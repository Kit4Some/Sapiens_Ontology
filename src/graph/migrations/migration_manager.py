"""
Migration Manager for Neo4j.

Handles schema migrations with:
- Version tracking in database
- Ordered execution
- Rollback support
- Dry-run mode
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.graph.migrations.base_migration import BaseMigration

logger = structlog.get_logger(__name__)


class MigrationStatus(str, Enum):
    """Status of a migration."""

    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Migration record."""

    version: str
    description: str
    status: MigrationStatus
    applied_at: datetime | None = None
    rolled_back_at: datetime | None = None
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "status": self.status.value,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class MigrationResult:
    """Result of migration operation."""

    success: bool
    operation: str  # "migrate", "rollback", "status"
    migrations_applied: list[str] = field(default_factory=list)
    migrations_rolled_back: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    current_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "migrations_applied": self.migrations_applied,
            "migrations_rolled_back": self.migrations_rolled_back,
            "errors": self.errors,
            "current_version": self.current_version,
        }


class MigrationManager:
    """
    Manages Neo4j schema migrations.

    Tracks applied migrations in a Migration node in the database.

    Usage:
        ```python
        manager = MigrationManager()

        # Register migrations
        manager.register(Migration001())
        manager.register(Migration002())

        # Apply all pending
        result = await manager.migrate()

        # Rollback last
        result = await manager.rollback()

        # Check status
        status = await manager.get_status()
        ```
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
    ) -> None:
        self._client = client or get_ontology_client()
        self._migrations: dict[str, BaseMigration] = {}

    def register(self, migration: BaseMigration) -> None:
        """
        Register a migration.

        Args:
            migration: Migration instance
        """
        self._migrations[migration.version] = migration
        logger.debug("Migration registered", version=migration.version)

    def register_many(self, migrations: list[BaseMigration]) -> None:
        """Register multiple migrations."""
        for migration in migrations:
            self.register(migration)

    async def setup(self) -> None:
        """
        Setup migration tracking infrastructure.

        Creates the Migration node and indexes.
        """
        await self._client.connect()

        # Create Migration node constraint
        try:
            await self._client.execute_cypher(
                "CREATE CONSTRAINT migration_version IF NOT EXISTS "
                "FOR (m:Migration) REQUIRE m.version IS UNIQUE"
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

        logger.info("Migration infrastructure setup complete")

    async def migrate(
        self,
        target_version: str | None = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Apply pending migrations.

        Args:
            target_version: Stop at this version (None = apply all)
            dry_run: Show what would be done without applying

        Returns:
            MigrationResult with details
        """
        result = MigrationResult(success=False, operation="migrate")

        await self.setup()

        # Get applied migrations
        applied = await self._get_applied_migrations()
        applied_versions = {m["version"] for m in applied}

        # Get pending migrations in order
        pending = []
        for version in sorted(self._migrations.keys()):
            if version not in applied_versions:
                pending.append(self._migrations[version])
                if target_version and version == target_version:
                    break

        if not pending:
            logger.info("No pending migrations")
            result.success = True
            result.current_version = max(applied_versions) if applied_versions else None
            return result

        logger.info(f"Found {len(pending)} pending migrations")

        if dry_run:
            logger.info("Dry run - no changes will be made")
            result.migrations_applied = [m.version for m in pending]
            result.success = True
            return result

        # Apply migrations
        for migration in pending:
            logger.info(
                "Applying migration",
                version=migration.version,
                description=migration.description,
            )

            start_time = datetime.utcnow()

            try:
                # Validate
                if not await migration.validate(self._client):
                    raise ValueError(f"Migration {migration.version} validation failed")

                # Apply
                await migration.up(self._client)

                # Record success
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                await self._record_migration(
                    migration,
                    MigrationStatus.APPLIED,
                    duration_ms=duration,
                )

                result.migrations_applied.append(migration.version)

                logger.info(
                    "Migration applied",
                    version=migration.version,
                    duration_ms=round(duration, 2),
                )

            except Exception as e:
                error_msg = f"Migration {migration.version} failed: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)

                # Record failure
                await self._record_migration(
                    migration,
                    MigrationStatus.FAILED,
                    error=str(e),
                )

                # Stop on failure
                break

        result.success = len(result.errors) == 0
        result.current_version = (
            result.migrations_applied[-1] if result.migrations_applied else None
        )

        return result

    async def rollback(
        self,
        steps: int = 1,
        target_version: str | None = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Rollback migrations.

        Args:
            steps: Number of migrations to rollback
            target_version: Rollback to this version
            dry_run: Show what would be done without applying

        Returns:
            MigrationResult with details
        """
        result = MigrationResult(success=False, operation="rollback")

        await self.setup()

        # Get applied migrations in reverse order
        applied = await self._get_applied_migrations()
        applied_versions = sorted(
            [m["version"] for m in applied if m["status"] == MigrationStatus.APPLIED.value],
            reverse=True,
        )

        if not applied_versions:
            logger.info("No migrations to rollback")
            result.success = True
            return result

        # Determine which migrations to rollback
        to_rollback = []
        if target_version:
            for version in applied_versions:
                if version <= target_version:
                    break
                to_rollback.append(version)
        else:
            to_rollback = applied_versions[:steps]

        if not to_rollback:
            logger.info("No migrations match rollback criteria")
            result.success = True
            return result

        logger.info(f"Rolling back {len(to_rollback)} migrations")

        if dry_run:
            result.migrations_rolled_back = to_rollback
            result.success = True
            return result

        # Rollback migrations
        for version in to_rollback:
            migration = self._migrations.get(version)

            if not migration:
                logger.warning(f"Migration {version} not found in registry")
                continue

            logger.info(
                "Rolling back migration",
                version=migration.version,
                description=migration.description,
            )

            try:
                await migration.down(self._client)

                # Update record
                await self._record_rollback(version)

                result.migrations_rolled_back.append(version)

                logger.info("Migration rolled back", version=version)

            except Exception as e:
                error_msg = f"Rollback of {version} failed: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                break

        result.success = len(result.errors) == 0

        # Get new current version
        remaining = await self._get_applied_migrations()
        remaining_versions = [
            m["version"] for m in remaining
            if m["status"] == MigrationStatus.APPLIED.value
        ]
        result.current_version = max(remaining_versions) if remaining_versions else None

        return result

    async def get_status(self) -> dict[str, Any]:
        """
        Get migration status.

        Returns:
            Status dictionary with applied and pending migrations
        """
        await self.setup()

        applied = await self._get_applied_migrations()
        applied_versions = {m["version"] for m in applied}

        pending = [
            {"version": v, "description": m.description}
            for v, m in sorted(self._migrations.items())
            if v not in applied_versions
        ]

        current = max(applied_versions) if applied_versions else None

        return {
            "current_version": current,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied": applied,
            "pending": pending,
        }

    async def _get_applied_migrations(self) -> list[dict[str, Any]]:
        """Get list of applied migrations from database."""
        query = """
        MATCH (m:Migration)
        RETURN m.version as version,
               m.description as description,
               m.status as status,
               m.applied_at as applied_at,
               m.error as error
        ORDER BY m.version
        """

        return await self._client.execute_cypher(query)

    async def _record_migration(
        self,
        migration: BaseMigration,
        status: MigrationStatus,
        duration_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Record migration in database."""
        query = """
        MERGE (m:Migration {version: $version})
        SET m.description = $description,
            m.status = $status,
            m.applied_at = datetime(),
            m.duration_ms = $duration_ms,
            m.error = $error
        """

        await self._client.execute_cypher(query, {
            "version": migration.version,
            "description": migration.description,
            "status": status.value,
            "duration_ms": duration_ms,
            "error": error,
        })

    async def _record_rollback(self, version: str) -> None:
        """Record rollback in database."""
        query = """
        MATCH (m:Migration {version: $version})
        SET m.status = $status,
            m.rolled_back_at = datetime()
        """

        await self._client.execute_cypher(query, {
            "version": version,
            "status": MigrationStatus.ROLLED_BACK.value,
        })


# Factory function
def create_migration_manager() -> MigrationManager:
    """Create a migration manager."""
    return MigrationManager()

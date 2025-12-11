"""
Neo4j Schema Migration System.

Provides database migration capabilities:
- Version-tracked schema changes
- Up/down migrations
- Migration history tracking
- Rollback support
- Dry-run mode

Usage:
    ```python
    from src.graph.migrations import MigrationManager

    manager = MigrationManager()

    # Apply all pending migrations
    await manager.migrate()

    # Rollback last migration
    await manager.rollback()

    # Check migration status
    status = await manager.get_status()
    ```
"""

from src.graph.migrations.migration_manager import (
    MigrationManager,
    Migration,
    MigrationResult,
    MigrationStatus,
)
from src.graph.migrations.base_migration import BaseMigration

__all__ = [
    "MigrationManager",
    "Migration",
    "MigrationResult",
    "MigrationStatus",
    "BaseMigration",
]

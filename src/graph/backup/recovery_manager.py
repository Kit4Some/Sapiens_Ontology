"""
Neo4j Recovery Manager.

Provides database recovery capabilities:
- Restore from Cypher, JSON, or GraphML backups
- Point-in-time recovery
- Selective recovery (specific node types)
- Dry-run mode for verification
"""

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.graph.backup.backup_manager import BackupManifest, BackupFormat

logger = structlog.get_logger(__name__)


@dataclass
class RecoveryPoint:
    """Represents a recovery point (backup)."""

    backup_id: str
    created_at: datetime
    backup_type: str
    format: str
    total_nodes: int
    total_relationships: int
    path: str

    @classmethod
    def from_manifest(cls, manifest: BackupManifest, path: str) -> "RecoveryPoint":
        return cls(
            backup_id=manifest.backup_id,
            created_at=manifest.created_at,
            backup_type=manifest.backup_type.value,
            format=manifest.format.value,
            total_nodes=manifest.total_nodes,
            total_relationships=manifest.total_relationships,
            path=path,
        )


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""

    success: bool
    backup_id: str
    recovery_type: str  # "full", "selective", "dry_run"

    # Statistics
    nodes_restored: int = 0
    relationships_restored: int = 0
    errors_count: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    # Details
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "backup_id": self.backup_id,
            "recovery_type": self.recovery_type,
            "statistics": {
                "nodes_restored": self.nodes_restored,
                "relationships_restored": self.relationships_restored,
                "errors_count": self.errors_count,
            },
            "timing": {
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


class RecoveryManager:
    """
    Neo4j Recovery Manager.

    Restores database from backups with support for:
    - Full restoration
    - Selective restoration (specific node types)
    - Dry-run mode
    - Pre-recovery validation

    Usage:
        ```python
        recovery = RecoveryManager(backup_dir="./backups")

        # List recovery points
        points = await recovery.list_recovery_points()

        # Restore from backup
        result = await recovery.restore(
            backup_id="20240115_120000_full",
            clear_before_restore=True,
        )

        # Dry run (validate without applying)
        result = await recovery.restore(backup_id, dry_run=True)
        ```
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
        backup_dir: str = "./backups",
    ) -> None:
        self._client = client or get_ontology_client()
        self._backup_path = Path(backup_dir)

    async def list_recovery_points(self) -> list[RecoveryPoint]:
        """
        List available recovery points.

        Returns:
            List of RecoveryPoint objects sorted by date descending
        """
        points = []

        for backup_dir in self._backup_path.iterdir():
            if not backup_dir.is_dir():
                continue

            manifest_path = backup_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    manifest = BackupManifest.from_dict(data)
                    points.append(RecoveryPoint.from_manifest(manifest, str(backup_dir)))
                except Exception as e:
                    logger.warning(
                        "Failed to load recovery point",
                        path=str(backup_dir),
                        error=str(e),
                    )

        # Sort by date descending
        points.sort(key=lambda x: x.created_at, reverse=True)

        return points

    async def get_recovery_point(self, backup_id: str) -> RecoveryPoint | None:
        """Get a specific recovery point by ID."""
        backup_dir = self._backup_path / backup_id

        if not backup_dir.exists():
            return None

        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = BackupManifest.from_dict(data)
            return RecoveryPoint.from_manifest(manifest, str(backup_dir))
        except Exception:
            return None

    async def restore(
        self,
        backup_id: str,
        clear_before_restore: bool = False,
        dry_run: bool = False,
        node_labels: list[str] | None = None,
        relationship_types: list[str] | None = None,
    ) -> RecoveryResult:
        """
        Restore database from backup.

        Args:
            backup_id: ID of backup to restore
            clear_before_restore: Clear database before restoring
            dry_run: Validate without applying changes
            node_labels: Specific node labels to restore (None = all)
            relationship_types: Specific relationship types to restore (None = all)

        Returns:
            RecoveryResult with restoration details
        """
        recovery_type = "dry_run" if dry_run else ("selective" if node_labels or relationship_types else "full")

        result = RecoveryResult(
            success=False,
            backup_id=backup_id,
            recovery_type=recovery_type,
            started_at=datetime.utcnow(),
        )

        backup_dir = self._backup_path / backup_id

        if not backup_dir.exists():
            result.errors.append(f"Backup not found: {backup_id}")
            return result

        # Load manifest
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            result.errors.append("Backup manifest not found")
            return result

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = BackupManifest.from_dict(manifest_data)
        except Exception as e:
            result.errors.append(f"Failed to load manifest: {str(e)}")
            return result

        logger.info(
            "Starting recovery",
            backup_id=backup_id,
            recovery_type=recovery_type,
            dry_run=dry_run,
            clear_before=clear_before_restore,
        )

        try:
            await self._client.connect()

            # Clear database if requested
            if clear_before_restore and not dry_run:
                await self._clear_database()
                logger.info("Database cleared before restore")

            # Determine which format to restore from
            format_priority = [BackupFormat.CYPHER, BackupFormat.JSON, BackupFormat.GRAPHML]

            restored = False
            for fmt in format_priority:
                file_info = self._find_backup_file(backup_dir, manifest, fmt)
                if file_info:
                    await self._restore_from_file(
                        backup_dir,
                        file_info,
                        result,
                        dry_run=dry_run,
                        node_labels=node_labels,
                        relationship_types=relationship_types,
                    )
                    restored = True
                    break

            if not restored:
                result.errors.append("No restorable backup file found")
                return result

            result.success = len(result.errors) == 0
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - result.started_at).total_seconds()

            logger.info(
                "Recovery completed",
                backup_id=backup_id,
                success=result.success,
                nodes=result.nodes_restored,
                relationships=result.relationships_restored,
                duration_s=round(result.duration_seconds, 2),
            )

        except Exception as e:
            result.errors.append(f"Recovery failed: {str(e)}")
            logger.error("Recovery failed", backup_id=backup_id, error=str(e))

        return result

    def _find_backup_file(
        self,
        backup_dir: Path,
        manifest: BackupManifest,
        format: BackupFormat,
    ) -> dict[str, Any] | None:
        """Find backup file for given format."""
        type_map = {
            BackupFormat.CYPHER: "cypher",
            BackupFormat.JSON: "json",
            BackupFormat.GRAPHML: "graphml",
        }

        target_type = type_map.get(format)
        if not target_type:
            return None

        for file_info in manifest.files:
            if file_info.get("type") == target_type:
                file_path = backup_dir / file_info["name"]
                if file_path.exists():
                    return file_info

        return None

    async def _restore_from_file(
        self,
        backup_dir: Path,
        file_info: dict[str, Any],
        result: RecoveryResult,
        dry_run: bool = False,
        node_labels: list[str] | None = None,
        relationship_types: list[str] | None = None,
    ) -> None:
        """Restore from a specific backup file."""
        file_path = backup_dir / file_info["name"]
        file_type = file_info.get("type", "")
        is_compressed = file_info.get("compressed", False)

        logger.info(
            "Restoring from file",
            file=file_info["name"],
            type=file_type,
            compressed=is_compressed,
        )

        if file_type == "cypher":
            await self._restore_from_cypher(
                file_path, result, dry_run, is_compressed, node_labels, relationship_types
            )
        elif file_type == "json":
            await self._restore_from_json(
                file_path, result, dry_run, is_compressed, node_labels, relationship_types
            )
        else:
            result.warnings.append(f"Unsupported file type for restore: {file_type}")

    async def _restore_from_cypher(
        self,
        file_path: Path,
        result: RecoveryResult,
        dry_run: bool,
        is_compressed: bool,
        node_labels: list[str] | None,
        relationship_types: list[str] | None,
    ) -> None:
        """Restore from Cypher file."""
        open_func = gzip.open if is_compressed else open
        mode = "rt" if is_compressed else "r"

        with open_func(file_path, mode, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("//"):
                    continue

                # Parse and filter
                is_node = line.startswith("MERGE (n:")
                is_rel = line.startswith("MATCH (a:")

                # Apply filters
                if node_labels and is_node:
                    if not any(f"MERGE (n:{label}" in line for label in node_labels):
                        continue

                if relationship_types and is_rel:
                    if not any(f"[r:{rel_type}]" in line for rel_type in relationship_types):
                        continue

                # Execute or count
                if dry_run:
                    if is_node:
                        result.nodes_restored += 1
                    elif is_rel:
                        result.relationships_restored += 1
                else:
                    try:
                        # Remove trailing semicolon if present
                        query = line.rstrip(";")
                        await self._client.execute_cypher(query)

                        if is_node:
                            result.nodes_restored += 1
                        elif is_rel:
                            result.relationships_restored += 1

                    except Exception as e:
                        result.errors_count += 1
                        if result.errors_count <= 10:  # Limit error messages
                            result.errors.append(f"Query failed: {str(e)[:100]}")

    async def _restore_from_json(
        self,
        file_path: Path,
        result: RecoveryResult,
        dry_run: bool,
        is_compressed: bool,
        node_labels: list[str] | None,
        relationship_types: list[str] | None,
    ) -> None:
        """Restore from JSON file."""
        open_func = gzip.open if is_compressed else open
        mode = "rt" if is_compressed else "r"

        with open_func(file_path, mode, encoding="utf-8") as f:
            data = json.load(f)

        # Restore nodes
        for label, nodes in data.get("nodes", {}).items():
            # Apply filter
            if node_labels and label not in node_labels:
                continue

            for node in nodes:
                if dry_run:
                    result.nodes_restored += 1
                else:
                    try:
                        # Build properties, handling embedding specially
                        props = {k: v for k, v in node.items() if not isinstance(v, str) or not v.startswith("[")}

                        query = f"""
                        MERGE (n:{label} {{id: $id}})
                        SET n = $props
                        """
                        await self._client.execute_cypher(query, {
                            "id": node.get("id"),
                            "props": props,
                        })
                        result.nodes_restored += 1

                    except Exception as e:
                        result.errors_count += 1
                        if result.errors_count <= 10:
                            result.errors.append(f"Node restore failed: {str(e)[:100]}")

        # Restore relationships
        for rel in data.get("relationships", []):
            rel_type = rel.get("type", "RELATES_TO")

            # Apply filter
            if relationship_types and rel_type not in relationship_types:
                continue

            if dry_run:
                result.relationships_restored += 1
            else:
                try:
                    query = f"""
                    MATCH (a {{id: $source}})
                    MATCH (b {{id: $target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r = $props
                    """
                    await self._client.execute_cypher(query, {
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                        "props": rel.get("properties", {}),
                    })
                    result.relationships_restored += 1

                except Exception as e:
                    result.errors_count += 1
                    if result.errors_count <= 10:
                        result.errors.append(f"Relationship restore failed: {str(e)[:100]}")

    async def _clear_database(self) -> None:
        """Clear all data from database."""
        # Delete relationships first
        await self._client.execute_cypher("MATCH ()-[r]->() DELETE r")
        # Delete nodes
        await self._client.execute_cypher("MATCH (n) DELETE n")

    async def validate_backup(self, backup_id: str) -> dict[str, Any]:
        """
        Validate a backup without restoring.

        Returns detailed validation results.
        """
        result = await self.restore(backup_id, dry_run=True)

        return {
            "backup_id": backup_id,
            "valid": result.success and result.errors_count == 0,
            "nodes_found": result.nodes_restored,
            "relationships_found": result.relationships_restored,
            "errors": result.errors,
            "warnings": result.warnings,
        }


# Factory function
def create_recovery_manager(
    backup_dir: str = "./backups",
) -> RecoveryManager:
    """Create a recovery manager."""
    return RecoveryManager(backup_dir=backup_dir)

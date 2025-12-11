"""
Neo4j Backup Manager.

Provides multiple backup strategies:
- Full export via Cypher statements
- JSON export for data portability
- GraphML export for graph analysis tools
- Incremental backups based on timestamps
- Compressed backups with checksums
"""

import asyncio
import gzip
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client

logger = structlog.get_logger(__name__)


class BackupFormat(str, Enum):
    """Supported backup formats."""

    CYPHER = "cypher"       # Cypher statements for replay
    JSON = "json"           # JSON for portability
    GRAPHML = "graphml"     # GraphML for graph tools
    FULL = "full"           # All formats


class BackupType(str, Enum):
    """Backup types."""

    FULL = "full"           # Complete graph backup
    INCREMENTAL = "incremental"  # Changes since last backup
    SCHEMA_ONLY = "schema_only"  # Only indexes and constraints


@dataclass
class BackupConfig:
    """Backup configuration."""

    # Paths
    backup_dir: str = "./backups"

    # Backup settings
    format: BackupFormat = BackupFormat.CYPHER
    backup_type: BackupType = BackupType.FULL
    compress: bool = True

    # Retention
    retention_days: int = 30
    max_backups: int = 100

    # Performance
    batch_size: int = 1000

    # Verification
    verify_after_backup: bool = True
    include_checksums: bool = True


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    backup_id: str
    backup_path: str
    backup_type: BackupType
    format: BackupFormat

    # Statistics
    node_count: int = 0
    relationship_count: int = 0
    file_size_bytes: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    # Verification
    checksum: str | None = None
    verified: bool = False

    # Errors
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "backup_id": self.backup_id,
            "backup_path": self.backup_path,
            "backup_type": self.backup_type.value,
            "format": self.format.value,
            "statistics": {
                "node_count": self.node_count,
                "relationship_count": self.relationship_count,
                "file_size_bytes": self.file_size_bytes,
                "file_size_mb": round(self.file_size_bytes / 1024 / 1024, 2),
            },
            "timing": {
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "verification": {
                "checksum": self.checksum,
                "verified": self.verified,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class BackupManifest:
    """Manifest file for backup metadata."""

    backup_id: str
    created_at: datetime
    backup_type: BackupType
    format: BackupFormat

    # Database info
    neo4j_version: str = ""
    database_name: str = "neo4j"

    # Contents
    files: list[dict[str, Any]] = field(default_factory=list)
    node_labels: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)

    # Statistics
    total_nodes: int = 0
    total_relationships: int = 0

    # Checksums
    checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at.isoformat(),
            "backup_type": self.backup_type.value,
            "format": self.format.value,
            "database": {
                "neo4j_version": self.neo4j_version,
                "database_name": self.database_name,
            },
            "contents": {
                "files": self.files,
                "node_labels": self.node_labels,
                "relationship_types": self.relationship_types,
            },
            "statistics": {
                "total_nodes": self.total_nodes,
                "total_relationships": self.total_relationships,
            },
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackupManifest":
        return cls(
            backup_id=data["backup_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            backup_type=BackupType(data["backup_type"]),
            format=BackupFormat(data["format"]),
            neo4j_version=data.get("database", {}).get("neo4j_version", ""),
            database_name=data.get("database", {}).get("database_name", "neo4j"),
            files=data.get("contents", {}).get("files", []),
            node_labels=data.get("contents", {}).get("node_labels", []),
            relationship_types=data.get("contents", {}).get("relationship_types", []),
            total_nodes=data.get("statistics", {}).get("total_nodes", 0),
            total_relationships=data.get("statistics", {}).get("total_relationships", 0),
            checksums=data.get("checksums", {}),
        )


class BackupManager:
    """
    Neo4j Backup Manager.

    Provides comprehensive backup capabilities with multiple formats,
    compression, verification, and retention management.

    Usage:
        ```python
        manager = BackupManager(config=BackupConfig(
            backup_dir="./backups",
            format=BackupFormat.CYPHER,
            compress=True,
            retention_days=30,
        ))

        # Create backup
        result = await manager.create_backup()

        # List backups
        backups = await manager.list_backups()

        # Verify backup
        is_valid = await manager.verify_backup(result.backup_id)

        # Cleanup old backups
        await manager.cleanup_old_backups()
        ```
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
        config: BackupConfig | None = None,
    ) -> None:
        self._client = client or get_ontology_client()
        self.config = config or BackupConfig()

        # Ensure backup directory exists
        self._backup_path = Path(self.config.backup_dir)
        self._backup_path.mkdir(parents=True, exist_ok=True)

    async def create_backup(
        self,
        backup_type: BackupType | None = None,
        format: BackupFormat | None = None,
        description: str = "",
    ) -> BackupResult:
        """
        Create a new backup.

        Args:
            backup_type: Override config backup type
            format: Override config format
            description: Optional backup description

        Returns:
            BackupResult with backup details
        """
        backup_type = backup_type or self.config.backup_type
        format = format or self.config.format

        # Generate backup ID
        timestamp = datetime.utcnow()
        backup_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{backup_type.value}"

        # Create backup directory
        backup_dir = self._backup_path / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        result = BackupResult(
            success=False,
            backup_id=backup_id,
            backup_path=str(backup_dir),
            backup_type=backup_type,
            format=format,
            started_at=timestamp,
        )

        logger.info(
            "Starting backup",
            backup_id=backup_id,
            backup_type=backup_type.value,
            format=format.value,
        )

        try:
            await self._client.connect()

            # Get database statistics
            stats = await self._get_database_stats()
            result.node_count = stats["nodes"]
            result.relationship_count = stats["relationships"]

            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                created_at=timestamp,
                backup_type=backup_type,
                format=format,
                total_nodes=stats["nodes"],
                total_relationships=stats["relationships"],
                node_labels=stats.get("labels", []),
                relationship_types=stats.get("rel_types", []),
            )

            # Export based on backup type
            if backup_type == BackupType.SCHEMA_ONLY:
                await self._export_schema(backup_dir, manifest)
            elif backup_type == BackupType.INCREMENTAL:
                await self._export_incremental(backup_dir, manifest)
            else:
                await self._export_full(backup_dir, format, manifest)

            # Calculate file sizes
            total_size = sum(
                f.stat().st_size for f in backup_dir.rglob("*") if f.is_file()
            )
            result.file_size_bytes = total_size

            # Write manifest
            manifest_path = backup_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

            # Verify if configured
            if self.config.verify_after_backup:
                result.verified = await self.verify_backup(backup_id)
                if not result.verified:
                    result.warnings.append("Backup verification failed")

            # Calculate checksum
            if self.config.include_checksums:
                result.checksum = await self._calculate_backup_checksum(backup_dir)

            result.success = True
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

            logger.info(
                "Backup completed",
                backup_id=backup_id,
                nodes=result.node_count,
                relationships=result.relationship_count,
                size_mb=round(result.file_size_bytes / 1024 / 1024, 2),
                duration_s=round(result.duration_seconds, 2),
            )

        except Exception as e:
            result.errors.append(f"Backup failed: {str(e)}")
            logger.error("Backup failed", backup_id=backup_id, error=str(e))

            # Cleanup failed backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

        return result

    async def _get_database_stats(self) -> dict[str, Any]:
        """Get database statistics for backup."""
        stats: dict[str, Any] = {"nodes": 0, "relationships": 0, "labels": [], "rel_types": []}

        # Count nodes
        node_query = "MATCH (n) RETURN count(n) as count"
        result = await self._client.execute_cypher(node_query)
        stats["nodes"] = result[0]["count"] if result else 0

        # Count relationships
        rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
        result = await self._client.execute_cypher(rel_query)
        stats["relationships"] = result[0]["count"] if result else 0

        # Get labels
        label_query = "CALL db.labels() YIELD label RETURN collect(label) as labels"
        result = await self._client.execute_cypher(label_query)
        stats["labels"] = result[0]["labels"] if result else []

        # Get relationship types
        rel_type_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        result = await self._client.execute_cypher(rel_type_query)
        stats["rel_types"] = result[0]["types"] if result else []

        return stats

    async def _export_schema(
        self,
        backup_dir: Path,
        manifest: BackupManifest,
    ) -> None:
        """Export schema (constraints and indexes) only."""
        schema_file = backup_dir / "schema.cypher"

        # Get constraints
        constraints_query = "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties"
        constraints = await self._client.execute_cypher(constraints_query)

        # Get indexes
        indexes_query = "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state"
        indexes = await self._client.execute_cypher(indexes_query)

        with open(schema_file, "w", encoding="utf-8") as f:
            f.write("// Neo4j Schema Export\n")
            f.write(f"// Generated: {datetime.utcnow().isoformat()}\n\n")

            f.write("// === CONSTRAINTS ===\n")
            for c in constraints:
                f.write(f"// {c.get('name')}: {c.get('type')} on {c.get('labelsOrTypes')}\n")

            f.write("\n// === INDEXES ===\n")
            for idx in indexes:
                f.write(f"// {idx.get('name')}: {idx.get('type')} on {idx.get('labelsOrTypes')} ({idx.get('state')})\n")

        manifest.files.append({
            "name": "schema.cypher",
            "type": "schema",
            "size": schema_file.stat().st_size,
        })

    async def _export_full(
        self,
        backup_dir: Path,
        format: BackupFormat,
        manifest: BackupManifest,
    ) -> None:
        """Export full database."""

        if format in (BackupFormat.CYPHER, BackupFormat.FULL):
            await self._export_cypher(backup_dir, manifest)

        if format in (BackupFormat.JSON, BackupFormat.FULL):
            await self._export_json(backup_dir, manifest)

        if format in (BackupFormat.GRAPHML, BackupFormat.FULL):
            await self._export_graphml(backup_dir, manifest)

    async def _export_cypher(
        self,
        backup_dir: Path,
        manifest: BackupManifest,
    ) -> None:
        """Export database as Cypher statements."""
        cypher_file = backup_dir / "data.cypher"

        with open(cypher_file, "w", encoding="utf-8") as f:
            f.write("// Neo4j Full Export (Cypher)\n")
            f.write(f"// Generated: {datetime.utcnow().isoformat()}\n")
            f.write(f"// Nodes: {manifest.total_nodes}, Relationships: {manifest.total_relationships}\n\n")

            # Export nodes by label
            f.write("// === NODES ===\n")
            for label in manifest.node_labels:
                f.write(f"\n// -- {label} nodes --\n")

                query = f"""
                MATCH (n:{label})
                RETURN n.id as id, properties(n) as props
                """

                results = await self._client.execute_cypher(query)

                for row in results:
                    props = row["props"]
                    # Remove embedding for readability (too large)
                    if "embedding" in props:
                        props["embedding"] = f"[{len(props['embedding'])} floats]"

                    props_str = json.dumps(props, ensure_ascii=False, default=str)
                    f.write(f"MERGE (n:{label} {{id: '{row['id']}'}}) SET n = {props_str};\n")

            # Export relationships
            f.write("\n// === RELATIONSHIPS ===\n")
            for rel_type in manifest.relationship_types:
                f.write(f"\n// -- {rel_type} relationships --\n")

                query = f"""
                MATCH (a)-[r:{rel_type}]->(b)
                RETURN a.id as source, b.id as target,
                       labels(a)[0] as source_label, labels(b)[0] as target_label,
                       properties(r) as props
                """

                results = await self._client.execute_cypher(query)

                for row in results:
                    props_str = json.dumps(row["props"], ensure_ascii=False, default=str) if row["props"] else "{}"
                    f.write(
                        f"MATCH (a:{row['source_label']} {{id: '{row['source']}'}}), "
                        f"(b:{row['target_label']} {{id: '{row['target']}'}}) "
                        f"MERGE (a)-[r:{rel_type}]->(b) SET r = {props_str};\n"
                    )

        # Compress if configured
        if self.config.compress:
            compressed_file = cypher_file.with_suffix(".cypher.gz")
            with open(cypher_file, "rb") as f_in:
                with gzip.open(compressed_file, "wb") as f_out:
                    f_out.writelines(f_in)
            cypher_file.unlink()
            manifest.files.append({
                "name": "data.cypher.gz",
                "type": "cypher",
                "compressed": True,
                "size": compressed_file.stat().st_size,
            })
        else:
            manifest.files.append({
                "name": "data.cypher",
                "type": "cypher",
                "compressed": False,
                "size": cypher_file.stat().st_size,
            })

    async def _export_json(
        self,
        backup_dir: Path,
        manifest: BackupManifest,
    ) -> None:
        """Export database as JSON."""
        data = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "total_nodes": manifest.total_nodes,
                "total_relationships": manifest.total_relationships,
            },
            "nodes": {},
            "relationships": [],
        }

        # Export nodes by label
        for label in manifest.node_labels:
            query = f"MATCH (n:{label}) RETURN properties(n) as props"
            results = await self._client.execute_cypher(query)
            data["nodes"][label] = [row["props"] for row in results]

        # Export relationships
        for rel_type in manifest.relationship_types:
            query = f"""
            MATCH (a)-[r:{rel_type}]->(b)
            RETURN a.id as source, b.id as target, type(r) as type, properties(r) as props
            """
            results = await self._client.execute_cypher(query)
            data["relationships"].extend([
                {
                    "source": row["source"],
                    "target": row["target"],
                    "type": row["type"],
                    "properties": row["props"],
                }
                for row in results
            ])

        # Write JSON
        json_file = backup_dir / "data.json"

        if self.config.compress:
            json_file = json_file.with_suffix(".json.gz")
            with gzip.open(json_file, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)
        else:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)

        manifest.files.append({
            "name": json_file.name,
            "type": "json",
            "compressed": self.config.compress,
            "size": json_file.stat().st_size,
        })

    async def _export_graphml(
        self,
        backup_dir: Path,
        manifest: BackupManifest,
    ) -> None:
        """Export database as GraphML."""
        graphml_file = backup_dir / "data.graphml"

        with open(graphml_file, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
            f.write('  <graph id="G" edgedefault="directed">\n')

            # Export nodes
            node_query = """
            MATCH (n)
            RETURN id(n) as neo_id, labels(n) as labels, n.id as id, n.name as name
            """
            nodes = await self._client.execute_cypher(node_query)

            for node in nodes:
                label = node["labels"][0] if node["labels"] else "Node"
                f.write(f'    <node id="{node["id"]}">\n')
                f.write(f'      <data key="label">{label}</data>\n')
                if node["name"]:
                    f.write(f'      <data key="name">{node["name"]}</data>\n')
                f.write('    </node>\n')

            # Export edges
            edge_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id as source, b.id as target, type(r) as type
            """
            edges = await self._client.execute_cypher(edge_query)

            for i, edge in enumerate(edges):
                f.write(f'    <edge id="e{i}" source="{edge["source"]}" target="{edge["target"]}">\n')
                f.write(f'      <data key="type">{edge["type"]}</data>\n')
                f.write('    </edge>\n')

            f.write('  </graph>\n')
            f.write('</graphml>\n')

        # Compress if configured
        if self.config.compress:
            compressed_file = graphml_file.with_suffix(".graphml.gz")
            with open(graphml_file, "rb") as f_in:
                with gzip.open(compressed_file, "wb") as f_out:
                    f_out.writelines(f_in)
            graphml_file.unlink()
            manifest.files.append({
                "name": "data.graphml.gz",
                "type": "graphml",
                "compressed": True,
                "size": compressed_file.stat().st_size,
            })
        else:
            manifest.files.append({
                "name": "data.graphml",
                "type": "graphml",
                "compressed": False,
                "size": graphml_file.stat().st_size,
            })

    async def _export_incremental(
        self,
        backup_dir: Path,
        manifest: BackupManifest,
    ) -> None:
        """Export incremental changes since last backup."""
        # Find last backup timestamp
        last_backup = await self._get_last_backup_time()

        if not last_backup:
            logger.warning("No previous backup found, performing full backup")
            await self._export_full(backup_dir, BackupFormat.CYPHER, manifest)
            return

        # Export nodes modified since last backup
        cypher_file = backup_dir / "incremental.cypher"

        with open(cypher_file, "w", encoding="utf-8") as f:
            f.write(f"// Incremental backup since {last_backup.isoformat()}\n\n")

            # Nodes with updated_at > last_backup
            query = """
            MATCH (n)
            WHERE n.updated_at > $since OR n.created_at > $since
            RETURN labels(n) as labels, properties(n) as props
            """

            results = await self._client.execute_cypher(query, {"since": last_backup.isoformat()})

            f.write(f"// {len(results)} modified nodes\n")
            for row in results:
                label = row["labels"][0] if row["labels"] else "Node"
                props = row["props"]
                if "embedding" in props:
                    props["embedding"] = f"[{len(props['embedding'])} floats]"
                props_str = json.dumps(props, ensure_ascii=False, default=str)
                f.write(f"MERGE (n:{label} {{id: '{props.get('id')}'}}) SET n = {props_str};\n")

        manifest.files.append({
            "name": "incremental.cypher",
            "type": "incremental",
            "since": last_backup.isoformat(),
            "size": cypher_file.stat().st_size,
        })

    async def _get_last_backup_time(self) -> datetime | None:
        """Get timestamp of last successful backup."""
        backups = await self.list_backups()
        if not backups:
            return None

        # Sort by creation time descending
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        for backup in backups:
            if backup.get("success", False):
                return datetime.fromisoformat(backup["created_at"])

        return None

    async def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate SHA256 checksum of backup files."""
        sha256 = hashlib.sha256()

        for file_path in sorted(backup_dir.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)

        return sha256.hexdigest()

    async def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity.

        Args:
            backup_id: ID of backup to verify

        Returns:
            True if backup is valid
        """
        backup_dir = self._backup_path / backup_id

        if not backup_dir.exists():
            logger.error("Backup not found", backup_id=backup_id)
            return False

        # Check manifest exists
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("Manifest not found", backup_id=backup_id)
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = BackupManifest.from_dict(manifest_data)

            # Verify all files exist
            for file_info in manifest.files:
                file_path = backup_dir / file_info["name"]
                if not file_path.exists():
                    logger.error("Missing file", backup_id=backup_id, file=file_info["name"])
                    return False

                # Verify file size
                if file_path.stat().st_size != file_info.get("size", 0):
                    logger.warning(
                        "File size mismatch",
                        backup_id=backup_id,
                        file=file_info["name"],
                    )

            logger.info("Backup verified", backup_id=backup_id)
            return True

        except Exception as e:
            logger.error("Backup verification failed", backup_id=backup_id, error=str(e))
            return False

    async def list_backups(self) -> list[dict[str, Any]]:
        """
        List all available backups.

        Returns:
            List of backup metadata dictionaries
        """
        backups = []

        for backup_dir in self._backup_path.iterdir():
            if not backup_dir.is_dir():
                continue

            manifest_path = backup_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    # Calculate total size
                    total_size = sum(
                        f.stat().st_size for f in backup_dir.rglob("*") if f.is_file()
                    )

                    backups.append({
                        "backup_id": manifest["backup_id"],
                        "created_at": manifest["created_at"],
                        "backup_type": manifest["backup_type"],
                        "format": manifest["format"],
                        "total_nodes": manifest.get("statistics", {}).get("total_nodes", 0),
                        "total_relationships": manifest.get("statistics", {}).get("total_relationships", 0),
                        "size_bytes": total_size,
                        "size_mb": round(total_size / 1024 / 1024, 2),
                        "path": str(backup_dir),
                        "success": True,
                    })
                except Exception as e:
                    backups.append({
                        "backup_id": backup_dir.name,
                        "path": str(backup_dir),
                        "success": False,
                        "error": str(e),
                    })

        # Sort by creation time descending
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return backups

    async def delete_backup(self, backup_id: str, force: bool = False) -> bool:
        """
        Delete a backup.

        Args:
            backup_id: ID of backup to delete
            force: Force deletion even if verification fails

        Returns:
            True if successfully deleted
        """
        backup_dir = self._backup_path / backup_id

        if not backup_dir.exists():
            logger.warning("Backup not found for deletion", backup_id=backup_id)
            return False

        try:
            shutil.rmtree(backup_dir)
            logger.info("Backup deleted", backup_id=backup_id)
            return True
        except Exception as e:
            logger.error("Failed to delete backup", backup_id=backup_id, error=str(e))
            return False

    async def cleanup_old_backups(self) -> dict[str, Any]:
        """
        Remove backups older than retention period.

        Returns:
            Cleanup statistics
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)

        backups = await self.list_backups()
        deleted = []
        kept = []
        errors = []

        for backup in backups:
            try:
                created_at = datetime.fromisoformat(backup["created_at"])

                if created_at < cutoff_date:
                    if await self.delete_backup(backup["backup_id"]):
                        deleted.append(backup["backup_id"])
                    else:
                        errors.append(backup["backup_id"])
                else:
                    kept.append(backup["backup_id"])

            except Exception as e:
                errors.append(f"{backup.get('backup_id', 'unknown')}: {str(e)}")

        # Also enforce max_backups limit
        remaining = await self.list_backups()
        if len(remaining) > self.config.max_backups:
            # Delete oldest backups
            to_delete = remaining[self.config.max_backups:]
            for backup in to_delete:
                if await self.delete_backup(backup["backup_id"]):
                    deleted.append(backup["backup_id"])

        result = {
            "deleted_count": len(deleted),
            "kept_count": len(kept),
            "deleted": deleted,
            "errors": errors,
            "cutoff_date": cutoff_date.isoformat(),
        }

        logger.info("Backup cleanup completed", **result)
        return result


# Factory function
def create_backup_manager(
    backup_dir: str = "./backups",
    **kwargs: Any,
) -> BackupManager:
    """Create a backup manager with custom configuration."""
    config = BackupConfig(backup_dir=backup_dir, **kwargs)
    return BackupManager(config=config)

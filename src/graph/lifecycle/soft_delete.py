"""
Soft Delete Manager for Neo4j.

Provides soft-delete functionality:
- Mark nodes as deleted without removing them
- Automatic filtering of deleted nodes in queries
- Restore deleted nodes
- Permanent purge after retention period
- Cascade soft-delete for related nodes
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client

logger = structlog.get_logger(__name__)


class DeletionPolicy(str, Enum):
    """Policies for handling related nodes during deletion."""

    CASCADE = "cascade"      # Soft-delete related nodes
    ORPHAN = "orphan"        # Leave related nodes (may become orphans)
    RESTRICT = "restrict"    # Prevent deletion if has relations
    SET_NULL = "set_null"    # Remove references to deleted node


@dataclass
class SoftDeleteConfig:
    """Configuration for soft-delete behavior."""

    # Retention
    retention_days: int = 30  # Days before permanent deletion eligible
    auto_purge: bool = False  # Automatically purge after retention

    # Cascade behavior
    default_policy: DeletionPolicy = DeletionPolicy.ORPHAN

    # Filtering
    auto_filter_deleted: bool = True  # Automatically exclude deleted nodes

    # Audit
    track_deleted_by: bool = True  # Track who deleted the node
    track_deletion_reason: bool = True  # Track why it was deleted


@dataclass
class SoftDeleteResult:
    """Result of a soft-delete operation."""

    success: bool
    operation: str  # "delete", "restore", "purge"

    # Counts
    nodes_affected: int = 0
    relationships_affected: int = 0

    # Details
    node_ids: list[str] = field(default_factory=list)
    cascade_deleted: list[str] = field(default_factory=list)

    # Errors
    errors: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)  # For RESTRICT policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "nodes_affected": self.nodes_affected,
            "relationships_affected": self.relationships_affected,
            "node_ids": self.node_ids,
            "cascade_deleted": self.cascade_deleted,
            "errors": self.errors,
            "blocked_by": self.blocked_by,
        }


class SoftDeleteManager:
    """
    Manages soft-delete operations for Neo4j nodes.

    Soft-delete marks nodes as deleted without removing them from the database.
    This allows for:
    - Recovery of accidentally deleted data
    - Audit trails
    - Gradual data cleanup

    Usage:
        ```python
        manager = SoftDeleteManager(config=SoftDeleteConfig(
            retention_days=30,
            default_policy=DeletionPolicy.CASCADE,
        ))

        # Soft delete a node
        result = await manager.soft_delete(
            node_id="entity-123",
            label="Entity",
            deleted_by="user@example.com",
            reason="Duplicate entry",
        )

        # Restore a deleted node
        result = await manager.restore(node_id="entity-123", label="Entity")

        # Purge old deleted nodes
        result = await manager.purge_expired()

        # Get all deleted nodes
        deleted = await manager.list_deleted(label="Entity")
        ```
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
        config: SoftDeleteConfig | None = None,
    ) -> None:
        self._client = client or get_ontology_client()
        self.config = config or SoftDeleteConfig()

    async def setup_schema(self) -> dict[str, Any]:
        """
        Setup schema for soft-delete support.

        Creates indexes on soft-delete fields for performance.
        """
        await self._client.connect()

        queries = [
            # Index on is_deleted for filtering
            "CREATE INDEX entity_is_deleted IF NOT EXISTS FOR (n:Entity) ON (n.is_deleted)",
            "CREATE INDEX chunk_is_deleted IF NOT EXISTS FOR (n:Chunk) ON (n.is_deleted)",
            "CREATE INDEX community_is_deleted IF NOT EXISTS FOR (n:Community) ON (n.is_deleted)",

            # Index on deleted_at for purge queries
            "CREATE INDEX entity_deleted_at IF NOT EXISTS FOR (n:Entity) ON (n.deleted_at)",
            "CREATE INDEX chunk_deleted_at IF NOT EXISTS FOR (n:Chunk) ON (n.deleted_at)",
        ]

        results = []
        for query in queries:
            try:
                await self._client.execute_cypher(query)
                results.append({"query": query[:50], "status": "created"})
            except Exception as e:
                if "already exists" in str(e).lower():
                    results.append({"query": query[:50], "status": "exists"})
                else:
                    results.append({"query": query[:50], "status": "error", "error": str(e)})

        logger.info("Soft-delete schema setup completed", results=results)
        return {"indexes_created": results}

    async def soft_delete(
        self,
        node_id: str,
        label: str = "Entity",
        policy: DeletionPolicy | None = None,
        deleted_by: str | None = None,
        reason: str | None = None,
    ) -> SoftDeleteResult:
        """
        Soft delete a node.

        Args:
            node_id: ID of the node to delete
            label: Node label
            policy: Deletion policy for related nodes
            deleted_by: User/system that deleted the node
            reason: Reason for deletion

        Returns:
            SoftDeleteResult with deletion details
        """
        policy = policy or self.config.default_policy
        result = SoftDeleteResult(success=False, operation="delete")

        await self._client.connect()

        logger.info(
            "Soft deleting node",
            node_id=node_id,
            label=label,
            policy=policy.value,
        )

        try:
            # Check if node exists and is not already deleted
            check_query = f"""
            MATCH (n:{label} {{id: $node_id}})
            RETURN n.id as id, coalesce(n.is_deleted, false) as is_deleted
            """
            check_result = await self._client.execute_cypher(check_query, {"node_id": node_id})

            if not check_result:
                result.errors.append(f"Node not found: {node_id}")
                return result

            if check_result[0].get("is_deleted", False):
                result.errors.append(f"Node already deleted: {node_id}")
                return result

            # Handle policy
            if policy == DeletionPolicy.RESTRICT:
                # Check for related nodes
                rel_query = f"""
                MATCH (n:{label} {{id: $node_id}})-[r]-()
                RETURN count(r) as rel_count
                """
                rel_result = await self._client.execute_cypher(rel_query, {"node_id": node_id})
                rel_count = rel_result[0].get("rel_count", 0) if rel_result else 0

                if rel_count > 0:
                    result.errors.append(f"Cannot delete: node has {rel_count} relationships")
                    result.blocked_by = [f"{rel_count} relationships"]
                    return result

            elif policy == DeletionPolicy.CASCADE:
                # Get related entities to cascade delete
                cascade_query = f"""
                MATCH (n:{label} {{id: $node_id}})-[:RELATES_TO]-(related:Entity)
                WHERE coalesce(related.is_deleted, false) = false
                RETURN related.id as id
                """
                cascade_result = await self._client.execute_cypher(cascade_query, {"node_id": node_id})
                result.cascade_deleted = [r["id"] for r in cascade_result]

            # Perform soft delete
            delete_query = f"""
            MATCH (n:{label} {{id: $node_id}})
            SET n.is_deleted = true,
                n.deleted_at = datetime(),
                n.deleted_by = $deleted_by,
                n.deletion_reason = $reason,
                n.pre_delete_state = {{
                    had_relationships: size((n)-[]-())
                }}
            RETURN n.id as id
            """

            params = {
                "node_id": node_id,
                "deleted_by": deleted_by if self.config.track_deleted_by else None,
                "reason": reason if self.config.track_deletion_reason else None,
            }

            delete_result = await self._client.execute_cypher(delete_query, params)

            if delete_result:
                result.nodes_affected = 1
                result.node_ids.append(node_id)

                # Cascade delete related nodes
                if policy == DeletionPolicy.CASCADE and result.cascade_deleted:
                    for related_id in result.cascade_deleted:
                        cascade_result = await self.soft_delete(
                            node_id=related_id,
                            label="Entity",
                            policy=DeletionPolicy.ORPHAN,  # Don't cascade further
                            deleted_by=deleted_by,
                            reason=f"Cascade from {node_id}",
                        )
                        if cascade_result.success:
                            result.nodes_affected += cascade_result.nodes_affected

                # Soft-delete relationships (mark them, don't remove)
                if policy != DeletionPolicy.RESTRICT:
                    rel_delete_query = f"""
                    MATCH (n:{label} {{id: $node_id}})-[r]-()
                    SET r.is_deleted = true,
                        r.deleted_at = datetime()
                    RETURN count(r) as affected
                    """
                    rel_result = await self._client.execute_cypher(rel_delete_query, {"node_id": node_id})
                    result.relationships_affected = rel_result[0].get("affected", 0) if rel_result else 0

                result.success = True

            logger.info(
                "Soft delete completed",
                node_id=node_id,
                nodes_affected=result.nodes_affected,
                relationships_affected=result.relationships_affected,
            )

        except Exception as e:
            result.errors.append(f"Soft delete failed: {str(e)}")
            logger.error("Soft delete failed", node_id=node_id, error=str(e))

        return result

    async def soft_delete_batch(
        self,
        node_ids: list[str],
        label: str = "Entity",
        policy: DeletionPolicy | None = None,
        deleted_by: str | None = None,
        reason: str | None = None,
    ) -> SoftDeleteResult:
        """
        Soft delete multiple nodes in batch.

        Args:
            node_ids: List of node IDs to delete
            label: Node label
            policy: Deletion policy
            deleted_by: User/system that deleted
            reason: Reason for deletion

        Returns:
            Combined SoftDeleteResult
        """
        policy = policy or self.config.default_policy
        result = SoftDeleteResult(success=False, operation="delete_batch")

        await self._client.connect()

        # Batch update query
        batch_query = f"""
        UNWIND $node_ids AS node_id
        MATCH (n:{label} {{id: node_id}})
        WHERE coalesce(n.is_deleted, false) = false
        SET n.is_deleted = true,
            n.deleted_at = datetime(),
            n.deleted_by = $deleted_by,
            n.deletion_reason = $reason
        RETURN n.id as id
        """

        params = {
            "node_ids": node_ids,
            "deleted_by": deleted_by if self.config.track_deleted_by else None,
            "reason": reason if self.config.track_deletion_reason else None,
        }

        try:
            batch_result = await self._client.execute_cypher(batch_query, params)
            result.node_ids = [r["id"] for r in batch_result]
            result.nodes_affected = len(result.node_ids)
            result.success = True

            logger.info(
                "Batch soft delete completed",
                requested=len(node_ids),
                affected=result.nodes_affected,
            )

        except Exception as e:
            result.errors.append(f"Batch soft delete failed: {str(e)}")
            logger.error("Batch soft delete failed", error=str(e))

        return result

    async def restore(
        self,
        node_id: str,
        label: str = "Entity",
        restore_relationships: bool = True,
    ) -> SoftDeleteResult:
        """
        Restore a soft-deleted node.

        Args:
            node_id: ID of the node to restore
            label: Node label
            restore_relationships: Also restore soft-deleted relationships

        Returns:
            SoftDeleteResult with restoration details
        """
        result = SoftDeleteResult(success=False, operation="restore")

        await self._client.connect()

        logger.info("Restoring node", node_id=node_id, label=label)

        try:
            # Check if node exists and is deleted
            check_query = f"""
            MATCH (n:{label} {{id: $node_id}})
            RETURN n.id as id, coalesce(n.is_deleted, false) as is_deleted
            """
            check_result = await self._client.execute_cypher(check_query, {"node_id": node_id})

            if not check_result:
                result.errors.append(f"Node not found: {node_id}")
                return result

            if not check_result[0].get("is_deleted", False):
                result.errors.append(f"Node is not deleted: {node_id}")
                return result

            # Restore node
            restore_query = f"""
            MATCH (n:{label} {{id: $node_id}})
            SET n.is_deleted = false,
                n.restored_at = datetime(),
                n.deleted_at = null,
                n.deleted_by = null,
                n.deletion_reason = null
            RETURN n.id as id
            """

            restore_result = await self._client.execute_cypher(restore_query, {"node_id": node_id})

            if restore_result:
                result.nodes_affected = 1
                result.node_ids.append(node_id)

                # Restore relationships
                if restore_relationships:
                    rel_restore_query = f"""
                    MATCH (n:{label} {{id: $node_id}})-[r]-()
                    WHERE coalesce(r.is_deleted, false) = true
                    SET r.is_deleted = false,
                        r.restored_at = datetime(),
                        r.deleted_at = null
                    RETURN count(r) as affected
                    """
                    rel_result = await self._client.execute_cypher(rel_restore_query, {"node_id": node_id})
                    result.relationships_affected = rel_result[0].get("affected", 0) if rel_result else 0

                result.success = True

            logger.info(
                "Restore completed",
                node_id=node_id,
                relationships_restored=result.relationships_affected,
            )

        except Exception as e:
            result.errors.append(f"Restore failed: {str(e)}")
            logger.error("Restore failed", node_id=node_id, error=str(e))

        return result

    async def purge(
        self,
        node_id: str,
        label: str = "Entity",
    ) -> SoftDeleteResult:
        """
        Permanently delete a soft-deleted node.

        Args:
            node_id: ID of the node to purge
            label: Node label

        Returns:
            SoftDeleteResult with purge details
        """
        result = SoftDeleteResult(success=False, operation="purge")

        await self._client.connect()

        logger.info("Purging node", node_id=node_id, label=label)

        try:
            # Only purge if already soft-deleted
            purge_query = f"""
            MATCH (n:{label} {{id: $node_id}})
            WHERE coalesce(n.is_deleted, false) = true
            DETACH DELETE n
            RETURN count(*) as deleted
            """

            purge_result = await self._client.execute_cypher(purge_query, {"node_id": node_id})

            deleted_count = purge_result[0].get("deleted", 0) if purge_result else 0

            if deleted_count > 0:
                result.nodes_affected = deleted_count
                result.node_ids.append(node_id)
                result.success = True
            else:
                result.errors.append(f"Node not found or not soft-deleted: {node_id}")

            logger.info("Purge completed", node_id=node_id, deleted=deleted_count)

        except Exception as e:
            result.errors.append(f"Purge failed: {str(e)}")
            logger.error("Purge failed", node_id=node_id, error=str(e))

        return result

    async def purge_expired(
        self,
        label: str | None = None,
        retention_days: int | None = None,
    ) -> SoftDeleteResult:
        """
        Permanently delete all nodes past retention period.

        Args:
            label: Specific label to purge (None = all labels)
            retention_days: Override config retention period

        Returns:
            SoftDeleteResult with purge details
        """
        retention_days = retention_days or self.config.retention_days
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        result = SoftDeleteResult(success=False, operation="purge_expired")

        await self._client.connect()

        logger.info(
            "Purging expired nodes",
            label=label or "all",
            retention_days=retention_days,
            cutoff=cutoff_date.isoformat(),
        )

        try:
            labels = [label] if label else ["Entity", "Chunk", "Community"]

            for lbl in labels:
                purge_query = f"""
                MATCH (n:{lbl})
                WHERE coalesce(n.is_deleted, false) = true
                  AND n.deleted_at < datetime($cutoff)
                WITH n LIMIT 1000
                DETACH DELETE n
                RETURN count(*) as deleted
                """

                # Repeat until no more to delete (batch deletion)
                total_deleted = 0
                while True:
                    purge_result = await self._client.execute_cypher(
                        purge_query,
                        {"cutoff": cutoff_date.isoformat()},
                    )
                    deleted = purge_result[0].get("deleted", 0) if purge_result else 0

                    if deleted == 0:
                        break

                    total_deleted += deleted

                result.nodes_affected += total_deleted
                if total_deleted > 0:
                    result.node_ids.append(f"{lbl}:{total_deleted}")

            result.success = True

            logger.info(
                "Purge expired completed",
                total_purged=result.nodes_affected,
            )

        except Exception as e:
            result.errors.append(f"Purge expired failed: {str(e)}")
            logger.error("Purge expired failed", error=str(e))

        return result

    async def list_deleted(
        self,
        label: str = "Entity",
        limit: int = 100,
        include_details: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List all soft-deleted nodes.

        Args:
            label: Node label
            limit: Maximum results
            include_details: Include deletion details

        Returns:
            List of deleted node information
        """
        await self._client.connect()

        if include_details:
            query = f"""
            MATCH (n:{label})
            WHERE coalesce(n.is_deleted, false) = true
            RETURN n.id as id,
                   n.name as name,
                   n.deleted_at as deleted_at,
                   n.deleted_by as deleted_by,
                   n.deletion_reason as reason,
                   size((n)-[]-()) as relationships
            ORDER BY n.deleted_at DESC
            LIMIT $limit
            """
        else:
            query = f"""
            MATCH (n:{label})
            WHERE coalesce(n.is_deleted, false) = true
            RETURN n.id as id, n.name as name
            ORDER BY n.deleted_at DESC
            LIMIT $limit
            """

        results = await self._client.execute_cypher(query, {"limit": limit})
        return results

    async def get_deletion_stats(self) -> dict[str, Any]:
        """
        Get statistics about soft-deleted nodes.

        Returns:
            Statistics dictionary
        """
        await self._client.connect()

        stats_query = """
        CALL {
            MATCH (n:Entity) WHERE coalesce(n.is_deleted, true) = true
            RETURN 'deleted_entities' as label, count(n) as count
            UNION ALL
            MATCH (n:Entity) WHERE coalesce(n.is_deleted, false) = false
            RETURN 'active_entities' as label, count(n) as count
            UNION ALL
            MATCH (n:Chunk) WHERE coalesce(n.is_deleted, true) = true
            RETURN 'deleted_chunks' as label, count(n) as count
            UNION ALL
            MATCH (n:Chunk) WHERE coalesce(n.is_deleted, false) = false
            RETURN 'active_chunks' as label, count(n) as count
        }
        RETURN label, count
        """

        results = await self._client.execute_cypher(stats_query)
        stats = {r["label"]: r["count"] for r in results}

        # Calculate retention info
        cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)

        expire_query = """
        MATCH (n)
        WHERE coalesce(n.is_deleted, false) = true
          AND n.deleted_at < datetime($cutoff)
        RETURN count(n) as eligible_for_purge
        """

        expire_result = await self._client.execute_cypher(expire_query, {"cutoff": cutoff.isoformat()})
        eligible = expire_result[0].get("eligible_for_purge", 0) if expire_result else 0

        return {
            "counts": stats,
            "retention": {
                "retention_days": self.config.retention_days,
                "cutoff_date": cutoff.isoformat(),
                "eligible_for_purge": eligible,
            },
        }


# Factory function
def create_soft_delete_manager(
    retention_days: int = 30,
    **kwargs: Any,
) -> SoftDeleteManager:
    """Create a soft delete manager with custom configuration."""
    config = SoftDeleteConfig(retention_days=retention_days, **kwargs)
    return SoftDeleteManager(config=config)

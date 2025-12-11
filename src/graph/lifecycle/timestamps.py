"""
Timestamp Management for Neo4j Nodes.

Provides automatic timestamp tracking:
- created_at: When the node was first created
- updated_at: When the node was last modified
- version: Incremental version number

Works with existing nodes to add timestamps retroactively.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client

logger = structlog.get_logger(__name__)


@dataclass
class TimestampConfig:
    """Configuration for timestamp management."""

    # Fields to track
    track_created_at: bool = True
    track_updated_at: bool = True
    track_version: bool = True

    # Auto-update behavior
    auto_update_on_change: bool = True

    # Timezone
    use_utc: bool = True


class TimestampManager:
    """
    Manages timestamps for Neo4j nodes.

    Provides utilities for:
    - Adding timestamps to new nodes
    - Updating timestamps on modification
    - Backfilling timestamps for existing nodes
    - Querying by timestamp ranges

    Usage:
        ```python
        manager = TimestampManager()

        # Setup schema (indexes on timestamp fields)
        await manager.setup_schema()

        # Add timestamps to existing nodes without them
        result = await manager.backfill_timestamps(label="Entity")

        # Query nodes by creation date
        recent = await manager.get_nodes_created_after(
            label="Entity",
            after=datetime(2024, 1, 1),
        )

        # Get modification history
        history = await manager.get_modification_history(
            node_id="entity-123",
            label="Entity",
        )
        ```
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
        config: TimestampConfig | None = None,
    ) -> None:
        self._client = client or get_ontology_client()
        self.config = config or TimestampConfig()

    async def setup_schema(self) -> dict[str, Any]:
        """
        Setup indexes for timestamp fields.

        Returns:
            Schema setup results
        """
        await self._client.connect()

        queries = [
            # created_at indexes
            "CREATE INDEX entity_created_at IF NOT EXISTS FOR (n:Entity) ON (n.created_at)",
            "CREATE INDEX chunk_created_at IF NOT EXISTS FOR (n:Chunk) ON (n.created_at)",
            "CREATE INDEX community_created_at IF NOT EXISTS FOR (n:Community) ON (n.created_at)",

            # updated_at indexes
            "CREATE INDEX entity_updated_at IF NOT EXISTS FOR (n:Entity) ON (n.updated_at)",
            "CREATE INDEX chunk_updated_at IF NOT EXISTS FOR (n:Chunk) ON (n.updated_at)",

            # version index (for optimistic locking queries)
            "CREATE INDEX entity_version IF NOT EXISTS FOR (n:Entity) ON (n.version)",
        ]

        results = []
        for query in queries:
            try:
                await self._client.execute_cypher(query)
                results.append({"query": query[:60], "status": "created"})
            except Exception as e:
                if "already exists" in str(e).lower():
                    results.append({"query": query[:60], "status": "exists"})
                else:
                    results.append({"query": query[:60], "status": "error", "error": str(e)})

        logger.info("Timestamp schema setup completed", results=results)
        return {"indexes_created": results}

    async def backfill_timestamps(
        self,
        label: str = "Entity",
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """
        Add timestamps to existing nodes that don't have them.

        Args:
            label: Node label to backfill
            batch_size: Number of nodes per batch

        Returns:
            Backfill statistics
        """
        await self._client.connect()

        logger.info("Starting timestamp backfill", label=label)

        now = datetime.utcnow().isoformat()
        total_updated = 0

        # Backfill created_at
        if self.config.track_created_at:
            created_query = f"""
            MATCH (n:{label})
            WHERE n.created_at IS NULL
            WITH n LIMIT $batch_size
            SET n.created_at = datetime($now)
            RETURN count(n) as updated
            """

            while True:
                result = await self._client.execute_cypher(
                    created_query,
                    {"batch_size": batch_size, "now": now},
                )
                updated = result[0].get("updated", 0) if result else 0

                if updated == 0:
                    break

                total_updated += updated
                logger.info(f"Backfilled created_at for {updated} nodes", total=total_updated)

        # Backfill updated_at (set to created_at or now)
        if self.config.track_updated_at:
            updated_query = f"""
            MATCH (n:{label})
            WHERE n.updated_at IS NULL
            WITH n LIMIT $batch_size
            SET n.updated_at = coalesce(n.created_at, datetime($now))
            RETURN count(n) as updated
            """

            while True:
                result = await self._client.execute_cypher(
                    updated_query,
                    {"batch_size": batch_size, "now": now},
                )
                updated = result[0].get("updated", 0) if result else 0

                if updated == 0:
                    break

        # Backfill version
        if self.config.track_version:
            version_query = f"""
            MATCH (n:{label})
            WHERE n.version IS NULL
            WITH n LIMIT $batch_size
            SET n.version = 1
            RETURN count(n) as updated
            """

            while True:
                result = await self._client.execute_cypher(
                    version_query,
                    {"batch_size": batch_size},
                )
                updated = result[0].get("updated", 0) if result else 0

                if updated == 0:
                    break

        logger.info("Timestamp backfill completed", label=label, total_updated=total_updated)

        return {
            "label": label,
            "nodes_updated": total_updated,
            "fields_added": {
                "created_at": self.config.track_created_at,
                "updated_at": self.config.track_updated_at,
                "version": self.config.track_version,
            },
        }

    async def update_timestamp(
        self,
        node_id: str,
        label: str = "Entity",
        increment_version: bool = True,
    ) -> dict[str, Any]:
        """
        Update the timestamp for a specific node.

        Args:
            node_id: Node ID
            label: Node label
            increment_version: Whether to increment version

        Returns:
            Updated timestamp info
        """
        await self._client.connect()

        if increment_version and self.config.track_version:
            query = f"""
            MATCH (n:{label} {{id: $node_id}})
            SET n.updated_at = datetime(),
                n.version = coalesce(n.version, 0) + 1
            RETURN n.updated_at as updated_at, n.version as version
            """
        else:
            query = f"""
            MATCH (n:{label} {{id: $node_id}})
            SET n.updated_at = datetime()
            RETURN n.updated_at as updated_at, n.version as version
            """

        result = await self._client.execute_cypher(query, {"node_id": node_id})

        if result:
            return {
                "node_id": node_id,
                "updated_at": result[0].get("updated_at"),
                "version": result[0].get("version"),
            }

        return {"node_id": node_id, "error": "Node not found"}

    async def get_nodes_created_after(
        self,
        label: str,
        after: datetime,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get nodes created after a specific date.

        Args:
            label: Node label
            after: Datetime threshold
            limit: Maximum results

        Returns:
            List of matching nodes
        """
        await self._client.connect()

        query = f"""
        MATCH (n:{label})
        WHERE n.created_at >= datetime($after)
        RETURN n.id as id, n.name as name, n.created_at as created_at
        ORDER BY n.created_at DESC
        LIMIT $limit
        """

        return await self._client.execute_cypher(query, {
            "after": after.isoformat(),
            "limit": limit,
        })

    async def get_nodes_modified_after(
        self,
        label: str,
        after: datetime,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get nodes modified after a specific date.

        Args:
            label: Node label
            after: Datetime threshold
            limit: Maximum results

        Returns:
            List of matching nodes
        """
        await self._client.connect()

        query = f"""
        MATCH (n:{label})
        WHERE n.updated_at >= datetime($after)
        RETURN n.id as id, n.name as name, n.updated_at as updated_at, n.version as version
        ORDER BY n.updated_at DESC
        LIMIT $limit
        """

        return await self._client.execute_cypher(query, {
            "after": after.isoformat(),
            "limit": limit,
        })

    async def get_node_history(
        self,
        node_id: str,
        label: str = "Entity",
    ) -> dict[str, Any]:
        """
        Get timestamp history for a node.

        Args:
            node_id: Node ID
            label: Node label

        Returns:
            Node timestamp information
        """
        await self._client.connect()

        query = f"""
        MATCH (n:{label} {{id: $node_id}})
        RETURN n.id as id,
               n.name as name,
               n.created_at as created_at,
               n.updated_at as updated_at,
               n.version as version,
               n.deleted_at as deleted_at,
               n.restored_at as restored_at
        """

        result = await self._client.execute_cypher(query, {"node_id": node_id})

        if result:
            return result[0]

        return {"error": "Node not found", "node_id": node_id}

    async def get_timestamp_stats(
        self,
        label: str = "Entity",
    ) -> dict[str, Any]:
        """
        Get statistics about timestamps.

        Args:
            label: Node label

        Returns:
            Statistics dictionary
        """
        await self._client.connect()

        stats_query = f"""
        MATCH (n:{label})
        RETURN count(n) as total,
               count(n.created_at) as has_created_at,
               count(n.updated_at) as has_updated_at,
               count(n.version) as has_version,
               min(n.created_at) as oldest_created,
               max(n.created_at) as newest_created,
               max(n.updated_at) as last_updated,
               avg(n.version) as avg_version
        """

        result = await self._client.execute_cypher(stats_query)

        if result:
            data = result[0]
            total = data.get("total", 0)
            return {
                "label": label,
                "total_nodes": total,
                "coverage": {
                    "created_at": f"{data.get('has_created_at', 0)}/{total}",
                    "updated_at": f"{data.get('has_updated_at', 0)}/{total}",
                    "version": f"{data.get('has_version', 0)}/{total}",
                },
                "timeline": {
                    "oldest_created": data.get("oldest_created"),
                    "newest_created": data.get("newest_created"),
                    "last_updated": data.get("last_updated"),
                },
                "versioning": {
                    "average_version": round(data.get("avg_version", 0) or 0, 2),
                },
            }

        return {"error": "Query failed"}


def ensure_timestamps(query: str, operation: str = "MERGE") -> str:
    """
    Utility function to add timestamp SET clauses to a Cypher query.

    Args:
        query: Original Cypher query
        operation: "MERGE" or "CREATE"

    Returns:
        Modified query with timestamp handling
    """
    if operation == "MERGE":
        # Add ON CREATE and ON MATCH clauses
        timestamp_clause = """
        ON CREATE SET n.created_at = datetime(), n.updated_at = datetime(), n.version = 1
        ON MATCH SET n.updated_at = datetime(), n.version = coalesce(n.version, 0) + 1
        """
    else:
        # Just set created_at for CREATE
        timestamp_clause = """
        SET n.created_at = datetime(), n.updated_at = datetime(), n.version = 1
        """

    # Insert before RETURN clause if present
    if "RETURN" in query.upper():
        return_idx = query.upper().index("RETURN")
        return query[:return_idx] + timestamp_clause + query[return_idx:]
    else:
        return query + timestamp_clause


# Factory function
def create_timestamp_manager() -> TimestampManager:
    """Create a timestamp manager with default configuration."""
    return TimestampManager()

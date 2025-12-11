"""
Migration 002: Soft Delete Support.

Adds soft-delete fields to nodes:
- is_deleted: boolean flag
- deleted_at: timestamp
- deleted_by: who deleted
- deletion_reason: why deleted
"""

from typing import Any

from src.graph.migrations.base_migration import BaseMigration


class Migration002SoftDelete(BaseMigration):
    """Add soft-delete support to nodes."""

    version = "002"
    description = "Add soft-delete fields (is_deleted, deleted_at, deleted_by)"
    dependencies = ["001"]

    async def up(self, client: Any) -> None:
        """Add soft-delete fields and indexes."""

        # Set default is_deleted = false for existing nodes
        default_queries = [
            """
            MATCH (n:Entity)
            WHERE n.is_deleted IS NULL
            SET n.is_deleted = false
            """,
            """
            MATCH (n:Chunk)
            WHERE n.is_deleted IS NULL
            SET n.is_deleted = false
            """,
            """
            MATCH (n:Community)
            WHERE n.is_deleted IS NULL
            SET n.is_deleted = false
            """,
        ]

        for query in default_queries:
            await client.execute_cypher(query)

        # Create indexes for soft-delete queries
        indexes = [
            "CREATE INDEX entity_is_deleted IF NOT EXISTS FOR (n:Entity) ON (n.is_deleted)",
            "CREATE INDEX chunk_is_deleted IF NOT EXISTS FOR (n:Chunk) ON (n.is_deleted)",
            "CREATE INDEX community_is_deleted IF NOT EXISTS FOR (n:Community) ON (n.is_deleted)",
            "CREATE INDEX entity_deleted_at IF NOT EXISTS FOR (n:Entity) ON (n.deleted_at)",
            "CREATE INDEX chunk_deleted_at IF NOT EXISTS FOR (n:Chunk) ON (n.deleted_at)",
        ]

        for query in indexes:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

    async def down(self, client: Any) -> None:
        """Remove soft-delete fields and indexes."""

        # Drop indexes
        drops = [
            "DROP INDEX chunk_deleted_at IF EXISTS",
            "DROP INDEX entity_deleted_at IF EXISTS",
            "DROP INDEX community_is_deleted IF EXISTS",
            "DROP INDEX chunk_is_deleted IF EXISTS",
            "DROP INDEX entity_is_deleted IF EXISTS",
        ]

        for query in drops:
            try:
                await client.execute_cypher(query)
            except Exception:
                pass

        # Remove soft-delete properties
        remove_queries = [
            "MATCH (n:Entity) REMOVE n.is_deleted, n.deleted_at, n.deleted_by, n.deletion_reason",
            "MATCH (n:Chunk) REMOVE n.is_deleted, n.deleted_at, n.deleted_by, n.deletion_reason",
            "MATCH (n:Community) REMOVE n.is_deleted, n.deleted_at, n.deleted_by, n.deletion_reason",
        ]

        for query in remove_queries:
            await client.execute_cypher(query)

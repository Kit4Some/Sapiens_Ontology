"""
Migration 003: Timestamp Fields.

Adds timestamp tracking to nodes:
- created_at: when node was created
- updated_at: when node was last modified
- version: optimistic locking version number
"""

from typing import Any

from src.graph.migrations.base_migration import BaseMigration


class Migration003Timestamps(BaseMigration):
    """Add timestamp fields to nodes."""

    version = "003"
    description = "Add timestamp fields (created_at, updated_at, version)"
    dependencies = ["001", "002"]

    async def up(self, client: Any) -> None:
        """Add timestamp fields and backfill existing nodes."""

        # Backfill created_at for existing nodes
        backfill_queries = [
            """
            MATCH (n:Entity)
            WHERE n.created_at IS NULL
            SET n.created_at = datetime(),
                n.updated_at = datetime(),
                n.version = 1
            """,
            """
            MATCH (n:Chunk)
            WHERE n.created_at IS NULL
            SET n.created_at = datetime(),
                n.updated_at = datetime(),
                n.version = 1
            """,
            """
            MATCH (n:Community)
            WHERE n.created_at IS NULL
            SET n.created_at = datetime(),
                n.updated_at = datetime(),
                n.version = 1
            """,
        ]

        for query in backfill_queries:
            await client.execute_cypher(query)

        # Create indexes for timestamp queries
        indexes = [
            "CREATE INDEX entity_created_at IF NOT EXISTS FOR (n:Entity) ON (n.created_at)",
            "CREATE INDEX chunk_created_at IF NOT EXISTS FOR (n:Chunk) ON (n.created_at)",
            "CREATE INDEX community_created_at IF NOT EXISTS FOR (n:Community) ON (n.created_at)",
            "CREATE INDEX entity_updated_at IF NOT EXISTS FOR (n:Entity) ON (n.updated_at)",
            "CREATE INDEX chunk_updated_at IF NOT EXISTS FOR (n:Chunk) ON (n.updated_at)",
            "CREATE INDEX entity_version IF NOT EXISTS FOR (n:Entity) ON (n.version)",
        ]

        for query in indexes:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

    async def down(self, client: Any) -> None:
        """Remove timestamp fields and indexes."""

        # Drop indexes
        drops = [
            "DROP INDEX entity_version IF EXISTS",
            "DROP INDEX chunk_updated_at IF EXISTS",
            "DROP INDEX entity_updated_at IF EXISTS",
            "DROP INDEX community_created_at IF EXISTS",
            "DROP INDEX chunk_created_at IF EXISTS",
            "DROP INDEX entity_created_at IF EXISTS",
        ]

        for query in drops:
            try:
                await client.execute_cypher(query)
            except Exception:
                pass

        # Remove timestamp properties
        remove_queries = [
            "MATCH (n:Entity) REMOVE n.created_at, n.updated_at, n.version",
            "MATCH (n:Chunk) REMOVE n.created_at, n.updated_at, n.version",
            "MATCH (n:Community) REMOVE n.created_at, n.updated_at, n.version",
        ]

        for query in remove_queries:
            await client.execute_cypher(query)

"""
Migration 001: Initial Schema.

Creates the base schema with:
- Uniqueness constraints on node IDs
- Property indexes for fast lookups
- Vector indexes for similarity search
- Full-text indexes for text search
"""

from typing import Any

from src.graph.migrations.base_migration import BaseMigration


class Migration001InitialSchema(BaseMigration):
    """Initial database schema setup."""

    version = "001"
    description = "Initial schema with constraints, indexes, and vector indexes"

    async def up(self, client: Any) -> None:
        """Create initial schema."""

        # Uniqueness constraints
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
        ]

        for query in constraints:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

        # Property indexes
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX chunk_source IF NOT EXISTS FOR (n:Chunk) ON (n.source)",
            "CREATE INDEX community_level IF NOT EXISTS FOR (n:Community) ON (n.level)",
        ]

        for query in indexes:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

        # Vector indexes
        vector_indexes = [
            """
            CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            """
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (n:Chunk) ON (n.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
        ]

        for query in vector_indexes:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

        # Full-text indexes
        fulltext_indexes = [
            """
            CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name, n.type, n.description]
            OPTIONS {
                indexConfig: {
                    `fulltext.analyzer`: 'standard',
                    `fulltext.eventually_consistent`: false
                }
            }
            """,
            """
            CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
            FOR (n:Chunk) ON EACH [n.text]
            OPTIONS {
                indexConfig: {
                    `fulltext.analyzer`: 'standard',
                    `fulltext.eventually_consistent`: false
                }
            }
            """,
            """
            CREATE FULLTEXT INDEX community_fulltext IF NOT EXISTS
            FOR (n:Community) ON EACH [n.summary]
            OPTIONS {
                indexConfig: {
                    `fulltext.analyzer`: 'standard',
                    `fulltext.eventually_consistent`: false
                }
            }
            """,
        ]

        for query in fulltext_indexes:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

    async def down(self, client: Any) -> None:
        """Remove initial schema."""

        # Drop indexes (in reverse order)
        drops = [
            "DROP INDEX community_fulltext IF EXISTS",
            "DROP INDEX chunk_fulltext IF EXISTS",
            "DROP INDEX entity_fulltext IF EXISTS",
            "DROP INDEX chunk_embedding IF EXISTS",
            "DROP INDEX entity_embedding IF EXISTS",
            "DROP INDEX community_level IF EXISTS",
            "DROP INDEX chunk_source IF EXISTS",
            "DROP INDEX entity_type IF EXISTS",
            "DROP INDEX entity_name IF EXISTS",
            "DROP CONSTRAINT community_id IF EXISTS",
            "DROP CONSTRAINT chunk_id IF EXISTS",
            "DROP CONSTRAINT entity_id IF EXISTS",
        ]

        for query in drops:
            try:
                await client.execute_cypher(query)
            except Exception:
                pass  # Ignore errors on drop

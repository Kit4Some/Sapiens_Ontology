"""
Migration 004: Lifecycle Composite Indexes.

Adds composite indexes for common lifecycle queries:
- Active entities (is_deleted = false)
- Recently modified entities
- Entities by type and status
"""

from typing import Any

from src.graph.migrations.base_migration import BaseMigration


class Migration004LifecycleIndexes(BaseMigration):
    """Add composite indexes for lifecycle queries."""

    version = "004"
    description = "Add composite indexes for lifecycle management queries"
    dependencies = ["002", "003"]

    async def up(self, client: Any) -> None:
        """Create composite indexes for efficient lifecycle queries."""

        # Composite indexes for common query patterns
        indexes = [
            # Active entities by type (most common query pattern)
            "CREATE INDEX entity_type_active IF NOT EXISTS FOR (n:Entity) ON (n.type, n.is_deleted)",

            # Recently updated entities
            "CREATE INDEX entity_updated_active IF NOT EXISTS FOR (n:Entity) ON (n.updated_at, n.is_deleted)",

            # Source document tracking
            "CREATE INDEX chunk_source_active IF NOT EXISTS FOR (n:Chunk) ON (n.source, n.is_deleted)",

            # Provenance tracking
            "CREATE INDEX entity_source_doc IF NOT EXISTS FOR (n:Entity) ON (n.source_doc_id)",
            "CREATE INDEX entity_pipeline IF NOT EXISTS FOR (n:Entity) ON (n.pipeline_id)",
        ]

        for query in indexes:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

        # Add relationship indexes for soft-delete
        rel_indexes = [
            # Note: Neo4j 5.x supports relationship property indexes
            "CREATE INDEX rel_relates_to_deleted IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.is_deleted)",
            "CREATE INDEX rel_contains_deleted IF NOT EXISTS FOR ()-[r:CONTAINS]-() ON (r.is_deleted)",
        ]

        for query in rel_indexes:
            try:
                await client.execute_cypher(query)
            except Exception as e:
                # Relationship indexes may not be supported in all Neo4j versions
                if "already exists" not in str(e).lower():
                    pass  # Silently skip if not supported

    async def down(self, client: Any) -> None:
        """Remove composite indexes."""

        drops = [
            "DROP INDEX rel_contains_deleted IF EXISTS",
            "DROP INDEX rel_relates_to_deleted IF EXISTS",
            "DROP INDEX entity_pipeline IF EXISTS",
            "DROP INDEX entity_source_doc IF EXISTS",
            "DROP INDEX chunk_source_active IF EXISTS",
            "DROP INDEX entity_updated_active IF EXISTS",
            "DROP INDEX entity_type_active IF EXISTS",
        ]

        for query in drops:
            try:
                await client.execute_cypher(query)
            except Exception:
                pass

"""
Ontology Graph Client Module.

Neo4j client for ontology-based knowledge graph operations.
Supports schema management, CRUD operations, and advanced search capabilities.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ClientError

from src.config.settings import get_settings
from src.graph.schema import (
    ChunkNode,
    CommunityNode,
    EntityNode,
    FulltextSearchResult,
    NodeLabel,
    Relationship,
    RelationType,
    VectorSearchResult,
)

logger = structlog.get_logger(__name__)


class OntologyGraphClient:
    """
    Neo4j client for ontology graph operations.

    Provides schema management, node/relationship CRUD,
    vector similarity search, and full-text search capabilities.
    """

    # Schema creation queries
    SCHEMA_CONSTRAINTS = [
        # Uniqueness constraints
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
    ]

    SCHEMA_INDEXES = [
        # Property indexes for fast lookups
        "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
        "CREATE INDEX chunk_source IF NOT EXISTS FOR (n:Chunk) ON (n.source)",
        "CREATE INDEX community_level IF NOT EXISTS FOR (n:Community) ON (n.level)",
    ]

    # Relationship property indexes for efficient traversal and filtering
    RELATIONSHIP_INDEXES = [
        # RELATES_TO weight index for weighted graph traversal
        "CREATE INDEX rel_weight IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.weight)",
        # RELATES_TO predicate index for specific relation type queries
        "CREATE INDEX rel_predicate IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.predicate)",
        # RELATES_TO confidence index for filtering by confidence threshold
        "CREATE INDEX rel_confidence IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.confidence)",
        # BELONGS_TO weight index for community membership queries
        "CREATE INDEX belongs_to_weight IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.weight)",
    ]

    # Vector indexes (Neo4j 5.x with vector index support)
    VECTOR_INDEXES = [
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

    # Full-text indexes (using standard analyzer for multilingual support including Korean)
    FULLTEXT_INDEXES = [
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

    # Relationship full-text indexes for semantic search on predicates
    RELATIONSHIP_FULLTEXT_INDEXES = [
        # Note: Neo4j fulltext indexes on relationships require Neo4j 4.3+
        # This enables searching for relations by predicate description
        """
        CREATE FULLTEXT INDEX relates_to_fulltext IF NOT EXISTS
        FOR ()-[r:RELATES_TO]-() ON EACH [r.predicate, r.description]
        OPTIONS {
            indexConfig: {
                `fulltext.analyzer`: 'standard',
                `fulltext.eventually_consistent`: false
            }
        }
        """,
    ]

    def __init__(self, settings: Any = None) -> None:
        """Initialize the client with optional custom settings."""
        self._driver: AsyncDriver | None = None
        self._settings = settings or get_settings().neo4j

    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        if self._driver is not None:
            return

        self._driver = AsyncGraphDatabase.driver(
            self._settings.uri,
            auth=(self._settings.username, self._settings.password.get_secret_value()),
            max_connection_pool_size=self._settings.max_connection_pool_size,
        )
        await self._driver.verify_connectivity()
        logger.info("Connected to Neo4j", uri=self._settings.uri)

    async def close(self) -> None:
        """Close the database connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        if self._driver is None:
            await self.connect()

        assert self._driver is not None  # Type guard for mypy
        async with self._driver.session(database=self._settings.database) as session:
            yield session

    # =========================================================================
    # Schema Management
    # =========================================================================

    async def setup_schema(self, vector_dimensions: int = 1536) -> dict[str, Any]:
        """
        Create all schema constraints, indexes, and vector indexes.

        Args:
            vector_dimensions: Dimension of vector embeddings (default: 1536 for OpenAI)

        Returns:
            Dictionary with creation results for each index type
        """
        results: dict[str, list[Any]] = {
            "constraints": [],
            "indexes": [],
            "relationship_indexes": [],
            "vector_indexes": [],
            "fulltext_indexes": [],
            "relationship_fulltext_indexes": [],
            "errors": [],
        }

        async with self.session() as session:
            # Create constraints
            for query in self.SCHEMA_CONSTRAINTS:
                try:
                    await session.run(query)
                    results["constraints"].append({"query": query[:50], "status": "created"})
                    logger.info("Constraint created", query=query[:50])
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        results["constraints"].append({"query": query[:50], "status": "exists"})
                    else:
                        results["errors"].append({"query": query[:50], "error": str(e)})
                        logger.warning("Constraint creation failed", error=str(e))

            # Create property indexes
            for query in self.SCHEMA_INDEXES:
                try:
                    await session.run(query)
                    results["indexes"].append({"query": query[:50], "status": "created"})
                    logger.info("Index created", query=query[:50])
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        results["indexes"].append({"query": query[:50], "status": "exists"})
                    else:
                        results["errors"].append({"query": query[:50], "error": str(e)})

            # Create relationship property indexes
            for query in self.RELATIONSHIP_INDEXES:
                try:
                    await session.run(query)
                    results["relationship_indexes"].append({"query": query[:50], "status": "created"})
                    logger.info("Relationship index created", query=query[:50])
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        results["relationship_indexes"].append({"query": query[:50], "status": "exists"})
                    else:
                        # Relationship indexes may not be supported in all Neo4j versions
                        results["errors"].append({"query": query[:50], "error": str(e)})
                        logger.warning("Relationship index creation failed", error=str(e))

            # Create vector indexes (with dimension parameter)
            for query_template in self.VECTOR_INDEXES:
                query = query_template.replace("1536", str(vector_dimensions))
                try:
                    await session.run(query)
                    results["vector_indexes"].append({"status": "created"})
                    logger.info("Vector index created")
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        results["vector_indexes"].append({"status": "exists"})
                    else:
                        results["errors"].append({"query": "vector_index", "error": str(e)})
                        logger.warning("Vector index creation failed", error=str(e))

            # Create full-text indexes
            for query in self.FULLTEXT_INDEXES:
                try:
                    await session.run(query)
                    results["fulltext_indexes"].append({"status": "created"})
                    logger.info("Fulltext index created")
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        results["fulltext_indexes"].append({"status": "exists"})
                    else:
                        results["errors"].append({"query": "fulltext_index", "error": str(e)})

            # Create relationship full-text indexes
            for query in self.RELATIONSHIP_FULLTEXT_INDEXES:
                try:
                    await session.run(query)
                    results["relationship_fulltext_indexes"].append({"status": "created"})
                    logger.info("Relationship fulltext index created")
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        results["relationship_fulltext_indexes"].append({"status": "exists"})
                    else:
                        # Relationship fulltext indexes require Neo4j 4.3+
                        results["errors"].append({"query": "relationship_fulltext", "error": str(e)})
                        logger.warning("Relationship fulltext index creation failed", error=str(e))

        logger.info("Schema setup completed", results=results)
        return results

    async def get_data_diagnostics(self) -> dict[str, Any]:
        """
        Comprehensive data diagnostics for troubleshooting.

        Returns:
            Dictionary with counts, index status, and sample data
        """
        diagnostics: dict[str, Any] = {
            "counts": {},
            "indexes": {"vector": [], "fulltext": []},
            "samples": {},
            "health": {"has_data": False, "has_embeddings": False, "issues": []},
        }

        async with self.session() as session:
            # Count nodes by label
            count_queries = {
                "entities": "MATCH (n:Entity) RETURN count(n) as count",
                "chunks": "MATCH (n:Chunk) RETURN count(n) as count",
                "communities": "MATCH (n:Community) RETURN count(n) as count",
                "relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            }

            for name, query in count_queries.items():
                try:
                    result = await session.run(query)
                    record = await result.single()
                    diagnostics["counts"][name] = record["count"] if record else 0
                except Exception as e:
                    diagnostics["counts"][name] = f"Error: {str(e)}"

            # Check for embeddings
            try:
                embed_query = """
                MATCH (n:Entity)
                WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
                RETURN count(n) as count
                """
                result = await session.run(embed_query)
                record = await result.single()
                diagnostics["counts"]["entities_with_embeddings"] = record["count"] if record else 0

                chunk_embed_query = """
                MATCH (n:Chunk)
                WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
                RETURN count(n) as count
                """
                result = await session.run(chunk_embed_query)
                record = await result.single()
                diagnostics["counts"]["chunks_with_embeddings"] = record["count"] if record else 0
            except Exception as e:
                diagnostics["health"]["issues"].append(f"Embedding check failed: {str(e)}")

            # Check indexes
            try:
                index_query = "SHOW INDEXES YIELD name, type, state, labelsOrTypes"
                result = await session.run(index_query)
                records = await result.data()
                for rec in records:
                    idx_info = {
                        "name": rec.get("name"),
                        "state": rec.get("state"),
                        "labels": rec.get("labelsOrTypes"),
                    }
                    if rec.get("type") == "VECTOR":
                        diagnostics["indexes"]["vector"].append(idx_info)
                    elif rec.get("type") == "FULLTEXT":
                        diagnostics["indexes"]["fulltext"].append(idx_info)
            except Exception as e:
                diagnostics["health"]["issues"].append(f"Index check failed: {str(e)}")

            # Get sample entities
            try:
                sample_query = """
                MATCH (n:Entity)
                RETURN n.id as id, n.name as name, n.type as type,
                       n.description as description
                LIMIT 5
                """
                result = await session.run(sample_query)
                diagnostics["samples"]["entities"] = await result.data()
            except Exception as e:
                diagnostics["health"]["issues"].append(f"Sample query failed: {str(e)}")

            # Get sample chunks
            try:
                chunk_sample_query = """
                MATCH (n:Chunk)
                RETURN n.id as id, substring(n.text, 0, 100) as text_preview, n.source as source
                LIMIT 3
                """
                result = await session.run(chunk_sample_query)
                diagnostics["samples"]["chunks"] = await result.data()
            except Exception as e:
                pass  # Optional, don't add to issues

        # Determine health status
        entity_count = diagnostics["counts"].get("entities", 0)
        if isinstance(entity_count, int):
            diagnostics["health"]["has_data"] = entity_count > 0
            embed_count = diagnostics["counts"].get("entities_with_embeddings", 0)
            if isinstance(embed_count, int):
                diagnostics["health"]["has_embeddings"] = embed_count > 0

        # Add recommendations
        if not diagnostics["health"]["has_data"]:
            diagnostics["health"]["issues"].append(
                "No entities found. Please ingest documents using the Data Ingestion pipeline."
            )
        elif not diagnostics["health"]["has_embeddings"]:
            diagnostics["health"]["issues"].append(
                "Entities exist but have no embeddings. Vector search will not work. "
                "Re-ingest documents or run embedding generation."
            )

        logger.info("Data diagnostics completed", diagnostics=diagnostics)
        return diagnostics

    async def get_schema(self) -> dict[str, Any]:
        """
        Retrieve the complete database schema.

        Returns:
            Dictionary containing node labels, relationship types,
            properties, constraints, and indexes
        """
        schema: dict[str, Any] = {
            "node_labels": [],
            "relationship_types": [],
            "node_properties": {},
            "relationship_properties": {},
            "constraints": [],
            "indexes": [],
        }

        async with self.session() as session:
            # Get node labels with properties
            node_query = """
            CALL db.schema.nodeTypeProperties()
            YIELD nodeLabels, propertyName, propertyTypes, mandatory
            RETURN nodeLabels,
                   collect({
                       name: propertyName,
                       types: propertyTypes,
                       mandatory: mandatory
                   }) as properties
            """
            result = await session.run(node_query)
            records = await result.data()
            for record in records:
                labels = record.get("nodeLabels", [])
                label_key = ":".join(labels) if labels else "Unknown"
                schema["node_labels"].extend(labels)
                schema["node_properties"][label_key] = record.get("properties", [])

            # Get relationship types with properties
            rel_query = """
            CALL db.schema.relTypeProperties()
            YIELD relType, propertyName, propertyTypes, mandatory
            RETURN relType,
                   collect({
                       name: propertyName,
                       types: propertyTypes,
                       mandatory: mandatory
                   }) as properties
            """
            result = await session.run(rel_query)
            records = await result.data()
            for record in records:
                rel_type = record.get("relType", "").replace(":`", "").replace("`", "")
                schema["relationship_types"].append(rel_type)
                schema["relationship_properties"][rel_type] = record.get("properties", [])

            # Get constraints
            constraint_query = "SHOW CONSTRAINTS"
            result = await session.run(constraint_query)
            schema["constraints"] = await result.data()

            # Get indexes
            index_query = "SHOW INDEXES"
            result = await session.run(index_query)
            schema["indexes"] = await result.data()

        # Deduplicate
        schema["node_labels"] = list(set(schema["node_labels"]))
        schema["relationship_types"] = list(set(schema["relationship_types"]))

        return schema

    # =========================================================================
    # Node Operations
    # =========================================================================

    async def create_entity(
        self,
        entity: EntityNode,
        merge: bool = True,
    ) -> dict[str, Any]:
        """
        Create or merge an Entity node.

        Args:
            entity: EntityNode model with node properties
            merge: If True, use MERGE (upsert); if False, use CREATE

        Returns:
            Created/updated node properties
        """
        operation = "MERGE" if merge else "CREATE"
        query = f"""
        {operation} (e:Entity {{id: $id}})
        SET e.name = $name,
            e.type = $type,
            e.embedding = $embedding,
            e.properties = $properties,
            e.updated_at = datetime()
        ON CREATE SET e.created_at = datetime()
        RETURN e {{.*}} as node
        """

        params = {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "embedding": entity.embedding,
            "properties": entity.properties,
        }

        async with self.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            logger.info("Entity created/updated", entity_id=entity.id)
            return record["node"] if record else {}

    async def create_chunk(
        self,
        chunk: ChunkNode,
        merge: bool = True,
    ) -> dict[str, Any]:
        """Create or merge a Chunk node."""
        operation = "MERGE" if merge else "CREATE"
        query = f"""
        {operation} (c:Chunk {{id: $id}})
        SET c.text = $text,
            c.embedding = $embedding,
            c.source = $source,
            c.position = $position,
            c.updated_at = datetime()
        ON CREATE SET c.created_at = datetime()
        RETURN c {{.*}} as node
        """

        params = {
            "id": chunk.id,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "source": chunk.source,
            "position": chunk.position,
        }

        async with self.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            logger.info("Chunk created/updated", chunk_id=chunk.id)
            return record["node"] if record else {}

    async def create_community(
        self,
        community: CommunityNode,
        merge: bool = True,
    ) -> dict[str, Any]:
        """Create or merge a Community node."""
        operation = "MERGE" if merge else "CREATE"
        query = f"""
        {operation} (c:Community {{id: $id}})
        SET c.level = $level,
            c.summary = $summary,
            c.member_count = $member_count,
            c.updated_at = datetime()
        ON CREATE SET c.created_at = datetime()
        RETURN c {{.*}} as node
        """

        params = {
            "id": community.id,
            "level": community.level,
            "summary": community.summary,
            "member_count": community.member_count,
        }

        async with self.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            logger.info("Community created/updated", community_id=community.id)
            return record["node"] if record else {}

    async def create_relationship(
        self,
        relationship: Relationship,
        source_label: NodeLabel = NodeLabel.ENTITY,
        target_label: NodeLabel = NodeLabel.ENTITY,
    ) -> dict[str, Any]:
        """
        Create a relationship between two nodes.

        Args:
            relationship: Relationship model
            source_label: Label of source node
            target_label: Label of target node

        Returns:
            Relationship properties
        """
        query = f"""
        MATCH (source:{source_label.value} {{id: $source_id}})
        MATCH (target:{target_label.value} {{id: $target_id}})
        MERGE (source)-[r:{relationship.type.value}]->(target)
        SET r.weight = $weight,
            r.properties = $properties,
            r.updated_at = datetime()
        ON CREATE SET r.created_at = datetime()
        RETURN type(r) as type, r {{.*}} as properties
        """

        params = {
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "weight": relationship.weight,
            "properties": relationship.properties,
        }

        async with self.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            logger.info(
                "Relationship created",
                source=relationship.source_id,
                target=relationship.target_id,
                type=relationship.type.value,
            )
            return {"type": record["type"], "properties": record["properties"]} if record else {}

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def vector_search(
        self,
        embedding: list[float],
        node_label: NodeLabel = NodeLabel.ENTITY,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Perform vector similarity search using vector index.

        Args:
            embedding: Query vector embedding
            node_label: Node label to search (Entity or Chunk)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of VectorSearchResult sorted by similarity
        """
        index_name = f"{node_label.value.lower()}_embedding"

        query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        WHERE score >= $min_score
        RETURN node.id as node_id,
               labels(node)[0] as node_label,
               score,
               properties(node) as properties
        ORDER BY score DESC
        """

        params = {
            "index_name": index_name,
            "top_k": top_k,
            "embedding": embedding,
            "min_score": min_score,
        }

        async with self.session() as session:
            result = await session.run(query, params)
            records = await result.data()

        results = [
            VectorSearchResult(
                node_id=r["node_id"],
                node_label=r["node_label"],
                score=r["score"],
                properties=r["properties"],
            )
            for r in records
        ]

        logger.info(
            "Vector search completed",
            node_label=node_label.value,
            results_count=len(results),
        )
        return results

    async def fulltext_search(
        self,
        query_text: str,
        node_label: NodeLabel | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[FulltextSearchResult]:
        """
        Perform full-text search across nodes.

        Args:
            query_text: Search query string (supports Lucene syntax)
            node_label: Optional node label to search (None = search all indexes)
            top_k: Number of results to return
            min_score: Minimum relevance score threshold

        Returns:
            List of FulltextSearchResult sorted by relevance
        """
        # Escape special Lucene characters for safer queries
        escaped_query = self._escape_lucene_query(query_text)

        results: list[FulltextSearchResult] = []

        async with self.session() as session:
            if node_label:
                # Search specific index
                index_name = f"{node_label.value.lower()}_fulltext"
                query = """
                CALL db.index.fulltext.queryNodes($index_name, $query_text)
                YIELD node, score
                WHERE score >= $min_score
                RETURN node.id as node_id,
                       labels(node)[0] as node_label,
                       score,
                       CASE labels(node)[0]
                           WHEN 'Entity' THEN coalesce(node.name, '')
                           WHEN 'Chunk' THEN coalesce(node.text, '')
                           WHEN 'Community' THEN coalesce(node.summary, '')
                           ELSE ''
                       END as text
                ORDER BY score DESC
                LIMIT $top_k
                """
                params = {
                    "index_name": index_name,
                    "query_text": escaped_query,
                    "top_k": top_k,
                    "min_score": min_score,
                }
                result = await session.run(query, params)
                records = await result.data()

                for r in records:
                    results.append(
                        FulltextSearchResult(
                            node_id=r["node_id"],
                            node_label=r["node_label"],
                            score=r["score"],
                            text=r["text"] or "",
                            highlights=[],
                        )
                    )
            else:
                # Search ALL fulltext indexes and merge results
                indexes = [
                    ("entity_fulltext", "Entity", "name"),
                    ("chunk_fulltext", "Chunk", "text"),
                    ("community_fulltext", "Community", "summary"),
                ]

                all_records: list[dict[str, Any]] = []
                per_index_limit = max(top_k // len(indexes) + 2, 5)

                for index_name, label, text_field in indexes:
                    try:
                        query = f"""
                        CALL db.index.fulltext.queryNodes($index_name, $query_text)
                        YIELD node, score
                        WHERE score >= $min_score
                        RETURN node.id as node_id,
                               '{label}' as node_label,
                               score,
                               coalesce(node.{text_field}, '') as text
                        ORDER BY score DESC
                        LIMIT $limit
                        """
                        params = {
                            "index_name": index_name,
                            "query_text": escaped_query,
                            "min_score": min_score,
                            "limit": per_index_limit,
                        }
                        result = await session.run(query, params)
                        records = await result.data()
                        all_records.extend(records)
                        logger.debug(
                            "Fulltext index search",
                            index=index_name,
                            results=len(records),
                        )
                    except Exception as e:
                        logger.debug(f"Index {index_name} search failed: {e}")
                        continue

                # Sort by score and deduplicate
                all_records.sort(key=lambda x: x["score"], reverse=True)
                seen_ids: set[str] = set()
                for r in all_records:
                    if r["node_id"] not in seen_ids and len(results) < top_k:
                        seen_ids.add(r["node_id"])
                        results.append(
                            FulltextSearchResult(
                                node_id=r["node_id"],
                                node_label=r["node_label"],
                                score=r["score"],
                                text=r["text"] or "",
                                highlights=[],
                            )
                        )

        logger.info(
            "Fulltext search completed",
            query=query_text[:50],
            results_count=len(results),
            searched_all=node_label is None,
        )
        return results

    def _escape_lucene_query(self, query: str) -> str:
        """
        Escape special Lucene query characters.

        This helps handle Korean and special characters properly.
        """
        # Special chars that need escaping in Lucene
        special_chars = r'+-&|!(){}[]^"~*?:\/'
        escaped = []
        for char in query:
            if char in special_chars:
                escaped.append(f"\\{char}")
            else:
                escaped.append(char)
        return "".join(escaped)

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    async def get_neighbors(
        self,
        node_id: str,
        node_label: NodeLabel = NodeLabel.ENTITY,
        relationship_types: list[RelationType] | None = None,
        direction: str = "BOTH",  # "OUTGOING", "INCOMING", "BOTH"
        max_hops: int = 1,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get N-hop neighbors of a node.

        Args:
            node_id: Starting node ID
            node_label: Label of starting node
            relationship_types: Filter by relationship types (None = all)
            direction: Traversal direction
            max_hops: Maximum number of hops (1-5)
            limit: Maximum number of results

        Returns:
            Dictionary with nodes and relationships
        """
        max_hops = min(max_hops, 5)  # Safety limit

        # Build relationship pattern
        if relationship_types:
            rel_types = "|".join(rt.value for rt in relationship_types)
            rel_pattern = f"[r:{rel_types}*1..{max_hops}]"
        else:
            rel_pattern = f"[r*1..{max_hops}]"

        # Build direction pattern
        if direction == "OUTGOING":
            pattern = f"-{rel_pattern}->"
        elif direction == "INCOMING":
            pattern = f"<-{rel_pattern}-"
        else:
            pattern = f"-{rel_pattern}-"

        query = f"""
        MATCH (start:{node_label.value} {{id: $node_id}})
        MATCH path = (start){pattern}(neighbor)
        WITH neighbor,
             relationships(path) as rels,
             length(path) as distance
        RETURN DISTINCT
               neighbor.id as neighbor_id,
               labels(neighbor)[0] as neighbor_label,
               properties(neighbor) as properties,
               distance,
               [rel in rels | {{type: type(rel), properties: properties(rel)}}] as relationships
        ORDER BY distance ASC, neighbor_id
        LIMIT $limit
        """

        params = {
            "node_id": node_id,
            "limit": limit,
        }

        async with self.session() as session:
            result = await session.run(query, params)
            records = await result.data()

        logger.info(
            "Neighbors retrieved",
            node_id=node_id,
            max_hops=max_hops,
            results_count=len(records),
        )

        return {
            "start_node": node_id,
            "max_hops": max_hops,
            "neighbors": records,
            "total_count": len(records),
        }

    async def get_paths(
        self,
        source_id: str,
        target_id: str,
        source_label: NodeLabel = NodeLabel.ENTITY,
        target_label: NodeLabel = NodeLabel.ENTITY,
        max_hops: int = 4,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find shortest paths between two nodes.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID
            source_label: Label of source node
            target_label: Label of target node
            max_hops: Maximum path length
            limit: Maximum number of paths

        Returns:
            List of paths with nodes and relationships
        """
        query = f"""
        MATCH (source:{source_label.value} {{id: $source_id}})
        MATCH (target:{target_label.value} {{id: $target_id}})
        MATCH path = shortestPath((source)-[*1..{max_hops}]-(target))
        RETURN [node in nodes(path) | {{
                   id: node.id,
                   label: labels(node)[0],
                   name: coalesce(node.name, node.text, node.summary, node.id)
               }}] as nodes,
               [rel in relationships(path) | {{
                   type: type(rel),
                   properties: properties(rel)
               }}] as relationships,
               length(path) as path_length
        LIMIT $limit
        """

        params = {
            "source_id": source_id,
            "target_id": target_id,
            "limit": limit,
        }

        async with self.session() as session:
            result = await session.run(query, params)
            records: list[dict[str, Any]] = await result.data()

        logger.info(
            "Paths found",
            source=source_id,
            target=target_id,
            path_count=len(records),
        )

        return records

    # =========================================================================
    # Generic Query Execution
    # =========================================================================

    async def execute_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a raw Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records: list[dict[str, Any]] = await result.data()

        logger.debug(
            "Cypher executed",
            query=query[:100],
            param_count=len(parameters) if parameters else 0,
            result_count=len(records),
        )

        return records

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a write Cypher query with transaction.

        Returns summary statistics.
        """
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            summary = await result.consume()

        stats = {
            "nodes_created": summary.counters.nodes_created,
            "nodes_deleted": summary.counters.nodes_deleted,
            "relationships_created": summary.counters.relationships_created,
            "relationships_deleted": summary.counters.relationships_deleted,
            "properties_set": summary.counters.properties_set,
        }

        logger.info("Write query executed", **stats)
        return stats

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def bulk_create_entities(
        self,
        entities: list[EntityNode],
        batch_size: int = 500,
    ) -> dict[str, int]:
        """
        Bulk create/merge Entity nodes.

        Args:
            entities: List of EntityNode models
            batch_size: Number of nodes per transaction

        Returns:
            Statistics about created/updated nodes
        """
        query = """
        UNWIND $batch as entity
        MERGE (e:Entity {id: entity.id})
        SET e.name = entity.name,
            e.type = entity.type,
            e.embedding = entity.embedding,
            e.properties = entity.properties,
            e.updated_at = datetime()
        ON CREATE SET e.created_at = datetime()
        RETURN count(*) as processed
        """

        total_processed = 0
        for i in range(0, len(entities), batch_size):
            batch = [e.model_dump() for e in entities[i : i + batch_size]]
            async with self.session() as session:
                result = await session.run(query, {"batch": batch})
                record = await result.single()
                total_processed += record["processed"] if record else 0

        logger.info("Bulk entity creation completed", total=total_processed)
        return {"processed": total_processed, "total": len(entities)}

    async def bulk_create_relationships(
        self,
        relationships: list[Relationship],
        batch_size: int = 500,
    ) -> dict[str, int]:
        """Bulk create relationships."""
        # Group by relationship type for efficient processing
        by_type: dict[str, list[Relationship]] = {}
        for rel in relationships:
            by_type.setdefault(rel.type.value, []).append(rel)

        total_processed = 0
        for rel_type, rels in by_type.items():
            query = f"""
            UNWIND $batch as rel
            MATCH (source {{id: rel.source_id}})
            MATCH (target {{id: rel.target_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r.weight = rel.weight,
                r.properties = rel.properties,
                r.updated_at = datetime()
            ON CREATE SET r.created_at = datetime()
            RETURN count(*) as processed
            """

            for i in range(0, len(rels), batch_size):
                batch = [r.model_dump() for r in rels[i : i + batch_size]]
                async with self.session() as session:
                    result = await session.run(query, {"batch": batch})
                    record = await result.single()
                    total_processed += record["processed"] if record else 0

        logger.info("Bulk relationship creation completed", total=total_processed)
        return {"processed": total_processed, "total": len(relationships)}


# Singleton instance
_client: OntologyGraphClient | None = None


def get_ontology_client() -> OntologyGraphClient:
    """Get the singleton OntologyGraphClient instance."""
    global _client
    if _client is None:
        _client = OntologyGraphClient()
    return _client

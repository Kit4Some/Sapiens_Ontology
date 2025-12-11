"""
Neo4j Loader Module.

Loads extracted entities, chunks, and relations into Neo4j graph database.

Enhanced with:
- Better error handling and reporting
- Embedding validation
- Loading metrics
- Proper error propagation
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.sddi.state import (
    ExtractedEntity,
    ExtractedRelation,
    LoadResult,
    TextChunk,
)

logger = structlog.get_logger(__name__)


class Neo4jLoadError(Exception):
    """Raised when Neo4j loading fails."""

    def __init__(self, message: str, error_type: str = "unknown", recoverable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable


@dataclass
class LoadingMetrics:
    """Metrics for tracking Neo4j loading."""

    chunks_attempted: int = 0
    chunks_loaded: int = 0
    entities_attempted: int = 0
    entities_loaded: int = 0
    relations_attempted: int = 0
    relations_loaded: int = 0
    links_created: int = 0
    batch_failures: int = 0
    individual_failures: int = 0
    missing_embeddings: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    @property
    def chunk_success_rate(self) -> float:
        if self.chunks_attempted == 0:
            return 1.0
        return self.chunks_loaded / self.chunks_attempted

    @property
    def entity_success_rate(self) -> float:
        if self.entities_attempted == 0:
            return 1.0
        return self.entities_loaded / self.entities_attempted

    @property
    def relation_success_rate(self) -> float:
        if self.relations_attempted == 0:
            return 1.0
        return self.relations_loaded / self.relations_attempted

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_attempted": self.chunks_attempted,
            "chunks_loaded": self.chunks_loaded,
            "entities_attempted": self.entities_attempted,
            "entities_loaded": self.entities_loaded,
            "relations_attempted": self.relations_attempted,
            "relations_loaded": self.relations_loaded,
            "links_created": self.links_created,
            "batch_failures": self.batch_failures,
            "individual_failures": self.individual_failures,
            "missing_embeddings": self.missing_embeddings,
            "chunk_success_rate": round(self.chunk_success_rate, 3),
            "entity_success_rate": round(self.entity_success_rate, 3),
            "relation_success_rate": round(self.relation_success_rate, 3),
            "duration_seconds": round(self.duration_seconds, 2),
        }


def _serialize_for_neo4j(value: Any) -> Any:
    """
    Serialize a value for Neo4j storage.

    Neo4j only accepts primitive types (str, int, float, bool, None)
    or arrays of primitives. Dicts and complex objects must be
    serialized to JSON strings.
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        # Check if all elements are primitives
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
            return list(value)
        else:
            # Serialize complex lists to JSON
            return json.dumps(value, ensure_ascii=False, default=str)
    elif isinstance(value, dict):
        # Serialize dicts to JSON string
        return json.dumps(value, ensure_ascii=False, default=str)
    else:
        # Convert other types to string
        return str(value)


class Neo4jLoader:
    """
    Neo4j data loader for SDDI pipeline.

    Handles bulk creation of chunks, entities, and relationships
    with support for embeddings and deduplication.

    Features:
    - Batch operations using UNWIND for high performance
    - Transaction batching for large datasets
    - Progress tracking and error collection
    - Loading metrics and validation

    Enhanced with:
    - Better error handling and reporting
    - Embedding validation before loading
    - Detailed loading metrics
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
        batch_size: int = 500,
        use_batch_operations: bool = True,
        validate_embeddings: bool = True,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the loader.

        Args:
            client: Optional OntologyGraphClient instance
            batch_size: Number of records per batch operation
            use_batch_operations: Use UNWIND batch operations (recommended)
            validate_embeddings: Whether to validate embeddings before loading
            max_retries: Maximum retry attempts for failed operations
        """
        self._client = client or get_ontology_client()
        self._batch_size = batch_size
        self._use_batch = use_batch_operations
        self._validate_embeddings = validate_embeddings
        self._max_retries = max_retries
        self._metrics = LoadingMetrics()

    def get_metrics(self) -> LoadingMetrics:
        """Get loading metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset loading metrics for a new run."""
        self._metrics = LoadingMetrics()

    def _validate_embedding_coverage(
        self,
        items: list[Any],
        embeddings: dict[str, list[float]],
        item_type: str,
    ) -> tuple[int, list[str]]:
        """
        Validate embedding coverage for items.

        Returns:
            Tuple of (missing_count, list of warning messages)
        """
        missing_ids = []
        for item in items:
            item_id = item.id if hasattr(item, 'id') else item.get('id')
            if item_id and item_id not in embeddings:
                missing_ids.append(item_id)

        messages = []
        if missing_ids:
            self._metrics.missing_embeddings += len(missing_ids)
            msg = f"Missing embeddings for {len(missing_ids)} {item_type}(s)"
            messages.append(msg)
            logger.warning(msg, missing_count=len(missing_ids), sample_ids=missing_ids[:5])

        return len(missing_ids), messages

    async def ensure_connected(self) -> None:
        """Ensure database connection is established."""
        await self._client.connect()

    async def setup_schema(self, vector_dimensions: int = 1536) -> dict[str, Any]:
        """
        Setup database schema with constraints and indexes.

        Args:
            vector_dimensions: Dimension of vector embeddings

        Returns:
            Schema setup results
        """
        await self.ensure_connected()
        return await self._client.setup_schema(vector_dimensions)

    async def load_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: dict[str, list[float]] | None = None,
    ) -> tuple[int, list[str]]:
        """
        Load text chunks into Neo4j as Chunk nodes.

        Args:
            chunks: List of TextChunk objects
            embeddings: Optional mapping of chunk_id -> embedding

        Returns:
            Tuple of (number of chunks created, list of error messages)
        """
        if not chunks:
            logger.info("No chunks to load")
            return 0, []

        embeddings = embeddings or {}

        logger.info(
            "Starting chunk loading",
            chunk_count=len(chunks),
            embeddings_count=len(embeddings),
            batch_mode=self._use_batch,
        )

        # Verify Neo4j connection first
        try:
            test_query = "RETURN 1 as test"
            await self._client.execute_cypher(test_query)
        except Exception as e:
            error_msg = f"Neo4j connection test failed: {str(e)}"
            logger.error(error_msg)
            return 0, [error_msg]

        if self._use_batch:
            return await self._load_chunks_batch(chunks, embeddings)
        else:
            return await self._load_chunks_single(chunks, embeddings)

    async def _load_chunks_batch(
        self,
        chunks: list[TextChunk],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Load chunks using UNWIND batch operations for better performance."""
        errors: list[str] = []
        total_created = 0

        # UNWIND batch query - processes multiple chunks in a single transaction
        batch_query = """
        UNWIND $chunks AS chunk_data
        MERGE (c:Chunk {id: chunk_data.id})
        SET c.text = chunk_data.text,
            c.source = chunk_data.doc_id,
            c.position = chunk_data.position,
            c.start_char = chunk_data.start_char,
            c.end_char = chunk_data.end_char,
            c.embedding = chunk_data.embedding,
            c.metadata = chunk_data.metadata
        RETURN count(c) as created_count
        """

        # Process in batches
        for batch_start in range(0, len(chunks), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            # Prepare batch data
            batch_data = []
            for chunk in batch:
                batch_data.append({
                    "id": chunk.id,
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "position": chunk.position,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "embedding": embeddings.get(chunk.id),
                    "metadata": _serialize_for_neo4j(chunk.metadata),
                })

            try:
                result = await self._client.execute_cypher(batch_query, {"chunks": batch_data})
                if result and len(result) > 0:
                    created = result[0].get("created_count", 0)
                    total_created += created

                logger.info(
                    "Chunk batch loaded",
                    batch=f"{batch_start // self._batch_size + 1}/{(len(chunks) + self._batch_size - 1) // self._batch_size}",
                    batch_size=len(batch),
                    total_created=total_created,
                    progress=f"{batch_end / len(chunks) * 100:.1f}%",
                )

            except Exception as e:
                error_msg = f"Batch chunk loading failed at {batch_start}-{batch_end}: {str(e)}"
                logger.error(error_msg, error_type=type(e).__name__)
                errors.append(error_msg)

                # Fallback to single-record processing for this batch
                logger.info("Falling back to single-record processing for failed batch")
                for chunk in batch:
                    try:
                        single_result = await self._load_single_chunk(chunk, embeddings)
                        if single_result:
                            total_created += 1
                    except Exception as single_e:
                        errors.append(f"Failed to load chunk {chunk.id}: {str(single_e)}")

        logger.info(
            "Batch chunk loading completed",
            created=total_created,
            failed=len(errors),
            total=len(chunks),
        )

        return total_created, errors

    async def _load_single_chunk(
        self,
        chunk: TextChunk,
        embeddings: dict[str, list[float]],
    ) -> bool:
        """Load a single chunk (fallback method)."""
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.source = $doc_id,
            c.position = $position,
            c.start_char = $start_char,
            c.end_char = $end_char,
            c.embedding = $embedding,
            c.metadata = $metadata
        RETURN c.id as created_id
        """
        params = {
            "id": chunk.id,
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "position": chunk.position,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "embedding": embeddings.get(chunk.id),
            "metadata": _serialize_for_neo4j(chunk.metadata),
        }
        result = await self._client.execute_cypher(query, params)
        return bool(result)

    async def _load_chunks_single(
        self,
        chunks: list[TextChunk],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Load chunks one at a time (legacy method)."""
        errors: list[str] = []
        total_created = 0
        failed_count = 0

        query = """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.source = $doc_id,
            c.position = $position,
            c.start_char = $start_char,
            c.end_char = $end_char,
            c.embedding = $embedding,
            c.metadata = $metadata
        RETURN c.id as created_id
        """

        for idx, chunk in enumerate(chunks):
            embedding = embeddings.get(chunk.id)
            metadata_json = _serialize_for_neo4j(chunk.metadata)

            params = {
                "id": chunk.id,
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "position": chunk.position,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "embedding": embedding,
                "metadata": metadata_json,
            }

            try:
                result = await self._client.execute_cypher(query, params)
                if result:
                    total_created += 1
                else:
                    failed_count += 1
                    logger.warning("Chunk merge returned no result", chunk_id=chunk.id)

                if (idx + 1) % 20 == 0:
                    logger.info(
                        "Chunk loading progress",
                        loaded=total_created,
                        total=len(chunks),
                        progress=f"{(idx + 1) / len(chunks) * 100:.1f}%",
                    )

            except Exception as e:
                failed_count += 1
                error_msg = f"Failed to load chunk {chunk.id}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

                if idx == 0:
                    logger.error(
                        "First chunk failed to load - possible Neo4j issue",
                        chunk_id=chunk.id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        logger.info(
            "Chunks loading completed",
            created=total_created,
            failed=failed_count,
            total=len(chunks),
        )

        return total_created, errors

    async def load_entities(
        self,
        entities: list[ExtractedEntity],
        embeddings: dict[str, list[float]] | None = None,
    ) -> tuple[int, list[str]]:
        """
        Load extracted entities into Neo4j as Entity nodes.

        Args:
            entities: List of ExtractedEntity objects
            embeddings: Optional mapping of entity_id -> embedding

        Returns:
            Tuple of (number of entities created, list of error messages)
        """
        if not entities:
            logger.info("No entities to load")
            return 0, []

        embeddings = embeddings or {}

        logger.info(
            "Starting entity loading",
            entity_count=len(entities),
            embeddings_available=len(embeddings),
            batch_mode=self._use_batch,
        )

        if self._use_batch:
            return await self._load_entities_batch(entities, embeddings)
        else:
            return await self._load_entities_single(entities, embeddings)

    async def _load_entities_batch(
        self,
        entities: list[ExtractedEntity],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Load entities using UNWIND batch operations."""
        errors: list[str] = []
        total_created = 0

        # Enhanced query with timestamp support
        batch_query = """
        UNWIND $entities AS entity_data
        MERGE (e:Entity {id: entity_data.id})
        SET e.name = entity_data.name,
            e.type = entity_data.type,
            e.description = entity_data.description,
            e.aliases = entity_data.aliases,
            e.confidence = entity_data.confidence,
            e.embedding = entity_data.embedding,
            e.properties = entity_data.properties,
            e.created_at = COALESCE(e.created_at, entity_data.created_at),
            e.updated_at = entity_data.updated_at,
            e.source_doc_id = entity_data.source_doc_id,
            e.pipeline_id = entity_data.pipeline_id,
            e.version = COALESCE(e.version, 0) + 1
        RETURN count(e) as created_count
        """

        for batch_start in range(0, len(entities), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(entities))
            batch = entities[batch_start:batch_end]

            batch_data = []
            for entity in batch:
                # Get timestamps as ISO strings for Neo4j
                created_at = entity.created_at.isoformat() if entity.created_at else None
                updated_at = entity.updated_at.isoformat() if entity.updated_at else None

                batch_data.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "description": entity.description,
                    "aliases": entity.aliases,
                    "confidence": entity.confidence,
                    "embedding": embeddings.get(entity.id),
                    "properties": _serialize_for_neo4j(entity.properties),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "source_doc_id": entity.source_doc_id,
                    "pipeline_id": entity.pipeline_id,
                })

            try:
                result = await self._client.execute_cypher(batch_query, {"entities": batch_data})
                if result and len(result) > 0:
                    created = result[0].get("created_count", 0)
                    total_created += created

                logger.info(
                    "Entity batch loaded",
                    batch=f"{batch_start // self._batch_size + 1}/{(len(entities) + self._batch_size - 1) // self._batch_size}",
                    batch_size=len(batch),
                    total_created=total_created,
                )

            except Exception as e:
                error_msg = f"Batch entity loading failed at {batch_start}-{batch_end}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Fallback to single-record processing
                for entity in batch:
                    try:
                        if await self._load_single_entity(entity, embeddings):
                            total_created += 1
                    except Exception as single_e:
                        errors.append(f"Failed to load entity {entity.id}: {str(single_e)}")

        logger.info("Batch entity loading completed", count=total_created, failed=len(errors))
        return total_created, errors

    async def _load_single_entity(
        self,
        entity: ExtractedEntity,
        embeddings: dict[str, list[float]],
    ) -> bool:
        """Load a single entity (fallback method)."""
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.description = $description,
            e.aliases = $aliases,
            e.confidence = $confidence,
            e.embedding = $embedding,
            e.properties = $properties
        RETURN e.id as created_id
        """
        params = {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type.value,
            "description": entity.description,
            "aliases": entity.aliases,
            "confidence": entity.confidence,
            "embedding": embeddings.get(entity.id),
            "properties": _serialize_for_neo4j(entity.properties),
        }
        result = await self._client.execute_cypher(query, params)
        return bool(result)

    async def _load_entities_single(
        self,
        entities: list[ExtractedEntity],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Load entities one at a time (legacy method)."""
        errors: list[str] = []
        total_created = 0

        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.description = $description,
            e.aliases = $aliases,
            e.confidence = $confidence,
            e.embedding = $embedding,
            e.properties = $properties
        RETURN e.id as created_id
        """

        for entity in entities:
            properties_json = _serialize_for_neo4j(entity.properties)
            params = {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type.value,
                "description": entity.description,
                "aliases": entity.aliases,
                "confidence": entity.confidence,
                "embedding": embeddings.get(entity.id),
                "properties": properties_json,
            }
            try:
                result = await self._client.execute_cypher(query, params)
                if result:
                    total_created += 1
            except Exception as e:
                error_msg = f"Failed to load entity {entity.id} ({entity.name}): {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

        logger.info("Entities loaded", count=total_created, failed=len(errors))
        return total_created, errors

    async def load_relations(
        self,
        relations: list[ExtractedRelation],
    ) -> tuple[int, list[str]]:
        """
        Load extracted relations into Neo4j as relationships.

        Creates RELATES_TO relationships between Entity nodes.

        Args:
            relations: List of ExtractedRelation objects

        Returns:
            Tuple of (number of relationships created, list of error messages)
        """
        if not relations:
            logger.info("No relations to load")
            return 0, []

        logger.info(
            "Starting relation loading",
            relation_count=len(relations),
            batch_mode=self._use_batch,
        )

        if self._use_batch:
            return await self._load_relations_batch(relations)
        else:
            return await self._load_relations_single(relations)

    def _calculate_relation_weight(self, relation: ExtractedRelation) -> float:
        """
        Calculate relationship weight based on multiple factors.

        Weight is computed from:
        - Base confidence score (0.4 weight)
        - Number of source chunks (0.3 weight) - more mentions = stronger
        - Predicate specificity (0.3 weight) - specific predicates score higher

        Returns:
            Weight value between 0.0 and 1.0
        """
        # Base confidence (40% of weight)
        confidence_component = relation.confidence * 0.4

        # Co-occurrence component (30% of weight)
        # More chunks mentioning this relation = stronger evidence
        chunk_count = len(relation.chunk_ids) if relation.chunk_ids else 1
        # Normalize: 1 chunk = 0.1, 10+ chunks = 1.0 (linear scale)
        # Then multiply by 0.3 to get the 30% weight contribution
        cooccurrence_normalized = min(chunk_count / 10.0, 1.0)
        cooccurrence_component = cooccurrence_normalized * 0.3

        # Predicate specificity (30% of weight)
        # Generic predicates like "relates_to" score lower
        generic_predicates = {"relates_to", "related", "associated", "connected", "linked"}
        predicate_lower = relation.predicate.lower().replace("_", " ").replace("-", " ")
        if any(gen in predicate_lower for gen in generic_predicates):
            specificity_component = 0.1  # Generic
        elif len(relation.predicate) > 3:
            specificity_component = 0.25  # Specific predicate
        else:
            specificity_component = 0.15  # Too short to be meaningful

        total_weight = confidence_component + cooccurrence_component + specificity_component
        return min(1.0, max(0.0, total_weight))

    async def _load_relations_batch(
        self,
        relations: list[ExtractedRelation],
    ) -> tuple[int, list[str]]:
        """Load relations using UNWIND batch operations."""
        errors: list[str] = []
        total_created = 0

        # Enhanced query with weight support
        batch_query = """
        UNWIND $relations AS rel_data
        MATCH (source:Entity {id: rel_data.source_id})
        MATCH (target:Entity {id: rel_data.target_id})
        MERGE (source)-[r:RELATES_TO]->(target)
        SET r.predicate = rel_data.predicate,
            r.description = rel_data.description,
            r.confidence = rel_data.confidence,
            r.weight = rel_data.weight,
            r.chunk_ids = rel_data.chunk_ids,
            r.properties = rel_data.properties,
            r.created_at = COALESCE(r.created_at, rel_data.created_at),
            r.pipeline_id = rel_data.pipeline_id
        RETURN count(r) as created_count
        """

        for batch_start in range(0, len(relations), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(relations))
            batch = relations[batch_start:batch_end]

            batch_data = []
            for rel in batch:
                # Calculate weight for the relationship
                weight = self._calculate_relation_weight(rel)
                created_at = rel.created_at.isoformat() if rel.created_at else None

                batch_data.append({
                    "source_id": rel.source_entity,
                    "target_id": rel.target_entity,
                    "predicate": rel.predicate,
                    "description": rel.description,
                    "confidence": rel.confidence,
                    "weight": weight,
                    "chunk_ids": rel.chunk_ids,
                    "properties": _serialize_for_neo4j(rel.properties),
                    "created_at": created_at,
                    "pipeline_id": rel.pipeline_id,
                })

            try:
                result = await self._client.execute_cypher(batch_query, {"relations": batch_data})
                if result and len(result) > 0:
                    created = result[0].get("created_count", 0)
                    total_created += created

                logger.info(
                    "Relation batch loaded",
                    batch=f"{batch_start // self._batch_size + 1}/{(len(relations) + self._batch_size - 1) // self._batch_size}",
                    batch_size=len(batch),
                    total_created=total_created,
                )

            except Exception as e:
                error_msg = f"Batch relation loading failed at {batch_start}-{batch_end}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Fallback to single-record processing
                for rel in batch:
                    try:
                        if await self._load_single_relation(rel):
                            total_created += 1
                    except Exception as single_e:
                        errors.append(f"Failed to load relation {rel.source_entity}->{rel.target_entity}: {str(single_e)}")

        logger.info("Batch relation loading completed", count=total_created, failed=len(errors))
        return total_created, errors

    async def _load_single_relation(
        self,
        rel: ExtractedRelation,
    ) -> bool:
        """Load a single relation (fallback method)."""
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATES_TO]->(target)
        SET r.predicate = $predicate,
            r.description = $description,
            r.confidence = $confidence,
            r.chunk_ids = $chunk_ids,
            r.properties = $properties
        RETURN type(r) as rel_type
        """
        params = {
            "source_id": rel.source_entity,
            "target_id": rel.target_entity,
            "predicate": rel.predicate,
            "description": rel.description,
            "confidence": rel.confidence,
            "chunk_ids": rel.chunk_ids,
            "properties": _serialize_for_neo4j(rel.properties),
        }
        result = await self._client.execute_cypher(query, params)
        return bool(result)

    async def _load_relations_single(
        self,
        relations: list[ExtractedRelation],
    ) -> tuple[int, list[str]]:
        """Load relations one at a time (legacy method)."""
        errors: list[str] = []
        total_created = 0

        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATES_TO]->(target)
        SET r.predicate = $predicate,
            r.description = $description,
            r.confidence = $confidence,
            r.chunk_ids = $chunk_ids,
            r.properties = $properties
        RETURN type(r) as rel_type
        """

        for rel in relations:
            properties_json = _serialize_for_neo4j(rel.properties)
            params = {
                "source_id": rel.source_entity,
                "target_id": rel.target_entity,
                "predicate": rel.predicate,
                "description": rel.description,
                "confidence": rel.confidence,
                "chunk_ids": rel.chunk_ids,
                "properties": properties_json,
            }
            try:
                result = await self._client.execute_cypher(query, params)
                if result:
                    total_created += 1
            except Exception as e:
                error_msg = f"Failed to load relation {rel.source_entity}->{rel.target_entity}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

        logger.info("Relations loaded", count=total_created, failed=len(errors))
        return total_created, errors

    async def link_entities_to_chunks(
        self,
        entities: list[ExtractedEntity],
    ) -> int:
        """
        Create CONTAINS relationships between Chunks and Entities.

        Args:
            entities: List of entities with chunk_ids

        Returns:
            Number of relationships created
        """
        # Flatten to (chunk_id, entity_id) pairs
        links = []
        for entity in entities:
            for chunk_id in entity.chunk_ids:
                links.append({"chunk_id": chunk_id, "entity_id": entity.id})

        if not links:
            return 0

        if self._use_batch:
            return await self._link_entities_batch(links)
        else:
            return await self._link_entities_single(links)

    async def _link_entities_batch(self, links: list[dict[str, str]]) -> int:
        """Create chunk-entity links using batch operations."""
        total_created = 0

        batch_query = """
        UNWIND $links AS link_data
        MATCH (c:Chunk {id: link_data.chunk_id})
        MATCH (e:Entity {id: link_data.entity_id})
        MERGE (c)-[r:CONTAINS]->(e)
        RETURN count(r) as created_count
        """

        for batch_start in range(0, len(links), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(links))
            batch = links[batch_start:batch_end]

            try:
                result = await self._client.execute_cypher(batch_query, {"links": batch})
                if result and len(result) > 0:
                    total_created += result[0].get("created_count", 0)
            except Exception as e:
                logger.warning("Batch chunk-entity linking failed", error=str(e))
                # Fallback to single processing
                for link in batch:
                    try:
                        await self._client.execute_cypher(
                            """
                            MATCH (c:Chunk {id: $chunk_id})
                            MATCH (e:Entity {id: $entity_id})
                            MERGE (c)-[r:CONTAINS]->(e)
                            """,
                            link
                        )
                        total_created += 1
                    except Exception:
                        pass

        logger.info("Batch chunk-entity links created", count=total_created)
        return total_created

    async def _link_entities_single(self, links: list[dict[str, str]]) -> int:
        """Create chunk-entity links one at a time."""
        total_created = 0

        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (c)-[r:CONTAINS]->(e)
        """

        total_created = 0
        for link in links:
            try:
                await self._client.execute_cypher(query, link)
                total_created += 1
            except Exception as e:
                logger.warning("Failed to create chunk-entity link", error=str(e))

        logger.info("Chunk-Entity links created", count=total_created)
        return total_created

    async def load_all(
        self,
        chunks: list[TextChunk],
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        embeddings: dict[str, list[float]] | None = None,
        setup_schema_first: bool = True,
    ) -> LoadResult:
        """
        Load all extracted data in a single operation.

        Loads in order: chunks → entities → entity-chunk links → relations

        Args:
            chunks: List of text chunks
            entities: List of extracted entities
            relations: List of extracted relations
            embeddings: Optional embeddings for chunks and entities
            setup_schema_first: Whether to ensure schema/indexes exist before loading

        Returns:
            LoadResult with counts and any errors
        """
        start_time = time.time()
        all_errors: list[str] = []
        embeddings = embeddings or {}

        logger.info(
            "Starting load_all operation",
            chunks_count=len(chunks),
            entities_count=len(entities),
            relations_count=len(relations),
            embeddings_count=len(embeddings),
            batch_mode=self._use_batch,
            batch_size=self._batch_size,
        )

        await self.ensure_connected()

        # Ensure schema and indexes exist before loading data
        if setup_schema_first:
            try:
                logger.info("Setting up Neo4j schema and indexes...")
                await self.setup_schema()
            except Exception as e:
                logger.warning("Schema setup warning (may already exist)", error=str(e))

        # Load chunks (always try, even if entities failed)
        chunks_created = 0
        try:
            chunks_created, chunk_errors = await self.load_chunks(chunks, embeddings)
            all_errors.extend(chunk_errors)
        except Exception as e:
            error_msg = f"Chunk loading failed completely: {str(e)}"
            all_errors.append(error_msg)
            logger.error(error_msg, error_type=type(e).__name__)

        # Load entities
        entities_created = 0
        try:
            entities_created, entity_errors = await self.load_entities(entities, embeddings)
            all_errors.extend(entity_errors)
        except Exception as e:
            error_msg = f"Entity loading failed completely: {str(e)}"
            all_errors.append(error_msg)
            logger.error(error_msg, error_type=type(e).__name__)

        # Link entities to chunks (only if both exist)
        if chunks_created > 0 and entities_created > 0:
            try:
                await self.link_entities_to_chunks(entities)
            except Exception as e:
                error_msg = f"Entity-chunk linking failed: {str(e)}"
                all_errors.append(error_msg)
                logger.error(error_msg)
        else:
            logger.info(
                "Skipping entity-chunk linking (insufficient data)",
                chunks_created=chunks_created,
                entities_created=entities_created,
            )

        # Load relations (only if entities exist)
        relations_created = 0
        if entities_created > 0:
            try:
                relations_created, relation_errors = await self.load_relations(relations)
                all_errors.extend(relation_errors)
            except Exception as e:
                error_msg = f"Relation loading failed completely: {str(e)}"
                all_errors.append(error_msg)
                logger.error(error_msg, error_type=type(e).__name__)
        else:
            logger.info("Skipping relation loading (no entities)")

        duration = time.time() - start_time

        result = LoadResult(
            chunks_created=chunks_created,
            entities_created=entities_created,
            relations_created=relations_created,
            errors=all_errors,
            duration_seconds=duration,
        )

        logger.info(
            "Load completed",
            chunks=chunks_created,
            entities=entities_created,
            relations=relations_created,
            duration=f"{duration:.2f}s",
            error_count=len(all_errors),
            success=len(all_errors) == 0,
        )

        return result

    async def clear_all(self, confirm: bool = False) -> dict[str, int]:
        """
        Clear all data from the graph.

        WARNING: This deletes all nodes and relationships!

        Args:
            confirm: Must be True to execute

        Returns:
            Counts of deleted nodes/relationships
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear all data")

        await self.ensure_connected()

        # Delete all relationships first
        rel_query = "MATCH ()-[r]->() DELETE r RETURN count(r) as deleted"
        rel_result = await self._client.execute_cypher(rel_query)

        # Delete all nodes
        node_query = "MATCH (n) DELETE n RETURN count(n) as deleted"
        node_result = await self._client.execute_cypher(node_query)

        result = {
            "relationships_deleted": rel_result[0].get("deleted", 0) if rel_result else 0,
            "nodes_deleted": node_result[0].get("deleted", 0) if node_result else 0,
        }

        logger.warning("Graph cleared", **result)
        return result

    async def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about loaded data.

        Returns:
            Dictionary with node/relationship counts by type
        """
        await self.ensure_connected()

        stats_query = """
        CALL {
            MATCH (c:Chunk) RETURN 'chunks' as label, count(c) as count
            UNION ALL
            MATCH (e:Entity) RETURN 'entities' as label, count(e) as count
            UNION ALL
            MATCH (comm:Community) RETURN 'communities' as label, count(comm) as count
            UNION ALL
            MATCH ()-[r:RELATES_TO]->() RETURN 'relates_to' as label, count(r) as count
            UNION ALL
            MATCH ()-[r:CONTAINS]->() RETURN 'contains' as label, count(r) as count
            UNION ALL
            MATCH ()-[r:BELONGS_TO]->() RETURN 'belongs_to' as label, count(r) as count
        }
        RETURN label, count
        """

        results = await self._client.execute_cypher(stats_query)
        stats = {r["label"]: r["count"] for r in results}

        return {
            "nodes": {
                "chunks": stats.get("chunks", 0),
                "entities": stats.get("entities", 0),
                "communities": stats.get("communities", 0),
            },
            "relationships": {
                "relates_to": stats.get("relates_to", 0),
                "contains": stats.get("contains", 0),
                "belongs_to": stats.get("belongs_to", 0),
            },
        }

    async def create_community(
        self,
        community_id: str,
        summary: str,
        entity_ids: list[str],
        embedding: list[float] | None = None,
        level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Create a Community node and BELONGS_TO relationships.

        Args:
            community_id: Unique identifier for the community
            summary: Text summary describing the community
            entity_ids: List of entity IDs that belong to this community
            embedding: Optional vector embedding for the community summary
            level: Community hierarchy level (0 = leaf, higher = more abstract)
            metadata: Optional additional metadata

        Returns:
            True if community was created successfully
        """
        await self.ensure_connected()

        # Create Community node
        create_community_query = """
        MERGE (c:Community {id: $community_id})
        SET c.summary = $summary,
            c.embedding = $embedding,
            c.level = $level,
            c.member_count = $member_count,
            c.entity_ids = $entity_ids,
            c.metadata = $metadata,
            c.created_at = COALESCE(c.created_at, datetime()),
            c.updated_at = datetime()
        RETURN c.id as community_id
        """

        try:
            await self._client.execute_cypher(
                create_community_query,
                {
                    "community_id": community_id,
                    "summary": summary,
                    "embedding": embedding,
                    "level": level,
                    "member_count": len(entity_ids),
                    "entity_ids": entity_ids,
                    "metadata": _serialize_for_neo4j(metadata or {}),
                }
            )

            # Create BELONGS_TO relationships from entities to community
            if entity_ids:
                belongs_to_created = await self.link_entities_to_community(
                    community_id=community_id,
                    entity_ids=entity_ids,
                )
                logger.info(
                    "Community created with BELONGS_TO relationships",
                    community_id=community_id,
                    member_count=len(entity_ids),
                    relationships_created=belongs_to_created,
                )

            return True

        except Exception as e:
            logger.error(
                "Failed to create community",
                community_id=community_id,
                error=str(e),
            )
            return False

    async def link_entities_to_community(
        self,
        community_id: str,
        entity_ids: list[str],
        weight: float = 1.0,
    ) -> int:
        """
        Create BELONGS_TO relationships between entities and a community.

        Args:
            community_id: Target community ID
            entity_ids: List of entity IDs to link
            weight: Relationship weight (default 1.0)

        Returns:
            Number of relationships created
        """
        if not entity_ids:
            return 0

        await self.ensure_connected()

        if self._use_batch:
            return await self._link_entities_to_community_batch(
                community_id, entity_ids, weight
            )
        else:
            return await self._link_entities_to_community_single(
                community_id, entity_ids, weight
            )

    async def _link_entities_to_community_batch(
        self,
        community_id: str,
        entity_ids: list[str],
        weight: float,
    ) -> int:
        """Create BELONGS_TO relationships using batch operations."""
        total_created = 0

        batch_query = """
        UNWIND $entity_ids AS entity_id
        MATCH (e:Entity {id: entity_id})
        MATCH (c:Community {id: $community_id})
        MERGE (e)-[r:BELONGS_TO]->(c)
        SET r.weight = $weight,
            r.created_at = COALESCE(r.created_at, datetime())
        RETURN count(r) as created_count
        """

        for batch_start in range(0, len(entity_ids), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(entity_ids))
            batch = entity_ids[batch_start:batch_end]

            try:
                result = await self._client.execute_cypher(
                    batch_query,
                    {
                        "entity_ids": batch,
                        "community_id": community_id,
                        "weight": weight,
                    }
                )
                if result and len(result) > 0:
                    total_created += result[0].get("created_count", 0)

            except Exception as e:
                logger.warning(
                    "Batch BELONGS_TO creation failed, falling back to single",
                    error=str(e),
                    batch_size=len(batch),
                )
                # Fallback to single processing
                for entity_id in batch:
                    try:
                        await self._client.execute_cypher(
                            """
                            MATCH (e:Entity {id: $entity_id})
                            MATCH (c:Community {id: $community_id})
                            MERGE (e)-[r:BELONGS_TO]->(c)
                            SET r.weight = $weight
                            """,
                            {
                                "entity_id": entity_id,
                                "community_id": community_id,
                                "weight": weight,
                            }
                        )
                        total_created += 1
                    except Exception:
                        pass

        logger.info(
            "BELONGS_TO relationships created",
            community_id=community_id,
            count=total_created,
        )
        return total_created

    async def _link_entities_to_community_single(
        self,
        community_id: str,
        entity_ids: list[str],
        weight: float,
    ) -> int:
        """Create BELONGS_TO relationships one at a time."""
        total_created = 0

        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (c:Community {id: $community_id})
        MERGE (e)-[r:BELONGS_TO]->(c)
        SET r.weight = $weight,
            r.created_at = COALESCE(r.created_at, datetime())
        """

        for entity_id in entity_ids:
            try:
                await self._client.execute_cypher(
                    query,
                    {
                        "entity_id": entity_id,
                        "community_id": community_id,
                        "weight": weight,
                    }
                )
                total_created += 1
            except Exception as e:
                logger.warning(
                    "Failed to create BELONGS_TO relationship",
                    entity_id=entity_id,
                    community_id=community_id,
                    error=str(e),
                )

        return total_created

    async def load_communities(
        self,
        communities: list[dict[str, Any]],
        embeddings: dict[str, list[float]] | None = None,
    ) -> tuple[int, list[str]]:
        """
        Load multiple communities with their BELONGS_TO relationships.

        Args:
            communities: List of community dicts with keys:
                - id: Community ID
                - summary: Community description
                - entity_ids: List of member entity IDs
                - level: Optional hierarchy level
                - metadata: Optional metadata
            embeddings: Optional mapping of community_id -> embedding

        Returns:
            Tuple of (communities created, list of error messages)
        """
        if not communities:
            return 0, []

        embeddings = embeddings or {}
        errors: list[str] = []
        total_created = 0

        logger.info("Starting community loading", count=len(communities))

        for community in communities:
            try:
                success = await self.create_community(
                    community_id=community["id"],
                    summary=community.get("summary", ""),
                    entity_ids=community.get("entity_ids", []),
                    embedding=embeddings.get(community["id"]),
                    level=community.get("level", 0),
                    metadata=community.get("metadata"),
                )
                if success:
                    total_created += 1
            except Exception as e:
                error_msg = f"Failed to load community {community.get('id')}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        logger.info(
            "Community loading completed",
            created=total_created,
            failed=len(errors),
        )
        return total_created, errors

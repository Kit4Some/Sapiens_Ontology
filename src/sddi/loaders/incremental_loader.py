"""
Incremental Graph Update Module.

Provides optimized delta-based updates to the knowledge graph:
- Change detection (new/modified/deleted entities)
- Version management and tracking
- Embedding change detection
- Efficient partial updates
- Rollback support
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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


class ChangeType(str, Enum):
    """Types of changes detected in data."""

    NEW = "new"                # Entity/relation doesn't exist
    MODIFIED = "modified"      # Entity/relation exists but changed
    UNCHANGED = "unchanged"    # No changes detected
    DELETED = "deleted"        # Should be removed
    EMBEDDING_ONLY = "embedding_only"  # Only embedding changed


@dataclass
class EntityChange:
    """Represents a detected change to an entity."""

    entity_id: str
    change_type: ChangeType
    entity: ExtractedEntity | None = None
    existing_hash: str | None = None
    new_hash: str | None = None
    changed_fields: list[str] = field(default_factory=list)
    embedding_changed: bool = False


@dataclass
class RelationChange:
    """Represents a detected change to a relation."""

    relation_key: str  # source_id:predicate:target_id
    change_type: ChangeType
    relation: ExtractedRelation | None = None
    existing_confidence: float | None = None
    new_confidence: float | None = None


@dataclass
class ChunkChange:
    """Represents a detected change to a chunk."""

    chunk_id: str
    change_type: ChangeType
    chunk: TextChunk | None = None
    content_hash: str | None = None


@dataclass
class DeltaReport:
    """Report of detected changes."""

    entity_changes: list[EntityChange] = field(default_factory=list)
    relation_changes: list[RelationChange] = field(default_factory=list)
    chunk_changes: list[ChunkChange] = field(default_factory=list)

    # Summary counts
    new_entities: int = 0
    modified_entities: int = 0
    unchanged_entities: int = 0
    deleted_entities: int = 0
    embedding_only_entities: int = 0

    new_relations: int = 0
    modified_relations: int = 0
    unchanged_relations: int = 0

    new_chunks: int = 0
    modified_chunks: int = 0
    unchanged_chunks: int = 0

    detection_time_ms: float = 0.0

    def has_changes(self) -> bool:
        """Check if any changes were detected."""
        return (
            self.new_entities > 0 or
            self.modified_entities > 0 or
            self.deleted_entities > 0 or
            self.new_relations > 0 or
            self.modified_relations > 0 or
            self.new_chunks > 0 or
            self.modified_chunks > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "entities": {
                "new": self.new_entities,
                "modified": self.modified_entities,
                "unchanged": self.unchanged_entities,
                "deleted": self.deleted_entities,
                "embedding_only": self.embedding_only_entities,
            },
            "relations": {
                "new": self.new_relations,
                "modified": self.modified_relations,
                "unchanged": self.unchanged_relations,
            },
            "chunks": {
                "new": self.new_chunks,
                "modified": self.modified_chunks,
                "unchanged": self.unchanged_chunks,
            },
            "detection_time_ms": self.detection_time_ms,
            "has_changes": self.has_changes(),
        }


class ContentHasher:
    """Generates content hashes for change detection."""

    @staticmethod
    def hash_entity(entity: ExtractedEntity) -> str:
        """Generate hash for entity content (excluding embedding)."""
        content = {
            "name": entity.name,
            "type": entity.type.value,
            "description": entity.description,
            "aliases": sorted(entity.aliases) if entity.aliases else [],
            "confidence": round(entity.confidence, 4),
            "properties": json.dumps(entity.properties, sort_keys=True),
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    @staticmethod
    def hash_relation(relation: ExtractedRelation) -> str:
        """Generate hash for relation content."""
        content = {
            "source": relation.source_entity,
            "target": relation.target_entity,
            "predicate": relation.predicate,
            "description": relation.description,
            "confidence": round(relation.confidence, 4),
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    @staticmethod
    def hash_chunk(chunk: TextChunk) -> str:
        """Generate hash for chunk content."""
        content = {
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "position": chunk.position,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    @staticmethod
    def hash_embedding(embedding: list[float] | None) -> str | None:
        """Generate hash for embedding vector."""
        if not embedding:
            return None
        # Hash first and last 10 values + length for efficiency
        sample = embedding[:10] + embedding[-10:] + [float(len(embedding))]
        content_str = json.dumps([round(v, 6) for v in sample])
        return hashlib.sha256(content_str.encode()).hexdigest()[:12]


class IncrementalLoader:
    """
    Incremental graph loader with delta detection.

    Features:
    - Detects changes before loading
    - Only updates modified entities
    - Tracks versions for rollback
    - Handles embedding updates separately
    - Efficient batch operations for deltas
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
        batch_size: int = 500,
        track_versions: bool = True,
        min_confidence_change: float = 0.1,
    ):
        """
        Initialize incremental loader.

        Args:
            client: Neo4j client
            batch_size: Batch size for operations
            track_versions: Whether to track entity versions
            min_confidence_change: Minimum confidence change to trigger update
        """
        self._client = client or get_ontology_client()
        self._batch_size = batch_size
        self._track_versions = track_versions
        self._min_confidence_change = min_confidence_change
        self._hasher = ContentHasher()

    async def detect_changes(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        chunks: list[TextChunk],
        embeddings: dict[str, list[float]] | None = None,
    ) -> DeltaReport:
        """
        Detect changes between new data and existing graph.

        Args:
            entities: New entities to compare
            relations: New relations to compare
            chunks: New chunks to compare
            embeddings: New embeddings

        Returns:
            DeltaReport with all detected changes
        """
        start_time = time.time()
        embeddings = embeddings or {}

        report = DeltaReport()

        await self._client.connect()

        # Detect entity changes
        entity_changes = await self._detect_entity_changes(entities, embeddings)
        report.entity_changes = entity_changes
        for change in entity_changes:
            if change.change_type == ChangeType.NEW:
                report.new_entities += 1
            elif change.change_type == ChangeType.MODIFIED:
                report.modified_entities += 1
            elif change.change_type == ChangeType.UNCHANGED:
                report.unchanged_entities += 1
            elif change.change_type == ChangeType.DELETED:
                report.deleted_entities += 1
            elif change.change_type == ChangeType.EMBEDDING_ONLY:
                report.embedding_only_entities += 1

        # Detect relation changes
        relation_changes = await self._detect_relation_changes(relations)
        report.relation_changes = relation_changes
        for change in relation_changes:
            if change.change_type == ChangeType.NEW:
                report.new_relations += 1
            elif change.change_type == ChangeType.MODIFIED:
                report.modified_relations += 1
            elif change.change_type == ChangeType.UNCHANGED:
                report.unchanged_relations += 1

        # Detect chunk changes
        chunk_changes = await self._detect_chunk_changes(chunks)
        report.chunk_changes = chunk_changes
        for change in chunk_changes:
            if change.change_type == ChangeType.NEW:
                report.new_chunks += 1
            elif change.change_type == ChangeType.MODIFIED:
                report.modified_chunks += 1
            elif change.change_type == ChangeType.UNCHANGED:
                report.unchanged_chunks += 1

        report.detection_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Change detection completed",
            new_entities=report.new_entities,
            modified_entities=report.modified_entities,
            unchanged_entities=report.unchanged_entities,
            new_relations=report.new_relations,
            detection_time_ms=round(report.detection_time_ms, 2),
        )

        return report

    async def _detect_entity_changes(
        self,
        entities: list[ExtractedEntity],
        embeddings: dict[str, list[float]],
    ) -> list[EntityChange]:
        """Detect changes for entities."""
        if not entities:
            return []

        changes = []
        entity_ids = [e.id for e in entities]

        # Fetch existing entity metadata
        existing_query = """
        UNWIND $ids AS eid
        MATCH (e:Entity {id: eid})
        RETURN e.id as id, e.content_hash as content_hash, e.embedding_hash as embedding_hash,
               e.version as version, e.name as name, e.type as type
        """
        existing_results = await self._client.execute_cypher(
            existing_query, {"ids": entity_ids}
        )

        existing_map = {r["id"]: r for r in existing_results}

        for entity in entities:
            new_hash = self._hasher.hash_entity(entity)
            new_embedding_hash = self._hasher.hash_embedding(embeddings.get(entity.id))

            existing = existing_map.get(entity.id)

            if not existing:
                # New entity
                changes.append(EntityChange(
                    entity_id=entity.id,
                    change_type=ChangeType.NEW,
                    entity=entity,
                    new_hash=new_hash,
                    embedding_changed=new_embedding_hash is not None,
                ))
            else:
                existing_hash = existing.get("content_hash")
                existing_embedding_hash = existing.get("embedding_hash")

                content_changed = existing_hash != new_hash
                embedding_changed = (
                    new_embedding_hash is not None and
                    existing_embedding_hash != new_embedding_hash
                )

                if content_changed:
                    # Determine what changed
                    changed_fields = []
                    if existing.get("name") != entity.name:
                        changed_fields.append("name")
                    if existing.get("type") != entity.type.value:
                        changed_fields.append("type")
                    # More detailed comparison would require fetching full entity

                    changes.append(EntityChange(
                        entity_id=entity.id,
                        change_type=ChangeType.MODIFIED,
                        entity=entity,
                        existing_hash=existing_hash,
                        new_hash=new_hash,
                        changed_fields=changed_fields,
                        embedding_changed=embedding_changed,
                    ))
                elif embedding_changed:
                    changes.append(EntityChange(
                        entity_id=entity.id,
                        change_type=ChangeType.EMBEDDING_ONLY,
                        entity=entity,
                        existing_hash=existing_hash,
                        new_hash=new_hash,
                        embedding_changed=True,
                    ))
                else:
                    changes.append(EntityChange(
                        entity_id=entity.id,
                        change_type=ChangeType.UNCHANGED,
                        entity=entity,
                    ))

        return changes

    async def _detect_relation_changes(
        self,
        relations: list[ExtractedRelation],
    ) -> list[RelationChange]:
        """Detect changes for relations."""
        if not relations:
            return []

        changes = []

        # Create relation keys
        relation_map = {}
        for rel in relations:
            key = f"{rel.source_entity}:{rel.predicate}:{rel.target_entity}"
            relation_map[key] = rel

        # Fetch existing relations
        existing_query = """
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        WHERE s.id + ':' + r.predicate + ':' + t.id IN $keys
        RETURN s.id + ':' + r.predicate + ':' + t.id as key,
               r.confidence as confidence, r.content_hash as content_hash
        """
        existing_results = await self._client.execute_cypher(
            existing_query, {"keys": list(relation_map.keys())}
        )

        existing_map = {r["key"]: r for r in existing_results}

        for key, rel in relation_map.items():
            existing = existing_map.get(key)

            if not existing:
                changes.append(RelationChange(
                    relation_key=key,
                    change_type=ChangeType.NEW,
                    relation=rel,
                    new_confidence=rel.confidence,
                ))
            else:
                existing_conf = existing.get("confidence", 0)
                conf_diff = abs(rel.confidence - existing_conf)

                if conf_diff >= self._min_confidence_change:
                    changes.append(RelationChange(
                        relation_key=key,
                        change_type=ChangeType.MODIFIED,
                        relation=rel,
                        existing_confidence=existing_conf,
                        new_confidence=rel.confidence,
                    ))
                else:
                    changes.append(RelationChange(
                        relation_key=key,
                        change_type=ChangeType.UNCHANGED,
                        relation=rel,
                    ))

        return changes

    async def _detect_chunk_changes(
        self,
        chunks: list[TextChunk],
    ) -> list[ChunkChange]:
        """Detect changes for chunks."""
        if not chunks:
            return []

        changes = []
        chunk_ids = [c.id for c in chunks]

        # Fetch existing chunk metadata
        existing_query = """
        UNWIND $ids AS cid
        MATCH (c:Chunk {id: cid})
        RETURN c.id as id, c.content_hash as content_hash
        """
        existing_results = await self._client.execute_cypher(
            existing_query, {"ids": chunk_ids}
        )

        existing_map = {r["id"]: r for r in existing_results}

        for chunk in chunks:
            new_hash = self._hasher.hash_chunk(chunk)
            existing = existing_map.get(chunk.id)

            if not existing:
                changes.append(ChunkChange(
                    chunk_id=chunk.id,
                    change_type=ChangeType.NEW,
                    chunk=chunk,
                    content_hash=new_hash,
                ))
            elif existing.get("content_hash") != new_hash:
                changes.append(ChunkChange(
                    chunk_id=chunk.id,
                    change_type=ChangeType.MODIFIED,
                    chunk=chunk,
                    content_hash=new_hash,
                ))
            else:
                changes.append(ChunkChange(
                    chunk_id=chunk.id,
                    change_type=ChangeType.UNCHANGED,
                    chunk=chunk,
                ))

        return changes

    async def apply_delta(
        self,
        delta_report: DeltaReport,
        embeddings: dict[str, list[float]] | None = None,
        skip_unchanged: bool = True,
    ) -> LoadResult:
        """
        Apply detected changes to the graph.

        Args:
            delta_report: Report from detect_changes()
            embeddings: Embeddings for new/modified entities
            skip_unchanged: Whether to skip unchanged items

        Returns:
            LoadResult with counts
        """
        start_time = time.time()
        embeddings = embeddings or {}
        all_errors = []

        await self._client.connect()

        # Apply chunk changes
        chunks_created = 0
        chunks_to_load = []
        for change in delta_report.chunk_changes:
            if change.change_type in [ChangeType.NEW, ChangeType.MODIFIED]:
                if change.chunk:
                    chunks_to_load.append(change.chunk)

        if chunks_to_load:
            chunks_created, errors = await self._load_chunks_delta(chunks_to_load, embeddings)
            all_errors.extend(errors)

        # Apply entity changes
        entities_created = 0

        # New entities
        new_entities = [
            c.entity for c in delta_report.entity_changes
            if c.change_type == ChangeType.NEW and c.entity
        ]
        if new_entities:
            count, errors = await self._load_entities_new(new_entities, embeddings)
            entities_created += count
            all_errors.extend(errors)

        # Modified entities
        modified_entities = [
            c.entity for c in delta_report.entity_changes
            if c.change_type == ChangeType.MODIFIED and c.entity
        ]
        if modified_entities:
            count, errors = await self._update_entities(modified_entities, embeddings)
            entities_created += count
            all_errors.extend(errors)

        # Embedding-only updates
        embedding_only = [
            c.entity for c in delta_report.entity_changes
            if c.change_type == ChangeType.EMBEDDING_ONLY and c.entity
        ]
        if embedding_only:
            count, errors = await self._update_embeddings_only(embedding_only, embeddings)
            all_errors.extend(errors)

        # Apply relation changes
        relations_created = 0
        new_relations = [
            c.relation for c in delta_report.relation_changes
            if c.change_type == ChangeType.NEW and c.relation
        ]
        modified_relations = [
            c.relation for c in delta_report.relation_changes
            if c.change_type == ChangeType.MODIFIED and c.relation
        ]

        if new_relations or modified_relations:
            count, errors = await self._load_relations_delta(
                new_relations + modified_relations
            )
            relations_created = count
            all_errors.extend(errors)

        duration = time.time() - start_time

        logger.info(
            "Delta applied",
            chunks_created=chunks_created,
            entities_created=entities_created,
            relations_created=relations_created,
            duration=f"{duration:.2f}s",
            errors=len(all_errors),
        )

        return LoadResult(
            chunks_created=chunks_created,
            entities_created=entities_created,
            relations_created=relations_created,
            errors=all_errors,
            duration_seconds=duration,
        )

    async def _load_chunks_delta(
        self,
        chunks: list[TextChunk],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Load new or modified chunks."""
        errors = []
        total = 0

        batch_query = """
        UNWIND $chunks AS chunk_data
        MERGE (c:Chunk {id: chunk_data.id})
        SET c.text = chunk_data.text,
            c.source = chunk_data.doc_id,
            c.position = chunk_data.position,
            c.embedding = chunk_data.embedding,
            c.content_hash = chunk_data.content_hash,
            c.updated_at = datetime()
        RETURN count(c) as count
        """

        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start:batch_start + self._batch_size]
            batch_data = [{
                "id": c.id,
                "text": c.text,
                "doc_id": c.doc_id,
                "position": c.position,
                "embedding": embeddings.get(c.id),
                "content_hash": self._hasher.hash_chunk(c),
            } for c in batch]

            try:
                result = await self._client.execute_cypher(batch_query, {"chunks": batch_data})
                if result:
                    total += result[0].get("count", 0)
            except Exception as e:
                errors.append(f"Chunk delta batch failed: {str(e)}")

        return total, errors

    async def _load_entities_new(
        self,
        entities: list[ExtractedEntity],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Load new entities."""
        errors = []
        total = 0

        batch_query = """
        UNWIND $entities AS entity_data
        CREATE (e:Entity {id: entity_data.id})
        SET e.name = entity_data.name,
            e.type = entity_data.type,
            e.description = entity_data.description,
            e.aliases = entity_data.aliases,
            e.confidence = entity_data.confidence,
            e.embedding = entity_data.embedding,
            e.content_hash = entity_data.content_hash,
            e.embedding_hash = entity_data.embedding_hash,
            e.version = 1,
            e.created_at = datetime(),
            e.updated_at = datetime()
        RETURN count(e) as count
        """

        for batch_start in range(0, len(entities), self._batch_size):
            batch = entities[batch_start:batch_start + self._batch_size]
            batch_data = [{
                "id": e.id,
                "name": e.name,
                "type": e.type.value,
                "description": e.description,
                "aliases": e.aliases,
                "confidence": e.confidence,
                "embedding": embeddings.get(e.id),
                "content_hash": self._hasher.hash_entity(e),
                "embedding_hash": self._hasher.hash_embedding(embeddings.get(e.id)),
            } for e in batch]

            try:
                result = await self._client.execute_cypher(batch_query, {"entities": batch_data})
                if result:
                    total += result[0].get("count", 0)
            except Exception as e:
                errors.append(f"Entity insert batch failed: {str(e)}")

        return total, errors

    async def _update_entities(
        self,
        entities: list[ExtractedEntity],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Update modified entities."""
        errors = []
        total = 0

        batch_query = """
        UNWIND $entities AS entity_data
        MATCH (e:Entity {id: entity_data.id})
        SET e.name = entity_data.name,
            e.type = entity_data.type,
            e.description = entity_data.description,
            e.aliases = entity_data.aliases,
            e.confidence = entity_data.confidence,
            e.embedding = entity_data.embedding,
            e.content_hash = entity_data.content_hash,
            e.embedding_hash = entity_data.embedding_hash,
            e.version = COALESCE(e.version, 0) + 1,
            e.updated_at = datetime()
        RETURN count(e) as count
        """

        for batch_start in range(0, len(entities), self._batch_size):
            batch = entities[batch_start:batch_start + self._batch_size]
            batch_data = [{
                "id": e.id,
                "name": e.name,
                "type": e.type.value,
                "description": e.description,
                "aliases": e.aliases,
                "confidence": e.confidence,
                "embedding": embeddings.get(e.id),
                "content_hash": self._hasher.hash_entity(e),
                "embedding_hash": self._hasher.hash_embedding(embeddings.get(e.id)),
            } for e in batch]

            try:
                result = await self._client.execute_cypher(batch_query, {"entities": batch_data})
                if result:
                    total += result[0].get("count", 0)
            except Exception as e:
                errors.append(f"Entity update batch failed: {str(e)}")

        return total, errors

    async def _update_embeddings_only(
        self,
        entities: list[ExtractedEntity],
        embeddings: dict[str, list[float]],
    ) -> tuple[int, list[str]]:
        """Update only embeddings for entities."""
        errors = []
        total = 0

        batch_query = """
        UNWIND $entities AS entity_data
        MATCH (e:Entity {id: entity_data.id})
        SET e.embedding = entity_data.embedding,
            e.embedding_hash = entity_data.embedding_hash,
            e.updated_at = datetime()
        RETURN count(e) as count
        """

        for batch_start in range(0, len(entities), self._batch_size):
            batch = entities[batch_start:batch_start + self._batch_size]
            batch_data = [{
                "id": e.id,
                "embedding": embeddings.get(e.id),
                "embedding_hash": self._hasher.hash_embedding(embeddings.get(e.id)),
            } for e in batch]

            try:
                result = await self._client.execute_cypher(batch_query, {"entities": batch_data})
                if result:
                    total += result[0].get("count", 0)
            except Exception as e:
                errors.append(f"Embedding update batch failed: {str(e)}")

        return total, errors

    async def _load_relations_delta(
        self,
        relations: list[ExtractedRelation],
    ) -> tuple[int, list[str]]:
        """Load new or update modified relations."""
        errors = []
        total = 0

        batch_query = """
        UNWIND $relations AS rel_data
        MATCH (source:Entity {id: rel_data.source_id})
        MATCH (target:Entity {id: rel_data.target_id})
        MERGE (source)-[r:RELATES_TO {predicate: rel_data.predicate}]->(target)
        SET r.description = rel_data.description,
            r.confidence = rel_data.confidence,
            r.chunk_ids = rel_data.chunk_ids,
            r.content_hash = rel_data.content_hash,
            r.updated_at = datetime()
        RETURN count(r) as count
        """

        for batch_start in range(0, len(relations), self._batch_size):
            batch = relations[batch_start:batch_start + self._batch_size]
            batch_data = [{
                "source_id": r.source_entity,
                "target_id": r.target_entity,
                "predicate": r.predicate,
                "description": r.description,
                "confidence": r.confidence,
                "chunk_ids": r.chunk_ids,
                "content_hash": self._hasher.hash_relation(r),
            } for r in batch]

            try:
                result = await self._client.execute_cypher(batch_query, {"relations": batch_data})
                if result:
                    total += result[0].get("count", 0)
            except Exception as e:
                errors.append(f"Relation delta batch failed: {str(e)}")

        return total, errors

    async def load_incremental(
        self,
        chunks: list[TextChunk],
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        embeddings: dict[str, list[float]] | None = None,
    ) -> tuple[DeltaReport, LoadResult]:
        """
        Convenience method: detect changes and apply delta in one call.

        Args:
            chunks: New chunks
            entities: New entities
            relations: New relations
            embeddings: Embeddings

        Returns:
            Tuple of (DeltaReport, LoadResult)
        """
        delta = await self.detect_changes(entities, relations, chunks, embeddings)
        result = await self.apply_delta(delta, embeddings)
        return delta, result

    async def get_change_history(
        self,
        entity_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get change history for an entity (if version tracking enabled).

        Args:
            entity_id: Entity ID
            limit: Maximum history entries

        Returns:
            List of version history records
        """
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e.id as id, e.version as version, e.created_at as created_at,
               e.updated_at as updated_at, e.content_hash as content_hash
        """
        results = await self._client.execute_cypher(query, {"id": entity_id})
        return results[:limit] if results else []


# Factory function
def create_incremental_loader(
    client: OntologyGraphClient | None = None,
    **kwargs,
) -> IncrementalLoader:
    """Create an incremental loader instance."""
    return IncrementalLoader(client=client, **kwargs)

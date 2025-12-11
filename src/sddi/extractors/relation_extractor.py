"""
Relation Extractor Module.

LLM-based relation extraction for knowledge graph construction.
Extracts relationships between entities from text.

Enhanced with:
- Extraction quality metrics
- Entity existence validation
- Retry logic with backoff
- Proper error propagation
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.sddi.state import (
    ExtractedEntity,
    ExtractedRelation,
    TextChunk,
    Triplet,
)

logger = structlog.get_logger(__name__)


class RelationExtractionError(Exception):
    """Raised when relation extraction fails critically."""

    def __init__(self, message: str, error_type: str = "unknown", recoverable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable


@dataclass
class RelationExtractionMetrics:
    """Metrics for tracking relation extraction quality."""

    chunks_processed: int = 0
    chunks_with_relations: int = 0
    total_relations: int = 0
    relations_filtered_by_confidence: int = 0
    relations_with_invalid_entities: int = 0
    extraction_errors: int = 0
    retry_count: int = 0
    empty_extractions: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    @property
    def extraction_rate(self) -> float:
        """Percentage of chunks that yielded relations."""
        if self.chunks_processed == 0:
            return 0.0
        return self.chunks_with_relations / self.chunks_processed

    @property
    def avg_relations_per_chunk(self) -> float:
        """Average relations per chunk (only chunks with relations)."""
        if self.chunks_with_relations == 0:
            return 0.0
        return self.total_relations / self.chunks_with_relations

    @property
    def duration_seconds(self) -> float:
        """Extraction duration in seconds."""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_with_relations": self.chunks_with_relations,
            "total_relations": self.total_relations,
            "relations_filtered_by_confidence": self.relations_filtered_by_confidence,
            "relations_with_invalid_entities": self.relations_with_invalid_entities,
            "extraction_errors": self.extraction_errors,
            "retry_count": self.retry_count,
            "empty_extractions": self.empty_extractions,
            "extraction_rate": round(self.extraction_rate, 3),
            "avg_relations_per_chunk": round(self.avg_relations_per_chunk, 2),
            "duration_seconds": round(self.duration_seconds, 2),
        }


class RelationExtractionOutput(BaseModel):
    """Schema for LLM relation extraction output."""

    relations: list[dict[str, Any]] = Field(description="List of extracted relations")


RELATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert relation extraction system for knowledge graph construction.
Given a text and a list of entities found in it, extract all meaningful relationships between these entities.

## Known Entities
{entities}

## Guidelines
1. Only extract relations between the provided entities
2. Use clear, concise predicates (verbs or verb phrases)
3. Predicates should be in UPPER_SNAKE_CASE format
4. Each relation should be directional (source → target)
5. Assign confidence scores based on how explicit the relation is in text
6. Include a brief description explaining the relationship

## Common Predicate Types
- WORKS_FOR, EMPLOYED_BY: Employment relationships
- LOCATED_IN, BASED_IN: Geographic relationships
- FOUNDED, CREATED: Creation relationships
- OWNS, ACQUIRED: Ownership relationships
- PART_OF, BELONGS_TO: Membership/containment
- RELATED_TO: Generic association
- COLLABORATES_WITH: Partnership
- REPORTS_TO: Hierarchy
- PRODUCES, DEVELOPS: Production
- USES, UTILIZES: Usage relationships
- OCCURRED_ON, HAPPENED_AT: Temporal/spatial events

## Output Format
Return a JSON object with a "relations" array containing:
- source: Source entity name (must match an entity from the list)
- target: Target entity name (must match an entity from the list)
- predicate: Relationship type in UPPER_SNAKE_CASE
- description: Brief description of the relationship
- confidence: Score 0.0-1.0 based on explicitness in text

Only return valid JSON, no additional text.""",
        ),
        (
            "human",
            """Extract relations from this text, using only the entities provided:

---
{text}
---

Return relations as JSON:""",
        ),
    ]
)


class RelationExtractor:
    """
    LLM-based relation extractor for knowledge graph construction.

    Extracts relationships between previously identified entities.

    Enhanced with:
    - Extraction quality metrics
    - Entity existence validation
    - Retry logic with backoff
    - Configurable timeouts
    """

    def __init__(
        self,
        llm: BaseChatModel,
        min_confidence: float = 0.5,
        allowed_predicates: list[str] | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        validate_entities: bool = True,
        progress_callback: Callable[[str, float, str, dict[str, Any]], None] | None = None,
    ) -> None:
        """
        Initialize the relation extractor.

        Args:
            llm: LangChain chat model for extraction
            min_confidence: Minimum confidence threshold
            allowed_predicates: Optional whitelist of predicates
            max_retries: Maximum retry attempts for transient failures
            retry_delay: Base delay between retries (exponential backoff)
            validate_entities: Whether to validate entity references
            progress_callback: Optional callback for progress updates
        """
        self._llm = llm
        self._min_confidence = min_confidence
        self._allowed_predicates = allowed_predicates
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._validate_entities = validate_entities
        self._progress_callback = progress_callback
        self._parser = JsonOutputParser(pydantic_object=RelationExtractionOutput)
        self._chain = RELATION_EXTRACTION_PROMPT | self._llm | self._parser
        self._metrics = RelationExtractionMetrics()

    def get_metrics(self) -> RelationExtractionMetrics:
        """Get extraction metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset extraction metrics for a new run."""
        self._metrics = RelationExtractionMetrics()

    def _generate_relation_id(
        self,
        source: str,
        target: str,
        predicate: str,
    ) -> str:
        """Generate a deterministic relation ID."""
        key = f"{source.lower()}:{predicate}:{target.lower()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _format_entities_for_prompt(
        self,
        entities: list[ExtractedEntity],
    ) -> str:
        """Format entity list for prompt injection."""
        lines = []
        for entity in entities:
            aliases_str = f" (aliases: {', '.join(entity.aliases)})" if entity.aliases else ""
            lines.append(f"- {entity.name} [{entity.type.value}]{aliases_str}")
        return "\n".join(lines)

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to UPPER_SNAKE_CASE."""
        # Remove extra whitespace
        normalized = predicate.strip()
        # Replace spaces and hyphens with underscores
        normalized = normalized.replace(" ", "_").replace("-", "_")
        # Convert to uppercase
        normalized = normalized.upper()
        # Remove consecutive underscores
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized

    def _find_entity_by_name(
        self,
        name: str,
        entities: list[ExtractedEntity],
    ) -> ExtractedEntity | None:
        """Find entity by name or alias."""
        name_lower = name.lower().strip()
        for entity in entities:
            if entity.name.lower() == name_lower:
                return entity
            if any(alias.lower() == name_lower for alias in entity.aliases):
                return entity
        return None

    async def _call_llm_with_retry(
        self,
        text: str,
        entities_str: str,
        chunk_id: str,
    ) -> list[dict[str, Any]]:
        """Call LLM with retry logic for transient failures."""
        import asyncio

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                result = await self._chain.ainvoke(
                    {
                        "text": text,
                        "entities": entities_str,
                    }
                )
                return result.get("relations", [])

            except Exception as e:
                error_str = str(e).lower()

                # Non-recoverable errors
                if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                    raise RelationExtractionError(
                        f"API authentication failed: {e}",
                        error_type="auth_error",
                        recoverable=False,
                    ) from e

                # Rate limit - retry with longer delay
                if "rate limit" in error_str or "429" in error_str:
                    wait_time = self._retry_delay * (attempt + 1) * 2
                    logger.warning(
                        "Rate limited, waiting before retry",
                        chunk_id=chunk_id,
                        wait_seconds=wait_time,
                        attempt=attempt + 1,
                    )
                    self._metrics.retry_count += 1
                    await asyncio.sleep(wait_time)
                    last_error = e
                    continue

                # Timeout or other transient errors
                wait_time = self._retry_delay * (attempt + 1)
                self._metrics.retry_count += 1
                await asyncio.sleep(wait_time)
                last_error = e

        self._metrics.extraction_errors += 1
        raise RelationExtractionError(
            f"Relation extraction failed after {self._max_retries} attempts: {last_error}",
            error_type="extraction_failed",
            recoverable=True,
        )

    async def extract(
        self,
        chunk: TextChunk,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        """
        Extract relations from a text chunk given known entities.

        Args:
            chunk: Text chunk to process
            entities: List of entities found in this chunk

        Returns:
            List of extracted relations

        Raises:
            RelationExtractionError: If extraction fails critically
        """
        self._metrics.chunks_processed += 1

        if len(entities) < 2:
            logger.debug("Not enough entities for relation extraction", chunk_id=chunk.id)
            return []

        # Filter entities relevant to this chunk
        chunk_entities = [e for e in entities if chunk.id in e.chunk_ids]

        if len(chunk_entities) < 2:
            # Fall back to all entities if filtering too aggressive
            # But limit to a reasonable number to avoid overwhelming the LLM
            chunk_entities = entities[:20]
            logger.debug(
                "Falling back to all entities (limited)",
                chunk_id=chunk.id,
                entity_count=len(chunk_entities),
            )

        entities_str = self._format_entities_for_prompt(chunk_entities)

        # Build entity lookup for validation
        entity_by_name: dict[str, ExtractedEntity] = {}
        for e in chunk_entities:
            entity_by_name[e.name.lower()] = e
            for alias in e.aliases:
                entity_by_name[alias.lower()] = e

        try:
            relations_data = await self._call_llm_with_retry(chunk.text, entities_str, chunk.id)
        except RelationExtractionError:
            raise
        except Exception as e:
            self._metrics.extraction_errors += 1
            logger.error(
                "Relation extraction failed unexpectedly",
                chunk_id=chunk.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

        if not relations_data:
            self._metrics.empty_extractions += 1
            logger.debug("No relations extracted", chunk_id=chunk.id)
            return []

        relations = []
        filtered_by_confidence = 0
        invalid_entities = 0

        for rel_data in relations_data:
            try:
                source_name = rel_data.get("source", "").strip()
                target_name = rel_data.get("target", "").strip()
                predicate = self._normalize_predicate(rel_data.get("predicate", "RELATED_TO"))
                confidence = float(rel_data.get("confidence", 0.7))

                # Validate source and target exist
                source_entity = self._find_entity_by_name(source_name, chunk_entities)
                target_entity = self._find_entity_by_name(target_name, chunk_entities)

                if not source_entity or not target_entity:
                    invalid_entities += 1
                    if self._validate_entities:
                        logger.debug(
                            "Skipping relation with unknown entity",
                            source=source_name,
                            target=target_name,
                            source_found=source_entity is not None,
                            target_found=target_entity is not None,
                        )
                        continue
                    else:
                        # Create placeholder if validation disabled (not recommended)
                        logger.warning(
                            "Creating relation with unvalidated entities",
                            source=source_name,
                            target=target_name,
                        )

                # Filter by confidence
                if confidence < self._min_confidence:
                    filtered_by_confidence += 1
                    continue

                # Filter by allowed predicates
                if self._allowed_predicates and predicate not in self._allowed_predicates:
                    continue

                relation = ExtractedRelation(
                    id=self._generate_relation_id(source_entity.id, target_entity.id, predicate),
                    source_entity=source_entity.id,
                    target_entity=target_entity.id,
                    predicate=predicate,
                    description=rel_data.get("description", ""),
                    chunk_ids=[chunk.id],
                    confidence=confidence,
                    properties={
                        "source_name": source_entity.name,
                        "target_name": target_entity.name,
                    },
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                relations.append(relation)

            except Exception as e:
                logger.warning("Failed to parse relation", rel_data=rel_data, error=str(e))
                continue

        # Update metrics
        self._metrics.relations_filtered_by_confidence += filtered_by_confidence
        self._metrics.relations_with_invalid_entities += invalid_entities
        self._metrics.total_relations += len(relations)
        if relations:
            self._metrics.chunks_with_relations += 1

        logger.info(
            "Relations extracted from chunk",
            chunk_id=chunk.id,
            relation_count=len(relations),
            filtered_by_confidence=filtered_by_confidence,
            invalid_entities=invalid_entities,
        )
        return relations

    async def extract_batch(
        self,
        chunks: list[TextChunk],
        entities: list[ExtractedEntity],
        deduplicate: bool = True,
        max_concurrent: int = 3,
        batch_timeout: float = 180.0,
    ) -> list[ExtractedRelation]:
        """
        Extract relations from multiple chunks with concurrent processing.

        Args:
            chunks: List of text chunks
            entities: All entities extracted from chunks
            deduplicate: Whether to merge duplicate relations
            max_concurrent: Max concurrent LLM calls
            batch_timeout: Timeout for each batch in seconds

        Returns:
            List of extracted relations

        Raises:
            RelationExtractionError: If extraction fails critically
        """
        import asyncio

        if not chunks:
            logger.warning("No chunks provided for relation extraction")
            return []

        if len(entities) < 2:
            logger.warning("Not enough entities for relation extraction", entity_count=len(entities))
            return []

        # Reset metrics for this batch run
        self.reset_metrics()
        self._metrics.start_time = datetime.utcnow()

        all_relations: list[ExtractedRelation] = []
        failed_chunks: list[str] = []
        total_chunks = len(chunks)

        # Process in concurrent batches
        for i in range(0, total_chunks, max_concurrent):
            batch_chunks = chunks[i : i + max_concurrent]
            batch_num = (i // max_concurrent) + 1
            total_batches = (total_chunks + max_concurrent - 1) // max_concurrent

            logger.info(
                "Processing relation extraction batch",
                batch=batch_num,
                total_batches=total_batches,
                chunks_in_batch=len(batch_chunks),
            )

            # Create tasks for concurrent extraction
            tasks = [self.extract(chunk, entities) for chunk in batch_chunks]

            try:
                # Run concurrently with timeout
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=batch_timeout,
                )

                for idx, result in enumerate(results):
                    chunk = batch_chunks[idx]
                    if isinstance(result, RelationExtractionError):
                        if not result.recoverable:
                            logger.error(
                                "Non-recoverable relation extraction error",
                                chunk_id=chunk.id,
                                error=str(result),
                            )
                            raise result
                        logger.warning(
                            "Chunk relation extraction failed (recoverable)",
                            chunk_id=chunk.id,
                            error=str(result),
                        )
                        failed_chunks.append(chunk.id)
                    elif isinstance(result, Exception):
                        logger.warning(
                            "Chunk relation extraction failed",
                            chunk_id=chunk.id,
                            error=str(result),
                            error_type=type(result).__name__,
                        )
                        failed_chunks.append(chunk.id)
                    elif result:
                        all_relations.extend(result)

            except TimeoutError:
                logger.error(
                    "Relation extraction batch timed out",
                    batch=batch_num,
                    timeout_seconds=batch_timeout,
                )
                failed_chunks.extend([c.id for c in batch_chunks])
                self._metrics.extraction_errors += len(batch_chunks)
                continue

            logger.info(
                "Batch relation extraction completed",
                batch=batch_num,
                relations_found=sum(len(r) for r in results if isinstance(r, list)),
                total_relations=len(all_relations),
            )

            # Invoke progress callback for real-time updates
            if self._progress_callback:
                processed_chunks = min(i + max_concurrent, total_chunks)
                progress_percent = processed_chunks / total_chunks * 100
                # Scale progress to 0.50 ~ 0.70 range (relation extraction phase)
                scaled_progress = 0.50 + (progress_percent / 100) * 0.20
                self._progress_callback(
                    "extract_relations",
                    scaled_progress,
                    f"Extracting relations: {processed_chunks}/{total_chunks} chunks",
                    {
                        "relations_so_far": len(all_relations),
                        "chunks_processed": processed_chunks,
                        "total_chunks": total_chunks,
                        "batch_index": batch_num,
                    },
                )

        if deduplicate:
            before_dedup = len(all_relations)
            all_relations = self._deduplicate_relations(all_relations)
            logger.info(
                "Relation deduplication completed",
                before=before_dedup,
                after=len(all_relations),
            )

        # Finalize metrics
        self._metrics.end_time = datetime.utcnow()
        self._metrics.total_relations = len(all_relations)

        logger.info(
            "All relation extraction completed",
            chunk_count=total_chunks,
            relation_count=len(all_relations),
            failed_chunks=len(failed_chunks),
            metrics=self._metrics.to_dict(),
        )

        return all_relations

    def _deduplicate_relations(
        self,
        relations: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        """
        Merge duplicate relations.

        Combines chunk_ids and takes highest confidence.
        """
        relation_map: dict[str, ExtractedRelation] = {}

        for relation in relations:
            if relation.id in relation_map:
                existing = relation_map[relation.id]
                merged_chunks = list(set(existing.chunk_ids + relation.chunk_ids))
                merged_confidence = max(existing.confidence, relation.confidence)
                merged_description = existing.description or relation.description

                relation_map[relation.id] = ExtractedRelation(
                    id=relation.id,
                    source_entity=existing.source_entity,
                    target_entity=existing.target_entity,
                    predicate=existing.predicate,
                    description=merged_description,
                    chunk_ids=merged_chunks,
                    confidence=merged_confidence,
                    properties={**existing.properties, **relation.properties},
                )
            else:
                relation_map[relation.id] = relation

        return list(relation_map.values())

    def relations_to_triplets(
        self,
        relations: list[ExtractedRelation],
        entities: list[ExtractedEntity],
    ) -> list[Triplet]:
        """
        Convert relations to knowledge graph triplets.

        Args:
            relations: List of extracted relations
            entities: List of entities (for type lookup)

        Returns:
            List of Triplet objects
        """
        # Build entity lookup
        entity_map = {e.id: e for e in entities}

        triplets = []
        for relation in relations:
            source_entity = entity_map.get(relation.source_entity)
            target_entity = entity_map.get(relation.target_entity)

            if not source_entity or not target_entity:
                continue

            triplet = Triplet(
                subject=source_entity.name,
                predicate=relation.predicate,
                object=target_entity.name,
                subject_type=source_entity.type,
                object_type=target_entity.type,
                confidence=relation.confidence,
                source_chunk_id=relation.chunk_ids[0] if relation.chunk_ids else None,
            )
            triplets.append(triplet)

        return triplets


# Coref-aware relation extraction prompt
COREF_RELATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert relation extraction system with coreference resolution.
Given a text and entities, extract relations while resolving pronouns and coreferences.

## Known Entities
{entities}

## Guidelines
1. Resolve pronouns (he, she, it, they) to their antecedent entities
2. Handle coreferences (the company → Microsoft, the CEO → Satya Nadella)
3. Extract implicit relations when strongly implied
4. Use consistent entity names (prefer the canonical form from the entity list)

## Output Format
JSON with "relations" array containing: source, target, predicate, description, confidence

Only return valid JSON.""",
        ),
        (
            "human",
            """Extract relations with coreference resolution:

---
{text}
---

Return relations as JSON:""",
        ),
    ]
)


class CorefRelationExtractor(RelationExtractor):
    """
    Relation extractor with coreference resolution.

    Resolves pronouns and indirect references to extract
    relations that may be implicit in the text.
    """

    def __init__(self, llm: BaseChatModel, **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self._coref_chain = COREF_RELATION_PROMPT | self._llm | self._parser

    async def extract(
        self,
        chunk: TextChunk,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        """Extract relations with coreference resolution."""
        if len(entities) < 2:
            return []

        chunk_entities = [e for e in entities if chunk.id in e.chunk_ids] or entities[:10]
        entities_str = self._format_entities_for_prompt(chunk_entities)

        try:
            result = await self._coref_chain.ainvoke(
                {
                    "text": chunk.text,
                    "entities": entities_str,
                }
            )
            relations_data = result.get("relations", [])
        except Exception as e:
            logger.warning("Coref extraction failed, falling back", error=str(e))
            return await super().extract(chunk, entities)

        # Process relations (same as parent class)
        relations = []
        for rel_data in relations_data:
            try:
                source_name = rel_data.get("source", "").strip()
                target_name = rel_data.get("target", "").strip()
                predicate = self._normalize_predicate(rel_data.get("predicate", "RELATED_TO"))
                confidence = float(rel_data.get("confidence", 0.7))

                source_entity = self._find_entity_by_name(source_name, chunk_entities)
                target_entity = self._find_entity_by_name(target_name, chunk_entities)

                if not source_entity or not target_entity:
                    continue

                if confidence < self._min_confidence:
                    continue

                relation = ExtractedRelation(
                    id=self._generate_relation_id(source_entity.id, target_entity.id, predicate),
                    source_entity=source_entity.id,
                    target_entity=target_entity.id,
                    predicate=predicate,
                    description=rel_data.get("description", ""),
                    chunk_ids=[chunk.id],
                    confidence=confidence,
                    properties={
                        "source_name": source_entity.name,
                        "target_name": target_entity.name,
                        "coref_resolved": True,
                    },
                )
                relations.append(relation)
            except Exception:
                continue

        return relations

"""
Entity Extractor Module.

LLM-based Named Entity Recognition for knowledge graph construction.
Enhanced with validation, retry logic, and extraction quality metrics.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.sddi.state import EntityType, ExtractedEntity, TextChunk

logger = structlog.get_logger(__name__)


class EntityExtractionError(Exception):
    """Raised when entity extraction fails critically."""

    def __init__(self, message: str, error_type: str = "unknown", recoverable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable


@dataclass
class ExtractionMetrics:
    """Metrics for tracking extraction quality."""

    chunks_processed: int = 0
    chunks_with_entities: int = 0
    total_entities: int = 0
    entities_filtered_by_confidence: int = 0
    entities_filtered_by_type: int = 0
    extraction_errors: int = 0
    retry_count: int = 0
    empty_extractions: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    @property
    def extraction_rate(self) -> float:
        """Percentage of chunks that yielded entities."""
        if self.chunks_processed == 0:
            return 0.0
        return self.chunks_with_entities / self.chunks_processed

    @property
    def avg_entities_per_chunk(self) -> float:
        """Average entities per chunk (only chunks with entities)."""
        if self.chunks_with_entities == 0:
            return 0.0
        return self.total_entities / self.chunks_with_entities

    @property
    def duration_seconds(self) -> float:
        """Extraction duration in seconds."""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_with_entities": self.chunks_with_entities,
            "total_entities": self.total_entities,
            "entities_filtered_by_confidence": self.entities_filtered_by_confidence,
            "entities_filtered_by_type": self.entities_filtered_by_type,
            "extraction_errors": self.extraction_errors,
            "retry_count": self.retry_count,
            "empty_extractions": self.empty_extractions,
            "extraction_rate": round(self.extraction_rate, 3),
            "avg_entities_per_chunk": round(self.avg_entities_per_chunk, 2),
            "duration_seconds": round(self.duration_seconds, 2),
        }


class EntityExtractionOutput(BaseModel):
    """Schema for LLM entity extraction output."""

    entities: list[dict[str, Any]] = Field(description="List of extracted entities")


ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert Named Entity Recognition (NER) system.
Extract all named entities from the given text and classify them into the following types:

## Entity Types
- PERSON: People, characters, historical figures
- ORGANIZATION: Companies, institutions, agencies, teams
- LOCATION: Places, cities, countries, regions, addresses
- DATE: Dates, time periods, years, seasons
- EVENT: Events, incidents, occasions, meetings
- CONCEPT: Abstract concepts, theories, methodologies, ideas
- PRODUCT: Products, services, brands, software
- TECHNOLOGY: Technologies, tools, frameworks, programming languages
- METRIC: Numbers, statistics, measurements, KPIs
- DOCUMENT: Documents, reports, articles, papers

## Guidelines
1. Extract ALL entities, even if they appear multiple times
2. Use the most specific entity type possible
3. Include brief descriptions when context provides them
4. Capture alternative names/aliases when mentioned
5. Assign confidence scores (0.0-1.0) based on certainty
6. For ambiguous entities, prefer the most contextually relevant type

## Output Format
Return a JSON object with an "entities" array containing objects with:
- name: The entity mention as it appears in text
- type: One of the entity types above
- description: Brief description based on context (optional)
- aliases: List of alternative names mentioned (optional)
- confidence: Confidence score 0.0-1.0

Only return valid JSON, no additional text.""",
        ),
        (
            "human",
            """Extract all named entities from the following text:

---
{text}
---

Return the entities as JSON:""",
        ),
    ]
)


class EntityExtractor:
    """
    LLM-based entity extractor for knowledge graph construction.

    Uses structured output parsing to extract named entities
    with their types, descriptions, and confidence scores.

    Enhanced with:
    - Extraction quality metrics
    - Configurable retry logic
    - Minimum extraction validation
    - Proper error propagation
    """

    def __init__(
        self,
        llm: BaseChatModel,
        entity_types: list[EntityType] | None = None,
        min_confidence: float = 0.5,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        min_text_length: int = 50,
        fail_on_empty: bool = False,
    ) -> None:
        """
        Initialize the entity extractor.

        Args:
            llm: LangChain chat model for extraction
            entity_types: Filter for specific entity types (None = all)
            min_confidence: Minimum confidence threshold
            max_retries: Maximum retry attempts for transient failures
            retry_delay: Base delay between retries (exponential backoff)
            min_text_length: Minimum chunk text length to process
            fail_on_empty: Whether to raise error if no entities found
        """
        self._llm = llm
        self._entity_types = entity_types
        self._min_confidence = min_confidence
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._min_text_length = min_text_length
        self._fail_on_empty = fail_on_empty
        self._parser = JsonOutputParser(pydantic_object=EntityExtractionOutput)
        self._chain = ENTITY_EXTRACTION_PROMPT | self._llm | self._parser
        self._metrics = ExtractionMetrics()

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a deterministic entity ID."""
        key = f"{entity_type}:{name.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _parse_entity_type(self, type_str: str) -> EntityType:
        """Parse entity type string to enum."""
        type_upper = type_str.upper().strip()
        try:
            return EntityType(type_upper)
        except ValueError:
            # Map common variations
            type_map = {
                "ORG": EntityType.ORGANIZATION,
                "LOC": EntityType.LOCATION,
                "GPE": EntityType.LOCATION,
                "PER": EntityType.PERSON,
                "TIME": EntityType.DATE,
                "TECH": EntityType.TECHNOLOGY,
                "COMPANY": EntityType.ORGANIZATION,
                "PLACE": EntityType.LOCATION,
            }
            return type_map.get(type_upper, EntityType.OTHER)

    def get_metrics(self) -> ExtractionMetrics:
        """Get extraction metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset extraction metrics for a new run."""
        self._metrics = ExtractionMetrics()

    async def _call_llm_with_retry(
        self,
        text: str,
        chunk_id: str,
    ) -> list[dict[str, Any]]:
        """Call LLM with retry logic for transient failures."""
        import asyncio

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                result = await self._chain.ainvoke({"text": text})
                return result.get("entities", [])

            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON parsing failed, retrying",
                    chunk_id=chunk_id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                last_error = e

            except Exception as e:
                error_str = str(e).lower()

                # Non-recoverable errors - don't retry
                if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                    raise EntityExtractionError(
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

                # Timeout - retry with backoff
                if "timeout" in error_str:
                    wait_time = self._retry_delay * (attempt + 1)
                    logger.warning(
                        "Timeout, retrying",
                        chunk_id=chunk_id,
                        wait_seconds=wait_time,
                        attempt=attempt + 1,
                    )
                    self._metrics.retry_count += 1
                    await asyncio.sleep(wait_time)
                    last_error = e
                    continue

                # Other errors - retry with backoff
                wait_time = self._retry_delay * (attempt + 1)
                self._metrics.retry_count += 1
                await asyncio.sleep(wait_time)
                last_error = e

        # All retries exhausted
        self._metrics.extraction_errors += 1
        raise EntityExtractionError(
            f"Entity extraction failed after {self._max_retries} attempts: {last_error}",
            error_type="extraction_failed",
            recoverable=True,
        )

    async def extract(
        self,
        chunk: TextChunk,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from a single text chunk.

        Args:
            chunk: Text chunk to process

        Returns:
            List of extracted entities

        Raises:
            EntityExtractionError: If extraction fails and fail_on_empty is True
        """
        self._metrics.chunks_processed += 1

        # Skip chunks that are too short
        if len(chunk.text.strip()) < self._min_text_length:
            logger.debug(
                "Skipping short chunk",
                chunk_id=chunk.id,
                text_length=len(chunk.text),
                min_length=self._min_text_length,
            )
            return []

        try:
            logger.debug("Starting entity extraction", chunk_id=chunk.id, text_length=len(chunk.text))

            # Use retry-enabled LLM call
            entities_data = await self._call_llm_with_retry(chunk.text, chunk.id)
            logger.debug("LLM response received", chunk_id=chunk.id, entities_count=len(entities_data))

        except EntityExtractionError:
            raise  # Re-raise extraction errors
        except Exception as e:
            self._metrics.extraction_errors += 1
            logger.error(
                "Entity extraction failed unexpectedly",
                chunk_id=chunk.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            if self._fail_on_empty:
                raise EntityExtractionError(
                    f"Unexpected error during extraction: {e}",
                    error_type="unexpected_error",
                    recoverable=True,
                ) from e
            return []

        # Check for empty extraction
        if not entities_data:
            self._metrics.empty_extractions += 1
            logger.warning(
                "No entities extracted from chunk",
                chunk_id=chunk.id,
                text_preview=chunk.text[:100] if chunk.text else "EMPTY",
            )
            if self._fail_on_empty and len(chunk.text) > 100:
                raise EntityExtractionError(
                    f"No entities found in substantial text (chunk: {chunk.id})",
                    error_type="empty_extraction",
                    recoverable=True,
                )
            return []

        entities = []
        filtered_by_confidence = 0
        filtered_by_type = 0

        for entity_data in entities_data:
            try:
                entity_type = self._parse_entity_type(entity_data.get("type", "OTHER"))

                # Filter by entity types if specified
                if self._entity_types and entity_type not in self._entity_types:
                    filtered_by_type += 1
                    continue

                confidence = float(entity_data.get("confidence", 0.8))

                # Filter by confidence threshold
                if confidence < self._min_confidence:
                    filtered_by_confidence += 1
                    continue

                name = entity_data.get("name", "").strip()
                if not name:
                    continue

                entity = ExtractedEntity(
                    id=self._generate_entity_id(name, entity_type.value),
                    name=name,
                    type=entity_type,
                    description=entity_data.get("description", ""),
                    aliases=entity_data.get("aliases", []),
                    chunk_ids=[chunk.id],
                    confidence=confidence,
                    properties=entity_data.get("properties", {}),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                entities.append(entity)

            except Exception as e:
                logger.warning("Failed to parse entity", entity_data=entity_data, error=str(e))
                continue

        # Update metrics
        self._metrics.entities_filtered_by_confidence += filtered_by_confidence
        self._metrics.entities_filtered_by_type += filtered_by_type
        self._metrics.total_entities += len(entities)
        if entities:
            self._metrics.chunks_with_entities += 1

        logger.info(
            "Entities extracted from chunk",
            chunk_id=chunk.id,
            entity_count=len(entities),
            filtered_by_confidence=filtered_by_confidence,
            filtered_by_type=filtered_by_type,
        )
        return entities

    async def extract_batch(
        self,
        chunks: list[TextChunk],
        deduplicate: bool = True,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from multiple chunks.

        Args:
            chunks: List of text chunks to process
            deduplicate: Whether to merge duplicate entities

        Returns:
            List of extracted entities (deduplicated if requested)
        """
        all_entities: list[ExtractedEntity] = []

        for chunk in chunks:
            chunk_entities = await self.extract(chunk)
            all_entities.extend(chunk_entities)

        if deduplicate:
            all_entities = self._deduplicate_entities(all_entities)

        logger.info(
            "Batch entity extraction completed",
            chunk_count=len(chunks),
            entity_count=len(all_entities),
        )
        return all_entities

    def _deduplicate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """
        Merge duplicate entities by ID.

        Combines chunk_ids and takes highest confidence.
        """
        entity_map: dict[str, ExtractedEntity] = {}

        for entity in entities:
            if entity.id in entity_map:
                existing = entity_map[entity.id]
                # Merge chunk_ids
                merged_chunks = list(set(existing.chunk_ids + entity.chunk_ids))
                # Take higher confidence
                merged_confidence = max(existing.confidence, entity.confidence)
                # Merge aliases
                merged_aliases = list(set(existing.aliases + entity.aliases))
                # Update description if empty
                merged_description = existing.description or entity.description

                entity_map[entity.id] = ExtractedEntity(
                    id=entity.id,
                    name=existing.name,  # Keep first occurrence
                    type=existing.type,
                    description=merged_description,
                    aliases=merged_aliases,
                    chunk_ids=merged_chunks,
                    confidence=merged_confidence,
                    properties={**existing.properties, **entity.properties},
                )
            else:
                entity_map[entity.id] = entity

        return list(entity_map.values())


# Prompt for batch extraction (more efficient for multiple chunks)
BATCH_ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert Named Entity Recognition system.
Extract all unique named entities from the provided text chunks.

## Entity Types
PERSON, ORGANIZATION, LOCATION, DATE, EVENT, CONCEPT, PRODUCT, TECHNOLOGY, METRIC, DOCUMENT

## Output Format
Return JSON with an "entities" array. Each entity should have:
- name: Entity name
- type: Entity type from the list above
- description: Brief description
- confidence: Score 0.0-1.0
- source_chunks: List of chunk indices where this entity appears (0-indexed)

Deduplicate entities across chunks - list each unique entity only once.""",
        ),
        (
            "human",
            """Extract entities from these text chunks:

{chunks_text}

Return deduplicated entities as JSON:""",
        ),
    ]
)


class BatchEntityExtractor(EntityExtractor):
    """
    Optimized entity extractor for batch processing.

    Processes multiple chunks in a single LLM call for efficiency.

    Enhanced with:
    - Improved error recovery
    - Better progress tracking
    - Configurable timeouts
    - Extraction validation
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_chunks_per_batch: int = 5,
        batch_timeout: float = 120.0,
        min_extraction_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize batch entity extractor.

        Args:
            llm: LangChain chat model
            max_chunks_per_batch: Max chunks per LLM call
            batch_timeout: Timeout for batch processing in seconds
            min_extraction_rate: Minimum expected extraction rate (warning if below)
            **kwargs: Additional arguments for EntityExtractor
        """
        super().__init__(llm, **kwargs)
        self._max_chunks_per_batch = max_chunks_per_batch
        self._batch_timeout = batch_timeout
        self._min_extraction_rate = min_extraction_rate
        self._batch_chain = BATCH_ENTITY_PROMPT | self._llm | self._parser

    async def extract_batch(
        self,
        chunks: list[TextChunk],
        deduplicate: bool = True,
        max_concurrent: int = 2,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from chunks using batched LLM calls with concurrency.

        Args:
            chunks: List of text chunks
            deduplicate: Whether to deduplicate entities
            max_concurrent: Max concurrent batch extractions

        Returns:
            List of extracted entities

        Raises:
            EntityExtractionError: If extraction fails critically
        """
        import asyncio

        if not chunks:
            logger.warning("No chunks provided for extraction")
            return []

        # Reset metrics for this batch run
        self.reset_metrics()
        self._metrics.start_time = datetime.utcnow()

        all_entities: list[ExtractedEntity] = []
        failed_batches: list[int] = []
        total_batches = (len(chunks) + self._max_chunks_per_batch - 1) // self._max_chunks_per_batch

        # Filter out short chunks
        valid_chunks = [c for c in chunks if len(c.text.strip()) >= self._min_text_length]
        if len(valid_chunks) < len(chunks):
            logger.info(
                "Filtered short chunks",
                original=len(chunks),
                valid=len(valid_chunks),
                min_length=self._min_text_length,
            )

        if not valid_chunks:
            logger.warning("No valid chunks after filtering")
            return []

        # Create all batch groups
        batch_groups = [
            valid_chunks[i : i + self._max_chunks_per_batch]
            for i in range(0, len(valid_chunks), self._max_chunks_per_batch)
        ]

        # Process batch groups concurrently
        for group_start in range(0, len(batch_groups), max_concurrent):
            concurrent_batches = batch_groups[group_start : group_start + max_concurrent]
            batch_nums = list(range(group_start + 1, group_start + len(concurrent_batches) + 1))

            logger.info(
                "Processing entity extraction batches",
                batches=batch_nums,
                total_batches=total_batches,
            )

            # Create tasks for concurrent extraction
            tasks = [self._extract_batch_single(batch) for batch in concurrent_batches]

            try:
                # Run with configurable timeout
                timeout = self._batch_timeout * len(concurrent_batches)
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )

                for idx, result in enumerate(results):
                    batch_num = batch_nums[idx]
                    if isinstance(result, EntityExtractionError):
                        if not result.recoverable:
                            # Non-recoverable error - stop processing
                            logger.error(
                                "Non-recoverable extraction error",
                                batch=batch_num,
                                error=str(result),
                            )
                            raise result
                        logger.warning(
                            "Batch entity extraction failed (recoverable)",
                            batch=batch_num,
                            error=str(result),
                        )
                        failed_batches.append(batch_num)
                    elif isinstance(result, Exception):
                        logger.warning(
                            "Batch entity extraction failed",
                            batch=batch_num,
                            error=str(result),
                            error_type=type(result).__name__,
                        )
                        failed_batches.append(batch_num)
                    else:
                        all_entities.extend(result)
                        logger.info(
                            "Batch extraction completed",
                            batch=batch_num,
                            entities_found=len(result),
                            total_entities=len(all_entities),
                        )

            except TimeoutError:
                logger.error(
                    "Entity extraction timed out",
                    batches=batch_nums,
                    timeout_seconds=timeout,
                )
                failed_batches.extend(batch_nums)
                self._metrics.extraction_errors += len(batch_nums)
                continue

        if deduplicate:
            before_dedup = len(all_entities)
            all_entities = self._deduplicate_entities(all_entities)
            logger.info(
                "Entity deduplication completed",
                before=before_dedup,
                after=len(all_entities),
            )

        # Finalize metrics
        self._metrics.end_time = datetime.utcnow()
        self._metrics.total_entities = len(all_entities)
        self._metrics.chunks_processed = len(valid_chunks)

        # Check extraction quality
        if self._metrics.extraction_rate < self._min_extraction_rate and len(valid_chunks) > 5:
            logger.warning(
                "Low extraction rate detected",
                extraction_rate=round(self._metrics.extraction_rate, 3),
                min_expected=self._min_extraction_rate,
                chunks_processed=self._metrics.chunks_processed,
                entities_found=len(all_entities),
            )

        logger.info(
            "All entity extraction completed",
            chunk_count=len(chunks),
            valid_chunks=len(valid_chunks),
            entity_count=len(all_entities),
            failed_batches=len(failed_batches),
            extraction_rate=round(self._metrics.extraction_rate, 3),
            metrics=self._metrics.to_dict(),
        )

        return all_entities

    async def _extract_batch_single(
        self,
        chunks: list[TextChunk],
        retry_count: int = 0,
    ) -> list[ExtractedEntity]:
        """Process a single batch of chunks with retry logic."""
        import asyncio

        # Format chunks for prompt
        chunks_text = "\n\n".join(
            f"[Chunk {i}] (ID: {chunk.id})\n{chunk.text}" for i, chunk in enumerate(chunks)
        )

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                logger.debug(
                    "Calling LLM for entity extraction",
                    chunk_count=len(chunks),
                    attempt=attempt + 1,
                )
                result = await self._batch_chain.ainvoke({"chunks_text": chunks_text})
                entities_data = result.get("entities", [])
                logger.debug("LLM response received", entities_count=len(entities_data))

                if not entities_data and len(chunks_text) > 100:
                    self._metrics.empty_extractions += 1
                    logger.warning(
                        "LLM returned 0 entities for non-trivial text",
                        chunk_count=len(chunks),
                        text_preview=chunks_text[:200],
                    )

                # Success - process entities
                break

            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON parsing failed in batch extraction",
                    attempt=attempt + 1,
                    error=str(e),
                )
                last_error = e
                await asyncio.sleep(self._retry_delay * (attempt + 1))
                continue

            except Exception as e:
                error_str = str(e).lower()

                # Non-recoverable: Auth errors
                if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                    raise EntityExtractionError(
                        f"LLM API authentication failed: {e}",
                        error_type="auth_error",
                        recoverable=False,
                    ) from e

                # Rate limit - longer wait
                if "rate limit" in error_str or "429" in error_str:
                    wait_time = self._retry_delay * (attempt + 1) * 2
                    logger.warning(
                        "Rate limited, waiting before retry",
                        wait_seconds=wait_time,
                        attempt=attempt + 1,
                    )
                    self._metrics.retry_count += 1
                    await asyncio.sleep(wait_time)
                    last_error = e
                    continue

                # Timeout - retry with backoff
                if "timeout" in error_str:
                    wait_time = self._retry_delay * (attempt + 1)
                    logger.warning(
                        "Timeout, retrying",
                        wait_seconds=wait_time,
                        attempt=attempt + 1,
                    )
                    self._metrics.retry_count += 1
                    await asyncio.sleep(wait_time)
                    last_error = e
                    continue

                # Other errors
                logger.warning(
                    "Batch extraction error, retrying",
                    error=str(e),
                    error_type=type(e).__name__,
                    attempt=attempt + 1,
                )
                self._metrics.retry_count += 1
                await asyncio.sleep(self._retry_delay * (attempt + 1))
                last_error = e

        else:
            # All retries exhausted - fallback to individual extraction
            logger.warning(
                "Batch extraction failed after retries, falling back to individual extraction",
                retries=self._max_retries,
                last_error=str(last_error) if last_error else "unknown",
            )
            self._metrics.extraction_errors += 1

            # Fallback to individual chunk extraction
            fallback_entities = []
            for chunk in chunks:
                try:
                    chunk_entities = await self.extract(chunk)
                    fallback_entities.extend(chunk_entities)
                except EntityExtractionError as e:
                    if not e.recoverable:
                        raise
                    logger.warning("Individual chunk extraction failed", chunk_id=chunk.id, error=str(e))
                except Exception as e:
                    logger.warning("Individual chunk extraction failed", chunk_id=chunk.id, error=str(e))

            return fallback_entities

        # Process successful extraction
        entities = []
        filtered_count = 0

        for entity_data in entities_data:
            try:
                entity_type = self._parse_entity_type(entity_data.get("type", "OTHER"))

                # Filter by entity types if specified
                if self._entity_types and entity_type not in self._entity_types:
                    filtered_count += 1
                    continue

                confidence = float(entity_data.get("confidence", 0.8))

                if confidence < self._min_confidence:
                    filtered_count += 1
                    continue

                name = entity_data.get("name", "").strip()
                if not name:
                    continue

                # Map source_chunks indices to chunk IDs
                source_indices = entity_data.get("source_chunks", [0])
                chunk_ids = [chunks[idx].id for idx in source_indices if idx < len(chunks)]

                entity = ExtractedEntity(
                    id=self._generate_entity_id(name, entity_type.value),
                    name=name,
                    type=entity_type,
                    description=entity_data.get("description", ""),
                    aliases=entity_data.get("aliases", []),
                    chunk_ids=chunk_ids or [chunks[0].id],
                    confidence=confidence,
                    properties={},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                entities.append(entity)

            except Exception as e:
                logger.warning("Failed to parse batch entity", error=str(e))
                continue

        if entities:
            self._metrics.chunks_with_entities += len(chunks)

        logger.debug(
            "Batch single extraction completed",
            entities_found=len(entities),
            filtered=filtered_count,
            chunks=len(chunks),
        )

        return entities

"""
Resilient Entity Extractor with Advanced Error Handling.

Features:
- Circuit Breaker pattern for API protection
- Exponential backoff with jitter
- Adaptive batch sizing
- Progress checkpointing
- Dead letter queue for failed items
- Health check before processing
- Server error (502, 503, 504) specific handling
"""

import asyncio
import hashlib
import json
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.sddi.state import EntityType, ExtractedEntity, TextChunk

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Error Classification
# =============================================================================


class ErrorCategory(str, Enum):
    """Categorization of errors for appropriate handling."""

    TRANSIENT = "transient"  # 502, 503, 504, timeout - retry with backoff
    RATE_LIMIT = "rate_limit"  # 429 - retry with longer wait
    CLIENT_ERROR = "client_error"  # 400, 401, 403 - don't retry
    PARSE_ERROR = "parse_error"  # JSON parsing failed - retry with different format
    UNKNOWN = "unknown"  # Unknown error - retry with caution


@dataclass
class ClassifiedError:
    """Error with classification and metadata."""

    category: ErrorCategory
    message: str
    original_error: Exception
    retry_after: float = 0.0  # Suggested wait time
    recoverable: bool = True

    @classmethod
    def from_exception(cls, error: Exception) -> "ClassifiedError":
        """Classify an exception into appropriate category."""
        error_str = str(error).lower()
        error_full = repr(error).lower()

        # Check for HTML error responses (502, 503, etc.)
        if "<html>" in error_str or "bad gateway" in error_str or "502" in error_str:
            return cls(
                category=ErrorCategory.TRANSIENT,
                message="Server returned 502 Bad Gateway",
                original_error=error,
                retry_after=5.0,
                recoverable=True,
            )

        if "503" in error_str or "service unavailable" in error_str:
            return cls(
                category=ErrorCategory.TRANSIENT,
                message="Server returned 503 Service Unavailable",
                original_error=error,
                retry_after=10.0,
                recoverable=True,
            )

        if "504" in error_str or "gateway timeout" in error_str:
            return cls(
                category=ErrorCategory.TRANSIENT,
                message="Server returned 504 Gateway Timeout",
                original_error=error,
                retry_after=15.0,
                recoverable=True,
            )

        if "500" in error_str or "internal server error" in error_str:
            return cls(
                category=ErrorCategory.TRANSIENT,
                message="Server returned 500 Internal Server Error",
                original_error=error,
                retry_after=5.0,
                recoverable=True,
            )

        if "timeout" in error_str or "timed out" in error_str:
            return cls(
                category=ErrorCategory.TRANSIENT,
                message="Request timed out",
                original_error=error,
                retry_after=3.0,
                recoverable=True,
            )

        if "connection" in error_str and ("reset" in error_str or "refused" in error_str):
            return cls(
                category=ErrorCategory.TRANSIENT,
                message="Connection error",
                original_error=error,
                retry_after=5.0,
                recoverable=True,
            )

        if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
            # Try to extract retry-after from error
            retry_after = 60.0  # Default
            if "retry-after" in error_full:
                try:
                    import re
                    match = re.search(r"retry-after[:\s]+(\d+)", error_full)
                    if match:
                        retry_after = float(match.group(1))
                except Exception:
                    pass

            return cls(
                category=ErrorCategory.RATE_LIMIT,
                message="Rate limit exceeded",
                original_error=error,
                retry_after=retry_after,
                recoverable=True,
            )

        if "401" in error_str or "unauthorized" in error_str or "api key" in error_str:
            return cls(
                category=ErrorCategory.CLIENT_ERROR,
                message="Authentication failed",
                original_error=error,
                retry_after=0,
                recoverable=False,
            )

        if "403" in error_str or "forbidden" in error_str:
            return cls(
                category=ErrorCategory.CLIENT_ERROR,
                message="Access forbidden",
                original_error=error,
                retry_after=0,
                recoverable=False,
            )

        if "400" in error_str or "bad request" in error_str:
            return cls(
                category=ErrorCategory.CLIENT_ERROR,
                message="Bad request",
                original_error=error,
                retry_after=0,
                recoverable=False,
            )

        if isinstance(error, json.JSONDecodeError):
            return cls(
                category=ErrorCategory.PARSE_ERROR,
                message="JSON parsing failed",
                original_error=error,
                retry_after=1.0,
                recoverable=True,
            )

        return cls(
            category=ErrorCategory.UNKNOWN,
            message=str(error),
            original_error=error,
            retry_after=2.0,
            recoverable=True,
        )


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.

    Prevents cascading failures by stopping requests to a failing service.
    """

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying again
    half_open_max_calls: int = 3  # Test calls in half-open state

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                # Enough successful calls, close circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failure during recovery, reopen
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN after half-open failure")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker OPEN after threshold failures",
                failures=self.failure_count,
                threshold=self.failure_threshold,
            )

    def get_wait_time(self) -> float:
        """Get wait time before next retry."""
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            return max(0, self.recovery_timeout - elapsed)
        return 0


# =============================================================================
# Adaptive Batch Sizing
# =============================================================================


@dataclass
class AdaptiveBatchConfig:
    """Configuration for adaptive batch sizing."""

    initial_size: int = 5
    min_size: int = 1
    max_size: int = 10
    increase_threshold: float = 0.9  # Success rate to increase
    decrease_threshold: float = 0.5  # Success rate to decrease
    increase_factor: float = 1.5
    decrease_factor: float = 0.5
    window_size: int = 10  # Recent operations to consider


class AdaptiveBatchSizer:
    """Dynamically adjusts batch size based on success rate."""

    def __init__(self, config: AdaptiveBatchConfig | None = None):
        self.config = config or AdaptiveBatchConfig()
        self.current_size = self.config.initial_size
        self.results: deque[bool] = deque(maxlen=self.config.window_size)

    def record_result(self, success: bool) -> None:
        """Record operation result."""
        self.results.append(success)
        self._adjust_size()

    def _adjust_size(self) -> None:
        """Adjust batch size based on recent success rate."""
        if len(self.results) < 3:
            return  # Not enough data

        success_rate = sum(self.results) / len(self.results)

        if success_rate >= self.config.increase_threshold:
            new_size = min(
                int(self.current_size * self.config.increase_factor),
                self.config.max_size,
            )
            if new_size != self.current_size:
                logger.info(
                    "Increasing batch size",
                    old_size=self.current_size,
                    new_size=new_size,
                    success_rate=round(success_rate, 2),
                )
                self.current_size = new_size
        elif success_rate <= self.config.decrease_threshold:
            new_size = max(
                int(self.current_size * self.config.decrease_factor),
                self.config.min_size,
            )
            if new_size != self.current_size:
                logger.info(
                    "Decreasing batch size",
                    old_size=self.current_size,
                    new_size=new_size,
                    success_rate=round(success_rate, 2),
                )
                self.current_size = new_size

    def get_size(self) -> int:
        """Get current batch size."""
        return self.current_size


# =============================================================================
# Progress Checkpointing
# =============================================================================


@dataclass
class ProcessingCheckpoint:
    """Checkpoint for resumable processing."""

    pipeline_id: str
    total_chunks: int
    processed_indices: set[int] = field(default_factory=set)
    extracted_entities: list[ExtractedEntity] = field(default_factory=list)
    failed_indices: set[int] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def progress_percent(self) -> float:
        """Get processing progress percentage."""
        if self.total_chunks == 0:
            return 0.0
        return len(self.processed_indices) / self.total_chunks * 100

    @property
    def remaining_indices(self) -> list[int]:
        """Get indices not yet processed."""
        all_indices = set(range(self.total_chunks))
        return sorted(all_indices - self.processed_indices - self.failed_indices)

    def mark_processed(self, index: int, entities: list[ExtractedEntity]) -> None:
        """Mark a chunk as processed."""
        self.processed_indices.add(index)
        self.extracted_entities.extend(entities)
        self.updated_at = datetime.utcnow()

    def mark_failed(self, index: int) -> None:
        """Mark a chunk as failed."""
        self.failed_indices.add(index)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint."""
        return {
            "pipeline_id": self.pipeline_id,
            "total_chunks": self.total_chunks,
            "processed_indices": list(self.processed_indices),
            "failed_indices": list(self.failed_indices),
            "entities_count": len(self.extracted_entities),
            "progress_percent": round(self.progress_percent, 2),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Dead Letter Queue
# =============================================================================


@dataclass
class DeadLetterItem:
    """Item that failed processing and was moved to DLQ."""

    chunk: TextChunk
    error: ClassifiedError
    attempts: int
    first_failure: datetime
    last_failure: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "error_category": self.error.category.value,
            "error_message": self.error.message,
            "attempts": self.attempts,
            "first_failure": self.first_failure.isoformat(),
            "last_failure": self.last_failure.isoformat(),
        }


class DeadLetterQueue:
    """Queue for items that repeatedly fail processing."""

    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts
        self.items: dict[str, DeadLetterItem] = {}
        self.attempt_counts: dict[str, int] = {}

    def should_move_to_dlq(self, chunk_id: str) -> bool:
        """Check if item should be moved to DLQ."""
        return self.attempt_counts.get(chunk_id, 0) >= self.max_attempts

    def record_attempt(self, chunk_id: str) -> int:
        """Record an attempt and return new count."""
        self.attempt_counts[chunk_id] = self.attempt_counts.get(chunk_id, 0) + 1
        return self.attempt_counts[chunk_id]

    def add(self, chunk: TextChunk, error: ClassifiedError) -> None:
        """Add item to DLQ."""
        now = datetime.utcnow()
        attempts = self.attempt_counts.get(chunk.id, 1)

        if chunk.id in self.items:
            item = self.items[chunk.id]
            item.error = error
            item.attempts = attempts
            item.last_failure = now
        else:
            self.items[chunk.id] = DeadLetterItem(
                chunk=chunk,
                error=error,
                attempts=attempts,
                first_failure=now,
                last_failure=now,
            )

        logger.warning(
            "Item moved to dead letter queue",
            chunk_id=chunk.id,
            error_category=error.category.value,
            attempts=attempts,
        )

    def get_all(self) -> list[DeadLetterItem]:
        """Get all DLQ items."""
        return list(self.items.values())

    def get_retryable(self, category: ErrorCategory | None = None) -> list[DeadLetterItem]:
        """Get items that might be retryable."""
        retryable = []
        for item in self.items.values():
            if item.error.recoverable:
                if category is None or item.error.category == category:
                    retryable.append(item)
        return retryable

    def clear(self) -> int:
        """Clear DLQ and return count of cleared items."""
        count = len(self.items)
        self.items.clear()
        self.attempt_counts.clear()
        return count


# =============================================================================
# Enhanced Extraction Metrics
# =============================================================================


@dataclass
class ResilientExtractionMetrics:
    """Extended metrics for resilient extraction."""

    # Basic metrics
    chunks_processed: int = 0
    chunks_with_entities: int = 0
    total_entities: int = 0
    entities_filtered: int = 0

    # Error tracking
    transient_errors: int = 0
    rate_limit_errors: int = 0
    client_errors: int = 0
    parse_errors: int = 0
    unknown_errors: int = 0

    # Retry tracking
    retry_attempts: int = 0
    successful_retries: int = 0

    # Circuit breaker
    circuit_opens: int = 0
    circuit_rejects: int = 0

    # Batch sizing
    batch_size_adjustments: int = 0
    min_batch_size_used: int = 0
    max_batch_size_used: int = 0

    # DLQ
    items_in_dlq: int = 0

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        total_attempts = self.chunks_processed + self.total_errors
        if total_attempts == 0:
            return 0.0
        return self.chunks_processed / total_attempts

    @property
    def total_errors(self) -> int:
        return (
            self.transient_errors
            + self.rate_limit_errors
            + self.client_errors
            + self.parse_errors
            + self.unknown_errors
        )

    def record_error(self, error: ClassifiedError) -> None:
        """Record error by category."""
        if error.category == ErrorCategory.TRANSIENT:
            self.transient_errors += 1
        elif error.category == ErrorCategory.RATE_LIMIT:
            self.rate_limit_errors += 1
        elif error.category == ErrorCategory.CLIENT_ERROR:
            self.client_errors += 1
        elif error.category == ErrorCategory.PARSE_ERROR:
            self.parse_errors += 1
        else:
            self.unknown_errors += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_with_entities": self.chunks_with_entities,
            "total_entities": self.total_entities,
            "entities_filtered": self.entities_filtered,
            "errors": {
                "transient": self.transient_errors,
                "rate_limit": self.rate_limit_errors,
                "client": self.client_errors,
                "parse": self.parse_errors,
                "unknown": self.unknown_errors,
                "total": self.total_errors,
            },
            "retries": {
                "attempts": self.retry_attempts,
                "successful": self.successful_retries,
            },
            "circuit_breaker": {
                "opens": self.circuit_opens,
                "rejects": self.circuit_rejects,
            },
            "batch_sizing": {
                "adjustments": self.batch_size_adjustments,
                "min_used": self.min_batch_size_used,
                "max_used": self.max_batch_size_used,
            },
            "dlq_items": self.items_in_dlq,
            "success_rate": round(self.success_rate, 3),
            "duration_seconds": round(self.duration_seconds, 2),
        }


# =============================================================================
# Prompts
# =============================================================================


class EntityExtractionOutput(BaseModel):
    """Schema for LLM entity extraction output."""

    entities: list[dict[str, Any]] = Field(description="List of extracted entities")


ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert Named Entity Recognition (NER) system.
Extract all named entities from the given text and classify them.

## Entity Types
- PERSON: People, characters, historical figures
- ORGANIZATION: Companies, institutions, agencies
- LOCATION: Places, cities, countries, regions
- DATE: Dates, time periods, years
- EVENT: Events, incidents, occasions
- CONCEPT: Abstract concepts, theories, ideas
- PRODUCT: Products, services, brands
- TECHNOLOGY: Technologies, tools, frameworks
- METRIC: Numbers, statistics, measurements
- DOCUMENT: Documents, reports, articles

## Output Format
Return ONLY valid JSON with an "entities" array:
{{"entities": [
  {{"name": "entity name", "type": "TYPE", "description": "brief description", "confidence": 0.9}}
]}}

Extract ALL entities. Assign confidence scores 0.0-1.0.
If no entities found, return {{"entities": []}}""",
        ),
        (
            "human",
            """Extract entities from this text:

{text}

Return JSON only:""",
        ),
    ]
)


BATCH_ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert NER system. Extract unique named entities from text chunks.

## Entity Types
PERSON, ORGANIZATION, LOCATION, DATE, EVENT, CONCEPT, PRODUCT, TECHNOLOGY, METRIC, DOCUMENT

## Output Format
Return ONLY valid JSON:
{{"entities": [
  {{"name": "...", "type": "...", "description": "...", "confidence": 0.9, "source_chunks": [0,1]}}
]}}

Deduplicate across chunks. source_chunks = 0-indexed list of chunk numbers.
If no entities, return {{"entities": []}}""",
        ),
        (
            "human",
            """Extract entities from these chunks:

{chunks_text}

Return JSON only:""",
        ),
    ]
)


# =============================================================================
# Resilient Entity Extractor
# =============================================================================


class ResilientEntityExtractor:
    """
    Production-grade entity extractor with resilience patterns.

    Features:
    - Circuit breaker for API protection
    - Exponential backoff with jitter
    - Adaptive batch sizing
    - Progress checkpointing
    - Dead letter queue
    - Comprehensive error classification
    """

    def __init__(
        self,
        llm: BaseChatModel,
        min_confidence: float = 0.5,
        entity_types: list[EntityType] | None = None,
        min_text_length: int = 50,
        # Retry configuration
        max_retries: int = 5,
        base_retry_delay: float = 2.0,
        max_retry_delay: float = 60.0,
        # Circuit breaker configuration
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 30.0,
        # Batch configuration
        initial_batch_size: int = 5,
        max_concurrent_batches: int = 2,
        # DLQ configuration
        max_dlq_attempts: int = 5,
        # Health check
        enable_health_check: bool = True,
        # Progress callback
        progress_callback: Callable[[str, float, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._llm = llm
        self._min_confidence = min_confidence
        self._entity_types = entity_types
        self._min_text_length = min_text_length
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._max_concurrent = max_concurrent_batches
        self._enable_health_check = enable_health_check
        self._progress_callback = progress_callback

        # Initialize components
        self._parser = JsonOutputParser(pydantic_object=EntityExtractionOutput)
        self._single_chain = ENTITY_EXTRACTION_PROMPT | self._llm | self._parser
        self._batch_chain = BATCH_ENTITY_PROMPT | self._llm | self._parser

        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
        )

        self._batch_sizer = AdaptiveBatchSizer(
            AdaptiveBatchConfig(initial_size=initial_batch_size)
        )

        self._dlq = DeadLetterQueue(max_attempts=max_dlq_attempts)
        self._metrics = ResilientExtractionMetrics()
        self._checkpoint: ProcessingCheckpoint | None = None

    def _calculate_backoff(self, attempt: int, base_delay: float = 0.0) -> float:
        """Calculate exponential backoff with jitter."""
        if base_delay == 0:
            base_delay = self._base_retry_delay

        # Exponential backoff: base * 2^attempt
        delay = base_delay * (2 ** attempt)

        # Cap at max delay
        delay = min(delay, self._max_retry_delay)

        # Add jitter (0.5 to 1.5 of delay)
        jitter = delay * (0.5 + random.random())

        return jitter

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate deterministic entity ID."""
        key = f"{entity_type}:{name.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _parse_entity_type(self, type_str: str) -> EntityType:
        """Parse entity type string to enum."""
        type_upper = type_str.upper().strip()
        try:
            return EntityType(type_upper)
        except ValueError:
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

    async def health_check(self) -> bool:
        """
        Check if LLM API is healthy before processing.

        Returns:
            True if API is responsive, False otherwise
        """
        if not self._enable_health_check:
            return True

        try:
            test_text = "Test entity: OpenAI is a company."
            result = await asyncio.wait_for(
                self._single_chain.ainvoke({"text": test_text}),
                timeout=30.0,
            )
            return "entities" in result
        except Exception as e:
            logger.warning("Health check failed", error=str(e))
            return False

    async def _call_with_resilience(
        self,
        chain: Any,
        inputs: dict[str, Any],
        context: str = "extraction",
    ) -> dict[str, Any]:
        """
        Call LLM with full resilience: circuit breaker, retries, backoff.
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            wait_time = self._circuit_breaker.get_wait_time()
            self._metrics.circuit_rejects += 1
            logger.warning(
                "Circuit breaker OPEN, request rejected",
                wait_seconds=round(wait_time, 1),
            )
            raise Exception(f"Circuit breaker open. Wait {wait_time:.1f}s before retry.")

        last_error: ClassifiedError | None = None

        for attempt in range(self._max_retries):
            try:
                result = await asyncio.wait_for(
                    chain.ainvoke(inputs),
                    timeout=120.0,  # 2 minute timeout per call
                )

                # Success - record and return
                self._circuit_breaker.record_success()
                if attempt > 0:
                    self._metrics.successful_retries += 1

                return result

            except asyncio.TimeoutError:
                classified = ClassifiedError(
                    category=ErrorCategory.TRANSIENT,
                    message="Request timed out",
                    original_error=asyncio.TimeoutError("LLM call timed out"),
                    retry_after=5.0,
                    recoverable=True,
                )
                last_error = classified
                self._metrics.record_error(classified)

            except Exception as e:
                classified = ClassifiedError.from_exception(e)
                last_error = classified
                self._metrics.record_error(classified)

                # Non-recoverable errors
                if not classified.recoverable:
                    self._circuit_breaker.record_failure()
                    raise

            # Calculate wait time with backoff
            wait_time = self._calculate_backoff(
                attempt,
                last_error.retry_after if last_error else self._base_retry_delay,
            )

            logger.warning(
                f"{context} failed, retrying",
                attempt=attempt + 1,
                max_attempts=self._max_retries,
                error_category=last_error.category.value if last_error else "unknown",
                error_message=last_error.message if last_error else "unknown",
                wait_seconds=round(wait_time, 2),
            )

            self._metrics.retry_attempts += 1
            await asyncio.sleep(wait_time)

        # All retries exhausted
        self._circuit_breaker.record_failure()
        if self._circuit_breaker.state == CircuitState.OPEN:
            self._metrics.circuit_opens += 1

        error_msg = f"All {self._max_retries} retries exhausted"
        if last_error:
            error_msg += f": {last_error.message}"

        raise Exception(error_msg)

    async def extract_single(self, chunk: TextChunk) -> list[ExtractedEntity]:
        """Extract entities from a single chunk."""
        if len(chunk.text.strip()) < self._min_text_length:
            return []

        try:
            result = await self._call_with_resilience(
                self._single_chain,
                {"text": chunk.text},
                context=f"chunk_{chunk.id}",
            )

            entities_data = result.get("entities", [])
            return self._parse_entities(entities_data, [chunk])

        except Exception as e:
            classified = ClassifiedError.from_exception(e)
            self._dlq.record_attempt(chunk.id)

            if self._dlq.should_move_to_dlq(chunk.id):
                self._dlq.add(chunk, classified)
                self._metrics.items_in_dlq += 1

            return []

    def _parse_entities(
        self,
        entities_data: list[dict[str, Any]],
        source_chunks: list[TextChunk],
    ) -> list[ExtractedEntity]:
        """Parse raw entity data into ExtractedEntity objects."""
        entities = []

        for entity_data in entities_data:
            try:
                entity_type = self._parse_entity_type(entity_data.get("type", "OTHER"))

                # Filter by entity types
                if self._entity_types and entity_type not in self._entity_types:
                    self._metrics.entities_filtered += 1
                    continue

                confidence = float(entity_data.get("confidence", 0.8))
                if confidence < self._min_confidence:
                    self._metrics.entities_filtered += 1
                    continue

                name = entity_data.get("name", "").strip()
                if not name:
                    continue

                # Get source chunk IDs
                source_indices = entity_data.get("source_chunks", [0])
                chunk_ids = [
                    source_chunks[idx].id
                    for idx in source_indices
                    if idx < len(source_chunks)
                ]

                if not chunk_ids:
                    chunk_ids = [source_chunks[0].id] if source_chunks else []

                entity = ExtractedEntity(
                    id=self._generate_entity_id(name, entity_type.value),
                    name=name,
                    type=entity_type,
                    description=entity_data.get("description", ""),
                    aliases=entity_data.get("aliases", []),
                    chunk_ids=chunk_ids,
                    confidence=confidence,
                    properties={},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                entities.append(entity)
                self._metrics.total_entities += 1

            except Exception as e:
                logger.warning("Failed to parse entity", error=str(e))
                continue

        if entities:
            self._metrics.chunks_with_entities += 1

        return entities

    async def _process_batch(
        self,
        chunks: list[TextChunk],
        batch_index: int,
    ) -> list[ExtractedEntity]:
        """Process a single batch of chunks."""
        # Format chunks for prompt
        chunks_text = "\n\n".join(
            f"[Chunk {i}]\n{chunk.text}" for i, chunk in enumerate(chunks)
        )

        try:
            result = await self._call_with_resilience(
                self._batch_chain,
                {"chunks_text": chunks_text},
                context=f"batch_{batch_index}",
            )

            entities_data = result.get("entities", [])
            entities = self._parse_entities(entities_data, chunks)

            self._batch_sizer.record_result(True)
            self._metrics.chunks_processed += len(chunks)

            logger.info(
                "Batch processed successfully",
                batch=batch_index,
                chunks=len(chunks),
                entities=len(entities),
            )

            return entities

        except Exception as e:
            self._batch_sizer.record_result(False)

            logger.warning(
                "Batch processing failed, falling back to individual extraction",
                batch=batch_index,
                error=str(e),
            )

            # Fallback to individual extraction
            fallback_entities = []
            for chunk in chunks:
                chunk_entities = await self.extract_single(chunk)
                fallback_entities.extend(chunk_entities)
                self._metrics.chunks_processed += 1

            return fallback_entities

    async def extract_batch(
        self,
        chunks: list[TextChunk],
        pipeline_id: str | None = None,
        resume_checkpoint: ProcessingCheckpoint | None = None,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from multiple chunks with full resilience.

        Args:
            chunks: Text chunks to process
            pipeline_id: Optional pipeline identifier for checkpointing
            resume_checkpoint: Optional checkpoint to resume from

        Returns:
            List of extracted entities
        """
        if not chunks:
            return []

        # Reset metrics
        self._metrics = ResilientExtractionMetrics()

        # Run health check
        if self._enable_health_check:
            is_healthy = await self.health_check()
            if not is_healthy:
                logger.error("LLM API health check failed, aborting extraction")
                raise Exception("LLM API is not healthy")

        # Initialize or resume checkpoint
        if resume_checkpoint:
            self._checkpoint = resume_checkpoint
            logger.info(
                "Resuming from checkpoint",
                processed=len(resume_checkpoint.processed_indices),
                remaining=len(resume_checkpoint.remaining_indices),
            )
        else:
            self._checkpoint = ProcessingCheckpoint(
                pipeline_id=pipeline_id or "unknown",
                total_chunks=len(chunks),
            )

        # Filter valid chunks
        valid_chunks = [c for c in chunks if len(c.text.strip()) >= self._min_text_length]

        if not valid_chunks:
            logger.warning("No valid chunks after filtering")
            return []

        # Create index mapping
        chunk_indices = {chunk.id: idx for idx, chunk in enumerate(chunks)}

        # Get remaining chunks to process
        if self._checkpoint:
            remaining_indices = self._checkpoint.remaining_indices
            chunks_to_process = [chunks[i] for i in remaining_indices if i < len(chunks)]
        else:
            chunks_to_process = valid_chunks

        all_entities: list[ExtractedEntity] = []

        # Add entities from checkpoint
        if self._checkpoint:
            all_entities.extend(self._checkpoint.extracted_entities)

        # Process in adaptive batches
        batch_index = 0
        while chunks_to_process:
            # Get current batch size
            current_batch_size = self._batch_sizer.get_size()
            self._metrics.min_batch_size_used = min(
                self._metrics.min_batch_size_used or current_batch_size,
                current_batch_size,
            )
            self._metrics.max_batch_size_used = max(
                self._metrics.max_batch_size_used,
                current_batch_size,
            )

            # Create batch
            batch = chunks_to_process[:current_batch_size]
            chunks_to_process = chunks_to_process[current_batch_size:]

            # Process batch
            batch_entities = await self._process_batch(batch, batch_index)
            all_entities.extend(batch_entities)

            # Update checkpoint
            if self._checkpoint:
                for chunk in batch:
                    idx = chunk_indices.get(chunk.id, 0)
                    self._checkpoint.mark_processed(idx, [])

            batch_index += 1

            # Log progress
            progress = (
                len(self._checkpoint.processed_indices) / self._checkpoint.total_chunks * 100
                if self._checkpoint
                else 0
            )
            processed_chunks = (
                len(self._checkpoint.processed_indices) if self._checkpoint else 0
            )
            total_chunks_count = (
                self._checkpoint.total_chunks if self._checkpoint else len(chunks)
            )
            logger.info(
                "Extraction progress",
                batch=batch_index,
                progress=f"{progress:.1f}%",
                entities_so_far=len(all_entities),
                batch_size=current_batch_size,
            )

            # Invoke progress callback for real-time updates
            if self._progress_callback:
                # Scale progress to 0.20 ~ 0.45 range (entity extraction phase)
                scaled_progress = 0.20 + (progress / 100) * 0.25
                self._progress_callback(
                    "extract_entities",
                    scaled_progress,
                    f"Extracting entities: {processed_chunks}/{total_chunks_count} chunks",
                    {
                        "entities_so_far": len(all_entities),
                        "chunks_processed": processed_chunks,
                        "total_chunks": total_chunks_count,
                        "batch_index": batch_index,
                    },
                )

        # Deduplicate
        all_entities = self._deduplicate_entities(all_entities)

        # Finalize
        self._metrics.end_time = datetime.utcnow()
        self._metrics.items_in_dlq = len(self._dlq.items)

        logger.info(
            "Entity extraction completed",
            total_chunks=len(chunks),
            valid_chunks=len(valid_chunks),
            total_entities=len(all_entities),
            metrics=self._metrics.to_dict(),
        )

        return all_entities

    def _deduplicate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Merge duplicate entities by ID."""
        entity_map: dict[str, ExtractedEntity] = {}

        for entity in entities:
            if entity.id in entity_map:
                existing = entity_map[entity.id]
                merged_chunks = list(set(existing.chunk_ids + entity.chunk_ids))
                merged_confidence = max(existing.confidence, entity.confidence)
                merged_aliases = list(set(existing.aliases + entity.aliases))
                merged_description = existing.description or entity.description

                entity_map[entity.id] = ExtractedEntity(
                    id=entity.id,
                    name=existing.name,
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

    def get_metrics(self) -> ResilientExtractionMetrics:
        """Get extraction metrics."""
        return self._metrics

    def get_checkpoint(self) -> ProcessingCheckpoint | None:
        """Get current processing checkpoint."""
        return self._checkpoint

    def get_dead_letter_queue(self) -> DeadLetterQueue:
        """Get dead letter queue."""
        return self._dlq

    def get_circuit_breaker_state(self) -> CircuitState:
        """Get circuit breaker state."""
        return self._circuit_breaker.state

    async def retry_dlq_items(self) -> list[ExtractedEntity]:
        """
        Retry items in the dead letter queue.

        Returns:
            Entities extracted from successfully retried items
        """
        retryable = self._dlq.get_retryable(ErrorCategory.TRANSIENT)
        if not retryable:
            return []

        logger.info("Retrying DLQ items", count=len(retryable))

        entities = []
        for item in retryable:
            try:
                chunk_entities = await self.extract_single(item.chunk)
                entities.extend(chunk_entities)
                # Remove from DLQ on success
                del self._dlq.items[item.chunk.id]
            except Exception as e:
                logger.warning(
                    "DLQ retry failed",
                    chunk_id=item.chunk.id,
                    error=str(e),
                )

        return entities

"""
Telemetry Module for Ontology Reasoning System.

Provides opt-in telemetry collection for:
- User prompts and query patterns
- Knowledge graph extraction data
- Performance metrics and usage statistics

Privacy-first design:
- Disabled by default (opt-in)
- Data anonymization support
- Configurable sampling rate
- Batch processing for efficiency
"""

import asyncio
import hashlib
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
import structlog

from src.config.settings import TelemetrySettings

logger = structlog.get_logger(__name__)


class TelemetryEventType(str, Enum):
    """Types of telemetry events."""

    QUERY = "query"
    KG_EXTRACTION = "kg_extraction"
    METRIC = "metric"
    ERROR = "error"
    SESSION = "session"


@dataclass
class TelemetryEvent:
    """Represents a single telemetry event."""

    event_type: TelemetryEventType
    timestamp: datetime
    session_id: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "data": self.data,
            "metadata": self.metadata,
        }


class TelemetryCollector:
    """
    Collects and sends telemetry data to the configured endpoint.

    Features:
    - Opt-in by default (disabled unless explicitly enabled)
    - Data anonymization
    - Sampling support
    - Batch processing
    - Async HTTP sending
    """

    def __init__(self, settings: TelemetrySettings) -> None:
        """
        Initialize the telemetry collector.

        Args:
            settings: Telemetry configuration settings
        """
        self._settings = settings
        self._enabled = settings.enabled
        self._batch: list[TelemetryEvent] = []
        self._batch_lock = asyncio.Lock()
        self._session_id = self._generate_session_id()
        self._last_flush_time = time.time()
        self._http_client: httpx.AsyncClient | None = None
        self._flush_task: asyncio.Task[None] | None = None

        if self._enabled:
            logger.info(
                "Telemetry enabled",
                endpoint=settings.endpoint_url,
                sampling_rate=settings.sampling_rate,
                anonymize=settings.anonymize_data,
            )

    def _generate_session_id(self) -> str:
        """Generate an anonymous session ID."""
        raw_id = str(uuid.uuid4())
        if self._settings.anonymize_data:
            return hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        return raw_id

    def _should_sample(self) -> bool:
        """Determine if this event should be sampled."""
        return random.random() < self._settings.sampling_rate

    def _anonymize_text(self, text: str) -> str:
        """
        Anonymize text content.

        Applies hashing to preserve patterns while protecting content.
        """
        if not self._settings.anonymize_data:
            return text

        # Hash the text but preserve length info
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        return f"[ANON:{text_hash}:len={len(text)}]"

    def _anonymize_ip(self, ip: str) -> str:
        """Anonymize IP address."""
        if not self._settings.anonymize_data:
            return ip

        # For IPv4, keep first two octets
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"

        # For IPv6, hash it
        return hashlib.sha256(ip.encode()).hexdigest()[:8]

    def _anonymize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Anonymize sensitive fields in data.

        Args:
            data: Raw data dictionary

        Returns:
            Anonymized data dictionary
        """
        if not self._settings.anonymize_data:
            return data

        anonymized = {}
        sensitive_keys = {"ip", "user_id", "email", "name", "api_key"}

        for key, value in data.items():
            if key.lower() in sensitive_keys:
                if isinstance(value, str):
                    anonymized[key] = hashlib.sha256(value.encode()).hexdigest()[:12]
                else:
                    anonymized[key] = "[REDACTED]"
            elif key in {"query", "prompt", "content", "text"}:
                # Anonymize query content but keep structure
                if isinstance(value, str):
                    anonymized[key] = self._anonymize_text(value)
                else:
                    anonymized[key] = value
            else:
                anonymized[key] = value

        return anonymized

    async def track_query(
        self,
        query: str,
        response: dict[str, Any],
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Track a query event.

        Args:
            query: The user's query
            response: Response data (answer, confidence, etc.)
            latency_ms: Query latency in milliseconds
            metadata: Additional metadata
        """
        if not self._enabled or not self._settings.collect_prompts:
            return

        if not self._should_sample():
            return

        data = {
            "query": query,
            "query_length": len(query),
            "response_confidence": response.get("confidence", 0),
            "response_type": response.get("answer_type", "unknown"),
            "latency_ms": latency_ms,
            "iterations": response.get("iterations", 0),
            "evidence_count": response.get("evidence_count", 0),
        }

        event = TelemetryEvent(
            event_type=TelemetryEventType.QUERY,
            timestamp=datetime.now(UTC),
            session_id=self._session_id,
            data=self._anonymize_data(data),
            metadata=metadata or {},
        )

        await self._add_to_batch(event)

    async def track_kg_extraction(
        self,
        document_type: str,
        entity_count: int,
        relation_count: int,
        chunk_count: int,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Track knowledge graph extraction event.

        Args:
            document_type: Type of source document
            entity_count: Number of extracted entities
            relation_count: Number of extracted relations
            chunk_count: Number of processed chunks
            latency_ms: Processing latency in milliseconds
            metadata: Additional metadata
        """
        if not self._enabled or not self._settings.collect_kg_data:
            return

        if not self._should_sample():
            return

        data = {
            "document_type": document_type,
            "entity_count": entity_count,
            "relation_count": relation_count,
            "chunk_count": chunk_count,
            "latency_ms": latency_ms,
        }

        event = TelemetryEvent(
            event_type=TelemetryEventType.KG_EXTRACTION,
            timestamp=datetime.now(UTC),
            session_id=self._session_id,
            data=data,
            metadata=metadata or {},
        )

        await self._add_to_batch(event)

    async def track_metric(
        self,
        metric_name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Track a custom metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        if not self._enabled or not self._settings.collect_metrics:
            return

        if not self._should_sample():
            return

        data = {
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {},
        }

        event = TelemetryEvent(
            event_type=TelemetryEventType.METRIC,
            timestamp=datetime.now(UTC),
            session_id=self._session_id,
            data=data,
        )

        await self._add_to_batch(event)

    async def track_error(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Track an error event.

        Args:
            error_type: Type/class of the error
            error_message: Error message
            context: Additional context about the error
        """
        if not self._enabled:
            return

        data = {
            "error_type": error_type,
            "error_message": self._anonymize_text(error_message) if self._settings.anonymize_data else error_message,
            "context": self._anonymize_data(context or {}),
        }

        event = TelemetryEvent(
            event_type=TelemetryEventType.ERROR,
            timestamp=datetime.now(UTC),
            session_id=self._session_id,
            data=data,
        )

        await self._add_to_batch(event)

    async def _add_to_batch(self, event: TelemetryEvent) -> None:
        """Add event to batch and flush if needed."""
        async with self._batch_lock:
            self._batch.append(event)

            should_flush = (
                len(self._batch) >= self._settings.batch_size
                or time.time() - self._last_flush_time >= self._settings.flush_interval_seconds
            )

            if should_flush:
                await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Send batched events to the telemetry endpoint."""
        if not self._batch:
            return

        events_to_send = self._batch.copy()
        self._batch = []
        self._last_flush_time = time.time()

        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=30.0)

            headers = {"Content-Type": "application/json"}
            if self._settings.api_key:
                headers["Authorization"] = f"Bearer {self._settings.api_key.get_secret_value()}"

            payload = {
                "events": [event.to_dict() for event in events_to_send],
                "batch_id": str(uuid.uuid4()),
                "sent_at": datetime.now(UTC).isoformat(),
            }

            response = await self._http_client.post(
                self._settings.endpoint_url,
                json=payload,
                headers=headers,
            )

            if response.status_code >= 400:
                logger.warning(
                    "Telemetry send failed",
                    status_code=response.status_code,
                    events_count=len(events_to_send),
                )
            else:
                logger.debug(
                    "Telemetry batch sent",
                    events_count=len(events_to_send),
                )

        except Exception as e:
            logger.warning(
                "Telemetry send error",
                error=str(e),
                events_count=len(events_to_send),
            )
            # Don't re-add events to batch to avoid infinite loop

    async def flush(self) -> None:
        """Force flush all pending events."""
        async with self._batch_lock:
            await self._flush_batch()

    async def close(self) -> None:
        """Close the telemetry collector and flush remaining events."""
        await self.flush()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._enabled

    def start_background_flush(self) -> None:
        """Start background flush task."""
        if not self._enabled:
            return

        async def _background_flush() -> None:
            while True:
                await asyncio.sleep(self._settings.flush_interval_seconds)
                async with self._batch_lock:
                    if self._batch:
                        await self._flush_batch()

        self._flush_task = asyncio.create_task(_background_flush())

    def stop_background_flush(self) -> None:
        """Stop background flush task."""
        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None


# Global telemetry collector instance
_telemetry_collector: TelemetryCollector | None = None


def get_telemetry_collector() -> TelemetryCollector | None:
    """Get the global telemetry collector instance."""
    return _telemetry_collector


def init_telemetry(settings: TelemetrySettings) -> TelemetryCollector:
    """
    Initialize the global telemetry collector.

    Args:
        settings: Telemetry configuration settings

    Returns:
        Initialized TelemetryCollector instance
    """
    global _telemetry_collector
    _telemetry_collector = TelemetryCollector(settings)
    return _telemetry_collector


def telemetry_decorator(
    event_type: TelemetryEventType = TelemetryEventType.METRIC,
) -> Callable[..., Any]:
    """
    Decorator to automatically track function execution.

    Args:
        event_type: Type of telemetry event to create

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            collector = get_telemetry_collector()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                if collector and collector.is_enabled:
                    await collector.track_metric(
                        metric_name=f"function.{func.__name__}",
                        value=latency_ms,
                        tags={"status": "success"},
                    )

                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                if collector and collector.is_enabled:
                    await collector.track_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"function": func.__name__, "latency_ms": latency_ms},
                    )

                raise

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            collector = get_telemetry_collector()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                if collector and collector.is_enabled:
                    asyncio.create_task(
                        collector.track_metric(
                            metric_name=f"function.{func.__name__}",
                            value=latency_ms,
                            tags={"status": "success"},
                        )
                    )

                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                if collector and collector.is_enabled:
                    asyncio.create_task(
                        collector.track_error(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            context={"function": func.__name__, "latency_ms": latency_ms},
                        )
                    )

                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

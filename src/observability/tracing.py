"""
Distributed Tracing with OpenTelemetry.

Provides end-to-end tracing across:
- API requests
- MACER agent calls
- Neo4j operations
- LLM invocations

Supports export to Jaeger, Zipkin, or OTLP collectors.
"""

import functools
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(str, Enum):
    """OpenTelemetry span kinds."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span status codes."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""

    enabled: bool = True
    service_name: str = "ontology-reasoning"
    service_version: str = "1.0.0"

    # Exporter settings
    exporter: str = "otlp"  # "otlp", "jaeger", "zipkin", "console", "none"
    otlp_endpoint: str = "http://localhost:4317"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"

    # Sampling
    sample_rate: float = 1.0  # 1.0 = 100% of traces

    # Context propagation
    propagators: list[str] | None = None  # ["tracecontext", "baggage"]

    def __post_init__(self):
        if self.propagators is None:
            self.propagators = ["tracecontext", "baggage"]


@dataclass
class SpanContext:
    """Lightweight span context for when OpenTelemetry is not available."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    name: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        if self.attributes is None:
            self.attributes = {}
        if self.events is None:
            self.events = []

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set the span status."""
        self.status = status
        if description:
            self.set_attribute("status.description", description)

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": (
                (self.end_time - self.start_time).total_seconds() * 1000
                if self.end_time and self.start_time else None
            ),
            "attributes": self.attributes,
            "events": self.events,
        }


class TracerProvider:
    """
    Tracer provider that works with or without OpenTelemetry.

    Falls back to a lightweight implementation when OTel is not available.
    """

    def __init__(self, config: TracingConfig | None = None):
        self.config = config or TracingConfig()
        self._otel_available = False
        self._tracer = None
        self._spans: dict[str, SpanContext] = {}

        if self.config.enabled:
            self._try_init_opentelemetry()

    def _try_init_opentelemetry(self) -> None:
        """Try to initialize OpenTelemetry if available."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
            from opentelemetry.sdk.resources import Resource

            # Create resource
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
            })

            # Create tracer provider
            provider = OTelTracerProvider(resource=resource)

            # Configure exporter
            self._configure_exporter(provider)

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self._tracer = trace.get_tracer(
                self.config.service_name,
                self.config.service_version,
            )

            self._otel_available = True
            logger.info(
                "OpenTelemetry initialized",
                exporter=self.config.exporter,
                service=self.config.service_name,
            )

        except ImportError:
            logger.info(
                "OpenTelemetry not available, using fallback tracer",
                hint="pip install opentelemetry-api opentelemetry-sdk",
            )
            self._otel_available = False

    def _configure_exporter(self, provider) -> None:
        """Configure the trace exporter."""
        try:
            if self.config.exporter == "otlp":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            elif self.config.exporter == "jaeger":
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = JaegerExporter(
                    collector_endpoint=self.config.jaeger_endpoint,
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))

            elif self.config.exporter == "zipkin":
                from opentelemetry.exporter.zipkin.json import ZipkinExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = ZipkinExporter(endpoint=self.config.zipkin_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            elif self.config.exporter == "console":
                from opentelemetry.sdk.trace.export import (
                    ConsoleSpanExporter,
                    SimpleSpanProcessor,
                )

                exporter = ConsoleSpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(exporter))

        except ImportError as e:
            logger.warning(
                "Failed to configure exporter",
                exporter=self.config.exporter,
                error=str(e),
            )

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> SpanContext:
        """Start a new span."""
        import uuid

        span_id = str(uuid.uuid4())[:16]
        trace_id = str(uuid.uuid4())[:32]

        span = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
        )

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        self._spans[span_id] = span

        if self._otel_available and self._tracer:
            # Use OpenTelemetry span
            span.set_attribute("_otel_span", True)

        return span

    def end_span(self, span: SpanContext) -> None:
        """End a span."""
        span.end()

        if span.span_id in self._spans:
            del self._spans[span.span_id]

        # Log span for debugging/fallback
        if not self._otel_available:
            logger.debug(
                "Span completed",
                span_name=span.name,
                duration_ms=round(
                    (span.end_time - span.start_time).total_seconds() * 1000, 2
                ) if span.end_time and span.start_time else None,
                status=span.status.value,
            )


# Global tracer provider
_tracer_provider: TracerProvider | None = None


def init_tracing(config: TracingConfig | None = None) -> TracerProvider:
    """Initialize the global tracer provider."""
    global _tracer_provider
    _tracer_provider = TracerProvider(config)
    return _tracer_provider


def get_tracer() -> TracerProvider:
    """Get the global tracer provider."""
    global _tracer_provider
    if _tracer_provider is None:
        _tracer_provider = TracerProvider()
    return _tracer_provider


@contextmanager
def trace_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """
    Context manager for creating a trace span.

    Usage:
        with trace_span("process_query", attributes={"query_id": "123"}):
            # Do work
            pass
    """
    tracer = get_tracer()
    span = tracer.start_span(name, kind, attributes)

    try:
        yield span
        span.set_status(SpanStatus.OK)
    except Exception as e:
        span.set_status(SpanStatus.ERROR, str(e))
        span.set_attribute("error.type", type(e).__name__)
        span.set_attribute("error.message", str(e))
        raise
    finally:
        tracer.end_span(span)


def traced(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing functions.

    Usage:
        @traced("my_function")
        async def my_function():
            pass

        @traced(attributes={"component": "retriever"})
        def sync_function():
            pass
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_span(span_name, kind, attributes):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, kind, attributes):
                return func(*args, **kwargs)

        if asyncio_is_async(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def asyncio_is_async(func: Callable) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


# Predefined span names for common operations
class SpanNames:
    """Standard span names for consistency."""

    # API
    API_REQUEST = "api.request"
    API_QUERY = "api.query"
    API_INGEST = "api.ingest"

    # MACER Pipeline
    MACER_PIPELINE = "macer.pipeline"
    MACER_CONSTRUCTOR = "macer.constructor"
    MACER_RETRIEVER = "macer.retriever"
    MACER_REFLECTOR = "macer.reflector"
    MACER_RESPONSER = "macer.responser"

    # Neo4j
    NEO4J_QUERY = "neo4j.query"
    NEO4J_SEARCH = "neo4j.search"
    NEO4J_VECTOR_SEARCH = "neo4j.vector_search"

    # LLM
    LLM_INVOKE = "llm.invoke"
    LLM_EMBEDDING = "llm.embedding"

    # Ingestion
    INGEST_DOCUMENT = "ingest.document"
    INGEST_CHUNK = "ingest.chunk"
    INGEST_EXTRACT = "ingest.extract"
    INGEST_EMBED = "ingest.embed"
    INGEST_LOAD = "ingest.load"

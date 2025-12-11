"""
Observability Module.

Provides comprehensive observability for the Ontology Reasoning System:
- Structured logging with JSON output and correlation IDs
- Metrics collection with Prometheus export
- Distributed tracing with OpenTelemetry
- Alerting rules and webhook integrations
"""

from src.observability.logging import (
    configure_logging,
    get_logger,
    LogContext,
    correlation_id_var,
)
from src.observability.correlation import (
    CorrelationMiddleware,
    RequestTimingMiddleware,
    get_correlation_id,
    set_correlation_id,
)
from src.observability.metrics import (
    MetricsRegistry,
    QueryMetricsCollector,
    get_metrics_registry,
    record_query_latency,
    record_macer_iteration,
)
from src.observability.tracing import (
    TracingConfig,
    init_tracing,
    get_tracer,
    trace_span,
)
from src.observability.alerting import (
    AlertRule,
    AlertManager,
    AlertSeverity,
    get_alert_manager,
)
from src.observability.telemetry import (
    TelemetryCollector,
    TelemetryEvent,
    TelemetryEventType,
    get_telemetry_collector,
    init_telemetry,
    telemetry_decorator,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "LogContext",
    "correlation_id_var",
    # Correlation
    "CorrelationMiddleware",
    "RequestTimingMiddleware",
    "get_correlation_id",
    "set_correlation_id",
    # Metrics
    "MetricsRegistry",
    "QueryMetricsCollector",
    "get_metrics_registry",
    "record_query_latency",
    "record_macer_iteration",
    # Tracing
    "TracingConfig",
    "init_tracing",
    "get_tracer",
    "trace_span",
    # Alerting
    "AlertRule",
    "AlertManager",
    "AlertSeverity",
    "get_alert_manager",
    # Telemetry
    "TelemetryCollector",
    "TelemetryEvent",
    "TelemetryEventType",
    "get_telemetry_collector",
    "init_telemetry",
    "telemetry_decorator",
]

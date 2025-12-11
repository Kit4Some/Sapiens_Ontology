"""
Structured Logging Configuration.

Configures structlog for:
- JSON output in production
- Colored console output in development
- Correlation ID injection
- Request/response logging
- Performance timing
"""

import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Literal

import structlog
from structlog.types import EventDict, WrappedLogger

# Context variable for correlation ID
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Context variable for additional log context
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


class LogContext:
    """
    Context manager for adding temporary context to logs.

    Usage:
        with LogContext(user_id="123", action="query"):
            logger.info("Processing request")
    """

    def __init__(self, **kwargs: Any):
        self._context = kwargs
        self._token = None

    def __enter__(self):
        current = _log_context.get().copy()
        current.update(self._context)
        self._token = _log_context.set(current)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            _log_context.reset(self._token)
        return False


def add_correlation_id(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add correlation ID to log events."""
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_log_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add context variables to log events."""
    context = _log_context.get()
    if context:
        event_dict.update(context)
    return event_dict


def add_timestamp(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add ISO timestamp to log events."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_service_info(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add service information to log events."""
    event_dict["service"] = "ontology-reasoning"
    return event_dict


def censor_sensitive_data(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Censor sensitive data from logs."""
    sensitive_keys = {
        "password", "api_key", "secret", "token", "authorization",
        "apikey", "api-key", "bearer", "credential", "private_key"
    }

    def censor_value(key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: censor_value(k, v) for k, v in value.items()}
        elif isinstance(value, str) and any(s in key.lower() for s in sensitive_keys):
            return "***REDACTED***"
        return value

    for key in list(event_dict.keys()):
        event_dict[key] = censor_value(key, event_dict[key])

    return event_dict


def configure_logging(
    level: str = "INFO",
    format: Literal["json", "console"] = "json",
    service_name: str = "ontology-reasoning",
) -> None:
    """
    Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format ("json" for production, "console" for development)
        service_name: Service name for log identification
    """
    # Shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_correlation_id,
        add_log_context,
        censor_sensitive_data,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        renderer = structlog.processors.JSONRenderer()
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Pre-configured loggers for common components
def get_api_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for API layer."""
    return get_logger("api")


def get_workflow_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for workflow/agent layer."""
    return get_logger("workflow")


def get_graph_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for graph/Neo4j layer."""
    return get_logger("graph")


def get_ingestion_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for ingestion pipeline."""
    return get_logger("ingestion")

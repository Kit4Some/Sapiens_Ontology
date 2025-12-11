"""
Request Correlation ID Middleware.

Provides unique correlation IDs for request tracing across services:
- Generates or extracts correlation IDs from headers
- Propagates IDs to all downstream calls
- Adds IDs to response headers
"""

import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.observability.logging import correlation_id_var, get_logger

logger = get_logger(__name__)

# Standard correlation ID headers
CORRELATION_ID_HEADERS = [
    "X-Correlation-ID",
    "X-Request-ID",
    "X-Trace-ID",
]


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())[:12]


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in context."""
    correlation_id_var.set(correlation_id)


def extract_correlation_id(request: Request) -> str | None:
    """Extract correlation ID from request headers."""
    for header in CORRELATION_ID_HEADERS:
        value = request.headers.get(header)
        if value:
            return value
    return None


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle correlation IDs for request tracing.

    Features:
    - Extracts correlation ID from incoming request headers
    - Generates new ID if none provided
    - Adds correlation ID to response headers
    - Sets correlation ID in context for logging

    Usage:
        app.add_middleware(CorrelationMiddleware)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Extract or generate correlation ID
        correlation_id = extract_correlation_id(request)
        if not correlation_id:
            correlation_id = generate_correlation_id()

        # Set in context
        token = correlation_id_var.set(correlation_id)

        # Log request start
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            client_host=request.client.host if request.client else None,
        )

        try:
            # Process request
            response = await call_next(request)

            # Add correlation ID to response
            response.headers["X-Correlation-ID"] = correlation_id

            # Log request completion
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )

            return response

        except Exception as e:
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

        finally:
            # Reset context
            correlation_id_var.reset(token)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request timing.

    Adds timing information to response headers and logs.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        import time

        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Log slow requests
        if duration_ms > 5000:  # > 5 seconds
            logger.warning(
                "Slow request detected",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )

        return response

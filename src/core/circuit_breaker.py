"""
Circuit Breaker Pattern Implementation.

Protects external service calls (LLM, Neo4j, APIs) from cascading failures
by failing fast when a service is unavailable.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service unavailable, requests fail immediately
- HALF_OPEN: Testing if service recovered

Based on Microsoft's Circuit Breaker pattern and Netflix Hystrix.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Generic, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(str, Enum):
    """Types of failures with different weights."""

    TIMEOUT = "timeout"           # Weight: 1
    CONNECTION = "connection"     # Weight: 1
    SERVER_ERROR = "server_error"  # Weight: 2
    AUTH_ERROR = "auth_error"     # Weight: 3 (likely won't self-heal)
    RATE_LIMIT = "rate_limit"     # Weight: 2
    UNKNOWN = "unknown"           # Weight: 1


FAILURE_WEIGHTS = {
    FailureType.TIMEOUT: 1,
    FailureType.CONNECTION: 1,
    FailureType.SERVER_ERROR: 2,
    FailureType.AUTH_ERROR: 3,
    FailureType.RATE_LIMIT: 2,
    FailureType.UNKNOWN: 1,
}


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure threshold to trip the circuit
    failure_threshold: int = 5

    # Weighted failure threshold (alternative to count)
    weighted_threshold: float = 10.0

    # Time window for counting failures (seconds)
    failure_window: float = 60.0

    # Time to wait before testing recovery (seconds)
    recovery_timeout: float = 30.0

    # Number of successful calls in half-open to close circuit
    success_threshold: int = 3

    # Timeout for individual calls (seconds)
    call_timeout: float = 30.0

    # Whether to use weighted failures
    use_weighted_failures: bool = True

    # Excluded exceptions (won't trigger circuit)
    excluded_exceptions: tuple[type[Exception], ...] = ()

    # Name for logging
    name: str = "default"


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Rejected when circuit is open
    timeout_requests: int = 0
    state_changes: int = 0

    # Current window stats
    window_failures: int = 0
    window_successes: int = 0
    weighted_failures: float = 0.0

    # Timing
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    last_state_change: datetime | None = None
    total_open_time_ms: float = 0.0

    # Consecutive counts
    consecutive_successes: int = 0
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "timeout_requests": self.timeout_requests,
            "success_rate": self.success_rate,
            "window_failures": self.window_failures,
            "weighted_failures": self.weighted_failures,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None,
        }

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


@dataclass
class FailureRecord:
    """Record of a failure event."""

    timestamp: datetime
    failure_type: FailureType
    weight: float
    exception_type: str
    message: str


class CircuitBreakerOpen(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, circuit_name: str, recovery_time: datetime):
        self.circuit_name = circuit_name
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. "
            f"Recovery attempt at {recovery_time.isoformat()}"
        )


class CircuitBreaker:
    """
    Circuit Breaker for protecting external service calls.

    Usage:
        cb = CircuitBreaker(CircuitBreakerConfig(name="llm"))

        # As decorator
        @cb.protect
        async def call_llm(prompt: str) -> str:
            ...

        # Or manually
        async with cb:
            result = await external_service()
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._failures: list[FailureRecord] = []
        self._half_open_successes = 0
        self._opened_at: datetime | None = None
        self._lock = asyncio.Lock()

        # State change callbacks
        self._state_callbacks: list[Callable[[CircuitState, CircuitState], None]] = []

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        return self._metrics

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    def on_state_change(
        self, callback: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    def _classify_exception(self, exc: Exception) -> FailureType:
        """Classify exception into failure type."""
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__.lower()

        # Check for excluded exceptions
        if isinstance(exc, self.config.excluded_exceptions):
            return None

        # Timeout
        if any(x in exc_str for x in ["timeout", "timed out", "deadline"]):
            return FailureType.TIMEOUT
        if "timeout" in exc_type:
            return FailureType.TIMEOUT

        # Connection errors
        if any(x in exc_str for x in [
            "connection", "network", "unreachable", "refused",
            "reset", "broken pipe", "dns"
        ]):
            return FailureType.CONNECTION

        # Auth errors
        if any(x in exc_str for x in [
            "auth", "api key", "unauthorized", "forbidden",
            "invalid_api_key", "access denied"
        ]):
            return FailureType.AUTH_ERROR

        # Rate limit
        if any(x in exc_str for x in ["rate limit", "429", "too many", "throttle"]):
            return FailureType.RATE_LIMIT

        # Server errors
        if any(x in exc_str for x in ["500", "502", "503", "504", "server error"]):
            return FailureType.SERVER_ERROR

        return FailureType.UNKNOWN

    def _clean_old_failures(self) -> None:
        """Remove failures outside the time window."""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.config.failure_window)

        self._failures = [f for f in self._failures if f.timestamp > cutoff]

        # Recalculate weighted failures
        self._metrics.weighted_failures = sum(f.weight for f in self._failures)
        self._metrics.window_failures = len(self._failures)

    def _should_trip(self) -> bool:
        """Determine if circuit should trip to OPEN."""
        self._clean_old_failures()

        if self.config.use_weighted_failures:
            return self._metrics.weighted_failures >= self.config.weighted_threshold
        else:
            return self._metrics.window_failures >= self.config.failure_threshold

    def _should_attempt_reset(self) -> bool:
        """Determine if we should try to reset the circuit."""
        if self._opened_at is None:
            return True

        recovery_time = self._opened_at + timedelta(
            seconds=self.config.recovery_timeout
        )
        return datetime.utcnow() >= recovery_time

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1
        self._metrics.last_state_change = datetime.utcnow()

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.utcnow()
            logger.warning(
                "Circuit breaker OPENED",
                name=self.config.name,
                failures=self._metrics.window_failures,
                weighted_failures=self._metrics.weighted_failures,
            )
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_successes = 0
            logger.info(
                "Circuit breaker HALF-OPEN (testing recovery)",
                name=self.config.name,
            )
        elif new_state == CircuitState.CLOSED:
            if self._opened_at:
                open_duration = (datetime.utcnow() - self._opened_at).total_seconds()
                self._metrics.total_open_time_ms += open_duration * 1000

            self._opened_at = None
            self._failures.clear()
            self._metrics.weighted_failures = 0
            self._metrics.window_failures = 0
            self._metrics.consecutive_failures = 0

            logger.info(
                "Circuit breaker CLOSED (recovered)",
                name=self.config.name,
            )

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(
                    "State change callback failed",
                    error=str(e),
                )

    async def _record_success(self) -> None:
        """Record a successful call."""
        self._metrics.total_requests += 1
        self._metrics.successful_requests += 1
        self._metrics.last_success_time = datetime.utcnow()
        self._metrics.consecutive_successes += 1
        self._metrics.consecutive_failures = 0

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            self._metrics.window_successes += 1

            if self._half_open_successes >= self.config.success_threshold:
                await self._transition_to(CircuitState.CLOSED)

    async def _record_failure(
        self,
        exc: Exception,
        failure_type: FailureType | None = None,
    ) -> None:
        """Record a failed call."""
        self._metrics.total_requests += 1
        self._metrics.failed_requests += 1
        self._metrics.last_failure_time = datetime.utcnow()
        self._metrics.consecutive_failures += 1
        self._metrics.consecutive_successes = 0

        if isinstance(exc, asyncio.TimeoutError):
            self._metrics.timeout_requests += 1

        # Classify failure if not provided
        if failure_type is None:
            failure_type = self._classify_exception(exc)

        # Skip excluded failures
        if failure_type is None:
            return

        # Record failure
        weight = FAILURE_WEIGHTS.get(failure_type, 1)
        record = FailureRecord(
            timestamp=datetime.utcnow(),
            failure_type=failure_type,
            weight=weight,
            exception_type=type(exc).__name__,
            message=str(exc)[:200],
        )
        self._failures.append(record)

        # Update metrics
        self._metrics.weighted_failures += weight
        self._metrics.window_failures += 1

        logger.debug(
            "Circuit breaker recorded failure",
            name=self.config.name,
            failure_type=failure_type.value,
            weight=weight,
            total_weighted=self._metrics.weighted_failures,
        )

        # Check if should trip
        if self._state == CircuitState.CLOSED and self._should_trip():
            await self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open returns to open
            await self._transition_to(CircuitState.OPEN)

    async def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns True if request can proceed, False otherwise.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            elif self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            elif self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open
                return True

        return False

    async def __aenter__(self):
        """Async context manager entry."""
        if not await self.allow_request():
            recovery_time = self._opened_at + timedelta(
                seconds=self.config.recovery_timeout
            ) if self._opened_at else datetime.utcnow()

            self._metrics.rejected_requests += 1
            raise CircuitBreakerOpen(self.config.name, recovery_time)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        async with self._lock:
            if exc_val is None:
                await self._record_success()
            else:
                await self._record_failure(exc_val)

        return False  # Don't suppress exception

    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to protect a function with circuit breaker.

        Usage:
            @circuit_breaker.protect
            async def call_service():
                ...
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.call_timeout,
                )

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failures.clear()
        self._opened_at = None
        self._half_open_successes = 0
        self._metrics.window_failures = 0
        self._metrics.weighted_failures = 0
        self._metrics.consecutive_failures = 0

        logger.info(
            "Circuit breaker manually reset",
            name=self.config.name,
        )

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.config.name,
            "state": self._state.value,
            "metrics": self._metrics.to_dict(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "weighted_threshold": self.config.weighted_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
            },
            "recovery_time": (
                (self._opened_at + timedelta(seconds=self.config.recovery_timeout)).isoformat()
                if self._opened_at else None
            ),
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to circuit breakers for different services.
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()

    def register(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        if name in self._breakers:
            return self._breakers[name]

        cfg = config or CircuitBreakerConfig(name=name)
        cfg.name = name
        breaker = CircuitBreaker(cfg)
        self._breakers[name] = breaker

        logger.debug("Registered circuit breaker", name=name)
        return breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name in self._breakers:
            return self._breakers[name]
        return self.register(name, config)

    def get_all_status(self) -> dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

        logger.info("All circuit breakers reset")


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from global registry."""
    return _registry.get_or_create(name, config)


def get_all_circuit_breakers_status() -> dict[str, Any]:
    """Get status of all registered circuit breakers."""
    return _registry.get_all_status()


# Pre-configured circuit breakers for common services
LLM_CIRCUIT_CONFIG = CircuitBreakerConfig(
    name="llm",
    failure_threshold=5,
    weighted_threshold=10.0,
    failure_window=60.0,
    recovery_timeout=30.0,
    success_threshold=2,
    call_timeout=60.0,
)

NEO4J_CIRCUIT_CONFIG = CircuitBreakerConfig(
    name="neo4j",
    failure_threshold=3,
    weighted_threshold=6.0,
    failure_window=30.0,
    recovery_timeout=15.0,
    success_threshold=2,
    call_timeout=30.0,
)

EMBEDDING_CIRCUIT_CONFIG = CircuitBreakerConfig(
    name="embedding",
    failure_threshold=5,
    weighted_threshold=8.0,
    failure_window=60.0,
    recovery_timeout=20.0,
    success_threshold=3,
    call_timeout=30.0,
)


def get_llm_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for LLM calls."""
    return get_circuit_breaker("llm", LLM_CIRCUIT_CONFIG)


def get_neo4j_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Neo4j calls."""
    return get_circuit_breaker("neo4j", NEO4J_CIRCUIT_CONFIG)


def get_embedding_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for embedding calls."""
    return get_circuit_breaker("embedding", EMBEDDING_CIRCUIT_CONFIG)

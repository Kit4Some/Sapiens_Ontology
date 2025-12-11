"""
Retry Handler with Exponential Backoff.

Provides robust retry mechanisms for pipeline operations:
- Configurable retry attempts and delays
- Exponential backoff with jitter
- Retryable vs non-retryable error classification
- Async and sync support
"""

import asyncio
import random
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Error that can be retried."""

    pass


class NonRetryableError(Exception):
    """Error that should not be retried."""

    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0      # seconds
    max_delay: float = 60.0         # seconds
    exponential_base: float = 2.0
    jitter: float = 0.1             # 10% jitter

    # Error classification
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
        RetryableError,
    )
    non_retryable_exceptions: tuple[type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
        NonRetryableError,
    )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        # Explicitly non-retryable
        if isinstance(error, self.non_retryable_exceptions):
            return False

        # Explicitly retryable
        if isinstance(error, self.retryable_exceptions):
            return True

        # Check error message for common retryable patterns
        error_str = str(error).lower()
        retryable_patterns = [
            "timeout", "timed out",
            "connection", "network",
            "rate limit", "429",
            "503", "502", "504",
            "temporary", "retry",
            "unavailable",
        ]

        for pattern in retryable_patterns:
            if pattern in error_str:
                return True

        return False


class RetryHandler:
    """
    Handles retry logic with exponential backoff.

    Usage:
        handler = RetryHandler(RetryConfig(max_retries=3))

        @handler.wrap
        async def my_function():
            ...

        # Or manually
        result = await handler.execute(my_function, arg1, arg2)
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self._total_retries = 0
        self._total_failures = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_retries": self._total_retries,
            "total_failures": self._total_failures,
        }

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        on_retry: Callable[[int, Exception], None] | None = None,
        **kwargs,
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute (can be async or sync)
            *args: Positional arguments for func
            on_retry: Optional callback called on each retry
            **kwargs: Keyword arguments for func

        Returns:
            Result of the function

        Raises:
            Last exception if all retries exhausted
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Handle both async and sync functions
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success - reset consecutive failures
                return result

            except Exception as e:
                last_error = e

                # Check if retryable
                if not self.config.is_retryable(e):
                    logger.warning(
                        "Non-retryable error, failing immediately",
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    self._total_failures += 1
                    raise

                # Check if we have retries left
                if attempt >= self.config.max_retries:
                    logger.error(
                        "All retries exhausted",
                        attempt=attempt + 1,
                        max_retries=self.config.max_retries,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    self._total_failures += 1
                    raise

                # Calculate delay
                delay = self.config.get_delay(attempt)
                self._total_retries += 1

                logger.warning(
                    "Retry scheduled",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    delay_seconds=f"{delay:.2f}",
                    error_type=type(e).__name__,
                    error=str(e)[:100],
                )

                # Call retry callback if provided
                if on_retry:
                    try:
                        on_retry(attempt + 1, e)
                    except Exception:
                        pass

                # Wait before retry
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Retry loop completed without result or error")

    def wrap(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to wrap a function with retry logic.

        Usage:
            @handler.wrap
            async def my_function():
                ...
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory for retry with exponential backoff.

    Usage:
        @with_retry(max_retries=3, initial_delay=1.0)
        async def my_function():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
    )
    handler = RetryHandler(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        return handler.wrap(func)

    return decorator


class RetryContext:
    """
    Context manager for retry operations.

    Usage:
        async with RetryContext(config) as ctx:
            for attempt in ctx.attempts():
                try:
                    result = await do_something()
                    ctx.success()
                    break
                except Exception as e:
                    if not ctx.should_retry(e):
                        raise
                    await ctx.wait()
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self._attempt = 0
        self._succeeded = False
        self._last_error: Exception | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and not self._succeeded:
            logger.error(
                "Retry context exited with error",
                attempts=self._attempt,
                error_type=exc_type.__name__ if exc_type else None,
            )
        return False

    def attempts(self):
        """Generator for attempt numbers."""
        while self._attempt <= self.config.max_retries:
            yield self._attempt
            self._attempt += 1

    def should_retry(self, error: Exception) -> bool:
        """Check if should retry given the error."""
        self._last_error = error

        if not self.config.is_retryable(error):
            return False

        if self._attempt >= self.config.max_retries:
            return False

        return True

    async def wait(self) -> None:
        """Wait before next retry attempt."""
        delay = self.config.get_delay(self._attempt)
        await asyncio.sleep(delay)

    def success(self) -> None:
        """Mark operation as successful."""
        self._succeeded = True


# Pre-configured retry handlers for common scenarios
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    retryable_exceptions=(
        TimeoutError,
        ConnectionError,
        RetryableError,
    ),
)

NEO4J_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
)

EMBEDDING_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)

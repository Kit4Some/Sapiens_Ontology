"""
SDDI Pipeline Reliability Module.

Provides production-grade reliability features:
- Dead-Letter Queue (DLQ) for failed documents
- Retry mechanisms with exponential backoff
- Per-document latency tracking
- Document size validation and limits
"""

from src.sddi.reliability.dead_letter_queue import (
    DeadLetterQueue,
    DLQEntry,
    DLQStatus,
    InMemoryDLQ,
    FileDLQ,
)
from src.sddi.reliability.retry_handler import (
    RetryConfig,
    RetryHandler,
    RetryableError,
    NonRetryableError,
    with_retry,
)
from src.sddi.reliability.metrics import (
    DocumentMetrics,
    PipelineMetricsCollector,
    LatencyStats,
    StepTimer,
)
from src.sddi.reliability.validation import (
    DocumentValidator,
    ValidationConfig,
    ValidationResult,
    SizeExceededError,
)

__all__ = [
    # Dead-Letter Queue
    "DeadLetterQueue",
    "DLQEntry",
    "DLQStatus",
    "InMemoryDLQ",
    "FileDLQ",
    # Retry
    "RetryConfig",
    "RetryHandler",
    "RetryableError",
    "NonRetryableError",
    "with_retry",
    # Metrics
    "DocumentMetrics",
    "PipelineMetricsCollector",
    "LatencyStats",
    "StepTimer",
    # Validation
    "DocumentValidator",
    "ValidationConfig",
    "ValidationResult",
    "SizeExceededError",
]

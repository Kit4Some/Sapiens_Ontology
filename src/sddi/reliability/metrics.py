"""
Pipeline Metrics Collection.

Provides detailed per-document and per-step metrics:
- Latency tracking at document and step level
- Success/failure rates
- Throughput calculation
- Resource usage (tokens, embeddings)
"""

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LatencyStats:
    """Statistical summary of latencies."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_dev_ms: float = 0.0

    _values: list[float] = field(default_factory=list, repr=False)

    def add(self, latency_ms: float) -> None:
        """Add a latency measurement."""
        self._values.append(latency_ms)
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)

    def compute(self) -> None:
        """Compute statistical summaries."""
        if not self._values:
            return

        self.mean_ms = self.total_ms / self.count

        sorted_values = sorted(self._values)
        self.median_ms = statistics.median(sorted_values)

        # Percentiles
        n = len(sorted_values)
        self.p95_ms = sorted_values[int(n * 0.95)] if n >= 20 else self.max_ms
        self.p99_ms = sorted_values[int(n * 0.99)] if n >= 100 else self.max_ms

        # Standard deviation
        if len(self._values) > 1:
            self.std_dev_ms = statistics.stdev(self._values)

    def to_dict(self) -> dict[str, Any]:
        self.compute()
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else 0,
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
        }


@dataclass
class DocumentMetrics:
    """Metrics for a single document."""

    document_id: str
    pipeline_id: str
    source: str = ""

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    total_latency_ms: float = 0.0

    # Per-step latencies (ms)
    step_latencies: dict[str, float] = field(default_factory=dict)

    # Counts
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    embedding_count: int = 0

    # Sizes
    content_size_bytes: int = 0
    content_size_chars: int = 0

    # Resource usage
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    embedding_dimensions: int = 0

    # Status
    success: bool = False
    error_message: str | None = None
    failed_step: str | None = None

    def record_step(self, step: str, latency_ms: float) -> None:
        """Record latency for a specific step."""
        self.step_latencies[step] = latency_ms

    def finish(self, success: bool = True, error: str | None = None) -> None:
        """Mark document processing as finished."""
        self.end_time = datetime.utcnow()
        self.total_latency_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pipeline_id": self.pipeline_id,
            "source": self.source,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "step_latencies": {k: round(v, 2) for k, v in self.step_latencies.items()},
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "embedding_count": self.embedding_count,
            "content_size_bytes": self.content_size_bytes,
            "content_size_chars": self.content_size_chars,
            "llm_tokens_input": self.llm_tokens_input,
            "llm_tokens_output": self.llm_tokens_output,
            "success": self.success,
            "error_message": self.error_message,
            "failed_step": self.failed_step,
        }


class PipelineMetricsCollector:
    """
    Collects and aggregates pipeline metrics.

    Provides:
    - Per-document metrics tracking
    - Per-step latency aggregation
    - Success/failure rates
    - Throughput calculation
    """

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.start_time = datetime.utcnow()
        self.end_time: datetime | None = None

        # Document metrics
        self._documents: dict[str, DocumentMetrics] = {}

        # Aggregate latencies by step
        self._step_latencies: dict[str, LatencyStats] = {}

        # Overall latency
        self._document_latencies = LatencyStats()

        # Counters
        self._total_documents = 0
        self._successful_documents = 0
        self._failed_documents = 0
        self._total_chunks = 0
        self._total_entities = 0
        self._total_relations = 0
        self._total_embeddings = 0

        # Size tracking
        self._total_bytes = 0
        self._total_chars = 0

        # Token tracking
        self._total_llm_tokens_input = 0
        self._total_llm_tokens_output = 0

    def start_document(
        self,
        document_id: str,
        source: str = "",
        content_size_bytes: int = 0,
        content_size_chars: int = 0,
    ) -> DocumentMetrics:
        """Start tracking a document."""
        metrics = DocumentMetrics(
            document_id=document_id,
            pipeline_id=self.pipeline_id,
            source=source,
            content_size_bytes=content_size_bytes,
            content_size_chars=content_size_chars,
        )
        self._documents[document_id] = metrics
        self._total_documents += 1
        self._total_bytes += content_size_bytes
        self._total_chars += content_size_chars

        return metrics

    def get_document(self, document_id: str) -> DocumentMetrics | None:
        """Get metrics for a document."""
        return self._documents.get(document_id)

    def record_step_latency(
        self,
        document_id: str,
        step: str,
        latency_ms: float,
    ) -> None:
        """Record latency for a specific step."""
        # Document-level
        if document_id in self._documents:
            self._documents[document_id].record_step(step, latency_ms)

        # Aggregate by step
        if step not in self._step_latencies:
            self._step_latencies[step] = LatencyStats()
        self._step_latencies[step].add(latency_ms)

    def finish_document(
        self,
        document_id: str,
        success: bool = True,
        error: str | None = None,
        failed_step: str | None = None,
        chunk_count: int = 0,
        entity_count: int = 0,
        relation_count: int = 0,
        embedding_count: int = 0,
        llm_tokens_input: int = 0,
        llm_tokens_output: int = 0,
    ) -> DocumentMetrics | None:
        """Finish tracking a document."""
        if document_id not in self._documents:
            return None

        metrics = self._documents[document_id]
        metrics.finish(success, error)
        metrics.failed_step = failed_step
        metrics.chunk_count = chunk_count
        metrics.entity_count = entity_count
        metrics.relation_count = relation_count
        metrics.embedding_count = embedding_count
        metrics.llm_tokens_input = llm_tokens_input
        metrics.llm_tokens_output = llm_tokens_output

        # Update aggregates
        self._document_latencies.add(metrics.total_latency_ms)

        if success:
            self._successful_documents += 1
        else:
            self._failed_documents += 1

        self._total_chunks += chunk_count
        self._total_entities += entity_count
        self._total_relations += relation_count
        self._total_embeddings += embedding_count
        self._total_llm_tokens_input += llm_tokens_input
        self._total_llm_tokens_output += llm_tokens_output

        return metrics

    def finish_pipeline(self) -> None:
        """Mark pipeline as finished."""
        self.end_time = datetime.utcnow()

        logger.info(
            "Pipeline metrics finalized",
            pipeline_id=self.pipeline_id,
            total_documents=self._total_documents,
            successful=self._successful_documents,
            failed=self._failed_documents,
            duration_seconds=(self.end_time - self.start_time).total_seconds(),
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        self._document_latencies.compute()
        for stats in self._step_latencies.values():
            stats.compute()

        duration_seconds = 0.0
        if self.end_time:
            duration_seconds = (self.end_time - self.start_time).total_seconds()
        else:
            duration_seconds = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "pipeline_id": self.pipeline_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(duration_seconds, 2),

            # Document counts
            "documents": {
                "total": self._total_documents,
                "successful": self._successful_documents,
                "failed": self._failed_documents,
                "success_rate": round(
                    self._successful_documents / max(self._total_documents, 1), 4
                ),
            },

            # Output counts
            "outputs": {
                "chunks": self._total_chunks,
                "entities": self._total_entities,
                "relations": self._total_relations,
                "embeddings": self._total_embeddings,
            },

            # Size metrics
            "sizes": {
                "total_bytes": self._total_bytes,
                "total_chars": self._total_chars,
                "avg_bytes_per_doc": round(
                    self._total_bytes / max(self._total_documents, 1), 2
                ),
            },

            # Token usage
            "tokens": {
                "llm_input": self._total_llm_tokens_input,
                "llm_output": self._total_llm_tokens_output,
                "total": self._total_llm_tokens_input + self._total_llm_tokens_output,
            },

            # Throughput
            "throughput": {
                "docs_per_second": round(
                    self._total_documents / max(duration_seconds, 0.001), 4
                ),
                "chunks_per_second": round(
                    self._total_chunks / max(duration_seconds, 0.001), 4
                ),
                "chars_per_second": round(
                    self._total_chars / max(duration_seconds, 0.001), 2
                ),
            },

            # Latency statistics
            "latency": {
                "document": self._document_latencies.to_dict(),
                "by_step": {
                    step: stats.to_dict()
                    for step, stats in self._step_latencies.items()
                },
            },
        }

    def get_all_document_metrics(self) -> list[dict[str, Any]]:
        """Get metrics for all documents."""
        return [m.to_dict() for m in self._documents.values()]

    def get_failed_documents(self) -> list[dict[str, Any]]:
        """Get metrics for failed documents only."""
        return [
            m.to_dict()
            for m in self._documents.values()
            if not m.success
        ]


class StepTimer:
    """
    Context manager for timing pipeline steps.

    Usage:
        async with StepTimer(collector, doc_id, "extract_entities") as timer:
            # Do work
            pass
        # Latency automatically recorded
    """

    def __init__(
        self,
        collector: PipelineMetricsCollector,
        document_id: str,
        step_name: str,
    ):
        self.collector = collector
        self.document_id = document_id
        self.step_name = step_name
        self._start_time: float = 0

    async def __aenter__(self):
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self._start_time) * 1000
        self.collector.record_step_latency(
            self.document_id,
            self.step_name,
            latency_ms,
        )
        return False

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self._start_time) * 1000
        self.collector.record_step_latency(
            self.document_id,
            self.step_name,
            latency_ms,
        )
        return False

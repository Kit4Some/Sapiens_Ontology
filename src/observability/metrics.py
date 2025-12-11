"""
Metrics Collection and Export.

Provides comprehensive metrics for the Ontology Reasoning System:
- Query latency tracking (p50, p95, p99)
- MACER iteration metrics
- Component-level error rates
- Prometheus-compatible export
"""

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ComponentName(str, Enum):
    """Component names for error tracking."""
    API = "api"
    CONSTRUCTOR = "constructor"
    RETRIEVER = "retriever"
    REFLECTOR = "reflector"
    RESPONSER = "responser"
    NEO4J = "neo4j"
    LLM = "llm"
    EMBEDDING = "embedding"
    INGESTION = "ingestion"


@dataclass
class HistogramBuckets:
    """Configurable histogram buckets."""
    # Latency buckets in milliseconds
    LATENCY_MS: list[float] = field(default_factory=lambda: [
        10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000
    ])
    # Iteration count buckets
    ITERATIONS: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7, 10, 15, 20])
    # Confidence score buckets
    CONFIDENCE: list[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
    ])


@dataclass
class PercentileStats:
    """Statistical percentiles for a metric."""
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    mean: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    _values: list[float] = field(default_factory=list, repr=False)

    def add(self, value: float) -> None:
        """Add a value to the distribution."""
        self._values.append(value)
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def compute(self) -> None:
        """Compute percentiles."""
        if not self._values:
            return

        self.mean = self.sum / self.count
        sorted_values = sorted(self._values)
        n = len(sorted_values)

        self.p50 = sorted_values[int(n * 0.50)]
        self.p75 = sorted_values[int(n * 0.75)]
        self.p90 = sorted_values[int(n * 0.90)]
        self.p95 = sorted_values[min(int(n * 0.95), n - 1)]
        self.p99 = sorted_values[min(int(n * 0.99), n - 1)]

    def to_dict(self) -> dict[str, Any]:
        self.compute()
        return {
            "count": self.count,
            "sum": round(self.sum, 2),
            "min": round(self.min, 2) if self.min != float("inf") else 0,
            "max": round(self.max, 2),
            "mean": round(self.mean, 2),
            "p50": round(self.p50, 2),
            "p75": round(self.p75, 2),
            "p90": round(self.p90, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    correlation_id: str | None = None
    query_text: str = ""

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    total_latency_ms: float = 0.0

    # MACER metrics
    macer_iterations: int = 0
    constructor_latency_ms: float = 0.0
    retriever_latency_ms: float = 0.0
    reflector_latency_ms: float = 0.0
    responser_latency_ms: float = 0.0

    # Results
    confidence_score: float = 0.0
    evidence_count: int = 0
    subgraph_nodes: int = 0
    subgraph_edges: int = 0

    # Status
    success: bool = False
    error_message: str | None = None
    error_component: str | None = None

    def finish(
        self,
        success: bool = True,
        error: str | None = None,
        error_component: str | None = None,
    ) -> None:
        """Mark query as finished."""
        self.end_time = datetime.utcnow()
        self.total_latency_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error
        self.error_component = error_component

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "correlation_id": self.correlation_id,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "macer_iterations": self.macer_iterations,
            "agent_latencies": {
                "constructor_ms": round(self.constructor_latency_ms, 2),
                "retriever_ms": round(self.retriever_latency_ms, 2),
                "reflector_ms": round(self.reflector_latency_ms, 2),
                "responser_ms": round(self.responser_latency_ms, 2),
            },
            "confidence_score": round(self.confidence_score, 3),
            "evidence_count": self.evidence_count,
            "subgraph_size": {
                "nodes": self.subgraph_nodes,
                "edges": self.subgraph_edges,
            },
            "success": self.success,
            "error_message": self.error_message,
            "error_component": self.error_component,
        }


class QueryMetricsCollector:
    """
    Collects and aggregates query metrics.

    Provides:
    - Per-query tracking
    - Latency percentiles (p50, p95, p99)
    - MACER iteration distribution
    - Error rate by component
    - Confidence score distribution
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._queries: dict[str, QueryMetrics] = {}
        self._completed_queries: list[QueryMetrics] = []

        # Aggregate stats
        self._latency_stats = PercentileStats()
        self._confidence_stats = PercentileStats()
        self._iteration_stats = PercentileStats()

        # Per-agent latencies
        self._agent_latencies: dict[str, PercentileStats] = {
            "constructor": PercentileStats(),
            "retriever": PercentileStats(),
            "reflector": PercentileStats(),
            "responser": PercentileStats(),
        }

        # Error tracking
        self._total_queries = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._errors_by_component: dict[str, int] = defaultdict(int)

        # Time window tracking
        self._window_start = datetime.utcnow()

    def start_query(
        self,
        query_id: str,
        query_text: str = "",
        correlation_id: str | None = None,
    ) -> QueryMetrics:
        """Start tracking a query."""
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=query_text[:100],
            correlation_id=correlation_id,
        )
        self._queries[query_id] = metrics
        return metrics

    def get_query(self, query_id: str) -> QueryMetrics | None:
        """Get metrics for a query."""
        return self._queries.get(query_id)

    def record_agent_latency(
        self,
        query_id: str,
        agent: str,
        latency_ms: float,
    ) -> None:
        """Record latency for a specific agent."""
        if query_id not in self._queries:
            return

        metrics = self._queries[query_id]
        agent_lower = agent.lower()

        if agent_lower == "constructor":
            metrics.constructor_latency_ms = latency_ms
        elif agent_lower == "retriever":
            metrics.retriever_latency_ms += latency_ms  # Accumulate for iterations
        elif agent_lower == "reflector":
            metrics.reflector_latency_ms += latency_ms  # Accumulate for iterations
        elif agent_lower == "responser":
            metrics.responser_latency_ms = latency_ms

        # Add to agent stats
        if agent_lower in self._agent_latencies:
            self._agent_latencies[agent_lower].add(latency_ms)

    def record_macer_iteration(self, query_id: str) -> None:
        """Record a MACER iteration for a query."""
        if query_id in self._queries:
            self._queries[query_id].macer_iterations += 1

    def record_results(
        self,
        query_id: str,
        confidence: float,
        evidence_count: int,
        subgraph_nodes: int = 0,
        subgraph_edges: int = 0,
    ) -> None:
        """Record query results."""
        if query_id not in self._queries:
            return

        metrics = self._queries[query_id]
        metrics.confidence_score = confidence
        metrics.evidence_count = evidence_count
        metrics.subgraph_nodes = subgraph_nodes
        metrics.subgraph_edges = subgraph_edges

    def finish_query(
        self,
        query_id: str,
        success: bool = True,
        error: str | None = None,
        error_component: str | None = None,
    ) -> QueryMetrics | None:
        """Finish tracking a query."""
        if query_id not in self._queries:
            return None

        metrics = self._queries.pop(query_id)
        metrics.finish(success, error, error_component)

        # Update aggregates
        self._total_queries += 1
        if success:
            self._successful_queries += 1
        else:
            self._failed_queries += 1
            if error_component:
                self._errors_by_component[error_component] += 1

        # Update stats
        self._latency_stats.add(metrics.total_latency_ms)
        self._confidence_stats.add(metrics.confidence_score)
        self._iteration_stats.add(metrics.macer_iterations)

        # Store in history
        self._completed_queries.append(metrics)
        if len(self._completed_queries) > self.max_history:
            self._completed_queries.pop(0)

        logger.info(
            "Query completed",
            query_id=query_id,
            latency_ms=round(metrics.total_latency_ms, 2),
            iterations=metrics.macer_iterations,
            confidence=round(metrics.confidence_score, 3),
            success=success,
        )

        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all query metrics."""
        duration_seconds = (datetime.utcnow() - self._window_start).total_seconds()

        return {
            "window_start": self._window_start.isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "queries": {
                "total": self._total_queries,
                "successful": self._successful_queries,
                "failed": self._failed_queries,
                "success_rate": round(
                    self._successful_queries / max(self._total_queries, 1), 4
                ),
                "in_progress": len(self._queries),
            },
            "throughput": {
                "queries_per_second": round(
                    self._total_queries / max(duration_seconds, 1), 4
                ),
                "queries_per_minute": round(
                    self._total_queries / max(duration_seconds / 60, 1), 2
                ),
            },
            "latency": self._latency_stats.to_dict(),
            "confidence": self._confidence_stats.to_dict(),
            "macer_iterations": self._iteration_stats.to_dict(),
            "agent_latencies": {
                agent: stats.to_dict()
                for agent, stats in self._agent_latencies.items()
            },
            "errors_by_component": dict(self._errors_by_component),
        }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Query counters
        lines.append("# HELP ontology_queries_total Total number of queries processed")
        lines.append("# TYPE ontology_queries_total counter")
        lines.append(f"ontology_queries_total{{status=\"success\"}} {self._successful_queries}")
        lines.append(f"ontology_queries_total{{status=\"failed\"}} {self._failed_queries}")
        lines.append("")

        # Query latency histogram
        lines.append("# HELP ontology_query_latency_ms Query latency in milliseconds")
        lines.append("# TYPE ontology_query_latency_ms summary")
        latency = self._latency_stats
        latency.compute()
        lines.append(f'ontology_query_latency_ms{{quantile="0.5"}} {latency.p50}')
        lines.append(f'ontology_query_latency_ms{{quantile="0.9"}} {latency.p90}')
        lines.append(f'ontology_query_latency_ms{{quantile="0.95"}} {latency.p95}')
        lines.append(f'ontology_query_latency_ms{{quantile="0.99"}} {latency.p99}')
        lines.append(f"ontology_query_latency_ms_sum {latency.sum}")
        lines.append(f"ontology_query_latency_ms_count {latency.count}")
        lines.append("")

        # MACER iterations histogram
        lines.append("# HELP ontology_macer_iterations MACER iterations per query")
        lines.append("# TYPE ontology_macer_iterations summary")
        iterations = self._iteration_stats
        iterations.compute()
        lines.append(f'ontology_macer_iterations{{quantile="0.5"}} {iterations.p50}')
        lines.append(f'ontology_macer_iterations{{quantile="0.95"}} {iterations.p95}')
        lines.append(f"ontology_macer_iterations_sum {iterations.sum}")
        lines.append(f"ontology_macer_iterations_count {iterations.count}")
        lines.append("")

        # Confidence score histogram
        lines.append("# HELP ontology_confidence_score Answer confidence score")
        lines.append("# TYPE ontology_confidence_score summary")
        confidence = self._confidence_stats
        confidence.compute()
        lines.append(f'ontology_confidence_score{{quantile="0.5"}} {confidence.p50}')
        lines.append(f'ontology_confidence_score{{quantile="0.95"}} {confidence.p95}')
        lines.append(f"ontology_confidence_score_sum {confidence.sum}")
        lines.append(f"ontology_confidence_score_count {confidence.count}")
        lines.append("")

        # Per-agent latencies
        for agent, stats in self._agent_latencies.items():
            stats.compute()
            lines.append(f"# HELP ontology_agent_{agent}_latency_ms {agent.title()} agent latency")
            lines.append(f"# TYPE ontology_agent_{agent}_latency_ms summary")
            lines.append(f'ontology_agent_{agent}_latency_ms{{quantile="0.5"}} {stats.p50}')
            lines.append(f'ontology_agent_{agent}_latency_ms{{quantile="0.95"}} {stats.p95}')
            lines.append(f"ontology_agent_{agent}_latency_ms_count {stats.count}")
            lines.append("")

        # Error rate by component
        lines.append("# HELP ontology_errors_total Errors by component")
        lines.append("# TYPE ontology_errors_total counter")
        for component, count in self._errors_by_component.items():
            lines.append(f'ontology_errors_total{{component="{component}"}} {count}')
        lines.append("")

        # Active queries gauge
        lines.append("# HELP ontology_queries_in_progress Queries currently in progress")
        lines.append("# TYPE ontology_queries_in_progress gauge")
        lines.append(f"ontology_queries_in_progress {len(self._queries)}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        self._queries.clear()
        self._completed_queries.clear()
        self._latency_stats = PercentileStats()
        self._confidence_stats = PercentileStats()
        self._iteration_stats = PercentileStats()
        self._agent_latencies = {
            "constructor": PercentileStats(),
            "retriever": PercentileStats(),
            "reflector": PercentileStats(),
            "responser": PercentileStats(),
        }
        self._total_queries = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._errors_by_component.clear()
        self._window_start = datetime.utcnow()

        logger.info("Query metrics reset")


class MetricsRegistry:
    """
    Central registry for all metrics collectors.

    Provides unified access to:
    - Query metrics
    - Ingestion metrics
    - System metrics
    """

    def __init__(self):
        self.query_metrics = QueryMetricsCollector()
        self._custom_counters: dict[str, int] = defaultdict(int)
        self._custom_gauges: dict[str, float] = {}
        self._start_time = datetime.utcnow()

    def increment_counter(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a custom counter."""
        key = self._make_key(name, labels)
        self._custom_counters[key] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a custom gauge value."""
        key = self._make_key(name, labels)
        self._custom_gauges[key] = value

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "uptime_seconds": round(uptime, 2),
            "query_metrics": self.query_metrics.get_summary(),
            "custom_counters": dict(self._custom_counters),
            "custom_gauges": dict(self._custom_gauges),
        }

    def get_prometheus_output(self) -> str:
        """Get all metrics in Prometheus format."""
        lines = []

        # Uptime
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        lines.append("# HELP ontology_uptime_seconds Time since service start")
        lines.append("# TYPE ontology_uptime_seconds gauge")
        lines.append(f"ontology_uptime_seconds {uptime}")
        lines.append("")

        # Query metrics
        lines.append(self.query_metrics.get_prometheus_metrics())

        # Custom counters
        for key, value in self._custom_counters.items():
            lines.append(f"ontology_custom_counter_{key} {value}")

        # Custom gauges
        for key, value in self._custom_gauges.items():
            lines.append(f"ontology_custom_gauge_{key} {value}")

        return "\n".join(lines)


# Global metrics registry
_metrics_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


# Convenience functions
def record_query_latency(query_id: str, latency_ms: float) -> None:
    """Record query latency."""
    registry = get_metrics_registry()
    if query_id in registry.query_metrics._queries:
        registry.query_metrics._queries[query_id].total_latency_ms = latency_ms


def record_macer_iteration(query_id: str) -> None:
    """Record a MACER iteration."""
    registry = get_metrics_registry()
    registry.query_metrics.record_macer_iteration(query_id)


class QueryTimer:
    """
    Context manager for timing queries.

    Usage:
        async with QueryTimer(query_id, "constructor") as timer:
            # Do work
            pass
    """

    def __init__(
        self,
        query_id: str,
        agent: str | None = None,
    ):
        self.query_id = query_id
        self.agent = agent
        self._start_time: float = 0

    async def __aenter__(self):
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self._start_time) * 1000

        registry = get_metrics_registry()
        if self.agent:
            registry.query_metrics.record_agent_latency(
                self.query_id, self.agent, latency_ms
            )

        return False

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self._start_time) * 1000

        registry = get_metrics_registry()
        if self.agent:
            registry.query_metrics.record_agent_latency(
                self.query_id, self.agent, latency_ms
            )

        return False

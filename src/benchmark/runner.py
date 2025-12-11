"""
Benchmark Runner.

Executes benchmark scenarios and collects metrics.
"""

import asyncio
import json
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # General settings
    name: str = "benchmark"
    description: str = ""
    warmup_iterations: int = 3
    iterations: int = 10

    # Concurrency settings
    concurrent_users: int = 1
    ramp_up_seconds: int = 0

    # Timing
    timeout_seconds: int = 300

    # Output
    output_dir: str = "./benchmark_results"
    save_raw_results: bool = True


@dataclass
class LatencyStats:
    """Latency statistics."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p75_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_dev_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "LatencyStats":
        """Calculate statistics from latency samples."""
        if not samples_ms:
            return cls()

        sorted_samples = sorted(samples_ms)
        n = len(sorted_samples)

        def percentile(p: float) -> float:
            k = (n - 1) * (p / 100)
            f = int(k)
            c = f + 1 if f < n - 1 else f
            return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])

        return cls(
            min_ms=min(sorted_samples),
            max_ms=max(sorted_samples),
            mean_ms=statistics.mean(sorted_samples),
            median_ms=statistics.median(sorted_samples),
            p50_ms=percentile(50),
            p75_ms=percentile(75),
            p90_ms=percentile(90),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            std_dev_ms=statistics.stdev(sorted_samples) if n > 1 else 0.0,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p75_ms": round(self.p75_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    description: str
    started_at: datetime
    completed_at: datetime | None = None

    # Counts
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0

    # Latency
    latency_samples_ms: list[float] = field(default_factory=list)
    latency_stats: LatencyStats | None = None

    # Throughput
    throughput_ops_per_sec: float = 0.0

    # Custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: list[str] = field(default_factory=list)

    def calculate_stats(self) -> None:
        """Calculate statistics from collected samples."""
        if self.latency_samples_ms:
            self.latency_stats = LatencyStats.from_samples(self.latency_samples_ms)

        if self.completed_at and self.started_at:
            duration_seconds = (self.completed_at - self.started_at).total_seconds()
            if duration_seconds > 0:
                self.throughput_ops_per_sec = self.successful_iterations / duration_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            ),
            "iterations": {
                "total": self.total_iterations,
                "successful": self.successful_iterations,
                "failed": self.failed_iterations,
                "success_rate": (
                    self.successful_iterations / self.total_iterations * 100
                    if self.total_iterations > 0 else 0
                ),
            },
            "latency": self.latency_stats.to_dict() if self.latency_stats else None,
            "throughput_ops_per_sec": round(self.throughput_ops_per_sec, 2),
            "custom_metrics": self.custom_metrics,
            "errors": self.errors[:10],  # Limit errors in output
        }


class BenchmarkScenario(ABC):
    """Abstract base class for benchmark scenarios."""

    name: str = "scenario"
    description: str = ""

    @abstractmethod
    async def setup(self) -> None:
        """Setup before benchmark runs."""
        pass

    @abstractmethod
    async def run_iteration(self) -> tuple[bool, float, dict[str, Any]]:
        """
        Run a single benchmark iteration.

        Returns:
            Tuple of (success, latency_ms, custom_metrics)
        """
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Cleanup after benchmark runs."""
        pass


class BenchmarkRunner:
    """
    Runs benchmark scenarios and collects metrics.

    Features:
    - Warmup iterations
    - Concurrent execution
    - Progress tracking
    - Result persistence
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._results: list[BenchmarkResult] = []

    async def run_scenario(
        self,
        scenario: BenchmarkScenario,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        """
        Run a benchmark scenario.

        Args:
            scenario: The scenario to run
            progress_callback: Optional callback for progress updates

        Returns:
            Benchmark result with collected metrics
        """
        result = BenchmarkResult(
            name=scenario.name,
            description=scenario.description,
            started_at=datetime.utcnow(),
        )

        logger.info(
            "Starting benchmark",
            scenario=scenario.name,
            iterations=self.config.iterations,
            concurrent_users=self.config.concurrent_users,
        )

        try:
            # Setup
            await scenario.setup()

            # Warmup
            if self.config.warmup_iterations > 0:
                logger.info("Running warmup iterations", count=self.config.warmup_iterations)
                for i in range(self.config.warmup_iterations):
                    try:
                        await asyncio.wait_for(
                            scenario.run_iteration(),
                            timeout=self.config.timeout_seconds,
                        )
                    except Exception as e:
                        logger.warning("Warmup iteration failed", error=str(e))

            # Run iterations
            if self.config.concurrent_users > 1:
                await self._run_concurrent(scenario, result, progress_callback)
            else:
                await self._run_sequential(scenario, result, progress_callback)

            # Calculate statistics
            result.completed_at = datetime.utcnow()
            result.calculate_stats()

            logger.info(
                "Benchmark completed",
                scenario=scenario.name,
                success_rate=f"{result.successful_iterations / result.total_iterations * 100:.1f}%",
                mean_latency_ms=result.latency_stats.mean_ms if result.latency_stats else 0,
                throughput=f"{result.throughput_ops_per_sec:.2f} ops/sec",
            )

        except Exception as e:
            result.errors.append(f"Benchmark failed: {str(e)}")
            result.completed_at = datetime.utcnow()
            logger.error("Benchmark failed", error=str(e))

        finally:
            # Teardown
            try:
                await scenario.teardown()
            except Exception as e:
                logger.warning("Teardown failed", error=str(e))

        # Save results
        self._results.append(result)
        if self.config.save_raw_results:
            self._save_result(result)

        return result

    async def _run_sequential(
        self,
        scenario: BenchmarkScenario,
        result: BenchmarkResult,
        progress_callback: Callable[[int, int], None] | None,
    ) -> None:
        """Run iterations sequentially."""
        for i in range(self.config.iterations):
            result.total_iterations += 1

            try:
                success, latency_ms, custom_metrics = await asyncio.wait_for(
                    scenario.run_iteration(),
                    timeout=self.config.timeout_seconds,
                )

                if success:
                    result.successful_iterations += 1
                    result.latency_samples_ms.append(latency_ms)
                else:
                    result.failed_iterations += 1

                # Merge custom metrics
                for key, value in custom_metrics.items():
                    if key not in result.custom_metrics:
                        result.custom_metrics[key] = []
                    result.custom_metrics[key].append(value)

            except asyncio.TimeoutError:
                result.failed_iterations += 1
                result.errors.append(f"Iteration {i + 1} timed out")

            except Exception as e:
                result.failed_iterations += 1
                result.errors.append(f"Iteration {i + 1} failed: {str(e)}")

            if progress_callback:
                progress_callback(i + 1, self.config.iterations)

    async def _run_concurrent(
        self,
        scenario: BenchmarkScenario,
        result: BenchmarkResult,
        progress_callback: Callable[[int, int], None] | None,
    ) -> None:
        """Run iterations concurrently."""
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        completed = 0

        async def run_with_semaphore(iteration: int) -> tuple[bool, float, dict]:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        scenario.run_iteration(),
                        timeout=self.config.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    return False, 0.0, {"error": "timeout"}
                except Exception as e:
                    return False, 0.0, {"error": str(e)}

        # Create all tasks
        tasks = [
            run_with_semaphore(i)
            for i in range(self.config.iterations)
        ]

        # Run with ramp-up if configured
        if self.config.ramp_up_seconds > 0:
            delay_per_task = self.config.ramp_up_seconds / len(tasks)
            for task in tasks:
                asyncio.create_task(task)
                await asyncio.sleep(delay_per_task)
            results = await asyncio.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks)

        # Process results
        for i, (success, latency_ms, custom_metrics) in enumerate(results):
            result.total_iterations += 1

            if success:
                result.successful_iterations += 1
                result.latency_samples_ms.append(latency_ms)
            else:
                result.failed_iterations += 1
                if "error" in custom_metrics:
                    result.errors.append(f"Iteration {i + 1}: {custom_metrics['error']}")

            completed += 1
            if progress_callback:
                progress_callback(completed, self.config.iterations)

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.name}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info("Benchmark result saved", path=str(filepath))

    def get_results(self) -> list[dict[str, Any]]:
        """Get all benchmark results."""
        return [r.to_dict() for r in self._results]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all benchmark runs."""
        if not self._results:
            return {"message": "No benchmarks have been run"}

        return {
            "total_benchmarks": len(self._results),
            "benchmarks": [
                {
                    "name": r.name,
                    "success_rate": r.successful_iterations / r.total_iterations * 100 if r.total_iterations > 0 else 0,
                    "mean_latency_ms": r.latency_stats.mean_ms if r.latency_stats else 0,
                    "p99_latency_ms": r.latency_stats.p99_ms if r.latency_stats else 0,
                    "throughput_ops_per_sec": r.throughput_ops_per_sec,
                }
                for r in self._results
            ],
        }

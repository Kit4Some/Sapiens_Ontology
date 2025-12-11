"""
Reliable SDDI Pipeline with Production-Grade Features.

Extends the base SDDI pipeline with:
- Dead-Letter Queue (DLQ) for failed documents
- Retry mechanisms with exponential backoff
- Per-document and per-step latency metrics
- Document validation and size limits
- Graceful degradation and partial success handling
- Replay capability for failed ingestions

This module provides enterprise-grade reliability for knowledge graph ingestion.
"""

import asyncio
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.sddi.loaders.neo4j_loader import Neo4jLoader
from src.sddi.pipeline import (
    PipelineProgress,
    SDDIPipeline,
    _broadcast_progress,
)
from src.sddi.reliability import (
    DeadLetterQueue,
    DocumentMetrics,
    DocumentValidator,
    DLQEntry,
    DLQStatus,
    FileDLQ,
    InMemoryDLQ,
    LatencyStats,
    PipelineMetricsCollector,
    RetryConfig,
    RetryHandler,
    StepTimer,
    ValidationConfig,
    ValidationResult,
)
from src.sddi.state import (
    LoadResult,
    LoadStatus,
    RawDocument,
    SDDIPipelineState,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ReliabilityConfig:
    """
    Configuration for pipeline reliability features.

    Attributes:
        enable_dlq: Enable Dead-Letter Queue for failed documents
        enable_retry: Enable automatic retry with backoff
        enable_metrics: Enable detailed metrics collection
        enable_validation: Enable document validation before processing

        dlq_backend: DLQ backend type ("memory" or "file")
        dlq_path: Path for file-based DLQ
        dlq_max_retries: Maximum retry attempts for DLQ entries
        dlq_ttl_hours: Hours before DLQ entries expire

        retry_max_attempts: Maximum retry attempts per step
        retry_initial_delay: Initial delay in seconds
        retry_max_delay: Maximum delay in seconds
        retry_exponential_base: Exponential backoff base

        validation_max_size_bytes: Maximum document size in bytes
        validation_max_size_chars: Maximum document size in characters
        validation_max_batch_size: Maximum documents per batch

        partial_success: Continue processing if some documents fail
        fail_fast: Stop on first error (overrides partial_success)
    """

    # Feature toggles
    enable_dlq: bool = True
    enable_retry: bool = True
    enable_metrics: bool = True
    enable_validation: bool = True

    # DLQ configuration
    dlq_backend: str = "memory"  # "memory" or "file"
    dlq_path: str = "./dlq"
    dlq_max_retries: int = 3
    dlq_ttl_hours: int = 72

    # Retry configuration
    retry_max_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential_base: float = 2.0
    retry_jitter: float = 0.1

    # Validation configuration
    validation_max_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    validation_max_size_chars: int = 5_000_000
    validation_max_batch_size: int = 100

    # Behavior
    partial_success: bool = True
    fail_fast: bool = False

    def to_retry_config(self) -> RetryConfig:
        """Convert to RetryConfig."""
        return RetryConfig(
            max_retries=self.retry_max_attempts,
            initial_delay=self.retry_initial_delay,
            max_delay=self.retry_max_delay,
            exponential_base=self.retry_exponential_base,
            jitter=self.retry_jitter,
        )

    def to_validation_config(self) -> ValidationConfig:
        """Convert to ValidationConfig."""
        return ValidationConfig(
            max_document_size_bytes=self.validation_max_size_bytes,
            max_document_size_chars=self.validation_max_size_chars,
            max_documents_per_batch=self.validation_max_batch_size,
        )


@dataclass
class DocumentProcessingResult:
    """Result of processing a single document."""

    document_id: str
    success: bool
    chunks_created: int = 0
    entities_created: int = 0
    relations_created: int = 0

    # Timing
    latency_ms: float = 0.0
    step_latencies: dict[str, float] = field(default_factory=dict)

    # Error info (if failed)
    error_message: str | None = None
    error_type: str | None = None
    failed_step: str | None = None
    dlq_entry_id: str | None = None

    # Validation info
    validation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "success": self.success,
            "chunks_created": self.chunks_created,
            "entities_created": self.entities_created,
            "relations_created": self.relations_created,
            "latency_ms": round(self.latency_ms, 2),
            "step_latencies": {k: round(v, 2) for k, v in self.step_latencies.items()},
            "error_message": self.error_message,
            "error_type": self.error_type,
            "failed_step": self.failed_step,
            "dlq_entry_id": self.dlq_entry_id,
            "validation_warnings": self.validation_warnings,
        }


@dataclass
class ReliablePipelineResult:
    """Result of reliable pipeline execution."""

    pipeline_id: str
    success: bool
    partial_success: bool = False

    # Aggregate counts
    documents_processed: int = 0
    documents_succeeded: int = 0
    documents_failed: int = 0
    documents_skipped: int = 0

    total_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0

    # Timing
    total_latency_ms: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    # Per-document results
    document_results: list[DocumentProcessingResult] = field(default_factory=list)

    # DLQ info
    dlq_entries_created: int = 0

    # Metrics summary
    metrics_summary: dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "success": self.success,
            "partial_success": self.partial_success,
            "summary": {
                "documents_processed": self.documents_processed,
                "documents_succeeded": self.documents_succeeded,
                "documents_failed": self.documents_failed,
                "documents_skipped": self.documents_skipped,
                "success_rate": round(
                    self.documents_succeeded / max(self.documents_processed, 1), 4
                ),
            },
            "outputs": {
                "total_chunks": self.total_chunks,
                "total_entities": self.total_entities,
                "total_relations": self.total_relations,
            },
            "timing": {
                "total_latency_ms": round(self.total_latency_ms, 2),
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "avg_latency_per_doc_ms": round(
                    self.total_latency_ms / max(self.documents_processed, 1), 2
                ),
            },
            "dlq": {
                "entries_created": self.dlq_entries_created,
            },
            "metrics": self.metrics_summary,
            "errors": self.errors,
            "warnings": self.warnings,
            "documents": [r.to_dict() for r in self.document_results],
        }


# =============================================================================
# Reliable Pipeline
# =============================================================================


class ReliablePipeline:
    """
    Production-grade SDDI Pipeline with reliability features.

    Wraps the base SDDIPipeline with:
    - Document validation before processing
    - Retry logic with exponential backoff
    - Dead-Letter Queue for failed documents
    - Comprehensive metrics collection
    - Partial success handling
    - Replay capability

    Usage:
        ```python
        pipeline = ReliablePipeline(
            llm=llm,
            embeddings=embeddings,
            neo4j_loader=loader,
            config=ReliabilityConfig(
                enable_dlq=True,
                enable_retry=True,
                enable_metrics=True,
            ),
        )

        result = await pipeline.run(documents)

        # Check results
        print(f"Success rate: {result.documents_succeeded}/{result.documents_processed}")
        print(f"Average latency: {result.metrics_summary['latency']['document']['mean_ms']}ms")

        # Replay failed documents
        replay_result = await pipeline.replay_dlq()
        ```
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        neo4j_loader: Neo4jLoader | None = None,
        config: ReliabilityConfig | None = None,
        # Base pipeline config
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_entity_confidence: float = 0.5,
        min_relation_confidence: float = 0.5,
        max_concurrent_chunks: int = 5,
        batch_size: int = 10,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> None:
        """
        Initialize the reliable pipeline.

        Args:
            llm: LangChain chat model for extraction
            embeddings: Embedding model for vectorization
            neo4j_loader: Optional pre-configured Neo4j loader
            config: Reliability configuration
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_entity_confidence: Minimum confidence for entities
            min_relation_confidence: Minimum confidence for relations
            max_concurrent_chunks: Max concurrent chunk processing
            batch_size: Batch size for processing
            progress_callback: Optional callback for progress updates
        """
        self.config = config or ReliabilityConfig()
        self._llm = llm
        self._embeddings = embeddings
        self._loader = neo4j_loader or Neo4jLoader()
        self._progress_callback = progress_callback
        self._pipeline_id: str = ""

        # Store base pipeline config
        self._base_config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_entity_confidence": min_entity_confidence,
            "min_relation_confidence": min_relation_confidence,
            "max_concurrent_chunks": max_concurrent_chunks,
            "batch_size": batch_size,
        }

        # Initialize base pipeline
        self._pipeline = SDDIPipeline(
            llm=llm,
            embeddings=embeddings,
            neo4j_loader=self._loader,
            progress_callback=progress_callback,
            **self._base_config,
        )

        # Initialize reliability components
        self._dlq: DeadLetterQueue | None = None
        self._validator: DocumentValidator | None = None
        self._retry_handler: RetryHandler | None = None
        self._metrics: PipelineMetricsCollector | None = None

        self._setup_reliability_components()

    def _setup_reliability_components(self) -> None:
        """Initialize reliability components based on config."""

        # DLQ
        if self.config.enable_dlq:
            if self.config.dlq_backend == "file":
                self._dlq = FileDLQ(self.config.dlq_path)
            else:
                self._dlq = InMemoryDLQ()
            logger.info(
                "DLQ initialized",
                backend=self.config.dlq_backend,
                path=self.config.dlq_path if self.config.dlq_backend == "file" else None,
            )

        # Validator
        if self.config.enable_validation:
            self._validator = DocumentValidator(self.config.to_validation_config())
            logger.info(
                "Validator initialized",
                max_size_mb=self.config.validation_max_size_bytes / 1024 / 1024,
            )

        # Retry handler
        if self.config.enable_retry:
            self._retry_handler = RetryHandler(self.config.to_retry_config())
            logger.info(
                "Retry handler initialized",
                max_retries=self.config.retry_max_attempts,
                initial_delay=self.config.retry_initial_delay,
            )

    def _report_progress(
        self,
        step: str,
        progress: float,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Report progress to callback and SSE subscribers."""
        if self._progress_callback:
            try:
                self._progress_callback(step, progress, message)
            except Exception as e:
                logger.warning("Progress callback failed", error=str(e))

        if self._pipeline_id:
            progress_obj = PipelineProgress(
                pipeline_id=self._pipeline_id,
                step=step,
                progress=progress,
                message=message,
                details=details or {},
            )
            _broadcast_progress(progress_obj)

    # =========================================================================
    # Main API
    # =========================================================================

    async def run(
        self,
        documents: list[dict[str, Any] | RawDocument],
        pipeline_id: str | None = None,
    ) -> ReliablePipelineResult:
        """
        Run the reliable pipeline with all safety features.

        Args:
            documents: List of documents to process
            pipeline_id: Optional pipeline run identifier

        Returns:
            ReliablePipelineResult with detailed metrics and results
        """
        self._pipeline_id = pipeline_id or str(uuid.uuid4())[:12]
        start_time = datetime.utcnow()

        # Initialize metrics collector
        if self.config.enable_metrics:
            self._metrics = PipelineMetricsCollector(self._pipeline_id)

        result = ReliablePipelineResult(
            pipeline_id=self._pipeline_id,
            success=False,
            start_time=start_time,
        )

        logger.info(
            "Reliable pipeline started",
            pipeline_id=self._pipeline_id,
            document_count=len(documents),
            config={
                "dlq": self.config.enable_dlq,
                "retry": self.config.enable_retry,
                "metrics": self.config.enable_metrics,
                "validation": self.config.enable_validation,
            },
        )

        self._report_progress(
            "started",
            0.0,
            f"Reliable pipeline started with {len(documents)} document(s)",
            {
                "document_count": len(documents),
                "reliability_config": {
                    "dlq_enabled": self.config.enable_dlq,
                    "retry_enabled": self.config.enable_retry,
                    "metrics_enabled": self.config.enable_metrics,
                    "validation_enabled": self.config.enable_validation,
                },
            },
        )

        try:
            # Phase 1: Validation
            validated_docs, validation_results = await self._validate_documents(
                documents, result
            )

            if not validated_docs and self.config.fail_fast:
                result.errors.append("All documents failed validation")
                return self._finalize_result(result)

            # Phase 2: Process documents
            await self._process_documents(validated_docs, validation_results, result)

            # Determine success status
            result.success = result.documents_failed == 0
            result.partial_success = (
                result.documents_succeeded > 0 and result.documents_failed > 0
            )

        except Exception as e:
            error_msg = f"Pipeline failed with unhandled error: {str(e)}"
            logger.error(error_msg, traceback=traceback.format_exc())
            result.errors.append(error_msg)
            result.success = False

            self._report_progress(
                "failed",
                -1,
                error_msg,
                {"error": str(e), "traceback": traceback.format_exc()},
            )

        return self._finalize_result(result)

    async def run_single(
        self,
        document: dict[str, Any] | RawDocument,
        pipeline_id: str | None = None,
    ) -> DocumentProcessingResult:
        """
        Process a single document with full reliability features.

        Args:
            document: Document to process
            pipeline_id: Optional pipeline run identifier

        Returns:
            DocumentProcessingResult with detailed metrics
        """
        result = await self.run([document], pipeline_id)
        if result.document_results:
            return result.document_results[0]

        # Return empty result if no document was processed
        doc_id = document.get("id", "unknown") if isinstance(document, dict) else document.id
        return DocumentProcessingResult(
            document_id=doc_id,
            success=False,
            error_message="Document was not processed",
        )

    async def replay_dlq(
        self,
        max_entries: int = 10,
        filter_status: DLQStatus | None = DLQStatus.PENDING,
    ) -> ReliablePipelineResult:
        """
        Replay failed documents from the Dead-Letter Queue.

        Args:
            max_entries: Maximum entries to replay
            filter_status: Only replay entries with this status

        Returns:
            ReliablePipelineResult for replayed documents
        """
        if not self._dlq:
            raise RuntimeError("DLQ not enabled")

        self._pipeline_id = f"replay-{str(uuid.uuid4())[:8]}"
        result = ReliablePipelineResult(
            pipeline_id=self._pipeline_id,
            success=False,
            start_time=datetime.utcnow(),
        )

        # Get entries ready for retry
        entries = await self._dlq.peek(count=max_entries)
        entries_to_process = [
            e for e in entries
            if e.is_ready_for_retry() and (filter_status is None or e.status == filter_status)
        ]

        if not entries_to_process:
            logger.info("No DLQ entries ready for replay")
            result.success = True
            return self._finalize_result(result)

        logger.info(
            "Starting DLQ replay",
            pipeline_id=self._pipeline_id,
            entry_count=len(entries_to_process),
        )

        self._report_progress(
            "replay_started",
            0.0,
            f"Replaying {len(entries_to_process)} failed document(s)",
        )

        # Initialize metrics
        if self.config.enable_metrics:
            self._metrics = PipelineMetricsCollector(self._pipeline_id)

        for idx, entry in enumerate(entries_to_process):
            progress = (idx + 1) / len(entries_to_process)
            self._report_progress(
                "replaying",
                progress,
                f"Replaying document {idx + 1}/{len(entries_to_process)}",
            )

            # Mark as retrying
            await self._dlq.mark_retrying(entry.entry_id)

            # Reconstruct document
            doc = RawDocument(
                id=entry.document_id,
                content=entry.document_content,
                source=entry.document_metadata.get("source", "dlq_replay"),
                metadata=entry.document_metadata,
            )

            # Process with retry
            doc_result = await self._process_single_document(doc, None)
            result.document_results.append(doc_result)

            if doc_result.success:
                result.documents_succeeded += 1
                await self._dlq.mark_resolved(entry.entry_id)
            else:
                result.documents_failed += 1
                await self._dlq.mark_failed_retry(
                    entry.entry_id,
                    Exception(doc_result.error_message or "Unknown error"),
                )

            result.documents_processed += 1
            result.total_chunks += doc_result.chunks_created
            result.total_entities += doc_result.entities_created
            result.total_relations += doc_result.relations_created

        result.success = result.documents_failed == 0
        result.partial_success = (
            result.documents_succeeded > 0 and result.documents_failed > 0
        )

        return self._finalize_result(result)

    async def get_dlq_stats(self) -> dict[str, Any]:
        """Get statistics about the Dead-Letter Queue."""
        if not self._dlq:
            return {"error": "DLQ not enabled"}
        return await self._dlq.get_stats()

    async def get_dlq_entries(
        self,
        count: int = 50,
        status: DLQStatus | None = None,
    ) -> list[dict[str, Any]]:
        """Get entries from the DLQ."""
        if not self._dlq:
            return []

        entries = await self._dlq.peek(count=count)
        if status:
            entries = [e for e in entries if e.status == status]
        return [e.to_dict() for e in entries]

    def get_retry_stats(self) -> dict[str, Any]:
        """Get retry handler statistics."""
        if not self._retry_handler:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._retry_handler.stats,
        }

    # =========================================================================
    # Internal Processing
    # =========================================================================

    async def _validate_documents(
        self,
        documents: list[dict[str, Any] | RawDocument],
        result: ReliablePipelineResult,
    ) -> tuple[list[RawDocument], dict[str, ValidationResult]]:
        """
        Validate documents before processing.

        Returns:
            Tuple of (validated_documents, validation_results_map)
        """
        validated_docs: list[RawDocument] = []
        validation_results: dict[str, ValidationResult] = {}

        self._report_progress(
            "validation",
            0.02,
            f"Validating {len(documents)} document(s)...",
        )

        for idx, doc in enumerate(documents):
            # Convert to RawDocument if needed
            if isinstance(doc, dict):
                try:
                    raw_doc = RawDocument(**doc)
                except Exception as e:
                    result.errors.append(f"Invalid document format at index {idx}: {str(e)}")
                    result.documents_skipped += 1
                    continue
            else:
                raw_doc = doc

            # Validate if enabled
            if self._validator:
                validation = self._validator.validate_document(
                    document_id=raw_doc.id,
                    content=raw_doc.content,
                    source=raw_doc.source,
                )
                validation_results[raw_doc.id] = validation

                if not validation.valid:
                    error_msgs = [e["message"] for e in validation.errors]
                    result.errors.append(
                        f"Document {raw_doc.id} failed validation: {'; '.join(error_msgs)}"
                    )
                    result.documents_skipped += 1

                    # Add to DLQ if enabled
                    if self._dlq:
                        await self._dlq.push_error(
                            document_id=raw_doc.id,
                            document_content=raw_doc.content,
                            error=ValueError(f"Validation failed: {'; '.join(error_msgs)}"),
                            failed_step="validation",
                            pipeline_id=self._pipeline_id,
                            document_metadata={"source": raw_doc.source, **raw_doc.metadata},
                            max_retries=0,  # Don't retry validation failures
                        )
                        result.dlq_entries_created += 1

                    if self.config.fail_fast:
                        break
                    continue

                # Add warnings to result
                if validation.warnings:
                    result.warnings.extend(validation.warnings)
            else:
                # Basic validation without validator
                if not raw_doc.content.strip():
                    result.errors.append(f"Document {raw_doc.id} is empty")
                    result.documents_skipped += 1
                    continue

            validated_docs.append(raw_doc)

        logger.info(
            "Validation completed",
            total=len(documents),
            valid=len(validated_docs),
            skipped=result.documents_skipped,
        )

        return validated_docs, validation_results

    async def _process_documents(
        self,
        documents: list[RawDocument],
        validation_results: dict[str, ValidationResult],
        result: ReliablePipelineResult,
    ) -> None:
        """Process validated documents through the pipeline."""

        if not documents:
            return

        self._report_progress(
            "processing",
            0.05,
            f"Processing {len(documents)} validated document(s)...",
        )

        # Process documents
        for idx, doc in enumerate(documents):
            progress = 0.05 + (idx / len(documents)) * 0.90
            self._report_progress(
                "processing",
                progress,
                f"Processing document {idx + 1}/{len(documents)}: {doc.id}",
                {"current_document": doc.id, "index": idx},
            )

            # Get validation result for this doc
            validation = validation_results.get(doc.id)

            # Process with retry and metrics
            doc_result = await self._process_single_document(doc, validation)
            result.document_results.append(doc_result)

            # Update aggregates
            result.documents_processed += 1
            if doc_result.success:
                result.documents_succeeded += 1
                result.total_chunks += doc_result.chunks_created
                result.total_entities += doc_result.entities_created
                result.total_relations += doc_result.relations_created
            else:
                result.documents_failed += 1
                if doc_result.dlq_entry_id:
                    result.dlq_entries_created += 1

            # Fail fast if configured
            if not doc_result.success and self.config.fail_fast:
                result.errors.append(f"Fail-fast triggered by document: {doc.id}")
                break

    async def _process_single_document(
        self,
        doc: RawDocument,
        validation: ValidationResult | None,
    ) -> DocumentProcessingResult:
        """
        Process a single document with retry and metrics.

        This is the core processing method that handles:
        - Metrics tracking per document
        - Retry logic for transient failures
        - DLQ for persistent failures
        """
        doc_start_time = time.perf_counter()

        doc_result = DocumentProcessingResult(
            document_id=doc.id,
            success=False,
            validation_warnings=validation.warnings if validation else [],
        )

        # Start metrics tracking
        if self._metrics:
            self._metrics.start_document(
                document_id=doc.id,
                source=doc.source,
                content_size_bytes=len(doc.content.encode("utf-8")),
                content_size_chars=len(doc.content),
            )

        async def process_with_pipeline() -> SDDIPipelineState:
            """Inner function to run pipeline (for retry wrapper)."""
            return await self._pipeline.run(
                documents=[doc],
                pipeline_id=f"{self._pipeline_id}-{doc.id[:8]}",
            )

        try:
            # Execute with or without retry
            if self._retry_handler and self.config.enable_retry:
                final_state = await self._retry_handler.execute(
                    process_with_pipeline,
                    on_retry=lambda attempt, error: logger.warning(
                        "Retrying document",
                        document_id=doc.id,
                        attempt=attempt,
                        error=str(error),
                    ),
                )
            else:
                final_state = await process_with_pipeline()

            # Extract results from state
            load_result = final_state.get("load_result", LoadResult())
            load_status = final_state.get("load_status", LoadStatus.FAILED)

            if load_status == LoadStatus.COMPLETED:
                doc_result.success = True
                doc_result.chunks_created = load_result.chunks_created
                doc_result.entities_created = load_result.entities_created
                doc_result.relations_created = load_result.relations_created
            elif load_status == LoadStatus.PARTIAL:
                doc_result.success = True  # Consider partial as success
                doc_result.chunks_created = load_result.chunks_created
                doc_result.entities_created = load_result.entities_created
                doc_result.relations_created = load_result.relations_created
                doc_result.validation_warnings.append("Partial load - some items may have failed")
            else:
                doc_result.success = False
                doc_result.error_message = "; ".join(final_state.get("errors", ["Unknown error"]))
                doc_result.failed_step = final_state.get("current_step", "unknown")

        except Exception as e:
            doc_result.success = False
            doc_result.error_message = str(e)
            doc_result.error_type = type(e).__name__
            doc_result.failed_step = "pipeline_execution"

            logger.error(
                "Document processing failed",
                document_id=doc.id,
                error_type=doc_result.error_type,
                error=doc_result.error_message,
            )

            # Add to DLQ if enabled
            if self._dlq:
                entry_id = await self._dlq.push_error(
                    document_id=doc.id,
                    document_content=doc.content,
                    error=e,
                    failed_step=doc_result.failed_step,
                    pipeline_id=self._pipeline_id,
                    document_metadata={"source": doc.source, **doc.metadata},
                    max_retries=self.config.dlq_max_retries,
                    ttl_hours=self.config.dlq_ttl_hours,
                )
                doc_result.dlq_entry_id = entry_id

        # Calculate timing
        doc_result.latency_ms = (time.perf_counter() - doc_start_time) * 1000

        # Finish metrics tracking
        if self._metrics:
            self._metrics.finish_document(
                document_id=doc.id,
                success=doc_result.success,
                error=doc_result.error_message,
                failed_step=doc_result.failed_step,
                chunk_count=doc_result.chunks_created,
                entity_count=doc_result.entities_created,
                relation_count=doc_result.relations_created,
            )

        return doc_result

    def _finalize_result(
        self,
        result: ReliablePipelineResult,
    ) -> ReliablePipelineResult:
        """Finalize the pipeline result with metrics summary."""
        result.end_time = datetime.utcnow()
        result.total_latency_ms = (
            (result.end_time - result.start_time).total_seconds() * 1000
        )

        # Add metrics summary
        if self._metrics:
            self._metrics.finish_pipeline()
            result.metrics_summary = self._metrics.get_summary()

        # Report completion
        status = "completed" if result.success else ("partial" if result.partial_success else "failed")

        self._report_progress(
            status,
            1.0 if result.success else -1,
            f"Pipeline {status}: {result.documents_succeeded}/{result.documents_processed} documents succeeded",
            {
                "documents_succeeded": result.documents_succeeded,
                "documents_failed": result.documents_failed,
                "documents_skipped": result.documents_skipped,
                "total_entities": result.total_entities,
                "total_relations": result.total_relations,
                "dlq_entries": result.dlq_entries_created,
            },
        )

        logger.info(
            "Reliable pipeline finished",
            pipeline_id=self._pipeline_id,
            success=result.success,
            partial_success=result.partial_success,
            documents_succeeded=result.documents_succeeded,
            documents_failed=result.documents_failed,
            documents_skipped=result.documents_skipped,
            total_latency_ms=round(result.total_latency_ms, 2),
        )

        return result


# =============================================================================
# Factory Functions
# =============================================================================


def create_reliable_pipeline(
    llm: BaseChatModel,
    embeddings: Embeddings,
    neo4j_loader: Neo4jLoader | None = None,
    config: ReliabilityConfig | None = None,
    **kwargs: Any,
) -> ReliablePipeline:
    """
    Factory function to create a reliable pipeline.

    Args:
        llm: LangChain chat model
        embeddings: Embedding model
        neo4j_loader: Optional Neo4j loader
        config: Reliability configuration
        **kwargs: Additional pipeline configuration

    Returns:
        Configured ReliablePipeline instance
    """
    return ReliablePipeline(
        llm=llm,
        embeddings=embeddings,
        neo4j_loader=neo4j_loader,
        config=config,
        **kwargs,
    )


def create_production_pipeline(
    llm: BaseChatModel,
    embeddings: Embeddings,
    neo4j_loader: Neo4jLoader | None = None,
    dlq_path: str = "./dlq",
    **kwargs: Any,
) -> ReliablePipeline:
    """
    Create a production-ready pipeline with recommended settings.

    Args:
        llm: LangChain chat model
        embeddings: Embedding model
        neo4j_loader: Optional Neo4j loader
        dlq_path: Path for file-based DLQ
        **kwargs: Additional pipeline configuration

    Returns:
        Production-configured ReliablePipeline
    """
    config = ReliabilityConfig(
        # Enable all features
        enable_dlq=True,
        enable_retry=True,
        enable_metrics=True,
        enable_validation=True,

        # Production DLQ settings
        dlq_backend="file",
        dlq_path=dlq_path,
        dlq_max_retries=3,
        dlq_ttl_hours=168,  # 7 days

        # Robust retry settings
        retry_max_attempts=3,
        retry_initial_delay=2.0,
        retry_max_delay=120.0,
        retry_exponential_base=2.0,

        # Conservative validation
        validation_max_size_bytes=50 * 1024 * 1024,  # 50 MB
        validation_max_size_chars=10_000_000,
        validation_max_batch_size=50,

        # Partial success for resilience
        partial_success=True,
        fail_fast=False,
    )

    return ReliablePipeline(
        llm=llm,
        embeddings=embeddings,
        neo4j_loader=neo4j_loader,
        config=config,
        **kwargs,
    )


def create_development_pipeline(
    llm: BaseChatModel,
    embeddings: Embeddings,
    neo4j_loader: Neo4jLoader | None = None,
    **kwargs: Any,
) -> ReliablePipeline:
    """
    Create a development pipeline with fast-fail and minimal overhead.

    Args:
        llm: LangChain chat model
        embeddings: Embedding model
        neo4j_loader: Optional Neo4j loader
        **kwargs: Additional pipeline configuration

    Returns:
        Development-configured ReliablePipeline
    """
    config = ReliabilityConfig(
        # Minimal features for fast iteration
        enable_dlq=True,
        enable_retry=False,  # Fail fast in dev
        enable_metrics=True,
        enable_validation=True,

        # In-memory DLQ for dev
        dlq_backend="memory",
        dlq_max_retries=1,

        # Quick retry
        retry_max_attempts=1,
        retry_initial_delay=0.5,

        # Fail fast for quick feedback
        partial_success=False,
        fail_fast=True,
    )

    return ReliablePipeline(
        llm=llm,
        embeddings=embeddings,
        neo4j_loader=neo4j_loader,
        config=config,
        **kwargs,
    )

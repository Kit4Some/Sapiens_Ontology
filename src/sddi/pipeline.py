"""
SDDI LangGraph Pipeline.

Schema-Driven Data Integration pipeline using LangGraph for orchestration.
Pipeline: Ingest → Chunk → Extract Entities → Extract Relations → Embed → Load

Supports parallel processing for large files with real-time progress tracking.

Enhanced with:
- Extraction quality validation
- Configurable quality thresholds
- Detailed metrics and monitoring
- Proper error propagation
"""

import asyncio
import contextlib
import hashlib
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from src.sddi.extractors.entity_extractor import (
    BatchEntityExtractor,
    ExtractionMetrics,
)
from src.sddi.extractors.resilient_extractor import (
    ResilientEntityExtractor,
    ResilientExtractionMetrics,
    ProcessingCheckpoint,
)
from src.sddi.extractors.relation_extractor import (
    RelationExtractionError,
    RelationExtractionMetrics,
    RelationExtractor,
)
from src.sddi.loaders.neo4j_loader import Neo4jLoader
from src.sddi.loaders.incremental_loader import (
    IncrementalLoader,
    DeltaReport,
    ChangeType,
    create_incremental_loader,
)
from src.sddi.state import (
    LoadResult,
    LoadStatus,
    RawDocument,
    SDDIPipelineState,
    TextChunk,
)

logger = structlog.get_logger(__name__)


class PipelineValidationError(Exception):
    """Raised when pipeline validation fails."""

    def __init__(self, message: str, step: str, metrics: dict[str, Any] | None = None):
        super().__init__(message)
        self.step = step
        self.metrics = metrics or {}


@dataclass
class PipelineQualityConfig:
    """Configuration for pipeline quality thresholds."""

    # Minimum extraction rates
    min_entity_extraction_rate: float = 0.1  # At least 10% of chunks should have entities
    min_relation_extraction_rate: float = 0.05  # At least 5% of chunks should have relations

    # Minimum counts
    min_entities_per_document: int = 1  # At least 1 entity per document
    min_total_entities: int = 1  # Total entities required

    # Error thresholds
    max_extraction_error_rate: float = 0.5  # Max 50% error rate allowed
    max_empty_extraction_rate: float = 0.8  # Max 80% empty extractions

    # Embedding validation
    require_embeddings: bool = True  # Embeddings required for loading

    # Strict mode - fail on quality issues
    strict_mode: bool = False

# Type for progress callback: (step_name, progress_percent, message)
ProgressCallback = Callable[[str, float, str], None]


@dataclass
class PipelineProgress:
    """Real-time pipeline progress information."""

    pipeline_id: str
    step: str
    progress: float  # 0.0 to 1.0
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # Real-time counts for frontend display
    entities_created: int = 0
    relations_created: int = 0
    chunks_created: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "step": self.step,
            "progress": self.progress,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "entities_created": self.entities_created,
            "relations_created": self.relations_created,
            "chunks_created": self.chunks_created,
        }


# Global progress store for SSE streaming
_pipeline_progress: dict[str, list[PipelineProgress]] = {}
_pipeline_subscribers: dict[str, list[asyncio.Queue[PipelineProgress]]] = {}


def get_pipeline_progress(pipeline_id: str) -> list[PipelineProgress]:
    """Get all progress updates for a pipeline."""
    return _pipeline_progress.get(pipeline_id, [])


async def subscribe_to_pipeline(pipeline_id: str) -> AsyncGenerator[PipelineProgress, None]:
    """Subscribe to real-time progress updates for a pipeline."""
    queue: asyncio.Queue[PipelineProgress] = asyncio.Queue()

    if pipeline_id not in _pipeline_subscribers:
        _pipeline_subscribers[pipeline_id] = []
    _pipeline_subscribers[pipeline_id].append(queue)

    try:
        while True:
            try:
                progress = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield progress
                if progress.step == "completed" or progress.step == "failed":
                    break
            except TimeoutError:
                # Send heartbeat
                yield PipelineProgress(
                    pipeline_id=pipeline_id,
                    step="heartbeat",
                    progress=-1,
                    message="Connection alive",
                )
    finally:
        if pipeline_id in _pipeline_subscribers:
            _pipeline_subscribers[pipeline_id].remove(queue)
            if not _pipeline_subscribers[pipeline_id]:
                del _pipeline_subscribers[pipeline_id]


def _broadcast_progress(progress: PipelineProgress) -> None:
    """Broadcast progress to all subscribers."""
    # Store progress
    if progress.pipeline_id not in _pipeline_progress:
        _pipeline_progress[progress.pipeline_id] = []
    _pipeline_progress[progress.pipeline_id].append(progress)

    # Broadcast to subscribers
    if progress.pipeline_id in _pipeline_subscribers:
        for queue in _pipeline_subscribers[progress.pipeline_id]:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(progress)


class SDDIPipeline:
    """
    SDDI Pipeline orchestrator using LangGraph.

    Implements a 6-stage pipeline for knowledge graph construction:
    1. Ingest: Accept raw documents
    2. Chunk: Split documents into text chunks
    3. Extract Entities: Named entity recognition using LLM
    4. Extract Relations: Relation extraction between entities
    5. Embed: Generate vector embeddings for chunks and entities
    6. Load: Persist to Neo4j graph database

    Enhanced with:
    - Extraction quality validation
    - Configurable quality thresholds
    - Detailed metrics tracking
    - Proper error propagation
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        neo4j_loader: Neo4jLoader | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_entity_confidence: float = 0.5,
        min_relation_confidence: float = 0.5,
        progress_callback: ProgressCallback | None = None,
        max_concurrent_chunks: int = 5,
        batch_size: int = 10,
        quality_config: PipelineQualityConfig | None = None,
        use_incremental_loading: bool = False,
        use_resilient_extractor: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        """
        Initialize the SDDI pipeline.

        Args:
            llm: LangChain chat model for extraction
            embeddings: Embedding model for vectorization
            neo4j_loader: Optional pre-configured loader
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_entity_confidence: Minimum confidence for entities
            min_relation_confidence: Minimum confidence for relations
            progress_callback: Optional callback for progress updates
            max_concurrent_chunks: Max concurrent chunk processing for parallelization
            batch_size: Batch size for embedding generation
            quality_config: Quality validation configuration
            use_incremental_loading: Use delta-based incremental updates (default: False)
            use_resilient_extractor: Use production-grade resilient extractor (default: True)
            circuit_breaker_threshold: Failures before circuit opens (default: 5)
            circuit_breaker_timeout: Seconds before circuit recovery attempt (default: 30)
            max_retries: Maximum retry attempts per operation (default: 5)
        """
        self._llm = llm
        self._embeddings = embeddings
        self._loader = neo4j_loader or Neo4jLoader()
        self._progress_callback = progress_callback
        self._max_concurrent = max_concurrent_chunks
        self._batch_size = batch_size
        self._pipeline_id: str = ""
        self._quality_config = quality_config or PipelineQualityConfig()
        self._use_incremental = use_incremental_loading
        self._use_resilient = use_resilient_extractor
        self._incremental_loader: IncrementalLoader | None = None
        self._last_delta_report: DeltaReport | None = None

        # Initialize incremental loader if enabled
        if use_incremental_loading:
            self._incremental_loader = create_incremental_loader()

        # Metrics storage
        self._entity_metrics: ExtractionMetrics | ResilientExtractionMetrics | None = None
        self._relation_metrics: RelationExtractionMetrics | None = None
        self._entity_checkpoint: ProcessingCheckpoint | None = None

        # Text splitter configuration
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Entity extractor - choose resilient or legacy based on config
        if use_resilient_extractor:
            self._resilient_entity_extractor: ResilientEntityExtractor | None = ResilientEntityExtractor(
                llm=llm,
                min_confidence=min_entity_confidence,
                max_retries=max_retries,
                circuit_failure_threshold=circuit_breaker_threshold,
                circuit_recovery_timeout=circuit_breaker_timeout,
                initial_batch_size=batch_size,
                max_concurrent_batches=max_concurrent_chunks,
                enable_health_check=True,
                progress_callback=self._report_progress,
            )
            self._entity_extractor: BatchEntityExtractor | None = None
            logger.info(
                "Using ResilientEntityExtractor",
                circuit_threshold=circuit_breaker_threshold,
                max_retries=max_retries,
            )
        else:
            self._resilient_entity_extractor = None
            self._entity_extractor = BatchEntityExtractor(
                llm=llm,
                min_confidence=min_entity_confidence,
                max_retries=3,
                retry_delay=2.0,
                min_text_length=50,
                fail_on_empty=self._quality_config.strict_mode,
            )
            logger.info("Using legacy BatchEntityExtractor")

        self._relation_extractor = RelationExtractor(
            llm=llm,
            min_confidence=min_relation_confidence,
            max_retries=3,
            retry_delay=2.0,
            validate_entities=True,
            progress_callback=self._report_progress,
        )

        # Build the workflow
        self._workflow = self._build_workflow()

    def _validate_entity_extraction(
        self,
        entities: list[Any],
        chunks: list[TextChunk],
        documents: list[Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate entity extraction quality.

        Supports both legacy BatchEntityExtractor and ResilientEntityExtractor metrics.

        Returns:
            Tuple of (is_valid, list of warning/error messages)
        """
        messages: list[str] = []
        is_valid = True

        # Get metrics from appropriate extractor
        if self._use_resilient and self._resilient_entity_extractor:
            # ResilientEntityExtractor - metrics already stored in self._entity_metrics
            # during extraction, but get fresh copy to be safe
            metrics = self._resilient_entity_extractor.get_metrics()
            self._entity_metrics = metrics

            # Extract values for validation (ResilientExtractionMetrics format)
            chunks_processed = metrics.chunks_processed
            chunks_with_entities = metrics.chunks_with_entities
            extraction_rate = chunks_with_entities / max(chunks_processed, 1)
            total_errors = metrics.total_errors
            error_rate = total_errors / max(chunks_processed, 1) if chunks_processed > 0 else 0
            # ResilientExtractionMetrics doesn't track empty_extractions the same way
            empty_rate = (chunks_processed - chunks_with_entities) / max(chunks_processed, 1) if chunks_processed > 0 else 0

        elif self._entity_extractor:
            # Legacy BatchEntityExtractor
            metrics = self._entity_extractor.get_metrics()
            self._entity_metrics = metrics

            chunks_processed = metrics.chunks_processed
            extraction_rate = metrics.extraction_rate
            error_rate = metrics.extraction_errors / max(chunks_processed, 1) if chunks_processed > 0 else 0
            empty_rate = metrics.empty_extractions / max(chunks_processed, 1) if chunks_processed > 0 else 0

        else:
            # No extractor available - skip metric-based validation
            logger.warning("No entity extractor available for metrics validation")
            chunks_processed = len(chunks)
            extraction_rate = len(entities) / max(chunks_processed, 1)
            error_rate = 0
            empty_rate = 0

        # Check minimum total entities
        if len(entities) < self._quality_config.min_total_entities:
            msg = f"Insufficient entities extracted: {len(entities)} < {self._quality_config.min_total_entities}"
            messages.append(msg)
            if self._quality_config.strict_mode:
                is_valid = False

        # Check entities per document
        entities_per_doc = len(entities) / max(len(documents), 1)
        if entities_per_doc < self._quality_config.min_entities_per_document:
            msg = f"Low entity density: {entities_per_doc:.2f} per document"
            messages.append(msg)

        # Check extraction rate
        if extraction_rate < self._quality_config.min_entity_extraction_rate:
            msg = f"Low extraction rate: {extraction_rate:.2%} < {self._quality_config.min_entity_extraction_rate:.0%}"
            messages.append(msg)
            if self._quality_config.strict_mode:
                is_valid = False

        # Check error rate
        if chunks_processed > 0 and error_rate > self._quality_config.max_extraction_error_rate:
            msg = f"High extraction error rate: {error_rate:.2%}"
            messages.append(msg)
            if self._quality_config.strict_mode:
                is_valid = False

        # Check empty extraction rate
        if chunks_processed > 0 and empty_rate > self._quality_config.max_empty_extraction_rate:
            msg = f"High empty extraction rate: {empty_rate:.2%}"
            messages.append(msg)

        return is_valid, messages

    def _validate_relation_extraction(
        self,
        relations: list[Any],
        entities: list[Any],
        chunks: list[TextChunk],
    ) -> tuple[bool, list[str]]:
        """
        Validate relation extraction quality.

        Returns:
            Tuple of (is_valid, list of warning/error messages)
        """
        messages: list[str] = []
        is_valid = True

        # Get metrics from extractor
        metrics = self._relation_extractor.get_metrics()
        self._relation_metrics = metrics

        # Skip validation if not enough entities
        if len(entities) < 2:
            messages.append("Not enough entities for relation extraction")
            return True, messages

        # Check extraction rate (optional for relations)
        if metrics.extraction_rate < self._quality_config.min_relation_extraction_rate:
            msg = f"Low relation extraction rate: {metrics.extraction_rate:.2%}"
            messages.append(msg)

        # Check for invalid entity references
        if metrics.relations_with_invalid_entities > 0:
            ratio = metrics.relations_with_invalid_entities / max(metrics.total_relations + metrics.relations_with_invalid_entities, 1)
            if ratio > 0.5:
                msg = f"High invalid entity rate in relations: {ratio:.2%}"
                messages.append(msg)

        return is_valid, messages

    def _validate_embeddings(
        self,
        embeddings: dict[str, list[float]],
        chunks: list[TextChunk],
        entities: list[Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate embedding generation.

        Returns:
            Tuple of (is_valid, list of warning/error messages)
        """
        messages: list[str] = []
        is_valid = True

        expected_count = len(chunks) + len(entities)
        actual_count = len(embeddings)

        if actual_count < expected_count:
            missing = expected_count - actual_count
            msg = f"Missing embeddings: {missing} of {expected_count}"
            messages.append(msg)

            if self._quality_config.require_embeddings:
                missing_ratio = missing / max(expected_count, 1)
                # More than 10% missing in strict mode
                if missing_ratio > 0.1 and self._quality_config.strict_mode:
                    is_valid = False

        return is_valid, messages

    async def _validate_embedding_dimensions(
        self,
        embeddings: dict[str, list[float]],
    ) -> list[str]:
        """
        Validate embedding dimensions against expected configuration.

        Checks that all embeddings have the expected number of dimensions
        (default 1536 for OpenAI text-embedding-3-small) to prevent
        index mismatch errors in Neo4j.

        Args:
            embeddings: Dictionary mapping item IDs to embedding vectors

        Returns:
            List of validation error messages (empty if valid)
        """
        from src.config.settings import get_settings

        settings = get_settings()

        # Check if validation is enabled
        if not settings.ingestion.validate_embedding_dimensions:
            return []

        if not embeddings:
            return []

        expected_dim = settings.ingestion.embedding_dimensions
        errors: list[str] = []
        dimension_counts: dict[int, int] = {}

        # Check a sample of embeddings for performance
        sample_size = min(100, len(embeddings))
        sample_ids = list(embeddings.keys())[:sample_size]

        for item_id in sample_ids:
            embedding = embeddings[item_id]
            dim = len(embedding)
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1

            if dim != expected_dim:
                if len(errors) < 5:  # Limit error messages
                    errors.append(
                        f"Embedding dimension mismatch for '{item_id}': "
                        f"got {dim}, expected {expected_dim}"
                    )

        # Log dimension distribution
        if len(dimension_counts) > 1:
            logger.warning(
                "Inconsistent embedding dimensions detected",
                dimension_counts=dimension_counts,
                expected=expected_dim,
            )

        # Summary error if mismatches found
        mismatched_dims = {d for d in dimension_counts if d != expected_dim}
        if mismatched_dims:
            errors.insert(
                0,
                f"Embedding dimension validation failed: expected {expected_dim}, "
                f"but found dimensions {sorted(mismatched_dims)}. "
                f"Check INGESTION_EMBEDDING_DIMENSIONS setting or embedding model."
            )

        return errors

    def _report_progress(
        self,
        step: str,
        progress: float,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Report progress to callback and SSE subscribers."""
        # Legacy callback
        if self._progress_callback:
            try:
                self._progress_callback(step, progress, message)
            except Exception as e:
                logger.warning("Progress callback failed", error=str(e))

        # Broadcast to SSE subscribers
        if self._pipeline_id:
            # Extract counts from details if available
            details_dict = details or {}
            entities_count = details_dict.get("entities_so_far", 0)
            relations_count = details_dict.get("relations_so_far", 0)
            chunks_count = details_dict.get("chunks_processed", 0)

            progress_obj = PipelineProgress(
                pipeline_id=self._pipeline_id,
                step=step,
                progress=progress,
                message=message,
                details=details_dict,
                entities_created=entities_count,
                relations_created=relations_count,
                chunks_created=chunks_count,
            )
            _broadcast_progress(progress_obj)

    def _build_workflow(self) -> StateGraph[SDDIPipelineState]:
        """Build the LangGraph workflow."""

        workflow: StateGraph[SDDIPipelineState] = StateGraph(SDDIPipelineState)

        # Add nodes
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("chunk", self._chunk_node)
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("extract_relations", self._extract_relations_node)
        workflow.add_node("embed", self._embed_node)
        workflow.add_node("load", self._load_node)

        # Define edges (linear pipeline)
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "chunk")
        workflow.add_edge("chunk", "extract_entities")
        workflow.add_edge("extract_entities", "extract_relations")
        workflow.add_edge("extract_relations", "embed")
        workflow.add_edge("embed", "load")
        workflow.add_edge("load", END)

        return workflow

    # =========================================================================
    # Pipeline Nodes
    # =========================================================================

    async def _ingest_node(self, state: SDDIPipelineState) -> dict[str, Any]:
        """
        Ingest node: Validate and prepare raw documents.
        """
        raw_data = state.get("raw_data", [])
        pipeline_id = state.get("pipeline_id") or str(uuid.uuid4())[:8]

        self._report_progress("ingest", 0.05, f"Validating {len(raw_data)} document(s)...")

        logger.info(
            "Ingest started",
            pipeline_id=pipeline_id,
            document_count=len(raw_data),
        )

        # Validate documents
        validated_docs = []
        errors = state.get("errors", [])

        for doc in raw_data:
            if isinstance(doc, dict):
                try:
                    doc = RawDocument(**doc)
                except Exception as e:
                    errors.append(f"Invalid document format: {str(e)}")
                    continue

            if not doc.content.strip():
                errors.append(f"Empty document: {doc.id}")
                continue

            validated_docs.append(doc)

        return {
            "raw_data": validated_docs,
            "pipeline_id": pipeline_id,
            "current_step": "ingest",
            "errors": errors,
            "load_status": LoadStatus.IN_PROGRESS,
        }

    async def _chunk_node(self, state: SDDIPipelineState) -> dict[str, Any]:
        """
        Chunk node: Split documents into text chunks.
        """
        raw_data = state.get("raw_data", [])
        pipeline_id = state.get("pipeline_id", "")

        self._report_progress("chunk", 0.10, f"Chunking {len(raw_data)} document(s)...")

        logger.info("Chunking started", document_count=len(raw_data))

        chunks: list[TextChunk] = []

        for doc in raw_data:
            # Split document into chunks
            doc_chunks = self._text_splitter.split_text(doc.content)

            for idx, chunk_text in enumerate(doc_chunks):
                # Generate deterministic chunk ID
                chunk_id = self._generate_chunk_id(doc.id, idx)

                # Calculate character positions (approximate)
                start_char = sum(len(c) for c in doc_chunks[:idx])
                end_char = start_char + len(chunk_text)

                chunk = TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    doc_id=doc.id,
                    position=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "source": doc.source,
                        "pipeline_id": pipeline_id,
                        **doc.metadata,
                    },
                )
                chunks.append(chunk)

        logger.info("Chunking completed", chunk_count=len(chunks))

        self._report_progress("chunk", 0.15, f"Created {len(chunks)} chunks")

        return {
            "chunks": chunks,
            "current_step": "chunk",
        }

    async def _extract_entities_node(self, state: SDDIPipelineState) -> dict[str, Any]:
        """
        Entity extraction node: Extract named entities from chunks.

        Uses either ResilientEntityExtractor (default) or legacy BatchEntityExtractor.
        ResilientEntityExtractor provides:
        - Circuit breaker for API protection
        - Exponential backoff with jitter
        - Adaptive batch sizing
        - Progress checkpointing
        - Dead letter queue for failed items
        """
        chunks = state.get("chunks", [])
        errors = state.get("errors", [])

        extractor_type = "ResilientEntityExtractor" if self._use_resilient else "BatchEntityExtractor"
        self._report_progress(
            "extract_entities",
            0.20,
            f"Extracting entities from {len(chunks)} chunks using {extractor_type}...",
            {
                "total_chunks": len(chunks),
                "batch_size": self._batch_size,
                "extractor": extractor_type,
            },
        )

        logger.info(
            "Entity extraction started",
            chunk_count=len(chunks),
            extractor=extractor_type,
            sample_chunk_length=len(chunks[0].text) if chunks else 0,
        )

        # Log first chunk sample for debugging
        if chunks:
            logger.debug(
                "First chunk sample",
                chunk_id=chunks[0].id,
                text_preview=chunks[0].text[:200] if chunks[0].text else "EMPTY",
            )

        all_entities = []

        try:
            if self._use_resilient and self._resilient_entity_extractor:
                # Use resilient extractor with all advanced features
                all_entities = await self._resilient_entity_extractor.extract_batch(
                    chunks,
                    pipeline_id=self._pipeline_id,
                    resume_checkpoint=self._entity_checkpoint,
                )

                # Store metrics and checkpoint
                self._entity_metrics = self._resilient_entity_extractor.get_metrics()
                self._entity_checkpoint = self._resilient_entity_extractor.get_checkpoint()

                # Log circuit breaker state
                cb_state = self._resilient_entity_extractor.get_circuit_breaker_state()
                logger.info(
                    "Resilient extraction completed",
                    entities=len(all_entities),
                    circuit_state=cb_state.value,
                    metrics=self._entity_metrics.to_dict() if self._entity_metrics else {},
                )

                # Check for items in DLQ
                dlq = self._resilient_entity_extractor.get_dead_letter_queue()
                if dlq.items:
                    dlq_count = len(dlq.items)
                    errors.append(f"DLQ: {dlq_count} chunks failed after all retries")
                    logger.warning(
                        "Items in dead letter queue",
                        count=dlq_count,
                        items=[item.to_dict() for item in dlq.get_all()[:5]],  # Log first 5
                    )

            else:
                # Use legacy batch extractor
                batch_size = self._batch_size
                total_batches = (len(chunks) + batch_size - 1) // batch_size

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(chunks))
                    batch_chunks = chunks[start_idx:end_idx]

                    # Progress update per batch
                    batch_progress = 0.20 + (batch_idx / total_batches) * 0.25
                    self._report_progress(
                        "extract_entities",
                        batch_progress,
                        f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_chunks)} chunks)...",
                        {
                            "batch": batch_idx + 1,
                            "total_batches": total_batches,
                            "chunks_processed": start_idx,
                            "total_chunks": len(chunks),
                        },
                    )

                    # Extract entities from batch
                    if self._entity_extractor:
                        batch_entities = await self._entity_extractor.extract_batch(
                            batch_chunks,
                            deduplicate=False,  # Deduplicate at the end
                        )
                        all_entities.extend(batch_entities)

                    logger.info(
                        "Batch extraction completed",
                        batch=batch_idx + 1,
                        entities_in_batch=len(batch_entities) if 'batch_entities' in dir() else 0,
                        total_so_far=len(all_entities),
                    )

                # Store metrics for legacy extractor
                if self._entity_extractor:
                    self._entity_metrics = self._entity_extractor.get_metrics()

                # Deduplicate entities globally
                if all_entities:
                    seen_ids: set[str] = set()
                    unique_entities = []
                    for entity in all_entities:
                        if entity.id not in seen_ids:
                            seen_ids.add(entity.id)
                            unique_entities.append(entity)
                    all_entities = unique_entities

        except Exception as e:
            import traceback
            error_msg = f"Entity extraction failed: {str(e)}"
            logger.error(error_msg, traceback=traceback.format_exc())
            errors.append(error_msg)

            # For resilient extractor, try to recover entities from checkpoint
            if self._use_resilient and self._resilient_entity_extractor:
                checkpoint = self._resilient_entity_extractor.get_checkpoint()
                if checkpoint and checkpoint.extracted_entities:
                    all_entities = checkpoint.extracted_entities
                    logger.info(
                        "Recovered entities from checkpoint after error",
                        recovered_count=len(all_entities),
                    )

        # Validate entity extraction quality
        raw_data = state.get("raw_data", [])
        is_valid, validation_messages = self._validate_entity_extraction(
            all_entities, chunks, raw_data
        )

        for msg in validation_messages:
            logger.warning(f"Entity extraction validation: {msg}")
            errors.append(f"Validation: {msg}")

        if not all_entities and chunks:
            warning_msg = "CRITICAL: No entities extracted from chunks. Check LLM API connection and chunk content."
            logger.error(warning_msg)
            errors.append(warning_msg)

            if self._quality_config.strict_mode:
                raise PipelineValidationError(
                    "Entity extraction failed - no entities found",
                    step="extract_entities",
                    metrics=self._entity_metrics.to_dict() if self._entity_metrics else {},
                )

        if not is_valid and self._quality_config.strict_mode:
            raise PipelineValidationError(
                "Entity extraction quality validation failed",
                step="extract_entities",
                metrics=self._entity_metrics.to_dict() if self._entity_metrics else {},
            )

        logger.info(
            "Entity extraction completed",
            entity_count=len(all_entities),
            validation_passed=is_valid,
            extractor=extractor_type,
            metrics=self._entity_metrics.to_dict() if self._entity_metrics else {},
        )

        self._report_progress(
            "extract_entities",
            0.45,
            f"Extracted {len(all_entities)} unique entities",
            {
                "entities_count": len(all_entities),
                "validation_passed": is_valid,
                "extractor": extractor_type,
                "metrics": self._entity_metrics.to_dict() if self._entity_metrics else {},
            },
        )

        return {
            "entities": all_entities,
            "current_step": "extract_entities",
            "errors": errors,
        }

    async def _extract_relations_node(self, state: SDDIPipelineState) -> dict[str, Any]:
        """
        Relation extraction node: Extract relations between entities.
        """
        chunks = state.get("chunks", [])
        entities = state.get("entities", [])
        errors = state.get("errors", [])

        self._report_progress(
            "extract_relations", 0.50,
            f"Extracting relations between {len(entities)} entities..."
        )

        logger.info(
            "Relation extraction started",
            chunk_count=len(chunks),
            entity_count=len(entities),
        )

        # Skip relation extraction if no entities
        if len(entities) < 2:
            logger.warning("Skipping relation extraction - not enough entities")
            return {
                "relations": [],
                "triplets": [],
                "current_step": "extract_relations",
                "errors": errors + ["Skipped relation extraction: insufficient entities"],
            }

        try:
            relations = await self._relation_extractor.extract_batch(
                chunks,
                entities,
                deduplicate=True,
            )
        except RelationExtractionError as e:
            if not e.recoverable:
                raise
            logger.error(f"Relation extraction error (recoverable): {e}")
            errors.append(f"Relation extraction error: {e}")
            relations = []
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            errors.append(f"Relation extraction failed: {e}")
            relations = []

        # Convert to triplets
        triplets = self._relation_extractor.relations_to_triplets(relations, entities)

        # Validate relation extraction
        is_valid, validation_messages = self._validate_relation_extraction(
            relations, entities, chunks
        )

        for msg in validation_messages:
            logger.warning(f"Relation extraction validation: {msg}")
            errors.append(f"Validation: {msg}")

        logger.info(
            "Relation extraction completed",
            relation_count=len(relations),
            triplet_count=len(triplets),
            validation_passed=is_valid,
            metrics=self._relation_metrics.to_dict() if self._relation_metrics else {},
        )

        self._report_progress(
            "extract_relations", 0.70,
            f"Extracted {len(relations)} relations",
            {
                "relation_count": len(relations),
                "triplet_count": len(triplets),
                "validation_passed": is_valid,
                "metrics": self._relation_metrics.to_dict() if self._relation_metrics else {},
            },
        )

        return {
            "relations": relations,
            "triplets": triplets,
            "current_step": "extract_relations",
            "errors": errors,
        }

    async def _embed_node(self, state: SDDIPipelineState) -> dict[str, Any]:
        """
        Embedding node: Generate vector embeddings for chunks and entities.

        Uses batch processing with progress tracking.
        """
        chunks = state.get("chunks", [])
        entities = state.get("entities", [])

        total_items = len(chunks) + len(entities)
        self._report_progress(
            "embed",
            0.70,
            f"Generating embeddings for {total_items} items...",
            {"total_chunks": len(chunks), "total_entities": len(entities)},
        )

        logger.info(
            "Embedding started",
            chunk_count=len(chunks),
            entity_count=len(entities),
        )

        embeddings: dict[str, list[float]] = {}

        # Embed chunks in batches for better performance
        if chunks:
            chunk_texts = [c.text for c in chunks]
            batch_size = self._batch_size * 10  # Larger batches for embeddings
            total_chunk_batches = (len(chunk_texts) + batch_size - 1) // batch_size

            for batch_idx in range(total_chunk_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunk_texts))
                batch_texts = chunk_texts[start_idx:end_idx]
                batch_chunks = chunks[start_idx:end_idx]

                # Progress: 0.70 to 0.82 for chunks
                batch_progress = 0.70 + (batch_idx / max(total_chunk_batches, 1)) * 0.12
                self._report_progress(
                    "embed",
                    batch_progress,
                    f"Embedding chunks batch {batch_idx + 1}/{total_chunk_batches}...",
                    {
                        "type": "chunks",
                        "batch": batch_idx + 1,
                        "total_batches": total_chunk_batches,
                        "processed": start_idx,
                        "total": len(chunks),
                    },
                )

                batch_embeddings = await self._embeddings.aembed_documents(batch_texts)

                for chunk, embedding in zip(batch_chunks, batch_embeddings, strict=True):
                    embeddings[chunk.id] = embedding

        # Embed entities (using name + description)
        if entities:
            entity_texts = [
                f"{e.name}: {e.description}" if e.description else e.name for e in entities
            ]
            batch_size = self._batch_size * 10
            total_entity_batches = (len(entity_texts) + batch_size - 1) // batch_size

            for batch_idx in range(total_entity_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(entity_texts))
                batch_texts = entity_texts[start_idx:end_idx]
                batch_entities = entities[start_idx:end_idx]

                # Progress: 0.82 to 0.88 for entities
                batch_progress = 0.82 + (batch_idx / max(total_entity_batches, 1)) * 0.06
                self._report_progress(
                    "embed",
                    batch_progress,
                    f"Embedding entities batch {batch_idx + 1}/{total_entity_batches}...",
                    {
                        "type": "entities",
                        "batch": batch_idx + 1,
                        "total_batches": total_entity_batches,
                        "processed": start_idx,
                        "total": len(entities),
                    },
                )

                batch_embeddings = await self._embeddings.aembed_documents(batch_texts)

                for entity, embedding in zip(batch_entities, batch_embeddings, strict=True):
                    embeddings[entity.id] = embedding

        logger.info("Embedding completed", embedding_count=len(embeddings))

        # Validate embedding dimensions if enabled
        validation_errors = await self._validate_embedding_dimensions(embeddings)
        if validation_errors:
            for error in validation_errors:
                logger.error("Embedding validation failed", error=error)
            # Add errors to state but continue (non-blocking by default)
            return {
                "embeddings": embeddings,
                "current_step": "embed",
                "errors": validation_errors,
            }

        self._report_progress(
            "embed",
            0.88,
            f"Created {len(embeddings)} embeddings",
            {"total_embeddings": len(embeddings)},
        )

        return {
            "embeddings": embeddings,
            "current_step": "embed",
        }

    async def _load_node(self, state: SDDIPipelineState) -> dict[str, Any]:
        """
        Load node: Persist extracted data to Neo4j.
        """
        chunks = state.get("chunks", [])
        entities = state.get("entities", [])
        relations = state.get("relations", [])
        embeddings = state.get("embeddings", {})
        errors = state.get("errors", [])

        # Choose loading strategy: incremental or full
        use_incremental = self._use_incremental and self._incremental_loader is not None

        if use_incremental:
            self._report_progress(
                "load", 0.85,
                f"Detecting changes for {len(entities)} entities, {len(relations)} relations..."
            )

            logger.info(
                "Incremental loading: detecting changes",
                chunks=len(chunks),
                entities=len(entities),
                relations=len(relations),
            )
        else:
            self._report_progress(
                "load", 0.90,
                f"Loading {len(chunks)} chunks, {len(entities)} entities, {len(relations)} relations to Neo4j..."
            )

            logger.info(
                "Full loading to Neo4j started",
                chunks=len(chunks),
                entities=len(entities),
                relations=len(relations),
            )

        try:
            if use_incremental and self._incremental_loader:
                # Incremental loading with delta detection
                delta_report, load_result = await self._incremental_loader.load_incremental(
                    chunks=chunks,
                    entities=entities,
                    relations=relations,
                    embeddings=embeddings,
                )
                self._last_delta_report = delta_report

                logger.info(
                    "Delta detection completed",
                    new_entities=delta_report.new_entities,
                    modified_entities=delta_report.modified_entities,
                    unchanged_entities=delta_report.unchanged_entities,
                    new_relations=delta_report.new_relations,
                    detection_time_ms=round(delta_report.detection_time_ms, 2),
                )

                self._report_progress(
                    "load", 0.92,
                    f"Delta: {delta_report.new_entities} new, {delta_report.modified_entities} modified entities"
                )
            else:
                # Standard full loading
                load_result = await self._loader.load_all(
                    chunks=chunks,
                    entities=entities,
                    relations=relations,
                    embeddings=embeddings,
                )

            load_status = LoadStatus.COMPLETED if not load_result.errors else LoadStatus.PARTIAL
            errors.extend(load_result.errors)

        except Exception as e:
            logger.error("Loading failed", error=str(e))
            load_result = LoadResult(errors=[str(e)])
            load_status = LoadStatus.FAILED
            errors.append(f"Load failed: {str(e)}")

        logger.info(
            "Loading completed",
            status=load_status.value,
            chunks_created=load_result.chunks_created,
            entities_created=load_result.entities_created,
            relations_created=load_result.relations_created,
        )

        self._report_progress(
            "load", 0.95,
            f"Loaded: {load_result.chunks_created} chunks, {load_result.entities_created} entities, {load_result.relations_created} relations"
        )

        # Detect and create communities after loading entities and relations
        communities_created = 0
        if load_result.entities_created > 0 and load_result.relations_created > 0:
            try:
                communities_created = await self._detect_and_create_communities(
                    entities=entities,
                    relations=relations,
                    embeddings=embeddings,
                )
                logger.info("Communities created", count=communities_created)
            except Exception as e:
                logger.warning("Community detection failed (non-fatal)", error=str(e))
                errors.append(f"Community detection failed: {str(e)}")

        self._report_progress(
            "load", 1.0,
            f"Complete: {load_result.entities_created} entities, {load_result.relations_created} relations, {communities_created} communities",
            {"communities_created": communities_created},
        )

        return {
            "load_status": load_status,
            "load_result": load_result,
            "current_step": "load",
            "errors": errors,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_chunk_id(self, doc_id: str, position: int) -> str:
        """Generate deterministic chunk ID."""
        key = f"{doc_id}:chunk:{position}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def _detect_and_create_communities(
        self,
        entities: list[Any],
        relations: list[Any],
        embeddings: dict[str, list[float]] | None = None,
        min_community_size: int = 2,
    ) -> int:
        """
        Detect communities from entity relationships and create Community nodes.

        Uses a simple connected components approach that groups entities
        based on their relationship connections. For more advanced community
        detection (Louvain, Label Propagation), Neo4j GDS library is required.

        Args:
            entities: List of extracted entities
            relations: List of extracted relations
            embeddings: Optional embeddings for generating community summaries
            min_community_size: Minimum entities required to form a community

        Returns:
            Number of communities created
        """
        if not entities or not relations:
            return 0

        self._report_progress(
            "community_detection", 0.96,
            f"Detecting communities from {len(entities)} entities..."
        )

        # Build adjacency list from relations
        entity_ids = {e.id for e in entities}
        adjacency: dict[str, set[str]] = {eid: set() for eid in entity_ids}

        for rel in relations:
            source = rel.source_entity
            target = rel.target_entity
            if source in adjacency and target in adjacency:
                adjacency[source].add(target)
                adjacency[target].add(source)  # Bidirectional

        # Find connected components using BFS
        visited: set[str] = set()
        communities: list[list[str]] = []

        for entity_id in entity_ids:
            if entity_id in visited:
                continue

            # BFS to find all connected entities
            component: list[str] = []
            queue = [entity_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= min_community_size:
                communities.append(component)

        logger.info(
            "Community detection completed",
            total_communities=len(communities),
            entity_coverage=sum(len(c) for c in communities) / max(len(entity_ids), 1),
        )

        if not communities:
            return 0

        # Create community nodes with summaries
        entity_map = {e.id: e for e in entities}
        embeddings = embeddings or {}
        created_count = 0

        for idx, component in enumerate(communities):
            community_id = f"community_{self._pipeline_id}_{idx}"

            # Generate summary from entity names and types
            member_entities = [entity_map.get(eid) for eid in component if eid in entity_map]
            member_names = [e.name for e in member_entities if e][:10]  # Limit for summary
            member_types = list({e.type.value for e in member_entities if e})[:5]

            summary = f"Community of {len(component)} entities. "
            if member_types:
                summary += f"Types: {', '.join(member_types)}. "
            if member_names:
                summary += f"Members include: {', '.join(member_names[:5])}"
                if len(member_names) > 5:
                    summary += f" and {len(member_names) - 5} more."

            # Generate community embedding from average of member embeddings
            community_embedding = None
            member_embeddings = [embeddings.get(eid) for eid in component if eid in embeddings]
            if member_embeddings:
                # Simple average of member embeddings
                dim = len(member_embeddings[0])
                community_embedding = [
                    sum(emb[i] for emb in member_embeddings) / len(member_embeddings)
                    for i in range(dim)
                ]

            try:
                success = await self._loader.create_community(
                    community_id=community_id,
                    summary=summary,
                    entity_ids=component,
                    embedding=community_embedding,
                    level=0,  # Leaf level
                    metadata={
                        "pipeline_id": self._pipeline_id,
                        "detection_method": "connected_components",
                        "member_count": len(component),
                        "entity_types": member_types,
                    },
                )
                if success:
                    created_count += 1

            except Exception as e:
                logger.warning(
                    "Failed to create community",
                    community_id=community_id,
                    error=str(e),
                )

        self._report_progress(
            "community_detection", 0.98,
            f"Created {created_count} communities"
        )

        return created_count

    # =========================================================================
    # Public API
    # =========================================================================

    async def run(
        self,
        documents: list[dict[str, Any] | RawDocument],
        pipeline_id: str | None = None,
    ) -> SDDIPipelineState:
        """
        Run the complete SDDI pipeline.

        Args:
            documents: List of documents to process
            pipeline_id: Optional pipeline run identifier

        Returns:
            Final pipeline state with all extracted data
        """
        # Convert dicts to RawDocument if needed
        raw_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                raw_docs.append(RawDocument(**doc))
            else:
                raw_docs.append(doc)

        self._pipeline_id = pipeline_id or str(uuid.uuid4())[:8]

        initial_state: SDDIPipelineState = {
            "raw_data": raw_docs,
            "pipeline_id": self._pipeline_id,
            "chunks": [],
            "entities": [],
            "relations": [],
            "triplets": [],
            "embeddings": {},
            "load_status": LoadStatus.PENDING,
            "load_result": LoadResult(),
            "current_step": "init",
            "errors": [],
            "metadata": {},
        }

        # Compile and run workflow
        app = self._workflow.compile()

        logger.info(
            "SDDI Pipeline started",
            pipeline_id=self._pipeline_id,
            document_count=len(raw_docs),
        )

        # Broadcast start event
        self._report_progress(
            "started",
            0.0,
            f"Pipeline started with {len(raw_docs)} document(s)",
            {"document_count": len(raw_docs)},
        )

        try:
            final_state: SDDIPipelineState = await app.ainvoke(initial_state)  # type: ignore[arg-type,assignment]

            # Broadcast completion event
            load_result = final_state.get("load_result")
            self._report_progress(
                "completed",
                1.0,
                "Pipeline completed successfully",
                {
                    "entities_created": load_result.entities_created if load_result else 0,
                    "relations_created": load_result.relations_created if load_result else 0,
                    "chunks_created": load_result.chunks_created if load_result else 0,
                    "errors": final_state.get("errors", []),
                },
            )

            logger.info(
                "SDDI Pipeline completed",
                pipeline_id=self._pipeline_id,
                status=final_state.get("load_status", LoadStatus.FAILED).value,
                entities=len(final_state.get("entities", [])),
                relations=len(final_state.get("relations", [])),
            )

            return final_state

        except Exception as e:
            # Broadcast failure event
            self._report_progress(
                "failed",
                -1,
                f"Pipeline failed: {str(e)}",
                {"error": str(e)},
            )
            raise

    async def run_step(
        self,
        step: Literal["ingest", "chunk", "extract_entities", "extract_relations", "embed", "load"],
        state: SDDIPipelineState,
    ) -> SDDIPipelineState:
        """
        Run a single pipeline step.

        Useful for debugging or step-by-step execution.

        Args:
            step: Name of the step to run
            state: Current pipeline state

        Returns:
            Updated state after running the step
        """
        step_methods = {
            "ingest": self._ingest_node,
            "chunk": self._chunk_node,
            "extract_entities": self._extract_entities_node,
            "extract_relations": self._extract_relations_node,
            "embed": self._embed_node,
            "load": self._load_node,
        }

        if step not in step_methods:
            raise ValueError(f"Unknown step: {step}")

        method = step_methods[step]
        update = await method(state)

        # Merge update into state
        merged: SDDIPipelineState = {**state, **update}  # type: ignore[typeddict-item]
        return merged

    def get_last_delta_report(self) -> DeltaReport | None:
        """
        Get the delta report from the last incremental loading operation.

        Returns:
            DeltaReport if incremental loading was used, None otherwise
        """
        return self._last_delta_report

    def is_incremental_enabled(self) -> bool:
        """Check if incremental loading is enabled."""
        return self._use_incremental and self._incremental_loader is not None

    async def detect_changes_only(
        self,
        chunks: list[TextChunk],
        entities: list,
        relations: list,
        embeddings: dict[str, list[float]] | None = None,
    ) -> DeltaReport | None:
        """
        Detect changes without applying them (dry run).

        Useful for previewing what would be updated.

        Args:
            chunks: Text chunks
            entities: Extracted entities
            relations: Extracted relations
            embeddings: Entity embeddings

        Returns:
            DeltaReport with detected changes, or None if incremental not enabled
        """
        if not self._incremental_loader:
            logger.warning("Incremental loader not enabled, cannot detect changes")
            return None

        return await self._incremental_loader.detect_changes(
            entities=entities,
            relations=relations,
            chunks=chunks,
            embeddings=embeddings,
        )

    # =========================================================================
    # Resilient Extractor Access Methods
    # =========================================================================

    def is_resilient_extractor_enabled(self) -> bool:
        """Check if resilient extractor is enabled."""
        return self._use_resilient and self._resilient_entity_extractor is not None

    def get_extraction_checkpoint(self) -> ProcessingCheckpoint | None:
        """
        Get the current extraction checkpoint for resuming.

        Returns:
            ProcessingCheckpoint if available, None otherwise
        """
        return self._entity_checkpoint

    def get_circuit_breaker_state(self) -> str:
        """
        Get the current circuit breaker state.

        Returns:
            State string: 'closed', 'open', or 'half_open'
        """
        if self._resilient_entity_extractor:
            return self._resilient_entity_extractor.get_circuit_breaker_state().value
        return "not_available"

    def get_dead_letter_queue_items(self) -> list[dict]:
        """
        Get items in the dead letter queue.

        Returns:
            List of failed item dictionaries
        """
        if self._resilient_entity_extractor:
            dlq = self._resilient_entity_extractor.get_dead_letter_queue()
            return [item.to_dict() for item in dlq.get_all()]
        return []

    async def retry_failed_extractions(self) -> list:
        """
        Retry items in the dead letter queue.

        Returns:
            List of entities extracted from retried items
        """
        if self._resilient_entity_extractor:
            return await self._resilient_entity_extractor.retry_dlq_items()
        return []

    def get_extraction_metrics(self) -> dict:
        """
        Get detailed extraction metrics.

        Returns:
            Dictionary with extraction metrics
        """
        if self._entity_metrics:
            return self._entity_metrics.to_dict()
        return {}

    async def run_with_resume(
        self,
        documents: list,
        pipeline_id: str | None = None,
        resume_from_checkpoint: bool = True,
    ) -> SDDIPipelineState:
        """
        Run the pipeline with automatic checkpoint resume support.

        If a checkpoint exists for this pipeline, it will resume from where
        it left off. Useful for recovering from failures.

        Args:
            documents: List of documents to process
            pipeline_id: Pipeline identifier (required for resume)
            resume_from_checkpoint: Whether to attempt resume (default: True)

        Returns:
            Final pipeline state
        """
        if resume_from_checkpoint and pipeline_id and self._entity_checkpoint:
            if self._entity_checkpoint.pipeline_id == pipeline_id:
                logger.info(
                    "Resuming from checkpoint",
                    pipeline_id=pipeline_id,
                    processed=len(self._entity_checkpoint.processed_indices),
                    progress=f"{self._entity_checkpoint.progress_percent:.1f}%",
                )

        return await self.run(documents, pipeline_id=pipeline_id)


def create_sddi_pipeline(
    llm: BaseChatModel,
    embeddings: Embeddings,
    **kwargs: Any,
) -> SDDIPipeline:
    """
    Factory function to create an SDDI pipeline.

    Args:
        llm: LangChain chat model
        embeddings: Embedding model
        **kwargs: Additional pipeline configuration

    Returns:
        Configured SDDIPipeline instance
    """
    return SDDIPipeline(llm=llm, embeddings=embeddings, **kwargs)

"""
SDDI Pipeline Module.

Schema-Driven Data Integration pipeline for ontology processing.
Implements Semantic Layer → Kinetic Layer → Dynamic Layer architecture.

Pipeline Stages:
1. Ingest: Accept raw documents
2. Chunk: Split into text segments
3. Extract Entities: LLM-based NER
4. Extract Relations: Relation extraction
5. Embed: Vector embeddings
6. Load: Persist to Neo4j

Reliability Features (NEW):
- Dead-Letter Queue (DLQ) for failed documents
- Retry mechanisms with exponential backoff
- Per-document latency tracking (mean, p95, p99)
- Document size validation and limits
"""

from src.sddi.extractors import EntityExtractor, RelationExtractor
from src.sddi.loaders import Neo4jLoader
from src.sddi.pipeline import SDDIPipeline, create_sddi_pipeline
from src.sddi.state import (
    EmbeddingResult,
    # Enums
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    LoadResult,
    LoadStatus,
    # Models
    RawDocument,
    # State
    SDDIPipelineState,
    TextChunk,
    Triplet,
)

# Reliable Pipeline (NEW)
from src.sddi.reliable_pipeline import (
    ReliablePipeline,
    ReliabilityConfig,
    ReliablePipelineResult,
    DocumentProcessingResult,
    create_reliable_pipeline,
    create_production_pipeline,
    create_development_pipeline,
)

# Reliability Components (NEW)
from src.sddi.reliability import (
    # Dead-Letter Queue
    DeadLetterQueue,
    DLQEntry,
    DLQStatus,
    InMemoryDLQ,
    FileDLQ,
    # Retry
    RetryConfig,
    RetryHandler,
    RetryableError,
    NonRetryableError,
    with_retry,
    # Metrics
    DocumentMetrics,
    PipelineMetricsCollector,
    LatencyStats,
    # Validation
    DocumentValidator,
    ValidationConfig,
    ValidationResult,
    SizeExceededError,
)

__all__ = [
    # Enums
    "EntityType",
    "LoadStatus",
    # Models
    "RawDocument",
    "TextChunk",
    "ExtractedEntity",
    "ExtractedRelation",
    "Triplet",
    "EmbeddingResult",
    "LoadResult",
    # State
    "SDDIPipelineState",
    # Pipeline (Legacy)
    "SDDIPipeline",
    "create_sddi_pipeline",
    # Reliable Pipeline (NEW)
    "ReliablePipeline",
    "ReliabilityConfig",
    "ReliablePipelineResult",
    "DocumentProcessingResult",
    "create_reliable_pipeline",
    "create_production_pipeline",
    "create_development_pipeline",
    # Reliability - DLQ
    "DeadLetterQueue",
    "DLQEntry",
    "DLQStatus",
    "InMemoryDLQ",
    "FileDLQ",
    # Reliability - Retry
    "RetryConfig",
    "RetryHandler",
    "RetryableError",
    "NonRetryableError",
    "with_retry",
    # Reliability - Metrics
    "DocumentMetrics",
    "PipelineMetricsCollector",
    "LatencyStats",
    # Reliability - Validation
    "DocumentValidator",
    "ValidationConfig",
    "ValidationResult",
    "SizeExceededError",
    # Components
    "EntityExtractor",
    "RelationExtractor",
    "Neo4jLoader",
]

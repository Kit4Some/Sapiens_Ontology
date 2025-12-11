"""
Document Validation and Size Limits.

Provides input validation for the pipeline:
- Document size limits
- Content validation
- Format validation
- Resource estimation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ValidationErrorType(str, Enum):
    """Types of validation errors."""

    SIZE_EXCEEDED = "size_exceeded"
    EMPTY_CONTENT = "empty_content"
    INVALID_FORMAT = "invalid_format"
    MISSING_FIELD = "missing_field"
    INVALID_ENCODING = "invalid_encoding"
    TOO_MANY_DOCUMENTS = "too_many_documents"


class SizeExceededError(Exception):
    """Raised when document size exceeds limits."""

    def __init__(
        self,
        message: str,
        actual_size: int,
        max_size: int,
        document_id: str | None = None,
    ):
        super().__init__(message)
        self.actual_size = actual_size
        self.max_size = max_size
        self.document_id = document_id


@dataclass
class ValidationConfig:
    """Configuration for document validation."""

    # Size limits
    max_document_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_document_size_chars: int = 5_000_000          # 5M characters
    max_documents_per_batch: int = 100
    max_total_batch_size_bytes: int = 100 * 1024 * 1024  # 100 MB

    # Content requirements
    min_content_length: int = 10
    max_empty_ratio: float = 0.9  # Max 90% whitespace

    # Chunk size estimation
    estimated_chunk_size: int = 1000
    estimated_entities_per_chunk: int = 5
    estimated_relations_per_entity: float = 1.5

    # Resource limits
    max_estimated_chunks: int = 10000
    max_estimated_entities: int = 50000
    max_estimated_relations: int = 75000
    max_estimated_embeddings: int = 60000

    # Warnings
    warn_size_bytes: int = 1 * 1024 * 1024  # 1 MB warning threshold
    warn_chunk_count: int = 500


@dataclass
class ValidationResult:
    """Result of document validation."""

    valid: bool = True
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Document info
    document_id: str = ""
    size_bytes: int = 0
    size_chars: int = 0

    # Estimates
    estimated_chunks: int = 0
    estimated_entities: int = 0
    estimated_relations: int = 0
    estimated_embeddings: int = 0

    # Resource estimates
    estimated_llm_tokens: int = 0
    estimated_embedding_calls: int = 0
    estimated_processing_time_seconds: float = 0.0

    def add_error(
        self,
        error_type: ValidationErrorType,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a validation error."""
        self.valid = False
        self.errors.append({
            "type": error_type.value,
            "message": message,
            "details": details or {},
        })

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "document_id": self.document_id,
            "size_bytes": self.size_bytes,
            "size_chars": self.size_chars,
            "estimates": {
                "chunks": self.estimated_chunks,
                "entities": self.estimated_entities,
                "relations": self.estimated_relations,
                "embeddings": self.estimated_embeddings,
                "llm_tokens": self.estimated_llm_tokens,
                "embedding_calls": self.estimated_embedding_calls,
                "processing_time_seconds": round(self.estimated_processing_time_seconds, 2),
            },
        }


@dataclass
class BatchValidationResult:
    """Result of batch validation."""

    valid: bool = True
    document_results: list[ValidationResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Batch totals
    total_documents: int = 0
    valid_documents: int = 0
    invalid_documents: int = 0
    total_size_bytes: int = 0
    total_size_chars: int = 0

    # Aggregate estimates
    total_estimated_chunks: int = 0
    total_estimated_entities: int = 0
    total_estimated_relations: int = 0
    total_estimated_embeddings: int = 0
    total_estimated_processing_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": {
                "total_documents": self.total_documents,
                "valid_documents": self.valid_documents,
                "invalid_documents": self.invalid_documents,
                "total_size_bytes": self.total_size_bytes,
                "total_size_chars": self.total_size_chars,
            },
            "estimates": {
                "chunks": self.total_estimated_chunks,
                "entities": self.total_estimated_entities,
                "relations": self.total_estimated_relations,
                "embeddings": self.total_estimated_embeddings,
                "processing_time_seconds": round(self.total_estimated_processing_time, 2),
            },
            "documents": [r.to_dict() for r in self.document_results],
        }


class DocumentValidator:
    """
    Validates documents before pipeline processing.

    Checks:
    - Size limits (bytes and characters)
    - Content requirements (non-empty, minimal whitespace)
    - Resource estimates (chunks, entities, embeddings)
    - Batch limits
    """

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def validate_document(
        self,
        document_id: str,
        content: str,
        source: str = "",
    ) -> ValidationResult:
        """
        Validate a single document.

        Args:
            document_id: Document identifier
            content: Document content
            source: Document source

        Returns:
            ValidationResult with errors, warnings, and estimates
        """
        result = ValidationResult(document_id=document_id)

        # Calculate sizes
        content_bytes = content.encode("utf-8")
        result.size_bytes = len(content_bytes)
        result.size_chars = len(content)

        # Check size limits
        if result.size_bytes > self.config.max_document_size_bytes:
            result.add_error(
                ValidationErrorType.SIZE_EXCEEDED,
                f"Document size ({result.size_bytes:,} bytes) exceeds maximum "
                f"({self.config.max_document_size_bytes:,} bytes)",
                {
                    "actual_bytes": result.size_bytes,
                    "max_bytes": self.config.max_document_size_bytes,
                },
            )

        if result.size_chars > self.config.max_document_size_chars:
            result.add_error(
                ValidationErrorType.SIZE_EXCEEDED,
                f"Document length ({result.size_chars:,} chars) exceeds maximum "
                f"({self.config.max_document_size_chars:,} chars)",
                {
                    "actual_chars": result.size_chars,
                    "max_chars": self.config.max_document_size_chars,
                },
            )

        # Check content
        if not content or not content.strip():
            result.add_error(
                ValidationErrorType.EMPTY_CONTENT,
                "Document content is empty",
            )
        elif len(content.strip()) < self.config.min_content_length:
            result.add_error(
                ValidationErrorType.EMPTY_CONTENT,
                f"Document content too short (minimum {self.config.min_content_length} chars)",
            )

        # Check whitespace ratio
        if content:
            non_whitespace = len(content.replace(" ", "").replace("\n", "").replace("\t", ""))
            whitespace_ratio = 1 - (non_whitespace / len(content))
            if whitespace_ratio > self.config.max_empty_ratio:
                result.add_warning(
                    f"Document has high whitespace ratio ({whitespace_ratio:.1%})"
                )

        # Calculate estimates
        self._calculate_estimates(result)

        # Check resource estimates
        if result.estimated_chunks > self.config.max_estimated_chunks:
            result.add_error(
                ValidationErrorType.SIZE_EXCEEDED,
                f"Estimated chunks ({result.estimated_chunks:,}) exceeds maximum "
                f"({self.config.max_estimated_chunks:,})",
            )

        # Add size warning
        if result.size_bytes > self.config.warn_size_bytes:
            result.add_warning(
                f"Large document ({result.size_bytes / 1024 / 1024:.1f} MB) - "
                f"processing may take longer"
            )

        if result.estimated_chunks > self.config.warn_chunk_count:
            result.add_warning(
                f"Document will produce ~{result.estimated_chunks} chunks - "
                f"consider splitting"
            )

        if not result.valid:
            logger.warning(
                "Document validation failed",
                document_id=document_id,
                errors=len(result.errors),
                size_bytes=result.size_bytes,
            )

        return result

    def validate_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> BatchValidationResult:
        """
        Validate a batch of documents.

        Args:
            documents: List of document dicts with 'id', 'content', 'source'

        Returns:
            BatchValidationResult with per-document and aggregate results
        """
        result = BatchValidationResult()
        result.total_documents = len(documents)

        # Check batch size
        if len(documents) > self.config.max_documents_per_batch:
            result.valid = False
            result.errors.append({
                "type": ValidationErrorType.TOO_MANY_DOCUMENTS.value,
                "message": f"Batch size ({len(documents)}) exceeds maximum "
                          f"({self.config.max_documents_per_batch})",
            })
            return result

        # Validate each document
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            content = doc.get("content", "")
            source = doc.get("source", "")

            doc_result = self.validate_document(doc_id, content, source)
            result.document_results.append(doc_result)

            if doc_result.valid:
                result.valid_documents += 1
            else:
                result.invalid_documents += 1
                result.valid = False

            # Aggregate sizes
            result.total_size_bytes += doc_result.size_bytes
            result.total_size_chars += doc_result.size_chars

            # Aggregate estimates
            result.total_estimated_chunks += doc_result.estimated_chunks
            result.total_estimated_entities += doc_result.estimated_entities
            result.total_estimated_relations += doc_result.estimated_relations
            result.total_estimated_embeddings += doc_result.estimated_embeddings
            result.total_estimated_processing_time += doc_result.estimated_processing_time_seconds

        # Check total batch size
        if result.total_size_bytes > self.config.max_total_batch_size_bytes:
            result.valid = False
            result.errors.append({
                "type": ValidationErrorType.SIZE_EXCEEDED.value,
                "message": f"Total batch size ({result.total_size_bytes / 1024 / 1024:.1f} MB) "
                          f"exceeds maximum ({self.config.max_total_batch_size_bytes / 1024 / 1024:.1f} MB)",
            })

        # Check aggregate estimates
        if result.total_estimated_entities > self.config.max_estimated_entities:
            result.warnings.append(
                f"Batch will produce ~{result.total_estimated_entities:,} entities - "
                f"consider processing in smaller batches"
            )

        if result.total_estimated_processing_time > 300:  # 5 minutes
            result.warnings.append(
                f"Estimated processing time: {result.total_estimated_processing_time / 60:.1f} minutes"
            )

        logger.info(
            "Batch validation completed",
            total=result.total_documents,
            valid=result.valid_documents,
            invalid=result.invalid_documents,
            total_size_mb=result.total_size_bytes / 1024 / 1024,
        )

        return result

    def _calculate_estimates(self, result: ValidationResult) -> None:
        """Calculate processing estimates for a document."""

        # Estimate chunks based on size
        result.estimated_chunks = max(
            1,
            result.size_chars // self.config.estimated_chunk_size
        )

        # Estimate entities (based on chunk count)
        result.estimated_entities = (
            result.estimated_chunks * self.config.estimated_entities_per_chunk
        )

        # Estimate relations
        result.estimated_relations = int(
            result.estimated_entities * self.config.estimated_relations_per_entity
        )

        # Estimate embeddings (chunks + entities)
        result.estimated_embeddings = result.estimated_chunks + result.estimated_entities

        # Estimate LLM tokens (rough: ~1.5 tokens per character for input)
        result.estimated_llm_tokens = int(result.size_chars * 1.5)

        # Estimate embedding API calls (batch of 100)
        result.estimated_embedding_calls = max(1, result.estimated_embeddings // 100)

        # Estimate processing time (rough)
        # - Chunking: ~0.01s per chunk
        # - Entity extraction: ~0.5s per chunk (LLM)
        # - Relation extraction: ~0.3s per chunk (LLM)
        # - Embedding: ~0.1s per batch
        # - Loading: ~0.01s per item
        chunk_time = result.estimated_chunks * 0.01
        entity_time = result.estimated_chunks * 0.5
        relation_time = result.estimated_chunks * 0.3
        embed_time = result.estimated_embedding_calls * 0.1
        load_time = (result.estimated_chunks + result.estimated_entities + result.estimated_relations) * 0.01

        result.estimated_processing_time_seconds = (
            chunk_time + entity_time + relation_time + embed_time + load_time
        )

    def estimate_resources(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Estimate resources needed to process documents.

        Returns detailed resource estimates without full validation.
        """
        batch_result = self.validate_batch(documents)

        return {
            "documents": batch_result.total_documents,
            "total_size_mb": round(batch_result.total_size_bytes / 1024 / 1024, 2),
            "estimated_chunks": batch_result.total_estimated_chunks,
            "estimated_entities": batch_result.total_estimated_entities,
            "estimated_relations": batch_result.total_estimated_relations,
            "estimated_embeddings": batch_result.total_estimated_embeddings,
            "estimated_llm_tokens": sum(
                r.estimated_llm_tokens for r in batch_result.document_results
            ),
            "estimated_processing_time_minutes": round(
                batch_result.total_estimated_processing_time / 60, 2
            ),
            "warnings": batch_result.warnings,
        }

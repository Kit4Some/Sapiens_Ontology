"""
Integration Validation Framework.

Provides end-to-end validation for the Ontology-Based Knowledge Graph Reasoning System:
- Dynamic Context Extraction (query classification, complexity assessment)
- Document Ingestion Validation
- Entity/Relation Extraction Validation
- Query Processing Validation
- Evidence Retrieval Validation
- Answer Generation Validation
- End-to-End Pipeline Validation
"""

from src.validation.context_extractor import (
    ContextExtractor,
    QueryType,
    Complexity,
    DocumentContext,
    QueryContext,
    ValidationContext,
)
from src.validation.pipeline_validator import (
    PipelineValidator,
    ValidationStep,
    StepResult,
    ValidationReport,
)
from src.validation.step_validators import (
    IngestionValidator,
    EntityExtractionValidator,
    RelationExtractionValidator,
    QueryProcessingValidator,
    RetrievalValidator,
    ResponseValidator,
)

__all__ = [
    # Context extraction
    "ContextExtractor",
    "QueryType",
    "Complexity",
    "DocumentContext",
    "QueryContext",
    "ValidationContext",
    # Pipeline validation
    "PipelineValidator",
    "ValidationStep",
    "StepResult",
    "ValidationReport",
    # Step validators
    "IngestionValidator",
    "EntityExtractionValidator",
    "RelationExtractionValidator",
    "QueryProcessingValidator",
    "RetrievalValidator",
    "ResponseValidator",
]

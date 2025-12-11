"""
Step Validators for Integration Validation Framework.

Provides individual validators for each pipeline step:
- IngestionValidator: Document ingestion to Neo4j
- EntityExtractionValidator: Entity extraction quality
- RelationExtractionValidator: Relation extraction and cardinality
- QueryProcessingValidator: Query processing pipeline
- RetrievalValidator: Evidence retrieval coverage
- ResponseValidator: Answer generation quality
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.validation.context_extractor import (
    ValidationContext,
    DocumentContext,
    QueryContext,
    Domain,
    DOMAIN_ENTITY_TYPES,
    DOMAIN_RELATION_TYPES,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Base Classes and Enums
# =============================================================================


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationMetric:
    """A single validation metric."""
    name: str
    expected: Any
    actual: Any
    passed: bool
    message: str = ""
    severity: str = "error"  # "error", "warning", "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class StepValidationResult:
    """Result of a single step validation."""
    step_name: str
    status: ValidationStatus
    metrics: list[ValidationMetric] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED

    @property
    def failed_metrics(self) -> list[ValidationMetric]:
        return [m for m in self.metrics if not m.passed and m.severity == "error"]

    @property
    def warning_metrics(self) -> list[ValidationMetric]:
        return [m for m in self.metrics if not m.passed and m.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "passed": self.passed,
            "metrics": [m.to_dict() for m in self.metrics],
            "failed_count": len(self.failed_metrics),
            "warning_count": len(self.warning_metrics),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "details": self.details,
            "errors": self.errors,
        }


class BaseValidator(ABC):
    """Abstract base class for step validators."""

    def __init__(self, context: ValidationContext | None = None):
        self.context = context
        self._start_time: float = 0.0

    @property
    @abstractmethod
    def step_name(self) -> str:
        """Name of this validation step."""
        pass

    @abstractmethod
    async def validate(self, data: dict[str, Any]) -> StepValidationResult:
        """
        Perform validation on the provided data.

        Args:
            data: Data to validate (specific to each step)

        Returns:
            StepValidationResult with metrics and status
        """
        pass

    def _create_result(
        self,
        metrics: list[ValidationMetric],
        details: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ) -> StepValidationResult:
        """Create a validation result from metrics."""
        # Determine status based on metrics
        has_failures = any(not m.passed and m.severity == "error" for m in metrics)
        has_warnings = any(not m.passed and m.severity == "warning" for m in metrics)

        if has_failures:
            status = ValidationStatus.FAILED
        elif has_warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED

        return StepValidationResult(
            step_name=self.step_name,
            status=status,
            metrics=metrics,
            details=details or {},
            errors=errors or [],
        )


# =============================================================================
# Step 1: Ingestion Validator
# =============================================================================


class IngestionValidator(BaseValidator):
    """
    Validates document ingestion results.

    Checks:
    - Document was successfully loaded
    - Chunks were created with proper structure
    - Embeddings were generated
    - Metadata was preserved
    """

    @property
    def step_name(self) -> str:
        return "document_ingestion"

    async def validate(self, data: dict[str, Any]) -> StepValidationResult:
        """
        Validate ingestion results.

        Expected data format:
        {
            "doc_id": str,
            "status": str,  # "COMPLETED", "FAILED", etc.
            "chunk_count": int,
            "chunks_with_embeddings": int,
            "total_chars": int,
            "metadata": dict,
            "neo4j_diagnostics": dict,  # Optional: from get_data_diagnostics()
        }
        """
        metrics: list[ValidationMetric] = []
        details: dict[str, Any] = {}

        # Get context-aware thresholds
        doc_ctx = self.context.document if self.context else None
        min_chunks = self._get_min_chunks(doc_ctx)
        min_embedding_coverage = 0.95 if self.context and self.context.strict_mode else 0.90

        # 1. Check completion status
        status = data.get("status", "UNKNOWN")
        metrics.append(ValidationMetric(
            name="ingestion_status",
            expected="COMPLETED",
            actual=status,
            passed=status == "COMPLETED",
            message="Document ingestion completed successfully" if status == "COMPLETED" else f"Ingestion failed with status: {status}",
        ))

        # 2. Check chunk count
        chunk_count = data.get("chunk_count", 0)
        metrics.append(ValidationMetric(
            name="chunk_count",
            expected=f">= {min_chunks}",
            actual=chunk_count,
            passed=chunk_count >= min_chunks,
            message=f"Created {chunk_count} chunks (min: {min_chunks})",
            severity="error" if chunk_count == 0 else "warning" if chunk_count < min_chunks else "info",
        ))

        # 3. Check embedding coverage
        chunks_with_embeddings = data.get("chunks_with_embeddings", 0)
        embedding_coverage = chunks_with_embeddings / max(chunk_count, 1)
        metrics.append(ValidationMetric(
            name="embedding_coverage",
            expected=f">= {min_embedding_coverage * 100:.0f}%",
            actual=f"{embedding_coverage * 100:.1f}%",
            passed=embedding_coverage >= min_embedding_coverage,
            message=f"{chunks_with_embeddings}/{chunk_count} chunks have embeddings",
        ))

        # 4. Check metadata preservation
        metadata = data.get("metadata", {})
        has_source = "source" in metadata or "file_path" in metadata
        metrics.append(ValidationMetric(
            name="metadata_preserved",
            expected=True,
            actual=has_source,
            passed=has_source,
            message="Source metadata preserved" if has_source else "Missing source metadata",
            severity="warning",
        ))

        # 5. Check Neo4j diagnostics if available
        neo4j_diag = data.get("neo4j_diagnostics", {})
        if neo4j_diag:
            counts = neo4j_diag.get("counts", {})
            details["neo4j_counts"] = counts

            # Verify chunks in database
            db_chunk_count = counts.get("chunks", 0)
            if isinstance(db_chunk_count, int):
                metrics.append(ValidationMetric(
                    name="chunks_in_database",
                    expected=chunk_count,
                    actual=db_chunk_count,
                    passed=db_chunk_count >= chunk_count,
                    message=f"Database has {db_chunk_count} chunks",
                    severity="warning",
                ))

        # Add document context details
        if doc_ctx:
            details["document"] = {
                "name": doc_ctx.name,
                "format": doc_ctx.format.value,
                "domain": doc_ctx.domain.value,
                "size_chars": doc_ctx.size_chars,
            }

        return self._create_result(metrics, details)

    def _get_min_chunks(self, doc_ctx: DocumentContext | None) -> int:
        """Get minimum expected chunks based on document context."""
        if not doc_ctx:
            return 5

        # Base minimum on document size
        if doc_ctx.size_chars < 1000:
            return 1
        elif doc_ctx.size_chars < 5000:
            return 3
        elif doc_ctx.size_chars < 20000:
            return 10
        else:
            return 20


# =============================================================================
# Step 2: Entity Extraction Validator
# =============================================================================


class EntityExtractionValidator(BaseValidator):
    """
    Validates entity extraction results.

    Checks:
    - Entity count within expected range
    - Entity types match expected domain types
    - Entities have proper structure (id, name, type)
    - Entity confidence scores
    - Duplicate detection
    """

    @property
    def step_name(self) -> str:
        return "entity_extraction"

    async def validate(self, data: dict[str, Any]) -> StepValidationResult:
        """
        Validate entity extraction results.

        Expected data format:
        {
            "entities": list[dict],  # List of extracted entities
            "entity_count": int,
            "entity_types": dict[str, int],  # type -> count
            "avg_confidence": float,
            "duplicates_removed": int,
        }
        """
        metrics: list[ValidationMetric] = []
        details: dict[str, Any] = {}

        # Get context-aware expectations
        doc_ctx = self.context.document if self.context else None
        expected_types = self._get_expected_types(doc_ctx)
        min_entities = 3 if self.context and self.context.strict_mode else 1

        entities = data.get("entities", [])
        entity_count = data.get("entity_count", len(entities))
        entity_types = data.get("entity_types", {})

        # 1. Check entity count
        metrics.append(ValidationMetric(
            name="entity_count",
            expected=f">= {min_entities}",
            actual=entity_count,
            passed=entity_count >= min_entities,
            message=f"Extracted {entity_count} entities",
            severity="error" if entity_count == 0 else "warning" if entity_count < min_entities else "info",
        ))

        # 2. Check entity type coverage
        if expected_types and entity_types:
            matched_types = set(entity_types.keys()) & set(expected_types)
            type_coverage = len(matched_types) / max(len(expected_types), 1)
            metrics.append(ValidationMetric(
                name="entity_type_coverage",
                expected=f">= 30% of expected types",
                actual=f"{type_coverage * 100:.1f}% ({len(matched_types)}/{len(expected_types)})",
                passed=type_coverage >= 0.3 or entity_count > 0,
                message=f"Matched types: {list(matched_types)[:5]}",
                severity="warning",
            ))
            details["expected_types"] = expected_types
            details["actual_types"] = list(entity_types.keys())

        # 3. Check entity structure
        valid_structure_count = 0
        for entity in entities[:50]:  # Sample first 50
            has_required = all(k in entity for k in ["id", "name", "type"])
            if has_required:
                valid_structure_count += 1

        sample_size = min(50, len(entities))
        if sample_size > 0:
            structure_validity = valid_structure_count / sample_size
            metrics.append(ValidationMetric(
                name="entity_structure_valid",
                expected="100%",
                actual=f"{structure_validity * 100:.1f}%",
                passed=structure_validity >= 0.95,
                message=f"{valid_structure_count}/{sample_size} entities have valid structure",
            ))

        # 4. Check average confidence
        avg_confidence = data.get("avg_confidence", 0.0)
        if avg_confidence > 0:
            min_confidence = 0.6 if self.context and self.context.strict_mode else 0.5
            metrics.append(ValidationMetric(
                name="avg_confidence",
                expected=f">= {min_confidence}",
                actual=f"{avg_confidence:.2f}",
                passed=avg_confidence >= min_confidence,
                message=f"Average entity confidence: {avg_confidence:.2f}",
                severity="warning",
            ))

        # 5. Report duplicates
        duplicates_removed = data.get("duplicates_removed", 0)
        if duplicates_removed > 0:
            metrics.append(ValidationMetric(
                name="duplicates_handled",
                expected="< 20% of total",
                actual=duplicates_removed,
                passed=duplicates_removed < entity_count * 0.2,
                message=f"{duplicates_removed} duplicate entities removed",
                severity="info",
            ))

        details["entity_type_distribution"] = entity_types

        return self._create_result(metrics, details)

    def _get_expected_types(self, doc_ctx: DocumentContext | None) -> list[str]:
        """Get expected entity types based on document domain."""
        if not doc_ctx:
            return DOMAIN_ENTITY_TYPES[Domain.GENERAL]
        return doc_ctx.expected_entity_types or DOMAIN_ENTITY_TYPES.get(
            doc_ctx.domain, DOMAIN_ENTITY_TYPES[Domain.GENERAL]
        )


# =============================================================================
# Step 3: Relation Extraction Validator
# =============================================================================


class RelationExtractionValidator(BaseValidator):
    """
    Validates relation extraction results.

    Checks:
    - Relation count appropriate for entity count
    - Relation types match expected domain predicates
    - Source/target entities exist
    - Cardinality constraints (if defined)
    - Relation confidence scores
    """

    @property
    def step_name(self) -> str:
        return "relation_extraction"

    async def validate(self, data: dict[str, Any]) -> StepValidationResult:
        """
        Validate relation extraction results.

        Expected data format:
        {
            "relations": list[dict],  # List of extracted relations
            "relation_count": int,
            "entity_count": int,  # For ratio calculation
            "predicate_types": dict[str, int],  # predicate -> count
            "avg_confidence": float,
            "orphan_relations": int,  # Relations with missing entities
            "cardinality_violations": list[dict],  # Optional
        }
        """
        metrics: list[ValidationMetric] = []
        details: dict[str, Any] = {}

        # Get context-aware expectations
        doc_ctx = self.context.document if self.context else None
        expected_predicates = self._get_expected_predicates(doc_ctx)

        relations = data.get("relations", [])
        relation_count = data.get("relation_count", len(relations))
        entity_count = data.get("entity_count", 0)
        predicate_types = data.get("predicate_types", {})

        # 1. Check relation count relative to entities
        # Heuristic: expect 0.5-3 relations per entity
        min_ratio = 0.3
        max_ratio = 5.0
        if entity_count > 0:
            ratio = relation_count / entity_count
            metrics.append(ValidationMetric(
                name="relation_entity_ratio",
                expected=f"{min_ratio}-{max_ratio} relations per entity",
                actual=f"{ratio:.2f}",
                passed=min_ratio <= ratio <= max_ratio,
                message=f"{relation_count} relations for {entity_count} entities",
                severity="warning",
            ))

        # 2. Check predicate type coverage
        if expected_predicates and predicate_types:
            # Normalize predicate names for comparison
            normalized_expected = {p.upper().replace(" ", "_") for p in expected_predicates}
            normalized_actual = {p.upper().replace(" ", "_") for p in predicate_types.keys()}

            matched_predicates = normalized_actual & normalized_expected
            coverage = len(matched_predicates) / max(len(expected_predicates), 1)

            metrics.append(ValidationMetric(
                name="predicate_type_coverage",
                expected=f">= 20% of expected predicates",
                actual=f"{coverage * 100:.1f}%",
                passed=coverage >= 0.2 or relation_count > 0,
                message=f"Matched predicates: {list(matched_predicates)[:5]}",
                severity="warning",
            ))
            details["expected_predicates"] = expected_predicates
            details["actual_predicates"] = list(predicate_types.keys())

        # 3. Check for orphan relations
        orphan_relations = data.get("orphan_relations", 0)
        if relation_count > 0:
            orphan_ratio = orphan_relations / relation_count
            metrics.append(ValidationMetric(
                name="orphan_relations",
                expected="0%",
                actual=f"{orphan_ratio * 100:.1f}% ({orphan_relations})",
                passed=orphan_relations == 0,
                message="Relations without valid source/target entities",
                severity="warning" if orphan_ratio < 0.1 else "error",
            ))

        # 4. Check relation structure
        valid_structure_count = 0
        for rel in relations[:50]:  # Sample first 50
            has_required = all(k in rel for k in ["source_entity", "target_entity", "predicate"])
            if has_required:
                valid_structure_count += 1

        sample_size = min(50, len(relations))
        if sample_size > 0:
            structure_validity = valid_structure_count / sample_size
            metrics.append(ValidationMetric(
                name="relation_structure_valid",
                expected="100%",
                actual=f"{structure_validity * 100:.1f}%",
                passed=structure_validity >= 0.95,
                message=f"{valid_structure_count}/{sample_size} relations have valid structure",
            ))

        # 5. Check cardinality violations
        cardinality_violations = data.get("cardinality_violations", [])
        if cardinality_violations:
            metrics.append(ValidationMetric(
                name="cardinality_compliance",
                expected="No violations",
                actual=f"{len(cardinality_violations)} violations",
                passed=False,
                message=f"Cardinality constraints violated: {cardinality_violations[:3]}",
                severity="warning",
            ))
            details["cardinality_violations"] = cardinality_violations[:10]

        # 6. Average confidence
        avg_confidence = data.get("avg_confidence", 0.0)
        if avg_confidence > 0:
            min_confidence = 0.5 if self.context and self.context.strict_mode else 0.4
            metrics.append(ValidationMetric(
                name="avg_relation_confidence",
                expected=f">= {min_confidence}",
                actual=f"{avg_confidence:.2f}",
                passed=avg_confidence >= min_confidence,
                message=f"Average relation confidence: {avg_confidence:.2f}",
                severity="warning",
            ))

        details["predicate_distribution"] = predicate_types

        return self._create_result(metrics, details)

    def _get_expected_predicates(self, doc_ctx: DocumentContext | None) -> list[str]:
        """Get expected relation predicates based on document domain."""
        if not doc_ctx:
            return DOMAIN_RELATION_TYPES[Domain.GENERAL]
        return doc_ctx.expected_relation_types or DOMAIN_RELATION_TYPES.get(
            doc_ctx.domain, DOMAIN_RELATION_TYPES[Domain.GENERAL]
        )


# =============================================================================
# Step 4: Query Processing Validator
# =============================================================================


class QueryProcessingValidator(BaseValidator):
    """
    Validates query processing results.

    Checks:
    - Topic entities were extracted
    - Entities were retrieved from graph
    - Subgraph was built
    - Query evolution (if any) was reasonable
    """

    @property
    def step_name(self) -> str:
        return "query_processing"

    async def validate(self, data: dict[str, Any]) -> StepValidationResult:
        """
        Validate query processing (Constructor phase) results.

        Expected data format:
        {
            "query": str,
            "topic_entities": list[dict],  # Extracted topic entities
            "retrieved_entities": list[dict],  # Entities found in graph
            "subgraph": dict,  # Built subgraph {nodes: [], edges: []}
            "processing_time_ms": float,
        }
        """
        metrics: list[ValidationMetric] = []
        details: dict[str, Any] = {}

        # Get context-aware expectations
        query_ctx = self.context.query if self.context else None

        topic_entities = data.get("topic_entities", [])
        retrieved_entities = data.get("retrieved_entities", [])
        subgraph = data.get("subgraph", {})
        query = data.get("query", "")

        # 1. Check topic entity extraction
        min_topics = 1
        metrics.append(ValidationMetric(
            name="topic_entity_extraction",
            expected=f">= {min_topics} topic entities",
            actual=len(topic_entities),
            passed=len(topic_entities) >= min_topics,
            message=f"Extracted {len(topic_entities)} topic entities from query",
            severity="error" if len(topic_entities) == 0 else "info",
        ))

        # 2. Check entity retrieval
        # This may be 0 if it's a new knowledge base
        if topic_entities:
            retrieval_ratio = len(retrieved_entities) / len(topic_entities) if topic_entities else 0
            metrics.append(ValidationMetric(
                name="entity_retrieval",
                expected=">= 50% of topic entities found",
                actual=f"{retrieval_ratio * 100:.1f}%",
                passed=retrieval_ratio >= 0.5 or len(retrieved_entities) > 0,
                message=f"Retrieved {len(retrieved_entities)} entities from graph",
                severity="warning" if retrieval_ratio < 0.5 else "info",
            ))

        # 3. Check subgraph construction
        subgraph_nodes = subgraph.get("nodes", [])
        subgraph_edges = subgraph.get("edges", [])
        has_subgraph = len(subgraph_nodes) > 0

        metrics.append(ValidationMetric(
            name="subgraph_construction",
            expected="Non-empty subgraph if entities found",
            actual=f"{len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges",
            passed=has_subgraph or len(retrieved_entities) == 0,
            message="Subgraph built successfully" if has_subgraph else "No subgraph built",
            severity="warning" if not has_subgraph and len(retrieved_entities) > 0 else "info",
        ))

        # 4. Processing time (informational)
        processing_time = data.get("processing_time_ms", 0)
        if processing_time > 0:
            max_time = 5000  # 5 seconds
            metrics.append(ValidationMetric(
                name="processing_time",
                expected=f"< {max_time}ms",
                actual=f"{processing_time:.0f}ms",
                passed=processing_time < max_time,
                message=f"Query processing took {processing_time:.0f}ms",
                severity="warning" if processing_time > max_time else "info",
            ))

        # Add query context details
        if query_ctx:
            details["query_context"] = {
                "type": query_ctx.query_type.value,
                "complexity": query_ctx.complexity.value,
                "potential_entities": query_ctx.potential_entities,
            }

        details["topic_entities"] = [e.get("name", e) for e in topic_entities[:5]]
        details["retrieved_count"] = len(retrieved_entities)

        return self._create_result(metrics, details)


# =============================================================================
# Step 5: Retrieval Validator
# =============================================================================


class RetrievalValidator(BaseValidator):
    """
    Validates evidence retrieval results.

    Checks:
    - Evidence count meets complexity requirements
    - Evidence types are appropriate
    - Relevance scores meet threshold
    - Evidence diversity (multiple sources)
    """

    @property
    def step_name(self) -> str:
        return "evidence_retrieval"

    async def validate(self, data: dict[str, Any]) -> StepValidationResult:
        """
        Validate evidence retrieval (Retriever phase) results.

        Expected data format:
        {
            "evidence": list[dict],  # Collected evidence
            "evidence_count": int,
            "evidence_types": dict[str, int],  # type -> count
            "avg_relevance": float,
            "sufficiency_score": float,
            "iteration_count": int,
        }
        """
        metrics: list[ValidationMetric] = []
        details: dict[str, Any] = {}

        # Get context-aware expectations
        query_ctx = self.context.query if self.context else None
        min_evidence = query_ctx.min_evidence_count if query_ctx else 3
        min_relevance = query_ctx.min_relevance_score if query_ctx else 0.5

        evidence = data.get("evidence", [])
        evidence_count = data.get("evidence_count", len(evidence))
        evidence_types = data.get("evidence_types", {})
        avg_relevance = data.get("avg_relevance", 0.0)
        sufficiency_score = data.get("sufficiency_score", 0.0)

        # 1. Check evidence count
        metrics.append(ValidationMetric(
            name="evidence_count",
            expected=f">= {min_evidence}",
            actual=evidence_count,
            passed=evidence_count >= min_evidence,
            message=f"Collected {evidence_count} evidence pieces",
            severity="warning" if evidence_count < min_evidence else "info",
        ))

        # 2. Check evidence type diversity
        if evidence_types:
            type_count = len(evidence_types)
            metrics.append(ValidationMetric(
                name="evidence_diversity",
                expected=">= 2 evidence types",
                actual=f"{type_count} types",
                passed=type_count >= 2 or evidence_count < 3,
                message=f"Evidence types: {list(evidence_types.keys())}",
                severity="info",
            ))
            details["evidence_type_distribution"] = evidence_types

        # 3. Check average relevance
        if avg_relevance > 0:
            metrics.append(ValidationMetric(
                name="avg_relevance",
                expected=f">= {min_relevance}",
                actual=f"{avg_relevance:.2f}",
                passed=avg_relevance >= min_relevance,
                message=f"Average evidence relevance: {avg_relevance:.2f}",
                severity="warning" if avg_relevance < min_relevance else "info",
            ))

        # 4. Check sufficiency score
        min_sufficiency = 0.6 if self.context and self.context.strict_mode else 0.4
        metrics.append(ValidationMetric(
            name="sufficiency_score",
            expected=f">= {min_sufficiency}",
            actual=f"{sufficiency_score:.2f}",
            passed=sufficiency_score >= min_sufficiency,
            message=f"Evidence sufficiency: {sufficiency_score:.2f}",
            severity="warning" if sufficiency_score < min_sufficiency else "info",
        ))

        # 5. Check iteration count (efficiency)
        iteration_count = data.get("iteration_count", 0)
        max_iterations = 5
        if iteration_count > 0:
            metrics.append(ValidationMetric(
                name="retrieval_efficiency",
                expected=f"<= {max_iterations} iterations",
                actual=iteration_count,
                passed=iteration_count <= max_iterations,
                message=f"Retrieval completed in {iteration_count} iterations",
                severity="info",
            ))

        return self._create_result(metrics, details)


# =============================================================================
# Step 6: Response Validator
# =============================================================================


class ResponseValidator(BaseValidator):
    """
    Validates response generation results.

    Checks:
    - Answer was generated
    - Confidence meets threshold
    - Answer type is appropriate
    - Supporting evidence is cited
    - Language matches query language
    """

    @property
    def step_name(self) -> str:
        return "response_generation"

    async def validate(self, data: dict[str, Any]) -> StepValidationResult:
        """
        Validate response generation (Responser phase) results.

        Expected data format:
        {
            "answer": str,
            "answer_type": str,  # "direct", "inferred", "uncertain", "no_data"
            "confidence": float,
            "supporting_evidence": list[str],
            "reasoning_steps": int,
            "explanation": str,
            "caveats": list[str],
        }
        """
        metrics: list[ValidationMetric] = []
        details: dict[str, Any] = {}

        # Get context-aware expectations
        query_ctx = self.context.query if self.context else None
        min_confidence = query_ctx.min_confidence if query_ctx else 0.5
        expected_answer_type = query_ctx.expected_answer_type.value if query_ctx else None

        answer = data.get("answer", "")
        answer_type = data.get("answer_type", "unknown")
        confidence = data.get("confidence", 0.0)
        supporting_evidence = data.get("supporting_evidence", [])

        # 1. Check answer generation
        has_answer = bool(answer and len(answer.strip()) > 10)
        metrics.append(ValidationMetric(
            name="answer_generated",
            expected=True,
            actual=has_answer,
            passed=has_answer or answer_type == "no_data",
            message="Answer generated successfully" if has_answer else "No answer generated",
            severity="error" if not has_answer and answer_type != "no_data" else "info",
        ))

        # 2. Check answer type
        valid_types = ["direct", "inferred", "uncertain", "no_data"]
        is_valid_type = answer_type in valid_types
        metrics.append(ValidationMetric(
            name="answer_type_valid",
            expected=f"One of {valid_types}",
            actual=answer_type,
            passed=is_valid_type,
            message=f"Answer type: {answer_type}",
            severity="warning" if not is_valid_type else "info",
        ))

        # 3. Check confidence
        if answer_type not in ["no_data", "failed"]:
            metrics.append(ValidationMetric(
                name="confidence",
                expected=f">= {min_confidence}",
                actual=f"{confidence:.2f}",
                passed=confidence >= min_confidence,
                message=f"Answer confidence: {confidence:.2f}",
                severity="warning" if confidence < min_confidence else "info",
            ))

        # 4. Check evidence citation
        if answer_type in ["direct", "inferred"]:
            has_evidence = len(supporting_evidence) > 0
            metrics.append(ValidationMetric(
                name="evidence_cited",
                expected=">= 1 supporting evidence",
                actual=len(supporting_evidence),
                passed=has_evidence,
                message=f"{len(supporting_evidence)} evidence pieces cited",
                severity="warning" if not has_evidence else "info",
            ))

        # 5. Check answer length (reasonable length for the query type)
        if has_answer:
            answer_len = len(answer)
            min_len = 20
            max_len = 5000
            metrics.append(ValidationMetric(
                name="answer_length",
                expected=f"{min_len}-{max_len} chars",
                actual=answer_len,
                passed=min_len <= answer_len <= max_len,
                message=f"Answer length: {answer_len} characters",
                severity="info",
            ))

        # 6. Check explanation provided
        explanation = data.get("explanation", "")
        has_explanation = bool(explanation and len(explanation.strip()) > 0)
        metrics.append(ValidationMetric(
            name="explanation_provided",
            expected=True,
            actual=has_explanation,
            passed=has_explanation,
            message="Reasoning explanation provided" if has_explanation else "No explanation",
            severity="info",
        ))

        # Add details
        details["answer_type"] = answer_type
        details["confidence"] = confidence
        details["evidence_count"] = len(supporting_evidence)
        details["reasoning_steps"] = data.get("reasoning_steps", 0)
        details["caveats"] = data.get("caveats", [])

        return self._create_result(metrics, details)

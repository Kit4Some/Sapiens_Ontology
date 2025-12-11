"""
Pipeline Validator for Integration Validation Framework.

Orchestrates end-to-end validation of the complete pipeline:
- Document Ingestion → Entity Extraction → Relation Extraction
- Query Processing → Evidence Retrieval → Response Generation

Provides comprehensive validation reports with metrics and recommendations.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog

from src.validation.context_extractor import (
    ContextExtractor,
    ValidationContext,
)
from src.validation.step_validators import (
    BaseValidator,
    ValidationStatus,
    StepValidationResult,
    IngestionValidator,
    EntityExtractionValidator,
    RelationExtractionValidator,
    QueryProcessingValidator,
    RetrievalValidator,
    ResponseValidator,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ValidationStep(str, Enum):
    """Pipeline validation steps."""
    INGESTION = "ingestion"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    QUERY_PROCESSING = "query_processing"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    RESPONSE_GENERATION = "response_generation"


class PipelinePhase(str, Enum):
    """High-level pipeline phases."""
    INGESTION_PHASE = "ingestion"  # Ingestion + Entity + Relation
    QUERY_PHASE = "query"          # Query Processing + Retrieval + Response


@dataclass
class StepResult:
    """Result from executing a validation step."""
    step: ValidationStep
    validation_result: StepValidationResult
    execution_time_ms: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.validation_result.passed if not self.skipped else True

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.value,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "execution_time_ms": self.execution_time_ms,
            "validation": self.validation_result.to_dict() if not self.skipped else None,
        }


@dataclass
class ValidationReport:
    """Complete validation report for a pipeline run."""
    pipeline_id: str
    phase: PipelinePhase
    steps: list[StepResult] = field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.PASSED
    start_time: str = ""
    end_time: str = ""
    total_duration_ms: float = 0.0
    context: ValidationContext | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.utcnow().isoformat() + "Z"

    @property
    def passed(self) -> bool:
        return self.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]

    @property
    def failed_steps(self) -> list[StepResult]:
        return [s for s in self.steps if not s.passed and not s.skipped]

    @property
    def passed_steps(self) -> list[StepResult]:
        return [s for s in self.steps if s.passed]

    @property
    def warning_steps(self) -> list[StepResult]:
        return [
            s for s in self.steps
            if s.validation_result.status == ValidationStatus.WARNING
        ]

    def finalize(self) -> None:
        """Finalize the report after all steps complete."""
        self.end_time = datetime.utcnow().isoformat() + "Z"

        # Determine overall status
        if any(not s.passed and not s.skipped for s in self.steps):
            self.overall_status = ValidationStatus.FAILED
        elif any(s.validation_result.status == ValidationStatus.WARNING for s in self.steps if not s.skipped):
            self.overall_status = ValidationStatus.WARNING
        else:
            self.overall_status = ValidationStatus.PASSED

        # Generate summary
        self.summary = {
            "total_steps": len(self.steps),
            "passed_steps": len(self.passed_steps),
            "failed_steps": len(self.failed_steps),
            "warning_steps": len(self.warning_steps),
            "skipped_steps": len([s for s in self.steps if s.skipped]),
        }

        # Generate recommendations
        self._generate_recommendations()

    def _generate_recommendations(self) -> None:
        """Generate recommendations based on validation results."""
        recommendations = []

        for step in self.steps:
            if step.skipped:
                continue

            result = step.validation_result

            # Check failed metrics
            for metric in result.failed_metrics:
                if metric.severity == "error":
                    recommendations.append(
                        f"[{step.step.value}] Fix: {metric.name} - {metric.message}"
                    )

            # Check warning metrics
            for metric in result.warning_metrics:
                recommendations.append(
                    f"[{step.step.value}] Consider: {metric.name} - {metric.message}"
                )

        # Add phase-specific recommendations
        if self.phase == PipelinePhase.INGESTION_PHASE:
            if any(s.step == ValidationStep.ENTITY_EXTRACTION and not s.passed for s in self.steps):
                recommendations.append(
                    "Consider adjusting entity extraction prompts or confidence thresholds"
                )
            if any(s.step == ValidationStep.RELATION_EXTRACTION and not s.passed for s in self.steps):
                recommendations.append(
                    "Review relation extraction schema and predicate definitions"
                )

        elif self.phase == PipelinePhase.QUERY_PHASE:
            if any(s.step == ValidationStep.QUERY_PROCESSING and not s.passed for s in self.steps):
                recommendations.append(
                    "Ensure topic entities exist in the knowledge graph"
                )
            if any(s.step == ValidationStep.EVIDENCE_RETRIEVAL and not s.passed for s in self.steps):
                recommendations.append(
                    "Consider expanding retrieval strategies or adjusting relevance thresholds"
                )

        self.recommendations = recommendations[:10]  # Limit to top 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "phase": self.phase.value,
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "recommendations": self.recommendations,
            "context": self.context.to_dict() if self.context else None,
        }


# =============================================================================
# Pipeline Validator
# =============================================================================


class PipelineValidator:
    """
    End-to-end pipeline validator.

    Orchestrates validation of all pipeline steps and generates
    comprehensive reports with metrics and recommendations.

    Usage:
        validator = PipelineValidator()

        # Validate ingestion phase
        report = await validator.validate_ingestion(
            doc_path="/path/to/doc.pdf",
            ingestion_result={...},
            entity_result={...},
            relation_result={...},
        )

        # Validate query phase
        report = await validator.validate_query(
            query="What is X?",
            query_result={...},
            retrieval_result={...},
            response_result={...},
        )

        # Or run full end-to-end validation
        report = await validator.validate_full_pipeline(...)
    """

    def __init__(
        self,
        context_extractor: ContextExtractor | None = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the pipeline validator.

        Args:
            context_extractor: Optional custom context extractor
            strict_mode: Use stricter validation thresholds
        """
        self.context_extractor = context_extractor or ContextExtractor()
        self.strict_mode = strict_mode

        # Initialize validators (will be configured with context per run)
        self._validators: dict[ValidationStep, type[BaseValidator]] = {
            ValidationStep.INGESTION: IngestionValidator,
            ValidationStep.ENTITY_EXTRACTION: EntityExtractionValidator,
            ValidationStep.RELATION_EXTRACTION: RelationExtractionValidator,
            ValidationStep.QUERY_PROCESSING: QueryProcessingValidator,
            ValidationStep.EVIDENCE_RETRIEVAL: RetrievalValidator,
            ValidationStep.RESPONSE_GENERATION: ResponseValidator,
        }

    async def validate_ingestion(
        self,
        doc_path: str | None = None,
        doc_content: str | None = None,
        ingestion_result: dict[str, Any] | None = None,
        entity_result: dict[str, Any] | None = None,
        relation_result: dict[str, Any] | None = None,
        pipeline_id: str | None = None,
    ) -> ValidationReport:
        """
        Validate the ingestion phase of the pipeline.

        Args:
            doc_path: Path to the ingested document
            doc_content: Document content (for context extraction)
            ingestion_result: Results from ingestion step
            entity_result: Results from entity extraction
            relation_result: Results from relation extraction
            pipeline_id: Optional pipeline identifier

        Returns:
            ValidationReport with results for all ingestion steps
        """
        import uuid
        import time

        pipeline_id = pipeline_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        # Extract context
        context = self.context_extractor.extract_full_context(
            document_path=doc_path,
            document_content=doc_content,
            validation_mode="ingestion_only",
            strict_mode=self.strict_mode,
        )

        report = ValidationReport(
            pipeline_id=pipeline_id,
            phase=PipelinePhase.INGESTION_PHASE,
            context=context,
        )

        # Run step validations
        steps_data = [
            (ValidationStep.INGESTION, ingestion_result),
            (ValidationStep.ENTITY_EXTRACTION, entity_result),
            (ValidationStep.RELATION_EXTRACTION, relation_result),
        ]

        for step, data in steps_data:
            step_result = await self._run_step_validation(step, data, context)
            report.steps.append(step_result)

        # Finalize report
        report.total_duration_ms = (time.time() - start_time) * 1000
        report.finalize()

        logger.info(
            "Ingestion validation completed",
            pipeline_id=pipeline_id,
            status=report.overall_status.value,
            duration_ms=report.total_duration_ms,
        )

        return report

    async def validate_query(
        self,
        query: str,
        query_result: dict[str, Any] | None = None,
        retrieval_result: dict[str, Any] | None = None,
        response_result: dict[str, Any] | None = None,
        pipeline_id: str | None = None,
    ) -> ValidationReport:
        """
        Validate the query phase of the pipeline.

        Args:
            query: The user query
            query_result: Results from query processing (Constructor)
            retrieval_result: Results from evidence retrieval (Retriever)
            response_result: Results from response generation (Responser)
            pipeline_id: Optional pipeline identifier

        Returns:
            ValidationReport with results for all query steps
        """
        import uuid
        import time

        pipeline_id = pipeline_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        # Extract context
        context = self.context_extractor.extract_full_context(
            query=query,
            validation_mode="query_only",
            strict_mode=self.strict_mode,
        )

        report = ValidationReport(
            pipeline_id=pipeline_id,
            phase=PipelinePhase.QUERY_PHASE,
            context=context,
        )

        # Run step validations
        steps_data = [
            (ValidationStep.QUERY_PROCESSING, query_result),
            (ValidationStep.EVIDENCE_RETRIEVAL, retrieval_result),
            (ValidationStep.RESPONSE_GENERATION, response_result),
        ]

        for step, data in steps_data:
            step_result = await self._run_step_validation(step, data, context)
            report.steps.append(step_result)

        # Finalize report
        report.total_duration_ms = (time.time() - start_time) * 1000
        report.finalize()

        logger.info(
            "Query validation completed",
            pipeline_id=pipeline_id,
            status=report.overall_status.value,
            duration_ms=report.total_duration_ms,
        )

        return report

    async def validate_full_pipeline(
        self,
        doc_path: str | None = None,
        doc_content: str | None = None,
        query: str | None = None,
        ingestion_result: dict[str, Any] | None = None,
        entity_result: dict[str, Any] | None = None,
        relation_result: dict[str, Any] | None = None,
        query_result: dict[str, Any] | None = None,
        retrieval_result: dict[str, Any] | None = None,
        response_result: dict[str, Any] | None = None,
        pipeline_id: str | None = None,
    ) -> ValidationReport:
        """
        Validate the complete end-to-end pipeline.

        Args:
            doc_path: Path to the ingested document
            doc_content: Document content
            query: The user query
            ingestion_result: Results from ingestion step
            entity_result: Results from entity extraction
            relation_result: Results from relation extraction
            query_result: Results from query processing
            retrieval_result: Results from evidence retrieval
            response_result: Results from response generation
            pipeline_id: Optional pipeline identifier

        Returns:
            ValidationReport with results for all steps
        """
        import uuid
        import time

        pipeline_id = pipeline_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        # Extract full context
        context = self.context_extractor.extract_full_context(
            document_path=doc_path,
            document_content=doc_content,
            query=query,
            validation_mode="full",
            strict_mode=self.strict_mode,
        )

        # Create combined report
        report = ValidationReport(
            pipeline_id=pipeline_id,
            phase=PipelinePhase.INGESTION_PHASE,  # Will show both phases
            context=context,
        )

        # Run all step validations
        all_steps_data = [
            (ValidationStep.INGESTION, ingestion_result),
            (ValidationStep.ENTITY_EXTRACTION, entity_result),
            (ValidationStep.RELATION_EXTRACTION, relation_result),
            (ValidationStep.QUERY_PROCESSING, query_result),
            (ValidationStep.EVIDENCE_RETRIEVAL, retrieval_result),
            (ValidationStep.RESPONSE_GENERATION, response_result),
        ]

        for step, data in all_steps_data:
            step_result = await self._run_step_validation(step, data, context)
            report.steps.append(step_result)

        # Finalize report
        report.total_duration_ms = (time.time() - start_time) * 1000
        report.finalize()

        logger.info(
            "Full pipeline validation completed",
            pipeline_id=pipeline_id,
            status=report.overall_status.value,
            duration_ms=report.total_duration_ms,
        )

        return report

    async def _run_step_validation(
        self,
        step: ValidationStep,
        data: dict[str, Any] | None,
        context: ValidationContext,
    ) -> StepResult:
        """Run validation for a single step."""
        import time

        start_time = time.time()

        if data is None:
            # Skip this step
            return StepResult(
                step=step,
                validation_result=StepValidationResult(
                    step_name=step.value,
                    status=ValidationStatus.SKIPPED,
                ),
                skipped=True,
                skip_reason="No data provided",
            )

        try:
            # Create validator instance with context
            validator_class = self._validators[step]
            validator = validator_class(context=context)

            # Run validation
            result = await validator.validate(data)

            execution_time = (time.time() - start_time) * 1000

            return StepResult(
                step=step,
                validation_result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(
                "Step validation failed",
                step=step.value,
                error=str(e),
            )

            execution_time = (time.time() - start_time) * 1000

            return StepResult(
                step=step,
                validation_result=StepValidationResult(
                    step_name=step.value,
                    status=ValidationStatus.FAILED,
                    errors=[str(e)],
                ),
                execution_time_ms=execution_time,
            )


# =============================================================================
# Utility Functions
# =============================================================================


def create_pipeline_validator(
    strict_mode: bool = False,
) -> PipelineValidator:
    """
    Create a configured pipeline validator.

    Args:
        strict_mode: Use stricter validation thresholds

    Returns:
        Configured PipelineValidator instance
    """
    return PipelineValidator(strict_mode=strict_mode)


async def validate_ingestion_quick(
    ingestion_data: dict[str, Any],
    doc_path: str | None = None,
) -> dict[str, Any]:
    """
    Quick ingestion validation without full pipeline context.

    Args:
        ingestion_data: Ingestion results to validate
        doc_path: Optional document path

    Returns:
        Validation result dictionary
    """
    validator = PipelineValidator()
    report = await validator.validate_ingestion(
        doc_path=doc_path,
        ingestion_result=ingestion_data,
    )
    return report.to_dict()


async def validate_query_quick(
    query: str,
    response_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Quick query validation without full pipeline context.

    Args:
        query: The user query
        response_data: Response generation results

    Returns:
        Validation result dictionary
    """
    validator = PipelineValidator()
    report = await validator.validate_query(
        query=query,
        response_result=response_data,
    )
    return report.to_dict()

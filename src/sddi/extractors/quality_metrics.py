"""
Extraction Quality Metrics Module.

Provides comprehensive quality metrics for entity and relation extraction:
- Confidence score distribution
- Entity type distribution
- Extraction coverage metrics
- Validation against ground truth (when available)
- Anomaly detection
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from collections import defaultdict

import structlog

from src.sddi.state import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    TextChunk,
)

logger = structlog.get_logger(__name__)


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"  # 90%+
    GOOD = "good"           # 75-90%
    FAIR = "fair"           # 60-75%
    POOR = "poor"           # 40-60%
    CRITICAL = "critical"   # <40%


@dataclass
class ConfidenceMetrics:
    """Confidence score metrics."""
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    below_threshold: int = 0
    above_threshold: int = 0
    threshold: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 3),
            "median": round(self.median, 3),
            "std_dev": round(self.std_dev, 3),
            "min": round(self.min, 3),
            "max": round(self.max, 3),
            "below_threshold": self.below_threshold,
            "above_threshold": self.above_threshold,
            "high_confidence_ratio": round(
                self.above_threshold / max(self.above_threshold + self.below_threshold, 1), 3
            ),
        }


@dataclass
class TypeDistribution:
    """Entity/relation type distribution."""
    counts: dict[str, int] = field(default_factory=dict)
    percentages: dict[str, float] = field(default_factory=dict)
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "percentages": {k: round(v, 3) for k, v in self.percentages.items()},
            "total": self.total,
        }


@dataclass
class CoverageMetrics:
    """Extraction coverage metrics."""
    chunks_processed: int = 0
    chunks_with_entities: int = 0
    chunks_with_relations: int = 0
    avg_entities_per_chunk: float = 0.0
    avg_relations_per_chunk: float = 0.0
    empty_chunk_ratio: float = 0.0
    dense_chunk_ratio: float = 0.0  # >5 entities

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_with_entities": self.chunks_with_entities,
            "chunks_with_relations": self.chunks_with_relations,
            "entity_coverage": round(
                self.chunks_with_entities / max(self.chunks_processed, 1), 3
            ),
            "relation_coverage": round(
                self.chunks_with_relations / max(self.chunks_processed, 1), 3
            ),
            "avg_entities_per_chunk": round(self.avg_entities_per_chunk, 2),
            "avg_relations_per_chunk": round(self.avg_relations_per_chunk, 2),
            "empty_chunk_ratio": round(self.empty_chunk_ratio, 3),
            "dense_chunk_ratio": round(self.dense_chunk_ratio, 3),
        }


@dataclass
class AnomalyReport:
    """Detected anomalies in extraction."""
    anomaly_type: str
    severity: str  # low, medium, high
    description: str
    affected_items: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "affected_count": len(self.affected_items),
            "affected_items": self.affected_items[:10],  # First 10 only
            "recommendation": self.recommendation,
        }


@dataclass
class ValidationResult:
    """Validation against ground truth."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_entities: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1_score": round(self.f1_score, 3),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class ExtractionQualityReport:
    """Comprehensive extraction quality report."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    document_id: str = ""

    # Core metrics
    entity_count: int = 0
    relation_count: int = 0
    unique_entity_types: int = 0
    unique_predicates: int = 0

    # Detailed metrics
    entity_confidence: ConfidenceMetrics = field(default_factory=ConfidenceMetrics)
    relation_confidence: ConfidenceMetrics = field(default_factory=ConfidenceMetrics)
    entity_type_distribution: TypeDistribution = field(default_factory=TypeDistribution)
    predicate_distribution: TypeDistribution = field(default_factory=TypeDistribution)
    coverage: CoverageMetrics = field(default_factory=CoverageMetrics)

    # Quality assessment
    overall_quality: QualityLevel = QualityLevel.FAIR
    quality_score: float = 0.0
    anomalies: list[AnomalyReport] = field(default_factory=list)

    # Validation (if ground truth available)
    validation: ValidationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "document_id": self.document_id,
            "summary": {
                "entity_count": self.entity_count,
                "relation_count": self.relation_count,
                "unique_entity_types": self.unique_entity_types,
                "unique_predicates": self.unique_predicates,
                "overall_quality": self.overall_quality.value,
                "quality_score": round(self.quality_score, 2),
            },
            "entity_confidence": self.entity_confidence.to_dict(),
            "relation_confidence": self.relation_confidence.to_dict(),
            "entity_type_distribution": self.entity_type_distribution.to_dict(),
            "predicate_distribution": self.predicate_distribution.to_dict(),
            "coverage": self.coverage.to_dict(),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "validation": self.validation.to_dict() if self.validation else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class ExtractionQualityAnalyzer:
    """
    Analyzes extraction quality and generates comprehensive reports.

    Features:
    - Confidence distribution analysis
    - Entity type balance checking
    - Coverage metrics
    - Anomaly detection
    - Optional ground truth validation
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        min_entities_per_chunk: float = 0.5,
        max_entities_per_chunk: float = 20.0,
        type_imbalance_threshold: float = 0.5,  # >50% in one type is imbalanced
    ) -> None:
        """
        Initialize quality analyzer.

        Args:
            confidence_threshold: Threshold for "high confidence"
            min_entities_per_chunk: Expected minimum entities per chunk
            max_entities_per_chunk: Expected maximum entities per chunk
            type_imbalance_threshold: Threshold for type imbalance warning
        """
        self._confidence_threshold = confidence_threshold
        self._min_entities = min_entities_per_chunk
        self._max_entities = max_entities_per_chunk
        self._imbalance_threshold = type_imbalance_threshold

        # Historical metrics for comparison
        self._history: list[ExtractionQualityReport] = []

    def analyze(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        chunks: list[TextChunk] | None = None,
        document_id: str = "",
        ground_truth_entities: list[dict[str, str]] | None = None,
    ) -> ExtractionQualityReport:
        """
        Analyze extraction quality and generate report.

        Args:
            entities: Extracted entities
            relations: Extracted relations
            chunks: Original text chunks (for coverage analysis)
            document_id: Document identifier
            ground_truth_entities: Optional ground truth for validation

        Returns:
            Comprehensive quality report
        """
        report = ExtractionQualityReport(document_id=document_id)

        # Basic counts
        report.entity_count = len(entities)
        report.relation_count = len(relations)

        if not entities:
            report.overall_quality = QualityLevel.CRITICAL
            report.quality_score = 0.0
            report.anomalies.append(AnomalyReport(
                anomaly_type="no_entities",
                severity="high",
                description="No entities were extracted",
                recommendation="Check if LLM is responding correctly or if text contains extractable entities",
            ))
            return report

        # Confidence metrics
        report.entity_confidence = self._analyze_confidence(
            [e.confidence for e in entities]
        )
        if relations:
            report.relation_confidence = self._analyze_confidence(
                [r.confidence for r in relations]
            )

        # Type distribution
        report.entity_type_distribution = self._analyze_type_distribution(
            [e.type.value for e in entities]
        )
        report.unique_entity_types = len(report.entity_type_distribution.counts)

        if relations:
            report.predicate_distribution = self._analyze_type_distribution(
                [r.predicate for r in relations]
            )
            report.unique_predicates = len(report.predicate_distribution.counts)

        # Coverage metrics
        if chunks:
            report.coverage = self._analyze_coverage(entities, relations, chunks)

        # Anomaly detection
        report.anomalies = self._detect_anomalies(
            entities, relations, chunks, report
        )

        # Validation against ground truth
        if ground_truth_entities:
            report.validation = self._validate_against_ground_truth(
                entities, ground_truth_entities
            )

        # Calculate overall quality score
        report.quality_score = self._calculate_quality_score(report)
        report.overall_quality = self._score_to_level(report.quality_score)

        # Store in history
        self._history.append(report)
        if len(self._history) > 100:
            self._history.pop(0)

        logger.info(
            "Quality analysis completed",
            document_id=document_id,
            quality=report.overall_quality.value,
            score=round(report.quality_score, 2),
            entities=report.entity_count,
            relations=report.relation_count,
            anomalies=len(report.anomalies),
        )

        return report

    def _analyze_confidence(self, confidences: list[float]) -> ConfidenceMetrics:
        """Analyze confidence score distribution."""
        if not confidences:
            return ConfidenceMetrics()

        return ConfidenceMetrics(
            mean=statistics.mean(confidences),
            median=statistics.median(confidences),
            std_dev=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            min=min(confidences),
            max=max(confidences),
            below_threshold=sum(1 for c in confidences if c < self._confidence_threshold),
            above_threshold=sum(1 for c in confidences if c >= self._confidence_threshold),
            threshold=self._confidence_threshold,
        )

    def _analyze_type_distribution(self, types: list[str]) -> TypeDistribution:
        """Analyze type distribution."""
        if not types:
            return TypeDistribution()

        counts = defaultdict(int)
        for t in types:
            counts[t] += 1

        total = len(types)
        percentages = {k: v / total for k, v in counts.items()}

        return TypeDistribution(
            counts=dict(counts),
            percentages=percentages,
            total=total,
        )

    def _analyze_coverage(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        chunks: list[TextChunk],
    ) -> CoverageMetrics:
        """Analyze extraction coverage."""
        if not chunks:
            return CoverageMetrics()

        # Count entities per chunk
        chunk_entity_counts = defaultdict(int)
        for entity in entities:
            for chunk_id in entity.chunk_ids:
                chunk_entity_counts[chunk_id] += 1

        # Count relations per chunk
        chunk_relation_counts = defaultdict(int)
        for relation in relations:
            for chunk_id in relation.chunk_ids:
                chunk_relation_counts[chunk_id] += 1

        total_chunks = len(chunks)
        chunks_with_entities = len(chunk_entity_counts)
        chunks_with_relations = len(chunk_relation_counts)

        entity_counts = list(chunk_entity_counts.values())
        relation_counts = list(chunk_relation_counts.values())

        empty_chunks = total_chunks - chunks_with_entities
        dense_chunks = sum(1 for c in entity_counts if c > 5)

        return CoverageMetrics(
            chunks_processed=total_chunks,
            chunks_with_entities=chunks_with_entities,
            chunks_with_relations=chunks_with_relations,
            avg_entities_per_chunk=statistics.mean(entity_counts) if entity_counts else 0.0,
            avg_relations_per_chunk=statistics.mean(relation_counts) if relation_counts else 0.0,
            empty_chunk_ratio=empty_chunks / max(total_chunks, 1),
            dense_chunk_ratio=dense_chunks / max(chunks_with_entities, 1),
        )

    def _detect_anomalies(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        chunks: list[TextChunk] | None,
        report: ExtractionQualityReport,
    ) -> list[AnomalyReport]:
        """Detect anomalies in extraction results."""
        anomalies = []

        # 1. Low confidence ratio
        if report.entity_confidence.mean < 0.6:
            anomalies.append(AnomalyReport(
                anomaly_type="low_confidence",
                severity="medium",
                description=f"Average entity confidence is low ({report.entity_confidence.mean:.2f})",
                affected_items=[e.name for e in entities if e.confidence < 0.6][:10],
                recommendation="Consider using LLM refinement or reviewing extraction prompts",
            ))

        # 2. Type imbalance
        if report.entity_type_distribution.percentages:
            max_percentage = max(report.entity_type_distribution.percentages.values())
            if max_percentage > self._imbalance_threshold:
                dominant_type = max(
                    report.entity_type_distribution.percentages,
                    key=report.entity_type_distribution.percentages.get
                )
                anomalies.append(AnomalyReport(
                    anomaly_type="type_imbalance",
                    severity="low",
                    description=f"Entity types are imbalanced ({dominant_type}: {max_percentage:.0%})",
                    affected_items=[dominant_type],
                    recommendation="Check if document naturally has this distribution or if extraction is biased",
                ))

        # 3. Missing relations
        if entities and not relations:
            anomalies.append(AnomalyReport(
                anomaly_type="no_relations",
                severity="medium",
                description="No relations extracted despite having entities",
                affected_items=[],
                recommendation="Review relation extraction prompts or check if text contains explicit relationships",
            ))

        # 4. Relation-entity ratio
        if entities and relations:
            ratio = len(relations) / len(entities)
            if ratio < 0.1:
                anomalies.append(AnomalyReport(
                    anomaly_type="low_relation_ratio",
                    severity="low",
                    description=f"Very few relations per entity ({ratio:.2f})",
                    affected_items=[],
                    recommendation="Consider using coreference-aware relation extraction",
                ))

        # 5. Coverage issues
        if chunks and report.coverage.empty_chunk_ratio > 0.5:
            anomalies.append(AnomalyReport(
                anomaly_type="low_coverage",
                severity="high",
                description=f"Many chunks have no entities ({report.coverage.empty_chunk_ratio:.0%})",
                affected_items=[],
                recommendation="Check chunk size or text content quality",
            ))

        # 6. Duplicate names (potential resolution issue)
        name_counts = defaultdict(list)
        for entity in entities:
            name_counts[entity.name.lower()].append(entity.id)

        duplicates = [(name, ids) for name, ids in name_counts.items() if len(ids) > 1]
        if duplicates:
            anomalies.append(AnomalyReport(
                anomaly_type="duplicate_entities",
                severity="medium",
                description=f"Found {len(duplicates)} entity names with multiple IDs",
                affected_items=[name for name, _ in duplicates[:10]],
                recommendation="Run entity resolution to merge duplicates",
            ))

        # 7. Very short descriptions
        no_description = [e.name for e in entities if not e.description]
        if len(no_description) > len(entities) * 0.5:
            anomalies.append(AnomalyReport(
                anomaly_type="missing_descriptions",
                severity="low",
                description=f"{len(no_description)} entities have no description",
                affected_items=no_description[:10],
                recommendation="Consider enriching entities with descriptions",
            ))

        return anomalies

    def _validate_against_ground_truth(
        self,
        entities: list[ExtractedEntity],
        ground_truth: list[dict[str, str]],
    ) -> ValidationResult:
        """
        Validate extracted entities against ground truth.

        Ground truth format: [{"name": "...", "type": "..."}]
        """
        extracted_names = {e.name.lower() for e in entities}
        truth_names = {gt["name"].lower() for gt in ground_truth}

        true_positives = len(extracted_names & truth_names)
        false_positives = len(extracted_names - truth_names)
        false_negatives = len(truth_names - extracted_names)

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.0001)

        matched = [
            (name, next((e.type.value for e in entities if e.name.lower() == name), ""))
            for name in extracted_names & truth_names
        ]

        return ValidationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            matched_entities=matched,
        )

    def _calculate_quality_score(self, report: ExtractionQualityReport) -> float:
        """Calculate overall quality score (0-100)."""
        score = 0.0
        weights = {
            "confidence": 30,
            "coverage": 25,
            "diversity": 15,
            "relations": 20,
            "anomalies": 10,
        }

        # Confidence score (30%)
        if report.entity_confidence.mean > 0:
            confidence_score = min(report.entity_confidence.mean / 0.9, 1.0)
            score += weights["confidence"] * confidence_score

        # Coverage score (25%)
        if report.coverage.chunks_processed > 0:
            coverage_score = 1.0 - report.coverage.empty_chunk_ratio
            score += weights["coverage"] * coverage_score

        # Type diversity score (15%)
        if report.unique_entity_types > 0:
            diversity_score = min(report.unique_entity_types / 5, 1.0)  # 5+ types is good
            score += weights["diversity"] * diversity_score

        # Relation score (20%)
        if report.entity_count > 0:
            if report.relation_count > 0:
                ratio = min(report.relation_count / report.entity_count, 1.0)
                score += weights["relations"] * ratio
            else:
                score += weights["relations"] * 0.3  # Partial credit

        # Anomaly penalty (10%)
        high_severity = sum(1 for a in report.anomalies if a.severity == "high")
        medium_severity = sum(1 for a in report.anomalies if a.severity == "medium")
        anomaly_penalty = min((high_severity * 0.5 + medium_severity * 0.2), 1.0)
        score += weights["anomalies"] * (1.0 - anomaly_penalty)

        return score

    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 75:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.FAIR
        elif score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def get_historical_trend(self, last_n: int = 10) -> dict[str, Any]:
        """Get historical quality trend."""
        recent = self._history[-last_n:]
        if not recent:
            return {"available": False}

        scores = [r.quality_score for r in recent]
        entity_counts = [r.entity_count for r in recent]

        return {
            "available": True,
            "reports_analyzed": len(recent),
            "avg_quality_score": round(statistics.mean(scores), 2),
            "quality_trend": "improving" if scores[-1] > scores[0] else "declining",
            "avg_entities": round(statistics.mean(entity_counts), 1),
            "latest_quality": recent[-1].overall_quality.value if recent else None,
        }

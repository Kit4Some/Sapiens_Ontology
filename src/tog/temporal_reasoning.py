"""
Enhanced Temporal Reasoning Module.

Provides sophisticated temporal analysis for evidence scoring:
- Date/time extraction and normalization
- Temporal relationship detection (before, after, during)
- Temporal ordering in reasoning chains
- Korean temporal expression support
- Event sequence analysis
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class TemporalRelation(str, Enum):
    """Types of temporal relationships between events."""

    BEFORE = "before"           # X happened before Y
    AFTER = "after"             # X happened after Y
    DURING = "during"           # X happened during Y
    SIMULTANEOUS = "simultaneous"  # X and Y happened at same time
    OVERLAPS = "overlaps"       # X and Y overlap in time
    CONTAINS = "contains"       # X contains Y temporally
    UNKNOWN = "unknown"         # Temporal relationship unclear


class TemporalPrecision(str, Enum):
    """Precision level of temporal information."""

    EXACT_DATE = "exact_date"    # Specific date (2024-01-15)
    MONTH = "month"              # Month precision (January 2024)
    QUARTER = "quarter"          # Quarter precision (Q1 2024)
    YEAR = "year"                # Year precision (2024)
    DECADE = "decade"            # Decade precision (2020s)
    CENTURY = "century"          # Century precision (21st century)
    RELATIVE = "relative"        # Relative expression (last week)
    APPROXIMATE = "approximate"  # Approximate (around 2020)
    UNKNOWN = "unknown"


@dataclass
class TemporalExpression:
    """Extracted temporal expression with metadata."""

    raw_text: str                            # Original text
    normalized_start: datetime | None = None  # Normalized start date
    normalized_end: datetime | None = None    # Normalized end date (for ranges)
    precision: TemporalPrecision = TemporalPrecision.UNKNOWN
    is_range: bool = False
    is_relative: bool = False
    confidence: float = 1.0

    @property
    def year(self) -> int | None:
        """Get year if available."""
        if self.normalized_start:
            return self.normalized_start.year
        return None

    def overlaps_with(self, other: "TemporalExpression") -> bool:
        """Check if this expression overlaps with another."""
        if not self.normalized_start or not other.normalized_start:
            return False

        self_end = self.normalized_end or self.normalized_start
        other_end = other.normalized_end or other.normalized_start

        return (
            self.normalized_start <= other_end and
            self_end >= other.normalized_start
        )

    def distance_years(self, other: "TemporalExpression") -> float | None:
        """Calculate distance in years between expressions."""
        if not self.normalized_start or not other.normalized_start:
            return None

        delta = abs((self.normalized_start - other.normalized_start).days)
        return delta / 365.25


@dataclass
class TemporalContext:
    """Temporal context extracted from text."""

    expressions: list[TemporalExpression] = field(default_factory=list)
    temporal_relations: list[tuple[str, TemporalRelation, str]] = field(default_factory=list)
    has_temporal_ordering: bool = False
    primary_timeframe: TemporalExpression | None = None

    @property
    def has_temporal_info(self) -> bool:
        return len(self.expressions) > 0

    @property
    def year_range(self) -> tuple[int | None, int | None]:
        """Get the range of years mentioned."""
        years = [e.year for e in self.expressions if e.year]
        if not years:
            return None, None
        return min(years), max(years)


class TemporalReasoningEngine:
    """
    Enhanced temporal reasoning for evidence scoring and chain analysis.

    Features:
    - Multi-format date extraction (ISO, natural language, Korean)
    - Temporal relationship inference
    - Event sequence validation
    - Temporal consistency checking in reasoning chains
    """

    def __init__(self):
        """Initialize the temporal reasoning engine."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for temporal extraction."""
        # ISO format dates
        self._iso_date = re.compile(
            r"\b(\d{4})-(\d{2})-(\d{2})\b"
        )

        # Year patterns (1800-2099)
        self._year = re.compile(
            r"\b(1[89]\d{2}|20[0-9]{2})\b"
        )

        # Month-Year patterns
        self._month_year = re.compile(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
            re.IGNORECASE
        )

        # Korean month-year
        self._korean_month_year = re.compile(
            r"(\d{4})년\s*(\d{1,2})월"
        )

        # Korean year
        self._korean_year = re.compile(
            r"(\d{4})년"
        )

        # Decade patterns
        self._decade = re.compile(
            r"\b(1[89]\d0|20[0-2]0)s?\b|\b(1[89]\d0|20[0-2]0)년대\b"
        )

        # Quarter patterns
        self._quarter = re.compile(
            r"\b[Qq]([1-4])\s+(\d{4})\b|\b(\d{4})\s+[Qq]([1-4])\b"
        )

        # Relative temporal expressions (English)
        self._relative_en = re.compile(
            r"\b(last|next|this|previous|following|past|recent|current)\s+(week|month|year|decade|quarter|century)\b",
            re.IGNORECASE
        )

        # Relative temporal expressions (Korean)
        self._relative_kr = re.compile(
            r"(지난|다음|이번|작년|올해|내년|최근|현재|당시|그때)"
        )

        # Temporal ordering keywords (English)
        self._order_before_en = re.compile(
            r"\b(before|prior to|preceding|earlier than|until)\b",
            re.IGNORECASE
        )
        self._order_after_en = re.compile(
            r"\b(after|following|since|subsequent to|later than|from)\b",
            re.IGNORECASE
        )
        self._order_during_en = re.compile(
            r"\b(during|while|throughout|amid|in the course of)\b",
            re.IGNORECASE
        )
        self._order_simultaneous_en = re.compile(
            r"\b(at the same time|simultaneously|concurrently|meanwhile)\b",
            re.IGNORECASE
        )

        # Temporal ordering keywords (Korean)
        self._order_before_kr = re.compile(
            r"(전에|이전에|앞서|~전|보다 먼저)"
        )
        self._order_after_kr = re.compile(
            r"(후에|이후에|뒤에|~후|부터|다음에)"
        )
        self._order_during_kr = re.compile(
            r"(동안|중에|사이에|~중)"
        )

        # Duration patterns
        self._duration_en = re.compile(
            r"\b(\d+)\s*(years?|months?|weeks?|days?|decades?)\b",
            re.IGNORECASE
        )
        self._duration_kr = re.compile(
            r"(\d+)\s*(년|개월|주|일|세기)"
        )

        # Approximate temporal markers
        self._approximate = re.compile(
            r"\b(around|approximately|about|circa|roughly|nearly)\s+(\d{4})\b|약\s*(\d{4})년|(\d{4})년경",
            re.IGNORECASE
        )

    def extract_temporal_context(self, text: str) -> TemporalContext:
        """
        Extract comprehensive temporal context from text.

        Args:
            text: Input text

        Returns:
            TemporalContext with all extracted temporal information
        """
        expressions = []

        # Extract ISO dates
        for match in self._iso_date.finditer(text):
            try:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                dt = datetime(year, month, day)
                expressions.append(TemporalExpression(
                    raw_text=match.group(0),
                    normalized_start=dt,
                    precision=TemporalPrecision.EXACT_DATE,
                    confidence=1.0,
                ))
            except ValueError:
                pass

        # Extract month-year (English)
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        for match in self._month_year.finditer(text):
            month_name = match.group(1).lower()
            year = int(match.group(2))
            month = month_map.get(month_name, 1)
            expressions.append(TemporalExpression(
                raw_text=match.group(0),
                normalized_start=datetime(year, month, 1),
                normalized_end=datetime(year, month, 28),  # Simplified
                precision=TemporalPrecision.MONTH,
                is_range=True,
                confidence=0.95,
            ))

        # Extract Korean month-year
        for match in self._korean_month_year.finditer(text):
            year = int(match.group(1))
            month = int(match.group(2))
            if 1 <= month <= 12:
                expressions.append(TemporalExpression(
                    raw_text=match.group(0),
                    normalized_start=datetime(year, month, 1),
                    normalized_end=datetime(year, month, 28),
                    precision=TemporalPrecision.MONTH,
                    is_range=True,
                    confidence=0.95,
                ))

        # Extract quarters
        for match in self._quarter.finditer(text):
            if match.group(1) and match.group(2):
                quarter, year = int(match.group(1)), int(match.group(2))
            else:
                year, quarter = int(match.group(3)), int(match.group(4))

            start_month = (quarter - 1) * 3 + 1
            expressions.append(TemporalExpression(
                raw_text=match.group(0),
                normalized_start=datetime(year, start_month, 1),
                normalized_end=datetime(year, start_month + 2, 28),
                precision=TemporalPrecision.QUARTER,
                is_range=True,
                confidence=0.9,
            ))

        # Extract decades
        for match in self._decade.finditer(text):
            decade_str = match.group(0).replace('s', '').replace('년대', '')
            decade = int(decade_str)
            expressions.append(TemporalExpression(
                raw_text=match.group(0),
                normalized_start=datetime(decade, 1, 1),
                normalized_end=datetime(decade + 9, 12, 31),
                precision=TemporalPrecision.DECADE,
                is_range=True,
                confidence=0.8,
            ))

        # Extract approximate years
        for match in self._approximate.finditer(text):
            year_str = match.group(2) or match.group(3) or match.group(4)
            if year_str:
                year = int(year_str)
                expressions.append(TemporalExpression(
                    raw_text=match.group(0),
                    normalized_start=datetime(year - 2, 1, 1),
                    normalized_end=datetime(year + 2, 12, 31),
                    precision=TemporalPrecision.APPROXIMATE,
                    is_range=True,
                    confidence=0.7,
                ))

        # Extract standalone years (only if not already captured)
        captured_years = {e.year for e in expressions if e.year}
        for match in self._year.finditer(text):
            year = int(match.group(1))
            if year not in captured_years:
                expressions.append(TemporalExpression(
                    raw_text=match.group(0),
                    normalized_start=datetime(year, 1, 1),
                    normalized_end=datetime(year, 12, 31),
                    precision=TemporalPrecision.YEAR,
                    is_range=True,
                    confidence=0.85,
                ))
                captured_years.add(year)

        # Extract relative expressions
        for match in self._relative_en.finditer(text):
            expressions.append(TemporalExpression(
                raw_text=match.group(0),
                precision=TemporalPrecision.RELATIVE,
                is_relative=True,
                confidence=0.6,
            ))

        for match in self._relative_kr.finditer(text):
            expressions.append(TemporalExpression(
                raw_text=match.group(0),
                precision=TemporalPrecision.RELATIVE,
                is_relative=True,
                confidence=0.6,
            ))

        # Detect temporal relations
        temporal_relations = self._detect_temporal_relations(text)

        # Determine primary timeframe
        primary = None
        if expressions:
            # Prefer most precise non-relative expression
            sorted_exprs = sorted(
                [e for e in expressions if not e.is_relative],
                key=lambda x: (
                    -x.confidence,
                    ["exact_date", "month", "quarter", "year", "decade", "approximate"].index(x.precision.value)
                    if x.precision.value in ["exact_date", "month", "quarter", "year", "decade", "approximate"]
                    else 10
                )
            )
            if sorted_exprs:
                primary = sorted_exprs[0]

        return TemporalContext(
            expressions=expressions,
            temporal_relations=temporal_relations,
            has_temporal_ordering=len(temporal_relations) > 0,
            primary_timeframe=primary,
        )

    def _detect_temporal_relations(
        self,
        text: str,
    ) -> list[tuple[str, TemporalRelation, str]]:
        """Detect temporal ordering relationships in text."""
        relations = []

        # Check for ordering keywords
        has_before = bool(self._order_before_en.search(text) or self._order_before_kr.search(text))
        has_after = bool(self._order_after_en.search(text) or self._order_after_kr.search(text))
        has_during = bool(self._order_during_en.search(text) or self._order_during_kr.search(text))
        has_simultaneous = bool(self._order_simultaneous_en.search(text))

        # Create generic relations based on detected keywords
        if has_before:
            relations.append(("event_a", TemporalRelation.BEFORE, "event_b"))
        if has_after:
            relations.append(("event_a", TemporalRelation.AFTER, "event_b"))
        if has_during:
            relations.append(("event_a", TemporalRelation.DURING, "event_b"))
        if has_simultaneous:
            relations.append(("event_a", TemporalRelation.SIMULTANEOUS, "event_b"))

        return relations

    def compute_temporal_alignment_score(
        self,
        query_context: TemporalContext,
        evidence_context: TemporalContext,
    ) -> dict[str, Any]:
        """
        Compute detailed temporal alignment score between query and evidence.

        Returns:
            Dict with temporal alignment details
        """
        result = {
            "score": 0.5,  # Default neutral
            "alignment_type": "neutral",
            "details": {},
            "temporal_match": False,
            "temporal_consistency": True,
        }

        # No temporal info in query - neutral alignment
        if not query_context.has_temporal_info:
            result["alignment_type"] = "query_no_temporal"
            return result

        # No temporal info in evidence - slightly negative
        if not evidence_context.has_temporal_info:
            result["score"] = 0.4
            result["alignment_type"] = "evidence_no_temporal"
            return result

        # Both have temporal info - compute alignment
        query_years = query_context.year_range
        evidence_years = evidence_context.year_range

        if query_years[0] and evidence_years[0]:
            # Check for year overlap
            query_min, query_max = query_years
            evidence_min, evidence_max = evidence_years

            # Exact match
            if query_min == evidence_min and query_max == evidence_max:
                result["score"] = 1.0
                result["alignment_type"] = "exact_match"
                result["temporal_match"] = True
            # Overlap
            elif query_min <= evidence_max and query_max >= evidence_min:
                overlap_start = max(query_min, evidence_min)
                overlap_end = min(query_max, evidence_max)
                total_range = max(query_max, evidence_max) - min(query_min, evidence_min) + 1
                overlap_ratio = (overlap_end - overlap_start + 1) / total_range
                result["score"] = 0.6 + (overlap_ratio * 0.4)
                result["alignment_type"] = "overlap"
                result["temporal_match"] = True
                result["details"]["overlap_ratio"] = overlap_ratio
            # Close proximity (within 5 years)
            elif min(abs(query_min - evidence_max), abs(query_max - evidence_min)) <= 5:
                distance = min(abs(query_min - evidence_max), abs(query_max - evidence_min))
                result["score"] = 0.5 + (0.2 * (5 - distance) / 5)
                result["alignment_type"] = "close_proximity"
                result["details"]["year_distance"] = distance
            # Distant
            else:
                distance = min(abs(query_min - evidence_max), abs(query_max - evidence_min))
                result["score"] = max(0.1, 0.4 - (distance / 50))
                result["alignment_type"] = "distant"
                result["details"]["year_distance"] = distance
                result["temporal_consistency"] = False

        # Check temporal relation consistency
        if query_context.has_temporal_ordering and evidence_context.has_temporal_ordering:
            # If both mention ordering, check consistency
            query_rels = {r[1] for r in query_context.temporal_relations}
            evidence_rels = {r[1] for r in evidence_context.temporal_relations}

            # Contradicting relations
            contradictions = [
                (TemporalRelation.BEFORE, TemporalRelation.AFTER),
                (TemporalRelation.AFTER, TemporalRelation.BEFORE),
            ]
            for q_rel, e_rel in contradictions:
                if q_rel in query_rels and e_rel in evidence_rels:
                    result["score"] *= 0.5
                    result["temporal_consistency"] = False
                    result["details"]["relation_conflict"] = True

            # Matching relations boost
            if query_rels & evidence_rels:
                result["score"] = min(1.0, result["score"] * 1.1)
                result["details"]["matching_relations"] = list(query_rels & evidence_rels)

        # Precision bonus - higher precision evidence is better
        if evidence_context.primary_timeframe:
            precision_bonus = {
                TemporalPrecision.EXACT_DATE: 0.05,
                TemporalPrecision.MONTH: 0.04,
                TemporalPrecision.QUARTER: 0.03,
                TemporalPrecision.YEAR: 0.02,
                TemporalPrecision.DECADE: 0.01,
            }
            bonus = precision_bonus.get(evidence_context.primary_timeframe.precision, 0)
            result["score"] = min(1.0, result["score"] + bonus)
            result["details"]["precision_bonus"] = bonus

        return result

    def validate_temporal_chain(
        self,
        evidence_list: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Validate temporal consistency across a reasoning chain.

        Args:
            evidence_list: List of evidence with temporal context

        Returns:
            Validation result with consistency score
        """
        if len(evidence_list) < 2:
            return {
                "valid": True,
                "consistency_score": 1.0,
                "issues": [],
            }

        issues = []
        temporal_sequence = []

        for i, ev in enumerate(evidence_list):
            text = ev.get("content", "")
            context = self.extract_temporal_context(text)
            if context.primary_timeframe and context.primary_timeframe.year:
                temporal_sequence.append({
                    "index": i,
                    "year": context.primary_timeframe.year,
                    "precision": context.primary_timeframe.precision,
                })

        # Check for temporal jumps
        if len(temporal_sequence) >= 2:
            for i in range(len(temporal_sequence) - 1):
                curr = temporal_sequence[i]
                next_item = temporal_sequence[i + 1]
                gap = abs(next_item["year"] - curr["year"])

                if gap > 50:  # Large temporal jump
                    issues.append(f"Large temporal gap ({gap} years) between evidence {curr['index']} and {next_item['index']}")

        consistency_score = 1.0 - (len(issues) * 0.2)

        return {
            "valid": len(issues) == 0,
            "consistency_score": max(0.0, consistency_score),
            "issues": issues,
            "temporal_sequence": temporal_sequence,
        }


# Global engine instance
_engine: TemporalReasoningEngine | None = None


def get_temporal_engine() -> TemporalReasoningEngine:
    """Get the global temporal reasoning engine."""
    global _engine
    if _engine is None:
        _engine = TemporalReasoningEngine()
    return _engine


def compute_enhanced_temporal_alignment(
    query: str,
    evidence_text: str,
) -> dict[str, Any]:
    """
    Convenience function for computing temporal alignment.

    Args:
        query: Query text
        evidence_text: Evidence text

    Returns:
        Temporal alignment score and details
    """
    engine = get_temporal_engine()
    query_context = engine.extract_temporal_context(query)
    evidence_context = engine.extract_temporal_context(evidence_text)
    return engine.compute_temporal_alignment_score(query_context, evidence_context)

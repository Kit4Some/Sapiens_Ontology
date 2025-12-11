"""
Negative Evidence Detection and Processing Module.

Handles:
- Negation detection in text
- Contradiction identification between evidence pieces
- Evidence polarity scoring
- Conflicting claim detection
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class EvidencePolarity(str, Enum):
    """Polarity classification for evidence."""

    POSITIVE = "positive"       # Supports the claim
    NEGATIVE = "negative"       # Contradicts the claim
    NEUTRAL = "neutral"         # Neither supports nor contradicts
    UNCERTAIN = "uncertain"     # Cannot determine polarity


class ContradictionType(str, Enum):
    """Types of contradictions between evidence pieces."""

    DIRECT_NEGATION = "direct_negation"       # "X is A" vs "X is not A"
    VALUE_CONFLICT = "value_conflict"         # "X is A" vs "X is B"
    TEMPORAL_CONFLICT = "temporal_conflict"   # Different time claims
    ENTITY_CONFLICT = "entity_conflict"       # Different entity for same role
    LOGICAL_CONFLICT = "logical_conflict"     # Logically incompatible
    NONE = "none"


@dataclass
class NegationSpan:
    """Detected negation span in text."""

    start: int
    end: int
    negation_word: str
    scope_start: int      # Start of negation scope
    scope_end: int        # End of negation scope
    negated_text: str     # Text being negated


@dataclass
class PolarityResult:
    """Result of polarity analysis."""

    polarity: EvidencePolarity
    confidence: float
    negations: list[NegationSpan] = field(default_factory=list)
    negated_claims: list[str] = field(default_factory=list)
    positive_indicators: int = 0
    negative_indicators: int = 0
    reasoning: str = ""


@dataclass
class ContradictionResult:
    """Result of contradiction detection between two evidence pieces."""

    has_contradiction: bool
    contradiction_type: ContradictionType
    confidence: float
    conflicting_claims: list[tuple[str, str]] = field(default_factory=list)
    explanation: str = ""


class NegativeEvidenceDetector:
    """
    Detects and analyzes negative evidence and contradictions.

    Features:
    - Negation word detection with scope analysis
    - Evidence polarity classification
    - Inter-evidence contradiction detection
    - Korean and English support
    """

    def __init__(self):
        """Initialize the negative evidence detector."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for negation and contradiction detection."""

        # English negation patterns
        self._negation_words_en = re.compile(
            r"\b(not|no|never|none|neither|nor|nobody|nothing|nowhere|"
            r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
            r"doesn't|don't|didn't|won't|wouldn't|can't|cannot|couldn't|"
            r"shouldn't|shan't|mustn't|needn't|"
            r"without|lack|lacking|lacks|lacked|"
            r"deny|denies|denied|denying|"
            r"refuse|refuses|refused|refusing|"
            r"fail|fails|failed|failing|"
            r"unable|impossible|unlikely)\b",
            re.IGNORECASE
        )

        # Korean negation patterns
        self._negation_words_kr = re.compile(
            r"(아니다|아닙니다|아니에요|아니야|"
            r"없다|없습니다|없어요|없어|"
            r"못하다|못합니다|못해요|못해|"
            r"안하다|안합니다|안해요|안해|"
            r"않다|않습니다|않아요|않아|"
            r"~지 않다|~지 않습니다|"
            r"불가능|불가|거부|거절|"
            r"결코|전혀|절대로|절대)"
        )

        # Contrast/contradiction markers (English)
        self._contrast_en = re.compile(
            r"\b(but|however|although|though|yet|despite|"
            r"nevertheless|nonetheless|on the contrary|"
            r"in contrast|conversely|instead|rather than|"
            r"contrary to|unlike|whereas|while)\b",
            re.IGNORECASE
        )

        # Contrast/contradiction markers (Korean)
        self._contrast_kr = re.compile(
            r"(하지만|그러나|그렇지만|반면|반대로|"
            r"대신|오히려|달리|~와 달리|"
            r"~에도 불구하고|그럼에도)"
        )

        # Correction patterns (English)
        self._correction_en = re.compile(
            r"\b(actually|in fact|in reality|"
            r"correctly|the truth is|"
            r"mistakenly|incorrectly|wrongly|"
            r"false|untrue|inaccurate|"
            r"misconception|myth|"
            r"no longer|used to be|formerly|previously)\b",
            re.IGNORECASE
        )

        # Correction patterns (Korean)
        self._correction_kr = re.compile(
            r"(사실|실제로|정확히|올바르게|"
            r"잘못|틀리게|부정확|"
            r"오해|착각|더 이상|"
            r"이전에는|예전에는|과거에는)"
        )

        # Value comparison patterns
        self._value_pattern = re.compile(
            r"(?:is|was|are|were|be|been|being)\s+(?:not\s+)?(\w+)",
            re.IGNORECASE
        )

        # Numeric comparison patterns
        self._numeric_pattern = re.compile(
            r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(years?|months?|days?|hours?|"
            r"percent|%|dollars?|\$|million|billion|km|miles?|kg|pounds?)",
            re.IGNORECASE
        )

    def detect_negations(self, text: str) -> list[NegationSpan]:
        """
        Detect negation words and their scope in text.

        Args:
            text: Input text

        Returns:
            List of detected negation spans
        """
        negations = []

        # English negations
        for match in self._negation_words_en.finditer(text):
            # Simple scope heuristic: negation applies to following clause
            # until punctuation or end of sentence
            scope_start = match.start()
            remaining_text = text[match.end():]

            # Find end of scope (next major punctuation or end)
            scope_end_rel = len(remaining_text)
            for punct in ['.', '!', '?', ',', ';', ':', 'but', 'however', 'although']:
                idx = remaining_text.lower().find(punct)
                if idx != -1 and idx < scope_end_rel:
                    scope_end_rel = idx

            scope_end = match.end() + scope_end_rel
            negated_text = text[match.end():scope_end].strip()

            negations.append(NegationSpan(
                start=match.start(),
                end=match.end(),
                negation_word=match.group(0).lower(),
                scope_start=scope_start,
                scope_end=scope_end,
                negated_text=negated_text[:100],  # Limit length
            ))

        # Korean negations
        for match in self._negation_words_kr.finditer(text):
            # Korean negation typically affects preceding phrase
            scope_start = max(0, match.start() - 50)
            preceding_text = text[scope_start:match.start()]

            negations.append(NegationSpan(
                start=match.start(),
                end=match.end(),
                negation_word=match.group(0),
                scope_start=scope_start,
                scope_end=match.end(),
                negated_text=preceding_text.strip()[-50:],
            ))

        return negations

    def analyze_polarity(
        self,
        evidence_text: str,
        query: str | None = None,
    ) -> PolarityResult:
        """
        Analyze the polarity of evidence relative to a query.

        Args:
            evidence_text: Evidence text to analyze
            query: Optional query for context-aware analysis

        Returns:
            PolarityResult with polarity classification
        """
        negations = self.detect_negations(evidence_text)
        negation_count = len(negations)

        # Count contrast markers
        contrast_count_en = len(self._contrast_en.findall(evidence_text))
        contrast_count_kr = len(self._contrast_kr.findall(evidence_text))
        contrast_count = contrast_count_en + contrast_count_kr

        # Count correction markers
        correction_count_en = len(self._correction_en.findall(evidence_text))
        correction_count_kr = len(self._correction_kr.findall(evidence_text))
        correction_count = correction_count_en + correction_count_kr

        # Determine polarity
        negative_indicators = negation_count + contrast_count + correction_count
        positive_indicators = 0

        # Check for positive indicators
        positive_words = re.findall(
            r"\b(confirmed|verified|proved|proven|established|"
            r"indeed|certainly|definitely|clearly|obviously|"
            r"correct|true|accurate|right|valid|"
            r"supports?|confirms?|demonstrates?|shows?)\b",
            evidence_text,
            re.IGNORECASE
        )
        positive_words_kr = re.findall(
            r"(확인|검증|증명|입증|"
            r"분명|확실|명확|올바|정확|"
            r"맞다|사실|지지|보여)",
            evidence_text
        )
        positive_indicators = len(positive_words) + len(positive_words_kr)

        # Classify polarity
        if negative_indicators > positive_indicators + 1:
            polarity = EvidencePolarity.NEGATIVE
            confidence = min(0.95, 0.6 + (negative_indicators * 0.1))
            reasoning = f"Detected {negation_count} negations, {contrast_count} contrasts, {correction_count} corrections"
        elif positive_indicators > negative_indicators + 1:
            polarity = EvidencePolarity.POSITIVE
            confidence = min(0.95, 0.6 + (positive_indicators * 0.1))
            reasoning = f"Detected {positive_indicators} positive indicators"
        elif negation_count == 0 and positive_indicators == 0:
            polarity = EvidencePolarity.NEUTRAL
            confidence = 0.7
            reasoning = "No strong polarity indicators found"
        else:
            polarity = EvidencePolarity.UNCERTAIN
            confidence = 0.5
            reasoning = f"Mixed signals: {positive_indicators} positive, {negative_indicators} negative"

        # Context-aware adjustment if query provided
        if query and negations:
            # Check if negation directly relates to query entities
            query_lower = query.lower()
            for negation in negations:
                negated_lower = negation.negated_text.lower()
                # If negated text overlaps with query terms, stronger negative
                query_words = set(query_lower.split())
                negated_words = set(negated_lower.split())
                overlap = query_words & negated_words
                if len(overlap) >= 2:
                    confidence = min(0.95, confidence + 0.1)
                    reasoning += f"; negation directly relates to query"

        return PolarityResult(
            polarity=polarity,
            confidence=confidence,
            negations=negations,
            negated_claims=[n.negated_text for n in negations if n.negated_text],
            positive_indicators=positive_indicators,
            negative_indicators=negative_indicators,
            reasoning=reasoning,
        )

    def detect_contradiction(
        self,
        evidence_a: str,
        evidence_b: str,
    ) -> ContradictionResult:
        """
        Detect contradictions between two evidence pieces.

        Args:
            evidence_a: First evidence text
            evidence_b: Second evidence text

        Returns:
            ContradictionResult with contradiction details
        """
        conflicts = []
        contradiction_type = ContradictionType.NONE
        confidence = 0.0

        # Check for direct negation patterns
        # Pattern: "X is Y" vs "X is not Y"
        polarity_a = self.analyze_polarity(evidence_a)
        polarity_b = self.analyze_polarity(evidence_b)

        # Check polarity opposition
        if (polarity_a.polarity == EvidencePolarity.POSITIVE and
            polarity_b.polarity == EvidencePolarity.NEGATIVE):
            contradiction_type = ContradictionType.DIRECT_NEGATION
            confidence = 0.6

        elif (polarity_a.polarity == EvidencePolarity.NEGATIVE and
              polarity_b.polarity == EvidencePolarity.POSITIVE):
            contradiction_type = ContradictionType.DIRECT_NEGATION
            confidence = 0.6

        # Check for value conflicts
        values_a = self._value_pattern.findall(evidence_a.lower())
        values_b = self._value_pattern.findall(evidence_b.lower())

        # Extract numeric values
        numbers_a = self._numeric_pattern.findall(evidence_a)
        numbers_b = self._numeric_pattern.findall(evidence_b)

        # Check for conflicting numbers with same units
        for num_a, unit_a in numbers_a:
            for num_b, unit_b in numbers_b:
                if unit_a.lower().rstrip('s') == unit_b.lower().rstrip('s'):  # Same unit
                    try:
                        val_a = float(num_a.replace(',', ''))
                        val_b = float(num_b.replace(',', ''))
                        # Significant difference (>20%)
                        if val_a != 0 and abs(val_a - val_b) / val_a > 0.2:
                            conflicts.append((
                                f"{num_a} {unit_a}",
                                f"{num_b} {unit_b}"
                            ))
                            if contradiction_type == ContradictionType.NONE:
                                contradiction_type = ContradictionType.VALUE_CONFLICT
                                confidence = 0.7
                    except ValueError:
                        pass

        # Check for "no longer" / "used to be" patterns
        no_longer_a = bool(re.search(r"\bno longer\b|\bused to\b|\bformerly\b", evidence_a, re.IGNORECASE))
        no_longer_b = bool(re.search(r"\bno longer\b|\bused to\b|\bformerly\b", evidence_b, re.IGNORECASE))

        if no_longer_a != no_longer_b:
            # One mentions temporal change, the other doesn't
            if contradiction_type == ContradictionType.NONE:
                contradiction_type = ContradictionType.TEMPORAL_CONFLICT
                confidence = 0.5

        # Check for Korean contradiction patterns
        kr_negation_a = bool(self._negation_words_kr.search(evidence_a))
        kr_negation_b = bool(self._negation_words_kr.search(evidence_b))

        if kr_negation_a != kr_negation_b:
            # One has negation, other doesn't - potential contradiction
            if contradiction_type == ContradictionType.NONE:
                contradiction_type = ContradictionType.DIRECT_NEGATION
                confidence = max(confidence, 0.5)

        explanation = ""
        if contradiction_type != ContradictionType.NONE:
            explanation = f"Detected {contradiction_type.value}"
            if conflicts:
                explanation += f" with {len(conflicts)} conflicting claims"

        return ContradictionResult(
            has_contradiction=contradiction_type != ContradictionType.NONE,
            contradiction_type=contradiction_type,
            confidence=confidence,
            conflicting_claims=conflicts,
            explanation=explanation,
        )

    def find_contradictions_in_list(
        self,
        evidence_list: list[dict[str, Any]],
    ) -> list[tuple[int, int, ContradictionResult]]:
        """
        Find all contradictions in a list of evidence.

        Args:
            evidence_list: List of evidence dicts with 'content' key

        Returns:
            List of (index_a, index_b, contradiction_result) tuples
        """
        contradictions = []

        for i in range(len(evidence_list)):
            for j in range(i + 1, len(evidence_list)):
                text_a = evidence_list[i].get("content", "")
                text_b = evidence_list[j].get("content", "")

                if not text_a or not text_b:
                    continue

                result = self.detect_contradiction(text_a, text_b)
                if result.has_contradiction:
                    contradictions.append((i, j, result))

        return contradictions


class NegativeEvidenceScorer:
    """
    Scores evidence considering negative indicators and contradictions.

    Provides adjusted relevance scores based on:
    - Evidence polarity relative to query
    - Contradiction with other evidence
    - Negation impact on answer support
    """

    def __init__(self):
        """Initialize the negative evidence scorer."""
        self._detector = NegativeEvidenceDetector()

    def compute_negative_adjusted_score(
        self,
        base_score: float,
        evidence_text: str,
        query: str,
        other_evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compute score adjusted for negative evidence factors.

        Args:
            base_score: Base relevance score
            evidence_text: Evidence text
            query: Query text
            other_evidence: Other evidence texts for contradiction checking

        Returns:
            Dict with adjusted score and factors
        """
        # Analyze polarity
        polarity_result = self._detector.analyze_polarity(evidence_text, query)

        # Start with base score
        adjusted_score = base_score

        # Adjust for polarity
        if polarity_result.polarity == EvidencePolarity.NEGATIVE:
            # Negative evidence can still be relevant (answering "not X")
            # But may not support a positive answer
            adjusted_score *= 0.85
        elif polarity_result.polarity == EvidencePolarity.UNCERTAIN:
            adjusted_score *= 0.95

        # Check for contradictions with other evidence
        contradiction_penalty = 0.0
        contradictions_found = []

        if other_evidence:
            for i, other_text in enumerate(other_evidence):
                result = self._detector.detect_contradiction(evidence_text, other_text)
                if result.has_contradiction:
                    contradictions_found.append({
                        "other_index": i,
                        "type": result.contradiction_type.value,
                        "confidence": result.confidence,
                    })
                    contradiction_penalty += result.confidence * 0.1

        # Apply contradiction penalty (max 30% reduction)
        adjusted_score *= (1.0 - min(0.3, contradiction_penalty))

        # Determine if evidence supports or contradicts answer
        supports_answer = polarity_result.polarity != EvidencePolarity.NEGATIVE

        return {
            "original_score": base_score,
            "adjusted_score": max(0.0, min(1.0, adjusted_score)),
            "polarity": polarity_result.polarity.value,
            "polarity_confidence": polarity_result.confidence,
            "supports_answer": supports_answer,
            "negation_count": len(polarity_result.negations),
            "negated_claims": polarity_result.negated_claims,
            "contradiction_penalty": contradiction_penalty,
            "contradictions": contradictions_found,
            "reasoning": polarity_result.reasoning,
        }


# Global instances
_detector: NegativeEvidenceDetector | None = None
_scorer: NegativeEvidenceScorer | None = None


def get_negative_evidence_detector() -> NegativeEvidenceDetector:
    """Get the global negative evidence detector."""
    global _detector
    if _detector is None:
        _detector = NegativeEvidenceDetector()
    return _detector


def get_negative_evidence_scorer() -> NegativeEvidenceScorer:
    """Get the global negative evidence scorer."""
    global _scorer
    if _scorer is None:
        _scorer = NegativeEvidenceScorer()
    return _scorer


def analyze_evidence_polarity(
    evidence_text: str,
    query: str | None = None,
) -> PolarityResult:
    """Convenience function for polarity analysis."""
    return get_negative_evidence_detector().analyze_polarity(evidence_text, query)


def detect_evidence_contradiction(
    evidence_a: str,
    evidence_b: str,
) -> ContradictionResult:
    """Convenience function for contradiction detection."""
    return get_negative_evidence_detector().detect_contradiction(evidence_a, evidence_b)

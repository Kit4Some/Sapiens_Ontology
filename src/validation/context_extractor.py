"""
Dynamic Context Extraction Module (Section 5.0).

Extracts and classifies context from documents and queries for validation:
- Document format and language detection
- Domain classification
- Query type classification (DEFINITION, COMPARISON, PROCEDURE, etc.)
- Complexity assessment (simple, moderate, complex, multi-hop)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    MD = "md"
    HTML = "html"
    TXT = "txt"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    UNKNOWN = "unknown"


class Language(str, Enum):
    """Supported languages."""
    KOREAN = "ko"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Domain(str, Enum):
    """Domain classifications."""
    SECURITY = "security"
    FINANCE = "finance"
    MEDICAL = "medical"
    LEGAL = "legal"
    TECH = "tech"
    ACADEMIC = "academic"
    GENERAL = "general"


class QueryType(str, Enum):
    """Query type classifications."""
    DEFINITION = "definition"      # "뭐야", "무엇", "what is", "define"
    COMPARISON = "comparison"      # "차이", "비교", "vs", "difference"
    PROCEDURE = "procedure"        # "어떻게", "방법", "how to", "steps"
    CAUSAL = "causal"              # "왜", "원인", "why", "because"
    LIST = "list"                  # "종류", "목록", "list", "types"
    PRECAUTION = "precaution"      # "주의", "조심", "careful", "avoid"
    TEMPORAL = "temporal"          # "언제", "when", "date", "time"
    LOCATION = "location"          # "어디", "where", "location"
    QUANTITATIVE = "quantitative"  # "얼마", "몇", "how many", "count"
    RELATIONAL = "relational"      # "관계", "연결", "related to"
    GENERAL = "general"            # None of the above


class Complexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"          # Single concept, direct lookup
    MODERATE = "moderate"      # 2-3 concepts, some reasoning
    COMPLEX = "complex"        # Multiple concepts, synthesis
    MULTI_HOP = "multi_hop"    # Requires traversing multiple relationships


class AnswerType(str, Enum):
    """Expected answer types."""
    DIRECT = "direct"          # Direct from graph
    INFERRED = "inferred"      # Requires reasoning
    UNCERTAIN = "uncertain"    # Partial answer


# =============================================================================
# Query Type Detection Patterns
# =============================================================================


QUERY_TYPE_PATTERNS: dict[QueryType, dict[str, list[str]]] = {
    QueryType.DEFINITION: {
        "ko": ["뭐야", "무엇", "무엇인가", "뭔가", "정의", "설명해", "알려줘", "이란"],
        "en": ["what is", "what are", "define", "definition", "explain", "describe", "meaning of"],
    },
    QueryType.COMPARISON: {
        "ko": ["차이", "비교", "다른점", "차이점", "구분", "비슷", "같은점"],
        "en": ["difference", "compare", "vs", "versus", "differ", "similar", "distinction"],
    },
    QueryType.PROCEDURE: {
        "ko": ["어떻게", "방법", "절차", "과정", "단계", "순서", "하려면"],
        "en": ["how to", "how do", "steps", "procedure", "process", "method", "way to"],
    },
    QueryType.CAUSAL: {
        "ko": ["왜", "이유", "원인", "때문", "결과", "영향"],
        "en": ["why", "reason", "because", "cause", "result", "effect", "impact"],
    },
    QueryType.LIST: {
        "ko": ["종류", "목록", "나열", "어떤것들", "무엇들", "리스트", "유형"],
        "en": ["list", "types", "kinds", "examples", "enumerate", "what are the"],
    },
    QueryType.PRECAUTION: {
        "ko": ["주의", "조심", "위험", "피해야", "금지", "하면안되", "주의사항"],
        "en": ["careful", "caution", "avoid", "danger", "risk", "warning", "don't"],
    },
    QueryType.TEMPORAL: {
        "ko": ["언제", "시간", "날짜", "기간", "얼마나 오래"],
        "en": ["when", "date", "time", "period", "duration", "how long"],
    },
    QueryType.LOCATION: {
        "ko": ["어디", "장소", "위치", "곳"],
        "en": ["where", "location", "place", "located"],
    },
    QueryType.QUANTITATIVE: {
        "ko": ["얼마", "몇", "개수", "수량", "비율", "퍼센트"],
        "en": ["how many", "how much", "count", "number", "percentage", "ratio"],
    },
    QueryType.RELATIONAL: {
        "ko": ["관계", "연결", "연관", "관련"],
        "en": ["related", "relationship", "connection", "associated", "linked"],
    },
}


# Query type to expected answer type and confidence mapping
QUERY_TYPE_EXPECTATIONS: dict[QueryType, dict[str, Any]] = {
    QueryType.DEFINITION: {"answer_type": AnswerType.DIRECT, "min_confidence": 0.75},
    QueryType.COMPARISON: {"answer_type": AnswerType.INFERRED, "min_confidence": 0.60},
    QueryType.PROCEDURE: {"answer_type": AnswerType.DIRECT, "min_confidence": 0.65},
    QueryType.CAUSAL: {"answer_type": AnswerType.INFERRED, "min_confidence": 0.55},
    QueryType.LIST: {"answer_type": AnswerType.DIRECT, "min_confidence": 0.70},
    QueryType.PRECAUTION: {"answer_type": AnswerType.INFERRED, "min_confidence": 0.50},
    QueryType.TEMPORAL: {"answer_type": AnswerType.DIRECT, "min_confidence": 0.70},
    QueryType.LOCATION: {"answer_type": AnswerType.DIRECT, "min_confidence": 0.70},
    QueryType.QUANTITATIVE: {"answer_type": AnswerType.DIRECT, "min_confidence": 0.65},
    QueryType.RELATIONAL: {"answer_type": AnswerType.INFERRED, "min_confidence": 0.55},
    QueryType.GENERAL: {"answer_type": AnswerType.INFERRED, "min_confidence": 0.50},
}


# Complexity assessment thresholds
COMPLEXITY_THRESHOLDS: dict[Complexity, dict[str, Any]] = {
    Complexity.SIMPLE: {"min_evidence": 3, "min_relevance": 0.6, "max_concepts": 1},
    Complexity.MODERATE: {"min_evidence": 5, "min_relevance": 0.5, "max_concepts": 3},
    Complexity.COMPLEX: {"min_evidence": 8, "min_relevance": 0.4, "max_concepts": 5},
    Complexity.MULTI_HOP: {"min_evidence": 10, "min_relevance": 0.3, "max_concepts": 10},
}


# Domain detection keywords
DOMAIN_KEYWORDS: dict[Domain, dict[str, list[str]]] = {
    Domain.SECURITY: {
        "ko": ["보안", "취약점", "해킹", "공격", "방어", "암호", "인증", "권한", "침입", "악성코드",
               "랜섬웨어", "피싱", "CVE", "레드팀", "블루팀", "침투테스트", "SIEM", "SOC"],
        "en": ["security", "vulnerability", "attack", "defense", "encryption", "authentication",
               "malware", "ransomware", "phishing", "penetration", "CVE", "red team", "blue team",
               "SIEM", "SOC", "threat", "exploit", "firewall", "IDS", "IPS"],
    },
    Domain.FINANCE: {
        "ko": ["금융", "투자", "주식", "채권", "펀드", "은행", "대출", "이자", "수익률", "리스크"],
        "en": ["finance", "investment", "stock", "bond", "fund", "bank", "loan", "interest",
               "return", "risk", "portfolio", "trading", "market", "asset", "equity"],
    },
    Domain.MEDICAL: {
        "ko": ["의료", "질병", "치료", "진단", "증상", "약물", "환자", "수술", "병원"],
        "en": ["medical", "disease", "treatment", "diagnosis", "symptom", "drug", "patient",
               "surgery", "hospital", "clinical", "therapy", "pharmaceutical"],
    },
    Domain.LEGAL: {
        "ko": ["법률", "판례", "소송", "계약", "규정", "조항", "재판", "변호사", "법원"],
        "en": ["legal", "law", "case", "contract", "regulation", "statute", "court",
               "attorney", "litigation", "jurisdiction", "precedent", "compliance"],
    },
    Domain.TECH: {
        "ko": ["기술", "소프트웨어", "프로그래밍", "API", "서버", "클라우드", "데이터베이스", "네트워크"],
        "en": ["technology", "software", "programming", "API", "server", "cloud", "database",
               "network", "system", "framework", "architecture", "deployment", "kubernetes"],
    },
    Domain.ACADEMIC: {
        "ko": ["연구", "논문", "학술", "이론", "실험", "방법론", "분석", "결과"],
        "en": ["research", "paper", "academic", "theory", "experiment", "methodology",
               "analysis", "study", "findings", "hypothesis", "peer review"],
    },
}


# Domain-specific expected entity types
DOMAIN_ENTITY_TYPES: dict[Domain, list[str]] = {
    Domain.SECURITY: ["Vulnerability", "Attack", "Defense", "Tool", "Actor", "TTP", "CVE", "Malware"],
    Domain.FINANCE: ["Company", "Asset", "Transaction", "Regulation", "Metric", "Person", "Market"],
    Domain.MEDICAL: ["Disease", "Treatment", "Drug", "Symptom", "Procedure", "Anatomy", "Patient"],
    Domain.LEGAL: ["Law", "Case", "Party", "Jurisdiction", "Precedent", "Statute", "Contract"],
    Domain.TECH: ["System", "Component", "Protocol", "Standard", "Framework", "API", "Service"],
    Domain.ACADEMIC: ["Concept", "Theory", "Method", "Finding", "Author", "Publication"],
    Domain.GENERAL: ["Concept", "Person", "Organization", "Event", "Location", "Date", "Product"],
}


# Domain-specific expected relation types
DOMAIN_RELATION_TYPES: dict[Domain, list[str]] = {
    Domain.SECURITY: ["EXPLOITS", "DEFENDS_AGAINST", "TARGETS", "USES", "MITIGATES", "DETECTS"],
    Domain.FINANCE: ["OWNS", "INVESTS_IN", "REGULATES", "TRADES", "REPORTS_TO", "MANAGES"],
    Domain.MEDICAL: ["TREATS", "CAUSES", "INDICATES", "INTERACTS_WITH", "PREVENTS", "DIAGNOSES"],
    Domain.LEGAL: ["CITES", "OVERRULES", "APPLIES_TO", "DEFINES", "RESTRICTS", "GOVERNS"],
    Domain.TECH: ["IMPLEMENTS", "DEPENDS_ON", "EXTENDS", "COMMUNICATES_WITH", "CONTAINS"],
    Domain.ACADEMIC: ["REFERENCES", "SUPPORTS", "CONTRADICTS", "BUILDS_ON", "PROPOSES"],
    Domain.GENERAL: ["RELATES_TO", "PART_OF", "DESCRIBES", "MENTIONS", "LOCATED_IN", "WORKS_FOR"],
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DocumentContext:
    """Context extracted from a document."""

    path: str = ""
    name: str = ""
    format: DocumentFormat = DocumentFormat.UNKNOWN
    language: Language = Language.UNKNOWN
    domain: Domain = Domain.GENERAL
    size_bytes: int = 0
    size_chars: int = 0

    # Detected characteristics
    detected_keywords: list[str] = field(default_factory=list)
    detected_entity_types: list[str] = field(default_factory=list)
    domain_confidence: float = 0.0

    # Expectations
    expected_entity_types: list[str] = field(default_factory=list)
    expected_relation_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "format": self.format.value,
            "language": self.language.value,
            "domain": self.domain.value,
            "size_bytes": self.size_bytes,
            "size_chars": self.size_chars,
            "detected_keywords": self.detected_keywords,
            "domain_confidence": self.domain_confidence,
            "expected_entity_types": self.expected_entity_types,
            "expected_relation_types": self.expected_relation_types,
        }


@dataclass
class QueryContext:
    """Context extracted from a query."""

    query: str = ""
    language: Language = Language.UNKNOWN
    query_type: QueryType = QueryType.GENERAL
    complexity: Complexity = Complexity.SIMPLE

    # Classification details
    detected_patterns: list[str] = field(default_factory=list)
    concept_count: int = 0
    type_confidence: float = 0.0
    complexity_confidence: float = 0.0

    # Expectations
    expected_answer_type: AnswerType = AnswerType.INFERRED
    min_confidence: float = 0.5
    min_evidence_count: int = 3
    min_relevance_score: float = 0.5

    # Extracted entities from query
    potential_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "language": self.language.value,
            "query_type": self.query_type.value,
            "complexity": self.complexity.value,
            "detected_patterns": self.detected_patterns,
            "concept_count": self.concept_count,
            "type_confidence": self.type_confidence,
            "complexity_confidence": self.complexity_confidence,
            "expected_answer_type": self.expected_answer_type.value,
            "min_confidence": self.min_confidence,
            "min_evidence_count": self.min_evidence_count,
            "min_relevance_score": self.min_relevance_score,
            "potential_entities": self.potential_entities,
        }


@dataclass
class ValidationContext:
    """Combined validation context."""

    document: DocumentContext | None = None
    query: QueryContext | None = None

    # Overall validation parameters
    validation_mode: str = "full"  # "full", "ingestion_only", "query_only"
    strict_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict() if self.document else None,
            "query": self.query.to_dict() if self.query else None,
            "validation_mode": self.validation_mode,
            "strict_mode": self.strict_mode,
        }


# =============================================================================
# Context Extractor
# =============================================================================


class ContextExtractor:
    """
    Extracts and classifies context from documents and queries.

    Provides:
    - Document format and language detection
    - Domain classification based on content
    - Query type classification
    - Complexity assessment
    - Expectation setting for validation
    """

    def __init__(self):
        self._query_patterns = QUERY_TYPE_PATTERNS
        self._domain_keywords = DOMAIN_KEYWORDS

    def extract_document_context(
        self,
        path: str | None = None,
        content: str | None = None,
        format_hint: str | None = None,
    ) -> DocumentContext:
        """
        Extract context from a document.

        Args:
            path: Document file path
            content: Document content (optional, for content-based detection)
            format_hint: Format hint if known

        Returns:
            DocumentContext with extracted information
        """
        ctx = DocumentContext()

        # Extract path info
        if path:
            ctx.path = path
            p = Path(path)
            ctx.name = p.name
            ctx.format = self._detect_format(p.suffix, format_hint)

        # Analyze content if provided
        if content:
            ctx.size_bytes = len(content.encode("utf-8"))
            ctx.size_chars = len(content)
            ctx.language = self._detect_language(content)
            ctx.domain, ctx.domain_confidence, ctx.detected_keywords = self._detect_domain(content)

        # Set expectations based on domain
        ctx.expected_entity_types = DOMAIN_ENTITY_TYPES.get(ctx.domain, DOMAIN_ENTITY_TYPES[Domain.GENERAL])
        ctx.expected_relation_types = DOMAIN_RELATION_TYPES.get(ctx.domain, DOMAIN_RELATION_TYPES[Domain.GENERAL])

        logger.info(
            "Document context extracted",
            name=ctx.name,
            format=ctx.format.value,
            language=ctx.language.value,
            domain=ctx.domain.value,
            domain_confidence=ctx.domain_confidence,
        )

        return ctx

    def extract_query_context(self, query: str) -> QueryContext:
        """
        Extract context from a query.

        Args:
            query: The user query

        Returns:
            QueryContext with classification and expectations
        """
        ctx = QueryContext(query=query)

        # Detect language
        ctx.language = self._detect_language(query)

        # Classify query type
        ctx.query_type, ctx.detected_patterns, ctx.type_confidence = self._classify_query_type(query)

        # Assess complexity
        ctx.complexity, ctx.concept_count, ctx.complexity_confidence = self._assess_complexity(query)

        # Extract potential entities
        ctx.potential_entities = self._extract_potential_entities(query)

        # Set expectations based on query type and complexity
        expectations = QUERY_TYPE_EXPECTATIONS.get(ctx.query_type, QUERY_TYPE_EXPECTATIONS[QueryType.GENERAL])
        ctx.expected_answer_type = expectations["answer_type"]
        ctx.min_confidence = expectations["min_confidence"]

        complexity_thresholds = COMPLEXITY_THRESHOLDS.get(ctx.complexity, COMPLEXITY_THRESHOLDS[Complexity.SIMPLE])
        ctx.min_evidence_count = complexity_thresholds["min_evidence"]
        ctx.min_relevance_score = complexity_thresholds["min_relevance"]

        logger.info(
            "Query context extracted",
            query=query[:50],
            language=ctx.language.value,
            type=ctx.query_type.value,
            complexity=ctx.complexity.value,
            concepts=ctx.concept_count,
        )

        return ctx

    def extract_full_context(
        self,
        document_path: str | None = None,
        document_content: str | None = None,
        query: str | None = None,
        validation_mode: str = "full",
        strict_mode: bool = False,
    ) -> ValidationContext:
        """
        Extract full validation context from document and/or query.

        Args:
            document_path: Path to document
            document_content: Document content
            query: User query
            validation_mode: "full", "ingestion_only", or "query_only"
            strict_mode: Use stricter validation thresholds

        Returns:
            Complete ValidationContext
        """
        ctx = ValidationContext(validation_mode=validation_mode, strict_mode=strict_mode)

        if document_path or document_content:
            ctx.document = self.extract_document_context(document_path, document_content)

        if query:
            ctx.query = self.extract_query_context(query)

        # Adjust expectations for strict mode
        if strict_mode and ctx.query:
            ctx.query.min_confidence *= 1.2
            ctx.query.min_evidence_count = int(ctx.query.min_evidence_count * 1.5)
            ctx.query.min_relevance_score = min(0.8, ctx.query.min_relevance_score * 1.2)

        return ctx

    def _detect_format(self, suffix: str, hint: str | None = None) -> DocumentFormat:
        """Detect document format from file extension."""
        if hint:
            suffix = f".{hint.lower().strip('.')}"

        suffix = suffix.lower()
        format_map = {
            ".pdf": DocumentFormat.PDF,
            ".docx": DocumentFormat.DOCX,
            ".doc": DocumentFormat.DOCX,
            ".md": DocumentFormat.MD,
            ".markdown": DocumentFormat.MD,
            ".html": DocumentFormat.HTML,
            ".htm": DocumentFormat.HTML,
            ".txt": DocumentFormat.TXT,
            ".json": DocumentFormat.JSON,
            ".csv": DocumentFormat.CSV,
            ".xml": DocumentFormat.XML,
            ".yaml": DocumentFormat.YAML,
            ".yml": DocumentFormat.YAML,
        }
        return format_map.get(suffix, DocumentFormat.UNKNOWN)

    def _detect_language(self, text: str) -> Language:
        """Detect text language (Korean, English, or mixed)."""
        if not text:
            return Language.UNKNOWN

        # Count Korean characters (Hangul range: AC00-D7A3)
        korean_chars = len(re.findall(r'[\uAC00-\uD7A3]', text))
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', text))

        total = korean_chars + english_chars
        if total == 0:
            return Language.UNKNOWN

        korean_ratio = korean_chars / total
        english_ratio = english_chars / total

        if korean_ratio > 0.7:
            return Language.KOREAN
        elif english_ratio > 0.7:
            return Language.ENGLISH
        elif korean_ratio > 0.2 and english_ratio > 0.2:
            return Language.MIXED
        elif korean_ratio > english_ratio:
            return Language.KOREAN
        else:
            return Language.ENGLISH

    def _detect_domain(self, text: str) -> tuple[Domain, float, list[str]]:
        """
        Detect domain from text content.

        Returns:
            Tuple of (domain, confidence, detected_keywords)
        """
        text_lower = text.lower()
        domain_scores: dict[Domain, tuple[int, list[str]]] = {}

        for domain, keywords in self._domain_keywords.items():
            matches = []
            for lang_keywords in keywords.values():
                for keyword in lang_keywords:
                    if keyword.lower() in text_lower:
                        matches.append(keyword)
            domain_scores[domain] = (len(matches), matches)

        # Find domain with highest score
        best_domain = Domain.GENERAL
        best_score = 0
        best_keywords: list[str] = []

        for domain, (score, keywords) in domain_scores.items():
            if score > best_score:
                best_score = score
                best_domain = domain
                best_keywords = keywords

        # Calculate confidence (normalized by max possible matches)
        max_possible = sum(
            len(kw)
            for kw in self._domain_keywords.get(best_domain, {}).values()
        )
        confidence = min(1.0, best_score / max(max_possible * 0.3, 1))

        # Require minimum score to assign non-general domain
        if best_score < 2:
            return Domain.GENERAL, 0.0, []

        return best_domain, confidence, best_keywords[:10]  # Limit keywords

    def _classify_query_type(self, query: str) -> tuple[QueryType, list[str], float]:
        """
        Classify query type based on patterns.

        Returns:
            Tuple of (query_type, detected_patterns, confidence)
        """
        query_lower = query.lower()
        type_scores: dict[QueryType, list[str]] = {}

        for qtype, patterns in self._query_patterns.items():
            matches = []
            for lang_patterns in patterns.values():
                for pattern in lang_patterns:
                    if pattern.lower() in query_lower:
                        matches.append(pattern)
            if matches:
                type_scores[qtype] = matches

        # Find best match
        best_type = QueryType.GENERAL
        best_patterns: list[str] = []
        best_score = 0

        for qtype, patterns in type_scores.items():
            if len(patterns) > best_score:
                best_score = len(patterns)
                best_type = qtype
                best_patterns = patterns

        # Calculate confidence
        confidence = min(1.0, best_score * 0.4) if best_score > 0 else 0.3

        return best_type, best_patterns, confidence

    def _assess_complexity(self, query: str) -> tuple[Complexity, int, float]:
        """
        Assess query complexity.

        Returns:
            Tuple of (complexity, concept_count, confidence)
        """
        # Count potential concepts (rough heuristic)
        # - Named entities (capitalized words)
        # - Technical terms (words with numbers or special patterns)
        # - Quoted terms

        concepts = set()

        # Capitalized words (potential entities)
        caps_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        concepts.update(caps_words)

        # Korean nouns (followed by particles)
        korean_nouns = re.findall(r'[\uAC00-\uD7A3]+(?=[은는이가을를])', query)
        concepts.update(korean_nouns)

        # Quoted terms
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        concepts.update(quoted)

        # Technical terms (e.g., "CVE-2024-1234", "API", "REST")
        tech_terms = re.findall(r'\b[A-Z]{2,}(?:-\d+)*\b', query)
        concepts.update(tech_terms)

        concept_count = len(concepts)

        # Determine complexity
        if concept_count <= 1:
            complexity = Complexity.SIMPLE
        elif concept_count <= 3:
            complexity = Complexity.MODERATE
        elif concept_count <= 5:
            complexity = Complexity.COMPLEX
        else:
            complexity = Complexity.MULTI_HOP

        # Check for complexity indicators
        complexity_indicators = [
            "and", "또한", "그리고", "관계", "relationship", "between",
            "compared to", "비교", "차이", "통해", "through"
        ]
        indicator_count = sum(1 for ind in complexity_indicators if ind.lower() in query.lower())

        if indicator_count >= 2 and complexity != Complexity.MULTI_HOP:
            # Upgrade complexity
            complexity_order = [Complexity.SIMPLE, Complexity.MODERATE, Complexity.COMPLEX, Complexity.MULTI_HOP]
            current_idx = complexity_order.index(complexity)
            complexity = complexity_order[min(current_idx + 1, len(complexity_order) - 1)]

        confidence = 0.7 if concept_count > 0 else 0.5

        return complexity, concept_count, confidence

    def _extract_potential_entities(self, query: str) -> list[str]:
        """Extract potential entity names from query."""
        entities = []

        # Capitalized sequences
        caps_sequences = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', query)
        entities.extend(caps_sequences)

        # Korean noun phrases (simplified)
        korean_phrases = re.findall(r'[\uAC00-\uD7A3]{2,}', query)
        entities.extend(korean_phrases)

        # Quoted terms
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        entities.extend(quoted)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e.lower() not in seen and len(e) > 1:
                seen.add(e.lower())
                unique_entities.append(e)

        return unique_entities[:10]  # Limit to 10

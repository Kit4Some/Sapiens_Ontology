"""
Domain Profile Management.

Provides domain-specific extraction configurations:
- Pre-built profiles for common domains (tech, medical, legal, finance)
- Custom profile support via YAML/JSON
- Profile inheritance and composition
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.sddi.extractors.schema.entity_schema import (
    EntityTypeDefinition,
    EntityTypeRegistry,
    create_entity_type,
    create_base_entity_registry,
)
from src.sddi.extractors.schema.relation_schema import (
    PredicateDefinition,
    PredicateRegistry,
    create_predicate,
    create_base_predicate_registry,
)

logger = structlog.get_logger(__name__)


@dataclass
class ExtractionSchema:
    """
    Combined extraction schema with entity types and predicates.

    This is the primary schema object used by extractors.
    """

    entity_registry: EntityTypeRegistry
    predicate_registry: PredicateRegistry

    # Extraction settings
    min_entity_confidence: float = 0.5
    min_relation_confidence: float = 0.5
    max_entities_per_chunk: int = 50
    max_relations_per_chunk: int = 100

    # Prompt customization
    entity_guidelines: list[str] = field(default_factory=list)
    relation_guidelines: list[str] = field(default_factory=list)
    few_shot_examples: list[dict[str, Any]] = field(default_factory=list)

    # Language support
    primary_language: str = "en"
    supported_languages: list[str] = field(default_factory=lambda: ["en"])

    def get_entity_types_prompt(self, include_examples: bool = True) -> str:
        """Generate entity types section for prompts."""
        return self.entity_registry.generate_prompt_section(include_examples)

    def get_predicates_prompt(
        self,
        include_examples: bool = True,
        categories: list[str] | None = None,
    ) -> str:
        """Generate predicates section for prompts."""
        return self.predicate_registry.generate_prompt_section(
            include_examples, categories
        )

    def get_guidelines_prompt(self, for_entities: bool = True) -> str:
        """Generate guidelines section for prompts."""
        guidelines = self.entity_guidelines if for_entities else self.relation_guidelines
        if not guidelines:
            return ""
        lines = ["## Guidelines"]
        for i, guideline in enumerate(guidelines, 1):
            lines.append(f"{i}. {guideline}")
        return "\n".join(lines)

    def resolve_entity_type(self, type_str: str) -> str:
        """Resolve entity type name or alias."""
        return self.entity_registry.resolve_type(type_str)

    def resolve_predicate(self, predicate: str) -> str:
        """Resolve predicate name or alias."""
        return self.predicate_registry.resolve_name(predicate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_registry": self.entity_registry.to_dict(),
            "predicate_registry": self.predicate_registry.to_dict(),
            "min_entity_confidence": self.min_entity_confidence,
            "min_relation_confidence": self.min_relation_confidence,
            "max_entities_per_chunk": self.max_entities_per_chunk,
            "max_relations_per_chunk": self.max_relations_per_chunk,
            "entity_guidelines": self.entity_guidelines,
            "relation_guidelines": self.relation_guidelines,
            "primary_language": self.primary_language,
            "supported_languages": self.supported_languages,
        }


@dataclass
class DomainProfile:
    """
    Domain-specific extraction profile.

    Combines entity types, predicates, and domain knowledge.
    """

    name: str
    display_name: str
    description: str

    # Schema components
    entity_types: list[EntityTypeDefinition] = field(default_factory=list)
    predicates: list[PredicateDefinition] = field(default_factory=list)

    # Inheritance
    base_profiles: list[str] = field(default_factory=list)  # Names of profiles to inherit from
    include_base_entities: bool = True
    include_base_predicates: bool = True

    # Domain-specific settings
    entity_guidelines: list[str] = field(default_factory=list)
    relation_guidelines: list[str] = field(default_factory=list)

    # Confidence thresholds (can override defaults)
    min_entity_confidence: float | None = None
    min_relation_confidence: float | None = None

    # Language
    primary_language: str = "en"
    supported_languages: list[str] = field(default_factory=lambda: ["en"])

    # Metadata
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "entity_types": [e.to_dict() for e in self.entity_types],
            "predicates": [p.to_dict() for p in self.predicates],
            "base_profiles": self.base_profiles,
            "include_base_entities": self.include_base_entities,
            "include_base_predicates": self.include_base_predicates,
            "entity_guidelines": self.entity_guidelines,
            "relation_guidelines": self.relation_guidelines,
            "min_entity_confidence": self.min_entity_confidence,
            "min_relation_confidence": self.min_relation_confidence,
            "primary_language": self.primary_language,
            "supported_languages": self.supported_languages,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainProfile":
        entity_types = [
            EntityTypeDefinition.from_dict(e)
            for e in data.get("entity_types", [])
        ]
        predicates = [
            PredicateDefinition.from_dict(p)
            for p in data.get("predicates", [])
        ]

        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            description=data.get("description", ""),
            entity_types=entity_types,
            predicates=predicates,
            base_profiles=data.get("base_profiles", []),
            include_base_entities=data.get("include_base_entities", True),
            include_base_predicates=data.get("include_base_predicates", True),
            entity_guidelines=data.get("entity_guidelines", []),
            relation_guidelines=data.get("relation_guidelines", []),
            min_entity_confidence=data.get("min_entity_confidence"),
            min_relation_confidence=data.get("min_relation_confidence"),
            primary_language=data.get("primary_language", "en"),
            supported_languages=data.get("supported_languages", ["en"]),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            tags=data.get("tags", []),
        )


class DomainProfileManager:
    """
    Manages domain profiles and creates extraction schemas.

    Features:
    - Profile registration and lookup
    - Profile inheritance resolution
    - Schema generation from profiles
    """

    def __init__(self):
        self._profiles: dict[str, DomainProfile] = {}
        self._register_builtin_profiles()

    def _register_builtin_profiles(self) -> None:
        """Register built-in domain profiles."""
        for profile in BUILTIN_PROFILES:
            self.register(profile)

    def register(self, profile: DomainProfile) -> None:
        """Register a domain profile."""
        self._profiles[profile.name.lower()] = profile
        logger.debug("Registered domain profile", name=profile.name)

    def get(self, name: str) -> DomainProfile | None:
        """Get a profile by name."""
        return self._profiles.get(name.lower())

    def list_profiles(self) -> list[str]:
        """List all registered profile names."""
        return list(self._profiles.keys())

    def create_schema(
        self,
        profile_name: str | None = None,
        profile: DomainProfile | None = None,
    ) -> ExtractionSchema:
        """
        Create an extraction schema from a profile.

        Args:
            profile_name: Name of registered profile to use
            profile: Profile object (takes precedence over name)

        Returns:
            ExtractionSchema configured for the profile
        """
        if profile is None:
            if profile_name:
                profile = self.get(profile_name)
                if not profile:
                    raise ValueError(f"Profile not found: {profile_name}")
            else:
                # Default: base profile with all types
                profile = DomainProfile(
                    name="default",
                    display_name="Default",
                    description="Default extraction profile",
                    include_base_entities=True,
                    include_base_predicates=True,
                )

        # Create registries
        entity_registry = EntityTypeRegistry()
        predicate_registry = PredicateRegistry()

        # Add base types if requested
        if profile.include_base_entities:
            base_entity_reg = create_base_entity_registry()
            for type_def in base_entity_reg.get_all(enabled_only=False):
                entity_registry.register(type_def)

        if profile.include_base_predicates:
            base_pred_reg = create_base_predicate_registry()
            for pred_def in base_pred_reg.get_all(enabled_only=False):
                predicate_registry.register(pred_def)

        # Handle inheritance
        for base_name in profile.base_profiles:
            base_profile = self.get(base_name)
            if base_profile:
                for type_def in base_profile.entity_types:
                    entity_registry.register(type_def)
                for pred_def in base_profile.predicates:
                    predicate_registry.register(pred_def)

        # Add profile-specific types
        for type_def in profile.entity_types:
            entity_registry.register(type_def)

        for pred_def in profile.predicates:
            predicate_registry.register(pred_def)

        # Build guidelines
        entity_guidelines = profile.entity_guidelines or [
            "Extract ALL entities, even if they appear multiple times",
            "Use the most specific entity type possible",
            "Include brief descriptions when context provides them",
            "Assign confidence scores (0.0-1.0) based on certainty",
        ]

        relation_guidelines = profile.relation_guidelines or [
            "Only extract relations between the provided entities",
            "Use clear, concise predicates in UPPER_SNAKE_CASE",
            "Each relation should be directional (source → target)",
            "Assign confidence scores based on how explicit the relation is",
        ]

        return ExtractionSchema(
            entity_registry=entity_registry,
            predicate_registry=predicate_registry,
            min_entity_confidence=profile.min_entity_confidence or 0.5,
            min_relation_confidence=profile.min_relation_confidence or 0.5,
            entity_guidelines=entity_guidelines,
            relation_guidelines=relation_guidelines,
            primary_language=profile.primary_language,
            supported_languages=profile.supported_languages,
        )

    def load_from_file(self, path: str | Path) -> DomainProfile:
        """Load a profile from YAML or JSON file."""
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            return load_profile_from_yaml(path)
        elif path.suffix == ".json":
            return load_profile_from_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def save_profile(self, profile: DomainProfile, path: str | Path) -> None:
        """Save a profile to file."""
        path = Path(path)
        data = profile.to_dict()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                raise ImportError("PyYAML required for YAML support")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


def load_profile_from_yaml(path: str | Path) -> DomainProfile:
    """Load a domain profile from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML support: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return DomainProfile.from_dict(data)


def load_profile_from_json(path: str | Path) -> DomainProfile:
    """Load a domain profile from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DomainProfile.from_dict(data)


# =============================================================================
# Built-in Domain Profiles
# =============================================================================

TECH_PROFILE = DomainProfile(
    name="technology",
    display_name="Technology",
    description="Technology and software development domain",
    entity_types=[
        create_entity_type(
            name="API",
            description="Application Programming Interfaces",
            examples=["REST API", "GraphQL API", "OpenAI API"],
            aliases=["ENDPOINT", "SERVICE"],
        ),
        create_entity_type(
            name="DATABASE",
            description="Database systems and data stores",
            examples=["PostgreSQL", "MongoDB", "Redis"],
            aliases=["DB", "DATASTORE"],
        ),
        create_entity_type(
            name="CLOUD_SERVICE",
            description="Cloud computing services",
            examples=["AWS Lambda", "Google Cloud Storage", "Azure Functions"],
        ),
        create_entity_type(
            name="PROGRAMMING_LANGUAGE",
            description="Programming and scripting languages",
            examples=["Python", "JavaScript", "Rust"],
            aliases=["LANG"],
        ),
        create_entity_type(
            name="FRAMEWORK",
            description="Software frameworks and libraries",
            examples=["React", "Django", "TensorFlow"],
            aliases=["LIBRARY", "LIB"],
        ),
        create_entity_type(
            name="VERSION",
            description="Software versions and releases",
            examples=["v2.0", "Python 3.11", "Node.js 18 LTS"],
        ),
    ],
    predicates=[
        create_predicate(
            name="DEPENDS_ON",
            description="Software dependency relationship",
            examples=["React DEPENDS_ON Node.js"],
            category="technical",
        ),
        create_predicate(
            name="IMPLEMENTS",
            description="Implementation relationship",
            examples=["Django IMPLEMENTS MVC"],
            category="technical",
            aliases=["FOLLOWS"],
        ),
        create_predicate(
            name="INTEGRATES_WITH",
            description="Integration between systems",
            examples=["Slack INTEGRATES_WITH Jira"],
            symmetric=True,
            category="technical",
        ),
        create_predicate(
            name="RUNS_ON",
            description="Platform/runtime relationship",
            examples=["Docker RUNS_ON Linux"],
            category="technical",
        ),
        create_predicate(
            name="SUPERSEDES",
            description="Version succession",
            examples=["Python 3 SUPERSEDES Python 2"],
            category="technical",
            inverse="SUPERSEDED_BY",
        ),
    ],
    entity_guidelines=[
        "Include version numbers when mentioned",
        "Distinguish between programming languages and frameworks",
        "Extract API names including their providers",
    ],
    tags=["tech", "software", "development"],
)

MEDICAL_PROFILE = DomainProfile(
    name="medical",
    display_name="Medical/Healthcare",
    description="Medical and healthcare domain",
    entity_types=[
        create_entity_type(
            name="DISEASE",
            description="Diseases, conditions, disorders",
            examples=["COVID-19", "Diabetes", "Hypertension"],
            aliases=["CONDITION", "DISORDER", "ILLNESS"],
        ),
        create_entity_type(
            name="DRUG",
            description="Medications, drugs, treatments",
            examples=["Aspirin", "Metformin", "Remdesivir"],
            aliases=["MEDICATION", "MEDICINE", "TREATMENT"],
        ),
        create_entity_type(
            name="SYMPTOM",
            description="Medical symptoms and signs",
            examples=["fever", "headache", "fatigue"],
        ),
        create_entity_type(
            name="PROCEDURE",
            description="Medical procedures and surgeries",
            examples=["MRI", "appendectomy", "blood test"],
            aliases=["SURGERY", "TEST"],
        ),
        create_entity_type(
            name="ANATOMY",
            description="Body parts and anatomical structures",
            examples=["heart", "liver", "brain"],
            aliases=["BODY_PART", "ORGAN"],
        ),
        create_entity_type(
            name="GENE",
            description="Genes and genetic markers",
            examples=["BRCA1", "TP53", "EGFR"],
            aliases=["GENETIC_MARKER"],
        ),
    ],
    predicates=[
        create_predicate(
            name="TREATS",
            description="Drug treats condition",
            examples=["Aspirin TREATS headache"],
            source_types=["DRUG"],
            target_types=["DISEASE", "SYMPTOM"],
            category="medical",
        ),
        create_predicate(
            name="CAUSES",
            description="Causal relationship",
            examples=["Smoking CAUSES lung cancer"],
            category="medical",
        ),
        create_predicate(
            name="SYMPTOM_OF",
            description="Symptom indicates condition",
            examples=["Fever SYMPTOM_OF COVID-19"],
            source_types=["SYMPTOM"],
            target_types=["DISEASE"],
            category="medical",
        ),
        create_predicate(
            name="DIAGNOSES",
            description="Procedure diagnoses condition",
            examples=["MRI DIAGNOSES tumor"],
            source_types=["PROCEDURE"],
            target_types=["DISEASE"],
            category="medical",
        ),
        create_predicate(
            name="INTERACTS_WITH",
            description="Drug-drug interaction",
            examples=["Aspirin INTERACTS_WITH Warfarin"],
            source_types=["DRUG"],
            target_types=["DRUG"],
            symmetric=True,
            category="medical",
        ),
        create_predicate(
            name="ASSOCIATED_WITH",
            description="Gene-disease association",
            examples=["BRCA1 ASSOCIATED_WITH breast cancer"],
            source_types=["GENE"],
            target_types=["DISEASE"],
            category="medical",
        ),
    ],
    min_entity_confidence=0.6,  # Higher threshold for medical
    tags=["medical", "healthcare", "clinical"],
)

FINANCE_PROFILE = DomainProfile(
    name="finance",
    display_name="Finance",
    description="Financial and business domain",
    entity_types=[
        create_entity_type(
            name="FINANCIAL_INSTRUMENT",
            description="Stocks, bonds, derivatives, securities",
            examples=["AAPL stock", "Treasury bonds", "S&P 500"],
            aliases=["STOCK", "BOND", "SECURITY", "TICKER"],
        ),
        create_entity_type(
            name="CURRENCY",
            description="Currencies and monetary units",
            examples=["USD", "Bitcoin", "Euro"],
            aliases=["MONEY"],
        ),
        create_entity_type(
            name="FINANCIAL_METRIC",
            description="Financial metrics and ratios",
            examples=["P/E ratio", "ROI", "EBITDA"],
            aliases=["RATIO", "KPI"],
        ),
        create_entity_type(
            name="REGULATION",
            description="Financial regulations and compliance",
            examples=["SEC Rule 10b-5", "Basel III", "MiFID II"],
            aliases=["COMPLIANCE", "RULE"],
        ),
        create_entity_type(
            name="TRANSACTION",
            description="Financial transactions",
            examples=["IPO", "merger", "acquisition"],
            aliases=["DEAL"],
        ),
    ],
    predicates=[
        create_predicate(
            name="INVESTED_IN",
            description="Investment relationship",
            examples=["Berkshire INVESTED_IN Apple"],
            category="finance",
        ),
        create_predicate(
            name="REGULATED_BY",
            description="Regulatory oversight",
            examples=["Bank REGULATED_BY SEC"],
            category="finance",
        ),
        create_predicate(
            name="MERGED_WITH",
            description="Corporate merger",
            examples=["Sprint MERGED_WITH T-Mobile"],
            symmetric=True,
            category="finance",
        ),
        create_predicate(
            name="LISTED_ON",
            description="Stock exchange listing",
            examples=["Apple LISTED_ON NASDAQ"],
            source_types=["ORGANIZATION"],
            category="finance",
        ),
        create_predicate(
            name="VALUED_AT",
            description="Valuation relationship",
            examples=["Startup VALUED_AT $1B"],
            category="finance",
        ),
    ],
    tags=["finance", "business", "investment"],
)

LEGAL_PROFILE = DomainProfile(
    name="legal",
    display_name="Legal",
    description="Legal and regulatory domain",
    entity_types=[
        create_entity_type(
            name="LAW",
            description="Laws, statutes, regulations",
            examples=["GDPR", "CCPA", "Sarbanes-Oxley Act"],
            aliases=["STATUTE", "ACT", "LEGISLATION"],
        ),
        create_entity_type(
            name="COURT",
            description="Courts and tribunals",
            examples=["Supreme Court", "EU Court of Justice"],
        ),
        create_entity_type(
            name="CASE",
            description="Legal cases",
            examples=["Brown v. Board of Education", "Roe v. Wade"],
            aliases=["LAWSUIT", "LITIGATION"],
        ),
        create_entity_type(
            name="CONTRACT",
            description="Legal contracts and agreements",
            examples=["NDA", "SLA", "Employment Agreement"],
            aliases=["AGREEMENT"],
        ),
        create_entity_type(
            name="LEGAL_TERM",
            description="Legal terminology and concepts",
            examples=["negligence", "liability", "jurisdiction"],
        ),
    ],
    predicates=[
        create_predicate(
            name="GOVERNS",
            description="Law governs subject matter",
            examples=["GDPR GOVERNS data privacy"],
            source_types=["LAW"],
            category="legal",
        ),
        create_predicate(
            name="FILED_IN",
            description="Case filed in court",
            examples=["Case FILED_IN District Court"],
            source_types=["CASE"],
            target_types=["COURT"],
            category="legal",
        ),
        create_predicate(
            name="CITES",
            description="Legal citation",
            examples=["Case A CITES Case B"],
            source_types=["CASE", "DOCUMENT"],
            target_types=["CASE", "LAW"],
            category="legal",
        ),
        create_predicate(
            name="AMENDS",
            description="Law amends another",
            examples=["Amendment AMENDS Constitution"],
            source_types=["LAW"],
            target_types=["LAW"],
            category="legal",
        ),
        create_predicate(
            name="PARTY_TO",
            description="Entity is party to agreement",
            examples=["Company PARTY_TO Contract"],
            target_types=["CONTRACT", "CASE"],
            category="legal",
        ),
    ],
    tags=["legal", "compliance", "regulatory"],
)

ACADEMIC_PROFILE = DomainProfile(
    name="academic",
    display_name="Academic/Research",
    description="Academic and research domain",
    entity_types=[
        create_entity_type(
            name="RESEARCHER",
            description="Researchers and academics",
            examples=["Dr. Jane Smith", "Prof. John Doe"],
            aliases=["SCIENTIST", "PROFESSOR", "SCHOLAR"],
        ),
        create_entity_type(
            name="PUBLICATION",
            description="Academic publications",
            examples=["Nature paper", "arXiv preprint"],
            aliases=["PAPER", "ARTICLE", "JOURNAL"],
        ),
        create_entity_type(
            name="INSTITUTION",
            description="Academic institutions",
            examples=["MIT", "Stanford University", "CERN"],
            aliases=["UNIVERSITY", "RESEARCH_CENTER"],
        ),
        create_entity_type(
            name="RESEARCH_FIELD",
            description="Fields of study",
            examples=["Machine Learning", "Quantum Physics"],
            aliases=["FIELD", "DISCIPLINE"],
        ),
        create_entity_type(
            name="DATASET",
            description="Research datasets",
            examples=["ImageNet", "MNIST", "Common Crawl"],
        ),
        create_entity_type(
            name="METHOD",
            description="Research methods and algorithms",
            examples=["Transformer", "BERT", "Gradient Descent"],
            aliases=["ALGORITHM", "MODEL"],
        ),
    ],
    predicates=[
        create_predicate(
            name="AUTHORED_BY",
            description="Publication authored by researcher",
            examples=["Paper AUTHORED_BY Dr. Smith"],
            source_types=["PUBLICATION"],
            target_types=["RESEARCHER", "PERSON"],
            inverse="AUTHORED",
            category="academic",
        ),
        create_predicate(
            name="AFFILIATED_WITH",
            description="Researcher affiliated with institution",
            examples=["Dr. Smith AFFILIATED_WITH MIT"],
            source_types=["RESEARCHER", "PERSON"],
            target_types=["INSTITUTION", "ORGANIZATION"],
            category="academic",
        ),
        create_predicate(
            name="CITES",
            description="Publication cites another",
            examples=["Paper A CITES Paper B"],
            source_types=["PUBLICATION"],
            target_types=["PUBLICATION"],
            category="academic",
        ),
        create_predicate(
            name="TRAINED_ON",
            description="Model trained on dataset",
            examples=["BERT TRAINED_ON Wikipedia"],
            source_types=["METHOD"],
            target_types=["DATASET"],
            category="academic",
        ),
        create_predicate(
            name="ADVANCES",
            description="Research advances a field",
            examples=["Paper ADVANCES NLP"],
            source_types=["PUBLICATION", "METHOD"],
            target_types=["RESEARCH_FIELD"],
            category="academic",
        ),
    ],
    tags=["academic", "research", "scientific"],
)

BUILTIN_PROFILES = [
    TECH_PROFILE,
    MEDICAL_PROFILE,
    FINANCE_PROFILE,
    LEGAL_PROFILE,
    ACADEMIC_PROFILE,
]


# Global profile manager
_profile_manager: DomainProfileManager | None = None


def get_profile_manager() -> DomainProfileManager:
    """Get the global profile manager."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = DomainProfileManager()
    return _profile_manager

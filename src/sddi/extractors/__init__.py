"""
SDDI Extractors Module.

Entity and relation extraction with:
- Hybrid NER (SpaCy + Patterns + LLM)
- Multi-stage entity resolution
- Canonical entity registry
- Extraction quality metrics
- Dynamic schema-based extraction (NEW)
- Configurable domain profiles (NEW)
"""

from src.sddi.extractors.entity_extractor import EntityExtractor, EntityType
from src.sddi.extractors.relation_extractor import RelationExtractor

# Advanced extraction components
from src.sddi.extractors.entity_resolver import (
    EntityResolver,
    EntityCluster,
    ResolutionCandidate,
)
from src.sddi.extractors.hybrid_extractor import (
    HybridEntityExtractor,
    ExtractionStrategy,
)
from src.sddi.extractors.quality_metrics import (
    ExtractionQualityAnalyzer,
    ExtractionQualityReport,
    QualityLevel,
)
from src.sddi.extractors.canonical_registry import (
    CanonicalEntityRegistry,
    CanonicalEntity,
    AliasType,
    AliasMatch,
    create_registry_with_defaults,
)

# Dynamic Schema System
from src.sddi.extractors.schema.entity_schema import (
    EntityTypeDefinition,
    EntityTypeRegistry,
    create_entity_type,
    create_base_entity_registry,
    BASE_ENTITY_TYPES,
)
from src.sddi.extractors.schema.relation_schema import (
    PredicateDefinition,
    PredicateRegistry,
    create_predicate,
    create_base_predicate_registry,
    BASE_PREDICATES,
)
from src.sddi.extractors.schema.domain_profile import (
    ExtractionSchema,
    DomainProfile,
    DomainProfileManager,
    get_profile_manager,
    BUILTIN_PROFILES,
)
from src.sddi.extractors.schema.prompt_builder import (
    DynamicPromptBuilder,
    PromptTemplate,
    create_prompt_builder,
)

# Configurable Extractors
from src.sddi.extractors.configurable_extractor import (
    ConfigurableEntityExtractor,
    ConfigurableRelationExtractor,
    create_entity_extractor,
    create_relation_extractor,
    create_extractors_for_domain,
)

# Resilient Extractor (Production-grade with Circuit Breaker, Adaptive Batching, DLQ)
from src.sddi.extractors.resilient_extractor import (
    ResilientEntityExtractor,
    ResilientExtractionMetrics,
    CircuitBreaker,
    CircuitState,
    AdaptiveBatchSizer,
    AdaptiveBatchConfig,
    ProcessingCheckpoint,
    DeadLetterQueue,
    DeadLetterItem,
    ClassifiedError,
    ErrorCategory,
)

__all__ = [
    # Core extractors (legacy)
    "EntityExtractor",
    "EntityType",
    "RelationExtractor",
    # Entity resolution
    "EntityResolver",
    "EntityCluster",
    "ResolutionCandidate",
    # Hybrid extraction
    "HybridEntityExtractor",
    "ExtractionStrategy",
    # Quality metrics
    "ExtractionQualityAnalyzer",
    "ExtractionQualityReport",
    "QualityLevel",
    # Canonical registry
    "CanonicalEntityRegistry",
    "CanonicalEntity",
    "AliasType",
    "AliasMatch",
    "create_registry_with_defaults",
    # Dynamic Schema System
    "EntityTypeDefinition",
    "EntityTypeRegistry",
    "create_entity_type",
    "create_base_entity_registry",
    "BASE_ENTITY_TYPES",
    "PredicateDefinition",
    "PredicateRegistry",
    "create_predicate",
    "create_base_predicate_registry",
    "BASE_PREDICATES",
    "ExtractionSchema",
    "DomainProfile",
    "DomainProfileManager",
    "get_profile_manager",
    "BUILTIN_PROFILES",
    "DynamicPromptBuilder",
    "PromptTemplate",
    "create_prompt_builder",
    # Configurable Extractors
    "ConfigurableEntityExtractor",
    "ConfigurableRelationExtractor",
    "create_entity_extractor",
    "create_relation_extractor",
    "create_extractors_for_domain",
    # Resilient Extractor
    "ResilientEntityExtractor",
    "ResilientExtractionMetrics",
    "CircuitBreaker",
    "CircuitState",
    "AdaptiveBatchSizer",
    "AdaptiveBatchConfig",
    "ProcessingCheckpoint",
    "DeadLetterQueue",
    "DeadLetterItem",
    "ClassifiedError",
    "ErrorCategory",
]

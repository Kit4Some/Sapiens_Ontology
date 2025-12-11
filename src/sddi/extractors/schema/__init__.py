"""
Dynamic Extraction Schema System.

Provides configurable, domain-specific schemas for entity and relation extraction:
- Dynamic entity type definitions with aliases and validation
- Type inheritance hierarchy (e.g., Person → Employee → Engineer)
- Configurable relation predicates with type constraints
- Cardinality constraints (1:1, 1:N, N:1, N:M)
- Domain profiles (tech, medical, legal, finance, academic)
- Custom schema support via YAML/JSON
- Dynamic prompt generation from schemas
- JSON-LD and Turtle ontology export
"""

from src.sddi.extractors.schema.entity_schema import (
    EntityTypeDefinition,
    EntityTypeRegistry,
    create_entity_type,
    create_base_entity_registry,
    create_extended_entity_registry,
    BASE_ENTITY_TYPES,
    EXTENDED_ENTITY_TYPES,
)
from src.sddi.extractors.schema.relation_schema import (
    Cardinality,
    PredicateDefinition,
    PredicateRegistry,
    create_predicate,
    create_base_predicate_registry,
    BASE_PREDICATES,
)
from src.sddi.extractors.schema.domain_profile import (
    DomainProfile,
    DomainProfileManager,
    ExtractionSchema,
    get_profile_manager,
    load_profile_from_yaml,
    load_profile_from_json,
    BUILTIN_PROFILES,
)
from src.sddi.extractors.schema.prompt_builder import (
    DynamicPromptBuilder,
    PromptTemplate,
    create_prompt_builder,
)
from src.sddi.extractors.schema.ontology_export import (
    OntologyExporter,
    export_ontology_jsonld,
    export_ontology_turtle,
    validate_ontology,
    JSONLD_CONTEXT,
)

__all__ = [
    # Entity schema
    "EntityTypeDefinition",
    "EntityTypeRegistry",
    "create_entity_type",
    "create_base_entity_registry",
    "create_extended_entity_registry",
    "BASE_ENTITY_TYPES",
    "EXTENDED_ENTITY_TYPES",
    # Relation schema
    "Cardinality",
    "PredicateDefinition",
    "PredicateRegistry",
    "create_predicate",
    "create_base_predicate_registry",
    "BASE_PREDICATES",
    # Domain profiles
    "DomainProfile",
    "DomainProfileManager",
    "ExtractionSchema",
    "get_profile_manager",
    "load_profile_from_yaml",
    "load_profile_from_json",
    "BUILTIN_PROFILES",
    # Prompt builder
    "DynamicPromptBuilder",
    "PromptTemplate",
    "create_prompt_builder",
    # Ontology export
    "OntologyExporter",
    "export_ontology_jsonld",
    "export_ontology_turtle",
    "validate_ontology",
    "JSONLD_CONTEXT",
]

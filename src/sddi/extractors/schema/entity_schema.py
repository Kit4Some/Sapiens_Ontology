"""
Dynamic Entity Type Schema.

Defines configurable entity types for extraction:
- Custom entity type definitions with descriptions
- Type inheritance hierarchy (e.g., Person → Employee → Engineer)
- Type aliases and variations
- Validation rules
- Examples for few-shot prompting
"""

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EntityTypeDefinition:
    """
    Definition of an entity type for extraction.

    Supports inheritance hierarchy for type specialization:
    - PERSON (base)
      └── EMPLOYEE (parent_type=PERSON)
          └── ENGINEER (parent_type=EMPLOYEE)

    Child types inherit properties from parent types and can be
    used where parent types are expected.

    Attributes:
        name: Internal type name (e.g., "PERSON", "ORGANIZATION")
        display_name: Human-readable name (e.g., "Person", "Organization")
        description: Description of what this type represents
        parent_type: Parent type name for inheritance (None = root type)
        abstract: Whether this type can be instantiated directly
        examples: Example entities of this type
        aliases: Alternative names that map to this type
        validation_patterns: Regex patterns for validation
        properties: Expected properties for this entity type
        inherited_properties: Properties inherited from parent (computed)
        priority: Priority for disambiguation (higher = preferred)
    """

    name: str
    display_name: str
    description: str

    # Inheritance
    parent_type: str | None = None  # Parent type name for inheritance
    abstract: bool = False  # Abstract types cannot be instantiated directly

    examples: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    validation_patterns: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    priority: int = 50
    enabled: bool = True

    # Extraction hints
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "parent_type": self.parent_type,
            "abstract": self.abstract,
            "examples": self.examples,
            "aliases": self.aliases,
            "properties": self.properties,
            "priority": self.priority,
            "keywords": self.keywords,
        }

    def to_prompt_format(self, include_examples: bool = True, include_parent: bool = False) -> str:
        """Format for inclusion in extraction prompts."""
        result = f"- {self.name}"
        if include_parent and self.parent_type:
            result += f" (extends {self.parent_type})"
        result += f": {self.description}"
        if include_examples and self.examples:
            examples_str = ", ".join(self.examples[:3])
            result += f" (e.g., {examples_str})"
        return result

    def is_subtype_of(self, other_type: str, registry: "EntityTypeRegistry") -> bool:
        """
        Check if this type is a subtype of another type.

        Args:
            other_type: Type name to check against
            registry: Registry for looking up parent types

        Returns:
            True if this type is same as or descends from other_type
        """
        if self.name.upper() == other_type.upper():
            return True

        if not self.parent_type:
            return False

        parent_def = registry.get(self.parent_type)
        if not parent_def:
            return False

        return parent_def.is_subtype_of(other_type, registry)

    def get_all_properties(self, registry: "EntityTypeRegistry") -> list[str]:
        """
        Get all properties including inherited ones.

        Args:
            registry: Registry for looking up parent types

        Returns:
            Combined list of own and inherited properties
        """
        own_props = list(self.properties)

        if not self.parent_type:
            return own_props

        parent_def = registry.get(self.parent_type)
        if not parent_def:
            return own_props

        # Parent properties come first, then own properties
        parent_props = parent_def.get_all_properties(registry)
        return parent_props + [p for p in own_props if p not in parent_props]

    def get_ancestry(self, registry: "EntityTypeRegistry") -> list[str]:
        """
        Get list of ancestor types from root to this type.

        Returns:
            List of type names from root ancestor to this type
        """
        if not self.parent_type:
            return [self.name]

        parent_def = registry.get(self.parent_type)
        if not parent_def:
            return [self.name]

        return parent_def.get_ancestry(registry) + [self.name]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityTypeDefinition":
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"].title()),
            description=data.get("description", ""),
            parent_type=data.get("parent_type"),
            abstract=data.get("abstract", False),
            examples=data.get("examples", []),
            aliases=data.get("aliases", []),
            validation_patterns=data.get("validation_patterns", []),
            properties=data.get("properties", []),
            priority=data.get("priority", 50),
            enabled=data.get("enabled", True),
            keywords=data.get("keywords", []),
            exclude_keywords=data.get("exclude_keywords", []),
        )


class EntityTypeRegistry:
    """
    Registry for managing entity type definitions.

    Provides:
    - Registration of custom entity types
    - Type lookup by name or alias
    - Type inheritance hierarchy support
    - Validation of entity types
    - Prompt generation helpers

    Inheritance Features:
    - Types can specify a parent_type for inheritance
    - Child types inherit properties from parents
    - is_compatible() checks if a type can be used where another is expected
    - get_subtypes() retrieves all descendants of a type
    """

    def __init__(self):
        self._types: dict[str, EntityTypeDefinition] = {}
        self._alias_map: dict[str, str] = {}  # alias -> type_name
        self._children_map: dict[str, list[str]] = {}  # parent -> children

    def register(self, type_def: EntityTypeDefinition) -> None:
        """Register an entity type definition."""
        name_upper = type_def.name.upper()
        self._types[name_upper] = type_def

        # Register aliases
        for alias in type_def.aliases:
            self._alias_map[alias.upper()] = name_upper

        # Track inheritance relationships
        if type_def.parent_type:
            parent_upper = type_def.parent_type.upper()
            if parent_upper not in self._children_map:
                self._children_map[parent_upper] = []
            if name_upper not in self._children_map[parent_upper]:
                self._children_map[parent_upper].append(name_upper)

        logger.debug(
            "Registered entity type",
            name=name_upper,
            parent=type_def.parent_type,
            aliases=type_def.aliases,
        )

    def register_many(self, type_defs: list[EntityTypeDefinition]) -> None:
        """Register multiple entity types."""
        for type_def in type_defs:
            self.register(type_def)

    def get(self, name: str) -> EntityTypeDefinition | None:
        """Get entity type by name or alias."""
        name_upper = name.upper()

        # Direct lookup
        if name_upper in self._types:
            return self._types[name_upper]

        # Alias lookup
        if name_upper in self._alias_map:
            return self._types[self._alias_map[name_upper]]

        return None

    def resolve_type(self, name: str) -> str:
        """Resolve type name or alias to canonical type name."""
        name_upper = name.upper()

        if name_upper in self._types:
            return name_upper

        if name_upper in self._alias_map:
            return self._alias_map[name_upper]

        # Return as-is if not found (will be treated as OTHER)
        return name_upper

    def get_all(self, enabled_only: bool = True) -> list[EntityTypeDefinition]:
        """Get all registered entity types."""
        types = list(self._types.values())
        if enabled_only:
            types = [t for t in types if t.enabled]
        return sorted(types, key=lambda t: (-t.priority, t.name))

    def get_type_names(self, enabled_only: bool = True) -> list[str]:
        """Get list of all type names."""
        return [t.name for t in self.get_all(enabled_only)]

    def contains(self, name: str) -> bool:
        """Check if type or alias is registered."""
        name_upper = name.upper()
        return name_upper in self._types or name_upper in self._alias_map

    # =========================================================================
    # Inheritance Methods
    # =========================================================================

    def is_compatible(self, child_type: str, parent_type: str) -> bool:
        """
        Check if child_type can be used where parent_type is expected.

        A type is compatible if:
        - It is the same type
        - It is a subtype (descendant) of the expected type

        Args:
            child_type: The type to check
            parent_type: The expected type

        Returns:
            True if child_type is same as or descends from parent_type
        """
        child_type = self.resolve_type(child_type)
        parent_type = self.resolve_type(parent_type)

        if child_type == parent_type:
            return True

        child_def = self.get(child_type)
        if not child_def:
            return False

        return child_def.is_subtype_of(parent_type, self)

    def get_subtypes(self, type_name: str, recursive: bool = True) -> list[str]:
        """
        Get all subtypes of a given type.

        Args:
            type_name: Parent type name
            recursive: Include descendants of descendants

        Returns:
            List of subtype names
        """
        type_upper = type_name.upper()
        direct_children = self._children_map.get(type_upper, [])

        if not recursive:
            return list(direct_children)

        all_subtypes = list(direct_children)
        for child in direct_children:
            all_subtypes.extend(self.get_subtypes(child, recursive=True))

        return all_subtypes

    def get_supertypes(self, type_name: str) -> list[str]:
        """
        Get all ancestor types of a given type.

        Args:
            type_name: Child type name

        Returns:
            List of ancestor type names (from immediate parent to root)
        """
        type_def = self.get(type_name)
        if not type_def or not type_def.parent_type:
            return []

        ancestors = [type_def.parent_type.upper()]
        parent_ancestors = self.get_supertypes(type_def.parent_type)
        ancestors.extend(parent_ancestors)

        return ancestors

    def get_root_types(self) -> list[EntityTypeDefinition]:
        """Get all types that have no parent (root types)."""
        return [t for t in self._types.values() if not t.parent_type]

    def get_type_hierarchy(self) -> dict[str, Any]:
        """
        Get the full type hierarchy as a nested dictionary.

        Returns:
            Nested dict representing the type tree
        """
        def build_subtree(type_name: str) -> dict[str, Any]:
            type_def = self.get(type_name)
            children = self._children_map.get(type_name.upper(), [])
            return {
                "name": type_name,
                "display_name": type_def.display_name if type_def else type_name,
                "abstract": type_def.abstract if type_def else False,
                "children": [build_subtree(child) for child in children],
            }

        roots = self.get_root_types()
        return {
            "hierarchy": [build_subtree(r.name) for r in roots]
        }

    def validate_hierarchy(self) -> list[str]:
        """
        Validate the type hierarchy for issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for type_name, type_def in self._types.items():
            if type_def.parent_type:
                parent_upper = type_def.parent_type.upper()
                if parent_upper not in self._types:
                    errors.append(
                        f"Type {type_name} has undefined parent type: {type_def.parent_type}"
                    )

                # Check for cycles
                visited = {type_name}
                current = parent_upper
                while current:
                    if current in visited:
                        errors.append(f"Cycle detected in type hierarchy involving: {type_name}")
                        break
                    visited.add(current)
                    current_def = self._types.get(current)
                    current = current_def.parent_type.upper() if current_def and current_def.parent_type else None

        return errors

    # =========================================================================
    # Prompt Generation
    # =========================================================================

    def generate_prompt_section(
        self,
        include_examples: bool = True,
        include_disabled: bool = False,
        include_hierarchy: bool = False,
    ) -> str:
        """Generate entity types section for prompts."""
        types = self.get_all(enabled_only=not include_disabled)
        lines = ["## Entity Types"]

        if include_hierarchy:
            # Group by hierarchy
            roots = [t for t in types if not t.parent_type]
            for root in roots:
                lines.append(root.to_prompt_format(include_examples, include_parent=True))
                self._add_children_to_prompt(root.name, types, lines, include_examples, indent=1)
        else:
            for type_def in types:
                lines.append(type_def.to_prompt_format(include_examples))

        return "\n".join(lines)

    def _add_children_to_prompt(
        self,
        parent_name: str,
        types: list[EntityTypeDefinition],
        lines: list[str],
        include_examples: bool,
        indent: int,
    ) -> None:
        """Recursively add child types to prompt with indentation."""
        children = [t for t in types if t.parent_type and t.parent_type.upper() == parent_name.upper()]
        for child in children:
            prefix = "  " * indent
            lines.append(prefix + child.to_prompt_format(include_examples, include_parent=True))
            self._add_children_to_prompt(child.name, types, lines, include_examples, indent + 1)

    def to_dict(self) -> dict[str, Any]:
        """Export registry to dictionary."""
        return {
            "types": [t.to_dict() for t in self._types.values()],
            "alias_map": dict(self._alias_map),
            "children_map": dict(self._children_map),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityTypeRegistry":
        """Create registry from dictionary."""
        registry = cls()
        for type_data in data.get("types", []):
            type_def = EntityTypeDefinition.from_dict(type_data)
            registry.register(type_def)
        return registry

    def clear(self) -> None:
        """Clear all registered types."""
        self._types.clear()
        self._alias_map.clear()
        self._children_map.clear()


def create_entity_type(
    name: str,
    description: str,
    examples: list[str] | None = None,
    aliases: list[str] | None = None,
    parent_type: str | None = None,
    abstract: bool = False,
    properties: list[str] | None = None,
    **kwargs,
) -> EntityTypeDefinition:
    """
    Factory function to create an entity type definition.

    Args:
        name: Type name
        description: Description of the type
        examples: Example entities
        aliases: Alternative names
        parent_type: Parent type for inheritance
        abstract: Whether type can be instantiated
        properties: Properties specific to this type
        **kwargs: Additional fields

    Returns:
        EntityTypeDefinition instance
    """
    return EntityTypeDefinition(
        name=name.upper(),
        display_name=name.title().replace("_", " "),
        description=description,
        parent_type=parent_type.upper() if parent_type else None,
        abstract=abstract,
        examples=examples or [],
        aliases=[a.upper() for a in (aliases or [])],
        properties=properties or [],
        **kwargs,
    )


# =============================================================================
# Pre-defined Entity Types (Base)
# =============================================================================

BASE_ENTITY_TYPES = [
    # ==========================================================================
    # Root Types (Base Classes)
    # ==========================================================================
    create_entity_type(
        name="PERSON",
        description="People, characters, historical figures, individuals",
        examples=["Elon Musk", "Marie Curie", "John Smith"],
        aliases=["PER", "HUMAN", "INDIVIDUAL"],
        properties=["name", "birth_date", "nationality", "occupation"],
        keywords=["mr", "mrs", "dr", "prof", "ceo", "founder"],
    ),
    create_entity_type(
        name="ORGANIZATION",
        description="Companies, institutions, agencies, teams, groups",
        examples=["Apple Inc.", "United Nations", "MIT"],
        aliases=["ORG", "COMPANY", "INSTITUTION", "AGENCY"],
        properties=["name", "founded_date", "headquarters", "industry"],
        keywords=["inc", "corp", "ltd", "llc", "foundation", "institute"],
    ),
    create_entity_type(
        name="LOCATION",
        description="Places, cities, countries, regions, addresses, geographical features",
        examples=["New York City", "Japan", "Mount Everest"],
        aliases=["LOC", "GPE", "PLACE", "GEO", "FACILITY"],
        properties=["name", "coordinates", "country", "type"],
        keywords=["city", "country", "state", "region", "street"],
    ),
    create_entity_type(
        name="DATE",
        description="Dates, time periods, years, seasons, timestamps",
        examples=["2024", "January 15th", "Q3 2023", "last week"],
        aliases=["TIME", "DATETIME", "PERIOD", "TEMPORAL"],
        properties=["value", "precision", "timezone"],
    ),
    create_entity_type(
        name="EVENT",
        description="Events, incidents, occasions, meetings, conferences",
        examples=["World War II", "CES 2024", "Annual Meeting"],
        aliases=["INCIDENT", "OCCASION", "MEETING"],
        properties=["name", "date", "location", "participants"],
    ),
    create_entity_type(
        name="CONCEPT",
        description="Abstract concepts, theories, methodologies, ideas, fields of study",
        examples=["Machine Learning", "Quantum Theory", "Agile Development"],
        aliases=["IDEA", "THEORY", "METHODOLOGY", "FIELD"],
        properties=["name", "domain", "related_concepts"],
    ),
    create_entity_type(
        name="PRODUCT",
        description="Products, services, brands, software applications",
        examples=["iPhone 15", "Windows 11", "ChatGPT"],
        aliases=["SERVICE", "BRAND", "APP", "SOFTWARE"],
        properties=["name", "manufacturer", "release_date", "version"],
    ),
    create_entity_type(
        name="TECHNOLOGY",
        description="Technologies, tools, frameworks, programming languages, protocols",
        examples=["Python", "Kubernetes", "REST API", "TCP/IP"],
        aliases=["TECH", "TOOL", "FRAMEWORK", "LANGUAGE", "PROTOCOL"],
        properties=["name", "category", "version", "license"],
    ),
    create_entity_type(
        name="METRIC",
        description="Numbers, statistics, measurements, KPIs, quantities",
        examples=["$1.5 billion", "95% accuracy", "10,000 users"],
        aliases=["NUMBER", "STATISTIC", "QUANTITY", "KPI", "MEASURE"],
        properties=["value", "unit", "context", "timestamp"],
    ),
    create_entity_type(
        name="DOCUMENT",
        description="Documents, reports, articles, papers, publications, files",
        examples=["Annual Report 2023", "RFC 2616", "GDPR"],
        aliases=["REPORT", "ARTICLE", "PAPER", "PUBLICATION", "FILE"],
        properties=["title", "author", "date", "type"],
    ),
]


# =============================================================================
# Extended Entity Types (With Inheritance)
# =============================================================================

EXTENDED_ENTITY_TYPES = [
    # --------------------------------------------------------------------------
    # Person Subtypes
    # --------------------------------------------------------------------------
    create_entity_type(
        name="EMPLOYEE",
        description="Employees, workers, staff members of organizations",
        examples=["John Smith (Software Engineer)", "Jane Doe (Marketing Manager)"],
        parent_type="PERSON",
        properties=["employee_id", "department", "title", "hire_date"],
    ),
    create_entity_type(
        name="EXECUTIVE",
        description="C-level executives, directors, board members",
        examples=["Tim Cook (CEO)", "Satya Nadella (CEO)"],
        parent_type="EMPLOYEE",
        properties=["executive_level", "board_positions"],
    ),
    create_entity_type(
        name="RESEARCHER",
        description="Scientists, academics, research professionals",
        examples=["Dr. Fei-Fei Li", "Geoffrey Hinton"],
        parent_type="PERSON",
        properties=["institution", "research_areas", "h_index"],
    ),

    # --------------------------------------------------------------------------
    # Organization Subtypes
    # --------------------------------------------------------------------------
    create_entity_type(
        name="CORPORATION",
        description="For-profit companies, businesses, enterprises",
        examples=["Microsoft Corporation", "Toyota Motor Corp"],
        parent_type="ORGANIZATION",
        properties=["stock_symbol", "market_cap", "ceo"],
    ),
    create_entity_type(
        name="NONPROFIT",
        description="Non-profit organizations, NGOs, foundations",
        examples=["Red Cross", "Wikipedia Foundation"],
        parent_type="ORGANIZATION",
        properties=["mission", "tax_status"],
    ),
    create_entity_type(
        name="GOVERNMENT_AGENCY",
        description="Government departments, agencies, bureaus",
        examples=["FBI", "NASA", "EPA"],
        parent_type="ORGANIZATION",
        aliases=["GOV_AGENCY", "AGENCY"],
        properties=["jurisdiction", "parent_department"],
    ),
    create_entity_type(
        name="UNIVERSITY",
        description="Universities, colleges, academic institutions",
        examples=["MIT", "Stanford University", "Oxford"],
        parent_type="ORGANIZATION",
        properties=["ranking", "enrollment", "endowment"],
    ),

    # --------------------------------------------------------------------------
    # Location Subtypes
    # --------------------------------------------------------------------------
    create_entity_type(
        name="CITY",
        description="Cities, towns, municipalities",
        examples=["San Francisco", "Tokyo", "London"],
        parent_type="LOCATION",
        properties=["population", "mayor", "area"],
    ),
    create_entity_type(
        name="COUNTRY",
        description="Countries, nations, sovereign states",
        examples=["United States", "Japan", "Germany"],
        parent_type="LOCATION",
        properties=["capital", "population", "gdp", "government_type"],
    ),
    create_entity_type(
        name="FACILITY",
        description="Buildings, facilities, physical structures",
        examples=["Googleplex", "Pentagon", "CERN"],
        parent_type="LOCATION",
        properties=["address", "capacity", "purpose"],
    ),

    # --------------------------------------------------------------------------
    # Event Subtypes
    # --------------------------------------------------------------------------
    create_entity_type(
        name="CONFERENCE",
        description="Conferences, summits, symposiums",
        examples=["NeurIPS 2024", "CES", "Davos Forum"],
        parent_type="EVENT",
        properties=["organizer", "topics", "attendance"],
    ),
    create_entity_type(
        name="PRODUCT_LAUNCH",
        description="Product launches, releases, announcements",
        examples=["iPhone 15 Launch", "AWS re:Invent keynote"],
        parent_type="EVENT",
        properties=["product", "presenter"],
    ),

    # --------------------------------------------------------------------------
    # Technology Subtypes
    # --------------------------------------------------------------------------
    create_entity_type(
        name="PROGRAMMING_LANGUAGE",
        description="Programming languages, scripting languages",
        examples=["Python", "JavaScript", "Rust"],
        parent_type="TECHNOLOGY",
        aliases=["LANG", "PL"],
        properties=["paradigm", "typing", "creator"],
    ),
    create_entity_type(
        name="FRAMEWORK",
        description="Software frameworks, libraries, SDKs",
        examples=["React", "TensorFlow", "Spring Boot"],
        parent_type="TECHNOLOGY",
        properties=["language", "use_case"],
    ),
    create_entity_type(
        name="DATABASE",
        description="Database systems, data stores",
        examples=["PostgreSQL", "MongoDB", "Redis"],
        parent_type="TECHNOLOGY",
        aliases=["DB", "DATASTORE"],
        properties=["type", "query_language"],
    ),
]


def create_base_entity_registry() -> EntityTypeRegistry:
    """Create a registry with base entity types only (no inheritance)."""
    registry = EntityTypeRegistry()
    registry.register_many(BASE_ENTITY_TYPES)
    return registry


def create_extended_entity_registry() -> EntityTypeRegistry:
    """
    Create a registry with base types and extended types with inheritance.

    This includes the full type hierarchy:
    - PERSON → EMPLOYEE → EXECUTIVE
    - PERSON → RESEARCHER
    - ORGANIZATION → CORPORATION, NONPROFIT, GOVERNMENT_AGENCY, UNIVERSITY
    - LOCATION → CITY, COUNTRY, FACILITY
    - EVENT → CONFERENCE, PRODUCT_LAUNCH
    - TECHNOLOGY → PROGRAMMING_LANGUAGE, FRAMEWORK, DATABASE
    """
    registry = EntityTypeRegistry()
    # Register base types first (parents must exist before children)
    registry.register_many(BASE_ENTITY_TYPES)
    # Then register extended types with inheritance
    registry.register_many(EXTENDED_ENTITY_TYPES)

    # Validate hierarchy
    errors = registry.validate_hierarchy()
    if errors:
        logger.warning("Entity type hierarchy validation errors", errors=errors)

    return registry

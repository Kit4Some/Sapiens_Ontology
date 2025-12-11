"""
Dynamic Relation Predicate Schema.

Defines configurable relation predicates for extraction:
- Custom predicate definitions with descriptions
- Source/target type constraints
- Cardinality constraints (1:1, 1:N, N:1, N:M)
- Symmetry and inverse relationships
- Domain-specific predicate sets
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Cardinality(str, Enum):
    """
    Relationship cardinality constraints.

    Defines how many instances can participate on each side of a relationship.
    """

    ONE_TO_ONE = "1:1"      # Each source has exactly one target, and vice versa
    ONE_TO_MANY = "1:N"     # Each source can have many targets, but each target has one source
    MANY_TO_ONE = "N:1"     # Each source has one target, but each target can have many sources
    MANY_TO_MANY = "N:M"    # No restrictions on either side

    @classmethod
    def from_string(cls, value: str) -> "Cardinality":
        """Parse cardinality from string representation."""
        mapping = {
            "1:1": cls.ONE_TO_ONE,
            "1:n": cls.ONE_TO_MANY,
            "1:N": cls.ONE_TO_MANY,
            "n:1": cls.MANY_TO_ONE,
            "N:1": cls.MANY_TO_ONE,
            "n:m": cls.MANY_TO_MANY,
            "N:M": cls.MANY_TO_MANY,
            "n:n": cls.MANY_TO_MANY,
            "N:N": cls.MANY_TO_MANY,
        }
        return mapping.get(value, cls.MANY_TO_MANY)

    def allows_multiple_sources(self) -> bool:
        """Check if multiple sources can point to the same target."""
        return self in [Cardinality.MANY_TO_ONE, Cardinality.MANY_TO_MANY]

    def allows_multiple_targets(self) -> bool:
        """Check if a source can point to multiple targets."""
        return self in [Cardinality.ONE_TO_MANY, Cardinality.MANY_TO_MANY]


@dataclass
class PredicateDefinition:
    """
    Definition of a relation predicate for extraction.

    Attributes:
        name: Predicate name in UPPER_SNAKE_CASE (e.g., "WORKS_FOR")
        display_name: Human-readable name (e.g., "Works For")
        description: Description of what this relation represents
        examples: Example usage of this predicate
        source_types: Valid source entity types (empty = any)
        target_types: Valid target entity types (empty = any)
        cardinality: Relationship cardinality (1:1, 1:N, N:1, N:M)
        max_source_count: Maximum sources per target (None = unlimited)
        max_target_count: Maximum targets per source (None = unlimited)
        inverse: Inverse predicate name (e.g., WORKS_FOR <-> EMPLOYS)
        symmetric: Whether relation is symmetric (A-B implies B-A)
        transitive: Whether relation is transitive (A-B, B-C implies A-C)
        reflexive: Whether relation can be reflexive (A-A allowed)
        aliases: Alternative predicate names
        priority: Priority for disambiguation
    """

    name: str
    display_name: str
    description: str
    examples: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)

    # Cardinality constraints
    cardinality: Cardinality = Cardinality.MANY_TO_MANY
    max_source_count: int | None = None  # Max sources pointing to one target
    max_target_count: int | None = None  # Max targets from one source

    # Relationship properties
    inverse: str | None = None
    symmetric: bool = False
    transitive: bool = False  # A→B, B→C implies A→C
    reflexive: bool = False   # A→A allowed

    aliases: list[str] = field(default_factory=list)
    priority: int = 50
    enabled: bool = True

    # Semantic hints
    keywords: list[str] = field(default_factory=list)
    category: str = "general"  # employment, ownership, location, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "examples": self.examples,
            "source_types": self.source_types,
            "target_types": self.target_types,
            "cardinality": self.cardinality.value,
            "max_source_count": self.max_source_count,
            "max_target_count": self.max_target_count,
            "inverse": self.inverse,
            "symmetric": self.symmetric,
            "transitive": self.transitive,
            "reflexive": self.reflexive,
            "aliases": self.aliases,
            "priority": self.priority,
            "category": self.category,
        }

    def to_prompt_format(self, include_examples: bool = True) -> str:
        """Format for inclusion in extraction prompts."""
        result = f"- {self.name}: {self.description}"
        if include_examples and self.examples:
            result += f" (e.g., {self.examples[0]})"
        return result

    def validates_types(self, source_type: str, target_type: str) -> bool:
        """Check if source/target types are valid for this predicate."""
        source_valid = not self.source_types or source_type.upper() in [t.upper() for t in self.source_types]
        target_valid = not self.target_types or target_type.upper() in [t.upper() for t in self.target_types]
        return source_valid and target_valid

    def validates_cardinality(
        self,
        source_id: str,
        target_id: str,
        existing_relations: list[tuple[str, str]],
    ) -> tuple[bool, str | None]:
        """
        Validate cardinality constraints against existing relations.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            existing_relations: List of (source_id, target_id) tuples

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check reflexive constraint
        if source_id == target_id and not self.reflexive:
            return False, f"Predicate {self.name} does not allow reflexive relations (A→A)"

        # Count existing relations
        sources_to_target = sum(1 for s, t in existing_relations if t == target_id)
        targets_from_source = sum(1 for s, t in existing_relations if s == source_id)

        # Check cardinality constraints
        if not self.cardinality.allows_multiple_sources() and sources_to_target >= 1:
            return False, f"Predicate {self.name} ({self.cardinality.value}) allows only one source per target"

        if not self.cardinality.allows_multiple_targets() and targets_from_source >= 1:
            return False, f"Predicate {self.name} ({self.cardinality.value}) allows only one target per source"

        # Check explicit count limits
        if self.max_source_count and sources_to_target >= self.max_source_count:
            return False, f"Predicate {self.name} allows max {self.max_source_count} sources per target"

        if self.max_target_count and targets_from_source >= self.max_target_count:
            return False, f"Predicate {self.name} allows max {self.max_target_count} targets per source"

        return True, None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredicateDefinition":
        # Parse cardinality
        cardinality_str = data.get("cardinality", "N:M")
        if isinstance(cardinality_str, str):
            cardinality = Cardinality.from_string(cardinality_str)
        else:
            cardinality = cardinality_str

        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"].replace("_", " ").title()),
            description=data.get("description", ""),
            examples=data.get("examples", []),
            source_types=data.get("source_types", []),
            target_types=data.get("target_types", []),
            cardinality=cardinality,
            max_source_count=data.get("max_source_count"),
            max_target_count=data.get("max_target_count"),
            inverse=data.get("inverse"),
            symmetric=data.get("symmetric", False),
            transitive=data.get("transitive", False),
            reflexive=data.get("reflexive", False),
            aliases=data.get("aliases", []),
            priority=data.get("priority", 50),
            enabled=data.get("enabled", True),
            keywords=data.get("keywords", []),
            category=data.get("category", "general"),
        )


class PredicateRegistry:
    """
    Registry for managing predicate definitions.

    Provides:
    - Registration of custom predicates
    - Predicate lookup by name or alias
    - Type constraint validation
    - Prompt generation helpers
    """

    def __init__(self):
        self._predicates: dict[str, PredicateDefinition] = {}
        self._alias_map: dict[str, str] = {}  # alias -> predicate_name
        self._categories: dict[str, list[str]] = {}  # category -> predicate_names

    def register(self, predicate: PredicateDefinition) -> None:
        """Register a predicate definition."""
        name_upper = predicate.name.upper().replace(" ", "_")
        self._predicates[name_upper] = predicate

        # Register aliases
        for alias in predicate.aliases:
            self._alias_map[alias.upper().replace(" ", "_")] = name_upper

        # Register by category
        if predicate.category not in self._categories:
            self._categories[predicate.category] = []
        if name_upper not in self._categories[predicate.category]:
            self._categories[predicate.category].append(name_upper)

        logger.debug(
            "Registered predicate",
            name=name_upper,
            category=predicate.category,
        )

    def register_many(self, predicates: list[PredicateDefinition]) -> None:
        """Register multiple predicates."""
        for predicate in predicates:
            self.register(predicate)

    def get(self, name: str) -> PredicateDefinition | None:
        """Get predicate by name or alias."""
        name_upper = name.upper().replace(" ", "_").replace("-", "_")

        # Direct lookup
        if name_upper in self._predicates:
            return self._predicates[name_upper]

        # Alias lookup
        if name_upper in self._alias_map:
            return self._predicates[self._alias_map[name_upper]]

        return None

    def resolve_name(self, name: str) -> str:
        """Resolve predicate name or alias to canonical name."""
        name_upper = name.upper().replace(" ", "_").replace("-", "_")

        if name_upper in self._predicates:
            return name_upper

        if name_upper in self._alias_map:
            return self._alias_map[name_upper]

        return name_upper

    def get_all(self, enabled_only: bool = True) -> list[PredicateDefinition]:
        """Get all registered predicates."""
        predicates = list(self._predicates.values())
        if enabled_only:
            predicates = [p for p in predicates if p.enabled]
        return sorted(predicates, key=lambda p: (-p.priority, p.name))

    def get_by_category(self, category: str) -> list[PredicateDefinition]:
        """Get predicates by category."""
        names = self._categories.get(category, [])
        return [self._predicates[name] for name in names if name in self._predicates]

    def get_categories(self) -> list[str]:
        """Get all predicate categories."""
        return list(self._categories.keys())

    def validate_relation(
        self,
        predicate: str,
        source_type: str,
        target_type: str,
    ) -> tuple[bool, str | None]:
        """
        Validate a relation against predicate type constraints.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pred_def = self.get(predicate)
        if not pred_def:
            return True, None  # Unknown predicates are allowed

        if not pred_def.validates_types(source_type, target_type):
            return False, (
                f"Predicate {predicate} does not allow "
                f"{source_type} -> {target_type}"
            )

        return True, None

    def validate_cardinality(
        self,
        predicate: str,
        source_id: str,
        target_id: str,
        existing_relations: list[tuple[str, str]],
    ) -> tuple[bool, str | None]:
        """
        Validate a relation against predicate cardinality constraints.

        Args:
            predicate: Predicate name
            source_id: Source entity ID
            target_id: Target entity ID
            existing_relations: List of existing (source_id, target_id) tuples

        Returns:
            Tuple of (is_valid, error_message)
        """
        pred_def = self.get(predicate)
        if not pred_def:
            return True, None  # Unknown predicates have no constraints

        return pred_def.validates_cardinality(source_id, target_id, existing_relations)

    def validate_full(
        self,
        predicate: str,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        existing_relations: list[tuple[str, str]] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Full validation of a relation against all constraints.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Type validation
        type_valid, type_error = self.validate_relation(predicate, source_type, target_type)
        if not type_valid and type_error:
            errors.append(type_error)

        # Cardinality validation
        if existing_relations is not None:
            card_valid, card_error = self.validate_cardinality(
                predicate, source_id, target_id, existing_relations
            )
            if not card_valid and card_error:
                errors.append(card_error)

        return len(errors) == 0, errors

    def get_inverse(self, predicate: str) -> str | None:
        """Get the inverse predicate if defined."""
        pred_def = self.get(predicate)
        if pred_def and pred_def.inverse:
            return pred_def.inverse
        return None

    def is_symmetric(self, predicate: str) -> bool:
        """Check if predicate is symmetric."""
        pred_def = self.get(predicate)
        return pred_def.symmetric if pred_def else False

    def generate_prompt_section(
        self,
        include_examples: bool = True,
        categories: list[str] | None = None,
    ) -> str:
        """Generate predicates section for prompts."""
        lines = ["## Common Predicate Types"]

        if categories:
            # Group by specified categories
            for category in categories:
                predicates = self.get_by_category(category)
                if predicates:
                    lines.append(f"\n### {category.title()}")
                    for pred in predicates:
                        if pred.enabled:
                            lines.append(pred.to_prompt_format(include_examples))
        else:
            # All predicates, grouped by category
            for category in sorted(self._categories.keys()):
                predicates = self.get_by_category(category)
                enabled = [p for p in predicates if p.enabled]
                if enabled:
                    lines.append(f"\n### {category.title()}")
                    for pred in enabled:
                        lines.append(pred.to_prompt_format(include_examples))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export registry to dictionary."""
        return {
            "predicates": [p.to_dict() for p in self._predicates.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredicateRegistry":
        """Create registry from dictionary."""
        registry = cls()
        for pred_data in data.get("predicates", []):
            pred_def = PredicateDefinition.from_dict(pred_data)
            registry.register(pred_def)
        return registry

    def clear(self) -> None:
        """Clear all registered predicates."""
        self._predicates.clear()
        self._alias_map.clear()
        self._categories.clear()


def create_predicate(
    name: str,
    description: str,
    examples: list[str] | None = None,
    source_types: list[str] | None = None,
    target_types: list[str] | None = None,
    cardinality: str | Cardinality = Cardinality.MANY_TO_MANY,
    category: str = "general",
    **kwargs,
) -> PredicateDefinition:
    """
    Factory function to create a predicate definition.

    Args:
        name: Predicate name
        description: Description of the predicate
        examples: Example usages
        source_types: Allowed source entity types
        target_types: Allowed target entity types
        cardinality: Relationship cardinality ("1:1", "1:N", "N:1", "N:M")
        category: Category for grouping
        **kwargs: Additional fields (inverse, symmetric, transitive, reflexive, etc.)

    Returns:
        PredicateDefinition instance
    """
    # Parse cardinality if string
    if isinstance(cardinality, str):
        cardinality = Cardinality.from_string(cardinality)

    return PredicateDefinition(
        name=name.upper().replace(" ", "_"),
        display_name=name.replace("_", " ").title(),
        description=description,
        examples=examples or [],
        source_types=[t.upper() for t in (source_types or [])],
        target_types=[t.upper() for t in (target_types or [])],
        cardinality=cardinality,
        category=category,
        **kwargs,
    )


# =============================================================================
# Pre-defined Predicates (Base)
# =============================================================================

BASE_PREDICATES = [
    # ==========================================================================
    # Employment Relations
    # ==========================================================================
    create_predicate(
        name="WORKS_FOR",
        description="Employment relationship - person works for organization",
        examples=["John WORKS_FOR Google"],
        source_types=["PERSON"],
        target_types=["ORGANIZATION"],
        cardinality="N:M",  # Person can work for multiple orgs, orgs have many employees
        inverse="EMPLOYS",
        category="employment",
        aliases=["EMPLOYED_BY", "WORKS_AT"],
    ),
    create_predicate(
        name="EMPLOYS",
        description="Organization employs person",
        examples=["Google EMPLOYS John"],
        source_types=["ORGANIZATION"],
        target_types=["PERSON"],
        cardinality="1:N",  # One org can employ many people
        inverse="WORKS_FOR",
        category="employment",
        aliases=["HIRES"],
    ),
    create_predicate(
        name="FOUNDED",
        description="Person or organization founded another entity",
        examples=["Elon Musk FOUNDED SpaceX"],
        source_types=["PERSON", "ORGANIZATION"],
        target_types=["ORGANIZATION", "PRODUCT"],
        cardinality="N:M",  # Multiple founders possible, founder can found multiple entities
        category="creation",
        aliases=["CREATED", "ESTABLISHED", "STARTED"],
    ),
    create_predicate(
        name="CEO_OF",
        description="Person is CEO of organization",
        examples=["Tim Cook CEO_OF Apple"],
        source_types=["PERSON"],
        target_types=["ORGANIZATION"],
        cardinality="1:1",  # One CEO per org at a time, one org per CEO role
        max_target_count=1,  # A person can only be CEO of one company typically
        category="employment",
        aliases=["LEADS", "HEADS"],
    ),

    # ==========================================================================
    # Location Relations
    # ==========================================================================
    create_predicate(
        name="LOCATED_IN",
        description="Entity is located in a place",
        examples=["Apple LOCATED_IN California"],
        target_types=["LOCATION"],
        cardinality="N:1",  # Many entities in one location, entity has one primary location
        category="location",
        aliases=["BASED_IN", "HEADQUARTERED_IN", "IN"],
    ),
    create_predicate(
        name="BORN_IN",
        description="Person was born in a location",
        examples=["Einstein BORN_IN Germany"],
        source_types=["PERSON"],
        target_types=["LOCATION"],
        cardinality="N:1",  # Many people born in one place, but person born in one place
        max_target_count=1,  # Person can only be born in one location
        category="location",
    ),

    # ==========================================================================
    # Ownership Relations
    # ==========================================================================
    create_predicate(
        name="OWNS",
        description="Ownership relationship",
        examples=["Microsoft OWNS GitHub"],
        cardinality="1:N",  # One owner, owner can own multiple things
        category="ownership",
        aliases=["POSSESSES", "HAS"],
    ),
    create_predicate(
        name="ACQUIRED",
        description="Entity acquired another entity",
        examples=["Microsoft ACQUIRED LinkedIn"],
        source_types=["ORGANIZATION"],
        target_types=["ORGANIZATION", "PRODUCT"],
        cardinality="N:1",  # Multiple acquirers possible (consortium), but typically one
        max_source_count=1,  # Usually one acquirer
        category="ownership",
        aliases=["BOUGHT", "PURCHASED"],
    ),
    create_predicate(
        name="SUBSIDIARY_OF",
        description="Organization is subsidiary of parent",
        examples=["Instagram SUBSIDIARY_OF Meta"],
        source_types=["ORGANIZATION"],
        target_types=["ORGANIZATION"],
        cardinality="N:1",  # Many subsidiaries per parent, one parent per subsidiary
        max_target_count=1,  # Subsidiary has one direct parent
        inverse="PARENT_OF",
        transitive=True,  # If A is subsidiary of B, and B of C, then A is subsidiary of C
        category="ownership",
    ),

    # ==========================================================================
    # Membership/Composition Relations
    # ==========================================================================
    create_predicate(
        name="PART_OF",
        description="Entity is part of another entity",
        examples=["GPU PART_OF Computer"],
        cardinality="N:M",  # Part can be in multiple wholes, whole has multiple parts
        transitive=True,  # If A part of B, B part of C, then A part of C
        category="membership",
        aliases=["BELONGS_TO", "MEMBER_OF", "COMPONENT_OF"],
    ),
    create_predicate(
        name="CONTAINS",
        description="Entity contains another entity",
        examples=["Computer CONTAINS GPU"],
        cardinality="1:N",  # Container has many parts
        inverse="PART_OF",
        transitive=True,
        category="membership",
        aliases=["INCLUDES", "HAS_PART"],
    ),

    # ==========================================================================
    # Association Relations
    # ==========================================================================
    create_predicate(
        name="RELATED_TO",
        description="Generic association between entities",
        examples=["AI RELATED_TO Machine Learning"],
        cardinality="N:M",  # No restrictions
        symmetric=True,
        reflexive=True,  # A can be related to itself
        category="association",
        aliases=["ASSOCIATED_WITH", "CONNECTED_TO"],
        priority=10,  # Lower priority - use more specific predicates when possible
    ),
    create_predicate(
        name="COLLABORATES_WITH",
        description="Partnership or collaboration",
        examples=["Google COLLABORATES_WITH NASA"],
        cardinality="N:M",
        symmetric=True,
        category="association",
        aliases=["PARTNERS_WITH", "WORKS_WITH"],
    ),
    create_predicate(
        name="COMPETES_WITH",
        description="Competition relationship",
        examples=["Google COMPETES_WITH Microsoft"],
        cardinality="N:M",
        symmetric=True,
        category="association",
        aliases=["RIVAL_OF"],
    ),

    # ==========================================================================
    # Hierarchy Relations
    # ==========================================================================
    create_predicate(
        name="REPORTS_TO",
        description="Reporting relationship in hierarchy",
        examples=["Manager REPORTS_TO Director"],
        source_types=["PERSON"],
        target_types=["PERSON"],
        cardinality="N:1",  # Many report to one, person reports to one manager
        max_target_count=1,  # Person has one direct manager
        inverse="MANAGES",
        transitive=True,  # Indirect reporting chain
        category="hierarchy",
    ),
    create_predicate(
        name="MANAGES",
        description="Management relationship",
        examples=["Director MANAGES Manager"],
        source_types=["PERSON"],
        target_types=["PERSON", "ORGANIZATION"],
        cardinality="1:N",  # One manager, many reports
        inverse="REPORTS_TO",
        category="hierarchy",
        aliases=["SUPERVISES", "OVERSEES"],
    ),

    # ==========================================================================
    # Production/Usage Relations
    # ==========================================================================
    create_predicate(
        name="PRODUCES",
        description="Entity produces something",
        examples=["Apple PRODUCES iPhone"],
        source_types=["ORGANIZATION", "PERSON"],
        target_types=["PRODUCT", "TECHNOLOGY"],
        cardinality="1:N",  # One producer (primary), producer makes many products
        category="production",
        aliases=["MANUFACTURES", "CREATES", "DEVELOPS"],
    ),
    create_predicate(
        name="USES",
        description="Entity uses something",
        examples=["Netflix USES AWS"],
        cardinality="N:M",  # Many users, user uses many things
        category="production",
        aliases=["UTILIZES", "EMPLOYS_TECH"],
    ),
    create_predicate(
        name="BUILT_WITH",
        description="Something is built with technology",
        examples=["App BUILT_WITH Python"],
        target_types=["TECHNOLOGY"],
        cardinality="N:M",  # Multiple techs per product, tech used in many products
        category="production",
        aliases=["DEVELOPED_WITH", "IMPLEMENTED_IN"],
    ),

    # ==========================================================================
    # Temporal Relations
    # ==========================================================================
    create_predicate(
        name="OCCURRED_ON",
        description="Event occurred on a date",
        examples=["Launch OCCURRED_ON 2024-01-15"],
        source_types=["EVENT"],
        target_types=["DATE"],
        cardinality="N:1",  # Many events on same date, but event has one occurrence date
        max_target_count=1,  # Event occurs on one specific date
        category="temporal",
        aliases=["HAPPENED_ON", "TOOK_PLACE_ON"],
    ),
    create_predicate(
        name="FOUNDED_ON",
        description="Entity was founded on a date",
        examples=["Apple FOUNDED_ON 1976"],
        source_types=["ORGANIZATION"],
        target_types=["DATE"],
        cardinality="N:1",  # Many orgs founded on same date
        max_target_count=1,  # Org has one founding date
        category="temporal",
    ),

    # ==========================================================================
    # Knowledge/Education Relations
    # ==========================================================================
    create_predicate(
        name="STUDIED_AT",
        description="Person studied at institution",
        examples=["Einstein STUDIED_AT ETH Zurich"],
        source_types=["PERSON"],
        target_types=["ORGANIZATION"],
        cardinality="N:M",  # Many students at institution, person can study at multiple
        category="education",
        aliases=["ATTENDED", "GRADUATED_FROM"],
    ),
    create_predicate(
        name="AUTHORED",
        description="Person authored document",
        examples=["Einstein AUTHORED Relativity Paper"],
        source_types=["PERSON"],
        target_types=["DOCUMENT"],
        cardinality="N:M",  # Multiple authors possible, author writes multiple docs
        category="creation",
        aliases=["WROTE", "PUBLISHED"],
    ),
]


def create_base_predicate_registry() -> PredicateRegistry:
    """Create a registry with base predicates."""
    registry = PredicateRegistry()
    registry.register_many(BASE_PREDICATES)
    return registry

"""
Ontology Export Module.

Provides export capabilities for the ontology schema in various formats:
- JSON-LD (Linked Data)
- RDF/Turtle
- Simple JSON

Enables interoperability with external systems and tools.
"""

from datetime import datetime
from typing import Any

import structlog

from src.sddi.extractors.schema.entity_schema import (
    EntityTypeRegistry,
    create_extended_entity_registry,
)
from src.sddi.extractors.schema.relation_schema import (
    PredicateRegistry,
    create_base_predicate_registry,
)

logger = structlog.get_logger(__name__)


# JSON-LD Context for the ontology
JSONLD_CONTEXT = {
    "@vocab": "http://schema.ontology.local/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    # Entity type mappings
    "EntityType": "owl:Class",
    "Predicate": "owl:ObjectProperty",
    # Common properties
    "name": "rdfs:label",
    "description": "rdfs:comment",
    "parent_type": "rdfs:subClassOf",
    "examples": "skos:example",
    "aliases": "skos:altLabel",
    # Predicate properties
    "domain": "rdfs:domain",
    "range": "rdfs:range",
    "inverse": "owl:inverseOf",
    "symmetric": "owl:SymmetricProperty",
    "transitive": "owl:TransitiveProperty",
    # Cardinality
    "cardinality": {"@id": "owl:cardinality", "@type": "xsd:string"},
    "maxSourceCount": {"@id": "owl:maxCardinality", "@type": "xsd:integer"},
    "maxTargetCount": {"@id": "owl:maxCardinality", "@type": "xsd:integer"},
}


class OntologyExporter:
    """
    Exports ontology schema to various formats.

    Supports:
    - JSON-LD for Linked Data compatibility
    - Simple JSON for internal use
    - RDF/Turtle for semantic web tools
    """

    def __init__(
        self,
        entity_registry: EntityTypeRegistry | None = None,
        predicate_registry: PredicateRegistry | None = None,
        base_uri: str = "http://ontology.local/",
    ):
        """
        Initialize exporter with registries.

        Args:
            entity_registry: Entity type registry (defaults to extended)
            predicate_registry: Predicate registry (defaults to base)
            base_uri: Base URI for the ontology
        """
        self.entity_registry = entity_registry or create_extended_entity_registry()
        self.predicate_registry = predicate_registry or create_base_predicate_registry()
        self.base_uri = base_uri.rstrip("/") + "/"

    def export_jsonld(
        self,
        include_hierarchy: bool = True,
        include_examples: bool = True,
    ) -> dict[str, Any]:
        """
        Export ontology as JSON-LD.

        Args:
            include_hierarchy: Include type hierarchy information
            include_examples: Include example values

        Returns:
            JSON-LD document as dictionary
        """
        doc = {
            "@context": JSONLD_CONTEXT,
            "@id": f"{self.base_uri}ontology",
            "@type": "owl:Ontology",
            "name": "Knowledge Graph Ontology",
            "description": "Entity types and predicates for the knowledge graph",
            "version": "1.0.0",
            "created": datetime.utcnow().isoformat() + "Z",
            "entityTypes": self._export_entity_types(include_hierarchy, include_examples),
            "predicates": self._export_predicates(include_examples),
        }

        if include_hierarchy:
            doc["typeHierarchy"] = self.entity_registry.get_type_hierarchy()

        return doc

    def _export_entity_types(
        self,
        include_hierarchy: bool,
        include_examples: bool,
    ) -> list[dict[str, Any]]:
        """Export entity types as JSON-LD nodes."""
        types = []

        for type_def in self.entity_registry.get_all(enabled_only=False):
            type_node = {
                "@id": f"{self.base_uri}types/{type_def.name}",
                "@type": "EntityType",
                "name": type_def.name,
                "displayName": type_def.display_name,
                "description": type_def.description,
                "abstract": type_def.abstract,
                "properties": type_def.properties,
            }

            if include_hierarchy and type_def.parent_type:
                type_node["parent_type"] = {
                    "@id": f"{self.base_uri}types/{type_def.parent_type}"
                }

            if type_def.aliases:
                type_node["aliases"] = type_def.aliases

            if include_examples and type_def.examples:
                type_node["examples"] = type_def.examples

            if type_def.keywords:
                type_node["keywords"] = type_def.keywords

            types.append(type_node)

        return types

    def _export_predicates(self, include_examples: bool) -> list[dict[str, Any]]:
        """Export predicates as JSON-LD nodes."""
        predicates = []

        for pred_def in self.predicate_registry.get_all(enabled_only=False):
            pred_node = {
                "@id": f"{self.base_uri}predicates/{pred_def.name}",
                "@type": ["Predicate"],
                "name": pred_def.name,
                "displayName": pred_def.display_name,
                "description": pred_def.description,
                "category": pred_def.category,
                "cardinality": pred_def.cardinality.value,
            }

            # Add property characteristics
            if pred_def.symmetric:
                pred_node["@type"].append("owl:SymmetricProperty")
            if pred_def.transitive:
                pred_node["@type"].append("owl:TransitiveProperty")
            if pred_def.reflexive:
                pred_node["reflexive"] = True

            # Domain (source types)
            if pred_def.source_types:
                pred_node["domain"] = [
                    {"@id": f"{self.base_uri}types/{t}"} for t in pred_def.source_types
                ]

            # Range (target types)
            if pred_def.target_types:
                pred_node["range"] = [
                    {"@id": f"{self.base_uri}types/{t}"} for t in pred_def.target_types
                ]

            # Inverse
            if pred_def.inverse:
                pred_node["inverse"] = {
                    "@id": f"{self.base_uri}predicates/{pred_def.inverse}"
                }

            # Cardinality constraints
            if pred_def.max_source_count:
                pred_node["maxSourceCount"] = pred_def.max_source_count
            if pred_def.max_target_count:
                pred_node["maxTargetCount"] = pred_def.max_target_count

            if pred_def.aliases:
                pred_node["aliases"] = pred_def.aliases

            if include_examples and pred_def.examples:
                pred_node["examples"] = pred_def.examples

            predicates.append(pred_node)

        return predicates

    def export_json(self) -> dict[str, Any]:
        """
        Export ontology as simple JSON (no JSON-LD semantics).

        Returns:
            Plain JSON dictionary
        """
        return {
            "version": "1.0.0",
            "created": datetime.utcnow().isoformat() + "Z",
            "entityTypes": self.entity_registry.to_dict(),
            "predicates": self.predicate_registry.to_dict(),
            "hierarchy": self.entity_registry.get_type_hierarchy(),
        }

    def export_turtle(self) -> str:
        """
        Export ontology as RDF/Turtle format.

        Returns:
            Turtle-formatted string
        """
        lines = [
            "# Ontology Export - RDF/Turtle Format",
            f"# Generated: {datetime.utcnow().isoformat()}Z",
            "",
            "@prefix : <{base}> .".format(base=self.base_uri),
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
            "",
            "# =============================================================================",
            "# Entity Types",
            "# =============================================================================",
            "",
        ]

        # Export entity types
        for type_def in self.entity_registry.get_all(enabled_only=False):
            lines.append(f":types/{type_def.name} a owl:Class ;")
            lines.append(f'    rdfs:label "{type_def.name}" ;')
            lines.append(f'    rdfs:comment "{self._escape_turtle(type_def.description)}" ;')

            if type_def.parent_type:
                lines.append(f"    rdfs:subClassOf :types/{type_def.parent_type} ;")

            if type_def.aliases:
                for alias in type_def.aliases:
                    lines.append(f'    skos:altLabel "{alias}" ;')

            # Remove trailing semicolon and add period
            lines[-1] = lines[-1].rstrip(" ;") + " ."
            lines.append("")

        lines.extend([
            "# =============================================================================",
            "# Predicates",
            "# =============================================================================",
            "",
        ])

        # Export predicates
        for pred_def in self.predicate_registry.get_all(enabled_only=False):
            pred_types = ["owl:ObjectProperty"]
            if pred_def.symmetric:
                pred_types.append("owl:SymmetricProperty")
            if pred_def.transitive:
                pred_types.append("owl:TransitiveProperty")

            lines.append(f":predicates/{pred_def.name} a {', '.join(pred_types)} ;")
            lines.append(f'    rdfs:label "{pred_def.name}" ;')
            lines.append(f'    rdfs:comment "{self._escape_turtle(pred_def.description)}" ;')

            if pred_def.source_types:
                for source_type in pred_def.source_types:
                    lines.append(f"    rdfs:domain :types/{source_type} ;")

            if pred_def.target_types:
                for target_type in pred_def.target_types:
                    lines.append(f"    rdfs:range :types/{target_type} ;")

            if pred_def.inverse:
                lines.append(f"    owl:inverseOf :predicates/{pred_def.inverse} ;")

            lines[-1] = lines[-1].rstrip(" ;") + " ."
            lines.append("")

        return "\n".join(lines)

    def _escape_turtle(self, text: str) -> str:
        """Escape special characters for Turtle format."""
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def validate_ontology(self) -> dict[str, Any]:
        """
        Validate the ontology for consistency issues.

        Returns:
            Validation report with errors and warnings
        """
        errors = []
        warnings = []

        # Validate entity type hierarchy
        hierarchy_errors = self.entity_registry.validate_hierarchy()
        errors.extend([{"type": "hierarchy", "message": e} for e in hierarchy_errors])

        # Check predicate domain/range references
        all_types = set(self.entity_registry.get_type_names(enabled_only=False))

        for pred_def in self.predicate_registry.get_all(enabled_only=False):
            for source_type in pred_def.source_types:
                if source_type not in all_types:
                    warnings.append({
                        "type": "predicate_domain",
                        "predicate": pred_def.name,
                        "message": f"Source type '{source_type}' not in entity registry",
                    })

            for target_type in pred_def.target_types:
                if target_type not in all_types:
                    warnings.append({
                        "type": "predicate_range",
                        "predicate": pred_def.name,
                        "message": f"Target type '{target_type}' not in entity registry",
                    })

            # Check inverse consistency
            if pred_def.inverse:
                inverse_def = self.predicate_registry.get(pred_def.inverse)
                if not inverse_def:
                    warnings.append({
                        "type": "inverse",
                        "predicate": pred_def.name,
                        "message": f"Inverse predicate '{pred_def.inverse}' not found",
                    })
                elif inverse_def.inverse != pred_def.name:
                    warnings.append({
                        "type": "inverse_mismatch",
                        "predicate": pred_def.name,
                        "message": f"Inverse predicate '{pred_def.inverse}' does not reference back",
                    })

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "entity_type_count": len(all_types),
            "predicate_count": len(self.predicate_registry.get_all(enabled_only=False)),
        }


def export_ontology_jsonld(
    entity_registry: EntityTypeRegistry | None = None,
    predicate_registry: PredicateRegistry | None = None,
    base_uri: str = "http://ontology.local/",
) -> dict[str, Any]:
    """
    Convenience function to export ontology as JSON-LD.

    Args:
        entity_registry: Optional entity registry
        predicate_registry: Optional predicate registry
        base_uri: Base URI for the ontology

    Returns:
        JSON-LD document
    """
    exporter = OntologyExporter(entity_registry, predicate_registry, base_uri)
    return exporter.export_jsonld()


def export_ontology_turtle(
    entity_registry: EntityTypeRegistry | None = None,
    predicate_registry: PredicateRegistry | None = None,
    base_uri: str = "http://ontology.local/",
) -> str:
    """
    Convenience function to export ontology as Turtle.

    Args:
        entity_registry: Optional entity registry
        predicate_registry: Optional predicate registry
        base_uri: Base URI for the ontology

    Returns:
        Turtle-formatted string
    """
    exporter = OntologyExporter(entity_registry, predicate_registry, base_uri)
    return exporter.export_turtle()


def validate_ontology(
    entity_registry: EntityTypeRegistry | None = None,
    predicate_registry: PredicateRegistry | None = None,
) -> dict[str, Any]:
    """
    Convenience function to validate ontology consistency.

    Args:
        entity_registry: Optional entity registry
        predicate_registry: Optional predicate registry

    Returns:
        Validation report
    """
    exporter = OntologyExporter(entity_registry, predicate_registry)
    return exporter.validate_ontology()

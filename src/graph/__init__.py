"""
Knowledge Graph Module.

Neo4j-based knowledge graph operations and GDS algorithm integrations.
"""

from src.graph.community import CommunityDetector
from src.graph.neo4j_client import (
    OntologyGraphClient,
    get_ontology_client,
)
from src.graph.schema import (
    ChunkNode,
    CommunityNode,
    EntityNode,
    FulltextSearchResult,
    NodeLabel,
    Relationship,
    RelationType,
    VectorSearchResult,
)

__all__ = [
    # Schema
    "NodeLabel",
    "RelationType",
    "EntityNode",
    "ChunkNode",
    "CommunityNode",
    "Relationship",
    "VectorSearchResult",
    "FulltextSearchResult",
    # Client
    "OntologyGraphClient",
    "get_ontology_client",
    # Community Detection
    "CommunityDetector",
]

"""
Graph Schema Models.

Defines node labels, relationship types, and property schemas for the ontology graph.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeLabel(str, Enum):
    """Node labels in the ontology graph."""

    ENTITY = "Entity"
    CHUNK = "Chunk"
    COMMUNITY = "Community"


class RelationType(str, Enum):
    """Relationship types in the ontology graph."""

    RELATES_TO = "RELATES_TO"  # Entity-Entity semantic relationship
    CONTAINS = "CONTAINS"      # Document/Chunk contains Entity
    MENTIONS = "MENTIONS"      # Chunk mentions Entity (with position info)
    BELONGS_TO = "BELONGS_TO"  # Entity belongs to Community
    PARENT = "PARENT"          # Community hierarchy (child -> parent)
    DERIVED_FROM = "DERIVED_FROM"  # Entity derived from another Entity
    SUPPORTS = "SUPPORTS"      # Evidence supports Answer


class EntityNode(BaseModel):
    """Entity node schema."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (e.g., Person, Organization, Concept)")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class ChunkNode(BaseModel):
    """Chunk node schema for text segments."""

    id: str = Field(..., description="Unique identifier")
    text: str = Field(..., description="Text content")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    source: str = Field(..., description="Source document identifier")
    position: int = Field(..., description="Position in source document")


class CommunityNode(BaseModel):
    """Community node schema for hierarchical clustering."""

    id: str = Field(..., description="Unique identifier")
    level: int = Field(..., description="Hierarchy level (0 = leaf)")
    summary: str = Field(default="", description="Community summary")
    member_count: int = Field(default=0, description="Number of members")


class Relationship(BaseModel):
    """Generic relationship schema."""

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    type: RelationType = Field(..., description="Relationship type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    weight: float = Field(default=1.0, description="Relationship weight/strength")


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""

    node_id: str
    node_label: str
    score: float
    properties: dict[str, Any] = Field(default_factory=dict)


class FulltextSearchResult(BaseModel):
    """Result from full-text search."""

    node_id: str
    node_label: str
    score: float
    text: str
    highlights: list[str] = Field(default_factory=list)

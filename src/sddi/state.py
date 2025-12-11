"""
SDDI Pipeline State and Models.

Defines the state schema and data models for the Schema-Driven Data Integration pipeline.

Includes:
- Timestamp tracking (created_at, updated_at)
- Soft-delete support (is_deleted, deleted_at)
- Version tracking for optimistic locking
"""

from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Supported entity types for extraction."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"
    METRIC = "METRIC"
    DOCUMENT = "DOCUMENT"
    OTHER = "OTHER"


class LoadStatus(str, Enum):
    """Status of data loading operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RawDocument(BaseModel):
    """Raw input document."""

    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Raw text content")
    source: str = Field(default="unknown", description="Document source")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class TextChunk(BaseModel):
    """Text chunk after splitting."""

    id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text content")
    doc_id: str = Field(..., description="Parent document ID")
    position: int = Field(..., description="Position in document")
    start_char: int = Field(default=0, description="Start character index")
    end_char: int = Field(default=0, description="End character index")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class ExtractedEntity(BaseModel):
    """
    Entity extracted from text.

    Includes lifecycle management fields:
    - Timestamps: created_at, updated_at
    - Soft-delete: is_deleted, deleted_at, deleted_by
    - Versioning: version
    - Provenance: source_doc_id, pipeline_id
    """

    # Core fields
    id: str = Field(..., description="Entity identifier")
    name: str = Field(..., description="Entity name/mention")
    type: EntityType = Field(..., description="Entity type")
    description: str = Field(default="", description="Entity description")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    chunk_ids: list[str] = Field(default_factory=list, description="Source chunk IDs")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")

    # Timestamp fields
    created_at: datetime | None = Field(default=None, description="When entity was created")
    updated_at: datetime | None = Field(default=None, description="When entity was last updated")

    # Soft-delete fields
    is_deleted: bool = Field(default=False, description="Whether entity is soft-deleted")
    deleted_at: datetime | None = Field(default=None, description="When entity was deleted")
    deleted_by: str | None = Field(default=None, description="Who deleted the entity")
    deletion_reason: str | None = Field(default=None, description="Why entity was deleted")

    # Versioning
    version: int = Field(default=1, ge=1, description="Version number for optimistic locking")

    # Provenance
    source_doc_id: str | None = Field(default=None, description="Source document ID")
    pipeline_id: str | None = Field(default=None, description="Extraction pipeline ID")

    def soft_delete(self, deleted_by: str | None = None, reason: str | None = None) -> None:
        """Mark entity as soft-deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        self.deleted_by = deleted_by
        self.deletion_reason = reason

    def restore(self) -> None:
        """Restore a soft-deleted entity."""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
        self.deletion_reason = None
        self.updated_at = datetime.utcnow()

    def touch(self) -> None:
        """Update the updated_at timestamp and increment version."""
        self.updated_at = datetime.utcnow()
        self.version += 1


class ExtractedRelation(BaseModel):
    """
    Relation extracted between entities.

    Includes lifecycle management fields for consistency with ExtractedEntity.
    """

    # Core fields
    id: str = Field(..., description="Relation identifier")
    source_entity: str = Field(..., description="Source entity ID or name")
    target_entity: str = Field(..., description="Target entity ID or name")
    predicate: str = Field(..., description="Relationship type/predicate")
    description: str = Field(default="", description="Relation description")
    chunk_ids: list[str] = Field(default_factory=list, description="Source chunk IDs")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")

    # Timestamp fields
    created_at: datetime | None = Field(default=None, description="When relation was created")
    updated_at: datetime | None = Field(default=None, description="When relation was last updated")

    # Soft-delete fields
    is_deleted: bool = Field(default=False, description="Whether relation is soft-deleted")
    deleted_at: datetime | None = Field(default=None, description="When relation was deleted")

    # Provenance
    pipeline_id: str | None = Field(default=None, description="Extraction pipeline ID")


class Triplet(BaseModel):
    """Knowledge graph triplet (subject, predicate, object)."""

    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Predicate/relationship")
    object: str = Field(..., description="Object entity")
    subject_type: EntityType = Field(default=EntityType.OTHER, description="Subject entity type")
    object_type: EntityType = Field(default=EntityType.OTHER, description="Object entity type")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: str | None = Field(default=None, description="Source chunk")


class EmbeddingResult(BaseModel):
    """Embedding result for a text item."""

    id: str = Field(..., description="Item identifier")
    item_type: str = Field(..., description="Type: 'chunk' or 'entity'")
    embedding: list[float] = Field(..., description="Vector embedding")
    model: str = Field(default="", description="Embedding model used")


class LoadResult(BaseModel):
    """Result of loading data to Neo4j."""

    chunks_created: int = Field(default=0)
    entities_created: int = Field(default=0)
    relations_created: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0)


class SDDIPipelineState(TypedDict, total=False):
    """
    State for the SDDI LangGraph pipeline.

    Flows through: Ingest → Chunk → Extract Entities → Extract Relations → Embed → Load
    """

    # Input
    raw_data: list[RawDocument]

    # After chunking
    chunks: list[TextChunk]

    # After entity extraction
    entities: list[ExtractedEntity]

    # After relation extraction
    relations: list[ExtractedRelation]

    # Normalized triplets
    triplets: list[Triplet]

    # After embedding
    embeddings: dict[str, list[float]]  # id -> embedding

    # Loading status
    load_status: LoadStatus
    load_result: LoadResult

    # Pipeline metadata
    pipeline_id: str
    current_step: str
    errors: list[str]
    metadata: dict[str, Any]

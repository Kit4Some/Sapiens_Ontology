"""
SDDI Embedders Module.

Provides versioned embedding support:
- Version tracking for embeddings
- Model migration support
- Compatibility checking
"""

from src.sddi.embedders.versioned_embedder import (
    EmbeddingModel,
    EmbeddingVersion,
    VersionedEmbedding,
    VersionedEmbedder,
    EmbeddingVersionStore,
    EmbeddingMigrator,
    create_versioned_embedder,
    # Pre-defined versions
    OPENAI_ADA_002_VERSION,
    OPENAI_3_SMALL_VERSION,
    OPENAI_3_LARGE_VERSION,
)

__all__ = [
    "EmbeddingModel",
    "EmbeddingVersion",
    "VersionedEmbedding",
    "VersionedEmbedder",
    "EmbeddingVersionStore",
    "EmbeddingMigrator",
    "create_versioned_embedder",
    "OPENAI_ADA_002_VERSION",
    "OPENAI_3_SMALL_VERSION",
    "OPENAI_3_LARGE_VERSION",
]

"""
SDDI Chunkers Module.

Provides intelligent document chunking:
- Structure-aware chunking (respects sections, code blocks, tables)
- Semantic boundary preservation
- Section hierarchy context
"""

from src.sddi.chunkers.structure_aware_chunker import (
    ChunkType,
    HeadingLevel,
    StructuralElement,
    StructuredChunk,
    ChunkerConfig,
    DocumentStructureParser,
    StructureAwareChunker,
    create_chunker,
)

__all__ = [
    "ChunkType",
    "HeadingLevel",
    "StructuralElement",
    "StructuredChunk",
    "ChunkerConfig",
    "DocumentStructureParser",
    "StructureAwareChunker",
    "create_chunker",
]

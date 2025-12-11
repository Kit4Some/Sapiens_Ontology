"""
Structure-Aware Document Chunking.

Intelligently splits documents by recognizing structure:
- Markdown headers (# ## ### etc.)
- HTML sections (<section>, <article>, <h1-h6>)
- Code blocks (```, indented blocks)
- Tables (markdown, HTML)
- Lists (ordered, unordered)
- Paragraphs and sentences

Preserves semantic boundaries to avoid splitting mid-concept.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ChunkType(str, Enum):
    """Types of content chunks."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST = "list"
    QUOTE = "quote"
    SECTION = "section"
    METADATA = "metadata"
    MIXED = "mixed"


class HeadingLevel(int, Enum):
    """Document heading levels."""

    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


@dataclass
class StructuralElement:
    """A structural element from the document."""

    element_type: ChunkType
    content: str
    start_pos: int
    end_pos: int
    level: int = 0  # For headings
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.content)


@dataclass
class StructuredChunk:
    """A chunk with structural context."""

    text: str
    chunk_type: ChunkType
    chunk_index: int

    # Structural context
    section_title: str | None = None
    section_level: int = 0
    parent_sections: list[str] = field(default_factory=list)

    # Position info
    start_char: int = 0
    end_char: int = 0

    # Content metadata
    has_code: bool = False
    has_table: bool = False
    has_list: bool = False
    language: str | None = None  # For code blocks

    # Quality indicators
    is_complete: bool = True  # Not split mid-sentence
    overlap_with_prev: int = 0
    overlap_with_next: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chunk_type": self.chunk_type.value,
            "chunk_index": self.chunk_index,
            "section_title": self.section_title,
            "section_level": self.section_level,
            "parent_sections": self.parent_sections,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "has_code": self.has_code,
            "has_table": self.has_table,
            "has_list": self.has_list,
            "language": self.language,
            "is_complete": self.is_complete,
        }


@dataclass
class ChunkerConfig:
    """Configuration for structure-aware chunker."""

    # Size limits
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    target_chunk_size: int = 1000

    # Overlap settings
    overlap_size: int = 100
    overlap_on_boundaries: bool = False  # Don't overlap at section boundaries

    # Structure preservation
    preserve_code_blocks: bool = True  # Don't split code blocks
    preserve_tables: bool = True        # Don't split tables
    preserve_lists: bool = True         # Try to keep lists together

    # Splitting behavior
    split_on_sentence: bool = True      # Prefer sentence boundaries
    split_on_paragraph: bool = True     # Prefer paragraph boundaries

    # Section handling
    include_heading_in_chunk: bool = True  # Include section heading in each chunk
    max_heading_context: int = 2            # Max parent headings to include


class DocumentStructureParser:
    """
    Parses document structure into elements.

    Recognizes:
    - Markdown structure
    - HTML structure
    - Plain text paragraphs
    """

    # Markdown patterns
    MD_HEADING = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    MD_CODE_BLOCK = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    MD_INLINE_CODE = re.compile(r'`[^`]+`')
    MD_TABLE = re.compile(r'^\|.+\|$\n^\|[-:| ]+\|$\n(^\|.+\|$\n?)+', re.MULTILINE)
    MD_LIST_ITEM = re.compile(r'^(\s*)[-*+]\s+.+$|^(\s*)\d+\.\s+.+$', re.MULTILINE)
    MD_QUOTE = re.compile(r'^>\s+.+$(\n^>\s+.+$)*', re.MULTILINE)
    MD_HR = re.compile(r'^---+$|^\*\*\*+$|^___+$', re.MULTILINE)

    # HTML patterns
    HTML_HEADING = re.compile(r'<h([1-6])>(.+?)</h\1>', re.IGNORECASE | re.DOTALL)
    HTML_CODE = re.compile(r'<code>(.+?)</code>|<pre>(.+?)</pre>', re.IGNORECASE | re.DOTALL)
    HTML_TABLE = re.compile(r'<table>(.+?)</table>', re.IGNORECASE | re.DOTALL)
    HTML_LIST = re.compile(r'<[uo]l>(.+?)</[uo]l>', re.IGNORECASE | re.DOTALL)

    def __init__(self):
        self._elements: list[StructuralElement] = []

    def parse(self, text: str) -> list[StructuralElement]:
        """
        Parse document into structural elements.

        Args:
            text: Document text

        Returns:
            List of StructuralElement objects
        """
        self._elements = []

        # Detect format
        is_markdown = self._is_markdown(text)
        is_html = self._is_html(text)

        if is_markdown:
            self._parse_markdown(text)
        elif is_html:
            self._parse_html(text)
        else:
            self._parse_plain_text(text)

        # Sort by position
        self._elements.sort(key=lambda e: e.start_pos)

        # Fill gaps with paragraph elements
        self._fill_gaps(text)

        return self._elements

    def _is_markdown(self, text: str) -> bool:
        """Check if text appears to be markdown."""
        indicators = [
            self.MD_HEADING.search(text),
            self.MD_CODE_BLOCK.search(text),
            self.MD_TABLE.search(text),
            '```' in text,
            bool(re.search(r'\[.+\]\(.+\)', text)),  # Links
        ]
        return sum(bool(i) for i in indicators) >= 2

    def _is_html(self, text: str) -> bool:
        """Check if text appears to be HTML."""
        return bool(re.search(r'<[a-z]+[^>]*>', text, re.IGNORECASE))

    def _parse_markdown(self, text: str) -> None:
        """Parse markdown structure."""

        # Code blocks (parse first to avoid conflicts)
        for match in self.MD_CODE_BLOCK.finditer(text):
            lang = match.group(1) or None
            self._elements.append(StructuralElement(
                element_type=ChunkType.CODE_BLOCK,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                metadata={"language": lang},
            ))

        # Headings
        for match in self.MD_HEADING.finditer(text):
            level = len(match.group(1))
            self._elements.append(StructuralElement(
                element_type=ChunkType.HEADING,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                level=level,
                metadata={"title": match.group(2).strip()},
            ))

        # Tables
        for match in self.MD_TABLE.finditer(text):
            self._elements.append(StructuralElement(
                element_type=ChunkType.TABLE,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
            ))

        # Block quotes
        for match in self.MD_QUOTE.finditer(text):
            self._elements.append(StructuralElement(
                element_type=ChunkType.QUOTE,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
            ))

    def _parse_html(self, text: str) -> None:
        """Parse HTML structure."""

        # Headings
        for match in self.HTML_HEADING.finditer(text):
            level = int(match.group(1))
            self._elements.append(StructuralElement(
                element_type=ChunkType.HEADING,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                level=level,
                metadata={"title": self._strip_html(match.group(2))},
            ))

        # Code blocks
        for match in self.HTML_CODE.finditer(text):
            self._elements.append(StructuralElement(
                element_type=ChunkType.CODE_BLOCK,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
            ))

        # Tables
        for match in self.HTML_TABLE.finditer(text):
            self._elements.append(StructuralElement(
                element_type=ChunkType.TABLE,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
            ))

        # Lists
        for match in self.HTML_LIST.finditer(text):
            self._elements.append(StructuralElement(
                element_type=ChunkType.LIST,
                content=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
            ))

    def _parse_plain_text(self, text: str) -> None:
        """Parse plain text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        pos = 0

        for para in paragraphs:
            if para.strip():
                start = text.find(para, pos)
                self._elements.append(StructuralElement(
                    element_type=ChunkType.PARAGRAPH,
                    content=para,
                    start_pos=start,
                    end_pos=start + len(para),
                ))
                pos = start + len(para)

    def _fill_gaps(self, text: str) -> None:
        """Fill gaps between elements with paragraph elements."""
        if not self._elements:
            # Whole document is one paragraph
            self._elements.append(StructuralElement(
                element_type=ChunkType.PARAGRAPH,
                content=text,
                start_pos=0,
                end_pos=len(text),
            ))
            return

        new_elements = []
        prev_end = 0

        for elem in self._elements:
            # Check for gap
            if elem.start_pos > prev_end:
                gap_text = text[prev_end:elem.start_pos]
                if gap_text.strip():
                    new_elements.append(StructuralElement(
                        element_type=ChunkType.PARAGRAPH,
                        content=gap_text,
                        start_pos=prev_end,
                        end_pos=elem.start_pos,
                    ))

            new_elements.append(elem)
            prev_end = elem.end_pos

        # Check for trailing content
        if prev_end < len(text):
            trailing = text[prev_end:]
            if trailing.strip():
                new_elements.append(StructuralElement(
                    element_type=ChunkType.PARAGRAPH,
                    content=trailing,
                    start_pos=prev_end,
                    end_pos=len(text),
                ))

        self._elements = new_elements

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text).strip()


class StructureAwareChunker:
    """
    Structure-aware document chunker.

    Splits documents while preserving semantic structure:
    - Respects section boundaries
    - Keeps code blocks together
    - Preserves tables
    - Maintains heading context
    """

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()
        self._parser = DocumentStructureParser()

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[StructuredChunk]:
        """
        Split document into structured chunks.

        Args:
            text: Document text
            metadata: Optional document metadata

        Returns:
            List of StructuredChunk objects
        """
        if not text or not text.strip():
            return []

        # Parse structure
        elements = self._parser.parse(text)

        # Build section hierarchy
        section_stack = self._build_section_stack(elements)

        # Create chunks
        chunks = self._create_chunks(text, elements, section_stack)

        logger.info(
            "Document chunked",
            total_chunks=len(chunks),
            original_length=len(text),
        )

        return chunks

    def _build_section_stack(
        self, elements: list[StructuralElement]
    ) -> dict[int, list[str]]:
        """
        Build section hierarchy from heading elements.

        Returns dict mapping position to list of parent section titles.
        """
        section_stack: dict[int, list[str]] = {}
        current_sections: list[tuple[int, str]] = []  # (level, title)

        for elem in elements:
            if elem.element_type == ChunkType.HEADING:
                level = elem.level
                title = elem.metadata.get("title", "")

                # Pop sections at same or higher level
                while current_sections and current_sections[-1][0] >= level:
                    current_sections.pop()

                current_sections.append((level, title))

            # Record current hierarchy for this position
            section_stack[elem.start_pos] = [s[1] for s in current_sections]

        return section_stack

    def _create_chunks(
        self,
        text: str,
        elements: list[StructuralElement],
        section_stack: dict[int, list[str]],
    ) -> list[StructuredChunk]:
        """Create chunks from structural elements."""

        chunks: list[StructuredChunk] = []
        current_chunk_parts: list[StructuralElement] = []
        current_size = 0
        chunk_index = 0

        for elem in elements:
            elem_size = elem.length

            # Special handling for code blocks and tables
            if elem.element_type == ChunkType.CODE_BLOCK and self.config.preserve_code_blocks:
                # Flush current chunk first
                if current_chunk_parts:
                    chunk = self._finalize_chunk(
                        current_chunk_parts, chunk_index, section_stack
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_parts = []
                    current_size = 0

                # Code block as its own chunk (or split if too large)
                if elem_size <= self.config.max_chunk_size:
                    chunk = self._create_single_element_chunk(
                        elem, chunk_index, section_stack
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                else:
                    # Split large code block
                    sub_chunks = self._split_large_element(elem, chunk_index, section_stack)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                continue

            if elem.element_type == ChunkType.TABLE and self.config.preserve_tables:
                # Similar handling for tables
                if current_chunk_parts:
                    chunk = self._finalize_chunk(
                        current_chunk_parts, chunk_index, section_stack
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_parts = []
                    current_size = 0

                chunk = self._create_single_element_chunk(
                    elem, chunk_index, section_stack
                )
                chunks.append(chunk)
                chunk_index += 1
                continue

            # Check if adding this element exceeds max size
            if current_size + elem_size > self.config.max_chunk_size:
                # Flush current chunk
                if current_chunk_parts:
                    chunk = self._finalize_chunk(
                        current_chunk_parts, chunk_index, section_stack
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_parts = []
                    current_size = 0

                # Handle large elements
                if elem_size > self.config.max_chunk_size:
                    sub_chunks = self._split_large_element(elem, chunk_index, section_stack)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    continue

            # Add element to current chunk
            current_chunk_parts.append(elem)
            current_size += elem_size

            # Check if we should start new chunk (section boundary)
            if elem.element_type == ChunkType.HEADING:
                if current_size >= self.config.min_chunk_size:
                    chunk = self._finalize_chunk(
                        current_chunk_parts, chunk_index, section_stack
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Start new chunk with heading context
                    if self.config.include_heading_in_chunk and current_chunk_parts:
                        # Keep heading for next chunk
                        current_chunk_parts = [elem]
                        current_size = elem_size
                    else:
                        current_chunk_parts = []
                        current_size = 0

        # Flush remaining
        if current_chunk_parts:
            chunk = self._finalize_chunk(
                current_chunk_parts, chunk_index, section_stack
            )
            chunks.append(chunk)

        return chunks

    def _finalize_chunk(
        self,
        parts: list[StructuralElement],
        index: int,
        section_stack: dict[int, list[str]],
    ) -> StructuredChunk:
        """Create a StructuredChunk from parts."""

        # Combine text
        text = "\n\n".join(p.content for p in parts)

        # Determine type
        chunk_type = ChunkType.MIXED
        if len(parts) == 1:
            chunk_type = parts[0].element_type

        # Get section context
        first_pos = parts[0].start_pos
        sections = section_stack.get(first_pos, [])

        parent_sections = sections[:-1] if len(sections) > 1 else []
        section_title = sections[-1] if sections else None
        section_level = len(sections)

        # Detect content types
        has_code = any(p.element_type == ChunkType.CODE_BLOCK for p in parts)
        has_table = any(p.element_type == ChunkType.TABLE for p in parts)
        has_list = any(p.element_type == ChunkType.LIST for p in parts)

        # Get language for code blocks
        language = None
        for p in parts:
            if p.element_type == ChunkType.CODE_BLOCK and p.metadata.get("language"):
                language = p.metadata["language"]
                break

        return StructuredChunk(
            text=text.strip(),
            chunk_type=chunk_type,
            chunk_index=index,
            section_title=section_title,
            section_level=section_level,
            parent_sections=parent_sections[-self.config.max_heading_context:],
            start_char=parts[0].start_pos,
            end_char=parts[-1].end_pos,
            has_code=has_code,
            has_table=has_table,
            has_list=has_list,
            language=language,
            is_complete=True,
        )

    def _create_single_element_chunk(
        self,
        elem: StructuralElement,
        index: int,
        section_stack: dict[int, list[str]],
    ) -> StructuredChunk:
        """Create a chunk from a single element."""

        sections = section_stack.get(elem.start_pos, [])

        return StructuredChunk(
            text=elem.content.strip(),
            chunk_type=elem.element_type,
            chunk_index=index,
            section_title=sections[-1] if sections else None,
            section_level=len(sections),
            parent_sections=sections[:-1] if len(sections) > 1 else [],
            start_char=elem.start_pos,
            end_char=elem.end_pos,
            has_code=elem.element_type == ChunkType.CODE_BLOCK,
            has_table=elem.element_type == ChunkType.TABLE,
            has_list=elem.element_type == ChunkType.LIST,
            language=elem.metadata.get("language"),
            is_complete=True,
        )

    def _split_large_element(
        self,
        elem: StructuralElement,
        start_index: int,
        section_stack: dict[int, list[str]],
    ) -> list[StructuredChunk]:
        """Split a large element into multiple chunks."""

        chunks = []
        content = elem.content
        chunk_index = start_index

        # For code blocks, try to split on newlines
        if elem.element_type == ChunkType.CODE_BLOCK:
            lines = content.split('\n')
            current_lines = []
            current_size = 0

            for line in lines:
                if current_size + len(line) > self.config.target_chunk_size and current_lines:
                    chunk_text = '\n'.join(current_lines)
                    chunk = StructuredChunk(
                        text=chunk_text,
                        chunk_type=ChunkType.CODE_BLOCK,
                        chunk_index=chunk_index,
                        section_title=section_stack.get(elem.start_pos, [None])[-1] if section_stack.get(elem.start_pos) else None,
                        has_code=True,
                        language=elem.metadata.get("language"),
                        is_complete=False,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_lines = []
                    current_size = 0

                current_lines.append(line)
                current_size += len(line) + 1

            if current_lines:
                chunk_text = '\n'.join(current_lines)
                chunks.append(StructuredChunk(
                    text=chunk_text,
                    chunk_type=ChunkType.CODE_BLOCK,
                    chunk_index=chunk_index,
                    has_code=True,
                    language=elem.metadata.get("language"),
                    is_complete=len(chunks) == 0,  # Only first chunk is incomplete
                ))

        else:
            # For paragraphs, try to split on sentences
            if self.config.split_on_sentence:
                chunks = self._split_on_sentences(content, elem, start_index, section_stack)
            else:
                chunks = self._split_by_size(content, elem, start_index, section_stack)

        return chunks

    def _split_on_sentences(
        self,
        content: str,
        elem: StructuralElement,
        start_index: int,
        section_stack: dict[int, list[str]],
    ) -> list[StructuredChunk]:
        """Split content on sentence boundaries."""

        # Sentence boundary pattern
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z가-힣])')
        sentences = sentence_end.split(content)

        chunks = []
        current_text = ""
        chunk_index = start_index

        for sentence in sentences:
            if len(current_text) + len(sentence) > self.config.target_chunk_size and current_text:
                chunks.append(StructuredChunk(
                    text=current_text.strip(),
                    chunk_type=elem.element_type,
                    chunk_index=chunk_index,
                    is_complete=True,
                ))
                chunk_index += 1
                current_text = ""

            current_text += sentence + " "

        if current_text.strip():
            chunks.append(StructuredChunk(
                text=current_text.strip(),
                chunk_type=elem.element_type,
                chunk_index=chunk_index,
                is_complete=True,
            ))

        return chunks

    def _split_by_size(
        self,
        content: str,
        elem: StructuralElement,
        start_index: int,
        section_stack: dict[int, list[str]],
    ) -> list[StructuredChunk]:
        """Split content by character count."""

        chunks = []
        chunk_index = start_index
        pos = 0

        while pos < len(content):
            end = min(pos + self.config.target_chunk_size, len(content))

            # Try to find a good break point
            if end < len(content):
                # Look for space near the end
                space_pos = content.rfind(' ', pos, end)
                if space_pos > pos:
                    end = space_pos

            chunk_text = content[pos:end].strip()
            if chunk_text:
                chunks.append(StructuredChunk(
                    text=chunk_text,
                    chunk_type=elem.element_type,
                    chunk_index=chunk_index,
                    is_complete=False,
                ))
                chunk_index += 1

            pos = end

        return chunks

    def get_chunk_stats(self, chunks: list[StructuredChunk]) -> dict[str, Any]:
        """Get statistics about chunks."""

        if not chunks:
            return {"count": 0}

        sizes = [len(c.text) for c in chunks]
        types = {}
        for c in chunks:
            types[c.chunk_type.value] = types.get(c.chunk_type.value, 0) + 1

        return {
            "count": len(chunks),
            "total_chars": sum(sizes),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "types": types,
            "with_code": sum(1 for c in chunks if c.has_code),
            "with_table": sum(1 for c in chunks if c.has_table),
            "complete_chunks": sum(1 for c in chunks if c.is_complete),
        }


def create_chunker(
    max_chunk_size: int = 1500,
    target_chunk_size: int = 1000,
    overlap_size: int = 100,
    preserve_structure: bool = True,
) -> StructureAwareChunker:
    """
    Factory function to create a chunker with common settings.

    Args:
        max_chunk_size: Maximum chunk size in characters
        target_chunk_size: Target chunk size
        overlap_size: Overlap between chunks
        preserve_structure: Whether to preserve code blocks and tables

    Returns:
        Configured StructureAwareChunker
    """
    config = ChunkerConfig(
        max_chunk_size=max_chunk_size,
        target_chunk_size=target_chunk_size,
        overlap_size=overlap_size,
        preserve_code_blocks=preserve_structure,
        preserve_tables=preserve_structure,
    )

    return StructureAwareChunker(config)

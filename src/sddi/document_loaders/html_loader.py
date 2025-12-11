"""
HTML Document Loader for SDDI Pipeline.

Loads HTML files and extracts text content for knowledge graph construction.
"""

import re
import uuid
from html.parser import HTMLParser
from typing import Any, BinaryIO

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument


class HTMLTextExtractor(HTMLParser):
    """Simple HTML parser to extract text content."""

    # Tags to skip entirely (content won't be extracted)
    SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "math"}

    # Block-level tags that should add newlines
    BLOCK_TAGS = {
        "p", "div", "section", "article", "aside", "header", "footer",
        "nav", "main", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "dl", "dt", "dd",
        "table", "tr", "th", "td",
        "blockquote", "pre", "hr", "br",
        "form", "fieldset", "legend",
    }

    # Heading tags for structure extraction
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self, extract_structure: bool = True):
        super().__init__()
        self.extract_structure = extract_structure
        self.text_parts: list[str] = []
        self.current_tag: str | None = None
        self.skip_depth: int = 0
        self.headings: list[tuple[str, str]] = []  # (level, text)
        self.links: list[tuple[str, str]] = []  # (href, text)
        self.title: str = ""
        self.meta_description: str = ""
        self._current_heading: str | None = None
        self._current_link: str | None = None
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.current_tag = tag

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if self.skip_depth > 0:
            return

        # Handle block tags
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

        # Track headings
        if tag in self.HEADING_TAGS and self.extract_structure:
            self._current_heading = tag

        # Track links
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href and not href.startswith("#"):
                self._current_link = href
                self._link_text = []

        # Handle meta tags
        if tag == "meta":
            attrs_dict = dict(attrs)
            name = attrs_dict.get("name", "").lower()
            content = attrs_dict.get("content", "")
            if name == "description" and content:
                self.meta_description = content

        # Handle images (extract alt text)
        if tag == "img":
            attrs_dict = dict(attrs)
            alt = attrs_dict.get("alt", "")
            if alt:
                self.text_parts.append(f"[Image: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return

        if self.skip_depth > 0:
            return

        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

        # Save heading
        if tag == self._current_heading:
            heading_text = "".join(self.text_parts[-10:]).strip()  # Approximate
            if heading_text:
                self.headings.append((self._current_heading, heading_text))
            self._current_heading = None

        # Save link
        if tag == "a" and self._current_link:
            link_text = "".join(self._link_text).strip()
            if link_text:
                self.links.append((self._current_link, link_text))
            self._current_link = None
            self._link_text = []

        self.current_tag = None

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return

        # Clean whitespace
        text = data.strip()
        if not text:
            return

        self.text_parts.append(text)

        # Track link text
        if self._current_link is not None:
            self._link_text.append(text)

        # Track title
        if self.current_tag == "title":
            self.title = text

    def get_text(self) -> str:
        """Get the extracted text content."""
        text = " ".join(self.text_parts)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def get_structured_content(self) -> str:
        """Get structured content with headings and metadata."""
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}")

        if self.meta_description:
            parts.append(f"Description: {self.meta_description}")

        if parts:
            parts.append("")  # Blank line

        # Add main text
        parts.append(self.get_text())

        return "\n".join(parts)


class HTMLLoader(BaseDocumentLoader):
    """
    Loader for HTML files.

    Extracts text content from HTML documents, handling:
    - Text extraction with proper whitespace
    - Heading structure
    - Link extraction
    - Meta information (title, description)

    Args:
        extract_structure: Whether to extract document structure (headings, etc.)
        include_links: Whether to include extracted links in metadata
        split_by_heading: Whether to split into documents by heading
        min_heading_level: Minimum heading level for splitting (1-6)
    """

    SUPPORTED_EXTENSIONS = ["html", "htm", "xhtml"]

    def __init__(
        self,
        extract_structure: bool = True,
        include_links: bool = True,
        split_by_heading: bool = False,
        min_heading_level: int = 2,
    ) -> None:
        self.extract_structure = extract_structure
        self.include_links = include_links
        self.split_by_heading = split_by_heading
        self.min_heading_level = min_heading_level

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Load HTML file into RawDocument(s)."""
        content = file.read()
        if isinstance(content, bytes):
            # Try to detect encoding from meta tag, fallback to utf-8
            content = self._decode_html(content)

        # Parse HTML
        parser = HTMLTextExtractor(extract_structure=self.extract_structure)
        try:
            parser.feed(content)
        except Exception:
            # Fallback: just extract text with regex
            text = self._simple_extract(content)
            return [
                RawDocument(
                    id=f"{filename}:0:{uuid.uuid4().hex[:8]}",
                    content=text,
                    source=filename,
                    metadata={"source": filename, "format": "html", **(metadata or {})},
                )
            ]

        base_metadata = {
            "source": filename,
            "format": "html",
            "title": parser.title,
            "description": parser.meta_description,
            **(metadata or {}),
        }

        if self.include_links:
            base_metadata["links"] = parser.links[:50]  # Limit links

        documents: list[RawDocument] = []

        if self.split_by_heading and parser.headings:
            documents = self._split_by_headings(
                content, parser, filename, base_metadata
            )
        else:
            # Single document
            text = parser.get_structured_content()
            if text.strip():
                documents.append(
                    RawDocument(
                        id=f"{filename}:0:{uuid.uuid4().hex[:8]}",
                        content=text,
                        source=filename,
                        metadata={
                            **base_metadata,
                            "headings": [h[1] for h in parser.headings],
                        },
                    )
                )

        return documents

    def _decode_html(self, content: bytes) -> str:
        """Decode HTML content, detecting encoding if possible."""
        # Try common encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = content.decode(encoding)
                # Check for meta charset
                charset_match = re.search(
                    r'<meta[^>]+charset=["\']?([^"\'\s>]+)', text, re.I
                )
                if charset_match:
                    detected = charset_match.group(1).lower()
                    if detected != encoding:
                        return content.decode(detected)
                return text
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback
        return content.decode("utf-8", errors="replace")

    def _simple_extract(self, content: str) -> str:
        """Simple text extraction using regex (fallback)."""
        # Remove script and style
        text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.S | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Clean entities
        text = re.sub(r"&[^;]+;", " ", text)
        # Clean whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_by_headings(
        self,
        content: str,
        parser: HTMLTextExtractor,
        filename: str,
        base_metadata: dict[str, Any],
    ) -> list[RawDocument]:
        """Split document by heading tags."""
        documents: list[RawDocument] = []

        # Find heading positions in original HTML
        heading_pattern = re.compile(
            r"<(h[1-6])[^>]*>(.*?)</\1>",
            re.I | re.S,
        )

        sections: list[tuple[str, str, int, int]] = []
        for match in heading_pattern.finditer(content):
            level = int(match.group(1)[1])
            if level <= self.min_heading_level:
                heading_text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                sections.append((
                    match.group(1),
                    heading_text,
                    match.start(),
                    match.end(),
                ))

        if not sections:
            # No suitable headings, return single document
            return [
                RawDocument(
                    id=f"{filename}:0:{uuid.uuid4().hex[:8]}",
                    content=parser.get_structured_content(),
                    source=filename,
                    metadata=base_metadata,
                )
            ]

        # Extract content between headings
        for idx, (tag, heading, start, end) in enumerate(sections):
            # Get content until next heading or end
            if idx < len(sections) - 1:
                section_end = sections[idx + 1][2]
            else:
                section_end = len(content)

            section_html = content[start:section_end]

            # Parse section
            section_parser = HTMLTextExtractor(extract_structure=False)
            try:
                section_parser.feed(section_html)
                section_text = section_parser.get_text()
            except Exception:
                section_text = self._simple_extract(section_html)

            if section_text.strip():
                documents.append(
                    RawDocument(
                        id=f"{filename}:{idx}:{uuid.uuid4().hex[:8]}",
                        content=f"{heading}\n\n{section_text}",
                        source=filename,
                        metadata={
                            **base_metadata,
                            "section_heading": heading,
                            "section_level": int(tag[1]),
                            "section_index": idx,
                        },
                    )
                )

        return documents

"""
Markdown Document Loader.
"""

import re
from pathlib import Path
from typing import BinaryIO

import structlog

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument

logger = structlog.get_logger(__name__)


class MarkdownLoader(BaseDocumentLoader):
    """
    Loader for Markdown files (.md, .markdown).

    Can either load as a single document or split by headings.
    """

    SUPPORTED_EXTENSIONS = ["md", "markdown"]

    def __init__(
        self,
        encoding: str = "utf-8",
        split_by_headings: bool = False,
        heading_level: int = 2,
        remove_code_blocks: bool = False,
    ):
        """
        Initialize the markdown loader.

        Args:
            encoding: Text encoding to use
            split_by_headings: If True, split document by headings
            heading_level: Heading level to split on (1-6)
            remove_code_blocks: If True, remove code blocks from content
        """
        self.encoding = encoding
        self.split_by_headings = split_by_headings
        self.heading_level = heading_level
        self.remove_code_blocks = remove_code_blocks

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load a markdown file."""
        logger.info("Loading markdown file", filename=filename)

        try:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode(self.encoding)

            content = content.strip()

            if not content:
                logger.warning("Empty markdown file", filename=filename)
                return []

            # Optionally remove code blocks
            if self.remove_code_blocks:
                content = self._remove_code_blocks(content)

            # Split by headings or return as single document
            if self.split_by_headings:
                return self._split_by_headings(content, filename, metadata)

            doc = RawDocument(
                id=self._generate_doc_id(filename),
                content=content,
                source=filename,
                metadata={
                    "file_type": "markdown",
                    "encoding": self.encoding,
                    "char_count": len(content),
                    "headings": self._extract_headings(content),
                    **(metadata or {}),
                },
            )

            logger.info(
                "Markdown file loaded",
                filename=filename,
                char_count=len(content),
            )

            return [doc]

        except UnicodeDecodeError as e:
            logger.error("Encoding error", filename=filename, error=str(e))
            raise ValueError(f"Failed to decode {filename}: {e}")

    async def load_from_path(
        self,
        path: Path,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load a markdown file from path."""
        with open(path, "rb") as f:
            return await self.load(f, path.name, metadata)

    def _remove_code_blocks(self, content: str) -> str:
        """Remove fenced code blocks from markdown."""
        # Remove fenced code blocks (```...```)
        content = re.sub(r"```[\s\S]*?```", "", content)
        # Remove inline code (`...`)
        content = re.sub(r"`[^`]+`", "", content)
        return content

    def _extract_headings(self, content: str) -> list[str]:
        """Extract all headings from markdown content."""
        pattern = r"^(#{1,6})\s+(.+)$"
        headings = []
        for match in re.finditer(pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append(f"h{level}: {text}")
        return headings

    def _split_by_headings(
        self,
        content: str,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Split markdown content by headings."""
        pattern = rf"^(#{{{self.heading_level}}})\s+(.+)$"

        sections = []
        current_section = {"title": "Introduction", "content": ""}
        current_start = 0

        for match in re.finditer(pattern, content, re.MULTILINE):
            # Save previous section
            section_content = content[current_start : match.start()].strip()
            if section_content:
                current_section["content"] = section_content
                sections.append(current_section.copy())

            # Start new section
            current_section = {
                "title": match.group(2).strip(),
                "content": "",
            }
            current_start = match.end()

        # Add final section
        final_content = content[current_start:].strip()
        if final_content:
            current_section["content"] = final_content
            sections.append(current_section)

        # Convert to RawDocuments
        documents = []
        for idx, section in enumerate(sections):
            if not section["content"]:
                continue

            doc = RawDocument(
                id=self._generate_doc_id(filename, idx),
                content=section["content"],
                source=filename,
                metadata={
                    "file_type": "markdown",
                    "section_title": section["title"],
                    "section_index": idx,
                    **(metadata or {}),
                },
            )
            documents.append(doc)

        logger.info(
            "Markdown file split",
            filename=filename,
            section_count=len(documents),
        )

        return documents

"""
Plain Text Document Loader.
"""

from pathlib import Path
from typing import BinaryIO

import structlog

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument

logger = structlog.get_logger(__name__)


class TextLoader(BaseDocumentLoader):
    """
    Loader for plain text files (.txt).

    Simply reads the file content as a single document.
    """

    SUPPORTED_EXTENSIONS = ["txt", "text"]

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the text loader.

        Args:
            encoding: Text encoding to use (default: utf-8)
        """
        self.encoding = encoding

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load a text file."""
        logger.info("Loading text file", filename=filename)

        try:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode(self.encoding)

            # Clean up content
            content = content.strip()

            if not content:
                logger.warning("Empty text file", filename=filename)
                return []

            doc = RawDocument(
                id=self._generate_doc_id(filename),
                content=content,
                source=filename,
                metadata={
                    "file_type": "text",
                    "encoding": self.encoding,
                    "char_count": len(content),
                    **(metadata or {}),
                },
            )

            logger.info(
                "Text file loaded",
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
        """Load a text file from path."""
        with open(path, "rb") as f:
            return await self.load(f, path.name, metadata)

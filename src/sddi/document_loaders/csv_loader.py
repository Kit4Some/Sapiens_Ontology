"""
CSV Document Loader.
"""

import csv
import io
from pathlib import Path
from typing import BinaryIO

import structlog

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument

logger = structlog.get_logger(__name__)


class CSVLoader(BaseDocumentLoader):
    """
    Loader for CSV files (.csv).

    Can create one document per row or combine all rows into one document.
    """

    SUPPORTED_EXTENSIONS = ["csv"]

    def __init__(
        self,
        encoding: str = "utf-8",
        content_columns: list[str] | None = None,
        id_column: str | None = None,
        one_doc_per_row: bool = True,
        delimiter: str = ",",
        row_template: str | None = None,
    ):
        """
        Initialize the CSV loader.

        Args:
            encoding: Text encoding to use
            content_columns: Columns to use for document content (None = all)
            id_column: Column to use for document IDs (optional)
            one_doc_per_row: If True, create one document per row
            delimiter: CSV delimiter character
            row_template: Template for formatting rows (e.g., "{name}: {description}")
        """
        self.encoding = encoding
        self.content_columns = content_columns
        self.id_column = id_column
        self.one_doc_per_row = one_doc_per_row
        self.delimiter = delimiter
        self.row_template = row_template

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load a CSV file."""
        logger.info("Loading CSV file", filename=filename)

        try:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode(self.encoding)

            if not content.strip():
                logger.warning("Empty CSV file", filename=filename)
                return []

            # Parse CSV
            reader = csv.DictReader(
                io.StringIO(content),
                delimiter=self.delimiter,
            )

            rows = list(reader)
            if not rows:
                logger.warning("CSV file has no data rows", filename=filename)
                return []

            headers = reader.fieldnames or []

            logger.info(
                "CSV parsed",
                filename=filename,
                row_count=len(rows),
                columns=headers,
            )

            if self.one_doc_per_row:
                return self._load_one_per_row(rows, headers, filename, metadata)
            else:
                return self._load_combined(rows, headers, filename, metadata)

        except csv.Error as e:
            logger.error("CSV parsing error", filename=filename, error=str(e))
            raise ValueError(f"Failed to parse CSV {filename}: {e}")
        except UnicodeDecodeError as e:
            logger.error("Encoding error", filename=filename, error=str(e))
            raise ValueError(f"Failed to decode {filename}: {e}")

    async def load_from_path(
        self,
        path: Path,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load a CSV file from path."""
        with open(path, "rb") as f:
            return await self.load(f, path.name, metadata)

    def _format_row(self, row: dict, headers: list[str]) -> str:
        """Format a row as text content."""
        if self.row_template:
            try:
                return self.row_template.format(**row)
            except KeyError:
                pass

        # Use content columns or all columns
        columns = self.content_columns or headers

        parts = []
        for col in columns:
            if col in row and row[col]:
                parts.append(f"{col}: {row[col]}")

        return "\n".join(parts)

    def _load_one_per_row(
        self,
        rows: list[dict],
        headers: list[str],
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Create one document per CSV row."""
        documents = []

        for idx, row in enumerate(rows):
            # Generate document ID
            if self.id_column and self.id_column in row:
                doc_id = str(row[self.id_column])
            else:
                doc_id = self._generate_doc_id(filename, idx)

            # Format content
            content = self._format_row(row, headers)
            if not content.strip():
                continue

            doc = RawDocument(
                id=doc_id,
                content=content,
                source=filename,
                metadata={
                    "file_type": "csv",
                    "row_index": idx,
                    "columns": headers,
                    "row_data": row,
                    **(metadata or {}),
                },
            )
            documents.append(doc)

        logger.info(
            "CSV rows loaded",
            filename=filename,
            document_count=len(documents),
        )

        return documents

    def _load_combined(
        self,
        rows: list[dict],
        headers: list[str],
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Combine all CSV rows into one document."""
        parts = []
        for idx, row in enumerate(rows):
            formatted = self._format_row(row, headers)
            if formatted.strip():
                parts.append(f"[Row {idx + 1}]\n{formatted}")

        content = "\n\n".join(parts)

        if not content.strip():
            logger.warning("No content extracted from CSV", filename=filename)
            return []

        doc = RawDocument(
            id=self._generate_doc_id(filename),
            content=content,
            source=filename,
            metadata={
                "file_type": "csv",
                "row_count": len(rows),
                "columns": headers,
                **(metadata or {}),
            },
        )

        logger.info(
            "CSV combined into single document",
            filename=filename,
            row_count=len(rows),
        )

        return [doc]

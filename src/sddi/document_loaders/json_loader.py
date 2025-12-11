"""
JSON Document Loader for SDDI Pipeline.

Loads JSON files and converts them to RawDocument format.
Supports both single objects and arrays of objects.
Optimized for large files with streaming support.
"""

import json
import uuid
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Generator

import structlog

# Try to import ijson for streaming JSON parsing
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument

logger = structlog.get_logger(__name__)


class JSONLoader(BaseDocumentLoader):
    """
    Robust loader for JSON files.

    Supports various JSON structures:
    - Single object: Converts to one document
    - Array of objects: Each object becomes a document
    - Nested structures: Flattened with dot notation keys
    - Large files: Uses streaming parser (ijson) for files > threshold

    Args:
        content_keys: List of keys to use as content (if None, uses all)
        id_key: Key to use for document ID (if present)
        flatten: Whether to flatten nested objects
        array_mode: How to handle arrays - 'split' (each item = doc) or 'combine'
        text_keys: Priority keys to look for text content
        streaming_threshold_mb: Use streaming parser for files larger than this (MB)
    """

    SUPPORTED_EXTENSIONS = ["json"]

    # Common keys that typically contain main text content
    DEFAULT_TEXT_KEYS = [
        "text", "content", "body", "description", "message",
        "title", "name", "summary", "abstract", "paragraph",
        "sentence", "value", "data", "answer", "question",
    ]

    def __init__(
        self,
        content_keys: list[str] | None = None,
        id_key: str | None = None,
        flatten: bool = True,
        array_mode: str = "split",
        text_keys: list[str] | None = None,
        streaming_threshold_mb: float = 10.0,
    ) -> None:
        self.content_keys = content_keys
        self.id_key = id_key
        self.flatten = flatten
        self.array_mode = array_mode
        self.text_keys = text_keys or self.DEFAULT_TEXT_KEYS
        self.streaming_threshold_bytes = int(streaming_threshold_mb * 1024 * 1024)

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """
        Load JSON file into RawDocument(s).

        Automatically chooses between standard and streaming parser
        based on file size.
        """
        logger.info("Starting JSON file load", filename=filename)

        try:
            # Read content to determine size and parsing strategy
            content_bytes = file.read()
            file_size = len(content_bytes)

            logger.info(
                "JSON file read",
                filename=filename,
                size_bytes=file_size,
                size_mb=round(file_size / (1024 * 1024), 2),
            )

            base_metadata = {
                "source": filename,
                "format": "json",
                "file_size": file_size,
                **(metadata or {}),
            }

            # Choose parsing strategy
            if file_size > self.streaming_threshold_bytes and IJSON_AVAILABLE:
                logger.info(
                    "Using streaming parser for large file",
                    filename=filename,
                    threshold_mb=self.streaming_threshold_bytes / (1024 * 1024),
                )
                documents = await self._load_streaming(
                    BytesIO(content_bytes), filename, base_metadata
                )
            else:
                if file_size > self.streaming_threshold_bytes and not IJSON_AVAILABLE:
                    logger.warning(
                        "ijson not available, using standard parser for large file",
                        filename=filename,
                        size_mb=round(file_size / (1024 * 1024), 2),
                    )
                documents = await self._load_standard(
                    content_bytes, filename, base_metadata
                )

            logger.info(
                "JSON file loaded successfully",
                filename=filename,
                documents_count=len(documents),
            )

            return documents

        except json.JSONDecodeError as e:
            logger.error(
                "JSON parsing error",
                filename=filename,
                error=str(e),
                line=e.lineno,
                column=e.colno,
            )
            # Try to salvage what we can
            return await self._load_with_recovery(file, filename, metadata or {})

        except UnicodeDecodeError as e:
            logger.error(
                "Encoding error - trying alternative encodings",
                filename=filename,
                error=str(e),
            )
            return await self._load_with_encoding_fallback(
                content_bytes, filename, metadata or {}
            )

        except Exception as e:
            logger.error(
                "Unexpected error loading JSON",
                filename=filename,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ValueError(f"Failed to load JSON file {filename}: {str(e)}") from e

    async def _load_standard(
        self,
        content_bytes: bytes,
        filename: str,
        metadata: dict[str, Any],
    ) -> list[RawDocument]:
        """Standard JSON parsing for smaller files."""
        # Try UTF-8 first, then other encodings
        content = None
        for encoding in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
            try:
                content = content_bytes.decode(encoding)
                logger.debug(f"Successfully decoded with {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise UnicodeDecodeError("utf-8", content_bytes, 0, 1, "Failed all encoding attempts")

        # Parse JSON
        data = json.loads(content)

        return self._process_json_data(data, filename, metadata)

    async def _load_streaming(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any],
    ) -> list[RawDocument]:
        """Streaming JSON parsing for large files using ijson."""
        if not IJSON_AVAILABLE:
            logger.warning("ijson not available, falling back to standard parser")
            file.seek(0)
            content_bytes = file.read()
            return await self._load_standard(content_bytes, filename, metadata)

        documents: list[RawDocument] = []
        doc_count = 0
        batch_size = 1000

        try:
            # Detect JSON structure first
            file.seek(0)
            first_bytes = file.read(100).decode("utf-8", errors="ignore").strip()
            file.seek(0)

            is_array = first_bytes.startswith("[")

            if is_array:
                # Parse array items one by one
                logger.info("Streaming array JSON", filename=filename)

                parser = ijson.items(file, "item")

                for item in parser:
                    doc = self._create_document(
                        item,
                        filename,
                        doc_count,
                        {**metadata, "array_index": doc_count},
                    )
                    if doc:
                        documents.append(doc)
                        doc_count += 1

                        if doc_count % batch_size == 0:
                            logger.info(
                                "Streaming progress",
                                filename=filename,
                                documents_processed=doc_count,
                            )
            else:
                # For objects, parse key-value pairs
                logger.info("Streaming object JSON", filename=filename)

                # Parse the entire object using ijson.kvitems
                parser = ijson.kvitems(file, "")
                current_obj: dict[str, Any] = {}

                for key, value in parser:
                    current_obj[key] = value

                # Create document from the collected object
                if current_obj:
                    doc = self._create_document(
                        current_obj,
                        filename,
                        0,
                        metadata,
                    )
                    if doc:
                        documents.append(doc)
                        doc_count = 1

            logger.info(
                "Streaming complete",
                filename=filename,
                total_documents=doc_count,
            )

        except Exception as e:
            # Catch any ijson-related errors
            logger.error(
                "Streaming parser error",
                filename=filename,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback to standard parsing
            file.seek(0)
            content_bytes = file.read()
            return await self._load_standard(content_bytes, filename, metadata)

        return documents

    async def _load_with_recovery(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any],
    ) -> list[RawDocument]:
        """
        Attempt to recover from malformed JSON.
        Tries line-by-line JSON parsing (JSONL format).
        """
        logger.info("Attempting JSON recovery (JSONL mode)", filename=filename)

        file.seek(0)
        content_bytes = file.read()

        documents: list[RawDocument] = []

        # Try decoding
        for encoding in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
            try:
                content = content_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error("Could not decode file with any encoding", filename=filename)
            return []

        # Try parsing line by line (JSONL format)
        lines = content.split("\n")
        valid_count = 0
        error_count = 0

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                doc = self._create_document(
                    data,
                    filename,
                    idx,
                    {**metadata, "source": filename, "format": "jsonl", "line": idx},
                )
                if doc:
                    documents.append(doc)
                    valid_count += 1
            except json.JSONDecodeError:
                error_count += 1
                continue

        if valid_count > 0:
            logger.info(
                "JSONL recovery successful",
                filename=filename,
                valid_lines=valid_count,
                error_lines=error_count,
            )
        else:
            logger.warning(
                "JSONL recovery failed - no valid JSON lines",
                filename=filename,
            )

        return documents

    async def _load_with_encoding_fallback(
        self,
        content_bytes: bytes,
        filename: str,
        metadata: dict[str, Any],
    ) -> list[RawDocument]:
        """Try multiple encodings to load the file."""
        encodings = ["utf-8-sig", "cp949", "euc-kr", "gb2312", "big5", "latin-1", "iso-8859-1"]

        for encoding in encodings:
            try:
                content = content_bytes.decode(encoding)
                data = json.loads(content)

                logger.info(
                    "Successfully loaded with fallback encoding",
                    filename=filename,
                    encoding=encoding,
                )

                return self._process_json_data(
                    data,
                    filename,
                    {**metadata, "encoding": encoding},
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

        logger.error("All encoding attempts failed", filename=filename)
        return []

    def _process_json_data(
        self,
        data: Any,
        filename: str,
        metadata: dict[str, Any],
    ) -> list[RawDocument]:
        """Process parsed JSON data into documents."""
        documents: list[RawDocument] = []

        if isinstance(data, list):
            if self.array_mode == "split":
                logger.info(
                    "Processing JSON array",
                    filename=filename,
                    items_count=len(data),
                )

                for idx, item in enumerate(data):
                    doc = self._create_document(
                        item,
                        filename,
                        idx,
                        {**metadata, "array_index": idx},
                    )
                    if doc:
                        documents.append(doc)

                    # Log progress for large arrays
                    if (idx + 1) % 1000 == 0:
                        logger.info(
                            "Array processing progress",
                            filename=filename,
                            processed=idx + 1,
                            total=len(data),
                        )
            else:
                # Combine mode - treat entire array as one document
                doc = self._create_document(data, filename, 0, metadata)
                if doc:
                    documents.append(doc)
        else:
            # Single object or primitive
            doc = self._create_document(data, filename, 0, metadata)
            if doc:
                documents.append(doc)

        return documents

    def _create_document(
        self,
        data: Any,
        filename: str,
        index: int,
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Create a RawDocument from JSON data."""
        if data is None:
            return None

        if isinstance(data, dict):
            return self._create_document_from_dict(data, filename, index, metadata)
        elif isinstance(data, list):
            # Handle nested arrays
            return self._create_document_from_list(data, filename, index, metadata)
        else:
            # Primitive value (string, number, boolean)
            return self._create_document_from_primitive(data, filename, index, metadata)

    def _create_document_from_dict(
        self,
        data: dict[str, Any],
        filename: str,
        index: int,
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Create document from dictionary."""
        # Extract ID if specified
        doc_id = None
        if self.id_key and self.id_key in data:
            doc_id = str(data[self.id_key])

        # Generate content
        content = self._extract_content_from_dict(data)

        if not content or not content.strip():
            logger.debug(
                "Empty content from dict, skipping",
                filename=filename,
                index=index,
                keys=list(data.keys())[:10],  # First 10 keys for debugging
            )
            return None

        # Generate doc ID if not provided
        if not doc_id:
            content_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]
            doc_id = f"{filename}:{index}:{content_hash}"

        return RawDocument(
            id=doc_id,
            content=content,
            source=filename,
            metadata={
                **metadata,
                "original_keys": list(data.keys()),
                "content_length": len(content),
            },
        )

    def _create_document_from_list(
        self,
        data: list[Any],
        filename: str,
        index: int,
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Create document from list (nested array)."""
        content_parts = []

        for item in data:
            if isinstance(item, dict):
                text = self._extract_content_from_dict(item)
                if text:
                    content_parts.append(text)
            elif isinstance(item, str):
                content_parts.append(item)
            elif item is not None:
                content_parts.append(str(item))

        content = "\n\n".join(content_parts)

        if not content.strip():
            return None

        content_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]

        return RawDocument(
            id=f"{filename}:{index}:{content_hash}",
            content=content,
            source=filename,
            metadata={
                **metadata,
                "array_length": len(data),
                "content_length": len(content),
            },
        )

    def _create_document_from_primitive(
        self,
        data: Any,
        filename: str,
        index: int,
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Create document from primitive value."""
        content = str(data).strip()

        if not content:
            return None

        content_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]

        return RawDocument(
            id=f"{filename}:{index}:{content_hash}",
            content=content,
            source=filename,
            metadata={
                **metadata,
                "content_length": len(content),
                "original_type": type(data).__name__,
            },
        )

    def _extract_content_from_dict(self, data: dict[str, Any]) -> str:
        """
        Extract text content from dictionary.

        Prioritizes text-like keys, then falls back to full conversion.
        """
        # If content_keys specified, use only those
        if self.content_keys:
            parts = []
            flat_data = self._flatten_dict(data) if self.flatten else data

            for key in self.content_keys:
                if key in flat_data:
                    value = flat_data[key]
                    if isinstance(value, str):
                        parts.append(f"{key}: {value}")
                    elif value is not None:
                        parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")

            if parts:
                return "\n".join(parts)

        # Try to find text content in priority keys
        for key in self.text_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and value.strip():
                    # Found primary text content
                    other_parts = []
                    for k, v in data.items():
                        if k != key and v is not None:
                            if isinstance(v, str):
                                other_parts.append(f"{k}: {v}")
                            elif isinstance(v, (int, float, bool)):
                                other_parts.append(f"{k}: {v}")

                    if other_parts:
                        return f"{value}\n\n[Metadata]\n" + "\n".join(other_parts)
                    return value

        # Fall back to full dict conversion
        if self.flatten:
            flat_data = self._flatten_dict(data)
            return self._dict_to_text(flat_data)
        else:
            return self._dict_to_text(data)

    def _flatten_dict(
        self,
        data: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Flatten nested dictionary with dot notation."""
        items: list[tuple[str, Any]] = []

        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep).items())
            elif isinstance(value, list):
                # Handle list specially
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    # Simple list - join as string
                    items.append((new_key, ", ".join(str(v) for v in value)))
                else:
                    # Complex list - JSON serialize
                    items.append((new_key, json.dumps(value, ensure_ascii=False)))
            else:
                items.append((new_key, value))

        return dict(items)

    def _dict_to_text(self, data: dict[str, Any]) -> str:
        """Convert dictionary to readable text format."""
        lines = []

        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, str):
                if value.strip():
                    lines.append(f"{key}: {value}")
            elif isinstance(value, (dict, list)):
                json_str = json.dumps(value, ensure_ascii=False, indent=2)
                lines.append(f"{key}: {json_str}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    async def load_from_path(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Load JSON file from filesystem path."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        logger.info("Loading JSON from path", path=str(path))

        with open(path, "rb") as f:
            return await self.load(f, path.name, metadata)

    def load_sync(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Synchronous version of load for non-async contexts."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.load(file, filename, metadata))

"""
YAML Document Loader for SDDI Pipeline.

Loads YAML files and converts them to RawDocument format.
Supports single and multi-document YAML files.
"""

import uuid
from typing import Any, BinaryIO

import yaml

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument


class YAMLLoader(BaseDocumentLoader):
    """
    Loader for YAML files.

    Supports:
    - Single YAML documents
    - Multi-document YAML files (separated by ---)
    - Various YAML structures (mappings, sequences, scalars)

    Args:
        content_keys: List of keys to use as content (if None, uses all)
        id_key: Key to use for document ID (if present)
        split_documents: Whether to split multi-document YAML into separate docs
        flatten: Whether to flatten nested structures
    """

    SUPPORTED_EXTENSIONS = ["yaml", "yml"]

    def __init__(
        self,
        content_keys: list[str] | None = None,
        id_key: str | None = None,
        split_documents: bool = True,
        flatten: bool = True,
    ) -> None:
        self.content_keys = content_keys
        self.id_key = id_key
        self.split_documents = split_documents
        self.flatten = flatten

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Load YAML file into RawDocument(s)."""
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        base_metadata = {
            "source": filename,
            "format": "yaml",
            **(metadata or {}),
        }

        documents: list[RawDocument] = []

        # Parse all YAML documents in the file
        yaml_docs = list(yaml.safe_load_all(content))

        if self.split_documents:
            for idx, yaml_doc in enumerate(yaml_docs):
                if yaml_doc is None:
                    continue

                doc = self._create_document(
                    yaml_doc,
                    filename,
                    idx,
                    {**base_metadata, "yaml_doc_index": idx},
                )
                if doc:
                    documents.append(doc)
        else:
            # Combine all YAML documents
            combined_content = []
            for idx, yaml_doc in enumerate(yaml_docs):
                if yaml_doc is None:
                    continue
                combined_content.append(f"--- Document {idx + 1} ---")
                combined_content.append(self._data_to_text(yaml_doc))

            if combined_content:
                documents.append(
                    RawDocument(
                        id=f"{filename}:combined:{uuid.uuid4().hex[:8]}",
                        content="\n\n".join(combined_content),
                        source=filename,
                        metadata={**base_metadata, "yaml_doc_count": len(yaml_docs)},
                    )
                )

        return documents

    def _create_document(
        self,
        data: Any,
        filename: str,
        index: int,
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Create a RawDocument from YAML data."""
        if data is None:
            return None

        if isinstance(data, dict):
            # Extract ID if specified
            doc_id = None
            if self.id_key and self.id_key in data:
                doc_id = str(data[self.id_key])

            # Flatten if needed
            if self.flatten:
                flat_data = self._flatten_dict(data)
            else:
                flat_data = data

            # Extract content
            if self.content_keys:
                content_parts = []
                for key in self.content_keys:
                    if key in flat_data:
                        content_parts.append(f"{key}: {flat_data[key]}")
                content = "\n".join(content_parts)
            else:
                content = self._data_to_text(flat_data)

            if not content.strip():
                return None

            return RawDocument(
                id=doc_id or f"{filename}:{index}:{uuid.uuid4().hex[:8]}",
                content=content,
                source=filename,
                metadata={**metadata, "yaml_type": "mapping"},
            )

        elif isinstance(data, list):
            # Handle list at root level
            content_parts = []
            for idx, item in enumerate(data):
                content_parts.append(f"[{idx}] {self._data_to_text(item)}")

            content = "\n".join(content_parts)
            if not content.strip():
                return None

            return RawDocument(
                id=f"{filename}:{index}:{uuid.uuid4().hex[:8]}",
                content=content,
                source=filename,
                metadata={**metadata, "yaml_type": "sequence", "item_count": len(data)},
            )

        else:
            # Scalar value
            content = str(data)
            if not content.strip():
                return None

            return RawDocument(
                id=f"{filename}:{index}:{uuid.uuid4().hex[:8]}",
                content=content,
                source=filename,
                metadata={**metadata, "yaml_type": "scalar"},
            )

    def _flatten_dict(
        self,
        data: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Flatten nested dictionary with dot notation."""
        items: list[tuple[str, Any]] = []

        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep).items())
            elif isinstance(value, list):
                # Convert list to string or expand based on content
                if all(isinstance(v, (str, int, float, bool)) for v in value):
                    items.append((new_key, ", ".join(str(v) for v in value)))
                else:
                    items.append((new_key, self._list_to_text(value)))
            else:
                items.append((new_key, value))

        return dict(items)

    def _data_to_text(self, data: Any, indent: int = 0) -> str:
        """Convert YAML data to readable text format."""
        prefix = "  " * indent

        if data is None:
            return f"{prefix}null"

        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._data_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return "\n".join(lines)

        elif isinstance(data, list):
            return self._list_to_text(data, indent)

        else:
            return f"{prefix}{data}"

    def _list_to_text(self, data: list, indent: int = 0) -> str:
        """Convert a list to text format."""
        prefix = "  " * indent
        lines = []

        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(self._data_to_text(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")

        return "\n".join(lines)

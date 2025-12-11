"""
JSON-LD Document Loader for SDDI Pipeline.

Loads JSON-LD (Linked Data) files and extracts semantic information
for knowledge graph construction.
"""

import json
import uuid
from typing import Any, BinaryIO

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument


class JSONLDLoader(BaseDocumentLoader):
    """
    Loader for JSON-LD (JSON for Linked Data) files.

    JSON-LD is a method of encoding Linked Data using JSON.
    This loader extracts entities and their relationships from
    JSON-LD structures, preserving semantic information.

    Handles:
    - @context: Namespace definitions
    - @id: Resource identifiers
    - @type: Type declarations
    - @graph: Multiple resources
    - Nested objects and references

    Args:
        extract_context: Whether to include @context info in metadata
        preserve_iris: Whether to preserve full IRIs or use local names
        split_graph: Whether to split @graph items into separate documents
    """

    SUPPORTED_EXTENSIONS = ["jsonld", "json-ld"]

    def __init__(
        self,
        extract_context: bool = True,
        preserve_iris: bool = False,
        split_graph: bool = True,
    ) -> None:
        self.extract_context = extract_context
        self.preserve_iris = preserve_iris
        self.split_graph = split_graph

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Load JSON-LD file into RawDocument(s)."""
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        data = json.loads(content)
        base_metadata = {
            "source": filename,
            "format": "json-ld",
            **(metadata or {}),
        }

        # Extract context if present
        context = {}
        if "@context" in data and self.extract_context:
            context = self._extract_context(data["@context"])
            base_metadata["jsonld_context"] = context

        documents: list[RawDocument] = []

        # Handle @graph (multiple resources)
        if "@graph" in data:
            graph_items = data["@graph"]
            if self.split_graph:
                for idx, item in enumerate(graph_items):
                    doc = self._create_document_from_resource(
                        item,
                        filename,
                        idx,
                        context,
                        {**base_metadata, "graph_index": idx},
                    )
                    if doc:
                        documents.append(doc)
            else:
                # Combine all graph items
                doc = self._create_combined_document(
                    graph_items, filename, context, base_metadata
                )
                if doc:
                    documents.append(doc)
        else:
            # Single resource
            doc = self._create_document_from_resource(
                data, filename, 0, context, base_metadata
            )
            if doc:
                documents.append(doc)

        return documents

    def _extract_context(self, context: Any) -> dict[str, str]:
        """Extract namespace prefixes from @context."""
        if isinstance(context, str):
            return {"@vocab": context}
        elif isinstance(context, dict):
            result = {}
            for key, value in context.items():
                if isinstance(value, str):
                    result[key] = value
                elif isinstance(value, dict) and "@id" in value:
                    result[key] = value["@id"]
            return result
        elif isinstance(context, list):
            result = {}
            for item in context:
                result.update(self._extract_context(item))
            return result
        return {}

    def _create_document_from_resource(
        self,
        resource: dict[str, Any],
        filename: str,
        index: int,
        context: dict[str, str],
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Create a document from a JSON-LD resource."""
        if not isinstance(resource, dict):
            return None

        # Extract key JSON-LD properties
        resource_id = resource.get("@id", "")
        resource_type = resource.get("@type", [])
        if isinstance(resource_type, str):
            resource_type = [resource_type]

        # Build content from properties
        content_parts = []

        # Add type information
        if resource_type:
            type_names = [self._resolve_name(t, context) for t in resource_type]
            content_parts.append(f"Type: {', '.join(type_names)}")

        # Add ID if present
        if resource_id:
            content_parts.append(f"ID: {self._resolve_name(resource_id, context)}")

        # Process other properties
        for key, value in resource.items():
            if key.startswith("@"):
                continue

            prop_name = self._resolve_name(key, context)
            prop_value = self._format_value(value, context)

            if prop_value:
                content_parts.append(f"{prop_name}: {prop_value}")

        content = "\n".join(content_parts)
        if not content.strip():
            return None

        # Generate document ID
        doc_id = resource_id if resource_id else f"{filename}:{index}:{uuid.uuid4().hex[:8]}"

        return RawDocument(
            id=doc_id,
            content=content,
            source=filename,
            metadata={
                **metadata,
                "jsonld_id": resource_id,
                "jsonld_types": resource_type,
            },
        )

    def _create_combined_document(
        self,
        resources: list[dict[str, Any]],
        filename: str,
        context: dict[str, str],
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Create a single document from multiple resources."""
        content_parts = []

        for idx, resource in enumerate(resources):
            if not isinstance(resource, dict):
                continue

            resource_id = resource.get("@id", f"Resource {idx}")
            resource_type = resource.get("@type", [])
            if isinstance(resource_type, str):
                resource_type = [resource_type]

            # Add section header
            type_str = ", ".join(self._resolve_name(t, context) for t in resource_type)
            header = f"[{self._resolve_name(resource_id, context)}]"
            if type_str:
                header += f" ({type_str})"
            content_parts.append(header)

            # Add properties
            for key, value in resource.items():
                if key.startswith("@"):
                    continue

                prop_name = self._resolve_name(key, context)
                prop_value = self._format_value(value, context)

                if prop_value:
                    content_parts.append(f"  {prop_name}: {prop_value}")

            content_parts.append("")  # Blank line between resources

        content = "\n".join(content_parts)
        if not content.strip():
            return None

        return RawDocument(
            id=f"{filename}:combined:{uuid.uuid4().hex[:8]}",
            content=content,
            source=filename,
            metadata={**metadata, "resource_count": len(resources)},
        )

    def _resolve_name(self, uri: str, context: dict[str, str]) -> str:
        """Resolve a URI to a readable name."""
        if self.preserve_iris:
            return uri

        # Check if it's a prefixed name (e.g., "schema:name")
        if ":" in uri and not uri.startswith("http"):
            prefix, local = uri.split(":", 1)
            if prefix in context:
                return local
            return uri

        # Extract local name from full URI
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]

        return uri

    def _format_value(self, value: Any, context: dict[str, str]) -> str:
        """Format a JSON-LD value for text output."""
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, (int, float, bool)):
            return str(value)

        if isinstance(value, dict):
            # Handle typed values
            if "@value" in value:
                return str(value["@value"])
            # Handle references
            if "@id" in value:
                return self._resolve_name(value["@id"], context)
            # Handle language-tagged strings
            if "@language" in value and "@value" in value:
                return f"{value['@value']} ({value['@language']})"
            # Nested object - summarize
            return json.dumps(value, ensure_ascii=False)

        if isinstance(value, list):
            formatted = [self._format_value(v, context) for v in value]
            return ", ".join(f for f in formatted if f)

        return str(value)

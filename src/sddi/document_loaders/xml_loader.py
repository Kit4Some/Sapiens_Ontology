"""
XML Document Loader for SDDI Pipeline.

Loads XML files and converts them to RawDocument format.
Supports various XML structures including RDF/XML.

Uses defusedxml for secure parsing of untrusted XML data.
"""

import io
import uuid
import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DefusedET
from typing import Any, BinaryIO

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument


class XMLLoader(BaseDocumentLoader):
    """
    Loader for XML files.

    Supports:
    - Standard XML documents
    - RDF/XML format
    - Configurable element-to-document mapping
    - Namespace handling

    Args:
        root_elements: List of element tags to treat as document roots
                      (if None, uses immediate children of root)
        text_elements: List of element tags to extract text from
        include_attributes: Whether to include element attributes
        strip_namespaces: Whether to remove namespace prefixes
    """

    SUPPORTED_EXTENSIONS = ["xml", "rdf"]

    def __init__(
        self,
        root_elements: list[str] | None = None,
        text_elements: list[str] | None = None,
        include_attributes: bool = True,
        strip_namespaces: bool = True,
    ) -> None:
        self.root_elements = root_elements
        self.text_elements = text_elements
        self.include_attributes = include_attributes
        self.strip_namespaces = strip_namespaces

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Load XML file into RawDocument(s)."""
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        # Parse XML securely using defusedxml
        root = DefusedET.fromstring(content)

        # Extract namespaces
        namespaces = self._extract_namespaces(content)

        base_metadata = {
            "source": filename,
            "format": "xml",
            "namespaces": namespaces,
            **(metadata or {}),
        }

        documents: list[RawDocument] = []

        # Find document root elements
        if self.root_elements:
            # Find specific elements as document roots
            for elem_name in self.root_elements:
                for idx, elem in enumerate(root.iter()):
                    local_name = self._get_local_name(elem.tag)
                    if local_name == elem_name:
                        doc = self._element_to_document(
                            elem,
                            filename,
                            idx,
                            {**base_metadata, "root_element": elem_name},
                        )
                        if doc:
                            documents.append(doc)
        else:
            # Use immediate children of root
            for idx, child in enumerate(root):
                doc = self._element_to_document(
                    child,
                    filename,
                    idx,
                    base_metadata,
                )
                if doc:
                    documents.append(doc)

        # If no children processed, treat entire document as one
        if not documents:
            doc = self._element_to_document(root, filename, 0, base_metadata)
            if doc:
                documents.append(doc)

        return documents

    def _extract_namespaces(self, content: str) -> dict[str, str]:
        """Extract namespace declarations from XML."""
        namespaces = {}
        try:
            # Simple regex-free approach using defusedxml
            for event, elem in DefusedET.iterparse(
                io.StringIO(content),
                events=["start-ns"],
            ):
                prefix, uri = elem
                namespaces[prefix if prefix else "default"] = uri
        except Exception:
            pass
        return namespaces

    def _get_local_name(self, tag: str) -> str:
        """Extract local name from a potentially namespaced tag."""
        if "}" in tag:
            return tag.split("}")[-1]
        return tag

    def _element_to_document(
        self,
        element: ET.Element,
        filename: str,
        index: int,
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Convert an XML element to a RawDocument."""
        content_parts = []

        # Add element name
        elem_name = self._get_local_name(element.tag)
        if self.strip_namespaces:
            content_parts.append(f"Element: {elem_name}")
        else:
            content_parts.append(f"Element: {element.tag}")

        # Add attributes
        if self.include_attributes and element.attrib:
            for attr, value in element.attrib.items():
                attr_name = self._get_local_name(attr) if self.strip_namespaces else attr
                content_parts.append(f"  @{attr_name}: {value}")

        # Add element text
        if element.text and element.text.strip():
            content_parts.append(f"  Text: {element.text.strip()}")

        # Process child elements
        child_content = self._process_children(element, depth=1)
        if child_content:
            content_parts.append(child_content)

        content = "\n".join(content_parts)
        if not content.strip():
            return None

        # Generate ID from element attributes or index
        doc_id = None
        for id_attr in ["id", "ID", "Id", "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"]:
            if id_attr in element.attrib:
                doc_id = element.attrib[id_attr]
                break

        if not doc_id:
            doc_id = f"{filename}:{elem_name}:{index}:{uuid.uuid4().hex[:8]}"

        return RawDocument(
            id=doc_id,
            content=content,
            source=filename,
            metadata={
                **metadata,
                "element_name": elem_name,
                "attributes": dict(element.attrib),
            },
        )

    def _process_children(self, element: ET.Element, depth: int = 0) -> str:
        """Recursively process child elements."""
        if depth > 10:  # Prevent infinite recursion
            return ""

        lines = []
        indent = "  " * (depth + 1)

        for child in element:
            child_name = self._get_local_name(child.tag)
            if self.strip_namespaces:
                lines.append(f"{indent}{child_name}:")
            else:
                lines.append(f"{indent}{child.tag}:")

            # Add attributes
            if self.include_attributes and child.attrib:
                for attr, value in child.attrib.items():
                    attr_name = self._get_local_name(attr) if self.strip_namespaces else attr
                    lines.append(f"{indent}  @{attr_name}: {value}")

            # Add text content
            if child.text and child.text.strip():
                # Check if this is a text element we want to extract
                if self.text_elements is None or child_name in self.text_elements:
                    lines.append(f"{indent}  {child.text.strip()}")

            # Process nested children
            if len(child) > 0:
                nested = self._process_children(child, depth + 1)
                if nested:
                    lines.append(nested)

            # Add tail text (text after closing tag)
            if child.tail and child.tail.strip():
                lines.append(f"{indent}  {child.tail.strip()}")

        return "\n".join(lines)


class RDFXMLLoader(XMLLoader):
    """
    Specialized loader for RDF/XML format.

    Extracts RDF triples and resources from RDF/XML documents.
    """

    SUPPORTED_EXTENSIONS = ["rdf", "owl"]

    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"

    def __init__(self) -> None:
        super().__init__(
            include_attributes=True,
            strip_namespaces=False,  # Keep namespaces for RDF
        )

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Load RDF/XML file into RawDocument(s)."""
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        # Parse XML securely using defusedxml
        root = DefusedET.fromstring(content)
        namespaces = self._extract_namespaces(content)

        base_metadata = {
            "source": filename,
            "format": "rdf-xml",
            "namespaces": namespaces,
            **(metadata or {}),
        }

        documents: list[RawDocument] = []

        # Find all RDF resources (elements with rdf:about or rdf:ID)
        for idx, elem in enumerate(root.iter()):
            about = elem.get(f"{{{self.RDF_NS}}}about")
            rdf_id = elem.get(f"{{{self.RDF_NS}}}ID")

            if about or rdf_id:
                doc = self._rdf_resource_to_document(
                    elem,
                    filename,
                    idx,
                    about or rdf_id,
                    base_metadata,
                )
                if doc:
                    documents.append(doc)

        # If no RDF resources found, fall back to standard XML processing
        if not documents:
            return await super().load(
                __import__("io").BytesIO(content.encode("utf-8")),
                filename,
                metadata,
            )

        return documents

    def _rdf_resource_to_document(
        self,
        element: ET.Element,
        filename: str,
        index: int,
        resource_uri: str,
        metadata: dict[str, Any],
    ) -> RawDocument | None:
        """Convert an RDF resource element to a RawDocument."""
        content_parts = []

        # Resource URI
        content_parts.append(f"Resource: {resource_uri}")

        # Resource type
        elem_type = self._get_local_name(element.tag)
        if elem_type != "Description":
            content_parts.append(f"Type: {elem_type}")

        # Process properties (child elements)
        for child in element:
            prop_name = self._get_local_name(child.tag)

            # Check for resource reference
            resource_ref = child.get(f"{{{self.RDF_NS}}}resource")
            if resource_ref:
                content_parts.append(f"{prop_name}: -> {resource_ref}")
            elif child.text and child.text.strip():
                # Literal value
                datatype = child.get(f"{{{self.RDF_NS}}}datatype")
                lang = child.get(f"{{{self.XML_NS}}}lang") if hasattr(self, "XML_NS") else None

                value = child.text.strip()
                if datatype:
                    value += f" (type: {self._get_local_name(datatype)})"
                if lang:
                    value += f" (@{lang})"

                content_parts.append(f"{prop_name}: {value}")
            elif len(child) > 0:
                # Nested resource or blank node
                content_parts.append(f"{prop_name}: [nested]")

        content = "\n".join(content_parts)
        if not content.strip():
            return None

        return RawDocument(
            id=resource_uri,
            content=content,
            source=filename,
            metadata={
                **metadata,
                "rdf_type": elem_type,
                "resource_uri": resource_uri,
            },
        )

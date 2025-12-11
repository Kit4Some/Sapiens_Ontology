"""
Document Loaders for SDDI Pipeline.

Supports multiple file formats:
- Text: .txt, .text
- Markdown: .md, .markdown
- CSV: .csv
- PDF: .pdf
- Microsoft Word: .docx, .doc
- JSON: .json
- JSON-LD: .jsonld, .json-ld
- XML: .xml, .rdf, .owl
- YAML: .yaml, .yml
- HTML: .html, .htm, .xhtml
"""

from src.sddi.document_loaders.base import BaseDocumentLoader, DocumentLoaderFactory
from src.sddi.document_loaders.text_loader import TextLoader
from src.sddi.document_loaders.markdown_loader import MarkdownLoader
from src.sddi.document_loaders.csv_loader import CSVLoader
from src.sddi.document_loaders.pdf_loader import PDFLoader
from src.sddi.document_loaders.docx_loader import DOCXLoader, DOCLoader
from src.sddi.document_loaders.json_loader import JSONLoader
from src.sddi.document_loaders.jsonld_loader import JSONLDLoader
from src.sddi.document_loaders.xml_loader import XMLLoader, RDFXMLLoader
from src.sddi.document_loaders.yaml_loader import YAMLLoader
from src.sddi.document_loaders.html_loader import HTMLLoader

__all__ = [
    "BaseDocumentLoader",
    "DocumentLoaderFactory",
    "TextLoader",
    "MarkdownLoader",
    "CSVLoader",
    "PDFLoader",
    "DOCXLoader",
    "DOCLoader",
    "JSONLoader",
    "JSONLDLoader",
    "XMLLoader",
    "RDFXMLLoader",
    "YAMLLoader",
    "HTMLLoader",
]

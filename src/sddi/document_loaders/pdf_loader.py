"""
PDF Document Loader.
"""

import io
from pathlib import Path
from typing import BinaryIO

import structlog

from src.sddi.document_loaders.base import BaseDocumentLoader
from src.sddi.state import RawDocument

logger = structlog.get_logger(__name__)


class PDFLoader(BaseDocumentLoader):
    """
    Loader for PDF files (.pdf).

    Uses PyMuPDF (fitz) for text extraction with fallback to PyPDF2.
    """

    SUPPORTED_EXTENSIONS = ["pdf"]

    def __init__(
        self,
        one_doc_per_page: bool = False,
        extract_images: bool = False,
        ocr_enabled: bool = False,
    ):
        """
        Initialize the PDF loader.

        Args:
            one_doc_per_page: If True, create one document per page
            extract_images: If True, extract text from images (requires OCR)
            ocr_enabled: If True, use OCR for scanned PDFs
        """
        self.one_doc_per_page = one_doc_per_page
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled

    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load a PDF file."""
        logger.info("Loading PDF file", filename=filename)

        content = file.read()

        # Try PyMuPDF first, then fallback to PyPDF2
        try:
            return await self._load_with_pymupdf(content, filename, metadata)
        except ImportError:
            logger.warning("PyMuPDF not available, trying PyPDF2")
            try:
                return await self._load_with_pypdf2(content, filename, metadata)
            except ImportError:
                raise ImportError(
                    "No PDF library available. Install either:\n"
                    "  pip install PyMuPDF  (recommended)\n"
                    "  pip install pypdf"
                )

    async def load_from_path(
        self,
        path: Path,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load a PDF file from path."""
        with open(path, "rb") as f:
            return await self.load(f, path.name, metadata)

    async def _load_with_pymupdf(
        self,
        content: bytes,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load PDF using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF

        pdf_doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(pdf_doc)

        logger.info("PDF opened with PyMuPDF", filename=filename, pages=page_count)

        if self.one_doc_per_page:
            return self._extract_per_page_pymupdf(pdf_doc, filename, metadata)

        # Extract all text
        all_text = []
        for page_num, page in enumerate(pdf_doc):
            text = page.get_text("text")
            if text.strip():
                all_text.append(f"[Page {page_num + 1}]\n{text}")

        pdf_doc.close()

        combined_text = "\n\n".join(all_text)

        if not combined_text.strip():
            logger.warning("No text extracted from PDF", filename=filename)
            return []

        doc = RawDocument(
            id=self._generate_doc_id(filename),
            content=combined_text,
            source=filename,
            metadata={
                "file_type": "pdf",
                "page_count": page_count,
                "extractor": "pymupdf",
                **(metadata or {}),
            },
        )

        logger.info(
            "PDF loaded",
            filename=filename,
            pages=page_count,
            char_count=len(combined_text),
        )

        return [doc]

    def _extract_per_page_pymupdf(
        self,
        pdf_doc,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Extract one document per page using PyMuPDF."""
        documents = []

        for page_num, page in enumerate(pdf_doc):
            text = page.get_text("text").strip()
            if not text:
                continue

            doc = RawDocument(
                id=self._generate_doc_id(filename, page_num),
                content=text,
                source=filename,
                metadata={
                    "file_type": "pdf",
                    "page_number": page_num + 1,
                    "total_pages": len(pdf_doc),
                    "extractor": "pymupdf",
                    **(metadata or {}),
                },
            )
            documents.append(doc)

        pdf_doc.close()

        logger.info(
            "PDF pages extracted",
            filename=filename,
            document_count=len(documents),
        )

        return documents

    async def _load_with_pypdf2(
        self,
        content: bytes,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Load PDF using PyPDF2 (fallback)."""
        from pypdf import PdfReader

        pdf_reader = PdfReader(io.BytesIO(content))
        page_count = len(pdf_reader.pages)

        logger.info("PDF opened with PyPDF2", filename=filename, pages=page_count)

        if self.one_doc_per_page:
            return self._extract_per_page_pypdf2(pdf_reader, filename, metadata)

        # Extract all text
        all_text = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text and text.strip():
                all_text.append(f"[Page {page_num + 1}]\n{text}")

        combined_text = "\n\n".join(all_text)

        if not combined_text.strip():
            logger.warning("No text extracted from PDF", filename=filename)
            return []

        doc = RawDocument(
            id=self._generate_doc_id(filename),
            content=combined_text,
            source=filename,
            metadata={
                "file_type": "pdf",
                "page_count": page_count,
                "extractor": "pypdf2",
                **(metadata or {}),
            },
        )

        logger.info(
            "PDF loaded",
            filename=filename,
            pages=page_count,
            char_count=len(combined_text),
        )

        return [doc]

    def _extract_per_page_pypdf2(
        self,
        pdf_reader,
        filename: str,
        metadata: dict | None = None,
    ) -> list[RawDocument]:
        """Extract one document per page using PyPDF2."""
        documents = []
        page_count = len(pdf_reader.pages)

        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            doc = RawDocument(
                id=self._generate_doc_id(filename, page_num),
                content=text.strip(),
                source=filename,
                metadata={
                    "file_type": "pdf",
                    "page_number": page_num + 1,
                    "total_pages": page_count,
                    "extractor": "pypdf2",
                    **(metadata or {}),
                },
            )
            documents.append(doc)

        logger.info(
            "PDF pages extracted",
            filename=filename,
            document_count=len(documents),
        )

        return documents

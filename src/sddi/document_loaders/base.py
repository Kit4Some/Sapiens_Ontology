"""
Base Document Loader and Factory.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Any

import structlog

from src.sddi.state import RawDocument

logger = structlog.get_logger(__name__)


class BaseDocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    Each loader handles a specific file format and converts it
    to RawDocument objects for the SDDI pipeline.
    """

    # Supported file extensions for this loader
    SUPPORTED_EXTENSIONS: list[str] = []

    @abstractmethod
    async def load(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """
        Load a file and convert to RawDocument objects.

        Args:
            file: File-like object with binary content
            filename: Original filename
            metadata: Optional metadata to attach to documents

        Returns:
            List of RawDocument objects
        """
        pass

    async def load_from_path(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """
        Load a file from filesystem path.

        Default implementation opens the file and delegates to load().
        Subclasses can override for custom behavior.

        Args:
            path: Path to the file
            metadata: Optional metadata to attach to documents

        Returns:
            List of RawDocument objects
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        logger.info("Loading file from path", path=str(path), loader=self.__class__.__name__)

        with open(path, "rb") as f:
            return await self.load(f, path.name, metadata)

    def _generate_doc_id(self, filename: str, index: int = 0) -> str:
        """Generate a unique document ID."""
        import hashlib
        key = f"{filename}:{index}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if this loader supports the given file extension."""
        ext = extension.lower().lstrip(".")
        return ext in [e.lower().lstrip(".") for e in cls.SUPPORTED_EXTENSIONS]


class DocumentLoaderFactory:
    """
    Factory for creating document loaders based on file type.

    Provides centralized registration and retrieval of document loaders.
    """

    _loaders: dict[str, type[BaseDocumentLoader]] = {}

    @classmethod
    def register(cls, loader_class: type[BaseDocumentLoader]) -> None:
        """Register a loader class for its supported extensions."""
        for ext in loader_class.SUPPORTED_EXTENSIONS:
            ext = ext.lower().lstrip(".")
            cls._loaders[ext] = loader_class
            logger.debug(f"Registered loader for .{ext}: {loader_class.__name__}")

    @classmethod
    def get_loader(cls, filename: str, **loader_kwargs) -> BaseDocumentLoader:
        """
        Get appropriate loader for a file.

        Args:
            filename: Name of the file (used to determine extension)
            **loader_kwargs: Additional arguments passed to loader constructor

        Returns:
            Instance of appropriate loader

        Raises:
            ValueError: If no loader supports the file extension
        """
        ext = Path(filename).suffix.lower().lstrip(".")

        if ext not in cls._loaders:
            supported = list(cls._loaders.keys())
            raise ValueError(
                f"Unsupported file extension: .{ext}. "
                f"Supported: {', '.join('.' + e for e in supported)}"
            )

        loader_class = cls._loaders[ext]
        logger.info(
            "Creating document loader",
            filename=filename,
            extension=ext,
            loader=loader_class.__name__,
        )
        return loader_class(**loader_kwargs)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of all supported file extensions."""
        return list(cls._loaders.keys())

    @classmethod
    async def load_file(
        cls,
        file: BinaryIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
        **loader_kwargs,
    ) -> list[RawDocument]:
        """
        Convenience method to load a file with the appropriate loader.

        Args:
            file: File-like object
            filename: Original filename
            metadata: Optional metadata
            **loader_kwargs: Additional arguments passed to loader constructor

        Returns:
            List of RawDocument objects
        """
        loader = cls.get_loader(filename, **loader_kwargs)

        try:
            documents = await loader.load(file, filename, metadata)

            logger.info(
                "File loaded successfully",
                filename=filename,
                documents_count=len(documents),
                loader=loader.__class__.__name__,
            )

            return documents

        except Exception as e:
            logger.error(
                "Failed to load file",
                filename=filename,
                error=str(e),
                error_type=type(e).__name__,
                loader=loader.__class__.__name__,
            )
            raise

    @classmethod
    async def load_files(
        cls,
        files: list[tuple[BinaryIO, str, dict[str, Any] | None]],
    ) -> tuple[list[RawDocument], list[str]]:
        """
        Load multiple files, collecting results and errors.

        Args:
            files: List of (file, filename, metadata) tuples

        Returns:
            Tuple of (documents, errors) where errors is list of error messages
        """
        all_documents: list[RawDocument] = []
        errors: list[str] = []

        for file, filename, metadata in files:
            try:
                documents = await cls.load_file(file, filename, metadata)
                all_documents.extend(documents)
            except Exception as e:
                error_msg = f"Failed to load {filename}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return all_documents, errors


# Auto-register loaders when module is imported
def _register_default_loaders() -> None:
    """Register all default document loaders."""
    try:
        from src.sddi.document_loaders.text_loader import TextLoader
        from src.sddi.document_loaders.markdown_loader import MarkdownLoader
        from src.sddi.document_loaders.csv_loader import CSVLoader
        from src.sddi.document_loaders.pdf_loader import PDFLoader
        from src.sddi.document_loaders.json_loader import JSONLoader
        from src.sddi.document_loaders.jsonld_loader import JSONLDLoader
        from src.sddi.document_loaders.xml_loader import XMLLoader, RDFXMLLoader
        from src.sddi.document_loaders.yaml_loader import YAMLLoader
        from src.sddi.document_loaders.html_loader import HTMLLoader
        from src.sddi.document_loaders.docx_loader import DOCXLoader, DOCLoader

        # Basic text formats
        DocumentLoaderFactory.register(TextLoader)
        DocumentLoaderFactory.register(MarkdownLoader)
        DocumentLoaderFactory.register(CSVLoader)
        DocumentLoaderFactory.register(PDFLoader)

        # Microsoft Office formats
        DocumentLoaderFactory.register(DOCXLoader)
        DocumentLoaderFactory.register(DOCLoader)

        # Structured data formats
        DocumentLoaderFactory.register(JSONLoader)
        DocumentLoaderFactory.register(JSONLDLoader)
        DocumentLoaderFactory.register(XMLLoader)
        DocumentLoaderFactory.register(RDFXMLLoader)  # .rdf, .owl
        DocumentLoaderFactory.register(YAMLLoader)
        DocumentLoaderFactory.register(HTMLLoader)

        logger.info(
            "Document loaders registered",
            supported_extensions=DocumentLoaderFactory.get_supported_extensions(),
        )

    except ImportError as e:
        logger.warning(f"Some loaders could not be registered: {e}")


_register_default_loaders()

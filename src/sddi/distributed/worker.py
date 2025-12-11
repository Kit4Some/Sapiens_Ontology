"""
Celery Worker for Distributed Document Processing.

Provides async task processing for:
- Document chunking
- Entity extraction
- Relation extraction
- Embedding generation
- Graph loading
"""

import asyncio
import os
from datetime import datetime
from typing import Any

import structlog

# Celery setup with graceful degradation
try:
    from celery import Celery, Task
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None
    Task = object
    AsyncResult = None

logger = structlog.get_logger(__name__)

# =============================================================================
# Celery Configuration
# =============================================================================

# Default broker and backend URLs (Redis)
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

if CELERY_AVAILABLE:
    celery_app = Celery(
        "ontology_pipeline",
        broker=BROKER_URL,
        backend=BACKEND_URL,
    )

    # Celery configuration
    celery_app.conf.update(
        # Task settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,

        # Task routing
        task_routes={
            "src.sddi.distributed.worker.process_document_task": {"queue": "documents"},
            "src.sddi.distributed.worker.process_chunk_task": {"queue": "chunks"},
            "src.sddi.distributed.worker.extract_entities_task": {"queue": "extraction"},
            "src.sddi.distributed.worker.generate_embeddings_task": {"queue": "embeddings"},
            "src.sddi.distributed.worker.load_to_graph_task": {"queue": "loading"},
        },

        # Concurrency settings
        worker_concurrency=4,
        worker_prefetch_multiplier=2,

        # Task time limits
        task_time_limit=600,  # 10 minutes hard limit
        task_soft_time_limit=540,  # 9 minutes soft limit

        # Result settings
        result_expires=86400,  # 24 hours
        task_track_started=True,

        # Retry settings
        task_default_retry_delay=30,
        task_max_retries=3,

        # Rate limiting
        task_annotations={
            "src.sddi.distributed.worker.extract_entities_task": {
                "rate_limit": "10/m",  # 10 per minute (LLM rate limiting)
            },
            "src.sddi.distributed.worker.generate_embeddings_task": {
                "rate_limit": "20/m",  # 20 per minute
            },
        },
    )

    class BaseTask(Task):
        """Base task class with common functionality."""

        abstract = True
        autoretry_for = (Exception,)
        retry_kwargs = {"max_retries": 3}
        retry_backoff = True
        retry_backoff_max = 300  # 5 minutes
        retry_jitter = True

        def on_failure(self, exc, task_id, args, kwargs, einfo):
            """Handle task failure."""
            logger.error(
                "Task failed",
                task_id=task_id,
                task_name=self.name,
                error=str(exc),
            )

        def on_success(self, retval, task_id, args, kwargs):
            """Handle task success."""
            logger.info(
                "Task completed",
                task_id=task_id,
                task_name=self.name,
            )
else:
    celery_app = None

    class BaseTask:
        """Dummy base task when Celery is not available."""
        pass


# =============================================================================
# Helper Functions
# =============================================================================


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


def ensure_celery():
    """Ensure Celery is available."""
    if not CELERY_AVAILABLE:
        raise ImportError(
            "Celery is not installed. Install with: pip install celery[redis]"
        )


# =============================================================================
# Task Definitions
# =============================================================================

if CELERY_AVAILABLE:

    @celery_app.task(bind=True, base=BaseTask, name="process_document")
    def process_document_task(
        self,
        document_data: dict[str, Any],
        job_id: str,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process a single document through chunking.

        Args:
            document_data: Document content and metadata
            job_id: Parent job ID for tracking
            settings: Optional processing settings

        Returns:
            Dict with chunks and metadata
        """
        from src.sddi.pipeline import chunk_text

        logger.info(
            "Processing document",
            job_id=job_id,
            task_id=self.request.id,
            doc_source=document_data.get("source", "unknown"),
        )

        try:
            content = document_data.get("content", "")
            metadata = document_data.get("metadata", {})

            chunk_size = (settings or {}).get("chunk_size", 1000)
            chunk_overlap = (settings or {}).get("chunk_overlap", 200)

            # Chunk the document
            chunks = chunk_text(content, chunk_size, chunk_overlap)

            # Add metadata to chunks
            result_chunks = []
            for i, chunk in enumerate(chunks):
                result_chunks.append({
                    "id": f"{job_id}_chunk_{i}",
                    "text": chunk,
                    "source": document_data.get("source"),
                    "chunk_index": i,
                    "metadata": {
                        **metadata,
                        "parent_doc": document_data.get("source"),
                    },
                })

            return {
                "status": "success",
                "job_id": job_id,
                "task_id": self.request.id,
                "chunks_count": len(result_chunks),
                "chunks": result_chunks,
                "source": document_data.get("source"),
            }

        except Exception as e:
            logger.error(
                "Document processing failed",
                job_id=job_id,
                error=str(e),
            )
            raise

    @celery_app.task(bind=True, base=BaseTask, name="process_chunk")
    def process_chunk_task(
        self,
        chunk_data: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        """
        Process a single chunk (placeholder for future use).

        Args:
            chunk_data: Chunk content and metadata
            job_id: Parent job ID

        Returns:
            Processed chunk data
        """
        return {
            "status": "success",
            "job_id": job_id,
            "task_id": self.request.id,
            "chunk_id": chunk_data.get("id"),
        }

    @celery_app.task(bind=True, base=BaseTask, name="extract_entities")
    def extract_entities_task(
        self,
        chunks: list[dict[str, Any]],
        job_id: str,
        llm_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Extract entities from chunks using LLM.

        Args:
            chunks: List of chunk data
            job_id: Parent job ID
            llm_config: LLM configuration

        Returns:
            Dict with extracted entities
        """
        from src.sddi.extractors.entity_extractor import BatchEntityExtractor
        from src.sddi.state import TextChunk

        logger.info(
            "Extracting entities",
            job_id=job_id,
            task_id=self.request.id,
            chunks_count=len(chunks),
        )

        try:
            # Setup LLM
            llm = _get_llm(llm_config)

            # Create extractor
            extractor = BatchEntityExtractor(llm=llm, max_chunks_per_batch=5)

            # Convert to TextChunk objects
            text_chunks = [
                TextChunk(
                    id=c.get("id", f"chunk_{i}"),
                    text=c.get("text", ""),
                    source=c.get("source"),
                    metadata=c.get("metadata", {}),
                )
                for i, c in enumerate(chunks)
            ]

            # Extract entities
            entities = run_async(extractor.extract_batch(text_chunks))

            # Convert to serializable format
            entities_data = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "source_chunk_ids": e.source_chunk_ids,
                    "confidence": e.confidence,
                }
                for e in entities
            ]

            return {
                "status": "success",
                "job_id": job_id,
                "task_id": self.request.id,
                "entities_count": len(entities_data),
                "entities": entities_data,
            }

        except Exception as e:
            logger.error(
                "Entity extraction failed",
                job_id=job_id,
                error=str(e),
            )
            raise

    @celery_app.task(bind=True, base=BaseTask, name="extract_relations")
    def extract_relations_task(
        self,
        chunks: list[dict[str, Any]],
        entities: list[dict[str, Any]],
        job_id: str,
        llm_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Extract relations between entities.

        Args:
            chunks: Source chunks
            entities: Extracted entities
            job_id: Parent job ID
            llm_config: LLM configuration

        Returns:
            Dict with extracted relations
        """
        from src.sddi.extractors.relation_extractor import RelationExtractor
        from src.sddi.state import TextChunk, ExtractedEntity

        logger.info(
            "Extracting relations",
            job_id=job_id,
            task_id=self.request.id,
            entities_count=len(entities),
        )

        try:
            llm = _get_llm(llm_config)
            extractor = RelationExtractor(llm=llm)

            # Convert to objects
            text_chunks = [
                TextChunk(
                    id=c.get("id"),
                    text=c.get("text", ""),
                    source=c.get("source"),
                )
                for c in chunks
            ]

            entity_objects = [
                ExtractedEntity(
                    id=e.get("id"),
                    name=e.get("name"),
                    type=e.get("type"),
                    description=e.get("description"),
                    source_chunk_ids=e.get("source_chunk_ids", []),
                    confidence=e.get("confidence", 0.8),
                )
                for e in entities
            ]

            # Extract relations
            relations = run_async(
                extractor.extract_batch(text_chunks, entity_objects)
            )

            # Convert to serializable format
            relations_data = [
                {
                    "source_entity_id": r.source_entity_id,
                    "target_entity_id": r.target_entity_id,
                    "relation_type": r.relation_type,
                    "description": r.description,
                    "confidence": r.confidence,
                }
                for r in relations
            ]

            return {
                "status": "success",
                "job_id": job_id,
                "task_id": self.request.id,
                "relations_count": len(relations_data),
                "relations": relations_data,
            }

        except Exception as e:
            logger.error(
                "Relation extraction failed",
                job_id=job_id,
                error=str(e),
            )
            raise

    @celery_app.task(bind=True, base=BaseTask, name="generate_embeddings")
    def generate_embeddings_task(
        self,
        items: list[dict[str, Any]],
        item_type: str,  # "chunk" or "entity"
        job_id: str,
        embedding_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate embeddings for chunks or entities.

        Args:
            items: Items to embed (chunks or entities)
            item_type: Type of items ("chunk" or "entity")
            job_id: Parent job ID
            embedding_config: Embedding model configuration

        Returns:
            Dict with embeddings mapped by ID
        """
        from src.llm.cached_embeddings import create_cached_embeddings

        logger.info(
            "Generating embeddings",
            job_id=job_id,
            task_id=self.request.id,
            item_type=item_type,
            items_count=len(items),
        )

        try:
            # Get embedding model
            embeddings = _get_embeddings(embedding_config)
            cached_embeddings = create_cached_embeddings(embeddings)

            # Get texts to embed
            if item_type == "chunk":
                texts = [item.get("text", "") for item in items]
            else:
                texts = [
                    f"{item.get('name', '')} - {item.get('description', '')}"
                    for item in items
                ]

            # Generate embeddings
            embedding_vectors = run_async(cached_embeddings.embed_documents(texts))

            # Map embeddings to IDs
            embeddings_by_id = {}
            for item, vector in zip(items, embedding_vectors):
                embeddings_by_id[item.get("id")] = vector

            return {
                "status": "success",
                "job_id": job_id,
                "task_id": self.request.id,
                "embeddings_count": len(embeddings_by_id),
                "embeddings": embeddings_by_id,
            }

        except Exception as e:
            logger.error(
                "Embedding generation failed",
                job_id=job_id,
                error=str(e),
            )
            raise

    @celery_app.task(bind=True, base=BaseTask, name="load_to_graph")
    def load_to_graph_task(
        self,
        chunks: list[dict[str, Any]],
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        embeddings: dict[str, list[float]],
        job_id: str,
    ) -> dict[str, Any]:
        """
        Load processed data to Neo4j graph.

        Args:
            chunks: Chunk data
            entities: Entity data
            relations: Relation data
            embeddings: Embeddings by ID
            job_id: Parent job ID

        Returns:
            Load statistics
        """
        from src.sddi.loaders.neo4j_loader import Neo4jLoader, LoaderConfig
        from src.sddi.state import TextChunk, ExtractedEntity, ExtractedRelation

        logger.info(
            "Loading to graph",
            job_id=job_id,
            task_id=self.request.id,
            chunks=len(chunks),
            entities=len(entities),
            relations=len(relations),
        )

        try:
            # Create loader
            loader = Neo4jLoader(LoaderConfig(batch_size=500))

            # Convert to objects
            chunk_objects = [
                TextChunk(
                    id=c.get("id"),
                    text=c.get("text", ""),
                    source=c.get("source"),
                    embedding=embeddings.get(c.get("id")),
                )
                for c in chunks
            ]

            entity_objects = [
                ExtractedEntity(
                    id=e.get("id"),
                    name=e.get("name"),
                    type=e.get("type"),
                    description=e.get("description"),
                    embedding=embeddings.get(e.get("id")),
                )
                for e in entities
            ]

            relation_objects = [
                ExtractedRelation(
                    source_entity_id=r.get("source_entity_id"),
                    target_entity_id=r.get("target_entity_id"),
                    relation_type=r.get("relation_type"),
                    description=r.get("description"),
                )
                for r in relations
            ]

            # Load to graph
            result = run_async(
                loader.load_all(chunk_objects, entity_objects, relation_objects)
            )

            return {
                "status": "success",
                "job_id": job_id,
                "task_id": self.request.id,
                "loaded_chunks": result.get("chunks_loaded", 0),
                "loaded_entities": result.get("entities_loaded", 0),
                "loaded_relations": result.get("relations_loaded", 0),
            }

        except Exception as e:
            logger.error(
                "Graph loading failed",
                job_id=job_id,
                error=str(e),
            )
            raise

else:
    # Dummy tasks when Celery is not available
    def process_document_task(*args, **kwargs):
        raise ImportError("Celery is not installed")

    def process_chunk_task(*args, **kwargs):
        raise ImportError("Celery is not installed")

    def extract_entities_task(*args, **kwargs):
        raise ImportError("Celery is not installed")

    def extract_relations_task(*args, **kwargs):
        raise ImportError("Celery is not installed")

    def generate_embeddings_task(*args, **kwargs):
        raise ImportError("Celery is not installed")

    def load_to_graph_task(*args, **kwargs):
        raise ImportError("Celery is not installed")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_llm(config: dict[str, Any] | None = None):
    """Get LLM instance based on config."""
    from src.config.settings import get_settings

    settings = get_settings()
    config = config or {}

    provider = config.get("provider", "openai")
    model = config.get("model", settings.llm.reasoning_model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = config.get("api_key") or (
            settings.llm.anthropic_api_key.get_secret_value()
            if settings.llm.anthropic_api_key else None
        )
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=settings.llm.temperature,
            top_p=settings.llm.top_p,
        )
    else:
        from langchain_openai import ChatOpenAI
        api_key = config.get("api_key") or (
            settings.llm.openai_api_key.get_secret_value()
            if settings.llm.openai_api_key else None
        )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=settings.llm.temperature,
            seed=settings.llm.seed,
            top_p=settings.llm.top_p,
        )


def _get_embeddings(config: dict[str, Any] | None = None):
    """Get embedding model based on config."""
    from langchain_openai import OpenAIEmbeddings
    from src.config.settings import get_settings

    settings = get_settings()
    config = config or {}

    model = config.get("model", settings.llm.embedding_model)
    api_key = config.get("api_key") or (
        settings.llm.openai_api_key.get_secret_value()
        if settings.llm.openai_api_key else None
    )

    return OpenAIEmbeddings(model=model, api_key=api_key)


# =============================================================================
# Task Utilities
# =============================================================================


def get_task_status(task_id: str) -> dict[str, Any]:
    """Get status of a Celery task."""
    ensure_celery()

    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "result": result.result if result.ready() else None,
    }


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """Revoke/cancel a Celery task."""
    ensure_celery()

    celery_app.control.revoke(task_id, terminate=terminate)
    return True

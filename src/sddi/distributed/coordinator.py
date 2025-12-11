"""
Distributed Pipeline Coordinator.

Manages distributed document processing jobs:
- Job creation and tracking
- Task orchestration
- Progress monitoring
- Result aggregation
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.sddi.distributed.worker import (
    CELERY_AVAILABLE,
    process_document_task,
    extract_entities_task,
    extract_relations_task,
    generate_embeddings_task,
    load_to_graph_task,
    get_task_status,
)

logger = structlog.get_logger(__name__)


class JobStatus(str, Enum):
    """Distributed job status."""
    PENDING = "pending"
    CHUNKING = "chunking"
    EXTRACTING_ENTITIES = "extracting_entities"
    EXTRACTING_RELATIONS = "extracting_relations"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    LOADING = "loading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Information about a Celery task."""
    task_id: str
    task_type: str
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None


@dataclass
class DistributedJob:
    """A distributed processing job."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Input data
    documents: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    # Task tracking
    tasks: list[TaskInfo] = field(default_factory=list)
    current_stage: str = "pending"

    # Results
    chunks: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    # Statistics
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0

    # Error tracking
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_stage": self.current_stage,
            "progress": {
                "total_documents": self.total_documents,
                "processed_documents": self.processed_documents,
                "total_chunks": self.total_chunks,
                "total_entities": self.total_entities,
                "total_relations": self.total_relations,
            },
            "tasks": [
                {
                    "task_id": t.task_id,
                    "task_type": t.task_type,
                    "status": t.status,
                }
                for t in self.tasks
            ],
            "errors": self.errors,
        }


class DistributedPipelineCoordinator:
    """
    Coordinates distributed document processing.

    Orchestrates Celery tasks across multiple workers for:
    - Parallel document processing
    - Entity/relation extraction
    - Embedding generation
    - Graph loading

    Provides fallback to local processing when Celery is unavailable.
    """

    def __init__(self):
        self._jobs: dict[str, DistributedJob] = {}
        self._celery_available = CELERY_AVAILABLE

    @property
    def is_distributed(self) -> bool:
        """Check if distributed processing is available."""
        return self._celery_available

    async def create_job(
        self,
        documents: list[dict[str, Any]],
        settings: dict[str, Any] | None = None,
    ) -> DistributedJob:
        """
        Create a new distributed processing job.

        Args:
            documents: List of document data dicts
            settings: Processing settings

        Returns:
            Created job instance
        """
        job_id = str(uuid.uuid4())

        job = DistributedJob(
            job_id=job_id,
            documents=documents,
            settings=settings or {},
            total_documents=len(documents),
        )

        self._jobs[job_id] = job

        logger.info(
            "Created distributed job",
            job_id=job_id,
            documents_count=len(documents),
            distributed=self._celery_available,
        )

        return job

    async def start_job(self, job_id: str) -> DistributedJob:
        """
        Start processing a job.

        Args:
            job_id: Job ID to start

        Returns:
            Updated job instance
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job already started: {job_id}")

        job.started_at = datetime.utcnow()

        if self._celery_available:
            await self._start_distributed(job)
        else:
            await self._start_local(job)

        return job

    async def _start_distributed(self, job: DistributedJob) -> None:
        """Start distributed processing with Celery."""
        try:
            # Stage 1: Chunking
            job.status = JobStatus.CHUNKING
            job.current_stage = "chunking"

            chunking_tasks = []
            for doc in job.documents:
                task = process_document_task.delay(
                    document_data=doc,
                    job_id=job.job_id,
                    settings=job.settings,
                )
                chunking_tasks.append(task)
                job.tasks.append(TaskInfo(
                    task_id=task.id,
                    task_type="chunking",
                    started_at=datetime.utcnow(),
                ))

            # Wait for chunking to complete
            all_chunks = []
            for task in chunking_tasks:
                result = task.get(timeout=300)  # 5 minute timeout
                if result.get("status") == "success":
                    all_chunks.extend(result.get("chunks", []))
                    job.processed_documents += 1

            job.chunks = all_chunks
            job.total_chunks = len(all_chunks)

            # Stage 2: Entity Extraction
            job.status = JobStatus.EXTRACTING_ENTITIES
            job.current_stage = "entity_extraction"

            # Process chunks in batches
            batch_size = job.settings.get("extraction_batch_size", 20)
            entity_tasks = []

            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                task = extract_entities_task.delay(
                    chunks=batch,
                    job_id=job.job_id,
                    llm_config=job.settings.get("llm_config"),
                )
                entity_tasks.append(task)
                job.tasks.append(TaskInfo(
                    task_id=task.id,
                    task_type="entity_extraction",
                    started_at=datetime.utcnow(),
                ))

            # Collect entities
            all_entities = []
            for task in entity_tasks:
                result = task.get(timeout=600)  # 10 minute timeout
                if result.get("status") == "success":
                    all_entities.extend(result.get("entities", []))

            job.entities = all_entities
            job.total_entities = len(all_entities)

            # Stage 3: Relation Extraction
            job.status = JobStatus.EXTRACTING_RELATIONS
            job.current_stage = "relation_extraction"

            relation_task = extract_relations_task.delay(
                chunks=all_chunks,
                entities=all_entities,
                job_id=job.job_id,
                llm_config=job.settings.get("llm_config"),
            )
            job.tasks.append(TaskInfo(
                task_id=relation_task.id,
                task_type="relation_extraction",
                started_at=datetime.utcnow(),
            ))

            relation_result = relation_task.get(timeout=600)
            if relation_result.get("status") == "success":
                job.relations = relation_result.get("relations", [])
                job.total_relations = len(job.relations)

            # Stage 4: Embedding Generation
            job.status = JobStatus.GENERATING_EMBEDDINGS
            job.current_stage = "embedding_generation"

            # Generate embeddings for chunks
            chunk_embed_task = generate_embeddings_task.delay(
                items=all_chunks,
                item_type="chunk",
                job_id=job.job_id,
                embedding_config=job.settings.get("embedding_config"),
            )

            # Generate embeddings for entities
            entity_embed_task = generate_embeddings_task.delay(
                items=all_entities,
                item_type="entity",
                job_id=job.job_id,
                embedding_config=job.settings.get("embedding_config"),
            )

            job.tasks.extend([
                TaskInfo(task_id=chunk_embed_task.id, task_type="chunk_embeddings"),
                TaskInfo(task_id=entity_embed_task.id, task_type="entity_embeddings"),
            ])

            # Collect embeddings
            chunk_embed_result = chunk_embed_task.get(timeout=600)
            entity_embed_result = entity_embed_task.get(timeout=600)

            job.embeddings.update(chunk_embed_result.get("embeddings", {}))
            job.embeddings.update(entity_embed_result.get("embeddings", {}))

            # Stage 5: Load to Graph
            job.status = JobStatus.LOADING
            job.current_stage = "graph_loading"

            load_task = load_to_graph_task.delay(
                chunks=all_chunks,
                entities=all_entities,
                relations=job.relations,
                embeddings=job.embeddings,
                job_id=job.job_id,
            )
            job.tasks.append(TaskInfo(
                task_id=load_task.id,
                task_type="graph_loading",
                started_at=datetime.utcnow(),
            ))

            load_result = load_task.get(timeout=600)

            # Mark completed
            job.status = JobStatus.COMPLETED
            job.current_stage = "completed"
            job.completed_at = datetime.utcnow()

            logger.info(
                "Distributed job completed",
                job_id=job.job_id,
                chunks=job.total_chunks,
                entities=job.total_entities,
                relations=job.total_relations,
            )

        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(str(e))
            logger.error(
                "Distributed job failed",
                job_id=job.job_id,
                error=str(e),
            )
            raise

    async def _start_local(self, job: DistributedJob) -> None:
        """
        Fallback to local processing when Celery is unavailable.

        Uses the existing pipeline for processing.
        """
        from src.sddi.pipeline import SDDIPipeline

        logger.info(
            "Starting local processing (Celery unavailable)",
            job_id=job.job_id,
        )

        try:
            pipeline = SDDIPipeline()

            # Process each document
            for doc in job.documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                result = await pipeline.process(
                    raw_data=content,
                    source=doc.get("source", "unknown"),
                    metadata=metadata,
                )

                if result.get("status") == "success":
                    job.processed_documents += 1
                    job.total_chunks += result.get("chunks_count", 0)
                    job.total_entities += result.get("entities_count", 0)
                    job.total_relations += result.get("relations_count", 0)
                else:
                    job.errors.append(result.get("error", "Unknown error"))

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(str(e))
            raise

    async def get_job(self, job_id: str) -> DistributedJob | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get detailed job status with task states."""
        job = self._jobs.get(job_id)
        if not job:
            return {"error": f"Job not found: {job_id}"}

        status = job.to_dict()

        # Update task statuses from Celery
        if self._celery_available:
            for task_info in job.tasks:
                try:
                    task_status = get_task_status(task_info.task_id)
                    task_info.status = task_status.get("status", "unknown")
                except Exception:
                    pass

        return status

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        # Cancel Celery tasks
        if self._celery_available:
            from src.sddi.distributed.worker import revoke_task
            for task_info in job.tasks:
                try:
                    revoke_task(task_info.task_id, terminate=True)
                except Exception:
                    pass

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()

        return True

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List jobs with optional status filter."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return [j.to_dict() for j in jobs[:limit]]

    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Remove old completed jobs."""
        cutoff = datetime.utcnow()
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=max_age_hours)

        to_remove = []
        for job_id, job in self._jobs.items():
            if job.completed_at and job.completed_at < cutoff:
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]

        return len(to_remove)


# =============================================================================
# Global Coordinator Instance
# =============================================================================

_coordinator: DistributedPipelineCoordinator | None = None


def get_distributed_coordinator() -> DistributedPipelineCoordinator:
    """Get the global coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = DistributedPipelineCoordinator()
    return _coordinator

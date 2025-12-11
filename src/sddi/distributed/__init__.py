"""
Distributed Pipeline Module.

Provides horizontal scaling for document ingestion using Celery.
"""

from src.sddi.distributed.worker import (
    celery_app,
    process_document_task,
    process_chunk_task,
    extract_entities_task,
    generate_embeddings_task,
)
from src.sddi.distributed.coordinator import (
    DistributedPipelineCoordinator,
    JobStatus,
    DistributedJob,
)

__all__ = [
    "celery_app",
    "process_document_task",
    "process_chunk_task",
    "extract_entities_task",
    "generate_embeddings_task",
    "DistributedPipelineCoordinator",
    "JobStatus",
    "DistributedJob",
]

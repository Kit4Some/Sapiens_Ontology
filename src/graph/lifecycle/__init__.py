"""
Graph Lifecycle Management.

Provides data lifecycle features:
- Soft-delete mechanism for nodes and relationships
- Timestamp tracking (created_at, updated_at, deleted_at)
- Data retention policies
- Audit trail support
"""

from src.graph.lifecycle.soft_delete import (
    SoftDeleteManager,
    SoftDeleteConfig,
    SoftDeleteResult,
    DeletionPolicy,
)
from src.graph.lifecycle.timestamps import (
    TimestampManager,
    ensure_timestamps,
)

__all__ = [
    # Soft Delete
    "SoftDeleteManager",
    "SoftDeleteConfig",
    "SoftDeleteResult",
    "DeletionPolicy",
    # Timestamps
    "TimestampManager",
    "ensure_timestamps",
]

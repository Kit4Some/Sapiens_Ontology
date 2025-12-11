"""
Dead-Letter Queue (DLQ) for Failed Documents.

Stores failed documents for later inspection and retry:
- Preserves original document and error context
- Supports multiple backends (in-memory, file, database)
- Enables manual or automatic retry
- Provides cleanup and expiration
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
import uuid

import structlog

logger = structlog.get_logger(__name__)


class DLQStatus(str, Enum):
    """Status of a DLQ entry."""

    PENDING = "pending"        # Awaiting retry
    RETRYING = "retrying"      # Currently being retried
    RESOLVED = "resolved"      # Successfully processed on retry
    ABANDONED = "abandoned"    # Exceeded max retries, given up
    EXPIRED = "expired"        # TTL exceeded


@dataclass
class DLQEntry:
    """Entry in the Dead-Letter Queue."""

    entry_id: str
    document_id: str
    document_content: str
    document_metadata: dict[str, Any]

    # Error information
    error_type: str
    error_message: str
    error_traceback: str | None = None
    failed_step: str = "unknown"

    # Pipeline context
    pipeline_id: str = ""
    chunk_index: int | None = None

    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    status: DLQStatus = DLQStatus.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_retry_at: datetime | None = None
    next_retry_at: datetime | None = None
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "document_id": self.document_id,
            "document_content": self.document_content,
            "document_metadata": self.document_metadata,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "failed_step": self.failed_step,
            "pipeline_id": self.pipeline_id,
            "chunk_index": self.chunk_index,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DLQEntry":
        return cls(
            entry_id=data["entry_id"],
            document_id=data["document_id"],
            document_content=data["document_content"],
            document_metadata=data.get("document_metadata", {}),
            error_type=data["error_type"],
            error_message=data["error_message"],
            error_traceback=data.get("error_traceback"),
            failed_step=data.get("failed_step", "unknown"),
            pipeline_id=data.get("pipeline_id", ""),
            chunk_index=data.get("chunk_index"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            status=DLQStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            last_retry_at=datetime.fromisoformat(data["last_retry_at"]) if data.get("last_retry_at") else None,
            next_retry_at=datetime.fromisoformat(data["next_retry_at"]) if data.get("next_retry_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
        )

    def can_retry(self) -> bool:
        """Check if this entry can be retried."""
        if self.status in (DLQStatus.RESOLVED, DLQStatus.ABANDONED, DLQStatus.EXPIRED):
            return False
        if self.retry_count >= self.max_retries:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def is_ready_for_retry(self) -> bool:
        """Check if enough time has passed for next retry."""
        if not self.can_retry():
            return False
        if self.next_retry_at and datetime.utcnow() < self.next_retry_at:
            return False
        return True


class DeadLetterQueue(ABC):
    """Abstract base class for Dead-Letter Queue implementations."""

    @abstractmethod
    async def push(self, entry: DLQEntry) -> str:
        """Add an entry to the DLQ. Returns entry_id."""
        pass

    @abstractmethod
    async def pop(self) -> DLQEntry | None:
        """Get and remove the next entry ready for retry."""
        pass

    @abstractmethod
    async def peek(self, count: int = 10) -> list[DLQEntry]:
        """View entries without removing them."""
        pass

    @abstractmethod
    async def get(self, entry_id: str) -> DLQEntry | None:
        """Get a specific entry by ID."""
        pass

    @abstractmethod
    async def update(self, entry: DLQEntry) -> bool:
        """Update an existing entry."""
        pass

    @abstractmethod
    async def remove(self, entry_id: str) -> bool:
        """Remove an entry from the queue."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get number of entries in the queue."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all entries. Returns count deleted."""
        pass

    async def push_error(
        self,
        document_id: str,
        document_content: str,
        error: Exception,
        failed_step: str = "unknown",
        pipeline_id: str = "",
        document_metadata: dict[str, Any] | None = None,
        max_retries: int = 3,
        ttl_hours: int = 72,
    ) -> str:
        """
        Convenience method to push a failed document.

        Args:
            document_id: ID of the failed document
            document_content: Content of the document
            error: The exception that caused the failure
            failed_step: Pipeline step where failure occurred
            pipeline_id: ID of the pipeline run
            document_metadata: Original document metadata
            max_retries: Maximum retry attempts
            ttl_hours: Hours before entry expires

        Returns:
            Entry ID
        """
        import traceback

        entry = DLQEntry(
            entry_id=str(uuid.uuid4()),
            document_id=document_id,
            document_content=document_content,
            document_metadata=document_metadata or {},
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            failed_step=failed_step,
            pipeline_id=pipeline_id,
            max_retries=max_retries,
            expires_at=datetime.utcnow() + timedelta(hours=ttl_hours),
        )

        entry_id = await self.push(entry)

        logger.warning(
            "Document added to DLQ",
            entry_id=entry_id,
            document_id=document_id,
            error_type=entry.error_type,
            failed_step=failed_step,
        )

        return entry_id

    async def mark_retrying(self, entry_id: str) -> bool:
        """Mark an entry as being retried."""
        entry = await self.get(entry_id)
        if not entry:
            return False

        entry.status = DLQStatus.RETRYING
        entry.last_retry_at = datetime.utcnow()
        entry.updated_at = datetime.utcnow()
        return await self.update(entry)

    async def mark_resolved(self, entry_id: str) -> bool:
        """Mark an entry as successfully resolved."""
        entry = await self.get(entry_id)
        if not entry:
            return False

        entry.status = DLQStatus.RESOLVED
        entry.updated_at = datetime.utcnow()

        logger.info(
            "DLQ entry resolved",
            entry_id=entry_id,
            document_id=entry.document_id,
            retry_count=entry.retry_count,
        )

        return await self.update(entry)

    async def mark_failed_retry(
        self,
        entry_id: str,
        error: Exception,
        backoff_minutes: int = 5,
    ) -> bool:
        """Mark a retry attempt as failed."""
        entry = await self.get(entry_id)
        if not entry:
            return False

        entry.retry_count += 1
        entry.error_message = str(error)
        entry.updated_at = datetime.utcnow()

        if entry.retry_count >= entry.max_retries:
            entry.status = DLQStatus.ABANDONED
            logger.error(
                "DLQ entry abandoned (max retries exceeded)",
                entry_id=entry_id,
                document_id=entry.document_id,
                retry_count=entry.retry_count,
            )
        else:
            entry.status = DLQStatus.PENDING
            # Exponential backoff for next retry
            backoff = backoff_minutes * (2 ** entry.retry_count)
            entry.next_retry_at = datetime.utcnow() + timedelta(minutes=backoff)
            logger.warning(
                "DLQ retry failed, scheduled for later",
                entry_id=entry_id,
                document_id=entry.document_id,
                retry_count=entry.retry_count,
                next_retry_at=entry.next_retry_at.isoformat(),
            )

        return await self.update(entry)

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the DLQ."""
        entries = await self.peek(count=10000)  # Get all

        stats = {
            "total": len(entries),
            "by_status": {},
            "by_step": {},
            "by_error_type": {},
            "ready_for_retry": 0,
            "oldest_entry": None,
            "newest_entry": None,
        }

        for entry in entries:
            # By status
            status = entry.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # By step
            step = entry.failed_step
            stats["by_step"][step] = stats["by_step"].get(step, 0) + 1

            # By error type
            error_type = entry.error_type
            stats["by_error_type"][error_type] = stats["by_error_type"].get(error_type, 0) + 1

            # Ready for retry
            if entry.is_ready_for_retry():
                stats["ready_for_retry"] += 1

            # Oldest/newest
            if stats["oldest_entry"] is None or entry.created_at < stats["oldest_entry"]:
                stats["oldest_entry"] = entry.created_at
            if stats["newest_entry"] is None or entry.created_at > stats["newest_entry"]:
                stats["newest_entry"] = entry.created_at

        if stats["oldest_entry"]:
            stats["oldest_entry"] = stats["oldest_entry"].isoformat()
        if stats["newest_entry"]:
            stats["newest_entry"] = stats["newest_entry"].isoformat()

        return stats


class InMemoryDLQ(DeadLetterQueue):
    """In-memory Dead-Letter Queue implementation."""

    def __init__(self):
        self._entries: dict[str, DLQEntry] = {}

    async def push(self, entry: DLQEntry) -> str:
        self._entries[entry.entry_id] = entry
        return entry.entry_id

    async def pop(self) -> DLQEntry | None:
        for entry_id, entry in list(self._entries.items()):
            if entry.is_ready_for_retry():
                del self._entries[entry_id]
                return entry
        return None

    async def peek(self, count: int = 10) -> list[DLQEntry]:
        entries = list(self._entries.values())
        entries.sort(key=lambda e: e.created_at)
        return entries[:count]

    async def get(self, entry_id: str) -> DLQEntry | None:
        return self._entries.get(entry_id)

    async def update(self, entry: DLQEntry) -> bool:
        if entry.entry_id in self._entries:
            self._entries[entry.entry_id] = entry
            return True
        return False

    async def remove(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    async def size(self) -> int:
        return len(self._entries)

    async def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count


class FileDLQ(DeadLetterQueue):
    """
    File-based Dead-Letter Queue implementation.

    Stores entries as JSON files in a directory.
    Suitable for persistence across restarts.
    """

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, entry_id: str) -> Path:
        return self._dir / f"{entry_id}.json"

    async def push(self, entry: DLQEntry) -> str:
        path = self._entry_path(entry.entry_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
        return entry.entry_id

    async def pop(self) -> DLQEntry | None:
        for path in sorted(self._dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = DLQEntry.from_dict(data)
                if entry.is_ready_for_retry():
                    path.unlink()
                    return entry
            except Exception as e:
                logger.warning(f"Failed to read DLQ entry: {path}", error=str(e))
        return None

    async def peek(self, count: int = 10) -> list[DLQEntry]:
        entries = []
        for path in sorted(self._dir.glob("*.json"))[:count]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(DLQEntry.from_dict(data))
            except Exception:
                pass
        return entries

    async def get(self, entry_id: str) -> DLQEntry | None:
        path = self._entry_path(entry_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DLQEntry.from_dict(data)
        except Exception:
            return None

    async def update(self, entry: DLQEntry) -> bool:
        path = self._entry_path(entry.entry_id)
        if not path.exists():
            return False
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    async def remove(self, entry_id: str) -> bool:
        path = self._entry_path(entry_id)
        if path.exists():
            path.unlink()
            return True
        return False

    async def size(self) -> int:
        return len(list(self._dir.glob("*.json")))

    async def clear(self) -> int:
        count = 0
        for path in self._dir.glob("*.json"):
            path.unlink()
            count += 1
        return count

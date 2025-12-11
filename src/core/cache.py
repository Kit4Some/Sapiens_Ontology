"""
Multi-Layer Caching System.

Provides tiered caching with:
- L1: In-memory LRU cache (fast, limited size)
- L2: Redis cache (distributed, larger capacity)
- TTL-based expiration
- Cache statistics and monitoring
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CacheLevel(str, Enum):
    """Cache storage levels."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with metadata."""

    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_requests: int = 0
    bytes_stored: int = 0
    items_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_requests": self.total_requests,
            "hit_rate_percent": round(self.hit_rate, 2),
            "bytes_stored": self.bytes_stored,
            "items_count": self.items_count,
        }


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all cached values. Returns count of cleared items."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class LRUCache(CacheBackend):
    """
    Thread-safe LRU (Least Recently Used) in-memory cache.

    Features:
    - O(1) get/set operations
    - Automatic eviction of least recently used items
    - TTL-based expiration
    - Size-based limits (items and bytes)
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_bytes: int = 100 * 1024 * 1024,  # 100 MB
        default_ttl: int | None = 3600,  # 1 hour
    ):
        self._max_size = max_size
        self._max_bytes = max_bytes
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
        self._current_bytes = 0

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate

    def _evict_if_needed(self) -> None:
        """Evict oldest items if cache exceeds limits."""
        # Evict by count
        while len(self._cache) >= self._max_size:
            key, entry = self._cache.popitem(last=False)
            self._current_bytes -= entry.size_bytes
            self._stats.evictions += 1
            logger.debug("Cache eviction (count)", key=key[:50])

        # Evict by bytes
        while self._current_bytes > self._max_bytes and self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_bytes -= entry.size_bytes
            self._stats.evictions += 1
            logger.debug("Cache eviction (bytes)", key=key[:50])

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._current_bytes -= entry.size_bytes
            self._stats.expirations += 1

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        async with self._lock:
            self._stats.total_requests += 1

            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired():
                self._cache.pop(key)
                self._current_bytes -= entry.size_bytes
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1

            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        async with self._lock:
            # Calculate expiration
            expires_at = None
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if effective_ttl is not None:
                expires_at = time.time() + effective_ttl

            # Estimate size
            size_bytes = self._estimate_size(value)

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_bytes -= old_entry.size_bytes

            # Evict if needed
            self._evict_if_needed()

            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )
            self._cache[key] = entry
            self._current_bytes += size_bytes

            self._stats.items_count = len(self._cache)
            self._stats.bytes_stored = self._current_bytes

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._current_bytes -= entry.size_bytes
                self._stats.items_count = len(self._cache)
                self._stats.bytes_stored = self._current_bytes
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired():
                self._cache.pop(key)
                self._current_bytes -= entry.size_bytes
                self._stats.expirations += 1
                return False
            return True

    async def clear(self) -> int:
        """Clear all cached values."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_bytes = 0
            self._stats.items_count = 0
            self._stats.bytes_stored = 0
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class RedisCache(CacheBackend):
    """
    Redis-based distributed cache backend.

    Features:
    - Distributed caching across multiple nodes
    - Automatic serialization/deserialization
    - Connection pooling
    - Graceful fallback on connection failure
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "ontology:",
        default_ttl: int = 3600,
        max_connections: int = 10,
    ):
        self._url = url
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._max_connections = max_connections
        self._client = None
        self._stats = CacheStats()
        self._connected = False

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(
                    self._url,
                    max_connections=self._max_connections,
                    decode_responses=False,
                )
                # Test connection
                await self._client.ping()
                self._connected = True
                logger.info("Redis cache connected", url=self._url[:30])
            except ImportError:
                logger.warning("redis package not installed")
                self._connected = False
                return None
            except Exception as e:
                logger.warning("Redis connection failed", error=str(e))
                self._connected = False
                return None
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        self._stats.total_requests += 1

        client = await self._get_client()
        if client is None:
            self._stats.misses += 1
            return None

        try:
            data = await client.get(self._make_key(key))
            if data is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            return pickle.loads(data)  # nosec B301 - data from internal Redis cache only
        except Exception as e:
            logger.error("Redis get failed", key=key[:50], error=str(e))
            self._stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in Redis."""
        client = await self._get_client()
        if client is None:
            return False

        try:
            data = pickle.dumps(value)
            effective_ttl = ttl if ttl is not None else self._default_ttl

            await client.set(
                self._make_key(key),
                data,
                ex=effective_ttl,
            )
            return True
        except Exception as e:
            logger.error("Redis set failed", key=key[:50], error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        client = await self._get_client()
        if client is None:
            return False

        try:
            result = await client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error("Redis delete failed", key=key[:50], error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        client = await self._get_client()
        if client is None:
            return False

        try:
            return await client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error("Redis exists failed", key=key[:50], error=str(e))
            return False

    async def clear(self) -> int:
        """Clear all cached values with prefix."""
        client = await self._get_client()
        if client is None:
            return 0

        try:
            pattern = f"{self._prefix}*"
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.error("Redis clear failed", error=str(e))
            return 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected


class TieredCache:
    """
    Two-tier caching system with L1 (memory) and L2 (Redis).

    Read path: L1 -> L2 -> Source
    Write path: Source -> L2 -> L1

    Features:
    - Automatic promotion from L2 to L1 on access
    - Write-through to both tiers
    - Graceful degradation if L2 unavailable
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_max_bytes: int = 100 * 1024 * 1024,
        l1_ttl: int = 300,  # 5 minutes
        redis_url: str | None = None,
        redis_ttl: int = 3600,  # 1 hour
        namespace: str = "default",
    ):
        self._namespace = namespace
        self._l1 = LRUCache(
            max_size=l1_max_size,
            max_bytes=l1_max_bytes,
            default_ttl=l1_ttl,
        )

        self._l2: RedisCache | None = None
        if redis_url:
            self._l2 = RedisCache(
                url=redis_url,
                prefix=f"ontology:{namespace}:",
                default_ttl=redis_ttl,
            )

    async def get(self, key: str) -> Any | None:
        """Get value from cache, checking L1 then L2."""
        # Try L1 first
        value = await self._l1.get(key)
        if value is not None:
            return value

        # Try L2 if available
        if self._l2:
            value = await self._l2.get(key)
            if value is not None:
                # Promote to L1
                await self._l1.set(key, value)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        l1_ttl: int | None = None,
        l2_ttl: int | None = None,
    ) -> bool:
        """Set value in both cache tiers."""
        # Write to L1
        l1_success = await self._l1.set(key, value, l1_ttl)

        # Write to L2 if available
        l2_success = True
        if self._l2:
            l2_success = await self._l2.set(key, value, l2_ttl)

        return l1_success and l2_success

    async def delete(self, key: str) -> bool:
        """Delete from both tiers."""
        l1_deleted = await self._l1.delete(key)
        l2_deleted = True
        if self._l2:
            l2_deleted = await self._l2.delete(key)
        return l1_deleted or l2_deleted

    async def clear(self) -> dict[str, int]:
        """Clear both cache tiers."""
        l1_count = await self._l1.clear()
        l2_count = 0
        if self._l2:
            l2_count = await self._l2.clear()
        return {"l1_cleared": l1_count, "l2_cleared": l2_count}

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics."""
        stats = {
            "namespace": self._namespace,
            "l1": self._l1.get_stats().to_dict(),
        }
        if self._l2:
            stats["l2"] = self._l2.get_stats().to_dict()
            stats["l2_connected"] = self._l2.is_connected
        return stats


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a deterministic cache key from arguments.

    Uses SHA256 hash for consistent, collision-resistant keys.
    """
    key_parts = []

    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True, default=str))
        else:
            key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        if isinstance(v, (dict, list)):
            key_parts.append(f"{k}={json.dumps(v, sort_keys=True, default=str)}")
        else:
            key_parts.append(f"{k}={v}")

    combined = "|".join(key_parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


# =============================================================================
# Specialized Caches
# =============================================================================


class QueryResultCache:
    """
    Cache for query results.

    Keys are based on:
    - Query text (normalized)
    - Max iterations
    - Context hash
    """

    def __init__(
        self,
        cache: TieredCache | None = None,
        default_ttl: int = 1800,  # 30 minutes
    ):
        self._cache = cache or TieredCache(
            l1_max_size=500,
            l1_max_bytes=50 * 1024 * 1024,
            l1_ttl=300,
            namespace="query_results",
        )
        self._default_ttl = default_ttl

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        return " ".join(query.lower().split())

    def _make_key(
        self,
        query: str,
        max_iterations: int = 5,
        context: dict | None = None,
    ) -> str:
        """Generate cache key for query."""
        normalized = self._normalize_query(query)
        return generate_cache_key(
            "query",
            normalized,
            max_iterations=max_iterations,
            context=context or {},
        )

    async def get(
        self,
        query: str,
        max_iterations: int = 5,
        context: dict | None = None,
    ) -> dict | None:
        """Get cached query result."""
        key = self._make_key(query, max_iterations, context)
        result = await self._cache.get(key)

        if result:
            logger.debug("Query cache hit", query=query[:50])

        return result

    async def set(
        self,
        query: str,
        result: dict,
        max_iterations: int = 5,
        context: dict | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Cache query result."""
        key = self._make_key(query, max_iterations, context)
        effective_ttl = ttl or self._default_ttl

        success = await self._cache.set(key, result, l1_ttl=300, l2_ttl=effective_ttl)

        if success:
            logger.debug("Query result cached", query=query[:50])

        return success

    async def invalidate(
        self,
        query: str,
        max_iterations: int = 5,
        context: dict | None = None,
    ) -> bool:
        """Invalidate cached query result."""
        key = self._make_key(query, max_iterations, context)
        return await self._cache.delete(key)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class EmbeddingCache:
    """
    Cache for text embeddings.

    Significantly reduces API calls for repeated texts.
    Uses content-based hashing for deduplication.
    """

    def __init__(
        self,
        cache: TieredCache | None = None,
        default_ttl: int = 86400,  # 24 hours (embeddings are stable)
    ):
        self._cache = cache or TieredCache(
            l1_max_size=10000,
            l1_max_bytes=200 * 1024 * 1024,  # 200 MB
            l1_ttl=3600,  # 1 hour in memory
            namespace="embeddings",
        )
        self._default_ttl = default_ttl

    def _make_key(self, text: str, model: str = "default") -> str:
        """Generate cache key for embedding."""
        # Normalize text
        normalized = " ".join(text.split())
        return generate_cache_key("embedding", normalized, model=model)

    async def get(self, text: str, model: str = "default") -> list[float] | None:
        """Get cached embedding."""
        key = self._make_key(text, model)
        return await self._cache.get(key)

    async def get_many(
        self,
        texts: list[str],
        model: str = "default",
    ) -> tuple[dict[int, list[float]], list[int]]:
        """
        Get multiple embeddings from cache.

        Returns:
            Tuple of (cached embeddings by index, missing indices)
        """
        cached = {}
        missing = []

        for i, text in enumerate(texts):
            embedding = await self.get(text, model)
            if embedding is not None:
                cached[i] = embedding
            else:
                missing.append(i)

        return cached, missing

    async def set(
        self,
        text: str,
        embedding: list[float],
        model: str = "default",
        ttl: int | None = None,
    ) -> bool:
        """Cache embedding."""
        key = self._make_key(text, model)
        effective_ttl = ttl or self._default_ttl
        return await self._cache.set(key, embedding, l1_ttl=3600, l2_ttl=effective_ttl)

    async def set_many(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model: str = "default",
        ttl: int | None = None,
    ) -> int:
        """Cache multiple embeddings. Returns count of successfully cached."""
        success_count = 0
        for text, embedding in zip(texts, embeddings):
            if await self.set(text, embedding, model, ttl):
                success_count += 1
        return success_count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class SubgraphCache:
    """
    Cache for frequently accessed subgraphs.

    Caches:
    - Entity neighborhoods (1-hop, 2-hop)
    - Community subgraphs
    - Path queries
    """

    def __init__(
        self,
        cache: TieredCache | None = None,
        default_ttl: int = 600,  # 10 minutes (graphs change more often)
    ):
        self._cache = cache or TieredCache(
            l1_max_size=200,
            l1_max_bytes=50 * 1024 * 1024,
            l1_ttl=300,
            namespace="subgraphs",
        )
        self._default_ttl = default_ttl

    async def get_neighborhood(
        self,
        entity_id: str,
        hops: int = 1,
    ) -> dict | None:
        """Get cached entity neighborhood."""
        key = generate_cache_key("neighborhood", entity_id, hops=hops)
        return await self._cache.get(key)

    async def set_neighborhood(
        self,
        entity_id: str,
        subgraph: dict,
        hops: int = 1,
        ttl: int | None = None,
    ) -> bool:
        """Cache entity neighborhood."""
        key = generate_cache_key("neighborhood", entity_id, hops=hops)
        return await self._cache.set(key, subgraph, l2_ttl=ttl or self._default_ttl)

    async def get_community(self, community_id: str) -> dict | None:
        """Get cached community subgraph."""
        key = generate_cache_key("community", community_id)
        return await self._cache.get(key)

    async def set_community(
        self,
        community_id: str,
        subgraph: dict,
        ttl: int | None = None,
    ) -> bool:
        """Cache community subgraph."""
        key = generate_cache_key("community", community_id)
        return await self._cache.set(key, subgraph, l2_ttl=ttl or self._default_ttl)

    async def invalidate_entity(self, entity_id: str) -> None:
        """Invalidate all caches related to an entity."""
        for hops in [1, 2, 3]:
            key = generate_cache_key("neighborhood", entity_id, hops=hops)
            await self._cache.delete(key)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


# =============================================================================
# Global Cache Instances
# =============================================================================

_query_cache: QueryResultCache | None = None
_embedding_cache: EmbeddingCache | None = None
_subgraph_cache: SubgraphCache | None = None


def init_caches(
    redis_url: str | None = None,
    query_cache_size: int = 500,
    embedding_cache_size: int = 10000,
    subgraph_cache_size: int = 200,
) -> dict[str, Any]:
    """
    Initialize global cache instances.

    Args:
        redis_url: Optional Redis URL for L2 cache
        query_cache_size: Max items in query cache
        embedding_cache_size: Max items in embedding cache
        subgraph_cache_size: Max items in subgraph cache

    Returns:
        Dict with cache references
    """
    global _query_cache, _embedding_cache, _subgraph_cache

    _query_cache = QueryResultCache(
        TieredCache(
            l1_max_size=query_cache_size,
            redis_url=redis_url,
            namespace="query_results",
        )
    )

    _embedding_cache = EmbeddingCache(
        TieredCache(
            l1_max_size=embedding_cache_size,
            redis_url=redis_url,
            namespace="embeddings",
        )
    )

    _subgraph_cache = SubgraphCache(
        TieredCache(
            l1_max_size=subgraph_cache_size,
            redis_url=redis_url,
            namespace="subgraphs",
        )
    )

    logger.info(
        "Caches initialized",
        redis_enabled=redis_url is not None,
        query_cache_size=query_cache_size,
        embedding_cache_size=embedding_cache_size,
        subgraph_cache_size=subgraph_cache_size,
    )

    return {
        "query_cache": _query_cache,
        "embedding_cache": _embedding_cache,
        "subgraph_cache": _subgraph_cache,
    }


def get_query_cache() -> QueryResultCache:
    """Get the global query cache."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryResultCache()
    return _query_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_subgraph_cache() -> SubgraphCache:
    """Get the global subgraph cache."""
    global _subgraph_cache
    if _subgraph_cache is None:
        _subgraph_cache = SubgraphCache()
    return _subgraph_cache


def get_all_cache_stats() -> dict[str, Any]:
    """Get statistics from all caches."""
    return {
        "query_cache": get_query_cache().get_stats(),
        "embedding_cache": get_embedding_cache().get_stats(),
        "subgraph_cache": get_subgraph_cache().get_stats(),
    }


async def clear_all_caches() -> dict[str, Any]:
    """Clear all caches."""
    return {
        "query_cache": await get_query_cache()._cache.clear(),
        "embedding_cache": await get_embedding_cache()._cache.clear(),
        "subgraph_cache": await get_subgraph_cache()._cache.clear(),
    }

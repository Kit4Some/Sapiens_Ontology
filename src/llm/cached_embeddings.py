"""
Cached Embeddings Wrapper.

Provides a caching layer around embedding models to reduce API calls.
"""

from typing import Any

import structlog

from src.core.cache import get_embedding_cache

logger = structlog.get_logger(__name__)


class CachedEmbeddings:
    """
    Wrapper around LangChain embeddings with caching.

    Features:
    - Caches embeddings by content hash
    - Batch processing with partial cache hits
    - Statistics tracking
    - Fallback to original embeddings on cache miss
    """

    def __init__(self, embeddings: Any, model_name: str = "default"):
        """
        Initialize cached embeddings.

        Args:
            embeddings: The underlying embedding model (e.g., OpenAIEmbeddings)
            model_name: Model identifier for cache namespacing
        """
        self._embeddings = embeddings
        self._model_name = model_name
        self._cache = get_embedding_cache()
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query with caching.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        self._total_requests += 1

        # Try cache first
        cached = await self._cache.get(text, self._model_name)
        if cached is not None:
            self._cache_hits += 1
            return cached

        # Cache miss - generate embedding
        self._cache_misses += 1

        if hasattr(self._embeddings, "aembed_query"):
            embedding = await self._embeddings.aembed_query(text)
        else:
            embedding = self._embeddings.embed_query(text)

        # Cache the result
        await self._cache.set(text, embedding, self._model_name)

        return embedding

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._total_requests += len(texts)

        # Check cache for all texts
        cached_embeddings, missing_indices = await self._cache.get_many(
            texts, self._model_name
        )

        self._cache_hits += len(cached_embeddings)
        self._cache_misses += len(missing_indices)

        # If all cached, return immediately
        if not missing_indices:
            logger.debug(
                "All embeddings cached",
                count=len(texts),
            )
            return [cached_embeddings[i] for i in range(len(texts))]

        # Get missing texts
        missing_texts = [texts[i] for i in missing_indices]

        logger.debug(
            "Embedding cache partial hit",
            total=len(texts),
            cached=len(cached_embeddings),
            missing=len(missing_texts),
        )

        # Generate embeddings for missing texts
        if hasattr(self._embeddings, "aembed_documents"):
            new_embeddings = await self._embeddings.aembed_documents(missing_texts)
        else:
            new_embeddings = self._embeddings.embed_documents(missing_texts)

        # Cache new embeddings
        await self._cache.set_many(missing_texts, new_embeddings, self._model_name)

        # Combine cached and new embeddings
        all_embeddings = {}

        # Add cached embeddings
        for idx, embedding in cached_embeddings.items():
            all_embeddings[idx] = embedding

        # Add new embeddings
        for i, idx in enumerate(missing_indices):
            all_embeddings[idx] = new_embeddings[i]

        # Return in original order
        return [all_embeddings[i] for i in range(len(texts))]

    def embed_query_sync(self, text: str) -> list[float]:
        """Synchronous embed_query for compatibility."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.embed_query(text))

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embed_documents for compatibility."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.embed_documents(texts))

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self._total_requests == 0:
            return 0.0
        return (self._cache_hits / self._total_requests) * 100

    def get_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(self.cache_hit_rate, 2),
            "model_name": self._model_name,
            "cache_stats": self._cache.get_stats(),
        }

    # Delegate other attributes to underlying embeddings
    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)


def create_cached_embeddings(
    embeddings: Any,
    model_name: str | None = None,
) -> CachedEmbeddings:
    """
    Create a cached wrapper around embeddings.

    Args:
        embeddings: The underlying embedding model
        model_name: Optional model name override

    Returns:
        CachedEmbeddings wrapper
    """
    # Try to get model name from embeddings
    if model_name is None:
        model_name = getattr(embeddings, "model", "default")

    return CachedEmbeddings(embeddings, model_name)

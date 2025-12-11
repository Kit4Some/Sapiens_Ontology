"""
Versioned Embedding System.

Provides embedding version management for:
- Tracking which model/version generated each embedding
- Detecting version mismatches
- Supporting migration when switching models
- Backward compatibility checks

Key concepts:
- EmbeddingVersion: Identifies the model and version
- VersionedEmbedder: Wraps embedding models with versioning
- EmbeddingMigrator: Handles re-embedding when models change
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__name__)


class EmbeddingModel(str, Enum):
    """Known embedding models."""

    # OpenAI
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"

    # Anthropic (via voyage or similar)
    VOYAGE_2 = "voyage-2"
    VOYAGE_LARGE_2 = "voyage-large-2"

    # Open source
    BGE_SMALL = "bge-small-en-v1.5"
    BGE_BASE = "bge-base-en-v1.5"
    BGE_LARGE = "bge-large-en-v1.5"
    E5_SMALL = "e5-small-v2"
    E5_BASE = "e5-base-v2"
    E5_LARGE = "e5-large-v2"

    # Local
    NOMIC_EMBED = "nomic-embed-text"
    MXBAI_EMBED = "mxbai-embed-large"

    # Custom/unknown
    CUSTOM = "custom"


@dataclass
class EmbeddingVersion:
    """
    Identifies an embedding model version.

    Used to track which model generated each embedding
    and detect version mismatches.
    """

    model: str
    dimensions: int
    version: str = "1.0.0"
    provider: str = "unknown"
    normalized: bool = True  # Whether embeddings are L2 normalized

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version_id(self) -> str:
        """Unique identifier for this version."""
        key = f"{self.provider}:{self.model}:{self.dimensions}:{self.version}"
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]

    def is_compatible_with(self, other: "EmbeddingVersion") -> bool:
        """Check if embeddings from another version are compatible."""
        # Must have same dimensions
        if self.dimensions != other.dimensions:
            return False

        # Same model family is compatible
        if self.model == other.model:
            return True

        # Some models are known to be compatible
        compatible_pairs = [
            (EmbeddingModel.OPENAI_3_SMALL.value, EmbeddingModel.OPENAI_3_LARGE.value),
        ]

        for a, b in compatible_pairs:
            if (self.model == a and other.model == b) or \
               (self.model == b and other.model == a):
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "version": self.version,
            "provider": self.provider,
            "normalized": self.normalized,
            "version_id": self.version_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingVersion":
        return cls(
            model=data["model"],
            dimensions=data["dimensions"],
            version=data.get("version", "1.0.0"),
            provider=data.get("provider", "unknown"),
            normalized=data.get("normalized", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VersionedEmbedding:
    """An embedding with version information."""

    embedding: list[float]
    version: EmbeddingVersion
    text_hash: str  # Hash of original text
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding,
            "version": self.version.to_dict(),
            "text_hash": self.text_hash,
            "created_at": self.created_at.isoformat(),
        }


# Pre-defined version configurations
OPENAI_ADA_002_VERSION = EmbeddingVersion(
    model=EmbeddingModel.OPENAI_ADA_002.value,
    dimensions=1536,
    version="1.0.0",
    provider="openai",
)

OPENAI_3_SMALL_VERSION = EmbeddingVersion(
    model=EmbeddingModel.OPENAI_3_SMALL.value,
    dimensions=1536,
    version="1.0.0",
    provider="openai",
)

OPENAI_3_LARGE_VERSION = EmbeddingVersion(
    model=EmbeddingModel.OPENAI_3_LARGE.value,
    dimensions=3072,
    version="1.0.0",
    provider="openai",
)


class VersionedEmbedder:
    """
    Wrapper around embedding models that adds version tracking.

    Provides:
    - Version metadata with each embedding
    - Batch embedding with progress tracking
    - Caching support
    - Version compatibility checking
    """

    def __init__(
        self,
        embeddings: Embeddings,
        version: EmbeddingVersion,
        cache: dict[str, VersionedEmbedding] | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize versioned embedder.

        Args:
            embeddings: LangChain embeddings instance
            version: Version information for this embedder
            cache: Optional cache for embeddings (text_hash -> embedding)
            use_cache: Whether to use caching
        """
        self._embeddings = embeddings
        self._version = version
        self._cache = cache if cache is not None else {}
        self._use_cache = use_cache

        # Stats
        self._total_embedded = 0
        self._cache_hits = 0

    @property
    def version(self) -> EmbeddingVersion:
        return self._version

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_embedded": self._total_embedded,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._cache_hits / max(self._total_embedded, 1),
        }

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def embed_text(self, text: str) -> VersionedEmbedding:
        """
        Embed a single text with version tracking.

        Args:
            text: Text to embed

        Returns:
            VersionedEmbedding with version metadata
        """
        text_hash = self._hash_text(text)

        # Check cache
        if self._use_cache and text_hash in self._cache:
            self._cache_hits += 1
            return self._cache[text_hash]

        # Generate embedding
        embedding = await self._embeddings.aembed_query(text)
        self._total_embedded += 1

        versioned = VersionedEmbedding(
            embedding=embedding,
            version=self._version,
            text_hash=text_hash,
        )

        # Cache
        if self._use_cache:
            self._cache[text_hash] = versioned

        return versioned

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100,
        show_progress: bool = False,
    ) -> list[VersionedEmbedding]:
        """
        Embed multiple texts with version tracking.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding API calls
            show_progress: Whether to log progress

        Returns:
            List of VersionedEmbedding objects
        """
        results: list[VersionedEmbedding] = []
        texts_to_embed: list[tuple[int, str]] = []  # (original_index, text)

        # Check cache first
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if self._use_cache and text_hash in self._cache:
                self._cache_hits += 1
                results.append(self._cache[text_hash])
            else:
                texts_to_embed.append((i, text))
                results.append(None)  # Placeholder

        if not texts_to_embed:
            return results

        # Batch embed remaining texts
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[batch_idx:batch_idx + batch_size]
            batch_texts = [t for _, t in batch]

            if show_progress:
                current_batch = batch_idx // batch_size + 1
                logger.info(
                    "Embedding batch",
                    batch=f"{current_batch}/{total_batches}",
                    size=len(batch_texts),
                )

            # Generate embeddings
            embeddings = await self._embeddings.aembed_documents(batch_texts)
            self._total_embedded += len(batch_texts)

            # Create versioned embeddings
            for (original_idx, text), embedding in zip(batch, embeddings, strict=True):
                text_hash = self._hash_text(text)
                versioned = VersionedEmbedding(
                    embedding=embedding,
                    version=self._version,
                    text_hash=text_hash,
                )

                results[original_idx] = versioned

                if self._use_cache:
                    self._cache[text_hash] = versioned

        return results

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.debug("Embedding cache cleared")


class EmbeddingVersionStore:
    """
    Stores and manages embedding version metadata in Neo4j.

    Tracks:
    - Which version was used for each node's embedding
    - When embeddings were generated
    - Version migration history
    """

    def __init__(self, neo4j_client: Any):
        self.neo4j = neo4j_client

    async def initialize_schema(self) -> None:
        """Create schema for version tracking."""

        queries = [
            # Version metadata node
            """
            CREATE CONSTRAINT embedding_version_id IF NOT EXISTS
            FOR (v:EmbeddingVersion)
            REQUIRE v.version_id IS UNIQUE
            """,
            # Index on embedding version property
            """
            CREATE INDEX entity_embedding_version IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding_version)
            """,
            """
            CREATE INDEX chunk_embedding_version IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding_version)
            """,
        ]

        for query in queries:
            try:
                await self.neo4j.execute_write(query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Schema creation warning", error=str(e))

    async def register_version(self, version: EmbeddingVersion) -> None:
        """Register an embedding version in the database."""

        query = """
        MERGE (v:EmbeddingVersion {version_id: $version_id})
        SET v.model = $model,
            v.dimensions = $dimensions,
            v.version = $version,
            v.provider = $provider,
            v.normalized = $normalized,
            v.registered_at = datetime()
        """

        await self.neo4j.execute_write(
            query,
            version_id=version.version_id,
            model=version.model,
            dimensions=version.dimensions,
            version=version.version,
            provider=version.provider,
            normalized=version.normalized,
        )

        logger.info(
            "Registered embedding version",
            version_id=version.version_id,
            model=version.model,
        )

    async def get_current_version(self) -> EmbeddingVersion | None:
        """Get the most recently used embedding version."""

        query = """
        MATCH (v:EmbeddingVersion)
        RETURN v
        ORDER BY v.registered_at DESC
        LIMIT 1
        """

        result = await self.neo4j.execute_read(query)

        if result and result[0]["v"]:
            node = result[0]["v"]
            return EmbeddingVersion(
                model=node["model"],
                dimensions=node["dimensions"],
                version=node["version"],
                provider=node["provider"],
                normalized=node.get("normalized", True),
            )

        return None

    async def get_version_stats(self) -> dict[str, Any]:
        """Get statistics about embedding versions in use."""

        query = """
        MATCH (e:Entity)
        WHERE e.embedding_version IS NOT NULL
        WITH e.embedding_version AS version, count(e) AS entity_count
        RETURN version, entity_count, 'entity' AS node_type

        UNION ALL

        MATCH (c:Chunk)
        WHERE c.embedding_version IS NOT NULL
        WITH c.embedding_version AS version, count(c) AS chunk_count
        RETURN version, chunk_count AS entity_count, 'chunk' AS node_type
        """

        results = await self.neo4j.execute_read(query)

        stats = {}
        for record in results or []:
            version = record["version"]
            node_type = record["node_type"]
            count = record["entity_count"]

            if version not in stats:
                stats[version] = {"entities": 0, "chunks": 0}

            if node_type == "entity":
                stats[version]["entities"] = count
            else:
                stats[version]["chunks"] = count

        return stats

    async def find_outdated_nodes(
        self,
        current_version: EmbeddingVersion,
        node_type: str = "Entity",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Find nodes with outdated or missing embedding versions."""

        query = f"""
        MATCH (n:{node_type})
        WHERE n.embedding_version IS NULL
           OR n.embedding_version <> $current_version_id
        RETURN n.id AS id, n.name AS name, n.embedding_version AS old_version
        LIMIT $limit
        """

        results = await self.neo4j.execute_read(
            query,
            current_version_id=current_version.version_id,
            limit=limit,
        )

        return [
            {
                "id": r["id"],
                "name": r["name"],
                "old_version": r["old_version"],
            }
            for r in results or []
        ]


class EmbeddingMigrator:
    """
    Handles migration of embeddings when switching models.

    Supports:
    - Full re-embedding of all content
    - Incremental migration (batch by batch)
    - Version tracking during migration
    - Rollback support
    """

    def __init__(
        self,
        neo4j_client: Any,
        embedder: VersionedEmbedder,
        version_store: EmbeddingVersionStore,
    ):
        self.neo4j = neo4j_client
        self.embedder = embedder
        self.version_store = version_store
        self._migration_in_progress = False

    async def check_migration_needed(self) -> dict[str, Any]:
        """Check if migration is needed."""

        current_version = await self.version_store.get_current_version()
        target_version = self.embedder.version

        if current_version is None:
            return {
                "needed": True,
                "reason": "no_existing_version",
                "current": None,
                "target": target_version.to_dict(),
            }

        if current_version.version_id == target_version.version_id:
            return {
                "needed": False,
                "reason": "same_version",
                "current": current_version.to_dict(),
                "target": target_version.to_dict(),
            }

        if not target_version.is_compatible_with(current_version):
            # Count outdated nodes
            outdated_entities = await self.version_store.find_outdated_nodes(
                target_version, "Entity", limit=1
            )
            outdated_chunks = await self.version_store.find_outdated_nodes(
                target_version, "Chunk", limit=1
            )

            return {
                "needed": True,
                "reason": "incompatible_version",
                "current": current_version.to_dict(),
                "target": target_version.to_dict(),
                "has_outdated_entities": len(outdated_entities) > 0,
                "has_outdated_chunks": len(outdated_chunks) > 0,
            }

        return {
            "needed": False,
            "reason": "compatible_version",
            "current": current_version.to_dict(),
            "target": target_version.to_dict(),
        }

    async def migrate_entities(
        self,
        batch_size: int = 100,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """
        Migrate entity embeddings to new version.

        Args:
            batch_size: Number of entities to process per batch
            progress_callback: Optional callback(processed, total, message)

        Returns:
            Migration statistics
        """
        if self._migration_in_progress:
            raise RuntimeError("Migration already in progress")

        self._migration_in_progress = True
        target_version = self.embedder.version

        try:
            # Find all outdated entities
            query = """
            MATCH (e:Entity)
            WHERE e.embedding_version IS NULL
               OR e.embedding_version <> $version_id
            RETURN count(e) AS total
            """
            result = await self.neo4j.execute_read(
                query,
                version_id=target_version.version_id,
            )
            total = result[0]["total"] if result else 0

            if total == 0:
                return {"migrated": 0, "total": 0, "status": "no_migration_needed"}

            logger.info(
                "Starting entity migration",
                total=total,
                target_version=target_version.version_id,
            )

            migrated = 0
            errors = 0

            while True:
                # Get batch of entities
                query = """
                MATCH (e:Entity)
                WHERE e.embedding_version IS NULL
                   OR e.embedding_version <> $version_id
                RETURN e.id AS id, e.name AS name, e.description AS description
                LIMIT $batch_size
                """

                entities = await self.neo4j.execute_read(
                    query,
                    version_id=target_version.version_id,
                    batch_size=batch_size,
                )

                if not entities:
                    break

                # Generate embeddings
                texts = [
                    f"{e['name']} {e.get('description', '')}"
                    for e in entities
                ]

                versioned_embeddings = await self.embedder.embed_texts(texts)

                # Update entities
                for entity, versioned in zip(entities, versioned_embeddings, strict=True):
                    try:
                        update_query = """
                        MATCH (e:Entity {id: $id})
                        SET e.embedding = $embedding,
                            e.embedding_version = $version_id,
                            e.embedding_updated_at = datetime()
                        """
                        await self.neo4j.execute_write(
                            update_query,
                            id=entity["id"],
                            embedding=versioned.embedding,
                            version_id=target_version.version_id,
                        )
                        migrated += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to migrate entity",
                            entity_id=entity["id"],
                            error=str(e),
                        )
                        errors += 1

                if progress_callback:
                    progress_callback(migrated, total, f"Migrated {migrated}/{total} entities")

            # Register version
            await self.version_store.register_version(target_version)

            return {
                "migrated": migrated,
                "errors": errors,
                "total": total,
                "status": "completed",
                "version": target_version.to_dict(),
            }

        finally:
            self._migration_in_progress = False

    async def migrate_chunks(
        self,
        batch_size: int = 100,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Migrate chunk embeddings to new version."""

        if self._migration_in_progress:
            raise RuntimeError("Migration already in progress")

        self._migration_in_progress = True
        target_version = self.embedder.version

        try:
            # Count outdated chunks
            query = """
            MATCH (c:Chunk)
            WHERE c.embedding_version IS NULL
               OR c.embedding_version <> $version_id
            RETURN count(c) AS total
            """
            result = await self.neo4j.execute_read(
                query,
                version_id=target_version.version_id,
            )
            total = result[0]["total"] if result else 0

            if total == 0:
                return {"migrated": 0, "total": 0, "status": "no_migration_needed"}

            logger.info(
                "Starting chunk migration",
                total=total,
                target_version=target_version.version_id,
            )

            migrated = 0
            errors = 0

            while True:
                # Get batch of chunks
                query = """
                MATCH (c:Chunk)
                WHERE c.embedding_version IS NULL
                   OR c.embedding_version <> $version_id
                RETURN c.id AS id, c.text AS text
                LIMIT $batch_size
                """

                chunks = await self.neo4j.execute_read(
                    query,
                    version_id=target_version.version_id,
                    batch_size=batch_size,
                )

                if not chunks:
                    break

                # Generate embeddings
                texts = [c["text"] for c in chunks]
                versioned_embeddings = await self.embedder.embed_texts(texts)

                # Update chunks
                for chunk, versioned in zip(chunks, versioned_embeddings, strict=True):
                    try:
                        update_query = """
                        MATCH (c:Chunk {id: $id})
                        SET c.embedding = $embedding,
                            c.embedding_version = $version_id,
                            c.embedding_updated_at = datetime()
                        """
                        await self.neo4j.execute_write(
                            update_query,
                            id=chunk["id"],
                            embedding=versioned.embedding,
                            version_id=target_version.version_id,
                        )
                        migrated += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to migrate chunk",
                            chunk_id=chunk["id"],
                            error=str(e),
                        )
                        errors += 1

                if progress_callback:
                    progress_callback(migrated, total, f"Migrated {migrated}/{total} chunks")

            return {
                "migrated": migrated,
                "errors": errors,
                "total": total,
                "status": "completed",
                "version": target_version.to_dict(),
            }

        finally:
            self._migration_in_progress = False

    async def migrate_all(
        self,
        batch_size: int = 100,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Migrate all embeddings (entities and chunks)."""

        entity_result = await self.migrate_entities(batch_size, progress_callback)
        chunk_result = await self.migrate_chunks(batch_size, progress_callback)

        return {
            "entities": entity_result,
            "chunks": chunk_result,
            "status": "completed",
        }


def create_versioned_embedder(
    model_name: str = "text-embedding-3-small",
    provider: str = "openai",
    api_key: str | None = None,
    dimensions: int | None = None,
) -> VersionedEmbedder:
    """
    Factory function to create a VersionedEmbedder.

    Args:
        model_name: Name of the embedding model
        provider: Provider (openai, anthropic, local)
        api_key: API key for the provider
        dimensions: Override dimensions (for models that support it)

    Returns:
        Configured VersionedEmbedder
    """
    from pydantic import SecretStr

    # Determine dimensions based on model
    default_dimensions = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "voyage-2": 1024,
        "voyage-large-2": 1536,
    }

    dims = dimensions or default_dimensions.get(model_name, 1536)

    # Create version
    version = EmbeddingVersion(
        model=model_name,
        dimensions=dims,
        provider=provider,
    )

    # Create embeddings instance based on provider
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=SecretStr(api_key) if api_key else None,
            dimensions=dimensions,  # Only pass if explicitly set
        )
    elif provider == "local":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return VersionedEmbedder(embeddings, version)

"""
Canonical Entity Registry.

Neo4j-backed registry for storing canonical entities and mapping
variations (aliases, translations, abbreviations) to them.

Handles cases like:
- "레드팀", "Red Team", "적색팀" → canonical: "Red Team"
- "Microsoft Corp", "MSFT", "마이크로소프트" → canonical: "Microsoft Corporation"
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.sddi.extractors.entity_extractor import EntityType

logger = structlog.get_logger(__name__)


class AliasType(str, Enum):
    """Type of alias relationship."""

    EXACT = "exact"           # Exact canonical name
    ABBREVIATION = "abbreviation"  # Shortened form (IBM, MS)
    TRANSLATION = "translation"    # Different language
    SYNONYM = "synonym"           # Alternative name
    TYPO = "typo"                 # Common misspelling
    HISTORICAL = "historical"     # Old name (e.g., "Facebook" → "Meta")


@dataclass
class CanonicalEntity:
    """Canonical entity record."""

    canonical_id: str
    canonical_name: str
    entity_type: EntityType
    description: str | None = None
    aliases: dict[str, AliasType] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    source: str = "system"  # system, user, auto


@dataclass
class AliasMatch:
    """Result of alias lookup."""

    canonical_entity: CanonicalEntity
    matched_alias: str
    alias_type: AliasType
    similarity_score: float
    match_method: str  # exact, fuzzy, embedding


class CanonicalEntityRegistry:
    """
    Neo4j-backed registry for canonical entities.

    Provides:
    - Canonical entity storage and retrieval
    - Alias management (add, remove, lookup)
    - Fuzzy and embedding-based matching
    - Batch operations for efficiency

    Neo4j Schema:
    - (:CanonicalEntity {canonical_id, name, type, description, embedding, ...})
    - (:EntityAlias {alias, alias_type, confidence})
    - (:CanonicalEntity)-[:HAS_ALIAS]->(:EntityAlias)
    """

    def __init__(
        self,
        neo4j_client: Any,
        embedding_provider: Any | None = None,
        fuzzy_threshold: float = 0.85,
        embedding_threshold: float = 0.88,
        auto_create: bool = True,
    ):
        """
        Initialize registry.

        Args:
            neo4j_client: Neo4j client for database operations
            embedding_provider: Provider for generating embeddings
            fuzzy_threshold: Threshold for fuzzy matching
            embedding_threshold: Threshold for embedding similarity
            auto_create: Auto-create canonical entities when not found
        """
        self.neo4j = neo4j_client
        self.embedding_provider = embedding_provider
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.auto_create = auto_create

        # In-memory cache for frequently accessed entities
        self._cache: dict[str, CanonicalEntity] = {}
        self._alias_cache: dict[str, str] = {}  # alias → canonical_id

    async def initialize_schema(self) -> None:
        """Create Neo4j constraints and indexes for canonical entities."""

        queries = [
            # Constraints
            """
            CREATE CONSTRAINT canonical_entity_id IF NOT EXISTS
            FOR (c:CanonicalEntity)
            REQUIRE c.canonical_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT entity_alias_unique IF NOT EXISTS
            FOR (a:EntityAlias)
            REQUIRE (a.alias, a.entity_type) IS UNIQUE
            """,

            # Indexes
            """
            CREATE INDEX canonical_entity_name IF NOT EXISTS
            FOR (c:CanonicalEntity) ON (c.name)
            """,
            """
            CREATE INDEX canonical_entity_type IF NOT EXISTS
            FOR (c:CanonicalEntity) ON (c.entity_type)
            """,
            """
            CREATE INDEX entity_alias_text IF NOT EXISTS
            FOR (a:EntityAlias) ON (a.alias)
            """,

            # Full-text index for fuzzy search
            """
            CREATE FULLTEXT INDEX canonical_entity_fulltext IF NOT EXISTS
            FOR (c:CanonicalEntity) ON EACH [c.name, c.description]
            """,
            """
            CREATE FULLTEXT INDEX entity_alias_fulltext IF NOT EXISTS
            FOR (a:EntityAlias) ON EACH [a.alias]
            """,

            # Vector index for embedding search
            """
            CREATE VECTOR INDEX canonical_entity_embedding IF NOT EXISTS
            FOR (c:CanonicalEntity) ON (c.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """,
        ]

        for query in queries:
            try:
                await self.neo4j.execute_write(query)
            except Exception as e:
                # Index/constraint might already exist
                if "already exists" not in str(e).lower():
                    logger.warning("Schema creation warning", query=query[:50], error=str(e))

        logger.info("Canonical entity registry schema initialized")

    async def resolve_to_canonical(
        self,
        name: str,
        entity_type: EntityType | None = None,
        context: str | None = None,
    ) -> CanonicalEntity | None:
        """
        Resolve a name to its canonical entity.

        Resolution order:
        1. Exact match in cache
        2. Exact match in Neo4j
        3. Fuzzy match (Levenshtein)
        4. Embedding similarity match
        5. Auto-create if enabled

        Args:
            name: Entity name to resolve
            entity_type: Optional type filter
            context: Optional context for disambiguation

        Returns:
            CanonicalEntity if found/created, None otherwise
        """
        normalized = self._normalize(name)

        # Stage 1: Check cache
        cache_key = f"{entity_type or 'any'}:{normalized}"
        if cache_key in self._alias_cache:
            canonical_id = self._alias_cache[cache_key]
            if canonical_id in self._cache:
                return self._cache[canonical_id]

        # Stage 2: Exact match in Neo4j
        canonical = await self._exact_match(normalized, entity_type)
        if canonical:
            self._update_cache(canonical, normalized)
            return canonical

        # Stage 3: Fuzzy match
        canonical = await self._fuzzy_match(name, entity_type)
        if canonical:
            self._update_cache(canonical, normalized)
            return canonical

        # Stage 4: Embedding match
        if self.embedding_provider:
            canonical = await self._embedding_match(name, entity_type, context)
            if canonical:
                self._update_cache(canonical, normalized)
                return canonical

        # Stage 5: Auto-create if enabled
        if self.auto_create and entity_type:
            canonical = await self.create_canonical(
                name=name,
                entity_type=entity_type,
                source="auto",
                confidence=0.8,
            )
            return canonical

        return None

    async def create_canonical(
        self,
        name: str,
        entity_type: EntityType,
        description: str | None = None,
        aliases: list[tuple[str, AliasType]] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "user",
        confidence: float = 1.0,
    ) -> CanonicalEntity:
        """
        Create a new canonical entity.

        Args:
            name: Canonical name
            entity_type: Entity type
            description: Optional description
            aliases: Optional list of (alias, type) tuples
            metadata: Optional metadata
            source: Source of creation
            confidence: Confidence score

        Returns:
            Created CanonicalEntity
        """
        import hashlib

        canonical_id = hashlib.sha256(
            f"{entity_type.value}:{name.lower()}".encode()
        ).hexdigest()[:16]

        # Generate embedding if provider available
        embedding = None
        if self.embedding_provider:
            try:
                embedding = await self.embedding_provider.embed_text(name)
            except Exception as e:
                logger.warning("Failed to generate embedding", name=name, error=str(e))

        now = datetime.utcnow()
        entity = CanonicalEntity(
            canonical_id=canonical_id,
            canonical_name=name,
            entity_type=entity_type,
            description=description,
            aliases={name.lower(): AliasType.EXACT},
            metadata=metadata or {},
            embedding=embedding,
            created_at=now,
            updated_at=now,
            confidence=confidence,
            source=source,
        )

        # Add provided aliases
        if aliases:
            for alias, alias_type in aliases:
                entity.aliases[alias.lower()] = alias_type

        # Store in Neo4j
        await self._store_canonical(entity)

        # Update cache
        self._update_cache(entity, name.lower())

        logger.info(
            "Created canonical entity",
            canonical_id=canonical_id,
            name=name,
            type=entity_type.value,
            aliases_count=len(entity.aliases),
        )

        return entity

    async def add_alias(
        self,
        canonical_id: str,
        alias: str,
        alias_type: AliasType = AliasType.SYNONYM,
        confidence: float = 1.0,
    ) -> bool:
        """
        Add an alias to a canonical entity.

        Args:
            canonical_id: ID of canonical entity
            alias: New alias to add
            alias_type: Type of alias
            confidence: Confidence that this alias is correct

        Returns:
            True if added successfully
        """
        query = """
        MATCH (c:CanonicalEntity {canonical_id: $canonical_id})
        MERGE (a:EntityAlias {alias: $alias, entity_type: c.entity_type})
        ON CREATE SET
            a.alias_type = $alias_type,
            a.confidence = $confidence,
            a.created_at = datetime()
        ON MATCH SET
            a.alias_type = $alias_type,
            a.confidence = CASE WHEN a.confidence < $confidence
                           THEN $confidence ELSE a.confidence END,
            a.updated_at = datetime()
        MERGE (c)-[:HAS_ALIAS]->(a)
        RETURN c.canonical_id AS id
        """

        result = await self.neo4j.execute_write(
            query,
            canonical_id=canonical_id,
            alias=alias.lower(),
            alias_type=alias_type.value,
            confidence=confidence,
        )

        if result:
            # Update cache
            cache_key = f"any:{alias.lower()}"
            self._alias_cache[cache_key] = canonical_id

            logger.info(
                "Added alias",
                canonical_id=canonical_id,
                alias=alias,
                alias_type=alias_type.value,
            )
            return True

        return False

    async def remove_alias(self, canonical_id: str, alias: str) -> bool:
        """Remove an alias from a canonical entity."""

        query = """
        MATCH (c:CanonicalEntity {canonical_id: $canonical_id})-[r:HAS_ALIAS]->(a:EntityAlias {alias: $alias})
        DELETE r
        WITH a
        WHERE NOT exists((a)<-[:HAS_ALIAS]-())
        DELETE a
        RETURN count(*) AS deleted
        """

        await self.neo4j.execute_write(
            query,
            canonical_id=canonical_id,
            alias=alias.lower(),
        )

        # Remove from cache
        for key in list(self._alias_cache.keys()):
            if key.endswith(f":{alias.lower()}"):
                del self._alias_cache[key]

        return True

    async def merge_entities(
        self,
        source_id: str,
        target_id: str,
        keep_aliases: bool = True,
    ) -> CanonicalEntity | None:
        """
        Merge one canonical entity into another.

        All aliases from source are transferred to target.
        Source entity is deleted.

        Args:
            source_id: ID of entity to merge from (will be deleted)
            target_id: ID of entity to merge into (will be kept)
            keep_aliases: Whether to keep source's aliases

        Returns:
            Updated target entity
        """
        if keep_aliases:
            # Transfer aliases
            query = """
            MATCH (source:CanonicalEntity {canonical_id: $source_id})-[:HAS_ALIAS]->(a:EntityAlias)
            MATCH (target:CanonicalEntity {canonical_id: $target_id})
            MERGE (target)-[:HAS_ALIAS]->(a)
            """
            await self.neo4j.execute_write(query, source_id=source_id, target_id=target_id)

        # Delete source entity
        query = """
        MATCH (source:CanonicalEntity {canonical_id: $source_id})
        DETACH DELETE source
        """
        await self.neo4j.execute_write(query, source_id=source_id)

        # Clear caches
        if source_id in self._cache:
            del self._cache[source_id]

        self._alias_cache = {k: v for k, v in self._alias_cache.items() if v != source_id}

        # Return updated target
        return await self.get_by_id(target_id)

    async def get_by_id(self, canonical_id: str) -> CanonicalEntity | None:
        """Get canonical entity by ID."""

        if canonical_id in self._cache:
            return self._cache[canonical_id]

        query = """
        MATCH (c:CanonicalEntity {canonical_id: $canonical_id})
        OPTIONAL MATCH (c)-[:HAS_ALIAS]->(a:EntityAlias)
        RETURN c, collect(a) AS aliases
        """

        result = await self.neo4j.execute_read(query, canonical_id=canonical_id)

        if result and result[0]["c"]:
            entity = self._parse_neo4j_result(result[0])
            self._cache[canonical_id] = entity
            return entity

        return None

    async def search(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[AliasMatch]:
        """
        Search for canonical entities by name or alias.

        Uses full-text search with fuzzy matching.

        Args:
            query: Search query
            entity_type: Optional type filter
            limit: Maximum results

        Returns:
            List of matching entities with scores
        """
        cypher = """
        CALL db.index.fulltext.queryNodes('canonical_entity_fulltext', $query)
        YIELD node, score
        WHERE ($entity_type IS NULL OR node.entity_type = $entity_type)
        RETURN node, score, 'fulltext' AS match_type
        ORDER BY score DESC
        LIMIT $limit

        UNION

        CALL db.index.fulltext.queryNodes('entity_alias_fulltext', $query)
        YIELD node AS alias_node, score
        MATCH (c:CanonicalEntity)-[:HAS_ALIAS]->(alias_node)
        WHERE ($entity_type IS NULL OR c.entity_type = $entity_type)
        RETURN c AS node, score, 'alias' AS match_type
        ORDER BY score DESC
        LIMIT $limit
        """

        results = await self.neo4j.execute_read(
            cypher,
            query=f"{query}~",  # Fuzzy search
            entity_type=entity_type.value if entity_type else None,
            limit=limit,
        )

        matches = []
        seen_ids = set()

        for record in results or []:
            node = record["node"]
            if node["canonical_id"] in seen_ids:
                continue
            seen_ids.add(node["canonical_id"])

            entity = CanonicalEntity(
                canonical_id=node["canonical_id"],
                canonical_name=node["name"],
                entity_type=EntityType(node["entity_type"]),
                description=node.get("description"),
            )

            matches.append(AliasMatch(
                canonical_entity=entity,
                matched_alias=query,
                alias_type=AliasType.SYNONYM,
                similarity_score=record["score"],
                match_method=record["match_type"],
            ))

        return matches

    async def get_all_aliases(
        self,
        canonical_id: str,
    ) -> dict[str, AliasType]:
        """Get all aliases for a canonical entity."""

        query = """
        MATCH (c:CanonicalEntity {canonical_id: $canonical_id})-[:HAS_ALIAS]->(a:EntityAlias)
        RETURN a.alias AS alias, a.alias_type AS alias_type
        """

        results = await self.neo4j.execute_read(query, canonical_id=canonical_id)

        return {
            r["alias"]: AliasType(r["alias_type"])
            for r in results or []
        }

    async def bulk_import(
        self,
        entities: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> tuple[int, int]:
        """
        Bulk import canonical entities.

        Args:
            entities: List of entity dicts with name, type, aliases, etc.
            batch_size: Batch size for Neo4j operations

        Returns:
            Tuple of (created_count, error_count)
        """
        created = 0
        errors = 0

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]

            for entity_data in batch:
                try:
                    await self.create_canonical(
                        name=entity_data["name"],
                        entity_type=EntityType(entity_data["type"]),
                        description=entity_data.get("description"),
                        aliases=[
                            (a["alias"], AliasType(a["type"]))
                            for a in entity_data.get("aliases", [])
                        ],
                        metadata=entity_data.get("metadata", {}),
                        source=entity_data.get("source", "import"),
                    )
                    created += 1
                except Exception as e:
                    logger.warning(
                        "Failed to import entity",
                        name=entity_data.get("name"),
                        error=str(e),
                    )
                    errors += 1

        logger.info(
            "Bulk import completed",
            created=created,
            errors=errors,
            total=len(entities),
        )

        return created, errors

    async def export(
        self,
        entity_type: EntityType | None = None,
    ) -> list[dict[str, Any]]:
        """Export canonical entities as list of dicts."""

        query = """
        MATCH (c:CanonicalEntity)
        WHERE $entity_type IS NULL OR c.entity_type = $entity_type
        OPTIONAL MATCH (c)-[:HAS_ALIAS]->(a:EntityAlias)
        RETURN c, collect(a) AS aliases
        """

        results = await self.neo4j.execute_read(
            query,
            entity_type=entity_type.value if entity_type else None,
        )

        exported = []
        for record in results or []:
            node = record["c"]
            aliases = record["aliases"]

            exported.append({
                "canonical_id": node["canonical_id"],
                "name": node["name"],
                "type": node["entity_type"],
                "description": node.get("description"),
                "aliases": [
                    {"alias": a["alias"], "type": a["alias_type"]}
                    for a in aliases if a
                ],
                "metadata": node.get("metadata", {}),
                "created_at": node.get("created_at"),
            })

        return exported

    def clear_cache(self) -> None:
        """Clear in-memory caches."""
        self._cache.clear()
        self._alias_cache.clear()
        logger.debug("Registry cache cleared")

    # Private methods

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        import unicodedata

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Lowercase and strip
        text = text.lower().strip()
        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _update_cache(self, entity: CanonicalEntity, alias: str) -> None:
        """Update caches with entity and alias."""

        self._cache[entity.canonical_id] = entity

        # Cache all known aliases
        for a in entity.aliases:
            cache_key = f"{entity.entity_type.value}:{a}"
            self._alias_cache[cache_key] = entity.canonical_id
            cache_key = f"any:{a}"
            self._alias_cache[cache_key] = entity.canonical_id

        # Also cache the lookup alias
        cache_key = f"{entity.entity_type.value}:{alias}"
        self._alias_cache[cache_key] = entity.canonical_id

    async def _exact_match(
        self,
        normalized_name: str,
        entity_type: EntityType | None,
    ) -> CanonicalEntity | None:
        """Find exact match in Neo4j."""

        query = """
        MATCH (c:CanonicalEntity)-[:HAS_ALIAS]->(a:EntityAlias {alias: $alias})
        WHERE $entity_type IS NULL OR c.entity_type = $entity_type
        OPTIONAL MATCH (c)-[:HAS_ALIAS]->(all_aliases:EntityAlias)
        RETURN c, collect(all_aliases) AS aliases
        LIMIT 1
        """

        result = await self.neo4j.execute_read(
            query,
            alias=normalized_name,
            entity_type=entity_type.value if entity_type else None,
        )

        if result and result[0]["c"]:
            return self._parse_neo4j_result(result[0])

        return None

    async def _fuzzy_match(
        self,
        name: str,
        entity_type: EntityType | None,
    ) -> CanonicalEntity | None:
        """Find fuzzy match using full-text search."""

        query = """
        CALL db.index.fulltext.queryNodes('entity_alias_fulltext', $query)
        YIELD node, score
        WHERE score > $threshold
        MATCH (c:CanonicalEntity)-[:HAS_ALIAS]->(node)
        WHERE $entity_type IS NULL OR c.entity_type = $entity_type
        OPTIONAL MATCH (c)-[:HAS_ALIAS]->(all_aliases:EntityAlias)
        RETURN c, collect(all_aliases) AS aliases, score
        ORDER BY score DESC
        LIMIT 1
        """

        result = await self.neo4j.execute_read(
            query,
            query=f"{name}~",  # Fuzzy search syntax
            threshold=self.fuzzy_threshold,
            entity_type=entity_type.value if entity_type else None,
        )

        if result and result[0]["c"]:
            return self._parse_neo4j_result(result[0])

        return None

    async def _embedding_match(
        self,
        name: str,
        entity_type: EntityType | None,
        context: str | None,
    ) -> CanonicalEntity | None:
        """Find match using embedding similarity."""

        if not self.embedding_provider:
            return None

        try:
            # Generate embedding for query
            query_text = f"{name} {context}" if context else name
            query_embedding = await self.embedding_provider.embed_text(query_text)

            # Vector similarity search in Neo4j
            query = """
            CALL db.index.vector.queryNodes(
                'canonical_entity_embedding',
                $k,
                $embedding
            ) YIELD node, score
            WHERE score > $threshold
            AND ($entity_type IS NULL OR node.entity_type = $entity_type)
            OPTIONAL MATCH (node)-[:HAS_ALIAS]->(a:EntityAlias)
            RETURN node AS c, collect(a) AS aliases, score
            ORDER BY score DESC
            LIMIT 1
            """

            result = await self.neo4j.execute_read(
                query,
                k=5,
                embedding=query_embedding,
                threshold=self.embedding_threshold,
                entity_type=entity_type.value if entity_type else None,
            )

            if result and result[0]["c"]:
                return self._parse_neo4j_result(result[0])

        except Exception as e:
            logger.warning("Embedding match failed", name=name, error=str(e))

        return None

    async def _store_canonical(self, entity: CanonicalEntity) -> None:
        """Store canonical entity in Neo4j."""

        # Create canonical entity node
        query = """
        MERGE (c:CanonicalEntity {canonical_id: $canonical_id})
        SET c.name = $name,
            c.entity_type = $entity_type,
            c.description = $description,
            c.embedding = $embedding,
            c.metadata = $metadata,
            c.confidence = $confidence,
            c.source = $source,
            c.created_at = CASE WHEN c.created_at IS NULL
                           THEN datetime() ELSE c.created_at END,
            c.updated_at = datetime()
        RETURN c
        """

        await self.neo4j.execute_write(
            query,
            canonical_id=entity.canonical_id,
            name=entity.canonical_name,
            entity_type=entity.entity_type.value,
            description=entity.description,
            embedding=entity.embedding,
            metadata=entity.metadata,
            confidence=entity.confidence,
            source=entity.source,
        )

        # Create aliases
        for alias, alias_type in entity.aliases.items():
            await self.add_alias(
                entity.canonical_id,
                alias,
                alias_type,
                confidence=entity.confidence,
            )

    def _parse_neo4j_result(self, record: dict[str, Any]) -> CanonicalEntity:
        """Parse Neo4j result into CanonicalEntity."""

        node = record["c"]
        aliases_nodes = record.get("aliases", [])

        aliases = {}
        for a in aliases_nodes:
            if a:
                aliases[a["alias"]] = AliasType(a.get("alias_type", "synonym"))

        return CanonicalEntity(
            canonical_id=node["canonical_id"],
            canonical_name=node["name"],
            entity_type=EntityType(node["entity_type"]),
            description=node.get("description"),
            aliases=aliases,
            metadata=node.get("metadata", {}),
            embedding=node.get("embedding"),
            confidence=node.get("confidence", 1.0),
            source=node.get("source", "unknown"),
        )


# Pre-built canonical entity sets for common domains
COMMON_TECH_ENTITIES = [
    {
        "name": "Microsoft Corporation",
        "type": "organization",
        "aliases": [
            {"alias": "microsoft", "type": "synonym"},
            {"alias": "msft", "type": "abbreviation"},
            {"alias": "ms", "type": "abbreviation"},
            {"alias": "마이크로소프트", "type": "translation"},
        ],
    },
    {
        "name": "Google LLC",
        "type": "organization",
        "aliases": [
            {"alias": "google", "type": "synonym"},
            {"alias": "googl", "type": "abbreviation"},
            {"alias": "alphabet", "type": "synonym"},
            {"alias": "구글", "type": "translation"},
        ],
    },
    {
        "name": "Amazon.com Inc.",
        "type": "organization",
        "aliases": [
            {"alias": "amazon", "type": "synonym"},
            {"alias": "amzn", "type": "abbreviation"},
            {"alias": "aws", "type": "abbreviation"},
            {"alias": "아마존", "type": "translation"},
        ],
    },
    {
        "name": "Red Team",
        "type": "concept",
        "description": "Security testing team that simulates attacks",
        "aliases": [
            {"alias": "red team", "type": "exact"},
            {"alias": "redteam", "type": "synonym"},
            {"alias": "레드팀", "type": "translation"},
            {"alias": "적색팀", "type": "translation"},
            {"alias": "공격팀", "type": "synonym"},
        ],
    },
    {
        "name": "Blue Team",
        "type": "concept",
        "description": "Security team that defends against attacks",
        "aliases": [
            {"alias": "blue team", "type": "exact"},
            {"alias": "blueteam", "type": "synonym"},
            {"alias": "블루팀", "type": "translation"},
            {"alias": "청색팀", "type": "translation"},
            {"alias": "방어팀", "type": "synonym"},
        ],
    },
]


async def create_registry_with_defaults(
    neo4j_client: Any,
    embedding_provider: Any | None = None,
    include_tech_entities: bool = True,
) -> CanonicalEntityRegistry:
    """
    Create a registry and optionally populate with common entities.

    Args:
        neo4j_client: Neo4j client
        embedding_provider: Optional embedding provider
        include_tech_entities: Whether to import common tech entities

    Returns:
        Initialized registry
    """
    registry = CanonicalEntityRegistry(
        neo4j_client=neo4j_client,
        embedding_provider=embedding_provider,
    )

    await registry.initialize_schema()

    if include_tech_entities:
        await registry.bulk_import(COMMON_TECH_ENTITIES)

    return registry

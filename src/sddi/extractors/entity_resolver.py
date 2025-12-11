"""
Entity Resolution Module.

Multi-stage entity resolution with:
- Exact match (normalized strings)
- Fuzzy string matching (Levenshtein distance)
- Embedding similarity (cosine similarity)
- Cross-lingual matching (multilingual embeddings)
- Canonical entity registry integration
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import defaultdict

import structlog
from langchain_core.embeddings import Embeddings

from src.sddi.state import ExtractedEntity, EntityType

logger = structlog.get_logger(__name__)


@dataclass
class ResolutionCandidate:
    """A candidate match during entity resolution."""
    entity: ExtractedEntity
    score: float
    match_type: str  # exact, fuzzy, embedding, cross_lingual


@dataclass
class ResolutionResult:
    """Result of entity resolution."""
    canonical_entity: ExtractedEntity
    merged_entities: list[ExtractedEntity]
    total_mentions: int
    confidence: float
    resolution_method: str


@dataclass
class EntityCluster:
    """Cluster of entities that refer to the same real-world entity."""
    canonical: ExtractedEntity
    members: list[ExtractedEntity] = field(default_factory=list)
    all_names: set[str] = field(default_factory=set)
    all_aliases: set[str] = field(default_factory=set)
    total_chunk_ids: set[str] = field(default_factory=set)

    def add_member(self, entity: ExtractedEntity) -> None:
        """Add an entity to this cluster."""
        self.members.append(entity)
        self.all_names.add(entity.name.lower())
        self.all_aliases.update(a.lower() for a in entity.aliases)
        self.total_chunk_ids.update(entity.chunk_ids)

    def to_merged_entity(self) -> ExtractedEntity:
        """Create a single merged entity from the cluster."""
        # Collect all unique names and aliases
        all_variations = self.all_names | self.all_aliases
        all_variations.discard(self.canonical.name.lower())

        # Merge properties from all members
        merged_properties = {}
        for member in self.members:
            merged_properties.update(member.properties)

        # Take the best description
        best_description = self.canonical.description
        for member in self.members:
            if len(member.description) > len(best_description):
                best_description = member.description

        # Calculate aggregate confidence
        confidences = [self.canonical.confidence] + [m.confidence for m in self.members]
        avg_confidence = sum(confidences) / len(confidences)

        return ExtractedEntity(
            id=self.canonical.id,
            name=self.canonical.name,
            type=self.canonical.type,
            description=best_description,
            aliases=list(all_variations),
            chunk_ids=list(self.total_chunk_ids),
            confidence=min(avg_confidence * 1.1, 1.0),  # Boost confidence for merged entities
            properties={
                **merged_properties,
                "merged_count": len(self.members) + 1,
                "original_names": list(self.all_names),
            },
        )


class EntityResolver:
    """
    Multi-stage entity resolution system.

    Resolution stages:
    1. Exact match (case-insensitive, normalized)
    2. Fuzzy string matching (Levenshtein distance)
    3. Embedding similarity (cosine similarity)
    4. Cross-lingual matching (for multilingual entities)
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        fuzzy_threshold: float = 0.85,
        embedding_threshold: float = 0.90,
        cross_lingual_threshold: float = 0.88,
        enable_fuzzy: bool = True,
        enable_embedding: bool = True,
        enable_cross_lingual: bool = True,
    ) -> None:
        """
        Initialize entity resolver.

        Args:
            embeddings: Embedding model for semantic similarity
            fuzzy_threshold: Minimum fuzzy match score (0-1)
            embedding_threshold: Minimum embedding similarity (0-1)
            cross_lingual_threshold: Minimum cross-lingual similarity (0-1)
            enable_fuzzy: Enable fuzzy string matching
            enable_embedding: Enable embedding-based matching
            enable_cross_lingual: Enable cross-lingual matching
        """
        self._embeddings = embeddings
        self._fuzzy_threshold = fuzzy_threshold
        self._embedding_threshold = embedding_threshold
        self._cross_lingual_threshold = cross_lingual_threshold
        self._enable_fuzzy = enable_fuzzy
        self._enable_embedding = enable_embedding
        self._enable_cross_lingual = enable_cross_lingual

        # Cache for embeddings
        self._embedding_cache: dict[str, list[float]] = {}

        # Statistics
        self._stats = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "embedding_matches": 0,
            "cross_lingual_matches": 0,
            "total_resolved": 0,
            "total_input": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Get resolution statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics."""
        for key in self._stats:
            self._stats[key] = 0

    # =========================================================================
    # String Normalization
    # =========================================================================

    def _normalize_string(self, s: str) -> str:
        """Normalize string for comparison."""
        # Lowercase
        normalized = s.lower().strip()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove common suffixes/prefixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        return normalized

    def _normalize_for_id(self, name: str, entity_type: str) -> str:
        """Generate normalized ID key."""
        normalized = self._normalize_string(name)
        return f"{entity_type}:{normalized}"

    def _get_all_name_variations(self, entity: ExtractedEntity) -> set[str]:
        """Get all normalized name variations for an entity."""
        variations = {self._normalize_string(entity.name)}
        for alias in entity.aliases:
            variations.add(self._normalize_string(alias))
        return variations

    # =========================================================================
    # Fuzzy String Matching
    # =========================================================================

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Calculate fuzzy similarity score (0-1)."""
        s1_norm = self._normalize_string(s1)
        s2_norm = self._normalize_string(s2)

        if s1_norm == s2_norm:
            return 1.0

        max_len = max(len(s1_norm), len(s2_norm))
        if max_len == 0:
            return 1.0

        distance = self._levenshtein_distance(s1_norm, s2_norm)
        return 1.0 - (distance / max_len)

    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity (better for short strings)."""
        s1_norm = self._normalize_string(s1)
        s2_norm = self._normalize_string(s2)

        if s1_norm == s2_norm:
            return 1.0

        len1, len2 = len(s1_norm), len(s2_norm)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Jaro similarity
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1_norm[i] != s2_norm[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1_norm[i] != s2_norm[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

        # Winkler modification
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1_norm[i] == s2_norm[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)

    def _best_fuzzy_score(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> float:
        """Get best fuzzy match score between two entities."""
        names1 = self._get_all_name_variations(entity1)
        names2 = self._get_all_name_variations(entity2)

        best_score = 0.0
        for n1 in names1:
            for n2 in names2:
                # Use both algorithms and take the best
                levenshtein_score = self._fuzzy_similarity(n1, n2)
                jaro_score = self._jaro_winkler_similarity(n1, n2)
                score = max(levenshtein_score, jaro_score)
                best_score = max(best_score, score)

        return best_score

    # =========================================================================
    # Embedding-based Matching
    # =========================================================================

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text (with caching)."""
        if self._embeddings is None:
            return None

        cache_key = text.lower().strip()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            embedding = await self._embeddings.aembed_query(text)
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning("Failed to get embedding", text=text[:50], error=str(e))
            return None

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _embedding_similarity(
        self,
        entity1: ExtractedEntity,
        entity2: ExtractedEntity,
    ) -> float:
        """Calculate embedding-based similarity between entities."""
        if self._embeddings is None:
            return 0.0

        # Create rich text representation for embedding
        text1 = f"{entity1.name} ({entity1.type.value}): {entity1.description}"
        text2 = f"{entity2.name} ({entity2.type.value}): {entity2.description}"

        emb1 = await self._get_embedding(text1)
        emb2 = await self._get_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        return self._cosine_similarity(emb1, emb2)

    # =========================================================================
    # Cross-lingual Matching
    # =========================================================================

    def _is_likely_cross_lingual_pair(self, name1: str, name2: str) -> bool:
        """Check if two names are likely cross-lingual variants."""
        # Check if one is primarily ASCII and other is not
        ascii_ratio1 = sum(1 for c in name1 if ord(c) < 128) / max(len(name1), 1)
        ascii_ratio2 = sum(1 for c in name2 if ord(c) < 128) / max(len(name2), 1)

        # One primarily ASCII, other primarily non-ASCII
        return (ascii_ratio1 > 0.8 and ascii_ratio2 < 0.5) or (ascii_ratio2 > 0.8 and ascii_ratio1 < 0.5)

    async def _cross_lingual_similarity(
        self,
        entity1: ExtractedEntity,
        entity2: ExtractedEntity,
    ) -> float:
        """
        Calculate cross-lingual similarity.

        Uses multilingual embeddings to match entities across languages.
        e.g., "레드팀" ↔ "Red Team" ↔ "적색팀"
        """
        if self._embeddings is None:
            return 0.0

        # Check if this is a potential cross-lingual pair
        if not self._is_likely_cross_lingual_pair(entity1.name, entity2.name):
            return 0.0

        # Must be same entity type for cross-lingual match
        if entity1.type != entity2.type:
            return 0.0

        # Use embedding similarity for cross-lingual matching
        return await self._embedding_similarity(entity1, entity2)

    # =========================================================================
    # Main Resolution Logic
    # =========================================================================

    async def resolve(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """
        Resolve and deduplicate a list of entities.

        Args:
            entities: List of extracted entities

        Returns:
            List of resolved/merged entities
        """
        if not entities:
            return []

        self._stats["total_input"] += len(entities)

        logger.info(
            "Starting entity resolution",
            input_count=len(entities),
            fuzzy_enabled=self._enable_fuzzy,
            embedding_enabled=self._enable_embedding,
            cross_lingual_enabled=self._enable_cross_lingual,
        )

        # Group entities by type for more efficient matching
        entities_by_type: dict[EntityType, list[ExtractedEntity]] = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)

        # Resolve each type group separately
        resolved_entities: list[ExtractedEntity] = []
        for entity_type, type_entities in entities_by_type.items():
            type_resolved = await self._resolve_type_group(type_entities)
            resolved_entities.extend(type_resolved)

        self._stats["total_resolved"] += len(resolved_entities)

        logger.info(
            "Entity resolution completed",
            input_count=len(entities),
            output_count=len(resolved_entities),
            reduction=f"{(1 - len(resolved_entities) / max(len(entities), 1)) * 100:.1f}%",
            stats=self._stats,
        )

        return resolved_entities

    async def _resolve_type_group(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Resolve entities within a single type group."""
        if len(entities) <= 1:
            return entities

        # Build clusters
        clusters: list[EntityCluster] = []
        processed: set[str] = set()

        for entity in entities:
            if entity.id in processed:
                continue

            # Find or create cluster for this entity
            cluster = await self._find_or_create_cluster(entity, clusters, entities, processed)
            if cluster not in clusters:
                clusters.append(cluster)

            processed.add(entity.id)

        # Convert clusters to merged entities
        return [cluster.to_merged_entity() for cluster in clusters]

    async def _find_or_create_cluster(
        self,
        entity: ExtractedEntity,
        existing_clusters: list[EntityCluster],
        all_entities: list[ExtractedEntity],
        processed: set[str],
    ) -> EntityCluster:
        """Find matching cluster or create new one."""
        entity_names = self._get_all_name_variations(entity)

        # Stage 1: Exact match with existing clusters
        for cluster in existing_clusters:
            if entity_names & cluster.all_names or entity_names & cluster.all_aliases:
                cluster.add_member(entity)
                self._stats["exact_matches"] += 1
                return cluster

        # Stage 2: Fuzzy match with existing clusters
        if self._enable_fuzzy:
            for cluster in existing_clusters:
                score = self._best_fuzzy_score(entity, cluster.canonical)
                if score >= self._fuzzy_threshold:
                    cluster.add_member(entity)
                    self._stats["fuzzy_matches"] += 1
                    logger.debug(
                        "Fuzzy match found",
                        entity=entity.name,
                        canonical=cluster.canonical.name,
                        score=score,
                    )
                    return cluster

        # Stage 3: Embedding similarity with existing clusters
        if self._enable_embedding and self._embeddings:
            for cluster in existing_clusters:
                score = await self._embedding_similarity(entity, cluster.canonical)
                if score >= self._embedding_threshold:
                    cluster.add_member(entity)
                    self._stats["embedding_matches"] += 1
                    logger.debug(
                        "Embedding match found",
                        entity=entity.name,
                        canonical=cluster.canonical.name,
                        score=score,
                    )
                    return cluster

        # Stage 4: Cross-lingual matching
        if self._enable_cross_lingual and self._embeddings:
            for cluster in existing_clusters:
                score = await self._cross_lingual_similarity(entity, cluster.canonical)
                if score >= self._cross_lingual_threshold:
                    cluster.add_member(entity)
                    self._stats["cross_lingual_matches"] += 1
                    logger.info(
                        "Cross-lingual match found",
                        entity=entity.name,
                        canonical=cluster.canonical.name,
                        score=score,
                    )
                    return cluster

        # No match found - create new cluster
        new_cluster = EntityCluster(canonical=entity)
        new_cluster.all_names.add(self._normalize_string(entity.name))
        new_cluster.all_aliases.update(self._normalize_string(a) for a in entity.aliases)
        new_cluster.total_chunk_ids.update(entity.chunk_ids)

        return new_cluster

    # =========================================================================
    # Batch Resolution with Progress
    # =========================================================================

    async def resolve_with_progress(
        self,
        entities: list[ExtractedEntity],
        progress_callback: Callable | None = None,
    ) -> list[ExtractedEntity]:
        """
        Resolve entities with progress callback.

        Args:
            entities: List of entities to resolve
            progress_callback: Optional callback(progress: float, message: str)

        Returns:
            Resolved entity list
        """
        if progress_callback:
            progress_callback(0.0, "Starting entity resolution...")

        # Pre-compute embeddings if enabled
        if self._enable_embedding and self._embeddings:
            if progress_callback:
                progress_callback(0.1, "Computing entity embeddings...")

            for i, entity in enumerate(entities):
                text = f"{entity.name} ({entity.type.value}): {entity.description}"
                await self._get_embedding(text)

                if progress_callback and (i + 1) % 10 == 0:
                    progress = 0.1 + (i / len(entities)) * 0.4
                    progress_callback(progress, f"Computed embeddings for {i + 1}/{len(entities)} entities")

        if progress_callback:
            progress_callback(0.5, "Resolving entity clusters...")

        result = await self.resolve(entities)

        if progress_callback:
            progress_callback(1.0, f"Resolution complete: {len(entities)} -> {len(result)} entities")

        return result

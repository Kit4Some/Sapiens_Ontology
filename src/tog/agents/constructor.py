"""
Constructor Agent for MACER.

Responsible for:
1. Topic Entity extraction from questions
2. Initial entity retrieval via Vector + Full-text search (with fuzzy matching)
3. Seed SubGraph construction with adaptive hop depth (1-3 hops based on question type)
4. Sub-question decomposition for multi-hop questions
5. Bridge entity detection for connecting topic entities
"""

import hashlib
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.tog.prompts import SEED_SUBGRAPH_PROMPT, TOPIC_ENTITY_EXTRACTION_PROMPT
from src.tog.state import (
    MACERState,
    QuestionType,
    SubGraph,
    SubGraphEdge,
    SubGraphNode,
    TopicEntity,
    is_multihop_question,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Fuzzy Matching Utilities (Korean + English Support)
# =============================================================================


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
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


def get_character_ngrams(text: str, n: int = 2) -> set[str]:
    """
    Generate character n-grams from text.

    Better for Korean text matching since Korean characters are syllable blocks.
    """
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """
    Calculate similarity using character n-grams (Jaccard similarity).

    More suitable for Korean text than Levenshtein distance.
    """
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    if s1_lower == s2_lower:
        return 1.0

    if not s1_lower or not s2_lower:
        return 0.0

    ngrams1 = get_character_ngrams(s1_lower, n)
    ngrams2 = get_character_ngrams(s2_lower, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def is_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    return any("\uac00" <= char <= "\ud7a3" or "\u1100" <= char <= "\u11ff" for char in text)


# Type compatibility mapping for semantic validation
ENTITY_TYPE_COMPATIBILITY: dict[str, set[str]] = {
    # Person-related types
    "PERSON": {"PERSON", "ARTIST", "MUSICIAN", "BAND_MEMBER", "ATHLETE", "ACTOR", "WRITER", "POLITICIAN", "SCIENTIST"},
    "ARTIST": {"ARTIST", "PERSON", "MUSICIAN", "ACTOR", "WRITER"},
    "MUSICIAN": {"MUSICIAN", "PERSON", "ARTIST", "BAND_MEMBER", "SINGER", "COMPOSER"},
    "ATHLETE": {"ATHLETE", "PERSON", "PLAYER", "SPORTSPERSON"},
    # Organization-related types
    "ORGANIZATION": {"ORGANIZATION", "COMPANY", "BAND", "GROUP", "INSTITUTION", "AGENCY"},
    "BAND": {"BAND", "MUSICAL_GROUP", "ORGANIZATION", "GROUP"},
    "COMPANY": {"COMPANY", "ORGANIZATION", "CORPORATION", "BUSINESS"},
    # Location-related types
    "LOCATION": {"LOCATION", "CITY", "COUNTRY", "PLACE", "TOWN", "REGION", "STATE", "AREA"},
    "CITY": {"CITY", "LOCATION", "TOWN", "MUNICIPALITY", "PLACE"},
    "COUNTRY": {"COUNTRY", "LOCATION", "NATION", "STATE"},
    "PLACE": {"PLACE", "LOCATION", "CITY", "TOWN", "REGION", "AREA"},
    # Work-related types
    "WORK": {"WORK", "ALBUM", "SONG", "MOVIE", "BOOK", "PRODUCT"},
    "ALBUM": {"ALBUM", "WORK", "MUSIC_ALBUM", "RECORD"},
    "SONG": {"SONG", "WORK", "TRACK", "MUSIC"},
}


def is_semantically_compatible(topic_type: str, candidate_type: str) -> bool:
    """
    Check if candidate entity type is semantically compatible with topic type.

    This prevents matching completely unrelated entities (e.g., a PERSON query
    matching an ORGANIZATION entity via fuzzy string matching).

    Args:
        topic_type: Type of the topic entity being searched for
        candidate_type: Type of the candidate entity found in database

    Returns:
        True if types are compatible, False otherwise
    """
    if not topic_type or not candidate_type:
        return True  # Allow if type info is missing

    topic_upper = topic_type.upper().strip()
    candidate_upper = candidate_type.upper().strip()

    # Exact match
    if topic_upper == candidate_upper:
        return True

    # Check compatibility mapping
    compatible_set = ENTITY_TYPE_COMPATIBILITY.get(topic_upper, set())
    if candidate_upper in compatible_set:
        return True

    # Reverse check - candidate might be a broader category
    candidate_compatible_set = ENTITY_TYPE_COMPATIBILITY.get(candidate_upper, set())
    if topic_upper in candidate_compatible_set:
        return True

    # Generic types are always compatible
    generic_types = {"ENTITY", "CONCEPT", "THING", "UNKNOWN", "OTHER"}
    return topic_upper in generic_types or candidate_upper in generic_types


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity ratio between two strings (0.0-1.0).

    Uses n-gram similarity for Korean, Levenshtein for others.
    """
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    if s1_lower == s2_lower:
        return 1.0

    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 1.0

    # Use n-gram similarity for Korean text (better results)
    if is_korean(s1) or is_korean(s2):
        return ngram_similarity(s1_lower, s2_lower, n=2)

    # Use Levenshtein for non-Korean
    distance = levenshtein_distance(s1_lower, s2_lower)
    return 1.0 - (distance / max_len)


def find_best_match(
    query: str,
    candidates: list[dict[str, Any]],
    name_key: str = "name",
    min_similarity: float = 0.5,
) -> list[tuple[dict[str, Any], float]]:
    """
    Find best matching candidates using fuzzy matching.

    Args:
        query: Query string to match
        candidates: List of candidate dicts
        name_key: Key to use for matching
        min_similarity: Minimum similarity threshold (lowered for better recall)

    Returns:
        List of (candidate, similarity_score) tuples, sorted by score
    """
    matches = []
    query_lower = query.lower().strip()

    for candidate in candidates:
        name = str(candidate.get(name_key, "")).lower().strip()
        if not name:
            continue

        # Check exact containment first (high priority)
        if query_lower in name or name in query_lower:
            matches.append((candidate, 0.95))
            continue

        # Check if query words are contained in name (for multi-word queries)
        query_words = query_lower.split()
        if len(query_words) > 1:
            word_matches = sum(1 for w in query_words if w in name)
            if word_matches >= len(query_words) * 0.5:
                matches.append((candidate, 0.8))
                continue

        # Calculate fuzzy similarity
        similarity = string_similarity(query, name)
        if similarity >= min_similarity:
            matches.append((candidate, similarity))

        # Also check description if available
        description = str(candidate.get("description", "")).lower().strip()
        if description and query_lower in description and not any(c[0] == candidate for c in matches):
            matches.append((candidate, 0.7))

    # Sort by similarity descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


class ConstructorAgent:
    """
    Constructor Agent for MACER framework.

    Builds the initial reasoning context by:
    1. Extracting topic entities from the question
    2. Retrieving matching entities from Neo4j (fulltext + vector + fuzzy)
    3. Constructing a seed subgraph with adaptive hop depth (1-3 hops)
    4. Decomposing multi-hop questions into sub-questions
    5. Detecting bridge entities for multi-hop reasoning
    """

    # Hop depth configuration based on question type
    HOP_DEPTH_CONFIG = {
        QuestionType.FACTOID.value: 1,
        QuestionType.YESNO.value: 1,
        QuestionType.AGGREGATION.value: 2,
        QuestionType.COMPARISON.value: 2,
        QuestionType.MULTIHOP.value: 3,
        QuestionType.BRIDGE.value: 3,
    }

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Any | None = None,
        neo4j_client: OntologyGraphClient | None = None,
        max_topic_entities: int = 5,
        max_seed_nodes: int = 100,  # Increased for deeper traversal
        include_community_context: bool = True,
        enable_deep_traversal: bool = True,
    ) -> None:
        """
        Initialize the Constructor agent.

        Args:
            llm: LangChain chat model
            embeddings: Embedding model for vector search (optional but recommended)
            neo4j_client: Neo4j client for graph operations
            max_topic_entities: Maximum topic entities to extract
            max_seed_nodes: Maximum nodes in seed subgraph
            include_community_context: Whether to include community info
            enable_deep_traversal: Enable adaptive multi-hop traversal
        """
        self._llm = llm
        self._embeddings = embeddings
        self._client = neo4j_client or get_ontology_client()
        self._max_topic_entities = max_topic_entities
        self._max_seed_nodes = max_seed_nodes
        self._include_community = include_community_context
        self._enable_deep_traversal = enable_deep_traversal

        # Build chains
        self._parser = JsonOutputParser()
        self._extraction_chain = TOPIC_ENTITY_EXTRACTION_PROMPT | self._llm | self._parser
        self._subgraph_chain = SEED_SUBGRAPH_PROMPT | self._llm | self._parser

    def _get_hop_depth(self, question_type: str) -> int:
        """
        Determine appropriate hop depth based on question type.

        Args:
            question_type: Type of question

        Returns:
            Number of hops for subgraph traversal
        """
        if not self._enable_deep_traversal:
            return 1
        return self.HOP_DEPTH_CONFIG.get(question_type, 2)

    async def check_data_availability(self) -> dict[str, Any]:
        """
        Check if there is data available in the knowledge graph.

        Returns:
            Dictionary with data availability status and diagnostics
        """
        await self._client.connect()
        diagnostics = await self._client.get_data_diagnostics()
        return diagnostics

    async def extract_topic_entities(self, question: str) -> list[TopicEntity]:
        """
        Extract topic entities from the question using LLM.

        Args:
            question: Natural language question

        Returns:
            List of identified topic entities
        """
        logger.info("Extracting topic entities", question=question[:100])

        try:
            result = await self._extraction_chain.ainvoke({"question": question})
            entities_data = result.get("topic_entities", [])
        except Exception as e:
            logger.error("Topic entity extraction failed", error=str(e))
            return []

        topic_entities = []
        for entity_data in entities_data[: self._max_topic_entities]:
            entity = TopicEntity(
                id=self._generate_temp_id(entity_data.get("name", "")),
                name=entity_data.get("name", ""),
                type=entity_data.get("type", "CONCEPT"),
                relevance_score=1.0 if entity_data.get("is_primary", False) else 0.8,
                source="llm_extraction",
            )
            topic_entities.append(entity)

        logger.info("Topic entities extracted", count=len(topic_entities))
        return topic_entities

    async def retrieve_entities(
        self,
        topic_entities: list[TopicEntity],
        original_query: str = "",
    ) -> list[dict[str, Any]]:
        """
        Retrieve matching entities from Neo4j using multiple strategies:
        1. Vector similarity search (if embeddings available)
        2. Full-text search across all indexes
        3. Exact/partial name matching
        4. Fuzzy matching as fallback

        Args:
            topic_entities: Extracted topic entities
            original_query: Original question (for vector search)

        Returns:
            List of matched entities from database
        """
        await self._client.connect()
        retrieved: list[dict[str, Any]] = []

        # First, check data availability
        diagnostics = await self.check_data_availability()
        if not diagnostics["health"]["has_data"]:
            logger.warning(
                "No data in Neo4j",
                issues=diagnostics["health"]["issues"],
            )
            return []

        entity_count = diagnostics["counts"].get("entities", 0)
        has_embeddings = diagnostics["health"]["has_embeddings"]
        logger.info(
            "Data availability check",
            entity_count=entity_count,
            has_embeddings=has_embeddings,
        )

        # Strategy 1: Vector search on the original query (if embeddings available)
        if self._embeddings and has_embeddings and original_query:
            try:
                logger.info("Attempting vector search", query=original_query[:50])
                query_embedding = await self._embeddings.aembed_query(original_query)

                from src.graph.schema import NodeLabel

                vector_results = await self._client.vector_search(
                    embedding=query_embedding,
                    node_label=NodeLabel.ENTITY,
                    top_k=10,
                    min_score=0.3,
                )

                logger.info("Vector search results", count=len(vector_results))

                for result in vector_results:
                    props = result.properties or {}
                    retrieved.append(
                        {
                            "id": result.node_id,
                            "name": props.get("name", result.node_id),
                            "type": props.get("type", "Entity"),
                            "description": props.get("description", ""),
                            "label": result.node_label,
                            "score": result.score,
                            "source": "vector_search",
                            "topic_entity": original_query,
                        }
                    )
            except Exception as e:
                logger.warning("Vector search failed", error=str(e))

        # Strategy 2 & 3: Fulltext and exact match for each topic entity
        for topic in topic_entities:
            logger.info("Searching for topic entity", topic_name=topic.name, topic_type=topic.type)

            # Try full-text search first
            try:
                fulltext_results = await self._client.fulltext_search(
                    query_text=topic.name,
                    top_k=5,
                    min_score=0.1,  # Lower threshold for better recall
                )

                logger.info(
                    "Fulltext search results",
                    topic=topic.name,
                    results_count=len(fulltext_results),
                )

                for result in fulltext_results:
                    retrieved.append(
                        {
                            "id": result.node_id,
                            "name": result.text,
                            "label": result.node_label,
                            "score": result.score,
                            "source": "fulltext",
                            "topic_entity": topic.name,
                        }
                    )
            except Exception as e:
                logger.warning("Fulltext search failed", entity=topic.name, error=str(e))

            # Also try exact/fuzzy match
            try:
                query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                   OR ANY(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS toLower($name))
                   OR toLower(coalesce(e.description, '')) CONTAINS toLower($name)
                RETURN e.id as id, e.name as name, e.type as type,
                       e.description as description, labels(e)[0] as label
                LIMIT 5
                """
                results = await self._client.execute_cypher(query, {"name": topic.name})

                logger.info(
                    "Entity name match results",
                    topic=topic.name,
                    results_count=len(results),
                )

                for r in results:
                    if not any(existing["id"] == r["id"] for existing in retrieved):
                        retrieved.append(
                            {
                                "id": r["id"],
                                "name": r["name"],
                                "type": r.get("type", "Unknown"),
                                "description": r.get("description", ""),
                                "label": r.get("label", "Entity"),
                                "score": 0.9,
                                "source": "exact_match",
                                "topic_entity": topic.name,
                            }
                        )
            except Exception as e:
                logger.warning("Entity name lookup failed", entity=topic.name, error=str(e))

        # If no entities found, try fuzzy matching against all entities
        if not retrieved and topic_entities:
            logger.info("No exact matches found, trying fuzzy matching")
            try:
                # Get all entity names for fuzzy matching
                all_entities_query = """
                MATCH (e:Entity)
                RETURN e.id as id, e.name as name, e.type as type,
                       e.description as description, labels(e)[0] as label
                LIMIT 100
                """
                all_entities = await self._client.execute_cypher(all_entities_query)

                if all_entities:
                    for topic in topic_entities:
                        # Find fuzzy matches with stricter threshold
                        matches = find_best_match(
                            topic.name,
                            all_entities,
                            name_key="name",
                            min_similarity=0.8,  # 80% similarity threshold (increased from 50%)
                        )

                        if matches:
                            logger.info(
                                "Fuzzy matches found",
                                topic=topic.name,
                                topic_type=topic.type,
                                match_count=len(matches),
                                best_match=matches[0][0].get("name") if matches else None,
                                best_score=matches[0][1] if matches else 0,
                            )

                            valid_matches_count = 0
                            for entity, similarity in matches[:5]:  # Check top 5, take up to 3 valid
                                if valid_matches_count >= 3:
                                    break

                                # Type compatibility check - prevent PERSON matching ORGANIZATION, etc.
                                candidate_type = entity.get("type", "Unknown")
                                if not is_semantically_compatible(topic.type, candidate_type):
                                    logger.debug(
                                        "Skipping fuzzy match due to type incompatibility",
                                        topic=topic.name,
                                        topic_type=topic.type,
                                        candidate=entity.get("name"),
                                        candidate_type=candidate_type,
                                    )
                                    continue

                                if not any(existing["id"] == entity["id"] for existing in retrieved):
                                    retrieved.append(
                                        {
                                            "id": entity["id"],
                                            "name": entity["name"],
                                            "type": candidate_type,
                                            "description": entity.get("description", ""),
                                            "label": entity.get("label", "Entity"),
                                            "score": similarity,
                                            "source": "fuzzy_match",
                                            "topic_entity": topic.name,
                                            "topic_type": topic.type,
                                            "fuzzy_suggestion": f"Did you mean: '{entity['name']}'?",
                                        }
                                    )
                                    valid_matches_count += 1

                            if valid_matches_count == 0 and matches:
                                logger.warning(
                                    "All fuzzy matches rejected due to type incompatibility",
                                    topic=topic.name,
                                    topic_type=topic.type,
                                    rejected_count=len(matches),
                                )
            except Exception as e:
                logger.warning("Fuzzy matching failed", error=str(e))

        # If still no entities found, try searching chunks directly
        if not retrieved and topic_entities:
            logger.info("No entities found, falling back to chunk search")
            try:
                # Search chunks that contain any of the topic entity names
                topic_names = [t.name for t in topic_entities]
                chunk_query = """
                MATCH (c:Chunk)
                WHERE ANY(name IN $names WHERE toLower(c.text) CONTAINS toLower(name))
                RETURN c.id as id, c.text as text, c.source as source
                LIMIT 10
                """
                chunk_results = await self._client.execute_cypher(
                    chunk_query, {"names": topic_names}
                )

                logger.info("Chunk fallback search results", count=len(chunk_results))

                # Extract any entities linked to these chunks
                if chunk_results:
                    chunk_ids = [c["id"] for c in chunk_results]
                    entity_from_chunk_query = """
                    MATCH (c:Chunk)-[:CONTAINS]->(e:Entity)
                    WHERE c.id IN $chunk_ids
                    RETURN DISTINCT e.id as id, e.name as name, e.type as type,
                           e.description as description
                    LIMIT 10
                    """
                    linked_entities = await self._client.execute_cypher(
                        entity_from_chunk_query, {"chunk_ids": chunk_ids}
                    )

                    for r in linked_entities:
                        retrieved.append(
                            {
                                "id": r["id"],
                                "name": r["name"],
                                "type": r.get("type", "Unknown"),
                                "description": r.get("description", ""),
                                "label": "Entity",
                                "score": 0.7,
                                "source": "chunk_link",
                                "topic_entity": topic_names[0] if topic_names else "",
                            }
                        )

                    logger.info("Entities from chunk links", count=len(linked_entities))
            except Exception as e:
                logger.warning("Chunk fallback search failed", error=str(e))

        # Deduplicate by ID
        seen_ids = set()
        unique_retrieved = []
        for entity in retrieved:
            if entity["id"] not in seen_ids:
                seen_ids.add(entity["id"])
                unique_retrieved.append(entity)

        logger.info(
            "Entity retrieval completed",
            total_retrieved=len(unique_retrieved),
            sources={e.get("source") for e in unique_retrieved},
        )
        return unique_retrieved

    async def build_seed_subgraph(
        self,
        retrieved_entities: list[dict[str, Any]],
        question_type: str = QuestionType.FACTOID.value,
    ) -> SubGraph:
        """
        Build seed subgraph with adaptive hop depth based on question type.

        Args:
            retrieved_entities: Retrieved entities from search
            question_type: Type of question for determining hop depth

        Returns:
            Seed SubGraph for reasoning
        """
        if not retrieved_entities:
            return SubGraph()

        nodes: dict[str, SubGraphNode] = {}
        edges: list[SubGraphEdge] = []

        # Determine hop depth based on question type
        max_hops = self._get_hop_depth(question_type)
        logger.info(
            "Building subgraph with adaptive depth",
            question_type=question_type,
            max_hops=max_hops,
        )

        # Add retrieved entities as seed nodes
        seed_entity_ids = []
        for entity in retrieved_entities[: self._max_topic_entities]:
            node = SubGraphNode(
                id=entity["id"],
                name=entity.get("name", "Unknown"),
                type=entity.get("type", entity.get("label", "Entity")),
                properties={
                    "description": entity.get("description", ""),
                    "score": entity.get("score", 1.0),
                },
                distance_from_topic=0,
                relevance_score=entity.get("score", 1.0),
            )
            nodes[node.id] = node
            seed_entity_ids.append(entity["id"])

        # Get multi-hop neighbors for seed entities
        await self._client.connect()

        # BFS traversal for multi-hop neighbors
        current_frontier = seed_entity_ids.copy()
        visited = set(seed_entity_ids)

        for hop in range(1, max_hops + 1):
            if len(nodes) >= self._max_seed_nodes:
                break

            next_frontier = []
            # Limit per-hop exploration based on remaining capacity
            nodes_per_entity = max(3, (self._max_seed_nodes - len(nodes)) // max(len(current_frontier), 1))

            for entity_id in current_frontier:
                if len(nodes) >= self._max_seed_nodes:
                    break

                try:
                    neighbor_result = await self._client.get_neighbors(
                        node_id=entity_id,
                        max_hops=1,
                        limit=nodes_per_entity,
                    )

                    for neighbor in neighbor_result.get("neighbors", []):
                        neighbor_id = neighbor.get("neighbor_id")
                        if not neighbor_id or neighbor_id in visited:
                            continue

                        visited.add(neighbor_id)
                        props = neighbor.get("properties", {})

                        # Calculate relevance score based on hop distance
                        base_relevance = 0.9 - (hop * 0.15)  # Decay per hop
                        node = SubGraphNode(
                            id=neighbor_id,
                            name=props.get("name", neighbor_id),
                            type=props.get("type", neighbor.get("neighbor_label", "Entity")),
                            properties=props,
                            distance_from_topic=hop,
                            relevance_score=max(0.3, base_relevance),
                        )
                        nodes[neighbor_id] = node
                        next_frontier.append(neighbor_id)

                        # Add edges
                        for rel in neighbor.get("relationships", []):
                            edge = SubGraphEdge(
                                source_id=entity_id,
                                target_id=neighbor_id,
                                relation_type=rel.get("type", "RELATED_TO"),
                                predicate=rel.get("properties", {}).get("predicate", ""),
                                properties=rel.get("properties", {}),
                                confidence=1.0,
                            )
                            edges.append(edge)

                except Exception as e:
                    logger.debug("Neighbor retrieval failed", entity_id=entity_id, hop=hop, error=str(e))

            current_frontier = next_frontier
            if not current_frontier:
                break

            logger.debug(
                "Hop traversal completed",
                hop=hop,
                nodes_in_frontier=len(next_frontier),
                total_nodes=len(nodes),
            )

        # For multi-hop questions, try to find bridge entities connecting topic entities
        if max_hops >= 2 and len(seed_entity_ids) >= 2:
            await self._find_bridge_entities(seed_entity_ids, nodes, edges)

        # Include community context if enabled
        if self._include_community and seed_entity_ids:
            await self._add_community_context(list(nodes.keys())[:20], nodes, edges)

        # Determine center entity
        center_id = seed_entity_ids[0] if seed_entity_ids else None

        subgraph = SubGraph(
            nodes=list(nodes.values()),
            edges=edges,
            center_entity_id=center_id,
        )

        logger.info(
            "Seed subgraph built",
            node_count=subgraph.node_count(),
            edge_count=subgraph.edge_count(),
            max_hops_used=max_hops,
            bridge_search=max_hops >= 2,
        )

        return subgraph

    async def _find_bridge_entities(
        self,
        seed_entity_ids: list[str],
        nodes: dict[str, SubGraphNode],
        edges: list[SubGraphEdge],
    ) -> None:
        """
        Find bridge entities that connect multiple topic entities.

        Bridge entities are crucial for multi-hop reasoning as they
        link different parts of the question.

        Args:
            seed_entity_ids: IDs of seed/topic entities
            nodes: Current node dictionary (modified in place)
            edges: Current edge list (modified in place)
        """
        try:
            # Find shortest paths between all pairs of topic entities
            for i, start_id in enumerate(seed_entity_ids):
                for end_id in seed_entity_ids[i + 1:]:
                    # Query for shortest path
                    path_query = """
                    MATCH path = shortestPath((start:Entity {id: $start_id})-[*1..4]-(end:Entity {id: $end_id}))
                    WHERE start <> end
                    UNWIND nodes(path) as n
                    WITH n WHERE n.id <> $start_id AND n.id <> $end_id
                    RETURN DISTINCT
                        n.id as id,
                        n.name as name,
                        n.type as type,
                        n.description as description
                    LIMIT 5
                    """
                    try:
                        bridge_results = await self._client.execute_cypher(
                            path_query,
                            {"start_id": start_id, "end_id": end_id},
                        )

                        for result in bridge_results:
                            bridge_id = result.get("id")
                            if bridge_id and bridge_id not in nodes:
                                node = SubGraphNode(
                                    id=bridge_id,
                                    name=result.get("name", bridge_id),
                                    type=result.get("type", "Entity"),
                                    properties={
                                        "description": result.get("description", ""),
                                        "is_bridge": True,
                                        "connects": [start_id, end_id],
                                    },
                                    distance_from_topic=1,  # Bridge entities are close to topics
                                    relevance_score=0.85,  # High relevance for bridges
                                )
                                nodes[bridge_id] = node
                                logger.debug(
                                    "Bridge entity found",
                                    bridge=result.get("name"),
                                    connects=(start_id, end_id),
                                )
                    except Exception as e:
                        logger.debug(
                            "Bridge path query failed",
                            start=start_id,
                            end=end_id,
                            error=str(e),
                        )

            # Query edges between all nodes
            node_ids = list(nodes.keys())
            if len(node_ids) > 1:
                edge_query = """
                MATCH (a)-[r]->(b)
                WHERE a.id IN $ids AND b.id IN $ids AND a.id <> b.id
                RETURN DISTINCT
                    a.id as source,
                    b.id as target,
                    type(r) as rel_type,
                    properties(r) as props
                """
                try:
                    edge_results = await self._client.execute_cypher(
                        edge_query,
                        {"ids": node_ids},
                    )
                    existing_edges = {(e.source_id, e.target_id, e.relation_type) for e in edges}
                    for result in edge_results:
                        edge_key = (result["source"], result["target"], result["rel_type"])
                        if edge_key not in existing_edges:
                            edge = SubGraphEdge(
                                source_id=result["source"],
                                target_id=result["target"],
                                relation_type=result["rel_type"],
                                predicate=result.get("props", {}).get("predicate", ""),
                                properties=result.get("props", {}),
                                confidence=1.0,
                            )
                            edges.append(edge)
                            existing_edges.add(edge_key)
                except Exception as e:
                    logger.debug("Edge completion query failed", error=str(e))

        except Exception as e:
            logger.debug("Bridge entity search failed", error=str(e))

    async def _add_community_context(
        self,
        entity_ids: list[str],
        nodes: dict[str, SubGraphNode],
        edges: list[SubGraphEdge],
    ) -> None:
        """Add community information to subgraph."""
        try:
            query = """
            MATCH (e:Entity)-[:BELONGS_TO]->(c:Community)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id as id, c.summary as summary, c.level as level,
                   c.member_count as member_count, collect(e.id) as member_entities
            LIMIT 5
            """
            communities = await self._client.execute_cypher(query, {"entity_ids": entity_ids})

            for comm in communities:
                comm_id = comm.get("id")
                if comm_id and comm_id not in nodes:
                    node = SubGraphNode(
                        id=comm_id,
                        name=f"Community L{comm.get('level', 0)}",
                        type="Community",
                        properties={
                            "summary": comm.get("summary", ""),
                            "level": comm.get("level", 0),
                            "member_count": comm.get("member_count", 0),
                        },
                        distance_from_topic=1,
                        relevance_score=0.6,
                    )
                    nodes[comm_id] = node

                    # Add edges to member entities
                    for member_id in comm.get("member_entities", []):
                        edge = SubGraphEdge(
                            source_id=member_id,
                            target_id=comm_id,
                            relation_type="BELONGS_TO",
                            predicate="belongs_to",
                            confidence=1.0,
                        )
                        edges.append(edge)

        except Exception as e:
            logger.debug("Community context retrieval failed", error=str(e))

    async def decompose_question(
        self,
        question: str,
        question_type: str,
        topic_entities: list[TopicEntity],
    ) -> list[str]:
        """
        Decompose complex questions into sub-questions for multi-hop reasoning.

        Args:
            question: Original question
            question_type: Type of question
            topic_entities: Extracted topic entities

        Returns:
            List of sub-questions (empty if decomposition not needed)
        """
        # Only decompose multi-hop, comparison, and bridge questions
        if question_type not in [
            QuestionType.MULTIHOP.value,
            QuestionType.COMPARISON.value,
            QuestionType.BRIDGE.value,
        ]:
            return []

        try:
            decompose_prompt = f"""Decompose this question into simpler sub-questions that can be answered step by step.
Each sub-question should focus on finding a single piece of information.

Question: {question}

Topic entities found: {[e.name for e in topic_entities]}

Return JSON with format:
{{
    "sub_questions": ["sub-question 1", "sub-question 2", ...],
    "reasoning": "brief explanation of the decomposition"
}}

Rules:
- For comparison questions: create sub-questions to get properties of each entity being compared
- For multi-hop questions: break into steps following the reasoning chain
- For bridge questions: identify the intermediate entity needed
- Keep sub-questions simple and focused on one fact each
- Maximum 4 sub-questions
"""
            result = await self._llm.ainvoke(decompose_prompt)
            content = result.content if hasattr(result, "content") else str(result)

            # Parse JSON from response
            import json
            import re

            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                sub_questions = parsed.get("sub_questions", [])[:4]  # Limit to 4
                logger.info(
                    "Question decomposed",
                    original=question[:50],
                    sub_question_count=len(sub_questions),
                )
                return sub_questions
        except Exception as e:
            logger.debug("Question decomposition failed", error=str(e))

        return []

    async def construct(self, state: MACERState) -> dict[str, Any]:
        """
        Main construction method for MACER pipeline.

        Enhanced with multi-hop reasoning support:
        - Adaptive hop depth based on question type
        - Sub-question decomposition for complex questions
        - Bridge entity detection

        Args:
            state: Current MACER state

        Returns:
            State updates with topic entities, seed subgraph, and sub-questions
        """
        question = state.get("original_query", "")
        question_type = state.get("question_type", QuestionType.FACTOID.value)

        logger.info(
            "Constructor agent starting",
            question=question[:100],
            question_type=question_type,
        )

        # Step 0: Check data availability first
        diagnostics = await self.check_data_availability()
        has_data = diagnostics["health"]["has_data"]

        if not has_data:
            logger.warning(
                "No data in knowledge graph - early termination",
                issues=diagnostics["health"]["issues"],
            )
            return {
                "topic_entities": [],
                "retrieved_entities": [],
                "current_subgraph": SubGraph(),
                "subgraph_history": [],
                "current_query": question,
                "sub_questions": [],
                "iteration": 0,
                "should_terminate": True,
                "metadata": {
                    "no_data": True,
                    "diagnostics": diagnostics,
                    "message": "Knowledge graph is empty. Please ingest documents first.",
                },
            }

        # Step 1: Extract topic entities
        topic_entities = await self.extract_topic_entities(question)

        # Step 2: Retrieve matching entities (with vector search support)
        retrieved_entities = await self.retrieve_entities(
            topic_entities,
            original_query=question,
        )

        # Update topic entities with resolved IDs (immutably)
        updated_topic_entities = []
        for topic in topic_entities:
            resolved_id = None
            for retrieved in retrieved_entities:
                if retrieved.get("topic_entity") == topic.name:
                    resolved_id = retrieved["id"]
                    break
            if resolved_id and resolved_id != topic.id:
                # Create new TopicEntity with updated ID using model_copy
                updated_topic = topic.model_copy(update={"id": resolved_id})
                updated_topic_entities.append(updated_topic)
            else:
                updated_topic_entities.append(topic)
        topic_entities = updated_topic_entities

        # Step 3: Decompose question if needed for multi-hop reasoning
        sub_questions = await self.decompose_question(
            question, question_type, topic_entities
        )

        # Step 4: Build seed subgraph with adaptive hop depth
        subgraph = await self.build_seed_subgraph(
            retrieved_entities,
            question_type=question_type,
        )

        # Determine if we have enough data to proceed
        no_results = len(retrieved_entities) == 0 and subgraph.node_count() == 0

        # Check subgraph quality for multi-hop questions
        subgraph_metrics = subgraph.get_subgraph_metrics()
        is_multihop = is_multihop_question(question)

        logger.info(
            "Constructor completed",
            topic_entities=len(topic_entities),
            retrieved=len(retrieved_entities),
            subgraph_nodes=subgraph.node_count(),
            subgraph_edges=subgraph.edge_count(),
            max_path_length=subgraph_metrics.get("max_path_length", 0),
            bridge_entities=subgraph_metrics.get("bridge_entity_count", 0),
            sub_questions=len(sub_questions),
            no_results=no_results,
        )

        return {
            "topic_entities": topic_entities,
            "retrieved_entities": retrieved_entities,
            "current_subgraph": subgraph,
            "subgraph_history": [subgraph] if subgraph.node_count() > 0 else [],
            "current_query": question,
            "sub_questions": sub_questions,
            "iteration": 0,
            "metadata": {
                "no_data": False,
                "no_results": no_results,
                "question_type": question_type,
                "is_multihop": is_multihop,
                "subgraph_metrics": subgraph_metrics,
                "diagnostics_summary": {
                    "entity_count": diagnostics["counts"].get("entities", 0),
                    "has_embeddings": diagnostics["health"]["has_embeddings"],
                },
            },
        }

    def _generate_temp_id(self, name: str) -> str:
        """Generate temporary ID for unresolved entities."""
        return f"temp_{hashlib.sha256(name.encode()).hexdigest()[:8]}"

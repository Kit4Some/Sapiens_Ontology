"""
Retriever Agent for MACER.

Responsible for:
1. Vector Similarity Search
2. Multi-hop Graph Traversal
3. Community Summary Search
4. Text2Cypher Natural Language Queries
5. Evidence Collection and Ranking
6. Path-based Evidence Extraction for Multi-hop Reasoning
7. Reasoning Chain Detection and Tracking
8. Evidence-Grounded 4-Component Scoring
"""

import hashlib
import re
import uuid
from typing import Any

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.graph.schema import NodeLabel
from src.text2cypher import CypherGenerationResult, Text2CypherGenerator
from src.tog.prompts import (
    ENTITY_EXTRACTION_FOR_SCORING_PROMPT,
    EVIDENCE_GROUNDED_RETRIEVAL_PROMPT,
    EVIDENCE_RANKING_PROMPT,
    RETRIEVAL_STRATEGY_PROMPT,
)
from src.tog.state import (
    Evidence,
    EvidenceConfidenceType,
    EvidenceType,
    MACERState,
    QuestionType,
    SubGraph,
    SubGraphEdge,
    SubGraphNode,
)
from src.tog.temporal_reasoning import (
    TemporalReasoningEngine,
    compute_enhanced_temporal_alignment,
    get_temporal_engine,
)
from src.tog.negative_evidence import (
    NegativeEvidenceScorer,
    get_negative_evidence_scorer,
    analyze_evidence_polarity,
    EvidencePolarity,
)

logger = structlog.get_logger(__name__)


class RetrievalStrategy:
    """Encapsulates retrieval strategy decision."""

    VECTOR_SEARCH = "VECTOR_SEARCH"
    GRAPH_TRAVERSAL = "GRAPH_TRAVERSAL"
    COMMUNITY_SEARCH = "COMMUNITY_SEARCH"
    TEXT2CYPHER = "TEXT2CYPHER"
    HYBRID = "HYBRID"


class RetrieverAgent:
    """
    Retriever Agent for MACER framework.

    Retrieves evidence from the knowledge graph using multiple strategies:
    - Vector similarity for semantic search
    - Graph traversal for structural exploration
    - Community summaries for high-level context
    - Text2Cypher for natural language to Cypher translation
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings | None = None,
        neo4j_client: OntologyGraphClient | None = None,
        max_evidence_per_iteration: int = 10,
        max_traversal_depth: int = 3,
        enable_text2cypher: bool = True,
    ) -> None:
        """
        Initialize the Retriever agent.

        Args:
            llm: LangChain chat model
            embeddings: Embedding model for vector search
            neo4j_client: Neo4j client
            max_evidence_per_iteration: Max evidence pieces to collect per iteration
            max_traversal_depth: Maximum graph traversal depth
            enable_text2cypher: Whether to enable Text2Cypher retrieval
        """
        self._llm = llm
        self._embeddings = embeddings
        self._client = neo4j_client or get_ontology_client()
        self._max_evidence = max_evidence_per_iteration
        self._max_depth = max_traversal_depth
        self._enable_text2cypher = enable_text2cypher

        # Build chains
        self._parser = JsonOutputParser()
        self._strategy_chain = RETRIEVAL_STRATEGY_PROMPT | self._llm | self._parser
        self._ranking_chain = EVIDENCE_RANKING_PROMPT | self._llm | self._parser
        self._evidence_grounded_chain = EVIDENCE_GROUNDED_RETRIEVAL_PROMPT | self._llm | self._parser
        self._entity_extraction_chain = ENTITY_EXTRACTION_FOR_SCORING_PROMPT | self._llm | self._parser

        # Initialize Text2Cypher generator
        self._text2cypher: Text2CypherGenerator | None = None
        if enable_text2cypher:
            self._text2cypher = Text2CypherGenerator(
                llm=llm,
                embeddings=embeddings,
                neo4j_client=self._client,
                enable_entity_resolution=True,
                enable_self_healing=True,
                max_healing_retries=2,
            )

        # Cache for entity extraction to avoid redundant LLM calls
        self._entity_cache: dict[str, set[str]] = {}

    # =========================================================================
    # Evidence-Grounded Scoring Methods (4-Component Scoring)
    # =========================================================================

    def _extract_entities_fast(self, text: str) -> set[str]:
        """
        Fast entity extraction using regex patterns (no LLM call).
        Used for computing entity overlap score.
        """
        if text in self._entity_cache:
            return self._entity_cache[text]

        entities: set[str] = set()

        # Extract capitalized words/phrases (likely named entities)
        # Pattern matches sequences of capitalized words
        capitalized_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        for match in re.findall(capitalized_pattern, text):
            if len(match) > 2:  # Skip very short matches
                entities.add(match.lower())

        # Extract quoted strings
        quoted_pattern = r'"([^"]+)"'
        for match in re.findall(quoted_pattern, text):
            entities.add(match.lower())

        # Extract dates and years
        date_pattern = r"\b(\d{4}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2},?\s+\d{4})\b"
        for match in re.findall(date_pattern, text):
            entities.add(match.lower())

        # Extract Korean named entities (sequences of Korean characters)
        korean_pattern = r"[\uAC00-\uD7A3]{2,}"
        for match in re.findall(korean_pattern, text):
            if len(match) >= 2:
                entities.add(match)

        self._entity_cache[text] = entities
        return entities

    def _compute_entity_overlap(self, query: str, evidence_text: str) -> float:
        """
        Compute Jaccard similarity of entities between query and evidence.
        Weight: 40% of final score.
        """
        query_entities = self._extract_entities_fast(query)
        evidence_entities = self._extract_entities_fast(evidence_text)

        if not query_entities or not evidence_entities:
            return 0.0

        intersection = query_entities & evidence_entities
        union = query_entities | evidence_entities

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _compute_relationship_match(self, query: str, evidence_text: str) -> float:
        """
        Check if evidence contains the target relationship.
        Weight: 30% of final score.
        """
        # Common relationship verbs
        relationship_verbs = [
            "founded", "created", "established", "started",
            "played for", "worked at", "employed by", "member of",
            "married", "divorced", "born", "died",
            "surpassed", "exceeded", "broke", "set",
            "won", "lost", "defeated", "beat",
            "located in", "based in", "headquarters",
            "directed", "produced", "wrote", "composed",
            "invented", "discovered", "developed",
            "served as", "acted as", "known as",
            # Korean verbs
            "설립", "창립", "출생", "사망", "수상", "기록",
        ]

        query_lower = query.lower()
        evidence_lower = evidence_text.lower()

        # Find relationship verbs in query
        query_verbs = [v for v in relationship_verbs if v in query_lower]

        if not query_verbs:
            # No specific relationship in query, give partial score
            return 0.5

        # Check if evidence contains any of the same relationship verbs
        for verb in query_verbs:
            if verb in evidence_lower:
                return 1.0  # Explicit match

        # Check for related/implied relationships
        for verb in relationship_verbs:
            if verb in evidence_lower:
                return 0.5  # Implied match

        return 0.0  # No relationship found

    def _compute_temporal_alignment(self, query: str, evidence_text: str) -> float:
        """
        Enhanced temporal alignment using the TemporalReasoningEngine.

        Supports:
        - Multiple date formats (ISO, natural language, Korean)
        - Temporal relationships (before, after, during)
        - Temporal ordering in reasoning chains
        - Precision-aware scoring

        Weight: 20% of final score.
        """
        try:
            result = compute_enhanced_temporal_alignment(query, evidence_text)
            return result["score"]
        except Exception as e:
            logger.warning("Enhanced temporal alignment failed, using fallback", error=str(e))
            return self._compute_temporal_alignment_fallback(query, evidence_text)

    def _compute_temporal_alignment_fallback(self, query: str, evidence_text: str) -> float:
        """
        Fallback temporal alignment using simple year matching.
        """
        year_pattern = r"\b(1[89]\d{2}|20[0-2]\d)\b"
        query_years = set(re.findall(year_pattern, query))
        evidence_years = set(re.findall(year_pattern, evidence_text))

        if not query_years:
            return 0.5
        if not evidence_years:
            return 0.5
        if query_years & evidence_years:
            return 1.0
        for qy in query_years:
            qy_int = int(qy)
            for ey in evidence_years:
                ey_int = int(ey)
                if abs(qy_int - ey_int) <= 5:
                    return 0.7
        return 0.2

    def _compute_temporal_alignment_detailed(
        self, query: str, evidence_text: str
    ) -> dict[str, Any]:
        """
        Get detailed temporal alignment including relationship analysis.
        """
        return compute_enhanced_temporal_alignment(query, evidence_text)

    def _compute_answer_presence(
        self, evidence_text: str, expected_type: str | None = None
    ) -> tuple[float, str | None]:
        """
        Check if evidence contains a candidate answer of expected type.
        Weight: 10% of final score.
        Returns: (score, extracted_answer)
        """
        entities = self._extract_entities_fast(evidence_text)

        if not entities:
            return 0.0, None

        # Try to find an answer based on expected type
        if expected_type:
            expected_lower = expected_type.lower()

            # Person names (multiple capitalized words)
            if expected_lower in ["person", "people", "who"]:
                person_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
                matches = re.findall(person_pattern, evidence_text)
                if matches:
                    return 1.0, matches[0]

            # Organizations
            if expected_lower in ["organization", "org", "company", "team"]:
                org_indicators = ["Inc", "Corp", "Ltd", "LLC", "FC", "Club", "Team"]
                for entity in entities:
                    if any(ind.lower() in entity for ind in org_indicators):
                        return 1.0, entity
                # Return first capitalized phrase as fallback
                cap_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", evidence_text)
                if cap_matches:
                    return 0.7, cap_matches[0]

            # Locations
            if expected_lower in ["location", "place", "where", "city", "country"]:
                # Return first capitalized word/phrase
                loc_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", evidence_text)
                if loc_matches:
                    return 0.8, loc_matches[0]

            # Dates
            if expected_lower in ["date", "when", "year", "time"]:
                date_matches = re.findall(r"\b(\d{4}|\w+\s+\d{1,2},?\s+\d{4})\b", evidence_text)
                if date_matches:
                    return 1.0, date_matches[0]

        # Generic: just return if entities exist
        if entities:
            return 0.5, list(entities)[0]

        return 0.0, None

    def _classify_evidence_confidence(
        self, query: str, evidence_text: str, entity_overlap: float, relationship_match: float
    ) -> EvidenceConfidenceType:
        """
        Classify evidence as EXPLICIT, IMPLICIT, or INFERRED.
        """
        # High entity overlap + high relationship match = EXPLICIT
        if entity_overlap >= 0.3 and relationship_match >= 0.8:
            return EvidenceConfidenceType.EXPLICIT

        # Medium scores = IMPLICIT
        if entity_overlap >= 0.1 or relationship_match >= 0.5:
            return EvidenceConfidenceType.IMPLICIT

        # Low scores = INFERRED
        return EvidenceConfidenceType.INFERRED

    def compute_evidence_grounded_score(
        self,
        query: str,
        evidence_text: str,
        expected_answer_type: str | None = None,
        other_evidence_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compute enhanced 5-component evidence-grounded score.

        Components (weights):
        - Entity Overlap: 35%
        - Relationship Match: 25%
        - Temporal Alignment: 20% (Enhanced with temporal reasoning)
        - Answer Presence: 10%
        - Negative Evidence Adjustment: 10% (NEW)

        Enhanced features:
        - Advanced temporal relationship detection (before/after/during)
        - Negative evidence and contradiction detection
        - Polarity-aware scoring

        Returns dict with all scores and classifications.
        """
        # Core component scores
        entity_overlap = self._compute_entity_overlap(query, evidence_text)
        relationship_match = self._compute_relationship_match(query, evidence_text)
        temporal_alignment = self._compute_temporal_alignment(query, evidence_text)
        answer_presence, extracted_answer = self._compute_answer_presence(
            evidence_text, expected_answer_type
        )

        # Enhanced temporal analysis
        temporal_details = self._compute_temporal_alignment_detailed(query, evidence_text)

        # Negative evidence analysis
        polarity_result = analyze_evidence_polarity(evidence_text, query)
        supports_answer = polarity_result.polarity != EvidencePolarity.NEGATIVE

        # Compute negative evidence adjustment
        negative_adjustment = 1.0  # Neutral by default
        if polarity_result.polarity == EvidencePolarity.NEGATIVE:
            negative_adjustment = 0.7  # Penalty for negative evidence
        elif polarity_result.polarity == EvidencePolarity.UNCERTAIN:
            negative_adjustment = 0.9  # Minor penalty for uncertainty

        # Check for contradictions with other evidence
        contradiction_penalty = 0.0
        if other_evidence_texts:
            scorer = get_negative_evidence_scorer()
            neg_result = scorer.compute_negative_adjusted_score(
                base_score=1.0,
                evidence_text=evidence_text,
                query=query,
                other_evidence=other_evidence_texts,
            )
            contradiction_penalty = neg_result.get("contradiction_penalty", 0.0)

        # Compute weighted final score with new components
        base_score = (
            entity_overlap * 0.35
            + relationship_match * 0.25
            + temporal_alignment * 0.20
            + answer_presence * 0.10
            + (negative_adjustment * 0.10)  # Negative evidence component
        )

        # Apply contradiction penalty
        final_score = base_score * (1.0 - min(0.3, contradiction_penalty))

        # Classify confidence type
        confidence_type = self._classify_evidence_confidence(
            query, evidence_text, entity_overlap, relationship_match
        )

        return {
            # Core component scores
            "entity_overlap_score": entity_overlap,
            "relationship_match_score": relationship_match,
            "temporal_alignment_score": temporal_alignment,
            "answer_presence_score": answer_presence,
            "final_relevance_score": min(1.0, max(0.0, final_score)),
            "confidence_type": confidence_type,
            "extracted_answer": extracted_answer,
            "supporting_entities": list(self._extract_entities_fast(evidence_text)),
            # Enhanced temporal details
            "temporal_details": {
                "alignment_type": temporal_details.get("alignment_type", "unknown"),
                "temporal_match": temporal_details.get("temporal_match", False),
                "temporal_consistency": temporal_details.get("temporal_consistency", True),
            },
            # Negative evidence details
            "negative_evidence": {
                "polarity": polarity_result.polarity.value,
                "supports_answer": supports_answer,
                "negation_count": len(polarity_result.negations),
                "negated_claims": polarity_result.negated_claims[:3],
                "negative_adjustment": negative_adjustment,
                "contradiction_penalty": contradiction_penalty,
            },
        }

    async def score_evidence_with_llm(
        self,
        question: str,
        evidence_list: list[Evidence],
        prior_evidence: list[Evidence] | None = None,
    ) -> dict[str, Any]:
        """
        Use LLM to score evidence with the Evidence-Grounded Retrieval prompt.
        Returns scored evidence with gaps identified.
        """
        if not evidence_list:
            return {
                "scored_evidence": [],
                "retrieval_confidence": 0.0,
                "gaps_identified": ["No evidence found"],
                "recommended_next_query": question,
            }

        # Format evidence for prompt
        candidate_str = "\n".join([
            f"{i+1}. [{e.evidence_type.value}] {e.content[:300]}"
            for i, e in enumerate(evidence_list[:15])
        ])

        prior_str = ""
        if prior_evidence:
            prior_str = "\n".join([
                f"- {e.content[:200]}" for e in prior_evidence[:5]
            ])

        try:
            result = await self._evidence_grounded_chain.ainvoke({
                "sub_question": question,
                "candidate_evidence": candidate_str,
                "prior_evidence": prior_str or "None",
            })

            return result

        except Exception as e:
            logger.warning("LLM evidence scoring failed, using fast scoring", error=str(e))
            # Fallback to fast scoring
            return {
                "scored_evidence": [],
                "retrieval_confidence": 0.5,
                "gaps_identified": ["LLM scoring failed, using heuristic scoring"],
                "recommended_next_query": None,
            }

    # =========================================================================
    # Original Methods
    # =========================================================================

    async def determine_strategy(self, state: MACERState) -> dict[str, Any]:
        """
        Determine the best retrieval strategy based on current state.

        Args:
            state: Current MACER state

        Returns:
            Strategy decision with parameters
        """
        question = state.get("current_query", state.get("original_query", ""))
        subgraph = state.get("current_subgraph", SubGraph())
        evidence = state.get("evidence", [])
        iteration = state.get("iteration", 0)

        try:
            result: dict[str, Any] = await self._strategy_chain.ainvoke(
                {
                    "question": question,
                    "subgraph_size": subgraph.node_count(),
                    "evidence_count": len(evidence),
                    "iteration": iteration,
                }
            )
            return result
        except Exception as e:
            logger.warning("Strategy determination failed, using default", error=str(e))
            return {
                "primary_strategy": RetrievalStrategy.HYBRID,
                "search_depth": 2,
                "focus_entities": [],
                "search_terms": [question],
            }

    async def vector_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of similar entities
        """
        if not self._embeddings:
            logger.warning("No embeddings model available, skipping vector search")
            return []

        await self._client.connect()
        results = []

        try:
            # Embed the query
            logger.info("Generating query embedding for vector search", query=query[:50])
            query_embedding = await self._embeddings.aembed_query(query)

            # Search entities with quality threshold
            try:
                entity_results = await self._client.vector_search(
                    embedding=query_embedding,
                    node_label=NodeLabel.ENTITY,
                    top_k=top_k,
                    min_score=0.5,  # Increased threshold for better precision (was 0.3)
                )
                logger.info("Entity vector search results", count=len(entity_results))

                for r in entity_results:
                    results.append(
                        {
                            "id": r.node_id,
                            "label": r.node_label,
                            "score": r.score,
                            "properties": r.properties,
                            "source": "vector_entity",
                        }
                    )
            except Exception as e:
                logger.warning("Entity vector search failed (index may not exist)", error=str(e))

            # Search chunks for context
            try:
                chunk_results = await self._client.vector_search(
                    embedding=query_embedding,
                    node_label=NodeLabel.CHUNK,
                    top_k=top_k,
                    min_score=0.5,  # Increased threshold for better precision (was 0.3)
                )
                logger.info("Chunk vector search results", count=len(chunk_results))

                for r in chunk_results:
                    results.append(
                        {
                            "id": r.node_id,
                            "label": r.node_label,
                            "score": r.score,
                            "properties": r.properties,
                            "source": "vector_chunk",
                        }
                    )
            except Exception as e:
                logger.warning("Chunk vector search failed (index may not exist)", error=str(e))

            return results

        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []

    async def fallback_chunk_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Fallback search using fulltext search on chunks when vector search fails.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of chunk results
        """
        await self._client.connect()
        results = []

        try:
            # Try fulltext search on chunks
            logger.info("Attempting fallback fulltext chunk search", query=query[:50])
            fulltext_results = await self._client.fulltext_search(
                query_text=query,
                node_label=NodeLabel.CHUNK,
                top_k=top_k,
                min_score=0.1,
            )

            for r in fulltext_results:
                text = r.text or ""
                if text:  # Only add if text is not empty
                    results.append(
                        {
                            "id": r.node_id,
                            "label": r.node_label,
                            "score": r.score,
                            "properties": {"text": text},
                            "source": "fulltext_chunk",
                        }
                    )

            logger.info(
                "Fallback chunk search results",
                count=len(results),
                sample_text_lengths=[len(r["properties"]["text"]) for r in results[:3]] if results else [],
            )

        except Exception as e:
            logger.warning("Fallback chunk search failed", error=str(e))

        # Also try direct text match if fulltext fails
        if not results:
            try:
                direct_query = """
                MATCH (c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower($query)
                RETURN c.id as id, c.text as text, c.source as source
                LIMIT $limit
                """
                direct_results = await self._client.execute_cypher(
                    direct_query, {"query": query, "limit": top_k}
                )

                for r in direct_results:
                    text = r.get("text") or ""
                    if text:  # Only add if text is not empty
                        results.append(
                            {
                                "id": r.get("id") or f"chunk_{len(results)}",
                                "label": "Chunk",
                                "score": 0.5,
                                "properties": {"text": text, "source": r.get("source") or ""},
                                "source": "direct_chunk_match",
                            }
                        )

                logger.info(
                    "Direct chunk match results",
                    count=len(results),
                    sample_text_lengths=[len(r["properties"]["text"]) for r in results[:3]] if results else [],
                )

            except Exception as e:
                logger.warning("Direct chunk match failed", error=str(e))

        return results

    async def graph_traversal(
        self,
        subgraph: SubGraph,
        depth: int = 2,
        focus_node_ids: list[str] | None = None,
    ) -> tuple[list[SubGraphNode], list[SubGraphEdge]]:
        """
        Perform multi-hop graph traversal to expand subgraph.

        Args:
            subgraph: Current subgraph
            depth: Traversal depth
            focus_node_ids: Specific nodes to expand from

        Returns:
            Tuple of (new_nodes, new_edges)
        """
        await self._client.connect()

        # Determine starting nodes
        if focus_node_ids:
            start_nodes = focus_node_ids
        else:
            # Start from highest relevance nodes
            sorted_nodes = sorted(
                subgraph.nodes,
                key=lambda n: n.relevance_score,
                reverse=True,
            )
            start_nodes = [n.id for n in sorted_nodes[:5]]

        new_nodes: dict[str, SubGraphNode] = {}
        new_edges: list[SubGraphEdge] = []
        existing_node_ids = {n.id for n in subgraph.nodes}

        for node_id in start_nodes:
            try:
                result = await self._client.get_neighbors(
                    node_id=node_id,
                    max_hops=depth,
                    limit=20,
                )

                for neighbor in result.get("neighbors", []):
                    neighbor_id = neighbor.get("neighbor_id")
                    distance = neighbor.get("distance", 1)

                    # Skip existing nodes
                    if neighbor_id in existing_node_ids or neighbor_id in new_nodes:
                        # Still add edges
                        for rel in neighbor.get("relationships", []):
                            edge = SubGraphEdge(
                                source_id=node_id,
                                target_id=neighbor_id,
                                relation_type=rel.get("type", "RELATED_TO"),
                                predicate=rel.get("properties", {}).get("predicate", ""),
                                properties=rel.get("properties", {}),
                            )
                            new_edges.append(edge)
                        continue

                    props = neighbor.get("properties", {})
                    node = SubGraphNode(
                        id=neighbor_id,
                        name=props.get("name", neighbor_id),
                        type=props.get("type", neighbor.get("neighbor_label", "Entity")),
                        properties=props,
                        distance_from_topic=distance,
                        relevance_score=max(0.3, 1.0 - (distance * 0.2)),
                    )
                    new_nodes[neighbor_id] = node

                    # Add edges
                    for rel in neighbor.get("relationships", []):
                        edge = SubGraphEdge(
                            source_id=node_id,
                            target_id=neighbor_id,
                            relation_type=rel.get("type", "RELATED_TO"),
                            predicate=rel.get("properties", {}).get("predicate", ""),
                            properties=rel.get("properties", {}),
                        )
                        new_edges.append(edge)

            except Exception as e:
                logger.debug("Graph traversal failed for node", node_id=node_id, error=str(e))

        logger.info(
            "Graph traversal completed",
            new_nodes=len(new_nodes),
            new_edges=len(new_edges),
        )

        return list(new_nodes.values()), new_edges

    async def community_search(
        self,
        query: str,
        subgraph: SubGraph,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search community summaries for high-level context.

        Args:
            query: Search query
            subgraph: Current subgraph
            top_k: Number of communities

        Returns:
            List of relevant community summaries
        """
        await self._client.connect()

        try:
            # Full-text search on community summaries
            results = await self._client.fulltext_search(
                query_text=query,
                node_label=NodeLabel.COMMUNITY,
                top_k=top_k,
                min_score=0.5,  # Increased threshold for better precision (was 0.3)
            )

            communities = []
            for r in results:
                communities.append(
                    {
                        "id": r.node_id,
                        "summary": r.text,
                        "score": r.score,
                        "source": "community_search",
                    }
                )

            # Also get communities connected to current subgraph entities
            entity_ids = [n.id for n in subgraph.nodes if n.type != "Community"][:10]
            if entity_ids:
                query = """
                MATCH (e:Entity)-[:BELONGS_TO]->(c:Community)
                WHERE e.id IN $entity_ids
                RETURN DISTINCT c.id as id, c.summary as summary, c.level as level
                LIMIT $top_k
                """
                connected = await self._client.execute_cypher(
                    query, {"entity_ids": entity_ids, "top_k": top_k}
                )

                for comm in connected:
                    if not any(c["id"] == comm["id"] for c in communities):
                        communities.append(
                            {
                                "id": comm["id"],
                                "summary": comm.get("summary", ""),
                                "level": comm.get("level", 0),
                                "score": 0.7,
                                "source": "connected_community",
                            }
                        )

            return communities

        except Exception as e:
            logger.error("Community search failed", error=str(e))
            return []

    async def text2cypher_search(
        self,
        query: str,
        max_results: int = 20,
    ) -> tuple[list[dict[str, Any]], CypherGenerationResult | None]:
        """
        Generate and execute a Cypher query from natural language.

        Uses the Text2Cypher generator to translate the question into
        a Cypher query and execute it against the knowledge graph.

        Args:
            query: Natural language question
            max_results: Maximum number of results to return

        Returns:
            Tuple of (results, generation_result)
        """
        if not self._text2cypher:
            logger.debug("Text2Cypher not enabled, skipping")
            return [], None

        await self._client.connect()

        try:
            # Generate Cypher with entity resolution for better accuracy
            gen_result = await self._text2cypher.generate_with_entity_resolution(
                question=query,
                validate=True,
            )

            logger.info(
                "Text2Cypher generated",
                cypher=gen_result.cypher[:100] if gen_result.cypher else "None",
                confidence=gen_result.confidence,
                is_valid=gen_result.is_valid,
                healing_attempts=gen_result.healing_attempts,
            )

            if not gen_result.is_valid:
                logger.warning("Generated Cypher is invalid, skipping execution")
                return [], gen_result

            # Execute the generated query
            try:
                # Add LIMIT if not present
                cypher = gen_result.cypher.strip()
                if "LIMIT" not in cypher.upper():
                    cypher = f"{cypher} LIMIT {max_results}"

                results = await self._client.execute_cypher(
                    cypher,
                    gen_result.parameters or None,
                )

                logger.info("Text2Cypher query executed", results_count=len(results))

                # Format results for evidence extraction
                formatted_results = []
                for r in results[:max_results]:
                    # Convert Neo4j result to dict format compatible with evidence extraction
                    formatted = {
                        "id": r.get("id", r.get("n.id", str(hash(str(r)))[:12])),
                        "label": r.get("label", r.get("type", "Entity")),
                        "score": gen_result.confidence,
                        "properties": {k: v for k, v in r.items() if v is not None},
                        "source": "text2cypher",
                        "cypher_query": gen_result.cypher[:200],
                    }
                    formatted_results.append(formatted)

                return formatted_results, gen_result

            except Exception as e:
                logger.warning("Text2Cypher execution failed", error=str(e))
                return [], gen_result

        except Exception as e:
            logger.error("Text2Cypher generation failed", error=str(e))
            return [], None

    def extract_path_evidence(
        self,
        subgraph: SubGraph,
        topic_entity_ids: list[str],
        reasoning_chain_id: str | None = None,
    ) -> list[Evidence]:
        """
        Extract path-based evidence from subgraph for multi-hop reasoning.

        Finds paths connecting topic entities and creates evidence that
        captures the entire reasoning chain.

        Args:
            subgraph: Current reasoning subgraph
            topic_entity_ids: IDs of topic entities to find paths between
            reasoning_chain_id: Optional ID to group related evidence

        Returns:
            List of PATH type Evidence objects
        """
        if len(topic_entity_ids) < 2 or subgraph.node_count() < 2:
            return []

        path_evidence_list = []
        chain_id = reasoning_chain_id or f"chain_{uuid.uuid4().hex[:8]}"

        # Find paths between all pairs of topic entities
        processed_pairs = set()
        for i, start_id in enumerate(topic_entity_ids):
            for end_id in topic_entity_ids[i + 1:]:
                pair_key = tuple(sorted([start_id, end_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # Find all paths between the two entities
                paths = subgraph.find_paths(start_id, end_id, max_length=5)

                for path_idx, path in enumerate(paths[:3]):  # Limit to top 3 paths
                    if len(path) < 2:
                        continue

                    # Get the path with predicates
                    path_with_predicates = subgraph.get_path_with_predicates(path)

                    # Build human-readable path content
                    path_parts = []
                    path_nodes = []
                    path_predicates = []

                    for source_name, predicate, target_name in path_with_predicates:
                        if not path_parts:
                            path_parts.append(source_name)
                        path_parts.append(f"→[{predicate}]→")
                        path_parts.append(target_name)
                        path_predicates.append(predicate)

                    for node_id in path:
                        node = subgraph.get_node(node_id)
                        path_nodes.append(node.name if node else node_id)

                    content = " ".join(path_parts)
                    path_length = len(path) - 1  # Number of hops

                    # Calculate relevance based on path length and node relevance
                    avg_relevance = 0.0
                    for node_id in path:
                        node = subgraph.get_node(node_id)
                        if node:
                            avg_relevance += node.relevance_score
                    avg_relevance /= len(path) if path else 1

                    # Shorter paths are generally more relevant
                    length_factor = 1.0 - (path_length - 1) * 0.1
                    relevance = avg_relevance * max(0.5, length_factor)

                    evidence = Evidence(
                        id=self._generate_evidence_id(f"path_{start_id}_{end_id}_{path_idx}_{content}"),
                        content=f"[Reasoning Path] {content}",
                        evidence_type=EvidenceType.PATH,
                        source_nodes=path,
                        source_edges=[f"{path[j]}->{path[j+1]}" for j in range(len(path) - 1)],
                        relevance_score=min(1.0, relevance),
                        supports_answer=True,
                        hop_index=0,
                        path_length=path_length,
                        reasoning_chain_id=chain_id,
                        path_nodes=path_nodes,
                        path_predicates=path_predicates,
                    )
                    path_evidence_list.append(evidence)

        # Also extract evidence from bridge entities
        bridge_entities = subgraph.get_bridge_entities()
        for bridge in bridge_entities:
            # Find what the bridge connects
            connects = bridge.properties.get("connects", [])
            content = f"[Bridge Entity] {bridge.name} connects multiple reasoning paths"

            if connects:
                connected_names = []
                for conn_id in connects:
                    node = subgraph.get_node(conn_id)
                    if node:
                        connected_names.append(node.name)
                if connected_names:
                    content = f"[Bridge Entity] {bridge.name} connects: {', '.join(connected_names)}"

            evidence = Evidence(
                id=self._generate_evidence_id(f"bridge_{bridge.id}"),
                content=content,
                evidence_type=EvidenceType.PATH,
                source_nodes=[bridge.id],
                relevance_score=0.85,
                supports_answer=True,
                hop_index=1,
                path_length=2,
                reasoning_chain_id=chain_id,
                path_nodes=[bridge.name],
                path_predicates=[],
            )
            path_evidence_list.append(evidence)

        logger.info(
            "Path evidence extracted",
            path_count=len(path_evidence_list),
            topic_pairs=len(processed_pairs),
            bridge_entities=len(bridge_entities),
        )

        return path_evidence_list

    def extract_evidence(
        self,
        subgraph: SubGraph,
        search_results: list[dict[str, Any]],
        community_results: list[dict[str, Any]],
        text2cypher_results: list[dict[str, Any]] | None = None,
        topic_entity_ids: list[str] | None = None,
        question_type: str = QuestionType.FACTOID.value,
    ) -> list[Evidence]:
        """
        Extract evidence from retrieved data.

        Enhanced with multi-hop support:
        - Tracks hop_index based on node distance from topic
        - Extracts path-based evidence for multi-hop questions
        - Groups related evidence into reasoning chains

        Args:
            subgraph: Current subgraph
            search_results: Vector search results
            community_results: Community search results
            text2cypher_results: Results from Text2Cypher queries
            topic_entity_ids: IDs of topic entities (for path finding)
            question_type: Type of question for adaptive extraction

        Returns:
            List of Evidence objects
        """
        evidence_list = []
        text2cypher_results = text2cypher_results or []
        topic_entity_ids = topic_entity_ids or []

        logger.info(
            "Starting evidence extraction",
            subgraph_nodes=subgraph.node_count(),
            subgraph_edges=subgraph.edge_count(),
            search_results=len(search_results),
            community_results=len(community_results),
            text2cypher_results=len(text2cypher_results),
            question_type=question_type,
        )

        # Check if this is a multi-hop question
        is_multihop = question_type in [
            QuestionType.MULTIHOP.value,
            QuestionType.BRIDGE.value,
            QuestionType.COMPARISON.value,
        ]

        # Evidence from subgraph relationships
        edge_evidence_count = 0
        for edge in subgraph.edges:
            try:
                source_node = subgraph.get_node(edge.source_id)
                target_node = subgraph.get_node(edge.target_id)

                if source_node and target_node:
                    content = (
                        f"{source_node.name} {edge.predicate or edge.relation_type} {target_node.name}"
                    )

                    # Calculate hop index based on distance from topic entities
                    min_distance = min(
                        source_node.distance_from_topic,
                        target_node.distance_from_topic,
                    )

                    evidence = Evidence(
                        id=self._generate_evidence_id(content),
                        content=content,
                        evidence_type=EvidenceType.DIRECT,
                        source_nodes=[edge.source_id, edge.target_id],
                        source_edges=[f"{edge.source_id}->{edge.target_id}"],
                        relevance_score=min(source_node.relevance_score, target_node.relevance_score),
                        supports_answer=True,
                        hop_index=min_distance,
                        path_length=1,
                    )
                    evidence_list.append(evidence)
                    edge_evidence_count += 1
            except Exception as e:
                logger.warning("Failed to create edge evidence", edge=str(edge), error=str(e))

        # Evidence from entity properties
        node_evidence_count = 0
        for node in subgraph.nodes:
            try:
                if node.properties.get("description"):
                    content = f"{node.name}: {node.properties['description']}"
                    evidence = Evidence(
                        id=self._generate_evidence_id(content),
                        content=content,
                        evidence_type=EvidenceType.DIRECT,
                        source_nodes=[node.id],
                        relevance_score=node.relevance_score,
                        supports_answer=True,
                        hop_index=node.distance_from_topic,
                        path_length=0,
                    )
                    evidence_list.append(evidence)
                    node_evidence_count += 1
            except Exception as e:
                logger.warning("Failed to create node evidence", node=node.name, error=str(e))

        # Evidence from vector search (chunks)
        chunk_evidence_count = 0
        logger.info(
            "Processing search results for chunk evidence",
            search_results_count=len(search_results),
            source_types=[r.get("source", "unknown") for r in search_results[:5]],
        )
        for result in search_results:
            try:
                source_type = result.get("source", "")
                if source_type in ["vector_chunk", "fulltext_chunk", "direct_chunk_match"]:
                    props = result.get("properties", {})
                    # Use `or ""` to handle both missing key AND None value
                    text = props.get("text") or ""
                    if text:
                        text = str(text)[:500]  # Ensure string and truncate
                        # Clamp relevance_score to [0.0, 1.0] - fulltext scores can exceed 1.0
                        raw_score = float(result.get("score") or 0.5)
                        clamped_score = max(0.0, min(1.0, raw_score))
                        evidence = Evidence(
                            id=self._generate_evidence_id(text),
                            content=text,
                            evidence_type=EvidenceType.CONTEXTUAL,
                            source_nodes=[result.get("id") or "chunk"],
                            relevance_score=clamped_score,
                            supports_answer=True,
                            hop_index=0,  # Chunk evidence is considered direct
                            path_length=0,
                        )
                        evidence_list.append(evidence)
                        chunk_evidence_count += 1
                    else:
                        logger.debug("Skipping chunk with empty text", source=source_type, result_id=result.get("id"))
                elif source_type in ["vector_entity"]:
                    # Also extract evidence from vector entity search results
                    props = result.get("properties", {})
                    name = props.get("name") or ""
                    description = props.get("description") or ""
                    if name or description:
                        content = f"{name}: {description}" if description else name
                        # Clamp relevance_score to [0.0, 1.0]
                        raw_score = float(result.get("score") or 0.5)
                        clamped_score = max(0.0, min(1.0, raw_score))
                        evidence = Evidence(
                            id=self._generate_evidence_id(content),
                            content=content[:500],
                            evidence_type=EvidenceType.DIRECT,
                            source_nodes=[result.get("id") or "entity"],
                            relevance_score=clamped_score,
                            supports_answer=True,
                            hop_index=0,
                            path_length=0,
                        )
                        evidence_list.append(evidence)
                        chunk_evidence_count += 1
            except Exception as e:
                logger.warning("Failed to create chunk evidence", source=result.get("source"), error=str(e))

        logger.debug(
            "Evidence extraction progress",
            edge_evidence=edge_evidence_count,
            node_evidence=node_evidence_count,
            chunk_evidence=chunk_evidence_count,
        )

        # Evidence from community summaries
        community_evidence_count = 0
        for comm in community_results:
            try:
                summary = comm.get("summary") or ""
                if summary:
                    # Clamp relevance_score to [0.0, 1.0]
                    raw_score = float(comm.get("score") or 0.5)
                    clamped_score = max(0.0, min(1.0, raw_score))
                    evidence = Evidence(
                        id=self._generate_evidence_id(summary),
                        content=f"[Community Context] {summary[:500]}",
                        evidence_type=EvidenceType.COMMUNITY,
                        source_nodes=[comm.get("id") or "community"],
                        relevance_score=clamped_score,
                        supports_answer=True,
                        hop_index=1,  # Community summaries provide higher-level context
                        path_length=0,
                    )
                    evidence_list.append(evidence)
                    community_evidence_count += 1
            except Exception as e:
                logger.warning("Failed to create community evidence", comm_id=comm.get("id"), error=str(e))

        # Evidence from Text2Cypher query results
        cypher_evidence_count = 0
        for result in text2cypher_results:
            try:
                props = result.get("properties", {})
                # Build content from all available properties
                content_parts = []
                for key, value in props.items():
                    if value and key not in ("id", "embedding", "chunk_embedding", "entity_embedding"):
                        if isinstance(value, str) and len(value) > 200:
                            value = value[:200] + "..."
                        content_parts.append(f"{key}: {value}")

                if content_parts:
                    content = f"[Cypher Query Result] {'; '.join(content_parts)}"
                    # Clamp relevance_score to [0.0, 1.0]
                    raw_score = float(result.get("score") or 0.7)
                    clamped_score = max(0.0, min(1.0, raw_score))
                    evidence = Evidence(
                        id=self._generate_evidence_id(content),
                        content=content[:500],
                        evidence_type=EvidenceType.DIRECT,  # Direct from query
                        source_nodes=[result.get("id") or "cypher_result"],
                        relevance_score=clamped_score,
                        supports_answer=True,
                        hop_index=0,  # Cypher results are direct
                        path_length=0,
                    )
                    evidence_list.append(evidence)
                    cypher_evidence_count += 1
            except Exception as e:
                logger.warning("Failed to create cypher evidence", result_id=result.get("id"), error=str(e))

        logger.info(
            "Evidence extraction completed",
            edge_evidence=edge_evidence_count,
            node_evidence=node_evidence_count,
            chunk_evidence=chunk_evidence_count,
            community_evidence=community_evidence_count,
            cypher_evidence=cypher_evidence_count,
            total_before_dedup=len(evidence_list),
        )

        # Extract path-based evidence for multi-hop questions
        if is_multihop and len(topic_entity_ids) >= 2:
            path_evidence = self.extract_path_evidence(
                subgraph,
                topic_entity_ids,
            )
            evidence_list.extend(path_evidence)
            logger.info(
                "Added path evidence for multi-hop question",
                path_evidence_count=len(path_evidence),
            )

        # Deduplicate by ID
        seen_ids = set()
        unique_evidence = []
        for ev in evidence_list:
            if ev.id not in seen_ids:
                seen_ids.add(ev.id)
                unique_evidence.append(ev)

        return unique_evidence[: self._max_evidence * 3]  # Allow more buffer for multi-hop

    async def rank_evidence(
        self,
        question: str,
        evidence_list: list[Evidence],
        use_llm: bool = False,
        expected_answer_type: str | None = None,
    ) -> tuple[dict[str, float], list[Evidence], list[str]]:
        """
        Rank evidence using 4-component scoring system.

        Uses fast heuristic scoring by default, with optional LLM fallback.
        Components: Entity Overlap (40%), Relationship Match (30%),
                   Temporal Alignment (20%), Answer Presence (10%)

        Args:
            question: Current query
            evidence_list: List of evidence to rank
            use_llm: Whether to use LLM for scoring (slower but more accurate)
            expected_answer_type: Expected type of answer (person/org/location/date)

        Returns:
            Tuple of (rankings dict, updated evidence list, gaps identified)
        """
        if not evidence_list:
            return {}, [], ["No evidence to rank"]

        rankings: dict[str, float] = {}
        updated_evidence: list[Evidence] = []
        gaps_identified: list[str] = []

        # Use fast 4-component scoring
        explicit_count = 0
        implicit_count = 0
        inferred_count = 0

        for ev in evidence_list:
            # Compute 4-component score
            scores = self.compute_evidence_grounded_score(
                query=question,
                evidence_text=ev.content,
                expected_answer_type=expected_answer_type,
            )

            # Update evidence with new scores
            updated_ev = ev.model_copy(update={
                "relevance_score": scores["final_relevance_score"],
                "entity_overlap_score": scores["entity_overlap_score"],
                "relationship_match_score": scores["relationship_match_score"],
                "temporal_alignment_score": scores["temporal_alignment_score"],
                "answer_presence_score": scores["answer_presence_score"],
                "confidence_type": scores["confidence_type"],
                "extracted_answer": scores["extracted_answer"],
                "supporting_entities": scores["supporting_entities"],
            })

            updated_evidence.append(updated_ev)
            rankings[ev.id] = scores["final_relevance_score"]

            # Count confidence types
            if scores["confidence_type"] == EvidenceConfidenceType.EXPLICIT:
                explicit_count += 1
            elif scores["confidence_type"] == EvidenceConfidenceType.IMPLICIT:
                implicit_count += 1
            else:
                inferred_count += 1

        # Identify gaps based on evidence quality
        if explicit_count == 0:
            gaps_identified.append("No explicit evidence found - answers require inference")
        if all(ev.entity_overlap_score < 0.2 for ev in updated_evidence):
            gaps_identified.append("Low entity overlap - may need different search terms")
        if all(ev.relationship_match_score < 0.5 for ev in updated_evidence):
            gaps_identified.append("Relationship not clearly stated in evidence")
        if all(ev.temporal_alignment_score < 0.5 for ev in updated_evidence):
            gaps_identified.append("Temporal context unclear or mismatched")

        # Optionally use LLM for more accurate scoring
        if use_llm and len(evidence_list) <= 15:
            try:
                llm_result = await self.score_evidence_with_llm(question, evidence_list)
                if llm_result.get("gaps_identified"):
                    gaps_identified.extend(llm_result["gaps_identified"])

                # Merge LLM scores with fast scores (average them)
                for scored in llm_result.get("scored_evidence", []):
                    ev_id = scored.get("evidence_id")
                    llm_score = scored.get("final_relevance_score", 0.5)
                    if ev_id in rankings:
                        # Average of fast and LLM scores
                        rankings[ev_id] = (rankings[ev_id] + llm_score) / 2

            except Exception as e:
                logger.debug("LLM scoring skipped", error=str(e))

        logger.info(
            "Evidence ranked with 4-component scoring",
            total=len(evidence_list),
            explicit=explicit_count,
            implicit=implicit_count,
            inferred=inferred_count,
            gaps=len(gaps_identified),
        )

        return rankings, updated_evidence, gaps_identified

    async def retrieve(self, state: MACERState) -> dict[str, Any]:
        """
        Main retrieval method for MACER pipeline.

        Enhanced with multi-hop support:
        - Uses sub-questions for targeted retrieval
        - Extracts path-based evidence
        - Tracks reasoning chains

        Args:
            state: Current MACER state

        Returns:
            State updates with new evidence, expanded subgraph, and reasoning chains
        """
        question = state.get("current_query", state.get("original_query", ""))
        subgraph = state.get("current_subgraph", SubGraph())
        existing_evidence = state.get("evidence", [])
        iteration = state.get("iteration", 0)
        metadata = state.get("metadata", {})
        question_type = state.get("question_type", QuestionType.FACTOID.value)
        sub_questions = state.get("sub_questions", [])
        topic_entities = state.get("topic_entities", [])
        existing_chains = state.get("reasoning_chains", [])

        # Get topic entity IDs
        topic_entity_ids = [te.id for te in topic_entities if hasattr(te, "id")]

        logger.info(
            "Retriever agent starting",
            question=question[:100],
            iteration=iteration,
            question_type=question_type,
            subgraph_nodes=subgraph.node_count(),
            existing_evidence=len(existing_evidence),
            sub_questions=len(sub_questions),
        )

        # Early exit for NO_DATA scenario
        if metadata.get("no_data", False):
            logger.warning("Retriever: NO_DATA scenario - skipping retrieval")
            return {
                "current_subgraph": subgraph,
                "evidence": existing_evidence,
                "evidence_rankings": {},
                "reasoning_chains": existing_chains,
            }

        # Check if we should terminate early
        if state.get("should_terminate", False):
            logger.info("Retriever: Termination flag set - skipping retrieval")
            return {
                "current_subgraph": subgraph,
                "evidence": existing_evidence,
                "evidence_rankings": {},
                "reasoning_chains": existing_chains,
            }

        # Determine strategy
        strategy = await self.determine_strategy(state)
        primary_strategy = strategy.get("primary_strategy", RetrievalStrategy.HYBRID)
        search_depth = strategy.get("search_depth", 2)

        # For multi-hop questions, increase search depth
        if question_type in [QuestionType.MULTIHOP.value, QuestionType.BRIDGE.value]:
            search_depth = max(search_depth, 3)

        logger.info("Retrieval strategy determined", strategy=primary_strategy, depth=search_depth)

        # Execute retrieval based on strategy
        search_results: list[dict[str, Any]] = []
        community_results: list[dict[str, Any]] = []
        text2cypher_results: list[dict[str, Any]] = []
        new_nodes: list[SubGraphNode] = []
        new_edges: list[SubGraphEdge] = []

        # Search for main question
        if primary_strategy in [RetrievalStrategy.VECTOR_SEARCH, RetrievalStrategy.HYBRID]:
            search_results = await self.vector_search(question, top_k=10)
            logger.info("Vector search completed", results_count=len(search_results))

            # Also search for sub-questions to gather targeted evidence
            for sub_q in sub_questions[:3]:  # Limit to first 3 sub-questions
                sub_results = await self.vector_search(sub_q, top_k=5)
                search_results.extend(sub_results)
                logger.debug("Sub-question search completed", sub_question=sub_q[:50], results=len(sub_results))

        if primary_strategy in [RetrievalStrategy.GRAPH_TRAVERSAL, RetrievalStrategy.HYBRID]:
            focus_entities = strategy.get("focus_entities", [])
            # Convert names to IDs
            focus_ids = []
            for name in focus_entities:
                for node in subgraph.nodes:
                    if name.lower() in node.name.lower():
                        focus_ids.append(node.id)
                        break

            if subgraph.node_count() > 0:
                new_nodes, new_edges = await self.graph_traversal(
                    subgraph,
                    depth=search_depth,
                    focus_node_ids=focus_ids if focus_ids else None,
                )
                logger.info("Graph traversal completed", new_nodes=len(new_nodes), new_edges=len(new_edges))
            else:
                logger.warning("No nodes in subgraph, skipping graph traversal")

        if primary_strategy in [RetrievalStrategy.COMMUNITY_SEARCH, RetrievalStrategy.HYBRID]:
            community_results = await self.community_search(question, subgraph)
            logger.info("Community search completed", results_count=len(community_results))

        # Text2Cypher: Generate and execute natural language query
        if primary_strategy in [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.HYBRID]:
            text2cypher_results, gen_result = await self.text2cypher_search(question, max_results=15)
            if gen_result:
                logger.info(
                    "Text2Cypher completed",
                    results_count=len(text2cypher_results),
                    cypher_valid=gen_result.is_valid,
                    confidence=gen_result.confidence,
                )

            # For multi-hop questions, also try sub-questions
            if question_type in [QuestionType.MULTIHOP.value, QuestionType.BRIDGE.value]:
                for sub_q in sub_questions[:2]:
                    sub_cypher_results, _ = await self.text2cypher_search(sub_q, max_results=10)
                    text2cypher_results.extend(sub_cypher_results)

        # Fallback: If no results found from any method, try direct chunk search
        if not search_results and not new_nodes and not community_results and not text2cypher_results:
            logger.warning("No results from primary methods, attempting fallback chunk search")
            search_results = await self.fallback_chunk_search(question, top_k=15)

        # Update subgraph
        updated_subgraph = SubGraph(
            nodes=subgraph.nodes + new_nodes,
            edges=subgraph.edges + new_edges,
            center_entity_id=subgraph.center_entity_id,
        )

        # Extract new evidence from all sources (with multi-hop support)
        new_evidence = self.extract_evidence(
            updated_subgraph,
            search_results,
            community_results,
            text2cypher_results,
            topic_entity_ids=topic_entity_ids,
            question_type=question_type,
        )

        # Merge with existing evidence (avoid duplicates)
        existing_ids = {ev.id for ev in existing_evidence}
        unique_new_evidence = [ev for ev in new_evidence if ev.id not in existing_ids]

        logger.info(
            "Evidence extraction completed",
            new_evidence_count=len(unique_new_evidence),
            path_evidence_count=sum(1 for e in unique_new_evidence if e.evidence_type == EvidenceType.PATH),
            total_evidence_before_rank=len(existing_evidence) + len(unique_new_evidence),
        )

        # Rank all evidence using 4-component scoring
        all_evidence = existing_evidence + unique_new_evidence
        rankings, updated_evidence, gaps_identified = await self.rank_evidence(
            question=question,
            evidence_list=all_evidence,
            use_llm=False,  # Use fast scoring by default
            expected_answer_type=metadata.get("expected_answer_type"),
        )

        # Sort by relevance and limit
        updated_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        final_evidence = updated_evidence[: self._max_evidence * 3]

        # Count evidence by confidence type
        explicit_evidence = [e for e in final_evidence if e.confidence_type == EvidenceConfidenceType.EXPLICIT]
        implicit_evidence = [e for e in final_evidence if e.confidence_type == EvidenceConfidenceType.IMPLICIT]

        # Extract reasoning chains from path evidence
        reasoning_chains = list(existing_chains)
        for ev in final_evidence:
            if ev.evidence_type == EvidenceType.PATH and ev.reasoning_chain_id:
                # Group evidence by chain ID
                chain_evidence_ids = [e.id for e in final_evidence if e.reasoning_chain_id == ev.reasoning_chain_id]
                if chain_evidence_ids and chain_evidence_ids not in reasoning_chains:
                    reasoning_chains.append(chain_evidence_ids)

        logger.info(
            "Retriever completed",
            iteration=iteration,
            new_nodes=len(new_nodes),
            new_evidence=len(unique_new_evidence),
            total_evidence=len(final_evidence),
            explicit_evidence=len(explicit_evidence),
            implicit_evidence=len(implicit_evidence),
            reasoning_chains=len(reasoning_chains),
            gaps_identified=len(gaps_identified),
            subgraph_total_nodes=updated_subgraph.node_count(),
        )

        # Note: iteration is NOT incremented here - only Reflector increments iteration
        return {
            "current_subgraph": updated_subgraph,
            "evidence": final_evidence,
            "evidence_rankings": rankings,
            "reasoning_chains": reasoning_chains,
            "gaps_identified": gaps_identified,
        }

    def _generate_evidence_id(self, content: str) -> str:
        """Generate deterministic evidence ID."""
        return f"ev_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

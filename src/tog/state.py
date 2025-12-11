"""
MACER State and Models for Think-on-Graph 3.0.

Defines the state schema and data models for the MACER reasoning framework:
- Meta-cognitive
- Adaptive
- Chain-of-thought with
- Evidence-based
- Reasoning
"""

from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class ReasoningAction(str, Enum):
    """Actions available during reasoning."""

    EXPLORE = "explore"  # Expand subgraph
    FOCUS = "focus"  # Narrow down search
    REFINE = "refine"  # Refine query
    BACKTRACK = "backtrack"  # Go back to previous state
    CONCLUDE = "conclude"  # Ready to answer


class EvidenceType(str, Enum):
    """Types of evidence collected."""

    DIRECT = "direct"  # Direct entity/relation match
    INFERRED = "inferred"  # Inferred from reasoning
    CONTEXTUAL = "contextual"  # From surrounding context
    COMMUNITY = "community"  # From community summaries
    PATH = "path"  # Multi-hop reasoning path (chain of evidence)


class EvidenceConfidenceType(str, Enum):
    """Evidence confidence classification for multi-hop QA."""

    EXPLICIT = "explicit"  # Passage directly states the answer
    IMPLICIT = "implicit"  # Passage implies the answer through context
    INFERRED = "inferred"  # Answer requires combining with other evidence


class QuestionType(str, Enum):
    """Types of questions for adaptive processing."""

    # Original types (HotpotQA-style)
    FACTOID = "factoid"  # Single fact lookup (who, what, when, where)
    COMPARISON = "comparison"  # Compare two or more entities
    MULTIHOP = "multihop"  # Requires chaining multiple facts
    AGGREGATION = "aggregation"  # Requires aggregating multiple pieces
    BRIDGE = "bridge"  # Bridge entity questions (who/where/when)
    YESNO = "yesno"  # Yes/No questions

    # Extended types (General Document QA)
    DEFINITION = "definition"  # "What is X?", "Define X" - requires explanation
    PROCEDURE = "procedure"  # "How to X?", "Steps for X" - requires step-by-step
    CAUSE_EFFECT = "cause_effect"  # "Why X?", "What causes X?" - requires causal reasoning
    LIST = "list"  # "List X", "Types of X" - requires enumeration
    NARRATIVE = "narrative"  # Complex explanation requiring synthesis
    OPINION = "opinion"  # Analysis, evaluation, or subjective response


class AnswerClassification(str, Enum):
    """Answer confidence classification for synthesis."""

    CONFIDENT = "confident"  # overall_confidence > 0.8, all steps >= 0.8
    PROBABLE = "probable"  # 0.6 < overall_confidence <= 0.8
    UNCERTAIN = "uncertain"  # 0.4 < overall_confidence <= 0.6
    INSUFFICIENT = "insufficient"  # overall_confidence <= 0.4 or critical gaps


class ChainValidationStatus(str, Enum):
    """Status of reasoning chain validation."""

    VALID = "valid"  # All steps have answers with evidence
    PARTIAL = "partial"  # Some steps missing evidence but answer derivable
    INVALID = "invalid"  # Critical steps missing or circular dependencies


class GroundingSource(str, Enum):
    """Source of answer grounding priority."""

    GRAPH_ONLY = "graph_only"  # Answer entirely from graph evidence
    GRAPH_PRIMARY = "graph_primary"  # Primarily graph with minor inferences
    LLM_SUPPLEMENTED = "llm_supplemented"  # Graph supplemented with LLM knowledge
    LLM_ONLY = "llm_only"  # No graph evidence, LLM knowledge only


class HallucinationRisk(str, Enum):
    """Risk level for hallucination in generated answer."""

    LOW = "low"  # Strong graph evidence, high confidence
    MEDIUM = "medium"  # Partial graph evidence, some inference
    HIGH = "high"  # Mostly LLM knowledge, weak evidence


class TopicEntity(BaseModel):
    """Entity identified as topic from the query."""

    id: str = Field(..., description="Entity ID in graph")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="extraction", description="How entity was identified")


class SubGraphNode(BaseModel):
    """Node in the reasoning subgraph."""

    id: str = Field(..., description="Node ID")
    name: str = Field(..., description="Node name")
    type: str = Field(..., description="Node type")
    properties: dict[str, Any] = Field(default_factory=dict)
    distance_from_topic: int = Field(default=0, description="Hops from topic entity")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)


class SubGraphEdge(BaseModel):
    """Edge in the reasoning subgraph."""

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relation_type: str = Field(..., description="Relationship type")
    predicate: str = Field(default="", description="Specific predicate")
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SubGraph(BaseModel):
    """Subgraph for reasoning context with multi-hop analysis support."""

    nodes: list[SubGraphNode] = Field(default_factory=list)
    edges: list[SubGraphEdge] = Field(default_factory=list)
    center_entity_id: str | None = Field(default=None, description="Main topic entity")

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def get_node(self, node_id: str) -> SubGraphNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_neighbors(self, node_id: str) -> list[str]:
        neighbors = []
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbors.append(edge.target_id)
            elif edge.target_id == node_id:
                neighbors.append(edge.source_id)
        return list(set(neighbors))

    def get_edge(self, source_id: str, target_id: str) -> SubGraphEdge | None:
        """Get edge between two nodes."""
        for edge in self.edges:
            if (edge.source_id == source_id and edge.target_id == target_id) or \
               (edge.source_id == target_id and edge.target_id == source_id):
                return edge
        return None

    def add_node(self, node: SubGraphNode) -> "SubGraph":
        """Add a node immutably, returning a new SubGraph."""
        if any(n.id == node.id for n in self.nodes):
            return self  # Node already exists
        return self.model_copy(update={"nodes": self.nodes + [node]})

    def add_edge(self, edge: SubGraphEdge) -> "SubGraph":
        """Add an edge immutably, returning a new SubGraph."""
        return self.model_copy(update={"edges": self.edges + [edge]})

    def merge(self, other: "SubGraph") -> "SubGraph":
        """Merge with another subgraph immutably."""
        existing_node_ids = {n.id for n in self.nodes}
        new_nodes = [n for n in other.nodes if n.id not in existing_node_ids]

        existing_edges = {(e.source_id, e.target_id, e.relation_type) for e in self.edges}
        new_edges = [
            e for e in other.edges
            if (e.source_id, e.target_id, e.relation_type) not in existing_edges
        ]

        return self.model_copy(update={
            "nodes": self.nodes + new_nodes,
            "edges": self.edges + new_edges,
        })

    def prune_low_relevance(self, threshold: float = 0.3) -> "SubGraph":
        """Remove low-relevance nodes immutably."""
        kept_nodes = [n for n in self.nodes if n.relevance_score >= threshold]
        kept_node_ids = {n.id for n in kept_nodes}
        kept_edges = [
            e for e in self.edges
            if e.source_id in kept_node_ids and e.target_id in kept_node_ids
        ]
        return self.model_copy(update={"nodes": kept_nodes, "edges": kept_edges})

    # =========================================================================
    # Multi-hop Analysis Methods
    # =========================================================================

    def find_paths(
        self,
        start_node_id: str,
        end_node_id: str,
        max_length: int = 5,
    ) -> list[list[str]]:
        """
        Find all paths between two nodes using BFS.

        Args:
            start_node_id: Starting node ID
            end_node_id: Target node ID
            max_length: Maximum path length (hops)

        Returns:
            List of paths, each path is a list of node IDs
        """
        if start_node_id == end_node_id:
            return [[start_node_id]]

        # Build adjacency list
        adjacency: dict[str, list[str]] = {}
        for node in self.nodes:
            adjacency[node.id] = []
        for edge in self.edges:
            if edge.source_id in adjacency:
                adjacency[edge.source_id].append(edge.target_id)
            if edge.target_id in adjacency:
                adjacency[edge.target_id].append(edge.source_id)

        # BFS for all paths
        paths: list[list[str]] = []
        queue: list[list[str]] = [[start_node_id]]

        while queue:
            path = queue.pop(0)
            if len(path) > max_length + 1:
                continue

            current = path[-1]
            if current == end_node_id:
                paths.append(path)
                continue

            for neighbor in adjacency.get(current, []):
                if neighbor not in path:  # Avoid cycles
                    queue.append(path + [neighbor])

        return paths

    def find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
    ) -> list[str] | None:
        """Find shortest path between two nodes."""
        paths = self.find_paths(start_node_id, end_node_id, max_length=10)
        if not paths:
            return None
        return min(paths, key=len)

    def get_path_with_predicates(
        self,
        path: list[str],
    ) -> list[tuple[str, str, str]]:
        """
        Get path with edge predicates.

        Returns list of (source_name, predicate, target_name) tuples.
        """
        result = []
        for i in range(len(path) - 1):
            source_id, target_id = path[i], path[i + 1]
            source_node = self.get_node(source_id)
            target_node = self.get_node(target_id)
            edge = self.get_edge(source_id, target_id)

            source_name = source_node.name if source_node else source_id
            target_name = target_node.name if target_node else target_id
            predicate = edge.predicate or edge.relation_type if edge else "related_to"

            result.append((source_name, predicate, target_name))
        return result

    def get_bridge_entities(self) -> list[SubGraphNode]:
        """
        Identify bridge entities - nodes with high betweenness.

        Bridge entities connect different parts of the subgraph
        and are critical for multi-hop reasoning.
        """
        if len(self.nodes) < 3:
            return []

        # Calculate simple betweenness: how many shortest paths pass through each node
        betweenness: dict[str, int] = {n.id: 0 for n in self.nodes}
        node_ids = [n.id for n in self.nodes]

        for i, start in enumerate(node_ids):
            for end in node_ids[i + 1:]:
                paths = self.find_paths(start, end, max_length=5)
                for path in paths:
                    for node_id in path[1:-1]:  # Exclude start and end
                        betweenness[node_id] += 1

        # Return nodes with above-average betweenness
        if not betweenness:
            return []
        avg_betweenness = sum(betweenness.values()) / len(betweenness)
        bridge_ids = {nid for nid, b in betweenness.items() if b > avg_betweenness}

        return [n for n in self.nodes if n.id in bridge_ids]

    def get_longest_path_length(self) -> int:
        """Get length of longest path in subgraph (diameter approximation)."""
        if len(self.nodes) < 2:
            return 0

        max_length = 0
        node_ids = [n.id for n in self.nodes]

        # Sample a subset for large graphs
        sample_size = min(10, len(node_ids))
        import random
        sampled = random.sample(node_ids, sample_size) if len(node_ids) > sample_size else node_ids

        for start in sampled:
            for end in sampled:
                if start != end:
                    paths = self.find_paths(start, end, max_length=10)
                    if paths:
                        longest = max(len(p) - 1 for p in paths)
                        max_length = max(max_length, longest)

        return max_length

    def is_connected(self, node_id_1: str, node_id_2: str) -> bool:
        """Check if two nodes are connected by any path."""
        return len(self.find_paths(node_id_1, node_id_2, max_length=10)) > 0

    def count_components(self) -> int:
        """Count number of connected components in the subgraph."""
        if not self.nodes:
            return 0

        visited: set[str] = set()
        components = 0

        def dfs(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for neighbor in self.get_neighbors(node_id):
                dfs(neighbor)

        for node in self.nodes:
            if node.id not in visited:
                dfs(node.id)
                components += 1

        return components

    def get_subgraph_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics about the subgraph for reasoning."""
        bridge_entities = self.get_bridge_entities()
        return {
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "component_count": self.count_components(),
            "is_connected": self.count_components() <= 1,
            "max_path_length": self.get_longest_path_length(),
            "bridge_entity_count": len(bridge_entities),
            "bridge_entities": [n.name for n in bridge_entities],
            "avg_node_relevance": (
                sum(n.relevance_score for n in self.nodes) / len(self.nodes)
                if self.nodes else 0.0
            ),
        }


class EvidencePolarity(str, Enum):
    """Polarity classification for evidence."""

    POSITIVE = "positive"       # Supports the claim
    NEGATIVE = "negative"       # Contradicts the claim
    NEUTRAL = "neutral"         # Neither supports nor contradicts
    UNCERTAIN = "uncertain"     # Cannot determine polarity


class Evidence(BaseModel):
    """Evidence piece supporting an answer."""

    id: str = Field(..., description="Evidence identifier")
    content: str = Field(..., description="Evidence content/statement")
    evidence_type: EvidenceType = Field(default=EvidenceType.DIRECT)
    source_nodes: list[str] = Field(default_factory=list, description="Source node IDs")
    source_edges: list[str] = Field(default_factory=list, description="Source edge descriptions")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    supports_answer: bool = Field(default=True, description="Whether it supports or contradicts")

    # Enhanced negative evidence tracking
    polarity: EvidencePolarity = Field(default=EvidencePolarity.NEUTRAL, description="Evidence polarity")
    negation_count: int = Field(default=0, description="Number of negations detected")
    negated_claims: list[str] = Field(default_factory=list, description="Claims being negated")
    contradiction_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Contradiction level with other evidence")

    # Multi-hop reasoning extensions
    hop_index: int = Field(default=0, description="Position in reasoning chain (0=direct)")
    path_length: int = Field(default=1, description="Number of hops from query entity")
    parent_evidence_ids: list[str] = Field(default_factory=list, description="Evidence this builds upon")
    reasoning_chain_id: str | None = Field(default=None, description="ID of chain this belongs to")
    path_nodes: list[str] = Field(default_factory=list, description="Ordered nodes in path (for PATH type)")
    path_predicates: list[str] = Field(default_factory=list, description="Predicates along path")

    # Evidence-Grounded scoring components (5-component scoring - enhanced)
    entity_overlap_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Jaccard similarity of entities (35% weight)")
    relationship_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relationship match score (25% weight)")
    temporal_alignment_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Temporal context alignment (20% weight)")
    answer_presence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Answer presence score (10% weight)")
    negative_evidence_adjustment: float = Field(default=1.0, ge=0.0, le=1.0, description="Negative evidence adjustment (10% weight)")
    confidence_type: EvidenceConfidenceType = Field(default=EvidenceConfidenceType.IMPLICIT, description="EXPLICIT/IMPLICIT/INFERRED")
    extracted_answer: str | None = Field(default=None, description="Candidate answer extracted from evidence")
    supporting_entities: list[str] = Field(default_factory=list, description="Named entities supporting this evidence")

    # Enhanced temporal context tracking
    temporal_alignment_type: str = Field(default="unknown", description="Type of temporal alignment (exact_match, overlap, close_proximity, distant)")
    temporal_match: bool = Field(default=False, description="Whether temporal context matches query")
    temporal_consistency: bool = Field(default=True, description="Whether evidence is temporally consistent")

    def with_score(self, new_score: float) -> "Evidence":
        """Return a new Evidence with updated relevance score."""
        return self.model_copy(update={"relevance_score": max(0.0, min(1.0, new_score))})

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if evidence meets quality threshold."""
        return self.relevance_score >= threshold and self.evidence_type in [EvidenceType.DIRECT, EvidenceType.PATH]

    def is_part_of_chain(self) -> bool:
        """Check if evidence is part of a reasoning chain."""
        return len(self.parent_evidence_ids) > 0 or self.hop_index > 0 or self.evidence_type == EvidenceType.PATH

    def get_path_representation(self) -> str:
        """Get human-readable path representation for PATH evidence."""
        if self.evidence_type != EvidenceType.PATH or not self.path_nodes:
            return self.content

        path_parts = []
        for i, node in enumerate(self.path_nodes):
            path_parts.append(node)
            if i < len(self.path_predicates):
                path_parts.append(f"→[{self.path_predicates[i]}]→")
        return "".join(path_parts)


class ReasoningStep(BaseModel):
    """A single step in the reasoning chain."""

    step_number: int = Field(..., description="Step number in chain")
    action: ReasoningAction = Field(..., description="Action taken")
    thought: str = Field(..., description="Reasoning thought")
    observation: str = Field(default="", description="What was observed")
    query_evolution: str | None = Field(default=None, description="How query was refined")
    subgraph_change: str | None = Field(default=None, description="How subgraph changed")
    new_evidence: list[str] = Field(default_factory=list, description="New evidence IDs")
    sufficiency_delta: float = Field(default=0.0, description="Change in sufficiency")


class QueryEvolution(BaseModel):
    """Record of query evolution during reasoning."""

    original: str = Field(..., description="Original query")
    refined: str = Field(..., description="Refined query")
    reason: str = Field(default="", description="Why query was refined")
    sub_questions: list[str] = Field(default_factory=list, description="Decomposed sub-questions")


class SufficiencyAssessment(BaseModel):
    """Assessment of evidence sufficiency."""

    score: float = Field(..., ge=0.0, le=1.0, description="Sufficiency score")
    has_enough_evidence: bool = Field(default=False)
    missing_aspects: list[str] = Field(default_factory=list, description="What's missing")
    confidence_factors: dict[str, float] = Field(default_factory=dict)
    recommendation: ReasoningAction = Field(default=ReasoningAction.EXPLORE)
    reasoning: str = Field(default="", description="Explanation of assessment")


class AnswerType(str, Enum):
    """Types of generated answers."""

    DIRECT = "direct"        # Answer directly from graph evidence
    INFERRED = "inferred"    # Answer derived through reasoning
    UNCERTAIN = "uncertain"  # Partial answer with caveats
    NO_DATA = "no_data"      # Knowledge graph empty or no matches
    FAILED = "failed"        # Generation error occurred


class Answer(BaseModel):
    """
    Generated answer from the reasoning process.

    Represents the final output of the MACER pipeline including
    the answer text, supporting evidence, and confidence metrics.
    """

    # Core answer content
    id: str = Field(..., description="Answer identifier")
    query: str = Field(..., description="Original query this answers")
    content: str = Field(..., description="The answer text")
    answer_type: AnswerType = Field(default=AnswerType.DIRECT)

    # Confidence and quality
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence")
    sufficiency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Evidence sufficiency")

    # Supporting information
    supporting_evidence: list[str] = Field(default_factory=list, description="Evidence IDs supporting answer")
    contradicting_evidence: list[str] = Field(default_factory=list, description="Evidence IDs contradicting")
    source_entities: list[str] = Field(default_factory=list, description="Entity IDs used")

    # Explanation and reasoning
    explanation: str = Field(default="", description="Explanation of how answer was derived")
    reasoning_steps: int = Field(default=0, description="Number of reasoning steps taken")
    iterations_used: int = Field(default=0, description="Number of retrieval iterations")

    # Caveats and limitations
    caveats: list[str] = Field(default_factory=list, description="Limitations or uncertainties")

    # Metadata
    created_at: str | None = Field(default=None, description="When answer was generated")
    pipeline_id: str | None = Field(default=None, description="Pipeline that generated this")

    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if answer meets confidence threshold."""
        return self.confidence >= threshold and self.answer_type in [AnswerType.DIRECT, AnswerType.INFERRED]

    def has_contradictions(self) -> bool:
        """Check if there are contradicting evidence pieces."""
        return len(self.contradicting_evidence) > 0


# =============================================================================
# Provenance Tracking Models (for Answer Synthesis)
# =============================================================================


class EvidenceChainLink(BaseModel):
    """A single link in the evidence chain for provenance tracking."""

    claim: str = Field(..., description="The claim being supported")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence IDs supporting this claim")
    hop_path: list[str] = Field(default_factory=list, description="Entity path: entity1 → relation → entity2")
    step_confidences: list[float] = Field(default_factory=list, description="Confidence at each step")
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum of step confidences")

    def calculate_final_confidence(self) -> float:
        """Calculate final confidence as minimum of step confidences."""
        if not self.step_confidences:
            return 0.0
        return min(self.step_confidences)


class ChainValidation(BaseModel):
    """Validation result for the reasoning chain."""

    status: ChainValidationStatus = Field(default=ChainValidationStatus.INVALID)
    steps_validated: int = Field(default=0, description="Number of steps successfully validated")
    steps_total: int = Field(default=0, description="Total number of steps in chain")
    issues: list[str] = Field(default_factory=list, description="Validation issues found")
    circular_dependencies: list[str] = Field(default_factory=list, description="Detected circular dependencies")
    missing_evidence_steps: list[int] = Field(default_factory=list, description="Step indices missing evidence")

    def is_valid_for_synthesis(self) -> bool:
        """Check if chain is valid enough for answer synthesis."""
        return self.status in [ChainValidationStatus.VALID, ChainValidationStatus.PARTIAL]

    def get_validation_ratio(self) -> float:
        """Get ratio of validated steps."""
        if self.steps_total == 0:
            return 0.0
        return self.steps_validated / self.steps_total


class AnswerProvenance(BaseModel):
    """Full provenance tracking for a synthesized answer."""

    evidence_chain: list[EvidenceChainLink] = Field(
        default_factory=list, description="Chain of evidence links for each claim"
    )
    primary_sources: list[str] = Field(
        default_factory=list, description="Evidence IDs for main answer claims"
    )
    supporting_sources: list[str] = Field(
        default_factory=list, description="Evidence IDs for contextual support"
    )
    grounding_source: GroundingSource = Field(
        default=GroundingSource.LLM_ONLY, description="Primary source of answer grounding"
    )

    def get_all_evidence_ids(self) -> list[str]:
        """Get all unique evidence IDs used in provenance."""
        all_ids = set(self.primary_sources + self.supporting_sources)
        for link in self.evidence_chain:
            all_ids.update(link.evidence_ids)
        return list(all_ids)

    def get_grounding_description(self) -> str:
        """Get human-readable grounding description."""
        descriptions = {
            GroundingSource.GRAPH_ONLY: "Based on knowledge graph evidence",
            GroundingSource.GRAPH_PRIMARY: "Based primarily on knowledge graph with minor inferences",
            GroundingSource.LLM_SUPPLEMENTED: "Based on knowledge graph supplemented with general knowledge",
            GroundingSource.LLM_ONLY: "Based on general knowledge (no graph evidence found)",
        }
        return descriptions.get(self.grounding_source, "Unknown grounding source")


class AlternativeAnswer(BaseModel):
    """Alternative answer for uncertain cases."""

    answer: str = Field(..., description="Alternative answer text")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in this alternative")
    reason: str = Field(default="", description="Why this is an alternative")
    supporting_evidence_ids: list[str] = Field(default_factory=list, description="Evidence supporting this alternative")


class AnswerComponents(BaseModel):
    """Structured components of the synthesized answer."""

    direct_answer: str = Field(default="", description="Concise direct answer")
    supporting_facts: list[str] = Field(default_factory=list, description="Facts from evidence")
    inferred_facts: list[str] = Field(default_factory=list, description="Facts derived through inference")
    caveats: list[str] = Field(default_factory=list, description="Limitations or uncertainties")


class SynthesizedAnswer(BaseModel):
    """
    Fully synthesized answer with complete provenance tracking.

    This model represents the output of the Answer Synthesis Agent,
    including chain validation, evidence attribution, and confidence classification.
    """

    # Core answer
    final_answer: str = Field(..., description="The synthesized answer text")
    answer_classification: AnswerClassification = Field(
        default=AnswerClassification.INSUFFICIENT,
        description="Confidence classification"
    )
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall answer confidence")

    # Chain validation
    chain_validation: ChainValidation = Field(
        default_factory=ChainValidation,
        description="Validation result for reasoning chain"
    )

    # Provenance
    provenance: AnswerProvenance = Field(
        default_factory=AnswerProvenance,
        description="Full provenance tracking"
    )

    # Structured components
    answer_components: AnswerComponents = Field(
        default_factory=AnswerComponents,
        description="Structured answer components"
    )

    # Alternatives (for uncertain answers)
    alternative_answers: list[AlternativeAnswer] = Field(
        default_factory=list,
        description="Alternative answers if uncertain"
    )

    # Missing information
    missing_information: list[str] = Field(
        default_factory=list,
        description="Information that would improve confidence"
    )

    # Hallucination risk assessment
    hallucination_risk: HallucinationRisk = Field(
        default=HallucinationRisk.HIGH,
        description="Risk level for hallucination"
    )

    def is_confident_answer(self) -> bool:
        """Check if answer is confident enough to present directly."""
        return self.answer_classification in [
            AnswerClassification.CONFIDENT,
            AnswerClassification.PROBABLE
        ]

    def should_show_alternatives(self) -> bool:
        """Check if alternatives should be shown."""
        return (
            self.answer_classification in [
                AnswerClassification.UNCERTAIN,
                AnswerClassification.INSUFFICIENT
            ]
            and len(self.alternative_answers) > 0
        )

    def get_formatted_answer(self) -> str:
        """Get formatted answer with appropriate caveats based on classification."""
        prefix = self.provenance.get_grounding_description()

        if self.answer_classification == AnswerClassification.CONFIDENT:
            return f"{prefix}: {self.final_answer}"
        elif self.answer_classification == AnswerClassification.PROBABLE:
            caveats = ", ".join(self.answer_components.caveats[:2]) if self.answer_components.caveats else ""
            caveat_text = f" (Note: {caveats})" if caveats else ""
            return f"{prefix}: {self.final_answer}{caveat_text}"
        elif self.answer_classification == AnswerClassification.UNCERTAIN:
            return f"{prefix} (uncertain): {self.final_answer}"
        else:
            missing = ", ".join(self.missing_information[:3]) if self.missing_information else "more evidence"
            return f"Unable to provide a confident answer. Missing: {missing}"

    @classmethod
    def from_classification_score(cls, score: float) -> AnswerClassification:
        """Determine classification from confidence score."""
        if score > 0.8:
            return AnswerClassification.CONFIDENT
        elif score > 0.6:
            return AnswerClassification.PROBABLE
        elif score > 0.4:
            return AnswerClassification.UNCERTAIN
        else:
            return AnswerClassification.INSUFFICIENT


class MACERState(TypedDict, total=False):
    """
    State for the MACER reasoning framework.

    MACER = Meta-cognitive Adaptive Chain-of-thought with Evidence-based Reasoning

    Flows through: Constructor → Retriever → Reflector → (loop) → Responser
    """

    # Query State
    original_query: str  # Original user question
    current_query: str  # Current (potentially refined) query
    query_history: list[QueryEvolution]  # History of query refinements
    question_type: str  # Type of question (factoid, multihop, comparison, etc.)
    sub_questions: list[str]  # Decomposed sub-questions for multi-hop

    # Graph State
    topic_entities: list[TopicEntity]  # Identified topic entities
    retrieved_entities: list[dict[str, Any]]  # Raw retrieved entities
    current_subgraph: SubGraph  # Current reasoning subgraph
    subgraph_history: list[SubGraph]  # History of subgraph states

    # Evidence State
    evidence: list[Evidence]  # Collected evidence
    evidence_rankings: dict[str, float]  # Evidence ID -> relevance score
    reasoning_chains: list[list[str]]  # Detected multi-hop chains (lists of evidence IDs)
    gaps_identified: list[str]  # Identified gaps in evidence (for further retrieval)

    # Reasoning State
    reasoning_path: list[ReasoningStep]  # Chain of reasoning steps
    sufficiency_score: float  # Current sufficiency score (0-1)
    sufficiency_assessment: SufficiencyAssessment  # Detailed assessment
    path_completeness: float  # Score for how complete reasoning paths are (0-1)

    # Iteration Control
    iteration: int  # Current iteration number
    max_iterations: int  # Maximum allowed iterations
    should_terminate: bool  # Whether to stop reasoning

    # Output State
    final_answer: str | None  # Generated answer
    confidence: float  # Answer confidence (0-1)
    explanation: str  # Explanation of reasoning
    answer_type: str  # Type of answer (direct, inferred, uncertain)

    # Metadata
    pipeline_id: str
    errors: list[str]
    metadata: dict[str, Any]


# =============================================================================
# Factory Functions
# =============================================================================


def detect_question_type(query: str) -> QuestionType:
    """
    Detect the type of question for adaptive processing.

    Args:
        query: The user's question

    Returns:
        Detected QuestionType
    """
    query_lower = query.lower()

    # =========================================================================
    # Extended Question Types (General Document QA) - Check first
    # =========================================================================

    # Definition patterns - "What is X?", "Define X"
    definition_patterns = [
        r"what is\s+(?:a|an|the)?\s*\w+",  # "What is X?"
        r"^define\b", r"definition of\b",  # "Define X", "Definition of X"
        r"explain what\b", r"describe what\b",  # "Explain what X is"
        r"란\s*(?:무엇|뭐)", r"이란\s*(?:무엇|뭐)",  # Korean: "X란 무엇"
        r"정의(?:는|가|를)", r"개념(?:은|이)",  # Korean: "정의", "개념"
        r"무엇(?:인가|입니까|인지|이야)",  # Korean: "무엇인가요"
        r"뭐(?:야|예요|인가요)",  # Korean: "뭐야"
    ]
    import re
    if any(re.search(p, query_lower) for p in definition_patterns):
        # Distinguish from simple factoid "What is the capital of X?"
        factoid_indicators = ["capital of", "name of", "수도는", "이름은"]
        if not any(ind in query_lower for ind in factoid_indicators):
            return QuestionType.DEFINITION

    # Procedure patterns - "How to X?", "Steps for X"
    procedure_patterns = [
        r"how (?:do|can|to|should)\s+(?:i|we|you)", r"steps (?:to|for)\b",
        r"process (?:of|for)\b", r"way to\b", r"method (?:to|for|of)\b",
        r"guide (?:to|for)\b", r"tutorial\b", r"instruction",
        r"어떻게\s+(?:하|만들|설정|사용|구현)",  # Korean: "어떻게 하"
        r"방법(?:은|이|을)", r"절차(?:는|가)", r"과정(?:은|이)",  # Korean: "방법", "절차"
        r"하는\s*(?:법|방법)", r"만드는\s*(?:법|방법)",  # Korean: "하는 법"
    ]
    if any(re.search(p, query_lower) for p in procedure_patterns):
        return QuestionType.PROCEDURE

    # Cause/Effect patterns - "Why X?", "What causes X?"
    cause_effect_patterns = [
        r"^why\b", r"why (?:is|are|does|do|did|was|were)\b",
        r"what (?:causes|caused|leads to|led to)\b",
        r"reason (?:for|why|behind)\b", r"because of what\b",
        r"due to what\b", r"result in\b", r"consequence of\b",
        r"^왜\b", r"왜\s+(?:그런|이런|저런)",  # Korean: "왜"
        r"이유(?:는|가|를)", r"원인(?:은|이|을)",  # Korean: "이유", "원인"
        r"때문(?:에|인지)", r"결과(?:로|는)",  # Korean: "때문에"
    ]
    if any(re.search(p, query_lower) for p in cause_effect_patterns):
        return QuestionType.CAUSE_EFFECT

    # List patterns - "List X", "Types of X"
    list_patterns = [
        r"^list\b", r"enumerate\b", r"what are (?:the|some|all)\s+(?:types|kinds|examples)",
        r"give (?:me\s+)?(?:some\s+)?examples of\b",
        r"types of\b", r"kinds of\b", r"categories of\b",
        r"나열", r"종류(?:는|가|를)", r"유형(?:은|이|을)",  # Korean: "나열", "종류"
        r"예(?:시|를)\s*(?:들어|줘)", r"목록",  # Korean: "예시", "목록"
    ]
    if any(re.search(p, query_lower) for p in list_patterns):
        return QuestionType.LIST

    # Narrative/Opinion patterns - Complex explanation, analysis
    narrative_patterns = [
        r"explain\s+(?:in\s+detail|how|why|the)", r"describe\s+(?:in\s+detail|how|the)",
        r"elaborate\b", r"discuss\b", r"analyze\b", r"analyse\b",
        r"what do you think\b", r"your opinion\b", r"assessment of\b",
        r"설명(?:해|하다|해\s*줘)", r"분석(?:해|하다)",  # Korean: "설명해줘", "분석해"
        r"자세히", r"상세(?:히|하게)",  # Korean: "자세히"
    ]
    if any(re.search(p, query_lower) for p in narrative_patterns):
        # Distinguish narrative from opinion
        opinion_indicators = [
            "think", "opinion", "believe", "view", "perspective",
            "생각", "의견", "견해", "평가",
        ]
        if any(ind in query_lower for ind in opinion_indicators):
            return QuestionType.OPINION
        return QuestionType.NARRATIVE

    # =========================================================================
    # Original Question Types (HotpotQA-style)
    # =========================================================================

    # Yes/No patterns
    yesno_patterns = [
        "is ", "are ", "was ", "were ", "does ", "do ", "did ",
        "can ", "could ", "will ", "would ", "has ", "have ", "had ",
        "인가요", "입니까", "맞나요", "인지", "할까요",
    ]
    if any(query_lower.startswith(p) or p in query_lower for p in yesno_patterns):
        # Check if it's a simple yes/no or comparative
        if any(w in query_lower for w in ["both", "같이", "둘 다", "either"]):
            return QuestionType.COMPARISON
        return QuestionType.YESNO

    # Comparison patterns
    comparison_patterns = [
        "compare", "difference", "similar", "versus", "vs", "between",
        "better", "worse", "more", "less", "같은", "다른", "차이",
        "비교", "어느 것이", "which one",
    ]
    if any(p in query_lower for p in comparison_patterns):
        return QuestionType.COMPARISON

    # Multi-hop indicator patterns (require chaining facts)
    multihop_patterns = [
        # English patterns
        "who also", "which also", "that also",
        "where did .* work", "what did .* do after",
        "who was .* 's", "who is .* 's",
        "the .* of the .* of", "in which .* did",
        "born in the same", "worked with", "collaborated",
        # Korean patterns
        "와/과 함께", "의 .* 의", "어디서", "누구와",
        "그 후에", "이전에",
    ]
    if any(p in query_lower for p in multihop_patterns):
        return QuestionType.MULTIHOP

    # Bridge entity patterns (connect two entities through intermediate)
    bridge_patterns = [
        "what connects", "how is .* related to",
        "relationship between", "link between", "connection",
        "연결", "관계", "연관",
    ]
    if any(p in query_lower for p in bridge_patterns):
        return QuestionType.BRIDGE

    # Aggregation patterns
    aggregation_patterns = [
        "how many", "how much", "count", "total", "all",
        "list all", "what are all", "몇 개", "얼마나", "모든", "전부",
    ]
    if any(p in query_lower for p in aggregation_patterns):
        return QuestionType.AGGREGATION

    # Default to factoid for simple questions
    return QuestionType.FACTOID


def is_multihop_question(query: str) -> bool:
    """
    Check if question requires multi-hop reasoning.

    Args:
        query: The user's question

    Returns:
        True if question likely requires multi-hop reasoning
    """
    question_type = detect_question_type(query)
    return question_type in [
        QuestionType.MULTIHOP,
        QuestionType.BRIDGE,
        QuestionType.COMPARISON,
    ]


def create_initial_state(
    query: str,
    max_iterations: int = 5,
    pipeline_id: str | None = None,
) -> MACERState:
    """
    Create an initial MACER state for a new query.

    Args:
        query: The user's question
        max_iterations: Maximum reasoning iterations allowed
        pipeline_id: Optional pipeline identifier

    Returns:
        A properly initialized MACERState
    """
    import uuid

    # Detect question type for adaptive processing
    question_type = detect_question_type(query)

    # For multi-hop questions, increase max iterations if needed
    effective_max_iterations = max_iterations
    if question_type in [QuestionType.MULTIHOP, QuestionType.BRIDGE, QuestionType.COMPARISON]:
        effective_max_iterations = max(max_iterations, 7)  # At least 7 iterations for complex questions

    return MACERState(
        original_query=query,
        current_query=query,
        query_history=[],
        question_type=question_type.value,
        sub_questions=[],
        topic_entities=[],
        retrieved_entities=[],
        current_subgraph=SubGraph(),
        subgraph_history=[],
        evidence=[],
        evidence_rankings={},
        reasoning_chains=[],
        reasoning_path=[],
        sufficiency_score=0.0,
        path_completeness=0.0,
        iteration=0,
        max_iterations=effective_max_iterations,
        should_terminate=False,
        final_answer=None,
        confidence=0.0,
        explanation="",
        answer_type="",
        pipeline_id=pipeline_id or str(uuid.uuid4())[:8],
        errors=[],
        metadata={"question_type": question_type.value},
    )


def merge_evidence_lists(
    existing: list[Evidence],
    new: list[Evidence],
    max_items: int = 30,
) -> list[Evidence]:
    """
    Merge evidence lists immutably, removing duplicates and limiting size.

    Args:
        existing: Current evidence list
        new: New evidence to add
        max_items: Maximum items in result

    Returns:
        Merged and deduplicated evidence list
    """
    existing_ids = {ev.id for ev in existing}
    unique_new = [ev for ev in new if ev.id not in existing_ids]
    merged = existing + unique_new

    # Sort by relevance and limit
    merged.sort(key=lambda e: e.relevance_score, reverse=True)
    return merged[:max_items]


def calculate_path_completeness(
    evidence: list[Evidence],
    subgraph: SubGraph,
    question_type: str,
) -> float:
    """
    Calculate how complete the reasoning paths are for multi-hop questions.

    Args:
        evidence: Collected evidence
        subgraph: Current reasoning subgraph
        question_type: Type of question

    Returns:
        Path completeness score (0.0 to 1.0)
    """
    if not evidence:
        return 0.0

    # For simple factoid questions, path completeness is less critical
    if question_type == QuestionType.FACTOID.value:
        return 1.0 if evidence else 0.0

    # Count evidence with path information
    path_evidence = [e for e in evidence if e.is_part_of_chain()]
    path_ratio = len(path_evidence) / len(evidence) if evidence else 0.0

    # Check subgraph connectivity
    metrics = subgraph.get_subgraph_metrics()
    connectivity_score = 1.0 if metrics["is_connected"] else 0.5

    # Check if we have bridge entities for complex questions
    bridge_score = 1.0
    if question_type in [QuestionType.BRIDGE.value, QuestionType.MULTIHOP.value] and metrics["bridge_entity_count"] == 0:
        bridge_score = 0.3  # Penalize missing bridges

    # Check path length adequacy
    path_length_score = min(1.0, metrics["max_path_length"] / 3)  # Expect 2-3 hop paths

    # Check evidence chain coverage
    chain_coverage = 0.0
    if path_evidence:
        max_hop = max(e.hop_index for e in path_evidence)
        if max_hop > 0:
            # Check we have evidence at each hop level
            hops_covered = {e.hop_index for e in path_evidence}
            chain_coverage = len(hops_covered) / (max_hop + 1)

    # Weighted combination
    weights = {
        "path_ratio": 0.25,
        "connectivity": 0.20,
        "bridge": 0.20,
        "path_length": 0.15,
        "chain_coverage": 0.20,
    }

    completeness = (
        weights["path_ratio"] * path_ratio +
        weights["connectivity"] * connectivity_score +
        weights["bridge"] * bridge_score +
        weights["path_length"] * path_length_score +
        weights["chain_coverage"] * chain_coverage
    )

    return min(1.0, max(0.0, completeness))


def calculate_progress(state: MACERState) -> dict[str, Any]:
    """
    Calculate progress metrics from current state.

    Args:
        state: Current MACER state

    Returns:
        Dict with progress metrics
    """
    evidence = state.get("evidence", [])
    subgraph = state.get("current_subgraph", SubGraph())
    question_type = state.get("question_type", QuestionType.FACTOID.value)

    # Calculate path completeness
    path_completeness = calculate_path_completeness(evidence, subgraph, question_type)

    # Get subgraph metrics
    subgraph_metrics = subgraph.get_subgraph_metrics() if subgraph.nodes else {}

    return {
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 5),
        "progress_percent": (
            state.get("iteration", 0) / max(state.get("max_iterations", 5), 1) * 100
        ),
        "sufficiency_score": state.get("sufficiency_score", 0.0),
        "evidence_count": len(evidence),
        "path_evidence_count": sum(1 for e in evidence if e.is_part_of_chain()),
        "subgraph_size": subgraph.node_count(),
        "subgraph_edge_count": subgraph.edge_count(),
        "max_path_length": subgraph_metrics.get("max_path_length", 0),
        "bridge_entities": subgraph_metrics.get("bridge_entity_count", 0),
        "path_completeness": path_completeness,
        "question_type": question_type,
        "sub_questions_count": len(state.get("sub_questions", [])),
        "reasoning_chains_count": len(state.get("reasoning_chains", [])),
        "errors": len(state.get("errors", [])),
        "should_terminate": state.get("should_terminate", False),
    }

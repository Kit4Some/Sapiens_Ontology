"""
Advanced Response Synthesizer for High-Quality Knowledge Graph Answers.

Transforms raw evidence and graph data into sophisticated, multi-layered responses
with deep analysis, relationship discovery, and contextual explanations.

Features:
- Multi-layer response structure (Summary, Analysis, Connections, Implications)
- Deep graph pattern analysis
- Relationship chain reasoning
- Impact and dependency analysis
- Bilingual support (Korean/English)
- Rich explanations with context
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.tog.state import (
    Evidence,
    EvidenceType,
    MACERState,
    ReasoningStep,
    SubGraph,
    SubGraphEdge,
    SubGraphNode,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Response Structure Types
# =============================================================================


class ResponseDepth(str, Enum):
    """Response detail level."""
    BRIEF = "brief"          # 1-2 sentences
    STANDARD = "standard"    # Paragraph with key points
    DETAILED = "detailed"    # Multi-section with analysis
    COMPREHENSIVE = "comprehensive"  # Full report with all sections


class AnalysisType(str, Enum):
    """Types of analysis in response."""
    DEFINITION = "definition"
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    ARCHITECTURE = "architecture"
    PROCESS = "process"
    IMPACT = "impact"
    TIMELINE = "timeline"


@dataclass
class EntityCluster:
    """Cluster of related entities."""
    name: str
    entities: list[str]
    cluster_type: str  # component, dependency, related, hierarchy
    description: str
    relationships: list[tuple[str, str, str]]  # (source, predicate, target)


@dataclass
class RelationshipChain:
    """Chain of relationships forming a reasoning path."""
    chain_id: str
    entities: list[str]
    predicates: list[str]
    description: str
    confidence: float
    hop_count: int


@dataclass
class PatternInsight:
    """Pattern discovered in the graph."""
    pattern_type: str  # hub, bridge, cluster, hierarchy, cycle
    entities_involved: list[str]
    description: str
    significance: str  # Why this pattern matters


@dataclass
class ImpactAnalysis:
    """Analysis of entity impact and dependencies."""
    entity: str
    direct_dependencies: list[str]
    indirect_dependencies: list[str]
    dependents: list[str]  # Things that depend on this
    impact_description: str


@dataclass
class StructuredSection:
    """A section of the structured response."""
    title: str
    title_ko: str
    content: str
    content_ko: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class AdvancedResponse:
    """Full structured response with all components."""

    # Core answer
    executive_summary: str
    executive_summary_ko: str

    # Detailed sections
    definition_section: StructuredSection | None = None
    architecture_section: StructuredSection | None = None
    relationship_section: StructuredSection | None = None
    component_section: StructuredSection | None = None
    technology_section: StructuredSection | None = None
    process_section: StructuredSection | None = None
    impact_section: StructuredSection | None = None

    # Analysis results
    entity_clusters: list[EntityCluster] = field(default_factory=list)
    relationship_chains: list[RelationshipChain] = field(default_factory=list)
    pattern_insights: list[PatternInsight] = field(default_factory=list)
    impact_analysis: list[ImpactAnalysis] = field(default_factory=list)

    # Metadata
    confidence: float = 0.5
    evidence_count: int = 0
    graph_depth: int = 0
    response_depth: ResponseDepth = ResponseDepth.STANDARD
    analysis_types: list[AnalysisType] = field(default_factory=list)

    # Provenance
    primary_entities: list[str] = field(default_factory=list)
    key_relationships: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)

    def to_formatted_response(self, include_korean: bool = True) -> str:
        """Generate the final formatted response string."""
        sections = []

        # Executive Summary
        if include_korean:
            sections.append(f"## 개요 (Overview)\n\n{self.executive_summary_ko}\n\n{self.executive_summary}")
        else:
            sections.append(f"## Overview\n\n{self.executive_summary}")

        # Definition
        if self.definition_section:
            sections.append(self._format_section(self.definition_section, include_korean))

        # Architecture/Structure
        if self.architecture_section:
            sections.append(self._format_section(self.architecture_section, include_korean))

        # Components
        if self.component_section:
            sections.append(self._format_section(self.component_section, include_korean))

        # Relationships
        if self.relationship_section:
            sections.append(self._format_section(self.relationship_section, include_korean))

        # Technology Stack
        if self.technology_section:
            sections.append(self._format_section(self.technology_section, include_korean))

        # Process/Workflow
        if self.process_section:
            sections.append(self._format_section(self.process_section, include_korean))

        # Impact Analysis
        if self.impact_section:
            sections.append(self._format_section(self.impact_section, include_korean))

        # Pattern Insights
        if self.pattern_insights:
            sections.append(self._format_patterns(include_korean))

        # Relationship Chains
        if self.relationship_chains:
            sections.append(self._format_chains(include_korean))

        # Confidence and Sources
        sections.append(self._format_metadata(include_korean))

        return "\n\n".join(sections)

    def _format_section(self, section: StructuredSection, include_korean: bool) -> str:
        """Format a single section."""
        if include_korean:
            return f"## {section.title_ko} ({section.title})\n\n{section.content_ko}\n\n{section.content}"
        return f"## {section.title}\n\n{section.content}"

    def _format_patterns(self, include_korean: bool) -> str:
        """Format pattern insights."""
        lines = []
        if include_korean:
            lines.append("## 발견된 패턴 (Discovered Patterns)")
        else:
            lines.append("## Discovered Patterns")

        for pattern in self.pattern_insights[:5]:
            lines.append(f"\n### {pattern.pattern_type.title()}")
            lines.append(f"- **Entities**: {', '.join(pattern.entities_involved[:5])}")
            lines.append(f"- **Description**: {pattern.description}")
            lines.append(f"- **Significance**: {pattern.significance}")

        return "\n".join(lines)

    def _format_chains(self, include_korean: bool) -> str:
        """Format relationship chains."""
        lines = []
        if include_korean:
            lines.append("## 관계 체인 (Relationship Chains)")
        else:
            lines.append("## Relationship Chains")

        for chain in self.relationship_chains[:5]:
            chain_str = " → ".join(
                f"{chain.entities[i]} --[{chain.predicates[i]}]→ {chain.entities[i+1]}"
                for i in range(len(chain.predicates))
            ) if chain.predicates else " → ".join(chain.entities)

            lines.append(f"\n**{chain.description}** (confidence: {chain.confidence:.0%})")
            lines.append(f"```\n{chain_str}\n```")

        return "\n".join(lines)

    def _format_metadata(self, include_korean: bool) -> str:
        """Format metadata section."""
        lines = []
        if include_korean:
            lines.append("---")
            lines.append(f"**신뢰도 (Confidence)**: {self.confidence:.0%}")
            lines.append(f"**분석된 엔티티 (Entities Analyzed)**: {len(self.primary_entities)}")
            lines.append(f"**증거 수 (Evidence Count)**: {self.evidence_count}")
            lines.append(f"**그래프 깊이 (Graph Depth)**: {self.graph_depth} hops")
        else:
            lines.append("---")
            lines.append(f"**Confidence**: {self.confidence:.0%}")
            lines.append(f"**Entities Analyzed**: {len(self.primary_entities)}")
            lines.append(f"**Evidence Count**: {self.evidence_count}")
            lines.append(f"**Graph Depth**: {self.graph_depth} hops")

        return "\n".join(lines)


# =============================================================================
# Advanced Prompts for Rich Response Generation
# =============================================================================


DEEP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert knowledge analyst that creates comprehensive, insightful responses.

## Task
Generate a sophisticated, multi-layered analysis of the topic based on knowledge graph evidence.
Your response should be rich with insights, not just a list of facts.

## Input
- Question: {question}
- Topic Entity: {topic_entity}
- Graph Evidence: {graph_evidence}
- Relationships Found: {relationships}
- Entity Clusters: {entity_clusters}
- Patterns Discovered: {patterns}

## Response Requirements

### 1. Executive Summary (2-3 sentences)
Capture the essence of what this entity/topic IS and WHY it matters.
Start with "X는..." in Korean and "X is..." in English.

### 2. Definition & Core Concept
- What is it fundamentally?
- What problem does it solve?
- What category/domain does it belong to?

### 3. Architecture & Structure (if applicable)
- What are its main components?
- How do components interact?
- What is the hierarchy?
Draw ASCII diagrams if helpful.

### 4. Relationships & Dependencies
- What does it depend on?
- What depends on it?
- How does it relate to similar things?
Explain the significance of key relationships.

### 5. Technology & Implementation (if applicable)
- What technologies are used?
- Why were these choices made?
- What are the technical implications?

### 6. Impact & Implications
- Why is this important?
- What are the consequences?
- What should users know?

## Writing Style
- Use clear structure with headers
- Explain connections, not just list facts
- Provide context for why things matter
- Use analogies when helpful
- Be specific with examples from evidence

## Output Format
Return a JSON object with:
{{
    "executive_summary": "2-3 sentence summary in English",
    "executive_summary_ko": "2-3 sentence summary in Korean",
    "definition": {{
        "content": "English definition section",
        "content_ko": "Korean definition section"
    }},
    "architecture": {{
        "content": "English architecture description with ASCII diagram if applicable",
        "content_ko": "Korean architecture description"
    }},
    "components": {{
        "content": "English component analysis",
        "content_ko": "Korean component analysis"
    }},
    "relationships": {{
        "content": "English relationship analysis",
        "content_ko": "Korean relationship analysis"
    }},
    "technology": {{
        "content": "English technology analysis",
        "content_ko": "Korean technology analysis"
    }},
    "impact": {{
        "content": "English impact analysis",
        "content_ko": "Korean impact analysis"
    }},
    "key_insights": ["insight1", "insight2", "insight3"],
    "confidence": 0.0-1.0
}}

IMPORTANT: Generate SUBSTANTIVE content, not placeholders. Each section should have at least 3-4 sentences of meaningful analysis."""),
    ("human", "Generate comprehensive analysis:")
])


RELATIONSHIP_CHAIN_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at analyzing relationship chains in knowledge graphs.

## Task
Analyze the relationship chains and explain what they reveal about the topic.

## Input
- Topic Entity: {topic_entity}
- Relationship Chains: {chains}
- Question Context: {question}

## Analysis Requirements
For each significant chain:
1. Describe what the chain reveals
2. Explain why this connection matters
3. Identify any patterns (e.g., dependency chains, hierarchies)

## Output Format
Return a JSON object with:
{{
    "chain_analyses": [
        {{
            "chain_description": "human-readable description",
            "chain_description_ko": "Korean description",
            "significance": "why this matters",
            "pattern_type": "dependency/hierarchy/association/composition"
        }}
    ],
    "overall_insight": "what these chains collectively reveal",
    "overall_insight_ko": "Korean overall insight"
}}"""),
    ("human", "Analyze the relationship chains:")
])


PATTERN_DISCOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at discovering meaningful patterns in knowledge graphs.

## Task
Identify and explain significant patterns in the graph structure.

## Input
- Central Entity: {topic_entity}
- Nodes: {nodes}
- Edges: {edges}
- Entity Types Distribution: {type_distribution}

## Patterns to Look For
1. **Hub Pattern**: Entity with many connections
2. **Bridge Pattern**: Entity connecting otherwise separate clusters
3. **Hierarchy Pattern**: Clear parent-child relationships
4. **Cluster Pattern**: Groups of tightly connected entities
5. **Chain Pattern**: Linear sequences of relationships
6. **Star Pattern**: Central entity with many direct connections

## Output Format
Return a JSON object with:
{{
    "patterns": [
        {{
            "pattern_type": "hub|bridge|hierarchy|cluster|chain|star",
            "entities_involved": ["entity1", "entity2"],
            "description": "what this pattern represents",
            "description_ko": "Korean description",
            "significance": "why this pattern matters",
            "significance_ko": "Korean significance"
        }}
    ],
    "graph_characteristics": {{
        "is_densely_connected": true/false,
        "has_clear_hierarchy": true/false,
        "dominant_pattern": "most prominent pattern type",
        "complexity_level": "simple|moderate|complex"
    }}
}}"""),
    ("human", "Discover patterns in the graph:")
])


# =============================================================================
# Advanced Response Synthesizer
# =============================================================================


class AdvancedResponseSynthesizer:
    """
    Advanced synthesizer that creates rich, multi-layered responses
    from knowledge graph evidence.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        default_depth: ResponseDepth = ResponseDepth.DETAILED,
        include_korean: bool = True,
        max_chains: int = 10,
        max_patterns: int = 5,
    ):
        self._llm = llm
        self._default_depth = default_depth
        self._include_korean = include_korean
        self._max_chains = max_chains
        self._max_patterns = max_patterns

        # Build chains
        self._json_parser = JsonOutputParser()
        self._str_parser = StrOutputParser()

        self._deep_analysis_chain = DEEP_ANALYSIS_PROMPT | self._llm | self._json_parser
        self._chain_analysis_chain = RELATIONSHIP_CHAIN_ANALYSIS_PROMPT | self._llm | self._json_parser
        self._pattern_discovery_chain = PATTERN_DISCOVERY_PROMPT | self._llm | self._json_parser

    # =========================================================================
    # Graph Analysis Methods
    # =========================================================================

    def analyze_subgraph_structure(
        self,
        subgraph: SubGraph,
    ) -> dict[str, Any]:
        """
        Analyze the structure of the subgraph.

        Returns detailed structural metrics and characteristics.
        """
        if not subgraph or subgraph.node_count() == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0,
                "type_distribution": {},
                "hub_entities": [],
                "bridge_entities": [],
                "max_depth": 0,
            }

        # Count edges per node
        edge_counts: dict[str, int] = defaultdict(int)
        for edge in subgraph.edges:
            edge_counts[edge.source_id] += 1
            edge_counts[edge.target_id] += 1

        # Find hubs (nodes with many connections)
        avg_edges = sum(edge_counts.values()) / len(edge_counts) if edge_counts else 0
        hub_threshold = max(3, avg_edges * 1.5)
        hub_entities = [
            node.name for node in subgraph.nodes
            if edge_counts.get(node.id, 0) >= hub_threshold
        ]

        # Type distribution
        type_distribution: dict[str, int] = defaultdict(int)
        for node in subgraph.nodes:
            type_distribution[node.type] += 1

        # Find bridge entities
        bridge_entities = [e.name for e in subgraph.get_bridge_entities()]

        # Calculate max depth
        max_depth = max((n.distance_from_topic for n in subgraph.nodes), default=0)

        # Calculate density
        n = subgraph.node_count()
        e = subgraph.edge_count()
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        density = e / max_edges if max_edges > 0 else 0

        return {
            "node_count": n,
            "edge_count": e,
            "density": density,
            "type_distribution": dict(type_distribution),
            "hub_entities": hub_entities,
            "bridge_entities": bridge_entities,
            "max_depth": max_depth,
            "avg_connections": avg_edges,
        }

    def extract_entity_clusters(
        self,
        subgraph: SubGraph,
        evidence: list[Evidence],
    ) -> list[EntityCluster]:
        """
        Extract clusters of related entities from the graph.
        """
        clusters = []

        # Group by entity type
        type_groups: dict[str, list[SubGraphNode]] = defaultdict(list)
        for node in subgraph.nodes:
            type_groups[node.type].append(node)

        # Create clusters for each type with 2+ entities
        for entity_type, nodes in type_groups.items():
            if len(nodes) >= 2:
                # Find relationships within cluster
                node_ids = {n.id for n in nodes}
                cluster_relationships = []
                for edge in subgraph.edges:
                    if edge.source_id in node_ids and edge.target_id in node_ids:
                        source_name = next((n.name for n in nodes if n.id == edge.source_id), "?")
                        target_name = next((n.name for n in nodes if n.id == edge.target_id), "?")
                        cluster_relationships.append(
                            (source_name, edge.predicate or edge.relation_type, target_name)
                        )

                clusters.append(EntityCluster(
                    name=f"{entity_type} Cluster",
                    entities=[n.name for n in nodes],
                    cluster_type="type_group",
                    description=f"Group of {len(nodes)} {entity_type} entities",
                    relationships=cluster_relationships,
                ))

        # Find component clusters (things that are parts of something)
        component_edges = [
            e for e in subgraph.edges
            if any(kw in (e.predicate or "").lower()
                   for kw in ["포함", "구성", "include", "contain", "part", "component", "has"])
        ]

        if component_edges:
            components = set()
            parents = set()
            for edge in component_edges:
                source = subgraph.get_node(edge.source_id)
                target = subgraph.get_node(edge.target_id)
                if source and target:
                    parents.add(source.name)
                    components.add(target.name)

            if components:
                clusters.append(EntityCluster(
                    name="Component Cluster",
                    entities=list(components),
                    cluster_type="component",
                    description=f"Components/parts of: {', '.join(list(parents)[:3])}",
                    relationships=[(e.source_id, e.predicate or "contains", e.target_id)
                                   for e in component_edges[:10]],
                ))

        # Find dependency clusters
        dependency_edges = [
            e for e in subgraph.edges
            if any(kw in (e.predicate or "").lower()
                   for kw in ["사용", "의존", "use", "depend", "require", "based on"])
        ]

        if dependency_edges:
            dependencies = set()
            dependents = set()
            for edge in dependency_edges:
                source = subgraph.get_node(edge.source_id)
                target = subgraph.get_node(edge.target_id)
                if source and target:
                    dependents.add(source.name)
                    dependencies.add(target.name)

            if dependencies:
                clusters.append(EntityCluster(
                    name="Dependency Cluster",
                    entities=list(dependencies),
                    cluster_type="dependency",
                    description=f"Dependencies used by: {', '.join(list(dependents)[:3])}",
                    relationships=[(e.source_id, e.predicate or "uses", e.target_id)
                                   for e in dependency_edges[:10]],
                ))

        return clusters

    def extract_relationship_chains(
        self,
        subgraph: SubGraph,
        topic_entity_ids: list[str],
    ) -> list[RelationshipChain]:
        """
        Extract meaningful relationship chains from the graph.
        """
        chains = []
        chain_id = 0

        if not topic_entity_ids or subgraph.node_count() < 2:
            return chains

        # Find paths from each topic entity
        for start_id in topic_entity_ids[:3]:
            # BFS to find paths
            visited = {start_id}
            queue = [(start_id, [start_id], [])]  # (current, path, predicates)

            while queue and chain_id < self._max_chains:
                current, path, predicates = queue.pop(0)

                if len(path) >= 2:
                    # Create chain from this path
                    entity_names = []
                    for node_id in path:
                        node = subgraph.get_node(node_id)
                        entity_names.append(node.name if node else node_id)

                    # Generate description
                    if predicates:
                        desc = f"{entity_names[0]} "
                        for i, pred in enumerate(predicates):
                            desc += f"--[{pred}]--> {entity_names[i+1]} "
                    else:
                        desc = " → ".join(entity_names)

                    chains.append(RelationshipChain(
                        chain_id=f"chain_{chain_id}",
                        entities=entity_names,
                        predicates=predicates,
                        description=desc.strip(),
                        confidence=0.8 - (len(path) - 2) * 0.1,  # Decay with length
                        hop_count=len(path) - 1,
                    ))
                    chain_id += 1

                # Continue BFS
                if len(path) < 5:  # Max depth
                    for edge in subgraph.edges:
                        next_id = None
                        pred = edge.predicate or edge.relation_type

                        if edge.source_id == current and edge.target_id not in visited:
                            next_id = edge.target_id
                        elif edge.target_id == current and edge.source_id not in visited:
                            next_id = edge.source_id

                        if next_id:
                            visited.add(next_id)
                            queue.append((next_id, path + [next_id], predicates + [pred]))

        # Sort by confidence
        chains.sort(key=lambda c: c.confidence, reverse=True)
        return chains[:self._max_chains]

    def analyze_impact(
        self,
        subgraph: SubGraph,
        topic_entity_ids: list[str],
    ) -> list[ImpactAnalysis]:
        """
        Analyze the impact and dependencies of topic entities.
        """
        analyses = []

        for entity_id in topic_entity_ids[:5]:
            entity = subgraph.get_node(entity_id)
            if not entity:
                continue

            direct_deps = []
            indirect_deps = []
            dependents = []

            # Find direct dependencies (things this entity uses)
            for edge in subgraph.edges:
                if edge.source_id == entity_id:
                    target = subgraph.get_node(edge.target_id)
                    if target:
                        pred = (edge.predicate or "").lower()
                        if any(kw in pred for kw in ["use", "depend", "require", "based", "사용"]):
                            direct_deps.append(target.name)
                        elif any(kw in pred for kw in ["include", "contain", "has", "포함"]):
                            pass  # These are components, not dependencies
                        else:
                            direct_deps.append(target.name)

                elif edge.target_id == entity_id:
                    source = subgraph.get_node(edge.source_id)
                    if source:
                        pred = (edge.predicate or "").lower()
                        if any(kw in pred for kw in ["use", "depend", "require", "사용"]):
                            dependents.append(source.name)

            # Find indirect dependencies (2-hop)
            for dep_name in direct_deps:
                dep_node = next((n for n in subgraph.nodes if n.name == dep_name), None)
                if dep_node:
                    for edge in subgraph.edges:
                        if edge.source_id == dep_node.id:
                            target = subgraph.get_node(edge.target_id)
                            if target and target.name not in direct_deps:
                                indirect_deps.append(target.name)

            # Generate description
            if direct_deps or dependents:
                desc_parts = []
                if direct_deps:
                    desc_parts.append(f"Depends on: {', '.join(direct_deps[:5])}")
                if dependents:
                    desc_parts.append(f"Required by: {', '.join(dependents[:5])}")
                if indirect_deps:
                    desc_parts.append(f"Transitively depends on: {', '.join(indirect_deps[:3])}")
                description = ". ".join(desc_parts)
            else:
                description = "No significant dependencies detected."

            analyses.append(ImpactAnalysis(
                entity=entity.name,
                direct_dependencies=direct_deps,
                indirect_dependencies=indirect_deps[:5],
                dependents=dependents,
                impact_description=description,
            ))

        return analyses

    # =========================================================================
    # Response Generation
    # =========================================================================

    async def generate_deep_analysis(
        self,
        question: str,
        topic_entity: str,
        evidence: list[Evidence],
        clusters: list[EntityCluster],
        patterns: list[dict],
        relationships: list[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """
        Generate deep analysis using LLM.
        """
        # Format evidence
        evidence_str = "\n".join([
            f"- [{e.evidence_type.value}] {e.content[:200]}"
            for e in sorted(evidence, key=lambda x: x.relevance_score, reverse=True)[:15]
        ])

        # Format relationships
        rel_str = "\n".join([
            f"- {src} --[{pred}]--> {tgt}"
            for src, pred, tgt in relationships[:20]
        ])

        # Format clusters
        cluster_str = "\n".join([
            f"- {c.name}: {', '.join(c.entities[:5])}"
            for c in clusters[:5]
        ])

        # Format patterns
        pattern_str = "\n".join([
            f"- {p.get('pattern_type', 'unknown')}: {p.get('description', '')}"
            for p in patterns[:5]
        ]) if patterns else "No significant patterns detected."

        try:
            result = await self._deep_analysis_chain.ainvoke({
                "question": question,
                "topic_entity": topic_entity,
                "graph_evidence": evidence_str,
                "relationships": rel_str,
                "entity_clusters": cluster_str,
                "patterns": pattern_str,
            })
            return result
        except Exception as e:
            logger.error("Deep analysis generation failed", error=str(e))
            return self._generate_fallback_analysis(topic_entity, evidence, relationships)

    def _generate_fallback_analysis(
        self,
        topic_entity: str,
        evidence: list[Evidence],
        relationships: list[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """Generate fallback analysis when LLM fails."""

        # Extract facts from evidence
        facts = []
        for ev in evidence[:10]:
            if ev.evidence_type == EvidenceType.DIRECT:
                facts.append(ev.content[:150])

        # Build basic sections
        summary = f"{topic_entity}은(는) 지식 그래프에서 {len(evidence)}개의 증거와 연결되어 있습니다."
        summary_en = f"{topic_entity} is connected to {len(evidence)} pieces of evidence in the knowledge graph."

        return {
            "executive_summary": summary_en,
            "executive_summary_ko": summary,
            "definition": {
                "content": f"Based on available evidence: {'; '.join(facts[:3])}",
                "content_ko": f"수집된 증거를 바탕으로: {'; '.join(facts[:3])}",
            },
            "relationships": {
                "content": f"Found {len(relationships)} relationships.",
                "content_ko": f"{len(relationships)}개의 관계가 발견되었습니다.",
            },
            "key_insights": facts[:3],
            "confidence": 0.5,
        }

    async def discover_patterns(
        self,
        topic_entity: str,
        subgraph: SubGraph,
    ) -> list[dict]:
        """
        Discover patterns in the graph using LLM.
        """
        if subgraph.node_count() < 3:
            return []

        # Prepare inputs
        nodes_str = "\n".join([
            f"- {n.name} ({n.type})"
            for n in subgraph.nodes[:30]
        ])

        edges_str = "\n".join([
            f"- {e.source_id} --[{e.predicate or e.relation_type}]--> {e.target_id}"
            for e in subgraph.edges[:30]
        ])

        # Type distribution
        type_dist = defaultdict(int)
        for n in subgraph.nodes:
            type_dist[n.type] += 1

        try:
            result = await self._pattern_discovery_chain.ainvoke({
                "topic_entity": topic_entity,
                "nodes": nodes_str,
                "edges": edges_str,
                "type_distribution": dict(type_dist),
            })
            return result.get("patterns", [])
        except Exception as e:
            logger.warning("Pattern discovery failed", error=str(e))
            return []

    async def synthesize_advanced_response(
        self,
        state: MACERState,
        depth: ResponseDepth | None = None,
    ) -> AdvancedResponse:
        """
        Main method to synthesize an advanced, structured response.
        """
        depth = depth or self._default_depth

        question = state.get("original_query", "")
        evidence = state.get("evidence", [])
        subgraph = state.get("current_subgraph", SubGraph())
        topic_entities = state.get("topic_entities", [])
        reasoning_path = state.get("reasoning_path", [])

        # Get topic entity info
        topic_entity_ids = [te.id for te in topic_entities if hasattr(te, "id")]
        topic_entity_names = [te.name for te in topic_entities if hasattr(te, "name")]
        primary_topic = topic_entity_names[0] if topic_entity_names else "Unknown"

        logger.info(
            "Starting advanced synthesis",
            question=question[:50],
            evidence_count=len(evidence),
            subgraph_nodes=subgraph.node_count(),
            topic_entities=topic_entity_names[:3],
        )

        # 1. Analyze graph structure
        structure = self.analyze_subgraph_structure(subgraph)

        # 2. Extract entity clusters
        clusters = self.extract_entity_clusters(subgraph, evidence)

        # 3. Extract relationship chains
        chains = self.extract_relationship_chains(subgraph, topic_entity_ids)

        # 4. Analyze impact
        impacts = self.analyze_impact(subgraph, topic_entity_ids)

        # 5. Discover patterns (async)
        patterns = await self.discover_patterns(primary_topic, subgraph)

        # 6. Extract relationships for analysis
        relationships = [
            (
                subgraph.get_node(e.source_id).name if subgraph.get_node(e.source_id) else e.source_id,
                e.predicate or e.relation_type,
                subgraph.get_node(e.target_id).name if subgraph.get_node(e.target_id) else e.target_id,
            )
            for e in subgraph.edges
        ]

        # 7. Generate deep analysis with LLM
        analysis = await self.generate_deep_analysis(
            question=question,
            topic_entity=primary_topic,
            evidence=evidence,
            clusters=clusters,
            patterns=patterns,
            relationships=relationships,
        )

        # 8. Build the response
        response = AdvancedResponse(
            executive_summary=analysis.get("executive_summary", ""),
            executive_summary_ko=analysis.get("executive_summary_ko", ""),
            confidence=analysis.get("confidence", 0.5),
            evidence_count=len(evidence),
            graph_depth=structure.get("max_depth", 0),
            response_depth=depth,
            primary_entities=topic_entity_names,
            key_relationships=[f"{s} → {p} → {t}" for s, p, t in relationships[:10]],
            sources_used=[e.id for e in evidence[:10]],
        )

        # Add sections from analysis
        if analysis.get("definition"):
            response.definition_section = StructuredSection(
                title="Definition & Core Concept",
                title_ko="정의 및 핵심 개념",
                content=analysis["definition"].get("content", ""),
                content_ko=analysis["definition"].get("content_ko", ""),
            )

        if analysis.get("architecture"):
            response.architecture_section = StructuredSection(
                title="Architecture & Structure",
                title_ko="아키텍처 및 구조",
                content=analysis["architecture"].get("content", ""),
                content_ko=analysis["architecture"].get("content_ko", ""),
            )

        if analysis.get("components"):
            response.component_section = StructuredSection(
                title="Components",
                title_ko="구성 요소",
                content=analysis["components"].get("content", ""),
                content_ko=analysis["components"].get("content_ko", ""),
            )

        if analysis.get("relationships"):
            response.relationship_section = StructuredSection(
                title="Relationships & Dependencies",
                title_ko="관계 및 의존성",
                content=analysis["relationships"].get("content", ""),
                content_ko=analysis["relationships"].get("content_ko", ""),
            )

        if analysis.get("technology"):
            response.technology_section = StructuredSection(
                title="Technology Stack",
                title_ko="기술 스택",
                content=analysis["technology"].get("content", ""),
                content_ko=analysis["technology"].get("content_ko", ""),
            )

        if analysis.get("impact"):
            response.impact_section = StructuredSection(
                title="Impact & Implications",
                title_ko="영향 및 의미",
                content=analysis["impact"].get("content", ""),
                content_ko=analysis["impact"].get("content_ko", ""),
            )

        # Add analysis results
        response.entity_clusters = clusters
        response.relationship_chains = chains
        response.impact_analysis = impacts

        # Convert patterns to PatternInsight objects
        for p in patterns[:self._max_patterns]:
            response.pattern_insights.append(PatternInsight(
                pattern_type=p.get("pattern_type", "unknown"),
                entities_involved=p.get("entities_involved", []),
                description=p.get("description", ""),
                significance=p.get("significance", ""),
            ))

        logger.info(
            "Advanced synthesis completed",
            confidence=response.confidence,
            sections_generated=sum([
                1 for s in [
                    response.definition_section,
                    response.architecture_section,
                    response.component_section,
                    response.relationship_section,
                    response.technology_section,
                    response.impact_section,
                ] if s is not None
            ]),
            clusters=len(response.entity_clusters),
            chains=len(response.relationship_chains),
            patterns=len(response.pattern_insights),
        )

        return response

    def format_response(
        self,
        response: AdvancedResponse,
        include_korean: bool | None = None,
    ) -> str:
        """
        Format the advanced response into a readable string.
        """
        if include_korean is None:
            include_korean = self._include_korean

        return response.to_formatted_response(include_korean)


# =============================================================================
# Factory Function
# =============================================================================


def create_advanced_synthesizer(
    llm: BaseChatModel,
    response_depth: ResponseDepth = ResponseDepth.DETAILED,
    include_korean: bool = True,
) -> AdvancedResponseSynthesizer:
    """
    Create an advanced response synthesizer.

    Args:
        llm: Language model for analysis
        response_depth: Default response detail level
        include_korean: Include Korean translations

    Returns:
        Configured AdvancedResponseSynthesizer
    """
    return AdvancedResponseSynthesizer(
        llm=llm,
        default_depth=response_depth,
        include_korean=include_korean,
    )

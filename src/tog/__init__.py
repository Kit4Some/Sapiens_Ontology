"""
ToG (Think-on-Graph) 3.0 MACER Module.

Meta-cognitive Adaptive Chain-of-thought with Evidence-based Reasoning.

Architecture:
    Constructor → Retriever → Reflector ←→ (loop) → Responser

Components:
- State: MACERState and supporting models
- Agents: Constructor, Retriever, Reflector, Responser
- Prompts: LLM prompt templates for each agent
- Advanced Synthesizer: Multi-layer response generation with deep graph analysis (NEW)
"""

from src.tog.agents import (
    ConstructorAgent,
    ReflectorAgent,
    ResponserAgent,
    RetrievalStrategy,
    RetrieverAgent,
)
from src.tog.state import (
    Evidence,
    EvidenceType,
    # State
    MACERState,
    QueryEvolution,
    # Enums
    ReasoningAction,
    ReasoningStep,
    SubGraph,
    SubGraphEdge,
    SubGraphNode,
    SufficiencyAssessment,
    # Models
    TopicEntity,
)
from src.tog.advanced_synthesizer import (
    AdvancedResponseSynthesizer,
    AdvancedResponse,
    ResponseDepth,
    EntityCluster,
    RelationshipChain,
    PatternInsight,
    ImpactAnalysis,
    create_advanced_synthesizer,
)

__all__ = [
    # Enums
    "ReasoningAction",
    "EvidenceType",
    "ResponseDepth",
    # Models
    "TopicEntity",
    "SubGraphNode",
    "SubGraphEdge",
    "SubGraph",
    "Evidence",
    "ReasoningStep",
    "QueryEvolution",
    "SufficiencyAssessment",
    # Advanced Synthesis Models
    "AdvancedResponse",
    "EntityCluster",
    "RelationshipChain",
    "PatternInsight",
    "ImpactAnalysis",
    # State
    "MACERState",
    # Agents
    "ConstructorAgent",
    "RetrieverAgent",
    "RetrievalStrategy",
    "ReflectorAgent",
    "ResponserAgent",
    # Advanced Synthesizer
    "AdvancedResponseSynthesizer",
    "create_advanced_synthesizer",
]

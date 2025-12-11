"""
ToG 3.0 MACER Agents Module.

MACER = Meta-cognitive Adaptive Chain-of-thought with Evidence-based Reasoning

Agents:
- Constructor: Topic entity extraction and seed subgraph construction
- Retriever: Multi-strategy evidence retrieval (vector, graph, community)
- Reflector: Meta-cognitive assessment and adaptive reasoning control
- Responser: Evidence synthesis and answer generation
"""

from src.tog.agents.constructor import ConstructorAgent
from src.tog.agents.reflector import ReflectorAgent
from src.tog.agents.responser import ResponserAgent
from src.tog.agents.retriever import RetrievalStrategy, RetrieverAgent

__all__ = [
    "ConstructorAgent",
    "RetrieverAgent",
    "RetrievalStrategy",
    "ReflectorAgent",
    "ResponserAgent",
]

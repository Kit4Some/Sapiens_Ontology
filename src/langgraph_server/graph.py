"""
LangGraph Server Graph Definition.

Exports the compiled MACER workflow graph for LangGraph Server.
This graph implements ToG 3.0 (Think-on-Graph) with MACER framework:
- Constructor: Extract topic entities and build seed subgraph
- Retriever: Multi-strategy evidence retrieval
- Reflector: Meta-cognitive assessment and iteration control
- Responser: Evidence synthesis and answer generation
"""

import os
import uuid
from typing import Any, Literal

import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from src.graph.neo4j_client import get_ontology_client
from src.tog.agents import (
    ConstructorAgent,
    ReflectorAgent,
    ResponserAgent,
    RetrieverAgent,
)
from src.tog.state import MACERState, SubGraph

logger = structlog.get_logger(__name__)

# =============================================================================
# Configuration from Environment
# =============================================================================

OPENAI_API_KEY = os.getenv("LLM_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_REASONING_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-3-small")
SUFFICIENCY_THRESHOLD = float(os.getenv("TOG_CONFIDENCE_THRESHOLD", "0.75"))
MAX_ITERATIONS = int(os.getenv("TOG_MAX_REASONING_DEPTH", "5"))

# Deterministic response settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_SEED = int(os.getenv("LLM_SEED", "42"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))

# =============================================================================
# Initialize LLM and Embeddings
# =============================================================================


def get_llm() -> ChatOpenAI:
    """Get configured LLM instance."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        top_p=LLM_TOP_P,
        api_key=OPENAI_API_KEY,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Get configured embeddings instance."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


# =============================================================================
# Initialize Agents
# =============================================================================

_llm = get_llm()
_embeddings = get_embeddings()
_neo4j_client = get_ontology_client()

_constructor = ConstructorAgent(llm=_llm, neo4j_client=_neo4j_client)
_retriever = RetrieverAgent(llm=_llm, embeddings=_embeddings, neo4j_client=_neo4j_client)
_reflector = ReflectorAgent(
    llm=_llm,
    sufficiency_threshold=SUFFICIENCY_THRESHOLD,
    max_iterations=MAX_ITERATIONS,
)
_responser = ResponserAgent(llm=_llm)


# =============================================================================
# Node Definitions
# =============================================================================


async def constructor_node(state: MACERState) -> dict[str, Any]:
    """
    Constructor Node: Extract topic entities and build seed subgraph.

    Input: original_query
    Output: topic_entities, retrieved_entities, current_subgraph
    """
    logger.info(
        "=== CONSTRUCTOR NODE START ===",
        query=state.get("original_query", "")[:50],
        iteration_in=state.get("iteration", 0),
    )
    try:
        result = await _constructor.construct(state)
        logger.info(
            "=== CONSTRUCTOR NODE END ===",
            topic_entities=len(result.get("topic_entities", [])),
            retrieved_entities=len(result.get("retrieved_entities", [])),
            subgraph_nodes=result.get("current_subgraph", SubGraph()).node_count(),
        )
        return result
    except Exception as e:
        logger.error("Constructor failed", error=str(e))
        return {
            "errors": state.get("errors", []) + [f"Constructor error: {str(e)}"],
            "topic_entities": [],
            "current_subgraph": SubGraph(),
            "iteration": 0,
        }


async def retriever_node(state: MACERState) -> dict[str, Any]:
    """
    Retriever Node: Execute retrieval strategies and collect evidence.

    Strategies: Vector search, Graph traversal, Community search
    """
    iteration = state.get("iteration", 0)
    logger.info(
        "=== RETRIEVER NODE START ===",
        iteration_in=iteration,
        evidence_count_in=len(state.get("evidence", [])),
        subgraph_nodes_in=state.get("current_subgraph", SubGraph()).node_count(),
    )
    try:
        result = await _retriever.retrieve(state)
        logger.info(
            "=== RETRIEVER NODE END ===",
            evidence_count_out=len(result.get("evidence", [])),
            subgraph_nodes_out=result.get("current_subgraph", SubGraph()).node_count(),
        )
        return result
    except Exception as e:
        logger.error("Retriever failed", error=str(e))
        return {
            "errors": state.get("errors", []) + [f"Retriever error: {str(e)}"],
        }


async def reflector_node(state: MACERState) -> dict[str, Any]:
    """
    Reflector Node: Meta-cognitive assessment and iteration control.

    Evaluates evidence sufficiency and decides next action.
    """
    iteration = state.get("iteration", 0)
    logger.info(
        "=== REFLECTOR NODE START ===",
        iteration_in=iteration,
        evidence_count=len(state.get("evidence", [])),
        current_sufficiency=state.get("sufficiency_score", 0.0),
    )
    try:
        result = await _reflector.reflect(state)
        logger.info(
            "=== REFLECTOR NODE END ===",
            iteration_out=result.get("iteration", 0),
            sufficiency_out=result.get("sufficiency_score", 0.0),
            should_terminate=result.get("should_terminate", False),
        )
        return result
    except Exception as e:
        logger.error("Reflector failed", error=str(e))
        return {
            "errors": state.get("errors", []) + [f"Reflector error: {str(e)}"],
            "should_terminate": True,
        }


async def responser_node(state: MACERState) -> dict[str, Any]:
    """
    Responser Node: Synthesize evidence and generate final answer.
    """
    logger.info(
        "=== RESPONSER NODE START ===",
        iteration_final=state.get("iteration", 0),
        evidence_count=len(state.get("evidence", [])),
        retrieved_entities=len(state.get("retrieved_entities", [])),
        sufficiency_score=state.get("sufficiency_score", 0.0),
    )
    try:
        result = await _responser.respond(state)
        logger.info(
            "=== RESPONSER NODE END ===",
            confidence=result.get("confidence", 0.0),
            answer_type=result.get("answer_type", "UNKNOWN"),
            answer_length=len(result.get("final_answer", "") or ""),
        )
        return result
    except Exception as e:
        logger.error("Responser failed", error=str(e))
        return {
            "errors": state.get("errors", []) + [f"Responser error: {str(e)}"],
            "final_answer": f"Unable to generate answer due to error: {str(e)}",
            "confidence": 0.1,
        }


# =============================================================================
# Routing Logic
# =============================================================================


def should_continue(state: MACERState) -> Literal["continue", "respond"]:
    """
    Determine whether to continue reasoning or generate response.

    Termination conditions:
    1. should_terminate flag is True
    2. Max iterations reached
    3. Sufficiency threshold met
    """
    should_terminate = state.get("should_terminate", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)
    sufficiency = state.get("sufficiency_score", 0.0)
    evidence_count = len(state.get("evidence", []))

    logger.info(
        "=== ROUTING DECISION ===",
        iteration=iteration,
        max_iterations=max_iterations,
        sufficiency=f"{sufficiency:.2%}",
        threshold=f"{SUFFICIENCY_THRESHOLD:.2%}",
        should_terminate=should_terminate,
        evidence_count=evidence_count,
    )

    if should_terminate:
        logger.info(">>> ROUTE: responser (should_terminate=True)")
        return "respond"

    if iteration >= max_iterations:
        logger.info(">>> ROUTE: responser (max iterations reached)")
        return "respond"

    if sufficiency >= SUFFICIENCY_THRESHOLD:
        logger.info(">>> ROUTE: responser (sufficiency threshold met)")
        return "respond"

    logger.info(
        ">>> ROUTE: retriever (continuing reasoning)",
        remaining=max_iterations - iteration,
    )
    return "continue"


# =============================================================================
# Build Graph
# =============================================================================


def build_macer_graph() -> StateGraph[MACERState]:
    """
    Build the MACER reasoning workflow graph.

    Flow: Constructor → Retriever → Reflector ↔ (loop) → Responser
    """
    workflow: StateGraph[MACERState] = StateGraph(MACERState)

    # Add nodes
    workflow.add_node("constructor", constructor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("responser", responser_node)

    # Set entry point
    workflow.set_entry_point("constructor")

    # Linear edges
    workflow.add_edge("constructor", "retriever")
    workflow.add_edge("retriever", "reflector")

    # Conditional edge from reflector
    workflow.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "continue": "retriever",
            "respond": "responser",
        },
    )

    # Terminal edge
    workflow.add_edge("responser", END)

    return workflow


# =============================================================================
# Compiled Graph Export (for LangGraph Server)
# =============================================================================

# Build and compile the graph
# Note: LangGraph Server handles persistence automatically,
# so we don't provide a custom checkpointer here.
_workflow = build_macer_graph()

# Export the compiled graph for LangGraph Server
graph = _workflow.compile()


# =============================================================================
# Utility Functions
# =============================================================================


def create_initial_state(query: str, max_iterations: int = MAX_ITERATIONS) -> MACERState:
    """Create initial state for a new reasoning session."""
    return {
        "original_query": query,
        "current_query": query,
        "query_history": [],
        "topic_entities": [],
        "retrieved_entities": [],
        "current_subgraph": SubGraph(),
        "subgraph_history": [],
        "evidence": [],
        "evidence_rankings": {},
        "reasoning_path": [],
        "sufficiency_score": 0.0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_terminate": False,
        "final_answer": None,
        "confidence": 0.0,
        "explanation": "",
        "pipeline_id": str(uuid.uuid4())[:8],
        "errors": [],
        "metadata": {},
    }


async def run_macer(query: str, thread_id: str | None = None) -> MACERState:
    """
    Run the MACER workflow for a given query.

    Args:
        query: Natural language question
        thread_id: Optional thread ID for checkpointing

    Returns:
        Final MACERState with answer
    """
    initial_state = create_initial_state(query)

    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    logger.info("Starting MACER workflow", query=query[:100], thread_id=thread_id)

    final_state = await graph.ainvoke(initial_state, config=config)

    logger.info(
        "MACER workflow completed",
        confidence=final_state.get("confidence", 0),
        iterations=final_state.get("iteration", 0),
    )

    return final_state

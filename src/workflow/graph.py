"""
LangGraph Workflow for MACER Reasoning.

Orchestrates the Think-on-Graph 3.0 MACER agents:
Constructor → Retriever → Reflector ←→ (loop) → Responser

Includes intent classification to handle non-knowledge queries gracefully.
"""

import re
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, cast

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.tog.agents import (
    ConstructorAgent,
    ReflectorAgent,
    ResponserAgent,
    RetrieverAgent,
)
from src.tog.state import MACERState, SubGraph

logger = structlog.get_logger(__name__)


# =============================================================================
# Intent Classification
# =============================================================================

# Greeting patterns (Korean and English)
GREETING_PATTERNS = [
    # Korean greetings
    r"^안녕[하세요]*[?!]*$",
    r"^안뇽[?!]*$",
    r"^하이[?!]*$",
    r"^헬로[?!]*$",
    r"^반가워[요]?[?!]*$",
    r"^처음 뵙겠습니다[?!\.]*$",
    r"^좋은 ?(아침|오후|저녁)[이에요요]*[?!\.]*$",
    # English greetings
    r"^h(i|ello|ey)[?!\.]*$",
    r"^good ?(morning|afternoon|evening|night)[?!\.]*$",
    r"^greetings?[?!\.]*$",
    r"^what'?s? ?up[?!\.]*$",
    r"^how are you[?!\.]*$",
    r"^howdy[?!\.]*$",
]

# Small talk patterns
SMALL_TALK_PATTERNS = [
    # Korean
    r"^뭐해[요]?[?]*$",
    r"^뭐하고 ?있어[요]?[?]*$",
    r"^잘 ?지내[요]?[?]*$",
    r"^오늘 ?날씨[는가]?[?]*$",
    r"^심심해[요]?[?]*$",
    r"^배고파[요]?[?]*$",
    # English
    r"^how'?s? ?it ?going[?]*$",
    r"^what'?s? ?new[?]*$",
    r"^how'?s? ?your ?day[?]*$",
]

# System/capability questions
SYSTEM_PATTERNS = [
    # Korean
    r"^너는? ?누구[야니]?[?]*$",
    r"^넌? ?뭐야[?]*$",
    r"^뭘? ?할 ?수 ?있어[?]*$",
    r"^어떻게 ?사용해[?]*$",
    r"^도움말[?!\.]*$",
    # English
    r"^who ?are ?you[?]*$",
    r"^what ?are ?you[?]*$",
    r"^what ?can ?you ?do[?]*$",
    r"^help[?!\.]*$",
    r"^how ?do ?i ?use[?]*$",
]


def classify_query_intent(query: str) -> dict[str, Any]:
    """
    Classify query intent to determine appropriate handling.

    Returns:
        Dict with intent type and response info
    """
    query_lower = query.lower().strip()

    # Check for greetings
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return {
                "intent": "GREETING",
                "is_knowledge_query": False,
                "response": _get_greeting_response(query),
            }

    # Check for small talk
    for pattern in SMALL_TALK_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return {
                "intent": "SMALL_TALK",
                "is_knowledge_query": False,
                "response": _get_small_talk_response(query),
            }

    # Check for system/help questions
    for pattern in SYSTEM_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return {
                "intent": "SYSTEM",
                "is_knowledge_query": False,
                "response": _get_system_response(query),
            }

    # Default: knowledge query
    return {
        "intent": "KNOWLEDGE",
        "is_knowledge_query": True,
        "response": None,
    }


def _get_greeting_response(query: str) -> str:
    """Generate a greeting response."""
    # Detect language
    is_korean = any('\uac00' <= c <= '\ud7a3' for c in query)

    if is_korean:
        return """안녕하세요! 👋

저는 지식 그래프 기반 추론 시스템입니다.
업로드된 문서에서 추출된 지식을 바탕으로 질문에 답변해드립니다.

**사용 방법:**
1. 먼저 'Data Ingestion' 탭에서 문서를 업로드하세요
2. 그 다음 여기서 질문해주세요

**예시 질문:**
- "레드팀이 뭐야?"
- "머신러닝과 딥러닝의 차이점은?"
- "특정 개념에 대해 설명해줘"

무엇이든 물어보세요! 😊"""
    else:
        return """Hello! 👋

I'm a knowledge graph-based reasoning system.
I can answer questions based on knowledge extracted from uploaded documents.

**How to use:**
1. First, upload documents in the 'Data Ingestion' tab
2. Then ask your questions here

**Example questions:**
- "What is machine learning?"
- "Explain the difference between X and Y"
- "Tell me about a specific concept"

Feel free to ask anything! 😊"""


def _get_small_talk_response(query: str) -> str:
    """Generate a small talk response."""
    is_korean = any('\uac00' <= c <= '\ud7a3' for c in query)

    if is_korean:
        return """저는 AI 어시스턴트라서 개인적인 일상은 없지만,
당신의 질문에 답변하는 것이 제 역할이에요! 🤖

**지식 그래프 질문을 해주세요!**
업로드된 문서를 바탕으로 다양한 질문에 답변해드릴 수 있어요.

예: "~에 대해 알려줘", "~가 뭐야?", "~와 ~의 관계는?" 등"""
    else:
        return """As an AI assistant, I don't have personal experiences,
but I'm here to help answer your questions! 🤖

**Try asking a knowledge question!**
I can answer various questions based on uploaded documents.

Examples: "Tell me about...", "What is...?", "How does X relate to Y?" etc."""


def _get_system_response(query: str) -> str:
    """Generate a system/help response."""
    is_korean = any('\uac00' <= c <= '\ud7a3' for c in query)

    if is_korean:
        return """## 지식 그래프 추론 시스템 🧠

저는 Think-on-Graph (ToG) 3.0 MACER 프레임워크를 사용하는
지식 그래프 기반 추론 시스템입니다.

### 기능
- **문서 처리**: PDF, TXT, JSON, MD 등 다양한 형식 지원
- **엔티티 추출**: 문서에서 주요 개념과 관계 자동 추출
- **지식 그래프 구축**: Neo4j 기반 지식 그래프 생성
- **추론 응답**: 그래프를 탐색하여 질문에 답변

### 사용 방법
1. **문서 업로드**: 'Data Ingestion' 탭에서 문서 업로드
2. **질문하기**: 업로드된 문서 내용에 대해 질문
3. **결과 확인**: 추론 과정과 함께 답변 확인

### 현재 상태
- 지식 그래프에 문서가 업로드되어 있는지 확인하세요
- 벡터 임베딩이 있으면 의미 검색이 더 정확합니다"""
    else:
        return """## Knowledge Graph Reasoning System 🧠

I'm a knowledge graph-based reasoning system using the
Think-on-Graph (ToG) 3.0 MACER framework.

### Features
- **Document Processing**: Support for PDF, TXT, JSON, MD, etc.
- **Entity Extraction**: Automatic extraction of concepts and relations
- **Knowledge Graph**: Neo4j-based knowledge graph construction
- **Reasoning**: Graph traversal to answer questions

### How to Use
1. **Upload Documents**: Use 'Data Ingestion' tab
2. **Ask Questions**: Query about uploaded document content
3. **View Results**: See reasoning process and answers

### Current Status
- Check if documents are uploaded to the knowledge graph
- Vector embeddings enable more accurate semantic search"""


class OntologyReasoningWorkflow:
    """
    LangGraph-based MACER reasoning workflow.

    Provides a complete pipeline for knowledge graph reasoning
    with adaptive iteration and meta-cognitive control.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings | None = None,
        sufficiency_threshold: float = 0.75,
        max_iterations: int = 5,
        enable_checkpointing: bool = True,
        checkpointing_backend: Literal["memory", "postgres"] = "memory",
        postgres_uri: str | None = None,
    ) -> None:
        """
        Initialize the reasoning workflow.

        Args:
            llm: LangChain chat model
            embeddings: Embedding model for vector search
            sufficiency_threshold: Score threshold to conclude
            max_iterations: Maximum reasoning iterations
            enable_checkpointing: Whether to enable state checkpointing
            checkpointing_backend: Backend for checkpointing ("memory" or "postgres")
            postgres_uri: PostgreSQL connection URI (required for postgres backend)
        """
        self._llm = llm
        self._embeddings = embeddings
        self._sufficiency_threshold = sufficiency_threshold
        self._max_iterations = max_iterations

        # Initialize agents (pass embeddings to Constructor for vector search)
        self._constructor = ConstructorAgent(llm=llm, embeddings=embeddings)
        self._retriever = RetrieverAgent(llm=llm, embeddings=embeddings)
        self._reflector = ReflectorAgent(
            llm=llm,
            sufficiency_threshold=sufficiency_threshold,
            max_iterations=max_iterations,
        )
        self._responser = ResponserAgent(llm=llm)

        # Build workflow
        self._workflow = self._build_workflow()

        # Checkpointer setup
        self._checkpointer: BaseCheckpointSaver | None = None
        self._checkpointing_backend = checkpointing_backend
        self._postgres_uri = postgres_uri

        if enable_checkpointing:
            self._checkpointer = self._create_checkpointer(checkpointing_backend, postgres_uri)

        # Compiled app (lazy initialization)
        self._app: Any = None

    def _create_checkpointer(
        self,
        backend: Literal["memory", "postgres"],
        postgres_uri: str | None = None,
    ) -> BaseCheckpointSaver:
        """
        Create a checkpointer based on the specified backend.

        Args:
            backend: Checkpointing backend type
            postgres_uri: PostgreSQL connection URI for postgres backend

        Returns:
            Configured checkpointer instance
        """
        if backend == "postgres":
            if not postgres_uri:
                logger.warning(
                    "PostgreSQL URI not provided, falling back to memory checkpointer"
                )
                return MemorySaver()

            try:
                from langgraph.checkpoint.postgres import PostgresSaver

                logger.info(
                    "Initializing PostgreSQL checkpointer",
                    uri=postgres_uri.split("@")[-1] if "@" in postgres_uri else "***",
                )
                return PostgresSaver.from_conn_string(postgres_uri)
            except ImportError:
                logger.warning(
                    "langgraph-checkpoint-postgres not installed, "
                    "falling back to memory checkpointer. "
                    "Install with: pip install langgraph-checkpoint-postgres"
                )
                return MemorySaver()
            except Exception as e:
                logger.error(
                    "Failed to initialize PostgreSQL checkpointer",
                    error=str(e),
                )
                return MemorySaver()

        # Default: memory checkpointer
        return MemorySaver()

    def _build_workflow(self) -> StateGraph[MACERState]:
        """Build the LangGraph workflow."""

        workflow: StateGraph[MACERState] = StateGraph(MACERState)

        # =====================================================================
        # Node Definitions
        # =====================================================================

        async def constructor_node(state: MACERState) -> dict[str, Any]:
            """
            Constructor Node: Extract topic entities and build seed subgraph.
            """
            logger.info(
                "=== CONSTRUCTOR NODE START ===",
                query=state.get("original_query", "")[:50],
                iteration_in=state.get("iteration", 0),
            )
            try:
                result = await self._constructor.construct(state)
                logger.info(
                    "=== CONSTRUCTOR NODE END ===",
                    topic_entities=len(result.get("topic_entities", [])),
                    retrieved_entities=len(result.get("retrieved_entities", [])),
                    subgraph_nodes=result.get("current_subgraph", SubGraph()).node_count(),
                    iteration_out=result.get("iteration", 0),
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
            """
            iteration = state.get("iteration", 0)
            logger.info(
                "=== RETRIEVER NODE START ===",
                iteration_in=iteration,
                evidence_count_in=len(state.get("evidence", [])),
                subgraph_nodes_in=state.get("current_subgraph", SubGraph()).node_count(),
            )
            try:
                result = await self._retriever.retrieve(state)
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
            Reflector Node: Assess sufficiency and decide next action.
            """
            iteration = state.get("iteration", 0)
            logger.info(
                "=== REFLECTOR NODE START ===",
                iteration_in=iteration,
                evidence_count=len(state.get("evidence", [])),
                current_sufficiency=state.get("sufficiency_score", 0.0),
            )
            try:
                result = await self._reflector.reflect(state)
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
                    "should_terminate": True,  # Force termination on error
                }

        async def responser_node(state: MACERState) -> dict[str, Any]:
            """
            Responser Node: Synthesize evidence and generate answer.
            """
            logger.info(
                "=== RESPONSER NODE START ===",
                iteration_final=state.get("iteration", 0),
                evidence_count=len(state.get("evidence", [])),
                retrieved_entities=len(state.get("retrieved_entities", [])),
                sufficiency_score=state.get("sufficiency_score", 0.0),
            )
            try:
                result = await self._responser.respond(state)
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

        # =====================================================================
        # Add Nodes
        # =====================================================================

        workflow.add_node("constructor", constructor_node)
        workflow.add_node("retriever", retriever_node)
        workflow.add_node("reflector", reflector_node)
        workflow.add_node("responser", responser_node)

        # =====================================================================
        # Edge Definitions
        # =====================================================================

        # Entry point
        workflow.set_entry_point("constructor")

        # Linear edges
        workflow.add_edge("constructor", "retriever")
        workflow.add_edge("retriever", "reflector")

        # Conditional edge from reflector
        def should_continue(state: MACERState) -> Literal["continue", "respond"]:
            """Determine whether to continue reasoning or respond."""
            should_terminate = state.get("should_terminate", False)
            iteration = state.get("iteration", 0)
            max_iterations = state.get("max_iterations", self._max_iterations)
            sufficiency = state.get("sufficiency_score", 0.0)
            evidence_count = len(state.get("evidence", []))
            metadata = state.get("metadata", {})
            subgraph = state.get("current_subgraph", SubGraph())

            logger.info(
                "=== ROUTING DECISION ===",
                iteration=iteration,
                max_iterations=max_iterations,
                sufficiency=f"{sufficiency:.2%}",
                threshold=f"{self._sufficiency_threshold:.2%}",
                should_terminate=should_terminate,
                evidence_count=evidence_count,
                subgraph_nodes=subgraph.node_count(),
            )

            # Check for NO_DATA scenario - immediate termination
            if metadata.get("no_data", False):
                logger.info(">>> ROUTE: responser (NO_DATA - knowledge graph empty)")
                return "respond"

            # Check termination conditions
            if should_terminate:
                logger.info(">>> ROUTE: responser (should_terminate=True)")
                return "respond"

            if iteration >= max_iterations:
                logger.info(">>> ROUTE: responser (max iterations reached)", iteration=iteration)
                return "respond"

            if sufficiency >= self._sufficiency_threshold:
                logger.info(
                    ">>> ROUTE: responser (sufficiency threshold met)", sufficiency=sufficiency
                )
                return "respond"

            # If no progress after 2 iterations (no evidence, no subgraph nodes), give up
            if iteration >= 2 and evidence_count == 0 and subgraph.node_count() == 0:
                logger.info(">>> ROUTE: responser (no progress after multiple iterations)")
                return "respond"

            logger.info(
                ">>> ROUTE: retriever (continuing reasoning)",
                iteration=iteration,
                remaining=max_iterations - iteration,
            )
            return "continue"

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

    def compile(self) -> Any:
        """Compile the workflow into an executable app."""
        if self._app is None:
            self._app = self._workflow.compile(checkpointer=self._checkpointer)
        return self._app

    async def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> MACERState:
        """
        Execute the reasoning workflow.

        Includes intent classification to handle non-knowledge queries
        (greetings, small talk, help) without going through full pipeline.

        Args:
            query: Natural language question
            context: Optional additional context
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final MACERState with answer
        """
        # Check intent first
        intent_result = classify_query_intent(query)

        if not intent_result["is_knowledge_query"]:
            # Handle non-knowledge queries directly
            logger.info(
                "Non-knowledge query detected",
                query=query[:50],
                intent=intent_result["intent"],
            )

            return cast(MACERState, {
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
                "sufficiency_score": 1.0,
                "iteration": 0,
                "max_iterations": self._max_iterations,
                "should_terminate": True,
                "final_answer": intent_result["response"],
                "confidence": 1.0,
                "explanation": f"Query classified as {intent_result['intent']} - responded directly.",
                "pipeline_id": thread_id or str(uuid.uuid4())[:8],
                "errors": [],
                "metadata": {
                    **(context or {}),
                    "intent": intent_result["intent"],
                    "is_knowledge_query": False,
                },
            })

        # Proceed with full reasoning workflow for knowledge queries
        app = self.compile()

        # Initialize state
        initial_state: MACERState = {
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
            "max_iterations": self._max_iterations,
            "should_terminate": False,
            "final_answer": None,
            "confidence": 0.0,
            "explanation": "",
            "pipeline_id": thread_id or str(uuid.uuid4())[:8],
            "errors": [],
            "metadata": {
                **(context or {}),
                "intent": "KNOWLEDGE",
                "is_knowledge_query": True,
            },
        }

        # Configure for checkpointing
        config = {}
        if self._checkpointer and thread_id:
            config["configurable"] = {"thread_id": thread_id}

        logger.info("Starting MACER workflow", query=query[:100], thread_id=thread_id)

        # Execute workflow
        final_state = await app.ainvoke(initial_state, config=config)

        logger.info(
            "MACER workflow completed",
            confidence=final_state.get("confidence", 0),
            iterations=final_state.get("iteration", 0),
            answer_length=len(final_state.get("final_answer", "") or ""),
        )

        # Cast final_state to MACERState (validated by workflow execution)
        return cast(MACERState, final_state)

    async def stream(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream the reasoning workflow execution.

        Yields state updates as each node completes.

        Args:
            query: Natural language question
            context: Optional additional context
            thread_id: Optional thread ID for checkpointing

        Yields:
            Dict with node name and state update
        """
        app = self.compile()

        # Initialize state
        initial_state: MACERState = {
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
            "max_iterations": self._max_iterations,
            "should_terminate": False,
            "final_answer": None,
            "confidence": 0.0,
            "explanation": "",
            "pipeline_id": thread_id or str(uuid.uuid4())[:8],
            "errors": [],
            "metadata": context or {},
        }

        # Configure for checkpointing
        config = {}
        if self._checkpointer and thread_id:
            config["configurable"] = {"thread_id": thread_id}

        logger.info("Streaming MACER workflow", query=query[:100])

        # Stream execution
        async for event in app.astream(initial_state, config=config):
            for node_name, state_update in event.items():
                yield {
                    "node": node_name,
                    "update": state_update,
                    "iteration": state_update.get("iteration", 0),
                    "sufficiency": state_update.get("sufficiency_score", 0.0),
                }

    async def stream_events(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream detailed events during workflow execution.

        Yields more granular events including node start/end.

        Args:
            query: Natural language question
            context: Optional additional context
            thread_id: Optional thread ID

        Yields:
            Event dictionaries with type and data
        """
        app = self.compile()

        initial_state: MACERState = {
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
            "max_iterations": self._max_iterations,
            "should_terminate": False,
            "final_answer": None,
            "confidence": 0.0,
            "explanation": "",
            "pipeline_id": thread_id or str(uuid.uuid4())[:8],
            "errors": [],
            "metadata": context or {},
        }

        config = {}
        if self._checkpointer and thread_id:
            config["configurable"] = {"thread_id": thread_id}

        logger.info("Streaming MACER events", query=query[:100])

        async for event in app.astream_events(initial_state, config=config, version="v2"):
            event_type = event.get("event", "")
            event_data = event.get("data", {})

            if event_type == "on_chain_start":
                yield {
                    "type": "workflow_start",
                    "name": event.get("name", ""),
                    "run_id": event.get("run_id", ""),
                }
            elif event_type == "on_chain_end":
                yield {
                    "type": "workflow_end",
                    "name": event.get("name", ""),
                    "output": event_data.get("output", {}),
                }
            elif event_type.startswith("on_") and "node" in event_type.lower():
                yield {
                    "type": event_type,
                    "name": event.get("name", ""),
                    "data": event_data,
                }

    def get_state(self, thread_id: str) -> MACERState | None:
        """
        Get the current state for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Current state or None if not found
        """
        if not self._checkpointer:
            return None

        app = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = app.get_state(config)
            return state.values if state else None
        except Exception as e:
            logger.warning("Failed to get state", thread_id=thread_id, error=str(e))
            return None


def create_ontology_reasoning_workflow(
    llm: BaseChatModel,
    embeddings: Embeddings | None = None,
    sufficiency_threshold: float = 0.75,
    max_iterations: int = 5,
    enable_checkpointing: bool = True,
    checkpointing_backend: Literal["memory", "postgres"] = "memory",
    postgres_uri: str | None = None,
) -> OntologyReasoningWorkflow:
    """
    Factory function to create an ontology reasoning workflow.

    Args:
        llm: LangChain chat model
        embeddings: Optional embedding model
        sufficiency_threshold: Score threshold to conclude
        max_iterations: Maximum reasoning iterations
        enable_checkpointing: Whether to enable state checkpointing
        checkpointing_backend: Backend for checkpointing ("memory" or "postgres")
        postgres_uri: PostgreSQL connection URI (required for postgres backend)

    Returns:
        Configured OntologyReasoningWorkflow instance
    """
    return OntologyReasoningWorkflow(
        llm=llm,
        embeddings=embeddings,
        sufficiency_threshold=sufficiency_threshold,
        max_iterations=max_iterations,
        enable_checkpointing=enable_checkpointing,
        checkpointing_backend=checkpointing_backend,
        postgres_uri=postgres_uri,
    )


# =============================================================================
# Streaming Utilities
# =============================================================================


async def stream_reasoning(
    workflow: OntologyReasoningWorkflow,
    query: str,
    callback: Callable[[dict[str, Any]], Any] | None = None,
) -> MACERState | None:
    """
    Stream reasoning workflow with optional callback.

    Args:
        workflow: The reasoning workflow
        query: Natural language question
        callback: Optional async callback for each update

    Returns:
        Final state
    """
    final_state = None

    async for update in workflow.stream(query):
        node = update.get("node")
        iteration = update.get("iteration", 0)
        sufficiency = update.get("sufficiency", 0.0)

        logger.info(
            "Workflow update",
            node=node,
            iteration=iteration,
            sufficiency=f"{sufficiency:.0%}",
        )

        if callback:
            await callback(update)

        # Keep track of state for final return
        if node == "responser":
            final_state = update.get("update", {})

    return cast(MACERState, final_state) if final_state else None


class ReasoningStreamHandler:
    """Handler for processing streaming reasoning updates."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.current_node: str = ""
        self.final_answer: str | None = None

    async def on_update(self, update: dict[str, Any]) -> None:
        """Handle a workflow update."""
        self.updates.append(update)
        self.current_node = update.get("node", "")

        state = update.get("update", {})
        if state.get("final_answer"):
            self.final_answer = state["final_answer"]

    def get_progress(self) -> dict[str, Any]:
        """Get current progress summary."""
        return {
            "updates_received": len(self.updates),
            "current_node": self.current_node,
            "has_answer": self.final_answer is not None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def reason_over_graph(
    query: str,
    llm: BaseChatModel,
    embeddings: Embeddings | None = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    """
    High-level convenience function for graph reasoning.

    Args:
        query: Natural language question
        llm: LangChain chat model
        embeddings: Optional embedding model
        max_iterations: Maximum iterations

    Returns:
        Dict with answer, confidence, and explanation
    """
    workflow = create_ontology_reasoning_workflow(
        llm=llm,
        embeddings=embeddings,
        max_iterations=max_iterations,
    )

    state = await workflow.run(query)

    return {
        "query": query,
        "answer": state.get("final_answer", ""),
        "confidence": state.get("confidence", 0.0),
        "explanation": state.get("explanation", ""),
        "iterations": state.get("iteration", 0),
        "evidence_count": len(state.get("evidence", [])),
        "errors": state.get("errors", []),
    }

"""
Pytest Configuration and Shared Fixtures.

This module provides shared fixtures for testing the Ontology Reasoning System.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.config.settings import Settings, get_settings
from src.graph.neo4j_client import OntologyGraphClient
from src.tog.state import (
    Evidence,
    EvidenceType,
    MACERState,
    QueryEvolution,
    ReasoningAction,
    ReasoningStep,
    SubGraph,
    SubGraphEdge,
    SubGraphNode,
)


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Settings Fixtures
# =============================================================================


@pytest.fixture
def test_settings() -> Settings:
    """Provide test settings with mock values."""
    with patch.dict(
        "os.environ",
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password123",
            "LLM_PROVIDER": "openai",
            "LLM_OPENAI_API_KEY": "test-api-key",
        },
    ):
        # Clear cache and get fresh settings
        get_settings.cache_clear()
        return get_settings()


# =============================================================================
# Mock LLM Fixtures
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM that returns configurable responses."""
    llm = MagicMock(spec=BaseChatModel)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mock response"))
    return llm


@pytest.fixture
def mock_llm_json_response() -> MagicMock:
    """Create a mock LLM that returns JSON responses."""
    llm = MagicMock(spec=BaseChatModel)

    async def mock_ainvoke(*args: Any, **kwargs: Any) -> AIMessage:
        return AIMessage(
            content='{"result": "success", "entities": [], "relations": []}'
        )

    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    return llm


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create a mock embeddings model."""
    embeddings = MagicMock()
    # Return 1536-dimensional vectors (OpenAI embedding size)
    embeddings.embed_documents = MagicMock(
        return_value=[[0.1] * 1536 for _ in range(10)]
    )
    embeddings.embed_query = MagicMock(return_value=[0.1] * 1536)
    embeddings.aembed_documents = AsyncMock(
        return_value=[[0.1] * 1536 for _ in range(10)]
    )
    embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    return embeddings


# =============================================================================
# Neo4j Client Fixtures
# =============================================================================


@pytest.fixture
def mock_neo4j_client() -> MagicMock:
    """Create a mock Neo4j client."""
    client = MagicMock(spec=OntologyGraphClient)

    # Connection methods
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    # Schema methods
    client.get_schema = AsyncMock(
        return_value={
            "node_labels": ["Entity", "Chunk", "Community"],
            "relationship_types": ["RELATES_TO", "CONTAINS", "BELONGS_TO"],
            "node_properties": {
                "Entity": ["id", "name", "type", "description"],
                "Chunk": ["id", "text", "source"],
            },
            "indexes": [],
        }
    )

    # Search methods
    client.vector_search = AsyncMock(return_value=[])
    client.fulltext_search = AsyncMock(return_value=[])

    # Query methods
    client.execute_cypher = AsyncMock(return_value=[])
    client.get_neighbors = AsyncMock(return_value={"nodes": [], "edges": []})

    return client


@pytest.fixture
async def real_neo4j_client(
    test_settings: Settings,
) -> AsyncGenerator[OntologyGraphClient, None]:
    """Create a real Neo4j client for integration tests."""
    client = OntologyGraphClient(settings=test_settings.neo4j)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()


# =============================================================================
# State Fixtures
# =============================================================================


@pytest.fixture
def sample_subgraph_node() -> SubGraphNode:
    """Create a sample subgraph node."""
    return SubGraphNode(
        id="entity_1",
        name="Test Entity",
        type="CONCEPT",
        properties={"description": "A test entity"},
        relevance_score=0.9,
    )


@pytest.fixture
def sample_subgraph_edge() -> SubGraphEdge:
    """Create a sample subgraph edge."""
    return SubGraphEdge(
        source_id="entity_1",
        target_id="entity_2",
        relation_type="RELATES_TO",
        properties={"weight": 1.0},
    )


@pytest.fixture
def sample_subgraph(
    sample_subgraph_node: SubGraphNode,
    sample_subgraph_edge: SubGraphEdge,
) -> SubGraph:
    """Create a sample subgraph."""
    node2 = SubGraphNode(
        id="entity_2",
        name="Another Entity",
        type="CONCEPT",
        properties={},
        relevance_score=0.8,
    )
    return SubGraph(
        nodes=[sample_subgraph_node, node2],
        edges=[sample_subgraph_edge],
    )


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Create sample evidence items."""
    return [
        Evidence(
            id="ev_1",
            content="Direct evidence from graph",
            evidence_type=EvidenceType.DIRECT,
            source_nodes=["entity_1"],
            relevance_score=0.95,
        ),
        Evidence(
            id="ev_2",
            content="Inferred evidence",
            evidence_type=EvidenceType.INFERRED,
            source_nodes=["entity_2"],
            relevance_score=0.8,
        ),
        Evidence(
            id="ev_3",
            content="Context from chunk",
            evidence_type=EvidenceType.CONTEXTUAL,
            source_nodes=["chunk_1"],
            relevance_score=0.7,
        ),
    ]


@pytest.fixture
def sample_reasoning_step() -> ReasoningStep:
    """Create a sample reasoning step."""
    return ReasoningStep(
        step_number=1,
        action=ReasoningAction.EXPLORE,
        thought="Exploring the graph for relevant entities",
        observation="Found 5 related entities",
        query_evolution=None,
        subgraph_change="Added 3 new nodes",
        new_evidence=["ev_1", "ev_2"],
        sufficiency_delta=0.2,
    )


@pytest.fixture
def sample_macer_state(
    sample_subgraph: SubGraph,
    sample_evidence: list[Evidence],
    sample_reasoning_step: ReasoningStep,
) -> MACERState:
    """Create a sample MACER state."""
    return MACERState(
        original_query="What is the relationship between A and B?",
        current_query="What is the relationship between A and B?",
        query_history=[],
        topic_entities=["A", "B"],
        retrieved_entities=[],
        current_subgraph=sample_subgraph,
        subgraph_history=[],
        evidence=sample_evidence,
        evidence_rankings={},
        reasoning_path=[sample_reasoning_step],
        sufficiency_score=0.6,
        iteration=1,
        max_iterations=5,
        should_terminate=False,
        final_answer=None,
        confidence=0.0,
        explanation="",
        pipeline_id="test_123",
        errors=[],
        metadata={},
    )


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_text_chunk() -> str:
    """Provide a sample text chunk for extraction tests."""
    return """
    Apple Inc. is a multinational technology company headquartered in Cupertino, California.
    Tim Cook has been the CEO of Apple since 2011. The company was founded by Steve Jobs,
    Steve Wozniak, and Ronald Wayne in 1976. Apple is known for products like the iPhone,
    iPad, and MacBook.
    """


@pytest.fixture
def sample_cypher_queries() -> dict[str, str]:
    """Provide sample Cypher queries for testing."""
    return {
        "valid_match": "MATCH (n:Entity) RETURN n LIMIT 10",
        "valid_where": "MATCH (n:Entity) WHERE n.name = 'Test' RETURN n",
        "valid_relationship": "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a, r, b",
        "invalid_syntax": "MATCH (n:Entity RETURN n",
        "invalid_label": "MATCH (n:NonExistent) RETURN n",
        "dangerous_delete": "MATCH (n) DETACH DELETE n",
    }


@pytest.fixture
def sample_questions() -> list[dict[str, str]]:
    """Provide sample questions for testing."""
    return [
        {
            "question": "Who is the CEO of Apple?",
            "expected_entities": ["Apple", "CEO"],
            "complexity": "simple",
        },
        {
            "question": "What products does Apple make?",
            "expected_entities": ["Apple", "products"],
            "complexity": "simple",
        },
        {
            "question": "How is Tim Cook related to Steve Jobs through Apple?",
            "expected_entities": ["Tim Cook", "Steve Jobs", "Apple"],
            "complexity": "complex",
        },
    ]


# =============================================================================
# API Test Fixtures
# =============================================================================


@pytest.fixture
def api_client() -> Generator[Any, None, None]:
    """Create a test client for FastAPI."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_api_client() -> AsyncGenerator[Any, None]:
    """Create an async test client for FastAPI."""
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


# =============================================================================
# Utility Functions
# =============================================================================


def create_mock_chain_response(response_dict: dict[str, Any]) -> MagicMock:
    """Create a mock chain that returns a specific response."""
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=response_dict)
    return chain


def create_mock_entity(
    entity_id: str,
    name: str,
    entity_type: str = "CONCEPT",
) -> dict[str, Any]:
    """Create a mock entity dictionary."""
    return {
        "id": entity_id,
        "name": name,
        "type": entity_type,
        "description": f"Test entity: {name}",
        "aliases": [],
        "confidence": 0.9,
    }


def create_mock_relation(
    source_id: str,
    target_id: str,
    relation_type: str = "RELATES_TO",
) -> dict[str, Any]:
    """Create a mock relation dictionary."""
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": relation_type,
        "properties": {},
        "confidence": 0.85,
    }

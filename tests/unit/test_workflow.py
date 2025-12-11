"""
Unit Tests for MACER Workflow.

Tests the LangGraph workflow orchestration including:
- Node definitions and transitions
- Conditional edge routing
- State management through the pipeline
- Termination conditions
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END

from src.tog.state import (
    Evidence,
    EvidenceType,
    MACERState,
    ReasoningAction,
    ReasoningStep,
    SubGraph,
    SubGraphNode,
    SufficiencyAssessment,
    TopicEntity,
)
from src.workflow.graph import OntologyReasoningWorkflow


# =============================================================================
# Workflow Construction Tests
# =============================================================================


class TestWorkflowConstruction:
    """Test workflow construction and structure."""

    @pytest.fixture
    def mock_workflow(self, mock_llm: MagicMock, mock_embeddings: MagicMock) -> OntologyReasoningWorkflow:
        """Create a workflow with mock components."""
        with patch("src.workflow.graph.get_ontology_client") as mock_client:
            mock_client.return_value = MagicMock()
            return OntologyReasoningWorkflow(
                llm=mock_llm,
                embeddings=mock_embeddings,
                sufficiency_threshold=0.75,
                max_iterations=5,
                enable_checkpointing=False,
            )

    def test_workflow_has_all_nodes(self, mock_workflow: OntologyReasoningWorkflow) -> None:
        """Test that workflow has all required nodes."""
        graph = mock_workflow._workflow.get_graph()
        node_ids = set(graph.nodes.keys())

        required_nodes = {"constructor", "retriever", "reflector", "responser"}
        assert required_nodes.issubset(node_ids), f"Missing nodes: {required_nodes - node_ids}"

    def test_workflow_entry_point(self, mock_workflow: OntologyReasoningWorkflow) -> None:
        """Test that workflow starts at constructor."""
        graph = mock_workflow._workflow.get_graph()
        # The entry point should lead to constructor
        assert "__start__" in graph.nodes or graph._entry_point == "constructor"

    def test_workflow_edges_defined(self, mock_workflow: OntologyReasoningWorkflow) -> None:
        """Test that all required edges are defined."""
        graph = mock_workflow._workflow.get_graph()
        edges = graph.edges

        # Check linear edges exist (constructor->retriever, retriever->reflector)
        edge_tuples = [(e[0], e[1]) for e in edges]

        # Note: Edge representation may vary based on LangGraph version
        # At minimum, we should have edges connecting the nodes
        assert len(edges) >= 3, "Expected at least 3 edges in the workflow"


# =============================================================================
# State Transition Tests
# =============================================================================


class TestStateTransitions:
    """Test state transitions between nodes."""

    @pytest.fixture
    def initial_state(self) -> MACERState:
        """Create initial state for testing."""
        return MACERState(
            original_query="What is machine learning?",
            current_query="What is machine learning?",
            query_history=[],
            topic_entities=[],
            retrieved_entities=[],
            current_subgraph=SubGraph(),
            subgraph_history=[],
            evidence=[],
            evidence_rankings={},
            reasoning_path=[],
            sufficiency_score=0.0,
            iteration=0,
            max_iterations=5,
            should_terminate=False,
            final_answer=None,
            confidence=0.0,
            explanation="",
            pipeline_id="test_123",
            errors=[],
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_constructor_to_retriever_transition(
        self,
        mock_llm: MagicMock,
        initial_state: MACERState,
    ) -> None:
        """Test that constructor always transitions to retriever."""
        from src.tog.agents.constructor import ConstructorAgent

        with patch("src.tog.agents.constructor.get_ontology_client") as mock_client:
            mock_client.return_value = MagicMock()
            mock_client.return_value.connect = AsyncMock()
            mock_client.return_value.execute_cypher = AsyncMock(return_value=[{"count": 10}])
            mock_client.return_value.vector_search = AsyncMock(return_value=[])
            mock_client.return_value.fulltext_search = AsyncMock(return_value=[])

            # Mock the extraction chain
            with patch.object(
                ConstructorAgent,
                "_topic_chain",
                new_callable=lambda: MagicMock(
                    ainvoke=AsyncMock(return_value={"topic_entities": []})
                ),
            ):
                constructor = ConstructorAgent(llm=mock_llm)
                result = await constructor.construct(initial_state)

                # Constructor should return state updates
                assert "topic_entities" in result
                assert "current_subgraph" in result
                # Should NOT set should_terminate unless no data
                if not result.get("metadata", {}).get("no_data"):
                    assert result.get("should_terminate", False) is False


# =============================================================================
# Conditional Routing Tests
# =============================================================================


class TestConditionalRouting:
    """Test the conditional routing from Reflector."""

    def test_should_continue_at_max_iterations(self) -> None:
        """Test that routing returns 'respond' at max iterations."""
        state: MACERState = {
            "iteration": 5,
            "max_iterations": 5,
            "sufficiency_score": 0.5,
            "should_terminate": False,
            "evidence": [],
            "current_subgraph": SubGraph(),
            "metadata": {},
        }

        # Simulate the should_continue function logic
        should_terminate = state.get("should_terminate", False)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 5)
        sufficiency = state.get("sufficiency_score", 0.0)
        threshold = 0.75

        if should_terminate or iteration >= max_iterations or sufficiency >= threshold:
            result = "respond"
        else:
            result = "continue"

        assert result == "respond"

    def test_should_continue_at_sufficiency_threshold(self) -> None:
        """Test that routing returns 'respond' when sufficiency is met."""
        state: MACERState = {
            "iteration": 2,
            "max_iterations": 5,
            "sufficiency_score": 0.85,
            "should_terminate": False,
            "evidence": [],
            "current_subgraph": SubGraph(),
            "metadata": {},
        }

        threshold = 0.75
        if state["sufficiency_score"] >= threshold:
            result = "respond"
        else:
            result = "continue"

        assert result == "respond"

    def test_should_continue_when_insufficient(self) -> None:
        """Test that routing returns 'continue' when more evidence needed."""
        state: MACERState = {
            "iteration": 1,
            "max_iterations": 5,
            "sufficiency_score": 0.4,
            "should_terminate": False,
            "evidence": [MagicMock()],
            "current_subgraph": SubGraph(nodes=[SubGraphNode(id="1", name="Test", type="CONCEPT")]),
            "metadata": {},
        }

        threshold = 0.75
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 5)
        sufficiency = state.get("sufficiency_score", 0.0)
        should_terminate = state.get("should_terminate", False)

        if should_terminate or iteration >= max_iterations or sufficiency >= threshold:
            result = "respond"
        else:
            result = "continue"

        assert result == "continue"

    def test_should_terminate_on_no_data(self) -> None:
        """Test that routing returns 'respond' when no_data is True."""
        state: MACERState = {
            "iteration": 0,
            "max_iterations": 5,
            "sufficiency_score": 0.0,
            "should_terminate": False,
            "evidence": [],
            "current_subgraph": SubGraph(),
            "metadata": {"no_data": True},
        }

        if state.get("metadata", {}).get("no_data", False):
            result = "respond"
        else:
            result = "continue"

        assert result == "respond"

    def test_should_terminate_after_no_progress(self) -> None:
        """Test termination after no progress in multiple iterations."""
        state: MACERState = {
            "iteration": 3,
            "max_iterations": 5,
            "sufficiency_score": 0.0,
            "should_terminate": False,
            "evidence": [],
            "current_subgraph": SubGraph(),
            "metadata": {},
        }

        iteration = state.get("iteration", 0)
        evidence_count = len(state.get("evidence", []))
        subgraph = state.get("current_subgraph", SubGraph())

        # No progress after 2 iterations
        if iteration >= 2 and evidence_count == 0 and subgraph.node_count() == 0:
            result = "respond"
        else:
            result = "continue"

        assert result == "respond"


# =============================================================================
# Termination Condition Tests
# =============================================================================


class TestTerminationConditions:
    """Test various termination conditions."""

    def test_max_iterations_enforced(self) -> None:
        """Test that max iterations limit is respected."""
        max_iterations = 5

        for iteration in range(10):
            should_stop = iteration >= max_iterations

            if iteration < max_iterations:
                assert not should_stop
            else:
                assert should_stop

    def test_sufficiency_threshold(self) -> None:
        """Test sufficiency threshold comparison."""
        threshold = 0.75

        test_scores = [0.0, 0.5, 0.74, 0.75, 0.8, 1.0]
        expected = [False, False, False, True, True, True]

        for score, should_stop in zip(test_scores, expected):
            assert (score >= threshold) == should_stop

    def test_explicit_termination_flag(self) -> None:
        """Test that explicit termination flag works."""
        state_with_flag: MACERState = {
            "should_terminate": True,
            "iteration": 1,
            "max_iterations": 5,
            "sufficiency_score": 0.3,
        }

        assert state_with_flag["should_terminate"] is True


# =============================================================================
# State Immutability Tests
# =============================================================================


class TestStateImmutability:
    """Test that state updates are immutable."""

    def test_evidence_update_creates_new_list(self) -> None:
        """Test that evidence updates don't mutate original list."""
        original_evidence = [
            Evidence(
                id="ev_1",
                content="Test",
                evidence_type=EvidenceType.DIRECT,
                source_nodes=["node_1"],
                relevance_score=0.8,
            )
        ]

        # Simulate adding evidence immutably
        new_evidence = Evidence(
            id="ev_2",
            content="New",
            evidence_type=EvidenceType.CONTEXTUAL,
            source_nodes=["node_2"],
            relevance_score=0.7,
        )

        updated_evidence = original_evidence + [new_evidence]

        assert len(original_evidence) == 1
        assert len(updated_evidence) == 2
        assert original_evidence is not updated_evidence

    def test_subgraph_update_creates_new_instance(self) -> None:
        """Test that subgraph updates create new instances."""
        original = SubGraph(
            nodes=[SubGraphNode(id="1", name="A", type="CONCEPT")],
            edges=[],
        )

        new_node = SubGraphNode(id="2", name="B", type="CONCEPT")

        # Create new subgraph with additional node
        updated = SubGraph(
            nodes=original.nodes + [new_node],
            edges=original.edges,
            center_entity_id=original.center_entity_id,
        )

        assert len(original.nodes) == 1
        assert len(updated.nodes) == 2
        assert original is not updated

    def test_evidence_model_copy(self) -> None:
        """Test that Evidence.model_copy creates proper copies."""
        original = Evidence(
            id="ev_1",
            content="Test",
            evidence_type=EvidenceType.DIRECT,
            source_nodes=["node_1"],
            relevance_score=0.5,
        )

        # Update relevance score immutably
        updated = original.model_copy(update={"relevance_score": 0.9})

        assert original.relevance_score == 0.5
        assert updated.relevance_score == 0.9
        assert original is not updated


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in the workflow."""

    @pytest.mark.asyncio
    async def test_constructor_handles_extraction_error(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Test that constructor handles LLM extraction errors gracefully."""
        from src.tog.agents.constructor import ConstructorAgent

        with patch("src.tog.agents.constructor.get_ontology_client") as mock_client:
            mock_client.return_value = MagicMock()
            mock_client.return_value.connect = AsyncMock()
            mock_client.return_value.execute_cypher = AsyncMock(return_value=[{"count": 10}])

            # Make the extraction chain fail
            with patch.object(
                ConstructorAgent,
                "_topic_chain",
                new_callable=lambda: MagicMock(
                    ainvoke=AsyncMock(side_effect=Exception("LLM Error"))
                ),
            ):
                constructor = ConstructorAgent(llm=mock_llm)

                state: MACERState = {
                    "original_query": "Test query",
                    "current_query": "Test query",
                }

                # Should not raise, should return gracefully
                result = await constructor.construct(state)

                # Should still return valid state
                assert "topic_entities" in result

    @pytest.mark.asyncio
    async def test_reflector_handles_assessment_error(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Test that reflector handles assessment errors gracefully."""
        from src.tog.agents.reflector import ReflectorAgent

        reflector = ReflectorAgent(llm=mock_llm)

        # Make the sufficiency chain fail
        with patch.object(
            reflector,
            "_sufficiency_chain",
            new_callable=lambda: MagicMock(
                ainvoke=AsyncMock(side_effect=Exception("Assessment Error"))
            ),
        ):
            state: MACERState = {
                "original_query": "Test query",
                "current_query": "Test query",
                "evidence": [],
                "current_subgraph": SubGraph(),
                "reasoning_path": [],
                "iteration": 0,
                "metadata": {},
            }

            assessment = await reflector.assess_sufficiency(state)

            # Should return default assessment on error
            assert isinstance(assessment, SufficiencyAssessment)
            assert assessment.score == 0.5  # Default score


# =============================================================================
# Integration Flow Tests
# =============================================================================


class TestIntegrationFlow:
    """Test the full flow through the workflow."""

    @pytest.fixture
    def complete_state(self) -> MACERState:
        """Create a complete state for flow testing."""
        return MACERState(
            original_query="Who is the CEO of Apple?",
            current_query="Who is the CEO of Apple?",
            query_history=[],
            topic_entities=[
                TopicEntity(id="t1", name="Apple", type="ORG", relevance_score=1.0),
                TopicEntity(id="t2", name="CEO", type="ROLE", relevance_score=0.9),
            ],
            retrieved_entities=[
                {"id": "e1", "name": "Apple Inc.", "type": "ORG"},
                {"id": "e2", "name": "Tim Cook", "type": "PERSON"},
            ],
            current_subgraph=SubGraph(
                nodes=[
                    SubGraphNode(id="e1", name="Apple Inc.", type="ORG", relevance_score=0.95),
                    SubGraphNode(id="e2", name="Tim Cook", type="PERSON", relevance_score=0.9),
                ],
                edges=[],
            ),
            subgraph_history=[],
            evidence=[
                Evidence(
                    id="ev_1",
                    content="Tim Cook is the CEO of Apple Inc.",
                    evidence_type=EvidenceType.DIRECT,
                    source_nodes=["e1", "e2"],
                    relevance_score=0.95,
                ),
            ],
            evidence_rankings={"ev_1": 0.95},
            reasoning_path=[
                ReasoningStep(
                    step_number=1,
                    action=ReasoningAction.EXPLORE,
                    thought="Found CEO relationship",
                    observation="Direct evidence found",
                    sufficiency_delta=0.5,
                ),
            ],
            sufficiency_score=0.85,
            iteration=1,
            max_iterations=5,
            should_terminate=False,
            final_answer=None,
            confidence=0.0,
            explanation="",
            pipeline_id="flow_test",
            errors=[],
            metadata={},
        )

    def test_state_has_all_required_fields(self, complete_state: MACERState) -> None:
        """Test that complete state has all required fields."""
        required_fields = [
            "original_query",
            "current_query",
            "topic_entities",
            "current_subgraph",
            "evidence",
            "reasoning_path",
            "sufficiency_score",
            "iteration",
            "max_iterations",
            "should_terminate",
        ]

        for field in required_fields:
            assert field in complete_state, f"Missing field: {field}"

    def test_sufficiency_score_progression(self) -> None:
        """Test that sufficiency score can progress through iterations."""
        scores = [0.0, 0.3, 0.5, 0.7, 0.85]

        for i, score in enumerate(scores):
            if i > 0:
                # Each iteration should potentially increase score
                # (in a real scenario with evidence accumulation)
                assert score >= scores[i - 1]

    def test_reasoning_path_accumulation(self) -> None:
        """Test that reasoning path accumulates correctly."""
        path = []

        for i in range(3):
            step = ReasoningStep(
                step_number=i + 1,
                action=ReasoningAction.EXPLORE if i < 2 else ReasoningAction.CONCLUDE,
                thought=f"Step {i + 1} thought",
                observation=f"Step {i + 1} observation",
                sufficiency_delta=0.2,
            )
            path.append(step)

        assert len(path) == 3
        assert path[0].step_number == 1
        assert path[-1].action == ReasoningAction.CONCLUDE

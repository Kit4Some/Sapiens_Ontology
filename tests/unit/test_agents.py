"""
Unit Tests for MACER Agents.

Tests the Constructor, Retriever, Reflector, and Responser agents.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tog.agents import (
    ConstructorAgent,
    ReflectorAgent,
    ResponserAgent,
    RetrieverAgent,
)
from src.tog.state import (
    Evidence,
    EvidenceType,
    MACERState,
    ReasoningAction,
    ReasoningStep,
    SubGraph,
    SubGraphEdge,
    SubGraphNode,
    SufficiencyAssessment,
)


# =============================================================================
# Constructor Agent Tests
# =============================================================================


class TestConstructorAgent:
    """Test cases for ConstructorAgent."""

    @pytest.fixture
    def constructor_agent(self, mock_llm: MagicMock) -> ConstructorAgent:
        """Create a ConstructorAgent with mock LLM."""
        return ConstructorAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_extract_topic_entities(
        self,
        constructor_agent: ConstructorAgent,
    ) -> None:
        """Test topic entity extraction from query."""
        # Mock the chain response
        with patch.object(
            constructor_agent,
            "_topic_chain",
            new_callable=lambda: MagicMock(
                ainvoke=AsyncMock(
                    return_value={
                        "entities": ["Apple", "CEO", "Tim Cook"],
                        "entity_types": {
                            "Apple": "ORGANIZATION",
                            "CEO": "ROLE",
                            "Tim Cook": "PERSON",
                        },
                    }
                )
            ),
        ):
            entities = await constructor_agent.extract_topic_entities(
                "Who is the CEO of Apple?"
            )

            assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_construct_creates_seed_subgraph(
        self,
        constructor_agent: ConstructorAgent,
        sample_macer_state: MACERState,
        mock_neo4j_client: MagicMock,
    ) -> None:
        """Test that construct() creates a seed subgraph."""
        # Mock the neo4j client
        with patch.object(
            constructor_agent, "_neo4j_client", mock_neo4j_client
        ):
            mock_neo4j_client.vector_search.return_value = [
                MagicMock(
                    node_id="entity_1",
                    name="Test Entity",
                    node_label="Entity",
                    score=0.9,
                )
            ]
            mock_neo4j_client.get_neighbors.return_value = {
                "nodes": [],
                "edges": [],
            }

            # Mock topic extraction
            with patch.object(
                constructor_agent,
                "extract_topic_entities",
                new_callable=lambda: AsyncMock(return_value=["Test"]),
            ):
                result = await constructor_agent.construct(sample_macer_state)

                assert "topic_entities" in result
                assert "current_subgraph" in result

    @pytest.mark.asyncio
    async def test_construct_handles_no_entities(
        self,
        constructor_agent: ConstructorAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test construct() handles case with no matching entities."""
        with patch.object(
            constructor_agent,
            "extract_topic_entities",
            new_callable=lambda: AsyncMock(return_value=[]),
        ):
            result = await constructor_agent.construct(sample_macer_state)

            assert "topic_entities" in result
            assert result["topic_entities"] == []


# =============================================================================
# Retriever Agent Tests
# =============================================================================


class TestRetrieverAgent:
    """Test cases for RetrieverAgent."""

    @pytest.fixture
    def retriever_agent(
        self,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> RetrieverAgent:
        """Create a RetrieverAgent with mocks."""
        return RetrieverAgent(llm=mock_llm, embeddings=mock_embeddings)

    @pytest.mark.asyncio
    async def test_vector_search(
        self,
        retriever_agent: RetrieverAgent,
        mock_neo4j_client: MagicMock,
    ) -> None:
        """Test vector similarity search."""
        with patch.object(
            retriever_agent, "_neo4j_client", mock_neo4j_client
        ):
            mock_neo4j_client.vector_search.return_value = [
                {"id": "e1", "name": "Entity 1", "score": 0.95},
                {"id": "e2", "name": "Entity 2", "score": 0.85},
            ]

            results = await retriever_agent.vector_search(
                query="Test query",
                top_k=5,
            )

            assert len(results) == 2
            mock_neo4j_client.vector_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_collects_evidence(
        self,
        retriever_agent: RetrieverAgent,
        sample_macer_state: MACERState,
        mock_neo4j_client: MagicMock,
    ) -> None:
        """Test that retrieve() collects evidence."""
        with patch.object(
            retriever_agent, "_neo4j_client", mock_neo4j_client
        ):
            # Mock strategy chain
            with patch.object(
                retriever_agent,
                "_strategy_chain",
                new_callable=lambda: MagicMock(
                    ainvoke=AsyncMock(
                        return_value={
                            "primary_strategy": "hybrid",
                            "search_depth": 2,
                        }
                    )
                ),
            ):
                mock_neo4j_client.vector_search.return_value = []
                mock_neo4j_client.fulltext_search.return_value = []

                result = await retriever_agent.retrieve(sample_macer_state)

                assert "evidence" in result

    @pytest.mark.asyncio
    async def test_rank_evidence(
        self,
        retriever_agent: RetrieverAgent,
        sample_evidence: list[Evidence],
    ) -> None:
        """Test evidence ranking."""
        ranked = retriever_agent.rank_evidence(
            evidence=sample_evidence,
            query="test query",
        )

        assert len(ranked) == len(sample_evidence)
        # Should be sorted by relevance
        scores = [e.relevance_score for e in ranked]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Reflector Agent Tests
# =============================================================================


class TestReflectorAgent:
    """Test cases for ReflectorAgent."""

    @pytest.fixture
    def reflector_agent(self, mock_llm: MagicMock) -> ReflectorAgent:
        """Create a ReflectorAgent with mock LLM."""
        return ReflectorAgent(
            llm=mock_llm,
            sufficiency_threshold=0.75,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    async def test_assess_sufficiency(
        self,
        reflector_agent: ReflectorAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test sufficiency assessment."""
        with patch.object(
            reflector_agent,
            "_sufficiency_chain",
            new_callable=lambda: MagicMock(
                ainvoke=AsyncMock(
                    return_value={
                        "sufficiency_score": 0.7,
                        "has_enough_evidence": False,
                        "missing_aspects": ["More context needed"],
                        "completeness_score": 0.6,
                        "reliability_score": 0.8,
                        "consistency_score": 0.7,
                        "recommendation": "EXPLORE",
                        "reasoning": "Need more evidence",
                    }
                )
            ),
        ):
            assessment = await reflector_agent.assess_sufficiency(
                sample_macer_state
            )

            assert isinstance(assessment, SufficiencyAssessment)
            assert 0 <= assessment.score <= 1
            assert assessment.recommendation in ReasoningAction

    @pytest.mark.asyncio
    async def test_evolve_query(
        self,
        reflector_agent: ReflectorAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test query evolution."""
        with patch.object(
            reflector_agent,
            "_query_evolution_chain",
            new_callable=lambda: MagicMock(
                ainvoke=AsyncMock(
                    return_value={
                        "evolved_query": "What is the specific relationship?",
                        "reasoning": "Narrowing focus",
                        "evolution_type": "REFINE",
                        "sub_questions": [],
                    }
                )
            ),
        ):
            evolution = await reflector_agent.evolve_query(sample_macer_state)

            if evolution:
                assert evolution.refined != sample_macer_state["current_query"]

    @pytest.mark.asyncio
    async def test_should_continue_at_max_iterations(
        self,
        reflector_agent: ReflectorAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test that should_continue returns False at max iterations."""
        state = {**sample_macer_state, "iteration": 5, "max_iterations": 5}

        should_continue, action = await reflector_agent.should_continue(state)

        assert not should_continue
        assert action == ReasoningAction.CONCLUDE

    @pytest.mark.asyncio
    async def test_should_continue_at_threshold(
        self,
        reflector_agent: ReflectorAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test that should_continue returns False at sufficiency threshold."""
        state = {**sample_macer_state, "sufficiency_score": 0.85}

        should_continue, action = await reflector_agent.should_continue(state)

        assert not should_continue
        assert action == ReasoningAction.CONCLUDE

    @pytest.mark.asyncio
    async def test_reflect_updates_state(
        self,
        reflector_agent: ReflectorAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test that reflect() updates state correctly."""
        with patch.object(
            reflector_agent,
            "assess_sufficiency",
            new_callable=lambda: AsyncMock(
                return_value=SufficiencyAssessment(
                    score=0.6,
                    has_enough_evidence=False,
                    missing_aspects=["More data"],
                    recommendation=ReasoningAction.EXPLORE,
                    reasoning="Continuing exploration",
                )
            ),
        ):
            with patch.object(
                reflector_agent,
                "should_continue",
                new_callable=lambda: AsyncMock(
                    return_value=(True, ReasoningAction.EXPLORE)
                ),
            ):
                result = await reflector_agent.reflect(sample_macer_state)

                assert "sufficiency_score" in result
                assert "reasoning_path" in result
                assert "iteration" in result

    def test_create_reasoning_step(
        self,
        reflector_agent: ReflectorAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test reasoning step creation."""
        step = reflector_agent.create_reasoning_step(
            state=sample_macer_state,
            action=ReasoningAction.EXPLORE,
            thought="Exploring graph",
            observation="Found new entities",
        )

        assert isinstance(step, ReasoningStep)
        assert step.action == ReasoningAction.EXPLORE
        assert step.thought == "Exploring graph"


# =============================================================================
# Responser Agent Tests
# =============================================================================


class TestResponserAgent:
    """Test cases for ResponserAgent."""

    @pytest.fixture
    def responser_agent(self, mock_llm: MagicMock) -> ResponserAgent:
        """Create a ResponserAgent with mock LLM."""
        return ResponserAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_synthesize_evidence(
        self,
        responser_agent: ResponserAgent,
        sample_evidence: list[Evidence],
        sample_reasoning_step: ReasoningStep,
    ) -> None:
        """Test evidence synthesis."""
        with patch.object(
            responser_agent,
            "_synthesis_chain",
            new_callable=lambda: MagicMock(
                ainvoke=AsyncMock(
                    return_value={
                        "synthesized_facts": ["Fact 1", "Fact 2"],
                        "inferences": ["Inference 1"],
                        "contradictions": [],
                        "key_evidence_used": ["ev_1"],
                        "answer_confidence": 0.8,
                    }
                )
            ),
        ):
            result = await responser_agent.synthesize_evidence(
                question="What is the answer?",
                evidence=sample_evidence,
                reasoning_path=[sample_reasoning_step],
            )

            assert "synthesized_facts" in result
            assert "inferences" in result

    @pytest.mark.asyncio
    async def test_generate_answer(
        self,
        responser_agent: ResponserAgent,
    ) -> None:
        """Test answer generation."""
        synthesis = {
            "synthesized_facts": ["Fact 1"],
            "inferences": [],
            "contradictions": [],
            "answer_confidence": 0.8,
        }

        with patch.object(
            responser_agent,
            "_answer_chain",
            new_callable=lambda: MagicMock(
                ainvoke=AsyncMock(
                    return_value={
                        "answer": "The answer is X.",
                        "confidence": 0.85,
                        "answer_type": "DEFINITIVE",
                        "supporting_evidence": [],
                        "caveats": [],
                        "explanation": "Based on the evidence...",
                    }
                )
            ),
        ):
            result = await responser_agent.generate_answer(
                question="What is X?",
                synthesis=synthesis,
            )

            assert "answer" in result
            assert "confidence" in result
            assert 0 <= result["confidence"] <= 1

    def test_calculate_final_confidence(
        self,
        responser_agent: ResponserAgent,
        sample_evidence: list[Evidence],
        sample_reasoning_step: ReasoningStep,
    ) -> None:
        """Test final confidence calculation."""
        confidence = responser_agent.calculate_final_confidence(
            synthesis_confidence=0.8,
            answer_confidence=0.75,
            evidence=sample_evidence,
            reasoning_path=[sample_reasoning_step],
        )

        assert 0 <= confidence <= 1

    def test_select_key_evidence(
        self,
        responser_agent: ResponserAgent,
        sample_evidence: list[Evidence],
    ) -> None:
        """Test key evidence selection."""
        selected = responser_agent.select_key_evidence(
            evidence=sample_evidence,
            max_items=2,
        )

        assert len(selected) <= 2
        # Should prefer diverse types and high relevance
        assert all(isinstance(e, Evidence) for e in selected)

    @pytest.mark.asyncio
    async def test_respond_full_pipeline(
        self,
        responser_agent: ResponserAgent,
        sample_macer_state: MACERState,
    ) -> None:
        """Test full respond pipeline."""
        with patch.object(
            responser_agent,
            "synthesize_evidence",
            new_callable=lambda: AsyncMock(
                return_value={
                    "synthesized_facts": ["Fact"],
                    "inferences": [],
                    "contradictions": [],
                    "answer_confidence": 0.8,
                }
            ),
        ):
            with patch.object(
                responser_agent,
                "generate_answer",
                new_callable=lambda: AsyncMock(
                    return_value={
                        "answer": "The answer",
                        "confidence": 0.8,
                        "answer_type": "DEFINITIVE",
                        "caveats": [],
                        "explanation": "",
                    }
                ),
            ):
                with patch.object(
                    responser_agent,
                    "explain_reasoning",
                    new_callable=lambda: AsyncMock(return_value="Explanation"),
                ):
                    result = await responser_agent.respond(sample_macer_state)

                    assert "final_answer" in result
                    assert "confidence" in result
                    assert "explanation" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentIntegration:
    """Integration tests for agent interactions."""

    @pytest.mark.asyncio
    async def test_constructor_retriever_flow(
        self,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
        mock_neo4j_client: MagicMock,
        sample_macer_state: MACERState,
    ) -> None:
        """Test flow from Constructor to Retriever."""
        constructor = ConstructorAgent(llm=mock_llm)
        retriever = RetrieverAgent(llm=mock_llm, embeddings=mock_embeddings)

        # This would test the actual flow in a real integration test
        # For unit tests, we verify the state shape compatibility
        assert "current_subgraph" in sample_macer_state
        assert "evidence" in sample_macer_state

    @pytest.mark.asyncio
    async def test_reflector_responser_flow(
        self,
        mock_llm: MagicMock,
        sample_macer_state: MACERState,
    ) -> None:
        """Test flow from Reflector to Responser."""
        reflector = ReflectorAgent(llm=mock_llm)
        responser = ResponserAgent(llm=mock_llm)

        # Verify state compatibility
        assert "sufficiency_score" in sample_macer_state
        assert "reasoning_path" in sample_macer_state

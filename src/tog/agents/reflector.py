"""
Reflector Agent for MACER (Core Component).

Responsible for:
1. Sufficiency Score evaluation (Meta-cognitive)
2. Query Evolution: decomposition and refinement
3. SubGraph Evolution: node/edge expansion and pruning
4. Iteration control and termination decisions
5. Multi-hop path completeness assessment
6. Reasoning chain tracking and validation
7. 4-Dimension Evaluation (Completeness, Coverage, Consistency, Convergence)
"""

from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

from src.tog.prompts import (
    FOUR_DIMENSION_REFLECTION_PROMPT,
    ITERATION_CONTROL_PROMPT,
    QUERY_EVOLUTION_PROMPT,
    SUBGRAPH_EVOLUTION_PROMPT,
    SUFFICIENCY_ASSESSMENT_PROMPT,
    format_evidence_for_prompt,
    format_reasoning_path_for_prompt,
    format_subgraph_for_prompt,
)
from src.tog.state import (
    Evidence,
    EvidenceType,
    MACERState,
    QueryEvolution,
    QuestionType,
    ReasoningAction,
    ReasoningStep,
    SubGraph,
    SufficiencyAssessment,
    calculate_path_completeness,
)

logger = structlog.get_logger(__name__)


class ReflectorAgent:
    """
    Reflector Agent - The meta-cognitive core of MACER.

    Performs adaptive reasoning by:
    - Assessing evidence sufficiency
    - Evolving queries based on what's missing
    - Guiding subgraph expansion/contraction
    - Deciding when to continue or conclude
    """

    def __init__(
        self,
        llm: BaseChatModel,
        sufficiency_threshold: float = 0.75,
        max_iterations: int = 5,
        enable_query_evolution: bool = True,
        enable_subgraph_evolution: bool = True,
    ) -> None:
        """
        Initialize the Reflector agent.

        Args:
            llm: LangChain chat model
            sufficiency_threshold: Score threshold to conclude
            max_iterations: Maximum reasoning iterations
            enable_query_evolution: Whether to refine queries
            enable_subgraph_evolution: Whether to evolve subgraph
        """
        self._llm = llm
        self._sufficiency_threshold = sufficiency_threshold
        self._max_iterations = max_iterations
        self._enable_query_evolution = enable_query_evolution
        self._enable_subgraph_evolution = enable_subgraph_evolution

        # Build chains
        self._parser = JsonOutputParser()
        self._sufficiency_chain = SUFFICIENCY_ASSESSMENT_PROMPT | self._llm | self._parser
        self._query_evolution_chain = QUERY_EVOLUTION_PROMPT | self._llm | self._parser
        self._subgraph_evolution_chain = SUBGRAPH_EVOLUTION_PROMPT | self._llm | self._parser
        self._iteration_chain = ITERATION_CONTROL_PROMPT | self._llm | self._parser
        self._four_dimension_chain = FOUR_DIMENSION_REFLECTION_PROMPT | self._llm | self._parser

        # Track previous sufficiency scores for early stop detection
        self._previous_scores: list[float] = []

    async def assess_sufficiency(self, state: MACERState) -> SufficiencyAssessment:
        """
        Assess whether current evidence is sufficient to answer the question.

        This is the core meta-cognitive function, enhanced with multi-hop awareness.

        Args:
            state: Current MACER state

        Returns:
            SufficiencyAssessment with detailed evaluation
        """
        question = state.get("original_query", "")
        evidence = state.get("evidence", [])
        subgraph = state.get("current_subgraph", SubGraph())
        reasoning_path = state.get("reasoning_path", [])
        question_type = state.get("question_type", QuestionType.FACTOID.value)
        sub_questions = state.get("sub_questions", [])
        reasoning_chains = state.get("reasoning_chains", [])

        # Check if this is a multi-hop question
        is_multihop = question_type in [
            QuestionType.MULTIHOP.value,
            QuestionType.BRIDGE.value,
            QuestionType.COMPARISON.value,
        ]

        # Calculate path completeness for multi-hop questions
        path_completeness = calculate_path_completeness(evidence, subgraph, question_type)

        # Count path-based evidence
        path_evidence_count = sum(1 for e in evidence if e.evidence_type == EvidenceType.PATH)

        # Format inputs for prompt with multi-hop context
        evidence_str = format_evidence_for_prompt(
            [
                {
                    "evidence_type": e.evidence_type.value,
                    "content": e.content,
                    "relevance_score": e.relevance_score,
                    "hop_index": e.hop_index if hasattr(e, "hop_index") else 0,
                    "path_length": e.path_length if hasattr(e, "path_length") else 0,
                }
                for e in evidence
            ]
        )

        key_entities = [
            n.name
            for n in sorted(subgraph.nodes, key=lambda x: x.relevance_score, reverse=True)[:5]
        ]

        # Get bridge entities for multi-hop questions
        bridge_entities = subgraph.get_bridge_entities() if is_multihop else []
        bridge_names = [b.name for b in bridge_entities[:3]]

        reasoning_str = format_reasoning_path_for_prompt(
            [
                {"step_number": s.step_number, "action": s.action.value, "thought": s.thought}
                for s in reasoning_path
            ]
        )

        # Add multi-hop context to prompt
        multihop_context = ""
        if is_multihop:
            multihop_context = f"""
Multi-hop Question Analysis:
- Question type: {question_type}
- Sub-questions: {len(sub_questions)} ({', '.join(sub_questions[:2][:50])}...)
- Path evidence count: {path_evidence_count}
- Reasoning chains: {len(reasoning_chains)}
- Path completeness: {path_completeness:.2f}
- Bridge entities: {', '.join(bridge_names) if bridge_names else 'None identified'}
"""

        try:
            result = await self._sufficiency_chain.ainvoke(
                {
                    "question": question,
                    "evidence": evidence_str,
                    "node_count": subgraph.node_count(),
                    "edge_count": subgraph.edge_count(),
                    "key_entities": ", ".join(key_entities),
                    "reasoning_path": reasoning_str + multihop_context,
                }
            )

            # Parse recommendation
            recommendation_str = result.get("recommendation", "EXPLORE")
            try:
                recommendation = ReasoningAction(recommendation_str.lower())
            except ValueError:
                recommendation = ReasoningAction.EXPLORE

            # Calculate base sufficiency score
            base_score = float(result.get("sufficiency_score", 0.5))

            # Adjust score based on path completeness for multi-hop questions
            if is_multihop:
                # Path completeness has significant impact on multi-hop sufficiency
                path_weight = 0.3
                adjusted_score = (base_score * (1 - path_weight)) + (path_completeness * path_weight)

                # Penalize if we have no path evidence but it's a multi-hop question
                if path_evidence_count == 0 and len(evidence) > 0:
                    adjusted_score *= 0.8  # 20% penalty

                # Penalize if no bridge entities found for bridge questions
                if question_type == QuestionType.BRIDGE.value and not bridge_entities:
                    adjusted_score *= 0.85  # 15% penalty

                base_score = adjusted_score

            # Check if we have enough evidence
            has_enough = result.get("has_enough_evidence", False)

            # For multi-hop questions, require path evidence
            if is_multihop and path_evidence_count == 0:
                has_enough = False

            # Update missing aspects for multi-hop
            missing_aspects = result.get("missing_aspects", [])
            if is_multihop:
                if path_evidence_count == 0:
                    missing_aspects.insert(0, "No reasoning paths found connecting topic entities")
                if not bridge_entities and question_type == QuestionType.BRIDGE.value:
                    missing_aspects.insert(0, "Bridge entities not identified")
                if len(reasoning_chains) == 0 and len(sub_questions) > 0:
                    missing_aspects.insert(0, "Sub-questions not resolved into reasoning chains")

            assessment = SufficiencyAssessment(
                score=min(1.0, max(0.0, base_score)),
                has_enough_evidence=has_enough,
                missing_aspects=missing_aspects,
                confidence_factors={
                    "completeness": float(result.get("completeness_score", 0.5)),
                    "reliability": float(result.get("reliability_score", 0.5)),
                    "consistency": float(result.get("consistency_score", 0.5)),
                    "path_completeness": path_completeness,
                },
                recommendation=recommendation,
                reasoning=result.get("reasoning", ""),
            )

            logger.info(
                "Sufficiency assessed",
                score=assessment.score,
                has_enough=assessment.has_enough_evidence,
                recommendation=assessment.recommendation.value,
                is_multihop=is_multihop,
                path_completeness=path_completeness,
                path_evidence=path_evidence_count,
            )

            return assessment

        except Exception as e:
            logger.error("Sufficiency assessment failed", error=str(e))
            return SufficiencyAssessment(
                score=0.5,
                has_enough_evidence=False,
                missing_aspects=["Unable to assess"],
                recommendation=ReasoningAction.EXPLORE,
                reasoning=f"Assessment failed: {str(e)}",
            )

    async def evaluate_four_dimensions(
        self,
        state: MACERState,
    ) -> dict[str, Any]:
        """
        Evaluate reasoning quality using 4 dimensions.

        Dimensions:
        1. COMPLETENESS: How much of the reasoning chain is completed
        2. COVERAGE: How well evidence covers the question requirements
        3. CONSISTENCY: Internal consistency of reasoning (no contradictions)
        4. CONVERGENCE: How confidently we can determine the final answer

        Args:
            state: Current MACER state

        Returns:
            4-dimension evaluation with action recommendation
        """
        question = state.get("original_query", "")
        evidence = state.get("evidence", [])
        sub_questions = state.get("sub_questions", [])
        reasoning_path = state.get("reasoning_path", [])
        iteration = state.get("iteration", 0)

        # Format evidence pool
        evidence_pool_str = self._format_evidence_pool(evidence)

        # Format step results (from reasoning path)
        step_results_str = format_reasoning_path_for_prompt([
            {"step_number": s.step_number, "action": s.action.value, "thought": s.thought}
            for s in reasoning_path
        ])

        try:
            result = await self._four_dimension_chain.ainvoke({
                "question": question,
                "sub_questions": str(sub_questions) if sub_questions else "None (single-hop question)",
                "step_results": step_results_str or "No steps yet",
                "evidence_pool": evidence_pool_str,
                "iteration": iteration,
                "max_iterations": self._max_iterations,
            })

            # Extract dimension scores
            dimension_scores = result.get("dimension_scores", {})
            completeness = float(dimension_scores.get("completeness", 0.5))
            coverage = float(dimension_scores.get("coverage", 0.5))
            consistency = float(dimension_scores.get("consistency", 0.8))
            convergence = float(dimension_scores.get("convergence", 0.5))

            # Calculate overall sufficiency (weighted average)
            overall_sufficiency = (
                completeness * 0.25 +
                coverage * 0.25 +
                consistency * 0.25 +
                convergence * 0.25
            )

            # Determine action based on decision matrix
            action = result.get("action", "CONTINUE_CHAIN")
            action_details = result.get("action_details", {})

            # Check for early stop
            early_stop = result.get("early_stop_check", {})
            should_early_stop = early_stop.get("should_stop", False)

            # Track scores for early stop detection
            self._previous_scores.append(overall_sufficiency)
            if len(self._previous_scores) > 3:
                self._previous_scores.pop(0)

            # Check if improvement is stagnating (2 consecutive low improvements)
            if len(self._previous_scores) >= 3:
                recent_improvement = self._previous_scores[-1] - self._previous_scores[-2]
                prev_improvement = self._previous_scores[-2] - self._previous_scores[-3]
                if recent_improvement < 0.05 and prev_improvement < 0.05:
                    should_early_stop = True

            logger.info(
                "4-dimension evaluation completed",
                completeness=completeness,
                coverage=coverage,
                consistency=consistency,
                convergence=convergence,
                overall_sufficiency=overall_sufficiency,
                action=action,
                early_stop=should_early_stop,
            )

            return {
                "dimension_scores": {
                    "completeness": completeness,
                    "coverage": coverage,
                    "consistency": consistency,
                    "convergence": convergence,
                },
                "dimension_details": result.get("dimension_details", {}),
                "overall_sufficiency": overall_sufficiency,
                "action": action,
                "action_details": action_details,
                "early_stop": should_early_stop,
                "early_stop_reason": early_stop.get("reason", ""),
            }

        except Exception as e:
            logger.error("4-dimension evaluation failed", error=str(e))
            # Fallback to simple heuristic evaluation
            return self._fallback_four_dimension_evaluation(evidence, sub_questions, reasoning_path)

    def _format_evidence_pool(self, evidence: list[Evidence]) -> str:
        """Format evidence pool with confidence types for 4-dimension evaluation."""
        if not evidence:
            return "No evidence collected."

        lines = []
        for i, ev in enumerate(evidence[:20], 1):
            ev_type = ev.evidence_type.value
            conf_type = "UNKNOWN"
            if hasattr(ev, 'confidence_type'):
                conf_type = ev.confidence_type.value

            lines.append(
                f"{i}. [{ev_type}][{conf_type}] (score: {ev.relevance_score:.2f}) "
                f"{ev.content[:200]}"
            )

        return "\n".join(lines)

    def _fallback_four_dimension_evaluation(
        self,
        evidence: list[Evidence],
        sub_questions: list[str],
        reasoning_path: list[ReasoningStep],
    ) -> dict[str, Any]:
        """Fallback heuristic evaluation when LLM fails."""
        # Simple heuristic calculations
        completeness = len(reasoning_path) / max(len(sub_questions), 1) if sub_questions else 0.5
        completeness = min(1.0, completeness)

        coverage = min(1.0, len(evidence) / 10) if evidence else 0.0

        # Assume high consistency if no contradictions detected
        consistency = 0.8

        # Convergence based on evidence scores
        if evidence:
            top_scores = sorted([e.relevance_score for e in evidence], reverse=True)[:5]
            convergence = sum(top_scores) / len(top_scores) if top_scores else 0.5
        else:
            convergence = 0.0

        overall_sufficiency = (completeness + coverage + consistency + convergence) / 4

        # Determine action
        if overall_sufficiency >= 0.8:
            action = "FINALIZE"
        elif coverage < 0.5:
            action = "RETRIEVE_MORE"
        elif completeness < 0.8:
            action = "CONTINUE_CHAIN"
        else:
            action = "RE_RANK_EVIDENCE"

        return {
            "dimension_scores": {
                "completeness": completeness,
                "coverage": coverage,
                "consistency": consistency,
                "convergence": convergence,
            },
            "dimension_details": {"note": "Fallback heuristic evaluation"},
            "overall_sufficiency": overall_sufficiency,
            "action": action,
            "action_details": {},
            "early_stop": False,
            "early_stop_reason": "",
        }

    async def evolve_query(self, state: MACERState) -> QueryEvolution | None:
        """
        Evolve the query based on current understanding and gaps.

        Args:
            state: Current MACER state

        Returns:
            QueryEvolution if query was refined, None otherwise
        """
        if not self._enable_query_evolution:
            return None

        original_question = state.get("original_query", "")
        current_query = state.get("current_query", original_question)
        evidence = state.get("evidence", [])
        assessment = state.get("sufficiency_assessment")
        missing_aspects = assessment.missing_aspects if assessment else []

        # Format current evidence
        evidence_str = format_evidence_for_prompt(
            [
                {
                    "evidence_type": e.evidence_type.value,
                    "content": e.content[:200],
                    "relevance_score": e.relevance_score,
                }
                for e in evidence[:10]
            ]
        )

        try:
            result = await self._query_evolution_chain.ainvoke(
                {
                    "original_question": original_question,
                    "current_query": current_query,
                    "current_evidence": evidence_str,
                    "missing_aspects": ", ".join(missing_aspects)
                    if missing_aspects
                    else "None identified",
                }
            )

            evolved_query = result.get("evolved_query", current_query)

            # Only create evolution if query actually changed
            if evolved_query.lower().strip() != current_query.lower().strip():
                evolution = QueryEvolution(
                    original=current_query,
                    refined=evolved_query,
                    reason=result.get("reasoning", ""),
                    sub_questions=result.get("sub_questions", []),
                )

                logger.info(
                    "Query evolved",
                    evolution_type=result.get("evolution_type", "unknown"),
                    original=current_query[:50],
                    refined=evolved_query[:50],
                )

                return evolution

            return None

        except Exception as e:
            logger.warning("Query evolution failed", error=str(e))
            return None

    async def evolve_subgraph(
        self,
        state: MACERState,
    ) -> dict[str, Any]:
        """
        Decide how to evolve the subgraph based on current state.

        Returns guidance for the Retriever agent.

        Args:
            state: Current MACER state

        Returns:
            Dict with evolution directives
        """
        if not self._enable_subgraph_evolution:
            return {"action": "MAINTAIN"}

        question = state.get("current_query", state.get("original_query", ""))
        subgraph = state.get("current_subgraph", SubGraph())
        evidence = state.get("evidence", [])
        assessment = state.get("sufficiency_assessment")

        # Summarize current state
        subgraph_summary = format_subgraph_for_prompt(
            {
                "nodes": [{"name": n.name, "type": n.type} for n in subgraph.nodes[:15]],
                "edges": [
                    {
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "relation_type": e.relation_type,
                    }
                    for e in subgraph.edges[:15]
                ],
            }
        )

        evidence_summary = f"{len(evidence)} pieces of evidence, avg relevance: {sum(e.relevance_score for e in evidence) / len(evidence) if evidence else 0:.2f}"

        sufficiency_str = (
            f"Score: {assessment.score:.2f}, Missing: {', '.join(assessment.missing_aspects[:3])}"
            if assessment
            else "Not assessed"
        )

        try:
            result = await self._subgraph_evolution_chain.ainvoke(
                {
                    "question": question,
                    "subgraph_summary": subgraph_summary,
                    "evidence_summary": evidence_summary,
                    "sufficiency_assessment": sufficiency_str,
                }
            )

            logger.info(
                "Subgraph evolution decided",
                action=result.get("action", "MAINTAIN"),
                reasoning=result.get("reasoning", "")[:100],
            )

            return dict(result)

        except Exception as e:
            logger.warning("Subgraph evolution failed", error=str(e))
            return {"action": "MAINTAIN", "reasoning": f"Evolution failed: {str(e)}"}

    async def should_continue(self, state: MACERState) -> tuple[bool, ReasoningAction]:
        """
        Decide whether to continue reasoning or conclude.

        Enhanced with multi-hop awareness for better termination decisions.

        Args:
            state: Current MACER state

        Returns:
            Tuple of (should_continue, next_action)
        """
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", self._max_iterations)
        sufficiency_score = state.get("sufficiency_score", 0.0)
        evidence = state.get("evidence", [])
        reasoning_path = state.get("reasoning_path", [])
        metadata = state.get("metadata", {})
        subgraph = state.get("current_subgraph", SubGraph())
        question_type = state.get("question_type", QuestionType.FACTOID.value)
        path_completeness = state.get("path_completeness", 0.0)
        reasoning_chains = state.get("reasoning_chains", [])

        # Check if this is a multi-hop question
        is_multihop = question_type in [
            QuestionType.MULTIHOP.value,
            QuestionType.BRIDGE.value,
            QuestionType.COMPARISON.value,
        ]

        # Check for termination flag from earlier stages
        if state.get("should_terminate", False):
            logger.info("Termination flag already set")
            return False, ReasoningAction.CONCLUDE

        # Check for NO_DATA or NO_RESULTS scenarios
        if metadata.get("no_data", False):
            logger.info("NO_DATA scenario - cannot continue")
            return False, ReasoningAction.CONCLUDE

        # If no evidence found after multiple iterations, conclude
        if iteration >= 2 and not evidence and subgraph.node_count() == 0:
            logger.info("No progress after multiple iterations - concluding")
            return False, ReasoningAction.CONCLUDE

        # Hard stop at max iterations
        if iteration >= max_iterations:
            logger.info("Max iterations reached, concluding")
            return False, ReasoningAction.CONCLUDE

        # For multi-hop questions, check path completeness before concluding
        if is_multihop:
            # Count path evidence
            path_evidence_count = sum(1 for e in evidence if e.evidence_type == EvidenceType.PATH)

            # Don't conclude too early for multi-hop if paths aren't complete
            if path_completeness < 0.5 and path_evidence_count == 0 and iteration < max_iterations - 1:
                logger.info(
                    "Multi-hop: continuing despite sufficiency - paths incomplete",
                    path_completeness=path_completeness,
                    path_evidence=path_evidence_count,
                )
                return True, ReasoningAction.EXPLORE

            # Require higher threshold for multi-hop questions
            multihop_threshold = self._sufficiency_threshold + 0.1
            if sufficiency_score >= multihop_threshold and path_completeness >= 0.6:
                logger.info(
                    "Multi-hop sufficiency met",
                    score=sufficiency_score,
                    path_completeness=path_completeness,
                )
                return False, ReasoningAction.CONCLUDE
        else:
            # Quick conclusion if high sufficiency for simple questions
            if sufficiency_score >= self._sufficiency_threshold:
                logger.info("Sufficiency threshold met", score=sufficiency_score)
                return False, ReasoningAction.CONCLUDE

        # Calculate recent progress
        recent_progress = "No progress data"
        if len(reasoning_path) >= 2:
            recent_steps = reasoning_path[-2:]
            recent_deltas = [s.sufficiency_delta for s in recent_steps]
            avg_delta = sum(recent_deltas) / len(recent_deltas)
            recent_progress = f"Avg sufficiency delta: {avg_delta:.2f}"

        # Add multi-hop context
        if is_multihop:
            recent_progress += f", Path completeness: {path_completeness:.2f}, Chains: {len(reasoning_chains)}"

        try:
            result = await self._iteration_chain.ainvoke(
                {
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "sufficiency_score": sufficiency_score,
                    "evidence_count": len(evidence),
                    "recent_progress": recent_progress,
                }
            )

            should_continue = result.get("should_continue", True)
            action_str = result.get("next_action", "EXPLORE")

            try:
                next_action = ReasoningAction(action_str.lower())
            except ValueError:
                next_action = (
                    ReasoningAction.EXPLORE if should_continue else ReasoningAction.CONCLUDE
                )

            # Override for multi-hop: don't conclude if path incomplete
            if not should_continue and is_multihop and path_completeness < 0.5 and iteration < max_iterations - 1:
                logger.info("Overriding conclude for multi-hop - paths incomplete")
                should_continue = True
                next_action = ReasoningAction.EXPLORE

            logger.info(
                "Iteration control decision",
                continue_reasoning=should_continue,
                next_action=next_action.value,
                is_multihop=is_multihop,
                path_completeness=path_completeness,
                reasoning=result.get("reasoning", "")[:100],
            )

            return should_continue, next_action

        except Exception as e:
            logger.warning("Iteration control failed", error=str(e))
            # Default: continue if not at max and low sufficiency
            if iteration < max_iterations and sufficiency_score < self._sufficiency_threshold:
                return True, ReasoningAction.EXPLORE
            return False, ReasoningAction.CONCLUDE

    def create_reasoning_step(
        self,
        state: MACERState,
        action: ReasoningAction,
        thought: str,
        observation: str = "",
        query_evolution: QueryEvolution | None = None,
        subgraph_change: str | None = None,
        new_evidence_ids: list[str] | None = None,
    ) -> ReasoningStep:
        """
        Create a new reasoning step for the chain.

        Args:
            state: Current state
            action: Action being taken
            thought: Reasoning thought
            observation: What was observed
            query_evolution: If query was evolved
            subgraph_change: Description of subgraph change
            new_evidence_ids: IDs of new evidence

        Returns:
            New ReasoningStep
        """
        reasoning_path = state.get("reasoning_path", [])
        previous_score = state.get("sufficiency_score", 0.0)
        current_assessment = state.get("sufficiency_assessment")
        current_score = current_assessment.score if current_assessment else previous_score

        step = ReasoningStep(
            step_number=len(reasoning_path) + 1,
            action=action,
            thought=thought,
            observation=observation,
            query_evolution=query_evolution.refined if query_evolution else None,
            subgraph_change=subgraph_change,
            new_evidence=new_evidence_ids or [],
            sufficiency_delta=current_score - previous_score,
        )

        return step

    async def reflect(self, state: MACERState) -> dict[str, Any]:
        """
        Main reflection method for MACER pipeline.

        Performs meta-cognitive assessment and decides next steps.
        Enhanced with multi-hop path completeness tracking.

        Args:
            state: Current MACER state

        Returns:
            State updates with assessment, evolutions, reasoning step, and path completeness
        """
        iteration = state.get("iteration", 0)
        metadata = state.get("metadata", {})
        question_type = state.get("question_type", QuestionType.FACTOID.value)
        evidence = state.get("evidence", [])
        subgraph = state.get("current_subgraph", SubGraph())

        logger.info("Reflector agent starting", iteration=iteration, question_type=question_type)

        # Check for NO_DATA scenario from Constructor
        if metadata.get("no_data", False):
            logger.warning("NO_DATA scenario detected - immediate termination")
            assessment = SufficiencyAssessment(
                score=0.0,
                has_enough_evidence=False,
                missing_aspects=["Knowledge graph is empty - no data to search"],
                confidence_factors={"completeness": 0.0, "reliability": 0.0, "consistency": 0.0, "path_completeness": 0.0},
                recommendation=ReasoningAction.CONCLUDE,
                reasoning="Cannot proceed - knowledge graph contains no data. Please ingest documents first.",
            )
            reasoning_step = ReasoningStep(
                step_number=1,
                action=ReasoningAction.CONCLUDE,
                thought="Knowledge graph is empty. Unable to find any relevant information.",
                observation="No entities or evidence available.",
                sufficiency_delta=0.0,
            )
            return {
                "sufficiency_score": 0.0,
                "sufficiency_assessment": assessment,
                "path_completeness": 0.0,
                "reasoning_path": [reasoning_step],
                "should_terminate": True,
                "iteration": 1,
                "metadata": {**metadata, "termination_reason": "NO_DATA"},
            }

        # Check for NO_RESULTS scenario (data exists but nothing matched)
        if metadata.get("no_results", False) and iteration == 0:
            logger.warning("NO_RESULTS scenario - no entities matched the query")
            # Give it one more chance with evolved query if it's early
            if iteration < 2 and not evidence:
                logger.info("Attempting query refinement for better results")
                # Continue to normal assessment but bias toward REFINE

        # Step 1: Assess sufficiency (includes path completeness calculation)
        assessment = await self.assess_sufficiency(state)

        # Step 1.5: 4-Dimension Evaluation for enhanced reflection
        four_dim_result = await self.evaluate_four_dimensions(state)
        dimension_scores = four_dim_result.get("dimension_scores", {})

        # Log 4-dimension scores
        logger.info(
            "4-dimension scores",
            completeness=dimension_scores.get("completeness", 0.0),
            coverage=dimension_scores.get("coverage", 0.0),
            consistency=dimension_scores.get("consistency", 0.0),
            convergence=dimension_scores.get("convergence", 0.0),
            overall=four_dim_result.get("overall_sufficiency", 0.0),
            recommended_action=four_dim_result.get("action", "UNKNOWN"),
        )

        # Calculate path completeness for state update
        path_completeness = calculate_path_completeness(evidence, subgraph, question_type)

        # Check for early stop from 4-dimension evaluation
        if four_dim_result.get("early_stop", False):
            logger.info(
                "Early stop triggered by 4-dimension evaluation",
                reason=four_dim_result.get("early_stop_reason", ""),
            )

        # Step 2: Decide if we should continue
        should_continue, next_action = await self.should_continue(
            {
                **state,
                "sufficiency_score": assessment.score,
                "sufficiency_assessment": assessment,
                "path_completeness": path_completeness,
            }
        )

        # Step 3: If continuing, evolve query and subgraph
        query_evolution = None
        subgraph_evolution = None

        if should_continue and next_action != ReasoningAction.CONCLUDE:
            # Evolve query if needed (now triggers on EXPLORE too for better evidence discovery)
            # This ensures query variation across iterations to prevent stagnant scores
            if next_action in [ReasoningAction.REFINE, ReasoningAction.FOCUS, ReasoningAction.EXPLORE]:
                query_evolution = await self.evolve_query(
                    {
                        **state,
                        "sufficiency_assessment": assessment,
                    }
                )
                logger.info(
                    "Query evolution triggered",
                    action=next_action.value,
                    evolved=query_evolution.refined[:50] if query_evolution else "None",
                )

            # Decide subgraph evolution
            subgraph_evolution = await self.evolve_subgraph(
                {
                    **state,
                    "sufficiency_assessment": assessment,
                }
            )

        # Step 4: Create reasoning step
        thought = self._generate_thought(assessment, next_action, query_evolution)

        # Include path completeness in observation for multi-hop questions
        is_multihop = question_type in [
            QuestionType.MULTIHOP.value,
            QuestionType.BRIDGE.value,
            QuestionType.COMPARISON.value,
        ]
        if is_multihop:
            observation = f"Sufficiency: {assessment.score:.2f}, Path Completeness: {path_completeness:.2f}, Missing: {', '.join(assessment.missing_aspects[:2])}"
        else:
            observation = f"Sufficiency: {assessment.score:.2f}, Missing: {', '.join(assessment.missing_aspects[:2])}"

        reasoning_step = self.create_reasoning_step(
            state=state,
            action=next_action,
            thought=thought,
            observation=observation,
            query_evolution=query_evolution,
            subgraph_change=subgraph_evolution.get("action") if subgraph_evolution else None,
        )

        # Update query history
        query_history = list(state.get("query_history", []))
        if query_evolution:
            query_history.append(query_evolution)

        # Update reasoning path
        reasoning_path = list(state.get("reasoning_path", []))
        reasoning_path.append(reasoning_step)

        # Count path evidence for logging
        path_evidence_count = sum(1 for e in evidence if e.evidence_type == EvidenceType.PATH)

        logger.info(
            "Reflector completed",
            sufficiency=assessment.score,
            path_completeness=path_completeness,
            path_evidence=path_evidence_count,
            should_continue=should_continue,
            next_action=next_action.value,
        )

        # Consider early stop from 4-dimension evaluation
        if four_dim_result.get("early_stop", False) and iteration >= 2:
            should_continue = False

        return {
            "sufficiency_score": assessment.score,
            "sufficiency_assessment": assessment,
            "path_completeness": path_completeness,
            "current_query": query_evolution.refined
            if query_evolution
            else state.get("current_query"),
            "query_history": query_history,
            "reasoning_path": reasoning_path,
            "should_terminate": not should_continue,
            "iteration": state.get("iteration", 0) + 1,
            "metadata": {
                **(state.get("metadata", {})),
                "last_action": next_action.value,
                "subgraph_evolution": subgraph_evolution,
                "path_completeness": path_completeness,
                "four_dimension_scores": dimension_scores,
                "four_dimension_action": four_dim_result.get("action", ""),
                "early_stop": four_dim_result.get("early_stop", False),
            },
        }

    def _generate_thought(
        self,
        assessment: SufficiencyAssessment,
        action: ReasoningAction,
        query_evolution: QueryEvolution | None,
    ) -> str:
        """Generate a thought description for the reasoning step."""
        parts = [f"Evidence sufficiency is {assessment.score:.0%}."]

        if assessment.missing_aspects:
            parts.append(f"Missing: {', '.join(assessment.missing_aspects[:2])}.")

        if action == ReasoningAction.CONCLUDE:
            parts.append("Sufficient evidence collected, ready to conclude.")
        elif action == ReasoningAction.EXPLORE:
            parts.append("Need to explore more of the graph.")
        elif action == ReasoningAction.FOCUS:
            parts.append("Narrowing focus to most relevant areas.")
        elif action == ReasoningAction.REFINE:
            if query_evolution:
                parts.append(f"Refining query to: '{query_evolution.refined[:50]}...'")
            else:
                parts.append("Attempting to refine the search approach.")
        elif action == ReasoningAction.BACKTRACK:
            parts.append("Evidence contradictory or unhelpful, backtracking.")

        return " ".join(parts)

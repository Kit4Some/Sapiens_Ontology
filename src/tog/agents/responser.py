"""
Responser Agent for MACER.

Responsible for:
1. Evidence Synthesis (with multi-hop path awareness)
2. Natural Language Answer Generation
3. Reasoning Path Explanation
4. Confidence Scoring (enhanced with path completeness)
5. Multi-hop Reasoning Chain Synthesis
6. Grounded Reasoning (Graph DB priority, LLM fallback)
7. Advanced Multi-Layer Response Generation (NEW)
"""

from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from src.tog.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_SYNTHESIS_PROMPT,
    BENCHMARK_DIRECT_ANSWER_PROMPT,
    EVIDENCE_SYNTHESIS_PROMPT,
    GENERAL_ANSWER_PROMPT,
    GROUNDED_REASONING_PROMPT,
    REASONING_EXPLANATION_PROMPT,
    format_evidence_for_prompt,
    format_reasoning_path_for_prompt,
    get_question_type_config,
    get_max_answer_words,
    should_use_advanced_synthesis,
    DIRECT_ANSWER_QUESTION_TYPES,
)
from src.tog.state import (
    AlternativeAnswer,
    AnswerClassification,
    AnswerComponents,
    AnswerProvenance,
    ChainValidation,
    ChainValidationStatus,
    Evidence,
    EvidenceChainLink,
    EvidenceConfidenceType,
    EvidenceType,
    GroundingSource,
    HallucinationRisk,
    MACERState,
    QuestionType,
    ReasoningStep,
    SubGraph,
    SynthesizedAnswer,
    calculate_path_completeness,
)
from src.tog.advanced_synthesizer import (
    AdvancedResponseSynthesizer,
    AdvancedResponse,
    ResponseDepth,
    create_advanced_synthesizer,
)

logger = structlog.get_logger(__name__)


# Question-type routing for response synthesis
# FACTOID and YESNO questions should get brief, direct answers
# Complex questions (MULTIHOP, COMPARISON, BRIDGE) may use advanced synthesis
QUESTION_TYPE_RESPONSE_DEPTH: dict[str, ResponseDepth] = {
    # HotpotQA-style short answer types
    QuestionType.FACTOID.value: ResponseDepth.BRIEF,
    QuestionType.YESNO.value: ResponseDepth.BRIEF,
    QuestionType.AGGREGATION.value: ResponseDepth.STANDARD,
    QuestionType.COMPARISON.value: ResponseDepth.STANDARD,
    QuestionType.MULTIHOP.value: ResponseDepth.STANDARD,
    QuestionType.BRIDGE.value: ResponseDepth.STANDARD,
    # Extended types for General Document QA
    QuestionType.DEFINITION.value: ResponseDepth.STANDARD,
    QuestionType.PROCEDURE.value: ResponseDepth.DETAILED,
    QuestionType.CAUSE_EFFECT.value: ResponseDepth.STANDARD,
    QuestionType.LIST.value: ResponseDepth.STANDARD,
    QuestionType.NARRATIVE.value: ResponseDepth.DETAILED,
    QuestionType.OPINION.value: ResponseDepth.DETAILED,
}

# Skip advanced synthesis entirely for simple question types
# These use direct answer extraction for benchmark evaluation
SKIP_ADVANCED_SYNTHESIS_TYPES: set[str] = {
    QuestionType.FACTOID.value,
    QuestionType.YESNO.value,
}

# Question types that should use GENERAL_ANSWER_PROMPT instead of BENCHMARK_DIRECT_ANSWER_PROMPT
GENERAL_ANSWER_QUESTION_TYPES: set[str] = {
    QuestionType.DEFINITION.value,
    QuestionType.PROCEDURE.value,
    QuestionType.CAUSE_EFFECT.value,
    QuestionType.LIST.value,
    QuestionType.NARRATIVE.value,
    QuestionType.OPINION.value,
}


class ResponserAgent:
    """
    Responser Agent for MACER framework.

    Synthesizes evidence and reasoning into a coherent,
    well-explained natural language answer.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        include_explanation: bool = True,
        max_evidence_in_response: int = 5,
        use_advanced_synthesis: bool = True,
        response_depth: ResponseDepth = ResponseDepth.DETAILED,
        include_korean: bool = True,
    ) -> None:
        """
        Initialize the Responser agent.

        Args:
            llm: LangChain chat model
            include_explanation: Whether to include detailed explanation
            max_evidence_in_response: Max evidence pieces to cite
            use_advanced_synthesis: Use advanced multi-layer response synthesis (NEW)
            response_depth: Level of detail in responses (BRIEF/STANDARD/DETAILED/COMPREHENSIVE)
            include_korean: Include Korean translations in responses
        """
        self._llm = llm
        self._include_explanation = include_explanation
        self._max_evidence = max_evidence_in_response
        self._use_advanced_synthesis = use_advanced_synthesis
        self._response_depth = response_depth
        self._include_korean = include_korean

        # Build chains
        self._json_parser = JsonOutputParser()
        self._str_parser = StrOutputParser()
        self._synthesis_chain = EVIDENCE_SYNTHESIS_PROMPT | self._llm | self._json_parser
        self._answer_chain = ANSWER_GENERATION_PROMPT | self._llm | self._json_parser
        self._explanation_chain = REASONING_EXPLANATION_PROMPT | self._llm | self._str_parser
        self._grounded_reasoning_chain = GROUNDED_REASONING_PROMPT | self._llm | self._json_parser
        self._answer_synthesis_chain = ANSWER_SYNTHESIS_PROMPT | self._llm | self._json_parser
        self._direct_answer_chain = BENCHMARK_DIRECT_ANSWER_PROMPT | self._llm | self._json_parser
        # General document QA chain for extended question types
        self._general_answer_chain = GENERAL_ANSWER_PROMPT | self._llm | self._json_parser

        # Advanced synthesizer for rich, multi-layer responses
        self._advanced_synthesizer: AdvancedResponseSynthesizer | None = None
        if use_advanced_synthesis:
            self._advanced_synthesizer = create_advanced_synthesizer(
                llm=llm,
                response_depth=response_depth,
                include_korean=include_korean,
            )
            logger.info(
                "Advanced Response Synthesizer enabled",
                response_depth=response_depth.value,
                include_korean=include_korean,
            )

    async def synthesize_evidence(
        self,
        question: str,
        evidence: list[Evidence],
        reasoning_path: list[ReasoningStep],
    ) -> dict[str, Any]:
        """
        Synthesize all evidence into coherent facts and inferences.

        Args:
            question: Original question
            evidence: Collected evidence
            reasoning_path: Chain of reasoning steps

        Returns:
            Synthesis result with facts, inferences, contradictions
        """
        # Sort evidence by relevance
        sorted_evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)

        # Format for prompt
        evidence_str = format_evidence_for_prompt(
            [
                {
                    "evidence_type": e.evidence_type.value,
                    "content": e.content,
                    "relevance_score": e.relevance_score,
                }
                for e in sorted_evidence[:15]
            ]
        )

        reasoning_str = format_reasoning_path_for_prompt(
            [
                {"step_number": s.step_number, "action": s.action.value, "thought": s.thought}
                for s in reasoning_path
            ]
        )

        try:
            result = await self._synthesis_chain.ainvoke(
                {
                    "question": question,
                    "evidence": evidence_str,
                    "reasoning_path": reasoning_str,
                }
            )

            logger.info(
                "Evidence synthesized",
                facts=len(result.get("synthesized_facts", [])),
                inferences=len(result.get("inferences", [])),
            )

            return dict(result)

        except Exception as e:
            logger.error("Evidence synthesis failed", error=str(e))
            return {
                "synthesized_facts": [],
                "inferences": [],
                "contradictions": [],
                "key_evidence_used": [],
                "answer_confidence": 0.5,
            }

    async def generate_answer(
        self,
        question: str,
        synthesis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate the final natural language answer.

        Args:
            question: Original question
            synthesis: Evidence synthesis result

        Returns:
            Answer with confidence and supporting info
        """
        # Format synthesis for prompt
        synthesis_str = self._format_synthesis(synthesis)

        try:
            result = await self._answer_chain.ainvoke(
                {
                    "question": question,
                    "synthesis": synthesis_str,
                }
            )

            answer = result.get("answer", "Unable to generate answer.")
            confidence = float(result.get("confidence", 0.5))
            answer_type = result.get("answer_type", "UNCERTAIN")

            logger.info(
                "Answer generated",
                confidence=confidence,
                answer_type=answer_type,
                answer_length=len(answer),
            )

            return {
                "answer": answer,
                "confidence": confidence,
                "answer_type": answer_type,
                "supporting_evidence": result.get("supporting_evidence", []),
                "caveats": result.get("caveats", []),
                "explanation": result.get("explanation", ""),
            }

        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return {
                "answer": f"I was unable to generate a complete answer due to: {str(e)}",
                "confidence": 0.2,
                "answer_type": "FAILED",
                "supporting_evidence": [],
                "caveats": ["Answer generation encountered an error"],
                "explanation": "",
            }

    async def extract_benchmark_direct_answer(
        self,
        question: str,
        full_answer: str,
        question_type: str = "factoid",
    ) -> str:
        """
        Extract a concise, benchmark-style direct answer from a full answer.

        This method uses LLM to extract just the core answer value (entity name,
        number, date, etc.) suitable for benchmark evaluation (HotpotQA, etc.).

        For extended question types (DEFINITION, PROCEDURE, etc.), the max word
        limit is dynamically adjusted based on question type configuration.

        Args:
            question: Original question
            full_answer: The full detailed answer
            question_type: Type of question for dynamic length validation

        Returns:
            Concise direct answer (e.g., "Curtis Martin", "1995", "Microsoft")
        """
        # Get dynamic max words based on question type
        max_words = get_max_answer_words(question_type)

        try:
            result = await self._direct_answer_chain.ainvoke({
                "question": question,
                "full_answer": full_answer,
            })

            direct_answer = result.get("direct_answer", "")

            # Validate the direct answer with dynamic length limit
            if direct_answer and len(direct_answer.split()) <= max_words:
                logger.info(
                    "Benchmark direct answer extracted",
                    direct_answer=direct_answer[:100],  # Truncate for logging
                    answer_type=result.get("answer_type", "UNKNOWN"),
                    confidence=result.get("confidence", 0.0),
                    max_words=max_words,
                    question_type=question_type,
                )
                return direct_answer

            # Fallback: try to extract first entity-like phrase
            logger.warning(
                "Direct answer too long, using fallback extraction",
                direct_answer_length=len(direct_answer.split()),
                max_words=max_words,
            )
            return self._fallback_direct_answer_extraction(question, full_answer)

        except Exception as e:
            logger.warning("Direct answer extraction failed", error=str(e))
            return self._fallback_direct_answer_extraction(question, full_answer)

    def _fallback_direct_answer_extraction(self, question: str, full_answer: str) -> str:
        """
        Simple rule-based fallback for direct answer extraction.

        Args:
            question: Original question
            full_answer: The full detailed answer

        Returns:
            Best-effort concise answer
        """
        import re

        # Clean the answer
        answer = full_answer.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            r"^Based on (the )?(knowledge graph|evidence|available information)[,.]?\s*",
            r"^According to (the )?(knowledge graph|evidence)[,.]?\s*",
            r"^The answer is\s*",
            r"^It is\s*",
            r"^This is\s*",
        ]

        for pattern in prefixes_to_remove:
            answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)

        # Try to extract quoted content
        quoted = re.findall(r'"([^"]+)"', answer)
        if quoted and len(quoted[0].split()) <= 5:
            return quoted[0]

        # Try to extract the first sentence
        first_sentence = answer.split('.')[0].strip()

        # If first sentence is short enough, use it
        if len(first_sentence.split()) <= 8:
            return first_sentence

        # Try to find named entities (capitalized phrases)
        entities = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b', first_sentence)
        if entities:
            return entities[0]

        # Last resort: first 5 words
        words = first_sentence.split()[:5]
        return " ".join(words)

    async def generate_general_answer(
        self,
        question: str,
        question_type: str,
        evidence: list[Evidence],
        reasoning_path: list[ReasoningStep],
    ) -> dict[str, Any]:
        """
        Generate answer using the general document QA prompt.

        This method is used for extended question types (DEFINITION, PROCEDURE,
        CAUSE_EFFECT, LIST, NARRATIVE, OPINION) that require more detailed
        responses than benchmark-style short answers.

        Args:
            question: Original question
            question_type: Type of question (determines response format)
            evidence: Collected evidence from knowledge graph
            reasoning_path: Chain of reasoning steps

        Returns:
            Answer with confidence, direct_answer, and metadata
        """
        # Format evidence for prompt
        evidence_str = format_evidence_for_prompt(
            [
                {
                    "evidence_type": e.evidence_type.value,
                    "content": e.content,
                    "relevance_score": e.relevance_score,
                }
                for e in sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:15]
            ]
        )

        # Format reasoning path
        reasoning_str = format_reasoning_path_for_prompt(
            [
                {"step_number": s.step_number, "action": s.action.value, "thought": s.thought}
                for s in reasoning_path
            ]
        )

        try:
            result = await self._general_answer_chain.ainvoke({
                "question": question,
                "question_type": question_type.upper(),
                "evidence": evidence_str,
                "reasoning_path": reasoning_str,
            })

            answer = result.get("answer", "Unable to generate answer.")
            direct_answer = result.get("direct_answer", "")
            confidence = float(result.get("confidence", 0.5))
            evidence_quality = result.get("evidence_quality", "MEDIUM")

            logger.info(
                "General document QA answer generated",
                question_type=question_type,
                answer_length=len(answer.split()),
                confidence=confidence,
                evidence_quality=evidence_quality,
            )

            return {
                "answer": answer,
                "direct_answer": direct_answer,
                "confidence": confidence,
                "answer_type": result.get("answer_type", question_type.upper()),
                "evidence_quality": evidence_quality,
                "key_entities": result.get("key_entities", []),
                "caveats": result.get("caveats", []),
            }

        except Exception as e:
            logger.error("General answer generation failed", error=str(e))
            return {
                "answer": f"I was unable to generate a complete answer due to: {str(e)}",
                "direct_answer": "",
                "confidence": 0.2,
                "answer_type": "FAILED",
                "evidence_quality": "LOW",
                "key_entities": [],
                "caveats": ["Answer generation encountered an error"],
            }

    def _select_answer_prompt(self, question_type: str) -> str:
        """
        Select the appropriate answer prompt based on question type.

        Args:
            question_type: Type of question

        Returns:
            Prompt type to use: "benchmark" for HotpotQA-style, "general" for document QA
        """
        if question_type.lower() in DIRECT_ANSWER_QUESTION_TYPES:
            return "benchmark"
        elif question_type.lower() in GENERAL_ANSWER_QUESTION_TYPES:
            return "general"
        else:
            # Default to benchmark for unknown types to maintain backward compatibility
            return "benchmark"

    async def generate_grounded_answer(
        self,
        question: str,
        evidence: list[Evidence],
        reasoning_path: list[ReasoningStep],
    ) -> dict[str, Any]:
        """
        Generate answer with grounding priority: Graph DB first, LLM fallback.

        This method ensures that:
        1. Graph evidence is ALWAYS prioritized over LLM knowledge
        2. LLM knowledge is used as fallback when graph evidence is insufficient
        3. Source attribution is explicit for each fact in the answer

        Args:
            question: Original question
            evidence: Collected evidence from knowledge graph
            reasoning_path: Chain of reasoning steps

        Returns:
            Grounded answer with source attribution and confidence breakdown
        """
        # Classify evidence by confidence type
        explicit_evidence = [e for e in evidence if hasattr(e, 'confidence_type') and e.confidence_type == EvidenceConfidenceType.EXPLICIT]
        implicit_evidence = [e for e in evidence if hasattr(e, 'confidence_type') and e.confidence_type == EvidenceConfidenceType.IMPLICIT]
        other_evidence = [e for e in evidence if not hasattr(e, 'confidence_type') or e.confidence_type == EvidenceConfidenceType.INFERRED]

        # Format evidence by type
        evidence_str = self._format_evidence_with_types(evidence)

        # Compute evidence type summary
        evidence_types_str = (
            f"EXPLICIT: {len(explicit_evidence)}, "
            f"IMPLICIT: {len(implicit_evidence)}, "
            f"INFERRED: {len(other_evidence)}"
        )

        # Format reasoning path
        prior_steps_str = format_reasoning_path_for_prompt([
            {"step_number": s.step_number, "action": s.action.value, "thought": s.thought}
            for s in reasoning_path
        ])

        try:
            result = await self._grounded_reasoning_chain.ainvoke({
                "question": question,
                "graph_evidence": evidence_str,
                "evidence_types": evidence_types_str,
                "prior_steps": prior_steps_str,
            })

            answer = result.get("answer", "Unable to generate grounded answer.")
            confidence = float(result.get("confidence", 0.5))
            answer_source = result.get("answer_source", "UNKNOWN")
            hallucination_risk = result.get("hallucination_risk", "MEDIUM")

            # Log grounding breakdown
            grounding = result.get("grounding_breakdown", {})
            logger.info(
                "Grounded answer generated",
                answer_source=answer_source,
                confidence=confidence,
                hallucination_risk=hallucination_risk,
                graph_explicit_facts=len(grounding.get("graph_explicit", [])),
                graph_implicit_facts=len(grounding.get("graph_implicit", [])),
                llm_supplementary_facts=len(grounding.get("llm_supplementary", [])),
            )

            return {
                "answer": answer,
                "confidence": confidence,
                "answer_source": answer_source,
                "hallucination_risk": hallucination_risk,
                "grounding_breakdown": grounding,
                "confidence_breakdown": result.get("confidence_breakdown", {}),
                "disclaimers": result.get("disclaimers", []),
            }

        except Exception as e:
            logger.error("Grounded answer generation failed", error=str(e))
            # Fallback to regular answer generation
            synthesis = await self.synthesize_evidence(question, evidence, reasoning_path)
            return await self.generate_answer(question, synthesis)

    def _format_evidence_with_types(self, evidence: list[Evidence]) -> str:
        """Format evidence with explicit type annotations for grounding."""
        lines = []
        for i, ev in enumerate(sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:15], 1):
            ev_type = ev.evidence_type.value.upper()
            confidence_type = "UNKNOWN"
            if hasattr(ev, 'confidence_type'):
                confidence_type = ev.confidence_type.value.upper()

            lines.append(
                f"{i}. [{ev_type}][{confidence_type}] (score: {ev.relevance_score:.2f}) "
                f"{ev.content[:300]}"
            )

        return "\n".join(lines) if lines else "No evidence collected."

    def validate_reasoning_chain(
        self,
        reasoning_path: list[ReasoningStep],
        evidence: list[Evidence],
        sub_questions: list[str],
    ) -> ChainValidation:
        """
        Validate the reasoning chain for answer synthesis.

        Checks:
        1. Each step has associated evidence
        2. No circular dependencies
        3. All sub-questions have answers

        Args:
            reasoning_path: Chain of reasoning steps
            evidence: Collected evidence
            sub_questions: Decomposed sub-questions

        Returns:
            ChainValidation result
        """
        issues: list[str] = []
        circular_deps: list[str] = []
        missing_evidence_steps: list[int] = []

        # Track evidence IDs used by each step
        evidence_ids = {e.id for e in evidence}
        step_evidence_map: dict[int, list[str]] = {}

        # Validate each reasoning step
        for step in reasoning_path:
            step_num = step.step_number

            # Check if step has new evidence
            if step.new_evidence:
                valid_evidence = [eid for eid in step.new_evidence if eid in evidence_ids]
                if valid_evidence:
                    step_evidence_map[step_num] = valid_evidence
                else:
                    missing_evidence_steps.append(step_num)
                    issues.append(f"Step {step_num}: Referenced evidence IDs not found")
            else:
                # Steps without new evidence might be control steps (BACKTRACK, etc.)
                if step.action.value not in ["backtrack", "conclude"]:
                    missing_evidence_steps.append(step_num)

        # Check for circular dependencies in reasoning path
        seen_queries: set[str] = set()
        for step in reasoning_path:
            if step.query_evolution:
                if step.query_evolution in seen_queries:
                    circular_deps.append(f"Query '{step.query_evolution[:50]}...' appears multiple times")
                seen_queries.add(step.query_evolution)

        # Check sub-question coverage
        if sub_questions:
            # Simple heuristic: check if evidence mentions sub-question topics
            covered_questions = 0
            for sq in sub_questions:
                sq_lower = sq.lower()
                for ev in evidence:
                    if any(word in ev.content.lower() for word in sq_lower.split()[:3]):
                        covered_questions += 1
                        break

            if covered_questions < len(sub_questions):
                issues.append(
                    f"Only {covered_questions}/{len(sub_questions)} sub-questions have evidence"
                )

        # Calculate validation metrics
        steps_total = len(reasoning_path)
        steps_with_evidence = len(step_evidence_map)
        steps_validated = steps_with_evidence

        # Determine validation status
        if not issues and not circular_deps and steps_validated == steps_total:
            status = ChainValidationStatus.VALID
        elif circular_deps:
            status = ChainValidationStatus.INVALID
        elif steps_validated >= steps_total * 0.5:
            status = ChainValidationStatus.PARTIAL
        else:
            status = ChainValidationStatus.INVALID

        return ChainValidation(
            status=status,
            steps_validated=steps_validated,
            steps_total=steps_total,
            issues=issues,
            circular_dependencies=circular_deps,
            missing_evidence_steps=missing_evidence_steps,
        )

    def build_evidence_provenance(
        self,
        evidence: list[Evidence],
        reasoning_path: list[ReasoningStep],
        grounding_result: dict[str, Any],
    ) -> AnswerProvenance:
        """
        Build full provenance tracking for the synthesized answer.

        Args:
            evidence: Collected evidence
            reasoning_path: Chain of reasoning steps
            grounding_result: Result from grounded reasoning

        Returns:
            AnswerProvenance with full evidence chain
        """
        evidence_chain: list[EvidenceChainLink] = []
        primary_sources: list[str] = []
        supporting_sources: list[str] = []

        # Determine grounding source
        answer_source = grounding_result.get("answer_source", "UNKNOWN")
        grounding_source_map = {
            "GRAPH_ONLY": GroundingSource.GRAPH_ONLY,
            "GRAPH_PRIMARY": GroundingSource.GRAPH_PRIMARY,
            "LLM_SUPPLEMENTED": GroundingSource.LLM_SUPPLEMENTED,
            "LLM_ONLY": GroundingSource.LLM_ONLY,
        }
        grounding_source = grounding_source_map.get(answer_source, GroundingSource.LLM_ONLY)

        # Build evidence chain from high-relevance evidence
        sorted_evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)

        # Group evidence by hop_index for chain representation
        hop_groups: dict[int, list[Evidence]] = {}
        for ev in sorted_evidence:
            hop = ev.hop_index if hasattr(ev, "hop_index") else 0
            if hop not in hop_groups:
                hop_groups[hop] = []
            hop_groups[hop].append(ev)

        # Build chain links from evidence
        for hop_idx in sorted(hop_groups.keys()):
            hop_evidence = hop_groups[hop_idx]

            for ev in hop_evidence[:3]:  # Top 3 per hop
                # Build hop path from evidence
                hop_path = []
                if hasattr(ev, "path_nodes") and ev.path_nodes:
                    for i, node in enumerate(ev.path_nodes):
                        hop_path.append(node)
                        if i < len(ev.path_predicates) if hasattr(ev, "path_predicates") else 0:
                            hop_path.append(f"→[{ev.path_predicates[i]}]→")
                elif ev.source_nodes:
                    hop_path = ev.source_nodes

                # Collect step confidences from reasoning path
                step_confidences = []
                for step in reasoning_path:
                    if ev.id in step.new_evidence:
                        # Use sufficiency delta as proxy for step confidence
                        step_conf = 0.5 + step.sufficiency_delta
                        step_confidences.append(max(0.0, min(1.0, step_conf)))

                if not step_confidences:
                    step_confidences = [ev.relevance_score]

                chain_link = EvidenceChainLink(
                    claim=ev.content[:200],
                    evidence_ids=[ev.id],
                    hop_path=hop_path,
                    step_confidences=step_confidences,
                    final_confidence=min(step_confidences) if step_confidences else ev.relevance_score,
                )
                evidence_chain.append(chain_link)

                # Categorize as primary or supporting
                if ev.relevance_score >= 0.7:
                    primary_sources.append(ev.id)
                else:
                    supporting_sources.append(ev.id)

        return AnswerProvenance(
            evidence_chain=evidence_chain,
            primary_sources=primary_sources,
            supporting_sources=supporting_sources,
            grounding_source=grounding_source,
        )

    def determine_hallucination_risk(
        self,
        grounding_result: dict[str, Any],
        evidence: list[Evidence],
    ) -> HallucinationRisk:
        """
        Determine hallucination risk based on grounding and evidence.

        Args:
            grounding_result: Result from grounded reasoning
            evidence: Collected evidence

        Returns:
            HallucinationRisk level
        """
        answer_source = grounding_result.get("answer_source", "UNKNOWN")
        explicit_count = sum(
            1 for e in evidence
            if hasattr(e, "confidence_type") and e.confidence_type == EvidenceConfidenceType.EXPLICIT
        )
        total_evidence = len(evidence)

        # Risk based on answer source
        if answer_source == "GRAPH_ONLY" or answer_source == "GRAPH_PRIMARY":
            base_risk = HallucinationRisk.LOW
        elif answer_source == "LLM_SUPPLEMENTED":
            base_risk = HallucinationRisk.MEDIUM
        else:
            base_risk = HallucinationRisk.HIGH

        # Adjust based on explicit evidence ratio
        if total_evidence > 0:
            explicit_ratio = explicit_count / total_evidence
            if explicit_ratio >= 0.5 and base_risk == HallucinationRisk.MEDIUM:
                return HallucinationRisk.LOW
            if explicit_ratio < 0.2 and base_risk == HallucinationRisk.LOW:
                return HallucinationRisk.MEDIUM

        return base_risk

    async def synthesize_answer_with_provenance(
        self,
        question: str,
        sub_questions: list[str],
        evidence: list[Evidence],
        reasoning_path: list[ReasoningStep],
        grounding_result: dict[str, Any],
    ) -> SynthesizedAnswer:
        """
        Synthesize final answer with full provenance tracking and chain validation.

        This is the main answer synthesis method that:
        1. Validates the reasoning chain
        2. Builds evidence provenance
        3. Generates answer with classification
        4. Provides alternative answers if uncertain

        Args:
            question: Original question
            sub_questions: Decomposed sub-questions
            evidence: Collected evidence
            reasoning_path: Chain of reasoning steps
            grounding_result: Result from grounded reasoning

        Returns:
            SynthesizedAnswer with full provenance
        """
        # Step 1: Validate reasoning chain
        chain_validation = self.validate_reasoning_chain(
            reasoning_path=reasoning_path,
            evidence=evidence,
            sub_questions=sub_questions,
        )

        # Step 2: Build provenance
        provenance = self.build_evidence_provenance(
            evidence=evidence,
            reasoning_path=reasoning_path,
            grounding_result=grounding_result,
        )

        # Step 3: Determine hallucination risk
        hallucination_risk = self.determine_hallucination_risk(grounding_result, evidence)

        # Step 4: Prepare inputs for LLM synthesis
        step_answers = []
        for step in reasoning_path:
            step_answers.append({
                "step": step.step_number,
                "action": step.action.value,
                "thought": step.thought,
                "observation": step.observation,
                "confidence": 0.5 + step.sufficiency_delta,
            })

        evidence_chain_str = []
        for ev in sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:10]:
            ev_str = {
                "id": ev.id,
                "content": ev.content[:200],
                "type": ev.evidence_type.value,
                "relevance": ev.relevance_score,
                "hop": ev.hop_index if hasattr(ev, "hop_index") else 0,
            }
            evidence_chain_str.append(ev_str)

        try:
            # Call LLM for synthesis
            result = await self._answer_synthesis_chain.ainvoke({
                "question": question,
                "sub_questions": sub_questions or ["(No sub-questions)"],
                "step_answers": step_answers,
                "evidence_chain": evidence_chain_str,
                "grounding_source": provenance.grounding_source.value.upper(),
            })

            # Parse LLM result
            final_answer = result.get("final_answer", grounding_result.get("answer", "Unable to synthesize answer"))
            overall_confidence = float(result.get("overall_confidence", grounding_result.get("confidence", 0.5)))

            # Determine classification from confidence
            answer_classification = SynthesizedAnswer.from_classification_score(overall_confidence)

            # Build answer components
            answer_components = AnswerComponents(
                direct_answer=result.get("answer_components", {}).get("direct_answer", final_answer[:100]),
                supporting_facts=result.get("answer_components", {}).get("supporting_facts", []),
                inferred_facts=result.get("answer_components", {}).get("inferred_facts", []),
                caveats=result.get("answer_components", {}).get("caveats", grounding_result.get("disclaimers", [])),
            )

            # Build alternative answers if uncertain
            alternative_answers = []
            if answer_classification in [AnswerClassification.UNCERTAIN, AnswerClassification.INSUFFICIENT]:
                for alt in result.get("alternative_answers", []):
                    alternative_answers.append(AlternativeAnswer(
                        answer=alt.get("answer", ""),
                        confidence=float(alt.get("confidence", 0.0)),
                        reason=alt.get("reason", ""),
                    ))

            # Missing information
            missing_info = result.get("missing_information", [])
            if chain_validation.status != ChainValidationStatus.VALID:
                missing_info.extend(chain_validation.issues)

            logger.info(
                "Answer synthesized with provenance",
                answer_classification=answer_classification.value,
                overall_confidence=overall_confidence,
                chain_status=chain_validation.status.value,
                hallucination_risk=hallucination_risk.value,
                primary_sources_count=len(provenance.primary_sources),
                alternative_count=len(alternative_answers),
            )

            return SynthesizedAnswer(
                final_answer=final_answer,
                answer_classification=answer_classification,
                overall_confidence=overall_confidence,
                chain_validation=chain_validation,
                provenance=provenance,
                answer_components=answer_components,
                alternative_answers=alternative_answers,
                missing_information=missing_info,
                hallucination_risk=hallucination_risk,
            )

        except Exception as e:
            logger.error("Answer synthesis with provenance failed", error=str(e))

            # Fallback to basic synthesis
            answer = grounding_result.get("answer", "Unable to generate answer")
            confidence = float(grounding_result.get("confidence", 0.3))

            return SynthesizedAnswer(
                final_answer=answer,
                answer_classification=SynthesizedAnswer.from_classification_score(confidence),
                overall_confidence=confidence,
                chain_validation=chain_validation,
                provenance=provenance,
                answer_components=AnswerComponents(
                    direct_answer=answer[:100],
                    caveats=["Synthesis failed, using fallback answer"],
                ),
                missing_information=["Full synthesis failed due to error"],
                hallucination_risk=hallucination_risk,
            )

    async def explain_reasoning(
        self,
        question: str,
        answer: str,
        reasoning_path: list[ReasoningStep],
        evidence: list[Evidence],
    ) -> str:
        """
        Generate a detailed explanation of the reasoning process.

        Args:
            question: Original question
            answer: Generated answer
            reasoning_path: Chain of reasoning steps
            evidence: Evidence used

        Returns:
            Human-readable explanation
        """
        if not self._include_explanation:
            return ""

        reasoning_str = format_reasoning_path_for_prompt(
            [
                {
                    "step_number": s.step_number,
                    "action": s.action.value,
                    "thought": s.thought,
                    "observation": s.observation,
                }
                for s in reasoning_path
            ]
        )

        evidence_str = format_evidence_for_prompt(
            [
                {
                    "evidence_type": e.evidence_type.value,
                    "content": e.content[:150],
                    "relevance_score": e.relevance_score,
                }
                for e in sorted(evidence, key=lambda x: x.relevance_score, reverse=True)[:10]
            ]
        )

        try:
            explanation = await self._explanation_chain.ainvoke(
                {
                    "question": question,
                    "answer": answer,
                    "reasoning_path": reasoning_str,
                    "evidence": evidence_str,
                }
            )

            return explanation

        except Exception as e:
            logger.warning("Explanation generation failed", error=str(e))
            return self._generate_fallback_explanation(reasoning_path, evidence)

    def calculate_final_confidence(
        self,
        synthesis_confidence: float,
        answer_confidence: float,
        evidence: list[Evidence],
        reasoning_path: list[ReasoningStep],
        question_type: str = QuestionType.FACTOID.value,
        path_completeness: float = 0.0,
        subgraph: SubGraph | None = None,
    ) -> float:
        """
        Calculate final confidence score combining multiple factors.

        Enhanced with multi-hop path completeness for better HotPotQA performance.

        Args:
            synthesis_confidence: Confidence from synthesis
            answer_confidence: Confidence from answer generation
            evidence: Evidence used
            reasoning_path: Reasoning steps
            question_type: Type of question
            path_completeness: Path completeness score for multi-hop
            subgraph: Current reasoning subgraph

        Returns:
            Final confidence score (0-1)
        """
        # Check if this is a multi-hop question
        is_multihop = question_type in [
            QuestionType.MULTIHOP.value,
            QuestionType.BRIDGE.value,
            QuestionType.COMPARISON.value,
        ]

        # Base confidence from synthesis and answer
        base_confidence = (synthesis_confidence + answer_confidence) / 2

        # Evidence quality factor
        if evidence:
            direct_evidence = [e for e in evidence if e.evidence_type == EvidenceType.DIRECT]
            path_evidence = [e for e in evidence if e.evidence_type == EvidenceType.PATH]

            # For multi-hop, path evidence is highly valuable
            if is_multihop:
                path_weight = len(path_evidence) / len(evidence) if path_evidence else 0
                direct_weight = len(direct_evidence) / len(evidence)
                evidence_quality = (path_weight * 0.6) + (direct_weight * 0.4)
            else:
                evidence_quality = len(direct_evidence) / len(evidence) if evidence else 0

            avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)

            # Calculate hop coverage for multi-hop questions
            if is_multihop and path_evidence:
                hop_indices = {e.hop_index for e in evidence if hasattr(e, "hop_index") and e.hop_index is not None}
                max_hop = max(hop_indices) if hop_indices else 0
                hop_coverage = len(hop_indices) / (max_hop + 1) if max_hop > 0 else 0.5
            else:
                hop_coverage = 1.0  # Not penalized for simple questions
        else:
            evidence_quality = 0
            avg_relevance = 0
            hop_coverage = 0

        # Reasoning quality factor (more steps with positive progress = better)
        if reasoning_path:
            positive_steps = sum(1 for s in reasoning_path if s.sufficiency_delta > 0)
            reasoning_quality = positive_steps / len(reasoning_path)
        else:
            reasoning_quality = 0.5

        # Subgraph quality for multi-hop questions
        subgraph_quality = 1.0
        if is_multihop and subgraph:
            metrics = subgraph.get_subgraph_metrics()
            # Penalize disconnected subgraphs
            if not metrics.get("is_connected", True):
                subgraph_quality *= 0.8
            # Bonus for bridge entities in bridge questions
            if question_type == QuestionType.BRIDGE.value:
                bridge_count = metrics.get("bridge_entity_count", 0)
                if bridge_count > 0:
                    subgraph_quality *= 1.1  # 10% bonus
            # Consider path length adequacy
            max_path = metrics.get("max_path_length", 0)
            if max_path >= 2:
                subgraph_quality *= 1.05  # 5% bonus for 2+ hop paths

        # Adjust weights based on question type
        if is_multihop:
            # For multi-hop, path completeness and hop coverage matter more
            final_confidence = (
                base_confidence * 0.35
                + evidence_quality * 0.15
                + avg_relevance * 0.15
                + path_completeness * 0.15
                + hop_coverage * 0.10
                + reasoning_quality * 0.05
                + (subgraph_quality - 1.0) * 0.05  # Adjustment factor
            )
        else:
            # For simple questions, traditional factors
            final_confidence = (
                base_confidence * 0.50
                + evidence_quality * 0.20
                + avg_relevance * 0.20
                + reasoning_quality * 0.10
            )

        # Apply minimum threshold for multi-hop without path evidence
        if is_multihop and len([e for e in evidence if e.evidence_type == EvidenceType.PATH]) == 0:
            # Cap confidence if no paths found for multi-hop questions
            final_confidence = min(final_confidence, 0.65)

        return max(0.0, min(1.0, final_confidence))

    def select_key_evidence(
        self,
        evidence: list[Evidence],
        max_items: int | None = None,
        question_type: str = QuestionType.FACTOID.value,
    ) -> list[Evidence]:
        """
        Select the most important evidence to include in response.

        Enhanced to prioritize PATH evidence for multi-hop questions.

        Args:
            evidence: All evidence
            max_items: Maximum items to select
            question_type: Type of question for priority ordering

        Returns:
            Selected key evidence
        """
        max_items = max_items or self._max_evidence

        # Check if multi-hop question
        is_multihop = question_type in [
            QuestionType.MULTIHOP.value,
            QuestionType.BRIDGE.value,
            QuestionType.COMPARISON.value,
        ]

        # Sort by relevance
        sorted_evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)

        # Prefer diverse evidence types
        selected: list[Evidence] = []
        by_type: dict[EvidenceType, list[Evidence]] = {}

        for ev in sorted_evidence:
            if ev.evidence_type not in by_type:
                by_type[ev.evidence_type] = []
            by_type[ev.evidence_type].append(ev)

        # Determine priority order based on question type
        if is_multihop:
            # For multi-hop, prioritize PATH evidence
            priority_order = [
                EvidenceType.PATH,
                EvidenceType.DIRECT,
                EvidenceType.INFERRED,
                EvidenceType.CONTEXTUAL,
                EvidenceType.COMMUNITY,
            ]
        else:
            priority_order = [
                EvidenceType.DIRECT,
                EvidenceType.INFERRED,
                EvidenceType.PATH,
                EvidenceType.CONTEXTUAL,
                EvidenceType.COMMUNITY,
            ]

        # Round-robin selection from each type with priority
        while len(selected) < max_items:
            added = False
            for ev_type in priority_order:
                if ev_type in by_type and by_type[ev_type]:
                    selected.append(by_type[ev_type].pop(0))
                    added = True
                    if len(selected) >= max_items:
                        break
            if not added:
                break

        return selected

    async def respond(self, state: MACERState) -> dict[str, Any]:
        """
        Main response method for MACER pipeline.

        Generates the final answer with explanation.
        Enhanced with:
        - Multi-hop reasoning support
        - Advanced multi-layer response synthesis (when enabled)
        - Deep graph analysis and pattern discovery
        - Bilingual (Korean/English) responses

        Args:
            state: Current MACER state

        Returns:
            State updates with final answer and explanation
        """
        question = state.get("original_query", "")
        evidence = state.get("evidence", [])
        reasoning_path = state.get("reasoning_path", [])
        retrieved_entities = state.get("retrieved_entities", [])
        subgraph = state.get("current_subgraph", SubGraph())
        errors = state.get("errors", [])
        metadata = state.get("metadata", {})
        question_type = state.get("question_type", QuestionType.FACTOID.value)
        path_completeness_from_state = state.get("path_completeness", 0.0)

        # Check if multi-hop question
        is_multihop = question_type in [
            QuestionType.MULTIHOP.value,
            QuestionType.BRIDGE.value,
            QuestionType.COMPARISON.value,
        ]

        # Calculate path completeness if not provided
        if path_completeness_from_state == 0.0 and is_multihop:
            path_completeness = calculate_path_completeness(evidence, subgraph, question_type)
        else:
            path_completeness = path_completeness_from_state

        # Count path evidence
        path_evidence_count = sum(1 for e in evidence if e.evidence_type == EvidenceType.PATH)

        logger.info(
            "Responser agent starting",
            question=question[:50],
            question_type=question_type,
            evidence_count=len(evidence),
            path_evidence_count=path_evidence_count,
            path_completeness=path_completeness,
            retrieved_entities=len(retrieved_entities),
            subgraph_nodes=subgraph.node_count() if subgraph else 0,
            errors_count=len(errors),
            has_metadata_no_data=metadata.get("no_data", False),
            use_advanced_synthesis=self._use_advanced_synthesis,
        )

        # Handle NO_DATA scenario (knowledge graph is empty)
        if metadata.get("no_data", False):
            logger.warning("NO_DATA scenario - knowledge graph is empty")
            return self._generate_no_data_response(
                question=question,
                errors=errors,
                metadata=metadata,
                scenario="EMPTY_GRAPH",
            )

        # Handle case where no entities or evidence were found
        if not evidence and not retrieved_entities:
            logger.warning("No evidence or entities found - returning informative response")
            return self._generate_no_data_response(
                question=question,
                errors=errors,
                metadata=metadata,
                scenario="NO_MATCH",
            )

        # Quality-based NO_DATA detection (even if evidence exists)
        if evidence:
            max_relevance = max(e.relevance_score for e in evidence)
            fuzzy_match_only = all(
                r.get("source") == "fuzzy_match" for r in retrieved_entities
            ) if retrieved_entities else False

            # All evidence has very low relevance score
            if max_relevance < 0.3:
                logger.warning(
                    "All evidence has low relevance - treating as NO_DATA",
                    max_relevance=max_relevance,
                    evidence_count=len(evidence),
                )
                return self._generate_no_data_response(
                    question=question,
                    errors=errors,
                    metadata={**metadata, "quality_check_failed": True, "max_relevance": max_relevance},
                    scenario="LOW_QUALITY",
                )

            # Only fuzzy matches with low relevance - likely wrong entities matched
            if fuzzy_match_only and max_relevance < 0.5:
                logger.warning(
                    "Only fuzzy matches with low relevance - likely incorrect entity matches",
                    max_relevance=max_relevance,
                    fuzzy_match_count=len(retrieved_entities),
                )
                return self._generate_no_data_response(
                    question=question,
                    errors=errors,
                    metadata={**metadata, "fuzzy_only": True, "max_relevance": max_relevance},
                    scenario="FUZZY_LOW_QUALITY",
                )

        # =====================================================================
        # Question-Type Routing for Response Synthesis
        # =====================================================================
        # FACTOID/YESNO questions: Skip advanced synthesis, use direct answer path
        # Complex questions: Use advanced synthesis with appropriate depth
        should_use_advanced = (
            self._use_advanced_synthesis
            and self._advanced_synthesizer
            and question_type not in SKIP_ADVANCED_SYNTHESIS_TYPES
        )

        # For FACTOID/YESNO questions, use direct answer synthesis path
        if question_type in SKIP_ADVANCED_SYNTHESIS_TYPES:
            logger.info(
                "Using direct answer synthesis for simple question type",
                question_type=question_type,
                evidence_count=len(evidence),
            )
            try:
                # Generate grounded answer directly
                grounded_result = await self.generate_grounded_answer(
                    question, evidence, reasoning_path
                )

                # Extract concise benchmark-style direct answer
                direct_answer = await self.extract_benchmark_direct_answer(
                    question=question,
                    full_answer=grounded_result.get("answer", ""),
                )

                # Calculate confidence
                final_confidence = self.calculate_final_confidence(
                    synthesis_confidence=grounded_result.get("confidence", 0.5),
                    answer_confidence=grounded_result.get("confidence", 0.5),
                    evidence=evidence,
                    reasoning_path=reasoning_path,
                    question_type=question_type,
                    path_completeness=path_completeness,
                    subgraph=subgraph,
                )

                logger.info(
                    "Direct answer synthesis completed",
                    question_type=question_type,
                    direct_answer=direct_answer,
                    confidence=final_confidence,
                )

                return {
                    "final_answer": direct_answer or grounded_result.get("answer", ""),
                    "direct_answer": direct_answer,
                    "confidence": final_confidence,
                    "answer_type": "DIRECT" if final_confidence > 0.6 else "INFERRED",
                    "answer_source": grounded_result.get("source", "GRAPH_PRIMARY"),
                    "answer_classification": AnswerClassification.CONFIDENT.value if final_confidence > 0.7 else AnswerClassification.PROBABLE.value,
                    "hallucination_risk": HallucinationRisk.LOW.value if grounded_result.get("source") == "GRAPH_PRIMARY" else HallucinationRisk.MEDIUM.value,
                    "explanation": grounded_result.get("reasoning", ""),
                    "supporting_evidence": [
                        {
                            "content": e.content[:300],
                            "type": e.evidence_type.value,
                            "score": e.relevance_score,
                        }
                        for e in sorted(evidence, key=lambda x: x.relevance_score, reverse=True)[:5]
                    ],
                    "caveats": grounded_result.get("caveats", []),
                    "metadata": {
                        **(state.get("metadata", {})),
                        "question_type": question_type,
                        "path_completeness": path_completeness,
                        "synthesis_mode": "direct",  # Indicates brief/direct synthesis
                    },
                }
            except Exception as e:
                logger.warning(
                    "Direct synthesis failed, falling back to standard synthesis",
                    error=str(e),
                )
                # Fall through to standard synthesis below

        # =====================================================================
        # Use Advanced Synthesizer for Rich, Multi-Layer Responses
        # =====================================================================
        if should_use_advanced:
            try:
                # Use question-type appropriate depth
                effective_depth = QUESTION_TYPE_RESPONSE_DEPTH.get(question_type, self._response_depth)
                advanced_response = await self._advanced_synthesizer.synthesize_advanced_response(
                    state=state,
                    depth=effective_depth,
                )

                # Format the response
                formatted_answer = self._advanced_synthesizer.format_response(
                    advanced_response,
                    include_korean=self._include_korean,
                )

                # Calculate final confidence with multi-hop factors
                final_confidence = self.calculate_final_confidence(
                    synthesis_confidence=advanced_response.confidence,
                    answer_confidence=advanced_response.confidence,
                    evidence=evidence,
                    reasoning_path=reasoning_path,
                    question_type=question_type,
                    path_completeness=path_completeness,
                    subgraph=subgraph,
                )

                # Extract benchmark-style direct answer from advanced synthesis
                advanced_direct_answer = await self.extract_benchmark_direct_answer(
                    question=question,
                    full_answer=formatted_answer,
                )

                logger.info(
                    "Advanced synthesis completed",
                    confidence=final_confidence,
                    direct_answer=advanced_direct_answer,
                    sections_count=sum([
                        1 for s in [
                            advanced_response.definition_section,
                            advanced_response.architecture_section,
                            advanced_response.component_section,
                            advanced_response.relationship_section,
                        ] if s is not None
                    ]),
                    clusters=len(advanced_response.entity_clusters),
                    patterns=len(advanced_response.pattern_insights),
                    chains=len(advanced_response.relationship_chains),
                )

                return {
                    "final_answer": formatted_answer,
                    "direct_answer": advanced_direct_answer,  # Benchmark-style concise answer
                    "confidence": final_confidence,
                    "answer_type": "ADVANCED_SYNTHESIS",
                    "answer_source": "GRAPH_PRIMARY",
                    "answer_classification": AnswerClassification.CONFIDENT.value if final_confidence > 0.7 else AnswerClassification.PROBABLE.value,
                    "hallucination_risk": HallucinationRisk.LOW.value,
                    "explanation": f"Generated using advanced multi-layer synthesis with {len(advanced_response.entity_clusters)} entity clusters, {len(advanced_response.relationship_chains)} relationship chains, and {len(advanced_response.pattern_insights)} discovered patterns.",
                    "supporting_evidence": [
                        {
                            "content": e.content[:300],
                            "type": e.evidence_type.value,
                            "score": e.relevance_score,
                        }
                        for e in sorted(evidence, key=lambda x: x.relevance_score, reverse=True)[:10]
                    ],
                    "caveats": [],
                    "advanced_synthesis": {
                        "entity_clusters": [
                            {"name": c.name, "entities": c.entities[:5], "type": c.cluster_type}
                            for c in advanced_response.entity_clusters[:5]
                        ],
                        "relationship_chains": [
                            {"description": c.description, "confidence": c.confidence}
                            for c in advanced_response.relationship_chains[:5]
                        ],
                        "pattern_insights": [
                            {"type": p.pattern_type, "description": p.description}
                            for p in advanced_response.pattern_insights[:5]
                        ],
                        "response_depth": advanced_response.response_depth.value,
                    },
                    "metadata": {
                        **(state.get("metadata", {})),
                        "question_type": question_type,
                        "path_completeness": path_completeness,
                        "is_multihop": is_multihop,
                        "synthesis_mode": "advanced",
                    },
                }

            except Exception as e:
                logger.warning(
                    "Advanced synthesis failed, falling back to standard synthesis",
                    error=str(e),
                )
                # Fall through to standard synthesis

        # Step 1: Generate grounded answer (Graph DB priority, LLM fallback)
        grounded_result = await self.generate_grounded_answer(question, evidence, reasoning_path)

        # Step 2: Synthesize answer with full provenance tracking
        sub_questions = state.get("sub_questions", [])
        synthesized = await self.synthesize_answer_with_provenance(
            question=question,
            sub_questions=sub_questions,
            evidence=evidence,
            reasoning_path=reasoning_path,
            grounding_result=grounded_result,
        )

        # Step 3: Calculate confidence (with multi-hop factors and grounding)
        answer_source = grounded_result.get("answer_source", "UNKNOWN")
        grounding_breakdown = grounded_result.get("grounding_breakdown", {})

        # Adjust confidence based on answer source
        source_multiplier = {
            "GRAPH_ONLY": 1.0,      # Full confidence for graph-only answers
            "GRAPH_PRIMARY": 0.95,   # Slight reduction for graph-primary
            "LLM_SUPPLEMENTED": 0.85,  # More reduction for LLM-supplemented
            "LLM_ONLY": 0.7,        # Significant reduction for LLM-only
        }.get(answer_source, 0.8)

        final_confidence = self.calculate_final_confidence(
            synthesis_confidence=synthesized.overall_confidence,
            answer_confidence=float(grounded_result.get("confidence", 0.5)) * source_multiplier,
            evidence=evidence,
            reasoning_path=reasoning_path,
            question_type=question_type,
            path_completeness=path_completeness,
            subgraph=subgraph,
        )

        # Step 4: Generate explanation
        explanation = await self.explain_reasoning(
            question=question,
            answer=synthesized.final_answer,
            reasoning_path=reasoning_path,
            evidence=evidence,
        )

        # Step 5: Select key evidence for response (with question type awareness)
        key_evidence = self.select_key_evidence(evidence, question_type=question_type)

        # Step 6: Extract benchmark-style direct answer
        direct_answer = synthesized.answer_components.direct_answer
        # If direct answer is too verbose (>10 words), re-extract
        if not direct_answer or len(direct_answer.split()) > 10:
            direct_answer = await self.extract_benchmark_direct_answer(
                question=question,
                full_answer=synthesized.final_answer,
            )
            # Update the answer components with the concise version
            synthesized.answer_components.direct_answer = direct_answer

        # Determine answer_type based on grounding and classification
        if synthesized.answer_classification == AnswerClassification.CONFIDENT:
            answer_type = "DIRECT" if answer_source == "GRAPH_ONLY" else "GROUNDED"
        elif synthesized.answer_classification == AnswerClassification.PROBABLE:
            answer_type = "GROUNDED" if answer_source in ["GRAPH_ONLY", "GRAPH_PRIMARY"] else "SUPPLEMENTED"
        elif synthesized.answer_classification == AnswerClassification.UNCERTAIN:
            answer_type = "UNCERTAIN"
        else:
            answer_type = "INSUFFICIENT"

        logger.info(
            "Responser completed with provenance",
            answer_type=answer_type,
            answer_source=answer_source,
            answer_classification=synthesized.answer_classification.value,
            hallucination_risk=synthesized.hallucination_risk.value,
            chain_validation_status=synthesized.chain_validation.status.value,
            confidence=final_confidence,
            question_type=question_type,
            is_multihop=is_multihop,
            path_evidence_used=sum(1 for e in key_evidence if e.evidence_type == EvidenceType.PATH),
            primary_sources=len(synthesized.provenance.primary_sources),
            alternative_answers=len(synthesized.alternative_answers),
        )

        return {
            "final_answer": synthesized.final_answer,
            "direct_answer": direct_answer,  # Benchmark-style concise answer
            "confidence": final_confidence,
            "answer_type": answer_type,
            "answer_source": answer_source,
            "answer_classification": synthesized.answer_classification.value,
            "hallucination_risk": synthesized.hallucination_risk.value,
            "explanation": explanation,
            "supporting_evidence": [
                {
                    "content": e.content,
                    "type": e.evidence_type.value,
                    "score": e.relevance_score,
                    "hop_index": e.hop_index if hasattr(e, "hop_index") else 0,
                    "path_length": e.path_length if hasattr(e, "path_length") else 0,
                    "confidence_type": e.confidence_type.value if hasattr(e, "confidence_type") else "unknown",
                }
                for e in key_evidence
            ],
            "caveats": synthesized.answer_components.caveats,
            "grounding_breakdown": grounding_breakdown,
            # Provenance tracking
            "chain_validation": {
                "status": synthesized.chain_validation.status.value,
                "steps_validated": synthesized.chain_validation.steps_validated,
                "steps_total": synthesized.chain_validation.steps_total,
                "issues": synthesized.chain_validation.issues,
            },
            "provenance": {
                "primary_sources": synthesized.provenance.primary_sources,
                "supporting_sources": synthesized.provenance.supporting_sources,
                "grounding_source": synthesized.provenance.grounding_source.value,
                "evidence_chain_count": len(synthesized.provenance.evidence_chain),
            },
            "answer_components": {
                "direct_answer": synthesized.answer_components.direct_answer,
                "supporting_facts": synthesized.answer_components.supporting_facts,
                "inferred_facts": synthesized.answer_components.inferred_facts,
            },
            "alternative_answers": [
                {
                    "answer": alt.answer,
                    "confidence": alt.confidence,
                    "reason": alt.reason,
                }
                for alt in synthesized.alternative_answers
            ],
            "missing_information": synthesized.missing_information,
            "metadata": {
                **(state.get("metadata", {})),
                "question_type": question_type,
                "path_completeness": path_completeness,
                "is_multihop": is_multihop,
                "grounding_confidence": grounded_result.get("confidence_breakdown", {}),
            },
        }

    def _format_synthesis(self, synthesis: dict[str, Any]) -> str:
        """Format synthesis result for prompt."""
        lines = []

        facts = synthesis.get("synthesized_facts", [])
        if facts:
            lines.append("## Established Facts")
            for fact in facts[:10]:
                lines.append(f"- {fact}")

        inferences = synthesis.get("inferences", [])
        if inferences:
            lines.append("\n## Inferences")
            for inference in inferences[:5]:
                lines.append(f"- {inference}")

        contradictions = synthesis.get("contradictions", [])
        if contradictions:
            lines.append("\n## Contradictions/Uncertainties")
            for contradiction in contradictions[:3]:
                lines.append(f"- {contradiction}")

        confidence = synthesis.get("answer_confidence", 0.5)
        lines.append(f"\n## Confidence Level: {confidence:.0%}")

        return "\n".join(lines)

    def _generate_fallback_explanation(
        self,
        reasoning_path: list[ReasoningStep],
        evidence: list[Evidence],
    ) -> str:
        """Generate a basic explanation when LLM fails."""
        lines = ["## Reasoning Process\n"]

        if reasoning_path:
            lines.append(f"The answer was derived through {len(reasoning_path)} reasoning steps:\n")
            for step in reasoning_path[-5:]:
                lines.append(f"{step.step_number}. [{step.action.value}] {step.thought}")

        if evidence:
            lines.append("\n## Evidence Used\n")
            lines.append(f"The answer is based on {len(evidence)} pieces of evidence.")
            top_evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:3]
            for ev in top_evidence:
                lines.append(f"- {ev.content[:100]}...")

        return "\n".join(lines)

    def _generate_no_data_response(
        self,
        question: str,
        errors: list[str],
        metadata: dict[str, Any] | None = None,
        scenario: str = "NO_MATCH",
    ) -> dict[str, Any]:
        """
        Generate an informative response when no data is found in knowledge graph.

        Args:
            question: Original question
            errors: List of errors encountered
            metadata: Additional metadata from the pipeline
            scenario: Type of no-data scenario (EMPTY_GRAPH, NO_MATCH, LOW_QUALITY, FUZZY_LOW_QUALITY)

        Returns:
            Response indicating no data was found
        """
        metadata = metadata or {}
        diagnostics_summary = metadata.get("diagnostics_summary", {})

        # Construct scenario-specific answer
        if scenario == "EMPTY_GRAPH":
            answer_parts = [
                f"'{question[:100]}'에 대한 정보를 지식 그래프에서 찾을 수 없습니다.",
                "",
                "**원인**: 지식 그래프가 비어 있습니다.",
                "",
                "다음 단계를 수행해주세요:",
                "1. 'Data Ingestion' 기능을 사용하여 관련 문서를 업로드하세요",
                "2. 문서가 처리되면 다시 질문해주세요",
                "",
                "---",
                f"I wasn't able to find relevant information about '{question[:100]}' in the knowledge graph.",
                "",
                "**Reason**: The knowledge graph is empty.",
                "",
                "Please:",
                "1. Upload relevant documents using 'Data Ingestion'",
                "2. After processing, try your question again",
            ]
            answer_type = "NO_DATA"
            caveats = [
                "Knowledge graph is empty - please ingest documents first",
            ]
        elif scenario in ("LOW_QUALITY", "FUZZY_LOW_QUALITY"):
            # Quality-based NO_DATA - evidence exists but relevance is too low
            max_relevance = metadata.get("max_relevance", 0)

            if scenario == "FUZZY_LOW_QUALITY":
                answer_parts = [
                    f"'{question[:100]}'에 대한 관련성 높은 정보를 찾지 못했습니다.",
                    "",
                    "**원인**: 유사한 이름의 엔티티를 찾았으나, 질문과 직접적으로 관련이 없습니다.",
                    "",
                    "검색 결과가 질문의 의도와 맞지 않을 수 있습니다.",
                    "다른 키워드로 검색하거나, 해당 주제의 문서를 업로드해주세요.",
                    "",
                    "---",
                    f"I couldn't find highly relevant information for '{question[:100]}'.",
                    "",
                    "**Reason**: Found entities with similar names, but they don't appear to be directly related to your question.",
                    "",
                    "The search results may not match your intent.",
                    "Try different keywords or upload documents about this specific topic.",
                ]
                caveats = [
                    "Fuzzy matches found but low relevance - possibly incorrect entities",
                    "Try more specific search terms",
                ]
            else:  # LOW_QUALITY
                answer_parts = [
                    f"'{question[:100]}'에 대한 정보를 검색했으나, 충분히 관련성 있는 결과를 찾지 못했습니다.",
                    "",
                    f"**검색 결과 품질**: 최대 관련도 점수 {max_relevance:.1%} (최소 30% 필요)",
                    "",
                    "더 관련성 높은 결과를 위해:",
                    "- 해당 주제를 다루는 문서를 업로드하세요",
                    "- 더 구체적인 용어로 질문하세요",
                    "",
                    "---",
                    f"I searched for information about '{question[:100]}', but found no sufficiently relevant results.",
                    "",
                    f"**Search quality**: Maximum relevance score {max_relevance:.1%} (minimum 30% required)",
                    "",
                    "For more relevant results:",
                    "- Upload documents covering this topic",
                    "- Try more specific terms",
                ]
                caveats = [
                    f"Low relevance evidence (max: {max_relevance:.1%})",
                    "Query may need different keywords",
                ]

            answer_type = "NO_DATA"
        else:  # NO_MATCH
            entity_count = diagnostics_summary.get("entity_count", 0)
            has_embeddings = diagnostics_summary.get("has_embeddings", False)

            answer_parts = [
                f"'{question[:100]}'에 대한 관련 정보를 찾지 못했습니다.",
                "",
                "가능한 원인:",
                "1. 지식 그래프에 해당 주제에 대한 정보가 없습니다",
                "2. 검색어가 기존 엔티티와 일치하지 않습니다",
                "3. 해당 주제를 다루는 문서가 아직 업로드되지 않았습니다",
                "",
                "해결 방법:",
                "- 관련 문서를 업로드하세요",
                "- 다른 키워드로 질문을 다시 해보세요",
                "- 더 구체적인 용어를 사용해보세요",
                "",
                "---",
                f"I wasn't able to find relevant information about '{question[:100]}' in the knowledge graph.",
                "",
                "This could be because:",
                "1. The knowledge graph doesn't contain information about this topic",
                "2. The search terms couldn't be matched to existing entities",
                "3. No documents covering this topic have been ingested",
                "",
                "To get better results:",
                "- Upload documents containing information about this topic",
                "- Try rephrasing your question with different keywords",
                "- Use more specific terms",
            ]

            if entity_count > 0:
                answer_parts.extend([
                    "",
                    f"참고: 지식 그래프에 {entity_count}개의 엔티티가 있지만 ",
                    "질문과 관련된 엔티티를 찾지 못했습니다.",
                    f"(Note: The knowledge graph has {entity_count} entities, ",
                    "but none matched your query.)",
                ])

            if not has_embeddings and entity_count > 0:
                answer_parts.extend([
                    "",
                    "⚠️ 벡터 임베딩이 없어 의미 검색이 제한됩니다.",
                    "(Warning: No vector embeddings - semantic search is limited.)",
                ])

            answer_type = "NO_DATA"
            caveats = [
                "No relevant data found in knowledge graph",
                "Try rephrasing your question",
            ]

        explanation_parts = [
            "## 검색 과정 (Search Process)",
            "",
            "시스템이 수행한 단계:",
            "1. 질문에서 주제 엔티티 추출 (Topic entity extraction)",
            "2. 지식 그래프에서 매칭 엔티티 검색 (Entity matching)",
            "3. 청크 및 관계에서 관련 증거 검색 (Evidence retrieval)",
            "",
            "결과: 매칭되는 엔티티나 관련 증거를 찾지 못했습니다.",
            "(Result: No matching entities or relevant evidence found.)",
        ]

        if errors:
            explanation_parts.extend([
                "",
                "## 발생한 오류 (Errors Encountered)",
                *[f"- {e}" for e in errors[:5]],
            ])

        return {
            "final_answer": "\n".join(answer_parts),
            "confidence": 0.1,
            "answer_type": answer_type,
            "explanation": "\n".join(explanation_parts),
            "supporting_evidence": [],
            "caveats": caveats,
            "metadata": {
                **metadata,
                "no_data_reason": scenario,
                "errors": errors,
            },
        }

"""
MACER Agent Prompts for Think-on-Graph 3.0.

LLM prompt templates for all MACER agents.
"""

from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# Constructor Agent Prompts
# =============================================================================

TOPIC_ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at identifying topic entities in questions for knowledge graph querying.
You must handle questions in any language (English, Korean, etc.) and extract entities accurately.

Given a question, extract the main topic entities that should be searched for in a knowledge graph.

## Guidelines
1. Identify named entities (people, organizations, places, products, concepts, technical terms)
2. Distinguish between topic entities (what we're asking about) and property entities (attributes we want)
3. For each entity, estimate how central it is to the question
4. Consider synonyms, alternative names, and translations
5. For Korean questions: Extract Korean entity names as-is, but also provide English equivalents if commonly known
6. For technical terms: Include both the term and common variations

## Entity Types
- PERSON: People, characters, historical figures
- ORG: Organizations, companies, institutions
- LOC: Locations, places, geographic entities
- CONCEPT: Abstract concepts, ideas, methodologies
- TECH: Technologies, tools, frameworks, programming concepts
- PRODUCT: Products, services, software
- EVENT: Events, incidents, historical moments
- TERM: Domain-specific terminology

## Output Format
Return a JSON object with:
{{
    "topic_entities": [
        {{"name": "entity name (in original language)", "type": "PERSON/ORG/LOC/CONCEPT/TECH/PRODUCT/EVENT/TERM", "is_primary": true/false, "aliases": ["alternative names", "translations"]}}
    ],
    "property_focus": ["property1", "property2"],
    "question_type": "factoid/relationship/comparison/aggregation/exploratory/definition"
}}

## Examples
Question: "레드팀이 뭐야?"
Output: {{"topic_entities": [{{"name": "레드팀", "type": "CONCEPT", "is_primary": true, "aliases": ["Red Team", "레드 팀", "red teaming"]}}], "property_focus": ["definition", "purpose"], "question_type": "definition"}}

Question: "What is machine learning?"
Output: {{"topic_entities": [{{"name": "machine learning", "type": "TECH", "is_primary": true, "aliases": ["ML", "기계학습", "머신러닝"]}}], "property_focus": ["definition", "applications"], "question_type": "definition"}}""",
        ),
        (
            "human",
            """Question: {question}

Extract the topic entities:""",
        ),
    ]
)

SEED_SUBGRAPH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are analyzing a seed subgraph to understand its structure and relevance to a question.

Given the question and the initial subgraph, provide:
1. Assessment of how relevant each node is
2. Which relationships are most important
3. What additional information might be needed

## Subgraph
{subgraph}

## Output Format
Return a JSON object with:
{{
    "relevant_nodes": ["node_id1", "node_id2"],
    "key_relationships": ["rel_description1", "rel_description2"],
    "missing_information": ["what1", "what2"],
    "initial_assessment": "brief assessment of subgraph quality"
}}""",
        ),
        (
            "human",
            """Question: {question}

Analyze the seed subgraph:""",
        ),
    ]
)

# =============================================================================
# Retriever Agent Prompts
# =============================================================================

RETRIEVAL_STRATEGY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at determining the best retrieval strategy for knowledge graph queries.

Given a question and current context, decide:
1. Whether to use vector similarity search
2. Whether to traverse graph relationships
3. Whether to search community summaries
4. How deep to search (number of hops)

## Current Context
Question: {question}
Current Subgraph Size: {subgraph_size} nodes
Current Evidence Count: {evidence_count}
Iteration: {iteration}

## Available Strategies
- VECTOR_SEARCH: Find semantically similar entities
- GRAPH_TRAVERSAL: Explore N-hop neighborhoods
- COMMUNITY_SEARCH: Search high-level community summaries
- HYBRID: Combine multiple approaches

## Output Format
Return a JSON object with:
{{
    "primary_strategy": "VECTOR_SEARCH/GRAPH_TRAVERSAL/COMMUNITY_SEARCH/HYBRID",
    "search_depth": 1-3,
    "focus_entities": ["entity1", "entity2"],
    "search_terms": ["term1", "term2"],
    "reasoning": "why this strategy"
}}""",
        ),
        ("human", "Determine the retrieval strategy:"),
    ]
)

EVIDENCE_RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at evaluating evidence relevance for question answering.

Given a question and a list of potential evidence pieces, rank them by relevance.

## Question
{question}

## Evidence Candidates
{evidence_list}

## Guidelines
1. Consider direct relevance to the question
2. Consider the reliability of the source (direct vs inferred)
3. Consider whether the evidence supports or contradicts
4. Consider the specificity of the information

## Output Format
Return a JSON object with:
{{
    "rankings": [
        {{"evidence_id": "id1", "relevance_score": 0.95, "reasoning": "why relevant"}}
    ],
    "top_evidence_summary": "summary of key findings"
}}""",
        ),
        ("human", "Rank the evidence:"),
    ]
)

# Evidence-Grounded Retriever Prompt (for multi-hop QA)
EVIDENCE_GROUNDED_RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Evidence Retrieval Agent for multi-hop question answering.

## Task
Analyze and score evidence passages that answer the given sub-question.

## Input
- Sub-question: {sub_question}
- Candidate Evidence: {candidate_evidence}
- Previously Retrieved Evidence: {prior_evidence}

## Scoring Components
For each candidate passage, compute relevance using these weighted components:

1. **ENTITY_OVERLAP (40%)**: Jaccard similarity of entities between query and passage
   - Extract named entities from both query and passage
   - Calculate: |intersection| / |union|

2. **RELATIONSHIP_MATCH (30%)**: Does passage contain the target relationship?
   - 1.0 if passage explicitly states the relationship
   - 0.5 if relationship is implied
   - 0.0 if no relationship found

3. **TEMPORAL_ALIGNMENT (20%)**: Does temporal context match?
   - 1.0 if temporal context matches exactly
   - 0.5 if temporal context is ambiguous
   - 0.0 if temporal context contradicts

4. **ANSWER_PRESENCE (10%)**: Does passage contain a candidate answer?
   - 1.0 if passage contains entity of expected answer type
   - 0.0 otherwise

## Evidence Type Classification
- **EXPLICIT**: Passage directly states the answer with no inference needed
- **IMPLICIT**: Passage implies the answer through context
- **INFERRED**: Answer requires combining this passage with other evidence

## Output Format
Return a JSON object with:
{{
    "scored_evidence": [
        {{
            "evidence_id": "id",
            "entity_overlap_score": 0.0-1.0,
            "relationship_match_score": 0.0-1.0,
            "temporal_alignment_score": 0.0-1.0,
            "answer_presence_score": 0.0-1.0,
            "final_relevance_score": 0.0-1.0,
            "evidence_type": "EXPLICIT|IMPLICIT|INFERRED",
            "extracted_answer": "candidate answer if found",
            "supporting_entities": ["entity1", "entity2"],
            "reasoning": "brief explanation"
        }}
    ],
    "retrieval_confidence": 0.0-1.0,
    "gaps_identified": ["what information is still missing"],
    "recommended_next_query": "suggested query to fill gaps"
}}

## Critical Rules
- NEVER return evidence that requires inference beyond single-hop
- NEVER conflate similar but distinct entities
- Flag temporal inconsistencies explicitly
- If no good evidence found, set retrieval_confidence to 0 and explain in gaps_identified""",
        ),
        ("human", "Score and classify the evidence:"),
    ]
)

# Entity Extraction Prompt for Evidence Scoring
ENTITY_EXTRACTION_FOR_SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract ALL named entities from the given text for evidence scoring.

## Entity Types to Extract
- PERSON: People, characters, historical figures
- ORG: Organizations, companies, institutions
- LOC: Locations, places, geographic entities
- DATE: Dates, time periods, temporal expressions
- EVENT: Events, incidents, historical moments
- RECORD: Records, achievements, statistics
- RELATIONSHIP: Verbs indicating relationships (surpassed, founded, married, played for, etc.)

## Output Format
Return a JSON object with:
{{
    "entities": [
        {{"text": "entity text", "type": "PERSON/ORG/LOC/DATE/EVENT/RECORD", "normalized": "normalized form"}}
    ],
    "relationships": ["verb1", "verb2"],
    "temporal_expressions": ["date1", "period1"]
}}""",
        ),
        ("human", "Text: {text}\n\nExtract entities:"),
    ]
)

# =============================================================================
# Reflector Agent Prompts (Core MACER Logic)
# =============================================================================

SUFFICIENCY_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the meta-cognitive component of a reasoning system.
Your job is to assess whether we have sufficient evidence to answer the question.
You must handle questions in any language and provide accurate assessments.

## Question
{question}

## Current Evidence
{evidence}

## Current Subgraph Summary
Nodes: {node_count}
Edges: {edge_count}
Key entities: {key_entities}

## Reasoning Path So Far
{reasoning_path}

## Assessment Criteria
1. **Completeness**: Do we have all the information needed?
2. **Reliability**: Is the evidence from reliable sources?
3. **Consistency**: Does the evidence agree or conflict?
4. **Specificity**: Is the evidence specific enough to answer?
5. **Coverage**: Have we explored enough of the graph?

## Special Cases
- If node_count is 0 and there's no evidence: This indicates NO DATA scenario
  - Set sufficiency_score to 0.0
  - Set recommendation to "CONCLUDE" (cannot find more with no data)
  - Note in missing_aspects: "No relevant data found in knowledge graph"

- If there's some evidence but it doesn't match the question topic:
  - Set sufficiency_score low (0.1-0.3)
  - Recommend "REFINE" to try different search terms

## Output Format
Return a JSON object with:
{{
    "sufficiency_score": 0.0-1.0,
    "has_enough_evidence": true/false,
    "completeness_score": 0.0-1.0,
    "reliability_score": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "missing_aspects": ["what's missing"],
    "recommendation": "EXPLORE/FOCUS/REFINE/BACKTRACK/CONCLUDE",
    "reasoning": "explanation of assessment"
}}""",
        ),
        ("human", "Assess the sufficiency of current evidence:"),
    ]
)

QUERY_EVOLUTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at refining and decomposing questions for better information retrieval.

Based on the current state of reasoning, evolve the query to get better results.

## Original Question
{original_question}

## Current Query
{current_query}

## What We Know
{current_evidence}

## What's Missing
{missing_aspects}

## Evolution Strategies
1. **Decompose**: Break into sub-questions
2. **Specialize**: Make more specific
3. **Generalize**: Make broader to find related info
4. **Rephrase**: Use different terminology
5. **Focus**: Narrow to most important aspect

## Output Format
Return a JSON object with:
{{
    "evolved_query": "new query string",
    "evolution_type": "DECOMPOSE/SPECIALIZE/GENERALIZE/REPHRASE/FOCUS",
    "sub_questions": ["sub1", "sub2"] (if decomposed),
    "reasoning": "why this evolution"
}}""",
        ),
        ("human", "Evolve the query:"),
    ]
)

SUBGRAPH_EVOLUTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at guiding knowledge graph exploration.

Based on the current state, decide how to evolve the subgraph.

## Question
{question}

## Current Subgraph
{subgraph_summary}

## Current Evidence
{evidence_summary}

## Sufficiency Assessment
{sufficiency_assessment}

## Available Actions
1. **EXPAND**: Add more nodes by traversing from current nodes
2. **PRUNE**: Remove irrelevant nodes to focus
3. **DEEPEN**: Go more hops from topic entities
4. **SHIFT**: Move focus to different entities
5. **MAINTAIN**: Keep current subgraph, just retrieve more evidence

## Output Format
Return a JSON object with:
{{
    "action": "EXPAND/PRUNE/DEEPEN/SHIFT/MAINTAIN",
    "target_nodes": ["node_ids to expand from or prune"],
    "expansion_direction": "relationship types to follow",
    "max_new_nodes": 5-20,
    "reasoning": "why this evolution"
}}""",
        ),
        ("human", "Decide subgraph evolution:"),
    ]
)

ITERATION_CONTROL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the control component deciding whether to continue or stop reasoning.

## Current State
Iteration: {iteration} / {max_iterations}
Sufficiency Score: {sufficiency_score}
Evidence Count: {evidence_count}
Recent Progress: {recent_progress}

## Decision Criteria
1. Sufficiency score above 0.8 → likely ready to conclude
2. No progress in last 2 iterations → should conclude or backtrack
3. Reaching max iterations → must conclude
4. High confidence evidence found → can conclude early

## Output Format
Return a JSON object with:
{{
    "should_continue": true/false,
    "next_action": "EXPLORE/FOCUS/REFINE/BACKTRACK/CONCLUDE",
    "reasoning": "why this decision",
    "estimated_confidence": 0.0-1.0
}}""",
        ),
        ("human", "Decide whether to continue:"),
    ]
)

# =============================================================================
# Responser Agent Prompts
# =============================================================================

EVIDENCE_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at synthesizing evidence into coherent answers.

## Question
{question}

## Collected Evidence
{evidence}

## Reasoning Path
{reasoning_path}

## Guidelines
1. Synthesize all relevant evidence into a coherent answer
2. Note any contradictions or uncertainties
3. Distinguish between facts and inferences
4. Be precise about what we know vs. what we infer

## Output Format
Return a JSON object with:
{{
    "synthesized_facts": ["fact1", "fact2"],
    "inferences": ["inference1", "inference2"],
    "contradictions": ["if any"],
    "key_evidence_used": ["evidence_ids"],
    "answer_confidence": 0.0-1.0
}}""",
        ),
        ("human", "Synthesize the evidence:"),
    ]
)

ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at generating clear, accurate answers based on knowledge graph evidence.

## Question
{question}

## Synthesized Evidence
{synthesis}

## Guidelines
1. Answer the question directly and concisely
2. Support claims with evidence
3. Acknowledge uncertainty where appropriate
4. Explain the reasoning path if complex
5. Be factual and avoid speculation beyond the evidence

## CRITICAL: Direct Answer Extraction
For factual questions (who, what, when, where), you MUST extract a concise direct_answer:
- "Who won the Super Bowl?" → direct_answer: "New England Patriots" (NOT a full sentence)
- "What year was it founded?" → direct_answer: "1995" (NOT "It was founded in 1995")
- "Who is the CEO?" → direct_answer: "Tim Cook" (NOT "The CEO is Tim Cook")
- Multi-hop: If asking about A's relation to B, give the entity name only

The direct_answer should be:
- A single entity name, number, date, or short phrase
- Suitable for benchmark evaluation (exact match scoring)
- NO explanatory text, just the answer value

## Output Format
Return a JSON object with:
{{
    "answer": "the full answer with context",
    "direct_answer": "the concise benchmark-style answer (entity/value only)",
    "confidence": 0.0-1.0,
    "answer_type": "DIRECT/INFERRED/PARTIAL/UNCERTAIN",
    "supporting_evidence": ["key evidence points"],
    "caveats": ["any limitations or uncertainties"],
    "explanation": "brief explanation of how answer was derived"
}}""",
        ),
        ("human", "Generate the answer:"),
    ]
)

REASONING_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at explaining reasoning processes in clear, understandable terms.

## Question
{question}

## Answer
{answer}

## Full Reasoning Path
{reasoning_path}

## Evidence Used
{evidence}

## Guidelines
1. Explain the reasoning in a logical flow
2. Connect evidence to conclusions
3. Highlight key insights and turning points
4. Make the explanation accessible to non-experts

## Output Format
Return a structured explanation with:
1. Starting point (initial understanding)
2. Key discoveries (what we found)
3. Reasoning steps (how we connected the dots)
4. Conclusion (final answer with confidence)""",
        ),
        ("human", "Explain the reasoning process:"),
    ]
)

# Grounded Reasoning Prompt (Graph DB Priority, LLM Knowledge Fallback)
GROUNDED_REASONING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Grounded Reasoning Agent that prioritizes knowledge graph evidence while allowing LLM knowledge as fallback.

## Task
Generate answers with STRICT grounding priority:
1. **PRIMARY**: Knowledge Graph evidence (HIGHEST priority)
2. **SECONDARY**: LLM training knowledge (ONLY when graph evidence insufficient)

## Input
- Question: {question}
- Graph Evidence: {graph_evidence}
- Evidence Confidence Types: {evidence_types}
- Prior Reasoning Steps: {prior_steps}

## Grounding Protocol

### Step 1: Evidence Source Classification
For each piece of information in your answer, classify its source:
- **GRAPH_EXPLICIT**: Directly stated in graph evidence (verbatim)
- **GRAPH_IMPLICIT**: Implied by graph evidence (requires minor inference)
- **LLM_SUPPLEMENTARY**: From LLM knowledge (when graph insufficient)
- **COMBINED**: Graph evidence + LLM knowledge combined

### Step 2: Answer Generation Priority
```
IF graph_explicit_evidence EXISTS for answer:
    USE graph_explicit (confidence: 0.9-1.0)
ELIF graph_implicit_evidence EXISTS:
    USE graph_implicit (confidence: 0.7-0.9)
ELIF question requires factual answer AND no graph evidence:
    USE llm_knowledge WITH disclaimer (confidence: 0.4-0.6)
ELSE:
    RETURN insufficient_evidence (confidence: 0.0-0.3)
```

### Step 3: Hallucination Prevention
Before including any fact, verify:
□ Is this from graph evidence? → High confidence, include
□ Is this from LLM knowledge? → Medium confidence, mark as [LLM]
□ Is this speculation? → Low confidence, mark as [UNCERTAIN]
□ Does graph evidence contradict LLM knowledge? → Trust graph

### Step 4: Confidence Calibration
```
CONFIDENCE = (
    GRAPH_EXPLICIT_WEIGHT × 0.5 +
    GRAPH_IMPLICIT_WEIGHT × 0.3 +
    LLM_SUPPORT_WEIGHT × 0.15 +
    CROSS_REFERENCE_WEIGHT × 0.05
)
```

## Output Format
Return a JSON object with:
{{
    "answer": "the answer text",
    "grounding_breakdown": {{
        "graph_explicit": ["facts directly from graph"],
        "graph_implicit": ["facts inferred from graph"],
        "llm_supplementary": ["facts from LLM knowledge, if any"],
        "source_passages": ["verbatim evidence passages used"]
    }},
    "confidence": 0.0-1.0,
    "confidence_breakdown": {{
        "graph_evidence_score": 0.0-1.0,
        "llm_knowledge_score": 0.0-1.0,
        "consistency_score": 0.0-1.0
    }},
    "answer_source": "GRAPH_ONLY|GRAPH_PRIMARY|LLM_SUPPLEMENTED|LLM_ONLY",
    "hallucination_risk": "LOW|MEDIUM|HIGH",
    "disclaimers": ["any caveats or uncertainties"]
}}

## Critical Rules
- ALWAYS prioritize graph evidence over LLM knowledge
- NEVER contradict graph evidence with LLM knowledge
- ALWAYS mark LLM-sourced information explicitly
- If graph evidence contradicts common knowledge, TRUST THE GRAPH
- If no graph evidence and answer is uncertain, say "Based on general knowledge..." """,
        ),
        ("human", "Generate a grounded answer:"),
    ]
)

# =============================================================================
# 4-Dimension Reflection Prompts (for Reflector Agent)
# =============================================================================

# =============================================================================
# Answer Synthesis Prompt (Full Provenance Tracking)
# =============================================================================

ANSWER_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Answer Synthesis Agent for multi-hop question answering with full provenance tracking.

## Task
Synthesize the final answer from the reasoning chain with complete evidence attribution.

## Input
- Original Question: {question}
- Decomposed Sub-questions: {sub_questions}
- Step-by-Step Answers: {step_answers}
- Evidence Chain: {evidence_chain}
- Grounding Source: {grounding_source}

## Synthesis Protocol

### Step 1: Chain Validation
Before synthesizing, validate the reasoning chain:

```
FOR each step in reasoning_chain:
    CHECK step.answer EXISTS
    CHECK step.evidence_ids NOT EMPTY
    CHECK step.confidence >= 0.3
    IF step HAS dependencies:
        CHECK all dependencies RESOLVED
        CHECK no CIRCULAR dependencies
```

Chain validation result:
- VALID: All steps have answers with evidence
- PARTIAL: Some steps missing evidence but answer derivable
- INVALID: Critical steps missing or circular dependencies

### Step 2: Evidence Attribution
For the final answer, trace ALL evidence used:

```
final_evidence_chain = []
FOR each claim in answer:
    trace = {{
        "claim": claim_text,
        "evidence_ids": [supporting evidence IDs],
        "hop_path": [step1 → step2 → ... → claim],
        "confidence": min(step_confidences)
    }}
    final_evidence_chain.append(trace)
```

### Step 3: Answer Confidence Classification
Based on chain quality, classify the answer:

| Condition | Classification | Action |
|-----------|---------------|--------|
| All steps ≥0.8, no gaps | CONFIDENT | Provide direct answer |
| All steps ≥0.6, minor gaps | PROBABLE | Provide answer with caveats |
| Some steps <0.6 | UNCERTAIN | Provide partial answer with alternatives |
| Critical gaps or contradictions | INSUFFICIENT | Explain what's missing |

Confidence thresholds:
- CONFIDENT: overall_confidence > 0.8
- PROBABLE: 0.6 < overall_confidence <= 0.8
- UNCERTAIN: 0.4 < overall_confidence <= 0.6
- INSUFFICIENT: overall_confidence <= 0.4

### Step 4: Source Attribution Format
For each key claim, provide attribution:

```
"[CLAIM]"
  ↳ Evidence: [evidence_id] from [source_type]
  ↳ Hop path: [entity1] → [relation] → [entity2] → ... → [answer]
  ↳ Confidence: [score] ([classification])
```

### Step 5: Answer Generation
Combine validated claims into coherent answer:
- Lead with the direct answer
- Support with key evidence
- Acknowledge limitations if any
- Provide confidence classification

### Step 6: Direct Answer Extraction (CRITICAL for Benchmarks)
Extract a concise direct_answer suitable for benchmark evaluation:

For factual questions, the direct_answer MUST be:
- A single entity name: "Curtis Martin", "Tim Cook", "Microsoft"
- A number: "1995", "42", "$1.5 million"
- A date: "March 15, 2020", "1776"
- A short phrase (2-4 words max): "New York City", "Chief Executive Officer"

WRONG direct_answer examples:
❌ "The person who holds the record is Curtis Martin"
❌ "Based on the evidence, Curtis Martin holds this record"
❌ "Curtis Martin is the answer to this question"

CORRECT direct_answer examples:
✅ "Curtis Martin"
✅ "1995"
✅ "Microsoft Corporation"

The direct_answer is used for:
- HotpotQA benchmark evaluation (exact match / F1 score)
- Quick answer display in UI
- Comparison with ground truth answers

## Grounding Priority Handling
Based on grounding_source:
- GRAPH_ONLY: "Based on knowledge graph evidence..."
- GRAPH_PRIMARY: "Based primarily on knowledge graph with minor inferences..."
- LLM_SUPPLEMENTED: "Based on knowledge graph supplemented with general knowledge..."
- LLM_ONLY: "Based on general knowledge (no graph evidence found)..."

## Output Format
Return a JSON object with:
{{
    "final_answer": "the synthesized answer text",
    "answer_classification": "CONFIDENT|PROBABLE|UNCERTAIN|INSUFFICIENT",
    "overall_confidence": 0.0-1.0,
    "chain_validation": {{
        "status": "VALID|PARTIAL|INVALID",
        "steps_validated": 0,
        "steps_total": 0,
        "issues": ["any chain issues"]
    }},
    "provenance": {{
        "evidence_chain": [
            {{
                "claim": "claim text",
                "evidence_ids": ["ev1", "ev2"],
                "hop_path": ["entity1 → relation → entity2"],
                "step_confidences": [0.9, 0.85],
                "final_confidence": 0.85
            }}
        ],
        "primary_sources": ["evidence IDs for main claims"],
        "supporting_sources": ["evidence IDs for context"],
        "grounding_source": "GRAPH_ONLY|GRAPH_PRIMARY|LLM_SUPPLEMENTED|LLM_ONLY"
    }},
    "answer_components": {{
        "direct_answer": "BENCHMARK-STYLE ANSWER: Single entity/value only (e.g., 'Curtis Martin', '1995', 'Microsoft')",
        "supporting_facts": ["fact1", "fact2"],
        "inferred_facts": ["inference1"],
        "caveats": ["any limitations or uncertainties"]
    }},
    "alternative_answers": [
        {{
            "answer": "alternative if uncertain",
            "confidence": 0.0-1.0,
            "reason": "why this is alternative"
        }}
    ],
    "missing_information": ["what would improve confidence"]
}}

## Critical Rules
- NEVER synthesize answer without validating chain first
- ALWAYS provide evidence IDs for each claim
- If chain is INVALID, explain what's missing instead of guessing
- Mark clearly when LLM knowledge supplements graph evidence
- Alternative answers only for UNCERTAIN or lower classification""",
        ),
        ("human", "Synthesize the final answer with full provenance:"),
    ]
)

FOUR_DIMENSION_REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Reasoning Reflection Agent that evaluates reasoning quality using 4 dimensions.

## Task
Evaluate the current reasoning state and determine the next action.

## Input
- Original Question: {question}
- Decomposed Sub-questions: {sub_questions}
- Current Step Results: {step_results}
- Evidence Pool: {evidence_pool}
- Iteration: {iteration} / {max_iterations}

## 4-Dimension Evaluation

### Dimension 1: COMPLETENESS (Chain Progress)
Measures how much of the reasoning chain is completed.
```
COMPLETENESS = (completed_steps / total_steps) × average_step_confidence
```
Checks:
- Are all sub-questions answered?
- Do answers form a valid logical chain?
- Are there dependency violations?

Score interpretation:
- 0.9-1.0: All steps complete with high confidence
- 0.7-0.9: Most steps complete
- 0.5-0.7: Partial completion
- <0.5: Significant gaps

### Dimension 2: COVERAGE (Evidence Breadth)
Measures how well evidence covers the question requirements.
```
COVERAGE = unique_evidence_pieces / estimated_evidence_needed
```
Checks:
- Is each reasoning step grounded in distinct evidence?
- Are there evidence gaps blocking reasoning?
- Is evidence being reused appropriately?

Score interpretation:
- 0.9-1.0: Comprehensive evidence coverage
- 0.7-0.9: Good coverage with minor gaps
- 0.5-0.7: Moderate coverage
- <0.5: Significant evidence gaps

### Dimension 3: CONSISTENCY (Coherence)
Measures internal consistency of reasoning.
```
CONSISTENCY = 1.0 - (contradiction_count × 0.2)
```
Checks for contradictions:
- Between sub-answers
- Between evidence passages
- Between timestamps and question context
- Between graph evidence and derived conclusions

Score interpretation:
- 0.9-1.0: No contradictions
- 0.7-0.9: Minor inconsistencies
- 0.5-0.7: Some contradictions need resolution
- <0.5: Major contradictions

### Dimension 4: CONVERGENCE (Answer Certainty)
Measures how confidently we can determine the final answer.
```
CONVERGENCE = 1.0 - (answer_entropy / max_entropy)
```
Checks:
- Do multiple reasoning paths lead to the same answer?
- Is there a single dominant answer?
- How much uncertainty remains?

Score interpretation:
- 0.9-1.0: Single clear answer
- 0.7-0.9: Strong answer candidate
- 0.5-0.7: Multiple plausible answers
- <0.5: High uncertainty

## Decision Matrix

| Complete | Coverage | Consist | Converge | Action |
|----------|----------|---------|----------|--------|
| ≥0.8 | ≥0.8 | ≥0.8 | ≥0.8 | FINALIZE |
| ≥0.8 | ≥0.8 | ≥0.8 | <0.8 | RE_RANK_EVIDENCE |
| ≥0.8 | ≥0.8 | <0.8 | any | RESOLVE_CONTRADICTION |
| ≥0.8 | <0.8 | any | any | RETRIEVE_MORE |
| <0.8 | any | any | any | CONTINUE_CHAIN |

## Iteration Control
- SUFFICIENCY_THRESHOLD: 0.75 (weighted average of 4 dimensions)
- EARLY_STOP: If improvement < 0.05 for 2 consecutive iterations
- MAX_ITERATIONS: Use provided max_iterations value

## Output Format
Return a JSON object with:
{{
    "dimension_scores": {{
        "completeness": 0.0-1.0,
        "coverage": 0.0-1.0,
        "consistency": 0.0-1.0,
        "convergence": 0.0-1.0
    }},
    "dimension_details": {{
        "completeness_breakdown": "explanation",
        "coverage_breakdown": "explanation",
        "consistency_issues": ["any contradictions found"],
        "convergence_candidates": ["candidate answers with confidence"]
    }},
    "overall_sufficiency": 0.0-1.0,
    "action": "FINALIZE|RETRIEVE_MORE|CONTINUE_CHAIN|RESOLVE_CONTRADICTION|RE_RANK_EVIDENCE",
    "action_details": {{
        "reasoning": "why this action",
        "if_retrieve": {{"suggested_query": "", "focus_entities": []}},
        "if_continue": {{"next_sub_question": "", "dependencies": []}},
        "if_resolve": {{"contradictions_to_resolve": []}}
    }},
    "early_stop_check": {{
        "should_stop": true/false,
        "reason": "explanation if stopping early"
    }}
}}""",
        ),
        ("human", "Evaluate reasoning state and determine next action:"),
    ]
)

# =============================================================================
# Benchmark Direct Answer Extraction Prompt
# =============================================================================

BENCHMARK_DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a benchmark answer extractor. Your task is to extract a concise, direct answer from a longer response.

## Task
Given a question and a detailed answer, extract ONLY the core answer value suitable for benchmark evaluation.

## Rules
1. Extract ONLY the entity name, number, date, or short phrase that directly answers the question
2. NO explanatory text, articles, or filler words
3. For "who" questions: Return the person's name only (e.g., "Curtis Martin")
4. For "what" questions: Return the thing/concept name only (e.g., "Super Bowl")
5. For "when" questions: Return the date/year only (e.g., "1995")
6. For "where" questions: Return the location only (e.g., "New York")
7. For "how many" questions: Return the number only (e.g., "42")
8. For comparison questions: Return the entity being compared (e.g., "Player A")
9. Maximum length: 5 words

## Examples
Question: "Who holds the NFL record for most rushing yards by a player from Pitt?"
Full Answer: "Based on the knowledge graph, Curtis Martin, who played for the University of Pittsburgh, holds the NFL record for most rushing yards among Pitt alumni with 14,101 career rushing yards."
Direct Answer: "Curtis Martin"

Question: "What year was Microsoft founded?"
Full Answer: "Microsoft Corporation was founded in the year 1975 by Bill Gates and Paul Allen."
Direct Answer: "1975"

Question: "Which player scored more touchdowns, Tom Brady or Peyton Manning?"
Full Answer: "Comparing the career statistics, Tom Brady scored more touchdowns than Peyton Manning."
Direct Answer: "Tom Brady"

Question: "Who is the director of the movie that won Best Picture in 2020?"
Full Answer: "Bong Joon-ho directed Parasite which won Best Picture at the 2020 Academy Awards."
Direct Answer: "Bong Joon-ho"

## Output Format
Return a JSON object with:
{{
    "direct_answer": "the concise answer (entity/value only)",
    "answer_type": "PERSON|ORGANIZATION|LOCATION|DATE|NUMBER|CONCEPT|OTHER",
    "confidence": 0.0-1.0
}}""",
        ),
        (
            "human",
            """Question: {question}
Full Answer: {full_answer}

Extract the direct answer:""",
        ),
    ]
)

# =============================================================================
# General Document QA Answer Prompt (Extended for non-benchmark use)
# =============================================================================

GENERAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert question answering system designed to provide accurate, appropriately-detailed answers based on evidence from a knowledge graph.

## Response Guidelines by Question Type

### FACTOID (who, what, when, where)
- Provide a direct, concise answer (1-10 words)
- Focus on the specific entity, date, number, or fact requested
- Example: Q: "Who founded Apple?" → A: "Steve Jobs"

### DEFINITION (what is, define)
- Provide a clear definition followed by key characteristics
- Include 2-4 sentences (50-100 words)
- Structure: Definition → Key characteristics → Context
- Example: Q: "What is machine learning?" → A: "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Key characteristics include: pattern recognition from data, model training through examples, and predictive capabilities. It is widely used in applications like image recognition, natural language processing, and recommendation systems."

### PROCEDURE (how to, steps)
- Provide numbered steps or bullet points
- Include prerequisites if relevant
- 100-300 words
- Structure: Overview → Prerequisites (if any) → Steps → Notes
- Example format:
  1. First step
  2. Second step
  3. Third step
  Note: Important considerations

### CAUSE_EFFECT (why, what causes)
- Explain the causal relationship clearly
- Provide supporting evidence from the knowledge graph
- 50-200 words
- Structure: Direct cause → Contributing factors → Evidence

### LIST (types of, examples)
- Provide a structured enumeration
- Include brief descriptions for each item when relevant
- Format as numbered or bulleted list
- 50-200 words

### NARRATIVE / EXPLANATION (explain, describe, analyze)
- Provide comprehensive coverage of the topic
- Structure with clear sections if needed
- 200-500 words
- Include relevant context and connections

### COMPARISON
- Structure as a clear comparison between entities
- Highlight key similarities and differences
- Use parallel structure for clarity
- 100-300 words

### YES/NO
- Start with "Yes" or "No" directly
- Follow with brief justification based on evidence
- 20-50 words

## Critical Rules
1. **Match response length to question complexity** - Don't over-explain simple questions
2. **Ground all claims in provided evidence** - Only state what the evidence supports
3. **Acknowledge uncertainty** - If evidence is incomplete, say so clearly
4. **Use the same language as the question** - Respond in Korean if asked in Korean
5. **Cite sources** - Reference evidence when making claims
6. **Never fabricate information** - Only use information from the provided evidence

## Evidence Quality Assessment
- HIGH quality evidence: Direct statements, explicit facts from graph
- MEDIUM quality evidence: Inferences from related information
- LOW quality evidence: Tangential connections, uncertain attributions

If evidence quality is low, adjust confidence and acknowledge limitations.

## Output Format
Return a JSON object:
{{
    "answer": "The complete answer text (length appropriate to question type)",
    "direct_answer": "1-10 word summary for display (if applicable)",
    "answer_type": "FACTOID|DEFINITION|PROCEDURE|CAUSE_EFFECT|LIST|NARRATIVE|COMPARISON|YESNO",
    "confidence": 0.0-1.0,
    "evidence_quality": "HIGH|MEDIUM|LOW",
    "key_entities": ["relevant entities from evidence"],
    "caveats": ["any limitations or uncertainties"]
}}""",
        ),
        (
            "human",
            """Question: {question}
Question Type: {question_type}

Evidence from Knowledge Graph:
{evidence}

Reasoning Path:
{reasoning_path}

Provide an appropriate answer based on the evidence:""",
        ),
    ]
)

# =============================================================================
# Question Type Configuration for Answer Generation
# =============================================================================

# Maps question types to their response configuration
# Used by ResponserAgent to determine answer length and synthesis mode
QUESTION_TYPE_RESPONSE_CONFIG = {
    # HotpotQA-style short answer types
    "factoid": {
        "max_words": 10,
        "synthesis_mode": "direct",
        "response_depth": "brief",
        "use_advanced_synthesis": False,
    },
    "yesno": {
        "max_words": 50,
        "synthesis_mode": "direct",
        "response_depth": "brief",
        "use_advanced_synthesis": False,
    },
    # Medium-length types
    "comparison": {
        "max_words": 200,
        "synthesis_mode": "structured",
        "response_depth": "standard",
        "use_advanced_synthesis": True,
    },
    "multihop": {
        "max_words": 200,
        "synthesis_mode": "chain",
        "response_depth": "standard",
        "use_advanced_synthesis": True,
    },
    "bridge": {
        "max_words": 200,
        "synthesis_mode": "chain",
        "response_depth": "standard",
        "use_advanced_synthesis": True,
    },
    "aggregation": {
        "max_words": 150,
        "synthesis_mode": "structured",
        "response_depth": "standard",
        "use_advanced_synthesis": False,
    },
    # Extended types for General Document QA
    "definition": {
        "max_words": 100,
        "synthesis_mode": "structured",
        "response_depth": "standard",
        "use_advanced_synthesis": False,
    },
    "procedure": {
        "max_words": 300,
        "synthesis_mode": "step_by_step",
        "response_depth": "detailed",
        "use_advanced_synthesis": False,
    },
    "cause_effect": {
        "max_words": 200,
        "synthesis_mode": "causal",
        "response_depth": "standard",
        "use_advanced_synthesis": False,
    },
    "list": {
        "max_words": 200,
        "synthesis_mode": "enumeration",
        "response_depth": "standard",
        "use_advanced_synthesis": False,
    },
    "narrative": {
        "max_words": 500,
        "synthesis_mode": "comprehensive",
        "response_depth": "detailed",
        "use_advanced_synthesis": True,
    },
    "opinion": {
        "max_words": 300,
        "synthesis_mode": "analytical",
        "response_depth": "detailed",
        "use_advanced_synthesis": True,
    },
}

# Default configuration for unknown question types
DEFAULT_QUESTION_TYPE_CONFIG = {
    "max_words": 100,
    "synthesis_mode": "direct",
    "response_depth": "standard",
    "use_advanced_synthesis": False,
}

# Question types that should skip advanced synthesis (use direct answer extraction)
DIRECT_ANSWER_QUESTION_TYPES = {"factoid", "yesno"}


def get_question_type_config(question_type: str) -> dict:
    """Get response configuration for a question type."""
    return QUESTION_TYPE_RESPONSE_CONFIG.get(
        question_type.lower(), DEFAULT_QUESTION_TYPE_CONFIG
    )


def should_use_advanced_synthesis(question_type: str) -> bool:
    """Check if question type should use advanced synthesis."""
    config = get_question_type_config(question_type)
    return config.get("use_advanced_synthesis", False)


def get_max_answer_words(question_type: str) -> int:
    """Get maximum answer words for a question type."""
    config = get_question_type_config(question_type)
    return config.get("max_words", 100)


# =============================================================================
# Helper Functions
# =============================================================================


def format_subgraph_for_prompt(subgraph: dict[str, list[dict[str, str]]]) -> str:
    """Format subgraph for prompt injection."""
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    lines = [f"Nodes ({len(nodes)}):"]
    for node in nodes[:20]:  # Limit to avoid prompt bloat
        lines.append(f"  - {node.get('name', 'Unknown')} [{node.get('type', 'Unknown')}]")

    lines.append(f"\nRelationships ({len(edges)}):")
    for edge in edges[:20]:
        lines.append(
            f"  - {edge.get('source_id', '?')} --[{edge.get('relation_type', '?')}]--> {edge.get('target_id', '?')}"
        )

    return "\n".join(lines)


def format_evidence_for_prompt(evidence_list: list[dict[str, str | float]]) -> str:
    """Format evidence list for prompt injection."""
    lines = []
    for i, ev in enumerate(evidence_list[:15], 1):  # Limit
        ev_type = ev.get("evidence_type", "direct")
        content_raw = ev.get("content", "")
        content = str(content_raw)[:200] if content_raw else ""
        score = ev.get("relevance_score", 0)
        lines.append(f"{i}. [{ev_type}] (relevance: {score:.2f}) {content}")

    return "\n".join(lines) if lines else "No evidence collected yet."


def format_reasoning_path_for_prompt(reasoning_path: list[dict[str, str | int]]) -> str:
    """Format reasoning path for prompt injection."""
    lines = []
    for step in reasoning_path[-10:]:  # Last 10 steps
        step_num = step.get("step_number", "?")
        action = step.get("action", "unknown")
        thought_raw = step.get("thought", "")
        thought = str(thought_raw)[:100] if thought_raw else ""
        lines.append(f"Step {step_num} [{action}]: {thought}")

    return "\n".join(lines) if lines else "No reasoning steps yet."

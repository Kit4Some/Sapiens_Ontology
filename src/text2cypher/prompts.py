"""
Text2Cypher Prompt Templates.

LLM prompts for natural language to Cypher translation.
"""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are an expert Neo4j Cypher query generator.
Your task is to convert natural language questions into valid Cypher queries.

## Database Schema
{schema}

## Guidelines
1. **Use ONLY schema elements**: Only use node labels, relationship types, and properties that exist in the schema above
2. **Parameterize values**: Use $param syntax for literal values when possible
3. **Case sensitivity**: Property values are case-sensitive; use toLower() for case-insensitive matching
4. **Null handling**: Use COALESCE or IS NOT NULL to handle missing properties
5. **Performance**: Prefer indexed properties in WHERE clauses; avoid cartesian products
6. **Limit results**: Always include LIMIT unless counting or aggregating

## Cypher Best Practices
- Use MATCH for required patterns, OPTIONAL MATCH for optional ones
- Use WITH for query chaining and intermediate aggregations
- Use DISTINCT to avoid duplicates when appropriate
- Use ORDER BY before LIMIT for consistent results

## Output Format
Return ONLY the Cypher query. No explanations, no markdown formatting, no backticks."""

SYSTEM_PROMPT_WITH_EXAMPLES = """You are an expert Neo4j Cypher query generator.
Your task is to convert natural language questions into valid Cypher queries.

## Database Schema
{schema}

## Similar Examples
{examples}

## Guidelines
1. **Use ONLY schema elements**: Only use node labels, relationship types, and properties that exist in the schema
2. **Follow example patterns**: Use similar query patterns as shown in the examples above
3. **Parameterize values**: Use $param syntax for literal values
4. **Case sensitivity**: Use toLower() for case-insensitive string matching
5. **Limit results**: Include LIMIT unless aggregating

## Output Format
Return ONLY the Cypher query. No explanations or markdown."""

SYSTEM_PROMPT_ENTITY_RESOLUTION = """You are an expert Neo4j Cypher query generator with entity resolution capabilities.
Convert natural language questions to Cypher, resolving entity mentions to database values.

## Database Schema
{schema}

## Entity Mapping
The following entities were found in the question and mapped to database values:
{entity_mappings}

## Guidelines
1. Use the resolved entity values from the mapping above, not the original mentions
2. Use exact matches for resolved entities
3. Handle cases where entities might not exist with OPTIONAL MATCH or existence checks
4. Follow schema constraints strictly

## Output Format
Return ONLY the Cypher query."""

# =============================================================================
# Self-Healing Prompt
# =============================================================================

SELF_HEALING_PROMPT = """The following Cypher query failed validation:

## Original Question
{question}

## Failed Query
{failed_query}

## Error Message
{error_message}

## Database Schema
{schema}

## Instructions
Fix the Cypher query to address the error. Common issues:
- Using non-existent labels, types, or properties (check schema)
- Syntax errors (missing parentheses, brackets, quotes)
- Invalid relationship patterns
- Missing RETURN clause
- Type mismatches in comparisons

Return ONLY the corrected Cypher query."""

# =============================================================================
# Chat Prompt Templates
# =============================================================================

TEXT2CYPHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_BASE),
        ("human", "Question: {question}\n\nGenerate the Cypher query:"),
    ]
)

TEXT2CYPHER_WITH_EXAMPLES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_WITH_EXAMPLES),
        ("human", "Question: {question}\n\nGenerate the Cypher query:"),
    ]
)

TEXT2CYPHER_ENTITY_RESOLUTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_ENTITY_RESOLUTION),
        ("human", "Question: {question}\n\nGenerate the Cypher query with resolved entities:"),
    ]
)

SELF_HEALING_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "You are an expert at fixing Cypher query errors."), ("human", SELF_HEALING_PROMPT)]
)

# =============================================================================
# Few-Shot Example Template
# =============================================================================

EXAMPLE_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", "Question: {question}"), ("ai", "{cypher}")]
)


def create_few_shot_prompt(examples: list[dict[str, str]]) -> FewShotChatMessagePromptTemplate:
    """
    Create a few-shot prompt from examples.

    Args:
        examples: List of {"question": str, "cypher": str} dicts

    Returns:
        FewShotChatMessagePromptTemplate
    """
    return FewShotChatMessagePromptTemplate(
        example_prompt=EXAMPLE_TEMPLATE,
        examples=examples,
    )


# =============================================================================
# Schema Formatting
# =============================================================================


def format_schema_for_prompt(schema: dict[str, Any]) -> str:
    """
    Format Neo4j schema for prompt injection.

    Args:
        schema: Schema dict from OntologyGraphClient.get_schema()

    Returns:
        Formatted schema string
    """
    lines = []

    # Node labels with properties
    lines.append("### Node Labels")
    node_props = schema.get("node_properties", {})
    for label in schema.get("node_labels", []):
        props = node_props.get(label, [])
        prop_names = [p.get("name", "") for p in props if p.get("name")]
        if prop_names:
            lines.append(f"(:{label}) - Properties: {', '.join(prop_names)}")
        else:
            lines.append(f"(:{label})")

    # Relationship types with properties
    lines.append("\n### Relationship Types")
    rel_props = schema.get("relationship_properties", {})
    for rel_type in schema.get("relationship_types", []):
        props = rel_props.get(rel_type, [])
        prop_names = [p.get("name", "") for p in props if p.get("name")]
        if prop_names:
            lines.append(f"[:{rel_type}] - Properties: {', '.join(prop_names)}")
        else:
            lines.append(f"[:{rel_type}]")

    # Indexes (useful for query optimization hints)
    indexes = schema.get("indexes", [])
    if indexes:
        lines.append("\n### Indexes (prefer these properties in WHERE clauses)")
        for idx in indexes[:10]:  # Limit to avoid prompt bloat
            idx_name = idx.get("name", "")
            idx_type = idx.get("type", "")
            if "vector" not in idx_type.lower() and "fulltext" not in idx_type.lower():
                lines.append(f"- {idx_name}: {idx_type}")

    return "\n".join(lines)


def format_entity_mappings(mappings: dict[str, dict[str, Any]]) -> str:
    """
    Format entity mappings for prompt injection.

    Args:
        mappings: Dict of mention -> {resolved_value, node_label, property, score}

    Returns:
        Formatted mapping string
    """
    lines = []
    for mention, resolved in mappings.items():
        value = resolved.get("resolved_value", mention)
        label = resolved.get("node_label", "Entity")
        prop = resolved.get("property", "name")
        score = resolved.get("score", 1.0)
        lines.append(f'- "{mention}" → (:{label} {{{prop}: "{value}"}}) [confidence: {score:.2f}]')

    return "\n".join(lines) if lines else "No entity mappings found."

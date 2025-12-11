"""
Dynamic Prompt Builder.

Generates extraction prompts dynamically based on schema configuration:
- Entity extraction prompts from entity registry
- Relation extraction prompts from predicate registry
- Few-shot example injection
- Multi-language support
"""

from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate

from src.sddi.extractors.schema.domain_profile import ExtractionSchema

logger = structlog.get_logger(__name__)


@dataclass
class PromptTemplate:
    """A generated prompt template with metadata."""

    template: ChatPromptTemplate
    template_type: str  # "entity" or "relation"
    variables: list[str]
    schema_version: str = "1.0"

    def format(self, **kwargs) -> str:
        """Format the template with variables."""
        messages = self.template.format_messages(**kwargs)
        return "\n".join(m.content for m in messages)


class DynamicPromptBuilder:
    """
    Builds extraction prompts dynamically from schema.

    Features:
    - Schema-driven prompt generation
    - Few-shot example injection
    - Language customization
    - Output format specification
    """

    def __init__(self, schema: ExtractionSchema):
        self.schema = schema

    def build_entity_extraction_prompt(
        self,
        include_examples: bool = True,
        few_shot_count: int = 0,
        output_format: str = "json",
        language: str | None = None,
    ) -> PromptTemplate:
        """
        Build entity extraction prompt from schema.

        Args:
            include_examples: Include entity type examples
            few_shot_count: Number of few-shot examples to include
            output_format: Output format (json, xml, markdown)
            language: Override language

        Returns:
            PromptTemplate ready for use
        """
        lang = language or self.schema.primary_language

        # Build entity types section
        entity_types_section = self.schema.get_entity_types_prompt(include_examples)

        # Build guidelines section
        guidelines_section = self.schema.get_guidelines_prompt(for_entities=True)

        # Build few-shot examples
        few_shot_section = ""
        if few_shot_count > 0 and self.schema.few_shot_examples:
            examples = self.schema.few_shot_examples[:few_shot_count]
            few_shot_section = self._format_few_shot_examples(examples, "entity")

        # Build output format section
        output_section = self._get_entity_output_format(output_format)

        # Language instruction
        lang_instruction = self._get_language_instruction(lang)

        # Compose system prompt
        system_prompt = f"""You are an expert Named Entity Recognition (NER) system.
Extract all named entities from the given text and classify them according to the types below.

{entity_types_section}

{guidelines_section}
{lang_instruction}
{output_section}
{few_shot_section}
Only return valid {output_format.upper()}, no additional text."""

        # Human prompt
        human_prompt = """Extract all named entities from the following text:

---
{text}
---

Return the entities:"""

        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt.strip()),
            ("human", human_prompt.strip()),
        ])

        return PromptTemplate(
            template=template,
            template_type="entity",
            variables=["text"],
        )

    def build_relation_extraction_prompt(
        self,
        include_examples: bool = True,
        categories: list[str] | None = None,
        few_shot_count: int = 0,
        output_format: str = "json",
        language: str | None = None,
    ) -> PromptTemplate:
        """
        Build relation extraction prompt from schema.

        Args:
            include_examples: Include predicate examples
            categories: Predicate categories to include (None = all)
            few_shot_count: Number of few-shot examples
            output_format: Output format
            language: Override language

        Returns:
            PromptTemplate ready for use
        """
        lang = language or self.schema.primary_language

        # Build predicates section
        predicates_section = self.schema.get_predicates_prompt(
            include_examples, categories
        )

        # Build guidelines section
        guidelines_section = self.schema.get_guidelines_prompt(for_entities=False)

        # Build output format section
        output_section = self._get_relation_output_format(output_format)

        # Language instruction
        lang_instruction = self._get_language_instruction(lang)

        # Few-shot section
        few_shot_section = ""
        if few_shot_count > 0 and self.schema.few_shot_examples:
            examples = [e for e in self.schema.few_shot_examples if "relations" in e]
            few_shot_section = self._format_few_shot_examples(examples[:few_shot_count], "relation")

        # Compose system prompt
        system_prompt = f"""You are an expert relation extraction system for knowledge graph construction.
Given a text and a list of entities found in it, extract all meaningful relationships between these entities.

## Known Entities
{{entities}}

{predicates_section}

{guidelines_section}
{lang_instruction}
{output_section}
{few_shot_section}
Only return valid {output_format.upper()}, no additional text."""

        # Human prompt
        human_prompt = """Extract relations from this text, using only the entities provided:

---
{text}
---

Return relations:"""

        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt.strip()),
            ("human", human_prompt.strip()),
        ])

        return PromptTemplate(
            template=template,
            template_type="relation",
            variables=["text", "entities"],
        )

    def build_batch_entity_prompt(
        self,
        include_examples: bool = True,
        output_format: str = "json",
    ) -> PromptTemplate:
        """Build prompt for batch entity extraction."""

        entity_types_section = self.schema.get_entity_types_prompt(include_examples)
        entity_type_names = ", ".join(self.schema.entity_registry.get_type_names())

        output_section = self._get_entity_output_format(output_format, batch=True)

        system_prompt = f"""You are an expert Named Entity Recognition system.
Extract all unique named entities from the provided text chunks.

{entity_types_section}

{output_section}

Deduplicate entities across chunks - list each unique entity only once.
Only return valid {output_format.upper()}."""

        human_prompt = """Extract entities from these text chunks:

{chunks_text}

Return deduplicated entities:"""

        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt.strip()),
            ("human", human_prompt.strip()),
        ])

        return PromptTemplate(
            template=template,
            template_type="entity_batch",
            variables=["chunks_text"],
        )

    def build_coref_relation_prompt(
        self,
        output_format: str = "json",
    ) -> PromptTemplate:
        """Build prompt for relation extraction with coreference resolution."""

        predicates_section = self.schema.get_predicates_prompt(include_examples=False)

        system_prompt = f"""You are an expert relation extraction system with coreference resolution.
Given a text and entities, extract relations while resolving pronouns and coreferences.

## Known Entities
{{entities}}

## Guidelines
1. Resolve pronouns (he, she, it, they) to their antecedent entities
2. Handle coreferences (the company → Microsoft, the CEO → Satya Nadella)
3. Extract implicit relations when strongly implied
4. Use consistent entity names (prefer the canonical form from the entity list)

{predicates_section}

## Output Format
JSON with "relations" array containing: source, target, predicate, description, confidence

Only return valid JSON."""

        human_prompt = """Extract relations with coreference resolution:

---
{text}
---

Return relations:"""

        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt.strip()),
            ("human", human_prompt.strip()),
        ])

        return PromptTemplate(
            template=template,
            template_type="relation_coref",
            variables=["text", "entities"],
        )

    def _get_entity_output_format(
        self,
        format_type: str,
        batch: bool = False,
    ) -> str:
        """Get output format specification for entity extraction."""

        if format_type == "json":
            chunk_field = '\n- source_chunks: List of chunk indices where entity appears (0-indexed)' if batch else ''
            return f"""## Output Format
Return a JSON object with an "entities" array containing objects with:
- name: The entity mention as it appears in text
- type: One of the entity types above
- description: Brief description based on context (optional)
- aliases: List of alternative names mentioned (optional)
- confidence: Confidence score 0.0-1.0{chunk_field}"""

        elif format_type == "xml":
            return """## Output Format
Return XML with <entities> root containing <entity> elements:
<entities>
  <entity>
    <name>Entity Name</name>
    <type>ENTITY_TYPE</type>
    <description>Brief description</description>
    <confidence>0.9</confidence>
  </entity>
</entities>"""

        else:
            return """## Output Format
Return entities in markdown list format:
- **Entity Name** (TYPE): Description [confidence: 0.9]"""

    def _get_relation_output_format(self, format_type: str) -> str:
        """Get output format specification for relation extraction."""

        if format_type == "json":
            return """## Output Format
Return a JSON object with a "relations" array containing:
- source: Source entity name (must match an entity from the list)
- target: Target entity name (must match an entity from the list)
- predicate: Relationship type in UPPER_SNAKE_CASE
- description: Brief description of the relationship
- confidence: Score 0.0-1.0 based on explicitness in text"""

        elif format_type == "xml":
            return """## Output Format
Return XML with <relations> root:
<relations>
  <relation>
    <source>Source Entity</source>
    <target>Target Entity</target>
    <predicate>PREDICATE_NAME</predicate>
    <confidence>0.9</confidence>
  </relation>
</relations>"""

        else:
            return """## Output Format
Return relations in markdown:
- Source Entity **PREDICATE** Target Entity [confidence: 0.9]"""

    def _get_language_instruction(self, language: str) -> str:
        """Get language-specific instruction."""

        language_map = {
            "en": "",  # No instruction needed for English
            "ko": "\n## Language\nThe text is in Korean. Extract entities in their original Korean form, but classify using English type names.",
            "ja": "\n## Language\nThe text is in Japanese. Extract entities in their original Japanese form.",
            "zh": "\n## Language\nThe text is in Chinese. Extract entities in their original Chinese form.",
            "es": "\n## Language\nThe text is in Spanish. Extract entities preserving Spanish names.",
            "de": "\n## Language\nThe text is in German. Extract entities preserving German names.",
            "fr": "\n## Language\nThe text is in French. Extract entities preserving French names.",
        }

        if language not in language_map:
            return f"\n## Language\nThe text may be in {language}. Preserve original entity names."

        return language_map[language]

    def _format_few_shot_examples(
        self,
        examples: list[dict[str, Any]],
        example_type: str,
    ) -> str:
        """Format few-shot examples for prompt."""

        if not examples:
            return ""

        lines = ["\n## Examples"]

        for i, example in enumerate(examples, 1):
            lines.append(f"\n### Example {i}")

            if example_type == "entity":
                lines.append(f"Input: {example.get('text', '')[:200]}...")
                entities = example.get("entities", [])
                lines.append("Output:")
                for entity in entities[:3]:
                    lines.append(f"  - {entity.get('name')} [{entity.get('type')}]")

            elif example_type == "relation":
                lines.append(f"Input: {example.get('text', '')[:200]}...")
                relations = example.get("relations", [])
                lines.append("Output:")
                for rel in relations[:3]:
                    lines.append(
                        f"  - {rel.get('source')} {rel.get('predicate')} {rel.get('target')}"
                    )

        return "\n".join(lines)


def create_prompt_builder(
    profile_name: str | None = None,
    schema: ExtractionSchema | None = None,
) -> DynamicPromptBuilder:
    """
    Factory function to create a prompt builder.

    Args:
        profile_name: Name of domain profile to use
        schema: Pre-built schema (takes precedence)

    Returns:
        DynamicPromptBuilder configured with schema
    """
    if schema is None:
        from src.sddi.extractors.schema.domain_profile import get_profile_manager
        manager = get_profile_manager()
        schema = manager.create_schema(profile_name)

    return DynamicPromptBuilder(schema)

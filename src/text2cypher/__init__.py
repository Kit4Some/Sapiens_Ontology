"""
Text2Cypher Engine Module.

Natural language to Cypher query translation with:
- Schema-aware generation
- Few-shot example selection (semantic similarity)
- Entity resolution
- Query validation and self-healing
"""

from src.text2cypher.examples import (
    DEFAULT_EXAMPLES,
    EXAMPLE_CATEGORIES,
    CypherExample,
    FewShotExampleStore,
)
from src.text2cypher.generator import (
    CypherGenerationResult,
    Text2CypherGenerator,
    Text2CypherGeneratorFactory,
)
from src.text2cypher.prompts import (
    TEXT2CYPHER_ENTITY_RESOLUTION_PROMPT,
    TEXT2CYPHER_PROMPT,
    TEXT2CYPHER_WITH_EXAMPLES_PROMPT,
    format_entity_mappings,
    format_schema_for_prompt,
)
from src.text2cypher.validator import (
    CypherSanitizer,
    CypherValidator,
    HealingResult,
    ValidationResult,
)

__all__ = [
    # Generator
    "Text2CypherGenerator",
    "Text2CypherGeneratorFactory",
    "CypherGenerationResult",
    # Validator
    "CypherValidator",
    "CypherSanitizer",
    "ValidationResult",
    "HealingResult",
    # Examples
    "FewShotExampleStore",
    "CypherExample",
    "DEFAULT_EXAMPLES",
    "EXAMPLE_CATEGORIES",
    # Prompts
    "TEXT2CYPHER_PROMPT",
    "TEXT2CYPHER_WITH_EXAMPLES_PROMPT",
    "TEXT2CYPHER_ENTITY_RESOLUTION_PROMPT",
    "format_schema_for_prompt",
    "format_entity_mappings",
]

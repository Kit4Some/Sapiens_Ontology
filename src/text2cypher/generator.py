"""
Text2Cypher Generator Module.

Advanced natural language to Cypher translation with:
- Schema-aware generation
- Few-shot example selection
- Entity resolution
- Query validation and self-healing
"""

import re
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.text2cypher.examples import FewShotExampleStore
from src.text2cypher.prompts import (
    TEXT2CYPHER_ENTITY_RESOLUTION_PROMPT,
    TEXT2CYPHER_PROMPT,
    TEXT2CYPHER_WITH_EXAMPLES_PROMPT,
    format_entity_mappings,
    format_schema_for_prompt,
)
from src.text2cypher.validator import CypherValidator

logger = structlog.get_logger(__name__)


class CypherGenerationResult(BaseModel):
    """Result of Cypher generation."""

    cypher: str = Field(..., description="Generated Cypher query")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Generation confidence")
    is_valid: bool = Field(default=False, description="Whether query passed validation")
    entities_detected: list[str] = Field(
        default_factory=list, description="Detected entity mentions"
    )
    entity_mappings: dict[str, Any] = Field(
        default_factory=dict, description="Entity resolution mappings"
    )
    examples_used: int = Field(default=0, description="Number of few-shot examples used")
    healing_attempts: int = Field(default=0, description="Number of self-healing attempts")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Query parameters")


@dataclass
class EntityResolution:
    """Result of entity resolution."""

    mention: str
    resolved_value: str
    node_label: str
    property_name: str
    score: float


class Text2CypherGenerator:
    """
    Advanced Text2Cypher generator with schema injection, few-shot examples,
    entity resolution, and self-healing capabilities.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings | None = None,
        neo4j_client: OntologyGraphClient | None = None,
        example_store: FewShotExampleStore | None = None,
        num_examples: int = 3,
        enable_entity_resolution: bool = True,
        enable_self_healing: bool = True,
        max_healing_retries: int = 3,
    ) -> None:
        """
        Initialize the Text2Cypher generator.

        Args:
            llm: LangChain chat model for generation
            embeddings: Embedding model for similarity search
            neo4j_client: Neo4j client for schema and entity lookup
            example_store: Few-shot example store
            num_examples: Number of similar examples to include
            enable_entity_resolution: Whether to resolve entities to DB values
            enable_self_healing: Whether to attempt query repair on failure
            max_healing_retries: Maximum self-healing attempts
        """
        self._llm = llm
        self._embeddings = embeddings
        self._client = neo4j_client or get_ontology_client()
        self._num_examples = num_examples
        self._enable_entity_resolution = enable_entity_resolution
        self._enable_self_healing = enable_self_healing
        self._max_healing_retries = max_healing_retries

        # Initialize example store
        self._example_store = example_store or FewShotExampleStore(embeddings=embeddings)

        # Initialize validator
        self._validator = CypherValidator(
            llm=llm if enable_self_healing else None,
            neo4j_client=self._client,
        )

        # Build generation chains
        self._base_chain = TEXT2CYPHER_PROMPT | self._llm | StrOutputParser()
        self._examples_chain = TEXT2CYPHER_WITH_EXAMPLES_PROMPT | self._llm | StrOutputParser()
        self._entity_chain = TEXT2CYPHER_ENTITY_RESOLUTION_PROMPT | self._llm | StrOutputParser()

        # Caches
        self._schema_cache: dict[str, Any] | None = None

    async def _get_schema(self) -> dict[str, Any]:
        """Get cached database schema."""
        if self._schema_cache is None:
            await self._client.connect()
            self._schema_cache = await self._client.get_schema()
        return self._schema_cache

    def _clean_cypher_response(self, response: str) -> str:
        """Clean up LLM response to extract pure Cypher."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            lines = lines[1:]  # Remove opening ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove closing ```
            response = "\n".join(lines)

        # Handle ```cypher prefix
        if response.lower().startswith("cypher"):
            response = response[6:].strip()

        return response.strip()

    def _extract_entity_mentions(self, question: str) -> list[str]:
        """
        Extract potential entity mentions from a question.

        Uses heuristics to identify likely entity references.
        """
        mentions = []

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', question)
        mentions.extend(quoted)
        quoted = re.findall(r"'([^']+)'", question)
        mentions.extend(quoted)

        # Capitalized phrases (potential proper nouns)
        # Match 1-4 consecutive capitalized words
        capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", question)
        # Filter out common words and question starters
        stop_words = {
            "What",
            "Who",
            "Where",
            "When",
            "How",
            "Why",
            "Which",
            "Find",
            "Show",
            "List",
            "Get",
            "Tell",
            "Give",
            "The",
            "This",
            "That",
            "These",
            "Those",
        }
        mentions.extend([m for m in capitalized if m not in stop_words])

        return list(set(mentions))

    async def _resolve_entities(
        self,
        mentions: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        Resolve entity mentions to database values using fuzzy matching.

        Args:
            mentions: List of entity mentions from the question

        Returns:
            Dict mapping mentions to resolved values
        """
        if not mentions:
            return {}

        mappings = {}

        for mention in mentions:
            # Try fulltext search first
            try:
                results = await self._client.fulltext_search(
                    query_text=mention,
                    top_k=1,
                    min_score=0.5,
                )

                if results:
                    best = results[0]
                    mappings[mention] = {
                        "resolved_value": best.text or mention,
                        "node_label": best.node_label,
                        "property": "name",
                        "score": best.score,
                        "node_id": best.node_id,
                    }
                    continue
            except Exception as e:
                logger.debug("Fulltext search failed for entity", mention=mention, error=str(e))

            # Fall back to exact match
            try:
                query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($mention)
                RETURN e.id as id, e.name as name, e.type as type
                LIMIT 1
                """
                cypher_results = await self._client.execute_cypher(query, {"mention": mention})

                if cypher_results:
                    match = cypher_results[0]
                    mappings[mention] = {
                        "resolved_value": match["name"],
                        "node_label": "Entity",
                        "property": "name",
                        "score": 0.8,
                        "node_id": match["id"],
                    }
            except Exception as e:
                logger.debug("Entity lookup failed", mention=mention, error=str(e))

        logger.info("Entity resolution completed", resolved=len(mappings), total=len(mentions))
        return mappings

    def _calculate_confidence(
        self,
        cypher: str,
        is_valid: bool,
        healing_attempts: int,
        entity_resolutions: int,
    ) -> float:
        """Calculate confidence score for generated query."""
        base_confidence = 0.7

        # Penalize for healing attempts
        healing_penalty = healing_attempts * 0.1

        # Boost for successful validation
        validation_boost = 0.2 if is_valid else -0.2

        # Boost for entity resolution
        resolution_boost = min(entity_resolutions * 0.05, 0.15)

        # Query complexity penalty (very long queries might be less reliable)
        complexity_penalty = 0.1 if len(cypher) > 500 else 0

        confidence = (
            base_confidence
            - healing_penalty
            + validation_boost
            + resolution_boost
            - complexity_penalty
        )
        return max(0.0, min(1.0, confidence))

    async def generate(
        self,
        question: str,
        use_examples: bool = True,
        validate: bool = True,
    ) -> CypherGenerationResult:
        """
        Generate Cypher query from natural language question.

        Args:
            question: Natural language question
            use_examples: Whether to include few-shot examples
            validate: Whether to validate and heal the query

        Returns:
            CypherGenerationResult with generated query and metadata
        """
        logger.info("Generating Cypher", question=question[:100])

        # Get schema
        schema = await self._get_schema()
        schema_str = format_schema_for_prompt(schema)

        # Get similar examples
        examples_str = ""
        examples_used = 0
        if use_examples and self._embeddings:
            similar_examples = await self._example_store.get_similar_examples(
                question=question,
                k=self._num_examples,
            )
            if similar_examples:
                examples_str = self._example_store.format_examples_for_prompt(similar_examples)
                examples_used = len(similar_examples)

        # Generate query
        if examples_str:
            cypher = await self._examples_chain.ainvoke(
                {
                    "schema": schema_str,
                    "examples": examples_str,
                    "question": question,
                }
            )
        else:
            cypher = await self._base_chain.ainvoke(
                {
                    "schema": schema_str,
                    "question": question,
                }
            )

        cypher = self._clean_cypher_response(cypher)

        # Extract detected entities
        entities_detected = self._extract_entity_mentions(question)

        # Validate and heal
        is_valid = False
        healing_attempts = 0

        if validate:
            if self._enable_self_healing:
                healing_result = await self._validator.validate_and_heal(
                    cypher=cypher,
                    question=question,
                    max_retries=self._max_healing_retries,
                )
                is_valid = healing_result.success
                healing_attempts = healing_result.attempts - 1  # First attempt isn't healing
                if healing_result.healed_query:
                    cypher = healing_result.healed_query
            else:
                validation_result = await self._validator.validate_syntax(cypher)
                is_valid = validation_result.is_valid

        # Calculate confidence
        confidence = self._calculate_confidence(
            cypher=cypher,
            is_valid=is_valid,
            healing_attempts=healing_attempts,
            entity_resolutions=0,
        )

        logger.info(
            "Cypher generated",
            cypher=cypher[:200],
            is_valid=is_valid,
            confidence=confidence,
        )

        return CypherGenerationResult(
            cypher=cypher,
            confidence=confidence,
            is_valid=is_valid,
            entities_detected=entities_detected,
            entity_mappings={},
            examples_used=examples_used,
            healing_attempts=healing_attempts,
            parameters={},
        )

    async def generate_with_entity_resolution(
        self,
        question: str,
        validate: bool = True,
    ) -> CypherGenerationResult:
        """
        Generate Cypher with entity resolution.

        Resolves entity mentions in the question to actual database values
        before generating the query, improving accuracy.

        Args:
            question: Natural language question
            validate: Whether to validate and heal the query

        Returns:
            CypherGenerationResult with resolved entities
        """
        logger.info("Generating Cypher with entity resolution", question=question[:100])

        # Extract and resolve entities
        entities_detected = self._extract_entity_mentions(question)
        entity_mappings = {}

        if self._enable_entity_resolution and entities_detected:
            entity_mappings = await self._resolve_entities(entities_detected)

        # Get schema
        schema = await self._get_schema()
        schema_str = format_schema_for_prompt(schema)

        # Generate query with entity mappings
        if entity_mappings:
            entity_mappings_str = format_entity_mappings(entity_mappings)
            cypher = await self._entity_chain.ainvoke(
                {
                    "schema": schema_str,
                    "entity_mappings": entity_mappings_str,
                    "question": question,
                }
            )
        else:
            # Fall back to standard generation
            return await self.generate(question, validate=validate)

        cypher = self._clean_cypher_response(cypher)

        # Build parameters from resolved entities
        parameters = {}
        for mention, resolution in entity_mappings.items():
            param_name = re.sub(r"[^a-zA-Z0-9_]", "_", mention.lower())
            parameters[param_name] = resolution["resolved_value"]

        # Validate and heal
        is_valid = False
        healing_attempts = 0

        if validate:
            if self._enable_self_healing:
                healing_result = await self._validator.validate_and_heal(
                    cypher=cypher,
                    question=question,
                    max_retries=self._max_healing_retries,
                )
                is_valid = healing_result.success
                healing_attempts = healing_result.attempts - 1
                if healing_result.healed_query:
                    cypher = healing_result.healed_query
            else:
                validation_result = await self._validator.validate_syntax(cypher)
                is_valid = validation_result.is_valid

        # Calculate confidence
        confidence = self._calculate_confidence(
            cypher=cypher,
            is_valid=is_valid,
            healing_attempts=healing_attempts,
            entity_resolutions=len(entity_mappings),
        )

        logger.info(
            "Cypher generated with entity resolution",
            cypher=cypher[:200],
            is_valid=is_valid,
            entities_resolved=len(entity_mappings),
        )

        return CypherGenerationResult(
            cypher=cypher,
            confidence=confidence,
            is_valid=is_valid,
            entities_detected=entities_detected,
            entity_mappings=entity_mappings,
            examples_used=0,  # Entity resolution doesn't use examples
            healing_attempts=healing_attempts,
            parameters=parameters,
        )

    async def execute(
        self,
        question: str,
        use_entity_resolution: bool = True,
    ) -> tuple[CypherGenerationResult, list[dict[str, Any]]]:
        """
        Generate and execute a Cypher query.

        Convenience method that generates the query and executes it.

        Args:
            question: Natural language question
            use_entity_resolution: Whether to use entity resolution

        Returns:
            Tuple of (generation_result, query_results)
        """
        if use_entity_resolution:
            result = await self.generate_with_entity_resolution(question)
        else:
            result = await self.generate(question)

        if not result.is_valid:
            logger.warning("Executing potentially invalid query", cypher=result.cypher[:100])

        try:
            query_results = await self._client.execute_cypher(
                result.cypher,
                result.parameters or None,
            )
        except Exception as e:
            logger.error("Query execution failed", error=str(e))
            query_results = []

        return result, query_results

    def add_example(
        self,
        question: str,
        cypher: str,
        category: str = "general",
    ) -> None:
        """
        Add a new few-shot example.

        Args:
            question: Natural language question
            cypher: Corresponding Cypher query
            category: Example category
        """
        self._example_store.add_example(
            question=question,
            cypher=cypher,
            category=category,
        )

    async def refresh_schema(self) -> None:
        """Force refresh of cached schema."""
        self._schema_cache = None
        await self._get_schema()


class Text2CypherGeneratorFactory:
    """Factory for creating Text2Cypher generators with common configurations."""

    @staticmethod
    def create_basic(llm: BaseChatModel) -> Text2CypherGenerator:
        """Create a basic generator without examples or entity resolution."""
        return Text2CypherGenerator(
            llm=llm,
            embeddings=None,
            enable_entity_resolution=False,
            enable_self_healing=False,
        )

    @staticmethod
    def create_with_examples(
        llm: BaseChatModel,
        embeddings: Embeddings,
    ) -> Text2CypherGenerator:
        """Create a generator with few-shot examples."""
        return Text2CypherGenerator(
            llm=llm,
            embeddings=embeddings,
            enable_entity_resolution=False,
            enable_self_healing=True,
        )

    @staticmethod
    def create_full(
        llm: BaseChatModel,
        embeddings: Embeddings,
    ) -> Text2CypherGenerator:
        """Create a fully-featured generator with all capabilities."""
        return Text2CypherGenerator(
            llm=llm,
            embeddings=embeddings,
            enable_entity_resolution=True,
            enable_self_healing=True,
            max_healing_retries=3,
        )

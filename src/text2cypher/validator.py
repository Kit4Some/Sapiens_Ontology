"""
Cypher Validator Module.

Validates Cypher queries and provides self-healing capabilities.
"""

import re
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.text2cypher.prompts import (
    SELF_HEALING_CHAT_PROMPT,
    format_schema_for_prompt,
)

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of Cypher validation."""

    is_valid: bool
    error_message: str | None = None
    error_type: str | None = None  # syntax, semantic, runtime
    suggestions: list[str] | None = None


@dataclass
class HealingResult:
    """Result of self-healing attempt."""

    success: bool
    original_query: str
    healed_query: str | None
    attempts: int
    errors: list[str]


class CypherValidator:
    """
    Cypher query validator with self-healing capabilities.

    Validates Cypher syntax using Neo4j EXPLAIN and provides
    LLM-based query correction when validation fails.
    """

    # Common Cypher syntax patterns for basic validation
    REQUIRED_CLAUSES = re.compile(r"\b(RETURN|CREATE|MERGE|DELETE|SET|REMOVE)\b", re.IGNORECASE)
    MATCH_PATTERN = re.compile(r"\bMATCH\b", re.IGNORECASE)
    DANGEROUS_PATTERNS = [
        re.compile(r"\bDETACH\s+DELETE\b", re.IGNORECASE),
        re.compile(r"\bDROP\b", re.IGNORECASE),
        re.compile(r"\bCALL\s+\{", re.IGNORECASE),  # Subqueries can be dangerous
    ]

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        neo4j_client: OntologyGraphClient | None = None,
        allow_mutations: bool = False,
    ) -> None:
        """
        Initialize the validator.

        Args:
            llm: LLM for self-healing (optional)
            neo4j_client: Neo4j client for EXPLAIN validation
            allow_mutations: Whether to allow CREATE/DELETE queries
        """
        self._llm = llm
        self._client = neo4j_client or get_ontology_client()
        self._allow_mutations = allow_mutations
        self._schema_cache: dict[str, Any] | None = None
        self._healing_chain: Any = None

        if llm:
            self._healing_chain = SELF_HEALING_CHAT_PROMPT | llm | StrOutputParser()

    async def _get_schema(self) -> dict[str, Any]:
        """Get cached database schema."""
        if self._schema_cache is None:
            await self._client.connect()
            self._schema_cache = await self._client.get_schema()
        return self._schema_cache

    def validate_syntax_basic(self, cypher: str) -> ValidationResult:
        """
        Perform basic syntax validation without database connection.

        Checks for common issues like:
        - Missing RETURN clause
        - Unbalanced parentheses/brackets
        - Dangerous operations
        """
        cypher = cypher.strip()

        if not cypher:
            return ValidationResult(
                is_valid=False, error_message="Empty query", error_type="syntax"
            )

        # Check for required clauses
        if not self.REQUIRED_CLAUSES.search(cypher):
            return ValidationResult(
                is_valid=False,
                error_message="Query must contain RETURN, CREATE, MERGE, DELETE, SET, or REMOVE clause",
                error_type="syntax",
                suggestions=["Add a RETURN clause to specify what to output"],
            )

        # Check for dangerous operations
        if not self._allow_mutations:
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern.search(cypher):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Dangerous operation detected: {pattern.pattern}",
                        error_type="security",
                        suggestions=["Remove destructive operations or enable allow_mutations"],
                    )

        # Check balanced parentheses
        if cypher.count("(") != cypher.count(")"):
            return ValidationResult(
                is_valid=False,
                error_message="Unbalanced parentheses",
                error_type="syntax",
                suggestions=["Check that all ( have matching )"],
            )

        # Check balanced brackets
        if cypher.count("[") != cypher.count("]"):
            return ValidationResult(
                is_valid=False,
                error_message="Unbalanced brackets",
                error_type="syntax",
                suggestions=["Check that all [ have matching ]"],
            )

        # Check balanced braces
        if cypher.count("{") != cypher.count("}"):
            return ValidationResult(
                is_valid=False,
                error_message="Unbalanced braces",
                error_type="syntax",
                suggestions=["Check that all { have matching }"],
            )

        # Check for common typos
        if re.search(r"\bMATCN\b|\bRETURN\s*$|\bWHRE\b", cypher, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                error_message="Possible typo in Cypher keywords",
                error_type="syntax",
            )

        return ValidationResult(is_valid=True)

    async def validate_syntax(self, cypher: str) -> ValidationResult:
        """
        Validate Cypher query syntax using Neo4j EXPLAIN.

        Args:
            cypher: Cypher query to validate

        Returns:
            ValidationResult with status and any error details
        """
        # First do basic validation
        basic_result = self.validate_syntax_basic(cypher)
        if not basic_result.is_valid:
            return basic_result

        # Use EXPLAIN to validate against database
        try:
            await self._client.connect()
            await self._client.execute_cypher(f"EXPLAIN {cypher}")
            return ValidationResult(is_valid=True)

        except Exception as e:
            error_str = str(e)

            # Parse error type
            if "SyntaxError" in error_str or "Invalid input" in error_str:
                error_type = "syntax"
            elif "not defined" in error_str.lower() or "unknown" in error_str.lower():
                error_type = "semantic"
            else:
                error_type = "runtime"

            # Extract suggestions from error
            suggestions = self._extract_suggestions_from_error(error_str)

            return ValidationResult(
                is_valid=False,
                error_message=error_str,
                error_type=error_type,
                suggestions=suggestions,
            )

    def _extract_suggestions_from_error(self, error: str) -> list[str]:
        """Extract helpful suggestions from error messages."""
        suggestions = []

        error_lower = error.lower()

        if "unknown function" in error_lower:
            suggestions.append(
                "Check function name spelling and availability in your Neo4j version"
            )

        if "type mismatch" in error_lower:
            suggestions.append("Ensure property types match comparison values")

        if "variable" in error_lower and "not defined" in error_lower:
            suggestions.append("Define variables in MATCH or WITH before using them")

        if "label" in error_lower or "relationship type" in error_lower:
            suggestions.append("Verify label/relationship type exists in schema")

        if "property" in error_lower and (
            "does not exist" in error_lower or "unknown" in error_lower
        ):
            suggestions.append("Check property name against schema")

        return suggestions

    async def validate_against_schema(self, cypher: str) -> ValidationResult:
        """
        Validate that query only uses schema elements that exist.

        Checks node labels, relationship types, and property names.
        """
        schema = await self._get_schema()

        valid_labels = set(schema.get("node_labels", []))
        valid_rel_types = set(schema.get("relationship_types", []))

        # Extract labels from query
        label_pattern = re.compile(r":(\w+)(?:\s*\{|\s*\)|\s*\]|\s*-|\s*<)")
        found_labels = set(label_pattern.findall(cypher))

        # Check labels
        invalid_labels = found_labels - valid_labels - valid_rel_types
        if invalid_labels:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown labels/types: {', '.join(invalid_labels)}",
                error_type="semantic",
                suggestions=[f"Valid labels: {', '.join(sorted(valid_labels)[:10])}..."],
            )

        return ValidationResult(is_valid=True)

    async def validate_and_heal(
        self,
        cypher: str,
        question: str,
        max_retries: int = 3,
    ) -> HealingResult:
        """
        Validate Cypher and attempt self-healing if invalid.

        Uses LLM to regenerate query based on error feedback.

        Args:
            cypher: Original Cypher query
            question: Original natural language question
            max_retries: Maximum healing attempts

        Returns:
            HealingResult with healed query if successful
        """
        if not self._healing_chain:
            # No LLM available, just validate
            result = await self.validate_syntax(cypher)
            return HealingResult(
                success=result.is_valid,
                original_query=cypher,
                healed_query=cypher if result.is_valid else None,
                attempts=1,
                errors=[result.error_message] if result.error_message else [],
            )

        errors: list[str] = []
        current_query = cypher
        schema = await self._get_schema()
        schema_str = format_schema_for_prompt(schema)

        for attempt in range(max_retries + 1):
            # Validate current query
            result = await self.validate_syntax(current_query)

            if result.is_valid:
                # Also check schema compliance
                schema_result = await self.validate_against_schema(current_query)
                if schema_result.is_valid:
                    logger.info(
                        "Query validated",
                        attempts=attempt + 1,
                        healed=attempt > 0,
                    )
                    return HealingResult(
                        success=True,
                        original_query=cypher,
                        healed_query=current_query,
                        attempts=attempt + 1,
                        errors=errors,
                    )
                else:
                    result = schema_result

            # Record error
            if result.error_message:
                errors.append(result.error_message)

            # Stop if max retries reached
            if attempt >= max_retries:
                break

            # Attempt healing
            logger.info(
                "Attempting query healing",
                attempt=attempt + 1,
                error=result.error_message[:100] if result.error_message else "",
            )

            try:
                healed = await self._healing_chain.ainvoke(
                    {
                        "question": question,
                        "failed_query": current_query,
                        "error_message": result.error_message,
                        "schema": schema_str,
                    }
                )

                # Clean up response
                healed = self._clean_cypher_response(healed)
                current_query = healed

            except Exception as e:
                errors.append(f"Healing failed: {str(e)}")
                logger.error("Healing attempt failed", error=str(e))
                break

        logger.warning(
            "Query healing failed",
            attempts=max_retries + 1,
            errors=errors,
        )

        return HealingResult(
            success=False,
            original_query=cypher,
            healed_query=None,
            attempts=max_retries + 1,
            errors=errors,
        )

    def _clean_cypher_response(self, response: str) -> str:
        """Clean up LLM response to extract pure Cypher."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first line (```cypher or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        # Remove any trailing explanations after the query
        # Cypher queries typically end with a clause keyword or closing bracket
        lines = response.split("\n")
        query_lines = []
        for line in lines:
            stripped = line.strip()
            # Stop at obvious non-Cypher content
            if stripped.startswith("//") and "explanation" in stripped.lower():
                break
            if stripped.startswith("Note:") or stripped.startswith("This query"):
                break
            query_lines.append(line)

        return "\n".join(query_lines).strip()


class CypherSanitizer:
    """
    Sanitizes Cypher queries to prevent injection attacks.
    """

    # Patterns that should not appear in user-provided values
    INJECTION_PATTERNS = [
        re.compile(r";\s*(MATCH|CREATE|DELETE|DROP|CALL)", re.IGNORECASE),
        re.compile(r"\]\s*-\s*\["),  # Relationship injection
        re.compile(r"\}\s*\)\s*-"),  # Node property injection
    ]

    @classmethod
    def sanitize_value(cls, value: str) -> str:
        """
        Sanitize a string value for use in Cypher.

        Escapes special characters and validates against injection patterns.
        """
        # Check for injection attempts
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(value):
                raise ValueError("Potential Cypher injection detected in value")

        # Escape special characters
        sanitized = value.replace("\\", "\\\\")
        sanitized = sanitized.replace("'", "\\'")
        sanitized = sanitized.replace('"', '\\"')

        return sanitized

    @classmethod
    def validate_parameter_name(cls, name: str) -> bool:
        """Validate that parameter name is safe."""
        return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))

    @classmethod
    def build_safe_query(
        cls,
        template: str,
        parameters: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """
        Build a parameterized query safely.

        Args:
            template: Query template with $param placeholders
            parameters: Parameter values

        Returns:
            Tuple of (query, validated_parameters)
        """
        validated_params: dict[str, Any] = {}

        for key, value in parameters.items():
            if not cls.validate_parameter_name(key):
                raise ValueError(f"Invalid parameter name: {key}")

            if isinstance(value, str):
                # String values are passed as parameters (safe)
                validated_params[key] = value
            elif isinstance(value, (int, float, bool, type(None))):
                validated_params[key] = value
            elif isinstance(value, list):
                # Validate list contents
                validated_params[key] = [
                    cls.sanitize_value(v) if isinstance(v, str) else v for v in value
                ]
            else:
                raise ValueError(f"Unsupported parameter type: {type(value)}")

        return template, validated_params

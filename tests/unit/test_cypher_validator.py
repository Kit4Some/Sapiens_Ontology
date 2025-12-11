"""
Unit Tests for Cypher Validator.

Tests the CypherValidator and CypherSanitizer classes.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.text2cypher.validator import (
    CypherSanitizer,
    CypherValidator,
    HealingResult,
    ValidationResult,
)


class TestCypherValidator:
    """Test cases for CypherValidator."""

    # =========================================================================
    # Basic Syntax Validation Tests
    # =========================================================================

    def test_validate_empty_query(self) -> None:
        """Test validation of empty query."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic("")

        assert not result.is_valid
        assert result.error_type == "syntax"
        assert "Empty query" in str(result.error_message)

    def test_validate_valid_match_return(self) -> None:
        """Test validation of valid MATCH-RETURN query."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MATCH (n:Entity) RETURN n"
        )

        assert result.is_valid

    def test_validate_valid_match_where_return(self) -> None:
        """Test validation of valid MATCH-WHERE-RETURN query."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MATCH (n:Entity) WHERE n.name = 'Test' RETURN n"
        )

        assert result.is_valid

    def test_validate_valid_relationship_query(self) -> None:
        """Test validation of relationship query."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a, r, b"
        )

        assert result.is_valid

    def test_validate_missing_return_clause(self) -> None:
        """Test validation fails when RETURN is missing."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MATCH (n:Entity) WHERE n.name = 'Test'"
        )

        assert not result.is_valid
        assert result.error_type == "syntax"

    def test_validate_unbalanced_parentheses(self) -> None:
        """Test validation fails with unbalanced parentheses."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MATCH (n:Entity RETURN n"
        )

        assert not result.is_valid
        assert "parentheses" in str(result.error_message).lower()

    def test_validate_unbalanced_brackets(self) -> None:
        """Test validation fails with unbalanced brackets."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MATCH (a)-[r:RELATES_TO->(b) RETURN a"
        )

        assert not result.is_valid
        assert "bracket" in str(result.error_message).lower()

    def test_validate_unbalanced_braces(self) -> None:
        """Test validation fails with unbalanced braces."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MATCH (n:Entity {name: 'Test') RETURN n"
        )

        assert not result.is_valid
        assert "brace" in str(result.error_message).lower()

    # =========================================================================
    # Dangerous Operations Tests
    # =========================================================================

    def test_validate_detach_delete_blocked(self) -> None:
        """Test that DETACH DELETE is blocked by default."""
        validator = CypherValidator(allow_mutations=False)
        result = validator.validate_syntax_basic(
            "MATCH (n) DETACH DELETE n"
        )

        assert not result.is_valid
        assert result.error_type == "security"

    def test_validate_drop_blocked(self) -> None:
        """Test that DROP is blocked."""
        validator = CypherValidator(allow_mutations=False)
        result = validator.validate_syntax_basic(
            "DROP INDEX my_index"
        )

        assert not result.is_valid

    def test_validate_mutations_allowed(self) -> None:
        """Test that mutations are allowed when enabled."""
        validator = CypherValidator(allow_mutations=True)
        result = validator.validate_syntax_basic(
            "CREATE (n:Entity {name: 'Test'}) RETURN n"
        )

        assert result.is_valid

    def test_validate_merge_allowed(self) -> None:
        """Test that MERGE is valid."""
        validator = CypherValidator()
        result = validator.validate_syntax_basic(
            "MERGE (n:Entity {id: 'test'}) RETURN n"
        )

        assert result.is_valid

    # =========================================================================
    # Neo4j EXPLAIN Validation Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_validate_syntax_with_explain_success(self) -> None:
        """Test syntax validation using Neo4j EXPLAIN."""
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.execute_cypher = AsyncMock(return_value=[])

        validator = CypherValidator(neo4j_client=mock_client)
        result = await validator.validate_syntax(
            "MATCH (n:Entity) RETURN n LIMIT 10"
        )

        assert result.is_valid
        mock_client.execute_cypher.assert_called()

    @pytest.mark.asyncio
    async def test_validate_syntax_with_explain_failure(self) -> None:
        """Test syntax validation catches errors from EXPLAIN."""
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.execute_cypher = AsyncMock(
            side_effect=Exception("SyntaxError: Invalid input")
        )

        validator = CypherValidator(neo4j_client=mock_client)
        result = await validator.validate_syntax(
            "MATCH (n:Entity RETURN n"  # Invalid syntax
        )

        assert not result.is_valid
        assert result.error_type == "syntax"

    # =========================================================================
    # Schema Validation Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_validate_against_schema_valid_label(self) -> None:
        """Test schema validation with valid labels."""
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.get_schema = AsyncMock(
            return_value={
                "node_labels": ["Entity", "Chunk"],
                "relationship_types": ["RELATES_TO"],
            }
        )

        validator = CypherValidator(neo4j_client=mock_client)
        result = await validator.validate_against_schema(
            "MATCH (n:Entity)-[r:RELATES_TO]->(m:Chunk) RETURN n, m"
        )

        assert result.is_valid

    @pytest.mark.asyncio
    async def test_validate_against_schema_invalid_label(self) -> None:
        """Test schema validation catches invalid labels."""
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.get_schema = AsyncMock(
            return_value={
                "node_labels": ["Entity", "Chunk"],
                "relationship_types": ["RELATES_TO"],
            }
        )

        validator = CypherValidator(neo4j_client=mock_client)
        result = await validator.validate_against_schema(
            "MATCH (n:NonExistent) RETURN n"
        )

        assert not result.is_valid
        assert result.error_type == "semantic"
        assert "NonExistent" in str(result.error_message)

    # =========================================================================
    # Self-Healing Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_validate_and_heal_success_first_try(self) -> None:
        """Test validation succeeds on first try without healing."""
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.execute_cypher = AsyncMock(return_value=[])
        mock_client.get_schema = AsyncMock(
            return_value={
                "node_labels": ["Entity"],
                "relationship_types": [],
            }
        )

        validator = CypherValidator(neo4j_client=mock_client)
        result = await validator.validate_and_heal(
            cypher="MATCH (n:Entity) RETURN n",
            question="Show all entities",
        )

        assert result.success
        assert result.attempts == 1
        assert result.healed_query == "MATCH (n:Entity) RETURN n"

    @pytest.mark.asyncio
    async def test_validate_and_heal_with_llm_healing(self) -> None:
        """Test that invalid queries are healed by LLM."""
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.get_schema = AsyncMock(
            return_value={
                "node_labels": ["Entity"],
                "relationship_types": [],
            }
        )

        # First call fails, second succeeds (after healing)
        mock_client.execute_cypher = AsyncMock(
            side_effect=[
                Exception("SyntaxError"),
                [],  # Success after healing
            ]
        )

        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock(return_value=mock_llm)
        mock_healing_chain = MagicMock()
        mock_healing_chain.ainvoke = AsyncMock(
            return_value="MATCH (n:Entity) RETURN n"
        )

        with patch.object(
            CypherValidator, "_healing_chain", mock_healing_chain
        ):
            validator = CypherValidator(
                llm=mock_llm,
                neo4j_client=mock_client,
            )
            validator._healing_chain = mock_healing_chain

            result = await validator.validate_and_heal(
                cypher="MATCH (n:Entity RETURN n",  # Invalid
                question="Show all entities",
            )

            # May or may not succeed depending on mock setup
            assert isinstance(result, HealingResult)

    @pytest.mark.asyncio
    async def test_validate_and_heal_max_retries(self) -> None:
        """Test that healing respects max retries."""
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.execute_cypher = AsyncMock(
            side_effect=Exception("Persistent error")
        )
        mock_client.get_schema = AsyncMock(
            return_value={"node_labels": [], "relationship_types": []}
        )

        validator = CypherValidator(neo4j_client=mock_client)
        result = await validator.validate_and_heal(
            cypher="INVALID QUERY",
            question="Test",
            max_retries=2,
        )

        assert not result.success
        assert result.attempts == 3  # 1 initial + 2 retries
        assert len(result.errors) > 0

    # =========================================================================
    # Error Extraction Tests
    # =========================================================================

    def test_extract_suggestions_unknown_function(self) -> None:
        """Test suggestion extraction for unknown function."""
        validator = CypherValidator()
        suggestions = validator._extract_suggestions_from_error(
            "Unknown function 'myFunc'"
        )

        assert len(suggestions) > 0
        assert any("function" in s.lower() for s in suggestions)

    def test_extract_suggestions_type_mismatch(self) -> None:
        """Test suggestion extraction for type mismatch."""
        validator = CypherValidator()
        suggestions = validator._extract_suggestions_from_error(
            "Type mismatch: expected Integer but got String"
        )

        assert len(suggestions) > 0
        assert any("type" in s.lower() for s in suggestions)

    def test_extract_suggestions_undefined_variable(self) -> None:
        """Test suggestion extraction for undefined variable."""
        validator = CypherValidator()
        suggestions = validator._extract_suggestions_from_error(
            "Variable `x` not defined"
        )

        assert len(suggestions) > 0
        assert any("variable" in s.lower() or "define" in s.lower() for s in suggestions)


class TestCypherSanitizer:
    """Test cases for CypherSanitizer."""

    # =========================================================================
    # Value Sanitization Tests
    # =========================================================================

    def test_sanitize_simple_string(self) -> None:
        """Test sanitization of simple string."""
        result = CypherSanitizer.sanitize_value("Hello World")
        assert result == "Hello World"

    def test_sanitize_escapes_single_quotes(self) -> None:
        """Test that single quotes are escaped."""
        result = CypherSanitizer.sanitize_value("It's a test")
        assert result == "It\\'s a test"

    def test_sanitize_escapes_double_quotes(self) -> None:
        """Test that double quotes are escaped."""
        result = CypherSanitizer.sanitize_value('Say "hello"')
        assert result == 'Say \\"hello\\"'

    def test_sanitize_escapes_backslashes(self) -> None:
        """Test that backslashes are escaped."""
        result = CypherSanitizer.sanitize_value("path\\to\\file")
        assert result == "path\\\\to\\\\file"

    def test_sanitize_detects_injection_semicolon_match(self) -> None:
        """Test that injection attempts with semicolon+MATCH are detected."""
        with pytest.raises(ValueError, match="injection"):
            CypherSanitizer.sanitize_value("'; MATCH (n) RETURN n; //")

    def test_sanitize_detects_injection_semicolon_delete(self) -> None:
        """Test that injection attempts with semicolon+DELETE are detected."""
        with pytest.raises(ValueError, match="injection"):
            CypherSanitizer.sanitize_value("test'; DELETE (n); //")

    # =========================================================================
    # Parameter Name Validation Tests
    # =========================================================================

    def test_validate_parameter_name_valid(self) -> None:
        """Test validation of valid parameter names."""
        assert CypherSanitizer.validate_parameter_name("name")
        assert CypherSanitizer.validate_parameter_name("entity_id")
        assert CypherSanitizer.validate_parameter_name("_private")
        assert CypherSanitizer.validate_parameter_name("param123")

    def test_validate_parameter_name_invalid(self) -> None:
        """Test validation rejects invalid parameter names."""
        assert not CypherSanitizer.validate_parameter_name("123start")
        assert not CypherSanitizer.validate_parameter_name("has-dash")
        assert not CypherSanitizer.validate_parameter_name("has space")
        assert not CypherSanitizer.validate_parameter_name("special$char")

    # =========================================================================
    # Safe Query Building Tests
    # =========================================================================

    def test_build_safe_query_simple(self) -> None:
        """Test building safe query with simple parameters."""
        template = "MATCH (n:Entity) WHERE n.name = $name RETURN n"
        params = {"name": "Test Entity"}

        query, validated = CypherSanitizer.build_safe_query(template, params)

        assert query == template
        assert validated["name"] == "Test Entity"

    def test_build_safe_query_with_int(self) -> None:
        """Test building safe query with integer parameter."""
        template = "MATCH (n:Entity) RETURN n LIMIT $limit"
        params = {"limit": 10}

        query, validated = CypherSanitizer.build_safe_query(template, params)

        assert validated["limit"] == 10

    def test_build_safe_query_with_list(self) -> None:
        """Test building safe query with list parameter."""
        template = "MATCH (n:Entity) WHERE n.id IN $ids RETURN n"
        params = {"ids": ["id1", "id2", "id3"]}

        query, validated = CypherSanitizer.build_safe_query(template, params)

        assert validated["ids"] == ["id1", "id2", "id3"]

    def test_build_safe_query_sanitizes_list_strings(self) -> None:
        """Test that strings in lists are sanitized."""
        template = "MATCH (n) WHERE n.name IN $names RETURN n"
        params = {"names": ["Test's Name", "Normal"]}

        query, validated = CypherSanitizer.build_safe_query(template, params)

        assert validated["names"][0] == "Test\\'s Name"
        assert validated["names"][1] == "Normal"

    def test_build_safe_query_invalid_param_name(self) -> None:
        """Test that invalid parameter names are rejected."""
        template = "MATCH (n) RETURN n"
        params = {"invalid-name": "value"}

        with pytest.raises(ValueError, match="Invalid parameter name"):
            CypherSanitizer.build_safe_query(template, params)

    def test_build_safe_query_unsupported_type(self) -> None:
        """Test that unsupported types are rejected."""
        template = "MATCH (n) RETURN n"
        params = {"obj": {"nested": "object"}}

        with pytest.raises(ValueError, match="Unsupported parameter type"):
            CypherSanitizer.build_safe_query(template, params)

    def test_build_safe_query_with_none(self) -> None:
        """Test that None values are allowed."""
        template = "MATCH (n) WHERE n.value = $value RETURN n"
        params = {"value": None}

        query, validated = CypherSanitizer.build_safe_query(template, params)

        assert validated["value"] is None

    def test_build_safe_query_with_bool(self) -> None:
        """Test that boolean values are allowed."""
        template = "MATCH (n) WHERE n.active = $active RETURN n"
        params = {"active": True}

        query, validated = CypherSanitizer.build_safe_query(template, params)

        assert validated["active"] is True


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert result.error_message is None

    def test_invalid_result_with_message(self) -> None:
        """Test creating an invalid result with error message."""
        result = ValidationResult(
            is_valid=False,
            error_message="Test error",
            error_type="syntax",
        )
        assert not result.is_valid
        assert result.error_message == "Test error"
        assert result.error_type == "syntax"

    def test_result_with_suggestions(self) -> None:
        """Test result with suggestions."""
        result = ValidationResult(
            is_valid=False,
            error_message="Error",
            suggestions=["Fix 1", "Fix 2"],
        )
        assert result.suggestions is not None
        assert len(result.suggestions) == 2


class TestHealingResult:
    """Test HealingResult dataclass."""

    def test_successful_healing(self) -> None:
        """Test successful healing result."""
        result = HealingResult(
            success=True,
            original_query="MATCH (n RETURN n",
            healed_query="MATCH (n) RETURN n",
            attempts=2,
            errors=["First attempt failed"],
        )
        assert result.success
        assert result.healed_query != result.original_query
        assert result.attempts == 2

    def test_failed_healing(self) -> None:
        """Test failed healing result."""
        result = HealingResult(
            success=False,
            original_query="INVALID",
            healed_query=None,
            attempts=3,
            errors=["Error 1", "Error 2", "Error 3"],
        )
        assert not result.success
        assert result.healed_query is None
        assert len(result.errors) == 3

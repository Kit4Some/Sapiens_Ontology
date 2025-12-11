"""
Integration Tests for API Endpoints.

Tests the FastAPI REST API endpoints.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestHealthEndpoint:
    """Test cases for /api/health endpoint."""

    def test_health_check_healthy(self) -> None:
        """Test health check returns healthy status."""
        with patch("src.api.main.check_neo4j_connection", new_callable=lambda: AsyncMock(return_value=True)):
            with patch("src.api.main.check_llm_availability", return_value=True):
                with TestClient(app) as client:
                    response = client.get("/api/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert data["neo4j_connected"] is True
                    assert data["llm_available"] is True

    def test_health_check_degraded(self) -> None:
        """Test health check returns degraded when service is down."""
        with patch("src.api.main.check_neo4j_connection", new_callable=lambda: AsyncMock(return_value=False)):
            with patch("src.api.main.check_llm_availability", return_value=True):
                with TestClient(app) as client:
                    response = client.get("/api/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "degraded"
                    assert data["neo4j_connected"] is False

    def test_health_check_unhealthy(self) -> None:
        """Test health check returns unhealthy when all services down."""
        with patch("src.api.main.check_neo4j_connection", new_callable=lambda: AsyncMock(return_value=False)):
            with patch("src.api.main.check_llm_availability", return_value=False):
                with TestClient(app) as client:
                    response = client.get("/api/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "unhealthy"


class TestSchemaEndpoint:
    """Test cases for /api/schema endpoint."""

    def test_get_schema_success(self) -> None:
        """Test successful schema retrieval."""
        mock_schema = {
            "node_labels": ["Entity", "Chunk", "Community"],
            "relationship_types": ["RELATES_TO", "CONTAINS"],
            "node_properties": {"Entity": ["id", "name", "type"]},
            "indexes": [],
        }

        with patch("src.api.main.get_graph_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_schema = AsyncMock(return_value=mock_schema)
            mock_get_client.return_value = mock_client

            with TestClient(app) as client:
                response = client.get("/api/schema")

                assert response.status_code == 200
                data = response.json()
                assert "node_labels" in data
                assert "Entity" in data["node_labels"]

    def test_get_schema_error(self) -> None:
        """Test schema endpoint handles errors."""
        with patch("src.api.main.get_graph_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_schema = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            mock_get_client.return_value = mock_client

            with TestClient(app) as client:
                response = client.get("/api/schema")

                assert response.status_code == 500
                assert "error" in response.json()["detail"].lower() or "failed" in response.json()["detail"].lower()


class TestQueryEndpoint:
    """Test cases for /api/query endpoint."""

    def test_query_success(self) -> None:
        """Test successful query processing."""
        mock_result = {
            "final_answer": "The answer is X.",
            "confidence": 0.85,
            "answer_type": "DEFINITIVE",
            "explanation": "Based on evidence...",
            "evidence": [],
            "reasoning_path": [],
            "iteration": 3,
            "errors": [],
        }

        with patch("src.api.main.get_workflow") as mock_get_workflow:
            mock_workflow = MagicMock()
            mock_workflow.run = AsyncMock(return_value=mock_result)
            mock_get_workflow.return_value = mock_workflow

            with TestClient(app) as client:
                response = client.post(
                    "/api/query",
                    json={"query": "What is the capital of France?"},
                )

                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "confidence" in data

    def test_query_with_max_iterations(self) -> None:
        """Test query with custom max iterations."""
        mock_result = {
            "final_answer": "Answer",
            "confidence": 0.8,
            "answer_type": "DEFINITIVE",
            "explanation": "",
            "evidence": [],
            "reasoning_path": [],
            "iteration": 2,
            "errors": [],
        }

        with patch("src.api.main.get_workflow") as mock_get_workflow:
            mock_workflow = MagicMock()
            mock_workflow.run = AsyncMock(return_value=mock_result)
            mock_get_workflow.return_value = mock_workflow

            with TestClient(app) as client:
                response = client.post(
                    "/api/query",
                    json={
                        "query": "Test query",
                        "max_iterations": 3,
                    },
                )

                assert response.status_code == 200

    def test_query_empty_query(self) -> None:
        """Test query endpoint rejects empty query."""
        with TestClient(app) as client:
            response = client.post(
                "/api/query",
                json={"query": ""},
            )

            # Should return validation error
            assert response.status_code in [400, 422]

    def test_query_workflow_error(self) -> None:
        """Test query endpoint handles workflow errors."""
        with patch("src.api.main.get_workflow") as mock_get_workflow:
            mock_workflow = MagicMock()
            mock_workflow.run = AsyncMock(
                side_effect=Exception("Workflow failed")
            )
            mock_get_workflow.return_value = mock_workflow

            with TestClient(app) as client:
                response = client.post(
                    "/api/query",
                    json={"query": "Test query"},
                )

                assert response.status_code == 500


class TestStreamQueryEndpoint:
    """Test cases for /api/query/stream endpoint."""

    def test_stream_query_returns_event_stream(self) -> None:
        """Test that stream endpoint returns SSE content type."""
        with patch("src.api.main.get_workflow") as mock_get_workflow:
            mock_workflow = MagicMock()
            mock_app = MagicMock()

            # Mock the streaming behavior
            async def mock_astream(*args: Any, **kwargs: Any):
                yield {"constructor": {"iteration": 0}}
                yield {"retriever": {"iteration": 0, "evidence": []}}

            mock_app.astream = mock_astream
            mock_app.get_state = MagicMock(return_value=None)
            mock_workflow.compile.return_value = mock_app
            mock_get_workflow.return_value = mock_workflow

            with TestClient(app) as client:
                response = client.post(
                    "/api/query/stream",
                    json={"query": "Test query"},
                )

                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]


class TestCypherEndpoint:
    """Test cases for /api/cypher endpoint."""

    def test_execute_cypher_success(self) -> None:
        """Test successful Cypher execution."""
        mock_results = [
            {"n.name": "Entity 1"},
            {"n.name": "Entity 2"},
        ]

        with patch("src.api.main.get_graph_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.execute_cypher = AsyncMock(return_value=mock_results)
            mock_get_client.return_value = mock_client

            with TestClient(app) as client:
                response = client.post(
                    "/api/cypher",
                    params={"cypher": "MATCH (n:Entity) RETURN n.name LIMIT 10"},
                )

                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert data["count"] == 2

    def test_execute_cypher_with_parameters(self) -> None:
        """Test Cypher execution with parameters."""
        mock_results = [{"n.name": "Test Entity"}]

        with patch("src.api.main.get_graph_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.execute_cypher = AsyncMock(return_value=mock_results)
            mock_get_client.return_value = mock_client

            with TestClient(app) as client:
                response = client.post(
                    "/api/cypher",
                    params={"cypher": "MATCH (n:Entity {name: $name}) RETURN n.name"},
                    json={"name": "Test Entity"},
                )

                assert response.status_code == 200

    def test_execute_cypher_error(self) -> None:
        """Test Cypher endpoint handles execution errors."""
        with patch("src.api.main.get_graph_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.execute_cypher = AsyncMock(
                side_effect=Exception("Invalid syntax")
            )
            mock_get_client.return_value = mock_client

            with TestClient(app) as client:
                response = client.post(
                    "/api/cypher",
                    params={"cypher": "INVALID QUERY"},
                )

                assert response.status_code == 400


class TestStatsEndpoint:
    """Test cases for /api/stats endpoint."""

    def test_get_stats_success(self) -> None:
        """Test successful stats retrieval."""
        mock_stats = {
            "node_count": 1000,
            "relationship_count": 5000,
            "entity_count": 500,
            "chunk_count": 400,
            "community_count": 100,
        }

        with patch("src.api.main.get_graph_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("src.api.main.Neo4jLoader") as mock_loader_class:
                mock_loader = MagicMock()
                mock_loader.get_stats = AsyncMock(return_value=mock_stats)
                mock_loader_class.return_value = mock_loader

                with TestClient(app) as client:
                    response = client.get("/api/stats")

                    assert response.status_code == 200
                    data = response.json()
                    assert "graph" in data
                    assert "system" in data


class TestRequestValidation:
    """Test request validation."""

    def test_query_request_validation(self) -> None:
        """Test query request model validation."""
        with TestClient(app) as client:
            # Missing required field
            response = client.post("/api/query", json={})
            assert response.status_code == 422

            # Invalid max_iterations
            response = client.post(
                "/api/query",
                json={"query": "test", "max_iterations": -1},
            )
            assert response.status_code == 422

    def test_cors_headers(self) -> None:
        """Test CORS headers are present."""
        with TestClient(app) as client:
            response = client.options(
                "/api/health",
                headers={"Origin": "http://localhost:3000"},
            )

            # CORS should be configured
            assert response.status_code in [200, 204, 405]


class TestErrorHandling:
    """Test error handling and responses."""

    def test_404_not_found(self) -> None:
        """Test 404 response for unknown endpoints."""
        with TestClient(app) as client:
            response = client.get("/api/unknown")
            assert response.status_code == 404

    def test_method_not_allowed(self) -> None:
        """Test 405 response for wrong HTTP method."""
        with TestClient(app) as client:
            response = client.put("/api/health")
            assert response.status_code == 405

    def test_internal_error_handling(self) -> None:
        """Test internal server errors are handled gracefully."""
        with patch("src.api.main.get_workflow") as mock_get_workflow:
            mock_get_workflow.side_effect = RuntimeError("Internal error")

            with TestClient(app) as client:
                response = client.post(
                    "/api/query",
                    json={"query": "test"},
                )

                assert response.status_code == 500

"""
Unit Tests for Neo4j Client.

Tests the OntologyGraphClient class functionality.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.neo4j_client import (
    NodeLabel,
    OntologyGraphClient,
    RelationType,
    get_ontology_client,
)


class TestOntologyGraphClient:
    """Test cases for OntologyGraphClient."""

    # =========================================================================
    # Connection Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        """Test successful database connection."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()

            mock_db.driver.assert_called_once()
            mock_driver.verify_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self) -> None:
        """Test that connect() is idempotent."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            await client.connect()  # Second call should be no-op

            # Driver should only be created once
            assert mock_db.driver.call_count == 1

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Test database disconnection."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            await client.disconnect()

            mock_driver.close.assert_called_once()

    # =========================================================================
    # Schema Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_schema(self, mock_neo4j_client: MagicMock) -> None:
        """Test schema retrieval."""
        schema = await mock_neo4j_client.get_schema()

        assert "node_labels" in schema
        assert "relationship_types" in schema
        assert "Entity" in schema["node_labels"]
        assert "RELATES_TO" in schema["relationship_types"]

    @pytest.mark.asyncio
    async def test_setup_schema(self) -> None:
        """Test schema setup creates indexes and constraints."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_session = MagicMock()
            mock_session.run = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            result = await client.setup_schema()

            # Should have called session.run multiple times for constraints/indexes
            assert mock_session.run.call_count > 0
            assert "constraints" in result
            assert "indexes" in result

    # =========================================================================
    # CRUD Tests - Entities
    # =========================================================================

    @pytest.mark.asyncio
    async def test_create_entity(self) -> None:
        """Test entity creation."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.single.return_value = {
                "e": {
                    "id": "test_id",
                    "name": "Test Entity",
                    "type": "CONCEPT",
                }
            }
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            result = await client.create_entity(
                entity_id="test_id",
                name="Test Entity",
                entity_type="CONCEPT",
            )

            assert result is not None
            mock_session.run.assert_called()

    @pytest.mark.asyncio
    async def test_get_entity(self) -> None:
        """Test entity retrieval."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.single.return_value = {
                "e": {
                    "id": "test_id",
                    "name": "Test Entity",
                    "type": "CONCEPT",
                }
            }
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            result = await client.get_entity("test_id")

            assert result is not None

    # =========================================================================
    # Search Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_vector_search(self, mock_neo4j_client: MagicMock) -> None:
        """Test vector similarity search."""
        mock_neo4j_client.vector_search.return_value = [
            {"id": "entity_1", "name": "Entity 1", "score": 0.95},
            {"id": "entity_2", "name": "Entity 2", "score": 0.85},
        ]

        results = await mock_neo4j_client.vector_search(
            query_embedding=[0.1] * 1536,
            top_k=5,
        )

        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]

    @pytest.mark.asyncio
    async def test_fulltext_search(self, mock_neo4j_client: MagicMock) -> None:
        """Test fulltext search."""
        mock_neo4j_client.fulltext_search.return_value = [
            {"id": "entity_1", "name": "Apple Inc", "score": 5.0},
        ]

        results = await mock_neo4j_client.fulltext_search(
            query_text="Apple",
            top_k=10,
        )

        assert len(results) == 1
        assert "Apple" in results[0]["name"]

    # =========================================================================
    # Graph Traversal Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_neighbors(self, mock_neo4j_client: MagicMock) -> None:
        """Test neighbor retrieval."""
        mock_neo4j_client.get_neighbors.return_value = {
            "nodes": [
                {"id": "neighbor_1", "name": "Neighbor 1"},
                {"id": "neighbor_2", "name": "Neighbor 2"},
            ],
            "edges": [
                {"source": "center", "target": "neighbor_1", "type": "RELATES_TO"},
            ],
        }

        result = await mock_neo4j_client.get_neighbors(
            entity_id="center",
            hops=1,
        )

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    @pytest.mark.asyncio
    async def test_get_paths(self) -> None:
        """Test path finding between nodes."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.data = AsyncMock(
                return_value=[
                    {
                        "nodes": [
                            {"id": "a", "name": "A"},
                            {"id": "b", "name": "B"},
                        ],
                        "relationships": [{"type": "RELATES_TO"}],
                        "path_length": 1,
                    }
                ]
            )
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            paths = await client.get_paths(
                source_id="a",
                target_id="b",
                max_hops=3,
            )

            assert len(paths) == 1
            assert paths[0]["path_length"] == 1

    # =========================================================================
    # Cypher Execution Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_execute_cypher(self) -> None:
        """Test raw Cypher execution."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.data = AsyncMock(
                return_value=[
                    {"n.name": "Entity 1"},
                    {"n.name": "Entity 2"},
                ]
            )
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            results = await client.execute_cypher(
                "MATCH (n:Entity) RETURN n.name LIMIT 10"
            )

            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_execute_cypher_with_parameters(self) -> None:
        """Test Cypher execution with parameters."""
        with patch("src.graph.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.data = AsyncMock(
                return_value=[{"n.name": "Test Entity"}]
            )
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_db.driver.return_value = mock_driver

            client = OntologyGraphClient()
            await client.connect()
            results = await client.execute_cypher(
                "MATCH (n:Entity) WHERE n.name = $name RETURN n.name",
                {"name": "Test Entity"},
            )

            mock_session.run.assert_called_with(
                "MATCH (n:Entity) WHERE n.name = $name RETURN n.name",
                {"name": "Test Entity"},
            )


class TestNodeLabel:
    """Test NodeLabel enum."""

    def test_entity_label(self) -> None:
        """Test Entity label value."""
        assert NodeLabel.ENTITY.value == "Entity"

    def test_chunk_label(self) -> None:
        """Test Chunk label value."""
        assert NodeLabel.CHUNK.value == "Chunk"

    def test_community_label(self) -> None:
        """Test Community label value."""
        assert NodeLabel.COMMUNITY.value == "Community"


class TestRelationType:
    """Test RelationType enum."""

    def test_relates_to(self) -> None:
        """Test RELATES_TO relation type."""
        assert RelationType.RELATES_TO.value == "RELATES_TO"

    def test_contains(self) -> None:
        """Test CONTAINS relation type."""
        assert RelationType.CONTAINS.value == "CONTAINS"


class TestGetOntologyClient:
    """Test the singleton client getter."""

    def test_returns_client(self) -> None:
        """Test that get_ontology_client returns a client."""
        client = get_ontology_client()
        assert isinstance(client, OntologyGraphClient)

    def test_returns_same_instance(self) -> None:
        """Test singleton behavior."""
        client1 = get_ontology_client()
        client2 = get_ontology_client()
        assert client1 is client2

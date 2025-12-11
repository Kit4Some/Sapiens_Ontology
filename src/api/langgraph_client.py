"""
LangGraph Server Client.

Client for communicating with the LangGraph Server API.
"""

import os
from typing import Any, AsyncGenerator

import httpx
import structlog

logger = structlog.get_logger(__name__)

LANGGRAPH_SERVER_URL = os.getenv("LANGGRAPH_SERVER_URL", "http://langgraph:8000")


class LangGraphClient:
    """Client for LangGraph Server API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or LANGGRAPH_SERVER_URL
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(300.0),  # 5 minute timeout for reasoning
            )
        return self._client

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if LangGraph Server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning("LangGraph health check failed", error=str(e))
            return False

    async def run_macer(
        self,
        query: str,
        thread_id: str | None = None,
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """
        Run MACER workflow via LangGraph Server.

        Args:
            query: Natural language question
            thread_id: Optional thread ID for checkpointing
            max_iterations: Maximum reasoning iterations

        Returns:
            Final state with answer
        """
        client = await self._get_client()

        # Create initial input
        input_data = {
            "original_query": query,
            "current_query": query,
            "query_history": [],
            "topic_entities": [],
            "retrieved_entities": [],
            "current_subgraph": {"nodes": [], "edges": [], "center_entity_id": None},
            "subgraph_history": [],
            "evidence": [],
            "evidence_rankings": {},
            "reasoning_path": [],
            "sufficiency_score": 0.0,
            "iteration": 0,
            "max_iterations": max_iterations,
            "should_terminate": False,
            "final_answer": None,
            "confidence": 0.0,
            "explanation": "",
            "pipeline_id": thread_id or "",
            "errors": [],
            "metadata": {},
        }

        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        logger.info("Calling LangGraph Server", query=query[:50], thread_id=thread_id)

        try:
            # Call the graph endpoint
            response = await client.post(
                "/graphs/macer/invoke",
                json={
                    "input": input_data,
                    "config": config,
                },
            )
            response.raise_for_status()
            result = response.json()

            logger.info(
                "LangGraph Server response received",
                confidence=result.get("output", {}).get("confidence", 0),
            )

            return result.get("output", {})

        except httpx.HTTPStatusError as e:
            logger.error("LangGraph Server HTTP error", status=e.response.status_code)
            raise
        except Exception as e:
            logger.error("LangGraph Server call failed", error=str(e))
            raise

    async def stream_macer(
        self,
        query: str,
        thread_id: str | None = None,
        max_iterations: int = 5,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream MACER workflow via LangGraph Server.

        Args:
            query: Natural language question
            thread_id: Optional thread ID for checkpointing
            max_iterations: Maximum reasoning iterations

        Yields:
            State updates from each node
        """
        client = await self._get_client()

        input_data = {
            "original_query": query,
            "current_query": query,
            "query_history": [],
            "topic_entities": [],
            "retrieved_entities": [],
            "current_subgraph": {"nodes": [], "edges": [], "center_entity_id": None},
            "subgraph_history": [],
            "evidence": [],
            "evidence_rankings": {},
            "reasoning_path": [],
            "sufficiency_score": 0.0,
            "iteration": 0,
            "max_iterations": max_iterations,
            "should_terminate": False,
            "final_answer": None,
            "confidence": 0.0,
            "explanation": "",
            "pipeline_id": thread_id or "",
            "errors": [],
            "metadata": {},
        }

        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        logger.info("Streaming from LangGraph Server", query=query[:50])

        try:
            async with client.stream(
                "POST",
                "/graphs/macer/stream",
                json={
                    "input": input_data,
                    "config": config,
                    "stream_mode": "updates",
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        data = json.loads(line[6:])
                        yield data

        except Exception as e:
            logger.error("LangGraph Server stream failed", error=str(e))
            raise


# Singleton client instance
_langgraph_client: LangGraphClient | None = None


def get_langgraph_client() -> LangGraphClient:
    """Get singleton LangGraph client."""
    global _langgraph_client
    if _langgraph_client is None:
        _langgraph_client = LangGraphClient()
    return _langgraph_client

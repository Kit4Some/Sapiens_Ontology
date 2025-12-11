"""
Benchmark Scenarios.

Predefined scenarios for testing system performance:
- Query latency at various graph sizes
- Ingestion throughput
- Concurrent load handling
- Cache effectiveness
"""

import asyncio
import random
import time
import uuid
from typing import Any

import structlog

from src.benchmark.runner import BenchmarkScenario

logger = structlog.get_logger(__name__)


class QueryLatencyScenario(BenchmarkScenario):
    """
    Benchmark query latency at various graph sizes.

    Tests:
    - Simple entity lookups
    - Full-text search
    - Vector similarity search
    - Multi-hop traversals
    """

    name = "query_latency"
    description = "Measure query latency for different query types"

    def __init__(
        self,
        query_types: list[str] | None = None,
        sample_queries: list[str] | None = None,
    ):
        self.query_types = query_types or ["fulltext", "entity_lookup", "vector"]
        self.sample_queries = sample_queries or [
            "What is machine learning?",
            "How does neural network work?",
            "Explain natural language processing",
            "What are knowledge graphs?",
            "Define artificial intelligence",
        ]
        self._client = None
        self._current_query_type = "fulltext"

    async def setup(self) -> None:
        """Initialize Neo4j client."""
        from src.graph.neo4j_client import get_ontology_client
        self._client = get_ontology_client()
        await self._client.connect()

    async def run_iteration(self) -> tuple[bool, float, dict[str, Any]]:
        """Run a single query and measure latency."""
        # Select random query type and query
        self._current_query_type = random.choice(self.query_types)
        query = random.choice(self.sample_queries)

        start_time = time.perf_counter()

        try:
            if self._current_query_type == "fulltext":
                results = await self._client.fulltext_search(
                    query_text=query,
                    top_k=10,
                    min_score=0.1,
                )
                result_count = len(results)

            elif self._current_query_type == "entity_lookup":
                # Simple entity lookup by name pattern
                cypher = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN e
                LIMIT 10
                """
                results = await self._client.execute_cypher(
                    cypher, {"query": query.split()[0]}
                )
                result_count = len(results)

            elif self._current_query_type == "vector":
                # Skip if no embeddings available
                try:
                    from langchain_openai import OpenAIEmbeddings
                    from src.config.settings import get_settings

                    settings = get_settings()
                    api_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None

                    if api_key:
                        embeddings = OpenAIEmbeddings(api_key=api_key)
                        query_embedding = await embeddings.aembed_query(query)
                        results = await self._client.vector_search(
                            embedding=query_embedding,
                            top_k=10,
                        )
                        result_count = len(results)
                    else:
                        result_count = 0
                except Exception:
                    result_count = 0

            else:
                # Multi-hop query
                cypher = """
                MATCH (e:Entity)-[r*1..2]-(related:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN e, related
                LIMIT 20
                """
                results = await self._client.execute_cypher(
                    cypher, {"query": query.split()[0]}
                )
                result_count = len(results)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return True, latency_ms, {
                "query_type": self._current_query_type,
                "result_count": result_count,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return False, latency_ms, {
                "query_type": self._current_query_type,
                "error": str(e),
            }

    async def teardown(self) -> None:
        """Close Neo4j client."""
        if self._client:
            await self._client.close()


class IngestionThroughputScenario(BenchmarkScenario):
    """
    Benchmark document ingestion throughput.

    Tests:
    - Chunking speed
    - Entity extraction throughput
    - Embedding generation rate
    - Graph loading speed
    """

    name = "ingestion_throughput"
    description = "Measure document ingestion throughput"

    def __init__(
        self,
        document_size_chars: int = 5000,
        documents_per_iteration: int = 1,
    ):
        self.document_size_chars = document_size_chars
        self.documents_per_iteration = documents_per_iteration
        self._sample_text = self._generate_sample_text()

    def _generate_sample_text(self) -> str:
        """Generate sample document text."""
        paragraphs = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "Knowledge graphs represent real-world entities and the relationships between them. They are used to integrate information from multiple sources and to answer complex queries that involve multiple entities.",
            "Ontologies provide a shared vocabulary and formal specification of concepts and relationships within a domain. They enable semantic interoperability and knowledge sharing across systems.",
        ]

        # Repeat to reach desired size
        text = ""
        while len(text) < self.document_size_chars:
            text += random.choice(paragraphs) + "\n\n"

        return text[:self.document_size_chars]

    async def setup(self) -> None:
        """Initialize pipeline components."""
        pass

    async def run_iteration(self) -> tuple[bool, float, dict[str, Any]]:
        """Run ingestion and measure throughput."""
        from src.sddi.pipeline import SDDIPipeline

        start_time = time.perf_counter()

        try:
            pipeline = SDDIPipeline()

            chunks_count = 0
            entities_count = 0

            for i in range(self.documents_per_iteration):
                result = await pipeline.process(
                    raw_data=self._sample_text,
                    source=f"benchmark_doc_{uuid.uuid4().hex[:8]}",
                    skip_loading=True,  # Skip Neo4j loading for pure throughput test
                )

                chunks_count += result.get("chunks_count", 0)
                entities_count += result.get("entities_count", 0)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return True, latency_ms, {
                "documents_processed": self.documents_per_iteration,
                "total_chunks": chunks_count,
                "total_entities": entities_count,
                "chars_processed": self.document_size_chars * self.documents_per_iteration,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return False, latency_ms, {"error": str(e)}

    async def teardown(self) -> None:
        """Cleanup."""
        pass


class ConcurrentLoadScenario(BenchmarkScenario):
    """
    Benchmark concurrent query handling.

    Tests system behavior under load:
    - Response time degradation
    - Error rates
    - Resource utilization
    """

    name = "concurrent_load"
    description = "Test system behavior under concurrent load"

    def __init__(
        self,
        queries: list[str] | None = None,
    ):
        self.queries = queries or [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "How do neural networks work?",
            "Define knowledge representation",
            "What are semantic web technologies?",
        ]
        self._workflow = None

    async def setup(self) -> None:
        """Initialize workflow."""
        from src.workflow.graph import create_ontology_reasoning_workflow
        from src.config.settings import get_settings
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        settings = get_settings()
        api_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None

        if api_key:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=api_key,
                temperature=settings.llm.temperature,
                seed=settings.llm.seed,
                top_p=settings.llm.top_p,
            )
            embeddings = OpenAIEmbeddings(api_key=api_key)

            self._workflow = create_ontology_reasoning_workflow(
                llm=llm,
                embeddings=embeddings,
                max_iterations=3,  # Lower for benchmarking
            )

    async def run_iteration(self) -> tuple[bool, float, dict[str, Any]]:
        """Run a query through the workflow."""
        if not self._workflow:
            return False, 0.0, {"error": "Workflow not initialized"}

        query = random.choice(self.queries)
        start_time = time.perf_counter()

        try:
            result = await self._workflow.run(
                query=query,
                thread_id=str(uuid.uuid4()),
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return True, latency_ms, {
                "query": query[:50],
                "confidence": result.get("confidence", 0),
                "iterations": result.get("iteration", 0),
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return False, latency_ms, {"error": str(e)}

    async def teardown(self) -> None:
        """Cleanup."""
        pass


class CacheEffectivenessScenario(BenchmarkScenario):
    """
    Benchmark cache effectiveness.

    Compares performance with and without caching:
    - Query result cache hit rates
    - Embedding cache savings
    - Overall latency reduction
    """

    name = "cache_effectiveness"
    description = "Measure cache effectiveness and hit rates"

    def __init__(
        self,
        queries: list[str] | None = None,
        repeat_probability: float = 0.7,
    ):
        self.queries = queries or [
            "What is machine learning?",
            "Explain deep learning",
            "Define neural networks",
            "What are transformers?",
            "How does NLP work?",
        ]
        self.repeat_probability = repeat_probability
        self._recent_queries: list[str] = []
        self._cache = None

    async def setup(self) -> None:
        """Initialize cache."""
        from src.core.cache import get_query_cache
        self._cache = get_query_cache()

    async def run_iteration(self) -> tuple[bool, float, dict[str, Any]]:
        """Run query with cache measurement."""
        # Decide whether to repeat a recent query or use a new one
        if self._recent_queries and random.random() < self.repeat_probability:
            query = random.choice(self._recent_queries)
            is_repeat = True
        else:
            query = random.choice(self.queries)
            is_repeat = False

        start_time = time.perf_counter()

        try:
            # Check cache
            cached_result = await self._cache.get(query)
            cache_hit = cached_result is not None

            if cache_hit:
                # Cache hit - return cached result
                latency_ms = (time.perf_counter() - start_time) * 1000
                return True, latency_ms, {
                    "cache_hit": True,
                    "is_repeat_query": is_repeat,
                }

            # Cache miss - simulate processing
            # In a real scenario, this would run the actual query
            await asyncio.sleep(0.1)  # Simulate query processing

            # Store in cache
            await self._cache.set(query, {"answer": f"Answer to: {query}"})

            # Track for potential repeats
            self._recent_queries.append(query)
            if len(self._recent_queries) > 10:
                self._recent_queries.pop(0)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return True, latency_ms, {
                "cache_hit": False,
                "is_repeat_query": is_repeat,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return False, latency_ms, {"error": str(e)}

    async def teardown(self) -> None:
        """Get cache statistics."""
        if self._cache:
            stats = self._cache.get_stats()
            logger.info("Cache statistics", **stats)


class GraphScaleScenario(BenchmarkScenario):
    """
    Test query performance at different graph scales.

    Measures how query latency changes as graph size increases.
    """

    name = "graph_scale"
    description = "Test query performance vs graph size"

    def __init__(self):
        self._client = None
        self._graph_stats = {}

    async def setup(self) -> None:
        """Get graph statistics."""
        from src.graph.neo4j_client import get_ontology_client

        self._client = get_ontology_client()
        await self._client.connect()

        # Get current graph size
        stats_query = """
        MATCH (e:Entity)
        WITH count(e) as entity_count
        MATCH (c:Chunk)
        WITH entity_count, count(c) as chunk_count
        MATCH ()-[r]->()
        RETURN entity_count, chunk_count, count(r) as relationship_count
        """
        result = await self._client.execute_cypher(stats_query)
        if result:
            self._graph_stats = result[0]

        logger.info("Graph statistics", **self._graph_stats)

    async def run_iteration(self) -> tuple[bool, float, dict[str, Any]]:
        """Run query and measure against graph size."""
        # Mix of query types
        queries = [
            # Simple lookup
            ("simple", "MATCH (e:Entity) RETURN e LIMIT 100"),
            # Pattern match
            ("pattern", "MATCH (e:Entity)-[r]->(e2:Entity) RETURN e, r, e2 LIMIT 100"),
            # Aggregation
            ("aggregation", "MATCH (e:Entity) RETURN e.type, count(*) as count ORDER BY count DESC LIMIT 10"),
            # Path query
            ("path", "MATCH path = (e:Entity)-[*1..2]-(e2:Entity) RETURN path LIMIT 50"),
        ]

        query_type, query = random.choice(queries)
        start_time = time.perf_counter()

        try:
            result = await self._client.execute_cypher(query)
            latency_ms = (time.perf_counter() - start_time) * 1000

            return True, latency_ms, {
                "query_type": query_type,
                "result_count": len(result),
                "graph_entities": self._graph_stats.get("entity_count", 0),
                "graph_relationships": self._graph_stats.get("relationship_count", 0),
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return False, latency_ms, {"error": str(e)}

    async def teardown(self) -> None:
        """Close client."""
        if self._client:
            await self._client.close()

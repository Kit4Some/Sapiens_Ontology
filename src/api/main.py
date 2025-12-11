"""
FastAPI Application for Ontology Reasoning System.

REST API with SSE streaming for MACER reasoning workflow.
"""

import asyncio
import io
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field, SecretStr

from src.config.settings import get_settings
from src.observability.logging import configure_logging
from src.observability.correlation import CorrelationMiddleware, RequestTimingMiddleware
from src.observability.metrics import get_metrics_registry
from src.observability.tracing import TracingConfig, init_tracing
from src.observability.alerting import (
    get_alert_manager,
    start_alert_checker,
    WebhookConfig,
    AlertChannel,
)
from src.observability.telemetry import (
    init_telemetry,
    get_telemetry_collector,
    TelemetryCollector,
)
from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.llm.provider import (
    LLMProviderChain,
    ProviderConfig,
    ProviderType,
    create_provider_configs_from_settings,
    get_llm_chain,
)
from src.tog.state import MACERState, SubGraph
from src.workflow.graph import (
    OntologyReasoningWorkflow,
    create_ontology_reasoning_workflow,
)

settings = get_settings()

# Configure structured logging
configure_logging(
    level=settings.log_level,
    format=settings.observability.log_format,
    service_name=settings.app_name,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# Global State
# =============================================================================

_workflow: OntologyReasoningWorkflow | None = None
_neo4j_client: OntologyGraphClient | None = None
_llm_chain: LLMProviderChain | None = None
_ingestion_jobs: dict[str, dict[str, Any]] = {}
_alert_checker_task: asyncio.Task | None = None
_telemetry_collector: TelemetryCollector | None = None


class IngestionStatus(str, Enum):
    """Status of an ingestion job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def get_workflow() -> OntologyReasoningWorkflow:
    """Get the global workflow instance."""
    global _workflow
    if _workflow is None:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    return _workflow


def get_graph_client() -> OntologyGraphClient:
    """Get the global Neo4j client."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = get_ontology_client()
    return _neo4j_client


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global _workflow, _neo4j_client, _llm_chain, _alert_checker_task, _telemetry_collector

    logger.info("Starting Ontology Reasoning API", version=settings.app_version)

    # Initialize telemetry if enabled
    if settings.telemetry.enabled:
        _telemetry_collector = init_telemetry(settings.telemetry)
        _telemetry_collector.start_background_flush()
        logger.info(
            "Telemetry initialized",
            endpoint=settings.telemetry.endpoint_url,
            sampling_rate=settings.telemetry.sampling_rate,
        )

    # Initialize tracing if enabled
    if settings.observability.tracing_enabled:
        tracing_config = TracingConfig(
            enabled=True,
            service_name=settings.app_name,
            exporter=settings.observability.tracing_exporter,
            otlp_endpoint=settings.observability.tracing_endpoint,
            sample_rate=settings.observability.tracing_sample_rate,
        )
        init_tracing(tracing_config)
        logger.info("Distributed tracing initialized", exporter=settings.observability.tracing_exporter)

    # Initialize alerting if enabled
    if settings.observability.alerting_enabled:
        alert_manager = get_alert_manager()

        # Configure webhooks
        if settings.observability.slack_webhook_url:
            alert_manager.configure_webhook(WebhookConfig(
                channel=AlertChannel.SLACK,
                url=settings.observability.slack_webhook_url,
            ))

        if settings.observability.pagerduty_routing_key:
            alert_manager.configure_webhook(WebhookConfig(
                channel=AlertChannel.PAGERDUTY,
                url="https://events.pagerduty.com/v2/enqueue",
                pagerduty_routing_key=settings.observability.pagerduty_routing_key,
            ))

        # Start alert checker background task
        _alert_checker_task = await start_alert_checker(
            interval_seconds=settings.observability.alert_check_interval,
            metrics_registry=get_metrics_registry(),
        )
        logger.info("Alerting initialized")

    # Initialize caching if enabled
    if settings.cache.enabled:
        from src.core.cache import init_caches
        init_caches(
            redis_url=settings.cache.redis_url,
            query_cache_size=settings.cache.query_cache_size,
            embedding_cache_size=settings.cache.embedding_cache_size,
            subgraph_cache_size=settings.cache.subgraph_cache_size,
        )
        logger.info(
            "Caching initialized",
            redis_enabled=settings.cache.redis_url is not None,
        )

    # Initialize Neo4j client
    _neo4j_client = get_ontology_client()
    try:
        await _neo4j_client.connect()
        logger.info("Neo4j connected")

        # Setup schema and indexes on startup
        try:
            schema_result = await _neo4j_client.setup_schema()
            logger.info("Neo4j schema initialized", result=schema_result)
        except Exception as e:
            logger.warning("Schema setup warning (indexes may already exist)", error=str(e))

    except Exception as e:
        logger.warning("Neo4j connection failed", error=str(e))

    # Initialize LLM Provider Chain with Failover
    try:
        openai_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
        anthropic_key = settings.llm.anthropic_api_key.get_secret_value() if settings.llm.anthropic_api_key else None
        azure_key = settings.llm.azure_openai_api_key.get_secret_value() if settings.llm.azure_openai_api_key else None

        _llm_chain = get_llm_chain(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            azure_endpoint=settings.llm.azure_openai_endpoint,
            azure_api_key=azure_key,
            azure_deployment=settings.llm.azure_openai_deployment,
            force_reinit=True,
        )

        logger.info(
            "LLM Provider Chain initialized",
            providers=[p.provider_type.value for p in _llm_chain.providers],
            current_provider=_llm_chain.current_provider.provider_type.value,
            failover_enabled=settings.llm.enable_failover,
        )

        # Initialize workflow with LLM from chain
        from langchain_openai import OpenAIEmbeddings

        llm = _llm_chain.get_llm()
        embeddings = OpenAIEmbeddings(
            model=settings.llm.embedding_model,
            api_key=openai_key,
        )

        _workflow = create_ontology_reasoning_workflow(
            llm=llm,
            embeddings=embeddings,
            sufficiency_threshold=settings.tog.confidence_threshold,
            max_iterations=settings.tog.max_reasoning_depth,
            checkpointing_backend=settings.checkpointing.backend,
            postgres_uri=settings.checkpointing.postgres_uri if settings.checkpointing.backend == "postgres" else None,
        )
        logger.info(
            "Workflow initialized with failover-enabled LLM",
            checkpointing_backend=settings.checkpointing.backend,
        )

    except Exception as e:
        logger.warning("LLM chain initialization failed", error=str(e))
        _llm_chain = None
        _workflow = None

    yield

    # Shutdown
    logger.info("Shutting down Ontology Reasoning API")

    # Cancel alert checker task
    if _alert_checker_task and not _alert_checker_task.done():
        _alert_checker_task.cancel()
        try:
            await _alert_checker_task
        except asyncio.CancelledError:
            logger.info("Alert checker task cancelled")

    # Close Neo4j connection
    if _neo4j_client:
        await _neo4j_client.close()
        logger.info("Neo4j connection closed")

    # Close telemetry collector
    if _telemetry_collector:
        _telemetry_collector.stop_background_flush()
        await _telemetry_collector.close()
        logger.info("Telemetry collector closed")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Ontology Reasoning API",
    version="1.0.0",
    description="ToG 3.0 MACER + LangGraph + Neo4j based Knowledge Graph Reasoning System",
    lifespan=lifespan,
)

# CORS for Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Electron app uses file:// or localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Observability middleware
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(CorrelationMiddleware)


# =============================================================================
# Request/Response Models
# =============================================================================


class LLMConfig(BaseModel):
    """Dynamic LLM configuration from frontend."""

    provider: str = Field(default="openai", description="LLM provider: openai or anthropic")
    model: str = Field(default="gpt-4o-mini", description="Model API name")
    api_key: str = Field(default="", description="API key for the provider")


class QueryRequest(BaseModel):
    """Request model for reasoning queries."""

    query: str = Field(..., min_length=1, description="Natural language query")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Maximum reasoning iterations")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    llm_config: LLMConfig | None = Field(default=None, description="Dynamic LLM configuration")


class ReasoningStep(BaseModel):
    """A step in the reasoning process."""

    iteration: int
    node: str  # constructor, retriever, reflector, responser
    action: str = ""
    message: str
    sufficiency_score: float | None = None
    evidence_count: int = 0
    timestamp: str


class ThinkingStep(BaseModel):
    """A thinking step showing internal reasoning."""

    step: int
    thought: str
    action: str
    observation: str
    confidence: float = 0.0


class PatternInsight(BaseModel):
    """A discovered pattern in the knowledge graph."""

    pattern_type: str  # hub, bridge, cluster, chain, star
    entities: list[str]
    description: str
    significance: str


class SupportingEvidence(BaseModel):
    """Evidence supporting the answer."""

    content: str
    source_type: str  # DIRECT, INFERRED, CONTEXTUAL
    relevance: float
    entity_names: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response model for completed queries."""

    id: str
    query: str
    answer: str
    confidence: float
    answer_type: str = "DIRECT"
    reasoning_path: list[ReasoningStep]
    explanation: str
    iteration_count: int
    entities_found: int
    evidence_count: int
    errors: list[str] = Field(default_factory=list)
    # Extended fields for frontend display
    thinking_process: list[ThinkingStep] = Field(default_factory=list)
    patterns: list[PatternInsight] = Field(default_factory=list)
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    direct_answer: str | None = None  # Concise answer for benchmarks


class StreamEvent(BaseModel):
    """SSE stream event."""

    event: str  # "step", "answer", "error", "done"
    data: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    neo4j_connected: bool
    llm_available: bool
    version: str
    environment: str


class SchemaResponse(BaseModel):
    """Graph schema response."""

    node_labels: list[str]
    relationship_types: list[str]
    node_properties: dict[str, list[dict[str, Any]]]
    indexes: list[dict[str, Any]]


class IngestionJobResponse(BaseModel):
    """Response for ingestion job creation."""

    job_id: str
    status: str
    message: str
    files_count: int
    supported_formats: list[str]


class IngestionStatusResponse(BaseModel):
    """Response for ingestion job status."""

    job_id: str
    status: str
    progress: float  # 0.0 to 1.0
    message: str
    files_processed: int
    files_total: int
    entities_created: int
    relations_created: int
    chunks_created: int
    errors: list[str]
    started_at: str | None
    completed_at: str | None


# =============================================================================
# Helper Functions
# =============================================================================


def create_initial_state(query: str, max_iterations: int) -> MACERState:
    """Create initial state for workflow."""
    return {
        "original_query": query,
        "current_query": query,
        "query_history": [],
        "topic_entities": [],
        "retrieved_entities": [],
        "current_subgraph": SubGraph(),
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
        "pipeline_id": str(uuid.uuid4())[:8],
        "errors": [],
        "metadata": {},
    }


def format_reasoning_path(reasoning_path: list[Any]) -> list[ReasoningStep]:
    """Format reasoning path for API response."""
    steps: list[ReasoningStep] = []
    for step in reasoning_path:
        steps.append(
            ReasoningStep(
                iteration=step.step_number if hasattr(step, "step_number") else 0,
                node=step.action.value if hasattr(step, "action") else "unknown",
                action=step.action.value if hasattr(step, "action") else "",
                message=step.thought if hasattr(step, "thought") else str(step),
                sufficiency_score=None,
                evidence_count=len(step.new_evidence) if hasattr(step, "new_evidence") else 0,
                timestamp=datetime.utcnow().isoformat(),
            )
        )
    return steps


def get_step_message(node_name: str, state: dict[str, Any]) -> str:
    """Generate human-readable message for a step."""
    messages = {
        "constructor": "Extracting topic entities from query...",
        "retriever": f"Retrieving evidence (iteration {state.get('iteration', 0)})",
        "reflector": f"Assessing sufficiency: {state.get('sufficiency_score', 0):.0%}",
        "responser": "Generating final answer...",
    }
    return messages.get(node_name, f"Processing {node_name}...")


async def check_neo4j_connection() -> bool:
    """Check Neo4j connectivity."""
    try:
        client = get_graph_client()
        await client.execute_cypher("RETURN 1")
        return True
    except Exception:
        return False


def check_llm_availability() -> bool:
    """Check LLM availability."""
    return _workflow is not None


def create_workflow_with_config(llm_config: LLMConfig | None) -> OntologyReasoningWorkflow:
    """Create a workflow with dynamic LLM configuration using failover chain."""
    if llm_config is None or not llm_config.api_key:
        # Use global workflow with default settings
        workflow = get_workflow()
        return workflow

    try:
        from langchain_openai import OpenAIEmbeddings

        # Create a custom LLM chain for this request
        configs: list[ProviderConfig] = []
        priority = 1

        # Add the requested provider first
        if llm_config.provider == "anthropic":
            configs.append(ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key=llm_config.api_key,
                model=llm_config.model,
                priority=priority,
            ))
            priority += 1
        elif llm_config.provider == "azure":
            configs.append(ProviderConfig(
                provider_type=ProviderType.AZURE_OPENAI,
                api_key=llm_config.api_key,
                model=llm_config.model,
                azure_endpoint=settings.llm.azure_openai_endpoint,
                priority=priority,
            ))
            priority += 1
        else:
            # OpenAI
            configs.append(ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=llm_config.api_key,
                model=llm_config.model,
                priority=priority,
            ))
            priority += 1

        # Add fallback providers from settings (if available and different)
        openai_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
        anthropic_key = settings.llm.anthropic_api_key.get_secret_value() if settings.llm.anthropic_api_key else None

        if openai_key and llm_config.provider != "openai":
            configs.append(ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=openai_key,
                model=settings.llm.reasoning_model,
                priority=priority,
            ))
            priority += 1

        if anthropic_key and llm_config.provider != "anthropic":
            configs.append(ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key=anthropic_key,
                model=settings.llm.anthropic_model,
                priority=priority,
            ))
            priority += 1

        # Add local LLM as final fallback
        configs.append(ProviderConfig(
            provider_type=ProviderType.LOCAL,
            model=settings.llm.local_llm_model,
            local_base_url=settings.llm.local_llm_base_url,
            priority=999,
        ))

        # Create chain and get LLM
        custom_chain = LLMProviderChain(configs)
        llm = custom_chain.get_llm()

        # Embeddings (always use OpenAI for now)
        embedding_key = llm_config.api_key if llm_config.provider == "openai" else openai_key
        embeddings = OpenAIEmbeddings(
            model=settings.llm.embedding_model,
            api_key=embedding_key,
        )

        logger.info(
            "Created workflow with custom LLM chain",
            requested_provider=llm_config.provider,
            providers=[p.provider_type.value for p in custom_chain.providers],
        )

        return create_ontology_reasoning_workflow(
            llm=llm,
            embeddings=embeddings,
            sufficiency_threshold=settings.tog.confidence_threshold,
            max_iterations=settings.tog.max_reasoning_depth,
        )
    except Exception as e:
        logger.error("Failed to create workflow with custom config", error=str(e))
        # Fall back to global workflow
        return get_workflow()


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns system status including Neo4j and LLM availability.
    """
    neo4j_ok = await check_neo4j_connection()
    llm_ok = check_llm_availability()

    status = "healthy" if (neo4j_ok and llm_ok) else "degraded"
    if not neo4j_ok and not llm_ok:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        neo4j_connected=neo4j_ok,
        llm_available=llm_ok,
        version=settings.app_version,
        environment=settings.environment,
    )


@app.get("/api/llm/health", tags=["System"])
async def get_llm_health() -> dict[str, Any]:
    """
    Get detailed LLM provider health status.

    Returns health status for all configured providers including:
    - Current active provider
    - Health status per provider
    - Failover count
    - Latency metrics
    """
    if _llm_chain is None:
        return {
            "status": "unavailable",
            "message": "LLM chain not initialized",
            "providers": [],
        }

    try:
        # Perform health check on all providers
        await _llm_chain.health_check_all()

        return {
            "status": "available",
            **_llm_chain.get_health_status(),
            "failover_enabled": settings.llm.enable_failover,
        }
    except Exception as e:
        logger.error("LLM health check failed", error=str(e))
        return {
            "status": "error",
            "message": str(e),
            "providers": [],
        }


@app.post("/api/llm/failover", tags=["System"])
async def trigger_failover() -> dict[str, Any]:
    """
    Manually trigger failover to next available provider.

    Useful for testing or when current provider is known to be degraded.
    """
    if _llm_chain is None:
        raise HTTPException(status_code=503, detail="LLM chain not initialized")

    current = _llm_chain.current_provider.provider_type.value

    # Force health check to find next available
    await _llm_chain.health_check_all()

    new_provider = _llm_chain.current_provider.provider_type.value

    return {
        "previous_provider": current,
        "current_provider": new_provider,
        "switched": current != new_provider,
        "failover_count": _llm_chain.failover_count,
    }


# =============================================================================
# Observability Endpoints
# =============================================================================


@app.get("/metrics", tags=["Observability"])
async def prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    from src.observability.metrics import get_prometheus_metrics

    metrics_text = get_prometheus_metrics()
    return PlainTextResponse(content=metrics_text, media_type="text/plain")


@app.get("/api/metrics", tags=["Observability"])
async def get_metrics() -> dict[str, Any]:
    """
    Get all metrics in JSON format.

    Returns detailed metrics including:
    - Query latency percentiles (p50, p95, p99)
    - MACER iteration statistics
    - Error rates
    - Request counts
    """
    registry = get_metrics_registry()
    return registry.get_all_metrics()


@app.get("/api/observability/alerts", tags=["Observability"])
async def get_active_alerts() -> dict[str, Any]:
    """
    Get currently active alerts.

    Returns:
    - List of active alerts with severity and description
    - Alert manager status
    - Registered rules count
    """
    alert_manager = get_alert_manager()
    return alert_manager.get_status()


@app.get("/api/observability/alerts/history", tags=["Observability"])
async def get_alert_history(limit: int = 100) -> dict[str, Any]:
    """
    Get alert history.

    Args:
        limit: Maximum number of alerts to return (default: 100)

    Returns:
        Recent alerts with timestamps and resolution status
    """
    alert_manager = get_alert_manager()
    history = alert_manager.get_alert_history(limit=limit)
    return {
        "count": len(history),
        "alerts": [alert.to_dict() for alert in history],
    }


@app.post("/api/observability/alerts/{rule_name}/acknowledge", tags=["Observability"])
async def acknowledge_alert(rule_name: str) -> dict[str, Any]:
    """
    Acknowledge an active alert.

    Args:
        rule_name: The name of the alert rule to acknowledge

    Returns:
        Success status
    """
    alert_manager = get_alert_manager()
    success = alert_manager.acknowledge_alert(rule_name)

    if not success:
        raise HTTPException(status_code=404, detail=f"No active alert found for rule: {rule_name}")

    return {
        "acknowledged": True,
        "rule_name": rule_name,
    }


@app.get("/api/observability/tracing", tags=["Observability"])
async def get_tracing_status() -> dict[str, Any]:
    """
    Get distributed tracing status.

    Returns:
    - Whether tracing is enabled
    - Exporter configuration
    - Sample rate
    """
    from src.observability.tracing import get_tracer

    tracer = get_tracer()
    return {
        "enabled": settings.observability.tracing_enabled,
        "exporter": settings.observability.tracing_exporter if settings.observability.tracing_enabled else None,
        "endpoint": settings.observability.tracing_endpoint if settings.observability.tracing_enabled else None,
        "sample_rate": settings.observability.tracing_sample_rate,
        "opentelemetry_available": tracer._otel_available,
        "active_spans": len(tracer._spans),
    }


# =============================================================================
# Cache Endpoints
# =============================================================================


@app.get("/api/cache/stats", tags=["Cache"])
async def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics.

    Returns:
    - Hit/miss rates for each cache type
    - Memory usage
    - Items count
    """
    from src.core.cache import get_all_cache_stats

    return {
        "enabled": settings.cache.enabled,
        "redis_configured": settings.cache.redis_url is not None,
        **get_all_cache_stats(),
    }


@app.post("/api/cache/clear", tags=["Cache"])
async def clear_caches(
    cache_type: str | None = None,
) -> dict[str, Any]:
    """
    Clear cache(s).

    Args:
        cache_type: Optional specific cache to clear (query, embedding, subgraph).
                   If not provided, clears all caches.

    Returns:
        Count of cleared items per cache
    """
    from src.core.cache import (
        get_query_cache,
        get_embedding_cache,
        get_subgraph_cache,
        clear_all_caches,
    )

    if cache_type is None:
        result = await clear_all_caches()
        logger.info("All caches cleared", result=result)
        return {"cleared": "all", **result}

    if cache_type == "query":
        count = await get_query_cache()._cache.clear()
        return {"cleared": "query", "items": count}
    elif cache_type == "embedding":
        count = await get_embedding_cache()._cache.clear()
        return {"cleared": "embedding", "items": count}
    elif cache_type == "subgraph":
        count = await get_subgraph_cache()._cache.clear()
        return {"cleared": "subgraph", "items": count}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cache type: {cache_type}. Valid types: query, embedding, subgraph"
        )


@app.get("/api/cache/config", tags=["Cache"])
async def get_cache_config() -> dict[str, Any]:
    """
    Get cache configuration.

    Returns current cache settings including sizes, TTLs, and Redis status.
    """
    return {
        "enabled": settings.cache.enabled,
        "redis_url": settings.cache.redis_url[:30] + "..." if settings.cache.redis_url else None,
        "query_cache": {
            "max_size": settings.cache.query_cache_size,
            "ttl_seconds": settings.cache.query_cache_ttl,
        },
        "embedding_cache": {
            "max_size": settings.cache.embedding_cache_size,
            "ttl_seconds": settings.cache.embedding_cache_ttl,
        },
        "subgraph_cache": {
            "max_size": settings.cache.subgraph_cache_size,
            "ttl_seconds": settings.cache.subgraph_cache_ttl,
        },
        "l1_max_bytes": settings.cache.l1_max_bytes,
    }


# =============================================================================
# Graph Endpoints
# =============================================================================


@app.get("/api/schema", response_model=SchemaResponse, tags=["Graph"])
async def get_graph_schema() -> SchemaResponse:
    """
    Get graph database schema.

    Returns node labels, relationship types, and available indexes.
    """
    try:
        client = get_graph_client()
        schema = await client.get_schema()

        return SchemaResponse(
            node_labels=schema.get("node_labels", []),
            relationship_types=schema.get("relationship_types", []),
            node_properties=schema.get("node_properties", {}),
            indexes=schema.get("indexes", [])[:20],  # Limit response size
        )
    except Exception as e:
        logger.error("Schema retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve schema: {str(e)}") from e


@app.post("/api/query", response_model=QueryResponse, tags=["Reasoning"])
async def process_query(request: QueryRequest, use_cache: bool = True) -> QueryResponse:
    """
    Process a reasoning query (synchronous).

    Executes the full MACER workflow and returns the complete result.

    Args:
        request: The query request
        use_cache: Whether to use query result caching (default: True)
    """
    from src.observability.metrics import QueryTimer
    from src.observability.tracing import trace_span, SpanKind
    from src.core.cache import get_query_cache

    query_id = str(uuid.uuid4())
    metrics_registry = get_metrics_registry()
    query_cache = get_query_cache()

    logger.info("Processing query", query_id=query_id, query=request.query[:100])

    # Check cache first (if enabled and no custom LLM config)
    if use_cache and settings.cache.enabled and request.llm_config is None:
        cached_result = await query_cache.get(
            request.query,
            max_iterations=request.max_iterations,
            context=request.context,
        )
        if cached_result:
            logger.info("Query cache hit", query_id=query_id)
            return QueryResponse(
                id=query_id,
                query=request.query,
                answer=cached_result.get("answer", ""),
                confidence=cached_result.get("confidence", 0.0),
                answer_type=cached_result.get("answer_type", "CACHED"),
                reasoning_path=cached_result.get("reasoning_path", []),
                explanation=cached_result.get("explanation", "") + " (cached)",
                iteration_count=cached_result.get("iteration_count", 0),
                entities_found=cached_result.get("entities_found", 0),
                evidence_count=cached_result.get("evidence_count", 0),
                errors=[],
            )

    workflow = create_workflow_with_config(request.llm_config)

    with trace_span("api.query", kind=SpanKind.SERVER, attributes={"query_id": query_id}):
        with QueryTimer(query_id, metrics_registry.query_metrics) as timer:
            try:
                result = await workflow.run(
                    query=request.query,
                    context=request.context,
                    thread_id=query_id,
                )

                # Record metrics
                iteration_count = result.get("iteration", 0)
                confidence = result.get("confidence", 0.0)
                timer.record_iterations(iteration_count)
                timer.record_confidence(confidence)

                # Extract thinking process from reasoning path
                thinking_process = []
                for idx, step in enumerate(result.get("reasoning_path", [])):
                    if hasattr(step, "thought") or isinstance(step, dict):
                        step_data = step if isinstance(step, dict) else step.__dict__
                        thinking_process.append(ThinkingStep(
                            step=idx + 1,
                            thought=step_data.get("thought", ""),
                            action=step_data.get("action", {}).value if hasattr(step_data.get("action", {}), "value") else str(step_data.get("action", "")),
                            observation=step_data.get("observation", ""),
                            confidence=0.5 + float(step_data.get("sufficiency_delta", 0)),
                        ))

                # Extract patterns from advanced synthesis if available
                patterns = []
                advanced_synthesis = result.get("advanced_synthesis", {})
                for pattern in advanced_synthesis.get("pattern_insights", []):
                    patterns.append(PatternInsight(
                        pattern_type=pattern.get("pattern_type", "unknown"),
                        entities=pattern.get("entities_involved", []),
                        description=pattern.get("description", ""),
                        significance=pattern.get("significance", ""),
                    ))

                # Extract supporting evidence
                supporting_evidence = []
                for ev in result.get("evidence", [])[:10]:  # Top 10
                    ev_data = ev if isinstance(ev, dict) else ev.__dict__
                    supporting_evidence.append(SupportingEvidence(
                        content=ev_data.get("content", "")[:300],
                        source_type=ev_data.get("evidence_type", {}).value if hasattr(ev_data.get("evidence_type", {}), "value") else str(ev_data.get("evidence_type", "UNKNOWN")),
                        relevance=float(ev_data.get("relevance_score", 0)),
                        entity_names=ev_data.get("entities_mentioned", [])[:5],
                    ))

                # Extract direct answer (concise version for benchmarks)
                # Priority: direct_answer field > answer_components.direct_answer > fallback extraction
                direct_answer = result.get("direct_answer")
                if not direct_answer or len(str(direct_answer).split()) > 10:
                    direct_answer = result.get("answer_components", {}).get("direct_answer")
                if not direct_answer or len(str(direct_answer).split()) > 10:
                    # Fallback: try to extract first meaningful phrase
                    full_answer = result.get("final_answer", "")
                    if full_answer:
                        import re
                        # Remove common verbose prefixes
                        clean = re.sub(r"^(Based on|According to|The answer is|It is).*?,\s*", "", full_answer, flags=re.IGNORECASE)
                        first_sentence = clean.split('.')[0].strip()
                        # Take first short phrase
                        direct_answer = first_sentence[:100] if len(first_sentence) > 100 else first_sentence

                response = QueryResponse(
                    id=query_id,
                    query=request.query,
                    answer=result.get("final_answer") or "Unable to generate answer",
                    confidence=confidence,
                    answer_type=result.get("answer_type", "UNCERTAIN"),
                    reasoning_path=format_reasoning_path(result.get("reasoning_path", [])),
                    explanation=result.get("explanation", ""),
                    iteration_count=iteration_count,
                    entities_found=len(result.get("retrieved_entities", [])),
                    evidence_count=len(result.get("evidence", [])),
                    errors=result.get("errors", []),
                    thinking_process=thinking_process,
                    patterns=patterns,
                    supporting_evidence=supporting_evidence,
                    direct_answer=direct_answer,
                )

                # Cache successful results (if enabled and no custom LLM config)
                if use_cache and settings.cache.enabled and request.llm_config is None:
                    if confidence >= 0.5:  # Only cache reasonably confident results
                        await query_cache.set(
                            request.query,
                            {
                                "answer": response.answer,
                                "confidence": response.confidence,
                                "answer_type": response.answer_type,
                                "reasoning_path": [step.model_dump() for step in response.reasoning_path],
                                "explanation": response.explanation,
                                "iteration_count": response.iteration_count,
                                "entities_found": response.entities_found,
                                "evidence_count": response.evidence_count,
                            },
                            max_iterations=request.max_iterations,
                            context=request.context,
                        )

                # Track query telemetry (if enabled)
                telemetry = get_telemetry_collector()
                if telemetry and telemetry.is_enabled:
                    await telemetry.track_query(
                        query=request.query,
                        response={
                            "confidence": response.confidence,
                            "answer_type": response.answer_type,
                            "iterations": response.iteration_count,
                            "evidence_count": response.evidence_count,
                        },
                        latency_ms=timer.elapsed_ms,
                        metadata={"query_id": query_id},
                    )

                return response

            except Exception as e:
                timer.record_error()
                logger.error("Query processing failed", query_id=query_id, error=str(e))
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}") from e


@app.post("/api/query/stream", tags=["Reasoning"])
async def stream_query(request: QueryRequest) -> StreamingResponse:
    """
    Process a reasoning query with SSE streaming.

    Streams progress updates as the workflow executes.
    """
    workflow = create_workflow_with_config(request.llm_config)

    async def event_generator() -> AsyncGenerator[str, None]:
        query_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": query_id}}
        initial_state = create_initial_state(request.query, request.max_iterations)

        logger.info("Starting streaming query", query_id=query_id)

        try:
            # Stream workflow execution
            app = workflow.compile()

            async for event in app.astream(initial_state, config=config):
                for node_name, state_update in event.items():
                    step_event = StreamEvent(
                        event="step",
                        data={
                            "query_id": query_id,
                            "node": node_name,
                            "iteration": state_update.get("iteration", 0),
                            "sufficiency_score": state_update.get("sufficiency_score", 0),
                            "current_query": state_update.get("current_query", request.query),
                            "evidence_count": len(state_update.get("evidence", [])),
                            "entities_count": len(state_update.get("topic_entities", [])),
                            "message": get_step_message(node_name, state_update),
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )
                    yield f"data: {step_event.model_dump_json()}\n\n"
                    await asyncio.sleep(0.05)  # Small delay for UI updates

            # Get final state
            final_state = app.get_state(config)
            if final_state and final_state.values:
                values = final_state.values
                answer_event = StreamEvent(
                    event="answer",
                    data={
                        "query_id": query_id,
                        "answer": values.get("final_answer", ""),
                        "confidence": values.get("confidence", 0.0),
                        "answer_type": values.get("answer_type", "UNCERTAIN"),
                        "explanation": values.get("explanation", ""),
                        "iteration_count": values.get("iteration", 0),
                        "entities_found": len(values.get("retrieved_entities", [])),
                        "evidence_count": len(values.get("evidence", [])),
                        "reasoning_path": [
                            {
                                "iteration": s.step_number if hasattr(s, "step_number") else i,
                                "action": s.action.value if hasattr(s, "action") else "unknown",
                                "message": s.thought if hasattr(s, "thought") else str(s),
                            }
                            for i, s in enumerate(values.get("reasoning_path", []))
                        ],
                    },
                )
                yield f"data: {answer_event.model_dump_json()}\n\n"

            # Done event
            yield f"data: {StreamEvent(event='done', data={'query_id': query_id}).model_dump_json()}\n\n"

            logger.info("Streaming query completed", query_id=query_id)

        except Exception as e:
            logger.error("Streaming query failed", query_id=query_id, error=str(e))
            error_event = StreamEvent(event="error", data={"message": str(e), "query_id": query_id})
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/cypher", tags=["Graph"])
async def execute_cypher(
    cypher: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute a raw Cypher query.

    For advanced users and debugging. Use with caution.
    """
    try:
        client = get_graph_client()
        results = await client.execute_cypher(cypher, parameters)
        return {
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        logger.error("Cypher execution failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Cypher execution failed: {str(e)}") from e


class Text2CypherRequest(BaseModel):
    """Request model for Text2Cypher generation."""

    question: str = Field(..., min_length=1, description="Natural language question")
    execute: bool = Field(default=False, description="Whether to execute the generated query")
    use_entity_resolution: bool = Field(default=True, description="Use entity resolution for better accuracy")
    llm_config: LLMConfig | None = Field(default=None, description="Dynamic LLM configuration")


class Text2CypherResponse(BaseModel):
    """Response model for Text2Cypher generation."""

    cypher: str
    confidence: float
    is_valid: bool
    entities_detected: list[str]
    entity_mappings: dict[str, Any]
    healing_attempts: int
    results: list[dict[str, Any]] | None = None
    results_count: int | None = None


@app.post("/api/text2cypher", tags=["Graph"], response_model=Text2CypherResponse)
async def text2cypher(request: Text2CypherRequest) -> Text2CypherResponse:
    """
    Generate Cypher query from natural language.

    Uses schema-aware generation with optional entity resolution
    and self-healing validation. Optionally executes the query.

    Returns the generated Cypher, confidence score, and optionally
    the query results.
    """
    from src.text2cypher import Text2CypherGenerator
    from langchain_openai import OpenAIEmbeddings

    try:
        # Get or create LLM
        if request.llm_config and request.llm_config.api_key:
            configs: list[ProviderConfig] = []
            if request.llm_config.provider == "anthropic":
                configs.append(ProviderConfig(
                    provider_type=ProviderType.ANTHROPIC,
                    api_key=request.llm_config.api_key,
                    model=request.llm_config.model,
                    priority=1,
                ))
            else:
                configs.append(ProviderConfig(
                    provider_type=ProviderType.OPENAI,
                    api_key=request.llm_config.api_key,
                    model=request.llm_config.model,
                    priority=1,
                ))
            llm_chain = LLMProviderChain(configs)
            llm = llm_chain.get_llm()
            embedding_key = request.llm_config.api_key if request.llm_config.provider == "openai" else None
        elif _llm_chain:
            llm = _llm_chain.get_llm()
            embedding_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
        else:
            raise HTTPException(status_code=503, detail="LLM not available")

        # Create embeddings for entity resolution
        embeddings = OpenAIEmbeddings(api_key=embedding_key) if embedding_key else None

        # Create Text2Cypher generator
        client = get_graph_client()
        generator = Text2CypherGenerator(
            llm=llm,
            embeddings=embeddings,
            neo4j_client=client,
            enable_entity_resolution=request.use_entity_resolution,
            enable_self_healing=True,
            max_healing_retries=2,
        )

        # Generate Cypher
        if request.use_entity_resolution and embeddings:
            gen_result = await generator.generate_with_entity_resolution(
                question=request.question,
                validate=True,
            )
        else:
            gen_result = await generator.generate(
                question=request.question,
                use_examples=True,
                validate=True,
            )

        logger.info(
            "Text2Cypher generated",
            question=request.question[:50],
            cypher=gen_result.cypher[:100] if gen_result.cypher else "None",
            confidence=gen_result.confidence,
            is_valid=gen_result.is_valid,
        )

        # Optionally execute the query
        results = None
        results_count = None
        if request.execute and gen_result.is_valid:
            try:
                cypher = gen_result.cypher.strip()
                if "LIMIT" not in cypher.upper():
                    cypher = f"{cypher} LIMIT 100"
                results = await client.execute_cypher(cypher, gen_result.parameters or None)
                results_count = len(results)
                logger.info("Text2Cypher query executed", results_count=results_count)
            except Exception as e:
                logger.warning("Text2Cypher execution failed", error=str(e))
                # Return generation result without execution
                pass

        return Text2CypherResponse(
            cypher=gen_result.cypher,
            confidence=gen_result.confidence,
            is_valid=gen_result.is_valid,
            entities_detected=gen_result.entities_detected,
            entity_mappings=gen_result.entity_mappings,
            healing_attempts=gen_result.healing_attempts,
            results=results,
            results_count=results_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text2Cypher failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Text2Cypher failed: {str(e)}") from e


# =============================================================================
# Ontology Export Endpoints
# =============================================================================


class OntologyExportFormat(str, Enum):
    """Export format for ontology."""
    JSONLD = "jsonld"
    JSON = "json"
    TURTLE = "turtle"


class OntologyExportRequest(BaseModel):
    """Request for ontology export."""
    format: OntologyExportFormat = Field(default=OntologyExportFormat.JSONLD)
    include_hierarchy: bool = Field(default=True, description="Include type hierarchy")
    include_examples: bool = Field(default=True, description="Include example values")
    base_uri: str = Field(default="http://ontology.local/", description="Base URI for the ontology")


@app.get("/api/ontology", tags=["Ontology"])
async def get_ontology(
    format: str = "jsonld",
    include_hierarchy: bool = True,
    include_examples: bool = True,
) -> dict[str, Any]:
    """
    Export the ontology schema.

    Returns the complete ontology including entity types, predicates,
    cardinality constraints, and type hierarchy.

    Args:
        format: Export format (jsonld, json, turtle)
        include_hierarchy: Include type hierarchy information
        include_examples: Include example values

    Returns:
        Ontology document in requested format
    """
    try:
        from src.sddi.extractors.schema import (
            OntologyExporter,
            create_extended_entity_registry,
            create_base_predicate_registry,
        )

        exporter = OntologyExporter(
            entity_registry=create_extended_entity_registry(),
            predicate_registry=create_base_predicate_registry(),
        )

        if format == "turtle":
            return {
                "format": "turtle",
                "content": exporter.export_turtle(),
            }
        elif format == "json":
            return exporter.export_json()
        else:  # jsonld
            return exporter.export_jsonld(
                include_hierarchy=include_hierarchy,
                include_examples=include_examples,
            )

    except Exception as e:
        logger.error("Ontology export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}") from e


@app.get("/api/ontology/validate", tags=["Ontology"])
async def validate_ontology_endpoint() -> dict[str, Any]:
    """
    Validate the ontology schema for consistency.

    Checks:
    - Type hierarchy validity (no cycles, all parents exist)
    - Predicate domain/range validity
    - Inverse predicate consistency

    Returns:
        Validation report with errors and warnings
    """
    try:
        from src.sddi.extractors.schema import (
            validate_ontology as do_validate,
            create_extended_entity_registry,
            create_base_predicate_registry,
        )

        result = do_validate(
            entity_registry=create_extended_entity_registry(),
            predicate_registry=create_base_predicate_registry(),
        )

        logger.info(
            "Ontology validated",
            valid=result["valid"],
            errors=len(result["errors"]),
            warnings=len(result["warnings"]),
        )

        return result

    except Exception as e:
        logger.error("Ontology validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e


@app.get("/api/ontology/hierarchy", tags=["Ontology"])
async def get_type_hierarchy() -> dict[str, Any]:
    """
    Get the entity type inheritance hierarchy.

    Returns a tree structure showing type relationships:
    - PERSON → EMPLOYEE → EXECUTIVE
    - ORGANIZATION → CORPORATION, UNIVERSITY, etc.

    Returns:
        Type hierarchy as nested dictionary
    """
    try:
        from src.sddi.extractors.schema import create_extended_entity_registry

        registry = create_extended_entity_registry()
        return registry.get_type_hierarchy()

    except Exception as e:
        logger.error("Type hierarchy fetch failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}") from e


@app.get("/api/ontology/types", tags=["Ontology"])
async def get_entity_types(include_extended: bool = True) -> dict[str, Any]:
    """
    Get all registered entity types.

    Args:
        include_extended: Include extended types with inheritance

    Returns:
        List of entity type definitions
    """
    try:
        from src.sddi.extractors.schema import (
            create_base_entity_registry,
            create_extended_entity_registry,
        )

        if include_extended:
            registry = create_extended_entity_registry()
        else:
            registry = create_base_entity_registry()

        return {
            "types": registry.to_dict()["types"],
            "count": len(registry.get_type_names()),
        }

    except Exception as e:
        logger.error("Entity types fetch failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}") from e


@app.get("/api/ontology/predicates", tags=["Ontology"])
async def get_predicates() -> dict[str, Any]:
    """
    Get all registered predicates with cardinality constraints.

    Returns:
        List of predicate definitions including cardinality, domain, and range
    """
    try:
        from src.sddi.extractors.schema import create_base_predicate_registry

        registry = create_base_predicate_registry()

        return {
            "predicates": registry.to_dict()["predicates"],
            "count": len(registry.get_all()),
            "categories": registry.get_categories(),
        }

    except Exception as e:
        logger.error("Predicates fetch failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}") from e


@app.get("/api/stats", tags=["System"])
async def get_stats() -> dict[str, Any]:
    """
    Get system statistics.

    Returns counts of nodes, relationships, and other metrics.
    """
    try:
        client = get_graph_client()
        from src.sddi.loaders.neo4j_loader import Neo4jLoader

        loader = Neo4jLoader(client=client)
        stats = await loader.get_stats()

        return {
            "graph": stats,
            "system": {
                "workflow_available": _workflow is not None,
                "environment": settings.environment,
            },
        }
    except Exception as e:
        logger.error("Stats retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# SDDI Ingestion Endpoints
# =============================================================================


async def run_ingestion_pipeline(
    job_id: str,
    files_data: list[tuple[str, bytes]],
    llm_config: LLMConfig | None,
) -> None:
    """Background task to run the SDDI pipeline."""
    global _ingestion_jobs

    total_size_mb = sum(len(content) for _, content in files_data) / (1024 * 1024)
    logger.info(
        "Starting ingestion pipeline",
        job_id=job_id,
        files_count=len(files_data),
        total_size_mb=round(total_size_mb, 2),
    )

    _ingestion_jobs[job_id]["status"] = IngestionStatus.PROCESSING.value
    _ingestion_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
    _ingestion_jobs[job_id]["message"] = "Initializing pipeline..."

    try:
        # Import SDDI components
        from src.sddi.document_loaders import DocumentLoaderFactory
        from src.sddi.pipeline import SDDIPipeline
        from src.sddi.loaders.neo4j_loader import Neo4jLoader

        # Setup LLM with failover chain and embeddings
        from langchain_openai import OpenAIEmbeddings

        openai_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
        anthropic_key = settings.llm.anthropic_api_key.get_secret_value() if settings.llm.anthropic_api_key else None

        if llm_config and llm_config.api_key:
            # Create custom LLM chain with user's provider as primary
            configs: list[ProviderConfig] = []
            priority = 1

            if llm_config.provider == "anthropic":
                configs.append(ProviderConfig(
                    provider_type=ProviderType.ANTHROPIC,
                    api_key=llm_config.api_key,
                    model=llm_config.model,
                    priority=priority,
                ))
                priority += 1
            elif llm_config.provider == "azure":
                configs.append(ProviderConfig(
                    provider_type=ProviderType.AZURE_OPENAI,
                    api_key=llm_config.api_key,
                    model=llm_config.model,
                    azure_endpoint=settings.llm.azure_openai_endpoint,
                    priority=priority,
                ))
                priority += 1
            else:
                configs.append(ProviderConfig(
                    provider_type=ProviderType.OPENAI,
                    api_key=llm_config.api_key,
                    model=llm_config.model,
                    priority=priority,
                ))
                priority += 1

            # Add fallback providers
            if openai_key and llm_config.provider != "openai":
                configs.append(ProviderConfig(
                    provider_type=ProviderType.OPENAI,
                    api_key=openai_key,
                    model=settings.llm.reasoning_model,
                    priority=priority,
                ))
                priority += 1

            if anthropic_key and llm_config.provider != "anthropic":
                configs.append(ProviderConfig(
                    provider_type=ProviderType.ANTHROPIC,
                    api_key=anthropic_key,
                    model=settings.llm.anthropic_model,
                    priority=priority,
                ))
                priority += 1

            # Local LLM fallback
            configs.append(ProviderConfig(
                provider_type=ProviderType.LOCAL,
                model=settings.llm.local_llm_model,
                local_base_url=settings.llm.local_llm_base_url,
                priority=999,
            ))

            llm_chain = LLMProviderChain(configs)
            embedding_key = llm_config.api_key if llm_config.provider == "openai" else openai_key
        else:
            # Use global LLM chain or create default
            if _llm_chain is not None:
                llm_chain = _llm_chain
            else:
                configs = create_provider_configs_from_settings(
                    openai_api_key=openai_key,
                    anthropic_api_key=anthropic_key,
                )
                llm_chain = LLMProviderChain(configs)
            embedding_key = openai_key

        llm = llm_chain.get_llm()
        embeddings = OpenAIEmbeddings(api_key=embedding_key)

        logger.info(
            "Ingestion LLM chain initialized",
            providers=[p.provider_type.value for p in llm_chain.providers],
            current_provider=llm_chain.current_provider.provider_type.value,
        )

        # Create progress callback to update job status
        def progress_callback(step: str, progress: float, message: str) -> None:
            # Scale progress: file loading = 0-30%, pipeline = 30-100%
            scaled_progress = 0.3 + (progress * 0.7)
            _ingestion_jobs[job_id]["progress"] = scaled_progress
            _ingestion_jobs[job_id]["message"] = f"[{step}] {message}"
            logger.debug(
                "Pipeline progress",
                job_id=job_id,
                step=step,
                progress=round(progress, 2),
                message=message,
            )

        # Create pipeline with progress callback
        neo4j_loader = Neo4jLoader(client=get_graph_client())
        pipeline = SDDIPipeline(
            llm=llm,
            embeddings=embeddings,
            neo4j_loader=neo4j_loader,
            progress_callback=progress_callback,
        )

        # Process files with detailed progress tracking
        all_documents = []
        errors = []
        files_total = len(files_data)

        logger.info("Starting document loading phase", job_id=job_id, files_total=files_total)

        for idx, (filename, content) in enumerate(files_data):
            file_size_mb = len(content) / (1024 * 1024)
            logger.info(
                f"Processing file {idx + 1}/{files_total}",
                filename=filename,
                size_mb=round(file_size_mb, 2),
            )

            _ingestion_jobs[job_id]["message"] = f"Loading file ({idx + 1}/{files_total}): {filename} ({file_size_mb:.1f}MB)"
            # File loading uses 0-30% progress
            _ingestion_jobs[job_id]["progress"] = (idx / files_total) * 0.3

            try:
                # Load file using appropriate loader
                loader = DocumentLoaderFactory.get_loader(filename)

                logger.info(
                    "Using document loader",
                    filename=filename,
                    loader=loader.__class__.__name__,
                )

                documents = await loader.load(
                    io.BytesIO(content),
                    filename,
                    metadata={"job_id": job_id},
                )

                doc_count = len(documents)
                all_documents.extend(documents)

                logger.info(
                    "File loaded successfully",
                    filename=filename,
                    documents_created=doc_count,
                    total_documents=len(all_documents),
                )

                _ingestion_jobs[job_id]["files_processed"] = idx + 1
                _ingestion_jobs[job_id]["message"] = f"Loaded {filename}: {doc_count} documents"

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                error_msg = f"Failed to process {filename}: {str(e)}"
                logger.error(
                    "Document loading failed",
                    filename=filename,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=error_trace,
                )
                errors.append(error_msg)
                _ingestion_jobs[job_id]["errors"].append(error_msg)

        # Log loading phase summary
        logger.info(
            "Document loading phase complete",
            job_id=job_id,
            total_documents=len(all_documents),
            errors_count=len(errors),
        )

        if not all_documents:
            error_detail = (
                f"No documents were successfully loaded from {files_total} file(s). "
                f"Errors: {'; '.join(errors) if errors else 'No specific errors captured'}"
            )
            raise ValueError(error_detail)

        # Run SDDI pipeline (progress will be updated via callback)
        _ingestion_jobs[job_id]["message"] = "Starting knowledge extraction pipeline..."
        _ingestion_jobs[job_id]["progress"] = 0.3

        result = await pipeline.run(all_documents, pipeline_id=job_id)

        # Update job status
        load_result = result.get("load_result")
        _ingestion_jobs[job_id].update({
            "status": IngestionStatus.COMPLETED.value,
            "progress": 1.0,
            "message": "Ingestion completed successfully",
            "entities_created": load_result.entities_created if load_result else 0,
            "relations_created": load_result.relations_created if load_result else 0,
            "chunks_created": load_result.chunks_created if load_result else 0,
            "errors": errors + (result.get("errors", [])),
            "completed_at": datetime.utcnow().isoformat(),
        })

        logger.info(
            "Ingestion completed",
            job_id=job_id,
            entities=load_result.entities_created if load_result else 0,
            relations=load_result.relations_created if load_result else 0,
        )

        # Track KG extraction telemetry (if enabled)
        telemetry = get_telemetry_collector()
        if telemetry and telemetry.is_enabled:
            # Calculate total processing time
            started_at = _ingestion_jobs[job_id].get("started_at")
            completed_at = _ingestion_jobs[job_id].get("completed_at")
            if started_at and completed_at:
                from datetime import datetime as dt
                start = dt.fromisoformat(started_at)
                end = dt.fromisoformat(completed_at)
                latency_ms = (end - start).total_seconds() * 1000
            else:
                latency_ms = 0

            # Determine dominant document type from files
            doc_types = [f[0].rsplit(".", 1)[-1].lower() for f in files_data if "." in f[0]]
            dominant_type = max(set(doc_types), key=doc_types.count) if doc_types else "unknown"

            await telemetry.track_kg_extraction(
                document_type=dominant_type,
                entity_count=load_result.entities_created if load_result else 0,
                relation_count=load_result.relations_created if load_result else 0,
                chunk_count=load_result.chunks_created if load_result else 0,
                latency_ms=latency_ms,
                metadata={"job_id": job_id, "files_count": len(files_data)},
            )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error("Ingestion failed", job_id=job_id, error=str(e), traceback=error_trace)
        _ingestion_jobs[job_id].update({
            "status": IngestionStatus.FAILED.value,
            "message": f"Ingestion failed: {str(e)}",
            "errors": _ingestion_jobs[job_id].get("errors", []) + [str(e)],
            "completed_at": datetime.utcnow().isoformat(),
        })


@app.post("/api/ingest", response_model=IngestionJobResponse, tags=["Ingestion"])
async def upload_and_ingest(
    files: list[UploadFile] = File(..., description="Files to ingest"),
    llm_provider: str = Form(default="openai", description="LLM provider"),
    llm_model: str = Form(default="gpt-4o-mini", description="LLM model"),
    llm_api_key: str = Form(default="", description="API key"),
) -> IngestionJobResponse:
    """
    Upload files and ingest into knowledge graph.

    Supports: .txt, .md, .csv, .pdf files.
    Runs SDDI pipeline in background and returns job ID for status tracking.
    """
    from src.sddi.document_loaders import DocumentLoaderFactory

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate file types
    supported = DocumentLoaderFactory.get_supported_extensions()
    files_data: list[tuple[str, bytes]] = []

    for file in files:
        if not file.filename:
            continue

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{ext}. Supported: {', '.join('.' + e for e in supported)}",
            )

        content = await file.read()
        files_data.append((file.filename, content))

    if not files_data:
        raise HTTPException(status_code=400, detail="No valid files to process")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    _ingestion_jobs[job_id] = {
        "status": IngestionStatus.PENDING.value,
        "progress": 0.0,
        "message": "Job created, waiting to start...",
        "files_processed": 0,
        "files_total": len(files_data),
        "entities_created": 0,
        "relations_created": 0,
        "chunks_created": 0,
        "errors": [],
        "started_at": None,
        "completed_at": None,
    }

    # Create LLM config
    llm_config = LLMConfig(
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
    ) if llm_api_key else None

    # Start background task using asyncio.create_task for reliable async execution
    import asyncio
    asyncio.create_task(run_ingestion_pipeline(job_id, files_data, llm_config))

    logger.info("Ingestion job created", job_id=job_id, files=len(files_data))

    return IngestionJobResponse(
        job_id=job_id,
        status=IngestionStatus.PENDING.value,
        message=f"Ingestion job created. Processing {len(files_data)} file(s).",
        files_count=len(files_data),
        supported_formats=[f".{ext}" for ext in supported],
    )


@app.get("/api/ingest/{job_id}/stream", tags=["Ingestion"])
async def stream_ingestion_progress(job_id: str) -> StreamingResponse:
    """
    Stream real-time progress updates for an ingestion job via SSE.

    Provides detailed step-by-step progress including:
    - Current processing step
    - Batch progress for entity extraction
    - Embedding generation progress
    - Final results
    """
    from src.sddi.pipeline import subscribe_to_pipeline, PipelineProgress

    async def event_generator() -> AsyncGenerator[str, None]:
        # First, check if job exists
        if job_id not in _ingestion_jobs:
            error_event = StreamEvent(
                event="error",
                data={"message": f"Job not found: {job_id}", "job_id": job_id},
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
            return

        # Send current job status first
        job = _ingestion_jobs[job_id]
        status_event = StreamEvent(
            event="status",
            data={
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"],
                "entities_created": job.get("entities_created", 0),
                "relations_created": job.get("relations_created", 0),
                "chunks_created": job.get("chunks_created", 0),
            },
        )
        yield f"data: {status_event.model_dump_json()}\n\n"

        # If job already completed, don't need to stream
        if job["status"] in ["completed", "failed"]:
            done_event = StreamEvent(event="done", data={"job_id": job_id})
            yield f"data: {done_event.model_dump_json()}\n\n"
            return

        # Subscribe to real-time progress updates
        try:
            async for progress in subscribe_to_pipeline(job_id):
                if progress.step == "heartbeat":
                    yield f": heartbeat\n\n"
                    continue

                progress_event = StreamEvent(
                    event="progress",
                    data={
                        "job_id": job_id,
                        "step": progress.step,
                        "progress": progress.progress,
                        "message": progress.message,
                        "details": progress.details,
                        "timestamp": progress.timestamp,
                        # Include real-time counts at root level for frontend
                        "entities_created": progress.entities_created,
                        "relations_created": progress.relations_created,
                        "chunks_created": progress.chunks_created,
                    },
                )
                yield f"data: {progress_event.model_dump_json()}\n\n"

                # Update job dict with current counts for polling clients
                if job_id in _ingestion_jobs:
                    _ingestion_jobs[job_id]["progress"] = progress.progress
                    _ingestion_jobs[job_id]["message"] = f"[{progress.step}] {progress.message}"
                    if progress.entities_created > 0:
                        _ingestion_jobs[job_id]["entities_created"] = progress.entities_created
                    if progress.relations_created > 0:
                        _ingestion_jobs[job_id]["relations_created"] = progress.relations_created
                    if progress.chunks_created > 0:
                        _ingestion_jobs[job_id]["chunks_created"] = progress.chunks_created

                # Check for completion
                if progress.step in ["completed", "failed"]:
                    # Update job status from pipeline progress
                    if job_id in _ingestion_jobs:
                        _ingestion_jobs[job_id]["status"] = (
                            IngestionStatus.COMPLETED.value
                            if progress.step == "completed"
                            else IngestionStatus.FAILED.value
                        )
                        _ingestion_jobs[job_id]["progress"] = 1.0 if progress.step == "completed" else job["progress"]
                        _ingestion_jobs[job_id]["message"] = progress.message
                        if progress.details:
                            _ingestion_jobs[job_id].update({
                                "entities_created": progress.details.get("entities_created", 0),
                                "relations_created": progress.details.get("relations_created", 0),
                                "chunks_created": progress.details.get("chunks_created", 0),
                            })

                    done_event = StreamEvent(event="done", data={"job_id": job_id})
                    yield f"data: {done_event.model_dump_json()}\n\n"
                    break

        except Exception as e:
            logger.error("Ingestion streaming failed", job_id=job_id, error=str(e))
            error_event = StreamEvent(event="error", data={"message": str(e), "job_id": job_id})
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/ingest/{job_id}", response_model=IngestionStatusResponse, tags=["Ingestion"])
async def get_ingestion_status(job_id: str) -> IngestionStatusResponse:
    """
    Get status of an ingestion job.

    Returns current progress and results.
    """
    if job_id not in _ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _ingestion_jobs[job_id]

    return IngestionStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        files_processed=job["files_processed"],
        files_total=job["files_total"],
        entities_created=job["entities_created"],
        relations_created=job["relations_created"],
        chunks_created=job["chunks_created"],
        errors=job["errors"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
    )


@app.get("/api/ingest/formats/supported", tags=["Ingestion"])
async def get_supported_formats() -> dict[str, Any]:
    """
    Get list of supported file formats for ingestion.
    """
    from src.sddi.document_loaders import DocumentLoaderFactory

    extensions = DocumentLoaderFactory.get_supported_extensions()

    format_info = {
        "txt": {"name": "Plain Text", "description": "Simple text files"},
        "text": {"name": "Plain Text", "description": "Simple text files"},
        "md": {"name": "Markdown", "description": "Markdown formatted documents"},
        "markdown": {"name": "Markdown", "description": "Markdown formatted documents"},
        "csv": {"name": "CSV", "description": "Comma-separated values (one doc per row)"},
        "pdf": {"name": "PDF", "description": "PDF documents with text extraction"},
    }

    return {
        "supported_extensions": [f".{ext}" for ext in extensions],
        "formats": {ext: format_info.get(ext, {"name": ext.upper(), "description": ""}) for ext in extensions},
    }


# =============================================================================
# Diagnostic Endpoints
# =============================================================================


@app.get("/api/diagnostic/neo4j", tags=["Diagnostic"])
async def diagnose_neo4j() -> dict[str, Any]:
    """
    Diagnose Neo4j database status.

    Returns detailed information about:
    - Connection status
    - Node and relationship counts
    - Index status (vector, fulltext)
    - Sample data preview
    """
    try:
        client = get_graph_client()

        # Get basic stats
        stats_query = """
        CALL {
            MATCH (e:Entity) RETURN 'Entity' as label, count(e) as count
            UNION ALL
            MATCH (c:Chunk) RETURN 'Chunk' as label, count(c) as count
            UNION ALL
            MATCH (comm:Community) RETURN 'Community' as label, count(comm) as count
        }
        RETURN label, count
        """
        stats_result = await client.execute_cypher(stats_query)
        node_counts = {r["label"]: r["count"] for r in stats_result}

        # Get relationship counts
        rel_query = """
        CALL {
            MATCH ()-[r:RELATES_TO]->() RETURN 'RELATES_TO' as type, count(r) as count
            UNION ALL
            MATCH ()-[r:CONTAINS]->() RETURN 'CONTAINS' as type, count(r) as count
            UNION ALL
            MATCH ()-[r:BELONGS_TO]->() RETURN 'BELONGS_TO' as type, count(r) as count
        }
        RETURN type, count
        """
        rel_result = await client.execute_cypher(rel_query)
        rel_counts = {r["type"]: r["count"] for r in rel_result}

        # Check indexes
        index_query = "SHOW INDEXES"
        indexes = await client.execute_cypher(index_query)

        vector_indexes = [
            {"name": idx.get("name"), "state": idx.get("state"), "type": idx.get("type")}
            for idx in indexes
            if idx.get("type") == "VECTOR"
        ]
        fulltext_indexes = [
            {"name": idx.get("name"), "state": idx.get("state"), "type": idx.get("type")}
            for idx in indexes
            if idx.get("type") == "FULLTEXT"
        ]

        # Check for embeddings
        embedding_query = """
        MATCH (e:Entity) WHERE e.embedding IS NOT NULL
        RETURN count(e) as entities_with_embedding
        """
        embedding_result = await client.execute_cypher(embedding_query)
        entities_with_embeddings = embedding_result[0]["entities_with_embedding"] if embedding_result else 0

        chunk_embedding_query = """
        MATCH (c:Chunk) WHERE c.embedding IS NOT NULL
        RETURN count(c) as chunks_with_embedding
        """
        chunk_emb_result = await client.execute_cypher(chunk_embedding_query)
        chunks_with_embeddings = chunk_emb_result[0]["chunks_with_embedding"] if chunk_emb_result else 0

        # Sample entities
        sample_query = """
        MATCH (e:Entity)
        RETURN e.id as id, e.name as name, e.type as type,
               e.description as description,
               CASE WHEN e.embedding IS NOT NULL THEN true ELSE false END as has_embedding
        LIMIT 5
        """
        sample_entities = await client.execute_cypher(sample_query)

        # Sample chunks
        chunk_sample_query = """
        MATCH (c:Chunk)
        RETURN c.id as id, substring(c.text, 0, 100) as text_preview, c.source as source,
               CASE WHEN c.embedding IS NOT NULL THEN true ELSE false END as has_embedding
        LIMIT 3
        """
        sample_chunks = await client.execute_cypher(chunk_sample_query)

        return {
            "status": "connected",
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "embeddings": {
                "entities_with_embeddings": entities_with_embeddings,
                "chunks_with_embeddings": chunks_with_embeddings,
                "total_entities": node_counts.get("Entity", 0),
                "total_chunks": node_counts.get("Chunk", 0),
            },
            "indexes": {
                "vector_indexes": vector_indexes,
                "fulltext_indexes": fulltext_indexes,
                "vector_index_count": len(vector_indexes),
                "fulltext_index_count": len(fulltext_indexes),
            },
            "sample_entities": sample_entities,
            "sample_chunks": sample_chunks,
            "recommendations": _generate_recommendations(
                node_counts, entities_with_embeddings, chunks_with_embeddings,
                len(vector_indexes), len(fulltext_indexes)
            ),
        }

    except Exception as e:
        logger.error("Neo4j diagnosis failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "recommendations": ["Check Neo4j connection settings", "Ensure Neo4j is running"],
        }


def _generate_recommendations(
    node_counts: dict[str, int],
    entities_with_embeddings: int,
    chunks_with_embeddings: int,
    vector_index_count: int,
    fulltext_index_count: int,
) -> list[str]:
    """Generate recommendations based on diagnostic results."""
    recommendations = []

    total_entities = node_counts.get("Entity", 0)
    total_chunks = node_counts.get("Chunk", 0)

    if total_entities == 0 and total_chunks == 0:
        recommendations.append("No data in Neo4j. Please upload documents using the Data Ingestion feature.")

    if total_entities > 0 and entities_with_embeddings == 0:
        recommendations.append("Entities exist but have no embeddings. Re-run ingestion to generate embeddings.")

    if total_chunks > 0 and chunks_with_embeddings == 0:
        recommendations.append("Chunks exist but have no embeddings. Re-run ingestion to generate embeddings.")

    if vector_index_count < 2:
        recommendations.append("Vector indexes missing. Restart the API server to create indexes.")

    if fulltext_index_count < 3:
        recommendations.append("Fulltext indexes missing. Restart the API server to create indexes.")

    if entities_with_embeddings > 0 and vector_index_count >= 2 and fulltext_index_count >= 3:
        recommendations.append("System appears ready for querying.")

    return recommendations


@app.post("/api/diagnostic/test-search", tags=["Diagnostic"])
async def test_search(query: str) -> dict[str, Any]:
    """
    Test search functionality with a sample query.

    Tests both fulltext and vector search capabilities.
    """
    try:
        client = get_graph_client()
        results: dict[str, Any] = {
            "query": query,
            "fulltext_search": {"status": "not_tested", "results": []},
            "entity_lookup": {"status": "not_tested", "results": []},
        }

        # Test fulltext search
        try:
            fulltext_results = await client.fulltext_search(
                query_text=query,
                top_k=5,
                min_score=0.1,
            )
            results["fulltext_search"] = {
                "status": "success",
                "count": len(fulltext_results),
                "results": [
                    {
                        "node_id": r.node_id,
                        "label": r.node_label,
                        "text": r.text[:100] if r.text else "",
                        "score": r.score,
                    }
                    for r in fulltext_results
                ],
            }
        except Exception as e:
            results["fulltext_search"] = {"status": "error", "error": str(e)}

        # Test entity lookup
        try:
            entity_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query)
               OR toLower(coalesce(e.description, '')) CONTAINS toLower($query)
            RETURN e.id as id, e.name as name, e.type as type, e.description as description
            LIMIT 5
            """
            entity_results = await client.execute_cypher(entity_query, {"query": query})
            results["entity_lookup"] = {
                "status": "success",
                "count": len(entity_results),
                "results": entity_results,
            }
        except Exception as e:
            results["entity_lookup"] = {"status": "error", "error": str(e)}

        return results

    except Exception as e:
        logger.error("Search test failed", error=str(e))
        return {"status": "error", "error": str(e)}


@app.post("/api/diagnostic/setup-schema", tags=["Diagnostic"])
async def setup_schema_endpoint() -> dict[str, Any]:
    """
    Manually trigger schema and index setup.

    Use this if indexes are missing after startup.
    """
    try:
        client = get_graph_client()
        result = await client.setup_schema()
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        logger.error("Schema setup failed", error=str(e))
        return {"status": "error", "error": str(e)}


@app.post("/api/diagnostic/test-extraction", tags=["Diagnostic"])
async def test_extraction(
    text: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = "",
) -> dict[str, Any]:
    """
    Test entity extraction on a sample text.

    Useful for diagnosing LLM API issues.
    """
    import traceback
    from src.sddi.extractors.entity_extractor import BatchEntityExtractor
    from src.sddi.state import TextChunk

    results: dict[str, Any] = {
        "input_text": text[:200] + "..." if len(text) > 200 else text,
        "steps": [],
    }

    try:
        # Step 1: Setup LLM
        results["steps"].append({"step": "setup_llm", "status": "starting"})

        if llm_api_key:
            if llm_provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model=llm_model,
                    temperature=settings.llm.temperature,
                    top_p=settings.llm.top_p,
                    api_key=llm_api_key,
                )
            else:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=llm_model,
                    temperature=settings.llm.temperature,
                    seed=settings.llm.seed,
                    top_p=settings.llm.top_p,
                    api_key=llm_api_key,
                )
        else:
            from langchain_openai import ChatOpenAI
            api_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
            if not api_key:
                return {
                    "status": "error",
                    "error": "No API key provided and no default API key configured",
                    "steps": results["steps"],
                }
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=settings.llm.temperature,
                seed=settings.llm.seed,
                top_p=settings.llm.top_p,
                api_key=api_key,
            )

        results["steps"][-1]["status"] = "success"
        results["steps"][-1]["llm_provider"] = llm_provider
        results["steps"][-1]["llm_model"] = llm_model

        # Step 2: Create test chunk
        results["steps"].append({"step": "create_chunk", "status": "starting"})

        chunk = TextChunk(
            id="test_chunk_001",
            text=text,
            doc_id="test_doc",
            position=0,
            start_char=0,
            end_char=len(text),
            metadata={"source": "test"},
        )

        results["steps"][-1]["status"] = "success"
        results["steps"][-1]["chunk_length"] = len(text)

        # Step 3: Extract entities
        results["steps"].append({"step": "extract_entities", "status": "starting"})

        extractor = BatchEntityExtractor(llm=llm, min_confidence=0.3)
        entities = await extractor.extract_batch([chunk], deduplicate=True)

        results["steps"][-1]["status"] = "success"
        results["steps"][-1]["entities_found"] = len(entities)

        # Format entities for response
        results["entities"] = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type.value,
                "description": e.description,
                "confidence": e.confidence,
            }
            for e in entities
        ]

        results["status"] = "success"
        results["summary"] = f"Extracted {len(entities)} entities from {len(text)} characters"

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("Test extraction failed", error=str(e), traceback=error_trace)
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = error_trace
        if results["steps"]:
            results["steps"][-1]["status"] = "failed"
            results["steps"][-1]["error"] = str(e)

    return results


@app.post("/api/diagnostic/test-pipeline", tags=["Diagnostic"])
async def test_pipeline(
    text: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = "",
) -> dict[str, Any]:
    """
    Test the complete SDDI pipeline on sample text.

    Runs: Chunk → Extract Entities → Extract Relations → Embed → Load
    """
    import traceback
    from src.sddi.pipeline import SDDIPipeline
    from src.sddi.loaders.neo4j_loader import Neo4jLoader
    from src.sddi.state import RawDocument

    results: dict[str, Any] = {
        "input_text": text[:200] + "..." if len(text) > 200 else text,
        "pipeline_steps": {},
    }

    try:
        # Setup LLM and embeddings
        if llm_api_key:
            if llm_provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                from langchain_openai import OpenAIEmbeddings

                llm = ChatAnthropic(
                    model=llm_model,
                    temperature=settings.llm.temperature,
                    top_p=settings.llm.top_p,
                    api_key=llm_api_key,
                )
                # Need OpenAI key for embeddings
                embed_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else llm_api_key
                embeddings = OpenAIEmbeddings(api_key=embed_key)
            else:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                llm = ChatOpenAI(
                    model=llm_model,
                    temperature=settings.llm.temperature,
                    seed=settings.llm.seed,
                    top_p=settings.llm.top_p,
                    api_key=llm_api_key,
                )
                embeddings = OpenAIEmbeddings(api_key=llm_api_key)
        else:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            api_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
            if not api_key:
                return {"status": "error", "error": "No API key configured"}
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=settings.llm.temperature,
                seed=settings.llm.seed,
                top_p=settings.llm.top_p,
                api_key=api_key,
            )
            embeddings = OpenAIEmbeddings(api_key=api_key)

        results["pipeline_steps"]["llm_setup"] = "success"

        # Create pipeline
        neo4j_loader = Neo4jLoader(client=get_graph_client())
        pipeline = SDDIPipeline(
            llm=llm,
            embeddings=embeddings,
            neo4j_loader=neo4j_loader,
            chunk_size=500,
            chunk_overlap=50,
        )

        # Create test document
        doc = RawDocument(
            id="test_doc_001",
            content=text,
            source="diagnostic_test",
            metadata={"test": True},
        )

        results["pipeline_steps"]["pipeline_created"] = "success"

        # Run pipeline
        logger.info("Running diagnostic pipeline test")
        final_state = await pipeline.run([doc], pipeline_id="diag_test")

        # Collect results
        results["pipeline_steps"]["pipeline_completed"] = "success"
        results["chunks_created"] = len(final_state.get("chunks", []))
        results["entities_extracted"] = len(final_state.get("entities", []))
        results["relations_extracted"] = len(final_state.get("relations", []))
        results["embeddings_created"] = len(final_state.get("embeddings", {}))

        load_result = final_state.get("load_result")
        if load_result:
            results["neo4j_loaded"] = {
                "chunks": load_result.chunks_created,
                "entities": load_result.entities_created,
                "relations": load_result.relations_created,
            }

        results["errors"] = final_state.get("errors", [])
        results["load_status"] = final_state.get("load_status", "unknown")
        if hasattr(results["load_status"], "value"):
            results["load_status"] = results["load_status"].value

        # Entity details
        entities = final_state.get("entities", [])
        results["entity_details"] = [
            {"name": e.name, "type": e.type.value, "confidence": e.confidence}
            for e in entities[:10]  # First 10
        ]

        results["status"] = "success" if results["entities_extracted"] > 0 else "warning"
        results["summary"] = (
            f"Chunks: {results['chunks_created']}, "
            f"Entities: {results['entities_extracted']}, "
            f"Relations: {results['relations_extracted']}"
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("Test pipeline failed", error=str(e), traceback=error_trace)
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = error_trace

    return results


@app.post("/api/diagnostic/load-sample-data", tags=["Diagnostic"])
async def load_sample_data() -> dict[str, Any]:
    """
    Load sample Red Team data directly into Neo4j for testing.

    This bypasses the LLM extraction and loads pre-defined entities and chunks.
    """
    import hashlib

    client = get_graph_client()
    await client.connect()

    # Setup schema first
    try:
        await client.setup_schema()
    except Exception as e:
        logger.warning("Schema setup warning", error=str(e))

    results: dict[str, Any] = {
        "chunks_created": 0,
        "entities_created": 0,
        "relations_created": 0,
        "errors": [],
    }

    # Sample data about Red Team
    sample_entities = [
        {
            "id": hashlib.sha256("CONCEPT:레드팀".encode()).hexdigest()[:16],
            "name": "레드팀",
            "type": "CONCEPT",
            "description": "AI 시스템의 취약점을 찾기 위해 적대적 공격을 시뮬레이션하는 보안 테스트 방법론",
            "aliases": ["Red Team", "레드 팀", "적색팀"],
        },
        {
            "id": hashlib.sha256("CONCEPT:AI Red Teaming".encode()).hexdigest()[:16],
            "name": "AI Red Teaming",
            "type": "CONCEPT",
            "description": "AI 모델과 시스템의 보안 취약점을 평가하기 위한 공격 시뮬레이션 기법",
            "aliases": ["AI 레드티밍", "인공지능 레드팀"],
        },
        {
            "id": hashlib.sha256("ORGANIZATION:Microsoft AIRT".encode()).hexdigest()[:16],
            "name": "Microsoft AI Red Team",
            "type": "ORGANIZATION",
            "description": "2018년 설립된 Microsoft의 AI 보안 테스트 전문 팀 (AIRT)",
            "aliases": ["AIRT", "MS AI Red Team"],
        },
        {
            "id": hashlib.sha256("CONCEPT:보안 테스트".encode()).hexdigest()[:16],
            "name": "보안 테스트",
            "type": "CONCEPT",
            "description": "시스템의 보안 취약점을 발견하기 위한 테스트 활동",
            "aliases": ["Security Testing", "침투 테스트"],
        },
        {
            "id": hashlib.sha256("CONCEPT:적대적 공격".encode()).hexdigest()[:16],
            "name": "적대적 공격",
            "type": "CONCEPT",
            "description": "AI 시스템을 속이거나 오작동하게 만드는 악의적 입력",
            "aliases": ["Adversarial Attack", "적대적 예제"],
        },
    ]

    sample_chunks = [
        {
            "id": hashlib.sha256("chunk:redteam:1".encode()).hexdigest()[:16],
            "text": "레드팀(Red Team)은 AI 시스템의 보안 취약점을 찾기 위해 적대적 공격을 시뮬레이션하는 보안 테스트 방법론입니다. 레드팀은 실제 공격자의 관점에서 시스템을 테스트하여 잠재적인 위협을 발견합니다.",
            "doc_id": "sample_doc_001",
            "position": 0,
            "source": "sample_redteam_doc",
        },
        {
            "id": hashlib.sha256("chunk:redteam:2".encode()).hexdigest()[:16],
            "text": "AI Red Teaming은 AI 모델과 시스템의 보안 취약점을 평가하기 위한 공격 시뮬레이션 기법입니다. 프롬프트 인젝션, 탈옥, 적대적 예제 등 다양한 공격 기법을 사용합니다.",
            "doc_id": "sample_doc_001",
            "position": 1,
            "source": "sample_redteam_doc",
        },
        {
            "id": hashlib.sha256("chunk:redteam:3".encode()).hexdigest()[:16],
            "text": "Microsoft AI Red Team(AIRT)은 2018년에 설립되어 Microsoft의 AI 제품과 서비스의 보안을 테스트합니다. AIRT는 LLM 보안, 컴퓨터 비전 모델 공격, 멀티모달 시스템 취약점 등을 연구합니다.",
            "doc_id": "sample_doc_001",
            "position": 2,
            "source": "sample_redteam_doc",
        },
    ]

    sample_relations = [
        {
            "source_id": sample_entities[0]["id"],  # 레드팀
            "target_id": sample_entities[3]["id"],  # 보안 테스트
            "type": "IS_A",
            "predicate": "is a type of",
        },
        {
            "source_id": sample_entities[1]["id"],  # AI Red Teaming
            "target_id": sample_entities[0]["id"],  # 레드팀
            "type": "SPECIALIZES",
            "predicate": "specializes",
        },
        {
            "source_id": sample_entities[2]["id"],  # Microsoft AIRT
            "target_id": sample_entities[1]["id"],  # AI Red Teaming
            "type": "PERFORMS",
            "predicate": "performs",
        },
        {
            "source_id": sample_entities[0]["id"],  # 레드팀
            "target_id": sample_entities[4]["id"],  # 적대적 공격
            "type": "USES",
            "predicate": "uses",
        },
    ]

    try:
        # Load entities one by one (maximum compatibility)
        for entity in sample_entities:
            entity_query = """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.type = $type,
                e.description = $description,
                e.aliases = $aliases,
                e.confidence = 0.95
            """
            await client.execute_cypher(entity_query, entity)
            results["entities_created"] += 1

        # Load chunks one by one
        for chunk in sample_chunks:
            chunk_query = """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.source = $source,
                c.doc_id = $doc_id,
                c.position = $position
            """
            await client.execute_cypher(chunk_query, chunk)
            results["chunks_created"] += 1

        # Load relations one by one
        for rel in sample_relations:
            relation_query = """
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})
            MERGE (source)-[r:RELATES_TO]->(target)
            SET r.type = $type,
                r.predicate = $predicate
            """
            await client.execute_cypher(relation_query, rel)
            results["relations_created"] += 1

        # Create chunk-entity links
        link_query = """
        MATCH (c:Chunk), (e:Entity)
        WHERE c.text CONTAINS e.name
        MERGE (c)-[:CONTAINS]->(e)
        """
        await client.execute_cypher(link_query)

        results["status"] = "success"
        results["message"] = (
            f"Loaded {results['entities_created']} entities, "
            f"{results['chunks_created']} chunks, "
            f"{results['relations_created']} relations"
        )

        # Verify data
        verify_query = "MATCH (e:Entity) RETURN count(e) as count"
        verify_result = await client.execute_cypher(verify_query)
        results["verification"] = {
            "total_entities": verify_result[0]["count"] if verify_result else 0
        }

    except Exception as e:
        import traceback
        logger.error("Sample data load failed", error=str(e))
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


@app.delete("/api/diagnostic/clear-data", tags=["Diagnostic"])
async def clear_all_data() -> dict[str, Any]:
    """
    Clear all data from Neo4j (for testing purposes).

    WARNING: This deletes ALL nodes and relationships!
    """
    client = get_graph_client()
    await client.connect()

    try:
        # Delete all nodes and relationships
        delete_query = "MATCH (n) DETACH DELETE n"
        await client.execute_cypher(delete_query)

        # Verify
        count_query = "MATCH (n) RETURN count(n) as count"
        result = await client.execute_cypher(count_query)
        remaining = result[0]["count"] if result else 0

        return {
            "status": "success",
            "message": "All data cleared",
            "remaining_nodes": remaining,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# Embeddings Management
# =============================================================================


@app.post("/api/embeddings/regenerate", tags=["Embeddings"])
async def regenerate_embeddings(
    llm_provider: str = Form(default="openai"),
    llm_api_key: str = Form(default=""),
    batch_size: int = Form(default=100),
    include_entities: bool = Form(default=True),
    include_chunks: bool = Form(default=True),
) -> dict[str, Any]:
    """
    Regenerate embeddings for existing entities and chunks.

    This is useful when:
    - Entities/chunks exist but have no embeddings (legacy data)
    - Embeddings need to be updated with a new model
    - Original ingestion failed at the embedding stage

    Args:
        llm_provider: LLM provider (openai for embeddings)
        llm_api_key: API key for embedding generation
        batch_size: Number of items to process per batch
        include_entities: Whether to regenerate entity embeddings
        include_chunks: Whether to regenerate chunk embeddings

    Returns:
        Summary of embedding regeneration results
    """
    client = get_graph_client()
    await client.connect()

    results: dict[str, Any] = {
        "status": "success",
        "entities_processed": 0,
        "chunks_processed": 0,
        "entities_with_embeddings_before": 0,
        "chunks_with_embeddings_before": 0,
        "entities_with_embeddings_after": 0,
        "chunks_with_embeddings_after": 0,
        "errors": [],
    }

    try:
        # Setup embeddings
        from langchain_openai import OpenAIEmbeddings

        api_key = llm_api_key or (
            settings.llm.openai_api_key.get_secret_value()
            if settings.llm.openai_api_key else None
        )
        if not api_key:
            return {"status": "error", "error": "No OpenAI API key provided"}

        embeddings = OpenAIEmbeddings(api_key=api_key)

        # Get current embedding counts
        count_query = """
        MATCH (e:Entity)
        WITH count(e) as total,
             sum(CASE WHEN e.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_emb
        RETURN total, with_emb as with_embedding
        """
        entity_count = await client.execute_cypher(count_query)
        if entity_count:
            results["entities_with_embeddings_before"] = entity_count[0]["with_embedding"]

        chunk_count_query = """
        MATCH (c:Chunk)
        WITH count(c) as total,
             sum(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_emb
        RETURN total, with_emb as with_embedding
        """
        chunk_count = await client.execute_cypher(chunk_count_query)
        if chunk_count:
            results["chunks_with_embeddings_before"] = chunk_count[0]["with_embedding"]

        # Process entities
        if include_entities:
            logger.info("Regenerating entity embeddings...")

            # Get entities without embeddings first, then all entities
            entity_query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NULL
            RETURN e.id as id, e.name as name, coalesce(e.description, '') as description
            """
            entities_without = await client.execute_cypher(entity_query)

            if entities_without:
                # Process in batches
                for i in range(0, len(entities_without), batch_size):
                    batch = entities_without[i:i + batch_size]
                    texts = [
                        f"{e['name']}: {e['description']}" if e['description'] else e['name']
                        for e in batch
                    ]

                    try:
                        batch_embeddings = await embeddings.aembed_documents(texts)

                        # Update each entity
                        for entity, embedding in zip(batch, batch_embeddings):
                            update_query = """
                            MATCH (e:Entity {id: $id})
                            SET e.embedding = $embedding
                            """
                            await client.execute_cypher(update_query, {
                                "id": entity["id"],
                                "embedding": embedding,
                            })
                            results["entities_processed"] += 1

                        logger.info(
                            f"Processed entity batch {i // batch_size + 1}",
                            batch_size=len(batch),
                            total_processed=results["entities_processed"],
                        )
                    except Exception as e:
                        error_msg = f"Entity batch {i // batch_size + 1} failed: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)

        # Process chunks
        if include_chunks:
            logger.info("Regenerating chunk embeddings...")

            chunk_query = """
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL
            RETURN c.id as id, c.text as text
            """
            chunks_without = await client.execute_cypher(chunk_query)

            if chunks_without:
                for i in range(0, len(chunks_without), batch_size):
                    batch = chunks_without[i:i + batch_size]
                    texts = [c["text"] for c in batch]

                    try:
                        batch_embeddings = await embeddings.aembed_documents(texts)

                        for chunk, embedding in zip(batch, batch_embeddings):
                            update_query = """
                            MATCH (c:Chunk {id: $id})
                            SET c.embedding = $embedding
                            """
                            await client.execute_cypher(update_query, {
                                "id": chunk["id"],
                                "embedding": embedding,
                            })
                            results["chunks_processed"] += 1

                        logger.info(
                            f"Processed chunk batch {i // batch_size + 1}",
                            batch_size=len(batch),
                            total_processed=results["chunks_processed"],
                        )
                    except Exception as e:
                        error_msg = f"Chunk batch {i // batch_size + 1} failed: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)

        # Get final counts
        entity_final = await client.execute_cypher(count_query)
        if entity_final:
            results["entities_with_embeddings_after"] = entity_final[0]["with_embedding"]

        chunk_final = await client.execute_cypher(chunk_count_query)
        if chunk_final:
            results["chunks_with_embeddings_after"] = chunk_final[0]["with_embedding"]

        results["summary"] = (
            f"Processed {results['entities_processed']} entities, "
            f"{results['chunks_processed']} chunks. "
            f"Entities with embeddings: {results['entities_with_embeddings_before']} → "
            f"{results['entities_with_embeddings_after']}. "
            f"Chunks with embeddings: {results['chunks_with_embeddings_before']} → "
            f"{results['chunks_with_embeddings_after']}."
        )

        logger.info("Embedding regeneration completed", **results)

    except Exception as e:
        import traceback
        logger.error("Embedding regeneration failed", error=str(e), traceback=traceback.format_exc())
        results["status"] = "error"
        results["error"] = str(e)

    return results


@app.get("/api/embeddings/status", tags=["Embeddings"])
async def get_embeddings_status() -> dict[str, Any]:
    """
    Get current embedding status for entities and chunks.

    Returns counts of items with and without embeddings.
    """
    client = get_graph_client()
    await client.connect()

    try:
        # Entity embedding status
        entity_query = """
        MATCH (e:Entity)
        WITH count(e) as total,
             sum(CASE WHEN e.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
        RETURN total, with_embedding, total - with_embedding as without_embedding
        """
        entity_result = await client.execute_cypher(entity_query)

        # Chunk embedding status
        chunk_query = """
        MATCH (c:Chunk)
        WITH count(c) as total,
             sum(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
        RETURN total, with_embedding, total - with_embedding as without_embedding
        """
        chunk_result = await client.execute_cypher(chunk_query)

        entity_stats = entity_result[0] if entity_result else {"total": 0, "with_embedding": 0, "without_embedding": 0}
        chunk_stats = chunk_result[0] if chunk_result else {"total": 0, "with_embedding": 0, "without_embedding": 0}

        # Calculate percentages
        entity_pct = (entity_stats["with_embedding"] / entity_stats["total"] * 100) if entity_stats["total"] > 0 else 0
        chunk_pct = (chunk_stats["with_embedding"] / chunk_stats["total"] * 100) if chunk_stats["total"] > 0 else 0

        needs_regeneration = (
            entity_stats["without_embedding"] > 0 or
            chunk_stats["without_embedding"] > 0
        )

        return {
            "entities": {
                "total": entity_stats["total"],
                "with_embedding": entity_stats["with_embedding"],
                "without_embedding": entity_stats["without_embedding"],
                "coverage_percent": round(entity_pct, 1),
            },
            "chunks": {
                "total": chunk_stats["total"],
                "with_embedding": chunk_stats["with_embedding"],
                "without_embedding": chunk_stats["without_embedding"],
                "coverage_percent": round(chunk_pct, 1),
            },
            "needs_regeneration": needs_regeneration,
            "recommendation": (
                "Run POST /api/embeddings/regenerate to generate missing embeddings"
                if needs_regeneration else
                "All items have embeddings - system ready for semantic search"
            ),
        }

    except Exception as e:
        logger.error("Failed to get embedding status", error=str(e))
        return {"status": "error", "error": str(e)}


# =============================================================================
# Validation Endpoints
# =============================================================================


class ValidationRequest(BaseModel):
    """Request model for validation endpoints."""
    doc_path: str | None = Field(default=None, description="Document path")
    doc_content: str | None = Field(default=None, description="Document content")
    query: str | None = Field(default=None, description="Query to validate")
    strict_mode: bool = Field(default=False, description="Use stricter thresholds")
    ingestion_result: dict[str, Any] | None = Field(default=None)
    entity_result: dict[str, Any] | None = Field(default=None)
    relation_result: dict[str, Any] | None = Field(default=None)
    query_result: dict[str, Any] | None = Field(default=None)
    retrieval_result: dict[str, Any] | None = Field(default=None)
    response_result: dict[str, Any] | None = Field(default=None)


@app.post("/api/validation/ingestion", tags=["Validation"])
async def validate_ingestion(request: ValidationRequest) -> dict[str, Any]:
    """
    Validate document ingestion results.

    Validates:
    - Document ingestion status and chunk creation
    - Entity extraction quality and coverage
    - Relation extraction and cardinality compliance

    Returns detailed validation report with metrics and recommendations.
    """
    try:
        from src.validation import PipelineValidator

        validator = PipelineValidator(strict_mode=request.strict_mode)
        report = await validator.validate_ingestion(
            doc_path=request.doc_path,
            doc_content=request.doc_content,
            ingestion_result=request.ingestion_result,
            entity_result=request.entity_result,
            relation_result=request.relation_result,
        )

        return report.to_dict()

    except Exception as e:
        logger.error("Ingestion validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e


@app.post("/api/validation/query", tags=["Validation"])
async def validate_query(request: ValidationRequest) -> dict[str, Any]:
    """
    Validate query processing results.

    Validates:
    - Query processing and topic entity extraction
    - Evidence retrieval coverage and relevance
    - Response generation quality and confidence

    Returns detailed validation report with metrics and recommendations.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        from src.validation import PipelineValidator

        validator = PipelineValidator(strict_mode=request.strict_mode)
        report = await validator.validate_query(
            query=request.query,
            query_result=request.query_result,
            retrieval_result=request.retrieval_result,
            response_result=request.response_result,
        )

        return report.to_dict()

    except Exception as e:
        logger.error("Query validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e


@app.post("/api/validation/full", tags=["Validation"])
async def validate_full_pipeline(request: ValidationRequest) -> dict[str, Any]:
    """
    Validate the complete end-to-end pipeline.

    Validates all steps:
    1. Document Ingestion
    2. Entity Extraction
    3. Relation Extraction
    4. Query Processing
    5. Evidence Retrieval
    6. Response Generation

    Returns comprehensive validation report with all metrics and recommendations.
    """
    try:
        from src.validation import PipelineValidator

        validator = PipelineValidator(strict_mode=request.strict_mode)
        report = await validator.validate_full_pipeline(
            doc_path=request.doc_path,
            doc_content=request.doc_content,
            query=request.query,
            ingestion_result=request.ingestion_result,
            entity_result=request.entity_result,
            relation_result=request.relation_result,
            query_result=request.query_result,
            retrieval_result=request.retrieval_result,
            response_result=request.response_result,
        )

        return report.to_dict()

    except Exception as e:
        logger.error("Full pipeline validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e


@app.post("/api/validation/context", tags=["Validation"])
async def extract_context(
    doc_path: str | None = None,
    doc_content: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """
    Extract validation context from document and/or query.

    Performs dynamic context extraction:
    - Document format, language, and domain detection
    - Query type classification (definition, comparison, procedure, etc.)
    - Complexity assessment (simple, moderate, complex, multi-hop)
    - Sets validation expectations based on context

    Useful for understanding how the validation framework will evaluate results.
    """
    try:
        from src.validation import ContextExtractor

        extractor = ContextExtractor()
        context = extractor.extract_full_context(
            document_path=doc_path,
            document_content=doc_content,
            query=query,
        )

        return context.to_dict()

    except Exception as e:
        logger.error("Context extraction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Context extraction failed: {str(e)}") from e


@app.get("/api/validation/query-types", tags=["Validation"])
async def get_query_types() -> dict[str, Any]:
    """
    Get all supported query type classifications.

    Returns the query types used by the validation framework:
    - DEFINITION: "what is X?"
    - COMPARISON: "difference between X and Y"
    - PROCEDURE: "how to do X"
    - CAUSAL: "why does X happen"
    - LIST: "types of X"
    - etc.

    Includes pattern examples for each type in Korean and English.
    """
    from src.validation.context_extractor import (
        QueryType,
        QUERY_TYPE_PATTERNS,
        QUERY_TYPE_EXPECTATIONS,
    )

    result = {}
    for qtype in QueryType:
        patterns = QUERY_TYPE_PATTERNS.get(qtype, {})
        expectations = QUERY_TYPE_EXPECTATIONS.get(qtype, {})

        result[qtype.value] = {
            "patterns": {
                "korean": patterns.get("ko", [])[:5],
                "english": patterns.get("en", [])[:5],
            },
            "expected_answer_type": expectations.get("answer_type", "inferred").value if hasattr(expectations.get("answer_type"), "value") else str(expectations.get("answer_type", "inferred")),
            "min_confidence": expectations.get("min_confidence", 0.5),
        }

    return {
        "query_types": result,
        "count": len(QueryType),
    }


@app.get("/api/validation/complexity-levels", tags=["Validation"])
async def get_complexity_levels() -> dict[str, Any]:
    """
    Get complexity level thresholds used for validation.

    Returns the complexity levels and their associated validation thresholds:
    - SIMPLE: Single concept, direct lookup
    - MODERATE: 2-3 concepts, some reasoning
    - COMPLEX: Multiple concepts, synthesis required
    - MULTI_HOP: Requires traversing multiple relationships
    """
    from src.validation.context_extractor import (
        Complexity,
        COMPLEXITY_THRESHOLDS,
    )

    result = {}
    for complexity in Complexity:
        thresholds = COMPLEXITY_THRESHOLDS.get(complexity, {})
        result[complexity.value] = {
            "min_evidence": thresholds.get("min_evidence", 3),
            "min_relevance": thresholds.get("min_relevance", 0.5),
            "max_concepts": thresholds.get("max_concepts", 1),
        }

    return {
        "complexity_levels": result,
        "count": len(Complexity),
    }


@app.get("/api/validation/domains", tags=["Validation"])
async def get_domains() -> dict[str, Any]:
    """
    Get domain classifications and their expected entity/relation types.

    Returns domains supported by the validation framework:
    - SECURITY: Cybersecurity domain
    - FINANCE: Financial domain
    - MEDICAL: Healthcare domain
    - LEGAL: Legal domain
    - TECH: Technology domain
    - ACADEMIC: Research/academic domain
    - GENERAL: General purpose
    """
    from src.validation.context_extractor import (
        Domain,
        DOMAIN_KEYWORDS,
        DOMAIN_ENTITY_TYPES,
        DOMAIN_RELATION_TYPES,
    )

    result = {}
    for domain in Domain:
        keywords = DOMAIN_KEYWORDS.get(domain, {})
        entity_types = DOMAIN_ENTITY_TYPES.get(domain, [])
        relation_types = DOMAIN_RELATION_TYPES.get(domain, [])

        result[domain.value] = {
            "keywords": {
                "korean": keywords.get("ko", [])[:5],
                "english": keywords.get("en", [])[:5],
            },
            "expected_entity_types": entity_types,
            "expected_relation_types": relation_types,
        }

    return {
        "domains": result,
        "count": len(Domain),
    }


# =============================================================================
# Query Optimization Endpoints
# =============================================================================


@app.post("/api/query/analyze", tags=["Query Optimization"])
async def analyze_cypher_query(query: str) -> dict[str, Any]:
    """
    Analyze a Cypher query for performance characteristics.

    Returns:
    - Complexity assessment
    - Estimated cost
    - Optimization suggestions
    - Warnings about potential issues
    """
    from src.graph.query_optimizer import get_query_optimizer

    optimizer = get_query_optimizer()
    analysis = optimizer.analyze_query(query)

    return {
        "complexity": analysis.complexity.value,
        "estimated_cost": analysis.estimated_cost,
        "has_limit": analysis.has_limit,
        "has_where": analysis.has_where,
        "pattern_count": analysis.pattern_count,
        "uses_index": analysis.uses_index,
        "warnings": analysis.warnings,
        "suggestions": analysis.suggestions,
    }


@app.post("/api/query/optimize", tags=["Query Optimization"])
async def optimize_cypher_query(
    query: str,
    max_results: int | None = None,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    """
    Optimize a Cypher query by adding limits and analyzing performance.

    Args:
        query: The Cypher query to optimize
        max_results: Maximum results (overrides default)
        timeout_ms: Query timeout in milliseconds

    Returns:
        Optimized query with metadata
    """
    from src.graph.query_optimizer import get_query_optimizer

    optimizer = get_query_optimizer()
    optimized_query, metadata = optimizer.optimize_query(
        query, max_results=max_results, timeout_ms=timeout_ms
    )

    return {
        "original_query": query,
        "optimized_query": optimized_query,
        **metadata,
    }


@app.get("/api/query/optimization-config", tags=["Query Optimization"])
async def get_query_optimization_config() -> dict[str, Any]:
    """
    Get current query optimization configuration.

    Returns:
    - Default and max limits
    - Timeout settings
    - Auto-optimization flags
    """
    return {
        "default_limit": settings.neo4j.default_query_limit,
        "max_limit": settings.neo4j.max_query_limit,
        "default_timeout_ms": settings.neo4j.query_timeout_ms,
        "max_timeout_ms": settings.neo4j.max_query_timeout_ms,
        "auto_inject_limit": settings.neo4j.auto_inject_limit,
    }


# =============================================================================
# Factory Function
# =============================================================================


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )

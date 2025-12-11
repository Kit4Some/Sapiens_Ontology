"""
Application Settings - Pydantic Settings for configuration management.

Supports environment variables and .env file loading.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_to_lowercase(v: str) -> str:
    """Normalize string to lowercase."""
    if isinstance(v, str):
        return v.lower()
    return v


class Neo4jSettings(BaseSettings):
    """Neo4j database connection settings."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_")

    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    username: str = Field(default="neo4j", description="Neo4j username")
    password: SecretStr = Field(default=SecretStr("password"), description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    max_connection_pool_size: int = Field(default=50, description="Connection pool size")

    # Query optimization settings
    query_timeout_ms: int = Field(default=30000, description="Query timeout in milliseconds")
    max_query_timeout_ms: int = Field(default=120000, description="Maximum query timeout")
    default_query_limit: int = Field(default=1000, description="Default LIMIT for queries")
    max_query_limit: int = Field(default=10000, description="Maximum LIMIT for queries")
    auto_inject_limit: bool = Field(default=True, description="Auto-inject LIMIT clauses")


class LLMSettings(BaseSettings):
    """LLM provider settings with multi-provider failover support."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    # Primary provider (case-insensitive via BeforeValidator)
    provider: Annotated[
        Literal["openai", "anthropic", "azure", "local"],
        BeforeValidator(normalize_to_lowercase),
    ] = Field(default="openai", description="Primary LLM provider")

    # API Keys
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key")

    # Azure OpenAI settings
    azure_openai_api_key: SecretStr | None = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: str | None = Field(default=None, description="Azure OpenAI endpoint URL")
    azure_openai_deployment: str | None = Field(default=None, description="Azure OpenAI deployment name")
    azure_openai_api_version: str = Field(default="2024-02-01", description="Azure OpenAI API version")

    # Local LLM settings (Ollama)
    local_llm_base_url: str = Field(default="http://localhost:11434", description="Local LLM base URL")
    local_llm_model: str = Field(default="llama3.2", description="Local LLM model name")

    # Model configurations
    reasoning_model: str = Field(
        default="gpt-4o-mini", description="Model for complex reasoning"
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022", description="Anthropic model name"
    )
    cypher_model: str = Field(
        default="gpt-4o-mini", description="Model for Cypher generation"
    )
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")

    temperature: float = Field(default=0.0, description="LLM temperature")
    seed: int = Field(default=42, description="Random seed for reproducibility (OpenAI/Azure only)")
    top_p: float = Field(default=1.0, description="Nucleus sampling threshold")
    max_tokens: int = Field(default=4096, description="Max tokens for response")

    # Failover settings
    enable_failover: bool = Field(default=True, description="Enable automatic provider failover")
    failover_timeout: int = Field(default=30, description="Timeout before failover (seconds)")
    max_retries_per_provider: int = Field(default=3, description="Max retries before failover")


class ToGSettings(BaseSettings):
    """ToG 3.0 MACER specific settings."""

    model_config = SettingsConfigDict(env_prefix="TOG_")

    max_reasoning_depth: int = Field(default=5, description="Maximum reasoning depth")
    exploration_width: int = Field(default=10, description="Graph exploration width")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold")
    enable_meta_cognition: bool = Field(default=True, description="Enable meta-cognitive layer")

    # QA Mode settings (General Document QA support)
    qa_mode: Annotated[
        Literal["benchmark", "general", "auto"],
        BeforeValidator(normalize_to_lowercase),
    ] = Field(
        default="auto",
        description="QA mode: 'benchmark' for HotpotQA-style short answers, "
                    "'general' for document QA with flexible answer lengths, "
                    "'auto' to detect based on question type"
    )
    default_max_answer_words: int = Field(
        default=100,
        description="Default max words for answers (overridden by question type in auto mode)"
    )
    enable_extended_question_types: bool = Field(
        default=True,
        description="Enable extended question types (DEFINITION, PROCEDURE, CAUSE_EFFECT, LIST, NARRATIVE, OPINION)"
    )
    enable_narrative_answers: bool = Field(
        default=True,
        description="Allow long narrative answers for complex questions"
    )


class WorkflowSettings(BaseSettings):
    """LangGraph workflow settings."""

    model_config = SettingsConfigDict(env_prefix="WORKFLOW_")

    max_iterations: int = Field(default=10, description="Maximum workflow iterations")
    timeout_seconds: int = Field(default=120, description="Workflow timeout in seconds")
    enable_checkpointing: bool = Field(default=True, description="Enable state checkpointing")


class APISettings(BaseSettings):
    """FastAPI server settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0", description="API host")  # nosec B104 - intentional for container deployment
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")
    max_upload_size_mb: int = Field(default=1024, description="Max file upload size in MB (default 1GB)")


class IngestionSettings(BaseSettings):
    """SDDI Ingestion pipeline settings."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size: int = Field(default=1000, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    batch_size: int = Field(default=10, description="Batch size for parallel processing")
    max_concurrent_chunks: int = Field(default=5, description="Max concurrent chunk processing")
    min_entity_confidence: float = Field(default=0.5, description="Min entity confidence threshold")
    min_relation_confidence: float = Field(default=0.5, description="Min relation confidence threshold")

    # Embedding settings
    embedding_dimensions: int = Field(
        default=1536,
        description="Expected embedding dimensions (must match Neo4j vector index). "
                    "Common values: OpenAI text-embedding-3-small=1536, text-embedding-3-large=3072, "
                    "Ollama models vary (768, 1024, etc.)"
    )
    validate_embedding_dimensions: bool = Field(
        default=True,
        description="Validate embedding dimensions before loading to Neo4j"
    )


class DistributedSettings(BaseSettings):
    """Distributed processing settings (Celery)."""

    model_config = SettingsConfigDict(env_prefix="DISTRIBUTED_")

    # Enable/disable distributed processing
    enabled: bool = Field(default=False, description="Enable distributed processing")

    # Celery configuration
    broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL (Redis recommended)"
    )
    result_backend: str = Field(
        default="redis://localhost:6379/1",
        description="Celery result backend URL"
    )

    # Worker configuration
    worker_concurrency: int = Field(default=4, description="Worker concurrency")
    worker_prefetch_multiplier: int = Field(default=2, description="Prefetch multiplier")

    # Task configuration
    task_time_limit: int = Field(default=600, description="Task time limit in seconds")
    task_soft_time_limit: int = Field(default=540, description="Task soft time limit")

    # Rate limiting
    extraction_rate_limit: str = Field(default="10/m", description="Entity extraction rate limit")
    embedding_rate_limit: str = Field(default="20/m", description="Embedding generation rate limit")

    # Batch sizes
    extraction_batch_size: int = Field(default=20, description="Chunks per extraction batch")
    embedding_batch_size: int = Field(default=50, description="Items per embedding batch")


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring settings."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_")

    # Logging
    log_format: Literal["json", "console"] = Field(
        default="json", description="Log format (json for production, console for development)"
    )

    # Tracing
    tracing_enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    tracing_exporter: Literal["otlp", "jaeger", "zipkin", "console", "none"] = Field(
        default="otlp", description="Trace exporter type"
    )
    tracing_endpoint: str = Field(default="http://localhost:4317", description="Tracing collector endpoint")
    tracing_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Trace sampling rate")

    # Metrics
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")

    # Alerting
    alerting_enabled: bool = Field(default=False, description="Enable alerting")
    alert_check_interval: int = Field(default=60, description="Alert check interval in seconds")
    slack_webhook_url: str | None = Field(default=None, description="Slack webhook URL for alerts")
    pagerduty_routing_key: str | None = Field(default=None, description="PagerDuty routing key")


class CheckpointingSettings(BaseSettings):
    """LangGraph checkpointing settings."""

    model_config = SettingsConfigDict(env_prefix="CHECKPOINT_")

    backend: Literal["memory", "postgres"] = Field(
        default="memory", description="Checkpointing backend (memory or postgres)"
    )
    postgres_uri: str = Field(
        default="postgresql://ontology:ontology_pass@localhost:5432/ontology_checkpoints",
        description="PostgreSQL connection URI for checkpointing",
    )


class TelemetrySettings(BaseSettings):
    """Telemetry settings for opt-in data collection."""

    model_config = SettingsConfigDict(env_prefix="TELEMETRY_")

    # Master switch (opt-in, default off)
    enabled: bool = Field(default=False, description="Enable telemetry collection (opt-in)")

    # Collection settings
    collect_prompts: bool = Field(default=True, description="Collect user prompts")
    collect_kg_data: bool = Field(default=True, description="Collect KG extraction data")
    collect_metrics: bool = Field(default=True, description="Collect performance metrics")

    # Privacy settings
    anonymize_data: bool = Field(default=True, description="Anonymize collected data")
    sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Sampling rate (0.0-1.0)"
    )

    # Endpoint settings
    endpoint_url: str = Field(
        default="https://telemetry.sapiens.team/v1/collect",
        description="Telemetry collection endpoint",
    )
    api_key: SecretStr | None = Field(default=None, description="Telemetry API key")

    # Batching
    batch_size: int = Field(default=50, description="Events per batch before sending")
    flush_interval_seconds: int = Field(default=300, description="Flush interval in seconds")


class CacheSettings(BaseSettings):
    """Caching configuration settings."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    # Enable/disable caching
    enabled: bool = Field(default=True, description="Enable caching")

    # Redis configuration (optional L2 cache)
    redis_url: str | None = Field(default=None, description="Redis URL for distributed cache")

    # Query result cache
    query_cache_size: int = Field(default=500, description="Max query results in L1 cache")
    query_cache_ttl: int = Field(default=1800, description="Query cache TTL in seconds")

    # Embedding cache
    embedding_cache_size: int = Field(default=10000, description="Max embeddings in L1 cache")
    embedding_cache_ttl: int = Field(default=86400, description="Embedding cache TTL in seconds")

    # Subgraph cache
    subgraph_cache_size: int = Field(default=200, description="Max subgraphs in L1 cache")
    subgraph_cache_ttl: int = Field(default=600, description="Subgraph cache TTL in seconds")

    # Memory limits
    l1_max_bytes: int = Field(default=100 * 1024 * 1024, description="Max L1 cache size in bytes")


class Settings(BaseSettings):
    """Main application settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(default="Ontology Reasoning System", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    # Sub-settings
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tog: ToGSettings = Field(default_factory=ToGSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    api: APISettings = Field(default_factory=APISettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    checkpointing: CheckpointingSettings = Field(default_factory=CheckpointingSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

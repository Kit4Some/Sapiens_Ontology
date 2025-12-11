"""
LLM Provider with Automatic Failover.

Implements multi-provider support with automatic failover chain:
OpenAI -> Anthropic -> Azure OpenAI -> Local LLM

Features:
- Health checking per provider
- Automatic failover on API errors
- Retry with exponential backoff
- Provider-specific error handling
- Metrics and logging
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import SecretStr

logger = structlog.get_logger(__name__)


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class ProviderStatus(str, Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderHealth:
    """Health status for a provider."""
    provider: ProviderType
    status: ProviderStatus
    last_check: datetime
    last_success: datetime | None = None
    last_error: str | None = None
    error_count: int = 0
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    provider_type: ProviderType
    api_key: str | None = None
    model: str | None = None
    temperature: float = 0.0
    seed: int = 42
    top_p: float = 1.0
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    # Azure-specific
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str = "2024-02-01"

    # Local LLM specific
    local_base_url: str = "http://localhost:11434"

    # Priority (lower = higher priority)
    priority: int = 100

    # Whether this provider is enabled
    enabled: bool = True


@dataclass
class ProviderMetrics:
    """Metrics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    failovers_triggered: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


class LLMProvider:
    """
    Single LLM provider wrapper with health checking and retry logic.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._llm: BaseChatModel | None = None
        self._health = ProviderHealth(
            provider=config.provider_type,
            status=ProviderStatus.UNKNOWN,
            last_check=datetime.utcnow(),
        )
        self._metrics = ProviderMetrics()
        self._initialized = False

    @property
    def provider_type(self) -> ProviderType:
        return self.config.provider_type

    @property
    def health(self) -> ProviderHealth:
        return self._health

    @property
    def metrics(self) -> ProviderMetrics:
        return self._metrics

    @property
    def is_available(self) -> bool:
        """Check if provider is available for use."""
        return (
            self.config.enabled
            and self._health.status != ProviderStatus.UNHEALTHY
            and self._health.error_count < 5  # Circuit breaker threshold
        )

    def _create_llm(self) -> BaseChatModel:
        """Create LangChain LLM instance based on provider type."""
        if self.config.provider_type == ProviderType.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.config.model or "gpt-4o-mini",
                temperature=self.config.temperature,
                seed=self.config.seed,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                api_key=SecretStr(self.config.api_key) if self.config.api_key else None,
                timeout=self.config.timeout,
                max_retries=0,  # We handle retries ourselves
            )

        elif self.config.provider_type == ProviderType.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.config.model or "claude-3-5-sonnet-20241022",
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                api_key=SecretStr(self.config.api_key) if self.config.api_key else None,
                timeout=self.config.timeout,
            )

        elif self.config.provider_type == ProviderType.AZURE_OPENAI:
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI(
                azure_deployment=self.config.azure_deployment or self.config.model,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.azure_api_version,
                api_key=SecretStr(self.config.api_key) if self.config.api_key else None,
                temperature=self.config.temperature,
                seed=self.config.seed,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=0,
            )

        elif self.config.provider_type == ProviderType.LOCAL:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.config.model or "llama3.2",
                base_url=self.config.local_base_url,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        else:
            raise ValueError(f"Unsupported provider type: {self.config.provider_type}")

    def initialize(self) -> None:
        """Initialize the LLM instance."""
        if self._initialized:
            return

        try:
            self._llm = self._create_llm()
            self._initialized = True
            logger.info(
                "LLM provider initialized",
                provider=self.config.provider_type.value,
                model=self.config.model,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize LLM provider",
                provider=self.config.provider_type.value,
                error=str(e),
            )
            self._health.status = ProviderStatus.UNHEALTHY
            self._health.last_error = str(e)
            raise

    async def health_check(self) -> ProviderHealth:
        """Perform health check on the provider."""
        if not self._initialized:
            try:
                self.initialize()
            except Exception:
                return self._health

        start_time = time.time()

        try:
            # Simple health check query
            test_message = HumanMessage(content="Reply with 'OK' only.")
            response = await self._llm.ainvoke([test_message])

            latency_ms = (time.time() - start_time) * 1000

            self._health = ProviderHealth(
                provider=self.config.provider_type,
                status=ProviderStatus.HEALTHY,
                last_check=datetime.utcnow(),
                last_success=datetime.utcnow(),
                error_count=0,
                latency_ms=latency_ms,
            )

            logger.debug(
                "Health check passed",
                provider=self.config.provider_type.value,
                latency_ms=latency_ms,
            )

        except Exception as e:
            self._health.error_count += 1
            self._health.last_check = datetime.utcnow()
            self._health.last_error = str(e)

            if self._health.error_count >= 3:
                self._health.status = ProviderStatus.UNHEALTHY
            else:
                self._health.status = ProviderStatus.DEGRADED

            logger.warning(
                "Health check failed",
                provider=self.config.provider_type.value,
                error=str(e),
                error_count=self._health.error_count,
            )

        return self._health

    def _categorize_error(self, error: Exception) -> tuple[str, bool]:
        """
        Categorize error and determine if failover is needed.

        Returns:
            Tuple of (error_category, should_failover)
        """
        error_str = str(error).lower()

        # Authentication errors - failover immediately
        if any(x in error_str for x in ["api key", "authentication", "unauthorized", "invalid_api_key"]):
            return "auth_error", True

        # Credit/billing errors - failover immediately
        if any(x in error_str for x in ["credit", "billing", "quota", "insufficient"]):
            return "billing_error", True

        # Rate limit errors - might retry, then failover
        if any(x in error_str for x in ["rate limit", "429", "too many requests"]):
            return "rate_limit", True

        # Model not found - failover
        if any(x in error_str for x in ["model not found", "does not exist", "invalid model"]):
            return "model_error", True

        # Timeout - retry first
        if any(x in error_str for x in ["timeout", "timed out"]):
            return "timeout", False

        # Server errors - retry first
        if any(x in error_str for x in ["500", "502", "503", "504", "server error"]):
            return "server_error", False

        # Connection errors - retry first
        if any(x in error_str for x in ["connection", "network", "unreachable"]):
            return "connection_error", False

        # Unknown error - retry first
        return "unknown", False

    async def invoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """
        Invoke the LLM with retry logic.

        Args:
            messages: Messages to send to the LLM
            **kwargs: Additional arguments for the LLM

        Returns:
            LLM response message

        Raises:
            Exception if all retries fail
        """
        if not self._initialized:
            self.initialize()

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            start_time = time.time()
            self._metrics.total_requests += 1

            try:
                response = await self._llm.ainvoke(messages, **kwargs)

                # Success
                latency_ms = (time.time() - start_time) * 1000
                self._metrics.successful_requests += 1
                self._metrics.total_latency_ms += latency_ms
                self._health.error_count = 0
                self._health.last_success = datetime.utcnow()
                self._health.status = ProviderStatus.HEALTHY

                return response

            except Exception as e:
                last_error = e
                self._metrics.failed_requests += 1
                self._health.error_count += 1
                self._health.last_error = str(e)

                error_category, should_failover = self._categorize_error(e)

                logger.warning(
                    "LLM invocation failed",
                    provider=self.config.provider_type.value,
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error_category=error_category,
                    should_failover=should_failover,
                    error=str(e),
                )

                # If should failover, don't retry
                if should_failover:
                    self._health.status = ProviderStatus.UNHEALTHY
                    raise

                # Retry with exponential backoff
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        # All retries failed
        self._health.status = ProviderStatus.DEGRADED
        raise last_error or Exception("All retries failed")

    def get_llm(self) -> BaseChatModel:
        """Get the underlying LLM instance."""
        if not self._initialized:
            self.initialize()
        return self._llm


class LLMProviderChain:
    """
    Chain of LLM providers with automatic failover.

    Attempts providers in priority order, failing over to the next
    provider when one fails.
    """

    def __init__(
        self,
        configs: list[ProviderConfig],
        health_check_interval: int = 300,  # 5 minutes
    ) -> None:
        """
        Initialize provider chain.

        Args:
            configs: List of provider configurations
            health_check_interval: Seconds between health checks
        """
        # Sort by priority and filter enabled
        sorted_configs = sorted(
            [c for c in configs if c.enabled],
            key=lambda x: x.priority
        )

        self._providers: list[LLMProvider] = [
            LLMProvider(config) for config in sorted_configs
        ]
        self._health_check_interval = health_check_interval
        self._last_health_check: datetime | None = None
        self._current_provider_idx = 0
        self._failover_count = 0

        if not self._providers:
            raise ValueError("At least one enabled provider is required")

        logger.info(
            "LLM provider chain initialized",
            providers=[p.provider_type.value for p in self._providers],
        )

    @property
    def providers(self) -> list[LLMProvider]:
        return self._providers

    @property
    def current_provider(self) -> LLMProvider:
        return self._providers[self._current_provider_idx]

    @property
    def failover_count(self) -> int:
        return self._failover_count

    def get_health_status(self) -> dict[str, Any]:
        """Get health status for all providers."""
        return {
            "providers": [p.health.to_dict() for p in self._providers],
            "current_provider": self.current_provider.provider_type.value,
            "failover_count": self._failover_count,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
        }

    async def health_check_all(self) -> list[ProviderHealth]:
        """Perform health check on all providers."""
        results = []

        for provider in self._providers:
            try:
                health = await provider.health_check()
                results.append(health)
            except Exception as e:
                logger.error(
                    "Health check failed",
                    provider=provider.provider_type.value,
                    error=str(e),
                )

        self._last_health_check = datetime.utcnow()

        # Reset to first healthy provider
        for idx, provider in enumerate(self._providers):
            if provider.is_available:
                if idx != self._current_provider_idx:
                    logger.info(
                        "Switching to healthy provider",
                        from_provider=self.current_provider.provider_type.value,
                        to_provider=provider.provider_type.value,
                    )
                    self._current_provider_idx = idx
                break

        return results

    def _get_next_available_provider(self, start_idx: int = 0) -> int | None:
        """Get index of next available provider."""
        for i in range(len(self._providers)):
            idx = (start_idx + i) % len(self._providers)
            if self._providers[idx].is_available:
                return idx
        return None

    async def invoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """
        Invoke LLM with automatic failover.

        Tries current provider first, then fails over to next available
        provider on failure.

        Args:
            messages: Messages to send to the LLM
            **kwargs: Additional arguments for the LLM

        Returns:
            LLM response message

        Raises:
            Exception if all providers fail
        """
        errors: list[tuple[ProviderType, str]] = []
        tried_providers: set[int] = set()

        # Start with current provider
        provider_idx = self._current_provider_idx

        while len(tried_providers) < len(self._providers):
            # Skip if already tried
            if provider_idx in tried_providers:
                next_idx = self._get_next_available_provider(provider_idx + 1)
                if next_idx is None or next_idx in tried_providers:
                    break
                provider_idx = next_idx
                continue

            tried_providers.add(provider_idx)
            provider = self._providers[provider_idx]

            if not provider.is_available:
                provider_idx = (provider_idx + 1) % len(self._providers)
                continue

            try:
                response = await provider.invoke(messages, **kwargs)

                # Update current provider on success
                if provider_idx != self._current_provider_idx:
                    self._current_provider_idx = provider_idx

                return response

            except Exception as e:
                errors.append((provider.provider_type, str(e)))

                logger.warning(
                    "Provider failed, attempting failover",
                    failed_provider=provider.provider_type.value,
                    error=str(e),
                )

                # Find next available provider
                next_idx = self._get_next_available_provider(provider_idx + 1)

                if next_idx is not None and next_idx not in tried_providers:
                    self._failover_count += 1
                    provider._metrics.failovers_triggered += 1

                    logger.info(
                        "Failing over to next provider",
                        from_provider=provider.provider_type.value,
                        to_provider=self._providers[next_idx].provider_type.value,
                        total_failovers=self._failover_count,
                    )

                    provider_idx = next_idx
                else:
                    break

        # All providers failed
        error_details = "; ".join([f"{p.value}: {e}" for p, e in errors])
        raise Exception(f"All LLM providers failed: {error_details}")

    def get_llm(self) -> BaseChatModel:
        """
        Get the current LLM instance.

        Note: This returns the underlying LLM without failover protection.
        Use invoke() for automatic failover.
        """
        return self.current_provider.get_llm()


# =============================================================================
# Factory Functions
# =============================================================================


def create_provider_configs_from_settings(
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_key: str | None = None,
    azure_deployment: str | None = None,
    openai_model: str = "gpt-4o-mini",
    anthropic_model: str = "claude-3-5-sonnet-20241022",
    local_model: str = "llama3.2",
    local_base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    seed: int = 42,
    top_p: float = 1.0,
) -> list[ProviderConfig]:
    """
    Create provider configurations from API keys.

    Priority:
    1. OpenAI (if key provided)
    2. Anthropic (if key provided)
    3. Azure OpenAI (if configured)
    4. Local LLM (always available as fallback)
    """
    configs: list[ProviderConfig] = []
    priority = 1

    if openai_api_key:
        configs.append(ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key=openai_api_key,
            model=openai_model,
            priority=priority,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
        ))
        priority += 1

    if anthropic_api_key:
        configs.append(ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key=anthropic_api_key,
            model=anthropic_model,
            priority=priority,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
        ))
        priority += 1

    if azure_endpoint and azure_api_key:
        configs.append(ProviderConfig(
            provider_type=ProviderType.AZURE_OPENAI,
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            priority=priority,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
        ))
        priority += 1

    # Local LLM as fallback (lowest priority)
    configs.append(ProviderConfig(
        provider_type=ProviderType.LOCAL,
        model=local_model,
        local_base_url=local_base_url,
        priority=999,  # Lowest priority
        enabled=True,  # Always enabled as fallback
        temperature=temperature,
        seed=seed,
        top_p=top_p,
    ))

    return configs


_llm_chain: LLMProviderChain | None = None


def get_llm_chain(
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_key: str | None = None,
    azure_deployment: str | None = None,
    force_reinit: bool = False,
) -> LLMProviderChain:
    """
    Get or create the global LLM provider chain.

    Args:
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        azure_endpoint: Azure OpenAI endpoint
        azure_api_key: Azure OpenAI API key
        azure_deployment: Azure OpenAI deployment name
        force_reinit: Force reinitialization

    Returns:
        LLMProviderChain instance
    """
    global _llm_chain

    if _llm_chain is not None and not force_reinit:
        return _llm_chain

    configs = create_provider_configs_from_settings(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_deployment=azure_deployment,
    )

    _llm_chain = LLMProviderChain(configs)
    return _llm_chain

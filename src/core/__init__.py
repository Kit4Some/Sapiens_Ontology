"""
Core Infrastructure Module.

Provides foundational patterns and utilities:
- Circuit Breaker for fault tolerance
- Common patterns and abstractions
"""

from src.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
    FailureType,
    CircuitMetrics,
    get_circuit_breaker,
    get_all_circuit_breakers_status,
    get_llm_circuit_breaker,
    get_neo4j_circuit_breaker,
    get_embedding_circuit_breaker,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitBreakerRegistry",
    "CircuitState",
    "FailureType",
    "CircuitMetrics",
    # Factory functions
    "get_circuit_breaker",
    "get_all_circuit_breakers_status",
    "get_llm_circuit_breaker",
    "get_neo4j_circuit_breaker",
    "get_embedding_circuit_breaker",
]

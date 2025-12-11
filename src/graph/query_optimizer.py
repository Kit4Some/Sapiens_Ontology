"""
Neo4j Query Optimizer.

Provides query optimization for large-scale graph operations:
- Query timeout management
- LIMIT clause injection
- Query complexity analysis
- Index usage hints
- Query plan caching
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class QueryAnalysis:
    """Analysis results for a Cypher query."""

    original_query: str
    complexity: QueryComplexity
    estimated_cost: float
    has_limit: bool
    has_where: bool
    pattern_count: int
    uses_index: bool
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class QueryConfig:
    """Configuration for query optimization."""

    # Default limits
    default_limit: int = 1000
    max_limit: int = 10000

    # Timeouts (milliseconds)
    default_timeout_ms: int = 30000  # 30 seconds
    max_timeout_ms: int = 120000  # 2 minutes

    # Complexity thresholds
    pattern_count_threshold: int = 3  # Warn if more patterns
    depth_threshold: int = 3  # Warn if deep traversals

    # Auto-optimization
    auto_inject_limit: bool = True
    warn_on_full_scan: bool = True


class QueryOptimizer:
    """
    Optimizes Cypher queries for performance and scalability.

    Features:
    - Query complexity analysis
    - Automatic LIMIT injection
    - Timeout configuration
    - Index usage detection
    - Query suggestions
    """

    # Patterns that indicate expensive operations
    EXPENSIVE_PATTERNS = [
        r"MATCH\s*\([^)]*\)\s*-\s*\[\s*\*",  # Variable-length paths
        r"MATCH\s*\([^)]*\)\s*-\s*\[\s*:\w+\s*\*",  # Type-specific var-length
        r"MATCH\s*\([^)]*\)\s*,\s*\([^)]*\)",  # Multiple patterns (Cartesian)
        r"OPTIONAL\s+MATCH",  # Optional matches
        r"shortestPath",  # Shortest path algorithm
        r"allShortestPaths",  # All shortest paths
    ]

    # Patterns that use indexes efficiently
    INDEX_PATTERNS = [
        r"WHERE\s+\w+\.id\s*=",  # ID lookup
        r"WHERE\s+\w+\.name\s*=",  # Name lookup
        r"CALL\s+db\.index",  # Explicit index call
        r"USING\s+INDEX",  # Index hint
    ]

    def __init__(self, config: QueryConfig | None = None):
        self.config = config or QueryConfig()
        self._query_cache: dict[str, QueryAnalysis] = {}

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a Cypher query for performance characteristics.

        Args:
            query: The Cypher query to analyze

        Returns:
            QueryAnalysis with complexity assessment and suggestions
        """
        # Check cache
        cache_key = query.strip()
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        # Normalize query
        normalized = " ".join(query.split())

        # Count patterns
        pattern_count = len(re.findall(r"MATCH\s*\(", normalized, re.IGNORECASE))

        # Check for LIMIT
        has_limit = bool(re.search(r"\bLIMIT\s+\d+", normalized, re.IGNORECASE))

        # Check for WHERE
        has_where = bool(re.search(r"\bWHERE\b", normalized, re.IGNORECASE))

        # Check for index usage
        uses_index = any(
            re.search(pattern, normalized, re.IGNORECASE)
            for pattern in self.INDEX_PATTERNS
        )

        # Count expensive operations
        expensive_count = sum(
            1 for pattern in self.EXPENSIVE_PATTERNS
            if re.search(pattern, normalized, re.IGNORECASE)
        )

        # Determine complexity
        complexity = self._calculate_complexity(
            pattern_count, expensive_count, has_limit, has_where, uses_index
        )

        # Estimate cost (relative score)
        estimated_cost = self._estimate_cost(
            pattern_count, expensive_count, has_limit, uses_index
        )

        # Generate warnings and suggestions
        warnings, suggestions = self._generate_recommendations(
            normalized, pattern_count, has_limit, has_where, uses_index, expensive_count
        )

        analysis = QueryAnalysis(
            original_query=query,
            complexity=complexity,
            estimated_cost=estimated_cost,
            has_limit=has_limit,
            has_where=has_where,
            pattern_count=pattern_count,
            uses_index=uses_index,
            warnings=warnings,
            suggestions=suggestions,
        )

        # Cache analysis
        self._query_cache[cache_key] = analysis

        return analysis

    def _calculate_complexity(
        self,
        pattern_count: int,
        expensive_count: int,
        has_limit: bool,
        has_where: bool,
        uses_index: bool,
    ) -> QueryComplexity:
        """Calculate query complexity level."""
        score = 0

        # Pattern complexity
        score += pattern_count * 2

        # Expensive operations
        score += expensive_count * 5

        # Mitigating factors
        if has_limit:
            score -= 2
        if has_where:
            score -= 1
        if uses_index:
            score -= 2

        if score <= 2:
            return QueryComplexity.LOW
        elif score <= 5:
            return QueryComplexity.MEDIUM
        elif score <= 10:
            return QueryComplexity.HIGH
        else:
            return QueryComplexity.VERY_HIGH

    def _estimate_cost(
        self,
        pattern_count: int,
        expensive_count: int,
        has_limit: bool,
        uses_index: bool,
    ) -> float:
        """Estimate relative query cost (0-100)."""
        base_cost = 10.0

        # Pattern cost
        base_cost += pattern_count * 10

        # Expensive operation cost
        base_cost += expensive_count * 20

        # Discount for LIMIT
        if has_limit:
            base_cost *= 0.5

        # Discount for index usage
        if uses_index:
            base_cost *= 0.7

        return min(100.0, base_cost)

    def _generate_recommendations(
        self,
        query: str,
        pattern_count: int,
        has_limit: bool,
        has_where: bool,
        uses_index: bool,
        expensive_count: int,
    ) -> tuple[list[str], list[str]]:
        """Generate warnings and optimization suggestions."""
        warnings = []
        suggestions = []

        # No LIMIT warning
        if not has_limit and "RETURN" in query.upper():
            if "count(" not in query.lower() and "sum(" not in query.lower():
                warnings.append("Query has no LIMIT clause - may return large result sets")
                suggestions.append("Add LIMIT clause to prevent unbounded results")

        # No WHERE warning
        if not has_where and pattern_count > 0:
            if "RETURN" in query.upper():
                warnings.append("Query has no WHERE clause - may perform full scan")
                suggestions.append("Add WHERE clause to filter results")

        # Variable-length path warning
        if re.search(r"\[\s*\*", query):
            warnings.append("Variable-length paths can be expensive")
            suggestions.append("Consider adding depth limits (e.g., [*1..3])")

        # Multiple patterns warning
        if pattern_count > self.config.pattern_count_threshold:
            warnings.append(f"Query has {pattern_count} MATCH patterns - potential Cartesian product")
            suggestions.append("Consider breaking into multiple queries or using WITH clauses")

        # No index usage suggestion
        if not uses_index and has_where:
            suggestions.append("Consider adding index hints (USING INDEX) for better performance")

        # Expensive operations
        if expensive_count > 0:
            warnings.append(f"Query contains {expensive_count} potentially expensive operations")

        return warnings, suggestions

    def optimize_query(
        self,
        query: str,
        max_results: int | None = None,
        timeout_ms: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Optimize a Cypher query by injecting limits and configuration.

        Args:
            query: The original Cypher query
            max_results: Maximum results to return (uses default if not specified)
            timeout_ms: Query timeout in milliseconds

        Returns:
            Tuple of (optimized_query, metadata)
        """
        analysis = self.analyze_query(query)
        optimized = query
        metadata = {
            "original_complexity": analysis.complexity.value,
            "estimated_cost": analysis.estimated_cost,
            "modifications": [],
        }

        # Inject LIMIT if needed and configured
        if self.config.auto_inject_limit and not analysis.has_limit:
            if "RETURN" in optimized.upper():
                # Don't add LIMIT to aggregation queries
                aggregations = ["count(", "sum(", "avg(", "min(", "max(", "collect("]
                has_aggregation = any(agg in optimized.lower() for agg in aggregations)

                if not has_aggregation:
                    limit = max_results or self.config.default_limit
                    limit = min(limit, self.config.max_limit)

                    # Find RETURN clause and add LIMIT after
                    # Handle ORDER BY if present
                    if "ORDER BY" in optimized.upper():
                        # Add LIMIT after ORDER BY
                        optimized = re.sub(
                            r"(ORDER\s+BY\s+[^;]+?)(?=\s*$|\s*;)",
                            rf"\1 LIMIT {limit}",
                            optimized,
                            flags=re.IGNORECASE,
                        )
                    else:
                        # Add LIMIT after RETURN clause
                        optimized = re.sub(
                            r"(RETURN\s+.+?)(?=\s*$|\s*;)",
                            rf"\1 LIMIT {limit}",
                            optimized,
                            flags=re.IGNORECASE,
                        )

                    metadata["modifications"].append(f"Added LIMIT {limit}")
                    metadata["limit_added"] = limit

        # Calculate timeout
        effective_timeout = timeout_ms or self.config.default_timeout_ms
        effective_timeout = min(effective_timeout, self.config.max_timeout_ms)
        metadata["timeout_ms"] = effective_timeout

        # Log if high complexity
        if analysis.complexity in (QueryComplexity.HIGH, QueryComplexity.VERY_HIGH):
            logger.warning(
                "High complexity query detected",
                complexity=analysis.complexity.value,
                estimated_cost=analysis.estimated_cost,
                warnings=analysis.warnings,
            )

        return optimized, metadata

    def get_query_hints(self, labels: list[str], properties: list[str]) -> list[str]:
        """
        Generate index hints for a query.

        Args:
            labels: Node labels being queried
            properties: Properties being filtered on

        Returns:
            List of USING INDEX hints
        """
        hints = []

        # Common indexed properties
        indexed_props = {"id", "name", "type", "source"}

        for label in labels:
            for prop in properties:
                if prop.lower() in indexed_props:
                    hints.append(f"USING INDEX {label.lower()}:{label}({prop})")

        return hints

    def suggest_pagination(
        self,
        query: str,
        page_size: int = 100,
    ) -> dict[str, str]:
        """
        Suggest pagination strategy for a query.

        Args:
            query: The original query
            page_size: Items per page

        Returns:
            Dict with paginated query templates
        """
        # Find what's being returned
        return_match = re.search(r"RETURN\s+(.+?)(?:\s+ORDER|\s+LIMIT|\s*$)", query, re.IGNORECASE)
        if not return_match:
            return {"error": "Could not identify RETURN clause"}

        return_items = return_match.group(1).strip()

        # Generate paginated versions
        paginated = {
            "count_query": re.sub(
                r"RETURN\s+.+",
                f"RETURN count({return_items.split(',')[0].strip()}) AS total",
                query,
                flags=re.IGNORECASE,
            ),
            "page_query": f"{query.rstrip(';').rstrip()} SKIP $skip LIMIT {page_size}",
            "page_size": page_size,
            "usage": "Execute count_query first, then page_query with skip=0, skip=page_size, etc.",
        }

        return paginated


class QueryBuilder:
    """
    Safe query builder with automatic optimization.

    Helps construct Cypher queries with proper limits and safeguards.
    """

    def __init__(self, optimizer: QueryOptimizer | None = None):
        self.optimizer = optimizer or QueryOptimizer()
        self._parts: list[str] = []
        self._params: dict[str, Any] = {}
        self._limit: int | None = None
        self._skip: int | None = None

    def match(self, pattern: str) -> "QueryBuilder":
        """Add MATCH clause."""
        self._parts.append(f"MATCH {pattern}")
        return self

    def optional_match(self, pattern: str) -> "QueryBuilder":
        """Add OPTIONAL MATCH clause."""
        self._parts.append(f"OPTIONAL MATCH {pattern}")
        return self

    def where(self, condition: str) -> "QueryBuilder":
        """Add WHERE clause."""
        self._parts.append(f"WHERE {condition}")
        return self

    def and_where(self, condition: str) -> "QueryBuilder":
        """Add AND condition to WHERE."""
        self._parts.append(f"AND {condition}")
        return self

    def with_clause(self, items: str) -> "QueryBuilder":
        """Add WITH clause."""
        self._parts.append(f"WITH {items}")
        return self

    def return_clause(self, items: str) -> "QueryBuilder":
        """Add RETURN clause."""
        self._parts.append(f"RETURN {items}")
        return self

    def order_by(self, items: str) -> "QueryBuilder":
        """Add ORDER BY clause."""
        self._parts.append(f"ORDER BY {items}")
        return self

    def limit(self, count: int) -> "QueryBuilder":
        """Set LIMIT."""
        self._limit = count
        return self

    def skip(self, count: int) -> "QueryBuilder":
        """Set SKIP."""
        self._skip = count
        return self

    def param(self, name: str, value: Any) -> "QueryBuilder":
        """Add a parameter."""
        self._params[name] = value
        return self

    def build(self) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """
        Build the final query with optimization.

        Returns:
            Tuple of (query, parameters, metadata)
        """
        query = "\n".join(self._parts)

        # Add SKIP and LIMIT
        if self._skip is not None:
            query += f"\nSKIP {self._skip}"
        if self._limit is not None:
            query += f"\nLIMIT {self._limit}"

        # Optimize
        optimized, metadata = self.optimizer.optimize_query(query)

        return optimized, self._params, metadata

    def reset(self) -> "QueryBuilder":
        """Reset the builder."""
        self._parts = []
        self._params = {}
        self._limit = None
        self._skip = None
        return self


# =============================================================================
# Optimized Query Templates
# =============================================================================


class OptimizedQueries:
    """
    Collection of optimized query templates for common operations.
    """

    @staticmethod
    def entity_by_id(entity_id: str, limit: int = 1) -> tuple[str, dict[str, Any]]:
        """Get entity by ID with index hint."""
        return (
            """
            MATCH (e:Entity {id: $entity_id})
            RETURN e
            LIMIT $limit
            """,
            {"entity_id": entity_id, "limit": limit},
        )

    @staticmethod
    def entity_neighbors(entity_id: str, hops: int = 1, limit: int = 100) -> tuple[str, dict[str, Any]]:
        """Get entity neighborhood with depth limit."""
        return (
            f"""
            MATCH (e:Entity {{id: $entity_id}})
            CALL {{
                WITH e
                MATCH path = (e)-[*1..{hops}]-(neighbor)
                WHERE neighbor:Entity OR neighbor:Chunk
                RETURN neighbor, relationships(path) AS rels
                LIMIT $limit
            }}
            RETURN neighbor, rels
            """,
            {"entity_id": entity_id, "limit": limit},
        )

    @staticmethod
    def vector_search(
        embedding: list[float],
        label: str = "Entity",
        top_k: int = 10,
        min_score: float = 0.7,
    ) -> tuple[str, dict[str, Any]]:
        """Vector similarity search with limit."""
        index_name = f"{label.lower()}_embedding"
        return (
            f"""
            CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding)
            YIELD node, score
            WHERE score >= $min_score
            RETURN node, score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            {"embedding": embedding, "top_k": top_k, "min_score": min_score},
        )

    @staticmethod
    def fulltext_search(
        query: str,
        label: str = "Entity",
        limit: int = 20,
    ) -> tuple[str, dict[str, Any]]:
        """Full-text search with limit."""
        index_name = f"{label.lower()}_fulltext"
        return (
            f"""
            CALL db.index.fulltext.queryNodes('{index_name}', $query)
            YIELD node, score
            WHERE score > 0
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query": query, "limit": limit},
        )

    @staticmethod
    def multi_hop_path(
        start_id: str,
        end_id: str,
        max_hops: int = 5,
        limit: int = 10,
    ) -> tuple[str, dict[str, Any]]:
        """Find paths between entities with depth limit."""
        return (
            f"""
            MATCH path = shortestPath(
                (start:Entity {{id: $start_id}})-[*1..{max_hops}]-(end:Entity {{id: $end_id}})
            )
            RETURN path, length(path) AS hops
            ORDER BY hops
            LIMIT $limit
            """,
            {"start_id": start_id, "end_id": end_id, "limit": limit},
        )

    @staticmethod
    def community_entities(
        community_id: str,
        limit: int = 100,
    ) -> tuple[str, dict[str, Any]]:
        """Get entities in a community with limit."""
        return (
            """
            MATCH (c:Community {id: $community_id})<-[:BELONGS_TO]-(e:Entity)
            RETURN e
            LIMIT $limit
            """,
            {"community_id": community_id, "limit": limit},
        )

    @staticmethod
    def paginated_entities(
        skip: int = 0,
        limit: int = 100,
        label: str | None = None,
        order_by: str = "name",
    ) -> tuple[str, dict[str, Any]]:
        """Paginated entity retrieval."""
        label_filter = f":{label}" if label else ""
        return (
            f"""
            MATCH (e:Entity{label_filter})
            RETURN e
            ORDER BY e.{order_by}
            SKIP $skip
            LIMIT $limit
            """,
            {"skip": skip, "limit": limit},
        )


# =============================================================================
# Global Optimizer Instance
# =============================================================================

_optimizer: QueryOptimizer | None = None


def get_query_optimizer() -> QueryOptimizer:
    """Get the global query optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer


def init_query_optimizer(config: QueryConfig | None = None) -> QueryOptimizer:
    """Initialize the global query optimizer with custom config."""
    global _optimizer
    _optimizer = QueryOptimizer(config)
    return _optimizer

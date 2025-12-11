"""
Graph Integrity Checker.

Validates referential integrity in the Neo4j graph:
- Orphan nodes (no relationships)
- Dangling references (relationships to non-existent nodes)
- Schema violations
- Data consistency issues
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client

logger = structlog.get_logger(__name__)


class IssueType(str, Enum):
    """Types of integrity issues."""

    ORPHAN_ENTITY = "orphan_entity"           # Entity with no relationships
    ORPHAN_CHUNK = "orphan_chunk"             # Chunk with no entities
    DANGLING_RELATION = "dangling_relation"   # Relation to non-existent entity
    MISSING_REQUIRED = "missing_required"     # Missing required property
    INVALID_TYPE = "invalid_type"             # Invalid entity type
    DUPLICATE_ID = "duplicate_id"             # Duplicate ID (shouldn't happen with constraints)
    CIRCULAR_REFERENCE = "circular_reference" # Self-referencing relationship
    INCONSISTENT_STATE = "inconsistent_state" # Deleted node with active relations


class IssueSeverity(str, Enum):
    """Severity levels for issues."""

    ERROR = "error"       # Must be fixed
    WARNING = "warning"   # Should be investigated
    INFO = "info"         # For information only


@dataclass
class IntegrityIssue:
    """Represents an integrity issue found in the graph."""

    issue_type: IssueType
    severity: IssueSeverity
    description: str
    node_id: str | None = None
    node_label: str | None = None
    relationship_type: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "node_id": self.node_id,
            "node_label": self.node_label,
            "relationship_type": self.relationship_type,
            "details": self.details,
        }


@dataclass
class IntegrityReport:
    """Report of integrity check results."""

    checked_at: datetime = field(default_factory=datetime.utcnow)
    is_healthy: bool = True

    # Counts
    total_nodes: int = 0
    total_relationships: int = 0
    issues_found: int = 0

    # Issues by type
    errors: list[IntegrityIssue] = field(default_factory=list)
    warnings: list[IntegrityIssue] = field(default_factory=list)
    info: list[IntegrityIssue] = field(default_factory=list)

    # Statistics
    orphan_entities: int = 0
    orphan_chunks: int = 0
    dangling_relations: int = 0
    inconsistent_states: int = 0

    # Duration
    duration_seconds: float = 0.0

    def add_issue(self, issue: IntegrityIssue) -> None:
        """Add an issue to the report."""
        if issue.severity == IssueSeverity.ERROR:
            self.errors.append(issue)
            self.is_healthy = False
        elif issue.severity == IssueSeverity.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)

        self.issues_found += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "is_healthy": self.is_healthy,
            "summary": {
                "total_nodes": self.total_nodes,
                "total_relationships": self.total_relationships,
                "issues_found": self.issues_found,
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.info),
            },
            "statistics": {
                "orphan_entities": self.orphan_entities,
                "orphan_chunks": self.orphan_chunks,
                "dangling_relations": self.dangling_relations,
                "inconsistent_states": self.inconsistent_states,
            },
            "duration_seconds": round(self.duration_seconds, 2),
            "issues": {
                "errors": [e.to_dict() for e in self.errors],
                "warnings": [w.to_dict() for w in self.warnings],
                "info": [i.to_dict() for i in self.info],
            },
        }


class IntegrityChecker:
    """
    Checks referential integrity of the Neo4j graph.

    Performs comprehensive checks for:
    - Orphan nodes (entities without relationships, chunks without entities)
    - Dangling references (relationships pointing to deleted/missing nodes)
    - Schema violations (missing required properties, invalid types)
    - State inconsistencies (deleted nodes with active relationships)

    Usage:
        ```python
        checker = IntegrityChecker()

        # Full integrity check
        report = await checker.check_all()

        if not report.is_healthy:
            print(f"Found {len(report.errors)} errors")
            for error in report.errors:
                print(f"  - {error.description}")

        # Specific checks
        orphans = await checker.find_orphan_entities()
        dangling = await checker.find_dangling_relations()
        ```
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
    ) -> None:
        self._client = client or get_ontology_client()

    async def check_all(self) -> IntegrityReport:
        """
        Perform comprehensive integrity check.

        Returns:
            IntegrityReport with all issues found
        """
        start_time = datetime.utcnow()
        report = IntegrityReport()

        await self._client.connect()

        logger.info("Starting comprehensive integrity check")

        # Get counts
        report.total_nodes, report.total_relationships = await self._get_counts()

        # Run all checks
        await self._check_orphan_entities(report)
        await self._check_orphan_chunks(report)
        await self._check_dangling_relations(report)
        await self._check_inconsistent_states(report)
        await self._check_required_properties(report)
        await self._check_circular_references(report)

        report.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            "Integrity check completed",
            is_healthy=report.is_healthy,
            errors=len(report.errors),
            warnings=len(report.warnings),
            duration_s=round(report.duration_seconds, 2),
        )

        return report

    async def _get_counts(self) -> tuple[int, int]:
        """Get total node and relationship counts."""
        node_query = "MATCH (n) RETURN count(n) as count"
        rel_query = "MATCH ()-[r]->() RETURN count(r) as count"

        node_result = await self._client.execute_cypher(node_query)
        rel_result = await self._client.execute_cypher(rel_query)

        return (
            node_result[0]["count"] if node_result else 0,
            rel_result[0]["count"] if rel_result else 0,
        )

    async def _check_orphan_entities(self, report: IntegrityReport) -> None:
        """Find entities with no relationships."""
        query = """
        MATCH (e:Entity)
        WHERE NOT (e)--()
          AND coalesce(e.is_deleted, false) = false
        RETURN e.id as id, e.name as name, e.type as type
        LIMIT 100
        """

        results = await self._client.execute_cypher(query)
        report.orphan_entities = len(results)

        for row in results:
            report.add_issue(IntegrityIssue(
                issue_type=IssueType.ORPHAN_ENTITY,
                severity=IssueSeverity.WARNING,
                description=f"Entity '{row['name']}' has no relationships",
                node_id=row["id"],
                node_label="Entity",
                details={"type": row["type"]},
            ))

    async def _check_orphan_chunks(self, report: IntegrityReport) -> None:
        """Find chunks that don't contain any entities."""
        query = """
        MATCH (c:Chunk)
        WHERE NOT (c)-[:CONTAINS]->(:Entity)
          AND coalesce(c.is_deleted, false) = false
        RETURN c.id as id, c.source as source, substring(c.text, 0, 50) as preview
        LIMIT 100
        """

        results = await self._client.execute_cypher(query)
        report.orphan_chunks = len(results)

        for row in results:
            report.add_issue(IntegrityIssue(
                issue_type=IssueType.ORPHAN_CHUNK,
                severity=IssueSeverity.INFO,
                description=f"Chunk from '{row['source']}' contains no entities",
                node_id=row["id"],
                node_label="Chunk",
                details={"preview": row["preview"]},
            ))

    async def _check_dangling_relations(self, report: IntegrityReport) -> None:
        """Find relationships where source or target is missing/deleted."""
        # Check for RELATES_TO with deleted endpoints
        query = """
        MATCH (a)-[r:RELATES_TO]->(b)
        WHERE (coalesce(a.is_deleted, false) = true OR coalesce(b.is_deleted, false) = true)
          AND coalesce(r.is_deleted, false) = false
        RETURN a.id as source_id, b.id as target_id, r.predicate as predicate,
               coalesce(a.is_deleted, false) as source_deleted,
               coalesce(b.is_deleted, false) as target_deleted
        LIMIT 100
        """

        results = await self._client.execute_cypher(query)
        report.dangling_relations = len(results)

        for row in results:
            deleted_side = "source" if row["source_deleted"] else "target"
            report.add_issue(IntegrityIssue(
                issue_type=IssueType.DANGLING_RELATION,
                severity=IssueSeverity.ERROR,
                description=f"Relationship {row['predicate']} has deleted {deleted_side}",
                node_id=row["source_id"],
                relationship_type="RELATES_TO",
                details={
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "predicate": row["predicate"],
                },
            ))

    async def _check_inconsistent_states(self, report: IntegrityReport) -> None:
        """Find deleted nodes that still have active relationships."""
        query = """
        MATCH (n)
        WHERE coalesce(n.is_deleted, false) = true
        WITH n, size([r IN [(n)-[r]-() | r] WHERE coalesce(r.is_deleted, false) = false]) as active_rels
        WHERE active_rels > 0
        RETURN labels(n)[0] as label, n.id as id, n.name as name, active_rels
        LIMIT 100
        """

        results = await self._client.execute_cypher(query)
        report.inconsistent_states = len(results)

        for row in results:
            report.add_issue(IntegrityIssue(
                issue_type=IssueType.INCONSISTENT_STATE,
                severity=IssueSeverity.ERROR,
                description=f"Deleted {row['label']} '{row['name']}' has {row['active_rels']} active relationships",
                node_id=row["id"],
                node_label=row["label"],
                details={"active_relationships": row["active_rels"]},
            ))

    async def _check_required_properties(self, report: IntegrityReport) -> None:
        """Check for missing required properties."""
        # Entities must have id, name, type
        query = """
        MATCH (e:Entity)
        WHERE e.id IS NULL OR e.name IS NULL OR e.type IS NULL
        RETURN e.id as id, e.name as name, e.type as type,
               CASE WHEN e.id IS NULL THEN 'id' ELSE '' END +
               CASE WHEN e.name IS NULL THEN ',name' ELSE '' END +
               CASE WHEN e.type IS NULL THEN ',type' ELSE '' END as missing
        LIMIT 50
        """

        results = await self._client.execute_cypher(query)

        for row in results:
            report.add_issue(IntegrityIssue(
                issue_type=IssueType.MISSING_REQUIRED,
                severity=IssueSeverity.ERROR,
                description=f"Entity missing required properties: {row['missing'].strip(',')}",
                node_id=row["id"],
                node_label="Entity",
                details={"missing_properties": row["missing"].strip(",").split(",")},
            ))

    async def _check_circular_references(self, report: IntegrityReport) -> None:
        """Find self-referencing relationships."""
        query = """
        MATCH (n)-[r]->(n)
        RETURN labels(n)[0] as label, n.id as id, n.name as name, type(r) as rel_type
        LIMIT 50
        """

        results = await self._client.execute_cypher(query)

        for row in results:
            report.add_issue(IntegrityIssue(
                issue_type=IssueType.CIRCULAR_REFERENCE,
                severity=IssueSeverity.WARNING,
                description=f"{row['label']} '{row['name']}' has self-referencing {row['rel_type']}",
                node_id=row["id"],
                node_label=row["label"],
                relationship_type=row["rel_type"],
            ))

    async def find_orphan_entities(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Find entities with no relationships.

        Args:
            limit: Maximum results

        Returns:
            List of orphan entity details
        """
        await self._client.connect()

        query = """
        MATCH (e:Entity)
        WHERE NOT (e)--()
          AND coalesce(e.is_deleted, false) = false
        RETURN e.id as id, e.name as name, e.type as type,
               e.created_at as created_at
        ORDER BY e.created_at DESC
        LIMIT $limit
        """

        return await self._client.execute_cypher(query, {"limit": limit})

    async def find_dangling_relations(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Find relationships with missing/deleted endpoints.

        Args:
            limit: Maximum results

        Returns:
            List of dangling relationship details
        """
        await self._client.connect()

        query = """
        MATCH (a)-[r:RELATES_TO]->(b)
        WHERE (coalesce(a.is_deleted, false) = true OR coalesce(b.is_deleted, false) = true)
          AND coalesce(r.is_deleted, false) = false
        RETURN a.id as source_id, a.name as source_name,
               b.id as target_id, b.name as target_name,
               r.predicate as predicate, type(r) as rel_type
        LIMIT $limit
        """

        return await self._client.execute_cypher(query, {"limit": limit})

    async def get_integrity_stats(self) -> dict[str, Any]:
        """
        Get quick integrity statistics.

        Returns:
            Statistics dictionary
        """
        await self._client.connect()

        query = """
        CALL {
            // Total counts
            MATCH (n) RETURN 'total_nodes' as metric, count(n) as value
            UNION ALL
            MATCH ()-[r]->() RETURN 'total_relationships' as metric, count(r) as value
            UNION ALL
            // Orphan entities
            MATCH (e:Entity)
            WHERE NOT (e)--() AND coalesce(e.is_deleted, false) = false
            RETURN 'orphan_entities' as metric, count(e) as value
            UNION ALL
            // Deleted nodes
            MATCH (n) WHERE coalesce(n.is_deleted, false) = true
            RETURN 'deleted_nodes' as metric, count(n) as value
            UNION ALL
            // Active entities
            MATCH (e:Entity) WHERE coalesce(e.is_deleted, false) = false
            RETURN 'active_entities' as metric, count(e) as value
        }
        RETURN metric, value
        """

        results = await self._client.execute_cypher(query)
        return {r["metric"]: r["value"] for r in results}


# Factory function
def create_integrity_checker() -> IntegrityChecker:
    """Create an integrity checker."""
    return IntegrityChecker()

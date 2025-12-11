"""
Graph Integrity Repair.

Provides repair operations for integrity issues:
- Remove orphan nodes
- Fix dangling references
- Cascade soft-delete
- Clean inconsistent states
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client
from src.graph.integrity.integrity_checker import IntegrityChecker, IntegrityReport

logger = structlog.get_logger(__name__)


class RepairAction(str, Enum):
    """Types of repair actions."""

    DELETE_ORPHAN = "delete_orphan"
    SOFT_DELETE_ORPHAN = "soft_delete_orphan"
    CASCADE_DELETE = "cascade_delete"
    REMOVE_DANGLING_REL = "remove_dangling_rel"
    SOFT_DELETE_DANGLING = "soft_delete_dangling"
    FIX_INCONSISTENT = "fix_inconsistent"


@dataclass
class RepairResult:
    """Result of a repair operation."""

    success: bool
    action: RepairAction
    items_repaired: int = 0
    items_skipped: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action.value,
            "items_repaired": self.items_repaired,
            "items_skipped": self.items_skipped,
            "details": self.details,
            "errors": self.errors,
        }


class IntegrityRepair:
    """
    Repairs integrity issues in the Neo4j graph.

    Provides both automatic and manual repair operations.

    Usage:
        ```python
        repair = IntegrityRepair()

        # Auto-repair all issues (with dry-run first)
        preview = await repair.auto_repair(dry_run=True)
        print(f"Would repair {preview.items_repaired} items")

        # Actually repair
        result = await repair.auto_repair(dry_run=False)

        # Specific repairs
        await repair.remove_orphan_entities(soft_delete=True)
        await repair.fix_dangling_relations()
        await repair.cascade_soft_delete(node_id="...", label="Entity")
        ```
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
    ) -> None:
        self._client = client or get_ontology_client()
        self._checker = IntegrityChecker(client=self._client)

    async def auto_repair(
        self,
        dry_run: bool = True,
        soft_delete: bool = True,
    ) -> dict[str, RepairResult]:
        """
        Automatically repair all detected integrity issues.

        Args:
            dry_run: Preview changes without applying
            soft_delete: Use soft-delete instead of hard delete

        Returns:
            Dictionary of repair results by action type
        """
        results: dict[str, RepairResult] = {}

        await self._client.connect()

        logger.info(
            "Starting auto-repair",
            dry_run=dry_run,
            soft_delete=soft_delete,
        )

        # Get current integrity report
        report = await self._checker.check_all()

        if report.is_healthy:
            logger.info("No integrity issues found")
            return results

        # Repair dangling relations first (most critical)
        if report.dangling_relations > 0:
            results["dangling_relations"] = await self.fix_dangling_relations(
                soft_delete=soft_delete,
                dry_run=dry_run,
            )

        # Fix inconsistent states
        if report.inconsistent_states > 0:
            results["inconsistent_states"] = await self.fix_inconsistent_states(
                dry_run=dry_run,
            )

        # Handle orphan entities (optional - may be intentional)
        if report.orphan_entities > 0:
            results["orphan_entities"] = await self.remove_orphan_entities(
                soft_delete=soft_delete,
                dry_run=dry_run,
            )

        logger.info(
            "Auto-repair completed",
            repairs=len(results),
            dry_run=dry_run,
        )

        return results

    async def remove_orphan_entities(
        self,
        soft_delete: bool = True,
        dry_run: bool = False,
        limit: int = 1000,
    ) -> RepairResult:
        """
        Remove or soft-delete orphan entities.

        Args:
            soft_delete: Use soft-delete instead of hard delete
            dry_run: Preview without applying
            limit: Maximum entities to process

        Returns:
            RepairResult with details
        """
        action = RepairAction.SOFT_DELETE_ORPHAN if soft_delete else RepairAction.DELETE_ORPHAN
        result = RepairResult(success=False, action=action)

        await self._client.connect()

        # Find orphans
        find_query = """
        MATCH (e:Entity)
        WHERE NOT (e)--()
          AND coalesce(e.is_deleted, false) = false
        RETURN e.id as id, e.name as name
        LIMIT $limit
        """

        orphans = await self._client.execute_cypher(find_query, {"limit": limit})

        if not orphans:
            result.success = True
            return result

        result.details = orphans

        if dry_run:
            result.items_repaired = len(orphans)
            result.success = True
            logger.info(f"[DRY RUN] Would {'soft-delete' if soft_delete else 'delete'} {len(orphans)} orphan entities")
            return result

        # Apply repair
        if soft_delete:
            repair_query = """
            MATCH (e:Entity)
            WHERE NOT (e)--()
              AND coalesce(e.is_deleted, false) = false
            WITH e LIMIT $limit
            SET e.is_deleted = true,
                e.deleted_at = datetime(),
                e.deletion_reason = 'auto_repair:orphan_entity'
            RETURN count(e) as repaired
            """
        else:
            repair_query = """
            MATCH (e:Entity)
            WHERE NOT (e)--()
              AND coalesce(e.is_deleted, false) = false
            WITH e LIMIT $limit
            DELETE e
            RETURN count(*) as repaired
            """

        try:
            repair_result = await self._client.execute_cypher(repair_query, {"limit": limit})
            result.items_repaired = repair_result[0].get("repaired", 0) if repair_result else 0
            result.success = True

            logger.info(
                "Orphan entities repaired",
                count=result.items_repaired,
                action="soft_delete" if soft_delete else "delete",
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.error("Failed to repair orphan entities", error=str(e))

        return result

    async def fix_dangling_relations(
        self,
        soft_delete: bool = True,
        dry_run: bool = False,
    ) -> RepairResult:
        """
        Fix relationships with deleted endpoints.

        Args:
            soft_delete: Soft-delete relations instead of removing
            dry_run: Preview without applying

        Returns:
            RepairResult with details
        """
        action = RepairAction.SOFT_DELETE_DANGLING if soft_delete else RepairAction.REMOVE_DANGLING_REL
        result = RepairResult(success=False, action=action)

        await self._client.connect()

        # Find dangling relations
        find_query = """
        MATCH (a)-[r:RELATES_TO]->(b)
        WHERE (coalesce(a.is_deleted, false) = true OR coalesce(b.is_deleted, false) = true)
          AND coalesce(r.is_deleted, false) = false
        RETURN a.id as source_id, b.id as target_id, r.predicate as predicate
        LIMIT 1000
        """

        dangling = await self._client.execute_cypher(find_query)

        if not dangling:
            result.success = True
            return result

        result.details = dangling

        if dry_run:
            result.items_repaired = len(dangling)
            result.success = True
            logger.info(f"[DRY RUN] Would fix {len(dangling)} dangling relations")
            return result

        # Apply repair
        if soft_delete:
            repair_query = """
            MATCH (a)-[r:RELATES_TO]->(b)
            WHERE (coalesce(a.is_deleted, false) = true OR coalesce(b.is_deleted, false) = true)
              AND coalesce(r.is_deleted, false) = false
            SET r.is_deleted = true,
                r.deleted_at = datetime()
            RETURN count(r) as repaired
            """
        else:
            repair_query = """
            MATCH (a)-[r:RELATES_TO]->(b)
            WHERE (coalesce(a.is_deleted, false) = true OR coalesce(b.is_deleted, false) = true)
              AND coalesce(r.is_deleted, false) = false
            DELETE r
            RETURN count(*) as repaired
            """

        try:
            repair_result = await self._client.execute_cypher(repair_query)
            result.items_repaired = repair_result[0].get("repaired", 0) if repair_result else 0
            result.success = True

            logger.info(
                "Dangling relations repaired",
                count=result.items_repaired,
                action="soft_delete" if soft_delete else "delete",
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.error("Failed to fix dangling relations", error=str(e))

        return result

    async def fix_inconsistent_states(
        self,
        dry_run: bool = False,
    ) -> RepairResult:
        """
        Fix deleted nodes that have active relationships.

        Soft-deletes the relationships of deleted nodes.

        Args:
            dry_run: Preview without applying

        Returns:
            RepairResult with details
        """
        result = RepairResult(success=False, action=RepairAction.FIX_INCONSISTENT)

        await self._client.connect()

        # Find inconsistent nodes
        find_query = """
        MATCH (n)-[r]-()
        WHERE coalesce(n.is_deleted, false) = true
          AND coalesce(r.is_deleted, false) = false
        RETURN DISTINCT labels(n)[0] as label, n.id as id, count(r) as active_rels
        LIMIT 1000
        """

        inconsistent = await self._client.execute_cypher(find_query)

        if not inconsistent:
            result.success = True
            return result

        result.details = inconsistent

        if dry_run:
            result.items_repaired = len(inconsistent)
            result.success = True
            logger.info(f"[DRY RUN] Would fix {len(inconsistent)} inconsistent nodes")
            return result

        # Soft-delete relationships of deleted nodes
        repair_query = """
        MATCH (n)-[r]-()
        WHERE coalesce(n.is_deleted, false) = true
          AND coalesce(r.is_deleted, false) = false
        SET r.is_deleted = true,
            r.deleted_at = datetime()
        RETURN count(r) as repaired
        """

        try:
            repair_result = await self._client.execute_cypher(repair_query)
            result.items_repaired = repair_result[0].get("repaired", 0) if repair_result else 0
            result.success = True

            logger.info(
                "Inconsistent states fixed",
                relations_soft_deleted=result.items_repaired,
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.error("Failed to fix inconsistent states", error=str(e))

        return result

    async def cascade_soft_delete(
        self,
        node_id: str,
        label: str = "Entity",
        max_depth: int = 2,
        dry_run: bool = False,
    ) -> RepairResult:
        """
        Cascade soft-delete to related nodes.

        Args:
            node_id: Starting node ID
            label: Node label
            max_depth: Maximum relationship depth to cascade
            dry_run: Preview without applying

        Returns:
            RepairResult with details
        """
        result = RepairResult(success=False, action=RepairAction.CASCADE_DELETE)

        await self._client.connect()

        # Find related nodes to cascade
        find_query = f"""
        MATCH (start:{label} {{id: $node_id}})
        MATCH path = (start)-[*1..{max_depth}]-(related)
        WHERE coalesce(related.is_deleted, false) = false
        RETURN DISTINCT labels(related)[0] as label, related.id as id, related.name as name,
               length(path) as depth
        ORDER BY depth
        LIMIT 100
        """

        related = await self._client.execute_cypher(find_query, {"node_id": node_id})

        if not related:
            result.success = True
            logger.info("No related nodes to cascade delete")
            return result

        result.details = related

        if dry_run:
            result.items_repaired = len(related)
            result.success = True
            logger.info(f"[DRY RUN] Would cascade soft-delete {len(related)} related nodes")
            return result

        # Apply cascade soft-delete
        cascade_query = f"""
        MATCH (start:{label} {{id: $node_id}})
        MATCH path = (start)-[*1..{max_depth}]-(related)
        WHERE coalesce(related.is_deleted, false) = false
        SET related.is_deleted = true,
            related.deleted_at = datetime(),
            related.deletion_reason = 'cascade_from:' + $node_id
        WITH related
        MATCH (related)-[r]-()
        SET r.is_deleted = true,
            r.deleted_at = datetime()
        RETURN count(DISTINCT related) as nodes_deleted
        """

        try:
            cascade_result = await self._client.execute_cypher(cascade_query, {"node_id": node_id})
            result.items_repaired = cascade_result[0].get("nodes_deleted", 0) if cascade_result else 0
            result.success = True

            logger.info(
                "Cascade soft-delete completed",
                source=node_id,
                affected=result.items_repaired,
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.error("Cascade soft-delete failed", node_id=node_id, error=str(e))

        return result


# Factory function
def create_integrity_repair() -> IntegrityRepair:
    """Create an integrity repair instance."""
    return IntegrityRepair()

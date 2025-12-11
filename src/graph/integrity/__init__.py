"""
Graph Integrity Management.

Provides referential integrity checking and repair:
- Orphan node detection
- Dangling reference detection
- Cascade operations
- Integrity repair utilities
"""

from src.graph.integrity.integrity_checker import (
    IntegrityChecker,
    IntegrityReport,
    IntegrityIssue,
    IssueType,
    IssueSeverity,
)
from src.graph.integrity.integrity_repair import (
    IntegrityRepair,
    RepairResult,
    RepairAction,
)

__all__ = [
    # Checker
    "IntegrityChecker",
    "IntegrityReport",
    "IntegrityIssue",
    "IssueType",
    "IssueSeverity",
    # Repair
    "IntegrityRepair",
    "RepairResult",
    "RepairAction",
]

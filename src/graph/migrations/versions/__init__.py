"""
Migration Versions.

Contains all database migrations in order.
Import and register with MigrationManager.
"""

from src.graph.migrations.versions.m001_initial_schema import Migration001InitialSchema
from src.graph.migrations.versions.m002_soft_delete import Migration002SoftDelete
from src.graph.migrations.versions.m003_timestamps import Migration003Timestamps
from src.graph.migrations.versions.m004_lifecycle_indexes import Migration004LifecycleIndexes

# All migrations in order
ALL_MIGRATIONS = [
    Migration001InitialSchema(),
    Migration002SoftDelete(),
    Migration003Timestamps(),
    Migration004LifecycleIndexes(),
]

__all__ = [
    "Migration001InitialSchema",
    "Migration002SoftDelete",
    "Migration003Timestamps",
    "Migration004LifecycleIndexes",
    "ALL_MIGRATIONS",
]

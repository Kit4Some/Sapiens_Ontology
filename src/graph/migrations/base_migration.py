"""
Base Migration Class.

Defines the interface for database migrations.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BaseMigration(ABC):
    """
    Abstract base class for database migrations.

    Each migration should:
    1. Have a unique version identifier
    2. Implement up() for applying changes
    3. Implement down() for reverting changes
    4. Be idempotent where possible

    Example:
        ```python
        class Migration001AddTimestamps(BaseMigration):
            version = "001"
            description = "Add timestamp fields to Entity nodes"

            async def up(self, client) -> None:
                await client.execute_cypher('''
                    MATCH (e:Entity)
                    WHERE e.created_at IS NULL
                    SET e.created_at = datetime(),
                        e.updated_at = datetime()
                ''')

            async def down(self, client) -> None:
                await client.execute_cypher('''
                    MATCH (e:Entity)
                    REMOVE e.created_at, e.updated_at
                ''')
        ```
    """

    # Migration metadata - override in subclass
    version: str = "000"
    description: str = "Base migration"
    dependencies: list[str] = []  # List of version IDs this depends on

    @abstractmethod
    async def up(self, client: Any) -> None:
        """
        Apply the migration.

        Args:
            client: OntologyGraphClient instance
        """
        pass

    @abstractmethod
    async def down(self, client: Any) -> None:
        """
        Revert the migration.

        Args:
            client: OntologyGraphClient instance
        """
        pass

    async def validate(self, client: Any) -> bool:
        """
        Validate migration can be applied.

        Override for custom validation logic.

        Args:
            client: OntologyGraphClient instance

        Returns:
            True if migration can be applied
        """
        return True

    def __repr__(self) -> str:
        return f"Migration({self.version}: {self.description})"

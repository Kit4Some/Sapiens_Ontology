"""
Authorization Module - Role-Based Access Control (RBAC).

Provides fine-grained access control:
- Hierarchical roles with inheritance
- Granular permissions
- Resource-level access control
- Policy-based authorization
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """System permissions."""

    # Entity permissions
    ENTITY_READ = "entity:read"
    ENTITY_CREATE = "entity:create"
    ENTITY_UPDATE = "entity:update"
    ENTITY_DELETE = "entity:delete"

    # Relation permissions
    RELATION_READ = "relation:read"
    RELATION_CREATE = "relation:create"
    RELATION_UPDATE = "relation:update"
    RELATION_DELETE = "relation:delete"

    # Query permissions
    QUERY_EXECUTE = "query:execute"
    QUERY_ADMIN = "query:admin"  # Raw Cypher execution

    # Ingestion permissions
    INGEST_READ = "ingest:read"
    INGEST_CREATE = "ingest:create"
    INGEST_DELETE = "ingest:delete"

    # System permissions
    SYSTEM_STATUS = "system:status"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_ADMIN = "system:admin"

    # User management
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"

    # API Key management
    APIKEY_READ = "apikey:read"
    APIKEY_CREATE = "apikey:create"
    APIKEY_REVOKE = "apikey:revoke"


class ResourceType(str, Enum):
    """Resource types for access control."""

    ENTITY = "entity"
    RELATION = "relation"
    CHUNK = "chunk"
    COMMUNITY = "community"
    DOCUMENT = "document"
    QUERY = "query"
    JOB = "job"
    USER = "user"
    APIKEY = "apikey"
    SYSTEM = "system"


@dataclass
class Role:
    """
    Role definition with permissions.

    Roles can inherit from parent roles.
    """

    name: str
    description: str
    permissions: set[Permission] = field(default_factory=set)
    parent_roles: list[str] = field(default_factory=list)
    is_system_role: bool = False

    def get_all_permissions(self, role_registry: dict[str, "Role"]) -> set[Permission]:
        """Get all permissions including inherited ones."""
        all_perms = set(self.permissions)

        for parent_name in self.parent_roles:
            parent = role_registry.get(parent_name)
            if parent:
                all_perms.update(parent.get_all_permissions(role_registry))

        return all_perms


@dataclass
class ResourcePolicy:
    """
    Policy for resource-level access control.

    Defines who can access specific resources.
    """

    resource_type: ResourceType
    resource_id: str | None = None  # None means all resources of type
    allowed_roles: list[str] = field(default_factory=list)
    allowed_users: list[str] = field(default_factory=list)
    denied_users: list[str] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)

    def matches(self, resource_type: ResourceType, resource_id: str | None) -> bool:
        """Check if policy matches the resource."""
        if self.resource_type != resource_type:
            return False
        if self.resource_id is not None and self.resource_id != resource_id:
            return False
        return True


class RBACManager:
    """
    Role-Based Access Control Manager.

    Features:
    - Hierarchical role management
    - Permission checking with inheritance
    - Resource-level policies
    - Audit trail integration
    """

    def __init__(self):
        self._roles: dict[str, Role] = {}
        self._user_roles: dict[str, set[str]] = {}  # user_id -> role names
        self._policies: list[ResourcePolicy] = []

        # Initialize default system roles
        self._init_system_roles()

    def _init_system_roles(self):
        """Initialize default system roles."""
        # Viewer - read-only access
        viewer = Role(
            name="viewer",
            description="Read-only access to entities and queries",
            permissions={
                Permission.ENTITY_READ,
                Permission.RELATION_READ,
                Permission.QUERY_EXECUTE,
                Permission.SYSTEM_STATUS,
            },
            is_system_role=True,
        )

        # Editor - can create and modify entities
        editor = Role(
            name="editor",
            description="Can create and modify entities",
            permissions={
                Permission.ENTITY_CREATE,
                Permission.ENTITY_UPDATE,
                Permission.RELATION_CREATE,
                Permission.RELATION_UPDATE,
                Permission.INGEST_READ,
                Permission.INGEST_CREATE,
            },
            parent_roles=["viewer"],
            is_system_role=True,
        )

        # Manager - can delete and manage
        manager = Role(
            name="manager",
            description="Full entity management including deletion",
            permissions={
                Permission.ENTITY_DELETE,
                Permission.RELATION_DELETE,
                Permission.INGEST_DELETE,
                Permission.USER_READ,
                Permission.APIKEY_READ,
                Permission.APIKEY_CREATE,
            },
            parent_roles=["editor"],
            is_system_role=True,
        )

        # Admin - full system access
        admin = Role(
            name="admin",
            description="Full system administration access",
            permissions={
                Permission.QUERY_ADMIN,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_ADMIN,
                Permission.USER_CREATE,
                Permission.USER_UPDATE,
                Permission.USER_DELETE,
                Permission.APIKEY_REVOKE,
            },
            parent_roles=["manager"],
            is_system_role=True,
        )

        # Register system roles
        for role in [viewer, editor, manager, admin]:
            self._roles[role.name] = role

        logger.info("System roles initialized", roles=list(self._roles.keys()))

    def create_role(
        self,
        name: str,
        description: str,
        permissions: list[Permission] | None = None,
        parent_roles: list[str] | None = None,
    ) -> Role:
        """Create a new custom role."""
        if name in self._roles:
            raise ValueError(f"Role '{name}' already exists")

        # Validate parent roles
        for parent in parent_roles or []:
            if parent not in self._roles:
                raise ValueError(f"Parent role '{parent}' does not exist")

        role = Role(
            name=name,
            description=description,
            permissions=set(permissions or []),
            parent_roles=parent_roles or [],
            is_system_role=False,
        )

        self._roles[name] = role
        logger.info("Role created", role=name, permissions=len(role.permissions))
        return role

    def delete_role(self, name: str) -> bool:
        """Delete a custom role."""
        if name not in self._roles:
            return False

        role = self._roles[name]
        if role.is_system_role:
            raise ValueError(f"Cannot delete system role '{name}'")

        # Remove role from all users
        for user_id, roles in self._user_roles.items():
            roles.discard(name)

        del self._roles[name]
        logger.info("Role deleted", role=name)
        return True

    def get_role(self, name: str) -> Role | None:
        """Get a role by name."""
        return self._roles.get(name)

    def list_roles(self, include_system: bool = True) -> list[Role]:
        """List all roles."""
        roles = list(self._roles.values())
        if not include_system:
            roles = [r for r in roles if not r.is_system_role]
        return roles

    def assign_role(self, user_id: str, role_name: str):
        """Assign a role to a user."""
        if role_name not in self._roles:
            raise ValueError(f"Role '{role_name}' does not exist")

        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()

        self._user_roles[user_id].add(role_name)
        logger.info("Role assigned", user_id=user_id, role=role_name)

    def revoke_role(self, user_id: str, role_name: str):
        """Revoke a role from a user."""
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role_name)
            logger.info("Role revoked", user_id=user_id, role=role_name)

    def get_user_roles(self, user_id: str) -> list[str]:
        """Get all roles assigned to a user."""
        return list(self._user_roles.get(user_id, set()))

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        """Get all permissions for a user (including inherited)."""
        permissions = set()

        for role_name in self._user_roles.get(user_id, set()):
            role = self._roles.get(role_name)
            if role:
                permissions.update(role.get_all_permissions(self._roles))

        return permissions

    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        user_roles: list[str] | None = None,
    ) -> bool:
        """
        Check if a user has a specific permission.

        Args:
            user_id: The user ID
            permission: The permission to check
            user_roles: Optional list of roles (from JWT/API key)
        """
        # Get roles from user assignment or provided roles
        roles_to_check = set()
        if user_roles:
            roles_to_check.update(user_roles)
        roles_to_check.update(self._user_roles.get(user_id, set()))

        # Check each role
        for role_name in roles_to_check:
            role = self._roles.get(role_name)
            if role:
                all_perms = role.get_all_permissions(self._roles)
                if permission in all_perms:
                    return True

        return False

    def check_resource_access(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> bool:
        """
        Check if user can access a specific resource.

        Combines permission checks with resource policies.
        """
        # First check if explicitly denied
        for policy in self._policies:
            if policy.matches(resource_type, resource_id):
                if user_id in policy.denied_users:
                    return False

        # Check policies for explicit access
        for policy in self._policies:
            if policy.matches(resource_type, resource_id):
                if user_id in policy.allowed_users:
                    return True

                # Check roles
                roles = set(user_roles or [])
                roles.update(self._user_roles.get(user_id, set()))
                if any(r in policy.allowed_roles for r in roles):
                    return True

        # Fall back to permission-based check
        read_permission = Permission(f"{resource_type.value}:read")
        return self.check_permission(user_id, read_permission, user_roles)

    def add_policy(self, policy: ResourcePolicy):
        """Add a resource access policy."""
        self._policies.append(policy)
        logger.info(
            "Policy added",
            resource_type=policy.resource_type.value,
            resource_id=policy.resource_id,
        )

    def remove_policy(
        self,
        resource_type: ResourceType,
        resource_id: str | None = None,
    ):
        """Remove policies matching the resource."""
        self._policies = [
            p for p in self._policies if not p.matches(resource_type, resource_id)
        ]


# =============================================================================
# FastAPI Integration
# =============================================================================


def check_permission(permission: Permission):
    """
    FastAPI dependency to check permissions.

    Usage:
        @app.get("/admin/config")
        async def get_config(
            _: None = Depends(check_permission(Permission.SYSTEM_CONFIG))
        ):
            return {"config": "data"}
    """
    from fastapi import Depends, HTTPException, Request, status

    from src.security.authentication import get_current_user, User

    async def permission_checker(
        request: Request,
        user: User | None = Depends(get_current_user),
    ):
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        rbac = get_rbac_manager()
        if not rbac.check_permission(user.id, permission, user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value}",
            )

        return None

    return Depends(permission_checker)


def check_resource_access(resource_type: ResourceType):
    """
    FastAPI dependency for resource-level access control.

    Usage:
        @app.get("/entity/{entity_id}")
        async def get_entity(
            entity_id: str,
            _: None = Depends(check_resource_access(ResourceType.ENTITY)),
        ):
            return {"entity_id": entity_id}
    """
    from fastapi import Depends, HTTPException, Request, status

    from src.security.authentication import get_current_user, User

    async def resource_checker(
        request: Request,
        user: User | None = Depends(get_current_user),
    ):
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        # Extract resource ID from path params
        resource_id = request.path_params.get("id") or request.path_params.get(
            f"{resource_type.value}_id"
        )

        rbac = get_rbac_manager()
        if not rbac.check_resource_access(
            user.id, resource_type, resource_id, user.roles
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to {resource_type.value}: {resource_id}",
            )

        return None

    return Depends(resource_checker)


# =============================================================================
# Global Instance
# =============================================================================

_rbac_manager: RBACManager | None = None


def get_rbac_manager() -> RBACManager:
    """Get global RBAC manager instance."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def init_rbac_manager() -> RBACManager:
    """Initialize and return RBAC manager."""
    global _rbac_manager
    _rbac_manager = RBACManager()
    return _rbac_manager

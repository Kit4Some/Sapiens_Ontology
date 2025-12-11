"""
Security Module.

Provides comprehensive security features:
- Authentication (JWT + API Key)
- Authorization (RBAC)
- Rate limiting
- Audit logging
- Secure credential handling
"""

from src.security.authentication import (
    AuthenticationMiddleware,
    JWTAuthenticator,
    APIKeyAuthenticator,
    get_current_user,
    require_auth,
    require_role,
)
from src.security.authorization import (
    Permission,
    Role,
    RBACManager,
    check_permission,
)
from src.security.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    get_audit_logger,
)
from src.security.rate_limiter import (
    RateLimiter,
    RateLimitMiddleware,
    RateLimitConfig,
)

__all__ = [
    # Authentication
    "AuthenticationMiddleware",
    "JWTAuthenticator",
    "APIKeyAuthenticator",
    "get_current_user",
    "require_auth",
    "require_role",
    # Authorization
    "Permission",
    "Role",
    "RBACManager",
    "check_permission",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "get_audit_logger",
    # Rate Limiting
    "RateLimiter",
    "RateLimitMiddleware",
    "RateLimitConfig",
]

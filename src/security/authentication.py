"""
Authentication Module.

Provides authentication mechanisms:
- JWT (JSON Web Token) authentication
- API Key authentication
- Combined authentication middleware
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class AuthMethod(str, Enum):
    """Authentication methods."""
    JWT = "jwt"
    API_KEY = "api_key"
    NONE = "none"


@dataclass
class User:
    """Authenticated user information."""

    id: str
    username: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    auth_method: AuthMethod = AuthMethod.NONE
    metadata: dict[str, Any] = field(default_factory=dict)

    # Session info
    authenticated_at: datetime | None = None
    expires_at: datetime | None = None

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles or "admin" in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions or "admin" in self.roles

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "roles": self.roles,
            "permissions": self.permissions,
            "auth_method": self.auth_method.value,
        }


@dataclass
class APIKey:
    """API Key information."""

    key_id: str
    key_hash: str  # Hashed key, never store plaintext
    name: str
    user_id: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    is_active: bool = True
    rate_limit: int = 1000  # Requests per hour

    def is_valid(self) -> bool:
        """Check if API key is valid and not expired."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


class JWTAuthenticator:
    """
    JWT-based authentication.

    Features:
    - Token generation and validation
    - Refresh token support
    - Configurable expiration
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_expire = timedelta(minutes=access_token_expire_minutes)
        self._refresh_expire = timedelta(days=refresh_token_expire_days)

        # Check for JWT library
        try:
            import jwt
            self._jwt = jwt
            self._available = True
        except ImportError:
            logger.warning("PyJWT not installed, JWT auth unavailable")
            self._available = False

    def create_access_token(
        self,
        user: User,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create an access token for user."""
        if not self._available:
            raise RuntimeError("JWT library not available")

        now = datetime.utcnow()
        expires = now + self._access_expire

        payload = {
            "sub": user.id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "iat": now,
            "exp": expires,
            "type": "access",
        }

        if additional_claims:
            payload.update(additional_claims)

        return self._jwt.encode(payload, self._secret_key, algorithm=self._algorithm)

    def create_refresh_token(self, user: User) -> str:
        """Create a refresh token for user."""
        if not self._available:
            raise RuntimeError("JWT library not available")

        now = datetime.utcnow()
        expires = now + self._refresh_expire

        payload = {
            "sub": user.id,
            "iat": now,
            "exp": expires,
            "type": "refresh",
            "jti": secrets.token_hex(16),  # Unique token ID
        }

        return self._jwt.encode(payload, self._secret_key, algorithm=self._algorithm)

    def validate_token(self, token: str) -> User | None:
        """Validate a JWT token and return user info."""
        if not self._available:
            return None

        try:
            payload = self._jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )

            # Check token type
            if payload.get("type") != "access":
                logger.warning("Invalid token type", type=payload.get("type"))
                return None

            return User(
                id=payload["sub"],
                username=payload.get("username", ""),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                auth_method=AuthMethod.JWT,
                authenticated_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
            )

        except self._jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except self._jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None

    def refresh_access_token(self, refresh_token: str) -> tuple[str, str] | None:
        """Use refresh token to get new access and refresh tokens."""
        if not self._available:
            return None

        try:
            payload = self._jwt.decode(
                refresh_token,
                self._secret_key,
                algorithms=[self._algorithm],
            )

            if payload.get("type") != "refresh":
                return None

            # Create new user object for token generation
            user = User(
                id=payload["sub"],
                username="",  # Will be filled from DB in real implementation
                auth_method=AuthMethod.JWT,
            )

            new_access = self.create_access_token(user)
            new_refresh = self.create_refresh_token(user)

            return new_access, new_refresh

        except Exception as e:
            logger.warning("Refresh token validation failed", error=str(e))
            return None


class APIKeyAuthenticator:
    """
    API Key-based authentication.

    Features:
    - Secure key generation
    - Key hashing (keys are never stored in plaintext)
    - Rate limiting per key
    - Key rotation support
    """

    def __init__(self):
        self._keys: dict[str, APIKey] = {}  # key_id -> APIKey
        self._key_lookup: dict[str, str] = {}  # hash -> key_id

    @staticmethod
    def generate_key() -> tuple[str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (key_id, plaintext_key)
            The plaintext key should be shown to user once, then discarded.
        """
        key_id = f"key_{secrets.token_hex(8)}"
        plaintext_key = f"ont_{secrets.token_urlsafe(32)}"
        return key_id, plaintext_key

    @staticmethod
    def hash_key(plaintext_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(plaintext_key.encode()).hexdigest()

    def register_key(
        self,
        key_id: str,
        plaintext_key: str,
        name: str,
        user_id: str,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        expires_at: datetime | None = None,
        rate_limit: int = 1000,
    ) -> APIKey:
        """Register a new API key."""
        key_hash = self.hash_key(plaintext_key)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            roles=roles or [],
            permissions=permissions or [],
            expires_at=expires_at,
            rate_limit=rate_limit,
        )

        self._keys[key_id] = api_key
        self._key_lookup[key_hash] = key_id

        logger.info("API key registered", key_id=key_id, name=name)
        return api_key

    def validate_key(self, plaintext_key: str) -> User | None:
        """Validate an API key and return user info."""
        key_hash = self.hash_key(plaintext_key)

        key_id = self._key_lookup.get(key_hash)
        if not key_id:
            logger.warning("Unknown API key")
            return None

        api_key = self._keys.get(key_id)
        if not api_key or not api_key.is_valid():
            logger.warning("Invalid or expired API key", key_id=key_id)
            return None

        # Update last used timestamp
        api_key.last_used_at = datetime.utcnow()

        return User(
            id=api_key.user_id,
            username=f"api_key:{api_key.name}",
            roles=api_key.roles,
            permissions=api_key.permissions,
            auth_method=AuthMethod.API_KEY,
            authenticated_at=datetime.utcnow(),
            expires_at=api_key.expires_at,
            metadata={"key_id": api_key.key_id, "rate_limit": api_key.rate_limit},
        )

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            api_key = self._keys[key_id]
            api_key.is_active = False
            logger.info("API key revoked", key_id=key_id)
            return True
        return False

    def list_keys(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List API keys, optionally filtered by user."""
        keys = []
        for api_key in self._keys.values():
            if user_id and api_key.user_id != user_id:
                continue
            keys.append({
                "key_id": api_key.key_id,
                "name": api_key.name,
                "user_id": api_key.user_id,
                "roles": api_key.roles,
                "is_active": api_key.is_active,
                "created_at": api_key.created_at.isoformat(),
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            })
        return keys


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for authentication.

    Supports both JWT and API Key authentication.
    Adds user info to request state.
    """

    def __init__(
        self,
        app,
        jwt_authenticator: JWTAuthenticator | None = None,
        api_key_authenticator: APIKeyAuthenticator | None = None,
        exclude_paths: list[str] | None = None,
        require_auth: bool = False,
    ):
        super().__init__(app)
        self._jwt_auth = jwt_authenticator
        self._api_key_auth = api_key_authenticator
        self._exclude_paths = exclude_paths or [
            "/api/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics",
        ]
        self._require_auth = require_auth

    async def dispatch(self, request: Request, call_next):
        """Process authentication for each request."""
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self._exclude_paths):
            return await call_next(request)

        user = None

        # Try JWT authentication (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer ") and self._jwt_auth:
            token = auth_header[7:]
            user = self._jwt_auth.validate_token(token)

        # Try API Key authentication
        if not user and self._api_key_auth:
            api_key = request.headers.get("X-API-Key")
            if api_key:
                user = self._api_key_auth.validate_key(api_key)

        # Check if authentication is required
        if self._require_auth and not user:
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Add user to request state
        request.state.user = user
        request.state.is_authenticated = user is not None

        # Log authentication
        if user:
            logger.debug(
                "Request authenticated",
                user_id=user.id,
                auth_method=user.auth_method.value,
                path=request.url.path,
            )

        return await call_next(request)


# =============================================================================
# FastAPI Dependencies
# =============================================================================

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(request: Request) -> User | None:
    """
    FastAPI dependency to get current authenticated user.

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            if not user:
                raise HTTPException(status_code=401)
            return {"user": user.username}
    """
    return getattr(request.state, "user", None)


async def get_required_user(request: Request) -> User:
    """
    FastAPI dependency that requires authentication.

    Raises 401 if not authenticated.
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication for a route.

    Usage:
        @app.get("/protected")
        @require_auth
        async def protected_route(request: Request):
            user = request.state.user
            return {"user": user.username}
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Find request in args or kwargs
        request = kwargs.get("request")
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

        if not request or not getattr(request.state, "is_authenticated", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        return await func(*args, **kwargs)

    return wrapper


def require_role(*roles: str) -> Callable:
    """
    Decorator to require specific roles.

    Usage:
        @app.get("/admin")
        @require_role("admin")
        async def admin_route(request: Request):
            return {"message": "Admin access"}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            user = getattr(request.state, "user", None) if request else None

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not any(user.has_role(role) for role in roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required roles: {', '.join(roles)}",
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_permission(*permissions: str) -> Callable:
    """
    Decorator to require specific permissions.

    Usage:
        @app.delete("/entity/{id}")
        @require_permission("entity:delete")
        async def delete_entity(request: Request, id: str):
            return {"deleted": id}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            user = getattr(request.state, "user", None) if request else None

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not any(user.has_permission(perm) for perm in permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required permissions: {', '.join(permissions)}",
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Global Instances
# =============================================================================

_jwt_authenticator: JWTAuthenticator | None = None
_api_key_authenticator: APIKeyAuthenticator | None = None


def init_authentication(
    jwt_secret: str | None = None,
    jwt_expire_minutes: int = 30,
) -> tuple[JWTAuthenticator | None, APIKeyAuthenticator]:
    """Initialize global authentication instances."""
    global _jwt_authenticator, _api_key_authenticator

    if jwt_secret:
        _jwt_authenticator = JWTAuthenticator(
            secret_key=jwt_secret,
            access_token_expire_minutes=jwt_expire_minutes,
        )

    _api_key_authenticator = APIKeyAuthenticator()

    return _jwt_authenticator, _api_key_authenticator


def get_jwt_authenticator() -> JWTAuthenticator | None:
    """Get global JWT authenticator."""
    return _jwt_authenticator


def get_api_key_authenticator() -> APIKeyAuthenticator:
    """Get global API key authenticator."""
    global _api_key_authenticator
    if _api_key_authenticator is None:
        _api_key_authenticator = APIKeyAuthenticator()
    return _api_key_authenticator

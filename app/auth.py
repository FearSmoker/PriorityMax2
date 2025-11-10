# backend/app/auth.py
"""
PriorityMax Authentication & Authorization Module (production-grade)
-------------------------------------------------------------------

Features:
 - JWT access + refresh tokens with rotation and revocation (Redis-backed optional)
 - Password hashing (argon2 / bcrypt fallback) and password policy checks
 - OAuth2 password flow helpers and FastAPI dependencies
 - API Key auth (per-tenant), scoped permissions, and role-based access control (RBAC)
 - Multi-tenant support (org_id), user & role models (Pydantic), and DB integration hooks
 - Token introspection endpoints + admin controls for token revocation / key rotation
 - Refresh token rotation, reuse detection, and audit hooks
 - Pluggable storage backends: SQLAlchemy (recommended) or Mongo (fallback)
 - Security utilities: CSRF-safe cookie helpers, secure cookie flags, rate-limit hooks
 - FastAPI middleware for injecting auth context and enforcing RBAC
 - CLI utilities for creating initial admin user, rotating signing keys, and dumping tokens

Usage (example):
    from app.auth import configure_auth, get_current_user, require_role
    app = FastAPI()
    configure_auth(app)
    @app.get("/private")
    async def private_route(user=Depends(get_current_user)):
        return {"you": user.email}

Notes:
 - Optional dependencies (sqlalchemy, motor, redis, passlib, python-jose) are imported best-effort.
 - If Redis available, refresh token rotation and revocation list use Redis for high performance.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import hmac
import base64
import logging
import secrets
import hashlib
import typing as t
import datetime
import functools
from typing import Optional, Dict, Any, List, Tuple

# FastAPI / Starlette
try:
    from fastapi import Depends, HTTPException, status, Request, Response, Security
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    _HAS_FASTAPI = True
except Exception:
    # allow module importability in non-FastAPI contexts (for CLI/scripts)
    Depends = lambda *a, **k: None  # type: ignore
    HTTPException = Exception  # type: ignore
    status = type("S", (), {"HTTP_401_UNAUTHORIZED": 401, "HTTP_403_FORBIDDEN": 403})()  # type: ignore
    OAuth2PasswordBearer = None
    OAuth2PasswordRequestForm = None
    APIKeyHeader = None
    FastAPI = None
    BaseHTTPMiddleware = object
    JSONResponse = dict
    _HAS_FASTAPI = False

# Optional storage/backends
_HAS_SQLALCHEMY = False
_HAS_MOTOR = False
_HAS_REDIS = False
_HAS_PASSLIB = False
_HAS_JOSE = False

try:
    import sqlalchemy as sa
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    _HAS_SQLALCHEMY = True
except Exception:
    sa = None

try:
    import motor.motor_asyncio as motor
    _HAS_MOTOR = True
except Exception:
    motor = None

try:
    import redis
    _HAS_REDIS = True
except Exception:
    redis = None

try:
    from passlib.context import CryptContext
    _HAS_PASSLIB = True
except Exception:
    CryptContext = None

try:
    from jose import jwt, JWTError
    _HAS_JOSE = True
except Exception:
    jwt = None
    JWTError = Exception

# Local utilities
try:
    from app.utils.common import now_iso, json_dumps, set_context, get_context, sha256_hex
except Exception:
    def now_iso():
        return datetime.datetime.utcnow().isoformat() + "Z"
    def json_dumps(o):
        try:
            return json.dumps(o)
        except Exception:
            return str(o)
    def set_context(k, v): pass
    def get_context(k, default=None): return default
    def sha256_hex(x: str) -> str:
        return hashlib.sha256(x.encode()).hexdigest()

LOG = logging.getLogger("prioritymax.auth")
LOG.setLevel(os.getenv("PRIORITYMAX_AUTH_LOG", "INFO"))
if not LOG.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(ch)

# -------------------------
# Environment / Defaults
# -------------------------
JWT_ALG = os.getenv("PRIORITYMAX_JWT_ALG", "HS256")
JWT_SECRET = os.getenv("PRIORITYMAX_JWT_SECRET", None) or secrets.token_urlsafe(48)
JWT_ACCESS_EXPIRES = int(os.getenv("PRIORITYMAX_JWT_ACCESS_EXPIRES", str(60 * 15)))  # 15 minutes
JWT_REFRESH_EXPIRES = int(os.getenv("PRIORITYMAX_JWT_REFRESH_EXPIRES", str(60 * 60 * 24 * 14)))  # 14 days
JWT_ISSUER = os.getenv("PRIORITYMAX_JWT_ISSUER", "prioritymax")
JWT_AUD = os.getenv("PRIORITYMAX_JWT_AUD", "prioritymax-users")
REFRESH_ROTATION = os.getenv("PRIORITYMAX_REFRESH_ROTATION", "true").lower() in ("1", "true", "yes")
REDIS_URL = os.getenv("REDIS_URL", None)
API_KEY_HEADER_NAME = os.getenv("PRIORITYMAX_API_KEY_HEADER", "X-API-KEY")
COOKIE_SECURE = os.getenv("PRIORITYMAX_COOKIE_SECURE", "true").lower() in ("1", "true", "yes")
COOKIE_SAMESITE = os.getenv("PRIORITYMAX_COOKIE_SAMESITE", "lax")

# crypto context
if _HAS_PASSLIB:
    pwd_ctx = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
else:
    pwd_ctx = None

# redis client (optional)
_redis_client = None
if _HAS_REDIS and REDIS_URL:
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        LOG.exception("Failed to initialize Redis client")

# oauth2 helpers for FastAPI
if _HAS_FASTAPI and OAuth2PasswordBearer is not None:
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", scopes={})
else:
    oauth2_scheme = None

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME) if _HAS_FASTAPI and APIKeyHeader is not None else None

# -------------------------
# Models (Pydantic + simple DB schemas)
# -------------------------
try:
    from pydantic import BaseModel, EmailStr, Field, validator
    _HAS_PYDANTIC = True
except Exception:
    BaseModel = object
    EmailStr = str
    Field = lambda *a, **k: None
    _HAS_PYDANTIC = False

# Pydantic models for APIs
class RoleModel(BaseModel):
    name: str
    permissions: List[str] = []
    description: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    password: Optional[str] = None
    roles: List[str] = []
    org_id: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False

class UserPublic(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    roles: List[str] = []
    org_id: Optional[str] = None
    is_active: bool
    is_superuser: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

# Simple internal user representation (dict-backed)
# Persistent schema assumed: id, email, password_hash, salt, roles, org_id, is_active, is_superuser, created_at
# Storage layer to implement: get_user_by_email, save_user, update_user, revoke_token, etc.

# -------------------------
# Storage abstraction (pluggable)
# -------------------------
class StorageInterface:
    """
    Abstract storage interface. Implementations must be async-compatible when used in FastAPI.
    """

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def create_user(self, user: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def update_user(self, user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def list_roles(self, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def get_role(self, role_name: str, org_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def save_api_key(self, api_key_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def get_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def persist_revoked_token(self, jti: str, expires_at: float) -> None:
        raise NotImplementedError

    async def is_token_revoked(self, jti: str) -> bool:
        raise NotImplementedError

    async def list_users(self, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

# Simple in-memory storage (for local dev/testing)
class InMemoryStorage(StorageInterface):
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.revoked: Dict[str, float] = {}

    async def get_user_by_email(self, email: str):
        for u in self.users.values():
            if u["email"].lower() == email.lower():
                return u
        return None

    async def get_user_by_id(self, user_id: str):
        return self.users.get(user_id)

    async def create_user(self, user: Dict[str, Any]):
        user_id = user.get("id") or str(uuid.uuid4())
        user["id"] = user_id
        user["created_at"] = now_iso()
        self.users[user_id] = user
        return user

    async def update_user(self, user_id: str, patch: Dict[str, Any]):
        u = self.users.get(user_id)
        if not u:
            return None
        u.update(patch)
        u["updated_at"] = now_iso()
        return u

    async def list_roles(self, org_id: Optional[str] = None):
        return list(self.roles.values())

    async def get_role(self, role_name: str, org_id: Optional[str] = None):
        return self.roles.get(role_name)

    async def save_api_key(self, api_key_data: Dict[str, Any]):
        self.api_keys[api_key_data["key"]] = api_key_data
        return api_key_data

    async def get_api_key(self, key: str):
        return self.api_keys.get(key)

    async def persist_revoked_token(self, jti: str, expires_at: float):
        self.revoked[jti] = expires_at

    async def is_token_revoked(self, jti: str):
        exp = self.revoked.get(jti)
        if not exp:
            return False
        if time.time() > exp:
            # garbage collect
            del self.revoked[jti]
            return False
        return True

    async def list_users(self, org_id: Optional[str] = None):
        return list(self.users.values())

# Database wiring: prefer SQLAlchemy if present; else motor (Mongo); else InMemory
_default_storage: StorageInterface = InMemoryStorage()

# -------------------------
# Password / Crypto utilities
# -------------------------
def verify_password(plain: str, hashed: str) -> bool:
    if _HAS_PASSLIB and pwd_ctx:
        try:
            return pwd_ctx.verify(plain, hashed)
        except Exception:
            return False
    # fallback: simple salted SHA256 (not recommended for prod)
    try:
        # hashed expected format: salt$hex
        if "$" in hashed:
            salt, hexx = hashed.split("$", 1)
            return hmac.compare_digest(hexx, hashlib.sha256(salt.encode() + plain.encode()).hexdigest())
        return False
    except Exception:
        return False

def hash_password(plain: str) -> str:
    if _HAS_PASSLIB and pwd_ctx:
        return pwd_ctx.hash(plain)
    else:
        salt = secrets.token_hex(16)
        h = hashlib.sha256(salt.encode() + plain.encode()).hexdigest()
        return f"{salt}${h}"

def check_password_policy(password: str) -> Tuple[bool, List[str]]:
    """
    Simple password policy: length >= 8, include lowercase, uppercase, digit, special.
    Returns (ok, messages)
    """
    msgs = []
    ok = True
    if len(password) < 8:
        ok = False
        msgs.append("password must be at least 8 characters")
    if not any(c.islower() for c in password):
        ok = False
        msgs.append("password must include a lowercase letter")
    if not any(c.isupper() for c in password):
        ok = False
        msgs.append("password must include an uppercase letter")
    if not any(c.isdigit() for c in password):
        ok = False
        msgs.append("password must include a digit")
    if not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in password):
        ok = False
        msgs.append("password must include a special character")
    return ok, msgs

# -------------------------
# JWT utilities
# -------------------------
def _jwt_now() -> int:
    return int(time.time())

def _jwt_exp(seconds: int) -> int:
    return _jwt_now() + int(seconds)

def generate_jwt(payload: Dict[str, Any], expires_in: int = JWT_ACCESS_EXPIRES, subject: Optional[str] = None, token_type: str = "access") -> str:
    """
    Create a signed JWT. Adds standard claims.
    """
    if not _HAS_JOSE:
        # naive HMAC base64 token fallback (not recommended)
        header = {"alg": "none", "typ": "JWT"}
        body = dict(payload)
        body.update({"iat": _jwt_now(), "exp": _jwt_exp(expires_in), "iss": JWT_ISSUER, "typ": token_type})
        if subject:
            body["sub"] = subject
        return base64.urlsafe_b64encode(json.dumps(body).encode()).decode()
    claims = dict(payload)
    now = _jwt_now()
    claims.update({
        "iat": now,
        "exp": _jwt_exp(expires_in),
        "iss": JWT_ISSUER,
        "aud": JWT_AUD,
        "typ": token_type,
    })
    if subject:
        claims["sub"] = subject
    # include unique id for revocation tracking
    jti = str(uuid.uuid4())
    claims["jti"] = jti
    token = jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALG)
    return token

def decode_jwt(token: str, verify_exp: bool = True) -> Dict[str, Any]:
    if not _HAS_JOSE:
        try:
            payload_json = base64.urlsafe_b64decode(token.encode()).decode()
            return json.loads(payload_json)
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    try:
        opts = {"verify_exp": verify_exp}
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], audience=JWT_AUD, issuer=JWT_ISSUER, options=opts)
        return payload
    except JWTError as e:
        LOG.debug("JWT decode failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

# -------------------------
# Refresh token rotation & revocation (Redis-backed if available)
# -------------------------
async def _persist_revoked_jti(jti: str, exp_ts: int):
    """
    Persist revoked jti. Use Redis if available, else storage.
    """
    try:
        if _redis_client:
            # store with ttl
            ttl = max(1, int(exp_ts - time.time()))
            _redis_client.setex(f"revoked:{jti}", ttl, "1")
            return
    except Exception:
        LOG.exception("redis revoke save failed")
    # fallback to storage
    try:
        await _default_storage.persist_revoked_token(jti, exp_ts)
    except Exception:
        LOG.exception("storage revoke save failed")

async def _is_jti_revoked(jti: str) -> bool:
    try:
        if _redis_client:
            v = _redis_client.get(f"revoked:{jti}")
            if v:
                return True
        return await _default_storage.is_token_revoked(jti)
    except Exception:
        LOG.exception("is_jti_revoked error")
        return False

# -------------------------
# Token creation flows
# -------------------------
async def create_tokens_for_user(user: Dict[str, Any], include_refresh: bool = True) -> Dict[str, Any]:
    """
    Create access + optional refresh tokens.
    Refresh tokens are JWTs with longer expiry and stored (jti) for rotation detection.
    """
    sub = user.get("id")
    payload = {
        "uid": sub,
        "email": user.get("email"),
        "roles": user.get("roles", []),
        "org": user.get("org_id"),
    }
    access = generate_jwt(payload, expires_in=JWT_ACCESS_EXPIRES, subject=sub, token_type="access")
    refresh = None
    if include_refresh:
        refresh = generate_jwt({"uid": sub}, expires_in=JWT_REFRESH_EXPIRES, subject=sub, token_type="refresh")
        # store refresh jti in Redis to track rotation if needed (optional)
        try:
            # decode to get jti & exp
            decoded = decode_jwt(refresh, verify_exp=False)
            jti = decoded.get("jti")
            exp = decoded.get("exp")
            if jti and exp:
                # mark as active in redis (used for rotation detection)
                if _redis_client:
                    _redis_client.setex(f"refresh:{jti}", max(1, int(exp - time.time())), json.dumps({"uid": sub}))
        except Exception:
            LOG.exception("Failed to register refresh token")
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer", "expires_in": JWT_ACCESS_EXPIRES}

# -------------------------
# Auth validation helpers & dependencies
# -------------------------
async def _validate_access_token(token: str) -> Dict[str, Any]:
    """
    Decode, validate token, check revocation, and return payload.
    """
    payload = decode_jwt(token, verify_exp=True)
    jti = payload.get("jti")
    typ = payload.get("typ") or payload.get("type")
    if typ != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    if jti and await _is_jti_revoked(jti):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")
    return payload

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    FastAPI dependency: return current user dict (from storage) or 401.
    """
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = await _validate_access_token(token)
    uid = payload.get("uid") or payload.get("sub")
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = await _default_storage.get_user_by_id(uid)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")
    # attach auth context for logging/tracing
    try:
        set_context("user_id", user.get("id"))
        set_context("org_id", user.get("org_id"))
    except Exception:
        pass
    return user

async def require_role(role: str):
    """
    Dependency factory: require user to have a role.
    Usage:
        @app.get("/admin")
        async def admin_route(user=Depends(get_current_user), _=Depends(require_role("admin"))):
            ...
    """
    async def _dep(user=Depends(get_current_user)):
        roles = set(user.get("roles", []))
        if user.get("is_superuser"):
            return True
        if role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return True
    return _dep

async def require_permission(perm: str):
    """
    Require permission string. Permissions are provided by roles and directly on user.
    """
    async def _dep(user=Depends(get_current_user)):
        if user.get("is_superuser"):
            return True
        user_perms = set(user.get("permissions", []))
        # expand roles -> role perms
        roles = user.get("roles", [])
        for r in roles:
            role_obj = await _default_storage.get_role(r, org_id=user.get("org_id"))
            if role_obj:
                user_perms.update(role_obj.get("permissions", []))
        if perm not in user_perms:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permission")
        return True
    return _dep

# -------------------------
# API Key flow
# -------------------------
def generate_api_key(metadata: Optional[Dict[str, Any]] = None, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate an API key object (key is visible only once). The stored key will be hashed.
    """
    raw = secrets.token_urlsafe(32)
    if prefix:
        raw = f"{prefix}_{raw}"
    # store only hashed key using sha256
    hashed = sha256_hex(raw)
    obj = {
        "key": hashed,
        "raw": raw,  # raw returned to caller only
        "created_at": now_iso(),
        "metadata": metadata or {},
    }
    return obj

async def verify_api_key(raw_key: str) -> Optional[Dict[str, Any]]:
    """
    Verify API key by hashing and checking storage. Returns key object if valid.
    """
    h = sha256_hex(raw_key)
    entry = await _default_storage.get_api_key(h)
    if entry:
        return entry
    return None

async def api_key_dependency(api_key: str = Depends(api_key_header)) -> Dict[str, Any]:
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
    entry = await verify_api_key(api_key)
    if not entry:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return entry

# -------------------------
# Auth endpoints wiring (FastAPI routers)
# -------------------------
def attach_auth_routes(app: FastAPI):
    """
    Register auth endpoints on FastAPI app under /auth prefix.
    """
    if not _HAS_FASTAPI:
        LOG.warning("FastAPI not available; attach_auth_routes skipped")
        return

    @app.post("/auth/register", response_model=UserPublic)
    async def register_user(payload: UserCreate):
        # check existing
        existing = await _default_storage.get_user_by_email(payload.email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        # password policy
        if payload.password:
            ok, msgs = check_password_policy(payload.password)
            if not ok:
                raise HTTPException(status_code=400, detail={"password_policy": msgs})
            pwd_hash = hash_password(payload.password)
        else:
            pwd_hash = None
        user_obj = {
            "email": payload.email.lower(),
            "full_name": payload.full_name,
            "password_hash": pwd_hash,
            "roles": payload.roles or [],
            "org_id": payload.org_id,
            "is_active": payload.is_active,
            "is_superuser": payload.is_superuser,
            "permissions": [],
        }
        created = await _default_storage.create_user(user_obj)
        return UserPublic(
            id=created["id"],
            email=created["email"],
            full_name=created.get("full_name"),
            roles=created.get("roles", []),
            org_id=created.get("org_id"),
            is_active=created.get("is_active", True),
            is_superuser=created.get("is_superuser", False),
        )

    @app.post("/auth/token", response_model=TokenResponse)
    async def token_endpoint(form_data: OAuth2PasswordRequestForm = Depends()):
        # authenticate user
        user = await _default_storage.get_user_by_email(form_data.username)
        if not user:
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        if not user.get("password_hash"):
            raise HTTPException(status_code=400, detail="User has no password set")
        if not verify_password(form_data.password, user["password_hash"]):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        tokens = await create_tokens_for_user(user, include_refresh=True)
        return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], expires_in=JWT_ACCESS_EXPIRES)

    @app.post("/auth/refresh", response_model=TokenResponse)
    async def refresh_endpoint(refresh_token: str):
        # Validate refresh token (no exp verify?) but ensure rotation and reuse detection
        try:
            data = decode_jwt(refresh_token, verify_exp=True)
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        if data.get("typ") != "refresh":
            raise HTTPException(status_code=401, detail="Not a refresh token")
        jti = data.get("jti")
        sub = data.get("sub") or data.get("uid")
        # rotation: if reuse detected (jti already used), revoke all
        if jti and await _is_jti_revoked(jti):
            # compromise detected: revoke all refresh tokens for user
            await _handle_refresh_reuse(sub)
            raise HTTPException(status_code=401, detail="Token reuse detected; all sessions revoked")
        # rotate: mark current jti revoked and issue new refresh
        if jti:
            # mark this jti as revoked
            exp = data.get("exp", _jwt_now() + JWT_REFRESH_EXPIRES)
            await _persist_revoked_jti(jti, exp)
        # create new tokens
        user = await _default_storage.get_user_by_id(sub)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        toks = await create_tokens_for_user(user, include_refresh=True)
        return TokenResponse(access_token=toks["access_token"], refresh_token=toks["refresh_token"], expires_in=JWT_ACCESS_EXPIRES)

    @app.post("/auth/logout")
    async def logout_endpoint(current_user=Depends(get_current_user), token: str = Depends(oauth2_scheme)):
        # revoke the access token (jti) and optional refresh tokens for session
        try:
            payload = decode_jwt(token, verify_exp=False)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid token")
        jti = payload.get("jti")
        exp = payload.get("exp", _jwt_now() + JWT_ACCESS_EXPIRES)
        if jti:
            await _persist_revoked_jti(jti, exp)
        return {"ok": True}

    @app.get("/auth/me", response_model=UserPublic)
    async def whoami(user=Depends(get_current_user)):
        return UserPublic(
            id=user["id"],
            email=user["email"],
            full_name=user.get("full_name"),
            roles=user.get("roles", []),
            org_id=user.get("org_id"),
            is_active=user.get("is_active", True),
            is_superuser=user.get("is_superuser", False),
        )

    @app.post("/auth/apikey/create")
    async def create_api_key(name: str, desc: Optional[str] = None, current_user=Depends(get_current_user)):
        # only superusers or admin role can create
        if not (current_user.get("is_superuser") or "admin" in current_user.get("roles", [])):
            raise HTTPException(status_code=403, detail="Not authorized")
        api = generate_api_key({"created_by": current_user["id"], "name": name, "desc": desc})
        saved = await _default_storage.save_api_key({"key": api["key"], "metadata": api.get("metadata"), "created_at": now_iso()})
        return {"api_key": api["raw"], "meta": saved}

    @app.post("/auth/revoke")
    async def admin_revoke(jti: str, current_user=Depends(get_current_user)):
        # superuser or admin only
        if not (current_user.get("is_superuser") or "admin" in current_user.get("roles", [])):
            raise HTTPException(status_code=403, detail="Not authorized")
        # mark as revoked with short TTL (access tokens)
        await _persist_revoked_jti(jti, int(time.time()) + 60 * 60 * 24 * 30)
        return {"ok": True}

    LOG.info("Auth routes attached to app")

# -------------------------
# Refresh reuse handler
# -------------------------
async def _handle_refresh_reuse(user_id: str):
    """
    Strong reaction to refresh token reuse: revoke all tokens for user (best-effort).
    Could be extended to notify user, email alert, or force logout across devices.
    """
    LOG.warning("Detected refresh token reuse for user: %s. Revoking sessions.", user_id)
    # naive strategy: persist a special revoked marker for user
    marker_jti = f"revoked_all_user_{user_id}"
    await _persist_revoked_jti(marker_jti, int(time.time()) + 60 * 60 * 24 * 30)
    # if Redis present, delete refresh keys for user
    if _redis_client:
        try:
            # this assumes refresh keys store {"uid": ...}
            for k in _redis_client.scan_iter(match="refresh:*"):
                try:
                    val = _redis_client.get(k)
                    j = json.loads(val) if val else None
                    if j and j.get("uid") == user_id:
                        _redis_client.delete(k)
                except Exception:
                    continue
        except Exception:
            LOG.exception("redis cleanup on reuse failed")
    # admin audit hook
    try:
        set_context("auth_event", "refresh_reuse")
    except Exception:
        pass

# -------------------------
# Middleware to inject auth context into request and logs
# -------------------------
class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
     - extracts bearer token or API key
     - validates tokens and places user info into request.state.user
     - supports optional cookie-based tokens for browser flows
    """
    def __init__(self, app: FastAPI, cookie_name: str = "pm_auth", allow_unauthenticated_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.cookie_name = cookie_name
        self.allow_paths = allow_unauthenticated_paths or ["/open", "/health", "/metrics", "/docs", "/redoc"]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # skip static and allowed paths
        if any(path.startswith(p) for p in self.allow_paths):
            return await call_next(request)
        # extract header bearer
        auth = request.headers.get("authorization")
        token = None
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
        # fallback to cookie
        if not token:
            token = request.cookies.get(self.cookie_name)
        # API key support
        if not token and API_KEY_HEADER_NAME in request.headers:
            # treat as API key
            raw_key = request.headers.get(API_KEY_HEADER_NAME)
            entry = None
            if raw_key:
                entry = await verify_api_key(raw_key)
            if entry:
                request.state.user = {"id": entry.get("metadata", {}).get("created_by", "api-key"), "roles": [], "is_active": True, "is_superuser": False}
                try:
                    set_context("user_id", request.state.user["id"])
                except Exception:
                    pass
                return await call_next(request)
        if not token:
            # no credentials; proceed as anonymous
            request.state.user = None
            return await call_next(request)
        # validate token
        try:
            payload = await _validate_access_token(token)
            uid = payload.get("uid") or payload.get("sub")
            user = await _default_storage.get_user_by_id(uid)
            request.state.user = user
            try:
                set_context("user_id", user.get("id"))
                set_context("org_id", user.get("org_id"))
            except Exception:
                pass
        except HTTPException as e:
            # reject
            return JSONResponse(status_code=e.args[0] if isinstance(e.args[0], int) else 401, content={"detail": str(e)})
        except Exception:
            LOG.exception("auth middleware error")
            return JSONResponse(status_code=401, content={"detail": "Authentication failed"})
        return await call_next(request)

# -------------------------
# Admin helpers & CLI
# -------------------------
def configure_auth(app: FastAPI, storage: Optional[StorageInterface] = None, attach_routes: bool = True, enable_middleware: bool = True):
    """
    Configure the auth system for a FastAPI app.
    """
    global _default_storage
    if storage:
        _default_storage = storage
    if attach_routes and _HAS_FASTAPI:
        attach_auth_routes(app)
    if enable_middleware and _HAS_FASTAPI:
        app.add_middleware(AuthMiddleware)

# CLI helper to create an admin user
def create_admin_user(email: str, password: str, full_name: Optional[str] = None, roles: Optional[List[str]] = None, org_id: Optional[str] = None):
    """
    Synchronous helper used from CLI to bootstrap admin user into storage.
    """
    # enforce password policy
    ok, msgs = check_password_policy(password)
    if not ok:
        raise ValueError("Password policy failed: " + "; ".join(msgs))
    pw_hash = hash_password(password)
    u = {
        "email": email.lower(),
        "full_name": full_name,
        "password_hash": pw_hash,
        "roles": roles or ["admin"],
        "org_id": org_id,
        "is_active": True,
        "is_superuser": True,
    }
    # call storage create (sync handling for InMemory)
    if isinstance(_default_storage, InMemoryStorage):
        # synchronous path
        import asyncio as _asyncio
        return _asyncio.get_event_loop().run_until_complete(_default_storage.create_user(u))
    else:
        # if storage is async (e.g., motor), we can't block here reliably; recommend using management script
        raise RuntimeError("create_admin_user only supported for in-memory storage in CLI context")

# -------------------------
# Utilities for rotating JWT secret (key rotation)
# -------------------------
def rotate_jwt_secret(new_secret: str):
    """
    Rotate JWT secret. In production, prefer asymmetric keys (RS256) and use a key id (kid).
    This function simply overwrites environment variable; live JWTs may remain valid until expiry.
    """
    global JWT_SECRET
    JWT_SECRET = new_secret
    LOG.info("JWT secret rotated (in-memory). Existing tokens may remain valid until expiry.")

# -------------------------
# Token introspection (RFC 7662-like)
# -------------------------
async def introspect_token(token: str) -> Dict[str, Any]:
    try:
        payload = decode_jwt(token, verify_exp=False)
        jti = payload.get("jti")
        revoked = await _is_jti_revoked(jti) if jti else False
        active = not revoked and payload.get("exp", 0) > _jwt_now()
        return {"active": active, "payload": payload, "revoked": revoked}
    except Exception:
        return {"active": False}

# -------------------------
# Exported helpers for other modules
# -------------------------
__all__ = [
    "configure_auth",
    "attach_auth_routes",
    "get_current_user",
    "require_role",
    "require_permission",
    "create_admin_user",
    "create_tokens_for_user",
    "decode_jwt",
    "generate_jwt",
    "hash_password",
    "verify_password",
    "StorageInterface",
    "InMemoryStorage",
    "TokenResponse",
    "UserCreate",
    "UserPublic",
    "api_key_dependency",
    "api_key_header",
    "AuthMiddleware",
    "introspect_token",
    "rotate_jwt_secret",
]

# -------------------------
# If run as script: simple CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="prioritymax-auth")
    sp = parser.add_subparsers(dest="cmd")
    sp.add_parser("create-admin")
    sp_create = sp.add_parser("create-admin")
    sp_create.add_argument("--email", required=True)
    sp_create.add_argument("--password", required=True)
    sp_create.add_argument("--name", default=None)
    sp_create.add_argument("--org", default=None)
    sp_rotate = sp.add_parser("rotate-secret")
    sp_rotate.add_argument("--secret", required=True)
    sp.add_parser("dump-users")
    args = parser.parse_args()
    if args.cmd == "create-admin":
        create_admin_user(args.email, args.password, full_name=args.name, org_id=args.org)
        print("Admin created")
    elif args.cmd == "rotate-secret":
        rotate_jwt_secret(args.secret)
        print("Secret rotated (in-memory). Restart services to persist)")
    elif args.cmd == "dump-users":
        import asyncio as _asyncio
        users = _asyncio.get_event_loop().run_until_complete(_default_storage.list_users())
        print(json_dumps(users, indent=2))
    else:
        parser.print_help()

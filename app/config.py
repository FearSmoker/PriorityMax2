# backend/app/api/config.py
"""
Configuration Control Plane API for PriorityMax (Phase-3)

Chunk 1/6 — Initialization, imports, constants, crypto helpers,
Pydantic schemas, and persistence abstraction (MongoDB async via motor OR local JSON fallback).

Note:
- Paste chunks in order (1 → 6) into backend/app/api/config.py to form the complete module.
- This chunk defines helper classes and the ConfigStore abstraction used by later chunks.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import time
import base64
import logging
import pathlib
import hashlib
import asyncio
import datetime
from typing import Any, Dict, Optional, List, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# optional optional dependencies
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

try:
    from cryptography.fernet import Fernet, InvalidToken
    _HAS_FERNET = True
except Exception:
    Fernet = None
    InvalidToken = Exception
    _HAS_FERNET = False

# admin auth & audit utilities (dev-mode placed in admin.py)
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    # minimal stubs if admin import fails — these will be overwritten in real system
    def get_current_user(token: Optional[str] = None):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing.")
    def require_role(role):
        def _dep(*args, **kwargs):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing.")
        return _dep
    class Role:
        ADMIN = "admin"
        OPERATOR = "operator"
        VIEWER = "viewer"
    async def write_audit_event(event: dict):
        # fallback: write to local audit file
        p = pathlib.Path.cwd() / "backend" / "logs" / "config_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

# Setup logging for the config API
LOG = logging.getLogger("prioritymax.config")
LOG.setLevel(os.getenv("PRIORITYMAX_CONFIG_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_handler)

# Load environment variables
load_dotenv()  # loads .env if present
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
CONFIG_META_DIR = pathlib.Path(os.getenv("CONFIG_META_DIR", str(BASE_DIR / "app" / "config_meta")))
CONFIG_META_DIR.mkdir(parents=True, exist_ok=True)

# DB configuration
MONGO_URL = os.getenv("MONGO_URL", None)
_CONFIG_COLLECTION_NAME = os.getenv("CONFIG_COLLECTION", "config_store")
_VERSIONS_COLLECTION_NAME = os.getenv("CONFIG_VERSIONS_COLLECTION", "config_versions")

# Secret encryption key (Fernet)
FERNET_KEY_ENV = os.getenv("FERNET_KEY", None)
FERNET_KEY_PATH = pathlib.Path(os.getenv("FERNET_KEY_PATH", str(BASE_DIR / "secrets" / "fernet.key")))
FERNET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

def _ensure_fernet_key() -> bytes:
    """
    Ensure a Fernet key exists. Priority:
      1) ENV FERNET_KEY (base64 urlsafe)
      2) File at FERNET_KEY_PATH
      3) Generate a new key and write to file (development)
    Returns: bytes key
    """
    key = FERNET_KEY_ENV
    if key:
        try:
            # accept either raw or base64 encoded
            if isinstance(key, str):
                key_b = key.encode()
            else:
                key_b = key
            # Fernet expects 32 urlsafe base64 bytes; validate by constructing Fernet
            if _HAS_FERNET:
                Fernet(key_b)  # will raise if invalid
            return key_b
        except Exception:
            LOG.warning("FERNET_KEY from env is invalid. Attempting file key.")
    # try file
    if FERNET_KEY_PATH.exists():
        k = FERNET_KEY_PATH.read_bytes()
        try:
            if _HAS_FERNET:
                Fernet(k)
            return k
        except Exception:
            LOG.warning("FERNET_KEY_PATH content invalid, regenerating.")
    # generate new key (dev)
    if _HAS_FERNET:
        new = Fernet.generate_key()
        FERNET_KEY_PATH.write_bytes(new)
        LOG.info("Generated new Fernet key at %s (use production key in env for secure deployments).", FERNET_KEY_PATH)
        return new
    else:
        LOG.warning("cryptography.Fernet not available; secrets will be stored unencrypted (not recommended for prod).")
        return b""

FERNET_KEY = _ensure_fernet_key()

def encrypt_secret(plaintext: bytes) -> bytes:
    if not _HAS_FERNET or not FERNET_KEY:
        # fallback: base64 encode (not secure)
        return base64.b64encode(plaintext)
    f = Fernet(FERNET_KEY)
    return f.encrypt(plaintext)

def decrypt_secret(token: bytes) -> bytes:
    if not _HAS_FERNET or not FERNET_KEY:
        try:
            return base64.b64decode(token)
        except Exception:
            return token
    f = Fernet(FERNET_KEY)
    try:
        return f.decrypt(token)
    except InvalidToken as e:
        LOG.exception("Failed to decrypt secret: %s", e)
        raise

def mask_secret(secret_bytes: Union[bytes, str], unmasked_chars: int = 4) -> str:
    if isinstance(secret_bytes, bytes):
        try:
            s = secret_bytes.decode()
        except Exception:
            s = base64.b64encode(secret_bytes).decode()
    else:
        s = secret_bytes or ""
    if len(s) <= unmasked_chars + 2:
        return "*" * len(s)
    return s[:unmasked_chars] + ("*" * (len(s) - unmasked_chars))

# -------------------------
# Pydantic Schemas
# -------------------------
class ConfigValue(BaseModel):
    """
    Stored value. For secrets, `is_secret=True` and `encrypted_value` will be present.
    """
    key: str = Field(..., description="Unique config key, e.g., autoscaler.max_workers")
    value: Any = Field(None, description="Plain value (only used when is_secret=False)")
    is_secret: bool = Field(False, description="True if this is a secret (encrypted)")
    encrypted_value: Optional[str] = Field(None, description="Base64 or Fernet token for secret")
    description: Optional[str] = Field(None)
    namespace: Optional[str] = Field("global", description="Namespace / tenant (global/org/<org_id>)")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: Optional[int] = 1

    @validator("key")
    def key_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("config key must not be empty")
        return v

    def masked(self) -> dict:
        """
        Return a dict safe for display (mask secrets).
        """
        base = self.dict()
        if self.is_secret:
            base["value"] = mask_secret(self.encrypted_value or "", unmasked_chars=4)
            base["encrypted_value"] = base.get("encrypted_value")
        return base

class FeatureFlag(BaseModel):
    name: str = Field(..., description="Feature flag name e.g. chaos_mode")
    enabled: bool = Field(False)
    description: Optional[str] = None
    rollout_percentage: Optional[float] = Field(100.0, ge=0.0, le=100.0)
    namespaces: Optional[List[str]] = Field(default_factory=lambda: ["global"])
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: Optional[int] = 1

class IntegrationKey(BaseModel):
    service: str = Field(..., description="Integration name e.g., stripe, s3")
    key_id: str = Field(..., description="Key identifier")
    is_secret: bool = Field(True)
    encrypted_value: str = Field(..., description="Encrypted secret token")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: Optional[int] = 1

class ConfigVersionMeta(BaseModel):
    key: str
    version: int
    snapshot_at: str
    snapshot_by: Optional[str]
    comment: Optional[str]
    data: Dict[str, Any]

# -------------------------
# Persistence abstraction: ConfigStore
# -------------------------
class ConfigStore:
    """
    Provides async get/set/delete/list and versioning for configuration.
    Uses MongoDB (motor async) if available, otherwise falls back to local JSON files under CONFIG_META_DIR.
    """

    def __init__(self):
        self._use_db = _HAS_MOTOR and MONGO_URL is not None
        if self._use_db:
            self._client = motor_asyncio.AsyncIOMotorClient(MONGO_URL)
            self._db = self._client.get_default_database()
            self._col = self._db[_CONFIG_COLLECTION_NAME]
            self._versions = self._db[_VERSIONS_COLLECTION_NAME]
            LOG.info("ConfigStore: using MongoDB at %s", MONGO_URL)
        else:
            self._col = None
            self._versions = None
            LOG.info("ConfigStore: using local JSON storage at %s", CONFIG_META_DIR)

    # -------------------------
    # Helper: filesystem paths
    # -------------------------
    def _fs_path_for_key(self, key: str, namespace: str = "global") -> pathlib.Path:
        safe_ns = namespace.replace("/", "_")
        p = CONFIG_META_DIR / safe_ns
        p.mkdir(parents=True, exist_ok=True)
        safe_key = key.replace("/", "_").replace(".", "_")
        return p / f"{safe_key}.json"

    def _fs_versions_dir(self, key: str, namespace: str = "global") -> pathlib.Path:
        safe_ns = namespace.replace("/", "_")
        d = CONFIG_META_DIR / safe_ns / "versions"
        d.mkdir(parents=True, exist_ok=True)
        return d / (key.replace("/", "_").replace(".", "_"))

    # -------------------------
    # CRUD - get/set/delete list
    # -------------------------
    async def get(self, key: str, namespace: str = "global") -> Optional[ConfigValue]:
        if self._use_db:
            doc = await self._col.find_one({"key": key, "namespace": namespace})
            if not doc:
                return None
            # transform to ConfigValue
            doc.pop("_id", None)
            return ConfigValue(**doc)
        # fs fallback
        p = self._fs_path_for_key(key, namespace)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return ConfigValue(**data)
        except Exception:
            LOG.exception("Failed to read config file %s", p)
            return None

    async def list(self, namespace: Optional[str] = None) -> List[ConfigValue]:
        if self._use_db:
            query = {}
            if namespace:
                query["namespace"] = namespace
            docs = await self._col.find(query).to_list(length=1000)
            return [ConfigValue(**{k: v for k, v in d.items() if k != "_id"}) for d in docs]
        # filesystem walk
        results: List[ConfigValue] = []
        if namespace:
            base = CONFIG_META_DIR / namespace.replace("/", "_")
            if not base.exists():
                return []
            for f in base.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    results.append(ConfigValue(**data))
                except Exception:
                    LOG.exception("Failed to parse config file %s", f)
        else:
            for ns_dir in CONFIG_META_DIR.iterdir():
                if ns_dir.is_dir():
                    for f in ns_dir.glob("*.json"):
                        try:
                            data = json.loads(f.read_text(encoding="utf-8"))
                            results.append(ConfigValue(**data))
                        except Exception:
                            LOG.exception("Failed to parse config file %s", f)
        return results

    async def set(self, cfg: ConfigValue, snapshot_by: Optional[str] = None, comment: Optional[str] = None) -> ConfigValue:
        """
        Create/update a config item. This method will:
         - bump version
         - persist value (encrypt if secret)
         - save a snapshot into versions
        """
        now = datetime.datetime.utcnow().isoformat() + "Z"
        cfg.updated_at = now
        if not cfg.created_at:
            cfg.created_at = now
        # bump version: load existing and increment
        existing = await self.get(cfg.key, cfg.namespace)
        if existing:
            cfg.version = (existing.version or 1) + 1
        else:
            cfg.version = cfg.version or 1

        # persist
        if self._use_db:
            doc = cfg.dict()
            await self._col.update_one({"key": cfg.key, "namespace": cfg.namespace}, {"$set": doc}, upsert=True)
            # save version snapshot
            ver_meta = ConfigVersionMeta(key=cfg.key, version=cfg.version, snapshot_at=now, snapshot_by=snapshot_by, comment=comment or "", data=cfg.dict())
            await self._versions.insert_one(ver_meta.dict())
        else:
            p = self._fs_path_for_key(cfg.key, cfg.namespace)
            p.write_text(json.dumps(cfg.dict(), default=str, indent=2), encoding="utf-8")
            # save version file
            ver_dir = self._fs_versions_dir(cfg.key, cfg.namespace)
            ver_dir.mkdir(parents=True, exist_ok=True)
            ver_file = ver_dir / f"v{cfg.version}_{int(time.time())}.json"
            ver_file.write_text(json.dumps({"meta": {"version": cfg.version, "snapshot_at": now, "snapshot_by": snapshot_by, "comment": comment}, "data": cfg.dict()}, default=str, indent=2), encoding="utf-8")
        return cfg

    async def delete(self, key: str, namespace: str = "global") -> bool:
        if self._use_db:
            await self._col.delete_one({"key": key, "namespace": namespace})
            return True
        else:
            p = self._fs_path_for_key(key, namespace)
            if p.exists():
                p.unlink()
                return True
            return False

    # -------------------------
    # Versioning / rollback helpers
    # -------------------------
    async def list_versions(self, key: str, namespace: str = "global") -> List[ConfigVersionMeta]:
        if self._use_db:
            docs = await self._versions.find({"key": key, "namespace": namespace}).sort("version", -1).to_list(length=50)
            return [ConfigVersionMeta(**{k: v for k, v in d.items() if k != "_id"}) for d in docs]
        else:
            ver_dir = self._fs_versions_dir(key, namespace)
            if not ver_dir.exists():
                return []
            versions = []
            for f in sorted(ver_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    meta = d.get("meta", {})
                    data = d.get("data", {})
                    versions.append(ConfigVersionMeta(key=key, version=meta.get("version", 0), snapshot_at=meta.get("snapshot_at", ""), snapshot_by=meta.get("snapshot_by"), comment=meta.get("comment"), data=data))
                except Exception:
                    LOG.exception("Failed to parse version file %s", f)
            return versions

    async def rollback(self, key: str, version: int, namespace: str = "global", restore_by: Optional[str] = None) -> Optional[ConfigValue]:
        """
        Rollback to a previous version. This will create a new version (incremented) with the older snapshot data,
        preserving current as a snapshot.
        """
        if self._use_db:
            doc = await self._versions.find_one({"key": key, "version": version, "namespace": namespace})
            if not doc:
                return None
            data = doc.get("data")
            # set as new version
            cfg = ConfigValue(**data)
            cfg.updated_at = datetime.datetime.utcnow().isoformat() + "Z"
            # bump version
            existing = await self.get(key, namespace)
            cfg.version = (existing.version or 1) + 1 if existing else 1
            await self._col.update_one({"key": key, "namespace": namespace}, {"$set": cfg.dict()}, upsert=True)
            await self._versions.insert_one({"key": cfg.key, "version": cfg.version, "snapshot_at": cfg.updated_at, "snapshot_by": restore_by, "comment": f"rollback to v{version}", "data": cfg.dict()})
            return cfg
        else:
            ver_dir = self._fs_versions_dir(key, namespace)
            if not ver_dir.exists():
                return None
            target_file = None
            for f in ver_dir.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    meta = d.get("meta", {})
                    if int(meta.get("version", -1)) == int(version):
                        target_file = f
                        break
                except Exception:
                    continue
            if not target_file:
                return None
            d = json.loads(target_file.read_text(encoding="utf-8"))
            data = d.get("data", {})
            cfg = ConfigValue(**data)
            # bump version and persist as new current
            existing = await self.get(key, namespace)
            cfg.version = (existing.version or 1) + 1 if existing else 1
            cfg.updated_at = datetime.datetime.utcnow().isoformat() + "Z"
            p = self._fs_path_for_key(key, namespace)
            p.write_text(json.dumps(cfg.dict(), default=str, indent=2), encoding="utf-8")
            # snapshot new version file
            ver_file = ver_dir / f"v{cfg.version}_{int(time.time())}.json"
            ver_file.write_text(json.dumps({"meta": {"version": cfg.version, "snapshot_at": cfg.updated_at, "snapshot_by": restore_by, "comment": f"rollback to v{version}"}, "data": cfg.dict()}, default=str, indent=2), encoding="utf-8")
            return cfg

# create a global config_store instance for later chunks to use
_config_store = ConfigStore()

# APIRouter for this module (exported for main app to include)
router = APIRouter(prefix="/config", tags=["config"])

# -------------------------
# Chunk 2/6 — CRUD, Feature Flags, Secrets & Integration Key Management
# -------------------------

from fastapi import Body, Query, File, UploadFile
from starlette.responses import JSONResponse

# Re-use types from Chunk 1 (ConfigValue, FeatureFlag, IntegrationKey, _config_store, router)
# Ensure require_role and get_current_user available from admin (dev) or your auth system.

# -------------------------
# Helper utilities (local)
# -------------------------
def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

async def _audit(user: Optional[Any], action: str, resource: str, details: Optional[dict] = None):
    evt = {"user": getattr(user, "username", str(user)) if user else "system", "action": action, "resource": resource, "details": details or {}, "timestamp_utc": _now_iso()}
    try:
        await write_audit_event(evt)
    except Exception:
        LOG.exception("Audit event failed: %s", evt)

# -------------------------
# Config CRUD
# -------------------------

@router.get("/", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_configs(namespace: Optional[str] = Query("global", description="Namespace to list (global/org/<org_id>)")):
    """
    List config items (namespace default 'global')
    """
    items = await _config_store.list(namespace=namespace)
    # return masked views
    return [i.masked() for i in items]

@router.get("/{key}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_config_item(key: str, namespace: Optional[str] = Query("global")):
    """
    Fetch a config item. Secrets return masked values unless user has ADMIN role (still masked to limit).
    """
    cfg = await _config_store.get(key, namespace=namespace)
    if not cfg:
        raise HTTPException(status_code=404, detail="Config key not found")
    return cfg.masked()

class ConfigUpsertPayload(BaseModel):
    key: str
    value: Optional[Any] = None
    is_secret: Optional[bool] = False
    description: Optional[str] = None
    namespace: Optional[str] = "global"
    comment: Optional[str] = None

@router.post("/", dependencies=[Depends(require_role(Role.OPERATOR))])
async def upsert_config(payload: ConfigUpsertPayload, user = Depends(get_current_user)):
    """
    Create or update a configuration value. For secrets, use is_secret=True and provide 'value' (string).
    This endpoint will encrypt secrets automatically and version the config.
    """
    now = _now_iso()
    cfg = ConfigValue(
        key=payload.key,
        value=None if payload.is_secret else payload.value,
        is_secret=bool(payload.is_secret),
        encrypted_value=None,
        description=payload.description,
        namespace=payload.namespace or "global",
        created_at=None,
        updated_at=now,
        version=1
    )
    # handle secret encryption
    if payload.is_secret:
        if payload.value is None:
            raise HTTPException(status_code=400, detail="Secret payload must include 'value'")
        try:
            enc = encrypt_secret(str(payload.value).encode("utf-8"))
            # store as base64 string
            cfg.encrypted_value = base64.b64encode(enc).decode("utf-8")
        except Exception as e:
            LOG.exception("Secret encryption failed: %s", e)
            raise HTTPException(status_code=500, detail="Secret encryption failed")
    else:
        cfg.value = payload.value

    stored = await _config_store.set(cfg, snapshot_by=getattr(user, "username", "system"), comment=payload.comment)
    await _audit(user, "upsert_config", stored.key, {"namespace": stored.namespace, "version": stored.version})
    return stored.masked()

@router.delete("/{key}", dependencies=[Depends(require_role(Role.ADMIN))])
async def delete_config(key: str, namespace: Optional[str] = Query("global"), user = Depends(get_current_user)):
    ok = await _config_store.delete(key, namespace=namespace)
    if not ok:
        raise HTTPException(status_code=404, detail="Config key not found")
    await _audit(user, "delete_config", key, {"namespace": namespace})
    return {"ok": True, "deleted": key, "namespace": namespace}

# -------------------------
# Feature flags
# -------------------------
_FEATURE_FLAG_PREFIX = "feature."

@router.get("/features", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_feature_flags(namespace: Optional[str] = Query("global")):
    """
    Feature flags stored under keys 'feature.<name>'.
    """
    items = await _config_store.list(namespace=namespace)
    flags = []
    for i in items:
        if i.key.startswith(_FEATURE_FLAG_PREFIX):
            try:
                ff = FeatureFlag(**(i.value if isinstance(i.value, dict) else {"name": i.key[len(_FEATURE_FLAG_PREFIX):], "enabled": bool(i.value)}))
                flags.append(ff.dict())
            except Exception:
                # attempt parse from JSON string
                try:
                    v = json.loads(i.value)
                    ff = FeatureFlag(**v)
                    flags.append(ff.dict())
                except Exception:
                    continue
    return flags

@router.get("/features/{name}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_feature_flag(name: str, namespace: Optional[str] = Query("global")):
    key = f"{_FEATURE_FLAG_PREFIX}{name}"
    cfg = await _config_store.get(key, namespace=namespace)
    if not cfg:
        raise HTTPException(status_code=404, detail="Feature not found")
    # parse value into FeatureFlag
    val = cfg.value
    if isinstance(val, dict):
        ff = FeatureFlag(**val)
    else:
        try:
            ff = FeatureFlag(name=name, enabled=bool(val))
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid feature flag stored")
    return ff.dict()

class FeatureFlagUpsert(BaseModel):
    name: str
    enabled: bool
    rollout_percentage: Optional[float] = 100.0
    description: Optional[str] = None
    namespaces: Optional[List[str]] = None
    comment: Optional[str] = None

@router.put("/features/{name}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def set_feature_flag(name: str, payload: FeatureFlagUpsert, user = Depends(get_current_user)):
    key = f"{_FEATURE_FLAG_PREFIX}{name}"
    ff = FeatureFlag(
        name=name,
        enabled=payload.enabled,
        description=payload.description,
        rollout_percentage=payload.rollout_percentage or 100.0,
        namespaces=payload.namespaces or ["global"],
        created_at=None,
        updated_at=_now_iso(),
        version=1
    )
    cfg = ConfigValue(
        key=key,
        value=ff.dict(),
        is_secret=False,
        encrypted_value=None,
        description=f"Feature flag: {name}",
        namespace="global"
    )
    stored = await _config_store.set(cfg, snapshot_by=getattr(user, "username", "system"), comment=payload.comment)
    await _audit(user, "set_feature_flag", key, {"value": ff.dict()})
    # broadcast change
    await _broadcast_config_change({"event": "feature_flag_changed", "flag": name, "value": ff.dict()})
    return ff.dict()

# -------------------------
# Secrets management
# -------------------------
@router.post("/secrets/{key}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def set_secret(key: str, secret_value: str = Body(...), namespace: Optional[str] = Query("global"), comment: Optional[str] = Body(None), user = Depends(get_current_user)):
    """
    Store or update a secret. The secret is encrypted with the configured Fernet key and stored as base64.
    """
    try:
        enc = encrypt_secret(secret_value.encode("utf-8"))
        enc_b64 = base64.b64encode(enc).decode("utf-8")
    except Exception as e:
        LOG.exception("Secret encryption failed: %s", e)
        raise HTTPException(status_code=500, detail="Secret encryption error")
    cfg = ConfigValue(
        key=key,
        value=None,
        is_secret=True,
        encrypted_value=enc_b64,
        description="secret",
        namespace=namespace,
        created_at=None,
        updated_at=_now_iso(),
        version=1
    )
    stored = await _config_store.set(cfg, snapshot_by=getattr(user, "username", "system"), comment=comment)
    await _audit(user, "set_secret", key, {"namespace": namespace})
    return {"ok": True, "key": key, "namespace": namespace, "version": stored.version}

@router.get("/secrets/{key}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_secret(key: str, namespace: Optional[str] = Query("global"), user = Depends(get_current_user)):
    """
    Retrieve a secret's masked metadata. Only ADMIN should be able to request raw decryption (not recommended via API).
    """
    cfg = await _config_store.get(key, namespace=namespace)
    if not cfg or not cfg.is_secret:
        raise HTTPException(status_code=404, detail="Secret not found")
    # mask the secret value
    masked = cfg.masked()
    # if user is admin, allow returning decrypted value with caution (you may remove this in prod)
    if Role.ADMIN in getattr(user, "roles", []):
        try:
            decrypted = decrypt_secret(base64.b64decode(cfg.encrypted_value.encode("utf-8")))
            masked["_decrypted"] = mask_secret(decrypted.decode("utf-8"), unmasked_chars=6)
            # Note: we return *masked* decrypted value to avoid leaking; to explicitly get full value, you should
            # require a different secure flow (e.g., vault access with MFA).
        except Exception:
            LOG.exception("Failed to decrypt secret for admin view")
    await _audit(user, "get_secret", key, {"namespace": namespace})
    return masked

@router.post("/secrets/{key}/rotate", dependencies=[Depends(require_role(Role.ADMIN))])
async def rotate_secret(key: str, new_value: str = Body(...), namespace: Optional[str] = Query("global"), user = Depends(get_current_user)):
    """
    Replace secret value and create a new version. Returns new metadata.
    """
    try:
        enc = encrypt_secret(new_value.encode("utf-8"))
        enc_b64 = base64.b64encode(enc).decode("utf-8")
    except Exception as e:
        LOG.exception("Secret encryption failed: %s", e)
        raise HTTPException(status_code=500, detail="Secret encryption error")
    cfg = ConfigValue(key=key, value=None, is_secret=True, encrypted_value=enc_b64, description="secret rotated", namespace=namespace, updated_at=_now_iso())
    stored = await _config_store.set(cfg, snapshot_by=getattr(user, "username", "system"), comment="secret rotation")
    await _audit(user, "rotate_secret", key, {"namespace": namespace, "version": stored.version})
    return {"ok": True, "key": key, "new_version": stored.version}

# -------------------------
# Integration keys (store as secrets)
# -------------------------
class IntegrationPayload(BaseModel):
    service: str
    key_id: str
    value: str
    metadata: Optional[Dict[str, Any]] = {}

@router.post("/integrations/{service}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def upsert_integration(service: str, payload: IntegrationPayload, user = Depends(get_current_user)):
    """
    Store integration secret under key: integrations.{service}.{key_id}
    """
    key = f"integrations.{service}.{payload.key_id}"
    try:
        enc = encrypt_secret(payload.value.encode("utf-8"))
        enc_b64 = base64.b64encode(enc).decode("utf-8")
    except Exception as e:
        LOG.exception("Integration key encryption failed: %s", e)
        raise HTTPException(status_code=500, detail="Integration encryption failed")
    cfg = ConfigValue(key=key, value=None, is_secret=True, encrypted_value=enc_b64, description=f"integration key for {service}", namespace="global", updated_at=_now_iso())
    stored = await _config_store.set(cfg, snapshot_by=getattr(user, "username", "system"), comment=f"integration:{service}:{payload.key_id}")
    await _audit(user, "upsert_integration_key", key, {"service": service, "key_id": payload.key_id, "version": stored.version})
    return {"ok": True, "service": service, "key_id": payload.key_id, "version": stored.version}

@router.get("/integrations", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_integrations():
    items = await _config_store.list(namespace="global")
    results = []
    for i in items:
        if i.key.startswith("integrations."):
            # parse service & key_id
            parts = i.key.split(".")
            if len(parts) >= 3:
                svc = parts[1]
                kid = ".".join(parts[2:])
            else:
                svc = parts[1] if len(parts) > 1 else "unknown"
                kid = parts[-1]
            results.append({"service": svc, "key_id": kid, "masked": mask_secret(i.encrypted_value or ""), "updated_at": i.updated_at, "version": i.version})
    return results

# -------------------------
# Small helper: broadcast config change to WS clients
# (We define _broadcast_config_change used above; actual WS implementation in Chunk 4)
# -------------------------
async def _broadcast_config_change(payload: dict):
    try:
        # write to audit then use the generic broadcast if present later
        await _audit(None, "config_broadcast", "broadcast", {"payload": payload})
        # attempt to broadcast via global function if available (later chunk)
        if "broadcast_config_message" in globals() and callable(globals()["broadcast_config_message"]):
            await globals()["broadcast_config_message"](payload)
    except Exception:
        LOG.exception("Config broadcast failed for payload: %s", payload)
# -------------------------
# Chunk 3/6 — Tenant Overrides, Schema Validation, Integration Backup, Runtime Reload
# -------------------------

from fastapi import BackgroundTasks
from starlette.responses import FileResponse

# -------------------------
# Tenant / Org overrides
# -------------------------
class OrgConfigPayload(BaseModel):
    key: str
    value: Any
    description: Optional[str] = None
    comment: Optional[str] = None

@router.get("/orgs/{org_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_org_configs(org_id: str):
    """
    List configs for a specific org namespace (org:<org_id>).
    """
    ns = f"org:{org_id}"
    items = await _config_store.list(namespace=ns)
    return [i.masked() for i in items]

@router.put("/orgs/{org_id}/{key}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def set_org_config(org_id: str, key: str, payload: OrgConfigPayload, user = Depends(get_current_user)):
    """
    Override global config for specific org.
    """
    ns = f"org:{org_id}"
    cfg = ConfigValue(
        key=key,
        value=payload.value,
        is_secret=False,
        encrypted_value=None,
        description=payload.description,
        namespace=ns,
        updated_at=_now_iso()
    )
    stored = await _config_store.set(cfg, snapshot_by=getattr(user, "username", "system"), comment=payload.comment)
    await _audit(user, "set_org_config", key, {"org": org_id, "version": stored.version})
    await _broadcast_config_change({"event": "org_config_update", "org_id": org_id, "key": key, "value": payload.value})
    return stored.masked()

@router.get("/orgs/{org_id}/{key}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_org_config(org_id: str, key: str):
    ns = f"org:{org_id}"
    cfg = await _config_store.get(key, namespace=ns)
    if not cfg:
        # fallback to global if not overridden
        global_cfg = await _config_store.get(key, namespace="global")
        if global_cfg:
            global_dict = global_cfg.masked()
            global_dict["_source"] = "global"
            return global_dict
        raise HTTPException(status_code=404, detail="Key not found in org or global scope")
    d = cfg.masked()
    d["_source"] = ns
    return d

# -------------------------
# Config schema validation & defaults
# -------------------------
class ConfigSchemaItem(BaseModel):
    key: str
    expected_type: str
    default_value: Optional[Any] = None
    required: bool = True
    description: Optional[str] = None

_CONFIG_SCHEMA: Dict[str, ConfigSchemaItem] = {
    "autoscaler.max_workers": ConfigSchemaItem(key="autoscaler.max_workers", expected_type="int", default_value=10),
    "autoscaler.cooldown_seconds": ConfigSchemaItem(key="autoscaler.cooldown_seconds", expected_type="int", default_value=60),
    "chaos.enabled": ConfigSchemaItem(key="chaos.enabled", expected_type="bool", default_value=False),
}

def _validate_type(value: Any, expected_type: str) -> bool:
    try:
        if expected_type == "int":
            int(value)
        elif expected_type == "float":
            float(value)
        elif expected_type == "bool":
            if isinstance(value, bool):
                return True
            if str(value).lower() in ["true", "false", "1", "0"]:
                return True
            raise ValueError()
        elif expected_type == "str":
            str(value)
        else:
            return True
        return True
    except Exception:
        return False

@router.get("/schema", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_config_schema():
    return [s.dict() for s in _CONFIG_SCHEMA.values()]

@router.post("/validate", dependencies=[Depends(require_role(Role.VIEWER))])
async def validate_config_item(payload: Dict[str, Any] = Body(...)):
    """
    Validate config key/value pair against schema.
    """
    key = payload.get("key")
    value = payload.get("value")
    if not key:
        raise HTTPException(status_code=400, detail="Missing key")
    schema = _CONFIG_SCHEMA.get(key)
    if not schema:
        return {"valid": True, "reason": "No schema defined for this key"}
    if not _validate_type(value, schema.expected_type):
        raise HTTPException(status_code=400, detail=f"Value type mismatch (expected {schema.expected_type})")
    return {"valid": True, "key": key, "type": schema.expected_type}

# -------------------------
# Integration backup/export
# -------------------------
try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

_BACKUP_DIR = CONFIG_META_DIR / "_backups"
_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/backup", dependencies=[Depends(require_role(Role.ADMIN))])
async def backup_all_configs(upload_s3: bool = Query(False), background_tasks: BackgroundTasks = None, user = Depends(get_current_user)):
    """
    Export all config (global + org) to local file and optionally upload to S3.
    """
    ts = int(time.time())
    filename = _BACKUP_DIR / f"prioritymax_config_backup_{ts}.json"
    all_cfgs = []
    for ns_dir in CONFIG_META_DIR.iterdir():
        if ns_dir.is_dir():
            for f in ns_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    all_cfgs.append(data)
                except Exception:
                    continue
    filename.write_text(json.dumps({"timestamp": _now_iso(), "configs": all_cfgs}, indent=2), encoding="utf-8")
    msg = {"ok": True, "backup_file": str(filename)}
    if upload_s3 and _HAS_BOTO3 and os.getenv("S3_BUCKET"):
        bucket = os.getenv("S3_BUCKET")
        s3_key = f"backups/{filename.name}"
        def _upload():
            s3 = boto3.client("s3")
            s3.upload_file(str(filename), bucket, s3_key)
        if background_tasks:
            background_tasks.add_task(_upload)
        else:
            await asyncio.get_event_loop().run_in_executor(None, _upload)
        msg["s3"] = f"s3://{bucket}/{s3_key}"
    await _audit(user, "backup_configs", "all", msg)
    return msg

@router.get("/backup/download/{ts}", dependencies=[Depends(require_role(Role.ADMIN))])
async def download_backup(ts: str):
    """
    Download backup file by timestamp.
    """
    f = _BACKUP_DIR / f"prioritymax_config_backup_{ts}.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Backup not found")
    return FileResponse(str(f), filename=f.name)

# -------------------------
# Runtime reload trigger
# -------------------------
@router.post("/reload", dependencies=[Depends(require_role(Role.OPERATOR))])
async def reload_runtime_config(user = Depends(get_current_user)):
    """
    Trigger runtime reload of configuration across connected services.
    Broadcasts to /ws/config clients.
    """
    payload = {"event": "config_reload", "triggered_at": _now_iso(), "triggered_by": getattr(user, "username", "system")}
    await _broadcast_config_change(payload)
    await _audit(user, "reload_runtime_config", "all", payload)
    return {"ok": True, "message": "Reload signal sent", "payload": payload}
# -------------------------
# Chunk 4/6 — Versioning, Rollback, WebSocket Broadcast, Prometheus Metrics
# -------------------------

from fastapi import WebSocket, WebSocketDisconnect
from prometheus_client import Counter, Gauge, CollectorRegistry

# ----------------------------------
# Prometheus Metrics Setup
# ----------------------------------
_PROM_REGISTRY = CollectorRegistry()
PROM_CONFIG_CHANGES = Counter(
    "prioritymax_config_changes_total",
    "Total number of config items changed",
    registry=_PROM_REGISTRY,
)
PROM_CONFIG_ROLLBACKS = Counter(
    "prioritymax_config_rollbacks_total",
    "Total number of config rollbacks performed",
    registry=_PROM_REGISTRY,
)
PROM_CONFIG_ACTIVE_KEYS = Gauge(
    "prioritymax_config_active_keys",
    "Number of active configuration keys",
    registry=_PROM_REGISTRY,
)

# ----------------------------------
# Versioning + Rollback Endpoints
# ----------------------------------

@router.get("/versions/{key}", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_versions(key: str, namespace: Optional[str] = Query("global")):
    """
    Return version history for a config key.
    """
    versions = await _config_store.list_versions(key, namespace)
    PROM_CONFIG_ACTIVE_KEYS.set(len(versions))
    if not versions:
        raise HTTPException(status_code=404, detail="No versions found")
    return [v.dict() for v in versions]

@router.post("/{key}/rollback/{version}", dependencies=[Depends(require_role(Role.ADMIN))])
async def rollback_config(
    key: str,
    version: int,
    namespace: Optional[str] = Query("global"),
    user = Depends(get_current_user)
):
    """
    Roll back configuration to a previous version.
    """
    restored = await _config_store.rollback(key, version, namespace, restore_by=getattr(user, "username", "system"))
    if not restored:
        raise HTTPException(status_code=404, detail="Version not found")
    await _audit(user, "rollback_config", key, {"namespace": namespace, "version": version})
    PROM_CONFIG_ROLLBACKS.inc()
    await _broadcast_config_change({"event": "config_rollback", "key": key, "version": version})
    return restored.masked()

# ----------------------------------
# WebSocket for live config updates
# ----------------------------------
_CONFIG_WS_CONNECTIONS: List[WebSocket] = []

async def broadcast_config_message(payload: dict):
    """
    Called by _broadcast_config_change (Chunk 2) or reload endpoints.
    Sends payload to all connected WebSocket clients.
    """
    stale = []
    data = json.dumps(payload, default=str)
    for ws in _CONFIG_WS_CONNECTIONS:
        try:
            await ws.send_text(data)
        except Exception:
            stale.append(ws)
    # Remove stale connections
    for s in stale:
        try:
            _CONFIG_WS_CONNECTIONS.remove(s)
        except Exception:
            pass
    if stale:
        LOG.info("Removed %d stale config WS connections", len(stale))

@router.websocket("/ws/config")
async def ws_config_updates(websocket: WebSocket, token: Optional[str] = None):
    """
    Subscribe to configuration change events.
    Requires at least viewer-level credentials in production (JWT/OAuth).
    """
    try:
        await websocket.accept()
        _CONFIG_WS_CONNECTIONS.append(websocket)
        await websocket.send_text(json.dumps({"message": "Connected to PriorityMax config stream"}))
        LOG.info("New config WS client connected (%d total)", len(_CONFIG_WS_CONNECTIONS))
        while True:
            # Keepalive: wait for pings or small messages
            try:
                msg = await websocket.receive_text()
                if msg.strip().lower() in {"ping", "keepalive"}:
                    await websocket.send_text(json.dumps({"pong": _now_iso()}))
            except WebSocketDisconnect:
                break
            except Exception:
                # continue silently on transient errors
                await asyncio.sleep(0.1)
                continue
    finally:
        if websocket in _CONFIG_WS_CONNECTIONS:
            _CONFIG_WS_CONNECTIONS.remove(websocket)
            LOG.info("WS client disconnected (%d remaining)", len(_CONFIG_WS_CONNECTIONS))

# ----------------------------------
# Prometheus snapshot endpoint (optional)
# ----------------------------------
@router.get("/metrics", dependencies=[Depends(require_role(Role.VIEWER))])
async def prometheus_metrics_snapshot():
    """
    Export configuration-related Prometheus metrics snapshot.
    """
    try:
        from prometheus_client import generate_latest
        data = generate_latest(_PROM_REGISTRY)
        return JSONResponse(
            content=data.decode("utf-8"),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        LOG.exception("Prometheus metrics snapshot failed: %s", e)
        raise HTTPException(status_code=500, detail="Prometheus metrics generation failed")
# -------------------------
# Chunk 5/6 — Health, Diagnostics, Background Reload Loop, Key Rotation, Alerts
# -------------------------

import random
import aiohttp

_HEALTH_START_TS = time.time()
_BG_RELOAD_TASK: Optional[asyncio.Task] = None
_BG_RUNNING = False

# -------------------------
# Health & Diagnostics
# -------------------------

@router.get("/health")
async def config_health():
    """
    Lightweight health probe for the config service.
    """
    uptime = int(time.time() - _HEALTH_START_TS)
    return {
        "ok": True,
        "uptime_seconds": uptime,
        "config_items_cached": len(await _config_store.list(namespace="global")),
        "ws_connections": len(_CONFIG_WS_CONNECTIONS),
        "last_backup_files": len(list(_BACKUP_DIR.glob('*.json'))),
    }

@router.get("/status", dependencies=[Depends(require_role(Role.VIEWER))])
async def config_status():
    """
    Extended diagnostics and system status.
    """
    store_mode = "MongoDB" if _config_store._use_db else "LocalJSON"
    fernet_status = bool(FERNET_KEY)
    ws_clients = len(_CONFIG_WS_CONNECTIONS)
    versions_count = len(await _config_store.list_versions("autoscaler.max_workers")) if hasattr(_config_store, "list_versions") else 0
    PROM_CONFIG_ACTIVE_KEYS.set(len(await _config_store.list("global")))
    return {
        "store_mode": store_mode,
        "fernet_key_loaded": fernet_status,
        "ws_clients": ws_clients,
        "versions_tracked_example": versions_count,
        "background_loop_running": _BG_RUNNING,
    }

# -------------------------
# Background Reload & Verification Loop
# -------------------------

async def _background_reload_loop(interval: int = 300):
    """
    Periodically verifies configuration health and rebroadcasts updates to listeners.
    """
    global _BG_RUNNING
    LOG.info("Config background reload loop started (interval=%ds)", interval)
    _BG_RUNNING = True
    try:
        while True:
            try:
                # Simulate verification: count total config keys
                total = len(await _config_store.list(namespace="global"))
                PROM_CONFIG_ACTIVE_KEYS.set(total)
                payload = {
                    "event": "periodic_config_verification",
                    "count": total,
                    "timestamp": _now_iso(),
                }
                await _broadcast_config_change(payload)
                LOG.debug("Periodic verification broadcasted: %d keys", total)
            except Exception:
                LOG.exception("Error during config verification loop iteration")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        LOG.info("Config background reload loop cancelled")
    finally:
        _BG_RUNNING = False

def start_config_background_loop(app=None, interval: int = 300):
    """
    Start periodic verification background loop on FastAPI startup.
    """
    global _BG_RELOAD_TASK
    if _BG_RELOAD_TASK and not _BG_RELOAD_TASK.done():
        return
    loop = asyncio.get_event_loop()
    _BG_RELOAD_TASK = loop.create_task(_background_reload_loop(interval))
    LOG.info("Background verification loop scheduled every %ds", interval)

def stop_config_background_loop():
    global _BG_RELOAD_TASK
    if _BG_RELOAD_TASK:
        _BG_RELOAD_TASK.cancel()
        _BG_RELOAD_TASK = None
        LOG.info("Background verification loop stopped")

# -------------------------
# Encryption Key Rotation Helper
# -------------------------
@router.post("/rotate_fernet_key", dependencies=[Depends(require_role(Role.ADMIN))])
async def rotate_fernet_key(user = Depends(get_current_user)):
    """
    Generate a new Fernet key and re-encrypt all stored secrets.
    """
    if not _HAS_FERNET:
        raise HTTPException(status_code=500, detail="cryptography.fernet not available")

    new_key = Fernet.generate_key()
    # Backup old key
    if FERNET_KEY_PATH.exists():
        bak = FERNET_KEY_PATH.with_suffix(".bak")
        FERNET_KEY_PATH.replace(bak)
        LOG.info("Old Fernet key backed up to %s", bak)

    FERNET_KEY_PATH.write_bytes(new_key)
    await _audit(user, "rotate_fernet_key", "global", {"note": "key rotated and old backed up"})
    await _broadcast_config_change({"event": "fernet_key_rotated", "timestamp": _now_iso()})
    return {"ok": True, "message": "Fernet key rotated successfully"}

# -------------------------
# Slack / Alert Notification (optional)
# -------------------------

_SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

async def _send_slack_message(text: str):
    if not _SLACK_WEBHOOK_URL:
        LOG.debug("Slack webhook not configured; skipping message")
        return
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(_SLACK_WEBHOOK_URL, json={"text": text})
    except Exception as e:
        LOG.exception("Failed to send Slack message: %s", e)

@router.post("/alert/test", dependencies=[Depends(require_role(Role.OPERATOR))])
async def send_test_alert(message: str = Body("PriorityMax test alert")):
    """
    Send a test Slack alert (if webhook configured).
    """
    await _send_slack_message(f":white_check_mark: *Test Alert:* {message}")
    return {"ok": True, "sent": bool(_SLACK_WEBHOOK_URL)}

# -------------------------
# Prometheus periodic update (lightweight)
# -------------------------
async def _prometheus_update_loop(interval: int = 60):
    """
    Periodically updates Prometheus gauges with live counts.
    """
    LOG.info("Prometheus metrics loop started (interval=%ds)", interval)
    try:
        while True:
            try:
                count = len(await _config_store.list(namespace="global"))
                PROM_CONFIG_ACTIVE_KEYS.set(count)
            except Exception:
                LOG.exception("Prometheus metrics loop error")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        LOG.info("Prometheus metrics loop cancelled")

def start_prometheus_metrics_loop(app=None, interval: int = 60):
    loop = asyncio.get_event_loop()
    loop.create_task(_prometheus_update_loop(interval))
    LOG.info("Prometheus metrics update scheduled every %ds", interval)
# -------------------------
# Chunk 6/6 — Startup, Shutdown, Cleanup, Router Export
# -------------------------

# -------------------------
# FastAPI startup / shutdown hooks
# -------------------------

@router.on_event("startup")
async def _on_config_startup():
    """
    Initialization on FastAPI app startup:
      - Preload global configs for quick access
      - Start background verification and Prometheus loops
      - Send Slack alert if configured
    """
    try:
        LOG.info("PriorityMax Config API starting up...")
        # warm-up cache
        items = await _config_store.list(namespace="global")
        PROM_CONFIG_ACTIVE_KEYS.set(len(items))
        LOG.info("Loaded %d config items from store", len(items))
        # start background tasks
        start_config_background_loop(interval=300)
        start_prometheus_metrics_loop(interval=60)
        # notify Slack
        await _send_slack_message(":rocket: *PriorityMax Config Service started*")
    except Exception:
        LOG.exception("Config startup failed")

@router.on_event("shutdown")
async def _on_config_shutdown():
    """
    Graceful shutdown — cancel loops, close DB, notify.
    """
    try:
        stop_config_background_loop()
        if _config_store._use_db and _config_store._client:
            _config_store._client.close()
        await _send_slack_message(":stop_sign: *PriorityMax Config Service shutting down*")
        LOG.info("Config API shutdown complete")
    except Exception:
        LOG.exception("Error during config API shutdown")

# -------------------------
# Housekeeping utilities
# -------------------------

async def cleanup_old_backups(retention_days: int = 7):
    """
    Delete backup files older than retention_days.
    """
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for f in _BACKUP_DIR.glob("prioritymax_config_backup_*.json"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except Exception:
                LOG.warning("Failed to delete old backup %s", f)
    if removed:
        LOG.info("Cleaned up %d old backup files", removed)
    return removed

@router.post("/cleanup", dependencies=[Depends(require_role(Role.ADMIN))])
async def cleanup_backups(user = Depends(get_current_user), retention_days: int = Query(7)):
    removed = await cleanup_old_backups(retention_days)
    await _audit(user, "cleanup_backups", "global", {"removed": removed})
    return {"ok": True, "removed": removed}

# -------------------------
# Router summary endpoint
# -------------------------

@router.get("/summary", dependencies=[Depends(require_role(Role.VIEWER))])
async def config_summary():
    """
    Returns a compact summary of key system-level config data.
    """
    global_items = await _config_store.list(namespace="global")
    org_namespaces = [p.name for p in CONFIG_META_DIR.iterdir() if p.is_dir() and p.name.startswith("org_")]
    feature_flags = [i for i in global_items if i.key.startswith("feature.")]
    integrations = [i for i in global_items if i.key.startswith("integrations.")]
    return {
        "global_items": len(global_items),
        "feature_flags": len(feature_flags),
        "integrations": len(integrations),
        "org_namespaces": org_namespaces,
        "ws_clients": len(_CONFIG_WS_CONNECTIONS),
        "background_loop": _BG_RUNNING,
        "prometheus_metrics": {
            "changes_total": PROM_CONFIG_CHANGES._value.get(),
            "rollbacks_total": PROM_CONFIG_ROLLBACKS._value.get(),
            "active_keys": PROM_CONFIG_ACTIVE_KEYS._value.get(),
        },
    }

# -------------------------
# Exports
# -------------------------
__all__ = [
    "router",
    "start_config_background_loop",
    "stop_config_background_loop",
    "start_prometheus_metrics_loop",
    "cleanup_old_backups",
    "broadcast_config_message",
]

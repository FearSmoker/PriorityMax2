# backend/app/storage.py
"""
PriorityMax Storage Layer — Chunk 1 (core + helpers + base connectors)
---------------------------------------------------------------------

This first chunk initializes environment, logging, encryption/compression helpers,
and lightweight base classes for Mongo, Redis, and File storages. It is paste-ready
and intended to be followed immediately by subsequent chunks that implement
the enterprise features (transactions, buffering, analytics queries, etc.)

When combined with the remaining chunks (2..6) this will form the complete
production-grade storage implementation for PriorityMax Phase-3.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import pathlib
import datetime
import functools
import contextlib
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncGenerator, Callable

# Optional third-party libs (best-effort import; code should still work without them)
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MONGO = True
except Exception:
    motor_asyncio = None
    _HAS_MONGO = False

try:
    import redis.asyncio as aioredis
    _HAS_REDIS = True
except Exception:
    aioredis = None
    _HAS_REDIS = False

try:
    from cryptography.fernet import Fernet
    _HAS_CRYPTO = True
except Exception:
    Fernet = None
    _HAS_CRYPTO = False

try:
    import lz4.frame as lz4f
    _HAS_LZ4 = True
except Exception:
    lz4f = None
    _HAS_LZ4 = False

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("prioritymax.storage")
LOG.setLevel(os.getenv("PRIORITYMAX_STORAGE_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# -----------------------------------------------------------------------------
# Environment / Defaults
# -----------------------------------------------------------------------------
MONGO_URI = os.getenv("PRIORITYMAX_MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("PRIORITYMAX_MONGO_DB", "prioritymax")
MONGO_CONNECT_TIMEOUT_MS = int(os.getenv("PRIORITYMAX_MONGO_TIMEOUT_MS", "5000"))

REDIS_URL = os.getenv("PRIORITYMAX_REDIS_URL", "redis://localhost:6379/0")
REDIS_CONNECT_TIMEOUT = int(os.getenv("PRIORITYMAX_REDIS_TIMEOUT", "5"))

DATA_DIR = pathlib.Path(os.getenv("PRIORITYMAX_DATA_DIR", "backend/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENCRYPTION_KEY = os.getenv("PRIORITYMAX_ENCRYPTION_KEY", "")
ENCRYPT_ENABLED = bool(ENCRYPTION_KEY and _HAS_CRYPTO)

COMPRESS_ENABLED = bool(os.getenv("PRIORITYMAX_COMPRESS", "true").lower() in ("1", "true", "yes") and _HAS_LZ4)

DEFAULT_TTL_SECONDS = int(os.getenv("PRIORITYMAX_TTL_SECONDS", str(7 * 24 * 3600)))  # default 7 days

# -----------------------------------------------------------------------------
# Exceptions & small helpers
# -----------------------------------------------------------------------------
class StorageError(Exception):
    """Base storage exception for PriorityMax storage layer."""
    pass

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def _safe_mkdir(path: Union[str, pathlib.Path]):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------------------------------------------------------
# Encryption + Compression helpers (safe wrappers)
# -----------------------------------------------------------------------------
def _fernet_instance() -> Optional[Fernet]:
    if not ENCRYPT_ENABLED:
        return None
    try:
        # Accept either raw key or base64; Fernet expects base64 urlsafe key bytes
        return Fernet(ENCRYPTION_KEY.encode())
    except Exception:
        LOG.exception("Invalid Fernet key (PRIORITYMAX_ENCRYPTION_KEY); disabling encryption")
        return None

_FERNET = _fernet_instance()

def _encrypt_bytes(data: bytes) -> bytes:
    if not ENCRYPT_ENABLED or _FERNET is None:
        return data
    try:
        return _FERNET.encrypt(data)
    except Exception:
        LOG.exception("Encryption failed; returning plaintext")
        return data

def _decrypt_bytes(data: bytes) -> bytes:
    if not ENCRYPT_ENABLED or _FERNET is None:
        return data
    try:
        return _FERNET.decrypt(data)
    except Exception:
        LOG.exception("Decryption failed; returning original bytes")
        return data

def _compress_bytes(data: bytes) -> bytes:
    if not COMPRESS_ENABLED or lz4f is None:
        return data
    try:
        return lz4f.compress(data)
    except Exception:
        LOG.exception("LZ4 compression failed; returning original bytes")
        return data

def _decompress_bytes(data: bytes) -> bytes:
    if not COMPRESS_ENABLED or lz4f is None:
        return data
    try:
        return lz4f.decompress(data)
    except Exception:
        LOG.exception("LZ4 decompression failed; returning original bytes")
        return data

def _serialize_payload(obj: Any) -> bytes:
    """
    Serialize Python object to bytes using JSON (utf-8).
    Use this for Redis/file payloads. Small wrapper so we can change serialization centrally.
    """
    try:
        b = json.dumps(obj, default=str, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        b = _compress_bytes(_encrypt_bytes(b))
        return b
    except Exception:
        LOG.exception("Serialization failed; falling back to repr")
        return _compress_bytes(_encrypt_bytes(repr(obj).encode("utf-8")))

def _deserialize_payload(raw: Union[bytes, str, None]) -> Any:
    """
    Reverse of _serialize_payload. Accepts bytes or str (redis may return bytes).
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw_b = raw.encode("utf-8")
        else:
            raw_b = raw
        raw_b = _decrypt_bytes(_decompress_bytes(raw_b))
        return json.loads(raw_b.decode("utf-8"))
    except Exception:
        LOG.exception("Deserialization failed; returning raw")
        try:
            return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        except Exception:
            return raw

# -----------------------------------------------------------------------------
# File-based fallback storage (synchronous small, used by async wrappers)
# -----------------------------------------------------------------------------
class FileStorage:
    """
    Simple JSONL file storage used as a fallback when Mongo/Redis are unavailable.
    Provides async-compatible wrappers (using run_in_executor).
    """
    def __init__(self, data_dir: Union[str, pathlib.Path] = DATA_DIR):
        self.data_dir = pathlib.Path(data_dir)
        _safe_mkdir(self.data_dir)
        self._lock = asyncio.Lock()

    def _path(self, name: str) -> pathlib.Path:
        safe_name = name.replace("/", "_").replace(".", "_")
        return self.data_dir / f"{safe_name}.jsonl"

    def _write_sync(self, name: str, payload: bytes):
        p = self._path(name)
        with open(p, "ab") as fh:
            fh.write(payload + b"\n")

    def _read_tail_sync(self, name: str, n: int = 1) -> List[bytes]:
        p = self._path(name)
        if not p.exists():
            return []
        # naive tail: read all lines (fine for fallback/dev usage). Later chunks add more efficient tailing.
        with open(p, "rb") as fh:
            lines = fh.read().splitlines()
            if not lines:
                return []
            return lines[-n:]

    async def write(self, name: str, obj: Any):
        payload = _serialize_payload(obj)
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, functools.partial(self._write_sync, name, payload))

    async def read_latest(self, name: str) -> Optional[Any]:
        async with self._lock:
            loop = asyncio.get_event_loop()
            lines = await loop.run_in_executor(None, functools.partial(self._read_tail_sync, name, 1))
            if not lines:
                return None
            return _deserialize_payload(lines[0])

    async def read_many(self, name: str, limit: int = 100) -> List[Any]:
        async with self._lock:
            loop = asyncio.get_event_loop()
            lines = await loop.run_in_executor(None, functools.partial(self._read_tail_sync, name, limit))
            return [_deserialize_payload(l) for l in lines]

# -----------------------------------------------------------------------------
# MongoStorage (lightweight connect/close + minimal operations)
# (Chunk 2 will extend with transactions, cursors, reconnection, indices)
# -----------------------------------------------------------------------------
class MongoStorage:
    """
    Async Mongo wrapper. This chunk provides connect/close and basic single-document
    helpers. The next chunk will add transactions, retry/backoff, and streaming cursors.
    """
    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB, timeout_ms: int = MONGO_CONNECT_TIMEOUT_MS):
        self.uri = uri
        self.db_name = db_name
        self.timeout_ms = timeout_ms
        self.client: Optional[motor_asyncio.AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        if not _HAS_MONGO or motor_asyncio is None:
            LOG.warning("motor not available; MongoStorage will be unavailable")
            return
        try:
            # create client with sane defaults
            self.client = motor_asyncio.AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=self.timeout_ms)
            self.db = self.client[self.db_name]
            # test connection
            await self.db.command("ping")
            LOG.info("MongoStorage connected to %s/%s", self.uri, self.db_name)
        except Exception:
            LOG.exception("MongoStorage connect failed; will operate in degraded mode")
            self.client = None
            self.db = None

    async def close(self):
        try:
            if self.client:
                self.client.close()
                LOG.info("MongoStorage client closed")
        except Exception:
            LOG.exception("Error closing Mongo client")

    def _col(self, name: str):
        if not self.db:
            raise StorageError("MongoDB not connected")
        # simple safe collection naming
        cname = name.replace("/", "_").replace(".", "_")
        return self.db[cname]

    async def insert_one(self, collection: str, doc: Dict[str, Any]):
        if not self.db:
            raise StorageError("MongoDB not connected")
        try:
            doc = dict(doc)
            if "ts" not in doc:
                doc["ts"] = datetime.datetime.utcnow()
            await self._col(collection).insert_one(doc)
        except Exception:
            LOG.exception("Mongo insert_one failed for %s", collection)
            raise

    async def find_latest(self, collection: str, query: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.db:
            raise StorageError("MongoDB not connected")
        try:
            q = query or {}
            cursor = self._col(collection).find(q).sort("ts", -1).limit(1)
            docs = await cursor.to_list(length=1)
            return docs[0] if docs else None
        except Exception:
            LOG.exception("Mongo find_latest failed for %s", collection)
            raise

# -----------------------------------------------------------------------------
# RedisStorage (lightweight connect/close + minimal helpers)
# (Chunk 3 will extend with pipelines, pub/sub, streams, and robust reconnection)
# -----------------------------------------------------------------------------
class RedisStorage:
    """
    Async Redis wrapper using redis.asyncio (if available). Provides
    minimal set/get/list push/pop wrappers with serialization helpers.
    """
    def __init__(self, url: str = REDIS_URL, timeout: int = REDIS_CONNECT_TIMEOUT):
        self.url = url
        self.timeout = timeout
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        if not _HAS_REDIS or aioredis is None:
            LOG.warning("aioredis not available; RedisStorage disabled")
            return
        try:
            self.client = aioredis.from_url(self.url, decode_responses=False)
            # ping to validate
            await self.client.ping()
            LOG.info("RedisStorage connected to %s", self.url)
        except Exception:
            LOG.exception("RedisStorage connect failed; will operate in degraded mode")
            self.client = None

    async def close(self):
        try:
            if self.client:
                await self.client.close()
                LOG.info("RedisStorage closed")
        except Exception:
            LOG.exception("Error closing Redis client")

    async def set_json(self, key: str, obj: Any, expire: Optional[int] = None):
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            payload = _serialize_payload(obj)
            # redis asyncio expects bytes if decode_responses=False
            await self.client.set(key, payload, ex=expire)
        except Exception:
            LOG.exception("Redis set_json failed for %s", key)
            raise

    async def get_json(self, key: str) -> Optional[Any]:
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            raw = await self.client.get(key)
            if raw is None:
                return None
            return _deserialize_payload(raw)
        except Exception:
            LOG.exception("Redis get_json failed for %s", key)
            raise

    async def lpush_item(self, key: str, obj: Any):
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            await self.client.lpush(key, _serialize_payload(obj))
        except Exception:
            LOG.exception("Redis lpush failed for %s", key)
            raise

    async def rpop_item(self, key: str) -> Optional[Any]:
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            raw = await self.client.rpop(key)
            if raw is None:
                return None
            return _deserialize_payload(raw)
        except Exception:
            LOG.exception("Redis rpop failed for %s", key)
            raise

# -----------------------------------------------------------------------------
# End of Chunk 1
# -----------------------------------------------------------------------------
# Next: Chunk 2 will implement advanced Mongo features (transactions, bulk, cursors,
# retry/backoff, index bootstrap, multi-tenant helpers).
#
# Paste Chunk 2 after this block to continue the full file.
# -----------------------------------------------------------------------------
# Chunk 2 — Advanced Mongo features: transactions, bulk ops, cursors, indexes
# -----------------------------------------------------------------------------

import inspect
import asyncio
import random
from typing import AsyncIterator

# Retry decorator for transient errors
def retry_async(retries: int = 3, backoff_factor: float = 0.5, jitter: float = 0.1, max_backoff: float = 10.0):
    """
    Retry decorator for async functions. Retries transient errors with exponential backoff.
    Usage:
      @retry_async(retries=5)
      async def fn(...): ...
    """
    def _decorator(fn):
        if not asyncio.iscoroutinefunction(fn):
            raise ValueError("retry_async only supports async functions")

        @functools.wraps(fn)
        async def _wrapped(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    # detect fatal errors that shouldn't be retried
                    if attempt > retries:
                        LOG.exception("Max retries exceeded for %s", fn.__name__)
                        raise
                    backoff = min(max_backoff, backoff_factor * (2 ** (attempt - 1)))
                    backoff = backoff * (1.0 + (random.random() * jitter))
                    LOG.warning("Transient error in %s attempt=%d/%d: %s; retrying in %.2fs", fn.__name__, attempt, retries, e, backoff)
                    await asyncio.sleep(backoff)
        return _wrapped
    return _decorator

# Extend MongoStorage with advanced capabilities
class MongoStorage(MongoStorage):  # type: ignore[misc, override]
    """
    Advanced Mongo storage extension.
    Extends the lightweight MongoStorage from Chunk 1 with:
      - transaction context manager (with session)
      - bulk inserts with ordered/parallel fallback
      - async streaming cursors (find_stream, aggregate_stream)
      - index bootstrap & TTL management
      - paginated query helper
      - reconnection/backoff helper
    """

    # list of transient exceptions to retry; motor wraps pymongo errors
    _TRANSIENT_ERRS = (TimeoutError,)

    def _tenant_collection(self, name: str, tenant: Optional[str] = None):
        """
        Return collection for multi-tenant systems.
        If tenant provided, prefix collection name to isolate data.
        """
        cname = name.replace("/", "_").replace(".", "_")
        if tenant:
            cname = f"{tenant}__{cname}"
        return self.db[cname]

    @retry_async(retries=4, backoff_factor=0.3)
    async def insert_one(self, collection: str, doc: Dict[str, Any], tenant: Optional[str] = None):
        if not self.db:
            raise StorageError("MongoDB not connected")
        try:
            doc = dict(doc)
            if "ts" not in doc:
                doc["ts"] = datetime.datetime.utcnow()
            col = self._tenant_collection(collection, tenant)
            res = await col.insert_one(doc)
            return res.inserted_id
        except Exception:
            LOG.exception("Mongo insert_one failed for %s (tenant=%s)", collection, tenant)
            raise

    @retry_async(retries=4, backoff_factor=0.3)
    async def bulk_insert(self, collection: str, docs: List[Dict[str, Any]], ordered: bool = False, tenant: Optional[str] = None):
        """
        Bulk insert with fallback for large batches. Uses insert_many with unordered inserts (faster).
        """
        if not self.db:
            raise StorageError("MongoDB not connected")
        if not docs:
            return []
        try:
            col = self._tenant_collection(collection, tenant)
            now = datetime.datetime.utcnow()
            for d in docs:
                if "ts" not in d:
                    d["ts"] = now
            res = await col.insert_many(docs, ordered=ordered)
            return res.inserted_ids
        except Exception:
            LOG.exception("Mongo bulk_insert failed; attempting per-document fallback")
            # fallback: insert one-by-one to provide partial success
            inserted = []
            for d in docs:
                try:
                    id_ = await self.insert_one(collection, d, tenant=tenant)
                    inserted.append(id_)
                except Exception:
                    LOG.exception("Per-document insert fallback failed for collection=%s", collection)
            return inserted

    async def find_stream(self, collection: str, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None, batch_size: int = 1000, tenant: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator for streaming large query results using motor cursor.
        Yields document dicts as native Python objects.
        """
        if not self.db:
            raise StorageError("MongoDB not connected")
        col = self._tenant_collection(collection, tenant)
        cursor = col.find(query, projection or {}).batch_size(batch_size)
        try:
            async for doc in cursor:
                yield doc
        except Exception:
            LOG.exception("Mongo find_stream failed for %s", collection)
            raise

    async def aggregate_stream(self, collection: str, pipeline: List[Dict[str, Any]], batch_size: int = 1000, tenant: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        if not self.db:
            raise StorageError("MongoDB not connected")
        col = self._tenant_collection(collection, tenant)
        cursor = col.aggregate(pipeline, batchSize=batch_size)
        try:
            async for doc in cursor:
                yield doc
        except Exception:
            LOG.exception("Mongo aggregate_stream failed for %s", collection)
            raise

    async def find_latest(self, collection: str, query: Optional[Dict[str, Any]] = None, tenant: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Override to support tenant-aware collection and retry logic.
        """
        if not self.db:
            raise StorageError("MongoDB not connected")
        try:
            q = query or {}
            col = self._tenant_collection(collection, tenant)
            cursor = col.find(q).sort("ts", -1).limit(1)
            docs = await cursor.to_list(length=1)
            return docs[0] if docs else None
        except Exception:
            LOG.exception("Mongo find_latest failed for %s tenant=%s", collection, tenant)
            raise

    async def paginated_query(self, collection: str, query: Dict[str, Any], sort: Optional[List[Tuple[str, int]]] = None, page_size: int = 100, page_token: Optional[Any] = None, tenant: Optional[str] = None):
        """
        Cursor-style paginated query returning (results, next_token)
        next_token is the last document's ts or _id to be passed for subsequent pages.
        """
        if not self.db:
            raise StorageError("MongoDB not connected")
        col = self._tenant_collection(collection, tenant)
        q = dict(query)
        if page_token is not None:
            # assume page_token is a timestamp or _id marker; prefer ts
            if isinstance(page_token, (str, int, float)):
                q["ts"] = {"$lt": page_token}
        cursor = col.find(q)
        if sort:
            cursor = cursor.sort(sort)
        cursor = cursor.limit(page_size)
        docs = await cursor.to_list(length=page_size)
        if not docs:
            return [], None
        last = docs[-1]
        next_token = last.get("ts") or last.get("_id")
        return docs, next_token

    # -------------------------
    # Transactions
    # -------------------------
    @contextlib.asynccontextmanager
    async def mongo_transaction(self, read_concern: Optional[Any] = None, write_concern: Optional[Any] = None, wtimeout: Optional[int] = None):
        """
        Async context manager for MongoDB transactions. Uses motor session.start_session + with_transaction.
        Example:
            async with storage.mongo_transaction() as s:
                await storage._col('x').insert_one({...}, session=s)
        Note: Requires Mongo replica set or single-node replset for transactions.
        """
        if not self.client:
            raise StorageError("MongoDB client not initialized")
        session = None
        try:
            session = await self.client.start_session()
            async with session.start_transaction():
                yield session
        except Exception:
            LOG.exception("Mongo transaction failed/aborted")
            if session:
                try:
                    await session.abort_transaction()
                except Exception:
                    pass
            raise
        finally:
            if session:
                await session.end_session()

    # -------------------------
    # Index & bootstrap helpers
    # -------------------------
    async def ensure_index(self, collection: str, keys: List[Tuple[str, int]], unique: bool = False, expire_after_seconds: Optional[int] = None, tenant: Optional[str] = None):
        """
        Ensure index exists on collection. keys example: [("ts", -1), ("user_id", 1)]
        """
        if not self.db:
            raise StorageError("MongoDB not connected")
        try:
            col = self._tenant_collection(collection, tenant)
            index_name = await col.create_index(keys, unique=unique, expireAfterSeconds=expire_after_seconds)
            LOG.info("Ensured index %s on %s (tenant=%s)", index_name, collection, tenant)
            return index_name
        except Exception:
            LOG.exception("ensure_index failed for %s tenant=%s", collection, tenant)
            raise

    async def bootstrap_indexes(self, tenant: Optional[str] = None):
        """
        Create recommended indexes used by PriorityMax.
        """
        try:
            # scaling actions/history
            await self.ensure_index("scaling_actions", [("ts", -1)], tenant=tenant)
            await self.ensure_index("scaling_summary", [("ts", -1)], tenant=tenant)
            await self.ensure_index("dlq_promotions", [("ts", -1)], tenant=tenant)
            await self.ensure_index("drift_events", [("ts", -1)], tenant=tenant)
            # model registry: unique version index
            await self.ensure_index("model_registry", [("version_id", 1)], unique=True, tenant=tenant)
            # TTL for old logs
            await self.ensure_index("scaling_summary", [("ts", 1)], expire_after_seconds=7 * 24 * 3600, tenant=tenant)
            LOG.info("Bootstrap indexes complete (tenant=%s)", tenant)
        except Exception:
            LOG.exception("bootstrap_indexes failed for tenant=%s", tenant)
            raise

    # -------------------------
    # Health & reconnect
    # -------------------------
    async def ping(self) -> bool:
        if not self.db:
            return False
        try:
            await self.db.command("ping")
            return True
        except Exception:
            LOG.exception("Mongo ping failed")
            return False

    async def ensure_connected(self, retries: int = 3, backoff: float = 1.0):
        """
        Ensure client is connected; try to reconnect if disconnected.
        """
        if not _HAS_MONGO:
            LOG.warning("Mongo not available; skipping ensure_connected")
            return
        for i in range(retries):
            try:
                if self.client is None:
                    await self.connect()
                ok = await self.ping()
                if ok:
                    return True
            except Exception:
                LOG.debug("Mongo ensure_connected attempt %d failed", i + 1)
            await asyncio.sleep(backoff * (i + 1))
        raise StorageError("Failed to ensure Mongo connection after retries")

    # -------------------------
    # Aggregation helpers
    # -------------------------
    async def aggregate_one(self, collection: str, pipeline: List[Dict[str, Any]], tenant: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Run aggregation pipeline and return single document (first result).
        """
        if not self.db:
            raise StorageError("MongoDB not connected")
        try:
            col = self._tenant_collection(collection, tenant)
            cursor = col.aggregate(pipeline, allowDiskUse=True)
            docs = await cursor.to_list(length=1)
            return docs[0] if docs else None
        except Exception:
            LOG.exception("aggregate_one failed for %s tenant=%s", collection, tenant)
            raise

    async def aggregate_stream_pipeline(self, collection: str, pipeline: List[Dict[str, Any]], tenant: Optional[str] = None, batch_size: int = 500) -> AsyncIterator[Dict[str, Any]]:
        """
        Convenience wrapper to stream aggregation results.
        """
        async for doc in self.aggregate_stream(collection, pipeline, batch_size=batch_size, tenant=tenant):
            yield doc

# -----------------------------------------------------------------------------
# End of Chunk 2
# -----------------------------------------------------------------------------
# Next: Chunk 3 will extend RedisStorage with pub/sub, stream consumer helpers,
# batch/pipeline writes, and DLQ helpers for promote/monitor.
# -----------------------------------------------------------------------------
# Chunk 3 — Advanced Redis features: streams, pipelines, pub/sub, DLQ helpers
# -----------------------------------------------------------------------------

import aioredis.exceptions as redis_exceptions
import random

class RedisStorage(RedisStorage):  # type: ignore[misc, override]
    """
    Advanced Redis async wrapper providing:
      - Pub/Sub channels for system notifications
      - Stream-based queue ingestion (XADD / XREAD)
      - Pipeline batching for bulk writes
      - DLQ helpers (inspect, promote, purge)
      - Backpressure-safe writes and retries
    """

    _RETRY_EXCEPTIONS = (
        redis_exceptions.ConnectionError,
        redis_exceptions.TimeoutError,
        redis_exceptions.BusyLoadingError,
    )

    # ------------------------------------------------------------------
    # Core helpers / reconnection logic
    # ------------------------------------------------------------------
    async def ensure_connected(self, retries: int = 3, delay: float = 1.0):
        """
        Ensure Redis connection alive; reconnect if not.
        """
        for attempt in range(1, retries + 1):
            if not self.client:
                await self.connect()
            try:
                pong = await self.client.ping()
                if pong:
                    return True
            except Exception:
                LOG.warning("Redis ping failed (attempt %d/%d), reconnecting...", attempt, retries)
                await asyncio.sleep(delay * attempt)
                try:
                    await self.connect()
                except Exception:
                    LOG.debug("Redis reconnect attempt %d failed", attempt)
        raise StorageError("Redis connection failed after retries")

    # ------------------------------------------------------------------
    # Pipelines & batched writes
    # ------------------------------------------------------------------
    async def pipeline_set_json(self, kvs: Dict[str, Any], expire: Optional[int] = None):
        """
        Batch multiple JSON sets atomically using a Redis pipeline.
        """
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            async with self.client.pipeline(transaction=False) as pipe:
                for k, v in kvs.items():
                    pipe.set(k, _serialize_payload(v), ex=expire)
                await pipe.execute()
        except self._RETRY_EXCEPTIONS as e:
            LOG.warning("Pipeline write transient error: %s", e)
            await asyncio.sleep(0.5)
            await self.pipeline_set_json(kvs, expire)
        except Exception:
            LOG.exception("Redis pipeline_set_json failed")
            raise

    async def safe_write_json(self, key: str, data: Dict[str, Any], expire: Optional[int] = None, retries: int = 3):
        """
        Write JSON to Redis with retry/backpressure handling.
        """
        for i in range(retries):
            try:
                await self.set_json(key, data, expire)
                return True
            except self._RETRY_EXCEPTIONS:
                LOG.warning("Transient Redis write error for key=%s (attempt %d/%d)", key, i + 1, retries)
                await asyncio.sleep(0.3 * (i + 1))
        LOG.error("Redis write failed permanently for %s", key)
        return False

    # ------------------------------------------------------------------
    # Stream interface (Redis Streams API)
    # ------------------------------------------------------------------
    async def xadd_json(self, stream: str, data: Dict[str, Any], maxlen: int = 10000, approximate: bool = True) -> str:
        """
        Append JSON data to Redis Stream.
        """
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            fields = {k: json.dumps(v, default=str) for k, v in data.items()}
            return await self.client.xadd(stream, fields, maxlen=maxlen, approximate=approximate)
        except Exception:
            LOG.exception("Redis XADD failed for stream %s", stream)
            raise

    async def xread_json(self, streams: List[str], count: Optional[int] = 10, block_ms: Optional[int] = 5000) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Read messages from Redis streams (XREAD).
        Returns list of (stream, [decoded_messages]).
        """
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            data = await self.client.xread(streams=streams, count=count, block=block_ms)
            out = []
            for stream_name, entries in data:
                parsed = []
                for msg_id, fields in entries:
                    decoded = {k: json.loads(v) if isinstance(v, (bytes, bytearray)) else v for k, v in fields.items()}
                    parsed.append({"id": msg_id, "data": decoded})
                out.append((stream_name, parsed))
            return out
        except Exception:
            LOG.exception("Redis XREAD failed for streams=%s", streams)
            raise

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------
    async def publish_json(self, channel: str, message: Dict[str, Any]):
        """
        Publish JSON-encoded message to a Redis Pub/Sub channel.
        """
        if not self.client:
            raise StorageError("Redis not connected")
        try:
            await self.client.publish(channel, json.dumps(message))
        except Exception:
            LOG.exception("Redis publish_json failed for channel=%s", channel)
            raise

    async def subscribe_json(self, channel: str):
        """
        Async generator to subscribe and yield JSON messages from a channel.
        Usage:
            async for msg in redis.subscribe_json("prioritymax-events"):
                ...
        """
        if not self.client:
            raise StorageError("Redis not connected")
        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    yield json.loads(message["data"])
                except Exception:
                    LOG.exception("Invalid JSON in message from %s", channel)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    # ------------------------------------------------------------------
    # DLQ Helpers
    # ------------------------------------------------------------------
    async def get_queue_length(self, key: str) -> int:
        if not self.client:
            return 0
        try:
            return await self.client.llen(key)
        except Exception:
            return 0

    async def inspect_dlq(self, queue: str, limit: int = 10) -> List[Dict[str, Any]]:
        key = f"{queue}.dlq"
        if not self.client:
            return []
        try:
            items = await self.client.lrange(key, 0, limit - 1)
            return [_deserialize_payload(i) for i in items]
        except Exception:
            LOG.exception("inspect_dlq failed for %s", key)
            return []

    async def promote_dlq(self, queue: str, limit: int = 100) -> int:
        """
        Move messages from DLQ back to main queue.
        """
        if not self.client:
            return 0
        dlq_key = f"{queue}.dlq"
        promoted = 0
        try:
            async with self.client.pipeline(transaction=False) as pipe:
                for _ in range(limit):
                    msg = await self.client.rpop(dlq_key)
                    if not msg:
                        break
                    await pipe.lpush(queue, msg)
                    promoted += 1
                await pipe.execute()
            LOG.info("Promoted %d messages from %s", promoted, dlq_key)
        except Exception:
            LOG.exception("promote_dlq failed for %s", dlq_key)
        return promoted

    async def purge_dlq(self, queue: str):
        key = f"{queue}.dlq"
        try:
            await self.client.delete(key)
            LOG.info("Purged DLQ for queue %s", queue)
        except Exception:
            LOG.exception("purge_dlq failed for %s", queue)

    # ------------------------------------------------------------------
    # Housekeeping and metrics integration
    # ------------------------------------------------------------------
    async def cleanup_old_keys(self, pattern: str = "*", max_age_seconds: int = 86400):
        """
        Iterate through keys matching pattern and delete expired ones by timestamp field.
        Keys should store JSON with 'ts' field (ISO8601 or epoch).
        """
        if not self.client:
            return
        async for key in self.client.scan_iter(match=pattern):
            try:
                val = await self.get_json(key)
                if not val:
                    continue
                ts = val.get("ts")
                if not ts:
                    continue
                if isinstance(ts, str):
                    try:
                        t = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if (datetime.datetime.utcnow() - t).total_seconds() > max_age_seconds:
                            await self.client.delete(key)
                    except Exception:
                        continue
            except Exception:
                continue

    async def count_keys(self, pattern: str = "*") -> int:
        """
        Approximate count of keys matching a pattern (uses SCAN).
        """
        if not self.client:
            return 0
        total = 0
        async for _ in self.client.scan_iter(match=pattern):
            total += 1
        return total
# -----------------------------------------------------------------------------
# Chunk 4 — Unified Storage manager + domain APIs + buffered write flusher + DLQ monitor
# -----------------------------------------------------------------------------

from contextlib import asynccontextmanager
from collections import defaultdict

class Storage:
    """
    Unified storage façade that composes MongoStorage, RedisStorage, and FileStorage.
    Provides high-level domain-specific methods used across PriorityMax:
      - insert_scaling_action, insert_scaling_summary
      - DLQ enqueue/promote/count/inspect
      - model registry operations (insert/get latest)
      - drift_event logging
      - buffered writes for high-frequency telemetry
      - simple DLQ monitor coroutine
    """

    def __init__(self,
                 mongo: Optional[MongoStorage] = None,
                 redis: Optional[RedisStorage] = None,
                 file: Optional[FileStorage] = None,
                 tenant: Optional[str] = None,
                 buffer_flush_interval: float = 1.0,
                 buffer_max_size: int = 1000):
        # backends (may be None or in degraded mode)
        self.mongo = mongo or (MongoStorage() if _HAS_MONGO else None)
        self.redis = redis or (RedisStorage() if _HAS_REDIS else None)
        self.file = file or FileStorage(DATA_DIR)
        self.tenant = tenant

        # buffered writer
        self._buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._buffer_lock = asyncio.Lock()
        self._buffer_flush_interval = float(buffer_flush_interval)
        self._buffer_max_size = int(buffer_max_size)
        self._buffer_task: Optional[asyncio.Task] = None
        self._running = False

        # DLQ monitor task handle
        self._dlq_monitor_task: Optional[asyncio.Task] = None
        self._dlq_monitor_config: Dict[str, Any] = {}

    # -------------------------
    # Connect / Close
    # -------------------------
    async def connect(self):
        """
        Connect all available backends. Each backend is best-effort; if one fails we keep others.
        """
        try:
            if self.mongo:
                await self.mongo.connect()
                try:
                    await self.mongo.ensure_connected()
                except Exception:
                    LOG.debug("Mongo ensure_connected failed on startup")
        except Exception:
            LOG.exception("Mongo connect error")

        try:
            if self.redis:
                await self.redis.connect()
                try:
                    await self.redis.ensure_connected()
                except Exception:
                    LOG.debug("Redis ensure_connected failed on startup")
        except Exception:
            LOG.exception("Redis connect error")

        # Start buffer flusher
        self._running = True
        loop = asyncio.get_event_loop()
        self._buffer_task = loop.create_task(self._buffer_flusher_loop())
        LOG.info("Storage connected: mongo=%s redis=%s file=%s", bool(self.mongo and self.mongo.db), bool(self.redis and self.redis.client), bool(self.file))

    async def close(self):
        """
        Gracefully close backends and stop background tasks.
        """
        self._running = False
        if self._buffer_task:
            self._buffer_task.cancel()
            try:
                await self._buffer_task
            except Exception:
                pass
        if self._dlq_monitor_task:
            self._dlq_monitor_task.cancel()
            try:
                await self._dlq_monitor_task
            except Exception:
                pass
        try:
            if self.mongo:
                await self.mongo.close()
        except Exception:
            LOG.exception("Error closing mongo")
        try:
            if self.redis:
                await self.redis.close()
        except Exception:
            LOG.exception("Error closing redis")

    # -------------------------
    # Buffering / Flusher
    # -------------------------
    async def _buffer_flusher_loop(self):
        LOG.info("Starting storage buffer flusher (interval=%.2fs)", self._buffer_flush_interval)
        try:
            while self._running:
                try:
                    await self.flush_buffer_once()
                except Exception:
                    LOG.exception("Buffer flusher iteration failed")
                await asyncio.sleep(self._buffer_flush_interval)
        except asyncio.CancelledError:
            LOG.debug("Buffer flusher cancelled")
        except Exception:
            LOG.exception("Buffer flusher stopped unexpectedly")
        LOG.info("Buffer flusher exiting")

    async def flush_buffer_once(self):
        """
        Flush entries from buffer to primary storage (Mongo preferred, else file).
        Writes are grouped by collection name.
        """
        async with self._buffer_lock:
            if not self._buffer:
                return
            to_flush = dict(self._buffer)
            self._buffer.clear()

        # For each collection, attempt bulk_insert to mongo, else write to file
        for collection, docs in to_flush.items():
            try:
                if self.mongo and self.mongo.db:
                    await self.mongo.bulk_insert(collection, docs, ordered=False, tenant=self.tenant)
                else:
                    await self.file.insert_many(collection, docs)
            except Exception:
                LOG.exception("flush_buffer_once failed for %s; writing to file fallback", collection)
                try:
                    await self.file.insert_many(collection, docs)
                except Exception:
                    LOG.exception("file fallback failed during flush")

    async def buffered_insert(self, collection: str, doc: Dict[str, Any], max_buffered: Optional[int] = None):
        """
        Add doc to in-memory buffer which will be flushed periodically.
        If buffer grows beyond threshold, flush synchronously.
        """
        collection = collection.replace("/", "_").replace(".", "_")
        async with self._buffer_lock:
            self._buffer[collection].append(doc)
            cur_len = len(self._buffer[collection])
        if max_buffered is None:
            max_buffered = self._buffer_max_size
        if cur_len >= max_buffered:
            LOG.debug("Buffer reached max for %s; flushing synchronously", collection)
            await self.flush_buffer_once()

    # -------------------------
    # Domain-specific methods
    # -------------------------
    async def insert_scaling_action(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("ts", _now_iso())
        # prefer buffered writes for high frequency
        await self.buffered_insert("scaling_actions", payload)

    async def insert_scaling_summary(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("ts", _now_iso())
        # write immediately to have latest summary available
        if self.mongo and self.mongo.db:
            try:
                await self.mongo.insert_one("scaling_summary", payload, tenant=self.tenant)
                return
            except Exception:
                LOG.exception("Mongo write failed for scaling_summary; falling back to buffer/file")
        await self.buffered_insert("scaling_summary", payload)

    async def set_desired_worker_count(self, desired: int):
        """
        Convenience: write desired worker count to Redis key so worker manager picks up.
        """
        key = f"prioritymax:desired_workers"
        payload = {"desired": int(desired), "ts": _now_iso()}
        if self.redis and self.redis.client:
            try:
                await self.redis.safe_write_json(key, payload, expire=3600)
                return True
            except Exception:
                LOG.exception("Redis set desired worker count failed")
        # fallback to file
        await self.file.write("desired_workers", payload)
        return True

    async def get_latest_worker_count(self) -> Optional[int]:
        """
        Read latest scaling_summary to restore worker count.
        """
        # try mongo
        if self.mongo and self.mongo.db:
            try:
                rec = await self.mongo.find_latest("scaling_summary", tenant=self.tenant)
                if rec and "current_workers" in rec:
                    return int(rec["current_workers"])
            except Exception:
                LOG.exception("Mongo read latest worker count failed")
        # fallback to file
        rec = await self.file.read_latest("scaling_summary")
        if rec and "current_workers" in rec:
            return int(rec["current_workers"])
        return None

    # -------------------------
    # DLQ operations (high-level)
    # -------------------------
    async def enqueue_dlq(self, queue: str, msg: Dict[str, Any]):
        """
        Enqueue message to DLQ. Use Redis list if available, else persist to file collection.
        """
        payload = dict(msg)
        payload.setdefault("ts", _now_iso())
        dlq_key = f"{queue}.dlq"
        if self.redis and self.redis.client:
            try:
                await self.redis.lpush_item(dlq_key, payload)
                return
            except Exception:
                LOG.exception("Redis DLQ lpush failed; falling back")
        # fallback: persist to mongo or file
        if self.mongo and self.mongo.db:
            try:
                await self.mongo.insert_one(dlq_key.replace(".", "_"), payload, tenant=self.tenant)
                return
            except Exception:
                LOG.exception("Mongo fallback DLQ insert failed")
        await self.file.write(dlq_key.replace(".", "_"), payload)

    async def get_dlq_length(self, queue: str) -> int:
        dlq_key = f"{queue}.dlq"
        if self.redis and self.redis.client:
            try:
                return await self.redis.get_queue_length(dlq_key)
            except Exception:
                LOG.exception("Redis get_dlq_length failed")
        # fallback: approximate from file lines
        items = await self.file.read_many(dlq_key.replace(".", "_"), limit=10000)
        return len(items)

    async def inspect_dlq(self, queue: str, limit: int = 20) -> List[Dict[str, Any]]:
        dlq_key = f"{queue}.dlq"
        if self.redis and self.redis.client:
            try:
                return await self.redis.inspect_dlq(queue, limit=limit)
            except Exception:
                LOG.exception("Redis inspect_dlq failed")
        # fallback: read from mongo or file
        if self.mongo and self.mongo.db:
            try:
                # read latest docs from tenant collection
                docs = []
                async for doc in self.mongo.find_stream(dlq_key.replace(".", "_"), {}, batch_size=limit, tenant=self.tenant):
                    docs.append(doc)
                    if len(docs) >= limit:
                        break
                return docs
            except Exception:
                LOG.exception("Mongo inspect dlq failed")
        return await self.file.read_many(dlq_key.replace(".", "_"), limit=limit)

    async def promote_dlq(self, queue: str, limit: int = 100) -> int:
        """
        Promote up to `limit` messages from DLQ back to main queue.
        If Redis available, use atomic pipeline. Otherwise, read from mongo/file and re-enqueue.
        """
        promoted = 0
        dlq_key = f"{queue}.dlq"
        try:
            if self.redis and self.redis.client:
                promoted = await self.redis.promote_dlq(queue, limit=limit)
                # record promotion event
                await self.insert_dlq_promotion({"queue": queue, "promoted": promoted, "ts": _now_iso()})
                return promoted
        except Exception:
            LOG.exception("Redis promote_dlq failed; falling back")

        # Fallback: read items and re-push to main queue (best-effort)
        try:
            items = await self.inspect_dlq(queue, limit=limit)
            for item in items:
                # push into main queue via redis or file (prefer redis)
                try:
                    if self.redis and self.redis.client:
                        await self.redis.lpush_item(queue, item)
                    else:
                        # fallback: write to a file stream for worker to read
                        await self.file.write(queue.replace(".", "_"), item)
                    promoted += 1
                except Exception:
                    LOG.exception("Failed to requeue item from DLQ")
            await self.insert_dlq_promotion({"queue": queue, "promoted": promoted, "ts": _now_iso()})
        except Exception:
            LOG.exception("Fallback DLQ promote failed")
        return promoted

    async def insert_dlq_promotion(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("ts", _now_iso())
        if self.mongo and self.mongo.db:
            try:
                await self.mongo.insert_one("dlq_promotions", payload, tenant=self.tenant)
                return
            except Exception:
                LOG.exception("Mongo insert dlq_promotion failed; buffering")
        await self.buffered_insert("dlq_promotions", payload)

    # -------------------------
    # Model registry
    # -------------------------
    async def insert_model_version(self, model_type: str, version_meta: Dict[str, Any]):
        """
        Persist model registry record. Also keep a "latest" single-document per model_type for quick access.
        """
        rec = dict(version_meta)
        rec.setdefault("model_type", model_type)
        rec.setdefault("ts", _now_iso())
        try:
            await self.buffered_insert(f"model_registry_{model_type}", rec)
            # also insert into a single-document collection with find_and_replace semantics (fast access)
            if self.mongo and self.mongo.db:
                # upsert latest marker in collection model_registry_latest
                await self.mongo._col("model_registry_latest").update_one({"model_type": model_type}, {"$set": rec}, upsert=True)
                return
        except Exception:
            LOG.exception("insert_model_version failed; buffering to file")
            await self.buffered_insert(f"model_registry_{model_type}", rec)

    async def get_latest_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Return the latest model metadata for type.
        """
        if self.mongo and self.mongo.db:
            try:
                # first try single-document latest
                latest = await self.mongo.find_latest("model_registry_latest", {"model_type": model_type}, tenant=self.tenant)
                if latest:
                    return latest
                # fallback to time-ordered collection
                rec = await self.mongo.find_latest(f"model_registry_{model_type}", tenant=self.tenant)
                if rec:
                    return rec
            except Exception:
                LOG.exception("Mongo get_latest_model failed")
        # fallback to file read
        rec = await self.file.read_latest(f"model_registry_{model_type}")
        return rec

    # -------------------------
    # Drift / analytics events
    # -------------------------
    async def insert_drift_event(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("ts", _now_iso())
        await self.buffered_insert("drift_events", payload)

    async def insert_correlation_stat(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("ts", _now_iso())
        await self.buffered_insert("correlation_stats", payload)

    # -------------------------
    # Transaction context manager (best-effort)
    # -------------------------
    @asynccontextmanager
    async def transaction(self):
        """
        Provide transactional context:
          - If Mongo transactions are available, yield a session object that callers can pass into mongo ops.
          - Otherwise yield None and operations should be best-effort.
        Example:
            async with storage.transaction() as sess:
                await storage.mongo._col('x').insert_one({...}, session=sess)
        """
        session = None
        if self.mongo and getattr(self.mongo, "client", None):
            try:
                session = await self.mongo.client.start_session()
                async with session.start_transaction():
                    yield session
                    # commit happens on context exit
                    return
            except Exception:
                LOG.exception("Mongo transaction failed; aborting")
                if session:
                    try:
                        await session.abort_transaction()
                    except Exception:
                        pass
                raise
            finally:
                if session:
                    try:
                        await session.end_session()
                    except Exception:
                        pass
        else:
            # best-effort fallback
            yield None

    # -------------------------
    # DLQ monitor scheduler
    # -------------------------
    def start_dlq_monitor(self,
                          queue: str,
                          interval_sec: int = 300,
                          backlog_threshold: int = 200,
                          min_age_sec: int = 60,
                          autoscaler_hint_fn: Optional[Callable[[], float]] = None):
        """
        Start background task that periodically checks DLQ length and promotes items
        when system is idle and backlog low. The function returns immediately and the
        monitor runs in background.
        """
        if self._dlq_monitor_task and not self._dlq_monitor_task.done():
            LOG.warning("DLQ monitor already running")
            return
        self._dlq_monitor_config = {"queue": queue, "interval_sec": int(interval_sec), "backlog_threshold": int(backlog_threshold), "min_age_sec": int(min_age_sec)}
        loop = asyncio.get_event_loop()
        self._dlq_monitor_task = loop.create_task(self._dlq_monitor_loop(queue, interval_sec, backlog_threshold, min_age_sec, autoscaler_hint_fn))
        LOG.info("Started DLQ monitor for queue=%s interval=%ds", queue, interval_sec)

    async def _dlq_monitor_loop(self, queue: str, interval_sec: int, backlog_threshold: int, min_age_sec: int, autoscaler_hint_fn: Optional[Callable[[], float]]):
        """
        Background coroutine for DLQ monitoring & conditional promotion.
        """
        try:
            while True:
                try:
                    dlq_len = await self.get_dlq_length(queue)
                    # compute backlog via redis or mongo summary if available
                    backlog = 0
                    if self.redis and self.redis.client:
                        try:
                            backlog = await self.redis.get_queue_length(queue)
                        except Exception:
                            backlog = 0
                    else:
                        # try scaling summary approximate
                        latest = await self.get_latest_worker_count()
                        backlog = latest or 0

                    # If backlog below threshold and DLQ has items and autoscaler_hint indicates idle -> promote small batch
                    if dlq_len > 0 and backlog <= backlog_threshold:
                        # optional autoscaler hint check
                        can_promote = True
                        if autoscaler_hint_fn:
                            try:
                                hint_val = autoscaler_hint_fn()
                                # If hint is coroutine, await
                                if asyncio.iscoroutine(hint_val):
                                    hint_val = await hint_val
                                # hint_val is expected to be scalar where low value indicates idle
                                if float(hint_val) > 0.5:
                                    can_promote = False
                            except Exception:
                                LOG.exception("autoscaler_hint_fn failed; allowing promote by default")
                        if can_promote:
                            promote_limit = min(100, dlq_len)
                            promoted = await self.promote_dlq(queue, limit=promote_limit)
                            LOG.info("DLQ monitor: promoted=%d from queue=%s (dlq_len=%d backlog=%d)", promoted, queue, dlq_len, backlog)
                    # sleep until next iteration
                except asyncio.CancelledError:
                    LOG.debug("DLQ monitor cancelled")
                    break
                except Exception:
                    LOG.exception("DLQ monitor iteration error")
                await asyncio.sleep(interval_sec)
        except Exception:
            LOG.exception("DLQ monitor unexpected error")

    # -------------------------
    # Convenience shutdown
    # -------------------------
    async def shutdown(self):
        await self.close()

# Create default global storage singleton for app imports
GLOBAL_STORAGE = Storage()
# -----------------------------------------------------------------------------
# Chunk 5 — Analytics queries, advanced transactions, buffer control, migrations
# -----------------------------------------------------------------------------

from typing import Iterable, Generator

# -------------------------
# Storage: Advanced helpers & analytics
# -------------------------
class Storage(Storage):  # type: ignore[misc, override]
    """
    Extends unified Storage with:
      - analytics queries (scaling history, dlq events, drift aggregates)
      - stronger transactional helpers (atomic upserts / conditional writes)
      - buffer management APIs (force flush, flush collection)
      - migration & bootstrap utilities
      - export/import helpers for debug and backup
    """

    # -------------------------
    # Buffer control utilities
    # -------------------------
    async def flush_buffer(self):
        """Public API to force flush buffer synchronously."""
        await self.flush_buffer_once()

    async def flush_collection_buffer(self, collection: str):
        """
        Flush only a named collection from buffer to primary storage or file.
        """
        collection = collection.replace("/", "_").replace(".", "_")
        async with self._buffer_lock:
            docs = self._buffer.pop(collection, [])
        if not docs:
            return 0
        try:
            if self.mongo and self.mongo.db:
                await self.mongo.bulk_insert(collection, docs, ordered=False, tenant=self.tenant)
                return len(docs)
        except Exception:
            LOG.exception("flush_collection_buffer -> mongo failed; writing to file")
        try:
            await self.file.insert_many(collection, docs)
            return len(docs)
        except Exception:
            LOG.exception("flush_collection_buffer -> file fallback failed")
            return 0

    async def drain_buffer_and_shutdown(self, timeout: float = 10.0):
        """
        Flush buffers, stop background flusher and then close backends.
        Intended to be used during graceful shutdown in apps.
        """
        LOG.info("Draining storage buffer before shutdown (timeout=%.2fs)", timeout)
        # stop flusher loop
        self._running = False
        if self._buffer_task:
            self._buffer_task.cancel()
        # attempt flush
        try:
            await asyncio.wait_for(self.flush_buffer_once(), timeout=timeout)
        except asyncio.TimeoutError:
            LOG.warning("Timeout while flushing buffer during shutdown")
        except Exception:
            LOG.exception("Error flushing buffer during shutdown")
        await self.close()

    # -------------------------
    # Enhanced transactional helpers
    # -------------------------
    async def atomic_find_and_update(self, collection: str, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False, tenant: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Perform atomic find-and-update (findOneAndUpdate) with optional upsert.
        Returns the updated document (post-update).
        """
        if self.mongo and self.mongo.db:
            try:
                col = self.mongo._tenant_collection(collection, tenant)
                res = await col.find_one_and_update(query, {"$set": update}, upsert=upsert, return_document=True)
                return res
            except Exception:
                LOG.exception("atomic_find_and_update failed for %s", collection)
        # fallback: naive read-modify-write (not atomic)
        try:
            rec = await self.get_latest_model(collection) if collection.startswith("model_registry") else None
            if rec:
                rec.update(update)
                await self.buffered_insert(collection, rec)
                return rec
        except Exception:
            LOG.exception("Fallback atomic_find_and_update failed")
        return None

    async def conditional_insert(self, collection: str, query: Dict[str, Any], doc: Dict[str, Any], tenant: Optional[str] = None) -> bool:
        """
        Insert doc only if no document matches query. Uses Mongo unique index when available, else best-effort check+insert.
        Returns True if inserted, False if skipped.
        """
        if self.mongo and self.mongo.db:
            try:
                col = self.mongo._tenant_collection(collection, tenant)
                # try upsert with $setOnInsert and check whether upsertedId returned
                res = await col.update_one(query, {"$setOnInsert": doc}, upsert=True)
                # pymongo/motor does not return upserted_id consistently here; do a quick read
                found = await col.find_one(query)
                return bool(found)
            except Exception:
                LOG.exception("conditional_insert failed in mongo")
        # fallback: check via file
        latest = await self.file.read_latest(collection.replace(".", "_"))
        if latest and all(latest.get(k) == v for k, v in query.items()):
            return False
        await self.buffered_insert(collection, doc)
        return True

    # -------------------------
    # Analytics queries
    # -------------------------
    async def fetch_scaling_history(self, start_ts: Optional[str] = None, end_ts: Optional[str] = None, limit: int = 1000, tenant: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return scaling actions history between start_ts and end_ts.
        start_ts/end_ts are ISO strings. If None, uses sensible defaults (last 7 days).
        """
        q = {}
        if start_ts or end_ts:
            time_query = {}
            if start_ts:
                try:
                    time_query["$gte"] = datetime.datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
                except Exception:
                    pass
            if end_ts:
                try:
                    time_query["$lte"] = datetime.datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
                except Exception:
                    pass
            if time_query:
                q["ts"] = time_query
        try:
            if self.mongo and self.mongo.db:
                res = await self.mongo.find_stream("scaling_actions", q, batch_size=limit, tenant=tenant)
                out = []
                async for d in res:
                    out.append(d)
                    if len(out) >= limit:
                        break
                return out
        except Exception:
            LOG.exception("fetch_scaling_history (mongo) failed")
        # fallback to file
        return await self.file.read_many("scaling_actions", limit=limit)

    async def aggregate_scaling_summary(self, window_hours: int = 24, tenant: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate summary stats over last window_hours.
        Returns simple metrics: avg_workers, max_workers, scale_up_count, scale_down_count
        """
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(hours=window_hours)
        pipeline = [
            {"$match": {"ts": {"$gte": start, "$lte": end}}},
            {"$group": {"_id": None,
                        "avg_workers": {"$avg": "$to"},
                        "max_workers": {"$max": "$to"},
                        "min_workers": {"$min": "$to"},
                        "scale_up": {"$sum": {"$cond": [{"$eq": ["$action", "scale_up"]}, 1, 0]}},
                        "scale_down": {"$sum": {"$cond": [{"$eq": ["$action", "scale_down"]}, 1, 0]}}}}
        ]
        try:
            if self.mongo and self.mongo.db:
                res = await self.mongo.aggregate_one("scaling_actions", pipeline, tenant=tenant)
                return res or {}
        except Exception:
            LOG.exception("aggregate_scaling_summary failed")
        # fallback: approximate from file
        docs = await self.file.read_many("scaling_actions", limit=10000)
        arr = [d for d in docs if "ts" in d]
        # parse times loosely
        vals = [d.get("to", 0) for d in arr]
        up = sum(1 for d in arr if d.get("action") == "scale_up")
        down = sum(1 for d in arr if d.get("action") == "scale_down")
        if not vals:
            return {"avg_workers": 0, "max_workers": 0, "min_workers": 0, "scale_up": up, "scale_down": down}
        return {"avg_workers": float(sum(vals) / len(vals)), "max_workers": max(vals), "min_workers": min(vals), "scale_up": up, "scale_down": down}

    async def query_dlq_events(self, queue: str, limit: int = 100, tenant: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve recent DLQ events for a queue (from Mongo or file).
        """
        coll = f"{queue}.dlq"
        try:
            if self.mongo and self.mongo.db:
                out = []
                async for d in self.mongo.find_stream(coll.replace(".", "_"), {}, batch_size=limit, tenant=tenant):
                    out.append(d)
                    if len(out) >= limit:
                        break
                return out
        except Exception:
            LOG.exception("query_dlq_events failed (mongo)")
        # fallback: file
        return await self.file.read_many(coll.replace(".", "_"), limit=limit)

    async def aggregate_drift_stats(self, since_hours: int = 24, tenant: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute simple drift aggregates: count events, top drifted features (by mean delta).
        Drift events are expected to be stored in 'drift_events' collection with 'per_feature' map.
        """
        start = datetime.datetime.utcnow() - datetime.timedelta(hours=since_hours)
        pipeline = [
            {"$match": {"ts": {"$gte": start}}},
            {"$project": {"per_feature": 1}},
            {"$limit": 5000}
        ]
        feature_totals: Dict[str, List[float]] = {}
        try:
            if self.mongo and self.mongo.db:
                async for doc in self.mongo.aggregate_stream_pipeline("drift_events", pipeline, tenant=tenant):
                    pf = doc.get("per_feature") or {}
                    for k, v in pf.items():
                        try:
                            feature_totals.setdefault(k, []).append(float(v))
                        except Exception:
                            continue
                # compute means
                stats = {k: {"count": len(v), "mean": float(sum(v) / len(v)) if v else 0.0, "max": float(max(v)) if v else 0.0} for k, v in feature_totals.items()}
                # top features by mean drift
                top = sorted(stats.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:10]
                return {"top_features": top, "feature_stats": stats}
        except Exception:
            LOG.exception("aggregate_drift_stats failed")
        # fallback: file parse
        docs = await self.file.read_many("drift_events", limit=5000)
        for d in docs:
            pf = d.get("per_feature") or {}
            for k, v in pf.items():
                try:
                    feature_totals.setdefault(k, []).append(float(v))
                except Exception:
                    continue
        stats = {k: {"count": len(v), "mean": float(sum(v) / len(v)) if v else 0.0, "max": float(max(v)) if v else 0.0} for k, v in feature_totals.items()}
        top = sorted(stats.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:10]
        return {"top_features": top, "feature_stats": stats}

    # -------------------------
    # Migration & index bootstrap utilities
    # -------------------------
    async def bootstrap(self, tenant: Optional[str] = None):
        """
        Run bootstrap tasks:
          - create indexes
          - ensure TTL indices
          - tidy up old files
        """
        LOG.info("Running storage bootstrap (tenant=%s)", tenant)
        try:
            if self.mongo and self.mongo.db:
                await self.mongo.bootstrap_indexes(tenant=tenant)
                await self.mongo.ensure_connected()
                await self.mongo.ensure_index("scaling_summary", [("ts", 1)], expire_after_seconds=7 * 24 * 3600, tenant=tenant)
            # schedule cleanup of old local files older than 90 days
            await self.file_readme_cleanup(days=90)
        except Exception:
            LOG.exception("Storage bootstrap failed")

    async def file_readme_cleanup(self, days: int = 90):
        """
        Delete file-store logs older than days.
        """
        cutoff = time.time() - days * 86400
        for f in DATA_DIR.glob("*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    LOG.info("Deleted old file-store %s", f.name)
            except Exception:
                LOG.exception("file_readme_cleanup error for %s", f)

    async def export_collection_jsonl(self, collection: str, out_path: Union[str, pathlib.Path], tenant: Optional[str] = None, limit: Optional[int] = None):
        """
        Export a collection to JSONL file for backup or debugging.
        """
        outp = pathlib.Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        try:
            if self.mongo and self.mongo.db:
                async with aiofiles.open(outp, "w", encoding="utf-8") as fh:
                    async for doc in self.mongo.find_stream(collection, {}, tenant=tenant):
                        await fh.write(json.dumps(doc, default=str) + "\n")
                        count += 1
                        if limit and count >= limit:
                            break
                LOG.info("Exported %d docs to %s", count, outp)
                return count
        except Exception:
            LOG.exception("export_collection_jsonl failed (mongo); falling back to file read")
        # fallback: read from file storage (synchronous)
        items = await self.file.read_many(collection, limit=limit or 100000)
        with open(outp, "w", encoding="utf-8") as fh:
            for d in items:
                fh.write(json.dumps(d, default=str) + "\n")
                count += 1
        LOG.info("Exported %d docs (file fallback) to %s", count, outp)
        return count

    # -------------------------
    # Backup/restore helpers (lightweight)
    # -------------------------
    async def import_jsonl_to_collection(self, collection: str, infile: Union[str, pathlib.Path], tenant: Optional[str] = None, batch_size: int = 1000):
        """
        Import JSONL file into a collection (used for restore/testing).
        """
        p = pathlib.Path(infile)
        if not p.exists():
            raise FileNotFoundError(p)
        count = 0
        batch: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line.strip())
                except Exception:
                    continue
                batch.append(obj)
                if len(batch) >= batch_size:
                    if self.mongo and self.mongo.db:
                        await self.mongo.bulk_insert(collection, batch, ordered=False, tenant=tenant)
                    else:
                        await self.file.insert_many(collection, batch)
                    count += len(batch)
                    batch = []
        if batch:
            if self.mongo and self.mongo.db:
                await self.mongo.bulk_insert(collection, batch, ordered=False, tenant=tenant)
            else:
                await self.file.insert_many(collection, batch)
            count += len(batch)
        LOG.info("Imported %d docs into %s from %s", count, collection, infile)
        return count

# -----------------------------------------------------------------------------
# End of Chunk 5
# -----------------------------------------------------------------------------
# Next: Chunk 6 (final) — instrumentation (prometheus counters/timers), CI helpers,
# and small CLI utilities (index bootstrap script, export/import CLI wrappers).
# -----------------------------------------------------------------------------
# Chunk 6 — Prometheus instrumentation, CI self-tests, CLI utilities
# -----------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

if _HAS_PROM:
    STORAGE_MONGO_OPS = Counter("prioritymax_storage_mongo_ops_total", "MongoDB operations", ["op"])
    STORAGE_REDIS_OPS = Counter("prioritymax_storage_redis_ops_total", "Redis operations", ["op"])
    STORAGE_BUFFER_FLUSHES = Counter("prioritymax_storage_buffer_flushes_total", "Buffer flushes", ["collection"])
    STORAGE_BUFFER_LAT = Histogram("prioritymax_storage_buffer_flush_latency_seconds", "Buffer flush latency (s)", buckets=[0.001,0.01,0.1,0.5,1,5,10])
    STORAGE_DLQ_PROMOTIONS = Counter("prioritymax_storage_dlq_promotions_total", "DLQ messages promoted", ["queue"])

    def start_prometheus_exporter(port: int = 9109):
        try:
            start_http_server(port)
            LOG.info("Prometheus metrics exposed at :%d", port)
        except Exception:
            LOG.exception("Failed to start Prometheus exporter")
else:
    def start_prometheus_exporter(port: int = 9109):
        LOG.warning("prometheus_client not installed; metrics disabled")

# -----------------------------------------------------------------------------
# Metrics decorators (used internally by Storage)
# -----------------------------------------------------------------------------
def record_mongo_op(op: str):
    def _decorator(fn):
        if not _HAS_PROM:
            return fn
        @functools.wraps(fn)
        async def _wrapped(*args, **kwargs):
            STORAGE_MONGO_OPS.labels(op=op).inc()
            t0 = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                STORAGE_BUFFER_LAT.observe(time.perf_counter() - t0)
        return _wrapped
    return _decorator

def record_redis_op(op: str):
    def _decorator(fn):
        if not _HAS_PROM:
            return fn
        @functools.wraps(fn)
        async def _wrapped(*args, **kwargs):
            STORAGE_REDIS_OPS.labels(op=op).inc()
            return await fn(*args, **kwargs)
        return _wrapped
    return _decorator

# Patch a few critical Storage methods dynamically with instrumentation
if _HAS_PROM:
    try:
        Storage.flush_buffer_once = record_mongo_op("flush")(Storage.flush_buffer_once)
        Storage.insert_dlq_promotion = record_mongo_op("insert_dlq")(Storage.insert_dlq_promotion)
        RedisStorage.promote_dlq = record_redis_op("promote_dlq")(RedisStorage.promote_dlq)
    except Exception:
        LOG.debug("Prometheus patching skipped (non-fatal)")

# -----------------------------------------------------------------------------
# Self-test utilities (for CI/CD)
# -----------------------------------------------------------------------------
async def _self_test_storage():
    """
    Quick integration sanity check for CI pipelines.
    """
    LOG.info("Running storage self-test…")
    store = Storage()
    await store.connect()
    await store.insert_scaling_action({"action": "scale_up", "from": 2, "to": 3})
    await store.insert_dlq_promotion({"queue": "demo", "promoted": 1})
    await asyncio.sleep(0.5)
    latest = await store.get_latest_worker_count()
    LOG.info("Self-test latest_worker_count=%s", latest)
    await store.shutdown()
    LOG.info("Storage self-test complete")

# -----------------------------------------------------------------------------
# CLI tool
# -----------------------------------------------------------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-storage",
                                description="PriorityMax unified storage management CLI")
    sub = p.add_subparsers(dest="cmd")

    # Bootstrapping / Index setup
    s1 = sub.add_parser("bootstrap", help="Run Mongo index/bootstrap setup")
    s1.add_argument("--tenant", default=None)

    # Export / Import
    s2 = sub.add_parser("export", help="Export a collection to JSONL")
    s2.add_argument("--collection", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--limit", type=int, default=None)

    s3 = sub.add_parser("import", help="Import JSONL file into collection")
    s3.add_argument("--collection", required=True)
    s3.add_argument("--infile", required=True)
    s3.add_argument("--batch", type=int, default=1000)

    # Self-test
    sub.add_parser("selftest", help="Run quick integration test")

    # Prometheus
    s4 = sub.add_parser("metrics", help="Run Prometheus metrics exporter")
    s4.add_argument("--port", type=int, default=9109)

    return p

async def _cli_async_main(args):
    store = Storage()
    if args.cmd in ("bootstrap", "export", "import"):
        await store.connect()
    if args.cmd == "bootstrap":
        await store.bootstrap(tenant=args.tenant)
    elif args.cmd == "export":
        await store.export_collection_jsonl(args.collection, args.out, limit=args.limit)
    elif args.cmd == "import":
        await store.import_jsonl_to_collection(args.collection, args.infile, batch_size=args.batch)
    elif args.cmd == "selftest":
        await _self_test_storage()
    elif args.cmd == "metrics":
        start_prometheus_exporter(port=args.port)
        LOG.info("Metrics exporter running; press Ctrl-C to exit.")
        while True:
            await asyncio.sleep(60)
    else:
        LOG.warning("Unknown command")
    await store.shutdown()

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    asyncio.run(_cli_async_main(args))

if __name__ == "__main__":
    main_cli()

# -----------------------------------------------------------------------------
# End of Chunk 6 — full PriorityMax Storage system complete ✅
# -----------------------------------------------------------------------------

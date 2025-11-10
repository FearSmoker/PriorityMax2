# backend/app/api/tasks.py
"""
PriorityMax Tasks API — Chunk 1/7
Initialization, logging, Redis & Mongo setup, Pydantic schemas, enums, and persistence helpers.

Paste chunks 1 → 7 in order to assemble the full api/tasks.py module.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import math
import asyncio
import logging
import pathlib
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body, Query, status
from pydantic import BaseModel, Field, validator

# Optional/third-party: prefer async libs, degrade gracefully
try:
    import aioredis
    _HAS_AIOREDIS = True
except Exception:
    aioredis = None
    _HAS_AIOREDIS = False

try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

# Optional metrics placeholder (actual metrics wired in later chunk)
try:
    from prometheus_client import Counter, Gauge, CollectorRegistry
    _HAS_PROM = True
except Exception:
    Counter = Gauge = CollectorRegistry = None
    _HAS_PROM = False

# Integration placeholders for ML/predictor/rl/autoscaler modules
try:
    from app.ml.predictor import Predictor  # type: ignore
except Exception:
    Predictor = None

try:
    from app.ml.rl_agent import RLAgent  # type: ignore
except Exception:
    RLAgent = None

try:
    from app.autoscaler import Autoscaler  # type: ignore
except Exception:
    Autoscaler = None

# Admin & audit helpers (expected to exist in app.api.admin)
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    # stubs if admin module not present — these will raise on use
    def get_current_user(*a, **k):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
    def require_role(r):
        def _dep(*a, **k):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
        return _dep
    class Role:
        ADMIN = "admin"
        OPERATOR = "operator"
        VIEWER = "viewer"
    async def write_audit_event(evt: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "tasks_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.tasks")
LOG.setLevel(os.getenv("PRIORITYMAX_TASKS_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_handler)

# Base directories & meta
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
TASKS_META_DIR = pathlib.Path(os.getenv("TASKS_META_DIR", str(BASE_DIR / "app" / "tasks_meta")))
TASKS_META_DIR.mkdir(parents=True, exist_ok=True)

_TASKS_FS = TASKS_META_DIR / "tasks.json"        # fallback metadata store
_TASKS_HISTORY_FS = TASKS_META_DIR / "history.jsonl"

# Ensure fallback files exist
if not _TASKS_FS.exists():
    _TASKS_FS.write_text(json.dumps({}), encoding="utf-8")
if not _TASKS_HISTORY_FS.exists():
    _TASKS_HISTORY_FS.write_text("", encoding="utf-8")

# Environment-driven config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MONGO_URL = os.getenv("MONGO_URL", None)
DEFAULT_VISIBILITY_SECONDS = int(os.getenv("TASK_VISIBILITY_SECONDS", "60"))
DEFAULT_MAX_RETRIES = int(os.getenv("TASK_DEFAULT_MAX_RETRIES", "5"))
DEFAULT_DLQ_KEY = os.getenv("TASK_DLQ_KEY", "prioritymax:dlq")
TASKS_QUEUE_KEY_PREFIX = os.getenv("TASK_QUEUE_PREFIX", "prioritymax:queue:")
TASKS_INFLIGHT_KEY_PREFIX = os.getenv("TASK_INFLIGHT_PREFIX", "prioritymax:inflight:")
TASKS_METADATA_COLLECTION = os.getenv("TASKS_METADATA_COLLECTION", "prioritymax_tasks_meta")

# Redis & Mongo async clients (initialized lazily)
_redis_client: Optional[Any] = None
_mongo_client: Optional[Any] = None
_tasks_meta_col = None

async def get_redis():
    """
    Return an aioredis client. Lazy-initializes for reuse.
    If aioredis is not installed or connection fails, returns None (and the system falls back to FS).
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not _HAS_AIOREDIS:
        LOG.info("aioredis not installed — queue persistence will use filesystem fallback")
        return None
    try:
        _redis_client = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        # test ping
        await _redis_client.ping()
        LOG.info("Connected to Redis at %s", REDIS_URL)
        return _redis_client
    except Exception:
        LOG.exception("Failed to connect to Redis; falling back to filesystem queue")
        _redis_client = None
        return None

def get_redis_sync():
    """
    Synchronous accessor for blocking contexts (not used often).
    """
    # intentionally simple: not creating sync redis if aioredis unavailable
    return None

# Mongo client initialization (async Motor)
if _HAS_MOTOR and MONGO_URL:
    try:
        _mongo_client = motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        _tasks_meta_db = _mongo_client.get_default_database()
        _tasks_meta_col = _tasks_meta_db.get_collection(TASKS_METADATA_COLLECTION)
        LOG.info("Tasks metadata: using MongoDB at %s", MONGO_URL)
    except Exception:
        _tasks_meta_col = None
        LOG.exception("Failed to connect to Mongo; using filesystem fallback for metadata")
else:
    _tasks_meta_col = None
    LOG.info("Tasks metadata: using filesystem fallback at %s", _TASKS_FS)

# -------------------------
# Enums & Models
# -------------------------
class TaskStatus(str, Enum):
    PENDING = "pending"
    RESERVED = "reserved"   # dequeued and currently processing (inflight)
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"           # moved to DLQ
    CANCELED = "canceled"

class TaskPriority(int, Enum):
    LOW = 10
    MEDIUM = 50
    HIGH = 100
    CRITICAL = 200

class TaskRetryPolicy(BaseModel):
    max_retries: int = Field(DEFAULT_MAX_RETRIES, ge=0)
    backoff_seconds: Optional[int] = Field(30, ge=0)  # base backoff
    backoff_factor: Optional[float] = Field(2.0, ge=1.0)
    retry_on_exceptions: Optional[List[str]] = Field(default_factory=lambda: [])

class TaskCreate(BaseModel):
    task_type: str = Field(..., description="Logical task type, used for routing")
    payload: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: Optional[str] = Field(None, description="Tenant / org id for multi-tenant isolation")
    priority: TaskPriority = TaskPriority.MEDIUM
    eta: Optional[str] = Field(None, description="ISO timestamp when task becomes visible")
    retries: Optional[int] = Field(0)
    retry_policy: Optional[TaskRetryPolicy] = Field(default_factory=TaskRetryPolicy)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("eta", pre=True, always=False)
    def validate_eta(cls, v):
        if v is None or v == "":
            return None
        try:
            # normalize to ISO Z
            return datetime.datetime.fromisoformat(v.replace("Z", "+00:00")).isoformat() + "Z"
        except Exception:
            raise ValueError("eta must be an ISO8601 timestamp")

class Task(BaseModel):
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    tenant_id: Optional[str]
    priority: TaskPriority
    status: TaskStatus
    created_at: str
    visible_at: str
    attempts: int = 0
    last_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    retry_policy: TaskRetryPolicy

class TaskResult(BaseModel):
    task_id: str
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processed_at: Optional[str] = None
    attempt: Optional[int] = None

# -------------------------
# Persistence helpers (filesystem fallback)
# -------------------------
def _fs_read_tasks() -> Dict[str, Dict[str, Any]]:
    try:
        raw = _TASKS_FS.read_text(encoding="utf-8")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        LOG.exception("Failed to read tasks FS store")
        return {}

def _fs_write_tasks(data: Dict[str, Dict[str, Any]]):
    try:
        tmp = _TASKS_FS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        tmp.replace(_TASKS_FS)
    except Exception:
        LOG.exception("Failed to write tasks FS store")

def _fs_append_history(record: Dict[str, Any]):
    try:
        with open(_TASKS_HISTORY_FS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        LOG.exception("Failed to append to tasks history file")

async def _persist_task_meta(task: Task):
    """
    Persist task metadata to Mongo if available, otherwise to FS.
    """
    data = task.dict()
    if _tasks_meta_col is not None:
        try:
            await _tasks_meta_col.update_one({"task_id": task.task_id}, {"$set": data}, upsert=True)
            return
        except Exception:
            LOG.exception("Mongo write failed; falling back to FS for task metadata")
    # filesystem fallback
    all_tasks = _fs_read_tasks()
    all_tasks[task.task_id] = data
    _fs_write_tasks(all_tasks)

async def _load_task_meta(task_id: str) -> Optional[Task]:
    if _tasks_meta_col is not None:
        try:
            doc = await _tasks_meta_col.find_one({"task_id": task_id})
            if not doc:
                return None
            doc.pop("_id", None)
            return Task(**doc)
        except Exception:
            LOG.exception("Mongo read failed; falling back to FS for task metadata")
    all_tasks = _fs_read_tasks()
    d = all_tasks.get(task_id)
    if not d:
        return None
    try:
        return Task(**d)
    except Exception:
        LOG.exception("Failed to parse task meta from FS for %s", task_id)
        return None

async def _delete_task_meta(task_id: str):
    if _tasks_meta_col is not None:
        try:
            await _tasks_meta_col.delete_one({"task_id": task_id})
            return
        except Exception:
            LOG.exception("Mongo delete failed; falling back to FS deletion")
    all_tasks = _fs_read_tasks()
    if task_id in all_tasks:
        del all_tasks[task_id]
        _fs_write_tasks(all_tasks)

# -------------------------
# Queue key helpers
# -------------------------
def _queue_key_for(task_type: str, tenant_id: Optional[str] = None) -> str:
    # allow per-tenant queues if tenant_id provided
    if tenant_id:
        return f"{TASKS_QUEUE_KEY_PREFIX}{tenant_id}:{task_type}"
    return f"{TASKS_QUEUE_KEY_PREFIX}{task_type}"

def _inflight_key_for(task_type: str, tenant_id: Optional[str] = None) -> str:
    if tenant_id:
        return f"{TASKS_INFLIGHT_KEY_PREFIX}{tenant_id}:{task_type}"
    return f"{TASKS_INFLIGHT_KEY_PREFIX}{task_type}"

# -------------------------
# Router
# -------------------------
router = APIRouter(prefix="/tasks", tags=["tasks"])

# Placeholder for WS broadcast from later chunk
_TASKS_WS_CONNECTIONS: List[Any] = []

async def _broadcast_task_event(event: Dict[str, Any]):
    """
    Placeholder; implemented in later chunk with WebSocket handling.
    """
    try:
        # write audit as minimum
        await write_audit_event({"source": "tasks_event", "event": event, "ts": datetime.datetime.utcnow().isoformat() + "Z"})
    except Exception:
        LOG.exception("tasks audit write failed for event %s", event)

# End of Chunk 1/7
# -------------------------
# Chunk 2/7 — Core Queue Helpers (enqueue, dequeue, ack/fail/retry, DLQ)
# -------------------------

import random
from contextlib import asynccontextmanager

# -------------------------
# Internal helper utilities
# -------------------------
def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def _task_key(task: Task) -> str:
    """Unique redis key for a task (for metadata indexing)."""
    return f"prioritymax:task:{task.task_id}"

def _visibility_deadline(seconds: int) -> str:
    """Compute new visibility time (ISO)."""
    return (datetime.datetime.utcnow() + datetime.timedelta(seconds=seconds)).isoformat() + "Z"

async def _redis_zadd(redis, key: str, score: float, member: str):
    """Safe ZADD wrapper."""
    try:
        await redis.zadd(key, {member: score})
    except Exception:
        LOG.exception("Redis ZADD failed for key %s", key)

# -------------------------
# Enqueue task
# -------------------------
async def _enqueue_task(task: Task, redis=None):
    """
    Add a new task to Redis (sorted set by priority) or filesystem fallback.
    """
    redis = redis or await get_redis()
    score = float(task.priority) + random.random() * 0.0001  # small jitter
    queue_key = _queue_key_for(task.task_type, task.tenant_id)

    if redis:
        try:
            await redis.zadd(queue_key, {task.task_id: score})
            await _persist_task_meta(task)
            _fs_append_history({"event": "enqueue", "task_id": task.task_id, "queue": queue_key, "ts": _now_iso()})
            return True
        except Exception:
            LOG.exception("Redis enqueue failed, falling back to FS")

    # fallback FS mode
    all_tasks = _fs_read_tasks()
    all_tasks[task.task_id] = task.dict()
    _fs_write_tasks(all_tasks)
    _fs_append_history({"event": "enqueue_fs", "task_id": task.task_id, "ts": _now_iso()})
    return True

# -------------------------
# Dequeue task (with visibility timeout)
# -------------------------
async def _dequeue_task(task_type: str, tenant_id: Optional[str] = None, visibility_timeout: int = DEFAULT_VISIBILITY_SECONDS, redis=None) -> Optional[Task]:
    """
    Atomically pop the highest-priority task from queue (lowest score ZRANGE).
    Moves it to inflight queue with visibility timeout.
    """
    redis = redis or await get_redis()
    queue_key = _queue_key_for(task_type, tenant_id)
    inflight_key = _inflight_key_for(task_type, tenant_id)

    if redis:
        try:
            async with redis.pipeline(transaction=True) as pipe:
                # get one lowest-score item
                items = await redis.zrange(queue_key, 0, 0)
                if not items:
                    return None
                task_id = items[0]
                # remove from main queue
                await pipe.zrem(queue_key, task_id)
                # add to inflight with expiry score (timestamp)
                expiry = time.time() + visibility_timeout
                await pipe.zadd(inflight_key, {task_id: expiry})
                await pipe.execute()
            # load task metadata
            task = await _load_task_meta(task_id)
            if not task:
                LOG.warning("Task %s not found in meta after dequeue", task_id)
                return None
            # mark reserved
            task.status = TaskStatus.RESERVED
            task.visible_at = _visibility_deadline(visibility_timeout)
            await _persist_task_meta(task)
            await write_audit_event({"event": "dequeue", "task_id": task_id, "queue": queue_key, "ts": _now_iso()})
            await _broadcast_task_event({"event": "task_reserved", "task_id": task_id, "queue": queue_key})
            return task
        except Exception:
            LOG.exception("Redis dequeue failed; falling back to FS")

    # fallback FS mode (inefficient)
    all_tasks = _fs_read_tasks()
    if not all_tasks:
        return None
    # pick highest-priority pending task
    candidates = [
        Task(**t)
        for t in all_tasks.values()
        if t.get("status") == TaskStatus.PENDING
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x.priority)
    task = candidates[0]
    task.status = TaskStatus.RESERVED
    task.visible_at = _visibility_deadline(visibility_timeout)
    all_tasks[task.task_id] = task.dict()
    _fs_write_tasks(all_tasks)
    _fs_append_history({"event": "dequeue_fs", "task_id": task.task_id, "ts": _now_iso()})
    return task

# -------------------------
# Acknowledge task
# -------------------------
async def _ack_task(task_id: str, redis=None):
    redis = redis or await get_redis()
    if redis:
        try:
            # remove from inflight
            keys = await redis.keys(f"{TASKS_INFLIGHT_KEY_PREFIX}*")
            for k in keys:
                await redis.zrem(k, task_id)
            task = await _load_task_meta(task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                await _persist_task_meta(task)
            await _broadcast_task_event({"event": "task_completed", "task_id": task_id, "ts": _now_iso()})
            await write_audit_event({"event": "ack", "task_id": task_id, "ts": _now_iso()})
            return True
        except Exception:
            LOG.exception("Redis ack failed")
    # FS fallback
    all_tasks = _fs_read_tasks()
    if task_id in all_tasks:
        t = all_tasks[task_id]
        t["status"] = TaskStatus.COMPLETED
        all_tasks[task_id] = t
        _fs_write_tasks(all_tasks)
        _fs_append_history({"event": "ack_fs", "task_id": task_id, "ts": _now_iso()})
        return True
    return False

# -------------------------
# Fail task (with retry policy)
# -------------------------
async def _fail_task(task_id: str, error: str, redis=None):
    redis = redis or await get_redis()
    task = await _load_task_meta(task_id)
    if not task:
        return False
    task.attempts += 1
    task.last_error = error
    rp = task.retry_policy or TaskRetryPolicy()
    if task.attempts < rp.max_retries:
        # schedule retry after backoff
        delay = int(rp.backoff_seconds * (rp.backoff_factor ** (task.attempts - 1)))
        task.status = TaskStatus.PENDING
        task.visible_at = _visibility_deadline(delay)
        await _persist_task_meta(task)
        if redis:
            try:
                await redis.zadd(_queue_key_for(task.task_type, task.tenant_id), {task.task_id: float(task.priority)})
            except Exception:
                LOG.exception("Redis requeue on fail failed")
        _fs_append_history({"event": "retry_scheduled", "task_id": task_id, "delay": delay, "ts": _now_iso()})
        await _broadcast_task_event({"event": "task_retry_scheduled", "task_id": task_id, "delay": delay})
    else:
        # exhausted retries
        task.status = TaskStatus.DEAD
        await _persist_task_meta(task)
        if redis:
            try:
                await redis.lpush(DEFAULT_DLQ_KEY, json.dumps(task.dict(), default=str))
            except Exception:
                LOG.exception("Redis DLQ push failed")
        _fs_append_history({"event": "dead_letter", "task_id": task_id, "ts": _now_iso()})
        await _broadcast_task_event({"event": "task_dead_lettered", "task_id": task_id})
    await write_audit_event({"event": "fail", "task_id": task_id, "error": error, "ts": _now_iso()})
    return True

# -------------------------
# Visibility timeout watchdog
# -------------------------
async def _requeue_expired_inflight(redis=None):
    """
    Scan inflight queues for tasks whose visibility timeout expired and requeue them.
    """
    redis = redis or await get_redis()
    if not redis:
        return 0
    total_requeued = 0
    try:
        keys = await redis.keys(f"{TASKS_INFLIGHT_KEY_PREFIX}*")
        now = time.time()
        for key in keys:
            expired = await redis.zrangebyscore(key, 0, now)
            for task_id in expired:
                # remove from inflight
                await redis.zrem(key, task_id)
                # requeue task
                task = await _load_task_meta(task_id)
                if task and task.status == TaskStatus.RESERVED:
                    task.status = TaskStatus.PENDING
                    await _persist_task_meta(task)
                    await redis.zadd(_queue_key_for(task.task_type, task.tenant_id), {task.task_id: float(task.priority)})
                    total_requeued += 1
                    _fs_append_history({"event": "requeued_expired", "task_id": task_id, "ts": _now_iso()})
                    await _broadcast_task_event({"event": "task_requeued_expired", "task_id": task_id})
        return total_requeued
    except Exception:
        LOG.exception("Failed to requeue expired inflight tasks")
        return 0

# -------------------------
# DLQ management
# -------------------------
async def _get_dlq_items(limit: int = 100, redis=None) -> List[Dict[str, Any]]:
    redis = redis or await get_redis()
    if not redis:
        return []
    try:
        items = await redis.lrange(DEFAULT_DLQ_KEY, 0, limit - 1)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except Exception:
                continue
        return result
    except Exception:
        LOG.exception("Failed to read DLQ")
        return []

async def _purge_dlq(redis=None):
    redis = redis or await get_redis()
    if not redis:
        return 0
    try:
        n = await redis.llen(DEFAULT_DLQ_KEY)
        await redis.delete(DEFAULT_DLQ_KEY)
        _fs_append_history({"event": "dlq_purged", "count": n, "ts": _now_iso()})
        await _broadcast_task_event({"event": "dlq_purged", "count": n})
        return n
    except Exception:
        LOG.exception("Failed to purge DLQ")
        return 0

# End of Chunk 2/7
# -------------------------
# Chunk 3/7 — AI Scheduling (Predictor / RL Agent) + Autoscaler Hooks
# -------------------------

from typing import Callable, Awaitable

# Feature flags (configurable via env)
AI_SCHEDULER_ENABLED = os.getenv("AI_SCHEDULER_ENABLED", "true").lower() in ("1", "true", "yes")
AI_PREDICTOR_INTERVAL = int(os.getenv("AI_PREDICTOR_INTERVAL", "15"))  # seconds between predictor runs
RL_INFERENCE_INTERVAL = int(os.getenv("RL_INFERENCE_INTERVAL", "10"))   # seconds between RL runs
AUTOSCALER_PUB_CHANNEL = os.getenv("AUTOSCALER_PUB_CHANNEL", "prioritymax:autoscaler:hints")

# ML clients (singletons)
_predictor_client: Optional[Any] = None
_rl_agent_client: Optional[Any] = None
_autoscaler_client: Optional[Any] = None

async def init_ml_clients(force_reload: bool = False):
    """
    Initialize Predictor, RLAgent, and Autoscaler instances if available.
    This is safe to call multiple times.
    """
    global _predictor_client, _rl_agent_client, _autoscaler_client
    if _predictor_client is None or force_reload:
        if Predictor is not None:
            try:
                _predictor_client = Predictor()  # assume predictor class has default ctor
                LOG.info("Predictor client initialized")
            except Exception:
                LOG.exception("Failed to init Predictor")
                _predictor_client = None
        else:
            LOG.info("No Predictor implementation found; predictor disabled")
    if _rl_agent_client is None or force_reload:
        if RLAgent is not None:
            try:
                _rl_agent_client = RLAgent()
                LOG.info("RLAgent client initialized")
            except Exception:
                LOG.exception("Failed to init RLAgent")
                _rl_agent_client = None
        else:
            LOG.info("No RLAgent implementation found; RL scheduling disabled")
    if _autoscaler_client is None or force_reload:
        if Autoscaler is not None:
            try:
                _autoscaler_client = Autoscaler()
                LOG.info("Autoscaler client initialized")
            except Exception:
                LOG.exception("Failed to init Autoscaler client")
                _autoscaler_client = None
        else:
            LOG.info("No Autoscaler integration found; will publish hints to Redis channel if available")

async def _predict_queue_metrics() -> Dict[str, Any]:
    """
    Run predictor client to forecast queue demand and latency.
    Returns a dict of { "queue": {"predicted_rate": float, "predicted_latency_s": float, ...}, ... }
    If predictor unavailable, returns empty dict.
    """
    await init_ml_clients()
    try:
        # gather current queue lengths to feed predictor
        redis = await get_redis()
        queues = {}
        if redis:
            try:
                keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
                for k in keys:
                    # optionally parse queue name
                    length = await redis.zcard(k)
                    queues[k] = {"length": length}
            except Exception:
                LOG.exception("Failed to sample redis queues for predictor")
        # call predictor
        if _predictor_client:
            try:
                # predictor may expect a structured input, we forward current snapshot
                forecast = await asyncio.get_event_loop().run_in_executor(None, lambda: _predictor_client.predict_snapshot(queues))
                # expected forecast structure: {queue_key: {"rate": ..., "latency_s": ...}}
                await write_audit_event({"event": "predictor_run", "input_snapshot": queues, "forecast_summary": {k: v for k, v in (list(forecast.items())[:5])}, "ts": _now_iso()})
                return forecast or {}
            except Exception:
                LOG.exception("Predictor call failed")
                return {}
        # no predictor
        return {}
    except Exception:
        LOG.exception("Predictor wrapper error")
        return {}

async def _rl_infer_action(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call RL agent to compute scheduling adjustments.
    Input snapshot is a dict representing current system state (queues, worker counts, recent latencies).
    Returns actions like:
      { "scale": {"queue_key": suggested_consumer_count, ...}, "priority_overrides": {task_type: delta}, "throttle": {...} }
    If RL agent not available, returns defaults (empty).
    """
    await init_ml_clients()
    if not _rl_agent_client:
        return {}
    try:
        # RL agent might be synchronous; run in threadpool
        action = await asyncio.get_event_loop().run_in_executor(None, lambda: _rl_agent_client.infer(snapshot))
        # action validation and normalization
        if not isinstance(action, dict):
            LOG.warning("RL agent returned non-dict action; ignoring")
            return {}
        await write_audit_event({"event": "rl_inference", "snapshot_summary": {k: snapshot.get(k) for k in list(snapshot.keys())[:5]}, "action_summary": action, "ts": _now_iso()})
        return action
    except Exception:
        LOG.exception("RL inference failed")
        return {}

async def _compute_priority_override_for_task(task: Task) -> int:
    """
    Compute new priority for a task using RL suggestions and heuristics.
    Returns new priority integer (TaskPriority-compatible).
    """
    # baseline from task
    base = int(task.priority)
    try:
        # fetch a small system snapshot to pass to RL
        # sample queue sizes and recent latencies
        redis = await get_redis()
        snapshot = {"queues": {}, "task": {"task_type": task.task_type, "tenant_id": task.tenant_id, "attempts": task.attempts}}
        if redis:
            try:
                keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
                for k in keys:
                    snapshot["queues"][k] = {"length": await redis.zcard(k)}
            except Exception:
                LOG.exception("Failed to sample redis inside priority override")
        # call RL
        action = await _rl_infer_action(snapshot)
        overrides = action.get("priority_overrides") if isinstance(action, dict) else {}
        # find override for this task type
        key = f"{task.tenant_id or 'global'}:{task.task_type}"
        override_delta = 0
        if overrides:
            # try direct task_type, then tenant scoped
            override_delta = int(overrides.get(task.task_type, overrides.get(key, 0) or 0))
        new_priority = max(1, min(1000, base + override_delta))
        return new_priority
    except Exception:
        LOG.exception("Failed computing priority override; returning base priority")
        return base

async def _publish_autoscaler_hint(hint: Dict[str, Any]):
    """
    Publish an autoscaler hint: either call Autoscaler client or publish to Redis pub/sub channel.
    Hint example: {"queue": "prioritymax:queue:email", "suggested_consumers": 5, "reason": "predicted_burst"}
    """
    try:
        await init_ml_clients()
        await write_audit_event({"event": "autoscaler_hint", "hint": hint, "ts": _now_iso()})
    except Exception:
        LOG.exception("Failed to audit autoscaler hint")

    # If an Autoscaler client is available, invoke it
    if _autoscaler_client:
        try:
            if hasattr(_autoscaler_client, "apply_hint"):
                # running in executor if blocking
                await asyncio.get_event_loop().run_in_executor(None, lambda: _autoscaler_client.apply_hint(hint))
                return True
        except Exception:
            LOG.exception("Autoscaler client apply_hint failed")

    # fallback: publish to Redis channel so k8s operator or separate service can consume hints
    try:
        redis = await get_redis()
        if redis:
            await redis.publish(AUTOSCALER_PUB_CHANNEL, json.dumps(hint, default=str))
            LOG.debug("Published autoscaler hint to channel %s: %s", AUTOSCALER_PUB_CHANNEL, hint)
            return True
    except Exception:
        LOG.exception("Failed to publish autoscaler hint to redis channel")
    return False

async def _ai_scheduler_iteration():
    """
    Single iteration that runs predictor + RL and publishes autoscaler hints.
    Intended to be called periodically by background loop.
    """
    if not AI_SCHEDULER_ENABLED:
        LOG.debug("AI scheduler disabled by feature flag")
        return
    try:
        # collect a compact snapshot of current queues and inflight counts
        redis = await get_redis()
        snapshot = {"queues": {}, "inflight": {}, "time": _now_iso()}
        if redis:
            try:
                keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
                for k in keys:
                    snapshot["queues"][k] = {"length": await redis.zcard(k)}
                inflight_keys = await redis.keys(f"{TASKS_INFLIGHT_KEY_PREFIX}*")
                for k in inflight_keys:
                    # approximate inflight by zcard
                    snapshot["inflight"][k] = {"count": await redis.zcard(k)}
            except Exception:
                LOG.exception("Failed to sample redis for AI scheduler")
        # predictor forecast
        forecast = await _predict_queue_metrics()
        # merge forecasts into snapshot
        snapshot["forecast"] = forecast
        # run RL inference
        rl_action = await _rl_infer_action(snapshot)
        # parse RL action to produce autoscaler hints
        hints = []
        if rl_action.get("scale"):
            for qk, val in rl_action["scale"].items():
                try:
                    suggested = int(val)
                    hints.append({"queue": qk, "suggested_consumers": suggested, "source": "rl_agent", "ts": _now_iso()})
                except Exception:
                    continue
        # predictor-driven hints (if predictor provides rate > threshold)
        for qk, f in (forecast or {}).items():
            try:
                rate = float(f.get("predicted_rate", 0.0) or 0.0)
                # simple heuristic: suggest consumers = ceil(rate / 10)
                if rate > 5.0:
                    suggested = max(1, int(math.ceil(rate / 10.0)))
                    hints.append({"queue": qk, "suggested_consumers": suggested, "source": "predictor", "ts": _now_iso()})
            except Exception:
                continue
        # deduplicate hints by queue (keep max suggested)
        hint_map: Dict[str, Dict[str, Any]] = {}
        for h in hints:
            q = h.get("queue")
            if not q:
                continue
            if q not in hint_map or (h.get("suggested_consumers", 0) > hint_map[q].get("suggested_consumers", 0)):
                hint_map[q] = h
        # publish hints
        for h in hint_map.values():
            try:
                await _publish_autoscaler_hint(h)
            except Exception:
                LOG.exception("Failed publishing autoscaler hint %s", h)
        # broadcast an aggregated AI decision event
        try:
            await _broadcast_task_event({"event": "ai_scheduler_iteration", "snapshot_summary": {k: snapshot["queues"].get(k, {}) for k in list(snapshot["queues"].keys())[:5]}, "hints": list(hint_map.values()), "ts": _now_iso()})
        except Exception:
            LOG.exception("Failed broadcasting AI scheduler event")
    except Exception:
        LOG.exception("AI scheduler iteration failed")

# -------------------------
# Retrain trigger helpers (safe)
# -------------------------
async def trigger_retrain_predictor(reason: str = "manual", sample_window_hours: int = 24):
    """
    Enqueue or trigger an offline predictor retrain job.
    Implementation is intentionally non-destructive: it creates a retrain 'task' in tasks metadata
    and optionally writes an entry to a retrain queue (Redis list) for a training worker to pick up.
    """
    retrain_id = f"retrain_predictor_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    payload = {"retrain_id": retrain_id, "reason": reason, "sample_window_hours": sample_window_hours, "created_at": _now_iso()}
    # write small metadata record
    try:
        p = TASKS_META_DIR / "retrain_jobs.jsonl"
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        # optional publish to redis retrain channel
        redis = await get_redis()
        if redis:
            await redis.lpush("prioritymax:retrain:predictor", json.dumps(payload))
        await write_audit_event({"event": "trigger_retrain_predictor", "payload": payload})
        return payload
    except Exception:
        LOG.exception("Failed to trigger retrain_predictor")
        return None

async def trigger_retrain_rl(reason: str = "manual", config: Optional[Dict[str, Any]] = None):
    """
    Enqueue RL retraining job similarly.
    """
    retrain_id = f"retrain_rl_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    payload = {"retrain_id": retrain_id, "reason": reason, "config": config or {}, "created_at": _now_iso()}
    try:
        p = TASKS_META_DIR / "retrain_rl.jsonl"
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        redis = await get_redis()
        if redis:
            await redis.lpush("prioritymax:retrain:rl", json.dumps(payload))
        await write_audit_event({"event": "trigger_retrain_rl", "payload": payload})
        return payload
    except Exception:
        LOG.exception("Failed to trigger retrain_rl")
        return None

# -------------------------
# Background AI scheduler loop registration
# -------------------------
_AI_BG_TASK: Optional[asyncio.Task] = None
_AI_BG_RUNNING = False

async def _ai_background_loop(predictor_interval: int = AI_PREDICTOR_INTERVAL):
    """
    Background loop that runs AI scheduler periodically.
    """
    global _AI_BG_RUNNING
    _AI_BG_RUNNING = True
    LOG.info("AI scheduler background loop starting (interval=%ds)", predictor_interval)
    try:
        await init_ml_clients()
        while True:
            try:
                await _ai_scheduler_iteration()
            except Exception:
                LOG.exception("AI scheduler iteration raised")
            await asyncio.sleep(predictor_interval)
    except asyncio.CancelledError:
        LOG.info("AI scheduler background loop cancelled")
    finally:
        _AI_BG_RUNNING = False
        LOG.info("AI scheduler background loop stopped")

def start_ai_scheduler(loop: Optional[asyncio.AbstractEventLoop] = None, interval: int = AI_PREDICTOR_INTERVAL):
    global _AI_BG_TASK
    if _AI_BG_TASK and not _AI_BG_TASK.done():
        return
    loop = loop or asyncio.get_event_loop()
    _AI_BG_TASK = loop.create_task(_ai_background_loop(interval))
    LOG.info("AI scheduler background task scheduled")

def stop_ai_scheduler():
    global _AI_BG_TASK
    if _AI_BG_TASK:
        _AI_BG_TASK.cancel()
        _AI_BG_TASK = None
        LOG.info("AI scheduler background task stopped")

# End of Chunk 3/7
# -------------------------
# Chunk 4/7 — REST Endpoints (Task Lifecycle + Queue Analytics + Prometheus Metrics)
# -------------------------

from fastapi import Request
from fastapi.responses import JSONResponse

# Prometheus metrics initialization
if _HAS_PROM:
    _PROM_REG = CollectorRegistry()
    TASKS_ENQUEUED = Counter("prioritymax_tasks_enqueued_total", "Total number of enqueued tasks", registry=_PROM_REG)
    TASKS_DEQUEUED = Counter("prioritymax_tasks_dequeued_total", "Total number of dequeued tasks", registry=_PROM_REG)
    TASKS_ACKED = Counter("prioritymax_tasks_acked_total", "Total number of acknowledged tasks", registry=_PROM_REG)
    TASKS_FAILED = Counter("prioritymax_tasks_failed_total", "Total number of failed tasks", registry=_PROM_REG)
    TASKS_DLQ_COUNT = Gauge("prioritymax_dlq_items", "Number of items in Dead Letter Queue", registry=_PROM_REG)
    TASKS_QUEUE_LENGTH = Gauge("prioritymax_queue_length", "Tasks per queue", ["queue"], registry=_PROM_REG)
else:
    _PROM_REG = None
    TASKS_ENQUEUED = TASKS_DEQUEUED = TASKS_ACKED = TASKS_FAILED = TASKS_DLQ_COUNT = TASKS_QUEUE_LENGTH = None

# -------------------------
# Task endpoints
# -------------------------
@router.post("/enqueue", dependencies=[Depends(require_role(Role.OPERATOR))])
async def enqueue_task(payload: TaskCreate = Body(...), user=Depends(get_current_user)):
    """
    Enqueue a single task. Applies optional AI-based priority override.
    """
    redis = await get_redis()
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    base_task = Task(
        task_id=task_id,
        task_type=payload.task_type,
        payload=payload.payload,
        tenant_id=payload.tenant_id,
        priority=payload.priority,
        status=TaskStatus.PENDING,
        created_at=_now_iso(),
        visible_at=_now_iso(),
        attempts=payload.retries or 0,
        metadata=payload.metadata or {},
        retry_policy=payload.retry_policy or TaskRetryPolicy()
    )

    # AI-based dynamic priority adjustment
    try:
        new_priority = await _compute_priority_override_for_task(base_task)
        base_task.priority = TaskPriority(new_priority)
    except Exception:
        LOG.exception("Priority override computation failed; keeping base priority")

    await _enqueue_task(base_task, redis)
    if TASKS_ENQUEUED:
        TASKS_ENQUEUED.inc()
    await write_audit_event({
        "event": "enqueue_task",
        "task_id": task_id,
        "user": getattr(user, "username", "unknown"),
        "priority": int(base_task.priority),
        "ts": _now_iso()
    })
    await _broadcast_task_event({"event": "task_enqueued", "task_id": task_id, "priority": int(base_task.priority)})
    return {"ok": True, "task_id": task_id, "priority": int(base_task.priority)}

@router.post("/bulk_enqueue", dependencies=[Depends(require_role(Role.OPERATOR))])
async def bulk_enqueue(tasks: List[TaskCreate], user=Depends(get_current_user)):
    """
    Enqueue multiple tasks in one request.
    """
    redis = await get_redis()
    task_ids = []
    for t in tasks:
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = Task(
            task_id=task_id,
            task_type=t.task_type,
            payload=t.payload,
            tenant_id=t.tenant_id,
            priority=t.priority,
            status=TaskStatus.PENDING,
            created_at=_now_iso(),
            visible_at=_now_iso(),
            attempts=t.retries or 0,
            metadata=t.metadata or {},
            retry_policy=t.retry_policy or TaskRetryPolicy()
        )
        try:
            new_p = await _compute_priority_override_for_task(task)
            task.priority = TaskPriority(new_p)
        except Exception:
            pass
        await _enqueue_task(task, redis)
        task_ids.append(task_id)
    if TASKS_ENQUEUED:
        TASKS_ENQUEUED.inc(len(task_ids))
    await write_audit_event({
        "event": "bulk_enqueue",
        "count": len(task_ids),
        "user": getattr(user, "username", "unknown"),
        "ts": _now_iso()
    })
    await _broadcast_task_event({"event": "bulk_enqueued", "count": len(task_ids)})
    return {"ok": True, "task_ids": task_ids}

@router.post("/dequeue", dependencies=[Depends(require_role(Role.OPERATOR))])
async def dequeue_task(
    task_type: str = Query(...),
    tenant_id: Optional[str] = Query(None),
    visibility_timeout: int = Query(DEFAULT_VISIBILITY_SECONDS),
    user=Depends(get_current_user)
):
    """
    Dequeue next available task for processing.
    """
    redis = await get_redis()
    task = await _dequeue_task(task_type, tenant_id, visibility_timeout, redis)
    if not task:
        return {"ok": False, "message": "No task available"}
    if TASKS_DEQUEUED:
        TASKS_DEQUEUED.inc()
    await write_audit_event({"event": "dequeue_task", "task_id": task.task_id, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    await _broadcast_task_event({"event": "task_dequeued", "task_id": task.task_id})
    return {"ok": True, "task": task.dict()}

@router.post("/ack", dependencies=[Depends(require_role(Role.OPERATOR))])
async def ack_task(task_id: str = Query(...), user=Depends(get_current_user)):
    """
    Acknowledge completion of a task.
    """
    success = await _ack_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    if TASKS_ACKED:
        TASKS_ACKED.inc()
    await write_audit_event({"event": "ack_task", "task_id": task_id, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    return {"ok": True, "task_id": task_id}

@router.post("/fail", dependencies=[Depends(require_role(Role.OPERATOR))])
async def fail_task(task_id: str = Query(...), error: str = Query("unknown_error"), user=Depends(get_current_user)):
    """
    Mark a task as failed; retries automatically handled.
    """
    success = await _fail_task(task_id, error)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    if TASKS_FAILED:
        TASKS_FAILED.inc()
    await write_audit_event({"event": "fail_task", "task_id": task_id, "error": error, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    return {"ok": True, "task_id": task_id}

@router.get("/status/{task_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def task_status(task_id: str):
    """
    Fetch current metadata for a task.
    """
    task = await _load_task_meta(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.delete("/{task_id}", dependencies=[Depends(require_role(Role.ADMIN))])
async def cancel_task(task_id: str, user=Depends(get_current_user)):
    """
    Cancel a pending or reserved task.
    """
    task = await _load_task_meta(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.status = TaskStatus.CANCELED
    await _persist_task_meta(task)
    await _delete_task_meta(task_id)
    await write_audit_event({"event": "cancel_task", "task_id": task_id, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    await _broadcast_task_event({"event": "task_canceled", "task_id": task_id})
    return {"ok": True, "task_id": task_id}

# -------------------------
# Queue Introspection
# -------------------------
@router.get("/queue/length", dependencies=[Depends(require_role(Role.VIEWER))])
async def queue_length():
    redis = await get_redis()
    queues = {}
    if redis:
        try:
            keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
            for k in keys:
                count = await redis.zcard(k)
                queues[k] = count
                if TASKS_QUEUE_LENGTH:
                    TASKS_QUEUE_LENGTH.labels(queue=k).set(count)
        except Exception:
            LOG.exception("Queue length sampling failed")
    else:
        all_tasks = _fs_read_tasks()
        for t in all_tasks.values():
            q = _queue_key_for(t["task_type"], t.get("tenant_id"))
            queues[q] = queues.get(q, 0) + 1
    return queues

@router.get("/queue/snapshot", dependencies=[Depends(require_role(Role.VIEWER))])
async def queue_snapshot():
    """
    Return snapshot of queues, inflight, and DLQ sizes.
    """
    redis = await get_redis()
    snapshot = {"queues": {}, "inflight": {}, "dlq": 0, "time": _now_iso()}
    if redis:
        try:
            keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
            for k in keys:
                snapshot["queues"][k] = {"length": await redis.zcard(k)}
            inflight_keys = await redis.keys(f"{TASKS_INFLIGHT_KEY_PREFIX}*")
            for k in inflight_keys:
                snapshot["inflight"][k] = {"count": await redis.zcard(k)}
            dlq_count = await redis.llen(DEFAULT_DLQ_KEY)
            snapshot["dlq"] = dlq_count
            if TASKS_DLQ_COUNT:
                TASKS_DLQ_COUNT.set(dlq_count)
        except Exception:
            LOG.exception("Snapshot collection failed")
    return snapshot

# -------------------------
# DLQ Inspection
# -------------------------
@router.get("/dlq", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_dlq(limit: int = Query(50)):
    items = await _get_dlq_items(limit)
    if TASKS_DLQ_COUNT:
        TASKS_DLQ_COUNT.set(len(items))
    return {"ok": True, "count": len(items), "items": items}

@router.post("/dlq/purge", dependencies=[Depends(require_role(Role.ADMIN))])
async def purge_dlq(user=Depends(get_current_user)):
    count = await _purge_dlq()
    await write_audit_event({"event": "purge_dlq", "count": count, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    return {"ok": True, "removed": count}

# -------------------------
# Analytics Endpoints
# -------------------------
@router.get("/analytics/throughput", dependencies=[Depends(require_role(Role.VIEWER))])
async def throughput_analytics(hours: int = Query(1)):
    """
    Estimate tasks processed per minute in the last N hours.
    """
    cutoff = time.time() - hours * 3600
    count = 0
    try:
        with open(_TASKS_HISTORY_FS, "r", encoding="utf-8") as fh:
            for line in fh:
                if '"event": "ack"' in line or '"event": "ack_fs"' in line:
                    try:
                        obj = json.loads(line)
                        ts = obj.get("ts")
                        if not ts:
                            continue
                        t = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                        if t >= cutoff:
                            count += 1
                    except Exception:
                        continue
    except Exception:
        pass
    per_min = count / (hours * 60)
    return {"ok": True, "tasks_processed_per_min": round(per_min, 2)}

@router.get("/analytics/failure_rate", dependencies=[Depends(require_role(Role.VIEWER))])
async def failure_rate(hours: int = Query(1)):
    """
    Calculate task failure ratio over a window.
    """
    cutoff = time.time() - hours * 3600
    total = fails = 0
    try:
        with open(_TASKS_HISTORY_FS, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                ev = obj.get("event")
                if ev in ("ack", "ack_fs"):
                    ts = obj.get("ts")
                    if ts and datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() >= cutoff:
                        total += 1
                elif ev in ("fail", "dead_letter"):
                    ts = obj.get("ts")
                    if ts and datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() >= cutoff:
                        fails += 1
    except Exception:
        LOG.exception("Failed to compute failure rate")
    rate = (fails / total) * 100 if total > 0 else 0.0
    return {"ok": True, "failure_rate_percent": round(rate, 2), "window_hours": hours}

# End of Chunk 4/7
# -------------------------
# Chunk 5/7 — WebSocket, Worker Heartbeats, Prometheus scrape endpoint, Live broadcast
# -------------------------

from fastapi import WebSocket, WebSocketDisconnect
from starlette.responses import PlainTextResponse

# WebSocket connections list for tasks events
_TASKS_WS_CONNECTIONS: List[WebSocket] = []

# Optional simple token for WS auth (set via ENV)
TASKS_WS_AUTH_TOKEN = os.getenv("TASKS_WS_AUTH_TOKEN", None)

# Worker registry to track heartbeats
_WORKER_REGISTRY: Dict[str, Dict[str, Any]] = {}  # worker_id -> {"last_seen": ts, "status": {...}}

# WebSocket endpoint for tasks stream
@router.websocket("/ws/tasks")
async def ws_tasks(websocket: WebSocket, token: Optional[str] = None):
    """
    WebSocket for real-time task events.
    If TASKS_WS_AUTH_TOKEN is set, require it as query param 'token'.
    """
    # simple token guard
    if TASKS_WS_AUTH_TOKEN and token != TASKS_WS_AUTH_TOKEN:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    _TASKS_WS_CONNECTIONS.append(websocket)
    LOG.info("Tasks WS connected — %d clients", len(_TASKS_WS_CONNECTIONS))
    try:
        await websocket.send_text(json.dumps({"message": "connected", "time": _now_iso()}))
        while True:
            try:
                msg = await websocket.receive_text()
                # support ping / client subscribe messages (no-op)
                if msg.strip().lower() in ("ping", "keepalive"):
                    await websocket.send_text(json.dumps({"pong": _now_iso()}))
            except WebSocketDisconnect:
                break
            except Exception:
                # ignore and continue
                await asyncio.sleep(0.1)
                continue
    finally:
        try:
            _TASKS_WS_CONNECTIONS.remove(websocket)
        except Exception:
            pass
        LOG.info("Tasks WS disconnected — %d clients", len(_TASKS_WS_CONNECTIONS))

# Broadcast implementation (overrides earlier placeholder)
async def _broadcast_task_event(event: Dict[str, Any]):
    """
    Broadcast event to connected websocket clients (best-effort).
    Also updates lightweight in-memory metrics where appropriate.
    """
    # ensure event has timestamp
    event.setdefault("ts", _now_iso())
    # write audit record for the event
    try:
        await write_audit_event({"source": "tasks_broadcast", "event": event, "ts": event.get("ts")})
    except Exception:
        LOG.exception("Failed to write audit for task event broadcast")

    # update Prometheus counters where applicable
    try:
        ev = event.get("event", "")
        if ev == "task_enqueued" and TASKS_ENQUEUED:
            TASKS_ENQUEUED.inc()
        elif ev == "task_dequeued" and TASKS_DEQUEUED:
            TASKS_DEQUEUED.inc()
        elif ev == "task_completed" and TASKS_ACKED:
            TASKS_ACKED.inc()
        elif ev == "task_failed" and TASKS_FAILED:
            TASKS_FAILED.inc()
    except Exception:
        LOG.exception("Failed updating prom counters for event %s", event.get("event"))

    # send to websocket clients
    stale = []
    data = json.dumps(event, default=str)
    for ws in list(_TASKS_WS_CONNECTIONS):
        try:
            await ws.send_text(data)
        except Exception:
            stale.append(ws)
    for s in stale:
        try:
            _TASKS_WS_CONNECTIONS.remove(s)
        except Exception:
            pass

# -------------------------
# Worker heartbeat endpoint
# -------------------------
class WorkerHeartbeatPayload(BaseModel):
    worker_id: str
    host: Optional[str] = None
    pid: Optional[int] = None
    queue_types: Optional[List[str]] = Field(default_factory=list)
    capacity: Optional[int] = 1
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

@router.post("/worker/heartbeat", dependencies=[Depends(require_role(Role.OPERATOR))])
async def worker_heartbeat(payload: WorkerHeartbeatPayload = Body(...), user = Depends(get_current_user)):
    """
    Workers call this to indicate liveness and capacity.
    This helps detect dead workers and provide inputs to autoscaler / RL.
    """
    now = time.time()
    _WORKER_REGISTRY[payload.worker_id] = {
        "last_seen": now,
        "host": payload.host,
        "pid": payload.pid,
        "queue_types": payload.queue_types or [],
        "capacity": payload.capacity or 1,
        "metadata": payload.metadata or {},
        "reported_at": _now_iso()
    }
    # broadcast worker update
    await _broadcast_task_event({"event": "worker_heartbeat", "worker_id": payload.worker_id, "host": payload.host, "capacity": payload.capacity, "ts": _now_iso()})
    await write_audit_event({"event": "worker_heartbeat", "worker_id": payload.worker_id, "user": getattr(user, "username", "system"), "ts": _now_iso()})
    return {"ok": True, "worker_id": payload.worker_id, "ts": _now_iso()}

@router.get("/worker/list", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_workers():
    """
    Return current worker registry with liveness (last_seen) data.
    """
    res = []
    now = time.time()
    for wid, meta in _WORKER_REGISTRY.items():
        age = now - meta.get("last_seen", 0)
        res.append({"worker_id": wid, "last_seen_seconds": int(age), **meta})
    return {"workers": res, "count": len(res)}

# -------------------------
# Prometheus scrape endpoint for tasks subsystem
# -------------------------
@router.get("/metrics")
async def tasks_metrics():
    """
    Expose Prometheus metrics for tasks subsystem.
    """
    if not _HAS_PROM or _PROM_REG is None:
        raise HTTPException(status_code=404, detail="Prometheus not enabled")
    try:
        # update gauges (queue lengths, dlq) before expose
        try:
            redis = await get_redis()
            if redis and TASKS_QUEUE_LENGTH:
                keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
                for k in keys:
                    try:
                        TASKS_QUEUE_LENGTH.labels(queue=k).set(await redis.zcard(k))
                    except Exception:
                        pass
            if redis and TASKS_DLQ_COUNT:
                try:
                    TASKS_DLQ_COUNT.set(await redis.llen(DEFAULT_DLQ_KEY))
                except Exception:
                    pass
        except Exception:
            LOG.exception("Failed to update gauges before metrics scrape")
        # generate latest metrics text
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        payload = generate_latest(_PROM_REG)
        return PlainTextResponse(payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        LOG.exception("Failed to generate metrics")
        raise HTTPException(status_code=500, detail="Failed to render metrics")

# -------------------------
# Worker health utilities
# -------------------------
async def _prune_stale_workers(threshold_seconds: int = 60):
    """
    Remove workers that have not heartbeated within threshold_seconds.
    """
    now = time.time()
    removed = []
    for wid, meta in list(_WORKER_REGISTRY.items()):
        if now - meta.get("last_seen", 0) > threshold_seconds:
            removed.append(wid)
            _WORKER_REGISTRY.pop(wid, None)
    if removed:
        await _broadcast_task_event({"event": "workers_pruned", "removed": removed, "ts": _now_iso()})
    return removed

# -------------------------
# Small helper: push a task lifecycle event (used by internal code)
# -------------------------
async def _emit_task_lifecycle_event(task_id: str, event_name: str, extra: Optional[Dict[str, Any]] = None):
    payload = {"event": event_name, "task_id": task_id, "ts": _now_iso()}
    if extra:
        payload.update(extra)
    await _broadcast_task_event(payload)

# End of Chunk 5/7
# -------------------------
# Chunk 6/7 — Reporting, Analytics Exports, Admin Utilities
# -------------------------

import gzip
import csv
from starlette.responses import FileResponse

_REPORTS_DIR = TASKS_META_DIR / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/reports", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_task_reports():
    """
    List available generated report files.
    """
    files = []
    for f in sorted(_REPORTS_DIR.glob("*.jsonl.gz"), key=lambda x: x.stat().st_mtime, reverse=True):
        files.append({"file": f.name, "modified": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()})
    return {"reports": files, "count": len(files)}

@router.post("/reports/generate", dependencies=[Depends(require_role(Role.OPERATOR))])
async def generate_task_report(
    hours: int = Query(24),
    format: str = Query("jsonl.gz"),
    upload_s3: bool = Query(False),
    user = Depends(get_current_user)
):
    """
    Generate compressed JSONL report for recent tasks and events.
    """
    cutoff = time.time() - hours * 3600
    outfile = _REPORTS_DIR / f"tasks_report_{int(time.time())}.{format}"
    written = 0
    try:
        if format == "jsonl.gz":
            with gzip.open(outfile, "wt", encoding="utf-8") as gz:
                # include task metadata
                tasks = _fs_read_tasks()
                for t in tasks.values():
                    try:
                        created = datetime.datetime.fromisoformat(t.get("created_at").replace("Z", "+00:00")).timestamp()
                        if created >= cutoff:
                            gz.write(json.dumps(t, default=str) + "\n")
                            written += 1
                    except Exception:
                        continue
                # include history
                with open(_TASKS_HISTORY_FS, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            ev = json.loads(line)
                            ts = datetime.datetime.fromisoformat(ev.get("ts").replace("Z", "+00:00")).timestamp()
                            if ts >= cutoff:
                                gz.write(json.dumps(ev, default=str) + "\n")
                                written += 1
                        except Exception:
                            continue
        elif format == "csv":
            with open(outfile, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["task_id", "type", "status", "priority", "created_at", "attempts", "tenant_id"])
                for t in _fs_read_tasks().values():
                    try:
                        created = datetime.datetime.fromisoformat(t.get("created_at").replace("Z", "+00:00")).timestamp()
                        if created >= cutoff:
                            writer.writerow([
                                t.get("task_id"),
                                t.get("task_type"),
                                t.get("status"),
                                t.get("priority"),
                                t.get("created_at"),
                                t.get("attempts"),
                                t.get("tenant_id")
                            ])
                            written += 1
                    except Exception:
                        continue
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
    except Exception:
        LOG.exception("Failed generating task report")
        raise HTTPException(status_code=500, detail="Report generation failed")

    await write_audit_event({
        "event": "generate_task_report",
        "user": getattr(user, "username", "unknown"),
        "entries": written,
        "file": str(outfile),
        "ts": _now_iso()
    })

    # Optional S3 upload
    uploaded = None
    if upload_s3 and _HAS_BOTO3 and S3_BUCKET:
        try:
            s3 = boto3.client("s3")
            key = f"reports/{outfile.name}"
            s3.upload_file(str(outfile), S3_BUCKET, key)
            uploaded = f"s3://{S3_BUCKET}/{key}"
        except Exception:
            LOG.exception("S3 upload failed")

    return {"ok": True, "entries": written, "file": outfile.name, "uploaded": uploaded}

@router.get("/reports/download/{filename}", dependencies=[Depends(require_role(Role.VIEWER))])
async def download_task_report(filename: str):
    f = _REPORTS_DIR / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(f), filename=f.name)

# -------------------------
# Aggregated analytics summary
# -------------------------
@router.get("/analytics/summary", dependencies=[Depends(require_role(Role.VIEWER))])
async def analytics_summary():
    """
    Returns aggregate stats: total, completed, failed, DLQ count.
    """
    tasks = _fs_read_tasks()
    summary = {"total": len(tasks), "completed": 0, "failed": 0, "pending": 0, "dlq": 0}
    for t in tasks.values():
        s = t.get("status")
        if s == TaskStatus.COMPLETED:
            summary["completed"] += 1
        elif s == TaskStatus.FAILED:
            summary["failed"] += 1
        elif s == TaskStatus.PENDING:
            summary["pending"] += 1
    try:
        redis = await get_redis()
        if redis:
            summary["dlq"] = await redis.llen(DEFAULT_DLQ_KEY)
    except Exception:
        pass
    if TASKS_DLQ_COUNT:
        TASKS_DLQ_COUNT.set(summary["dlq"])
    return summary

@router.post("/analytics/export_s3", dependencies=[Depends(require_role(Role.ADMIN))])
async def export_to_s3(hours: int = Query(24), user = Depends(get_current_user)):
    """
    Exports a recent dataset to S3 directly (shortcut to reports/generate).
    """
    if not _HAS_BOTO3 or not S3_BUCKET:
        raise HTTPException(status_code=400, detail="S3 not configured")
    report = await generate_task_report(hours=hours, upload_s3=True, user=user)
    await write_audit_event({"event": "analytics_export_s3", "file": report.get("file"), "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    return report

# -------------------------
# Admin utilities
# -------------------------
@router.post("/admin/requeue_expired", dependencies=[Depends(require_role(Role.ADMIN))])
async def requeue_expired_tasks(user = Depends(get_current_user)):
    """
    Force manual requeue of expired inflight tasks.
    """
    count = await _requeue_expired_inflight()
    await write_audit_event({"event": "requeue_expired_manual", "count": count, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    await _broadcast_task_event({"event": "manual_requeue_expired", "count": count})
    return {"ok": True, "requeued": count}

@router.post("/admin/clear_all", dependencies=[Depends(require_role(Role.ADMIN))])
async def clear_all_tasks(confirm: bool = Query(False), user = Depends(get_current_user)):
    """
    Dangerous endpoint: purge all queues and metadata. Requires confirm=true.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true required")
    redis = await get_redis()
    if redis:
        try:
            keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
            for k in keys:
                await redis.delete(k)
            keys = await redis.keys(f"{TASKS_INFLIGHT_KEY_PREFIX}*")
            for k in keys:
                await redis.delete(k)
            await redis.delete(DEFAULT_DLQ_KEY)
        except Exception:
            LOG.exception("Failed to purge redis queues")
    # purge metadata files
    try:
        _TASKS_FS.write_text("{}", encoding="utf-8")
        _TASKS_HISTORY_FS.write_text("", encoding="utf-8")
    except Exception:
        LOG.exception("Failed to clear FS stores")
    await write_audit_event({"event": "clear_all_tasks", "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    await _broadcast_task_event({"event": "all_tasks_cleared", "ts": _now_iso()})
    return {"ok": True, "message": "All queues and metadata cleared"}

# End of Chunk 6/7
# -------------------------
# Chunk 7/7 — Background Maintenance, Health Endpoints, Startup/Shutdown, Exports
# -------------------------

_MAINTENANCE_TASK: Optional[asyncio.Task] = None
_MAINTENANCE_RUNNING = False
_MAINTENANCE_INTERVAL = int(os.getenv("TASK_MAINTENANCE_INTERVAL", "60"))  # seconds

async def _maintenance_loop(interval: int = _MAINTENANCE_INTERVAL):
    """
    Periodic cleanup and requeue loop:
      - Requeue expired inflight tasks
      - Prune stale worker heartbeats
      - Emit periodic metrics updates
    """
    global _MAINTENANCE_RUNNING
    _MAINTENANCE_RUNNING = True
    LOG.info("Tasks maintenance loop started (interval=%ds)", interval)
    try:
        while True:
            try:
                # Requeue expired inflight tasks
                requeued = await _requeue_expired_inflight()
                if requeued:
                    await write_audit_event({"event": "maintenance_requeue_expired", "count": requeued, "ts": _now_iso()})
                    await _broadcast_task_event({"event": "maintenance_requeue_expired", "count": requeued})
                # Prune dead workers
                removed = await _prune_stale_workers()
                if removed:
                    await write_audit_event({"event": "maintenance_pruned_workers", "removed": removed, "ts": _now_iso()})
                # Update metrics gauges
                try:
                    redis = await get_redis()
                    if redis and TASKS_QUEUE_LENGTH:
                        keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
                        for k in keys:
                            TASKS_QUEUE_LENGTH.labels(queue=k).set(await redis.zcard(k))
                    if redis and TASKS_DLQ_COUNT:
                        TASKS_DLQ_COUNT.set(await redis.llen(DEFAULT_DLQ_KEY))
                except Exception:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("Maintenance loop iteration failed")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        LOG.info("Tasks maintenance loop cancelled")
    finally:
        _MAINTENANCE_RUNNING = False
        LOG.info("Tasks maintenance loop stopped")

def start_tasks_maintenance(loop: Optional[asyncio.AbstractEventLoop] = None, interval: int = _MAINTENANCE_INTERVAL):
    global _MAINTENANCE_TASK
    if _MAINTENANCE_TASK and not _MAINTENANCE_TASK.done():
        return
    loop = loop or asyncio.get_event_loop()
    _MAINTENANCE_TASK = loop.create_task(_maintenance_loop(interval))
    LOG.info("Tasks maintenance background task scheduled")

def stop_tasks_maintenance():
    global _MAINTENANCE_TASK
    if _MAINTENANCE_TASK:
        _MAINTENANCE_TASK.cancel()
        _MAINTENANCE_TASK = None
        LOG.info("Tasks maintenance background task stopped")

# -------------------------
# Health & diagnostics
# -------------------------
@router.get("/health")
async def tasks_health():
    """
    Simple health check for tasks subsystem.
    """
    redis_ok = False
    mongo_ok = False
    try:
        r = await get_redis()
        if r:
            await r.ping()
            redis_ok = True
    except Exception:
        redis_ok = False
    if _tasks_meta_col is not None:
        try:
            await _tasks_meta_col.estimated_document_count()
            mongo_ok = True
        except Exception:
            mongo_ok = False
    return {
        "ok": True,
        "redis_connected": redis_ok,
        "mongo_connected": mongo_ok,
        "ai_scheduler_running": _AI_BG_RUNNING,
        "maintenance_running": _MAINTENANCE_RUNNING,
        "workers": len(_WORKER_REGISTRY),
        "queues_ws_clients": len(_TASKS_WS_CONNECTIONS)
    }

@router.get("/status", dependencies=[Depends(require_role(Role.VIEWER))])
async def tasks_status():
    """
    High-level subsystem status for dashboard.
    """
    redis = await get_redis()
    queues = {}
    if redis:
        try:
            keys = await redis.keys(f"{TASKS_QUEUE_KEY_PREFIX}*")
            for k in keys:
                queues[k] = await redis.zcard(k)
        except Exception:
            LOG.exception("Failed to read queue lengths for status")
    return {
        "queues": queues,
        "active_workers": len(_WORKER_REGISTRY),
        "ai_scheduler_running": _AI_BG_RUNNING,
        "maintenance_running": _MAINTENANCE_RUNNING,
        "time": _now_iso()
    }

# -------------------------
# Startup / Shutdown hooks
# -------------------------
@router.on_event("startup")
async def _on_tasks_startup():
    try:
        LOG.info("PriorityMax Tasks API starting up...")
        start_ai_scheduler()
        start_tasks_maintenance()
        await _broadcast_task_event({"event": "tasks_api_startup", "ts": _now_iso()})
    except Exception:
        LOG.exception("Tasks API startup failed")

@router.on_event("shutdown")
async def _on_tasks_shutdown():
    try:
        stop_tasks_maintenance()
        stop_ai_scheduler()
        await _broadcast_task_event({"event": "tasks_api_shutdown", "ts": _now_iso()})
        LOG.info("PriorityMax Tasks API shutdown complete")
    except Exception:
        LOG.exception("Tasks API shutdown error")

# -------------------------
# Final exports
# -------------------------
__all__ = [
    "router",
    "start_ai_scheduler",
    "stop_ai_scheduler",
    "start_tasks_maintenance",
    "stop_tasks_maintenance",
    "_enqueue_task",
    "_dequeue_task",
    "_ack_task",
    "_fail_task",
    "_broadcast_task_event",
]

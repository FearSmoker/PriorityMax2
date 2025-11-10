# backend/app/api/chaos.py
"""
PriorityMax Chaos API — Chunk 1/6
Initialization, logging, persistence abstraction, schemas, enums and helpers.

Paste chunks 1 → 6 in order to assemble the full api/chaos.py module.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import enum
import shutil
import asyncio
import logging
import pathlib
import datetime
import base64
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks, status
from pydantic import BaseModel, Field, validator

# Optional features
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

try:
    from prometheus_client import Counter, Gauge, CollectorRegistry
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False
    Counter = Gauge = CollectorRegistry = None

# auth / audit helpers (expected to exist)
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    # fallback stubs (should be replaced by your real admin module)
    def get_current_user(*args, **kwargs):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
    def require_role(role):
        def _dep(*a, **k):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
        return _dep
    class Role:
        ADMIN = "admin"
        OPERATOR = "operator"
        VIEWER = "viewer"
    async def write_audit_event(event: dict):
        # minimal local audit
        p = pathlib.Path.cwd() / "backend" / "logs" / "chaos_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

# try to reuse config store to check chaos.enabled feature flag (if available)
try:
    from app.api.config import _config_store  # type: ignore
    _HAS_CONFIG_STORE = True
except Exception:
    _HAS_CONFIG_STORE = False
    _config_store = None  # type: ignore

# Logging setup
LOG = logging.getLogger("prioritymax.chaos")
LOG.setLevel(os.getenv("PRIORITYMAX_CHAOS_LOG_LEVEL", "INFO"))
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_ch)

# base directories & persistence
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
CHAOS_META_DIR = pathlib.Path(os.getenv("CHAOS_META_DIR", str(BASE_DIR / "app" / "chaos_meta")))
CHAOS_META_DIR.mkdir(parents=True, exist_ok=True)

_CHAOS_EXPERIMENTS_COLLECTION = os.getenv("CHAOS_EXPERIMENTS_COLLECTION", "chaos_experiments")
_CHAOS_ACTIONS_COLLECTION = os.getenv("CHAOS_ACTIONS_COLLECTION", "chaos_actions")

MONGO_URL = os.getenv("MONGO_URL", None)

# Setup storage (Mongo or local JSON files)
if _HAS_MOTOR and MONGO_URL:
    try:
        _mongo_client = motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        _chaos_db = _mongo_client.get_default_database()
        experiments_col = _chaos_db[_CHAOS_EXPERIMENTS_COLLECTION]
        actions_col = _chaos_db[_CHAOS_ACTIONS_COLLECTION]
        LOG.info("Chaos store: using MongoDB at %s", MONGO_URL)
    except Exception:
        experiments_col = actions_col = None
        LOG.exception("Failed connecting to MongoDB; falling back to filesystem")
else:
    experiments_col = actions_col = None
    LOG.info("Chaos store: using filesystem fallback at %s", CHAOS_META_DIR)

# filesystem fallback files
_EXPS_FS = CHAOS_META_DIR / "experiments.json"
_ACTIONS_FS = CHAOS_META_DIR / "actions.jsonl"
# ensure files exist
if not _EXPS_FS.exists():
    _EXPS_FS.write_text(json.dumps({}), encoding="utf-8")
if not _ACTIONS_FS.exists():
    _ACTIONS_FS.write_text("", encoding="utf-8")

# -------------------------------
# Enums & Models
# -------------------------------

class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"

class FaultType(str, enum.Enum):
    KILL_POD = "kill_pod"
    CPU_STRESS = "cpu_stress"
    MEM_STRESS = "mem_stress"
    LATENCY = "latency"           # inject artificial latency to network
    DROP_CONNECTION = "drop_connection"  # drop connection to Redis/Mongo
    PAUSE_WORKER = "pause_worker"
    THROTTLE_INGESTION = "throttle_ingestion"
    CUSTOM_COMMAND = "custom_command"

class ScopeType(str, enum.Enum):
    GLOBAL = "global"
    QUEUE = "queue"
    NODE = "node"
    POD = "pod"
    SERVICE = "service"

class SafetyLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ChaosTarget(BaseModel):
    """
    Defines where to apply the fault.
    Examples:
      {"type": "queue", "name": "email_queue"}
      {"type": "pod", "namespace": "default", "label_selector": "app=worker"}
    """
    type: ScopeType
    name: Optional[str] = None
    namespace: Optional[str] = None
    label_selector: Optional[str] = None
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChaosExperimentCreate(BaseModel):
    """
    Request to create a chaos experiment.
    """
    name: str = Field(..., description="Human friendly name")
    fault: FaultType
    target: ChaosTarget
    duration_seconds: int = Field(..., ge=1, le=60*60*6)  # max 6 hours by default
    intensity: Optional[int] = Field(1, ge=0, le=100)  # interpreted by drivers
    safety_level: SafetyLevel = SafetyLevel.MEDIUM
    dry_run: bool = Field(False)
    cooldown_seconds: int = Field(60, description="policy cooldown after experiment ends")
    created_by: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("duration_seconds")
    def max_duration_guard(cls, v):
        if v <= 0:
            raise ValueError("duration_seconds must be > 0")
        return v

class ChaosExperiment(ChaosExperimentCreate):
    experiment_id: str
    created_at: str
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    version: int = 1

class ChaosActionRecord(BaseModel):
    action_id: str
    experiment_id: str
    action_type: str
    timestamp: str
    details: Dict[str, Any] = Field(default_factory=dict)

class ChaosReportItem(BaseModel):
    experiment_id: str
    name: str
    fault: FaultType
    target: ChaosTarget
    status: ExperimentStatus
    started_at: Optional[str]
    stopped_at: Optional[str]
    created_by: Optional[str]
    duration_seconds: Optional[int]
    result: Optional[Dict[str, Any]] = None

# -------------------------------
# Persistence helpers
# -------------------------------

def _fs_read_experiments() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(_EXPS_FS.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        LOG.exception("Failed reading experiments fs store")
        return {}

def _fs_write_experiments(data: Dict[str, Dict[str, Any]]):
    try:
        tmp = _EXPS_FS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        tmp.replace(_EXPS_FS)
    except Exception:
        LOG.exception("Failed writing experiments fs store")

async def _persist_experiment(exp: ChaosExperiment):
    """
    Persist experiment object to Mongo or local filesystem.
    """
    doc = exp.dict()
    if experiments_col is not None:
        # remove any _id keys and upsert
        await experiments_col.update_one({"experiment_id": exp.experiment_id}, {"$set": doc}, upsert=True)
    else:
        all_ex = _fs_read_experiments()
        all_ex[exp.experiment_id] = doc
        _fs_write_experiments(all_ex)

async def _load_experiment(experiment_id: str) -> Optional[ChaosExperiment]:
    if experiments_col is not None:
        doc = await experiments_col.find_one({"experiment_id": experiment_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return ChaosExperiment(**doc)
    all_ex = _fs_read_experiments()
    d = all_ex.get(experiment_id)
    if not d:
        return None
    return ChaosExperiment(**d)

async def _list_experiments_from_store(status: Optional[List[ExperimentStatus]] = None) -> List[ChaosExperiment]:
    results: List[ChaosExperiment] = []
    if experiments_col is not None:
        query = {}
        if status:
            query["status"] = {"$in": [s.value for s in status]}
        docs = await experiments_col.find(query).to_list(length=1000)
        for d in docs:
            d.pop("_id", None)
            results.append(ChaosExperiment(**d))
        return results
    all_ex = _fs_read_experiments()
    for d in all_ex.values():
        try:
            ce = ChaosExperiment(**d)
            if status and ce.status not in status:
                continue
            results.append(ce)
        except Exception:
            continue
    return results

async def _append_action_record(record: ChaosActionRecord):
    line = json.dumps(record.dict(), default=str)
    if actions_col is not None:
        await actions_col.insert_one(record.dict())
    else:
        with open(_ACTIONS_FS, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

# -------------------------------
# Safety & feature checks
# -------------------------------

async def _is_chaos_enabled() -> bool:
    """
    Check config feature flag chaos.enabled if config store available.
    Defaults to False when unknown.
    """
    try:
        if _HAS_CONFIG_STORE and _config_store is not None:
            cfg = await _config_store.get("feature.chaos_mode", namespace="global")
            if cfg and cfg.value:
                # cfg.value may be dict or boolean
                if isinstance(cfg.value, dict):
                    return bool(cfg.value.get("enabled", False))
                return bool(cfg.value)
        return False
    except Exception:
        LOG.exception("Failed reading chaos mode flag; defaulting to False")
        return False

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

async def _audit(user: Any, action: str, resource: str, details: Optional[dict] = None):
    try:
        evt = {"user": getattr(user, "username", str(user)) if user else "system", "action": action, "resource": resource, "details": details or {}, "timestamp_utc": _now_iso()}
        await write_audit_event(evt)
    except Exception:
        LOG.exception("Audit failed for action %s resource %s", action, resource)

# -------------------------------
# Websocket broadcast placeholder (filled in later chunk)
# -------------------------------
# _CHAOS_WS_CONNECTIONS and broadcast function will be implemented in chunk 4/5,
# but we declare a safe placeholder reference so other chunks can call it now.
_CHAOS_WS_CONNECTIONS: List[Any] = []

async def _broadcast_chaos_event(payload: dict):
    """
    Broadcast chaos event to connected WS clients. Implemented more fully in later chunks.
    Here we try to send to connections in _CHAOS_WS_CONNECTIONS if available.
    """
    try:
        if not _CHAOS_WS_CONNECTIONS:
            return
        data = json.dumps(payload, default=str)
        stale = []
        for ws in list(_CHAOS_WS_CONNECTIONS):
            try:
                await ws.send_text(data)  # WebSocket expected
            except Exception:
                stale.append(ws)
        for s in stale:
            try:
                _CHAOS_WS_CONNECTIONS.remove(s)
            except Exception:
                pass
    except Exception:
        LOG.exception("Broadcast chaos event failed")

# -------------------------------
# Prometheus metrics (optional)
# -------------------------------
if _HAS_PROM:
    _PROM_REG = CollectorRegistry()
    PROM_CH_EXPERIMENTS_TOTAL = Counter("prioritymax_chaos_experiments_total", "Total chaos experiments created", registry=_PROM_REG)
    PROM_CH_ACTIVE = Gauge("prioritymax_chaos_active_experiments", "Active chaos experiments", registry=_PROM_REG)
    PROM_CH_INJECTIONS = Counter("prioritymax_chaos_injections_total", "Total fault injections executed", registry=_PROM_REG)
else:
    PROM_CH_EXPERIMENTS_TOTAL = PROM_CH_ACTIVE = PROM_CH_INJECTIONS = None
    _PROM_REG = None

# -------------------------------
# API Router
# -------------------------------
router = APIRouter(prefix="/chaos", tags=["chaos"])
# -------------------------
# Chunk 2/6 — Experiment CRUD & Lifecycle Management
# -------------------------

from fastapi import Path
from typing import Set

# In-memory runtime state
_RUNNING_EXPERIMENT_TASKS: Dict[str, asyncio.Task] = {}   # experiment_id -> asyncio.Task
_EXPERIMENT_LOCK = asyncio.Lock()
_COOLDOWNS: Dict[str, float] = {}  # experiment key or target -> timestamp when cooldown expires

# -------------------------
# Helper: cooldown checks
# -------------------------
def _cooldown_key_for_experiment(exp: ChaosExperiment) -> str:
    # Use a coarse key to avoid overlapping experiments on same target: e.g., "<type>:<name>"
    tgt = exp.target
    k = f"{tgt.type.value}:{tgt.name or tgt.label_selector or 'any'}"
    return k

def _is_in_cooldown(exp: ChaosExperiment) -> Tuple[bool, Optional[int]]:
    k = _cooldown_key_for_experiment(exp)
    ts = _COOLDOWNS.get(k, 0)
    now = time.time()
    if ts > now:
        remaining = int(ts - now)
        return True, remaining
    return False, None

def _set_cooldown(exp: ChaosExperiment):
    k = _cooldown_key_for_experiment(exp)
    _COOLDOWNS[k] = time.time() + max(0, exp.cooldown_seconds)

# -------------------------
# Internal runner orchestration
# -------------------------
async def _run_experiment_background(exp_id: str):
    """
    Internal background runner. Loads experiment, updates status, executes driver, persists results,
    and ensures auto-stop after duration or on stop request.
    """
    LOG.info("Background run created for experiment %s", exp_id)
    async with _EXPERIMENT_LOCK:
        exp = await _load_experiment(exp_id)
        if not exp:
            LOG.error("Experiment %s disappeared before start", exp_id)
            return
        if exp.status not in (ExperimentStatus.PENDING, ExperimentStatus.STOPPED):
            LOG.warning("Experiment %s in status %s cannot be started", exp_id, exp.status)
            return
        # Safety check: chaos feature flag
        enabled = await _is_chaos_enabled()
        if not enabled and not exp.dry_run:
            exp.status = ExperimentStatus.FAILED
            exp.result = {"error": "chaos_mode_disabled"}
            await _persist_experiment(exp)
            await _audit(None, "start_experiment_blocked", exp.experiment_id, {"reason": "chaos_mode_disabled"})
            return
        # cooldown check
        in_cd, remaining = _is_in_cooldown(exp)
        if in_cd:
            exp.status = ExperimentStatus.FAILED
            exp.result = {"error": "cooldown_active", "remaining_seconds": remaining}
            await _persist_experiment(exp)
            await _audit(None, "start_experiment_blocked", exp.experiment_id, {"reason": "cooldown_active", "remaining_seconds": remaining})
            return

        # mark started
        exp.started_at = _now_iso()
        exp.status = ExperimentStatus.RUNNING
        await _persist_experiment(exp)
        await _audit(None, "experiment_started", exp.experiment_id, {"name": exp.name, "fault": exp.fault})
        # broadcast
        await _broadcast_chaos_event({"event": "experiment_started", "experiment_id": exp.experiment_id, "name": exp.name, "fault": exp.fault, "started_at": exp.started_at})

    # Execute driver
    action_rec_start = ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="execute_start", timestamp=_now_iso(), details={"dry_run": exp.dry_run, "intensity": exp.intensity})
    await _append_action_record(action_rec_start)
    if PROM_CH_EXPERIMENTS_TOTAL:
        try:
            PROM_CH_EXPERIMENTS_TOTAL.inc()
            PROM_CH_ACTIVE.inc()
        except Exception:
            pass

    # find driver function name
    driver_fn_name = f"_driver_{exp.fault.value}"
    driver_fn = globals().get(driver_fn_name)
    result: Dict[str, Any] = {"status": "not_executed"}
    try:
        if exp.dry_run:
            LOG.info("Experiment %s dry-run mode; skipping actual driver", exp.experiment_id)
            result = {"status": "dry_run", "note": "no destructive actions executed"}
        else:
            if not callable(driver_fn):
                LOG.error("No driver implemented for fault %s (expected %s)", exp.fault, driver_fn_name)
                result = {"status": "failed", "error": f"driver_missing:{driver_fn_name}"}
            else:
                # driver may be sync or async
                maybe_coro = driver_fn(exp)
                if asyncio.iscoroutine(maybe_coro):
                    result = await maybe_coro
                else:
                    # run in threadpool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: driver_fn(exp))
        # record action
        action_rec_exec = ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="executed_driver", timestamp=_now_iso(), details={"result": result})
        await _append_action_record(action_rec_exec)
        if PROM_CH_INJECTIONS:
            try:
                PROM_CH_INJECTIONS.inc()
            except Exception:
                pass
    except Exception as e:
        LOG.exception("Driver execution failed for experiment %s: %s", exp.experiment_id, e)
        result = {"status": "failed", "error": str(e)}

    # schedule auto-stop after duration unless stopped earlier
    stop_time = time.time() + int(exp.duration_seconds)
    try:
        while time.time() < stop_time:
            # check if stop requested
            cur = await _load_experiment(exp.experiment_id)
            if not cur or cur.status == ExperimentStatus.STOPPING:
                LOG.info("Experiment %s stop requested or removed; exiting early", exp.experiment_id)
                break
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        LOG.info("Background task for %s cancelled", exp.experiment_id)

    # finalization: attempt to heal / revert driver actions
    try:
        # attempt to call driver revert if available
        revert_fn_name = f"_driver_{exp.fault.value}_revert"
        revert_fn = globals().get(revert_fn_name)
        revert_result = {"status": "no_revert"}
        if exp.dry_run:
            revert_result = {"status": "dry_run_no_revert"}
        else:
            if callable(revert_fn):
                maybe = revert_fn(exp)
                if asyncio.iscoroutine(maybe):
                    revert_result = await maybe
                else:
                    loop = asyncio.get_event_loop()
                    revert_result = await loop.run_in_executor(None, lambda: revert_fn(exp))
        action_rec_revert = ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="revert_driver", timestamp=_now_iso(), details={"revert_result": revert_result})
        await _append_action_record(action_rec_revert)
    except Exception:
        LOG.exception("Revert step failed for experiment %s", exp.experiment_id)

    # finalize experiment record
    async with _EXPERIMENT_LOCK:
        cur = await _load_experiment(exp.experiment_id)
        if cur:
            cur.stopped_at = _now_iso()
            cur.status = ExperimentStatus.COMPLETED if result.get("status") not in ("failed",) else ExperimentStatus.FAILED
            cur.result = result
            cur.version = (cur.version or 1) + 1
            await _persist_experiment(cur)
            await _audit(None, "experiment_completed", cur.experiment_id, {"result": result})
            _set_cooldown(cur)
            if PROM_CH_ACTIVE:
                try:
                    PROM_CH_ACTIVE.set(max(0, (PROM_CH_ACTIVE._value.get() or 1) - 1))
                except Exception:
                    pass
            await _broadcast_chaos_event({"event": "experiment_completed", "experiment_id": cur.experiment_id, "status": cur.status, "result": result})
    # cleanup runtime task entry
    try:
        _RUNNING_EXPERIMENT_TASKS.pop(exp.experiment_id, None)
    except Exception:
        pass

# -------------------------
# API Endpoints: create/list/get/start/stop/delete
# -------------------------

class CreateExperimentResponse(BaseModel):
    experiment_id: str
    status: str

@router.post("/experiments", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_experiment(payload: ChaosExperimentCreate, user = Depends(get_current_user)):
    """
    Create a chaos experiment. Returns experiment_id.
    """
    # safety: max duration cap can also be enforced here
    exp_id = f"exp_{uuid.uuid4().hex[:10]}"
    now = _now_iso()
    exp = ChaosExperiment(**payload.dict(), experiment_id=exp_id, created_at=now, status=ExperimentStatus.PENDING, version=1)
    exp.created_by = getattr(user, "username", "system")
    await _persist_experiment(exp)
    await _audit(user, "create_experiment", exp.experiment_id, {"name": exp.name, "fault": exp.fault, "target": exp.target.dict()})
    if PROM_CH_EXPERIMENTS_TOTAL:
        try:
            PROM_CH_EXPERIMENTS_TOTAL.inc()
        except Exception:
            pass
    return CreateExperimentResponse(experiment_id=exp.experiment_id, status=exp.status)

@router.get("/experiments", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_experiments(status: Optional[List[ExperimentStatus]] = Query(None)):
    """
    List experiments. Optional filter by status (repeatable query params).
    """
    st_list = [s for s in status] if status else None
    exps = await _list_experiments_from_store(status=st_list)
    return [e.dict() for e in exps]

@router.get("/experiments/{experiment_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_experiment(experiment_id: str = Path(...)):
    exp = await _load_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp

@router.post("/experiments/{experiment_id}/start", dependencies=[Depends(require_role(Role.OPERATOR))])
async def start_experiment(experiment_id: str, background_tasks: BackgroundTasks = None, user = Depends(get_current_user)):
    """
    Start an existing experiment. Launches background runner which will execute the driver, monitor TTL and auto-stop.
    """
    exp = await _load_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.status == ExperimentStatus.RUNNING:
        return {"ok": True, "message": "Experiment already running", "experiment_id": experiment_id}
    # Update status to pending->running and persist
    exp.status = ExperimentStatus.PENDING
    exp.started_at = None
    await _persist_experiment(exp)

    # schedule background runner
    loop = asyncio.get_event_loop()
    task = loop.create_task(_run_experiment_background(experiment_id))
    _RUNNING_EXPERIMENT_TASKS[experiment_id] = task
    await _audit(user, "start_experiment", experiment_id, {"started_by": getattr(user, "username", "system")})
    await _broadcast_chaos_event({"event": "start_requested", "experiment_id": experiment_id})
    return {"ok": True, "experiment_id": experiment_id, "scheduled": True}

@router.post("/experiments/{experiment_id}/stop", dependencies=[Depends(require_role(Role.OPERATOR))])
async def stop_experiment(experiment_id: str, user = Depends(get_current_user)):
    """
    Signal an experiment to stop. The runner will detect STOPPING status and attempt revert.
    """
    exp = await _load_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.status not in (ExperimentStatus.RUNNING, ExperimentStatus.PENDING):
        return {"ok": True, "message": f"Experiment already in status {exp.status}", "experiment_id": experiment_id}
    exp.status = ExperimentStatus.STOPPING
    await _persist_experiment(exp)
    await _append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=experiment_id, action_type="stop_requested", timestamp=_now_iso(), details={"requested_by": getattr(user, "username", "system")} ))
    await _audit(user, "stop_experiment", experiment_id, {"requested_by": getattr(user, "username", "system")})
    # attempt to cancel the running task if exists
    task = _RUNNING_EXPERIMENT_TASKS.get(experiment_id)
    if task and not task.done():
        try:
            task.cancel()
        except Exception:
            LOG.exception("Failed to cancel task for experiment %s", experiment_id)
    await _broadcast_chaos_event({"event": "stop_requested", "experiment_id": experiment_id})
    return {"ok": True, "experiment_id": experiment_id, "stopping": True}

@router.delete("/experiments/{experiment_id}", dependencies=[Depends(require_role(Role.ADMIN))])
async def delete_experiment(experiment_id: str, user = Depends(get_current_user)):
    """
    Delete an experiment record (does not attempt to revert running actions).
    Use stop_experiment first to request stop.
    """
    # stop if running
    try:
        await stop_experiment(experiment_id, user)
    except Exception:
        pass
    # remove from store
    if experiments_col is not None:
        await experiments_col.delete_one({"experiment_id": experiment_id})
    else:
        exs = _fs_read_experiments()
        if experiment_id in exs:
            del exs[experiment_id]
            _fs_write_experiments(exs)
    await _audit(user, "delete_experiment", experiment_id, {})
    await _broadcast_chaos_event({"event": "experiment_deleted", "experiment_id": experiment_id})
    return {"ok": True, "deleted": experiment_id}

@router.get("/experiments/{experiment_id}/actions", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_experiment_actions(experiment_id: str, limit: int = Query(100)):
    """
    Return the action records for the experiment (from actions.jsonl or DB).
    """
    records = []
    if actions_col is not None:
        docs = await actions_col.find({"experiment_id": experiment_id}).sort("timestamp", -1).to_list(length=limit)
        for d in docs:
            d.pop("_id", None)
            records.append(d)
        return records
    # read actions file and filter
    if _ACTIONS_FS.exists():
        with open(_ACTIONS_FS, "r", encoding="utf-8") as fh:
            for line in reversed(fh.read().splitlines()):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get("experiment_id") == experiment_id:
                        records.append(r)
                        if len(records) >= limit:
                            break
                except Exception:
                    continue
    return records

# -------------------------
# Utility: quick dry-run trigger endpoint for safety testing
# -------------------------
@router.post("/experiments/{experiment_id}/dry_run", dependencies=[Depends(require_role(Role.OPERATOR))])
async def dry_run_experiment(experiment_id: str, user = Depends(get_current_user)):
    """
    Execute an experiment in dry-run mode (no destructive actions). Useful for previewing sequences.
    This will run the same orchestration but with dry_run=True override.
    """
    exp = await _load_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    # clone as ephemeral experiment with dry_run set
    tmp_id = f"dry_{uuid.uuid4().hex[:8]}"
    tmp = ChaosExperiment(**exp.dict())
    tmp.experiment_id = tmp_id
    tmp.dry_run = True
    tmp.created_at = _now_iso()
    tmp.started_at = None
    tmp.stopped_at = None
    tmp.status = ExperimentStatus.PENDING
    await _persist_experiment(tmp)
    # schedule run
    loop = asyncio.get_event_loop()
    task = loop.create_task(_run_experiment_background(tmp_id))
    _RUNNING_EXPERIMENT_TASKS[tmp_id] = task
    await _audit(user, "dry_run_started", experiment_id, {"dry_experiment_id": tmp_id})
    return {"ok": True, "dry_experiment_id": tmp_id}

# End of Chunk 2/6
# -------------------------
# Chunk 3/6 — Local Fault Injection Drivers (CPU, Memory, Pause Worker, Drop Connection)
# -------------------------

import multiprocessing
import signal
import subprocess
import threading

# Optional libs
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

try:
    import redis
    _HAS_REDIS = True
except Exception:
    redis = None
    _HAS_REDIS = False

# Safety gate for destructive actions
_CHAOS_ALLOW_DESTRUCTIVE = os.getenv("CHAOS_ALLOW_DESTRUCTIVE", "false").lower() in ("1", "true", "yes")

# runtime map for driver subprocesses and metadata
_DRIVER_PROCS: Dict[str, List[Dict[str, Any]]] = {}  # experiment_id -> list of {"proc": Process/Thread/subproc, "type": "...", "meta": {...}}

def _register_driver_proc(exp_id: str, entry: Dict[str, Any]):
    lst = _DRIVER_PROCS.setdefault(exp_id, [])
    lst.append(entry)

def _clear_driver_procs(exp_id: str):
    try:
        _DRIVER_PROCS.pop(exp_id, None)
    except KeyError:
        pass

def _terminate_driver_procs(exp_id: str):
    entries = _DRIVER_PROCS.get(exp_id, [])[:]
    for e in entries:
        proc = e.get("proc")
        try:
            # multiprocessing.Process
            if isinstance(proc, multiprocessing.Process):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
            # subprocess.Popen
            elif isinstance(proc, subprocess.Popen):
                proc.kill()
                proc.wait(timeout=5)
            # thread — can't forcibly stop; rely on cooperative flag
            elif isinstance(proc, threading.Thread):
                # try to set flag in meta for cooperative stop
                meta = e.get("meta", {})
                meta["stop_requested"] = True
            else:
                # unknown type; attempt to call .terminate / .kill
                if hasattr(proc, "terminate"):
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        except Exception:
            LOG.exception("Failed to terminate driver proc for exp %s entry %s", exp_id, e)
    # finally clear
    _clear_driver_procs(exp_id)

# -------------------------
# Helper functions for stress processes
# -------------------------
def _cpu_busy_loop(duration: int, intensity: int, stop_flag_path: Optional[str] = None):
    """
    CPU-bound busy loop that runs for `duration` seconds.
    intensity parameter is ignored inside this simple loop, but could be used to spin/nap ratio.
    This function is run in a separate process.
    """
    end = time.time() + duration
    try:
        while time.time() < end:
            x = 0
            # do some math to keep CPU busy
            for i in range(10000):
                x += i ^ (i << 1)
            # cooperative stop via flag file
            if stop_flag_path and os.path.exists(stop_flag_path):
                break
    except KeyboardInterrupt:
        pass

def _alloc_memory_mb(megabytes: int, duration: int, stop_flag_path: Optional[str] = None):
    """
    Allocate lists to consume approximate megabytes and hold for duration.
    Runs in a separate process.
    """
    try:
        allocated = []
        mb_chunk = b"x" * 1024 * 1024  # 1 MB chunk
        for n in range(megabytes):
            allocated.append(mb_chunk)
            # small sleep to avoid OOM too fast
            time.sleep(0.01)
            if stop_flag_path and os.path.exists(stop_flag_path):
                break
        # hold for duration or until stop flag
        end = time.time() + duration
        while time.time() < end:
            if stop_flag_path and os.path.exists(stop_flag_path):
                break
            time.sleep(0.5)
    except Exception:
        pass

# -------------------------
# Driver: CPU stress
# -------------------------
def _driver_cpu_stress(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Spawn N worker processes that perform busy computation for exp.duration_seconds.
    intensity (1-100) may be used to determine number of processes: intensity% of CPU count.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "cpu_stress not executed in dry run"}

    if not _CHAOS_ALLOW_DESTRUCTIVE:
        LOG.warning("Destructive actions disabled; set CHAOS_ALLOW_DESTRUCTIVE=true to allow CPU stress")
        return {"status": "blocked", "reason": "destructive_disabled"}

    cpu_count = multiprocessing.cpu_count() or 1
    # determine number of processes from intensity (e.g., intensity 50 -> half of cpus)
    proc_count = max(1, int(round(cpu_count * (max(1, exp.intensity) / 100.0))))
    stop_flag = str(CHAOS_META_DIR / f"{exp.experiment_id}.stop")
    # ensure any previous stop flag removed
    try:
        if os.path.exists(stop_flag):
            os.remove(stop_flag)
    except Exception:
        pass

    procs = []
    for i in range(proc_count):
        p = multiprocessing.Process(target=_cpu_busy_loop, args=(int(exp.duration_seconds), int(exp.intensity), stop_flag))
        p.daemon = True
        p.start()
        procs.append({"proc": p, "type": "cpu_stress", "meta": {"pid": p.pid}})
        _register_driver_proc(exp.experiment_id, procs[-1])

    LOG.info("Started %d cpu_stress processes for experiment %s", len(procs), exp.experiment_id)
    return {"status": "started", "proc_count": len(procs), "proc_pids": [p["meta"]["pid"] for p in procs]}

def _driver_cpu_stress_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Attempt to stop cpu stress processes started by this experiment.
    """
    try:
        # create stop flag to let processes exit cooperatively
        stop_flag = str(CHAOS_META_DIR / f"{exp.experiment_id}.stop")
        with open(stop_flag, "w", encoding="utf-8") as fh:
            fh.write("stop")
    except Exception:
        pass
    # wait briefly, then force terminate
    time.sleep(1)
    _terminate_driver_procs(exp.experiment_id)
    # remove stop flag
    try:
        if os.path.exists(stop_flag):
            os.remove(stop_flag)
    except Exception:
        pass
    LOG.info("Reverted cpu stress for experiment %s", exp.experiment_id)
    return {"status": "reverted"}

# -------------------------
# Driver: Memory stress
# -------------------------
def _driver_mem_stress(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Allocate memory in MB equal to intensity%*available_memory (capped) or intensity MB.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "mem_stress not executed in dry run"}

    if not _CHAOS_ALLOW_DESTRUCTIVE:
        LOG.warning("Destructive actions disabled; set CHAOS_ALLOW_DESTRUCTIVE=true to allow mem stress")
        return {"status": "blocked", "reason": "destructive_disabled"}

    # compute target MB
    target_mb = int(max(1, exp.intensity))  # by default treat intensity as MB if psutil absent
    try:
        if _HAS_PSUTIL:
            mem = psutil.virtual_memory()
            # use intensity as percent of total
            if exp.intensity and exp.intensity <= 100:
                target_mb = int((mem.total / (1024 * 1024)) * (exp.intensity / 100.0))
                # cap to 80% of available to avoid OOM
                target_mb = int(min(target_mb, max(1, (mem.available / (1024 * 1024)) * 0.8)))
    except Exception:
        LOG.exception("psutil memory calculation failed; falling back to intensity MB")

    # spawn process to allocate
    p = multiprocessing.Process(target=_alloc_memory_mb, args=(target_mb, int(exp.duration_seconds), str(CHAOS_META_DIR / f"{exp.experiment_id}.stop")))
    p.daemon = True
    p.start()
    _register_driver_proc(exp.experiment_id, {"proc": p, "type": "mem_stress", "meta": {"pid": p.pid, "target_mb": target_mb}})
    LOG.info("Started mem_stress process pid=%s target_mb=%s for exp=%s", p.pid, target_mb, exp.experiment_id)
    return {"status": "started", "pid": p.pid, "target_mb": target_mb}

def _driver_mem_stress_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    try:
        stop_flag = str(CHAOS_META_DIR / f"{exp.experiment_id}.stop")
        with open(stop_flag, "w", encoding="utf-8") as fh:
            fh.write("stop")
    except Exception:
        pass
    time.sleep(1)
    _terminate_driver_procs(exp.experiment_id)
    try:
        if os.path.exists(stop_flag):
            os.remove(stop_flag)
    except Exception:
        pass
    LOG.info("Reverted mem stress for experiment %s", exp.experiment_id)
    return {"status": "reverted"}

# -------------------------
# Driver: Pause worker (SIGSTOP / SIGCONT)
# -------------------------
def _driver_pause_worker(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Pause worker processes by sending SIGSTOP to processes matching label_selector or name.
    This requires psutil. If psutil is not available, returns not_implemented.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "pause_worker not executed in dry run"}

    if not _HAS_PSUTIL:
        LOG.warning("psutil not available; cannot pause workers")
        return {"status": "blocked", "reason": "psutil_missing"}

    selector = exp.target.label_selector or exp.target.name
    if not selector:
        LOG.warning("No selector provided to pause_worker")
        return {"status": "blocked", "reason": "no_selector"}

    matched = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = p.info.get("name", "") or ""
            cmd = " ".join(p.info.get("cmdline") or [])
            if selector in name or selector in cmd:
                try:
                    p.suspend()
                    matched.append({"pid": p.pid, "name": name, "cmdline": cmd})
                    _register_driver_proc(exp.experiment_id, {"proc": p, "type": "pause_worker", "meta": {"pid": p.pid}})
                except Exception:
                    LOG.exception("Failed to suspend process %s", p.pid)
        except Exception:
            continue
    LOG.info("Paused %d processes matching selector '%s' for exp %s", len(matched), selector, exp.experiment_id)
    return {"status": "paused", "matched": matched}

def _driver_pause_worker_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    # resume any registered psutil processes
    entries = _DRIVER_PROCS.get(exp.experiment_id, [])[:]
    resumed = []
    for e in entries:
        if e.get("type") == "pause_worker":
            proc = e.get("proc")
            try:
                # psutil.Process instance may be stale if read from earlier; try by pid
                if isinstance(proc, psutil.Process):
                    proc.resume()
                    resumed.append({"pid": proc.pid})
                else:
                    pid = e.get("meta", {}).get("pid")
                    if pid:
                        try:
                            p = psutil.Process(pid)
                            p.resume()
                            resumed.append({"pid": pid})
                        except Exception:
                            pass
            except Exception:
                LOG.exception("Failed to resume process entry %s", e)
    # clear entries
    _terminate_driver_procs(exp.experiment_id)
    LOG.info("Resumed %d paused processes for exp %s", len(resumed), exp.experiment_id)
    return {"status": "resumed", "resumed": resumed}

# -------------------------
# Driver: Drop connection (Redis)
# -------------------------
def _driver_drop_connection(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    For Redis, uses CLIENT PAUSE to block server processing for the given duration (ms).
    This is less-destructive than firewall rules. Requires redis-py.
    If redis isn't available, returns not_implemented.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "drop_connection not executed in dry run"}

    if not _CHAOS_ALLOW_DESTRUCTIVE:
        LOG.warning("Destructive actions disabled; set CHAOS_ALLOW_DESTRUCTIVE=true to allow drop_connection")
        return {"status": "blocked", "reason": "destructive_disabled"}

    target = exp.target
    # expect target.extra to include {"redis_url": "redis://..."} or label to resolve
    redis_url = target.extra.get("redis_url") if target.extra else None
    if not redis_url and not _HAS_REDIS:
        LOG.warning("redis not configured or redis lib missing")
        return {"status": "blocked", "reason": "redis_missing"}
    try:
        if _HAS_REDIS:
            r = redis.from_url(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            ms = int(exp.duration_seconds * 1000)
            # CLIENT PAUSE will block all clients for ms - ensure operator aware
            r.execute_command("CLIENT", "PAUSE", ms)
            _append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="redis_client_pause", timestamp=_now_iso(), details={"ms": ms, "redis_url": redis_url}))
            # nothing to register for revert; Redis will resume automatically after ms
            LOG.info("Issued CLIENT PAUSE for %d ms on %s", ms, redis_url or "default")
            return {"status": "paused", "ms": ms}
        else:
            return {"status": "blocked", "reason": "redis_lib_missing"}
    except Exception as e:
        LOG.exception("Failed to execute redis client pause: %s", e)
        return {"status": "failed", "error": str(e)}

def _driver_drop_connection_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    # for CLIENT PAUSE revert is automatic; nothing to do
    return {"status": "noop", "note": "redis CLIENT PAUSE auto-reverts after duration"}

# -------------------------
# Driver: throttle_ingestion (placeholder)
# -------------------------
def _driver_throttle_ingestion(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Placeholder for ingestion throttling. Implementation is environment-specific.
    Could implement by adding config flag that workers check, or by applying rate-limiting rules.
    """
    LOG.info("throttle_ingestion requested for exp %s but not implemented in local driver", exp.experiment_id)
    # record action
    _append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="throttle_ingestion_requested", timestamp=_now_iso(), details={"intensity": exp.intensity}))
    return {"status": "not_implemented", "note": "use worker-level configuration or gateway rate limiter to effect ingestion throttle"}

def _driver_throttle_ingestion_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    return {"status": "noop"}

# -------------------------
# Driver: custom_command (placeholder, runs provided shell command)
# -------------------------
def _driver_custom_command(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Executes a custom shell command provided in target.extra['command'].
    THIS IS POTENTIALLY DANGEROUS. Only allowed if CHAOS_ALLOW_DESTRUCTIVE is true.
    """
    cmd = exp.target.extra.get("command") if exp.target and exp.target.extra else None
    if not cmd:
        return {"status": "blocked", "reason": "no_command_provided"}
    if exp.dry_run:
        return {"status": "dry_run", "command": cmd}
    if not _CHAOS_ALLOW_DESTRUCTIVE:
        return {"status": "blocked", "reason": "destructive_disabled"}
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _register_driver_proc(exp.experiment_id, {"proc": p, "type": "custom_command", "meta": {"cmd": cmd, "pid": p.pid}})
        out, err = p.communicate(timeout=exp.duration_seconds + 5)
        return {"status": "executed", "returncode": p.returncode, "stdout": (out.decode("utf-8", errors="ignore")[:2000] if out else ""), "stderr": (err.decode("utf-8", errors="ignore")[:2000] if err else "")}
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        return {"status": "timeout"}
    except Exception as e:
        LOG.exception("custom command failed: %s", e)
        return {"status": "failed", "error": str(e)}

def _driver_custom_command_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    # attempt to kill any registered custom_command procs
    _terminate_driver_procs(exp.experiment_id)
    return {"status": "reverted"}

# -------------------------
# Unknown driver fallback
# -------------------------
def _driver_unknown(exp: ChaosExperiment) -> Dict[str, Any]:
    LOG.error("Unknown driver requested: %s", exp.fault)
    return {"status": "failed", "error": "unknown_driver"}

# End of Chunk 3/6
# -------------------------
# Chunk 4/6 — Kubernetes-mode Drivers (pod kill, cpu stress, network latency, cordon/uncordon)
# -------------------------

import shlex
import tempfile
import inspect
from typing import cast

# Optional Kubernetes client
try:
    from kubernetes import client as k8s_client, config as k8s_config, utils as k8s_utils
    _HAS_K8S = True
    # attempt to load in-cluster or kubeconfig lazily in driver
except Exception:
    k8s_client = None
    k8s_config = None
    k8s_utils = None
    _HAS_K8S = False

# Helper: run kubectl command as fallback
def _run_kubectl(cmd: str, timeout: int = 30) -> Tuple[int, str, str]:
    """
    Execute a kubectl command string. Returns (returncode, stdout, stderr).
    """
    try:
        parts = shlex.split(cmd)
        p = subprocess.Popen(parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out.decode("utf-8", errors="ignore"), err.decode("utf-8", errors="ignore")
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)

# Helper: annotate / label pods (used for marking experiments)
def _k8s_label_pods(namespace: str, selector: str, labels: Dict[str, str]) -> Dict[str, Any]:
    """
    Attempts to label pods matching selector with given labels.
    """
    out = {"applied": 0, "errors": []}
    label_str = ",".join([f"{k}={v}" for k, v in labels.items()])
    if _HAS_K8S:
        try:
            # lazy load config
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            v1 = k8s_client.CoreV1Api()
            pods = v1.list_namespaced_pod(namespace=namespace, label_selector=selector).items
            for p in pods:
                name = p.metadata.name
                body = {"metadata": {"labels": {**(p.metadata.labels or {}), **labels}}}
                v1.patch_namespaced_pod(name, namespace, body)
                out["applied"] += 1
            return out
        except Exception as e:
            out["errors"].append(str(e))
            return out
    # kubectl fallback
    cmd = f"kubectl -n {namespace} label pods -l '{selector}' {label_str} --overwrite"
    rc, so, se = _run_kubectl(cmd)
    if rc == 0:
        out["applied"] = so.count("\n")
    else:
        out["errors"].append(se or so)
    return out

# -------------------------
# K8s Driver: kill_pod
# -------------------------
def _driver_kill_pod(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Delete pods matching the target selector. This will trigger K8s to reschedule pods if Deployment/ReplicaSet exists.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "kill_pod not executed in dry run"}

    if not _CHAOS_ALLOW_DESTRUCTIVE:
        LOG.warning("Destructive actions disabled; kill_pod blocked")
        return {"status": "blocked", "reason": "destructive_disabled"}

    tgt = exp.target
    namespace = tgt.namespace or "default"
    selector = tgt.label_selector or tgt.name
    if not selector:
        return {"status": "blocked", "reason": "no_selector_provided"}

    # label pods for UI tracing
    label_res = _k8s_label_pods(namespace, selector, {"prioritymax-chaos": exp.experiment_id})
    try:
        if _HAS_K8S:
            try:
                # attempt to load config
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1 = k8s_client.CoreV1Api()
                pods = v1.list_namespaced_pod(namespace=namespace, label_selector=selector).items
                deleted = []
                for p in pods:
                    name = p.metadata.name
                    v1.delete_namespaced_pod(name, namespace, body=k8s_client.V1DeleteOptions(grace_period_seconds=30))
                    deleted.append(name)
                    # record action per-pod
                    awaitable = _append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="k8s_delete_pod", timestamp=_now_iso(), details={"pod": name, "namespace": namespace} ))
                    # schedule append asynchronously (safe to fire and forget)
                    try:
                        asyncio.get_event_loop().create_task(awaitable)
                    except Exception:
                        pass
                return {"status": "deleted", "pods": deleted, "labeling": label_res}
            except Exception as e:
                LOG.exception("K8s client delete failed: %s", e)
                # fallback to kubectl
        # kubectl fallback
        cmd = f"kubectl -n {namespace} delete pods -l '{selector}' --grace-period=30"
        rc, so, se = _run_kubectl(cmd, timeout=60)
        if rc != 0:
            return {"status": "failed", "error": se or so, "labeling": label_res}
        deleted = [ln for ln in so.splitlines() if ln.strip()]
        # record single action
        awaitable = _append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="kubectl_delete_pods", timestamp=_now_iso(), details={"cmd": cmd, "output": deleted}))
        try:
            asyncio.get_event_loop().create_task(awaitable)
        except Exception:
            pass
        return {"status": "deleted", "deleted_lines": deleted, "labeling": label_res}
    except Exception as e:
        LOG.exception("kill_pod driver error: %s", e)
        return {"status": "failed", "error": str(e)}

def _driver_kill_pod_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    After deletion, K8s controllers will usually recreate pods. Revert is a noop, but we remove labels added for UI.
    """
    tgt = exp.target
    namespace = tgt.namespace or "default"
    selector = tgt.label_selector or tgt.name or ""
    try:
        # remove label by setting it to empty value
        _k8s_label_pods(namespace, selector, {"prioritymax-chaos": None})
    except Exception:
        LOG.exception("Failed to remove labels during revert")
    return {"status": "reverted_label_cleanup"}

# -------------------------
# K8s Driver: cpu_stress (via 'stress' or 'stress-ng' inside pod)
# -------------------------
def _driver_cpu_stress_pod(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Execute CPU stress inside matching pods by running a command like:
      kubectl exec -n <ns> <pod> -- stress --cpu N --timeout <s>
    This requires the stress/stress-ng binary to be present inside the container image.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "cpu_stress_pod not executed in dry run"}
    if not _CHAOS_ALLOW_DESTRUCTIVE:
        return {"status": "blocked", "reason": "destructive_disabled"}

    tgt = exp.target
    namespace = tgt.namespace or "default"
    selector = tgt.label_selector or tgt.name
    if not selector:
        return {"status": "blocked", "reason": "no_selector"}

    # find pod names
    pods = []
    try:
        if _HAS_K8S:
            try:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1 = k8s_client.CoreV1Api()
                items = v1.list_namespaced_pod(namespace=namespace, label_selector=selector).items
                pods = [p.metadata.name for p in items]
            except Exception:
                LOG.exception("k8s client list pods failed; falling back to kubectl")
        if not pods:
            rc, so, se = _run_kubectl(f"kubectl -n {namespace} get pods -l '{selector}' -o name")
            if rc == 0:
                pods = [ln.replace("pod/", "").strip() for ln in so.splitlines() if ln.strip()]

        stress_cmds = []
        for pod in pods:
            # calculate cpu workers per pod based on intensity (use intensity% of CPU cores in pod node)
            cpu_count = multiprocessing.cpu_count()
            workers = max(1, int(round(cpu_count * (max(1, exp.intensity) / 100.0))))
            cmd = f"kubectl -n {namespace} exec {pod} -- timeout {int(exp.duration_seconds)}s stress --cpu {workers} || true"
            # run asynchronously as subprocess
            p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _register_driver_proc(exp.experiment_id, {"proc": p, "type": "k8s_cpu_stress", "meta": {"pod": pod, "cmd": cmd, "pid": p.pid}})
            stress_cmds.append({"pod": pod, "cmd": cmd, "pid": p.pid})
            # record action
            try:
                asyncio.get_event_loop().create_task(_append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="k8s_cpu_stress_started", timestamp=_now_iso(), details={"pod": pod, "cmd": cmd})))
            except Exception:
                pass
        return {"status": "started", "targets": stress_cmds}
    except Exception as e:
        LOG.exception("cpu_stress_pod failed: %s", e)
        return {"status": "failed", "error": str(e)}

def _driver_cpu_stress_pod_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    # attempt to kill any started subprocesses
    _terminate_driver_procs(exp.experiment_id)
    return {"status": "reverted"}

# -------------------------
# K8s Driver: network latency using tc (requires privileged exec in pod or sidecar)
# -------------------------
def _driver_latency(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Use 'tc qdisc' commands to add latency to network interface inside pods.
    This requires the container to have 'tc' available and sufficient privileges.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "latency not executed in dry run"}
    if not _CHAOS_ALLOW_DESTRUCTIVE:
        return {"status": "blocked", "reason": "destructive_disabled"}

    tgt = exp.target
    namespace = tgt.namespace or "default"
    selector = tgt.label_selector or tgt.name
    if not selector:
        return {"status": "blocked", "reason": "no_selector"}

    latency_ms = int(exp.intensity or 100)  # interpret intensity as ms
    pods = []
    try:
        # find pods (re-use logic)
        if _HAS_K8S:
            try:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1 = k8s_client.CoreV1Api()
                items = v1.list_namespaced_pod(namespace=namespace, label_selector=selector).items
                pods = [p.metadata.name for p in items]
            except Exception:
                LOG.exception("k8s client list pods failed; falling back to kubectl")
        if not pods:
            rc, so, se = _run_kubectl(f"kubectl -n {namespace} get pods -l '{selector}' -o name")
            if rc == 0:
                pods = [ln.replace("pod/", "").strip() for ln in so.splitlines() if ln.strip()]

        commands_run = []
        for pod in pods:
            # construct tc command to add latency on eth0
            tc_cmd = f"kubectl -n {namespace} exec {pod} -- tc qdisc add dev eth0 root netem delay {latency_ms}ms"
            rc, so, se = _run_kubectl(tc_cmd, timeout=20)
            commands_run.append({"pod": pod, "cmd": tc_cmd, "rc": rc, "stdout": so, "stderr": se})
            # record action
            try:
                asyncio.get_event_loop().create_task(_append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="k8s_tc_add", timestamp=_now_iso(), details={"pod": pod, "latency_ms": latency_ms})))
            except Exception:
                pass
        return {"status": "tc_applied", "results": commands_run}
    except Exception as e:
        LOG.exception("latency driver error: %s", e)
        return {"status": "failed", "error": str(e)}

def _driver_latency_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Remove tc rules from affected pods.
    """
    tgt = exp.target
    namespace = tgt.namespace or "default"
    selector = tgt.label_selector or tgt.name
    pods = []
    try:
        if _HAS_K8S:
            try:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1 = k8s_client.CoreV1Api()
                items = v1.list_namespaced_pod(namespace=namespace, label_selector=selector).items
                pods = [p.metadata.name for p in items]
            except Exception:
                LOG.exception("k8s client list pods failed; falling back to kubectl")
        if not pods:
            rc, so, se = _run_kubectl(f"kubectl -n {namespace} get pods -l '{selector}' -o name")
            if rc == 0:
                pods = [ln.replace("pod/", "").strip() for ln in so.splitlines() if ln.strip()]

        results = []
        for pod in pods:
            tc_rm = f"kubectl -n {namespace} exec {pod} -- tc qdisc del dev eth0 root netem || true"
            rc, so, se = _run_kubectl(tc_rm, timeout=20)
            results.append({"pod": pod, "rc": rc, "stdout": so, "stderr": se})
        return {"status": "tc_removed", "results": results}
    except Exception as e:
        LOG.exception("latency revert failed: %s", e)
        return {"status": "failed", "error": str(e)}

# -------------------------
# K8s Driver: cordon/uncordon node
# -------------------------
def _driver_node_cordon(exp: ChaosExperiment) -> Dict[str, Any]:
    """
    Cordon a node (mark unschedulable) and optionally uncordon after revert.
    Target.name expected to be node name.
    """
    if exp.dry_run:
        return {"status": "dry_run", "note": "node_cordon not executed in dry run"}

    if not _CHAOS_ALLOW_DESTRUCTIVE:
        return {"status": "blocked", "reason": "destructive_disabled"}

    node_name = exp.target.name
    if not node_name:
        return {"status": "blocked", "reason": "no_node_name"}

    try:
        if _HAS_K8S:
            try:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1 = k8s_client.CoreV1Api()
                body = {"spec": {"unschedulable": True}}
                v1.patch_node(node_name, body)
                asyncio.get_event_loop().create_task(_append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="k8s_node_cordon", timestamp=_now_iso(), details={"node": node_name})))
                return {"status": "cordoned", "node": node_name}
            except Exception:
                LOG.exception("k8s client cordon failed; falling back to kubectl")
        cmd = f"kubectl cordon {node_name}"
        rc, so, se = _run_kubectl(cmd)
        if rc != 0:
            return {"status": "failed", "error": se or so}
        asyncio.get_event_loop().create_task(_append_action_record(ChaosActionRecord(action_id=f"act_{uuid.uuid4().hex[:8]}", experiment_id=exp.experiment_id, action_type="kubectl_node_cordon", timestamp=_now_iso(), details={"node": node_name})))
        return {"status": "cordoned", "node": node_name}
    except Exception as e:
        LOG.exception("node cordon error: %s", e)
        return {"status": "failed", "error": str(e)}

def _driver_node_cordon_revert(exp: ChaosExperiment) -> Dict[str, Any]:
    node_name = exp.target.name
    if not node_name:
        return {"status": "blocked", "reason": "no_node_name"}
    try:
        if _HAS_K8S:
            try:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1 = k8s_client.CoreV1Api()
                body = {"spec": {"unschedulable": False}}
                v1.patch_node(node_name, body)
                return {"status": "uncordoned", "node": node_name}
            except Exception:
                LOG.exception("k8s client uncordon failed; falling back to kubectl")
        cmd = f"kubectl uncordon {node_name}"
        rc, so, se = _run_kubectl(cmd)
        if rc != 0:
            return {"status": "failed", "error": se or so}
        return {"status": "uncordoned", "node": node_name}
    except Exception as e:
        LOG.exception("node uncordon revert_error: %s", e)
        return {"status": "failed", "error": str(e)}

# -------------------------
# Fallback driver mapping for k8s faults to generic names used earlier
# -------------------------
# Map FaultType values to k8s drivers where appropriate
# These functions will be discovered by _run_experiment_background via naming convention
globals()["_driver_kill_pod"] = _driver_kill_pod
globals()["_driver_cpu_stress_pod"] = _driver_cpu_stress_pod
globals()["_driver_latency"] = _driver_latency
globals()["_driver_node_cordon"] = _driver_node_cordon

# End of Chunk 4/6
# -------------------------
# Chunk 5/6 — Observability, WebSocket, Policies, Safety Enforcement, Reports & Exports
# -------------------------

from fastapi import WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse
from prometheus_client import generate_latest
import gzip
import io

# Optional S3 for exports
try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

# -------------------------
# WebSocket: /ws/chaos
# -------------------------
_CHAOS_WS_CONNECTIONS: List[WebSocket] = []

@router.websocket("/ws/chaos")
async def ws_chaos(websocket: WebSocket, token: Optional[str] = None):
    """
    WebSocket endpoint for real-time chaos events.
    In production you should validate token / user scopes here.
    """
    await websocket.accept()
    _CHAOS_WS_CONNECTIONS.append(websocket)
    LOG.info("Chaos WS connected: %d clients", len(_CHAOS_WS_CONNECTIONS))
    try:
        await websocket.send_text(json.dumps({"message": "connected", "time": _now_iso()}))
        while True:
            try:
                msg = await websocket.receive_text()
                # simple keepalive
                if msg.strip().lower() in ("ping", "keepalive"):
                    await websocket.send_text(json.dumps({"pong": _now_iso()}))
            except WebSocketDisconnect:
                break
            except Exception:
                # ignore stray messages; keep connection alive
                await asyncio.sleep(0.1)
                continue
    finally:
        try:
            _CHAOS_WS_CONNECTIONS.remove(websocket)
        except Exception:
            pass
        LOG.info("Chaos WS disconnected: %d clients", len(_CHAOS_WS_CONNECTIONS))

# Replace earlier placeholder broadcast function with real one
async def _broadcast_chaos_event(payload: dict):
    """
    Broadcast to WS clients and append to action log (audit).
    """
    try:
        # write to audit stream
        await write_audit_event({"source": "chaos_broadcast", "payload": payload, "timestamp": _now_iso()})
    except Exception:
        LOG.exception("Failed to write audit for broadcast")
    data = json.dumps(payload, default=str)
    stale = []
    for ws in list(_CHAOS_WS_CONNECTIONS):
        try:
            await ws.send_text(data)
        except Exception:
            stale.append(ws)
    for s in stale:
        try:
            _CHAOS_WS_CONNECTIONS.remove(s)
        except Exception:
            pass

# -------------------------
# Prometheus metrics endpoint
# -------------------------
@router.get("/metrics")
async def chaos_metrics():
    """
    Return Prometheus metrics for chaos subsystem.
    """
    if not _HAS_PROM or _PROM_REG is None:
        raise HTTPException(status_code=404, detail="Prometheus not enabled in this build")
    try:
        payload = generate_latest(_PROM_REG)
        return JSONResponse(content=payload.decode("utf-8"), media_type="text/plain; version=0.0.4; charset=utf-8")
    except Exception:
        LOG.exception("Failed to generate prometheus metrics")
        raise HTTPException(status_code=500, detail="Metrics generation failed")

# -------------------------
# Policy management
# -------------------------
class ChaosPolicyCreate(BaseModel):
    policy_id: Optional[str] = None
    name: str
    allowed_faults: List[FaultType] = Field(default_factory=lambda: [FaultType.CPU_STRESS, FaultType.MEM_STRESS, FaultType.KILL_POD])
    max_concurrent: int = Field(1, ge=1)
    max_duration_seconds: int = Field(3600, ge=1)
    allowed_namespaces: Optional[List[str]] = Field(default_factory=lambda: ["*"])
    cooldown_seconds: int = Field(300, ge=0)
    description: Optional[str] = None

    @validator("policy_id", pre=True, always=True)
    def set_policy_id(cls, v):
        return v or f"policy_{uuid.uuid4().hex[:8]}"

class ChaosPolicy(ChaosPolicyCreate):
    created_at: str
    updated_at: str
    version: int = 1

_POLICIES_FS = CHAOS_META_DIR / "policies.json"
if not _POLICIES_FS.exists():
    _POLICIES_FS.write_text(json.dumps({}), encoding="utf-8")

def _fs_read_policies() -> Dict[str, Dict[str, Any]]:
    try:
        p = json.loads(_POLICIES_FS.read_text(encoding="utf-8"))
        if not isinstance(p, dict):
            return {}
        return p
    except Exception:
        LOG.exception("Failed reading policies store")
        return {}

def _fs_write_policies(data: Dict[str, Dict[str, Any]]):
    try:
        tmp = _POLICIES_FS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        tmp.replace(_POLICIES_FS)
    except Exception:
        LOG.exception("Failed writing policies store")

async def _save_policy(policy: ChaosPolicy):
    if experiments_col is not None:
        # policies can be stored in DB collection named 'chaos_policies'
        col = _chaos_db.get_collection(os.getenv("CHAOS_POLICIES_COLLECTION", "chaos_policies"))
        await col.update_one({"policy_id": policy.policy_id}, {"$set": policy.dict()}, upsert=True)
    else:
        data = _fs_read_policies()
        data[policy.policy_id] = policy.dict()
        _fs_write_policies(data)

async def _get_policy(policy_id: str) -> Optional[ChaosPolicy]:
    if experiments_col is not None:
        col = _chaos_db.get_collection(os.getenv("CHAOS_POLICIES_COLLECTION", "chaos_policies"))
        doc = await col.find_one({"policy_id": policy_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return ChaosPolicy(**doc)
    data = _fs_read_policies()
    d = data.get(policy_id)
    if not d:
        return None
    return ChaosPolicy(**d)

async def _list_policies() -> List[ChaosPolicy]:
    if experiments_col is not None:
        col = _chaos_db.get_collection(os.getenv("CHAOS_POLICIES_COLLECTION", "chaos_policies"))
        docs = await col.find({}).to_list(length=1000)
        return [ChaosPolicy(**{k: v for k, v in d.items() if k != "_id"}) for d in docs]
    data = _fs_read_policies()
    return [ChaosPolicy(**v) for v in data.values()]

@router.post("/policies", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_policy(payload: ChaosPolicyCreate, user = Depends(get_current_user)):
    now = _now_iso()
    p = ChaosPolicy(**payload.dict(), created_at=now, updated_at=now, version=1)
    await _save_policy(p)
    await _audit(user, "create_policy", p.policy_id, {"name": p.name})
    return p

@router.get("/policies", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_policies():
    policies = await _list_policies()
    return policies

@router.get("/policies/{policy_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_policy(policy_id: str):
    p = await _get_policy(policy_id)
    if not p:
        raise HTTPException(status_code=404, detail="Policy not found")
    return p

@router.put("/policies/{policy_id}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def update_policy(policy_id: str, payload: ChaosPolicyCreate, user = Depends(get_current_user)):
    existing = await _get_policy(policy_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Policy not found")
    now = _now_iso()
    new_version = (existing.version or 1) + 1
    p = ChaosPolicy(**payload.dict(), policy_id=policy_id, created_at=existing.created_at, updated_at=now, version=new_version)
    await _save_policy(p)
    await _audit(user, "update_policy", policy_id, {"name": p.name})
    return p

@router.delete("/policies/{policy_id}", dependencies=[Depends(require_role(Role.ADMIN))])
async def delete_policy(policy_id: str, user = Depends(get_current_user)):
    if experiments_col is not None:
        col = _chaos_db.get_collection(os.getenv("CHAOS_POLICIES_COLLECTION", "chaos_policies"))
        await col.delete_one({"policy_id": policy_id})
    else:
        data = _fs_read_policies()
        if policy_id in data:
            del data[policy_id]
            _fs_write_policies(data)
    await _audit(user, "delete_policy", policy_id, {})
    return {"ok": True, "deleted": policy_id}

# -------------------------
# Policy enforcement helper
# -------------------------
async def _enforce_policies_for_experiment(exp: ChaosExperiment) -> Tuple[bool, Optional[str]]:
    """
    Enforce policies:
      - Is fault allowed in any policy for the target namespace?
      - Check concurrent limits for target
      - Check max_duration cap
    Returns (allowed: bool, reason_if_blocked)
    """
    policies = await _list_policies()
    # find any policy that allows this fault
    allowed_policies = [p for p in policies if exp.fault in p.allowed_faults and ( "*" in p.allowed_namespaces or (exp.target.namespace and exp.target.namespace in p.allowed_namespaces) )]
    if not allowed_policies:
        return False, "no_policy_allows_fault"
    # check max concurrent for the most restrictive policy
    max_concurrent = min([p.max_concurrent for p in allowed_policies]) if allowed_policies else 1
    # count active experiments with same target key
    current = await _list_experiments_from_store(status=[ExperimentStatus.RUNNING, ExperimentStatus.PENDING])
    same_target = [e for e in current if e.target.type == exp.target.type and (e.target.name == exp.target.name or e.target.label_selector == exp.target.label_selector)]
    if len(same_target) >= max_concurrent:
        return False, "concurrent_limit_reached"
    # check duration cap
    max_duration = min([p.max_duration_seconds for p in allowed_policies])
    if exp.duration_seconds > max_duration:
        return False, f"duration_exceeds_policy_max ({max_duration}s)"
    return True, None

# -------------------------
# Reports & Exports
# -------------------------
_REPORTS_DIR = CHAOS_META_DIR / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/reports", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_reports():
    """
    List generated report files in the reports directory.
    """
    files = []
    for f in sorted(_REPORTS_DIR.glob("*.jsonl.gz"), key=lambda p: p.stat().st_mtime, reverse=True):
        files.append({"name": f.name, "path": str(f), "mtime": f.stat().st_mtime})
    return files

@router.post("/reports/generate", dependencies=[Depends(require_role(Role.OPERATOR))])
async def generate_report(since_hours: int = Query(24), upload_s3: bool = Query(False), user = Depends(get_current_user)):
    """
    Generate a compressed JSONL report of experiments/actions from the last `since_hours`.
    Optionally upload to S3 if configured.
    """
    cutoff = time.time() - since_hours * 3600
    outfile = _REPORTS_DIR / f"chaos_report_{int(time.time())}.jsonl.gz"
    count = 0
    with gzip.open(outfile, "wt", encoding="utf-8") as gz:
        # include experiments
        exps = await _list_experiments_from_store()
        for e in exps:
            created_ts = datetime.datetime.fromisoformat(e.created_at.replace("Z", "+00:00")).timestamp() if e.created_at else 0
            if created_ts >= cutoff:
                gz.write(json.dumps({"type": "experiment", "data": e.dict()}, default=str) + "\n")
                count += 1
        # include recent actions
        if _ACTIONS_FS.exists():
            with open(_ACTIONS_FS, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                        ts = datetime.datetime.fromisoformat(r.get("timestamp").replace("Z", "+00:00")).timestamp() if r.get("timestamp") else 0
                        if ts >= cutoff:
                            gz.write(json.dumps({"type": "action", "data": r}, default=str) + "\n")
                            count += 1
                    except Exception:
                        continue
    result = {"ok": True, "file": str(outfile), "entries": count}
    await _audit(user, "generate_report", "chaos_reports", {"file": str(outfile), "entries": count})
    if upload_s3 and _HAS_BOTO3 and os.getenv("S3_BUCKET"):
        s3 = boto3.client("s3")
        key = f"chaos/reports/{outfile.name}"
        try:
            s3.upload_file(str(outfile), os.getenv("S3_BUCKET"), key)
            result["s3"] = f"s3://{os.getenv('S3_BUCKET')}/{key}"
        except Exception:
            LOG.exception("Failed to upload report to s3")
    return result

@router.get("/reports/download/{filename}", dependencies=[Depends(require_role(Role.VIEWER))])
async def download_report(filename: str):
    p = _REPORTS_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(p), filename=p.name, media_type="application/gzip")

# -------------------------
# Safety enforcement: max concurrent experiments overall
# -------------------------
_MAX_CONCURRENT_GLOBAL = int(os.getenv("CHAOS_MAX_CONCURRENT_GLOBAL", "5"))

async def _check_global_concurrency_allowed() -> bool:
    running = await _list_experiments_from_store(status=[ExperimentStatus.RUNNING, ExperimentStatus.PENDING])
    return len(running) < _MAX_CONCURRENT_GLOBAL

# -------------------------
# Simple summary endpoints
# -------------------------
@router.get("/summary", dependencies=[Depends(require_role(Role.VIEWER))])
async def chaos_summary():
    exps = await _list_experiments_from_store()
    counts = {s.value: 0 for s in ExperimentStatus}
    for e in exps:
        counts[e.status.value] = counts.get(e.status.value, 0) + 1
    return {
        "total_experiments": len(exps),
        "by_status": counts,
        "active_ws_clients": len(_CHAOS_WS_CONNECTIONS),
        "prometheus_enabled": _HAS_PROM,
    }

# End of Chunk 5/6
# -------------------------
# Chunk 6/6 — Background Recovery Loop, Cleanup, Startup/Shutdown, Health, Exports
# -------------------------

# Background task management
_CHAOS_BG_TASK: Optional[asyncio.Task] = None
_BG_RUNNING = False
_RECOVERY_INTERVAL = int(os.getenv("CHAOS_RECOVERY_INTERVAL", "60"))  # seconds

async def _background_recovery_loop():
    """
    Periodically checks for:
      - Experiments that exceeded duration but still marked RUNNING
      - Stopped/FAILED experiments with lingering driver processes
      - Cleans expired cooldowns
    """
    global _BG_RUNNING
    _BG_RUNNING = True
    LOG.info("Chaos recovery loop started (interval=%ds)", _RECOVERY_INTERVAL)
    while True:
        try:
            # Cleanup cooldowns
            now = time.time()
            expired_keys = [k for k, v in _COOLDOWNS.items() if v < now]
            for k in expired_keys:
                _COOLDOWNS.pop(k, None)

            # Find long-running experiments
            exps = await _list_experiments_from_store(status=[ExperimentStatus.RUNNING])
            for e in exps:
                if e.started_at:
                    try:
                        start_ts = datetime.datetime.fromisoformat(e.started_at.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        continue
                    if time.time() - start_ts > e.duration_seconds + 30:
                        LOG.warning("Experiment %s exceeded duration; forcing stop", e.experiment_id)
                        e.status = ExperimentStatus.STOPPING
                        await _persist_experiment(e)
                        _terminate_driver_procs(e.experiment_id)
                        e.status = ExperimentStatus.COMPLETED
                        e.stopped_at = _now_iso()
                        await _persist_experiment(e)
                        await _broadcast_chaos_event({"event": "auto_stopped", "experiment_id": e.experiment_id})
                        await _audit(None, "auto_stop_experiment", e.experiment_id, {"reason": "duration_exceeded"})
            await asyncio.sleep(_RECOVERY_INTERVAL)
        except asyncio.CancelledError:
            LOG.info("Chaos recovery loop cancelled")
            break
        except Exception:
            LOG.exception("Error in chaos recovery loop")
            await asyncio.sleep(_RECOVERY_INTERVAL)
    _BG_RUNNING = False
    LOG.info("Chaos recovery loop ended")

def start_chaos_background_loop(loop: Optional[asyncio.AbstractEventLoop] = None):
    global _CHAOS_BG_TASK
    if _CHAOS_BG_TASK and not _CHAOS_BG_TASK.done():
        return
    loop = loop or asyncio.get_event_loop()
    _CHAOS_BG_TASK = loop.create_task(_background_recovery_loop())
    LOG.info("Chaos background loop scheduled")

def stop_chaos_background_loop():
    global _CHAOS_BG_TASK
    if _CHAOS_BG_TASK:
        _CHAOS_BG_TASK.cancel()
        _CHAOS_BG_TASK = None
        LOG.info("Chaos background loop stopped")

# -------------------------
# Cleanup utilities
# -------------------------
async def cleanup_old_experiments(retention_days: int = 7) -> int:
    """
    Deletes experiment records older than `retention_days` days.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
    count = 0
    if experiments_col is not None:
        docs = await experiments_col.find({}).to_list(length=10000)
        for d in docs:
            created = d.get("created_at")
            try:
                if created and datetime.datetime.fromisoformat(created.replace("Z", "+00:00")) < cutoff:
                    await experiments_col.delete_one({"experiment_id": d["experiment_id"]})
                    count += 1
            except Exception:
                continue
    else:
        exs = _fs_read_experiments()
        for k, v in list(exs.items()):
            created = v.get("created_at")
            try:
                if created and datetime.datetime.fromisoformat(created.replace("Z", "+00:00")) < cutoff:
                    del exs[k]
                    count += 1
            except Exception:
                continue
        _fs_write_experiments(exs)
    return count

@router.post("/cleanup", dependencies=[Depends(require_role(Role.ADMIN))])
async def cleanup_expired(user = Depends(get_current_user), retention_days: int = Query(7)):
    """
    Delete experiment records older than `retention_days`.
    """
    removed = await cleanup_old_experiments(retention_days)
    await _audit(user, "cleanup_old_experiments", "chaos", {"removed": removed})
    return {"ok": True, "removed": removed}

# -------------------------
# Health & diagnostics
# -------------------------
@router.get("/health")
async def chaos_health():
    return {
        "ok": True,
        "mongo_connected": experiments_col is not None,
        "background_running": _BG_RUNNING,
        "active_ws": len(_CHAOS_WS_CONNECTIONS),
        "cooldowns": len(_COOLDOWNS),
    }

@router.get("/status", dependencies=[Depends(require_role(Role.VIEWER))])
async def chaos_status():
    exps = await _list_experiments_from_store()
    running = len([e for e in exps if e.status == ExperimentStatus.RUNNING])
    failed = len([e for e in exps if e.status == ExperimentStatus.FAILED])
    completed = len([e for e in exps if e.status == ExperimentStatus.COMPLETED])
    return {
        "running": running,
        "failed": failed,
        "completed": completed,
        "background_running": _BG_RUNNING,
        "active_ws_clients": len(_CHAOS_WS_CONNECTIONS),
        "prometheus_enabled": _HAS_PROM,
    }

# -------------------------
# Startup & Shutdown Hooks
# -------------------------
@router.on_event("startup")
async def _on_chaos_startup():
    try:
        LOG.info("PriorityMax Chaos API starting up...")
        start_chaos_background_loop()
        await _broadcast_chaos_event({"event": "chaos_api_startup", "time": _now_iso()})
    except Exception:
        LOG.exception("Chaos startup failed")

@router.on_event("shutdown")
async def _on_chaos_shutdown():
    try:
        stop_chaos_background_loop()
        _terminate_driver_procs("*")  # kill any remaining stress processes
        await _broadcast_chaos_event({"event": "chaos_api_shutdown", "time": _now_iso()})
        LOG.info("Chaos API shutdown complete")
    except Exception:
        LOG.exception("Chaos shutdown error")

# -------------------------
# Final Exports
# -------------------------
__all__ = [
    "router",
    "start_chaos_background_loop",
    "stop_chaos_background_loop",
    "cleanup_old_experiments",
]

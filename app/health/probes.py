# backend/app/health/probes.py
"""
PriorityMax Health & Probes module (production-grade)

Provides:
 - /health/liveness        -> minimal liveness check (quick)
 - /health/readiness       -> readiness check (can be heavier, cached)
 - /health/diagnostics     -> deep diagnostics JSON (redis, mongo, models, disk, workers)
 - /health/self_heal       -> admin-only self-healing actions (safe defaults)
 - /health/metrics         -> prometheus scrape for health subsystem (optional)
 - /health/raw             -> raw low-level probe invocation (for debugging)
 
Notes:
 - Self-heal actions are intentionally conservative: they publish hints to a restart channel,
   optionally integrate with Kubernetes client if available, and log/audit actions.
 - All external checks are defensive and degrade to "unknown" rather than crashing the service.
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
import shutil
import psutil
import platform
import datetime
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException, Depends, Body, Query, status
from pydantic import BaseModel, Field

# Attempt to import optional dependencies (graceful fallback)
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

try:
    from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _HAS_PROM = True
except Exception:
    CollectorRegistry = Gauge = generate_latest = CONTENT_TYPE_LATEST = None
    _HAS_PROM = False

# Kubernetes optional client for self-heal pod restart (only used if installed)
try:
    from kubernetes import client as k8s_client, config as k8s_config
    _HAS_K8S = True
except Exception:
    k8s_client = None
    k8s_config = None
    _HAS_K8S = False

# Admin helpers (auth + auditing)
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    # fallback stubs (read-only)
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
        # simple fallback: append to a local audit file
        p = pathlib.Path.cwd() / "backend" / "logs" / "health_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.health")
LOG.setLevel(os.getenv("PRIORITYMAX_HEALTH_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Router
router = APIRouter(prefix="/health", tags=["health"])

# Configurable environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MONGO_URL = os.getenv("MONGO_URL", None)
ML_MODELS_DIR = pathlib.Path(os.getenv("ML_MODELS_DIR", str(pathlib.Path(__file__).resolve().parents[2] / "app" / "ml" / "models")))
WORKER_REGISTRY_KEY = os.getenv("WORKER_REGISTRY_KEY", "prioritymax:workers")  # used if redis present
SELF_HEAL_RESTART_CHANNEL = os.getenv("SELF_HEAL_RESTART_CHANNEL", "prioritymax:self_heal:restart")
PROBE_CACHE_TTL = int(os.getenv("PROBE_CACHE_TTL", "8"))  # seconds to cache readiness probes
DISK_WARN_THRESHOLD_PERCENT = float(os.getenv("DISK_WARN_PCT", "10.0"))  # warn if free disk < threshold %

# Async clients (lazy)
_redis_client: Optional[Any] = None
_mongo_client: Optional[Any] = None

async def get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not _HAS_AIOREDIS:
        return None
    try:
        _redis_client = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        await _redis_client.ping()
        return _redis_client
    except Exception:
        LOG.exception("Health: failed connecting to Redis")
        _redis_client = None
        return None

def get_mongo_sync():
    # synchronous helper (rarely used)
    return None

if _HAS_MOTOR and MONGO_URL:
    try:
        _mongo_client = motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    except Exception:
        LOG.exception("Health: failed connecting to Mongo")

# Probe cache to avoid hammering upstream
_probe_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"ts": float, "result": {...}}

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    v = _probe_cache.get(key)
    if not v:
        return None
    if (time.time() - v["ts"]) > PROBE_CACHE_TTL:
        _probe_cache.pop(key, None)
        return None
    return v["result"]

def _cache_set(key: str, result: Dict[str, Any]):
    _probe_cache[key] = {"ts": time.time(), "result": result}

# -----------------------------
# Pydantic response models
# -----------------------------
class ComponentHealth(BaseModel):
    name: str
    status: str = Field(..., description="ok | warn | fail | unknown")
    details: Optional[Dict[str, Any]] = None

class HealthReport(BaseModel):
    app: str = "prioritymax"
    version: Optional[str] = None
    ts: str
    overall: str = Field(..., description="ok | degraded | fail")
    components: List[ComponentHealth] = Field(default_factory=list)

class SelfHealRequest(BaseModel):
    action: str = Field(..., description="restart_worker | restart_pod | flush_cache | noop")
    target: Optional[str] = Field(None, description="worker_id, pod name, or other target")
    reason: Optional[str] = Field(None, description="human-friendly reason")
    force: bool = Field(False, description="bypass safety checks (admin only)")

class SelfHealResponse(BaseModel):
    ok: bool
    action: str
    target: Optional[str]
    message: Optional[str]

# -----------------------------
# Low-level component checks
# -----------------------------
async def _check_redis() -> ComponentHealth:
    name = "redis"
    if not _HAS_AIOREDIS:
        return ComponentHealth(name=name, status="unknown", details={"reason": "aioredis_not_installed"})
    try:
        r = await get_redis()
        if not r:
            return ComponentHealth(name=name, status="warn", details={"reason": "connection_unavailable"})
        # attempt a ping + sample few keys
        try:
            pong = await r.ping()
        except Exception:
            pong = False
        if not pong:
            return ComponentHealth(name=name, status="fail", details={"reason": "ping_failed"})
        info = {}
        try:
            # sample worker registry length if present
            keys = await r.keys(f"{WORKER_REGISTRY_KEY}*")
            info["sampled_keys"] = len(keys)
        except Exception:
            pass
        return ComponentHealth(name=name, status="ok", details=info)
    except Exception:
        LOG.exception("Health: redis check error")
        return ComponentHealth(name=name, status="fail", details={"reason": "exception"})

async def _check_mongo() -> ComponentHealth:
    name = "mongo"
    if not _HAS_MOTOR or not MONGO_URL:
        return ComponentHealth(name=name, status="unknown", details={"reason": "mongo_not_configured"})
    try:
        # cheap ping by listing db names or calling server_info
        await _mongo_client.admin.command("ping")
        return ComponentHealth(name=name, status="ok")
    except Exception:
        LOG.exception("Health: mongo check failed")
        return ComponentHealth(name=name, status="fail", details={"reason": "ping_failed"})

async def _check_ml_models() -> ComponentHealth:
    name = "ml_models"
    try:
        if not ML_MODELS_DIR.exists():
            return ComponentHealth(name=name, status="warn", details={"reason": "models_dir_missing", "path": str(ML_MODELS_DIR)})
        files = list(ML_MODELS_DIR.glob("*"))
        info = {"model_files": [f.name for f in files][:20], "count": len(files)}
        if len(files) == 0:
            return ComponentHealth(name=name, status="warn", details=info)
        return ComponentHealth(name=name, status="ok", details=info)
    except Exception:
        LOG.exception("Health: ml models check error")
        return ComponentHealth(name=name, status="fail", details={"reason": "exception"})

async def _check_disk() -> ComponentHealth:
    name = "disk"
    try:
        root = pathlib.Path("/")
        usage = shutil.disk_usage(str(root))
        free_pct = (usage.free / usage.total) * 100.0
        details = {"total": usage.total, "free": usage.free, "free_pct": round(free_pct, 2)}
        status = "ok" if free_pct > DISK_WARN_THRESHOLD_PERCENT else "warn"
        return ComponentHealth(name=name, status=status, details=details)
    except Exception:
        LOG.exception("Health: disk check failed")
        return ComponentHealth(name=name, status="unknown", details={"reason": "exception"})

async def _check_cpu_memory() -> ComponentHealth:
    name = "system"
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        details = {"cpu_percent": cpu, "mem_percent": mem.percent, "mem_total": mem.total}
        # simple heuristic: warn if cpu > 90% or mem > 90%
        if cpu > 90 or mem.percent > 90:
            status = "warn"
        else:
            status = "ok"
        return ComponentHealth(name=name, status=status, details=details)
    except Exception:
        LOG.exception("Health: cpu/mem check failed")
        return ComponentHealth(name=name, status="unknown", details={"reason": "exception"})

async def _check_worker_registry() -> ComponentHealth:
    name = "workers"
    try:
        r = await get_redis()
        if not r:
            return ComponentHealth(name=name, status="unknown", details={"reason": "redis_missing"})
        # assume worker registry keys are like WORKER_REGISTRY_KEY:worker_id
        try:
            keys = await r.keys(f"{WORKER_REGISTRY_KEY}*")
            # fetch last-seen times for a few workers
            sample = {}
            now = time.time()
            for k in keys[:20]:
                try:
                    val = await r.hgetall(k)
                    last = float(val.get("last_seen", "0") or "0")
                    sample[k] = {"last_seen": last, "age_seconds": int(now - last)}
                except Exception:
                    sample[k] = {"raw": "unreadable"}
            details = {"worker_count": len(keys), "sample": sample}
            status = "ok"
            # warn if there are zero workers
            if len(keys) == 0:
                status = "warn"
            return ComponentHealth(name=name, status=status, details=details)
        except Exception:
            LOG.exception("Health: worker registry sample failed")
            return ComponentHealth(name=name, status="unknown", details={"reason": "sample_failed"})
    except Exception:
        LOG.exception("Health: worker registry check failed")
        return ComponentHealth(name=name, status="fail", details={"reason": "exception"})

async def _check_autoscaler() -> ComponentHealth:
    name = "autoscaler"
    # best-effort: try contacting autoscaler if there's a client or check last hint timestamp in redis
    try:
        r = await get_redis()
        if not r:
            return ComponentHealth(name=name, status="unknown", details={"reason": "redis_missing"})
        # try to read last autoscaler hint key (if exists in redis)
        hint_key = "prioritymax:autoscaler:last_hint"
        try:
            raw = await r.get(hint_key)
            if raw:
                return ComponentHealth(name=name, status="ok", details={"last_hint": raw})
            else:
                return ComponentHealth(name=name, status="warn", details={"reason": "no_recent_hints"})
        except Exception:
            return ComponentHealth(name=name, status="unknown", details={"reason": "no_hint_key"})
    except Exception:
        LOG.exception("Health: autoscaler check failed")
        return ComponentHealth(name=name, status="fail", details={"reason": "exception"})

# -----------------------------
# Compose readiness/liveness
# -----------------------------
async def _compose_health(deep: bool = False) -> HealthReport:
    """
    Build structured health report. If deep==False use a smaller subset of checks and cached result.
    """
    now = datetime.datetime.utcnow().isoformat() + "Z"
    # use cache for readiness if deep==False
    cache_key = "readiness_shallow" if not deep else f"readiness_deep_{int(time.time() // PROBE_CACHE_TTL)}"
    if not deep:
        cached = _cache_get(cache_key)
        if cached:
            return HealthReport(**cached)

    components: List[ComponentHealth] = []
    overall = "ok"

    # always run these quick checks
    redis_comp = await _check_redis()
    components.append(redis_comp)
    if redis_comp.status != "ok" and overall == "ok":
        overall = "degraded" if redis_comp.status == "warn" else "fail"

    mongo_comp = await _check_mongo()
    components.append(mongo_comp)
    if mongo_comp.status != "ok" and overall == "ok":
        overall = "degraded" if mongo_comp.status == "warn" else "fail"

    disk_comp = await _check_disk()
    components.append(disk_comp)
    if disk_comp.status != "ok" and overall == "ok":
        overall = "degraded" if disk_comp.status == "warn" else "fail"

    sys_comp = await _check_cpu_memory()
    components.append(sys_comp)
    if sys_comp.status != "ok" and overall == "ok":
        overall = "degraded" if sys_comp.status == "warn" else "fail"

    # deeper checks when requested
    if deep:
        models_comp = await _check_ml_models()
        components.append(models_comp)
        if models_comp.status != "ok" and overall == "ok":
            overall = "degraded" if models_comp.status == "warn" else "fail"

        workers_comp = await _check_worker_registry()
        components.append(workers_comp)
        if workers_comp.status != "ok" and overall == "ok":
            overall = "degraded" if workers_comp.status == "warn" else "fail"

        autoscaler_comp = await _check_autoscaler()
        components.append(autoscaler_comp)
        if autoscaler_comp.status != "ok" and overall == "ok":
            overall = "degraded" if autoscaler_comp.status == "warn" else "fail"

    report = HealthReport(app="prioritymax", version=os.getenv("PRIORITYMAX_VERSION", None), ts=now, overall=overall, components=components)
    if not deep:
        _cache_set(cache_key, report.dict())
    return report

# -----------------------------
# Endpoints
# -----------------------------
@router.get("/liveness", summary="K8s liveness probe (fast)", tags=["health"])
async def liveness():
    """
    Quick liveness probe used by Kubernetes.
    - Returns HTTP 200 when process is responsive.
    - Keep extremely fast and conservative.
    """
    return {"status": "alive", "ts": datetime.datetime.utcnow().isoformat() + "Z"}

@router.get("/readiness", summary="K8s readiness probe (cached, cheap)", tags=["health"])
async def readiness(deep: bool = Query(False, description="If true run a deeper readiness check (slower)")):
    """
    Readiness probe. When used by K8s, keep deep=False.
    For an operator-run health check you can pass deep=true to include ML models, workers, autoscaler.
    Results are cached for PROBE_CACHE_TTL seconds when deep=False.
    """
    report = await _compose_health(deep=deep)
    http_status = 200 if report.overall == "ok" else 503
    return report.dict()

@router.get("/diagnostics", summary="Deep diagnostics (debugging)", tags=["health"])
async def diagnostics():
    """
    Full diagnostics JSON — runs all deep checks (not cached).
    Designed for operator troubleshooting.
    """
    report = await _compose_health(deep=True)
    # Add extra runtime information
    extra = {
        "python_version": platform.python_version(),
        "process_pid": os.getpid(),
        "platform": platform.platform()
    }
    payload = report.dict()
    payload["runtime"] = extra
    return payload

@router.get("/raw", summary="Run raw single probe", tags=["health"])
async def raw_probe(name: str = Query(..., description="component name: redis|mongo|disk|system|models|workers|autoscaler")):
    """
    Trigger a single low-level probe and return raw ComponentHealth.
    """
    mappings = {
        "redis": _check_redis,
        "mongo": _check_mongo,
        "disk": _check_disk,
        "system": _check_cpu_memory,
        "models": _check_ml_models,
        "workers": _check_worker_registry,
        "autoscaler": _check_autoscaler
    }
    fn = mappings.get(name)
    if not fn:
        raise HTTPException(status_code=400, detail="unknown probe name")
    ch = await fn()
    return ch.dict()

# Prometheus metrics endpoint for health subsystem (optional)
if _HAS_PROM:
    _PROM_REG = CollectorRegistry()
    _PROM_UP = Gauge("prioritymax_health_up", "Health overall (1=OK, 0=fail)", registry=_PROM_REG)
    _PROM_DISK_FREE = Gauge("prioritymax_health_disk_free_bytes", "Free disk bytes", registry=_PROM_REG)

    @router.get("/metrics")
    async def health_metrics():
        # refresh some values
        report = await _compose_health(deep=False)
        up = 1 if report.overall == "ok" else 0
        _PROM_UP.set(up)
        # disk free
        disk = await _check_disk()
        try:
            free = disk.details.get("free", 0) if disk.details else 0
            _PROM_DISK_FREE.set(free)
        except Exception:
            pass
        payload = generate_latest(_PROM_REG)
        return (payload, CONTENT_TYPE_LATEST)
else:
    @router.get("/metrics")
    async def health_metrics_disabled():
        raise HTTPException(status_code=404, detail="Prometheus client not available")

# -----------------------------
# Self-heal actions (admin only)
# -----------------------------
async def _publish_restart_hint(target: Optional[str], reason: Optional[str], via_redis: bool = True) -> Dict[str, Any]:
    """
    Publish a restart hint to the restart channel (non-destructive).
    This allows external operators or a Kubernetes operator to pick up and act.
    """
    payload = {"id": f"restart_hint_{int(time.time())}_{uuid.uuid4().hex[:6]}", "target": target, "reason": reason or "unspecified", "ts": datetime.datetime.utcnow().isoformat() + "Z"}
    try:
        if via_redis and _HAS_AIOREDIS:
            r = await get_redis()
            if r:
                await r.publish(SELF_HEAL_RESTART_CHANNEL, json.dumps(payload))
        # always log and audit
        await write_audit_event({"event": "self_heal_hint_published", "payload": payload})
        LOG.info("Self-heal hint published: %s", payload)
        return {"ok": True, "payload": payload}
    except Exception:
        LOG.exception("Failed to publish restart hint")
        return {"ok": False, "error": "publish_failed", "payload": payload}

async def _attempt_k8s_pod_restart(pod_name: str, namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Attempt to delete a pod (k8s will recreate via deployment/replica-set).
    This requires kubernetes client and proper permissions. Gated and admin-only.
    """
    if not _HAS_K8S:
        return {"ok": False, "error": "k8s_client_not_installed"}
    try:
        # load kube config (in-cluster or kubeconfig)
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        v1 = k8s_client.CoreV1Api()
        ns = namespace or os.getenv("POD_NAMESPACE", "default")
        # delete pod gracefully
        body = k8s_client.V1DeleteOptions(grace_period_seconds=30)
        resp = v1.delete_namespaced_pod(name=pod_name, namespace=ns, body=body)
        await write_audit_event({"event": "k8s_pod_restart_attempt", "pod": pod_name, "namespace": ns, "resp": repr(resp), "ts": _now_iso()})
        return {"ok": True, "pod": pod_name, "namespace": ns, "resp": str(resp)}
    except Exception as e:
        LOG.exception("Failed to restart pod %s", pod_name)
        return {"ok": False, "error": str(e)}

@router.post("/self_heal", dependencies=[Depends(require_role(Role.ADMIN))], summary="Admin self-heal actions")
async def self_heal(req: SelfHealRequest = Body(...), user = Depends(get_current_user)):
    """
    Admin-only endpoint to request self-healing actions.
    Actions supported:
     - restart_worker: publish a restart hint for a worker_id (target)
     - restart_pod: attempt k8s pod restart (if kubernetes client installed)
     - flush_cache: clear internal probe cache
     - noop: write audit and return status
    All actions are audited and logged.
    """
    action = req.action
    target = req.target
    reason = req.reason or "manual_admin"
    force = req.force

    # Safety: disallow restart_pod unless K8S client present or force=true
    if action == "restart_pod":
        if not _HAS_K8S and not force:
            raise HTTPException(status_code=400, detail="kubernetes client not available; set force=true to publish hint only")
        # if k8s client available, attempt delete
        if _HAS_K8S:
            res = await _attempt_k8s_pod_restart(target or "", namespace=os.getenv("POD_NAMESPACE", None))
            await write_audit_event({"event": "self_heal", "action": action, "target": target, "user": getattr(user, "username", "unknown"), "result": res, "ts": _now_iso()})
            return SelfHealResponse(ok=res.get("ok", False), action=action, target=target, message=str(res.get("error") or res.get("resp")))
        # fallback: publish restart hint
        res = await _publish_restart_hint(target, reason)
        await write_audit_event({"event": "self_heal_hint", "action": action, "target": target, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
        return SelfHealResponse(ok=res.get("ok", False), action=action, target=target, message="hint_published")

    elif action == "restart_worker":
        # publish hint for worker restart
        res = await _publish_restart_hint(target, reason)
        await write_audit_event({"event": "self_heal_restart_worker", "target": target, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
        return SelfHealResponse(ok=res.get("ok", False), action=action, target=target, message="hint_published")

    elif action == "flush_cache":
        _probe_cache.clear()
        await write_audit_event({"event": "self_heal_flush_cache", "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
        return SelfHealResponse(ok=True, action=action, target=None, message="cache_cleared")

    elif action == "noop":
        await write_audit_event({"event": "self_heal_noop", "user": getattr(user, "username", "unknown"), "target": target, "reason": reason, "ts": _now_iso()})
        return SelfHealResponse(ok=True, action=action, target=target, message="noop_logged")

    else:
        raise HTTPException(status_code=400, detail="unsupported self_heal action")

# -----------------------------
# Startup/Shutdown hooks
# -----------------------------
@router.on_event("startup")
async def _on_health_startup():
    LOG.info("Health probes startup: initializing clients")
    # warm up redis/mongo clients (non-blocking)
    try:
        await get_redis()
    except Exception:
        pass
    if _HAS_MOTOR and MONGO_URL:
        try:
            # test ping
            await _mongo_client.admin.command("ping")
        except Exception:
            LOG.exception("Health: mongo ping failed at startup")
    # create logs dir
    p = pathlib.Path.cwd() / "backend" / "logs"
    p.mkdir(parents=True, exist_ok=True)

@router.on_event("shutdown")
async def _on_health_shutdown():
    LOG.info("Health probes shutdown: closing clients")
    # close redis if exists
    global _redis_client
    try:
        if _redis_client is not None:
            await _redis_client.close()
            _redis_client = None
    except Exception:
        pass
    # close mongo asynchronously if exists
    try:
        if _mongo_client is not None:
            _mongo_client.close()
    except Exception:
        pass

# -----------------------------
# Utility: ISO now
# -----------------------------
def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

# End of file

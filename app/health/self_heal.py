# backend/app/health/self_heal.py
"""
PriorityMax Self-Heal Subsystem

Provides a safe, auditable, and rate-limited interface to:
 - publish restart hints for workers or services (Redis pub/sub)
 - attempt Kubernetes pod deletion (to trigger restart)
 - perform rolling restart for Deployments (k8s)
 - send notifications to webhooks / Slack / Ops channels
 - schedule and persist self-heal actions (via Redis lists)
 - admin-protected FastAPI endpoints to request and inspect self-heal actions

Design goals:
 - NON-DESTRUCTIVE by default: most actions publish "hints" that operators / operators' automation consume
 - Rate-limited: avoid restart loops or repeated destructive operations
 - Audited: every action is logged to audit store using write_audit_event
 - Safe fallbacks: works when Redis or kubernetes client isn't installed (publishes hints instead)
 - Dry-run mode: simulate actions without executing them
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
import functools
import datetime
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Body, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

# Optional dependencies (graceful fallback)
try:
    import aioredis
    _HAS_AIOREDIS = True
except Exception:
    aioredis = None
    _HAS_AIOREDIS = False

try:
    from kubernetes import client as k8s_client, config as k8s_config
    _HAS_K8S = True
except Exception:
    k8s_client = None
    k8s_config = None
    _HAS_K8S = False

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    requests = None
    _HAS_REQUESTS = False

# Admin helpers (auth + auditing) — use app.api.admin if available
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    # Minimal stubs for auth and auditing when admin module is not present
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
        # fallback: append to local audit file
        p = pathlib.Path.cwd() / "backend" / "logs" / "self_heal_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.self_heal")
LOG.setLevel(os.getenv("PRIORITYMAX_SELF_HEAL_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Router for API endpoints
router = APIRouter(prefix="/self_heal", tags=["self_heal"])

# Configuration (env-driven)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SELF_HEAL_CHANNEL = os.getenv("SELF_HEAL_RESTART_CHANNEL", "prioritymax:self_heal:restart")
SELF_HEAL_QUEUE = os.getenv("SELF_HEAL_QUEUE", "prioritymax:self_heal:queue")
SELF_HEAL_RATE_LIMIT_WINDOW = int(os.getenv("SELF_HEAL_RATE_LIMIT_WINDOW", "300"))  # seconds
SELF_HEAL_RATE_LIMIT_COUNT = int(os.getenv("SELF_HEAL_RATE_LIMIT_COUNT", "3"))  # allowed actions per window
SELF_HEAL_DEFAULT_DRY_RUN = os.getenv("SELF_HEAL_DEFAULT_DRY_RUN", "true").lower() in ("1", "true", "yes")
SELF_HEAL_WEBHOOKS = os.getenv("SELF_HEAL_WEBHOOKS", "")  # comma separated webhook URLs for notifications
K8S_IN_CLUSTER = os.getenv("K8S_IN_CLUSTER", "true").lower() in ("1", "true", "yes")

# Redis client (lazy)
_redis_client: Optional[Any] = None

async def get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not _HAS_AIOREDIS:
        LOG.info("aioredis not installed — Redis features disabled in self-heal")
        return None
    try:
        _redis_client = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        await _redis_client.ping()
        LOG.info("Self-heal redis connected at %s", REDIS_URL)
        return _redis_client
    except Exception:
        LOG.exception("Failed connecting to Redis for self-heal")
        _redis_client = None
        return None

# Internal rate-limit store (in-memory fallback if redis unavailable)
_rate_limit_store: Dict[str, List[float]] = {}  # key -> list of timestamps

def _now_ts() -> float:
    return time.time()

def _rate_limit_key(action: str, target: Optional[str]) -> str:
    t = target or "global"
    return f"self_heal:rl:{action}:{t}"

async def _is_rate_limited(action: str, target: Optional[str]) -> bool:
    """
    Rate-limit check: allow SELF_HEAL_RATE_LIMIT_COUNT actions per SELF_HEAL_RATE_LIMIT_WINDOW seconds.
    Prefer Redis (sorted set) if available, otherwise in-memory sliding window.
    """
    key = _rate_limit_key(action, target)
    r = await get_redis()
    now = _now_ts()
    window = SELF_HEAL_RATE_LIMIT_WINDOW
    limit = SELF_HEAL_RATE_LIMIT_COUNT

    if r:
        try:
            # Use Redis ZADD with score=ts, then ZREMRANGEBYSCORE to trim, ZCARD to get count
            pipe = r.pipeline()
            pipe.zadd(key, {str(uuid.uuid4().hex): now})
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zcard(key)
            pipe.expire(key, window + 10)
            res = await pipe.execute()
            count = int(res[2] or 0)
            LOG.debug("Rate limit (redis) for %s:%s -> %d", action, target, count)
            return count > limit
        except Exception:
            LOG.exception("Rate limit check (redis) failed, falling back to in-memory")
    # fallback in-memory
    timestamps = _rate_limit_store.setdefault(key, [])
    # remove old
    timestamps = [ts for ts in timestamps if ts >= now - window]
    timestamps.append(now)
    _rate_limit_store[key] = timestamps
    LOG.debug("Rate limit (mem) for %s:%s -> %d", action, target, len(timestamps))
    return len(timestamps) > limit

# Pydantic models for API
class SelfHealAction(BaseModel):
    action_id: str = Field(default_factory=lambda: f"heal_{uuid.uuid4().hex[:8]}")
    action: str  # restart_worker | restart_pod | rolling_restart | flush_cache | noop
    target: Optional[str] = None
    namespace: Optional[str] = None  # for k8s
    reason: Optional[str] = None
    dry_run: Optional[bool] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    requested_by: Optional[str] = None
    status: str = Field(default="pending")  # pending | executed | scheduled | failed
    result: Optional[Dict[str, Any]] = None

class SelfHealRequest(BaseModel):
    action: str
    target: Optional[str] = None
    namespace: Optional[str] = None
    reason: Optional[str] = None
    dry_run: Optional[bool] = None
    schedule_in_seconds: Optional[int] = None  # schedule delay

class SelfHealResponse(BaseModel):
    ok: bool
    action_id: str
    message: Optional[str] = None
    dry_run: Optional[bool] = None
    scheduled_at: Optional[str] = None

# Utility: persist action to queue (Redis list) and audit
async def _enqueue_action(action: SelfHealAction, r: Optional[Any] = None):
    """
    Persist action to SELF_HEAL_QUEUE (Redis list) or an FS fallback.
    """
    try:
        if r is None:
            r = await get_redis()
        payload = action.dict()
        if r:
            await r.lpush(SELF_HEAL_QUEUE, json.dumps(payload, default=str))
            LOG.info("Enqueued self-heal action to redis queue: %s", action.action_id)
        else:
            # FS fallback: write to file
            qf = pathlib.Path.cwd() / "backend" / "logs" / "self_heal_queue.jsonl"
            qf.parent.mkdir(parents=True, exist_ok=True)
            with open(qf, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
            LOG.info("Enqueued self-heal action to FS queue: %s", action.action_id)
        await write_audit_event({"event": "self_heal_enqueued", "action": payload})
        return True
    except Exception:
        LOG.exception("Failed to enqueue action")
        return False

# Notification helpers
async def _send_webhook_notification(payload: Dict[str, Any]):
    """
    Post a JSON payload to configured webhooks (one or more) synchronously.
    Non-blocking best-effort; returns list of results.
    """
    urls = [u.strip() for u in (SELF_HEAL_WEBHOOKS or "").split(",") if u.strip()]
    results = []
    if not urls:
        LOG.debug("No self-heal webhooks configured")
        return results
    for url in urls:
        try:
            if _HAS_REQUESTS:
                resp = requests.post(url, json=payload, timeout=5)
                results.append({"url": url, "status_code": resp.status_code})
            else:
                LOG.debug("requests not installed, skipping webhook post to %s", url)
                results.append({"url": url, "status": "requests_missing"})
        except Exception:
            LOG.exception("Webhook post to %s failed", url)
            results.append({"url": url, "error": "exception"})
    await write_audit_event({"event": "self_heal_webhook", "payload": payload, "results": results})
    return results

# Kubernetes helpers
async def _k8s_delete_pod(pod_name: str, namespace: Optional[str] = None, dry_run: bool = True):
    """
    Attempt to delete a pod (k8s will recreate it if controlled by a Deployment).
    Returns a dict with outcome details.
    """
    if not _HAS_K8S:
        return {"ok": False, "error": "kubernetes_client_not_installed"}
    ns = namespace or os.getenv("POD_NAMESPACE", "default")
    if dry_run:
        return {"ok": True, "action": "dry_run", "pod": pod_name, "namespace": ns}
    try:
        # load config
        try:
            if K8S_IN_CLUSTER:
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception:
                pass
        v1 = k8s_client.CoreV1Api()
        body = k8s_client.V1DeleteOptions(grace_period_seconds=30)
        resp = v1.delete_namespaced_pod(name=pod_name, namespace=ns, body=body)
        return {"ok": True, "resp": str(resp)}
    except Exception as e:
        LOG.exception("Failed to delete pod %s/%s", pod_name, ns)
        return {"ok": False, "error": str(e)}

async def _k8s_rolling_restart_deployment(deployment: str, namespace: Optional[str] = None, dry_run: bool = True):
    """
    Trigger a rolling restart by patching the deployment's annotation (kubectl rollout restart equivalent).
    """
    if not _HAS_K8S:
        return {"ok": False, "error": "kubernetes_client_not_installed"}
    ns = namespace or os.getenv("POD_NAMESPACE", "default")
    if dry_run:
        return {"ok": True, "action": "dry_run", "deployment": deployment, "namespace": ns}
    try:
        try:
            if K8S_IN_CLUSTER:
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception:
                pass
        apps = k8s_client.AppsV1Api()
        now = datetime.datetime.utcnow().isoformat() + "Z"
        patch = {"spec": {"template": {"metadata": {"annotations": {"prioritymax/self-heal-restart": now}}}}}
        resp = apps.patch_namespaced_deployment(name=deployment, namespace=ns, body=patch)
        return {"ok": True, "resp": str(resp)}
    except Exception as e:
        LOG.exception("Failed rolling restart for deployment %s/%s", deployment, ns)
        return {"ok": False, "error": str(e)}

# Core action execution (non-blocking)
async def _execute_action(action: SelfHealAction):
    """
    Execute or simulate the requested self-heal action.
    This function is safe to call from background tasks/workers.
    """
    LOG.info("Executing self-heal action %s (action=%s, target=%s)", action.action_id, action.action, action.target)
    dry_run = (action.dry_run if action.dry_run is not None else SELF_HEAL_DEFAULT_DRY_RUN)
    result = {"attempted_at": datetime.datetime.utcnow().isoformat() + "Z", "dry_run": dry_run}

    # map action to handler
    if action.action == "restart_worker":
        # publish a restart hint to redis channel
        payload = {"hint_id": action.action_id, "type": "restart_worker", "worker_id": action.target, "reason": action.reason or "operator_request", "ts": _now_ts()}
        r = await get_redis()
        if r:
            if dry_run:
                result.update({"published": False, "note": "dry_run"})
            else:
                try:
                    await r.publish(SELF_HEAL_CHANNEL, json.dumps(payload))
                    result.update({"published": True})
                except Exception:
                    LOG.exception("Failed publish restart_worker hint")
                    result.update({"published": False, "error": "publish_failed"})
        else:
            # FS fallback
            qf = pathlib.Path.cwd() / "backend" / "logs" / "self_heal_hints.jsonl"
            qf.parent.mkdir(parents=True, exist_ok=True)
            with open(qf, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
            result.update({"published": True, "fallback": "fs"})
        await write_audit_event({"event": "self_heal_restart_worker", "action": action.dict(), "result": result})
        await _send_webhook_notification({"type": "restart_worker", "action_id": action.action_id, "target": action.target, "dry_run": dry_run})
        return result

    elif action.action == "restart_pod":
        # attempt k8s pod delete (or dry run)
        pod = action.target
        ns = action.namespace
        res = await _k8s_delete_pod(pod, namespace=ns, dry_run=dry_run)
        result.update(res)
        await write_audit_event({"event": "self_heal_restart_pod", "action": action.dict(), "result": result})
        await _send_webhook_notification({"type": "restart_pod", "action_id": action.action_id, "target": pod, "namespace": ns, "dry_run": dry_run, "result": res})
        return result

    elif action.action == "rolling_restart":
        deployment = action.target
        ns = action.namespace
        res = await _k8s_rolling_restart_deployment(deployment, namespace=ns, dry_run=dry_run)
        result.update(res)
        await write_audit_event({"event": "self_heal_rolling_restart", "action": action.dict(), "result": result})
        await _send_webhook_notification({"type": "rolling_restart", "action_id": action.action_id, "target": deployment, "namespace": ns, "dry_run": dry_run, "result": res})
        return result

    elif action.action == "flush_cache":
        # flush probe cache (in-memory) and optionally instruct services to evict caches
        # we do an audit and publish a hint
        r = await get_redis()
        payload = {"hint_id": action.action_id, "type": "flush_cache", "target": action.target, "reason": action.reason or "operator_request", "ts": _now_ts()}
        if dry_run:
            result.update({"flushed": False, "note": "dry_run"})
        else:
            if r:
                try:
                    await r.publish(SELF_HEAL_CHANNEL, json.dumps(payload))
                    result.update({"published": True})
                except Exception:
                    LOG.exception("Failed to publish flush_cache hint")
                    result.update({"published": False, "error": "publish_failed"})
            else:
                qf = pathlib.Path.cwd() / "backend" / "logs" / "self_heal_hints.jsonl"
                qf.parent.mkdir(parents=True, exist_ok=True)
                with open(qf, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload) + "\n")
                result.update({"published": True, "fallback": "fs"})
        await write_audit_event({"event": "self_heal_flush_cache", "action": action.dict(), "result": result})
        await _send_webhook_notification({"type": "flush_cache", "action_id": action.action_id, "target": action.target, "dry_run": dry_run, "result": result})
        return result

    elif action.action == "noop":
        result.update({"ok": True, "note": "noop"})
        await write_audit_event({"event": "self_heal_noop", "action": action.dict(), "result": result})
        return result

    else:
        result.update({"ok": False, "error": "unsupported_action"})
        await write_audit_event({"event": "self_heal_unsupported", "action": action.dict(), "result": result})
        return result

# Worker: background processor that consumes SELF_HEAL_QUEUE and executes actions (for local deployments)
_SELF_HEAL_PROCESSOR_TASK: Optional[asyncio.Task] = None
_SELF_HEAL_PROCESSOR_RUNNING = False

async def _self_heal_processor_loop(poll_interval: int = 2):
    """
    Background task to process queued self-heal actions. It pops actions off the Redis list and executes them.
    Safe default: respects dry_run flag embedded into actions.
    """
    global _SELF_HEAL_PROCESSOR_RUNNING
    _SELF_HEAL_PROCESSOR_RUNNING = True
    LOG.info("Self-heal processor loop started")
    r = await get_redis()
    try:
        while True:
            try:
                if r:
                    raw = await r.rpop(SELF_HEAL_QUEUE)
                    if not raw:
                        await asyncio.sleep(poll_interval)
                        continue
                    try:
                        payload = json.loads(raw)
                        action = SelfHealAction(**payload)
                    except Exception:
                        LOG.exception("Malformed self-heal action payload: %s", raw)
                        continue
                    # execute action with rate limiting
                    if await _is_rate_limited(action.action, action.target):
                        LOG.warning("Self-heal action rate-limited: %s", action.action_id)
                        action.status = "failed"
                        action.result = {"error": "rate_limited"}
                        await write_audit_event({"event": "self_heal_rate_limited", "action": action.dict()})
                        continue
                    try:
                        action.status = "executing"
                        await write_audit_event({"event": "self_heal_execute_start", "action": action.dict()})
                        res = await _execute_action(action)
                        action.status = "executed"
                        action.result = res
                        await write_audit_event({"event": "self_heal_execute_end", "action": action.dict(), "result": res})
                    except Exception:
                        LOG.exception("Execution of self-heal action failed")
                        action.status = "failed"
                        action.result = {"error": "exception"}
                        await write_audit_event({"event": "self_heal_execute_error", "action": action.dict()})
                else:
                    # no redis configured: process FS queue file (not implemented as continuous poll to avoid complexity)
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("Self-heal processor loop iteration failed")
                await asyncio.sleep(poll_interval)
    finally:
        _SELF_HEAL_PROCESSOR_RUNNING = False
        LOG.info("Self-heal processor loop stopped")

def start_self_heal_processor(loop: Optional[asyncio.AbstractEventLoop] = None):
    global _SELF_HEAL_PROCESSOR_TASK
    if _SELF_HEAL_PROCESSOR_TASK and not _SELF_HEAL_PROCESSOR_TASK.done():
        return
    loop = loop or asyncio.get_event_loop()
    _SELF_HEAL_PROCESSOR_TASK = loop.create_task(_self_heal_processor_loop())
    LOG.info("Scheduled self-heal processor background task")

def stop_self_heal_processor():
    global _SELF_HEAL_PROCESSOR_TASK
    if _SELF_HEAL_PROCESSOR_TASK:
        _SELF_HEAL_PROCESSOR_TASK.cancel()
        _SELF_HEAL_PROCESSOR_TASK = None
        LOG.info("Stopped self-heal processor background task")

# -----------------------------
# API Endpoints (admin-protected)
# -----------------------------
@router.post("/request", dependencies=[Depends(require_role(Role.OPERATOR))])
async def request_self_heal(req: SelfHealRequest = Body(...), user = Depends(get_current_user)):
    """
    Request a self-heal action. This endpoint enqueues the action (persistent) and returns an action_id.
    - Actions: restart_worker, restart_pod, rolling_restart, flush_cache, noop
    - schedule_in_seconds: optional delay before processing
    - dry_run: optional explicit dry run flag (defaults to env)
    """
    # Validate action
    allowed = {"restart_worker", "restart_pod", "rolling_restart", "flush_cache", "noop"}
    if req.action not in allowed:
        raise HTTPException(status_code=400, detail=f"unsupported action {req.action}")

    dry = req.dry_run if req.dry_run is not None else SELF_HEAL_DEFAULT_DRY_RUN
    action = SelfHealAction(
        action=req.action,
        target=req.target,
        namespace=req.namespace,
        reason=req.reason,
        dry_run=dry,
        requested_by=(getattr(user, "username", None) or "unknown")
    )

    # rate-limit check immediately to avoid queueing many requests
    if await _is_rate_limited(action.action, action.target):
        await write_audit_event({"event": "self_heal_request_rate_limited", "action": action.dict(), "user": getattr(user, "username", "unknown")})
        raise HTTPException(status_code=429, detail="rate_limited")

    # schedule or enqueue now
    if req.schedule_in_seconds and int(req.schedule_in_seconds) > 0:
        scheduled_at = datetime.datetime.utcnow() + datetime.timedelta(seconds=int(req.schedule_in_seconds))
        action.status = "scheduled"
        await _enqueue_action(action)
        await write_audit_event({"event": "self_heal_scheduled", "action": action.dict(), "scheduled_at": scheduled_at.isoformat() + "Z", "user": getattr(user, "username", "unknown")})
        return SelfHealResponse(ok=True, action_id=action.action_id, message="scheduled", dry_run=dry, scheduled_at=scheduled_at.isoformat() + "Z")
    else:
        # immediate enqueue on queue
        ok = await _enqueue_action(action)
        if not ok:
            raise HTTPException(status_code=500, detail="enqueue_failed")
        await write_audit_event({"event": "self_heal_requested", "action": action.dict(), "user": getattr(user, "username", "unknown")})
        return SelfHealResponse(ok=True, action_id=action.action_id, message="enqueued", dry_run=dry, scheduled_at=None)

@router.get("/inspect", dependencies=[Depends(require_role(Role.VIEWER))])
async def inspect_queue(limit: int = Query(50), tail: bool = Query(True)):
    """
    Inspect pending self-heal actions (Redis list or FS queue fallback).
    """
    r = await get_redis()
    actions = []
    if r:
        try:
            # LRANGE to read queue items
            raw = await r.lrange(SELF_HEAL_QUEUE, 0, limit - 1)
            for item in raw:
                try:
                    actions.append(json.loads(item))
                except Exception:
                    actions.append({"raw": item})
            return {"count": len(actions), "actions": actions}
        except Exception:
            LOG.exception("Failed to inspect redis self-heal queue")
    # FS fallback: read last N lines
    qf = pathlib.Path.cwd() / "backend" / "logs" / "self_heal_queue.jsonl"
    if not qf.exists():
        return {"count": 0, "actions": []}
    with open(qf, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    if tail:
        lines = lines[-limit:]
    for l in lines:
        try:
            actions.append(json.loads(l))
        except Exception:
            actions.append({"raw": l.strip()})
    return {"count": len(actions), "actions": actions}

@router.post("/execute_now/{action_id}", dependencies=[Depends(require_role(Role.ADMIN))])
async def execute_action_now(action_id: str = Query(...), user = Depends(get_current_user)):
    """
    Force immediate execution of a queued action (admin-only).
    Attempts to locate the action in Redis queue or FS queue. If found, executes directly.
    """
    r = await get_redis()
    found = None
    if r:
        try:
            raw_items = await r.lrange(SELF_HEAL_QUEUE, 0, -1)
            for item in raw_items:
                try:
                    obj = json.loads(item)
                    if obj.get("action_id") == action_id:
                        found = obj
                        # remove this specific item using LREM
                        await r.lrem(SELF_HEAL_QUEUE, 0, item)
                        break
                except Exception:
                    continue
        except Exception:
            LOG.exception("Failed to scan redis queue for execute_now")
    if not found:
        # FS fallback: scan and rewrite file excluding target
        qf = pathlib.Path.cwd() / "backend" / "logs" / "self_heal_queue.jsonl"
        if qf.exists():
            kept = []
            with open(qf, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if obj.get("action_id") == action_id:
                            found = obj
                            continue
                    except Exception:
                        pass
                    kept.append(line)
            if found:
                with open(qf, "w", encoding="utf-8") as fh:
                    fh.writelines(kept)
    if not found:
        raise HTTPException(status_code=404, detail="action_not_found")
    action = SelfHealAction(**found)
    # execute directly (synchronous wrapper)
    if await _is_rate_limited(action.action, action.target):
        raise HTTPException(status_code=429, detail="rate_limited")
    # run action
    res = await _execute_action(action)
    action.status = "executed"
    action.result = res
    await write_audit_event({"event": "self_heal_execute_now", "action": action.dict(), "requested_by": getattr(user, "username", "unknown")})
    return {"ok": True, "action_id": action_id, "result": res}

@router.post("/healthcheck", dependencies=[Depends(require_role(Role.VIEWER))])
async def self_heal_health():
    """
    Simple status endpoint for the self-heal subsystem.
    """
    r = await get_redis()
    return {
        "redis": bool(r),
        "kubernetes_client": bool(_HAS_K8S),
        "processor_running": _SELF_HEAL_PROCESSOR_RUNNING,
        "queue_channel": SELF_HEAL_CHANNEL,
        "queue_name": SELF_HEAL_QUEUE
    }

# Startup/shutdown hooks to start processor when module loaded into FastAPI
@router.on_event("startup")
async def _on_self_heal_startup():
    # warm redis client
    try:
        await get_redis()
    except Exception:
        pass
    # start processor only if redis configured (we assume Redis deployment will handle queue consumption in production)
    if await get_redis():
        start_self_heal_processor()
    LOG.info("Self-heal subsystem started (processor_running=%s)", _SELF_HEAL_PROCESSOR_RUNNING)

@router.on_event("shutdown")
async def _on_self_heal_shutdown():
    stop_self_heal_processor()
    LOG.info("Self-heal subsystem stopped")

# Expose helper utilities (for unit tests / operators)
__all__ = [
    "router",
    "request_self_heal",
    "start_self_heal_processor",
    "stop_self_heal_processor",
    "execute_action_now",
    "inspect_queue",
    "get_redis",
]

# Backwards-compat shim for external callers
async def request_self_heal(action: str, target: Optional[str] = None, namespace: Optional[str] = None, reason: Optional[str] = None, dry_run: Optional[bool] = None, requester: Optional[str] = None, schedule_in_seconds: Optional[int] = None):
    """
    Programmatic helper to enqueue a self-heal action (used by other modules).
    Returns the action_id and enqueue result.
    """
    req = SelfHealRequest(action=action, target=target, namespace=namespace, reason=reason, dry_run=dry_run, schedule_in_seconds=schedule_in_seconds)
    # mimic API auth context for requested_by
    user = type("U", (), {"username": requester or "system"})()
    # validate
    if req.action not in {"restart_worker", "restart_pod", "rolling_restart", "flush_cache", "noop"}:
        raise ValueError("unsupported action")
    dry = req.dry_run if req.dry_run is not None else SELF_HEAL_DEFAULT_DRY_RUN
    action_obj = SelfHealAction(
        action=req.action,
        target=req.target,
        namespace=req.namespace,
        reason=req.reason,
        dry_run=dry,
        requested_by=requester or "system"
    )
    # rate-limit check
    if await _is_rate_limited(action_obj.action, action_obj.target):
        await write_audit_event({"event": "self_heal_request_rate_limited", "action": action_obj.dict(), "user": requester or "system"})
        return {"ok": False, "reason": "rate_limited"}
    # enqueue
    ok = await _enqueue_action(action_obj)
    if not ok:
        return {"ok": False, "reason": "enqueue_failed"}
    return {"ok": True, "action_id": action_obj.action_id}

# End of file

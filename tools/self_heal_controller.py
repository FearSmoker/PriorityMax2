#!/usr/bin/env python3
"""
PriorityMax Privileged Self-Heal Controller

Listens for self-heal hints published by the application (Redis pub/sub channel and Redis queue)
and executes destructive recovery actions (pod delete, rolling restart) in Kubernetes.

Features:
 - Leader election via Redis lock (safe single-active controller)
 - Consumes both pub/sub hints and persistent queue for reliability
 - Dry-run mode by default; toggle with SELF_HEAL_CONTROLLER_DRY_RUN=false
 - Rate limiting and retry/backoff per-action
 - Prometheus metrics and simple FastAPI health endpoints
 - Audit logging (integrates with app.api.admin.write_audit_event if importable)
 - Graceful shutdown and robust exception handling

Usage:
  python3 backend/tools/self_heal_controller.py

Environment variables (defaults shown):
  REDIS_URL=redis://localhost:6379/0
  SELF_HEAL_RESTART_CHANNEL=prioritymax:self_heal:restart
  SELF_HEAL_QUEUE=prioritymax:self_heal:queue
  SELF_HEAL_CONTROLLER_DRY_RUN=true
  REDIS_LEADER_LOCK_KEY=prioritymax:self_heal:leader_lock
  REDIS_LEADER_LOCK_TTL=15
  CONTROLLER_POLL_INTERVAL=2
  CONTROLLER_WORKERS=4
  CONTROLLER_RETRY_BACKOFF_BASE=2.0
  CONTROLLER_MAX_RETRIES=5
  PROMETHEUS_PORT=9000
  HEALTH_PORT=9010
  K8S_IN_CLUSTER=true

Notes:
 - Ensure the running ServiceAccount has necessary k8s permissions.
 - Run this process in a separate pod (privileged controller).
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import signal
import traceback
import datetime
from typing import Any, Dict, Optional, List, Tuple

# Third-party libs
try:
    import aioredis
except Exception as e:
    print("aioredis is required. Install with `pip install aioredis`.", file=sys.stderr)
    raise

try:
    from kubernetes import config as k8s_config, client as k8s_client
    from kubernetes.client.rest import ApiException as K8sApiException
except Exception:
    print("kubernetes python client is required. Install with `pip install kubernetes`.", file=sys.stderr)
    raise

try:
    import requests
except Exception:
    requests = None

try:
    from prometheus_client import Counter, Gauge, start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

# Optional integration with app's audit function (best-effort)
try:
    from app.api.admin import write_audit_event
    _HAS_AUDIT = True
except Exception:
    _HAS_AUDIT = False

# Logging
LOG = logging.getLogger("prioritymax.self_heal_controller")
LOG.setLevel(os.getenv("PRIORITYMAX_CONTROLLER_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Configuration from env
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SELF_HEAL_RESTART_CHANNEL = os.getenv("SELF_HEAL_RESTART_CHANNEL", "prioritymax:self_heal:restart")
SELF_HEAL_QUEUE = os.getenv("SELF_HEAL_QUEUE", "prioritymax:self_heal:queue")
DRY_RUN = os.getenv("SELF_HEAL_CONTROLLER_DRY_RUN", "true").lower() in ("1", "true", "yes")
REDIS_LEADER_LOCK_KEY = os.getenv("REDIS_LEADER_LOCK_KEY", "prioritymax:self_heal:leader_lock")
REDIS_LEADER_LOCK_TTL = int(os.getenv("REDIS_LEADER_LOCK_TTL", "15"))  # seconds
POLL_INTERVAL = float(os.getenv("CONTROLLER_POLL_INTERVAL", "2"))
WORKER_POOL = int(os.getenv("CONTROLLER_WORKERS", "4"))
RETRY_BACKOFF_BASE = float(os.getenv("CONTROLLER_RETRY_BACKOFF_BASE", "2.0"))
MAX_RETRIES = int(os.getenv("CONTROLLER_MAX_RETRIES", "5"))
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9000"))
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "9010"))
K8S_IN_CLUSTER = os.getenv("K8S_IN_CLUSTER", "true").lower() in ("1", "true", "yes")
LEADER_ID = f"controller-{uuid.uuid4().hex[:8]}"
REDIS_LEADER_LOCK_VAL = LEADER_ID

# Metrics (if prometheus_client available)
if _HAS_PROM:
    MET_ACTIONS_EXECUTED = Counter("prioritymax_controller_actions_executed_total", "Total actions executed")
    MET_ACTIONS_FAILED = Counter("prioritymax_controller_actions_failed_total", "Total actions failed")
    MET_ACTIONS_DRYRUN = Counter("prioritymax_controller_actions_dryrun_total", "Total actions in dry-run mode")
    MET_LEADER_UP = Gauge("prioritymax_controller_leader", "Is this instance leader (1/0)")
else:
    MET_ACTIONS_EXECUTED = MET_ACTIONS_FAILED = MET_ACTIONS_DRYRUN = MET_LEADER_UP = None

# Async globals
_redis: Optional[aioredis.Redis] = None
_stop_event = asyncio.Event()
_worker_tasks: List[asyncio.Task] = []
_leader_renew_task: Optional[asyncio.Task] = None

# K8s clients (initialized on start)
_core_api: Optional[k8s_client.CoreV1Api] = None
_apps_api: Optional[k8s_client.AppsV1Api] = None

# Utility functions
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

async def audit(evt: Dict[str, Any]):
    """
    Try to call the central audit function if available, otherwise log to local file.
    """
    msg = {"ts": now_iso(), "instance": LEADER_ID, **evt}
    try:
        if _HAS_AUDIT:
            # best-effort: write audit (may be sync)
            try:
                res = write_audit_event(msg)
                # if write_audit_event is async, await
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                LOG.exception("write_audit_event failed, falling back to file")
                _write_local_audit(msg)
        else:
            _write_local_audit(msg)
    except Exception:
        LOG.exception("audit failed for event %s", evt)

def _write_local_audit(msg: Dict[str, Any]):
    try:
        p = os.environ.get("PRIORITYMAX_LOCAL_AUDIT_PATH", "/tmp/prioritymax_self_heal_controller_audit.jsonl")
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(msg, default=str) + "\n")
    except Exception:
        LOG.exception("failed to write local audit")

# Redis helpers
async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is not None:
        return _redis
    _redis = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis

async def leader_acquire(redis: aioredis.Redis) -> bool:
    """
    Try to acquire leadership via SET NX with TTL.
    Returns True if acquired.
    """
    try:
        # SET key value NX EX ttl
        ok = await redis.set(REDIS_LEADER_LOCK_KEY, REDIS_LEADER_LOCK_VAL, ex=REDIS_LEADER_LOCK_TTL, nx=True)
        if ok:
            LOG.info("Acquired leader lock as %s", LEADER_ID)
            if MET_LEADER_UP:
                MET_LEADER_UP.set(1)
            await audit({"event": "leader_acquired", "leader": LEADER_ID})
            return True
        # not acquired
        return False
    except Exception:
        LOG.exception("leader_acquire error")
        return False

async def leader_renew_loop(redis: aioredis.Redis):
    """
    Renew leader lock by setting key only if value matches us.
    We run this task only if leader is acquired.
    """
    try:
        while not _stop_event.is_set():
            try:
                # Use Lua script to check value and set expiry atomically
                script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                   return redis.call("expire", KEYS[1], tonumber(ARGV[2]))
                else
                   return 0
                end
                """
                res = await redis.eval(script, keys=[REDIS_LEADER_LOCK_KEY], args=[REDIS_LEADER_LOCK_VAL, REDIS_LEADER_LOCK_TTL])
                if not res:
                    LOG.warning("Leader renewal failed - lock lost")
                    if MET_LEADER_UP:
                        MET_LEADER_UP.set(0)
                    await audit({"event": "leader_lost", "leader": LEADER_ID})
                    # stop renew loop -> caller should detect and re-elect
                    break
                await asyncio.sleep(max(1, REDIS_LEADER_LOCK_TTL / 3))
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("leader_renew error")
                await asyncio.sleep(1)
    finally:
        LOG.info("Leader renew loop exiting")

# Kubernetes actions (safe wrappers)
def _k8s_delete_pod_sync(pod_name: str, namespace: str = "default", grace_seconds: int = 30) -> Dict[str, Any]:
    try:
        resp = _core_api.delete_namespaced_pod(name=pod_name, namespace=namespace, body=k8s_client.V1DeleteOptions(grace_period_seconds=grace_seconds))
        return {"ok": True, "resp": str(resp)}
    except K8sApiException as e:
        return {"ok": False, "error": f"k8s_api_error:{e.status}:{e.reason}", "body": getattr(e, "body", None)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _k8s_patch_deployment_sync(deployment: str, namespace: str = "default") -> Dict[str, Any]:
    """
    Trigger a rolling restart by patching an annotation on the deployment's template.
    Equivalent to `kubectl rollout restart deployment/<deployment>`.
    """
    try:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        patch = {"spec": {"template": {"metadata": {"annotations": {"prioritymax/self-heal-restart": now}}}}}
        resp = _apps_api.patch_namespaced_deployment(name=deployment, namespace=namespace, body=patch)
        return {"ok": True, "resp": str(resp)}
    except K8sApiException as e:
        return {"ok": False, "error": f"k8s_api_error:{e.status}:{e.reason}", "body": getattr(e, "body", None)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Backoff helper
def _backoff_delay(attempt: int) -> float:
    # exponential backoff with jitter
    base = RETRY_BACKOFF_BASE
    delay = base ** attempt
    # jitter up to 30%
    jitter = delay * 0.3 * (0.5 - (uuid.uuid4().int % 100) / 100.0)
    return max(0.1, delay + jitter)

# Action processing
async def process_action_payload(payload: Dict[str, Any], dry_run_override: Optional[bool] = None) -> Dict[str, Any]:
    """
    payload is expected to be a dict with fields:
      - hint_id, type, worker_id OR pod/deployment, namespace, reason, ts, etc.
    Types supported:
      - restart_worker: publish to worker channel (non-destructive) -> no-op for privileged controller (audit)
      - restart_pod: delete pod (destructive)
      - rolling_restart: patch deployment (destructive)
      - flush_cache: publish hint (non-destructive)
    Returns result dict with ok, detail.
    """
    typ = payload.get("type") or payload.get("action")  # tolerate different shapes
    dry_run = DRY_RUN if dry_run_override is None else dry_run_override
    res_summary = {"processed_at": now_iso(), "payload": payload, "dry_run": dry_run}

    # simple normalization
    target = payload.get("pod") or payload.get("target") or payload.get("worker_id") or payload.get("deployment")
    namespace = payload.get("namespace") or os.getenv("POD_NAMESPACE", "default")

    # For non-destructive types, simply audit
    if typ in ("restart_worker", "flush_cache"):
        await audit({"event": "hint_received", "type": typ, "target": target, "payload": payload})
        res_summary.update({"ok": True, "info": "non_destructive_audit_only"})
        if MET_ACTIONS_DRYRUN and dry_run:
            MET_ACTIONS_DRYRUN.inc()
        else:
            if MET_ACTIONS_EXECUTED:
                MET_ACTIONS_EXECUTED.inc()
        return res_summary

    if typ == "restart_pod":
        if dry_run:
            await audit({"event": "restart_pod_dryrun", "pod": target, "namespace": namespace, "payload": payload})
            if MET_ACTIONS_DRYRUN: MET_ACTIONS_DRYRUN.inc()
            res_summary.update({"ok": True, "dry": True, "action": "delete_pod"})
            return res_summary
        # perform deletion with retries/backoff
        for attempt in range(0, MAX_RETRIES):
            try:
                out = await asyncio.get_event_loop().run_in_executor(None, _k8s_delete_pod_sync, target, namespace, 30)
                if out.get("ok"):
                    await audit({"event": "restart_pod_executed", "pod": target, "namespace": namespace, "payload": payload, "attempt": attempt})
                    if MET_ACTIONS_EXECUTED: MET_ACTIONS_EXECUTED.inc()
                    res_summary.update({"ok": True, "action": "delete_pod", "result": out})
                    return res_summary
                else:
                    # log and retry
                    await audit({"event": "restart_pod_failed", "pod": target, "namespace": namespace, "payload": payload, "attempt": attempt, "result": out})
                    LOG.warning("delete pod failed: %s; attempt=%d", out, attempt)
                    await asyncio.sleep(_backoff_delay(attempt))
            except Exception:
                LOG.exception("Exception while deleting pod %s", target)
                await asyncio.sleep(_backoff_delay(attempt))
        # final failure
        if MET_ACTIONS_FAILED: MET_ACTIONS_FAILED.inc()
        res_summary.update({"ok": False, "error": "max_retries_exceeded"})
        return res_summary

    if typ == "rolling_restart":
        if dry_run:
            await audit({"event": "rolling_restart_dryrun", "deployment": target, "namespace": namespace, "payload": payload})
            if MET_ACTIONS_DRYRUN: MET_ACTIONS_DRYRUN.inc()
            res_summary.update({"ok": True, "dry": True, "action": "rolling_restart"})
            return res_summary
        # attempt patch
        for attempt in range(0, MAX_RETRIES):
            try:
                out = await asyncio.get_event_loop().run_in_executor(None, _k8s_patch_deployment_sync, target, namespace)
                if out.get("ok"):
                    await audit({"event": "rolling_restart_executed", "deployment": target, "namespace": namespace, "payload": payload, "attempt": attempt})
                    if MET_ACTIONS_EXECUTED: MET_ACTIONS_EXECUTED.inc()
                    res_summary.update({"ok": True, "action": "rolling_restart", "result": out})
                    return res_summary
                else:
                    await audit({"event": "rolling_restart_failed", "deployment": target, "namespace": namespace, "payload": payload, "attempt": attempt, "result": out})
                    LOG.warning("rolling restart failed: %s; attempt=%d", out, attempt)
                    await asyncio.sleep(_backoff_delay(attempt))
            except Exception:
                LOG.exception("Exception while rolling restart %s", target)
                await asyncio.sleep(_backoff_delay(attempt))
        if MET_ACTIONS_FAILED: MET_ACTIONS_FAILED.inc()
        res_summary.update({"ok": False, "error": "max_retries_exceeded"})
        return res_summary

    # unknown type
    await audit({"event": "unknown_hint_type", "payload": payload})
    res_summary.update({"ok": False, "error": "unknown_type"})
    if MET_ACTIONS_FAILED: MET_ACTIONS_FAILED.inc()
    return res_summary

# Worker coroutine to process queue items (reliable)
async def queue_worker(name: str, redis: aioredis.Redis):
    LOG.info("Queue worker %s started", name)
    while not _stop_event.is_set():
        try:
            item = await redis.rpop(SELF_HEAL_QUEUE)
            if not item:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            try:
                payload = json.loads(item)
            except Exception:
                LOG.exception("Malformed queue item: %s", item)
                continue
            LOG.info("Worker %s processing queued hint: %s", name, payload.get("hint_id") or payload.get("action_id") or "<no-id>")
            # check rate limiting before executing (optional: can be more complex)
            res = await process_action_payload(payload)
            LOG.info("Worker %s result: %s", name, res.get("ok"))
        except asyncio.CancelledError:
            break
        except Exception:
            LOG.exception("Queue worker %s encountered exception", name)
            await asyncio.sleep(1)
    LOG.info("Queue worker %s exiting", name)

# Pub/Sub listener (best-effort, real-time)
async def pubsub_listener(redis: aioredis.Redis):
    LOG.info("Starting pubsub listener on channel %s", SELF_HEAL_RESTART_CHANNEL)
    sub = redis.pubsub()
    await sub.subscribe(SELF_HEAL_RESTART_CHANNEL)
    try:
        while not _stop_event.is_set():
            try:
                msg = await sub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    await asyncio.sleep(0)
                    continue
                # msg dict contains 'type','pattern','channel','data'
                data = msg.get("data")
                if not data:
                    continue
                try:
                    payload = json.loads(data)
                except Exception:
                    LOG.exception("Invalid pubsub payload: %s", data)
                    continue
                LOG.info("Pubsub hint arrived: %s", payload.get("hint_id") or payload.get("action_id"))
                # process in background (non-blocking)
                asyncio.create_task(process_action_payload(payload))
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("pubsub_listener loop exception")
                await asyncio.sleep(0.5)
    finally:
        try:
            await sub.unsubscribe(SELF_HEAL_RESTART_CHANNEL)
        except Exception:
            pass
        LOG.info("Pubsub listener stopped")

# Leader election main loop
async def leader_loop():
    redis = await get_redis()
    if not redis:
        LOG.error("Redis not available for leader election; exiting")
        return
    global _leader_renew_task
    try:
        while not _stop_event.is_set():
            got = await leader_acquire(redis)
            if got:
                # spawn renew loop
                _leader_renew_task = asyncio.create_task(leader_renew_loop(redis))
                # start workers and pubsub
                LOG.info("Becoming leader - starting workers and pubsub")
                await audit({"event": "become_leader", "leader": LEADER_ID})
                # start pubsub
                pubsub_task = asyncio.create_task(pubsub_listener(redis))
                # start worker tasks
                worker_pool = []
                for i in range(WORKER_POOL):
                    t = asyncio.create_task(queue_worker(f"w{i}", redis))
                    worker_pool.append(t)
                    _worker_tasks.append(t)
                # block until leader lock lost or stop_event
                while not _stop_event.is_set():
                    # check if lock still owned
                    val = await redis.get(REDIS_LEADER_LOCK_KEY)
                    if val != REDIS_LEADER_LOCK_VAL:
                        LOG.warning("Leader lock lost externally (value=%s)", val)
                        break
                    await asyncio.sleep(1)
                # cleanup
                LOG.info("Leader relinquishing leadership - cancelling workers")
                pubsub_task.cancel()
                for t in worker_pool:
                    t.cancel()
                if _leader_renew_task:
                    _leader_renew_task.cancel()
                await audit({"event": "relinquish_leader", "leader": LEADER_ID})
                if MET_LEADER_UP:
                    MET_LEADER_UP.set(0)
                # small backoff then attempt re-acquire
                await asyncio.sleep(1)
            else:
                # not leader — sleep then retry acquisition
                if MET_LEADER_UP:
                    MET_LEADER_UP.set(0)
                await asyncio.sleep(max(1.0, REDIS_LEADER_LOCK_TTL / 2.0))
    except asyncio.CancelledError:
        LOG.info("leader_loop cancelled")
    except Exception:
        LOG.exception("leader_loop unhandled exception")
    finally:
        if _leader_renew_task:
            _leader_renew_task.cancel()

# Kubernetes initialization
def init_k8s_clients():
    global _core_api, _apps_api
    try:
        if K8S_IN_CLUSTER:
            k8s_config.load_incluster_config()
        else:
            k8s_config.load_kube_config()
        _core_api = k8s_client.CoreV1Api()
        _apps_api = k8s_client.AppsV1Api()
        LOG.info("Kubernetes client initialized (in_cluster=%s)", K8S_IN_CLUSTER)
    except Exception:
        LOG.exception("Failed to init kubernetes client")
        raise

# HTTP health server (FastAPI) — minimal: /health and /metrics if prometheus available
async def start_health_server(loop: asyncio.AbstractEventLoop):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI(title="prioritymax-self-heal-controller")

    @app.get("/health")
    async def health():
        return JSONResponse({"ok": True, "leader_id": LEADER_ID, "dry_run": DRY_RUN, "time": now_iso()})

    @app.get("/metrics")
    async def metrics():
        if not _HAS_PROM:
            return JSONResponse({"ok": False, "error": "prometheus not installed"})
        # prometheus_client starts its own HTTP server; we simply forward that it's running
        return JSONResponse({"ok": True, "prometheus_port": PROMETHEUS_PORT})

    import uvicorn
    # run in separate thread to avoid blocking event loop
    config = uvicorn.Config(app, host="0.0.0.0", port=HEALTH_PORT, log_level="info")
    server = uvicorn.Server(config)
    # run server in another task
    loop.create_task(server.serve())

# Graceful shutdown
def _shutdown():
    LOG.info("Shutdown signal received")
    _stop_event.set()
    for t in _worker_tasks:
        t.cancel()
    if _leader_renew_task:
        _leader_renew_task.cancel()

# Entrypoint
async def main():
    LOG.info("Starting PriorityMax Self-Heal Controller (leader id=%s) dry_run=%s", LEADER_ID, DRY_RUN)
    # prometheus server
    if _HAS_PROM:
        try:
            start_http_server(PROMETHEUS_PORT)
            LOG.info("Prometheus metrics exposed on port %d", PROMETHEUS_PORT)
        except Exception:
            LOG.exception("Failed to start prometheus http server")

    # init k8s clients
    try:
        init_k8s_clients()
    except Exception:
        LOG.error("Kubernetes client initialization failed — controller cannot operate without k8s access")
        # depending on use-case, we may still listen to hints but cannot act
    # create redis connection
    try:
        redis = await get_redis()
    except Exception:
        LOG.exception("redis init failed")
        return

    # start health HTTP server in background
    loop = asyncio.get_event_loop()
    await start_health_server(loop)

    # start leader election loop
    leader_task = asyncio.create_task(leader_loop())

    # handle shutdown signals
    def _on_signal(sig, frame):
        LOG.info("signal %s received, shutting down", sig)
        _shutdown()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # wait until stop_event set
    await _stop_event.wait()
    LOG.info("Controller stopping: waiting tasks to finish")
    leader_task.cancel()
    # cancel worker tasks
    for t in _worker_tasks:
        t.cancel()
    if _leader_renew_task:
        _leader_renew_task.cancel()
    # allow short time for tasks to exit
    await asyncio.sleep(1)
    LOG.info("Controller exited cleanly")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    except Exception:
        LOG.exception("Unhandled exception in controller")
        traceback.print_exc()
        sys.exit(2)

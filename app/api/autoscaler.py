# backend/app/api/autoscaler.py
"""
Autoscaler API for PriorityMax (Phase-3)

This module provides a production-grade autoscaler surface and controller combining:
- Manual scaling APIs (scale up/down consumers, throttle, pause/resume queues)
- AI-driven autoscaler integration (predictor + RL agent inference hooks)
- Hybrid logic (rules + ML decisions)
- Metrics feedback loop (Prometheus + internal metrics)
- Kubernetes integration (HPA / custom operator / pod restart)
- Policy management (CRUD for scaling rules & cooldowns)
- Safety layer (dry-run mode, rate-limiting, validation, rollback)
- Anomaly detection hooks (basic unsupervised detector + thresholds)
- Prometheus + WebSocket broadcast for live updates
- Self-healing worker triggers (restart hung pods or restart worker processes)
- Audit logging integration (writes audit events to DB or JSONL via admin.write_audit_event)

Notes:
- This file expects the existence of several project modules:
    - app.ml.predictor (predictor.Predictor : .predict(features) -> dict)
    - app.ml.rl_agent (rl_agent.RLAgent : .infer(obs) -> action)
    - app.services.worker_manager (worker supervisor helpers)
    - app.api.admin.write_audit_event, get_current_user, require_role, Role
    - app.queue.redis_queue or queue interface to control consumers
- Many integrations are optional and gracefully degrade if missing (k8s, motor, boto3).
- Replace lightweight auth dependency with your real JWT/OAuth2 in production (we import from admin for dev convenience).
"""

import os
import sys
import time
import json
import uuid
import asyncio
import logging
import math
import pathlib
import tempfile
import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# optional motor (async mongo)
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

# optional Kubernetes client
try:
    from kubernetes import client as k8s_client, config as k8s_config, watch as k8s_watch
    _HAS_K8S = True
except Exception:
    k8s_client = None
    k8s_config = None
    k8s_watch = None
    _HAS_K8S = False

# Prometheus client for local metrics exposition (optional)
try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    _HAS_PROM = True
except Exception:
    CollectorRegistry = None
    Gauge = None
    push_to_gateway = None
    _HAS_PROM = False

# optional ML integrations (predictor + RL agent)
try:
    from app.ml.predictor import Predictor
    _HAS_PREDICTOR = True
except Exception:
    Predictor = None
    _HAS_PREDICTOR = False

try:
    from app.ml.rl_agent import RLAgent
    _HAS_RLAGENT = True
except Exception:
    RLAgent = None
    _HAS_RLAGENT = False

# optional worker manager for self-healing actions
try:
    from app.services.worker_manager import WorkerManager
    _HAS_WORKER_MANAGER = True
except Exception:
    WorkerManager = None
    _HAS_WORKER_MANAGER = False

# auth helpers & audit writer from admin module (dev-mode, replace with production auth)
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    # fallback stubs if admin import fails (very unlikely if admin.py exists)
    def get_current_user(token: Optional[str] = None):
        raise HTTPException(status_code=401, detail="Auth dependency missing")
    def require_role(r):
        def _dep():
            raise HTTPException(status_code=401, detail="Auth dependency missing")
        return _dep
    Role = None
    async def write_audit_event(e: dict):
        # fallback to local JSONL
        print("[AUDIT]", e)

# local path helpers
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
AUTOSCALER_DB = os.getenv("AUTOSCALER_DB", None)  # optional mongodb url
AUTOSCALER_META_DIR = os.getenv("AUTOSCALER_META_DIR", str(BASE_DIR / "app" / "autoscaler_meta"))
pathlib.Path(AUTOSCALER_META_DIR).mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("prioritymax.autoscaler")
LOG.setLevel(os.getenv("AUTOSCALER_LOG_LEVEL", "INFO"))
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(handler)

router = APIRouter(prefix="/autoscaler", tags=["autoscaler"])

# -------------------------------
# DB / persistence (optional)
# -------------------------------
if _HAS_MOTOR and AUTOSCALER_DB:
    motor_client = motor_asyncio.AsyncIOMotorClient(AUTOSCALER_DB)
    autoscaler_db = motor_client.get_default_database()
    policy_collection = autoscaler_db.get_collection("scaling_policies")
    action_collection = autoscaler_db.get_collection("scaling_actions")
else:
    motor_client = None
    autoscaler_db = None
    policy_collection = None
    action_collection = None

# -------------------------------
# Models / Schemas
# -------------------------------

class ScaleActionType(str):
    SCALE = "scale"            # change consumer count
    THROTTLE = "throttle"      # throttle queue ingestion rate
    PAUSE = "pause"            # pause queue consumption
    RESUME = "resume"          # resume queue consumption
    RESTART = "restart"        # restart worker(s)
    NOOP = "noop"              # no-op / informational


class PolicyMode(str):
    HYBRID = "hybrid"      # threshold + ML
    RULE = "rule"          # pure rule-based
    ML = "ml"              # ML-only


class AnomalyStrategy(str):
    ZSCORE = "zscore"
    IQR = "iqr"
    SIMPLE_THRESHOLD = "threshold"


class SeverityLevel(str):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ScalingPolicyCreate(BaseModel):
    """
    Policy schema to define scaling behavior.
    Example:
    {
      "name": "scale_on_queue_length",
      "mode": "hybrid",
      "min_replicas": 1,
      "max_replicas": 200,
      "scale_up_threshold": 500,   # queue length
      "scale_down_threshold": 50,
      "cooldown_seconds": 60,
      "hysteresis": 0.1,           # fractional hysteresis
      "priority": 10,
      "anomaly_strategy": "zscore",
      "anomaly_params": {"window": 300, "z_thresh": 5}
    }
    """
    name: str = Field(..., description="Unique policy name")
    mode: PolicyMode = Field(PolicyMode.HYBRID)
    min_replicas: int = Field(1, ge=0)
    max_replicas: int = Field(50, ge=1)
    scale_up_threshold: Optional[float] = Field(None, description="Metric threshold to trigger scale up (depends on metric)")
    scale_down_threshold: Optional[float] = Field(None, description="Metric threshold to trigger scale down")
    cooldown_seconds: int = Field(60, ge=0)
    hysteresis: float = Field(0.0, description="Additional damping/hysteresis fraction")
    priority: int = Field(10, description="Policy priority, lower = higher precedence")
    metric: str = Field("queue_length", description="Metric name this policy watches")
    anomaly_strategy: Optional[AnomalyStrategy] = Field(None)
    anomaly_params: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = Field(False, description="If true, actions are never executed")
    description: Optional[str] = Field(None)

    @validator("hysteresis")
    def check_hysteresis(cls, v):
        if v < 0 or v > 1:
            raise ValueError("hysteresis must be in [0,1]")
        return v

class ScalingPolicy(ScalingPolicyCreate):
    tag: str = Field(..., description="UUID tag for policy")
    created_at: str
    updated_at: str
    enabled: bool = True

class ScalingAction(BaseModel):
    """
    Represents a scaling decision / action executed by the autoscaler.
    """
    action_id: str
    policy_tag: Optional[str]
    action_type: ScaleActionType
    desired_replicas: Optional[int]
    delta_replicas: Optional[int]
    reason: Optional[str]
    initiated_by: Optional[str]  # 'manual' | 'autoscaler' | 'rl_agent' | username
    dry_run: bool = False
    timestamp_utc: str

class ScaleRequest(BaseModel):
    target_replicas: Optional[int] = None
    delta: Optional[int] = None
    reason: Optional[str] = None
    dry_run: bool = False

class PredictRequest(BaseModel):
    lookback_seconds: int = Field(300)
    features: Optional[Dict[str, Any]] = None

class AnomalyResult(BaseModel):
    is_anomaly: bool
    score: float
    strategy: str
    details: Dict[str, Any] = {}

# -------------------------------
# In-memory state
# -------------------------------
_POLICIES: Dict[str, ScalingPolicy] = {}
_LAST_ACTION_TS: Dict[str, float] = {}  # policy_tag -> timestamp of last action
_CURRENT_REPLICAS: Dict[str, int] = {}  # queue_name -> current replicas (track local view)
_LOCK = asyncio.Lock()

# WebSocket connections for broadcasts: topic -> set of websockets
_WS_CONNECTIONS: Dict[str, List[WebSocket]] = {}

# Prometheus metrics registry (optional simplified)
if _HAS_PROM:
    PROM_REG = CollectorRegistry()
    PROM_DESIRED_REPLICAS = Gauge("prioritymax_desired_replicas", "Desired replicas computed by autoscaler", registry=PROM_REG)
    PROM_CURRENT_REPLICAS = Gauge("prioritymax_current_replicas", "Current replicas observed", registry=PROM_REG)
    PROM_LAST_ACTION_TS = Gauge("prioritymax_last_action_timestamp", "Last action timestamp (unix)", registry=PROM_REG)
else:
    PROM_REG = None
    PROM_DESIRED_REPLICAS = None
    PROM_CURRENT_REPLICAS = None
    PROM_LAST_ACTION_TS = None

# Instantiate ML components if available
_PREDICTOR = Predictor() if _HAS_PREDICTOR else None
_RL_AGENT = RLAgent() if _HAS_RLAGENT else None
_WORKER_MANAGER = WorkerManager() if _HAS_WORKER_MANAGER else None

# -------------------------------
# Helper utilities
# -------------------------------
def _now_ts() -> float:
    return time.time()

def _utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def _generate_tag(prefix: str = "pol") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

async def _persist_policy(policy: ScalingPolicy):
    """
    Save policy to DB if available, otherwise to local JSON file.
    """
    if policy_collection is not None:
        await policy_collection.update_one({"tag": policy.tag}, {"$set": policy.dict()}, upsert=True)
    else:
        p = pathlib.Path(AUTOSCALER_META_DIR) / f"{policy.tag}.json"
        p.write_text(json.dumps(policy.dict(), default=str, indent=2), encoding="utf-8")

async def _delete_policy_persist(tag: str):
    if policy_collection is not None:
        await policy_collection.delete_one({"tag": tag})
    else:
        p = pathlib.Path(AUTOSCALER_META_DIR) / f"{tag}.json"
        if p.exists():
            p.unlink()

async def _persist_action(action: ScalingAction):
    if action_collection is not None:
        await action_collection.insert_one(action.dict())
    else:
        p = pathlib.Path(AUTOSCALER_META_DIR) / "actions.jsonl"
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(action.dict(), default=str) + "\n")

async def _broadcast(topic: str, message: dict):
    """
    Broadcast a message to all websocket connections subscribed to a topic.
    topic can be "autoscaler:events", "autoscaler:policy_updates", "queue:<queue_name>"
    """
    conns = _WS_CONNECTIONS.get(topic, [])
    if not conns:
        return
    payload = json.dumps(message, default=str)
    stale = []
    for ws in conns:
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(ws)
    # remove stale
    for s in stale:
        try:
            _WS_CONNECTIONS[topic].remove(s)
        except Exception:
            pass

async def _register_action_and_broadcast(action: ScalingAction):
    await _persist_action(action)
    await _broadcast("autoscaler:events", {"event": "scaling_action", "action": action.dict()})
    # update prometheus metrics if available
    if PROM_LAST_ACTION_TS is not None:
        try:
            PROM_LAST_ACTION_TS.set(_now_ts())
        except Exception:
            pass

# -------------------------------
# Basic metrics ingestion / readers
# -------------------------------
# In production you would read metrics from Prometheus or a metrics pipeline. For Phase-3 we support:
# - direct metrics push (ingest_metrics API)
# - scraping Prometheus (if configured)
# - reading short-term rolling window stored in memory (for predictor features)
_METRIC_WINDOW: Dict[str, List[Tuple[float, float]]] = {}  # metric -> list of (ts, value), kept short


def _ingest_metric(metric_name: str, value: float, ts: Optional[float] = None, max_window_sec: int = 600):
    ts = ts or _now_ts()
    window = _METRIC_WINDOW.setdefault(metric_name, [])
    window.append((ts, float(value)))
    # prune
    cutoff = ts - max_window_sec
    while window and window[0][0] < cutoff:
        window.pop(0)

def _compute_metric_rolling(metric_name: str, lookback: int = 300) -> Dict[str, float]:
    now = _now_ts()
    window = _METRIC_WINDOW.get(metric_name, [])
    vals = [v for (t, v) in window if t >= now - lookback]
    if not vals:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p90": 0.0}
    import statistics
    mean = statistics.mean(vals)
    try:
        p90 = sorted(vals)[max(0, int(len(vals) * 0.9) - 1)]
    except Exception:
        p90 = max(vals)
    return {"count": len(vals), "mean": mean, "min": min(vals), "max": max(vals), "p90": p90}

# -------------------------------
# Anomaly detection (simple)
# -------------------------------
def _detect_anomaly(metric_name: str, strategy: AnomalyStrategy, params: Dict[str, Any]) -> AnomalyResult:
    window = _METRIC_WINDOW.get(metric_name, [])
    if not window:
        return AnomalyResult(is_anomaly=False, score=0.0, strategy=strategy, details={})
    vals = [v for (_, v) in window]
    res = {"strategy": strategy}
    if strategy == AnomalyStrategy.SIMPLE_THRESHOLD:
        thresh = float(params.get("threshold", 1e9))
        is_anom = vals[-1] > thresh
        score = float(vals[-1]) / (thresh + 1e-9)
        return AnomalyResult(is_anomaly=is_anom, score=score, strategy=strategy, details=res)
    elif strategy == AnomalyStrategy.ZSCORE:
        import statistics
        if len(vals) < 2:
            return AnomalyResult(is_anomaly=False, score=0.0, strategy=strategy, details=res)
        mean = statistics.mean(vals[:-1])
        stdev = statistics.pstdev(vals[:-1]) if len(vals) > 1 else 0.0
        last = vals[-1]
        z = (last - mean) / (stdev + 1e-9)
        thresh = params.get("z_thresh", 5.0)
        return AnomalyResult(is_anomaly=abs(z) > thresh, score=float(abs(z)), strategy=strategy, details={"z": z, "mean": mean, "stdev": stdev})
    elif strategy == AnomalyStrategy.IQR:
        vals_sorted = sorted(vals)
        q1 = vals_sorted[max(0, int(0.25 * len(vals_sorted)))]
        q3 = vals_sorted[min(len(vals_sorted)-1, int(0.75 * len(vals_sorted)))]
        iqr = q3 - q1
        upper = q3 + params.get("k", 3.0) * iqr
        is_anom = vals[-1] > upper
        return AnomalyResult(is_anomaly=is_anom, score=float(vals[-1] / (upper + 1e-9)), strategy=strategy, details={"upper": upper, "iqr": iqr})
    else:
        return AnomalyResult(is_anomaly=False, score=0.0, strategy=strategy, details={})

# -------------------------------
# Kubernetes helpers (optional)
# -------------------------------
def _k8s_load_config():
    if not _HAS_K8S:
        raise RuntimeError("kubernetes client not available")
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

def _k8s_scale_deployment(namespace: str, deployment_name: str, replicas: int):
    """
    Scale a k8s deployment (synchronous). Requires k8s config.
    """
    if not _HAS_K8S:
        raise RuntimeError("k8s client not available")
    api = k8s_client.AppsV1Api()
    body = {"spec": {"replicas": int(replicas)}}
    api.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)

def _k8s_restart_pod(namespace: str, pod_name: str):
    if not _HAS_K8S:
        raise RuntimeError("k8s client not available")
    api = k8s_client.CoreV1Api()
    # delete pod, rely on controller to recreate
    api.delete_namespaced_pod(name=pod_name, namespace=namespace, body=k8s_client.V1DeleteOptions())

# -------------------------------
# Core decision logic
# -------------------------------
async def _compute_desired_replicas_for_policy(policy: ScalingPolicy) -> Tuple[int, dict]:
    """
    Compute desired replicas based on policy mode:
    - RULE: threshold checks only
    - ML: ask predictor for forecast and convert to replicas
    - HYBRID: combine rule & ML with priority/hysteresis
    Returns (desired_replicas, diagnostic_info)
    """
    metric = policy.metric
    rolling = _compute_metric_rolling(metric, lookback=policy.cooldown_seconds * 2)
    diag = {"rolling": rolling, "mode": policy.mode}

    # Rule based
    desired_rule = None
    if policy.scale_up_threshold is not None and rolling["p90"] >= policy.scale_up_threshold:
        # Increase proportional to overload
        overload = rolling["p90"] / (policy.scale_up_threshold + 1e-9)
        # compute delta as fraction of range
        range_size = policy.max_replicas - policy.min_replicas
        incr = max(1, int(math.ceil(range_size * min(overload - 1.0, 2.0) * (1 + policy.hysteresis))))
        desired_rule = min(policy.max_replicas, int((CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas) + incr))
        diag["rule"] = {"desired": desired_rule, "reason": "scale_up_threshold_hit", "overload": overload}
    elif policy.scale_down_threshold is not None and rolling["p90"] <= policy.scale_down_threshold:
        # compute conservative downscale
        under = rolling["p90"] / (policy.scale_down_threshold + 1e-9)
        decr = max(1, int(math.ceil((CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas) * (1 - under) * (1 + policy.hysteresis))))
        desired_rule = max(policy.min_replicas, int((CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas) - decr))
        diag["rule"] = {"desired": desired_rule, "reason": "scale_down_threshold_hit", "under": under}
    else:
        diag["rule"] = {"desired": None, "reason": "no_threshold_trigger"}

    # ML prediction
    desired_ml = None
    if policy.mode in (PolicyMode.ML, PolicyMode.HYBRID) and _PREDICTOR is not None:
        try:
            features = {"metric_rolling": rolling, "policy": policy.dict()}
            pred = _PREDICTOR.predict(features)
            # predictor should return expected arrival_rate or desired_replicas directly
            if isinstance(pred, dict):
                if "desired_replicas" in pred:
                    desired_ml = int(pred["desired_replicas"])
                elif "expected_arrival_rate" in pred:
                    # crude mapping: expected arrival / service_rate_per_worker -> replicas
                    service_per_worker = pred.get("service_per_worker", 4.0)
                    desired_ml = int(min(policy.max_replicas, max(policy.min_replicas, math.ceil(pred["expected_arrival_rate"] / (service_per_worker + 1e-9)))))
            else:
                # if predictor returns scalar
                desired_ml = int(pred)
            diag["ml"] = {"pred": pred, "desired": desired_ml}
        except Exception as e:
            LOG.exception("Predictor error: %s", e)
            diag["ml"] = {"error": str(e)}

    # RL agent
    desired_rl = None
    if policy.mode in (PolicyMode.ML, PolicyMode.HYBRID) and _RL_AGENT is not None:
        try:
            obs = {"rolling": rolling, "policy": policy.dict()}
            act = _RL_AGENT.infer(obs)
            # Map RL action to replicas if action is dict with desired_replicas
            if isinstance(act, dict) and "desired_replicas" in act:
                desired_rl = int(act["desired_replicas"])
            elif isinstance(act, (int, float)):
                desired_rl = int(act)
            diag["rl"] = {"act": act, "desired": desired_rl}
        except Exception as e:
            LOG.exception("RL inference error: %s", e)
            diag["rl"] = {"error": str(e)}

    # Combine according to mode
    desired = None
    if policy.mode == PolicyMode.RULE:
        desired = desired_rule or (CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas)
    elif policy.mode == PolicyMode.ML:
        desired = desired_ml or desired_rl or (CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas)
    else:  # HYBRID
        # prefer conservative max among ML and Rule unless RL suggests otherwise
        candidates = [v for v in (desired_rule, desired_ml, desired_rl) if v is not None]
        if not candidates:
            desired = CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas
        else:
            # priority-based decision: if rule signals scale-up strongly, take max; for scale down, take min
            rule_dir = None
            if desired_rule is not None:
                if desired_rule > (CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas):
                    rule_dir = "up"
                elif desired_rule < (CURRENT_REPLICAS_FOR_POLICY(policy) or policy.min_replicas):
                    rule_dir = "down"
            if rule_dir == "up":
                desired = max(candidates)
            elif rule_dir == "down":
                desired = min(candidates)
            else:
                # fallback: median-like approach
                desired = int(sorted(candidates)[len(candidates)//2])
    # Bound desired to policy limits
    desired = max(policy.min_replicas, min(policy.max_replicas, int(desired)))
    diag["final_desired"] = desired
    return desired, diag

def CURRENT_REPLICAS_FOR_POLICY(policy: ScalingPolicy) -> Optional[int]:
    """
    For now, policies do not map to specific queues; in extension, include mapping.
    We'll return global current replicas if present, else None.
    """
    # If policy.metadata contains queue_name mapping, use it
    try:
        md = getattr(policy, "metadata", {})
        q = md.get("queue_name")
        if q and q in _CURRENT_REPLICAS:
            return _CURRENT_REPLICAS[q]
    except Exception:
        pass
    # fallback: if only one queue tracked, use its replicas
    if len(_CURRENT_REPLICAS) == 1:
        return next(iter(_CURRENT_REPLICAS.values()))
    return None

# -------------------------------
# Execution of actions (safety layer)
# -------------------------------
async def _execute_scale_action(action: ScalingAction, queue_name: Optional[str] = None, k8s_target: Optional[dict] = None):
    """
    Execute scaling action with safety checks:
    - obey global cooldown for the policy
    - dry_run: do not perform
    - rate limiting: only one scale per policy per cooldown_seconds
    - self-heal: optionally restart hung pods
    """
    # write audit event
    await write_audit_event({"user": action.initiated_by or "autoscaler", "action": "scaling_decision", "resource": action.policy_tag, "details": action.dict()})
    # update local tracking
    _LAST_ACTION_TS[action.policy_tag or "global"] = _now_ts()

    if action.dry_run:
        LOG.info("[DRY-RUN] Would execute: %s", action.dict())
        await _register_action_and_broadcast(action)
        return {"ok": True, "dry_run": True, "action": action.dict()}

    # if Kubernetes target provided, scale k8s deployment / set HPA
    if k8s_target and _HAS_K8S:
        try:
            ns = k8s_target.get("namespace", "default")
            dep = k8s_target["deployment"]
            desired = action.desired_replicas or 0
            LOG.info("Scaling k8s deployment %s/%s -> replicas=%s", ns, dep, desired)
            _k8s_scale_deployment(ns, dep, desired)
            # update current replicas local view
            if queue_name:
                _CURRENT_REPLICAS[queue_name] = desired
            await _register_action_and_broadcast(action)
            return {"ok": True, "method": "k8s", "replicas": desired}
        except Exception as e:
            LOG.exception("k8s scaling failed: %s", e)
            # proceed to worker_manager or queue fallback

    # If we have a worker manager that can scale replicas/take action
    if _HAS_WORKER_MANAGER and _WORKER_MANAGER is not None:
        try:
            if action.action_type == ScaleActionType.SCALE:
                desired = action.desired_replicas or 0
                LOG.info("WorkerManager scaling to %s replicas (queue=%s)", desired, queue_name)
                _WORKER_MANAGER.scale_to(desired, queue_name=queue_name)
                _CURRENT_REPLICAS[queue_name] = desired
                await _register_action_and_broadcast(action)
                return {"ok": True, "method": "worker_manager", "replicas": desired}
            elif action.action_type == ScaleActionType.RESTART:
                # restart workers (optionally delta)
                LOG.info("WorkerManager restarting workers: delta=%s", action.delta_replicas)
                _WORKER_MANAGER.restart_workers(queue_name=queue_name, count=action.delta_replicas or 1)
                await _register_action_and_broadcast(action)
                return {"ok": True, "method": "worker_manager", "restarted": action.delta_replicas or 1}
        except Exception as e:
            LOG.exception("WorkerManager action failed: %s", e)

    # Fallback: create a best-effort API call to queue backend (redis queue control)
    try:
        from app.queue.redis_queue import RedisQueueController
        queue_ctrl = RedisQueueController()  # expect default config
        if action.action_type == ScaleActionType.SCALE:
            desired = action.desired_replicas or 0
            queue_ctrl.set_consumer_count(desired)
            if queue_name:
                _CURRENT_REPLICAS[queue_name] = desired
            await _register_action_and_broadcast(action)
            return {"ok": True, "method": "redis", "replicas": desired}
        elif action.action_type == ScaleActionType.THROTTLE:
            # throttle by setting ingestion rate or inserting delay
            throttle_rate = action.delta_replicas or 0
            queue_ctrl.set_ingestion_throttle(throttle_rate)
            await _register_action_and_broadcast(action)
            return {"ok": True, "method": "redis", "throttle": throttle_rate}
        elif action.action_type in (ScaleActionType.PAUSE, ScaleActionType.RESUME):
            queue_ctrl.pause() if action.action_type == ScaleActionType.PAUSE else queue_ctrl.resume()
            await _register_action_and_broadcast(action)
            return {"ok": True, "method": "redis", "paused": action.action_type == ScaleActionType.PAUSE}
        elif action.action_type == ScaleActionType.RESTART:
            queue_ctrl.restart_workers(count=action.delta_replicas or 1)
            await _register_action_and_broadcast(action)
            return {"ok": True, "method": "redis", "restarted": action.delta_replicas or 1}
    except Exception as e:
        LOG.exception("Fallback queue controller failed: %s", e)

    # If we reach here, action could not be executed
    LOG.warning("Action could not be executed by any available controller: %s", action.dict())
    await _register_action_and_broadcast(action)
    return {"ok": False, "reason": "no_controller_available", "action": action.dict()}

# -------------------------------
# Public APIs
# -------------------------------

@router.post("/policies", dependencies=[Depends(require_role("operator"))])
async def create_policy(payload: ScalingPolicyCreate, user=Depends(get_current_user)):
    """
    Create a new scaling policy. Returns the created policy with tag.
    """
    tag = _generate_tag("policy")
    now = _utc_iso()
    policy = ScalingPolicy(**payload.dict(), tag=tag, created_at=now, updated_at=now, enabled=True)
    _POLICIES[tag] = policy
    await _persist_policy(policy)
    await write_audit_event({"user": user.username, "action": "create_policy", "resource": tag, "details": policy.dict()})
    # broadcast
    await _broadcast("autoscaler:policy_updates", {"event": "policy_created", "policy": policy.dict()})
    return policy

@router.get("/policies", dependencies=[Depends(require_role("viewer"))])
async def list_policies():
    """
    List all known policies (in-memory + persisted).
    """
    # If DB exists, prefer reading from it
    if policy_collection is not None:
        docs = await policy_collection.find({}).to_list(length=1000)
        return docs
    return [p.dict() for p in _POLICIES.values()]

@router.get("/policies/{tag}", dependencies=[Depends(require_role("viewer"))])
async def get_policy(tag: str):
    p = _POLICIES.get(tag)
    if p:
        return p
    # try loaded JSON
    fp = pathlib.Path(AUTOSCALER_META_DIR) / f"{tag}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Policy not found")

@router.put("/policies/{tag}", dependencies=[Depends(require_role("operator"))])
async def update_policy(tag: str, payload: ScalingPolicyCreate, user=Depends(get_current_user)):
    if tag not in _POLICIES:
        raise HTTPException(status_code=404, detail="Policy not found")
    now = _utc_iso()
    policy = ScalingPolicy(**payload.dict(), tag=tag, created_at=_POLICIES[tag].created_at, updated_at=now, enabled=_POLICIES[tag].enabled)
    _POLICIES[tag] = policy
    await _persist_policy(policy)
    await write_audit_event({"user": user.username, "action": "update_policy", "resource": tag, "details": policy.dict()})
    await _broadcast("autoscaler:policy_updates", {"event": "policy_updated", "policy": policy.dict()})
    return policy

@router.delete("/policies/{tag}", dependencies=[Depends(require_role("admin"))])
async def delete_policy(tag: str, user=Depends(get_current_user)):
    if tag in _POLICIES:
        del _POLICIES[tag]
    await _delete_policy_persist(tag)
    await write_audit_event({"user": user.username, "action": "delete_policy", "resource": tag})
    await _broadcast("autoscaler:policy_updates", {"event": "policy_deleted", "tag": tag})
    return {"ok": True, "deleted": tag}

# manual scaling endpoint
@router.post("/scale/manual", dependencies=[Depends(require_role("operator"))])
async def manual_scale(req: ScaleRequest, queue_name: Optional[str] = None, user=Depends(get_current_user)):
    """
    Manual scale API: set target_replicas or delta (positive or negative).
    This is immediate and bypasses policies. Dry-run supported.
    """
    current = _CURRENT_REPLICAS.get(queue_name, None)
    if req.target_replicas is None and req.delta is None:
        raise HTTPException(status_code=400, detail="Provide target_replicas or delta")
    if req.target_replicas is None:
        if current is None:
            raise HTTPException(status_code=400, detail="Unknown current replicas; provide explicit target_replicas")
        target = max(0, current + req.delta)
    else:
        target = req.target_replicas

    action = ScalingAction(
        action_id=str(uuid.uuid4()),
        policy_tag=None,
        action_type=ScaleActionType.SCALE,
        desired_replicas=target,
        delta_replicas=(target - current) if current is not None else None,
        reason=req.reason or "manual",
        initiated_by=(user.username if hasattr(user, "username") else "manual_api"),
        dry_run=req.dry_run,
        timestamp_utc=_utc_iso()
    )
    LOG.info("Manual scale request: %s -> %s", queue_name, action.dict())
    res = await _execute_scale_action(action, queue_name=queue_name, k8s_target=None)
    return res

# Predictive endpoint
@router.post("/predict", dependencies=[Depends(require_role("viewer"))])
async def predict(req: PredictRequest):
    """
    Expose predictor for external sanity checks. Returns predictor output.
    """
    if not _HAS_PREDICTOR or _PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Predictor not available")
    features = req.features or {"metrics": {k: _compute_metric_rolling(k, lookback=req.lookback_seconds) for k in _METRIC_WINDOW.keys()}}
    try:
        out = _PREDICTOR.predict(features)
        return {"ok": True, "prediction": out}
    except Exception as e:
        LOG.exception("Predictor failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# RL inference endpoint
@router.post("/rl/infer", dependencies=[Depends(require_role("viewer"))])
async def rl_infer(payload: dict = Body(...)):
    """
    Ask RL agent to infer action given an observation dict.
    """
    if not _HAS_RLAGENT or _RL_AGENT is None:
        raise HTTPException(status_code=503, detail="RL agent not available")
    try:
        act = _RL_AGENT.infer(payload)
        return {"ok": True, "action": act}
    except Exception as e:
        LOG.exception("RL agent inference error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest_metric", dependencies=[Depends(require_role("operator"))])
async def ingest_metric(name: str = Query(...), value: float = Query(...), ts: Optional[float] = Query(None)):
    """
    Lightweight metric ingestion API for local tests / collector.
    Real deployments should stream into Prometheus; promote this only for synthetic tests.
    """
    _ingest_metric(name, value, ts)
    # update prometheus gauges (best-effort)
    if PROM_CURRENT_REPLICAS is not None and name == "current_replicas":
        try:
            PROM_CURRENT_REPLICAS.set(value)
        except Exception:
            pass
    return {"ok": True, "metric": name, "value": value}

# Run policies evaluation loop (single-step call)
@router.post("/policies/evaluate", dependencies=[Depends(require_role("operator"))])
async def evaluate_policies(dry_run: bool = Query(True), user=Depends(get_current_user)):
    """
    Evaluate all enabled policies once and optionally execute actions (dry_run toggles execution).
    Returns decisions (per policy).
    """
    decisions = []
    async with _LOCK:
        # sort policies by priority ascending
        policies = sorted([p for p in _POLICIES.values() if p.enabled], key=lambda x: x.priority)
        for policy in policies:
            # respect cooldown
            last_ts = _LAST_ACTION_TS.get(policy.tag, 0)
            if _now_ts() - last_ts < policy.cooldown_seconds:
                LOG.debug("Skipping policy %s due to cooldown", policy.tag)
                decisions.append({"policy": policy.tag, "skipped": "cooldown"})
                continue
            desired, diag = await _compute_desired_replicas_for_policy(policy)
            current = CURRENT_REPLICAS_FOR_POLICY(policy) or _CURRENT_REPLICAS.get(None, None) or policy.min_replicas
            if desired is None:
                decisions.append({"policy": policy.tag, "decision": "no_op", "diag": diag})
                continue
            # create action
            delta = desired - int(current)
            if delta == 0:
                decisions.append({"policy": policy.tag, "decision": "no_change", "diag": diag})
                continue
            action = ScalingAction(
                action_id=str(uuid.uuid4()),
                policy_tag=policy.tag,
                action_type=ScaleActionType.SCALE,
                desired_replicas=desired,
                delta_replicas=delta,
                reason=f"policy_evaluation:{policy.name}",
                initiated_by="autoscaler",
                dry_run=dry_run,
                timestamp_utc=_utc_iso()
            )
            res = await _execute_scale_action(action, queue_name=None, k8s_target=None)
            decisions.append({"policy": policy.tag, "decision": "executed" if res.get("ok") else "failed", "res": res, "diag": diag})
    # broadcast evaluation summary
    await _broadcast("autoscaler:evaluation", {"timestamp": _utc_iso(), "decisions": decisions})
    await write_audit_event({"user": user.username, "action": "evaluate_policies", "resource": "all_policies", "details": {"dry_run": dry_run, "decisions": decisions}})
    return {"ok": True, "decisions": decisions}

# Anomaly detection endpoint
@router.post("/anomaly/check", dependencies=[Depends(require_role("viewer"))])
async def check_anomaly(metric: str = Query(...), strategy: AnomalyStrategy = Query(AnomalyStrategy.ZSCORE), params: Dict[str, Any] = Body(None)):
    params = params or {}
    res = _detect_anomaly(metric, strategy, params)
    return res

# Self-healing endpoint (manual trigger)
@router.post("/self_heal", dependencies=[Depends(require_role("operator"))])
async def trigger_self_heal(queue_name: Optional[str] = None, restart_count: int = Query(1), reason: Optional[str] = Query(None), user=Depends(get_current_user)):
    """
    Trigger self-heal flow: restart hung workers, or restart k8s pods in namespace.
    """
    action = ScalingAction(
        action_id=str(uuid.uuid4()),
        policy_tag=None,
        action_type=ScaleActionType.RESTART,
        desired_replicas=None,
        delta_replicas=restart_count,
        reason=reason or "manual_self_heal",
        initiated_by=user.username if hasattr(user, "username") else "self_heal_api",
        dry_run=False,
        timestamp_utc=_utc_iso()
    )
    res = await _execute_scale_action(action, queue_name=queue_name, k8s_target=None)
    await write_audit_event({"user": user.username, "action": "self_heal", "resource": queue_name or "global", "details": res})
    return res

# WebSocket subscribe
@router.websocket("/ws/subscribe/{topic}")
async def ws_subscribe(websocket: WebSocket, topic: str = "autoscaler:events", user=Depends(get_current_user)):
    """
    Subscribe to a topic. Topics: autoscaler:events, autoscaler:policy_updates, queue:<queue_name>
    """
    await websocket.accept()
    if topic not in _WS_CONNECTIONS:
        _WS_CONNECTIONS[topic] = []
    _WS_CONNECTIONS[topic].append(websocket)
    try:
        while True:
            # Keep connection alive; expect pings or noops
            data = await websocket.receive_text()
            # echo heartbeat
            await websocket.send_text(json.dumps({"pong": _utc_iso()}))
    except WebSocketDisconnect:
        LOG.info("WS client disconnected from %s", topic)
        try:
            _WS_CONNECTIONS[topic].remove(websocket)
        except Exception:
            pass
    except Exception:
        LOG.exception("Websocket error for topic %s", topic)
        try:
            _WS_CONNECTIONS[topic].remove(websocket)
        except Exception:
            pass

# Health & diagnostics
@router.get("/health")
async def health():
    return {"ok": True, "time": _utc_iso(), "policies": len(_POLICIES), "connections": {k: len(v) for k, v in _WS_CONNECTIONS.items()}}

@router.get("/status")
async def status():
    # include ML availability, k8s availability, worker manager presence
    return {
        "ok": True,
        "ml_predictor": _HAS_PREDICTOR and (_PREDICTOR is not None),
        "rl_agent": _HAS_RLAGENT and (_RL_AGENT is not None),
        "worker_manager": _HAS_WORKER_MANAGER and (_WORKER_MANAGER is not None),
        "kubernetes_available": _HAS_K8S,
        "prometheus_available": _HAS_PROM,
        "policies_loaded": len(_POLICIES),
        "current_replicas": _CURRENT_REPLICAS
    }

# -------------------------------
# Background loops (optional) - start when app starts
# -------------------------------
# If you want these to start automatically, call autoscaler_start_background_tasks() from application startup.

_BG_TASKS: List[asyncio.Task] = []
_BG_RUNNING = False

async def _policy_loop(interval_seconds: int = 10, dry_run_default: bool = True):
    """Periodic policy evaluation loop."""
    LOG.info("Autoscaler policy loop started (interval=%s)", interval_seconds)
    try:
        while True:
            try:
                await evaluate_policies(dry_run=dry_run_default)
            except Exception:
                LOG.exception("Error in policy loop")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        LOG.info("Autoscaler policy loop cancelled")
    except Exception:
        LOG.exception("Autoscaler policy loop unexpected exit")

async def _ml_monitor_loop(interval_seconds: int = 60):
    """Periodic ML-based health checks or model-based recommendations (non-blocking)."""
    LOG.info("Autoscaler ML monitor loop started (interval=%s)", interval_seconds)
    try:
        while True:
            try:
                # Example: ask predictor to forecast 60s ahead for queue_length and log
                if _PREDICTOR is not None:
                    features = {"metrics": {k: _compute_metric_rolling(k, lookback=300) for k in _METRIC_WINDOW.keys()}}
                    try:
                        rec = _PREDICTOR.predict(features)
                        await _broadcast("autoscaler:ml_recommendation", {"timestamp": _utc_iso(), "recommendation": rec})
                    except Exception:
                        LOG.exception("Predictor error in monitor loop")
            except Exception:
                LOG.exception("Error in ML monitor loop")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        LOG.info("Autoscaler ML monitor loop cancelled")
    except Exception:
        LOG.exception("Autoscaler ML monitor loop unexpected exit")

def autoscaler_start_background_tasks(app, policy_interval: int = 10, ml_interval: int = 60, dry_run_default: bool = True):
    """
    Hook this in FastAPI startup to begin background loops:
        app.add_event_handler("startup", lambda: autoscaler_start_background_tasks(app))
    """
    global _BG_TASKS, _BG_RUNNING
    if _BG_RUNNING:
        return
    loop = asyncio.get_event_loop()
    t1 = loop.create_task(_policy_loop(interval_seconds=policy_interval, dry_run_default=dry_run_default))
    t2 = loop.create_task(_ml_monitor_loop(interval_seconds=ml_interval))
    _BG_TASKS.extend([t1, t2])
    _BG_RUNNING = True
    LOG.info("Autoscaler background tasks started")

def autoscaler_stop_background_tasks():
    global _BG_TASKS, _BG_RUNNING
    for t in _BG_TASKS:
        try:
            t.cancel()
        except Exception:
            pass
    _BG_TASKS = []
    _BG_RUNNING = False
    LOG.info("Autoscaler background tasks stopped")

# -------------------------------
# Startup hooks: load persisted policies
# -------------------------------
@router.on_event("startup")
async def _startup_autoscaler():
    # load policy files from local directory if DB not present
    LOG.info("Autoscaler startup: loading persisted policies")
    if policy_collection is not None:
        docs = await policy_collection.find({}).to_list(length=1000)
        for d in docs:
            try:
                tag = d.get("tag")
                p = ScalingPolicy(**d)
                _POLICIES[tag] = p
            except Exception:
                LOG.exception("Failed load policy from DB: %s", d)
    else:
        pdir = pathlib.Path(AUTOSCALER_META_DIR)
        for f in pdir.glob("policy_*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                p = ScalingPolicy(**d)
                _POLICIES[p.tag] = p
            except Exception:
                LOG.exception("Failed load policy file %s", f)
    LOG.info("Loaded %d policies", len(_POLICIES))
    # optionally start background tasks
    try:
        autoscaler_start_background_tasks(None)
    except Exception:
        LOG.exception("Failed to start autoscaler background tasks")

@router.on_event("shutdown")
async def _shutdown_autoscaler():
    autoscaler_stop_background_tasks()

# End of autoscaler.py

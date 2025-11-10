# backend/app/autoscaler.py
"""
PriorityMax Intelligent Autoscaler (Enterprise Edition)
------------------------------------------------------

This version integrates:
 - Predictive ML scaling (LightGBM/RandomForest)
 - RL-based adaptive scaling (PPO policy)
 - Rule-based safeguards (CPU/memory/backlog thresholds)
 - Hybrid decision fusion with weighting and hysteresis
 - Kubernetes operator integration (HPA hooks)
 - Prometheus metrics + REST API exposure
 - Scaling history persistence + audit trail (Mongo or Redis)
 - Fault-tolerant background scheduling & health recovery
 - DLQ self-healing hooks (integrates with RedisQueue)
 - Analytics overlay: scale decision correlation, rolling statistics
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import uuid
import asyncio
import datetime
import logging
import statistics
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# ---------------------------
# Internal PriorityMax imports
# ---------------------------
try:
    from app.utils.time_utils import now_ts, iso_now, async_sleep, compute_backoff
    from app.utils.logger import get_logger, StructuredLoggerAdapter
    from app.metrics import get_metric_snapshot
    from app.ml.predictor import PREDICTOR_MANAGER
    from app.ml.rl_agent_prod import RLAgentProd
    from app.queue.redis_queue import RedisQueue
    from app.storage import Storage
except Exception as e:
    raise ImportError(f"autoscaler requires full PriorityMax backend context: {e}")

# ---------------------------
# External optional libs
# ---------------------------
try:
    from prometheus_client import Gauge, Counter
    _HAS_PROM = True
except Exception:
    Gauge = Counter = None
    _HAS_PROM = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except Exception:
    _HAS_AIOHTTP = False

# ---------------------------
# Logging setup
# ---------------------------
LOG = get_logger("prioritymax.autoscaler")
LAD = StructuredLoggerAdapter(LOG, {"component": "autoscaler"})

# ---------------------------
# Configurable parameters
# ---------------------------
AUTOSCALER_MODE = os.getenv("PRIORITYMAX_AUTOSCALER_MODE", "hybrid")  # hybrid | rule | ml | rl
AUTOSCALER_INTERVAL = float(os.getenv("PRIORITYMAX_AUTOSCALER_INTERVAL", "10"))
AUTOSCALER_COOLDOWN = float(os.getenv("PRIORITYMAX_AUTOSCALER_COOLDOWN", "30"))
AUTOSCALER_MAX_SCALE = int(os.getenv("PRIORITYMAX_AUTOSCALER_MAX", "100"))
AUTOSCALER_MIN_SCALE = int(os.getenv("PRIORITYMAX_AUTOSCALER_MIN", "1"))
AUTOSCALER_STEP = int(os.getenv("PRIORITYMAX_AUTOSCALER_STEP", "2"))
AUTOSCALER_HYSTERESIS = float(os.getenv("PRIORITYMAX_AUTOSCALER_HYSTERESIS", "0.15"))
AUTOSCALER_JITTER = float(os.getenv("PRIORITYMAX_AUTOSCALER_JITTER", "0.05"))
AUTOSCALER_PREDICTIVE_HORIZON = int(os.getenv("PRIORITYMAX_PREDICTIVE_HORIZON", "60"))
AUTOSCALER_CPU_UP = float(os.getenv("PRIORITYMAX_CPU_UP", "0.8"))
AUTOSCALER_CPU_DOWN = float(os.getenv("PRIORITYMAX_CPU_DOWN", "0.35"))
AUTOSCALER_LAT_UP = float(os.getenv("PRIORITYMAX_LAT_UP", "1.5"))
AUTOSCALER_LAT_DOWN = float(os.getenv("PRIORITYMAX_LAT_DOWN", "0.7"))
AUTOSCALER_BACKLOG_UP = int(os.getenv("PRIORITYMAX_BACKLOG_UP", "1000"))
AUTOSCALER_BACKLOG_DOWN = int(os.getenv("PRIORITYMAX_BACKLOG_DOWN", "50"))
AUTOSCALER_QUEUE = os.getenv("PRIORITYMAX_AUTOSCALER_QUEUE", "default")
RL_AGENT_PATH = os.getenv("PRIORITYMAX_RL_MODEL_PATH", "app/ml/models/rl_agent.pt")
ENABLE_K8S_INTEGRATION = os.getenv("PRIORITYMAX_K8S_AUTOSCALE", "false").lower() in ("1", "true", "yes")
K8S_NAMESPACE = os.getenv("PRIORITYMAX_K8S_NAMESPACE", "default")
K8S_DEPLOYMENT = os.getenv("PRIORITYMAX_K8S_DEPLOYMENT", "prioritymax-worker")
PROM_PORT = int(os.getenv("PRIORITYMAX_PROMETHEUS_PORT", "9001"))

# ---------------------------
# Prometheus metrics
# ---------------------------
if _HAS_PROM:
    SCALE_DECISIONS = Counter("prioritymax_autoscaler_decisions_total", "Autoscaling decisions made", ["action"])
    SCALE_SCORE = Gauge("prioritymax_autoscaler_score", "Composite autoscaler decision score")
    SCALE_HINT = Gauge("prioritymax_autoscaler_hint", "RL or ML model hint value")
    SCALE_WORKERS = Gauge("prioritymax_autoscaler_workers", "Current worker count")
    SCALE_COOLDOWN = Gauge("prioritymax_autoscaler_cooldown_sec", "Cooldown seconds")
else:
    SCALE_DECISIONS = SCALE_SCORE = SCALE_HINT = SCALE_WORKERS = SCALE_COOLDOWN = None

# ---------------------------
# State and Data Models
# ---------------------------
class ScalingHistory:
    """Tracks per-decision history for audit and analytics."""
    def __init__(self, maxlen: int = 500):
        self._entries: List[Dict[str, Any]] = []
        self._maxlen = maxlen

    def add(self, entry: Dict[str, Any]):
        entry["timestamp"] = iso_now()
        self._entries.append(entry)
        if len(self._entries) > self._maxlen:
            self._entries.pop(0)

    def summary(self):
        up = len([e for e in self._entries if e["action"] == "scale_up"])
        down = len([e for e in self._entries if e["action"] == "scale_down"])
        steady = len([e for e in self._entries if e["action"] == "steady"])
        return {"up": up, "down": down, "steady": steady, "total": len(self._entries)}

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._entries[-n:]

# ---------------------------------------------------------------------
# Autoscaler state machine
# ---------------------------------------------------------------------
class AutoscalerState:
    """Maintains live runtime state for the autoscaler."""
    def __init__(self):
        self.current_workers = AUTOSCALER_MIN_SCALE
        self.last_action = "steady"
        self.last_score = 0.0
        self.last_hint = 0.0
        self.last_scaled_at = 0.0
        self.cooldown_active = False
        self.history = ScalingHistory()
        self.recovery_mode = False
        self.health_status = "healthy"

    def update(self, action: str, score: float, hint: float):
        self.last_action = action
        self.last_score = score
        self.last_hint = hint
        self.last_scaled_at = now_ts()
        if _HAS_PROM:
            SCALE_WORKERS.set(self.current_workers)
            SCALE_SCORE.set(score)
            SCALE_HINT.set(hint)

    def is_in_cooldown(self):
        if (now_ts() - self.last_scaled_at) < AUTOSCALER_COOLDOWN:
            self.cooldown_active = True
            return True
        self.cooldown_active = False
        return False

STATE = AutoscalerState()

# ---------------------------------------------------------------------
# Exception wrapper for resilient async loops
# ---------------------------------------------------------------------
class ResilientLoop:
    """Utility to wrap autoscaler background loops with automatic retry/backoff."""
    def __init__(self, label: str, func: Callable, interval: float):
        self.label = label
        self.func = func
        self.interval = interval
        self._running = False
        self._task = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        LAD.info("ResilientLoop '%s' started", self.label)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            LAD.info("ResilientLoop '%s' stopped", self.label)

    async def _loop(self):
        attempt = 0
        while self._running:
            try:
                await self.func()
                attempt = 0
                await async_sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                attempt += 1
                delay = compute_backoff(attempt, base=2, factor=1.5, max_delay=30)
                LAD.warning("Loop '%s' error (%s); retrying in %.2fs", self.label, e, delay)
                await async_sleep(delay)
# ---------------------------
# Chunk 2 — PriorityMaxAutoscaler core (initialization, metrics aggregation,
# decision fusion, and action application)
# ---------------------------

class PriorityMaxAutoscaler:
    """
    Main autoscaler class implementing hybrid ML/RL/rule-based scaling.
    This chunk contains:
      - init / model loading
      - metric aggregation helpers
      - decision fusion (rule + predictor + RL)
      - safe apply (scale up/down) including Kubernetes hook
      - DLQ self-heal hints
    """
    def __init__(self,
                 queue_name: str = AUTOSCALER_QUEUE,
                 queue_client: Optional[RedisQueue] = None,
                 storage: Optional[Storage] = None,
                 rl_agent_path: Optional[str] = None):
        self.queue_name = queue_name
        self.queue = queue_client or RedisQueue()
        self.storage = storage or Storage()
        self.rl_agent_path = rl_agent_path or RL_AGENT_PATH
        self.predictor = PREDICTOR_MANAGER
        self.rl_agent: Optional[RLAgentProd] = None
        self.mode = AUTOSCALER_MODE
        self.loop: Optional[ResilientLoop] = None
        self._k8s_session: Optional[Any] = None
        self._scale_lock = asyncio.Lock()
        self._manual_override_until: float = 0.0

    # -------------------------
    # Initialization & loading
    # -------------------------
    async def initialize(self):
        LAD.info("Initializing PriorityMaxAutoscaler (mode=%s)", self.mode)
        # Try load predictor model if available
        try:
            self.predictor.load_latest()
            LAD.info("Predictor loaded: %s", getattr(self.predictor, "last_loaded_meta", None))
        except Exception:
            LAD.exception("Predictor load failed; ML hints disabled")

        # Try load RL agent (best-effort)
        try:
            self.rl_agent = RLAgentProd(checkpoint_path=self.rl_agent_path)
            LAD.info("RL agent loaded from %s", self.rl_agent_path)
        except Exception:
            LAD.exception("RL agent load failed; RL hints disabled")
            self.rl_agent = None

        # Initialize current workers from storage or default
        try:
            cur = await self.storage.get_latest_worker_count()
            if cur and isinstance(cur, int):
                STATE.current_workers = cur
                LAD.info("Restored worker count from storage: %d", cur)
        except Exception:
            LAD.debug("Could not restore worker count from storage; using default")

        # Start background loop
        self.loop = ResilientLoop(label="autoscaler-main", func=self._loop_once, interval=AUTOSCALER_INTERVAL)
        await self.loop.start()

    # -------------------------
    # Shutdown
    # -------------------------
    async def shutdown(self):
        LAD.info("Shutting down autoscaler")
        if self.loop:
            await self.loop.stop()

    # -------------------------
    # Metric aggregation
    # -------------------------
    async def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics from queue and system:
         - queue backlog, pending, inflight
         - average latency, throughput (from metrics module)
         - CPU/memory utilization (if available)
         - DLQ sizes
        """
        out: Dict[str, Any] = {"queue": {}, "system": {}, "timestamp": now_ts()}
        try:
            q = await self.queue.get_queue_stats(self.queue_name)
            out["queue"].update(q)
        except Exception:
            LAD.exception("Failed to fetch queue stats")
            out["queue"].update({"backlog": 0, "pending": 0, "inflight": 0})

        try:
            m = get_metric_snapshot()
            out["system"].update(m)
        except Exception:
            LAD.debug("metrics snapshot not available")
            out["system"].update({"avg_latency": 0.0, "throughput": 0.0, "cpu_util": 0.0, "mem_util": 0.0})

        try:
            dlq_len = await self.queue.get_dlq_length(self.queue_name)
            out["queue"]["dlq_backlog"] = dlq_len
        except Exception:
            out["queue"]["dlq_backlog"] = 0

        return out

    # -------------------------
    # Decision fusion
    # -------------------------
    async def _compute_rule_score(self, metrics: Dict[str, Any]) -> float:
        """Simple rule-based scoring (-2 .. +2)."""
        score = 0.0
        q = metrics.get("queue", {})
        sysm = metrics.get("system", {})
        backlog = int(q.get("backlog", 0))
        latency = float(sysm.get("avg_latency", 0.0))
        cpu = float(sysm.get("cpu_util", 0.0))

        if backlog >= AUTOSCALER_BACKLOG_UP:
            score += 1.2
        elif backlog <= AUTOSCALER_BACKLOG_DOWN:
            score -= 1.0

        if latency >= AUTOSCALER_LAT_UP:
            score += 1.0
        elif latency <= AUTOSCALER_LAT_DOWN:
            score -= 0.8

        if cpu >= AUTOSCALER_CPU_UP:
            score += 0.6
        elif cpu <= AUTOSCALER_CPU_DOWN:
            score -= 0.4

        # Normalize and clamp
        score = max(min(score, 2.0), -2.0)
        LAD.debug("Rule score: %.3f (backlog=%d lat=%.3f cpu=%.3f)", score, backlog, latency, cpu)
        return score

    async def _compute_ml_hint(self, metrics: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Use predictor to forecast next N seconds backlog/latency and return a normalized hint.
        The predictor returns arbitrary scale; we map it to [-1, +1].
        """
        hint = 0.0
        debug = {}
        try:
            feat = {
                "queue_len": int(metrics["queue"].get("backlog", 0)),
                "avg_latency": float(metrics["system"].get("avg_latency", 0.0)),
                "cpu": float(metrics["system"].get("cpu_util", 0.0)),
                "throughput": float(metrics["system"].get("throughput", 0.0)),
                "dlq": int(metrics["queue"].get("dlq_backlog", 0))
            }
            preds = self.predictor.predict(feat)
            # prediction semantics: higher prediction → more work incoming
            raw = float(preds.get("prediction", 0.0))
            # Map raw to [-1,1] via tanh-like scaling
            hint = math.tanh(raw / max(1.0, feat["queue_len"] + 1.0) * 3.0)
            debug = {"pred_raw": raw}
            LAD.debug("ML hint: raw=%.4f mapped=%.4f", raw, hint)
        except Exception:
            LAD.exception("Predictor inference error")
        return hint, debug

    async def _compute_rl_hint(self, metrics: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Query RL agent for suggested action magnitude.
        Expect RL agent to return continuous score ~[-1,1] where +1 strongly suggests scale_up.
        """
        hint = 0.0
        debug = {}
        if not self.rl_agent:
            return hint, debug
        try:
            obs = [
                float(metrics["queue"].get("backlog", 0)) / max(1.0, AUTOSCALER_BACKLOG_UP),
                float(metrics["system"].get("avg_latency", 0.0)) / max(0.1, AUTOSCALER_LAT_UP),
                float(metrics["system"].get("cpu_util", 0.0))
            ]
            # RLAgentProd.infer may be async
            if asyncio.iscoroutinefunction(self.rl_agent.infer):
                raw = await self.rl_agent.infer(obs)
            else:
                raw = self.rl_agent.infer(obs)
            hint = float(raw)
            debug = {"rl_raw": raw}
            LAD.debug("RL hint: %.4f", hint)
        except Exception:
            LAD.exception("RL inference failed")
        return hint, debug

    async def _fuse_scores(self,
                           rule_score: float,
                           ml_hint: float,
                           rl_hint: float,
                           metrics: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Combine rule_score, ml_hint, rl_hint into composite score.
        Weighting strategy:
          - rule: base stability (0.5)
          - ml: predictive (0.3)
          - rl: adapt & fine-tune (0.2)
        We apply dynamic weighting if RL or ML not available.
        """
        w_rule = 0.5
        w_ml = 0.3 if self.predictor and getattr(self.predictor, "last_loaded_meta", None) else 0.0
        w_rl = 0.2 if self.rl_agent else 0.0
        # Normalize weights
        total = w_rule + w_ml + w_rl
        if total <= 0:
            # fallback to rule-only
            total = 1.0
            w_rule = 1.0
            w_ml = 0.0
            w_rl = 0.0
        w_rule /= total
        w_ml /= total
        w_rl /= total

        composite = w_rule * rule_score + w_ml * ml_hint + w_rl * rl_hint
        # apply hysteresis nudges if within small band: require stronger signal to change
        if abs(composite - STATE.last_score) < AUTOSCALER_HYSTERESIS:
            composite = STATE.last_score  # dampen oscillation
        composite = max(min(composite, 2.0), -2.0)
        debug = {"w_rule": w_rule, "w_ml": w_ml, "w_rl": w_rl, "composite_pre_clip": composite}
        LAD.debug("Fused composite score: %.4f (rule=%.3f ml=%.3f rl=%.3f)", composite, rule_score, ml_hint, rl_hint)
        return composite, debug

    # -------------------------
    # Action selection & safe apply
    # -------------------------
    def _interpret_score(self, score: float) -> str:
        """Map numeric score to discrete action"""
        if score > AUTOSCALER_HYSTERESIS:
            return "scale_up"
        if score < -AUTOSCALER_HYSTERESIS:
            return "scale_down"
        return "steady"

    async def _scale_k8s(self, new_replicas: int) -> bool:
        """
        Apply scaling by patching a Kubernetes Deployment replicas field.
        Uses aiohttp to call the Kubernetes API (in-cluster serviceaccount) or kubectl proxy.
        This is implemented best-effort; failures are logged but do not crash the autoscaler.
        """
        if not ENABLE_K8S_INTEGRATION:
            LAD.debug("K8s integration disabled; skipping k8s scale")
            return False
        # Try in-cluster: use serviceaccount token & API server env
        try:
            api_host = os.getenv("KUBERNETES_SERVICE_HOST")
            api_port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
            token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
            if not api_host or not os.path.exists(token_path):
                LAD.warning("K8s env not detected or token missing; skipping k8s scale")
                return False
            with open(token_path, "r") as fh:
                token = fh.read().strip()
            url = f"https://{api_host}:{api_port}/apis/apps/v1/namespaces/{K8S_NAMESPACE}/deployments/{K8S_DEPLOYMENT}"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/strategic-merge-patch+json"}
            payload = {"spec": {"replicas": int(new_replicas)}}
            # Use aiohttp session
            async with aiohttp.ClientSession() as sess:
                # disable SSL verify if CA missing
                ssl_ctx = None
                if os.path.exists(ca_path):
                    import ssl
                    ssl_ctx = ssl.create_default_context(cafile=ca_path)
                async with sess.patch(url, json=payload, headers=headers, ssl=ssl_ctx) as resp:
                    if resp.status in (200, 201):
                        LAD.info("K8s deployment scaled to %d replicas", new_replicas)
                        return True
                    else:
                        text = await resp.text()
                        LAD.warning("K8s scale failed status=%s body=%s", resp.status, text)
                        return False
        except Exception:
            LAD.exception("K8s scaling exception")
            return False

    async def _apply_scale(self, action: str, composite_score: float, debug: Dict[str, Any]):
        """
        Safely commit scale up/down with locks, cooldown check, persistence and optional k8s patch.
        """
        async with self._scale_lock:
            # manual override check
            if now_ts() < self._manual_override_until:
                LAD.info("Manual override active until %.3f; skipping autoscale", self._manual_override_until)
                return

            if STATE.is_in_cooldown():
                LAD.debug("Autoscaler in cooldown; skipping apply")
                return

            current = STATE.current_workers
            if action == "scale_up":
                target = min(AUTOSCALER_MAX_SCALE, current + AUTOSCALER_STEP)
            elif action == "scale_down":
                target = max(AUTOSCALER_MIN_SCALE, current - AUTOSCALER_STEP)
            else:
                target = current

            if target == current:
                LAD.debug("No-op scaling target == current (%d)", current)
                STATE.update("steady", composite_score, debug.get("hint", 0.0))
                return

            # Persist attempt to storage (optimistic)
            try:
                await self.storage.insert_scaling_action({
                    "from": current,
                    "to": target,
                    "action": action,
                    "score": composite_score,
                    "debug": debug,
                    "ts": iso_now()
                })
            except Exception:
                LAD.exception("Failed to persist scaling action")

            # Do the actual scale: first try k8s, else use worker manager hook (storage/queue-managed)
            k8s_ok = await self._scale_k8s(target)
            if not k8s_ok:
                # fallback: write desired worker count into storage so worker manager picks up
                try:
                    await self.storage.set_desired_worker_count(target)
                except Exception:
                    LAD.exception("Failed to write desired worker count to storage fallback")

            STATE.current_workers = target
            STATE.update(action, composite_score, debug.get("hint", 0.0))
            STATE.history.add({
                "action": action,
                "from": current,
                "to": target,
                "score": composite_score,
                "debug": debug
            })
            if SCALE_DECISIONS:
                SCALE_DECISIONS.labels(action=action).inc()
            LAD.info("Applied scaling action=%s current=%d target=%d score=%.4f", action, current, target, composite_score)

    # -------------------------
    # DLQ self-healing helpers
    # -------------------------
    async def _dlq_heal_suggestion(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        If DLQ backlog grows but overall system load is low, recommend promoting DLQ items.
        Returns a suggestion dict if conditions met.
        """
        dlq = int(metrics["queue"].get("dlq_backlog", 0))
        backlog = int(metrics["queue"].get("backlog", 0))
        cpu = float(metrics["system"].get("cpu_util", 0.0))
        if dlq > 0 and backlog < AUTOSCALER_BACKLOG_DOWN and cpu < AUTOSCALER_CPU_DOWN:
            return {"suggest": "promote_dlq", "dlq": dlq}
        return None

    async def _promote_dlq(self, limit: int = 100):
        """
        Promote DLQ messages back to main queue. This will call RedisQueue.promote_dlq.
        """
        try:
            promoted = await self.queue.promote_dlq(self.queue_name, limit=limit)
            LAD.info("Promoted %d messages from DLQ for queue %s", promoted, self.queue_name)
            await self.storage.insert_dlq_promotion({"queue": self.queue_name, "promoted": promoted, "ts": iso_now()})
            return promoted
        except Exception:
            LAD.exception("DLQ promote failed")
            return 0

    # -------------------------
    # One-shot loop iteration (called by ResilientLoop)
    # -------------------------
    async def _loop_once(self):
        """
        Single autoscaler tick:
         - collect metrics
         - compute rule/ml/rl hints
         - fuse into composite score
         - decide action and apply safely
         - handle DLQ healing suggestions
         - save history and emit metrics
        """
        metrics = await self._collect_metrics()
        rule_score = await self._compute_rule_score(metrics)
        ml_hint, ml_debug = await self._compute_ml_hint(metrics)
        rl_hint, rl_debug = await self._compute_rl_hint(metrics)
        composite, fuse_debug = await self._fuse_scores(rule_score, ml_hint, rl_hint, metrics)
        action = self._interpret_score(composite)

        debug = {"rule": rule_score, "ml": ml_debug, "rl": rl_debug, "fuse": fuse_debug}
        # store last hint for observability
        STATE.last_hint = (ml_hint or rl_hint)
        # Apply action
        await self._apply_scale(action, composite, debug)

        # DLQ healing suggestion
        suggestion = await self._dlq_heal_suggestion(metrics)
        if suggestion:
            LAD.info("DLQ heal suggestion: %s", suggestion)
            # rate-limit auto-promote: only promote small batches automatically
            promoted = await self._promote_dlq(limit= min(100, suggestion["dlq"]))
            LAD.info("DLQ auto-promote executed: promoted=%d", promoted)
# ---------------------------
# Chunk 3 — Health Monitor, Watchdog Loop & Recovery Logic
# ---------------------------

class AutoscalerHealthMonitor:
    """
    Background subsystem that monitors autoscaler health and takes
    corrective actions such as:
      - Detecting stalled scaling loops
      - Detecting runaway scaling oscillations
      - Detecting persistent DLQ growth or starvation
      - Triggering recovery mode and rollback to safe worker counts
      - Logging all health transitions
    """
    def __init__(self, autoscaler: PriorityMaxAutoscaler):
        self.autoscaler = autoscaler
        self._running = False
        self._task = None
        self._lock = asyncio.Lock()
        self._oscillation_window: List[str] = []
        self._last_healthy_ts = now_ts()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        LAD.info("AutoscalerHealthMonitor started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        LAD.info("AutoscalerHealthMonitor stopped")

    # -------------------------
    # Health checks
    # -------------------------
    async def _loop(self):
        """Monitor autoscaler health every 15 seconds."""
        while self._running:
            try:
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                LAD.exception("HealthMonitor loop error: %s", e)
            await async_sleep(15)

    async def _check_health(self):
        """Perform one health cycle."""
        async with self._lock:
            hist = STATE.history.recent(10)
            if not hist:
                return

            recent_actions = [h["action"] for h in hist[-5:]]
            up_count = recent_actions.count("scale_up")
            down_count = recent_actions.count("scale_down")

            # Detect oscillation pattern (frequent up-down flips)
            if up_count > 0 and down_count > 0:
                if recent_actions[-1] != STATE.last_action:
                    self._oscillation_window.append(STATE.last_action)
                    if len(self._oscillation_window) > 6:
                        self._oscillation_window.pop(0)
                    if len(set(self._oscillation_window[-4:])) > 2:
                        LAD.warning("Detected oscillation pattern in scaling")
                        await self._trigger_recovery(reason="oscillation")

            # Detect if autoscaler has been inactive for too long
            idle_time = now_ts() - STATE.last_scaled_at
            if idle_time > (AUTOSCALER_INTERVAL * 10):
                LAD.warning("Autoscaler idle for %.1fs; forcing refresh", idle_time)
                await self._trigger_recovery(reason="stalled_loop")

            # Detect if DLQ consistently growing without heal
            try:
                qstats = await self.autoscaler.queue.get_queue_stats(self.autoscaler.queue_name)
                dlq = int(qstats.get("dlq_backlog", 0))
                backlog = int(qstats.get("backlog", 0))
                if dlq > 500 and backlog < 50:
                    LAD.warning("DLQ growth detected (dlq=%d backlog=%d)", dlq, backlog)
                    await self._trigger_recovery(reason="dlq_overflow", extra={"dlq": dlq})
            except Exception:
                LAD.debug("Health DLQ check failed")

            # If reached here, mark healthy
            STATE.health_status = "healthy"
            self._last_healthy_ts = now_ts()

    # -------------------------
    # Recovery mode triggers
    # -------------------------
    async def _trigger_recovery(self, reason: str, extra: Optional[Dict[str, Any]] = None):
        """
        Transition autoscaler into recovery mode.
        This disables RL-driven scaling temporarily and resets worker count to median stable value.
        """
        LAD.warning("Autoscaler recovery mode triggered: reason=%s", reason)
        STATE.recovery_mode = True
        STATE.health_status = f"recovering ({reason})"
        try:
            hist = STATE.history.recent(20)
            stable_counts = [h["to"] for h in hist if h.get("action") != "steady"]
            if stable_counts:
                median_workers = int(statistics.median(stable_counts))
            else:
                median_workers = max(AUTOSCALER_MIN_SCALE, STATE.current_workers)
            LAD.info("Recovery fallback worker count = %d", median_workers)
            await self.autoscaler._apply_scale("steady", 0.0, {"reason": reason})
            STATE.current_workers = median_workers
            await self.autoscaler.storage.insert_recovery_event({
                "reason": reason,
                "median_workers": median_workers,
                "extra": extra or {},
                "ts": iso_now()
            })
            if SCALE_DECISIONS:
                SCALE_DECISIONS.labels(action="recovery").inc()
        except Exception:
            LAD.exception("Recovery procedure failed")

    async def _reset_recovery(self):
        """Reset recovery mode once stable for 5 intervals."""
        if STATE.recovery_mode:
            if now_ts() - self._last_healthy_ts > AUTOSCALER_INTERVAL * 5:
                STATE.recovery_mode = False
                STATE.health_status = "healthy"
                LAD.info("Autoscaler exited recovery mode")

# ---------------------------
# Integration with PriorityMaxAutoscaler
# ---------------------------
class AutoscalerOrchestrator:
    """
    Supervises both the main autoscaler loop and the health monitor,
    providing a unified control interface for FastAPI or CLI integration.
    """
    def __init__(self):
        self.autoscaler = PriorityMaxAutoscaler()
        self.health_monitor = AutoscalerHealthMonitor(self.autoscaler)
        self._started = False

    async def start(self):
        if self._started:
            return
        await self.autoscaler.initialize()
        await self.health_monitor.start()
        self._started = True
        LAD.info("AutoscalerOrchestrator started successfully")

    async def stop(self):
        if not self._started:
            return
        await self.health_monitor.stop()
        await self.autoscaler.shutdown()
        self._started = False
        LAD.info("AutoscalerOrchestrator stopped")

    async def restart(self):
        LAD.info("Restarting autoscaler orchestrator")
        await self.stop()
        await async_sleep(3)
        await self.start()

# ---------------------------
# Global singleton orchestrator
# ---------------------------
ORCHESTRATOR = AutoscalerOrchestrator()
# ---------------------------
# Chunk 4 — FastAPI Integration: Startup, Shutdown & REST Routes
# ---------------------------
try:
    from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from app.auth import require_admin
except Exception:
    # graceful fallback for non-API contexts
    APIRouter = None
    Depends = None
    HTTPException = Exception
    BackgroundTasks = None
    require_admin = lambda: None

if APIRouter:
    router = APIRouter(prefix="/autoscaler", tags=["autoscaler"])

    # -------------------------
    # API: status and live state
    # -------------------------
    @router.get("/status", summary="Get autoscaler live state")
    async def get_status():
        """Return current autoscaler status snapshot."""
        return JSONResponse({
            "current_workers": STATE.current_workers,
            "last_action": STATE.last_action,
            "last_score": round(STATE.last_score, 4),
            "last_hint": round(STATE.last_hint, 4),
            "health_status": STATE.health_status,
            "cooldown": STATE.is_in_cooldown(),
            "recovery_mode": STATE.recovery_mode,
            "summary": STATE.history.summary(),
            "recent": STATE.history.recent(5)
        })

    # -------------------------
    # API: manual trigger
    # -------------------------
    @router.post("/trigger", dependencies=[Depends(require_admin)],
                 summary="Manually trigger one autoscaler cycle")
    async def manual_trigger(background_tasks: BackgroundTasks):
        """Manually execute one autoscaling tick (for admin testing)."""
        if not ORCHESTRATOR._started:
            raise HTTPException(status_code=503, detail="Autoscaler not running")

        async def _trigger():
            try:
                await ORCHESTRATOR.autoscaler._loop_once()
            except Exception as e:
                LAD.exception("Manual trigger error: %s", e)
        background_tasks.add_task(_trigger)
        return {"message": "Autoscaler tick triggered in background"}

    # -------------------------
    # API: restart orchestrator
    # -------------------------
    @router.post("/restart", dependencies=[Depends(require_admin)],
                 summary="Restart autoscaler orchestrator")
    async def restart_autoscaler():
        try:
            await ORCHESTRATOR.restart()
            return {"message": "Autoscaler restarted"}
        except Exception as e:
            LAD.exception("Restart failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------
    # API: analytics snapshot
    # -------------------------
    @router.get("/analytics", summary="Autoscaler analytics summary")
    async def analytics_snapshot():
        """
        Return last 50 scale decisions and basic statistics
        for dashboard visualization.
        """
        hist = STATE.history.recent(50)
        up = sum(1 for h in hist if h["action"] == "scale_up")
        down = sum(1 for h in hist if h["action"] == "scale_down")
        steady = sum(1 for h in hist if h["action"] == "steady")
        avg_score = statistics.mean([h["score"] for h in hist]) if hist else 0.0
        return JSONResponse({
            "count": len(hist),
            "up": up,
            "down": down,
            "steady": steady,
            "avg_score": round(avg_score, 4),
            "recent": hist[-10:]
        })

    # -------------------------
    # API: scaling history
    # -------------------------
    @router.get("/history", summary="Get scaling decision history")
    async def get_history(limit: int = 20):
        return JSONResponse({"history": STATE.history.recent(limit)})

    # -------------------------
    # API: DLQ recovery manual trigger
    # -------------------------
    @router.post("/heal", dependencies=[Depends(require_admin)],
                 summary="Promote DLQ messages manually")
    async def promote_dlq(limit: int = 100):
        promoted = await ORCHESTRATOR.autoscaler._promote_dlq(limit=limit)
        return {"message": f"Promoted {promoted} DLQ messages"}

# ---------------------------
# FastAPI startup / shutdown hooks
# ---------------------------
async def start_autoscaler_service():
    """Hook for FastAPI startup."""
    LAD.info("Starting PriorityMax autoscaler orchestrator via FastAPI hook")
    try:
        await ORCHESTRATOR.start()
        LAD.info("Autoscaler orchestrator started")
    except Exception as e:
        LAD.exception("Autoscaler startup failed: %s", e)

async def stop_autoscaler_service():
    """Hook for FastAPI shutdown."""
    LAD.info("Stopping PriorityMax autoscaler orchestrator via FastAPI hook")
    try:
        await ORCHESTRATOR.stop()
        LAD.info("Autoscaler orchestrator stopped cleanly")
    except Exception as e:
        LAD.exception("Autoscaler shutdown failed: %s", e)

# ---------------------------
# Integration Helper for main.py
# ---------------------------
def register_autoscaler(app):
    """
    Register autoscaler routes and lifecycle events
    with the provided FastAPI app.
    """
    if not APIRouter:
        LAD.warning("FastAPI not available; autoscaler API not registered")
        return
    app.include_router(router)
    app.add_event_handler("startup", start_autoscaler_service)
    app.add_event_handler("shutdown", stop_autoscaler_service)
    LAD.info("Autoscaler API routes registered on FastAPI app")
# ---------------------------
# Chunk 5 — Kubernetes Operator / HPA Integration & Policy Engine
# ---------------------------

# Try the kubernetes client first (preferred)
try:
    from kubernetes import client as k8s_client, config as k8s_config, watch as k8s_watch
    _HAS_K8S_PY = True
except Exception:
    k8s_client = None
    k8s_config = None
    k8s_watch = None
    _HAS_K8S_PY = False

# aiohttp already imported earlier optionally as _HAS_AIOHTTP
import json as _json

class PolicyEngine:
    """
    Lightweight policy engine that converts high-level policies into scaling constraints.
    Policies may be specified as dicts, for example:
      {
        "min_workers": 2,
        "max_workers": 50,
        "scale_up_step": 2,
        "scale_down_step": 1,
        "cooldown": 30,
        "stabilization_window_sec": 120,
        "target_cpu_percent": 70
      }
    The engine evaluates policies with runtime metrics and returns resolved parameters.
    """
    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy or {}
        self.defaults = {
            "min_workers": AUTOSCALER_MIN_SCALE,
            "max_workers": AUTOSCALER_MAX_SCALE,
            "scale_up_step": AUTOSCALER_STEP,
            "scale_down_step": max(1, AUTOSCALER_STEP // 2),
            "cooldown": AUTOSCALER_COOLDOWN,
            "stabilization_window_sec": 120,
            "target_cpu_percent": 70,
        }

    def resolve(self, runtime_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge policy with defaults, and apply runtime overrides (e.g., burst mode).
        """
        out = dict(self.defaults)
        out.update(self.policy or {})
        # runtime override example: if CPU bursts > 90% temporarily allow larger step
        if runtime_metrics:
            cpu = float(runtime_metrics.get("system", {}).get("cpu_util", 0.0))
            if cpu > 0.9:
                out["scale_up_step"] = min(out["max_workers"], int(out["scale_up_step"] * 2))
        return out

class K8sOperator:
    """
    Operator that applies scaling decisions to Kubernetes.
    Uses official k8s client if available; otherwise uses in-cluster HTTP to API server.
    This is intentionally conservative (patch-style strategic merge).
    """
    def __init__(self, namespace: str = K8S_NAMESPACE, deployment: str = K8S_DEPLOYMENT):
        self.namespace = namespace
        self.deployment = deployment
        self._initialized = False
        self._api = None
        self._apps_v1 = None

    def initialize(self):
        """Load k8s config; prefer in-cluster then kubeconfig fallback."""
        if _HAS_K8S_PY:
            try:
                # try in-cluster first
                try:
                    k8s_config.load_incluster_config()
                    LAD.info("Loaded in-cluster Kubernetes config")
                except Exception:
                    k8s_config.load_kube_config()
                    LAD.info("Loaded kubeconfig for Kubernetes client")
                self._apps_v1 = k8s_client.AppsV1Api()
                self._initialized = True
                LAD.info("K8sOperator initialized (kubernetes python client)")
            except Exception:
                LAD.exception("Failed to initialize k8s client")
                self._initialized = False
        else:
            LAD.warning("kubernetes python client not available; K8sOperator will use HTTP fallback if possible")
            self._initialized = False

    async def patch_deployment_replicas(self, replicas: int) -> bool:
        """Patch deployment replicas via k8s client or HTTP fallback."""
        LAD.debug("Attempting to patch deployment %s/%s -> %d replicas", self.namespace, self.deployment, replicas)
        if _HAS_K8S_PY and self._apps_v1:
            try:
                body = {"spec": {"replicas": int(replicas)}}
                self._apps_v1.patch_namespaced_deployment(self.deployment, self.namespace, body)
                LAD.info("Patched deployment (k8s client) to %d replicas", replicas)
                return True
            except Exception:
                LAD.exception("k8s client patch failed")
                # fallthrough to HTTP attempt
        # HTTP fallback (in-cluster)
        if _HAS_AIOHTTP:
            try:
                api_host = os.getenv("KUBERNETES_SERVICE_HOST")
                api_port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
                token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
                ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
                if not api_host or not os.path.exists(token_path):
                    LAD.warning("K8s HTTP fallback not possible (no in-cluster info)")
                    return False
                with open(token_path, "r") as fh:
                    token = fh.read().strip()
                url = f"https://{api_host}:{api_port}/apis/apps/v1/namespaces/{self.namespace}/deployments/{self.deployment}"
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/strategic-merge-patch+json"}
                payload = {"spec": {"replicas": int(replicas)}}
                import ssl
                ssl_ctx = None
                if os.path.exists(ca_path):
                    ssl_ctx = ssl.create_default_context(cafile=ca_path)
                async with aiohttp.ClientSession() as sess:
                    async with sess.patch(url, headers=headers, json=payload, ssl=ssl_ctx) as resp:
                        txt = await resp.text()
                        if resp.status in (200, 201):
                            LAD.info("Patched deployment via HTTP fallback to %d replicas", replicas)
                            return True
                        LAD.warning("HTTP patch failed status=%s body=%s", resp.status, txt)
                        return False
            except Exception:
                LAD.exception("K8s HTTP fallback patch exception")
                return False
        LAD.warning("No method available to patch deployment; skipping")
        return False

    def get_deployment_spec_sync(self) -> Optional[Dict[str, Any]]:
        """Sync fetch for simpler CLI uses (blocking)."""
        if _HAS_K8S_PY and self._apps_v1:
            try:
                dep = self._apps_v1.read_namespaced_deployment(self.deployment, self.namespace)
                return dep.to_dict()
            except Exception:
                LAD.exception("Failed to read deployment via k8s client")
                return None
        LAD.warning("k8s python client not available for read")
        return None

class HPAController:
    """
    Controller that reconciles desired worker counts (from storage or ORCHESTRATOR)
    to Kubernetes HPA or Deployment replicas. It runs periodically and is safe/retryable.
    """
    def __init__(self,
                 k8s_operator: Optional[K8sOperator] = None,
                 poll_interval: float = 10.0,
                 policy_engine: Optional[PolicyEngine] = None,
                 storage: Optional[Storage] = None):
        self.k8s = k8s_operator or K8sOperator()
        self.poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.policy_engine = policy_engine or PolicyEngine()
        self.storage = storage or Storage()
        self._last_desired = None

    async def start(self):
        LAD.info("Starting HPAController (poll_interval=%.1f)", self.poll_interval)
        # initialize k8s operator
        self.k8s.initialize()
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        LAD.info("HPAController stopped")

    async def _get_desired_from_storage(self) -> Optional[int]:
        """
        Check storage for a desired worker count written by autoscaler fallback.
        """
        try:
            val = await self.storage.get_desired_worker_count()
            if val is None:
                return None
            return int(val)
        except Exception:
            LAD.debug("Failed to read desired worker count from storage")
            return None

    async def _loop(self):
        while self._running:
            try:
                # Read runtime metrics and policy
                metrics = await ORCHESTRATOR.autoscaler._collect_metrics()
                policy = self.policy_engine.resolve(metrics)
                # prefer desired worker from storage (autoscaler fallback)
                desired = await self._get_desired_from_storage()
                if desired is None:
                    desired = STATE.current_workers  # if none, use state
                # enforce policy bounds
                desired = max(policy["min_workers"], min(policy["max_workers"], int(desired)))
                if desired != self._last_desired:
                    LAD.info("Reconciling desired workers=%d (policy min=%d max=%d)", desired, policy["min_workers"], policy["max_workers"])
                    ok = await self.k8s.patch_deployment_replicas(desired)
                    if ok:
                        self._last_desired = desired
                        await self.storage.insert_hpa_reconcile({"desired": desired, "policy": policy, "ts": iso_now()})
                # optionally reconcile HPA object to match policy (not implemented here)
            except asyncio.CancelledError:
                break
            except Exception:
                LAD.exception("HPAController loop error")
            await async_sleep(self.poll_interval)

# ---------------------------
# Chunk 5b — Automatic HPA creation & reconciliation (extend K8sOperator / HPAController)
# ---------------------------

# Helper: build HPA spec
def _build_hpa_spec(deployment_name: str,
                    min_replicas: int,
                    max_replicas: int,
                    target_cpu_percent: int = 70,
                    stabilization_window_seconds: int = 300,
                    behavior: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create an autoscaling/v2 HPA spec dictionary (strategic-merge patch friendly).
    Uses metric type 'Resource' for CPU targetAverageUtilization by default.
    """
    behavior = behavior or {}
    spec = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": f"{deployment_name}-hpa",
            # namespace set by request path (don't include here for patch body)
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name
            },
            "minReplicas": int(min_replicas),
            "maxReplicas": int(max_replicas),
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": int(target_cpu_percent)
                        }
                    }
                }
            ],
        }
    }
    # optional behavior block for stabilization / scaleUp / scaleDown
    if behavior:
        spec["spec"]["behavior"] = behavior
    else:
        # sensible default behavior (stabilization + limited scale rate)
        spec["spec"]["behavior"] = {
            "scaleUp": {
                "stabilizationWindowSeconds": 0,
                "policies": [
                    {"type": "Percent", "value": 200, "periodSeconds": 60}
                ]
            },
            "scaleDown": {
                "stabilizationWindowSeconds": stabilization_window_seconds,
                "policies": [
                    {"type": "Percent", "value": 50, "periodSeconds": 60}
                ]
            }
        }
    return spec

# Extend K8sOperator to create/read/update HPA
def _ensure_hpa_k8sclient(apps_v1_api, autoscaling_v2_api, namespace: str, hpa_spec: Dict[str, Any]) -> bool:
    """Create or patch HPA using kubernetes client (synchronous helper)."""
    name = hpa_spec["metadata"]["name"]
    body = hpa_spec.copy()
    try:
        # try get
        existing = autoscaling_v2_api.read_namespaced_horizontal_pod_autoscaler(name, namespace)
        LAD.info("HPA '%s' exists; patching to desired spec", name)
        # patch (strategic merge)
        autoscaling_v2_api.patch_namespaced_horizontal_pod_autoscaler(name, namespace, body)
        return True
    except Exception as e:
        # create if not found
        try:
            LAD.info("Creating HPA '%s' in namespace '%s'", name, namespace)
            autoscaling_v2_api.create_namespaced_horizontal_pod_autoscaler(namespace, body)
            return True
        except Exception:
            LAD.exception("Failed to create/patch HPA: %s", e)
            return False

async def _ensure_hpa_http(namespace: str, hpa_spec: Dict[str, Any]) -> bool:
    """
    HTTP fallback to create/patch HPA via Kubernetes API.
    This function is async and uses aiohttp.
    """
    if not _HAS_AIOHTTP:
        LAD.warning("aiohttp not available; cannot create HPA via HTTP fallback")
        return False
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    api_host = os.getenv("KUBERNETES_SERVICE_HOST")
    api_port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
    if not api_host or not os.path.exists(token_path):
        LAD.warning("In-cluster k8s info missing; HTTP HPA creation not possible")
        return False
    with open(token_path, "r") as fh:
        token = fh.read().strip()
    name = hpa_spec["metadata"]["name"]
    url = f"https://{api_host}:{api_port}/apis/autoscaling/v2/namespaces/{namespace}/horizontalpodautoscalers"
    patch_url = f"{url}/{name}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/strategic-merge-patch+json"}
    import ssl
    ssl_ctx = None
    if os.path.exists(ca_path):
        ssl_ctx = ssl.create_default_context(cafile=ca_path)
    async with aiohttp.ClientSession() as sess:
        # try read first
        async with sess.get(patch_url, headers=headers, ssl=ssl_ctx) as resp:
            if resp.status == 200:
                # patch
                async with sess.patch(patch_url, json=hpa_spec, headers=headers, ssl=ssl_ctx) as presp:
                    if presp.status in (200, 201):
                        LAD.info("Patched HPA via HTTP fallback: %s", name)
                        return True
                    else:
                        txt = await presp.text()
                        LAD.warning("HTTP patch HPA failed status=%s body=%s", presp.status, txt)
                        # attempt create as fallback
            # create
        async with sess.post(url, json=hpa_spec, headers=headers, ssl=ssl_ctx) as cresp:
            if cresp.status in (200, 201):
                LAD.info("Created HPA via HTTP fallback: %s", name)
                return True
            txt = await cresp.text()
            LAD.warning("HTTP create HPA failed status=%s body=%s", cresp.status, txt)
            return False

# Attach methods to K8sOperator via monkey-patch style (keeps original class intact)
def _k8soperator_ensure_hpa(self, namespace: str, deployment_name: str,
                           min_replicas: int, max_replicas: int,
                           target_cpu_percent: int = 70,
                           stabilization_window_seconds: int = 300,
                           behavior: Optional[Dict[str, Any]] = None) -> bool:
    """
    Synchronous attempt to ensure HPA exists using kubernetes python client.
    Returns True on success, False otherwise. For async HTTP fallback use ensure_hpa_async.
    """
    spec = _build_hpa_spec(deployment_name, min_replicas, max_replicas, target_cpu_percent, stabilization_window_seconds, behavior)
    # try using client
    if _HAS_K8S_PY:
        try:
            autoscaling_api = k8s_client.AutoscalingV2Api()
            return _ensure_hpa_k8sclient(self._apps_v1, autoscaling_api, namespace, spec)
        except Exception:
            LAD.exception("Error ensuring HPA via kubernetes client")
            return False
    LAD.warning("kubernetes client not available for synchronous HPA ensure")
    return False

async def _k8soperator_ensure_hpa_async(self, namespace: str, deployment_name: str,
                                       min_replicas: int, max_replicas: int,
                                       target_cpu_percent: int = 70,
                                       stabilization_window_seconds: int = 300,
                                       behavior: Optional[Dict[str, Any]] = None) -> bool:
    spec = _build_hpa_spec(deployment_name, min_replicas, max_replicas, target_cpu_percent, stabilization_window_seconds, behavior)
    # prefer python client sync path if available
    if _HAS_K8S_PY:
        try:
            autoscaling_api = k8s_client.AutoscalingV2Api()
            ok = _ensure_hpa_k8sclient(self._apps_v1, autoscaling_api, namespace, spec)
            if ok:
                return True
        except Exception:
            LAD.exception("k8s client HPA ensure failed; falling back to HTTP")
    # HTTP fallback
    return await _ensure_hpa_http(namespace, spec)

# bind the functions to K8sOperator
setattr(K8sOperator, "ensure_hpa", _k8soperator_ensure_hpa)
setattr(K8sOperator, "ensure_hpa_async", _k8soperator_ensure_hpa_async)

# Now extend HPAController to call ensure_hpa during start/reconciliation
_orig_hpacontroller_start = HPAController.start

async def _hpacontroller_start_and_ensure_hpa(self):
    # call original start to set up loop & operator
    await _orig_hpacontroller_start(self)
    try:
        # derive policy bounds from policy engine defaults
        metrics = await ORCHESTRATOR.autoscaler._collect_metrics()
        policy = self.policy_engine.resolve(metrics)
        min_r = int(policy.get("min_workers", AUTOSCALER_MIN_SCALE))
        max_r = int(policy.get("max_workers", AUTOSCALER_MAX_SCALE))
        target_cpu = int(policy.get("target_cpu_percent", 70))
        LAD.info("Ensuring HPA object for deployment=%s namespace=%s (min=%d max=%d cpu=%d)", self.k8s.deployment, self.k8s.namespace, min_r, max_r, target_cpu)
        # try synchronous ensure first (useful for CLI contexts)
        try:
            ok = self.k8s.ensure_hpa(self.k8s.namespace, self.k8s.deployment, min_r, max_r, target_cpu, self.policy_engine.defaults["stabilization_window_sec"])
            if ok:
                LAD.info("HPA ensured via k8s client sync path")
                return
        except Exception:
            LAD.debug("Sync HPA ensure path failed; trying async HTTP ensure")
        # fallback to async ensure (HTTP or client)
        ok2 = await self.k8s.ensure_hpa_async(self.k8s.namespace, self.k8s.deployment, min_r, max_r, target_cpu, self.policy_engine.defaults["stabilization_window_sec"])
        if ok2:
            LAD.info("HPA ensured via async path")
        else:
            LAD.warning("Failed to ensure HPA object via any available path")
    except Exception:
        LAD.exception("Failed while attempting to ensure HPA during HPAController.start()")

# monkey-patch the HPAController.start
setattr(HPAController, "start", _hpacontroller_start_and_ensure_hpa)

# ---------------------------------------------------------------------
# RBAC note and user guidance (log at module import time)
# ---------------------------------------------------------------------
LAD.info("HPA auto-creation module loaded. Ensure service account has RBAC for horizontalpodautoscalers in namespace %s", K8S_NAMESPACE)
LAD.info("If running outside cluster, kubeconfig must be available to kubernetes python client.")

# ---------------------------
# Chunk 5c — Extend HPA spec with custom metrics (queue backlog / latency / RL score)
# ---------------------------

def _build_hpa_spec_with_custom_metrics(
    deployment_name: str,
    min_replicas: int,
    max_replicas: int,
    target_cpu_percent: int = 70,
    custom_metrics: Optional[List[Dict[str, Any]]] = None,
    stabilization_window_seconds: int = 300,
    behavior: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an autoscaling/v2 HPA spec that includes both CPU and custom metrics.
    Example custom metric:
        {
          "name": "prioritymax_queue_backlog",
          "type": "External",
          "target": {
             "type": "AverageValue",
             "averageValue": "500"  # desired backlog threshold
          }
        }
    """
    base_spec = _build_hpa_spec(
        deployment_name,
        min_replicas,
        max_replicas,
        target_cpu_percent,
        stabilization_window_seconds,
        behavior
    )

    # inject additional metrics
    metrics_block = base_spec["spec"]["metrics"]
    if custom_metrics:
        for m in custom_metrics:
            try:
                metrics_block.append({
                    "type": m.get("type", "External"),
                    m.get("type", "External").lower(): {
                        "metric": {
                            "name": m["name"]
                        },
                        "target": m["target"]
                    }
                })
            except Exception:
                LAD.exception("Invalid custom metric spec: %s", m)
    return base_spec


async def ensure_custom_metrics_hpa(
    namespace: str,
    deployment_name: str,
    min_replicas: int,
    max_replicas: int,
    cpu_target: int = 70,
    custom_metrics: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Asynchronous helper that ensures an HPA with custom metrics exists.
    Delegates to kubernetes client if available; falls back to aiohttp HTTP patch.
    """
    spec = _build_hpa_spec_with_custom_metrics(
        deployment_name,
        min_replicas,
        max_replicas,
        cpu_target,
        custom_metrics=custom_metrics
    )

    # prefer kubernetes client
    if _HAS_K8S_PY:
        try:
            autoscaling_api = k8s_client.AutoscalingV2Api()
            ok = _ensure_hpa_k8sclient(None, autoscaling_api, namespace, spec)
            if ok:
                LAD.info("Custom-metrics HPA ensured via kubernetes client")
                return True
        except Exception:
            LAD.exception("k8s client custom-metrics HPA ensure failed")

    # fallback HTTP
    LAD.debug("Falling back to HTTP ensure for custom-metrics HPA")
    return await _ensure_hpa_http(namespace, spec)


# integrate into K8sOperator
async def _k8soperator_ensure_custom_metrics_hpa(
    self,
    custom_metrics: List[Dict[str, Any]],
    min_replicas: int,
    max_replicas: int,
    cpu_target: int = 70
) -> bool:
    try:
        return await ensure_custom_metrics_hpa(
            self.namespace,
            self.deployment,
            min_replicas,
            max_replicas,
            cpu_target,
            custom_metrics
        )
    except Exception:
        LAD.exception("Failed to ensure custom-metrics HPA via operator")
        return False

setattr(K8sOperator, "ensure_custom_metrics_hpa", _k8soperator_ensure_custom_metrics_hpa)

# ---------------------------------------------------------------------
# Example integration during HPAController startup
# ---------------------------------------------------------------------
async def _hpacontroller_start_with_custom_metrics(self):
    """Ensure HPA with custom metrics on start (if policy enables)."""
    await _orig_hpacontroller_start(self)
    try:
        metrics = await ORCHESTRATOR.autoscaler._collect_metrics()
        policy = self.policy_engine.resolve(metrics)
        min_r = int(policy.get("min_workers", AUTOSCALER_MIN_SCALE))
        max_r = int(policy.get("max_workers", AUTOSCALER_MAX_SCALE))
        cpu_target = int(policy.get("target_cpu_percent", 70))
        custom_metrics_cfg = policy.get("custom_metrics", [
            {
                "name": "prioritymax_queue_backlog",
                "type": "External",
                "target": {
                    "type": "AverageValue",
                    "averageValue": "500"
                }
            },
            {
                "name": "prioritymax_rl_score",
                "type": "External",
                "target": {
                    "type": "AverageValue",
                    "averageValue": "0.5"
                }
            }
        ])
        LAD.info("Ensuring HPA with custom metrics for %s (metrics=%s)",
                 self.k8s.deployment, [m['name'] for m in custom_metrics_cfg])
        ok = await self.k8s.ensure_custom_metrics_hpa(
            custom_metrics_cfg,
            min_r,
            max_r,
            cpu_target
        )
        if ok:
            LAD.info("Custom metrics HPA ensured successfully")
        else:
            LAD.warning("Custom metrics HPA ensure failed")
    except Exception:
        LAD.exception("Error ensuring custom metrics HPA")

# override start again to call this version
setattr(HPAController, "start", _hpacontroller_start_with_custom_metrics)

LAD.info("Custom metrics HPA extension loaded: will attach External metrics (queue backlog, RL score) to HPA objects.")


# ---------------------------------------------------------------------
# Operator orchestrator & lifecycle
# ---------------------------------------------------------------------
class K8sOperatorService:
    """
    High-level service that runs the HPAController and monitors deployment state.
    """
    def __init__(self, hpa_poll_interval: float = 10.0):
        self.hpa_controller = HPAController(poll_interval=hpa_poll_interval)
        self._started = False

    async def start(self):
        LAD.info("Starting K8sOperatorService")
        await self.hpa_controller.start()
        self._started = True

    async def stop(self):
        LAD.info("Stopping K8sOperatorService")
        await self.hpa_controller.stop()
        self._started = False

# ---------------------------------------------------------------------
# Wire into orchestrator (start operator alongside autoscaler if enabled)
# ---------------------------------------------------------------------
K8S_OPERATOR = K8sOperatorService()

async def start_k8s_operator_service():
    try:
        await K8S_OPERATOR.start()
    except Exception:
        LAD.exception("Failed to start K8sOperatorService")

async def stop_k8s_operator_service():
    try:
        await K8S_OPERATOR.stop()
    except Exception:
        LAD.exception("Failed to stop K8sOperatorService")
# ---------------------------
# Chunk 6 — Persistence, Analytics & Audit Subsystem
# ---------------------------

class AutoscalerAnalytics:
    """
    Aggregates scaling events and provides analytical summaries
    for dashboards, monitoring, and ML feedback retraining.
    """
    def __init__(self, storage: Storage):
        self.storage = storage
        self._buffer: List[Dict[str, Any]] = []
        self._export_interval = 60
        self._running = False
        self._task = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        LAD.info("AutoscalerAnalytics started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        LAD.info("AutoscalerAnalytics stopped")

    async def _loop(self):
        """Periodic export loop."""
        while self._running:
            try:
                await self._export_snapshot()
            except asyncio.CancelledError:
                break
            except Exception:
                LAD.exception("Analytics exporter error")
            await async_sleep(self._export_interval)

    async def add_event(self, event: Dict[str, Any]):
        """Buffer a scaling or anomaly event for analytics."""
        event["ts"] = iso_now()
        self._buffer.append(event)
        if len(self._buffer) >= 20:
            await self._flush()

    async def _flush(self):
        """Flush events to storage."""
        if not self._buffer:
            return
        try:
            await self.storage.insert_many("scaling_analytics", self._buffer)
            LAD.debug("Flushed %d analytics events", len(self._buffer))
            self._buffer.clear()
        except Exception:
            LAD.exception("Failed to flush analytics events")

    async def _export_snapshot(self):
        """
        Compute aggregated metrics (rolling stats) for Prometheus or dashboards.
        """
        try:
            hist = STATE.history.recent(100)
            if not hist:
                return
            avg_score = float(statistics.mean([h["score"] for h in hist]))
            up = sum(1 for h in hist if h["action"] == "scale_up")
            down = sum(1 for h in hist if h["action"] == "scale_down")
            steady = sum(1 for h in hist if h["action"] == "steady")
            up_ratio = up / max(1, len(hist))
            down_ratio = down / max(1, len(hist))
            LAD.debug("Analytics export: avg=%.3f up=%.2f down=%.2f steady=%d", avg_score, up_ratio, down_ratio, steady)
            # Optionally push to Prometheus gauges
            if _HAS_PROM:
                SCALE_SCORE.set(avg_score)
                SCALE_DECISIONS.labels(action="scale_up").inc(up)
                SCALE_DECISIONS.labels(action="scale_down").inc(down)
            # persist snapshot
            await self.storage.insert_scaling_summary({
                "avg_score": avg_score,
                "up_ratio": up_ratio,
                "down_ratio": down_ratio,
                "steady": steady,
                "ts": iso_now()
            })
        except Exception:
            LAD.exception("Analytics snapshot export failed")

# ---------------------------
# Drift & Anomaly Detection
# ---------------------------
class AutoscalerDriftAnalyzer:
    """
    Detects long-term drift in scaling effectiveness or queue latency
    compared to predicted metrics.
    """
    def __init__(self, storage: Storage):
        self.storage = storage

    async def detect_drift(self):
        """
        Compare predicted vs actual scaling outcomes over time.
        Returns drift stats useful for retraining ML models.
        """
        try:
            records = await self.storage.fetch_scaling_history(limit=200)
            if not records:
                return None
            actual_scores = [r["score"] for r in records if r.get("score") is not None]
            target_workers = [r["to"] for r in records if r.get("to") is not None]
            avg_score = statistics.mean(actual_scores)
            stdev = statistics.pstdev(actual_scores)
            if stdev > 0.8 or abs(avg_score) > 1.0:
                LAD.warning("Scaling drift detected (avg_score=%.3f stdev=%.3f)", avg_score, stdev)
                await self.storage.insert_drift_event({"avg_score": avg_score, "stdev": stdev, "ts": iso_now()})
            return {"avg_score": avg_score, "stdev": stdev}
        except Exception:
            LAD.exception("Drift detection failed")
            return None

    async def analyze_correlation(self):
        """
        Correlate scaling actions with queue backlog improvements.
        """
        try:
            hist = await self.storage.fetch_scaling_history(limit=100)
            if not hist:
                return None
            improvements = []
            for h in hist:
                qstat_before = h.get("debug", {}).get("rule", 0)
                qstat_after = h.get("debug", {}).get("ml", {}).get("pred_raw", 0)
                if qstat_after and qstat_before:
                    diff = (qstat_before - qstat_after) / max(1, abs(qstat_before))
                    improvements.append(diff)
            if improvements:
                avg_improvement = statistics.mean(improvements)
                LAD.info("Avg queue improvement post-scale: %.2f%%", avg_improvement * 100)
                await self.storage.insert_correlation_stat({
                    "avg_improvement": avg_improvement,
                    "ts": iso_now()
                })
            return {"avg_improvement": avg_improvement}
        except Exception:
            LAD.exception("Correlation analysis failed")
            return None

# ---------------------------
# Integrated Analytics Controller
# ---------------------------
class AutoscalerTelemetry:
    """
    Combines analytics + drift analyzer + exporter into unified telemetry service.
    Runs continuously and provides diagnostic snapshots for dashboards.
    """
    def __init__(self, storage: Storage):
        self.analytics = AutoscalerAnalytics(storage)
        self.drift = AutoscalerDriftAnalyzer(storage)
        self._running = False
        self._task = None
        self.interval = 120  # seconds

    async def start(self):
        if self._running:
            return
        await self.analytics.start()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        LAD.info("AutoscalerTelemetry started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        await self.analytics.stop()
        LAD.info("AutoscalerTelemetry stopped")

    async def _loop(self):
        while self._running:
            try:
                await self.drift.detect_drift()
                await self.drift.analyze_correlation()
            except asyncio.CancelledError:
                break
            except Exception:
                LAD.exception("Telemetry loop error")
            await async_sleep(self.interval)

# ---------------------------
# Instantiate and wire telemetry into orchestrator
# ---------------------------
TELEMETRY = AutoscalerTelemetry(ORCHESTRATOR.autoscaler.storage)

async def start_telemetry_service():
    try:
        await TELEMETRY.start()
    except Exception:
        LAD.exception("Failed to start telemetry service")

async def stop_telemetry_service():
    try:
        await TELEMETRY.stop()
    except Exception:
        LAD.exception("Failed to stop telemetry service")
# ---------------------------
# Chunk 7 — CLI Interface, Admin Commands & Entrypoint
# ---------------------------

import argparse
import pprint

def _format_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    """Simple text table formatter for CLI."""
    if not rows:
        return "(no data)"
    widths = [max(len(str(r.get(c, ""))) for r in rows + [{c: c}]) + 2 for c in cols]
    header = "".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "".join("-" * w for w in widths)
    body = "\n".join("".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)) for r in rows)
    return f"{header}\n{sep}\n{body}"

async def _cmd_status(args):
    LAD.info("Fetching autoscaler status …")
    data = {
        "current_workers": STATE.current_workers,
        "last_action": STATE.last_action,
        "last_score": STATE.last_score,
        "last_hint": STATE.last_hint,
        "cooldown_active": STATE.is_in_cooldown(),
        "health_status": STATE.health_status,
        "recovery_mode": STATE.recovery_mode,
        "summary": STATE.history.summary()
    }
    pprint.pprint(data)

async def _cmd_history(args):
    hist = STATE.history.recent(args.limit)
    cols = ["timestamp", "action", "from", "to", "score"]
    print(_format_table(hist, cols))

async def _cmd_trigger(args):
    LAD.info("Manual autoscaler tick …")
    await ORCHESTRATOR.autoscaler._loop_once()
    LAD.info("Manual tick completed")

async def _cmd_restart(args):
    LAD.info("Restarting orchestrator …")
    await ORCHESTRATOR.restart()
    LAD.info("Restart complete")

async def _cmd_analyze(args):
    LAD.info("Running analytics & drift detection …")
    res1 = await TELEMETRY.drift.detect_drift()
    res2 = await TELEMETRY.drift.analyze_correlation()
    print("Drift:", res1)
    print("Correlation:", res2)

async def _cmd_test(args):
    """
    Dry-run a simulated autoscale decision with given metrics.
    Example:
      python -m app.autoscaler test --backlog 800 --latency 1.2 --cpu 0.7
    """
    metrics = {
        "queue": {"backlog": args.backlog, "dlq_backlog": 0},
        "system": {"avg_latency": args.latency, "cpu_util": args.cpu, "throughput": 500}
    }
    rule_score = await ORCHESTRATOR.autoscaler._compute_rule_score(metrics)
    ml_hint, _ = await ORCHESTRATOR.autoscaler._compute_ml_hint(metrics)
    rl_hint, _ = await ORCHESTRATOR.autoscaler._compute_rl_hint(metrics)
    comp, dbg = await ORCHESTRATOR.autoscaler._fuse_scores(rule_score, ml_hint, rl_hint, metrics)
    action = ORCHESTRATOR.autoscaler._interpret_score(comp)
    print(json.dumps({
        "rule_score": rule_score,
        "ml_hint": ml_hint,
        "rl_hint": rl_hint,
        "composite": comp,
        "action": action,
        "debug": dbg
    }, indent=2))

def _build_cli():
    p = argparse.ArgumentParser(prog="prioritymax-autoscaler", description="PriorityMax Autoscaler CLI")
    sub = p.add_subparsers(dest="cmd", help="Available commands")

    sub.add_parser("status", help="Show autoscaler status")
    h = sub.add_parser("history", help="Show recent scaling history")
    h.add_argument("--limit", type=int, default=10)
    sub.add_parser("trigger", help="Run one autoscaling tick manually")
    sub.add_parser("restart", help="Restart orchestrator service")
    sub.add_parser("analyze", help="Run analytics / drift analysis")
    t = sub.add_parser("test", help="Dry-run autoscale decision")
    t.add_argument("--backlog", type=int, default=100)
    t.add_argument("--latency", type=float, default=1.0)
    t.add_argument("--cpu", type=float, default=0.5)
    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()

    async def _run_async():
        if args.cmd == "status":
            await _cmd_status(args)
        elif args.cmd == "history":
            await _cmd_history(args)
        elif args.cmd == "trigger":
            await _cmd_trigger(args)
        elif args.cmd == "restart":
            await _cmd_restart(args)
        elif args.cmd == "analyze":
            await _cmd_analyze(args)
        elif args.cmd == "test":
            await _cmd_test(args)
        else:
            parser.print_help()

    try:
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        LAD.info("CLI interrupted")

# ---------------------------
# Module exports
# ---------------------------
__all__ = [
    "PriorityMaxAutoscaler",
    "AutoscalerOrchestrator",
    "AutoscalerHealthMonitor",
    "AutoscalerAnalytics",
    "AutoscalerTelemetry",
    "AutoscalerDriftAnalyzer",
    "start_autoscaler_service",
    "stop_autoscaler_service",
    "start_telemetry_service",
    "stop_telemetry_service",
    "register_autoscaler",
    "ORCHESTRATOR",
    "TELEMETRY",
    "main_cli"
]

if __name__ == "__main__":
    main_cli()

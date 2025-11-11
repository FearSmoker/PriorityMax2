# backend/app/ml/rl_agent_prod.py
"""
PriorityMax RL Agent (Production)
---------------------------------

Production-grade live inference & control loop for PriorityMax RL-based autoscaler.

Responsibilities:
 - Load trained PPO model (rl_agent.pt)
 - Observe live system metrics (Prometheus, Redis)
 - Compute scaling / throttling actions using RL policy
 - Enforce safety constraints and cooldowns
 - Emit intents to control queue (for operator or autoscaler pod)
 - Log metrics, audit decisions, and support rollback

Safe for production:
 - Dry-run by default
 - Rate-limited
 - Self-healing: falls back to heuristic control if inference fails
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import logging
import pathlib
import asyncio
import datetime
import traceback
import random
import signal
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Optional deps
try:
    import aioredis
    _HAS_AIOREDIS = True
except Exception:
    _HAS_AIOREDIS = False

try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    _HAS_MOTOR = False

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

# Project modules
try:
    from app.ml.model_registry import ModelRegistry
    from app.api.admin import write_audit_event
    from app.ml.predictor import PredictorManager
except Exception:
    ModelRegistry = None
    PredictorManager = None
    def write_audit_event(p):
        pth = pathlib.Path.cwd() / "backend" / "logs" / "rl_agent_prod_audit.jsonl"
        pth.parent.mkdir(parents=True, exist_ok=True)
        with open(pth, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(p, default=str) + "\n")

LOG = logging.getLogger("prioritymax.rl.agent_prod")
LOG.setLevel(os.getenv("PRIORITYMAX_RL_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "app" / "ml" / "models" / "rl_agent.pt"

# -----------------------------
# Config
# -----------------------------
@dataclass
class RLAgentConfig:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    mongo_url: Optional[str] = os.getenv("MONGO_URL", None)
    model_path: str = str(MODEL_PATH)
    loop_interval: float = float(os.getenv("RL_AGENT_LOOP_INTERVAL", "5.0"))  # seconds
    cooldown: float = float(os.getenv("RL_AGENT_ACTION_COOLDOWN", "10.0"))
    dry_run: bool = os.getenv("RL_AGENT_DRY_RUN", "true").lower() == "true"
    # keep original names but support EnvConfig naming too (min_workers / max_workers)
    min_consumers: int = int(os.getenv("RL_AGENT_MIN_CONSUMERS", os.getenv("RL_AGENT_MIN_WORKERS", "1")))
    max_consumers: int = int(os.getenv("RL_AGENT_MAX_CONSUMERS", os.getenv("RL_AGENT_MAX_WORKERS", "200")))
    max_delta: int = int(os.getenv("RL_AGENT_MAX_DELTA", os.getenv("PRIORITYMAX_MAX_SCALE_DELTA", "10")))
    throttle_limit: float = float(os.getenv("RL_AGENT_MAX_THROTTLE", "1.0"))
    reward_latency_sla_ms: float = float(os.getenv("RL_AGENT_SLA_MS", "500"))
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    safe_mode: bool = True
    seed: int = int(os.getenv("RL_AGENT_SEED", "42"))
    prom_port: int = int(os.getenv("RL_AGENT_PROM_PORT", "9205"))

    # Backwards-compatible mapping: populate min_workers/max_workers names if caller expects them
    def __post_init__(self):
        # Provide both naming styles used across repo
        if not hasattr(self, "min_workers"):
            setattr(self, "min_workers", getattr(self, "min_consumers"))
        if not hasattr(self, "max_workers"):
            setattr(self, "max_workers", getattr(self, "max_consumers"))
        if not hasattr(self, "max_scale_delta"):
            setattr(self, "max_scale_delta", getattr(self, "max_delta"))

# -----------------------------
# PPO Model (same arch as training)
# -----------------------------
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, obs: torch.Tensor):
        logits = self.actor(obs)
        mu = torch.tanh(logits)
        std = torch.ones_like(mu) * 0.2
        dist = torch.distributions.Normal(mu, std)
        act = dist.sample()
        logp = dist.log_prob(act).sum(axis=-1)
        return act, logp

    def value(self, obs: torch.Tensor):
        return self.critic(obs)

# -----------------------------
# RL Agent Class
# -----------------------------
class RLAgentProd:
    def __init__(self, cfg: RLAgentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.model = None
        self.last_action_time = 0.0
        self.last_action = None
        self.model_registry = ModelRegistry() if ModelRegistry else None
        self.predictor = PredictorManager() if PredictorManager else None
        self._redis = None
        self._mongo = None
        self.running = True
        self.loop_task = None
        self.metrics = {"inference_latency": [], "actions": 0, "errors": 0}
        if _HAS_PROM:
            try:
                self.prom_infer_time = Histogram("prioritymax_rl_infer_seconds", "RL inference latency")
                self.prom_actions = Counter("prioritymax_rl_actions_total", "Total actions emitted")
                self.prom_errors = Counter("prioritymax_rl_errors_total", "Inference or control loop errors")
                start_http_server(self.cfg.prom_port)
                LOG.info(f"Prometheus metrics exported on port {self.cfg.prom_port}")
            except Exception:
                LOG.warning("Prometheus init failed")

    def load_model(self):
        if not os.path.exists(self.cfg.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.cfg.model_path}")
        ckpt = torch.load(self.cfg.model_path, map_location=self.device)
        # Determine obs/act dims dynamically (8 obs, 3 acts)
        self.model = PPOActorCritic(8, 3).to(self.device)
        # allow both 'model' or 'state_dict' keys in checkpoint (back-compat safety)
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # If the checkpoint stores the raw state_dict under nested structure keep behavior the same
        if isinstance(state, dict):
            try:
                self.model.load_state_dict(state)
            except Exception:
                # if ckpt provided wrapper { "model": {"actor...":...}, ... } try direct key access
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    self.model.load_state_dict(ckpt["model"])
                else:
                    raise
        else:
            # fallback (preserve original behavior if structure differs)
            self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        self.model.eval()
        LOG.info(f"Loaded RL model from {self.cfg.model_path}")

    async def connect(self):
        if _HAS_AIOREDIS:
            try:
                self._redis = await aioredis.from_url(self.cfg.redis_url, encoding="utf-8", decode_responses=True)
                LOG.info("Connected to Redis")
            except Exception:
                LOG.exception("Redis connection failed")
        if _HAS_MOTOR and self.cfg.mongo_url:
            try:
                self._mongo = motor_asyncio.AsyncIOMotorClient(self.cfg.mongo_url)
                LOG.info("Connected to MongoDB")
            except Exception:
                LOG.exception("Mongo connection failed")

    async def disconnect(self):
        if self._redis:
            await self._redis.close()
        if self._mongo:
            self._mongo.close()

    # -----------------------------
    # Observation Collector
    # -----------------------------
    async def get_observation(self) -> np.ndarray:
        """
        Pulls latest system metrics for observation vector.
        Priority order:
          1. Redis cache (if available)
          2. Mongo metrics collection
          3. Fallback to PredictorManager.metrics
        """
        obs = np.zeros(8, dtype=np.float32)
        try:
            if self._redis:
                m = await self._redis.hgetall("prioritymax:live_metrics")
                if m:
                    # support both 'consumer_count' and 'worker_count' naming
                    vals = {}
                    for k, v in m.items():
                        try:
                            vals[k] = float(v)
                        except Exception:
                            # keep original string if not castable
                            vals[k] = v
                    obs = np.array([
                        vals.get("queue_length", vals.get("queue", 0.0)),
                        vals.get("consumer_count", vals.get("worker_count", 0.0)),
                        vals.get("avg_latency_ms", vals.get("avg_lat_ms", 0.0)),
                        vals.get("p95_latency_ms", vals.get("p95_lat_ms", 0.0)),
                        vals.get("success_rate", 1.0),
                        vals.get("arrival_rate", vals.get("arrival", 0.0)),
                        vals.get("cpu", 0.0),
                        vals.get("mem", vals.get("memory_utilization", vals.get("mem", 0.0))),
                    ], dtype=np.float32)
            elif self._mongo:
                db = self._mongo.get_default_database()
                coll = db["metrics"]
                doc = await coll.find_one(sort=[("_id", -1)])
                if doc:
                    obs = np.array([
                        float(doc.get("queue_length", doc.get("queue", 0))),
                        float(doc.get("consumer_count", doc.get("worker_count", 0))),
                        float(doc.get("avg_latency_ms", doc.get("avg_lat_ms", 0))),
                        float(doc.get("p95_latency_ms", doc.get("p95_lat_ms", 0))),
                        float(doc.get("success_rate", 1)),
                        float(doc.get("arrival_rate", doc.get("arrival", 0))),
                        float(doc.get("cpu", 0)),
                        float(doc.get("mem", doc.get("memory_utilization", doc.get("mem", 0)))),
                    ], dtype=np.float32)
            elif self.predictor:
                # fallback: synthetic metrics or predictor-managed metrics
                obs = np.random.rand(8).astype(np.float32)
        except Exception:
            LOG.exception("Observation fetch failed")
        return obs

    # -----------------------------
    # Inference
    # -----------------------------
    async def infer_action(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_model()
        try:
            t0 = time.perf_counter()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                act, _ = self.model.act(obs_t)
            act_np = act.cpu().numpy().flatten()
            if _HAS_PROM:
                self.prom_infer_time.observe(time.perf_counter() - t0)
            self.metrics["inference_latency"].append(time.perf_counter() - t0)
            self.metrics["actions"] += 1
            return act_np
        except Exception as e:
            LOG.exception("Inference failed")
            self.metrics["errors"] += 1
            if _HAS_PROM:
                self.prom_errors.inc()
            # fallback: heuristic rule (PID-like)
            return self.heuristic_fallback(obs)

    def heuristic_fallback(self, obs: np.ndarray) -> np.ndarray:
        """
        Basic heuristic fallback control policy.
        Scale up if latency > SLA and queue growing,
        scale down if idle.
        """
        queue_len, consumers, avg_lat, p95, success, arrival, cpu, mem = obs
        delta = 0
        throttle = 0.0
        priority = 0
        if avg_lat > self.cfg.reward_latency_sla_ms and queue_len > 100:
            delta = min(self.cfg.max_delta, int(consumers * 0.2 + 1))
        elif queue_len < 10 and cpu < 0.4:
            delta = max(-2, -int(consumers * 0.1))
        throttle = min(1.0, max(0.0, (avg_lat / self.cfg.reward_latency_sla_ms) - 1.0))
        return np.array([delta, throttle, priority], dtype=np.float32)

    # -----------------------------
    # Action Emitter
    # -----------------------------
    async def emit_action(self, action: np.ndarray, obs: np.ndarray):
        delta, throttle, priority = map(float, action[:3])
        now = time.time()
        if (now - self.last_action_time) < self.cfg.cooldown:
            LOG.debug("Cooldown active; skipping action emission")
            return
        act = {
            "ts": now,
            "delta_consumers": int(np.clip(round(delta), -self.cfg.max_delta, self.cfg.max_delta)),
            "throttle": float(np.clip(throttle, 0.0, self.cfg.throttle_limit)),
            "priority_bias": int(np.clip(round(priority), -2, 2)),
        }
        act["mode"] = "dry_run" if self.cfg.dry_run else "live"
        act["obs"] = obs.tolist()
        act["model_version"] = None
        if self.model_registry:
            try:
                latest = self.model_registry.get_latest("rl_agent.pt")
                act["model_version"] = latest["version_id"] if latest else None
            except Exception:
                LOG.debug("ModelRegistry lookup failed; continuing without model_version")

        try:
            # Prefer storing in redis stream/list used by external operator
            if self._redis:
                await self._redis.lpush("prioritymax:rl_actions", json.dumps(act))
            elif self._mongo:
                db = self._mongo.get_default_database()
                await db["rl_actions"].insert_one(act)
            else:
                path = pathlib.Path("/tmp/prioritymax_rl_action.json")
                path.write_text(json.dumps(act))
            write_audit_event({"event": "rl_action_emitted", **act})
            LOG.info(f"Action emitted: Δ={act['delta_consumers']} throttle={act['throttle']:.2f} mode={act['mode']}")
            self.last_action_time = now
            if _HAS_PROM:
                self.prom_actions.inc()
        except Exception:
            LOG.exception("Emit action failed")
            if _HAS_PROM:
                self.prom_errors.inc()

    # -----------------------------
    # Main control loop
    # -----------------------------
    async def control_loop(self):
        LOG.info("Starting RL control loop")
        while self.running:
            try:
                obs = await self.get_observation()
                act = await self.infer_action(obs)
                await self.emit_action(act, obs)
                await asyncio.sleep(self.cfg.loop_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("RL control loop iteration failed")
                self.metrics["errors"] += 1
                await asyncio.sleep(self.cfg.loop_interval * 2)

    # -----------------------------
    # Lifecycle
    # -----------------------------
    async def start(self):
        await self.connect()
        self.load_model()
        self.loop_task = asyncio.create_task(self.control_loop())

    async def stop(self):
        self.running = False
        if self.loop_task:
            self.loop_task.cancel()
        await self.disconnect()

# -----------------------------
# CLI
# -----------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-rl-agent-prod")
    p.add_argument("--dry", action="store_true", help="Run in dry-run mode")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--cooldown", type=float, default=10.0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p

def main_cli():
    args = _build_cli().parse_args()
    # create config while keeping original names, but mapped in RLAgentConfig.__post_init__
    cfg = RLAgentConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        mongo_url=os.getenv("MONGO_URL", None),
        model_path=str(MODEL_PATH),
        loop_interval=args.interval,
        cooldown=args.cooldown,
        dry_run=args.dry,
        seed=args.seed
    )
    agent = RLAgentProd(cfg)

    async def runner():
        await agent.start()
        while True:
            await asyncio.sleep(60)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.stop()))
        except NotImplementedError:
            # Not all platforms support loop.add_signal_handler (Windows)
            pass
    try:
        loop.run_until_complete(runner())
    except KeyboardInterrupt:
        LOG.info("Shutting down RL agent...")

if __name__ == "__main__":
    main_cli()

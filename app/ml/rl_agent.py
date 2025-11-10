# backend/app/ml/rl_agent.py
"""
PriorityMax RL Agent - Unified Runtime
-------------------------------------

This module implements a production-minded, full-featured RL Agent runtime for PriorityMax.
It unifies functionality discussed across the project:

 - Production controller (rl_agent_prod): loads a checkpoint, collects observations from
   Redis/Mongo/metrics cache, runs inference, emits scaling intents (safe by default).
 - Sandbox mode (rl_agent_sandbox features): can run against SimulatedRealEnv for local testing.
 - Hot-reload of model checkpoints for quick iteration / canary testing.
 - Heuristic fallback policy if model fails or is not available.
 - Prometheus metrics export (optional).
 - Optional FastAPI status / health endpoint for observability & debugging.
 - Async control loop with safety (cooldown, clipping, dry-run).
 - Audit logging (best-effort) and optional Mongo persistence of actions.
 - CLI entrypoints for running in prod, sandbox, or one-shot inference.
 - Supports GPU inference if torch.cuda is available.

Notes:
 - This module intentionally avoids performing destructive actions in-library.
   It writes intents to Redis / Mongo / local file for an external operator to apply.
 - Most destructive features are opt-in via configuration (dry_run=False).
 - Designed to be importable by other modules (e.g., Kubernetes operator, FastAPI app).
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
import signal
import threading
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple, Union, Callable

import numpy as np

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    _HAS_TORCH = False

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
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    _HAS_PROM = True
except Exception:
    start_http_server = Gauge = Counter = Histogram = None
    _HAS_PROM = False

try:
    from fastapi import FastAPI, APIRouter
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _HAS_FASTAPI = True
except Exception:
    _HAS_FASTAPI = False

# Project-local optional modules
try:
    from app.ml.real_env import SimulatedRealEnv, EnvConfig
    from app.ml.model_registry import ModelRegistry
    from app.api.admin import write_audit_event
    from app.ml.predictor import PredictorManager
except Exception:
    # graceful fallback stubs
    SimulatedRealEnv = None
    EnvConfig = None
    ModelRegistry = None
    PredictorManager = None

    def write_audit_event(payload: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "rl_agent_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.rl_agent")
LOG.setLevel(os.getenv("PRIORITYMAX_RL_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
MODELS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODELS_DIR", str(BASE_DIR / "app" / "ml" / "models")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RL_MODEL = MODELS_DIR / "rl_agent.pt"

# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class RLAgentConfig:
    """Configuration for RL Agent runtime"""
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    mongo_url: Optional[str] = os.getenv("MONGO_URL", None)
    model_path: str = str(DEFAULT_RL_MODEL)
    loop_interval: float = float(os.getenv("RL_AGENT_LOOP_INTERVAL", "5.0"))
    cooldown: float = float(os.getenv("RL_AGENT_ACTION_COOLDOWN", "10.0"))
    dry_run: bool = os.getenv("RL_AGENT_DRY_RUN", "true").lower() == "true"
    min_consumers: int = int(os.getenv("RL_AGENT_MIN_CONSUMERS", "1"))
    max_consumers: int = int(os.getenv("RL_AGENT_MAX_CONSUMERS", "200"))
    max_delta: int = int(os.getenv("RL_AGENT_MAX_DELTA", "10"))
    throttle_limit: float = float(os.getenv("RL_AGENT_MAX_THROTTLE", "1.0"))
    reward_latency_sla_ms: float = float(os.getenv("RL_AGENT_SLA_MS", "500"))
    device: str = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    hot_reload: bool = os.getenv("RL_AGENT_HOT_RELOAD", "true").lower() == "true"
    safe_mode: bool = True  # internal safeguard
    prom_port: int = int(os.getenv("RL_AGENT_PROM_PORT", "9205"))
    mode: str = os.getenv("RL_AGENT_MODE", "prod")  # prod | sandbox | test
    seed: int = int(os.getenv("RL_AGENT_SEED", "42"))
    audit_db_collection: str = os.getenv("RL_AGENT_AUDIT_COLL", "rl_audit")
    max_retries: int = int(os.getenv("RL_AGENT_MAX_RETRIES", "3"))

# ----------------------------
# PPO ActorCritic (shared architecture)
# ----------------------------
if _HAS_TORCH:
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
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)
            return action, logp

        def value(self, obs: torch.Tensor):
            return self.critic(obs)
else:
    PPOActorCritic = None  # type: ignore

# ----------------------------
# RL Agent Main Class
# ----------------------------
class RLAgent:
    """
    Unified RL Agent class for PriorityMax.

    Modes:
      - prod: live loop using Redis/Mongo and emits intents for external operator.
      - sandbox: runs against SimulatedRealEnv for testing/eval only (no live writes).
      - test: one-shot CLI inference & diagnostics.

    Public methods:
      - start(): start control loop (async)
      - stop(): stop control loop
      - run_once(): run one inference+emit cycle synchronously
      - status(): return current internal status dict
    """

    def __init__(self, cfg: Optional[RLAgentConfig] = None):
        self.cfg = cfg or RLAgentConfig()
        self.device = torch.device(self.cfg.device) if _HAS_TORCH else None
        self.model: Optional[torch.nn.Module] = None
        self.model_mtime: Optional[float] = None
        self.model_lock = threading.Lock()
        self.last_action_time: float = 0.0
        self.last_action: Optional[Dict[str, Any]] = None
        self.running: bool = False
        self.loop_task: Optional[asyncio.Task] = None
        self._redis = None
        self._mongo = None
        self.model_registry = ModelRegistry() if ModelRegistry else None
        self.predictor = PredictorManager() if PredictorManager else None
        self._shutdown = asyncio.Event()
        self._prom_started = False
        self.metrics = {
            "actions_emitted": 0,
            "inference_errors": 0,
            "inference_latency_s": [],
            "loop_cycles": 0,
        }
        # Prometheus metrics
        if _HAS_PROM:
            try:
                self.prom_infer_latency = Histogram("prioritymax_rl_infer_seconds", "RL inference latency seconds")
                self.prom_actions = Counter("prioritymax_rl_actions_total", "Total RL actions emitted")
                self.prom_errors = Counter("prioritymax_rl_errors_total", "Total RL errors")
                # start server lazily in start()
            except Exception:
                LOG.exception("Failed to init Prometheus metrics")
        # Seed randomness
        np.random.seed(self.cfg.seed)
        # If sandbox mode, prepare environment
        if self.cfg.mode == "sandbox" and SimulatedRealEnv and EnvConfig:
            try:
                self.env_cfg = EnvConfig(mode="sim", seed=self.cfg.seed)
                self.sim_env = SimulatedRealEnv(self.env_cfg)
            except Exception:
                self.sim_env = None
        else:
            self.sim_env = None

    # ------------------------
    # Model loading & hot-reload
    # ------------------------
    def load_model(self, force_path: Optional[str] = None) -> bool:
        """
        Load model from cfg.model_path or force_path.
        Returns True if model loaded successfully.
        """
        path = pathlib.Path(force_path or self.cfg.model_path)
        if not path.exists():
            LOG.warning("RL model checkpoint not found at %s", str(path))
            return False
        try:
            if not _HAS_TORCH:
                LOG.warning("Torch not available; cannot load RL model.")
                return False
            mtime = path.stat().st_mtime
            with self.model_lock:
                if self.model is not None and self.model_mtime == mtime:
                    # already loaded and unchanged
                    return True
                ckpt = torch.load(str(path), map_location=self.device)
                # expect ckpt to contain 'model' state_dict
                state = ckpt.get("model", ckpt)
                # instantiate architecture (obs=8, act=3 assumed)
                model = PPOActorCritic(8, 3)
                model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                self.model = model
                self.model_mtime = mtime
                LOG.info("Loaded RL model from %s", str(path))
                write_audit_event({"event": "rl_model_loaded", "path": str(path), "ts": time.time()})
            return True
        except Exception:
            LOG.exception("Failed to load RL model from %s", str(path))
            return False

    def maybe_hot_reload(self):
        """Check if file changed and hot-reload if enabled."""
        if not self.cfg.hot_reload:
            return
        path = pathlib.Path(self.cfg.model_path)
        if not path.exists():
            return
        try:
            mtime = path.stat().st_mtime
            if self.model_mtime is None or mtime > self.model_mtime + 1e-6:
                LOG.info("Detected model file change; reloading")
                self.load_model()
        except Exception:
            LOG.exception("Hot reload check failed")

    # ------------------------
    # Connectors: Redis & Mongo
    # ------------------------
    async def connect_redis(self):
        if not _HAS_AIOREDIS:
            LOG.debug("aioredis not installed; skipping redis connection")
            return
        if self._redis:
            return
        try:
            self._redis = await aioredis.from_url(self.cfg.redis_url, encoding="utf-8", decode_responses=True)
            LOG.info("Connected to Redis at %s", self.cfg.redis_url)
        except Exception:
            LOG.exception("Redis connection failed")

    async def connect_mongo(self):
        if not _HAS_MOTOR or not self.cfg.mongo_url:
            LOG.debug("motor not installed or mongo_url not provided; skipping mongo")
            return
        if self._mongo:
            return
        try:
            self._mongo = motor_asyncio.AsyncIOMotorClient(self.cfg.mongo_url)
            LOG.info("Connected to Mongo at %s", self.cfg.mongo_url)
        except Exception:
            LOG.exception("Mongo connection failed")

    async def close_connectors(self):
        try:
            if self._redis:
                await self._redis.close()
                self._redis = None
            if self._mongo:
                self._mongo.close()
                self._mongo = None
        except Exception:
            LOG.exception("Error closing connectors")

    # ------------------------
    # Observation collection
    # ------------------------
    async def get_observation(self) -> np.ndarray:
        """
        Build observation vector [queue_length, consumer_count, avg_latency_ms, p95_latency_ms,
        success_rate, arrival_rate, cpu, mem] as np.float32 array.
        Priority: Redis hash 'prioritymax:live_metrics' -> Mongo metrics collection -> predictor fallback -> simulated env.
        """
        obs = np.zeros(8, dtype=np.float32)
        # Try Redis
        try:
            if self._redis:
                m = await self._redis.hgetall("prioritymax:live_metrics")
                if m:
                    vals = {}
                    for k, v in m.items():
                        try:
                            vals[k] = float(v)
                        except Exception:
                            vals[k] = 0.0
                    obs = np.array([
                        vals.get("queue_length", 0.0),
                        vals.get("consumer_count", 0.0),
                        vals.get("avg_latency_ms", 0.0),
                        vals.get("p95_latency_ms", 0.0),
                        vals.get("success_rate", 1.0),
                        vals.get("arrival_rate", 0.0),
                        vals.get("cpu", 0.0),
                        vals.get("mem", 0.0),
                    ], dtype=np.float32)
                    return obs
            # Try Mongo
            if self._mongo:
                db = self._mongo.get_default_database()
                doc = await db["metrics"].find_one(sort=[("_id", -1)])
                if doc:
                    obs = np.array([
                        float(doc.get("queue_length", 0)),
                        float(doc.get("consumer_count", 0)),
                        float(doc.get("avg_latency_ms", 0)),
                        float(doc.get("p95_latency_ms", 0)),
                        float(doc.get("success_rate", 1)),
                        float(doc.get("arrival_rate", 0)),
                        float(doc.get("cpu", 0)),
                        float(doc.get("mem", 0)),
                    ], dtype=np.float32)
                    return obs
        except Exception:
            LOG.exception("Failed reading live metrics (redis/mongo)")

        # Predictor fallback
        try:
            if self.predictor:
                # best-effort: use predictor's scaler or synthetic sample
                # Here we simply return a random plausible vector (predictor integration would be more advanced)
                sample = np.random.rand(8).astype(np.float32)
                # scale some entries to realistic ranges
                sample[0] *= 200.0  # queue length
                sample[1] *= 50.0   # consumers
                sample[2] *= 1000.0 # avg latency ms
                sample[3] *= 1500.0 # p95
                sample[4] = float(max(0.0, min(1.0, 0.95 - 0.1 * sample[0] / 200.0)))
                return sample
        except Exception:
            LOG.exception("Predictor fallback failed")

        # Simulated fallback if enabled
        if self.sim_env:
            try:
                return self.sim_env._observe()
            except Exception:
                LOG.exception("Simulated env observe failed")

        # Last resort: zeros
        return obs

    # ------------------------
    # Inference
    # ------------------------
    async def infer_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Run model forward pass to produce action vector [delta_consumers, throttle, priority_bias].
        If model not present or fails, fallback to heuristic.
        Returns numpy array float32 len 3.
        """
        # Hot-reload check
        try:
            if self.cfg.hot_reload:
                self.maybe_hot_reload()
        except Exception:
            LOG.exception("Hot-reload check failed")

        # Model inference
        if self.model is None:
            # attempt load if not loaded
            self.load_model()

        if self.model is not None and _HAS_TORCH:
            try:
                t0 = time.perf_counter()
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    # model.act returns (action, logp)
                    act_t, _ = self.model.act(obs_t)
                # ensure numpy and shape
                act_np = act_t.cpu().numpy().astype(np.float32).ravel()
                latency = time.perf_counter() - t0
                self.metrics["inference_latency_s"].append(latency)
                if _HAS_PROM:
                    try:
                        self.prom_infer_latency.observe(latency)
                    except Exception:
                        pass
                return act_np
            except Exception:
                LOG.exception("Model inference failed; falling back")
                self.metrics["inference_errors"] += 1
                if _HAS_PROM:
                    try:
                        self.prom_errors.inc()
                    except Exception:
                        pass
                return self.heuristic_policy(obs)
        else:
            # no model available
            return self.heuristic_policy(obs)

    def heuristic_policy(self, obs: np.ndarray) -> np.ndarray:
        """Simple rule-based fallback policy."""
        queue_len, consumers, avg_lat, p95, success, arrival, cpu, mem = obs
        delta = 0
        throttle = 0.0
        priority = 0
        # aggressive if p95 >> SLA and queue grows
        if avg_lat > self.cfg.reward_latency_sla_ms and queue_len > (consumers * 2):
            delta = min(self.cfg.max_delta, max(1, int(math.ceil(queue_len / max(1, consumers)) - 1)))
        elif queue_len < max(5, int(consumers / 2)):
            delta = -1
        # throttle proportional to latency / sla
        throttle = float(min(self.cfg.throttle_limit, max(0.0, (avg_lat / max(1.0, self.cfg.reward_latency_sla_ms)) - 1.0)))
        # small priority boost if SLA violated
        if avg_lat > self.cfg.reward_latency_sla_ms:
            priority = 1
        return np.array([float(delta), float(throttle), float(priority)], dtype=np.float32)

    # ------------------------
    # Action emission
    # ------------------------
    async def emit_action(self, action: np.ndarray, obs: np.ndarray) -> Dict[str, Any]:
        """
        Emit action as an intent into Redis list 'prioritymax:rl_actions', Mongo collection, or local file.
        Enforces cooldown and clipping, and writes audit events.
        Returns the emitted action dict.
        """
        now = time.time()
        # Cooldown enforcement
        if (now - self.last_action_time) < self.cfg.cooldown:
            LOG.debug("Cooldown active; suppressing action emission")
            return {"skipped": True, "reason": "cooldown"}

        # Clip / sanitize action
        delta = int(np.clip(round(float(action[0])), -self.cfg.max_delta, self.cfg.max_delta))
        throttle = float(np.clip(float(action[1]) if len(action) > 1 else 0.0, 0.0, self.cfg.throttle_limit))
        priority = int(np.clip(round(float(action[2]) if len(action) > 2 else 0), -2, 2))

        # Ensure consumer limits
        # If we know current consumer count from obs, bound the new consumer count
        current_consumers = int(obs[1]) if obs is not None and len(obs) > 1 else None
        if current_consumers is not None:
            new_consumers = current_consumers + delta
            if new_consumers < self.cfg.min_consumers:
                delta = self.cfg.min_consumers - current_consumers
            elif new_consumers > self.cfg.max_consumers:
                delta = self.cfg.max_consumers - current_consumers

        emitted = {
            "ts": now,
            "delta_consumers": int(delta),
            "throttle": float(throttle),
            "priority_bias": int(priority),
            "dry_run": bool(self.cfg.dry_run),
            "obs": obs.tolist() if isinstance(obs, np.ndarray) else obs,
            "model_version": None,
        }
        # attach model version if available via registry or mtime
        try:
            if os.path.exists(self.cfg.model_path):
                emitted["model_version"] = pathlib.Path(self.cfg.model_path).name + f"@{int(pathlib.Path(self.cfg.model_path).stat().st_mtime)}"
        except Exception:
            pass

        # Write intent
        try:
            if self._redis:
                await self._redis.lpush("prioritymax:rl_actions", json.dumps(emitted))
            elif self._mongo:
                db = self._mongo.get_default_database()
                await db["rl_actions"].insert_one(emitted)
            else:
                # fallback: write to temp file
                p = pathlib.Path(tempfile.gettempdir()) / f"prioritymax_rl_action_{int(time.time()*1000)}.json"
                p.write_text(json.dumps(emitted))
        except Exception:
            LOG.exception("Failed to write RL action intent")
            if _HAS_PROM:
                try:
                    self.prom_errors.inc()
                except Exception:
                    pass

        # Audit & state update
        try:
            write_audit_event({"event": "rl_action_emitted", "action": emitted, "ts": now})
        except Exception:
            LOG.exception("Audit write failed")
        self.last_action_time = now
        self.last_action = emitted
        self.metrics["actions_emitted"] += 1
        if _HAS_PROM:
            try:
                self.prom_actions.inc()
            except Exception:
                pass
        return emitted

    # ------------------------
    # Single-run: infers and emits once (synchronous wrapper)
    # ------------------------
    async def run_once(self) -> Dict[str, Any]:
        """Perform a single observation->inference->emit cycle and return diagnostic dict."""
        obs = await self.get_observation()
        act = await self.infer_action(obs)
        emitted = await self.emit_action(act, obs)
        self.metrics["loop_cycles"] += 1
        return {"obs": obs.tolist(), "action_raw": act.tolist(), "emitted": emitted}

    # ------------------------
    # Control loop
    # ------------------------
    async def control_loop(self):
        """Main loop - runs until stop() called."""
        LOG.info("Starting RLAgent control loop in mode=%s interval=%ss", self.cfg.mode, self.cfg.loop_interval)
        # start Prometheus if configured
        if _HAS_PROM and not self._prom_started:
            try:
                start_http_server(self.cfg.prom_port)
                LOG.info("Prometheus metrics served at port %d", self.cfg.prom_port)
                self._prom_started = True
            except Exception:
                LOG.exception("Failed to start Prometheus server")
        # connect connectors
        try:
            await self.connect_redis()
            await self.connect_mongo()
        except Exception:
            LOG.exception("Failed to init connectors")
        # If sandbox mode, optionally preload model
        if self.cfg.mode == "sandbox" and self.sim_env:
            LOG.info("Sandbox mode: using simulated environment for observations")
        # warm model
        if self.model is None:
            self.load_model()
        # loop
        while not self._shutdown.is_set():
            try:
                cycle_start = time.perf_counter()
                # gather observation
                obs = await self.get_observation()
                # inference
                act = await self.infer_action(obs)
                # emit (respects cooldown)
                emitted = await self.emit_action(act, obs)
                # metrics
                self.metrics["loop_cycles"] += 1
                # sleep until next interval (account for cycle time)
                elapsed = time.perf_counter() - cycle_start
                to_sleep = max(0.0, self.cfg.loop_interval - elapsed)
                await asyncio.wait([self._shutdown.wait()], timeout=to_sleep)
            except asyncio.CancelledError:
                LOG.info("Control loop cancelled")
                break
            except Exception:
                LOG.exception("Control loop iteration failure")
                self.metrics["inference_errors"] += 1
                if _HAS_PROM:
                    try:
                        self.prom_errors.inc()
                    except Exception:
                        pass
                # backoff a bit on error
                await asyncio.sleep(min(5.0, self.cfg.loop_interval * 2.0))
        LOG.info("Control loop stopped")

    # ------------------------
    # Public lifecycle methods
    # ------------------------
    async def start(self):
        """Start agent (async) - schedules control loop on current event loop."""
        if self.running:
            LOG.warning("Agent already running")
            return
        self.running = True
        self._shutdown.clear()
        loop = asyncio.get_running_loop()
        self.loop_task = loop.create_task(self.control_loop())
        LOG.info("Agent started (task=%s)", self.loop_task)

    async def stop(self):
        """Stop agent gracefully."""
        if not self.running:
            LOG.warning("Agent not running")
            return
        self._shutdown.set()
        if self.loop_task:
            self.loop_task.cancel()
            try:
                await self.loop_task
            except Exception:
                pass
        await self.close_connectors()
        self.running = False
        LOG.info("Agent stopped")

    def status(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot (suitable for health endpoints)."""
        return {
            "running": self.running,
            "mode": self.cfg.mode,
            "last_action_time": self.last_action_time,
            "last_action": self.last_action,
            "model_loaded": bool(self.model),
            "metrics": {k: (float(np.mean(v)) if isinstance(v, list) and v else v) for k, v in self.metrics.items() if not isinstance(v, dict)},
        }

# ----------------------------
# FastAPI status server (optional)
# ----------------------------
class RLAgentAPI:
    """
    Optional lightweight status API to inspect and control the agent.
    Provides:
      - /status
      - /start
      - /stop
      - /run_once
      - /health
    """
    def __init__(self, agent: RLAgent, host: str = "0.0.0.0", port: int = 8000):
        if not _HAS_FASTAPI:
            raise RuntimeError("FastAPI not installed")
        self.agent = agent
        self.host = host
        self.port = port
        self.app = FastAPI(title="PriorityMax RL Agent API")
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        self.router = APIRouter(prefix="/rl")
        self._register_routes()
        self.app.include_router(self.router)

    def _register_routes(self):
        @self.router.get("/status")
        async def status():
            return JSONResponse(self.agent.status())

        @self.router.post("/start")
        async def start():
            if self.agent.running:
                return JSONResponse({"ok": False, "reason": "already_running"})
            loop = asyncio.get_running_loop()
            await self.agent.start()
            return JSONResponse({"ok": True})

        @self.router.post("/stop")
        async def stop():
            if not self.agent.running:
                return JSONResponse({"ok": False, "reason": "not_running"})
            await self.agent.stop()
            return JSONResponse({"ok": True})

        @self.router.post("/run_once")
        async def run_once():
            res = await self.agent.run_once()
            return JSONResponse(res)

        @self.router.get("/health")
        async def health():
            st = self.agent.status()
            healthy = (st.get("model_loaded", False) or self.cfg_mode_allows_no_model()) and (st.get("metrics", {}).get("actions_emitted", 0) >= 0)
            return JSONResponse({"healthy": healthy, "status": st})

    def cfg_mode_allows_no_model(self):
        # In sandbox/test mode, it's OK to not have model
        return self.agent.cfg.mode in ("sandbox", "test")

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

# ----------------------------
# CLI Entrypoints
# ----------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-rl-agent")
    p.add_argument("--mode", choices=["prod", "sandbox", "test"], default=os.getenv("RL_AGENT_MODE", "prod"))
    p.add_argument("--model", default=str(DEFAULT_RL_MODEL))
    p.add_argument("--interval", type=float, default=float(os.getenv("RL_AGENT_LOOP_INTERVAL", "5.0")))
    p.add_argument("--cooldown", type=float, default=float(os.getenv("RL_AGENT_ACTION_COOLDOWN", "10.0")))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-hot-reload", action="store_true")
    p.add_argument("--prom-port", type=int, default=int(os.getenv("RL_AGENT_PROM_PORT", "9205")))
    p.add_argument("--start-api", action="store_true", help="Start lightweight FastAPI status server")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--one-shot", action="store_true", help="Run a single inference+emit and exit (sync)")
    return p

def main_cli():
    args = _build_cli().parse_args()
    cfg = RLAgentConfig(
        model_path=args.model,
        loop_interval=args.interval,
        cooldown=args.cooldown,
        dry_run=args.dry_run or os.getenv("RL_AGENT_DRY_RUN", "true").lower() == "true",
        hot_reload=not args.no_hot_reload,
        prom_port=args.prom_port,
        mode=args.mode,
    )
    agent = RLAgent(cfg)

    async def runner():
        # start api server optionally in thread
        api_thread = None
        if args.start_api and _HAS_FASTAPI:
            api = RLAgentAPI(agent, port=args.api_port)
            api_thread = threading.Thread(target=api.run, daemon=True)
            api_thread.start()
        # Run one-shot or continuous
        if args.one_shot:
            res = await agent.run_once()
            print(json.dumps(res, indent=2))
            return
        await agent.start()
        # wait until signal
        while True:
            await asyncio.sleep(3600)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # graceful signal handling to stop agent
    def _on_signal(sig, frame):
        LOG.info("Received signal %s, stopping...", sig)
        try:
            loop.create_task(agent.stop())
        except Exception:
            pass
        time.sleep(0.5)
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    try:
        loop.run_until_complete(runner())
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    finally:
        try:
            loop.run_until_complete(agent.stop())
        except Exception:
            pass

# ----------------------------
# If imported, expose helper factory
# ----------------------------
def create_agent_from_env() -> RLAgent:
    cfg = RLAgentConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        mongo_url=os.getenv("MONGO_URL"),
        model_path=os.getenv("RL_AGENT_MODEL_PATH", str(DEFAULT_RL_MODEL)),
        loop_interval=float(os.getenv("RL_AGENT_LOOP_INTERVAL", "5.0")),
        cooldown=float(os.getenv("RL_AGENT_ACTION_COOLDOWN", "10.0")),
        dry_run=os.getenv("RL_AGENT_DRY_RUN", "true").lower() == "true",
        hot_reload=os.getenv("RL_AGENT_HOT_RELOAD", "true").lower() == "true",
        mode=os.getenv("RL_AGENT_MODE", "prod"),
        prom_port=int(os.getenv("RL_AGENT_PROM_PORT", "9205")),
    )
    return RLAgent(cfg)

# ----------------------------
# Unit / sanity test helper (lightweight)
# ----------------------------
def _sanity_check():
    """Quick run to assert core behavior (non-destructive)."""
    cfg = RLAgentConfig(mode="sandbox", model_path=str(DEFAULT_RL_MODEL), hot_reload=False)
    agent = RLAgent(cfg)
    # load model if exists (no error if absent)
    agent.load_model()
    async def test_run():
        await agent.connect_redis()
        # one-shot run
        res = await agent.run_once()
        print("Sanity run result:", res)
        await agent.close_connectors()
    try:
        asyncio.run(test_run())
    except Exception:
        LOG.exception("Sanity check failed")

# ----------------------------
# Module entrypoint
# ----------------------------
if __name__ == "__main__":
    main_cli()

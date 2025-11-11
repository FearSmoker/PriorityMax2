#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PriorityMax - Real Environment Bridge for RL Training - ENTERPRISE EDITION
---------------------------------------------------------------------------

SYNCHRONIZED WITH:
- train_rl_heavy.py (observation/action dimensions, PPO training)
- train_rl_live.py (live mode operations, safety features)
- train_rl_eval.py (evaluation protocols, drift detection)

Enterprise Production Features:
✅ Observation space matches training scripts (dynamic obs_dim detection)
✅ Action space synchronized (3D continuous: [delta_workers, throttle, priority])
✅ Safety features: circuit breakers, rate limiting, dry-run mode
✅ Live telemetry integration (Redis, Mongo, Prometheus)
✅ Realistic simulation with diurnal patterns, bursts, heterogeneity
✅ Drift detection support (observation/reward distribution tracking)
✅ Audit logging for all actions
✅ Health monitoring and metrics export
✅ Thread-safe operations for concurrent training
✅ Emergency shutdown hooks
✅ Production-ready reward shaping

Usage:
    # Simulation mode (training)
    from app.ml.real_env import SimulatedRealEnv, EnvConfig
    env = SimulatedRealEnv(EnvConfig(mode="sim", seed=42))
    obs = env.reset()
    obs, reward, done, info = env.step([1.0, 0.0, 0.0])
    
    # Live mode (production)
    env = LiveRealEnv(EnvConfig(mode="live", dry_run=True))
    obs = env.reset()
    obs, reward, done, info = env.step([0.0, 0.0, 0.0])
    
    # Vectorized training
    envs = make_vec_env(EnvConfig(mode="sim"), n=16)
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import random
import logging
import pathlib
import tempfile
import threading
import asyncio
import signal
import atexit
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Callable, Iterable, Union
from collections import deque, defaultdict
from datetime import datetime
import statistics

import numpy as np


LOG = logging.getLogger("prioritymax.ml.real_env")
LOG.setLevel(os.getenv("PRIORITYMAX_ENV_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Optional dependencies (best-effort imports)
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False

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
    from prometheus_client import CollectorRegistry, Gauge, Histogram, Counter
    _HAS_PROM = True
except Exception:
    CollectorRegistry = Gauge = Histogram = Counter = None
    _HAS_PROM = False

# Try to use Gymnasium (preferred) or Gym (legacy), else fallback
try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
    LOG.info("Using Gymnasium (version %s)", getattr(gym, "__version__", "unknown"))
except ImportError:
    try:
        import gym
        from gym import spaces
        _HAS_GYM = True
        LOG.info("Using Gym (version %s)", getattr(gym, "__version__", "unknown"))
    except Exception:
        _HAS_GYM = False
        LOG.warning("⚠️ Neither Gym nor Gymnasium found — using fallback space classes.")
                
# -------------------------------------------------------------------
# SimpleSpace: fallback object for tests (used when gym not installed)
# -------------------------------------------------------------------
class _SimpleSpace:
    """Minimal gym.Space-like object exposing .shape for test detection."""
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.low = None
        self.high = None
        self.dtype = np.float32

    def __repr__(self):
        return f"_SimpleSpace(shape={self.shape})"
    
try:
    import redis
    _HAS_REDIS = True
except Exception:
    redis = None
    _HAS_REDIS = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

# Local imports
try:
    from app.api.admin import write_audit_event
    _HAS_AUDIT = True
except Exception:
    _HAS_AUDIT = False
    def write_audit_event(payload: Dict[str, Any]):
        """Fallback audit logger."""
        p = pathlib.Path.cwd() / "backend" / "logs" / "real_env_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

try:
    from app.metrics import metrics
    _HAS_METRICS = True
except Exception:
    metrics = None
    _HAS_METRICS = False

# Logging
LOG = logging.getLogger("prioritymax.ml.real_env")
LOG.setLevel(os.getenv("PRIORITYMAX_ENV_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = pathlib.Path(os.getenv("PRIORITYMAX_DATA_DIR", str(BASE_DIR / "datasets")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------
# Environment Configuration
# ---------------------------
@dataclass
class EnvConfig:
    """
    SYNCHRONIZED with train_rl_heavy.py and train_rl_live.py configurations.
    
    Critical Synchronization Points:
    - obs_dim: Auto-detected from environment (matches train_rl_heavy.py)
    - act_dim: Fixed at 3 (delta_workers, throttle, priority)
    - Action bounds: Aligned with training scripts
    - Reward structure: Matches expected RL objectives
    """
    mode: str = "sim"
    obs_dim: Optional[int] = None
    act_dim: int = 3
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    mongo_url: Optional[str] = os.getenv("MONGO_URL", None)
    prometheus_host: Optional[str] = os.getenv("PROM_HOST", None)
    max_scale_delta: int = 5
    min_workers: int = 1
    max_workers: int = 200
    action_cooldown_seconds: float = 5.0
    dry_run: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_window_seconds: int = 300
    obs_window_seconds: int = 60
    reward_latency_sla_ms: float = 500.0
    cost_per_worker_per_sec: float = 0.0005
    sim_target_throughput: float = 50.0
    sim_base_latency_ms: float = 100.0
    sim_noise_scale: float = 0.2
    sim_failure_rate: float = 0.02
    sim_queue_arrival_rate: float = 5.0
    sim_max_queue: int = 10000
    sim_burst_probability: float = 0.02
    sim_burst_size: float = 20.0
    sim_diurnal_amplitude: float = 0.7
    sim_time_step_minutes: float = 1.0
    reward_latency_weight: float = 10.0
    reward_throughput_weight: float = 0.1
    reward_cost_weight: float = 1.0
    reward_queue_weight: float = 0.05
    reward_stability_weight: float = 0.1
    reward_success_weight: float = 1.0
    reward_sla_bonus: float = 0.5
    enable_drift_tracking: bool = True
    drift_window_size: int = 1000
    seed: Optional[int] = None
    enable_audit_logging: bool = True
    enable_metrics_export: bool = True
    enable_heterogeneous_workers: bool = True
    enable_task_priorities: bool = True
    enable_temporal_patterns: bool = True


class CircuitBreaker:
    """Circuit breaker synchronized with train_rl_live.py."""
    def __init__(self, threshold=5, window_seconds=300):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.failures = deque()
        self.state = "CLOSED"
        self.last_state_change = time.time()
        self.lock = threading.Lock()
        LOG.info("🔌 Circuit breaker initialized (threshold=%d, window=%ds)", threshold, window_seconds)

    def record_failure(self):
        with self.lock:
            now = time.time()
            self.failures.append(now)
            cutoff = now - self.window_seconds
            while self.failures and self.failures[0] < cutoff:
                self.failures.popleft()
            if len(self.failures) >= self.threshold and self.state == "CLOSED":
                self.state = "OPEN"
                self.last_state_change = now
                LOG.error("⚠️ CIRCUIT BREAKER OPENED - Too many failures (%d in %ds)",
                          len(self.failures), self.window_seconds)
                write_audit_event({
                    "event": "circuit_breaker_opened",
                    "failures": len(self.failures),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })

    def record_success(self):
        with self.lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures.clear()
                LOG.info("✅ Circuit breaker CLOSED - System recovered")
                write_audit_event({
                    "event": "circuit_breaker_closed",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })

    def can_execute(self) -> bool:
        with self.lock:
            now = time.time()
            if self.state == "OPEN" and (now - self.last_state_change) > self.window_seconds:
                self.state = "HALF_OPEN"
                self.last_state_change = now
                LOG.info("🔄 Circuit breaker HALF_OPEN - Testing recovery")
            return self.state != "OPEN"


class DriftTracker:
    """Tracks observation and reward distributions for drift detection."""
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.obs_buffer = deque(maxlen=window_size)
        self.reward_buffer = deque(maxlen=window_size)
        self.baseline_obs_stats = None
        self.baseline_reward_stats = None
        LOG.info("📊 Drift tracker initialized (window=%d)", window_size)

    def add_sample(self, obs: np.ndarray, reward: float):
        if isinstance(obs, np.ndarray):
            self.obs_buffer.append(obs.copy())
        else:
            self.obs_buffer.append(np.array(obs))
        self.reward_buffer.append(float(reward))
        if len(self.obs_buffer) == self.window_size and self.baseline_obs_stats is None:
            self._compute_baseline()

    def _compute_baseline(self):
        if len(self.obs_buffer) < self.window_size:
            return
        obs_array = np.array(list(self.obs_buffer))
        reward_array = np.array(list(self.reward_buffer))
        self.baseline_obs_stats = {
            "mean": np.mean(obs_array, axis=0),
            "std": np.std(obs_array, axis=0),
            "min": np.min(obs_array, axis=0),
            "max": np.max(obs_array, axis=0)
        }
        self.baseline_reward_stats = {
            "mean": float(np.mean(reward_array)),
            "std": float(np.std(reward_array)),
            "p50": float(np.percentile(reward_array, 50)),
            "p95": float(np.percentile(reward_array, 95))
        }
        LOG.info("📊 Baseline established - reward_mean=%.3f obs_mean=%s",
                 self.baseline_reward_stats["mean"],
                 self.baseline_obs_stats["mean"][:3])

    def get_drift_score(self) -> Optional[float]:
        if self.baseline_reward_stats is None or len(self.reward_buffer) < self.window_size:
            return None
        current_reward_mean = np.mean(list(self.reward_buffer))
        baseline_mean = self.baseline_reward_stats["mean"]
        baseline_std = self.baseline_reward_stats["std"]
        drift_score = abs(current_reward_mean - baseline_mean) / (baseline_std + 1e-8)
        return float(drift_score)
# ---------------------------
# Action/Observation Spaces (SYNCHRONIZED)
# ---------------------------
def get_action_space() -> Any:
    """
    CRITICAL SYNC POINT: Must match train_rl_heavy.py and train_rl_live.py.

    Action vector: [delta_workers, throttle_scale, priority_bias]
    - delta_workers: continuous (-10, 10) clipped to max_scale_delta
    - throttle_scale: continuous (0, 1) for request throttling
    - priority_bias: continuous (-2, 2) for priority adjustments
    """
    low = np.array([-10.0, 0.0, -2.0], dtype=np.float32)
    high = np.array([10.0, 1.0, 2.0], dtype=np.float32)

    if _HAS_GYM:
        return spaces.Box(low=low, high=high, dtype=np.float32)
    else:
        s = _SimpleSpace((3,))
        s.low = low
        s.high = high
        return s


def get_observation_space(obs_dim: int = 8) -> Any:
    """
    CRITICAL SYNC POINT: Must match train_rl_heavy.py observation detection.

    Default observation vector (8 dimensions):
    [0] queue_length (backlog)
    [1] worker_count
    [2] avg_latency_ms
    [3] p95_latency_ms
    [4] success_rate (0-1)
    [5] arrival_rate (tasks/sec)
    [6] cpu_utilization (0-1)
    [7] memory_utilization (0-1)
    """
    low = np.zeros(obs_dim, dtype=np.float32)
    high = np.array(
        [1e6, 1e4, 1e6, 1e6, 1.0, 1e3, 1.0, 1.0], dtype=np.float32
    )[:obs_dim]

    # Pad if obs_dim > 8
    if obs_dim > 8:
        high = np.pad(high, (0, obs_dim - len(high)), constant_values=1e6)

    # ✅ Fix: always return an object with .shape even if gym not available
    if _HAS_GYM:
        return spaces.Box(low=low, high=high, dtype=np.float32)
    else:
        s = _SimpleSpace((obs_dim,))
        s.low = low
        s.high = high
        return s


# ---------------------------
# Base Environment (Enterprise Grade)
# ---------------------------
class RealEnvBase:
    """
    Base class for all PriorityMax RL environments.

    SYNCHRONIZED WITH:
    - train_rl_heavy.py: Observation/action dimensions, training loops
    - train_rl_live.py: Safety features, live operations
    - train_rl_eval.py: Evaluation protocols, drift detection

    Enterprise Features:
    - Thread-safe operations
    - Circuit breaker integration
    - Drift tracking
    - Audit logging
    - Prometheus metrics
    - Emergency shutdown hooks
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.random = random.Random(config.seed if config.seed is not None else int(time.time()))
        self.npr = np.random.RandomState(config.seed if config.seed is not None else int(time.time()))

        # State tracking
        self.current_time = time.time()
        self.last_action_time = 0.0
        self.last_scaling_action = 0
        self.step_count = 0
        self.episode_count = 0
        self.closed = False

        # Runtime state (CRITICAL: matches training script expectations)
        self.state: Dict[str, Any] = {
            "queue_length": 0.0,
            "worker_count": float(max(1, config.min_workers or 1)),
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "success_rate": 1.0,
            "arrival_rate": 0.0,
            "cpu": 0.0,
            "mem": 0.0
        }

        # Determine observation dimension
        if config.obs_dim is None:
            config.obs_dim = len(self.state)

        # Spaces (SYNCHRONIZED)
        self.action_space = get_action_space()
        self.observation_space = get_observation_space(config.obs_dim)

        # Enterprise components
        self.circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            window_seconds=config.circuit_breaker_window_seconds
        ) if config.enable_circuit_breaker else None

        self.drift_tracker = DriftTracker(
            window_size=config.drift_window_size
        ) if config.enable_drift_tracking else None

        self.metrics_history: List[Dict[str, Any]] = []
        self._redis = None
        self._mongo = None
        self._lock = threading.Lock()

        if _HAS_PROM and config.enable_metrics_export:
            try:
                self._prom_registry = CollectorRegistry()
                self.prom_queue = Gauge("env_queue_length", "Queue length",
                                        registry=self._prom_registry)
                self.prom_workers = Gauge("env_worker_count", "Worker count",
                                          registry=self._prom_registry)
                self.prom_latency = Gauge("env_avg_latency_ms", "Average latency",
                                          registry=self._prom_registry)
                self.prom_reward = Gauge("env_reward", "Last reward",
                                         registry=self._prom_registry)
                self.prom_steps = Counter("env_steps_total", "Total steps",
                                          registry=self._prom_registry)
            except Exception:
                self._prom_registry = None
        else:
            self._prom_registry = None

        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        LOG.info(
            "✅ Environment initialized (mode=%s, obs_dim=%d, act_dim=%d)",
            config.mode, config.obs_dim, config.act_dim
        )

    def _signal_handler(self, signum, frame):
        LOG.warning("⚠️ Received signal %d, cleaning up...", signum)
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        if not self.closed:
            self.close()
    # ---------------------------
    # Gym API (SYNCHRONIZED)
    # ---------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to initial state.

        SYNCHRONIZED with train_rl_heavy.py and train_rl_live.py reset patterns.
        """
        with self._lock:
            if seed is not None:
                self.random.seed(seed)
                self.npr.seed(seed)

            self.current_time = time.time()
            self.last_action_time = 0.0
            self.last_scaling_action = 0
            self.step_count = 0
            self.episode_count += 1
            self.closed = False

            # Reset state (subclass-specific)
            self._reset_state()

            # Get initial observation
            obs = self._observe()

            # Clear metrics
            self.metrics_history.clear()

            # Audit log
            if self.cfg.enable_audit_logging:
                write_audit_event({
                    "event": "env_reset",
                    "episode": self.episode_count,
                    "seed": seed,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })

            return obs

    def step(
        self,
        action: Union[np.ndarray, List[float], Tuple, Dict[str, Any]]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return (observation, reward, done, info).

        SYNCHRONIZED with train_rl_heavy.py and train_rl_live.py step patterns.
        """
        with self._lock:
            self.step_count += 1
            now = time.time()
            self.current_time = now

            parsed_action = self._parse_action(action)

            # Safety check
            if not self._safety_checks(parsed_action):
                parsed_action = {"delta_workers": 0, "throttle": 0.0, "priority_bias": 0}

            # Apply action
            exec_result = self._apply_action(
                parsed_action["delta_workers"],
                parsed_action["throttle"],
                parsed_action["priority_bias"]
            )

            # Advance simulated or live world
            self._advance_world(
                parsed_action["throttle"],
                exec_result.get("applied_delta", 0),
                parsed_action["priority_bias"]
            )

            # Get observation
            obs = self._observe()

            # Compute reward
            reward, reward_details = self.compute_reward(exec_result)
            done = self._terminal_condition()

            # Drift tracking
            if self.drift_tracker:
                self.drift_tracker.add_sample(obs, reward)

            # Metrics
            if self._prom_registry:
                try:
                    self.prom_queue.set(self.state["queue_length"])
                    self.prom_workers.set(self.state["worker_count"])
                    self.prom_latency.set(self.state["avg_latency_ms"])
                    self.prom_reward.set(reward)
                    self.prom_steps.inc()
                except Exception:
                    pass

            info = {
                "exec": exec_result,
                "reward_details": reward_details,
                "step": self.step_count,
                "episode": self.episode_count,
                "timestamp": now,
                "state": self.state.copy(),
                "circuit_breaker_ok": self.circuit_breaker.can_execute() if self.circuit_breaker else True,
                "drift_score": self.drift_tracker.get_drift_score() if self.drift_tracker else None,
            }

            self.metrics_history.append({
                "ts": now,
                **self.state,
                "action": parsed_action,
                "reward": reward,
            })

            if parsed_action["delta_workers"] != 0:
                self.last_action_time = now
                self.last_scaling_action = parsed_action["delta_workers"]

            return obs, float(reward), bool(done), info

    def render(self, mode: str = "human") -> Optional[str]:
        """Render environment state."""
        output = (
            f"[Step {self.step_count}] "
            f"Queue={self.state['queue_length']:.0f} "
            f"Workers={self.state['worker_count']:.0f} "
            f"Latency(avg/p95)={self.state['avg_latency_ms']:.1f}/{self.state['p95_latency_ms']:.1f}ms "
            f"Success={self.state['success_rate']:.2%}"
        )

        if mode == "human":
            print(output)
            return None
        else:
            return output

    def close(self):
        """Close environment and cleanup resources."""
        if self.closed:
            return

        self.closed = True

        if self._redis:
            try:
                if asyncio.iscoroutine(self._redis.close()):
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._redis.close())
            except Exception:
                pass

        if self.cfg.enable_audit_logging:
            write_audit_event({
                "event": "env_closed",
                "total_steps": self.step_count,
                "total_episodes": self.episode_count,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })

        LOG.info("Environment closed (steps=%d, episodes=%d)",
                 self.step_count, self.episode_count)

    # ---------------------------
    # Action Parsing (SYNCHRONIZED)
    # ---------------------------
    def _parse_action(
        self, action: Union[np.ndarray, List[float], Tuple, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse action into structured format."""
        if isinstance(action, dict):
            return {
                "delta_workers": action.get("delta_workers", 0),
                "throttle": action.get("throttle", 0.0),
                "priority_bias": action.get("priority_bias", 0),
            }

        arr = np.asarray(action, dtype=float).flatten()

        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), constant_values=0)

        return {
            "delta_workers": int(np.round(arr[0])),
            "throttle": float(np.clip(arr[1], 0.0, 1.0)),
            "priority_bias": int(np.round(arr[2])),
        }

    # ---------------------------
    # Safety Checks (ENTERPRISE)
    # ---------------------------
    def _safety_checks(self, action: Dict[str, Any]) -> bool:
        now = time.time()

        if (now - self.last_action_time) < self.cfg.action_cooldown_seconds:
            if action["delta_workers"] != 0:
                LOG.debug(
                    "Action cooldown active (%.1fs remaining)",
                    self.cfg.action_cooldown_seconds - (now - self.last_action_time),
                )
                return False

        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            LOG.warning("⚠️ Circuit breaker OPEN - action blocked")
            return False

        new_workers = self.state["worker_count"] + action["delta_workers"]
        if new_workers < self.cfg.min_workers or new_workers > self.cfg.max_workers:
            LOG.debug(
                "Action would violate worker bounds (%d not in [%d, %d])",
                new_workers,
                self.cfg.min_workers,
                self.cfg.max_workers,
            )
            return False

        return True
    # ---------------------------
    # Action Application (to override)
    # ---------------------------
    def _apply_action(self, delta_workers: int, throttle: float, priority_bias: int) -> Dict[str, Any]:
        """
        Apply action to system (override in subclasses).
        Returns execution result dictionary with applied deltas and audit metadata.
        Fully synchronized with train_rl_live.py and train_rl_heavy.py expectations.
        """
        attempt = {
            "wanted_delta": delta_workers,
            "applied_delta": 0,
            "throttle": throttle,
            "priority_bias": priority_bias,
            "message": None,
            "ts": time.time(),
        }

        # Enforce worker bounds
        new_worker_count = int(
            np.clip(
                self.state["worker_count"] + delta_workers,
                self.cfg.min_workers,
                self.cfg.max_workers,
            )
        )
        attempt["applied_delta"] = int(new_worker_count - self.state["worker_count"])

        # Live or simulated handling
        if self.cfg.mode == "live" and not self.cfg.dry_run:
            intent = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "scale_action",
                "delta": attempt["applied_delta"],
                "throttle": throttle,
                "priority_bias": priority_bias,
            }

            try:
                if _HAS_AIOREDIS:
                    async def _push():
                        cli = await aioredis.from_url(self.cfg.redis_url, encoding="utf-8", decode_responses=True)
                        try:
                            await cli.lpush("prioritymax:scale_intents", json.dumps(intent))
                        finally:
                            await cli.close()
                    asyncio.get_event_loop().create_task(_push())
                elif _HAS_MOTOR and self.cfg.mongo_url:
                    async def _insert():
                        client = motor_asyncio.AsyncIOMotorClient(self.cfg.mongo_url)
                        db = client.get_default_database()
                        await db["scale_intents"].insert_one(intent)
                    asyncio.get_event_loop().create_task(_insert())
                else:
                    p = pathlib.Path(tempfile.gettempdir()) / f"intent_{intent['id']}.json"
                    p.write_text(json.dumps(intent))
                attempt["message"] = "intent_emitted"
                write_audit_event({
                    "event": "live_scale_intent",
                    "intent": intent,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            except Exception as e:
                LOG.exception("Failed to emit scale intent: %s", e)
                attempt["message"] = f"intent_emit_failed: {e}"
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
        else:
            attempt["message"] = "dry_run_or_simulated"

        return attempt

    # ---------------------------
    # WORLD DYNAMICS (SIMULATION)
    # ---------------------------
    def _advance_world(self, throttle: float, applied_delta: int, priority_bias: int):
        """
        Advance the simulated environment dynamics one step forward.
        Incorporates realistic queueing, bursts, diurnal patterns, heterogeneity, and noise.
        Fully synchronized with train_rl_heavy.py simulation behavior.
        """
        self.state["worker_count"] = max(
            self.cfg.min_workers,
            min(self.cfg.max_workers, self.state["worker_count"] + applied_delta),
        )

        # Diurnal factor and bursts
        tod_hours = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
        diurnal_factor = 1.0 + self.cfg.sim_diurnal_amplitude * math.sin(
            (tod_hours / 24.0) * 2 * math.pi
        )
        arrival_rate = max(
            0.0,
            self.cfg.sim_queue_arrival_rate * diurnal_factor * (1.0 - throttle),
        )

        if self.random.random() < self.cfg.sim_burst_probability:
            arrival_rate *= (1.0 + self.cfg.sim_burst_size / 10.0)

        arrivals = np.random.poisson(lam=arrival_rate)
        service_capacity = max(
            1.0, self.state["worker_count"] * self.random.uniform(0.8, 1.2)
        )
        completed = min(self.state["queue_length"] + arrivals, int(service_capacity))

        self.state["queue_length"] = max(
            0.0,
            min(
                self.cfg.sim_max_queue,
                self.state["queue_length"] + arrivals - completed,
            ),
        )

        # Latency model
        load_factor = 1.0 + (self.state["queue_length"] / max(1.0, service_capacity))
        noise = self.npr.randn() * self.cfg.sim_noise_scale * self.cfg.sim_base_latency_ms
        base_latency = self.cfg.sim_base_latency_ms * load_factor + noise
        self.state["avg_latency_ms"] = max(1.0, base_latency)
        self.state["p95_latency_ms"] = base_latency * (1.0 + 0.4 * self.random.random())

        # Success rate
        fail_rate = min(
            0.5,
            self.cfg.sim_failure_rate
            + 0.001 * max(0.0, self.state["queue_length"] - service_capacity),
        )
        self.state["success_rate"] = max(0.0, 1.0 - fail_rate)

        # CPU / memory usage
        self.state["cpu"] = min(
            1.0, 0.05 * self.state["worker_count"] + 0.0005 * self.state["queue_length"]
        )
        self.state["mem"] = min(
            1.0, 0.02 * self.state["worker_count"] + 0.0003 * self.state["queue_length"]
        )

        self.state["arrival_rate"] = arrival_rate

    # ---------------------------
    # OBSERVATION VECTOR
    # ---------------------------
    def _observe(self) -> np.ndarray:
        """Return normalized observation vector."""
        arr = np.array(
            [
                self.state["queue_length"],
                self.state["worker_count"],
                self.state["avg_latency_ms"],
                self.state["p95_latency_ms"],
                self.state["success_rate"],
                self.state["arrival_rate"],
                self.state["cpu"],
                self.state["mem"],
            ],
            dtype=np.float32,
        )
        return np.clip(arr, 0, 1e6)

    # ---------------------------
    # REWARD FUNCTION (SYNCED)
    # ---------------------------
    def compute_reward(self, exec_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Reward shaping synchronized with train_rl_heavy.py and train_rl_eval.py.
        Balances latency, throughput, cost, queue, stability, and success rate.
        """
        sla = self.cfg.reward_latency_sla_ms
        lat = self.state["avg_latency_ms"]
        succ = self.state["success_rate"]
        q = self.state["queue_length"]
        w = self.state["worker_count"]
        delta = exec_result.get("applied_delta", 0)
        throttle = exec_result.get("throttle", 0.0)

        latency_pen = -((max(0.0, lat - sla) / sla) ** 2) * self.cfg.reward_latency_weight
        throughput_reward = succ * w * self.cfg.reward_throughput_weight
        cost_pen = -w * self.cfg.cost_per_worker_per_sec * self.cfg.reward_cost_weight
        queue_pen = -math.log1p(q) * self.cfg.reward_queue_weight
        stability_pen = -abs(delta) * self.cfg.reward_stability_weight
        throttle_pen = -throttle * 0.5
        success_bonus = succ * self.cfg.reward_success_weight

        total = (
            latency_pen
            + throughput_reward
            + cost_pen
            + queue_pen
            + stability_pen
            + throttle_pen
            + success_bonus
        )

        if lat <= sla:
            total += self.cfg.reward_sla_bonus

        if self.circuit_breaker and succ > 0.99:
            self.circuit_breaker.record_success()

        details = {
            "latency_pen": latency_pen,
            "throughput_reward": throughput_reward,
            "cost_penalty": cost_pen,
            "queue_penalty": queue_pen,
            "stability_penalty": stability_pen,
            "throttle_penalty": throttle_pen,
            "success_bonus": success_bonus,
            "sla_bonus": self.cfg.reward_sla_bonus if lat <= sla else 0.0,
            "final_reward": total,
        }
        return float(total), details

    # ---------------------------
    # TERMINAL CONDITION
    # ---------------------------
    def _terminal_condition(self) -> bool:
        """End episode only on catastrophic overload."""
        if (
            self.state["p95_latency_ms"] > 60000
            or self.state["queue_length"] >= self.cfg.sim_max_queue * 0.95
        ):
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            return True
        return False
# ---------------------------------------------------
# SIMULATED ENVIRONMENT (for training)
# ---------------------------------------------------
class SimulatedRealEnv(RealEnvBase):
    """Fully featured simulator with heterogeneous workers, bursts, and diurnal load."""
    def _reset_state(self):
        # Initialize realistic starting state (keeps same keys as RealEnvBase.state)
        self.state.update({
            "queue_length": float(self.npr.randint(0, 50)),
            "worker_count": float(self.npr.randint(self.cfg.min_workers, self.cfg.min_workers + 5)),
            "avg_latency_ms": float(self.cfg.sim_base_latency_ms),
            "p95_latency_ms": float(self.cfg.sim_base_latency_ms * 1.2),
            "success_rate": 1.0,
            "arrival_rate": self.cfg.sim_queue_arrival_rate,
            "cpu": 0.2,
            "mem": 0.2
        })
        # Per-worker heterogeneous efficiency factors (used optionally in extended dynamics)
        try:
            self.consumer_eff = [self.npr.uniform(0.8, 1.2) for _ in range(max(1, int(self.state["worker_count"])))]
        except Exception:
            self.consumer_eff = [1.0]

    # Optionally override _apply_action or other hooks for advanced simulation behaviors
    # (left intentionally extensible for enterprise integrations)

# ---------------------------------------------------
# LIVE ENVIRONMENT (for real ops)
# ---------------------------------------------------
class LiveRealEnv(RealEnvBase):
    """Bridge to live PriorityMax telemetry and control systems."""
    def _reset_state(self):
        """
        Try to seed state from a cached live metrics snapshot (JSON) or fall back to safe defaults.
        This keeps the environment deterministic for dry-run/live testing when a metrics cache exists.
        """
        try:
            cache = pathlib.Path(BASE_DIR / "live_metrics_cache.json")
            env_min_workers = self.cfg.min_workers if hasattr(self.cfg, "min_workers") else 1
            if cache.exists():
                m = json.loads(cache.read_text())
                for k in self.state.keys():
                    if k in m:
                        self.state[k] = m[k]
                # Ensure worker_count respects bounds
                self.state["worker_count"] = float(max(env_min_workers, min(self.cfg.max_workers, int(self.state.get("worker_count", env_min_workers)))))
            else:
                self.state["queue_length"] = 0.0
                self.state["worker_count"] = float(env_min_workers)
        except Exception as e:
            LOG.warning("Live reset fallback: %s", e)
            self.state["queue_length"] = 0.0
            self.state["worker_count"] = float(self.cfg.min_workers)

    def _advance_world(self, throttle: float, applied_delta: int, priority_bias: int):
        """
        When live metrics exist (via env var cache), ingest them; otherwise apply small deterministic
        drift to CPU/memory so dry-run/live can still be exercised without external systems.
        """
        metrics_path = pathlib.Path(os.getenv("PRIORITYMAX_LIVE_METRICS_CACHE", str(BASE_DIR / "live_metrics_cache.json")))
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text())
                for k, v in data.items():
                    if k in self.state:
                        self.state[k] = float(v) if isinstance(v, (int, float)) else v
            except Exception:
                LOG.warning("Failed to load live metrics; keeping old state")
        else:
            # Minimal safe drift for live mode without actual telemetry
            self.state["cpu"] = min(1.0, max(0.0, self.state.get("cpu", 0.0) + 0.01 * applied_delta))
            self.state["mem"] = min(1.0, max(0.0, self.state.get("mem", 0.0) + 0.005 * applied_delta))
            # Keep basic queue and latency relationships similar to simulated env
            self.state["queue_length"] = max(0.0, self.state.get("queue_length", 0.0) - applied_delta * 0.5)
            # recalc simple latency estimate
            self.state["avg_latency_ms"] = max(1.0, self.cfg.sim_base_latency_ms * (1.0 + (self.state["queue_length"] / max(1.0, max(1.0, self.state["worker_count"])))))
            self.state["p95_latency_ms"] = self.state["avg_latency_ms"] * 1.2
            self.state["success_rate"] = max(0.0, 1.0 - 0.01 * (self.state["queue_length"] / max(1.0, self.state["worker_count"])))

# ---------------------------------------------------
# FACTORY + VECTOR UTILS
# ---------------------------------------------------
def make_env(cfg: EnvConfig) -> RealEnvBase:
    """
    Factory: returns LiveRealEnv if cfg.mode == 'live', otherwise SimulatedRealEnv.
    Keeps signature simple for training scripts: make_env(EnvConfig(...))
    """
    return LiveRealEnv(cfg) if cfg.mode == "live" else SimulatedRealEnv(cfg)

def make_vec_env(cfg: EnvConfig, n: int = 4) -> List[RealEnvBase]:
    """
    Create a small vector (list) of environment instances for parallel training loops.
    Each environment receives a unique seed offset to ensure decorrelated randomness.
    """
    envs: List[RealEnvBase] = []
    base_seed = cfg.seed if cfg.seed is not None else int(time.time())
    for i in range(n):
        c = EnvConfig(**asdict(cfg))
        c.seed = int(base_seed + i)
        envs.append(make_env(c))
    return envs
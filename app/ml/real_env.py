# backend/app/ml/real_env.py
"""
PriorityMax - Real Environment Bridge for RL training

Provides:
 - RealEnv: a Gym-like environment that interfaces with live PriorityMax subsystem
   (Redis queue, Mongo logs, Prometheus metrics) to provide observations and accept actions.
 - SimulatedRealEnv: a feature-complete simulator that mimics realistic queue & worker behavior
 - Reward shaping: latency, cost, success-rate, SLA penalties, smoothness penalty
 - Safety & rate-limiting: destructive ops are dry-run by default
 - Vectorized env creation & rollout helpers for batch RL training
 - CLI for quick testing locally

Notes:
 - Optional libs (aioredis, motor, prometheus_client, gym) are used only when installed.
 - In production, a privileged controller or Kubernetes operator should execute any real "scale" or "kill" actions.
 - This module focuses on modeling environment dynamics and exposing a clean RL interface.
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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable, Iterable, Union

import numpy as np
import pandas as pd

# Optional / best-effort imports
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
    from prometheus_client import CollectorRegistry, Gauge, Histogram
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

try:
    import gym
    from gym import spaces
    _HAS_GYM = True
except Exception:
    # Minimal fallback for spaces
    _HAS_GYM = False
    class spaces:
        class Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.low = low; self.high = high; self.shape = shape; self.dtype = dtype
        class Discrete:
            def __init__(self, n): self.n = n

# Local audit hook
try:
    from app.api.admin import write_audit_event
    _HAS_AUDIT = True
except Exception:
    _HAS_AUDIT = False
    def write_audit_event(payload: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "real_env_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

LOG = logging.getLogger("prioritymax.ml.real_env")
LOG.setLevel(os.getenv("PRIORITYMAX_ENV_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Base paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = pathlib.Path(os.getenv("PRIORITYMAX_DATA_DIR", str(BASE_DIR / "datasets")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Environment configuration dataclass
# -----------------------
@dataclass
class EnvConfig:
    # mode
    mode: str = "sim"  # 'sim' or 'live'
    # connectors
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    mongo_url: Optional[str] = os.getenv("MONGO_URL", None)
    prometheus_host: Optional[str] = os.getenv("PROM_HOST", None)
    # rate-limits & safety
    max_scale_delta: int = 5               # max consumers change per step
    min_consumers: int = 0
    max_consumers: int = 200
    action_cooldown_seconds: int = 5      # enforce cooldown between scaling actions
    dry_run: bool = True                   # don't perform destructive ops in live mode by default
    # observation / reward windows
    obs_window_seconds: int = 60
    reward_latency_sla_ms: int = 500
    cost_per_consumer_per_sec: float = 0.0005  # monetary proxy
    # simulation params (only used in sim mode)
    sim_target_throughput: float = 50.0
    sim_base_latency_ms: float = 100
    sim_noise_scale: float = 0.2
    sim_failure_rate: float = 0.02
    sim_queue_arrival_rate: float = 5.0  # avg tasks per second
    sim_max_queue: int = 1000
    seed: Optional[int] = None

# -----------------------
# Helpers: Observation / Action specs
# -----------------------
def default_action_space() -> Any:
    """
    Action structure (continuous vector) that the RL agent outputs:
      - delta_consumers: integer delta to adjust number of consumers (can be negative)
      - throttle_scale: continuous [0,1] fraction to throttle new tasks (0=no throttle, 1=full throttle)
      - priority_boost: integer to increase priority bias (discrete small int)
    We'll encode as a continuous vector and clip later.
    """
    # action vector: [delta_consumers (continuous), throttle_scale (0-1), priority_bias (-2..+2)]
    low = np.array([-10.0, 0.0, -2.0], dtype=np.float32)
    high = np.array([10.0, 1.0, 2.0], dtype=np.float32)
    if _HAS_GYM:
        return spaces.Box(low=low, high=high, dtype=np.float32)
    else:
        return {"low": low, "high": high}

def default_observation_space(n_features: int = 8) -> Any:
    """
    Observation vector includes:
     - queue_length (scalar)
     - consumer_count (scalar)
     - avg_latency_ms (scalar)
     - p95_latency_ms (scalar)
     - success_rate (scalar)
     - recent_arrival_rate (scalar)
     - rolling_cpu (scalar)
     - rolling_mem (scalar)
    """
    low = np.array([0.0] * n_features, dtype=np.float32)
    high = np.array([1e6, 1e4, 1e6, 1e6, 1.0, 1e3, 1.0, 1.0], dtype=np.float32)[:n_features]
    if _HAS_GYM:
        return spaces.Box(low=low, high=high, dtype=np.float32)
    else:
        return {"shape": (n_features,), "dtype": np.float32}

# -----------------------
# Base Environment (Gym-like)
# -----------------------
class RealEnvBase:
    """
    Base class for RealEnv and SimulatedRealEnv.

    Provides:
     - step(), reset(), render()
     - action clipping & safety
     - observation construction
     - reward shaping interface (override compute_reward)
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.random = random.Random(config.seed if config.seed is not None else int(time.time()))
        # state variables
        self.current_time = time.time()
        self.last_action_time = 0.0
        self.last_scaling_action = 0
        self.step_count = 0
        self.closed = False

        # runtime state shape
        self.state: Dict[str, Any] = {
            "queue_length": 0,
            "consumer_count": max(1, config.min_consumers or 1),
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "success_rate": 1.0,
            "arrival_rate": 0.0,
            "cpu": 0.0,
            "mem": 0.0
        }

        # observation & action spaces
        self.action_space = default_action_space()
        self.observation_space = default_observation_space(n_features=len(self.state))

        # metrics (for logging/traces)
        self.metrics_history: List[Dict[str, Any]] = []

        # connectors (lazy)
        self._redis = None
        self._mongo = None

        # lock for thread-safety
        self._lock = threading.Lock()

        # Prometheus metrics (best-effort)
        if _HAS_PROM:
            try:
                self._prom_registry = CollectorRegistry()
                self.prom_queue = Gauge("prioritymax_env_queue_length", "Queue length", registry=self._prom_registry)
                self.prom_consumers = Gauge("prioritymax_env_consumers", "Consumer count", registry=self._prom_registry)
            except Exception:
                self._prom_registry = None

    # -----------------------
    # Connectors
    # -----------------------
    async def connect_redis(self):
        if not _HAS_AIOREDIS:
            raise RuntimeError("aioredis not installed")
        if self._redis:
            return self._redis
        self._redis = await aioredis.from_url(self.cfg.redis_url, encoding="utf-8", decode_responses=True)
        return self._redis

    async def connect_mongo(self):
        if not _HAS_MOTOR:
            raise RuntimeError("motor not installed")
        if self._mongo:
            return self._mongo
        self._mongo = motor_asyncio.AsyncIOMotorClient(self.cfg.mongo_url)
        return self._mongo

    # -----------------------
    # Gym API
    # -----------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment state. Returns initial observation.
        """
        with self._lock:
            if seed is not None:
                self.random.seed(seed)
            self.current_time = time.time()
            self.last_action_time = 0.0
            self.last_scaling_action = 0
            self.step_count = 0
            self.closed = False
            # reset synthetic state (simulator overrides)
            self._reset_state()
            obs = self._observe()
            self.metrics_history.clear()
            return obs

    def step(self, action: Union[np.ndarray, List[float], Tuple[float, float, float], Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply action and advance environment by one timestep.

        Returns: obs, reward, done, info
        """
        with self._lock:
            self.step_count += 1
            now = time.time()
            self.current_time = now

            # parse action vector into structured actions
            act = self._parse_action(action)

            # enforce cooldown and safety
            if (now - self.last_action_time) < self.cfg.action_cooldown_seconds:
                # scale action down to zero if too soon (penalty will apply)
                if abs(act["delta_consumers"]) > 0:
                    LOG.debug("Action cooldown active: suppressing scaling action")
                    act["delta_consumers"] = 0

            # clip action to allowed max delta
            delta = int(np.clip(act["delta_consumers"], -self.cfg.max_scale_delta, self.cfg.max_scale_delta))
            throttle = float(np.clip(act["throttle_scale"], 0.0, 1.0))
            priority_bias = int(np.clip(round(act["priority_bias"]), -2, 2))

            # execute action (sim or live; if live & dry_run then only simulate effect)
            exec_result = self._apply_action(delta, throttle, priority_bias)

            # step world forward by one "time unit" (env tick)
            self._advance_world(throttle, delta, priority_bias)

            # compute observation & reward
            obs = self._observe()
            reward, reward_details = self.compute_reward(exec_result)

            # done condition (rare): catastrophes or max steps
            done = self._terminal_condition()

            # record metrics
            info = {"exec": exec_result, "reward_details": reward_details, "step": self.step_count}

            # update last action time if a scaling was performed
            if delta != 0:
                self.last_action_time = now
                self.last_scaling_action = delta

            # append to metrics history
            self.metrics_history.append({"ts": now, **self.state, "action": {"delta": delta, "throttle": throttle, "priority_bias": priority_bias}, "reward": reward})

            return obs, float(reward), bool(done), info

    def render(self, mode: str = "human"):
        s = f"t={self.step_count} queue={self.state['queue_length']} consumers={self.state['consumer_count']} avg_lat={self.state['avg_latency_ms']:.1f} p95={self.state['p95_latency_ms']:.1f} success={self.state['success_rate']:.2f}"
        if mode == "human":
            print(s)
        else:
            return s

    def close(self):
        self.closed = True

    # -----------------------
    # Parsing & Safety
    # -----------------------
    def _parse_action(self, action: Union[np.ndarray, List[float], Tuple[float, float, float], Dict[str, Any]]) -> Dict[str, Any]:
        # accept dict or array-like
        if isinstance(action, dict):
            return {"delta_consumers": action.get("delta_consumers", 0), "throttle_scale": action.get("throttle_scale", 0.0), "priority_bias": action.get("priority_bias", 0)}
        arr = np.asarray(action, dtype=float)
        if arr.ndim == 0:
            arr = np.asarray([arr])
        # ensure length 3 by padding
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), 'constant')
        return {"delta_consumers": int(round(arr[0])), "throttle_scale": float(arr[1]), "priority_bias": int(round(arr[2]))}

    def _apply_action(self, delta_consumers: int, throttle: float, priority_bias: int) -> Dict[str, Any]:
        """
        Apply action to live system or simulate.
        Returns a dict describing what was attempted and what succeeded.
        """
        attempt = {"wanted_delta": delta_consumers, "throttle": throttle, "priority_bias": priority_bias, "applied_delta": 0, "message": None}
        # Safety: clip final consumer count within min/max
        new_consumers = int(self.state["consumer_count"] + delta_consumers)
        new_consumers = max(self.cfg.min_consumers, min(self.cfg.max_consumers, new_consumers))

        applied_delta = new_consumers - int(self.state["consumer_count"])
        attempt["applied_delta"] = applied_delta

        if self.cfg.mode == "live" and not self.cfg.dry_run:
            # Here we would call the real autoscaler / k8s API / consumer supervisor to scale
            # This part is intentionally conservative: we do not perform destructive operations in this library.
            # Instead, external privileged controller should subscribe to events or poll a command queue.
            attempt["message"] = "live_apply: not implemented in-library (use operator)"
            # Write audit for external operator
            write_audit_event({"event": "env_scale_intent", "details": {"attempt": attempt, "ts": time.time()}})
        else:
            # In sim / dry_run mode just update local state (the change will be enacted in _advance_world)
            attempt["message"] = "dry_run_or_simulated"

        # Return what was attempted; actual state mutation is done by _advance_world for simulation fidelity
        return attempt

    # -----------------------
    # World dynamics (to be overridden / extended)
    # -----------------------
    def _reset_state(self):
        """Initialize state - default is a small warm queue and 1 consumer."""
        self.state["queue_length"] = max(0, int(self.random.gauss(10, 5)))
        self.state["consumer_count"] = max(1, int(self.random.randint(1, max(1, self.cfg.min_consumers or 1))))
        base_latency = max(1.0, float(self.cfg.sim_base_latency_ms or 100.0))
        self.state["avg_latency_ms"] = base_latency * (1.0 + self.random.random() * 0.1)
        self.state["p95_latency_ms"] = self.state["avg_latency_ms"] * 1.5
        self.state["success_rate"] = max(0.9, 1.0 - float(self.cfg.sim_failure_rate or 0.01))
        self.state["arrival_rate"] = float(self.cfg.sim_queue_arrival_rate or 1.0)
        self.state["cpu"] = 0.1
        self.state["mem"] = 0.1

    def _advance_world(self, throttle: float, applied_delta_consumers: int, priority_bias: int):
        """
        Advance the environment by one tick using simple queueing dynamics.
        This default implementation is used by SimulatedRealEnv; real env may override to use live metrics.
        """
        # scale consumers (this simulates controller effect applied)
        self.state["consumer_count"] = max(self.cfg.min_consumers, min(self.cfg.max_consumers, int(self.state["consumer_count"] + applied_delta_consumers)))

        # arrival process (Poisson)
        lam = max(0.0, self.cfg.sim_queue_arrival_rate * (1.0 - throttle))
        arrivals = np.random.poisson(lam=lam)
        # work done proportional to consumers
        service_capacity = max(0.0, self.state["consumer_count"] * 1.0)  # tasks per tick per consumer (unit)
        completed = int(min(self.state["queue_length"] + arrivals, int(service_capacity)))
        # update queue
        new_queue = max(0, (self.state["queue_length"] + arrivals) - completed)
        self.state["queue_length"] = min(self.cfg.sim_max_queue, new_queue)

        # latency model: inversely proportional to consumers / service capacity and queue length
        base = float(self.cfg.sim_base_latency_ms or 100.0)
        load_factor = 1.0 + (self.state["queue_length"] / max(1.0, max(1.0, service_capacity)))
        noise = (np.random.randn() * self.cfg.sim_noise_scale * base) if self.cfg.sim_noise_scale > 0 else 0.0
        avg_lat = base * load_factor + noise
        avg_lat = max(1.0, float(avg_lat))
        self.state["avg_latency_ms"] = avg_lat
        self.state["p95_latency_ms"] = avg_lat * (1.0 + 0.5 * self.random.random())

        # success/failure: more failures under heavier load
        failure_rate = min(0.5, float(self.cfg.sim_failure_rate) + 0.001 * max(0, self.state["queue_length"] - self.state["consumer_count"]))
        self.state["success_rate"] = max(0.0, 1.0 - failure_rate)

        # resource usage: cpu scales with consumer_count and queue_length
        self.state["cpu"] = min(1.0, 0.05 * self.state["consumer_count"] + 0.0005 * self.state["queue_length"])
        self.state["mem"] = min(1.0, 0.02 * self.state["consumer_count"] + 0.0002 * self.state["queue_length"])

    def _observe(self) -> np.ndarray:
        """
        Construct observation vector and normalize values to reasonable ranges.
        Return a numpy array matching observation_space.
        """
        arr = np.array([
            float(self.state["queue_length"]),
            float(self.state["consumer_count"]),
            float(self.state["avg_latency_ms"]),
            float(self.state["p95_latency_ms"]),
            float(self.state["success_rate"]),
            float(self.state["arrival_rate"]),
            float(self.state["cpu"]),
            float(self.state["mem"]),
        ], dtype=np.float32)
        # normalization: clip large values for numerical stability
        arr[0] = np.clip(arr[0], 0, 1e6)
        arr[1] = np.clip(arr[1], 0, 1e5)
        arr[2] = np.clip(arr[2], 0, 1e6)
        arr[3] = np.clip(arr[3], 0, 1e6)
        arr[4] = np.clip(arr[4], 0.0, 1.0)
        arr[5] = np.clip(arr[5], 0.0, 1e3)
        arr[6] = np.clip(arr[6], 0.0, 1.0)
        arr[7] = np.clip(arr[7], 0.0, 1.0)
        return arr

    # -----------------------
    # Reward shaping
    # -----------------------
    def compute_reward(self, exec_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward based on current state and last executed action.

        Components:
         - latency_penalty: negative if avg_latency_ms exceeds SLA
         - throughput_reward: positive for successful task completions (proxied by success_rate & consumer_count)
         - cost_penalty: small negative for active consumers (encourage fewer consumers)
         - stability_penalty: penalty for large scaling actions / flapping
         - throttle_penalty: penalty for heavy throttling (discourage blanket throttles)
        """
        sla = float(self.cfg.reward_latency_sla_ms or 500.0)
        avg_lat = float(self.state["avg_latency_ms"])
        p95 = float(self.state["p95_latency_ms"])
        success = float(self.state["success_rate"])
        consumers = int(self.state["consumer_count"])
        queue_len = int(self.state["queue_length"])

        # latency penalty grows quadratically past SLA
        latency_pen = -((max(0.0, avg_lat - sla) / sla) ** 2) * 10.0

        # throughput proxy: consumers * success rate (encourage throughput)
        throughput_reward = (consumers * success) * 0.1

        # cost per consumer
        cost_penalty = -consumers * float(self.cfg.cost_per_consumer_per_sec or 0.0005)

        # queue penalty (if queue grows, negative)
        queue_penalty = -math.log1p(queue_len) * 0.05

        # stability penalty: large scaling actions should be penalized
        stability_pen = -abs(exec_result.get("applied_delta", 0)) * 0.1

        # throttle penalty (encourage low throttling)
        throttle_pen = -float(exec_result.get("throttle", 0.0)) * 0.5

        # success reward (scale)
        success_reward = success * 1.0

        reward = latency_pen + throughput_reward + cost_penalty + queue_penalty + stability_pen + throttle_pen + success_reward

        # small bonus for meeting SLA
        if avg_lat <= sla:
            reward += 0.5

        details = {
            "latency_pen": latency_pen,
            "throughput_reward": throughput_reward,
            "cost_penalty": cost_penalty,
            "queue_penalty": queue_penalty,
            "stability_penalty": stability_pen,
            "throttle_penalty": throttle_pen,
            "success_reward": success_reward,
            "final_reward": reward,
            "avg_lat_ms": avg_lat,
            "p95_lat_ms": p95,
            "queue_len": queue_len,
            "consumers": consumers
        }
        return reward, details

    # -----------------------
    # Terminal / safety conditions
    # -----------------------
    def _terminal_condition(self) -> bool:
        # default: long-running non-episodic environment; allow optional termination if catastrophic
        # term if p95 latency exceeds some huge threshold or queue overflow
        if self.state["p95_latency_ms"] > 60_000:
            return True
        if self.state["queue_length"] >= self.cfg.sim_max_queue:
            # don't terminate immediately; training may learn to recover. Return False.
            return False
        # optionally limit steps per episode if desired (not by default)
        return False

# -----------------------
# SimulatedRealEnv: realistic simulator used for training
# -----------------------
class SimulatedRealEnv(RealEnvBase):
    """
    Extends RealEnvBase with richer simulation dynamics:
     - tasks have types & priorities
     - consumers have efficiency variances
     - background burst events & temporal patterns (diurnal)
     - integrates with ReplayBuffer or logging
    """

    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.task_types = ["io", "cpu", "db"]
        self.time_of_day = 0.0
        self.diurnal_amplitude = 0.7
        # consumer efficiency distribution
        self.consumer_efficiencies: List[float] = [1.0 for _ in range(max(1, int(self.state["consumer_count"])))]
        # internal RNG for numpy
        self.npr = np.random.RandomState(config.seed or int(time.time()))

    def _reset_state(self):
        super()._reset_state()
        self.time_of_day = self.npr.rand() * 24.0
        self.consumer_efficiencies = [float(max(0.5, min(1.5, self.npr.normal(loc=1.0, scale=0.1)))) for _ in range(max(1, int(self.state["consumer_count"])))]
        self.state["task_composition"] = {t: self.npr.rand() for t in self.task_types}
        total = sum(self.state["task_composition"].values())
        for k in self.state["task_composition"]:
            self.state["task_composition"][k] /= total

    def _advance_world(self, throttle: float, applied_delta_consumers: int, priority_bias: int):
        # update consumers list
        old_consumers = int(self.state["consumer_count"])
        new_consumers = max(1, old_consumers + applied_delta_consumers)
        if new_consumers > len(self.consumer_efficiencies):
            # add new consumers with baseline efficiency
            while len(self.consumer_efficiencies) < new_consumers:
                self.consumer_efficiencies.append(float(max(0.6, min(1.4, self.npr.normal(loc=1.0, scale=0.15)))))
        elif new_consumers < len(self.consumer_efficiencies):
            # remove consumers from the end
            self.consumer_efficiencies = self.consumer_efficiencies[:new_consumers]
        self.state["consumer_count"] = new_consumers

        # update time of day & arrival rate (diurnal pattern + bursts)
        self.time_of_day = (self.time_of_day + 1/60.0) % 24.0  # step ~1 minute per tick by default
        diurnal = 1.0 + self.diurnal_amplitude * math.sin((self.time_of_day / 24.0) * 2 * math.pi)
        base_arrival = self.cfg.sim_queue_arrival_rate * diurnal

        # random bursts
        if self.npr.rand() < 0.02:
            burst = int(max(1, self.npr.poisson(lam=20)))
        else:
            burst = 0
        lam = max(0.0, base_arrival * (1.0 - throttle) + burst)
        arrivals = int(self.npr.poisson(lam=lam))

        # compute service capacity considering consumer efficiencies
        capacity = sum(self.consumer_efficiencies) * 1.0  # units per tick
        completed = int(min(self.state["queue_length"] + arrivals, int(capacity)))
        new_queue = max(0, (self.state["queue_length"] + arrivals) - completed)
        self.state["queue_length"] = min(self.cfg.sim_max_queue, new_queue)

        # latency model: queuing + heterogeneity effect
        base = float(self.cfg.sim_base_latency_ms or 100.0)
        load_factor = 1.0 + (self.state["queue_length"] / max(1.0, capacity))
        heterogeneity_penalty = 1.0 + 0.1 * np.std(self.consumer_efficiencies)
        noise = (self.npr.randn() * self.cfg.sim_noise_scale * base)
        avg_lat = base * load_factor * heterogeneity_penalty + noise
        self.state["avg_latency_ms"] = max(1.0, float(avg_lat))
        self.state["p95_latency_ms"] = self.state["avg_latency_ms"] * (1.0 + 0.6 * self.npr.rand())

        # success rate: degrade with queue & failures
        failure_rate = min(0.5, float(self.cfg.sim_failure_rate) + 0.001 * max(0, self.state["queue_length"] - capacity))
        self.state["success_rate"] = max(0.0, 1.0 - failure_rate)

        # CPU/mem usage: influenced by consumer_count and task mix
        cpu_base = 0.05 * self.state["consumer_count"]
        mem_base = 0.02 * self.state["consumer_count"]
        self.state["cpu"] = float(min(1.0, cpu_base + 0.5 * self.npr.rand()))
        self.state["mem"] = float(min(1.0, mem_base + 0.3 * self.npr.rand()))

        # arrival_rate scalar
        self.state["arrival_rate"] = float(lam)

# -----------------------
# LiveRealEnv: fetches real telemetry (best-effort; no destructive ops)
# -----------------------
class LiveRealEnv(RealEnvBase):
    """
    Uses live telemetry (Redis / Mongo / Prometheus) to build observations and provides
    an action interface that writes intents into a command queue.

    IMPORTANT:
      - This class only writes "intents" (scale requests / throttle signals) into a Redis list or Mongo collection.
      - A privileged operator (Kubernetes operator or autoscaler pod) must read that queue and perform actions.
      - For safety, by default dry_run=True; set dry_run=False only when you have an operator that enforces RBAC and validation.
    """

    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self._redis_cli = None
        self._mongo_cli = None
        # a local mirror of last window stats for smoothing
        self.window_stats: List[Dict[str, Any]] = []

    async def connect(self):
        if _HAS_AIOREDIS:
            try:
                self._redis_cli = await aioredis.from_url(self.cfg.redis_url, encoding="utf-8", decode_responses=True)
            except Exception:
                LOG.exception("LiveRealEnv: cannot connect to Redis")
        if _HAS_MOTOR and self.cfg.mongo_url:
            try:
                self._mongo_cli = motor_asyncio.AsyncIOMotorClient(self.cfg.mongo_url)
            except Exception:
                LOG.exception("LiveRealEnv: cannot connect to Mongo")

    def _reset_state(self):
        # attempt to seed state from recent metrics if possible
        # fallback to sim-like defaults if connectors unavailable
        try:
            # synchronous best-effort: attempt to pull one snapshot from redis keys if present
            # NOTE: better to make this async in production; here we provide fallback sync approach
            if _HAS_AIOREDIS:
                # can't await here; leave as best-effort no-op
                pass
        except Exception:
            LOG.exception("LiveRealEnv reset failed; falling back to simulated reset")
        super()._reset_state()

    def _advance_world(self, throttle: float, applied_delta_consumers: int, priority_bias: int):
        # Live env doesn't simulate; instead it updates local state by sampling metrics
        # For safety, we emulate the immediate effect of an action in the local mirror,
        # but the true effect only manifests when operator executes the intent.
        # So apply optimistic local update for training RL agent with human-in-the-loop.
        self.state["consumer_count"] = max(self.cfg.min_consumers, min(self.cfg.max_consumers, int(self.state["consumer_count"] + applied_delta_consumers)))
        # sample latest metrics from Redis/Mongo if available (best-effort)
        # Here we implement a synchronous fallback: try reading from a local cache file written by a metric collector
        metrics_path = pathlib.Path(os.getenv("PRIORITYMAX_LIVE_METRICS_CACHE", str(BASE_DIR / "live_metrics_cache.json")))
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
                # expected keys: queue_length, avg_latency_ms, p95_latency_ms, success_rate, arrival_rate, cpu, mem
                for k in ("queue_length", "avg_latency_ms", "p95_latency_ms", "success_rate", "arrival_rate", "cpu", "mem"):
                    if k in m:
                        self.state[k] = m[k]
            except Exception:
                LOG.exception("Failed to read live metrics cache")
        # otherwise maintain previous state (or simulated)
        # record window stats
        self.window_stats.append({"ts": time.time(), **self.state})
        if len(self.window_stats) > 300:
            self.window_stats.pop(0)

    def _apply_action(self, delta_consumers: int, throttle: float, priority_bias: int) -> Dict[str, Any]:
        attempt = {"wanted_delta": delta_consumers, "throttle": throttle, "priority_bias": priority_bias, "applied_delta": 0, "message": None}
        new_consumers = int(max(self.cfg.min_consumers, min(self.cfg.max_consumers, int(self.state["consumer_count"] + delta_consumers))))
        attempt["applied_delta"] = new_consumers - int(self.state["consumer_count"])
        # Write intent into Redis list for operator to pick up
        intent = {"id": str(uuid.uuid4()), "ts": time.time(), "type": "scale", "delta": attempt["applied_delta"], "throttle": throttle, "priority_bias": priority_bias}
        try:
            if _HAS_AIOREDIS:
                # fire-and-forget push (best-effort)
                # We cannot await in sync method; schedule with asyncio
                async def _push():
                    cli = await aioredis.from_url(self.cfg.redis_url, encoding="utf-8", decode_responses=True)
                    try:
                        await cli.lpush("prioritymax:scale_intents", json.dumps(intent))
                    finally:
                        await cli.close()
                try:
                    asyncio.get_running_loop().create_task(_push())
                except RuntimeError:
                    # no running loop; run in new loop
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(_push())
            elif _HAS_MOTOR and self.cfg.mongo_url:
                # write into Mongo collection 'scale_intents' (async best-effort)
                try:
                    client = motor_asyncio.AsyncIOMotorClient(self.cfg.mongo_url)
                    db = client.get_default_database()
                    coll = db["scale_intents"]
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(coll.insert_one(intent))
                except Exception:
                    LOG.exception("Failed to write scale intent to mongo")
            else:
                # fallback: write to local file for operator
                p = pathlib.Path(tempfile.gettempdir()) / f"prioritymax_intent_{intent['id']}.json"
                p.write_text(json.dumps(intent))
            attempt["message"] = "intent_emitted"
            write_audit_event({"event": "scale_intent_emitted", "intent": intent})
        except Exception:
            LOG.exception("Failed to emit intent")
            attempt["message"] = "intent_emit_failed"
        return attempt

# -----------------------
# Vectorized env factory & rollout helpers
# -----------------------
def make_env(cfg: EnvConfig) -> RealEnvBase:
    """
    Factory to create appropriate env class depending on mode: 'sim' -> SimulatedRealEnv; 'live' -> LiveRealEnv
    """
    if cfg.mode == "live":
        return LiveRealEnv(cfg)
    return SimulatedRealEnv(cfg)

def make_vec_env(cfg: EnvConfig, n: int = 4) -> List[RealEnvBase]:
    """
    Return a list of independent env instances for parallelized experience collection.
    """
    envs = []
    for i in range(n):
        c = EnvConfig(**vars(cfg))
        # vary seed slightly
        c.seed = (cfg.seed or int(time.time())) + i
        envs.append(make_env(c))
    return envs

# -------------
# Rollout collector
# -------------
@dataclass
class RolloutResult:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)

def collect_rollout(env: RealEnvBase, policy_fn: Callable[[np.ndarray], np.ndarray], horizon: int = 1000, render: bool = False) -> RolloutResult:
    """
    Collect a single-agent rollout synchronously.
    policy_fn: function mapping observation -> action (numpy)
    """
    obs = env.reset()
    result = RolloutResult()
    for t in range(horizon):
        action = policy_fn(obs)
        next_obs, reward, done, info = env.step(action)
        result.observations.append(obs)
        result.actions.append(np.asarray(action))
        result.rewards.append(float(reward))
        result.dones.append(bool(done))
        result.infos.append(info)
        obs = next_obs
        if render:
            env.render()
        if done:
            break
    return result

def collect_batch_rollouts(envs: List[RealEnvBase], policy_fn: Callable[[np.ndarray], np.ndarray], horizon: int = 1000, max_steps: Optional[int] = None) -> List[RolloutResult]:
    """
    Collect rollouts concurrently across a list of envs (threaded).
    Returns list of RolloutResult corresponding to each env.
    """
    results: List[RolloutResult] = [RolloutResult() for _ in envs]
    obs_list = [env.reset() for env in envs]
    steps = 0
    while True:
        for i, env in enumerate(envs):
            if max_steps and steps >= max_steps:
                continue
            if len(results[i].dones) and results[i].dones[-1]:
                continue
            action = policy_fn(obs_list[i])
            next_obs, reward, done, info = env.step(action)
            results[i].observations.append(obs_list[i])
            results[i].actions.append(np.asarray(action))
            results[i].rewards.append(float(reward))
            results[i].dones.append(bool(done))
            results[i].infos.append(info)
            obs_list[i] = next_obs
            steps += 1
        # termination
        if max_steps and steps >= max_steps:
            break
        # if all finished
        if all(len(r.dones) and r.dones[-1] for r in results):
            break
    return results

# -----------------------
# Evaluation helpers
# -----------------------
def evaluate_policy(env: RealEnvBase, policy_fn: Callable[[np.ndarray], np.ndarray], episodes: int = 10, horizon: int = 1000) -> Dict[str, Any]:
    rewards = []
    latencies = []
    for e in range(episodes):
        r = collect_rollout(env, policy_fn, horizon=horizon)
        rewards.append(sum(r.rewards))
        # compute avg latency from env.metrics_history
        avg_lat = np.mean([m["avg_latency_ms"] for m in env.metrics_history]) if env.metrics_history else env.state["avg_latency_ms"]
        latencies.append(avg_lat)
    return {"avg_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)), "avg_latency": float(np.mean(latencies)), "episodes": episodes}

# -----------------------
# Simple random policy (for testing)
# -----------------------
def random_policy(obs: np.ndarray) -> np.ndarray:
    """
    Baseline policy: small random adjustments.
    """
    delta = int(np.random.randint(-1, 2))
    throttle = float(np.random.random() * 0.2)
    priority = int(np.random.randint(-1, 2))
    return np.array([delta, throttle, priority], dtype=np.float32)

# -----------------------
# CLI helper
# -----------------------
def _build_cli():
    import argparse
    parser = argparse.ArgumentParser(prog="prioritymax-real-env")
    parser.add_argument("--mode", choices=["sim", "live"], default="sim")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="parallel envs for batch collection")
    parser.add_argument("--policy", choices=["random", "noop"], default="random")
    parser.add_argument("--dump", help="path to save rollout JSON")
    return parser

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    cfg = EnvConfig(mode=args.mode, seed=args.seed)
    envs = make_vec_env(cfg, n=args.workers)
    policy_fn = random_policy if args.policy == "random" else (lambda o: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    if args.workers == 1:
        res = collect_rollout(envs[0], policy_fn, horizon=args.steps, render=args.render)
        print("Total reward:", sum(res.rewards))
        if args.dump:
            out = {"observations": [o.tolist() for o in res.observations], "actions": [a.tolist() for a in res.actions], "rewards": res.rewards}
            pathlib.Path(args.dump).write_text(json.dumps(out, indent=2))
    else:
        results = collect_batch_rollouts(envs, policy_fn, horizon=args.steps, max_steps=args.steps * args.workers)
        for i, r in enumerate(results):
            print(f"Env[{i}] reward={sum(r.rewards)} steps={len(r.rewards)}")
        if args.dump:
            out = []
            for r in results:
                out.append({"rewards": r.rewards, "actions": [a.tolist() for a in r.actions]})
            pathlib.Path(args.dump).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main_cli()

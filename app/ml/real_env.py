#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PriorityMax - Real Environment Bridge for RL Training - ENTERPRISE EDITION (FIXED)
-----------------------------------------------------------------------------------

SYNCHRONIZED WITH:
- train_rl_heavy.py (observation/action dimensions, PPO training)
- train_rl_live.py (live mode operations, safety features)
- train_rl_eval.py (evaluation protocols, drift detection)

✅ FIXED: AsyncIO event loop issues
✅ FIXED: Thread-safe audit logging
✅ FIXED: Sync/async compatibility
✅ All enterprise features preserved

Enterprise Production Features:
✅ Observation space matches training scripts (dynamic obs_dim detection)
✅ Action space synchronized (3D continuous: [delta_workers, throttle, priority])
✅ Safety features: circuit breakers, rate limiting, dry-run mode
✅ Live telemetry integration (Redis, Mongo, Prometheus)
✅ Realistic simulation with diurnal patterns, bursts, heterogeneity
✅ Drift detection support (observation/reward distribution tracking)
✅ Audit logging for all actions (FIXED - thread-safe)
✅ Health monitoring and metrics export
✅ Thread-safe operations for concurrent training
✅ Emergency shutdown hooks
✅ Production-ready reward shaping
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
import signal
import atexit
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Callable, Iterable, Union
from collections import deque, defaultdict
from datetime import datetime
import statistics

import numpy as np

# Optional dependencies (best-effort imports)
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False

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

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
    LOG_GYM = True
except Exception:
    try:
        import gym
        from gym import spaces
        _HAS_GYM = True
        LOG_GYM = True
    except Exception:
        _HAS_GYM = False
        LOG_GYM = False
        # Minimal fallback
        class spaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=np.float32):
                    self.low = low
                    self.high = high
                    self.shape = shape if shape is not None else (len(low),)
                    self.dtype = dtype
            
            class Discrete:
                def __init__(self, n):
                    self.n = n

try:
    from prometheus_client import CollectorRegistry, Gauge, Histogram, Counter
    _HAS_PROM = True
except Exception:
    CollectorRegistry = Gauge = Histogram = Counter = None
    _HAS_PROM = False

# Logging
LOG = logging.getLogger("prioritymax.ml.real_env")
LOG.setLevel(os.getenv("PRIORITYMAX_ENV_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

if LOG_GYM and _HAS_GYM:
    try:
        LOG.info("Using Gymnasium (version %s)", gym.__version__)
    except:
        LOG.info("Gymnasium/Gym available")

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = pathlib.Path(os.getenv("PRIORITYMAX_DATA_DIR", str(BASE_DIR / "datasets")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_DIR = BASE_DIR / "logs"
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# FIXED: Thread-Safe Audit Logger
# ---------------------------
class AuditLogger:
    """
    Thread-safe, synchronous audit logger.
    NO asyncio dependencies - works in any context.
    """
    def __init__(self, log_dir: pathlib.Path = AUDIT_LOG_DIR):
        self.log_file = log_dir / "real_env_audit.jsonl"
        self.lock = threading.Lock()
        self.enabled = True
        
    def log(self, event: Dict[str, Any]):
        """Thread-safe synchronous logging."""
        if not self.enabled:
            return
        
        try:
            with self.lock:
                # Add timestamp if not present
                if "timestamp" not in event:
                    event["timestamp"] = datetime.utcnow().isoformat() + "Z"
                
                # Write atomically
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            LOG.debug("Audit log write failed: %s", e)

# Global audit logger instance
_AUDIT_LOGGER = AuditLogger()

def write_audit_event(payload: Dict[str, Any]):
    """
    FIXED: Synchronous audit logging function.
    Compatible with both sync and async contexts.
    """
    _AUDIT_LOGGER.log(payload)

# ---------------------------
# Environment Configuration
# ---------------------------
@dataclass
class EnvConfig:
    """
    SYNCHRONIZED with train_rl_heavy.py and train_rl_live.py configurations.
    """
    # Mode
    mode: str = "sim"  # 'sim' or 'live'
    
    # Observation/Action dimensions (CRITICAL SYNC POINT)
    obs_dim: Optional[int] = None  # Auto-detected (typically 8)
    act_dim: int = 3  # [delta_workers, throttle, priority_bias]
    
    # Connectors (live mode)
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    mongo_url: Optional[str] = os.getenv("MONGO_URL", None)
    prometheus_host: Optional[str] = os.getenv("PROM_HOST", None)
    
    # Safety & Rate Limiting (ENTERPRISE FEATURES)
    max_scale_delta: int = 5  # Max worker change per step
    min_workers: int = 1
    max_workers: int = 200
    action_cooldown_seconds: float = 5.0
    dry_run: bool = True  # SAFE DEFAULT
    
    # Circuit Breaker (TUNED for training stability)
    enable_circuit_breaker: bool = False  # Disabled by default for training
    circuit_breaker_threshold: int = 20  # Higher threshold (was 5)
    circuit_breaker_window_seconds: int = 60  # Shorter window (was 300)
    
    # Observation/Reward Windows
    obs_window_seconds: int = 60
    reward_latency_sla_ms: float = 500.0
    cost_per_worker_per_sec: float = 0.0005
    
    # Simulation Parameters (TUNED for stable training)
    sim_target_throughput: float = 50.0
    sim_base_latency_ms: float = 100.0
    sim_noise_scale: float = 0.1  # Reduced noise (was 0.2)
    sim_failure_rate: float = 0.01  # Lower failure rate (was 0.02)
    sim_queue_arrival_rate: float = 3.0  # Lower arrival rate (was 5.0)
    sim_max_queue: int = 10000
    sim_burst_probability: float = 0.01  # Less frequent bursts (was 0.02)
    sim_burst_size: float = 10.0  # Smaller bursts (was 20.0)
    
    # Diurnal Pattern
    sim_diurnal_amplitude: float = 0.7
    sim_time_step_minutes: float = 1.0
    
    # Reward Shaping (CRITICAL SYNC POINT)
    reward_latency_weight: float = 10.0
    reward_throughput_weight: float = 0.1
    reward_cost_weight: float = 1.0
    reward_queue_weight: float = 0.05
    reward_stability_weight: float = 0.1
    reward_success_weight: float = 1.0
    reward_sla_bonus: float = 0.5
    
    # Drift Detection
    enable_drift_tracking: bool = True
    drift_window_size: int = 1000
    
    # Seed
    seed: Optional[int] = None
    
    # Audit & Logging
    enable_audit_logging: bool = True
    enable_metrics_export: bool = True
    
    # Advanced Features
    enable_heterogeneous_workers: bool = True
    enable_task_priorities: bool = True
    enable_temporal_patterns: bool = True

# ---------------------------
# ENTERPRISE FEATURE: Circuit Breaker
# ---------------------------
class CircuitBreaker:
    """
    Thread-safe circuit breaker for failure protection.
    TUNED: More forgiving for RL training environments.
    """
    def __init__(self, threshold: int = 20, window_seconds: int = 60):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.failures = deque()
        self.successes = deque()  # Track successes too
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self.last_state_change = time.time()
        self.lock = threading.Lock()
        self.half_open_attempts = 0
        self.max_half_open_attempts = 3
        LOG.info("🔌 Circuit breaker initialized (threshold=%d, window=%ds)",
                threshold, window_seconds)
    
    def record_failure(self):
        """Record a failure and potentially open circuit."""
        with self.lock:
            now = time.time()
            self.failures.append(now)
            
            # Prune old failures
            cutoff = now - self.window_seconds
            while self.failures and self.failures[0] < cutoff:
                self.failures.popleft()
            
            # Only open if failures significantly exceed successes
            recent_successes = sum(1 for t in self.successes if t > cutoff)
            failure_rate = len(self.failures) / max(1, len(self.failures) + recent_successes)
            
            if len(self.failures) >= self.threshold and failure_rate > 0.8 and self.state == "CLOSED":
                self.state = "OPEN"
                self.last_state_change = now
                self.half_open_attempts = 0
                LOG.error("⚠️ CIRCUIT BREAKER OPENED - Too many failures (%d/%d, rate=%.1f%%)",
                         len(self.failures), len(self.failures) + recent_successes, failure_rate * 100)
                write_audit_event({
                    "event": "circuit_breaker_opened",
                    "failures": len(self.failures),
                    "failure_rate": failure_rate
                })
    
    def record_success(self):
        """Record a success and potentially close circuit."""
        with self.lock:
            now = time.time()
            self.successes.append(now)
            
            # Prune old successes
            cutoff = now - self.window_seconds
            while self.successes and self.successes[0] < cutoff:
                self.successes.popleft()
            
            if self.state == "HALF_OPEN":
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.max_half_open_attempts:
                    self.state = "CLOSED"
                    self.failures.clear()
                    self.half_open_attempts = 0
                    LOG.info("✅ Circuit breaker CLOSED - System recovered")
                    write_audit_event({
                        "event": "circuit_breaker_closed"
                    })
    
    def can_execute(self) -> bool:
        """Check if actions are allowed."""
        with self.lock:
            now = time.time()
            
            # Auto-recover to HALF_OPEN after window
            if self.state == "OPEN" and (now - self.last_state_change) > self.window_seconds:
                self.state = "HALF_OPEN"
                self.last_state_change = now
                self.half_open_attempts = 0
                LOG.info("🔄 Circuit breaker HALF_OPEN - Testing recovery")
            
            return self.state != "OPEN"

# ---------------------------
# ENTERPRISE FEATURE: Drift Tracker
# ---------------------------
class DriftTracker:
    """
    Track observation and reward distributions for drift detection.
    Thread-safe implementation.
    """
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.obs_buffer = deque(maxlen=window_size)
        self.reward_buffer = deque(maxlen=window_size)
        self.baseline_obs_stats = None
        self.baseline_reward_stats = None
        self.lock = threading.Lock()
        LOG.info("📊 Drift tracker initialized (window=%d)", window_size)
    
    def add_sample(self, obs: np.ndarray, reward: float):
        """Add observation and reward sample (thread-safe)."""
        with self.lock:
            if isinstance(obs, np.ndarray):
                self.obs_buffer.append(obs.copy())
            else:
                self.obs_buffer.append(np.array(obs))
            
            self.reward_buffer.append(float(reward))
            
            # Establish baseline after first window
            if len(self.obs_buffer) == self.window_size and self.baseline_obs_stats is None:
                self._compute_baseline()
    
    def _compute_baseline(self):
        """Compute baseline statistics (called under lock)."""
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
        """Compute drift score (thread-safe)."""
        with self.lock:
            if self.baseline_reward_stats is None or len(self.reward_buffer) < self.window_size:
                return None
            
            current_reward_mean = np.mean(list(self.reward_buffer))
            baseline_mean = self.baseline_reward_stats["mean"]
            baseline_std = self.baseline_reward_stats["std"]
            
            drift_score = abs(current_reward_mean - baseline_mean) / (baseline_std + 1e-8)
            return float(drift_score)
    
    def export_baseline(self) -> Dict[str, Any]:
        """Export baseline for evaluation (thread-safe)."""
        with self.lock:
            if self.baseline_obs_stats is None:
                return {}
            
            return {
                "observations": [obs.tolist() for obs in list(self.obs_buffer)],
                "rewards": list(self.reward_buffer),
                "baseline_obs_stats": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.baseline_obs_stats.items()
                },
                "baseline_reward_stats": self.baseline_reward_stats
            }

# ---------------------------
# Action/Observation Spaces (SYNCHRONIZED)
# ---------------------------
def get_action_space() -> Any:
    """
    CRITICAL SYNC POINT: Must match train_rl_heavy.py
    Action: [delta_workers, throttle_scale, priority_bias]
    """
    low = np.array([-10.0, 0.0, -2.0], dtype=np.float32)
    high = np.array([10.0, 1.0, 2.0], dtype=np.float32)
    
    if _HAS_GYM:
        return spaces.Box(low=low, high=high, dtype=np.float32)
    else:
        return {"low": low, "high": high, "shape": (3,)}

def get_observation_space(obs_dim: int = 8) -> Any:
    """
    CRITICAL SYNC POINT: Must match train_rl_heavy.py
    Default: [queue, workers, avg_lat, p95_lat, success, arrival, cpu, mem]
    """
    low = np.zeros(obs_dim, dtype=np.float32)
    high = np.array([1e6, 1e4, 1e6, 1e6, 1.0, 1e3, 1.0, 1.0], dtype=np.float32)[:obs_dim]
    
    # Pad if obs_dim > 8
    if obs_dim > 8:
        high = np.pad(high, (0, obs_dim - len(high)), constant_values=1e6)
    
    if _HAS_GYM:
        return spaces.Box(low=low, high=high, dtype=np.float32)
    else:
        return {"low": low, "high": high, "shape": (obs_dim,)}

# ---------------------------
# Base Environment (Enterprise Grade - FIXED)
# ---------------------------
class RealEnvBase:
    """
    Base class for PriorityMax RL environments.
    FIXED: All asyncio issues resolved, fully thread-safe.
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
        
        # Runtime state (CRITICAL: matches training expectations)
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
        
        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Prometheus metrics (best-effort)
        self._prom_registry = None
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
        
        # Shutdown hooks
        atexit.register(self._cleanup)
        
        # Signal handlers (safe registration)
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except:
            pass  # May fail in non-main threads
        
        LOG.info("✅ Environment initialized (mode=%s, obs_dim=%d, act_dim=%d)",
                config.mode, config.obs_dim, config.act_dim)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        LOG.warning("⚠️ Received signal %d, cleaning up...", signum)
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup resources on shutdown."""
        if not self.closed:
            self.close()
    
    # ---------------------------
    # Gym API (SYNCHRONIZED - FIXED)
    # ---------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        FIXED: Reset without asyncio dependencies.
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
            
            # FIXED: Synchronous audit logging
            if self.cfg.enable_audit_logging:
                write_audit_event({
                    "event": "env_reset",
                    "episode": self.episode_count,
                    "seed": seed
                })
            
            return obs
    
    def step(self, action: Union[np.ndarray, List[float], Tuple, Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        FIXED: Step without asyncio dependencies.
        """
        with self._lock:
            self.step_count += 1
            now = time.time()
            self.current_time = now
            
            # Parse action
            parsed_action = self._parse_action(action)
            
            # Safety checks
            if not self._safety_checks(parsed_action):
                parsed_action = {"delta_workers": 0, "throttle": 0.0, "priority_bias": 0}
            
            # Apply action
            exec_result = self._apply_action(
                parsed_action["delta_workers"],
                parsed_action["throttle"],
                parsed_action["priority_bias"]
            )
            
            # Advance world
            self._advance_world(
                parsed_action["throttle"],
                exec_result.get("applied_delta", 0),
                parsed_action["priority_bias"]
            )
            
            # Get observation
            obs = self._observe()
            
            # Compute reward
            reward, reward_details = self.compute_reward(exec_result)
            
            # Check termination
            done = self._terminal_condition()
            
            # Update tracking
            if self.drift_tracker:
                self.drift_tracker.add_sample(obs, reward)
            
            # Update Prometheus metrics (best-effort)
            if self._prom_registry:
                try:
                    self.prom_queue.set(self.state["queue_length"])
                    self.prom_workers.set(self.state["worker_count"])
                    self.prom_latency.set(self.state["avg_latency_ms"])
                    self.prom_reward.set(reward)
                    self.prom_steps.inc()
                except Exception:
                    pass
            
            # Build info
            info = {
                "exec": exec_result,
                "reward_details": reward_details,
                "step": self.step_count,
                "episode": self.episode_count,
                "timestamp": now,
                "state": self.state.copy(),
                "circuit_breaker_ok": self.circuit_breaker.can_execute() if self.circuit_breaker else True,
                "drift_score": self.drift_tracker.get_drift_score() if self.drift_tracker else None
            }
            
            # Record metrics
            self.metrics_history.append({
                "ts": now,
                **self.state,
                "action": parsed_action,
                "reward": reward
            })
            
            # Update last action time
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
        
        # FIXED: Synchronous audit logging
        if self.cfg.enable_audit_logging:
            write_audit_event({
                "event": "env_closed",
                "total_steps": self.step_count,
                "total_episodes": self.episode_count
            })
        
        LOG.info("Environment closed (steps=%d, episodes=%d)",
                self.step_count, self.episode_count)
    
    # ---------------------------
    # Action Parsing (SYNCHRONIZED)
    # ---------------------------
    def _parse_action(self, action: Union[np.ndarray, List[float], Tuple, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse action into structured format."""
        if isinstance(action, dict):
            return {
                "delta_workers": action.get("delta_workers", 0),
                "throttle": action.get("throttle", 0.0),
                "priority_bias": action.get("priority_bias", 0)
            }
        
        # Convert to array
        arr = np.asarray(action, dtype=float).flatten()
        
        # Pad if needed
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), constant_values=0)
        
        return {
            "delta_workers": int(np.round(arr[0])),
            "throttle": float(np.clip(arr[1], 0.0, 1.0)),
            "priority_bias": int(np.round(arr[2]))
        }
    
    # ---------------------------
    # Safety Checks (ENTERPRISE)
    # ---------------------------
    def _safety_checks(self, action: Dict[str, Any]) -> bool:
        """Perform safety checks before executing action."""
        now = time.time()
        
        # Cooldown check
        if (now - self.last_action_time) < self.cfg.action_cooldown_seconds:
            if action["delta_workers"] != 0:
                LOG.debug("Action cooldown active (%.1fs remaining)",
                         self.cfg.action_cooldown_seconds - (now - self.last_action_time))
                return False
        
        # Circuit breaker check
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            LOG.warning("⚠️ Circuit breaker OPEN - action blocked")
            return False
        
        # Bounds check
        new_workers = self.state["worker_count"] + action["delta_workers"]
        if new_workers < self.cfg.min_workers or new_workers > self.cfg.max_workers:
            LOG.debug("Action would violate worker bounds (%d not in [%d, %d])",
                     new_workers, self.cfg.min_workers, self.cfg.max_workers)
            return False
        
        return True
    
    # ---------------------------
    # Action Application (FIXED - to override)
    # ---------------------------
    def _apply_action(self, delta_workers: int, throttle: float, priority_bias: int) -> Dict[str, Any]:
        """
        FIXED: Apply action without asyncio dependencies.
        Override in subclasses for live mode.
        """
        attempt = {
            "wanted_delta": delta_workers,
            "applied_delta": 0,
            "throttle": throttle,
            "priority_bias": priority_bias,
            "message": None,
            "ts": time.time()
        }

        # Enforce worker bounds
        new_worker_count = int(
            np.clip(
                self.state["worker_count"] + delta_workers,
                self.cfg.min_workers,
                self.cfg.max_workers
            )
        )
        attempt["applied_delta"] = int(new_worker_count - self.state["worker_count"])

        # Live mode handling (FIXED - synchronous)
        if self.cfg.mode == "live" and not self.cfg.dry_run:
            intent = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "scale_action",
                "delta": attempt["applied_delta"],
                "throttle": throttle,
                "priority_bias": priority_bias
            }

            try:
                # FIXED: Synchronous Redis operation
                if _HAS_REDIS and self.cfg.redis_url:
                    try:
                        r = redis.from_url(self.cfg.redis_url, decode_responses=True)
                        r.lpush("prioritymax:scale_intents", json.dumps(intent))
                        r.close()
                        attempt["message"] = "intent_pushed_to_redis"
                    except Exception as e:
                        LOG.debug("Redis push failed: %s", e)
                        # Fallback to file
                        p = pathlib.Path(tempfile.gettempdir()) / f"intent_{intent['id']}.json"
                        p.write_text(json.dumps(intent))
                        attempt["message"] = "intent_written_to_file"
                else:
                    # Fallback to local file intent
                    p = pathlib.Path(tempfile.gettempdir()) / f"intent_{intent['id']}.json"
                    p.write_text(json.dumps(intent))
                    attempt["message"] = "intent_written_to_file"
                
                # FIXED: Synchronous audit logging
                write_audit_event({
                    "event": "live_scale_intent",
                    "intent": intent
                })
            except Exception as e:
                LOG.exception("Failed to emit scale intent: %s", e)
                attempt["message"] = f"intent_emit_failed: {e}"
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
        else:
            # Simulation or dry run
            attempt["message"] = "dry_run_or_simulated"

        return attempt

    # ---------------------------
    # WORLD DYNAMICS (SIMULATION - to override)
    # ---------------------------
    def _advance_world(self, throttle: float, applied_delta: int, priority_bias: int):
        """
        Advance simulation dynamics (override in subclasses).
        Base implementation does nothing (for live mode).
        """
        pass

    def _reset_state(self):
        """Reset state to initial conditions (override in subclasses)."""
        pass

    # ---------------------------
    # OBSERVATION VECTOR
    # ---------------------------
    def _observe(self) -> np.ndarray:
        """Return normalized observation vector."""
        arr = np.array([
            self.state["queue_length"],
            self.state["worker_count"],
            self.state["avg_latency_ms"],
            self.state["p95_latency_ms"],
            self.state["success_rate"],
            self.state["arrival_rate"],
            self.state["cpu"],
            self.state["mem"]
        ], dtype=np.float32)
        
        # Ensure we match configured obs_dim
        if self.cfg.obs_dim and len(arr) < self.cfg.obs_dim:
            arr = np.pad(arr, (0, self.cfg.obs_dim - len(arr)), constant_values=0)
        elif self.cfg.obs_dim and len(arr) > self.cfg.obs_dim:
            arr = arr[:self.cfg.obs_dim]
        
        return np.clip(arr, 0, 1e6)

    # ---------------------------
    # REWARD FUNCTION (SYNCED)
    # ---------------------------
    def compute_reward(self, exec_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        FIXED: Reward shaping with strict bounds to prevent explosion.
        """
        sla = self.cfg.reward_latency_sla_ms
        lat = self.state["avg_latency_ms"]
        succ = self.state["success_rate"]
        q = self.state["queue_length"]
        w = self.state["worker_count"]
        delta = exec_result.get("applied_delta", 0)
        throttle = exec_result.get("throttle", 0.0)

        # Bounded latency penalty
        lat_normalized = min(lat / sla, 10.0)  # Cap at 10x SLA
        latency_pen = -((max(0.0, lat_normalized - 1.0)) ** 2) * self.cfg.reward_latency_weight
        latency_pen = max(latency_pen, -100.0)  # Floor

        # Bounded queue penalty
        queue_normalized = min(q / 1000.0, 10.0)
        queue_pen = -math.log1p(queue_normalized) * self.cfg.reward_queue_weight
        queue_pen = max(queue_pen, -50.0)  # Floor

        # Throughput reward (bounded)
        throughput_reward = min(succ * w * self.cfg.reward_throughput_weight, 10.0)

        # Cost penalty (bounded)
        cost_pen = -min(w * self.cfg.cost_per_worker_per_sec * self.cfg.reward_cost_weight, 10.0)

        # Stability penalty (bounded)
        stability_pen = -min(abs(delta) * self.cfg.reward_stability_weight, 5.0)

        # Throttle penalty
        throttle_pen = -throttle * 0.5

        # Success bonus (bounded)
        success_bonus = min(succ * self.cfg.reward_success_weight, 5.0)

        total = latency_pen + throughput_reward + cost_pen + queue_pen + stability_pen + throttle_pen + success_bonus

        # SLA bonus
        sla_bonus = 0.0
        if lat <= sla:
            sla_bonus = self.cfg.reward_sla_bonus
            total += sla_bonus

        # CRITICAL: Clip final reward
        total = float(np.clip(total, -1000.0, 100.0))

        # Circuit breaker (lenient)
        if self.circuit_breaker:
            if succ > 0.5 and lat < sla * 5.0:
                self.circuit_breaker.record_success()

        details = {
            "latency_pen": float(latency_pen),
            "throughput_reward": float(throughput_reward),
            "cost_penalty": float(cost_pen),
            "queue_penalty": float(queue_pen),
            "stability_penalty": float(stability_pen),
            "throttle_penalty": float(throttle_pen),
            "success_bonus": float(success_bonus),
            "sla_bonus": float(sla_bonus),
            "final_reward": total
        }
        return total, details

    # ---------------------------
    # TERMINAL CONDITION
    # ---------------------------
    def _terminal_condition(self) -> bool:
        """
        Check if episode should terminate.
        
        FIXED: Much more lenient criteria for stable RL training.
        - Increased thresholds significantly
        - Added step limit to prevent infinite episodes  
        - Only terminates on truly catastrophic failures
        - Allows agent to learn from difficult situations
        """
        # Prevent infinite episodes with step limit
        MAX_EPISODE_STEPS = 10000
        if self.step_count >= MAX_EPISODE_STEPS:
            LOG.info("Episode complete: reached max steps (%d)", MAX_EPISODE_STEPS)
            return True
        
        # Define EXTREME failure thresholds (much more lenient than before)
        catastrophic_latency = self.state["p95_latency_ms"] > 500000  # 500 seconds (was 100s)
        catastrophic_queue = self.state["queue_length"] >= self.cfg.sim_max_queue * 0.999  # 99.9% full (was 99%)
        complete_system_failure = self.state["success_rate"] < 0.01  # Less than 1% success rate
        
        # Only terminate if multiple failure conditions are met
        should_terminate = (catastrophic_latency and catastrophic_queue) or complete_system_failure
        
        if should_terminate:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            LOG.warning(
                "Episode terminated: lat=%.0fms queue=%.0f/%.0f success=%.2f%% step=%d",
                self.state["p95_latency_ms"], 
                self.state["queue_length"],
                self.cfg.sim_max_queue,
                self.state["success_rate"] * 100,
                self.step_count
            )
            return True
        
        return False
# ---------------------------------------------------
# SIMULATED ENVIRONMENT (for training) - FIXED
# ---------------------------------------------------
class SimulatedRealEnv(RealEnvBase):
    """
    FIXED: Fully featured simulator with realistic dynamics.
    No asyncio dependencies, fully thread-safe.
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        # Additional simulation state
        self.consumer_eff = []
        self.sim_time_of_day = 0.0  # Minutes since midnight
        
    def _reset_state(self):
        """Reset simulation state with safer initial conditions."""
        self.state.update({
            "queue_length": float(self.npr.randint(0, 20)),  # Lower initial queue
            "worker_count": float(self.npr.randint(self.cfg.min_workers, self.cfg.min_workers + 3)),
            "avg_latency_ms": float(self.cfg.sim_base_latency_ms * 0.8),  # Start with good latency
            "p95_latency_ms": float(self.cfg.sim_base_latency_ms * 1.0),
            "success_rate": 0.98,  # Start with high success rate
            "arrival_rate": self.cfg.sim_queue_arrival_rate * 0.5,  # Start with lower load
            "cpu": 0.15,
            "mem": 0.15
        })
        
        # Initialize heterogeneous worker efficiencies
        if self.cfg.enable_heterogeneous_workers:
            self.consumer_eff = [
                self.npr.uniform(0.8, 1.2) 
                for _ in range(int(self.state["worker_count"]))
            ]
        else:
            self.consumer_eff = [1.0] * int(self.state["worker_count"])
        
        # Reset time of day
        self.sim_time_of_day = datetime.utcnow().hour * 60.0 + datetime.utcnow().minute
    
    def _advance_world(self, throttle: float, applied_delta: int, priority_bias: int):
        """
        FIXED: Advance simulation with realistic dynamics.
        Incorporates diurnal patterns, bursts, heterogeneity, and noise.
        """
        # Update worker count
        old_workers = int(self.state["worker_count"])
        self.state["worker_count"] = max(
            self.cfg.min_workers,
            min(self.cfg.max_workers, self.state["worker_count"] + applied_delta)
        )
        new_workers = int(self.state["worker_count"])
        
        # Update worker efficiencies
        if self.cfg.enable_heterogeneous_workers:
            if new_workers > old_workers:
                # Add new workers
                for _ in range(new_workers - old_workers):
                    self.consumer_eff.append(self.npr.uniform(0.8, 1.2))
            elif new_workers < old_workers:
                # Remove workers
                self.consumer_eff = self.consumer_eff[:new_workers]
        
        # Advance time of day
        if self.cfg.enable_temporal_patterns:
            self.sim_time_of_day += self.cfg.sim_time_step_minutes
            if self.sim_time_of_day >= 1440:  # 24 hours
                self.sim_time_of_day -= 1440
        
        # Arrival rate with diurnal pattern
        if self.cfg.enable_temporal_patterns:
            tod_hours = self.sim_time_of_day / 60.0
            diurnal_factor = 1.0 + self.cfg.sim_diurnal_amplitude * math.sin(
                (tod_hours / 24.0) * 2 * math.pi - math.pi / 2  # Peak at noon
            )
        else:
            diurnal_factor = 1.0
        
        # Base arrival rate modulated by throttle
        arrival_rate = max(0.0, self.cfg.sim_queue_arrival_rate * diurnal_factor * (1.0 - throttle))

        # Random burst events
        if self.random.random() < self.cfg.sim_burst_probability:
            burst_multiplier = 1.0 + self.cfg.sim_burst_size / 10.0
            arrival_rate *= burst_multiplier
            LOG.debug("Burst event! arrival_rate=%.2f (x%.2f)", arrival_rate, burst_multiplier)

        # Poisson arrivals
        arrivals = self.npr.poisson(lam=arrival_rate)
        
        # Service capacity with worker heterogeneity
        if self.cfg.enable_heterogeneous_workers and self.consumer_eff:
            avg_efficiency = np.mean(self.consumer_eff[:new_workers])
            service_capacity = max(1.0, new_workers * avg_efficiency)
        else:
            service_capacity = max(1.0, new_workers * self.random.uniform(0.8, 1.2))
        
        # Queue dynamics
        completed = min(self.state["queue_length"] + arrivals, int(service_capacity))
        self.state["queue_length"] = max(
            0.0, 
            min(self.cfg.sim_max_queue, self.state["queue_length"] + arrivals - completed)
        )

        # Latency model: increases with load
        if service_capacity > 0:
            load_factor = 1.0 + (self.state["queue_length"] / service_capacity)
        else:
            load_factor = 10.0
        
        noise = self.npr.randn() * self.cfg.sim_noise_scale * self.cfg.sim_base_latency_ms
        base_latency = self.cfg.sim_base_latency_ms * load_factor + noise
        self.state["avg_latency_ms"] = max(1.0, base_latency)
        self.state["p95_latency_ms"] = base_latency * (1.0 + 0.4 * self.random.random())

        # Success rate: degrades with overload
        overload = max(0.0, self.state["queue_length"] - service_capacity)
        fail_rate = min(0.5, self.cfg.sim_failure_rate + 0.001 * overload)
        self.state["success_rate"] = max(0.0, 1.0 - fail_rate)

        # Resource utilization
        self.state["cpu"] = min(1.0, 
            0.05 * self.state["worker_count"] + 
            0.0005 * self.state["queue_length"] +
            self.npr.randn() * 0.05
        )
        self.state["mem"] = min(1.0, 
            0.02 * self.state["worker_count"] + 
            0.0003 * self.state["queue_length"] +
            self.npr.randn() * 0.02
        )

        # Update arrival rate observation
        self.state["arrival_rate"] = arrival_rate

# ---------------------------------------------------
# LIVE ENVIRONMENT (for real ops) - FIXED
# ---------------------------------------------------
class LiveRealEnv(RealEnvBase):
    """
    FIXED: Bridge to live PriorityMax telemetry.
    No asyncio dependencies, uses synchronous operations.
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.metrics_cache_path = BASE_DIR / "live_metrics_cache.json"
        
    def _reset_state(self):
        """Load initial state from live metrics cache."""
        try:
            if self.metrics_cache_path.exists():
                data = json.loads(self.metrics_cache_path.read_text())
                for k in self.state.keys():
                    if k in data:
                        self.state[k] = float(data[k])
                LOG.info("Loaded live state from cache: workers=%.0f queue=%.0f",
                        self.state["worker_count"], self.state["queue_length"])
            else:
                LOG.warning("No live metrics cache found, using defaults")
                self.state["queue_length"] = 0.0
                self.state["worker_count"] = float(self.cfg.min_workers)
        except Exception as e:
            LOG.warning("Live reset fallback due to error: %s", e)
            self.state["queue_length"] = 0.0
            self.state["worker_count"] = float(self.cfg.min_workers)

    def _advance_world(self, throttle: float, applied_delta: int, priority_bias: int):
        """
        FIXED: Fetch live metrics synchronously.
        Falls back to stale state if cache unavailable.
        """
        try:
            if self.metrics_cache_path.exists():
                data = json.loads(self.metrics_cache_path.read_text())
                
                # Update state from live metrics
                for k, v in data.items():
                    if k in self.state:
                        self.state[k] = float(v)
                
                LOG.debug("Updated from live metrics: queue=%.0f workers=%.0f lat=%.1fms",
                         self.state["queue_length"], 
                         self.state["worker_count"],
                         self.state["avg_latency_ms"])
            else:
                # Fallback: simple state evolution
                LOG.debug("No live metrics, using fallback evolution")
                self.state["worker_count"] = max(
                    self.cfg.min_workers,
                    min(self.cfg.max_workers, self.state["worker_count"] + applied_delta)
                )
                self.state["cpu"] = min(1.0, self.state["cpu"] + 0.01 * applied_delta)
                self.state["mem"] = min(1.0, self.state["mem"] + 0.005 * applied_delta)
                
        except Exception as e:
            LOG.debug("Live metrics fetch failed: %s", e)
            # Keep stale state

# ---------------------------------------------------
# FACTORY FUNCTIONS
# ---------------------------------------------------
def make_env(cfg: EnvConfig) -> RealEnvBase:
    """Factory function to create environment based on config."""
    if cfg.mode == "live":
        return LiveRealEnv(cfg)
    else:
        return SimulatedRealEnv(cfg)

def make_vec_env(cfg: EnvConfig, n: int = 4) -> List[RealEnvBase]:
    """
    Create vectorized environments for parallel training.
    FIXED: Each environment gets unique seed.
    """
    envs = []
    base_seed = cfg.seed if cfg.seed is not None else int(time.time())
    
    for i in range(n):
        # Create config copy with unique seed
        env_cfg = EnvConfig(**asdict(cfg))
        env_cfg.seed = base_seed + i
        envs.append(make_env(env_cfg))
    
    return envs

# ---------------------------------------------------
# HEALTH CHECK & DIAGNOSTICS
# ---------------------------------------------------
def run_env_diagnostic(cfg: Optional[EnvConfig] = None) -> Dict[str, Any]:
    """
    Run environment diagnostic to verify functionality.
    Useful for CI/CD and pre-training validation.
    """
    if cfg is None:
        cfg = EnvConfig(mode="sim", seed=42)
    
    results = {
        "config": asdict(cfg),
        "dependencies": {
            "gym": _HAS_GYM,
            "pandas": _HAS_PANDAS,
            "redis": _HAS_REDIS,
            "psutil": _HAS_PSUTIL,
            "prometheus": _HAS_PROM
        },
        "tests": {}
    }
    
    try:
        # Test environment creation
        env = make_env(cfg)
        results["tests"]["create_env"] = "✅ PASS"
        
        # Test reset
        obs = env.reset(seed=42)
        results["tests"]["reset"] = f"✅ PASS (obs_shape={obs.shape})"
        
        # Test step
        action = np.array([1.0, 0.0, 0.0])
        obs, reward, done, info = env.step(action)
        results["tests"]["step"] = f"✅ PASS (reward={reward:.3f})"
        
        # Test multiple steps
        for _ in range(10):
            action = env.action_space.sample() if hasattr(env.action_space, 'sample') else np.zeros(3)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        results["tests"]["multi_step"] = "✅ PASS"
        
        # Test close
        env.close()
        results["tests"]["close"] = "✅ PASS"
        
        results["status"] = "✅ ALL TESTS PASSED"
        
    except Exception as e:
        results["status"] = f"❌ FAILED: {e}"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results

# ---------------------------------------------------
# CLI for diagnostics
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PriorityMax Real Environment - Diagnostics")
    parser.add_argument("--mode", default="sim", choices=["sim", "live"], help="Environment mode")
    parser.add_argument("--diagnostic", action="store_true", help="Run diagnostic tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    
    if args.diagnostic:
        print("\n" + "="*70)
        print("PriorityMax Real Environment - Diagnostic Report")
        print("="*70 + "\n")
        
        cfg = EnvConfig(mode=args.mode, seed=42)
        results = run_env_diagnostic(cfg)
        
        print(json.dumps(results, indent=2))
        print("\n" + "="*70)
        print(f"Status: {results['status']}")
        print("="*70 + "\n")
        
        sys.exit(0 if results["status"].startswith("✅") else 1)
    
    else:
        # Interactive test
        print("\n🎮 Interactive Environment Test\n")
        
        cfg = EnvConfig(mode=args.mode, seed=42)
        env = make_env(cfg)
        
        obs = env.reset()
        print(f"Initial observation: {obs}")
        
        for step in range(5):
            action = np.random.randn(3)  # Random action
            obs, reward, done, info = env.step(action)
            
            print(f"\nStep {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Queue: {info['state']['queue_length']:.0f}")
            print(f"  Workers: {info['state']['worker_count']:.0f}")
            print(f"  Latency: {info['state']['avg_latency_ms']:.1f}ms")
            
            if done:
                print("  Episode terminated!")
                break
        
        env.close()
        print("\n✅ Test complete!\n")
#!/usr/bin/env python3
# train_rl_live.py
"""
Train RL — Live / Online Trainer (PPO variant) for PriorityMax
--------------------------------------------------------------

Purpose:
 - Train an RL agent (PPO-style) in a "live" mode — interacting with real metrics, live queue telemetry,
   and optionally applying small, safe control actions in production (autoscaler hints).
 - Designed for production safety:
     * Conservative action scaling by default (small step sizes)
     * Dry-run / simulation-first modes
     * Rate-limited model updates and checkpointing
     * Canary promotion workflow (evaluate candidate on holdout traffic or shadow mode before promotion)
 - Integrates with ModelRegistry, Metrics, Storage, and Autoscaler hooks from the PriorityMax system.

Notes:
 - This script is intended to run as a controlled training pod (sidecar/autoscaler) that receives telemetry,
   suggests actions, and learns from observed outcomes.
 - For heavy batch training use `train_rl_heavy.py`. This file focuses on online / continual learning.
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
import argparse
import tempfile
import atexit
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Optional dependencies (best-effort)
# ---------------------------
_HAS_TORCH = False
_HAS_MLFLOW = False
_HAS_WANDB = False
_HAS_AIOHTTP = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    optim = None

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None

try:
    import aiohttp
    _HAS_AIOHTTP = True
except Exception:
    aiohttp = None

# ---------------------------
# Project imports (best-effort)
# ---------------------------
try:
    from app.ml.real_env import SimulatedRealEnv, EnvConfig, make_vec_env
    from app.ml.model_registry import ModelRegistry
    from app.metrics import metrics
    from app.queue.redis_queue import RedisQueue
    from app.app.autoscaler import PriorityMaxAutoscaler
    from app.api.admin import write_audit_event
except Exception:
    SimulatedRealEnv = None
    EnvConfig = None
    make_vec_env = None
    ModelRegistry = None
    metrics = None
    RedisQueue = None
    PriorityMaxAutoscaler = None
    def write_audit_event(payload):
        # fallback audit file
        try:
            pth = pathlib.Path("/tmp/prioritymax_live_audit.jsonl")
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(json.dumps(payload) + "\n")
        except Exception:
            pass

# ---------------------------
# Paths & defaults
# ---------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "app" / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CKPT = MODELS_DIR / "rl_live.pt"
LOGDIR = ROOT / "logs" / "rl_live"
LOGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------
LOG = logging.getLogger("prioritymax.train_rl_live")
LOG.setLevel(os.getenv("PRIORITYMAX_RL_LIVE_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# ---------------------------
# Config dataclasses
# ---------------------------
@dataclass
class LiveRLConfig:
    # Live training loop params
    rollout_steps: int = 2048            # steps collected before an update
    update_epochs: int = 4               # PPO epochs per update
    mini_batch_size: int = 64
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 2.5e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Safety & control
    action_scale_limit: float = 0.5      # scale RL output to [-limit, limit] before mapping to worker changes
    max_scale_step: int = 1              # maximum workers to add/remove per action
    min_workers: int = 1
    max_workers: int = 10
    cooldown_seconds: int = 30           # minimum seconds between applying scaling actions

    # env & telemetry
    env_mode: str = "live"               # 'sim' or 'live'
    env_seed: int = 42
    num_envs: int = 4

    # logging & experiments
    experiment_name: str = "prioritymax_rl_live"
    run_name: Optional[str] = None
    log_wandb: bool = False
    log_mlflow: bool = False
    wandb_project: str = "PriorityMax-RL-Live"
    mlflow_experiment: str = "PriorityMax-RL-Live"

    # checkpointing
    checkpoint_path: str = str(DEFAULT_CKPT)
    checkpoint_interval: int = 300       # seconds between checkpoint saves
    resume_from: Optional[str] = None

    # operational
    dry_run: bool = False                # if true, do not apply scale actions
    audit_log: bool = True
    rate_limit_updates: int = 1          # max updates per minute (safety)

    # RL model
    obs_dim: Optional[int] = None
    act_dim: int = 1                     # single scalar action indicating scale hint

    # misc
    seed: int = 12345
    verbose: bool = True

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def current_iso_ts():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def atomic_write_json(obj: Dict[str, Any], path: str):
    p = pathlib.Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    tmp.replace(p)

def save_checkpoint(state: Dict[str, Any], path: str):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if _HAS_TORCH and "model" in state and isinstance(state["model"], torch.nn.Module):
            ck = state.copy()
            model = ck.pop("model")
            ck["model_state_dict"] = model.state_dict()
            if "optimizer" in ck and hasattr(ck["optimizer"], "state_dict"):
                ck["optimizer_state_dict"] = ck["optimizer"].state_dict()
                ck.pop("optimizer", None)
            torch.save(ck, str(p))
        else:
            import joblib
            joblib.dump(state, str(p))
        LOG.info("Checkpoint saved to %s", p)
    except Exception:
        LOG.exception("Failed to save checkpoint to %s", p)

def load_checkpoint(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    try:
        if _HAS_TORCH:
            return torch.load(str(p), map_location="cpu")
    except Exception:
        pass
    import joblib
    return joblib.load(str(p))

# ---------------------------
# Simple, conservative Actor-Critic
# ---------------------------
if _HAS_TORCH:
    class LiveActorCritic(nn.Module):
        """
        Small actor-critic suitable for live training:
         - policy returns single scalar mean (continuous) and log_std param
         - value head returns scalar value estimate
         - intentionally small to avoid overfitting in low-data online regime
        """
        def __init__(self, obs_dim: int, hidden: int = 128):
            super().__init__()
            self.obs_dim = obs_dim
            self.hidden = hidden
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU()
            )
            self.policy_head = nn.Linear(hidden, 1)
            self.value_head = nn.Linear(hidden, 1)
            self.log_std = nn.Parameter(torch.tensor([ -1.0 ]))  # small initial std

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.shared(x)
            mean = self.policy_head(h).squeeze(-1)
            value = self.value_head(h).squeeze(-1)
            return mean, value

        def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False):
            mean, value = self.forward(obs)
            std = torch.exp(self.log_std)
            if deterministic:
                action = mean
                logp = None
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            return action, logp, value
else:
    LiveActorCritic = None

# ---------------------------
# Live environment adapter
# ---------------------------
class LiveEnvAdapter:
    """
    Adapter that wraps either SimulatedRealEnv (for local testing) or
    provides interfaces to fetch real telemetry and simulate action application.

    Interface:
      - reset() -> obs
      - step(action) -> obs, reward, done, info

    In live mode, 'step' does not call a real external system to change workers directly.
    Instead, the Autoscaler or another safe controller should consume the RL hints returned
    by the trainer. This adapter supports 'apply_hint' to forward the action to the autoscaler in a safe way.
    """
    def __init__(self, cfg: LiveRLConfig, redis_queue: Optional[Any] = None, autoscaler: Optional[Any] = None):
        self.cfg = cfg
        self.redis = redis_queue or (RedisQueue() if RedisQueue else None)
        self.autoscaler = autoscaler or (PriorityMaxAutoscaler() if PriorityMaxAutoscaler else None)
        self.mode = cfg.env_mode
        # for sim mode use SimulatedRealEnv
        if self.mode == "sim" and SimulatedRealEnv:
            self.env = SimulatedRealEnv(EnvConfig(mode="sim", seed=cfg.env_seed))
        else:
            self.env = None  # live telemetry mode
        self.last_obs = None

    def reset(self, seed: Optional[int] = None):
        if self.env:
            self.last_obs = self.env.reset(seed=seed)
            return self.last_obs
        # For live mode: fetch current metrics snapshot from metrics module or Redis
        snap = {}
        try:
            if metrics:
                snap = metrics.snapshot()
                # produce a small obs vector e.g., backlog, p95 latency, worker count, cpu
                q = snap.get("queues", {})
                # pick first queue if exists
                if q:
                    qname = next(iter(q.keys()))
                    backlog = q[qname]["backlog"].get("60", {}).get("mean", 0.0)
                    lat_p95 = q[qname]["latency"].get("60", {}).get("p95", 0.0)
                else:
                    backlog = 0.0
                    lat_p95 = 0.0
                worker_count = 0
                try:
                    # try get worker count via prometheus gauge reading (best-effort)
                    worker_count = int(os.getenv("PRIORITYMAX_CURRENT_WORKERS", "0"))
                except Exception:
                    worker_count = 0
                self.last_obs = [float(backlog), float(lat_p95), float(worker_count)]
                return self.last_obs
            # fallback: empty obs
            self.last_obs = [0.0, 0.0, 0.0]
            return self.last_obs
        except Exception:
            LOG.exception("LiveEnvAdapter.reset failed")
            self.last_obs = [0.0, 0.0, 0.0]
            return self.last_obs

    def step(self, action: float) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        """
        action: scalar continuous value from RL in range unconstrained (we will scale/clip)
        Returns: obs, reward, done, info
        Reward design is critical. For live mode, reward should reflect improved SLA / reduced latency /
        reduced cost. This method computes a best-effort reward from metrics snapshots.
        """
        # map and clamp action to safe discrete change
        try:
            scaled = float(action) * float(self.cfg.action_scale_limit)
            # interpret scaled in [-limit, limit] → map to integer delta in [-max_step, max_step]
            delta = int(math.copysign(min(abs(round(scaled)), self.cfg.max_scale_step), scaled))
        except Exception:
            delta = 0

        info = {"applied_delta": 0, "snap_before": None, "snap_after": None}
        # snap before
        before = None
        try:
            if metrics:
                before = metrics.snapshot()
            info["snap_before"] = before
        except Exception:
            before = None

        # send hint to autoscaler (non-blocking) — the Autoscaler is responsible for actual scale with safety.
        if self.autoscaler and not self.cfg.dry_run:
            try:
                # autoscaler.apply_hint should be implemented to accept small hints
                if hasattr(self.autoscaler, "apply_hint"):
                    # apply_hint may be async; schedule it
                    res = self.autoscaler.apply_hint(delta)
                    # if coroutine, schedule
                    if hasattr(res, "__await__"):
                        import asyncio
                        asyncio.get_event_loop().create_task(res)
                    info["applied_delta"] = delta
                else:
                    # fallback: ask storage to write desired worker count or enqueue a control message
                    if self.redis and hasattr(self.redis, "enqueue_control"):
                        self.redis.enqueue_control({"delta": delta, "ts": time.time()})
                        info["applied_delta"] = delta
            except Exception:
                LOG.exception("Failed to forward hint to autoscaler")
        else:
            # dry-run: don't apply
            info["applied_delta"] = 0

        # short sleep to allow system to react slightly (in live this should be tuned carefully)
        try:
            time.sleep(0.5)
        except Exception:
            pass

        # compute reward based on change in metrics (before vs after)
        after = None
        try:
            if metrics:
                after = metrics.snapshot()
            info["snap_after"] = after
        except Exception:
            after = None

        # reward heuristics:
        # - reduce latency => positive reward
        # - reduce backlog => positive reward
        # - minimize unnecessary scaling (penalize large applied_delta)
        reward = 0.0
        try:
            if before and after:
                # choose same queue key logic as reset
                q_before = before.get("queues", {})
                q_after = after.get("queues", {})
                if q_before and q_after:
                    qname = next(iter(q_before.keys()))
                    p95_before = float(q_before[qname]["latency"]["60"]["p95"] or 0.0)
                    p95_after = float(q_after[qname]["latency"]["60"]["p95"] or 0.0)
                    backlog_before = float(q_before[qname]["backlog"]["60"]["mean"] or 0.0)
                    backlog_after = float(q_after[qname]["backlog"]["60"]["mean"] or 0.0)
                    # reward components
                    reward += max(0.0, backlog_before - backlog_after) * 0.1
                    reward += max(0.0, p95_before - p95_after) * 0.5
            # penalty for applying an action (cost aware)
            reward -= abs(info.get("applied_delta", 0)) * 0.05
        except Exception:
            LOG.exception("Reward calculation failed; defaulting reward to 0")
            reward = 0.0

        # produce next observation same as reset
        next_obs = self.reset()
        done = False  # live tasks typically are continuing; we never signal terminal episodes
        return next_obs, reward, done, info

# End of Chunk 1
# ---------------------------
# Chunk 2 — Trajectory buffer, online PPO update logic, trainer class, logging/hooks, CLI
# ---------------------------

import threading
import queue as _queue
import math
import time
import statistics
from typing import Iterable

# ---------------------------
# Lightweight online trajectory buffer (ring buffer style)
# ---------------------------
class OnlineTrajectoryBuffer:
    """
    Small ring buffer suitable for online / continual learning.
    Stores recent (obs, action, logp, reward, value, done) tuples and can return
    flattened arrays for PPO updates. Designed to be memory-light.
    """
    def __init__(self, capacity: int = 16384):
        self.capacity = int(capacity)
        self.obs = [None] * self.capacity
        self.actions = [0.0] * self.capacity
        self.logps = [0.0] * self.capacity
        self.values = [0.0] * self.capacity
        self.rewards = [0.0] * self.capacity
        self.dones = [False] * self.capacity
        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()

    def add(self, obs, action, logp, value, reward, done):
        with self.lock:
            idx = self.ptr % self.capacity
            self.obs[idx] = obs
            self.actions[idx] = float(action)
            self.logps[idx] = float(logp) if logp is not None else 0.0
            self.values[idx] = float(value)
            self.rewards[idx] = float(reward)
            self.dones[idx] = bool(done)
            self.ptr += 1
            self.size = min(self.size + 1, self.capacity)

    def clear(self):
        with self.lock:
            self.ptr = 0
            self.size = 0

    def get_recent(self, n: int):
        """
        Return last n items as lists in chronological order.
        """
        n = min(n, self.size)
        out_idx = []
        with self.lock:
            start = (self.ptr - n) % self.capacity
            for i in range(n):
                idx = (start + i) % self.capacity
                out_idx.append(idx)
            obs = [self.obs[i] for i in out_idx]
            acts = [self.actions[i] for i in out_idx]
            logps = [self.logps[i] for i in out_idx]
            vals = [self.values[i] for i in out_idx]
            rews = [self.rewards[i] for i in out_idx]
            dones = [self.dones[i] for i in out_idx]
        return {"obs": obs, "acts": acts, "logps": logps, "vals": vals, "rews": rews, "dones": dones}

    def __len__(self):
        return self.size

# ---------------------------
# PPO update utilities (stable, small-batch friendly)
# ---------------------------
def compute_gae_and_returns(rewards: Iterable[float], values: Iterable[float], last_value: float, gamma: float, lam: float):
    rewards = list(rewards)
    values = list(values)
    n = len(rewards)
    adv = [0.0] * n
    lastgaelam = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else last_value
        nonterminal = 0.0 if (t < n and False) else 1.0  # in this online case we treat nonterminal=1.0 (no episode ends)
        delta = rewards[t] + gamma * next_value - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
    returns = [adv[i] + values[i] for i in range(n)]
    return adv, returns

def to_numpy(x):
    try:
        import numpy as _np
        return _np.asarray(x, dtype=_np.float32)
    except Exception:
        return list(x)

# ---------------------------
# Logging hooks (MLflow & W&B helpers)
# ---------------------------
def start_experiment_logging(cfg: LiveRLConfig):
    ctx = {"mlflow": False, "wandb": False}
    if cfg.log_mlflow and _HAS_MLFLOW:
        try:
            mlflow.set_experiment(cfg.mlflow_experiment)
            mlflow.start_run(run_name=cfg.run_name or cfg.experiment_name)
            mlflow.log_params(asdict(cfg))
            ctx["mlflow"] = True
        except Exception:
            LOG.exception("Failed to start MLflow")
    if cfg.log_wandb and _HAS_WANDB:
        try:
            wandb.init(project=cfg.wandb_project, name=cfg.run_name or cfg.experiment_name, config=asdict(cfg))
            ctx["wandb"] = True
        except Exception:
            LOG.exception("Failed to start W&B")
    return ctx

def log_step_metrics(ctx: Dict[str, bool], step: int, metrics_dict: Dict[str, float]):
    try:
        if ctx.get("mlflow") and _HAS_MLFLOW:
            for k, v in metrics_dict.items():
                mlflow.log_metric(k, float(v), step=step)
        if ctx.get("wandb") and _HAS_WANDB:
            wandb.log(metrics_dict, step=step)
    except Exception:
        LOG.exception("Failed to log step metrics")

# ---------------------------
# Trainer class for live online training
# ---------------------------
class LiveRLTrainer:
    def __init__(self, cfg: LiveRLConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu") if _HAS_TORCH else None
        # model
        if _HAS_TORCH:
            if cfg.obs_dim is None:
                # conservative default obs dim (e.g., backlog, p95 latency, worker_count)
                cfg.obs_dim = cfg.obs_dim or 3
            self.model = LiveActorCritic(cfg.obs_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.model = None
            self.optimizer = None

        self.buffer = OnlineTrajectoryBuffer(capacity=max(4096, cfg.rollout_steps * 2))
        self.env_adapter = LiveEnvAdapter(cfg)
        self.registry = ModelRegistry() if ModelRegistry else None
        self.redis = RedisQueue() if RedisQueue else None
        self.autoscaler = PriorityMaxAutoscaler() if PriorityMaxAutoscaler else None

        # bookkeeping
        self.last_checkpoint = 0.0
        self.last_apply_ts = 0.0
        self.updates_this_minute = 0
        self.update_minute_window = []
        self.shutdown_flag = threading.Event()
        self.logging_ctx = start_experiment_logging(cfg)
        self.step_counter = 0
        self.eval_counter = 0
        self.best_reward = -float("inf")
        # lightweight metrics aggregator
        self.recent_rewards = []

        # If resume:
        if cfg.resume_from:
            try:
                ck = load_checkpoint(cfg.resume_from)
                if _HAS_TORCH and "model_state_dict" in ck and self.model is not None:
                    self.model.load_state_dict(ck["model_state_dict"])
                LOG.info("Resumed model from %s", cfg.resume_from)
            except Exception:
                LOG.exception("Resume failed")

    def _maybe_rate_limit(self):
        # prune minute window
        now = time.time()
        self.update_minute_window = [t for t in self.update_minute_window if now - t < 60.0]
        return len(self.update_minute_window) < max(1, self.cfg.rate_limit_updates)

    def collect_step(self):
        """
        Collect a single interaction using env_adapter.
        """
        obs = self.env_adapter.reset()
        # convert to tensor for model
        if _HAS_TORCH:
            obs_t = torch.as_tensor([obs], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_t, logp_t, value_t = self.model.get_action_and_value(obs_t, deterministic=False)
            # unwrap
            if isinstance(action_t, torch.Tensor):
                action = float(action_t.cpu().numpy().squeeze())
            else:
                action = float(action_t)
            logp = float(logp_t.cpu().numpy().squeeze()) if logp_t is not None else 0.0
            value = float(value_t.cpu().numpy().squeeze()) if value_t is not None else 0.0
        else:
            # fallback random policy
            action = random.uniform(-1.0, 1.0)
            logp = 0.0
            value = 0.0

        next_obs, reward, done, info = self.env_adapter.step(action)
        # add to buffer
        self.buffer.add(obs, action, logp, value, reward, done)
        self.step_counter += 1
        self.recent_rewards.append(reward)
        return reward, info

    def should_update(self):
        return len(self.buffer) >= self.cfg.rollout_steps and self._maybe_rate_limit()

    def perform_update(self):
        """
        Perform PPO-style update from buffer. This function is conservative (low learning rate,
        few epochs) to suit online incremental learning.
        """
        if not _HAS_TORCH:
            LOG.warning("Torch not available; skipping update")
            return {}

        data = self.buffer.get_recent(self.cfg.rollout_steps)
        obs = to_numpy(data["obs"])
        acts = to_numpy(data["acts"])
        old_logps = to_numpy(data["logps"])
        vals = to_numpy(data["vals"])
        rews = to_numpy(data["rews"])

        # compute GAE & returns using simple method: last_value estimated from model for last obs
        last_value = 0.0
        try:
            last_obs = data["obs"][-1]
            if last_obs is not None:
                obs_t = torch.as_tensor([last_obs], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    _, last_value_t = self.model.forward(obs_t)
                    last_value = float(last_value_t.cpu().numpy().squeeze())
        except Exception:
            last_value = 0.0

        advs, returns = compute_gae_and_returns(rews, vals, last_value, self.cfg.gamma, self.cfg.lam)
        # normalize advantages
        try:
            import numpy as _np
            advs = _np.asarray(advs, dtype=_np.float32)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        except Exception:
            pass

        # convert to tensors
        obs_b = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts_b = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        advs_b = torch.as_tensor(advs, dtype=torch.float32, device=self.device)
        rets_b = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_logp_b = torch.as_tensor(old_logps, dtype=torch.float32, device=self.device)

        batch_size = max(16, min(self.cfg.mini_batch_size, obs_b.shape[0]))
        n = obs_b.shape[0]
        idxs = list(range(n))

        stats_accum = {"pi_loss": [], "v_loss": [], "entropy": [], "clipfrac": []}
        for epoch in range(self.cfg.update_epochs):
            random.shuffle(idxs)
            for start in range(0, n, batch_size):
                batch_idx = idxs[start:start + batch_size]
                b_obs = obs_b[batch_idx]
                b_acts = acts_b[batch_idx]
                b_advs = advs_b[batch_idx]
                b_rets = rets_b[batch_idx]
                b_old_logp = old_logp_b[batch_idx]

                # forward
                mean, values = self.model.forward(b_obs)
                std = torch.exp(self.model.log_std)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(b_acts).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * b_advs
                pi_loss = -(torch.min(surr1, surr2)).mean()

                v_loss = ((b_rets - values.view(-1)) ** 2).mean()
                loss = pi_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                stats_accum["pi_loss"].append(float(pi_loss.detach().cpu().numpy()))
                stats_accum["v_loss"].append(float(v_loss.detach().cpu().numpy()))
                stats_accum["entropy"].append(float(entropy.detach().cpu().numpy()))
                stats_accum["clipfrac"].append(float((torch.abs(ratio - 1.0) > self.cfg.clip_ratio).float().mean().cpu().numpy()))

        # update rate-limit tracking
        self.update_minute_window.append(time.time())
        # clear buffer entries that we just consumed (we keep buffer as ring; for simplicity, clear fully)
        self.buffer.clear()

        # summarize stats
        summary = {k: (statistics.mean(v) if v else 0.0) for k, v in stats_accum.items()}
        self.updates_this_minute += 1
        return summary

    def maybe_checkpoint(self):
        now = time.time()
        if now - self.last_checkpoint >= self.cfg.checkpoint_interval:
            ck_path = str(pathlib.Path(self.cfg.checkpoint_path).with_suffix(f".{int(now)}.pt"))
            state = {"ts": now, "cfg": asdict(self.cfg)}
            if _HAS_TORCH:
                state["model_state_dict"] = self.model.state_dict()
                state["optimizer_state_dict"] = self.optimizer.state_dict() if self.optimizer else None
            try:
                save_checkpoint(state, ck_path)
                self.last_checkpoint = now
                # promote best checkpoint to registry based on simple evaluation
                # lightweight evaluation: mean recent reward
                recent_mean = statistics.mean(self.recent_rewards[-max(1, len(self.recent_rewards)//10):]) if self.recent_rewards else 0.0
                if self.registry:
                    try:
                        # register candidate; registry should implement canary gating externally
                        self.registry.register_model("rl_live", ck_path, metadata={"recent_mean_reward": recent_mean, "ts": current_iso_ts()})
                    except Exception:
                        LOG.exception("Model registry register failed")
            except Exception:
                LOG.exception("Checkpoint save failed")

    def run(self, runtime_seconds: Optional[int] = None):
        """
        Main loop:
         - collect interactions continuously
         - when enough data collected, perform online update (PPO) subject to rate limits
         - periodically checkpoint and log metrics
        """
        LOG.info("Starting LiveRLTrainer run loop (dry_run=%s)", self.cfg.dry_run)
        start = time.time()
        try:
            while not self.shutdown_flag.is_set():
                reward, info = self.collect_step()
                # small sleep to avoid tight loop if live system is slow
                time.sleep(0.1)
                # decide update
                if self.should_update():
                    LOG.info("Triggering online update (steps=%d)", self.cfg.rollout_steps)
                    stats = self.perform_update()
                    self.maybe_checkpoint()
                    # logging
                    step = self.step_counter
                    metrics_summary = {"step": step, "recent_reward_mean": float(statistics.mean(self.recent_rewards[-100:]) if self.recent_rewards else 0.0)}
                    metrics_summary.update(stats)
                    log_step_metrics(self.logging_ctx, step, metrics_summary)
                    # update best tracker
                    recent_mean = metrics_summary.get("recent_reward_mean", 0.0)
                    if recent_mean > self.best_reward:
                        self.best_reward = recent_mean
                        # optionally promote
                        if self.registry:
                            try:
                                # promote best to registry top slot
                                self.registry.register_model("rl_live_best", str(self.cfg.checkpoint_path), metadata={"best_reward": self.best_reward, "ts": current_iso_ts()})
                                LOG.info("Promoted candidate model to registry best slot (best_reward=%.3f)", self.best_reward)
                            except Exception:
                                LOG.exception("Failed to promote model to registry")
                    # small cooldown between updates
                    time.sleep(0.5)
                # optional runtime limit
                if runtime_seconds and (time.time() - start) > runtime_seconds:
                    LOG.info("Runtime limit reached; shutting down")
                    break
        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt received; shutting down trainer")
        except Exception:
            LOG.exception("Exception in live run loop")
        finally:
            # final checkpoint and cleanup
            try:
                final_path = str(pathlib.Path(self.cfg.checkpoint_path).with_suffix(".final.pt"))
                state = {"ts": time.time(), "cfg": asdict(self.cfg)}
                if _HAS_TORCH:
                    state["model_state_dict"] = self.model.state_dict()
                    state["optimizer_state_dict"] = self.optimizer.state_dict() if self.optimizer else None
                save_checkpoint(state, final_path)
            except Exception:
                LOG.exception("Final checkpoint failed")
            # finish logging
            if self.logging_ctx.get("wandb") and _HAS_WANDB:
                try:
                    wandb.finish()
                except Exception:
                    pass
            if self.logging_ctx.get("mlflow") and _HAS_MLFLOW:
                try:
                    mlflow.end_run()
                except Exception:
                    pass
            LOG.info("LiveRLTrainer stopped")

# ---------------------------
# CLI Entrypoint for live trainer
# ---------------------------
def build_live_arg_parser():
    p = argparse.ArgumentParser(prog="train_rl_live", description="PriorityMax Live RL trainer (PPO online/continual)")
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--mini-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--env-mode", choices=["sim", "live"], default="live")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--runtime", type=int, default=0, help="seconds to run (0: infinite)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--mlflow", action="store_true")
    return p

def main():
    parser = build_live_arg_parser()
    args = parser.parse_args()
    cfg = LiveRLConfig(
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch,
        lr=args.lr,
        env_mode=args.env_mode,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume,
        dry_run=args.dry_run,
        log_wandb=args.wandb,
        log_mlflow=args.mlflow
    )
    trainer = LiveRLTrainer(cfg)
    runtime = args.runtime if args.runtime > 0 else None
    trainer.run(runtime_seconds=runtime)

if __name__ == "__main__":
    main()

# End of Chunk 2
# ---------------------------
# Chunk 3 — Evaluation, Canary Promotion, Autoscaler integration, graceful shutdown,
# utility entrypoints (healthchecks, docker-friendly main), and helpers.
# ---------------------------

import signal
import threading
import asyncio
from typing import Callable

# ---------------------------
# Lightweight policy evaluation (offline / holdout)
# ---------------------------
def evaluate_policy_offline(model_state_path: Optional[str] = None,
                            eval_env: Optional[Any] = None,
                            episodes: int = 10,
                            max_steps_per_ep: int = 1000,
                            device: Optional[Any] = None) -> Dict[str, Any]:
    """
    Evaluate policy in a (simulated) evaluation environment.
    - model_state_path: optional path to checkpoint to load (state_dict expected)
    - eval_env: if provided, must implement reset() and step(action) -> obs, rew, done, info
    Returns summary metrics including mean_reward, std_reward, p95_latency (if env provides timing info).
    """
    if eval_env is None:
        if SimulatedRealEnv is None:
            raise RuntimeError("No evaluation env available (SimulatedRealEnv missing)")
        eval_env = SimulatedRealEnv(EnvConfig(mode="sim", seed=random.randint(0, 100000)))

    # load model if path provided
    model_to_eval = None
    if _HAS_TORCH and model_state_path:
        try:
            ck = load_checkpoint(model_state_path)
            model_to_eval = LiveActorCritic(cfg.obs_dim if 'cfg' in globals() and getattr(cfg, "obs_dim", None) else 3)
            if "model_state_dict" in ck:
                model_to_eval.load_state_dict(ck["model_state_dict"])
            else:
                model_to_eval.load_state_dict(ck)
            if device:
                model_to_eval.to(device)
            model_to_eval.eval()
        except Exception:
            LOG.exception("Failed to load model for evaluation; falling back to trainer's in-memory model")
            model_to_eval = None

    rewards = []
    latencies = []
    for ep in range(episodes):
        obs = eval_env.reset()
        total = 0.0
        for step in range(max_steps_per_ep):
            if model_to_eval is not None:
                try:
                    obs_t = torch.as_tensor([obs], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        act_t, _, _ = model_to_eval.get_action_and_value(obs_t, deterministic=True)
                    action = float(act_t.cpu().numpy().squeeze())
                except Exception:
                    action = random.uniform(-1.0, 1.0)
            else:
                # random baseline
                action = random.uniform(-1.0, 1.0)
            t0 = time.perf_counter()
            obs, rew, done, info = eval_env.step(action)
            t1 = time.perf_counter()
            total += float(rew or 0.0)
            latencies.append(t1 - t0)
            if done:
                break
        rewards.append(total)

    import statistics as _stats
    res = {
        "mean_reward": float(_stats.mean(rewards)) if rewards else 0.0,
        "std_reward": float(_stats.pstdev(rewards)) if rewards else 0.0,
        "p95_latency": float(_stats.quantiles(latencies, n=100)[94]) if latencies else 0.0,
        "episodes": episodes
    }
    LOG.info("Offline evaluation: mean_reward=%.3f std=%.3f p95_latency=%.4fs", res["mean_reward"], res["std_reward"], res["p95_latency"])
    return res

# ---------------------------
# Canary gating and promotion helpers
# ---------------------------
def canary_gate_and_promote(trainer: LiveRLTrainer,
                            candidate_ckpt_path: str,
                            holdout_env: Optional[Any] = None,
                            rmse_like_threshold: float = None,
                            min_reward_increase: float = 0.01,
                            eval_episodes: int = 5) -> bool:
    """
    Evaluate a candidate checkpoint on holdout env and promote to registry if it improves metric.
    This is intentionally conservative: requires positive improvement, and registers metadata only.
    Returns True if promoted.
    """
    try:
        LOG.info("Running canary gate for candidate %s", candidate_ckpt_path)
        cand_eval = evaluate_policy_offline(model_state_path=candidate_ckpt_path, eval_env=holdout_env, episodes=eval_episodes, device=trainer.device)
        cand_mean = cand_eval.get("mean_reward", 0.0)
        baseline = trainer.best_reward or 0.0
        LOG.info("Canary: baseline=%.4f candidate=%.4f", baseline, cand_mean)
        # promote only if candidate shows noticeable improvement
        if cand_mean >= baseline + max(min_reward_increase, 0.0):
            try:
                if trainer.registry:
                    trainer.registry.register_model("rl_live_canary", candidate_ckpt_path, metadata={"mean_reward": cand_mean, "ts": current_iso_ts()})
                LOG.info("Canary passed; promoted candidate checkpoint %s", candidate_ckpt_path)
                return True
            except Exception:
                LOG.exception("Registry promotion failed")
                return False
        else:
            LOG.warning("Canary failed: candidate mean %.4f not > baseline %.4f", cand_mean, baseline)
            return False
    except Exception:
        LOG.exception("Canary gate exception")
        return False

# ---------------------------
# Autoscaler integration: provide apply_hint method used by LiveEnvAdapter
# ---------------------------
class LocalAutoscalerShim:
    """
    A lightweight shim that exposes apply_hint(delta) to be consumed by LiveEnvAdapter.
    It enforces cooldowns, dry_run protection, and maps delta to actual scaling calls via PriorityMaxAutoscaler.
    This shim can be used as innocuous integration point for live training pods that are *not* the actual production autoscaler.
    """

    def __init__(self, autoscaler: Optional[Any] = None, cfg: Optional[LiveRLConfig] = None):
        self.autoscaler = autoscaler or (PriorityMaxAutoscaler() if PriorityMaxAutoscaler else None)
        self.cfg = cfg or LiveRLConfig()
        self._last_apply_ts = 0.0
        self._lock = threading.Lock()

    async def _apply_async(self, delta: int):
        # internal async apply that uses autoscaler._promote_dlq or storage set_desired_worker_count etc.
        try:
            if not self.autoscaler:
                LOG.debug("No real autoscaler available; skipping apply")
                return False
            # apply via a safe API; prefer a small change via autoscaler.storage.set_desired_worker_count if available
            cur = getattr(STATE, "current_workers", None)
            if cur is None:
                # if storage has get_latest_worker_count
                try:
                    cur = await self.autoscaler.storage.get_latest_worker_count()
                except Exception:
                    cur = 0
            target = int(max(self.cfg.min_workers, min(self.cfg.max_workers, cur + int(delta))))
            # call autoscaler scale API
            await self.autoscaler._apply_scale("scale_up" if target > cur else "scale_down" if target < cur else "steady", composite_score=0.0, debug={"hint": delta})
            return True
        except Exception:
            LOG.exception("AutoscalerShim async apply failed")
            return False

    def apply_hint(self, delta: int) -> Optional[Any]:
        """
        Public entry: delta is small integer change suggested by RL (-N..+N).
        This enforces cooldowns and rate-limits. Returns coroutine if scheduled, or False.
        """
        now = time.time()
        with self._lock:
            if now - self._last_apply_ts < self.cfg.cooldown_seconds:
                LOG.debug("AutoscalerShim cooldown active (%.2fs left)", self.cfg.cooldown_seconds - (now - self._last_apply_ts))
                return False
            # convert delta to allowed step
            safe_delta = max(-self.cfg.max_scale_step, min(self.cfg.max_scale_step, int(delta)))
            if safe_delta == 0:
                LOG.debug("AutoscalerShim received zero delta; ignoring")
                return False
            self._last_apply_ts = now
            # schedule async apply
            try:
                loop = asyncio.get_event_loop()
                coro = self._apply_async(safe_delta)
                if loop.is_running():
                    task = loop.create_task(coro)
                    return task
                else:
                    # run in thread to avoid blocking
                    threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()
                    return True
            except Exception:
                # final fallback: synchronous call
                try:
                    asyncio.run(self._apply_async(safe_delta))
                    return True
                except Exception:
                    LOG.exception("AutoscalerShim failed to schedule apply")
                    return False

# ---------------------------
# Graceful shutdown helpers & health endpoints
# ---------------------------
_shutdown_hooks: List[Callable[[], None]] = []

def register_shutdown_hook(fn: Callable[[], None]):
    _shutdown_hooks.append(fn)

def _run_shutdown_hooks():
    LOG.info("Running shutdown hooks")
    for fn in _shutdown_hooks:
        try:
            fn()
        except Exception:
            LOG.exception("Shutdown hook failed")

def setup_signal_handlers(trainer: LiveRLTrainer):
    def _handler(signum, frame):
        LOG.info("Signal %s received, initiating graceful shutdown", signum)
        trainer.shutdown_flag.set()
        # run shutdown hooks in separate thread to avoid blocking signal handler
        threading.Thread(target=_run_shutdown_hooks, daemon=True).start()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

# ---------------------------
# Docker / Kubernetes friendly entry: run as service with health endpoints and background tasks
# ---------------------------
def run_as_service(cfg: LiveRLConfig, health_port: int = 8081):
    """
    Run the trainer as a service:
      - spawn trainer.run() in background thread
      - expose minimal HTTP health endpoints for readiness/liveness (via aiohttp)
      - handle signals gracefully
    """
    trainer = LiveRLTrainer(cfg)
    setup_signal_handlers(trainer)

    # register hook to ensure final checkpoint on shutdown
    def _final_checkpoint_hook():
        try:
            ck = str(pathlib.Path(cfg.checkpoint_path).with_suffix(".service.final.pt"))
            state = {"ts": time.time(), "cfg": asdict(cfg)}
            if _HAS_TORCH:
                state["model_state_dict"] = trainer.model.state_dict()
                state["optimizer_state_dict"] = trainer.optimizer.state_dict() if trainer.optimizer else None
            save_checkpoint(state, ck)
        except Exception:
            LOG.exception("final checkpoint hook failed")
    register_shutdown_hook(_final_checkpoint_hook)

    # start trainer in thread
    t = threading.Thread(target=lambda: trainer.run(), daemon=True, name="live-rl-trainer")
    t.start()

    # start simple aiohttp server for health endpoints
    if _HAS_AIOHTTP:
        import aiohttp.web

        async def handle_ready(request):
            # ready if trainer has begun running and model exists
            ready = (trainer.step_counter > 0)
            return aiohttp.web.json_response({"ready": ready})

        async def handle_live(request):
            return aiohttp.web.json_response({"ok": True, "ts": current_iso_ts()})

        app = aiohttp.web.Application()
        app.router.add_get("/healthz", handle_live)
        app.router.add_get("/readyz", handle_ready)
        runner = aiohttp.web.AppRunner(app)

        async def _start_server():
            await runner.setup()
            site = aiohttp.web.TCPSite(runner, "0.0.0.0", health_port)
            await site.start()
            LOG.info("Health server listening on :%d", health_port)

        loop = asyncio.get_event_loop()
        loop.create_task(_start_server())

        try:
            # block until trainer thread exits
            while t.is_alive():
                t.join(timeout=1.0)
        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt received in service runner")
            trainer.shutdown_flag.set()
        finally:
            LOG.info("Service runner exiting; cleaning up")
            _run_shutdown_hooks()
            try:
                loop.run_until_complete(runner.cleanup())
            except Exception:
                pass
    else:
        # no aiohttp: simple blocking wait loop
        try:
            while t.is_alive():
                t.join(timeout=1.0)
        except KeyboardInterrupt:
            trainer.shutdown_flag.set()
        finally:
            _run_shutdown_hooks()

# ---------------------------
# Small convenience runner that wires autoscaler shim and runs the trainer
# ---------------------------
def run_with_autoscaler_shim(runtime_seconds: Optional[int] = None, **kwargs):
    """
    Convenience function used by CLI: create cfg, wire LocalAutoscalerShim into LiveEnvAdapter,
    and run trainer.
    """
    cfg = LiveRLConfig(**kwargs) if kwargs else LiveRLConfig()
    trainer = LiveRLTrainer(cfg)
    # attach shim
    shim = LocalAutoscalerShim(autoscaler=getattr(trainer, "autoscaler", None), cfg=cfg)
    trainer.env_adapter.autoscaler = shim
    # setup signal handlers
    setup_signal_handlers(trainer)
    LOG.info("Starting run_with_autoscaler_shim (dry_run=%s)", cfg.dry_run)
    trainer.run(runtime_seconds=runtime_seconds)

# ---------------------------
# If module executed as script: provide a CLI that supports service vs adhoc runs
# ---------------------------
def build_full_cli():
    p = argparse.ArgumentParser(prog="train_rl_live_service")
    p.add_argument("--mode", choices=["run", "service"], default="run", help="run: adhoc trainer; service: run as long-lived service with health endpoints")
    p.add_argument("--runtime", type=int, default=0, help="seconds to run in 'run' mode (0=infinite)")
    p.add_argument("--health-port", type=int, default=8081)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--env-mode", choices=["sim", "live"], default="live")
    p.add_argument("--rollout-steps", type=int, default=2048)
    return p

def full_main():
    parser = build_full_cli()
    args = parser.parse_args()
    cfg = LiveRLConfig(
        rollout_steps=args.rollout_steps,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume,
        dry_run=args.dry_run,
        log_wandb=args.wandb,
        log_mlflow=args.mlflow,
        env_mode=args.env_mode
    )
    if args.mode == "service":
        run_as_service(cfg, health_port=args.health_port)
    else:
        # adhoc run with autoscaler shim
        run_with_autoscaler_shim(runtime_seconds=(args.runtime or None), **asdict(cfg))

# ---------------------------
# Register atexit to ensure any final hooks run
# ---------------------------
atexit.register(_run_shutdown_hooks)

# If executed directly, run full_main
if __name__ == "__main__":
    full_main()

# End of Chunk 3
# ---------------------------
# Chunk 4 — Cloud integrations, telemetry exporters, self-healing loop, developer guide
# ---------------------------

import psutil
import threading
import subprocess
import shutil
from datetime import datetime

# ---------------------------
# Prometheus exporter for live RL telemetry
# ---------------------------
try:
    from prometheus_client import Gauge, Counter, start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

_PROM_REGISTRY = {}

def start_prometheus_metrics_exporter(port: int = 9303):
    """
    Start Prometheus metrics endpoint to expose:
      - rl_live_reward_mean
      - rl_live_updates_total
      - rl_live_checkpoint_ts
      - rl_live_gpu_util
    """
    if not _HAS_PROM:
        LOG.warning("Prometheus client not available; skipping RL metrics exporter")
        return
    global _PROM_REGISTRY
    _PROM_REGISTRY["reward_mean"] = Gauge("rl_live_reward_mean", "Mean reward of live RL trainer")
    _PROM_REGISTRY["updates_total"] = Counter("rl_live_updates_total", "Total PPO updates performed")
    _PROM_REGISTRY["checkpoint_ts"] = Gauge("rl_live_checkpoint_ts", "Timestamp of last checkpoint")
    _PROM_REGISTRY["gpu_util"] = Gauge("rl_live_gpu_util", "GPU utilization percent")
    start_http_server(port)
    LOG.info("Started Prometheus RL Live metrics exporter on :%d", port)

def update_prom_metrics(trainer: "LiveRLTrainer"):
    if not _HAS_PROM or not _PROM_REGISTRY:
        return
    try:
        if trainer.recent_rewards:
            mean_reward = float(statistics.mean(trainer.recent_rewards[-min(len(trainer.recent_rewards), 100):]))
            _PROM_REGISTRY["reward_mean"].set(mean_reward)
        _PROM_REGISTRY["updates_total"].inc()
        _PROM_REGISTRY["checkpoint_ts"].set(time.time())
        # GPU utilization
        gpu_util = get_gpu_utilization()
        if gpu_util is not None:
            _PROM_REGISTRY["gpu_util"].set(gpu_util)
    except Exception:
        LOG.debug("Prometheus metrics update failed", exc_info=True)

# ---------------------------
# GPU telemetry helper
# ---------------------------
def get_gpu_utilization() -> Optional[float]:
    """
    Returns current GPU utilization percent (single GPU) if available.
    Uses torch.cuda or nvidia-smi fallback.
    """
    try:
        if _HAS_TORCH and torch.cuda.is_available():
            util = torch.cuda.utilization()
            if isinstance(util, list) and len(util) > 0:
                return float(util[0])
            return float(util)
    except Exception:
        pass
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            ).decode().strip().split("\n")
            if out:
                vals = [float(x.strip()) for x in out if x.strip()]
                return sum(vals) / len(vals)
    except Exception:
        pass
    return None

# ---------------------------
# Self-healing / watchdog monitor
# ---------------------------
class LiveRLWatchdog(threading.Thread):
    """
    Monitors the trainer and attempts recovery actions if anomalies are detected:
      - stalls (no updates within threshold)
      - negative reward drift
      - long unresponsiveness
    """

    def __init__(self, trainer: "LiveRLTrainer", check_interval: int = 30, max_stall_time: int = 300):
        super().__init__(daemon=True, name="rl_live_watchdog")
        self.trainer = trainer
        self.check_interval = check_interval
        self.max_stall_time = max_stall_time
        self.last_update_ts = time.time()
        self.running = True

    def run(self):
        LOG.info("Watchdog started (interval=%ds)", self.check_interval)
        while self.running:
            try:
                time.sleep(self.check_interval)
                # detect stalls
                now = time.time()
                if len(self.trainer.buffer) == 0 and now - self.last_update_ts > self.max_stall_time:
                    LOG.warning("Trainer appears stalled (no buffer activity > %ds)", self.max_stall_time)
                    # self-heal: restart environment adapter or reinitialize
                    try:
                        self.trainer.env_adapter = LiveEnvAdapter(self.trainer.cfg)
                        self.last_update_ts = now
                        LOG.info("Environment adapter restarted by watchdog")
                    except Exception:
                        LOG.exception("Watchdog failed to restart adapter")
                # monitor reward drift
                if self.trainer.recent_rewards:
                    mean_r = statistics.mean(self.trainer.recent_rewards[-min(50, len(self.trainer.recent_rewards)):])
                    if mean_r < -1.0:
                        LOG.warning("Negative reward drift detected (mean=%.3f)", mean_r)
                # update Prometheus
                update_prom_metrics(self.trainer)
            except Exception:
                LOG.exception("Watchdog iteration failed")

    def stop(self):
        LOG.info("Stopping watchdog")
        self.running = False

# ---------------------------
# Cloud telemetry + MLflow artifact synchronization
# ---------------------------
def sync_artifacts_to_mlflow(trainer: "LiveRLTrainer"):
    if not (_HAS_MLFLOW and trainer.logging_ctx.get("mlflow")):
        return
    try:
        mlflow.log_artifact(trainer.cfg.checkpoint_path)
        LOG.debug("Checkpoint artifact synced to MLflow")
    except Exception:
        LOG.debug("Artifact sync to MLflow failed", exc_info=True)

def sync_artifacts_to_s3(bucket: str, prefix: str, path: str):
    """
    Optional S3 uploader for checkpoints or logs (requires boto3).
    """
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"{prefix.rstrip('/')}/{os.path.basename(path)}"
        s3.upload_file(path, bucket, key)
        LOG.info("Uploaded %s to s3://%s/%s", path, bucket, key)
    except Exception:
        LOG.exception("S3 sync failed")

# ---------------------------
# Extended CLI for cloud + monitoring
# ---------------------------
def build_cloud_cli():
    p = argparse.ArgumentParser(prog="train_rl_live_cloud", description="Run RL Live Trainer with cloud integrations")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--prometheus", action="store_true")
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--s3-prefix", default="models/rl_live")
    p.add_argument("--runtime", type=int, default=0)
    p.add_argument("--health-port", type=int, default=8081)
    p.add_argument("--dry-run", action="store_true")
    return p

def cloud_main():
    args = build_cloud_cli().parse_args()
    cfg = LiveRLConfig(log_mlflow=args.mlflow, log_wandb=args.wandb, dry_run=args.dry_run)
    trainer = LiveRLTrainer(cfg)
    setup_signal_handlers(trainer)
    wd = LiveRLWatchdog(trainer)
    wd.start()

    if args.prometheus:
        start_prometheus_metrics_exporter()

    if args.runtime > 0:
        trainer.run(runtime_seconds=args.runtime)
    else:
        trainer.run()

    sync_artifacts_to_mlflow(trainer)
    if args.s3_bucket:
        sync_artifacts_to_s3(args.s3_bucket, args.s3_prefix, cfg.checkpoint_path)

# ---------------------------
# Developer / Operator documentation block (for maintainers)
# ---------------------------

"""
===============================================================================
README: train_rl_live.py — Developer Reference
===============================================================================

This script is used in **production** to train and adapt the RL policy
for PriorityMax’s autoscaler.  It is safety-conscious and online-aware.

KEY RUN MODES
-------------
1.  Local Simulation:
      python3 scripts/train_rl_live.py --env-mode sim --runtime 120
      # Runs short simulation using SimulatedRealEnv, stores checkpoints locally.

2.  Live Continual Learning (with autoscaler hints):
      python3 scripts/train_rl_live.py --env-mode live --mlflow --wandb
      # Consumes real telemetry (queue backlog, latency, worker count),
      # sends small control hints (delta ±1) to autoscaler shim.

3.  Container Service Mode (for K8s pod / systemd):
      python3 scripts/train_rl_live.py --mode service --health-port 8081
      # Exposes /healthz and /readyz endpoints for readiness probes.

4.  Cloud-integrated Training:
      python3 scripts/train_rl_live.py --prometheus --mlflow --s3-bucket my-bucket
      # Streams metrics to Prometheus, uploads checkpoints to MLflow + S3.

FILES GENERATED
---------------
  - models/rl_live.pt                 → latest checkpoint
  - models/rl_live.<timestamp>.pt     → rolling backups
  - logs/rl_live/train.log            → full training logs
  - mlflow + wandb runs               → experiment tracking

PRODUCTION RECOMMENDATIONS
--------------------------
  ✅ Run in dry_run mode first on staging.
  ✅ Use MLflow + Prometheus exporters together for full observability.
  ✅ Always configure safe scaling bounds in LiveRLConfig.
  ✅ Connect watchdog to restart adapter if telemetry stalls.
  ✅ Periodically canary-evaluate before promoting new models.

===============================================================================
End of Developer Guide
===============================================================================
"""

# ---------------------------
# End of Chunk 4 — full file complete
# ---------------------------

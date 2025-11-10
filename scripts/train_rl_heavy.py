#!/usr/bin/env python3
# train_rl_heavy.py
"""
Train RL — Heavy / Production Trainer (PPO, multi-GPU ready)
------------------------------------------------------------

Purpose:
 - Train a production-grade PPO policy for PriorityMax autoscaling using
   synthetic and real workload environments.
 - Support multi-GPU / distributed training (torch.distributed and Ray options).
 - Integrated checkpointing, MLflow & W&B experiment logging, evaluation hooks,
   canary gating and model registry registration.

Usage examples:
  # single-node multi-GPU (DDP)
  python3 backend/scripts/train_rl_heavy.py --epochs 200 --steps-per-epoch 8192 --gpus 2 --exp-name pmax_ppo_v1

  # with Ray (if enabled)
  python3 backend/scripts/train_rl_heavy.py --use-ray --ray-address auto --epochs 1000 --gpus 8

Notes:
 - This file is intentionally defensive: it runs even if optional deps missing.
 - For full distributed runs, ensure cluster orchestration (SLURM / k8s / Ray) is configured.
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import shutil
import random
import atexit
import logging
import pathlib
import tempfile
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List, Sequence, Callable

# ---------------------------
# Optional heavy deps (best-effort import)
# ---------------------------
_HAS_TORCH = False
_HAS_RAY = False
_HAS_MLFLOW = False
_HAS_WANDB = False
_HAS_TQDM = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.nn.parallel import DistributedDataParallel as DDP
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    optim = None
    DDP = None

try:
    import ray
    from ray import tune
    _HAS_RAY = True
except Exception:
    ray = None
    tune = None

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
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = lambda x: x  # fallback iterator

# ---------------------------
# Project modules (best-effort)
# ---------------------------
try:
    # Prefer to import from package path if running within project
    from app.ml.real_env import SimulatedRealEnv, EnvConfig, make_vec_env
    from app.ml.rl_agent_sandbox import PPOActorCritic  # optional sandbox model
    from app.ml.model_registry import ModelRegistry
    from app.ml.rl_agent_prod import RLAgentProd
except Exception:
    # local fallbacks (placeholders)
    SimulatedRealEnv = None
    EnvConfig = None
    make_vec_env = None
    PPOActorCritic = None
    ModelRegistry = None
    RLAgentProd = None

# ---------------------------
# Paths & Defaults
# ---------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]  # backend/
DEFAULT_MODELS_DIR = ROOT / "app" / "ml" / "models"
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CKPT = DEFAULT_MODELS_DIR / "rl_agent.pt"
DEFAULT_LOGDIR = ROOT / "logs" / "rl_heavy"
DEFAULT_LOGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------
LOG = logging.getLogger("prioritymax.train_rl_heavy")
LOG.setLevel(os.getenv("PRIORITYMAX_TRAINER_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# ---------------------------
# Dataclasses: configs
# ---------------------------
@dataclass
class HeavyRLConfig:
    # training schedule
    epochs: int = 500
    steps_per_epoch: int = 8192
    update_epochs: int = 10
    mini_batch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # model
    hidden_dim: int = 256
    policy_hidden_layers: Tuple[int, ...] = (256, 256)
    value_hidden_layers: Tuple[int, ...] = (256, 256)
    activation: str = "relu"  # relu | tanh | gelu

    # environment
    env_name: str = "simulated_queue_v1"
    num_envs: int = 16
    env_seed: int = 1234
    obs_dim: Optional[int] = None  # autodetect from env
    act_dim: int = 3

    # devices & distribution
    gpus: int = 1
    use_ddp: bool = False
    use_ray: bool = False
    ray_address: Optional[str] = None

    # logging & experiment
    experiment_name: str = "prioritymax_rl_heavy"
    run_name: Optional[str] = None
    log_wandb: bool = True
    log_mlflow: bool = True
    wandb_project: str = "PriorityMax-RL"
    mlflow_experiment: str = "PriorityMax-RL"

    # checkpointing / evaluation
    checkpoint_dir: str = str(DEFAULT_MODELS_DIR)
    checkpoint_interval: int = 10
    eval_interval: int = 10
    eval_episodes: int = 8
    resume_from: Optional[str] = None

    # safety & pruning
    seed: int = 42
    checkpoint_top_k: int = 5  # keep top-k by metric
    save_best_by: str = "mean_reward"  # metric to sort by

    # misc
    num_workers: int = 4  # dataloader / sampling parallelism
    verbose: bool = True
    dry_run: bool = False

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
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

def ensure_dir(p: str):
    pth = pathlib.Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth

def save_json_atomic(data: Dict[str, Any], path: str):
    p = pathlib.Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)

def current_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def default_run_name(prefix: str = "ppo"):
    return f"{prefix}_{current_time_str()}_{uuid.uuid4().hex[:6]}"

# ---------------------------
# Checkpoint helpers
# ---------------------------
def save_checkpoint(state: Dict[str, Any], path: str, keep_latest: bool = True):
    """
    Save a checkpoint dictionary. For PyTorch model objects we save state_dict.
    """
    p = pathlib.Path(path)
    ensure_dir(str(p.parent))
    try:
        if _HAS_TORCH and "model" in state and isinstance(state["model"], torch.nn.Module):
            # Save model state_dict + optimizer state
            ckpt = state.copy()
            model = ckpt.pop("model")
            ckpt["model_state_dict"] = model.state_dict()
            # remove heavy objects if present
            if "optimizer" in ckpt and hasattr(ckpt["optimizer"], "state_dict"):
                try:
                    ckpt["optimizer_state_dict"] = ckpt["optimizer"].state_dict()
                    ckpt.pop("optimizer", None)
                except Exception:
                    ckpt.pop("optimizer", None)
            torch.save(ckpt, str(p))
        else:
            # generic save via joblib
            import joblib
            joblib.dump(state, str(p))
        LOG.info("Checkpoint written to %s", str(p))
    except Exception:
        LOG.exception("Failed to write checkpoint to %s", str(p))

def load_checkpoint(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    try:
        if _HAS_TORCH:
            ckpt = torch.load(str(p), map_location="cpu")
            return ckpt
    except Exception:
        pass
    import joblib
    return joblib.load(str(p))

# ---------------------------
# Activation selector
# ---------------------------
def select_activation(name: str):
    name = (name or "relu").lower()
    if not _HAS_TORCH:
        return None
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name == "leakyrelu":
        return lambda: nn.LeakyReLU(0.1)
    return nn.ReLU

# ---------------------------
# Small MLP block factory
# ---------------------------
def make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: Optional[int] = None, activation: str = "relu", final_activation: Optional[str] = None):
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for make_mlp")
    layers: List[nn.Module] = []
    act_cls = select_activation(activation)
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(act_cls())
        last = h
    if output_dim is not None:
        layers.append(nn.Linear(last, output_dim))
        if final_activation:
            fac = select_activation(final_activation)
            layers.append(fac())
    return nn.Sequential(*layers)

# ---------------------------
# Actor-Critic Model (configurable)
# ---------------------------
if _HAS_TORCH:
    class ActorCriticNet(nn.Module):
        """
        Configurable actor-critic with separate policy & value heads.
        Policy outputs continuous actions (mean), and optionally logstd parameter per action.
        """
        def __init__(self, obs_dim: int, act_dim: int, policy_hidden: Sequence[int] = (256,256), value_hidden: Sequence[int] = (256,256), activation: str = "relu"):
            super().__init__()
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.policy_net = make_mlp(obs_dim, policy_hidden, output_dim=act_dim, activation=activation)
            self.value_net = make_mlp(obs_dim, value_hidden, output_dim=1, activation=activation)
            # log std parameter (trainable)
            self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Return (action_mean, value)
            """
            action_mean = self.policy_net(obs)
            value = self.value_net(obs).squeeze(-1)
            return action_mean, value

        def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
    ActorCriticNet = None

# ---------------------------
# PPO Trainer core (skeleton)
# ---------------------------
class PPOTrainer:
    """
    High-level PPO trainer abstraction.
    Responsibilities:
      - Manage envs (vectorized)
      - Collect trajectories
      - Compute GAE and advantages
      - Perform minibatch updates
      - Checkpointing, evaluation, and logging
    """
    def __init__(self, cfg: HeavyRLConfig):
        self.cfg = cfg
        self.run_name = cfg.run_name or default_run_name("ppo")
        self.device = torch.device("cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu")
        if _HAS_TORCH and cfg.gpus > 0 and torch.cuda.is_available():
            # if single-node multi-GPU and use_ddp specified, caller must have launched via torch.distributed.launch or similar.
            LOG.info("Torch available. CUDA: %s, device=%s", torch.cuda.is_available(), self.device)
        else:
            LOG.info("Using device: %s", self.device)

        # envs and model will be created lazily
        self.envs = None
        self.model = None
        self.optimizer = None
        self.start_epoch = 1
        self.best_metrics: List[Tuple[str, float]] = []  # list of (path, metric) for top-k retention

        # model registry integration (best-effort)
        self.registry = ModelRegistry() if ModelRegistry else None

    # -------------------------
    # Environment factory
    # -------------------------
    def make_envs(self) -> Any:
        """
        Instantiates vectorized envs. Uses app.ml.make_vec_env if available;
        otherwise falls back to a basic local vectorization.
        """
        LOG.info("Creating %d envs (env_name=%s)", self.cfg.num_envs, self.cfg.env_name)
        if make_vec_env:
            try:
                env = make_vec_env(EnvConfig(mode="sim", seed=self.cfg.env_seed), n=self.cfg.num_envs)
                return env
            except Exception:
                LOG.exception("make_vec_env failed; falling back to simple wrappers")
        # Fallback: create list of independent envs wrapped in a minimal VecEnv shim
        envs = []
        if SimulatedRealEnv:
            for i in range(self.cfg.num_envs):
                envs.append(SimulatedRealEnv(EnvConfig(mode="sim", seed=self.cfg.env_seed + i)))
            return SimpleVecEnv(envs)
        raise RuntimeError("No environment available. Please implement make_vec_env or provide SimulatedRealEnv.")

    # -------------------------
    # Model init
    # -------------------------
    def init_model(self, obs_dim: int, act_dim: int):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required to init model")
        LOG.info("Initializing ActorCriticNet (obs=%d act=%d)", obs_dim, act_dim)
        self.model = ActorCriticNet(obs_dim, act_dim, policy_hidden=self.cfg.policy_hidden_layers, value_hidden=self.cfg.value_hidden_layers, activation=self.cfg.activation)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        # If resume and DDP, wrap model later as part of distributed init
        return self.model

    # -------------------------
    # Distributed setup placeholders
    # -------------------------
    def setup_distributed(self):
        """
        Placeholder for torch.distributed / Ray initialization.
        For DDP, users should launch via torch.distributed.launch or torchrun with WORLD_SIZE and RANK env vars set.
        This function can be extended to run local init_process_group in containerized clusters.
        """
        if not _HAS_TORCH:
            LOG.warning("Torch not available; skipping distributed setup")
            return
        if self.cfg.use_ddp:
            # Basic local init if env var present; else expect external launch
            try:
                if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "MASTER_ADDR" in os.environ:
                    rank = int(os.environ.get("RANK", "0"))
                    world_size = int(os.environ.get("WORLD_SIZE", "1"))
                    backend = os.environ.get("DDP_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
                    LOG.info("Initializing torch.distributed (backend=%s rank=%s world_size=%s)", backend, rank, world_size)
                    torch.distributed.init_process_group(backend=backend)
                    # wrap model in DDP after model created
                else:
                    LOG.info("No distributed env vars found; skipping in-process init. Expect external launcher for DDP.")
            except Exception:
                LOG.exception("Distributed init failed")
        if self.cfg.use_ray and _HAS_RAY:
            try:
                if self.cfg.ray_address:
                    ray.init(address=self.cfg.ray_address, ignore_reinit_error=True)
                else:
                    ray.init(ignore_reinit_error=True)
                LOG.info("Ray initialized (address=%s)", ray._private.services.get_node_ip_address())
            except Exception:
                LOG.exception("Ray initialization failed; continuing without Ray")

# ---------------------------
# Small VecEnv shim (fallback)
# ---------------------------
class SimpleVecEnv:
    """
    Simple synchronous vectorized environment wrapper for a list of envs.
    Provides:
      - reset() -> list of obs
      - step(actions_list) -> next_obs_list, reward_list, done_list, info_list
    """
    def __init__(self, envs: List[Any]):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self, seed: Optional[int] = None):
        obs = []
        for i, e in enumerate(self.envs):
            try:
                if hasattr(e, "reset"):
                    obs.append(e.reset(seed=(seed + i) if seed is not None else None))
                else:
                    obs.append(None)
            except Exception:
                LOG.exception("env reset failed for idx=%d", i)
                obs.append(None)
        return obs

    def step(self, actions: Sequence[Any]):
        next_obs = []
        rewards = []
        dones = []
        infos = []
        for i, e in enumerate(self.envs):
            try:
                a = actions[i]
                o, r, d, info = e.step(a)
                next_obs.append(o)
                rewards.append(r)
                dones.append(d)
                infos.append(info)
                if d:
                    o = e.reset()
            except Exception:
                LOG.exception("env step failed idx=%d", i)
                next_obs.append(None)
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
        return next_obs, rewards, dones, infos

# ---------------------------
# GAE computation (vectorized-friendly)
# ---------------------------
def compute_gae(rewards, values, last_value, gamma: float, lam: float):
    """
    rewards: sequence (T,) or array
    values: sequence (T,) - value estimates for each step
    returns: advantages (T,), returns_to_go (T,)
    Basic implementation for flattened trajectory. For vectorized per-env GAE, compute per-env.
    """
    T = len(rewards)
    adv = [0.0] * T
    lastgaelam = 0.0
    for t in reversed(range(T)):
        next_value = values[t+1] if (t+1) < len(values) else last_value
        delta = rewards[t] + gamma * next_value - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
    returns = [adv[i] + values[i] for i in range(T)]
    return adv, returns

# ---------------------------
# Chunk 2 — Data buffers, rollout collection, PPO update, evaluation, logging hooks
# ---------------------------

import statistics
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

# ---------------------------
# Trajectory Buffer (flattened, supports vectorized envs)
# ---------------------------
class TrajectoryBuffer:
    """
    Stores observations, actions, rewards, values, logps for a single collection cycle.
    Designed for flattened trajectories from vectorized envs.
    After collection, compute advantages (GAE) and returns, then provide minibatches.
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], act_dim: int, device: Optional[Any] = None):
        self.capacity = capacity
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logps = []
        self.dones = []
        self.infos = []
        self.device = device
        self.ptr = 0

    def add(self, obs, action, reward, value, logp, done, info=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.logps.append(float(logp) if logp is not None else 0.0)
        self.dones.append(bool(done))
        self.infos.append(info or {})
        self.ptr += 1

    def size(self):
        return self.ptr

    def clear(self):
        self.__init__(self.capacity, (), 0, device=self.device)

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        advs = [0.0] * self.ptr
        lastgaelam = 0.0
        for t in reversed(range(self.ptr)):
            next_value = self.values[t+1] if t+1 < self.ptr else last_value
            nonterminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            advs[t] = lastgaelam
        returns = [advs[i] + self.values[i] for i in range(self.ptr)]
        # convert to numpy arrays for batching
        try:
            import numpy as _np
            obs_arr = _np.asarray(self.obs, dtype=_np.float32)
            acts_arr = _np.asarray(self.actions, dtype=_np.float32)
            adv_arr = _np.asarray(advs, dtype=_np.float32)
            ret_arr = _np.asarray(returns, dtype=_np.float32)
            logp_arr = _np.asarray(self.logps, dtype=_np.float32)
        except Exception:
            obs_arr = self.obs
            acts_arr = self.actions
            adv_arr = advs
            ret_arr = returns
            logp_arr = self.logps
        # normalize advantages
        try:
            adv_mean = float(adv_arr.mean())
            adv_std = float(adv_arr.std()) if float(adv_arr.std()) > 1e-8 else 1.0
            adv_arr = (adv_arr - adv_mean) / (adv_std + 1e-8)
        except Exception:
            pass
        return {"obs": obs_arr, "acts": acts_arr, "advs": adv_arr, "rets": ret_arr, "logps": logp_arr}

    def minibatch_generator(self, data: Dict[str, Any], batch_size: int, shuffle: bool = True):
        """
        Yields minibatches of the flattened trajectory data.
        """
        n = len(data["advs"])
        indices = list(range(n))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {k: (data[k][batch_idx] if hasattr(data[k], "__getitem__") else [data[k][i] for i in batch_idx]) for k in data}

# ---------------------------
# Utility: safe tensor conversion
# ---------------------------
def to_tensor(x, device=None):
    if not _HAS_TORCH:
        return x
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return torch.from_numpy(x).to(device)
        if isinstance(x, list):
            return torch.tensor(x, dtype=torch.float32, device=device)
        return torch.tensor(x, dtype=torch.float32, device=device)
    except Exception:
        try:
            return torch.tensor(x, dtype=torch.float32, device=device)
        except Exception:
            return x

# ---------------------------
# Rollout collector
# ---------------------------
def collect_rollout(trainer: PPOTrainer, total_steps: int) -> Tuple[TrajectoryBuffer, float]:
    """
    Collect a flattened trajectory of length total_steps from trainer.envs.
    Returns TrajectoryBuffer and last_value for bootstrapping.
    """
    envs = trainer.envs or trainer.make_envs()
    trainer.envs = envs
    num_envs = getattr(envs, "num_envs", len(envs.envs) if hasattr(envs, "envs") else 1)
    buf = TrajectoryBuffer(capacity=total_steps, obs_shape=(trainer.cfg.obs_dim or 0,), act_dim=trainer.cfg.act_dim, device=trainer.device)

    # initial reset
    if hasattr(envs, "reset"):
        obs_list = envs.reset(seed=trainer.cfg.env_seed)
    else:
        obs_list = [None] * num_envs

    steps_collected = 0
    # We'll use a loop that collects actions for each env in sequence (vectorized)
    while steps_collected < total_steps:
        # prepare batch observation tensor
        if _HAS_TORCH:
            obs_tensor = to_tensor(obs_list, device=trainer.device)
            if isinstance(obs_tensor, torch.Tensor) and obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                action_tensor, logp_tensor, value_tensor = trainer.model.get_action(obs_tensor, deterministic=False)
            # ensure shapes
            if isinstance(action_tensor, torch.Tensor):
                actions = action_tensor.cpu().numpy()
            else:
                actions = action_tensor
            if isinstance(logp_tensor, torch.Tensor):
                logps = logp_tensor.cpu().numpy()
            else:
                logps = logp_tensor
            if isinstance(value_tensor, torch.Tensor):
                values = value_tensor.cpu().numpy()
            else:
                values = value_tensor
        else:
            # fallback: random actions
            actions = [0.0] * num_envs
            logps = [0.0] * num_envs
            values = [0.0] * num_envs

        # perform step
        next_obs, rewards, dones, infos = envs.step(actions)
        # store per-env
        for i in range(num_envs):
            buf.add(obs_list[i], actions[i], rewards[i], values[i], logps[i] if logps is not None else 0.0, dones[i], info=infos[i] if infos else {})
            steps_collected += 1
            if steps_collected >= total_steps:
                break
        obs_list = next_obs

    # compute last value bootstrap using last obs
    last_val = 0.0
    try:
        if _HAS_TORCH:
            last_obs_tensor = to_tensor(obs_list, device=trainer.device)
            if isinstance(last_obs_tensor, torch.Tensor):
                with torch.no_grad():
                    _, last_val = trainer.model.forward(last_obs_tensor)
                    if isinstance(last_val, torch.Tensor):
                        last_val = float(last_val.mean().cpu().numpy())
        else:
            last_val = 0.0
    except Exception:
        last_val = 0.0

    return buf, last_val

# ---------------------------
# PPO update step
# ---------------------------
def ppo_update(trainer: PPOTrainer, buffer_data: Dict[str, Any], updates: int):
    """
    Perform PPO policy/value updates using the flattened buffer_data.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for ppo_update")

    model = trainer.model
    optimizer = trainer.optimizer
    cfg = trainer.cfg

    obs_all = buffer_data["obs"]
    acts_all = buffer_data["acts"]
    advs_all = buffer_data["advs"]
    rets_all = buffer_data["rets"]
    old_logps_all = buffer_data["logps"]

    n = len(advs_all)
    minibatch_size = max(1, min(cfg.mini_batch_size, n))

    stats = {"pi_loss": [], "v_loss": [], "entropy": [], "clipfrac": []}

    for ep in range(updates):
        gen = trainer_buf_minibatches(obs_all, acts_all, advs_all, rets_all, old_logps_all, minibatch_size)
        for batch in gen:
            obs_b = to_tensor(batch["obs"], device=trainer.device)
            acts_b = to_tensor(batch["acts"], device=trainer.device)
            advs_b = to_tensor(batch["advs"], device=trainer.device)
            rets_b = to_tensor(batch["rets"], device=trainer.device)
            old_logp_b = to_tensor(batch["logps"], device=trainer.device)

            # forward
            action_mean, value_pred = model.forward(obs_b)
            # compute distribution and logp for actions
            std = torch.exp(model.log_std)
            dist = torch.distributions.Normal(action_mean, std)
            new_logp = dist.log_prob(acts_b).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_logp - old_logp_b)
            surr1 = ratio * advs_b
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advs_b
            pi_loss = -torch.min(surr1, surr2).mean()

            value_pred = value_pred.view(-1)
            v_loss = ((rets_b - value_pred) ** 2).mean()

            loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # stats
            stats["pi_loss"].append(pi_loss.item())
            stats["v_loss"].append(v_loss.item())
            stats["entropy"].append(float(entropy.item()))
            clipfrac = float((torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean().item())
            stats["clipfrac"].append(clipfrac)

    # aggregate stats
    summary = {k: (float(statistics.mean(v)) if v else 0.0) for k, v in stats.items()}
    return summary

# ---------------------------
# Helper: minibatch generator (numpy-friendly)
# ---------------------------
def trainer_buf_minibatches(obs, acts, advs, rets, logps, batch_size):
    try:
        import numpy as _np
        n = len(advs)
        indices = list(range(n))
        random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                "obs": _np.asarray(obs)[batch_idx],
                "acts": _np.asarray(acts)[batch_idx],
                "advs": _np.asarray(advs)[batch_idx],
                "rets": _np.asarray(rets)[batch_idx],
                "logps": _np.asarray(logps)[batch_idx]
            }
    except Exception:
        # fallback lists
        n = len(advs)
        idxs = list(range(n))
        random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start + batch_size]
            yield {
                "obs": [obs[i] for i in batch_idx],
                "acts": [acts[i] for i in batch_idx],
                "advs": [advs[i] for i in batch_idx],
                "rets": [rets[i] for i in batch_idx],
                "logps": [logps[i] for i in batch_idx]
            }

# ---------------------------
# Evaluation routine
# ---------------------------
def evaluate_policy(trainer: PPOTrainer, episodes: int = 8, deterministic: bool = True) -> Dict[str, Any]:
    """
    Run the current policy for a number of episodes in a fresh env and return aggregated metrics.
    """
    # create a separate single env for evaluation
    if SimulatedRealEnv:
        eval_env = SimulatedRealEnv(EnvConfig(mode="sim", seed=trainer.cfg.env_seed + 9999))
    else:
        raise RuntimeError("SimulatedRealEnv not available for evaluation")
    rewards = []
    latencies = []
    backlogs = []
    for ep in range(episodes):
        obs = eval_env.reset(seed=trainer.cfg.env_seed + ep)
        total_reward = 0.0
        for t in range(10000):
            if _HAS_TORCH:
                obs_t = to_tensor(obs, device=trainer.device)
                if isinstance(obs_t, torch.Tensor) and obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = trainer.model.get_action(obs_t, deterministic=deterministic)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()[0]
            else:
                action = 0.0
            obs, rew, done, info = eval_env.step(action)
            total_reward += rew
            # try extract latency/backlog proxies from info
            latencies.append(info.get("latency", 0.0) if isinstance(info, dict) else 0.0)
            backlogs.append(info.get("backlog", 0) if isinstance(info, dict) else 0)
            if done:
                break
        rewards.append(total_reward)
    res = {
        "mean_reward": float(statistics.mean(rewards)) if rewards else 0.0,
        "std_reward": float(statistics.pstdev(rewards)) if rewards else 0.0,
        "mean_latency": float(statistics.mean(latencies)) if latencies else 0.0,
        "p95_latency": float(sorted(latencies)[int(0.95 * len(latencies))]) if latencies else 0.0,
        "mean_backlog": float(statistics.mean(backlogs)) if backlogs else 0.0
    }
    return res

# ---------------------------
# Logging helpers (MLflow & W&B)
# ---------------------------
def init_loggers(cfg: HeavyRLConfig):
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

def log_metrics(ctx: Dict[str, Any], metrics: Dict[str, Any], step: Optional[int] = None):
    try:
        if ctx.get("mlflow") and _HAS_MLFLOW:
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v), step=step)
        if ctx.get("wandb") and _HAS_WANDB:
            wandb.log(metrics, step=step)
    except Exception:
        LOG.exception("Logging metrics failed")

# ---------------------------
# Checkpoint retention & registry promotion
# ---------------------------
def manage_checkpoint_retention(trainer: PPOTrainer, ckpt_path: str, metric_val: float):
    """
    Keep top-k checkpoints and optionally register best model in ModelRegistry.
    """
    # add to list and prune
    trainer.best_metrics.append((ckpt_path, float(metric_val)))
    trainer.best_metrics = sorted(trainer.best_metrics, key=lambda x: x[1], reverse=True)[:trainer.cfg.checkpoint_top_k]
    # remove files not in best_metrics
    keep = {p for p, _ in trainer.best_metrics}
    all_ckpts = list(pathlib.Path(trainer.cfg.checkpoint_dir).glob("rl_ckpt_*.pt"))
    for p in all_ckpts:
        if str(p) not in keep:
            try:
                p.unlink()
            except Exception:
                LOG.debug("Failed to remove old checkpoint %s", p)

    # promote top-1 to registry if available
    if trainer.registry and trainer.best_metrics:
        best_path = trainer.best_metrics[0][0]
        try:
            trainer.registry.register_model("rl_agent", best_path, metadata={"metric": trainer.best_metrics[0][1], "ts": current_time_str()})
            LOG.info("Promoted checkpoint %s to model registry", best_path)
        except Exception:
            LOG.exception("Model registry promotion failed for %s", best_path)

# ---------------------------
# Chunk 3 — Full PPO training loop, evaluation integration, CLI entrypoint
# ---------------------------

def train_heavy(cfg: HeavyRLConfig):
    """
    Core training entrypoint for PPO heavy trainer.
    - Initializes envs, model, optimizer, and loggers
    - Runs collection + PPO update + eval loops
    - Saves checkpoints and logs metrics
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for train_heavy")

    set_seed(cfg.seed)
    LOG.info("Starting heavy PPO training: epochs=%d steps/epoch=%d", cfg.epochs, cfg.steps_per_epoch)
    trainer = PPOTrainer(cfg)
    trainer.setup_distributed()
    ctx = init_loggers(cfg)

    # create envs and model
    envs = trainer.make_envs()
    trainer.envs = envs
    obs_dim = cfg.obs_dim or len(envs.reset()[0]) if hasattr(envs, "reset") else cfg.obs_dim
    act_dim = cfg.act_dim
    trainer.init_model(obs_dim, act_dim)

    best_metric = -float("inf")

    # optional resume
    if cfg.resume_from:
        try:
            ckpt = load_checkpoint(cfg.resume_from)
            if "model_state_dict" in ckpt:
                trainer.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt and trainer.optimizer:
                trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "epoch" in ckpt:
                trainer.start_epoch = ckpt["epoch"] + 1
            LOG.info("Resumed from checkpoint %s (epoch %d)", cfg.resume_from, trainer.start_epoch - 1)
        except Exception:
            LOG.exception("Failed to resume from %s", cfg.resume_from)

    # main training loop
    t0 = time.time()
    for epoch in range(trainer.start_epoch, cfg.epochs + 1):
        # collect rollout
        buf, last_val = collect_rollout(trainer, cfg.steps_per_epoch)
        data = buf.compute_advantages(last_val, cfg.gamma, cfg.lam)

        # PPO update
        stats = ppo_update(trainer, data, updates=cfg.update_epochs)
        total_steps = epoch * cfg.steps_per_epoch

        # periodic evaluation
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            eval_metrics = evaluate_policy(trainer, episodes=cfg.eval_episodes)
            reward = eval_metrics["mean_reward"]
            best_metric = max(best_metric, reward)
            log_metrics(ctx, {**stats, **eval_metrics, "epoch": epoch}, step=epoch)
            LOG.info("[EPOCH %d] reward=%.3f latency_p95=%.3f pi_loss=%.4f v_loss=%.4f",
                     epoch, reward, eval_metrics["p95_latency"], stats["pi_loss"], stats["v_loss"])

            # checkpoint if better
            if epoch % cfg.checkpoint_interval == 0 or reward >= best_metric:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"rl_ckpt_{epoch:04d}.pt")
                save_checkpoint({"model": trainer.model, "optimizer": trainer.optimizer, "epoch": epoch}, ckpt_path)
                manage_checkpoint_retention(trainer, ckpt_path, reward)

        else:
            log_metrics(ctx, {**stats, "epoch": epoch}, step=epoch)

        if cfg.verbose and epoch % 10 == 0:
            elapsed = time.time() - t0
            LOG.info("Epoch %d/%d done, elapsed=%.1fs", epoch, cfg.epochs, elapsed)

    # final checkpoint
    final_ckpt = os.path.join(cfg.checkpoint_dir, f"rl_final_{current_time_str()}.pt")
    save_checkpoint({"model": trainer.model, "optimizer": trainer.optimizer, "epoch": cfg.epochs}, final_ckpt)
    manage_checkpoint_retention(trainer, final_ckpt, best_metric)

    # finalize loggers
    if ctx.get("mlflow") and _HAS_MLFLOW:
        try:
            mlflow.log_metric("best_reward", best_metric)
            mlflow.end_run()
        except Exception:
            LOG.exception("Failed to close MLflow run")

    if ctx.get("wandb") and _HAS_WANDB:
        try:
            wandb.summary["best_reward"] = best_metric
            wandb.finish()
        except Exception:
            LOG.exception("Failed to close W&B run")

    LOG.info("Training complete. Best reward=%.3f checkpoint=%s", best_metric, final_ckpt)
    return {"best_reward": best_metric, "checkpoint": final_ckpt}

# ---------------------------
# CLI interface
# ---------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(description="PriorityMax Heavy PPO Trainer")
    
    # Training schedule (support both epoch-based and timestep-based)
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=8192, help="Steps per epoch")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total timesteps (alternative to epochs)")
    
    # Algorithm (for compatibility, only PPO supported)
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo"], help="RL algorithm (only PPO supported)")
    
    # Model and optimization
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--use-ddp", action="store_true", help="Enable DistributedDataParallel")
    parser.add_argument("--use-ray", action="store_true", help="Enable Ray distributed training")
    parser.add_argument("--ray-address", default=None, help="Ray cluster address")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Experiment tracking
    parser.add_argument("--exp-name", default="prioritymax_rl_heavy", help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    
    # Checkpointing
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (print config only)")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_MODELS_DIR), help="Checkpoint directory")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval (epochs)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint save interval")
    
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Auto-convert total_timesteps to epochs if provided
    if args.total_timesteps is not None and args.epochs is None:
        args.epochs = max(1, args.total_timesteps // args.steps_per_epoch)
        LOG.info("Auto-converted --total-timesteps=%d to --epochs=%d (steps-per-epoch=%d)", 
                 args.total_timesteps, args.epochs, args.steps_per_epoch)
    elif args.epochs is None:
        args.epochs = 500  # default
    
    # Validate algo (only PPO supported, but accept the arg for compatibility)
    if args.algo != "ppo":
        LOG.warning("Only PPO is supported. Ignoring --algo=%s", args.algo)
    
    cfg = HeavyRLConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        gpus=args.gpus,
        use_ddp=args.use_ddp,
        use_ray=args.use_ray,
        ray_address=args.ray_address,
        hidden_dim=args.hidden_dim,
        num_envs=args.envs,
        lr=args.lr,
        gamma=args.gamma,
        experiment_name=args.exp_name,
        log_wandb=args.wandb,
        log_mlflow=args.mlflow,
        resume_from=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        dry_run=args.dry_run,
    )
    
    if cfg.dry_run:
        LOG.warning("Running in DRY-RUN mode (no training, test configs only).")
        print(json.dumps(asdict(cfg), indent=2))
        return
    
    train_heavy(cfg)

if __name__ == "__main__":
    main()
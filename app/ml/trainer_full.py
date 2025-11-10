# backend/app/ml/trainer_full.py
"""
PriorityMax Unified Trainer
===========================

This module merges single-machine, distributed, and tuning trainers into one
production-grade training controller for PriorityMax.

Includes:
 - PPO (single-machine RL)
 - Predictor (LightGBM / sklearn)
 - Evaluation, logging, checkpointing
 - RLlib distributed PPO
 - Ray Tune & Optuna hyperparameter search
 - Torch.distributed launcher
 - Canary gating, MLflow, W&B integration
 - Unified CLI interface

Usage Examples:
---------------
  # Train PPO locally
  python3 backend/app/ml/trainer_full.py train_rl --epochs 300 --steps 2048

  # Train Predictor
  python3 backend/app/ml/trainer_full.py train_predictor --data datasets/queue_metrics.csv

  # Run distributed RLlib training
  python3 backend/app/ml/trainer_full.py rllib_train --epochs 100 --cpus 8

  # Run hyperparameter tuning (Ray Tune)
  python3 backend/app/ml/trainer_full.py tune --samples 10 --scheduler asha

  # Tune predictor with Optuna
  python3 backend/app/ml/trainer_full.py optuna_predictor --data datasets/queue_metrics.csv
"""

from __future__ import annotations
import os
import sys
import math
import time
import json
import uuid
import random
import logging
import pathlib
import shutil
import tempfile
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple, Callable

import numpy as np

# ==============================
# Optional dependencies (defensive imports)
# ==============================

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.optim import Adam
    from torch.nn.parallel import DistributedDataParallel as DDP
    _HAS_TORCH = True
except Exception:
    torch = nn = dist = mp = Adam = DDP = None
    _HAS_TORCH = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    lgb = None
    _HAS_LGB = False

try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

try:
    import ray
    from ray import tune
    from ray import air
    from ray.rllib.agents import ppo as rllib_ppo
    from ray.tune.registry import register_env
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    _HAS_RAY = True
except Exception:
    ray = tune = air = rllib_ppo = register_env = OptunaSearch = ASHAScheduler = PopulationBasedTraining = None
    _HAS_RAY = False

try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    optuna = None
    _HAS_OPTUNA = False

# ==============================
# Local Project Imports (best-effort)
# ==============================
ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ml.real_env import SimulatedRealEnv, EnvConfig, make_vec_env
    from ml.rl_agent_sandbox import PPOActorCritic
    from ml.model_registry import ModelRegistry
    from ml.predictor import PredictorManager, LightGBMPredictor
except Exception:
    SimulatedRealEnv = EnvConfig = make_vec_env = PPOActorCritic = None
    ModelRegistry = PredictorManager = LightGBMPredictor = None

# ==============================
# Logging Configuration
# ==============================
LOG = logging.getLogger("prioritymax.ml.trainer_full")
LOG.setLevel(os.getenv("PRIORITYMAX_TRAINER_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# ==============================
# Global Paths and Constants
# ==============================
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RL_CKPT = MODELS_DIR / "rl_agent.pt"
DEFAULT_PRED_CKPT = MODELS_DIR / "predictor_lgbm.pkl"
TUNE_DIR = MODELS_DIR / "tune"
TUNE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Dataclasses
# ==============================

@dataclass
class RLTrainConfig:
    epochs: int = 300
    steps_per_epoch: int = 2048
    hidden_dim: int = 128
    mini_batch_size: int = 256
    update_epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    seed: int = 42
    device: str = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    save_interval: int = 25
    eval_interval: int = 10
    checkpoint_path: str = str(DEFAULT_RL_CKPT)
    log_wandb: bool = False
    log_mlflow: bool = False
    wandb_project: str = "PriorityMax-RL"
    mlflow_experiment: str = "PriorityMax-RL"

@dataclass
class PredictorTrainConfig:
    data_path: Optional[str] = None
    target_col: str = "queue_next_sec"
    test_size: float = 0.2
    random_state: int = 42
    lgb_params: Optional[dict] = None
    n_estimators: int = 100
    early_stopping_rounds: int = 50
    checkpoint_path: str = str(DEFAULT_PRED_CKPT)
    log_wandb: bool = False
    log_mlflow: bool = False
    wandb_project: str = "PriorityMax-Predictor"
    mlflow_experiment: str = "PriorityMax-Predictor"

@dataclass
class RayClusterConfig:
    address: Optional[str] = os.getenv("RAY_ADDRESS", None)
    local_mode: bool = os.getenv("RAY_LOCAL_MODE", "false").lower() == "true"
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    include_dashboard: bool = False
    ignore_reinit_error: bool = True

@dataclass
class TuneJobConfig:
    name: str = "prioritymax_tune"
    num_samples: int = 10
    max_concurrent: int = 2
    stop: Dict[str, Any] = None
    resources_per_trial: Dict[str, Any] = None
    scheduler: Optional[str] = "asha"
    metric: str = "episode_reward_mean"
    mode: str = "max"
    local_dir: Optional[str] = str(TUNE_DIR)

# End of Chunk 1
# -------------------------------
# Next: Chunk 2 -> Core Utilities (save/load, seeding, logger setup, etc.)
# ==============================
# Core Utilities
# ==============================

def set_seed(seed: int):
    """Reproducibly set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: pathlib.Path):
    """Persist model/training state to disk, supporting both torch and generic pickled formats."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_TORCH and isinstance(state.get("model"), torch.nn.Module):
        # save only state_dict for PyTorch models
        state_to_save = state.copy()
        model = state_to_save.pop("model")
        state_to_save["model"] = model.state_dict()
        torch.save(state_to_save, str(path))
    else:
        import joblib
        joblib.dump(state, str(path))
    LOG.info("Checkpoint saved: %s", path)


def load_checkpoint(path: Union[str, pathlib.Path]) -> dict:
    """Load checkpoint from disk."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} not found")
    if _HAS_TORCH:
        try:
            return torch.load(str(path), map_location="cpu")
        except Exception:
            LOG.warning("Torch load failed; trying joblib")
    import joblib
    return joblib.load(str(path))


# ------------------------------
# MLflow / W&B logger helpers
# ------------------------------
def start_loggers(run_name: str, mlflow_exp: Optional[str] = None, wandb_proj: Optional[str] = None,
                  cfg: Optional[dict] = None) -> Dict[str, bool]:
    """Initialize MLflow and/or W&B sessions."""
    ctx = {"mlflow": False, "wandb": False}

    if mlflow_exp and _HAS_MLFLOW:
        try:
            mlflow.set_experiment(mlflow_exp)
            mlflow.start_run(run_name=run_name)
            if cfg:
                mlflow.log_params(cfg)
            ctx["mlflow"] = True
            LOG.info("MLflow run started: %s", mlflow_exp)
        except Exception:
            LOG.exception("MLflow initialization failed")

    if wandb_proj and _HAS_WANDB:
        try:
            wandb.init(project=wandb_proj, name=run_name, config=cfg or {})
            ctx["wandb"] = True
            LOG.info("Weights & Biases run started: %s", wandb_proj)
        except Exception:
            LOG.exception("W&B initialization failed")

    return ctx


def log_metrics(ctx: Dict[str, bool], metrics: Dict[str, float], step: int):
    """Send metrics to active loggers (MLflow/W&B)."""
    if ctx.get("mlflow") and _HAS_MLFLOW:
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            LOG.debug("MLflow metric logging failed")
    if ctx.get("wandb") and _HAS_WANDB:
        try:
            wandb.log({**metrics, "step": step})
        except Exception:
            LOG.debug("W&B metric logging failed")


def end_loggers(ctx: Dict[str, bool]):
    """Cleanly finalize logging sessions."""
    if ctx.get("mlflow") and _HAS_MLFLOW:
        try:
            mlflow.end_run()
        except Exception:
            LOG.debug("MLflow close failed")
    if ctx.get("wandb") and _HAS_WANDB:
        try:
            wandb.finish()
        except Exception:
            LOG.debug("W&B close failed")


# ==============================
# PPO / RL Utilities
# ==============================
if _HAS_TORCH:

    class ActorCritic(nn.Module):
        """Simple shared-network PPO Actor-Critic used for single-machine RL."""

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

        def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Sample an action and its log-probability from current policy."""
            logits = self.actor(obs)
            mu = torch.tanh(logits)
            std = torch.ones_like(mu) * 0.2
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)
            return action, logp

        def value(self, obs: torch.Tensor) -> torch.Tensor:
            return self.critic(obs).squeeze(-1)


def compute_gae(rewards: np.ndarray, values: np.ndarray, last_value: float,
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE-λ) for a trajectory.
    """
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else last_value) - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv


# End of Chunk 2
# -------------------------------
# Next: Chunk 3 → Single-machine RL training loop (PPO), evaluation and predictor training sections.
# ==============================
# Chunk 3: Single-machine RL trainer, evaluation, predictor training
# ==============================

from typing import Union
import pandas as pd

# ------------------------------
# Single-machine PPO training
# ------------------------------
def train_rl(rl_cfg: RLTrainConfig):
    """
    Train PPO on a single machine using SimulatedRealEnv and a simple ActorCritic model.
    This routine is production-oriented:
    - supports vectorized envs via make_vec_env
    - supports resume from checkpoint
    - logs metrics to MLflow/W&B when configured
    - periodically evaluates the policy
    """
    if SimulatedRealEnv is None or (not _HAS_TORCH and PPOActorCritic is None):
        raise RuntimeError("RL environment or PyTorch model not available. Ensure ml.real_env and torch are installed.")

    set_seed(rl_cfg.seed)
    device = torch.device(rl_cfg.device if _HAS_TORCH else "cpu")
    LOG.info("Starting single-machine RL training on device=%s", device)

    # Create vectorized envs
    n_envs = max(1, min(16, int(os.getenv("PRIORITYMAX_RL_ENVS", "4"))))
    try:
        envs = make_vec_env(EnvConfig(mode="sim", seed=rl_cfg.seed), n=n_envs)
    except Exception:
        # fallback: create list of env instances
        envs = [SimulatedRealEnv(EnvConfig(mode="sim", seed=rl_cfg.seed + i)) for i in range(n_envs)]

    # Determine observation and action dims from env
    sample_obs = envs[0].reset()
    if isinstance(sample_obs, (list, tuple, np.ndarray)):
        obs_dim = int(np.array(sample_obs).ravel().shape[0])
    elif isinstance(sample_obs, dict):
        # flatten dict observations
        obs_dim = int(np.concatenate([np.ravel(v) for v in sample_obs.values()]).shape[0])
    else:
        obs_dim = int(1)

    # simple continuous action space dimension heuristic (can be adapted per env)
    act_dim = int(getattr(envs[0], "action_dim", 3))

    # instantiate model (prefer user-supplied PPOActorCritic if available)
    if PPOActorCritic is not None:
        model = PPOActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=rl_cfg.hidden_dim).to(device)
    elif _HAS_TORCH:
        model = ActorCritic(obs_dim, act_dim, rl_cfg.hidden_dim).to(device)
    else:
        raise RuntimeError("No suitable actor-critic implementation available")

    optimizer = Adam(model.parameters(), lr=rl_cfg.lr) if _HAS_TORCH else None

    # Resume from checkpoint if exists
    ckpt_path = pathlib.Path(rl_cfg.checkpoint_path)
    if ckpt_path.exists():
        try:
            ckpt = load_checkpoint(ckpt_path)
            if _HAS_TORCH and isinstance(ckpt.get("model"), dict):
                model.load_state_dict(ckpt["model"])
            elif _HAS_TORCH and isinstance(ckpt.get("model"), torch.nn.Module):
                model.load_state_dict(ckpt["model"].state_dict())
            if "optimizer" in ckpt and optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            LOG.info("Resumed checkpoint from %s", ckpt_path)
        except Exception:
            LOG.exception("Failed to resume checkpoint; continuing from scratch")

    # Start loggers
    run_name = f"ppo_{int(time.time())}"
    log_ctx = start_loggers(run_name, mlflow_exp=(rl_cfg.mlflow_experiment if rl_cfg.log_mlflow else None),
                            wandb_proj=(rl_cfg.wandb_project if rl_cfg.log_wandb else None),
                            cfg=asdict(rl_cfg))

    total_steps = rl_cfg.steps_per_epoch * rl_cfg.epochs
    global_step = 0
    episode_returns = []
    episode_lengths = []

    # We'll collect data across envs into flattened buffers
    for epoch in range(1, rl_cfg.epochs + 1):
        obs_batch = []
        act_batch = []
        rew_batch = []
        val_batch = []
        logp_batch = []

        # reset all envs
        obs_list = [env.reset() for env in envs]
        done_mask = [False] * len(envs)
        ep_rets = [0.0] * len(envs)
        ep_lens = [0] * len(envs)

        steps_collected = 0
        while steps_collected < rl_cfg.steps_per_epoch:
            # prepare observation tensor
            obs_arr = np.stack([np.asarray(o).ravel() for o in obs_list], axis=0).astype(np.float32)
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)

            with torch.no_grad():
                action_t, logp_t = model.act(obs_t)
                value_t = model.value(obs_t).detach().cpu().numpy()

            actions = action_t.detach().cpu().numpy()
            logps = logp_t.detach().cpu().numpy()

            # step each env
            next_obs_list = []
            rewards = []
            dones = []
            infos = []
            for i, env in enumerate(envs):
                try:
                    nobs, rew, done, info = env.step(actions[i])
                except Exception:
                    # handle envs that expect different shapes or action formats
                    nobs, rew, done, info = env.step(actions[i].tolist() if hasattr(actions[i], "tolist") else float(actions[i]))
                next_obs_list.append(nobs)
                rewards.append(float(rew))
                dones.append(bool(done))
                infos.append(info)
                ep_rets[i] += float(rew)
                ep_lens[i] += 1

            # append to buffers
            for i in range(len(envs)):
                obs_batch.append(np.asarray(obs_list[i]).ravel().astype(np.float32))
                act_batch.append(np.asarray(actions[i]).astype(np.float32))
                rew_batch.append(float(rewards[i]))
                val_batch.append(float(value_t[i]))
                logp_batch.append(float(logps[i]))

            obs_list = next_obs_list
            steps_collected += len(envs)
            global_step += len(envs)

            # collect finished episodes and reset trackers
            for i, d in enumerate(dones):
                if d:
                    episode_returns.append(ep_rets[i])
                    episode_lengths.append(ep_lens[i])
                    ep_rets[i] = 0.0
                    ep_lens[i] = 0

        # compute last value for bootstrap (per-env last observation)
        last_vals = []
        for o in obs_list:
            t = torch.tensor(np.asarray(o).ravel()[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                last_vals.append(float(model.value(t).detach().cpu().item()))

        # compute advantages and returns using GAE per-trajectory flattened (approx)
        values = np.array(val_batch + last_vals, dtype=np.float32)
        rewards = np.array(rew_batch, dtype=np.float32)
        # naive return estimation (flattened)
        returns = []
        G = 0.0
        for r in rewards[::-1]:
            G = r + rl_cfg.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        advantages = returns - np.array(val_batch, dtype=np.float32)
        # normalize advantages
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert buffers to tensors
        obs_tensor = torch.tensor(np.array(obs_batch, dtype=np.float32), device=device)
        act_tensor = torch.tensor(np.array(act_batch, dtype=np.float32), device=device)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        old_logp_tensor = torch.tensor(np.array(logp_batch, dtype=np.float32), device=device)

        # PPO update epochs
        pi_losses = []
        v_losses = []
        clip_fracs = []
        dataset_size = obs_tensor.shape[0]
        for _ in range(rl_cfg.update_epochs):
            # create random minibatches
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, rl_cfg.mini_batch_size):
                mb_idx = idxs[start:start + rl_cfg.mini_batch_size]
                mb_obs = obs_tensor[mb_idx]
                mb_act = act_tensor[mb_idx]
                mb_adv = adv_tensor[mb_idx]
                mb_ret = ret_tensor[mb_idx]
                mb_old_logp = old_logp_tensor[mb_idx]

                # forward
                logits = model.actor(mb_obs)
                mu = torch.tanh(logits)
                std = torch.ones_like(mu) * 0.2
                dist = torch.distributions.Normal(mu, std)
                new_logp = dist.log_prob(mb_act).sum(axis=-1)
                ratio = torch.exp(new_logp - mb_old_logp)
                # clipped surrogate objective
                clip_adv = torch.clamp(ratio, 1.0 - rl_cfg.clip_range, 1.0 + rl_cfg.clip_range) * mb_adv
                pi_loss = -(torch.min(ratio * mb_adv, clip_adv)).mean()
                # value loss
                value_pred = model.value(mb_obs)
                v_loss = ((mb_ret - value_pred) ** 2).mean()
                entropy = dist.entropy().sum(axis=-1).mean()
                loss = pi_loss + rl_cfg.value_coef * v_loss - rl_cfg.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                clip_fracs.append((torch.abs(ratio - 1.0) > rl_cfg.clip_range).float().mean().item())

        # aggregate metrics
        metrics = {
            "epoch": epoch,
            "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "global_step": global_step
        }
        LOG.info("Epoch %d/%d metrics: %s", epoch, rl_cfg.epochs, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

        # log to MLflow/W&B
        log_metrics(log_ctx, metrics, step=epoch)

        # save checkpoint periodically
        if epoch % rl_cfg.save_interval == 0 or epoch == rl_cfg.epochs:
            ckpt_state = {"model": model, "optimizer": optimizer.state_dict() if optimizer is not None else None, "epoch": epoch, "meta": metrics}
            save_checkpoint(ckpt_state, pathlib.Path(rl_cfg.checkpoint_path))

        # run evaluation periodically
        if epoch % rl_cfg.eval_interval == 0 or epoch == rl_cfg.epochs:
            try:
                eval_res = evaluate_rl(model, device=str(device), episodes=5)
                LOG.info("Eval results: %s", eval_res)
                log_metrics(log_ctx, {"eval_avg_reward": eval_res.get("avg_reward", 0.0)}, step=epoch)
            except Exception:
                LOG.exception("Evaluation failed")

    # finalize loggers
    end_loggers(log_ctx)
    LOG.info("Finished training. Last checkpoint at %s", rl_cfg.checkpoint_path)
    return True


# ------------------------------
# RL Evaluation
# ------------------------------
def evaluate_rl(model_or_path: Union[str, torch.nn.Module], device: str = "cpu", episodes: int = 10) -> Dict[str, Any]:
    """
    Evaluate a saved model (path) or an in-memory model instance.
    Returns average reward and std.
    """
    if SimulatedRealEnv is None:
        raise RuntimeError("SimulatedRealEnv required for evaluation")

    device = torch.device(device if _HAS_TORCH else "cpu")

    # load model if path provided
    if isinstance(model_or_path, (str, pathlib.Path)):
        if not _HAS_TORCH:
            raise RuntimeError("Torch required to load model from disk")
        ckpt = load_checkpoint(model_or_path)
        # reconstruct model shape heuristically; user should provide shape for robust eval
        # We'll attempt to infer dims from env
        env = SimulatedRealEnv(EnvConfig(mode="sim", seed=0))
        obs0 = env.reset()
        obs_dim = int(np.array(obs0).ravel().shape[0])
        act_dim = int(getattr(env, "action_dim", 3))
        model = ActorCritic(obs_dim, act_dim, 128)
        state = ckpt.get("model", None)
        if state is None:
            raise RuntimeError("Checkpoint does not contain model state")
        model.load_state_dict(state if isinstance(state, dict) else state.get("model", {}))
    else:
        model = model_or_path

    model.to(device)
    model.eval()
    env = SimulatedRealEnv(EnvConfig(mode="sim", seed=42))
    rewards = []
    for ep in range(episodes):
        obs = env.reset(seed=42 + ep) if hasattr(env, "reset") else env.reset()
        total = 0.0
        for t in range(1000):
            obs_arr = torch.tensor(np.asarray(obs).ravel()[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, _ = model.act(obs_arr)
            action = action_t.detach().cpu().numpy()[0]
            obs, rew, done, info = env.step(action)
            total += float(rew)
            if done:
                break
        rewards.append(total)
    return {"avg_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)), "episodes": episodes}


# ------------------------------
# Predictor training & evaluation (LightGBM / sklearn fallback)
# ------------------------------
def train_predictor(cfg: PredictorTrainConfig) -> Dict[str, Any]:
    """
    Train a LightGBM or sklearn-based predictor on a CSV dataset.
    Returns metrics and saved checkpoint path.
    """
    if not cfg.data_path:
        raise ValueError("data_path is required")
    set_seed(cfg.random_state)

    data_path = pathlib.Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(cfg.data_path)
    df = pd.read_csv(str(data_path))
    if cfg.target_col not in df.columns:
        raise ValueError(f"target_col {cfg.target_col} not in dataset")

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    run_name = f"predictor_{int(time.time())}"
    log_ctx = start_loggers(run_name, mlflow_exp=(cfg.mlflow_experiment if cfg.log_mlflow else None),
                            wandb_proj=(cfg.wandb_project if cfg.log_wandb else None),
                            cfg=asdict(cfg))

    model = None
    ckpt_path = pathlib.Path(cfg.checkpoint_path)
    if _HAS_LGB:
        params = cfg.lgb_params or {"objective": "regression", "metric": "rmse", "verbosity": -1}
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        LOG.info("Training LightGBM with params: %s", params)
        model = lgb.train(params, dtrain, num_boost_round=cfg.n_estimators, valid_sets=[dval], early_stopping_rounds=cfg.early_stopping_rounds, verbose_eval=False)
        model.save_model(str(ckpt_path))
    elif _HAS_SKLEARN:
        LOG.info("Training RandomForest fallback")
        model = RandomForestRegressor(n_estimators=cfg.n_estimators, random_state=cfg.random_state)
        model.fit(X_train, y_train)
        import joblib
        joblib.dump(model, str(ckpt_path))
    else:
        raise RuntimeError("No supported ML library available for predictor training")

    # evaluation
    if _HAS_LGB:
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds)) if _HAS_SKLEARN else float("nan")
    metrics = {"rmse": rmse, "r2": r2}
    LOG.info("Predictor trained. Metrics: %s", metrics)

    log_metrics(log_ctx, metrics, step=0)
    end_loggers(log_ctx)

    return {"ckpt": str(ckpt_path), "metrics": metrics}


def evaluate_predictor(model_path: str, data_path: str, target_col: str = "queue_next_sec") -> Dict[str, Any]:
    """
    Load model from path and evaluate on CSV data_path.
    """
    data_path = pathlib.Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    df = pd.read_csv(str(data_path))
    if target_col not in df.columns:
        raise ValueError("target_col not found in dataset")
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # load model
    if pathlib.Path(model_path).suffix in [".txt", ".model"] and _HAS_LGB:
        model = lgb.Booster(model_file=str(model_path))
        preds = model.predict(X)
    else:
        import joblib
        model = joblib.load(str(model_path))
        preds = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds)) if _HAS_SKLEARN else float("nan")
    LOG.info("Predictor evaluation: RMSE=%.4f R2=%.4f", rmse, r2)
    return {"rmse": rmse, "r2": r2}

# End of Chunk 3
# -------------------------------
# Next: Chunk 4 → Canary gating, distributed RLlib launcher, and checkpoint conversion helpers.
# ==============================
# Chunk 4: Canary gating, RLlib distributed launcher, checkpoint conversion
# ==============================

import shutil
import tarfile

# ------------------------------
# Canary gating helpers
# ------------------------------
def canary_gate_predictor(candidate_ckpt: str, holdout_csv: str, rmse_threshold: float = 5.0) -> bool:
    """
    Evaluate candidate predictor on holdout CSV and decide whether it passes the canary gate.
    Returns True if RMSE <= rmse_threshold.
    """
    try:
        res = evaluate_predictor(candidate_ckpt, holdout_csv)
        rmse = float(res.get("rmse", float("inf")))
        passed = rmse <= float(rmse_threshold)
        LOG.info("Canary gate predictor: ckpt=%s rmse=%.4f threshold=%.4f passed=%s", candidate_ckpt, rmse, rmse_threshold, passed)
        return passed
    except Exception:
        LOG.exception("Canary gating evaluation failed")
        return False


def canary_gate_rl(candidate_checkpoint: str, episodes: int = 10, reward_threshold: float = 100.0) -> bool:
    """
    Run a short evaluation of an RL policy checkpoint and decide pass/fail based on avg reward.
    """
    try:
        res = evaluate_rl(candidate_checkpoint, device="cpu", episodes=episodes)
        avg = float(res.get("avg_reward", 0.0))
        passed = avg >= float(reward_threshold)
        LOG.info("Canary gate RL: ckpt=%s avg_reward=%.3f threshold=%.3f passed=%s", candidate_checkpoint, avg, reward_threshold, passed)
        return passed
    except Exception:
        LOG.exception("RL canary evaluation failed")
        return False

# ------------------------------
# RLlib helpers and launcher
# ------------------------------
def _ensure_ray_initialized(ray_cfg: Optional[RayClusterConfig] = None):
    if not _HAS_RAY:
        raise RuntimeError("Ray is not available. Install 'ray[rllib]' to use RLlib features.")
    cfg = ray_cfg or RayClusterConfig()
    if ray.is_initialized():
        LOG.debug("Ray already initialized")
        return
    init_kwargs = {}
    if cfg.address:
        init_kwargs["address"] = cfg.address
    if cfg.local_mode:
        init_kwargs["local_mode"] = True
    if cfg.num_cpus is not None:
        init_kwargs["num_cpus"] = int(cfg.num_cpus)
    if cfg.num_gpus is not None:
        init_kwargs["num_gpus"] = int(cfg.num_gpus)
    if cfg.include_dashboard:
        init_kwargs["include_dashboard"] = True
    LOG.info("Initializing Ray with %s", init_kwargs)
    ray.init(**init_kwargs, _ignore_reinit_error=cfg.ignore_reinit_error)


def _register_env_for_rllib():
    if not _HAS_RAY:
        LOG.debug("Ray not present; skipping env registration")
        return

    def creator(env_config):
        if SimulatedRealEnv is None or EnvConfig is None:
            raise RuntimeError("SimulatedRealEnv or EnvConfig not available for RLlib registration")
        cfg = EnvConfig(mode=env_config.get("mode", "sim"), seed=env_config.get("seed", 0))
        return SimulatedRealEnv(cfg)

    try:
        register_env("PriorityMaxSimEnv", lambda cfg: creator(cfg))
        LOG.info("Registered RLlib environment 'PriorityMaxSimEnv'")
    except Exception:
        LOG.exception("Failed to register RLlib env (it may already be registered)")

def run_rllib_ppo(rl_cfg: RLTrainConfig,
                  ray_cfg: Optional[RayClusterConfig] = None,
                  rllib_overrides: Optional[Dict[str, Any]] = None,
                  checkpoint_dir: Optional[str] = None,
                  restore_from: Optional[str] = None) -> bool:
    """
    Launch RLlib PPO training job.
    - Initializes Ray
    - Registers the env
    - Creates PPOTrainer with a sensible config derived from rl_cfg
    - Runs for rl_cfg.epochs iterations, saves checkpoints, and optionally converts them
    """
    if not _HAS_RAY:
        raise RuntimeError("Ray is required for RLlib training. Install ray[rllib].")

    ray_cfg = ray_cfg or RayClusterConfig()
    _ensure_ray_initialized(ray_cfg)
    _register_env_for_rllib()

    # default RLlib configuration mapped from RLTrainConfig
    rllib_conf = {
        "env": "PriorityMaxSimEnv",
        "framework": "torch",
        "num_workers": max(0, (ray_cfg.num_cpus or 1) - 1),
        "num_gpus": ray_cfg.num_gpus or 0,
        "train_batch_size": int(getattr(rl_cfg, "steps_per_epoch", 2048)),
        "sgd_minibatch_size": int(getattr(rl_cfg, "mini_batch_size", 256)),
        "model": {"fcnet_hiddens": [int(rl_cfg.hidden_dim), int(rl_cfg.hidden_dim)]},
        "lr": float(rl_cfg.lr),
        "gamma": float(rl_cfg.gamma),
        "lambda": float(rl_cfg.lam),
        "clip_param": float(rl_cfg.clip_range),
        "entropy_coeff": float(rl_cfg.entropy_coef),
        "vf_loss_coeff": float(rl_cfg.value_coef),
        "num_sgd_iter": int(rl_cfg.update_epochs),
        "log_level": "INFO",
    }
    if rllib_overrides:
        rllib_conf.update(rllib_overrides)

    # instantiate trainer
    try:
        trainer = rllib_ppo.PPOTrainer(config=rllib_conf, env="PriorityMaxSimEnv")
    except Exception as e:
        LOG.exception("Failed to create RLlib PPOTrainer: %s", e)
        raise

    checkpoint_dir = checkpoint_dir or str(MODELS_DIR / "rllib_checkpoints")
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # restore if requested
    if restore_from:
        try:
            trainer.restore(restore_from)
            LOG.info("Restored RLlib trainer from %s", restore_from)
        except Exception:
            LOG.exception("Failed to restore RLlib trainer from %s", restore_from)

    epochs = int(getattr(rl_cfg, "epochs", 100))
    save_interval = int(max(1, getattr(rl_cfg, "save_interval", max(1, epochs // 10))))
    try:
        for it in range(1, epochs + 1):
            result = trainer.train()
            reward_mean = result.get("episode_reward_mean", None)
            LOG.info("[RLlib] Iter %d/%d reward_mean=%s", it, epochs, reward_mean)
            if it % save_interval == 0 or it == epochs:
                ck = trainer.save(checkpoint_dir)
                LOG.info("[RLlib] Saved checkpoint: %s", ck)
                # best-effort convert
                try:
                    conv = convert_rllib_checkpoint_to_torch(trainer, ck, ModelsDir=MODELS_DIR, iter_id=it)
                    LOG.info("[RLlib] Converted checkpoint to torch: %s", conv)
                except Exception:
                    LOG.exception("[RLlib] Checkpoint conversion failed (continuing)")
    finally:
        try:
            trainer.stop()
        except Exception:
            pass
    return True


# ------------------------------
# Checkpoint conversion & archival
# ------------------------------
def convert_rllib_checkpoint_to_torch(trainer, rllib_checkpoint_path: str, ModelsDir: pathlib.Path = MODELS_DIR, iter_id: Optional[int] = None) -> Optional[str]:
    """
    Best-effort attempt to extract policy weights/state from RLlib trainer and save a PyTorch-style checkpoint.
    RLlib internals change across versions; this function makes multiple attempts and falls back to metadata.
    Returns path to converted file or None.
    """
    ModelsDir.mkdir(parents=True, exist_ok=True)
    out_path = ModelsDir / f"rl_agent_rllib_iter{iter_id or int(time.time())}.pt"
    try:
        policy = trainer.get_policy()
        # Try common access patterns
        state = None
        if hasattr(policy, "get_state"):
            try:
                state = policy.get_state()
            except Exception:
                state = None
        if state is None:
            # Try to extract PyTorch model inside policy
            try:
                # policy.model exists in many RLlib versions
                model_obj = getattr(policy, "model", None)
                if model_obj is not None and hasattr(model_obj, "state_dict"):
                    sd = model_obj.state_dict()
                    import torch
                    torch.save({"model": sd, "meta": {"rllib_checkpoint": rllib_checkpoint_path}}, str(out_path))
                    return str(out_path)
            except Exception:
                LOG.exception("Failed to extract policy.model.state_dict")
        # fallback: save the raw state as torch file
        try:
            import torch
            torch.save({"rllib_state": state, "meta": {"rllib_checkpoint": rllib_checkpoint_path}}, str(out_path))
            return str(out_path)
        except Exception:
            # as last resort write JSON metadata
            meta_path = ModelsDir / f"rl_agent_rllib_meta_{int(time.time())}.json"
            meta = {"rllib_checkpoint": rllib_checkpoint_path, "note": "could not convert state to torch"}
            meta_path.write_text(json.dumps(meta))
            return str(meta_path)
    except Exception:
        LOG.exception("RLlib -> torch conversion failed")
        return None


# ------------------------------
# Model promotion & archival helpers
# ------------------------------
def archive_checkpoint(src_path: str, archive_dir: Optional[str] = None) -> Optional[str]:
    """
    Archive a checkpoint into a tar.gz in the models archive directory.
    Returns archive path or None.
    """
    try:
        src = pathlib.Path(src_path)
        if not src.exists():
            raise FileNotFoundError(src_path)
        arch_dir = pathlib.Path(archive_dir or (MODELS_DIR / "archive"))
        arch_dir.mkdir(parents=True, exist_ok=True)
        dst = arch_dir / f"{src.stem}_{int(time.time())}.tar.gz"
        with tarfile.open(str(dst), "w:gz") as tf:
            if src.is_dir():
                for f in src.rglob("*"):
                    tf.add(str(f), arcname=str(f.relative_to(src)))
            else:
                tf.add(str(src), arcname=src.name)
        LOG.info("Archived %s -> %s", src, dst)
        return str(dst)
    except Exception:
        LOG.exception("Archive failed for %s", src_path)
        return None


def promote_model_to_registry(model_path: str, model_type: str = "predictor", metadata: Optional[dict] = None) -> bool:
    """
    Best-effort model registration: if a ModelRegistry is available, use it.
    Otherwise, move the model into models/registered/ with metadata JSON.
    """
    try:
        if ModelRegistry is not None:
            reg = ModelRegistry()
            # The registry implementation is project-specific; we attempt to call a standard method
            if hasattr(reg, "register"):
                reg.register(model_path, model_type=model_type, metadata=metadata or {})
                LOG.info("Registered model %s via ModelRegistry", model_path)
                return True
        # fallback local promotion
        reg_dir = MODELS_DIR / "registered" / model_type
        reg_dir.mkdir(parents=True, exist_ok=True)
        dest = reg_dir / pathlib.Path(model_path).name
        shutil.copy2(model_path, dest)
        meta_path = dest.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata or {}, default=str))
        LOG.info("Promoted model %s -> %s", model_path, dest)
        return True
    except Exception:
        LOG.exception("Model promotion failed for %s", model_path)
        return False

# End of Chunk 4
# -------------------------------
# Next: Chunk 5 → Ray Tune + Optuna tuning, torch.distributed launcher, and helper utilities for large-scale experiments.
# ==============================
# Chunk 5: Ray Tune, Optuna, torch.distributed launcher, tuning helpers
# ==============================

# ------------------------------
# Ray Tune scheduler factory
# ------------------------------
def build_tune_scheduler(name: Optional[str], metric: str = "episode_reward_mean", mode: str = "max"):
    if not _HAS_RAY:
        raise RuntimeError("Ray/Tune is required for Tune schedulers.")
    if not name:
        return None
    name = name.lower()
    if name == "asha":
        try:
            return ASHAScheduler(metric=metric, mode=mode)
        except Exception:
            LOG.exception("Failed to create ASHA scheduler")
            return None
    if name == "pbt":
        try:
            return PopulationBasedTraining(time_attr="training_iteration", perturbation_interval=5)
        except Exception:
            LOG.exception("Failed to create PBT scheduler")
            return None
    LOG.warning("Unsupported scheduler '%s' - using no scheduler", name)
    return None


# ------------------------------
# Make a Tune-compatible RLlib trainable
# ------------------------------
def make_rllib_trainable_for_tune(base_rl_cfg: RLTrainConfig) -> Callable:
    """
    Create a Ray Tune 'trainable' that runs a short RLlib training job for the trial.
    tune will call this with per-trial 'config' overrides.
    """
    if not _HAS_RAY:
        raise RuntimeError("Ray is required to use Tune wrappers")

    def trainable(config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        # Merge base config with trial overrides
        merged = asdict(base_rl_cfg)
        merged.update(config or {})
        # Minimal RLlib config for quick trials
        rllib_conf = {
            "env": "PriorityMaxSimEnv",
            "framework": "torch",
            "num_workers": int(merged.get("num_workers", 0)),
            "num_gpus": float(merged.get("num_gpus", 0)),
            "train_batch_size": int(merged.get("steps_per_epoch", 1024)),
            "sgd_minibatch_size": int(merged.get("mini_batch_size", 256)),
            "model": {"fcnet_hiddens": [int(merged.get("hidden_dim", 128)), int(merged.get("hidden_dim", 128))]},
            "lr": float(merged.get("lr", 3e-4)),
            "num_sgd_iter": int(merged.get("update_epochs", 3)),
            "log_level": "WARN",
        }
        _ensure_ray_initialized(RayClusterConfig(local_mode=True))
        _register_env_for_rllib()

        try:
            trainer = rllib_ppo.PPOTrainer(config=rllib_conf, env="PriorityMaxSimEnv")
        except Exception:
            LOG.exception("Failed to instantiate RLlib trainer in tune trainable")
            raise

        # quick training iterations (trial-level)
        n_iter = int(config.get("train_iters", 3))
        for it in range(1, n_iter + 1):
            res = trainer.train()
            # report primary metric back to Tune
            tune.report(episode_reward_mean=res.get("episode_reward_mean", float("nan")))
        # save checkpoint for the trial
        ck = trainer.save()
        tune.report(trial_checkpoint=ck)
        trainer.stop()

    return trainable


# ------------------------------
# Run a Ray Tune experiment
# ------------------------------
def run_tune_experiment(base_rl_cfg: RLTrainConfig,
                        search_space: Dict[str, Any],
                        tune_cfg: TuneJobConfig,
                        ray_cfg: Optional[RayClusterConfig] = None,
                        use_optuna: bool = False) -> Any:
    """
    Launch a Tune experiment using the provided base RL config and search space.
    - search_space should contain Tune search objects (tune.choice, tune.loguniform, etc.)
    """
    if not _HAS_RAY:
        raise RuntimeError("Ray/Tune is required to run experiments")

    ray_cfg = ray_cfg or RayClusterConfig()
    _ensure_ray_initialized(ray_cfg)
    _register_env_for_rllib()

    trainable = make_rllib_trainable_for_tune(base_rl_cfg)
    scheduler = build_tune_scheduler(tune_cfg.scheduler, metric=tune_cfg.metric, mode=tune_cfg.mode)

    search_alg = None
    if use_optuna:
        if not _HAS_OPTUNA:
            raise RuntimeError("Optuna not installed; cannot use OptunaSearch.")
        try:
            search_alg = OptunaSearch(metric=tune_cfg.metric, mode=tune_cfg.mode)
        except Exception:
            LOG.exception("Failed to initialize OptunaSearch; proceeding without it")
            search_alg = None

    resources = tune_cfg.resources_per_trial or {"cpu": 1, "gpu": 0}
    stop = tune_cfg.stop or {"training_iteration": 10}

    analysis = tune.run(
        trainable,
        name=tune_cfg.name,
        config=search_space,
        num_samples=tune_cfg.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial=resources,
        stop=stop,
        local_dir=tune_cfg.local_dir or str(TUNE_DIR),
        reuse_actors=False,
        max_concurrent_trials=tune_cfg.max_concurrent,
        fail_fast=False,
    )
    LOG.info("Tune finished. Best config: %s", analysis.get_best_config(metric=tune_cfg.metric, mode=tune_cfg.mode))
    return analysis


# ------------------------------
# Optuna objective for LightGBM predictor
# ------------------------------
def optuna_lightgbm_objective(data_path: str, target_col: str, trial: "optuna.trial.Trial") -> float:
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna is not installed")
    if not _HAS_LGB:
        raise RuntimeError("LightGBM required for this objective")

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col} not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    gbm = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval], early_stopping_rounds=20, verbose_eval=False)
    preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    return rmse


def run_optuna_predictor_study(data_path: str, target_col: str = "queue_next_sec", n_trials: int = 50, storage: Optional[str] = None, study_name: Optional[str] = None):
    """
    Run an Optuna study to tune LightGBM predictor hyperparameters.
    """
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna not installed")
    study_name = study_name or f"prioritymax_predictor_{int(time.time())}"
    storage = storage  # e.g., "sqlite:///optuna.db"
    study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name, load_if_exists=True)
    study.optimize(lambda tr: optuna_lightgbm_objective(data_path, target_col, tr), n_trials=n_trials)
    LOG.info("Optuna study complete. Best value: %s", study.best_value)
    return study


# ------------------------------
# Torch.distributed launcher helper
# ------------------------------
def launch_torch_distributed(world_size: int, run_fn: Callable[[int, int], None], dist_backend: str = "nccl", dist_url: Optional[str] = None):
    """
    Launch local torch.distributed processes using torch.multiprocessing.spawn.
    run_fn(local_rank, world_size) will be executed inside each spawned process.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for torch.distributed support")

    if dist_url is None:
        # choose random available port
        port = 29500 + (os.getpid() % 1000)
        dist_url = f"tcp://127.0.0.1:{port}"

    LOG.info("Launching torch.distributed: world_size=%d url=%s backend=%s", world_size, dist_url, dist_backend)

    def _entry(local_rank: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(int(dist_url.split(":")[-1]))
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(local_rank)
        try:
            dist.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=local_rank)
            run_fn(local_rank, world_size)
        finally:
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    try:
        mp.spawn(_entry, nprocs=world_size)
    except Exception:
        LOG.exception("Failed to spawn distributed processes")
        raise

# End of Chunk 5
# -------------------------------
# Next: Chunk 6 → Unified CLI that wires all commands & main() entrypoint.
# ==============================
# Chunk 6: Unified CLI & main entrypoint
# ==============================

import argparse

def _build_cli():
    p = argparse.ArgumentParser(prog="prioritymax-trainer-full", description="PriorityMax unified trainer CLI")
    sub = p.add_subparsers(dest="cmd")

    # --- Single-machine RL ---
    tr = sub.add_parser("train_rl", help="Train single-machine PPO RL agent")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--steps", type=int, default=2048)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--device", type=str, default="cpu")
    tr.add_argument("--checkpoint", type=str, default=str(DEFAULT_RL_CKPT))
    tr.add_argument("--wandb", action="store_true")
    tr.add_argument("--mlflow", action="store_true")

    ev = sub.add_parser("eval_rl", help="Evaluate PPO checkpoint")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--episodes", type=int, default=10)

    # --- Predictor ---
    tp = sub.add_parser("train_predictor", help="Train LightGBM/RandomForest predictor")
    tp.add_argument("--data", required=True)
    tp.add_argument("--target", default="queue_next_sec")
    tp.add_argument("--ckpt", default=str(DEFAULT_PRED_CKPT))
    tp.add_argument("--wandb", action="store_true")
    tp.add_argument("--mlflow", action="store_true")

    ep = sub.add_parser("evaluate_predictor", help="Evaluate predictor checkpoint")
    ep.add_argument("--ckpt", required=True)
    ep.add_argument("--data", required=True)
    ep.add_argument("--target", default="queue_next_sec")

    cg = sub.add_parser("canary_predictor", help="Run canary gate for predictor")
    cg.add_argument("--ckpt", required=True)
    cg.add_argument("--holdout", required=True)
    cg.add_argument("--threshold", type=float, default=5.0)

    # --- Distributed RLlib ---
    dr = sub.add_parser("rllib_train", help="Run distributed PPO training via RLlib")
    dr.add_argument("--epochs", type=int, default=100)
    dr.add_argument("--cpus", type=int, default=4)
    dr.add_argument("--gpus", type=int, default=0)
    dr.add_argument("--save_dir", default=str(MODELS_DIR / "rllib_checkpoints"))
    dr.add_argument("--restore", default=None)

    # --- Tune (Ray Tune hyperparameter search) ---
    tn = sub.add_parser("tune", help="Run Ray Tune experiment for PPO hyperparameters")
    tn.add_argument("--samples", type=int, default=10)
    tn.add_argument("--scheduler", type=str, default="asha")
    tn.add_argument("--local", action="store_true", help="Run Ray in local mode for testing")
    tn.add_argument("--optuna", action="store_true", help="Use Optuna search algorithm")
    tn.add_argument("--metric", default="episode_reward_mean")

    # --- Optuna Predictor Tuning ---
    op = sub.add_parser("optuna_predictor", help="Run Optuna tuning for LightGBM predictor")
    op.add_argument("--data", required=True)
    op.add_argument("--target", default="queue_next_sec")
    op.add_argument("--trials", type=int, default=30)

    # --- Torch Distributed ---
    td = sub.add_parser("torch_distributed", help="Launch torch.distributed multi-process training (demo)")
    td.add_argument("--world_size", type=int, default=2)

    return p


def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    # --- Single-machine RL ---
    if args.cmd == "train_rl":
        cfg = RLTrainConfig(
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            hidden_dim=args.hidden,
            lr=args.lr,
            device=args.device,
            checkpoint_path=args.checkpoint,
            log_wandb=args.wandb,
            log_mlflow=args.mlflow
        )
        train_rl(cfg)

    elif args.cmd == "eval_rl":
        res = evaluate_rl(args.checkpoint, device="cpu", episodes=args.episodes)
        print(json.dumps(res, indent=2))

    # --- Predictor ---
    elif args.cmd == "train_predictor":
        cfg = PredictorTrainConfig(
            data_path=args.data,
            target_col=args.target,
            checkpoint_path=args.ckpt,
            log_wandb=args.wandb,
            log_mlflow=args.mlflow
        )
        result = train_predictor(cfg)
        print(json.dumps(result, indent=2))

    elif args.cmd == "evaluate_predictor":
        res = evaluate_predictor(args.ckpt, args.data, target_col=args.target)
        print(json.dumps(res, indent=2))

    elif args.cmd == "canary_predictor":
        ok = canary_gate_predictor(args.ckpt, args.holdout, rmse_threshold=args.threshold)
        print(json.dumps({"passed": ok}, indent=2))

    # --- RLlib Distributed ---
    elif args.cmd == "rllib_train":
        rl_cfg = RLTrainConfig(epochs=args.epochs)
        ray_cfg = RayClusterConfig(num_cpus=args.cpus, num_gpus=args.gpus)
        run_rllib_ppo(rl_cfg, ray_cfg=ray_cfg, checkpoint_dir=args.save_dir, restore_from=args.restore)

    # --- Ray Tune ---
    elif args.cmd == "tune":
        base_cfg = RLTrainConfig()
        ray_cfg = RayClusterConfig(local_mode=args.local)
        tune_cfg = TuneJobConfig(num_samples=args.samples, scheduler=args.scheduler, metric=args.metric)
        search_space = {
            "lr": tune.loguniform(1e-5, 1e-3),
            "hidden_dim": tune.choice([64, 128, 256]),
            "clip_range": tune.choice([0.1, 0.2, 0.3]),
            "entropy_coef": tune.uniform(0.0, 0.05),
        }
        run_tune_experiment(base_cfg, search_space, tune_cfg, ray_cfg=ray_cfg, use_optuna=args.optuna)

    # --- Optuna Predictor ---
    elif args.cmd == "optuna_predictor":
        run_optuna_predictor_study(args.data, target_col=args.target, n_trials=args.trials)

    # --- Torch Distributed ---
    elif args.cmd == "torch_distributed":
        def worker_fn(local_rank: int, world_size: int):
            LOG.info(f"[RANK {local_rank}] running distributed worker")
            set_seed(42 + local_rank)
            # Example small PPO run or logging
            time.sleep(1)
            LOG.info(f"[RANK {local_rank}] completed")

        launch_torch_distributed(args.world_size, worker_fn)

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        LOG.info("Interrupted by user.")
    except Exception:
        LOG.exception("Fatal error in trainer CLI")

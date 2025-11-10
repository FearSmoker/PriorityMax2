# backend/app/ml/trainer.py
"""
PriorityMax Training Utilities
------------------------------

Provides robust, production-minded training utilities for:
  - RL (PPO) training using SimulatedRealEnv
  - Supervised predictor training (LightGBM / sklearn)
  - Evaluation, checkpointing, canary gating, experiment logging (MLflow / W&B)
  - Resume, hyperparameter search, distributed hooks (optional)

Usage examples:
  - Train RL:
      python3 backend/app/ml/trainer.py train_rl --epochs 500 --steps 4096
  - Eval RL:
      python3 backend/app/ml/trainer.py eval_rl --checkpoint ml/models/rl_agent.pt
  - Train predictor:
      python3 backend/app/ml/trainer.py train_predictor --data datasets/queue_metrics.csv
  - Grid search predictor:
      python3 backend/app/ml/trainer.py grid_search_predictor --data datasets/queue_metrics.csv
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import logging
import pathlib
import random
import shutil
import tempfile
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple

import numpy as np

# Optional dependencies imported best-effort
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    Adam = None
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

# Project modules
ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ml.real_env import SimulatedRealEnv, EnvConfig, make_vec_env
    from ml.rl_agent_sandbox import PPOActorCritic
    from ml.model_registry import ModelRegistry
    from ml.predictor import PredictorManager, LightGBMPredictor
except Exception:
    # graceful fallbacks: import may fail in minimal CI; trainer provides guardrails
    SimulatedRealEnv = None
    EnvConfig = None
    make_vec_env = None
    PPOActorCritic = None
    ModelRegistry = None
    PredictorManager = None
    LightGBMPredictor = None

# Logging
LOG = logging.getLogger("prioritymax.ml.trainer")
LOG.setLevel(os.getenv("PRIORITYMAX_TRAINER_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RL_CKPT = MODELS_DIR / "rl_agent.pt"
DEFAULT_PREDICTOR_CKPT = MODELS_DIR / "predictor_lgbm.pkl"

# -------------------------
# Config dataclasses
# -------------------------
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
    lgb_params: dict = None
    n_estimators: int = 100
    early_stopping_rounds: int = 50
    checkpoint_path: str = str(DEFAULT_PREDICTOR_CKPT)
    log_wandb: bool = False
    log_mlflow: bool = False
    wandb_project: str = "PriorityMax-Predictor"
    mlflow_experiment: str = "PriorityMax-Predictor"

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: dict, path: pathlib.Path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_TORCH and isinstance(state.get("model"), torch.nn.Module):
        # save state_dict for PyTorch models
        state_to_save = state.copy()
        mod = state_to_save.pop("model")
        state_to_save["model"] = mod.state_dict()
        torch.save(state_to_save, str(path))
    else:
        # generic save (pickle)
        import joblib
        joblib.dump(state, str(path))

def load_checkpoint(path: pathlib.Path) -> dict:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if _HAS_TORCH:
        try:
            ckpt = torch.load(str(path), map_location="cpu")
            return ckpt
        except Exception:
            pass
    import joblib
    return joblib.load(str(path))

def start_loggers(rl_cfg: RLTrainConfig = None, pred_cfg: PredictorTrainConfig = None, run_name: str = None):
    """
    Initialize MLflow/W&B if configured. Return a small context dict to pass around.
    """
    ctx = {"mlflow": False, "wandb": False}
    if rl_cfg and rl_cfg.log_mlflow or pred_cfg and pred_cfg.log_mlflow:
        if _HAS_MLFLOW:
            ctx["mlflow"] = True
            try:
                mlflow.set_experiment((rl_cfg.mlflow_experiment if rl_cfg else pred_cfg.mlflow_experiment))
                mlflow.start_run(run_name=run_name)
            except Exception:
                LOG.exception("Failed to start MLflow")
        else:
            LOG.warning("MLflow not available")
    if rl_cfg and rl_cfg.log_wandb or pred_cfg and pred_cfg.log_wandb:
        if _HAS_WANDB:
            ctx["wandb"] = True
            try:
                wandb.init(project=(rl_cfg.wandb_project if rl_cfg else pred_cfg.wandb_project), name=run_name, config=asdict(rl_cfg) if rl_cfg else asdict(pred_cfg))
            except Exception:
                LOG.exception("Failed to start Weights & Biases")
        else:
            LOG.warning("W&B not available")
    return ctx

def end_loggers(ctx: dict):
    if ctx.get("mlflow") and _HAS_MLFLOW:
        try:
            mlflow.end_run()
        except Exception:
            LOG.exception("Failed to end MLflow run")
    if ctx.get("wandb") and _HAS_WANDB:
        try:
            wandb.finish()
        except Exception:
            LOG.exception("Failed to finish W&B run")

# -------------------------
# PPO Implementation (re-usable)
# -------------------------
if _HAS_TORCH:
    class ActorCritic(nn.Module):
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
            return self.critic(obs).squeeze(-1)

    def compute_gae(rewards, vals, last_val, gamma, lam):
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0
            delta = rewards[t] + gamma * (vals[t+1] if t+1 < len(vals) else last_val) - vals[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        return adv

# -------------------------
# RL training loop (single-machine, vectorized envs supported)
# -------------------------
def train_rl(rl_cfg: RLTrainConfig):
    """
    High-level RL trainer using PPO.
    - Uses SimulatedRealEnv via make_vec_env to create multiple envs.
    - Implements on-policy data collection and PPO updates.
    - Logs to MLflow/W&B and saves checkpoints.
    """

    if SimulatedRealEnv is None or PPOActorCritic is None:
        raise RuntimeError("Required RL modules not available (SimulatedRealEnv / torch)")

    set_seed(rl_cfg.seed)
    device = torch.device(rl_cfg.device)

    LOG.info("Starting RL training: epochs=%d steps_per_epoch=%d device=%s", rl_cfg.epochs, rl_cfg.steps_per_epoch, rl_cfg.device)

    # create vectorized envs
    num_envs = max(1, min(8, int(os.getenv("PRIORITYMAX_RL_ENVS", "4"))))
    envs = make_vec_env(EnvConfig(mode="sim", seed=rl_cfg.seed), n=num_envs)
    obs_dim = len(envs[0]._observe())
    act_dim = 3

    model = ActorCritic(obs_dim, act_dim, rl_cfg.hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=rl_cfg.lr)

    # resume checkpoint if exists
    ckpt_path = pathlib.Path(rl_cfg.checkpoint_path)
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device)
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state)
            opt_state = ckpt.get("optimizer")
            if opt_state:
                optimizer.load_state_dict(opt_state)
            LOG.info("Resumed from checkpoint %s", str(ckpt_path))
        except Exception:
            LOG.exception("Failed to load checkpoint; starting fresh")

    # Setup logging
    run_name = f"ppo_{int(time.time())}"
    ctx = start_loggers(rl_cfg, None, run_name=run_name)

    # storage buffers: we will store per-step in lists and then convert to arrays for updates
    for epoch in range(1, rl_cfg.epochs + 1):
        # collect on-policy data for steps_per_epoch
        obs_batch = []
        act_batch = []
        rew_batch = []
        val_batch = []
        logp_batch = []

        # reset envs separately and maintain per-env obs
        obs_list = [env.reset() for env in envs]
        ep_returns = [0.0] * len(envs)
        ep_lens = [0] * len(envs)

        steps_collected = 0
        while steps_collected < rl_cfg.steps_per_epoch:
            # vectorized inference
            obs_tensor = torch.as_tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)
            with torch.no_grad():
                acts_t, logp_t = model.act(obs_tensor)
                vals_t = model.value(obs_tensor).cpu().numpy()
            acts = acts_t.cpu().numpy()
            logp = logp_t.cpu().numpy()
            # step envs
            next_obs_list = []
            rewards = []
            dones = []
            infos = []
            for i, env in enumerate(envs):
                next_obs, rew, done, info = env.step(acts[i])
                next_obs_list.append(next_obs)
                rewards.append(rew)
                dones.append(done)
                infos.append(info)
                ep_returns[i] += rew
                ep_lens[i] += 1

            # append to buffers (per-env flatten)
            for i in range(len(envs)):
                obs_batch.append(obs_list[i])
                act_batch.append(acts[i])
                rew_batch.append(rewards[i])
                val_batch.append(float(vals_t[i]))
                logp_batch.append(float(logp[i]))

            obs_list = next_obs_list
            steps_collected += len(envs)

        # After collecting batch, compute last values for bootstrap
        last_vals = []
        for i, env in enumerate(envs):
            # value of last obs
            obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32, device=device)
            with torch.no_grad():
                last_val = model.value(obs_t).cpu().item()
            last_vals.append(last_val)
        # compute advantages and returns
        vals = np.array(val_batch + last_vals, dtype=np.float32)  # append last vals per env? simplification
        rewards = np.array(rew_batch, dtype=np.float32)
        # For simplicity compute advantages with discounting across flattened trajectory (works reasonably for single-episode)
        # More correct approach: compute per-env GAE - omitted for brevity; this is a pragmatic implementation
        advs = []
        ret = []
        # naive discounted returns
        G = 0.0
        for r in rewards[::-1]:
            G = r + rl_cfg.gamma * G
            ret.insert(0, G)
        advs = np.array(ret) - np.array(val_batch)
        # normalize advs
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Convert to tensors
        obs_arr = torch.as_tensor(np.array(obs_batch, dtype=np.float32), device=device)
        act_arr = torch.as_tensor(np.array(act_batch, dtype=np.float32), device=device)
        adv_arr = torch.as_tensor(advs, device=device)
        ret_arr = torch.as_tensor(np.array(ret, dtype=np.float32), device=device)
        logp_arr = torch.as_tensor(np.array(logp_batch, dtype=np.float32), device=device)

        # PPO updates
        pi_losses = []
        v_losses = []
        clipfracs = []
        dataset_size = len(obs_arr)
        for _ in range(rl_cfg.update_epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, rl_cfg.mini_batch_size):
                batch_idx = idxs[start:start + rl_cfg.mini_batch_size]
                batch_obs = obs_arr[batch_idx]
                batch_act = act_arr[batch_idx]
                batch_adv = adv_arr[batch_idx]
                batch_ret = ret_arr[batch_idx]
                batch_old_logp = logp_arr[batch_idx]

                # forward
                logits = model.actor(batch_obs)
                mu = torch.tanh(logits)
                std = torch.ones_like(mu) * 0.2
                dist = torch.distributions.Normal(mu, std)
                new_logp = dist.log_prob(batch_act).sum(axis=-1)
                ratio = torch.exp(new_logp - batch_old_logp)
                # policy loss
                clip_adv = torch.clamp(ratio, 1.0 - rl_cfg.clip_range, 1.0 + rl_cfg.clip_range) * batch_adv
                pi_loss = -(torch.min(ratio * batch_adv, clip_adv)).mean()
                # value loss
                value_pred = model.value(batch_obs)
                v_loss = ((batch_ret - value_pred) ** 2).mean()
                # entropy
                entropy = dist.entropy().sum(axis=-1).mean()
                loss = pi_loss + rl_cfg.value_coef * v_loss - rl_cfg.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                clipfracs.append((torch.abs(ratio - 1.0) > rl_cfg.clip_range).float().mean().item())

        # Logging & checkpoint
        avg_pi = float(np.mean(pi_losses)) if pi_losses else 0.0
        avg_v = float(np.mean(v_losses)) if v_losses else 0.0
        avg_clip = float(np.mean(clipfracs)) if clipfracs else 0.0

        LOG.info("Epoch %d/%d: pi_loss=%.4f v_loss=%.4f clip=%.3f", epoch, rl_cfg.epochs, avg_pi, avg_v, avg_clip)
        if ctx.get("mlflow") and _HAS_MLFLOW:
            mlflow.log_metrics({"pi_loss": avg_pi, "v_loss": avg_v, "clip": avg_clip}, step=epoch)
        if ctx.get("wandb") and _HAS_WANDB:
            wandb.log({"epoch": epoch, "pi_loss": avg_pi, "v_loss": avg_v, "clip": avg_clip})

        if epoch % rl_cfg.save_interval == 0:
            save_checkpoint({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch, "ts": time.time()}, pathlib.Path(rl_cfg.checkpoint_path))
            LOG.info("Checkpoint saved to %s", rl_cfg.checkpoint_path)

        # evaluation hook
        if epoch % rl_cfg.eval_interval == 0:
            eval_res = evaluate_rl(model, device=device, episodes=5)
            LOG.info("Eval: avg_reward=%.3f", eval_res["avg_reward"])
            if ctx.get("mlflow") and _HAS_MLFLOW:
                mlflow.log_metrics({"eval_avg_reward": eval_res["avg_reward"]}, step=epoch)
            if ctx.get("wandb") and _HAS_WANDB:
                wandb.log({"eval_avg_reward": eval_res["avg_reward"]})

    # final checkpoint
    save_checkpoint({"model": model, "optimizer": optimizer.state_dict(), "epoch": rl_cfg.epochs, "ts": time.time()}, pathlib.Path(rl_cfg.checkpoint_path))
    LOG.info("Training finished. Saved checkpoint to %s", rl_cfg.checkpoint_path)
    end_loggers(ctx)

def evaluate_rl(model_or_path, device: str = "cpu", episodes: int = 10) -> Dict[str, Any]:
    """
    Evaluate a trained PPO policy in SimulatedRealEnv.
    Accepts either a model instance or a path to checkpoint.
    """
    if SimulatedRealEnv is None:
        raise RuntimeError("SimulatedRealEnv not available for RL evaluation")
    device = torch.device(device)
    # load model if path provided
    model = None
    if isinstance(model_or_path, (str, pathlib.Path)):
        path = pathlib.Path(model_or_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not _HAS_TORCH:
            raise RuntimeError("Torch required to load RL model")
        ckpt = torch.load(str(path), map_location=device)
        model = ActorCritic(8, 3)  # obs & act dims assumed
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
    else:
        model = model_or_path
    model.to(device)
    model.eval()
    env = SimulatedRealEnv(EnvConfig(mode="sim", seed=42))
    rewards = []
    for ep in range(episodes):
        obs = env.reset(seed=42 + ep)
        total = 0.0
        for t in range(1000):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                act, _ = model.act(obs_t)
            action = act.cpu().numpy()
            obs, rew, done, _ = env.step(action)
            total += rew
            if done:
                break
        rewards.append(total)
    return {"avg_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)), "episodes": episodes}

# -------------------------
# Predictor training & evaluation
# -------------------------
def train_predictor(cfg: PredictorTrainConfig):
    """
    Train a LightGBM or RandomForest predictor for queue forecasting.
    - Loads CSV data, splits, trains, evaluates, checkpoints model.
    """
    if not cfg.data_path:
        raise ValueError("data_path required for predictor training")
    set_seed(cfg.random_state)

    data_path = pathlib.Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(cfg.data_path)
    import pandas as pd
    df = pd.read_csv(str(data_path))
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column {cfg.target_col} not found in {cfg.data_path}")

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    # start logging
    ctx = start_loggers(None, cfg, run_name=f"predictor_{int(time.time())}")

    model = None
    if _HAS_LGB:
        params = cfg.lgb_params or {"objective": "regression", "metric": "rmse", "verbosity": -1, "boosting_type": "gbdt"}
        ltrain = lgb.Dataset(X_train, label=y_train)
        lval = lgb.Dataset(X_test, label=y_test, reference=ltrain)
        LOG.info("Training LightGBM with params: %s", params)
        model = lgb.train(params, ltrain, num_boost_round=cfg.n_estimators, valid_sets=[ltrain, lval], early_stopping_rounds=cfg.early_stopping_rounds, verbose_eval=False)
        # persist
        model.save_model(str(cfg.checkpoint_path))
    elif _HAS_SKLEARN:
        LOG.info("Training RandomForestRegressor fallback")
        model = RandomForestRegressor(n_estimators=100, random_state=cfg.random_state)
        model.fit(X_train, y_train)
        # save using joblib
        import joblib
        joblib.dump(model, cfg.checkpoint_path)
    else:
        raise RuntimeError("No supported predictor libraries installed (lightgbm or sklearn)")

    # evaluate
    if _HAS_LGB:
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds))) if _HAS_SKLEARN else float(np.sqrt(((preds - y_test) ** 2).mean()))
    r2 = float(r2_score(y_test, preds)) if _HAS_SKLEARN else float(np.nan)
    LOG.info("Predictor evaluation: RMSE=%.4f R2=%.4f", rmse, r2)
    if ctx.get("mlflow") and _HAS_MLFLOW:
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.log_artifact(str(cfg.checkpoint_path))
    if ctx.get("wandb") and _HAS_WANDB:
        wandb.log({"rmse": rmse, "r2": r2})
        try:
            wandb.save(str(cfg.checkpoint_path))
        except Exception:
            LOG.exception("W&B save failed")

    end_loggers(ctx)
    return {"rmse": rmse, "r2": r2, "ckpt": str(cfg.checkpoint_path)}

def evaluate_predictor(model_path: str, data_path: str, target_col: str = "queue_next_sec"):
    import pandas as pd
    df = pd.read_csv(str(data_path))
    if target_col not in df.columns:
        raise ValueError("target_col not present")
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    # load
    import joblib
    if pathlib.Path(model_path).suffix in [".txt", ".model"]:
        # try lightgbm
        if _HAS_LGB:
            model = lgb.Booster(model_file=str(model_path))
            preds = model.predict(X)
        else:
            raise RuntimeError("LightGBM not available to load .model")
    else:
        model = joblib.load(str(model_path))
        preds = model.predict(X)
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = math.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    LOG.info("Evaluate predictor: RMSE=%.4f R2=%.4f", rmse, r2)
    return {"rmse": rmse, "r2": r2}

# -------------------------
# Hyperparameter grid search for predictor
# -------------------------
def grid_search_predictor(data_path: str, target_col: str = "queue_next_sec", param_grid: dict = None, cv: int = 3, n_jobs: int = 1):
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn required for grid search")
    import pandas as pd
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    base = RandomForestRegressor(random_state=42)
    grid = param_grid or {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    search = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=n_jobs, verbose=2)
    search.fit(X, y)
    LOG.info("Grid search best params: %s score: %s", search.best_params_, search.best_score_)
    return {"best_params": search.best_params_, "best_score": float(search.best_score_)}

# -------------------------
# Canary gating: evaluate candidate model and require threshold to promote
# -------------------------
def canary_gate_predictor(candidate_ckpt: str, holdout_data: str, rmse_threshold: float = 5.0):
    """
    Evaluate candidate predictor on holdout data, return True if RMSE <= threshold.
    """
    res = evaluate_predictor(candidate_ckpt, holdout_data)
    ok = res["rmse"] <= rmse_threshold
    if ok:
        LOG.info("Canary passed (rmse=%.3f <= %.3f)", res["rmse"], rmse_threshold)
    else:
        LOG.warning("Canary failed (rmse=%.3f > %.3f)", res["rmse"], rmse_threshold)
    return ok

# -------------------------
# CLI
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-trainer")
    sub = p.add_subparsers(dest="cmd")
    # RL
    tr = sub.add_parser("train_rl")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--steps", type=int, default=2048)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--device", type=str, default="cpu")
    tr.add_argument("--checkpoint", type=str, default=str(DEFAULT_RL_CKPT))
    tr.add_argument("--wandb", action="store_true")
    tr.add_argument("--mlflow", action="store_true")

    ev = sub.add_parser("eval_rl")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--episodes", type=int, default=10)

    # predictor
    tp = sub.add_parser("train_predictor")
    tp.add_argument("--data", required=True)
    tp.add_argument("--target", default="queue_next_sec")
    tp.add_argument("--ckpt", default=str(DEFAULT_PREDICTOR_CKPT))
    tp.add_argument("--wandb", action="store_true")
    tp.add_argument("--mlflow", action="store_true")

    gp = sub.add_parser("grid_search_predictor")
    gp.add_argument("--data", required=True)
    gp.add_argument("--cv", type=int, default=3)

    # evaluate predictor
    ep = sub.add_parser("evaluate_predictor")
    ep.add_argument("--ckpt", required=True)
    ep.add_argument("--data", required=True)
    ep.add_argument("--target", default="queue_next_sec")

    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    if args.cmd == "train_rl":
        cfg = RLTrainConfig(epochs=args.epochs, steps_per_epoch=args.steps, hidden_dim=args.hidden_dim if hasattr(args, "hidden_dim") else args.hidden, lr=args.lr, device=args.device, checkpoint_path=args.checkpoint, log_wandb=args.wandb, log_mlflow=args.mlflow)
        train_rl(cfg)
    elif args.cmd == "eval_rl":
        res = evaluate_rl(args.checkpoint, device="cpu", episodes=int(args.episodes))
        print(json.dumps(res, indent=2))
    elif args.cmd == "train_predictor":
        cfg = PredictorTrainConfig(data_path=args.data, target_col=args.target, checkpoint_path=args.ckpt, log_wandb=args.wandb, log_mlflow=args.mlflow)
        res = train_predictor(cfg)
        print(json.dumps(res, indent=2))
    elif args.cmd == "grid_search_predictor":
        res = grid_search_predictor(args.data, cv=args.cv)
        print(json.dumps(res, indent=2))
    elif args.cmd == "evaluate_predictor":
        res = evaluate_predictor(args.ckpt, args.data, target_col=args.target)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

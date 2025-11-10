# backend/app/ml/trainer_distributed.py
"""
PriorityMax Distributed Trainer & Hyperparameter Tuning
------------------------------------------------------

Extended production-grade trainer that provides:

 - RLlib (Ray) PPO launcher for distributed RL training
 - Ray Tune integration for scalable hyperparameter search (grid, random, ASHA)
 - Optuna adapter / convenience for predictor tuning
 - Conversion helpers (RLlib -> Torch checkpoint best-effort)
 - Torch.distributed helper for multi-node/DP training (simple launcher)
 - MLflow / W&B integration helpers
 - CLI with many conveniences for local vs cluster runs
 - Defensive imports and helpful error messages

Save at: backend/app/ml/trainer_distributed.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import logging
import pathlib
import tempfile
import shutil
import signal
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Callable, Tuple, Union

# -------------------------
# Optional heavy dependencies (best-effort imports)
# -------------------------
# Ray & RLlib & Tune
try:
    import ray
    from ray import tune
    from ray import air
    from ray.rllib.agents import ppo as rllib_ppo
    from ray.tune.registry import register_env
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandForBOHB
    _HAS_RAY = True
except Exception:
    ray = tune = air = rllib_ppo = register_env = OptunaSearch = ASHAScheduler = PopulationBasedTraining = HyperBandForBOHB = None
    _HAS_RAY = False

# Optuna
try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    optuna = None
    _HAS_OPTUNA = False

# torch / torch.distributed
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    dist = None
    DDP = None
    _HAS_TORCH = False

# MLflow and wandb
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

# local project imports (best-effort)
ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ml.trainer import RLTrainConfig, PredictorTrainConfig, save_checkpoint, load_checkpoint, train_predictor, train_rl, evaluate_rl
    from ml.real_env import SimulatedRealEnv, EnvConfig, make_vec_env
    from ml.rl_agent_sandbox import PPOActorCritic
    from ml.model_registry import ModelRegistry
except Exception:
    # Provide graceful placeholders
    RLTrainConfig = None
    PredictorTrainConfig = None
    save_checkpoint = None
    load_checkpoint = None
    train_predictor = None
    train_rl = None
    evaluate_rl = None
    SimulatedRealEnv = None
    EnvConfig = None
    make_vec_env = None
    PPOActorCritic = None
    ModelRegistry = None

# Logging
LOG = logging.getLogger("prioritymax.trainer_distributed")
LOG.setLevel(os.getenv("PRIORITYMAX_TRAINER_DISTR_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TUNE_DIR = MODELS_DIR / "tune"

# -------------------------
# Dataclasses
# -------------------------
@dataclass
class RayClusterConfig:
    """
    Controls how ray.init is called for local / cluster runs.
    - address: None for local, "auto" or "ray://..." for clusters
    - local_mode: True for debugging inside same process
    - num_cpus, num_gpus: allocate resources (local mode)
    - redis_password: optional (cluster auth)
    """
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
    scheduler: Optional[str] = "asha"  # none|asha|pbt
    metric: str = "episode_reward_mean"
    mode: str = "max"
    local_dir: Optional[str] = None

# -------------------------
# Ray helpers
# -------------------------
def ensure_ray_initialized(cfg: RayClusterConfig):
    if not _HAS_RAY:
        raise RuntimeError("Ray is not installed. Install ray[rllib] and ray[tune] to use distributed training.")
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
    LOG.info("Calling ray.init(%s)", init_kwargs)
    ray.init(**init_kwargs, _ignore_reinit_error=cfg.ignore_reinit_error)

# -------------------------
# Env registry for RLlib
# -------------------------
def register_prioritymax_env():
    """
    Register a function 'PriorityMaxSimEnv' with RLlib that wraps SimulatedRealEnv.
    This allows RLlib trainer to create environments on workers.
    """
    if not _HAS_RAY:
        LOG.debug("Ray not available; skipping env registration")
        return

    def _creator(env_config):
        if SimulatedRealEnv is None or EnvConfig is None:
            raise RuntimeError("SimulatedRealEnv not available in runtime")
        cfg = EnvConfig(mode=env_config.get("mode", "sim"), seed=env_config.get("seed", 0))
        return SimulatedRealEnv(cfg)

    try:
        register_env("PriorityMaxSimEnv", lambda cfg: _creator(cfg))
        LOG.info("Registered RLlib env 'PriorityMaxSimEnv'")
    except Exception:
        LOG.exception("Failed to register RLlib env")

# -------------------------
# RLlib training launcher
# -------------------------
def run_rllib_ppo(rl_cfg: RLTrainConfig,
                  ray_cfg: Optional[RayClusterConfig] = None,
                  rllib_overrides: Optional[Dict[str, Any]] = None,
                  checkpoint_dir: Optional[str] = None,
                  restore_checkpoint: Optional[str] = None):
    """
    Launch a RLlib PPO training job using provided RLTrainConfig.

    This function:
    - Initializes Ray (local or cluster)
    - Registers environment
    - Builds RLlib config and creates PPOTrainer
    - Runs training iterations, logs, and saves checkpoints to checkpoint_dir
    - Optionally converts an RLlib checkpoint into a torch-friendly checkpoint
      for compatibility with the rest of the codebase (best-effort).

    NOTE: RLTrainConfig is expected to include fields like epochs, steps_per_epoch, lr, etc.
    """
    if not _HAS_RAY:
        raise RuntimeError("Ray and RLlib are required to run RLlib training.")

    ray_cfg = ray_cfg or RayClusterConfig()
    ensure_ray_initialized(ray_cfg)
    register_prioritymax_env()

    # Build base RLlib config
    rllib_config = {
        "env": "PriorityMaxSimEnv",
        "framework": "torch",
        "num_workers": max(0, (ray_cfg.num_cpus or 1) - 1),
        "num_gpus": (ray_cfg.num_gpus or 0),
        "train_batch_size": getattr(rl_cfg, "steps_per_epoch", 2048),
        "sgd_minibatch_size": getattr(rl_cfg, "mini_batch_size", 256),
        "model": {"fcnet_hiddens": [getattr(rl_cfg, "hidden_dim", 128), getattr(rl_cfg, "hidden_dim", 128)]},
        "lr": getattr(rl_cfg, "lr", 3e-4),
        "gamma": getattr(rl_cfg, "gamma", 0.99),
        "lambda": getattr(rl_cfg, "lam", 0.95),
        "clip_param": getattr(rl_cfg, "clip_range", 0.2),
        "entropy_coeff": getattr(rl_cfg, "entropy_coef", 0.01),
        "vf_loss_coeff": getattr(rl_cfg, "value_coef", 0.5),
        "num_sgd_iter": getattr(rl_cfg, "update_epochs", 10),
        "log_level": "INFO",
        # If users want observation normalization in RLlib, they can opt-in here.
    }

    # Merge overrides
    if rllib_overrides:
        rllib_config.update(rllib_overrides)

    # Create trainer
    trainer = rllib_ppo.PPOTrainer(config=rllib_config, env="PriorityMaxSimEnv")

    # Optionally restore from RLlib checkpoint
    if restore_checkpoint:
        try:
            LOG.info("Restoring RLlib trainer from checkpoint: %s", restore_checkpoint)
            trainer.restore(restore_checkpoint)
        except Exception:
            LOG.exception("Failed to restore RLlib checkpoint")

    checkpoint_dir = checkpoint_dir or str(MODELS_DIR / "rllib_checkpoints")
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    epochs = int(getattr(rl_cfg, "epochs", 100))
    save_interval = int(getattr(rl_cfg, "save_interval", max(10, epochs // 10)))

    try:
        for it in range(1, epochs + 1):
            res = trainer.train()
            LOG.info("[RLlib] Iter %d/%d: reward_mean=%s", it, epochs, res.get("episode_reward_mean"))
            # Optional: MLflow/W&B instrumentation could be added here
            if it % save_interval == 0 or it == epochs:
                ckpt = trainer.save(checkpoint_dir)
                LOG.info("[RLlib] Saved checkpoint at %s", ckpt)
                # Attempt to convert to standard PyTorch checkpoint for consumption by rl_agent_prod
                try:
                    converted = convert_rllib_checkpoint_to_torch(trainer, ckpt, ModelsDir=MODELS_DIR, iter_id=it)
                    LOG.info("[RLlib] Converted and wrote torch checkpoint: %s", converted)
                except Exception:
                    LOG.exception("[RLlib] Checkpoint conversion failed (non-fatal)")
    finally:
        trainer.stop()
    return True

# -------------------------
# Convert RLlib checkpoint to torch (best-effort)
# -------------------------
def convert_rllib_checkpoint_to_torch(trainer, rllib_checkpoint_path: str, ModelsDir: pathlib.Path = MODELS_DIR, iter_id: Optional[int] = None) -> Optional[str]:
    """
    Best-effort conversion: Extract policy weights/state from RLlib trainer and save a PyTorch-style checkpoint.
    Note: RLlib policies may serialize in different formats; this function tries common patterns.
    Returns path to converted file or None.
    """
    ModelsDir.mkdir(parents=True, exist_ok=True)
    out_path = ModelsDir / f"rl_agent_rllib_iter{iter_id or int(time.time())}.pt"
    try:
        # RLlib policy state retrieval logic (best-effort; may differ across RLlib versions)
        policy = trainer.get_policy()
        # Some versions provide get_state or get_weights
        state = None
        if hasattr(policy, "get_state"):
            state = policy.get_state()
        elif hasattr(policy, "get_weights"):
            state = policy.get_weights()
        else:
            # try to find nested model state
            try:
                model_state = policy.model.state_dict()
                state = {"model": model_state}
            except Exception:
                state = None
        # If policy returns numpy arrays etc., save generic
        if state is None:
            LOG.warning("Unable to extract policy state from RLlib trainer")
            # fallback: save small metadata file pointing to RLlib checkpoint
            meta = {"rllib_checkpoint": rllib_checkpoint_path, "ts": time.time()}
            meta_path = ModelsDir / f"rl_agent_rllib_meta_{int(time.time())}.json"
            meta_path.write_text(json.dumps(meta))
            return str(meta_path)
        # Try to coerce to torch state_dict if possible
        try:
            import torch
            # If state already contains 'model' or torch tensors, save as-is
            if isinstance(state, dict) and any(isinstance(v, (torch.Tensor, dict)) for v in state.values()):
                torch.save({"model": state}, str(out_path))
            else:
                # Save pickled JSON-ish state
                torch.save({"rllib_state": state}, str(out_path))
            return str(out_path)
        except Exception:
            # fallback: write JSON metadata
            out_meta = ModelsDir / f"rl_agent_rllib_state_{int(time.time())}.json"
            out_meta.write_text(json.dumps({"state_sample": str(type(state)), "rllib_checkpoint": rllib_checkpoint_path}, default=str))
            return str(out_meta)
    except Exception:
        LOG.exception("Failed to convert RLlib checkpoint")
        return None

# -------------------------
# Ray Tune utilities
# -------------------------
def build_tune_scheduler(name: Optional[str] = None, metric: str = "episode_reward_mean", mode: str = "max"):
    """
    Return a scheduler instance supported by Ray Tune
    name: 'asha' or 'pbt' or None
    """
    if name is None:
        return None
    if name == "asha":
        return ASHAScheduler(metric=metric, mode=mode)
    if name == "pbt":
        return PopulationBasedTraining(time_attr="training_iteration", perturbation_interval=5)
    # add more schedulers as needed
    return None

def make_rllib_trainable_for_tune(base_rl_cfg: RLTrainConfig):
    """
    Create a Ray Tune trainable function wrapper that runs short RLlib training inside each trial.
    This wrapper merges trial-specific config overrides into base_rl_cfg and runs a short run,
    reporting metrics back to Tune.
    """
    if not _HAS_RAY:
        raise RuntimeError("Ray is required to use tune wrappers")

    def trainable(config, checkpoint_dir=None):
        # merge config into base config
        merged = asdict(base_rl_cfg)
        merged.update(config)
        # create a lightweight RLlib config
        rllib_cfg = {
            "env": "PriorityMaxSimEnv",
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            "train_batch_size": int(merged.get("steps_per_epoch", 1024)),
            "sgd_minibatch_size": int(merged.get("mini_batch_size", 256)),
            "model": {"fcnet_hiddens": [merged.get("hidden_dim", 128), merged.get("hidden_dim", 128)]},
            "lr": merged.get("lr", 3e-4),
            "clip_param": merged.get("clip_range", 0.2),
            "num_sgd_iter": merged.get("update_epochs", 3),
            "log_level": "WARN",
        }
        register_prioritymax_env()
        trainer = rllib_ppo.PPOTrainer(config=rllib_cfg, env="PriorityMaxSimEnv")
        # run small number of iterations (controlled by config 'train_iters')
        niters = int(config.get("train_iters", 3))
        for it in range(niters):
            res = trainer.train()
            # report the primary metric (episode_reward_mean) to tune
            tune.report(episode_reward_mean=res.get("episode_reward_mean", float("nan")))
        # optionally save checkpoint for the trial
        ck = trainer.save()
        tune.report(trial_checkpoint=ck)
        trainer.stop()
    return trainable

def run_tune_experiment(base_rl_cfg: RLTrainConfig,
                        search_space: Dict[str, Any],
                        tune_cfg: TuneJobConfig,
                        ray_cfg: Optional[RayClusterConfig] = None,
                        use_optuna: bool = False):
    """
    Run a Ray Tune experiment using provided base RL config and search space.
    - search_space uses tune.choice, tune.loguniform, etc.
    - tune_cfg controls scheduling, samples, and resource config.
    """
    if not _HAS_RAY:
        raise RuntimeError("Ray is required for Tune experiments.")

    ray_cfg = ray_cfg or RayClusterConfig()
    ensure_ray_initialized(ray_cfg)
    register_prioritymax_env()

    trainable = make_rllib_trainable_for_tune(base_rl_cfg)
    scheduler = build_tune_scheduler(tune_cfg.scheduler, metric=tune_cfg.metric, mode=tune_cfg.mode)

    search_alg = None
    if use_optuna:
        if not _HAS_OPTUNA:
            raise RuntimeError("Optuna required for OptunaSearch")
        search_alg = OptunaSearch(metric=tune_cfg.metric, mode=tune_cfg.mode)

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
    LOG.info("Tune experiment finished. Best config: %s", analysis.get_best_config(metric=tune_cfg.metric, mode=tune_cfg.mode))
    return analysis

# -------------------------
# Optuna utilities for predictor tuning
# -------------------------
def optuna_lightgbm_objective(data_path: str, target_col: str, trial: "optuna.trial.Trial"):
    """
    Example Optuna objective to tune LightGBM hyperparameters for predictor training.
    This is purposely compact to run many trials quickly; extend for full production tuning.
    """
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna is not installed")

    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not in CSV")

    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    param = {
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

    gbm = lgb.train(param, dtrain, num_boost_round=500, valid_sets=[dval], early_stopping_rounds=20, verbose_eval=False)
    preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    return rmse

def run_optuna_predictor_study(data_path: str, target_col: str = "queue_next_sec", n_trials: int = 50, storage: Optional[str] = None, study_name: Optional[str] = None):
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna not installed")
    study_name = study_name or f"prioritymax_predictor_{int(time.time())}"
    study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name, load_if_exists=True)
    study.optimize(lambda t: optuna_lightgbm_objective(data_path, target_col, t), n_trials=n_trials)
    LOG.info("Optuna study complete. Best value: %s", study.best_value)
    return study

# -------------------------
# Torch.distributed launcher helper
# -------------------------
def launch_torch_distributed(world_size: int, node_rank: int = 0, dist_backend: str = "nccl", dist_url: Optional[str] = None, run_fn: Optional[Callable] = None):
    """
    Launch a simple torch.distributed run on the local machine across `world_size` processes.
    - world_size: total processes
    - node_rank: current node rank (0 if single-node)
    - dist_url: 'tcp://127.0.0.1:port' or env-provided URL
    - run_fn: function to be executed in each spawned process with signature fn(local_rank, world_size)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for torch.distributed")

    if dist_url is None:
        # pick a random port
        port = 29500 + (os.getpid() % 1000)
        dist_url = f"tcp://127.0.0.1:{port}"

    LOG.info("Launching torch.distributed: world_size=%d url=%s", world_size, dist_url)

    # spawn processes (use torch.multiprocessing.spawn)
    try:
        import torch.multiprocessing as mp
        def _entry(local_rank):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(int(dist_url.split(":")[-1]))
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(node_rank * world_size + local_rank)
            # initialize process group
            dist.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=int(os.environ["RANK"]))
            try:
                if run_fn:
                    run_fn(local_rank, world_size)
            finally:
                dist.destroy_process_group()
        mp.spawn(_entry, nprocs=world_size)
    except Exception:
        LOG.exception("Failed to spawn torch distributed processes")
        raise

# -------------------------
# Utility: MLflow & W&B helpers
# -------------------------
def init_mlflow_if_enabled(enabled: bool, experiment_name: str = "prioritymax"):
    if not enabled:
        return False
    if not _HAS_MLFLOW:
        LOG.warning("MLflow requested but not installed")
        return False
    try:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=f"{experiment_name}_{int(time.time())}")
        return True
    except Exception:
        LOG.exception("Failed to start MLflow")
        return False

def finish_mlflow_if_started(active: bool):
    if not active or not _HAS_MLFLOW:
        return
    try:
        mlflow.end_run()
    except Exception:
        LOG.exception("Failed to end MLflow run")

def init_wandb_if_enabled(enabled: bool, project: str = "PriorityMax"):
    if not enabled:
        return False
    if not _HAS_WANDB:
        LOG.warning("W&B requested but not installed")
        return False
    try:
        wandb.init(project=project, name=f"{project}_{int(time.time())}")
        return True
    except Exception:
        LOG.exception("Failed to init W&B")
        return False

def finish_wandb_if_started(active: bool):
    if not active or not _HAS_WANDB:
        return
    try:
        wandb.finish()
    except Exception:
        LOG.exception("Failed to finish W&B")

# -------------------------
# CLI glue
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-trainer-distributed")
    sub = p.add_subparsers(dest="cmd")

    # RLlib train
    rllib = sub.add_parser("rllib_train", help="Run RLlib (distributed) PPO training")
    rllib.add_argument("--epochs", type=int, default=200)
    rllib.add_argument("--steps", type=int, default=2048)
    rllib.add_argument("--cpus", type=int, default=4)
    rllib.add_argument("--gpus", type=int, default=0)
    rllib.add_argument("--checkpoint-dir", type=str, default=str(MODELS_DIR / "rllib_checkpoints"))
    rllib.add_argument("--restore", type=str, default=None)
    rllib.add_argument("--save-interval", type=int, default=10)

    # Tune search
    tune_p = sub.add_parser("tune", help="Run Ray Tune hyperparameter search")
    tune_p.add_argument("--name", type=str, default="prioritymax_tune")
    tune_p.add_argument("--samples", type=int, default=8)
    tune_p.add_argument("--scheduler", type=str, choices=["asha", "pbt", "none"], default="asha")
    tune_p.add_argument("--optuna", action="store_true", help="Use Optuna as search algorithm")
    tune_p.add_argument("--local-dir", type=str, default=str(TUNE_DIR))

    # Optuna predictor
    opt = sub.add_parser("optuna_predictor", help="Run Optuna for predictor hyperparams")
    opt.add_argument("--data", type=str, required=True)
    opt.add_argument("--target", type=str, default="queue_next_sec")
    opt.add_argument("--trials", type=int, default=50)
    opt.add_argument("--storage", type=str, default=None)

    # Torch.distributed local launcher
    td = sub.add_parser("torch_dist", help="Launch a simple torch.distributed job locally")
    td.add_argument("--world-size", type=int, default=2)

    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    if args.cmd == "rllib_train":
        from ml.trainer import RLTrainConfig as LocalRLTrainConfig  # local import for typing
        cfg = LocalRLTrainConfig(epochs=args.epochs, steps_per_epoch=args.steps, hidden_dim=128, lr=3e-4, save_interval=args.save_interval)
        ray_cfg = RayClusterConfig(num_cpus=args.cpus, num_gpus=args.gpus)
        run_rllib_ppo(cfg, ray_cfg, checkpoint_dir=args.checkpoint_dir, restore_checkpoint=args.restore)
    elif args.cmd == "tune":
        base_rl_cfg = RLTrainConfig(epochs=10, steps_per_epoch=1024, hidden_dim=128)
        tune_cfg = TuneJobConfig(name=args.name, num_samples=args.samples, scheduler=(None if args.scheduler == "none" else args.scheduler), local_dir=args.local_dir)
        # example search space - you should tune this
        search_space = {
            "lr": tune.loguniform(1e-5, 1e-3),
            "hidden_dim": tune.choice([64, 128, 256]),
            "train_iters": tune.choice([1, 2, 3]),
            "seed": tune.randint(0, 1000)
        }
        run_tune_experiment(base_rl_cfg, search_space, tune_cfg, ray_cfg=RayClusterConfig(local_mode=True), use_optuna=args.optuna)
    elif args.cmd == "optuna_predictor":
        study = run_optuna_predictor_study(args.data, args.target, n_trials=args.trials, storage=args.storage)
        print("Best trial:", study.best_trial.params, "value:", study.best_value)
    elif args.cmd == "torch_dist":
        def _dummy_fn(local_rank, world_size):
            import time
            LOG.info("Hello from local_rank=%d world_size=%d", local_rank, world_size)
            time.sleep(1)
        launch_torch_distributed(world_size=args.world_size, run_fn=_dummy_fn)
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

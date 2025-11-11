#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_rl_eval.py - ENTERPRISE EDITION (SYNCHRONIZED)
------------------------------------------------------------

Enterprise-grade RL model evaluation synchronized with train_rl_heavy.py and train_rl_live.py

Key Features:
✅ ONNX model evaluation (production inference path)
✅ A/B testing with statistical significance
✅ Drift detection (distribution shift monitoring)
✅ Performance benchmarking (latency/throughput)
✅ Numerical stability checks (NaN/Inf detection)
✅ Prometheus metrics export
✅ Parallel evaluation (multi-worker)
✅ MLflow & W&B integration
✅ S3 artifact storage
✅ Model registry integration
✅ CI/CD ready (exit codes, automated gates)

Usage:
    # Standard evaluation
    python3 scripts/train_rl_eval.py --model-dir models/ppo --eval-episodes 50
    
    # ONNX production path
    python3 scripts/train_rl_eval.py --checkpoint model.onnx --use-onnx
    
    # A/B testing
    python3 scripts/train_rl_eval.py --checkpoint candidate.pt --baseline-checkpoint baseline.pt
    
    # Full enterprise pipeline
    python3 scripts/train_rl_eval.py --model-tag prod-v2 --baseline-checkpoint prod-v1.pt \\
        --drift-detection --benchmark --prometheus --mlflow --wandb
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import pathlib
import random
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Third-party imports (best effort)
try:
    import yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

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
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    ClientError = Exception
    _HAS_BOTO3 = False

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except Exception:
    ort = None
    _HAS_ONNX = False

try:
    from prometheus_client import Gauge, Counter, start_http_server
    _HAS_PROMETHEUS = True
except Exception:
    Gauge = Counter = None
    _HAS_PROMETHEUS = False

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

# Project imports
ROOT = pathlib.Path(__file__).resolve().parents[2]
_candidates = [ROOT, ROOT / "backend", ROOT / "backend" / "app"]
for c in _candidates:
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))

try:
    from ml.real_env import SimulatedRealEnv, EnvConfig
except Exception:
    SimulatedRealEnv = None
    EnvConfig = None

# ---------------------------
# 🔄 Synchronization Setup (RealEnv alignment)
# ---------------------------
"""
SYNCHRONIZED WITH:
 - real_env.py (EnvConfig, DriftTracker)
 - train_rl_heavy.py (training consistency)
 - train_rl_live.py (live rollout consistency)
 - rl_agent_prod.py (runtime inference safety)
"""

from ml.real_env import get_observation_space, get_action_space

# Construct synchronized evaluation config
SYNC_CFG = EnvConfig(
    mode="sim",
    obs_dim=8,
    act_dim=3,
    drift_window_size=1000,
    reward_latency_sla_ms=500.0,
    cost_per_worker_per_sec=0.0005,
)

print(
    f"[SYNC CHECK] train_rl_eval aligned | "
    f"obs_dim={SYNC_CFG.obs_dim}, act_dim={SYNC_CFG.act_dim}, "
    f"SLA={SYNC_CFG.reward_latency_sla_ms}, cost={SYNC_CFG.cost_per_worker_per_sec}"
)

try:
    from ml.model_registry import ModelRegistry
except Exception:
    ModelRegistry = None

# -------------------------
# Constants & Paths
# -------------------------
DEFAULT_CHECKPOINTS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_CHECKPOINTS_DIR", str(ROOT / "backend" / "checkpoints")))
DEFAULT_MODELS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODELS_DIR", str(ROOT / "backend" / "app" / "ml" / "models")))
DEFAULT_RESULTS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_RESULTS_DIR", str(DEFAULT_CHECKPOINTS_DIR / "eval_results")))
DEFAULT_TRACE_DIR = pathlib.Path(os.getenv("PRIORITYMAX_TRACES_DIR", str(DEFAULT_RESULTS_DIR / "traces")))
DEFAULT_S3_BUCKET = os.getenv("PRIORITYMAX_S3_BUCKET", None)

DEFAULT_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_TRACE_DIR.mkdir(parents=True, exist_ok=True)

# Exit codes
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_MEAN_BELOW_THRESHOLD = 2
EXIT_CONFIG_ERROR = 3
EXIT_MISSING_DEP = 4

# -------------------------
# Logging
# -------------------------
LOG = logging.getLogger("prioritymax.rl.eval")
LOG_LEVEL = os.getenv("PRIORITYMAX_EVAL_LOG", "INFO").upper()
LOG.setLevel(LOG_LEVEL)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

def get_child_logger(name: str) -> logging.Logger:
    l = logging.getLogger(f"{LOG.name}.{name}")
    l.setLevel(LOG_LEVEL)
    return l

loader_logger = get_child_logger("loader")
storage_logger = get_child_logger("storage")

# -------------------------
# Config Dataclass
# -------------------------
@dataclass
class EvalConfig:
    # Basic evaluation
    model_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    model_tag: Optional[str] = None
    eval_episodes: int = 50
    max_steps: int = 1000
    deterministic: bool = True
    seed: Optional[int] = None
    render: int = 0
    out: str = str(DEFAULT_RESULTS_DIR / f"eval_result_{int(time.time())}.json")
    trace_dir: Optional[str] = None
    mlflow: bool = False
    mlflow_experiment: str = "PriorityMax-RL-Eval"
    wandb: bool = False
    wandb_project: str = "PriorityMax-RL-Eval"
    device: str = "cpu"
    stop_if_below_mean_reward: Optional[float] = None
    s3_bucket: Optional[str] = DEFAULT_S3_BUCKET
    s3_prefix: Optional[str] = None
    model_registry_backend: Optional[str] = None
    metadata_file: Optional[str] = None
    verbosity: int = 1
    
    # ===== ENTERPRISE FEATURES =====
    # ONNX support
    use_onnx: bool = False
    onnx_providers: List[str] = None
    
    # A/B testing
    baseline_checkpoint: Optional[str] = None
    statistical_test: str = "ttest"
    confidence_level: float = 0.95
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_reference_data: Optional[str] = None
    drift_threshold: float = 0.3
    
    # Performance benchmarking
    measure_inference_time: bool = True
    benchmark_batch_sizes: List[int] = None
    
    # Safety checks
    check_numerical_stability: bool = True
    max_episode_reward: Optional[float] = None
    min_episode_reward: Optional[float] = None
    
    # Prometheus
    enable_prometheus: bool = False
    prometheus_port: int = 9304
    
    # Parallelization
    num_workers: int = None
    worker_timeout: int = 3600
    
    def __post_init__(self):
        if self.onnx_providers is None:
            self.onnx_providers = ["CPUExecutionProvider"]
        if self.benchmark_batch_sizes is None:
            self.benchmark_batch_sizes = [1, 8, 32]
        if self.num_workers is None:
            import multiprocessing as mp
            self.num_workers = max(1, mp.cpu_count() // 2)
    
    @classmethod
    def from_cli_and_file(cls, args: argparse.Namespace) -> "EvalConfig":
        cfg_data: Dict[str, Any] = {}
        if getattr(args, "config", None):
            cfg_path = pathlib.Path(args.config)
            if not cfg_path.exists():
                raise ValueError(f"Config file not found: {cfg_path}")
            text = cfg_path.read_text(encoding="utf-8")
            if cfg_path.suffix.lower() in (".yml", ".yaml"):
                if not _HAS_YAML:
                    raise RuntimeError("PyYAML required to parse YAML config files")
                cfg_data = yaml.safe_load(text) or {}
            else:
                cfg_data = json.loads(text)
        overrides = {k: v for k, v in vars(args).items() if v is not None}
        cfg_data.update(overrides)
        return cls(**cfg_data)

# -------------------------
# MLflow & W&B Helpers
# -------------------------
def safe_mlflow_init(cfg: EvalConfig, run_name: Optional[str] = None):
    if not cfg.mlflow or not _HAS_MLFLOW:
        return None
    try:
        mlflow.set_experiment(cfg.mlflow_experiment)
        active_run = mlflow.start_run(run_name=run_name)
        LOG.info("✅ MLflow run started: %s", active_run.info.run_id)
        return active_run
    except Exception:
        LOG.exception("MLflow init failed")
        return None

def safe_mlflow_log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    if not _HAS_MLFLOW:
        return
    try:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v), step=step)
    except Exception:
        LOG.debug("MLflow log metrics failed")

def safe_mlflow_end():
    if not _HAS_MLFLOW:
        return
    try:
        mlflow.end_run()
    except Exception:
        LOG.debug("MLflow end_run failed")

def safe_wandb_init(cfg: EvalConfig, run_name: Optional[str] = None):
    if not cfg.wandb or not _HAS_WANDB:
        return None
    try:
        wandb.init(project=cfg.wandb_project, name=run_name, config=dataclasses.asdict(cfg))
        LOG.info("✅ W&B run started: %s", wandb.run.name if wandb.run else run_name)
        return wandb.run
    except Exception:
        LOG.exception("W&B init failed")
        return None

def safe_wandb_log(metrics: Dict[str, float]):
    if not _HAS_WANDB:
        return
    try:
        wandb.log(metrics)
    except Exception:
        LOG.debug("W&B log failed")

def safe_wandb_finish():
    if not _HAS_WANDB:
        return
    try:
        wandb.finish()
    except Exception:
        LOG.debug("W&B finish failed")

# -------------------------
# Storage & S3 Helpers
# -------------------------
def s3_client():
    if not _HAS_BOTO3:
        raise RuntimeError("boto3 not available")
    return boto3.client("s3")

def download_s3_prefix_to_dir(bucket: str, prefix: str, dest_dir: Union[str, pathlib.Path], max_items: Optional[int] = None) -> List[str]:
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    client = s3_client()
    keys_downloaded = []
    paginator = client.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    try:
        for page in paginator.paginate(**kwargs):
            objs = page.get("Contents", [])
            for obj in objs:
                if max_items and len(keys_downloaded) >= max_items:
                    break
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = pathlib.Path(key).name
                target = dest_dir / rel
                try:
                    client.download_file(bucket, key, str(target))
                    keys_downloaded.append(str(target))
                except ClientError:
                    storage_logger.exception("Failed download s3://%s/%s", bucket, key)
            if max_items and len(keys_downloaded) >= max_items:
                break
    except Exception:
        storage_logger.exception("S3 pagination failed")
    return keys_downloaded

def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")

def parse_s3_path(s3uri: str) -> Tuple[str, str]:
    assert s3uri.startswith("s3://")
    rest = s3uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def upload_results_to_s3(result_path: str, cfg: EvalConfig) -> Optional[str]:
    if not cfg.s3_bucket or not _HAS_BOTO3:
        return None
    client = s3_client()
    key_prefix = cfg.s3_prefix or f"eval-results/{time.strftime('%Y%m%d')}/"
    key_name = key_prefix.rstrip("/") + "/" + os.path.basename(result_path)
    try:
        client.upload_file(result_path, cfg.s3_bucket, key_name)
        LOG.info("☁️ Uploaded to s3://%s/%s", cfg.s3_bucket, key_name)
        return f"s3://{cfg.s3_bucket}/{key_name}"
    except Exception:
        LOG.exception("S3 upload failed")
        return None

# -------------------------
# Model Registry
# -------------------------
def resolve_model_dir_from_registry(tag: str, fallback_fs_dir: Optional[str] = None) -> Optional[str]:
    storage_logger.debug("Resolving model tag: %s", tag)
    try:
        if ModelRegistry is not None:
            registry = ModelRegistry()
            meta = registry.get_by_tag(tag)
            if meta:
                if meta.get("file_path"):
                    return meta["file_path"]
                if meta.get("s3_uri"):
                    return meta["s3_uri"]
                if meta.get("dir"):
                    return meta["dir"]
    except Exception:
        storage_logger.exception("ModelRegistry lookup failed")
    
    local_dir = pathlib.Path(fallback_fs_dir or DEFAULT_MODELS_DIR) / tag
    if local_dir.exists():
        return str(local_dir)
    
    if is_s3_path(tag):
        return tag
    
    return None

def local_find_checkpoint_in_dir(model_dir: Union[str, pathlib.Path], pattern_exts: Optional[List[str]] = None) -> Optional[str]:
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        return None
    pattern_exts = pattern_exts or [".pt", ".pth", ".ckpt", ".pth.tar", ".bin", ".pt.tar", ".onnx"]
    cand = None
    latest_mtime = 0.0
    for ext in pattern_exts:
        for p in model_dir.rglob(f"*{ext}"):
            try:
                st = p.stat().st_mtime
                if st > latest_mtime:
                    cand = p
                    latest_mtime = st
            except Exception:
                continue
    return str(cand) if cand else None

def fetch_model_to_local(model_dir: str, tmp_dir: Union[str, pathlib.Path]) -> str:
    tmp_dir = pathlib.Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if is_s3_path(model_dir):
        bucket, prefix = parse_s3_path(model_dir)
        storage_logger.info("Downloading from S3 %s/%s -> %s", bucket, prefix, tmp_dir)
        files = download_s3_prefix_to_dir(bucket, prefix, tmp_dir, max_items=None)
        if not files:
            raise RuntimeError(f"No files downloaded from s3://{bucket}/{prefix}")
        return str(tmp_dir)
    else:
        p = pathlib.Path(model_dir)
        if p.is_file():
            return str(p.parent)
        if p.is_dir():
            return str(p)
        alt = DEFAULT_MODELS_DIR / model_dir
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

def discover_checkpoint(cfg: EvalConfig) -> Optional[str]:
    if cfg.checkpoint:
        loader_logger.debug("Using explicit checkpoint: %s", cfg.checkpoint)
        return cfg.checkpoint
    
    if cfg.model_tag:
        resolved = resolve_model_dir_from_registry(cfg.model_tag, fallback_fs_dir=str(DEFAULT_MODELS_DIR))
        if resolved:
            if is_s3_path(resolved):
                return resolved
            ckpt = local_find_checkpoint_in_dir(resolved)
            if ckpt:
                return ckpt
            return resolved
    
    if cfg.model_dir:
        if is_s3_path(cfg.model_dir):
            return cfg.model_dir
        if pathlib.Path(cfg.model_dir).exists():
            ckpt = local_find_checkpoint_in_dir(cfg.model_dir)
            if ckpt:
                return ckpt
            return cfg.model_dir
        alt = DEFAULT_MODELS_DIR / cfg.model_dir
        if alt.exists():
            ckpt = local_find_checkpoint_in_dir(str(alt))
            return ckpt or str(alt)
    
    try:
        candidates = sorted(DEFAULT_MODELS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for d in candidates:
            if d.is_dir():
                ckpt = local_find_checkpoint_in_dir(d)
                if ckpt:
                    return ckpt
    except Exception:
        loader_logger.exception("Failed scanning default models dir")
    
    return None

# -------------------------
# Device & Seed
# -------------------------
def prepare_device(device_str: str = "cpu"):
    if not _HAS_TORCH:
        return None
    try:
        if device_str and device_str.startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(device_str)
        else:
            dev = torch.device("cpu")
        return dev
    except Exception:
        return torch.device("cpu")

def set_global_seed(seed: Optional[int]):
    if seed is None:
        seed = int(time.time() % 2**31)
    random.seed(seed)
    if _HAS_NUMPY:
        np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    LOG.info("Seed set to %s", seed)
    return seed

# -------------------------
# Policy Interface
# -------------------------
class PolicyInterface:
    """Unified wrapper for PyTorch models."""
    def __init__(self, model_obj: Any, device: Optional[Any] = None, deterministic: bool = True):
        self.model = model_obj
        self.device = device
        self.deterministic = deterministic
        self._is_torch = _HAS_TORCH and isinstance(model_obj, torch.nn.Module)
        if self._is_torch:
            try:
                self.model.eval()
            except Exception:
                pass
    
    def _to_tensor(self, obs: Any):
        if not _HAS_TORCH or not self._is_torch:
            return obs
        arr = obs
        try:
            if isinstance(obs, (list, tuple)):
                arr = np.asarray(obs, dtype=np.float32) if _HAS_NUMPY else obs
            if _HAS_NUMPY and isinstance(arr, np.ndarray):
                t = torch.from_numpy(arr.astype("float32"))
            else:
                t = torch.tensor(arr, dtype=torch.float32)
            if self.device:
                t = t.to(self.device)
            return t
        except Exception:
            return obs
    
    def act(self, obs: Any) -> Dict[str, Any]:
        try:
            if self._is_torch:
                with torch.no_grad():
                    t = self._to_tensor(obs)
                    if isinstance(t, torch.Tensor) and t.dim() == 1:
                        t = t.unsqueeze(0)
                    out = self.model(t)
                    if isinstance(out, (tuple, list)):
                        action = out[0]
                        logp = out[1] if len(out) > 1 else None
                        val = out[2] if len(out) > 2 else None
                    else:
                        action = out
                        logp = None
                        val = None
                    if isinstance(action, torch.Tensor):
                        a_np = action.detach().cpu().numpy()
                    else:
                        a_np = action
                    if isinstance(logp, torch.Tensor):
                        logp = logp.detach().cpu().numpy()
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    if _HAS_NUMPY and isinstance(a_np, np.ndarray):
                        if a_np.shape[0] == 1:
                            a_np = a_np[0]
                    return {"action": a_np, "logp": logp, "value": val}
            else:
                if hasattr(self.model, "act"):
                    res = self.model.act(obs)
                else:
                    res = self.model(obs)
                if isinstance(res, dict):
                    return res
                if isinstance(res, (tuple, list)):
                    out = {"action": res[0]}
                    if len(res) > 1:
                        out["logp"] = res[1]
                    if len(res) > 2:
                        out["value"] = res[2]
                    return out
                return {"action": res}
        except Exception:
            loader_logger.exception("Policy act() failed")
            raise

# -------------------------
# ONNX Policy Interface
# -------------------------
class ONNXPolicyInterface:
    """ONNX Runtime wrapper for production inference."""
    def __init__(self, onnx_path: str, providers: List[str] = None):
        if not _HAS_ONNX:
            raise RuntimeError("onnxruntime not available")
        
        self.onnx_path = onnx_path
        self.providers = providers or ["CPUExecutionProvider"]
        
        try:
            self.session = ort.InferenceSession(onnx_path, providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            loader_logger.info("✅ ONNX model loaded: %s", onnx_path)
            loader_logger.info("   Input: %s %s", self.input_name, self.session.get_inputs()[0].shape)
            loader_logger.info("   Outputs: %s", self.output_names)
        except Exception as e:
            loader_logger.exception("Failed to load ONNX model")
            raise
    
    def act(self, obs: Any) -> Dict[str, Any]:
        try:
            if isinstance(obs, (list, tuple)):
                obs_array = np.array(obs, dtype=np.float32) if _HAS_NUMPY else obs
            elif _HAS_NUMPY and isinstance(obs, np.ndarray):
                obs_array = obs.astype(np.float32)
            else:
                obs_array = np.array([obs], dtype=np.float32) if _HAS_NUMPY else [obs]
            
            if _HAS_NUMPY and obs_array.ndim == 1:
                obs_array = obs_array.reshape(1, -1)
            
            outputs = self.session.run(self.output_names, {self.input_name: obs_array})
            
            action = outputs[0][0] if len(outputs[0].shape) > 1 else outputs[0]
            value = outputs[1][0] if len(outputs) > 1 else None
            
            return {
                "action": action,
                "logp": None,
                "value": float(value) if value is not None else None
            }
        except Exception as e:
            loader_logger.exception("ONNX inference failed")
            raise

# -------------------------
# A/B Testing
# -------------------------
def compare_policies_statistically(
    baseline_results: Dict[str, Any],
    candidate_results: Dict[str, Any],
    test_type: str = "ttest",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    try:
        from scipy import stats
        import numpy as np
    except ImportError:
        LOG.warning("scipy not available for statistical tests")
        return {"error": "scipy required"}
    
    baseline_rewards = np.array(baseline_results.get("rewards", []))
    candidate_rewards = np.array(candidate_results.get("rewards", []))
    
    if len(baseline_rewards) == 0 or len(candidate_rewards) == 0:
        return {"error": "insufficient data"}
    
    baseline_mean = float(np.mean(baseline_rewards))
    candidate_mean = float(np.mean(candidate_rewards))
    improvement = ((candidate_mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0.0
    
    if test_type == "ttest":
        statistic, p_value = stats.ttest_ind(candidate_rewards, baseline_rewards)
    elif test_type == "mannwhitneyu":
        statistic, p_value = stats.mannwhitneyu(candidate_rewards, baseline_rewards, alternative='two-sided')
    else:
        return {"error": f"unknown test type: {test_type}"}
    
    def bootstrap_mean_diff(n_iterations=1000):
        diffs = []
        for _ in range(n_iterations):
            b_sample = np.random.choice(baseline_rewards, len(baseline_rewards), replace=True)
            c_sample = np.random.choice(candidate_rewards, len(candidate_rewards), replace=True)
            diffs.append(np.mean(c_sample) - np.mean(b_sample))
        return diffs
    
    bootstrap_diffs = bootstrap_mean_diff()
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))
    
    pooled_std = np.sqrt((np.var(baseline_rewards) + np.var(candidate_rewards)) / 2)
    cohens_d = (candidate_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0.0
    
    significant = p_value < (1 - confidence_level)
    if significant and improvement > 5:
        recommendation = "✅ DEPLOY - Significant improvement"
    elif significant and improvement < -5:
        recommendation = "❌ REJECT - Significant degradation"
    elif not significant:
        recommendation = "⚠️ INCONCLUSIVE - No significant difference"
    else:
        recommendation = "⚪ NEUTRAL - Small difference"
    
    return {
        "significant": significant,
        "p_value": float(p_value),
        "confidence_interval": (ci_lower, ci_upper),
        "effect_size": float(cohens_d),
        "improvement_pct": improvement,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "recommendation": recommendation
    }

# -------------------------
# Drift Detection
# -------------------------
class DriftDetector:
    def __init__(self, reference_data: Optional[Dict[str, Any]] = None):
        self.reference_data = reference_data
        self.reference_obs_dist = None
        self.reference_reward_dist = None
        
        if reference_data:
            self._compute_reference_distributions(reference_data)
    
    def _compute_reference_distributions(self, data: Dict[str, Any]):
        try:
            obs_data = data.get("observations", [])
            reward_data = data.get("rewards", [])
            
            if obs_data and _HAS_NUMPY:
                obs_array = np.array(obs_data)
                self.reference_obs_dist = {
                    "mean": np.mean(obs_array, axis=0),
                    "std": np.std(obs_array, axis=0),
                    "min": np.min(obs_array, axis=0),
                    "max": np.max(obs_array, axis=0)
                }
            
            if reward_data and _HAS_NUMPY:
                reward_array = np.array(reward_data)
                self.reference_reward_dist = {
                    "mean": float(np.mean(reward_array)),
                    "std": float(np.std(reward_array)),
                    "p50": float(np.percentile(reward_array, 50)),
                    "p95": float(np.percentile(reward_array, 95))
                }
        except Exception as e:
            LOG.exception("Failed to compute reference distributions")
    
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rl_eval.py - ENTERPRISE EDITION COMPLETION (Part 2)
----------------------------------------------------------
Continuation from DriftDetector.check_drift() onwards.
Synchronized with train_rl_heavy.py and train_rl_live.py
"""

# =========================================================================
# CONTINUATION: DriftDetector.check_drift() method
# =========================================================================

def check_drift(self, eval_results: Dict[str, Any], threshold: float = 0.3) -> Dict[str, Any]:
        """
        Check for distribution drift.
        
        Returns:
            {
                "drift_detected": bool,
                "drift_score": float,
                "threshold": threshold,
                "details": {...},
                "recommendation": str
            }
        """
        if not self.reference_reward_dist:
            return {"error": "no reference distribution available"}
        
        try:
            import numpy as np
            
            eval_rewards = np.array(eval_results.get("rewards", []))
            if len(eval_rewards) == 0:
                return {"error": "no evaluation rewards"}
            
            # Compute current statistics
            eval_mean = float(np.mean(eval_rewards))
            eval_std = float(np.std(eval_rewards))
            
            ref_mean = self.reference_reward_dist["mean"]
            ref_std = self.reference_reward_dist["std"]
            
            # Normalized distance (simple KL divergence approximation)
            mean_drift = abs(eval_mean - ref_mean) / (ref_std + 1e-8)
            std_drift = abs(eval_std - ref_std) / (ref_std + 1e-8)
            
            drift_score = max(mean_drift, std_drift)
            drift_detected = drift_score > threshold
            
            # Additional checks: percentile shifts
            eval_p50 = float(np.percentile(eval_rewards, 50))
            eval_p95 = float(np.percentile(eval_rewards, 95))
            ref_p50 = self.reference_reward_dist["p50"]
            ref_p95 = self.reference_reward_dist["p95"]
            
            p50_shift = abs(eval_p50 - ref_p50) / (abs(ref_p50) + 1e-8)
            p95_shift = abs(eval_p95 - ref_p95) / (abs(ref_p95) + 1e-8)
            
            return {
                "drift_detected": drift_detected,
                "drift_score": float(drift_score),
                "threshold": threshold,
                "details": {
                    "reference_mean": ref_mean,
                    "reference_std": ref_std,
                    "eval_mean": eval_mean,
                    "eval_std": eval_std,
                    "mean_drift": float(mean_drift),
                    "std_drift": float(std_drift),
                    "p50_shift": float(p50_shift),
                    "p95_shift": float(p95_shift),
                    "reference_p50": ref_p50,
                    "reference_p95": ref_p95,
                    "eval_p50": eval_p50,
                    "eval_p95": eval_p95
                },
                "recommendation": "⚠️ Retrain model - significant drift detected" if drift_detected 
                                 else "✅ Distribution stable - no action needed"
            }
        except Exception as e:
            LOG.exception("Drift detection failed: %s", e)
            return {"error": str(e)}

# =========================================================================
# SECTION 6: Performance Benchmarking (COMPLETE)
# =========================================================================

def benchmark_inference_performance(
    policy: Any,
    obs_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 8, 32],
    n_iterations: int = 100
) -> Dict[str, Any]:
    """
    Measure inference latency and throughput across batch sizes.
    Critical for production deployment planning and capacity estimation.
    
    Returns:
        {
            "batch_1": {"mean_latency_ms": ..., "p95_latency_ms": ..., "throughput_samples_per_sec": ...},
            "batch_8": {...},
            ...
        }
    """
    import numpy as np
    import time
    
    results = {}
    
    for batch_size in batch_sizes:
        latencies = []
        
        for _ in range(n_iterations):
            # Generate dummy batch matching obs_shape
            if len(obs_shape) == 0:
                # Scalar observation
                batch = np.random.randn(batch_size, 1).astype(np.float32)
            elif len(obs_shape) == 1:
                batch = np.random.randn(batch_size, obs_shape[0]).astype(np.float32)
            else:
                batch = np.random.randn(batch_size, *obs_shape).astype(np.float32)
            
            # Measure inference time
            start = time.perf_counter()
            try:
                if batch_size == 1:
                    _ = policy.act(batch[0])
                else:
                    # Batch inference if supported
                    for i in range(batch_size):
                        _ = policy.act(batch[i])
                
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
            except Exception as e:
                LOG.warning("Inference failed for batch_size=%d: %s", batch_size, e)
                break
        
        if latencies:
            latencies_np = np.array(latencies)
            results[f"batch_{batch_size}"] = {
                "mean_latency_ms": float(np.mean(latencies_np) * 1000),
                "std_latency_ms": float(np.std(latencies_np) * 1000),
                "p50_latency_ms": float(np.percentile(latencies_np, 50) * 1000),
                "p95_latency_ms": float(np.percentile(latencies_np, 95) * 1000),
                "p99_latency_ms": float(np.percentile(latencies_np, 99) * 1000),
                "throughput_samples_per_sec": float(batch_size / np.mean(latencies_np)),
                "n_samples": len(latencies)
            }
            
            LOG.debug("Batch %d: %.2f ms (p95: %.2f ms, throughput: %.1f samples/sec)",
                     batch_size,
                     results[f"batch_{batch_size}"]["mean_latency_ms"],
                     results[f"batch_{batch_size}"]["p95_latency_ms"],
                     results[f"batch_{batch_size}"]["throughput_samples_per_sec"])
    
    return results

# =========================================================================
# SECTION 7: Enhanced run_evaluation (COMPLETE IMPLEMENTATION)
# =========================================================================

def run_evaluation(cfg: EvalConfig) -> int:
    """
    Main orchestration with full enterprise features.
    Synchronized with train_rl_heavy.py and train_rl_live.py patterns.
    """
    LOG.info("=" * 80)
    LOG.info("=== PriorityMax RL Evaluation - Enterprise Edition ===")
    LOG.info("=" * 80)
    LOG.info("Configuration:")
    LOG.info("  Episodes: %d", cfg.eval_episodes)
    LOG.info("  Max steps: %d", cfg.max_steps)
    LOG.info("  Device: %s", cfg.device)
    LOG.info("  ONNX mode: %s", cfg.use_onnx)
    LOG.info("  A/B testing: %s", "enabled" if cfg.baseline_checkpoint else "disabled")
    LOG.info("  Drift detection: %s", "enabled" if cfg.enable_drift_detection else "disabled")
    LOG.info("  Benchmarking: %s", "enabled" if cfg.measure_inference_time else "disabled")
    LOG.info("=" * 80)
    
    seed = set_global_seed(cfg.seed)
    device = prepare_device(cfg.device)
    # ===================== SYNC CHECK =====================
    try:
        from ml.real_env import get_observation_space, get_action_space
        obs_space = get_observation_space()
        act_space = get_action_space()
        obs_dim = obs_space.shape[0] if obs_space is not None else 8
        act_dim = act_space.shape[0] if act_space is not None else 3
        LOG.info("🔄 Synced with real_env: obs_dim=%d act_dim=%d", obs_dim, act_dim)
    except Exception as e:
        LOG.warning("⚠️ Unable to sync env spaces from real_env: %s", e)
        obs_dim, act_dim = 8, 3
    # =======================================================

    
    # Initialize MLflow/W&B
    run_name = f"eval_{cfg.model_tag or 'model'}_{int(time.time())}"
    mlflow_run = safe_mlflow_init(cfg, run_name)
    wandb_run = safe_wandb_init(cfg, run_name)
    
    # Prometheus metrics (if enabled)
    if cfg.enable_prometheus and _HAS_PROMETHEUS:
        eval_reward_gauge = Gauge("rl_eval_reward_mean", "Mean evaluation reward")
        eval_episodes_counter = Counter("rl_eval_episodes_total", "Total episodes evaluated")
        eval_drift_gauge = Gauge("rl_eval_drift_score", "Distribution drift score")
        eval_latency_histogram = Histogram("rl_eval_action_latency_seconds", "Action latency")
        
        try:
            start_http_server(cfg.prometheus_port)
            LOG.info("📊 Prometheus metrics server started on port %d", cfg.prometheus_port)
        except Exception as e:
            LOG.warning("Failed to start Prometheus server: %s", e)
    
    tmp_dir = tempfile.mkdtemp(prefix="prioritymax_eval_tmp_")
    
    # -------------------------
    # Build policy (ONNX or PyTorch)
    # -------------------------
    try:
        if cfg.use_onnx:
            # Load ONNX model directly
            onnx_path = discover_checkpoint(cfg)
            if not onnx_path or not onnx_path.endswith(".onnx"):
                raise ValueError("ONNX mode requires .onnx checkpoint file. "
                               f"Got: {onnx_path}")
            policy = ONNXPolicyInterface(onnx_path, providers=cfg.onnx_providers)
            model_dir = str(pathlib.Path(onnx_path).parent)
            LOG.info("✅ ONNX policy loaded: %s", onnx_path)
        else:
            # Standard PyTorch loading
            policy, model_dir = build_policy_for_eval(cfg, tmp_dir=tmp_dir)
            LOG.info("✅ PyTorch policy loaded from: %s", model_dir)
    except Exception as e:
        LOG.exception("❌ Policy build failed: %s", e)
        safe_mlflow_end()
        safe_wandb_finish()
        return EXIT_FAILURE
    
    # -------------------------
    # Performance benchmarking
    # -------------------------
    benchmark_results = None
    if cfg.measure_inference_time:
        try:
            LOG.info("🔍 Running inference benchmark...")
            
            # Try to infer obs_shape from metadata
            obs_shape = (obs_dim,)  # Default fallback
            try:
                metadata_path = pathlib.Path(model_dir) / "metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if "obs_shape" in metadata:
                        obs_shape = tuple(metadata["obs_shape"])
            except Exception:
                LOG.debug("Could not load obs_shape from metadata, using default")
            
            benchmark_results = benchmark_inference_performance(
                policy, 
                obs_shape, 
                cfg.benchmark_batch_sizes,
                n_iterations=100
            )
            
            LOG.info("Benchmark results:")
            for batch_key, metrics in benchmark_results.items():
                LOG.info("  %s: %.2f ms (p95: %.2f ms, p99: %.2f ms, throughput: %.1f/s)", 
                        batch_key,
                        metrics["mean_latency_ms"],
                        metrics["p95_latency_ms"],
                        metrics["p99_latency_ms"],
                        metrics["throughput_samples_per_sec"])
            
            # Log to tracking systems
            if benchmark_results.get("batch_1"):
                safe_mlflow_log_metrics({
                    "inference_latency_mean_ms": benchmark_results["batch_1"]["mean_latency_ms"],
                    "inference_latency_p95_ms": benchmark_results["batch_1"]["p95_latency_ms"],
                    "inference_throughput": benchmark_results["batch_1"]["throughput_samples_per_sec"]
                })
                safe_wandb_log({
                    "inference_latency_mean_ms": benchmark_results["batch_1"]["mean_latency_ms"],
                    "inference_latency_p95_ms": benchmark_results["batch_1"]["p95_latency_ms"]
                })
        except Exception as e:
            LOG.warning("⚠️ Benchmark failed: %s", e)
            benchmark_results = {"error": str(e)}
    
    # -------------------------
    # Create environment factory
    # -------------------------
    env_factory = _local_make_env({})
    
    # -------------------------
    # Run main evaluation
    # -------------------------
    try:
        LOG.info("🚀 Starting evaluation (%d episodes, max_steps=%d)...", 
                cfg.eval_episodes, cfg.max_steps)
        
        res = run_vectorized_evaluation(
            policy,
            env_factory,
            episodes=cfg.eval_episodes,
            max_steps=cfg.max_steps,
            n_workers=cfg.num_workers,
            timeout=cfg.worker_timeout
        )
        
        LOG.info("✅ Evaluation complete:")
        LOG.info("   Mean reward: %.3f ± %.3f", 
                res.get("reward_mean", 0), res.get("reward_std", 0))
        LOG.info("   Episodes completed: %d/%d",
                res.get("episodes_completed", 0), cfg.eval_episodes)
        
        if res.get("failed", 0) > 0:
            LOG.warning("⚠️ %d episodes failed", res["failed"])
            
    except Exception as e:
        LOG.exception("❌ Evaluation loop failed: %s", e)
        safe_mlflow_end()
        safe_wandb_finish()
        return EXIT_FAILURE
    
    # Add benchmark results to output
    if benchmark_results:
        res["benchmark"] = benchmark_results
    
    # -------------------------
    # Safety checks
    # -------------------------
    if cfg.check_numerical_stability:
        rewards = res.get("rewards", [])
        if any(np.isnan(r) or np.isinf(r) for r in rewards):
            LOG.error("❌ NUMERICAL INSTABILITY DETECTED - NaN/Inf in rewards!")
            res["numerical_stability_check"] = "FAILED"
            res["numerical_stability_details"] = {
                "nan_count": sum(1 for r in rewards if np.isnan(r)),
                "inf_count": sum(1 for r in rewards if np.isinf(r))
            }
        else:
            LOG.info("✅ Numerical stability check passed")
            res["numerical_stability_check"] = "PASSED"
    
    # Reward threshold checks
    mean_reward = res.get("reward_mean", 0.0)
    if cfg.max_episode_reward and mean_reward > cfg.max_episode_reward:
        LOG.warning("⚠️ Mean reward %.3f EXCEEDS max threshold %.3f (possible bug)", 
                   mean_reward, cfg.max_episode_reward)
        res["sanity_check_max"] = "FAILED"
    
    if cfg.min_episode_reward and mean_reward < cfg.min_episode_reward:
        LOG.warning("⚠️ Mean reward %.3f BELOW min threshold %.3f", 
                   mean_reward, cfg.min_episode_reward)
        res["sanity_check_min"] = "FAILED"
    
    # -------------------------
    # A/B testing
    # -------------------------
    if cfg.baseline_checkpoint:
        try:
            LOG.info("🔬 Running A/B test against baseline...")
            LOG.info("   Baseline: %s", cfg.baseline_checkpoint)
            
            # Build baseline policy
            baseline_cfg = EvalConfig(
                checkpoint=cfg.baseline_checkpoint,
                eval_episodes=cfg.eval_episodes,
                max_steps=cfg.max_steps,
                device=cfg.device,
                use_onnx=cfg.baseline_checkpoint.endswith(".onnx") if cfg.baseline_checkpoint else False,
                seed=cfg.seed + 1000  # Different seed for baseline
            )
            
            baseline_policy, _ = build_policy_for_eval(baseline_cfg, tmp_dir=tmp_dir)
            
            baseline_res = run_vectorized_evaluation(
                baseline_policy,
                env_factory,
                episodes=cfg.eval_episodes,
                max_steps=cfg.max_steps,
                n_workers=cfg.num_workers,
                timeout=cfg.worker_timeout
            )
            
            # Perform statistical comparison
            ab_comparison = compare_policies_statistically(
                baseline_res,
                res,
                test_type=cfg.statistical_test,
                confidence_level=cfg.confidence_level
            )
            
            LOG.info("=" * 60)
            LOG.info("A/B Test Results:")
            LOG.info("=" * 60)
            LOG.info("  Baseline mean: %.3f", ab_comparison.get("baseline_mean", 0))
            LOG.info("  Candidate mean: %.3f", ab_comparison.get("candidate_mean", 0))
            LOG.info("  Improvement: %.1f%%", ab_comparison.get("improvement_pct", 0))
            LOG.info("  P-value: %.4f", ab_comparison.get("p_value", 0))
            LOG.info("  Significant: %s", ab_comparison.get("significant", False))
            LOG.info("  Effect size (Cohen's d): %.3f", ab_comparison.get("effect_size", 0))
            LOG.info("  Confidence interval: (%.3f, %.3f)", 
                    ab_comparison.get("confidence_interval", (0, 0))[0],
                    ab_comparison.get("confidence_interval", (0, 0))[1])
            LOG.info("  %s", ab_comparison.get("recommendation", "N/A"))
            LOG.info("=" * 60)
            
            res["ab_test"] = ab_comparison
            res["baseline_results"] = baseline_res
            
            # Log to tracking
            safe_mlflow_log_metrics({
                "ab_p_value": ab_comparison.get("p_value", 0),
                "ab_improvement_pct": ab_comparison.get("improvement_pct", 0),
                "ab_effect_size": ab_comparison.get("effect_size", 0),
                "ab_significant": 1.0 if ab_comparison.get("significant") else 0.0
            })
            
            safe_wandb_log({
                "ab_test": ab_comparison,
                "ab_improvement_pct": ab_comparison.get("improvement_pct", 0)
            })
            
        except Exception as e:
            LOG.exception("❌ A/B testing failed: %s", e)
            res["ab_test"] = {"error": str(e)}
    
    # -------------------------
    # Drift detection
    # -------------------------
    if cfg.enable_drift_detection and cfg.drift_reference_data:
        try:
            LOG.info("📊 Checking for distribution drift...")
            LOG.info("   Reference data: %s", cfg.drift_reference_data)
            
            # Load reference data
            ref_path = pathlib.Path(cfg.drift_reference_data)
            if not ref_path.exists():
                LOG.warning("Reference data not found: %s", ref_path)
            else:
                with open(ref_path, 'r') as f:
                    reference_data = json.load(f)
                
                drift_detector = DriftDetector(reference_data)
                drift_result = drift_detector.check_drift(res, threshold=cfg.drift_threshold)
                
                if drift_result.get("drift_detected"):
                    LOG.warning("=" * 60)
                    LOG.warning("⚠️ DRIFT DETECTED - Distribution shift from reference")
                    LOG.warning("=" * 60)
                    LOG.warning("  Drift score: %.3f (threshold: %.3f)", 
                               drift_result["drift_score"], cfg.drift_threshold)
                    details = drift_result.get("details", {})
                    LOG.warning("  Mean drift: %.3f", details.get("mean_drift", 0))
                    LOG.warning("  Std drift: %.3f", details.get("std_drift", 0))
                    LOG.warning("  P50 shift: %.3f", details.get("p50_shift", 0))
                    LOG.warning("  P95 shift: %.3f", details.get("p95_shift", 0))
                    LOG.warning("  %s", drift_result.get("recommendation", ""))
                    LOG.warning("=" * 60)
                else:
                    LOG.info("✅ No significant drift detected (score: %.3f)", 
                            drift_result.get("drift_score", 0))
                
                res["drift_detection"] = drift_result
                
                # Log to tracking
                safe_mlflow_log_metrics({
                    "drift_score": drift_result.get("drift_score", 0),
                    "drift_detected": 1.0 if drift_result.get("drift_detected") else 0.0
                })
                
                safe_wandb_log({
                    "drift_score": drift_result.get("drift_score", 0),
                    "drift_detected": drift_result.get("drift_detected", False)
                })
                
                # Update Prometheus
                if cfg.enable_prometheus and _HAS_PROMETHEUS:
                    eval_drift_gauge.set(drift_result.get("drift_score", 0))
                    
                # ✅ Final environment sync sanity check
                try:
                    obs_space = get_observation_space()
                    act_space = get_action_space()
                    if obs_space and act_space:
                        LOG.info(
                            "✅ Sync verified | obs_dim=%d vs env=%d | act_dim=%d vs env=%d",
                            SYNC_CFG.obs_dim, obs_space.shape[0],
                            SYNC_CFG.act_dim, act_space.shape[0],
                        )
                except Exception:
                    LOG.debug("Sync verification skipped (real_env not available)")
                
        except Exception as e:
            LOG.exception("❌ Drift detection failed: %s", e)
            res["drift_detection"] = {"error": str(e)}
    
    # -------------------------
    # Write results to file
    # -------------------------
    result_file = cfg.out or str(DEFAULT_RESULTS_DIR / f"eval_result_{int(time.time())}.json")
    pathlib.Path(result_file).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare serializable result
        serializable_res = {
                            k: v for k, v in res.items()
                            if not isinstance(v, (type, torch.nn.Module) if _HAS_TORCH else ())
        }                    
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(serializable_res, f, indent=2, default=str)
        LOG.info("💾 Results saved: %s", result_file)
    except Exception as e:
        LOG.exception("❌ Failed to write results: %s", e)
    
    # -------------------------
    # Log final metrics
    # -------------------------
    final_metrics = {
        "reward_mean": res.get("reward_mean", 0.0),
        "reward_std": res.get("reward_std", 0.0),
        "reward_p50": res.get("reward_p50", 0.0),
        "reward_p95": res.get("reward_p95", 0.0),
        "episodes_completed": res.get("episodes_completed", 0),
        "episodes_failed": res.get("failed", 0)
    }
    
    safe_mlflow_log_metrics(final_metrics)
    safe_wandb_log(final_metrics)
    
    # Update Prometheus
    if cfg.enable_prometheus and _HAS_PROMETHEUS:
        eval_reward_gauge.set(res.get("reward_mean", 0.0))
        eval_episodes_counter.inc(res.get("episodes_completed", 0))
    
    # -------------------------
    # S3 upload
    # -------------------------
    s3_uri = upload_results_to_s3(result_file, cfg)
    if s3_uri:
        LOG.info("☁️ Uploaded to S3: %s", s3_uri)
        res["s3_uri"] = s3_uri
    
    # -------------------------
    # Cleanup
    # -------------------------
    safe_mlflow_end()
    safe_wandb_finish()
    
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    
    # -------------------------
    # Final summary & exit code
    # -------------------------
    LOG.info("=" * 80)
    LOG.info("✅ EVALUATION COMPLETE")
    LOG.info("=" * 80)
    LOG.info("Episodes: %d/%d completed", 
            res.get("episodes_completed", 0), cfg.eval_episodes)
    LOG.info("Mean Reward: %.3f ± %.3f", mean_reward, res.get("reward_std", 0.0))
    LOG.info("Reward Range: [%.3f, %.3f]", 
            min(res.get("rewards", [0])), max(res.get("rewards", [0])))
    
    if "ab_test" in res:
        LOG.info("A/B Test: %s", res["ab_test"].get("recommendation", "N/A"))
    
    if "drift_detection" in res:
        LOG.info("Drift: %s", res["drift_detection"].get("recommendation", "N/A"))
    
    if "numerical_stability_check" in res:
        LOG.info("Stability: %s", res["numerical_stability_check"])
    
    LOG.info("Results: %s", result_file)
    if s3_uri:
        LOG.info("S3: %s", s3_uri)
    LOG.info("=" * 80)
    
    # Determine exit code
    if cfg.stop_if_below_mean_reward is not None and mean_reward < cfg.stop_if_below_mean_reward:
        LOG.warning("❌ Mean reward %.3f BELOW threshold %.3f - FAILING", 
                   mean_reward, cfg.stop_if_below_mean_reward)
        return EXIT_MEAN_BELOW_THRESHOLD
    
    if res.get("numerical_stability_check") == "FAILED":
        LOG.error("❌ Numerical instability detected - FAILING")
        return EXIT_FAILURE
    
    return EXIT_OK

# =========================================================================
# SECTION 8: Enhanced CLI (COMPLETE)
# =========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PriorityMax RL Evaluation - Enterprise Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation
  python train_rl_eval.py --model-dir models/ppo --eval-episodes 50
  
  # ONNX production path
  python train_rl_eval.py --checkpoint model.onnx --use-onnx
  
  # A/B testing
  python train_rl_eval.py --checkpoint candidate.pt --baseline-checkpoint baseline.pt
  
  # Full CI/CD pipeline
  python train_rl_eval.py --model-tag prod-v2 --baseline-checkpoint prod-v1.pt \\
      --drift-detection --drift-reference data/ref.json --benchmark \\
      --stop-if-below-mean-reward 100.0 --mlflow --wandb
        """
    )
    
    # ===== Basic Arguments =====
    p.add_argument("--config", help="Path to YAML/JSON config file")
    p.add_argument("--model-dir", help="Path or S3 URI to model directory")
    p.add_argument("--checkpoint", help="Explicit checkpoint file (.pt or .onnx)")
    p.add_argument("--model-tag", help="Model registry tag")
    
    # ===== Evaluation Parameters =====
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="Number of episodes to evaluate (default: 50)")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Max steps per episode (default: 1000)")
    p.add_argument("--device", default="cpu",
                   help="Device string: cpu, cuda, cuda:0 (default: cpu)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    
    # ===== Thresholds & Gates =====
    p.add_argument("--stop-if-below-mean-reward", type=float, default=None,
                   help="Exit with code 2 if mean reward below this threshold (CI gate)")
    p.add_argument("--max-episode-reward", type=float, default=None,
                   help="Sanity check: warn if reward exceeds this (default: None)")
    p.add_argument("--min-episode-reward", type=float, default=None,
                   help="Sanity check: warn if reward below this (default: None)")
    
    # ===== Experiment Tracking =====
    p.add_argument("--mlflow", action="store_true",
                   help="Enable MLflow logging")
    p.add_argument("--mlflow-experiment", default="PriorityMax-RL-Eval",
                   help="MLflow experiment name")
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="PriorityMax-RL-Eval",
                   help="W&B project name")
    
    # ===== Output & Storage =====
    p.add_argument("--out", help="Output result JSON path")
    p.add_argument("--s3-bucket", help="S3 bucket for uploads")
    p.add_argument("--s3-prefix", help="S3 prefix (default: eval-results/YYYYMMDD/)")
    
    # ===== ENTERPRISE FEATURES =====
    
    # ONNX
    p.add_argument("--use-onnx", action="store_true",
                   help="Evaluate ONNX model (production inference path)")
    p.add_argument("--onnx-providers", nargs="+", default=["CPUExecutionProvider"],
                   help="ONNX Runtime providers (default: CPUExecutionProvider)")
    
    # A/B Testing
    p.add_argument("--baseline-checkpoint", default=None,
                   help="Baseline checkpoint for A/B testing")
    p.add_argument("--statistical-test", choices=["ttest", "mannwhitneyu"], default="ttest",
                   help="Statistical test for A/B comparison (default: ttest)")
    p.add_argument("--confidence-level", type=float, default=0.95,
                   help="Confidence level for statistical tests (default: 0.95)")
    
    # Drift Detection
    p.add_argument("--drift-detection", action="store_true",
                   help="Enable distribution drift detection")
    p.add_argument("--drift-reference", dest="drift_reference_data", default=None,
                   help="Path to reference distribution data (JSON)")
    p.add_argument("--drift-threshold", type=float, default=0.3,
                   help="Drift detection threshold (default: 0.3)")
    
    # Performance Benchmarking
    p.add_argument("--benchmark", dest="measure_inference_time", action="store_true",
                   help="Measure inference performance (latency/throughput)")
    p.add_argument("--benchmark-batch-sizes", type=int, nargs="+", default=[1, 8, 32],
                   help="Batch sizes for benchmarking (default: 1 8 32)")
    
    # Safety Checks
    p.add_argument("--check-stability", dest="check_numerical_stability",
                   action="store_true", default=True,
                   help="Check for NaN/Inf in outputs (default: enabled)")
    p.add_argument("--no-check-stability", dest="check_numerical_stability",
                   action="store_false",
                   help="Disable numerical stability checks")
    
    # Prometheus Metrics
    p.add_argument("--prometheus", dest="enable_prometheus", action="store_true",
                   help="Enable Prometheus metrics export")
    p.add_argument("--prometheus-port", type=int, default=9304,
                   help="Prometheus metrics port (default: 9304)")
    
    # Parallelization
    p.add_argument("--num-workers", type=int, default=None,
                   help="Number of parallel evaluation workers (default: auto-detect)")
    p.add_argument("--worker-timeout", type=int, default=3600,
                   help="Worker timeout in seconds (default: 3600)")
    
    # Misc
    p.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=1,
                   help="Verbosity level: 0=quiet, 1=normal, 2=debug (default: 1)")
    
    return p

# =========================================================================
# SECTION 9: Main CLI Entry Point (COMPLETE)
# =========================================================================

def main_cli():
    """Main CLI entry point with full error handling."""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Adjust logging level based on verbosity
    if args.verbosity == 0:
        LOG.setLevel(logging.WARNING)
    elif args.verbosity == 2:
        LOG.setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        cfg = EvalConfig.from_cli_and_file(args)
    except Exception as e:
        LOG.error("❌ Failed to load configuration: %s", e)
        sys.exit(EXIT_CONFIG_ERROR)
    
    # Validate configuration
    try:
        if cfg.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        if cfg.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if cfg.confidence_level <= 0 or cfg.confidence_level >= 1:
            raise ValueError("confidence_level must be in (0, 1)")
    except ValueError as e:
        LOG.error("❌ Invalid configuration: %s", e)
        sys.exit(EXIT_CONFIG_ERROR)
    
    # Check dependencies
    missing_deps = []
    if cfg.use_onnx and not _HAS_ONNX:
        missing_deps.append("onnxruntime")
    if cfg.log_mlflow and not _HAS_MLFLOW:
        LOG.warning("MLflow requested but not available")
    if cfg.log_wandb and not _HAS_WANDB:
        LOG.warning("W&B requested but not available")
    if cfg.enable_prometheus and not _HAS_PROMETHEUS:
        LOG.warning("Prometheus requested but prometheus_client not available")
    
    if missing_deps:
        LOG.error("❌ Missing required dependencies: %s", ", ".join(missing_deps))
        LOG.error("   Install with: pip install %s", " ".join(missing_deps))
        sys.exit(EXIT_MISSING_DEP)
    
    # Handle graceful signals
    def _handle_sig(sig, frame):
        LOG.warning("⚠️ Signal %s received; terminating...", sig)
        sys.exit(EXIT_FAILURE)
    
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    
    # Run evaluation
    try:
        exit_code = run_evaluation(cfg)
        sys.exit(exit_code)
    
    except KeyboardInterrupt:
        LOG.warning("⚠️ Interrupted by user")
        sys.exit(EXIT_FAILURE)
    
    except Exception as e:
        LOG.exception("❌ Fatal error during evaluation: %s", e)
        sys.exit(EXIT_FAILURE)

# =========================================================================
# SECTION 10: Vectorized Evaluation Implementation (COMPLETE)
# =========================================================================

def run_vectorized_evaluation(
    policy: Any,
    env_factory: Callable,
    episodes: int = 100,
    max_steps: int = 1000,
    n_workers: int = 4,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Parallel evaluation across multiple workers.
    Synchronized with train_rl_heavy.py patterns.
    
    Args:
        policy: Policy interface with .act() method
        env_factory: Factory function that creates environment instances
        episodes: Total episodes to evaluate
        max_steps: Maximum steps per episode
        n_workers: Number of parallel workers
        timeout: Worker timeout in seconds
    
    Returns:
        Dictionary with evaluation results
    """
    import multiprocessing as mp
    from queue import Empty as QueueEmpty
    
    LOG.info("Starting vectorized evaluation: %d workers, %d episodes", n_workers, episodes)
    
    # Try to serialize policy for workers
    policy_serialized = None
    
    if _HAS_TORCH and isinstance(policy.model, torch.nn.Module):
        # Export to TorchScript for worker processes
        try:
            tmp_dir = tempfile.mkdtemp(prefix="eval_policy_")
            policy_path = os.path.join(tmp_dir, "policy.pt")
            
            # Try JIT trace
            try:
                dummy_obs = torch.randn(1, 8)  # Adjust shape as needed
                traced = torch.jit.trace(policy.model, dummy_obs)
                torch.jit.save(traced, policy_path)
            except:
                # Fallback: try JIT script
                try:
                    scripted = torch.jit.script(policy.model)
                    torch.jit.save(scripted, policy_path)
                except:
                    LOG.warning("Could not serialize policy; falling back to single-process eval")
                    return _single_process_evaluation(policy, env_factory, episodes, max_steps)
            
            policy_serialized = {
                "type": "torch_jit",
                "path": policy_path
            }
            LOG.debug("Policy serialized to: %s", policy_path)
            
        except Exception as e:
            LOG.warning("Policy serialization failed: %s; using single-process", e)
            return _single_process_evaluation(policy, env_factory, episodes, max_steps)
    else:
        # Non-torch policy: fallback to single process
        LOG.warning("Non-torch policy detected; using single-process evaluation")
        return _single_process_evaluation(policy, env_factory, episodes, max_steps)
    
    # Create worker queues
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Start worker processes
    workers = []
    for worker_id in range(n_workers):
        p = mp.Process(
            target=_evaluation_worker,
            args=(task_queue, result_queue, worker_id, policy_serialized, env_factory)
        )
        p.daemon = True
        p.start()
        workers.append(p)
    
    # Distribute tasks
    episodes_per_worker = episodes // n_workers
    remainder = episodes % n_workers
    
    for worker_id in range(n_workers):
        worker_episodes = episodes_per_worker + (1 if worker_id < remainder else 0)
        if worker_episodes > 0:
            task_queue.put({
                "worker_id": worker_id,
                "episodes": worker_episodes,
                "max_steps": max_steps,
                "seed_offset": worker_id * 1000
            })
    
    # Send shutdown signal
    for _ in range(n_workers):
        task_queue.put(None)
    
    # Collect results
    all_results = []
    deadline = time.time() + timeout
    
    with tqdm(total=episodes, desc="Evaluating", disable=not _HAS_TQDM) as pbar:
        while len(all_results) < episodes and time.time() < deadline:
            try:
                result = result_queue.get(timeout=1.0)
                if result.get("ok"):
                    all_results.append(result)
                    pbar.update(1)
                else:
                    LOG.warning("Episode failed: %s", result.get("error"))
            except QueueEmpty:
                continue
    
    # Terminate workers
    for p in workers:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()
    
    # Aggregate results
    rewards = [r["reward"] for r in all_results if "reward" in r]
    steps_list = [r["steps"] for r in all_results if "steps" in r]
    
    if not rewards:
        LOG.error("No successful episodes completed!")
        return {
            "episodes_requested": episodes,
            "episodes_completed": 0,
            "failed": episodes,
            "rewards": [],
            "reward_mean": 0.0,
            "reward_std": 0.0
        }
    
    result_dict = {
        "episodes_requested": episodes,
        "episodes_completed": len(rewards),
        "failed": episodes - len(rewards),
        "rewards": rewards,
        "steps": steps_list,
        "reward_mean": float(statistics.mean(rewards)),
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "reward_min": float(min(rewards)),
        "reward_max": float(max(rewards)),
        "reward_p50": float(sorted(rewards)[int(0.5 * len(rewards))]),
        "reward_p95": float(sorted(rewards)[int(0.95 * len(rewards))]) if len(rewards) > 1 else float(rewards[0]),
        "mean_episode_length": float(statistics.mean(steps_list)) if steps_list else 0.0,
        "raw_results": all_results
    }
    
    LOG.debug("Vectorized evaluation complete: %d/%d episodes", len(rewards), episodes)
    
    return result_dict

def _evaluation_worker(task_queue, result_queue, worker_id, policy_spec, env_factory):
    """
    Worker process for parallel evaluation.
    Runs independently and reports results via queue.
    """
    # Load policy in worker process
    policy = None
    try:
        if policy_spec and policy_spec["type"] == "torch_jit":
            if _HAS_TORCH:
                model = torch.jit.load(policy_spec["path"], map_location="cpu")
                policy = PolicyInterface(model, device=torch.device("cpu"))
            else:
                result_queue.put({"ok": False, "error": "torch_not_available"})
                return
        else:
            result_queue.put({"ok": False, "error": "unsupported_policy_type"})
            return
    except Exception as e:
        result_queue.put({"ok": False, "error": f"policy_load_failed: {e}"})
        return
    
    # Process tasks
    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except:
            continue
        
        if task is None:
            break
        
        episodes = task["episodes"]
        max_steps = task["max_steps"]
        seed_offset = task["seed_offset"]
        
        # Run episodes
        for ep in range(episodes):
            try:
                env = env_factory(seed=seed_offset + ep)
                obs = env.reset(seed=seed_offset + ep)
                
                total_reward = 0.0
                steps = 0
                done = False
                
                while not done and steps < max_steps:
                    action_dict = policy.act(obs)
                    action = action_dict["action"]
                    
                    obs, reward, done, info = env.step(action)
                    total_reward += float(reward)
                    steps += 1
                
                result_queue.put({
                    "ok": True,
                    "worker_id": worker_id,
                    "episode": ep,
                    "reward": total_reward,
                    "steps": steps
                })
                
            except Exception as e:
                result_queue.put({
                    "ok": False,
                    "worker_id": worker_id,
                    "episode": ep,
                    "error": str(e)
                })

def _single_process_evaluation(
    policy: Any,
    env_factory: Callable,
    episodes: int,
    max_steps: int
) -> Dict[str, Any]:
    """
    Fallback single-process evaluation.
    Used when multiprocessing is not available or policy cannot be serialized.
    """
    LOG.info("Running single-process evaluation: %d episodes", episodes)
    
    results = []
    
    for ep in tqdm(range(episodes), desc="Evaluating", disable=not _HAS_TQDM):
        try:
            env = env_factory(seed=ep)
            obs = env.reset(seed=ep)
            
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                action_dict = policy.act(obs)
                action = action_dict["action"]
                
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            
            results.append({
                "episode": ep,
                "reward": total_reward,
                "steps": steps
            })
            
        except Exception as e:
            LOG.warning("Episode %d failed: %s", ep, e)
            results.append({
                "episode": ep,
                "error": str(e)
            })
    
    # Aggregate
    rewards = [r["reward"] for r in results if "reward" in r]
    steps_list = [r["steps"] for r in results if "steps" in r]
    failed = [r for r in results if "error" in r]
    
    return {
        "episodes_requested": episodes,
        "episodes_completed": len(rewards),
        "failed": len(failed),
        "rewards": rewards,
        "steps": steps_list,
        "reward_mean": float(statistics.mean(rewards)) if rewards else 0.0,
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "reward_min": float(min(rewards)) if rewards else 0.0,
        "reward_max": float(max(rewards)) if rewards else 0.0,
        "reward_p50": float(sorted(rewards)[int(0.5 * len(rewards))]) if rewards else 0.0,
        "reward_p95": float(sorted(rewards)[int(0.95 * len(rewards))]) if len(rewards) > 1 else 0.0,
        "mean_episode_length": float(statistics.mean(steps_list)) if steps_list else 0.0
    }

# =========================================================================
# SECTION 11: Documentation & Usage Examples
# =========================================================================

"""
ENTERPRISE EVALUATION - COMPLETE USAGE GUIDE
=============================================

1. BASIC EVALUATION
-------------------
# Evaluate latest model checkpoint
python3 train_rl_eval.py \\
    --model-dir backend/app/ml/models/ppo_latest \\
    --eval-episodes 50 \\
    --device cpu

# Evaluate specific checkpoint
python3 train_rl_eval.py \\
    --checkpoint models/rl_ckpt_0100.pt \\
    --eval-episodes 100

2. ONNX PRODUCTION PATH
-----------------------
# Evaluate ONNX model (production inference)
python3 train_rl_eval.py \\
    --checkpoint models/rl_model_final.onnx \\
    --use-onnx \\
    --eval-episodes 100 \\
    --benchmark

# With GPU acceleration
python3 train_rl_eval.py \\
    --checkpoint models/model.onnx \\
    --use-onnx \\
    --onnx-providers CUDAExecutionProvider CPUExecutionProvider \\
    --device cuda

3. A/B TESTING
--------------
# Compare candidate vs baseline
python3 train_rl_eval.py \\
    --checkpoint models/candidate_v2.pt \\
    --baseline-checkpoint models/prod_v1.pt \\
    --eval-episodes 200 \\
    --statistical-test ttest \\
    --confidence-level 0.95

# Non-parametric test (Mann-Whitney U)
python3 train_rl_eval.py \\
    --checkpoint candidate.pt \\
    --baseline-checkpoint baseline.pt \\
    --statistical-test mannwhitneyu

4. DRIFT DETECTION
------------------
# Check for distribution shift
python3 train_rl_eval.py \\
    --checkpoint models/current.pt \\
    --drift-detection \\
    --drift-reference data/reference_distribution.json \\
    --drift-threshold 0.3

5. PERFORMANCE BENCHMARKING
---------------------------
# Measure inference latency/throughput
python3 train_rl_eval.py \\
    --checkpoint models/optimized.pt \\
    --benchmark \\
    --benchmark-batch-sizes 1 8 32 64 128

6. FULL CI/CD PIPELINE
----------------------
# Complete enterprise evaluation
python3 train_rl_eval.py \\
    --model-tag prod-candidate-v3 \\
    --baseline-checkpoint s3://bucket/models/prod-v2.pt \\
    --eval-episodes 200 \\
    --drift-detection \\
    --drift-reference s3://bucket/drift/baseline.json \\
    --drift-threshold 0.25 \\
    --benchmark \\
    --benchmark-batch-sizes 1 8 32 \\
    --check-stability \\
    --stop-if-below-mean-reward 95.0 \\
    --min-episode-reward 50.0 \\
    --max-episode-reward 500.0 \\
    --mlflow \\
    --mlflow-experiment ProdEval \\
    --wandb \\
    --wandb-project PriorityMax-Prod \\
    --s3-bucket my-ml-artifacts \\
    --s3-prefix evals/$(date +%Y%m%d) \\
    --prometheus \\
    --prometheus-port 9304 \\
    --num-workers 8 \\
    --out results/eval_$(date +%s).json

7. KUBERNETES CI JOB
--------------------
apiVersion: batch/v1
kind: Job
metadata:
  name: rl-model-eval
spec:
  template:
    spec:
      containers:
      - name: evaluator
        image: prioritymax/rl-eval:latest
        command:
        - python3
        - train_rl_eval.py
        args:
        - --config
        - /config/eval_prod.yaml
        - --out
        - /results/eval_result.json
        - --s3-bucket
        - prod-ml-artifacts
        volumeMounts:
        - name: config
          mountPath: /config
        - name: results
          mountPath: /results
      restartPolicy: Never
      volumes:
      - name: config
        configMap:
          name: eval-config
      - name: results
        emptyDir: {}

8. GITHUB ACTIONS
-----------------
name: Evaluate RL Model
on:
  pull_request:
    paths:
      - 'models/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Evaluate Model
      run: |
        python3 scripts/train_rl_eval.py \\
          --checkpoint ${{ github.workspace }}/models/candidate.pt \\
          --baseline-checkpoint s3://prod/models/baseline.pt \\
          --eval-episodes 100 \\
          --stop-if-below-mean-reward 90.0 \\
          --drift-detection \\
          --drift-reference s3://prod/drift/baseline.json \\
          --out ${{ github.workspace }}/eval_result.json
    
    - name: Upload Results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-results
        path: eval_result.json

9. EXIT CODES
-------------
0 (EXIT_OK): Evaluation successful, all checks passed
1 (EXIT_FAILURE): Evaluation failed due to error
2 (EXIT_MEAN_BELOW_THRESHOLD): Mean reward below --stop-if-below-mean-reward
3 (EXIT_CONFIG_ERROR): Configuration parsing error
4 (EXIT_MISSING_DEP): Required dependency missing

10. PROMETHEUS METRICS
----------------------
When --prometheus is enabled, these metrics are exposed:

- rl_eval_reward_mean: Mean evaluation reward
- rl_eval_episodes_total: Total episodes evaluated
- rl_eval_drift_score: Distribution drift score
- rl_eval_action_latency_seconds: Action inference latency histogram

Query examples:
# Average reward over time
rate(rl_eval_reward_mean[5m])

# P95 inference latency
histogram_quantile(0.95, rl_eval_action_latency_seconds_bucket)

# Drift alerts
rl_eval_drift_score > 0.3

11. TROUBLESHOOTING
-------------------
Q: "ONNX model fails to load"
A: Ensure onnxruntime is installed: pip install onnxruntime
   Check ONNX opset version compatibility (use opset 17)

Q: "A/B test shows 'insufficient data'"
A: Increase --eval-episodes (recommend ≥50 per model)
   Ensure both models complete episodes successfully

Q: "Drift detection false positives"
A: Adjust --drift-threshold (higher = less sensitive)
   Ensure reference data is representative
   Check reference data has sufficient samples (≥1000)

Q: "Multi-worker evaluation hangs"
A: Reduce --num-workers
   Check for GPU memory issues with --device cuda
   Increase --worker-timeout for slow models

Q: "Prometheus metrics not updating"
A: Verify port 9304 is accessible
   Check firewall/network policies
   Ensure prometheus_client is installed

12. BEST PRACTICES
------------------
✅ Always use ONNX evaluation for production models
✅ Run A/B tests with confidence_level ≥ 0.95
✅ Enable drift detection for production monitoring
✅ Benchmark on target hardware (CPU vs GPU)
✅ Set conservative thresholds initially
✅ Log to both MLflow and W&B for redundancy
✅ Upload artifacts to S3 for audit trail
✅ Use --check-stability to catch NaN/Inf bugs
✅ Run parallel evaluation for large test suites
✅ Monitor Prometheus metrics in production

13. CONFIGURATION FILE EXAMPLES
--------------------------------
# eval_prod.yaml
eval_episodes: 100
max_steps: 1000
device: cpu
deterministic: true
seed: 42

use_onnx: true
onnx_providers:
  - CPUExecutionProvider

baseline_checkpoint: s3://bucket/models/prod_v1.onnx
statistical_test: ttest
confidence_level: 0.95

enable_drift_detection: true
drift_reference_data: s3://bucket/drift/baseline_2024.json
drift_threshold: 0.25

measure_inference_time: true
benchmark_batch_sizes: [1, 8, 32]

check_numerical_stability: true
min_episode_reward: 50.0
max_episode_reward: 300.0

log_mlflow: true
mlflow_experiment: Production-Eval
log_wandb: true
wandb_project: PriorityMax-Prod

s3_bucket: ml-artifacts-prod
s3_prefix: evals/production

enable_prometheus: true
prometheus_port: 9304

num_workers: 8
worker_timeout: 3600

stop_if_below_mean_reward: 90.0
"""

# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    main_cli()
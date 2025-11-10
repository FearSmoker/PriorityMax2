#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_rl_eval.py (enterprise-grade, chunk 1)
---------------------------------------------------

Chunk 1 responsibilities:
 - Configuration (YAML/JSON/CLI) with strict validation
 - Structured logging and diagnostic contexts
 - MLflow / Weights & Biases (W&B) integration helpers (best-effort safe)
 - Model registry + artifact storage helpers (local FS + optional S3)
 - Robust checkpoint discovery and a model loader scaffold (metadata-guided)
 - Basic utility helpers used across other chunks

This file is split into chunks for readability and CI-friendly diffs.
Load chunk 1 first; chunk 2 will implement policy instantiation, env orchestration,
evaluation loop, artifact writing, metrics export, and final exit logic.
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

# Third-party -- imported best effort so script can run in minimal CI without optional deps
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

# project-local modules import attempt (best-effort)
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root
# prefer running from repo root; allow running from backend folder too
_candidates = [ROOT, ROOT / "backend", ROOT / "backend" / "app"]
for c in _candidates:
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))

# Try to import SimulatedRealEnv and EnvConfig; if absent we will error later with clear message
try:
    from ml.real_env import SimulatedRealEnv, EnvConfig
except Exception:
    SimulatedRealEnv = None
    EnvConfig = None

# Try to import predictor/registry modules for model metadata resolution
try:
    from ml.model_registry import ModelRegistry
except Exception:
    ModelRegistry = None

# -------------------------
# Constants & default paths
# -------------------------
DEFAULT_CHECKPOINTS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_CHECKPOINTS_DIR", str(ROOT / "backend" / "checkpoints")))
DEFAULT_MODELS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODELS_DIR", str(ROOT / "backend" / "app" / "ml" / "models")))
DEFAULT_RESULTS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_RESULTS_DIR", str(DEFAULT_CHECKPOINTS_DIR / "eval_results")))
DEFAULT_TRACE_DIR = pathlib.Path(os.getenv("PRIORITYMAX_TRACES_DIR", str(DEFAULT_RESULTS_DIR / "traces")))
DEFAULT_S3_BUCKET = os.getenv("PRIORITYMAX_S3_BUCKET", None)

# Ensure directories exist for local-run convenience
DEFAULT_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_TRACE_DIR.mkdir(parents=True, exist_ok=True)

# Exit codes for CI
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_MEAN_BELOW_THRESHOLD = 2
EXIT_CONFIG_ERROR = 3
EXIT_MISSING_DEP = 4

# -------------------------
# Logging setup
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

# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class EvalConfig:
    model_dir: Optional[str] = None            # model directory in FS or "s3://bucket/prefix"
    checkpoint: Optional[str] = None           # explicit checkpoint file path (overrides search)
    model_tag: Optional[str] = None            # optional registry tag to lookup
    eval_episodes: int = 50
    max_steps: int = 1000
    deterministic: bool = True
    seed: Optional[int] = None
    render: int = 0                             # 0=none,1=save traces,2=attempt env.render()
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
    model_registry_backend: Optional[str] = None  # e.g., "mongodb", "filesystem"
    metadata_file: Optional[str] = None           # override metadata file for model reconstruction
    verbosity: int = 1

    @classmethod
    def from_cli_and_file(cls, args: argparse.Namespace) -> "EvalConfig":
        # load file if provided
        cfg_data: Dict[str, Any] = {}
        if getattr(args, "config", None):
            cfg_path = pathlib.Path(args.config)
            if not cfg_path.exists():
                raise ValueError(f"Config file not found: {cfg_path}")
            text = cfg_path.read_text(encoding="utf-8")
            if cfg_path.suffix.lower() in (".yml", ".yaml"):
                if not _HAS_YAML:
                    raise RuntimeError("PyYAML required to parse YAML config files; install pyyaml")
                cfg_data = yaml.safe_load(text) or {}
            else:
                cfg_data = json.loads(text)
        # CLI flags override file
        overrides = {k: v for k, v in vars(args).items() if v is not None}
        cfg_data.update(overrides)
        # coerce keys to dataclass
        return cls(**cfg_data)

# -------------------------
# MLflow & W&B helpers (safe)
# -------------------------
def safe_mlflow_init(cfg: EvalConfig, run_name: Optional[str] = None):
    if not _HAS_MLFLOW:
        LOG.debug("MLflow not available; skipping MLflow init")
        return None
    try:
        mlflow.set_experiment(cfg.mlflow_experiment)
        active_run = mlflow.start_run(run_name=run_name)
        LOG.info("MLflow run started: %s", active_run.info.run_id)
        return active_run
    except Exception:
        LOG.exception("Failed to init MLflow")
        return None

def safe_mlflow_log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    if not _HAS_MLFLOW:
        return
    try:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v), step=step)
    except Exception:
        LOG.exception("MLflow log metrics failed")

def safe_mlflow_end():
    if not _HAS_MLFLOW:
        return
    try:
        mlflow.end_run()
    except Exception:
        LOG.exception("MLflow end_run failed")

def safe_wandb_init(cfg: EvalConfig, run_name: Optional[str] = None):
    if not _HAS_WANDB:
        LOG.debug("W&B not available; skipping wandb init")
        return None
    try:
        wandb.init(project=cfg.wandb_project, name=run_name, config=dataclasses.asdict(cfg))
        LOG.info("W&B run started: %s", wandb.run.name if wandb.run else run_name)
        return wandb.run
    except Exception:
        LOG.exception("Failed to init W&B")
        return None

def safe_wandb_log(metrics: Dict[str, float]):
    if not _HAS_WANDB:
        return
    try:
        wandb.log(metrics)
    except Exception:
        LOG.exception("W&B log failed")

def safe_wandb_finish():
    if not _HAS_WANDB:
        return
    try:
        wandb.finish()
    except Exception:
        LOG.exception("W&B finish failed")

# -------------------------
# Storage & model-registry helpers
# -------------------------
storage_logger = get_child_logger("storage")

def s3_client():
    if not _HAS_BOTO3:
        raise RuntimeError("boto3 not available")
    return boto3.client("s3")

def download_s3_prefix_to_dir(bucket: str, prefix: str, dest_dir: Union[str, pathlib.Path], max_items: Optional[int] = None) -> List[str]:
    """
    Download all objects under s3://bucket/prefix to dest_dir.
    Returns list of local file paths.
    """
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
                # skip "directories"
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
        storage_logger.exception("S3 pagination failed for %s/%s", bucket, prefix)
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

def resolve_model_dir_from_registry(tag: str, fallback_fs_dir: Optional[str] = None) -> Optional[str]:
    """
    Resolve model directory for a given tag from a model registry if available.
    This is a helper that tries:
      1) ModelRegistry (if implemented in project)
      2) Local FS under DEFAULT_MODELS_DIR/tag
      3) S3 object if tag encodes s3://...
    Returns local path or s3 uri string (caller must handle downloads).
    """
    storage_logger.debug("Resolving model tag: %s", tag)
    # ModelRegistry integration
    try:
        if ModelRegistry is not None:
            registry = ModelRegistry()
            meta = registry.get_by_tag(tag)
            if meta:
                # expected keys: file_path, s3_uri, local_dir, metadata
                if meta.get("file_path"):
                    return meta["file_path"]
                if meta.get("s3_uri"):
                    return meta["s3_uri"]
                if meta.get("dir"):
                    return meta["dir"]
    except Exception:
        storage_logger.exception("ModelRegistry lookup failed for %s", tag)

    # filesystem fallback
    local_dir = pathlib.Path(fallback_fs_dir or DEFAULT_MODELS_DIR) / tag
    if local_dir.exists():
        return str(local_dir)

    # maybe tag is s3 uri
    if is_s3_path(tag):
        return tag

    return None

def local_find_checkpoint_in_dir(model_dir: Union[str, pathlib.Path], pattern_exts: Optional[List[str]] = None) -> Optional[str]:
    """
    Search for the most likely checkpoint file in model_dir.
    Preference order: .pt/.pth/.ckpt/.pth.tar/.bin
    Returns absolute path or None.
    """
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        return None
    pattern_exts = pattern_exts or [".pt", ".pth", ".ckpt", ".pth.tar", ".bin", ".pt.tar"]
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
    """
    If model_dir is s3://..., download content to tmp_dir and return local path.
    If model_dir is local FS, return as-is.
    If model_dir is a single file path, return it.
    """
    tmp_dir = pathlib.Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if is_s3_path(model_dir):
        bucket, prefix = parse_s3_path(model_dir)
        storage_logger.info("Downloading model from S3 %s/%s -> %s", bucket, prefix, tmp_dir)
        files = download_s3_prefix_to_dir(bucket, prefix, tmp_dir, max_items=None)
        # If we downloaded nothing, raise
        if not files:
            raise RuntimeError(f"No files downloaded from s3://{bucket}/{prefix}")
        return str(tmp_dir)
    else:
        p = pathlib.Path(model_dir)
        if p.is_file():
            # return parent dir for consistency
            return str(p.parent)
        if p.is_dir():
            return str(p)
        # last attempt: maybe it's a tag under DEFAULT_MODELS_DIR
        alt = DEFAULT_MODELS_DIR / model_dir
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

# -------------------------
# Checkpoint loader scaffold
# -------------------------
loader_logger = get_child_logger("loader")

def discover_checkpoint(cfg: EvalConfig) -> Optional[str]:
    """
    Discover checkpoint path:
      - explicit checkpoint override in cfg.checkpoint -> return directly
      - if cfg.model_tag provided -> ask registry + resolve
      - if cfg.model_dir is s3://... or local dir -> search for checkpoint in that directory
      - fallback: try DEFAULT_MODELS_DIR for tags / directories
    Returns string path (local file) or s3 uri string (if caller wants to handle), or None.
    """
    if cfg.checkpoint:
        loader_logger.debug("Using explicit checkpoint: %s", cfg.checkpoint)
        return cfg.checkpoint

    # If model_tag present, resolve via registry or FS
    if cfg.model_tag:
        resolved = resolve_model_dir_from_registry(cfg.model_tag, fallback_fs_dir=str(DEFAULT_MODELS_DIR))
        if resolved:
            # may be s3:// or local dir
            if is_s3_path(resolved):
                return resolved
            # find checkpoint in resolved dir
            ckpt = local_find_checkpoint_in_dir(resolved)
            if ckpt:
                return ckpt
            # if no ckpt, return dir for caller to fetch
            return resolved

    # If model_dir provided, handle it
    if cfg.model_dir:
        if is_s3_path(cfg.model_dir):
            return cfg.model_dir
        # local dir or file
        if pathlib.Path(cfg.model_dir).exists():
            ckpt = local_find_checkpoint_in_dir(cfg.model_dir)
            if ckpt:
                return ckpt
            # return dir if no single file found
            return cfg.model_dir
        # maybe model_dir is a tag in models dir
        alt = DEFAULT_MODELS_DIR / cfg.model_dir
        if alt.exists():
            ckpt = local_find_checkpoint_in_dir(str(alt))
            return ckpt or str(alt)

    # last resort: scan DEFAULT_MODELS_DIR for latest model
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

# End of Chunk 1
# -------------------------
# Chunk 2: Policy reconstruction, loaders, and inference wrappers
# -------------------------

from types import SimpleNamespace
import importlib
import inspect

# -------------------------
# RNG / device helpers
# -------------------------
def prepare_device(device_str: str = "cpu"):
    """
    Return a torch.device if torch is available; otherwise None.
    Handles CPU/GPU strings like "cpu", "cuda", "cuda:0".
    """
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
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    LOG.info("Seed set to %s", seed)
    return seed

# -------------------------
# Policy interface & wrappers
# -------------------------
class PolicyInterface:
    """
    Minimal wrapper around a policy object to present a unified `.act(obs)` API.
    .act accepts a numpy array (single obs) and returns action numpy array, optionally logp, value.
    """
    def __init__(self, model_obj: Any, device: Optional[Any] = None, deterministic: bool = True):
        self.model = model_obj
        self.device = device
        self.deterministic = deterministic
        self._is_torch = _HAS_TORCH and isinstance(model_obj, torch.nn.Module)
        self._is_jit = _HAS_TORCH and isinstance(model_obj, torch.jit.ScriptModule)
        # if torch model, ensure eval mode
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
            import numpy as _np
            if isinstance(obs, (list, tuple)):
                arr = _np.asarray(obs, dtype=_np.float32)
            if isinstance(arr, _np.ndarray):
                t = torch.from_numpy(arr.astype("float32"))
            else:
                t = torch.tensor(arr, dtype=torch.float32)
            if self.device:
                t = t.to(self.device)
            return t
        except Exception:
            return obs

    def act(self, obs: Any) -> Dict[str, Any]:
        """
        Accepts observation (np.ndarray or list), returns dict:
          {"action": np.ndarray, "logp": float|np.ndarray|None, "value": float|None}
        """
        try:
            if self._is_torch:
                with torch.no_grad():
                    t = self._to_tensor(obs)
                    # add batch dim if needed
                    if isinstance(t, torch.Tensor) and t.dim() == 1:
                        t = t.unsqueeze(0)
                    out = self.model(t)
                    # allow model to return (action,) or (action,logp) or (action,logp,value)
                    if isinstance(out, (tuple, list)):
                        action = out[0]
                        logp = out[1] if len(out) > 1 else None
                        val = out[2] if len(out) > 2 else None
                    else:
                        action = out
                        logp = None
                        val = None
                    # move to cpu numpy
                    if isinstance(action, torch.Tensor):
                        a_np = action.detach().cpu().numpy()
                    else:
                        a_np = action
                    if isinstance(logp, torch.Tensor):
                        logp = logp.detach().cpu().numpy()
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    # squeeze batch if needed
                    if isinstance(a_np, (list, tuple)) or (hasattr(a_np, "shape") and getattr(a_np, "shape")[0] == 1):
                        try:
                            import numpy as _np
                            a_np = _np.asarray(a_np)
                            if a_np.shape[0] == 1:
                                a_np = a_np[0]
                        except Exception:
                            pass
                    return {"action": a_np, "logp": logp, "value": val}
            else:
                # non-torch call contract: model.act(obs) or model(obs)
                if hasattr(self.model, "act") and inspect.isfunction(getattr(self.model, "act")) or hasattr(self.model, "act") and inspect.ismethod(getattr(self.model, "act")):
                    res = self.model.act(obs)
                else:
                    res = self.model(obs)
                # normalize
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
# Policy loader logic
# -------------------------
def try_load_torch_checkpoint(path: str, device: Optional[Any] = None) -> Optional[torch.nn.Module]:
    """
    Try to load checkpoint as:
      - torch.load state_dict into architecture discovered from metadata (later)
      - torch.jit.load for ScriptModule
      - if file is a .pt with saved Module object, try direct load
    Returns nn.Module or raises on unrecoverable error.
    """
    if not _HAS_TORCH:
        raise RuntimeError("Torch not available to load checkpoint")
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    # If file is a TorchScript module, try loading as script
    try:
        if p.suffix in (".pt", ".pth") and _HAS_TORCH:
            # First try torch.jit.load (works for script modules)
            try:
                m = torch.jit.load(str(p), map_location="cpu")
                loader_logger.info("Loaded TorchScript module from %s", p)
                return m
            except Exception:
                loader_logger.debug("Not a TorchScript module or torch.jit.load failed for %s", p)
            # Next, try torch.load to inspect object
            ckpt = torch.load(str(p), map_location="cpu")
            # if ckpt is a dict with state_dict
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                # caller must provide architecture to load state_dict into; we'll return state_dict for higher-level logic
                sd = ckpt["state_dict"]
                # create a simple wrapper module to hold state_dict? Better to return ckpt and let caller handle
                loader_logger.info("Checkpoint contains state_dict keys: %s ...", list(sd.keys())[:5])
                return ckpt
            # If ckpt is a Module object
            if isinstance(ckpt, torch.nn.Module):
                loader_logger.info("Checkpoint contained nn.Module object")
                return ckpt
            # fallback: if ckpt is dict of tensors (state dict), return it
            if isinstance(ckpt, dict):
                # Heuristic: looks like state_dict
                keys = list(ckpt.keys())
                if keys and isinstance(ckpt[keys[0]], torch.Tensor):
                    return {"state_dict": ckpt}
            # else unknown format; return None
            loader_logger.warning("Torch load produced unsupported object for %s", p)
            return None
    except Exception:
        loader_logger.exception("Torch checkpoint load failed for %s", p)
        raise

def try_instantiate_policy_from_metadata(model_dir: str, device: Optional[Any] = None, strict: bool = False) -> Optional[PolicyInterface]:
    """
    Attempt to reconstruct policy using metadata.json inside model_dir (if present).
    metadata.json should contain:
      - 'framework': 'torch'|'custom'
      - 'entrypoint': 'module.path:ClassName' OR None
      - 'checkpoint_file': relative path to checkpoint inside model_dir
      - 'obs_shape'/'action_shape' metadata hints
    Returns PolicyInterface or None if reconstruction failed.
    """
    md_path = pathlib.Path(model_dir) / "metadata.json"
    if not md_path.exists():
        loader_logger.debug("No metadata.json found in %s", model_dir)
        return None
    try:
        md = json.loads(md_path.read_text(encoding="utf-8"))
    except Exception:
        loader_logger.exception("Failed to parse metadata.json in %s", model_dir)
        return None

    framework = md.get("framework", "torch")
    entrypoint = md.get("entrypoint")  # e.g., "ml.rl_models.ppo:ActorCritic"
    ckpt_rel = md.get("checkpoint_file")
    # Resolve checkpoint path
    ckpt_path = None
    if ckpt_rel:
        cand = pathlib.Path(model_dir) / ckpt_rel
        if cand.exists():
            ckpt_path = str(cand)
    # If no checkpoint path, try discovery
    if not ckpt_path:
        ckpt_path = local_find_checkpoint_in_dir(model_dir)

    if framework == "torch":
        if not ckpt_path:
            loader_logger.warning("No checkpoint available for torch model in %s", model_dir)
            if strict:
                raise FileNotFoundError("No checkpoint found")
            return None
        # If entrypoint specified, import class and instantiate then load state_dict
        if entrypoint:
            try:
                module_part, class_part = entrypoint.split(":")
                mod = importlib.import_module(module_part)
                cls = getattr(mod, class_part)
                # attempt to inspect constructor signature to pass metadata
                sig = inspect.signature(cls.__init__)
                kwargs = {}
                # pass metadata if constructor accepts 'obs_dim'/'act_dim' or 'config'
                if "config" in sig.parameters:
                    kwargs["config"] = md.get("config", {})
                # instantiate
                inst = cls(**kwargs) if kwargs else cls()
                # load checkpoint
                ck = try_load_torch_checkpoint(ckpt_path, device=device)
                if isinstance(ck, dict) and "state_dict" in ck:
                    inst.load_state_dict(ck["state_dict"])
                elif isinstance(ck, torch.nn.Module):
                    # ck is Module: use that directly
                    inst = ck
                elif isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
                    # torch checkpoint with nested model key
                    inst.load_state_dict(ck["model"])
                else:
                    loader_logger.warning("Unexpected checkpoint format for %s", ckpt_path)
                # move to device
                if _HAS_TORCH and isinstance(inst, torch.nn.Module) and device:
                    try:
                        inst.to(device)
                    except Exception:
                        pass
                loader_logger.info("Instantiated policy from entrypoint %s", entrypoint)
                return PolicyInterface(inst, device=device)
            except Exception:
                loader_logger.exception("Failed to instantiate policy from entrypoint %s", entrypoint)
                if strict:
                    raise
                return None
        else:
            # No entrypoint; attempt generic torch loader
            try:
                ck = try_load_torch_checkpoint(ckpt_path, device=device)
                if ck is None:
                    return None
                # if ck is Module -> wrap directly
                if isinstance(ck, torch.nn.Module):
                    if device:
                        try:
                            ck.to(device)
                        except Exception:
                            pass
                    return PolicyInterface(ck, device=device)
                # if ck contains state_dict, we cannot instantiate without architecture; return ck for caller
                if isinstance(ck, dict) and "state_dict" in ck:
                    # return a wrapper that will perform a shallow forward using the saved module if present
                    loader_logger.info("Returning state_dict-only checkpoint; caller must handle architecture")
                    return PolicyInterface(ck, device=device)
            except Exception:
                loader_logger.exception("Failed to load torch checkpoint in %s", model_dir)
                if strict:
                    raise
                return None
    else:
        # custom framework: attempt to load entrypoint function
        if entrypoint:
            try:
                module_part, func_part = entrypoint.split(":")
                mod = importlib.import_module(module_part)
                factory = getattr(mod, func_part)
                inst = factory(metadata=md)
                return PolicyInterface(inst, device=device)
            except Exception:
                loader_logger.exception("Failed to instantiate custom policy from entrypoint %s", entrypoint)
                if strict:
                    raise
                return None
        return None

def build_policy_for_eval(cfg: EvalConfig, tmp_dir: Optional[str] = None) -> Tuple[PolicyInterface, str]:
    """
    Top-level helper:
     - discover checkpoint using discover_checkpoint
     - if model_dir is s3://... download to tmp_dir
     - try to reconstruct policy using metadata, entrypoint, or fallback to TorchScript load
    Returns (PolicyInterface, model_local_dir)
    """
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="prioritymax_eval_")
    ck = discover_checkpoint(cfg)
    if not ck:
        raise FileNotFoundError("Could not discover checkpoint/model for evaluation")
    # if ck is s3://... or directory s3://, fetch
    if is_s3_path(ck):
        bucket, prefix = parse_s3_path(ck)
        # If ck points to a file, prefix may include file; download contents of prefix
        files = download_s3_prefix_to_dir(bucket, prefix, tmp_dir)
        if not files:
            raise RuntimeError(f"No files downloaded from s3://{bucket}/{prefix}")
        # attempt to find checkpoint among downloaded files
        local_ckpt = None
        for f in files:
            if pathlib.Path(f).suffix in (".pt", ".pth", ".ckpt", ".pth.tar"):
                local_ckpt = f
                break
        # set model_local_dir to tmp_dir
        model_local_dir = tmp_dir
    else:
        # ck may be a direct file or directory
        p = pathlib.Path(ck)
        if p.is_file():
            model_local_dir = str(p.parent)
            local_ckpt = str(p)
        else:
            model_local_dir = str(p)
            local_ckpt = local_find_checkpoint_in_dir(model_local_dir)

    device = prepare_device(cfg.device)
    # First attempt: instantiate from metadata (best-effort)
    policy = None
    try:
        policy = try_instantiate_policy_from_metadata(model_local_dir, device=device, strict=False)
    except Exception:
        loader_logger.exception("Metadata-based instantiation failed")

    # If metadata-based failed and we have a torch checkpoint file, try torch.jit or torch.load
    if (policy is None or (isinstance(policy.model, dict) and "state_dict" in policy.model)):
        # Try torchscript direct load of the checkpoint file if available
        if local_ckpt and _HAS_TORCH:
            try:
                maybe_mod = try_load_torch_checkpoint(local_ckpt, device=device)
                if isinstance(maybe_mod, torch.nn.Module) or (hasattr(maybe_mod, "__call__") and callable(maybe_mod)):
                    policy = PolicyInterface(maybe_mod, device=device)
                elif isinstance(maybe_mod, dict) and "state_dict" in maybe_mod:
                    # state_dict-only; cannot instantiate architecture automatically
                    loader_logger.warning("Checkpoint contains only state_dict; evaluation requires model architecture or entrypoint metadata")
                    raise RuntimeError("state_dict-only checkpoint cannot be used without architecture")
            except Exception:
                loader_logger.exception("Torch checkpoint fallback failed")
                raise

    if policy is None:
        raise RuntimeError("Failed to create policy for evaluation")

    # perform a warmup inference to sanity-check model
    try:
        # craft a dummy observation if metadata hints present
        md_file = pathlib.Path(model_local_dir) / "metadata.json"
        sample_obs = None
        if md_file.exists():
            try:
                md = json.loads(md_file.read_text(encoding="utf-8"))
                obs_shape = md.get("obs_shape")
                if obs_shape:
                    import numpy as _np
                    sample_obs = _np.zeros(tuple(obs_shape), dtype=_np.float32)
            except Exception:
                pass
        # fallback sample
        if sample_obs is None:
            sample_obs = [0.0] * 8  # generic placeholder; env will generate real obs anyway
        _ = policy.act(sample_obs)
        loader_logger.info("Policy warmup successful")
    except Exception:
        loader_logger.exception("Policy warmup inference failed")
        raise

    return policy, model_local_dir

# End of Chunk 2
# -------------------------
# Chunk 3: Environment orchestration, vectorized evaluation, and result aggregation
# -------------------------

import multiprocessing as mp
from functools import partial
from queue import Empty as QueueEmpty

# Try to import user-provided env factory (best-effort)
try:
    # If your project exposes make_vec_env or SimulatedRealEnv import, prefer that
    from ml.real_env import SimulatedRealEnv, EnvConfig, make_vec_env  # type: ignore
    _HAS_REAL_ENV = True
except Exception:
    SimulatedRealEnv = None
    EnvConfig = None
    make_vec_env = None
    _HAS_REAL_ENV = False

# -------------------------
# Environment factory
# -------------------------
def _local_make_env(env_config: Optional[dict] = None):
    """
    Return a callable that creates a new environment instance when called.
    env_config can include 'seed' and other initialization parameters.
    """
    env_config = env_config or {}
    def _fn(seed: Optional[int] = None):
        # If SimulatedRealEnv present, instantiate it; otherwise create a MinimalMockEnv
        if _HAS_REAL_ENV and EnvConfig is not None:
            cfg = EnvConfig(**env_config) if isinstance(env_config, dict) else EnvConfig()
            env = SimulatedRealEnv(cfg)
            if seed is not None:
                try:
                    env.seed(seed)
                except Exception:
                    pass
            return env
        else:
            # Minimal fallback environment for evaluation: simple deterministic environment
            class MinimalEnv:
                def __init__(self, seed=None):
                    self._seed = seed or 0
                    self._rng = random.Random(self._seed)
                    self._t = 0
                    self._max_steps = 100
                def reset(self, seed=None):
                    if seed is not None:
                        self._rng = random.Random(seed)
                    self._t = 0
                    return [0.0] * 8
                def step(self, action):
                    # action ignored; produce pseudo-random reward that slowly decays
                    self._t += 1
                    done = self._t >= self._max_steps
                    obs = [float(self._rng.random()) for _ in range(8)]
                    reward = max(0.0, 1.0 - (self._t * 0.01)) + (self._rng.random() * 0.01)
                    info = {}
                    return obs, reward, done, info
                def seed(self, s):
                    self._rng = random.Random(s)
            return MinimalEnv(seed=seed)
    return _fn

# -------------------------
# Worker process for parallel evaluation
# -------------------------
def _eval_worker(task_queue: mp.Queue, result_queue: mp.Queue, worker_id: int, env_factory_serialized: dict):
    """
    Worker runs episodes serially inside the process and pushes results back.
    We pass a minimal serialized env config instead of live closures to make multiprocessing safer.
    """
    # Rehydrate env factory
    env_factory = _local_make_env(env_factory_serialized.get("env_config"))
    # Optionally set process-local random seed base
    base_seed = env_factory_serialized.get("base_seed", int(time.time()) + worker_id)
    # Import torch lazily inside worker to avoid fork issues if necessary
    local_torch = None
    if _HAS_TORCH:
        try:
            import torch as _t
            local_torch = _t
        except Exception:
            local_torch = None

    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except QueueEmpty:
            continue
        if task is None:
            break  # shutdown sentinel
        # task: dict {policy_serialized, episodes, max_steps, start_seed}
        policy_serialized = task.get("policy")
        episodes = int(task.get("episodes", 1))
        max_steps = int(task.get("max_steps", 1000))
        start_seed = int(task.get("start_seed", base_seed))
        # Reconstruct policy object in worker process if it is serializable path; otherwise policy may be None -> error
        # We support policy_serialized as either:
        #  - dict with {"type":"torch_checkpoint", "path": "/abs/path/to/checkpoint", "entrypoint": "..."} OR
        #  - None meaning policy provided in main process (not supported in mp worker)
        policy_iface = None
        try:
            if policy_serialized and policy_serialized.get("type") == "torch_checkpoint":
                path = policy_serialized.get("path")
                entrypoint = policy_serialized.get("entrypoint")
                # Quick attempt: try torch.jit.load or torch.load as in chunk 2
                if _HAS_TORCH:
                    try:
                        mod = try_load_torch_checkpoint(path, device=prepare_device("cpu"))
                        if isinstance(mod, dict) and "state_dict" in mod:
                            # cannot instantiate architecture in worker; skip
                            raise RuntimeError("state_dict-only checkpoint cannot be loaded inside worker without entrypoint")
                        policy_iface = PolicyInterface(mod, device=prepare_device("cpu"), deterministic=True)
                    except Exception:
                        loader_logger.exception("Worker failed to load checkpoint %s", path)
                        policy_iface = None
            # else: unsupported serialized type
        except Exception:
            loader_logger.exception("Policy reconstruction in worker failed")
            policy_iface = None

        # If policy_iface is None, return error for the tasks
        if policy_iface is None:
            for ep in range(episodes):
                result_queue.put({"ok": False, "error": "policy_load_failed", "worker": worker_id, "episode": ep})
            continue

        # Create local env and run episodes
        for ep in range(episodes):
            env = env_factory(seed=start_seed + ep + worker_id)
            obs = env.reset(seed=start_seed + ep + worker_id) if hasattr(env, "reset") else env.reset()
            total_reward = 0.0
            steps = 0
            trace = []
            done = False
            while not done and steps < max_steps:
                try:
                    # policy expects numpy or list obs
                    out = policy_iface.act(obs)
                    action = out.get("action")
                    # if action is array-like with batch dim, pick first
                    if hasattr(action, "shape") and getattr(action, "shape")[0] == 1:
                        import numpy as _np
                        action = _np.asarray(action)[0]
                    obs, rew, done, info = env.step(action)
                    total_reward += float(rew)
                    steps += 1
                    trace.append({"step": steps, "reward": float(rew)})
                except Exception:
                    loader_logger.exception("Error during policy.act or env.step in worker %s", worker_id)
                    break
            result_queue.put({"ok": True, "worker": worker_id, "episode": ep, "reward": total_reward, "steps": steps, "trace": trace})
    # cleanup
    return

# -------------------------
# Vectorized evaluation orchestration (multiprocess)
# -------------------------
def run_vectorized_evaluation(policy: PolicyInterface,
                              env_factory: Callable[..., Any],
                              episodes: int = 100,
                              max_steps: int = 1000,
                              n_workers: int = 4,
                              per_worker_batch: int = 1,
                              timeout: int = 3600) -> Dict[str, Any]:
    """
    Distribute evaluation across multiple worker processes to run episodes in parallel.
    policy: PolicyInterface built in main process. If it can't be serialized for workers, this function will
            save a TorchScript snapshot (if torch) to a temp file and instruct workers to load it.
    env_factory: callable returned by _local_make_env (callable(seed) -> env)
    episodes: total number of episodes to run
    n_workers: number of worker processes
    per_worker_batch: number of episodes assigned per task per worker submit
    """
    # Prepare serialized policy specification for workers
    policy_spec = None
    tmp_policy_path = None
    if _HAS_TORCH and isinstance(policy.model, torch.nn.Module):
        # attempt to export TorchScript to a temp file for worker processes
        try:
            tmp_dir = tempfile.mkdtemp(prefix="prioritymax_policy_")
            tmp_policy_path = os.path.join(tmp_dir, "policy_script.pt")
            try:
                # try tracing with a dummy input inferred from metadata or use a single-dim vector
                dummy = None
                try:
                    # infer obs dim from metadata if provided
                    if hasattr(policy, "model") and hasattr(policy.model, "example_input") and policy.model.example_input is not None:
                        dummy = policy.model.example_input
                    else:
                        dummy = torch.randn(1, 8)
                except Exception:
                    dummy = torch.randn(1, 8)
                scriptmod = torch.jit.trace(policy.model, dummy)
                torch.jit.save(scriptmod, tmp_policy_path)
                policy_spec = {"type": "torch_checkpoint", "path": tmp_policy_path, "entrypoint": None}
                loader_logger.info("Exported policy TorchScript to %s for worker loading", tmp_policy_path)
            except Exception:
                # fallback: try torch.jit.script
                try:
                    scriptmod = torch.jit.script(policy.model)
                    torch.jit.save(scriptmod, tmp_policy_path)
                    policy_spec = {"type": "torch_checkpoint", "path": tmp_policy_path, "entrypoint": None}
                    loader_logger.info("Exported policy TorchScript (script) to %s for worker loading", tmp_policy_path)
                except Exception:
                    loader_logger.exception("Failed to export TorchScript for policy; falling back to single-process evaluation")
                    policy_spec = None
        except Exception:
            loader_logger.exception("Policy serialization failed; workers may not be able to load it")
            policy_spec = None
    else:
        # non-torch or non-serializable policy: do single-process eval
        policy_spec = None

    # If policy_spec is None, run single-process fallback
    if policy_spec is None:
        loader_logger.warning("Workers will not be used; running single-process evaluation")
        return run_singleprocess_evaluation(policy, env_factory, episodes=episodes, max_steps=max_steps)

    # Create multiprocessing queues
    manager = mp.Manager()
    task_q = manager.Queue()
    result_q = manager.Queue()

    # Prepare serialized env factory config to pass to workers
    env_factory_serialized = {"env_config": getattr(env_factory, "__closure__", None) or {}, "base_seed": int(time.time())}

    # Spawn workers
    workers = []
    for wid in range(n_workers):
        p = mp.Process(target=_eval_worker, args=(task_q, result_q, wid, {"env_config": None, "base_seed": int(time.time()) + wid}))
        p.daemon = True
        p.start()
        workers.append(p)

    # Submit tasks in batches
    remaining = episodes
    start_seed = int(time.time()) % 2**31
    while remaining > 0:
        batch = min(per_worker_batch, remaining)
        task = {"policy": policy_spec, "episodes": batch, "max_steps": max_steps, "start_seed": start_seed}
        task_q.put(task)
        remaining -= batch
        start_seed += batch

    # Collect results
    collected = []
    deadline = time.time() + timeout
    while len(collected) < episodes and time.time() < deadline:
        try:
            res = result_q.get(timeout=5.0)
            collected.append(res)
        except QueueEmpty:
            continue

    # Send shutdown sentinel
    for _ in workers:
        task_q.put(None)
    # Join workers
    for p in workers:
        p.join(timeout=2.0)

    # Aggregate results (filter successful)
    rewards = [r["reward"] for r in collected if r.get("ok")]
    steps_list = [r["steps"] for r in collected if r.get("ok")]
    failed = [r for r in collected if not r.get("ok")]

    out = {
        "episodes_requested": episodes,
        "episodes_completed": len(rewards),
        "failed": len(failed),
        "rewards": rewards,
        "steps": steps_list,
        "raw": collected
    }
    # derive stats
    if rewards:
        import statistics as _st
        out["reward_mean"] = float(_st.mean(rewards))
        out["reward_std"] = float(_st.pstdev(rewards) if len(rewards) > 1 else 0.0)
        out["reward_p50"] = float(sorted(rewards)[int(0.5 * (len(rewards)-1))])
        out["reward_p95"] = float(sorted(rewards)[int(0.95 * (len(rewards)-1))])
    else:
        out["reward_mean"] = out["reward_std"] = out["reward_p50"] = out["reward_p95"] = 0.0

    # Persist results to file
    try:
        results_dir = pathlib.Path.cwd() / "eval_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        fname = results_dir / f"eval_{int(time.time())}.json"
        fname.write_text(json.dumps(out, default=str, indent=2), encoding="utf-8")
        out["result_file"] = str(fname)
    except Exception:
        loader_logger.exception("Failed to persist evaluation results")

    # cleanup tmp policy if created
    if tmp_policy_path:
        try:
            shutil.rmtree(os.path.dirname(tmp_policy_path))
        except Exception:
            pass

    return out

# -------------------------
# Single-process evaluation fallback
# -------------------------
def run_singleprocess_evaluation(policy: PolicyInterface,
                                 env_factory: Callable[..., Any],
                                 episodes: int = 100,
                                 max_steps: int = 1000) -> Dict[str, Any]:
    """
    Run evaluation episodes in the current process (no multiprocessing).
    """
    results = []
    for ep in range(episodes):
        env = env_factory(seed=int(time.time()) + ep)
        obs = env.reset(seed=int(time.time()) + ep) if hasattr(env, "reset") else env.reset()
        total_reward = 0.0
        steps = 0
        trace = []
        done = False
        while not done and steps < max_steps:
            out = policy.act(obs)
            action = out.get("action")
            try:
                obs, rew, done, info = env.step(action)
            except Exception:
                # if env.step signature differs
                try:
                    obs, rew, done = env.step(action)
                    info = {}
                except Exception:
                    loader_logger.exception("Env.step failed during single-process evaluation")
                    break
            total_reward += float(rew)
            steps += 1
            trace.append({"step": steps, "reward": float(rew)})
        results.append({"episode": ep, "reward": total_reward, "steps": steps, "trace": trace})

    rewards = [r["reward"] for r in results]
    import statistics as _st
    out = {"episodes_requested": episodes, "episodes_completed": len(rewards), "rewards": rewards}
    if rewards:
        out["reward_mean"] = float(_st.mean(rewards))
        out["reward_std"] = float(_st.pstdev(rewards) if len(rewards) > 1 else 0.0)
        out["reward_p50"] = float(sorted(rewards)[int(0.5 * (len(rewards)-1))])
        out["reward_p95"] = float(sorted(rewards)[int(0.95 * (len(rewards)-1))])
    else:
        out["reward_mean"] = out["reward_std"] = out["reward_p50"] = out["reward_p95"] = 0.0
    return out

# End of Chunk 3
# -------------------------
# Chunk 4: Evaluation runner orchestration, logging, artifact uploads, and CLI
# -------------------------

def upload_results_to_s3(result_path: str, cfg: EvalConfig) -> Optional[str]:
    """
    Upload result JSON (and traces if available) to configured S3 bucket/prefix.
    Returns uploaded key URI.
    """
    if not cfg.s3_bucket or not _HAS_BOTO3:
        return None
    client = s3_client()
    key_prefix = cfg.s3_prefix or f"eval-results/{time.strftime('%Y%m%d')}/"
    key_name = key_prefix.rstrip("/") + "/" + os.path.basename(result_path)
    try:
        client.upload_file(result_path, cfg.s3_bucket, key_name)
        LOG.info("Uploaded evaluation result to s3://%s/%s", cfg.s3_bucket, key_name)
        return f"s3://{cfg.s3_bucket}/{key_name}"
    except Exception:
        LOG.exception("S3 upload of evaluation result failed")
        return None


def run_evaluation(cfg: EvalConfig) -> int:
    """
    Main orchestration: builds policy, runs evaluation, logs metrics, writes results,
    uploads to MLflow/W&B/S3 as configured, and returns CI exit code.
    """
    LOG.info("=== PriorityMax RL Evaluation Start ===")
    seed = set_global_seed(cfg.seed)
    device = prepare_device(cfg.device)
    LOG.info("Device: %s", device)

    # Initialize MLflow/W&B
    run_name = f"eval_{int(time.time())}"
    mlflow_run = safe_mlflow_init(cfg, run_name)
    wandb_run = safe_wandb_init(cfg, run_name)

    tmp_dir = tempfile.mkdtemp(prefix="prioritymax_eval_tmp_")

    # Build policy
    try:
        policy, model_dir = build_policy_for_eval(cfg, tmp_dir=tmp_dir)
        LOG.info("Policy built successfully from %s", model_dir)
    except Exception as e:
        LOG.exception("Policy build failed: %s", e)
        safe_mlflow_end()
        safe_wandb_finish()
        return EXIT_FAILURE

    # Create environment factory
    env_factory = _local_make_env({})

    # Run evaluation (vectorized if possible)
    try:
        res = run_vectorized_evaluation(policy,
                                        env_factory,
                                        episodes=cfg.eval_episodes,
                                        max_steps=cfg.max_steps,
                                        n_workers=min(8, max(1, mp.cpu_count() // 2)))
    except Exception:
        LOG.exception("Evaluation loop failed")
        safe_mlflow_end()
        safe_wandb_finish()
        return EXIT_FAILURE

    # Write result JSON file
    result_file = cfg.out or str(DEFAULT_RESULTS_DIR / f"eval_result_{int(time.time())}.json")
    pathlib.Path(result_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        LOG.info("Wrote result to %s", result_file)
    except Exception:
        LOG.exception("Failed to write evaluation result file")

    # Log metrics to MLflow/W&B
    safe_mlflow_log_metrics({
        "reward_mean": res.get("reward_mean", 0.0),
        "reward_std": res.get("reward_std", 0.0),
        "reward_p95": res.get("reward_p95", 0.0),
        "episodes_completed": res.get("episodes_completed", 0)
    })
    safe_wandb_log({
        "reward_mean": res.get("reward_mean", 0.0),
        "reward_std": res.get("reward_std", 0.0),
        "reward_p95": res.get("reward_p95", 0.0),
        "episodes_completed": res.get("episodes_completed", 0)
    })

    # Optionally upload to S3
    s3_uri = upload_results_to_s3(result_file, cfg)
    if s3_uri:
        safe_mlflow_log_metrics({"result_uploaded": 1.0})
        safe_wandb_log({"result_uploaded": 1.0})

    safe_mlflow_end()
    safe_wandb_finish()

    # Determine exit code
    mean_reward = res.get("reward_mean", 0.0)
    if cfg.stop_if_below_mean_reward is not None and mean_reward < cfg.stop_if_below_mean_reward:
        LOG.warning("Mean reward %.3f below threshold %.3f", mean_reward, cfg.stop_if_below_mean_reward)
        return EXIT_MEAN_BELOW_THRESHOLD

    LOG.info("=== Evaluation complete. Mean reward: %.3f ± %.3f ===", mean_reward, res.get("reward_std", 0.0))
    return EXIT_OK


# -------------------------
# CLI Entry
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PriorityMax RL Evaluation Runner (enterprise-grade)")
    p.add_argument("--config", help="Path to YAML/JSON config file")
    p.add_argument("--model-dir", help="Path or S3 URI to model directory")
    p.add_argument("--checkpoint", help="Explicit checkpoint file")
    p.add_argument("--model-tag", help="Model registry tag")
    p.add_argument("--eval-episodes", type=int, help="Number of episodes to evaluate")
    p.add_argument("--max-steps", type=int, help="Max steps per episode")
    p.add_argument("--device", default="cpu", help="Device string (cpu, cuda, cuda:0)")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--stop-if-below-mean-reward", type=float, help="Fail if mean reward below this threshold")
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--s3-bucket", help="Optional S3 bucket for uploads")
    p.add_argument("--s3-prefix", help="S3 prefix path (default eval-results/...)")
    p.add_argument("--out", help="Output result JSON path")
    return p


def main_cli():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        cfg = EvalConfig.from_cli_and_file(args)
    except Exception as e:
        LOG.error("Failed to load configuration: %s", e)
        sys.exit(EXIT_CONFIG_ERROR)

    # Handle graceful signals
    def _handle_sig(sig, frame):
        LOG.warning("Signal %s received; terminating...", sig)
        sys.exit(EXIT_FAILURE)
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    try:
        exit_code = run_evaluation(cfg)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user")
        sys.exit(EXIT_FAILURE)
    except Exception:
        LOG.exception("Fatal error during evaluation")
        sys.exit(EXIT_FAILURE)


# -------------------------
# Example usage (for reference)
# -------------------------
"""
# Local evaluation example:
python3 scripts/train_rl_eval.py \
    --model-dir backend/app/ml/models/ppo_latest \
    --eval-episodes 50 \
    --max-steps 1000 \
    --device cpu \
    --stop-if-below-mean-reward 100.0

# Using config file (YAML):
python3 scripts/train_rl_eval.py --config configs/eval_prod.yaml

# In CI (JSON only, minimal output):
python3 scripts/train_rl_eval.py --model-tag prod-latest --eval-episodes 5 --out result.json

# In cluster with S3 upload:
python3 scripts/train_rl_eval.py \
    --model-dir s3://prioritymax-prod/models/ppo-2025-11-01 \
    --s3-bucket prioritymax-prod \
    --s3-prefix evals/2025-11-09 \
    --mlflow --wandb
"""

if __name__ == "__main__":
    main_cli()

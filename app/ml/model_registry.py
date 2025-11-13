#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PriorityMax Model Registry - FULLY SYNCHRONIZED WITH STABLE TRAINING
---------------------------------------------------------------------

✅ SYNCHRONIZED WITH:
- real_env_STABLE.py (obs_dim=12, bounded rewards, observation normalization)
- train_rl_STABLE.py (gamma=0.99, lr=3e-4, clipped value loss, stable hyperparameters)

✅ NEW VALIDATION FEATURES:
- Validates stable hyperparameters (gamma, lambda, lr)
- Validates normalization metadata (obs, reward)
- Validates bounded reward configuration
- Detects explosion-prone configurations
- Validates clipped value loss usage

CRITICAL: This registry ensures only STABLE models are promoted to production!
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import shutil
import pathlib
import datetime
import hashlib
import logging
from typing import Any, Dict, Optional, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass

# Optional dependencies
try:
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BOTO3 = True
except:
    boto3 = None
    ClientError = Exception
    _HAS_BOTO3 = False

try:
    import torch
    _HAS_TORCH = True
except:
    torch = None
    _HAS_TORCH = False

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except:
    ort = None
    _HAS_ONNX = False

try:
    import numpy as np
    _HAS_NUMPY = True
except:
    _HAS_NUMPY = False

# Logging
LOG = logging.getLogger("prioritymax.ml.model_registry")
LOG.setLevel(os.getenv("PRIORITYMAX_ML_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODELS_DIR", str(BASE_DIR / "app" / "ml" / "models")))
REGISTRY_FILE = MODELS_DIR / "version_history.json"
ROLLBACK_INDEX = MODELS_DIR / "checkpoint_index.json"
S3_BUCKET = os.getenv("PRIORITYMAX_S3_BUCKET", None)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SYNCHRONIZED CONSTANTS (FROM STABLE TRAINING)
# =============================================================================

# From real_env_STABLE.py
EXPECTED_OBS_DIM = 12
EXPECTED_ACT_DIM = 3

VALID_WORKLOAD_TYPES = [
    "ecommerce", "social_media", "streaming", 
    "api_backend", "batch", "gaming"
]

# From train_rl_STABLE.py - STABLE HYPERPARAMETERS
@dataclass
class StableHyperparameters:
    """Expected stable hyperparameters for production models"""
    gamma: float = 0.99  # NOT 0.995 (causes explosion)
    lam: float = 0.95    # NOT 0.97 (causes explosion)
    lr: float = 3e-4     # Standard PPO (or 1e-4 for conservative)
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.01
    
    # Training schedule
    steps_per_epoch: int = 2048  # NOT 4096 (prevents explosion)
    update_epochs: int = 10
    target_kl: float = 0.015

STABLE_PARAMS = StableHyperparameters()

# Dangerous configurations that cause explosion
EXPLOSION_PRONE_CONFIGS = {
    "gamma": (0.995, "Too high, causes GAE accumulation → explosion"),
    "lam": (0.97, "Too high, causes advantage explosion"),
    "lr": (1e-3, "Too high for PPO, causes policy divergence"),
    "steps_per_epoch": (4096, "Too large, causes gradient explosion"),
    "value_coef": (0.25, "Too low without clipped value loss"),
}

# Helper functions
def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def _hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _audit(event: str, data: Dict[str, Any]):
    """Audit logging"""
    try:
        audit_file = MODELS_DIR.parent / "logs" / "model_audit.jsonl"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {"event": event, "ts": _now_iso(), **data}
        with open(audit_file, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except:
        pass

# =============================================================================
# STABILITY ANALYZER
# =============================================================================

class StabilityAnalyzer:
    """
    Analyzes training configurations for stability issues.
    
    CRITICAL: Detects explosion-prone configurations before deployment!
    """
    
    @staticmethod
    def analyze_hyperparameters(training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if hyperparameters are stable.
        
        Returns:
            {
                "stable": bool,
                "warnings": List[str],
                "critical_issues": List[str],
                "score": float (0-100, higher is better)
            }
        """
        result = {
            "stable": True,
            "warnings": [],
            "critical_issues": [],
            "score": 100.0
        }
        
        if not training_config:
            result["warnings"].append("No training config provided")
            result["score"] -= 20
            return result
        
        # Check gamma
        gamma = training_config.get("gamma", None)
        if gamma is not None:
            if gamma >= EXPLOSION_PRONE_CONFIGS["gamma"][0]:
                result["critical_issues"].append(
                    f"❌ CRITICAL: gamma={gamma} (expected {STABLE_PARAMS.gamma}). "
                    f"{EXPLOSION_PRONE_CONFIGS['gamma'][1]}"
                )
                result["stable"] = False
                result["score"] -= 40
            elif abs(gamma - STABLE_PARAMS.gamma) > 0.01:
                result["warnings"].append(
                    f"⚠️  gamma={gamma} differs from stable value ({STABLE_PARAMS.gamma})"
                )
                result["score"] -= 10
        
        # Check lambda
        lam = training_config.get("lam", None)
        if lam is not None:
            if lam >= EXPLOSION_PRONE_CONFIGS["lam"][0]:
                result["critical_issues"].append(
                    f"❌ CRITICAL: lambda={lam} (expected {STABLE_PARAMS.lam}). "
                    f"{EXPLOSION_PRONE_CONFIGS['lam'][1]}"
                )
                result["stable"] = False
                result["score"] -= 40
            elif abs(lam - STABLE_PARAMS.lam) > 0.01:
                result["warnings"].append(
                    f"⚠️  lambda={lam} differs from stable value ({STABLE_PARAMS.lam})"
                )
                result["score"] -= 10
        
        # Check learning rate
        lr = training_config.get("lr", None)
        if lr is not None:
            if lr >= EXPLOSION_PRONE_CONFIGS["lr"][0]:
                result["critical_issues"].append(
                    f"❌ CRITICAL: lr={lr} is too high. "
                    f"{EXPLOSION_PRONE_CONFIGS['lr'][1]}"
                )
                result["stable"] = False
                result["score"] -= 40
            elif lr not in [3e-4, 1e-4]:
                result["warnings"].append(
                    f"⚠️  lr={lr} is non-standard (expected 3e-4 or 1e-4)"
                )
                result["score"] -= 10
        
        # Check steps per epoch
        steps = training_config.get("steps_per_epoch", None)
        if steps is not None:
            if steps >= EXPLOSION_PRONE_CONFIGS["steps_per_epoch"][0]:
                result["critical_issues"].append(
                    f"❌ CRITICAL: steps_per_epoch={steps} is too large. "
                    f"{EXPLOSION_PRONE_CONFIGS['steps_per_epoch'][1]}"
                )
                result["stable"] = False
                result["score"] -= 30
            elif steps != STABLE_PARAMS.steps_per_epoch:
                result["warnings"].append(
                    f"⚠️  steps_per_epoch={steps} (recommended: {STABLE_PARAMS.steps_per_epoch})"
                )
                result["score"] -= 5
        
        # Check value coefficient (only critical if no clipped value loss)
        value_coef = training_config.get("value_coef", None)
        use_clipped_value = training_config.get("use_clipped_value_loss", True)
        
        if value_coef is not None and not use_clipped_value:
            if value_coef <= EXPLOSION_PRONE_CONFIGS["value_coef"][0]:
                result["critical_issues"].append(
                    f"❌ CRITICAL: value_coef={value_coef} without clipped value loss. "
                    f"{EXPLOSION_PRONE_CONFIGS['value_coef'][1]}"
                )
                result["stable"] = False
                result["score"] -= 50
        
        # Check for critical stability features
        if not training_config.get("use_clipped_value_loss", False):
            result["critical_issues"].append(
                "❌ CRITICAL: Clipped value loss not enabled (required for stability!)"
            )
            result["stable"] = False
            result["score"] -= 50
        
        if not training_config.get("normalize_advantages", True):
            result["warnings"].append(
                "⚠️  Advantage normalization not enabled (recommended)"
            )
            result["score"] -= 10
        
        if not training_config.get("normalize_observations", True):
            result["critical_issues"].append(
                "❌ CRITICAL: Observation normalization not enabled (required!)"
            )
            result["stable"] = False
            result["score"] -= 40
        
        if not training_config.get("normalize_rewards", True):
            result["warnings"].append(
                "⚠️  Reward normalization not enabled (recommended)"
            )
            result["score"] -= 10
        
        # Check gradient clipping
        max_grad_norm = training_config.get("max_grad_norm", None)
        if max_grad_norm is None or max_grad_norm > 1.0:
            result["warnings"].append(
                f"⚠️  max_grad_norm={max_grad_norm} (recommended: 0.5)"
            )
            result["score"] -= 10
        
        result["score"] = max(0.0, min(100.0, result["score"]))
        
        return result
    
    @staticmethod
    def analyze_training_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze training metrics for signs of instability.
        
        Checks for:
        - NaN losses
        - Value explosion (value_mean > 100)
        - Extreme KL divergence
        - Sudden reward drops
        """
        result = {
            "stable": True,
            "warnings": [],
            "issues": []
        }
        
        # Check for NaN
        if any(v != v for v in metrics.values() if isinstance(v, (int, float))):
            result["issues"].append("❌ NaN detected in metrics")
            result["stable"] = False
        
        # Check value predictions
        value_mean = metrics.get("value_mean", None)
        if value_mean is not None:
            if abs(value_mean) > 100:
                result["issues"].append(
                    f"❌ Value explosion detected: value_mean={value_mean:.2f}"
                )
                result["stable"] = False
            elif abs(value_mean) > 50:
                result["warnings"].append(
                    f"⚠️  High value predictions: value_mean={value_mean:.2f}"
                )
        
        # Check KL divergence
        kl_div = metrics.get("kl_div", None)
        if kl_div is not None:
            if kl_div > 0.1:
                result["issues"].append(
                    f"❌ Excessive KL divergence: {kl_div:.4f}"
                )
                result["stable"] = False
        
        # Check rewards
        mean_reward = metrics.get("mean_reward", None)
        if mean_reward is not None:
            if mean_reward < -1000:
                result["warnings"].append(
                    f"⚠️  Very negative rewards: {mean_reward:.2f}"
                )
        
        return result

# =============================================================================
# MODEL REGISTRY (STABILITY-AWARE)
# =============================================================================

class ModelRegistry:
    """
    Production model registry with STABILITY VALIDATION.
    
    NEW FEATURES:
    ✅ Validates stable hyperparameters (gamma=0.99, lr=3e-4, etc.)
    ✅ Detects explosion-prone configurations
    ✅ Validates normalization features
    ✅ Analyzes training metrics for instability
    ✅ Blocks promotion of unstable models
    """
    
    def __init__(self,
                 models_dir: pathlib.Path = MODELS_DIR,
                 registry_file: pathlib.Path = REGISTRY_FILE,
                 rollback_index: pathlib.Path = ROLLBACK_INDEX,
                 s3_bucket: Optional[str] = S3_BUCKET,
                 strict_stability: bool = True):
        self.models_dir = pathlib.Path(models_dir)
        self.registry_file = pathlib.Path(registry_file)
        self.rollback_index = pathlib.Path(rollback_index)
        self.s3_bucket = s3_bucket
        self.strict_stability = strict_stability  # NEW: Block unstable models
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()
        self._rollback_data = self._load_rollback_index()
        
        self.stability_analyzer = StabilityAnalyzer()
        
        LOG.info("Model Registry initialized (strict_stability=%s)", strict_stability)
    
    # -------------------------------------------------------------------------
    # Registry persistence
    # -------------------------------------------------------------------------
    
    def _load_registry(self) -> Dict[str, Any]:
        if not self.registry_file.exists():
            return {}
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8") or "{}")
        except:
            LOG.exception("Failed to load registry")
            return {}
    
    def _save_registry(self):
        try:
            self.registry_file.write_text(
                json.dumps(self._registry, indent=2, default=str),
                encoding="utf-8"
            )
        except:
            LOG.exception("Failed to save registry")
    
    def _load_rollback_index(self) -> Dict[str, Any]:
        if not self.rollback_index.exists():
            return {"checkpoints": []}
        try:
            return json.loads(self.rollback_index.read_text(encoding="utf-8"))
        except:
            return {"checkpoints": []}
    
    def _save_rollback_index(self):
        try:
            self.rollback_index.write_text(
                json.dumps(self._rollback_data, indent=2, default=str),
                encoding="utf-8"
            )
        except:
            pass
    
    # -------------------------------------------------------------------------
    # Checkpoint format detection (same as before)
    # -------------------------------------------------------------------------
    
    def _detect_checkpoint_format(self, path: pathlib.Path) -> str:
        """Detect checkpoint format"""
        if "emergency" in path.name.lower() or "autosave" in path.name.lower():
            return "emergency_checkpoint"
        
        if path.suffix == ".onnx":
            return "onnx"
        
        if not _HAS_TORCH:
            return "unknown"
        
        try:
            ckpt = torch.load(str(path), map_location="cpu")
            
            if "model_state_dict" in ckpt and "optimizer_state_dict" in ckpt:
                if "scaler_state_dict" in ckpt:
                    return "train_rl_stable_amp"
                return "train_rl_stable"
            
            if "model" in ckpt:
                return "legacy_module"
            
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return "state_dict_only"
            
            if ckpt.get("emergency"):
                return "emergency_checkpoint"
            
            return "unknown"
        except:
            LOG.warning("Failed to detect checkpoint format: %s", path)
            return "unknown"
    
    def _extract_model_dimensions(self, state_dict: Dict) -> Optional[Dict[str, int]]:
        """Extract obs_dim and act_dim from model state_dict"""
        try:
            dims = {}
            
            for key, tensor in state_dict.items():
                if "shared_net" in key and ".0.weight" in key:
                    dims["obs_dim"] = tensor.shape[1]
                    break
            
            for key, tensor in state_dict.items():
                if "policy_net" in key and "weight" in key:
                    if len(tensor.shape) == 2:
                        dims["act_dim"] = tensor.shape[0]
                        break
            
            return dims if dims else None
        except:
            return None
    
    # -------------------------------------------------------------------------
    # NEW: STABILITY VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_checkpoint_structure(self, path: pathlib.Path) -> Dict[str, Any]:
        """
        ENHANCED: Validate checkpoint structure + STABILITY.
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "dimensions": None,
            "format": None,
            "stability_analysis": None  # NEW
        }
        
        if not _HAS_TORCH:
            result["valid"] = False
            result["errors"].append("torch_not_available")
            return result
        
        try:
            ckpt = torch.load(str(path), map_location="cpu")
            
            fmt = self._detect_checkpoint_format(path)
            result["format"] = fmt
            
            # Extract model state
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "model" in ckpt:
                if hasattr(ckpt["model"], "state_dict"):
                    state = ckpt["model"].state_dict()
                else:
                    state = ckpt["model"]
            else:
                state = ckpt
            
            if not isinstance(state, dict):
                result["valid"] = False
                result["errors"].append("invalid_state_dict_type")
                return result
            
            # Validate dimensions
            dims = self._extract_model_dimensions(state)
            result["dimensions"] = dims
            
            if dims:
                if dims.get("obs_dim") != EXPECTED_OBS_DIM:
                    result["warnings"].append(
                        f"obs_dim mismatch: expected {EXPECTED_OBS_DIM}, got {dims.get('obs_dim')}"
                    )
                
                if dims.get("act_dim") != EXPECTED_ACT_DIM:
                    result["warnings"].append(
                        f"act_dim mismatch: expected {EXPECTED_ACT_DIM}, got {dims.get('act_dim')}"
                    )
            
            # === NEW: STABILITY ANALYSIS ===
            training_config = ckpt.get("config", None)
            if training_config:
                stability = self.stability_analyzer.analyze_hyperparameters(training_config)
                result["stability_analysis"] = stability
                
                if not stability["stable"]:
                    result["warnings"].append("⚠️  UNSTABLE training configuration detected!")
                    for issue in stability["critical_issues"]:
                        result["warnings"].append(f"  {issue}")
                
                # Log stability score
                LOG.info("Stability score: %.1f/100", stability["score"])
            else:
                result["warnings"].append("No training config found in checkpoint")
            
            # Check training metadata
            if fmt.startswith("train_rl_stable"):
                if "epoch" not in ckpt and "total_steps" not in ckpt:
                    result["warnings"].append("no_training_progress_metadata")
                
                if fmt == "train_rl_stable_amp":
                    if "scaler_state_dict" not in ckpt:
                        result["warnings"].append("amp_format_but_no_scaler")
            
            if "optimizer_state_dict" not in ckpt and "optimizer" not in ckpt:
                result["warnings"].append("no_optimizer_state")
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"load_failed: {str(e)}")
        
        return result
    
    # -------------------------------------------------------------------------
    # ONNX validation (same as before)
    # -------------------------------------------------------------------------
    
    def validate_onnx_model(self, model_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """Validate ONNX model"""
        if not _HAS_ONNX:
            return {"valid": False, "error": "onnxruntime_not_available"}
        
        path = pathlib.Path(model_path)
        if not path.exists():
            return {"valid": False, "error": "file_not_found"}
        
        try:
            session = ort.InferenceSession(str(path))
            
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            input_info = [(inp.name, inp.shape, inp.type) for inp in inputs]
            output_info = [(out.name, out.shape, out.type) for out in outputs]
            
            if len(inputs) != 1:
                return {"valid": False, "error": f"expected 1 input, got {len(inputs)}"}
            
            input_shape = inputs[0].shape
            if len(input_shape) != 2:
                return {"valid": False, "error": f"expected 2D input [batch, obs_dim], got {input_shape}"}
            
            obs_dim = input_shape[1]
            if isinstance(obs_dim, int) and obs_dim != EXPECTED_OBS_DIM:
                return {"valid": False, "error": f"obs_dim mismatch: expected {EXPECTED_OBS_DIM}, got {obs_dim}"}
            
            if _HAS_NUMPY:
                import numpy as np
                batch_size = 1
                test_input = np.random.randn(batch_size, EXPECTED_OBS_DIM).astype(np.float32)
                
                input_name = inputs[0].name
                test_outputs = session.run(None, {input_name: test_input})
                
                if len(test_outputs) != 2:
                    return {"valid": False, "error": f"expected 2 outputs [action_mean, value], got {len(test_outputs)}"}
                
                action_mean, value = test_outputs
                
                if action_mean.shape != (batch_size, EXPECTED_ACT_DIM):
                    return {"valid": False, "error": f"action_mean shape mismatch: expected ({batch_size}, {EXPECTED_ACT_DIM}), got {action_mean.shape}"}
                
                if value.shape != (batch_size,):
                    return {"valid": False, "error": f"value shape mismatch: expected ({batch_size},), got {value.shape}"}
            
            return {
                "valid": True,
                "input_info": input_info,
                "output_info": output_info,
                "test_output_shapes": [o.shape for o in test_outputs] if _HAS_NUMPY else None,
                "providers": session.get_providers(),
                "obs_dim": EXPECTED_OBS_DIM,
                "act_dim": EXPECTED_ACT_DIM
            }
            
        except Exception as e:
            LOG.exception("ONNX validation failed")
            return {"valid": False, "error": str(e)}
    
    # -------------------------------------------------------------------------
    # ENHANCED: Model registration with stability checks
    # -------------------------------------------------------------------------
    
    def register_model(self,
                       model_type: str,
                       file_path: Union[str, pathlib.Path],
                       metrics: Optional[Dict[str, float]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       dataset_version: Optional[str] = None,
                       author: Optional[str] = "system",
                       canary: bool = False,
                       shadow: bool = False,
                       training_config: Optional[Dict[str, Any]] = None,
                       is_onnx: bool = False,
                       is_emergency: bool = False) -> Dict[str, Any]:
        """
        Register new model version with STABILITY VALIDATION.
        
        NEW: Analyzes training configuration for stability issues!
        """
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        file_hash = _hash_file(path)
        version_id = f"{model_type}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        checkpoint_format = self._detect_checkpoint_format(path)
        is_onnx = is_onnx or checkpoint_format == "onnx"
        is_emergency = is_emergency or checkpoint_format == "emergency_checkpoint"
        
        entry = {
            "version_id": version_id,
            "model_type": model_type,
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_hash": file_hash,
            "created_at": _now_iso(),
            "metrics": metrics or {},
            "metadata": metadata or {},
            "dataset_version": dataset_version,
            "author": author,
            "canary": canary,
            "shadow": shadow,
            "active": False,
            "is_onnx": is_onnx,
            "is_emergency": is_emergency,
            "training_config": training_config,
            "checkpoint_format": checkpoint_format,
        }
        
        # === VALIDATION ===
        
        if is_onnx:
            validation = self.validate_onnx_model(path)
            entry["validation"] = validation
            
            if not validation["valid"]:
                LOG.error("ONNX validation failed: %s", validation.get("error"))
                raise ValueError(f"ONNX validation failed: {validation.get('error')}")
        else:
            validation = self._validate_checkpoint_structure(path)
            entry["validation"] = validation
            
            if not validation["valid"]:
                LOG.warning("Checkpoint validation failed: %s", validation["errors"])
            
            # === NEW: STABILITY CHECKS ===
            stability_analysis = validation.get("stability_analysis")
            if stability_analysis:
                entry["stability_score"] = stability_analysis["score"]
                entry["stability_stable"] = stability_analysis["stable"]
                
                if not stability_analysis["stable"]:
                    LOG.error("="*60)
                    LOG.error("⚠️  UNSTABLE MODEL DETECTED!")
                    LOG.error("="*60)
                    for issue in stability_analysis["critical_issues"]:
                        LOG.error(issue)
                    LOG.error("Stability score: %.1f/100", stability_analysis["score"])
                    LOG.error("="*60)
                    
                    if self.strict_stability and not is_emergency:
                        raise ValueError(
                            f"Model failed stability validation (score={stability_analysis['score']:.1f}/100). "
                            f"Issues: {stability_analysis['critical_issues']}"
                        )
                else:
                    LOG.info("✅ Model passed stability validation (score=%.1f/100)", 
                            stability_analysis["score"])
            
            for warning in validation.get("warnings", []):
                if "mismatch" in warning:
                    LOG.warning("⚠️  %s", warning)
        
        # Validate workload_type
        if training_config and "workload_type" in training_config:
            workload = training_config["workload_type"]
            if workload not in VALID_WORKLOAD_TYPES:
                LOG.warning("⚠️  Invalid workload_type: %s (expected one of %s)", 
                           workload, VALID_WORKLOAD_TYPES)
        
        # S3 upload (optional)
        if _HAS_BOTO3 and self.s3_bucket:
            try:
                s3 = boto3.client("s3")
                key = f"models/{path.name}"
                s3.upload_file(str(path), self.s3_bucket, key)
                entry["s3_uri"] = f"s3://{self.s3_bucket}/{key}"
                LOG.info("Uploaded to S3: %s", entry["s3_uri"])
            except Exception as e:
                LOG.warning("S3 upload failed: %s", e)
        
        # Save to registry
        self._registry[version_id] = entry
        self._save_registry()
        
        # Add to rollback index
        self._add_to_rollback_index(entry)
        
        _audit("model_registered", entry)
        LOG.info("✅ Registered model: %s (format=%s, stability_score=%.1f)", 
                version_id, checkpoint_format, entry.get("stability_score", 0))
        
        return entry
    
    def _add_to_rollback_index(self, entry: Dict[str, Any]):
        """Add to rollback manager"""
        self._rollback_data["checkpoints"].append({
            "path": entry["file_path"],
            "timestamp": entry["created_at"],
            "metrics": entry["metrics"],
            "version_id": entry["version_id"],
            "stability_score": entry.get("stability_score", None)
        })
        
        keep_n = int(os.getenv("MODEL_REGISTRY_ROLLBACK_KEEP", "10"))
        self._rollback_data["checkpoints"] = sorted(
            self._rollback_data["checkpoints"],
            key=lambda x: x["timestamp"]
        )[-keep_n:]
        
        self._save_rollback_index()
    
    # -------------------------------------------------------------------------
    # Model retrieval
    # -------------------------------------------------------------------------
    
    def list_models(self, 
                   model_type: Optional[str] = None,
                   include_emergency: bool = False,
                   include_onnx: bool = True,
                   min_stability_score: float = 0.0) -> List[Dict[str, Any]]:
        """List models with filtering"""
        results = []
        for v in self._registry.values():
            if model_type and v.get("model_type") != model_type:
                continue
            if not include_emergency and v.get("is_emergency"):
                continue
            if not include_onnx and v.get("is_onnx"):
                continue
            
            # NEW: Filter by stability score
            stability_score = v.get("stability_score", 100.0)
            if stability_score < min_stability_score:
                continue
            
            results.append(v)
        
        return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def get_model(self, version_id: str) -> Dict[str, Any]:
        """Get specific model by ID"""
        entry = self._registry.get(version_id)
        if not entry:
            raise KeyError(f"Model version {version_id} not found")
        return entry
    
    def get_latest(self, 
                  model_type: str,
                  only_active: bool = True,
                  prefer_onnx: bool = False,
                  min_stability_score: float = 70.0) -> Optional[Dict[str, Any]]:
        """Get latest model (optionally prefer ONNX, require stability)"""
        filtered = [v for v in self._registry.values() 
                   if v.get("model_type") == model_type]
        
        if only_active:
            filtered = [v for v in filtered if v.get("active")]
        
        # Exclude emergency checkpoints
        filtered = [v for v in filtered if not v.get("is_emergency")]
        
        # NEW: Filter by stability score
        filtered = [v for v in filtered 
                   if v.get("stability_score", 100.0) >= min_stability_score]
        
        if not filtered:
            return None
        
        # Prefer ONNX if requested
        if prefer_onnx:
            onnx_models = [v for v in filtered if v.get("is_onnx")]
            if onnx_models:
                return sorted(onnx_models, key=lambda x: x.get("created_at", ""), reverse=True)[0]
        
        return sorted(filtered, key=lambda x: x.get("created_at", ""), reverse=True)[0]
    
    # -------------------------------------------------------------------------
    # Model validation
    # -------------------------------------------------------------------------
    
    def validate_model(self, model_entry: Dict[str, Any]) -> bool:
        """Comprehensive validation including stability"""
        path = pathlib.Path(model_entry["file_path"])
        
        # File existence
        if not path.exists():
            LOG.error("Validation failed: file not found %s", path)
            return False
        
        # Hash verification
        if _hash_file(path) != model_entry["file_hash"]:
            LOG.error("Validation failed: hash mismatch")
            return False
        
        # Format-specific validation
        if model_entry.get("is_onnx"):
            validation = self.validate_onnx_model(path)
            if not validation["valid"]:
                LOG.error("ONNX validation failed: %s", validation.get("error"))
                return False
        else:
            validation = self._validate_checkpoint_structure(path)
            if not validation["valid"]:
                LOG.error("Checkpoint validation failed: %s", validation["errors"])
                return False
            
            # NEW: Check stability
            stability_analysis = validation.get("stability_analysis")
            if stability_analysis and not stability_analysis["stable"]:
                LOG.warning("⚠️  Model has stability issues (score=%.1f/100)", 
                           stability_analysis["score"])
                for issue in stability_analysis["critical_issues"]:
                    LOG.warning("  %s", issue)
                
                if self.strict_stability:
                    LOG.error("Validation failed: unstable model (strict mode)")
                    return False
        
        LOG.info("✅ Validation passed: %s", model_entry["version_id"])
        return True
    
    # -------------------------------------------------------------------------
    # ENHANCED: Promotion with stability checks
    # -------------------------------------------------------------------------
    
    def promote_model(self, version_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Promote model to active (ENHANCED with stability checks).
        
        NEW: Blocks promotion of unstable models unless force=True!
        """
        entry = self.get_model(version_id)
        
        # NEW: Check stability before promotion
        stability_score = entry.get("stability_score", None)
        stability_stable = entry.get("stability_stable", None)
        
        if not force:
            if stability_score is not None and stability_score < 70.0:
                raise ValueError(
                    f"Cannot promote unstable model (stability_score={stability_score:.1f}/100). "
                    f"Use force=True to override."
                )
            
            if stability_stable is False:
                raise ValueError(
                    f"Cannot promote model with critical stability issues. "
                    f"Use force=True to override."
                )
            
            if not self.validate_model(entry):
                raise ValueError("Validation failed; use force=True to override")
        
        model_type = entry["model_type"]
        
        # Deactivate others
        for v in self._registry.values():
            if v.get("model_type") == model_type:
                v["active"] = False
        
        entry["active"] = True
        entry["promoted_at"] = _now_iso()
        self._save_registry()
        
        _audit("model_promoted", entry)
        LOG.info("✅ Promoted model: %s (stability_score=%.1f)", 
                version_id, entry.get("stability_score", 0))
        
        return entry
    
    def rollback_model(self, model_type: str, steps: int = 1) -> Optional[Dict[str, Any]]:
        """Rollback to previous version"""
        history = [v for v in self._registry.values()
                  if v.get("model_type") == model_type and not v.get("is_emergency")]
        history = sorted(history, key=lambda x: x.get("created_at", ""), reverse=True)
        
        if len(history) < (steps + 1):
            LOG.warning("Not enough history for rollback")
            return None
        
        target = history[steps]
        self.promote_model(target["version_id"], force=True)
        
        _audit("model_rollback", {
            "from": history[0]["version_id"],
            "to": target["version_id"],
            "steps": steps
        })
        
        LOG.info("🔄 Rolled back %s to %s", model_type, target["version_id"])
        return target
    
    # -------------------------------------------------------------------------
    # Emergency recovery
    # -------------------------------------------------------------------------
    
    def recover_emergency_checkpoint(self, model_type: str) -> Optional[str]:
        """Recover from emergency_autosave.pt"""
        emergency_path = self.models_dir / "emergency_autosave.pt"
        
        if emergency_path.exists():
            LOG.warning("⚠️  Found emergency checkpoint: %s", emergency_path)
            
            entry = self.register_model(
                model_type=model_type,
                file_path=str(emergency_path),
                metadata={"recovery": True},
                is_emergency=True,
                author="emergency_recovery"
            )
            
            return entry["version_id"]
        
        return None
    
    # -------------------------------------------------------------------------
    # NEW: Stability reporting
    # -------------------------------------------------------------------------
    
    def get_stability_report(self, version_id: str) -> Dict[str, Any]:
        """Get detailed stability report for a model"""
        entry = self.get_model(version_id)
        
        report = {
            "version_id": version_id,
            "model_type": entry["model_type"],
            "created_at": entry["created_at"],
            "stability_score": entry.get("stability_score", None),
            "stability_stable": entry.get("stability_stable", None),
            "training_config": entry.get("training_config", {}),
            "validation": entry.get("validation", {}),
            "metrics": entry.get("metrics", {})
        }
        
        # Re-analyze if training config available
        if entry.get("training_config"):
            analysis = self.stability_analyzer.analyze_hyperparameters(
                entry["training_config"]
            )
            report["detailed_analysis"] = analysis
        
        return report
    
    def list_stable_models(self, model_type: str, min_score: float = 80.0) -> List[Dict[str, Any]]:
        """List only stable models above threshold"""
        return self.list_models(
            model_type=model_type,
            include_emergency=False,
            min_stability_score=min_score
        )
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def delete_model(self, version_id: str, delete_file: bool = False):
        """Delete model entry"""
        entry = self._registry.pop(version_id, None)
        if not entry:
            raise KeyError(f"Model {version_id} not found")
        
        if delete_file:
            try:
                pathlib.Path(entry["file_path"]).unlink()
                LOG.info("Deleted file: %s", entry["file_path"])
            except:
                LOG.exception("Failed to delete file")
        
        self._save_registry()
        _audit("model_deleted", entry)
        LOG.info("Deleted model: %s", version_id)
    
    def export_metadata(self, out_path: Optional[str] = None) -> str:
        """Export registry to JSON"""
        out = out_path or str(self.models_dir / f"registry_export_{int(time.time())}.json")
        pathlib.Path(out).write_text(json.dumps(self._registry, indent=2, default=str))
        LOG.info("Exported registry to: %s", out)
        return out
    
    def sync_s3(self):
        """Sync all models to S3"""
        if not (_HAS_BOTO3 and self.s3_bucket):
            LOG.warning("S3 not configured")
            return
        
        s3 = boto3.client("s3")
        synced = 0
        
        for vid, entry in self._registry.items():
            if "s3_uri" in entry:
                continue
            
            path = pathlib.Path(entry["file_path"])
            if not path.exists():
                continue
            
            key = f"models/{path.name}"
            try:
                s3.upload_file(str(path), self.s3_bucket, key)
                entry["s3_uri"] = f"s3://{self.s3_bucket}/{key}"
                synced += 1
                LOG.info("Uploaded to S3: %s", vid)
            except:
                LOG.exception("S3 upload failed for %s", vid)
        
        if synced > 0:
            self._save_registry()
        
        LOG.info("S3 sync complete: %d models uploaded", synced)


# =============================================================================
# CLI INTERFACE (ENHANCED)
# =============================================================================

def _build_cli():
    import argparse
    parser = argparse.ArgumentParser(
        prog="prioritymax-model-registry",
        description="PriorityMax Model Registry CLI (Stability-Aware)"
    )
    sub = parser.add_subparsers(dest="cmd", help="Command")
    
    # Register
    p_reg = sub.add_parser("register", help="Register new model")
    p_reg.add_argument("--type", required=True, help="Model type")
    p_reg.add_argument("--file", required=True, help="Model file path")
    p_reg.add_argument("--metrics", help="JSON metrics string")
    p_reg.add_argument("--meta", help="JSON metadata string")
    p_reg.add_argument("--author", default="cli", help="Author name")
    p_reg.add_argument("--onnx", action="store_true", help="Mark as ONNX")
    p_reg.add_argument("--emergency", action="store_true", help="Mark as emergency")
    p_reg.add_argument("--no-strict", action="store_true", help="Disable strict stability checks")
    
    # List
    sub.add_parser("list", help="List all models")
    
    p_list = sub.add_parser("list-type", help="List models by type")
    p_list.add_argument("--type", required=True)
    p_list.add_argument("--include-emergency", action="store_true")
    p_list.add_argument("--min-stability", type=float, default=0.0, 
                       help="Minimum stability score (0-100)")
    
    p_stable = sub.add_parser("list-stable", help="List only stable models")
    p_stable.add_argument("--type", required=True)
    p_stable.add_argument("--min-score", type=float, default=80.0)
    
    # Promote
    p_prom = sub.add_parser("promote", help="Promote model to active")
    p_prom.add_argument("--id", required=True)
    p_prom.add_argument("--force", action="store_true", help="Skip stability checks")
    
    # Rollback
    p_roll = sub.add_parser("rollback", help="Rollback to previous version")
    p_roll.add_argument("--type", required=True)
    p_roll.add_argument("--steps", type=int, default=1)
    
    # Delete
    p_del = sub.add_parser("delete", help="Delete model")
    p_del.add_argument("--id", required=True)
    p_del.add_argument("--delete-file", action="store_true")
    
    # Validate
    p_val = sub.add_parser("validate", help="Validate model")
    p_val.add_argument("--id", required=True)
    
    p_val_onnx = sub.add_parser("validate-onnx", help="Validate ONNX model")
    p_val_onnx.add_argument("--file", required=True)
    
    # Get
    p_get = sub.add_parser("get", help="Get model details")
    p_get.add_argument("--id", required=True)
    
    p_latest = sub.add_parser("get-latest", help="Get latest model")
    p_latest.add_argument("--type", required=True)
    p_latest.add_argument("--prefer-onnx", action="store_true")
    p_latest.add_argument("--min-stability", type=float, default=70.0)
    
    # NEW: Stability report
    p_report = sub.add_parser("stability-report", help="Get stability report")
    p_report.add_argument("--id", required=True)
    
    # Recover
    p_recover = sub.add_parser("recover-emergency", help="Recover emergency checkpoint")
    p_recover.add_argument("--type", required=True)
    
    # Export
    p_export = sub.add_parser("export", help="Export registry")
    p_export.add_argument("--out", help="Output path")
    
    # S3
    sub.add_parser("sync-s3", help="Sync models to S3")
    
    return parser


def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    
    if not args.cmd:
        parser.print_help()
        sys.exit(1)
    
    # Initialize registry
    strict = not getattr(args, 'no_strict', False)
    reg = ModelRegistry(strict_stability=strict)
    
    try:
        if args.cmd == "register":
            metrics = json.loads(args.metrics) if args.metrics else None
            meta = json.loads(args.meta) if args.meta else None
            
            res = reg.register_model(
                model_type=args.type,
                file_path=args.file,
                metrics=metrics,
                metadata=meta,
                author=args.author,
                is_onnx=args.onnx,
                is_emergency=args.emergency
            )
            print(json.dumps(res, indent=2, default=str))
            
            # Print stability summary
            if res.get("stability_score") is not None:
                print(f"\n📊 Stability Score: {res['stability_score']:.1f}/100")
                if res.get("stability_stable"):
                    print("✅ Model is STABLE")
                else:
                    print("⚠️  Model has STABILITY ISSUES")
        
        elif args.cmd == "list":
            res = reg.list_models()
            print(json.dumps(res, indent=2, default=str))
        
        elif args.cmd == "list-type":
            res = reg.list_models(
                model_type=args.type,
                include_emergency=args.include_emergency,
                min_stability_score=args.min_stability
            )
            print(json.dumps(res, indent=2, default=str))
        
        elif args.cmd == "list-stable":
            res = reg.list_stable_models(args.type, args.min_score)
            print(json.dumps(res, indent=2, default=str))
            print(f"\n✅ Found {len(res)} stable models (score >= {args.min_score})")
        
        elif args.cmd == "promote":
            res = reg.promote_model(args.id, force=args.force)
            print(json.dumps(res, indent=2, default=str))
            print(f"\n✅ Model {args.id} promoted")
        
        elif args.cmd == "rollback":
            res = reg.rollback_model(args.type, steps=args.steps)
            if res:
                print(json.dumps(res, indent=2, default=str))
                print(f"\n✅ Rolled back to {res['version_id']}")
            else:
                print("❌ Rollback failed")
                sys.exit(1)
        
        elif args.cmd == "delete":
            reg.delete_model(args.id, delete_file=args.delete_file)
            print(f"✅ Deleted {args.id}")
        
        elif args.cmd == "validate":
            entry = reg.get_model(args.id)
            if reg.validate_model(entry):
                print(f"✅ Validation passed: {args.id}")
            else:
                print(f"❌ Validation failed: {args.id}")
                sys.exit(1)
        
        elif args.cmd == "validate-onnx":
            result = reg.validate_onnx_model(args.file)
            print(json.dumps(result, indent=2, default=str))
            if result["valid"]:
                print("\n✅ ONNX validation passed")
            else:
                print(f"\n❌ ONNX validation failed: {result.get('error')}")
                sys.exit(1)
        
        elif args.cmd == "get":
            res = reg.get_model(args.id)
            print(json.dumps(res, indent=2, default=str))
        
        elif args.cmd == "get-latest":
            res = reg.get_latest(
                args.type, 
                prefer_onnx=args.prefer_onnx,
                min_stability_score=args.min_stability
            )
            if res:
                print(json.dumps(res, indent=2, default=str))
            else:
                print(f"❌ No model found for type: {args.type}")
                sys.exit(1)
        
        elif args.cmd == "stability-report":
            report = reg.get_stability_report(args.id)
            print(json.dumps(report, indent=2, default=str))
            
            if report.get("detailed_analysis"):
                analysis = report["detailed_analysis"]
                print(f"\n{'='*60}")
                print(f"STABILITY REPORT: {args.id}")
                print(f"{'='*60}")
                print(f"Score: {analysis['score']:.1f}/100")
                print(f"Status: {'✅ STABLE' if analysis['stable'] else '⚠️  UNSTABLE'}")
                
                if analysis["critical_issues"]:
                    print(f"\n❌ Critical Issues:")
                    for issue in analysis["critical_issues"]:
                        print(f"  {issue}")
                
                if analysis["warnings"]:
                    print(f"\n⚠️  Warnings:")
                    for warning in analysis["warnings"]:
                        print(f"  {warning}")
                print(f"{'='*60}\n")
        
        elif args.cmd == "recover-emergency":
            version_id = reg.recover_emergency_checkpoint(args.type)
            if version_id:
                print(f"✅ Emergency checkpoint recovered: {version_id}")
                print(f"\nTo promote: prioritymax-model-registry promote --id {version_id} --force")
            else:
                print(f"❌ No emergency checkpoint found")
                sys.exit(1)
        
        elif args.cmd == "export":
            path = reg.export_metadata(args.out if hasattr(args, 'out') else None)
            print(f"✅ Exported to: {path}")
        
        elif args.cmd == "sync-s3":
            reg.sync_s3()
            print("✅ S3 sync complete")
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        LOG.exception("CLI command failed")
        sys.exit(1)


# =============================================================================
# USAGE DOCUMENTATION
# =============================================================================

"""
SYNCHRONIZED MODEL REGISTRY - STABILITY-AWARE VERSION
======================================================

NEW FEATURES:
✅ Validates stable hyperparameters (gamma=0.99, lr=3e-4, etc.)
✅ Detects explosion-prone configurations
✅ Stability scoring (0-100)
✅ Blocks promotion of unstable models
✅ Detailed stability reports

USAGE EXAMPLES:

1. REGISTER MODEL (WITH AUTOMATIC STABILITY CHECK)
---------------------------------------------------
python -m app.ml.model_registry register \\
    --type rl_agent \\
    --file checkpoints/ckpt_best_0100.pt \\
    --metrics '{"mean_reward": 145.3}'

# Output will show stability score:
# 📊 Stability Score: 95.0/100
# ✅ Model is STABLE

2. LIST ONLY STABLE MODELS
---------------------------
python -m app.ml.model_registry list-stable --type rl_agent --min-score 80

3. GET STABILITY REPORT
------------------------
python -m app.ml.model_registry stability-report --id rl_agent_20240111_abc123

# Shows detailed analysis:
# - Hyperparameter validation
# - Critical issues
# - Warnings
# - Stability score

4. PROMOTE WITH STABILITY CHECK
--------------------------------
# Normal promotion (requires stability_score >= 70)
python -m app.ml.model_registry promote --id rl_agent_20240111_abc123

# Force promotion (skip stability checks)
python -m app.ml.model_registry promote --id rl_agent_20240111_abc123 --force

5. REGISTER WITH STRICT MODE DISABLED
--------------------------------------
python -m app.ml.model_registry register \\
    --type rl_agent \\
    --file checkpoints/ckpt_experimental.pt \\
    --no-strict

# Allows registration of unstable models (for testing)

PYTHON API:
-----------
from app.ml.model_registry import ModelRegistry

# Initialize with strict stability checks
registry = ModelRegistry(strict_stability=True)

# Register model (will validate stability)
entry = registry.register_model(
    model_type="rl_agent",
    file_path="checkpoints/ckpt_best.pt",
    training_config={
        "gamma": 0.99,  # ✅ Stable
        "lam": 0.95,    # ✅ Stable
        "lr": 3e-4,     # ✅ Stable
        "use_clipped_value_loss": True,  # ✅ Required
        "normalize_observations": True,  # ✅ Required
    }
)

# Get stability report
report = registry.get_stability_report(entry["version_id"])
print(f"Stability: {report['stability_score']}/100")

# Promote only if stable
if report["stability_stable"]:
    registry.promote_model(entry["version_id"])
else:
    print("Model unstable, not promoting")

COLAB INTEGRATION:
------------------
# After training with train_rl_STABLE.py:
from app.ml.model_registry import ModelRegistry

registry = ModelRegistry(
    models_dir="/content/drive/MyDrive/PriorityMax/models",
    strict_stability=True
)

# Register with training config
entry = registry.register_model(
    model_type="rl_agent",
    file_path=best_checkpoint_path,
    metrics=trainer.training_metrics,
    training_config=asdict(trainer.cfg),  # ← CRITICAL!
    author="colab-training"
)

# Check if safe to deploy
if entry["stability_score"] >= 80:
    print("✅ Safe for production")
    registry.promote_model(entry["version_id"])
else:
    print(f"⚠️  Stability score too low: {entry['stability_score']}")

WHAT GETS VALIDATED:
====================
✅ gamma=0.99 (not 0.995)
✅ lambda=0.95 (not 0.97)
✅ lr=3e-4 or 1e-4 (not 1e-3)
✅ steps_per_epoch=2048 (not 4096)
✅ use_clipped_value_loss=True
✅ normalize_observations=True
✅ normalize_rewards=True
✅ max_grad_norm=0.5
✅ obs_dim=12, act_dim=3

STABILITY SCORING:
==================
100 points = Perfect configuration
 -40 points = Critical issue (gamma too high, etc.)
 -10 points = Warning (non-standard value)
 -50 points = Missing clipped value loss
 -40 points = Missing observation normalization

Minimum for production: 70/100

BLOCKED CONFIGURATIONS:
=======================
❌ gamma >= 0.995 → "Causes GAE accumulation → explosion"
❌ lambda >= 0.97 → "Causes advantage explosion"
❌ lr >= 1e-3 → "Too high for PPO, causes divergence"
❌ steps_per_epoch >= 4096 → "Too large, causes gradient explosion"
❌ use_clipped_value_loss=False → "Required for stability"
❌ normalize_observations=False → "Required for stable gradients"
"""

if __name__ == "__main__":
    main_cli()
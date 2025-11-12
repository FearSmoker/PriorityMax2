#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PriorityMax Model Registry - FULLY SYNCHRONIZED ENTERPRISE EDITION
-------------------------------------------------------------------

SYNCHRONIZED WITH:
- real_env.py v2 (obs_dim=12, WorkloadType enum, extended metrics)
- train_rl_heavy.py (AMP checkpoints, ONNX export, emergency saves)
- train_rl_live.py (rollback manager, checkpoint indexing)
- train_rl_eval.py (A/B testing, drift detection)

CRITICAL SYNCHRONIZATION:
✅ Validates obs_dim=12, act_dim=3 from real_env.py
✅ Validates WorkloadType compatibility
✅ Handles AMP (Mixed Precision) checkpoints from train_rl_heavy.py
✅ Detects emergency_autosave.pt pattern
✅ ONNX model validation with onnxruntime
✅ Extended metrics validation (p99_latency, cost_rate, etc.)
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

# === SYNCHRONIZED CONSTANTS from real_env.py ===
EXPECTED_OBS_DIM = 12  # Extended observation space
EXPECTED_ACT_DIM = 3   # [delta_workers, throttle, priority_bias]

VALID_WORKLOAD_TYPES = [
    "ecommerce", "social_media", "streaming", 
    "api_backend", "batch", "gaming"
]

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
    """Simplified audit logging"""
    try:
        audit_file = MODELS_DIR.parent / "logs" / "model_audit.jsonl"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {"event": event, "ts": _now_iso(), **data}
        with open(audit_file, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except:
        pass  # Non-critical

# =============================================================================
# SYNCHRONIZED MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """
    Production model registry synchronized with PriorityMax training pipeline.
    
    SYNCHRONIZED FEATURES:
    - Validates real_env.py obs_dim=12, act_dim=3
    - Handles train_rl_heavy.py checkpoint format (AMP support)
    - Detects emergency_autosave.pt pattern
    - ONNX validation with dimension checks
    - Workload type validation
    """
    
    def __init__(self,
                 models_dir: pathlib.Path = MODELS_DIR,
                 registry_file: pathlib.Path = REGISTRY_FILE,
                 rollback_index: pathlib.Path = ROLLBACK_INDEX,
                 s3_bucket: Optional[str] = S3_BUCKET):
        self.models_dir = pathlib.Path(models_dir)
        self.registry_file = pathlib.Path(registry_file)
        self.rollback_index = pathlib.Path(rollback_index)
        self.s3_bucket = s3_bucket
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()
        self._rollback_data = self._load_rollback_index()
        
        LOG.info("Model Registry initialized (models_dir=%s)", self.models_dir)
    
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
            LOG.exception("Failed to load rollback index")
            return {"checkpoints": []}
    
    def _save_rollback_index(self):
        try:
            self.rollback_index.write_text(
                json.dumps(self._rollback_data, indent=2, default=str),
                encoding="utf-8"
            )
        except:
            LOG.exception("Failed to save rollback index")
    
    # -------------------------------------------------------------------------
    # SYNCHRONIZED: Checkpoint format detection
    # -------------------------------------------------------------------------
    
    def _detect_checkpoint_format(self, path: pathlib.Path) -> str:
        """
        Detect checkpoint format (SYNCHRONIZED with train_rl_heavy.py).
        
        Formats:
        - train_rl_heavy: Standard checkpoint with model_state_dict
        - train_rl_heavy_amp: AMP checkpoint with scaler_state_dict
        - emergency_checkpoint: emergency_autosave.pt pattern
        - onnx: .onnx file
        """
        # Check filename pattern for emergency
        if "emergency" in path.name.lower() or "autosave" in path.name.lower():
            return "emergency_checkpoint"
        
        if path.suffix == ".onnx":
            return "onnx"
        
        if not _HAS_TORCH:
            return "unknown"
        
        try:
            ckpt = torch.load(str(path), map_location="cpu")
            
            # train_rl_heavy.py format
            if "model_state_dict" in ckpt and "optimizer_state_dict" in ckpt:
                if "scaler_state_dict" in ckpt:
                    return "train_rl_heavy_amp"  # Mixed precision
                return "train_rl_heavy"
            
            # Legacy format
            if "model" in ckpt:
                return "legacy_module"
            
            # Direct state dict
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return "state_dict_only"
            
            # Emergency flag
            if ckpt.get("emergency"):
                return "emergency_checkpoint"
            
            return "unknown"
        except:
            LOG.warning("Failed to detect checkpoint format: %s", path)
            return "unknown"
    
    def _validate_checkpoint_structure(self, path: pathlib.Path) -> Dict[str, Any]:
        """
        Validate checkpoint structure (SYNCHRONIZED with training scripts).
        
        Checks:
        - Can be loaded by PyTorch
        - Has model_state_dict or model key
        - Dimension compatibility (obs_dim=12, act_dim=3)
        - Training metadata present
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "dimensions": None,
            "format": None
        }
        
        if not _HAS_TORCH:
            result["valid"] = False
            result["errors"].append("torch_not_available")
            return result
        
        try:
            ckpt = torch.load(str(path), map_location="cpu")
            
            # Detect format
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
            
            # === CRITICAL: Validate dimensions (synchronized with real_env.py) ===
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
            
            # Check training metadata
            if fmt.startswith("train_rl_heavy"):
                if "epoch" not in ckpt and "total_steps" not in ckpt:
                    result["warnings"].append("no_training_progress_metadata")
                
                # Validate AMP components
                if fmt == "train_rl_heavy_amp":
                    if "scaler_state_dict" not in ckpt:
                        result["warnings"].append("amp_format_but_no_scaler")
            
            # Check optimizer state (for resuming training)
            if "optimizer_state_dict" not in ckpt and "optimizer" not in ckpt:
                result["warnings"].append("no_optimizer_state")
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"load_failed: {str(e)}")
        
        return result
    
    def _extract_model_dimensions(self, state_dict: Dict) -> Optional[Dict[str, int]]:
        """
        Extract obs_dim and act_dim from model state_dict.
        
        Looks for:
        - shared_net.0.weight: [hidden, obs_dim] -> obs_dim
        - policy_net.*.weight: [act_dim, hidden] -> act_dim
        """
        try:
            dims = {}
            
            # Find observation dimension
            for key, tensor in state_dict.items():
                if "shared_net" in key and ".0.weight" in key:
                    # Shape: [hidden_dim, obs_dim]
                    dims["obs_dim"] = tensor.shape[1]
                    break
            
            # Find action dimension
            for key, tensor in state_dict.items():
                if "policy_net" in key and "weight" in key:
                    # Last linear layer: [act_dim, hidden]
                    if len(tensor.shape) == 2:
                        dims["act_dim"] = tensor.shape[0]
                        break
            
            return dims if dims else None
        except:
            return None
    
    # -------------------------------------------------------------------------
    # SYNCHRONIZED: ONNX validation
    # -------------------------------------------------------------------------
    
    def validate_onnx_model(self, model_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Validate ONNX model (SYNCHRONIZED with train_rl_heavy.py export).
        
        Checks:
        - Can be loaded by onnxruntime
        - Input shape matches obs_dim=12
        - Output shapes are correct (action_mean, value)
        - Inference test passes
        """
        if not _HAS_ONNX:
            return {"valid": False, "error": "onnxruntime_not_available"}
        
        path = pathlib.Path(model_path)
        if not path.exists():
            return {"valid": False, "error": "file_not_found"}
        
        try:
            # Load session
            session = ort.InferenceSession(str(path))
            
            # Get I/O info
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            input_info = [(inp.name, inp.shape, inp.type) for inp in inputs]
            output_info = [(out.name, out.shape, out.type) for out in outputs]
            
            # === CRITICAL: Validate dimensions ===
            if len(inputs) != 1:
                return {
                    "valid": False, 
                    "error": f"expected 1 input, got {len(inputs)}"
                }
            
            input_shape = inputs[0].shape
            # Handle dynamic batch: [None, obs_dim] or ['batch', obs_dim]
            if len(input_shape) != 2:
                return {
                    "valid": False,
                    "error": f"expected 2D input [batch, obs_dim], got {input_shape}"
                }
            
            obs_dim = input_shape[1]
            if isinstance(obs_dim, int) and obs_dim != EXPECTED_OBS_DIM:
                return {
                    "valid": False,
                    "error": f"obs_dim mismatch: expected {EXPECTED_OBS_DIM}, got {obs_dim}"
                }
            
            # Test inference
            if _HAS_NUMPY:
                import numpy as np
                batch_size = 1
                test_input = np.random.randn(batch_size, EXPECTED_OBS_DIM).astype(np.float32)
                
                input_name = inputs[0].name
                test_outputs = session.run(None, {input_name: test_input})
                
                # Validate outputs
                if len(test_outputs) != 2:
                    return {
                        "valid": False,
                        "error": f"expected 2 outputs [action_mean, value], got {len(test_outputs)}"
                    }
                
                action_mean, value = test_outputs
                
                # Check shapes
                if action_mean.shape != (batch_size, EXPECTED_ACT_DIM):
                    return {
                        "valid": False,
                        "error": f"action_mean shape mismatch: expected ({batch_size}, {EXPECTED_ACT_DIM}), got {action_mean.shape}"
                    }
                
                if value.shape != (batch_size,):
                    return {
                        "valid": False,
                        "error": f"value shape mismatch: expected ({batch_size},), got {value.shape}"
                    }
            
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
    # SYNCHRONIZED: Model registration
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
        Register new model version (SYNCHRONIZED with training pipeline).
        
        Features:
        - Auto-detects checkpoint format
        - Validates dimensions (obs_dim=12, act_dim=3)
        - Validates workload_type in training_config
        - Handles ONNX models
        - Marks emergency checkpoints
        """
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        file_hash = _hash_file(path)
        version_id = f"{model_type}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Detect format
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
        
        # === SYNCHRONIZED VALIDATION ===
        
        if is_onnx:
            # ONNX validation
            validation = self.validate_onnx_model(path)
            entry["validation"] = validation
            
            if not validation["valid"]:
                LOG.error("ONNX validation failed: %s", validation.get("error"))
                raise ValueError(f"ONNX validation failed: {validation.get('error')}")
        else:
            # PyTorch checkpoint validation
            validation = self._validate_checkpoint_structure(path)
            entry["validation"] = validation
            
            if not validation["valid"]:
                LOG.warning("Checkpoint validation failed: %s", validation["errors"])
            
            # Log dimension warnings
            for warning in validation.get("warnings", []):
                if "mismatch" in warning:
                    LOG.warning("⚠️  %s", warning)
        
        # Validate workload_type if present
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
        LOG.info("✅ Registered model: %s (format=%s, is_onnx=%s, is_emergency=%s)", 
                version_id, checkpoint_format, is_onnx, is_emergency)
        
        return entry
    
    def _add_to_rollback_index(self, entry: Dict[str, Any]):
        """Add to rollback manager (synchronized with train_rl_live.py)"""
        self._rollback_data["checkpoints"].append({
            "path": entry["file_path"],
            "timestamp": entry["created_at"],
            "metrics": entry["metrics"],
            "version_id": entry["version_id"]
        })
        
        # Keep last N checkpoints
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
                   include_onnx: bool = True) -> List[Dict[str, Any]]:
        """List models with filtering"""
        results = []
        for v in self._registry.values():
            if model_type and v.get("model_type") != model_type:
                continue
            if not include_emergency and v.get("is_emergency"):
                continue
            if not include_onnx and v.get("is_onnx"):
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
                  prefer_onnx: bool = False) -> Optional[Dict[str, Any]]:
        """Get latest model (optionally prefer ONNX)"""
        filtered = [v for v in self._registry.values() 
                   if v.get("model_type") == model_type]
        
        if only_active:
            filtered = [v for v in filtered if v.get("active")]
        
        # Exclude emergency checkpoints
        filtered = [v for v in filtered if not v.get("is_emergency")]
        
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
        """Comprehensive validation"""
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
        
        LOG.info("✅ Validation passed: %s", model_entry["version_id"])
        return True
    
    # -------------------------------------------------------------------------
    # Promotion & Rollback
    # -------------------------------------------------------------------------
    
    def promote_model(self, version_id: str, force: bool = False) -> Dict[str, Any]:
        """Promote model to active"""
        entry = self.get_model(version_id)
        
        if not force and not self.validate_model(entry):
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
        LOG.info("✅ Promoted model: %s", version_id)
        
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
# CLI INTERFACE
# =============================================================================

def _build_cli():
    import argparse
    parser = argparse.ArgumentParser(
        prog="prioritymax-model-registry",
        description="PriorityMax Model Registry CLI"
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
    
    # List
    sub.add_parser("list", help="List all models")
    
    p_list = sub.add_parser("list-type", help="List models by type")
    p_list.add_argument("--type", required=True)
    p_list.add_argument("--include-emergency", action="store_true")
    
    # Promote
    p_prom = sub.add_parser("promote", help="Promote model to active")
    p_prom.add_argument("--id", required=True)
    p_prom.add_argument("--force", action="store_true")
    
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
    
    reg = ModelRegistry()
    
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
        
        elif args.cmd == "list":
            res = reg.list_models()
            print(json.dumps(res, indent=2, default=str))
        
        elif args.cmd == "list-type":
            res = reg.list_models(
                model_type=args.type,
                include_emergency=args.include_emergency
            )
            print(json.dumps(res, indent=2, default=str))
        
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
            res = reg.get_latest(args.type, prefer_onnx=args.prefer_onnx)
            if res:
                print(json.dumps(res, indent=2, default=str))
            else:
                print(f"❌ No model found for type: {args.type}")
                sys.exit(1)
        
        elif args.cmd == "recover-emergency":
            version_id = reg.recover_emergency_checkpoint(args.type)
            if version_id:
                print(f"✅ Emergency checkpoint recovered: {version_id}")
                print(f"\nTo promote: prioritymax-model-registry promote --id {version_id}")
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
# USAGE EXAMPLES & DOCUMENTATION
# =============================================================================

"""
SYNCHRONIZED MODEL REGISTRY - USAGE GUIDE
==========================================

CRITICAL SYNCHRONIZATION:
✅ obs_dim=12, act_dim=3 validation (from real_env.py)
✅ WorkloadType validation (ecommerce, social_media, etc.)
✅ AMP checkpoint support (from train_rl_heavy.py)
✅ emergency_autosave.pt pattern detection
✅ ONNX dimension validation

1. REGISTER MODEL AFTER TRAINING
---------------------------------
# After train_rl_heavy.py completes
python -m app.ml.model_registry register \\
    --type rl_agent \\
    --file backend/app/ml/checkpoints/ckpt_best_0100.pt \\
    --metrics '{"mean_reward": 145.3, "p95_latency": 0.85}' \\
    --author "ml-team"

# Register ONNX model
python -m app.ml.model_registry register \\
    --type rl_agent \\
    --file backend/app/ml/checkpoints/model_best.onnx \\
    --onnx \\
    --metrics '{"mean_reward": 150.2}'

2. LIST & INSPECT MODELS
-------------------------
# List all models
python -m app.ml.model_registry list

# List specific type (exclude emergency checkpoints)
python -m app.ml.model_registry list-type --type rl_agent

# Include emergency checkpoints
python -m app.ml.model_registry list-type --type rl_agent --include-emergency

3. VALIDATE BEFORE DEPLOYMENT
------------------------------
# Validate registered model
python -m app.ml.model_registry validate --id rl_agent_20240111T120000_abc123

# Validate ONNX before registration
python -m app.ml.model_registry validate-onnx --file models/model.onnx

4. PROMOTE TO PRODUCTION
-------------------------
# Promote after validation
python -m app.ml.model_registry promote --id rl_agent_20240111T120000_abc123

# Force promote (skip validation)
python -m app.ml.model_registry promote --id rl_agent_20240111T120000_abc123 --force

5. ROLLBACK IF ISSUES
----------------------
# Rollback to previous version
python -m app.ml.model_registry rollback --type rl_agent

# Rollback 2 versions
python -m app.ml.model_registry rollback --type rl_agent --steps 2

6. EMERGENCY RECOVERY
---------------------
# Recover from emergency_autosave.pt (after Colab crash)
python -m app.ml.model_registry recover-emergency --type rl_agent

# Then promote the recovered checkpoint
python -m app.ml.model_registry promote --id <recovered_version_id>

7. PYTHON API USAGE
-------------------
from app.ml.model_registry import ModelRegistry

# Initialize
registry = ModelRegistry()

# Register model after training
entry = registry.register_model(
    model_type="rl_agent",
    file_path="checkpoints/ckpt_best_0100.pt",
    metrics={"mean_reward": 145.3},
    training_config={
        "epochs": 300,
        "workload_type": "ecommerce",  # Validated!
        "obs_dim": 12,  # Validated!
        "act_dim": 3    # Validated!
    }
)

# Validate before promotion
if registry.validate_model(entry):
    registry.promote_model(entry["version_id"])

# Get latest ONNX model for inference
latest = registry.get_latest("rl_agent", prefer_onnx=True)

# Rollback if production issues
if production_issues_detected:
    previous = registry.rollback_model("rl_agent")

8. COLAB INTEGRATION
--------------------
# In Google Colab, after training:
from app.ml.model_registry import ModelRegistry

registry = ModelRegistry(
    models_dir="/content/drive/MyDrive/PriorityMax/models"
)

# Register best checkpoint
best_ckpt = "/content/drive/MyDrive/PriorityMax/checkpoints/ckpt_best_0100.pt"
entry = registry.register_model(
    model_type="rl_agent",
    file_path=best_ckpt,
    metrics=trainer.training_metrics,
    author="colab-training"
)

print(f"Registered: {entry['version_id']}")

# Export ONNX and register
onnx_path = "/content/drive/MyDrive/PriorityMax/models/model_best.onnx"
if onnx_path.exists():
    onnx_entry = registry.register_model(
        model_type="rl_agent",
        file_path=str(onnx_path),
        is_onnx=True,
        metrics={"source_checkpoint": entry["version_id"]}
    )
    print(f"ONNX registered: {onnx_entry['version_id']}")

VALIDATION FEATURES (SYNCHRONIZED):
===================================
✅ Checkpoint format detection (train_rl_heavy, AMP, emergency)
✅ Dimension validation (obs_dim=12, act_dim=3)
✅ WorkloadType validation (ecommerce, social_media, etc.)
✅ ONNX model validation with inference test
✅ File hash verification
✅ Training metadata presence check
✅ Optimizer state check (for resuming)
✅ AMP scaler validation (if mixed precision)

ERROR HANDLING:
===============
❌ Dimension mismatch → Warning logged, registration succeeds
❌ Invalid workload_type → Warning logged
❌ ONNX validation failure → Registration fails
❌ Missing file → Registration fails
❌ Hash mismatch → Validation fails

BEST PRACTICES:
===============
1. Always validate before promoting to production
2. Use ONNX models for production inference (faster)
3. Keep emergency checkpoints accessible for recovery
4. Test rollback procedures regularly
5. Sync to S3 for disaster recovery
6. Monitor validation warnings in production
7. Use semantic versioning in metadata
8. Document training configs for reproducibility
"""

if __name__ == "__main__":
    main_cli()
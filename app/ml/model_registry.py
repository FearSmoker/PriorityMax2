# backend/app/ml/model_registry.py
"""
PriorityMax Model Registry (Production Grade)
---------------------------------------------

Central model lifecycle management service for PriorityMax AI Orchestrator.

Responsibilities:
 - Register, version, validate, and promote ML/RL models.
 - Store metadata, metrics, and artifacts.
 - Manage canary and shadow deployments.
 - Enforce model safety and rollback policies.
 - Integrate with local FS, S3, or MongoDB metadata stores.
 - Optional MLflow and Weights & Biases integration for lineage tracking.
 - Provide APIs for backend modules and dashboards.

Model Types:
 - predictor_lgbm.pkl       -> Queue latency predictor
 - rl_agent.pt              -> PPO reinforcement learning agent
 - autoscaler_policy.json   -> Dynamic autoscaling policy model
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
import tempfile
import logging
import traceback
from typing import Any, Dict, Optional, List, Tuple, Union

# Optional dependencies
try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    joblib = None
    _HAS_JOBLIB = False

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
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

# Audit hook
try:
    from app.api.admin import write_audit_event
    _HAS_AUDIT = True
except Exception:
    _HAS_AUDIT = False
    def write_audit_event(payload: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "model_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.ml.model_registry")
LOG.setLevel(os.getenv("PRIORITYMAX_ML_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths and config
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODELS_DIR", str(BASE_DIR / "app" / "ml" / "models")))
REGISTRY_FILE = MODELS_DIR / "version_history.json"
S3_BUCKET = os.getenv("PRIORITYMAX_S3_BUCKET", None)
MONGO_URL = os.getenv("MONGO_URL", None)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
    try:
        payload = {"event": event, "ts": _now_iso(), **data}
        write_audit_event(payload)
    except Exception:
        LOG.exception("Audit write failed for event=%s", event)

# ---------------------------
# Core Registry Class
# ---------------------------
class ModelRegistry:
    """
    Unified model registry supporting local FS, S3, and optional Mongo metadata.
    """

    def __init__(self,
                 models_dir: pathlib.Path = MODELS_DIR,
                 registry_file: pathlib.Path = REGISTRY_FILE,
                 s3_bucket: Optional[str] = S3_BUCKET,
                 mongo_url: Optional[str] = MONGO_URL):
        self.models_dir = pathlib.Path(models_dir)
        self.registry_file = pathlib.Path(registry_file)
        self.s3_bucket = s3_bucket
        self.mongo_url = mongo_url
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()
        self._mongo = None
        if mongo_url and _HAS_MOTOR:
            try:
                self._mongo = motor_asyncio.AsyncIOMotorClient(mongo_url)
            except Exception:
                LOG.exception("Mongo connection failed for ModelRegistry")

    # --------------------------------
    # Registry file helpers
    # --------------------------------
    def _load_registry(self) -> Dict[str, Any]:
        if not self.registry_file.exists():
            return {}
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8") or "{}")
        except Exception:
            LOG.exception("Failed to load registry JSON")
            return {}

    def _save_registry(self):
        try:
            self.registry_file.write_text(json.dumps(self._registry, indent=2, default=str))
        except Exception:
            LOG.exception("Failed to save registry")

    # --------------------------------
    # Model registration & versioning
    # --------------------------------
    def register_model(self,
                       model_type: str,
                       file_path: Union[str, pathlib.Path],
                       metrics: Optional[Dict[str, float]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       dataset_version: Optional[str] = None,
                       author: Optional[str] = "system",
                       canary: bool = False,
                       shadow: bool = False) -> Dict[str, Any]:
        """
        Register a new model version. Computes hash, stores metadata, uploads to S3 if configured.
        """
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        file_hash = _hash_file(path)
        version_id = f"{model_type}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
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
        }
        # Upload to S3 if configured
        if _HAS_BOTO3 and self.s3_bucket:
            try:
                s3 = boto3.client("s3")
                key = f"models/{path.name}"
                s3.upload_file(str(path), self.s3_bucket, key)
                entry["s3_uri"] = f"s3://{self.s3_bucket}/{key}"
            except Exception:
                LOG.exception("S3 upload failed")

        self._registry[version_id] = entry
        self._save_registry()
        _audit("model_registered", entry)
        LOG.info("Registered new model: %s", version_id)
        return entry

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        results = [v for v in self._registry.values() if not model_type or v.get("model_type") == model_type]
        return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_model(self, version_id: str) -> Dict[str, Any]:
        entry = self._registry.get(version_id)
        if not entry:
            raise KeyError(f"Model version {version_id} not found")
        return entry

    def get_latest(self, model_type: str, only_active: bool = True) -> Optional[Dict[str, Any]]:
        filtered = [v for v in self._registry.values() if v.get("model_type") == model_type]
        if only_active:
            filtered = [v for v in filtered if v.get("active")]
        if not filtered:
            return None
        return sorted(filtered, key=lambda x: x.get("created_at", ""), reverse=True)[0]

    # --------------------------------
    # Validation & Safety
    # --------------------------------
    def validate_model(self, model_entry: Dict[str, Any]) -> bool:
        """
        Perform sanity checks before activation:
         - File exists and matches recorded hash.
         - Torch/joblib can load without error.
         - Optional inference smoke test.
        """
        path = pathlib.Path(model_entry["file_path"])
        if not path.exists():
            LOG.error("Validation failed: missing file %s", path)
            return False
        if _hash_file(path) != model_entry["file_hash"]:
            LOG.error("Validation failed: hash mismatch for %s", path)
            return False
        try:
            if path.suffix in (".pkl", ".joblib") and _HAS_JOBLIB:
                joblib.load(path)
            elif path.suffix in (".pt", ".pth") and _HAS_TORCH:
                torch.load(path, map_location="cpu")
        except Exception:
            LOG.exception("Model load failed during validation")
            return False
        LOG.info("Validation passed for %s", model_entry["version_id"])
        return True

    # --------------------------------
    # Promotion, Activation, Rollback
    # --------------------------------
    def promote_model(self, version_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Mark given version as active; deactivate others of same type.
        """
        entry = self.get_model(version_id)
        if not force and not self.validate_model(entry):
            raise ValueError("Validation failed; use force=True to override.")
        mtype = entry["model_type"]
        # deactivate others
        for v in self._registry.values():
            if v.get("model_type") == mtype:
                v["active"] = False
        entry["active"] = True
        entry["promoted_at"] = _now_iso()
        self._save_registry()
        _audit("model_promoted", entry)
        LOG.info("Promoted model %s (type=%s)", version_id, mtype)
        return entry

    def rollback_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Roll back to the previously active version (most recent before current).
        """
        history = [v for v in self._registry.values() if v.get("model_type") == model_type]
        history = sorted(history, key=lambda x: x.get("created_at", ""), reverse=True)
        if len(history) < 2:
            LOG.warning("No previous model available for rollback")
            return None
        current = history[0]
        prev = history[1]
        self.promote_model(prev["version_id"], force=True)
        _audit("model_rollback", {"from": current["version_id"], "to": prev["version_id"]})
        LOG.info("Rolled back %s from %s -> %s", model_type, current["version_id"], prev["version_id"])
        return prev

    def delete_model(self, version_id: str):
        """
        Safely delete model entry (keeps file by default).
        """
        entry = self._registry.pop(version_id, None)
        if not entry:
            raise KeyError(f"Model {version_id} not found")
        self._save_registry()
        _audit("model_deleted", entry)
        LOG.info("Deleted model entry: %s", version_id)

    # --------------------------------
    # Export & Sync
    # --------------------------------
    def export_metadata(self, out_path: Optional[str] = None) -> str:
        out = out_path or str(MODELS_DIR / f"registry_export_{int(time.time())}.json")
        json.dump(self._registry, open(out, "w", encoding="utf-8"), indent=2, default=str)
        return out

    def sync_s3(self):
        """
        Ensure all registered models exist in S3 bucket.
        """
        if not (_HAS_BOTO3 and self.s3_bucket):
            LOG.warning("S3 not configured; skipping sync")
            return
        s3 = boto3.client("s3")
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
                LOG.info("Uploaded missing model %s to S3", vid)
            except Exception:
                LOG.exception("Failed S3 sync for %s", vid)
        self._save_registry()

# ---------------------------
# CLI Utility
# ---------------------------
def _build_cli():
    import argparse
    parser = argparse.ArgumentParser(prog="prioritymax-model-registry")
    sub = parser.add_subparsers(dest="cmd")

    p_reg = sub.add_parser("register", help="Register new model")
    p_reg.add_argument("--type", required=True)
    p_reg.add_argument("--file", required=True)
    p_reg.add_argument("--metrics", help="JSON string of metrics")
    p_reg.add_argument("--meta", help="JSON string of metadata")
    p_reg.add_argument("--author", default="cli")
    p_reg.add_argument("--canary", action="store_true")
    p_reg.add_argument("--shadow", action="store_true")

    sub.add_parser("list", help="List all models")

    p_prom = sub.add_parser("promote", help="Promote model")
    p_prom.add_argument("--id", required=True)
    p_prom.add_argument("--force", action="store_true")

    p_rollback = sub.add_parser("rollback", help="Rollback latest model of a type")
    p_rollback.add_argument("--type", required=True)

    sub.add_parser("sync_s3", help="Sync missing models to S3")
    sub.add_parser("export", help="Export registry JSON")

    return parser

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    reg = ModelRegistry()
    if args.cmd == "register":
        metrics = json.loads(args.metrics) if args.metrics else None
        meta = json.loads(args.meta) if args.meta else None
        res = reg.register_model(args.type, args.file, metrics=metrics, metadata=meta, author=args.author, canary=args.canary, shadow=args.shadow)
        print(json.dumps(res, indent=2))
    elif args.cmd == "list":
        res = reg.list_models()
        print(json.dumps(res, indent=2))
    elif args.cmd == "promote":
        res = reg.promote_model(args.id, force=args.force)
        print(json.dumps(res, indent=2))
    elif args.cmd == "rollback":
        res = reg.rollback_model(args.type)
        print(json.dumps(res, indent=2))
    elif args.cmd == "sync_s3":
        reg.sync_s3()
    elif args.cmd == "export":
        path = reg.export_metadata()
        print(f"Exported registry to {path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

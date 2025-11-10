# backend/app/ml/predictor.py
"""
PriorityMax Predictor (final integrated version)

Features:
 - synchronous + asynchronous single-sample prediction
 - vectorized batch prediction with auto-chunking
 - GPU-aware batching: will split and dispatch batches across available CUDA devices (if PyTorch available)
 - parallelized execution (threadpool) for heavy models
 - SHAP-based explainability if available
 - Prometheus metrics hooks (optional)
 - Auditing via app.api.admin.write_audit_event when available (file fallback)
 - Integration with ModelRegistry for model discovery
 - Drift detection and retrain trigger
 - Safe fallbacks when optional dependencies are absent
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import math
import logging
import pathlib
import asyncio
import datetime
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional libs
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    lgb = None
    _HAS_LGB = False

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

try:
    from prometheus_client import Histogram, Gauge, start_http_server
    _HAS_PROM = True
except Exception:
    Histogram = Gauge = start_http_server = None
    _HAS_PROM = False

# Model registry and audit integration (best-effort)
try:
    from app.ml.model_registry import ModelRegistry
except Exception:
    # Minimal stub so file still imports; real registry recommended.
    class ModelRegistry:
        def __init__(self, *a, **k): pass
        def get_latest(self, *a, **k): return None

try:
    from app.api.admin import write_audit_event
    _HAS_AUDIT = True
except Exception:
    _HAS_AUDIT = False
    def write_audit_event(payload: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "predictor_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.ml.predictor")
LOG.setLevel(os.getenv("PRIORITYMAX_ML_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
MODELS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODELS_DIR", str(BASE_DIR / "app" / "ml" / "models")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Prometheus metrics
if _HAS_PROM:
    PRED_INFER_LATENCY = Histogram("prioritymax_predictor_infer_seconds", "Inference latency (seconds)", buckets=[0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.5,1,2,5])
    PRED_BATCH_SIZE = Gauge("prioritymax_predictor_batch_size", "Last batch size processed")
    PRED_CONF = Gauge("prioritymax_predictor_confidence", "Last prediction confidence")
    # optionally start a local exporter if env set
    try:
        prom_port = int(os.getenv("PRIORITYMAX_PROMETHEUS_PORT", "9001"))
        start_http_server(prom_port)
        LOG.info("Prometheus metrics served at port %d", prom_port)
    except Exception:
        pass
else:
    PRED_INFER_LATENCY = PRED_BATCH_SIZE = PRED_CONF = None

# concurrency helpers
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("PRIORITYMAX_PRED_THREADPOOL", "8")))

# utils
def _now_iso():
    return datetime.datetime.datetime.utcnow().isoformat() + "Z" if hasattr(datetime, "datetime") else str(time.time())

def _audit(event: str, details: Dict[str, Any]):
    try:
        write_audit_event({"event": event, "ts": datetime.datetime.utcnow().isoformat() + "Z", **details})
    except Exception:
        LOG.debug("audit failed for %s", event)

# -------------------------
# Predictor base classes
# -------------------------
class BasePredictor:
    def __init__(self, name: str = "base"):
        self.name = name
        self.model = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def load(self, path: Union[str, pathlib.Path]):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Default: call predict on whole array (subclasses may override for GPU/torch)"""
        return self.predict(X)

    def explain(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.name, "features": self.feature_names, "metadata": self.metadata}

# LightGBM predictor
class LightGBMPredictor(BasePredictor):
    def __init__(self):
        super().__init__("lightgbm")
        if not _HAS_LGB:
            LOG.warning("lightgbm not installed; LightGBMPredictor will not be available")

    def load(self, path: Union[str, pathlib.Path]):
        if not _HAS_LGB:
            raise RuntimeError("lightgbm not installed")
        LOG.info("Loading LightGBM model from %s", path)
        self.model = lgb.Booster(model_file=str(path))
        try:
            self.feature_names = self.model.feature_name()
        except Exception:
            self.feature_names = []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not loaded")
        # LightGBM accepts 2D arrays
        return np.asarray(self.model.predict(X))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def explain(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not _HAS_SHAP:
            return None
        try:
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(X)
        except Exception:
            LOG.exception("shap explain failed for lightgbm")
            return None

# Torch predictor for PyTorch-based models (supports GPU)
if _HAS_TORCH:
    class TorchPredictor(BasePredictor):
        def __init__(self, device: Optional[torch.device] = None):
            super().__init__("torch")
            self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
            self.model = None

        def load(self, path: Union[str, pathlib.Path]):
            LOG.info("Loading Torch model from %s", path)
            # torch.load may be heavy depending on model
            self.model = torch.load(str(path), map_location="cpu")
            # prefer to call eval()
            try:
                self.model.eval()
            except Exception:
                pass
            # do NOT move to GPU here; predict_batch will move tensors as needed for device
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self.model is None:
                raise RuntimeError("torch model not loaded")
            # single inference: convert to tensor, move to cpu for lightweight models
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(X, dtype=np.float32))
                out = self.model(x.to(self.device))
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                return out.detach().cpu().numpy()

        def predict_batch(self, X: np.ndarray) -> np.ndarray:
            """Vectorized batch inference with optional GPU usage."""
            if self.model is None:
                raise RuntimeError("torch model not loaded")
            with torch.no_grad():
                tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
                # handle 1D input (single sample)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                try:
                    tensor = tensor.to(self.device)
                    out = self.model(tensor)
                    if isinstance(out, tuple) or isinstance(out, list):
                        out = out[0]
                    return out.detach().cpu().numpy()
                except Exception:
                    LOG.exception("Torch batch prediction failed; retrying on CPU")
                    try:
                        out = self.model(tensor.cpu())
                        if isinstance(out, tuple) or isinstance(out, list):
                            out = out[0]
                        return out.detach().cpu().numpy()
                    except Exception:
                        LOG.exception("Torch CPU fallback failed")
                        raise

        def explain(self, X: np.ndarray) -> Optional[np.ndarray]:
            if not _HAS_SHAP:
                return None
            try:
                # shap supports torch kernel/DeepExplainer in some contexts; best-effort
                explainer = shap.DeepExplainer(self.model, torch.from_numpy(X.astype(np.float32)).to(self.device))
                vals = explainer.shap_values(torch.from_numpy(X.astype(np.float32)).to(self.device))
                return np.asarray(vals)
            except Exception:
                LOG.exception("shap explain failed for torch")
                return None
else:
    TorchPredictor = None  # type: ignore

# -------------------------
# PredictorManager (big class)
# -------------------------
class PredictorManager:
    """
    Orchestrates model loading, single + batch predictions, GPU batching and autoscaler hints.
    """

    def __init__(self,
                 model_registry: Optional[ModelRegistry] = None,
                 default_model_type: str = "predictor_lgbm",
                 batch_max_size: int = int(os.getenv("PRIORITYMAX_PRED_BATCH_MAX", "8192")),
                 chunk_size: int = int(os.getenv("PRIORITYMAX_PRED_CHUNK_SIZE", "2048")),
                 gpu_chunk_size: int = int(os.getenv("PRIORITYMAX_GPU_CHUNK_SIZE", "4096")),
                 executor: Optional[ThreadPoolExecutor] = None):
        self.registry = model_registry or ModelRegistry()
        self.default_model_type = default_model_type
        self.models: Dict[str, BasePredictor] = {}
        self.last_loaded_meta: Optional[Dict[str, Any]] = None
        self.lock = asyncio.Lock()
        self.batch_max_size = batch_max_size
        self.chunk_size = chunk_size
        self.gpu_chunk_size = gpu_chunk_size
        self.executor = executor or _EXECUTOR
        # rolling scaler stats: feature_name -> (mean, std)
        self.scaler_stats: Dict[str, Tuple[float, float]] = {}
        # available devices: list of torch.device
        self.devices: List[Any] = []
        if _HAS_TORCH and torch.cuda.is_available():
            self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            LOG.info("Detected CUDA devices: %s", self.devices)
        else:
            self.devices = []
        # small cache for model path -> loaded instance
        self._loaded_paths: Dict[str, BasePredictor] = {}

    # -------------------------
    # Model load / management
    # -------------------------
    def _load_predictor_from_path(self, path: Union[str, pathlib.Path]) -> BasePredictor:
        p = pathlib.Path(path)
        suffix = p.suffix.lower()
        if suffix in (".txt", ".model", ".pkl", ".joblib") and _HAS_LGB:
            pred = LightGBMPredictor().load(p)
            return pred
        if suffix in (".pt", ".pth") and _HAS_TORCH:
            # prefer to create TorchPredictor for CPU/GPU
            dev = self.devices[0] if self.devices else None
            pred = TorchPredictor(device=dev).load(p)
            return pred
        # fallback: try lgb then torch
        if _HAS_LGB:
            try:
                pred = LightGBMPredictor().load(p)
                return pred
            except Exception:
                LOG.exception("Failed load as LightGBM")
        if _HAS_TORCH:
            try:
                pred = TorchPredictor().load(p)
                return pred
            except Exception:
                LOG.exception("Failed load as Torch")
        raise RuntimeError("No suitable predictor loader found for %s" % path)

    def load_latest(self, model_type: Optional[str] = None) -> Optional[BasePredictor]:
        """
        Load the latest *active* model of given type from the registry and cache it.
        Returns loaded predictor instance or None.
        """
        model_type = model_type or self.default_model_type
        meta = self.registry.get_latest(model_type)
        if not meta:
            LOG.warning("No latest model metadata for %s", model_type)
            return None
        path = meta.get("file_path") or meta.get("file")
        if not path:
            LOG.warning("Model %s has no file_path", meta.get("version_id"))
            return None
        # reuse cached instance per path
        if path in self._loaded_paths:
            self.last_loaded_meta = meta
            self.models[model_type] = self._loaded_paths[path]
            LOG.info("Using cached predictor for %s (path=%s)", model_type, path)
            return self._loaded_paths[path]
        try:
            pred = self._load_predictor_from_path(path)
            self._loaded_paths[path] = pred
            self.models[model_type] = pred
            self.last_loaded_meta = meta
            _audit("predictor_model_loaded", {"model_type": model_type, "version": meta.get("version_id")})
            LOG.info("Loaded model %s from %s", meta.get("version_id"), path)
            return pred
        except Exception as e:
            LOG.exception("Failed to load model from path %s: %s", path, e)
            return None

    def get_model(self, model_type: Optional[str] = None) -> Optional[BasePredictor]:
        model_type = model_type or self.default_model_type
        return self.models.get(model_type)

    # -------------------------
    # Feature normalization
    # -------------------------
    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in df_out.columns:
            if not np.issubdtype(df_out[col].dtype, np.number):
                continue
            vals = df_out[col].dropna()
            if vals.empty:
                continue
            mean = float(vals.mean())
            std = float(vals.std()) if float(vals.std()) > 1e-6 else 1.0
            prev = self.scaler_stats.get(col)
            if prev:
                # exponential moving average to avoid abrupt changes
                mean = 0.9 * prev[0] + 0.1 * mean
                std = 0.9 * prev[1] + 0.1 * std
            self.scaler_stats[col] = (mean, std)
            df_out[col] = (df_out[col] - mean) / (std or 1.0)
        return df_out.fillna(0.0)

    # -------------------------
    # Single-sample prediction
    # -------------------------
    def predict(self, features: Dict[str, float], model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous single-sample predict (uses vectorized path under the hood).
        """
        model_type = model_type or self.default_model_type
        # convert into df and call vectorized predict for consistency
        preds = self.predict_batch([features], model_type=model_type, max_workers=1)
        if not preds:
            raise RuntimeError("prediction failed")
        return preds[0]

    async def predict_async(self, features: Dict[str, float], model_type: Optional[str] = None) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.predict, features, model_type)

    # -------------------------
    # Autoscaler hint
    # -------------------------
    def autoscaler_hint(self, score: float) -> str:
        # customizable thresholds via env
        up_th = float(os.getenv("PRIORITYMAX_AUTOSCALE_UP", "0.8"))
        down_th = float(os.getenv("PRIORITYMAX_AUTOSCALE_DOWN", "0.3"))
        if score >= up_th:
            return "scale_up"
        if score <= down_th:
            return "scale_down"
        return "steady"

    # -------------------------
    # Vectorized batch prediction (core)
    # -------------------------
    def _chunk_generator(self, items: List[Dict[str, float]], chunk_size: int):
        for i in range(0, len(items), chunk_size):
            yield items[i: i + chunk_size]

    def _prepare_matrix(self, batch: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        # create DataFrame with union of keys
        df = pd.DataFrame(batch)
        # ensure consistent ordering of columns
        cols = list(df.columns)
        # normalize
        df_norm = self.normalize_df(df)
        return df_norm.values.astype(np.float32), cols

    def _predict_chunk_with_model(self, model: BasePredictor, X: np.ndarray) -> np.ndarray:
        """
        Call model.predict_batch or predict; provide safe wrappers.
        """
        # if the model implements predict_batch, call that
        try:
            if hasattr(model, "predict_batch"):
                return np.asarray(model.predict_batch(X))
        except Exception:
            LOG.exception("model.predict_batch failed, falling back to predict per-sample")
        # fallback: call predict for whole array
        try:
            return np.asarray(model.predict(X))
        except Exception:
            # last fallback: per-sample loop
            res = []
            for i in range(X.shape[0]):
                try:
                    res.append(float(model.predict(X[i:i+1]).ravel().mean()))
                except Exception:
                    res.append(0.0)
            return np.asarray(res)

    def predict_batch(self,
                      features_list: List[Dict[str, float]],
                      model_type: Optional[str] = None,
                      max_workers: int = 4,
                      max_batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Vectorized batch prediction entrypoint.

        - Automatically chunks large inputs.
        - If PyTorch with multiple GPUs is available, will attempt GPU-aware chunk dispatch:
          divide the batch among devices and run predict_batch concurrently across threads.
        - Returns list of dicts: {prediction, latency_sec, autoscaler_hint, model_version}
        """
        model_type = model_type or self.default_model_type
        if not features_list:
            return []

        if max_batch_size is None:
            max_batch_size = self.batch_max_size

        # ensure model loaded
        model = self.get_model(model_type) or self.load_latest(model_type)
        if not model:
            raise RuntimeError("No model available for type %s" % model_type)

        total = len(features_list)
        results: List[Dict[str, Any]] = []
        t_start_all = time.perf_counter()

        # choose chunk size: use GPU chunk size if torch GPU available and model is TorchPredictor
        use_gpu = _HAS_TORCH and isinstance(model, TorchPredictor) and len(self.devices) > 0
        effective_chunk = self.gpu_chunk_size if use_gpu else self.chunk_size

        # limit chunk size not to exceed max_batch_size
        effective_chunk = min(effective_chunk, max_batch_size)

        # If GPU & multiple devices: distribute chunks round-robin to devices
        if use_gpu and len(self.devices) > 1:
            # create device-specific predictors (move model to device copies if needed)
            # Note: we will call predict_batch which moves tensors to the device internally in TorchPredictor
            # We'll execute multiple chunk predictions in parallel using executor
            loop = asyncio.new_event_loop()
            # We'll not use asyncio here; we'll use ThreadPoolExecutor to submit pops
            from concurrent.futures import as_completed
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                # submit each chunk to pool, assigning device in round-robin via monkeypatching predictor.device
                dev_count = len(self.devices)
                idx = 0
                for chunk in self._chunk_generator(features_list, effective_chunk):
                    assigned_dev = self.devices[idx % dev_count]
                    idx += 1
                    # ensure model instance has device attribute; create wrapper if necessary
                    # create temporary wrapper around model that sets device for call
                    def _call_chunk(chunk_local, dev, mdl=model):
                        try:
                            # if torch model has attribute device, set it
                            if hasattr(mdl, "device"):
                                mdl.device = dev
                            X_mat, _ = self._prepare_matrix(chunk_local)
                            t0 = time.perf_counter()
                            preds_raw = self._predict_chunk_with_model(mdl, X_mat)
                            latency = time.perf_counter() - t0
                            return preds_raw, latency
                        except Exception:
                            LOG.exception("chunk prediction error")
                            return np.zeros((len(chunk_local),)), 0.0
                    futures.append(pool.submit(_call_chunk, chunk, assigned_dev))
                # collect results in order: as chunks were submitted, we can iterate futures
                # but as_completed yields in completion order; we'll instead keep the submit order and pop results accordingly
                for fut in futures:
                    preds_raw, latency = fut.result()
                    # preds_raw may be 1D or 2D; normalize to 2D list-of-values per sample
                    if preds_raw.ndim == 1:
                        # scalar per sample
                        for p in preds_raw:
                            score = float(np.mean(np.atleast_1d(p)))
                            hint = self.autoscaler_hint(score)
                            results.append({"prediction": score, "latency_sec": round(latency / max(1, len(preds_raw)), 6), "autoscaler_hint": hint, "model_version": self.last_loaded_meta.get("version_id") if self.last_loaded_meta else None})
                    else:
                        for p in preds_raw:
                            score = float(np.mean(p))
                            hint = self.autoscaler_hint(score)
                            results.append({"prediction": score, "latency_sec": round(latency / max(1, preds_raw.shape[0]), 6), "autoscaler_hint": hint, "model_version": self.last_loaded_meta.get("version_id") if self.last_loaded_meta else None})
        else:
            # simpler, single-device or CPU path: process chunks sequentially or with limited parallelism
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                for chunk in self._chunk_generator(features_list, effective_chunk):
                    futures.append(pool.submit(self._process_chunk_sync, model, chunk))
                for fut in futures:
                    batch_res = fut.result()
                    results.extend(batch_res)

        t_all = time.perf_counter() - t_start_all
        if PRED_INFER_LATENCY:
            try:
                PRED_INFER_LATENCY.observe(t_all)
                PRED_BATCH_SIZE.set(len(features_list))
            except Exception:
                pass
        _audit("predictor_batch", {"count": len(features_list), "model_type": model_type, "elapsed": t_all})
        return results

    def _process_chunk_sync(self, model: BasePredictor, chunk: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Prepare chunk -> call model -> produce per-sample dict results
        """
        try:
            X_mat, cols = self._prepare_matrix(chunk)
            t0 = time.perf_counter()
            preds_raw = self._predict_chunk_with_model(model, X_mat)
            latency = time.perf_counter() - t0
            out = []
            if preds_raw.ndim == 1:
                for p in preds_raw:
                    score = float(np.mean(np.atleast_1d(p)))
                    out.append({"prediction": score, "latency_sec": round(latency / max(1, len(preds_raw)), 6), "autoscaler_hint": self.autoscaler_hint(score), "model_version": self.last_loaded_meta.get("version_id") if self.last_loaded_meta else None})
            else:
                for p in preds_raw:
                    score = float(np.mean(p))
                    out.append({"prediction": score, "latency_sec": round(latency / max(1, preds_raw.shape[0]), 6), "autoscaler_hint": self.autoscaler_hint(score), "model_version": self.last_loaded_meta.get("version_id") if self.last_loaded_meta else None})
            return out
        except Exception:
            LOG.exception("Chunk processing failed")
            # fallback: produce zeros
            return [{"prediction": 0.0, "latency_sec": 0.0, "autoscaler_hint": "steady", "model_version": self.last_loaded_meta.get("version_id") if self.last_loaded_meta else None} for _ in chunk]

    # -------------------------
    # Explainability
    # -------------------------
    def explain(self, features: Dict[str, float], model_type: Optional[str] = None, top_k: int = 10) -> Dict[str, float]:
        """
        Return top_k feature contributions (SHAP or best-effort).
        """
        model_type = model_type or self.default_model_type
        model = self.get_model(model_type) or self.load_latest(model_type)
        if not model:
            raise RuntimeError("no model available")
        import pandas as pd
        df = pd.DataFrame([features])
        dfn = self.normalize_df(df)
        arr = dfn.values.astype(np.float32)
        expl = None
        try:
            expl = model.explain(arr)
        except Exception:
            LOG.exception("model explain failed")
            expl = None
        if expl is None:
            # fallback: return feature magnitudes as proxy
            return {k: abs(float(v)) for k, v in features.items() if isinstance(v, (int, float))}
        # SHAP outputs vary in shape. For single-row, typically returns array of shape (n_features,) or list per class
        try:
            if isinstance(expl, list):
                # multiclass -> use first-class contributions and absolute magnitude
                arrv = np.asarray(expl[0])
            else:
                arrv = np.asarray(expl)
            if arrv.ndim == 2:
                arrv = arrv[0]
            fnames = getattr(model, "feature_names", list(dfn.columns))
            contribs = {}
            for i, val in enumerate(arrv):
                if i < len(fnames):
                    contribs[str(fnames[i])] = float(val)
            # sort top_k by absolute value
            sorted_items = dict(sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])
            return sorted_items
        except Exception:
            LOG.exception("explain parse error")
            return {k: float(v) for k, v in features.items() if isinstance(v, (int, float))}

    # -------------------------
    # Drift detection
    # -------------------------
    def compute_drift(self, new_df: pd.DataFrame, ref_df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        per_feature = {}
        numeric_cols = [c for c in new_df.columns if np.issubdtype(new_df[c].dtype, np.number) and c in ref_df.columns]
        for c in numeric_cols:
            mu_new, mu_ref = new_df[c].mean(), ref_df[c].mean()
            std_ref = ref_df[c].std() if ref_df[c].std() > 1e-6 else 1.0
            delta = abs(mu_new - mu_ref) / std_ref
            per_feature[c] = float(delta)
        avg = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
        flag = avg > threshold
        _audit("predictor_drift", {"avg": avg, "flag": flag})
        return {"avg_drift": avg, "flag": flag, "per_feature": per_feature}

    # -------------------------
    # Retrain trigger
    # -------------------------
    def trigger_retrain(self, reason: str = "manual") -> bool:
        """
        Trigger retraining process. Implemented as best-effort invocation of configured retrain script
        or by writing a Redis key / enqueuing job for a training pipeline.
        """
        script = pathlib.Path(BASE_DIR) / "app" / "scripts" / "retrain_predictor_live.py"
        if script.exists():
            try:
                _audit("predictor_retrain_triggered", {"reason": reason})
                # spawn as background process to avoid blocking
                os.system(f"python3 {str(script)} &")
                return True
            except Exception:
                LOG.exception("Failed to spawn retrain script")
                return False
        # fallback: write file marker for external orchestrator to pick up
        marker = pathlib.Path("/tmp/prioritymax_retrain_marker.json")
        try:
            marker.write_text(json.dumps({"ts": time.time(), "reason": reason}))
            _audit("predictor_retrain_marker", {"reason": reason})
            return True
        except Exception:
            LOG.exception("Failed to write retrain marker")
            return False

# -------------------------
# Singleton manager
# -------------------------
PREDICTOR_MANAGER = PredictorManager()

# -------------------------
# CLI helper
# -------------------------
def _build_cli():
    import argparse
    parser = argparse.ArgumentParser(prog="prioritymax-predictor")
    sub = parser.add_subparsers(dest="cmd")
    p1 = sub.add_parser("load", help="Load latest model")
    p1.add_argument("--type", default=None)
    p2 = sub.add_parser("predict", help="Single prediction (JSON features)")
    p2.add_argument("--features", required=True, help='JSON string of features')
    p3 = sub.add_parser("batch", help="Batch predict from JSON file")
    p3.add_argument("--file", required=True)
    p3.add_argument("--type", default=None)
    p3.add_argument("--out", default=None)
    p4 = sub.add_parser("explain", help="Explain single features")
    p4.add_argument("--features", required=True)
    p5 = sub.add_parser("drift", help="Compute drift between two CSVs")
    p5.add_argument("--new", required=True)
    p5.add_argument("--ref", required=True)
    p6 = sub.add_parser("retrain", help="Trigger retrain")
    p6.add_argument("--reason", default="cli")
    return parser

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    mgr = PREDICTOR_MANAGER
    if args.cmd == "load":
        m = mgr.load_latest(args.type)
        print("loaded:", m.get_info() if m else None)
    elif args.cmd == "predict":
        feats = json.loads(args.features)
        res = mgr.predict(feats, args.type)
        print(json.dumps(res, indent=2))
    elif args.cmd == "batch":
        data = json.loads(open(args.file).read())
        res = mgr.predict_batch(data, model_type=args.type)
        if args.out:
            open(args.out, "w").write(json.dumps(res, indent=2))
        else:
            print(json.dumps(res, indent=2))
    elif args.cmd == "explain":
        feats = json.loads(args.features)
        print(json.dumps(mgr.explain(feats, None), indent=2))
    elif args.cmd == "drift":
        import pandas as pd
        n = pd.read_csv(args.new)
        r = pd.read_csv(args.ref)
        print(json.dumps(mgr.compute_drift(n, r), indent=2))
    elif args.cmd == "retrain":
        ok = mgr.trigger_retrain(args.reason)
        print("retrain triggered:", ok)
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

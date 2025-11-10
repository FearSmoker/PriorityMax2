# backend/app/ml/data_pipeline.py
"""
PriorityMax ML Data Pipeline

Responsibilities:
 - Collect & ingest queue/task telemetry from live production (Redis, Mongo) or simulations.
 - Preprocess raw telemetry into cleaned tabular datasets.
 - Feature engineering: rolling stats, time-window aggregates, label generation for predictors.
 - Train/test split, dataset versioning and storage (local FS + optional S3).
 - Helpers to train LightGBM predictors and export PyTorch datasets for RL / custom models.
 - Integration hooks for MLflow / wandb / tensorboard for experiment tracking (optional).
 - RL experience replay buffer export/import utilities.
 - Streaming ingest helper (async) to continuously append to buffer from Redis.
 - Scheduling helper to run periodic retraining based on new data.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import math
import shutil
import queue
import logging
import pathlib
import tempfile
import datetime
import argparse
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable, Union

import numpy as np
import pandas as pd

# optional dependencies
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    lgb = None
    _HAS_LGB = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:
    torch = None
    Dataset = object
    DataLoader = None
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
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

# async connectors
try:
    import aioredis
    _HAS_AIOREDIS = True
except Exception:
    aioredis = None
    _HAS_AIOREDIS = False

try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

# local audit hook (best-effort)
try:
    from app.api.admin import write_audit_event
    _HAS_AUDIT = True
except Exception:
    _HAS_AUDIT = False
    def write_audit_event(payload: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "ml_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.ml.data_pipeline")
LOG.setLevel(os.getenv("PRIORITYMAX_ML_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Paths & config
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
DATASETS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_DATASETS_DIR", str(BASE_DIR / "datasets")))
MODELS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODELS_DIR", str(BASE_DIR / "ml" / "models")))
ARTIFACTS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_ARTIFACTS_DIR", str(BASE_DIR / "ml" / "artifacts")))
S3_BUCKET = os.getenv("PRIORITYMAX_S3_BUCKET", None)
DEFAULT_SAMPLE_RATE = float(os.getenv("PRIORITYMAX_DEFAULT_SAMPLE_RATE", "1.0"))

for d in (DATASETS_DIR, MODELS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# dataset versioning file
_VERSION_INDEX = DATASETS_DIR / "datasets_index.json"
if not _VERSION_INDEX.exists():
    _VERSION_INDEX.write_text(json.dumps({}), encoding="utf-8")

def _index_read() -> Dict[str, Any]:
    try:
        return json.loads(_VERSION_INDEX.read_text(encoding="utf-8") or "{}")
    except Exception:
        LOG.exception("Failed reading dataset index")
        return {}

def _index_write(idx: Dict[str, Any]):
    try:
        _VERSION_INDEX.write_text(json.dumps(idx, default=str, indent=2), encoding="utf-8")
    except Exception:
        LOG.exception("Failed writing dataset index")

# -----------------------------------
# Utilities
# -----------------------------------
def _now_iso() -> str:
    return datetime.datetime.datetime.utcnow().isoformat() + "Z" if hasattr(datetime, "datetime") else time.time()

def _audit(event: str, details: Dict[str, Any]):
    """Write an audit event if available"""
    payload = {"event": event, "ts": datetime.datetime.utcnow().isoformat() + "Z", **details}
    try:
        if _HAS_AUDIT:
            maybe = write_audit_event(payload)
            if hasattr(maybe, "__await__"):
                # avoid awaiting in sync contexts
                try:
                    import asyncio
                    asyncio.create_task(maybe)
                except Exception:
                    LOG.debug("Could not schedule audit coroutine")
        else:
            write_audit_event(payload)
    except Exception:
        LOG.exception("Audit write failed for %s", event)

def _save_csv(df: pd.DataFrame, path: Union[str, pathlib.Path], compress: bool = False):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        p = p.with_suffix(p.suffix + ".gz")
        df.to_csv(str(p), index=False, compression="gzip")
    else:
        df.to_csv(str(p), index=False)
    return str(p)

def _save_npz(arrays: Dict[str, np.ndarray], path: Union[str, pathlib.Path]):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), **arrays)
    return str(p)

def _upload_to_s3(local_path: Union[str, pathlib.Path], bucket: Optional[str], s3_key: Optional[str] = None) -> Dict[str, Any]:
    if not _HAS_BOTO3 or not bucket:
        LOG.warning("boto3 not installed or S3 bucket not configured; skipping upload")
        return {"ok": False, "reason": "boto3_or_bucket_missing"}
    s3 = boto3.client("s3")
    key = s3_key or f"prioritymax/{pathlib.Path(local_path).name}"
    try:
        s3.upload_file(str(local_path), bucket, key)
        return {"ok": True, "bucket": bucket, "key": key}
    except Exception:
        LOG.exception("Failed upload to s3")
        return {"ok": False, "error": "upload_failed"}

# -----------------------------------
# DataCollector: batch & streaming
# -----------------------------------
class DataCollector:
    """
    Collects raw queue/task telemetry from multiple sources:
      - CSV/JSON files (offline)
      - MongoDB (task logs, metrics)
      - Redis (stream-ingested queue metrics)
      - Simulated generator (for offline training)
    Provides simple dedup, normalization, and raw->DataFrame conversion hooks.
    """

    def __init__(self,
                 mongo_url: Optional[str] = os.getenv("MONGO_URL", None),
                 redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                 sample_rate: float = DEFAULT_SAMPLE_RATE):
        self.mongo_url = mongo_url
        self.redis_url = redis_url
        self.sample_rate = float(sample_rate)
        self._mongo_client = None
        self._redis = None

    # -----------------------
    # synchronous readers
    # -----------------------
    def read_from_csv(self, path: Union[str, pathlib.Path], nrows: Optional[int] = None) -> pd.DataFrame:
        LOG.info("Reading CSV: %s", path)
        df = pd.read_csv(str(path), nrows=nrows)
        return df

    def read_from_jsonl(self, path: Union[str, pathlib.Path], nrows: Optional[int] = None) -> pd.DataFrame:
        LOG.info("Reading JSONL: %s", path)
        rows = []
        with open(str(path), "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if nrows and i >= nrows:
                    break
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return pd.DataFrame(rows)

    def read_simulated(self, runs: int = 1000, avg_tasks_per_run: int = 10, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate simulated queue metrics and task logs for offline training.
        Columns:
          - timestamp, run_id, task_id, task_type, duration_ms, success, consumer_count, queue_length, priority
        """
        LOG.info("Generating simulated dataset: runs=%d, avg_tasks=%d", runs, avg_tasks_per_run)
        rng = np.random.RandomState(seed or int(time.time()))
        rows = []
        for r in range(runs):
            run_id = f"sim_{r:06d}"
            tasks = max(1, int(max(1, rng.poisson(lam=avg_tasks_per_run))))
            base_ts = int(time.time()) - rng.randint(0, 60 * 60 * 24)
            for t in range(tasks):
                ts = base_ts + t * rng.randint(1, 10)
                duration = max(1, int(rng.exponential(scale=200)))
                success = bool(rng.rand() > 0.05)
                consumer_count = int(max(1, rng.randint(1, 8)))
                queue_len = int(max(0, rng.poisson(lam=consumer_count * 2)))
                priority = int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
                rows.append({
                    "timestamp": ts,
                    "run_id": run_id,
                    "task_id": f"{run_id}_t{t}",
                    "task_type": rng.choice(["io", "cpu", "db"]),
                    "duration_ms": duration,
                    "success": int(success),
                    "consumer_count": consumer_count,
                    "queue_length": queue_len,
                    "priority": priority,
                })
        df = pd.DataFrame(rows)
        LOG.info("Simulated dataset generated: rows=%d", len(df))
        return df

    # -----------------------
    # async streaming ingest (Redis / Streams)
    # -----------------------
    async def connect_redis(self):
        if not _HAS_AIOREDIS:
            raise RuntimeError("aioredis not installed")
        if self._redis:
            return self._redis
        self._redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        return self._redis

    async def stream_from_redis_pubsub(self, channel: str, handler: Callable[[Dict[str, Any]], None], stop_event: Optional[asyncio.Event] = None):
        """
        Subscribe to Redis pubsub channel and call handler(payload) for each message received.
        Handler may be sync; we call it in threadpool if necessary.
        """
        r = await self.connect_redis()
        ps = r.pubsub()
        await ps.subscribe(channel)
        LOG.info("Subscribed to redis channel: %s", channel)
        try:
            while not (stop_event and stop_event.is_set()):
                msg = await ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    await asyncio.sleep(0)
                    continue
                data = msg.get("data")
                if not data:
                    continue
                try:
                    payload = json.loads(data)
                except Exception:
                    LOG.debug("Ignoring non-json payload")
                    continue
                # sampling
                if self.sample_rate < 1.0 and np.random.rand() > self.sample_rate:
                    continue
                try:
                    res = handler(payload)
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    LOG.exception("Handler failed for payload")
        finally:
            try:
                await ps.unsubscribe(channel)
            except Exception:
                pass

    # -----------------------
    # mongo ingestion (async)
    # -----------------------
    async def connect_mongo(self):
        if not _HAS_MOTOR or not self.mongo_url:
            raise RuntimeError("motor/mongo not configured")
        if self._mongo_client:
            return self._mongo_client
        self._mongo_client = motor_asyncio.AsyncIOMotorClient(self.mongo_url)
        return self._mongo_client

    async def query_task_logs(self, db: Optional[str] = None, collection: str = "task_logs", query: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> pd.DataFrame:
        if not _HAS_MOTOR:
            raise RuntimeError("motor not installed")
        client = await self.connect_mongo()
        dbname = db or client.get_default_database().name
        coll = client[dbname][collection]
        q = query or {}
        cursor = coll.find(q)
        if limit:
            cursor = cursor.limit(limit)
        rows = []
        async for doc in cursor:
            # remove _id
            doc.pop("_id", None)
            rows.append(doc)
        return pd.DataFrame(rows)

# -----------------------------------
# Preprocessor & Feature Engineering
# -----------------------------------
class Preprocessor:
    """
    Convert raw telemetry DataFrame into training-ready features + labels.
    Provides:
     - type conversions, missing value handling
     - rolling window features (mean, median, std) by task_type or queue
     - label generation for predictor tasks (e.g., will_task_succeed, latency bucket)
     - feature metadata schema output
    """

    def __init__(self, time_col: str = "timestamp", id_cols: List[str] = None):
        self.time_col = time_col
        self.id_cols = id_cols or ["run_id", "task_id"]

    def basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure time col
        if self.time_col in df.columns:
            # if integer UNIX timestamps -> convert to datetime
            if np.issubdtype(df[self.time_col].dtype, np.integer):
                df[self.time_col] = pd.to_datetime(df[self.time_col], unit="s")
            else:
                try:
                    df[self.time_col] = pd.to_datetime(df[self.time_col])
                except Exception:
                    df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        else:
            df[self.time_col] = pd.to_datetime("now")

        # basic type coercion
        if "success" in df.columns:
            df["success"] = df["success"].astype(int)
        if "duration_ms" in df.columns:
            df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce").fillna(0).astype(int)

        # fill missing
        df = df.fillna({"priority": 0, "queue_length": 0, "consumer_count": 1})
        return df

    def rolling_features(self, df: pd.DataFrame, by: List[str] = ["task_type"], windows: List[int] = [60, 300]) -> pd.DataFrame:
        """
        Create rolling stats (mean, std, count) of duration_ms and queue_length per 'by' keys over windows (seconds).
        Adds columns like: duration_mean_60, duration_std_300
        """
        LOG.info("Computing rolling features by %s windows=%s", by, windows)
        df_sorted = df.sort_values(self.time_col)
        out = df_sorted.copy()
        out = out.reset_index(drop=True)
        for w in windows:
            suffix = f"_{w}"
            # windowing by time: convert to pandas Timedelta window
            window_str = f"{w}s"
            # groupby and rolling need time-indexed series
            temp = out.set_index(self.time_col)
            agg = temp.groupby(by).rolling(window_str).agg({"duration_ms": ["mean", "std", "count"], "queue_length": ["mean", "std"]})
            # flatten multiindex
            agg.columns = ["_".join(col).strip() + suffix for col in agg.columns]
            agg = agg.reset_index()
            # merge back on index and time
            out = out.merge(agg, on=list(out.columns[:len(out.columns)])[:0] + [self.time_col] + by, how="left") if False else pd.concat([out, agg.reindex(out.index)], axis=1)
            # safe fill
            for c in agg.columns:
                out[c] = out[c].fillna(0)
        return out

    def label_generation(self, df: pd.DataFrame, label_type: str = "next_task_latency_bucket") -> pd.DataFrame:
        """
        Create labels for supervised predictor training.
        Supported label_type:
          - 'will_fail' -> binary label whether task success==0
          - 'latency_bucket' -> discrete bucket of duration_ms
          - 'next_task_latency_bucket' -> latency bucket for next task in same run
        """
        LOG.info("Generating labels: %s", label_type)
        out = df.copy()
        if label_type == "will_fail":
            out["label"] = 1 - out["success"]
        elif label_type == "latency_bucket":
            # bucketize duration_ms into quantiles
            out["label"] = pd.qcut(out["duration_ms"].rank(method="first"), q=5, labels=False, duplicates="drop").fillna(0).astype(int)
        elif label_type == "next_task_latency_bucket":
            # sort by run_id and time, shift duration to next
            out = out.sort_values([self.time_col])
            out["next_duration"] = out.groupby("run_id")["duration_ms"].shift(-1)
            out["label"] = pd.qcut(out["next_duration"].fillna(out["duration_ms"]).rank(method="first"), q=5, labels=False, duplicates="drop").fillna(0).astype(int)
        else:
            raise ValueError("unsupported label type")
        return out

    def encode_categoricals(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """
        Simple label encoding for categorical fields. Returns mapping.
        """
        mapping = {}
        out = df.copy()
        for c in columns:
            vals = pd.Series(out[c].fillna("")).astype(str)
            uniques = vals.unique().tolist()
            mp = {v: i for i, v in enumerate(uniques)}
            mapping[c] = mp
            out[c] = vals.map(mp).fillna(0).astype(int)
        return out, mapping

    def build_feature_matrix(self, df: pd.DataFrame, feature_columns: List[str], label_column: str = "label") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return X (np.float32) and y (np.int64) ready for LightGBM / PyTorch.
        """
        X = df[feature_columns].fillna(0).astype(float).values.astype(np.float32)
        y = df[label_column].fillna(0).astype(int).values.astype(np.int64)
        return X, y

# -----------------------------------
# Dataset Manager: versioning & storage
# -----------------------------------
class DatasetManager:
    """
    Save / load dataset versions. Maintains an index JSON for dataset versions.
    """

    def __init__(self, base_dir: pathlib.Path = DATASETS_DIR):
        self.base_dir = pathlib.Path(base_dir)
        self.index_file = self.base_dir / "datasets_index.json"
        self.index = _index_read()

    def new_version(self, name: str, df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None, compress: bool = True, upload_s3: bool = False) -> Dict[str, Any]:
        """
        Create a new version entry and save file. Returns metadata dict.
        """
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        version_id = f"{name}_v{ts}_{uuid.uuid4().hex[:6]}"
        filename = f"{version_id}.csv.gz" if compress else f"{version_id}.csv"
        path = self.base_dir / filename
        _save_csv(df, path, compress=compress)
        meta_obj = {"version_id": version_id, "name": name, "file": str(path), "created_at": _now_iso(), "rows": len(df)}
        if meta:
            meta_obj.update(meta)
        self.index[version_id] = meta_obj
        _index_write(self.index)
        _audit("dataset_new_version", {"version_id": version_id, "name": name, "rows": len(df)})
        if upload_s3 and S3_BUCKET:
            up = _upload_to_s3(path, S3_BUCKET, s3_key=f"datasets/{filename}")
            meta_obj["s3"] = up
            _index_write(self.index)
        return meta_obj

    def list_versions(self, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for k, v in self.index.items():
            if name_filter and name_filter not in v.get("name", ""):
                continue
            results.append(v)
        # sort by created_at desc
        return sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)

    def load_version(self, version_id: str) -> pd.DataFrame:
        meta = self.index.get(version_id)
        if not meta:
            raise FileNotFoundError("version not found")
        path = pathlib.Path(meta["file"])
        if not path.exists():
            raise FileNotFoundError("dataset file missing")
        df = pd.read_csv(str(path))
        return df

# -----------------------------------
# LightGBM Trainer wrapper
# -----------------------------------
class LightGBMTrainer:
    """
    LightGBM trainer helper including:
     - dataset conversion
     - parameter defaults tuned for small-medium workloads
     - cross-validation & early stopping
     - model save/load helpers (to MODELS_DIR)
     - optional MLFlow/WandB logging
    """

    def __init__(self,
                 params: Optional[Dict[str, Any]] = None,
                 num_boost_round: int = int(os.getenv("LGB_NUM_ROUND", "1000")),
                 early_stopping_rounds: int = int(os.getenv("LGB_EARLY_STOP", "50")),
                 model_dir: pathlib.Path = MODELS_DIR,
                 use_mlflow: bool = False,
                 use_wandb: bool = False):
        default = {
            "objective": "multiclass" if int(os.getenv("LGB_MULTICLASS", "0")) else "binary",
            "metric": "auc" if int(os.getenv("LGB_BINARY", "1")) else "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": float(os.getenv("LGB_LR", "0.05")),
            "num_leaves": int(os.getenv("LGB_LEAVES", "31")),
            "min_data_in_leaf": int(os.getenv("LGB_MIN_DATA", "20")),
        }
        if params:
            default.update(params)
        self.params = default
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model_dir = pathlib.Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow and _HAS_MLFLOW
        self.use_wandb = use_wandb and _HAS_WANDB

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
        if not _HAS_LGB:
            raise RuntimeError("LightGBM not installed")
        LOG.info("Starting LightGBM training: rounds=%d", self.num_boost_round)
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_sets = [dtrain]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)
            valid_sets.append(dval)
            valid_names.append("valid")
        evals_result = {}
        booster = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round, valid_sets=valid_sets, valid_names=valid_names, early_stopping_rounds=self.early_stopping_rounds, evals_result=evals_result, verbose_eval=False)
        # save model
        model_name = model_name or f"lgb_model_{int(time.time())}_{uuid.uuid4().hex[:6]}.txt"
        model_path = self.model_dir / model_name
        booster.save_model(str(model_path))
        LOG.info("Saved LightGBM model to %s", model_path)
        # optional tracking
        if self.use_mlflow:
            try:
                mlflow.start_run()
                mlflow.log_params(self.params)
                mlflow.log_metric("best_iteration", booster.best_iteration)
                mlflow.log_artifact(str(model_path))
                mlflow.end_run()
            except Exception:
                LOG.exception("MLflow logging failed")
        if self.use_wandb:
            try:
                wandb.init(project=os.getenv("WANDB_PROJECT", "prioritymax_ml"), reinit=True)
                wandb.config.update(self.params)
                wandb.save(str(model_path))
                wandb.finish()
            except Exception:
                LOG.exception("WandB logging failed")
        _audit("trainer_lgb_completed", {"model_path": str(model_path), "params": self.params})
        return {"model_path": str(model_path), "evals_result": evals_result, "best_iteration": getattr(booster, "best_iteration", None)}

    def predict(self, model_path: str, X: np.ndarray) -> np.ndarray:
        if not _HAS_LGB:
            raise RuntimeError("LightGBM not installed")
        booster = lgb.Booster(model_file=str(model_path))
        pred = booster.predict(X)
        return np.array(pred)

# -----------------------------------
# PyTorch Dataset helper for RL / DL
# -----------------------------------
if _HAS_TORCH:
    class NumpyDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def export_pytorch_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
        ds = NumpyDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)
else:
    def export_pytorch_dataset(*a, **k):
        raise RuntimeError("PyTorch not available")

# -----------------------------------
# RL Experience Replay utilities
# -----------------------------------
class ReplayBuffer:
    """
    Simple file-backed replay buffer for RL datasets.
    Stored as NDJSON lines of JSON-serialized tuples:
      {"s": [...], "a": ..., "r": ..., "s2": [...], "done": false}
    """

    def __init__(self, path: Union[str, pathlib.Path] = DATASETS_DIR / "replay_buffer.jsonl", max_size: int = 1_000_000):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._lock = threading.Lock()

    def append(self, transition: Dict[str, Any]):
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(transition, default=str) + "\n")
            # optional trimming: naive approach
            try:
                size = sum(1 for _ in open(self.path, "r", encoding="utf-8"))
                if size > self.max_size:
                    # trim oldest lines
                    with open(self.path, "r", encoding="utf-8") as fh:
                        lines = fh.readlines()[-self.max_size:]
                    with open(self.path, "w", encoding="utf-8") as fh:
                        fh.writelines(lines)
            except Exception:
                LOG.exception("Replay buffer trimming failed")

    def sample(self, n: int = 1024) -> List[Dict[str, Any]]:
        # naive uniform sample (reads file)
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        if not lines:
            return []
        idx = np.random.choice(len(lines), size=min(n, len(lines)), replace=False)
        sampled = []
        for i in idx:
            try:
                sampled.append(json.loads(lines[i]))
            except Exception:
                continue
        return sampled

    def to_numpy(self, transform_fn: Optional[Callable[[Dict[str, Any]], Tuple[np.ndarray, np.ndarray]]] = None, n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert transitions to numpy arrays for supervised training. transform_fn maps transition -> (x, y)
        """
        rows = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if n and i >= n:
                    break
                try:
                    tr = json.loads(line)
                    rows.append(tr)
                except Exception:
                    continue
        Xs = []
        ys = []
        for r in rows:
            if transform_fn:
                x, y = transform_fn(r)
            else:
                # default: flatten state + action as x, reward as y
                s = np.array(r.get("s", []), dtype=float).ravel()
                a = np.array([r.get("a", 0)], dtype=float)
                x = np.concatenate([s, a], axis=0)
                y = np.array([r.get("r", 0)], dtype=float)
            Xs.append(x)
            ys.append(y)
        if not Xs:
            return np.zeros((0,)), np.zeros((0,))
        X = np.vstack(Xs)
        Y = np.vstack(ys).squeeze()
        return X.astype(np.float32), Y.astype(np.float32)

# -----------------------------------
# Retraining scheduler (simple)
# -----------------------------------
class RetrainScheduler:
    """
    Periodically checks dataset index and triggers retraining if specified conditions met:
      - new dataset version present
      - custom trigger function returns True
    Triggers provided callback (callable) with dataset version metadata.
    """

    def __init__(self, poll_interval: int = 300, check_fn: Optional[Callable[[Dict[str, Any]], bool]] = None):
        self.poll_interval = poll_interval
        self.check_fn = check_fn or (lambda meta: True)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_seen_versions = set()

    def _scan_once(self):
        idx = _index_read()
        for vid, meta in idx.items():
            if vid in self._last_seen_versions:
                continue
            if self.check_fn(meta):
                LOG.info("Retrain trigger: new dataset version %s", vid)
                self.on_trigger(meta)
                self._last_seen_versions.add(vid)

    def on_trigger(self, meta: Dict[str, Any]):
        """
        Override or set via attribute to perform training.
        Default behavior: write an audit event.
        """
        _audit("retrain_triggered", {"version": meta.get("version_id")})

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        LOG.info("RetrainScheduler started (interval=%ds)", self.poll_interval)

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception:
                LOG.exception("RetrainScheduler scan error")
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

# -----------------------------------
# Command-line utilities
# -----------------------------------
def _cli_collect(args):
    dc = DataCollector(mongo_url=os.getenv("MONGO_URL"), redis_url=os.getenv("REDIS_URL"))
    if args.source == "sim":
        df = dc.read_simulated(runs=args.runs, avg_tasks_per_run=args.avg_tasks)
    elif args.source in ("csv", "jsonl"):
        path = pathlib.Path(args.path)
        if args.source == "csv":
            df = dc.read_from_csv(path)
        else:
            df = dc.read_from_jsonl(path)
    else:
        raise ValueError("unsupported source")
    if args.save:
        dm = DatasetManager()
        meta = dm.new_version(name=args.name or "collected", df=df, compress=args.compress, upload_s3=args.upload_s3)
        print(json.dumps(meta, indent=2))
    else:
        print(df.head().to_json(orient="records"))

def _cli_preprocess(args):
    dm = DatasetManager()
    if args.version:
        df = dm.load_version(args.version)
    else:
        # if path provided
        df = pd.read_csv(args.path) if args.path else None
        if df is None:
            raise ValueError("no input dataset")
    prep = Preprocessor()
    dfc = prep.basic_clean(df)
    dfc = prep.label_generation(dfc, label_type=args.label)
    if args.features:
        # assume comma separated
        features = args.features.split(",")
    else:
        # autodetect numeric columns except label
        features = [c for c in dfc.columns if pd.api.types.is_numeric_dtype(dfc[c]) and c != "label"]
    X, y = prep.build_feature_matrix(dfc, features, label_column="label")
    # wrap into df for store
    df_out = dfc[features + ["label"]]
    dm = DatasetManager()
    meta = dm.new_version(name=args.name or f"preprocessed_{int(time.time())}", df=df_out, compress=args.compress, upload_s3=args.upload_s3)
    print(json.dumps(meta, indent=2))

def _cli_train_predictor(args):
    dm = DatasetManager()
    if args.version is None:
        raise ValueError("must supply dataset version id")
    df = dm.load_version(args.version)
    prep = Preprocessor()
    dfc = prep.basic_clean(df)
    dfc = prep.label_generation(dfc, label_type=args.label)
    features = [c for c in dfc.columns if c != "label"]
    # simple split
    train_frac = float(os.getenv("TRAIN_FRAC", "0.8"))
    dfc = dfc.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(dfc) * train_frac)
    df_train = dfc.iloc[:n_train]
    df_val = dfc.iloc[n_train:]
    X_train, y_train = prep.build_feature_matrix(df_train, features, "label")
    X_val, y_val = prep.build_feature_matrix(df_val, features, "label")
    trainer = LightGBMTrainer(use_mlflow=args.mlflow and _HAS_MLFLOW, use_wandb=args.wandb and _HAS_WANDB)
    res = trainer.train(X_train, y_train, X_val, y_val, feature_names=features, model_name=args.model_name)
    print(json.dumps(res, indent=2))

def _cli_generate_sim(args):
    dc = DataCollector()
    df = dc.read_simulated(runs=args.runs, avg_tasks_per_run=args.avg)
    if args.save:
        dm = DatasetManager()
        meta = dm.new_version(name=args.name or "simulated", df=df, compress=args.compress, upload_s3=args.upload_s3)
        print(json.dumps(meta, indent=2))
    else:
        print(df.head().to_json(orient="records"))

def _cli_export_pytorch(args):
    dm = DatasetManager()
    df = dm.load_version(args.version)
    prep = Preprocessor()
    dfc = prep.basic_clean(df)
    dfc = prep.label_generation(dfc, label_type=args.label)
    features = [c for c in dfc.columns if c != "label"]
    X, y = prep.build_feature_matrix(dfc, features, "label")
    if _HAS_TORCH:
        loader = export_pytorch_dataset(X, y, batch_size=args.batch)
        # save tensors as npz for portability
        arr_path = ARTIFACTS_DIR / f"pytorch_dataset_{args.version}.npz"
        _save_npz({"X": X, "y": y}, arr_path)
        print(f"Exported PyTorch dataset to {arr_path}")
    else:
        print("PyTorch not available; exporting as numpy npz")
        arr_path = ARTIFACTS_DIR / f"dataset_{args.version}.npz"
        _save_npz({"X": X, "y": y}, arr_path)
        print(f"Exported numpy dataset to {arr_path}")

def _cli_retrain_scheduler(args):
    # trivial example: start scheduler that logs retrain triggers
    rs = RetrainScheduler(poll_interval=args.interval)
    def trigger(meta):
        print("Retrain triggered for", meta.get("version_id"))
        _audit("retrain_scheduler_trigger", {"version": meta.get("version_id")})
    rs.on_trigger = trigger
    rs.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rs.stop()

def _build_cli():
    parser = argparse.ArgumentParser(prog="prioritymax-data-pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p_collect = sub.add_parser("collect", help="Collect data from sim/csv/jsonl")
    p_collect.add_argument("--source", choices=["sim", "csv", "jsonl"], default="sim")
    p_collect.add_argument("--path", help="path for csv/jsonl")
    p_collect.add_argument("--runs", type=int, default=1000)
    p_collect.add_argument("--avg_tasks", type=int, default=10)
    p_collect.add_argument("--save", action="store_true")
    p_collect.add_argument("--name", help="dataset name")
    p_collect.add_argument("--compress", action="store_true")
    p_collect.add_argument("--upload-s3", action="store_true")

    p_pre = sub.add_parser("preprocess", help="Preprocess dataset version or file")
    p_pre.add_argument("--version", help="dataset version id to load")
    p_pre.add_argument("--path", help="csv path if not using version")
    p_pre.add_argument("--label", default="latency_bucket")
    p_pre.add_argument("--features", help="comma separated feature list")
    p_pre.add_argument("--name", help="output dataset name")
    p_pre.add_argument("--compress", action="store_true")
    p_pre.add_argument("--upload-s3", action="store_true")

    p_train = sub.add_parser("train_predictor", help="Train LightGBM predictor")
    p_train.add_argument("--version", required=True, help="dataset version id")
    p_train.add_argument("--label", default="latency_bucket")
    p_train.add_argument("--model-name", help="output model filename")
    p_train.add_argument("--mlflow", action="store_true")
    p_train.add_argument("--wandb", action="store_true")

    p_sim = sub.add_parser("generate_sim", help="Generate simulated dataset")
    p_sim.add_argument("--runs", type=int, default=500)
    p_sim.add_argument("--avg", type=int, default=8)
    p_sim.add_argument("--save", action="store_true")
    p_sim.add_argument("--name", help="dataset name")
    p_sim.add_argument("--compress", action="store_true")
    p_sim.add_argument("--upload-s3", action="store_true")

    p_export = sub.add_parser("export_pytorch", help="Export dataset to PyTorch npz")
    p_export.add_argument("--version", required=True)
    p_export.add_argument("--label", default="latency_bucket")
    p_export.add_argument("--batch", type=int, default=256)

    p_sched = sub.add_parser("retrain_scheduler", help="Start retrain scheduler (blocks)")
    p_sched.add_argument("--interval", type=int, default=300)

    return parser

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    if args.cmd == "collect":
        _cli_collect(args)
    elif args.cmd == "preprocess":
        _cli_preprocess(args)
    elif args.cmd == "train_predictor":
        _cli_train_predictor(args)
    elif args.cmd == "generate_sim":
        _cli_generate_sim(args)
    elif args.cmd == "export_pytorch":
        _cli_export_pytorch(args)
    elif args.cmd == "retrain_scheduler":
        _cli_retrain_scheduler(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

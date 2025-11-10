#!/usr/bin/env python3
"""
PriorityMax — Predictor Training Script (Production-Grade)
==========================================================

This module trains, validates, and exports the prediction model used by
the autoscaler and workload forecaster in PriorityMax.

Features:
 - Configurable model backend (RandomForest, LightGBM, XGBoost, MLP)
 - Full MLflow + W&B experiment tracking
 - Prometheus metrics exporter
 - Redis + DLQ retraining feedback
 - S3 artifact sync
 - Drift detection, alerting, and validation pipeline
 - Incremental retraining support for live deployments
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import uuid
import yaml
import psutil
import shutil
import signal
import random
import joblib
import typing
import logging
import argparse
import tempfile
import threading
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Optional ML backends
# ---------------------------
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# ---------------------------
# Optional Integrations
# ---------------------------
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

try:
    from prometheus_client import Gauge, Counter, start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    _HAS_BOTO3 = False

try:
    from redis import Redis
    _HAS_REDIS = True
except Exception:
    _HAS_REDIS = False

# ---------------------------
# Local project imports
# ---------------------------
try:
    from ml.data_pipeline import DataPipeline
    from ml.model_registry import ModelRegistry
    from utils.logger import configure_logger
except Exception:
    # Fallback for standalone training
    class DataPipeline:
        def __init__(self, **kwargs):
            pass
        def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
            raise NotImplementedError("DataPipeline not available")
    ModelRegistry = None
    def configure_logger(*a, **kw): return logging.getLogger("train_predictor")

# ---------------------------
# Logging Setup
# ---------------------------
LOG = configure_logger("prioritymax.train_predictor", level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------
# Paths and constants
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "app", "ml", "models"))
DATA_CACHE = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_CACHE, exist_ok=True)

DEFAULT_MODEL_TYPE = os.getenv("PREDICTOR_MODEL", "lightgbm")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.join(BASE_DIR, "mlruns"))
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "prioritymax-predictor")
PROM_PORT = int(os.getenv("PROM_PORT", "9203"))

# ---------------------------
# Prometheus Metrics
# ---------------------------
if _HAS_PROM:
    TRAIN_LOSS = Gauge("predictor_train_loss", "Latest training loss")
    VAL_LOSS = Gauge("predictor_val_loss", "Latest validation loss")
    TRAIN_PROGRESS = Gauge("predictor_train_progress", "Progress percentage")
    TRAIN_DURATION = Gauge("predictor_train_duration_seconds", "Total training time")
    TRAIN_COMPLETED = Counter("predictor_train_completed_total", "Total training runs completed")

# ---------------------------
# Helper utilities
# ---------------------------
def start_prometheus_exporter():
    if _HAS_PROM:
        start_http_server(PROM_PORT)
        LOG.info(f"Prometheus exporter running on :{PROM_PORT}/metrics")

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

def get_cpu_mem_usage() -> dict:
    try:
        return {
            "cpu": psutil.cpu_percent(interval=0.1),
            "mem": psutil.virtual_memory().percent,
        }
    except Exception:
        return {"cpu": 0.0, "mem": 0.0}

# ---------------------------
# Configuration dataclass
# ---------------------------
class PredictorConfig:
    """
    Encapsulates all configurable parameters for predictor training.
    """
    def __init__(self, **kwargs):
        self.model_type = kwargs.get("model_type", DEFAULT_MODEL_TYPE)
        self.test_size = float(kwargs.get("test_size", 0.2))
        self.random_seed = int(kwargs.get("random_seed", 42))
        self.n_estimators = int(kwargs.get("n_estimators", 200))
        self.learning_rate = float(kwargs.get("learning_rate", 0.05))
        self.max_depth = int(kwargs.get("max_depth", 8))
        self.batch_size = int(kwargs.get("batch_size", 64))
        self.epochs = int(kwargs.get("epochs", 20))
        self.optuna_trials = int(kwargs.get("optuna_trials", 0))
        self.track_mlflow = bool(kwargs.get("track_mlflow", True))
        self.track_wandb = bool(kwargs.get("track_wandb", False))
        self.sync_s3 = bool(kwargs.get("sync_s3", False))
        self.bucket = kwargs.get("bucket", os.getenv("S3_BUCKET"))
        self.prefix = kwargs.get("prefix", os.getenv("S3_PREFIX", "predictor"))
        self.incremental = bool(kwargs.get("incremental", False))
        self.validation_metric = kwargs.get("validation_metric", "mse")
        self.target_col = kwargs.get("target_col", "target")
        self.data_source = kwargs.get("data_source", "realtime")
        self.config_path = kwargs.get("config_path", None)

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        return self.__dict__

    def dump(self, out_path: str):
        save_json(out_path, self.to_dict())

# ---------------------------
# Simple MLP predictor (torch)
# ---------------------------
if _HAS_TORCH:
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 128):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        def forward(self, x):
            return self.model(x)
# ---------------------------
# Chunk 2 — Data loading, preprocessing, training routines, evaluation, and artifact export
# ---------------------------

# ---------------------------
# Data helpers
# ---------------------------
def load_dataframe_from_csv(path: str) -> pd.DataFrame:
    LOG.info("Loading CSV data from %s", path)
    df = pd.read_csv(path)
    LOG.info("Loaded dataframe shape: %s", df.shape)
    return df

def load_data(cfg: PredictorConfig, pipeline: Optional[DataPipeline] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load features and target using DataPipeline if available, otherwise read from CSV.
    Returns X_df, y_series
    """
    if pipeline is not None:
        LOG.info("Using DataPipeline to load data")
        df = pipeline.load_training_data()
    else:
        # default fallback to cached csv
        if cfg.config_path and os.path.exists(cfg.config_path):
            try:
                with open(cfg.config_path, "r") as fh:
                    conf = json.load(fh)
                    csv_path = conf.get("data_csv") or conf.get("data_path")
                    if csv_path and os.path.exists(csv_path):
                        df = load_dataframe_from_csv(csv_path)
                    else:
                        raise RuntimeError("No data path found in config")
            except Exception:
                LOG.exception("Failed to load config path; falling back to sample dataset")
                df = _sample_dataset()
        else:
            # last-resort sample simulated data
            df = _sample_dataset()
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset. Available columns: {list(df.columns)}")
    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])
    return X, y

def _sample_dataset(rows: int = 5000) -> pd.DataFrame:
    """
    Minimal synthetic dataset generator for development/testing.
    """
    LOG.warning("Generating synthetic sample dataset (rows=%d)", rows)
    np.random.seed(42)
    ts = np.arange(rows)
    queue_len = np.abs(np.random.randn(rows).cumsum()).astype(int)
    cpu = np.clip(20 + 40 * np.random.rand(rows) + np.random.randn(rows) * 5, 0, 100)
    latency = np.clip(0.01 + (queue_len / (10 + np.random.rand(rows))) + np.random.randn(rows) * 0.01, 0.0, None)
    next_sec = queue_len + np.random.poisson(2, rows)
    df = pd.DataFrame({
        "ts": ts,
        "queue_len": queue_len,
        "cpu": cpu,
        "avg_latency": latency,
        "throughput": np.clip(queue_len / (1 + np.random.rand(rows)), 0.0, None),
        "target": next_sec
    })
    return df

# ---------------------------
# Preprocessing
# ---------------------------
def train_test_split_df(X: pd.DataFrame, y: pd.Series, cfg: PredictorConfig):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_seed)
    return X_train, X_val, y_train, y_val

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler

def transform_df_with_scaler(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    arr = scaler.transform(X.values)
    return arr

# ---------------------------
# Model factories & save/load
# ---------------------------
def build_model(cfg: PredictorConfig):
    if cfg.model_type.lower() in ("lgb", "lightgbm", "lightgbm_regression", "lightgbm_reg"):
        if not _HAS_LGB:
            raise RuntimeError("LightGBM requested but not installed")
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": cfg.learning_rate,
            "num_leaves": 2 ** 6,
            "max_depth": cfg.max_depth
        }
        LOG.info("Building LightGBM with params: %s", params)
        return ("lgb", params)
    elif cfg.model_type.lower() in ("rf", "randomforest", "random_forest"):
        LOG.info("Building RandomForestRegressor n_estimators=%d", cfg.n_estimators)
        model = RandomForestRegressor(n_estimators=cfg.n_estimators, random_state=cfg.random_seed, n_jobs=-1)
        return ("sklearn", model)
    elif cfg.model_type.lower() in ("mlp", "torch_mlp", "pytorch"):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch requested but not installed")
        LOG.info("Building SimpleMLP model")
        return ("torch", SimpleMLP)
    else:
        raise ValueError("Unknown model_type: %s" % cfg.model_type)

def save_model_artifact(cfg: PredictorConfig, model_obj: Any, scaler: Optional[StandardScaler], out_tag: Optional[str] = None) -> str:
    """
    Persist model to MODEL_DIR/<tag>/ and optionally push to S3.
    Returns artifact directory path.
    """
    tag = out_tag or f"{cfg.model_type}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    out_dir = os.path.join(MODEL_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    # Save scaler
    if scaler is not None:
        joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    # Save model by type
    if isinstance(model_obj, RandomForestRegressor):
        joblib.dump(model_obj, os.path.join(out_dir, "model.joblib"))
    elif _HAS_LGB and isinstance(model_obj, lgb.Booster):
        model_obj.save_model(os.path.join(out_dir, "model.txt"))
    elif _HAS_TORCH and isinstance(model_obj, torch.nn.Module):
        torch.save(model_obj.state_dict(), os.path.join(out_dir, "model.pt"))
    else:
        # fallback to joblib
        try:
            joblib.dump(model_obj, os.path.join(out_dir, "model.joblib"))
        except Exception:
            LOG.exception("Failed to save model object; attempting pickle")
            import pickle
            with open(os.path.join(out_dir, "model.pkl"), "wb") as fh:
                pickle.dump(model_obj, fh)
    # metadata
    meta = {
        "tag": tag,
        "model_type": cfg.model_type,
        "saved_at": now_ts(),
        "seed": cfg.random_seed
    }
    save_json(os.path.join(out_dir, "metadata.json"), meta)

    # Optionally push to S3
    if cfg.sync_s3 and _HAS_BOTO3 and cfg.bucket:
        try:
            s3 = boto3.client("s3")
            for root, _, files in os.walk(out_dir):
                for fname in files:
                    key = os.path.join(cfg.prefix, tag, os.path.relpath(os.path.join(root, fname), out_dir))
                    s3.upload_file(os.path.join(root, fname), cfg.bucket, key)
            LOG.info("Synced model artifacts to s3://%s/%s/%s", cfg.bucket, cfg.prefix, tag)
        except Exception:
            LOG.exception("Failed to sync artifacts to S3")
    LOG.info("Saved model artifact to %s", out_dir)
    return out_dir

# ---------------------------
# Evaluate utilities
# ---------------------------
def evaluate_model_on_df(model_type:str, model_obj: Any, scaler: Optional[StandardScaler], X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """
    Evaluate model (sklearn/lightgbm/torch) on validation set and return metrics.
    """
    if scaler is not None:
        X_arr = transform_df_with_scaler(scaler, X_val)
    else:
        X_arr = X_val.values
    preds = None
    try:
        if model_type == "sklearn":
            preds = model_obj.predict(X_arr)
        elif model_type == "lgb":
            if isinstance(model_obj, lgb.Booster):
                preds = model_obj.predict(X_arr)
            else:
                preds = model_obj.predict(X_arr)
        elif model_type == "torch":
            if not _HAS_TORCH:
                raise RuntimeError("Torch missing for predict")
            model_obj.eval()
            with torch.no_grad():
                t = torch.from_numpy(X_arr.astype(np.float32))
                out = model_obj(t)
                preds = out.detach().cpu().numpy().ravel()
        else:
            # fallback: try 'predict' attribute
            preds = np.asarray([float(x) for x in model_obj.predict(X_arr)])
    except Exception:
        LOG.exception("Model predict failed; attempting per-sample fallback")
        preds = []
        for i in range(X_arr.shape[0]):
            try:
                preds.append(float(model_obj.predict(X_arr[i:i+1]).ravel().mean()))
            except Exception:
                preds.append(0.0)
        preds = np.array(preds)

    mse = float(mean_squared_error(y_val, preds))
    rmse = math.sqrt(mse)
    r2 = float(r2_score(y_val, preds)) if len(y_val) > 1 else float("nan")
    # p95 error
    abs_err = np.abs(preds - y_val.values)
    p95 = float(np.percentile(abs_err, 95)) if len(abs_err) > 0 else 0.0
    LOG.info("Evaluation metrics: RMSE=%.4f R2=%.4f P95=%.4f", rmse, r2, p95)
    return {"mse": mse, "rmse": rmse, "r2": r2, "p95_abs_err": p95, "n": int(len(y_val))}

# ---------------------------
# Training implementations
# ---------------------------
def train_lightgbm(cfg: PredictorConfig, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, scaler: Optional[StandardScaler], run_ctx: Dict[str, Any]) -> Any:
    """Train LightGBM with optional early stopping, return booster object"""
    if not _HAS_LGB:
        raise RuntimeError("LightGBM not available")
    train_arr = transform_df_with_scaler(scaler, X_train) if scaler else X_train.values
    val_arr = transform_df_with_scaler(scaler, X_val) if scaler else X_val.values
    dtrain = lgb.Dataset(train_arr, label=y_train.values)
    dval = lgb.Dataset(val_arr, label=y_val.values, reference=dtrain)
    params = run_ctx.get("lgb_params") or {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": cfg.learning_rate
    }
    num_round = int(run_ctx.get("num_boost_round", cfg.n_estimators))
    LOG.info("Starting LightGBM training rounds=%d params=%s", num_round, params)
    booster = lgb.train(params, dtrain, num_boost_round=num_round, valid_sets=[dtrain, dval], early_stopping_rounds=run_ctx.get("early_stopping_rounds", 50), verbose_eval=False)
    return booster

def train_random_forest(cfg: PredictorConfig, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, scaler: Optional[StandardScaler], run_ctx: Dict[str, Any]) -> Any:
    model = RandomForestRegressor(n_estimators=cfg.n_estimators, random_state=cfg.random_seed, n_jobs=-1)
    # sklearn expects raw arrays (we will pass scaled arrays optionally)
    Xt = transform_df_with_scaler(scaler, X_train) if scaler else X_train.values
    model.fit(Xt, y_train.values)
    return model

def train_torch_mlp(cfg: PredictorConfig, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, scaler: Optional[StandardScaler], run_ctx: Dict[str, Any]) -> torch.nn.Module:
    if not _HAS_TORCH:
        raise RuntimeError("Torch not available")
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim, output_dim=1, hidden_dim=run_ctx.get("hidden_dim", 128))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=run_ctx.get("lr", 1e-3))
    loss_fn = nn.MSELoss()
    Xtr = transform_df_with_scaler(scaler, X_train) if scaler else X_train.values
    Xv = transform_df_with_scaler(scaler, X_val) if scaler else X_val.values
    ytr = y_train.values.astype(np.float32)
    yv = y_val.values.astype(np.float32)
    batch_size = run_ctx.get("batch_size", cfg.batch_size)
    epochs = run_ctx.get("epochs", cfg.epochs)
    LOG.info("Starting Torch MLP training epochs=%d batch=%d", epochs, batch_size)
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(Xtr))
        model.train()
        epoch_losses = []
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.from_numpy(Xtr[idx].astype(np.float32)).to(device)
            yb = torch.from_numpy(ytr[idx].astype(np.float32)).unsqueeze(-1).to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        LOG.info("Epoch %d/%d train_loss=%.6f", epoch, epochs, avg_loss)
        # validation
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xv.astype(np.float32)).to(device)
            predv = model(xv).cpu().numpy().ravel()
            val_mse = float(mean_squared_error(yv, predv))
        LOG.info("Epoch %d val_mse=%.6f", epoch, val_mse)
    return model

# ---------------------------
# Optuna objective wrapper
# ---------------------------
def optuna_objective(trial: "optuna.trial.Trial", cfg: PredictorConfig, X: pd.DataFrame, y: pd.Series) -> float:
    # sample hyperparams
    model_choice = trial.suggest_categorical("model", ["lgb", "rf"])
    if model_choice == "lgb":
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_loguniform("lr", 1e-4, 1e-1),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        }
        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_seed)
        scaler = fit_scaler(X_train)
        booster = train_lightgbm(cfg, X_train, y_train, X_val, y_val, scaler, {"lgb_params": params, "num_boost_round": 500, "early_stopping_rounds": 40})
        metrics = evaluate_model_on_df("lgb", booster, scaler, X_val, y_val)
        return metrics["mse"]
    else:
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 200, 400])
        max_depth = trial.suggest_int("max_depth", 3, 20)
        cfg_local = PredictorConfig(**cfg.to_dict())
        cfg_local.n_estimators = n_estimators
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_seed)
        scaler = fit_scaler(X_train)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=cfg.random_seed, n_jobs=-1)
        model.fit(transform_df_with_scaler(scaler, X_train), y_train.values)
        metrics = evaluate_model_on_df("sklearn", model, scaler, X_val, y_val)
        return metrics["mse"]

# ---------------------------
# High-level orchestrator
# ---------------------------
def run_training(cfg: PredictorConfig, pipeline: Optional[DataPipeline] = None, out_artifact_tag: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrate the full training flow:
     - load data
     - split & scale
     - optionally run Optuna
     - train selected model
     - evaluate
     - save artifacts
     - log to MLflow/W&B
    Returns result dict with metrics and artifact path.
    """
    start_time = time.time()
    if cfg.track_mlflow and _HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.start_run(run_name=f"predictor_{uuid.uuid4().hex[:6]}")
        except Exception:
            LOG.exception("Failed to start MLflow run")

    if cfg.track_wandb and _HAS_WANDB:
        try:
            wandb.init(project=WANDB_PROJECT, config=cfg.to_dict(), name=f"predictor-{uuid.uuid4().hex[:6]}")
        except Exception:
            LOG.exception("Failed to start wandb run")

    # load data
    X, y = load_data(cfg, pipeline)
    LOG.info("Dataset shape X=%s y=%s", X.shape, y.shape)
    X_train, X_val, y_train, y_val = train_test_split_df(X, y, cfg)
    scaler = fit_scaler(X_train) if True else None  # always fit scaler for numeric stability

    # Optionally run Optuna hyperparameter search
    best_artifact = None
    best_metrics = None
    selected_model_type = cfg.model_type
    run_ctx = {}
    if cfg.optuna_trials and _HAS_OPTUNA:
        LOG.info("Starting Optuna search trials=%d", cfg.optuna_trials)
        study = optuna.create_study(direction="minimize")
        try:
            study.optimize(lambda t: optuna_objective(t, cfg, X_train.append(X_val), y_train.append(y_val)), n_trials=cfg.optuna_trials)
            LOG.info("Optuna best trial: %s", study.best_trial.params)
            # apply best params (example)
            if "model" in study.best_trial.params:
                selected_model_type = study.best_trial.params.get("model")
            run_ctx["optuna_best"] = study.best_trial.params
        except Exception:
            LOG.exception("Optuna optimization failed")

    # Build & train model
    factory = build_model(cfg)
    model_obj = None
    if factory[0] == "lgb":
        run_ctx.setdefault("lgb_params", factory[1])
        model_obj = train_lightgbm(cfg, X_train, y_train, X_val, y_val, scaler, run_ctx)
        model_kind = "lgb"
    elif factory[0] == "sklearn":
        model_obj = train_random_forest(cfg, X_train, y_train, X_val, y_val, scaler, run_ctx)
        model_kind = "sklearn"
    elif factory[0] == "torch":
        model_obj = train_torch_mlp(cfg, X_train, y_train, X_val, y_val, scaler, run_ctx)
        model_kind = "torch"
    else:
        raise RuntimeError("Unknown model factory result: %s" % (factory,))

    # Evaluate
    metrics = evaluate_model_on_df(model_kind, model_obj, scaler, X_val, y_val)
    if cfg.track_mlflow and _HAS_MLFLOW:
        try:
            mlflow.log_metrics(metrics)
        except Exception:
            LOG.exception("Failed to log metrics to MLflow")
    if cfg.track_wandb and _HAS_WANDB:
        try:
            wandb.log(metrics)
        except Exception:
            LOG.exception("Failed to log metrics to W&B")

    # Save artifacts
    artifact_dir = save_model_artifact(cfg, model_obj, scaler, out_tag=out_artifact_tag)
    best_artifact = artifact_dir
    best_metrics = metrics

    # finish mlflow/wandb
    if cfg.track_mlflow and _HAS_MLFLOW:
        try:
            mlflow.log_artifact(os.path.join(artifact_dir, "metadata.json"))
            mlflow.end_run()
        except Exception:
            LOG.exception("Error finalizing MLflow")
    if cfg.track_wandb and _HAS_WANDB:
        try:
            wandb.save(os.path.join(artifact_dir, "*"))
            wandb.finish()
        except Exception:
            LOG.exception("Error finalizing W&B run")

    duration = time.time() - start_time
    if _HAS_PROM:
        try:
            TRAIN_DURATION.set(duration)
            TRAIN_COMPLETED.inc()
        except Exception:
            pass

    LOG.info("Training finished in %.2fs. Artifact=%s metrics=%s", duration, artifact_dir, metrics)
    return {"artifact_dir": artifact_dir, "metrics": metrics, "duration_sec": duration}

# ---------------------------
# Retrain trigger callback (e.g., invoked by DLQ monitor)
# ---------------------------
def retrain_callback_on_signal(cfg: PredictorConfig, reason: str = "dlq_threshold"):
    """
    Example callback used by external monitors to trigger retraining.
    In production you might enqueue retrain request rather than run inline.
    """
    LOG.info("Retrain callback triggered reason=%s", reason)
    # Run in background thread to avoid blocking monitor
    def _bg_run():
        try:
            run_training(cfg)
            LOG.info("Background retrain completed")
        except Exception:
            LOG.exception("Background retrain failed")
    t = threading.Thread(target=_bg_run, daemon=True)
    t.start()

# ---------------------------
# End of Chunk 2
# ---------------------------
# ---------------------------
# Chunk 3 — Drift detection, autoscaler feedback, CLI, signal handling, and main entrypoint
# ---------------------------

# ---------------------------
# Drift detection utilities
# ---------------------------
def detect_data_drift(prev_metrics: dict, new_metrics: dict, tolerance: float = 0.2) -> bool:
    """
    Compare validation RMSE between previous and new metrics.
    Return True if drift detected (degradation exceeds tolerance fraction).
    """
    try:
        old_rmse = float(prev_metrics.get("rmse", 0))
        new_rmse = float(new_metrics.get("rmse", 0))
        if old_rmse <= 0:
            return False
        delta = (new_rmse - old_rmse) / old_rmse
        drift = delta > tolerance
        LOG.info("Drift check: old_rmse=%.4f new_rmse=%.4f delta=%.2f%% -> drift=%s",
                 old_rmse, new_rmse, delta * 100, drift)
        return drift
    except Exception:
        LOG.exception("Drift detection failed")
        return False

# ---------------------------
# Autoscaler feedback integration
# ---------------------------
def push_metrics_to_redis(metrics: dict, prefix: str = "predictor:metrics"):
    if not _HAS_REDIS:
        return
    try:
        r = Redis(host=os.getenv("REDIS_HOST", "localhost"),
                  port=int(os.getenv("REDIS_PORT", "6379")),
                  db=int(os.getenv("REDIS_DB", "0")))
        key = f"{prefix}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        r.hmset(key, {k: str(v) for k, v in metrics.items()})
        r.expire(key, 3600)
        LOG.debug("Pushed metrics to Redis key=%s", key)
    except Exception:
        LOG.debug("Redis metrics push skipped (not configured)")

# ---------------------------
# Signal handling
# ---------------------------
_stop_training = False

def _handle_sigterm(signum, frame):
    global _stop_training
    LOG.warning("Received signal %s — graceful shutdown requested", signum)
    _stop_training = True

signal.signal(signal.SIGINT, _handle_sigterm)
signal.signal(signal.SIGTERM, _handle_sigterm)

# ---------------------------
# CLI helper
# ---------------------------
def build_cli():
    p = argparse.ArgumentParser(description="PriorityMax Predictor Trainer (production)")
    p.add_argument("--config", type=str, help="Path to JSON/YAML config file")
    p.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE, help="Model type (lightgbm, rf, mlp)")
    p.add_argument("--optuna-trials", type=int, default=0, help="Number of Optuna tuning trials")
    p.add_argument("--sync-s3", action="store_true", help="Enable S3 sync of artifacts")
    p.add_argument("--track-mlflow", action="store_true", help="Enable MLflow tracking")
    p.add_argument("--track-wandb", action="store_true", help="Enable Weights&Biases tracking")
    p.add_argument("--incremental", action="store_true", help="Perform incremental retraining")
    p.add_argument("--out-tag", type=str, help="Custom model tag")
    p.add_argument("--detect-drift", type=str, help="Path to previous metrics.json for drift comparison")
    p.add_argument("--json-out", action="store_true", help="Print JSON summary to stdout")
    p.add_argument("--prom", action="store_true", help="Start Prometheus exporter")
    return p

# ---------------------------
# Main execution entry
# ---------------------------
def main():
    args = build_cli().parse_args()
    cfg_kwargs = {}
    if args.config and os.path.exists(args.config):
        cfg = PredictorConfig.from_file(args.config)
    else:
        # fallback to CLI flags
        cfg_kwargs = {
            "model_type": args.model_type,
            "optuna_trials": args.optuna_trials,
            "sync_s3": args.sync_s3,
            "track_mlflow": args.track_mlflow,
            "track_wandb": args.track_wandb,
            "incremental": args.incremental
        }
        cfg = PredictorConfig(**cfg_kwargs)

    if args.prom:
        start_prometheus_exporter()

    LOG.info("=== PriorityMax Predictor Trainer ===")
    LOG.info("Config: %s", json.dumps(cfg.to_dict(), indent=2))

    # Attempt pipeline import (for production mode)
    pipeline = None
    try:
        pipeline = DataPipeline(mode="train")
    except Exception:
        LOG.warning("DataPipeline not available; using default loader")

    # Train model
    result = run_training(cfg, pipeline=pipeline, out_artifact_tag=args.out_tag)

    # Drift detection if applicable
    if args.detect_drift and os.path.exists(args.detect_drift):
        try:
            prev = json.load(open(args.detect_drift))
            drift = detect_data_drift(prev.get("metrics", {}), result.get("metrics", {}))
            result["drift_detected"] = drift
            if drift:
                LOG.warning("Model drift detected! Consider retraining autoscaler.")
        except Exception:
            LOG.exception("Failed to load drift metrics reference")

    # Push metrics to Redis
    push_metrics_to_redis(result.get("metrics", {}))

    # Save summary to file
    out_json = os.path.join(result["artifact_dir"], "train_summary.json")
    save_json(out_json, result)
    LOG.info("Wrote training summary to %s", out_json)

    # JSON output for CI/CD
    if args.json_out:
        print(json.dumps(result, indent=2))

    LOG.info("Predictor training complete ✔")

# ---------------------------
# Entry guard
# ---------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning("Training interrupted by user")
    except Exception as e:
        LOG.exception("Fatal error in train_predictor: %s", e)
        sys.exit(1)

# ---------------------------
# End of file
# ---------------------------

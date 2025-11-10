#!/usr/bin/env python3
"""
tests/test_train_predictor.py
=============================

End-to-end test suite for scripts/train_predictor.py

Validates:
 - synthetic dataset training
 - artifact creation & metadata
 - metrics correctness
 - Prometheus exporter
 - Redis metrics push (if available)
 - incremental retrain and drift detection logic

Run:
    pytest -v tests/test_train_predictor.py
"""

import os
import sys
import json
import tempfile
import time
import shutil
import signal
import logging
import pytest
from pathlib import Path

# Ensure root import
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

import train_predictor as tp


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Silence overly verbose loggers."""
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("prioritymax").setLevel(logging.INFO)


@pytest.fixture()
def temp_env(tmp_path):
    """Provide isolated dirs for model/artifact caching."""
    model_dir = tmp_path / "models"
    data_cache = tmp_path / "data"
    model_dir.mkdir()
    data_cache.mkdir()
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["DATA_CACHE"] = str(data_cache)
    return model_dir, data_cache


@pytest.mark.asyncio
async def test_synthetic_training_end_to_end(temp_env):
    """Train on synthetic dataset and assert artifacts + metrics."""
    model_dir, _ = temp_env
    cfg = tp.PredictorConfig(model_type="rf", track_mlflow=False, track_wandb=False)

    result = tp.run_training(cfg)

    # Assert output structure
    assert "artifact_dir" in result
    out_dir = Path(result["artifact_dir"])
    assert out_dir.exists()
    assert (out_dir / "metadata.json").exists()

    # Assert metrics correctness
    metrics = result.get("metrics", {})
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0
    assert metrics["p95_abs_err"] >= 0

    # Assert summary file
    summary_path = out_dir / "train_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert "duration_sec" in data

    # Prometheus exporter (optional)
    if tp._HAS_PROM:
        tp.start_prometheus_exporter()
        time.sleep(1.0)  # allow server to start
        assert tp.PROM_PORT > 0

    # Redis metrics push
    if tp._HAS_REDIS:
        tp.push_metrics_to_redis(metrics)
        # Redis keys may expire but ensure no exception


def test_drift_detection_no_drift():
    """Drift detection should return False for similar metrics."""
    m1 = {"rmse": 1.0}
    m2 = {"rmse": 1.05}
    assert tp.detect_data_drift(m1, m2, tolerance=0.2) is False


def test_drift_detection_drift_detected():
    """Drift detection should detect >20% degradation."""
    m1 = {"rmse": 1.0}
    m2 = {"rmse": 1.5}
    assert tp.detect_data_drift(m1, m2, tolerance=0.2) is True


def test_prometheus_exporter_runs(monkeypatch):
    """Prometheus exporter should start without error."""
    if not tp._HAS_PROM:
        pytest.skip("Prometheus not installed")
    monkeypatch.setenv("PROM_PORT", "9204")
    tp.start_prometheus_exporter()
    time.sleep(1.0)
    assert tp.PROM_PORT == 9204


def test_retrain_callback_thread(temp_env):
    """Retrain callback should spawn a background thread."""
    cfg = tp.PredictorConfig(model_type="rf")
    tp.retrain_callback_on_signal(cfg, reason="test_retrain")
    # Give it some time to start
    time.sleep(2.0)
    # Thread is daemonized, cannot assert join but should not raise


def test_graceful_signal_handling(monkeypatch):
    """Simulate SIGTERM and ensure flag set."""
    monkeypatch.setattr(tp, "_stop_training", False)
    tp._handle_sigterm(signal.SIGTERM, None)
    assert tp._stop_training is True


def test_cli_entry(tmp_path, monkeypatch):
    """End-to-end CLI execution using synthetic data."""
    config_path = tmp_path / "predictor_config.json"
    cfg = {
        "model_type": "rf",
        "track_mlflow": False,
        "track_wandb": False,
        "n_estimators": 5,
        "target_col": "target"
    }
    config_path.write_text(json.dumps(cfg))
    argv = [
        "train_predictor.py",
        "--config", str(config_path),
        "--json-out"
    ]
    monkeypatch.setattr(sys, "argv", argv)
    tp.main()  # should complete without exceptions
    # Verify at least one model dir created
    out_dirs = list(Path(os.getenv("MODEL_DIR")).glob("*"))
    assert len(out_dirs) > 0


@pytest.mark.parametrize("model_type", ["rf", "lightgbm", "mlp"])
def test_multiple_model_backends(temp_env, model_type):
    """Smoke test different model backends."""
    model_dir, _ = temp_env
    cfg = tp.PredictorConfig(model_type=model_type, track_mlflow=False, track_wandb=False)
    try:
        result = tp.run_training(cfg)
        assert "artifact_dir" in result
        assert Path(result["artifact_dir"]).exists()
    except RuntimeError as e:
        # skip if missing dependency (e.g. torch not installed)
        if "not available" in str(e).lower():
            pytest.skip(f"{model_type} backend not installed")


def test_artifact_persistence(temp_env):
    """Ensure model artifacts are persisted correctly."""
    model_dir, _ = temp_env
    cfg = tp.PredictorConfig(model_type="rf")
    result = tp.run_training(cfg)
    artifact_dir = Path(result["artifact_dir"])
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "train_summary.json").exists()
    LOG = logging.getLogger("test")
    LOG.info("Artifacts verified at %s", artifact_dir)

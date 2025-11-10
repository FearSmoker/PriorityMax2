"""
PriorityMax Pytest Configuration
--------------------------------

Centralized fixtures and stubs for all tests.

Features:
 - Global MLflow & W&B stubs (no network I/O)
 - Temporary working directory isolation
 - Async event loop fixture for asyncio-based tests
 - Auto-clean environment variables
 - Logging config to keep CI output clean
"""

import os
import sys
import json
import asyncio
import tempfile
import logging
import pathlib
import types
import pytest

# -----------------------------------------------------------------------------
# Logging setup for tests
# -----------------------------------------------------------------------------
LOG = logging.getLogger("prioritymax.tests")
LOG.setLevel(logging.WARNING)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# -----------------------------------------------------------------------------
# Global environment sanitization
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def clean_env_before_tests():
    """
    Clear out environment variables that could interfere with CI runs.
    """
    for var in [
        "MLFLOW_TRACKING_URI",
        "WANDB_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_BUCKET",
        "PROMETHEUS_PORT",
        "REDIS_URL"
    ]:
        os.environ.pop(var, None)
    # ensure deterministic locale/time
    os.environ["TZ"] = "UTC"
    yield

# -----------------------------------------------------------------------------
# Temporary directory for all I/O-heavy tests
# -----------------------------------------------------------------------------
@pytest.fixture(scope="function")
def temp_workdir(tmp_path, monkeypatch):
    """
    Create an isolated working directory for each test.
    Automatically switches CWD to this directory and restores afterward.
    """
    old_cwd = os.getcwd()
    tmp_dir = tmp_path / "workdir"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_dir)
    yield tmp_dir
    os.chdir(old_cwd)

# -----------------------------------------------------------------------------
# Asyncio event loop for async tests
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop():
    """
    Create an asyncio event loop for tests needing async operations.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# -----------------------------------------------------------------------------
# MLflow + WandB Stub Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def stub_mlflow(monkeypatch):
    """
    Replace MLflow APIs with lightweight in-memory stubs.
    Prevents network access and disk writes during testing.
    """
    class _MLFlowStub:
        def __init__(self):
            self.runs = []
            self.metrics = []

        def set_tracking_uri(self, *a, **kw): pass
        def set_experiment(self, *a, **kw): pass
        def start_run(self, *a, **kw): self.runs.append({"id": len(self.runs) + 1})
        def log_metric(self, key, value, step=None):
            self.metrics.append((key, value, step))
        def log_metrics(self, dct, step=None):
            for k, v in dct.items():
                self.metrics.append((k, v, step))
        def end_run(self):
            LOG.debug("MLflow stub: end_run()")

    stub = _MLFlowStub()
    monkeypatch.setitem(sys.modules, "mlflow", stub)
    return stub

@pytest.fixture(scope="session", autouse=True)
def stub_wandb(monkeypatch):
    """
    Replace WandB APIs with local mocks.
    Prevents real API calls and handles logging locally.
    """
    class _WandBStub:
        def __init__(self):
            self.logs = []
            self.runs = []

        def init(self, project=None, name=None, config=None):
            self.runs.append({"project": project, "name": name, "config": config})
            LOG.debug(f"WandB stub initialized: {project}/{name}")

        def log(self, metrics):
            self.logs.append(metrics)

        def finish(self):
            LOG.debug("WandB stub: finish()")

        def save(self, path):
            LOG.debug(f"WandB stub: save({path})")

    stub = _WandBStub()
    monkeypatch.setitem(sys.modules, "wandb", stub)
    return stub

# -----------------------------------------------------------------------------
# Dummy boto3 stub (for S3 uploads in evaluation tests)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def stub_boto3(monkeypatch):
    """
    Stub boto3 client to avoid AWS calls.
    """
    class _S3ClientStub:
        def __init__(self): self.uploaded = []
        def upload_file(self, file, bucket, key):
            LOG.debug(f"S3Stub upload_file({file}, {bucket}, {key})")
            self.uploaded.append({"bucket": bucket, "key": key, "file": file})

    class _Boto3Stub:
        def client(self, name):
            if name == "s3":
                return _S3ClientStub()
            raise ValueError(f"Unknown client {name}")

    stub = _Boto3Stub()
    monkeypatch.setitem(sys.modules, "boto3", stub)
    return stub

# -----------------------------------------------------------------------------
# Example fixture: FakePredictor for fast predictor tests
# -----------------------------------------------------------------------------
@pytest.fixture
def fake_predictor():
    """
    A lightweight fake predictor class for predictor-related tests.
    Returns constant predictions for deterministic unit testing.
    """
    class FakePredictor:
        def __init__(self):
            self.name = "fake"
        def predict(self, X):
            return [42.0 for _ in range(len(X))]
        def predict_batch(self, X):
            return [42.0 for _ in range(len(X))]
        def explain(self, X):
            return {"feat1": 0.5, "feat2": -0.1}
    return FakePredictor()

# -----------------------------------------------------------------------------
# Auto-use fixture: silence noisy libraries during CI
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def silence_external_lib_logs():
    """
    Reduce log noise from asyncio, FastAPI, and HTTPX during test runs.
    """
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    yield

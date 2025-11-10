# tests/test_train_rl_eval.py
import sys
import os
import json
import tempfile
import types
import pathlib
import pytest
import builtins

# Ensure project root is on sys.path so `scripts.train_rl_eval` can be imported.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the module under test
try:
    from scripts import train_rl_eval as treval
except Exception as e:
    # If import fails, raise with helpful message
    raise ImportError(f"Failed to import scripts.train_rl_eval: {e}")

# Basic smoke tests using monkeypatch to avoid heavy dependencies
@pytest.fixture
def tmp_result_dir(tmp_path, monkeypatch):
    d = tmp_path / "results"
    d.mkdir()
    yield str(d)

def make_min_cfg(tmp_dir: str, **overrides):
    """
    Create a simple SimpleNamespace-like config object compatible with run_evaluation.
    Only fields referenced by run_evaluation are provided.
    """
    cfg = types.SimpleNamespace()
    cfg.seed = overrides.get("seed", 42)
    cfg.device = overrides.get("device", "cpu")
    cfg.s3_bucket = overrides.get("s3_bucket", None)
    cfg.s3_prefix = overrides.get("s3_prefix", None)
    cfg.eval_episodes = overrides.get("eval_episodes", 3)
    cfg.max_steps = overrides.get("max_steps", 200)
    cfg.out = overrides.get("out", os.path.join(tmp_dir, "eval_result.json"))
    cfg.mlflow = overrides.get("mlflow", False)
    cfg.wandb = overrides.get("wandb", False)
    cfg.stop_if_below_mean_reward = overrides.get("stop_if_below_mean_reward", None)
    return cfg

def test_run_evaluation_success(monkeypatch, tmp_path):
    """
    Simulate a successful evaluation run by monkeypatching:
    - build_policy_for_eval -> returns dummy policy and model_dir
    - run_vectorized_evaluation -> returns deterministic metrics
    - safe_mlflow_init / safe_wandb_init / upload_results_to_s3 -> no-ops
    """
    tmp_dir = str(tmp_path)
    cfg = make_min_cfg(tmp_dir, eval_episodes=5, max_steps=100)

    # Create deterministic fake evaluation result
    fake_eval_result = {
        "reward_mean": 123.45,
        "reward_std": 1.23,
        "reward_p95": 125.0,
        "episodes_completed": 5,
        "per_episode": [120.0, 124.0, 123.0, 126.0, 123.0]
    }

    # Monkeypatch heavy functions
    monkeypatch.setattr(treval, "build_policy_for_eval", lambda cfg, tmp_dir=None: ("dummy_policy", "/model/dir"))
    monkeypatch.setattr(treval, "run_vectorized_evaluation", lambda policy, env_factory, episodes, max_steps, n_workers: fake_eval_result)
    monkeypatch.setattr(treval, "safe_mlflow_init", lambda cfg, run_name: None)
    monkeypatch.setattr(treval, "safe_wandb_init", lambda cfg, run_name: None)
    monkeypatch.setattr(treval, "safe_mlflow_log_metrics", lambda metrics: None)
    monkeypatch.setattr(treval, "safe_wandb_log", lambda metrics: None)
    monkeypatch.setattr(treval, "safe_mlflow_end", lambda: None)
    monkeypatch.setattr(treval, "safe_wandb_finish", lambda: None)
    # Avoid real S3 calls
    monkeypatch.setattr(treval, "upload_results_to_s3", lambda result_path, cfg: None)

    # Run evaluation
    exit_code = treval.run_evaluation(cfg)
    assert exit_code == treval.EXIT_OK

    # Validate output file exists and contents match expected structure
    assert os.path.exists(cfg.out)
    data = json.loads(open(cfg.out, "r", encoding="utf-8").read())
    assert "reward_mean" in data
    assert data["reward_mean"] == pytest.approx(fake_eval_result["reward_mean"], rel=1e-6)

def test_run_evaluation_fails_below_threshold(monkeypatch, tmp_path):
    """
    Ensure run_evaluation returns EXIT_MEAN_BELOW_THRESHOLD when mean reward < threshold.
    """
    tmp_dir = str(tmp_path)
    # set stop threshold higher than fake mean so the runner fails
    cfg = make_min_cfg(tmp_dir, eval_episodes=4, max_steps=50, out=os.path.join(tmp_dir, "eval_result.json"), stop_if_below_mean_reward=200.0)

    fake_eval_result = {
        "reward_mean": 50.0,
        "reward_std": 2.0,
        "reward_p95": 52.0,
        "episodes_completed": 4,
        "per_episode": [45.0, 50.0, 52.0, 53.0]
    }

    monkeypatch.setattr(treval, "build_policy_for_eval", lambda cfg, tmp_dir=None: ("dummy_policy", "/model/dir"))
    monkeypatch.setattr(treval, "run_vectorized_evaluation", lambda policy, env_factory, episodes, max_steps, n_workers: fake_eval_result)
    monkeypatch.setattr(treval, "safe_mlflow_init", lambda cfg, run_name: None)
    monkeypatch.setattr(treval, "safe_wandb_init", lambda cfg, run_name: None)
    monkeypatch.setattr(treval, "safe_mlflow_log_metrics", lambda metrics: None)
    monkeypatch.setattr(treval, "safe_wandb_log", lambda metrics: None)
    monkeypatch.setattr(treval, "safe_mlflow_end", lambda: None)
    monkeypatch.setattr(treval, "safe_wandb_finish", lambda: None)
    monkeypatch.setattr(treval, "upload_results_to_s3", lambda result_path, cfg: None)

    exit_code = treval.run_evaluation(cfg)
    assert exit_code == treval.EXIT_MEAN_BELOW_THRESHOLD

def test_run_evaluation_policy_build_fails(monkeypatch, tmp_path):
    """
    If policy build raises, the runner should return EXIT_FAILURE and not crash.
    """
    tmp_dir = str(tmp_path)
    cfg = make_min_cfg(tmp_dir, eval_episodes=2)

    def raise_build(cfg_arg, tmp_dir=None):
        raise RuntimeError("policy build error")
    monkeypatch.setattr(treval, "build_policy_for_eval", raise_build)
    monkeypatch.setattr(treval, "safe_mlflow_init", lambda cfg, run_name: None)
    monkeypatch.setattr(treval, "safe_wandb_init", lambda cfg, run_name: None)

    exit_code = treval.run_evaluation(cfg)
    assert exit_code == treval.EXIT_FAILURE

def test_cli_invocation_creates_result_file(monkeypatch, tmp_path, capsys):
    """
    Simulate CLI by calling main_cli with monkeypatched internals.
    Ensures main_cli writes output file and exits cleanly.
    """
    tmp_dir = str(tmp_path)
    out_path = os.path.join(tmp_dir, "eval_out.json")

    cfg_obj = make_min_cfg(tmp_dir, eval_episodes=1, out=out_path)

    # patch CLI helpers: EvalConfig.from_cli_and_file should return our cfg_obj
    class DummyArgs:
        pass

    dummy_args = DummyArgs()
    dummy_args.config = None
    monkeypatch.setattr(treval, "EvalConfig", types.SimpleNamespace(from_cli_and_file=lambda args: cfg_obj))
    # Patch run_evaluation to be a spy that writes file
    def fake_run(cfg):
        # write a result file
        with open(cfg.out, "w", encoding="utf-8") as f:
            json.dump({"reward_mean": 1.0, "episodes_completed": cfg.eval_episodes}, f)
        return treval.EXIT_OK
    monkeypatch.setattr(treval, "run_evaluation", fake_run)

    # Call CLI entrypoint (should call our patched functions)
    # main_cli uses argparse and sys.exit; capture SystemExit
    with pytest.raises(SystemExit) as se:
        treval.main_cli()
    # Expect normal exit (exit code 0)
    assert se.value.code == 0
    assert os.path.exists(out_path)
    data = json.loads(open(out_path, "r", encoding="utf-8").read())
    assert data["episodes_completed"] == cfg_obj.eval_episodes

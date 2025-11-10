#!/usr/bin/env python3
# backend/experiments/sim_runner.py
"""
PriorityMax Experiment & Simulation Runner
-----------------------------------------

Purpose
-------
A production-minded simulation runner used to:
 - create realistic queue/workload traces,
 - run large-scale simulation experiments (RL training / autoscaler evaluation),
 - evaluate autoscaler / predictor policies,
 - produce metrics, logs, and exportable artifacts (JSON, CSV, plots),
 - integrate with MLflow / Weights & Biases for experiment tracking (best-effort),
 - support parallel experiments (async + multiprocessing),
 - checkpoint simulation state and resume long runs.

Features
--------
 - WorkloadGenerators: Poisson, Bursty, Periodic, Trace-based
 - SimulatedRealEnv adapter integration hooks (if your ml.real_env.SimulatedRealEnv exists)
 - Pluggable policy callback: accept autoscaler, RL agent, or simple rule function
 - Result aggregation: per-second / rolling-window metrics, summary reports
 - Exporters: JSONL, CSV, local directory, MLflow/W&B integration
 - CLI with many controls for reproducibility & batch runs
 - Supports deterministic seeds, multi-process runs, and dry-run mode

Usage examples
--------------
# Single-run (interactive)
python3 backend/experiments/sim_runner.py run \
    --duration 60 --workload poisson --rate 50 --seed 42 --out-dir experiments/run1

# Batch grid search (vary burstiness)
python3 backend/experiments/sim_runner.py batch \
    --config experiments/batch_configs/grid.yaml

# Integrate with your RL autoscaler (best-effort)
python3 backend/experiments/sim_runner.py run \
    --policy-module app.autoscaler.PriorityMaxAutoscaler --policy-fn decide \
    --duration 600 --rate 200

Notes
-----
- This module is defensive: optional dependencies are imported best-effort and features switch off gracefully when libs are missing.
- Simulation is approximate and intended for stress-testing, not for production load generation against real upstream services.

Author: PriorityMax Team
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import atexit
import random
import shutil
import signal
import logging
import argparse
import tempfile
import pathlib
import itertools
import statistics
import threading
import traceback
import concurrent.futures
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Iterable, Union

# Optional libraries (best-effort)
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    _HAS_MATPLOTLIB = False

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

# Project modules (best-effort imports)
ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ml.real_env import SimulatedRealEnv, EnvConfig
except Exception:
    SimulatedRealEnv = None
    EnvConfig = None

try:
    from ml.predictor import PREDICTOR_MANAGER
except Exception:
    PREDICTOR_MANAGER = None

# Logging
LOG = logging.getLogger("prioritymax.experiments.sim_runner")
LOG.setLevel(os.getenv("PRIORITYMAX_EXPERIMENT_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Default directories
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = pathlib.Path(os.getenv("PRIORITYMAX_EXPERIMENTS_DIR", str(BASE_DIR / "experiments"))).resolve()
LOGS_DIR = EXPERIMENTS_DIR / "logs"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Type aliases
PolicyFn = Callable[[Dict[str, Any], float], Dict[str, Any]]
# Policy: accepts current metrics & timestamp -> returns action dict: {"action": "scale_up"/"scale_down"/"steady", "scale_to": int (optional)}

# -------------------------
# Dataclasses / Configs
# -------------------------
@dataclass
class ExperimentConfig:
    name: str = "sim_experiment"
    duration_sec: int = 300
    step_sec: float = 1.0
    seed: Optional[int] = None
    workload: str = "poisson"  # poisson, bursty, periodic, trace
    rate: float = 50.0  # avg events per second (for poisson)
    burst_rate: float = 500.0  # peak rate for bursty
    burst_prob: float = 0.05  # probability of burst window
    periodic_period: float = 60.0  # seconds
    periodic_amp: float = 1.5
    trace_path: Optional[str] = None
    policy_module: Optional[str] = None  # dotted module path to a policy (e.g., app.autoscaler)
    policy_fn: Optional[str] = None  # function name within module
    warmup_sec: int = 5
    out_dir: Optional[str] = None
    save_raw: bool = True
    use_mlflow: bool = False
    mlflow_experiment: str = "prioritymax_sim"
    use_wandb: bool = False
    wandb_project: str = "prioritymax-sim"
    parallel_workers: int = 4  # for parallel batch experiments
    dry_run: bool = False  # do not persist heavy artifacts
    # advanced
    enable_predictor_hint: bool = True  # call PREDICTOR_MANAGER for hints
    predictor_type: Optional[str] = None  # model type to ask predictor
    heartbeat_sec: float = 5.0  # for status printing

    def finalize(self):
        if not self.out_dir:
            ts = int(time.time())
            self.out_dir = str(EXPERIMENTS_DIR / f"{self.name}_{ts}_{uuid.uuid4().hex[:6]}")
        pathlib.Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        return self

# -------------------------
# Workload generators
# -------------------------
class WorkloadGenerator:
    """
    Abstract base for workload generators. Each generator yields counts per step (int) of new tasks.
    """
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed) if cfg.seed is not None else random.Random()

    def next_batch(self, t: float) -> int:
        raise NotImplementedError

class PoissonWorkload(WorkloadGenerator):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        if _HAS_NUMPY:
            self.np_rng = np.random.RandomState(cfg.seed)
        else:
            self.np_rng = None

    def next_batch(self, t: float) -> int:
        lam = self.cfg.rate * self.cfg.step_sec
        if self.np_rng is not None:
            return int(self.np_rng.poisson(lam=lam))
        # fallback: approximate via sum of Bernoullis
        count = 0
        p = lam / max(1.0, 1000.0)
        trials = int(1000)
        for _ in range(trials):
            if self.rng.random() < p:
                count += 1
        return count

class BurstyWorkload(WorkloadGenerator):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        if _HAS_NUMPY:
            self.np_rng = np.random.RandomState(cfg.seed)
        else:
            self.np_rng = None
        self.in_burst = False
        self.burst_ttl = 0

    def next_batch(self, t: float) -> int:
        # occasionally trigger a burst window
        if not self.in_burst and self.rng.random() < self.cfg.burst_prob:
            self.in_burst = True
            self.burst_ttl = max(1, int(self.rng.expovariate(1.0 / (self.cfg.step_sec * 10))))
        if self.in_burst:
            lam = self.cfg.burst_rate * self.cfg.step_sec
            self.burst_ttl -= 1
            if self.burst_ttl <= 0:
                self.in_burst = False
        else:
            lam = self.cfg.rate * self.cfg.step_sec
        if self.np_rng is not None:
            return int(self.np_rng.poisson(lam=lam))
        # fallback
        return int(self.rng.poissonvariate(lam) if hasattr(self.rng, "poissonvariate") else max(0, int(self.rng.expovariate(1.0/(lam+1e-6)))))

class PeriodicWorkload(WorkloadGenerator):
    def next_batch(self, t: float) -> int:
        # base poisson + sinusoidal multiplier
        base = self.cfg.rate * self.cfg.step_sec
        phase = (t % self.cfg.periodic_period) / max(1.0, self.cfg.periodic_period)
        multiplier = 1.0 + (self.cfg.periodic_amp - 1.0) * math.sin(2.0 * math.pi * phase)
        lam = max(0.0, base * multiplier)
        if _HAS_NUMPY:
            return int(np.random.RandomState(self.cfg.seed + int(t)).poisson(lam=lam))
        # fallback: approximate deterministic
        return int(round(lam))

class TraceWorkload(WorkloadGenerator):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        self.trace = []
        self.pos = 0
        if not cfg.trace_path:
            raise ValueError("trace_path required for TraceWorkload")
        self._load_trace(cfg.trace_path)

    def _load_trace(self, path: str):
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        # expect CSV with 'ts','count' or JSONL per-line {"ts":..., "count":...}
        try:
            if p.suffix.lower() in (".csv", ".tsv"):
                if _HAS_PANDAS:
                    df = pd.read_csv(str(p))
                    for _, r in df.iterrows():
                        self.trace.append((float(r.get("ts", 0.0)), int(r.get("count", 0))))
                else:
                    for line in p.read_text().splitlines():
                        parts = line.split(",")
                        if len(parts) >= 2:
                            self.trace.append((float(parts[0]), int(parts[1])))
            else:
                for line in p.read_text().splitlines():
                    obj = json.loads(line)
                    self.trace.append((float(obj.get("ts", 0)), int(obj.get("count", 0))))
        except Exception:
            LOG.exception("Failed to load trace")
            raise
        # normalize to per-step mapping
        self.trace.sort()
        if not self.trace:
            raise ValueError("trace file contained no events")

    def next_batch(self, t: float) -> int:
        if self.pos >= len(self.trace):
            return 0
        # advance until trace timestamp >= t
        ts, cnt = self.trace[self.pos]
        if ts <= t:
            self.pos += 1
            return cnt
        return 0
# -------------------------
# Simulation Core
# -------------------------
@dataclass
class SimulationState:
    t: float
    backlog: int
    inflight: int
    processed: int
    failed: int
    dlq: int


class SimCollector:
    """
    Lightweight collector storing per-step metrics and producing summaries.
    """
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.start_ts = time.time()

    def record(self, ts: float, state: SimulationState, step_new: int, step_processed: int, step_failed: int):
        self.rows.append({
            "ts": ts,
            "t": state.t,
            "backlog": state.backlog,
            "inflight": state.inflight,
            "processed_cumulative": state.processed,
            "processed_step": step_processed,
            "failed_cumulative": state.failed,
            "failed_step": step_failed,
            "dlq": state.dlq,
            "new_arrivals": step_new
        })

    def to_jsonl(self, path: Union[str, pathlib.Path]):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, default=str) + "\n")

    def to_csv(self, path: Union[str, pathlib.Path]):
        if not _HAS_PANDAS:
            p = pathlib.Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if not self.rows:
                return
            keys = list(self.rows[0].keys())
            with p.open("w", encoding="utf-8") as fh:
                fh.write(",".join(keys) + "\n")
                for r in self.rows:
                    fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
            return
        df = pd.DataFrame(self.rows)
        df.to_csv(str(path), index=False)

    def summary(self) -> Dict[str, Any]:
        if not self.rows:
            return {}
        total_processed = self.rows[-1]["processed_cumulative"]
        total_failed = self.rows[-1]["failed_cumulative"]
        avg_backlog = statistics.mean(r["backlog"] for r in self.rows)
        return {
            "total_processed": int(total_processed),
            "total_failed": int(total_failed),
            "avg_backlog": float(avg_backlog)
        }


# -------------------------
# Policy loader helpers
# -------------------------
def load_policy_from_module(dotted_path: str, fn_name: str) -> PolicyFn:
    if not dotted_path:
        raise ValueError("policy module path required")
    module = __import__(dotted_path, fromlist=[fn_name])
    if not hasattr(module, fn_name):
        raise AttributeError(f"Module {dotted_path} has no function {fn_name}")
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError("policy function must be callable")
    return fn


# -------------------------
# Simulation runner
# -------------------------
class SimRunner:
    """
    Main simulation runner.
    """
    def __init__(self, cfg: ExperimentConfig, policy: Optional[PolicyFn] = None):
        self.cfg = cfg
        self.cfg.finalize()
        self.policy = policy
        self.generator = self._make_generator(self.cfg)
        self.collector = SimCollector()
        self.state = SimulationState(t=0.0, backlog=0, inflight=0, processed=0, failed=0, dlq=0)
        self._stop_requested = False
        self._last_heartbeat = time.time()
        # hooks
        self.on_step_hooks: List[Callable[[int, SimulationState, Dict[str, Any]], None]] = []
        self.out_dir = pathlib.Path(self.cfg.out_dir)
        self.log_file = self.out_dir / "sim.log"

        fh = logging.FileHandler(str(self.log_file))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        LOG.addHandler(fh)

        # MLflow/W&B tracking
        self.mlflow_run = None
        self.wandb_run = None
        if self.cfg.use_mlflow and _HAS_MLFLOW and not self.cfg.dry_run:
            try:
                mlflow.set_experiment(self.cfg.mlflow_experiment)
                self.mlflow_run = mlflow.start_run(run_name=self.cfg.name)
                LOG.info("MLflow run started: %s", self.mlflow_run.info.run_id)
            except Exception:
                LOG.exception("MLflow start failed")
                self.mlflow_run = None
        if self.cfg.use_wandb and _HAS_WANDB and not self.cfg.dry_run:
            try:
                wandb.init(project=self.cfg.wandb_project, name=self.cfg.name, config=asdict(self.cfg))
                self.wandb_run = wandb
            except Exception:
                LOG.exception("W&B init failed")
                self.wandb_run = None

    def _make_generator(self, cfg: ExperimentConfig) -> WorkloadGenerator:
        if cfg.workload == "poisson":
            return PoissonWorkload(cfg)
        if cfg.workload == "bursty":
            return BurstyWorkload(cfg)
        if cfg.workload == "periodic":
            return PeriodicWorkload(cfg)
        if cfg.workload == "trace":
            return TraceWorkload(cfg)
        raise ValueError(f"unknown workload {cfg.workload}")

    def stop(self):
        self._stop_requested = True

    def _heartbeat(self, step: int):
        now = time.time()
        if now - self._last_heartbeat >= self.cfg.heartbeat_sec:
            LOG.info(
                "Sim %s step=%d t=%.1f backlog=%d processed=%d failed=%d dlq=%d",
                self.cfg.name,
                step,
                self.state.t,
                self.state.backlog,
                self.state.processed,
                self.state.failed,
                self.state.dlq,
            )
            self._last_heartbeat = now

    def _apply_policy(self, metrics: Dict[str, Any], t: float) -> Dict[str, Any]:
        """
        If a policy callback is present, call it. Otherwise use simple rule-based scaling hint.
        """
        if self.policy:
            try:
                return self.policy(metrics, t)
            except Exception:
                LOG.exception("Policy function raised; ignoring")
                return {"action": "steady"}

        backlog = metrics.get("backlog", 0)
        if backlog > max(10, self.cfg.rate * 2):
            return {"action": "scale_up"}
        if backlog < max(1, self.cfg.rate * 0.5):
            return {"action": "scale_down"}
        return {"action": "steady"}

    def _call_predictor_hint(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.cfg.enable_predictor_hint or PREDICTOR_MANAGER is None:
            return None
        try:
            feat = {
                "queue_len": int(metrics.get("backlog", 0)),
                "avg_latency": float(metrics.get("avg_latency", 0.0)),
                "cpu": float(metrics.get("cpu", 0.0)),
            }
            preds = PREDICTOR_MANAGER.predict(feat, model_type=self.cfg.predictor_type)
            return preds
        except Exception:
            LOG.exception("predictor hint failed")
            return None

    def rng_random(self) -> float:
        return random.Random(
            self.cfg.seed + int(self.state.t * 1000) if self.cfg.seed is not None else random.random()
        ).random()

    def _process_step(self, step_new: int, step_time: float) -> Tuple[int, int]:
        capacity = max(1, int(math.sqrt(max(1, self.state.backlog)) * self.cfg.step_sec * 1.0))
        to_process = min(self.state.backlog, capacity)
        failed = 0
        if to_process > 0:
            fail_prob = 0.001
            for _ in range(to_process):
                if self.rng_random() < fail_prob:
                    failed += 1
        self.state.backlog -= (to_process - failed)
        self.state.processed += (to_process - failed)
        self.state.failed += failed
        return to_process - failed, failed

    def run(self):
        LOG.info("Starting simulation %s -> %s", self.cfg.name, self.cfg.out_dir)
        steps = int(math.ceil(self.cfg.duration_sec / self.cfg.step_sec))
        self.state = SimulationState(t=0.0, backlog=0, inflight=0, processed=0, failed=0, dlq=0)

        env = None
        if SimulatedRealEnv:
            try:
                env = SimulatedRealEnv(EnvConfig(mode="sim"))
                LOG.debug("SimulatedRealEnv initialized")
            except Exception:
                LOG.exception("Failed to init SimulatedRealEnv; continuing without")

        for step in range(steps):
            if self._stop_requested:
                LOG.info("Stop requested; terminating simulation early at step %d", step)
                break
            t = step * self.cfg.step_sec
            self.state.t = t
            new_arrivals = int(self.generator.next_batch(t))
            self.state.backlog += new_arrivals

            metrics = {
                "t": t,
                "backlog": self.state.backlog,
                "inflight": self.state.inflight,
                "processed": self.state.processed,
                "failed": self.state.failed,
                "dlq": self.state.dlq,
            }
            predictor_hint = self._call_predictor_hint(metrics)
            if predictor_hint:
                metrics["predictor_hint"] = predictor_hint

            decision = self._apply_policy(metrics, t)
            processed, failed = self._process_step(new_arrivals, t)
            self.state.dlq += failed
            self.collector.record(t, self.state, new_arrivals, processed, failed)

            for hook in list(self.on_step_hooks):
                try:
                    hook(step, self.state, {"new": new_arrivals, "processed": processed, "failed": failed, "decision": decision})
                except Exception:
                    LOG.exception("on_step_hook failed")

            self._heartbeat(step)

        self._finalize()
        LOG.info("Simulation finished: %s", self.cfg.out_dir)
        return {"summary": self.collector.summary(), "out_dir": str(self.out_dir)}

    def _finalize(self):
        try:
            if self.cfg.save_raw and not self.cfg.dry_run:
                jsonl_path = pathlib.Path(self.cfg.out_dir) / "raw.jsonl"
                self.collector.to_jsonl(jsonl_path)
                csv_path = pathlib.Path(self.cfg.out_dir) / "raw.csv"
                self.collector.to_csv(csv_path)
                LOG.info("Saved raw outputs to %s and %s", jsonl_path, csv_path)
        except Exception:
            LOG.exception("Failed to save outputs")

        try:
            if self.mlflow_run:
                mlflow.log_params(asdict(self.cfg))
                mlflow.log_artifact(str(self.out_dir))
                mlflow.end_run()
            if self.wandb_run:
                self.wandb_run.log(self.collector.summary())
                self.wandb_run.save(str(self.out_dir))
                self.wandb_run.finish()
        except Exception:
            LOG.exception("Experiment tracking finalization failed")

    def add_on_step_hook(self, fn: Callable[[int, SimulationState, Dict[str, Any]], None]):
        self.on_step_hooks.append(fn)


# -------------------------
# Parallel & Analysis Tools
# -------------------------
def run_single_experiment(cfg: ExperimentConfig, policy: Optional[PolicyFn] = None) -> Dict[str, Any]:
    try:
        runner = SimRunner(cfg, policy)
        res = runner.run()
        return {"ok": True, "name": cfg.name, "result": res}
    except Exception:
        LOG.exception("run_single_experiment failed")
        return {"ok": False, "name": cfg.name, "error": traceback.format_exc()}


def run_parallel_batch(cfgs: List[ExperimentConfig], policy_factory: Optional[Callable[[ExperimentConfig], PolicyFn]] = None, max_workers: int = 4) -> List[Dict[str, Any]]:
    results = []
    if max_workers <= 1 or len(cfgs) == 1:
        for c in cfgs:
            policy = policy_factory(c) if policy_factory else None
            results.append(run_single_experiment(c, policy))
        return results

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for c in cfgs:
            futures.append(pool.submit(_process_worker_entry, json.dumps(asdict(c))))
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                LOG.exception("Process run failed")
                results.append({"ok": False, "error": traceback.format_exc()})
    return results


def _process_worker_entry(cfg_json: str) -> Dict[str, Any]:
    try:
        cfg_dict = json.loads(cfg_json)
        cfg = ExperimentConfig(**cfg_dict)
        runner = SimRunner(cfg)
        res = runner.run()
        return {"ok": True, "name": cfg.name, "result": res}
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


def plot_backlog(csv_path: Union[str, pathlib.Path], out_png: Union[str, pathlib.Path]):
    if not _HAS_PANDAS or not _HAS_MATPLOTLIB:
        LOG.warning("Plotting requires pandas & matplotlib")
        return
    df = pd.read_csv(str(csv_path))
    plt.figure(figsize=(12, 4))
    plt.plot(df["t"], df["backlog"], label="backlog")
    plt.plot(df["t"], df["processed_step"].cumsum(), label="processed_cum")
    plt.xlabel("t (s)")
    plt.ylabel("count")
    plt.legend()
    plt.title("Backlog over time")
    plt.tight_layout()
    plt.savefig(str(out_png))
    plt.close()
    LOG.info("Saved plot to %s", out_png)


# -------------------------
# CLI
# -------------------------
def parse_config_file(path: str) -> List[ExperimentConfig]:
    import yaml
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    raw = p.read_text()
    try:
        data = yaml.safe_load(raw)
    except Exception:
        data = json.loads(raw)
    cfgs = []
    if isinstance(data, list):
        for entry in data:
            cfg = ExperimentConfig(**entry)
            cfgs.append(cfg)
    elif isinstance(data, dict):
        entries = data.get("experiments", [])
        for entry in entries:
            cfg = ExperimentConfig(**entry)
            cfgs.append(cfg)
    else:
        raise ValueError("Unsupported config format")
    return cfgs


def cli_run(args):
    cfg = ExperimentConfig(
        name=args.name or "sim",
        duration_sec=args.duration,
        step_sec=args.step,
        seed=args.seed,
        workload=args.workload,
        rate=args.rate,
        burst_rate=args.burst_rate,
        burst_prob=args.burst_prob,
        periodic_period=args.periodic_period,
        periodic_amp=args.periodic_amp,
        trace_path=args.trace_path,
        policy_module=args.policy_module,
        policy_fn=args.policy_fn,
        warmup_sec=args.warmup,
        out_dir=args.out_dir,
        save_raw=not args.no_save,
        use_mlflow=args.mlflow,
        use_wandb=args.wandb,
        dry_run=args.dry_run,
    )
    policy = None
    if args.policy_module and args.policy_fn:
        try:
            policy = load_policy_from_module(args.policy_module, args.policy_fn)
        except Exception:
            LOG.exception("Failed to load policy; proceeding without policy")
            policy = None
    res = run_single_experiment(cfg, policy)
    print(json.dumps(res, indent=2))


def cli_batch(args):
    cfgs = parse_config_file(args.config)
    if args.out_dir:
        parent = pathlib.Path(args.out_dir)
        parent.mkdir(parents=True, exist_ok=True)
        for c in cfgs:
            c.out_dir = str(parent / pathlib.Path(c.out_dir).name)
    results = run_parallel_batch(cfgs, max_workers=args.workers)
    out_path = pathlib.Path(args.out_dir or EXPERIMENTS_DIR) / f"batch_results_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("Batch results saved to", out_path)


def build_arg_parser():
    p = argparse.ArgumentParser(prog="prioritymax-sim-runner")
    sub = p.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="Run a single simulation")
    run.add_argument("--name", type=str, default=None)
    run.add_argument("--duration", type=int, default=300)
    run.add_argument("--step", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--workload", choices=["poisson", "bursty", "periodic", "trace"], default="poisson")
    run.add_argument("--rate", type=float, default=50.0)
    run.add_argument("--burst-rate", type=float, default=500.0)
    run.add_argument("--burst-prob", type=float, default=0.05)
    run.add_argument("--periodic-period", type=float, default=60.0)
    run.add_argument("--periodic-amp", type=float, default=1.5)
    run.add_argument("--trace-path", type=str, default=None)
    run.add_argument("--policy-module", type=str, default=None)
    run.add_argument("--policy-fn", type=str, default=None)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--out-dir", type=str, default=None)
    run.add_argument("--no-save", action="store_true")
    run.add_argument("--mlflow", action="store_true")
    run.add_argument("--wandb", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cli_run)

    batch = sub.add_parser("batch", help="Run a batch/grid of simulations from config file")
    batch.add_argument("--config", required=True)
    batch.add_argument("--out-dir", required=False)
    batch.add_argument("--workers", type=int, default=4)
    batch.set_defaults(func=cli_batch)
    return p


def _install_signal_handlers(runner: Optional[SimRunner] = None):
    def _handler(signum, frame):
        LOG.info("Signal %s received; requesting stop", signum)
        if runner:
            runner.stop()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# backend/scripts/generate_sim_data.py
"""
PriorityMax — Synthetic Queue Metrics / Training Data Generator
---------------------------------------------------------------

Purpose
-------
Generate realistic synthetic datasets used to train the queue predictor and autoscaler.
Generates per-second (or configurable step) traces of:
  - incoming arrivals
  - backlog
  - processed
  - latency proxies
  - CPU/memory usage proxies
and computes supervised targets like next-step queue size, horizon max, or whether a burst occurs.

Features
--------
 - Workloads: poisson, bursty, periodic, trace-based
 - Vectorized generation (NumPy) for speed; pure-Python fallback
 - Multi-file & parallel generation for large datasets
 - Output CSV / JSONL / Parquet
 - Train/val/test split helper and optional shuffling
 - Optional S3 upload and local compression
 - Optional MLflow / W&B logging (best-effort)
 - CLI friendly & deterministic with seeds

Usage examples
--------------
# Generate 1 hour of per-second poisson arrivals, save CSV
python3 backend/scripts/generate_sim_data.py \
    --duration 3600 --step 1 --workload poisson --rate 30 \
    --out datasets/queue_metrics_1h.csv --format csv

# Generate 10 files in parallel (each 1 hour)
python3 backend/scripts/generate_sim_data.py \
    --duration 3600 --step 1 --workload bursty --rate 30 --files 10 \
    --out datasets/sim_runs --format parquet --parallel 4

# Generate labeled dataset for predictor: predict next-second queue length
python3 backend/scripts/generate_sim_data.py \
    --duration 86400 --step 1 --workload periodic --rate 20 \
    --target next_step --horizon 1 --out datasets/predictor_train.parquet

Notes
-----
This script is defensive: optional libraries are imported best-effort and features toggle gracefully
if the libs (numpy, pandas, boto3, mlflow, wandb, pyarrow) are not installed.
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import gzip
import shutil
import random
import tarfile
import argparse
import logging
import pathlib
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Optional dependencies (best-effort)
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
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

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
    import pyarrow  # for parquet
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

# Logging
LOG = logging.getLogger("prioritymax.synth")
LOG.setLevel(os.getenv("PRIORITYMAX_SYNTH_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults
DEFAULT_OUT_DIR = pathlib.Path.cwd() / "datasets"
DEFAULT_STEP = 1.0  # sec
DEFAULT_DURATION = 3600  # seconds (1 hour)
DEFAULT_FILES = 1

# -------------------------
# Dataclasses
# -------------------------
@dataclass
class GenConfig:
    duration: int = DEFAULT_DURATION
    step: float = DEFAULT_STEP
    workload: str = "poisson"  # poisson, bursty, periodic, trace
    rate: float = 30.0  # average arrivals per second
    burst_rate: float = 300.0
    burst_prob: float = 0.01
    periodic_period: float = 3600.0
    periodic_amp: float = 2.0
    jitter_std: float = 0.1  # noise on latency / cpu traces
    seed: Optional[int] = None
    out: Union[str, pathlib.Path] = str(DEFAULT_OUT_DIR)
    out_format: str = "csv"  # csv, jsonl, parquet
    files: int = DEFAULT_FILES
    parallel: int = 1
    compress: bool = False
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    target: Optional[str] = None  # e.g., 'next_step', 'horizon_max'
    horizon: int = 60  # seconds for horizon targets
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    use_mlflow: bool = False
    use_wandb: bool = False
    shuffle: bool = False
    trace_path: Optional[str] = None  # for trace workload
    verbose: bool = False

    def finalize(self):
        # normalize splits
        s = self.train_split + self.val_split + self.test_split
        if abs(s - 1.0) > 1e-6:
            # renormalize
            self.train_split /= s
            self.val_split /= s
            self.test_split /= s
        return self

# -------------------------
# Workload generators
# -------------------------
class BaseGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.seed = cfg.seed if cfg.seed is not None else random.randint(0, 2 ** 31 - 1)
        self.rng = random.Random(self.seed)
        if _HAS_NUMPY:
            self.np_rng = np.random.RandomState(self.seed)
        else:
            self.np_rng = None

    def generate_arrivals(self) -> List[int]:
        """Return sequence of arrivals per-step for duration steps."""
        raise NotImplementedError

class PoissonGenerator(BaseGenerator):
    def generate_arrivals(self) -> List[int]:
        steps = int(math.ceil(self.cfg.duration / self.cfg.step))
        lam = self.cfg.rate * self.cfg.step
        if self.np_rng is not None:
            arr = self.np_rng.poisson(lam=lam, size=steps).astype(int).tolist()
            return arr
        # fallback: approximate with sum of Bernoulli
        out = []
        for _ in range(steps):
            # use Poisson approximation via Knuth when lam small? use exponential inter-arrival method:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= self.rng.random()
            out.append(k - 1)
        return out

class BurstyGenerator(BaseGenerator):
    def generate_arrivals(self) -> List[int]:
        steps = int(math.ceil(self.cfg.duration / self.cfg.step))
        out = []
        in_burst = False
        burst_ttl = 0
        for i in range(steps):
            if not in_burst and self.rng.random() < self.cfg.burst_prob:
                in_burst = True
                burst_ttl = max(1, int(self.rng.expovariate(1.0 / 10.0)))  # burst length in steps (random)
            lam = (self.cfg.burst_rate if in_burst else self.cfg.rate) * self.cfg.step
            if self.np_rng is not None:
                val = int(self.np_rng.poisson(lam=lam))
            else:
                # simple approximation
                val = max(0, int(round(lam + self.rng.gauss(0, lam ** 0.5))))
            out.append(val)
            if in_burst:
                burst_ttl -= 1
                if burst_ttl <= 0:
                    in_burst = False
        return out

class PeriodicGenerator(BaseGenerator):
    def generate_arrivals(self) -> List[int]:
        steps = int(math.ceil(self.cfg.duration / self.cfg.step))
        out = []
        base = self.cfg.rate * self.cfg.step
        for i in range(steps):
            t = i * self.cfg.step
            phase = (t % self.cfg.periodic_period) / max(1.0, self.cfg.periodic_period)
            multiplier = 1.0 + (self.cfg.periodic_amp - 1.0) * math.sin(2 * math.pi * phase)
            lam = max(0.0, base * multiplier)
            if self.np_rng is not None:
                val = int(self.np_rng.poisson(lam=lam))
            else:
                val = max(0, int(round(lam + self.rng.gauss(0, math.sqrt(max(1.0, lam))))))
            out.append(val)
        return out

class TraceGenerator(BaseGenerator):
    def generate_arrivals(self) -> List[int]:
        # Expect trace file with per-step counts or timestamped counts
        if not self.cfg.trace_path:
            raise ValueError("trace_path required for trace workload")
        p = pathlib.Path(self.cfg.trace_path)
        if not p.exists():
            raise FileNotFoundError(self.cfg.trace_path)
        lines = p.read_text().splitlines()
        # Attempt CSV or JSONL parsing
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # either {"ts":..., "count":...} or just {"count": ...}
                cnt = int(obj.get("count") or obj.get("cnt") or obj.get("c") or 0)
                data.append(cnt)
            except Exception:
                parts = [s.strip() for s in line.split(",") if s.strip()]
                if len(parts) >= 2:
                    try:
                        cnt = int(parts[1])
                    except Exception:
                        cnt = int(float(parts[1]))
                    data.append(cnt)
                else:
                    # fallback interpret whole line as int
                    try:
                        data.append(int(line))
                    except Exception:
                        data.append(0)
        # if trace longer/shorter than steps, repeat/truncate
        steps = int(math.ceil(self.cfg.duration / self.cfg.step))
        if not data:
            return [0] * steps
        if len(data) >= steps:
            return data[:steps]
        # repeat to cover steps
        out = []
        i = 0
        while len(out) < steps:
            out.append(data[i % len(data)])
            i += 1
        return out

# -------------------------
# Simulation of queue dynamics (to produce labels)
# -------------------------
def simulate_queue(arrivals: List[int], cfg: GenConfig) -> Tuple[List[int], List[int], List[float], List[float]]:
    """
    Given arrivals per step, simulate:
      - backlog sequence
      - processed per step
      - latency proxy (e.g., average processing time per task)
      - cpu proxy (normalized 0..1)
    Returns (backlog, processed, latencies, cpu)
    """
    steps = len(arrivals)
    backlog = [0] * steps
    processed = [0] * steps
    latencies = [0.0] * steps
    cpu = [0.0] * steps

    cur_backlog = 0
    # simple processing capacity model: capacity scales with sqrt of backlog (diminishing returns) and base throughput
    base_capacity = max(1.0, cfg.rate * cfg.step * 0.5)
    for i in range(steps):
        new = arrivals[i]
        cur_backlog += new
        # capacity might scale with number of workers; simulate variable capacity
        capacity = int(max(1, base_capacity + math.sqrt(cur_backlog)))
        processed_now = min(cur_backlog, capacity)
        # failure/noise simulated
        fail_prob = 0.002  # small failure probability
        # adjust processed for failures
        fails = 0
        if processed_now > 0:
            # vectorized style would be faster; use probabilistic approximation
            if _HAS_NUMPY:
                fails = int(np.random.binomial(processed_now, fail_prob))
            else:
                # binomial approx
                fails = sum(1 for _ in range(processed_now) if random.random() < fail_prob)
        processed_success = processed_now - fails
        cur_backlog -= processed_success
        backlog[i] = cur_backlog
        processed[i] = processed_success
        # latency proxy: grows with backlog, add jitter
        latency = (1.0 + math.log1p(backlog[i])) * (1.0 + cfg.jitter_std * (random.random() - 0.5))
        latencies[i] = max(0.0, float(latency))
        # cpu proxy: normalize capacity usage to 0..1
        cpu_val = min(1.0, (processed_now / max(1.0, capacity)) * (0.2 + 0.8 * min(1.0, backlog[i] / max(1.0, cfg.rate))))
        cpu[i] = float(min(1.0, cpu_val + cfg.jitter_std * (random.random() - 0.5)))
    return backlog, processed, latencies, cpu

# -------------------------
# Label / target builders
# -------------------------
def build_targets(backlog: List[int], cfg: GenConfig) -> Dict[str, List[Union[int, float]]]:
    """
    Build supervised targets based on cfg.target:
      - next_step: backlog at t+1 (0 if out of bounds)
      - horizon_max: max backlog in next cfg.horizon seconds
      - binary_burst: 1 if backlog exceeds some threshold in horizon
    """
    steps = len(backlog)
    horizon_steps = max(1, int(math.ceil(cfg.horizon / cfg.step)))
    targets = {"next_step": [], "horizon_max": [], "binary_burst": []}
    for i in range(steps):
        nxt = backlog[i + 1] if i + 1 < steps else 0
        targets["next_step"].append(int(nxt))
        hi = max(backlog[i + 1: i + 1 + horizon_steps]) if i + 1 < steps else 0
        targets["horizon_max"].append(int(hi))
        thresh = max(1, int(cfg.rate * cfg.step * 2))
        targets["binary_burst"].append(1 if hi >= thresh else 0)
    return targets

# -------------------------
# I/O helpers
# -------------------------
def write_csv(path: Union[str, pathlib.Path], rows: List[Dict[str, Any]]):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    # use pandas if available for speed & correctness
    if _HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(str(p), index=False)
        return
    # pure python write: header + rows
    keys = list(rows[0].keys())
    with p.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, str):
                    vals.append('"{}"'.format(v.replace('"', '""')))
                else:
                    vals.append(str(v))
            fh.write(",".join(vals) + "\n")

def write_jsonl(path: Union[str, pathlib.Path], rows: List[Dict[str, Any]]):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

def write_parquet(path: Union[str, pathlib.Path], rows: List[Dict[str, Any]]):
    if not _HAS_PANDAS or not _HAS_PYARROW:
        raise RuntimeError("Parquet output requires pandas and pyarrow")
    df = pd.DataFrame(rows)
    df.to_parquet(str(path), index=False)

def compress_tar_gz(src_paths: List[Union[str, pathlib.Path]], dest_path: Union[str, pathlib.Path]):
    dest = pathlib.Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(dest), "w:gz") as tar:
        for p in src_paths:
            tar.add(str(p), arcname=pathlib.Path(p).name)
    return str(dest)

def upload_to_s3(local_path: Union[str, pathlib.Path], bucket: str, key: str):
    if not _HAS_BOTO3:
        raise RuntimeError("boto3 required to upload to s3")
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"

# -------------------------
# Core generation worker (single file)
# -------------------------
def generate_one_file(idx: int, cfg: GenConfig) -> Dict[str, Any]:
    """
    Generate a single file run. Returns metadata about generated artifact.
    """
    run_seed = (cfg.seed or 0) + idx * 10007
    cfg_local = GenConfig(**asdict(cfg))
    cfg_local.seed = run_seed
    cfg_local.finalize()
    LOG.info("Generating file idx=%d seed=%d duration=%d step=%s workload=%s", idx, run_seed, cfg_local.duration, cfg_local.step, cfg_local.workload)

    # Choose generator
    if cfg_local.workload == "poisson":
        gen = PoissonGenerator(cfg_local)
    elif cfg_local.workload == "bursty":
        gen = BurstyGenerator(cfg_local)
    elif cfg_local.workload == "periodic":
        gen = PeriodicGenerator(cfg_local)
    elif cfg_local.workload == "trace":
        gen = TraceGenerator(cfg_local)
    else:
        raise ValueError("Unknown workload: " + cfg_local.workload)

    arrivals = gen.generate_arrivals()
    backlog, processed, latencies, cpu = simulate_queue(arrivals, cfg_local)
    targets = build_targets(backlog, cfg_local) if cfg_local.target else {}

    # construct rows: ts, t, arrivals, backlog, processed, latency, cpu, plus targets
    rows = []
    steps = len(arrivals)
    base_ts = int(time.time())
    for i in range(steps):
        t = i * cfg_local.step
        ts = base_ts + int(t)
        row = {
            "ts": ts,
            "t": float(t),
            "arrivals": int(arrivals[i]),
            "backlog": int(backlog[i]),
            "processed": int(processed[i]),
            "latency": float(latencies[i]),
            "cpu": float(cpu[i]),
            "seed": int(cfg_local.seed),
            "run_idx": int(idx),
        }
        # attach targets if present
        if targets:
            if "next_step" in targets:
                row["target_next_step"] = int(targets["next_step"][i])
            if "horizon_max" in targets:
                row["target_horizon_max"] = int(targets["horizon_max"][i])
            if "binary_burst" in targets:
                row["target_binary_burst"] = int(targets["binary_burst"][i])
        rows.append(row)

    # optionally shuffle rows (useful for ML training if desired)
    if cfg_local.shuffle:
        if _HAS_NUMPY:
            arr = np.array(rows, dtype=object)
            np.random.RandomState(cfg_local.seed).shuffle(arr)
            rows = arr.tolist()
        else:
            random.Random(cfg_local.seed).shuffle(rows)

    # determine output path
    out_path = pathlib.Path(cfg_local.out)
    if cfg_local.files > 1 and out_path.is_dir():
        fname = f"sim_{idx}_{cfg_local.workload}_{cfg_local.duration}s.{cfg_local.out_format}"
        target_path = out_path / fname
    elif cfg_local.files > 1 and not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"sim_{idx}_{cfg_local.workload}_{cfg_local.duration}s.{cfg_local.out_format}"
        target_path = out_path / fname
    else:
        # if single file and out is file path, respect it (allow extension replacement)
        if out_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
            fname = f"sim_{idx}_{cfg_local.workload}_{cfg_local.duration}s.{cfg_local.out_format}"
            target_path = out_path / fname
        else:
            # if out path ends with extension use as is
            if str(cfg_local.out).endswith(cfg_local.out_format):
                target_path = pathlib.Path(cfg_local.out)
            else:
                # change extension
                target_path = pathlib.Path(str(cfg_local.out) + "." + cfg_local.out_format)

    # write file
    try:
        if cfg_local.out_format == "csv":
            write_csv(target_path, rows)
        elif cfg_local.out_format == "jsonl":
            write_jsonl(target_path, rows)
        elif cfg_local.out_format in ("parquet", "pq"):
            write_parquet(target_path, rows)
        else:
            raise ValueError("Unsupported out_format " + cfg_local.out_format)
        LOG.info("Wrote %d rows to %s", len(rows), target_path)
    except Exception:
        LOG.exception("Failed to write output file")
        raise

    uploaded = None
    if cfg_local.compress:
        tar_path = str(target_path) + ".tar.gz"
        compress_tar_gz([target_path], tar_path)
        LOG.info("Compressed output into %s", tar_path)
        if cfg_local.s3_bucket:
            key = f"{cfg_local.s3_prefix.rstrip('/')}/{pathlib.Path(tar_path).name}" if cfg_local.s3_prefix else pathlib.Path(tar_path).name
            uploaded = upload_to_s3(tar_path, cfg_local.s3_bucket, key)
            LOG.info("Uploaded compressed artifact to %s", uploaded)
    elif cfg_local.s3_bucket:
        # upload raw file
        key = f"{cfg_local.s3_prefix.rstrip('/')}/{pathlib.Path(target_path).name}" if cfg_local.s3_prefix else pathlib.Path(target_path).name
        uploaded = upload_to_s3(target_path, cfg_local.s3_bucket, key)
        LOG.info("Uploaded artifact to %s", uploaded)

    # optional MLflow / W&B logging
    if cfg_local.use_mlflow and _HAS_MLFLOW:
        try:
            mlflow.log_param("file_index", idx)
            mlflow.log_param("seed", cfg_local.seed)
            mlflow.log_artifact(str(target_path))
        except Exception:
            LOG.exception("MLflow logging failed")
    if cfg_local.use_wandb and _HAS_WANDB:
        try:
            wandb.run.summary[f"file_{idx}_rows"] = len(rows)
            # upload artifact if available
            if _HAS_WANDB:
                try:
                    wandb.save(str(target_path))
                except Exception:
                    LOG.exception("W&B save failed")
        except Exception:
            LOG.exception("W&B logging failed")

    return {"index": idx, "path": str(target_path), "rows": len(rows), "uploaded": uploaded, "seed": cfg_local.seed}

# -------------------------
# Orchestration
# -------------------------
def generate_batch(cfg: GenConfig) -> List[Dict[str, Any]]:
    cfg = cfg.finalize()
    out_dir = pathlib.Path(cfg.out)
    if cfg.files > 1 and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # single-threaded
    if cfg.parallel <= 1 or cfg.files == 1:
        results = []
        for idx in range(cfg.files):
            res = generate_one_file(idx, cfg)
            results.append(res)
        return results

    # parallel generation via multiprocessing
    import concurrent.futures
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.parallel) as pool:
        futures = {pool.submit(generate_one_file, idx, cfg): idx for idx in range(cfg.files)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception:
                LOG.exception("Failed generate idx=%s", idx)
                results.append({"index": idx, "error": "failed"})
    return results

# -------------------------
# Train/val/test split helper
# -------------------------
def split_and_save(path: Union[str, pathlib.Path], cfg: GenConfig):
    p = pathlib.Path(path)
    if not p.exists():
        LOG.warning("Cannot split non-existing file %s", p)
        return
    if cfg.out_format not in ("csv", "parquet", "jsonl"):
        LOG.warning("Split not supported for format %s", cfg.out_format)
        return
    # load using pandas if available
    if not _HAS_PANDAS:
        LOG.warning("Pandas not available; skipping splitting")
        return
    if cfg.out_format == "parquet":
        df = pd.read_parquet(str(p))
    elif cfg.out_format == "csv":
        df = pd.read_csv(str(p))
    else:
        # jsonl
        df = pd.read_json(str(p), lines=True)

    # shuffle if requested
    if cfg.shuffle:
        df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    n = len(df)
    ti = int(n * cfg.train_split)
    vi = ti + int(n * cfg.val_split)
    train = df.iloc[:ti]
    val = df.iloc[ti:vi]
    test = df.iloc[vi:]

    base = p.parent / (p.stem + "_split")
    base.mkdir(parents=True, exist_ok=True)
    def _save(df_part, suffix):
        target = base / f"{p.stem}_{suffix}.{cfg.out_format}"
        if cfg.out_format == "parquet":
            df_part.to_parquet(str(target), index=False)
        elif cfg.out_format == "csv":
            df_part.to_csv(str(target), index=False)
        else:
            df_part.to_json(str(target), orient="records", lines=True)
        LOG.info("Wrote split %s rows to %s", len(df_part), target)
    _save(train, "train")
    _save(val, "val")
    _save(test, "test")
    return {"train": len(train), "val": len(val), "test": len(test), "dir": str(base)}

# -------------------------
# CLI
# -------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="prioritymax-generate-sim-data")
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Total duration in seconds per file")
    p.add_argument("--step", type=float, default=DEFAULT_STEP, help="Time step size in seconds")
    p.add_argument("--workload", choices=["poisson","bursty","periodic","trace"], default="poisson")
    p.add_argument("--rate", type=float, default=30.0, help="Avg arrivals per second")
    p.add_argument("--burst-rate", type=float, default=300.0)
    p.add_argument("--burst-prob", type=float, default=0.01)
    p.add_argument("--periodic-period", type=float, default=3600.0)
    p.add_argument("--periodic-amp", type=float, default=2.0)
    p.add_argument("--jitter-std", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output path (file or directory)")
    p.add_argument("--format", type=str, choices=["csv","jsonl","parquet"], default="csv")
    p.add_argument("--files", type=int, default=DEFAULT_FILES, help="Number of files to generate")
    p.add_argument("--parallel", type=int, default=1, help="Parallel worker processes")
    p.add_argument("--compress", action="store_true", help="Compress outputs into tar.gz")
    p.add_argument("--s3-bucket", type=str, default=None)
    p.add_argument("--s3-prefix", type=str, default=None)
    p.add_argument("--target", type=str, choices=["next_step","horizon_max","binary_burst", None], default=None)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--use-mlflow", action="store_true")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--trace-path", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cli_main)
    return p

def cli_main(args):
    cfg = GenConfig(
        duration=args.duration,
        step=args.step,
        workload=args.workload,
        rate=args.rate,
        burst_rate=args.burst_rate,
        burst_prob=args.burst_prob,
        periodic_period=args.periodic_period,
        periodic_amp=args.periodic_amp,
        jitter_std=args.jitter_std,
        seed=args.seed,
        out=args.out,
        out_format=args.format,
        files=args.files,
        parallel=args.parallel,
        compress=args.compress,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        target=args.target,
        horizon=args.horizon,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        use_mlflow=args.use_mlflow,
        use_wandb=args.use_wandb,
        shuffle=args.shuffle,
        trace_path=args.trace_path,
        verbose=args.verbose
    ).finalize()

    if cfg.use_mlflow and _HAS_MLFLOW:
        try:
            mlflow.start_run(run_name=f"synthetic_gen_{int(time.time())}")
            mlflow.log_params(asdict(cfg))
        except Exception:
            LOG.exception("Failed to start MLflow run")

    if cfg.use_wandb and _HAS_WANDB:
        try:
            wandb.init(project="PriorityMax-SynthData", config=asdict(cfg))
        except Exception:
            LOG.exception("Failed to start W&B run")

    start = time.time()
    results = generate_batch(cfg)
    end = time.time()
    LOG.info("Generation completed in %.2fs. Results: %s", end - start, results)

    # splitting
    if cfg.files == 1:
        # find single output path and run split if target & splits configured
        path = results[0]["path"]
        if cfg.target and _HAS_PANDAS:
            split_res = split_and_save(path, cfg)
            LOG.info("Split result: %s", split_res)

    # finalize tracking
    if cfg.use_mlflow and _HAS_MLFLOW:
        try:
            for r in results:
                mlflow.log_metric("rows_written", r.get("rows", 0))
                mlflow.log_param(f"file_{r.get('index')}_path", r.get("path"))
            mlflow.end_run()
        except Exception:
            LOG.exception("MLflow finalization failed")

    if cfg.use_wandb and _HAS_WANDB:
        try:
            wandb.finish()
        except Exception:
            LOG.exception("W&B finalization failed")

    # compress if requested and single file output target was a directory of files
    if cfg.compress and cfg.files > 1:
        out_dir = pathlib.Path(cfg.out)
        tar_name = pathlib.Path(cfg.out).resolve().name + f"_sim_{int(time.time())}.tar.gz"
        tar_path = pathlib.Path(out_dir).parent / tar_name
        srcs = [r["path"] for r in results if "path" in r]
        compress_tar_gz(srcs, tar_path)
        LOG.info("Created archive %s", tar_path)
        if cfg.s3_bucket:
            key = f"{cfg.s3_prefix.rstrip('/')}/{tar_path.name}" if cfg.s3_prefix else tar_path.name
            uploaded = upload_to_s3(tar_path, cfg.s3_bucket, key)
            LOG.info("Uploaded archive to %s", uploaded)

# -------------------------
# Entrypoint
# -------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    args.func(args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PriorityMax Predictor Performance Benchmark
--------------------------------------------

Tests single and batch prediction performance using PredictorManager.

Metrics reported:
 - Avg latency (per sample)
 - Throughput (samples/sec)
 - CPU vs GPU utilization (if torch + CUDA)
 - Speedup factor (vectorized / single)
 - Prometheus metric scrape (optional)
"""

import os
import sys
import json
import time
import random
import asyncio
import statistics
import numpy as np
from typing import Dict, List

# Path adjustments if running standalone
sys.path.append(str(os.path.abspath(os.path.join(os.path.dirname(__file__), "../app"))))

from ml.predictor import PredictorManager, PREDICTOR_MANAGER

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

# -------- CONFIG --------
NUM_SAMPLES = int(os.getenv("PERF_NUM_SAMPLES", "5000"))
BATCH_SIZE = int(os.getenv("PERF_BATCH_SIZE", "512"))
MODEL_TYPE = os.getenv("PERF_MODEL_TYPE", "predictor_lgbm")

# Generate synthetic features
def generate_fake_features(n: int) -> List[Dict[str, float]]:
    data = []
    for i in range(n):
        data.append({
            "queue_length": random.randint(1, 100),
            "consumer_count": random.randint(1, 10),
            "priority": random.choice([0, 1, 2]),
            "cpu_util": random.uniform(0.1, 0.99),
            "mem_util": random.uniform(0.05, 0.95)
        })
    return data


async def benchmark_predictor():
    mgr: PredictorManager = PREDICTOR_MANAGER

    print(f"\n🚀 PriorityMax Predictor Benchmark")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Samples: {NUM_SAMPLES}, Batch size: {BATCH_SIZE}")
    print("-" * 60)

    # Warm up
    model = mgr.load_latest(MODEL_TYPE)
    if not model:
        print("❌ No model found in registry. Please register and activate one.")
        return

    # GPU info
    if HAS_TORCH and torch.cuda.is_available():
        print(f"🧠 CUDA device(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    else:
        print("⚙️ Using CPU-only inference")

    # Generate test data
    data = generate_fake_features(NUM_SAMPLES)

    # --- Single predictions (sequential) ---
    print("\n▶️ Running single prediction benchmark...")
    start = time.perf_counter()
    latencies = []
    for f in data:
        t0 = time.perf_counter()
        _ = mgr.predict(f, MODEL_TYPE)
        latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - start
    avg_latency = statistics.mean(latencies)
    throughput_single = NUM_SAMPLES / elapsed
    print(f"   Avg latency per sample: {avg_latency*1000:.3f} ms")
    print(f"   Throughput (samples/sec): {throughput_single:.2f}")
    print(f"   Total elapsed: {elapsed:.2f} sec")

    # --- Vectorized batch predictions ---
    print("\n⚡ Running vectorized batch benchmark...")
    start = time.perf_counter()
    preds = mgr.predict_batch(data, MODEL_TYPE, max_workers=4, max_batch_size=BATCH_SIZE)
    elapsed_vec = time.perf_counter() - start
    throughput_vec = NUM_SAMPLES / elapsed_vec
    print(f"   Total elapsed: {elapsed_vec:.2f} sec")
    print(f"   Effective throughput: {throughput_vec:.2f} samples/sec")
    print(f"   Avg latency per sample (effective): {(elapsed_vec/NUM_SAMPLES)*1000:.3f} ms")

    # --- Compare ---
    speedup = throughput_vec / throughput_single if throughput_single > 0 else float("nan")
    print("\n📈 Speedup Factor: {:.2f}×".format(speedup))

    # --- Optional GPU stats ---
    if HAS_TORCH and torch.cuda.is_available():
        try:
            util = torch.cuda.utilization()
        except Exception:
            util = None
        mem = torch.cuda.memory_allocated(0)/1e6 if torch.cuda.is_available() else 0
        print(f"   GPU Memory Usage: {mem:.2f} MB")
        if util:
            print(f"   GPU Utilization: {util}%")

    # --- Optional CPU load stats ---
    if HAS_PSUTIL:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        print(f"   CPU Utilization: {cpu}% | Memory: {mem}%")

    # --- Sanity check ---
    if preds:
        print("\n🧩 Example output sample:")
        print(json.dumps(preds[0], indent=2))

    print("\n✅ Benchmark complete.")
    print("-" * 60)
    return {
        "throughput_single": throughput_single,
        "throughput_batch": throughput_vec,
        "speedup": speedup,
        "avg_latency_single_ms": avg_latency*1000,
        "avg_latency_batch_ms": (elapsed_vec/NUM_SAMPLES)*1000
    }


if __name__ == "__main__":
    try:
        asyncio.run(benchmark_predictor())
    except KeyboardInterrupt:
        print("\nCancelled by user.")

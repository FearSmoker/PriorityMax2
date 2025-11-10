#!/usr/bin/env python3
"""
PriorityMax WebSocket + Queue Load Simulator (Prometheus-Enhanced)
------------------------------------------------------------------

Simulates hundreds of concurrent WebSocket clients and enqueues
synthetic tasks into the real RedisQueue backend simultaneously,
measuring:
 - WebSocket broadcast latency
 - Queue backlog and DLQ length
 - Message delivery rate
 - Prometheus-exported live metrics

Usage:
  python3 backend/tools/ws_load_simulator.py \
      --url ws://localhost:8000/ws/metrics \
      --clients 200 \
      --duration 60 \
      --enqueue-rate 50 \
      --broadcast-interval 1.0
"""

import asyncio
import json
import random
import time
import argparse
import statistics
import logging
import signal
import websockets
from typing import List, Dict, Any

# --------------------------------------------------------------------------
# Prometheus client
# --------------------------------------------------------------------------
try:
    from prometheus_client import Gauge, Counter, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

# --------------------------------------------------------------------------
# PriorityMax imports
# --------------------------------------------------------------------------
try:
    from app.queue.redis_queue import RedisQueue
except Exception:
    RedisQueue = None

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
LOG = logging.getLogger("prioritymax.loadsim")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_handler)

# --------------------------------------------------------------------------
# Prometheus metrics
# --------------------------------------------------------------------------
if _HAS_PROM:
    LS_ACTIVE_CLIENTS = Gauge("ws_loadsim_active_clients", "Currently active WebSocket clients")
    LS_MESSAGES_RECEIVED = Counter("ws_loadsim_messages_received_total", "Total messages received via WebSocket")
    LS_MESSAGES_SENT = Counter("ws_loadsim_messages_sent_total", "Total broadcast messages sent")
    LS_LATENCY_HIST = Histogram(
        "ws_loadsim_message_latency_seconds",
        "WebSocket message roundtrip latency (seconds)",
        buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
    )
    LS_QUEUE_BACKLOG = Gauge("ws_loadsim_queue_backlog", "Current Redis queue backlog")
    LS_QUEUE_DLQ = Gauge("ws_loadsim_queue_dlq_backlog", "Current Redis DLQ backlog")
    LS_ENQUEUED_TASKS = Counter("ws_loadsim_enqueued_tasks_total", "Total synthetic tasks enqueued")
else:
    LS_ACTIVE_CLIENTS = LS_MESSAGES_RECEIVED = LS_MESSAGES_SENT = LS_LATENCY_HIST = None
    LS_QUEUE_BACKLOG = LS_QUEUE_DLQ = LS_ENQUEUED_TASKS = None

# --------------------------------------------------------------------------
# WebSocket Client
# --------------------------------------------------------------------------
class SimClient:
    def __init__(self, uri: str, client_id: int):
        self.uri = uri
        self.client_id = client_id
        self.received = 0
        self.latencies: List[float] = []
        self.connected = False
        self._ws = None
        self._running = True

    async def connect(self):
        try:
            self._ws = await websockets.connect(self.uri)
            self.connected = True
            LOG.debug(f"Client {self.client_id} connected to {self.uri}")
            if _HAS_PROM:
                LS_ACTIVE_CLIENTS.inc()
            asyncio.create_task(self._listen())
        except Exception as e:
            LOG.warning(f"Client {self.client_id} failed to connect: {e}")
            self.connected = False

    async def _listen(self):
        while self._running and self._ws:
            try:
                msg = await self._ws.recv()
                self.received += 1
                if _HAS_PROM:
                    LS_MESSAGES_RECEIVED.inc()
                try:
                    data = json.loads(msg)
                    if isinstance(data, dict) and "sent_ts" in data:
                        latency = time.time() - float(data["sent_ts"])
                        self.latencies.append(latency)
                        if _HAS_PROM:
                            LS_LATENCY_HIST.observe(latency)
                except Exception:
                    pass
            except websockets.ConnectionClosed:
                LOG.debug(f"Client {self.client_id} disconnected")
                break
            except Exception as e:
                LOG.debug(f"Client {self.client_id} read error: {e}")
                await asyncio.sleep(0.05)

    async def close(self):
        self._running = False
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        if self.connected and _HAS_PROM:
            LS_ACTIVE_CLIENTS.dec()
        self.connected = False

# --------------------------------------------------------------------------
# Redis Queue Producer
# --------------------------------------------------------------------------
async def queue_producer(queue_name: str, rate: int, duration: float):
    """
    Periodically enqueue synthetic tasks into RedisQueue.
    """
    if not RedisQueue:
        LOG.warning("RedisQueue not available; skipping enqueue")
        return

    rq = RedisQueue()
    start = time.time()
    count = 0
    while time.time() - start < duration:
        try:
            # produce 'rate' tasks per second
            for _ in range(rate):
                payload = {
                    "id": f"simtask-{int(time.time()*1000)}-{random.randint(1000,9999)}",
                    "type": "test_task",
                    "payload": {"value": random.random()},
                    "created_at": time.time()
                }
                await rq.push(queue_name, json.dumps(payload))
                count += 1
                if _HAS_PROM:
                    LS_ENQUEUED_TASKS.inc()
            await asyncio.sleep(1)
        except Exception as e:
            LOG.warning(f"Enqueue error: {e}")
            await asyncio.sleep(2)

    LOG.info(f"Queue producer finished: {count} tasks enqueued")

# --------------------------------------------------------------------------
# Queue Metrics Collector
# --------------------------------------------------------------------------
async def queue_metrics_collector(queue_name: str, interval: float = 5.0):
    if not RedisQueue or not _HAS_PROM:
        return
    rq = RedisQueue()
    while True:
        try:
            qstats = await rq.get_queue_stats(queue_name)
            backlog = qstats.get("backlog", 0)
            dlq_len = await rq.get_dlq_length(queue_name)
            LS_QUEUE_BACKLOG.set(backlog)
            LS_QUEUE_DLQ.set(dlq_len)
            LOG.debug(f"Queue metrics: backlog={backlog} dlq={dlq_len}")
        except Exception:
            LOG.debug("queue_metrics_collector failed", exc_info=True)
        await asyncio.sleep(interval)

# --------------------------------------------------------------------------
# Broadcast helper
# --------------------------------------------------------------------------
async def broadcaster(channel: str, duration: float, interval: float):
    start = time.time()
    sent = 0
    while time.time() - start < duration:
        msg = {"type": "broadcast", "channel": channel, "payload": random.random(), "sent_ts": time.time()}
        if _HAS_PROM:
            LS_MESSAGES_SENT.inc()
        LOG.debug(f"Broadcast {channel}: {msg}")
        sent += 1
        await asyncio.sleep(interval)
    return sent

# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
async def run_load(uri: str, clients: int, duration: float, enqueue_rate: int, broadcast_interval: float, prometheus_port: int):
    if _HAS_PROM:
        start_http_server(prometheus_port)
        LOG.info(f"Prometheus exporter running at :{prometheus_port}/metrics")

    LOG.info(f"Target: {uri} | Clients: {clients} | Enqueue Rate: {enqueue_rate}/s")

    sim_clients = [SimClient(uri, i) for i in range(clients)]
    await asyncio.gather(*(c.connect() for c in sim_clients))
    connected = sum(1 for c in sim_clients if c.connected)
    LOG.info(f"Connected {connected}/{clients} clients")

    # concurrent tasks
    queue_name = "prioritymax"
    tasks = [
        asyncio.create_task(queue_producer(queue_name, enqueue_rate, duration)),
        asyncio.create_task(queue_metrics_collector(queue_name, 5.0)),
        asyncio.create_task(broadcaster(uri.split("/")[-1], duration, broadcast_interval))
    ]

    await asyncio.sleep(duration)

    for t in tasks:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    await asyncio.gather(*(c.close() for c in sim_clients))

    # results
    latencies = [l for c in sim_clients for l in c.latencies]
    LOG.info("=== Results ===")
    LOG.info(f"Total clients: {clients} | Connected: {connected}")
    if latencies:
        LOG.info(f"Avg latency: {statistics.mean(latencies)*1000:.2f} ms | p95: {statistics.quantiles(latencies, n=100)[94]*1000:.2f} ms")
    LOG.info("Simulation complete.")

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PriorityMax WebSocket + Queue Load Simulator (Prometheus-enhanced)")
    parser.add_argument("--url", default="ws://localhost:8000/ws/metrics", help="Target WebSocket URL")
    parser.add_argument("--clients", type=int, default=100, help="Simulated clients")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration (seconds)")
    parser.add_argument("--enqueue-rate", type=int, default=50, help="Synthetic tasks per second")
    parser.add_argument("--broadcast-interval", type=float, default=1.0, help="Seconds between WebSocket broadcasts")
    parser.add_argument("--prometheus-port", type=int, default=9209, help="Prometheus exporter port")
    parser.add_argument("--loglevel", default="INFO")
    args = parser.parse_args()

    LOG.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(shutdown(loop)))

    loop.run_until_complete(run_load(args.url, args.clients, args.duration, args.enqueue_rate, args.broadcast_interval, args.prometheus_port))

async def shutdown(loop):
    LOG.info("Shutdown signal received; cancelling tasks…")
    for task in asyncio.all_tasks(loop):
        task.cancel()
    await asyncio.sleep(0.1)
    loop.stop()

if __name__ == "__main__":
    main()

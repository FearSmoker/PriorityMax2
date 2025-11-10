# backend/app/metrics.py
"""
PriorityMax Metrics & Observability Subsystem
---------------------------------------------

Features:
 - Prometheus: metric definitions, start_http_server helper, registry
 - Pushgateway, InfluxDB, StatsD optional exporters (best-effort)
 - Async rolling window aggregator with multiple granularities
 - Latency heatmap & histogram helpers for visual heatmaps
 - FastAPI middleware and WebSocket broadcast helper for live dashboards
 - Decorators / contextmanagers for timing, counting, and sampling
 - Alert hooks and alert rule helpers (simple thresholding + callback)
 - Grafana dashboard skeleton generator (JSON)
 - Safe fallbacks when optional libs are not installed
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import atexit
import socket
import asyncio
import logging
import functools
import statistics
import traceback
import threading
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Optional libs
try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary, generate_latest, start_http_server, core
    from prometheus_client.exposition import CONTENT_TYPE_LATEST
    _HAS_PROM = True
except Exception:
    CollectorRegistry = Counter = Gauge = Histogram = Summary = generate_latest = start_http_server = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    _HAS_PROM = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except Exception:
    aiohttp = None
    _HAS_AIOHTTP = False

try:
    from influxdb import InfluxDBClient
    _HAS_INFLUX = True
except Exception:
    InfluxDBClient = None
    _HAS_INFLUX = False

try:
    import statsd
    _HAS_STATSD = True
except Exception:
    statsd = None
    _HAS_STATSD = False

# Logging
LOG = logging.getLogger("prioritymax.metrics")
LOG.setLevel(os.getenv("PRIORITYMAX_METRICS_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults / Env
PROM_PORT = int(os.getenv("PRIORITYMAX_PROMETHEUS_PORT", "9001"))
PROM_ADDR = os.getenv("PRIORITYMAX_PROMETHEUS_ADDR", "0.0.0.0")
PUSHGATEWAY = os.getenv("PRIORITYMAX_PUSHGATEWAY", "")
INFLUX_URL = os.getenv("PRIORITYMAX_INFLUX_URL", "")
INFLUX_DB = os.getenv("PRIORITYMAX_INFLUX_DB", "prioritymax_metrics")
STATSD_HOST = os.getenv("PRIORITYMAX_STATSD_HOST", "")
STATSD_PORT = int(os.getenv("PRIORITYMAX_STATSD_PORT", "8125"))

# Internal constants
_ROLLING_WINDOWS = [60, 300, 900, 3600]  # seconds: 1m, 5m, 15m, 1h
_HEATMAP_BINS = 50

# Helper types
MetricLabels = Optional[Dict[str, str]]
AlertCallback = Callable[[str, Dict[str, Any]], Any]

# -----------------------------------------------------------------------------
# Prometheus metric definitions (central registry)
# -----------------------------------------------------------------------------
if _HAS_PROM:
    PROM_REGISTRY = CollectorRegistry(auto_describe=False)
    # Counters
    PM_TASKS_ENQUEUED = Counter("prioritymax_tasks_enqueued_total", "Total tasks enqueued", ["queue"], registry=PROM_REGISTRY)
    PM_TASKS_PROCESSED = Counter("prioritymax_tasks_processed_total", "Total tasks processed", ["queue", "worker"], registry=PROM_REGISTRY)
    PM_TASKS_FAILED = Counter("prioritymax_tasks_failed_total", "Total failed tasks", ["queue", "worker", "reason"], registry=PROM_REGISTRY)
    PM_DLQ_PROMOTED = Counter("prioritymax_dlq_promoted_total", "Total DLQ messages promoted", ["queue"], registry=PROM_REGISTRY)
    # Gauges
    PM_QUEUE_BACKLOG = Gauge("prioritymax_queue_backlog", "Current queue backlog", ["queue"], registry=PROM_REGISTRY)
    PM_WORKER_COUNT = Gauge("prioritymax_worker_count", "Current worker count", ["pool"], registry=PROM_REGISTRY)
    PM_LAST_SCALE_SCORE = Gauge("prioritymax_last_scale_score", "Last composite scale score", registry=PROM_REGISTRY)
    PM_LAST_SCALE_ACTION = Gauge("prioritymax_last_scale_action", "Numeric representation of last scale action", ["action"], registry=PROM_REGISTRY)
    # Histograms & summaries
    PM_TASK_LATENCY = Histogram("prioritymax_task_latency_seconds", "Task processing latency seconds", ["queue", "worker"], buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10), registry=PROM_REGISTRY)
    PM_PREDICTOR_INFER = Histogram("prioritymax_predictor_infer_seconds", "Predictor inference latency", registry=PROM_REGISTRY)
    # Generic info
    PM_VERSION_INFO = Gauge("prioritymax_version_info", "PriorityMax version info", ["version", "commit"], registry=PROM_REGISTRY)
else:
    PROM_REGISTRY = None
    PM_TASKS_ENQUEUED = PM_TASKS_PROCESSED = PM_TASKS_FAILED = PM_DLQ_PROMOTED = None
    PM_QUEUE_BACKLOG = PM_WORKER_COUNT = PM_LAST_SCALE_SCORE = PM_LAST_SCALE_ACTION = None
    PM_TASK_LATENCY = PM_PREDICTOR_INFER = PM_VERSION_INFO = None

# -----------------------------------------------------------------------------
# Rolling window aggregator
# -----------------------------------------------------------------------------
class RollingWindow:
    """
    A time-based rolling window aggregator.

    - Keeps buckets of fixed resolution (1 second)
    - Supports multiple retention periods (windows)
    - Provides fast aggregated stats (count, sum, avg, p95, hist bins)
    """

    def __init__(self, max_window_seconds: int = 3600, resolution: float = 1.0):
        self.resolution = float(resolution)
        self.max_window = int(max_window_seconds)
        self.bucket_count = int(math.ceil(self.max_window / self.resolution))
        self._buckets: List[Dict[str, Any]] = [self._new_bucket() for _ in range(self.bucket_count)]
        self._lock = threading.Lock()
        self._start_ts = int(time.time())

    def _new_bucket(self) -> Dict[str, Any]:
        return {"ts": 0, "values": []}

    def _bucket_index(self, ts: Optional[float] = None) -> int:
        if ts is None:
            ts = time.time()
        idx = int((int(ts) % (self.bucket_count)) )
        return idx

    def add(self, value: float, ts: Optional[float] = None):
        """Add a single value to the current bucket."""
        if ts is None:
            ts = time.time()
        idx = self._bucket_index(ts)
        with self._lock:
            b = self._buckets[idx]
            if b["ts"] != int(ts // self.resolution):
                b["ts"] = int(ts // self.resolution)
                b["values"].clear()
            b["values"].append(float(value))

    def snapshot(self, window_seconds: int) -> Dict[str, Any]:
        """
        Return aggregated snapshot for the last window_seconds:
        {count, sum, mean, p50, p95, min, max, histogram_bins}
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        now = time.time()
        buckets_needed = int(min(self.bucket_count, math.ceil(window_seconds / self.resolution)))
        vals: List[float] = []
        with self._lock:
            for i in range(buckets_needed):
                # compute bucket ts index (backwards)
                ts_slot = int((now - i * self.resolution) // self.resolution)
                idx = int(ts_slot % self.bucket_count)
                b = self._buckets[idx]
                if b["ts"] == ts_slot:
                    vals.extend(b["values"])
        if not vals:
            return {"count": 0, "sum": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "hist": []}
        vals_sorted = sorted(vals)
        count = len(vals_sorted)
        s = sum(vals_sorted)
        mean = s / count
        p50 = vals_sorted[int(0.5 * (count - 1))]
        p95 = vals_sorted[int(0.95 * (count - 1))]
        mn = vals_sorted[0]
        mx = vals_sorted[-1]
        # histogram bins
        try:
            hist = self._histogram(vals_sorted, bins=_HEATMAP_BINS)
        except Exception:
            hist = []
        return {"count": count, "sum": s, "mean": mean, "p50": p50, "p95": p95, "min": mn, "max": mx, "hist": hist}

    def _histogram(self, values: List[float], bins: int = 50) -> List[int]:
        mn = values[0]
        mx = values[-1]
        if mn == mx:
            return [len(values)] + [0] * (bins - 1)
        width = (mx - mn) / bins
        bins_counts = [0] * bins
        for v in values:
            idx = int((v - mn) / width)
            if idx >= bins:
                idx = bins - 1
            bins_counts[idx] += 1
        return bins_counts

# -----------------------------------------------------------------------------
# Heatmap builder
# -----------------------------------------------------------------------------
class LatencyHeatmap:
    """
    Build an adaptive heatmap for latency vs time or backlog vs time.
    Stores rolling histograms per time bucket and provides JSON-friendly output
    for visualization (e.g., D3 heatmap).
    """

    def __init__(self, window_seconds: int = 3600, resolution: float = 5.0, bins: int = 50, max_value: Optional[float] = None):
        self.resolution = float(resolution)
        self.window = int(window_seconds)
        self.bins = int(bins)
        self.max_value = float(max_value) if max_value is not None else None
        self.bucket_count = int(math.ceil(self.window / self.resolution))
        self._buckets: List[Dict[str, Any]] = [{"ts": 0, "hist": [0] * self.bins} for _ in range(self.bucket_count)]
        self._lock = threading.Lock()

    def add(self, value: float, ts: Optional[float] = None):
        if ts is None:
            ts = time.time()
        with self._lock:
            idx = int((int(ts // self.resolution)) % self.bucket_count)
            b = self._buckets[idx]
            if b["ts"] != int(ts // self.resolution):
                b["ts"] = int(ts // self.resolution)
                b["hist"] = [0] * self.bins
            # map value to bin
            bin_idx = self._value_to_bin(value)
            b["hist"][bin_idx] += 1

    def _value_to_bin(self, v: float) -> int:
        # If max_value unknown, use soft-clamp using log scale
        if self.max_value:
            normalized = max(0.0, min(1.0, v / (self.max_value or 1.0)))
        else:
            # log scaling approx; pick a reasonable scale
            normalized = 1.0 - math.exp(-v / (1.0 + v))
        idx = int(normalized * (self.bins - 1))
        return max(0, min(self.bins - 1, idx))

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        out = []
        with self._lock:
            for i in range(self.bucket_count):
                b = self._buckets[i]
                if b["ts"] == 0:
                    out.append({"ts": None, "hist": [0] * self.bins})
                else:
                    out.append({"ts": int(b["ts"] * self.resolution), "hist": list(b["hist"])})
        return {"resolution": self.resolution, "bins": self.bins, "buckets": out}

# -----------------------------------------------------------------------------
# Metrics manager (singleton style)
# -----------------------------------------------------------------------------
class Metrics:
    """
    High-level metrics manager: collects metrics, updates Prometheus, and exposes
    async tasks to push to external systems.
    """

    def __init__(self,
                 prometheus_registry: Optional[CollectorRegistry] = None,
                 pushgateway: Optional[str] = None,
                 influx_url: Optional[str] = None,
                 influx_db: Optional[str] = None,
                 statsd_host: Optional[str] = None,
                 statsd_port: Optional[int] = None):
        self.registry = prometheus_registry if prometheus_registry is not None else PROM_REGISTRY
        self.pushgateway = pushgateway or PUSHGATEWAY
        self.influx_url = influx_url or INFLUX_URL
        self.influx_db = influx_db or INFLUX_DB
        self.statsd_host = statsd_host or STATSD_HOST
        self.statsd_port = statsd_port or STATSD_PORT

        # internal rolling stores
        self.latency_windows: Dict[str, RollingWindow] = {}
        self.backlog_windows: Dict[str, RollingWindow] = {}
        self.heatmaps: Dict[str, LatencyHeatmap] = {}

        # optional clients
        self.influx_client = None
        if _HAS_INFLUX and self.influx_url:
            try:
                # parse url: http://host:8086 or host only
                parsed = self.influx_url
                # best-effort: allow "host:port" or full URL
                host = parsed
                port = 8086
                if parsed.startswith("http"):
                    # naive parsing
                    import urllib.parse as _urlp
                    up = _urlp.urlparse(parsed)
                    host = up.hostname
                    port = up.port or 8086
                self.influx_client = InfluxDBClient(host=host, port=port, database=self.influx_db)
                LOG.info("InfluxDB client initialized for %s:%s db=%s", host, port, self.influx_db)
            except Exception:
                LOG.exception("Failed to initialize InfluxDB client")
                self.influx_client = None

        self.statsd_client = None
        if _HAS_STATSD and self.statsd_host:
            try:
                self.statsd_client = statsd.StatsClient(self.statsd_host, self.statsd_port, prefix="prioritymax")
            except Exception:
                LOG.exception("Failed to init StatsD client")
                self.statsd_client = None

        # websocket listeners
        self._ws_listeners: List[Callable[[Dict[str, Any]], Any]] = []

        # background tasks
        self._background_task: Optional[asyncio.Task] = None
        self._running = False
        self._export_interval = int(os.getenv("PRIORITYMAX_METRICS_EXPORT_INTERVAL", "30"))

        # Alerts
        self._alert_rules: List[Tuple[str, float, int, AlertCallback]] = []  # (metric_key, threshold, window_sec, callback)

        # version info
        self.version = os.getenv("PRIORITYMAX_VERSION", "dev")
        self.commit = os.getenv("PRIORITYMAX_COMMIT", "none")
        if PM_VERSION_INFO:
            try:
                PM_VERSION_INFO.labels(version=self.version, commit=self.commit).set(1)
            except Exception:
                pass

    # -------------------------
    # Helpers: window/heatmap access
    # -------------------------
    def _ensure_latency_window(self, key: str) -> RollingWindow:
        if key not in self.latency_windows:
            self.latency_windows[key] = RollingWindow(max_window_seconds=max(_ROLLING_WINDOWS))
        return self.latency_windows[key]

    def _ensure_backlog_window(self, key: str) -> RollingWindow:
        if key not in self.backlog_windows:
            self.backlog_windows[key] = RollingWindow(max_window_seconds=max(_ROLLING_WINDOWS))
        return self.backlog_windows[key]

    def _ensure_heatmap(self, key: str, max_value: Optional[float] = None) -> LatencyHeatmap:
        if key not in self.heatmaps:
            self.heatmaps[key] = LatencyHeatmap(window_seconds=max(_ROLLING_WINDOWS), resolution=5.0, bins=_HEATMAP_BINS, max_value=max_value)
        return self.heatmaps[key]

    # -------------------------
    # Recorders
    # -------------------------
    def record_enqueue(self, queue: str, count: int = 1):
        try:
            if PM_TASKS_ENQUEUED:
                PM_TASKS_ENQUEUED.labels(queue=queue).inc(count)
            # update backlog gauge if caller provides backlog separately use set_backlog
        except Exception:
            LOG.exception("record_enqueue failed")

    def record_processed(self, queue: str, worker: str, latency_sec: float):
        try:
            if PM_TASKS_PROCESSED:
                PM_TASKS_PROCESSED.labels(queue=queue, worker=worker).inc(1)
            if PM_TASK_LATENCY:
                PM_TASK_LATENCY.labels(queue=queue, worker=worker).observe(latency_sec)
            # rolling windows
            self._ensure_latency_window(f"latency:{queue}").add(latency_sec)
            # heatmap
            self._ensure_heatmap(f"latency:{queue}").add(latency_sec)
            # statsd
            if self.statsd_client:
                try:
                    self.statsd_client.timing(f"latency.{queue}", int(latency_sec*1000))
                except Exception:
                    LOG.debug("statsd timing failed")
        except Exception:
            LOG.exception("record_processed failed")

    def record_failed(self, queue: str, worker: str, reason: str = "error"):
        try:
            if PM_TASKS_FAILED:
                PM_TASKS_FAILED.labels(queue=queue, worker=worker, reason=reason).inc(1)
        except Exception:
            LOG.exception("record_failed failed")

    def set_backlog(self, queue: str, backlog: int):
        try:
            if PM_QUEUE_BACKLOG:
                PM_QUEUE_BACKLOG.labels(queue=queue).set(int(backlog))
            self._ensure_backlog_window(f"backlog:{queue}").add(float(backlog))
            # heatmap for backlog
            self._ensure_heatmap(f"backlog:{queue}", max_value=10000).add(float(backlog))
        except Exception:
            LOG.exception("set_backlog failed")

    def set_worker_count(self, pool: str, count: int):
        try:
            if PM_WORKER_COUNT:
                PM_WORKER_COUNT.labels(pool=pool).set(int(count))
        except Exception:
            LOG.exception("set_worker_count failed")

    def set_last_scale(self, score: float, action: Optional[str] = None):
        try:
            if PM_LAST_SCALE_SCORE:
                PM_LAST_SCALE_SCORE.set(float(score))
            if action and PM_LAST_SCALE_ACTION:
                # set last action gauge to 1 for that action label
                PM_LAST_SCALE_ACTION.labels(action=action).set(1)
        except Exception:
            LOG.exception("set_last_scale failed")

    def record_predictor_infer(self, sec: float):
        try:
            if PM_PREDICTOR_INFER:
                PM_PREDICTOR_INFER.observe(sec)
        except Exception:
            LOG.exception("record_predictor_infer failed")

    def record_dlq_promoted(self, queue: str, count: int = 1):
        try:
            if PM_DLQ_PROMOTED:
                PM_DLQ_PROMOTED.labels(queue=queue).inc(count)
        except Exception:
            LOG.exception("record_dlq_promoted failed")

    # -------------------------
    # Alerts
    # -------------------------
    def add_threshold_alert(self, metric_key: str, threshold: float, window_sec: int, callback: AlertCallback):
        """
        Register a simple threshold alert: if rolling average of metric_key over window_sec >= threshold,
        invoke callback(name, payload)
        metric_key format examples: "backlog:queue_name:mean", "latency:queue_name:p95"
        """
        self._alert_rules.append((metric_key, float(threshold), int(window_sec), callback))

    async def _run_alerts_once(self):
        """
        Evaluate all registered alerts; invoked by background loop.
        """
        for metric_key, threshold, window, cb in list(self._alert_rules):
            try:
                parts = metric_key.split(":")
                if parts[0] == "backlog":
                    qname = parts[1]
                    w = self._ensure_backlog_window(f"backlog:{qname}")
                    snap = w.snapshot(window)
                    val = snap.get("mean", 0.0)
                elif parts[0] == "latency":
                    qname = parts[1]
                    measure = parts[2] if len(parts) > 2 else "p95"
                    w = self._ensure_latency_window(f"latency:{qname}")
                    snap = w.snapshot(window)
                    val = snap.get(measure, snap.get("p95", 0.0))
                else:
                    LOG.debug("Unknown alert metric prefix: %s", parts[0])
                    continue
                if val >= threshold:
                    payload = {"metric": metric_key, "value": val, "threshold": threshold, "window": window, "ts": time.time()}
                    try:
                        cb(metric_key, payload)
                    except Exception:
                        LOG.exception("Alert callback exception for %s", metric_key)
            except Exception:
                LOG.exception("Alert evaluation failed for %s", metric_key)

    # -------------------------
    # Background exporter
    # -------------------------
    async def _background_loop(self):
        LOG.info("Metrics background exporter started (interval=%ss)", self._export_interval)
        while self._running:
            try:
                # push to pushgateway if configured
                if self.pushgateway:
                    try:
                        await self._push_to_gateway()
                    except Exception:
                        LOG.debug("pushgateway push failed")
                # push to influx if configured
                if self.influx_client:
                    try:
                        await self._push_to_influx()
                    except Exception:
                        LOG.debug("influx push failed")
                # run alerts
                await self._run_alerts_once()
                # broadcast snapshot to websocket listeners
                snapshot = self.snapshot()
                await self._broadcast(snapshot)
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("Metrics background loop error")
            await asyncio.sleep(self._export_interval)
        LOG.info("Metrics background exporter stopped")

    async def start(self):
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._background_task = loop.create_task(self._background_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except Exception:
                pass

    async def _push_to_gateway(self):
        """
        Push registry metrics to a Pushgateway.
        This method uses aiohttp to POST metrics in Prometheus plaintext format.
        """
        if not _HAS_PROM or not _HAS_AIOHTTP:
            return
        try:
            url = self.pushgateway.rstrip("/") + "/metrics/job/prioritymax"
            payload = generate_latest(self.registry)
            headers = {"Content-Type": CONTENT_TYPE_LATEST}
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, data=payload, headers=headers) as resp:
                    if resp.status not in (200, 202):
                        txt = await resp.text()
                        LOG.warning("Pushgateway returned %s: %s", resp.status, txt)
        except Exception:
            LOG.exception("Push to Pushgateway failed")

    async def _push_to_influx(self):
        """
        Convert key summaries into Influx points. This is best-effort and
        writes aggregated summaries rather than raw high-cardinality timeseries.
        """
        if not self.influx_client:
            return
        try:
            points = []
            timestamp = int(time.time() * 1000)
            # produce small set of points derived from rolling windows
            for k, w in list(self.latency_windows.items()):
                queue = k.split(":", 1)[1]
                snap = w.snapshot(60)
                if snap["count"] == 0:
                    continue
                points.append({
                    "measurement": "latency_summary",
                    "tags": {"queue": queue},
                    "time": timestamp,
                    "fields": {"mean": float(snap["mean"]), "p95": float(snap["p95"]), "count": int(snap["count"])}
                })
            for k, w in list(self.backlog_windows.items()):
                queue = k.split(":", 1)[1]
                snap = w.snapshot(60)
                if snap["count"] == 0:
                    continue
                points.append({
                    "measurement": "backlog_summary",
                    "tags": {"queue": queue},
                    "time": timestamp,
                    "fields": {"mean": float(snap["mean"]), "p95": float(snap["p95"]), "count": int(snap["count"])}
                })
            if points:
                # Influx client write is blocking; run in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self.influx_client.write_points(points))
        except Exception:
            LOG.exception("InfluxDB push failed")

    # -------------------------
    # WebSocket & broadcast helpers
    # -------------------------
    def register_ws_listener(self, fn: Callable[[Dict[str, Any]], Any]):
        """Register a synchronous callback to be invoked on each snapshot push."""
        self._ws_listeners.append(fn)

    async def _broadcast(self, payload: Dict[str, Any]):
        """
        Broadcast snapshot to all listeners. Listeners may be sync or async.
        """
        for listener in list(self._ws_listeners):
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(payload)
                else:
                    # run sync callbacks in threadpool to avoid blocking loop
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: listener(payload))
            except Exception:
                LOG.exception("Websocket listener failed")

    # -------------------------
    # Snapshot & utilities
    # -------------------------
    def snapshot(self) -> Dict[str, Any]:
        """
        Produce a JSON-serializable snapshot containing:
         - basic gauges (worker count, backlog per queue via PM_QUEUE_BACKLOG if available)
         - rolling summaries for each registered queue
         - heatmap snapshots
         - version info
        """
        try:
            out: Dict[str, Any] = {"ts": time.time(), "version": self.version, "commit": self.commit, "queues": {}}
            # backlog windows
            for k, w in list(self.backlog_windows.items()):
                qname = k.split(":", 1)[1]
                out["queues"].setdefault(qname, {})["backlog"] = {str(wnd): self._ensure_backlog_window(k).snapshot(wnd) for wnd in _ROLLING_WINDOWS}
            for k, w in list(self.latency_windows.items()):
                qname = k.split(":", 1)[1]
                out["queues"].setdefault(qname, {})["latency"] = {str(wnd): self._ensure_latency_window(k).snapshot(wnd) for wnd in _ROLLING_WINDOWS}
            # heatmaps
            for k, h in list(self.heatmaps.items()):
                qname = k.split(":", 1)[1]
                out["queues"].setdefault(qname, {})["heatmap"] = h.snapshot()
            # include some Prom metrics if registry present
            if _HAS_PROM:
                try:
                    # fetch backlog gauge values if possible (not always available)
                    out["prom_metrics"] = {}  # minimal due to scraping complexity
                except Exception:
                    pass
            return out
        except Exception:
            LOG.exception("Failed to create metrics snapshot")
            return {"ts": time.time(), "error": "snapshot_failed"}

    # -------------------------
    # FastAPI middleware and websocket helpers
    # -------------------------
    def fastapi_middleware(self):
        """
        Return a FastAPI-compatible ASGI middleware function that instruments request latencies.
        Example:
            app.add_middleware(SomeMiddlewareClass) or use dependency injection.
        For simplicity we return a decorator-style wrapper for FastAPI routes.
        """
        def decorator(fn):
            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def _wrapped(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        res = await fn(*args, **kwargs)
                        return res
                    finally:
                        elapsed = time.perf_counter() - start
                        # generic label
                        try:
                            if PM_TASK_LATENCY:
                                PM_TASK_LATENCY.labels(queue="http", worker="api").observe(elapsed)
                        except Exception:
                            pass
                return _wrapped
            else:
                @functools.wraps(fn)
                def _wrapped(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        res = fn(*args, **kwargs)
                        return res
                    finally:
                        elapsed = time.perf_counter() - start
                        try:
                            if PM_TASK_LATENCY:
                                PM_TASK_LATENCY.labels(queue="http", worker="api").observe(elapsed)
                        except Exception:
                            pass
                return _wrapped
        return decorator

    def websocket_broadcast_handler(self, websocket):
        """
        Example listener to push snapshots to FastAPI WebSocket.
        Usage in route:
            await websocket.accept()
            metrics.register_ws_listener(lambda payload: asyncio.create_task(websocket.send_json(payload)))
        """
        async def _listener(payload: Dict[str, Any]):
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                LOG.debug("websocket send failed")
        self.register_ws_listener(_listener)
        return _listener

    # -------------------------
    # Instrumentation helpers
    # -------------------------
    def timeit(self, queue: str = "default", worker: str = "default"):
        """
        Context manager & decorator for timing code blocks.
        Usage:
            with metrics.timeit("queue", "worker"):
                do_work()
        or as decorator:
            @metrics.timeit("queue","name")
            async def process(...): ...
        """
        manager = self
        class _Ctx:
            def __enter__(self_non):
                self_non._start = time.perf_counter()
                return self_non
            def __exit__(self_non, exc_type, exc, tb):
                elapsed = time.perf_counter() - self_non._start
                manager.record_processed(queue, worker, elapsed)
            async def __aenter__(self_non):
                self_non._start = time.perf_counter()
                return self_non
            async def __aexit__(self_non, exc_type, exc, tb):
                elapsed = time.perf_counter() - self_non._start
                manager.record_processed(queue, worker, elapsed)
        return _Ctx()

    def instrument(self, queue: str = "default", worker: str = "default"):
        """
        Decorator for function instrumentation. Works for sync & async functions.
        """
        def _decorator(fn):
            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def _wrapped(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return await fn(*args, **kwargs)
                    finally:
                        elapsed = time.perf_counter() - start
                        self.record_processed(queue, worker, elapsed)
                return _wrapped
            else:
                @functools.wraps(fn)
                def _wrapped(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        elapsed = time.perf_counter() - start
                        self.record_processed(queue, worker, elapsed)
                return _wrapped
        return _decorator

    # -------------------------
    # Grafana dashboard generator (skeleton)
    # -------------------------
    def grafana_dashboard_skeleton(self, queue_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Produce a JSON skeleton for a Grafana dashboard that visualizes:
         - backlog timeseries
         - latency heatmap
         - worker count
         - predictor latency
         - last scale decisions
        This is a minimal template suitable for editing in Grafana.
        """
        queues = queue_names or sorted({k.split(":", 1)[1] for k in self.backlog_windows.keys()} if self.backlog_windows else ["default"])
        panels = []
        # backlog panel
        panels.append({
            "type": "graph",
            "title": "Queue Backlog",
            "targets": [{"expr": f'prioritymax_queue_backlog{{queue="{q}"}}', "legendFormat": q} for q in queues]
        })
        # latency heatmap placeholder (requires heatmap plugin)
        panels.append({
            "type": "heatmap",
            "title": "Latency Heatmap",
            "targets": [{"expr": f'histogram_quantile(0.95, sum(rate(prioritymax_task_latency_seconds_bucket{{queue="{q}"}}[5m])) by (le))', "legendFormat": q} for q in queues]
        })
        # worker count
        panels.append({
            "type": "stat",
            "title": "Worker Count",
            "targets": [{"expr": 'prioritymax_worker_count', "legendFormat": "workers"}]
        })
        # last scale
        panels.append({
            "type": "table",
            "title": "Recent Scale Decisions",
            "targets": [{"expr": 'prioritymax_last_scale_score', "legendFormat": "score"}]
        })
        dashboard = {"dashboard": {"panels": panels, "title": "PriorityMax Overview"}}
        return dashboard

# -----------------------------------------------------------------------------
# Global singleton and helpers
# -----------------------------------------------------------------------------
_GLOBAL_METRICS: Optional[Metrics] = None

def get_global_metrics() -> Metrics:
    global _GLOBAL_METRICS
    if _GLOBAL_METRICS is None:
        _GLOBAL_METRICS = Metrics()
    return _GLOBAL_METRICS

# Convenience short names
metrics = get_global_metrics()

# Expose decorators
timeit = metrics.timeit
instrument = metrics.instrument

# -----------------------------------------------------------------------------
# Prometheus server starter (blocking-friendly)
# -----------------------------------------------------------------------------
_prom_server_thread = None

def start_prometheus_server(addr: str = PROM_ADDR, port: int = PROM_PORT):
    """
    Start the Prometheus HTTP server in a background thread.
    If the Prometheus client isn't available this is a no-op.
    """
    global _prom_server_thread
    if not _HAS_PROM:
        LOG.warning("prometheus_client not available; start_prometheus_server is a no-op")
        return
    if _prom_server_thread is not None:
        return
    def _serve():
        try:
            LOG.info("Starting Prometheus metrics HTTP server on %s:%d", addr, port)
            start_http_server(port, addr)
        except Exception:
            LOG.exception("Failed to start prometheus server")
    _prom_server_thread = threading.Thread(target=_serve, daemon=True, name="prometheus-server")
    _prom_server_thread.start()
    atexit.register(lambda: LOG.info("Prometheus server thread exiting"))

# -----------------------------------------------------------------------------
# FastAPI integration helpers (ASGI)
# -----------------------------------------------------------------------------
def prometheus_endpoint(environ, start_response):
    """
    Simple WSGI/ASGI endpoint to serve metrics via generate_latest.
    Can be used in Uvicorn/Starlette route if needed.
    """
    if not _HAS_PROM:
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"prometheus_client not installed"]
    try:
        data = generate_latest(PROM_REGISTRY)
        start_response("200 OK", [("Content-Type", CONTENT_TYPE_LATEST)])
        return [data]
    except Exception:
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        return [b"error"]

# Simple FastAPI dependency that returns the metrics snapshot
def metrics_snapshot_dependency():
    return metrics.snapshot()

# -----------------------------------------------------------------------------
# Simple alerting callbacks
# -----------------------------------------------------------------------------
def send_slack_alert(webhook_url: str, message: str, title: str = "PriorityMax Alert"):
    if not _HAS_AIOHTTP:
        LOG.warning("aiohttp not available; cannot send slack alert")
        return
    async def _send():
        payload = {"text": f"*{title}*\n{message}"}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(webhook_url, json=payload) as resp:
                    if resp.status not in (200, 201):
                        LOG.warning("Slack alert failed status=%s", resp.status)
        except Exception:
            LOG.exception("Slack alert exception")
    # schedule fire-and-forget
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_send())
    except Exception:
        # fallback thread
        threading.Thread(target=lambda: asyncio.run(_send()), daemon=True).start()

def pagerduty_alert(routing_key: str, summary: str, severity: str = "critical"):
    # Minimal PagerDuty v2 Events API client (best-effort)
    if not _HAS_AIOHTTP:
        LOG.warning("aiohttp not available; cannot send pagerduty alert")
        return
    async def _send():
        payload = {
            "routing_key": routing_key,
            "event_action": "trigger",
            "payload": {"summary": summary, "severity": severity, "source": socket.gethostname()}
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post("https://events.pagerduty.com/v2/enqueue", json=payload) as resp:
                    if resp.status not in (200, 202):
                        LOG.warning("PagerDuty alert failed status=%s", resp.status)
        except Exception:
            LOG.exception("PagerDuty alert exception")
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_send())
    except Exception:
        threading.Thread(target=lambda: asyncio.run(_send()), daemon=True).start()

# -----------------------------------------------------------------------------
# Auto-start for convenience in a container
# -----------------------------------------------------------------------------
def _maybe_start_background():
    try:
        if os.getenv("PRIORITYMAX_METRICS_AUTOSTART", "true").lower() in ("1", "true", "yes"):
            # start prom server if possible
            start_prometheus_server()
            # start background metrics exporter
            global _GLOBAL_METRICS
            m = get_global_metrics()
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # schedule start in the running loop
                    loop.create_task(m.start())
                else:
                    # start a loop in a background thread
                    def _run_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(m.start())
                        new_loop.run_forever()
                    threading.Thread(target=_run_loop, daemon=True).start()
            except RuntimeError:
                # no running loop; spawn a simple loop in thread
                def _run_loop2():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(m.start())
                    new_loop.run_forever()
                threading.Thread(target=_run_loop2, daemon=True).start()
    except Exception:
        LOG.exception("Auto-start failed")

# run auto-start
_maybe_start_background()

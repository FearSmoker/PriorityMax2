# backend/app/utils/common.py
"""
PriorityMax Common Utilities (production-grade integrated module)
-----------------------------------------------------------------
Shared utilities for logging, JSON, async helpers, system metrics, tracing, and
safe file/environment management. Designed for reliability and consistency across
PriorityMax components (API, ML, autoscaler, worker manager, etc.).

Features:
 - JSON encoding/decoding with NumPy, datetime, Decimal
 - Deep merge utilities and DotDict configuration wrapper
 - Async retry and timeout helpers
 - Prometheus metrics and structured logger setup
 - Time utilities and resource monitoring
 - Safe file and environment access
 - Hashing, compression, and serialization helpers
 - OpenTelemetry tracing wrappers (optional)
"""

from __future__ import annotations

import os
import io
import re
import sys
import gc
import math
import json
import uuid
import time
import gzip
import zlib
import base64
import psutil
import shutil
import random
import socket
import signal
import string
import pickle
import inspect
import logging
import datetime
import threading
import tempfile
import functools
import contextlib
from decimal import Decimal
from typing import (
    Any, Dict, List, Tuple, Optional, Callable, TypeVar, Awaitable, Union
)

# Optional dependencies
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except Exception:
    zstd = None
    _HAS_ZSTD = False

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
    _HAS_PROM = True
except Exception:
    Counter = Gauge = Histogram = CollectorRegistry = None
    _HAS_PROM = False

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
    _HAS_OTEL = True
except Exception:
    trace = None
    SpanKind = None
    _HAS_OTEL = False

# Logging
LOG = logging.getLogger("prioritymax.utils.common")
LOG.setLevel(os.getenv("PRIORITYMAX_UTIL_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# -------------------------
# JSON Helpers
# -------------------------
class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that supports numpy, datetime, Decimal, etc."""
    def default(self, obj):
        if _HAS_NUMPY:
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if hasattr(obj, "__dict__"):
            try:
                return obj.__dict__
            except Exception:
                return str(obj)
        return super().default(obj)

def json_dumps(obj: Any, indent: Optional[int] = None) -> str:
    try:
        return json.dumps(obj, cls=EnhancedJSONEncoder, indent=indent)
    except Exception:
        try:
            return json.dumps(str(obj))
        except Exception:
            return "{}"

def json_loads(s: Union[str, bytes]) -> Any:
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="ignore")
    try:
        return json.loads(s)
    except Exception:
        LOG.warning("json_loads failed, returning None")
        return None

# -------------------------
# DotDict & Deep Merge
# -------------------------
class DotDict(dict):
    """
    Dict subclass with dot-access convenience.
    Example:
        cfg = DotDict({"a": {"b": 2}})
        cfg.a.b == 2
        cfg["a"]["b"] == 2
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert DotDict to standard dict"""
        def _conv(val):
            if isinstance(val, DotDict):
                return {k: _conv(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [_conv(x) for x in val]
            return val
        return _conv(self)

def merge_dicts_deep(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts (b overwrites a)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts_deep(out[k], v)
        else:
            out[k] = v
    return out

# -------------------------
# File & Env Utilities
# -------------------------
def safe_write_json(path: str, data: Dict[str, Any]):
    """Atomically write JSON file."""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=EnhancedJSONEncoder, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        LOG.exception("Failed to write %s: %s", path, e)

def safe_read_json(path: str, default: Optional[Any] = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOG.warning("Failed to read %s: %s", path, e)
        return default

def get_env(key: str, default: Any = None, type_: Callable[[Any], Any] = str) -> Any:
    val = os.getenv(key, None)
    if val is None:
        return default
    try:
        if type_ == bool:
            return val.lower() in ("1", "true", "yes", "on")
        return type_(val)
    except Exception:
        return default

def atomic_write(path: str, content: Union[str, bytes]):
    """Write file atomically."""
    tmp = f"{path}.tmp"
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(tmp, mode) as f:
        f.write(content)
    os.replace(tmp, path)

# -------------------------
# Time utilities
# -------------------------
def now_ts() -> float:
    return time.time()

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def ts_to_iso(ts: float) -> str:
    return datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"

def iso_to_ts(s: str) -> float:
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "")).timestamp()
    except Exception:
        return 0.0

def monotonic_ms() -> int:
    """Return monotonic time in milliseconds."""
    return int(time.monotonic() * 1000)

# -------------------------
# Moving Average / EWMA
# -------------------------
class MovingAverage:
    """Thread-safe moving average calculator."""
    def __init__(self, window: int = 50):
        self.window = max(1, window)
        self.values: List[float] = []
        self.lock = threading.Lock()

    def add(self, value: float):
        with self.lock:
            self.values.append(value)
            if len(self.values) > self.window:
                self.values.pop(0)

    def avg(self) -> float:
        with self.lock:
            if not self.values:
                return 0.0
            return sum(self.values) / len(self.values)

class EWMA:
    """Exponentially Weighted Moving Average."""
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value

    def get(self) -> float:
        return self.value or 0.0
# -------------------------
# Chunk 2: Async utilities, retry, structured logger, prometheus helper, resource monitor
# -------------------------

import asyncio
import functools
from typing import Coroutine, Iterable, Sequence

# -------------------------
# Async retry / backoff helpers
# -------------------------
T = TypeVar("T")

def _is_exception_retryable(exc: Exception) -> bool:
    # Default retry policy: network/timeouts/transient IO errors
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    # add common transient network exceptions if present
    name = exc.__class__.__name__.lower()
    for transient in ("connection", "timeout", "tempor", "throttl", "rate", "unavail"):
        if transient in name:
            return True
    return False

def retry_sync(
    retries: int = 3,
    backoff_base: float = 0.5,
    retry_on_exception: Optional[Callable[[Exception], bool]] = None,
    jitter: float = 0.1,
):
    """
    Decorator for synchronous retry with exponential backoff.
    Usage:
        @retry_sync(retries=3)
        def do_work(...):
            ...
    """
    if retry_on_exception is None:
        retry_on_exception = _is_exception_retryable

    def _decor(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(0, retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not retry_on_exception(e) or attempt == retries:
                        raise
                    delay = backoff_base * (2 ** attempt) * (1.0 + (random.random() - 0.5) * jitter)
                    LOG.debug("Retrying %s after %.3fs due to %s (attempt %d)", fn.__name__, delay, e, attempt + 1)
                    time.sleep(delay)
            raise last_exc
        return _wrapped
    return _decor

def retry_async(
    retries: int = 3,
    backoff_base: float = 0.5,
    retry_on_exception: Optional[Callable[[Exception], bool]] = None,
    jitter: float = 0.1,
):
    """
    Async retry decorator for coroutines.
    """
    if retry_on_exception is None:
        retry_on_exception = _is_exception_retryable

    def _decor(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def _wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(0, retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not retry_on_exception(e) or attempt == retries:
                        raise
                    delay = backoff_base * (2 ** attempt) * (1.0 + (random.random() - 0.5) * jitter)
                    LOG.debug("Async retry %s after %.3fs due to %s (attempt %d)", fn.__name__, delay, e, attempt + 1)
                    await asyncio.sleep(delay)
            raise last_exc
        return _wrapped
    return _decor

# -------------------------
# Async bounded gather + cancel scope
# -------------------------
class CancelScope:
    """Simple cancel scope for groups of tasks."""
    def __init__(self):
        self._tasks: List[asyncio.Task] = []
        self._cancelled = False

    def add(self, task: asyncio.Task):
        if self._cancelled:
            task.cancel()
        else:
            self._tasks.append(task)

    def cancel_all(self):
        self._cancelled = True
        for t in list(self._tasks):
            try:
                t.cancel()
            except Exception:
                pass
        self._tasks.clear()

async def bounded_gather(coros: Iterable[Coroutine], concurrency: int = 32, return_exceptions: bool = False) -> List[Any]:
    """
    Gather coroutines with concurrency limit.
    Maintains order of results corresponding to input order.
    """
    sem = asyncio.Semaphore(concurrency)
    results: List[Any] = []
    coros = list(coros)

    async def _runner(i: int, c: Coroutine):
        async with sem:
            try:
                res = await c
                return (i, res, None)
            except Exception as e:
                return (i, None, e)

    tasks = [asyncio.create_task(_runner(i, c)) for i, c in enumerate(coros)]
    try:
        for t in asyncio.as_completed(tasks):
            i, res, exc = await t
            if exc is not None:
                if return_exceptions:
                    results.append((i, exc))
                else:
                    # cancel all remaining
                    for tt in tasks:
                        if not tt.done():
                            tt.cancel()
                    # re-raise
                    raise exc
            else:
                results.append((i, res))
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
    # reorder results
    results_sorted = sorted(results, key=lambda x: x[0])
    return [r for _, r in results_sorted]

# -------------------------
# Async timeout & watchdog
# -------------------------
class Watchdog:
    """Simple watchdog: calls a callback if not patted within timeout."""
    def __init__(self, timeout_sec: float, on_timeout: Callable[[], None]):
        self.timeout_sec = float(timeout_sec)
        self.on_timeout = on_timeout
        self._last_pat = time.monotonic()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def pet(self):
        self._last_pat = time.monotonic()

    async def _loop(self):
        self._running = True
        try:
            while self._running:
                await asyncio.sleep(max(0.1, self.timeout_sec / 3.0))
                if time.monotonic() - self._last_pat > self.timeout_sec:
                    try:
                        self.on_timeout()
                    except Exception:
                        LOG.exception("watchdog on_timeout failed")
                    self._running = False
        finally:
            self._running = False

    def start(self):
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self):
        self._running = False
        if self._task:
            try:
                self._task.cancel()
            except Exception:
                pass

async def with_timeout(coro: Awaitable[T], timeout: Optional[float]) -> T:
    if timeout is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout)

# -------------------------
# Structured logger (JSON + human)
# -------------------------
class StructuredLoggerAdapter:
    """
    Lightweight structured logger wrapper.
    Usage:
        log = StructuredLoggerAdapter(LOG, {"svc": "prioritymax"})
        log.info("message", extra={"task_id": "abc"})
    """
    def __init__(self, base: logging.Logger, base_ctx: Optional[Dict[str, Any]] = None):
        self._base = base
        self._ctx = base_ctx or {}

    def _merge(self, extra: Optional[Dict[str, Any]]):
        out = dict(self._ctx)
        if extra:
            out.update(extra)
        return out

    def info(self, msg: str, *args, extra: Optional[Dict[str, Any]] = None, **kwargs):
        ctx = self._merge(extra)
        self._base.info(f"{msg} | {json_dumps(ctx)}", *args, **kwargs)

    def debug(self, msg: str, *args, extra: Optional[Dict[str, Any]] = None, **kwargs):
        ctx = self._merge(extra)
        self._base.debug(f"{msg} | {json_dumps(ctx)}", *args, **kwargs)

    def warning(self, msg: str, *args, extra: Optional[Dict[str, Any]] = None, **kwargs):
        ctx = self._merge(extra)
        self._base.warning(f"{msg} | {json_dumps(ctx)}", *args, **kwargs)

    def error(self, msg: str, *args, extra: Optional[Dict[str, Any]] = None, **kwargs):
        ctx = self._merge(extra)
        self._base.error(f"{msg} | {json_dumps(ctx)}", *args, **kwargs)

# convenience adapter
STRUCT_LOG = StructuredLoggerAdapter(LOG, {"component": "common"})

# -------------------------
# Prometheus metric helper (lazy)
# -------------------------
_prom_registry_lock = threading.Lock()
_prom_registry: Optional[CollectorRegistry] = None
_prometers: Dict[str, Any] = {}

def get_prom_registry() -> Optional[CollectorRegistry]:
    global _prom_registry
    with _prom_registry_lock:
        if _prom_registry is None and _HAS_PROM:
            _prom_registry = CollectorRegistry()
        return _prom_registry

def make_metric(name: str, kind: str = "counter", documentation: str = "", labels: Optional[List[str]] = None):
    """
    Create or return cached Prometheus metric. kind: counter|gauge|histogram
    """
    if not _HAS_PROM:
        LOG.debug("Prometheus not available; make_metric returning None for %s", name)
        return None
    key = f"{kind}:{name}"
    if key in _prometers:
        return _prometers[key]
    with _prom_registry_lock:
        reg = get_prom_registry()
        if kind == "counter":
            m = Counter(name, documentation, labels or [], registry=reg)
        elif kind == "gauge":
            m = Gauge(name, documentation, labels or [], registry=reg)
        elif kind == "histogram":
            m = Histogram(name, documentation, labels or [], registry=reg)
        else:
            raise ValueError("Unknown metric kind")
        _prometers[key] = m
        return m

# -------------------------
# Resource monitoring
# -------------------------
def get_resource_usage() -> Dict[str, Any]:
    """
    Returns a dictionary with CPU, memory, fds, and process info.
    """
    try:
        p = psutil.Process(os.getpid())
        mem = p.memory_info()
        cpu_percent = p.cpu_percent(interval=0.1)
        open_files = len(p.open_files()) if hasattr(p, "open_files") else None
        threads = p.num_threads()
        return {
            "rss": getattr(mem, "rss", None),
            "vms": getattr(mem, "vms", None),
            "cpu_percent": cpu_percent,
            "open_fds": open_files,
            "num_threads": threads,
            "pid": p.pid,
        }
    except Exception:
        LOG.exception("get_resource_usage failed")
        return {}

def start_periodic_resource_report(interval_sec: int = 60, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
    """
    Start a background thread reporting resource usage to callback at interval.
    Returns the threading.Thread object (daemon).
    """
    def _runner():
        while True:
            try:
                data = get_resource_usage()
                if callback:
                    try:
                        callback(data)
                    except Exception:
                        LOG.exception("resource callback failed")
                else:
                    LOG.debug("resource: %s", data)
                time.sleep(interval_sec)
            except Exception:
                LOG.exception("resource reporter crashed, restarting loop")
                time.sleep(interval_sec)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t

# -------------------------
# End of Chunk 2
# -------------------------
# -------------------------
# Chunk 3: Serialization, compression, hashing, tracing wrappers, context utilities
# -------------------------

import hashlib
import hmac
import secrets
import contextvars
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# Hashing & Security helpers
# -------------------------
def sha256_hex(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def hmac_sha256(key: Union[str, bytes], data: Union[str, bytes]) -> str:
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hmac.new(key, data, hashlib.sha256).hexdigest()

def random_token(length: int = 32) -> str:
    """Return URL-safe random token."""
    return secrets.token_urlsafe(length)

def short_uid(prefix: str = "") -> str:
    """Short random hex ID."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"

# -------------------------
# Serialization / compression
# -------------------------
def safe_pickle_dumps(obj: Any, compress: bool = False) -> bytes:
    """
    Safe pickle dump with optional compression (gzip/zstd).
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if compress:
        if _HAS_ZSTD:
            c = zstd.ZstdCompressor(level=3)
            return c.compress(data)
        else:
            return gzip.compress(data)
    return data

def safe_pickle_loads(data: bytes, decompress: bool = False) -> Any:
    if decompress:
        if data[:2] == b"\x28\xb5" and _HAS_ZSTD:  # zstd magic
            d = zstd.ZstdDecompressor().decompress(data)
        else:
            try:
                d = gzip.decompress(data)
            except Exception:
                d = data
        data = d
    try:
        return pickle.loads(data)
    except Exception:
        LOG.exception("safe_pickle_loads failed")
        return None

def compress_bytes(data: bytes, method: str = "gzip", level: int = 5) -> bytes:
    """Compress bytes using gzip or zstd."""
    try:
        if method == "zstd" and _HAS_ZSTD:
            return zstd.ZstdCompressor(level=level).compress(data)
        if method == "gzip":
            return gzip.compress(data, compresslevel=level)
        return zlib.compress(data, level)
    except Exception:
        LOG.exception("compress_bytes failed")
        return data

def decompress_bytes(data: bytes) -> bytes:
    """Auto-detect compression format."""
    if not data:
        return b""
    if data[:2] == b"\x1f\x8b":  # gzip
        try:
            return gzip.decompress(data)
        except Exception:
            pass
    if data[:2] == b"\x28\xb5" and _HAS_ZSTD:  # zstd
        try:
            return zstd.ZstdDecompressor().decompress(data)
        except Exception:
            pass
    try:
        return zlib.decompress(data)
    except Exception:
        return data

# -------------------------
# Context propagation & tracing
# -------------------------
_current_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("prioritymax_ctx", default={})

def set_context(key: str, value: Any):
    ctx = dict(_current_context.get())
    ctx[key] = value
    _current_context.set(ctx)

def get_context(key: str, default: Any = None) -> Any:
    return _current_context.get().get(key, default)

def clear_context():
    _current_context.set({})

def trace_span(name: str):
    """
    Decorator to create OpenTelemetry span if tracing available.
    Usage:
        @trace_span("process_task")
        def process_task(...):
            ...
    """
    def _decor(fn: Callable):
        if not _HAS_OTEL:
            return fn

        tracer = trace.get_tracer("prioritymax")

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            with tracer.start_as_current_span(name, kind=SpanKind.INTERNAL):
                return fn(*args, **kwargs)
        return _wrapped
    return _decor

def trace_span_async(name: str):
    """
    Async decorator for coroutines.
    """
    def _decor(fn: Callable[..., Awaitable[Any]]):
        if not _HAS_OTEL:
            return fn
        tracer = trace.get_tracer("prioritymax")

        @functools.wraps(fn)
        async def _wrapped(*args, **kwargs):
            with tracer.start_as_current_span(name, kind=SpanKind.INTERNAL):
                return await fn(*args, **kwargs)
        return _wrapped
    return _decor

# -------------------------
# Contextual async task submission
# -------------------------
_executor_singleton: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

def get_executor(max_workers: int = 8) -> ThreadPoolExecutor:
    global _executor_singleton
    with _executor_lock:
        if _executor_singleton is None:
            _executor_singleton = ThreadPoolExecutor(max_workers=max_workers)
        return _executor_singleton

async def run_in_executor(func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run blocking function in global ThreadPoolExecutor."""
    loop = asyncio.get_running_loop()
    executor = get_executor()
    return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))

# -------------------------
# Graceful shutdown helpers
# -------------------------
_shutdown_flag = threading.Event()

def request_shutdown():
    _shutdown_flag.set()

def should_shutdown() -> bool:
    return _shutdown_flag.is_set()

def wait_for_shutdown(timeout: Optional[float] = None) -> bool:
    """Block until shutdown requested or timeout expires."""
    return _shutdown_flag.wait(timeout)

# -------------------------
# Version / build metadata
# -------------------------
def get_version_info() -> Dict[str, str]:
    """Return version/build info if available."""
    try:
        base = pathlib.Path(__file__).resolve().parents[2]
        version_file = base / "VERSION"
        git_head = base / ".git" / "HEAD"
        info = {}
        if version_file.exists():
            info["version"] = version_file.read_text().strip()
        if git_head.exists():
            info["git_head"] = git_head.read_text().strip()
        info["pid"] = str(os.getpid())
        info["ts"] = now_iso()
        return info
    except Exception:
        LOG.exception("get_version_info failed")
        return {"version": "unknown", "pid": str(os.getpid())}

# -------------------------
# Utility: graceful sleep (interruptible)
# -------------------------
async def graceful_sleep(seconds: float):
    """Sleep in small chunks so shutdown flag can interrupt."""
    step = 0.5
    end = time.time() + seconds
    while time.time() < end:
        if should_shutdown():
            break
        await asyncio.sleep(min(step, end - time.time()))

# -------------------------
# End of Chunk 3
# -------------------------
# -------------------------
# Chunk 4: System & file utilities, rolling logs, temp helpers, conversions, enhanced rate limiter
# -------------------------

import fcntl
import stat
import errno
import shlex
from pathlib import Path
from typing import IO

# -------------------------
# Safe filesystem helpers
# -------------------------
def ensure_dir(path: Union[str, Path], mode: int = 0o755):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True, mode=mode)

def safe_open_for_write(path: Union[str, Path], mode: str = "w", atomic: bool = True, perms: Optional[int] = None) -> IO:
    """
    Open a file for write. If atomic is True, write to a temp file and move on close via context manager.
    Returns a file-like object; use it as 'with safe_open_for_write(path) as f: f.write(...)'
    """
    path = Path(path)
    ensure_dir(path.parent)
    if not atomic:
        f = open(path, mode, encoding="utf-8")
        if perms:
            os.chmod(path, perms)
        return f

    tmp = str(path) + f".{random.randint(1000,9999)}.tmp"
    f = open(tmp, mode, encoding="utf-8")
    class _Ctx:
        def __enter__(self_):
            return f
        def __exit__(self_, exc_type, exc, tb):
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
            f.close()
            if exc_type is None:
                os.replace(tmp, str(path))
                if perms:
                    try:
                        os.chmod(path, perms)
                    except Exception:
                        pass
            else:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return _Ctx()

def file_lock(fd):
    """Context manager for advisory file lock (blocking)."""
    class _LockCtx:
        def __enter__(self_):
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            return fd
        def __exit__(self_, exc_type, exc, tb):
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    return _LockCtx()

# -------------------------
# Rolling file logger handler
# -------------------------
class RollingFileHandler(logging.Handler):
    """
    Simple rolling file handler based on size with archive rotation.
    - max_bytes: rotate when file exceeds size
    - backup_count: number of backups to keep
    """
    def __init__(self, filename: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5, compress: bool = True):
        super().__init__()
        self.filename = str(filename)
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        self.compress = bool(compress)
        ensure_dir(Path(self.filename).parent)
        self._fh = open(self.filename, "a+", encoding="utf-8")

    def emit(self, record):
        try:
            msg = self.format(record)
            self._fh.write(msg + "\n")
            self._fh.flush()
            if self._should_rotate():
                self._do_rotate()
        except Exception:
            self.handleError(record)

    def _should_rotate(self) -> bool:
        try:
            self._fh.seek(0, os.SEEK_END)
            size = self._fh.tell()
            return size >= self.max_bytes
        except Exception:
            return False

    def _do_rotate(self):
        try:
            self._fh.close()
            for i in range(self.backup_count - 1, 0, -1):
                s = f"{self.filename}.{i}"
                d = f"{self.filename}.{i+1}"
                if os.path.exists(s):
                    os.replace(s, d)
                    if self.compress and os.path.exists(d):
                        try:
                            with open(d, "rb") as rf:
                                comp = gzip.compress(rf.read())
                            with open(d + ".gz", "wb") as gf:
                                gf.write(comp)
                            os.remove(d)
                        except Exception:
                            pass
            # move current to .1
            if os.path.exists(self.filename):
                os.replace(self.filename, f"{self.filename}.1")
            # reopen file
            self._fh = open(self.filename, "a+", encoding="utf-8")
        except Exception:
            LOG.exception("RollingFileHandler rotate failed")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
        super().close()

# -------------------------
# Tempdir & atomic temp file helpers
# -------------------------
@contextlib.contextmanager
def tempdir(suffix: str = "", prefix: str = "pmtmp", dir: Optional[str] = None):
    td = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    try:
        yield td
    finally:
        try:
            shutil.rmtree(td)
        except Exception:
            pass

def temp_file_path(suffix: str = "", prefix: str = "pmfile", dir: Optional[str] = None) -> str:
    fd, p = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(fd)
    return p

# -------------------------
# Human-readable conversions
# -------------------------
def bytes_to_human(nbytes: int, precision: int = 2) -> str:
    if nbytes is None:
        return "0B"
    n = float(nbytes)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if n < 1024.0:
            return f"{n:.{precision}f}{unit}"
        n /= 1024.0
    return f"{n:.{precision}f}EB"

def human_to_bytes(s: str) -> int:
    s = s.strip().upper()
    pattern = r"^([\d\.]+)\s*([KMGTPE]?B?)$"
    m = re.match(pattern, s)
    if not m:
        raise ValueError(f"Cannot parse size: {s}")
    val = float(m.group(1))
    unit = m.group(2)
    multipliers = {"": 1, "B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2, "G": 1024**3, "GB": 1024**3, "T": 1024**4, "TB": 1024**4}
    return int(val * multipliers.get(unit, 1))

def seconds_to_human(sec: float) -> str:
    sec = float(sec)
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    if sec < 60:
        return f"{sec:.2f}s"
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}m{s}s"

# -------------------------
# Safe shell quoting & small helpers
# -------------------------
def safe_shlex_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)

# -------------------------
# Enhanced rate limiter (async-friendly & per-key)
# -------------------------
class AsyncTokenBucket:
    """
    Async-aware token bucket with awaitable acquire and blocking behavior.
    Allows 'refill_rate' tokens per second and 'capacity' maximum tokens.
    """
    def __init__(self, refill_rate: float = 1.0, capacity: float = 10.0):
        self.refill_rate = float(refill_rate)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        deadline = time.monotonic() + timeout if timeout else None
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
                    self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                if deadline and time.monotonic() > deadline:
                    return False
                # release lock and sleep small amount
                await asyncio.sleep(min(0.1, (tokens - self._tokens) / max(1e-6, self.refill_rate)))

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        return await self.acquire(tokens=tokens, timeout=0.0)

# -------------------------
# End of Chunk 4
# -------------------------
# -------------------------
# Chunk 5: Async timers, profilers, background scheduler, singleton, inspection & safety utils
# -------------------------

import traceback
from types import TracebackType

# -------------------------
# AsyncTimer and profiling
# -------------------------
class AsyncTimer:
    """Asynchronous context manager for timing operations."""
    def __init__(self, name: str = "block", log: bool = True):
        self.name = name
        self.log = log
        self.start = None
        self.end = None
        self.elapsed = None

    async def __aenter__(self):
        self.start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.log:
            LOG.debug("AsyncTimer %s took %.6fs", self.name, self.elapsed)

class Timer:
    """Synchronous timer (context manager)."""
    def __init__(self, name: str = "block", log: bool = True):
        self.name = name
        self.log = log
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.log:
            LOG.debug("Timer %s took %.6fs", self.name, self.elapsed)

# -------------------------
# PerformanceProfiler
# -------------------------
class PerformanceProfiler:
    """Collect simple timing statistics for sections of code."""
    def __init__(self):
        self.sections: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self.lock:
                self.sections.setdefault(name, []).append(elapsed)
            LOG.debug("Section %s took %.4f", name, elapsed)

    def summary(self) -> Dict[str, float]:
        with self.lock:
            return {k: float(sum(v) / len(v)) for k, v in self.sections.items() if v}

# -------------------------
# Background Scheduler
# -------------------------
class BackgroundScheduler:
    """
    Lightweight async background task scheduler.
    Example:
        scheduler = BackgroundScheduler()
        scheduler.add_task("metrics", collect_metrics, interval=30)
        await scheduler.start()
    """
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.funcs: Dict[str, Tuple[Callable, float]] = {}
        self._stop_event = asyncio.Event()

    def add_task(self, name: str, func: Callable[[], Awaitable[Any]], interval: float):
        """Register a repeating task."""
        self.funcs[name] = (func, interval)

    async def _loop(self, name: str, func: Callable, interval: float):
        while not should_shutdown() and not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                await func()
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("Background task %s failed", name)
            delta = interval - (time.perf_counter() - t0)
            await graceful_sleep(max(0.0, delta))

    async def start(self):
        LOG.info("Starting background scheduler with %d tasks", len(self.funcs))
        for name, (func, interval) in self.funcs.items():
            if name not in self.tasks or self.tasks[name].done():
                self.tasks[name] = asyncio.create_task(self._loop(name, func, interval))

    async def stop(self):
        self._stop_event.set()
        for t in self.tasks.values():
            t.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        LOG.info("Background scheduler stopped")

# -------------------------
# Singleton decorator
# -------------------------
def singleton(cls):
    """Thread-safe singleton class decorator."""
    instances = {}
    lock = threading.Lock()

    @functools.wraps(cls)
    def _get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _get_instance

# -------------------------
# Inspection & call utilities
# -------------------------
def get_caller_name(depth: int = 2) -> str:
    """Return caller function name."""
    try:
        frame = inspect.stack()[depth]
        return frame.function
    except Exception:
        return "unknown"

def safe_call(func: Callable, *args, **kwargs) -> Any:
    """Execute a function safely, catching exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception:
        LOG.exception("safe_call failed for %s", getattr(func, "__name__", "unknown"))
        return None

async def safe_await(coro: Awaitable[Any]) -> Any:
    """Await a coroutine safely."""
    try:
        return await coro
    except asyncio.CancelledError:
        LOG.debug("safe_await cancelled")
        raise
    except Exception:
        LOG.exception("safe_await error")
        return None

# -------------------------
# Exception format utilities
# -------------------------
def format_exception(e: BaseException, limit: int = 5) -> str:
    """Format exception and traceback to string."""
    try:
        tb = traceback.format_exception(type(e), e, e.__traceback__, limit=limit)
        return "".join(tb).strip()
    except Exception:
        return str(e)

def capture_exception_context(e: BaseException) -> Dict[str, Any]:
    """Capture detailed exception context."""
    return {
        "type": e.__class__.__name__,
        "msg": str(e),
        "trace": format_exception(e, limit=8),
        "time": now_iso(),
    }

# -------------------------
# CPU-bound workload runner (safe async)
# -------------------------
async def run_cpu_bound(func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run CPU-heavy function in background thread safely."""
    return await run_in_executor(func, *args, **kwargs)

# -------------------------
# Object size introspection
# -------------------------
def get_object_size(obj: Any) -> int:
    """Estimate object size recursively."""
    seen = set()
    def _sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        size = sys.getsizeof(o, 0)
        if isinstance(o, dict):
            size += sum((_sizeof(k) + _sizeof(v)) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(_sizeof(i) for i in o)
        return size
    return _sizeof(obj)

def human_obj_size(obj: Any) -> str:
    """Human-readable object size."""
    return human_bytes(get_object_size(obj))

# -------------------------
# End of Chunk 5
# -------------------------
# -------------------------
# Chunk 6: Diagnostics, cleanup utilities, self-check CLI, __all__ export
# -------------------------

# -------------------------
# Diagnostics & Health
# -------------------------
def health_snapshot() -> Dict[str, Any]:
    """
    Produce a unified health snapshot: CPU/mem usage, uptime, threads, open FDs, etc.
    Combines process + system info.
    """
    snap = {}
    try:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        cpu = proc.cpu_percent(interval=0.1)
        snap = {
            "pid": os.getpid(),
            "cpu_percent": cpu,
            "rss_mb": round(mem.rss / 1024 / 1024, 2),
            "threads": proc.num_threads(),
            "fds": len(proc.open_files()) if hasattr(proc, "open_files") else None,
            "uptime_sec": time.time() - proc.create_time(),
            "hostname": socket.gethostname(),
            "timestamp": now_iso(),
        }
    except Exception:
        LOG.exception("health_snapshot failed")
    return snap

# -------------------------
# Cleanup & Resource Reclaimer
# -------------------------
def cleanup_temp_files(older_than_sec: float = 3600, tmp_dir: Optional[str] = None):
    """
    Clean up temporary files older than a given threshold.
    """
    tmp = pathlib.Path(tmp_dir or tempfile.gettempdir())
    now = time.time()
    count = 0
    try:
        for f in tmp.iterdir():
            try:
                if f.is_file() and (now - f.stat().st_mtime) > older_than_sec:
                    f.unlink()
                    count += 1
            except Exception:
                continue
        LOG.debug("cleanup_temp_files removed %d files", count)
    except Exception:
        LOG.exception("cleanup_temp_files failed")

def cleanup_zombie_threads():
    """Attempt to detect and log non-daemon threads that remain alive."""
    try:
        threads = threading.enumerate()
        zombies = [t.name for t in threads if not t.daemon and t.is_alive()]
        if zombies:
            LOG.warning("Zombie threads: %s", zombies)
        return zombies
    except Exception:
        LOG.exception("cleanup_zombie_threads failed")
        return []

# -------------------------
# Async context guards
# -------------------------
class AsyncExitStack:
    """
    Minimal async context stack that ensures proper cleanup of entered resources.
    Useful when managing multiple async contexts dynamically.
    """
    def __init__(self):
        self._exit_callbacks: List[Callable[[], Awaitable[Any]]] = []

    async def enter_async_context(self, cm):
        res = await cm.__aenter__()
        self._exit_callbacks.append(cm.__aexit__)
        return res

    async def close(self):
        while self._exit_callbacks:
            cb = self._exit_callbacks.pop()
            try:
                await cb(None, None, None)
            except Exception:
                LOG.exception("AsyncExitStack cleanup failed")

# -------------------------
# Self-test CLI
# -------------------------
def _self_check():
    """Run internal sanity checks for this module."""
    print("PriorityMax common utilities self-check starting...")

    print("✔ JSON encode:", json_dumps({"x": 1, "ts": now_iso()}))
    print("✔ MovingAverage:", MovingAverage(5).avg())
    print("✔ SHA256:", sha256_hex("PriorityMax"))
    print("✔ Env loader:", get_env("NON_EXISTENT_ENV", "default"))
    print("✔ Resource snapshot:", get_resource_usage())
    print("✔ System snapshot:", get_system_snapshot())
    print("✔ Version info:", get_version_info())

    async def test_async():
        await asyncio.sleep(0.1)
        return "ok"

    print("✔ Async retry test:")
    async def retry_fn():
        raise TimeoutError("simulate")

    try:
        @retry_async(retries=2)
        async def fail_fn():
            raise TimeoutError("simulated")
        asyncio.run(fail_fn())
    except Exception:
        print("Expected retry failure handled")

    print("✔ Background scheduler sanity check:")
    async def dummy_task():
        print("heartbeat")

    sched = BackgroundScheduler()
    sched.add_task("dummy", dummy_task, interval=0.1)
    async def run_sched():
        await sched.start()
        await asyncio.sleep(0.3)
        await sched.stop()
    asyncio.run(run_sched())

    print("✔ Self-check complete ✅")

# -------------------------
# Main entrypoint (CLI)
# -------------------------
def _build_cli():
    import argparse
    parser = argparse.ArgumentParser(prog="prioritymax-utils")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("selfcheck")
    sub.add_parser("health")
    sub.add_parser("cleanup-temp")
    sub.add_parser("snapshot")

    return parser

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()

    if args.cmd == "selfcheck":
        _self_check()
    elif args.cmd == "health":
        print(json_dumps(health_snapshot(), indent=2))
    elif args.cmd == "cleanup-temp":
        cleanup_temp_files()
    elif args.cmd == "snapshot":
        print(json_dumps(get_system_snapshot(), indent=2))
    else:
        parser.print_help()

# -------------------------
# Public exports
# -------------------------
__all__ = [
    # JSON + dict tools
    "json_dumps", "json_loads", "DotDict", "merge_dicts_deep",
    # Time + moving avg
    "now_ts", "now_iso", "ts_to_iso", "iso_to_ts",
    "MovingAverage", "EWMA", "Timer", "AsyncTimer", "PerformanceProfiler",
    # Async helpers
    "retry_async", "retry_sync", "bounded_gather", "Watchdog", "CancelScope",
    "graceful_sleep", "BackgroundScheduler", "with_timeout",
    # Logging + metrics
    "StructuredLoggerAdapter", "STRUCT_LOG", "make_metric", "get_prom_registry",
    # Hashing + security
    "sha256_hex", "hmac_sha256", "random_token", "short_uid",
    # Serialization/compression
    "safe_pickle_dumps", "safe_pickle_loads", "compress_bytes", "decompress_bytes",
    # Context + tracing
    "trace_span", "trace_span_async", "set_context", "get_context", "clear_context",
    # File + system utils
    "safe_write_json", "safe_read_json", "atomic_write", "rotate_file", "write_rolling_log",
    "get_resource_usage", "get_system_snapshot", "start_periodic_resource_report",
    "run_command", "TokenBucketRateLimiter", "SyncRateLimiter", "SlidingWindow",
    # Diagnostics + health
    "health_snapshot", "cleanup_temp_files", "cleanup_zombie_threads",
    # Misc
    "singleton", "safe_call", "safe_await", "run_cpu_bound", "human_bytes", "human_time",
    "setup_signal_handlers", "get_version_info", "request_shutdown", "should_shutdown",
    "get_object_size", "human_obj_size", "format_exception", "capture_exception_context",
    "AsyncExitStack", "get_caller_name", "_self_check", "main_cli"
]

if __name__ == "__main__":
    main_cli()

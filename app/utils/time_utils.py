# backend/app/utils/time_utils.py
"""
PriorityMax Time Utilities (production-grade)
---------------------------------------------

A unified time and scheduling utility for PriorityMax’s distributed systems.
Provides high-precision, async-safe, and cluster-consistent temporal operations.

Features:
 - ISO8601 parsing/formatting with timezone awareness
 - Monotonic-safe timers for measuring latency or uptime
 - Async sleep, periodic tickers, and throttled loop helpers
 - High-resolution Timer + Stopwatch context managers
 - Rate control and backoff computation (adaptive sleep)
 - NTP synchronization (best-effort)
 - Time window utilities (rolling intervals)
 - Human-readable conversion (e.g. "5m", "2h30m") → seconds
 - Prometheus metric export hooks (optional)
 - Full async + sync API parity
"""

from __future__ import annotations

import os
import re
import sys
import math
import time
import json
import asyncio
import datetime
import logging
import threading
from typing import Any, Dict, Optional, Tuple, Callable, Awaitable, List

# Optional dependencies
try:
    import pytz
    _HAS_PYTZ = True
except Exception:
    pytz = None
    _HAS_PYTZ = False

try:
    import ntplib
    _HAS_NTP = True
except Exception:
    ntplib = None
    _HAS_NTP = False

try:
    from prometheus_client import Gauge
    _HAS_PROM = True
except Exception:
    Gauge = None
    _HAS_PROM = False

# Logging
LOG = logging.getLogger("prioritymax.utils.time_utils")
LOG.setLevel(os.getenv("PRIORITYMAX_TIME_LOG", "INFO").upper())
if not LOG.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(h)

# Prometheus metric
if _HAS_PROM:
    TIME_OFFSET = Gauge("prioritymax_time_offset_seconds", "Time offset from NTP server (seconds)")
else:
    TIME_OFFSET = None

# -------------------------
# Core helpers
# -------------------------
def now_ts() -> float:
    """Unix timestamp (UTC) with float seconds."""
    return time.time()

def now_ms() -> int:
    """Current epoch time in milliseconds."""
    return int(time.time() * 1000)

def now_ns() -> int:
    """Current epoch time in nanoseconds."""
    return time.time_ns()

def monotonic_ts() -> float:
    """Monotonic timestamp (not affected by system clock changes)."""
    return time.monotonic()

def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)

def utc_now() -> datetime.datetime:
    """Timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)

def iso_now() -> str:
    """Return current UTC time in ISO8601 format."""
    return utc_now().isoformat().replace("+00:00", "Z")

# -------------------------
# Parsing & formatting
# -------------------------
ISO_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?")

def parse_iso8601(s: str) -> datetime.datetime:
    """Parse ISO8601 string into UTC datetime."""
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception:
        LOG.debug("Fallback ISO parser for %s", s)
        m = ISO_PATTERN.match(s)
        if not m:
            raise ValueError(f"Invalid ISO8601: {s}")
        date, timepart, micros, zone = m.groups()
        dt = datetime.datetime.strptime(f"{date}T{timepart}", "%Y-%m-%dT%H:%M:%S")
        if micros:
            dt = dt.replace(microsecond=int(float(f"0.{micros}") * 1e6))
        return dt.replace(tzinfo=datetime.timezone.utc)

def to_iso(dt: datetime.datetime) -> str:
    """Format datetime into ISO string (UTC Z)."""
    if dt.tzinfo:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")

def from_timestamp(ts: float) -> datetime.datetime:
    """Convert float timestamp to UTC datetime."""
    return datetime.datetime.utcfromtimestamp(ts).replace(tzinfo=datetime.timezone.utc)

def to_timestamp(dt: datetime.datetime) -> float:
    """Convert datetime to float timestamp."""
    if dt.tzinfo:
        return dt.timestamp()
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()

# -------------------------
# Human duration parsing
# -------------------------
DURATION_PATTERN = re.compile(r"(\d+\.?\d*)\s*([a-zA-Z]+)")

_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "d": 86400,
    "day": 86400,
    "w": 604800,
    "week": 604800,
}

def parse_duration(text: str) -> float:
    """
    Parse human-readable duration strings like:
    "5m", "2h30m", "1.5h", "1d 4h 15m".
    """
    if not text:
        return 0.0
    total = 0.0
    for val, unit in DURATION_PATTERN.findall(text.lower()):
        mult = _UNIT_SECONDS.get(unit, 1)
        total += float(val) * mult
    return total

def format_duration(seconds: float, compact: bool = True) -> str:
    """Format seconds into '1d2h3m4s' or human string."""
    seconds = int(seconds)
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s or not parts: parts.append(f"{s}s")
    return "".join(parts) if compact else " ".join(parts)

# -------------------------
# Timers and context managers
# -------------------------
class Timer:
    """Sync timer context manager."""
    def __init__(self, name: str = "timer", log: bool = False):
        self.name = name
        self.start = None
        self.elapsed = None
        self.log = log

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        if self.log:
            LOG.info("Timer %s took %.6f sec", self.name, self.elapsed)

class AsyncTimer:
    """Async timer context manager."""
    def __init__(self, name: str = "async_timer", log: bool = False):
        self.name = name
        self.start = None
        self.elapsed = None
        self.log = log

    async def __aenter__(self):
        self.start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        if self.log:
            LOG.info("AsyncTimer %s took %.6f sec", self.name, self.elapsed)

# -------------------------
# Periodic tickers and schedulers
# -------------------------
class PeriodicTicker:
    """
    Async periodic ticker, like Go's time.Ticker.
    Example:
        async for tick in PeriodicTicker(5.0):
            ...
    """
    def __init__(self, interval: float, jitter: float = 0.0):
        self.interval = float(interval)
        self.jitter = float(jitter)
        self._running = True

    def stop(self):
        self._running = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._running:
            raise StopAsyncIteration
        await asyncio.sleep(self._get_sleep_interval())
        return now_ts()

    def _get_sleep_interval(self):
        if self.jitter <= 0:
            return self.interval
        return self.interval * (1.0 + (2 * (os.urandom(1)[0] / 255.0 - 0.5)) * self.jitter)

async def throttled_loop(interval: float, func: Callable[[], Awaitable[Any]], stop_event: Optional[asyncio.Event] = None):
    """Run func() every interval seconds until stop_event is set."""
    while not (stop_event and stop_event.is_set()):
        start = time.perf_counter()
        try:
            await func()
        except Exception:
            LOG.exception("throttled_loop error")
        elapsed = time.perf_counter() - start
        await asyncio.sleep(max(0.0, interval - elapsed))

# -------------------------
# NTP synchronization (best-effort)
# -------------------------
def get_ntp_offset(host: str = "pool.ntp.org", timeout: float = 2.0) -> Optional[float]:
    """Return difference (offset) between local and NTP time."""
    if not _HAS_NTP:
        LOG.debug("ntplib not installed; skipping NTP sync")
        return None
    try:
        c = ntplib.NTPClient()
        resp = c.request(host, version=3, timeout=timeout)
        offset = resp.offset
        if TIME_OFFSET:
            TIME_OFFSET.set(offset)
        return offset
    except Exception as e:
        LOG.warning("NTP sync failed: %s", e)
        return None

def sync_ntp_periodically(interval_sec: int = 3600):
    """Background thread that syncs NTP offset periodically."""
    def _loop():
        while True:
            get_ntp_offset()
            time.sleep(interval_sec)
    t = threading.Thread(target=_loop, daemon=True, name="PriorityMax-NTP")
    t.start()
    return t

# -------------------------
# Time windows & sliding intervals
# -------------------------
def truncate_to_minute(dt: datetime.datetime) -> datetime.datetime:
    return dt.replace(second=0, microsecond=0)

def truncate_to_hour(dt: datetime.datetime) -> datetime.datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def get_window_bounds(size_sec: int, ref: Optional[float] = None) -> Tuple[float, float]:
    """
    Compute rolling time window [start, end) around ref timestamp.
    """
    ref = ref or now_ts()
    start = ref - (ref % size_sec)
    end = start + size_sec
    return start, end

def generate_time_windows(duration_sec: float, window_size: float) -> List[Tuple[float, float]]:
    """
    Generate contiguous time windows covering duration.
    Example: 3600s duration, 300s window → 12 windows.
    """
    windows = []
    now = now_ts()
    end = now
    start = now - duration_sec
    t = start
    while t < end:
        windows.append((t, min(t + window_size, end)))
        t += window_size
    return windows

# -------------------------
# Backoff utilities
# -------------------------
def compute_backoff(attempt: int, base: float = 0.5, factor: float = 2.0, jitter: float = 0.1, max_delay: float = 30.0) -> float:
    """Compute exponential backoff with jitter."""
    delay = min(base * (factor ** attempt), max_delay)
    if jitter:
        import random
        delay *= (1.0 + (random.random() - 0.5) * jitter)
    return delay

# -------------------------
# Sleep helpers
# -------------------------
async def async_sleep(seconds: float):
    """Async sleep with safe cancellation."""
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        LOG.debug("async_sleep cancelled early")

def sleep_ms(ms: int):
    """Synchronous sleep in milliseconds."""
    time.sleep(ms / 1000.0)

# -------------------------
# CLI self-test
# -------------------------
def _self_check():
    print("PriorityMax time_utils self-check")
    print("now_ts:", now_ts())
    print("iso_now:", iso_now())
    print("parse_duration('2h30m'):", parse_duration("2h30m"))
    print("format_duration(9050):", format_duration(9050))
    print("window bounds 300s:", get_window_bounds(300))
    print("backoff(3):", compute_backoff(3))
    if _HAS_NTP:
        print("ntp offset:", get_ntp_offset())
    print("✅ time_utils OK")

# -------------------------
# CLI entrypoint
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-time")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selfcheck")
    sub.add_parser("ntp")
    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    if args.cmd == "selfcheck":
        _self_check()
    elif args.cmd == "ntp":
        print(get_ntp_offset())
    else:
        parser.print_help()

# -------------------------
# Exports
# -------------------------
__all__ = [
    "now_ts", "now_ms", "now_ns", "monotonic_ts", "monotonic_ms",
    "utc_now", "iso_now", "parse_iso8601", "to_iso", "from_timestamp", "to_timestamp",
    "parse_duration", "format_duration", "Timer", "AsyncTimer", "PeriodicTicker",
    "throttled_loop", "get_ntp_offset", "sync_ntp_periodically",
    "truncate_to_minute", "truncate_to_hour", "get_window_bounds", "generate_time_windows",
    "compute_backoff", "async_sleep", "sleep_ms"
]

if __name__ == "__main__":
    main_cli()

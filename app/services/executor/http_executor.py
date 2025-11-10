# backend/app/services/executor/http_executor.py
"""
PriorityMax HTTP Executor (production-minded)
---------------------------------------------

Provides robust, production-ready HTTP invocation utilities for PriorityMax:
  - Synchronous (requests) and asynchronous (aiohttp) invocation helpers
  - Retries with exponential backoff, jitter, and configurable status/code handling
  - Circuit breaker per-host/endpoint with half-open probe support
  - Rate limiting (token bucket) per-domain and global
  - Timeout configuration, connection pooling, keepalive
  - OAuth2 client credentials flow helper and JWT bearer injection
  - Pluggable request/response hooks (transformers, validators)
  - Async and sync bulk-parallel request helpers with concurrency control
  - Prometheus metrics (optional) and logging instrumentation
  - Best-effort OpenTelemetry tracing integration (if available)
  - CLI for quick testing and health checks
  - Clear error types and structured results for easy integration
  - Designed to be safe for use from FastAPI background tasks or in worker processes

Design choices & notes:
 - This module avoids executing arbitrary code; it's dedicated to making HTTP calls safely.
 - Optional dependencies are imported best-effort and fallbacks are used when absent.
 - Circuit breaker state is kept in-memory; for a clustered multi-instance product, move state to Redis.
 - Rate limiter implements a token-bucket with asyncio-friendly awaitable `acquire()` method.
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import logging
import asyncio
import threading
import functools
import random
import typing
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Optional libraries
_HAS_REQS = False
_HAS_AIOHTTP = False
_HAS_PROM = False
_HAS_OTEL = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAS_REQS = True
except Exception:
    requests = None
    HTTPAdapter = None
    Retry = None
    _HAS_REQS = False

try:
    import aiohttp
    from aiohttp import ClientTimeout, TCPConnector
    _HAS_AIOHTTP = True
except Exception:
    aiohttp = None
    ClientTimeout = None
    TCPConnector = None
    _HAS_AIOHTTP = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    _HAS_PROM = True
except Exception:
    Counter = Histogram = Gauge = start_http_server = None
    _HAS_PROM = False

# OpenTelemetry (best-effort)
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    _HAS_OTEL = True
except Exception:
    trace = None
    RequestsInstrumentor = None
    AioHttpClientInstrumentor = None
    _HAS_OTEL = False

# Logging
LOG = logging.getLogger("prioritymax.executor.http")
LOG.setLevel(os.getenv("PRIORITYMAX_HTTP_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults from env
DEFAULT_TIMEOUT = float(os.getenv("PRIORITYMAX_HTTP_TIMEOUT", "10.0"))
DEFAULT_CONNECT_TIMEOUT = float(os.getenv("PRIORITYMAX_HTTP_CONNECT_TIMEOUT", "3.0"))
DEFAULT_MAX_RETRIES = int(os.getenv("PRIORITYMAX_HTTP_RETRIES", "3"))
DEFAULT_BACKOFF_BASE = float(os.getenv("PRIORITYMAX_HTTP_BACKOFF_BASE", "0.5"))
DEFAULT_JITTER = float(os.getenv("PRIORITYMAX_HTTP_JITTER", "0.25"))
DEFAULT_MAX_CONCURRENCY = int(os.getenv("PRIORITYMAX_HTTP_MAX_CONCURRENCY", "64"))
DEFAULT_RATE_LIMIT = float(os.getenv("PRIORITYMAX_HTTP_RATE_LIMIT", "200.0"))  # tokens per minute global
PROMETHEUS_PORT = int(os.getenv("PRIORITYMAX_PROMETHEUS_PORT", "9003"))

# Prometheus metrics
if _HAS_PROM:
    HTTP_REQ_COUNT = Counter("prioritymax_http_requests_total", "Total HTTP requests", ["method", "host", "status"])
    HTTP_REQ_LATENCY = Histogram("prioritymax_http_request_seconds", "HTTP request latency seconds")
    HTTP_REQ_ERRORS = Counter("prioritymax_http_request_errors_total", "HTTP request errors", ["host", "error_type"])
    HTTP_CIRCUIT_TRIPS = Counter("prioritymax_http_circuit_trips_total", "Circuit breaker trips", ["host"])
    try:
        if os.getenv("PRIORITYMAX_PROMETHEUS_START", "false").lower() in ("1", "true", "yes"):
            start_http_server(PROMETHEUS_PORT)
            LOG.info("Prometheus metrics served at port %d", PROMETHEUS_PORT)
    except Exception:
        LOG.exception("Failed to start Prometheus server")
else:
    HTTP_REQ_COUNT = HTTP_REQ_LATENCY = HTTP_REQ_ERRORS = HTTP_CIRCUIT_TRIPS = None

# Type aliases
Headers = Dict[str, str]
Params = Dict[str, Union[str, int, float]]
JSONType = Any
HookFn = Callable[[Dict[str, Any]], Dict[str, Any]]  # transform request/response metadata

# Exceptions
class HTTPExecutorError(Exception):
    pass

class CircuitOpenError(HTTPExecutorError):
    pass

# -------------------------
# Utilities
# -------------------------
def _uid(prefix: str = "") -> str:
    return (prefix + "-" if prefix else "") + uuid.uuid4().hex[:8]

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# -------------------------
# Token bucket rate limiter
# -------------------------
class TokenBucket:
    """
    Simple token bucket rate limiter.
    rate: tokens per minute (float)
    capacity: max tokens (burst)
    """

    def __init__(self, rate_per_minute: float = DEFAULT_RATE_LIMIT, capacity: Optional[float] = None):
        self.rate = float(rate_per_minute) / 60.0  # convert to tokens per second
        self.capacity = capacity or max(1.0, self.rate * 10.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _add_tokens(self):
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        with self._lock:
            self._add_tokens()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Async-friendly acquisition. Returns True if acquired, False on timeout.
        """
        deadline = time.monotonic() + (timeout if timeout else 0) if timeout else None
        while True:
            with self._lock:
                self._add_tokens()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            if deadline and time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.01)

# -------------------------
# Circuit breaker (in-memory)
# -------------------------
@dataclass
class CircuitState:
    host: str
    failure_count: int = 0
    last_failure_ts: float = 0.0
    opened_until: float = 0.0  # timestamp until which circuit is open
    success_count_since_open: int = 0

class CircuitBreaker:
    """
    Basic circuit breaker:
      - trip_threshold: failures to trip
      - reset_timeout: seconds circuit stays open before half-open probing
      - half_open_successes: required consecutive successes to close circuit
    """

    def __init__(self, trip_threshold: int = 5, reset_timeout: float = 30.0, half_open_successes: int = 3):
        self.trip_threshold = trip_threshold
        self.reset_timeout = reset_timeout
        self.half_open_successes = half_open_successes
        self._states: Dict[str, CircuitState] = {}
        self._lock = threading.Lock()

    def _get_state(self, host: str) -> CircuitState:
        with self._lock:
            s = self._states.get(host)
            if not s:
                s = CircuitState(host=host)
                self._states[host] = s
            return s

    def record_success(self, host: str):
        s = self._get_state(host)
        with self._lock:
            now = time.time()
            s.failure_count = 0
            s.success_count_since_open += 1
            if s.opened_until and now > s.opened_until:
                # half-open window: require successes to close
                if s.success_count_since_open >= self.half_open_successes:
                    s.opened_until = 0
                    s.success_count_since_open = 0

    def record_failure(self, host: str):
        s = self._get_state(host)
        with self._lock:
            now = time.time()
            s.failure_count += 1
            s.last_failure_ts = now
            if s.failure_count >= self.trip_threshold:
                s.opened_until = now + self.reset_timeout
                s.success_count_since_open = 0
                if HTTP_CIRCUIT_TRIPS:
                    try:
                        HTTP_CIRCUIT_TRIPS.labels(host=host).inc()
                    except Exception:
                        pass

    def is_open(self, host: str) -> bool:
        s = self._get_state(host)
        now = time.time()
        if s.opened_until and now < s.opened_until:
            return True
        return False

    def time_to_recover(self, host: str) -> float:
        s = self._get_state(host)
        now = time.time()
        if s.opened_until and now < s.opened_until:
            return s.opened_until - now
        return 0.0

# -------------------------
# HTTP Executor config dataclass
# -------------------------
@dataclass
class HTTPRequestSpec:
    method: str = "GET"
    url: str = ""
    headers: Optional[Headers] = None
    params: Optional[Params] = None
    json: Optional[JSONType] = None
    data: Optional[Any] = None
    timeout: Optional[float] = None  # total timeout
    connect_timeout: Optional[float] = None
    allow_redirects: bool = True
    verify_ssl: bool = True
    auth: Optional[Tuple[str, str]] = None  # basic auth
    bearer_token: Optional[str] = None  # JWT / bearer insertion
    oauth2_client: Optional[Dict[str, Any]] = None  # client credentials config {token_url, client_id, client_secret, scope}
    retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    jitter: float = DEFAULT_JITTER
    allowed_statuses: Optional[List[int]] = None  # treat as success if status in list
    max_concurrency: Optional[int] = None  # override global concurrency
    host_rate_limit: Optional[float] = None  # tokens/min for this host
    metadata: Optional[Dict[str, Any]] = None  # opaque metadata
    hooks: Optional[Dict[str, HookFn]] = None  # {'pre': fn, 'post': fn}
    # internal:
    _timeout_tuple: Optional[Tuple[float, float]] = field(init=False, default=None)

    def __post_init__(self):
        connect = self.connect_timeout or DEFAULT_CONNECT_TIMEOUT
        tot = self.timeout or DEFAULT_TIMEOUT
        self._timeout_tuple = (connect, tot)

@dataclass
class HTTPResult:
    ok: bool
    status: Optional[int] = None
    text: Optional[str] = None
    json: Optional[Any] = None
    headers: Optional[Headers] = None
    elapsed: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# HTTPExecutor: core class
# -------------------------
class HTTPExecutor:
    def __init__(
        self,
        global_rate_limit_per_min: float = DEFAULT_RATE_LIMIT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        circuit: Optional[CircuitBreaker] = None,
        default_retry: int = DEFAULT_MAX_RETRIES,
        session_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.global_bucket = TokenBucket(rate_per_minute=global_rate_limit_per_min, capacity=global_rate_limit_per_min)
        self.host_buckets: Dict[str, TokenBucket] = {}
        self.max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(max_concurrency) if _HAS_AIOHTTP else None
        self.circuit = circuit or CircuitBreaker()
        self.default_retry = default_retry
        self.session_kwargs = session_kwargs or {}
        # sync requests session with connection pooling
        self._sync_session = None
        self._sync_session_lock = threading.Lock()
        # optional oauth2 token cache per client config (threadsafe)
        self._oauth_cache: Dict[str, Dict[str, Any]] = {}
        self._oauth_lock = threading.Lock()

        # instrument requests with OTel if available
        if _HAS_OTEL and RequestsInstrumentor:
            try:
                RequestsInstrumentor().instrument()
            except Exception:
                LOG.exception("Failed to instrument requests for OpenTelemetry")
        if _HAS_OTEL and AioHttpClientInstrumentor and _HAS_AIOHTTP:
            try:
                AioHttpClientInstrumentor().instrument()
            except Exception:
                LOG.exception("Failed to instrument aiohttp for OpenTelemetry")

    # -------------------------
    # Sync session initializer
    # -------------------------
    def _ensure_sync_session(self):
        with self._sync_session_lock:
            if self._sync_session is not None:
                return self._sync_session
            if not _HAS_REQS:
                LOG.warning("requests not available, sync calls will not function")
                return None
            s = requests.Session()
            # default adapter with retries; user can override session_kwargs
            retries = Retry(total=self.default_retry, backoff_factor=0.1, status_forcelist=(429, 500, 502, 503, 504))
            adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            # apply extra kwargs (e.g., cert, proxies)
            for k, v in self.session_kwargs.items():
                setattr(s, k, v)
            self._sync_session = s
            return s

    # -------------------------
    # OAuth2 client credentials flow helper
    # -------------------------
    def _get_oauth2_token(self, oauth_cfg: Dict[str, Any]) -> Optional[str]:
        """
        Simple cached client-credentials flow. oauth_cfg should contain:
          - token_url
          - client_id
          - client_secret
          - scope (optional)
        """
        if not _HAS_REQS:
            LOG.warning("requests not installed; cannot perform oauth2 flow")
            return None
        key = json.dumps(oauth_cfg, sort_keys=True)
        with self._oauth_lock:
            cached = self._oauth_cache.get(key)
            if cached and cached.get("expires_at", 0) > time.time() + 10:
                return cached.get("access_token")
        try:
            resp = requests.post(oauth_cfg["token_url"], data={
                "grant_type": "client_credentials",
                "client_id": oauth_cfg["client_id"],
                "client_secret": oauth_cfg["client_secret"],
                "scope": oauth_cfg.get("scope", "")
            }, timeout=(oauth_cfg.get("connect_timeout", DEFAULT_CONNECT_TIMEOUT), oauth_cfg.get("timeout", DEFAULT_TIMEOUT)))
            if resp.status_code == 200:
                blob = resp.json()
                token = blob.get("access_token")
                expires_in = int(blob.get("expires_in", 3600))
                with self._oauth_lock:
                    self._oauth_cache[key] = {"access_token": token, "expires_at": time.time() + expires_in - 30}
                return token
            else:
                LOG.warning("OAuth token fetch failed status=%s body=%s", resp.status_code, resp.text)
                return None
        except Exception:
            LOG.exception("OAuth token request failed")
            return None

    # -------------------------
    # Rate limiter helpers
    # -------------------------
    def _get_host_bucket(self, host: str, rate_per_min: Optional[float] = None) -> TokenBucket:
        if host not in self.host_buckets:
            if rate_per_min:
                self.host_buckets[host] = TokenBucket(rate_per_minute=rate_per_min)
            else:
                # default: proportional share of global
                self.host_buckets[host] = TokenBucket(rate_per_minute=max(1.0, self.global_bucket.rate * 60 * 0.5))
        return self.host_buckets[host]

    # -------------------------
    # Utility: build effective headers with auth
    # -------------------------
    def _build_headers(self, spec: HTTPRequestSpec) -> Headers:
        headers = dict(spec.headers or {})
        if spec.bearer_token:
            headers["Authorization"] = f"Bearer {spec.bearer_token}"
        elif spec.oauth2_client:
            token = self._get_oauth2_token(spec.oauth2_client)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    # -------------------------
    # Sync request implementation
    # -------------------------
    def request_sync(self, spec: HTTPRequestSpec) -> HTTPResult:
        """
        Perform a synchronous HTTP request with retries, circuit breaker, rate limiting.
        """
        if not _HAS_REQS:
            raise RuntimeError("requests library required for sync HTTPExecutor")

        # Basic parsing
        method = (spec.method or "GET").upper()
        url = spec.url
        host = typing.cast(str, (requests.utils.urlparse(url).netloc if _HAS_REQS else url))
        headers = self._build_headers(spec)
        allowed_statuses = set(spec.allowed_statuses or [])
        # Circuit check
        if self.circuit.is_open(host):
            ttl = self.circuit.time_to_recover(host)
            raise CircuitOpenError(f"Circuit open for host {host}, retry after {ttl:.1f}s")

        # Rate limit acquire
        host_bucket = self._get_host_bucket(host, rate_per_min=spec.host_rate_limit)
        if not host_bucket.try_acquire(1.0):
            # treat as transient throttle
            LOG.debug("Host rate limit exceeded for %s", host)
            return HTTPResult(ok=False, error="rate_limited", meta={"host": host})

        session = self._ensure_sync_session()
        if session is None:
            raise RuntimeError("Sync session unavailable")

        # Hook pre
        pre_hook = spec.hooks.get("pre") if spec.hooks else None
        if pre_hook:
            try:
                pre_hook({"method": method, "url": url, "spec": spec})
            except Exception:
                LOG.exception("pre-hook failed")

        last_err = None
        backoff_base = spec.backoff_base or DEFAULT_BACKOFF_BASE
        for attempt in range(0, (spec.retries or self.default_retry) + 1):
            t0 = time.perf_counter()
            try:
                timeout = spec._timeout_tuple or (DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT)
                resp = session.request(method, url, headers=headers, params=spec.params, json=spec.json, data=spec.data, timeout=timeout, allow_redirects=spec.allow_redirects, verify=spec.verify_ssl, auth=spec.auth)
                elapsed = time.perf_counter() - t0
                status = resp.status_code
                text = resp.text[:100000] if resp.text else ""
                json_body = None
                try:
                    json_body = resp.json()
                except Exception:
                    json_body = None
                # metrics
                if HTTP_REQ_COUNT:
                    try:
                        HTTP_REQ_COUNT.labels(method=method, host=host, status=str(status)).inc()
                        HTTP_REQ_LATENCY.observe(elapsed)
                    except Exception:
                        pass
                # success?
                if 200 <= status < 300 or status in allowed_statuses:
                    self.circuit.record_success(host)
                    # post hook
                    post_hook = spec.hooks.get("post") if spec.hooks else None
                    if post_hook:
                        try:
                            post_hook({"response": resp, "spec": spec})
                        except Exception:
                            LOG.exception("post-hook failed")
                    return HTTPResult(ok=True, status=status, text=text, json=json_body, headers=dict(resp.headers), elapsed=elapsed)
                # Retryable statuses: 429, 500-599
                if status == 429 or 500 <= status < 600:
                    last_err = f"status_{status}"
                    self.circuit.record_failure(host)
                    # backoff and retry unless exhausted
                    if attempt < (spec.retries or self.default_retry):
                        delay = backoff_base * (2 ** attempt) * (1.0 + (spec.jitter or DEFAULT_JITTER) * (random.random() - 0.5))
                        LOG.debug("Retryable status %d on %s: attempt %d, sleeping %.2fs", status, url, attempt, delay)
                        time.sleep(max(0.01, delay))
                        continue
                    else:
                        if HTTP_REQ_ERRORS:
                            try:
                                HTTP_REQ_ERRORS.labels(host=host, error_type=str(status)).inc()
                            except Exception:
                                pass
                        return HTTPResult(ok=False, status=status, text=text, json=json_body, elapsed=elapsed, error=f"status_{status}")
                # non-retryable status
                return HTTPResult(ok=False, status=status, text=text, json=json_body, elapsed=elapsed, error=f"status_{status}")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                last_err = str(e)
                self.circuit.record_failure(host)
                if HTTP_REQ_ERRORS:
                    try:
                        HTTP_REQ_ERRORS.labels(host=host, error_type=type(e).__name__).inc()
                    except Exception:
                        pass
                if attempt < (spec.retries or self.default_retry):
                    delay = backoff_base * (2 ** attempt) * (1.0 + (spec.jitter or DEFAULT_JITTER) * (random.random() - 0.5))
                    LOG.debug("HTTP request error, retrying attempt %d after %.2fs: %s", attempt, delay, e)
                    time.sleep(max(0.01, delay))
                    continue
                LOG.exception("HTTP request_sync failed finally for %s", url)
                return HTTPResult(ok=False, error=last_err, elapsed=elapsed, meta={"host": host})

    # -------------------------
    # Async request implementation
    # -------------------------
    async def request_async(self, spec: HTTPRequestSpec) -> HTTPResult:
        """
        Perform an async HTTP request using aiohttp with retries, backoff, circuit breaker and rate-limiting.
        """
        if not _HAS_AIOHTTP:
            # fallback to threadpool sync wrapper
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self.request_sync, spec))

        method = (spec.method or "GET").upper()
        url = spec.url
        parsed = aiohttp.ClientSession().get(url) if False else None  # keep lint happy
        host = typing.cast(str, (aiohttp.helpers.urlsplit(url).netloc if hasattr(aiohttp, "helpers") else url))
        if self.circuit.is_open(host):
            ttl = self.circuit.time_to_recover(host)
            raise CircuitOpenError(f"Circuit open for host {host}, retry after {ttl:.1f}s")

        host_bucket = self._get_host_bucket(host, rate_per_min=spec.host_rate_limit)
        acquired = await host_bucket.acquire(1.0, timeout=0.5)
        if not acquired:
            return HTTPResult(ok=False, error="rate_limited", meta={"host": host})

        # concurrency semaphore
        if self._sem:
            await self._sem.acquire()

        # build headers and timeout
        headers = self._build_headers(spec)
        timeout = ClientTimeout(total=spec.timeout or DEFAULT_TIMEOUT, connect=spec.connect_timeout or DEFAULT_CONNECT_TIMEOUT)

        connector = TCPConnector(limit=spec.max_concurrency or self.max_concurrency, ssl=spec.verify_ssl)
        # pre hook
        pre_hook = spec.hooks.get("pre") if spec.hooks else None
        if pre_hook:
            try:
                pre_hook({"method": method, "url": url, "spec": spec})
            except Exception:
                LOG.exception("pre-hook failed async")

        last_err = None
        backoff_base = spec.backoff_base or DEFAULT_BACKOFF_BASE
        # create session per request to respect connector/timeout; could reuse if needed
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for attempt in range(0, (spec.retries or self.default_retry) + 1):
                t0 = time.perf_counter()
                try:
                    async with session.request(method, url, headers=headers, params=spec.params, json=spec.json, data=spec.data, allow_redirects=spec.allow_redirects) as resp:
                        elapsed = time.perf_counter() - t0
                        status = resp.status
                        text = await resp.text()
                        json_body = None
                        try:
                            json_body = await resp.json()
                        except Exception:
                            json_body = None
                        # metrics
                        if HTTP_REQ_COUNT:
                            try:
                                HTTP_REQ_COUNT.labels(method=method, host=host, status=str(status)).inc()
                                HTTP_REQ_LATENCY.observe(elapsed)
                            except Exception:
                                pass
                        if 200 <= status < 300 or (spec.allowed_statuses and status in spec.allowed_statuses):
                            self.circuit.record_success(host)
                            # post hook
                            post_hook = spec.hooks.get("post") if spec.hooks else None
                            if post_hook:
                                try:
                                    post_hook({"response": resp, "spec": spec})
                                except Exception:
                                    LOG.exception("post-hook failed async")
                            if self._sem:
                                self._sem.release()
                            return HTTPResult(ok=True, status=status, text=text, json=json_body, headers=dict(resp.headers), elapsed=elapsed)
                        if status == 429 or 500 <= status < 600:
                            self.circuit.record_failure(host)
                            last_err = f"status_{status}"
                            if attempt < (spec.retries or self.default_retry):
                                delay = backoff_base * (2 ** attempt) * (1.0 + (spec.jitter or DEFAULT_JITTER) * (random.random() - 0.5))
                                await asyncio.sleep(max(0.01, delay))
                                continue
                            else:
                                if HTTP_REQ_ERRORS:
                                    try:
                                        HTTP_REQ_ERRORS.labels(host=host, error_type=str(status)).inc()
                                    except Exception:
                                        pass
                                if self._sem:
                                    self._sem.release()
                                return HTTPResult(ok=False, status=status, text=text, json=json_body, elapsed=elapsed, error=f"status_{status}")
                        # non-retryable
                        if self._sem:
                            self._sem.release()
                        return HTTPResult(ok=False, status=status, text=text, json=json_body, elapsed=elapsed, error=f"status_{status}")
                except asyncio.TimeoutError:
                    elapsed = time.perf_counter() - t0
                    last_err = "timeout"
                    self.circuit.record_failure(host)
                    if attempt < (spec.retries or self.default_retry):
                        delay = backoff_base * (2 ** attempt) * (1.0 + (spec.jitter or DEFAULT_JITTER) * (random.random() - 0.5))
                        await asyncio.sleep(max(0.01, delay))
                        continue
                    if self._sem:
                        self._sem.release()
                    return HTTPResult(ok=False, error="timeout", elapsed=elapsed)
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    last_err = str(e)
                    self.circuit.record_failure(host)
                    if attempt < (spec.retries or self.default_retry):
                        delay = backoff_base * (2 ** attempt) * (1.0 + (spec.jitter or DEFAULT_JITTER) * (random.random() - 0.5))
                        await asyncio.sleep(max(0.01, delay))
                        continue
                    LOG.exception("request_async final failure for %s", url)
                    if self._sem:
                        self._sem.release()
                    return HTTPResult(ok=False, error=last_err, elapsed=elapsed)
        # fallback return
        if self._sem:
            self._sem.release()
        return HTTPResult(ok=False, error=last_err)

    # -------------------------
    # Parallel bulk helpers
    # -------------------------
    async def bulk_request_async(self, specs: List[HTTPRequestSpec], concurrency: Optional[int] = None) -> List[HTTPResult]:
        """
        Perform a large list of HTTPRequests asynchronously with bounded concurrency.
        Returns results in same order.
        """
        if not _HAS_AIOHTTP:
            # fallback to running in threadpool sequentially (inefficient)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: [self.request_sync(s) for s in specs])

        sem = asyncio.Semaphore(concurrency or min(self.max_concurrency, len(specs)))
        results: List[Optional[HTTPResult]] = [None] * len(specs)

        async def _worker(i: int, s: HTTPRequestSpec):
            async with sem:
                try:
                    res = await self.request_async(s)
                except Exception as e:
                    LOG.exception("bulk_request worker failed")
                    res = HTTPResult(ok=False, error=str(e))
                results[i] = res

        await asyncio.gather(*[ _worker(i, s) for i, s in enumerate(specs) ])
        return typing.cast(List[HTTPResult], results)

    def bulk_request_sync(self, specs: List[HTTPRequestSpec], concurrency: Optional[int] = None) -> List[HTTPResult]:
        """
        Synchronous parallel bulk using ThreadPoolExecutor.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out: List[Optional[HTTPResult]] = [None] * len(specs)
        with ThreadPoolExecutor(max_workers=concurrency or min(self.max_concurrency, len(specs))) as pool:
            futures = { pool.submit(self.request_sync, specs[i]): i for i in range(len(specs)) }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    out[i] = fut.result()
                except Exception as e:
                    LOG.exception("bulk_request_sync item failed")
                    out[i] = HTTPResult(ok=False, error=str(e))
        return typing.cast(List[HTTPResult], out)

    # -------------------------
    # Health check helpers
    # -------------------------
    def ping(self, url: str, timeout: Optional[float] = 2.0) -> bool:
        try:
            spec = HTTPRequestSpec(method="GET", url=url, timeout=timeout, connect_timeout=min(0.5, timeout))
            res = self.request_sync(spec)
            return res.ok
        except Exception:
            return False

# -------------------------
# CLI for testing
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-http-executor")
    sub = p.add_subparsers(dest="cmd")
    req = sub.add_parser("req")
    req.add_argument("--method", default="GET")
    req.add_argument("--url", required=True)
    req.add_argument("--json", default=None)
    req.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ping = sub.add_parser("ping")
    ping.add_argument("--url", required=True)
    bulk = sub.add_parser("bulk")
    bulk.add_argument("--file", required=True, help="JSON file with list of {method,url,...}")
    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    ex = HTTPExecutor()
    if args.cmd == "req":
        j = json.loads(args.json) if args.json else None
        spec = HTTPRequestSpec(method=args.method, url=args.url, json=j, timeout=args.timeout)
        res = ex.request_sync(spec)
        print(json.dumps(res.__dict__, default=str, indent=2))
    elif args.cmd == "ping":
        ok = ex.ping(args.url)
        print("ok:", ok)
    elif args.cmd == "bulk":
        data = json.load(open(args.file))
        specs = []
        for item in data:
            spec = HTTPRequestSpec(**item)
            specs.append(spec)
        import asyncio
        res = asyncio.run(ex.bulk_request_async(specs))
        print(json.dumps([r.__dict__ for r in res], default=str, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

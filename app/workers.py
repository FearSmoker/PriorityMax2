# backend/app/workers.py
# ----------------------
# Chunk 1/7 — core imports, config, tracing + websocket analytics hooks,
# advanced TaskRecord, ExecResult, plugin system, and base utilities.
# ----------------------

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import math
import asyncio
import logging
import inspect
import random
import signal
import traceback
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable, Tuple, Union
from datetime import datetime, timedelta, timezone

# -------------------------
# Optional / best-effort integrations
# -------------------------
# Prometheus metrics (optional)
try:
    from prometheus_client import Gauge, Counter, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    Gauge = Counter = Histogram = start_http_server = None
    _HAS_PROM = False

# OpenTelemetry tracing (optional)
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import TracerProvider
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    _HAS_OTEL = True
except Exception:
    otel_trace = None
    SDKTracerProvider = None
    BatchSpanProcessor = ConsoleSpanExporter = None
    _HAS_OTEL = False

# Websocket manager (app-level) — for real-time worker analytics broadcasting
try:
    from app.websocket_manager import ws_manager  # expected to expose .broadcast(channel, payload)
    _HAS_WSM = True
except Exception:
    ws_manager = None
    _HAS_WSM = False

# Queue clients: prefer RedisQueue; fallback to stubs
try:
    from app.queue.redis_queue import RedisQueue
    _HAS_REDIS_QUEUE = True
except Exception:
    RedisQueue = None
    _HAS_REDIS_QUEUE = False

# Executors (container/function/http)
try:
    from app.services.executor.container_executor import ContainerExecutor
    from app.services.executor.function_executor import FunctionExecutor
    from app.services.executor.http_executor import HTTPExecutor
    _HAS_EXECUTORS = True
except Exception:
    ContainerExecutor = None
    FunctionExecutor = None
    HTTPExecutor = None
    _HAS_EXECUTORS = False

# Storage (Mongo/Mongo-like) for auditing & heartbeat persistence
try:
    from app.storage import Storage
    _HAS_STORAGE = True
except Exception:
    Storage = None
    _HAS_STORAGE = False

# Autoscaler (for hints/feedback)
try:
    from app.autoscaler import PriorityMaxAutoscaler
    _HAS_AUTOSCALER = True
except Exception:
    PriorityMaxAutoscaler = None
    _HAS_AUTOSCALER = False

# Metrics singleton
try:
    from app.metrics import metrics as global_metrics
except Exception:
    global_metrics = None

# tracing helper: initialize a no-op tracer if otel isn't available
if _HAS_OTEL:
    try:
        tracer = otel_trace.get_tracer(__name__)
    except Exception:
        tracer = None
else:
    tracer = None

# Logging
LOG = logging.getLogger("prioritymax.workers")
LOG.setLevel(os.getenv("PRIORITYMAX_WORKER_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# -------------------------
# Config (env-driven)
# -------------------------
DEFAULT_QUEUE = os.getenv("PRIORITYMAX_DEFAULT_QUEUE", "prioritymax")
WORKER_POOL_NAME = os.getenv("PRIORITYMAX_WORKER_POOL", "default")
WORKER_CONCURRENCY = int(os.getenv("PRIORITYMAX_WORKER_CONCURRENCY", "4"))
WORKER_PREFETCH = int(os.getenv("PRIORITYMAX_WORKER_PREFETCH", "1"))
DLQ_SUFFIX = os.getenv("PRIORITYMAX_DLQ_SUFFIX", ".dlq")
DLQ_PROMOTE_BATCH = int(os.getenv("PRIORITYMAX_DLQ_PROMOTE_BATCH", "100"))
DLQ_MONITOR_INTERVAL = int(os.getenv("PRIORITYMAX_DLQ_MONITOR_INTERVAL", "300"))
DLQ_PROMOTE_MIN_BACKLOG = int(os.getenv("PRIORITYMAX_DLQ_PROMOTE_MIN_BACKLOG", "10"))
SELF_HEAL_DEFAULT_DRY_RUN = os.getenv("PRIORITYMAX_SELF_HEAL_DEFAULT_DRY_RUN", "true").lower() in ("1", "true", "yes")
WORKER_RESTART_BACKOFF = float(os.getenv("PRIORITYMAX_WORKER_RESTART_BACKOFF", "2.0"))
MAX_TASK_RETRIES = int(os.getenv("PRIORITYMAX_TASK_MAX_RETRIES", "5"))
WORKER_SHUTDOWN_TIMEOUT = float(os.getenv("PRIORITYMAX_WORKER_SHUTDOWN_TIMEOUT", "10.0"))
WORKER_METRICS_CHANNEL = os.getenv("PRIORITYMAX_WORKER_METRICS_CHANNEL", "workers.metrics")
AUTOSCALER_FEEDBACK_INTERVAL = int(os.getenv("PRIORITYMAX_AUTOSCALER_FEEDBACK_INTERVAL", "15"))  # seconds

# Default timeouts / intervals
TASK_POP_TIMEOUT = float(os.getenv("PRIORITYMAX_TASK_POP_TIMEOUT", "5.0"))
HEARTBEAT_INTERVAL = int(os.getenv("PRIORITYMAX_WORKER_HEARTBEAT", "10"))  # seconds

# -------------------------
# Prometheus metrics registration (optional)
# -------------------------
if _HAS_PROM:
    PROM_ACTIVE_WORKERS = Gauge("prioritymax_active_workers", "Active worker coroutines", ["pool"])
    PROM_TASKS_PROCESSED = Counter("prioritymax_tasks_processed_total", "Total tasks processed", ["pool", "worker"])
    PROM_TASKS_FAILED = Counter("prioritymax_tasks_failed_total", "Total tasks failed", ["pool", "worker", "reason"])
    PROM_TASK_LATENCY = Histogram("prioritymax_task_latency_seconds", "Task processing latency seconds", ["pool", "worker"],
                                  buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5))
    PROM_DLQ_PROMOTED = Counter("prioritymax_dlq_promoted_total", "Total DLQ messages promoted", ["pool"])
else:
    PROM_ACTIVE_WORKERS = PROM_TASKS_PROCESSED = PROM_TASKS_FAILED = PROM_TASK_LATENCY = PROM_DLQ_PROMOTED = None

# -------------------------
# Plugin / hook system for enterprise features
# -------------------------
# Hooks: on_task_start, on_task_success, on_task_fail, on_dlq_push, on_retry, on_worker_heartbeat
HookFn = Callable[..., Any]

class HookRegistry:
    """
    Lightweight plugin registry to register callables for lifecycle events.
    Plugins may be added at runtime.
    """
    def __init__(self):
        self._hooks: Dict[str, List[HookFn]] = {}

    def register(self, name: str, fn: HookFn):
        if name not in self._hooks:
            self._hooks[name] = []
        self._hooks[name].append(fn)
        LOG.debug("Hook registered: %s -> %s", name, getattr(fn, "__name__", str(fn)))

    async def trigger(self, name: str, *args, **kwargs):
        if name not in self._hooks:
            return
        for fn in list(self._hooks[name]):
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn(*args, **kwargs)
                else:
                    # run sync hooks in threadpool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
            except Exception:
                LOG.exception("Hook %s failed", name)

# global hook registry instance
HOOKS = HookRegistry()

# -------------------------
# Utilities
# -------------------------
def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def safe_json_loads(s: Union[str, bytes, None]) -> Optional[dict]:
    if not s:
        return None
    try:
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return json.loads(s)
    except Exception:
        LOG.debug("safe_json_loads failed", exc_info=True)
        return None

def _short_id(prefix: str = "t") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

# -------------------------
# Advanced Task dataclass (supports typed payloads, provenance, and serialization plugins)
# -------------------------
@dataclass
class TaskRecord:
    id: str
    type: str
    payload: Dict[str, Any]
    retries: int = 0
    created_at: str = field(default_factory=now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_raw(raw: Union[str, bytes, dict]) -> "TaskRecord":
        """
        Accepts JSON string, bytes, dict. Handles older and newer payload shapes.
        """
        if isinstance(raw, dict):
            data = raw
        else:
            data = safe_json_loads(raw)
            if data is None:
                raise ValueError("Invalid raw task payload")
        tid = data.get("id") or _short_id("task")
        ttype = data.get("type") or data.get("task_type") or "generic"
        payload = data.get("payload") or data.get("data") or {}
        retries = int(data.get("retries", 0))
        meta = data.get("meta", {})
        created_at = data.get("created_at", now_iso())
        return TaskRecord(id=tid, type=ttype, payload=payload, retries=retries, created_at=created_at, meta=meta)

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "type": self.type,
            "payload": self.payload,
            "retries": self.retries,
            "meta": self.meta,
            "created_at": self.created_at
        })

# -------------------------
# Execution result dataclass (extended)
# -------------------------
@dataclass
class ExecResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_sec: float = 0.0
    should_retry: bool = False
    retry_delay: float = 0.0
    drop_to_dlq: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# Tracing context manager helper (no-op when otel missing)
# -------------------------
class tracing_span:
    """
    Context manager for spans. Usage:
        with tracing_span("task.exec", attributes={"task_id": id}):
            ...
    Falls back to no-op if OpenTelemetry not installed.
    """
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self._span = None

    def __enter__(self):
        if _HAS_OTEL and otel_trace:
            try:
                self._span = tracer.start_span(self.name)
                for k, v in self.attributes.items():
                    try:
                        self._span.set_attribute(k, v)
                    except Exception:
                        pass
            except Exception:
                self._span = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._span:
            try:
                self._span.end()
            except Exception:
                pass

    async def __aenter__(self):
        if _HAS_OTEL and otel_trace:
            try:
                self._span = tracer.start_span(self.name)
                for k, v in self.attributes.items():
                    try:
                        self._span.set_attribute(k, v)
                    except Exception:
                        pass
            except Exception:
                self._span = None
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._span:
            try:
                self._span.end()
            except Exception:
                pass

# -------------------------
# Worker analytics helper (publishes to websocket_manager if available)
# -------------------------
async def publish_worker_metrics(channel: str, payload: Dict[str, Any]):
    """
    Best-effort broadcast of worker metrics via ws_manager.
    Non-blocking — logs failures.
    """
    try:
        if _HAS_WSM and ws_manager:
            # create a fire-and-forget task to avoid blocking caller
            asyncio.create_task(ws_manager.broadcast(channel, payload))
        else:
            # fallback: log at debug level for offline mode
            LOG.debug("Worker metrics (no ws_manager): %s", json.dumps(payload))
    except Exception:
        LOG.exception("publish_worker_metrics failed")

# -------------------------
# End of chunk 1
# -------------------------
# -------------------------
# Chunk 2/7 — ExecutorManager, Worker, WorkerSupervisor (part 1)
# -------------------------

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine

# Threadpool for running sync tasks & blocking I/O
_GLOBAL_THREADPOOL = ThreadPoolExecutor(max_workers=int(os.getenv("PRIORITYMAX_THREADPOOL", "16")))


# -------------------------
# ExecutorManager
# -------------------------
class ExecutorManager:
    """
    Responsible for invoking actual task execution via one of:
      - FunctionExecutor (in-process Python function)
      - ContainerExecutor (run task in container)
      - HTTPExecutor (call external REST endpoint)
    If none available, falls back to a simple "noop" executor that simulates success.
    Executors should implement:
       async def run(self, task_record: TaskRecord, timeout: float) -> ExecResult
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # instantiate available executors
        self.container = ContainerExecutor() if ContainerExecutor is not None else None
        self.function = FunctionExecutor() if FunctionExecutor is not None else None
        self.http = HTTPExecutor() if HTTPExecutor is not None else None

    async def run(self, task: TaskRecord, timeout: Optional[float] = None) -> ExecResult:
        """
        Choose executor based on task.meta / type / config.
        Priority:
          1) task.meta.executor == "function" -> FunctionExecutor
          2) task.meta.executor == "container" -> ContainerExecutor
          3) task.meta.executor == "http" -> HTTPExecutor
          4) fallback: function -> http -> container if available
        """
        preferred = (task.meta.get("executor") or "").lower()
        # helper to attempt run
        async def _try_exec(exec_obj):
            try:
                if isinstance(exec_obj, ContainerExecutor) or isinstance(exec_obj, HTTPExecutor) or isinstance(exec_obj, FunctionExecutor):
                    # call their async run if available
                    if asyncio.iscoroutinefunction(exec_obj.run):
                        return await exec_obj.run(task, timeout=timeout)
                    else:
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(_GLOBAL_THREADPOOL, lambda: exec_obj.run(task, timeout=timeout))
                else:
                    # unknown object: try calling run sync in threadpool
                    if hasattr(exec_obj, "run"):
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(_GLOBAL_THREADPOOL, lambda: exec_obj.run(task, timeout=timeout))
            except Exception:
                LOG.exception("Executor run failed for %s", getattr(exec_obj, "__class__", exec_obj))
                return ExecResult(success=False, error="executor_failed", should_retry=True, retry_delay=1.0)
            return ExecResult(success=False, error="no_executor", should_retry=False)

        order = []
        if preferred:
            if preferred == "function":
                order = [self.function, self.http, self.container]
            elif preferred == "container":
                order = [self.container, self.function, self.http]
            elif preferred == "http":
                order = [self.http, self.function, self.container]
        else:
            # choose based on task.type heuristics
            if task.type.startswith("http:"):
                order = [self.http, self.function, self.container]
            elif task.type.startswith("fn:") or task.type.startswith("function:"):
                order = [self.function, self.http, self.container]
            else:
                order = [self.function, self.container, self.http]

        for exec_obj in order:
            if exec_obj is None:
                continue
            res = await _try_exec(exec_obj)
            if res and (res.success or res.should_retry or res.drop_to_dlq):
                return res
            # if exec returned a neutral response, try next
            if res and res.error == "no_executor":
                continue
        # Final fallback: simulate simple function processing (no-op success)
        try:
            # small simulated work
            await asyncio.sleep(0.01)
            return ExecResult(success=True, output={"message": "simulated-exec"}, duration_sec=0.01)
        except Exception:
            return ExecResult(success=False, error="fallback_failed", should_retry=True, retry_delay=1.0)


# -------------------------
# Worker coroutine
# -------------------------
class Worker:
    """
    Single worker coroutine that:
      - Pops tasks from RedisQueue (or configured queue)
      - Executes them via ExecutorManager
      - Handles retry logic, DLQ, metrics, tracing, hooks, and storage auditing
    """

    def __init__(self,
                 name: str,
                 queue_name: str = DEFAULT_QUEUE,
                 queue_client: Optional[RedisQueue] = None,
                 storage: Optional[Storage] = None,
                 executor_manager: Optional[ExecutorManager] = None,
                 pool: str = WORKER_POOL_NAME,
                 prefetch: int = WORKER_PREFETCH,
                 shutdown_event: Optional[asyncio.Event] = None):
        self.name = name
        self.queue_name = queue_name
        self.queue = queue_client or (RedisQueue() if _HAS_REDIS_QUEUE else None)
        self.storage = storage or (Storage() if _HAS_STORAGE else None)
        self.executor = executor_manager or ExecutorManager()
        self.pool = pool
        self.prefetch = prefetch
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = shutdown_event or asyncio.Event()
        self._last_heartbeat = time.time()
        self._backoff = WORKER_RESTART_BACKOFF
        # worker-level stats
        self.processed = 0
        self.failed = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self, graceful: bool = True):
        self._running = False
        self._shutdown_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=WORKER_SHUTDOWN_TIMEOUT)
            except asyncio.TimeoutError:
                LOG.warning("Worker %s did not shutdown in time; cancelling", self.name)
                self._task.cancel()
            except Exception:
                LOG.exception("Error waiting worker stop")
        # final heartbeat
        await self._report_heartbeat(terminated=True)

    async def _report_heartbeat(self, terminated: bool = False):
        try:
            self._last_heartbeat = time.time()
            if self.storage:
                try:
                    await self.storage.upsert_worker_heartbeat(self.pool, self.name, {"ts": self._last_heartbeat, "terminated": terminated})
                except Exception:
                    LOG.debug("storage upsert heartbeat failed")
            # publish metrics via websocket manager
            payload = {"worker": self.name, "pool": self.pool, "ts": self._last_heartbeat, "processed": self.processed, "failed": self.failed, "terminated": terminated}
            asyncio.create_task(publish_worker_metrics(WORKER_METRICS_CHANNEL, payload))
            if _HAS_PROM and PROM_ACTIVE_WORKERS:
                try:
                    PROM_ACTIVE_WORKERS.labels(pool=self.pool).set(1)
                except Exception:
                    pass
        except Exception:
            LOG.exception("report_heartbeat failed")

    async def _push_to_dlq(self, task: TaskRecord, reason: str):
        try:
            if not self.queue:
                LOG.warning("No queue client available; cannot push to DLQ")
                return False
            dlq_name = f"{self.queue_name}{DLQ_SUFFIX}"
            await self.queue.push(dlq_name, task.to_json())
            if _HAS_PROM and PROM_DLQ_PROMOTED:
                PROM_DLQ_PROMOTED.labels(pool=self.pool).inc(1)
            await HOOKS.trigger("on_dlq_push", task, reason)
            if self.storage:
                try:
                    await self.storage.insert_dlq_entry(self.pool, self.name, task.id, reason, task.to_json())
                except Exception:
                    LOG.debug("storage insert dlq entry failed")
            return True
        except Exception:
            LOG.exception("push_to_dlq failed")
            return False

    async def _maybe_retry_or_dlq(self, task: TaskRecord, exec_res: ExecResult):
        """
        Decide whether to re-enqueue with backoff or to push to DLQ based on exec_res and retries.
        """
        try:
            if exec_res.success:
                return
            # if explicit drop_to_dlq
            if exec_res.drop_to_dlq:
                await self._push_to_dlq(task, exec_res.error or "drop_to_dlq")
                return
            # increment retry counter
            task.retries = int(task.retries) + 1
            if task.retries > MAX_TASK_RETRIES or not exec_res.should_retry:
                LOG.warning("Task %s exceeded retries (%d) or non-retriable -> DLQ", task.id, task.retries)
                await self._push_to_dlq(task, exec_res.error or "max_retries")
                return
            # schedule re-enqueue with delay
            delay = exec_res.retry_delay or (2 ** min(6, task.retries))
            retry_payload = {
                "id": task.id,
                "type": task.type,
                "payload": task.payload,
                "retries": task.retries,
                "meta": task.meta
            }
            # use a simple delayed re-enqueue by pushing into a delayed stream or sorted set
            # Best-effort: use queue.push_delayed if supported else sleep then push
            if hasattr(self.queue, "push_delayed"):
                try:
                    await self.queue.push_delayed(self.queue_name, json.dumps(retry_payload), delay)
                    await HOOKS.trigger("on_retry", task, delay)
                    return
                except Exception:
                    LOG.debug("push_delayed not available or failed; fallback to sleep requeue")
            # fallback: schedule coroutine to push after delay
            async def _delayed_reenqueue():
                await asyncio.sleep(delay)
                try:
                    await self.queue.push(self.queue_name, json.dumps(retry_payload))
                    await HOOKS.trigger("on_retry", task, delay)
                except Exception:
                    LOG.exception("Delayed reenqueue failed")
            asyncio.create_task(_delayed_reenqueue())
            LOG.info("Task %s scheduled for retry after %.2fs (retry=%d)", task.id, delay, task.retries)
        except Exception:
            LOG.exception("maybe_retry_or_dlq failed; moving to DLQ")
            await self._push_to_dlq(task, "retry_decision_error")

    async def _process_task(self, raw_msg: Union[str, bytes, dict]) -> None:
        """
        Convert raw message to TaskRecord and execute it.
        Handles metrics, tracing, hooks, storage auditing.
        """
        task = None
        t_start = time.perf_counter()
        with tracing_span("task.process", {"worker": self.name}):
            try:
                task = TaskRecord.from_raw(raw_msg)
            except Exception:
                LOG.exception("Invalid task payload; pushing to DLQ")
                # best-effort: push raw into dlq
                try:
                    if self.queue:
                        dlq = f"{self.queue_name}{DLQ_SUFFIX}"
                        await self.queue.push(dlq, json.dumps({"raw": str(raw_msg)}))
                except Exception:
                    LOG.exception("Failed to push invalid payload to dlq")
                return

            await HOOKS.trigger("on_task_start", task, self.name)

            # execute
            try:
                # allow per-task timeout via task.meta.timeout
                timeout = float(task.meta.get("timeout", TASK_POP_TIMEOUT))
            except Exception:
                timeout = TASK_POP_TIMEOUT

            exec_res: ExecResult
            try:
                # run with timeout wrapper to ensure long-running tasks don't hang worker
                coro = self.executor.run(task, timeout=timeout)
                exec_res = await asyncio.wait_for(coro, timeout=(timeout * 2 if timeout and timeout > 0 else None))
            except asyncio.TimeoutError:
                exec_res = ExecResult(success=False, error="timeout", should_retry=True, retry_delay=5.0)
            except Exception as e:
                LOG.exception("Executor raised exception")
                exec_res = ExecResult(success=False, error=str(e), should_retry=True, retry_delay=1.0)

            duration = time.perf_counter() - t_start
            exec_res.duration_sec = duration

            if exec_res.success:
                self.processed += 1
                # metrics
                if _HAS_PROM and PROM_TASKS_PROCESSED:
                    try:
                        PROM_TASKS_PROCESSED.labels(pool=self.pool, worker=self.name).inc(1)
                        PROM_TASK_LATENCY.labels(pool=self.pool, worker=self.name).observe(duration)
                    except Exception:
                        pass
                # hooks & storage audit
                await HOOKS.trigger("on_task_success", task, exec_res, self.name)
                if self.storage:
                    try:
                        await self.storage.insert_task_audit(task.id, self.name, True, exec_res.output, duration)
                    except Exception:
                        LOG.debug("storage insert task audit failed")
            else:
                self.failed += 1
                if _HAS_PROM and PROM_TASKS_FAILED:
                    try:
                        PROM_TASKS_FAILED.labels(pool=self.pool, worker=self.name, reason=(exec_res.error or "unknown")).inc(1)
                    except Exception:
                        pass
                await HOOKS.trigger("on_task_fail", task, exec_res, self.name)
                if self.storage:
                    try:
                        await self.storage.insert_task_audit(task.id, self.name, False, exec_res.error, duration)
                    except Exception:
                        LOG.debug("storage insert task audit failed")
                # decide retry vs DLQ
                await self._maybe_retry_or_dlq(task, exec_res)

    async def _pop_task(self) -> Optional[Union[str, bytes, dict]]:
        """
        Pop a message from queue: best-effort support for various queue APIs.
        Prefer: queue.pop(queue_name, timeout=)
                queue.consume / queue.pull with ack interface.
        Fallback: raise NotImplementedError if no queue client available.
        """
        if not self.queue:
            # no queue client -> nothing to do
            await asyncio.sleep(1.0)
            return None

        # Try popular async signature: queue.pop(name, timeout)
        try:
            if asyncio.iscoroutinefunction(self.queue.pop):
                item = await self.queue.pop(self.queue_name, timeout=TASK_POP_TIMEOUT)
                return item
            else:
                # sync pop: run in threadpool
                loop = asyncio.get_event_loop()
                item = await loop.run_in_executor(_GLOBAL_THREADPOOL, lambda: self.queue.pop(self.queue_name, TASK_POP_TIMEOUT))
                return item
        except AttributeError:
            # maybe queue.consume(callback) style, not supported here
            LOG.debug("Queue client lacks pop() method; sleeping")
            await asyncio.sleep(1.0)
            return None
        except Exception:
            LOG.exception("pop_task failed")
            await asyncio.sleep(1.0)
            return None

    async def _run_loop(self):
        """
        Main worker loop — pop and process tasks until shutdown.
        Implements backoff on repeated failures.
        """
        LOG.info("Worker %s starting (pool=%s)", self.name, self.pool)
        await self._report_heartbeat()
        consecutive_errors = 0
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    raw = await self._pop_task()
                    if raw is None:
                        # no task available; heartbeat & sleep
                        await self._report_heartbeat()
                        await asyncio.sleep(0.1)
                        continue
                    # process task
                    await self._process_task(raw)
                    consecutive_errors = 0
                    # heartbeat occasionally
                    if random.random() < 0.1:
                        await self._report_heartbeat()
                except asyncio.CancelledError:
                    break
                except Exception:
                    LOG.exception("Worker main loop exception")
                    consecutive_errors += 1
                    # slow down on repeated errors
                    await asyncio.sleep(min(30.0, self._backoff * consecutive_errors))
        finally:
            LOG.info("Worker %s stopped", self.name)
            # ensure metrics decrement
            if _HAS_PROM and PROM_ACTIVE_WORKERS:
                try:
                    PROM_ACTIVE_WORKERS.labels(pool=self.pool).dec()
                except Exception:
                    pass


# -------------------------
# WorkerSupervisor (start / stop logic) — part 1
# -------------------------
class WorkerSupervisor:
    """
    Supervises a pool of Worker coroutines:
      - starts N workers
      - monitors heartbeats
      - restarts crashed workers with backoff
      - coordinates graceful shutdown
      - exposes simple scale_up / scale_down methods for worker count
    """
    def __init__(self,
                 pool_name: str = WORKER_POOL_NAME,
                 queue_name: str = DEFAULT_QUEUE,
                 concurrency: int = WORKER_CONCURRENCY,
                 storage: Optional[Storage] = None):
        self.pool_name = pool_name
        self.queue_name = queue_name
        self.concurrency = max(1, concurrency)
        self.storage = storage or (Storage() if _HAS_STORAGE else None)
        self.executor_manager = ExecutorManager()
        self.workers: Dict[str, Worker] = {}
        self._supervising_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._lock = asyncio.Lock()
        self._restart_backoffs: Dict[str, float] = {}
        self._monitor_interval = float(os.getenv("PRIORITYMAX_SUPERVISOR_MON_INTERVAL", "5.0"))
        # track desired count (for external autoscaler)
        self._desired_count = self.concurrency

    async def start(self):
        LOG.info("Starting WorkerSupervisor pool=%s desired=%d", self.pool_name, self._desired_count)
        # spawn initial workers
        async with self._lock:
            for i in range(self._desired_count):
                name = f"{self.pool_name}-w{i+1}"
                if name in self.workers:
                    continue
                w = Worker(name=name, queue_name=self.queue_name, storage=self.storage, executor_manager=self.executor_manager, pool=self.pool_name)
                w.start()
                self.workers[name] = w
        # start monitor task
        if not self._supervising_task:
            self._supervising_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        LOG.info("Stopping WorkerSupervisor pool=%s", self.pool_name)
        self._shutdown.set()
        if self._supervising_task:
            self._supervising_task.cancel()
            try:
                await self._supervising_task
            except Exception:
                pass
        # stop workers
        async with self._lock:
            stops = [w.stop(graceful=True) for w in list(self.workers.values())]
            await asyncio.gather(*stops, return_exceptions=True)
            self.workers.clear()

    async def _monitor_loop(self):
        """
        Periodically check worker heartbeats and restart missing workers up to desired count.
        """
        LOG.info("WorkerSupervisor monitor started for pool=%s", self.pool_name)
        try:
            while not self._shutdown.is_set():
                try:
                    await self._ensure_desired_workers()
                except Exception:
                    LOG.exception("Supervisor ensure loop failed")
                await asyncio.sleep(self._monitor_interval)
        except asyncio.CancelledError:
            pass
        LOG.info("WorkerSupervisor monitor stopped for pool=%s", self.pool_name)

    async def _ensure_desired_workers(self):
        """
        Ensure number of running workers equals _desired_count. Restart failed workers with backoff.
        """
        async with self._lock:
            current = len(self.workers)
            if current < self._desired_count:
                # spawn new workers
                for i in range(current, self._desired_count):
                    name = f"{self.pool_name}-w{i+1}"
                    if name in self.workers:
                        continue
                    # restart backoff logic
                    backoff = self._restart_backoffs.get(name, WORKER_RESTART_BACKOFF)
                    LOG.info("Spawning worker %s with backoff %.2fs", name, backoff)
                    try:
                        w = Worker(name=name, queue_name=self.queue_name, storage=self.storage, executor_manager=self.executor_manager, pool=self.pool_name)
                        w.start()
                        self.workers[name] = w
                        # reset backoff on success
                        self._restart_backoffs[name] = WORKER_RESTART_BACKOFF
                    except Exception:
                        LOG.exception("Failed to spawn worker %s", name)
                        # increase backoff for next attempt
                        self._restart_backoffs[name] = min(300.0, self._restart_backoffs.get(name, WORKER_RESTART_BACKOFF) * 2.0)
            elif current > self._desired_count:
                # scale down: stop extra workers
                stop_n = current - self._desired_count
                LOG.info("Scaling down %d workers", stop_n)
                removed = 0
                for name, w in list(self.workers.items()):
                    if removed >= stop_n:
                        break
                    try:
                        await w.stop(graceful=True)
                    except Exception:
                        LOG.debug("Error stopping worker %s", name)
                    self.workers.pop(name, None)
                    removed += 1

    # scale API
    async def scale_to(self, count: int):
        async with self._lock:
            self._desired_count = max(0, int(count))
            LOG.info("Desired worker count set to %d", self._desired_count)
            # ensure loop will observe change quickly
            return self._desired_count

    async def scale_up(self, step: int = 1):
        return await self.scale_to(self._desired_count + step)

    async def scale_down(self, step: int = 1):
        return await self.scale_to(max(0, self._desired_count - step))

# -------------------------
# End of Chunk 2
# -------------------------
# -------------------------
# Chunk 3/7 — Supervisor (continued), DLQ monitor, self-heal, k8s HPA/RBAC helpers
# -------------------------

import yaml  # used for generating k8s manifests; best-effort import
from typing import Set

# -------------------------
# WorkerSupervisor (continued) — autoscaler & health hooks
# -------------------------
    # (this code continues inside the WorkerSupervisor class definition from Chunk 2)

    async def drain(self, timeout: float = 60.0):
        """
        Gracefully drain current workers: stop accepting new tasks and wait for in-flight tasks to finish.
        Uses shutdown flag on workers and waits up to `timeout` seconds.
        """
        LOG.info("Draining workers for pool=%s timeout=%.1fs", self.pool_name, timeout)
        # mark desired count to 0 to prevent restarts
        await self.scale_to(0)
        # stop accepting new tasks by setting shutdown flag on each worker and waiting
        async with self._lock:
            stops = []
            for w in list(self.workers.values()):
                stops.append(w.stop(graceful=True))
            # wait concurrently
            try:
                await asyncio.wait_for(asyncio.gather(*stops, return_exceptions=True), timeout=timeout)
            except asyncio.TimeoutError:
                LOG.warning("Drain timed out; cancelling remaining workers")
                for w in list(self.workers.values()):
                    try:
                        if w._task:
                            w._task.cancel()
                    except Exception:
                        pass
            # clear list
            self.workers.clear()
        LOG.info("Drain complete for pool=%s", self.pool_name)

    async def report_health(self) -> Dict[str, Any]:
        """
        Return a health summary for this supervisor: worker counts, last heartbeats, processed/failure counts.
        """
        summary = {"pool": self.pool_name, "desired": self._desired_count, "running": len(self.workers), "workers": {}}
        async with self._lock:
            for name, w in list(self.workers.items()):
                summary["workers"][name] = {
                    "last_heartbeat": getattr(w, "_last_heartbeat", None),
                    "processed": w.processed,
                    "failed": w.failed,
                    "running": w._running
                }
        return summary

    async def register_autoscaler(self, autoscaler: PriorityMaxAutoscaler):
        """
        Register autoscaler so Supervisor can respond to scale signals (and provide feedback)
        """
        if not autoscaler:
            return
        self._autoscaler = autoscaler
        # start feedback loop to update autoscaler with worker metrics
        asyncio.create_task(self._autoscaler_feedback_loop())

    async def _autoscaler_feedback_loop(self):
        """
        Periodically compute simple signals (utilization, backlog per worker) and feed into autoscaler.
        This provides RL model with on-policy feedback and enables autoscaler to learn real-world effects.
        """
        while True:
            try:
                # basic metrics: backlog from queue, current worker count
                try:
                    if _HAS_REDIS_QUEUE:
                        rq = RedisQueue()
                        qstats = await rq.get_queue_stats(self.queue_name)
                        backlog = int(qstats.get("backlog", 0))
                    else:
                        backlog = 0
                except Exception:
                    backlog = 0
                running = len(self.workers)
                # simple utilization: backlog per worker
                util = float(backlog) / max(1.0, float(running))
                # create a small observation dict for autoscaler
                obs = {"backlog": backlog, "workers": running, "util": util, "timestamp": time.time()}
                try:
                    if hasattr(self, "_autoscaler") and self._autoscaler:
                        # autoscaler may expose an `observe` or `ingest` method
                        if hasattr(self._autoscaler, "observe"):
                            try:
                                self._autoscaler.observe(obs)
                            except Exception:
                                # best-effort: call as coroutine if needed
                                if asyncio.iscoroutinefunction(self._autoscaler.observe):
                                    await self._autoscaler.observe(obs)
                        elif hasattr(self._autoscaler, "ingest"):
                            maybe = self._autoscaler.ingest
                            if asyncio.iscoroutinefunction(maybe):
                                await maybe(obs)
                            else:
                                maybe(obs)
                except Exception:
                    LOG.debug("Autoscaler feedback ingest failed")
                # feed every AUTOSCALER_FEEDBACK_INTERVAL seconds
                await asyncio.sleep(AUTOSCALER_FEEDBACK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("Autoscaler feedback loop error")
                await asyncio.sleep(AUTOSCALER_FEEDBACK_INTERVAL)

# -------------------------
# DLQ monitor & promoter
# -------------------------
class DLQMonitor:
    """
    Periodically checks DLQ sizes and optionally promotes items back to main queue
    when system load is low. Integrates with WorkerSupervisor & Autoscaler.
    """
    def __init__(self,
                 queue_name: str = DEFAULT_QUEUE,
                 queue_client: Optional[RedisQueue] = None,
                 supervisor: Optional[WorkerSupervisor] = None,
                 storage: Optional[Storage] = None,
                 interval: int = DLQ_MONITOR_INTERVAL,
                 backlog_threshold: int = DLQ_PROMOTE_MIN_BACKLOG):
        self.queue_name = queue_name
        self.queue = queue_client or (RedisQueue() if _HAS_REDIS_QUEUE else None)
        self.supervisor = supervisor
        self.storage = storage or (Storage() if _HAS_STORAGE else None)
        self.interval = interval
        self.backlog_threshold = backlog_threshold
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_promote = 0

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run_loop(self):
        LOG.info("DLQMonitor started for queue=%s", self.queue_name)
        while self._running:
            try:
                await self._check_once()
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("DLQMonitor encountered error")
            await asyncio.sleep(self.interval)
        LOG.info("DLQMonitor stopped for queue=%s", self.queue_name)

    async def _check_once(self):
        if not self.queue:
            return
        dlq_name = f"{self.queue_name}{DLQ_SUFFIX}"
        try:
            dlq_len = await self.queue.get_dlq_length(self.queue_name)
        except Exception:
            try:
                dlq_len = await self.queue.length(dlq_name)
            except Exception:
                dlq_len = 0
        try:
            qstats = await self.queue.get_queue_stats(self.queue_name)
            backlog = int(qstats.get("backlog", 0))
        except Exception:
            backlog = 0

        LOG.debug("DLQMonitor: dlq_len=%d backlog=%d", dlq_len, backlog)
        # store metrics
        if global_metrics:
            try:
                global_metrics.set_backlog(self.queue_name, backlog)
            except Exception:
                pass

        # If DLQ has messages but backlog is low and system idle, promote small batches automatically
        if dlq_len > 0 and backlog < self.backlog_threshold:
            # rate-limit promotions to avoid feedback loops
            now = time.time()
            if now - self._last_promote < max(60, self.interval):
                LOG.debug("DLQMonitor: recently promoted, skipping")
                return
            promote_limit = min(DLQ_PROMOTE_BATCH, dlq_len)
            # Optionally consult supervisor and autoscaler hints before promoting
            safe_to_promote = True
            if self.supervisor:
                # ensure workers are present and not zero
                if len(self.supervisor.workers) <= 0:
                    safe_to_promote = False
            if safe_to_promote:
                promoted = await self._promote_batch(promote_limit)
                self._last_promote = time.time()
                LOG.info("DLQMonitor promoted %d messages from %s", promoted, dlq_name)
                if self.storage:
                    try:
                        await self.storage.insert_dlq_promotion({"queue": self.queue_name, "promoted": promoted, "ts": now_iso()})
                    except Exception:
                        LOG.debug("storage insert dlq promotion failed")

    async def _promote_batch(self, limit: int = 100) -> int:
        """
        Promote up to `limit` messages from DLQ back to main queue.
        Uses queue.promote_dlq if available; otherwise pop DLQ and push to main queue.
        """
        if not self.queue:
            return 0
        dlq_name = f"{self.queue_name}{DLQ_SUFFIX}"
        promoted = 0
        if hasattr(self.queue, "promote_dlq"):
            try:
                promoted = await self.queue.promote_dlq(self.queue_name, limit=limit)
                # metrics
                if global_metrics:
                    global_metrics.record_dlq_promoted(self.queue_name, count=promoted)
                return promoted
            except Exception:
                LOG.exception("promote_dlq failed")
        # fallback manual promote
        for _ in range(limit):
            try:
                item = await self.queue.pop(dlq_name, timeout=0.5)
                if not item:
                    break
                await self.queue.push(self.queue_name, item)
                promoted += 1
            except Exception:
                LOG.exception("manual DLQ promote failed")
                break
        if global_metrics:
            global_metrics.record_dlq_promoted(self.queue_name, count=promoted)
        return promoted

# -------------------------
# Self-Heal Controller (privileged minimal controller)
# -------------------------
class SelfHealController:
    """
    Minimal privileged controller that executes non-destructive repair actions.
    NOTE: destructive actions are dry-run by default (controlled by SELF_HEAL_DEFAULT_DRY_RUN).
    For production environment, prefer a Kubernetes operator or external privileged service.
    """
    def __init__(self, supervisor: WorkerSupervisor, queue_name: str = DEFAULT_QUEUE, dry_run: bool = SELF_HEAL_DEFAULT_DRY_RUN):
        self.supervisor = supervisor
        self.queue_name = queue_name
        self.dry_run = dry_run
        self._lock = asyncio.Lock()
        self._rate_limit_ts = 0

    async def attempt_self_heal(self):
        """
        Decide whether to trigger a heal action based on metrics and DLQ state.
        Actions may include promoting DLQ, restarting hung workers, or scaling.
        """
        async with self._lock:
            now = time.time()
            if now - self._rate_limit_ts < 60:
                # rate limit heals
                return False
            # simple heuristics: if DLQ backlog high and worker count >0 and system idle -> promote
            dlq_len = 0
            try:
                rq = RedisQueue() if _HAS_REDIS_QUEUE else None
                if rq:
                    dlq_len = await rq.get_dlq_length(self.queue_name)
            except Exception:
                pass
            # detect hung workers via heartbeat timestamps
            hung_workers: List[str] = []
            for name, w in self.supervisor.workers.items():
                if (time.time() - getattr(w, "_last_heartbeat", 0)) > (HEARTBEAT_INTERVAL * 5):
                    hung_workers.append(name)
            LOG.info("SelfHeal: dlq=%d hung_workers=%s", dlq_len, hung_workers)
            # choose action
            if dlq_len > DLQ_PROMOTE_MIN_BACKLOG and len(self.supervisor.workers) > 0:
                if self.dry_run:
                    LOG.info("SelfHeal (dry-run): would promote DLQ items")
                    self._rate_limit_ts = time.time()
                    return True
                else:
                    promoted = await DLQMonitor(queue_name=self.queue_name, queue_client=rq, supervisor=self.supervisor). _promote_batch(limit=DLQ_PROMOTE_BATCH)
                    LOG.info("SelfHeal: promoted %d", promoted)
                    self._rate_limit_ts = time.time()
                    return True
            if hung_workers:
                # attempt to restart hung workers
                for name in hung_workers:
                    if self.dry_run:
                        LOG.info("SelfHeal (dry-run): would restart worker %s", name)
                    else:
                        try:
                            # stop underlying worker task and remove; supervisor will restart if desired_count maintained
                            w = self.supervisor.workers.get(name)
                            if w:
                                await w.stop(graceful=False)
                                # remove to force restart
                                async with self.supervisor._lock:
                                    self.supervisor.workers.pop(name, None)
                        except Exception:
                            LOG.exception("SelfHeal restart worker failed")
                self._rate_limit_ts = time.time()
                return True
            return False

# -------------------------
# Kubernetes HPA & RBAC YAML helpers
# -------------------------
def generate_hpa_yaml(deployment_name: str,
                      namespace: str = os.getenv("K8S_NAMESPACE", "default"),
                      min_replicas: int = 1,
                      max_replicas: int = 10,
                      target_cpu_utilization_percentage: int = 70,
                      custom_metric_name: Optional[str] = None) -> str:
    """
    Produce an HPA YAML manifest string. If custom_metric_name provided, uses external metric template.
    """
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{deployment_name}-hpa", "namespace": namespace},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": deployment_name},
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
        }
    }
    if custom_metric_name:
        hpa["spec"]["metrics"] = [{
            "type": "External",
            "external": {
                "metric": {"name": custom_metric_name},
                "target": {"type": "AverageValue", "averageValue": "1"}
            }
        }]
    else:
        hpa["spec"]["metrics"] = [{
            "type": "Resource",
            "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": target_cpu_utilization_percentage}}
        }]
    return yaml.safe_dump(hpa)

def generate_rbac_yaml(service_account: str = "prioritymax-operator", namespace: str = os.getenv("K8S_NAMESPACE", "default")) -> str:
    """
    Minimal RBAC Role + RoleBinding + ServiceAccount for the autoscaler/operator.
    """
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": "prioritymax-operator-role", "namespace": namespace},
        "rules": [
            {"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["get", "list", "watch", "patch", "update"]},
            {"apiGroups": ["autoscaling"], "resources": ["horizontalpodautoscalers"], "verbs": ["get", "create", "patch", "update", "list"]},
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": [""], "resources": ["services"], "verbs": ["get", "list"]},
            {"apiGroups": ["metrics.k8s.io"], "resources": ["pods"], "verbs": ["get", "list"]}
        ]
    }
    sa = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": service_account, "namespace": namespace}}
    rb = {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding", "metadata": {"name": "prioritymax-operator-rb", "namespace": namespace}, "subjects": [{"kind": "ServiceAccount", "name": service_account, "namespace": namespace}], "roleRef": {"kind": "Role", "name": "prioritymax-operator-role", "apiGroup": "rbac.authorization.k8s.io"}}
    doc = "\n---\n".join([yaml.safe_dump(x) for x in [sa, role, rb]])
    return doc

# -------------------------
# Integration wiring helpers
# -------------------------
async def wire_supervisor_with_autoscaler(supervisor: WorkerSupervisor, autoscaler: PriorityMaxAutoscaler):
    """
    Helper to attach autoscaler and DLQMonitor to supervisor.
    """
    try:
        await supervisor.register_autoscaler(autoscaler)
    except Exception:
        LOG.debug("Failed to register autoscaler with supervisor")
    try:
        dlq = DLQMonitor(queue_name=supervisor.queue_name, queue_client=(RedisQueue() if _HAS_REDIS_QUEUE else None), supervisor=supervisor, storage=supervisor.storage)
        await dlq.start()
        # attach DLQ monitor instance to supervisor for lifecycle
        supervisor._dlq_monitor = dlq
    except Exception:
        LOG.exception("Failed to start DLQMonitor for supervisor")

# -------------------------
# End of Chunk 3
# -------------------------
# -------------------------
# Chunk 4/7 — Tracing, HookManager, ASGI health endpoints & admin CLI
# -------------------------

import contextlib
import traceback

# -------------------------
# OpenTelemetry / tracing initialization (best-effort)
# -------------------------
_HAS_OTEL = False
_otlp_exporter = None
_trace_provider = None
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    try:
        # Try OTLP exporter (grpc/http) if available
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
        _HAS_OTEL = True
    except Exception:
        OTLPSpanExporter = None
        _HAS_OTEL = True  # still True; we will fallback to Console exporter
    # Setup provider
    _trace_provider = TracerProvider()
    tracer = _trace_provider.get_tracer(__name__)
    # Prefer OTLP exporter if available and configured
    try:
        if OTLPSpanExporter and os.getenv("PRIORITYMAX_OTLP_ENDPOINT"):
            _otlp_exporter = OTLPSpanExporter(endpoint=os.getenv("PRIORITYMAX_OTLP_ENDPOINT"))
            _trace_provider.add_span_processor(BatchSpanProcessor(_otlp_exporter))
        else:
            _trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    except Exception:
        _trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    otel_trace.set_tracer_provider(_trace_provider)
except Exception:
    # no op fallback
    otel_trace = None
    tracer = None
    _HAS_OTEL = False

# tracing_span context manager (works with or without otel)
@contextlib.contextmanager
def tracing_span(name: str, attrs: Optional[Dict[str, Any]] = None):
    """
    Context manager to create a trace/span. Safe no-op when OpenTelemetry not installed.
    Usage:
        with tracing_span("task.exec", {"task_id": id}):
            ...
    """
    if _HAS_OTEL and tracer is not None:
        span = tracer.start_span(name)
        if attrs:
            for k, v in attrs.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        try:
            yield span
        except Exception as e:
            try:
                span.record_exception(e)
            except Exception:
                pass
            raise
        finally:
            try:
                span.end()
            except Exception:
                pass
    else:
        # fallback plain try/except wrapper
        try:
            yield None
        except Exception:
            LOG.debug("Exception in traced section: %s", traceback.format_exc())
            raise

# -------------------------
# HookManager — plugin / lifecycle hooks
# -------------------------
class HookManager:
    """
    Lightweight hooks system. Plugins can register callbacks for named events.
    Callbacks may be sync or async. Execution is best-effort (exceptions logged).
    """
    def __init__(self):
        self._hooks: Dict[str, List[Callable[..., Any]]] = {}
        self._lock = asyncio.Lock()

    def register(self, event: str, fn: Callable[..., Any]):
        self._hooks.setdefault(event, []).append(fn)
        LOG.debug("Hook registered: %s -> %s", event, getattr(fn, "__name__", str(fn)))

    async def trigger(self, event: str, *args, **kwargs):
        """
        Trigger hooks for event. Does not block main flow on hook failures.
        """
        hooks = list(self._hooks.get(event, []))
        for fn in hooks:
            try:
                if asyncio.iscoroutinefunction(fn):
                    asyncio.create_task(fn(*args, **kwargs))
                else:
                    # run sync hooks in threadpool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(_GLOBAL_THREADPOOL, lambda: fn(*args, **kwargs))
            except Exception:
                LOG.exception("Hook %s failed", getattr(fn, "__name__", str(fn)))

# global HOOKS instance (used earlier)
try:
    HOOKS  # type: ignore
except NameError:
    HOOKS = HookManager()

# Example hook: simple notifier that logs to storage and metrics
def _sample_on_task_success(task: Any, exec_res: Any, worker_name: str):
    try:
        LOG.info("Hook on_task_success: task=%s worker=%s", getattr(task, "id", "unknown"), worker_name)
    except Exception:
        pass

# register example hook
HOOKS.register("on_task_success", _sample_on_task_success)

# -------------------------
# ASGI helpers / HTTP endpoints for supervisor
# -------------------------
def make_supervisor_routes(supervisor: WorkerSupervisor):
    """
    Return FastAPI-style endpoint functions (async) that can be wired into your app routers.
    Example usage:
        app.add_api_route("/admin/supervisor/health", make_supervisor_routes(sup)["health"], methods=["GET"])
    """
    async def health(request=None):
        try:
            data = await supervisor.report_health()
            return {"ok": True, "data": data}
        except Exception:
            LOG.exception("Supervisor health endpoint error")
            return {"ok": False, "error": "health_failed"}

    async def drain(request=None):
        try:
            timeout = float(os.getenv("PRIORITYMAX_DRAIN_TIMEOUT", "60"))
            await supervisor.drain(timeout=timeout)
            return {"ok": True, "drained_to": 0}
        except Exception:
            LOG.exception("Supervisor drain failed")
            return {"ok": False, "error": "drain_failed"}

    async def scale(request=None):
        """
        Expect JSON body: {"count": N} or query param ?count=N
        """
        try:
            # request may be a FastAPI Request with .json() method or a simple dict
            count = None
            if request is not None:
                try:
                    if hasattr(request, "json"):
                        payload = await request.json()
                        count = payload.get("count")
                except Exception:
                    pass
                # fallback to query params
                try:
                    if hasattr(request, "query_params"):
                        count = count or int(request.query_params.get("count", count or 0))
                except Exception:
                    pass
            if count is None:
                return {"ok": False, "error": "missing_count"}
            await supervisor.scale_to(int(count))
            return {"ok": True, "desired": int(count)}
        except Exception:
            LOG.exception("Supervisor scale failed")
            return {"ok": False, "error": "scale_failed"}

    return {"health": health, "drain": drain, "scale": scale}

# -------------------------
# Admin CLI helpers: promote_dlq & supervisor admin CLI
# -------------------------
def cli_promote_dlq(queue_name: str = DEFAULT_QUEUE, limit: int = DLQ_PROMOTE_BATCH) -> int:
    """
    Synchronous CLI helper to promote DLQ items. Intended for scripts/admin use.
    """
    try:
        # try async promote if queue has sync helper else run loop
        rq = RedisQueue() if _HAS_REDIS_QUEUE else None
        if not rq:
            LOG.warning("No RedisQueue available for DLQ promote")
            return 0
        # Some queue implementations expose synchronous helper
        if hasattr(rq, "promote_dlq_sync"):
            return rq.promote_dlq_sync(queue_name, limit=limit)
        # else run async in event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # run in new loop in thread to avoid blocking running loop
            def _run():
                return asyncio.run(rq.promote_dlq(queue_name, limit=limit))
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as ex:
                f = ex.submit(_run)
                return int(f.result())
        else:
            return loop.run_until_complete(rq.promote_dlq(queue_name, limit=limit))
    except Exception:
        LOG.exception("CLI promote_dlq failed")
        return 0

def supervisor_admin_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-supervisor")
    sub = p.add_subparsers(dest="cmd")

    start = sub.add_parser("start")
    start.add_argument("--pool", default=WORKER_POOL_NAME)
    start.add_argument("--concurrency", type=int, default=WORKER_CONCURRENCY)

    stop = sub.add_parser("stop")
    stop.add_argument("--pool", default=WORKER_POOL_NAME)

    promote = sub.add_parser("promote_dlq")
    promote.add_argument("--queue", default=DEFAULT_QUEUE)
    promote.add_argument("--limit", type=int, default=DLQ_PROMOTE_BATCH)

    args = p.parse_args()
    if args.cmd == "promote_dlq":
        n = cli_promote_dlq(args.queue, args.limit)
        print("promoted:", n)
    else:
        p.print_help()

# -------------------------
# Publish worker metrics helper (used by Worker._report_heartbeat)
# -------------------------
async def publish_worker_metrics(channel: str, payload: Dict[str, Any]):
    """
    Simple publish helper that uses websocket_manager if available, else logs.
    """
    try:
        from app.websocket_manager import ws_manager  # runtime import to avoid circulars
        if ws_manager:
            await ws_manager.broadcast(channel, payload)
            return True
    except Exception:
        pass
    LOG.debug("Worker metrics: %s", payload)
    return False

# -------------------------
# End of Chunk 4
# -------------------------
# -------------------------
# Chunk 5/7 — Advanced OpenTelemetry, Hooks, Subsystem wiring, FastAPI routes
# -------------------------

# -------------------------
# Advanced OpenTelemetry exporters (Jaeger, OTLP HTTP fallback)
# -------------------------
try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    if _HAS_OTEL and _trace_provider:
        if os.getenv("PRIORITYMAX_JAEGER_HOST"):
            je = JaegerExporter(
                agent_host_name=os.getenv("PRIORITYMAX_JAEGER_HOST", "localhost"),
                agent_port=int(os.getenv("PRIORITYMAX_JAEGER_PORT", "6831")),
            )
            _trace_provider.add_span_processor(BatchSpanProcessor(je))
            LOG.info("Jaeger tracing exporter configured for host=%s", os.getenv("PRIORITYMAX_JAEGER_HOST"))
        elif os.getenv("PRIORITYMAX_OTLP_HTTP"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHTTPExporter
                otlp_http = OTLPHTTPExporter(endpoint=os.getenv("PRIORITYMAX_OTLP_HTTP"))
                _trace_provider.add_span_processor(BatchSpanProcessor(otlp_http))
                LOG.info("OTLP HTTP tracing exporter configured for %s", os.getenv("PRIORITYMAX_OTLP_HTTP"))
            except Exception:
                LOG.debug("Failed to init OTLP HTTP exporter")
except Exception:
    LOG.debug("Jaeger/OTLP HTTP exporter not configured")


# -------------------------
# Hook plugin examples
# -------------------------
async def slack_task_fail_notifier(task, exec_res, worker_name):
    """
    Sends Slack alert on task failure if webhook is configured.
    """
    webhook = os.getenv("PRIORITYMAX_SLACK_WEBHOOK")
    if not webhook or not _HAS_AIOHTTP:
        return
    try:
        msg = f":rotating_light: *Task Failed* `{task.id}` on `{worker_name}`\nError: `{exec_res.error}`"
        async with aiohttp.ClientSession() as sess:
            await sess.post(webhook, json={"text": msg})
    except Exception:
        LOG.debug("Slack notifier failed")

async def pagerduty_on_fail(task, exec_res, worker_name):
    """
    PagerDuty event trigger for severe task failures.
    """
    key = os.getenv("PRIORITYMAX_PD_ROUTING_KEY")
    if not key or not _HAS_AIOHTTP:
        return
    try:
        payload = {
            "routing_key": key,
            "event_action": "trigger",
            "payload": {
                "summary": f"Task {task.id} failed on {worker_name}: {exec_res.error}",
                "severity": "critical",
                "source": socket.gethostname(),
            }
        }
        async with aiohttp.ClientSession() as sess:
            await sess.post("https://events.pagerduty.com/v2/enqueue", json=payload)
    except Exception:
        LOG.debug("PagerDuty alert failed")

def email_on_success(task, exec_res, worker_name):
    """
    Sends a local email summary when a long task succeeds.
    Requires SENDMAIL_PATH environment variable or local `sendmail` binary.
    """
    sendmail = os.getenv("PRIORITYMAX_SENDMAIL_PATH", "/usr/sbin/sendmail")
    if not os.path.exists(sendmail):
        return
    try:
        import subprocess
        to_addr = os.getenv("PRIORITYMAX_NOTIFY_EMAIL", "ops@example.com")
        subject = f"Task {task.id} succeeded"
        body = f"Task {task.id} processed by {worker_name}\nOutput: {exec_res.output}"
        msg = f"Subject: {subject}\nTo: {to_addr}\n\n{body}"
        subprocess.Popen([sendmail, "-t"], stdin=subprocess.PIPE).communicate(msg.encode("utf-8"))
    except Exception:
        LOG.debug("email_on_success failed")

# Register hooks
HOOKS.register("on_task_fail", slack_task_fail_notifier)
HOOKS.register("on_task_fail", pagerduty_on_fail)
HOOKS.register("on_task_success", email_on_success)


# -------------------------
# Unified Worker Subsystem Startup / Shutdown
# -------------------------
class WorkerSubsystem:
    """
    Wraps Supervisor + DLQMonitor + SelfHeal + Metrics in a unified lifecycle.
    """
    def __init__(self, pool_name: str = WORKER_POOL_NAME, queue_name: str = DEFAULT_QUEUE):
        self.pool_name = pool_name
        self.queue_name = queue_name
        self.supervisor = WorkerSupervisor(pool_name=pool_name, queue_name=queue_name)
        self.dlq_monitor: Optional[DLQMonitor] = None
        self.self_heal: Optional[SelfHealController] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self):
        if self._running:
            return
        self._running = True
        LOG.info("Starting WorkerSubsystem for %s", self.pool_name)
        await self.supervisor.start()

        # DLQ monitor
        self.dlq_monitor = DLQMonitor(queue_name=self.queue_name, supervisor=self.supervisor)
        await self.dlq_monitor.start()

        # Self-heal controller
        self.self_heal = SelfHealController(supervisor=self.supervisor, queue_name=self.queue_name)
        asyncio.create_task(self._self_heal_loop())

        # start global metrics background loop (if not already)
        try:
            if global_metrics:
                await global_metrics.start()
        except Exception:
            LOG.debug("global_metrics.start() failed")

    async def stop(self):
        if not self._running:
            return
        LOG.info("Stopping WorkerSubsystem for %s", self.pool_name)
        self._running = False
        try:
            if self.dlq_monitor:
                await self.dlq_monitor.stop()
            await self.supervisor.stop()
        except Exception:
            LOG.exception("Error stopping subsystem")
        if global_metrics:
            await global_metrics.stop()
        self._shutdown_event.set()

    async def _self_heal_loop(self):
        while self._running:
            try:
                if self.self_heal:
                    await self.self_heal.attempt_self_heal()
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.debug("self_heal loop error")
            await asyncio.sleep(SELF_HEAL_INTERVAL)


# -------------------------
# FastAPI integration helper for /admin/worker routes
# -------------------------
def register_worker_routes(app, subsystem: WorkerSubsystem):
    """
    Wire admin and health routes for workers directly into a FastAPI app.
    """
    from fastapi import APIRouter, Request
    router = APIRouter(prefix="/admin/worker", tags=["Worker Admin"])

    routes = make_supervisor_routes(subsystem.supervisor)

    @router.get("/health")
    async def health(request: Request):
        return await routes["health"](request)

    @router.post("/drain")
    async def drain(request: Request):
        return await routes["drain"](request)

    @router.post("/scale")
    async def scale(request: Request):
        return await routes["scale"](request)

    @router.post("/promote_dlq")
    async def promote_dlq_route(limit: int = DLQ_PROMOTE_BATCH):
        n = cli_promote_dlq(subsystem.queue_name, limit)
        return {"ok": True, "promoted": n}

    app.include_router(router)
    LOG.info("Worker admin routes registered under /admin/worker")

# -------------------------
# Subsystem lifecycle glue for main.py
# -------------------------
async def start_worker_subsystem():
    """
    Convenience startup function for FastAPI startup event.
    Returns the WorkerSubsystem instance for shutdown management.
    """
    subsystem = WorkerSubsystem()
    await subsystem.start()
    LOG.info("Worker subsystem started")
    return subsystem

async def stop_worker_subsystem(subsystem: WorkerSubsystem):
    """
    Convenience shutdown function for FastAPI shutdown event.
    """
    if subsystem:
        await subsystem.stop()
        LOG.info("Worker subsystem stopped")

# -------------------------
# End of Chunk 5
# -------------------------
# -------------------------
# Chunk 6/7 — Admin metrics endpoints, full CLI, signal handlers, and test harness hooks
# -------------------------

import signal
import inspect
from typing import Coroutine

# -------------------------
# Extra metrics endpoint for supervisor introspection
# -------------------------
def make_metrics_routes(supervisor: WorkerSupervisor):
    """
    Returns ASGI-compatible handlers for exposing detailed worker metrics and traces.
    - /admin/worker/metrics => aggregated metrics for the pool (rolling windows, backlog, worker stats)
    """
    async def metrics(request=None):
        try:
            # basic summary
            health = await supervisor.report_health()
            # include rolling windows from global metrics if present
            gm = global_metrics if 'global_metrics' in globals() else None
            metrics_snapshot = gm.snapshot() if gm else {}
            return {"ok": True, "health": health, "metrics": metrics_snapshot}
        except Exception:
            LOG.exception("Supervisor metrics endpoint error")
            return {"ok": False, "error": "metrics_failed"}
    return {"metrics": metrics}

# -------------------------
# CLI: interactive management for worker subsystem + test hooks
# -------------------------
def full_worker_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-worker")
    sub = p.add_subparsers(dest="cmd")

    start = sub.add_parser("start", help="Start worker subsystem (blocking)")
    start.add_argument("--foreground", action="store_true", help="Run in foreground (do not daemonize)")

    stop = sub.add_parser("stop", help="Stop worker subsystem (via storage desired_count=0)")

    drain = sub.add_parser("drain", help="Drain workers gracefully")
    drain.add_argument("--timeout", type=int, default=60)

    promote = sub.add_parser("promote_dlq", help="Promote DLQ messages")
    promote.add_argument("--limit", type=int, default=DLQ_PROMOTE_BATCH)

    metrics_cmd = sub.add_parser("metrics", help="Dump metrics snapshot")

    test_cli = sub.add_parser("test_hooks", help="Run registered hooks for testing")

    args = p.parse_args()
    # naive single-process CLI implementation
    if args.cmd == "promote_dlq":
        n = cli_promote_dlq(DEFAULT_QUEUE, args.limit)
        print("promoted:", n)
    elif args.cmd == "metrics":
        gm = global_metrics if 'global_metrics' in globals() else None
        if gm:
            print(json.dumps(gm.snapshot(), indent=2))
        else:
            print("No global metrics instance")
    elif args.cmd == "test_hooks":
        print("Registered hooks:")
        for ev, hooks in HOOKS._hooks.items():
            print(f"Event: {ev}")
            for h in hooks:
                print("  -", getattr(h, "__name__", str(h)))
        # attempt to call each hook with dummy args (best-effort)
        for ev, hooks in HOOKS._hooks.items():
            for h in hooks:
                try:
                    if asyncio.iscoroutinefunction(h):
                        asyncio.run(h(type("T", (), {"id": "test"}), type("R", (), {"error": None, "output": "ok"}), "test-worker"))
                    else:
                        h(type("T", (), {"id": "test"}), type("R", (), {"error": None, "output": "ok"}), "test-worker")
                    print(f"Hook {getattr(h, '__name__', str(h))} executed")
                except Exception:
                    print(f"Hook {getattr(h, '__name__', str(h))} failed")
    else:
        p.print_help()

# -------------------------
# Signal handlers for graceful shutdown (intended for main.py wiring)
# -------------------------
def install_signal_handlers(loop: asyncio.AbstractEventLoop, subsystem: Optional[WorkerSubsystem] = None):
    """
    Install SIGINT/SIGTERM handlers that attempt an orderly shutdown of the subsystem.
    Should be called from main startup after event loop creation.
    """
    def _handle_signame(signame):
        LOG.info("Received signal %s: initiating shutdown", signame)
        if subsystem:
            async def _shutdown_and_stop():
                try:
                    await subsystem.stop()
                except Exception:
                    LOG.exception("Error during subsystem.stop()")
                finally:
                    LOG.info("Shutdown complete; stopping loop")
                    loop.stop()
            try:
                asyncio.run_coroutine_threadsafe(_shutdown_and_stop(), loop)
            except Exception:
                LOG.exception("Failed to schedule subsystem.stop()")
        else:
            # stop loop immediately
            try:
                for task in asyncio.all_tasks(loop):
                    task.cancel()
            except Exception:
                pass
            loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _handle_signame(signal.Signals(s).name))
        except NotImplementedError:
            # e.g., on Windows or restricted environments; fall back to default
            signal.signal(sig, lambda signum, frame: _handle_signame(signal.Signals(signum).name))

# -------------------------
# Test harness helpers (pytest fixtures)
# -------------------------
def pytest_worker_subsystem_fixture(tmp_path_factory=None):
    """
    Provide a simple fixture factory to spawn WorkerSubsystem in tests.
    Usage in pytest:
        @pytest.fixture
        def worker_subsystem(event_loop):
            return pytest_worker_subsystem_fixture()
    This is a best-effort helper; tests still need to run loop.run_until_complete on start/stop.
    """
    # create minimal subsystem pointing to ephemeral storage (if available)
    subsystem = WorkerSubsystem(pool_name="test-pool", queue_name="test-queue")
    return subsystem

# -------------------------
# Utility: wrap coroutine with timeout in tests
# -------------------------
async def run_with_timeout(coro: Coroutine, timeout: float = 5.0):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        LOG.warning("Coroutine timed out after %.2fs", timeout)
        return None

# -------------------------
# End of Chunk 6
# -------------------------
# -------------------------
# Chunk 7/7 — Entry point, orchestration, autoscaler feedback loop, Prometheus integration
# -------------------------

# -------------------------
# Autoscaler feedback orchestrator (background)
# -------------------------
class AutoscalerFeedbackOrchestrator:
    """
    Runs a feedback pipeline connecting WorkerSupervisor metrics → PredictorManager → Autoscaler.
    Useful when deploying as a standalone autoscaling daemon with in-process learning loops.
    """
    def __init__(self,
                 supervisor: WorkerSupervisor,
                 autoscaler: Optional[PriorityMaxAutoscaler] = None,
                 predictor: Optional[Any] = None,
                 interval: int = AUTOSCALER_FEEDBACK_INTERVAL):
        self.supervisor = supervisor
        self.autoscaler = autoscaler
        self.predictor = predictor or (PREDICTOR_MANAGER if "PREDICTOR_MANAGER" in globals() else None)
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        LOG.info("Starting AutoscalerFeedbackOrchestrator interval=%ds", self.interval)
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _loop(self):
        while self._running:
            try:
                health = await self.supervisor.report_health()
                summary = {
                    "timestamp": time.time(),
                    "running": health.get("running", 0),
                    "desired": health.get("desired", 0),
                }
                # Optionally add queue backlog & predictor hints
                if self.predictor:
                    features = {"queue_len": summary["running"], "workers": summary["desired"]}
                    try:
                        hint = self.predictor.predict(features)
                        summary["predictor_hint"] = hint
                    except Exception:
                        summary["predictor_hint"] = None
                if self.autoscaler and hasattr(self.autoscaler, "observe"):
                    try:
                        if asyncio.iscoroutinefunction(self.autoscaler.observe):
                            await self.autoscaler.observe(summary)
                        else:
                            self.autoscaler.observe(summary)
                    except Exception:
                        LOG.debug("Autoscaler.observe() failed")
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.debug("Feedback loop error")
                await asyncio.sleep(self.interval)

# -------------------------
# Prometheus standalone endpoint
# -------------------------
async def start_prometheus_server_async(port: int = 9095):
    """
    Start Prometheus exporter in background for standalone worker deployments.
    """
    if not _HAS_PROM:
        LOG.warning("prometheus_client not available; metrics endpoint disabled")
        return
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, lambda: start_http_server(port))
    LOG.info("Prometheus metrics server started on port %d", port)

# -------------------------
# Unified runner for CLI / container entrypoint
# -------------------------
async def run_worker_main():
    """
    Full orchestration runner used by container entrypoint or CLI.
    Starts WorkerSubsystem + Prometheus server + AutoscalerFeedbackOrchestrator.
    """
    LOG.info("PriorityMax Worker Main starting …")
    subsystem = WorkerSubsystem(pool_name=WORKER_POOL_NAME, queue_name=DEFAULT_QUEUE)
    await subsystem.start()
    orchestrator = AutoscalerFeedbackOrchestrator(supervisor=subsystem.supervisor)
    await orchestrator.start()
    await start_prometheus_server_async()

    # Install graceful signal handlers
    loop = asyncio.get_event_loop()
    install_signal_handlers(loop, subsystem)

    LOG.info("PriorityMax Worker running — press Ctrl+C to stop")
    # Wait until subsystem shutdown
    await subsystem._shutdown_event.wait()

    await orchestrator.stop()
    LOG.info("PriorityMax Worker main terminated cleanly.")

# -------------------------
# CLI wiring (advanced)
# -------------------------
def main_cli_entry():
    """
    CLI entrypoint combining all previous commands.
    """
    import argparse
    parser = argparse.ArgumentParser(description="PriorityMax Worker Subsystem CLI")
    sub = parser.add_subparsers(dest="cmd")

    parser_start = sub.add_parser("start", help="Start full worker subsystem (blocking)")
    parser_start.add_argument("--port", type=int, default=9095, help="Prometheus port")

    parser_scale = sub.add_parser("scale", help="Change desired worker count")
    parser_scale.add_argument("--count", type=int, required=True)

    parser_drain = sub.add_parser("drain", help="Drain all workers")
    parser_drain.add_argument("--timeout", type=int, default=60)

    parser_metrics = sub.add_parser("metrics", help="Dump worker metrics snapshot")

    parser_promote = sub.add_parser("promote_dlq", help="Promote DLQ items")
    parser_promote.add_argument("--limit", type=int, default=DLQ_PROMOTE_BATCH)

    args = parser.parse_args()

    if args.cmd == "start":
        asyncio.run(run_worker_main())
    elif args.cmd == "scale":
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ws = WorkerSubsystem()
        loop.run_until_complete(ws.start())
        loop.run_until_complete(ws.supervisor.scale_to(args.count))
        loop.run_until_complete(ws.stop())
        loop.close()
        print(f"Scaled to {args.count}")
    elif args.cmd == "drain":
        ws = WorkerSubsystem()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws.start())
        loop.run_until_complete(ws.supervisor.drain(timeout=args.timeout))
        loop.run_until_complete(ws.stop())
        loop.close()
    elif args.cmd == "metrics":
        gm = global_metrics if 'global_metrics' in globals() else None
        print(json.dumps(gm.snapshot() if gm else {"error": "no metrics"}, indent=2))
    elif args.cmd == "promote_dlq":
        n = cli_promote_dlq(DEFAULT_QUEUE, args.limit)
        print("promoted:", n)
    else:
        parser.print_help()

# -------------------------
# Main section (container entry)
# -------------------------
if __name__ == "__main__":
    """
    When executed directly, runs the complete PriorityMax Worker process.
    Equivalent to: `prioritymax-worker start`
    """
    try:
        asyncio.run(run_worker_main())
    except KeyboardInterrupt:
        LOG.info("Interrupted by user, shutting down …")
    except Exception as e:
        LOG.exception("Worker main crashed: %s", e)
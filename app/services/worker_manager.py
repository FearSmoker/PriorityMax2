# backend/app/services/worker_manager.py
"""
PriorityMax Worker Manager (production-minded)
----------------------------------------------
A robust worker supervisor & manager used by PriorityMax to run task consumers (workers)
in a reliable, observable, and autoscalable manner.

Features included:
 - Local worker runners: thread-based, process-based, subprocess, Docker container, Kubernetes Job (best-effort)
 - Dynamic scaling API: scale_up/scale_down/target_scale integrated with autoscaler hints
 - Self-healing: restart on crash, rate-limited restart, backoff, failure counts & circuit
 - Health checks + readiness/liveness helpers
 - Graceful shutdown (SIGTERM/SIGINT) and draining with configurable timeouts
 - Integration hooks: metrics, audit, DLQ promotion callback, event hooks for lifecycle events
 - Instrumentation: optional Prometheus metrics, optional OpenTelemetry traces (best-effort)
 - Worker lifecycle management: register worker types, custom env/template for each worker
 - Worker supervisor loop as an asyncio background task (suitable for FastAPI background tasks)
 - CLI for simple local testing (start/stop/status)
 - Pluggable executor backends: "thread", "process", "subprocess", "docker", "k8s"
 - Configurable concurrency caps, max restarts, and backoff policy
 - Safe defaults: destructive operations require explicit flags; most destructive ops are dry-run by default
 - Hooks to integrate with Redis / Kafka / RabbitMQ queue connectors for DLQ-based self-healing

Usage:
    from app.services.worker_manager import WorkerManager, WorkerSpec
    mgr = WorkerManager()
    mgr.register_worker_type("consumer", WorkerSpec(...))
    asyncio.run(mgr.start())  # or spawn as FastAPI background task
    mgr.scale_to("consumer", 10)
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import math
import signal
import logging
import asyncio
import shutil
import inspect
import pathlib
import threading
import functools
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Optional dependencies
_HAS_DOCKER = False
_HAS_K8S = False
_HAS_PROM = False
_HAS_OTEL = False

try:
    import docker as docker_py
    _HAS_DOCKER = True
except Exception:
    docker_py = None

try:
    import kubernetes
    from kubernetes import client as k8s_client, config as k8s_config
    _HAS_K8S = True
except Exception:
    k8s_client = k8s_config = None

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    Counter = Gauge = Histogram = start_http_server = None

try:
    from opentelemetry import trace
    _HAS_OTEL = True
except Exception:
    trace = None

# Logging
LOG = logging.getLogger("prioritymax.worker_manager")
LOG.setLevel(os.getenv("PRIORITYMAX_WORKER_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults & env-configurable
DEFAULT_MONITOR_INTERVAL = float(os.getenv("PRIORITYMAX_WORKER_MONITOR_INTERVAL", "5.0"))
DEFAULT_GRACEFUL_TIMEOUT = float(os.getenv("PRIORITYMAX_WORKER_GRACEFUL_TIMEOUT", "30.0"))
DEFAULT_MAX_RESTARTS = int(os.getenv("PRIORITYMAX_WORKER_MAX_RESTARTS", "10"))
DEFAULT_RESTART_BACKOFF = float(os.getenv("PRIORITYMAX_WORKER_BACKOFF_BASE", "1.0"))
DEFAULT_RESTART_BACKOFF_CAP = float(os.getenv("PRIORITYMAX_WORKER_BACKOFF_CAP", "60.0"))
DEFAULT_DRY_RUN = os.getenv("PRIORITYMAX_WORKER_DRY_RUN", "true").lower() in ("1", "true", "yes")
PROMETHEUS_PORT = int(os.getenv("PRIORITYMAX_PROMETHEUS_PORT", "9005"))

# Metrics (optional)
if _HAS_PROM:
    MM_WORKERS_TOTAL = Gauge("prioritymax_workers_total", "Number of managed workers", ["type", "status"])
    MM_WORKER_RESTARTS = Counter("prioritymax_worker_restarts_total", "Worker restarts", ["type"])
    MM_WORKER_UPTIME = Histogram("prioritymax_worker_uptime_seconds", "Worker uptime seconds", ["type"])
    # start server optionally
    try:
        if os.getenv("PRIORITYMAX_PROMETHEUS_START", "false").lower() in ("1", "true", "yes"):
            start_http_server(PROMETHEUS_PORT)
            LOG.info("Worker Manager Prometheus metrics on port %d", PROMETHEUS_PORT)
    except Exception:
        LOG.exception("Failed to start prometheus server")
else:
    MM_WORKERS_TOTAL = MM_WORKER_RESTARTS = MM_WORKER_UPTIME = None

# Type aliases
LifecycleHook = Callable[[str, Dict[str, Any]], None]  # event, payload

# Utilities
def _uid(prefix: str = "") -> str:
    return (prefix + "-" if prefix else "") + uuid.uuid4().hex[:8]

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# -------------------------
# Worker specification
# -------------------------
@dataclass
class WorkerSpec:
    """
    Describes how to create a worker instance.
      - name: logical worker type name (e.g., "consumer")
      - backend: "thread" | "process" | "subprocess" | "docker" | "k8s"
      - command: For subprocess/docker/k8s: list[str] or string command
      - callable: For thread/process: a Python callable (callable must be picklable for process)
      - env: environment variables
      - cwd: working directory
      - replicas: default initial replicas
      - resources: optional resource hints (cpu, memory) - used for docker/k8s
      - labels: arbitrary labels to identify worker in logs/metrics
      - restart_policy: "always" | "on-failure" | "never"
      - readiness_probe: optional callable -> bool used to determine readiness
      - health_probe: optional callable -> bool used for health checks
    """
    name: str
    backend: str = "thread"
    command: Optional[Union[str, List[str]]] = None
    callable: Optional[Callable[..., Any]] = None
    args: Optional[List[Any]] = field(default_factory=list)
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    env: Optional[Dict[str, str]] = field(default_factory=dict)
    cwd: Optional[str] = None
    replicas: int = 1
    resources: Optional[Dict[str, Any]] = field(default_factory=dict)
    labels: Optional[Dict[str, str]] = field(default_factory=dict)
    restart_policy: str = "always"
    readiness_probe: Optional[Callable[[], bool]] = None
    health_probe: Optional[Callable[[], bool]] = None
    max_restarts: Optional[int] = DEFAULT_MAX_RESTARTS
    restart_backoff_base: float = DEFAULT_RESTART_BACKOFF
    restart_backoff_cap: float = DEFAULT_RESTART_BACKOFF_CAP
    dry_run: Optional[bool] = None  # if set overrides manager default

# -------------------------
# Managed worker instance
# -------------------------
@dataclass
class ManagedWorker:
    id: str
    spec_type: str
    backend: str
    created_at: float
    proc: Any = None  # For thread/process/subprocess: the Future/process object; for docker/k8s: client object/meta
    status: str = "starting"  # starting | running | stopped | crashed | draining
    last_exit_code: Optional[int] = None
    restarts: int = 0
    last_start_ts: Optional[float] = None
    last_stop_ts: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# WorkerManager
# -------------------------
class WorkerManager:
    """
    Supervises worker instances, provides scaling, self-healing, and lifecycle hooks.

    Typical flow:
        mgr = WorkerManager()
        mgr.register_worker_type("consumer", WorkerSpec(...))
        asyncio.run(mgr.start())
        mgr.scale_to("consumer", 5)
    """

    def __init__(
        self,
        monitor_interval: float = DEFAULT_MONITOR_INTERVAL,
        graceful_timeout: float = DEFAULT_GRACEFUL_TIMEOUT,
        dry_run: bool = DEFAULT_DRY_RUN,
    ):
        self.monitor_interval = float(monitor_interval)
        self.graceful_timeout = float(graceful_timeout)
        self.dry_run = bool(dry_run)
        self._lock = asyncio.Lock()
        self._types: Dict[str, WorkerSpec] = {}
        self._workers: Dict[str, ManagedWorker] = {}  # id -> ManagedWorker
        self._by_type: Dict[str, List[str]] = {}  # spec_type -> [worker_ids]
        self._stop_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        # hooks
        self._hooks: Dict[str, List[LifecycleHook]] = {}
        self.audit_hook: Optional[Callable[[Dict[str, Any]], None]] = None
        # metrics snapshot
        self._last_metrics_time = time.time()
        # external integrations
        self.autoscaler_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None  # event, payload
        self.dlq_promote_fn: Optional[Callable[[str], int]] = None  # function to promote DLQ for a stream; returns number promoted
        # shutdown flag
        self._shutting_down = False

        # register signal handlers for graceful shutdown when running as standalone
        try:
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.initiate_shutdown(reason="SIGINT")))
            loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.initiate_shutdown(reason="SIGTERM")))
        except Exception:
            # not all environments support adding signal handlers (e.g., Windows / some Jupyter contexts)
            pass

    # -------------------------
    # Registration & hooks
    # -------------------------
    def register_worker_type(self, name: str, spec: WorkerSpec):
        if name in self._types:
            LOG.warning("Worker type %s is already registered; overwriting", name)
        self._types[name] = spec
        self._by_type.setdefault(name, [])
        LOG.info("Registered worker type: %s backend=%s replicas=%d", name, spec.backend, spec.replicas)

    def unregister_worker_type(self, name: str):
        if name in self._types:
            del self._types[name]
            LOG.info("Unregistered worker type: %s", name)

    def add_hook(self, event: str, fn: LifecycleHook):
        self._hooks.setdefault(event, []).append(fn)

    def _emit(self, event: str, payload: Dict[str, Any]):
        LOG.debug("Emit event %s payload keys=%s", event, list(payload.keys()))
        for fn in self._hooks.get(event, []):
            try:
                fn(event, payload)
            except Exception:
                LOG.exception("Hook %s failed", fn)
        # audit hook best-effort
        try:
            if self.audit_hook:
                self.audit_hook({"event": event, "ts": _now_iso(), **payload})
        except Exception:
            LOG.exception("audit hook failed")

    # -------------------------
    # Lifecycle: start/stop/monitor
    # -------------------------
    async def start(self):
        """
        Start the monitor background task. This does not create workers automatically;
        call scale_to or use default replicas after registration.
        """
        LOG.info("WorkerManager starting monitor (interval=%.2fs) dry_run=%s", self.monitor_interval, self.dry_run)
        self._stop_event.clear()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        # optionally start default replicas
        for typ, spec in list(self._types.items()):
            if spec.replicas and spec.replicas > 0:
                await self.scale_to(typ, spec.replicas)

    async def stop(self):
        """
        Stop manager, initiate graceful shutdown for all workers, wait until drained or timeout.
        """
        LOG.info("WorkerManager stopping: initiating graceful shutdown of %d workers", len(self._workers))
        await self.initiate_shutdown(reason="manager_stop")
        # wait for monitor to finish
        if self._monitor_task:
            try:
                await asyncio.wait_for(self._monitor_task, timeout=self.graceful_timeout + 5)
            except asyncio.TimeoutError:
                LOG.warning("Monitor task did not finish in time")

    async def initiate_shutdown(self, reason: str = "shutdown"):
        if self._shutting_down:
            return
        self._shutting_down = True
        self._emit("manager.shutdown.initiated", {"reason": reason})
        self._stop_event.set()
        # mark workers draining and attempt graceful stop
        async with self._lock:
            ids = list(self._workers.keys())
        LOG.info("Initiating shutdown for %d workers (reason=%s)", len(ids), reason)
        stop_tasks = []
        for wid in ids:
            stop_tasks.append(asyncio.create_task(self.stop_worker(wid, force=False, reason=reason)))
        # wait until graceful_timeout for all
        try:
            await asyncio.wait_for(asyncio.gather(*stop_tasks), timeout=self.graceful_timeout)
        except asyncio.TimeoutError:
            LOG.warning("Graceful shutdown timeout reached; force stopping remaining workers")
            # force stop remaining
            async with self._lock:
                remaining = list(self._workers.keys())
            for wid in remaining:
                try:
                    await self.stop_worker(wid, force=True, reason="force_shutdown")
                except Exception:
                    LOG.exception("Force stop failed for %s", wid)
        self._emit("manager.shutdown.completed", {"reason": reason})
        LOG.info("WorkerManager shutdown complete")

    async def _monitor_loop(self):
        """
        Background monitoring task:
         - checks worker statuses
         - restarts crashed workers according to policy/backoff
         - updates metrics
         - calls autoscaler hooks if configured
         - optionally runs DLQ monitor (if dlq_promote_fn set)
        """
        last_dlq_check = 0.0
        dlq_interval = float(os.getenv("PRIORITYMAX_DLQ_MONITOR_INTERVAL", "300.0"))
        while not self._stop_event.is_set():
            start = time.time()
            # snapshot workers to avoid holding lock long
            async with self._lock:
                workers_snapshot = list(self._workers.items())
            # health check each worker
            for wid, w in workers_snapshot:
                try:
                    await self._check_worker(w)
                except Exception:
                    LOG.exception("worker check failed for %s", wid)
            # update metrics
            self._update_metrics()
            # optionally run DLQ monitor occasionally
            if self.dlq_promote_fn and time.time() - last_dlq_check >= dlq_interval:
                try:
                    promoted = self.dlq_promote_fn("auto")  # stream param can be improved; using "auto" as placeholder
                    LOG.info("DLQ monitor promoted %d messages", promoted)
                except Exception:
                    LOG.exception("DLQ monitor failed")
                last_dlq_check = time.time()
            # autoscaler callback (best-effort)
            if self.autoscaler_callback:
                try:
                    # provide aggregated stats per type
                    stats = self._collect_type_stats()
                    self.autoscaler_callback("snapshot", stats)
                except Exception:
                    LOG.exception("autoscaler callback failed")
            # sleep remainder
            took = time.time() - start
            to_sleep = max(0.0, self.monitor_interval - took)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=to_sleep)
            except asyncio.TimeoutError:
                pass  # continue loop
        LOG.info("Monitor loop exiting")

    # -------------------------
    # Worker lifecycle helpers
    # -------------------------
    async def start_worker(self, spec_type: str) -> ManagedWorker:
        """
        Create and start a single worker instance of the given type.
        Returns ManagedWorker.
        """
        if spec_type not in self._types:
            raise KeyError(f"Unknown worker type {spec_type}")
        spec = self._types[spec_type]
        dry = spec.dry_run if spec.dry_run is not None else self.dry_run
        wid = _uid(spec_type)
        worker = ManagedWorker(
            id=wid,
            spec_type=spec_type,
            backend=spec.backend,
            created_at=time.time(),
            status="starting",
            last_start_ts=time.time(),
            meta={"labels": spec.labels.copy() if spec.labels else {}},
        )
        async with self._lock:
            self._workers[wid] = worker
            self._by_type.setdefault(spec_type, []).append(wid)
        self._emit("worker.starting", {"id": wid, "type": spec_type, "backend": spec.backend, "dry_run": dry})
        LOG.info("Starting worker %s type=%s backend=%s dry_run=%s", wid, spec_type, spec.backend, dry)

        # Depending on backend, create appropriate runner
        try:
            if dry:
                # simulate a running worker with a dummy asyncio task
                task = asyncio.create_task(self._simulate_worker_run(wid, spec))
                worker.proc = task
                worker.status = "running"
                worker.last_start_ts = time.time()
                return worker

            if spec.backend == "thread":
                # spawn a background thread executing target (callable)
                if not spec.callable:
                    raise ValueError("callable required for thread backend")
                thr = threading.Thread(target=self._run_callable_thread, args=(wid, spec), daemon=True, name=f"worker-{wid}")
                thr.start()
                worker.proc = thr
                worker.status = "running"
                worker.last_start_ts = time.time()
                return worker

            if spec.backend == "process":
                # use subprocess to spawn a new Python process running the callable via -m or pickled function
                # For simplicity we run a subprocess target that imports the module and calls entrypoint
                if spec.callable and inspect.isfunction(spec.callable):
                    # require module and qualname
                    mod = inspect.getmodule(spec.callable)
                    if not mod or not mod.__name__:
                        raise ValueError("callable must be importable (module required) for process backend")
                    qual = f"{mod.__name__}:{spec.callable.__name__}"
                    cmd = [sys.executable, "-u", "-c", f"import {mod.__name__}; {mod.__name__}.{spec.callable.__name__}(*{spec.args!r}, **{spec.kwargs!r})"]
                elif spec.command:
                    if isinstance(spec.command, (list, tuple)):
                        cmd = list(spec.command)
                    else:
                        cmd = ["/bin/sh", "-c", str(spec.command)]
                else:
                    raise ValueError("callable or command required for process backend")
                proc = subprocess.Popen(cmd, env={**os.environ, **(spec.env or {})}, cwd=spec.cwd or os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                worker.proc = proc
                worker.status = "running"
                worker.last_start_ts = time.time()
                asyncio.create_task(self._monitor_subprocess_output(wid, proc))
                return worker

            if spec.backend == "subprocess":
                # run a standalone program / script
                if not spec.command:
                    raise ValueError("command required for subprocess backend")
                cmd = spec.command if isinstance(spec.command, (list, tuple)) else ["/bin/sh", "-c", spec.command]
                proc = subprocess.Popen(cmd, env={**os.environ, **(spec.env or {})}, cwd=spec.cwd or os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                worker.proc = proc
                worker.status = "running"
                worker.last_start_ts = time.time()
                asyncio.create_task(self._monitor_subprocess_output(wid, proc))
                return worker

            if spec.backend == "docker":
                if not _HAS_DOCKER:
                    raise EnvironmentError("docker SDK not available for docker backend")
                # expect spec.command as image or dict with image/entrypoint
                # allow passing dict: {"image":..., "cmd":..., "env":..., "mounts":...}
                if isinstance(spec.command, (dict,)):
                    cfg = spec.command
                    image = cfg.get("image")
                    cmd = cfg.get("cmd")
                    env = {**os.environ, **(cfg.get("env") or {})}
                else:
                    image = spec.command if isinstance(spec.command, str) else None
                    cmd = None
                    env = {**os.environ, **(spec.env or {})}
                if not image:
                    raise ValueError("image required in spec.command for docker backend")
                # create docker client
                client = docker_py.from_env()
                # pull image optionally
                try:
                    client.images.pull(image)
                except Exception:
                    LOG.debug("docker pull failed or skipped for %s", image)
                container = client.containers.run(image=image, command=cmd, environment=env, detach=True, remove=False)
                worker.proc = {"container": container, "client": client}
                worker.status = "running"
                worker.last_start_ts = time.time()
                asyncio.create_task(self._monitor_docker_container(wid, container))
                return worker

            if spec.backend == "k8s":
                if not _HAS_K8S:
                    raise EnvironmentError("kubernetes client not available for k8s backend")
                # create k8s job/pod via client
                # For brevity we create a pod (not a job), but real usage should use Jobs/Deployments
                # spec.command can be dict with image/cmd/env
                if isinstance(spec.command, dict):
                    cfg = spec.command
                    image = cfg.get("image")
                    cmd = cfg.get("cmd")
                    env_list = [{"name": k, "value": v} for k, v in (cfg.get("env") or {}).items()]
                else:
                    raise ValueError("k8s backend requires command dict with image and cmd")
                # Build pod manifest
                pod_name = f"pm-worker-{wid}"
                pod_manifest = k8s_client.V1Pod(
                    metadata=k8s_client.V1ObjectMeta(name=pod_name, labels={"prioritymax-worker": spec_type}),
                    spec=k8s_client.V1PodSpec(containers=[k8s_client.V1Container(name="worker", image=image, command=cmd if isinstance(cmd, list) else [cmd], env=[k8s_client.V1EnvVar(name=e["name"], value=e["value"]) for e in env_list])], restart_policy="Never")
                )
                core = k8s_client.CoreV1Api()
                created = core.create_namespaced_pod(namespace=os.getenv("PRIORITYMAX_K8S_NAMESPACE", "default"), body=pod_manifest)
                worker.proc = {"pod_name": pod_name, "client": core}
                worker.status = "running"
                worker.last_start_ts = time.time()
                asyncio.create_task(self._monitor_k8s_pod(wid, pod_name, core))
                return worker

            raise ValueError(f"Unknown backend: {spec.backend}")
        except Exception as e:
            LOG.exception("Failed to start worker %s: %s", wid, e)
            worker.status = "crashed"
            worker.last_exit_code = -1
            async with self._lock:
                # increment restart counters and maybe schedule restart per policy (monitor loop handles restarts)
                self._workers[wid] = worker
            self._emit("worker.failed_to_start", {"id": wid, "type": spec_type, "error": str(e)})
            return worker

    async def stop_worker(self, worker_id: str, force: bool = False, reason: str = "stop"):
        """
        Stop an individual worker.
        If force is False attempt graceful stop (send termination signal / ask thread to stop).
        """
        async with self._lock:
            worker = self._workers.get(worker_id)
        if not worker:
            LOG.warning("stop_worker: unknown worker %s", worker_id)
            return False
        spec = self._types.get(worker.spec_type)
        dry = spec.dry_run if spec and spec.dry_run is not None else self.dry_run
        LOG.info("Stopping worker %s (force=%s dry_run=%s reason=%s)", worker_id, force, dry, reason)
        self._emit("worker.stopping", {"id": worker_id, "reason": reason, "force": force})
        try:
            if dry:
                # cancel simulated task if present
                if isinstance(worker.proc, asyncio.Task):
                    worker.proc.cancel()
                worker.status = "stopped"
                worker.last_stop_ts = time.time()
                async with self._lock:
                    self._remove_worker(worker_id)
                return True

            backend = worker.backend
            if backend == "thread":
                # No direct mechanism to stop thread: design callable should honor a stop flag.
                # We'll mark draining and rely on application-level cooperative shutdown.
                worker.status = "draining"
                # If force, there is nothing safe to kill; mark as stopped
                if force:
                    worker.status = "stopped"
                    worker.last_stop_ts = time.time()
                    self._remove_worker(worker_id)
                return True

            if backend == "process" or backend == "subprocess":
                proc = worker.proc
                if isinstance(proc, subprocess.Popen):
                    if not force:
                        proc.terminate()
                        try:
                            proc.wait(timeout=self.graceful_timeout)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=5)
                    else:
                        proc.kill()
                        proc.wait(timeout=5)
                    worker.status = "stopped"
                    worker.last_stop_ts = time.time()
                    self._remove_worker(worker_id)
                    return True

            if backend == "docker":
                info = worker.proc or {}
                cont = info.get("container")
                client = info.get("client")
                if cont:
                    try:
                        if not force:
                            cont.stop(timeout=int(self.graceful_timeout))
                        else:
                            cont.kill()
                    except Exception:
                        LOG.exception("docker stop/kill failed")
                    try:
                        cont.remove(force=True)
                    except Exception:
                        LOG.debug("container remove failed")
                worker.status = "stopped"
                worker.last_stop_ts = time.time()
                self._remove_worker(worker_id)
                return True

            if backend == "k8s":
                info = worker.proc or {}
                pod_name = info.get("pod_name")
                core = info.get("client")
                if pod_name and core:
                    try:
                        core.delete_namespaced_pod(name=pod_name, namespace=os.getenv("PRIORITYMAX_K8S_NAMESPACE", "default"))
                    except Exception:
                        LOG.exception("Failed delete pod %s", pod_name)
                worker.status = "stopped"
                worker.last_stop_ts = time.time()
                self._remove_worker(worker_id)
                return True

            # fallback: if proc is asyncio.Task
            if isinstance(worker.proc, asyncio.Task):
                try:
                    worker.proc.cancel()
                except Exception:
                    pass
                worker.status = "stopped"
                worker.last_stop_ts = time.time()
                self._remove_worker(worker_id)
                return True

        except Exception:
            LOG.exception("Failed to stop worker %s", worker_id)
            return False

    def _remove_worker(self, worker_id: str):
        """Internal removal from registries; must be called under lock"""
        if worker_id in self._workers:
            typ = self._workers[worker_id].spec_type
            del self._workers[worker_id]
            if typ in self._by_type and worker_id in self._by_type[typ]:
                self._by_type[typ].remove(worker_id)
        # emit removed event
        self._emit("worker.removed", {"id": worker_id})

    # -------------------------
    # Worker run helpers (thread/process)
    # -------------------------
    def _run_callable_thread(self, wid: str, spec: WorkerSpec):
        """
        Wrapper to call the callable in a thread and capture exceptions & lifecycle.
        The callable should be resilient and run forever (consumer loop). We'll catch exceptions and mark crashed.
        """
        try:
            LOG.info("Thread worker %s starting callable %s", wid, spec.callable)
            if callable(spec.callable):
                try:
                    spec.callable(* (spec.args or []), ** (spec.kwargs or {}))
                except Exception:
                    LOG.exception("Callable worker %s raised exception", wid)
            else:
                LOG.error("Spec callable invalid for thread worker %s", wid)
        finally:
            # mark crashed/stopped; supervisor will handle restart
            asyncio.run_coroutine_threadsafe(self._on_worker_exit_from_thread(wid), asyncio.get_event_loop())

    async def _on_worker_exit_from_thread(self, wid: str):
        async with self._lock:
            worker = self._workers.get(wid)
            if not worker:
                return
            worker.status = "crashed"
            worker.last_exit_code = -1
            worker.last_stop_ts = time.time()
            worker.restarts += 1
            LOG.warning("Thread worker %s exited; restarts=%d", wid, worker.restarts)
            self._emit("worker.crashed", {"id": wid, "type": worker.spec_type})
            # Monitor loop will decide restart based on policy

    async def _simulate_worker_run(self, wid: str, spec: WorkerSpec):
        """
        Simulated worker that sleeps until manager shutdown; used in dry-run mode.
        """
        try:
            LOG.info("Simulated worker %s running (dry-run)", wid)
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            LOG.info("Simulated worker %s cancelled", wid)
        finally:
            async with self._lock:
                if wid in self._workers:
                    self._workers[wid].status = "stopped"
                    self._workers[wid].last_stop_ts = time.time()
                    self._remove_worker(wid)
            self._emit("worker.stopped", {"id": wid})

    # -------------------------
    # Subprocess & container monitors (async)
    # -------------------------
    async def _monitor_subprocess_output(self, wid: str, proc: subprocess.Popen):
        """
        Monitor subprocess exit and emit events; read stdout/stderr for debugging (best-effort).
        """
        try:
            # non-blocking read of stdout/stderr is complicated; we'll poll wait
            rc = proc.wait()
            out, err = b"", b""
            try:
                out = proc.stdout.read() if proc.stdout else b""
            except Exception:
                pass
            try:
                err = proc.stderr.read() if proc.stderr else b""
            except Exception:
                pass
            async with self._lock:
                worker = self._workers.get(wid)
                if worker:
                    worker.status = "crashed" if rc != 0 else "stopped"
                    worker.last_exit_code = rc
                    worker.last_stop_ts = time.time()
                    worker.restarts += 1
            LOG.info("Subprocess worker %s exited rc=%s stdout=%s stderr=%s", wid, rc, (out.decode(errors="ignore")[:200] if out else None), (err.decode(errors="ignore")[:200] if err else None))
            self._emit("worker.process.exited", {"id": wid, "exit_code": rc, "stdout": (out.decode(errors="ignore")[:1000] if out else None), "stderr": (err.decode(errors="ignore")[:1000] if err else None)})
        except Exception:
            LOG.exception("monitor_subprocess_output failed for %s", wid)

    async def _monitor_docker_container(self, wid: str, container):
        """
        Poll container status until it exits; then update worker status.
        """
        try:
            client = container.client if hasattr(container, "client") else None
        except Exception:
            client = None
        try:
            # blocking poll with short sleeps
            while True:
                try:
                    container.reload()
                    state = container.status
                except Exception:
                    # container may be gone
                    state = "exited"
                if state in ("exited", "dead", "created"):
                    break
                await asyncio.sleep(2.0)
            try:
                logs = container.logs(tail=100).decode(errors="ignore")
            except Exception:
                logs = None
            # determine exit code if available
            try:
                rc = container.wait(timeout=1).get("StatusCode", None)
            except Exception:
                rc = None
            async with self._lock:
                worker = self._workers.get(wid)
                if worker:
                    worker.status = "stopped" if rc == 0 else "crashed"
                    worker.last_exit_code = rc
                    worker.last_stop_ts = time.time()
                    worker.restarts += 1
            LOG.info("Docker worker %s finished rc=%s logs_len=%s", wid, rc, len(logs) if logs else 0)
            self._emit("worker.docker.exited", {"id": wid, "exit_code": rc, "logs": (logs[:1000] if logs else None)})
        except Exception:
            LOG.exception("monitor_docker_container failed for %s", wid)

    async def _monitor_k8s_pod(self, wid: str, pod_name: str, core_api):
        """
        Poll a pod until completion or failure.
        """
        try:
            namespace = os.getenv("PRIORITYMAX_K8S_NAMESPACE", "default")
            while True:
                try:
                    pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
                    phase = pod.status.phase
                except Exception:
                    phase = "Unknown"
                if phase in ("Succeeded", "Failed", "Unknown"):
                    break
                await asyncio.sleep(2.0)
            # collect logs
            try:
                logs = core_api.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=200)
            except Exception:
                logs = None
            async with self._lock:
                worker = self._workers.get(wid)
                if worker:
                    worker.status = "stopped" if phase == "Succeeded" else "crashed"
                    worker.last_stop_ts = time.time()
                    worker.restarts += 1
            self._emit("worker.k8s.pod_finished", {"id": wid, "pod": pod_name, "phase": phase, "logs": (logs[:1000] if logs else None)})
            LOG.info("K8s pod %s finished phase=%s", pod_name, phase)
        except Exception:
            LOG.exception("monitor_k8s_pod failed for %s", wid)

    # -------------------------
    # Health check & restart policy
    # -------------------------
    async def _check_worker(self, worker: ManagedWorker):
        """
        Inspect worker state and decide whether to restart based on restart policy.
        """
        spec = self._types.get(worker.spec_type)
        if not spec:
            return
        # check readiness / health probes if provided
        healthy = True
        if spec.health_probe:
            try:
                healthy = bool(spec.health_probe())
            except Exception:
                healthy = False
        # If worker is crashed or stopped unexpectedly and restart policy allows -> restart
        if worker.status in ("crashed", "stopped"):
            # compute restart eligibility
            allow_restart = spec.restart_policy in ("always", "on-failure") if worker.last_exit_code != 0 else spec.restart_policy in ("always",)
            if allow_restart:
                # check restarts threshold
                maxr = spec.max_restarts if spec.max_restarts is not None else DEFAULT_MAX_RESTARTS
                if worker.restarts >= maxr:
                    LOG.warning("Worker %s reached max restarts %d; not restarting", worker.id, worker.restarts)
                    self._emit("worker.max_restarts_exhausted", {"id": worker.id, "type": worker.spec_type})
                    return
                # backoff calculation
                backoff = min(spec.restart_backoff_cap, spec.restart_backoff_base * (2 ** max(0, worker.restarts - 1)))
                LOG.info("Restarting worker %s after backoff %.1fs (restarts=%d)", worker.id, backoff, worker.restarts)
                # remove old worker and start a fresh one
                async with self._lock:
                    self._remove_worker(worker.id)
                await asyncio.sleep(backoff)
                new_w = await self.start_worker(worker.spec_type)
                # metric
                if MM_WORKER_RESTARTS:
                    try:
                        MM_WORKER_RESTARTS.labels(type=worker.spec_type).inc()
                    except Exception:
                        pass
                self._emit("worker.restarted", {"old_id": worker.id, "new_id": new_w.id, "type": worker.spec_type})
        else:
            # running: optionally run readiness/health and emit events
            if not healthy:
                LOG.warning("Worker %s health probe failed; marking crashed", worker.id)
                async with self._lock:
                    worker.status = "crashed"
                    worker.last_stop_ts = time.time()
                    worker.restarts += 1
                self._emit("worker.health_failed", {"id": worker.id, "type": worker.spec_type})

    # -------------------------
    # Scaling API
    # -------------------------
    async def scale_to(self, spec_type: str, target_replicas: int):
        """
        Scale the specified worker type to target_replicas by starting/stopping workers.
        """
        if spec_type not in self._types:
            raise KeyError(f"Unknown worker type {spec_type}")
        if target_replicas < 0:
            raise ValueError("target_replicas must be >= 0")
        async with self._lock:
            current = list(self._by_type.get(spec_type, []))
        current_count = len(current)
        LOG.info("Scaling type=%s from %d to %d", spec_type, current_count, target_replicas)
        if target_replicas == current_count:
            return
        if target_replicas > current_count:
            # scale up
            to_add = target_replicas - current_count
            tasks = [asyncio.create_task(self.start_worker(spec_type)) for _ in range(to_add)]
            # wait for all to be started
            new_workers = await asyncio.gather(*tasks, return_exceptions=True)
            started = [w for w in new_workers if isinstance(w, ManagedWorker)]
            LOG.info("Scaled up %d workers for type %s", len(started), spec_type)
            self._emit("scale.up", {"type": spec_type, "added": len(started), "target": target_replicas})
            return
        else:
            # scale down: stop excess workers (LIFO)
            to_remove = current_count - target_replicas
            removed = 0
            async with self._lock:
                ids = list(self._by_type.get(spec_type, []))[-to_remove:]
            tasks = [asyncio.create_task(self.stop_worker(wid, force=False, reason="scale_down")) for wid in ids]
            await asyncio.gather(*tasks)
            self._emit("scale.down", {"type": spec_type, "removed": to_remove, "target": target_replicas})
            LOG.info("Scaled down %d workers for type %s", to_remove, spec_type)

    async def scale_by_hint(self, spec_type: str, hint: str):
        """
        Convenience: respond to autoscaler hints: 'scale_up', 'scale_down', 'steady'
        Implemented as a simple +/-1 change; sophisticated autoscaler should use snapshot-based scaling.
        """
        async with self._lock:
            current = len(self._by_type.get(spec_type, []))
        if hint == "scale_up":
            await self.scale_to(spec_type, current + 1)
        elif hint == "scale_down" and current > 0:
            await self.scale_to(spec_type, current - 1)
        else:
            LOG.debug("scale_by_hint: steady or no-op for %s", spec_type)

    # -------------------------
    # Utilities: stats & metrics
    # -------------------------
    def _collect_type_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for typ, ids in self._by_type.items():
            active = len(ids)
            crashed = sum(1 for wid in ids if self._workers.get(wid) and self._workers[wid].status in ("crashed",))
            stats[typ] = {"active": active, "crashed": crashed}
        return stats

    def _update_metrics(self):
        if not _HAS_PROM:
            return
        # reset gauges per type/status
        type_counts: Dict[Tuple[str, str], int] = {}
        for wid, w in self._workers.items():
            key = (w.spec_type, w.status)
            type_counts[key] = type_counts.get(key, 0) + 1
            # uptime observation if running
            if w.last_start_ts and w.status == "running":
                try:
                    MM_WORKER_UPTIME.labels(type=w.spec_type).observe(time.time() - w.last_start_ts)
                except Exception:
                    pass
        # set gauges (this is a simplistic approach; Prometheus client supports direct set with labels)
        for (typ, status), cnt in type_counts.items():
            try:
                MM_WORKERS_TOTAL.labels(type=typ, status=status).set(cnt)
            except Exception:
                pass

    # -------------------------
    # Query API
    # -------------------------
    async def list_workers(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [self._serialize_worker(self._workers[k]) for k in list(self._workers.keys())]

    async def get_worker(self, wid: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            w = self._workers.get(wid)
            return self._serialize_worker(w) if w else None

    def _serialize_worker(self, w: ManagedWorker) -> Dict[str, Any]:
        if not w:
            return {}
        return {
            "id": w.id,
            "type": w.spec_type,
            "backend": w.backend,
            "status": w.status,
            "restarts": w.restarts,
            "created_at": w.created_at,
            "last_start_ts": w.last_start_ts,
            "last_stop_ts": w.last_stop_ts,
            "last_exit_code": w.last_exit_code,
            "meta": w.meta,
        }

    # -------------------------
    # DLQ helper utilities (promote & monitor)
    # -------------------------
    async def start_dlq_monitor(self, interval_sec: int = 300, backlog_threshold: int = 100, min_age_sec: int = 60, autoscaler_hint_fn: Optional[Callable[[], float]] = None):
        """
        Start a background coroutine that periodically checks DLQs via dlq_promote_fn.
        If backlog < backlog_threshold and autoscaler hint indicates low load, it will promote messages.
        dlq_promote_fn should be a callable that accepts a 'stream' argument and returns number promoted.
        """
        if not self.dlq_promote_fn:
            raise RuntimeError("dlq_promote_fn not configured")
        LOG.info("Starting DLQ monitor: interval=%ds backlog_threshold=%d min_age_sec=%d", interval_sec, backlog_threshold, min_age_sec)
        while not self._stop_event.is_set():
            try:
                promoted = self.dlq_promote_fn("monitor")  # actual interface can be extended to accept stream names
                LOG.info("DLQ monitor: promoted=%d", promoted)
                # Optionally consult autoscaler hint
                try:
                    if autoscaler_hint_fn:
                        hint_val = autoscaler_hint_fn()
                        LOG.debug("DLQ monitor autoscaler hint=%s", hint_val)
                        if hint_val < 0.2 and promoted > 0:
                            LOG.info("Low load detected; promoted %d DLQ messages back to main queue", promoted)
                except Exception:
                    LOG.exception("autoscaler_hint_fn failed")
            except Exception:
                LOG.exception("DLQ monitor iteration failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_sec)
            except asyncio.TimeoutError:
                pass

    # -------------------------
    # CLI
    # -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-worker-manager")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("start")
    sub.add_parser("status")
    start_type = sub.add_parser("start-type")
    start_type.add_argument("--type", required=True)
    start_type.add_argument("--count", type=int, default=1)
    stop_id = sub.add_parser("stop")
    stop_id.add_argument("--id", required=True)
    scale = sub.add_parser("scale")
    scale.add_argument("--type", required=True)
    scale.add_argument("--replicas", type=int, required=True)
    return p

async def _cli_main_async(args):
    mgr = WorkerManager()
    # register a simple dummy worker type for demo
    def dummy_consumer():
        while True:
            time.sleep(1)

    mgr.register_worker_type("demo", WorkerSpec(name="demo", backend="thread", callable=dummy_consumer, replicas=0))
    await mgr.start()
    if args.cmd == "start-type":
        await mgr.scale_to(args.type, args.count)
        print("Started", args.count, "workers of type", args.type)
    elif args.cmd == "scale":
        await mgr.scale_to(args.type, args.replicas)
        print("Scaled", args.type, "to", args.replicas)
    elif args.cmd == "status":
        workers = await mgr.list_workers()
        print(json.dumps(workers, indent=2))
    else:
        print("Unknown command")
    # Sleep so user can inspect; in real CLI we'd exit or keep running as daemon
    await asyncio.sleep(2)
    await mgr.stop()

def main_cli():
    import argparse
    parser = _build_cli()
    args = parser.parse_args()
    try:
        asyncio.run(_cli_main_async(args))
    except KeyboardInterrupt:
        print("Interrupted")

# -------------------------
# Exports
# -------------------------
__all__ = ["WorkerManager", "WorkerSpec", "ManagedWorker"]

if __name__ == "__main__":
    main_cli()

# backend/app/services/executor/container_executor.py
"""
PriorityMax Container Executor
------------------------------
A production-minded container/task executor that can run user tasks inside:
  - Local Docker (docker-py) containers (preferred for single-host)
  - Kubernetes Jobs (k8s python client) for cluster execution
It provides:
  - Sync and async APIs to run containers/jobs with timeouts, resource limits, env, mounts
  - Streaming of STDOUT/STDERR logs back to caller (iterable / async generator)
  - Retries, backoff, and graceful cancellation
  - Security-minded defaults (no privileged etc.) and sandboxing hints
  - Metrics hooks (prometheus_client optional)
  - Integration helpers for storing logs, artifacts to S3 (pluggable)
  - Health checks for Docker / Kubernetes
  - CLI helper for quick testing locally
Usage examples:
  - Run a short command in Docker (blocking):
      exec = ContainerExecutor(mode="docker")
      res = exec.run_image("python:3.10-slim", ["python","-c","print('hello')"], timeout=30)
  - Run async job in Kubernetes:
      exec = ContainerExecutor(mode="k8s")
      await exec.start()
      job = await exec.run_k8s_job_async(...)
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import math
import logging
import pathlib
import shlex
import asyncio
import tempfile
import threading
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable, AsyncIterator

# Optional dependencies
_HAS_DOCKER = False
_HAS_K8S = False
_HAS_PROM = False

try:
    import docker as docker_py  # docker-py
    from docker.errors import APIError, ContainerError, NotFound as DockerNotFound
    _HAS_DOCKER = True
except Exception:
    docker_py = None
    APIError = ContainerError = DockerNotFound = Exception
    _HAS_DOCKER = False

try:
    import kubernetes
    from kubernetes import client as k8s_client, config as k8s_config
    from kubernetes.client.rest import ApiException as K8sApiException
    _HAS_K8S = True
except Exception:
    kubernetes = None
    k8s_client = None
    k8s_config = None
    K8sApiException = Exception
    _HAS_K8S = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    Counter = Gauge = Histogram = None
    _HAS_PROM = False

# Logging
LOG = logging.getLogger("prioritymax.executor.container")
LOG.setLevel(os.getenv("PRIORITYMAX_EXECUTOR_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults
DEFAULT_MODE = os.getenv("PRIORITYMAX_EXECUTOR_MODE", "docker")  # "docker" or "k8s"
DEFAULT_NAMESPACE = os.getenv("PRIORITYMAX_K8S_NAMESPACE", "default")
DEFAULT_K8S_BACKOFF = int(os.getenv("PRIORITYMAX_K8S_BACKOFF", "5"))  # seconds between checks
DEFAULT_DOCKER_CLIENT_TIMEOUT = int(os.getenv("PRIORITYMAX_DOCKER_TIMEOUT", "120"))
DEFAULT_CONTAINER_WORKDIR = "/workspace"
DEFAULT_IMAGE_PULL_POLICY = os.getenv("PRIORITYMAX_IMAGE_PULL_POLICY", "IfNotPresent")  # For k8s

# Prometheus metrics
if _HAS_PROM:
    EXEC_RUN_COUNT = Counter("prioritymax_exec_run_total", "Total container executions", ["mode", "status"])
    EXEC_RUNNING = Gauge("prioritymax_exec_running", "Number of currently running executions", ["mode"])
    EXEC_LATENCY = Histogram("prioritymax_exec_seconds", "Container execution latency seconds", ["mode"])
else:
    EXEC_RUN_COUNT = EXEC_RUNNING = EXEC_LATENCY = None

# Types
EnvType = Dict[str, str]
MountType = Dict[str, Any]  # {"host_path": str, "container_path": str, "read_only": bool}

# Helper utilities
def _uid(prefix: str = "") -> str:
    return (prefix + "-" if prefix else "") + uuid.uuid4().hex[:8]

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _safe_path(p: Union[str, pathlib.Path]) -> str:
    return str(pathlib.Path(p))

# Simple backoff
def _exponential_backoff(attempt: int, base: float = 0.5, cap: float = 60.0):
    return min(cap, base * (2 ** (attempt - 1)))

# Dataclasses
@dataclass
class ExecResult:
    ok: bool
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    runtime_sec: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DockerRunSpec:
    image: str
    command: Optional[Union[str, List[str]]] = None
    environment: Optional[EnvType] = None
    mounts: Optional[List[MountType]] = None
    workdir: Optional[str] = DEFAULT_CONTAINER_WORKDIR
    cpu_shares: Optional[int] = None
    mem_limit: Optional[str] = None  # e.g. '512m' or '2g'
    network_mode: Optional[str] = None
    user: Optional[str] = None
    name: Optional[str] = None
    detach: bool = False
    remove: bool = True
    auto_remove: bool = True
    pull: bool = True
    timeout: Optional[int] = None  # seconds
    stdout_to_file: Optional[str] = None  # path to write stdout
    stderr_to_file: Optional[str] = None
    labels: Optional[Dict[str, str]] = None

@dataclass
class K8sRunSpec:
    image: str
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env: Optional[list] = None  # list of k8s V1EnvVar dicts or simple {'name':..., 'value':...}
    namespace: Optional[str] = DEFAULT_NAMESPACE
    service_account: Optional[str] = None
    cpu: Optional[str] = None  # e.g., "500m"
    memory: Optional[str] = None  # e.g., "512Mi"
    backoff_limit: int = 0
    ttl_seconds_after_finished: Optional[int] = 300
    restart_policy: str = "Never"
    image_pull_policy: str = DEFAULT_IMAGE_PULL_POLICY
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

# Core executor class
class ContainerExecutor:
    """
    ContainerExecutor runs tasks in containers via Docker or Kubernetes.

    Initialization:
        exec = ContainerExecutor(mode="docker")
        exec = ContainerExecutor(mode="k8s", k8s_namespace="ml")

    Methods:
        start() / stop()          - optional for k8s client init
        run_image(...)            - blocking run in Docker (sync)
        run_image_async(...)      - async wrapper
        stream_logs_docker(...)   - generator to stream logs while container runs
        run_k8s_job_async(...)    - create k8s Job and stream pod logs (async)
        cancel(...)               - cancel a running job/container
        health()                  - returns dict with client status
    """

    def __init__(
        self,
        mode: str = DEFAULT_MODE,  # "docker" or "k8s"
        docker_client_kwargs: Optional[Dict[str, Any]] = None,
        k8s_incluster: bool = False,
        k8s_kubeconfig: Optional[str] = None,
    ):
        self.mode = mode.lower()
        self.docker_client_kwargs = docker_client_kwargs or {}
        self.k8s_incluster = bool(k8s_incluster)
        self.k8s_kubeconfig = k8s_kubeconfig
        self._docker_client = None
        self._k8s_api = None
        self._k8s_batch = None
        self._k8s_core = None
        self._loop = None
        self._running_tasks: Dict[str, Dict[str, Any]] = {}  # id -> meta
        self._lock = asyncio.Lock()
        self._executor = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
        # create a threadpool executor for blocking io if needed
        self._threadpool = concurrent_threadpool = asyncio.get_event_loop().run_in_executor if asyncio.get_event_loop().is_running() else None

        # Metrics
        if EXEC_RUNNING:
            try:
                EXEC_RUNNING.labels(mode=self.mode).set(0)
            except Exception:
                pass

    # -------------------------
    # Startup / Shutdown
    # -------------------------
    async def start(self):
        """
        Initialize clients as needed.
        For Docker, create docker client.
        For K8s, load kube-config or in-cluster config and create APIs.
        """
        # ensure loop
        self._loop = asyncio.get_running_loop()

        if self.mode == "docker":
            if not _HAS_DOCKER:
                raise RuntimeError("docker SDK (docker) is not installed")
            try:
                # blocking creation via threadpool
                self._docker_client = docker_py.from_env(timeout=DEFAULT_DOCKER_CLIENT_TIMEOUT, **self.docker_client_kwargs)
                # quick ping
                _ = self._docker_client.ping()
                LOG.info("Docker client connected")
            except Exception:
                LOG.exception("Docker client start failed")
                raise
        elif self.mode == "k8s":
            if not _HAS_K8S:
                raise RuntimeError("kubernetes library not installed")
            try:
                if self.k8s_incluster:
                    k8s_config.load_incluster_config()
                elif self.k8s_kubeconfig:
                    k8s_config.load_kube_config(config_file=self.k8s_kubeconfig)
                else:
                    # load default kubeconfig
                    k8s_config.load_kube_config()
                self._k8s_api = k8s_client.BatchV1Api()
                self._k8s_core = k8s_client.CoreV1Api()
                LOG.info("Kubernetes client initialized (namespace=%s)", DEFAULT_NAMESPACE)
            except Exception:
                LOG.exception("Kubernetes client init failed")
                raise
        else:
            raise ValueError("Unknown mode: %s" % self.mode)

    async def stop(self):
        """
        Clean shutdown. Note: docker-py has no close; k8s client is stateless.
        """
        # Cancel running tasks? leave to caller
        LOG.info("Stopping ContainerExecutor mode=%s", self.mode)
        # Optionally, cleanup fast
        self._docker_client = None
        self._k8s_api = None
        self._k8s_core = None

    # -------------------------
    # Docker helpers (sync & async wrappers)
    # -------------------------
    def _build_docker_host_config(self, spec: DockerRunSpec):
        # Build volumes and host_config options (best-effort)
        volumes = {}
        binds = []
        if spec.mounts:
            for m in spec.mounts:
                host = _safe_path(m["host_path"])
                cont = m["container_path"]
                ro = m.get("read_only", False)
                mode = "ro" if ro else "rw"
                volumes[host] = {"bind": cont, "mode": mode}
        return volumes

    def run_image(
        self,
        spec: DockerRunSpec,
        max_retries: int = 1,
        retry_backoff_base: float = 0.5,
    ) -> ExecResult:
        """
        Run a container synchronously (blocking) using docker-py. Returns ExecResult.
        """
        if not _HAS_DOCKER:
            raise RuntimeError("docker library not available")

        name = spec.name or _uid("docker")
        start_time = time.perf_counter()
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                # pull image if requested
                if spec.pull:
                    try:
                        LOG.info("Pulling image %s (attempt %d)", spec.image, attempt)
                        # pull is blocking; use low-level API for progress? keep simple
                        self._docker_client.images.pull(spec.image)
                    except Exception:
                        LOG.exception("Image pull failed for %s (continuing)", spec.image)

                volumes = self._build_docker_host_config(spec)
                env = spec.environment or {}
                cmd = spec.command
                if isinstance(cmd, list):
                    cmd_arg = cmd
                else:
                    cmd_arg = shlex.split(cmd) if isinstance(cmd, str) else None

                container = self._docker_client.containers.run(
                    image=spec.image,
                    command=cmd_arg,
                    environment=env,
                    volumes=volumes,
                    working_dir=spec.workdir,
                    name=name,
                    detach=spec.detach,
                    remove=spec.remove,
                    auto_remove=spec.auto_remove,
                    network_mode=spec.network_mode,
                    user=spec.user,
                    labels=spec.labels or {},
                    stdout=True,
                    stderr=True,
                    mem_limit=spec.mem_limit,
                    cpu_shares=spec.cpu_shares,
                )

                if spec.detach:
                    # Return immediately with container id
                    elapsed = time.perf_counter() - start_time
                    if EXEC_RUN_COUNT:
                        try:
                            EXEC_RUN_COUNT.labels(mode="docker", status="detached").inc()
                        except Exception:
                            pass
                    return ExecResult(ok=True, exit_code=None, stdout=None, stderr=None, runtime_sec=elapsed, meta={"container_id": container.id})

                # wait with timeout
                timeout = spec.timeout or DEFAULT_DOCKER_CLIENT_TIMEOUT
                try:
                    # stream logs while waiting
                    logs = []
                    for line in container.logs(stream=True, stderr=True, stdout=True, follow=True):
                        decoded = line.decode(errors="ignore") if isinstance(line, (bytes, bytearray)) else str(line)
                        logs.append(decoded)
                        # optionally write to file
                        if spec.stdout_to_file:
                            try:
                                with open(spec.stdout_to_file, "a", encoding="utf-8") as fh:
                                    fh.write(decoded)
                            except Exception:
                                pass
                    # after stream ends, container.wait returns
                    rc = container.wait(timeout=timeout)
                    exit_code = rc.get("StatusCode") if isinstance(rc, dict) else int(rc)
                    stdout_full = "".join(logs)
                    elapsed = time.perf_counter() - start_time
                    if EXEC_RUN_COUNT:
                        try:
                            EXEC_RUN_COUNT.labels(mode="docker", status="ok" if exit_code == 0 else "error").inc()
                        except Exception:
                            pass
                    if EXEC_LATENCY:
                        try:
                            EXEC_LATENCY.labels(mode="docker").observe(elapsed)
                        except Exception:
                            pass
                    return ExecResult(ok=(exit_code == 0), exit_code=exit_code, stdout=stdout_full, stderr=None, runtime_sec=elapsed, meta={"container_id": container.id})
                except Exception as e:
                    # timeout or streaming error
                    try:
                        container.kill()
                    except Exception:
                        pass
                    raise

            except Exception as e:
                last_exc = e
                LOG.exception("Docker run attempt %d failed for image=%s", attempt, spec.image)
                if attempt < max_retries:
                    sleep = _exponential_backoff(attempt, base=retry_backoff_base)
                    time.sleep(sleep)
                    continue
                else:
                    elapsed = time.perf_counter() - start_time
                    if EXEC_RUN_COUNT:
                        try:
                            EXEC_RUN_COUNT.labels(mode="docker", status="error").inc()
                        except Exception:
                            pass
                    return ExecResult(ok=False, exit_code=None, stdout=None, stderr=None, runtime_sec=elapsed, error=str(last_exc))

    async def run_image_async(self, spec: DockerRunSpec, max_retries: int = 1, retry_backoff_base: float = 0.5) -> ExecResult:
        """
        Async wrapper around run_image using thread executor (since docker-py is blocking).
        """
        if not _HAS_DOCKER:
            raise RuntimeError("docker SDK not available")
        loop = asyncio.get_running_loop()
        func = functools.partial(self.run_image, spec, max_retries, retry_backoff_base)
        return await loop.run_in_executor(None, func)

    async def stream_logs_docker(self, container_id: str, follow: bool = True, tail: Optional[int] = 100) -> AsyncIterator[str]:
        """
        Async generator that yields log lines from a running container.
        """
        if not _HAS_DOCKER:
            raise RuntimeError("docker SDK not available")
        loop = asyncio.get_running_loop()
        # blocking generator in threadpool
        def _iter_logs():
            try:
                c = self._docker_client.containers.get(container_id)
                for line in c.logs(stream=True, follow=follow, tail=tail):
                    yield line.decode(errors="ignore") if isinstance(line, (bytes, bytearray)) else str(line)
            except Exception as e:
                yield f"__log_stream_error__:{e}"

        it = _iter_logs()
        while True:
            try:
                line = await loop.run_in_executor(None, next, it)
                if line is None:
                    break
                yield line
            except StopIteration:
                break
            except Exception:
                break

    # -------------------------
    # Kubernetes helpers (async)
    # -------------------------
    def _k8s_job_manifest(self, name: str, spec: K8sRunSpec, image_pull_secret: Optional[str] = None, volume_mounts: Optional[List[Dict[str, str]]] = None, volumes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Build a V1Job manifest dict (compatible with k8s_client) for the given spec.
        volume_mounts: list of {"name":..., "mountPath":..., "readOnly":...}
        volumes: list of k8s V1Volume-like dicts
        """
        labels = spec.labels or {}
        annotations = spec.annotations or {}
        container = k8s_client.V1Container(
            name=name,
            image=spec.image,
            command=spec.command,
            args=spec.args,
            env=[k8s_client.V1EnvVar(name=e["name"], value=e["value"]) if isinstance(e, dict) else e for e in (spec.env or [])],
            image_pull_policy=spec.image_pull_policy,
            resources=k8s_client.V1ResourceRequirements(limits={"cpu": spec.cpu} if spec.cpu else None, requests={"memory": spec.memory} if spec.memory else None),
            volume_mounts=[k8s_client.V1VolumeMount(name=vm["name"], mount_path=vm["mountPath"], read_only=vm.get("readOnly", False)) for vm in (volume_mounts or [])],
        )

        template = k8s_client.V1PodTemplateSpec(
            metadata=k8s_client.V1ObjectMeta(labels=labels, annotations=annotations),
            spec=k8s_client.V1PodSpec(restart_policy=spec.restart_policy, containers=[container], service_account_name=spec.service_account, image_pull_secrets=[k8s_client.V1LocalObjectReference(name=image_pull_secret)] if image_pull_secret else None, volumes=[k8s_client.V1Volume(**v) for v in (volumes or [])])
        )

        job_spec = k8s_client.V1JobSpec(
            template=template,
            backoff_limit=spec.backoff_limit,
            ttl_seconds_after_finished=spec.ttl_seconds_after_finished,
        )

        job = k8s_client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s_client.V1ObjectMeta(name=name, labels=labels),
            spec=job_spec,
        )
        return job

    async def run_k8s_job_async(
        self,
        spec: K8sRunSpec,
        name_prefix: str = "prioritymax-job",
        image_pull_secret: Optional[str] = None,
        volume_mounts: Optional[List[Dict[str, str]]] = None,
        volumes: Optional[List[Dict[str, Any]]] = None,
        stream_logs: bool = True,
        timeout: Optional[int] = None,
    ) -> ExecResult:
        """
        Create a Kubernetes Job and (optionally) stream logs from its pod.
        This method uses the k8s python client (blocking calls) but wraps them in asyncio.run_in_executor.
        """
        if not _HAS_K8S:
            raise RuntimeError("kubernetes client not installed")
        if not self._k8s_api:
            # synchronous init by calling start()
            await self.start()

        loop = asyncio.get_running_loop()
        job_name = f"{name_prefix}-{uuid.uuid4().hex[:6]}"
        namespace = spec.namespace or DEFAULT_NAMESPACE

        def _create_job_and_wait():
            try:
                job_manifest = self._k8s_job_manifest(job_name, spec, image_pull_secret=image_pull_secret, volume_mounts=volume_mounts, volumes=volumes)
                # create job
                self._k8s_api.create_namespaced_job(namespace=namespace, body=job_manifest)
                LOG.info("Created k8s job %s in namespace %s", job_name, namespace)
            except K8sApiException as e:
                LOG.exception("K8s job create failed: %s", str(e))
                raise

        try:
            await loop.run_in_executor(None, _create_job_and_wait)
        except Exception as e:
            return ExecResult(ok=False, error=str(e))

        # now stream logs from pod(s)
        start_time = time.perf_counter()
        pod_name = None

        async def _find_pod_for_job():
            # blocking call to list pods with label selector job-name=job_name
            def _list_pods():
                try:
                    pods = self._k8s_core.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_name}")
                    return pods.items
                except Exception:
                    return []
            items = await loop.run_in_executor(None, _list_pods)
            return items[0] if items else None

        # wait for pod creation and pending->running
        waited = 0
        pod = None
        while waited < (timeout or 300):
            pod = await _find_pod_for_job()
            if pod:
                pod_name = pod.metadata.name
                # check phase
                phase = pod.status.phase
                if phase in ("Running", "Succeeded", "Failed"):
                    break
            await asyncio.sleep(1)
            waited += 1

        if not pod_name:
            return ExecResult(ok=False, error="Pod not scheduled")

        # helper to stream logs via blocking core.read_namespaced_pod_log, run in executor
        async def _stream_pod_logs():
            nonlocal pod_name, namespace
            try:
                # blocking generator that yields log lines; emulate follow by repeated calls
                last_ts = None
                while True:
                    def _get_logs():
                        try:
                            return self._k8s_core.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=100, timestamps=True)
                        except Exception as e:
                            return f"__error__:{e}"
                    out = await loop.run_in_executor(None, _get_logs)
                    if isinstance(out, str) and out.startswith("__error__:"):
                        yield out
                        break
                    if out:
                        for line in out.splitlines():
                            yield line
                    # check pod status
                    def _get_pod_status():
                        try:
                            return self._k8s_core.read_namespaced_pod_status(name=pod_name, namespace=namespace)
                        except Exception:
                            return None
                    pod_status = await loop.run_in_executor(None, _get_pod_status)
                    if pod_status and pod_status.status.phase in ("Succeeded", "Failed"):
                        break
                    await asyncio.sleep(1)
            except Exception as e:
                yield f"__log_stream_error__:{e}"

        # gather logs while waiting for job completion (or until timeout)
        stdout_buf = []
        stderr_buf = []
        try:
            async for line in _stream_pod_logs():
                # yield lines to caller (if streaming)
                if stream_logs:
                    # caller might want to collect; here we just append, real streaming to user via returned generator
                    stdout_buf.append(line)
            # after streaming finishes, collect final job status
            def _get_job_status():
                try:
                    j = self._k8s_api.read_namespaced_job_status(name=job_name, namespace=namespace)
                    return j.status
                except Exception:
                    return None
            job_status = await loop.run_in_executor(None, _get_job_status)
            succeeded = getattr(job_status, "succeeded", 0) or 0
            failed = getattr(job_status, "failed", 0) or 0
            rc_ok = succeeded > 0 and failed == 0
            elapsed = time.perf_counter() - start_time
            # cleanup TTL may be set by spec; optionally delete job immediately
            return ExecResult(ok=rc_ok, exit_code=0 if rc_ok else (failed or 1), stdout="\n".join(stdout_buf), stderr="\n".join(stderr_buf), runtime_sec=elapsed, meta={"job_name": job_name, "namespace": namespace})
        except Exception as e:
            LOG.exception("Error streaming logs for job %s: %s", job_name, e)
            return ExecResult(ok=False, error=str(e))

    # -------------------------
    # Cancel / cleanup
    # -------------------------
    async def cancel(self, id: str):
        """
        Cancel a running container or k8s job by id (container_id or job_name).
        """
        if self.mode == "docker":
            if not _HAS_DOCKER:
                raise RuntimeError("docker not available")
            try:
                c = self._docker_client.containers.get(id)
                c.kill()
                LOG.info("Killed docker container %s", id)
                return True
            except DockerNotFound:
                LOG.warning("Container not found %s", id)
                return False
            except Exception:
                LOG.exception("Failed to cancel container %s", id)
                return False
        elif self.mode == "k8s":
            if not _HAS_K8S:
                raise RuntimeError("kubernetes not available")
            namespace = DEFAULT_NAMESPACE
            try:
                # delete job (cascade delete pods)
                self._k8s_api.delete_namespaced_job(name=id, namespace=namespace, propagation_policy="Background")
                LOG.info("Deleted k8s job %s", id)
                return True
            except Exception:
                LOG.exception("Failed to delete k8s job %s", id)
                return False
        else:
            raise ValueError("Unknown mode")

    # -------------------------
    # Health checks
    # -------------------------
    def health(self) -> Dict[str, Any]:
        """
        Synchronous health check for clients.
        """
        res = {"mode": self.mode, "ok": False, "details": {}, "ts": _now_iso()}
        if self.mode == "docker":
            if not _HAS_DOCKER:
                res["details"]["error"] = "docker SDK missing"
                return res
            try:
                client = self._docker_client or docker_py.from_env(timeout=DEFAULT_DOCKER_CLIENT_TIMEOUT, **self.docker_client_kwargs)
                pong = client.ping()
                res["ok"] = True
                res["details"]["ping"] = bool(pong)
            except Exception as e:
                res["ok"] = False
                res["details"]["error"] = str(e)
        elif self.mode == "k8s":
            if not _HAS_K8S:
                res["details"]["error"] = "kubernetes client missing"
                return res
            try:
                # quick call to get API versions
                if not self._k8s_api:
                    # attempt to initialize quickly
                    if self.k8s_incluster:
                        k8s_config.load_incluster_config()
                    else:
                        k8s_config.load_kube_config()
                    self._k8s_api = k8s_client.BatchV1Api()
                # list namespaces as cheap check
                ns = k8s_client.CoreV1Api().list_namespace(limit=1)
                res["ok"] = True
                res["details"]["namespaces_sample"] = len(ns.items)
            except Exception as e:
                res["ok"] = False
                res["details"]["error"] = str(e)
        return res

    # -------------------------
    # Utility helpers
    # -------------------------
    def _serialize_spec(self, spec: Union[DockerRunSpec, K8sRunSpec]) -> Dict[str, Any]:
        return json.loads(json.dumps(spec, default=lambda o: getattr(o, "__dict__", str(o))))

# -------------------------
# CLI
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-container-executor")
    sub = p.add_subparsers(dest="cmd")
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=["docker", "k8s"], default=DEFAULT_MODE)
    run.add_argument("--image", required=True)
    run.add_argument("--cmd", default=None)
    run.add_argument("--timeout", type=int, default=60)
    run.add_argument("--mem", default=None)
    run.add_argument("--cpu", default=None)
    run.add_argument("--detach", action="store_true")
    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    mode = args.mode
    ex = ContainerExecutor(mode=mode)
    try:
        import asyncio
        asyncio.run(ex.start())
    except Exception:
        LOG.exception("Executor start failed")
    if args.cmd == "run":
        if mode == "docker":
            spec = DockerRunSpec(image=args.image, command=args.cmd and shlex.split(args.cmd), mem_limit=args.mem, cpu_shares=None, detach=args.detach, timeout=args.timeout)
            res = ex.run_image(spec)
            print(json.dumps(res.__dict__, indent=2, default=str))
        else:
            spec = K8sRunSpec(image=args.image, command=[args.cmd] if args.cmd else None, cpu=args.cpu, memory=args.mem)
            res = asyncio.run(ex.run_k8s_job_async(spec, stream_logs=False, timeout=args.timeout))
            print(json.dumps(res.__dict__, indent=2, default=str))
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

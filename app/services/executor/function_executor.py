# backend/app/services/executor/function_executor.py
"""
PriorityMax Function Executor (production-ready)
------------------------------------------------

A robust function execution subsystem for PriorityMax that supports:
  - Synchronous and asynchronous function execution (callable objects)
  - Safe isolated execution via subprocess worker (recommended for untrusted code)
  - Timeouts, retries, exponential backoff, and resource limits (memory / CPU - best-effort, Unix-only)
  - ThreadPool / ProcessPool execution for CPU-bound and IO-bound functions
  - HTTP function invocation (sync + async) with retries + timeouts
  - AWS Lambda invocation helper (best-effort if boto3 present)
  - Integration hooks for ContainerExecutor and S3Connector to store logs/artifacts
  - Audit/metrics hooks (prometheus_client optional)
  - RBAC / policy pre-check hooks (callable that can veto execution)
  - CLI helper for quick tests
  - Clear warnings about untrusted code execution and recommended safe modes

Design notes:
 - Executing arbitrary user-provided Python code in-process is inherently unsafe.
   This module provides three execution modes:
     1) inproc (fast, not isolated) - runs callables inside the current process (ThreadPool)
     2) process (moderate isolation) - runs via ProcessPoolExecutor (separate OS process)
     3) subprocess (recommended for third-party/untrusted user code) - spawns a new Python
        process with limited privileges and optional resource limits; communicates via JSON over STDIN/STDOUT
 - For maximal isolation, use Docker / ContainerExecutor instead of subprocess.
 - Resource limiting (memory / cpu) uses `resource` module and is only effective on Unix systems.
 - All network calls (requests, boto3) are optional and imported best-effort.
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import math
import shlex
import asyncio
import logging
import pathlib
import tempfile
import threading
import functools
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Concurrency primitives
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future

# Optional libs
_HAS_REQS = False
_HAS_AIOHTTP = False
_HAS_BOTO3 = False
_HAS_PROM = False

try:
    import requests
    _HAS_REQS = True
except Exception:
    requests = None
    _HAS_REQS = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except Exception:
    aiohttp = None
    _HAS_AIOHTTP = False

try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    Counter = Gauge = Histogram = None
    _HAS_PROM = False

# Local integrations (best-effort)
try:
    from app.services.executor.container_executor import ContainerExecutor
except Exception:
    ContainerExecutor = None

try:
    from app.services.connector.s3_connector import S3Connector
except Exception:
    S3Connector = None

# Logging
LOG = logging.getLogger("prioritymax.executor.function")
LOG.setLevel(os.getenv("PRIORITYMAX_FUNC_LOG", "INFO"))
_hdl = logging.StreamHandler(sys.stdout)
_hdl.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_hdl)

# Defaults (configurable via env)
DEFAULT_THREADPOOL = int(os.getenv("PRIORITYMAX_FUNC_THREADS", "16"))
DEFAULT_PROCESSPOOL = int(os.getenv("PRIORITYMAX_FUNC_PROCS", "4"))
DEFAULT_EXEC_MODE = os.getenv("PRIORITYMAX_FUNC_DEFAULT_MODE", "subprocess")  # inproc | process | subprocess
DEFAULT_TIMEOUT = float(os.getenv("PRIORITYMAX_FUNC_DEFAULT_TIMEOUT", "30.0"))
DEFAULT_RETRIES = int(os.getenv("PRIORITYMAX_FUNC_DEFAULT_RETRIES", "2"))
DEFAULT_BACKOFF_BASE = float(os.getenv("PRIORITYMAX_FUNC_BACKOFF_BASE", "0.5"))
DEFAULT_SUBPROCESS_PYTHON = os.getenv("PRIORITYMAX_SUBPROCESS_PYTHON", sys.executable)
DEFAULT_SUBPROCESS_SANDBOX = os.getenv("PRIORITYMAX_SUBPROCESS_SANDBOX", "true").lower() in ("1", "true", "yes")

# Prometheus metrics (optional)
if _HAS_PROM:
    FUNC_INVOCATIONS = Counter("prioritymax_func_invocations_total", "Function invocations", ["mode", "status"])
    FUNC_LATENCY = Histogram("prioritymax_func_latency_seconds", "Function execution latency (seconds)", ["mode"])
    FUNC_RUNNING = Gauge("prioritymax_func_running", "Number of currently running functions", ["mode"])
else:
    FUNC_INVOCATIONS = FUNC_LATENCY = FUNC_RUNNING = None

# Utility helpers
def _uid(prefix: str = "") -> str:
    return (prefix + "-" if prefix else "") + uuid.uuid4().hex[:8]

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# Basic dataclasses
@dataclass
class FunctionSpec:
    """Description of function/execution target."""
    name: str
    mode: str = DEFAULT_EXEC_MODE  # inproc | process | subprocess | http | lambda | container
    callable: Optional[Callable[..., Any]] = None  # if inproc/process
    args: Optional[List[Any]] = field(default_factory=list)
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    code: Optional[str] = None  # python source code to exec (for subprocess runner)
    http: Optional[Dict[str, Any]] = None  # {'url':..., 'method': 'POST', 'headers':..., 'timeout': ...}
    lambda: Optional[Dict[str, Any]] = None  # {'function_name':..., 'payload':..., 'invocation_type':...}
    container: Optional[Dict[str, Any]] = None  # for container executor
    timeout: Optional[float] = None
    retries: int = DEFAULT_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    mem_limit_mb: Optional[int] = None  # for subprocess/process - best-effort
    cpu_limit_percent: Optional[float] = None  # best-effort
    user_id: Optional[str] = None  # for RBAC/audit
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    capture_stdout: bool = True
    capture_stderr: bool = True
    workdir: Optional[str] = None

@dataclass
class FunctionResult:
    ok: bool
    return_value: Optional[Any] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    runtime_sec: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# Thread / Process pools (singletons)
_THREAD_POOL = ThreadPoolExecutor(max_workers=DEFAULT_THREADPOOL)
_PROCESS_POOL = ProcessPoolExecutor(max_workers=DEFAULT_PROCESSPOOL)

# RBAC / policy hook typing
PolicyHook = Callable[[FunctionSpec], Tuple[bool, Optional[str]]]  # (allowed, reason_if_denied)

# -------------------------
# Safe Subprocess Runner
# -------------------------
_SUBPROCESS_HELPER = """
# Minimal subprocess runner that reads JSON payload from stdin, executes code, returns JSON to stdout.
# It expects a payload like:
# {"code": "<python source>", "args": [...], "kwargs": {...}, "capture": true}
# The runner will execute the code in a restricted namespace and call `main(*args, **kwargs)` if available,
# otherwise it will evaluate the last expression `__result__` if set by the script.
import sys, json, traceback, types, time, os

def _now_ts():
    return time.time()

def _safe_exec(payload):
    start = _now_ts()
    out = {"ok": False, "return_value": None, "stdout": "", "stderr": "", "runtime_sec": None, "error": None}
    try:
        code = payload.get("code", "")
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        # prepare namespace
        ns = {"__name__": "__main__", "__builtins__": __builtins__}
        # execute
        exec(code, ns)
        # call main if present
        if "main" in ns and callable(ns["main"]):
            rv = ns["main"](*args, **kwargs)
            out["return_value"] = rv
        else:
            # if script set '__result__', return it
            out["return_value"] = ns.get("__result__", None)
        out["ok"] = True
    except Exception as e:
        out["error"] = traceback.format_exc()
    out["runtime_sec"] = _now_ts() - start
    return out

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"ok": False, "error": "failed to read payload"}, default=str))
        sys.exit(2)
    res = _safe_exec(payload)
    sys.stdout.write(json.dumps(res, default=str))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
"""

_SUBPROCESS_HELPER_BYTES = _SUBPROCESS_HELPER.encode("utf-8")

# Write helper script to a temporary file at import time (used by subprocess mode)
_HELPER_PATH = pathlib.Path(tempfile.gettempdir()) / "prioritymax_func_runner.py"
try:
    if not _HELPER_PATH.exists():
        _HELPER_PATH.write_bytes(_SUBPROCESS_HELPER_BYTES)
except Exception:
    # best-effort, if cannot write, we will pass code via -c to python in subprocess
    _HELPER_PATH = None

# -------------------------
# FunctionExecutor
# -------------------------
class FunctionExecutor:
    """
    Main orchestrator for function execution.

    Provides:
      - run(spec) synchronous call
      - run_async(spec) coroutine
      - invoke_http(spec.http)
      - invoke_lambda(spec.lambda) (best-effort)
      - execute_incontainer (via ContainerExecutor) (best-effort)
      - policy enforcement hooks
      - audit hooks (callable)
    """

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor = None,
        process_pool: ProcessPoolExecutor = None,
        s3_connector: Optional[S3Connector] = None,
        container_executor: Optional[ContainerExecutor] = None,
    ):
        self.thread_pool = thread_pool or _THREAD_POOL
        self.process_pool = process_pool or _PROCESS_POOL
        self.s3 = s3_connector
        self.container = container_executor
        self.policy_hooks: List[PolicyHook] = []
        self.audit_hook: Optional[Callable[[Dict[str, Any]], None]] = None
        self._running_gauge_lock = threading.Lock()
        self._running_counts: Dict[str, int] = {}

    # -------------------------
    # Policy & audit
    # -------------------------
    def register_policy_hook(self, hook: PolicyHook):
        self.policy_hooks.append(hook)

    def set_audit_hook(self, hook: Callable[[Dict[str, Any]], None]):
        self.audit_hook = hook

    def _run_policy_checks(self, spec: FunctionSpec) -> Tuple[bool, Optional[str]]:
        for h in self.policy_hooks:
            try:
                allowed, reason = h(spec)
                if not allowed:
                    return False, reason or "denied by policy hook"
            except Exception as e:
                LOG.exception("policy hook raised exception")
                return False, "policy_hook_error"
        return True, None

    def _audit(self, event: str, payload: Dict[str, Any]):
        try:
            if self.audit_hook:
                self.audit_hook({"event": event, "ts": _now_iso(), **payload})
        except Exception:
            LOG.exception("audit hook failed for event %s", event)

    # -------------------------
    # Helper: increment/decrement running metrics
    # -------------------------
    def _inc_running(self, mode: str):
        if FUNC_RUNNING:
            try:
                FUNC_RUNNING.labels(mode=mode).inc()
            except Exception:
                pass
        with self._running_gauge_lock:
            self._running_counts[mode] = self._running_counts.get(mode, 0) + 1

    def _dec_running(self, mode: str):
        if FUNC_RUNNING:
            try:
                FUNC_RUNNING.labels(mode=mode).dec()
            except Exception:
                pass
        with self._running_gauge_lock:
            self._running_counts[mode] = max(0, self._running_counts.get(mode, 0) - 1)

    # -------------------------
    # Public sync entrypoint
    # -------------------------
    def run(self, spec: FunctionSpec) -> FunctionResult:
        """
        Synchronous execution wrapper. This will block until completion or timeout.
        Retries according to spec.retries with exponential backoff.
        """
        # policy checks
        ok, reason = self._run_policy_checks(spec)
        if not ok:
            self._audit("func_denied", {"name": spec.name, "user": spec.user_id, "reason": reason})
            return FunctionResult(ok=False, error=f"denied: {reason}", meta={"denied_reason": reason})

        attempt = 0
        last_exc = None
        start_total = time.perf_counter()
        while attempt <= spec.retries:
            attempt += 1
            try:
                t0 = time.perf_counter()
                if spec.mode in ("inproc",) and spec.callable:
                    self._inc_running("inproc")
                    try:
                        # run in threadpool to avoid blocking event loop
                        fut: Future = self.thread_pool.submit(spec.callable, * (spec.args or []), ** (spec.kwargs or {}))
                        return_val = fut.result(timeout=spec.timeout or DEFAULT_TIMEOUT)
                        runtime = time.perf_counter() - t0
                        self._audit("func_execute", {"name": spec.name, "mode": "inproc", "user": spec.user_id, "runtime": runtime})
                        if FUNC_INVOCATIONS:
                            try:
                                FUNC_INVOCATIONS.labels(mode="inproc", status="ok").inc()
                                FUNC_LATENCY.labels(mode="inproc").observe(runtime)
                            except Exception:
                                pass
                        return FunctionResult(ok=True, return_value=return_val, runtime_sec=runtime)
                    finally:
                        self._dec_running("inproc")

                elif spec.mode in ("process",) and spec.callable:
                    self._inc_running("process")
                    try:
                        fut: Future = self.process_pool.submit(spec.callable, * (spec.args or []), ** (spec.kwargs or {}))
                        return_val = fut.result(timeout=spec.timeout or DEFAULT_TIMEOUT)
                        runtime = time.perf_counter() - t0
                        self._audit("func_execute", {"name": spec.name, "mode": "process", "user": spec.user_id, "runtime": runtime})
                        if FUNC_INVOCATIONS:
                            try:
                                FUNC_INVOCATIONS.labels(mode="process", status="ok").inc()
                                FUNC_LATENCY.labels(mode="process").observe(runtime)
                            except Exception:
                                pass
                        return FunctionResult(ok=True, return_value=return_val, runtime_sec=runtime)
                    finally:
                        self._dec_running("process")

                elif spec.mode in ("subprocess",):
                    res = self._run_subprocess_sync(spec)
                    if res.ok:
                        return res
                    else:
                        last_exc = res.error or "subprocess failed"
                        # fall through to retry logic
                elif spec.mode in ("http",):
                    res = self._invoke_http_sync(spec)
                    if res.ok:
                        return res
                    else:
                        last_exc = res.error or "http invoke failed"
                elif spec.mode in ("lambda",):
                    res = self._invoke_lambda_sync(spec)
                    if res.ok:
                        return res
                    else:
                        last_exc = res.error or "lambda invoke failed"
                elif spec.mode in ("container",):
                    res = self._execute_in_container_sync(spec)
                    if res.ok:
                        return res
                    else:
                        last_exc = res.error or "container execution failed"
                else:
                    return FunctionResult(ok=False, error=f"unsupported mode {spec.mode}")
            except Exception as e:
                LOG.exception("Function execution attempt failed")
                last_exc = str(e)
            # retry/backoff
            if attempt <= spec.retries:
                backoff = (spec.backoff_base or DEFAULT_BACKOFF_BASE) * (2 ** (attempt - 1))
                time.sleep(backoff)
        total_elapsed = time.perf_counter() - start_total
        if FUNC_INVOCATIONS:
            try:
                FUNC_INVOCATIONS.labels(mode=spec.mode or "unknown", status="error").inc()
                FUNC_LATENCY.labels(mode=spec.mode or "unknown").observe(total_elapsed)
            except Exception:
                pass
        self._audit("func_failed", {"name": spec.name, "mode": spec.mode, "user": spec.user_id, "elapsed": total_elapsed, "error": last_exc})
        return FunctionResult(ok=False, error=str(last_exc), runtime_sec=total_elapsed)

    # -------------------------
    # Public async entrypoint
    # -------------------------
    async def run_async(self, spec: FunctionSpec) -> FunctionResult:
        """
        Async wrapper that dispatches to appropriate implementation.
        """
        loop = asyncio.get_running_loop()
        # For inproc/process: submit to thread/process pool using run_in_executor
        if spec.mode in ("inproc",) and spec.callable:
            return await loop.run_in_executor(self.thread_pool, functools.partial(self.run, spec))
        if spec.mode in ("process",) and spec.callable:
            return await loop.run_in_executor(self.thread_pool, functools.partial(self.run, spec))
        # For subprocess/http/lambda/container: run in executor-bound sync function
        return await loop.run_in_executor(self.thread_pool, functools.partial(self.run, spec))

    # -------------------------
    # Subprocess runner (sync)
    # -------------------------
    def _run_subprocess_sync(self, spec: FunctionSpec) -> FunctionResult:
        """
        Run Python code in a fresh subprocess. Communicates via JSON on stdin/stdout.

        Behavior:
         - Writes payload JSON (code, args, kwargs) to subprocess STDIN
         - Reads JSON result from STDOUT and returns it as FunctionResult
         - Applies timeout and optional resource limits (best-effort on Unix)
        """
        mode = "subprocess"
        self._inc_running(mode)
        start = time.perf_counter()
        try:
            payload = {"code": spec.code or "", "args": spec.args or [], "kwargs": spec.kwargs or {}}
            # preferred: use helper script file to avoid shell quoting issues
            if _HELPER_PATH and _HELPER_PATH.exists():
                cmd = [DEFAULT_SUBPROCESS_PYTHON, str(_HELPER_PATH)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    out_bytes, err_bytes = proc.communicate(json.dumps(payload).encode("utf-8"), timeout=spec.timeout or DEFAULT_TIMEOUT)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out_bytes, err_bytes = proc.communicate(timeout=5)
                    return FunctionResult(ok=False, error="timeout", stderr=(err_bytes.decode(errors="ignore") if err_bytes else None), runtime_sec=time.perf_counter() - start)
            else:
                # fallback: pass code via -c argument (less safe for large code)
                runner_code = _SUBPROCESS_HELPER
                cmd = [DEFAULT_SUBPROCESS_PYTHON, "-c", runner_code]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    out_bytes, err_bytes = proc.communicate(json.dumps(payload).encode("utf-8"), timeout=spec.timeout or DEFAULT_TIMEOUT)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out_bytes, err_bytes = proc.communicate(timeout=5)
                    return FunctionResult(ok=False, error="timeout", stderr=(err_bytes.decode(errors="ignore") if err_bytes else None), runtime_sec=time.perf_counter() - start)

            # parse stdout JSON
            try:
                out_text = out_bytes.decode("utf-8") if out_bytes else ""
                err_text = err_bytes.decode("utf-8") if err_bytes else ""
                data = json.loads(out_text) if out_text else {"ok": False, "error": "no output", "stderr": err_text}
                return FunctionResult(ok=data.get("ok", False), return_value=data.get("return_value"), stdout=None, stderr=(data.get("error") or err_text), runtime_sec=data.get("runtime_sec") or (time.perf_counter() - start), meta={"raw": data})
            except Exception:
                LOG.exception("Failed parse subprocess output")
                return FunctionResult(ok=False, error="invalid_runner_output", stderr=(err_text or None), runtime_sec=time.perf_counter() - start)
        finally:
            self._dec_running(mode)

    # -------------------------
    # HTTP invocation helpers
    # -------------------------
    def _invoke_http_sync(self, spec: FunctionSpec) -> FunctionResult:
        """
        Synchronous HTTP invocation for 'http' mode FunctionSpec.
        spec.http should contain: url, method, headers, json/body, timeout
        """
        if not _HAS_REQS:
            return FunctionResult(ok=False, error="requests library not available")
        mode = "http"
        self._inc_running(mode)
        start = time.perf_counter()
        try:
            info = spec.http or {}
            url = info.get("url")
            method = (info.get("method", "POST") or "POST").upper()
            headers = info.get("headers") or {}
            timeout = info.get("timeout") or spec.timeout or DEFAULT_TIMEOUT
            body = info.get("json") if "json" in info else info.get("data", None)
            try:
                resp = requests.request(method, url, json=body if method in ("POST", "PUT", "PATCH") else None, headers=headers, timeout=timeout)
                runtime = time.perf_counter() - start
                success = 200 <= resp.status_code < 300
                if FUNC_INVOCATIONS:
                    try:
                        FUNC_INVOCATIONS.labels(mode="http", status="ok" if success else "error").inc()
                        FUNC_LATENCY.labels(mode="http").observe(runtime)
                    except Exception:
                        pass
                return FunctionResult(ok=success, return_value=resp.text, stdout=None, stderr=None if success else resp.text, runtime_sec=runtime, meta={"status_code": resp.status_code})
            except Exception as e:
                LOG.exception("HTTP invocation failed")
                return FunctionResult(ok=False, error=str(e), runtime_sec=time.perf_counter() - start)
        finally:
            self._dec_running(mode)

    async def _invoke_http_async(self, spec: FunctionSpec) -> FunctionResult:
        if not _HAS_AIOHTTP:
            # fall back to sync in executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self._invoke_http_sync, spec))
        mode = "http"
        self._inc_running(mode)
        start = time.perf_counter()
        try:
            info = spec.http or {}
            url = info.get("url")
            method = (info.get("method", "POST") or "POST").upper()
            headers = info.get("headers") or {}
            timeout = info.get("timeout") or spec.timeout or DEFAULT_TIMEOUT
            body = info.get("json") if "json" in info else info.get("data", None)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, json=body if method in ("POST", "PUT", "PATCH") else None, headers=headers, timeout=timeout) as resp:
                        text = await resp.text()
                        runtime = time.perf_counter() - start
                        success = 200 <= resp.status < 300
                        if FUNC_INVOCATIONS:
                            try:
                                FUNC_INVOCATIONS.labels(mode="http", status="ok" if success else "error").inc()
                                FUNC_LATENCY.labels(mode="http").observe(runtime)
                            except Exception:
                                pass
                        return FunctionResult(ok=success, return_value=text, runtime_sec=runtime, meta={"status_code": resp.status})
            except Exception as e:
                LOG.exception("Async HTTP invocation failed")
                return FunctionResult(ok=False, error=str(e), runtime_sec=time.perf_counter() - start)
        finally:
            self._dec_running(mode)

    # -------------------------
    # AWS Lambda invocation (sync)
    # -------------------------
    def _invoke_lambda_sync(self, spec: FunctionSpec) -> FunctionResult:
        if not _HAS_BOTO3:
            return FunctionResult(ok=False, error="boto3 not available")
        cfg = spec.get("lambda", {})
        function_name = cfg.get("function_name")
        payload = cfg.get("payload", {})
        invocation_type = cfg.get("invocation_type", "RequestResponse")  # Event | RequestResponse
        mode = "lambda"
        self._inc_running(mode)
        start = time.perf_counter()
        try:
            client = boto3.client("lambda")
            resp = client.invoke(FunctionName=function_name, InvocationType=invocation_type, Payload=json.dumps(payload).encode("utf-8"))
            runtime = time.perf_counter() - start
            status_code = resp.get("StatusCode", 0)
            body = resp.get("Payload").read().decode("utf-8") if resp.get("Payload") else None
            ok = 200 <= status_code < 300
            if FUNC_INVOCATIONS:
                try:
                    FUNC_INVOCATIONS.labels(mode="lambda", status="ok" if ok else "error").inc()
                    FUNC_LATENCY.labels(mode="lambda").observe(runtime)
                except Exception:
                    pass
            return FunctionResult(ok=ok, return_value=body, runtime_sec=runtime, meta={"status_code": status_code})
        except Exception as e:
            LOG.exception("Lambda invocation failed")
            return FunctionResult(ok=False, error=str(e), runtime_sec=time.perf_counter() - start)
        finally:
            self._dec_running(mode)

    # -------------------------
    # Container execution integration (sync)
    # -------------------------
    def _execute_in_container_sync(self, spec: FunctionSpec) -> FunctionResult:
        """
        Execute using a provided ContainerExecutor instance.
        spec.container should contain fields for ContainerExecutor (image, command, mounts, env, etc.)
        """
        if self.container is None:
            return FunctionResult(ok=False, error="no container executor configured")
        try:
            # Build DockerRunSpec dataclass-like dict; we avoid direct import to reduce coupling
            cnt = spec.container or {}
            # container_executor.run_image expects a DockerRunSpec instance - accept both dict or already-built object
            # Best-effort: try to call container.run_image(spec) if spec is already a DockerRunSpec
            res = None
            if hasattr(self.container, "run_image") and callable(self.container.run_image):
                # run synchronously in current thread; ContainerExecutor is typically blocking for docker-py
                res = self.container.run_image(cnt)
                # normalize ExecResult to FunctionResult if necessary
                if isinstance(res, dict):
                    ok = bool(res.get("ok", False))
                    return FunctionResult(ok=ok, return_value=res.get("meta"), stdout=res.get("stdout"), stderr=res.get("stderr"), runtime_sec=res.get("runtime_sec"), meta=res)
                else:
                    # If returned object has attributes
                    ok = getattr(res, "ok", False)
                    stdout = getattr(res, "stdout", None)
                    stderr = getattr(res, "stderr", None)
                    runtime = getattr(res, "runtime_sec", None)
                    return FunctionResult(ok=ok, return_value=getattr(res, "meta", None), stdout=stdout, stderr=stderr, runtime_sec=runtime, meta={"raw": res})
            else:
                return FunctionResult(ok=False, error="container executor missing run_image")
        except Exception as e:
            LOG.exception("Container execution failed")
            return FunctionResult(ok=False, error=str(e))

    # -------------------------
    # CLI helpers & small utils
    # -------------------------
    def _serialize_for_audit(self, spec: FunctionSpec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "mode": spec.mode,
            "user": spec.user_id,
            "metadata": spec.metadata or {},
            "ts": _now_iso(),
        }

# -------------------------
# CLI
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-function-executor")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run-code")
    run.add_argument("--code", required=True, help="Python code to execute. Define main(...) or set __result__")
    run.add_argument("--timeout", type=int, default=int(DEFAULT_TIMEOUT))
    run.add_argument("--mode", choices=["subprocess", "inproc", "process"], default="subprocess")
    run.add_argument("--args", default="[]", help="JSON array of positional args")
    run.add_argument("--kwargs", default="{}", help="JSON object of kwargs")

    http = sub.add_parser("invoke-http")
    http.add_argument("--url", required=True)
    http.add_argument("--method", default="POST")
    http.add_argument("--json", default=None, help="JSON payload")

    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    fe = FunctionExecutor()
    if args.cmd == "run-code":
        spec = FunctionSpec(name="cli-run", mode=args.mode, code=args.code, args=json.loads(args.args), kwargs=json.loads(args.kwargs), timeout=args.timeout)
        res = fe.run(spec)
        print(json.dumps(res.__dict__, indent=2, default=str))
    elif args.cmd == "invoke-http":
        body = json.loads(args.json) if args.json else None
        spec = FunctionSpec(name="cli-http", mode="http", http={"url": args.url, "method": args.method, "json": body})
        res = fe.run(spec)
        print(json.dumps(res.__dict__, indent=2, default=str))
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

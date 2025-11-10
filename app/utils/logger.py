# backend/app/utils/logger.py
"""
PriorityMax Logger Utilities (production-grade)
-----------------------------------------------
Provides a comprehensive logging system used across PriorityMax services.

Features:
 - JSONFormatter and human-friendly formatter
 - Console + file handlers with atomic rotation & retention
 - Asynchronous logging handler (queue-based) to avoid blocking critical paths
 - Optional integration with OpenTelemetry trace/span IDs
 - Optional Sentry integration (best-effort)
 - Prometheus metrics hooks for log counts (optional)
 - FastAPI request middleware that injects request_id, user_id into log context
 - Contextual logger adapter for structured + contextual logging
 - Helpers to configure logging from environment variables / config
 - CLI self-check for verifying logging pipeline

Usage:
    from app.utils.logger import configure_logging, get_logger
    configure_logging(app_name="prioritymax", log_dir="/var/log/prioritymax")
    log = get_logger(__name__)
    log.info("hello", extra={"task_id": "123"})
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import atexit
import queue
import logging
import socket
import pathlib
import threading
import traceback
from typing import Any, Dict, Optional, Iterable, Tuple

# Optional deps
try:
    from opentelemetry.trace import get_current_span, SpanContext
    _HAS_OTEL = True
except Exception:
    _HAS_OTEL = False

try:
    import sentry_sdk
    _HAS_SENTRY = True
except Exception:
    sentry_sdk = None
    _HAS_SENTRY = False

try:
    from prometheus_client import Counter
    _HAS_PROM = True
except Exception:
    Counter = None
    _HAS_PROM = False

# Local imports (best-effort)
try:
    from app.utils.common import json_dumps, now_iso, get_version_info, set_context, get_context
except Exception:
    # Provide fallback implementations if import fails
    def json_dumps(obj):
        try:
            return json.dumps(obj)
        except Exception:
            return str(obj)
    def now_iso():
        import datetime
        return datetime.datetime.utcnow().isoformat() + "Z"
    def get_version_info():
        return {"version": "unknown"}
    def set_context(k, v): pass
    def get_context(k, default=None): return default

# -------------------------
# Constants & Env defaults
# -------------------------
DEFAULT_LOG_LEVEL = os.getenv("PRIORITYMAX_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_DIR = os.getenv("PRIORITYMAX_LOG_DIR", str(pathlib.Path.cwd() / "logs"))
DEFAULT_LOG_FILE = os.getenv("PRIORITYMAX_LOG_FILE", "prioritymax.log")
DEFAULT_MAX_BYTES = int(os.getenv("PRIORITYMAX_LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB
DEFAULT_BACKUP_COUNT = int(os.getenv("PRIORITYMAX_LOG_BACKUPS", "7"))
ASYNC_QUEUE_MAX = int(os.getenv("PRIORITYMAX_LOG_ASYNC_QSIZE", "10000"))
PROM_LOG_COUNTER_NAME = os.getenv("PRIORITYMAX_PROM_LOG_COUNTER", "prioritymax_logs_total")

# Prometheus metric
_LOG_COUNTER = None
if _HAS_PROM:
    try:
        _LOG_COUNTER = Counter(PROM_LOG_COUNTER_NAME, "Count of log records emitted", ["level"])
    except Exception:
        _LOG_COUNTER = None

# -------------------------
# Utilities
# -------------------------
def _make_request_id() -> str:
    return uuid.uuid4().hex

def _get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-host"

# -------------------------
# Formatters
# -------------------------
class JSONFormatter(logging.Formatter):
    """
    JSON formatter that attaches standard fields:
      - timestamp, level, logger, message, module, file, line
      - service, hostname, pid, version
      - optional: trace_id, span_id, request_id, user_id, extra
    """
    def __init__(self, service_name: str = "prioritymax", extra_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.service = service_name
        self.extra_fields = extra_fields or {}
        self.hostname = _get_hostname()
        self.pid = os.getpid()
        self.version = get_version_info().get("version", "unknown")

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": getattr(record, "module", None),
            "file": getattr(record, "pathname", None),
            "line": getattr(record, "lineno", None),
            "service": self.service,
            "hostname": self.hostname,
            "pid": self.pid,
            "version": self.version,
        }
        # include extra dict if present
        extra = {}
        for k, v in record.__dict__.items():
            if k in ("args", "msg", "levelname", "levelno", "name", "pathname", "lineno", "exc_info", "exc_text", "stack_info", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process"):
                continue
            extra[k] = v
        if extra:
            payload["extra"] = extra
        # include exception info
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # tracing context if available
        try:
            if _HAS_OTEL:
                span = get_current_span()
                if span is not None:
                    ctx = span.get_span_context()
                    trace_id = getattr(ctx, "trace_id", None)
                    span_id = getattr(ctx, "span_id", None)
                    if trace_id:
                        payload["trace_id"] = format(trace_id, '032x')
                    if span_id:
                        payload["span_id"] = format(span_id, '016x')
        except Exception:
            pass
        # merge service-level extras
        payload.update(self.extra_fields)
        return json_dumps(payload)

class HumanFormatter(logging.Formatter):
    """
    Human-friendly formatter, with optional colorization. Keeps some structured info.
    """
    def __init__(self, service_name: str = "prioritymax", color: bool = True):
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.service = service_name
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        # append contextual items if present
        extras = []
        req_id = getattr(record, "request_id", None) or get_context("request_id", None)
        if req_id:
            extras.append(f"req_id={req_id}")
        user = getattr(record, "user_id", None)
        if user:
            extras.append(f"user={user}")
        if extras:
            base = f"{base} | {' '.join(extras)}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base

# -------------------------
# Async Handler (Queue-based)
# -------------------------
class AsyncQueueHandler(logging.Handler):
    """
    Logging handler that enqueues records for an async worker thread to process.
    This avoids blocking the main thread on slow I/O (disk, network).
    """
    _worker_thread: Optional[threading.Thread] = None
    _queue: Optional[queue.Queue] = None
    _stop_event = threading.Event()
    _initialized = False
    _handlers_installed: Iterable[logging.Handler] = ()

    @classmethod
    def initialize(cls, handlers: Iterable[logging.Handler], max_qsize: int = ASYNC_QUEUE_MAX):
        if cls._initialized:
            return
        cls._queue = queue.Queue(maxsize=max_qsize)
        cls._handlers_installed = tuple(handlers)
        cls._stop_event.clear()
        cls._worker_thread = threading.Thread(target=cls._worker_loop, name="PriorityMax-LogWorker", daemon=True)
        cls._worker_thread.start()
        atexit.register(cls.shutdown)
        cls._initialized = True

    @classmethod
    def _worker_loop(cls):
        q = cls._queue
        if q is None:
            return
        while not cls._stop_event.is_set():
            try:
                record = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                for handler in cls._handlers_installed:
                    try:
                        handler.handle(record)
                    except Exception:
                        # we avoid crashing: swallow handler errors but log to stderr
                        try:
                            sys.stderr.write("log handler error: " + traceback.format_exc())
                        except Exception:
                            pass
                # Prometheus counter
                try:
                    if _LOG_COUNTER:
                        _LOG_COUNTER.labels(level=record.levelname.lower()).inc()
                except Exception:
                    pass
            finally:
                q.task_done()
        # flush remaining
        while not q.empty():
            record = q.get_nowait()
            for handler in cls._handlers_installed:
                try:
                    handler.handle(record)
                except Exception:
                    pass
            q.task_done()

    @classmethod
    def shutdown(cls):
        cls._stop_event.set()
        if cls._worker_thread:
            cls._worker_thread.join(timeout=2.0)
        cls._initialized = False

    def emit(self, record: logging.LogRecord):
        q = self.__class__._queue
        if not q:
            # fallback: handle synchronously on stdout
            try:
                for handler in self.__class__._handlers_installed:
                    handler.handle(record)
            except Exception:
                pass
            return
        try:
            q.put_nowait(record)
        except queue.Full:
            # If queue full, drop the message or handle fallback
            try:
                # increment drop metric if desired
                if _LOG_COUNTER:
                    _LOG_COUNTER.labels(level="dropped").inc()
            except Exception:
                pass
            # Optionally block briefly to avoid dropping critical logs
            try:
                q.put(record, timeout=0.5)
            except Exception:
                # last resort: write to stderr
                try:
                    sys.stderr.write("log queue full - dropping log: " + record.getMessage() + "\n")
                except Exception:
                    pass

# -------------------------
# File handler with rotation & retention
# -------------------------
class RotatingFileHandler(logging.handlers.RotatingFileHandler if hasattr(logging.handlers, "RotatingFileHandler") else logging.FileHandler):
    """
    Thin wrapper in case we want custom behavior later. Uses built-in RotatingFileHandler.
    """
    def __init__(self, filename: str, maxBytes: int = DEFAULT_MAX_BYTES, backupCount: int = DEFAULT_BACKUP_COUNT, encoding: str = "utf-8"):
        # If RotatingFileHandler not available, fall back to standard FileHandler
        if hasattr(logging.handlers, "RotatingFileHandler"):
            super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding)
        else:
            # fallback: ignore rotation params
            super().__init__(filename, encoding=encoding)

# -------------------------
# FastAPI middleware integration (request context)
# -------------------------
try:
    from fastapi import Request
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware
    _HAS_FASTAPI = True
except Exception:
    _HAS_FASTAPI = False
    Request = None
    FastAPI = None
    BaseHTTPMiddleware = object

class RequestIdFilter(logging.Filter):
    """
    Attach request_id and other context to log records (pulled from get_context()).
    """
    def filter(self, record: logging.LogRecord) -> bool:
        # inject common contextual fields
        record.request_id = get_context("request_id", None)
        record.user_id = get_context("user_id", None)
        return True

if _HAS_FASTAPI:
    class RequestContextMiddleware(BaseHTTPMiddleware):
        """
        FastAPI middleware that injects request id, user id, and populates contextvars.
        Also records request/response timing at INFO level.
        """
        def __init__(self, app: FastAPI, header_name: str = "X-Request-ID", service_name: str = "prioritymax"):
            super().__init__(app)
            self.header_name = header_name
            self.service_name = service_name

        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            req_id = request.headers.get(self.header_name) or _make_request_id()
            set_context("request_id", req_id)
            # optionally extract user id from headers/auth - best-effort
            try:
                uid = None
                if "authorization" in request.headers:
                    uid = request.headers.get("authorization")  # placeholder; real extraction needed
                set_context("user_id", uid)
            except Exception:
                pass
            # proceed
            try:
                response = await call_next(request)
            except Exception as e:
                # error path; log and re-raise
                logging.getLogger("uvicorn.error").exception("Unhandled exception for request %s", req_id)
                raise
            elapsed = time.perf_counter() - start
            # attach timing to response headers optionally
            try:
                response.headers["X-Request-Duration"] = f"{elapsed:.6f}"
            except Exception:
                pass
            # structured log for request lifecycle
            logging.getLogger("prioritymax.request").info("http_request", extra={
                "request_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_sec": elapsed
            })
            # clear context to prevent leak between requests in same thread
            set_context("request_id", None)
            set_context("user_id", None)
            return response

# -------------------------
# Configure logging
# -------------------------
_DEFAULT_CONFIGURED = False
_LOCK = threading.Lock()

def _ensure_log_dir(dirpath: str):
    try:
        p = pathlib.Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # ignore failure; filesystem issues shouldn't break logging config
        pass

def configure_logging(
    app_name: str = "prioritymax",
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    json: bool = True,
    async_worker: bool = True,
    sentry_dsn: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None
):
    """
    Configure root logging for PriorityMax services.

    Parameters:
      - app_name: service name inserted into logs
      - log_dir: directory to write log files
      - level: logging level (e.g. "INFO")
      - console: enable console handler
      - file: enable file handler
      - json: if True use JSONFormatter for file/console (file uses JSON by default)
      - async_worker: if True use AsyncQueueHandler that delegates to installed handlers
      - sentry_dsn: optional DSN string to initialize Sentry SDK
    """
    global _DEFAULT_CONFIGURED
    with _LOCK:
        if _DEFAULT_CONFIGURED:
            return
        level = (level or DEFAULT_LOG_LEVEL).upper()
        log_dir = log_dir or DEFAULT_LOG_DIR
        _ensure_log_dir(log_dir)

        root = logging.getLogger()
        root.setLevel(getattr(logging, level, logging.INFO))

        handlers = []

        # Console handler
        if console:
            ch = logging.StreamHandler(stream=sys.stdout)
            if json:
                ch.setFormatter(JSONFormatter(service_name=app_name, extra_fields=extra_fields))
            else:
                ch.setFormatter(HumanFormatter(service_name=app_name))
            ch.setLevel(getattr(logging, level, logging.INFO))
            handlers.append(ch)

        # File handler
        if file:
            logfile = os.path.join(log_dir, DEFAULT_LOG_FILE)
            fh = RotatingFileHandler(logfile, maxBytes=DEFAULT_MAX_BYTES, backupCount=DEFAULT_BACKUP_COUNT)
            fh.setLevel(getattr(logging, level, logging.INFO))
            # files often are better as JSON for log shipping
            fh.setFormatter(JSONFormatter(service_name=app_name, extra_fields=extra_fields))
            handlers.append(fh)

        # If async_worker requested: create AsyncQueueHandler that forwards to handlers
        if async_worker:
            try:
                AsyncQueueHandler.initialize(handlers)
                qh = AsyncQueueHandler()
                # add a small local fallback console handler if queue fails
                qh.setLevel(getattr(logging, level, logging.INFO))
                root.addHandler(qh)
            except Exception:
                # if we fail to initialize queue, add handlers directly
                for h in handlers:
                    root.addHandler(h)
        else:
            for h in handlers:
                root.addHandler(h)

        # Add request id filter to root so all handlers get request_id added
        root.addFilter(RequestIdFilter())

        # Sentry integration if DSN provided and present
        if sentry_dsn or os.getenv("SENTRY_DSN"):
            dsn = sentry_dsn or os.getenv("SENTRY_DSN")
            try:
                if _HAS_SENTRY:
                    sentry_sdk.init(dsn)
                    logging.getLogger("sentry").info("Sentry initialized")
                else:
                    logging.getLogger("sentry").warning("sentry_sdk not installed; skipping Sentry init")
            except Exception:
                logging.getLogger("sentry").exception("failed to initialize Sentry")

        _DEFAULT_CONFIGURED = True

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a standard logger configured for PriorityMax. call configure_logging first.
    """
    if name is None:
        name = "prioritymax"
    return logging.getLogger(name)

# -------------------------
# Structured Logger Adapter
# -------------------------
class StructuredLoggerAdapter(logging.LoggerAdapter):
    """
    Attach structured context to logs conveniently. Works well with JSONFormatter.
    Usage:
        logger = StructuredLoggerAdapter(get_logger(__name__), {"svc":"prioritymax"})
        logger.info("hello", extra={"task_id": "123"})
    """
    def process(self, msg, kwargs):
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        # merge adapter-provided context
        if isinstance(self.extra, dict):
            kwargs["extra"].update(self.extra)
        # also attach request context if available
        req_id = get_context("request_id", None)
        if req_id:
            kwargs["extra"].setdefault("request_id", req_id)
        user_id = get_context("user_id", None)
        if user_id:
            kwargs["extra"].setdefault("user_id", user_id)
        return msg, kwargs

# -------------------------
# Self-check CLI
# -------------------------
def _self_check():
    import tempfile
    print("PriorityMax logger self-check")
    tmp = tempfile.mkdtemp(prefix="prioritymax-logs-")
    print("tmp log dir:", tmp)
    configure_logging(app_name="prioritymax-test", log_dir=tmp, console=True, file=True, json=False, async_worker=True)
    log = get_logger("prioritymax.test")
    lad = StructuredLoggerAdapter(log, {"component": "selfcheck"})
    lad.info("this is a test info", extra={"sample": True})
    lad.warning("this is a test warning", extra={"sample": True})
    try:
        1/0
    except Exception:
        lad.exception("this is a test exception")
    # allow async worker to flush
    time.sleep(0.5)
    print("Logs written to:", tmp)
    print("Self-check complete")

# -------------------------
# Module CLI
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-logger")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selfcheck")
    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    if args.cmd == "selfcheck":
        _self_check()
    else:
        parser.print_help()

# -------------------------
# Exports
# -------------------------
__all__ = [
    "configure_logging",
    "get_logger",
    "JSONFormatter",
    "HumanFormatter",
    "AsyncQueueHandler",
    "RotatingFileHandler",
    "RequestContextMiddleware",
    "StructuredLoggerAdapter",
    "RequestIdFilter",
]

if __name__ == "__main__":
    main_cli()

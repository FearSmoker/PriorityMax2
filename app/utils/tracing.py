# backend/app/utils/tracing.py
"""
PriorityMax Tracing Utilities (production-grade)
------------------------------------------------

Provides OpenTelemetry-based tracing helpers, exporters (Jaeger/OTLP/Zipkin),
FastAPI middleware, async/span decorators, context propagation helpers,
and utilities to initialize and gracefully shut down tracing in PriorityMax
services (API, workers, RL agent, autoscaler, etc).

Design goals:
 - Best-effort imports: functions still work even if optional dependencies are missing.
 - Easy init via env vars or programmatic params.
 - FastAPI middleware that injects trace/span ids into logs and responses.
 - Decorators for sync/async functions to open named spans automatically.
 - Helpers for injecting/extracting trace context into HTTP headers.
 - Graceful shutdown hooks for flushing exporters.

Environment variables supported (defaults shown):
 - PRIORITYMAX_TRACING_ENABLED=true/false
 - PRIORITYMAX_TRACING_EXPORTER=otlp|jaeger|zipkin|stdout (default: otlp if OTLP lib present, else stdout)
 - PRIORITYMAX_TRACING_OTLP_ENDPOINT (e.g., http://otel-collector:4317)
 - PRIORITYMAX_TRACING_JAEGER_HOST (e.g., jaeger-agent)
 - PRIORITYMAX_TRACING_JAEGER_PORT (6831)
 - PRIORITYMAX_TRACING_SERVICE_NAME=prioritymax
 - PRIORITYMAX_TRACING_SAMPLER=always|probabilistic|parentbased (default: parentbased)
 - PRIORITYMAX_TRACING_PROBABILITY=0.1
 - PRIORITYMAX_TRACING_RESOURCE_ATTRS=json string of additional attributes

Usage:
    from app.utils.tracing import init_tracing, get_tracer, trace_span, FastAPITracingMiddleware
    init_tracing()
    tracer = get_tracer()
    @trace_span("do_work")
    def do_work(...): ...

Note:
 - This module prefers OpenTelemetry; if unavailable, it falls back to
   no-op functions that keep the rest of the app functional.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import functools
import asyncio
import typing
from typing import Any, Callable, Dict, Optional, Iterable, Awaitable

# Logging
LOG = logging.getLogger("prioritymax.utils.tracing")
LOG.setLevel(os.getenv("PRIORITYMAX_TRACING_LOG", "INFO"))
if not LOG.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(ch)

# Optional OpenTelemetry imports (best-effort)
_HAS_OTEL = False
_HAS_OTEL_EXPORTER_OTLP = False
_HAS_OTEL_EXPORTER_JAEGER = False
_HAS_OTEL_EXPORTER_ZIPKIN = False
_HAS_FASTAPI = False

try:
    from opentelemetry import trace, context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider, sampling
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    _HAS_OTEL = True
except Exception:
    trace = None
    context = None
    Resource = None
    TracerProvider = None
    sampling = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None

# OTLP exporter
if _HAS_OTEL:
    try:
        # newer opentelemetry exporters live in opentelemetry-exporter-otlp
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        _HAS_OTEL_EXPORTER_OTLP = True
    except Exception:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # http variant
            _HAS_OTEL_EXPORTER_OTLP = True
        except Exception:
            OTLPSpanExporter = None
            _HAS_OTEL_EXPORTER_OTLP = False

# Jaeger exporter
if _HAS_OTEL:
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        _HAS_OTEL_EXPORTER_JAEGER = True
    except Exception:
        JaegerExporter = None
        _HAS_OTEL_EXPORTER_JAEGER = False

# Zipkin exporter
if _HAS_OTEL:
    try:
        from opentelemetry.exporter.zipkin.json import ZipkinExporter
        _HAS_OTEL_EXPORTER_ZIPKIN = True
    except Exception:
        ZipkinExporter = None
        _HAS_OTEL_EXPORTER_ZIPKIN = False

# FastAPI middleware / Starlette integration
try:
    from fastapi import Request, FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware
    _HAS_FASTAPI = True
except Exception:
    Request = None
    FastAPI = None
    BaseHTTPMiddleware = object

# WSGI/ASGI instrumentations optional (requests, aiohttp)
try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except Exception:
    _HAS_AIOHTTP = False

# -----------------------------------------------------------------------------
# No-op fallbacks when OpenTelemetry is missing
# -----------------------------------------------------------------------------
if not _HAS_OTEL:
    LOG.warning("OpenTelemetry not available; tracing will be no-op")

    class _NoopSpan:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        def set_attribute(self, *a, **k): pass
        def set_status(self, *a, **k): pass
        def add_event(self, *a, **k): pass

    def init_tracing(*args, **kwargs):
        LOG.info("Tracing init called, but OpenTelemetry not installed - no-op")

    def get_tracer(name: Optional[str] = None):
        class NoopTracer:
            def start_as_current_span(self, name, context=None):
                return _NoopSpan()
            def start_span(self, name, context=None):
                return _NoopSpan()
        return NoopTracer()

    def trace_span(name: str):
        def _decor(fn):
            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def _wrapped(*a, **kw):
                    return await fn(*a, **kw)
                return _wrapped
            else:
                @functools.wraps(fn)
                def _wrapped(*a, **kw):
                    return fn(*a, **kw)
                return _wrapped
        return _decor

    class FastAPITracingMiddleware:
        def __init__(self, app: FastAPI, service_name: Optional[str] = None): pass

    def inject_trace_context(headers: Dict[str, str]):
        return headers

    def extract_trace_context(headers: Dict[str, str]):
        return {}

    def shutdown_tracing(timeout: float = 5.0):
        return None

else:
    # -----------------------------------------------------------------------------
    # Real OpenTelemetry implementation
    # -----------------------------------------------------------------------------
    from opentelemetry.trace import Tracer as OtelTracer  # type: ignore

    _TRACER_PROVIDER: Optional[TracerProvider] = None
    _INSTALLED_EXPORTERS: list = []
    _TRACER = None

    def _parse_resource_attributes(s: Optional[str]) -> Dict[str, str]:
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            LOG.warning("Failed to parse resource attrs: %s", s)
            return {}

    def _choose_sampler(name: str, probability: float):
        name = (name or "").lower()
        if name == "always":
            return sampling.ALWAYS_ON
        if name in ("probabilistic", "traceidratio", "parentbasedprobabilistic"):
            try:
                return sampling.TraceIdRatioBased(probability)
            except Exception:
                LOG.warning("TraceIdRatioBased not available, falling back to parentbased")
                return sampling.ParentBased(sampling.TraceIdRatioBased(probability))
        # default: ParentBased(TraceIdRatioBased(prob))
        try:
            return sampling.ParentBased(sampling.TraceIdRatioBased(probability))
        except Exception:
            return sampling.ParentBased(sampling.ALWAYS_ON)

    def init_tracing(
        service_name: Optional[str] = None,
        exporter: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        jaeger_host: Optional[str] = None,
        jaeger_port: Optional[int] = None,
        zipkin_endpoint: Optional[str] = None,
        sampler: Optional[str] = None,
        probability: Optional[float] = None,
        resource_attrs: Optional[Dict[str, str]] = None,
        console: bool = False,
    ):
        """
        Initialize the OpenTelemetry tracer provider and exporter(s).

        All parameters are optional; env vars are consulted as defaults.

        Typical env usage:
          PRIORITYMAX_TRACING_ENABLED=true
          PRIORITYMAX_TRACING_EXPORTER=jaeger
          PRIORITYMAX_TRACING_JAEGER_HOST=jaeger-agent
          PRIORITYMAX_TRACING_JAEGER_PORT=6831
          PRIORITYMAX_TRACING_SERVICE_NAME=prioritymax
        """
        global _TRACER_PROVIDER, _INSTALLED_EXPORTERS, _TRACER

        enabled = os.getenv("PRIORITYMAX_TRACING_ENABLED", "true").lower() in ("1", "true", "yes")
        if not enabled:
            LOG.info("Tracing disabled via env")
            return

        service_name = service_name or os.getenv("PRIORITYMAX_TRACING_SERVICE_NAME", "prioritymax")
        exporter = exporter or os.getenv("PRIORITYMAX_TRACING_EXPORTER", None)
        otlp_endpoint = otlp_endpoint or os.getenv("PRIORITYMAX_TRACING_OTLP_ENDPOINT", None)
        jaeger_host = jaeger_host or os.getenv("PRIORITYMAX_TRACING_JAEGER_HOST", os.getenv("JAEGER_HOST", None))
        jaeger_port = jaeger_port or int(os.getenv("PRIORITYMAX_TRACING_JAEGER_PORT", os.getenv("JAEGER_PORT", "6831")))
        zipkin_endpoint = zipkin_endpoint or os.getenv("PRIORITYMAX_TRACING_ZIPKIN_ENDPOINT", None)
        sampler = sampler or os.getenv("PRIORITYMAX_TRACING_SAMPLER", "parentbased")
        probability = float(probability or os.getenv("PRIORITYMAX_TRACING_PROBABILITY", "0.1"))
        resource_attrs = resource_attrs or _parse_resource_attributes(os.getenv("PRIORITYMAX_TRACING_RESOURCE_ATTRS", None))

        # create resource
        res_attrs = {"service.name": service_name}
        res_attrs.update(resource_attrs)
        resource = Resource.create(res_attrs)

        sampler_obj = _choose_sampler(sampler, probability)

        provider = TracerProvider(resource=resource, sampler=sampler_obj)
        _TRACER_PROVIDER = provider

        # Otel exporter selection
        chosen = exporter or (("otlp" if _HAS_OTEL_EXPORTER_OTLP else "stdout"))
        LOG.info("Initializing tracing: exporter=%s service=%s sampler=%s", chosen, service_name, sampler)

        processors = []

        try:
            if chosen == "otlp" and _HAS_OTEL_EXPORTER_OTLP:
                ep = otlp_endpoint or os.getenv("PRIORITYMAX_TRACING_OTLP_ENDPOINT", None)
                if not ep:
                    LOG.warning("OTLP exporter selected but endpoint not provided; falling back to console")
                else:
                    try:
                        otlp = OTLPSpanExporter(endpoint=ep)
                        processors.append(BatchSpanProcessor(otlp))
                        _INSTALLED_EXPORTERS.append(("otlp", otlp))
                        LOG.info("Configured OTLP exporter -> %s", ep)
                    except Exception:
                        LOG.exception("Failed to configure OTLP exporter")
            if chosen == "jaeger" and _HAS_OTEL_EXPORTER_JAEGER:
                try:
                    je = JaegerExporter(agent_host_name=jaeger_host, agent_port=jaeger_port)
                    processors.append(BatchSpanProcessor(je))
                    _INSTALLED_EXPORTERS.append(("jaeger", je))
                    LOG.info("Configured Jaeger exporter -> %s:%s", jaeger_host, jaeger_port)
                except Exception:
                    LOG.exception("Failed to configure Jaeger exporter")
            if chosen == "zipkin" and _HAS_OTEL_EXPORTER_ZIPKIN:
                try:
                    ze = ZipkinExporter(endpoint=zipkin_endpoint)
                    processors.append(BatchSpanProcessor(ze))
                    _INSTALLED_EXPORTERS.append(("zipkin", ze))
                    LOG.info("Configured Zipkin exporter -> %s", zipkin_endpoint)
                except Exception:
                    LOG.exception("Failed to configure Zipkin exporter")
            # Console exporter for debugging or fallback
            if console or chosen == "stdout" or not processors:
                try:
                    cs = ConsoleSpanExporter()
                    processors.append(BatchSpanProcessor(cs))
                    _INSTALLED_EXPORTERS.append(("console", cs))
                    LOG.info("ConsoleSpanExporter installed for local debugging")
                except Exception:
                    LOG.exception("Failed to configure console exporter")
        except Exception:
            LOG.exception("Error configuring exporters")

        # attach processors
        for p in processors:
            provider.add_span_processor(p)

        # set global provider
        try:
            trace.set_tracer_provider(provider)
            _TRACER = trace.get_tracer(service_name)
            LOG.info("TracerProvider set successfully")
        except Exception:
            LOG.exception("Failed to set tracer provider")

    def get_tracer(name: Optional[str] = None) -> OtelTracer:
        """Return an OpenTelemetry tracer instance."""
        if not _HAS_OTEL:
            raise RuntimeError("OpenTelemetry not available")
        return trace.get_tracer(name or "prioritymax")

    def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Decorator to create a span for the wrapped function (sync or async).
        Usage:
            @trace_span("process_task")
            def process_task(...): ...
        """
        def _decor(fn: Callable):
            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def _wrapped(*args, **kwargs):
                    tracer = get_tracer(fn.__module__ if hasattr(fn, "__module__") else "prioritymax")
                    with tracer.start_as_current_span(name) as span:
                        if attributes:
                            for k, v in attributes.items():
                                try:
                                    span.set_attribute(k, v)
                                except Exception:
                                    pass
                        try:
                            return await fn(*args, **kwargs)
                        except Exception as e:
                            try:
                                span.record_exception(e)
                            except Exception:
                                pass
                            raise
                return _wrapped
            else:
                @functools.wraps(fn)
                def _wrapped(*args, **kwargs):
                    tracer = get_tracer(fn.__module__ if hasattr(fn, "__module__") else "prioritymax")
                    with tracer.start_as_current_span(name) as span:
                        if attributes:
                            for k, v in attributes.items():
                                try:
                                    span.set_attribute(k, v)
                                except Exception:
                                    pass
                        try:
                            return fn(*args, **kwargs)
                        except Exception as e:
                            try:
                                span.record_exception(e)
                            except Exception:
                                pass
                            raise
                return _wrapped
        return _decor

    # FastAPI middleware for tracing
    if _HAS_FASTAPI:
        class FastAPITracingMiddleware(BaseHTTPMiddleware):
            """
            ASGI middleware that starts a span for each incoming request and injects trace ids into logs.
            Adds trace headers to response (traceparent) and populates context with trace ids.
            """
            def __init__(self, app: FastAPI, service_name: Optional[str] = None, header_name: str = "X-Request-ID"):
                super().__init__(app)
                self.service_name = service_name or os.getenv("PRIORITYMAX_TRACING_SERVICE_NAME", "prioritymax")
                self.header_name = header_name

            async def dispatch(self, request: Request, call_next):
                tracer = get_tracer(self.service_name)
                # Try to extract existing context from incoming headers
                try:
                    ctx = trace.propagation.get_global_textmap().extract(request.headers)
                except Exception:
                    ctx = None
                span_name = f"HTTP {request.method} {request.url.path}"
                # start span with extracted context
                try:
                    with tracer.start_as_current_span(span_name, context=ctx) as span:
                        # add useful attributes
                        try:
                            span.set_attribute("http.method", request.method)
                            span.set_attribute("http.scheme", request.url.scheme)
                            span.set_attribute("http.target", request.url.path)
                            span.set_attribute("http.host", request.url.hostname or "")
                            span.set_attribute("service.name", self.service_name)
                        except Exception:
                            pass
                        # attach request_id for logs and correlation
                        req_id = request.headers.get(self.header_name) or os.urandom(8).hex()
                        try:
                            span.set_attribute("request.id", req_id)
                        except Exception:
                            pass
                        # place trace ids in logging context (best-effort)
                        try:
                            span_ctx = span.get_span_context()
                            trace_id = format(span_ctx.trace_id, "032x") if span_ctx.trace_id else None
                            span_id = format(span_ctx.span_id, "016x") if span_ctx.span_id else None
                            # push into contextvars (if app uses our common.set_context)
                            try:
                                from app.utils.common import set_context
                                set_context("trace_id", trace_id)
                                set_context("span_id", span_id)
                                set_context("request_id", req_id)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # execute handler
                        response = await call_next(request)
                        # set attributes from response
                        try:
                            span.set_attribute("http.status_code", response.status_code)
                        except Exception:
                            pass
                        # add traceparent header if possible (W3C)
                        try:
                            # Using global propagator to inject
                            headers = {}
                            trace.get_current_span()  # ensure current span
                            trace.propagation.get_global_textmap().inject(headers)
                            for k, v in headers.items():
                                response.headers[k] = v
                        except Exception:
                            pass
                        return response
                except Exception:
                    # fallback: still call next if tracing failed
                    LOG.exception("Tracing middleware failed for request")
                    return await call_next(request)
    else:
        class FastAPITracingMiddleware(object):
            def __init__(self, app: FastAPI, service_name: Optional[str] = None, header_name: str = "X-Request-ID"):
                LOG.debug("FastAPI not available; FastAPITracingMiddleware is a no-op")

    # Propagation helpers: inject/extract trace context into headers
    def inject_trace_context(headers: Dict[str, str]):
        """Inject the current trace context into the provided headers mapping (mutates headers)."""
        try:
            trace.propagation.get_global_textmap().inject(headers)
        except Exception:
            LOG.debug("inject_trace_context failed")

    def extract_trace_context(headers: Dict[str, str]):
        """Extract a context dict from headers; returns context object usable by start_as_current_span."""
        try:
            return trace.propagation.get_global_textmap().extract(headers)
        except Exception:
            return None

    def shutdown_tracing(timeout: float = 5.0):
        """
        Flush and shutdown installed exporters and tracer provider.
        """
        global _TRACER_PROVIDER, _INSTALLED_EXPORTERS
        try:
            if _TRACER_PROVIDER is None:
                return
            # call shutdown on processors/exporters if available
            try:
                # The SDK provides shutdown() on provider in recent versions
                _TRACER_PROVIDER.shutdown()
            except Exception:
                LOG.debug("TracerProvider.shutdown not supported; attempting manual processor shutdown")
                try:
                    for proc in list(_TRACER_PROVIDER._active_span_processors):  # type: ignore
                        try:
                            proc.shutdown()
                        except Exception:
                            pass
                except Exception:
                    pass
            LOG.info("Tracing shutdown completed")
        except Exception:
            LOG.exception("shutdown_tracing failed")

# -----------------------------------------------------------------------------
# Utility wrappers for common patterns
# -----------------------------------------------------------------------------
def instrument_requests_session(session: Optional[Any] = None):
    """
    Instrument requests library to propagate trace headers automatically.
    Best-effort; requires opentelemetry-instrumentation-requests installed.
    """
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument(session=session)
        LOG.info("Requests instrumentation enabled")
    except Exception:
        LOG.debug("Requests instrumentation not available")

def instrument_aiohttp_client():
    """
    Instrument aiohttp client to inject/extract trace context.
    Requires opentelemetry-instrumentation-aiohttp-client installed.
    """
    try:
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
        AioHttpClientInstrumentor().instrument()
        LOG.info("aiohttp client instrumentation enabled")
    except Exception:
        LOG.debug("aiohttp client instrumentation not available")

# -----------------------------------------------------------------------------
# CLI helper for tracing control / debug
# -----------------------------------------------------------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-tracing")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("init", help="Initialize tracing using env vars")
    sub.add_parser("shutdown", help="Shutdown tracing gracefully")
    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    if args.cmd == "init":
        init_tracing(console=True)
        print("Tracing initialized (console).")
    elif args.cmd == "shutdown":
        shutdown_tracing()
        print("Tracing shutdown requested.")
    else:
        parser.print_help()

# -----------------------------------------------------------------------------
# Public API exports
# -----------------------------------------------------------------------------
__all__ = [
    "init_tracing",
    "get_tracer",
    "trace_span",
    "FastAPITracingMiddleware",
    "inject_trace_context",
    "extract_trace_context",
    "shutdown_tracing",
    "instrument_requests_session",
    "instrument_aiohttp_client",
    "main_cli",
]

# allow running as CLI
if __name__ == "__main__":
    main_cli()

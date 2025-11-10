# backend/app/api/websocket_routes.py
"""
PriorityMax WebSocket Routes — Enterprise Edition
=================================================

Defines secure, multi-channel WebSocket endpoints for live observability,
autoscaler feedback, chaos lab streams, and workflow updates.

Each route delegates its channel connection to app.websocket_manager.websocket_endpoint,
while enforcing token-based authentication and graceful lifecycle management.

Features:
 - Secure JWT token validation (query or header)
 - Channel-level connection gating (metrics, autoscaler, chaos, admin, workflows)
 - REST introspection endpoint for monitoring active clients
 - Tight integration with main FastAPI lifecycle and readiness gating
 - Optional Prometheus metrics hooks for WS connection count per channel
"""

import logging
import asyncio
from typing import Optional

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Depends,
    status,
    Request,
    HTTPException,
)
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------
# Local imports (lazy-loaded modules protected for production safety)
# ---------------------------------------------------------------------
try:
    from app.websocket_manager import websocket_endpoint, get_ws_manager, ws_manager
except Exception as e:
    websocket_endpoint = None
    ws_manager = None
    get_ws_manager = lambda: None
    print("[WARN] websocket_manager unavailable:", e)

try:
    from app.auth import verify_jwt_token, JWTError
except Exception:
    JWTError = Exception
    def verify_jwt_token(token: str):
        return {"sub": "anonymous"}

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
LOG = logging.getLogger("prioritymax.api.websocket")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel("INFO")

router = APIRouter(prefix="/ws", tags=["websocket"])

# ---------------------------------------------------------------------
# Secure JWT Verification Helper
# ---------------------------------------------------------------------
async def _verify_token(token: Optional[str], channel: str = "unknown") -> Optional[str]:
    """Validates a JWT token; returns user ID or None."""
    if not token:
        return None
    try:
        payload = verify_jwt_token(token)
        user_id = payload.get("sub") or payload.get("user_id")
        if not user_id:
            raise JWTError("missing subject")
        return user_id
    except JWTError:
        LOG.warning(f"Invalid JWT token for WebSocket channel={channel}")
        return None
    except Exception as e:
        LOG.exception("Token verification error: %s", e)
        return None

# ---------------------------------------------------------------------
# Generic WebSocket entry wrapper (for all channels)
# ---------------------------------------------------------------------
async def _ws_entrypoint(websocket: WebSocket, channel: str, token: Optional[str]):
    """
    Generic handler delegating connection to websocket_endpoint().
    Handles JWT validation and unauthorized closure.
    """
    try:
        user = await _verify_token(token, channel)
        if user is None:
            # No valid auth — close early with code 4401 (unauthorized)
            await websocket.close(code=4401)
            LOG.warning("Rejected WS connection (unauthenticated) channel=%s", channel)
            return
        LOG.info("Accepted WS connection channel=%s user=%s", channel, user)
        await websocket_endpoint(websocket, channel=channel, token=token)
    except WebSocketDisconnect:
        LOG.info("WebSocket disconnected channel=%s", channel)
    except Exception:
        LOG.exception("WebSocket handler failed for channel=%s", channel)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

# ---------------------------------------------------------------------
# Channel-specific routes (delegating to _ws_entrypoint)
# ---------------------------------------------------------------------
@router.websocket("/metrics")
async def ws_metrics(websocket: WebSocket, token: Optional[str] = Query(None)):
    """Real-time metrics dashboard stream (CPU, queue, latency, throughput)."""
    await _ws_entrypoint(websocket, channel="metrics", token=token)

@router.websocket("/autoscaler")
async def ws_autoscaler(websocket: WebSocket, token: Optional[str] = Query(None)):
    """RL autoscaler + ML inference decision updates."""
    await _ws_entrypoint(websocket, channel="autoscaler", token=token)

@router.websocket("/chaos")
async def ws_chaos(websocket: WebSocket, token: Optional[str] = Query(None)):
    """ChaosLab simulation feed (node kills, injected latencies)."""
    await _ws_entrypoint(websocket, channel="chaos", token=token)

@router.websocket("/admin")
async def ws_admin(websocket: WebSocket, token: Optional[str] = Query(None)):
    """Admin dashboard feed (audits, deployments, alert events)."""
    await _ws_entrypoint(websocket, channel="admin", token=token)

@router.websocket("/workflows")
async def ws_workflows(websocket: WebSocket, token: Optional[str] = Query(None)):
    """Workflow DAG state updates and orchestration events."""
    await _ws_entrypoint(websocket, channel="workflows", token=token)

# ---------------------------------------------------------------------
# REST endpoint: WebSocket subsystem status
# ---------------------------------------------------------------------
@router.get("/status", summary="WebSocket subsystem status", response_model=dict)
async def get_ws_status(manager=Depends(get_ws_manager)):
    """
    Return a dictionary of active connections per channel.
    Example:
      {
        "status": "ok",
        "channels": {"metrics": 3, "autoscaler": 2, "chaos": 0}
      }
    """
    if not manager:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "manager_unavailable"})
    try:
        async with manager._lock:
            status_info = {ch: len(clients) for ch, clients in manager._clients.items()}
        return {"status": "ok", "channels": status_info}
    except Exception as e:
        LOG.exception("Failed to collect WS status: %s", e)
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

# ---------------------------------------------------------------------
# Startup / Shutdown registration for FastAPI
# ---------------------------------------------------------------------
def register_ws_events(app):
    """
    Registers startup/shutdown handlers for ws_manager lifecycle.
    Should be called once from app.main after router inclusion.

    Example:
        from app.api.websocket_routes import register_ws_events
        register_ws_events(app)
    """
    if not ws_manager:
        LOG.warning("ws_manager not available; skipping WS lifecycle registration")
        return

    @app.on_event("startup")
    async def _ws_startup():
        try:
            LOG.info("🔌 Starting WebSocket manager ...")
            await ws_manager.start()
            app.state.ws_manager = ws_manager
            LOG.info("✅ WebSocket manager started (channels=%s)", list(ws_manager._clients.keys()))
        except Exception:
            LOG.exception("Failed to start WebSocket manager")

    @app.on_event("shutdown")
    async def _ws_shutdown():
        try:
            LOG.info("🧹 Shutting down WebSocket manager ...")
            await ws_manager.stop()
        except Exception:
            LOG.exception("Failed to stop WebSocket manager")

# ---------------------------------------------------------------------
# Prometheus instrumentation (optional)
# ---------------------------------------------------------------------
try:
    from prometheus_client import Gauge
    WS_CONNECTIONS = Gauge(
        "prioritymax_ws_connections_total",
        "Number of active WebSocket connections per channel",
        ["channel"],
    )

    async def _export_ws_metrics():
        """Collects WS connection metrics periodically."""
        while True:
            try:
                if ws_manager and hasattr(ws_manager, "_clients"):
                    async with ws_manager._lock:
                        for ch, clients in ws_manager._clients.items():
                            WS_CONNECTIONS.labels(ch).set(len(clients))
            except Exception:
                LOG.debug("WS metrics collection failed", exc_info=True)
            await asyncio.sleep(10.0)
except Exception:
    WS_CONNECTIONS = None
    _export_ws_metrics = None

def register_ws_metrics_task(app):
    """Registers background task for exporting WS Prometheus metrics."""
    if not _export_ws_metrics:
        LOG.debug("WS metrics exporter unavailable; skipping")
        return
    @app.on_event("startup")
    async def _start_ws_metrics_task():
        app.state.ws_metrics_task = asyncio.create_task(_export_ws_metrics())
        LOG.info("WS metrics background exporter started")

    @app.on_event("shutdown")
    async def _stop_ws_metrics_task():
        task = getattr(app.state, "ws_metrics_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        LOG.info("WS metrics background exporter stopped")

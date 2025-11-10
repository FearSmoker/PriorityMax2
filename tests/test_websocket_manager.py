# backend/tests/test_websocket_manager.py
"""
PriorityMax WebSocket Manager — Integration Test Suite
-------------------------------------------------------

This suite validates:
 - Client connections across multiple channels
 - Broadcast delivery and isolation
 - Disconnect cleanup
 - Heartbeat self-cleaning logic

Uses FastAPI TestClient and websockets (ASGI-level).
"""

import asyncio
import json
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# local imports
from app.websocket_manager import (
    ws_manager,
    websocket_endpoint,
    get_ws_manager,
)
from app.api.websocket_routes import router as ws_router, register_ws_events

# -----------------------------------------------------------------------------
# Setup: minimal FastAPI instance for testing
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def test_app():
    app = FastAPI()
    app.include_router(ws_router)
    register_ws_events(app)
    client = TestClient(app)
    yield client

# -----------------------------------------------------------------------------
# Async fixture to spin temporary ASGI app in asyncio context
# -----------------------------------------------------------------------------
@pytest.fixture
def asgi_app():
    app = FastAPI()
    app.include_router(ws_router)
    register_ws_events(app)
    return app

# -----------------------------------------------------------------------------
# Basic connection and broadcast tests
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_websocket_connect_and_broadcast(asgi_app):
    """
    Connect two clients to /ws/metrics, broadcast, ensure both receive.
    """
    from starlette.testclient import WebSocketTestSession
    from starlette.testclient import TestClient

    client = TestClient(asgi_app)
    # connect two clients
    with client.websocket_connect("/ws/metrics") as ws1, client.websocket_connect("/ws/metrics") as ws2:
        await asyncio.sleep(0.2)
        # broadcast via manager
        msg = {"event": "test_broadcast", "data": 42}
        await ws_manager.broadcast("metrics", msg)
        # allow event propagation
        data1 = ws1.receive_json(timeout=5)
        data2 = ws2.receive_json(timeout=5)
        assert data1 == msg
        assert data2 == msg
        # disconnect
        ws1.close()
        ws2.close()

    # confirm cleanup
    async with ws_manager._lock:
        metrics_clients = ws_manager._clients.get("metrics", [])
    assert len(metrics_clients) == 0

# -----------------------------------------------------------------------------
# Channel isolation test
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_channel_isolation(asgi_app):
    """
    Ensure broadcasts are isolated per channel (metrics vs autoscaler).
    """
    from starlette.testclient import TestClient
    client = TestClient(asgi_app)

    with client.websocket_connect("/ws/metrics") as ws_metrics, client.websocket_connect("/ws/autoscaler") as ws_auto:
        await asyncio.sleep(0.2)
        msg_metrics = {"channel": "metrics", "value": 111}
        msg_auto = {"channel": "autoscaler", "value": 999}
        await ws_manager.broadcast("metrics", msg_metrics)
        await ws_manager.broadcast("autoscaler", msg_auto)

        recv_metrics = ws_metrics.receive_json(timeout=5)
        recv_auto = ws_auto.receive_json(timeout=5)
        assert recv_metrics["channel"] == "metrics"
        assert recv_auto["channel"] == "autoscaler"

        ws_metrics.close()
        ws_auto.close()

# -----------------------------------------------------------------------------
# Disconnect cleanup test
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_disconnect_cleanup(asgi_app):
    """
    Connect client and close websocket; ensure manager removes it.
    """
    from starlette.testclient import TestClient
    client = TestClient(asgi_app)
    with client.websocket_connect("/ws/metrics") as ws:
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.5)
    async with ws_manager._lock:
        assert len(ws_manager._clients.get("metrics", [])) == 0

# -----------------------------------------------------------------------------
# Heartbeat loop test
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_heartbeat_loop(asgi_app):
    """
    Spawn dummy client and simulate timeout to trigger heartbeat removal.
    """
    # shorter heartbeat for test
    ws_manager._running = True
    task = asyncio.create_task(ws_manager._heartbeat_loop())
    try:
        dummy = type("Dummy", (), {"alive": True, "channel": "metrics", "last_heartbeat": 0.0})
        async with ws_manager._lock:
            ws_manager._clients["metrics"] = [dummy]
        await asyncio.sleep(WS_HEARTBEAT_INTERVAL * 2)
        async with ws_manager._lock:
            assert len(ws_manager._clients["metrics"]) == 0
    finally:
        ws_manager._running = False
        task.cancel()
        try:
            await task
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Stress test: broadcast many messages
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_broadcast_backpressure(asgi_app):
    """
    Test dropping messages when client outbox fills up.
    """
    from starlette.testclient import TestClient
    client = TestClient(asgi_app)
    with client.websocket_connect("/ws/metrics") as ws:
        # artificially fill outbox
        async with ws_manager._lock:
            ws_client = ws_manager._clients["metrics"][0]
            for _ in range(ws_client.outbox.maxsize):
                ws_client.outbox.put_nowait("spam")

        # broadcast a message — should drop due to full queue
        await ws_manager.broadcast("metrics", {"event": "should_drop"})
        ws.close()
    async with ws_manager._lock:
        assert len(ws_manager._clients.get("metrics", [])) == 0

# backend/app/websocket_manager.py
"""
PriorityMax WebSocket Manager (production-ready)
- Central hub with channels
- JWT auth (best-effort)
- Bounded per-client outbox and backpressure handling
- Heartbeat + auto-cleanup
- Integration hooks for metrics/alerts
- Safe auto-start helper (can be called from app startup)
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import threading
import functools
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect, Depends, status

# JWT helper (best-effort import)
try:
    from app.auth import verify_jwt_token
    from jose import JWTError
except Exception:
    verify_jwt_token = None
    JWTError = Exception  # fallback for catches

# metrics hook (best-effort)
try:
    from prometheus_client import Gauge, Counter
    _HAS_PROM = True
except Exception:
    Gauge = Counter = None
    _HAS_PROM = False

# local metrics getter (best-effort)
try:
    from app.metrics import get_global_metrics
except Exception:
    get_global_metrics = lambda: None

# Logging
LOG = logging.getLogger("prioritymax.websocket")
LOG.setLevel(os.getenv("PRIORITYMAX_WS_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Config (env-driven)
WS_HEARTBEAT_INTERVAL = float(os.getenv("PRIORITYMAX_WS_HEARTBEAT_INTERVAL", "20"))
WS_QUEUE_MAXSIZE = int(os.getenv("PRIORITYMAX_WS_QUEUE_MAXSIZE", "100"))
WS_BROADCAST_TIMEOUT = float(os.getenv("PRIORITYMAX_WS_BROADCAST_TIMEOUT", "5.0"))
WS_SECRET_KEY = os.getenv("PRIORITYMAX_JWT_SECRET", "supersecret")
WS_AUTOSTART = os.getenv("PRIORITYMAX_WS_AUTOSTART", "true").lower() in ("1", "true", "yes")

# Prometheus metrics (optional)
if _HAS_PROM:
    WS_CONNECTED_CLIENTS = Gauge("prioritymax_ws_connected_clients", "Connected WebSocket clients", ["channel"])
    WS_MESSAGES_SENT = Counter("prioritymax_ws_messages_sent_total", "Messages sent", ["channel"])
else:
    WS_CONNECTED_CLIENTS = None
    WS_MESSAGES_SENT = None

# -----------------------------------------------------------------------------
# Client representation
# -----------------------------------------------------------------------------
class WSClient:
    def __init__(self, websocket: WebSocket, channel: str, user: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.websocket = websocket
        self.channel = channel
        self.user = user or "anonymous"
        self.connected_at = time.time()
        self.outbox: asyncio.Queue = asyncio.Queue(maxsize=WS_QUEUE_MAXSIZE)
        self.last_heartbeat = time.time()
        self.alive = True
        # writer task pointer (managed by endpoint)
        self._writer_task: Optional[asyncio.Task] = None

    def __repr__(self):
        return f"<WSClient id={self.id[:8]} user={self.user} ch={self.channel}>"

# -----------------------------------------------------------------------------
# Central WebSocket manager
# -----------------------------------------------------------------------------
class WebSocketManager:
    def __init__(self):
        self._clients: Dict[str, List[WSClient]] = {}
        self._lock = asyncio.Lock()
        self._metrics = get_global_metrics()
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._on_connect_hooks: List[Callable[[WSClient], Any]] = []
        self._on_disconnect_hooks: List[Callable[[WSClient], Any]] = []
        self._on_message_hooks: List[Callable[[WSClient, Any], Any]] = []

    # -------------------------
    # Connection lifecycle
    # -------------------------
    async def connect(self, websocket: WebSocket, channel: str, token: Optional[str] = None) -> WSClient:
        # Try verify token if available
        user_id = "anonymous"
        if token and verify_jwt_token:
            try:
                payload = verify_jwt_token(token)
                user_id = payload.get("sub", "anonymous")
            except JWTError:
                LOG.warning("Invalid JWT token for websocket connection; rejecting")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

        # accept connection
        await websocket.accept()
        client = WSClient(websocket, channel, user=user_id)

        # register
        async with self._lock:
            self._clients.setdefault(channel, []).append(client)
            cur = len(self._clients[channel])
        if _HAS_PROM:
            try:
                WS_CONNECTED_CLIENTS.labels(channel=channel).set(cur)
            except Exception:
                LOG.debug("Prom metrics update failed on connect")

        LOG.info("WS connected %s", client)
        # call hooks (best-effort)
        for h in list(self._on_connect_hooks):
            try:
                if asyncio.iscoroutinefunction(h):
                    await h(client)
                else:
                    h(client)
            except Exception:
                LOG.exception("on_connect hook error")
        return client

    async def disconnect(self, client: WSClient):
        async with self._lock:
            lst = self._clients.get(client.channel, [])
            if client in lst:
                lst.remove(client)
                LOG.info("WS disconnected %s", client)
            else:
                LOG.debug("Client already removed %s", client)
            cur = len(self._clients.get(client.channel, []))
        if _HAS_PROM:
            try:
                WS_CONNECTED_CLIENTS.labels(channel=client.channel).set(cur)
            except Exception:
                pass

        # cancel writer task if exists
        if client._writer_task:
            try:
                client._writer_task.cancel()
            except Exception:
                pass

        for h in list(self._on_disconnect_hooks):
            try:
                if asyncio.iscoroutinefunction(h):
                    await h(client)
                else:
                    h(client)
            except Exception:
                LOG.exception("on_disconnect hook failed")

    # -------------------------
    # Broadcast API
    # -------------------------
    async def broadcast(self, channel: str, message: Union[str, Dict[str, Any]], *, drop_if_full: bool = True):
        """Broadcast to channel. If queue full, drop or block depending on drop_if_full."""
        if isinstance(message, dict):
            payload = json.dumps(message)
        else:
            payload = str(message)

        async with self._lock:
            clients = list(self._clients.get(channel, []))

        if not clients:
            return 0

        sent = 0
        for client in clients:
            try:
                if client.outbox.full():
                    if drop_if_full:
                        LOG.debug("Dropping message for %s due to full outbox", client)
                        continue
                    else:
                        # attempt to put with timeout
                        try:
                            await asyncio.wait_for(client.outbox.put(payload), timeout=WS_BROADCAST_TIMEOUT)
                        except asyncio.TimeoutError:
                            LOG.warning("Timeout queueing message to %s", client)
                            continue
                else:
                    await client.outbox.put(payload)
                sent += 1
            except Exception:
                LOG.exception("Error enqueueing broadcast to %s", client)

        if _HAS_PROM and sent:
            try:
                WS_MESSAGES_SENT.labels(channel=channel).inc(sent)
            except Exception:
                pass

        return sent

    async def send_direct(self, client: WSClient, message: Union[str, Dict[str, Any]]):
        if isinstance(message, dict):
            payload = json.dumps(message)
        else:
            payload = str(message)
        try:
            if client.outbox.full():
                LOG.debug("Direct send: outbox full for %s — dropping", client)
                return False
            await client.outbox.put(payload)
            if _HAS_PROM:
                try:
                    WS_MESSAGES_SENT.labels(channel=client.channel).inc(1)
                except Exception:
                    pass
            return True
        except Exception:
            LOG.exception("send_direct failed for %s", client)
            return False

    # -------------------------
    # Per-client writer
    # -------------------------
    async def _client_writer(self, client: WSClient):
        try:
            while client.alive:
                try:
                    msg = await client.outbox.get()
                except asyncio.CancelledError:
                    break
                try:
                    # send_text is awaited with timeout protection
                    send = client.websocket.send_text(msg)
                    await asyncio.wait_for(send, timeout=WS_BROADCAST_TIMEOUT)
                    client.last_heartbeat = time.time()
                except asyncio.TimeoutError:
                    LOG.warning("Timeout sending message to %s", client)
                except WebSocketDisconnect:
                    # let disconnect handler remove client
                    await self.disconnect(client)
                    break
                except Exception:
                    LOG.exception("Error in writer for %s", client)
                    # small sleep to avoid tight exception loop
                    await asyncio.sleep(0.1)
        finally:
            LOG.debug("Writer exiting for %s", client)

    # -------------------------
    # Incoming handler
    # -------------------------
    async def handle_incoming(self, client: WSClient):
        try:
            while client.alive:
                try:
                    # receive_text will raise WebSocketDisconnect as appropriate
                    data = await client.websocket.receive_text()
                except WebSocketDisconnect:
                    await self.disconnect(client)
                    break
                except Exception:
                    LOG.exception("receive_text error for %s", client)
                    await asyncio.sleep(0.1)
                    continue

                # update heartbeat timestamp
                client.last_heartbeat = time.time()

                # dispatch to on_message hooks (best-effort)
                for h in list(self._on_message_hooks):
                    try:
                        if asyncio.iscoroutinefunction(h):
                            await h(client, data)
                        else:
                            # run sync hooks in threadpool to avoid blocking loop
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, functools.partial(h, client, data))
                    except Exception:
                        LOG.exception("on_message hook failed for %s", client)
        except Exception:
            LOG.exception("Incoming loop unexpected error for %s", client)
        finally:
            # ensure cleanup
            try:
                await self.disconnect(client)
            except Exception:
                pass

    # -------------------------
    # Heartbeat cleanup loop
    # -------------------------
    async def _heartbeat_loop(self):
        LOG.info("WS heartbeat loop started (interval=%ds)", WS_HEARTBEAT_INTERVAL)
        try:
            while self._running:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
                now = time.time()
                stale: List[WSClient] = []
                async with self._lock:
                    for ch, clients in list(self._clients.items()):
                        keep = []
                        for c in clients:
                            # heartbeat: if no writes/reads for 3x interval, mark stale
                            if now - c.last_heartbeat > WS_HEARTBEAT_INTERVAL * 3:
                                LOG.info("Client timed out: %s", c)
                                c.alive = False
                                stale.append(c)
                                continue
                            keep.append(c)
                        self._clients[ch] = keep
                        if _HAS_PROM:
                            try:
                                WS_CONNECTED_CLIENTS.labels(channel=ch).set(len(keep))
                            except Exception:
                                pass
                # attempt to close stale connections
                for c in stale:
                    try:
                        await c.websocket.close()
                    except Exception:
                        pass
        except asyncio.CancelledError:
            LOG.debug("Heartbeat loop cancelled")
        except Exception:
            LOG.exception("Heartbeat loop failed")
        finally:
            LOG.info("WS heartbeat loop stopped")

    # -------------------------
    # Hook decorators
    # -------------------------
    def on_connect(self, fn: Callable[[WSClient], Any]):
        self._on_connect_hooks.append(fn)
        return fn

    def on_disconnect(self, fn: Callable[[WSClient], Any]):
        self._on_disconnect_hooks.append(fn)
        return fn

    def on_message(self, fn: Callable[[WSClient, Any], Any]):
        self._on_message_hooks.append(fn)
        return fn

    # -------------------------
    # Lifecycle control
    # -------------------------
    async def start(self):
        if self._running:
            return
        self._running = True
        # start heartbeat
        loop = asyncio.get_event_loop()
        self._heartbeat_task = loop.create_task(self._heartbeat_loop())
        LOG.info("WebSocketManager started")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        # cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except Exception:
                pass
        # close all clients
        async with self._lock:
            for ch, clients in list(self._clients.items()):
                for c in clients:
                    try:
                        await c.websocket.close()
                    except Exception:
                        pass
            self._clients.clear()
        LOG.info("WebSocketManager stopped")

# -----------------------------------------------------------------------------
# Singleton instance and FastAPI helpers
# -----------------------------------------------------------------------------
ws_manager = WebSocketManager()

async def get_ws_manager() -> WebSocketManager:
    return ws_manager

async def websocket_endpoint(
    websocket: WebSocket,
    channel: str,
    token: Optional[str] = None,
    manager: WebSocketManager = Depends(get_ws_manager)
):
    """
    Generic websocket endpoint handler for routes like:
        /ws/{channel}?token=...
    The FastAPI route should call this handler (it does the accept/looping).
    """
    # register client
    client = await manager.connect(websocket, channel, token)
    # start writer and reader tasks
    loop = asyncio.get_event_loop()
    writer_task = loop.create_task(manager._client_writer(client))
    # attach so disconnect cancels it
    client._writer_task = writer_task
    reader_task = loop.create_task(manager.handle_incoming(client))
    # wait for either to finish
    try:
        await asyncio.gather(writer_task, reader_task)
    except Exception:
        LOG.debug("websocket_endpoint gather ended")
    finally:
        # ensure disconnect
        try:
            await manager.disconnect(client)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Example hooks for logging
# -----------------------------------------------------------------------------
@ws_manager.on_connect
def _log_connect(client: WSClient):
    LOG.info("WS CONNECT: %s at %s", client, datetime.utcnow().isoformat())

@ws_manager.on_disconnect
def _log_disconnect(client: WSClient):
    LOG.info("WS DISCONNECT: %s", client)

# -----------------------------------------------------------------------------
# Safe background autostart helper
# -----------------------------------------------------------------------------
def _start_manager_in_thread():
    """
    Spawn a background thread with its own event loop to start the ws_manager.
    This is used when module is imported outside of an async context (e.g., gunicorn preload).
    """
    def _runner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(ws_manager.start())
            # keep the loop running to serve background heartbeat & tasks
            loop.run_forever()
        except Exception:
            LOG.exception("WS manager thread failed")
    t = threading.Thread(target=_runner, daemon=True, name="prioritymax-ws-thread")
    t.start()
    return t

def start_background_manager():
    """
    Start the manager safely:
      - if an asyncio loop is running, schedule start() on it
      - otherwise spawn a background thread with its own loop
    Call this from your FastAPI startup event or container entrypoint.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # schedule start on running loop
            loop.create_task(ws_manager.start())
            LOG.info("Scheduled ws_manager.start() on running loop")
            return None
        else:
            # not running, spawn thread
            return _start_manager_in_thread()
    except RuntimeError:
        # no running loop in this thread -> spawn thread
        return _start_manager_in_thread()

# Optionally auto-start if requested (safe)
if WS_AUTOSTART:
    try:
        start_background_manager()
    except Exception:
        LOG.exception("Auto-start of WebSocket manager failed")

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------

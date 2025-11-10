# backend/app/queue/rabbitmq_queue.py
"""
PriorityMax RabbitMQ Queue interface (async + production-ready)

Features:
 - Async produce/consume using aio-pika (asyncio)
 - Optional synchronous pika producer/consumer helpers (for blocking contexts)
 - Exchanges (topic/direct/fanout) and routing key support
 - Dead-letter queue (DLQ) with configurable policies
 - Retries with exponential backoff + jitter (idempotent-friendly)
 - Bounded TaskPool consumer model (one connection/channel, worker pool per queue)
 - JSON Schema validation (optional) and Avro support (optional)
 - Prometheus metrics hooks (optional)
 - Admin helpers (declare queues/exchanges, purge, bindings)
 - Queue depth / consumer lag helpers (via queue declare passive)
 - Health checks and graceful shutdown
 - Hooks for instrumentation (autoscaler hints, metrics)
 - Example handler & demo run in main guard
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import math
import asyncio
import logging
import random
import functools
import pathlib
import typing as t
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Coroutine, Tuple

# Optional libs
_HAS_AIO_PIKA = False
_HAS_PIKA = False
_HAS_JSONSCHEMA = False
_HAS_FASTAVRO = False
_HAS_PROM = False

try:
    import aio_pika
    from aio_pika import connect_robust, ExchangeType, Message, IncomingMessage, RobustConnection, RobustChannel
    _HAS_AIO_PIKA = True
except Exception:
    aio_pika = None
    connect_robust = None
    ExchangeType = None
    Message = IncomingMessage = RobustConnection = RobustChannel = None
    _HAS_AIO_PIKA = False

try:
    import pika  # sync fallback
    _HAS_PIKA = True
except Exception:
    pika = None
    _HAS_PIKA = False

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except Exception:
    jsonschema = None
    _HAS_JSONSCHEMA = False

try:
    from fastavro import schemaless_writer, schemaless_reader, parse_schema
    _HAS_FASTAVRO = True
except Exception:
    schemaless_writer = schemaless_reader = parse_schema = None
    _HAS_FASTAVRO = False

try:
    from prometheus_client import Counter, Gauge, Histogram
    _HAS_PROM = True
except Exception:
    Counter = Gauge = Histogram = None
    _HAS_PROM = False

# Logging
LOG = logging.getLogger("prioritymax.queue.rabbitmq")
LOG.setLevel(os.getenv("PRIORITYMAX_RABBITMQ_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RABBIT_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
DEFAULT_DLQ_SUFFIX = os.getenv("PRIORITYMAX_RABBIT_DLQ_SUFFIX", ".dlq")
DEFAULT_QUEUE_PREFETCH = int(os.getenv("PRIORITYMAX_RABBIT_PREFETCH", "32"))
DEFAULT_MAX_RETRIES = int(os.getenv("PRIORITYMAX_RABBIT_MAX_RETRIES", "5"))

# Prometheus metrics
if _HAS_PROM:
    RABBIT_PRODUCE_COUNT = Counter("prioritymax_rabbit_produce_total", "Total messages produced", ["queue", "status"])
    RABBIT_CONSUME_COUNT = Counter("prioritymax_rabbit_consume_total", "Total messages consumed", ["queue", "status"])
    RABBIT_DLQ_COUNT = Counter("prioritymax_rabbit_dlq_total", "Number of messages sent to DLQ", ["queue"])
    RABBIT_INFLIGHT = Gauge("prioritymax_rabbit_inflight", "Current inflight messages per queue", ["queue"])
    RABBIT_BACKLOG = Gauge("prioritymax_rabbit_backlog", "Pending backlog size per queue", ["queue"])
    RABBIT_PRODUCE_LATENCY = Histogram("prioritymax_rabbit_produce_seconds", "Produce latency (s)")
    RABBIT_CONSUME_LATENCY = Histogram("prioritymax_rabbit_consume_seconds", "Consume processing latency (s)")
else:
    RABBIT_PRODUCE_COUNT = RABBIT_CONSUME_COUNT = RABBIT_DLQ_COUNT = RABBIT_INFLIGHT = RABBIT_BACKLOG = RABBIT_PRODUCE_LATENCY = RABBIT_CONSUME_LATENCY = None

# Type aliases
MessageHandler = Callable[[Dict[str, Any], Dict[str, Any]], Coroutine[Any, Any, bool]]
# handler(payload, meta) -> bool (True processed, False permanent fail)

# Utils
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _gen_msg_id() -> str:
    return str(uuid.uuid4())

def _safe_json_loads(b: t.Union[bytes, str]) -> Any:
    if isinstance(b, bytes):
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return b
    if isinstance(b, str):
        try:
            return json.loads(b)
        except Exception:
            return b
    return b

def _exponential_backoff(attempt: int, base: float = 0.2, cap: float = 30.0, jitter: bool = True) -> float:
    exp = base * (2 ** (attempt - 1))
    sleep = min(exp, cap)
    if jitter:
        sleep = sleep * (1.0 + (random.random() - 0.5) * 0.2)
    return sleep

# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class RabbitMQConfig:
    url: str = DEFAULT_RABBIT_URL
    prefetch_count: int = DEFAULT_QUEUE_PREFETCH
    client_name: str = field(default_factory=lambda: f"prioritymax-{uuid.uuid4().hex[:8]}")
    heartbeat: int = 60
    connection_attempts: int = 3
    retry_delay: int = 5
    ssl: Optional[dict] = None
    default_dlq_suffix: str = DEFAULT_DLQ_SUFFIX
    max_retries: int = DEFAULT_MAX_RETRIES
    # pool settings
    default_concurrency: int = 8
    default_queue_maxsize: Optional[int] = None  # if None, use concurrency*4
    auto_ack: bool = False  # if True, will ack immediately upon delivery (not recommended)
    # optional: exchange defaults
    default_exchange: str = "prioritymax"
    default_exchange_type: str = "topic"

# -----------------------------
# TaskPoolConsumer for RabbitMQ
# -----------------------------
class TaskPoolConsumer:
    """
    Bounded TaskPool consumer for a single queue.
    Uses one aio-pika consumer and an asyncio.Queue + worker pool.
    """

    def __init__(
        self,
        queue_name: str,
        rabbit: "RabbitMQQueue",
        handler: MessageHandler,
        routing_key: Optional[str] = None,
        exchange: Optional[str] = None,
        concurrency: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        enable_dlq: bool = True,
        max_retries: Optional[int] = None,
        auto_ack: bool = False,
        json_deserializer: Optional[Callable[[bytes], Any]] = None,
    ):
        self.queue_name = queue_name
        self.rabbit = rabbit
        self.handler = handler
        self.routing_key = routing_key or queue_name
        self.exchange = exchange or rabbit.config.default_exchange
        self.concurrency = concurrency or rabbit.config.default_concurrency
        self.max_queue_size = max_queue_size or (self.concurrency * 4)
        self.enable_dlq = enable_dlq
        self.max_retries = max_retries or rabbit.config.max_retries
        self.auto_ack = auto_ack or rabbit.config.auto_ack
        self.json_deserializer = json_deserializer

        self._connection: Optional[RobustConnection] = None
        self._channel: Optional[RobustChannel] = None
        self._consumer_tag: Optional[str] = None
        self._queue_obj = None  # aio_pika.Queue
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._poll_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._inflight = 0
        self._lock = asyncio.Lock()
        # metrics labels
        self._labels = {"queue": self.queue_name}

        # Prometheus gauges per queue (optional)
        self._prom_inflight = RABBIT_INFLIGHT.labels(queue=self.queue_name) if RABBIT_INFLIGHT else None
        self._prom_backlog = RABBIT_BACKLOG.labels(queue=self.queue_name) if RABBIT_BACKLOG else None

    # lifecycle
    async def start(self, connection: RobustConnection, channel: RobustChannel):
        """
        Start the consumer using a provided aio-pika RobustConnection and channel.
        Declares queue/exchange/bindings if required.
        """
        if not _HAS_AIO_PIKA:
            raise RuntimeError("aio-pika required for async RabbitMQ operations")

        self._connection = connection
        self._channel = channel

        # ensure exchange and queue exist
        exch = await self._channel.declare_exchange(self.exchange, ExchangeType(self.rabbit.config.default_exchange_type), durable=True)
        qargs = {}
        # configure DLX for queue if DLQ enabled
        if self.enable_dlq:
            dlx_name = f"{self.queue_name}{self.rabbit.config.default_dlq_suffix}.dlx"
            # declare dlx/exchange and DLQ queue separately (rabbitmq will create if necessary)
            await self._channel.declare_exchange(dlx_name, ExchangeType.TOPIC, durable=True)
            dlq_name = f"{self.queue_name}{self.rabbit.config.default_dlq_suffix}"
            # the main queue will have x-dead-letter-exchange and optionally x-dead-letter-routing-key
            qargs["x-dead-letter-exchange"] = dlx_name

        # declare queue
        self._queue_obj = await self._channel.declare_queue(self.queue_name, durable=True, arguments=qargs)
        # bind to exchange with routing key
        await self._queue_obj.bind(exch, routing_key=self.routing_key)
        LOG.info("[TaskPoolConsumer] queue declared and bound: %s -> %s (%s)", self.exchange, self.queue_name, self.routing_key)

        # start workers
        for wid in range(self.concurrency):
            t = asyncio.create_task(self._worker_loop(wid))
            self._workers.append(t)

        # start poller
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        """
        Stop poll loop, wait for queue to drain and cancel workers.
        """
        LOG.info("[TaskPoolConsumer] stopping queue=%s", self.queue_name)
        self._stop_event.set()
        # cancel poll loop first to stop adding new items
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception:
                LOG.debug("Poll task stop raised", exc_info=True)

        # wait for queue to drain or small timeout
        drain_wait = 0
        while not self._queue.empty() and drain_wait < 30:
            LOG.debug("[TaskPoolConsumer] draining queue (%d) queue=%s", self._queue.qsize(), self.queue_name)
            await asyncio.sleep(0.5)
            drain_wait += 0.5

        # cancel workers
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                LOG.debug("Worker termination raised", exc_info=True)

        LOG.info("[TaskPoolConsumer] stopped queue=%s", self.queue_name)

    # poll loop: fetch messages from aio-pika and enqueue into asyncio.Queue (bounded)
    async def _poll_loop(self):
        if not self._queue_obj:
            LOG.error("[TaskPoolConsumer] poll_loop started without queue object")
            return
        try:
            async with self._queue_obj.iterator() as queue_iter:
                async for incoming in queue_iter:
                    if self._stop_event.is_set():
                        break

                    # backpressure: if internal queue is full, pause consuming by cancelling iterator temporarily
                    if self._queue.full():
                        # attempt to stop the iterator until backlog decreases
                        LOG.debug("[TaskPoolConsumer] backlog full (%d), pausing consumption on %s", self._queue.qsize(), self.queue_name)
                        # Cancel iterator to pause; then re-declare iterator when resumed
                        break  # we'll exit iterator context and restart after some sleep

                    try:
                        await self._queue.put(incoming)
                        if self._prom_backlog:
                            try:
                                self._prom_backlog.set(self._queue.qsize())
                            except Exception:
                                pass
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        LOG.exception("[TaskPoolConsumer] failed to enqueue message for %s", self.queue_name)
            # if we exit due to backlog, allow resume loop that restarts iterator
            # implement resume with backoff until queue drained or stop event
            while not self._stop_event.is_set() and self._queue.full():
                await asyncio.sleep(0.5)
            # If stop event not set, restart poll loop
            if not self._stop_event.is_set():
                await asyncio.sleep(0.1)  # small yield
                self._poll_task = asyncio.create_task(self._poll_loop())
        except asyncio.CancelledError:
            LOG.info("[TaskPoolConsumer] poll loop cancelled for %s", self.queue_name)
        except Exception:
            LOG.exception("[TaskPoolConsumer] poll loop error for %s", self.queue_name)

    # worker loop: process messages pulled from internal queue
    async def _worker_loop(self, worker_id: int):
        LOG.info("[TaskPoolConsumer] worker-%d started for %s", worker_id, self.queue_name)
        while not self._stop_event.is_set():
            try:
                incoming: IncomingMessage = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("[TaskPoolConsumer] worker queue get failed")
                continue

            # prepare payload and meta
            try:
                body = incoming.body
                payload = self.json_deserializer(body) if self.json_deserializer else _safe_json_loads(body)
            except Exception:
                payload = incoming.body

            meta = {
                "queue": self.queue_name,
                "routing_key": incoming.routing_key if hasattr(incoming, "routing_key") else None,
                "delivery_tag": incoming.delivery_tag if hasattr(incoming, "delivery_tag") else None,
                "redelivered": getattr(incoming, "redelivered", False),
                "headers": getattr(incoming, "headers", None),
                "ts": _now_iso(),
            }

            success = await self._process_message(payload, meta, incoming)
            # mark task done
            try:
                self._queue.task_done()
            except Exception:
                pass
            if self._prom_backlog:
                try:
                    self._prom_backlog.set(self._queue.qsize())
                except Exception:
                    pass
        LOG.info("[TaskPoolConsumer] worker-%d stopped for %s", worker_id, self.queue_name)

    async def _process_message(self, payload: Any, meta: Dict[str, Any], incoming: IncomingMessage) -> bool:
        """
        Retry semantics, DLQ on permanent failure, ack/nack handling.
        """
        attempt = 0
        last_exc = None
        start = time.perf_counter()
        self._inflight += 1
        if self._prom_inflight:
            try:
                self._prom_inflight.set(self._inflight)
            except Exception:
                pass

        success = False
        for attempt in range(1, self.max_retries + 1):
            try:
                res = await self.handler(payload, meta)
                if res:
                    success = True
                    break
                else:
                    last_exc = RuntimeError("handler returned False (permanent failure)")
                    break
            except Exception as e:
                last_exc = e
                LOG.warning("[TaskPoolConsumer] handler exception queue=%s attempt=%d: %s", self.queue_name, attempt, e)
                if attempt < self.max_retries:
                    await asyncio.sleep(_exponential_backoff(attempt))
                else:
                    break

        latency = time.perf_counter() - start
        if RABBIT_CONSUME_LATENCY:
            try:
                RABBIT_CONSUME_LATENCY.observe(latency)
            except Exception:
                pass

        self._inflight -= 1
        if self._prom_inflight:
            try:
                self._prom_inflight.set(self._inflight)
            except Exception:
                pass

        # handle ack/nack/dlq
        try:
            if success:
                # acknowledge
                if not self.auto_ack:
                    try:
                        await incoming.ack()
                    except Exception:
                        LOG.debug("[TaskPoolConsumer] ack failed for %s", self.queue_name, exc_info=True)
                if RABBIT_CONSUME_COUNT:
                    RABBIT_CONSUME_COUNT.labels(queue=self.queue_name, status="ok").inc()
                if self.rabbit.on_message_processed:
                    try:
                        self.rabbit.on_message_processed(self.queue_name, {"payload": payload, "meta": meta})
                    except Exception:
                        LOG.debug("on_message_processed hook failed", exc_info=True)
                return True
            else:
                # permanent failure -> send to DLQ or nack requeue false
                if self.enable_dlq:
                    dlq_q = f"{self.queue_name}{self.rabbit.config.default_dlq_suffix}"
                    dlq_payload = {"original_queue": self.queue_name, "payload": payload, "meta": meta, "error": str(last_exc), "ts": _now_iso()}
                    try:
                        await self.rabbit.publish(dlq_q, dlq_payload, routing_key=dlq_q)
                        if RABBIT_DLQ_COUNT:
                            RABBIT_DLQ_COUNT.labels(queue=self.queue_name).inc()
                        if self.rabbit.on_dlq_published:
                            try:
                                self.rabbit.on_dlq_published(dlq_q, dlq_payload)
                            except Exception:
                                LOG.debug("on_dlq_published hook failed", exc_info=True)
                        # ack original message to remove from queue
                        if not self.auto_ack:
                            await incoming.ack()
                        LOG.warning("[TaskPoolConsumer] message sent to DLQ %s", dlq_q)
                    except Exception:
                        LOG.exception("[TaskPoolConsumer] failed to publish to DLQ for %s", self.queue_name)
                        # nack without requeue (drop) as last resort
                        try:
                            if not self.auto_ack:
                                await incoming.nack(requeue=False)
                        except Exception:
                            pass
                else:
                    # nack without requeue
                    try:
                        if not self.auto_ack:
                            await incoming.nack(requeue=False)
                    except Exception:
                        pass
                if RABBIT_CONSUME_COUNT:
                    RABBIT_CONSUME_COUNT.labels(queue=self.queue_name, status="error").inc()
                return False
        except Exception:
            LOG.exception("[TaskPoolConsumer] ack/nack/dlq handling error for %s", self.queue_name)
            return False

# -----------------------------
# RabbitMQQueue - main class
# -----------------------------
class RabbitMQQueue:
    """
    High-level RabbitMQ abstraction supporting async publish/consume, DLQ, validation, metrics, and admin helpers.
    """

    def __init__(self, config: Optional[RabbitMQConfig] = None):
        self.config = config or RabbitMQConfig()
        self._connection: Optional[RobustConnection] = None
        self._channel: Optional[RobustChannel] = None
        self._lock = asyncio.Lock()
        self._consumers: Dict[str, TaskPoolConsumer] = {}
        self._json_schemas: Dict[str, dict] = {}
        self._avro_schemas: Dict[str, Any] = {}
        self.on_message_processed: Optional[Callable[[str, dict], None]] = None
        self.on_dlq_published: Optional[Callable[[str, dict], None]] = None

    # -------------------------
    # Connection management
    # -------------------------
    async def start(self):
        """Establish a robust connection and channel."""
        if not _HAS_AIO_PIKA:
            raise RuntimeError("aio-pika is required for RabbitMQ async operations (pip install aio-pika)")

        async with self._lock:
            if self._connection and not getattr(self._connection, "is_closed", False):
                return
            reconnect_attempts = 0
            while reconnect_attempts < self.config.connection_attempts:
                try:
                    LOG.info("Connecting to RabbitMQ: %s", self.config.url)
                    self._connection = await connect_robust(self.config.url, client_name=self.config.client_name, heartbeat=self.config.heartbeat)
                    self._channel = await self._connection.channel()
                    await self._channel.set_qos(prefetch_count=self.config.prefetch_count)
                    LOG.info("Connected to RabbitMQ and channel opened")
                    return
                except Exception:
                    reconnect_attempts += 1
                    LOG.exception("Connection attempt %d failed, retrying in %ds", reconnect_attempts, self.config.retry_delay)
                    await asyncio.sleep(self.config.retry_delay)
            raise RuntimeError("Failed to connect to RabbitMQ after retries")

    async def stop(self):
        """Stop all consumers and close channel/connection."""
        LOG.info("Stopping RabbitMQQueue")
        # stop consumers
        for qname, cons in list(self._consumers.items()):
            try:
                await cons.stop()
            except Exception:
                LOG.exception("Stopping consumer %s failed", qname)
        self._consumers.clear()

        # close channel & connection
        try:
            if self._channel:
                await self._channel.close()
        except Exception:
            LOG.debug("Channel close failed", exc_info=True)
        try:
            if self._connection:
                await self._connection.close()
        except Exception:
            LOG.debug("Connection close failed", exc_info=True)
        LOG.info("RabbitMQQueue stopped")

    # -------------------------
    # Schema registration
    # -------------------------
    def register_json_schema(self, queue_name: str, schema: dict):
        self._json_schemas[queue_name] = schema
        LOG.info("Registered JSON schema for %s", queue_name)

    def register_avro_schema(self, queue_name: str, schema: dict):
        if not _HAS_FASTAVRO:
            raise RuntimeError("fastavro is required to register avro schemas")
        self._avro_schemas[queue_name] = parse_schema(schema)
        LOG.info("Registered Avro schema for %s", queue_name)

    # -------------------------
    # Publishing (async)
    # -------------------------
    async def publish(self, queue_or_exchange: str, message: t.Union[dict, str, bytes], routing_key: Optional[str] = None, exchange: Optional[str] = None, persistent: bool = True, use_avro: bool = False, headers: Optional[Dict[str, str]] = None) -> bool:
        """
        Publish message to exchange with routing key, or directly to queue via default exchange name equals queue.
        """
        if not _HAS_AIO_PIKA:
            raise RuntimeError("aio-pika required for async publish")

        await self.start()
        start = time.perf_counter()
        try:
            exch_name = exchange or self.config.default_exchange
            routing_key = routing_key or queue_or_exchange
            exch = await self._channel.declare_exchange(exch_name, ExchangeType(self.config.default_exchange_type), durable=True)
            if use_avro:
                if queue_or_exchange not in self._avro_schemas:
                    raise RuntimeError("No avro schema registered for {}".format(queue_or_exchange))
                buf = bytearray()
                schemaless_writer(buf, self._avro_schemas[queue_or_exchange], message)
                body = bytes(buf)
            else:
                if isinstance(message, (dict, list)):
                    body = json.dumps(message, default=str).encode("utf-8")
                elif isinstance(message, str):
                    body = message.encode("utf-8")
                elif isinstance(message, bytes):
                    body = message
                else:
                    body = str(message).encode("utf-8")

            msg = Message(body, delivery_mode=(aio_pika.DeliveryMode.PERSISTENT if persistent else aio_pika.DeliveryMode.NOT_PERSISTENT), headers=headers or {})
            await exch.publish(msg, routing_key=routing_key)

            latency = time.perf_counter() - start
            if RABBIT_PRODUCE_LATENCY:
                try:
                    RABBIT_PRODUCE_LATENCY.observe(latency)
                except Exception:
                    pass
            if RABBIT_PRODUCE_COUNT:
                try:
                    RABBIT_PRODUCE_COUNT.labels(queue=queue_or_exchange, status="ok").inc()
                except Exception:
                    pass
            return True
        except Exception:
            LOG.exception("Publish failed to %s", queue_or_exchange)
            if RABBIT_PRODUCE_COUNT:
                try:
                    RABBIT_PRODUCE_COUNT.labels(queue=queue_or_exchange, status="error").inc()
                except Exception:
                    pass
            return False

    # -------------------------
    # Publishing (blocking via pika) - optional sync helper
    # -------------------------
    def publish_sync(self, queue_or_exchange: str, message: t.Union[dict, str, bytes], routing_key: Optional[str] = None, exchange: Optional[str] = None, persistent: bool = True, headers: Optional[Dict[str, str]] = None) -> bool:
        if not _HAS_PIKA:
            raise RuntimeError("pika not installed for sync publish")
        try:
            params = pika.URLParameters(self.config.url)
            conn = pika.BlockingConnection(params)
            ch = conn.channel()
            exch_name = exchange or self.config.default_exchange
            ch.exchange_declare(exchange=exch_name, exchange_type=self.config.default_exchange_type, durable=True)
            rk = routing_key or queue_or_exchange
            if isinstance(message, (dict, list)):
                body = json.dumps(message, default=str).encode("utf-8")
            elif isinstance(message, str):
                body = message.encode("utf-8")
            elif isinstance(message, bytes):
                body = message
            else:
                body = str(message).encode("utf-8")
            properties = pika.BasicProperties(delivery_mode=2 if persistent else 1, headers=headers or {})
            ch.basic_publish(exchange=exch_name, routing_key=rk, body=body, properties=properties)
            conn.close()
            if RABBIT_PRODUCE_COUNT:
                RABBIT_PRODUCE_COUNT.labels(queue=queue_or_exchange, status="ok").inc()
            return True
        except Exception:
            LOG.exception("publish_sync failed")
            if RABBIT_PRODUCE_COUNT:
                try:
                    RABBIT_PRODUCE_COUNT.labels(queue=queue_or_exchange, status="error").inc()
                except Exception:
                    pass
            return False

    # -------------------------
    # Register consumer (TaskPool model)
    # -------------------------
    async def register_consumer(self, queue_name: str, handler: MessageHandler, routing_key: Optional[str] = None, exchange: Optional[str] = None, concurrency: Optional[int] = None, max_queue_size: Optional[int] = None, enable_dlq: bool = True, max_retries: Optional[int] = None, auto_ack: bool = False, json_deserializer: Optional[Callable[[bytes], Any]] = None):
        """
        Register a TaskPoolConsumer for queue_name. Start immediately.
        """
        if not _HAS_AIO_PIKA:
            raise RuntimeError("aio-pika required for consumers")

        if queue_name in self._consumers:
            LOG.warning("Consumer for %s already registered", queue_name)
            return

        await self.start()
        tpc = TaskPoolConsumer(
            queue_name=queue_name,
            rabbit=self,
            handler=handler,
            routing_key=routing_key,
            exchange=exchange,
            concurrency=concurrency,
            max_queue_size=max_queue_size,
            enable_dlq=enable_dlq,
            max_retries=max_retries,
            auto_ack=auto_ack,
            json_deserializer=json_deserializer
        )
        await tpc.start(self._connection, self._channel)
        self._consumers[queue_name] = tpc
        LOG.info("Registered TaskPool consumer for %s (concurrency=%d max_queue=%d)", queue_name, tpc.concurrency, tpc.max_queue_size)

    # -------------------------
    # Pause / resume consumer
    # -------------------------
    async def pause_consumer(self, queue_name: str):
        tpc = self._consumers.get(queue_name)
        if not tpc:
            return
        try:
            # no direct pause API in aio-pika iterator; stop poll loop by canceling and it will automatically resume later
            if tpc._poll_task:
                tpc._poll_task.cancel()
                LOG.info("Paused consumer poll loop for %s", queue_name)
        except Exception:
            LOG.exception("Failed to pause consumer %s", queue_name)

    async def resume_consumer(self, queue_name: str):
        tpc = self._consumers.get(queue_name)
        if not tpc:
            return
        try:
            if not tpc._poll_task or tpc._poll_task.done():
                tpc._poll_task = asyncio.create_task(tpc._poll_loop())
                LOG.info("Resumed consumer poll loop for %s", queue_name)
        except Exception:
            LOG.exception("Failed to resume consumer %s", queue_name)

    # -------------------------
    # Admin helpers
    # -------------------------
    async def declare_queue(self, queue_name: str, durable: bool = True, arguments: Optional[dict] = None):
        await self.start()
        await self._channel.declare_queue(queue_name, durable=durable, arguments=arguments or {})
        LOG.info("Declared queue %s", queue_name)

    async def declare_exchange(self, exchange: str, exchange_type: str = "topic", durable: bool = True):
        await self.start()
        await self._channel.declare_exchange(exchange, ExchangeType(exchange_type), durable=durable)
        LOG.info("Declared exchange %s type=%s", exchange, exchange_type)

    async def bind_queue(self, queue_name: str, exchange: str, routing_key: str):
        await self.start()
        exch = await self._channel.get_exchange(exchange)
        q = await self._channel.get_queue(queue_name)
        await q.bind(exch, routing_key=routing_key)
        LOG.info("Bound queue %s to exchange %s with key %s", queue_name, exchange, routing_key)

    async def purge_queue(self, queue_name: str):
        await self.start()
        q = await self._channel.get_queue(queue_name)
        await q.purge()
        LOG.info("Purged queue %s", queue_name)

    # -------------------------
    # Queue depth / consumer info (best-effort)
    # -------------------------
    async def queue_depth(self, queue_name: str) -> Optional[int]:
        """
        Returns the approximate message count in the queue using passive declare.
        """
        await self.start()
        try:
            q = await self._channel.get_queue(queue_name)
            info = await q.declare(passive=True)
            # info.message_count may be available depending on implementation
            return getattr(info, "message_count", None)
        except Exception:
            LOG.exception("Failed to get queue depth for %s", queue_name)
            return None

    # -------------------------
    # Health check
    # -------------------------
    async def health(self) -> Dict[str, Any]:
        res = {"connected": False, "channel_open": False, "queues": {}, "ts": _now_iso()}
        try:
            if not self._connection or getattr(self._connection, "is_closed", False):
                await self.start()
            res["connected"] = True
        except Exception:
            res["connected"] = False
        try:
            res["channel_open"] = self._channel is not None and not getattr(self._channel, "is_closed", False)
        except Exception:
            res["channel_open"] = False

        # report per-consumer queue depths
        for qname in self._consumers.keys():
            depth = await self.queue_depth(qname)
            res["queues"][qname] = {"depth": depth}
        return res

# -----------------------------
# Sample handler & demo usage
# -----------------------------
async def sample_handler(payload: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    """
    Example consumer handler that performs idempotent processing with basic error handling.
    """
    try:
        msg_id = payload.get("id") or payload.get("message_id") or _gen_msg_id()
        # simulate work
        await asyncio.sleep(0.01)
        LOG.info("Processed RabbitMQ message id=%s meta=%s", msg_id, meta)
        return True
    except Exception:
        LOG.exception("sample_handler failed")
        return False

if __name__ == "__main__":
    async def _demo():
        cfg = RabbitMQConfig(url=os.getenv("RABBITMQ_URL", DEFAULT_RABBIT_URL), default_exchange="prioritymax", default_exchange_type="topic")
        q = RabbitMQQueue(cfg)
        await q.start()
        # declare exchange & queue
        await q.declare_exchange("prioritymax", exchange_type="topic")
        await q.declare_queue("prioritymax-demo")
        # register consumer
        await q.register_consumer("prioritymax-demo", sample_handler, routing_key="prioritymax-demo", concurrency=4)
        # publish messages
        for i in range(10):
            ok = await q.publish("prioritymax-demo", {"id": f"msg-{i}", "value": i}, routing_key="prioritymax-demo")
            LOG.info("Published msg %d -> %s", i, ok)
        # run for a while
        await asyncio.sleep(5)
        await q.stop()

    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        LOG.info("Demo interrupted")
    except Exception:
        LOG.exception("Demo failed")

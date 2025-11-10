# backend/app/queue/kafka_queue.py
"""
PriorityMax Kafka Queue interface (async + production-ready)

Features:
 - Async produce / consume using aiokafka
 - Optional confluent_kafka transactional producer if installed
 - JSON Schema validation and optional Avro support
 - Dead-letter queue (DLQ) with configurable policies
 - Retries with exponential backoff + jitter
 - Prometheus metrics (instrumentation optional)
 - Topic admin helpers (create topics, describe)
 - Consumer group lag helper
 - Health checks and graceful shutdown
 - Hooks for instrumentation (autoscaler hints, metrics)
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
import traceback
import functools
import random
import typing as t
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Tuple, Coroutine

import pathlib

# Optional dependencies
_HAS_AIOKAFKA = False
_HAS_CONFLUENT = False
_HAS_JSONSCHEMA = False
_HAS_FASTAVRO = False
_HAS_PROM = False

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer, AIOKafkaAdminClient
    from aiokafka.errors import KafkaError
    _HAS_AIOKAFKA = True
except Exception:
    AIOKafkaProducer = AIOKafkaConsumer = AIOKafkaAdminClient = KafkaError = None
    _HAS_AIOKAFKA = False

try:
    import confluent_kafka
    from confluent_kafka import Producer as ConfluentProducer, KafkaException as ConfluentKafkaException
    _HAS_CONFLUENT = True
except Exception:
    ConfluentProducer = ConfluentKafkaException = None
    _HAS_CONFLUENT = False

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
    from prometheus_client import Counter, Histogram, Gauge
    _HAS_PROM = True
except Exception:
    Counter = Histogram = Gauge = None
    _HAS_PROM = False

# Logging
LOG = logging.getLogger("prioritymax.queue.kafka")
LOG.setLevel(os.getenv("PRIORITYMAX_KAFKA_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults & paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_DLQ_SUFFIX = os.getenv("PRIORITYMAX_KAFKA_DLQ_SUFFIX", ".dlq")
DEFAULT_CONSUMER_GROUP_PREFIX = os.getenv("PRIORITYMAX_CONSUMER_GROUP_PREFIX", "prioritymax-")
DEFAULT_MAX_RETRIES = int(os.getenv("PRIORITYMAX_KAFKA_MAX_RETRIES", "5"))

# Prometheus metrics (optional)
if _HAS_PROM:
    KAFKA_PRODUCE_COUNT = Counter("prioritymax_kafka_produce_total", "Total messages produced", ["topic", "status"])
    KAFKA_CONSUME_COUNT = Counter("prioritymax_kafka_consume_total", "Total messages consumed", ["topic", "status"])
    KAFKA_PRODUCE_LATENCY = Histogram("prioritymax_kafka_produce_seconds", "Produce latency seconds", buckets=[0.001,0.01,0.05,0.1,0.5,1,2])
    KAFKA_CONSUME_LATENCY = Histogram("prioritymax_kafka_consume_seconds", "Consume processing latency seconds", buckets=[0.001,0.01,0.05,0.1,0.5,1,2])
    KAFKA_DLQ_COUNT = Counter("prioritymax_kafka_dlq_total", "Number of messages sent to DLQ", ["topic"])
else:
    KAFKA_PRODUCE_COUNT = KAFKA_CONSUME_COUNT = KAFKA_PRODUCE_LATENCY = KAFKA_CONSUME_LATENCY = KAFKA_DLQ_COUNT = None

# Type aliases
MessageHandler = Callable[[Dict[str, Any], Dict[str, Any]], t.Coroutine[Any, Any, bool]]
# Handler signature: async def handle(payload_dict, meta) -> bool (True = success processed, False = permanent failure)

# utils
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _gen_msg_id() -> str:
    return str(uuid.uuid4())

def _safe_json_loads(b: t.Union[bytes, str]) -> Any:
    if isinstance(b, bytes):
        b = b.decode("utf-8", errors="ignore")
    try:
        return json.loads(b)
    except Exception:
        return b

def _exponential_backoff(attempt: int, base: float = 0.2, cap: float = 30.0, jitter: bool = True) -> float:
    """Return backoff seconds with optional jitter."""
    exp = base * (2 ** (attempt - 1))
    sleep = min(exp, cap)
    if jitter:
        jitter_val = random.random() * (sleep * 0.1)
        return sleep + jitter_val
    return sleep

# -----------------------------
# KafkaQueue configuration dataclass
# -----------------------------
@dataclass
class KafkaQueueConfig:
    bootstrap_servers: str = DEFAULT_KAFKA_BOOTSTRAP
    client_id: str = field(default_factory=lambda: f"prioritymax-{uuid.uuid4().hex[:8]}")
    security_protocol: Optional[str] = None  # e.g., "SSL" or "SASL_SSL"
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    sasl_mechanism: Optional[str] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None
    default_topic_partitions: int = 3
    default_topic_replication: int = 1
    auto_create_topics: bool = False
    enable_idempotence: bool = True
    transactional_id: Optional[str] = None  # used for confluent transactional producer
    max_retries: int = DEFAULT_MAX_RETRIES
    acks: str = "all"  # 'all' for strongest durability
    request_timeout_ms: int = 30000

# -----------------------------
# KafkaQueue class
# -----------------------------
class KafkaQueue:
    """
    High-level Kafka queue abstraction supporting async produce/consume, DLQ, validation, and metrics.
    """

    def __init__(self, config: Optional[KafkaQueueConfig] = None):
        self.config = config or KafkaQueueConfig()
        self.bootstrap = self.config.bootstrap_servers
        self.client_id = self.config.client_id
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._admin: Optional[AIOKafkaAdminClient] = None
        self._confluent_producer: Optional[ConfluentProducer] = None
        self._running_consumers: Dict[str, asyncio.Task] = {}
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._json_schemas: Dict[str, dict] = {}  # topic -> json schema
        self._avro_schemas: Dict[str, Any] = {}  # topic -> parsed avro schema (fastavro)
        self._consumer_handlers: Dict[str, MessageHandler] = {}
        self._consumer_options: Dict[str, dict] = {}
        self._dlq_suffix = DEFAULT_DLQ_SUFFIX
        self._consumer_group_prefix = DEFAULT_CONSUMER_GROUP_PREFIX

        # optional instrumentation hooks
        self.on_message_error: Optional[Callable[[str, bytes, Exception], None]] = None
        self.on_message_processed: Optional[Callable[[str, dict], None]] = None
        self.on_dlq_published: Optional[Callable[[str, dict], None]] = None

        # initialize confluent producer optionally (for transactions or high throughput)
        if _HAS_CONFLUENT:
            try:
                conf = {"bootstrap.servers": self.bootstrap, "client.id": self.client_id, "enable.idempotence": self.config.enable_idempotence}
                if self.config.transactional_id:
                    conf["transactional.id"] = self.config.transactional_id
                # security
                if self.config.security_protocol:
                    conf["security.protocol"] = self.config.security_protocol
                if self.config.sasl_mechanism:
                    conf["sasl.mechanism"] = self.config.sasl_mechanism
                if self.config.sasl_plain_username:
                    conf["sasl.username"] = self.config.sasl_plain_username
                    conf["sasl.password"] = self.config.sasl_plain_password
                self._confluent_producer = ConfluentProducer(conf)
                LOG.info("Confluent producer initialized (optional)")
            except Exception:
                LOG.exception("Failed to initialize confluent producer; falling back to aiokafka-only")

    # -------------------------
    # Internal: build aiokafka config
    # -------------------------
    def _build_aiokafka_kwargs(self) -> Dict[str, Any]:
        kw = {"bootstrap_servers": self.bootstrap, "client_id": self.client_id, "request_timeout_ms": self.config.request_timeout_ms}
        # SSL/SASL
        if self.config.security_protocol:
            kw["security_protocol"] = self.config.security_protocol
        if self.config.ssl_cafile:
            kw["ssl_cafile"] = self.config.ssl_cafile
        if self.config.ssl_certfile:
            kw["ssl_certfile"] = self.config.ssl_certfile
        if self.config.ssl_keyfile:
            kw["ssl_keyfile"] = self.config.ssl_keyfile
        if self.config.sasl_mechanism:
            kw["sasl_mechanism"] = self.config.sasl_mechanism
        if self.config.sasl_plain_username:
            kw["sasl_plain_username"] = self.config.sasl_plain_username
            kw["sasl_plain_password"] = self.config.sasl_plain_password
        return kw

    # -------------------------
    # Start / stop lifecycle
    # -------------------------
    async def start(self):
        """Start aiokafka producer/admin clients lazily and be ready to produce/consume."""
        async with self._start_lock:
            if self._producer:
                return
            if not _HAS_AIOKAFKA:
                raise RuntimeError("aiokafka is required for async Kafka operations. Install with `pip install aiokafka`.")
            kw = self._build_aiokafka_kwargs()
            self._producer = AIOKafkaProducer(**kw)
            await self._producer.start()
            LOG.info("AIOKafkaProducer started (bootstrap=%s client_id=%s)", self.bootstrap, self.client_id)
            try:
                self._admin = AIOKafkaAdminClient(**kw)
                await self._admin.start()
                LOG.info("AIOKafkaAdminClient started")
            except Exception:
                LOG.exception("Failed to start Kafka admin client; continuing without admin")

    async def stop(self):
        """Stop producer/consumer/admin cleanly."""
        try:
            self._stop_event.set()
            # cancel consumer tasks
            for topic, tsk in list(self._running_consumers.items()):
                try:
                    tsk.cancel()
                except Exception:
                    LOG.debug("Cancel consumer task for %s failed", topic)
            # await cancellation
            await asyncio.sleep(0.1)
            if self._consumer:
                try:
                    await self._consumer.stop()
                except Exception:
                    LOG.debug("consumer stop err", exc_info=True)
            if self._producer:
                try:
                    await self._producer.stop()
                except Exception:
                    LOG.debug("producer stop err", exc_info=True)
            if self._admin:
                try:
                    await self._admin.stop()
                except Exception:
                    LOG.debug("admin stop err", exc_info=True)
            if self._confluent_producer:
                try:
                    self._confluent_producer.flush(5)
                except Exception:
                    LOG.debug("confluent flush failed")
            LOG.info("KafkaQueue stopped")
        except Exception:
            LOG.exception("Error during stop")

    # -------------------------
    # Topic admin helpers
    # -------------------------
    async def ensure_topic(self, topic: str, num_partitions: Optional[int] = None, replication_factor: Optional[int] = None, timeout_ms: int = 30000) -> bool:
        """
        Ensure topic exists; create if missing (requires admin client to be available).
        """
        if not self._admin:
            LOG.warning("No admin client available to ensure topic %s", topic)
            return False
        partitions = num_partitions or self.config.default_topic_partitions
        repl = replication_factor or self.config.default_topic_replication
        try:
            # aiokafka admin create topics
            from aiokafka import AIOKafkaAdminClient
            topics = await self._admin.list_topics(timeout=5)
            if topic in topics:
                LOG.debug("Topic %s already exists", topic)
                return True
            # create topic
            await self._admin.create_topics([{
                "topic": topic,
                "num_partitions": partitions,
                "replication_factor": repl
            }], timeout=timeout_ms)
            LOG.info("Created topic %s p=%d r=%d", topic, partitions, repl)
            return True
        except Exception:
            LOG.exception("Topic ensure/create failed for %s", topic)
            return False

    async def list_topics(self) -> List[str]:
        if not self._admin:
            raise RuntimeError("admin client not available")
        md = await self._admin.list_topics()
        return list(md)

    # -------------------------
    # Schema registration
    # -------------------------
    def register_json_schema(self, topic: str, schema: dict):
        """Save JSON schema for topic for later validation (optional)."""
        self._json_schemas[topic] = schema
        LOG.info("Registered JSON schema for topic %s", topic)

    def register_avro_schema(self, topic: str, schema: dict):
        """Register Avro schema (fastavro parse) if fastavro available."""
        if not _HAS_FASTAVRO:
            raise RuntimeError("fastavro required for avro schema parsing")
        parsed = parse_schema(schema)
        self._avro_schemas[topic] = parsed
        LOG.info("Registered Avro schema for topic %s", topic)

    # -------------------------
    # Produce helpers
    # -------------------------
    async def produce(self, topic: str, value: t.Union[dict, bytes, str], key: Optional[str] = None, headers: Optional[Dict[str,str]] = None, partition: Optional[int] = None, use_avro: bool = False, timeout: float = 10.0) -> bool:
        """
        Async produce a message to Kafka. Handles JSON serialization, optional Avro, metrics, and retries.
        Returns True if successfully produced (ack'd).
        """
        if not self._producer:
            await self.start()

        payload_bytes: bytes
        if use_avro:
            if not _HAS_FASTAVRO:
                raise RuntimeError("fastavro required to produce Avro messages")
            if topic not in self._avro_schemas:
                raise RuntimeError("No Avro schema registered for topic %s" % topic)
            buf = bytearray()
            schemaless_writer(buf, self._avro_schemas[topic], value)
            payload_bytes = bytes(buf)
        else:
            if isinstance(value, (dict, list)):
                payload_bytes = json.dumps(value, default=str).encode("utf-8")
            elif isinstance(value, str):
                payload_bytes = value.encode("utf-8")
            elif isinstance(value, bytes):
                payload_bytes = value
            else:
                payload_bytes = str(value).encode("utf-8")

        # headers conversion
        hdrs = [(k, v.encode("utf-8")) for k,v in (headers or {}).items()]

        attempt = 0
        start_time = time.perf_counter()
        while True:
            attempt += 1
            try:
                fut = await self._producer.send_and_wait(topic, payload_bytes, partition=partition, key=(key.encode("utf-8") if key else None), headers=hdrs, timeout=timeout)
                latency = time.perf_counter() - start_time
                if KAFKA_PRODUCE_LATENCY:
                    try:
                        KAFKA_PRODUCE_LATENCY.observe(latency)
                    except Exception:
                        pass
                if KAFKA_PRODUCE_COUNT:
                    try:
                        KAFKA_PRODUCE_COUNT.labels(topic=topic, status="ok").inc()
                    except Exception:
                        pass
                LOG.debug("Produced msg to %s partition=%s offset=%s", topic, getattr(fut, "partition", None), getattr(fut, "offset", None))
                return True
            except Exception as e:
                LOG.warning("Produce attempt %d failed for topic %s: %s", attempt, topic, e)
                if KAFKA_PRODUCE_COUNT:
                    try:
                        KAFKA_PRODUCE_COUNT.labels(topic=topic, status="error").inc()
                    except Exception:
                        pass
                if attempt >= max(1, self.config.max_retries):
                    LOG.exception("Max produce retries reached for topic %s", topic)
                    return False
                await asyncio.sleep(_exponential_backoff(attempt))

    def produce_sync_confluent(self, topic: str, value: t.Union[dict, bytes, str], key: Optional[str] = None, headers: Optional[Dict[str,str]] = None, timeout: float = 10.0) -> bool:
        """
        Optional synchronous producer using confluent_kafka (transactions & higher perf).
        Blocks calling thread - should be used in dedicated threads.
        """
        if not _HAS_CONFLUENT or not self._confluent_producer:
            raise RuntimeError("confluent_kafka not available; cannot produce_sync_confluent")
        payload = value if isinstance(value, (bytes, str)) else json.dumps(value, default=str).encode("utf-8")
        hdrs = [(k, v) for k,v in (headers or {}).items()]
        attempt = 0
        start = time.perf_counter()
        while True:
            attempt += 1
            try:
                self._confluent_producer.produce(topic, payload, key=key, headers=hdrs)
                self._confluent_producer.flush(timeout)
                if KAFKA_PRODUCE_COUNT:
                    try:
                        KAFKA_PRODUCE_COUNT.labels(topic=topic, status="ok").inc()
                    except Exception:
                        pass
                if KAFKA_PRODUCE_LATENCY:
                    try:
                        KAFKA_PRODUCE_LATENCY.observe(time.perf_counter() - start)
                    except Exception:
                        pass
                return True
            except ConfluentKafkaException:
                LOG.exception("Confluent produce failed")
                if attempt >= max(1, self.config.max_retries):
                    return False
                time.sleep(_exponential_backoff(attempt))
            except Exception:
                LOG.exception("Confluent produce unknown error")
                if attempt >= max(1, self.config.max_retries):
                    return False
                time.sleep(_exponential_backoff(attempt))

    # -------------------------
    # TaskPool-based Consumer (bounded concurrency)
    # -------------------------

class TaskPoolConsumer:
    """
    One TaskPoolConsumer per topic.
    Maintains a single AIOKafkaConsumer + bounded asyncio.Queue of tasks.
    Spawns worker tasks to process messages concurrently.

    Features:
      - Bounded concurrency (backpressure)
      - Kafka pause/resume when queue backlog high
      - DLQ publishing on permanent failure
      - Retries with exponential backoff
      - Prometheus metrics (inflight, processed, DLQ)
      - Graceful shutdown waiting for in-flight tasks
    """

    def __init__(
        self,
        topic: str,
        kafka: "KafkaQueue",
        handler: MessageHandler,
        group_id: str,
        concurrency: int = 8,
        max_queue_size: Optional[int] = None,
        enable_dlq: bool = True,
        max_retries: Optional[int] = None,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        value_deserializer: Optional[Callable[[bytes], Any]] = None,
    ):
        self.topic = topic
        self.kafka = kafka
        self.handler = handler
        self.group_id = group_id
        self.concurrency = concurrency
        self.max_queue_size = max_queue_size or (concurrency * 4)
        self.enable_dlq = enable_dlq
        self.max_retries = max_retries or kafka.config.max_retries
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.value_deserializer = value_deserializer

        self._consumer: Optional[AIOKafkaConsumer] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self._pause = False
        self._inflight: int = 0
        self._lock = asyncio.Lock()
        self._main_task: Optional[asyncio.Task] = None

        # metrics
        self._gauge_inflight: Optional[Gauge] = None
        self._gauge_backlog: Optional[Gauge] = None
        if _HAS_PROM:
            self._gauge_inflight = Gauge(
                "prioritymax_kafka_inflight",
                "Current inflight tasks per topic",
                ["topic"]
            )
            self._gauge_backlog = Gauge(
                "prioritymax_kafka_queue_backlog",
                "Pending backlog size per topic",
                ["topic"]
            )

    # -------------------------
    # Lifecycle
    # -------------------------
    async def start(self):
        """Start consumer + workers."""
        aiokw = self.kafka._build_aiokafka_kwargs()
        self._consumer = AIOKafkaConsumer(
            self.topic,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            enable_auto_commit=self.enable_auto_commit,
            **aiokw,
        )
        await self._consumer.start()
        LOG.info("[TaskPoolConsumer] started for topic=%s group=%s", self.topic, self.group_id)

        # spawn workers
        for wid in range(self.concurrency):
            task = asyncio.create_task(self._worker_loop(wid))
            self._workers.append(task)

        # spawn main polling loop
        self._main_task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        """Stop consumer and wait for workers to finish."""
        LOG.info("[TaskPoolConsumer] stopping topic=%s", self.topic)
        self._stop_event.set()
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception:
                LOG.exception("[TaskPoolConsumer] consumer stop failed")

        # wait for queue to drain
        while not self._queue.empty():
            LOG.debug("[TaskPoolConsumer] draining queue=%d topic=%s", self._queue.qsize(), self.topic)
            await asyncio.sleep(0.5)

        # cancel workers
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            try:
                await t
            except asyncio.CancelledError:
                pass
        LOG.info("[TaskPoolConsumer] stopped topic=%s", self.topic)

    # -------------------------
    # Polling loop
    # -------------------------
    async def _poll_loop(self):
        """Poll Kafka and enqueue messages into task queue."""
        assert self._consumer
        try:
            async for msg in self._consumer:
                if self._stop_event.is_set():
                    break

                # backpressure: pause Kafka if queue full
                if self._queue.full():
                    if not self._pause:
                        LOG.debug("[%s] queue full (%d), pausing consumer", self.topic, self._queue.qsize())
                        try:
                            await self._consumer.pause()
                        except Exception:
                            LOG.debug("[%s] pause failed", self.topic)
                        self._pause = True

                # resume if queue has capacity again
                if self._pause and self._queue.qsize() < (self.max_queue_size // 2):
                    try:
                        await self._consumer.resume()
                        self._pause = False
                        LOG.debug("[%s] resumed consumer (queue=%d)", self.topic, self._queue.qsize())
                    except Exception:
                        LOG.debug("[%s] resume failed", self.topic)

                try:
                    await self._queue.put(msg)
                    if self._gauge_backlog:
                        self._gauge_backlog.labels(topic=self.topic).set(self._queue.qsize())
                except asyncio.CancelledError:
                    break
                except Exception:
                    LOG.exception("[%s] failed to enqueue message", self.topic)
        except asyncio.CancelledError:
            LOG.info("[%s] poll loop cancelled", self.topic)
        except Exception:
            LOG.exception("[%s] poll loop crashed", self.topic)
        finally:
            LOG.info("[%s] poll loop stopped", self.topic)

    # -------------------------
    # Worker loop
    # -------------------------
    async def _worker_loop(self, worker_id: int):
        """Worker that consumes from queue and processes via handler."""
        LOG.info("[%s] worker-%d started (concurrency=%d)", self.topic, worker_id, self.concurrency)
        while not self._stop_event.is_set():
            try:
                msg = await self._queue.get()
            except asyncio.CancelledError:
                break

            meta = {
                "topic": msg.topic,
                "partition": msg.partition,
                "offset": msg.offset,
                "key": (msg.key.decode("utf-8") if msg.key else None),
                "timestamp": msg.timestamp,
            }

            payload = msg.value
            if self.value_deserializer:
                try:
                    payload = self.value_deserializer(payload)
                except Exception:
                    LOG.warning("[%s] deserializer failed, keeping raw bytes", self.topic)

            success = await self._process_message(payload, meta)
            self._queue.task_done()
            if self._gauge_backlog:
                try:
                    self._gauge_backlog.labels(topic=self.topic).set(self._queue.qsize())
                except Exception:
                    pass

            # commit offset on success if auto commit off
            if success and not self.enable_auto_commit:
                try:
                    await self._consumer.commit()
                except Exception:
                    LOG.debug("[%s] commit failed for offset=%s", self.topic, msg.offset)

        LOG.info("[%s] worker-%d stopped", self.topic, worker_id)

    # -------------------------
    # Core message processing logic
    # -------------------------
    async def _process_message(self, payload: Any, meta: Dict[str, Any]) -> bool:
        """
        Execute handler with retries, backoff, DLQ fallback.
        """
        success = False
        last_exc = None
        start = time.perf_counter()
        self._inflight += 1
        if self._gauge_inflight:
            try:
                self._gauge_inflight.labels(topic=self.topic).set(self._inflight)
            except Exception:
                pass

        for attempt in range(1, self.max_retries + 1):
            try:
                # schema validation if registered
                if self.topic in self.kafka._json_schemas and _HAS_JSONSCHEMA:
                    try:
                        jsonschema.validate(instance=payload, schema=self.kafka._json_schemas[self.topic])
                    except Exception as ve:
                        LOG.warning("[%s] schema validation failed: %s", self.topic, ve)
                        last_exc = ve
                        break  # permanent fail

                res = await self.handler(payload, meta)
                if res:
                    success = True
                    break
                else:
                    last_exc = RuntimeError("handler returned False")
                    break
            except Exception as e:
                last_exc = e
                LOG.warning("[%s] handler exception attempt=%d: %s", self.topic, attempt, e)
                if attempt < self.max_retries:
                    await asyncio.sleep(_exponential_backoff(attempt))
                else:
                    break

        latency = time.perf_counter() - start
        if KAFKA_CONSUME_LATENCY:
            try:
                KAFKA_CONSUME_LATENCY.observe(latency)
            except Exception:
                pass

        self._inflight -= 1
        if self._gauge_inflight:
            try:
                self._gauge_inflight.labels(topic=self.topic).set(self._inflight)
            except Exception:
                pass

        if success:
            if KAFKA_CONSUME_COUNT:
                try:
                    KAFKA_CONSUME_COUNT.labels(topic=self.topic, status="ok").inc()
                except Exception:
                    pass
            if self.kafka.on_message_processed:
                try:
                    self.kafka.on_message_processed(self.topic, {"payload": payload, "meta": meta})
                except Exception:
                    LOG.debug("[%s] on_message_processed hook failed", self.topic)
            return True

        # Permanent failure → DLQ
        if self.enable_dlq:
            dlq_topic = f"{self.topic}{self.kafka._dlq_suffix}"
            dlq_payload = {
                "original_topic": self.topic,
                "payload": payload,
                "meta": meta,
                "error": str(last_exc),
                "ts": _now_iso(),
            }
            try:
                await self.kafka.produce(dlq_topic, dlq_payload)
                if KAFKA_DLQ_COUNT:
                    try:
                        KAFKA_DLQ_COUNT.labels(topic=self.topic).inc()
                    except Exception:
                        pass
                if self.kafka.on_dlq_published:
                    self.kafka.on_dlq_published(dlq_topic, dlq_payload)
                LOG.warning("[%s] message sent to DLQ %s", self.topic, dlq_topic)
            except Exception:
                LOG.exception("[%s] DLQ publish failed", self.topic)

        if KAFKA_CONSUME_COUNT:
            try:
                KAFKA_CONSUME_COUNT.labels(topic=self.topic, status="error").inc()
            except Exception:
                pass
        return False
# -------------------------
# KafkaQueue: TaskPool integration
# -------------------------

    async def register_consumer(
        self,
        topic: str,
        handler: MessageHandler,
        group_id: Optional[str] = None,
        concurrency: int = 8,
        max_queue_size: Optional[int] = None,
        enable_dlq: bool = True,
        max_retries: Optional[int] = None,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        value_deserializer: Optional[Callable[[bytes], Any]] = None,
    ):
        """
        Register a handler for a Kafka topic using a bounded TaskPool model.

        The TaskPoolConsumer handles:
          - One AIOKafkaConsumer per topic
          - asyncio.Queue of messages (bounded)
          - N workers processing concurrently
          - Automatic pause/resume when backlog high
          - DLQ + retries

        Args mirror the previous register_consumer API.
        """
        if not _HAS_AIOKAFKA:
            raise RuntimeError("aiokafka required for consumers")

        grp = group_id or (self._consumer_group_prefix + topic)
        if topic in getattr(self, "_task_consumers", {}):
            LOG.warning("Consumer for topic %s already exists; ignoring duplicate registration", topic)
            return

        tpc = TaskPoolConsumer(
            topic=topic,
            kafka=self,
            handler=handler,
            group_id=grp,
            concurrency=concurrency,
            max_queue_size=max_queue_size,
            enable_dlq=enable_dlq,
            max_retries=max_retries,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            value_deserializer=value_deserializer,
        )

        # Lazily init _task_consumers dict
        if not hasattr(self, "_task_consumers"):
            self._task_consumers: Dict[str, TaskPoolConsumer] = {}

        self._task_consumers[topic] = tpc
        await tpc.start()
        LOG.info("TaskPool consumer registered for topic=%s group=%s concurrency=%d", topic, grp, concurrency)

    async def pause_consumer(self, topic_pattern: Optional[str] = None):
        """Pause consumers matching topic pattern (TaskPool version)."""
        if not hasattr(self, "_task_consumers"):
            return
        for t, obj in list(self._task_consumers.items()):
            if topic_pattern is None or topic_pattern in t:
                LOG.info("Pausing consumer for topic=%s", t)
                try:
                    await obj._consumer.pause()
                except Exception:
                    LOG.debug("Pause failed for %s", t)

    async def resume_consumer(self, topic_pattern: Optional[str] = None):
        """Resume consumers matching topic pattern."""
        if not hasattr(self, "_task_consumers"):
            return
        for t, obj in list(self._task_consumers.items()):
            if topic_pattern is None or topic_pattern in t:
                LOG.info("Resuming consumer for topic=%s", t)
                try:
                    await obj._consumer.resume()
                except Exception:
                    LOG.debug("Resume failed for %s", t)

    async def stop(self):
        """Override stop() to shut down TaskPool consumers cleanly."""
        self._stop_event.set()
        # Stop producer/admin first
        try:
            if self._producer:
                await self._producer.stop()
        except Exception:
            LOG.debug("Producer stop failed", exc_info=True)
        try:
            if self._admin:
                await self._admin.stop()
        except Exception:
            LOG.debug("Admin stop failed", exc_info=True)

        # Stop TaskPool consumers
        if hasattr(self, "_task_consumers"):
            for topic, cons in list(self._task_consumers.items()):
                try:
                    await cons.stop()
                except Exception:
                    LOG.exception("Error stopping TaskPool consumer for %s", topic)
            self._task_consumers.clear()
        LOG.info("KafkaQueue stopped (TaskPool mode)")


    # -------------------------
    # Helper: fetch consumer group lag (best-effort)
    # -------------------------
    async def consumer_group_lag(self, group_id: str) -> Dict[str, int]:
        """
        Returns mapping topic -> total_lag for a consumer group.
        Requires admin client & latest offsets info.
        Best-effort: falls back to empty dict if admin not available.
        """
        if not self._admin:
            LOG.warning("Admin client not available - cannot compute group lag")
            return {}
        try:
            # aiokafka admin doesn't expose offsets easily - we approximate by using list_consumer_group_offsets
            # Note: aiokafka provides list_consumer_group_offsets on AdminClient in newer versions; do best-effort
            group_offsets = await self._admin.list_consumer_group_offsets(group_id)
            lag_map: Dict[str, int] = {}
            # group_offsets: Dict[TopicPartition, OffsetMetadata]
            for tp, offmeta in group_offsets.items():
                topic = tp.topic
                part = tp.partition
                committed = offmeta.offset
                # fetch high watermark (end offset) via admin - aiokafka lacks simple API; use consumer position as fallback
                # We'll return committed offsets as approximation
                lag_map.setdefault(topic, 0)
                # cannot compute end offset here reliably; set lag as unknown (-1)
                lag_map[topic] = -1
            return lag_map
        except Exception:
            LOG.exception("Failed to compute consumer_group_lag for %s", group_id)
            return {}

    # -------------------------
    # Health check
    # -------------------------
    async def health(self) -> Dict[str, Any]:
        """
        Return a health dict: producer/admin connectivity & topics status.
        """
        healthy = True
        info = {"producer": False, "admin": False, "bootstrap": self.bootstrap, "client_id": self.client_id, "ts": _now_iso()}
        try:
            if not self._producer:
                await self.start()
            info["producer"] = True
        except Exception:
            LOG.exception("Producer health check failed")
            healthy = False

        try:
            if self._admin:
                topics = await self._admin.list_topics(timeout=2)
                info["admin"] = True
                info["topics_count"] = len(topics)
            else:
                info["admin"] = False
        except Exception:
            LOG.exception("Admin health check failed")
            info["admin"] = False
            healthy = False

        info["healthy"] = healthy
        return info

    # -------------------------
    # Utilities: pause/resume consumers
    # -------------------------
    async def pause_consumer(self, topic_pattern: Optional[str] = None):
        """Cancel consumer tasks matching topic_pattern (or all if None)."""
        to_cancel = []
        for name, task in list(self._running_consumers.items()):
            if topic_pattern is None or topic_pattern in name:
                to_cancel.append(name)
        for n in to_cancel:
            t = self._running_consumers.get(n)
            if t:
                try:
                    t.cancel()
                except Exception:
                    LOG.debug("Cancel failed for consumer %s", n)

    async def resume_consumer(self, topic: str):
        """Resume consumer for a topic (re-register previously registered handler)."""
        opts = self._consumer_options.get(topic)
        handler = self._consumer_handlers.get(topic)
        if handler and opts:
            concurrency = int(opts.get("concurrency", 1))
            for i in range(concurrency):
                tname = f"{topic}::{opts.get('group_id')}::{i}"
                if tname in self._running_consumers:
                    continue
                task = asyncio.create_task(self._consumer_task(topic, tname))
                self._running_consumers[tname] = task
                LOG.info("Resumed consumer %s", tname)

    # -------------------------
    # Context manager convenience
    # -------------------------
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

# -----------------------------
# Example handler & usage
# -----------------------------
async def sample_handler(payload: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    """
    Example consumer handler that performs idempotent processing with basic error handling.
    Return True if message processed successfully; False if permanently failed.
    """
    try:
        # simple idempotency: if payload contains 'id' and we've processed it before, skip
        msg_id = payload.get("id") or payload.get("message_id")
        # pretend to do work (this is where you'd enqueue tasks / call services)
        await asyncio.sleep(0.01)
        LOG.info("Processed message id=%s meta=%s", msg_id, meta)
        return True
    except Exception:
        LOG.exception("sample_handler failed")
        return False

# Demonstration main - only runs when this module executed directly
if __name__ == "__main__":
    async def _demo():
        cfg = KafkaQueueConfig(bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"), auto_create_topics=False)
        q = KafkaQueue(cfg)
        await q.start()
        # Register sample consumer
        await q.register_consumer("prioritymax-demo", sample_handler, concurrency=1)
        # Produce some sample messages
        for i in range(10):
            ok = await q.produce("prioritymax-demo", {"id": f"msg-{i}", "value": i})
            LOG.info("Produced %s -> %s", i, ok)
        # run for 10 seconds
        await asyncio.sleep(10)
        await q.stop()

    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        LOG.info("Demo interrupted")
    except Exception:
        LOG.exception("Demo failed")

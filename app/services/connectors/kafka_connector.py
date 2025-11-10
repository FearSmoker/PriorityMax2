# backend/app/services/connector/kafka_connector.py
"""
PriorityMax Kafka Connector (production-ready)
----------------------------------------------
Provides a robust, dual-mode Kafka connector used by PriorityMax:
 - Async mode (preferred) using aiokafka for producers/consumers
 - Synchronous mode fallback using confluent-kafka (Producer/AdminClient)
 - Topic creation / admin helpers, schema registry integration (Avro), TLS/SASL support
 - Transactional producer support (best-effort when confluent is available)
 - Pluggable serializers (JSON, Avro stub, protobuf stub)
 - Partitioning helpers and sticky/consistent key hashing
 - Health checks, graceful startup/shutdown, and instrumentation hooks (Prometheus optional)
 - Integration points for TaskPoolConsumer and KafkaQueue modules in PriorityMax
 - CLI helpers for basic admin tasks

Notes:
 - This connector intentionally supports both aiokafka (asyncio) and confluent-kafka
   so you can run both async consumer loops and sync admin/produce tasks.
 - For Avro serialization, an existing Schema Registry client is expected (confluent or fastavro-based).
 - TLS/SASL credentials are read from environment for Kubernetes secret integration.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import math
import logging
import asyncio
import pathlib
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional libs
_HAS_AIOKAFKA = False
_HAS_CONFLUENT = False
_HAS_PROM = False
_HAS_FASTAVRO = False

try:
    import aiokafka
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer, AIOKafkaClient
    _HAS_AIOKAFKA = True
except Exception:
    aiokafka = None
    AIOKafkaProducer = AIOKafkaConsumer = AIOKafkaClient = None
    _HAS_AIOKAFKA = False

try:
    import confluent_kafka
    from confluent_kafka import Producer as ConfluentProducer, Consumer as ConfluentConsumer, KafkaError, KafkaException, admin as confluent_admin
    from confluent_kafka.admin import AdminClient, NewTopic
    _HAS_CONFLUENT = True
except Exception:
    confluent_kafka = None
    ConfluentProducer = ConfluentConsumer = AdminClient = NewTopic = None
    _HAS_CONFLUENT = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    Counter = Gauge = Histogram = None
    _HAS_PROM = False

try:
    from fastavro import parse_schema, schemaless_writer, schemaless_reader
    _HAS_FASTAVRO = True
except Exception:
    parse_schema = schemaless_writer = schemaless_reader = None
    _HAS_FASTAVRO = False

# Logging
LOG = logging.getLogger("prioritymax.connector.kafka")
LOG.setLevel(os.getenv("PRIORITYMAX_KAFKA_LOG", "INFO"))
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(handler)

# Defaults and env-driven config
DEFAULT_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")  # PLAINTEXT, SSL, SASL_SSL
DEFAULT_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
DEFAULT_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
DEFAULT_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
DEFAULT_SSL_CA = os.getenv("KAFKA_SSL_CA", "")  # path to CA cert (optional)
DEFAULT_SSL_CERT = os.getenv("KAFKA_SSL_CERT", "")
DEFAULT_SSL_KEY = os.getenv("KAFKA_SSL_KEY", "")
DEFAULT_CLIENT_ID = os.getenv("KAFKA_CLIENT_ID", f"prioritymax-{uuid.uuid4().hex[:6]}")
DEFAULT_CONSUMER_GROUP_PREFIX = os.getenv("KAFKA_CONSUMER_GROUP_PREFIX", "prioritymax-")
DEFAULT_METRICS_PORT = int(os.getenv("PRIORITYMAX_PROMETHEUS_PORT", "9002"))
DEFAULT_DEFAULT_REPLICATION = int(os.getenv("KAFKA_DEFAULT_REPLICATION", "1"))
DEFAULT_DEFAULT_PARTITIONS = int(os.getenv("KAFKA_DEFAULT_PARTITIONS", "3"))

# Prometheus metrics (optional)
if _HAS_PROM:
    KAFKA_PRODUCE_COUNT = Counter("prioritymax_kafka_produce_total", "Kafka produce count", ["topic", "status"])
    KAFKA_PRODUCE_LATENCY = Histogram("prioritymax_kafka_produce_seconds", "Produce latency seconds")
    KAFKA_CONSUME_COUNT = Counter("prioritymax_kafka_consume_total", "Kafka consume count", ["topic", "status"])
    KAFKA_DLQ_COUNT = Counter("prioritymax_kafka_dlq_total", "Kafka DLQ published count", ["topic"])
    KAFKA_TOPIC_EXISTS = Gauge("prioritymax_kafka_topic_exists", "Topic existence (1 if exists)", ["topic"])
    # start local metrics server optionally if env requests
    try:
        if os.getenv("PRIORITYMAX_PROMETHEUS_START", "false").lower() in ("1", "true", "yes"):
            start_http_server(DEFAULT_METRICS_PORT)
            LOG.info("Prometheus metrics served at port %d", DEFAULT_METRICS_PORT)
    except Exception:
        LOG.exception("Failed to start local prometheus http server")
else:
    KAFKA_PRODUCE_COUNT = KAFKA_PRODUCE_LATENCY = KAFKA_CONSUME_COUNT = KAFKA_DLQ_COUNT = KAFKA_TOPIC_EXISTS = None

# Type aliases
SerializerFn = Callable[[Any], bytes]
DeserializerFn = Callable[[bytes], Any]

# Utilities
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _uid() -> str:
    return uuid.uuid4().hex

# Simple JSON serializer/deserializer
def json_serializer(obj: Any) -> bytes:
    return json.dumps(obj, default=str).encode("utf-8")

def json_deserializer(b: bytes) -> Any:
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return b

# Consistent partitioner (hash key -> partition)
def default_partitioner(key: Optional[bytes], all_partitions: List[int]) -> int:
    """
    Simple consistent hash partitioner for byte keys.
    """
    if key is None:
        # round-robin fallback: use uuid
        return all_partitions[hash(_uid()) % len(all_partitions)]
    try:
        h = 0
        for c in key:
            h = (h * 31 + c) & 0xFFFFFFFF
        return all_partitions[h % len(all_partitions)]
    except Exception:
        return all_partitions[0]

# Avro serializer stub (fastavro)
class AvroSerializer:
    def __init__(self, schema: dict):
        if not _HAS_FASTAVRO:
            raise RuntimeError("fastavro is required for AvroSerializer")
        self.schema = parse_schema(schema)

    def dumps(self, obj: dict) -> bytes:
        from io import BytesIO
        b = BytesIO()
        schemaless_writer(b, self.schema, obj)
        return b.getvalue()

    def loads(self, raw: bytes) -> dict:
        from io import BytesIO
        b = BytesIO(raw)
        return schemaless_reader(b, self.schema)

# -----------------------------------------------------------------------------
# KafkaConnector: top-level class offering both async & sync clients
# -----------------------------------------------------------------------------
class KafkaConnector:
    """
    Unified Kafka connector that offers:
      - Async producer/consumer using aiokafka
      - Sync producer/admin using confluent_kafka (fallback)
      - Topic admin helpers
      - Pluggable serializers and partitioner
      - Health checks and graceful shutdown
    """

    def __init__(
        self,
        bootstrap_servers: str = DEFAULT_BOOTSTRAP,
        client_id: str = DEFAULT_CLIENT_ID,
        security_protocol: str = DEFAULT_SECURITY_PROTOCOL,
        sasl_mechanism: str = DEFAULT_SASL_MECHANISM,
        sasl_username: str = DEFAULT_SASL_USERNAME,
        sasl_password: str = DEFAULT_SASL_PASSWORD,
        ssl_cafile: Optional[str] = DEFAULT_SSL_CA,
        ssl_certfile: Optional[str] = DEFAULT_SSL_CERT,
        ssl_keyfile: Optional[str] = DEFAULT_SSL_KEY,
        default_partitions: int = DEFAULT_DEFAULT_PARTITIONS,
        default_replication: int = DEFAULT_DEFAULT_REPLICATION,
        serializer: SerializerFn = json_serializer,
        deserializer: DeserializerFn = json_deserializer,
        partitioner: Callable[[Optional[bytes], List[int]], int] = default_partitioner,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.security_protocol = security_protocol
        self.sasl_mechanism = sasl_mechanism
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.ssl_cafile = ssl_cafile or None
        self.ssl_certfile = ssl_certfile or None
        self.ssl_keyfile = ssl_keyfile or None
        self.default_partitions = default_partitions
        self.default_replication = default_replication
        self.serializer = serializer
        self.deserializer = deserializer
        self.partitioner = partitioner

        # Async clients (aiokafka)
        self._async_producer: Optional[AIOKafkaProducer] = None
        self._async_client: Optional[AIOKafkaClient] = None
        self._async_started = False

        # Sync clients (confluent)
        self._sync_producer: Optional[ConfluentProducer] = None
        self._admin_client: Optional[AdminClient] = None

        # internal state
        self._started = False
        self._lock = asyncio.Lock()
        self._loop = None

        # metrics hooks
        self.on_produce_ack: Optional[Callable[[str, int, int, Any], None]] = None  # topic, partition, offset, metadata
        self.on_consumer_error: Optional[Callable[[str, Exception], None]] = None

    # -------------------------
    # Config builders
    # -------------------------
    def _base_conf(self) -> Dict[str, Any]:
        conf = {"bootstrap.servers": self.bootstrap_servers, "client.id": self.client_id}
        # confluent style keys
        if self.security_protocol:
            conf["security.protocol"] = self.security_protocol
        if self.sasl_mechanism:
            conf["sasl.mechanism"] = self.sasl_mechanism
        if self.sasl_username:
            conf["sasl.username"] = self.sasl_username
        if self.sasl_password:
            conf["sasl.password"] = self.sasl_password
        # SSL files if provided
        if self.ssl_cafile:
            conf["ssl.ca.location"] = self.ssl_cafile
        if self.ssl_certfile:
            conf["ssl.certificate.location"] = self.ssl_certfile
        if self.ssl_keyfile:
            conf["ssl.key.location"] = self.ssl_keyfile
        return conf

    def _aiokafka_conf(self) -> Dict[str, Any]:
        # aiokafka uses bootstrap_servers and security protocol kwargs differently
        conf = {"bootstrap_servers": self.bootstrap_servers, "client_id": self.client_id}
        # SASL/SSL mapping
        if self.security_protocol in ("SSL", "SASL_SSL"):
            ssl_context = None
            try:
                import ssl
                ssl_context = ssl.create_default_context(cafile=self.ssl_cafile) if self.ssl_cafile else ssl.create_default_context()
                if self.ssl_certfile and self.ssl_keyfile:
                    ssl_context.load_cert_chain(self.ssl_certfile, keyfile=self.ssl_keyfile)
                conf["security_protocol"] = "SSL"
                conf["ssl_context"] = ssl_context
            except Exception:
                LOG.exception("Failed to build ssl context for aiokafka")
        if self.security_protocol in ("SASL_PLAINTEXT", "SASL_SSL"):
            conf["sasl_mechanism"] = self.sasl_mechanism
            conf["sasl_plain_username"] = self.sasl_username
            conf["sasl_plain_password"] = self.sasl_password
        return conf

    # -------------------------
    # Async startup/shutdown
    # -------------------------
    async def start_async(self):
        """
        Initialize the async Kafka producer (AIOKafkaProducer). This is safe to call multiple times.
        """
        if not _HAS_AIOKAFKA:
            LOG.warning("aiokafka not installed; async producer unavailable")
            return

        async with self._lock:
            if self._async_started:
                return
            self._loop = asyncio.get_running_loop()
            conf = self._aiokafka_conf()
            self._async_client = AIOKafkaClient(bootstrap_servers=conf.get("bootstrap_servers"))
            # Create producer with value_serializer wrapper
            self._async_producer = AIOKafkaProducer(
                loop=self._loop,
                **{k: v for k, v in conf.items() if k != "bootstrap_servers"},
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
            )
            await self._async_producer.start()
            self._async_started = True
            LOG.info("Async aiokafka producer started (bootstrap=%s client=%s)", self.bootstrap_servers, self.client_id)

    async def stop_async(self):
        async with self._lock:
            if self._async_producer and self._async_started:
                try:
                    await self._async_producer.stop()
                except Exception:
                    LOG.exception("Failed to stop async producer")
            self._async_started = False
            LOG.info("Async aiokafka producer stopped")

    # -------------------------
    # Sync startup for admin & transactional produce
    # -------------------------
    def start_sync(self):
        """
        Initialize confluent-kafka AdminClient and Producer if available. This is synchronous and intended for admin tasks
        and optionally for a high-performance sync producer.
        """
        if not _HAS_CONFLUENT:
            LOG.warning("confluent-kafka not installed; sync admin/unified produce unavailable")
            return

        if self._admin_client:
            return
        conf = self._base_conf()
        try:
            self._admin_client = AdminClient(conf)
            LOG.info("Confluent AdminClient initialized")
        except Exception:
            LOG.exception("Failed to create AdminClient")

        try:
            # Producer config: enable.idempotence optional
            pconf = conf.copy()
            pconf.setdefault("enable.idempotence", True)
            self._sync_producer = ConfluentProducer(pconf)
            LOG.info("Confluent producer initialized")
        except Exception:
            LOG.exception("Failed to create Confluent Producer")

    def stop_sync(self):
        try:
            if self._sync_producer:
                self._sync_producer.flush(5.0)
        except Exception:
            LOG.exception("Failed to flush sync producer")
        self._sync_producer = None
        self._admin_client = None

    # -------------------------
    # Topic admin helpers
    # -------------------------
    def topic_exists_sync(self, topic: str, timeout: float = 5.0) -> bool:
        """
        Check if topic exists using AdminClient metadata (sync).
        """
        if not _HAS_CONFLUENT or not self._admin_client:
            LOG.warning("confluent admin not available to check topic existence")
            return False
        try:
            md = self._admin_client.list_topics(timeout=timeout)
            exists = topic in md.topics
            if KAFKA_TOPIC_EXISTS:
                try:
                    KAFKA_TOPIC_EXISTS.labels(topic=topic).set(1 if exists else 0)
                except Exception:
                    pass
            return exists
        except Exception:
            LOG.exception("topic_exists_sync failed for %s", topic)
            return False

    def create_topic_sync(self, topic: str, num_partitions: Optional[int] = None, replication: Optional[int] = None, config: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Create topic using AdminClient.NewTopic. Returns a dict with results per topic partition.
        """
        if not _HAS_CONFLUENT:
            raise RuntimeError("confluent-kafka required for create_topic_sync")
        num_partitions = num_partitions or self.default_partitions
        replication = replication or self.default_replication
        new_topic = NewTopic(topic, num_partitions=num_partitions, replication_factor=replication, config=config or {})
        fs = self._admin_client.create_topics([new_topic], request_timeout=timeout)
        res = {}
        for t, f in fs.items():
            try:
                f.result()
                res[t] = {"ok": True}
                LOG.info("Created topic %s", t)
            except Exception as e:
                LOG.exception("Failed to create topic %s: %s", t, e)
                res[t] = {"ok": False, "error": str(e)}
        return res

    def delete_topic_sync(self, topic: str, timeout: float = 30.0) -> Dict[str, Any]:
        if not _HAS_CONFLUENT:
            raise RuntimeError("confluent-kafka required for delete_topic_sync")
        fs = self._admin_client.delete_topics([topic], operation_timeout=timeout)
        res = {}
        for t, f in fs.items():
            try:
                f.result()
                res[t] = {"ok": True}
                LOG.info("Deleted topic %s", t)
            except Exception as e:
                LOG.exception("Failed to delete topic %s: %s", t, e)
                res[t] = {"ok": False, "error": str(e)}
        return res

    # -------------------------
    # Async produce (aiokafka)
    # -------------------------
    async def produce_async(
        self,
        topic: str,
        value: Any,
        key: Optional[Union[str, bytes]] = None,
        partition: Optional[int] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
        timestamp_ms: Optional[int] = None,
        serializer: Optional[SerializerFn] = None,
        partitioner: Optional[Callable[[Optional[bytes], List[int]], int]] = None,
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Produce a single message asynchronously using aiokafka.
        Returns (ok, partition, offset)
        """
        if not _HAS_AIOKAFKA:
            raise RuntimeError("aiokafka not installed for async produce")
        if not self._async_started or not self._async_producer:
            await self.start_async()

        ser = serializer or self.serializer
        part_fn = partitioner or self.partitioner

        val_bytes = ser(value) if not isinstance(value, (bytes, bytearray)) else value
        key_bytes = key.encode("utf-8") if isinstance(key, str) else key

        t0 = time.perf_counter()
        try:
            # Aiokafka allows specifying partition directly. If partition is None, leave it to kafka.
            fut = await self._async_producer.send_and_wait(topic, val_bytes, key=key_bytes, partition=partition, timestamp_ms=timestamp_ms, headers=headers)
            # send_and_wait returns RecordMetadata namedtuple (topic, partition, offset)
            partition_res = getattr(fut, "partition", None) or (fut[1] if isinstance(fut, tuple) else None)
            offset_res = getattr(fut, "offset", None) or (fut[2] if isinstance(fut, tuple) else None)
            elapsed = time.perf_counter() - t0
            if KAFKA_PRODUCE_LATENCY:
                try:
                    KAFKA_PRODUCE_LATENCY.observe(elapsed)
                except Exception:
                    pass
            if KAFKA_PRODUCE_COUNT:
                try:
                    KAFKA_PRODUCE_COUNT.labels(topic=topic, status="ok").inc()
                except Exception:
                    pass
            if self.on_produce_ack:
                try:
                    self.on_produce_ack(topic, partition_res, offset_res, None)
                except Exception:
                    LOG.debug("on_produce_ack hook failed", exc_info=True)
            return True, partition_res, offset_res
        except Exception as e:
            LOG.exception("produce_async failed for topic=%s", topic)
            if KAFKA_PRODUCE_COUNT:
                try:
                    KAFKA_PRODUCE_COUNT.labels(topic=topic, status="error").inc()
                except Exception:
                    pass
            if self.on_produce_ack:
                try:
                    self.on_produce_ack(topic, None, None, e)
                except Exception:
                    LOG.debug("on_produce_ack hook failed", exc_info=True)
            return False, None, None

    # -------------------------
    # Sync produce (confluent)
    # -------------------------
    def produce_sync(
        self,
        topic: str,
        value: Any,
        key: Optional[Union[str, bytes]] = None,
        partition: Optional[int] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
        serializer: Optional[SerializerFn] = None,
        on_delivery: Optional[Callable[[Any, Any], None]] = None,
        timeout: float = 10.0,
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Produce using confluent_kafka. Returns (ok, partition, offset)
        """
        if not _HAS_CONFLUENT:
            raise RuntimeError("confluent-kafka not installed for sync produce")
        if not self._sync_producer:
            self.start_sync()
        ser = serializer or self.serializer
        val_bytes = ser(value) if not isinstance(value, (bytes, bytearray)) else value
        key_bytes = key.encode("utf-8") if isinstance(key, str) else key

        delivered = {"ok": False, "partition": None, "offset": None, "error": None}
        def _delivery(err, msg):
            if err is not None:
                delivered["ok"] = False
                delivered["error"] = err
            else:
                delivered["ok"] = True
                delivered["partition"] = msg.partition()
                delivered["offset"] = msg.offset()
            if on_delivery:
                try:
                    on_delivery(err, msg)
                except Exception:
                    LOG.exception("on_delivery hook failed")

        try:
            self._sync_producer.produce(topic=topic, value=val_bytes, key=key_bytes, partition=partition if partition is not None else None, headers=headers, callback=_delivery)
            self._sync_producer.poll(0)  # serve callback
            self._sync_producer.flush(timeout)
            if KAFKA_PRODUCE_COUNT:
                try:
                    KAFKA_PRODUCE_COUNT.labels(topic=topic, status="ok" if delivered["ok"] else "error").inc()
                except Exception:
                    pass
            return delivered["ok"], delivered["partition"], delivered["offset"]
        except KafkaException as e:
            LOG.exception("produce_sync KafkaException for topic=%s", topic)
            if KAFKA_PRODUCE_COUNT:
                try:
                    KAFKA_PRODUCE_COUNT.labels(topic=topic, status="error").inc()
                except Exception:
                    pass
            return False, None, None
        except Exception:
            LOG.exception("produce_sync failed")
            return False, None, None

    # -------------------------
    # Transactional produce (best-effort)
    # -------------------------
    def produce_transactional_sync(self, topic: str, messages: List[Tuple[Optional[bytes], bytes]], transactional_id: Optional[str] = None, timeout: float = 30.0) -> bool:
        """
        Produce a batch of messages transactionally (requires confluent-kafka and broker support).
        messages: list of (key_bytes, value_bytes)
        """
        if not _HAS_CONFLUENT:
            raise RuntimeError("confluent-kafka required for transactional produce")
        if not self._sync_producer:
            self.start_sync()
        transactional_id = transactional_id or f"{self.client_id}-{_uid()}"
        # confluent python uses producer.init_transactions / begin_transaction / commit_transaction
        try:
            self._sync_producer.init_transactions(timeout=timeout)
            self._sync_producer.begin_transaction()
            for key_bytes, val_bytes in messages:
                self._sync_producer.produce(topic=topic, key=key_bytes, value=val_bytes)
            self._sync_producer.flush(timeout)
            self._sync_producer.commit_transaction(timeout)
            LOG.info("Transactional produce committed for topic=%s count=%d", topic, len(messages))
            return True
        except Exception:
            LOG.exception("Transactional produce failed, attempting abort")
            try:
                self._sync_producer.abort_transaction(timeout)
            except Exception:
                LOG.exception("Transactional abort failed")
            return False

    # -------------------------
    # Consumer helper (async)
    # -------------------------
    async def create_async_consumer(
        self,
        topics: List[str],
        group_id: Optional[str] = None,
        enable_auto_commit: bool = False,
        value_deserializer: Optional[DeserializerFn] = None,
        key_deserializer: Optional[DeserializerFn] = None,
        **aiokafka_kwargs,
    ) -> AIOKafkaConsumer:
        """
        Create and return an AIOKafkaConsumer configured with security settings and deserializers.
        Caller is responsible for starting/stopping the consumer.
        """
        if not _HAS_AIOKAFKA:
            raise RuntimeError("aiokafka required for async consumer")
        conf = self._aiokafka_conf()
        # prepare kwargs for AIOKafkaConsumer - pass group_id, enable_auto_commit etc.
        consumer = AIOKafkaConsumer(
            *topics,
            loop=asyncio.get_running_loop(),
            group_id=group_id or (DEFAULT_CONSUMER_GROUP_PREFIX + topics[0]),
            enable_auto_commit=enable_auto_commit,
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
            **aiokafka_kwargs,
            **{k: v for k, v in conf.items() if k not in ("bootstrap_servers",)},
        )
        # attach deserializer helpers (we'll wrap in TaskPoolConsumer so it can use value_deserializer)
        if value_deserializer:
            consumer._value_deserializer = value_deserializer  # best-effort attach
        if key_deserializer:
            consumer._key_deserializer = key_deserializer
        return consumer

    # -------------------------
    # Consumer helper (sync) - confluent consumer
    # -------------------------
    def create_sync_consumer(
        self,
        topics: List[str],
        group_id: Optional[str] = None,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        value_deserializer: Optional[DeserializerFn] = None,
    ) -> ConfluentConsumer:
        if not _HAS_CONFLUENT:
            raise RuntimeError("confluent-kafka required for sync consumer")
        conf = self._base_conf()
        conf.update({
            "group.id": group_id or (DEFAULT_CONSUMER_GROUP_PREFIX + (topics[0] if topics else "default")),
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": enable_auto_commit,
        })
        consumer = ConfluentConsumer(conf)
        consumer.subscribe(topics)
        # attach deserializer attributes for best-effort integration
        consumer._value_deserializer = value_deserializer
        return consumer

    # -------------------------
    # Health / status helpers
    # -------------------------
    def health_sync(self) -> Dict[str, Any]:
        """
        Synchronous health: checks admin client metadata and optionally broker reachability.
        """
        res = {"ok": False, "brokers": [], "topics": {}, "ts": _now_iso()}
        try:
            if _HAS_CONFLUENT and self._admin_client:
                md = self._admin_client.list_topics(timeout=5)
                res["brokers"] = [str(b) for b in md.brokers.values()] if hasattr(md, "brokers") else []
                # topics: partitions etc
                for tname, tmeta in md.topics.items():
                    res["topics"][tname] = {"partitions": len(tmeta.partitions) if getattr(tmeta, "partitions", None) else 0}
                res["ok"] = True
        except Exception:
            LOG.exception("health_sync failed")
            res["ok"] = False
        return res

    async def health_async(self) -> Dict[str, Any]:
        """
        Async health: attempt to start a short-lived producer/consumer or check metadata via aiokafka client.
        """
        res = {"ok": False, "ts": _now_iso()}
        if not _HAS_AIOKAFKA:
            res["ok"] = False
            return res
        try:
            # quick metadata fetch using AIOKafkaClient
            client = AIOKafkaClient(bootstrap_servers=self.bootstrap_servers, client_id=self.client_id)
            await client.bootstrap()
            md = await client.cluster_metadata()
            # cluster_metadata returns object mapping brokers & topics - best-effort parsing
            res["topics"] = {k: {"partitions": len(v.partitions) if hasattr(v, "partitions") else None} for k, v in md.topics.items()}
            await client.close()
            res["ok"] = True
        except Exception:
            LOG.exception("health_async failed")
            res["ok"] = False
        return res

    # -------------------------
    # Utility: serialize a batch (vectorized) for heavy batch produce
    # -------------------------
    def batch_serialize(self, messages: List[Tuple[Optional[Union[str, bytes]], Any]], serializer: Optional[SerializerFn] = None) -> List[Tuple[Optional[bytes], bytes]]:
        ser = serializer or self.serializer
        out = []
        for key, val in messages:
            keyb = key.encode("utf-8") if isinstance(key, str) else (key if key is None or isinstance(key, (bytes, bytearray)) else str(key).encode("utf-8"))
            valb = val if isinstance(val, (bytes, bytearray)) else ser(val)
            out.append((keyb, valb))
        return out

    # -------------------------
    # CLI helpers & utilities
    # -------------------------
    def _print_md(self, d: Dict[str, Any]):
        print(json.dumps(d, indent=2, default=str))

def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-kafka-connector")
    sub = p.add_subparsers(dest="cmd")

    health = sub.add_parser("health")
    health.add_argument("--mode", choices=["sync", "async"], default="sync")

    create = sub.add_parser("create-topic")
    create.add_argument("--topic", required=True)
    create.add_argument("--partitions", type=int, default=DEFAULT_DEFAULT_PARTITIONS)
    create.add_argument("--replication", type=int, default=DEFAULT_DEFAULT_REPLICATION)

    delete = sub.add_parser("delete-topic")
    delete.add_argument("--topic", required=True)

    produce = sub.add_parser("produce")
    produce.add_argument("--topic", required=True)
    produce.add_argument("--message", required=True)
    produce.add_argument("--key", default=None)

    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    conn = KafkaConnector()
    if args.cmd == "health":
        if args.mode == "sync":
            print(conn.health_sync())
        else:
            import asyncio
            print(asyncio.run(conn.health_async()))
    elif args.cmd == "create-topic":
        conn.start_sync()
        print(conn.create_topic_sync(args.topic, num_partitions=args.partitions, replication=args.replication))
    elif args.cmd == "delete-topic":
        conn.start_sync()
        print(conn.delete_topic_sync(args.topic))
    elif args.cmd == "produce":
        conn.start_sync()
        ok, p, o = conn.produce_sync(args.topic, json.loads(args.message) if args.message else None, key=args.key)
        print("produce:", ok, p, o)
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

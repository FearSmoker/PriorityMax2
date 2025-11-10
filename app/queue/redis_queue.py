# backend/app/queue/redis_queue.py
"""
PriorityMax Redis Queue (Streams + TaskPool consumer, production-ready)

Features:
 - Async Redis client (redis.asyncio or aioredis fallback)
 - Redis Streams + Consumer Groups primary mechanism (XADD, XREADGROUP, XACK)
 - Fallback list queue helpers (LPUSH / BRPOP) for simple setups
 - TaskPool consumer with bounded asyncio.Queue + worker pool per stream/consumer group
 - Dead-letter stream (DLQ) when messages exceed retries or permanent failure
 - Retries with exponential backoff + jitter
 - Prometheus metrics hooks (optional)
 - Admin helpers: create group, trim stream, read pending entries, claim stuck messages
 - Consumer-group lag/pending inspection helpers
 - Health checks and graceful shutdown
 - Hooks for instrumentation (on_processed, on_error, on_dlq)
 - CLI helper at the bottom for quick dev testing
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
import datetime
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Tuple, Coroutine

# Prefer redis.asyncio from redis-py 4.x
_HAS_REDIS_ASYNCIO = False
_HAS_OLD_AIOREDIS = False

try:
    import redis.asyncio as redis_async
    from redis.asyncio.client import Redis as RedisClient
    _HAS_REDIS_ASYNCIO = True
except Exception:
    redis_async = None
    RedisClient = None
    _HAS_REDIS_ASYNCIO = False
    try:
        import aioredis
        _HAS_OLD_AIOREDIS = True
    except Exception:
        aioredis = None
        _HAS_OLD_AIOREDIS = False

# Optional libs
_HAS_JSONSCHEMA = False
_HAS_PROM = False
try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except Exception:
    jsonschema = None

try:
    from prometheus_client import Counter, Gauge, Histogram
    _HAS_PROM = True
except Exception:
    Counter = Gauge = Histogram = None

# Logging
LOG = logging.getLogger("prioritymax.queue.redis")
LOG.setLevel(os.getenv("PRIORITYMAX_REDIS_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_STREAM_MAXLEN = int(os.getenv("PRIORITYMAX_STREAM_MAXLEN", "100000"))
DEFAULT_CONSUMER_PREFIX = os.getenv("PRIORITYMAX_REDIS_CONSUMER_PREFIX", "prioritymax-consumer-")
DEFAULT_MAX_RETRIES = int(os.getenv("PRIORITYMAX_REDIS_MAX_RETRIES", "5"))
DEFAULT_BLOCK_MS = int(os.getenv("PRIORITYMAX_REDIS_BLOCK_MS", "5000"))
DEFAULT_CLAIM_IDLE_MS = int(os.getenv("PRIORITYMAX_REDIS_CLAIM_IDLE_MS", "60000"))  # claim messages idle > 60s

# Prometheus metrics
if _HAS_PROM:
    REDIS_XADD_COUNT = Counter("prioritymax_redis_xadd_total", "Total xadd (stream) messages produced", ["stream", "status"])
    REDIS_CONSUME_COUNT = Counter("prioritymax_redis_consume_total", "Total stream messages processed", ["stream", "status"])
    REDIS_DLQ_COUNT = Counter("prioritymax_redis_dlq_total", "Total messages sent to DLQ", ["stream"])
    REDIS_INFLIGHT = Gauge("prioritymax_redis_inflight", "Current inflight tasks per stream", ["stream"])
    REDIS_BACKLOG = Gauge("prioritymax_redis_backlog", "Pending backlog size per stream (consumer group)", ["stream"])
    REDIS_XADD_LATENCY = Histogram("prioritymax_redis_xadd_seconds", "Latency for xadd", buckets=[0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1])
else:
    REDIS_XADD_COUNT = REDIS_CONSUME_COUNT = REDIS_DLQ_COUNT = REDIS_INFLIGHT = REDIS_BACKLOG = REDIS_XADD_LATENCY = None

# Types
MessageHandler = Callable[[Dict[str, Any], Dict[str, Any]], Coroutine[Any, Any, bool]]
# handler(payload, meta) -> bool (True = processed successfully)

# Utils
def _now_iso() -> str:
    return datetime.datetime.datetime.utcnow().isoformat() + "Z"

def _gen_id() -> str:
    return uuid.uuid4().hex

def _safe_json_loads(b: Any) -> Any:
    if isinstance(b, (bytes, bytearray)):
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

def _exponential_backoff(attempt: int, base: float = 0.2, cap: float = 30.0, jitter: bool = True):
    exp = base * (2 ** (attempt - 1))
    sleep = min(exp, cap)
    if jitter:
        sleep = sleep * (1.0 + (random.random() - 0.5) * 0.2)
    return sleep

# -------------------------
# Config dataclass
# -------------------------
@dataclass
class RedisQueueConfig:
    redis_url: str = DEFAULT_REDIS_URL
    stream_maxlen: int = DEFAULT_STREAM_MAXLEN
    consumer_name_prefix: str = DEFAULT_CONSUMER_PREFIX
    claim_idle_ms: int = DEFAULT_CLAIM_IDLE_MS
    block_ms: int = DEFAULT_BLOCK_MS
    max_retries: int = DEFAULT_MAX_RETRIES
    read_count: int = 10  # XREADGROUP COUNT
    trim_maxlen: Optional[int] = None  # optional stream trim
    # stream naming suffixes
    dlq_suffix: str = ".dlq"
    # fallback mode: "streams" or "lists"
    mode: str = "streams"

# -------------------------
# Low-level Redis client wrapper
# -------------------------
class RedisClientWrapper:
    def __init__(self, url: str = DEFAULT_REDIS_URL):
        self.url = url
        self._client: Optional[RedisClient] = None
        self._aioredis = None
        self._using_old = False

    async def start(self):
        if _HAS_REDIS_ASYNCIO:
            self._client = redis_async.from_url(self.url, decode_responses=False)
            LOG.info("Connected to redis.asyncio at %s", self.url)
        elif _HAS_OLD_AIOREDIS:
            # aioredis.create_redis_pool
            self._using_old = True
            self._aioredis = await aioredis.create_redis_pool(self.url)
            LOG.info("Connected to aioredis at %s", self.url)
        else:
            raise RuntimeError("No async redis client available. Install 'redis' (>=4.x) or 'aioredis'.")

    async def stop(self):
        try:
            if self._client:
                await self._client.close()
            if self._aioredis:
                self._aioredis.close()
                await self._aioredis.wait_closed()
        except Exception:
            LOG.exception("Redis client stop failed")

    # Stream commands (wrap redis-py and aioredis differences)
    async def xadd(self, stream: str, fields: Dict[str, Any], maxlen: Optional[int] = None, approximate: bool = True) -> str:
        if _HAS_REDIS_ASYNCIO and self._client:
            kwargs = {}
            if maxlen:
                kwargs["maxlen"] = maxlen
                if approximate:
                    kwargs["approximate"] = True
            # Redis expects mapping of bytes
            mapping = {k: (json.dumps(v) if not isinstance(v, (bytes, bytearray)) else v) for k, v in fields.items()}
            # redis-py returns bytes id or str depending decode_responses; we set decode_responses=False -> bytes
            idb = await self._client.xadd(stream, mapping, **kwargs)
            return idb.decode() if isinstance(idb, (bytes, bytearray)) else str(idb)
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            # aioredis xadd returns id as bytes
            mapping = {k: (json.dumps(v) if not isinstance(v, (bytes, bytearray)) else v) for k, v in fields.items()}
            idb = await self._aioredis.xadd(stream, mapping, max_len=maxlen)
            return idb.decode() if isinstance(idb, (bytes, bytearray)) else str(idb)
        else:
            raise RuntimeError("No redis client available for xadd")

    async def xgroup_create(self, stream: str, group: str, id: str = "$", mkstream: bool = True):
        if _HAS_REDIS_ASYNCIO and self._client:
            try:
                await self._client.xgroup_create(stream, group, id, mkstream=mkstream)
            except Exception as e:
                # group may already exist
                if "BUSYGROUP" in str(e).upper():
                    return
                raise
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            try:
                await self._aioredis.xgroup_create(stream, group, id, mkstream=mkstream)
            except Exception as e:
                if "BUSYGROUP" in str(e).upper():
                    return
                raise
        else:
            raise RuntimeError("No redis client available for xgroup_create")

    async def xreadgroup(self, group: str, consumer: str, streams: Dict[str, str], count: int = 10, block_ms: int = 5000):
        if _HAS_REDIS_ASYNCIO and self._client:
            res = await self._client.xreadgroup(groupname=group, consumername=consumer, streams=streams, count=count, block=block_ms)
            # returns list of (stream, [(id, {k: v}), ...])
            return res
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.xread_group(group, consumer, streams, count=count, block=block_ms)
        else:
            raise RuntimeError("No redis client available for xreadgroup")

    async def xack(self, stream: str, group: str, *ids: str):
        if _HAS_REDIS_ASYNCIO and self._client:
            return await self._client.xack(stream, group, *ids)
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.xack(stream, group, *ids)
        else:
            raise RuntimeError("No redis client available for xack")

    async def xpending(self, stream: str, group: str, start: str = "-", end: str = "+", count: int = 10):
        if _HAS_REDIS_ASYNCIO and self._client:
            return await self._client.xpending(stream, group, start=start, end=end, count=count)
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.xpending(stream, group, start, end, count)
        else:
            raise RuntimeError("No redis client available for xpending")

    async def xclaim(self, stream: str, group: str, consumer: str, min_idle_ms: int, *ids: str):
        if _HAS_REDIS_ASYNCIO and self._client:
            return await self._client.xclaim(stream, group, consumer, min_idle_ms, *ids)
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.xclaim(stream, group, consumer, min_idle_ms, *ids)
        else:
            raise RuntimeError("No redis client available for xclaim")

    async def xlen(self, stream: str) -> int:
        if _HAS_REDIS_ASYNCIO and self._client:
            return await self._client.xlen(stream)
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.xlen(stream)
        else:
            raise RuntimeError("No redis client available for xlen")

    async def xrange(self, stream: str, start: str = "-", end: str = "+", count: int = None):
        if _HAS_REDIS_ASYNCIO and self._client:
            return await self._client.xrange(stream, start=start, end=end, count=count)
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.xrange(stream, start, end, count)
        else:
            raise RuntimeError("No redis client available for xrange")

    async def delete(self, *keys: str):
        if _HAS_REDIS_ASYNCIO and self._client:
            return await self._client.delete(*keys)
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.delete(*keys)
        else:
            raise RuntimeError("No redis client available for delete")

    # list-based fallback helpers
    async def lpush(self, key: str, value: Any):
        if _HAS_REDIS_ASYNCIO and self._client:
            return await self._client.lpush(key, json.dumps(value))
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            return await self._aioredis.lpush(key, json.dumps(value))
        else:
            raise RuntimeError("No redis client available for lpush")

    async def brpop(self, key: str, timeout: int = 0):
        if _HAS_REDIS_ASYNCIO and self._client:
            res = await self._client.brpop(key, timeout=timeout)
            return res
        elif _HAS_OLD_AIOREDIS and self._aioredis:
            res = await self._aioredis.brpop(key, timeout=timeout)
            return res
        else:
            raise RuntimeError("No redis client available for brpop")

# -------------------------
# TaskPool Consumer (Streams + Groups)
# -------------------------
class StreamTaskPoolConsumer:
    """
    One StreamTaskPoolConsumer per (stream, consumer_group, consumer_name).
    Uses Redis Streams + Consumer Groups + a bounded asyncio.Queue + worker pool per consumer.
    """

    def __init__(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: Optional[str],
        redis: RedisClientWrapper,
        handler: MessageHandler,
        concurrency: int = 8,
        max_queue_size: Optional[int] = None,
        enable_dlq: bool = True,
        max_retries: Optional[int] = None,
        read_count: int = 10,
        block_ms: int = DEFAULT_BLOCK_MS,
        claim_idle_ms: int = DEFAULT_CLAIM_IDLE_MS,
        json_deserializer: Optional[Callable[[bytes], Any]] = None,
    ):
        self.stream = stream_name
        self.group = group_name
        self.consumer = consumer_name or (DEFAULT_CONSUMER_PREFIX + _gen_id()[:8])
        self.redis = redis
        self.handler = handler
        self.concurrency = concurrency
        self.max_queue_size = max_queue_size or (concurrency * 4)
        self.enable_dlq = enable_dlq
        self.max_retries = max_retries or DEFAULT_MAX_RETRIES
        self.read_count = read_count
        self.block_ms = block_ms
        self.claim_idle_ms = claim_idle_ms
        self.json_deserializer = json_deserializer

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._poll_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._inflight = 0
        self._lock = asyncio.Lock()

        # metrics labels
        self._labels = {"stream": self.stream}
        if _HAS_PROM:
            self._gauge_inflight = REDIS_INFLIGHT.labels(stream=self.stream)
            self._gauge_backlog = REDIS_BACKLOG.labels(stream=self.stream)
        else:
            self._gauge_inflight = self._gauge_backlog = None

    async def start(self, create_group_if_missing: bool = True):
        # ensure group exists
        try:
            if create_group_if_missing:
                await self.redis.xgroup_create(self.stream, self.group, id="$", mkstream=True)
        except Exception:
            # group may exist already — ignore
            LOG.debug("xgroup_create warning/ignored for %s/%s", self.stream, self.group)
        # start worker tasks
        for i in range(self.concurrency):
            t = asyncio.create_task(self._worker_loop(i))
            self._workers.append(t)
        # start poll loop
        self._poll_task = asyncio.create_task(self._poll_loop())
        LOG.info("Started StreamTaskPoolConsumer stream=%s group=%s consumer=%s concurrency=%d", self.stream, self.group, self.consumer, self.concurrency)

    async def stop(self):
        LOG.info("Stopping StreamTaskPoolConsumer %s/%s", self.stream, self.group)
        self._stop_event.set()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        # wait for queue to drain
        drain_wait = 0
        while not self._queue.empty() and drain_wait < 30:
            LOG.debug("Waiting for queue to drain (%d) for %s", self._queue.qsize(), self.stream)
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
                LOG.debug("Worker task exception on stop", exc_info=True)
        LOG.info("Stopped StreamTaskPoolConsumer %s/%s", self.stream, self.group)

    async def _poll_loop(self):
        """
        Poll XREADGROUP in a loop and enqueue messages into self._queue
        Implement backpressure: if queue full, sleep (do not read); claim pending stuck messages periodically.
        """
        while not self._stop_event.is_set():
            try:
                # backpressure check
                if self._queue.full():
                    LOG.debug("[%s] backlog full=%d sleeping", self.stream, self._queue.qsize())
                    # attempt to claim stuck pending entries as well
                    await asyncio.sleep(0.1)
                    continue

                # block until messages or timeout
                streams = {self.stream: ">"}
                res = await self.redis.xreadgroup(self.group, self.consumer, streams, count=self.read_count, block=self.block_ms)
                if not res:
                    # maybe there are pending messages owned by others that are idle — attempt to claim them
                    await self._claim_stuck_entries()
                    continue

                # res: list [(stream, [(id, {k: v}), ...]), ...]
                for stream, items in res:
                    # stream and items may be bytes or str depending on client; normalize
                    for item_id, data in items:
                        try:
                            # prepare payload: Redis returns mapping of field->bytes
                            payload = {}
                            for k, v in data.items():
                                # each v is bytes; try JSON decode
                                try:
                                    payload[k.decode() if isinstance(k, bytes) else k] = json.loads(v.decode() if isinstance(v, (bytes, bytearray)) else v)
                                except Exception:
                                    payload[k.decode() if isinstance(k, bytes) else k] = v.decode() if isinstance(v, (bytes, bytearray)) else v
                            meta = {"id": item_id.decode() if isinstance(item_id, (bytes, bytearray)) else str(item_id)}
                        except Exception:
                            payload = {b"k": data}
                            meta = {"id": item_id}
                        # enqueue message (block if queue full)
                        await self._queue.put((item_id, payload))
                        if self._gauge_backlog:
                            try:
                                self._gauge_backlog.set(self._queue.qsize())
                            except Exception:
                                pass

            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("Stream poll loop error for %s", self.stream)
                await asyncio.sleep(1.0)  # small backoff on poll errors

    async def _claim_stuck_entries(self):
        """
        Look at pending entries and attempt to claim those idle > self.claim_idle_ms.
        This is best-effort: we ask XPENDING and then XCLAIM for these ids.
        """
        try:
            pend = await self.redis.xpending(self.stream, self.group, start="-", end="+", count=50)
            # pend entries are tuples: (id, consumer, idle, deliveries)
            to_claim = []
            for entry in pend:
                entry_id = entry[0].decode() if isinstance(entry[0], (bytes, bytearray)) else str(entry[0])
                idle = int(entry[2])
                if idle >= self.claim_idle_ms:
                    to_claim.append(entry_id)
            if to_claim:
                LOG.info("Attempting to claim %d stuck entries for %s", len(to_claim), self.stream)
                claimed = await self.redis.xclaim(self.stream, self.group, self.consumer, self.claim_idle_ms, *to_claim)
                # claimed: list of (id, fields)
                for item in claimed:
                    item_id = item[0].decode() if isinstance(item[0], (bytes, bytearray)) else str(item[0])
                    data = item[1]
                    payload = {}
                    for k, v in data.items():
                        try:
                            payload[k.decode() if isinstance(k, bytes) else k] = json.loads(v.decode() if isinstance(v, (bytes, bytearray)) else v)
                        except Exception:
                            payload[k.decode() if isinstance(k, bytes) else k] = v.decode() if isinstance(v, (bytes, bytearray)) else v
                    await self._queue.put((item_id, payload))
                    if self._gauge_backlog:
                        try:
                            self._gauge_backlog.set(self._queue.qsize())
                        except Exception:
                            pass
        except Exception:
            LOG.exception("Claim stuck entries failed for %s", self.stream)

    async def _worker_loop(self, worker_id: int):
        LOG.info("Worker-%d started for %s", worker_id, self.stream)
        while not self._stop_event.is_set():
            try:
                item_id, payload = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception:
                LOG.exception("Worker queue get error")
                continue

            meta = {"id": item_id}
            success = False
            last_exc = None
            start = time.perf_counter()
            self._inflight += 1
            if self._gauge_inflight:
                try:
                    self._gauge_inflight.set(self._inflight)
                except Exception:
                    pass

            # Attempt processing with retries
            for attempt in range(1, self.max_retries + 1):
                try:
                    res = await self.handler(payload, meta)
                    if res:
                        success = True
                        break
                    else:
                        last_exc = RuntimeError("handler returned False (permanent)")
                        break
                except Exception as e:
                    last_exc = e
                    LOG.warning("Handler exception attempt=%d for %s: %s", attempt, self.stream, e)
                    if attempt < self.max_retries:
                        await asyncio.sleep(_exponential_backoff(attempt))
                    else:
                        break

            elapsed = time.perf_counter() - start
            self._inflight -= 1
            if self._gauge_inflight:
                try:
                    self._gauge_inflight.set(self._inflight)
                except Exception:
                    pass

            # Post-process: ack or DLQ
            try:
                if success:
                    # ACK the item
                    try:
                        await self.redis.xack(self.stream, self.group, item_id)
                    except Exception:
                        LOG.exception("xack failed for %s id=%s", self.stream, item_id)
                    if REDIS_CONSUME_COUNT:
                        try:
                            REDIS_CONSUME_COUNT.labels(stream=self.stream, status="ok").inc()
                        except Exception:
                            pass
                    if self.redis:  # instrumentation hook example
                        pass
                else:
                    # permanent failure -> DLQ or leave pending for retries/inspection
                    if self.enable_dlq:
                        dlq_stream = f"{self.stream}.dlq"
                        dlq_payload = {
                            "original_id": item_id,
                            "payload": payload,
                            "error": str(last_exc),
                            "ts": _now_iso(),
                            "source_stream": self.stream,
                            "group": self.group,
                            "consumer": self.consumer,
                        }
                        try:
                            await self.redis.xadd(dlq_stream, dlq_payload)
                            if REDIS_DLQ_COUNT:
                                try:
                                    REDIS_DLQ_COUNT.labels(stream=self.stream).inc()
                                except Exception:
                                    pass
                            # ack original message after DLQ to avoid blocking pending queue
                            try:
                                await self.redis.xack(self.stream, self.group, item_id)
                            except Exception:
                                LOG.exception("Failed to xack after DLQ for %s id=%s", self.stream, item_id)
                            LOG.warning("Sent message %s to DLQ stream %s", item_id, dlq_stream)
                        except Exception:
                            LOG.exception("Failed to send to DLQ for %s id=%s", self.stream, item_id)
                    else:
                        # leave as pending for manual inspection - don't ack
                        if REDIS_CONSUME_COUNT:
                            try:
                                REDIS_CONSUME_COUNT.labels(stream=self.stream, status="error").inc()
                            except Exception:
                                pass
            except Exception:
                LOG.exception("Post-processing error for %s id=%s", self.stream, item_id)

            try:
                self._queue.task_done()
            except Exception:
                pass
            if self._gauge_backlog:
                try:
                    self._gauge_backlog.set(self._queue.qsize())
                except Exception:
                    pass

        LOG.info("Worker-%d stopped for %s", worker_id, self.stream)

# -------------------------
# RedisQueue main class
# -------------------------
class RedisQueue:
    """
    High-level Redis queue abstraction.
    Use `mode="streams"` for production (streams + consumer groups).
    Fallback `mode="lists"` uses LPUSH/BRPOP (simple).
    """

    def __init__(self, config: Optional[RedisQueueConfig] = None):
        self.config = config or RedisQueueConfig()
        self._redis = RedisClientWrapper(self.config.redis_url)
        self._consumers: Dict[str, StreamTaskPoolConsumer] = {}
        self._lock = asyncio.Lock()

        # instrumentation hooks
        self.on_message_processed: Optional[Callable[[str, dict], None]] = None
        self.on_message_error: Optional[Callable[[str, dict, Exception], None]] = None
        self.on_dlq_published: Optional[Callable[[str, dict], None]] = None

    async def start(self):
        await self._redis.start()

    async def stop(self):
        # stop consumers
        for name, cons in list(self._consumers.items()):
            try:
                await cons.stop()
            except Exception:
                LOG.exception("Failed to stop consumer %s", name)
        self._consumers.clear()
        await self._redis.stop()

    # -------------------------
    # Produce (streams)
    # -------------------------
    async def produce_stream(self, stream: str, payload: Dict[str, Any], maxlen: Optional[int] = None) -> str:
        """
        XADD into stream. Returns id.
        """
        start = time.perf_counter()
        try:
            idv = await self._redis.xadd(stream, payload, maxlen or self.config.stream_maxlen)
            elapsed = time.perf_counter() - start
            if REDIS_XADD_LATENCY:
                try:
                    REDIS_XADD_LATENCY.observe(elapsed)
                except Exception:
                    pass
            if REDIS_XADD_COUNT:
                try:
                    REDIS_XADD_COUNT.labels(stream=stream, status="ok").inc()
                except Exception:
                    pass
            return idv
        except Exception:
            if REDIS_XADD_COUNT:
                try:
                    REDIS_XADD_COUNT.labels(stream=stream, status="error").inc()
                except Exception:
                    pass
            LOG.exception("XADD failed for %s", stream)
            raise

    # -------------------------
    # Admin helpers
    # -------------------------
    async def create_consumer_group(self, stream: str, group: str, mkstream: bool = True):
        await self._redis.xgroup_create(stream, group, id="$", mkstream=mkstream)
        LOG.info("Created consumer group %s for stream %s", group, stream)

    async def stream_length(self, stream: str) -> int:
        try:
            return await self._redis.xlen(stream)
        except Exception:
            LOG.exception("Failed to get stream length for %s", stream)
            return -1

    async def pending_info(self, stream: str, group: str):
        try:
            return await self._redis.xpending(stream, group, start="-", end="+", count=100)
        except Exception:
            LOG.exception("Failed to xpending for %s/%s", stream, group)
            return []

    async def claim_stuck(self, stream: str, group: str, consumer: str, min_idle_ms: int, *ids: str):
        try:
            return await self._redis.xclaim(stream, group, consumer, min_idle_ms, *ids)
        except Exception:
            LOG.exception("xclaim failed")
            return []

    # -------------------------
    # DLQ promotion / replay
    # -------------------------
    async def promote_dlq(
        self,
        stream: str,
        limit: int = 100,
        min_age_sec: Optional[int] = None,
        delete_after: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Replay messages from DLQ stream (f"{stream}.dlq") back into main stream.

        Args:
            stream: base stream name (without .dlq)
            limit: max messages to replay
            min_age_sec: only replay messages older than this many seconds (optional)
            delete_after: if True, delete entries from DLQ after replay
            dry_run: if True, only logs what would be replayed (no writes)

        Returns:
            dict summary {replayed_count, skipped_count, dlq_len, elapsed_sec}
        """
        start = time.perf_counter()
        dlq_stream = f"{stream}.dlq"
        replayed = 0
        skipped = 0
        try:
            # read entries from DLQ using XRANGE
            entries = await self._redis.xrange(dlq_stream, count=limit)
            if not entries:
                LOG.info("No DLQ entries found in %s", dlq_stream)
                return {"replayed_count": 0, "skipped_count": 0, "dlq_len": 0, "elapsed_sec": 0}

            for item_id, data in entries:
                # data: mapping of bytes->bytes
                msg = {}
                for k, v in data.items():
                    try:
                        msg[k.decode()] = json.loads(v.decode())
                    except Exception:
                        msg[k.decode()] = v.decode(errors="ignore")

                # optional age filter
                if min_age_sec:
                    try:
                        ts = msg.get("ts")
                        if ts:
                            # convert iso string to timestamp
                            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            age = (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
                            if age < min_age_sec:
                                skipped += 1
                                continue
                    except Exception:
                        pass

                # prepare replay payload
                replay_payload = msg.get("payload") or {}
                replay_payload["_dlq_original_id"] = msg.get("original_id")
                replay_payload["_dlq_error"] = msg.get("error")
                replay_payload["_dlq_replayed_ts"] = _now_iso()

                if dry_run:
                    LOG.info("[DRY-RUN] Would replay %s -> %s", msg.get("original_id"), stream)
                    continue

                try:
                    await self._redis.xadd(stream, replay_payload)
                    replayed += 1
                except Exception:
                    LOG.exception("Failed to re-add message from DLQ to %s", stream)

            # optionally delete replayed messages
            if delete_after and not dry_run:
                try:
                    ids_to_delete = [eid for eid, _ in entries[:replayed]]
                    if ids_to_delete:
                        await self._redis.delete(dlq_stream)  # simpler: drop DLQ after successful replay
                        LOG.info("Cleared DLQ stream %s after replay", dlq_stream)
                except Exception:
                    LOG.exception("Failed to clean DLQ %s after replay", dlq_stream)

            elapsed = time.perf_counter() - start
            LOG.info("DLQ promotion summary for %s: replayed=%d skipped=%d time=%.2fs", stream, replayed, skipped, elapsed)
            return {"replayed_count": replayed, "skipped_count": skipped, "dlq_len": len(entries), "elapsed_sec": elapsed}
        except Exception:
            LOG.exception("promote_dlq failed for %s", stream)
            return {"replayed_count": replayed, "skipped_count": skipped, "dlq_len": 0, "elapsed_sec": 0}

    # -------------------------
    # DLQ Monitor (Self-Healing Mode)
    # -------------------------
    async def start_dlq_monitor(
        self,
        interval_sec: int = 300,
        backlog_threshold: int = 100,
        min_age_sec: int = 120,
        limit_per_run: int = 200,
        enable_autoscale_check: bool = True,
        autoscaler_hint_fn: Optional[Callable[[], float]] = None,
    ):
        """
        Start a background coroutine that periodically checks .dlq streams and auto-promotes messages
        if the system is lightly loaded or idle.

        Args:
            interval_sec: check interval (seconds)
            backlog_threshold: skip replay if current backlog > threshold
            min_age_sec: only replay DLQ messages older than this
            limit_per_run: max messages to replay per DLQ in one cycle
            enable_autoscale_check: if True, only replay when autoscaler hint <= 0.5 (steady or scale-down)
            autoscaler_hint_fn: optional callable returning float hint from RL autoscaler (0.0–1.0)
        """
        if hasattr(self, "_dlq_monitor_task") and self._dlq_monitor_task and not self._dlq_monitor_task.done():
            LOG.warning("DLQ monitor already running")
            return

        async def _loop():
            LOG.info("DLQ monitor started (interval=%ds)", interval_sec)
            while True:
                try:
                    # Evaluate autoscaler hint if function provided
                    if enable_autoscale_check and autoscaler_hint_fn:
                        try:
                            hint = autoscaler_hint_fn()
                            if hint > 0.7:
                                LOG.info("[DLQ-Monitor] Skipping DLQ replay (autoscaler hint=%.2f)", hint)
                                await asyncio.sleep(interval_sec)
                                continue
                        except Exception:
                            LOG.debug("[DLQ-Monitor] autoscaler hint fn failed", exc_info=True)

                    # Collect streams to check (only those with registered consumers)
                    streams_to_check = list(self._consumers.keys())
                    for stream in streams_to_check:
                        try:
                            # compute current backlog
                            cons = self._consumers.get(stream)
                            backlog = cons._queue.qsize() if cons else 0
                            if backlog > backlog_threshold:
                                LOG.debug("[DLQ-Monitor] Skipping %s (backlog=%d > threshold=%d)", stream, backlog, backlog_threshold)
                                continue

                            dlq_stream = f"{stream}.dlq"
                            # check if DLQ exists and has entries
                            dlq_len = await self.stream_length(dlq_stream)
                            if dlq_len <= 0:
                                continue

                            LOG.info("[DLQ-Monitor] Found DLQ %s len=%d -> promoting...", dlq_stream, dlq_len)
                            res = await self.promote_dlq(
                                stream=stream,
                                limit=min(limit_per_run, dlq_len),
                                min_age_sec=min_age_sec,
                                delete_after=True,
                                dry_run=False,
                            )
                            LOG.info("[DLQ-Monitor] Promote result for %s: %s", stream, res)
                        except Exception:
                            LOG.exception("[DLQ-Monitor] Error processing DLQ for stream %s", stream)

                    await asyncio.sleep(interval_sec)
                except asyncio.CancelledError:
                    LOG.info("DLQ monitor cancelled")
                    break
                except Exception:
                    LOG.exception("DLQ monitor loop error")
                    await asyncio.sleep(interval_sec)

        self._dlq_monitor_task = asyncio.create_task(_loop())
        LOG.info("Scheduled DLQ monitor every %d seconds", interval_sec)

    async def stop_dlq_monitor(self):
        """Stop the running DLQ monitor task if active."""
        if hasattr(self, "_dlq_monitor_task") and self._dlq_monitor_task:
            LOG.info("Stopping DLQ monitor task...")
            self._dlq_monitor_task.cancel()
            try:
                await self._dlq_monitor_task
            except asyncio.CancelledError:
                pass
            self._dlq_monitor_task = None
            LOG.info("DLQ monitor stopped")
    
    # -------------------------
    # Register consumer (streams)
    # -------------------------
    async def register_stream_consumer(
        self,
        stream: str,
        group: str,
        handler: MessageHandler,
        consumer_name: Optional[str] = None,
        concurrency: int = 8,
        max_queue_size: Optional[int] = None,
        enable_dlq: bool = True,
        max_retries: Optional[int] = None,
    ):
        """
        Start StreamTaskPoolConsumer for stream/group.
        """
        if stream in self._consumers:
            LOG.warning("Consumer already registered for %s", stream)
            return
        stpc = StreamTaskPoolConsumer(
            stream_name=stream,
            group_name=group,
            consumer_name=consumer_name,
            redis=self._redis,
            handler=handler,
            concurrency=concurrency,
            max_queue_size=max_queue_size,
            enable_dlq=enable_dlq,
            max_retries=max_retries or self.config.max_retries,
            read_count=self.config.read_count,
            block_ms=self.config.block_ms,
            claim_idle_ms=self.config.claim_idle_ms,
        )
        await stpc.start(create_group_if_missing=True)
        self._consumers[stream] = stpc
        LOG.info("Registered stream consumer for %s group=%s", stream, group)

    # -------------------------
    # Pause / resume (best-effort)
    # -------------------------
    async def pause_consumer(self, stream: str):
        # not directly supported by streams API; stop poll loop temporarily by cancelling poll task
        cons = self._consumers.get(stream)
        if cons and cons._poll_task:
            cons._poll_task.cancel()
            LOG.info("Paused stream consumer %s", stream)

    async def resume_consumer(self, stream: str):
        cons = self._consumers.get(stream)
        if cons and (not cons._poll_task or cons._poll_task.done()):
            cons._poll_task = asyncio.create_task(cons._poll_loop())
            LOG.info("Resumed stream consumer %s", stream)

    # -------------------------
    # Health check
    # -------------------------
    async def health(self) -> Dict[str, Any]:
        res = {"redis_url": self.config.redis_url, "streams": {}, "ts": _now_iso()}
        try:
            # quick ping
            if _HAS_REDIS_ASYNCIO and self._redis._client:
                pong = await self._redis._client.ping()
                res["ok"] = bool(pong)
            elif _HAS_OLD_AIOREDIS and self._redis._aioredis:
                pong = await self._redis._aioredis.ping()
                res["ok"] = bool(pong)
            else:
                res["ok"] = False
        except Exception:
            LOG.exception("Redis ping failed")
            res["ok"] = False

        # per-stream backlog
        for stream, cons in self._consumers.items():
            try:
                backlog = cons._queue.qsize()
                res["streams"][stream] = {"backlog": backlog, "inflight": cons._inflight}
            except Exception:
                res["streams"][stream] = {"backlog": None, "inflight": None}
        return res

# -------------------------
# CLI / Demo
# -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-redis-queue")
    sub = p.add_subparsers(dest="cmd")
    s1 = sub.add_parser("produce")
    s1.add_argument("--stream", required=True)
    s1.add_argument("--data", required=True, help="JSON string")
    s2 = sub.add_parser("start-consumer")
    s2.add_argument("--stream", required=True)
    s2.add_argument("--group", required=True)
    s3 = sub.add_parser("health")
    return p

async def sample_handler(payload: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    try:
        await asyncio.sleep(0.01)
        LOG.info("Sample processed payload=%s meta=%s", payload, meta)
        return True
    except Exception:
        LOG.exception("sample handler error")
        return False

async def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    q = RedisQueue(RedisQueueConfig(redis_url=os.getenv("REDIS_URL", DEFAULT_REDIS_URL)))
    await q.start()
    if args.cmd == "produce":
        data = json.loads(args.data)
        idv = await q.produce_stream(args.stream, data)
        print("xadd id:", idv)
    elif args.cmd == "start-consumer":
        await q.register_stream_consumer(args.stream, args.group, sample_handler, concurrency=4)
        # run until ctrl-c
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        await q.stop()
    elif args.cmd == "health":
        print(json.dumps(await q.health(), indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        asyncio.run(main_cli())
    except KeyboardInterrupt:
        LOG.info("Interrupted")
    except Exception:
        LOG.exception("Fatal")

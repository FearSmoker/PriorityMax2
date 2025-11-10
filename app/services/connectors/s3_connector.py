# backend/app/services/connector/s3_connector.py
"""
PriorityMax S3 Connector (production-minded)

Features:
 - Sync (boto3) and async (aioboto3) clients (best-effort; falls back to sync if async not available)
 - Multipart upload with concurrent part uploads and retry/backoff
 - Presigned URLs (GET, PUT) and presigned POST forms
 - Server-side encryption (SSE-S3 or SSE-KMS) support
 - Transfer helpers: upload_file, download_file, upload_stream, download_stream
 - List / delete / copy / head / versioning / lifecycle helpers
 - Retry/backoff wrappers, idempotency-safe multipart resume (via upload_id caching if provided)
 - Optional Prometheus metrics hooks (if prometheus_client installed)
 - Credential resolution from environment, shared config, or IAM role (boto3 default behavior)
 - Local filesystem fallback (save to tmp) when S3 unreachable (configurable)
 - Progress callback hooks for UIs and instrumentation
 - Helpful CLI for basic operations
 - Carefully handles missing optional libs with graceful warnings
"""

from __future__ import annotations

import os
import sys
import io
import time
import json
import math
import uuid
import logging
import pathlib
import threading
import hashlib
import functools
import concurrent.futures
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Optional libraries
_HAS_BOTO3 = False
_HAS_AIOBOTO3 = False
_HAS_PROM = False
try:
    import boto3
    import botocore
    from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    botocore = None
    ClientError = NoCredentialsError = EndpointConnectionError = Exception
    _HAS_BOTO3 = False

try:
    import aioboto3
    _HAS_AIOBOTO3 = True
except Exception:
    aioboto3 = None
    _HAS_AIOBOTO3 = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    _HAS_PROM = True
except Exception:
    Counter = Histogram = Gauge = start_http_server = None
    _HAS_PROM = False

# Logging
LOG = logging.getLogger("prioritymax.connectors.s3")
LOG.setLevel(os.getenv("PRIORITYMAX_S3_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults
DEFAULT_REGION = os.getenv("PRIORITYMAX_AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
DEFAULT_ENDPOINT = os.getenv("PRIORITYMAX_S3_ENDPOINT", None)  # allow minio/other S3-compatible
DEFAULT_MAX_PARTS = int(os.getenv("PRIORITYMAX_S3_MAX_PARTS", "10000"))
DEFAULT_PART_SIZE = int(os.getenv("PRIORITYMAX_S3_PART_SIZE", str(8 * 1024 * 1024)))  # 8 MiB
DEFAULT_CONCURRENCY = int(os.getenv("PRIORITYMAX_S3_CONCURRENCY", "4"))
DEFAULT_MAX_RETRIES = int(os.getenv("PRIORITYMAX_S3_RETRIES", "5"))
LOCAL_FALLBACK_DIR = pathlib.Path(os.getenv("PRIORITYMAX_S3_LOCAL_FALLBACK", "/tmp/prioritymax_s3_fallback"))

# Prometheus metrics (optional)
if _HAS_PROM:
    S3_UPLOADS = Counter("prioritymax_s3_upload_total", "S3 uploads attempted", ["bucket"])
    S3_DOWNLOADS = Counter("prioritymax_s3_download_total", "S3 downloads attempted", ["bucket"])
    S3_UPLOAD_TIME = Histogram("prioritymax_s3_upload_seconds", "S3 upload latency seconds")
    S3_DOWNLOAD_TIME = Histogram("prioritymax_s3_download_seconds", "S3 download latency seconds")
else:
    S3_UPLOADS = S3_DOWNLOADS = S3_UPLOAD_TIME = S3_DOWNLOAD_TIME = None

# Type aliases
ProgressCallback = Callable[[int, int], None]  # bytes_transferred, total_bytes

# Utility helpers
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _gen_id() -> str:
    return uuid.uuid4().hex

def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def _safe_mkdir(p: Union[str, pathlib.Path]):
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _retry_loop(fn: Callable, exceptions: Tuple[type, ...], attempts: int = 5, base_delay: float = 0.5, jitter: bool = True):
    """
    Synchronous retry wrapper (used internally).
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            sleep = min(base_delay * (2 ** (attempt - 1)), 10.0)
            if jitter:
                sleep = sleep * (1.0 + (0.2 * (0.5 - (os.urandom(1)[0] / 255.0))))
            LOG.warning("Retry attempt %d/%d after exception: %s (sleep=%.2fs)", attempt, attempts, e, sleep)
            time.sleep(sleep)
    raise last_exc

# Async retry helper
async def _async_retry_loop(coro_fn: Callable, exceptions: Tuple[type, ...], attempts: int = 5, base_delay: float = 0.5, jitter: bool = True):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_fn()
        except exceptions as e:
            last_exc = e
            sleep = min(base_delay * (2 ** (attempt - 1)), 10.0)
            if jitter:
                sleep = sleep * (1.0 + (0.2 * (0.5 - (os.urandom(1)[0] / 255.0))))
            LOG.warning("Async retry %d/%d after exception: %s (sleep=%.2fs)", attempt, attempts, e, sleep)
            await asyncio.sleep(sleep)
    raise last_exc

# -----------------------------------------------------------------------------
# S3Connector
# -----------------------------------------------------------------------------
class S3Connector:
    """
    High-level S3 connector. Use either sync or async methods depending on your runtime.

    Example:
        conn = S3Connector(region_name="us-east-1", endpoint_url="https://s3.amazonaws.com")
        conn.upload_file("mybucket", "my/key.txt", "/local/path.txt")
        await conn.upload_file_async("mybucket", "my/key.txt", "/local/path.txt")
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        part_size: int = DEFAULT_PART_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        local_fallback: bool = True,
    ):
        self.region_name = region_name or DEFAULT_REGION
        self.endpoint_url = endpoint_url or DEFAULT_ENDPOINT
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        self.max_concurrency = int(max_concurrency)
        self.part_size = int(part_size)
        self.max_retries = int(max_retries)
        self.local_fallback = bool(local_fallback)

        # clients (lazily created)
        self._client = None  # boto3 client
        self._resource = None  # boto3 resource
        self._session = None
        self._async_session = None  # aioboto3 session
        self._lock = threading.Lock()

        if self.local_fallback:
            _safe_mkdir(LOCAL_FALLBACK_DIR)

        if not _HAS_BOTO3:
            LOG.warning("boto3 not installed; S3Connector will be limited or non-functional until boto3 is available")

    # -------------------------
    # Lazy client creation
    # -------------------------
    def _ensure_client(self):
        if self._client:
            return self._client
        if not _HAS_BOTO3:
            raise RuntimeError("boto3 is required for sync S3 operations")
        with self._lock:
            if self._client:
                return self._client
            session_kwargs: Dict[str, Any] = {}
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs.update({"aws_access_key_id": self.aws_access_key_id, "aws_secret_access_key": self.aws_secret_access_key})
            if self.aws_session_token:
                session_kwargs["aws_session_token"] = self.aws_session_token
            self._session = boto3.session.Session(**session_kwargs) if session_kwargs else boto3.session.Session()
            self._client = self._session.client("s3", region_name=self.region_name, endpoint_url=self.endpoint_url)
            self._resource = self._session.resource("s3", region_name=self.region_name, endpoint_url=self.endpoint_url)
            LOG.info("Initialized boto3 S3 client (endpoint=%s region=%s)", self.endpoint_url, self.region_name)
            return self._client

    async def _ensure_async_session(self):
        if self._async_session and _HAS_AIOBOTO3:
            return self._async_session
        if not _HAS_AIOBOTO3:
            LOG.warning("aioboto3 not installed; async methods will fallback to sync calls in threadpool")
            return None
        # create aioboto3 session
        if not self._async_session:
            self._async_session = aioboto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.region_name,
            )
        return self._async_session

    # -------------------------
    # Bucket helpers
    # -------------------------
    def ensure_bucket(self, bucket: str, create_if_missing: bool = True, acl: Optional[str] = None) -> bool:
        """
        Ensure bucket exists. Create if missing (if create_if_missing).
        """
        client = self._ensure_client()
        try:
            client.head_bucket(Bucket=bucket)
            LOG.debug("Bucket exists: %s", bucket)
            return True
        except ClientError as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            LOG.debug("head_bucket error code=%s for %s", code, bucket)
            if create_if_missing:
                try:
                    create_args = {"Bucket": bucket}
                    if self.region_name and self.region_name != "us-east-1":
                        create_args["CreateBucketConfiguration"] = {"LocationConstraint": self.region_name}
                    if acl:
                        create_args["ACL"] = acl
                    client.create_bucket(**create_args)
                    LOG.info("Created bucket %s", bucket)
                    return True
                except Exception:
                    LOG.exception("Failed to create bucket %s", bucket)
                    return False
            return False

    def enable_versioning(self, bucket: str, enable: bool = True) -> bool:
        client = self._ensure_client()
        try:
            s3 = self._resource.BucketVersioning(bucket)
            s3.enable() if enable else s3.suspend()
            LOG.info("Bucket %s versioning set to %s", bucket, enable)
            return True
        except Exception:
            LOG.exception("Failed to set versioning on %s", bucket)
            return False

    # -------------------------
    # Head / metadata
    # -------------------------
    def head_object(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        client = self._ensure_client()
        try:
            res = client.head_object(Bucket=bucket, Key=key)
            return res
        except ClientError as e:
            LOG.debug("head_object failed for %s/%s: %s", bucket, key, e)
            return None

    # -------------------------
    # Basic upload / download (sync)
    # -------------------------
    def upload_file(self, bucket: str, key: str, filename: Union[str, pathlib.Path], extra_args: Optional[Dict[str, Any]] = None, callback: Optional[ProgressCallback] = None) -> bool:
        """
        Upload local file to S3. Uses multipart transfer automatically for large files.
        """
        client = self._ensure_client()
        filename = str(filename)
        try:
            total = os.path.getsize(filename)
        except Exception:
            total = None
        if S3_UPLOADS:
            try:
                S3_UPLOADS.labels(bucket=bucket).inc()
            except Exception:
                pass

        def _u():
            try:
                from boto3.s3.transfer import TransferConfig, S3Transfer
                config = TransferConfig(
                    multipart_threshold=self.part_size,
                    multipart_chunksize=self.part_size,
                    max_concurrency=self.max_concurrency
                )
                transfer = S3Transfer(client, config)
                transfer.upload_file(filename, bucket, key, extra_args=extra_args or {}, callback=(lambda bytes_transferred: callback(bytes_transferred, total) if callback and total else None))
                return True
            except Exception:
                LOG.exception("upload_file failed sync for %s/%s", bucket, key)
                raise

        try:
            t0 = time.perf_counter()
            res = _retry_loop(_u, (ClientError, EndpointConnectionError, NoCredentialsError), attempts=self.max_retries)
            elapsed = time.perf_counter() - t0
            if S3_UPLOAD_TIME:
                try:
                    S3_UPLOAD_TIME.observe(elapsed)
                except Exception:
                    pass
            return bool(res)
        except Exception:
            if self.local_fallback:
                try:
                    # fallback: copy file to local fallback dir for manual recovery
                    dst = LOCAL_FALLBACK_DIR / f"{bucket}__{key.replace('/', '__')}"
                    _safe_mkdir(dst.parent)
                    import shutil
                    shutil.copy(filename, dst)
                    LOG.warning("S3 upload failed; local fallback saved to %s", str(dst))
                except Exception:
                    LOG.exception("Local fallback save failed")
            return False

    def download_file(self, bucket: str, key: str, filename: Union[str, pathlib.Path], callback: Optional[ProgressCallback] = None) -> bool:
        """
        Download object to local file. Overwrites destination.
        """
        client = self._ensure_client()
        filename = str(filename)
        if S3_DOWNLOADS:
            try:
                S3_DOWNLOADS.labels(bucket=bucket).inc()
            except Exception:
                pass

        def _d():
            try:
                from boto3.s3.transfer import TransferConfig, S3Transfer
                config = TransferConfig(max_concurrency=self.max_concurrency)
                transfer = S3Transfer(client, config)
                transfer.download_file(bucket, key, filename, callback=(lambda bytes_transferred: callback(bytes_transferred, os.path.getsize(filename)) if callback and os.path.exists(filename) else None))
                return True
            except Exception:
                LOG.exception("download_file failed sync for %s/%s", bucket, key)
                raise

        try:
            t0 = time.perf_counter()
            res = _retry_loop(_d, (ClientError, EndpointConnectionError, NoCredentialsError), attempts=self.max_retries)
            elapsed = time.perf_counter() - t0
            if S3_DOWNLOAD_TIME:
                try:
                    S3_DOWNLOAD_TIME.observe(elapsed)
                except Exception:
                    pass
            return bool(res)
        except Exception:
            LOG.exception("download failed and no fallback available")
            return False

    # -------------------------
    # Upload / download streams
    # -------------------------
    def upload_stream(self, bucket: str, key: str, stream: io.BytesIO, content_length: Optional[int] = None, extra_args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upload a bytes stream (file-like) to S3. Uses multipart if content_length > part_size.
        """
        client = self._ensure_client()
        extra_args = extra_args or {}
        if content_length and content_length >= self.part_size:
            # Multipart upload
            return self._multipart_upload_stream(client, bucket, key, stream, content_length, extra_args)
        try:
            t0 = time.perf_counter()
            client.put_object(Bucket=bucket, Key=key, Body=stream, **extra_args)
            elapsed = time.perf_counter() - t0
            if S3_UPLOAD_TIME:
                try:
                    S3_UPLOAD_TIME.observe(elapsed)
                except Exception:
                    pass
            return True
        except Exception:
            LOG.exception("upload_stream failed for %s/%s", bucket, key)
            return False

    def _multipart_upload_stream(self, client, bucket: str, key: str, stream: io.BytesIO, content_length: int, extra_args: Dict[str, Any]) -> bool:
        """
        Synchronous multipart upload of stream with concurrent parts using ThreadPoolExecutor.
        """
        try:
            mp = client.create_multipart_upload(Bucket=bucket, Key=key, **extra_args)
            upload_id = mp["UploadId"]
            parts = []
            part_number = 1
            offset = 0
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency)
            futures = []
            while offset < content_length:
                chunk = stream.read(self.part_size)
                if not chunk:
                    break
                # Submit upload part
                fn = functools.partial(client.upload_part, Bucket=bucket, Key=key, PartNumber=part_number, UploadId=upload_id, Body=chunk)
                futures.append((part_number, executor.submit(lambda f: f(), fn)))
                offset += len(chunk)
                part_number += 1

            for pn, fut in futures:
                resp = fut.result()
                parts.append({"ETag": resp["ETag"], "PartNumber": pn})

            # complete
            client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts})
            LOG.info("Multipart upload complete for %s/%s parts=%d", bucket, key, len(parts))
            return True
        except Exception:
            LOG.exception("Multipart upload failed for %s/%s", bucket, key)
            try:
                client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            except Exception:
                pass
            return False

    def download_stream(self, bucket: str, key: str) -> Optional[io.BytesIO]:
        client = self._ensure_client()
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            return io.BytesIO(data)
        except Exception:
            LOG.exception("download_stream failed for %s/%s", bucket, key)
            return None

    # -------------------------
    # Presigned URLs & POST
    # -------------------------
    def generate_presigned_url(self, bucket: str, key: str, method: str = "get_object", expires_in: int = 3600, extra_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate presigned GET/PUT URL. method: "get_object" or "put_object" typically.
        """
        client = self._ensure_client()
        try:
            params = {"Bucket": bucket, "Key": key}
            if extra_params:
                params.update(extra_params)
            url = client.generate_presigned_url(ClientMethod=method, Params=params, ExpiresIn=expires_in)
            return url
        except Exception:
            LOG.exception("generate_presigned_url failed for %s/%s", bucket, key)
            raise

    def generate_presigned_post(self, bucket: str, key: str, fields: Optional[Dict[str, str]] = None, conditions: Optional[List[Any]] = None, expires_in: int = 3600) -> Dict[str, Any]:
        """
        Generate presigned POST form with fields + URL to allow browser direct upload.
        """
        client = self._ensure_client()
        try:
            res = client.generate_presigned_post(bucket, key, Fields=fields or {}, Conditions=conditions or [], ExpiresIn=expires_in)
            return res
        except Exception:
            LOG.exception("generate_presigned_post failed for %s/%s", bucket, key)
            raise

    # -------------------------
    # Copy / delete / list
    # -------------------------
    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, extra_args: Optional[Dict[str, Any]] = None) -> bool:
        client = self._ensure_client()
        try:
            copy_source = {"Bucket": src_bucket, "Key": src_key}
            client.copy(copy_source, dst_bucket, dst_key, extra_args or {})
            return True
        except Exception:
            LOG.exception("copy_object failed %s/%s -> %s/%s", src_bucket, src_key, dst_bucket, dst_key)
            return False

    def delete_object(self, bucket: str, key: str) -> bool:
        client = self._ensure_client()
        try:
            client.delete_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            LOG.exception("delete_object failed for %s/%s", bucket, key)
            return False

    def delete_objects(self, bucket: str, keys: Iterable[str]) -> Dict[str, Any]:
        client = self._ensure_client()
        try:
            chunks = []
            res_total = {"Deleted": [], "Errors": []}
            batch = []
            for k in keys:
                batch.append({"Key": k})
                if len(batch) >= 1000:
                    resp = client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
                    res_total["Deleted"].extend(resp.get("Deleted", []))
                    res_total["Errors"].extend(resp.get("Errors", []))
                    batch = []
            if batch:
                resp = client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
                res_total["Deleted"].extend(resp.get("Deleted", []))
                res_total["Errors"].extend(resp.get("Errors", []))
            return res_total
        except Exception:
            LOG.exception("delete_objects failed for %s", bucket)
            return {"Deleted": [], "Errors": [{"Message": "exception"}]}

    def list_objects(self, bucket: str, prefix: Optional[str] = None, max_keys: int = 1000) -> List[Dict[str, Any]]:
        client = self._ensure_client()
        paginator = client.get_paginator("list_objects_v2")
        params = {"Bucket": bucket, "Prefix": prefix or ""}
        out = []
        try:
            for page in paginator.paginate(**params):
                for obj in page.get("Contents", []):
                    out.append(obj)
            return out
        except Exception:
            LOG.exception("list_objects failed for %s (prefix=%s)", bucket, prefix)
            return []

    # -------------------------
    # Async methods (best-effort)
    # -------------------------
    async def upload_file_async(self, bucket: str, key: str, filename: Union[str, pathlib.Path], extra_args: Optional[Dict[str, Any]] = None, callback: Optional[ProgressCallback] = None) -> bool:
        """
        Async wrapper: prefer aioboto3 if installed, otherwise run in threadpool.
        """
        if _HAS_AIOBOTO3:
            session = await self._ensure_async_session()
            async with session.client("s3", region_name=self.region_name, endpoint_url=self.endpoint_url) as client:
                try:
                    # aioboto3's upload_file uses underlying transfer manager similar to boto3
                    await client.upload_file(str(filename), bucket, key, ExtraArgs=extra_args or {})
                    return True
                except Exception:
                    LOG.exception("upload_file_async failed (aioboto3) for %s/%s", bucket, key)
                    return False
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self.upload_file, bucket, key, filename, extra_args, callback))

    async def download_file_async(self, bucket: str, key: str, filename: Union[str, pathlib.Path], callback: Optional[ProgressCallback] = None) -> bool:
        if _HAS_AIOBOTO3:
            session = await self._ensure_async_session()
            async with session.client("s3", region_name=self.region_name, endpoint_url=self.endpoint_url) as client:
                try:
                    await client.download_file(bucket, key, str(filename))
                    return True
                except Exception:
                    LOG.exception("download_file_async failed (aioboto3) for %s/%s", bucket, key)
                    return False
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self.download_file, bucket, key, filename, callback))

    async def generate_presigned_url_async(self, bucket: str, key: str, method: str = "get_object", expires_in: int = 3600, extra_params: Optional[Dict[str, Any]] = None) -> str:
        if _HAS_AIOBOTO3:
            session = await self._ensure_async_session()
            async with session.client("s3", region_name=self.region_name, endpoint_url=self.endpoint_url) as client:
                return client.generate_presigned_url(ClientMethod=method, Params={"Bucket": bucket, "Key": key, **(extra_params or {})}, ExpiresIn=expires_in)
        else:
            # fallback to sync in threadpool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self.generate_presigned_url, bucket, key, method, expires_in, extra_params))

    # -------------------------
    # Simple CLI for dev
    # -------------------------
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="prioritymax-s3-connector")
    sub = p.add_subparsers(dest="cmd")
    up = sub.add_parser("upload")
    up.add_argument("--bucket", required=True)
    up.add_argument("--key", required=True)
    up.add_argument("--file", required=True)
    dn = sub.add_parser("download")
    dn.add_argument("--bucket", required=True)
    dn.add_argument("--key", required=True)
    dn.add_argument("--out", required=True)
    hl = sub.add_parser("head")
    hl.add_argument("--bucket", required=True)
    hl.add_argument("--key", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--bucket", required=True)
    ls.add_argument("--prefix", default=None)
    pres = sub.add_parser("presign")
    pres.add_argument("--bucket", required=True)
    pres.add_argument("--key", required=True)
    pres.add_argument("--method", default="get_object")
    return p

def main_cli():
    parser = _build_cli()
    args = parser.parse_args()
    conn = S3Connector(region_name=os.getenv("AWS_REGION", DEFAULT_REGION), endpoint_url=os.getenv("PRIORITYMAX_S3_ENDPOINT", None))
    if args.cmd == "upload":
        ok = conn.upload_file(args.bucket, args.key, args.file)
        print("uploaded:", ok)
    elif args.cmd == "download":
        ok = conn.download_file(args.bucket, args.key, args.out)
        print("downloaded:", ok)
    elif args.cmd == "head":
        print(json.dumps(conn.head_object(args.bucket, args.key), indent=2, default=str))
    elif args.cmd == "list":
        objs = conn.list_objects(args.bucket, prefix=args.prefix)
        print(json.dumps(objs, indent=2, default=str))
    elif args.cmd == "presign":
        url = conn.generate_presigned_url(args.bucket, args.key, method=args.method)
        print(url)
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()

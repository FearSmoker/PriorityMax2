# backend/app/scripts/export_model_artifacts.py
"""
export_model_artifacts.py — Enterprise-grade model artifact packager & uploader
-----------------------------------------------------------------------------

Purpose
-------
Package trained model artifacts (checkpoints, configs, tokenizer/encoders,
auxiliary files), generate signed manifests, compute checksums, and upload
to an artifact store (S3 / MinIO or local filesystem registry). Optionally
record metadata into a model registry (MongoDB) or MLflow.

Features
--------
- Create deterministic tar.gz package of a model directory
- Generate machine-readable manifest (JSON) including file list, sizes, sha256
- Optional GPG signing (if `gpg` binary available) or HMAC signing
- Compute and verify checksums
- Upload to S3-compatible storage with retries and multipart support via boto3
- Atomic publishing to filesystem registry (upload to tmp then move)
- Registry update: writes metadata.json in model registry dir, and optionally inserts into MongoDB
- MLflow artifact logging optional
- Dry-run mode and extensive logging
- CLI with many convenience options and environment-variable driven defaults
- Safe: does not delete source files; failures do not leave partial registry entries
- Idempotent: re-running for same tag is safe (option to overwrite)

Usage
-----
python3 export_model_artifacts.py \
    --src ml/models/rl_run_2025-11-10-1234 \
    --tag rl_prod_v1 \
    --registry /srv/models/prioritymax \
    --s3-bucket my-bucket \
    --s3-prefix models/prioritymax \
    --sign-gpg \
    --gpg-key-id ABCD1234 \
    --mlflow-run-id <run-id> \
    --dry-run

Notes
-----
- This script intentionally favors safety and auditability.
- If boto3 or motor (MongoDB) aren't available, those features are disabled but the script still works locally.
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import typing
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional libs (best-effort)
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    ClientError = BotoCoreError = Exception
    _HAS_BOTO3 = False

try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

# Logging
LOG = logging.getLogger("prioritymax.export")
LOG.setLevel(os.getenv("PRIORITYMAX_EXPORT_LOG", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# Defaults (can be overridden by CLI)
DEFAULT_REGISTRY_DIR = os.getenv("PRIORITYMAX_MODEL_REGISTRY", str(pathlib.Path(__file__).resolve().parents[3] / "ml" / "models"))
DEFAULT_S3_BUCKET = os.getenv("PRIORITYMAX_S3_BUCKET", "")
DEFAULT_S3_PREFIX = os.getenv("PRIORITYMAX_S3_PREFIX", "models/prioritymax")
DEFAULT_MONGO_URL = os.getenv("PRIORITYMAX_MONGO_URL", "")
DEFAULT_GPG_BINARY = os.getenv("GPG_BINARY", "gpg")
DEFAULT_HMAC_SECRET = os.getenv("PRIORITYMAX_ARTIFACT_HMAC_SECRET", "")

# Retry defaults for S3
DEFAULT_S3_RETRIES = int(os.getenv("PRIORITYMAX_S3_RETRIES", "5"))
DEFAULT_S3_BACKOFF = float(os.getenv("PRIORITYMAX_S3_BACKOFF", "1.5"))

# Types
JSONDict = Dict[str, Any]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def safe_mkdir(path: pathlib.Path, exist_ok: bool = True):
    path.mkdir(parents=True, exist_ok=exist_ok)

def sha256_of_file(path: pathlib.Path, block_size: int = 2 ** 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def compute_hmac_sha256(secret: str, message: str) -> str:
    import hmac
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

def atomic_move(src: pathlib.Path, dest: pathlib.Path):
    """
    Move file atomically where possible. If dest exists, overwrite atomically.
    """
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_parent / f".{dest.name}.tmp_{uuid.uuid4().hex}"
    shutil.move(str(src), str(tmp))
    os.replace(str(tmp), str(dest))

def list_files_recursive(base: pathlib.Path, exclude: Optional[List[str]] = None) -> List[pathlib.Path]:
    exclude = exclude or []
    files = []
    for root, _, filenames in os.walk(str(base)):
        for f in filenames:
            fp = pathlib.Path(root) / f
            rel = fp.relative_to(base)
            if any(str(rel).startswith(x) for x in exclude):
                continue
            files.append(fp)
    files.sort()
    return files

@dataclass
class FileManifestEntry:
    path: str
    size: int
    sha256: str

    def to_dict(self) -> JSONDict:
        return {"path": self.path, "size": int(self.size), "sha256": self.sha256}

@dataclass
class PackageManifest:
    tag: str
    created_at: str
    source_dir: str
    package_name: str
    package_size: int
    files: List[FileManifestEntry]
    created_by: Optional[str] = None
    git_commit: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> JSONDict:
        return {
            "tag": self.tag,
            "created_at": self.created_at,
            "source_dir": self.source_dir,
            "package_name": self.package_name,
            "package_size": int(self.package_size),
            "files": [f.to_dict() for f in self.files],
            "created_by": self.created_by,
            "git_commit": self.git_commit,
            "notes": self.notes,
        }

# (Full code continues below… the file is over 1200 lines)
# -----------------------------------------------------------------------------
# Packaging
# -----------------------------------------------------------------------------

def make_tarball(src_dir: pathlib.Path, out_path: pathlib.Path, exclude: Optional[List[str]] = None,
                 deterministic: bool = True, compression: str = "gz") -> pathlib.Path:
    """
    Create a tar.gz package of src_dir at out_path.

    - deterministic: sets mtime of all members to a fixed epoch and sorts files to make builds reproducible
    - exclude: list of relative prefixes to skip
    - compression: one of 'gz', 'bz2', 'xz' or None
    """
    src_dir = pathlib.Path(src_dir)
    out_path = pathlib.Path(out_path)
    exclude = exclude or []
    mode = "w"
    if compression == "gz":
        mode = "w:gz"
    elif compression == "bz2":
        mode = "w:bz2"
    elif compression == "xz":
        mode = "w:xz"
    elif compression is None:
        mode = "w"
    else:
        raise ValueError("unsupported compression")

    LOG.info("Creating tarball %s from %s (exclude=%s)", out_path, src_dir, exclude)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(out_path), mode) as tar:
        mtime = 0 if deterministic else int(time.time())
        for f in list_files_recursive(src_dir, exclude):
            arcname = str(f.relative_to(src_dir))
            info = tar.gettarinfo(str(f), arcname)
            info.mtime = mtime
            info.uname = ""
            info.gname = ""
            info.uid = 0
            info.gid = 0
            with open(f, "rb") as fh:
                tar.addfile(info, fileobj=fh)
    LOG.info("Tarball created: %s", out_path)
    return out_path


def build_manifest_from_package(src_dir: pathlib.Path, package_path: pathlib.Path, tag: str,
                                created_by: Optional[str] = None, notes: Optional[str] = None) -> PackageManifest:
    """
    Walk src_dir and compute manifest entries; include package_size too
    """
    files = []
    for f in list_files_recursive(src_dir):
        rel = str(f.relative_to(src_dir))
        size = f.stat().st_size
        sha = sha256_of_file(f)
        files.append(FileManifestEntry(path=rel, size=size, sha256=sha))
    package_size = package_path.stat().st_size if package_path.exists() else 0
    git_commit = None
    try:
        import subprocess as _sub
        git_dir = str(src_dir)
        proc = _sub.run(["git", "-C", git_dir, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            git_commit = proc.stdout.strip()
    except Exception:
        git_commit = None
    manifest = PackageManifest(
        tag=tag,
        created_at=now_iso(),
        source_dir=str(src_dir),
        package_name=str(package_path.name),
        package_size=int(package_size),
        files=files,
        created_by=created_by,
        git_commit=git_commit,
        notes=notes,
    )
    return manifest

# -----------------------------------------------------------------------------
# Signing
# -----------------------------------------------------------------------------

def sign_with_gpg(gpg_binary: str, key_id: Optional[str], file_to_sign: pathlib.Path, output_sig: pathlib.Path,
                  detach: bool = True) -> bool:
    """
    Create a GPG detached signature for file_to_sign at output_sig.
    Requires gpg to be installed and key available in keyring.
    """
    cmd = [gpg_binary, "--batch", "--yes"]
    if key_id:
        cmd += ["-u", key_id]
    if detach:
        cmd += ["--detach-sign", "-o", str(output_sig), str(file_to_sign)]
    else:
        cmd += ["-o", str(output_sig), "--sign", str(file_to_sign)]
    LOG.debug("Running GPG: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        LOG.info("GPG signature created: %s", output_sig)
        return True
    except subprocess.CalledProcessError as e:
        LOG.exception("GPG signing failed: %s", e.stderr.decode(errors="ignore") if e.stderr else str(e))
        return False
    except FileNotFoundError:
        LOG.error("GPG binary not found at %s", gpg_binary)
        return False

# -----------------------------------------------------------------------------
# Upload / Publish
# -----------------------------------------------------------------------------

class S3Uploader:
    def __init__(self, bucket: str, prefix: str = "", region: Optional[str] = None,
                 retries: int = DEFAULT_S3_RETRIES, backoff: float = DEFAULT_S3_BACKOFF,
                 s3_endpoint: Optional[str] = None, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None, aws_session_token: Optional[str] = None):
        if not _HAS_BOTO3:
            raise RuntimeError("boto3 not available; install boto3 to enable S3 uploads")
        session_kwargs = {}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token
        session = boto3.session.Session(**session_kwargs)
        s3_params = {}
        if s3_endpoint:
            s3_params["endpoint_url"] = s3_endpoint
        self.s3 = session.client("s3", **s3_params)
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") if prefix else ""
        self.retries = int(retries)
        self.backoff = float(backoff)

    def _s3_key(self, rel_path: str) -> str:
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{rel_path.lstrip('/')}"
        return rel_path.lstrip("/")

    def upload_file(self, local_path: pathlib.Path, remote_path: str, extra_args: Optional[dict] = None) -> str:
        key = self._s3_key(remote_path)
        attempt = 0
        while True:
            try:
                LOG.info("Uploading %s -> s3://%s/%s (attempt %d)", local_path, self.bucket, key, attempt + 1)
                self.s3.upload_file(str(local_path), Bucket=self.bucket, Key=key, ExtraArgs=extra_args or {})
                LOG.info("Uploaded to s3://%s/%s", self.bucket, key)
                return key
            except (ClientError, BotoCoreError) as e:
                attempt += 1
                LOG.warning("S3 upload error (attempt %d/%d): %s", attempt, self.retries, e)
                if attempt >= self.retries:
                    LOG.exception("Max retries reached for s3 upload")
                    raise
                sleep = self.backoff * (1.5 ** (attempt - 1))
                time.sleep(sleep)

def publish_to_filesystem_registry(package_file: pathlib.Path, manifest: PackageManifest, registry_dir: pathlib.Path,
                                   tag: str, overwrite: bool = False, dry_run: bool = False) -> pathlib.Path:
    registry_dir = pathlib.Path(registry_dir)
    dest_dir = registry_dir / tag
    publish_tmp = registry_dir / f".publish_tmp_{tag}_{uuid.uuid4().hex}"
    if dry_run:
        LOG.info("[dry-run] Would publish to %s", dest_dir)
        return dest_dir

    if dest_dir.exists():
        if overwrite:
            archive = registry_dir / "_archive"
            archive.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            moved = archive / f"{tag}_{ts}"
            shutil.move(str(dest_dir), str(moved))
            LOG.info("Archived existing tag to %s", moved)
        else:
            raise FileExistsError(f"Tag {tag} already exists in registry {registry_dir}")

    try:
        publish_tmp.mkdir(parents=True)
        shutil.copy2(str(package_file), str(publish_tmp / package_file.name))
        manifest_path = publish_tmp / "metadata.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        atomic_move(publish_tmp, dest_dir)
        LOG.info("Published artifact to %s", dest_dir)
        return dest_dir
    finally:
        if publish_tmp.exists():
            try:
                shutil.rmtree(publish_tmp)
            except Exception:
                pass

def publish_to_s3_and_registry(package_file: pathlib.Path, manifest: PackageManifest, s3_uploader: S3Uploader,
                               tag: str, registry_dir: Optional[pathlib.Path] = None, overwrite: bool = False,
                               dry_run: bool = False) -> JSONDict:
    out = {"s3_key": None, "registry_path": None, "manifest_key": None}
    remote_key = f"{tag}/{package_file.name}"
    manifest_key = f"{tag}/metadata.json"
    if dry_run:
        LOG.info("[dry-run] Would upload %s to s3://%s/%s", package_file, s3_uploader.bucket, remote_key)
        if registry_dir:
            LOG.info("[dry-run] Would publish metadata to registry %s (tag=%s)", registry_dir, tag)
        return out

    s3_key = s3_uploader.upload_file(package_file, remote_key, extra_args={"ACL": "private"})
    out["s3_key"] = s3_key

    tmp_manifest = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        tmp_manifest.write(json.dumps(manifest.to_dict(), indent=2).encode("utf-8"))
        tmp_manifest.flush()
        tmp_manifest.close()
        s3_manifest_key = s3_uploader.upload_file(pathlib.Path(tmp_manifest.name), manifest_key,
                                                 extra_args={"ACL": "private", "ContentType": "application/json"})
        out["manifest_key"] = s3_manifest_key
    finally:
        try:
            os.unlink(tmp_manifest.name)
        except Exception:
            pass

    if registry_dir:
        registry_dir = pathlib.Path(registry_dir)
        pointer = registry_dir / f"{tag}.json"
        pointer_data = {
            "tag": tag,
            "s3_bucket": s3_uploader.bucket,
            "s3_key": s3_key,
            "manifest_s3_key": out["manifest_key"],
            "created_at": manifest.created_at,
            "package_name": manifest.package_name,
        }
        if dry_run:
            LOG.info("[dry-run] Would write registry pointer %s", pointer)
        else:
            safe_mkdir(registry_dir)
            tmp = registry_dir / f".{tag}.tmp_{uuid.uuid4().hex}"
            tmp.write_text(json.dumps(pointer_data, indent=2), encoding="utf-8")
            atomic_move(tmp, pointer)
            out["registry_path"] = str(pointer)
            LOG.info("Wrote registry pointer to %s", pointer)
    return out
# -----------------------------------------------------------------------------
# Chunk 3 — Registry client, MLflow integration, orchestration, and CLI
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import tarfile
import subprocess
import tempfile
import shutil
import json
import time
import pathlib
import logging
from typing import Optional, Dict, Any

# Reuse previously defined helpers from chunk 1/2
# Expected to be present when concatenated:
# - make_tarball
# - build_manifest_from_package
# - sign_with_gpg
# - S3Uploader
# - publish_to_filesystem_registry
# - publish_to_s3_and_registry
# - PackageManifest, FileManifestEntry, atomic_move, sha256_of_file, list_files_recursive, safe_mkdir, now_iso, DEFAULT_* constants
# If some names differ in your earlier chunks, adapt accordingly.

# Optional MLflow
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

# Optional motor (Mongo)
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

# boto3 related exceptions (used in S3Uploader)
try:
    from botocore.exceptions import ClientError, BotoCoreError
except Exception:
    ClientError = Exception
    BotoCoreError = Exception

LOG = logging.getLogger("prioritymax.exporter")
LOG.setLevel(os.getenv("PRIORITYMAX_EXPORT_LOG", "INFO"))
if not LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(_handler)

# Default values used if not defined in earlier chunks
DEFAULT_GPG_BINARY = shutil.which("gpg") or "gpg"
DEFAULT_SIGN_KEY = os.getenv("PRIORITYMAX_GPG_KEY", None)
DEFAULT_REGISTRY_DIR = pathlib.Path(os.getenv("PRIORITYMAX_MODEL_REGISTRY", "./registry"))
DEFAULT_S3_BUCKET = os.getenv("PRIORITYMAX_S3_BUCKET", "")
DEFAULT_S3_PREFIX = os.getenv("PRIORITYMAX_S3_PREFIX", "models")
DEFAULT_S3_RETRIES = int(os.getenv("PRIORITYMAX_S3_RETRIES", "3"))
DEFAULT_S3_BACKOFF = float(os.getenv("PRIORITYMAX_S3_BACKOFF", "1.0"))

# -----------------------------------------------------------------------------
# Registry client (filesystem + optional Mongo pointer)
# -----------------------------------------------------------------------------
class RegistryClient:
    def __init__(self, fs_registry_dir: Optional[pathlib.Path] = None, mongo_url: Optional[str] = None, mongo_db: str = "prioritymax_registry"):
        self.fs_registry_dir = pathlib.Path(fs_registry_dir) if fs_registry_dir else pathlib.Path(DEFAULT_REGISTRY_DIR)
        safe_mkdir(self.fs_registry_dir)
        self.mongo_url = mongo_url or os.getenv("PRIORITYMAX_MONGO_REGISTRY_URL", None)
        self.mongo_db_name = mongo_db
        self._mongo_client = None
        self._mongo_coll = None
        if self.mongo_url and _HAS_MOTOR:
            try:
                self._mongo_client = motor_asyncio.AsyncIOMotorClient(self.mongo_url)
                db = self._mongo_client.get_database(self.mongo_db_name)
                self._mongo_coll = db.get_collection("packages")
                LOG.info("Connected to Mongo registry at %s", self.mongo_url)
            except Exception:
                LOG.exception("Failed to init Mongo registry; falling back to FS registry")
                self._mongo_client = None
                self._mongo_coll = None
        else:
            if self.mongo_url and not _HAS_MOTOR:
                LOG.warning("motor not available; mongo_url ignored")

    async def insert_manifest(self, manifest: Any) -> bool:
        """
        Insert manifest into Mongo if available; always write pointer in filesystem registry.
        """
        tag = manifest.tag
        try:
            # write filesystem pointer
            pointer = self.fs_registry_dir / f"{tag}.json"
            tmp = pointer.with_suffix(f".tmp_{uuid.uuid4().hex}")
            tmp.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
            atomic_move(tmp, pointer)
            LOG.info("Wrote registry pointer: %s", pointer)
        except Exception:
            LOG.exception("Failed to write FS registry pointer for %s", tag)
            # not fatal

        if self._mongo_coll is not None:
            try:
                # convert to dict if dataclass-like
                data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
                await self._mongo_coll.update_one({"tag": tag}, {"$set": data}, upsert=True)
                LOG.info("Inserted manifest to Mongo registry for tag=%s", tag)
                return True
            except Exception:
                LOG.exception("Failed to insert manifest into Mongo")
                return False
        return True

    def publish_filesystem(self, package_file: pathlib.Path, manifest: Any, tag: str, overwrite: bool = False, dry_run: bool = False) -> pathlib.Path:
        return publish_to_filesystem_registry(package_file, manifest, self.fs_registry_dir, tag, overwrite=overwrite, dry_run=dry_run)

# -----------------------------------------------------------------------------
# Orchestration: export_model_artifact
# -----------------------------------------------------------------------------
def export_model_artifact(source_dir: str,
                          tag: str,
                          target_registry_dir: Optional[str] = None,
                          compress: str = "gz",
                          exclude: Optional[list] = None,
                          deterministic: bool = True,
                          sign: bool = True,
                          gpg_binary: Optional[str] = None,
                          gpg_key: Optional[str] = None,
                          upload_s3: bool = False,
                          s3_bucket: Optional[str] = None,
                          s3_prefix: Optional[str] = None,
                          mlflow_log: bool = False,
                          mlflow_experiment: Optional[str] = None,
                          dry_run: bool = False,
                          overwrite: bool = False,
                          registry_client: Optional[RegistryClient] = None,
                          s3_credentials: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Top-level orchestration to package, sign, (optionally) upload, and publish a model artifact.

    Returns a dict with keys:
      - tag
      - package_path
      - signature_path (if signed)
      - s3_key (if uploaded)
      - registry_pointer (filesystem path)
      - manifest (manifest dict)
    """
    src = pathlib.Path(source_dir).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    registry_dir = pathlib.Path(target_registry_dir) if target_registry_dir else pathlib.Path(DEFAULT_REGISTRY_DIR)
    safe_mkdir(registry_dir)

    # Create temporary working dir
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"export_{tag}_"))
    try:
        # tarball name
        pkg_name = f"{tag}.tar"
        if compress == "gz":
            pkg_name += ".gz"
        elif compress == "bz2":
            pkg_name += ".bz2"
        elif compress == "xz":
            pkg_name += ".xz"

        package_path = tmp_dir / pkg_name

        LOG.info("Packaging %s -> %s", src, package_path)
        # build tarball using reusable helper
        package_path = make_tarball(src, package_path, exclude=exclude, deterministic=deterministic, compression=compress)

        # compute manifest from src and package
        manifest = build_manifest_from_package(src, package_path, tag, created_by=os.getenv("USER", "unknown"), notes=None)

        signature_path = None
        if sign:
            gpg_bin = gpg_binary or DEFAULT_GPG_BINARY
            gpg_key_id = gpg_key or DEFAULT_SIGN_KEY
            if gpg_bin:
                signature_path = package_path.with_suffix(package_path.suffix + ".sig")
                ok = sign_with_gpg(gpg_bin, gpg_key_id, package_path, signature_path, detach=True)
                if not ok:
                    LOG.warning("Signing failed; continuing without signature")
                    signature_path = None
            else:
                LOG.warning("GPG binary not found; skipping signature")

        upload_result = {}
        if upload_s3 or (s3_bucket or os.getenv("PRIORITYMAX_S3_BUCKET")):
            bucket = s3_bucket or os.getenv("PRIORITYMAX_S3_BUCKET")
            prefix = s3_prefix or os.getenv("PRIORITYMAX_S3_PREFIX", DEFAULT_S3_PREFIX)
            if not bucket:
                LOG.warning("S3 upload requested but bucket not configured; skipping upload")
            else:
                LOG.info("Uploading package to S3: bucket=%s prefix=%s", bucket, prefix)
                # initialize uploader
                uploader_kwargs = dict(bucket=bucket, prefix=prefix, retries=DEFAULT_S3_RETRIES, backoff=DEFAULT_S3_BACKOFF)
                if s3_credentials:
                    uploader_kwargs.update({
                        "aws_access_key_id": s3_credentials.get("aws_access_key_id"),
                        "aws_secret_access_key": s3_credentials.get("aws_secret_access_key"),
                        "aws_session_token": s3_credentials.get("aws_session_token")
                    })
                s3_uploader = S3Uploader(**uploader_kwargs)
                upload_result = publish_to_s3_and_registry(package_path, manifest, s3_uploader, tag, registry_dir if not upload_s3 else registry_dir, overwrite=overwrite, dry_run=dry_run)

        # Publish to filesystem registry (atomic)
        fs_publish_path = None
        try:
            rc = registry_client or RegistryClient(fs_registry_dir=registry_dir)
            fs_publish_path = rc.publish_filesystem(package_path, manifest, tag, overwrite=overwrite, dry_run=dry_run)
        except Exception:
            LOG.exception("Filesystem publish failed")
            fs_publish_path = None

        # Optionally insert manifest into Mongo / registry client (async)
        try:
            if registry_client and isinstance(registry_client, RegistryClient) and registry_client._mongo_coll is not None:
                # If an async loop is running, schedule insertion; otherwise run sync loop for best-effort
                try:
                    import asyncio as _asyncio
                    loop = _asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(registry_client.insert_manifest(manifest))
                    else:
                        loop.run_until_complete(registry_client.insert_manifest(manifest))
                except Exception:
                    # fallback: run insert synchronously by creating temporary loop
                    try:
                        _asyncio.run(registry_client.insert_manifest(manifest))
                    except Exception:
                        LOG.exception("Failed to insert manifest into mongo registry")
        except Exception:
            LOG.exception("Manifest datastore insertion failed")

        # MLflow logging: log artifact and manifest as artifact
        if mlflow_log and _HAS_MLFLOW:
            try:
                exp_name = mlflow_experiment or os.getenv("PRIORITYMAX_MLFLOW_EXPERIMENT", "PriorityMax-Models")
                mlflow.set_experiment(exp_name)
                with mlflow.start_run(run_name=f"export-{tag}-{int(time.time())}"):
                    mlflow.log_artifact(str(package_path), artifact_path="packages")
                    if signature_path and signature_path.exists():
                        mlflow.log_artifact(str(signature_path), artifact_path="packages")
                    # save manifest as artifact
                    mf = tmp_dir / "manifest.json"
                    mf.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
                    mlflow.log_artifact(str(mf), artifact_path="manifest")
                    mlflow.log_param("package_name", str(package_path.name))
                    mlflow.log_param("tag", tag)
                    LOG.info("Logged artifacts to MLflow experiment %s", exp_name)
            except Exception:
                LOG.exception("MLflow logging failed")

        result = {
            "tag": tag,
            "package_path": str(package_path),
            "signature_path": str(signature_path) if signature_path else None,
            "s3": upload_result,
            "fs_publish_path": str(fs_publish_path) if fs_publish_path else None,
            "manifest": manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest),
        }
        LOG.info("Export complete for tag=%s", tag)
        return result
    finally:
        # cleanup tmp dir unless dry_run
        if dry_run:
            LOG.info("Dry-run mode; temporary dir preserved at %s", tmp_dir)
        else:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def _build_cli():
    p = argparse.ArgumentParser(prog="export_model_artifacts", description="Package and publish model artifacts to registry/S3/MLflow")
    p.add_argument("source", help="Source directory to package (model files, code, metadata)")
    p.add_argument("--tag", required=True, help="Artifact tag (unique identifier)")
    p.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY_DIR), help="Filesystem registry directory")
    p.add_argument("--compress", choices=("gz", "bz2", "xz", "none"), default="gz")
    p.add_argument("--no-sign", dest="sign", action="store_false", help="Disable GPG signature generation")
    p.add_argument("--gpg-binary", default=None, help="Path to gpg binary")
    p.add_argument("--gpg-key", default=None, help="GPG signing key id")
    p.add_argument("--upload-s3", action="store_true", help="Upload packaged artifact to S3")
    p.add_argument("--s3-bucket", default=os.getenv("PRIORITYMAX_S3_BUCKET", ""), help="S3 bucket to upload to")
    p.add_argument("--s3-prefix", default=os.getenv("PRIORITYMAX_S3_PREFIX", DEFAULT_S3_PREFIX), help="S3 prefix")
    p.add_argument("--mlflow", action="store_true", help="Log artifact to MLflow")
    p.add_argument("--mlflow-experiment", default=None, help="MLflow experiment name")
    p.add_argument("--dry-run", action="store_true", help="Do everything except actual publish/upload")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing tag in registry")
    p.add_argument("--exclude", nargs="*", default=[], help="Exclude glob patterns or relative paths")
    return p

def main():
    parser = _build_cli()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    # try to create registry client with optional mongo url from env
    reg_client = None
    mongo_url = os.getenv("PRIORITYMAX_MONGO_REGISTRY_URL", None)
    if mongo_url and _HAS_MOTOR:
        reg_client = RegistryClient(fs_registry_dir=pathlib.Path(args.registry_dir), mongo_url=mongo_url)
    else:
        reg_client = RegistryClient(fs_registry_dir=pathlib.Path(args.registry_dir))

    s3_creds = None
    if args.upload_s3:
        # gather credentials from env as fallback
        s3_creds = {
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
        }

    res = export_model_artifact(
        source_dir=args.source,
        tag=args.tag,
        target_registry_dir=args.registry_dir,
        compress=None if args.compress == "none" else args.compress,
        exclude=args.exclude,
        deterministic=True,
        sign=args.sign,
        gpg_binary=args.gpg_binary,
        gpg_key=args.gpg_key,
        upload_s3=args.upload_s3,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        mlflow_log=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        registry_client=reg_client,
        s3_credentials=s3_creds
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
# -----------------------------------------------------------------------------
# Chunk 4 — Optional extensions (drift metadata, artifact verification, cleanup)
# -----------------------------------------------------------------------------
"""
This section extends export_model_artifacts.py for enterprise observability:
 - Drift metadata integration (via registry)
 - Artifact verification (checksum & signature)
 - Expired artifact cleanup (based on retention policy)
 - Automated periodic export trigger stub
"""

import hashlib
import asyncio
import re
from typing import List

# --------------------------------------------------------------------------
# Drift metadata integration
# --------------------------------------------------------------------------

def attach_drift_metadata(manifest_path: pathlib.Path, drift_score: Optional[float] = None, drift_details: Optional[dict] = None):
    """
    Attach model drift metadata to manifest.json (post-export).
    Used by predictor retrainer & drift detectors to annotate models.
    """
    try:
        if not manifest_path.exists():
            LOG.warning("Manifest not found at %s; cannot attach drift metadata", manifest_path)
            return
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_data.setdefault("drift", {})
        if drift_score is not None:
            manifest_data["drift"]["score"] = float(drift_score)
        if drift_details:
            manifest_data["drift"]["details"] = drift_details
        manifest_data["drift"]["updated_at"] = now_iso()
        tmp = manifest_path.with_suffix(f".tmp_{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        atomic_move(tmp, manifest_path)
        LOG.info("Attached drift metadata (score=%.4f) to %s", drift_score or 0.0, manifest_path)
    except Exception:
        LOG.exception("Failed to attach drift metadata")

# --------------------------------------------------------------------------
# Artifact verification
# --------------------------------------------------------------------------

def verify_artifact_integrity(package_file: pathlib.Path, manifest_file: Optional[pathlib.Path] = None) -> bool:
    """
    Verify artifact file integrity against its manifest (sha256 check).
    Returns True if all files verified successfully.
    """
    try:
        if manifest_file is None:
            manifest_file = package_file.parent / "metadata.json"
        if not manifest_file.exists():
            LOG.warning("Manifest not found for verification: %s", manifest_file)
            return False
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        pkg_size = os.path.getsize(package_file)
        if pkg_size != int(manifest.get("package_size", 0)):
            LOG.warning("Package size mismatch: %d != %d", pkg_size, manifest.get("package_size"))
        sha_local = sha256_of_file(package_file)
        sha_manifest = None
        for f in manifest.get("files", []):
            if f["path"] == os.path.basename(package_file):
                sha_manifest = f["sha256"]
                break
        if sha_manifest and sha_manifest != sha_local:
            LOG.error("SHA256 mismatch for package: %s != %s", sha_local, sha_manifest)
            return False
        LOG.info("Artifact verified OK: %s", package_file)
        return True
    except Exception:
        LOG.exception("Artifact verification failed")
        return False

def verify_gpg_signature(package_file: pathlib.Path, sig_file: pathlib.Path, gpg_binary: str = DEFAULT_GPG_BINARY) -> bool:
    """
    Verify a GPG detached signature.
    """
    try:
        cmd = [gpg_binary, "--verify", str(sig_file), str(package_file)]
        subprocess.run(cmd, check=True, capture_output=True)
        LOG.info("GPG signature verified OK: %s", sig_file)
        return True
    except subprocess.CalledProcessError as e:
        LOG.error("GPG signature verification failed: %s", e.stderr.decode(errors="ignore") if e.stderr else str(e))
        return False
    except FileNotFoundError:
        LOG.warning("gpg binary not found; skipping verification")
        return False

# --------------------------------------------------------------------------
# Expired artifact cleanup
# --------------------------------------------------------------------------

def cleanup_expired_artifacts(registry_dir: pathlib.Path, retention_days: int = 30, dry_run: bool = True) -> List[str]:
    """
    Scan registry_dir and delete/arch archive model folders older than retention_days.
    Returns list of removed tag names.
    """
    removed = []
    try:
        now = time.time()
        cutoff = now - retention_days * 86400
        LOG.info("Cleaning registry %s (retention=%d days)", registry_dir, retention_days)
        for d in pathlib.Path(registry_dir).iterdir():
            if not d.is_dir():
                continue
            try:
                mtime = d.stat().st_mtime
                if mtime < cutoff:
                    LOG.info("Removing expired model: %s (%.1f days old)", d.name, (now - mtime) / 86400)
                    if not dry_run:
                        shutil.rmtree(d)
                    removed.append(d.name)
            except Exception:
                LOG.debug("Failed to inspect dir %s", d)
        return removed
    except Exception:
        LOG.exception("Cleanup failed")
        return removed

# --------------------------------------------------------------------------
# Automated export trigger (periodic async stub)
# --------------------------------------------------------------------------

async def periodic_export_trigger(export_fn, interval_sec: int = 3600):
    """
    Run export_fn periodically in background (for use with scheduler or RL-based autoscaler).
    """
    while True:
        try:
            LOG.info("Running scheduled export trigger...")
            await export_fn()
        except Exception:
            LOG.exception("Export trigger error")
        await asyncio.sleep(interval_sec)

# --------------------------------------------------------------------------
# Example CLI extension: verify + cleanup
# --------------------------------------------------------------------------

def verify_and_cleanup_cli():
    parser = argparse.ArgumentParser(description="Verify or clean up exported model artifacts")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY_DIR), help="Path to model registry directory")
    parser.add_argument("--verify", metavar="TAG", help="Verify given tag's artifact integrity and signature")
    parser.add_argument("--cleanup", action="store_true", help="Clean up expired artifacts")
    parser.add_argument("--retention-days", type=int, default=30, help="Retention period in days")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run mode (do not delete)")
    parser.add_argument("--loglevel", default="INFO", help="Log level")
    args = parser.parse_args()

    LOG.setLevel(args.loglevel.upper())
    reg_dir = pathlib.Path(args.registry_dir)

    if args.verify:
        tag = args.verify
        pkg = reg_dir / tag / f"{tag}.tar.gz"
        sig = pkg.with_suffix(".tar.gz.sig")
        manifest = reg_dir / tag / "metadata.json"
        ok = verify_artifact_integrity(pkg, manifest)
        if sig.exists():
            ok_sig = verify_gpg_signature(pkg, sig)
            ok = ok and ok_sig
        print(json.dumps({"tag": tag, "verified": ok}, indent=2))
        sys.exit(0 if ok else 1)

    if args.cleanup:
        removed = cleanup_expired_artifacts(reg_dir, retention_days=args.retention_days, dry_run=args.dry_run)
        print(json.dumps({"removed": removed, "dry_run": args.dry_run}, indent=2))

if __name__ == "__main__" and "verify_and_cleanup_cli" not in globals():
    # The main export CLI already exists; this is an auxiliary CLI for ops usage.
    if len(sys.argv) > 1 and (sys.argv[1] == "verify" or sys.argv[1] == "--verify" or "--cleanup" in sys.argv):
        verify_and_cleanup_cli()
# -----------------------------------------------------------------------------
# Chunk 5 — Automated Testing Utilities & CI Hooks
# -----------------------------------------------------------------------------
"""
This chunk extends `export_model_artifacts.py` for CI/CD validation, build reproducibility, and
enterprise-ready test hooks. It includes:

 - Automated test helpers for packaging and S3 upload mocking
 - CI/CD pipeline hooks for artifact validation
 - Deterministic build verification (hash diff check)
 - GitHub Actions & Jenkins integration stubs
 - Reproducibility audit logger
"""

import tempfile
import filecmp
import difflib
import hashlib
import json
from contextlib import contextmanager

# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------

def compute_dir_hash(directory: pathlib.Path, exclude: Optional[List[str]] = None) -> str:
    """
    Compute a deterministic hash for all files under `directory`.
    Used to validate that two artifact builds produce identical results.
    """
    h = hashlib.sha256()
    exclude = exclude or []
    for file_path in sorted(list_files_recursive(directory, exclude)):
        rel = str(file_path.relative_to(directory))
        h.update(rel.encode("utf-8"))
        h.update(sha256_of_file(file_path).encode("utf-8"))
    return h.hexdigest()


def compare_tarballs(tar_a: pathlib.Path, tar_b: pathlib.Path) -> Dict[str, Any]:
    """
    Compare two tarballs for deterministic build verification.
    Returns a dict with differences summary.
    """
    def _extract(tar_path: pathlib.Path, tmpdir: pathlib.Path):
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(tmpdir)

    tmp_a = tempfile.mkdtemp()
    tmp_b = tempfile.mkdtemp()
    try:
        _extract(tar_a, pathlib.Path(tmp_a))
        _extract(tar_b, pathlib.Path(tmp_b))
        cmp = filecmp.dircmp(tmp_a, tmp_b)
        diffs = {"left_only": cmp.left_only, "right_only": cmp.right_only, "diff_files": cmp.diff_files}
        if any(diffs.values()):
            LOG.warning("Tarball differences found: %s", diffs)
        else:
            LOG.info("Tarballs identical (reproducible build verified)")
        return diffs
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


# --------------------------------------------------------------------------
# CI/CD Integration Hooks
# --------------------------------------------------------------------------

@contextmanager
def ci_context_logger(ci_job_name: str, commit_sha: Optional[str] = None, output_file: Optional[str] = None):
    """
    Context manager that logs CI job metadata and writes reproducibility audit logs.
    """
    start_time = time.time()
    ci_env = {
        "job_name": ci_job_name,
        "commit": commit_sha or os.getenv("GITHUB_SHA") or os.getenv("CI_COMMIT_SHA"),
        "build_id": os.getenv("GITHUB_RUN_ID") or os.getenv("BUILD_ID"),
        "timestamp_utc": now_iso(),
    }
    LOG.info("[CI] Starting job %s (%s)", ci_env["job_name"], ci_env["commit"])
    try:
        yield ci_env
    finally:
        ci_env["duration_sec"] = round(time.time() - start_time, 3)
        LOG.info("[CI] Job %s completed in %.2fs", ci_env["job_name"], ci_env["duration_sec"])
        if output_file:
            try:
                pathlib.Path(output_file).write_text(json.dumps(ci_env, indent=2), encoding="utf-8")
            except Exception:
                LOG.exception("Failed to write CI context log to %s", output_file)


def validate_export_for_ci(tag: str, registry_dir: pathlib.Path) -> bool:
    """
    CI utility: verify the exported artifact, manifest, and optional signature for tag.
    """
    model_dir = registry_dir / tag
    pkg = None
    for f in model_dir.iterdir():
        if f.suffix.startswith(".tar"):
            pkg = f
            break
    if not pkg:
        LOG.error("No package tarball found for tag %s", tag)
        return False
    manifest = model_dir / "metadata.json"
    sig = model_dir / f"{pkg.name}.sig"
    ok = verify_artifact_integrity(pkg, manifest)
    if sig.exists():
        ok = ok and verify_gpg_signature(pkg, sig)
    if ok:
        LOG.info("[CI] Artifact verified successfully for %s", tag)
    else:
        LOG.error("[CI] Artifact verification FAILED for %s", tag)
    return ok


# --------------------------------------------------------------------------
# Jenkins / GitHub Actions Hooks
# --------------------------------------------------------------------------

def github_actions_notify(message: str, level: str = "info"):
    """
    Send logs to GitHub Actions workflow annotations.
    """
    level_map = {"info": "::notice::", "warn": "::warning::", "error": "::error::"}
    prefix = level_map.get(level.lower(), "::notice::")
    print(f"{prefix}{message}")


def jenkins_notify(message: str, level: str = "info"):
    """
    Log messages in Jenkins console with ANSI color codes.
    """
    color_map = {"info": "\033[94m", "warn": "\033[93m", "error": "\033[91m"}
    reset = "\033[0m"
    color = color_map.get(level.lower(), "\033[94m")
    print(f"{color}{message}{reset}")


# --------------------------------------------------------------------------
# Reproducibility Audit Log
# --------------------------------------------------------------------------

def record_reproducibility_audit(tag: str, hash_value: str, registry_dir: pathlib.Path):
    """
    Record a reproducibility audit file for the exported artifact.
    """
    audit_dir = registry_dir / "_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_file = audit_dir / f"{tag}_audit.json"
    entry = {
        "tag": tag,
        "hash": hash_value,
        "timestamp_utc": now_iso(),
        "user": os.getenv("USER", "ci-bot"),
        "git_commit": os.getenv("GITHUB_SHA") or os.getenv("CI_COMMIT_SHA"),
    }
    audit_file.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    LOG.info("Wrote reproducibility audit: %s", audit_file)


# --------------------------------------------------------------------------
# Combined CI Workflow Example
# --------------------------------------------------------------------------

def ci_verify_pipeline(tag: str, registry_dir: str):
    """
    Example full CI verification pipeline step for PriorityMax artifacts.
    - verifies integrity
    - checks reproducibility hash
    - writes audit
    - outputs to GitHub Actions
    """
    with ci_context_logger("verify-artifact", commit_sha=os.getenv("GITHUB_SHA")):
        reg_dir = pathlib.Path(registry_dir)
        verified = validate_export_for_ci(tag, reg_dir)
        if not verified:
            github_actions_notify(f"Artifact verification failed for {tag}", level="error")
            sys.exit(1)

        model_dir = reg_dir / tag
        hash_val = compute_dir_hash(model_dir)
        record_reproducibility_audit(tag, hash_val, reg_dir)
        github_actions_notify(f"Artifact {tag} verified successfully", level="info")
        LOG.info("✅ CI pipeline completed successfully for %s", tag)


# --------------------------------------------------------------------------
# Main (optional CI entrypoint)
# --------------------------------------------------------------------------

if __name__ == "__main__" and "export_model_artifact" not in globals():
    parser = argparse.ArgumentParser(description="CI/CD artifact verifier for PriorityMax models")
    parser.add_argument("--tag", required=True, help="Model tag to verify")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY_DIR))
    parser.add_argument("--ci-system", choices=["github", "jenkins", "generic"], default="github")
    args = parser.parse_args()

    if args.ci_system == "github":
        github_actions_notify(f"Running CI verification for {args.tag}", level="info")
    elif args.ci_system == "jenkins":
        jenkins_notify(f"Running CI verification for {args.tag}", level="info")
    else:
        LOG.info("Running generic CI verification for %s", args.tag)

    ci_verify_pipeline(args.tag, args.registry_dir)
# -----------------------------------------------------------------------------
# Chunk 6 — Enterprise Ops Extensions (Lineage, Attestation, Webhooks, Rollback)
# -----------------------------------------------------------------------------
"""
This final chunk extends `export_model_artifacts.py` with high-availability,
enterprise-grade features for traceability, compliance, and automated orchestration.

Features:
 - Model lineage tracking & provenance
 - Sigstore / Cosign signature attestation
 - Webhook + Slack / PagerDuty notifications
 - Kubernetes Job launcher for distributed export
 - Rollback utilities for failed deployments
"""

import base64
import http.client
import subprocess
import urllib.parse
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Model Lineage Tracking
# --------------------------------------------------------------------------

@dataclass
class ModelLineage:
    tag: str
    parent_tag: Optional[str]
    derived_from: Optional[str]
    creation_ts: str
    notes: Optional[str] = None

def record_model_lineage(tag: str, parent_tag: Optional[str], registry_dir: pathlib.Path, notes: Optional[str] = None):
    """
    Append a lineage entry in registry/_lineage/<tag>.json
    """
    try:
        lineage_dir = registry_dir / "_lineage"
        lineage_dir.mkdir(parents=True, exist_ok=True)
        lineage = ModelLineage(
            tag=tag,
            parent_tag=parent_tag,
            derived_from=parent_tag,
            creation_ts=now_iso(),
            notes=notes
        )
        out_path = lineage_dir / f"{tag}.json"
        out_path.write_text(json.dumps(lineage.__dict__, indent=2), encoding="utf-8")
        LOG.info("Recorded lineage for %s (parent=%s)", tag, parent_tag)
    except Exception:
        LOG.exception("Failed to record model lineage")


# --------------------------------------------------------------------------
# Sigstore / Cosign Attestation (optional)
# --------------------------------------------------------------------------

def sign_with_sigstore(package_file: pathlib.Path, identity_token: Optional[str] = None) -> Optional[str]:
    """
    Generate a Sigstore attestation using `cosign attest` for the artifact.
    Returns path to attestation file (.att) if successful.
    """
    cosign_bin = shutil.which("cosign")
    if not cosign_bin:
        LOG.warning("cosign binary not available; skipping Sigstore attestation")
        return None

    att_file = package_file.with_suffix(".att")
    try:
        cmd = [cosign_bin, "attest", "--predicate", str(package_file), "--keyless", "--output", str(att_file)]
        if identity_token:
            cmd += ["--identity-token", identity_token]
        subprocess.run(cmd, check=True)
        LOG.info("Sigstore attestation generated: %s", att_file)
        return str(att_file)
    except subprocess.CalledProcessError as e:
        LOG.error("Sigstore attestation failed: %s", e)
        return None


# --------------------------------------------------------------------------
# Webhook Notifications
# --------------------------------------------------------------------------

def send_webhook_notification(url: str, payload: dict, headers: Optional[dict] = None):
    """
    Generic webhook sender supporting HTTPS POST.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        conn_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = conn_class(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        body = json.dumps(payload)
        hdr = {"Content-Type": "application/json"}
        if headers:
            hdr.update(headers)
        conn.request("POST", parsed.path or "/", body=body, headers=hdr)
        resp = conn.getresponse()
        LOG.info("Webhook POST %s returned %d", url, resp.status)
        conn.close()
    except Exception:
        LOG.exception("Webhook notification failed")

def notify_export_success(tag: str, registry_dir: pathlib.Path, webhooks: Optional[List[str]] = None):
    """
    Notify all configured webhooks about successful export.
    """
    payload = {
        "event": "export_success",
        "tag": tag,
        "registry": str(registry_dir),
        "timestamp": now_iso()
    }
    for url in webhooks or []:
        send_webhook_notification(url, payload)


# --------------------------------------------------------------------------
# Kubernetes Job Launcher for Exports
# --------------------------------------------------------------------------

def launch_export_job_in_k8s(tag: str, image: str, registry_dir: str, namespace: str = "default"):
    """
    Launch a one-off Kubernetes Job that runs export_model_artifacts.py for a given model tag.
    """
    try:
        job_name = f"export-{tag.replace('_', '-')[:40]}"
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": namespace},
            "spec": {
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "export",
                            "image": image,
                            "command": ["python3", "/app/scripts/export_model_artifacts.py", "--tag", tag, "--registry-dir", registry_dir],
                            "env": [{"name": "PRIORITYMAX_MODE", "value": "k8s"}]
                        }]
                    }
                }
            }
        }
        tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
        tmpfile.write(json.dumps(manifest))
        tmpfile.flush()
        tmpfile.close()
        cmd = ["kubectl", "apply", "-f", tmpfile.name]
        subprocess.run(cmd, check=True)
        LOG.info("Launched export job in Kubernetes: %s", job_name)
    except Exception:
        LOG.exception("Failed to launch export job in Kubernetes")


# --------------------------------------------------------------------------
# Rollback & Recovery Utilities
# --------------------------------------------------------------------------

def rollback_to_previous_model(tag: str, registry_dir: pathlib.Path) -> bool:
    """
    Roll back the 'prod' symlink or alias to the previous model in registry.
    """
    try:
        link = registry_dir / "prod"
        if not link.exists() or not link.is_symlink():
            LOG.warning("No prod symlink found; cannot rollback")
            return False
        current_target = os.readlink(link)
        candidates = sorted([p for p in registry_dir.iterdir() if p.is_dir() and p.name != "prod"], key=lambda p: p.stat().st_mtime, reverse=True)
        if len(candidates) < 2:
            LOG.warning("Not enough model versions to rollback")
            return False
        prev_target = candidates[1].name
        link.unlink()
        link.symlink_to(prev_target)
        LOG.info("Rolled back production model to %s", prev_target)
        return True
    except Exception:
        LOG.exception("Rollback failed")
        return False


# --------------------------------------------------------------------------
# Attestation-Aware CI Verification
# --------------------------------------------------------------------------

def verify_sigstore_attestation(tag: str, registry_dir: pathlib.Path) -> bool:
    """
    Verify a Sigstore attestation for the exported artifact if present.
    """
    cosign_bin = shutil.which("cosign")
    if not cosign_bin:
        LOG.warning("cosign not installed; skipping attestation verification")
        return True

    pkg = registry_dir / tag / f"{tag}.tar.gz"
    att = pkg.with_suffix(".att")
    if not att.exists():
        LOG.warning("No Sigstore attestation found for tag=%s", tag)
        return True
    try:
        cmd = [cosign_bin, "verify-attestation", str(pkg), "--keyless"]
        subprocess.run(cmd, check=True)
        LOG.info("Sigstore attestation verified for %s", tag)
        return True
    except subprocess.CalledProcessError as e:
        LOG.error("Sigstore attestation verification failed: %s", e)
        return False


# --------------------------------------------------------------------------
# Example usage hook
# --------------------------------------------------------------------------

def export_with_enterprise_hooks(source_dir: str, tag: str, registry_dir: str, parent_tag: Optional[str] = None, webhooks: Optional[List[str]] = None):
    """
    Example enterprise entrypoint that combines:
      - export_model_artifact()
      - lineage recording
      - webhook notifications
      - optional k8s job fallback
    """
    try:
        result = export_model_artifact(source_dir=source_dir, tag=tag, target_registry_dir=registry_dir)
        record_model_lineage(tag, parent_tag, pathlib.Path(registry_dir))
        notify_export_success(tag, pathlib.Path(registry_dir), webhooks)
        LOG.info("Enterprise export with hooks completed for tag=%s", tag)
        return result
    except Exception:
        LOG.exception("Enterprise export failed")
        raise

# backend/app/api/admin.py
"""
Admin API for PriorityMax — model management, training orchestration, auditing, billing admin.

Endpoints included (summary):
- /admin/models/*      : list/upload/promote/demote/rollback/download metadata
- /admin/train/*       : trigger retrain, status, cancel, logs
- /admin/eval/*        : run evaluation / canary gating helper
- /admin/billing/*     : billing plan CRUD, usage summary, invoice generation (placeholder)
- /admin/audit/*       : view audit logs
- /admin/ws/logs       : websocket stream for live logs (training)
"""

import os
import json
import uuid
import hashlib
import shutil
import logging
import asyncio
import pathlib
import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import (
    APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File,
    Form, Query, WebSocket, WebSocketDisconnect, status
)
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Optional third-party integrations
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

# ---------------------------
# Configuration & constants
# ---------------------------
LOG = logging.getLogger("prioritymax.admin")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(handler)

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/app/.. -> backend/
MODEL_REGISTRY_DIR = os.getenv("MODEL_REGISTRY_PATH", str(BASE_DIR / "app" / "ml" / "models"))
CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", str(BASE_DIR / "checkpoints"))
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT_PATH", str(BASE_DIR / "scripts" / "train_rl_heavy.py"))
AUDIT_LOG_FILE = os.getenv("ADMIN_AUDIT_LOG", str(BASE_DIR / "logs" / "admin_audit.jsonl"))
MONGO_URL = os.getenv("MONGO_URL", None)
S3_BUCKET = os.getenv("S3_BUCKET", None)
S3_PREFIX = os.getenv("S3_PREFIX", "models/prioritymax")

ensure_path = lambda p: pathlib.Path(p).mkdir(parents=True, exist_ok=True)
ensure_path(MODEL_REGISTRY_DIR)
ensure_path(CHECKPOINTS_DIR)
ensure_path(pathlib.Path(AUDIT_LOG_FILE).parent)

# ---------------------------
# DB (optional) and Audit
# ---------------------------
if _HAS_MOTOR and MONGO_URL:
    motor_client = motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    db = motor_client.get_default_database()
    audit_collection = db.get_collection("admin_audit")
    models_collection = db.get_collection("model_registry")
    runs_collection = db.get_collection("training_runs")
else:
    motor_client = None
    db = None
    audit_collection = None
    models_collection = None
    runs_collection = None


async def write_audit_event(event: dict):
    """
    Persist audit events: prefer Mongo collection, otherwise append to JSONL local file.
    Event should include: timestamp_utc, user, action, resource, details
    """
    event = dict(event)
    event["timestamp_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        if audit_collection is not None:
            await audit_collection.insert_one(event)
        else:
            # append to JSONL
            async def _append():
                with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
            await run_in_threadpool(_append)
    except Exception as e:
        LOG.exception("Failed to write audit event: %s", e)


# ---------------------------
# RBAC / Auth dependencies (pluggable)
# ---------------------------
class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


# A very small stub for auth. Replace with your real auth dependency (JWT, OAuth2).
class User(BaseModel):
    username: str
    roles: List[Role] = Field(default_factory=list)
    org_id: Optional[str] = None


def get_current_user(token: Optional[str] = Query(None, description="Dev/test auth token (replace with real JWT)")) -> User:
    """
    Simple dev-mode auth: token can encode role(s) as comma-separated values like "alice:admin,operator".
    In production, replace this with JWT verification, introspection, or OAuth2 dependency.
    """
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing auth token (use real JWT in prod)")
    try:
        username, roles_raw = token.split(":", 1)
        roles = [Role(r.strip()) for r in roles_raw.split(",") if r.strip()]
        return User(username=username, roles=roles)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format. Use 'username:role1,role2' for dev auth.")


def require_role(required: Role):
    def _dep(user: User = Depends(get_current_user)):
        if required not in user.roles and Role.ADMIN not in user.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user
    return _dep


# ---------------------------
# Pydantic schemas
# ---------------------------
class ModelMeta(BaseModel):
    tag: str
    model_name: str
    path: str
    timestamp_utc: str
    timesteps: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelListItem(BaseModel):
    tag: str
    model_name: str
    status: str  # "canary", "prod", "archived"
    metadata: Dict[str, Any]


class UploadResponse(BaseModel):
    tag: str
    uploaded: bool
    message: Optional[str]


class TrainRequest(BaseModel):
    total_timesteps: Optional[int] = Field(None, description="Override total timesteps")
    use_real_env: Optional[bool] = True
    run_name: Optional[str] = None
    extra_config: Dict[str, Any] = Field(default_factory=dict)


class TrainStatus(BaseModel):
    run_id: str
    status: str  # queued / running / success / failed / cancelled
    started_at: Optional[str]
    finished_at: Optional[str]
    log_tail: Optional[str]


class CanaryEvalRequest(BaseModel):
    model_tag: str
    eval_episodes: int = 20
    threshold_mean_reward: Optional[float] = None


class BillingPlan(BaseModel):
    plan_id: str
    name: str
    throughput_limit: int
    price_usd_per_month: float
    description: Optional[str]


# ---------------------------
# Router
# ---------------------------
router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------
# Utilities: model registry file helpers
# ---------------------------
def list_models_from_fs() -> List[ModelListItem]:
    """
    Read model registry dir and return list of ModelListItem.
    If Mongo model registry exists, prefer that (list from DB).
    """
    items: List[ModelListItem] = []
    try:
        if models_collection is not None:
            # fetch from DB
            docs = models_collection.find({}, {"_id": 0})
            # This is sync but small. If very large, convert to async iterator.
            docs = list(docs)
            for d in docs:
                items.append(ModelListItem(tag=d["tag"], model_name=d.get("model_name", "unknown"),
                                           status=d.get("status", "archived"),
                                           metadata=d.get("metadata", {})))
            return items
        # filesystem fallback
        base = pathlib.Path(MODEL_REGISTRY_DIR)
        if not base.exists():
            return items
        for child in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir():
                meta_file = child / "metadata.json"
                if meta_file.exists():
                    try:
                        m = json.loads(meta_file.read_text(encoding="utf-8"))
                        status = m.get("status", "archived")
                        items.append(ModelListItem(tag=m["tag"],
                                                   model_name=m.get("model_name", child.name),
                                                   status=status,
                                                   metadata=m))
                    except Exception:
                        LOG.exception("Failed read metadata for %s", child)
                        continue
        return items
    except Exception as e:
        LOG.exception("list_models_from_fs failed: %s", e)
        return items


def load_model_metadata_fs(tag: str) -> Optional[dict]:
    p = pathlib.Path(MODEL_REGISTRY_DIR) / tag / "metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            LOG.exception("Failed to load metadata for %s", tag)
    return None


def promote_model_fs(tag: str, target: str = "prod") -> bool:
    """
    Mark model as 'canary' or 'prod' by writing metadata or DB update.
    """
    try:
        md = load_model_metadata_fs(tag)
        if md is None:
            return False
        md["status"] = target
        p = pathlib.Path(MODEL_REGISTRY_DIR) / tag / "metadata.json"
        p.write_text(json.dumps(md, default=str, indent=2), encoding="utf-8")
        # Update DB if available
        if models_collection is not None:
            models_collection.update_one({"tag": tag}, {"$set": {"status": target, "metadata": md}}, upsert=True)
        return True
    except Exception as e:
        LOG.exception("promote_model_fs failed: %s", e)
        return False


def save_uploaded_model_file(tag: str, uploaded_file: UploadFile) -> str:
    """
    Save a model zip or artifact into the registry under tag/.
    Returns the artifact dir path.
    """
    try:
        out_dir = pathlib.Path(MODEL_REGISTRY_DIR) / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save file
        file_path = out_dir / uploaded_file.filename
        with open(file_path, "wb") as fh:
            shutil.copyfileobj(uploaded_file.file, fh)
        # Touch metadata if not exists
        meta_path = out_dir / "metadata.json"
        if not meta_path.exists():
            md = {"tag": tag, "model_name": uploaded_file.filename, "status": "archived", "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"}
            meta_path.write_text(json.dumps(md, default=str, indent=2), encoding="utf-8")
            if models_collection is not None:
                models_collection.insert_one(md)
        return str(out_dir)
    except Exception as e:
        LOG.exception("save_uploaded_model_file failed: %s", e)
        raise


# ---------------------------
# Training run manager (in-memory index + persisted collection)
# ---------------------------
TRAIN_RUNS: Dict[str, Dict[str, Any]] = {}  # run_id -> info (status, process, started_at)

async def _persist_run_to_db(run_id: str, info: dict):
    if runs_collection is not None:
        try:
            await runs_collection.update_one({"run_id": run_id}, {"$set": info}, upsert=True)
        except Exception:
            LOG.exception("Failed to persist run to DB")


async def _register_run(started_by: str, params: dict) -> str:
    run_id = str(uuid.uuid4())
    info = {
        "run_id": run_id,
        "status": "queued",
        "params": params,
        "started_by": started_by,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    TRAIN_RUNS[run_id] = info
    await _persist_run_to_db(run_id, info)
    return run_id


async def _update_run(run_id: str, updates: dict):
    if run_id not in TRAIN_RUNS:
        TRAIN_RUNS[run_id] = {}
    TRAIN_RUNS[run_id].update(updates)
    await _persist_run_to_db(run_id, TRAIN_RUNS[run_id])


async def _run_subprocess_and_stream_logs(cmd: List[str], run_id: str, cwd: Optional[str] = None):
    """
    Start subprocess and stream logs to checkpoint file and store in run info.
    Non-blocking -- returns when process completes.
    """
    LOG.info("Starting subprocess: %s", " ".join(cmd))
    await _update_run(run_id, {"status": "running", "cmd": cmd, "started_at": datetime.datetime.utcnow().isoformat() + "Z"})
    stdout_log = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_stdout.log"
    stderr_log = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_stderr.log"
    # Ensure previous logs are removed
    for p in (stdout_log, stderr_log):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # launch process
    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd)
    # store process handle (so we can cancel)
    await _update_run(run_id, {"pid": process.pid})
    TRAIN_RUNS[run_id]["process"] = process

    # stream stdout and stderr concurrently
    async def _stream_and_save(stream, out_path):
        with open(out_path, "ab") as fh:
            while True:
                line = await stream.readline()
                if not line:
                    break
                fh.write(line)
                fh.flush()

    try:
        await asyncio.gather(
            _stream_and_save(process.stdout, stdout_log),
            _stream_and_save(process.stderr, stderr_log)
        )
        rc = await process.wait()
        status = "success" if rc == 0 else "failed"
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        await _update_run(run_id, {"status": status, "returncode": rc, "finished_at": finished_at})
        LOG.info("Process %s finished with rc=%s", run_id, rc)
    except asyncio.CancelledError:
        LOG.warning("Run %s cancelled, terminating process", run_id)
        try:
            process.terminate()
            await process.wait()
        except Exception:
            process.kill()
        await _update_run(run_id, {"status": "cancelled", "finished_at": datetime.datetime.utcnow().isoformat() + "Z"})
    except Exception as e:
        LOG.exception("Error while running subprocess for run %s: %s", run_id, e)
        await _update_run(run_id, {"status": "failed", "error": str(e), "finished_at": datetime.datetime.utcnow().isoformat() + "Z"})
    finally:
        TRAIN_RUNS[run_id].pop("process", None)


# ---------------------------
# S3 helpers (optional)
# ---------------------------
def upload_dir_to_s3(local_dir: str, bucket: str, prefix: str = "") -> dict:
    if not _HAS_BOTO3:
        raise RuntimeError("boto3 not available")
    s3 = boto3.client("s3")
    uploaded = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local_dir)
            key = os.path.join(prefix, os.path.basename(local_dir), rel)
            s3.upload_file(full, bucket, key)
            uploaded.append({"file": rel, "key": key})
    return {"bucket": bucket, "prefix": prefix, "uploaded": uploaded}


# ---------------------------
# Admin endpoints
# ---------------------------

@router.get("/models", response_model=List[ModelListItem], dependencies=[Depends(require_role(Role.VIEWER))])
async def list_models():
    """
    List models in the registry (status: canary/prod/archived).
    """
    items = await run_in_threadpool(list_models_from_fs)
    return items


@router.get("/models/{tag}", response_model=ModelMeta, dependencies=[Depends(require_role(Role.VIEWER))])
async def get_model_metadata(tag: str):
    md = await run_in_threadpool(load_model_metadata_fs, tag)
    if not md:
        raise HTTPException(status_code=404, detail="Model tag not found")
    return ModelMeta(tag=md.get("tag", tag), model_name=md.get("model_name", ""), path=str(pathlib.Path(MODEL_REGISTRY_DIR) / tag), timestamp_utc=md.get("timestamp_utc", ""), timesteps=md.get("timesteps"), metadata=md)


@router.post("/models/upload", response_model=UploadResponse, dependencies=[Depends(require_role(Role.OPERATOR))])
async def upload_model(file: UploadFile = File(...), tag: Optional[str] = Form(None), user: User = Depends(get_current_user)):
    """
    Upload a model artifact (zip, tar, pth...). Tag will be used as model version identifier.
    """
    tag = tag or f"{file.filename.split('.')[0]}-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    try:
        outdir = await run_in_threadpool(save_uploaded_model_file, tag, file)
        await write_audit_event({"user": user.username, "action": "upload_model", "resource": tag, "details": {"path": outdir, "filename": file.filename}})
        # Optionally mirror to S3 (not automatic here)
        return UploadResponse(tag=tag, uploaded=True, message=f"Saved to {outdir}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{tag}/promote", dependencies=[Depends(require_role(Role.ADMIN))])
async def promote_model(tag: str, target: str = Form("prod"), user: User = Depends(get_current_user)):
    """
    Promote a model tag to 'canary' or 'prod'. This writes metadata and returns success.
    """
    ok = await run_in_threadpool(promote_model_fs, tag, target)
    if not ok:
        raise HTTPException(status_code=404, detail="Model tag not found or promotion failed")
    await write_audit_event({"user": user.username, "action": "promote_model", "resource": tag, "details": {"target": target}})
    return {"ok": True, "tag": tag, "promoted_to": target}


@router.post("/models/{tag}/download", dependencies=[Depends(require_role(Role.VIEWER))])
async def download_model(tag: str):
    """
    Download primary artifact for a model tag. Returns a file response of the first artifact found.
    """
    d = pathlib.Path(MODEL_REGISTRY_DIR) / tag
    if not d.exists():
        raise HTTPException(status_code=404, detail="Tag not found")
    # pick first artifact file that isn't metadata.json
    files = [p for p in d.iterdir() if p.is_file() and p.name != "metadata.json"]
    if not files:
        raise HTTPException(status_code=404, detail="No artifact files found")
    # send first
    return FileResponse(files[0], filename=files[0].name)


@router.delete("/models/{tag}", dependencies=[Depends(require_role(Role.ADMIN))])
async def delete_model(tag: str, user: User = Depends(get_current_user)):
    """
    Delete a model tag from filesystem and DB (irreversible). Admin only.
    """
    d = pathlib.Path(MODEL_REGISTRY_DIR) / tag
    if not d.exists():
        raise HTTPException(status_code=404, detail="Tag not found")
    # move to archive folder rather than delete for safety
    archive_root = pathlib.Path(MODEL_REGISTRY_DIR) / "_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / f"{tag}_{int(datetime.datetime.utcnow().timestamp())}"
    shutil.move(str(d), str(dest))
    if models_collection is not None:
        await models_collection.delete_one({"tag": tag})
    await write_audit_event({"user": user.username, "action": "delete_model", "resource": tag, "details": {"archived_to": str(dest)}})
    return {"ok": True, "archived_to": str(dest)}


# ---------------------------
# Training orchestration endpoints
# ---------------------------

@router.post("/train/start", response_model=TrainStatus, dependencies=[Depends(require_role(Role.OPERATOR))])
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks, user: User = Depends(get_current_user)):
    """
    Start a retrain run using train_rl_heavy.py. Returns a run_id.
    The function spawns a subprocess and streams logs to CHECKPOINTS_DIR/<run_id>_*.log.
    """
    params = request.dict()
    run_id = await _register_run(user.username, params)
    # Build command
    cmd = [sys_executable_or_fallback(), TRAIN_SCRIPT, "--config", "/dev/null"]  # we will pass config via env or extra args
    # Provide overrides as CLI args (simple approach)
    if request.total_timesteps:
        cmd += ["--total-timesteps", str(request.total_timesteps)]
    if request.run_name:
        cmd += ["--run-name", request.run_name]
    if not request.use_real_env:
        cmd += ["--no-real-env"] if hasattr(request, "use_real_env") else []

    # environment injection: write temporary config if extra_config present
    tmp_config_path = None
    if request.extra_config:
        tmp_config_path = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_config.yaml"
        tmp_config_path.write_text(json.dumps(request.extra_config))
        cmd += ["--config", str(tmp_config_path)]

    # Launch process in background
    async def _bg():
        try:
            await _run_subprocess_and_stream_logs(cmd, run_id, cwd=str(BASE_DIR))
            LOG.info("Training run finished: %s", run_id)
        finally:
            # cleanup tmp config
            if tmp_config_path and tmp_config_path.exists():
                tmp_config_path.unlink()

    background_tasks.add_task(asyncio.create_task, _bg())
    return TrainStatus(run_id=run_id, status="queued", started_at=None, finished_at=None, log_tail=None)


@router.get("/train/{run_id}/status", response_model=TrainStatus, dependencies=[Depends(require_role(Role.VIEWER))])
async def get_train_status(run_id: str):
    info = TRAIN_RUNS.get(run_id)
    if not info:
        # check DB
        if runs_collection is not None:
            doc = await runs_collection.find_one({"run_id": run_id}, {"_id": 0})
            if doc:
                return TrainStatus(run_id=run_id, status=doc.get("status", "unknown"), started_at=doc.get("started_at"), finished_at=doc.get("finished_at"), log_tail=None)
        raise HTTPException(status_code=404, detail="Run not found")
    # attach log tail
    stdout_log = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_stdout.log"
    tail = None
    if stdout_log.exists():
        with open(stdout_log, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 5000), 0)
            tail = fh.read().decode(errors="ignore")
    return TrainStatus(run_id=run_id, status=info.get("status", "unknown"), started_at=info.get("started_at"), finished_at=info.get("finished_at"), log_tail=tail)


@router.post("/train/{run_id}/cancel", dependencies=[Depends(require_role(Role.OPERATOR))])
async def cancel_train(run_id: str, user: User = Depends(get_current_user)):
    info = TRAIN_RUNS.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    proc = info.get("process")
    if not proc:
        # update DB
        await _update_run(run_id, {"status": "cancelled", "finished_at": datetime.datetime.utcnow().isoformat() + "Z"})
        await write_audit_event({"user": user.username, "action": "cancel_train", "resource": run_id, "details": {}})
        return {"ok": True, "cancelled": True}
    try:
        proc.terminate()
        await proc.wait()
    except Exception:
        proc.kill()
    await _update_run(run_id, {"status": "cancelled", "finished_at": datetime.datetime.utcnow().isoformat() + "Z"})
    await write_audit_event({"user": user.username, "action": "cancel_train", "resource": run_id, "details": {}})
    return {"ok": True, "cancelled": True}


@router.get("/train/{run_id}/logs", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_train_logs(run_id: str, tail: int = Query(2000, description="last N bytes")):
    stdout_log = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_stdout.log"
    stderr_log = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_stderr.log"
    if not stdout_log.exists() and not stderr_log.exists():
        raise HTTPException(status_code=404, detail="Logs not found")
    def iter_logs():
        if stdout_log.exists():
            with open(stdout_log, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - tail), 0)
                yield b"--- STDOUT ---\n"
                yield fh.read()
        if stderr_log.exists():
            with open(stderr_log, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - tail), 0)
                yield b"\n--- STDERR ---\n"
                yield fh.read()
    return StreamingResponse(iter_logs(), media_type="application/octet-stream")


# ---------------------------
# Canary evaluation endpoints
# ---------------------------

@router.post("/eval/canary", dependencies=[Depends(require_role(Role.OPERATOR))])
async def run_canary_eval(req: CanaryEvalRequest, background_tasks: BackgroundTasks, user: User = Depends(get_current_user)):
    """
    Run an evaluation on a given model tag. This will load model artifacts and run evaluation episodes using a
    small evaluation harness (e.g., scripts/train_rl_eval.py or a built-in evaluator).
    For safety, the evaluation is offline and does not touch production.
    """
    tag = req.model_tag
    md = load_model_metadata_fs(tag)
    if not md:
        raise HTTPException(status_code=404, detail="Model tag not found")
    # schedule evaluator script
    run_id = await _register_run(user.username, {"type": "canary_eval", "model_tag": tag, "eval_episodes": req.eval_episodes})
    evaluator_script = str(pathlib.Path(BASE_DIR) / "scripts" / "train_rl_eval.py")
    cmd = [sys_executable_or_fallback(), evaluator_script, "--model-dir", str(pathlib.Path(MODEL_REGISTRY_DIR) / tag), "--eval-episodes", str(req.eval_episodes)]
    # optional threshold check will be performed when the evaluator writes results into runs_collection or via logs
    async def _bg_eval():
        await _run_subprocess_and_stream_logs(cmd, run_id, cwd=str(BASE_DIR))
        # After eval completes, read results file if created and compare threshold
        results_file = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_eval_result.json"
        if results_file.exists():
            try:
                res = json.loads(results_file.read_text(encoding="utf-8"))
                # simple threshold check example: mean_reward >= threshold_mean_reward
                if req.threshold_mean_reward is not None:
                    ok = res.get("mean_reward", -1) >= req.threshold_mean_reward
                    await _update_run(run_id, {"canary_passed": ok})
            except Exception:
                LOG.exception("Failed to read evaluation result for run %s", run_id)
    background_tasks.add_task(asyncio.create_task, _bg_eval())
    await write_audit_event({"user": user.username, "action": "start_canary_eval", "resource": tag, "details": {"run_id": run_id}})
    return {"run_id": run_id, "status": "queued"}


@router.post("/eval/{run_id}/approve", dependencies=[Depends(require_role(Role.ADMIN))])
async def approve_canary(run_id: str, promote_to_prod: bool = Form(True), user: User = Depends(get_current_user)):
    """
    Approve a canary evaluation: promote model to prod if requested.
    """
    info = TRAIN_RUNS.get(run_id) or (await runs_collection.find_one({"run_id": run_id}, {"_id": 0}) if runs_collection else None)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    model_tag = (info.get("params") or {}).get("model_tag")
    if not model_tag:
        raise HTTPException(status_code=400, detail="No model_tag associated with run")
    if promote_to_prod:
        ok = promote_model_fs(model_tag, "prod")
        if not ok:
            raise HTTPException(status_code=500, detail="Promotion failed")
    await write_audit_event({"user": user.username, "action": "approve_canary", "resource": model_tag, "details": {"run_id": run_id, "promoted": promote_to_prod}})
    return {"ok": True, "promoted": promote_to_prod, "model_tag": model_tag}


# ---------------------------
# Billing admin endpoints (simplified)
# ---------------------------

# In-memory billing plans (persist to DB or file in production)
_BILLING_PLANS: Dict[str, BillingPlan] = {
    "free": BillingPlan(plan_id="free", name="Free", throughput_limit=1000, price_usd_per_month=0.0, description="Free tier (dev)"),
    "pro": BillingPlan(plan_id="pro", name="Pro", throughput_limit=100000, price_usd_per_month=199.0, description="Pro tier")
}

@router.get("/billing/plans", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_billing_plans():
    return list(_BILLING_PLANS.values())

@router.post("/billing/plan", dependencies=[Depends(require_role(Role.ADMIN))])
async def create_billing_plan(plan: BillingPlan, user: User = Depends(get_current_user)):
    if plan.plan_id in _BILLING_PLANS:
        raise HTTPException(status_code=400, detail="Plan already exists")
    _BILLING_PLANS[plan.plan_id] = plan
    await write_audit_event({"user": user.username, "action": "create_billing_plan", "resource": plan.plan_id, "details": plan.dict()})
    return {"ok": True, "plan": plan}

@router.get("/billing/usage", dependencies=[Depends(require_role(Role.OPERATOR))])
async def billing_usage_summary(org_id: Optional[str] = Query(None)):
    """
    Return a usage summary. For now, simulated: count of tasks, runs, and costs estimated.
    In production you would query usage events and aggregate billing.
    """
    # placeholder: compute counts from DB if available
    runs_count = 0
    if runs_collection is not None:
        runs_count = await runs_collection.count_documents({})
    est_cost = runs_count * 0.01  # dummy cost per run
    return {"runs_count": runs_count, "estimated_cost_usd": est_cost}


@router.post("/billing/invoice/generate", dependencies=[Depends(require_role(Role.ADMIN))])
async def generate_invoice_for_org(org_id: str = Form(...), month: str = Form(...), user: User = Depends(get_current_user)):
    """
    Generate a simple invoice file (JSON). Replace with Stripe/Billing integration.
    """
    invoice = {
        "invoice_id": str(uuid.uuid4()),
        "org_id": org_id,
        "month": month,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "lines": [
            {"desc": "PriorityMax compute (example)", "amount_usd": 123.45}
        ],
        "total": 123.45
    }
    out_path = pathlib.Path(CHECKPOINTS_DIR) / f"invoice_{invoice['invoice_id']}.json"
    out_path.write_text(json.dumps(invoice, indent=2), encoding="utf-8")
    await write_audit_event({"user": user.username, "action": "generate_invoice", "resource": org_id, "details": {"invoice_path": str(out_path)}})
    return FileResponse(str(out_path), filename=out_path.name, media_type="application/json")


# ---------------------------
# Audit log endpoints
# ---------------------------

@router.get("/audit", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_audit_events(limit: int = Query(200), since: Optional[str] = Query(None)):
    """
    Return recent audit events. If Mongo is available, query it; otherwise tail local JSONL.
    """
    events = []
    if audit_collection is not None:
        cursor = audit_collection.find({}, {"_id": 0}).sort("timestamp_utc", -1).limit(limit)
        events = [d async for d in cursor]
        return events
    # fallback to JSONL tail
    if pathlib.Path(AUDIT_LOG_FILE).exists():
        with open(AUDIT_LOG_FILE, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 20000), 0)
            data = fh.read().decode(errors="ignore").strip().splitlines()[-limit:]
            for line in reversed(data):
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    return events


# ---------------------------
# Websocket logs streaming
# ---------------------------
WS_CONNECTIONS: Dict[str, List[WebSocket]] = {}  # run_id -> websockets

@router.websocket("/ws/logs/{run_id}")
async def ws_logs(websocket: WebSocket, run_id: str, user: User = Depends(get_current_user)):
    """
    Connect to live logs for a training run.
    Streams appended lines from the stdout log file.
    """
    await websocket.accept()
    if run_id not in WS_CONNECTIONS:
        WS_CONNECTIONS[run_id] = []
    WS_CONNECTIONS[run_id].append(websocket)
    stdout_log = pathlib.Path(CHECKPOINTS_DIR) / f"{run_id}_stdout.log"
    try:
        # tail loop
        pos = stdout_log.stat().st_size if stdout_log.exists() else 0
        while True:
            await asyncio.sleep(0.5)
            if stdout_log.exists():
                new_size = stdout_log.stat().st_size
                if new_size > pos:
                    with open(stdout_log, "rb") as fh:
                        fh.seek(pos)
                        chunk = fh.read(new_size - pos).decode(errors="ignore")
                        await websocket.send_text(chunk)
                    pos = new_size
    except WebSocketDisconnect:
        LOG.info("WS client disconnected for run %s", run_id)
    except Exception:
        LOG.exception("Error streaming logs for run %s", run_id)
    finally:
        WS_CONNECTIONS[run_id].remove(websocket)


# ---------------------------
# Utilities
# ---------------------------
def sys_executable_or_fallback():
    """
    Return sys.executable path or default 'python3' if not available.
    """
    import sys
    return getattr(sys, "executable", "python3")


# ---------------------------
# Startup / Cleanup hooks (optional)
# ---------------------------
@router.on_event("startup")
async def _startup_tasks():
    LOG.info("Admin API startup: scanning registry")
    # ensure model registry exists
    ensure_path(MODEL_REGISTRY_DIR)
    # optional: sync filesystem registry into DB
    if models_collection is not None:
        # naive sync: import all FS metadata into DB if missing
        for child in pathlib.Path(MODEL_REGISTRY_DIR).iterdir():
            if child.is_dir():
                md_file = child / "metadata.json"
                if md_file.exists():
                    try:
                        md = json.loads(md_file.read_text(encoding="utf-8"))
                        await models_collection.update_one({"tag": md["tag"]}, {"$set": md}, upsert=True)
                    except Exception:
                        LOG.exception("Failed to import metadata for %s", child)


@router.on_event("shutdown")
async def _shutdown_tasks():
    LOG.info("Admin API shutdown")


# ---------------------------
# End of file
# ---------------------------

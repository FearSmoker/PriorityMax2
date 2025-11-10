# backend/app/api/workflows.py
"""
PriorityMax Workflows API — Chunk 1/7
Initialization, logging, DB/Redis setup, Pydantic models & enums, FS fallback persistence helpers.

Assemble file by pasting chunks 1 → 7 in order.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import math
import asyncio
import logging
import pathlib
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Body, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator, root_validator

# Optional libs
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

try:
    import aioredis
    _HAS_AIOREDIS = True
except Exception:
    aioredis = None
    _HAS_AIOREDIS = False

try:
    from opentelemetry import trace as otel_trace
    _HAS_OTEL = True
except Exception:
    _HAS_OTEL = False

# Admin & audit helpers
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    def get_current_user(*a, **k):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
    def require_role(r):
        def _dep(*a, **k):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
        return _dep
    class Role:
        ADMIN = "admin"
        OPERATOR = "operator"
        VIEWER = "viewer"
    async def write_audit_event(evt: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "workflows_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.workflows")
LOG.setLevel(os.getenv("PRIORITYMAX_WORKFLOWS_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_handler)

# Base dirs & meta
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
WORKFLOWS_META_DIR = pathlib.Path(os.getenv("WORKFLOWS_META_DIR", str(BASE_DIR / "app" / "workflows_meta")))
WORKFLOWS_META_DIR.mkdir(parents=True, exist_ok=True)

_WORKFLOWS_FS = WORKFLOWS_META_DIR / "workflows.json"      # map workflow_id -> metadata (FS fallback)
_WORKFLOW_VERSIONS_DIR = WORKFLOWS_META_DIR / "versions"  # per-workflow versions (json files)
_RUNS_DIR = WORKFLOWS_META_DIR / "runs"                   # execution run logs
_WORKFLOWS_FS.parent.mkdir(parents=True, exist_ok=True)
_WORKFLOW_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
_RUNS_DIR.mkdir(parents=True, exist_ok=True)

if not _WORKFLOWS_FS.exists():
    _WORKFLOWS_FS.write_text(json.dumps({}), encoding="utf-8")

# Environment config
MONGO_URL = os.getenv("MONGO_URL", None)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WORKFLOW_RETENTION_DAYS = int(os.getenv("WORKFLOW_RETENTION_DAYS", "30"))
WORKFLOW_MAX_CONCURRENT_RUNS = int(os.getenv("WORKFLOW_MAX_CONCURRENT_RUNS", "10"))
WORKFLOW_EXECUTOR_DEFAULT = os.getenv("WORKFLOW_EXECUTOR_DEFAULT", "python")  # default executor type
S3_BUCKET = os.getenv("S3_BUCKET", None)

# DB clients (lazy init)
_mongo_client = None
_workflow_meta_col = None

if _HAS_MOTOR and MONGO_URL:
    try:
        _mongo_client = motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        _workflow_db = _mongo_client.get_default_database()
        _workflow_meta_col = _workflow_db.get_collection(os.getenv("WORKFLOWS_METADATA_COLLECTION", "prioritymax_workflows"))
        LOG.info("Workflows metadata: using MongoDB at %s", MONGO_URL)
    except Exception:
        _workflow_meta_col = None
        LOG.exception("Failed to connect to Mongo; using filesystem fallback for workflows metadata")
else:
    _workflow_meta_col = None
    LOG.info("Workflows metadata: using filesystem fallback at %s", _WORKFLOWS_FS)

_redis_client: Optional[Any] = None

async def get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not _HAS_AIOREDIS:
        LOG.info("aioredis not installed — redis features disabled for workflows")
        return None
    try:
        _redis_client = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        await _redis_client.ping()
        LOG.info("Workflows redis connected at %s", REDIS_URL)
        return _redis_client
    except Exception:
        LOG.exception("Failed to connect to Redis for workflows")
        _redis_client = None
        return None

# -------------------------
# Enums & Models
# -------------------------
class NodeType(str, Enum):
    TASK = "task"              # single task node
    PARALLEL = "parallel"      # parallel fork
    JOIN = "join"              # join node
    TIMER = "timer"            # delay/ETA node
    SWITCH = "switch"          # conditional routing
    SUBWORKFLOW = "subworkflow"# call another workflow
    START = "start"
    END = "end"

class ExecutorType(str, Enum):
    PYTHON = "python"          # inline python function
    CONTAINER = "container"    # run in container (k8s job)
    HTTP = "http"              # call REST endpoint
    EXTERNAL = "external"      # external service/webhook

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class TaskTemplate(BaseModel):
    """
    Template for a task node: contains executor type and executor-specific config.
    """
    executor: ExecutorType = Field(..., description="Executor type to run the task")
    config: Dict[str, Any] = Field(default_factory=dict, description="Executor-specific config (image, command, url, function path, etc.)")
    timeout_seconds: Optional[int] = Field(300, ge=1)
    retry_policy: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"max_retries": 3, "backoff_seconds": 5})
    sla_seconds: Optional[int] = Field(None, description="SLA time for this task")

class WorkflowNode(BaseModel):
    node_id: str
    type: NodeType
    name: str
    template: Optional[TaskTemplate] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # optional conditional expression for SWITCH nodes
    condition: Optional[str] = None
    # parallelism hint
    concurrency: Optional[int] = 1

    @validator("node_id", pre=True, always=True)
    def ensure_node_id(cls, v):
        return v or f"node_{uuid.uuid4().hex[:8]}"

class WorkflowEdge(BaseModel):
    source: str
    target: str
    condition: Optional[str] = None  # optional expression for conditional edges

class WorkflowSpec(BaseModel):
    """
    The core workflow spec contains nodes and edges forming a DAG.
    """
    name: str
    description: Optional[str] = None
    nodes: List[WorkflowNode] = Field(default_factory=list)
    edges: List[WorkflowEdge] = Field(default_factory=list)
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict)  # default variables for runs
    version: int = 1

    @root_validator
    def validate_nodes_edges(cls, values):
        nodes = values.get("nodes") or []
        edges = values.get("edges") or []
        node_ids = {n.node_id for n in nodes}
        for e in edges:
            if e.source not in node_ids:
                raise ValueError(f"Edge source {e.source} not in nodes")
            if e.target not in node_ids:
                raise ValueError(f"Edge target {e.target} not in nodes")
        return values

class WorkflowMeta(BaseModel):
    workflow_id: str
    spec: WorkflowSpec
    owner: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_at: str
    updated_at: str
    version_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    @validator("workflow_id", pre=True, always=True)
    def ensure_wid(cls, v):
        return v or f"wf_{uuid.uuid4().hex[:10]}"

class WorkflowRun(BaseModel):
    run_id: str
    workflow_id: str
    spec_version: int
    input_variables: Dict[str, Any]
    status: RunStatus = RunStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    logs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    current_nodes: Optional[List[str]] = Field(default_factory=list)
    attempts: int = 0

    @validator("run_id", pre=True, always=True)
    def mk_run_id(cls, v):
        return v or f"run_{uuid.uuid4().hex[:12]}"

# -------------------------
# Persistence helpers (FS fallback + Mongo if available)
# -------------------------
def _fs_read_workflows() -> Dict[str, Dict[str, Any]]:
    try:
        raw = _WORKFLOWS_FS.read_text(encoding="utf-8")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        LOG.exception("Failed to read workflows FS store")
        return {}

def _fs_write_workflows(data: Dict[str, Dict[str, Any]]):
    try:
        tmp = _WORKFLOWS_FS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        tmp.replace(_WORKFLOWS_FS)
    except Exception:
        LOG.exception("Failed to write workflows FS store")

async def _persist_workflow_meta(meta: WorkflowMeta):
    """
    Persist WorkflowMeta to Mongo if available; otherwise FS.
    Also save a versioned snapshot into versions directory.
    """
    data = meta.dict()
    if _workflow_meta_col is not None:
        try:
            await _workflow_meta_col.update_one({"workflow_id": meta.workflow_id}, {"$set": data}, upsert=True)
            return
        except Exception:
            LOG.exception("Mongo write failed; falling back to FS for workflow meta")
    # FS fallback
    all_wfs = _fs_read_workflows()
    all_wfs[meta.workflow_id] = data
    _fs_write_workflows(all_wfs)
    # version snapshot
    ver_file = _WORKFLOW_VERSIONS_DIR / f"{meta.workflow_id}_v{meta.spec.version}.json"
    try:
        ver_file.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    except Exception:
        LOG.exception("Failed to write workflow version snapshot")

async def _load_workflow_meta(workflow_id: str) -> Optional[WorkflowMeta]:
    if _workflow_meta_col is not None:
        try:
            doc = await _workflow_meta_col.find_one({"workflow_id": workflow_id})
            if not doc:
                return None
            doc.pop("_id", None)
            return WorkflowMeta(**doc)
        except Exception:
            LOG.exception("Mongo read failed; falling back to FS for workflow meta")
    all_wfs = _fs_read_workflows()
    doc = all_wfs.get(workflow_id)
    if not doc:
        return None
    try:
        return WorkflowMeta(**doc)
    except Exception:
        LOG.exception("Failed to parse workflow meta from FS for %s", workflow_id)
        return None

async def _delete_workflow_meta(workflow_id: str):
    if _workflow_meta_col is not None:
        try:
            await _workflow_meta_col.delete_one({"workflow_id": workflow_id})
            return
        except Exception:
            LOG.exception("Mongo delete failed; falling back to FS")
    all_wfs = _fs_read_workflows()
    if workflow_id in all_wfs:
        del all_wfs[workflow_id]
        _fs_write_workflows(all_wfs)
    # remove versions
    for f in _WORKFLOW_VERSIONS_DIR.glob(f"{workflow_id}_v*.json"):
        try:
            f.unlink()
        except Exception:
            pass

# -------------------------
# Run persistence (fs-based by default)
# -------------------------
def _run_file_for(run_id: str) -> pathlib.Path:
    return _RUNS_DIR / f"{run_id}.jsonl"

async def _append_run_log(run_id: str, entry: Dict[str, Any]):
    p = _run_file_for(run_id)
    try:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        LOG.exception("Failed to append run log for %s", run_id)

async def _persist_run_record(run: WorkflowRun):
    """
    Save a run summary record to workflow DB or FS.
    """
    data = run.dict()
    # try mongo runs collection if available
    if _workflow_meta_col is not None:
        try:
            runs_col = _workflow_meta_col.database.get_collection(os.getenv("WORKFLOW_RUNS_COLLECTION", "prioritymax_workflow_runs"))
            await runs_col.update_one({"run_id": run.run_id}, {"$set": data}, upsert=True)
            return
        except Exception:
            LOG.exception("Mongo write failed for run record; falling back to FS")
    # FS fallback: write a compact json file per run
    p = _RUNS_DIR / f"{run.run_id}.json"
    try:
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    except Exception:
        LOG.exception("Failed to persist run record to FS for %s", run.run_id)

async def _load_run(run_id: str) -> Optional[WorkflowRun]:
    if _workflow_meta_col is not None:
        try:
            runs_col = _workflow_meta_col.database.get_collection(os.getenv("WORKFLOW_RUNS_COLLECTION", "prioritymax_workflow_runs"))
            doc = await runs_col.find_one({"run_id": run_id})
            if not doc:
                return None
            doc.pop("_id", None)
            return WorkflowRun(**doc)
        except Exception:
            LOG.exception("Mongo read failed for run; falling back to FS")
    p = _run_file_for(run_id)
    if not p.exists():
        # maybe compact summary exists
        summary = _RUNS_DIR / f"{run_id}.json"
        if summary.exists():
            try:
                return WorkflowRun(**json.loads(summary.read_text(encoding="utf-8")))
            except Exception:
                LOG.exception("Failed to parse run summary for %s", run_id)
                return None
        return None
    # read logs and reconstruct basic run
    try:
        logs = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    continue
        # minimal run reconstruction (best-effort)
        # find header entry with run metadata if exists
        header = {}
        for e in logs:
            if e.get("_meta"):
                header = e.get("_meta")
                break
        run = WorkflowRun(
            run_id = run_id,
            workflow_id = header.get("workflow_id", "unknown"),
            spec_version = header.get("spec_version", 1),
            input_variables = header.get("input_variables", {}),
            status = header.get("status", RunStatus.PENDING),
            started_at = header.get("started_at"),
            completed_at = header.get("completed_at"),
            result = header.get("result"),
            logs = logs,
            current_nodes = header.get("current_nodes", []),
            attempts = header.get("attempts", 0)
        )
        return run
    except Exception:
        LOG.exception("Failed to reconstruct run from logs for %s", run_id)
        return None

# -------------------------
# Router
# -------------------------
router = APIRouter(prefix="/workflows", tags=["workflows"])

# Placeholder for websocket connections (implemented later)
_WORKFLOWS_WS: List[Any] = []

async def _broadcast_workflow_event(payload: Dict[str, Any]):
    """
    Placeholder broadcast function; implemented in a later chunk with WebSocket logic.
    """
    try:
        await write_audit_event({"source": "workflow_event", "payload": payload, "ts": datetime.datetime.utcnow().isoformat() + "Z"})
    except Exception:
        LOG.exception("Failed to write audit for workflow event")
    return

# End of Chunk 1/7
# -------------------------
# Chunk 2/7 — Workflow CRUD, Validation, Versioning, Import/Export
# -------------------------

from fastapi import Path, UploadFile, File
from networkx import DiGraph, is_directed_acyclic_graph  # networkx is optional; fallback implemented
try:
    import networkx as nx
    _HAS_NETWORKX = True
except Exception:
    nx = None
    _HAS_NETWORKX = False

# Utility: build a directed graph from spec and check for cycles
def _build_graph_from_spec(spec: WorkflowSpec) -> Any:
    """
    Returns a directed graph object compatible with networkx if available,
    otherwise returns a lightweight adjacency dict.
    """
    if _HAS_NETWORKX:
        G = nx.DiGraph()
        for n in spec.nodes:
            G.add_node(n.node_id)
        for e in spec.edges:
            G.add_edge(e.source, e.target)
        return G
    # fallback adjacency dict
    adj = {n.node_id: [] for n in spec.nodes}
    for e in spec.edges:
        adj.setdefault(e.source, []).append(e.target)
    return adj

def _is_acyclic(spec: WorkflowSpec) -> bool:
    """
    Returns True if DAG is acyclic; uses networkx if available, otherwise does DFS detection.
    """
    try:
        if _HAS_NETWORKX:
            G = _build_graph_from_spec(spec)
            return nx.is_directed_acyclic_graph(G)
        # fallback DFS
        adj = _build_graph_from_spec(spec)
        visited = set()
        recstack = set()

        def dfs(u):
            visited.add(u)
            recstack.add(u)
            for v in adj.get(u, []):
                if v not in visited:
                    if dfs(v):
                        return True
                elif v in recstack:
                    return True
            recstack.remove(u)
            return False

        for node in adj.keys():
            if node not in visited:
                if dfs(node):
                    return False
        return True
    except Exception:
        LOG.exception("Failed to determine acyclicity for workflow spec")
        return False

# -------------------------
# Create workflow
# -------------------------
@router.post("/", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_workflow(spec: WorkflowSpec = Body(...), name: Optional[str] = Query(None), user = Depends(get_current_user)):
    """
    Create a new workflow from spec. Validates DAG acyclicity and persists a versioned snapshot.
    Returns created WorkflowMeta.
    """
    # validate acyclic
    if not _is_acyclic(spec):
        raise HTTPException(status_code=400, detail="Workflow graph is cyclic — must be a DAG")

    now = datetime.datetime.utcnow().isoformat() + "Z"
    wf_meta = WorkflowMeta(
        workflow_id = f"wf_{uuid.uuid4().hex[:10]}",
        spec = spec,
        owner = getattr(user, "username", None),
        status = WorkflowStatus.DRAFT,
        created_at = now,
        updated_at = now,
        version_history = []
    )
    # persist
    await _persist_workflow_meta(wf_meta)
    await write_audit_event({"event": "create_workflow", "workflow_id": wf_meta.workflow_id, "user": getattr(user, "username", "unknown"), "ts": now})
    await _broadcast_workflow_event({"event": "workflow_created", "workflow_id": wf_meta.workflow_id, "owner": wf_meta.owner, "ts": now})
    return wf_meta

# -------------------------
# List workflows
# -------------------------
@router.get("/", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_workflows():
    """
    Return available workflows (basic metadata map).
    """
    results = []
    if _workflow_meta_col is not None:
        try:
            docs = await _workflow_meta_col.find({}).to_list(length=1000)
            for d in docs:
                d.pop("_id", None)
                results.append({"workflow_id": d.get("workflow_id"), "name": d.get("spec", {}).get("name"), "status": d.get("status"), "updated_at": d.get("updated_at")})
            return results
        except Exception:
            LOG.exception("Mongo list workflows failed; falling back to FS")
    # FS fallback
    all_wfs = _fs_read_workflows()
    for wid, doc in all_wfs.items():
        results.append({"workflow_id": wid, "name": doc.get("spec", {}).get("name"), "status": doc.get("status"), "updated_at": doc.get("updated_at")})
    return results

# -------------------------
# Get workflow by id
# -------------------------
@router.get("/{workflow_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_workflow(workflow_id: str = Path(...)):
    meta = await _load_workflow_meta(workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return meta

# -------------------------
# Update workflow (creates new version)
# -------------------------
@router.put("/{workflow_id}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def update_workflow(workflow_id: str, spec: WorkflowSpec = Body(...), user = Depends(get_current_user)):
    """
    Update a workflow's spec — stores a new version snapshot, increments version,
    and appends to version_history.
    """
    meta = await _load_workflow_meta(workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # validate DAG
    if not _is_acyclic(spec):
        raise HTTPException(status_code=400, detail="Workflow graph is cyclic — must be a DAG")

    now = datetime.datetime.utcnow().isoformat() + "Z"
    new_version = (meta.spec.version or 1) + 1
    # snapshot previous version
    hist_entry = {"version": meta.spec.version, "spec": meta.spec.dict(), "updated_at": meta.updated_at}
    meta.version_history = meta.version_history or []
    meta.version_history.append(hist_entry)
    # apply update
    meta.spec = spec
    meta.spec.version = new_version
    meta.updated_at = now
    # if name changed, update
    await _persist_workflow_meta(meta)
    await write_audit_event({"event": "update_workflow", "workflow_id": workflow_id, "user": getattr(user, "username", "unknown"), "new_version": new_version, "ts": now})
    await _broadcast_workflow_event({"event": "workflow_updated", "workflow_id": workflow_id, "new_version": new_version, "ts": now})
    return meta

# -------------------------
# Delete workflow
# -------------------------
@router.delete("/{workflow_id}", dependencies=[Depends(require_role(Role.ADMIN))])
async def delete_workflow(workflow_id: str, user = Depends(get_current_user)):
    meta = await _load_workflow_meta(workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # delete persisted metadata & versions
    await _delete_workflow_meta(workflow_id)
    now = datetime.datetime.utcnow().isoformat() + "Z"
    await write_audit_event({"event": "delete_workflow", "workflow_id": workflow_id, "user": getattr(user, "username", "unknown"), "ts": now})
    await _broadcast_workflow_event({"event": "workflow_deleted", "workflow_id": workflow_id, "ts": now})
    return {"ok": True, "deleted": workflow_id}

# -------------------------
# Export (download) workflow spec (single version or current)
# -------------------------
@router.get("/{workflow_id}/export", dependencies=[Depends(require_role(Role.VIEWER))])
async def export_workflow(workflow_id: str, version: Optional[int] = Query(None)):
    meta = await _load_workflow_meta(workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if version is None or version == meta.spec.version:
        payload = meta.spec.dict()
    else:
        # find version in history
        found = None
        for v in meta.version_history or []:
            if v.get("version") == version:
                found = v.get("spec")
                break
        if not found:
            raise HTTPException(status_code=404, detail="Version not found")
        payload = found
    filename = f"{workflow_id}_v{version or meta.spec.version}.json"
    temp = WORKFLOWS_META_DIR / filename
    temp.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    return FileResponse(str(temp), filename=filename, media_type="application/json")

# -------------------------
# Import workflow (upload JSON)
# -------------------------
@router.post("/import", dependencies=[Depends(require_role(Role.OPERATOR))])
async def import_workflow(file: UploadFile = File(...), user = Depends(get_current_user)):
    """
    Import a workflow JSON file (spec only). Creates a new workflow entry.
    """
    try:
        raw = await file.read()
        payload = json.loads(raw)
        spec = WorkflowSpec(**payload)
    except Exception:
        LOG.exception("Failed to parse uploaded workflow spec")
        raise HTTPException(status_code=400, detail="Invalid workflow spec")

    if not _is_acyclic(spec):
        raise HTTPException(status_code=400, detail="Workflow graph is cyclic — must be a DAG")

    now = datetime.datetime.utcnow().isoformat() + "Z"
    wf_meta = WorkflowMeta(
        workflow_id = f"wf_{uuid.uuid4().hex[:10]}",
        spec = spec,
        owner = getattr(user, "username", None),
        status = WorkflowStatus.DRAFT,
        created_at = now,
        updated_at = now,
        version_history = []
    )
    await _persist_workflow_meta(wf_meta)
    await write_audit_event({"event": "import_workflow", "workflow_id": wf_meta.workflow_id, "user": getattr(user, "username", "unknown"), "ts": now})
    await _broadcast_workflow_event({"event": "workflow_imported", "workflow_id": wf_meta.workflow_id, "ts": now})
    return {"ok": True, "workflow_id": wf_meta.workflow_id}

# -------------------------
# Quick validation endpoint (used by Designer)
# -------------------------
@router.post("/validate", dependencies=[Depends(require_role(Role.OPERATOR))])
async def validate_workflow_spec(spec: WorkflowSpec = Body(...)):
    """
    Validate a workflow spec for structural correctness:
      - nodes & edges reference correctness
      - DAG acyclicity
      - node templates presence for TASK nodes
    """
    # node/edge cross-check covered by pydantic root_validator
    # ensure TASK nodes have templates
    for n in spec.nodes:
        if n.type == NodeType.TASK and not n.template:
            raise HTTPException(status_code=400, detail=f"Task node {n.node_id} missing template")
    # acyclicity
    if not _is_acyclic(spec):
        raise HTTPException(status_code=400, detail="Workflow graph is cyclic")
    # basic template sanity
    for n in spec.nodes:
        if n.template:
            if n.template.executor == ExecutorType.PYTHON:
                # require 'callable' in config
                if "callable" not in n.template.config:
                    raise HTTPException(status_code=400, detail=f"Python executor node {n.node_id} missing 'callable' config")
            if n.template.executor == ExecutorType.CONTAINER:
                if "image" not in n.template.config:
                    raise HTTPException(status_code=400, detail=f"Container executor node {n.node_id} missing 'image' config")
    return {"ok": True, "message": "spec_valid"}

# -------------------------
# List versions for a workflow
# -------------------------
@router.get("/{workflow_id}/versions", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_workflow_versions(workflow_id: str):
    meta = await _load_workflow_meta(workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    versions = [{"version": meta.spec.version, "updated_at": meta.updated_at}]
    for v in meta.version_history or []:
        versions.append({"version": v.get("version"), "updated_at": v.get("updated_at")})
    return {"workflow_id": workflow_id, "versions": versions}

# -------------------------
# Rollback to a prior version
# -------------------------
@router.post("/{workflow_id}/rollback", dependencies=[Depends(require_role(Role.OPERATOR))])
async def rollback_workflow(workflow_id: str, version: int = Body(...), user = Depends(get_current_user)):
    meta = await _load_workflow_meta(workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if version == meta.spec.version:
        return {"ok": True, "message": "Already on requested version"}
    found_spec = None
    # locate version in history
    for v in meta.version_history or []:
        if v.get("version") == version:
            found_spec = v.get("spec")
            break
    if found_spec is None:
        raise HTTPException(status_code=404, detail="Version not found in history")
    # apply rollback: snapshot current to history, set spec to found, bump version
    now = datetime.datetime.utcnow().isoformat() + "Z"
    hist_entry = {"version": meta.spec.version, "spec": meta.spec.dict(), "updated_at": meta.updated_at}
    meta.version_history.append(hist_entry)
    meta.spec = WorkflowSpec(**found_spec)
    meta.spec.version = (meta.spec.version or 1) + 1
    meta.updated_at = now
    await _persist_workflow_meta(meta)
    await write_audit_event({"event": "rollback_workflow", "workflow_id": workflow_id, "to_version": version, "user": getattr(user, "username", "unknown"), "ts": now})
    await _broadcast_workflow_event({"event": "workflow_rolled_back", "workflow_id": workflow_id, "to_version": version, "ts": now})
    return {"ok": True, "workflow_id": workflow_id, "new_version": meta.spec.version}

# End of Chunk 2/7
# -------------------------
# Chunk 3/7 — Workflow Designer APIs, Template Registry, Validation & Preview
# -------------------------

_TEMPLATE_REGISTRY_FILE = WORKFLOWS_META_DIR / "templates.json"
if not _TEMPLATE_REGISTRY_FILE.exists():
    _TEMPLATE_REGISTRY_FILE.write_text(json.dumps({}, indent=2), encoding="utf-8")

def _fs_read_templates() -> Dict[str, Dict[str, Any]]:
    try:
        raw = _TEMPLATE_REGISTRY_FILE.read_text(encoding="utf-8")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        LOG.exception("Failed to read template registry")
        return {}

def _fs_write_templates(data: Dict[str, Dict[str, Any]]):
    try:
        tmp = _TEMPLATE_REGISTRY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(_TEMPLATE_REGISTRY_FILE)
    except Exception:
        LOG.exception("Failed to write template registry")

# -------------------------
# Template CRUD Endpoints
# -------------------------
class TemplateCreate(BaseModel):
    name: str
    executor: ExecutorType
    config: Dict[str, Any]
    description: Optional[str] = None

@router.post("/templates", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_template(payload: TemplateCreate, user=Depends(get_current_user)):
    templates = _fs_read_templates()
    template_id = f"tpl_{uuid.uuid4().hex[:8]}"
    templates[template_id] = {
        "template_id": template_id,
        "name": payload.name,
        "executor": payload.executor.value,
        "config": payload.config,
        "description": payload.description,
        "created_by": getattr(user, "username", "unknown"),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    _fs_write_templates(templates)
    await write_audit_event({
        "event": "create_template",
        "template_id": template_id,
        "executor": payload.executor.value,
        "user": getattr(user, "username", "unknown"),
        "ts": datetime.datetime.utcnow().isoformat() + "Z"
    })
    await _broadcast_workflow_event({"event": "template_created", "template_id": template_id})
    return {"ok": True, "template_id": template_id}

@router.get("/templates", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_templates():
    return list(_fs_read_templates().values())

@router.delete("/templates/{template_id}", dependencies=[Depends(require_role(Role.ADMIN))])
async def delete_template(template_id: str, user=Depends(get_current_user)):
    templates = _fs_read_templates()
    if template_id not in templates:
        raise HTTPException(status_code=404, detail="Template not found")
    templates.pop(template_id)
    _fs_write_templates(templates)
    await write_audit_event({
        "event": "delete_template",
        "template_id": template_id,
        "user": getattr(user, "username", "unknown"),
        "ts": datetime.datetime.utcnow().isoformat() + "Z"
    })
    await _broadcast_workflow_event({"event": "template_deleted", "template_id": template_id})
    return {"ok": True, "deleted": template_id}

# -------------------------
# Graph preview & validation for Designer
# -------------------------
@router.post("/preview", dependencies=[Depends(require_role(Role.OPERATOR))])
async def preview_workflow_graph(spec: WorkflowSpec = Body(...)):
    """
    Return lightweight structure summary for rendering in Designer frontend.
    """
    nodes_summary = [{"id": n.node_id, "name": n.name, "type": n.type.value} for n in spec.nodes]
    edges_summary = [{"from": e.source, "to": e.target, "condition": e.condition} for e in spec.edges]
    acyclic = _is_acyclic(spec)
    start_nodes = [n.node_id for n in spec.nodes if n.type == NodeType.START]
    end_nodes = [n.node_id for n in spec.nodes if n.type == NodeType.END]
    return {
        "nodes": nodes_summary,
        "edges": edges_summary,
        "is_acyclic": acyclic,
        "start_nodes": start_nodes,
        "end_nodes": end_nodes,
        "count_nodes": len(nodes_summary)
    }

@router.post("/validate/connectors", dependencies=[Depends(require_role(Role.OPERATOR))])
async def validate_connectors(spec: WorkflowSpec = Body(...)):
    """
    Validate each connector (edge) for basic correctness and reachability.
    """
    node_ids = {n.node_id for n in spec.nodes}
    invalid_edges = []
    for e in spec.edges:
        if e.source not in node_ids or e.target not in node_ids:
            invalid_edges.append(e)
    acyclic = _is_acyclic(spec)
    result = {"invalid_edges": len(invalid_edges), "is_acyclic": acyclic}
    return result

# -------------------------
# Thumbnail metadata for Designer Dashboard
# -------------------------
def _generate_thumbnail_meta(spec: WorkflowSpec) -> Dict[str, Any]:
    """
    Generate a minimal thumbnail JSON for UI display.
    """
    return {
        "name": spec.name,
        "nodes": len(spec.nodes),
        "edges": len(spec.edges),
        "start_nodes": len([n for n in spec.nodes if n.type == NodeType.START]),
        "end_nodes": len([n for n in spec.nodes if n.type == NodeType.END]),
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z"
    }

@router.post("/{workflow_id}/thumbnail", dependencies=[Depends(require_role(Role.OPERATOR))])
async def regenerate_thumbnail(workflow_id: str, user=Depends(get_current_user)):
    meta = await _load_workflow_meta(workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    thumb = _generate_thumbnail_meta(meta.spec)
    thumb_file = WORKFLOWS_META_DIR / f"{workflow_id}_thumbnail.json"
    thumb_file.write_text(json.dumps(thumb, indent=2, default=str), encoding="utf-8")
    await write_audit_event({
        "event": "regenerate_thumbnail",
        "workflow_id": workflow_id,
        "user": getattr(user, "username", "unknown"),
        "ts": datetime.datetime.utcnow().isoformat() + "Z"
    })
    await _broadcast_workflow_event({"event": "thumbnail_regenerated", "workflow_id": workflow_id})
    return {"ok": True, "thumbnail": thumb}

@router.get("/{workflow_id}/thumbnail", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_thumbnail(workflow_id: str):
    f = WORKFLOWS_META_DIR / f"{workflow_id}_thumbnail.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return JSONResponse(content=json.loads(f.read_text(encoding="utf-8")))

# -------------------------
# Workflow search endpoint (for Designer UI)
# -------------------------
@router.get("/search", dependencies=[Depends(require_role(Role.VIEWER))])
async def search_workflows(q: str = Query("")):
    """
    Simple full-text search by workflow name or description.
    """
    all_wfs = _fs_read_workflows()
    res = []
    for wf_id, wf in all_wfs.items():
        spec = wf.get("spec", {})
        name = (spec.get("name") or "").lower()
        desc = (spec.get("description") or "").lower()
        if q.lower() in name or q.lower() in desc:
            res.append({"workflow_id": wf_id, "name": spec.get("name"), "description": spec.get("description")})
    return {"results": res, "count": len(res)}

# End of Chunk 3/7
# -------------------------
# Chunk 4/7 — Execution Engine Core (start/stop runs, DAG executor, dependency manager)
# -------------------------

from concurrent.futures import ThreadPoolExecutor

# Executor registry will be populated by pluggable executors (Chunk 5)
# Each executor function must be async and accept (node: WorkflowNode, run_context: Dict) -> Dict[str, Any]
EXECUTOR_REGISTRY: Dict[str, Callable[[WorkflowNode, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}

# Small threadpool for any blocking sync call fallback
_THREAD_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("WORKFLOW_THREADPOOL_SIZE", "6")))

# Active runs in-memory map: run_id -> asyncio.Task
_ACTIVE_RUNS: Dict[str, asyncio.Task] = {}
_RUN_LOCK = asyncio.Lock()

def _safe_eval_condition(expr: str, variables: Dict[str, Any]) -> bool:
    """
    Evaluate a simple condition expression for SWITCH nodes.
    This intentionally uses a very limited eval environment to avoid granting full globals.
    Supported expressions: comparisons, boolean ops, numeric math referencing variables by name.
    Example: "payload.amount > 100 and user_plan == 'pro'"
    """
    if not expr:
        return False
    try:
        allowed_builtins = {"min": min, "max": max, "int": int, "float": float, "str": str, "len": len}
        # prepare locals — flatten variables into local namespace
        local_ns = dict(variables or {})
        result = eval(expr, {"__builtins__": allowed_builtins}, local_ns)
        return bool(result)
    except Exception:
        LOG.exception("Condition eval error: %s", expr)
        return False

async def _execute_node(node: WorkflowNode, run: WorkflowRun, run_ctx: Dict[str, Any], semaphore: Optional[asyncio.Semaphore] = None):
    """
    Execute a single node using the registered executor.
    Handles timeouts, retries, logging, SLA recording, and updates run logs.
    """
    entry = {
        "ts": _now_iso(),
        "node_id": node.node_id,
        "node_name": node.name,
        "type": node.type.value,
        "action": "start"
    }
    await _append_run_log(run.run_id, {"_meta": {"workflow_id": run.workflow_id, "spec_version": run.spec_version}})
    await _append_run_log(run.run_id, entry)
    await _broadcast_workflow_event({"event": "node_start", "run_id": run.run_id, "node_id": node.node_id, "ts": _now_iso()})

    # concurrency control
    if semaphore:
        await semaphore.acquire()

    attempts = 0
    last_err = None
    success = False
    result_payload = None

    # determine retry policy
    rp = node.template.retry_policy if node.template and getattr(node.template, "retry_policy", None) else {"max_retries": 1, "backoff_seconds": 5}
    max_retries = int(rp.get("max_retries", 1))
    backoff = int(rp.get("backoff_seconds", 5))
    timeout_seconds = node.template.timeout_seconds if node.template and node.template.timeout_seconds else None

    while attempts < max_retries and not success:
        attempts += 1
        try:
            executor_fn = EXECUTOR_REGISTRY.get(node.template.executor.value if node.template else WorkflowExecutor, None)
            if not executor_fn:
                raise RuntimeError(f"No executor registered for {node.template.executor if node.template else 'unknown'}")
            # call executor — if it's blocking we run in threadpool via run_in_executor
            if asyncio.iscoroutinefunction(executor_fn):
                coro = executor_fn(node, run_ctx)
            else:
                # wrap sync function
                coro = asyncio.get_event_loop().run_in_executor(_THREAD_POOL, lambda: executor_fn(node, run_ctx))
            if timeout_seconds:
                result_payload = await asyncio.wait_for(coro, timeout=timeout_seconds)
            else:
                result_payload = await coro
            success = True
            break
        except asyncio.TimeoutError:
            last_err = f"timeout after {timeout_seconds}s"
            LOG.warning("Node %s timed out on attempt %d/%d", node.node_id, attempts, max_retries)
        except Exception as e:
            last_err = str(e)
            LOG.exception("Node %s execution error on attempt %d/%d", node.node_id, attempts, max_retries)
        if not success and attempts < max_retries:
            await asyncio.sleep(backoff * (attempts or 1))

    # release semaphore if acquired
    if semaphore:
        try:
            semaphore.release()
        except Exception:
            pass

    entry_end = {
        "ts": _now_iso(),
        "node_id": node.node_id,
        "node_name": node.name,
        "action": "end",
        "success": success,
        "attempts": attempts,
        "last_error": last_err,
        "result": result_payload
    }
    await _append_run_log(run.run_id, entry_end)
    await _broadcast_workflow_event({"event": "node_end", "run_id": run.run_id, "node_id": node.node_id, "success": success, "ts": _now_iso()})
    return {"success": success, "attempts": attempts, "error": last_err, "result": result_payload}

async def _execute_workflow_run(run: WorkflowRun):
    """
    Core DAG executor: runs tasks respecting dependencies, parallelism and special node types.
    Writes logs to run file and persists run summary once complete/failed.
    """
    LOG.info("Starting workflow run %s for workflow %s", run.run_id, run.workflow_id)
    # mark running
    run.status = RunStatus.RUNNING
    run.started_at = _now_iso()
    await _persist_run_record(run)
    await _append_run_log(run.run_id, {"_meta": {"workflow_id": run.workflow_id, "spec_version": run.spec_version, "input_variables": run.input_variables, "status": "running", "started_at": run.started_at}})
    await _broadcast_workflow_event({"event": "run_started", "run_id": run.run_id, "workflow_id": run.workflow_id, "ts": run.started_at})

    # load workflow spec
    meta = await _load_workflow_meta(run.workflow_id)
    if not meta:
        run.status = RunStatus.FAILED
        run.completed_at = _now_iso()
        run.result = {"error": "workflow_meta_missing"}
        await _persist_run_record(run)
        await _append_run_log(run.run_id, {"event": "run_failed", "reason": "workflow_meta_missing", "ts": run.completed_at})
        await _broadcast_workflow_event({"event": "run_failed", "run_id": run.run_id, "reason": "workflow_meta_missing", "ts": run.completed_at})
        return

    spec = meta.spec
    # build adjacency and indegree
    adj = {n.node_id: [] for n in spec.nodes}
    indegree = {n.node_id: 0 for n in spec.nodes}
    node_map = {n.node_id: n for n in spec.nodes}
    for e in spec.edges:
        adj[e.source].append((e.target, e.condition))
        indegree[e.target] += 1

    # mapping for concurrency semaphores per node
    semaphores: Dict[str, asyncio.Semaphore] = {}
    for n in spec.nodes:
        semaphores[n.node_id] = asyncio.Semaphore(n.concurrency or 1)

    # ready queue: nodes with indegree 0
    ready = [nid for nid, deg in indegree.items() if deg == 0]
    running_tasks: Dict[str, asyncio.Task] = {}
    node_results: Dict[str, Dict[str, Any]] = {}
    current_nodes = []

    # run context passed to executors
    run_ctx = {
        "run_id": run.run_id,
        "workflow_id": run.workflow_id,
        "input": run.input_variables,
        "variables": dict(run.input_variables),
        "start_time": run.started_at,
    }

    # helper to mark node completion and enqueue children (respecting SWITCH conditions)
    async def _on_node_done(node_id: str, outcome: Dict[str, Any]):
        node_results[node_id] = outcome
        # find children
        for (child, cond) in adj.get(node_id, []):
            # if edge has condition, evaluate based on run_ctx and outcome
            allowed = True
            if cond:
                # prepare variables: exposed as 'result' and 'input'
                vars_for_cond = {"result": outcome.get("result"), "input": run_ctx["input"], **run_ctx.get("variables", {})}
                allowed = _safe_eval_condition(cond, vars_for_cond)
            if allowed:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

    # main loop: continue while ready nodes or running tasks exist
    try:
        while ready or running_tasks:
            # fire off all ready nodes up to concurrency (we'll start all and let semaphores limit)
            while ready:
                nid = ready.pop(0)
                node = node_map.get(nid)
                if not node:
                    continue
                # skip START/END node execution but log them
                if node.type in (NodeType.START, NodeType.END):
                    await _append_run_log(run.run_id, {"ts": _now_iso(), "node_id": nid, "action": "noop", "node_type": node.type.value})
                    # treat as completed
                    await _on_node_done(nid, {"success": True, "result": None})
                    continue
                # for SWITCH nodes, evaluate conditions to determine which child edges to consider.
                if node.type == NodeType.SWITCH:
                    # for switch, we consider it a virtual node: we evaluate condition expressions on outgoing edges and decrement indegrees for allowed ones
                    # prepare a fake outcome
                    outcome = {"success": True, "result": None}
                    # mark done immediately and let _on_node_done handle enqueue
                    await _on_node_done(nid, outcome)
                    continue

                # create coroutine to execute node
                sem = semaphores.get(nid)
                task_coro = _execute_node(node, run, run_ctx, sem)
                task = asyncio.create_task(task_coro)
                running_tasks[nid] = task
                current_nodes.append(nid)
                # attach completion callback to process children when done
                def _make_cb(node_id):
                    async def _cb(fut: asyncio.Future):
                        try:
                            res = fut.result()
                        except Exception as e:
                            res = {"success": False, "error": str(e)}
                        await _on_node_done(node_id, res)
                        # remove from running
                        running_tasks.pop(node_id, None)
                        if node_id in current_nodes:
                            current_nodes.remove(node_id)
                    return _cb
                task.add_done_callback(lambda fut, nid=nid: asyncio.create_task(_make_cb(nid)(fut)))
            # allow a short wait for tasks to progress
            if running_tasks:
                # wait for any task to complete or timeout small interval
                done, pending = await asyncio.wait(list(running_tasks.values()), timeout=1, return_when=asyncio.FIRST_COMPLETED)
                # loop continues to pick up newly ready nodes
                continue
            else:
                # no running tasks and no ready nodes -> break
                break

        # determine final status: if any node failed then run failed
        any_failed = any(not (node_results.get(nid, {}).get("success", False)) for nid in node_map.keys())
        run.completed_at = _now_iso()
        if any_failed:
            run.status = RunStatus.FAILED
            run.result = {"node_results": node_results}
            await _append_run_log(run.run_id, {"event": "run_failed", "node_results": node_results, "ts": run.completed_at})
            await _broadcast_workflow_event({"event": "run_failed", "run_id": run.run_id, "ts": run.completed_at})
        else:
            run.status = RunStatus.COMPLETED
            run.result = {"node_results": node_results}
            await _append_run_log(run.run_id, {"event": "run_completed", "node_results": node_results, "ts": run.completed_at})
            await _broadcast_workflow_event({"event": "run_completed", "run_id": run.run_id, "ts": run.completed_at})
    except Exception:
        LOG.exception("Unhandled exception in workflow run executor for %s", run.run_id)
        run.status = RunStatus.FAILED
        run.completed_at = _now_iso()
        run.result = {"error": "executor_crash"}
        await _append_run_log(run.run_id, {"event": "run_failed_unhandled", "ts": run.completed_at})
        await _broadcast_workflow_event({"event": "run_failed_unhandled", "run_id": run.run_id, "ts": run.completed_at})
    finally:
        # finalize run: persist summary record
        await _persist_run_record(run)
        # remove from active map
        async with _RUN_LOCK:
            if run.run_id in _ACTIVE_RUNS:
                _ACTIVE_RUNS.pop(run.run_id, None)
        return run

# -------------------------
# Public endpoints to start/stop runs and fetch run info
# -------------------------
class RunStartPayload(BaseModel):
    workflow_id: str
    input_variables: Optional[Dict[str, Any]] = Field(default_factory=dict)
    spec_version: Optional[int] = None
    run_id: Optional[str] = None

@router.post("/run/start", dependencies=[Depends(require_role(Role.OPERATOR))])
async def start_run(payload: RunStartPayload = Body(...), background: BackgroundTasks = None, user = Depends(get_current_user)):
    """
    Start a workflow run asynchronously. Returns run_id and basic metadata.
    """
    meta = await _load_workflow_meta(payload.workflow_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # choose spec version
    spec_version = payload.spec_version or meta.spec.version
    run = WorkflowRun(
        run_id = payload.run_id or f"run_{uuid.uuid4().hex[:12]}",
        workflow_id = payload.workflow_id,
        spec_version = spec_version,
        input_variables = payload.input_variables or {},
        status = RunStatus.PENDING,
        attempts = 0
    )
    # persist initial record and run header log
    await _persist_run_record(run)
    await _append_run_log(run.run_id, {"_meta": {"workflow_id": run.workflow_id, "spec_version": run.spec_version, "input_variables": run.input_variables, "status": "pending", "created_at": _now_iso()}})
    await write_audit_event({"event": "run_created", "run_id": run.run_id, "workflow_id": run.workflow_id, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    # schedule background execution
    async with _RUN_LOCK:
        if run.run_id in _ACTIVE_RUNS:
            raise HTTPException(status_code=400, detail="Run with id already active")
        task = asyncio.create_task(_execute_workflow_run(run))
        _ACTIVE_RUNS[run.run_id] = task
    await _broadcast_workflow_event({"event": "run_scheduled", "run_id": run.run_id, "workflow_id": run.workflow_id, "ts": _now_iso()})
    return {"ok": True, "run_id": run.run_id, "workflow_id": run.workflow_id}

@router.post("/run/stop", dependencies=[Depends(require_role(Role.OPERATOR))])
async def stop_run(run_id: str = Query(...), user = Depends(get_current_user)):
    """
    Request to cancel/stop an ongoing run. Best-effort: cancels the asyncio task.
    """
    async with _RUN_LOCK:
        t = _ACTIVE_RUNS.get(run_id)
        if not t:
            # if not active, mark run canceled in DB
            run = await _load_run(run_id)
            if run:
                run.status = RunStatus.CANCELED
                run.completed_at = _now_iso()
                await _persist_run_record(run)
                await _append_run_log(run_id, {"event": "run_canceled", "ts": run.completed_at})
                await _broadcast_workflow_event({"event": "run_canceled", "run_id": run_id, "ts": run.completed_at})
                return {"ok": True, "message": "Run canceled (not actively running)"}
            raise HTTPException(status_code=404, detail="Run not found")
        # cancel task
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        run = await _load_run(run_id)
        if run:
            run.status = RunStatus.CANCELED
            run.completed_at = _now_iso()
            await _persist_run_record(run)
            await _append_run_log(run_id, {"event": "run_canceled", "ts": run.completed_at})
        _ACTIVE_RUNS.pop(run_id, None)
    await _broadcast_workflow_event({"event": "run_canceled", "run_id": run_id, "ts": _now_iso()})
    await write_audit_event({"event": "run_stop_requested", "run_id": run_id, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    return {"ok": True, "run_id": run_id}

@router.get("/run/{run_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_run(run_id: str):
    run = await _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@router.get("/runs", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_runs(workflow_id: Optional[str] = Query(None), limit: int = Query(50)):
    """
    List recent runs. When Mongo is available, use DB; otherwise scan runs dir.
    """
    runs = []
    if _workflow_meta_col is not None:
        try:
            runs_col = _workflow_meta_col.database.get_collection(os.getenv("WORKFLOW_RUNS_COLLECTION", "prioritymax_workflow_runs"))
            q = {}
            if workflow_id:
                q["workflow_id"] = workflow_id
            docs = await runs_col.find(q).sort("started_at", -1).to_list(length=limit)
            for d in docs:
                d.pop("_id", None)
                runs.append(d)
            return {"runs": runs}
        except Exception:
            LOG.exception("Mongo query for runs failed; falling back to FS")
    # FS fallback: read run summary json files
    for f in sorted(_RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            runs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    if workflow_id:
        runs = [r for r in runs if r.get("workflow_id") == workflow_id]
    return {"runs": runs[:limit]}

# End of Chunk 4/7
# -------------------------
# Chunk 5/7 — Pluggable Executors (Python, HTTP, Container, External)
# -------------------------

import subprocess
import aiohttp
import importlib
import inspect

# -------------------------
# Python Function Executor
# -------------------------
async def executor_python(node: WorkflowNode, run_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a Python callable specified in node.template.config['callable'].
    Example callable string: "package.module:function_name"
    Passes run_ctx['input'] and node.metadata as parameters.
    """
    callable_path = node.template.config.get("callable")
    if not callable_path:
        raise RuntimeError("Python executor missing callable config")
    try:
        module_name, func_name = callable_path.split(":")
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name, None)
        if not fn:
            raise RuntimeError(f"Function {func_name} not found in {module_name}")
        if inspect.iscoroutinefunction(fn):
            result = await fn(run_ctx.get("input", {}), node.metadata or {})
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_THREAD_POOL, lambda: fn(run_ctx.get("input", {}), node.metadata or {}))
        await _append_run_log(run_ctx["run_id"], {"node": node.node_id, "executor": "python", "result": result, "ts": _now_iso()})
        return {"success": True, "result": result}
    except Exception as e:
        LOG.exception("Python executor error for node %s", node.node_id)
        return {"success": False, "error": str(e)}

# -------------------------
# HTTP Executor
# -------------------------
async def executor_http(node: WorkflowNode, run_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes an HTTP request.
    node.template.config must contain:
      - url
      - method (default POST)
      - headers (optional)
      - payload_template (optional): if present, formatted with run_ctx['input']
    """
    url = node.template.config.get("url")
    if not url:
        raise RuntimeError("HTTP executor missing URL config")
    method = node.template.config.get("method", "POST").upper()
    headers = node.template.config.get("headers", {"Content-Type": "application/json"})
    payload_tpl = node.template.config.get("payload_template")
    timeout = aiohttp.ClientTimeout(total=node.template.timeout_seconds or 10)
    payload = {}
    try:
        if payload_tpl:
            # allow formatting placeholders from input variables
            payload = json.loads(json.dumps(payload_tpl).format(**run_ctx.get("input", {})))
        else:
            payload = run_ctx.get("input", {})
    except Exception:
        payload = run_ctx.get("input", {})

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            if method == "GET":
                async with session.get(url, headers=headers) as resp:
                    data = await resp.json(content_type=None)
            else:
                async with session.request(method, url, headers=headers, json=payload) as resp:
                    data = await resp.json(content_type=None)
            await _append_run_log(run_ctx["run_id"], {"node": node.node_id, "executor": "http", "status": resp.status, "result": data, "ts": _now_iso()})
            return {"success": True, "result": data, "status": resp.status}
        except Exception as e:
            LOG.exception("HTTP executor error for %s", url)
            return {"success": False, "error": str(e)}

# -------------------------
# Container Executor (Kubernetes or Docker)
# -------------------------
async def executor_container(node: WorkflowNode, run_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a container image via subprocess (docker run) or k8s job.
    node.template.config must contain:
      - image
      - command (list or string)
    Optional ENV: WORKFLOW_CONTAINER_MODE=[docker|k8s]
    """
    image = node.template.config.get("image")
    if not image:
        raise RuntimeError("Container executor missing image config")

    cmd = node.template.config.get("command", [])
    if isinstance(cmd, str):
        cmd = cmd.split()
    mode = os.getenv("WORKFLOW_CONTAINER_MODE", "docker")

    try:
        if mode == "docker":
            docker_cmd = ["docker", "run", "--rm", image] + cmd
            LOG.info("Executing container: %s", " ".join(docker_cmd))
            proc = await asyncio.create_subprocess_exec(*docker_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            rc = proc.returncode
            result = stdout.decode().strip() if stdout else ""
            err = stderr.decode().strip() if stderr else ""
            await _append_run_log(run_ctx["run_id"], {"node": node.node_id, "executor": "container", "stdout": result, "stderr": err, "rc": rc, "ts": _now_iso()})
            success = rc == 0
            return {"success": success, "result": result, "stderr": err, "rc": rc}
        elif mode == "k8s":
            # K8s job creation (placeholder)
            await _append_run_log(run_ctx["run_id"], {"node": node.node_id, "executor": "container", "mode": "k8s", "image": image, "cmd": cmd, "ts": _now_iso()})
            return {"success": True, "result": f"k8s_job:{image}"}
        else:
            raise RuntimeError(f"Unsupported container mode {mode}")
    except Exception as e:
        LOG.exception("Container executor failed for %s", image)
        return {"success": False, "error": str(e)}

# -------------------------
# External Executor (Webhook or Custom Integration)
# -------------------------
async def executor_external(node: WorkflowNode, run_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Posts workflow event to an external service or webhook URL.
    Example config:
      {
        "url": "https://hooks.slack.com/...",
        "headers": {"Authorization": "Bearer ..."},
        "payload_template": {"text": "Workflow {workflow_id} run {run_id} completed."}
      }
    """
    cfg = node.template.config
    url = cfg.get("url")
    if not url:
        raise RuntimeError("External executor missing URL")
    headers = cfg.get("headers", {"Content-Type": "application/json"})
    payload = cfg.get("payload_template", {})
    # interpolate basic fields
    try:
        payload_json = json.loads(json.dumps(payload).format(**run_ctx))
    except Exception:
        payload_json = payload
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload_json) as resp:
                data = await resp.text()
                await _append_run_log(run_ctx["run_id"], {"node": node.node_id, "executor": "external", "url": url, "status": resp.status, "ts": _now_iso()})
                return {"success": True, "status": resp.status, "result": data}
        except Exception as e:
            LOG.exception("External executor error for node %s", node.node_id)
            return {"success": False, "error": str(e)}

# -------------------------
# Executor Registry Initialization
# -------------------------
def _register_executors():
    EXECUTOR_REGISTRY["python"] = executor_python
    EXECUTOR_REGISTRY["http"] = executor_http
    EXECUTOR_REGISTRY["container"] = executor_container
    EXECUTOR_REGISTRY["external"] = executor_external
    LOG.info("Workflow executors registered: %s", list(EXECUTOR_REGISTRY.keys()))

_register_executors()

# -------------------------
# Executor management endpoints (for admins)
# -------------------------
@router.get("/executors", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_executors():
    """
    List available executor types registered in system.
    """
    return {"executors": list(EXECUTOR_REGISTRY.keys())}

@router.post("/executors/reload", dependencies=[Depends(require_role(Role.ADMIN))])
async def reload_executors(user = Depends(get_current_user)):
    """
    Force reload executors registry (useful after new code deploys).
    """
    _register_executors()
    await write_audit_event({"event": "reload_executors", "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
    await _broadcast_workflow_event({"event": "executors_reloaded", "ts": _now_iso()})
    return {"ok": True, "executors": list(EXECUTOR_REGISTRY.keys())}

# End of Chunk 5/7
# -------------------------
# Chunk 6/7 — Observability, Tracing, WebSocket, Trace Replay, Prometheus metrics
# -------------------------

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, PlainTextResponse
import itertools

# Prometheus metrics for workflows subsystem (optional)
try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _HAS_PROM = True
    _WF_PROM_REG = CollectorRegistry()
    WF_RUNS_STARTED = Counter("prioritymax_workflow_runs_started_total", "Total workflow runs started", registry=_WF_PROM_REG)
    WF_RUNS_COMPLETED = Counter("prioritymax_workflow_runs_completed_total", "Total workflow runs completed", registry=_WF_PROM_REG)
    WF_RUNS_FAILED = Counter("prioritymax_workflow_runs_failed_total", "Total workflow runs failed", registry=_WF_PROM_REG)
    WF_RUNS_ACTIVE = Gauge("prioritymax_workflow_runs_active", "Active workflow runs", registry=_WF_PROM_REG)
except Exception:
    _HAS_PROM = False
    WF_RUNS_STARTED = WF_RUNS_COMPLETED = WF_RUNS_FAILED = WF_RUNS_ACTIVE = None
    generate_latest = CONTENT_TYPE_LATEST = None

# WebSocket connections for workflow events
_WORKFLOWS_WS: List[WebSocket] = []
WORKFLOWS_WS_AUTH_TOKEN = os.getenv("WORKFLOWS_WS_AUTH_TOKEN", None)

@router.websocket("/ws/workflows")
async def ws_workflows(websocket: WebSocket, token: Optional[str] = None):
    """
    WebSocket connection for live workflow events and logs.
    If WORKFLOWS_WS_AUTH_TOKEN is set, require it as query param 'token'.
    Clients may send {"action":"subscribe", "run_id":"..."} to filter by run.
    """
    if WORKFLOWS_WS_AUTH_TOKEN and token != WORKFLOWS_WS_AUTH_TOKEN:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    client = {"ws": websocket, "subscribed_runs": set()}
    _WORKFLOWS_WS.append(client)
    LOG.info("Workflow WS client connected (%d)", len(_WORKFLOWS_WS))
    try:
        while True:
            try:
                msg = await websocket.receive_text()
                try:
                    obj = json.loads(msg)
                    action = obj.get("action")
                    if action == "subscribe":
                        rid = obj.get("run_id")
                        if rid:
                            client["subscribed_runs"].add(rid)
                            await websocket.send_text(json.dumps({"subscribed": list(client["subscribed_runs"])}))
                    elif action == "unsubscribe":
                        rid = obj.get("run_id")
                        if rid and rid in client["subscribed_runs"]:
                            client["subscribed_runs"].remove(rid)
                    elif action == "ping":
                        await websocket.send_text(json.dumps({"pong": _now_iso()}))
                except Exception:
                    # ignore invalid json
                    continue
            except WebSocketDisconnect:
                break
            except Exception:
                await asyncio.sleep(0.1)
                continue
    finally:
        try:
            _WORKFLOWS_WS.remove(client)
        except Exception:
            pass
        LOG.info("Workflow WS disconnected (%d)", len(_WORKFLOWS_WS))

async def _broadcast_workflow_event(payload: Dict[str, Any]):
    """
    Broadcast workflow payload to websocket clients subscribing to run_id (if provided),
    otherwise broadcast to all.
    """
    payload.setdefault("ts", datetime.datetime.utcnow().isoformat() + "Z")
    # emit audit as well
    try:
        await write_audit_event({"source": "workflow_broadcast", "payload": payload, "ts": payload.get("ts")})
    except Exception:
        LOG.exception("Failed to write audit for workflow broadcast")

    # update Prometheus counters for run events
    try:
        ev = payload.get("event", "")
        if ev == "run_started" and WF_RUNS_STARTED:
            WF_RUNS_STARTED.inc()
            try:
                WF_RUNS_ACTIVE.inc()
            except Exception:
                pass
        elif ev == "run_completed" and WF_RUNS_COMPLETED:
            WF_RUNS_COMPLETED.inc()
            try:
                WF_RUNS_ACTIVE.dec()
            except Exception:
                pass
        elif ev == "run_failed" and WF_RUNS_FAILED:
            WF_RUNS_FAILED.inc()
            try:
                WF_RUNS_ACTIVE.dec()
            except Exception:
                pass
    except Exception:
        LOG.exception("Failed updating workflow prom counters")

    # send to connected sockets
    stale = []
    j = json.dumps(payload, default=str)
    for client in list(_WORKFLOWS_WS):
        ws = client.get("ws")
        try:
            subscribed = client.get("subscribed_runs", set())
            rid = payload.get("run_id")
            if subscribed and rid and rid not in subscribed:
                continue
            await ws.send_text(j)
        except Exception:
            stale.append(client)
    for s in stale:
        try:
            _WORKFLOWS_WS.remove(s)
        except Exception:
            pass

# -------------------------
# OpenTelemetry / Jaeger initialization endpoint
# -------------------------
@router.post("/observability/otel/init", dependencies=[Depends(require_role(Role.ADMIN))])
async def init_otel(service_name: Optional[str] = Body("prioritymax-workflows"), jaeger_host: Optional[str] = Body(None), jaeger_port: Optional[int] = Body(None), user = Depends(get_current_user)):
    if not _HAS_OTEL:
        raise HTTPException(status_code=404, detail="OpenTelemetry not available in environment")
    try:
        host = jaeger_host or os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST")
        port = jaeger_port or int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831"))
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        jaeger_exporter = JaegerExporter(agent_host_name=host, agent_port=port)
        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)
        otel_trace.set_tracer_provider(provider)
        await write_audit_event({"event": "init_otel", "service": service_name, "host": host, "port": port, "user": getattr(user, "username", "unknown"), "ts": _now_iso()})
        return {"ok": True, "service": service_name, "host": host, "port": port}
    except Exception:
        LOG.exception("Failed to initialize OTEL")
        raise HTTPException(status_code=500, detail="OTEL initialization failed")

# -------------------------
# Tracing helpers used by execution engine
# -------------------------
def _start_tracing_span(name: str):
    """
    Start a tracing span if OTEL is setup. Returns a context-manager like object.
    """
    if not _HAS_OTEL:
        class _Noop:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        return _Noop()
    tracer = otel_trace.get_tracer(__name__)
    return tracer.start_as_current_span(name)

# -------------------------
# Trace Replay endpoint (stream run logs frame-by-frame)
# -------------------------
def _iter_run_log_frames(run_id: str, delay_seconds: float = 0.2):
    """
    Generator that yields newline-delimited JSON frames from run log file,
    simulating a replay. Delay is applied between frames.
    """
    p = _run_file_for(run_id)
    if not p.exists():
        yield json.dumps({"error": "run_logs_not_found", "run_id": run_id}) + "\n"
        return
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                # yield each log line as-is
                yield line.strip() + "\n"
                time.sleep(delay_seconds)
            except GeneratorExit:
                break
            except Exception:
                continue

@router.get("/trace/replay/{run_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def trace_replay(run_id: str, delay_ms: int = Query(200)):
    """
    Stream run logs as a simple newline-delimited JSON stream for UI trace replay.
    """
    # streaming generator wrapper
    gen = lambda: _iter_run_log_frames(run_id, delay_seconds=(delay_ms / 1000.0))
    return StreamingResponse(gen(), media_type="application/x-ndjson")

# -------------------------
# Run logs fetch & stream endpoints
# -------------------------
@router.get("/run/{run_id}/logs", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_run_logs(run_id: str, tail: Optional[int] = Query(None)):
    p = _run_file_for(run_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Run logs not found")
    if tail:
        # return last N lines
        lines = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                lines.append(line)
        return {"run_id": run_id, "lines": lines[-tail:]}
    else:
        return StreamingResponse(open(p, "rb"), media_type="application/octet-stream")

# -------------------------
# Metrics scrape endpoint for workflows (Prometheus)
# -------------------------
@router.get("/metrics")
async def workflows_metrics():
    if not _HAS_PROM or _WF_PROM_REG is None:
        raise HTTPException(status_code=404, detail="Prometheus not enabled")
    try:
        payload = generate_latest(_WF_PROM_REG)
        return PlainTextResponse(payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        LOG.exception("Failed to render workflow prometheus metrics")
        raise HTTPException(status_code=500, detail="Failed to render metrics")

# End of Chunk 6/7
# -------------------------
# Chunk 7/7 — Background Maintenance, SLA Enforcement, Health, Startup/Shutdown, Exports
# -------------------------

_MAINTENANCE_TASK: Optional[asyncio.Task] = None
_MAINTENANCE_RUNNING = False
_MAINTENANCE_INTERVAL = int(os.getenv("WORKFLOW_MAINTENANCE_INTERVAL", "120"))  # seconds

async def _cleanup_old_runs(retention_days: int = WORKFLOW_RETENTION_DAYS):
    """
    Remove workflow run records older than retention_days.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
    removed = 0
    try:
        for f in _RUNS_DIR.glob("*.json"):
            mtime = datetime.datetime.utcfromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += 1
        for f in _RUNS_DIR.glob("*.jsonl"):
            mtime = datetime.datetime.utcfromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += 1
    except Exception:
        LOG.exception("Failed cleaning old runs")
    return removed

async def _enforce_sla():
    """
    Enforce SLAs for active runs by checking timestamps against each node's SLA.
    """
    now_ts = time.time()
    violations = []
    for f in _RUNS_DIR.glob("*.json"):
        try:
            run = json.loads(f.read_text(encoding="utf-8"))
            if run.get("status") not in (RunStatus.RUNNING, "running"):
                continue
            started = datetime.datetime.fromisoformat(run.get("started_at").replace("Z", "+00:00")).timestamp()
            elapsed = now_ts - started
            meta = await _load_workflow_meta(run.get("workflow_id"))
            if not meta:
                continue
            for n in meta.spec.nodes:
                if n.template and n.template.sla_seconds and elapsed > n.template.sla_seconds:
                    violations.append({"run_id": run.get("run_id"), "node_id": n.node_id, "sla": n.template.sla_seconds, "elapsed": elapsed})
                    await _broadcast_workflow_event({
                        "event": "sla_violation",
                        "run_id": run.get("run_id"),
                        "node_id": n.node_id,
                        "sla": n.template.sla_seconds,
                        "elapsed": round(elapsed, 2),
                        "ts": _now_iso()
                    })
        except Exception:
            continue
    return violations

async def _maintenance_loop():
    """
    Periodic background loop that cleans up old runs and enforces SLAs.
    """
    global _MAINTENANCE_RUNNING
    _MAINTENANCE_RUNNING = True
    LOG.info("Workflow maintenance loop started (interval=%ds)", _MAINTENANCE_INTERVAL)
    while True:
        try:
            cleaned = await _cleanup_old_runs()
            if cleaned:
                await write_audit_event({"event": "cleanup_old_runs", "count": cleaned, "ts": _now_iso()})
            violations = await _enforce_sla()
            if violations:
                await write_audit_event({"event": "sla_violations", "count": len(violations), "ts": _now_iso()})
            await asyncio.sleep(_MAINTENANCE_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception:
            LOG.exception("Workflow maintenance iteration failed")
            await asyncio.sleep(_MAINTENANCE_INTERVAL)
    _MAINTENANCE_RUNNING = False
    LOG.info("Workflow maintenance loop stopped")

def start_workflow_maintenance(loop: Optional[asyncio.AbstractEventLoop] = None):
    global _MAINTENANCE_TASK
    if _MAINTENANCE_TASK and not _MAINTENANCE_TASK.done():
        return
    loop = loop or asyncio.get_event_loop()
    _MAINTENANCE_TASK = loop.create_task(_maintenance_loop())
    LOG.info("Workflow maintenance background task scheduled")

def stop_workflow_maintenance():
    global _MAINTENANCE_TASK
    if _MAINTENANCE_TASK:
        _MAINTENANCE_TASK.cancel()
        _MAINTENANCE_TASK = None
        LOG.info("Workflow maintenance background task stopped")

# -------------------------
# Health & Diagnostics
# -------------------------
@router.get("/health")
async def workflows_health():
    redis_ok = False
    mongo_ok = False
    try:
        r = await get_redis()
        if r:
            await r.ping()
            redis_ok = True
    except Exception:
        pass
    if _workflow_meta_col is not None:
        try:
            await _workflow_meta_col.estimated_document_count()
            mongo_ok = True
        except Exception:
            pass
    return {
        "ok": True,
        "redis_connected": redis_ok,
        "mongo_connected": mongo_ok,
        "maintenance_running": _MAINTENANCE_RUNNING,
        "active_ws_clients": len(_WORKFLOWS_WS),
        "active_runs": len(_ACTIVE_RUNS)
    }

@router.get("/status", dependencies=[Depends(require_role(Role.VIEWER))])
async def workflows_status():
    active = list(_ACTIVE_RUNS.keys())
    return {
        "ok": True,
        "active_runs": len(active),
        "run_ids": active,
        "maintenance_running": _MAINTENANCE_RUNNING,
        "time": _now_iso()
    }

# -------------------------
# Startup / Shutdown Hooks
# -------------------------
@router.on_event("startup")
async def _on_workflows_startup():
    try:
        LOG.info("PriorityMax Workflows API starting up...")
        start_workflow_maintenance()
        await _broadcast_workflow_event({"event": "workflows_api_startup", "ts": _now_iso()})
    except Exception:
        LOG.exception("Workflows startup failed")

@router.on_event("shutdown")
async def _on_workflows_shutdown():
    try:
        stop_workflow_maintenance()
        await _broadcast_workflow_event({"event": "workflows_api_shutdown", "ts": _now_iso()})
        LOG.info("Workflows API shutdown complete")
    except Exception:
        LOG.exception("Workflows shutdown error")

# -------------------------
# Final Exports
# -------------------------
__all__ = [
    "router",
    "start_workflow_maintenance",
    "stop_workflow_maintenance",
    "_execute_workflow_run",
    "_register_executors",
    "EXECUTOR_REGISTRY",
]

# backend/app/api/predictor.py
"""
FastAPI service layer for PriorityMax Predictor

Exposes endpoints:
 - POST /predict            -> single inference (JSON features)
 - POST /batch_predict      -> batch inference (JSON array or NDJSON)
 - POST /explain            -> explanation (SHAP) for a single input or small batch
 - POST /drift_check        -> compute drift between two datasets (CSV upload or reference)
 - POST /trigger_retrain    -> trigger retraining (admin/operator)
 - GET  /predictor/info     -> info about loaded model(s)
 - GET  /predictor/health   -> lightweight health check

Security:
 - Attempts to reuse existing auth dependencies (app.auth.require_role/get_current_user).
 - Falls back to permissive/no-op auth if not available (use in dev only).
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import traceback
import tempfile
import pathlib
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form, status, Query
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

# Try to reuse your project's auth + audit utilities where possible
try:
    from app.auth import require_role, get_current_user, Role  # prefer app.auth
except Exception:
    # fallback to admin stubs if not present
    try:
        from app.api.admin import require_role, get_current_user, Role
    except Exception:
        # permissive fallback for dev environments
        def require_role(role):
            def _dep():
                return None
            return _dep
        async def get_current_user():
            return None
        class Role:
            ADMIN = "admin"
            OPERATOR = "operator"
            VIEWER = "viewer"

# audit helper (best-effort)
try:
    from app.api.admin import write_audit_event
except Exception:
    def write_audit_event(payload: Dict[str, Any]):
        p = pathlib.Path.cwd() / "backend" / "logs" / "predictor_api_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

# predictor manager
try:
    from app.ml.predictor import PREDICTOR_MANAGER, PredictorManager
except Exception:
    # If import fails, give a meaningful error at startup rather than failing import-time
    PREDICTOR_MANAGER = None
    PredictorManager = None

# prometheus optional
try:
    from prometheus_client import generate_latest, CollectorRegistry, Gauge, Histogram, CONTENT_TYPE_LATEST
    _HAS_PROM = True
except Exception:
    generate_latest = CollectorRegistry = Gauge = Histogram = CONTENT_TYPE_LATEST = None
    _HAS_PROM = False

LOG = logging.getLogger("prioritymax.api.predictor")
LOG.setLevel(os.getenv("PRIORITYMAX_API_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

router = APIRouter(prefix="/predictor", tags=["predictor"])

# Pydantic request/response schemas
class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name -> numeric value mapping")
    model_type: Optional[str] = Field("predictor_lgbm", description="Model type to use")

class PredictResponse(BaseModel):
    prediction: float
    latency_sec: float
    autoscaler_hint: Optional[str] = None
    model_version: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class BatchPredictRequest(BaseModel):
    inputs: List[Dict[str, float]] = Field(..., description="List of feature dicts")
    model_type: Optional[str] = Field("predictor_lgbm", description="Model type to use")
    batch_size: Optional[int] = Field(128, description="Internal batch size (for heavy models)")

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    count: int

class ExplainRequest(BaseModel):
    features: Dict[str, float]
    model_type: Optional[str] = Field("predictor_lgbm")
    top_k: Optional[int] = Field(10, description="Top-K feature contributions to return")

class ExplainResponse(BaseModel):
    explanations: Dict[str, float]
    model_version: Optional[str] = None

class DriftCheckResponse(BaseModel):
    avg_drift: float
    flag: bool
    per_feature: Dict[str, float]

class TriggerRetrainResponse(BaseModel):
    ok: bool
    message: Optional[str] = None

# Utility: make sure predictor manager exists
def _ensure_predictor_manager() -> PredictorManager:
    global PREDICTOR_MANAGER
    if PREDICTOR_MANAGER is None:
        raise HTTPException(status_code=500, detail="PredictorManager not available (check server logs)")
    return PREDICTOR_MANAGER

# -------------------------
# Endpoints
# -------------------------
@router.get("/health", summary="Predictor health")
async def predictor_health():
    mgr = PREDICTOR_MANAGER
    loaded = bool(mgr and mgr.last_loaded)
    return {"ok": True, "model_loaded": loaded, "model_info": getattr(mgr.last_loaded, "get", lambda k: None)("version_id") if mgr and mgr.last_loaded else None}

@router.get("/info", summary="Predictor info", dependencies=[Depends(require_role(Role.VIEWER))])
async def predictor_info():
    mgr = _ensure_predictor_manager()
    info = {}
    # list loaded model types
    with getattr(mgr, "lock", asyncio.Lock()):
        for k, v in mgr.models.items():
            info[k] = v.get_info() if hasattr(v, "get_info") else {"name": getattr(v, "name", None)}
    # last_loaded metadata
    info["last_loaded_meta"] = mgr.last_loaded or {}
    return info

@router.post("/predict", response_model=PredictResponse, summary="Single prediction (real-time)", dependencies=[Depends(require_role(Role.VIEWER))])
async def predict_endpoint(req: PredictRequest, background: BackgroundTasks, user = Depends(get_current_user)):
    mgr = _ensure_predictor_manager()
    start_ts = time.time()
    try:
        # prefer async path if available
        res = await mgr.predict_async(req.features, model_type=req.model_type)
        # res contains prediction fields as dict
        latency = res.get("latency_sec", round(time.time() - start_ts, 6))
        out = PredictResponse(
            prediction=float(res.get("prediction", 0.0)),
            latency_sec=latency,
            autoscaler_hint=res.get("autoscaler_hint"),
            model_version=res.get("model_version"),
            meta=res
        )
        # audit
        write_audit_event({"event": "predict_request", "user": getattr(user, "username", None), "features": list(req.features.keys()), "model_type": req.model_type, "result": {"prediction": out.prediction, "latency_sec": out.latency_sec}, "ts": time.time()})
        return out
    except Exception as e:
        LOG.exception("Predict failed")
        raise HTTPException(status_code=500, detail=f"prediction_failed: {str(e)}")

@router.post(
    "/batch_predict",
    response_model=BatchPredictResponse,
    summary="Batch predict (vectorized GPU/CPU)",
    dependencies=[Depends(require_role(Role.VIEWER))],
)
async def batch_predict_endpoint(
    req: BatchPredictRequest,
    user = Depends(get_current_user),
):
    """
    Efficient vectorized batch prediction endpoint.
    Uses PredictorManager.predict_batch() which automatically:
      - normalizes all features in one pass
      - uses GPU batching (if PyTorch + CUDA available)
      - splits large payloads into safe chunks
      - returns structured per-sample results
    """
    mgr = _ensure_predictor_manager()

    if not req.inputs or len(req.inputs) == 0:
        raise HTTPException(status_code=400, detail="Empty input list.")

    try:
        # Run in background threadpool (predict_batch uses vectorized execution internally)
        loop = asyncio.get_running_loop()
        preds = await loop.run_in_executor(
            None,
            mgr.predict_batch,
            req.inputs,
            req.model_type,
            4,  # max_workers (tuned for CPU/GPU)
            req.batch_size or 1024,
        )

        results = [PredictResponse(**p) for p in preds]

        write_audit_event({
            "event": "batch_predict_vectorized",
            "user": getattr(user, "username", None),
            "count": len(results),
            "model_type": req.model_type,
            "avg_pred": float(np.mean([r.prediction for r in results])) if results else None,
            "ts": time.time(),
        })

        return BatchPredictResponse(results=results, count=len(results))

    except Exception as e:
        LOG.exception("Batch predict failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

@router.post("/explain", response_model=ExplainResponse, summary="Explain prediction (SHAP)", dependencies=[Depends(require_role(Role.VIEWER))])
async def explain_endpoint(req: ExplainRequest, user = Depends(get_current_user)):
    mgr = _ensure_predictor_manager()
    # For SHAP, call the currently loaded model's explain method if available
    model = mgr.get_model(req.model_type)
    if not model:
        # attempt to load
        model = mgr.load_latest(req.model_type)
        if not model:
            raise HTTPException(status_code=404, detail="model_not_found")
    try:
        X_df = None
        # Build a single-row DataFrame like PredictorManager would
        import pandas as _pd
        X_df = _pd.DataFrame([req.features])
        # Normalize features (same as predict path)
        X_norm = mgr.normalize_features(X_df)
        arr = X_norm.values
        # call explain if model supports it
        expl = None
        if hasattr(model, "explain"):
            expl_vals = model.explain(arr)
            # interpret expl_vals depending on SHAP output shape
            if isinstance(expl_vals, list):
                # multiclass -> take first class contributions average
                expl_arr = expl_vals[0] if len(expl_vals) > 0 else expl_vals
            else:
                expl_arr = expl_vals
            # get feature names from model or X_df
            fnames = getattr(model, "feature_names", list(X_df.columns))
            # shap may return (n_features,) for single-row
            contributions = {}
            try:
                flat = list(expl_arr[0]) if hasattr(expl_arr, "__len__") and not isinstance(expl_arr, float) else [float(expl_arr)]
                for i, v in enumerate(flat):
                    if i < len(fnames):
                        contributions[str(fnames[i])] = float(v)
                # sort and return top_k
                top_k = int(req.top_k or 10)
                sorted_items = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])
                return ExplainResponse(explanations=sorted_items, model_version=mgr.last_loaded.get("version_id") if mgr.last_loaded else None)
            except Exception:
                LOG.exception("Explain parse failed")
        # fallback: not supported
        raise HTTPException(status_code=400, detail="explain_not_supported_for_model")
    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("Explain failed")
        raise HTTPException(status_code=500, detail=f"explain_failed: {str(e)}")

@router.post("/drift_check", response_model=DriftCheckResponse, summary="Compute drift between two CSV uploads", dependencies=[Depends(require_role(Role.VIEWER))])
async def drift_check(
    file_a: UploadFile = File(..., description="CSV file A (new data)"),
    file_b: UploadFile = File(..., description="CSV file B (reference)"),
    user = Depends(get_current_user)
):
    """
    Accepts two CSV files and computes drift using PredictorManager.compute_drift.
    This is intentionally resource-limited — large files should be processed offline.
    """
    mgr = _ensure_predictor_manager()
    try:
        # save both to tmp files
        tdir = tempfile.mkdtemp(prefix="prioritymax_drift_")
        path_a = pathlib.Path(tdir) / "a.csv"
        path_b = pathlib.Path(tdir) / "b.csv"
        contents_a = await file_a.read()
        contents_b = await file_b.read()
        path_a.write_bytes(contents_a)
        path_b.write_bytes(contents_b)
        import pandas as pd
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        res = mgr.compute_drift(df_a, df_b)
        write_audit_event({"event": "drift_check", "user": getattr(user, "username", None), "res": {"avg_drift": res.get("avg_drift")}, "ts": time.time()})
        return DriftCheckResponse(avg_drift=float(res.get("avg_drift", 0.0)), flag=bool(res.get("flag", False)), per_feature=res.get("per_feature", {}))
    except Exception as e:
        LOG.exception("Drift check failed")
        raise HTTPException(status_code=500, detail=f"drift_failed: {str(e)}")

@router.post("/trigger_retrain", response_model=TriggerRetrainResponse, summary="Trigger retraining (admin/operator)", dependencies=[Depends(require_role(Role.OPERATOR))])
async def trigger_retrain(background: BackgroundTasks, user = Depends(get_current_user), reason: Optional[str] = Query("manual_api", description="Reason for retrain")):
    mgr = _ensure_predictor_manager()
    try:
        # call trigger_retrain which may call a script; run in background
        def _fire():
            try:
                ok = mgr.trigger_retrain(reason=reason)
                write_audit_event({"event": "trigger_retrain", "user": getattr(user, "username", None), "reason": reason, "ok": bool(ok), "ts": time.time()})
            except Exception:
                LOG.exception("Retrain job invocation failed")
                write_audit_event({"event": "trigger_retrain_failed", "user": getattr(user, "username", None), "reason": reason, "ts": time.time()})
        background.add_task(_fire)
        return TriggerRetrainResponse(ok=True, message="retrain_job_scheduled")
    except Exception as e:
        LOG.exception("Trigger retrain failed")
        raise HTTPException(status_code=500, detail=f"trigger_failed: {str(e)}")

# Prometheus-compatible metrics for predictor (optional)
if _HAS_PROM:
    _PROM_REG = CollectorRegistry()
    _PROM_PRED_LAT = Histogram("prioritymax_predictor_inference_seconds", "Prediction latency seconds", registry=_PROM_REG)
    _PROM_CONF = Gauge("prioritymax_predictor_confidence", "Last prediction confidence", registry=_PROM_REG)

    @router.get("/metrics")
    async def predictor_metrics():
        try:
            # push any available metrics (if PredictorManager exports them)
            payload = generate_latest(_PROM_REG)
            return JSONResponse(content=payload, media_type=CONTENT_TYPE_LATEST)
        except Exception:
            raise HTTPException(status_code=500, detail="metrics_error")
else:
    @router.get("/metrics")
    async def predictor_metrics_disabled():
        raise HTTPException(status_code=404, detail="prometheus_not_available")

# -------------------------
# Startup / shutdown hooks
# -------------------------
@router.on_event("startup")
async def _on_startup():
    LOG.info("Predictor API startup: loading default models")
    try:
        mgr = _ensure_predictor_manager()
        # try to load the default predictor model
        try:
            mgr.load_latest("predictor_lgbm")
        except Exception:
            LOG.exception("Failed load default predictor at startup")
    except Exception:
        LOG.exception("PredictorManager not initialized at startup")

@router.on_event("shutdown")
async def _on_shutdown():
    LOG.info("Predictor API shutdown")

# -------------------------
# FastAPI app helper (optional)
# -------------------------
def create_predictor_app() -> FastAPI:
    """
    Helper to create a small FastAPI app that registers the predictor router.
    Use this if you want to run the predictor as a separate microservice.
    """
    app = FastAPI(title="PriorityMax Predictor API")
    app.include_router(router)
    return app

# -------------------------
# Exports
# -------------------------
__all__ = [
    "router",
    "create_predictor_app",
]

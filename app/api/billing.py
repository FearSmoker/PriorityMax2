# backend/app/api/billing.py
"""
PriorityMax Billing API (Phase-3)

Features included:
- Plans CRUD
- Subscriptions (create / update / cancel / seat management)
- Usage metering (report, aggregate)
- Invoices (generate, estimate, PDF generation, store to S3/local)
- Payments via Stripe (create invoice, attempt payment, subscription billing)
- Webhook handling for Stripe and external billing events
- Coupons / promo codes
- Billing analytics & exports (CSV)
- Dunning & retry management (background scheduler hooks)
- RBAC integration with app.api.admin (require_role, get_current_user)
- Audit logging via write_audit_event

Persistence:
- Prefer MongoDB (motor) if MONGO_URL env var provided
- Otherwise fallback to local JSON files under backend/app/billing_meta/

Environment variables:
- MONGO_URL (optional)
- STRIPE_API_KEY (optional)
- STRIPE_WEBHOOK_SECRET (optional)
- S3_BUCKET (optional) for invoice storage
- BILLING_META_DIR (optional) local fallback directory
- BILLING_CURRENCY (optional, default "usd")

Note: This module tries to be self-contained and degrades gracefully when optional libs are not present.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import hmac
import math
import csv
import base64
import shutil
import logging
import hashlib
import asyncio
import pathlib
import datetime
from typing import Optional, List, Dict, Any, Tuple

from fastapi import APIRouter, Depends, HTTPException, Body, Request, BackgroundTasks, Query, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

# optional dependencies
try:
    import motor.motor_asyncio as motor_asyncio
    _HAS_MOTOR = True
except Exception:
    motor_asyncio = None
    _HAS_MOTOR = False

try:
    import stripe
    _HAS_STRIPE = True
except Exception:
    stripe = None
    _HAS_STRIPE = False

try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    _HAS_REPORTLAB = True
except Exception:
    _HAS_REPORTLAB = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False

# admin and audit helpers (expected to exist)
try:
    from app.api.admin import get_current_user, require_role, Role, write_audit_event
except Exception:
    # minimal fallback stubs; in practice, admin.py exists
    def get_current_user(*args, **kwargs):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
    def require_role(r):
        def _dep(*args, **kwargs):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth dependency missing")
        return _dep
    class Role:
        ADMIN = "admin"
        OPERATOR = "operator"
        VIEWER = "viewer"
    async def write_audit_event(e: dict):
        # fallback: append to local file
        p = pathlib.Path.cwd() / "backend" / "logs" / "billing_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e, default=str) + "\n")

# Logging
LOG = logging.getLogger("prioritymax.billing")
LOG.setLevel(os.getenv("PRIORITYMAX_BILLING_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_handler)

# Config and directories
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
BILLING_META_DIR = pathlib.Path(os.getenv("BILLING_META_DIR", str(BASE_DIR / "app" / "billing_meta")))
BILLING_META_DIR.mkdir(parents=True, exist_ok=True)

INVOICE_DIR = BILLING_META_DIR / "invoices"
INVOICE_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URL = os.getenv("MONGO_URL", None)
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", None)
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", None)
S3_BUCKET = os.getenv("S3_BUCKET", None)
BILLING_CURRENCY = os.getenv("BILLING_CURRENCY", "usd")
DEFAULT_TAX_PERCENT = float(os.getenv("BILLING_DEFAULT_TAX_PERCENT", "0.0"))

if _HAS_STRIPE and STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

# Persistence collections/names
_PLANS_COLLECTION = os.getenv("BILLING_PLANS_COLLECTION", "billing_plans")
_SUBS_COLLECTION = os.getenv("BILLING_SUBS_COLLECTION", "billing_subscriptions")
_USAGE_COLLECTION = os.getenv("BILLING_USAGE_COLLECTION", "billing_usage")
_INVOICE_COLLECTION = os.getenv("BILLING_INVOICE_COLLECTION", "billing_invoices")
_COUPON_COLLECTION = os.getenv("BILLING_COUPON_COLLECTION", "billing_coupons")
_CUSTOMER_COLLECTION = os.getenv("BILLING_CUSTOMER_COLLECTION", "billing_customers")

# Mongo client if available
if _HAS_MOTOR and MONGO_URL:
    motor_client = motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    _billing_db = motor_client.get_default_database()
    plans_col = _billing_db[_PLANS_COLLECTION]
    subs_col = _billing_db[_SUBS_COLLECTION]
    usage_col = _billing_db[_USAGE_COLLECTION]
    invoice_col = _billing_db[_INVOICE_COLLECTION]
    coupon_col = _billing_db[_COUPON_COLLECTION]
    customer_col = _billing_db[_CUSTOMER_COLLECTION]
    LOG.info("Billing: using MongoDB at %s", MONGO_URL)
else:
    plans_col = subs_col = usage_col = invoice_col = coupon_col = customer_col = None
    LOG.info("Billing: using filesystem fallback at %s", BILLING_META_DIR)

# Router
router = APIRouter(prefix="/billing", tags=["billing"])

# ----------------------
# Pydantic models
# ----------------------
class Money(BaseModel):
    amount: int = Field(..., description="Amount in smallest currency unit (cents)")
    currency: str = Field(BILLING_CURRENCY)

class PlanCreate(BaseModel):
    plan_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    price_per_month: float = Field(..., ge=0.0, description="USD per month")
    included_quota: Optional[int] = Field(0, description="Included usage units per month")
    unit_price_overage: Optional[float] = Field(0.0, description="USD per extra unit beyond included_quota")
    seats_limit: Optional[int] = Field(0, description="0 => unlimited")
    features: Optional[Dict[str, Any]] = {}
    currency: Optional[str] = BILLING_CURRENCY
    billing_interval: Optional[str] = Field("monthly", description="monthly | yearly")
    active: bool = True

    @validator("plan_id", pre=True, always=True)
    def set_plan_id(cls, v, values):
        if v:
            return v
        # generate slug-like id
        name = values.get("name", "plan")
        slug = name.strip().lower().replace(" ", "_")
        return f"{slug}_{uuid.uuid4().hex[:6]}"

class Plan(PlanCreate):
    created_at: str
    updated_at: str
    version: int = 1

class CustomerCreate(BaseModel):
    customer_id: Optional[str] = None
    org_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

    @validator("customer_id", pre=True, always=True)
    def default_customer_id(cls, v):
        return v or f"cust_{uuid.uuid4().hex[:8]}"

class Customer(CustomerCreate):
    created_at: str
    updated_at: str

class SubscriptionCreate(BaseModel):
    customer_id: str
    plan_id: str
    seats: Optional[int] = 1
    trial_days: Optional[int] = 0
    start_date: Optional[str] = None  # ISO
    metadata: Optional[Dict[str, Any]] = {}

class Subscription(BaseModel):
    subscription_id: str
    customer_id: str
    plan_id: str
    seats: int
    status: str  # active / cancelled / past_due / trialing
    current_period_start: str
    current_period_end: str
    started_at: str
    canceled_at: Optional[str] = None
    trial_end: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class UsageRecord(BaseModel):
    usage_id: Optional[str] = None
    customer_id: str
    subscription_id: Optional[str] = None
    metric: str
    quantity: int
    timestamp: Optional[str] = None

    @validator("usage_id", pre=True, always=True)
    def default_usage_id(cls, v):
        return v or f"usage_{uuid.uuid4().hex[:10]}"

class InvoiceLineItem(BaseModel):
    description: str
    amount_cents: int
    quantity: int = 1
    metadata: Optional[Dict[str, Any]] = {}

class InvoiceCreate(BaseModel):
    customer_id: str
    subscription_id: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    lines: List[InvoiceLineItem]
    tax_percent: Optional[float] = DEFAULT_TAX_PERCENT
    currency: Optional[str] = BILLING_CURRENCY
    metadata: Optional[Dict[str, Any]] = {}

class Invoice(BaseModel):
    invoice_id: str
    customer_id: str
    subscription_id: Optional[str]
    created_at: str
    period_start: str
    period_end: str
    lines: List[InvoiceLineItem]
    subtotal_cents: int
    tax_cents: int
    total_cents: int
    currency: str
    paid: bool = False
    paid_at: Optional[str] = None
    pdf_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class CouponCreate(BaseModel):
    coupon_id: Optional[str] = None
    percent_off: Optional[float] = Field(None, ge=0.0, le=100.0)
    amount_off_cents: Optional[int] = Field(None, ge=0)
    duration_days: Optional[int] = Field(None, ge=1)
    max_redemptions: Optional[int] = Field(None, ge=1)
    expires_at: Optional[str] = None  # ISO

    @validator("coupon_id", pre=True, always=True)
    def default_coupon_id(cls, v):
        return v or f"coupon_{uuid.uuid4().hex[:8]}"

class Coupon(CouponCreate):
    created_at: str
    redemptions: int = 0

# ----------------------
# Persistence helpers (filesystem fallback)
# ----------------------
def _fs_write_json(p: pathlib.Path, data: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

def _fs_read_json(p: pathlib.Path) -> Optional[Any]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        LOG.exception("Failed to read JSON %s", p)
        return None

# FS paths
_PLANS_FS = BILLING_META_DIR / "plans.json"
_CUSTOMERS_FS = BILLING_META_DIR / "customers.json"
_SUBS_FS = BILLING_META_DIR / "subscriptions.json"
_USAGE_FS = BILLING_META_DIR / "usage.jsonl"  # append-only
_INVOICES_FS = BILLING_META_DIR / "invoices.json"
_COUPONS_FS = BILLING_META_DIR / "coupons.json"

# initialize FS files
for p in (_PLANS_FS, _CUSTOMERS_FS, _SUBS_FS, _INVOICES_FS, _COUPONS_FS):
    if not p.exists():
        _fs_write_json(p, {})

# ----------------------
# Utility helpers
# ----------------------
def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

async def _audit(user: Any, action: str, resource: str, details: Optional[dict] = None):
    evt = {"user": getattr(user, "username", str(user)) if user else "system", "action": action, "resource": resource, "details": details or {}, "timestamp_utc": _now_iso()}
    try:
        await write_audit_event(evt)
    except Exception:
        LOG.exception("Audit event failed: %s", evt)

def _money_to_cents(amount_float: float) -> int:
    return int(round(amount_float * 100))

def _cents_to_money_str(cents: int, currency: str = BILLING_CURRENCY) -> str:
    return f"{cents/100:.2f} {currency.upper()}"

# ----------------------
# Plans: CRUD
# ----------------------
async def _save_plan_to_store(plan: Plan):
    data = plan.dict()
    if plans_col is not None:
        await plans_col.update_one({"plan_id": plan.plan_id}, {"$set": data}, upsert=True)
    else:
        allp = _fs_read_json(_PLANS_FS) or {}
        allp[plan.plan_id] = data
        _fs_write_json(_PLANS_FS, allp)

async def _get_plan_from_store(plan_id: str) -> Optional[Plan]:
    if plans_col is not None:
        doc = await plans_col.find_one({"plan_id": plan_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return Plan(**doc)
    allp = _fs_read_json(_PLANS_FS) or {}
    d = allp.get(plan_id)
    if not d:
        return None
    return Plan(**d)

async def _list_plans_from_store() -> List[Plan]:
    if plans_col is not None:
        docs = await plans_col.find({}).to_list(length=1000)
        pl = []
        for d in docs:
            d.pop("_id", None)
            pl.append(Plan(**d))
        return pl
    allp = _fs_read_json(_PLANS_FS) or {}
    return [Plan(**v) for v in allp.values()]

async def _delete_plan_from_store(plan_id: str):
    if plans_col is not None:
        await plans_col.delete_one({"plan_id": plan_id})
    else:
        allp = _fs_read_json(_PLANS_FS) or {}
        if plan_id in allp:
            del allp[plan_id]
            _fs_write_json(_PLANS_FS, allp)

@router.post("/plans", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_plan(payload: PlanCreate, user = Depends(get_current_user)):
    now = _now_iso()
    plan = Plan(**payload.dict(), created_at=now, updated_at=now, version=1)
    await _save_plan_to_store(plan)
    await _audit(user, "create_plan", plan.plan_id, {"payload": payload.dict()})
    return plan

@router.get("/plans", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_plans(active_only: bool = Query(False)):
    plans = await _list_plans_from_store()
    if active_only:
        plans = [p for p in plans if p.active]
    return plans

@router.get("/plans/{plan_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_plan(plan_id: str):
    plan = await _get_plan_from_store(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return plan

@router.put("/plans/{plan_id}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def update_plan(plan_id: str, payload: PlanCreate, user = Depends(get_current_user)):
    existing = await _get_plan_from_store(plan_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Plan not found")
    now = _now_iso()
    # bump version
    new_version = (existing.version or 1) + 1
    plan = Plan(**payload.dict(), plan_id=plan_id, created_at=existing.created_at, updated_at=now, version=new_version)
    await _save_plan_to_store(plan)
    await _audit(user, "update_plan", plan_id, {"payload": payload.dict()})
    return plan

@router.delete("/plans/{plan_id}", dependencies=[Depends(require_role(Role.ADMIN))])
async def delete_plan(plan_id: str, user = Depends(get_current_user)):
    existing = await _get_plan_from_store(plan_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Plan not found")
    await _delete_plan_from_store(plan_id)
    await _audit(user, "delete_plan", plan_id, {})
    return {"ok": True, "deleted": plan_id}

# ----------------------
# Customers
# ----------------------
async def _save_customer_store(cust: Customer):
    data = cust.dict()
    if customer_col is not None:
        await customer_col.update_one({"customer_id": cust.customer_id}, {"$set": data}, upsert=True)
    else:
        allc = _fs_read_json(_CUSTOMERS_FS) or {}
        allc[cust.customer_id] = data
        _fs_write_json(_CUSTOMERS_FS, allc)

async def _get_customer_store(customer_id: str) -> Optional[Customer]:
    if customer_col is not None:
        doc = await customer_col.find_one({"customer_id": customer_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return Customer(**doc)
    allc = _fs_read_json(_CUSTOMERS_FS) or {}
    d = allc.get(customer_id)
    if not d:
        return None
    return Customer(**d)

@router.post("/customers", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_customer(payload: CustomerCreate, user = Depends(get_current_user)):
    now = _now_iso()
    cust = Customer(**payload.dict(), created_at=now, updated_at=now)
    await _save_customer_store(cust)
    await _audit(user, "create_customer", cust.customer_id, {"payload": payload.dict()})
    return cust

@router.get("/customers/{customer_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_customer(customer_id: str):
    cust = await _get_customer_store(customer_id)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    return cust

# ----------------------
# Subscriptions
# ----------------------
async def _save_subscription(sub: Subscription):
    data = sub.dict()
    if subs_col is not None:
        await subs_col.update_one({"subscription_id": sub.subscription_id}, {"$set": data}, upsert=True)
    else:
        allsubs = _fs_read_json(_SUBS_FS) or {}
        allsubs[sub.subscription_id] = data
        _fs_write_json(_SUBS_FS, allsubs)

async def _get_subscription(subscription_id: str) -> Optional[Subscription]:
    if subs_col is not None:
        doc = await subs_col.find_one({"subscription_id": subscription_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return Subscription(**doc)
    allsubs = _fs_read_json(_SUBS_FS) or {}
    d = allsubs.get(subscription_id)
    if not d:
        return None
    return Subscription(**d)

async def _list_subscriptions_for_customer(customer_id: str) -> List[Subscription]:
    if subs_col is not None:
        docs = await subs_col.find({"customer_id": customer_id}).to_list(length=100)
        return [Subscription(**{k: v for k, v in d.items() if k != "_id"}) for d in docs]
    allsubs = _fs_read_json(_SUBS_FS) or {}
    return [Subscription(**v) for v in allsubs.values() if v["customer_id"] == customer_id]

@router.post("/subscriptions", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_subscription(payload: SubscriptionCreate, user = Depends(get_current_user)):
    # validate customer & plan
    cust = await _get_customer_store(payload.customer_id)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    plan = await _get_plan_from_store(payload.plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    now = datetime.datetime.utcnow()
    start = now
    trial_end = None
    if payload.trial_days and payload.trial_days > 0:
        trial_end = (now + datetime.timedelta(days=int(payload.trial_days))).isoformat() + "Z"
    period_end = (start + datetime.timedelta(days=30)).isoformat() + "Z"
    subs_id = f"sub_{uuid.uuid4().hex[:10]}"
    sub = Subscription(
        subscription_id=subs_id,
        customer_id=payload.customer_id,
        plan_id=payload.plan_id,
        seats=payload.seats or 1,
        status="trialing" if trial_end else "active",
        current_period_start=start.isoformat() + "Z",
        current_period_end=period_end,
        started_at=start.isoformat() + "Z",
        canceled_at=None,
        trial_end=trial_end,
        metadata=payload.metadata or {}
    )
    await _save_subscription(sub)
    await _audit(user, "create_subscription", sub.subscription_id, {"customer": payload.customer_id, "plan": payload.plan_id})
    # optionally create Stripe subscription
    if _HAS_STRIPE and STRIPE_API_KEY:
        try:
            # Map plan -> stripe price if user provided metadata or plan contains a stripe id
            stripe_price_id = plan.features.get("stripe_price_id") if isinstance(plan.features, dict) else None
            if stripe_price_id:
                stripe_cust_id = cust.metadata.get("stripe_customer_id")
                if not stripe_cust_id:
                    # create stripe customer
                    stripe_c = stripe.Customer.create(email=cust.email, name=cust.name, metadata={"prioritymax_cust": cust.customer_id})
                    stripe_cust_id = stripe_c["id"]
                    # save it to customer metadata
                    cust.metadata["stripe_customer_id"] = stripe_cust_id
                    cust.updated_at = _now_iso()
                    await _save_customer_store(cust)
                stripe_sub = stripe.Subscription.create(customer=stripe_cust_id, items=[{"price": stripe_price_id}], expand=["latest_invoice.payment_intent"])
                sub.metadata["stripe_subscription_id"] = stripe_sub["id"]
                sub.status = stripe_sub["status"]
                await _save_subscription(sub)
        except Exception:
            LOG.exception("Stripe subscription creation failed")
    return sub

@router.post("/subscriptions/{subscription_id}/cancel", dependencies=[Depends(require_role(Role.OPERATOR))])
async def cancel_subscription(subscription_id: str, at_period_end: bool = Query(True), user = Depends(get_current_user)):
    sub = await _get_subscription(subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if sub.status == "cancelled":
        return {"ok": True, "message": "Already cancelled"}
    sub.status = "cancelled"
    sub.canceled_at = _now_iso()
    await _save_subscription(sub)
    await _audit(user, "cancel_subscription", subscription_id, {"at_period_end": at_period_end})
    # Stripe cancel if present
    try:
        sid = sub.metadata.get("stripe_subscription_id")
        if _HAS_STRIPE and sid:
            stripe.Subscription.delete(sid)
    except Exception:
        LOG.exception("Stripe cancel failed for subscription %s", subscription_id)
    return {"ok": True, "cancelled": subscription_id}

@router.get("/customers/{customer_id}/subscriptions", dependencies=[Depends(require_role(Role.VIEWER))])
async def list_customer_subscriptions(customer_id: str):
    subs = await _list_subscriptions_for_customer(customer_id)
    return subs

# ----------------------
# Usage Metering
# ----------------------
async def _append_usage_fs(record: UsageRecord):
    p = _USAGE_FS
    line = json.dumps(record.dict(), default=str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

async def _insert_usage_db(record: dict):
    if usage_col is not None:
        await usage_col.insert_one(record)

@router.post("/usage/report", dependencies=[Depends(require_role(Role.OPERATOR))])
async def report_usage(payload: UsageRecord, user = Depends(get_current_user)):
    payload.timestamp = payload.timestamp or _now_iso()
    rec = payload.dict()
    # persist
    try:
        if usage_col is not None:
            await _insert_usage_db(rec)
        else:
            await _append_usage_fs(payload)
    except Exception:
        LOG.exception("Failed to persist usage")
    await _audit(user, "report_usage", payload.usage_id, {"customer": payload.customer_id, "metric": payload.metric, "quantity": payload.quantity})
    return {"ok": True, "usage_id": payload.usage_id}

@router.get("/usage/aggregate", dependencies=[Depends(require_role(Role.VIEWER))])
async def aggregate_usage(customer_id: Optional[str] = None, metric: Optional[str] = None, since_days: int = Query(30)):
    """
    Aggregate usage records for reporting. If Mongo available, use aggregation pipeline.
    """
    since_ts = (datetime.datetime.utcnow() - datetime.timedelta(days=since_days)).isoformat() + "Z"
    aggregates = {}
    if usage_col is not None:
        match = {}
        if customer_id:
            match["customer_id"] = customer_id
        if metric:
            match["metric"] = metric
        match["timestamp"] = {"$gte": since_ts}
        pipeline = [{"$match": match}, {"$group": {"_id": {"customer": "$customer_id", "metric": "$metric"}, "total": {"$sum": "$quantity"}}}]
        docs = await usage_col.aggregate(pipeline).to_list(length=1000)
        for d in docs:
            key = f"{d['_id']['customer']}:{d['_id']['metric']}"
            aggregates[key] = d["total"]
    else:
        # read usage.jsonl and sum
        if _USAGE_FS.exists():
            with open(_USAGE_FS, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        if customer_id and r.get("customer_id") != customer_id:
                            continue
                        if metric and r.get("metric") != metric:
                            continue
                        if r.get("timestamp") < since_ts:
                            continue
                        key = f"{r.get('customer_id')}:{r.get('metric')}"
                        aggregates[key] = aggregates.get(key, 0) + int(r.get("quantity", 0))
                    except Exception:
                        continue
    return aggregates

# ----------------------
# Invoicing
# ----------------------
async def _save_invoice(inv: Invoice):
    data = inv.dict()
    if invoice_col is not None:
        await invoice_col.update_one({"invoice_id": inv.invoice_id}, {"$set": data}, upsert=True)
    else:
        allinv = _fs_read_json(_INVOICES_FS) or {}
        allinv[inv.invoice_id] = data
        _fs_write_json(_INVOICES_FS, allinv)
    # also write a JSON file for quick access
    p = INVOICE_DIR / f"{inv.invoice_id}.json"
    _fs_write_json(p, inv.dict())
    return inv

def _compute_invoice_totals(lines: List[InvoiceLineItem], tax_percent: float) -> Tuple[int, int, int]:
    subtotal = sum([li.amount_cents * li.quantity for li in lines])
    tax = int(math.ceil(subtotal * (tax_percent / 100.0)))
    total = subtotal + tax
    return subtotal, tax, total

def _generate_invoice_pdf(inv: Invoice, out_path: pathlib.Path) -> str:
    """
    Generate a simple PDF invoice using ReportLab if available.
    Otherwise write a JSON file and return that path.
    """
    if _HAS_REPORTLAB:
        c = canvas.Canvas(str(out_path), pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, f"PriorityMax Invoice: {inv.invoice_id}")
        c.setFont("Helvetica", 10)
        c.drawString(72, height - 96, f"Customer: {inv.customer_id}")
        c.drawString(72, height - 110, f"Created: {inv.created_at}")
        y = height - 150
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Description")
        c.drawString(400, y, "Amount")
        y -= 16
        c.setFont("Helvetica", 10)
        for li in inv.lines:
            c.drawString(72, y, li.description)
            c.drawString(400, y, _cents_to_money_str(li.amount_cents))
            y -= 14
            if y < 72:
                c.showPage()
                y = height - 72
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, f"Subtotal: {_cents_to_money_str(inv.subtotal_cents)}")
        y -= 16
        c.drawString(72, y, f"Tax: {_cents_to_money_str(inv.tax_cents)}")
        y -= 16
        c.drawString(72, y, f"Total: {_cents_to_money_str(inv.total_cents)}")
        c.save()
        return str(out_path)
    else:
        # fallback: write JSON file with .pdf.json extension
        out = out_path.with_suffix(".pdf.json")
        _fs_write_json(out, inv.dict())
        return str(out)

async def _store_invoice_asset(local_path: str, invoice_id: str):
    """
    Optionally upload invoice asset (PDF/JSON) to S3. Return stored path/URL.
    """
    if _HAS_BOTO3 and S3_BUCKET:
        s3 = boto3.client("s3")
        key = f"invoices/{invoice_id}/{pathlib.Path(local_path).name}"
        try:
            s3.upload_file(local_path, S3_BUCKET, key)
            url = f"s3://{S3_BUCKET}/{key}"
            return {"ok": True, "s3_key": key, "url": url}
        except Exception:
            LOG.exception("S3 upload failed for invoice %s", invoice_id)
    return {"ok": False, "local_path": local_path}

@router.post("/invoices/create", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_invoice(payload: InvoiceCreate, background_tasks: BackgroundTasks, user = Depends(get_current_user)):
    # validate customer
    cust = await _get_customer_store(payload.customer_id)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    now = _now_iso()
    pid = f"inv_{uuid.uuid4().hex[:10]}"
    period_start = payload.period_start or now
    period_end = payload.period_end or now
    subtotal, tax, total = _compute_invoice_totals(payload.lines, payload.tax_percent or 0.0)
    inv = Invoice(
        invoice_id=pid,
        customer_id=payload.customer_id,
        subscription_id=payload.subscription_id,
        created_at=now,
        period_start=period_start,
        period_end=period_end,
        lines=payload.lines,
        subtotal_cents=subtotal,
        tax_cents=tax,
        total_cents=total,
        currency=payload.currency or BILLING_CURRENCY,
        paid=False,
        paid_at=None,
        pdf_path=None,
        metadata=payload.metadata or {},
    )
    await _save_invoice(inv)
    await _audit(user, "create_invoice", pid, {"customer": payload.customer_id, "total_cents": total})
    # generate PDF or JSON in background
    out_pdf = INVOICE_DIR / f"{pid}.pdf"
    def _generate_and_store():
        try:
            pdf_path = _generate_invoice_pdf(inv, out_pdf)
            # store to DB path
            inv.pdf_path = pdf_path
            # update store
            if invoice_col is not None:
                # running sync in background thread; use motor? we call blocking file updates
                # but we can update via asyncio loop below; for simplicity write local
                pass
            _fs_write_json(INVOICE_DIR / f"{pid}.json", inv.dict())
            # optionally upload to s3
            _store_invoice_asset(pdf_path, pid)
        except Exception:
            LOG.exception("Failed to generate invoice asset for %s", pid)
    background_tasks.add_task(_generate_and_store)
    return inv

@router.get("/invoices/{invoice_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_invoice(invoice_id: str):
    # try DB
    if invoice_col is not None:
        doc = await invoice_col.find_one({"invoice_id": invoice_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Invoice not found")
        doc.pop("_id", None)
        return Invoice(**doc)
    allinv = _fs_read_json(_INVOICES_FS) or {}
    d = allinv.get(invoice_id)
    if not d:
        # try invoice file
        p = INVOICE_DIR / f"{invoice_id}.json"
        if p.exists():
            return Invoice(**json.loads(p.read_text(encoding="utf-8")))
        raise HTTPException(status_code=404, detail="Invoice not found")
    return Invoice(**d)

@router.get("/invoices/{invoice_id}/download", dependencies=[Depends(require_role(Role.VIEWER))])
async def download_invoice_asset(invoice_id: str):
    # return PDF if exists else JSON
    p_pdf = INVOICE_DIR / f"{invoice_id}.pdf"
    p_json = INVOICE_DIR / f"{invoice_id}.json"
    if p_pdf.exists():
        return FileResponse(str(p_pdf), filename=p_pdf.name, media_type="application/pdf")
    if p_json.exists():
        return FileResponse(str(p_json), filename=p_json.name, media_type="application/json")
    raise HTTPException(status_code=404, detail="Invoice asset not found")

# ----------------------
# Payment Integration: Stripe Webhooks & Payments
# ----------------------
def _verify_stripe_signature(payload: bytes, sig_header: str, secret: str) -> bool:
    """
    Verify Stripe webhook signature if STRIPE_WEBHOOK_SECRET is set.
    """
    if not _HAS_STRIPE or not secret:
        # no verification possible
        return False
    try:
        stripe.Webhook.construct_event(payload, sig_header, secret)
        return True
    except Exception:
        return False

@router.post("/webhooks/stripe")
async def stripe_webhook(req: Request):
    """
    Stripe webhook endpoint for handling invoice.paid, invoice.payment_failed, customer.subscription.deleted, etc.
    Must configure STRIPE_WEBHOOK_SECRET for validation.
    """
    raw = await req.body()
    sig_header = req.headers.get("stripe-signature", "")
    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(raw, sig_header, STRIPE_WEBHOOK_SECRET)
        except Exception as e:
            LOG.exception("Invalid stripe webhook signature: %s", e)
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        # best-effort parse (unsafe)
        try:
            event = json.loads(raw.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = event.get("type")
    data = event.get("data", {}).get("object", {})
    LOG.info("Stripe webhook received: %s", event_type)
    # basic event handling
    if event_type == "invoice.payment_succeeded":
        # find internal invoice via metadata or invoice number
        invnum = data.get("metadata", {}).get("prioritymax_invoice_id")
        amount_paid = data.get("amount_paid")
        currency = data.get("currency")
        if invnum:
            # mark invoice paid
            try:
                if invoice_col is not None:
                    await invoice_col.update_one({"invoice_id": invnum}, {"$set": {"paid": True, "paid_at": _now_iso(), "paid_amount_cents": amount_paid}})
                else:
                    allinv = _fs_read_json(_INVOICES_FS) or {}
                    v = allinv.get(invnum)
                    if v:
                        v["paid"] = True
                        v["paid_at"] = _now_iso()
                        v["paid_amount_cents"] = amount_paid
                        _fs_write_json(_INVOICES_FS, allinv)
                await _audit(None, "invoice_paid_webhook", invnum, {"amount_paid": amount_paid, "currency": currency})
            except Exception:
                LOG.exception("Failed to mark invoice paid for %s", invnum)
    elif event_type == "invoice.payment_failed":
        invnum = data.get("metadata", {}).get("prioritymax_invoice_id")
        if invnum:
            # mark invoice past_due and schedule dunning
            try:
                if invoice_col is not None:
                    await invoice_col.update_one({"invoice_id": invnum}, {"$set": {"paid": False, "status": "past_due"}})
                else:
                    allinv = _fs_read_json(_INVOICES_FS) or {}
                    v = allinv.get(invnum)
                    if v:
                        v["status"] = "past_due"
                        _fs_write_json(_INVOICES_FS, allinv)
                # schedule retry or dunning (stub here)
                await _audit(None, "invoice_payment_failed", invnum, {"info": data})
            except Exception:
                LOG.exception("Failed to mark invoice past_due for %s", invnum)
    elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        stripe_sub = data.get("id")
        # map stripe_sub to our subscription via metadata
        # iterate subscriptions in DB to find matching stripe id
        try:
            query = {"metadata.stripe_subscription_id": stripe_sub} if subs_col is not None else None
            if subs_col is not None:
                doc = await subs_col.find_one(query)
                if doc:
                    sub_id = doc.get("subscription_id")
                    await subs_col.update_one({"subscription_id": sub_id}, {"$set": {"status": data.get("status")}})
            else:
                allsubs = _fs_read_json(_SUBS_FS) or {}
                for sid, d in allsubs.items():
                    md = d.get("metadata", {})
                    if md.get("stripe_subscription_id") == stripe_sub:
                        d["status"] = data.get("status")
                _fs_write_json(_SUBS_FS, allsubs)
            await _audit(None, "stripe_subscription_event", stripe_sub, {"event": event_type})
        except Exception:
            LOG.exception("Error handling subscription event")
    # respond with 200
    return {"ok": True}

# ----------------------
# Coupons / promo codes
# ----------------------
async def _save_coupon(coupon: Coupon):
    data = coupon.dict()
    if coupon_col is not None:
        await coupon_col.update_one({"coupon_id": coupon.coupon_id}, {"$set": data}, upsert=True)
    else:
        allc = _fs_read_json(_COUPONS_FS) or {}
        allc[coupon.coupon_id] = data
        _fs_write_json(_COUPONS_FS, allc)

async def _get_coupon(coupon_id: str) -> Optional[Coupon]:
    if coupon_col is not None:
        doc = await coupon_col.find_one({"coupon_id": coupon_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return Coupon(**doc)
    allc = _fs_read_json(_COUPONS_FS) or {}
    d = allc.get(coupon_id)
    if not d:
        return None
    return Coupon(**d)

@router.post("/coupons", dependencies=[Depends(require_role(Role.OPERATOR))])
async def create_coupon(payload: CouponCreate, user = Depends(get_current_user)):
    now = _now_iso()
    coupon = Coupon(**payload.dict(), created_at=now, redemptions=0)
    await _save_coupon(coupon)
    await _audit(user, "create_coupon", coupon.coupon_id, {"payload": payload.dict()})
    return coupon

@router.get("/coupons/{coupon_id}", dependencies=[Depends(require_role(Role.VIEWER))])
async def get_coupon(coupon_id: str):
    c = await _get_coupon(coupon_id)
    if not c:
        raise HTTPException(status_code=404, detail="Coupon not found")
    return c

@router.post("/coupons/{coupon_id}/redeem", dependencies=[Depends(require_role(Role.OPERATOR))])
async def redeem_coupon(coupon_id: str, customer_id: str = Body(...), user = Depends(get_current_user)):
    coupon = await _get_coupon(coupon_id)
    if not coupon:
        raise HTTPException(status_code=404, detail="Coupon not found")
    if coupon.expires_at and coupon.expires_at < _now_iso():
        raise HTTPException(status_code=400, detail="Coupon expired")
    if coupon.max_redemptions and coupon.redemptions >= (coupon.max_redemptions or 0):
        raise HTTPException(status_code=400, detail="Coupon redemptions exceeded")
    # record redemption (for simple system we increment redemptions)
    coupon.redemptions = (coupon.redemptions or 0) + 1
    await _save_coupon(coupon)
    await _audit(user, "redeem_coupon", coupon_id, {"customer": customer_id})
    return {"ok": True, "coupon_id": coupon_id, "applied_to_customer": customer_id}

# ----------------------
# Billing reports & exports
# ----------------------
@router.get("/reports/invoices", dependencies=[Depends(require_role(Role.VIEWER))])
async def invoices_report(since_days: int = Query(30), format: Optional[str] = Query("json")):
    since_ts = (datetime.datetime.utcnow() - datetime.timedelta(days=since_days)).isoformat() + "Z"
    results = []
    if invoice_col is not None:
        docs = await invoice_col.find({"created_at": {"$gte": since_ts}}).to_list(length=1000)
        for d in docs:
            d.pop("_id", None)
            results.append(d)
    else:
        allinv = _fs_read_json(_INVOICES_FS) or {}
        for inv in allinv.values():
            if inv.get("created_at", "") >= since_ts:
                results.append(inv)
    if format == "csv":
        # stream CSV
        def iter_csv():
            header = ["invoice_id", "customer_id", "created_at", "total_cents", "paid"]
            yield ",".join(header) + "\n"
            for row in results:
                yield ",".join([str(row.get(h, "")) for h in header]) + "\n"
        return StreamingResponse(iter_csv(), media_type="text/csv")
    return results

@router.get("/reports/usage", dependencies=[Depends(require_role(Role.VIEWER))])
async def usage_report(metric: Optional[str] = None, since_days: int = Query(30)):
    aggregates = await aggregate_usage(customer_id=None, metric=metric, since_days=since_days)
    return aggregates

# ----------------------
# Dunning & retry policy (skeleton)
# ----------------------
_DUNNING_POLICIES = {
    "default": {"retries": 3, "intervals": [24, 48, 96], "escalation": {"email": True, "suspend_after": 7}}
}

async def _attempt_payment_for_invoice(invoice_id: str) -> bool:
    """
    Attempt to charge the customer's saved payment method (Stripe integration).
    Returns True if paid else False. This is a simplified flow.
    """
    # locate invoice
    inv = None
    if invoice_col is not None:
        doc = await invoice_col.find_one({"invoice_id": invoice_id})
        if doc:
            inv = Invoice(**{k: v for k, v in doc.items() if k != "_id"})
    else:
        allinv = _fs_read_json(_INVOICES_FS) or {}
        d = allinv.get(invoice_id)
        if d:
            inv = Invoice(**d)
    if not inv:
        LOG.warning("Invoice not found for payment attempt: %s", invoice_id)
        return False
    # find customer stripe id
    cust = await _get_customer_store(inv.customer_id)
    if cust and cust.metadata.get("stripe_customer_id") and _HAS_STRIPE:
        stripe_cust = cust.metadata.get("stripe_customer_id")
        try:
            # create PaymentIntent for invoice amount
            pi = stripe.PaymentIntent.create(amount=inv.total_cents, currency=inv.currency, customer=stripe_cust, metadata={"prioritymax_invoice_id": inv.invoice_id})
            # confirm (if automatic confirmation supported)
            # In a real flow, we should handle 3DS and asynchronous flows; here we attempt to confirm automatically
            stripe.PaymentIntent.confirm(pi["id"])
            # mark invoice paid
            if invoice_col is not None:
                await invoice_col.update_one({"invoice_id": inv.invoice_id}, {"$set": {"paid": True, "paid_at": _now_iso()}})
            else:
                allinv = _fs_read_json(_INVOICES_FS) or {}
                v = allinv.get(inv.invoice_id)
                if v:
                    v["paid"] = True
                    v["paid_at"] = _now_iso()
                    _fs_write_json(_INVOICES_FS, allinv)
            await _audit(None, "invoice_paid_auto", inv.invoice_id, {"method": "stripe"})
            return True
        except Exception:
            LOG.exception("Automatic payment attempt failed for %s", inv.invoice_id)
            return False
    return False

async def _dunning_worker_loop(interval_hours: int = 24):
    """
    Periodic worker to find unpaid invoices and attempt retries according to dunning policy.
    This function should be scheduled by your background runner (e.g., Kubernetes CronJob or FastAPI startup loop).
    """
    while True:
        try:
            LOG.info("Dunning loop waking up")
            # find unpaid invoices older than 24h (simple)
            cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=24)).isoformat() + "Z"
            unpaid = []
            if invoice_col is not None:
                docs = await invoice_col.find({"paid": False, "created_at": {"$lte": cutoff}}).to_list(length=1000)
                for d in docs:
                    d.pop("_id", None)
                    unpaid.append(d)
            else:
                allinv = _fs_read_json(_INVOICES_FS) or {}
                for inv in allinv.values():
                    if not inv.get("paid") and inv.get("created_at", "") <= cutoff:
                        unpaid.append(inv)
            for inv in unpaid:
                invoice_id = inv.get("invoice_id")
                success = await _attempt_payment_for_invoice(invoice_id)
                if not success:
                    # schedule retry or escalate
                    await _audit(None, "dunning_retry_failed", invoice_id, {"info": inv})
            LOG.info("Dunning loop sleeping for %d hours", interval_hours)
        except Exception:
            LOG.exception("Dunning worker error")
        await asyncio.sleep(interval_hours * 3600)

# ----------------------
# Health & diagnostics
# ----------------------
@router.get("/health")
async def billing_health():
    return {
        "ok": True,
        "time": _now_iso(),
        "mongo": bool(plans_col),
        "stripe": _HAS_STRIPE and bool(STRIPE_API_KEY),
        "s3": _HAS_BOTO3 and bool(S3_BUCKET),
    }

@router.get("/status", dependencies=[Depends(require_role(Role.VIEWER))])
async def billing_status():
    # simple stats
    plans = await _list_plans_from_store()
    customers_count = 0
    if customer_col is not None:
        customers_count = await customer_col.count_documents({})
    else:
        allc = _fs_read_json(_CUSTOMERS_FS) or {}
        customers_count = len(allc)
    return {
        "plans_count": len(plans),
        "customers_count": customers_count,
        "stripe_enabled": _HAS_STRIPE and bool(STRIPE_API_KEY),
        "s3_enabled": _HAS_BOTO3 and bool(S3_BUCKET),
    }

# ----------------------
# Startup hooks (optional)
# ----------------------
_BILLING_BG_TASK: Optional[asyncio.Task] = None

def start_billing_background_tasks(loop: Optional[asyncio.AbstractEventLoop] = None):
    """
    Start optional background tasks (dunning). Call this from FastAPI startup if desired.
    """
    global _BILLING_BG_TASK
    if _BILLING_BG_TASK and not _BILLING_BG_TASK.done():
        return
    loop = loop or asyncio.get_event_loop()
    _BILLING_BG_TASK = loop.create_task(_dunning_worker_loop(interval_hours=24))
    LOG.info("Billing background tasks started")

def stop_billing_background_tasks():
    global _BILLING_BG_TASK
    if _BILLING_BG_TASK:
        _BILLING_BG_TASK.cancel()
        _BILLING_BG_TASK = None
        LOG.info("Billing background tasks stopped")

# ----------------------
# Admin endpoints: invoice estimate, charge, export
# ----------------------
@router.post("/estimate-invoice", dependencies=[Depends(require_role(Role.OPERATOR))])
async def estimate_invoice(customer_id: str = Body(...), subscription_id: Optional[str] = Body(None), usage_window_days: int = Body(30)):
    """
    Estimate an invoice for a customer by aggregating usage & subscription cost.
    """
    cust = await _get_customer_store(customer_id)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    # sum usage for customer
    agg = await aggregate_usage(customer_id=customer_id, metric=None, since_days=usage_window_days)
    usage_total = sum(agg.values()) if isinstance(agg, dict) else 0
    # find subscription plan price
    plan_price = 0.0
    if subscription_id:
        sub = await _get_subscription(subscription_id)
        if sub:
            plan = await _get_plan_from_store(sub.plan_id)
            if plan:
                plan_price = plan.price_per_month
    # estimate extra usage cost using overage price (simplified)
    extra_cost = 0.0
    if subscription_id and sub and plan and plan.included_quota:
        overage = max(0, usage_total - plan.included_quota)
        extra_cost = overage * (plan.unit_price_overage or 0.0)
    subtotal = plan_price + extra_cost
    tax = subtotal * (DEFAULT_TAX_PERCENT / 100.0)
    total = subtotal + tax
    return {"subtotal": subtotal, "tax": tax, "total": total, "usage_total": usage_total}

@router.post("/charge-invoice/{invoice_id}", dependencies=[Depends(require_role(Role.OPERATOR))])
async def charge_invoice(invoice_id: str, user = Depends(get_current_user)):
    """
    Attempt to charge invoice via configured payment method (Stripe).
    """
    success = await _attempt_payment_for_invoice(invoice_id)
    await _audit(user, "charge_invoice", invoice_id, {"success": success})
    return {"invoice_id": invoice_id, "charged": success}

@router.get("/export/invoices/csv", dependencies=[Depends(require_role(Role.VIEWER))])
async def export_invoices_csv(since_days: int = Query(30)):
    """
    Export invoices as CSV for accounting.
    """
    since_ts = (datetime.datetime.utcnow() - datetime.timedelta(days=since_days)).isoformat() + "Z"
    rows = []
    if invoice_col is not None:
        docs = await invoice_col.find({"created_at": {"$gte": since_ts}}).to_list(length=1000)
        for d in docs:
            d.pop("_id", None)
            rows.append(d)
    else:
        allinv = _fs_read_json(_INVOICES_FS) or {}
        for inv in allinv.values():
            if inv.get("created_at", "") >= since_ts:
                rows.append(inv)
    def iter_csv():
        writer = csv.writer(sys.stdout)
        header = ["invoice_id", "customer_id", "created_at", "total_cents", "currency", "paid"]
        yield ",".join(header) + "\n"
        for r in rows:
            yield ",".join([str(r.get(c, "")) for c in header]) + "\n"
    return StreamingResponse(iter_csv(), media_type="text/csv")

# ----------------------
# Webhook verification endpoint for external integrations (e.g., accounting)
# ----------------------
@router.post("/webhooks/external")
async def external_webhook(req: Request):
    """
    Basic webhook endpoint that accepts events from external systems, verifies an 'X-Signature' header
    if provided (HMAC with shared secret), and stores events.
    Use env var EXTERNAL_WEBHOOK_SECRET.
    """
    secret = os.getenv("EXTERNAL_WEBHOOK_SECRET", None)
    body = await req.body()
    sig = req.headers.get("X-Signature", "")
    if secret:
        computed = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
        if not hmac.compare_digest(computed, sig):
            LOG.warning("Invalid external webhook signature")
            raise HTTPException(status_code=400, detail="Invalid signature")
    try:
        ev = json.loads(body.decode("utf-8"))
    except Exception:
        ev = {"raw": body.decode("utf-8", errors="ignore")}
    # store event (append to file)
    p = BILLING_META_DIR / "external_webhooks.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"received_at": _now_iso(), "event": ev}) + "\n")
    await _audit(None, "external_webhook", "external", {"event": ev})
    return {"ok": True}

# ----------------------
# End of file
# ----------------------

Here’s a complete README section you can paste into your project’s main README.md under the heading:

🧩 Configuration Control Plane (api/config.py)

It covers:

Overview

Setup

Environment Variables

Integration with FastAPI

Usage Examples

Security Practices

Optional Extensions

🧩 PriorityMax Configuration Control Plane (Phase-3)
Overview

backend/app/api/config.py implements a production-grade configuration control service for PriorityMax Cloud.
It provides dynamic runtime configuration management, feature-flagging, secrets encryption, versioning, rollback, and WebSocket-based live broadcasting to all services (autoscaler, RL agent, chaos module, etc.).

📦 Core Features
Category	Description
Config CRUD	Create, update, list, delete configuration items with version tracking.
Secrets Management	AES-256/Fernet encryption with key rotation and masked retrieval.
Feature Flags	Enable/disable features (chaos_mode, autoscaler_loop, etc.) globally or per tenant.
Tenant Overrides	Organization-scoped (org:<id>) configuration overrides with fallback to global.
Versioning / Rollback	Automatic version snapshot on every change; rollback via API.
Live Reload	WebSocket /ws/config broadcasts real-time updates to connected workers.
Backups	Export configs to JSON; optional upload to S3.
Prometheus Metrics	/config/metrics exposes live counters for monitoring.
Health / Diagnostics	/config/health and /config/status endpoints.
Slack Alerts	Optional integration for system notifications.
⚙️ Environment Variables

Create a .env file at your project root or define these in your environment:

# MongoDB connection (optional; falls back to local JSON store if omitted)
MONGO_URL=mongodb://localhost:27017/prioritymax

# Collection names
CONFIG_COLLECTION=config_store
CONFIG_VERSIONS_COLLECTION=config_versions

# Directory for local JSON fallback
CONFIG_META_DIR=backend/app/config_meta

# Fernet key management (32-byte base64 URL-safe key)
FERNET_KEY_PATH=backend/secrets/fernet.key
# or set directly:
# FERNET_KEY=your_base64_key_here

# Slack integration (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXXX/YYYY/ZZZZ

# S3 backup target (optional)
S3_BUCKET=prioritymax-backups
S3_PREFIX=config/

# Logging
PRIORITYMAX_CONFIG_LOG_LEVEL=INFO

🧩 Integration with FastAPI

Register the config API router in your main backend entrypoint (main.py):

# backend/app/main.py
from fastapi import FastAPI
from app.api import config  # import your config module

app = FastAPI(title="PriorityMax API")

# include router
app.include_router(config.router)

# optional: start background loops manually if not using router events
from app.api.config import start_config_background_loop, start_prometheus_metrics_loop
@app.on_event("startup")
async def startup_tasks():
    start_config_background_loop(app)
    start_prometheus_metrics_loop(app)


Once registered, the Config Control Plane is available under /config/*.

🌐 Example Usage
1️⃣ Create or update a config item
POST /config
{
  "key": "autoscaler.max_workers",
  "value": 25,
  "is_secret": false,
  "comment": "Increased max worker limit"
}

2️⃣ Create a secret
POST /config/secrets/redis_password
{
  "secret_value": "s3cr3tpass"
}

3️⃣ Set a feature flag
PUT /config/features/chaos_mode
{
  "name": "chaos_mode",
  "enabled": true,
  "description": "Enable chaos testing globally"
}

4️⃣ Rollback configuration
POST /config/autoscaler.max_workers/rollback/3

5️⃣ Subscribe to live updates
WebSocket → ws://localhost:8000/config/ws/config

6️⃣ Trigger runtime reload
POST /config/reload

🔐 Security & Access Control
Role	Permissions
Viewer	Read-only: list, get, metrics, health, WS subscribe.
Operator	CRUD on configs, feature flags, tenant overrides, trigger reload.
Admin	Full control including secrets rotation, key management, rollback, cleanup.

✅ Best Practices

Use a dedicated service account for automation with minimal role.

Never expose decrypted secrets via API.

Mount Fernet key via K8s Secret or Vault in production.

Enable HTTPS and authentication middleware for all /config/* endpoints.

Restrict /config/rotate_fernet_key and /config/cleanup to Admin only.

📊 Prometheus Metrics
Metric	Description
prioritymax_config_changes_total	Total config modifications
prioritymax_config_rollbacks_total	Rollbacks performed
prioritymax_config_active_keys	Number of active keys in store

Expose snapshot:

GET /config/metrics

☁️ Example Kubernetes Integration

Mount config service as a sidecar or microservice in your deployment:

containers:
  - name: prioritymax-config
    image: yourrepo/prioritymax-backend:latest
    envFrom:
      - configMapRef:
          name: prioritymax-env
      - secretRef:
          name: prioritymax-secrets
    ports:
      - containerPort: 8000


Workers, autoscaler, and RL agent components can subscribe to /ws/config for live updates.

🧠 Developer Notes

Default fallback storage: backend/app/config_meta/<namespace>/*.json

Every change → audit log written to backend/logs/config_audit.jsonl

Background tasks:

config_background_loop → verifies configuration every 5 min

prometheus_metrics_loop → updates metrics every 60 s

Rotation and cleanup utilities:

/config/rotate_fernet_key

/config/cleanup

✅ Quick Verification

Run locally:

uvicorn app.main:app --reload


Then test endpoints:

GET http://127.0.0.1:8000/config/health
GET http://127.0.0.1:8000/config/status


Expected output:

{
  "ok": true,
  "uptime_seconds": 42,
  "config_items_cached": 10,
  "ws_connections": 0
}

📚 Optional Extensions
Module	Purpose
app/api/admin.py	Admin control surface for model mgmt & training orchestration
app/api/autoscaler.py	Dynamic autoscaling logic integrating predictor & RL agent
app/api/config.py	Config control plane (this module)
app/ml/real_env.py	Live environment adapter for RL agent
app/services/worker_manager.py	Worker supervision & self-healing logic


It’s formatted like your other module docs (self-contained, production-ready) and covers:

Module purpose & architecture

Environment variables

Setup & registration

Example workflow spec JSON

Example executor configurations

WebSocket + tracing setup

Common admin & observability commands

🧩 PriorityMax Workflows API
Overview

api/workflows.py is the enterprise-grade workflow orchestration layer of PriorityMax Phase-3.
It turns the task scheduler into a full workflow engine — similar to Airflow, Temporal, or Camunda — supporting DAG-based pipelines, pluggable executors, visual workflow design, observability, and live execution control.

This API is designed to power:

Workflow creation, versioning, import/export

Visual workflow design with node/edge previews

DAG execution engine with AI-ready integration

SLA enforcement, retry, and monitoring

WebSocket live events & Jaeger tracing

🚀 FastAPI Integration

In your app/main.py (FastAPI entrypoint), simply include the router:

from fastapi import FastAPI
from app.api import workflows

app = FastAPI(title="PriorityMax Cloud")

# Register all routers
app.include_router(workflows.router)


This automatically registers:

/workflows/*
/workflows/ws/workflows
/workflows/run/*
/workflows/observability/*


No special middleware required.

⚙️ Environment Variables
Variable	Default	Description
MONGO_URL	None	MongoDB connection string for metadata & runs
REDIS_URL	redis://localhost:6379/0	Redis instance for queue/pubsub
WORKFLOWS_META_DIR	backend/app/workflows_meta	Local fallback store
WORKFLOW_RETENTION_DAYS	30	Cleanup threshold for old runs
WORKFLOW_MAX_CONCURRENT_RUNS	10	Limits simultaneous DAG executions
WORKFLOW_EXECUTOR_DEFAULT	python	Default executor type
WORKFLOW_CONTAINER_MODE	docker	docker or k8s mode for container executor
WORKFLOW_MAINTENANCE_INTERVAL	120	Background cleanup + SLA interval (seconds)
WORKFLOWS_WS_AUTH_TOKEN	(optional)	Token required for WebSocket connections
OTEL_EXPORTER_JAEGER_AGENT_HOST	(optional)	Jaeger host for tracing
OTEL_EXPORTER_JAEGER_AGENT_PORT	6831	Jaeger UDP port
🧠 Core Concepts
Workflow

A Workflow is a DAG of nodes (tasks, switches, timers, joins) connected by edges.
Each workflow is stored as a versioned JSON spec, validated for acyclicity and reference integrity.

Run

A Run is a single execution instance of a Workflow.
Each run maintains:

Status (PENDING → RUNNING → COMPLETED / FAILED)

Run log (.jsonl per run)

Input variables & per-node results

SLA and retry tracking

Node Types
Type	Description
task	Executable unit, linked to an executor (python/http/container/etc.)
parallel	Parallel branch start
join	Merge multiple branches
switch	Conditional routing using expressions
timer	Wait/delay node
subworkflow	Calls another workflow
start / end	DAG boundaries
🧩 Example Workflow Spec
{
  "name": "Order Processing",
  "description": "Demo e-commerce order pipeline",
  "version": 1,
  "nodes": [
    {
      "node_id": "start",
      "type": "start",
      "name": "Start"
    },
    {
      "node_id": "verify",
      "type": "task",
      "name": "Verify Order",
      "template": {
        "executor": "python",
        "config": {"callable": "workflows.tasks:verify_order"},
        "retry_policy": {"max_retries": 2, "backoff_seconds": 3}
      }
    },
    {
      "node_id": "payment",
      "type": "task",
      "name": "Process Payment",
      "template": {
        "executor": "http",
        "config": {
          "url": "https://payment.example.com/api/pay",
          "method": "POST",
          "headers": {"Content-Type": "application/json"}
        },
        "timeout_seconds": 15
      }
    },
    {
      "node_id": "end",
      "type": "end",
      "name": "End"
    }
  ],
  "edges": [
    {"source": "start", "target": "verify"},
    {"source": "verify", "target": "payment"},
    {"source": "payment", "target": "end"}
  ]
}

🔧 Executors
1️⃣ Python Executor

Runs Python functions within the backend process.

# workflows/tasks.py
def verify_order(payload, meta):
    print("Verifying order:", payload)
    return {"verified": True}


Config:

"template": {
  "executor": "python",
  "config": {"callable": "workflows.tasks:verify_order"}
}

2️⃣ HTTP Executor

Calls external REST APIs.

"template": {
  "executor": "http",
  "config": {
    "url": "https://api.my-service.com/process",
    "method": "POST",
    "headers": {"Authorization": "Bearer <token>"}
  }
}

3️⃣ Container Executor

Runs commands inside containers (Docker or K8s job mode).

"template": {
  "executor": "container",
  "config": {
    "image": "ubuntu:latest",
    "command": ["echo", "Hello from container"]
  }
}


Set mode via:

WORKFLOW_CONTAINER_MODE=docker   # or k8s

4️⃣ External Executor

Calls Slack/webhooks/third-party endpoints.

"template": {
  "executor": "external",
  "config": {
    "url": "https://hooks.slack.com/...",
    "payload_template": {
      "text": "Workflow {workflow_id} run {run_id} completed successfully."
    }
  }
}

📡 Real-Time Observability
WebSocket /ws/workflows

Receive events: run_started, node_start, node_end, run_completed, sla_violation, etc.

Optional token auth via ?token=WORKFLOWS_WS_AUTH_TOKEN

Example client:

const ws = new WebSocket("ws://localhost:8000/workflows/ws/workflows?token=abc");
ws.onmessage = e => console.log("Event:", JSON.parse(e.data));

Prometheus Metrics

Endpoint:

GET /workflows/metrics


Metrics include:

prioritymax_workflow_runs_started_total

prioritymax_workflow_runs_failed_total

prioritymax_workflow_runs_active

Jaeger / OpenTelemetry

Initialize tracing:

POST /workflows/observability/otel/init
{
  "service_name": "prioritymax-workflows",
  "jaeger_host": "localhost",
  "jaeger_port": 6831
}


This activates OTEL spans for run lifecycle events.

🧩 Admin & Maintenance
Cleanup old runs
POST /workflows/admin/cleanup

SLA enforcement & background maintenance

Runs automatically every WORKFLOW_MAINTENANCE_INTERVAL seconds.
Violations emit sla_violation events and audit entries.

Check subsystem health
GET /workflows/health

🧰 Developer Utilities
Endpoint	Description
/workflows/templates	Create/list/delete reusable task templates
/workflows/preview	Preview workflow DAG structure
/workflows/validate	Validate workflow spec
/workflows/run/start	Start a new workflow run
/workflows/run/{run_id}	Fetch run details
/workflows/run/{run_id}/logs	Download or stream run logs
/workflows/trace/replay/{run_id}	Replay workflow trace for UI visualization
🧾 Folder Structure (Quick Reference)
backend/
 ├── app/
 │   ├── api/
 │   │   ├── workflows.py      # ← this module
 │   │   └── tasks.py
 │   ├── ml/
 │   ├── utils/
 │   ├── main.py
 │   └── config.py
 ├── logs/
 │   └── workflows_audit.jsonl
 └── app/workflows_meta/
     ├── workflows.json
     ├── versions/
     ├── runs/
     ├── templates.json
     └── reports/

🧪 Quick Test via cURL
# 1. Create workflow
curl -X POST http://localhost:8000/workflows/ \
  -H "Content-Type: application/json" \
  -d @examples/order_workflow.json

# 2. Start a run
curl -X POST http://localhost:8000/workflows/run/start \
  -H "Content-Type: application/json" \
  -d '{"workflow_id":"wf_xxx","input_variables":{"order_id":123}}'

# 3. Stream events
wscat -c ws://localhost:8000/workflows/ws/workflows


✅ Result:
You now have a full-scale orchestration API capable of running dynamic, observable, and distributed workflows across PriorityMax Cloud.

📘 PriorityMax Unified Trainer — CLI & Usage Guide

This guide documents the complete command-line interface and runtime configuration for
backend/app/ml/trainer_full.py, which unifies:

PPO reinforcement learning (single-machine & distributed)

Predictor model training (LightGBM / RandomForest)

Ray RLlib distributed training

Ray Tune / Optuna hyperparameter tuning

Torch.distributed multi-process launcher

Canary gating & model registry promotion

⚙️ General CLI Structure
python3 backend/app/ml/trainer_full.py <command> [options...]


Example:

python3 backend/app/ml/trainer_full.py train_rl --epochs 200 --steps 4096 --device cuda


All logs and checkpoints are written to:

backend/app/ml/models/

🧩 Commands Overview
Command	Description
train_rl	Train PPO RL agent on simulated environment (single-machine).
eval_rl	Evaluate a trained RL checkpoint and compute average reward.
train_predictor	Train LightGBM or RandomForest predictor for queue forecasting.
evaluate_predictor	Evaluate saved predictor model on a CSV dataset.
canary_predictor	Run canary gate check on new predictor model before promotion.
rllib_train	Launch distributed PPO training using Ray RLlib.
tune	Run Ray Tune experiment for large-scale hyperparameter optimization.
optuna_predictor	Run Optuna tuning for LightGBM predictor hyperparameters.
torch_distributed	Launch torch.distributed local multi-process training test.
🧠 Reinforcement Learning (Single-Machine)
▶️ Train PPO RL Agent
python3 backend/app/ml/trainer_full.py train_rl \
    --epochs 300 \
    --steps 2048 \
    --lr 3e-4 \
    --device cuda \
    --checkpoint backend/app/ml/models/rl_agent.pt \
    --wandb --mlflow


✅ Features:

Vectorized environments (make_vec_env)

Periodic checkpointing and evaluation

Logging to W&B and MLflow

Resumes automatically if checkpoint exists

📊 Evaluate RL Agent
python3 backend/app/ml/trainer_full.py eval_rl \
    --checkpoint backend/app/ml/models/rl_agent.pt \
    --episodes 10


Expected output:

{
  "avg_reward": 96.75,
  "std_reward": 4.12,
  "episodes": 10
}

🔮 Predictor Models (LightGBM / sklearn)
🧠 Train Predictor
python3 backend/app/ml/trainer_full.py train_predictor \
    --data datasets/queue_metrics.csv \
    --target queue_next_sec \
    --ckpt backend/app/ml/models/predictor_lgbm.pkl \
    --wandb

🧪 Evaluate Predictor
python3 backend/app/ml/trainer_full.py evaluate_predictor \
    --ckpt backend/app/ml/models/predictor_lgbm.pkl \
    --data datasets/queue_metrics.csv


Output:

{
  "rmse": 0.1435,
  "r2": 0.948
}

🧰 Canary Gating
python3 backend/app/ml/trainer_full.py canary_predictor \
    --ckpt backend/app/ml/models/predictor_candidate.pkl \
    --holdout datasets/holdout.csv \
    --threshold 5.0


If the new model’s RMSE is below the threshold, it “passes” and can be promoted:

{ "passed": true }

☁️ Distributed RLlib Training (Ray)
🚀 Launch Ray PPO
python3 backend/app/ml/trainer_full.py rllib_train \
    --epochs 100 \
    --cpus 8 \
    --gpus 1 \
    --save_dir backend/app/ml/models/rllib_checkpoints


This:

Initializes Ray automatically

Registers the PriorityMaxSimEnv

Runs PPO across workers

Saves checkpoints every few iterations under:

backend/app/ml/models/rllib_checkpoints/


After training, RLlib checkpoints are automatically converted into .pt PyTorch-friendly weights.

🎛️ Ray Tune — Hyperparameter Optimization
🔍 Example Run
python3 backend/app/ml/trainer_full.py tune \
    --samples 10 \
    --scheduler asha \
    --optuna \
    --metric episode_reward_mean


This performs a grid/randomized search over:

lr ∈ [1e−5, 1e−3]

hidden_dim ∈ {64, 128, 256}

clip_range ∈ {0.1, 0.2, 0.3}

entropy_coef ∈ [0.0, 0.05]

and reports results to Ray Dashboard (if enabled).
Results are saved in:

backend/app/ml/models/tune/

⚗️ Optuna — Predictor Tuning
🔬 Example
python3 backend/app/ml/trainer_full.py optuna_predictor \
    --data datasets/queue_metrics.csv \
    --target queue_next_sec \
    --trials 50


This runs 50 trials to minimize RMSE by adjusting LightGBM parameters:

num_leaves

learning_rate

feature_fraction

bagging_fraction

min_data_in_leaf

After completion, it prints the best parameters and their score.

🧩 Torch.distributed Local Launcher

Test multi-GPU or multi-process distributed training on local machine:

python3 backend/app/ml/trainer_full.py torch_distributed --world_size 4


Each worker logs its rank, then runs a small job (placeholder example).

⚙️ Environment Variables
Variable	Description	Default
PRIORITYMAX_RL_ENVS	Number of parallel envs for PPO	4
PRIORITYMAX_TRAINER_LOG	Logging level	INFO
RAY_ADDRESS	Ray cluster address (for distributed runs)	None
RAY_LOCAL_MODE	If true, run Ray in-process (for debugging)	false
WANDB_API_KEY	Weights & Biases auth token	(required for W&B logging)
MLFLOW_TRACKING_URI	MLflow tracking server URL	(optional)
📦 Checkpoints & Artifacts
Type	Path	Notes
RL agent	backend/app/ml/models/rl_agent.pt	Updated every few epochs
Predictor	backend/app/ml/models/predictor_lgbm.pkl	Pickled model
RLlib	backend/app/ml/models/rllib_checkpoints/	Contains RLlib format
Tune results	backend/app/ml/models/tune/	Ray Tune logs & trial results
Archive	backend/app/ml/models/archive/	Tarballs for model versioning
💡 Tips & Best Practices

✅ Always run train_predictor with a clear --target and consistent schema (features must match future production data).
✅ When tuning with Ray or Optuna, set a global random seed for reproducibility:

export PYTHONHASHSEED=42


✅ Use RAY_LOCAL_MODE=true for debugging without a full Ray cluster.
✅ Use small --samples and --trials values first to validate setup.
✅ For multi-GPU nodes, prefer RLlib (it manages GPUs automatically).
✅ Logs are written both to stdout and W&B/MLflow if enabled.

📁 Example End-to-End Workflow
# 1️⃣ Train RL model locally
python3 backend/app/ml/trainer_full.py train_rl --epochs 100 --steps 2048

# 2️⃣ Evaluate and archive it
python3 backend/app/ml/trainer_full.py eval_rl --checkpoint backend/app/ml/models/rl_agent.pt
tar czf backend/app/ml/models/rl_agent_archive.tar.gz backend/app/ml/models/rl_agent.pt

# 3️⃣ Train predictor
python3 backend/app/ml/trainer_full.py train_predictor --data datasets/queue_metrics.csv

# 4️⃣ Run canary gate on holdout
python3 backend/app/ml/trainer_full.py canary_predictor \
    --ckpt backend/app/ml/models/predictor_lgbm.pkl \
    --holdout datasets/holdout.csv

# 5️⃣ Run distributed RLlib PPO
python3 backend/app/ml/trainer_full.py rllib_train --epochs 50 --cpus 8

# 6️⃣ Launch hyperparameter tuning
python3 backend/app/ml/trainer_full.py tune --samples 5 --scheduler asha --optuna

🧾 Expected Directory Layout After Training
backend/app/ml/
├── models/
│   ├── rl_agent.pt
│   ├── predictor_lgbm.pkl
│   ├── rllib_checkpoints/
│   ├── tune/
│   └── archive/
├── trainer_full.py
└── ...

✅ Quick Verification

Run:

python3 backend/app/ml/trainer_full.py --help


to confirm all commands are registered.

You’ll see:

Commands:
  train_rl            Train single-machine PPO RL agent
  eval_rl             Evaluate PPO checkpoint
  train_predictor     Train LightGBM/RandomForest predictor
  evaluate_predictor  Evaluate predictor checkpoint
  canary_predictor    Run canary gate for predictor
  rllib_train         Run distributed PPO training via RLlib
  tune                Run Ray Tune experiment for PPO hyperparameters
  optuna_predictor    Run Optuna tuning for LightGBM predictor
  torch_distributed   Launch torch.distributed multi-process training (demo)

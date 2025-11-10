# backend/app/ml/rl_agent_sandbox.py
"""
PriorityMax RL Agent - Sandbox
------------------------------

Purpose:
  - Lightweight, developer-friendly sandbox runtime for RL agents.
  - Safe: interacts only with SimulatedRealEnv (never calls live infra).
  - Useful for A/B testing, canary evaluation, debugging, replaying episodes,
    and local experimentation before promoting a model to `rl_agent_prod.py`.

Features:
  - Loads PPO-style checkpoints (rl_agent.pt) and runs inference.
  - Hot-reload model support (watch file changes).
  - Runs N parallel sandbox agents (threaded) against SimulatedRealEnv instances.
  - FastAPI debug/control server (start/stop, status, traces, re-run eval).
  - Optional MLflow / W&B logging.
  - Prometheus metrics export (optional).
  - Trace recording (save to JSON), replay mode, diff mode (compare two models).
  - Heuristic fallback and safety wrapper (guaranteed sandbox).
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import logging
import pathlib
import threading
import queue
import signal
import argparse
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project app to path for local imports
ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import SimulatedRealEnv from your real_env module
try:
    from ml.real_env import SimulatedRealEnv, EnvConfig, make_env, make_vec_env
except Exception:
    # Provide a minimal fallback to avoid import errors in some environments
    raise

# Optional observability libraries
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

try:
    from prometheus_client import start_http_server, Gauge, Histogram, Counter
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

# Optional lightweight web UI
try:
    from fastapi import FastAPI, APIRouter
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _HAS_FASTAPI = True
except Exception:
    _HAS_FASTAPI = False

# Logging
LOG = logging.getLogger("prioritymax.rl.sandbox")
LOG.setLevel(os.getenv("PRIORITYMAX_SANDBOX_LOG", "INFO"))
if not LOG.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(ch)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # backend/
MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL = MODELS_DIR / "rl_agent.pt"

# -----------------------
# Config dataclass
# -----------------------
@dataclass
class SandboxConfig:
    model_path: str = str(DEFAULT_MODEL)
    mode: str = "eval"  # eval|train|replay
    num_agents: int = 1
    env_seed: int = 1234
    steps_per_episode: int = 500
    episodes: int = 20
    loop_interval: float = 1.0  # seconds between agent ticks
    hot_reload: bool = False
    save_traces_dir: Optional[str] = None
    log_to_wandb: bool = False
    log_to_mlflow: bool = False
    prom_port: int = 9300
    api_port: int = 8008
    device: str = "cpu"  # or "cuda"
    deterministic: bool = True
    debug: bool = False

# -----------------------
# ActorCritic (same as training/inference)
# -----------------------
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, obs: torch.Tensor):
        logits = self.actor(obs)
        mu = torch.tanh(logits)
        std = torch.ones_like(mu) * 0.2
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp

    def value(self, obs: torch.Tensor):
        return self.critic(obs)

# -----------------------
# SandboxAgent: runs a single agent loop against one SimulatedRealEnv
# -----------------------
class SandboxAgent(threading.Thread):
    def __init__(self, agent_id: int, cfg: SandboxConfig, env_cfg: EnvConfig, model_loader: Callable[[], nn.Module], trace_queue: queue.Queue):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.model_loader = model_loader
        self.trace_queue = trace_queue
        self.device = torch.device(cfg.device)
        self.stop_event = threading.Event()
        self.model_ts = None
        self.model = None
        self._load_model_safe()
        self.env = SimulatedRealEnv(env_cfg)
        self.metrics = {"actions": 0, "errors": 0, "rewards": []}

    def _load_model_safe(self):
        """Load model if exists; keep None allowed (random policy fallback)."""
        path = pathlib.Path(self.cfg.model_path)
        if path.exists():
            try:
                ckpt = torch.load(str(path), map_location=self.device)
                # assume obs_dim=8, act_dim=3 as before
                model = PPOActorCritic(8, 3)
                model.load_state_dict(ckpt.get("model", ckpt))
                model.to(self.device)
                model.eval()
                self.model = model
                self.model_ts = path.stat().st_mtime
                LOG.info(f"[Agent {self.agent_id}] Model loaded from {path}")
            except Exception:
                LOG.exception(f"[Agent {self.agent_id}] Failed to load model; using fallback")
                self.model = None
        else:
            LOG.info(f"[Agent {self.agent_id}] No model found at {path}; using fallback policy")
            self.model = None

    def check_hot_reload(self):
        if not self.cfg.hot_reload:
            return
        path = pathlib.Path(self.cfg.model_path)
        if path.exists():
            mtime = path.stat().st_mtime
            if self.model_ts is None or mtime > (self.model_ts + 1e-6):
                LOG.info(f"[Agent {self.agent_id}] Hot-reloading model (mtime {mtime})")
                self._load_model_safe()

    def heuristic_policy(self, obs: np.ndarray) -> np.ndarray:
        """Simple safe sandbox heuristic - deterministic-ish"""
        queue_len, consumers, avg_lat, p95, success, arrival, cpu, mem = obs
        delta = 0
        throttle = 0.0
        priority = 0
        if avg_lat > 500 and queue_len > consumers * 2:
            delta = min(5, int(math.ceil(queue_len / max(1, consumers)) - 1))
        elif queue_len < max(2, consumers // 2):
            delta = -1
        throttle = min(1.0, max(0.0, (avg_lat - 200) / 1000.0))
        return np.array([delta, throttle, priority], dtype=np.float32)

    def model_policy(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return self.heuristic_policy(obs)
        try:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logp = self.model.act(obs_t)
            return action.cpu().numpy().astype(np.float32)
        except Exception:
            LOG.exception(f"[Agent {self.agent_id}] Model inference failed; falling back")
            self.metrics["errors"] += 1
            return self.heuristic_policy(obs)

    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        obs = self.env.reset(seed=self.env_cfg.seed + episode_idx)
        trace = {"agent_id": self.agent_id, "episode": episode_idx, "steps": []}
        total_reward = 0.0
        for t in range(self.cfg.steps_per_episode):
            if self.stop_event.is_set():
                break
            # optional hot-reload
            self.check_hot_reload()
            # get action
            action = self.model_policy(obs)
            # apply action in sim env (safe)
            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.metrics["actions"] += 1
            step_rec = {"t": t, "obs": obs.tolist(), "action": action.tolist(), "reward": float(reward), "done": bool(done), "info": info}
            trace["steps"].append(step_rec)
            obs = next_obs
            if done:
                break
            time.sleep(self.cfg.loop_interval)  # slow down for human-observable runs
        self.metrics["rewards"].append(total_reward)
        trace["total_reward"] = total_reward
        trace["metrics"] = self.metrics.copy()
        return trace

    def run(self):
        LOG.info(f"[Agent {self.agent_id}] Starting loop; mode={self.cfg.mode}")
        try:
            for e in range(self.cfg.episodes):
                if self.stop_event.is_set():
                    break
                trace = self.run_episode(e)
                # put trace on queue for external consumer or to save
                try:
                    self.trace_queue.put_nowait(trace)
                except queue.Full:
                    LOG.warning(f"[Agent {self.agent_id}] Trace queue full; dropping trace")
                # short rest between episodes
                time.sleep(0.2)
        except Exception:
            LOG.exception(f"[Agent {self.agent_id}] Error in run loop")
        LOG.info(f"[Agent {self.agent_id}] Finished run; actions={self.metrics['actions']} errors={self.metrics['errors']}")

    def stop(self):
        LOG.info(f"[Agent {self.agent_id}] Stop requested")
        self.stop_event.set()

# -----------------------
# SandboxRunner: manage multiple agents, traces, evaluation
# -----------------------
class SandboxRunner:
    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        self.env_cfg = EnvConfig(mode="sim", seed=cfg.env_seed)
        self.trace_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.agents: List[SandboxAgent] = []
        self._trace_store: List[Dict[str, Any]] = []
        self._running = False
        # Prometheus metrics
        if _HAS_PROM:
            try:
                self.prom_actions = Counter("prioritymax_sandbox_actions_total", "sandbox actions emitted")
                self.prom_episodes = Counter("prioritymax_sandbox_episodes_total", "sandbox episodes completed")
                start_http_server(self.cfg.prom_port)
                LOG.info(f"[SandboxRunner] Prometheus served at {self.cfg.prom_port}")
            except Exception:
                LOG.exception("[SandboxRunner] Prometheus init failed")

    def _model_loader(self):
        """Callable wrapper to lazy-load model; agents will call this."""
        path = pathlib.Path(self.cfg.model_path)
        if not path.exists():
            return None
        ckpt = torch.load(str(path), map_location=self.cfg.device)
        model = PPOActorCritic(8, 3)
        model.load_state_dict(ckpt.get("model", ckpt))
        model.eval()
        return model

    def start(self, num_agents: Optional[int] = None):
        if self._running:
            LOG.warning("[SandboxRunner] Already running")
            return
        na = num_agents or self.cfg.num_agents
        self.agents = []
        for i in range(na):
            agent = SandboxAgent(agent_id=i, cfg=self.cfg, env_cfg=self.env_cfg, model_loader=self._model_loader, trace_queue=self.trace_queue)
            self.agents.append(agent)
            agent.start()
        self._running = True
        # start trace consumer thread
        self._consumer_thread = threading.Thread(target=self._trace_consumer_loop, daemon=True)
        self._consumer_thread.start()
        LOG.info(f"[SandboxRunner] Started {na} agents")

    def stop(self):
        if not self._running:
            return
        for a in self.agents:
            a.stop()
        # join threads (with timeout)
        for a in self.agents:
            a.join(timeout=5.0)
        self._running = False
        LOG.info("[SandboxRunner] Stopped all agents")

    def _trace_consumer_loop(self):
        while self._running or not self.trace_queue.empty():
            try:
                trace = self.trace_queue.get(timeout=1.0)
            except Exception:
                continue
            # store trace in memory (and optionally disk)
            self._trace_store.append(trace)
            if self.cfg.save_traces_dir:
                # write to disk rotated by agent/episode
                outdir = pathlib.Path(self.cfg.save_traces_dir)
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"trace_agent{trace['agent_id']}_ep{trace['episode']}.json"
                try:
                    fname.write_text(json.dumps(trace, indent=2))
                except Exception:
                    LOG.exception("[SandboxRunner] Failed to persist trace")
            # metrics
            if _HAS_PROM:
                try:
                    self.prom_actions.inc(len(trace.get("steps", [])))
                    self.prom_episodes.inc(1)
                except Exception:
                    pass

    def get_traces(self, last_n: int = 10) -> List[Dict[str, Any]]:
        return list(self._trace_store[-last_n:])

    def clear_traces(self):
        self._trace_store.clear()

    def evaluate(self, episodes: int = 10) -> Dict[str, Any]:
        """Run a synchronous evaluation across a single env using the model or heuristic."""
        env = SimulatedRealEnv(self.env_cfg)
        # load model once for eval
        model = None
        path = pathlib.Path(self.cfg.model_path)
        if path.exists():
            try:
                ckpt = torch.load(str(path), map_location=self.cfg.device)
                m = PPOActorCritic(8, 3)
                m.load_state_dict(ckpt.get("model", ckpt))
                m.eval()
                model = m
            except Exception:
                LOG.exception("[SandboxRunner] Failed to load model for evaluation")
                model = None
        rewards = []
        for ep in range(episodes):
            obs = env.reset(seed=self.env_cfg.seed + ep)
            tot = 0.0
            for t in range(self.cfg.steps_per_episode):
                if model is not None:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        act, _ = model.act(obs_t)
                    action = act.cpu().numpy().astype(np.float32)
                else:
                    # heuristic
                    action = SandboxAgent(0, self.cfg, self.env_cfg, self._model_loader, self.trace_queue).heuristic_policy(obs)
                obs, rew, done, info = env.step(action)
                tot += rew
                if done:
                    break
            rewards.append(tot)
        res = {"avg_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)), "episodes": episodes}
        LOG.info(f"[SandboxRunner] Evaluation: {res}")
        return res

# -----------------------
# FastAPI Control Server (optional)
# -----------------------
class SandboxAPI:
    def __init__(self, runner: SandboxRunner, cfg: SandboxConfig):
        if not _HAS_FASTAPI:
            raise RuntimeError("FastAPI not installed")
        self.runner = runner
        self.cfg = cfg
        self.app = FastAPI(title="PriorityMax Sandbox API")
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        self.router = APIRouter(prefix="/sandbox")
        self._register_routes()
        self.app.include_router(self.router)

    def _register_routes(self):
        @self.router.get("/status")
        def status():
            return JSONResponse({"running": self.runner._running, "num_agents": len(self.runner.agents), "traces_stored": len(self.runner._trace_store)})

        @self.router.post("/start")
        def start(num_agents: int = 1):
            if self.runner._running:
                return JSONResponse({"ok": False, "message": "already_running"})
            self.runner.start(num_agents=num_agents)
            return JSONResponse({"ok": True})

        @self.router.post("/stop")
        def stop():
            if not self.runner._running:
                return JSONResponse({"ok": False, "message": "not_running"})
            self.runner.stop()
            return JSONResponse({"ok": True})

        @self.router.get("/traces")
        def traces(last_n: int = 10):
            return JSONResponse(self.runner.get_traces(last_n))

        @self.router.post("/evaluate")
        def evaluate(episodes: int = 10):
            res = self.runner.evaluate(episodes=episodes)
            return JSONResponse(res)

        @self.router.post("/clear_traces")
        def clear_traces():
            self.runner.clear_traces()
            return JSONResponse({"ok": True})

    def run(self, host="0.0.0.0", port=None):
        port = port or self.cfg.api_port
        LOG.info(f"[SandboxAPI] Starting FastAPI server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# -----------------------
# CLI & helpers
# -----------------------
def build_parser():
    p = argparse.ArgumentParser(prog="prioritymax-rl-sandbox")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to rl_agent.pt checkpoint")
    p.add_argument("--mode", choices=["eval", "replay", "train"], default="eval")
    p.add_argument("--num-agents", type=int, default=1)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--loop-interval", type=float, default=0.01)
    p.add_argument("--hot-reload", action="store_true")
    p.add_argument("--save-traces", type=str, default=None)
    p.add_argument("--prom-port", type=int, default=9300)
    p.add_argument("--api-port", type=int, default=8008)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p

def main_cli():
    args = build_parser().parse_args()
    cfg = SandboxConfig(
        model_path=args.model,
        mode=args.mode,
        num_agents=args.num_agents,
        env_seed=1234,
        steps_per_episode=args.steps,
        episodes=args.episodes,
        loop_interval=args.loop_interval,
        hot_reload=args.hot_reload,
        save_traces_dir=args.save_traces,
        log_to_wandb=args.wandb,
        log_to_mlflow=args.mlflow,
        prom_port=args.prom_port,
        api_port=args.api_port,
        device=args.device,
        debug=args.debug
    )

    runner = SandboxRunner(cfg)

    # optional trackers init
    if cfg.log_to_wandb and _HAS_WANDB:
        try:
            wandb.init(project="PriorityMax-Sandbox", config=asdict(cfg))
        except Exception:
            LOG.exception("W&B init failed")
    if cfg.log_to_mlflow and _HAS_MLFLOW:
        try:
            mlflow.set_experiment("PriorityMax-Sandbox")
            mlflow.start_run()
        except Exception:
            LOG.exception("MLflow init failed")

    # start runner
    runner.start(num_agents=cfg.num_agents)

    # optionally run API server in separate thread
    api_server = None
    if _HAS_FASTAPI:
        try:
            api = SandboxAPI(runner, cfg)
            api_server = threading.Thread(target=lambda: api.run(port=cfg.api_port), daemon=True)
            api_server.start()
        except Exception:
            LOG.exception("Failed to start sandbox API server")

    # wait for agents to finish episodes or be interrupted
    def _signal_handler(sig, frame):
        LOG.info("Received signal; stopping sandbox")
        runner.stop()
        if cfg.log_to_mlflow and _HAS_MLFLOW:
            try:
                mlflow.end_run()
            except Exception:
                pass
        if cfg.log_to_wandb and _HAS_WANDB:
            try:
                wandb.finish()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # block until finished
    try:
        while any(a.is_alive() for a in runner.agents):
            time.sleep(0.5)
    except KeyboardInterrupt:
        _signal_handler(None, None)

    # finalization
    LOG.info("All agents completed. Collected traces: %d", len(runner.get_traces(1000)))
    if cfg.save_traces_dir:
        LOG.info("Traces saved to %s", cfg.save_traces_dir)

    # finalize trackers
    if cfg.log_to_mlflow and _HAS_MLFLOW:
        mlflow.end_run()
    if cfg.log_to_wandb and _HAS_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main_cli()

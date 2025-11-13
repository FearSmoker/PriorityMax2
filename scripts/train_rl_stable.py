#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PriorityMax - STABLE RL Training Pipeline
------------------------------------------
✅ FIXED: Stable hyperparameters (epochs=500, steps=2048, lr=3e-4, gamma=0.99)
✅ FIXED: Clipped value loss to prevent explosion
✅ FIXED: Gradient clipping and normalization
✅ FIXED: NaN detection and recovery
✅ MAINTAINS: All complex features and vectorization
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import shutil
import random
import atexit
import signal
import logging
import pathlib
import argparse
import statistics
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple, List, Sequence
from collections import deque, defaultdict
from datetime import datetime

import numpy as np

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    print(f"✅ PyTorch {torch.__version__} detected")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch not found. Install with: pip install torch")
    sys.exit(1)

# Optional dependencies
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False
    print("⚠️  W&B not found (optional)")

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.disable = kwargs.get('disable', False)
            self.n = 0
        
        def __iter__(self):
            for item in self.iterable:
                self.n += 1
                if not self.disable and self.n % 10 == 0:
                    print(f"\r{self.desc}: {self.n}/{self.total}", end='', flush=True)
                yield item
            if not self.disable:
                print()
        
        def update(self, n=1):
            self.n += n
        
        def close(self):
            pass

# Import environment
try:
    from app.ml.real_env import (
        SimulatedRealEnv, 
        EnvConfig, 
        WorkloadType,
        get_observation_space,
        get_action_space
    )
    print("✅ Environment imported successfully")
except ImportError as e:
    print(f"❌ Failed to import environment: {e}")
    sys.exit(1)

# Paths
if 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules:
    if os.path.exists('/content/drive/MyDrive'):
        ROOT = pathlib.Path('/content/drive/MyDrive/PriorityMax')
    else:
        ROOT = pathlib.Path('/content/PriorityMax')
else:
    ROOT = pathlib.Path(__file__).resolve().parents[2]

ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG = logging.getLogger("prioritymax.train_rl")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
))
if not LOG.handlers:
    LOG.addHandler(handler)

# =============================================================================
# STABLE CONFIGURATION (RESEARCH-BACKED)
# =============================================================================

@dataclass
class TrainingConfig:
    """
    STABLE CONFIGURATION based on research and best practices
    ✅ More epochs with smaller steps (prevents explosion)
    ✅ Conservative learning rate
    ✅ Lower gamma (reduces GAE accumulation)
    ✅ Reduced value coefficient
    """
    
    # === SYNCHRONIZED WITH real_env.py ===
    obs_dim: int = 12
    act_dim: int = 3
    
    # === TRAINING SCHEDULE (STABLE) ===
    epochs: int = 500                      # MORE epochs
    steps_per_epoch: int = 2048           # SMALLER steps (from 4096)
    eval_episodes: int = 8
    eval_interval: int = 10
    
    # === PPO HYPERPARAMETERS (STABLE) ===
    gamma: float = 0.99                   # Reduced from 0.995
    lam: float = 0.95                     # Reduced from 0.97
    clip_ratio: float = 0.2               # Standard
    target_kl: float = 0.015              # Early stopping
    
    # === OPTIMIZATION (STABLE) ===
    lr: float = 3e-4                      # Standard PPO (not 1e-4)
    lr_schedule: str = "linear"           # Decay over time
    entropy_coef: float = 0.01            # Standard
    value_coef: float = 0.5               # Standard (clipping handles explosion)
    max_grad_norm: float = 0.5            # Strict clipping
    
    # === MODEL ===
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    
    # === MEMORY EFFICIENCY ===
    batch_size: int = 64
    minibatch_size: int = 64
    update_epochs: int = 10               # Standard PPO
    use_amp: bool = True
    
    # === ENVIRONMENT ===
    workload_type: str = "ecommerce"
    num_envs: int = 8
    max_episode_steps: int = 2000         # Match environment
    
    # === CHECKPOINTING ===
    checkpoint_interval: int = 10
    keep_top_k: int = 3
    auto_resume: bool = True
    emergency_save: bool = True
    
    # === LOGGING ===
    log_wandb: bool = True
    wandb_project: str = "PriorityMax-RL"
    wandb_entity: Optional[str] = None
    experiment_name: str = "ppo_stable"
    verbose: bool = True
    
    # === ONNX ===
    export_onnx: bool = True
    validate_onnx: bool = True
    
    # === EARLY STOPPING ===
    patience: int = 50
    min_delta: float = 0.5
    
    # === SEED ===
    seed: int = 42

# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(state: Dict[str, Any], path: str):
    """Save checkpoint atomically"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = path.with_suffix('.tmp')
    torch.save(state, tmp_path)
    tmp_path.replace(path)
    
    LOG.info(f"💾 Checkpoint saved: {path.name}")

def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint"""
    return torch.load(path, map_location=device)

def format_time(seconds: float) -> str:
    """Format seconds"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# =============================================================================
# LR SCHEDULER
# =============================================================================

class LRScheduler:
    """Learning rate scheduler"""
    
    def __init__(self, optimizer, schedule: str, initial_lr: float, 
                 total_steps: int, warmup_steps: int = 0):
        self.optimizer = optimizer
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            lr = self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.schedule == "cosine":
                lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.schedule == "linear":
                lr = self.initial_lr * (1 - progress)
            else:
                lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# =============================================================================
# ACTOR-CRITIC MODEL
# =============================================================================

class ActorCritic(nn.Module):
    """Production-grade actor-critic"""
    
    def __init__(self, obs_dim: int = 12, act_dim: int = 3, 
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 activation: str = "relu"):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh if activation == "tanh" else nn.GELU
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            act_fn(),
        )
        
        # Policy head
        policy_layers = []
        prev_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            policy_layers.extend([nn.Linear(prev_dim, h), act_fn()])
            prev_dim = h
        policy_layers.append(nn.Linear(prev_dim, act_dim))
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Value head
        value_layers = []
        prev_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            value_layers.extend([nn.Linear(prev_dim, h), act_fn()])
            prev_dim = h
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Orthogonal initialization"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.shared_net(obs)
        action_mean = self.policy_net(features)
        value = self.value_net(features).squeeze(-1)
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action"""
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions"""
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy

# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """Memory-efficient rollout buffer"""
    
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.values = torch.zeros(capacity, dtype=torch.float32)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs, action, reward, value, log_prob, done):
        """Add transition"""
        self.observations[self.ptr] = torch.from_numpy(np.array(obs))
        self.actions[self.ptr] = torch.from_numpy(np.array(action))
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = bool(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, lam: float):
        """Compute GAE advantages"""
        advantages = torch.zeros(self.size, dtype=torch.float32)
        last_gae = 0.0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + self.values[:self.size]
        
        # CRITICAL: Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def get_batch(self, batch_indices):
        """Get batch"""
        return {
            'observations': self.observations[batch_indices].to(self.device),
            'actions': self.actions[batch_indices].to(self.device),
            'advantages': self.advantages[batch_indices].to(self.device),
            'returns': self.returns[batch_indices].to(self.device),
            'old_log_probs': self.log_probs[batch_indices].to(self.device),
            'old_values': self.values[batch_indices].to(self.device)
        }
    
    def prepare_training(self, advantages, returns):
        """Store computed values"""
        self.advantages = advantages
        self.returns = returns
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0

# =============================================================================
# VECTORIZED ENVIRONMENT WRAPPER
# =============================================================================

class VectorizedEnvWrapper:
    """Proper vectorized environment wrapper"""
    
    def __init__(self, envs: List):
        self.envs = envs
        self.num_envs = len(envs)
        LOG.info(f"✅ VectorizedEnvWrapper initialized with {self.num_envs} environments")

    def reset(self, seed=None):
        """Reset all environments"""
        obs_list = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            try:
                obs = env.reset(seed=env_seed)
                obs_list.append(obs)
            except Exception as e:
                LOG.error(f"❌ Env {i} reset failed: {e}")
                obs_list.append(np.zeros(self.envs[0].cfg.obs_dim, dtype=np.float32))
        
        return obs_list

    def step(self, actions):
        """Step through all environments"""
        obs_list, rewards, dones, infos = [], [], [], []
        
        if isinstance(actions, np.ndarray):
            if actions.ndim == 1:
                actions = [actions for _ in range(self.num_envs)]
            else:
                actions = [actions[i] for i in range(self.num_envs)]
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            try:
                obs, reward, done, info = env.step(action)
                
                if done:
                    obs = env.reset()
                    info['terminal_observation'] = obs
                
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                
            except Exception as e:
                LOG.warning(f"⚠️ Env {i} step failed: {e}")
                obs_list.append(np.zeros(self.envs[0].cfg.obs_dim, dtype=np.float32))
                rewards.append(0.0)
                dones.append(True)
                infos.append({"error": str(e)})
        
        return obs_list, np.array(rewards), np.array(dones), infos

    def close(self):
        """Close all environments"""
        for i, env in enumerate(self.envs):
            try:
                env.close()
            except Exception as e:
                LOG.warning(f"⚠️ Env {i} close failed: {e}")

# =============================================================================
# PPO TRAINER (STABLE VERSION)
# =============================================================================

class PPOTrainer:
    """STABLE PPO trainer with explosion prevention"""
    
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        LOG.info("="*60)
        LOG.info("🚀 Initializing STABLE PPO Trainer")
        LOG.info("="*60)
        LOG.info(f"Device: {self.device}")
        LOG.info(f"Configuration: epochs={config.epochs}, steps={config.steps_per_epoch}")
        LOG.info(f"Hyperparameters: lr={config.lr}, gamma={config.gamma}, lambda={config.lam}")
        
        set_seed(config.seed)
        
        # Create environments
        self.envs = self._make_envs()
        
        # Create model
        self.model = ActorCritic(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        
        # LR scheduler
        total_steps = config.epochs * config.steps_per_epoch // config.batch_size
        self.lr_scheduler = LRScheduler(
            self.optimizer, 
            config.lr_schedule, 
            config.lr,
            total_steps,
            warmup_steps=total_steps // 10
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            capacity=config.steps_per_epoch * config.num_envs,
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            device=self.device
        )
        
        # Training state
        self.epoch = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.best_checkpoints = []
        self.no_improvement_count = 0
        
        # Metrics
        self.training_metrics = defaultdict(list)
        
        # W&B
        if config.log_wandb and _HAS_WANDB:
            try:
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=asdict(config)
                )
                LOG.info("✅ Weights & Biases initialized")
            except Exception as e:
                LOG.warning(f"⚠️  W&B init failed: {e}")
                self.cfg.log_wandb = False
        
        LOG.info("="*60)
        LOG.info("✅ Trainer initialized successfully")
        LOG.info("="*60)
    
    def _make_envs(self):
        """Create vectorized environments"""
        env_cfg = EnvConfig(
            mode="sim",
            obs_dim=self.cfg.obs_dim,
            act_dim=self.cfg.act_dim,
            workload_type=self.cfg.workload_type,
            seed=self.cfg.seed
        )
        
        envs = []
        for i in range(self.cfg.num_envs):
            env_cfg_copy = EnvConfig(**asdict(env_cfg))
            env_cfg_copy.seed = self.cfg.seed + i
            envs.append(SimulatedRealEnv(env_cfg_copy))
        
        LOG.info(f"✅ Created {self.cfg.num_envs} environments")
        
        return VectorizedEnvWrapper(envs)
    
    def _collect_rollout(self):
        """Collect trajectory data"""
        self.model.eval()
        
        obs_list = self.envs.reset(seed=self.cfg.seed + self.epoch)
        steps = 0
        episode_rewards = []
        current_episode_reward = [0.0] * self.cfg.num_envs
        
        with torch.no_grad():
            while steps < self.cfg.steps_per_epoch:
                obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
                
                actions, log_probs, values = self.model.get_action(obs_tensor)
                
                actions_np = actions.cpu().numpy()
                next_obs_list, rewards, dones, infos = self.envs.step(actions_np)
                
                for i in range(self.cfg.num_envs):
                    self.buffer.add(
                        obs_list[i],
                        actions_np[i],
                        rewards[i],
                        values[i].item(),
                        log_probs[i].item(),
                        dones[i]
                    )
                    
                    current_episode_reward[i] += rewards[i]
                    
                    if dones[i]:
                        episode_rewards.append(current_episode_reward[i])
                        current_episode_reward[i] = 0.0
                    
                    steps += 1
                    if steps >= self.cfg.steps_per_epoch:
                        break
                
                obs_list = next_obs_list
            
            last_obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
            _, last_values = self.model.forward(last_obs_tensor)
            last_value = last_values.mean().item()
        
        advantages, returns = self.buffer.compute_returns_and_advantages(
            last_value, self.cfg.gamma, self.cfg.lam
        )
        self.buffer.prepare_training(advantages, returns)
        
        metrics = {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0.0,
            'n_episodes': len(episode_rewards)
        }
        
        return metrics
    
    def _update_policy(self):
        """PPO policy update with CLIPPED VALUE LOSS"""
        self.model.train()
        
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_div': [],
            'clip_fraction': [],
            'value_mean': [],
            'value_std': []
        }
        
        indices = np.arange(self.buffer.size)
        
        for epoch_i in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, self.buffer.size, self.cfg.minibatch_size):
                end = min(start + self.cfg.minibatch_size, self.buffer.size)
                batch_idx = indices[start:end]
                
                batch = self.buffer.get_batch(batch_idx)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        log_probs, values, entropy = self.model.evaluate_actions(
                            batch['observations'], 
                            batch['actions']
                        )
                        
                        # Policy loss
                        ratio = torch.exp(log_probs - batch['old_log_probs'])
                        surr1 = ratio * batch['advantages']
                        surr2 = torch.clamp(
                            ratio, 
                            1 - self.cfg.clip_ratio, 
                            1 + self.cfg.clip_ratio
                        ) * batch['advantages']
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # CRITICAL: Clipped value loss (prevents explosion)
                        value_pred_clipped = batch['old_values'] + torch.clamp(
                            values - batch['old_values'],
                            -self.cfg.clip_ratio,
                            self.cfg.clip_ratio
                        )
                        value_loss_unclipped = (values - batch['returns']) ** 2
                        value_loss_clipped = (value_pred_clipped - batch['returns']) ** 2
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                        
                        # Total loss
                        loss = (policy_loss + 
                               self.cfg.value_coef * value_loss - 
                               self.cfg.entropy_coef * entropy.mean())
                    
                    # Check for NaN
                    if torch.isnan(loss):
                        LOG.error("❌ NaN loss detected! Skipping update")
                        continue
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.cfg.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                else:
                    log_probs, values, entropy = self.model.evaluate_actions(
                        batch['observations'], 
                        batch['actions']
                    )
                    
                    ratio = torch.exp(log_probs - batch['old_log_probs'])
                    surr1 = ratio * batch['advantages']
                    surr2 = torch.clamp(
                        ratio, 
                        1 - self.cfg.clip_ratio, 
                        1 + self.cfg.clip_ratio
                    ) * batch['advantages']
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Clipped value loss
                    value_pred_clipped = batch['old_values'] + torch.clamp(
                        values - batch['old_values'],
                        -self.cfg.clip_ratio,
                        self.cfg.clip_ratio
                    )
                    value_loss_unclipped = (values - batch['returns']) ** 2
                    value_loss_clipped = (value_pred_clipped - batch['returns']) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    
                    loss = (policy_loss + 
                           self.cfg.value_coef * value_loss - 
                           self.cfg.entropy_coef * entropy.mean())
                    
                    if torch.isnan(loss):
                        LOG.error("❌ NaN loss detected! Skipping update")
                        continue
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.cfg.max_grad_norm
                    )
                    self.optimizer.step()
                
                current_lr = self.lr_scheduler.step()
                
                with torch.no_grad():
                    kl_div = (batch['old_log_probs'] - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_ratio).float().mean().item()
                
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['kl_div'].append(kl_div)
                stats['clip_fraction'].append(clip_fraction)
                stats['value_mean'].append(values.mean().item())
                stats['value_std'].append(values.std().item())
            
            # Early stopping if KL too high
            mean_kl = np.mean(stats['kl_div'][-10:])
            if mean_kl > self.cfg.target_kl * 1.5:
                LOG.warning(f"⚠️  Early stopping at update epoch {epoch_i}: KL={mean_kl:.4f}")
                break
        
        return {k: np.mean(v) for k, v in stats.items()}
    
    def _evaluate(self):
        """Evaluate current policy"""
        self.model.eval()
        
        eval_env = SimulatedRealEnv(EnvConfig(
            mode="sim",
            obs_dim=self.cfg.obs_dim,
            act_dim=self.cfg.act_dim,
            workload_type=self.cfg.workload_type,
            seed=self.cfg.seed + 999999
        ))
        
        episode_rewards = []
        episode_lengths = []
        sla_violations = []
        total_costs = []
        
        with torch.no_grad():
            for ep in range(self.cfg.eval_episodes):
                obs = eval_env.reset(seed=self.cfg.seed + ep + 1000)
                total_reward = 0.0
                steps = 0
                
                for _ in range(self.cfg.max_episode_steps):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _, _ = self.model.get_action(obs_tensor, deterministic=True)
                    action_np = action.cpu().numpy()[0]
                    
                    obs, reward, done, info = eval_env.step(action_np)
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                sla_violations.append(info.get('sla_violations', 0))
                total_costs.append(info.get('total_cost', 0.0))
        
        eval_env.close()
        
        metrics = {
            'eval/mean_reward': np.mean(episode_rewards),
            'eval/std_reward': np.std(episode_rewards),
            'eval/mean_length': np.mean(episode_lengths),
            'eval/mean_sla_violations': np.mean(sla_violations),
            'eval/mean_cost': np.mean(total_costs)
        }
        
        return metrics
    
    def _save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint"""
        state = {
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'config': asdict(self.cfg),
            'training_metrics': dict(self.training_metrics)
        }
        
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        save_checkpoint(state, path)
        
        if is_best:
            self.best_checkpoints.append((path, self.best_reward))
            self.best_checkpoints.sort(key=lambda x: x[1], reverse=True)
            self.best_checkpoints = self.best_checkpoints[:self.cfg.keep_top_k]
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint"""
        LOG.info(f"📂 Loading checkpoint: {path}")
        
        state = load_checkpoint(path, self.device)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epoch = state.get('epoch', 0)
        self.total_steps = state.get('total_steps', 0)
        self.best_reward = state.get('best_reward', -float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in state:
            self.scaler.load_state_dict(state['scaler_state_dict'])
        
        LOG.info(f"✅ Resumed from epoch {self.epoch}, best reward: {self.best_reward:.2f}")
    
    def _export_onnx(self, save_path: str):
        """Export model to ONNX"""
        try:
            self.model.eval()
            dummy_input = torch.randn(1, self.cfg.obs_dim).to(self.device)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                save_path,
                input_names=['observation'],
                output_names=['action_mean', 'value'],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'action_mean': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                },
                opset_version=17,
                do_constant_folding=True
            )
            
            LOG.info(f"✅ ONNX model exported: {save_path}")
            return True
        except Exception as e:
            LOG.error(f"❌ ONNX export failed: {e}")
            return False
    
    def train(self):
        """Main training loop - STABLE"""
        LOG.info("\n" + "="*60)
        LOG.info("🎯 Starting STABLE Training")
        LOG.info("="*60)
        
        # Auto-resume
        if self.cfg.auto_resume:
            ckpts = sorted(CHECKPOINT_DIR.glob("ckpt_*.pt"))
            if ckpts:
                latest = ckpts[-1]
                try:
                    self._load_checkpoint(str(latest))
                except Exception as e:
                    LOG.warning(f"⚠️  Failed to load checkpoint: {e}")
        
        start_time = time.time()
        start_epoch = self.epoch + 1
        
        try:
            pbar = tqdm(
                range(start_epoch, self.cfg.epochs + 1),
                desc="Training Epochs",
                disable=not self.cfg.verbose,
                initial=start_epoch-1,
                total=self.cfg.epochs
            )
            
            for epoch in pbar:
                self.epoch = epoch
                epoch_start = time.time()
                
                # Collect rollout
                rollout_metrics = self._collect_rollout()
                self.total_steps += self.cfg.steps_per_epoch * self.cfg.num_envs
                
                # Update policy
                update_metrics = self._update_policy()
                
                # Clear buffer
                self.buffer.clear()
                
                # Combine metrics
                metrics = {**rollout_metrics, **update_metrics}
                metrics['epoch'] = epoch
                metrics['total_steps'] = self.total_steps
                metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                metrics['epoch_time'] = time.time() - epoch_start
                
                # CRITICAL: Monitor value predictions
                if abs(metrics.get('value_mean', 0)) > 100:
                    LOG.error(f"❌ VALUE EXPLOSION DETECTED! value_mean={metrics['value_mean']:.2f}")
                    LOG.error("   Stopping training to prevent further divergence")
                    break
                
                # Update progress bar
                pbar.set_postfix({
                    'reward': f"{rollout_metrics['mean_reward']:.1f}",
                    'value_mean': f"{metrics.get('value_mean', 0):.1f}",
                    'episodes': rollout_metrics['n_episodes']
                })
                
                # Periodic evaluation
                if epoch % self.cfg.eval_interval == 0 or epoch == self.cfg.epochs:
                    eval_metrics = self._evaluate()
                    metrics.update(eval_metrics)
                    
                    mean_reward = eval_metrics['eval/mean_reward']
                    
                    if mean_reward > self.best_reward + self.cfg.min_delta:
                        improvement = mean_reward - self.best_reward
                        self.best_reward = mean_reward
                        self.no_improvement_count = 0
                        
                        best_path = str(CHECKPOINT_DIR / f"ckpt_best_{epoch:04d}.pt")
                        self._save_checkpoint(best_path, is_best=True)
                        
                        LOG.info(f"🎉 New best! Reward: {mean_reward:.2f} (+{improvement:.2f})")
                    else:
                        self.no_improvement_count += 1
                    
                    LOG.info(
                        f"[Epoch {epoch}/{self.cfg.epochs}] "
                        f"Reward: {mean_reward:.2f} | "
                        f"SLA: {eval_metrics['eval/mean_sla_violations']:.1f} | "
                        f"Cost: ${eval_metrics['eval/mean_cost']:.2f} | "
                        f"Value: {metrics.get('value_mean', 0):.2f}±{metrics.get('value_std', 0):.2f}"
                    )
                
                # Periodic checkpoint
                if epoch % self.cfg.checkpoint_interval == 0:
                    ckpt_path = str(CHECKPOINT_DIR / f"ckpt_{epoch:04d}.pt")
                    self._save_checkpoint(ckpt_path, is_best=False)
                
                # Log to W&B
                if self.cfg.log_wandb and _HAS_WANDB:
                    try:
                        wandb.log(metrics, step=epoch)
                    except:
                        pass
                
                # Store metrics
                for k, v in metrics.items():
                    self.training_metrics[k].append(v)
                
                # Early stopping
                if self.no_improvement_count >= self.cfg.patience:
                    LOG.warning(f"⚠️  Early stopping: no improvement for {self.cfg.patience} evaluations")
                    break
            
            pbar.close()
            
            # Training complete
            total_time = time.time() - start_time
            LOG.info("\n" + "="*60)
            LOG.info("✅ Training Complete!")
            LOG.info("="*60)
            LOG.info(f"Total time: {format_time(total_time)}")
            LOG.info(f"Best reward: {self.best_reward:.2f}")
            LOG.info(f"Total steps: {self.total_steps:,}")
            
            # Save final checkpoint
            final_path = str(CHECKPOINT_DIR / f"ckpt_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            self._save_checkpoint(final_path, is_best=False)
            
            # Export ONNX
            if self.cfg.export_onnx and self.best_checkpoints:
                best_ckpt_path = self.best_checkpoints[0][0]
                self._load_checkpoint(best_ckpt_path)
                
                onnx_path = str(CHECKPOINT_DIR / f"model_best.onnx")
                self._export_onnx(onnx_path)
            
            # Final W&B summary
            if self.cfg.log_wandb and _HAS_WANDB:
                try:
                    wandb.summary['best_reward'] = self.best_reward
                    wandb.summary['total_time_hours'] = total_time / 3600
                    wandb.finish()
                except:
                    pass
            
            return {
                'best_reward': self.best_reward,
                'total_time': total_time,
                'total_steps': self.total_steps,
                'final_checkpoint': final_path
            }
        
        except KeyboardInterrupt:
            LOG.warning("\n⚠️  Training interrupted by user")
            return {'interrupted': True}
        
        except Exception as e:
            LOG.error(f"\n❌ Training failed: {e}")
            LOG.error(traceback.format_exc())
            return {'error': str(e)}
        
        finally:
            try:
                self.envs.close()
            except:
                pass

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PriorityMax STABLE RL Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=2048, help="Steps per epoch")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    
    # Environment
    parser.add_argument("--workload", type=str, default="ecommerce",
                       choices=["ecommerce", "social_media", "streaming", "api_backend", "batch", "gaming"],
                       help="Workload type")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    
    # Optimization
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", default=True, help="Use Weights & Biases")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--exp-name", type=str, default="ppo_stable", help="Experiment name")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    # Build config
    config = TrainingConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_interval=args.eval_interval,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        workload_type=args.workload,
        num_envs=args.num_envs,
        use_amp=args.use_amp,
        log_wandb=args.wandb,
        experiment_name=args.exp_name,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Print configuration
    LOG.info("\n" + "="*60)
    LOG.info("STABLE TRAINING CONFIGURATION")
    LOG.info("="*60)
    for key, value in asdict(config).items():
        LOG.info(f"{key:.<30} {value}")
    LOG.info("="*60 + "\n")
    
    # Create trainer and train
    trainer = PPOTrainer(config)
    results = trainer.train()
    
    # Print results
    LOG.info("\n" + "="*60)
    LOG.info("TRAINING RESULTS")
    LOG.info("="*60)
    for key, value in results.items():
        LOG.info(f"{key:.<30} {value}")
    LOG.info("="*60 + "\n")
    
    return results

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Module Synchronization Tests - Enterprise Version
Ensures `real_env.py`, `rl_agent.py`, and RL training scripts stay in sync.
"""

import importlib.util
import pathlib
import re
from types import ModuleType
from typing import Optional, Tuple, Dict, Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ML = ROOT / "app" / "ml"
SCRIPTS = ROOT / "scripts"

FILES = {
    "real_env": APP_ML / "real_env.py",
    "rl_agent": APP_ML / "rl_agent.py",
    "rl_agent_sandbox": APP_ML / "rl_agent_sandbox.py",
    "rl_agent_prod": APP_ML / "rl_agent_prod.py",
    "train_rl_heavy": SCRIPTS / "train_rl_heavy.py",
    "train_rl_live": SCRIPTS / "train_rl_live.py",
    "train_rl_eval": SCRIPTS / "train_rl_eval.py",
}

# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------
def load_module_from_path(name: str, path: pathlib.Path) -> Optional[ModuleType]:
    """Import module from arbitrary path."""
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    except Exception:
        return None


def read_source(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_first_number_after_pattern(src: str, pattern: str) -> Optional[float]:
    """Return numeric literal following a pattern like `key = 123`."""
    m = re.search(pattern + r"\s*=\s*([0-9.eE+-]+)", src)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def find_obs_act_dims_in_rl_agent_source(src: str) -> Tuple[Optional[int], Optional[int]]:
    """Find obs/action dims in PPO or RLAgent definitions."""
    obs = act = None
    # Look for obs_dim / act_dim hints in constructors
    m = re.search(r"obs_dim\s*[:=]\s*([0-9]+)", src)
    if m:
        obs = int(m.group(1))
    m2 = re.search(r"act_dim\s*[:=]\s*([0-9]+)", src)
    if m2:
        act = int(m2.group(1))
    # fallback: np.zeros
    if obs is None:
        m3 = re.search(r"np\.zeros\(\s*([0-9]+)\s*,", src)
        if m3:
            obs = int(m3.group(1))
    return obs, act


# ------------------------------------------------------------------------
# Improved extractor for real_env module
# ------------------------------------------------------------------------
def get_obs_act_from_real_env_module(mod: ModuleType) -> Tuple[Optional[int], Optional[int]]:
    """Try to call get_observation_space/get_action_space and extract .shape safely."""
    obs_dim = act_dim = None
    try:
        # --- Observation space ---
        if hasattr(mod, "get_observation_space"):
            try:
                sp = mod.get_observation_space()
            except TypeError:
                sp = mod.get_observation_space(8)
            print(f"[DEBUG] get_observation_space() returned: {type(sp)} attrs={dir(sp)}")
            if sp is not None:
                if hasattr(sp, "shape") and sp.shape is not None:
                    obs_dim = int(sp.shape[0]) if isinstance(sp.shape, tuple) else int(sp.shape)
                elif hasattr(sp, "low") and hasattr(sp, "high"):
                    # Derive dim from low/high arrays if available
                    try:
                        obs_dim = len(sp.low)
                    except Exception:
                        pass
                elif hasattr(sp, "__len__"):
                    obs_dim = len(sp)
        # --- Action space ---
        if hasattr(mod, "get_action_space"):
            sp = mod.get_action_space()
            print(f"[DEBUG] get_action_space() returned: {type(sp)} attrs={dir(sp)}")
            if sp is not None:
                if hasattr(sp, "shape") and sp.shape is not None:
                    act_dim = int(sp.shape[0]) if isinstance(sp.shape, tuple) else int(sp.shape)
                elif hasattr(sp, "low") and hasattr(sp, "high"):
                    try:
                        act_dim = len(sp.low)
                    except Exception:
                        pass
                elif hasattr(sp, "__len__"):
                    act_dim = len(sp)
    except Exception as e:
        print(f"[ERROR] Exception while reading obs/act: {e}")
    return obs_dim, act_dim


def extract_constants_from_source(src: str) -> Dict[str, Any]:
    keys = {
        "reward_latency_sla_ms": r"reward_latency_sla_ms",
        "cost_per_worker_per_sec": r"cost_per_worker_per_sec",
        "max_scale_delta": r"max_scale_delta",
        "max_delta": r"max_delta",
    }
    found = {}
    for k, pat in keys.items():
        val = extract_first_number_after_pattern(src, pat)
        if val is not None:
            found[k] = val
    return found


# ------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------
def test_files_exist():
    required = ["real_env", "rl_agent", "train_rl_live", "train_rl_eval"]
    missing = [k for k in required if not FILES[k].exists()]
    assert not missing, f"Missing required files: {missing}"


def test_real_env_exposes_spaces_or_shapes():
    """Ensure real_env defines get_observation_space/get_action_space correctly, compatible with Gymnasium."""
    path = FILES["real_env"]
    mod = load_module_from_path("real_env_for_tests", path)
    src = read_source(path)
    obs_dim = act_dim = None

    if not mod:
        pytest.skip("Could not import real_env module")

    # ---- Observation Space ----
    if hasattr(mod, "get_observation_space"):
        try:
            sp = mod.get_observation_space()
        except TypeError:
            sp = mod.get_observation_space(8)

        print("\n[DEBUG] get_observation_space() returned:", sp)
        print("[DEBUG] Type:", type(sp))
        print("[DEBUG] Dir:", dir(sp))

        # Gymnasium Box has attribute shape=(8,)
        try:
            if hasattr(sp, "shape") and sp.shape is not None:
                # If it's a tuple, take first dimension
                obs_dim = int(sp.shape[0]) if isinstance(sp.shape, tuple) else int(sp.shape)
            elif hasattr(sp, "low") and hasattr(sp.low, "__len__"):
                obs_dim = len(sp.low)
            elif hasattr(sp, "high") and hasattr(sp.high, "__len__"):
                obs_dim = len(sp.high)
            elif isinstance(sp, dict) and "shape" in sp:
                obs_dim = int(sp["shape"][0])
        except Exception as e:
            print(f"[DEBUG] Error reading obs_dim: {e}")

    # ---- Action Space ----
    if hasattr(mod, "get_action_space"):
        try:
            sp = mod.get_action_space()
        except TypeError:
            sp = mod.get_action_space()

        print("\n[DEBUG] get_action_space() returned:", sp)
        print("[DEBUG] Type:", type(sp))
        print("[DEBUG] Dir:", dir(sp))

        try:
            if hasattr(sp, "shape") and sp.shape is not None:
                act_dim = int(sp.shape[0]) if isinstance(sp.shape, tuple) else int(sp.shape)
            elif hasattr(sp, "low") and hasattr(sp.low, "__len__"):
                act_dim = len(sp.low)
            elif hasattr(sp, "high") and hasattr(sp.high, "__len__"):
                act_dim = len(sp.high)
            elif isinstance(sp, dict) and "shape" in sp:
                act_dim = int(sp["shape"][0])
        except Exception as e:
            print(f"[DEBUG] Error reading act_dim: {e}")

    # ---- Fallbacks ----
    if obs_dim is None:
        m2 = re.search(r"np\.zeros\(\s*([0-9]+)\s*,", src)
        if m2:
            obs_dim = int(m2.group(1))
    if act_dim is None:
        m3 = re.search(r"action vector.*\[(.*?)\]", src, re.IGNORECASE)
        if m3:
            act_dim = len([x for x in m3.group(1).split(",") if x.strip() != ""])

    print(f"\n[FINAL DEBUG] Detected real_env obs_dim={obs_dim}, act_dim={act_dim}")

    # ---- Assertions ----
    assert isinstance(obs_dim, int) and obs_dim > 0, (
        f"Failed to detect observation dim (obs_dim={obs_dim}) — "
        f"ensure get_observation_space() returns an object with .shape or .low attributes."
    )
    assert isinstance(act_dim, int) and act_dim > 0, (
        f"Failed to detect action dim (act_dim={act_dim}) — "
        f"ensure get_action_space() returns an object with .shape or .low attributes."
    )


def test_rl_agent_sync_with_real_env():
    """Ensure RL agent's expected dims match real_env."""
    real_mod = load_module_from_path("real_env_for_tests", FILES["real_env"])
    real_obs, real_act = get_obs_act_from_real_env_module(real_mod) if real_mod else (None, None)

    agent_src = read_source(FILES["rl_agent"])
    agent_obs, agent_act = find_obs_act_dims_in_rl_agent_source(agent_src)

    uses_getters = "get_observation_space" in agent_src and "get_action_space" in agent_src

    print(f"[DEBUG] real_env=(obs={real_obs}, act={real_act}), rl_agent=(obs={agent_obs}, act={agent_act})")

    if real_obs:
        assert agent_obs == real_obs, f"Mismatch: real_env obs_dim={real_obs} vs rl_agent obs_dim={agent_obs}"
    if real_act:
        assert agent_act == real_act, f"Mismatch: real_env act_dim={real_act} vs rl_agent act_dim={agent_act}"

    assert uses_getters, "rl_agent.py should use get_observation_space/get_action_space for sync validation."


def test_shared_constants_consistency():
    """Ensure constants (reward_latency_sla_ms, etc.) are identical across scripts."""
    targets = ["rl_agent", "train_rl_live", "train_rl_eval"]
    extracted = {}
    for key in targets:
        src = read_source(FILES[key])
        extracted[key] = extract_constants_from_source(src)

    all_keys = set().union(*(set(v.keys()) for v in extracted.values()))
    assert all_keys, "No shared numeric constants found across RL scripts."

    for const in ["reward_latency_sla_ms", "cost_per_worker_per_sec", "max_scale_delta", "max_delta"]:
        vals = {k: v[const] for k, v in extracted.items() if const in v}
        if len(vals) > 1:
            ref_name, ref_val = next(iter(vals.items()))
            for name, val in vals.items():
                assert abs(val - ref_val) < 1e-6, (
                    f"Constant '{const}' mismatch: {ref_name}={ref_val} vs {name}={val}"
                )


def test_presence_of_model_and_registry_helpers():
    src = read_source(FILES["rl_agent"])
    assert any(k in src for k in ["ModelRegistry", "model_registry"]), "rl_agent.py missing ModelRegistry reference."
    assert any(k in src.lower() for k in ["predictor", "predictor_manager"]), "rl_agent.py missing Predictor reference."
    assert "DEFAULT_RL_MODEL" in src or "model_path" in src, "rl_agent.py missing DEFAULT_RL_MODEL or model path constant."


def test_debug_print_shapes_if_possible():
    """Print shapes for debug."""
    mod = load_module_from_path("real_env_for_tests", FILES["real_env"])
    info = {}
    if mod:
        try:
            obs, act = get_obs_act_from_real_env_module(mod)
            info["real_env_obs"] = obs
            info["real_env_act"] = act
        except Exception:
            pass
    print("\n[SYNC DEBUG INFO]", info)

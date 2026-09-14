"""POPGym environment wrapper with synthetic-POMDP fallback.

Public surface: :class:`POPGymWrapper`.

The POPGym benchmark suite (Morad et al. 2023, "POPGym: Benchmarking Partially
Observable Reinforcement Learning") provides a wide range of memory-intensive
control tasks (Autoencode, Repeat Previous, etc.) with **fixed-horizon
episodes** and consistent observation shapes per environment.  This wrapper:

1. Builds an underlying :mod:`popgym` env when the optional dependency is
   installed.
2. Falls back to a deterministic synthetic POMDP that exercises the same
   outer-loop contract (``reset``/``step``/discrete action space) so the
   rest of the codebase stays import-clean without ``popgym``.
3. Optionally augments the observation with the previous reward and a
   normalised timestep, following the *MAMBA meta-RL* design
   (arXiv:2403.09859) which conditions the SSM on ``[o_t, r_t, t]`` so the
   model can learn reward- and time-dependent policies.

Gymnasium API
-------------
The wrapper exposes the modern 5-tuple ``step`` return ``(obs, reward, done,
truncated, info)`` so it slots straight into any Gymnasium-compatible
trainer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised only when popgym is installed
    import popgym  # type: ignore
    from gymnasium import spaces as _gym_spaces  # popgym depends on gymnasium

    POPGYM_AVAILABLE = True
except Exception:  # pragma: no cover
    POPGYM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Synthetic POMDP fallback
# ---------------------------------------------------------------------------
class _SyntheticPOMDP:
    """A tiny episodic POMDP used when :mod:`popgym` is not installed.

    The agent must output the same *binary* action as the most recent
    observation bit. Observations are 4-d binary vectors; the optimal policy
    achieves 100 % reward over a fixed horizon. This is just enough to keep
    the agent-side code path identical to the real POPGym flow.

    Per-step reward ∈ ``{0.0, 1.0}``; episode length = ``episode_len``.
    """

    def __init__(self, obs_dim: int = 4, episode_len: int = 32, seed: int = 0):
        self.obs_dim = obs_dim
        self.episode_len = episode_len
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._last_obs = self._rng.integers(0, 2, size=self.obs_dim).astype(
            np.float32)
        return self._last_obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._last_obs is not None, "call reset() before step()"
        # Optimal action: majority vote of last observation bits.
        majority = 1 if self._last_obs.sum() >= self.obs_dim / 2 else 0
        reward = float(action == majority)
        self._t += 1
        terminated = False
        truncated = self._t >= self.episode_len
        self._last_obs = self._rng.integers(0, 2, size=self.obs_dim).astype(
            np.float32)
        info: Dict[str, Any] = {"optimal_action": majority}
        return self._last_obs.copy(), reward, terminated, truncated, info


def _resolve_popgym_env_id(env_name: str) -> str:
    import gymnasium as gym
    registry = gym.envs.registry
    if env_name in registry:
        return env_name
    clean = env_name.replace("popgym:", "").replace("popgym-", "").rstrip("-v0")
    for suffix in ["Easy-v0", "-v0", "Medium-v0", "Hard-v0"]:
        cand = f"popgym-{clean}{suffix}"
        if cand in registry:
            return cand
    for k in registry.keys():
        if "popgym" in k.lower() and clean.lower() in k.lower():
            return k
    raise ValueError(f"Unknown POPGym environment: {env_name}")


def _compute_obs_dim(space: Any) -> int:
    from gymnasium import spaces
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    elif isinstance(space, spaces.Tuple):
        return sum(_compute_obs_dim(s) for s in space.spaces)
    elif isinstance(space, spaces.MultiDiscrete):
        return int(sum(space.nvec))
    elif isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    return int(getattr(space, "n", 1))


def _format_obs(obs: Any, space: Any) -> np.ndarray:
    from gymnasium import spaces
    if isinstance(space, spaces.Discrete):
        v = np.zeros(space.n, dtype=np.float32)
        v[int(obs)] = 1.0
        return v
    elif isinstance(space, spaces.Tuple):
        return np.concatenate([_format_obs(o, s) for o, s in zip(obs, space.spaces)], axis=0)
    elif isinstance(space, spaces.MultiDiscrete):
        parts = []
        for o, n in zip(obs, space.nvec):
            v = np.zeros(n, dtype=np.float32)
            v[int(o)] = 1.0
            parts.append(v)
        return np.concatenate(parts, axis=0)
    elif isinstance(space, spaces.Box):
        return np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _compute_act_dim(space: Any) -> int:
    from gymnasium import spaces
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    elif isinstance(space, spaces.MultiDiscrete):
        return int(np.prod(space.nvec))
    elif isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    return int(getattr(space, "n", 2))


def _convert_action(action: int, space: Any) -> Any:
    from gymnasium import spaces
    if isinstance(space, spaces.Discrete):
        return int(action)
    elif isinstance(space, spaces.MultiDiscrete):
        out = []
        rem = int(action)
        for dim in reversed(space.nvec):
            out.append(rem % dim)
            rem //= dim
        return list(reversed(out))
    return action


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class POPGymWrapper:
    """Unified POPGym / synthetic-POMDP wrapper with obs augmentation.

    Parameters
    ----------
    env_name:
        POPGym environment id (e.g. ``"Autoencode"``, ``"RepeatPrevious"``).
        Resolves to correct gymnasium IDs like ``"popgym-RepeatPreviousEasy-v0"``.
    augment_obs:
        When ``True``, concatenate ``[reward_{t-1}, t / episode_len]`` to the
        observation per the MAMBA meta-RL observation design.
    episode_len:
        Episode length for the synthetic POMDP fallback. Real POPGym envs
        already expose their own horizon — ignored there.
    """

    def __init__(self, env_name: str = "Autoencode",
                 augment_obs: bool = True,
                 episode_len: int = 32,
                 seed: int = 0):
        self.env_name = env_name
        self.augment_obs = augment_obs
        self.episode_len = episode_len
        self._seed = seed

        # ----- Build underlying env ------------------------------------
        self._use_synthetic = not POPGYM_AVAILABLE
        if self._use_synthetic:
            self._env = _SyntheticPOMDP(obs_dim=4, episode_len=episode_len,
                                        seed=seed)
            base_obs_dim = 4
            base_act_dim = 2
            self.is_discrete = True
            self._fallback_reason = "popgym package not available"
        else:
            try:
                import gymnasium as gym  # type: ignore

                real_id = _resolve_popgym_env_id(env_name)
                self._env = gym.make(real_id)
                self._real_env_id = real_id
                obs_space = self._env.observation_space
                act_space = self._env.action_space
                base_obs_dim = _compute_obs_dim(obs_space)
                base_act_dim = _compute_act_dim(act_space)
                self.is_discrete = isinstance(act_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete))
                self._fallback_reason = None
                self._use_synthetic = False
            except Exception as exc:  # pragma: no cover
                import warnings
                warnings.warn(
                    f"POPGymWrapper falling back to synthetic POMDP for '{env_name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._use_synthetic = True
                self._env = _SyntheticPOMDP(obs_dim=4,
                                            episode_len=episode_len,
                                            seed=seed)
                base_obs_dim = 4
                base_act_dim = 2
                self.is_discrete = True
                self._fallback_reason = str(exc)

        # ----- Cache dimensions ----------------------------------------
        self._base_obs_dim = base_obs_dim
        self._base_act_dim = base_act_dim
        # Optional augmentation adds 2 scalars: prev reward + normalised t.
        self._aug_extra = 2 if augment_obs else 0

        # ----- Episode state -------------------------------------------
        self._obs: Optional[np.ndarray] = None
        self._prev_reward: float = 0.0
        self._t: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        """Observation vector length **after** optional augmentation."""
        return self._base_obs_dim + self._aug_extra

    @property
    def act_dim(self) -> int:
        """Action cardinality (discrete) or dimensionality (continuous)."""
        return self._base_act_dim

    @property
    def using_synthetic(self) -> bool:
        """``True`` when the synthetic POMDP fallback is in use."""
        return self._use_synthetic

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the env and return the (optionally augmented) initial obs."""
        if self._use_synthetic:
            obs, info = self._env.reset(seed=seed)
            self._obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        else:
            kwargs = {} if seed is None else {"seed": seed}
            obs, info = self._env.reset(**kwargs)
            self._obs = _format_obs(obs, self._env.observation_space)
        self._prev_reward = 0.0
        self._t = 0
        return self._augment(self._obs, self._prev_reward, self._t,
                             self.episode_len), info

    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take ``action`` and return the (augmented) Gymnasium-5 tuple."""
        if self._use_synthetic:
            obs, reward, terminated, truncated, info = self._env.step(
                int(action))
            self._obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        else:
            real_act = _convert_action(action, self._env.action_space)
            obs, reward, terminated, truncated, info = self._env.step(real_act)
            reward = float(reward)
            self._obs = _format_obs(obs, self._env.observation_space)
        self._t += 1
        # Stash reward for the next observation's augmentation slot.
        self._prev_reward = float(reward)
        aug = self._augment(self._obs, self._prev_reward, self._t,
                            self.episode_len)
        info = dict(info) if info else {}
        info.setdefault("popgym_available", POPGYM_AVAILABLE)
        info.setdefault("synthetic", self._use_synthetic)
        return aug, reward, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    def _augment(self, obs: np.ndarray, prev_reward: float, t: int,
                 horizon: int) -> np.ndarray:
        """Concatenate ``[prev_reward, t/horizon]`` if ``augment_obs``."""
        if not self.augment_obs:
            return obs.astype(np.float32, copy=False)
        norm_t = float(t) / float(max(1, horizon))
        extra = np.array([prev_reward, norm_t], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32, copy=False), extra],
                              axis=0)

    def close(self) -> None:
        """Release underlying resources (no-op for the synthetic env)."""
        if not self._use_synthetic and hasattr(self._env, "close"):
            self._env.close()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    print(f"POPGYM_AVAILABLE={POPGYM_AVAILABLE}")

    env = POPGymWrapper(env_name="Autoencode", augment_obs=True,
                        episode_len=16, seed=42)

    obs, info = env.reset(seed=42)
    print(f"reset → obs.shape={obs.shape}  info={info}")
    assert obs.shape == (env.obs_dim,), (
        f"obs_dim mismatch: got {obs.shape[0]}, expected {env.obs_dim}")
    assert env.act_dim >= 2
    assert env.is_discrete

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        # Random policy — sufficient to exercise the API contract.
        action = int(np.random.default_rng(steps).integers(0, env.act_dim))
        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        steps += 1
        done = terminated or truncated
    print(f"random roll-out: steps={steps}  return={total_reward:.3f}  "
          f"info={info}")
    assert steps == env.episode_len, "synthetic env must hit episode_len"
    assert obs.shape == (env.obs_dim,)

    # --- No-augmentation path should be 2 dims shorter ---
    env2 = POPGymWrapper(env_name="Autoencode", augment_obs=False,
                         episode_len=8, seed=0)
    obs2, _ = env2.reset()
    assert obs2.shape[0] == env2.obs_dim
    assert obs2.shape[0] == env2._base_obs_dim
    print(f"no-aug obs_dim={env2.obs_dim}  (base={env2._base_obs_dim})")

    # --- Validate the aug vector contains (prev_reward, t/horizon) ---
    env3 = POPGymWrapper(env_name="Autoencode", augment_obs=True,
                         episode_len=10, seed=1)
    obs3, _ = env3.reset(seed=1)
    # After reset: prev_reward=0, t=0 → aug = [0.0, 0.0]
    np.testing.assert_allclose(obs3[-2:], [0.0, 0.0], atol=1e-6)
    # After one step with reward r, next obs must carry (r, 1/10).
    obs3_next, r, _, _, _ = env3.step(int(obs3[0] >= 0.5))
    np.testing.assert_allclose(obs3_next[-2:], [float(r), 0.1], atol=1e-6)
    print(f"aug vector after one step = {obs3_next[-2:].tolist()}  "
          f"(reward={r})")

    print("popgym_wrapper self-test ✓")

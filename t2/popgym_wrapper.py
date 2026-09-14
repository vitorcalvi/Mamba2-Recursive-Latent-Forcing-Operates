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


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class POPGymWrapper:
    """Unified POPGym / synthetic-POMDP wrapper with obs augmentation.

    Parameters
    ----------
    env_name:
        POPGym environment id (e.g. ``"Autoencode"``, ``"RepeatPreviousEasy"``).
        Ignored when :mod:`popgym` is unavailable; the synthetic fallback is
        used instead.
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
        else:
            # POPGym envs follow the gymnasium registry naming
            # ``popgym:<EnvName>-v0``.
            env_id = f"popgym:{env_name}-v0"
            try:
                import gymnasium as gym  # type: ignore

                self._env = gym.make(env_id)
            except Exception as exc:  # pragma: no cover
                # POPGym imports OK but the requested env isn't registered.
                # Fall back to synthetic so training can continue.
                self._use_synthetic = True
                self._env = _SyntheticPOMDP(obs_dim=4,
                                            episode_len=episode_len,
                                            seed=seed)
                base_obs_dim = 4
                base_act_dim = 2
                self.is_discrete = True
                self._fallback_reason = str(exc)
            else:
                obs_space = self._env.observation_space
                act_space = self._env.action_space
                base_obs_dim = int(np.prod(obs_space.shape))
                self.is_discrete = hasattr(act_space, "n")
                base_act_dim = int(act_space.n) if self.is_discrete else int(
                    np.prod(act_space.shape))
                self._fallback_reason = None

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
        else:
            kwargs = {} if seed is None else {"seed": seed}
            obs, info = self._env.reset(**kwargs)
        self._obs = np.asarray(obs, dtype=np.float32).reshape(-1)
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
        else:
            obs, reward, terminated, truncated, info = self._env.step(action)
            reward = float(reward)
        self._obs = np.asarray(obs, dtype=np.float32).reshape(-1)
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

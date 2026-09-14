"""DreamerV3-style actor-critic with symlog value transform and EMA target.

Public surface: :class:`DreamerActorCritic`.

Follows Hafner et al. 2024 ("Mastering Diverse Domains through World Models"):
- ``actor`` outputs a :class:`torch.distributions.Categorical` over discrete
  actions (or a :class:`Normal` for continuous control — only the discrete
  branch is exercised by the POPGym benchmark).
- ``critic`` regresses the symlog of the lambda-return, which keeps the
  target magnitude stable across the wide value range produced by
  POPGym's episodic auto-encoding tasks.
- ``target_critic`` is an exponential moving average of ``critic`` updated
  with ``target_tau`` per training step (Hafner uses ``tau=0.02``).
- :func:`compute_lambda_returns` implements the TD(λ) / eligibility-trace
  recurrence
  ``R_t^λ = r_t + γ · c_t · [(1-λ) · v̂_{t+1} + λ · R_{t+1}^λ]``
  where ``v̂`` is the EMA target value and ``c_t`` is the continue flag.

Both ``update_target`` and ``policy`` are written to be safe to call inside
``torch.no_grad`` blocks during the imagination rollout phase.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_model import ImaginedTrajectory, WorldModelState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mlp(in_d: int, hidden_d: int, out_d: int, depth: int = 3,
              act: str = "silu") -> nn.Sequential:
    """Build a depth-MLP with SiLU (DreamerV3 default) activations."""
    Act = nn.SiLU if act == "silu" else nn.GELU
    layers: List[nn.Module] = []
    d = in_d
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden_d), Act()]
        d = hidden_d
    layers.append(nn.Linear(d, out_d))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# DreamerV3 actor-critic
# ---------------------------------------------------------------------------
class DreamerActorCritic(nn.Module):
    """Actor-critic operating on imagined RSSM features.

    Parameters
    ----------
    state_dim:
        Dimensionality of the flattened RSSM feature vector
        (see :meth:`ImaginedTrajectory.features` → ``d_model + stoch_dim *
        n_categories``).
    act_dim:
        Number of discrete actions (or dimensionality of continuous action
        mean when ``act_type == "continuous"``).
    hidden:
        Width of every MLP hidden layer.
    act_type:
        ``"discrete"`` (default) → :class:`Categorical` policy.
        ``"continuous"`` → diagonal :class:`Normal` policy.
    gamma:
        Discount factor ``γ`` used in the lambda-return recurrence.
    lambda_:
        Trace mixing parameter ``λ ∈ [0, 1]``. ``λ=1`` ⇒ Monte-Carlo,
        ``λ=0`` ⇒ one-step TD.
    target_tau:
        EMA mixing rate for ``target_critic``:
        ``target ← (1-τ)·target + τ·critic``.
    """

    def __init__(self, state_dim: int, act_dim: int, hidden: int = 256,
                 act_type: str = "discrete", gamma: float = 0.997,
                 lambda_: float = 0.95, target_tau: float = 0.02):
        super().__init__()
        assert act_type in ("discrete", "continuous")
        self.state_dim, self.act_dim = state_dim, act_dim
        self.act_type = act_type
        self.gamma = gamma
        self.lambda_ = lambda_
        self.target_tau = target_tau

        # ── Networks ────────────────────────────────────────────────────
        self.actor = _make_mlp(state_dim, hidden, act_dim, depth=3)
        self.critic = _make_mlp(state_dim, hidden, 1, depth=3)
        # EMA target critic: a deepcopy frozen at init; the trainer calls
        # `update_target()` after each gradient step.
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Symlog utilities (DreamerV3 §3 — "symlog transform")
    # ------------------------------------------------------------------
    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        """sign(x) · log(|x| + 1) — numerically stable shrinkage of value scale."""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`symlog`."""
        return torch.sign(x) * (torch.expm1(torch.abs(x)))

    # ------------------------------------------------------------------
    # Lambda returns  (Hafner 2024 §B.2)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_lambda_returns(rewards: torch.Tensor,
                               values: torch.Tensor,
                               continues: torch.Tensor,
                               gamma: float,
                               lambda_: float) -> torch.Tensor:
        """Backward TD(λ) recurrence.

        Parameters
        ----------
        rewards, continues:
            ``[T, B]`` — predicted rewards and continuation flags along the
            imagined horizon. ``continues ∈ [0, 1]``.
        values:
            ``[T+1, B]`` — bootstrap values; ``values[T]`` is the value of
            the state *after* the last imagined step (used to bootstrap the
            final return).
        gamma, lambda_:
            See class docstring.

        Returns
        -------
        torch.Tensor
            ``[T, B]`` lambda-returns.
        """
        assert values.shape[0] == rewards.shape[0] + 1, (
            "values must be one step longer than rewards (bootstrap at T+1)")
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        # Bootstrap the last timestep with V(T) (no discount, no continue).
        next_return = values[-1]
        for t in reversed(range(T)):
            # Standard TD(λ) form: mix bootstrapped next-value with recursive
            # return; discount + continue gates both terms.
            mixed_value = (1.0 - lambda_) * values[t] + lambda_ * next_return
            next_return = rewards[t] + gamma * continues[t] * mixed_value
            returns[t] = next_return
        return returns

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def policy(self, state: WorldModelState) -> torch.distributions.Distribution:
        """Categorical/Normal policy over imagined states.

        Operates on the same flattened feature vector that
        :meth:`ImaginedTrajectory.features` produces.
        """
        feat = torch.cat([state.h, state.z.flatten(start_dim=1)], dim=-1)
        logits = self.actor(feat)
        if self.act_type == "discrete":
            return torch.distributions.Categorical(logits=logits)
        return torch.distributions.Normal(loc=logits, scale=torch.ones_like(logits))

    @torch.no_grad()
    def act(self, state: WorldModelState, sample: bool = True) -> torch.Tensor:
        """Sample (or take mode of) an action given an imagined state."""
        dist = self.policy(state)
        return dist.sample() if sample else dist.mode

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def actor_loss(self, imagined_traj: ImaginedTrajectory) -> torch.Tensor:
        """Reinforce with lambda-return baseline (REINFORCE w/ baseline).

        Implemented without the straight-through path: returns
        ``-E[(R_t^λ - symlog(V̂_θ(s_t))) · log π(a_t | s_t)]`` averaged over
        the (T, B) grid.
        """
        feats = imagined_traj.features()                  # [T, B, F]
        T, B = feats.shape[:2]
        actions = imagined_traj.actions                    # [T, B] or [T, B, A]
        rewards = imagined_traj.rewards                    # [T, B]
        continues = imagined_traj.continues                # [T, B]

        # Bootstrap values from the EMA target critic (detached).
        with torch.no_grad():
            values = self.target_critic(feats).squeeze(-1)     # [T, B]
            bootstrap = self.target_critic(feats[-1]).squeeze(-1)
            values = torch.cat([values, bootstrap.unsqueeze(0)], dim=0)
            returns = self.compute_lambda_returns(
                rewards, values, continues, self.gamma, self.lambda_)
            # Advantage in *original* value scale; symlog the baseline.
            sym_returns = self.symlog(returns)
            sym_baseline = values[:-1]                          # already symlog target
            advantage = sym_returns - sym_baseline              # [T, B]

        dist = self._policy_from_features(feats)
        if self.act_type == "discrete":
            # Categorical expects int64 indices [T, B]; world-model imagine
            # stores float-encoded actions for the dynamics, so cast here.
            if actions.dim() == 3 and actions.shape[-1] == self.act_dim:
                act_idx = actions.argmax(dim=-1)
            else:
                act_idx = actions.long()
            log_prob = dist.log_prob(act_idx)            # [T, B]
        else:
            # Continuous: actions [T, B, A], sum log_prob over action dim.
            log_prob = dist.log_prob(actions).sum(-1)           # [T, B]

        # Normalise advantage per imagined-batch to reduce variance.
        advantage = (advantage - advantage.mean()) / (
            advantage.std() + 1e-6)
        loss = -(advantage.detach() * log_prob).mean()
        return loss

    def critic_loss(self, imagined_traj: ImaginedTrajectory) -> torch.Tensor:
        """MSE between critic prediction and symlog of the lambda-return."""
        feats = imagined_traj.features()                  # [T, B, F]
        T, B = feats.shape[:2]
        rewards = imagined_traj.rewards                    # [T, B]
        continues = imagined_traj.continues                # [T, B]

        with torch.no_grad():
            values = self.target_critic(feats).squeeze(-1)     # [T, B]
            bootstrap = self.target_critic(feats[-1]).squeeze(-1)
            values = torch.cat([values, bootstrap.unsqueeze(0)], dim=0)
            returns = self.compute_lambda_returns(
                rewards, values, continues, self.gamma, self.lambda_)
            target = self.symlog(returns)                       # [T, B]

        pred = self.critic(feats).squeeze(-1)                   # [T, B]
        return F.mse_loss(pred, target)

    # ------------------------------------------------------------------
    # EMA target update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_target(self) -> None:
        """In-place EMA: ``target ← (1-τ)·target + τ·critic``."""
        tau = self.target_tau
        for tp, p in zip(self.target_critic.parameters(),
                         self.critic.parameters()):
            tp.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for tb, b in zip(self.target_critic.buffers(),
                         self.critic.buffers()):
            tb.copy_(b)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _policy_from_features(self, feats: torch.Tensor
                              ) -> torch.distributions.Distribution:
        logits = self.actor(feats)
        if self.act_type == "discrete":
            return torch.distributions.Categorical(logits=logits)
        return torch.distributions.Normal(loc=logits, scale=torch.ones_like(logits))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    from .world_model import Mamba2WorldModel

    torch.manual_seed(0)
    device = torch.device("cpu")

    # --- Build a tiny world model to manufacture an imagined trajectory ---
    obs_dim, act_dim = 6, 4
    wm = Mamba2WorldModel(obs_dim=obs_dim, act_dim=act_dim, d_model=32,
                          d_state=8, n_layers=2, stoch_dim=8,
                          n_categories=8).to(device)
    feat_dim = wm.d_model + wm.flat_z

    ac = DreamerActorCritic(state_dim=feat_dim, act_dim=act_dim,
                            hidden=64).to(device)
    print(f"DreamerActorCritic params="
          f"{sum(p.numel() for p in ac.parameters()):,}")

    # --- Symlog / symexp round-trip ---
    x = torch.randn(7) * 100.0
    err = (ac.symexp(ac.symlog(x)) - x).abs().max().item()
    print(f"symlog/symexp round-trip max-err = {err:.3e}")
    assert err < 1e-4, "symlog ∘ symexp must be near-identity"

    # --- Imagine a short trajectory ---
    s0 = wm.initial_state(batch_size=2, device=device)
    def _random_policy(s: WorldModelState) -> torch.Tensor:
        # World model's action_proj expects float actions of shape [B, act_dim].
        return torch.randint(0, act_dim, (s.h.shape[0],), device=device).float()

    traj = wm.imagine(s0, _random_policy, horizon=6)
    print("imagined:", traj.actions.shape, traj.rewards.shape,
          traj.continues.shape, traj.features().shape)
    assert traj.actions.shape == (6, 2)
    assert traj.rewards.shape == (6, 2)
    assert traj.continues.shape == (6, 2)
    assert traj.features().shape[:2] == (6, 2)

    # --- Lambda returns shape ---
    bootstrap = ac.target_critic(traj.features()[-1]).squeeze(-1)
    values = torch.cat([ac.target_critic(traj.features()).squeeze(-1),
                        bootstrap.unsqueeze(0)], dim=0)
    rets = ac.compute_lambda_returns(traj.rewards, values, traj.continues,
                                     ac.gamma, ac.lambda_)
    print("lambda-returns:", rets.shape)
    assert rets.shape == (6, 2)

    # --- Losses should be finite, scalar, and differentiable ---
    pi_loss = ac.actor_loss(traj)
    v_loss = ac.critic_loss(traj)
    print(f"actor_loss={pi_loss.item():.4f}  critic_loss={v_loss.item():.4f}")
    assert pi_loss.dim() == 0 and v_loss.dim() == 0
    assert torch.isfinite(pi_loss) and torch.isfinite(v_loss)
    (pi_loss + v_loss).backward()  # gradient flow check

    # --- EMA target update changes target weights but stays in sync ---
    pre = ac.target_critic[0].weight.detach().clone()
    ac.update_target()
    post = ac.target_critic[0].weight.detach().clone()
    delta = (post - pre).abs().mean().item()
    print(f"EMA |Δtarget| = {delta:.3e}  (expect > 0)")
    assert delta > 0.0, "target critic must move after update_target()"

    # --- Symlog baselines & discrete policy path ---
    dist = ac.policy(s0)
    print("policy:", type(dist).__name__,
          "logits shape:", dist.logits.shape)
    a = ac.act(s0)
    print("sampled action:", a.tolist())
    assert a.shape == (2,)

    print("actor_critic self-test ✓")

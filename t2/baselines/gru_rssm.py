"""GRURSSMWorldModel — Baseline DreamerV3 RSSM with GRU dynamics core.

Matched-parameter control for Mamba2WorldModel (Track 2 baseline). Swaps only
the dynamics core (Mamba-2 -> stacked nn.GRU), leaving the categorical
stochastic latent space, encoder, and prediction heads identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root workspace is in sys.path
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from t2.world_model import (
    WorldModelState,
    ImaginedTrajectory,
    _make_mlp,
    _sample_categorical,
    _flat,
)


class _GRUDynamicsCore(nn.Module):
    """Stacked nn.GRU dynamics core matching _DynamicsCore interface."""

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model * 2, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(seq)
        out, _ = self.gru(x)
        return self.norm(out)


class GRURSSMWorldModel(nn.Module):
    """Parameter-matched GRU-RSSM baseline model."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 256,
        d_state: int = 64,  # unused, kept for signature match
        n_layers: int = 4,
        stoch_dim: int = 32,
        n_categories: int = 32,
    ):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.d_model = d_model
        self.stoch_dim, self.n_categories = stoch_dim, n_categories
        self.flat_z = stoch_dim * n_categories

        self.encoder = _make_mlp(obs_dim, d_model, d_model)
        self.action_proj = nn.Linear(act_dim, d_model)
        self.dynamics = _GRUDynamicsCore(d_model, n_layers)
        self.prior_head = _make_mlp(d_model, d_model, self.flat_z)
        self.post_head = _make_mlp(d_model * 2, d_model, self.flat_z)
        self.decoder = _make_mlp(d_model + self.flat_z, d_model, obs_dim, depth=3)
        self.reward_head = _make_mlp(d_model + self.flat_z, d_model, 1)
        self.cont_head = _make_mlp(d_model + self.flat_z, d_model, 1)

    def initial_state(self, batch_size: int, device: torch.device | str) -> WorldModelState:
        h = torch.zeros(batch_size, self.d_model, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, self.n_categories, device=device)
        return WorldModelState(h, z)

    def step(
        self,
        prev: WorldModelState,
        action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[WorldModelState, torch.Tensor, torch.Tensor]:
        h_in = torch.cat([prev.h, self.action_proj(action)], dim=-1).unsqueeze(1)
        h_new = self.dynamics(h_in).squeeze(1)  # [B, D]
        prior_logits = self.prior_head(h_new).view(-1, self.stoch_dim, self.n_categories)

        if obs_embed is None:
            post_logits = prior_logits
        else:
            post_in = torch.cat([h_new, obs_embed], dim=-1)
            post_logits = self.post_head(post_in).view(-1, self.stoch_dim, self.n_categories)

        z_new = _sample_categorical(post_logits)
        return WorldModelState(h_new, z_new), prior_logits, post_logits

    def forward(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        reward_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        T, B = obs_seq.shape[:2]
        device = obs_seq.device
        state = self.initial_state(B, device)

        hs, zs, ps, qs = [], [], [], []
        recons, rps, cps = [], [], []
        embeds = self.encoder(obs_seq.flatten(0, 1)).view(T, B, -1)

        for t in range(T):
            state, p, q = self.step(state, act_seq[t], embeds[t])
            hs.append(state.h)
            zs.append(state.z)
            ps.append(p)
            qs.append(q)
            feat = torch.cat([state.h, _flat(state.z)], dim=-1)
            recons.append(self.decoder(feat))
            rps.append(self.reward_head(feat).squeeze(-1))
            cps.append(self.cont_head(feat).squeeze(-1))

        recon = torch.stack(recons)
        h = torch.stack(hs)
        z = torch.stack(zs)
        pL, qL = torch.stack(ps), torch.stack(qs)
        r_pred, c_pred = torch.stack(rps), torch.stack(cps)

        recon_loss = F.mse_loss(recon, obs_seq)
        reward_loss = F.mse_loss(r_pred, reward_seq)
        cont_loss = F.binary_cross_entropy_with_logits(c_pred, torch.ones_like(c_pred))
        log_q, log_p = F.log_softmax(qL, -1), F.log_softmax(pL, -1)
        kl = torch.clamp((log_q.exp() * (log_q - log_p)).sum(-1).sum(-1).mean(), min=0.0)
        total = recon_loss + reward_loss + cont_loss + 0.1 * kl

        return {
            "recon": recon_loss.detach(),
            "kl": kl.detach(),
            "reward": reward_loss.detach(),
            "continue": cont_loss.detach(),
            "total": total,
            "features": torch.cat([h, z.flatten(start_dim=2)], dim=-1),
        }

    @torch.no_grad()
    def imagine(
        self,
        initial_state: WorldModelState,
        policy_fn: Callable[[WorldModelState], torch.Tensor],
        horizon: int,
    ) -> ImaginedTrajectory:
        state = initial_state.detach()
        B, device = state.h.shape[0], state.h.device
        states, actions, rewards, conts = [state], [], [], []

        for _ in range(horizon):
            a = policy_fn(state)
            state, _, _ = self.step(state, a, None)
            feat = torch.cat([state.h, _flat(state.z)], dim=-1)
            actions.append(a)
            rewards.append(self.reward_head(feat).squeeze(-1))
            conts.append(torch.sigmoid(self.cont_head(feat).squeeze(-1)))
            states.append(state)

        return ImaginedTrajectory(
            states[1:],
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(conts),
        )


if __name__ == "__main__":
    device = torch.device("cpu")
    wm = GRURSSMWorldModel(obs_dim=8, act_dim=4, d_model=64, n_layers=2, stoch_dim=8, n_categories=8).to(device)
    print(f"GRURSSMWorldModel params = {sum(p.numel() for p in wm.parameters()):,}")

    T, B = 5, 3
    obs = torch.randn(T, B, 8)
    act = torch.randn(T, B, 4)
    rew = torch.randn(T, B)
    losses = wm(obs, act, rew)
    print("Forward losses:", {k: float(v.detach()) if torch.is_tensor(v) and v.dim() == 0 else tuple(v.shape) for k, v in losses.items()})

    s0 = wm.initial_state(B, device)
    s1, p, q = wm.step(s0, torch.zeros(B, 4), torch.randn(B, 64, device=device))
    print("Step:", s1.h.shape, s1.z.shape)

    traj = wm.imagine(s0, lambda s: torch.randn(s.h.shape[0], 4), horizon=4)
    print("Imagine:", traj.features().shape)

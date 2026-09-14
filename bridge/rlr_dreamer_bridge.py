"""bridge/rlr_dreamer_bridge.py — Track 1 (RLR) ↔ Track 2 (DreamerV3) Bridge.

Couples the RSSM recurrent state h_t with RLR latent looping.
Features:
- Projects RSSM h_t to RLR latent space
- Runs up to max_loops refinement iterations governed by ACTHaltingHead
- Supports 3 ablation modes: 'full' (active), 'frozen' (fixed weights), 'absent' (bypass)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from rlr.act_halting import ACTHaltingHead


class RLRDreamerBridge(nn.Module):
    """Bridge module wrapping RSSM state with RLR recursive latent scratchpad."""

    def __init__(
        self,
        rssm_state_dim: int,
        rlr_d_model: int = 128,
        max_loops: int = 8,
        tau: float = 0.0,
        ablation_mode: str = "full",
    ):
        super().__init__()
        assert ablation_mode in ("full", "frozen", "absent")
        self.rssm_state_dim = rssm_state_dim
        self.rlr_d_model = rlr_d_model
        self.max_loops = max_loops
        self._ablation_mode = ablation_mode

        self.in_proj = nn.Linear(rssm_state_dim, rlr_d_model)
        self.loop_cell = nn.GRUCell(rlr_d_model, rlr_d_model)
        self.out_proj = nn.Linear(rlr_d_model, rssm_state_dim)
        self.norm = nn.LayerNorm(rssm_state_dim)

        self.halting_head = ACTHaltingHead(
            d_model=rlr_d_model,
            tau_init=tau,
            tau_max=0.01,
        )

        if self._ablation_mode == "frozen":
            self._freeze_parameters()

    @property
    def ablation_mode(self) -> str:
        return self._ablation_mode

    @ablation_mode.setter
    def ablation_mode(self, mode: str) -> None:
        assert mode in ("full", "frozen", "absent")
        self._ablation_mode = mode
        if mode == "frozen":
            self._freeze_parameters()
        elif mode == "full":
            for p in self.parameters():
                p.requires_grad = True

    def _freeze_parameters(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self, h_t: torch.Tensor, return_cost: bool = False
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, torch.Tensor]:
        """Refine RSSM state h_t through recursive latent scratchpad.

        Parameters
        ----------
        h_t: [B, rssm_state_dim]
        return_cost: bool
            If True, also returns the differentiable ACT ponder cost.

        Returns
        -------
        augmented_h: [B, rssm_state_dim]
        n_loops_used: int
        ponder_cost: torch.Tensor (only if return_cost=True)
        """
        if self._ablation_mode == "absent":
            zero_cost = torch.zeros((), device=h_t.device)
            self.last_ponder_cost = zero_cost
            if return_cost:
                return h_t, 0, zero_cost
            return h_t, 0

        B = h_t.shape[0]
        x = self.in_proj(h_t)
        h_loop = torch.zeros(B, self.rlr_d_model, device=h_t.device, dtype=h_t.dtype)

        cum_rho = torch.zeros(B, device=h_t.device)
        loops_used = 0
        rho_history = []

        for loop_idx in range(self.max_loops):
            loops_used += 1
            h_loop = self.loop_cell(x, h_loop)
            rho = self.halting_head(h_loop)
            rho_history.append(rho)
            cum_rho = cum_rho + rho

            if self.halting_head.should_halt(cum_rho):
                break

        ponder_cost = self.halting_head.compute_ponder_cost(rho_history)
        self.last_ponder_cost = ponder_cost

        residual = self.out_proj(h_loop)
        out = self.norm(h_t + residual)

        if return_cost:
            return out, loops_used, ponder_cost
        return out, loops_used


if __name__ == "__main__":
    device = torch.device("cpu")
    bridge = RLRDreamerBridge(rssm_state_dim=64, rlr_d_model=32, max_loops=6, tau=0.01).to(device)

    h = torch.randn(4, 64)
    out_full, loops_full = bridge(h)
    print(f"Bridge [full]: output shape={out_full.shape}, loops used={loops_full}")

    bridge.ablation_mode = "frozen"
    out_frozen, loops_frozen = bridge(h)
    print(f"Bridge [frozen]: output shape={out_frozen.shape}, loops used={loops_frozen}")

    bridge.ablation_mode = "absent"
    out_absent, loops_absent = bridge(h)
    print(f"Bridge [absent]: output shape={out_absent.shape}, loops used={loops_absent}")
    assert torch.equal(out_absent, h)
    assert loops_absent == 0
    print("RLRDreamerBridge self-test complete.")

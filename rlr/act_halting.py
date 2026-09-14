"""
rlr/act_halting.py — F2 Fix: ACT-Style Halting with Ponder Cost
==============================================================

Reference
---------
Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural
Networks.* arXiv:1603.08983.

The classic ACT halting head emits a scalar ``ρ ∈ (0, 1)`` per loop and
halts as soon as the cumulative halting probability exceeds ``1 - ε``.
Training adds a *ponder cost* ``τ · Σ ρ`` to the task loss so that the
model pays for every extra loop it consumes.

This module is *self-contained* and can be plugged in front of any
``nn.Module`` that exposes a per-loop hidden state ``[B, D]``. In the
RLF engine the natural input is the post-norm hidden state at the
answer-start position (loop output before ``lm_head``).

Differences from the RLF engine's hard-HALT logic
--------------------------------------------------
- Soft signal: each loop gets a learned probability rather than a
  threshold on the § token.
- Ponder cost: training explicitly penalises loops spent beyond the
  minimum necessary.
- Adaptive τ: τ is grown from 0 → ``tau_max`` once the model plateaus,
  matching the "wait for the model to be competent before charging rent"
  schedule from the ACT paper.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional


class ACTHaltingHead(nn.Module):
    """Halting head that emits ρ ∈ (0, 1) per loop and tracks ACT ponder cost.

    Architecture
    ------------
    ``d_model → d_model // 4 → GELU → 1 → Sigmoid``

    The final layer is initialised with ``bias = -2.0`` so the sigmoid
    starts at ≈ 0.119. This deliberately biases the head *toward
    continuing* — the model must learn to halt, not the other way around.

    Parameters
    ----------
    d_model:
        Hidden size of the model emitting the per-loop hidden state.
    tau_init:
        Initial ponder-cost coefficient τ (warm-up start). Default 0.0.
    tau_max:
        Asymptotic τ (rent charge once plateau is detected).
    warmup_steps:
        Minimum number of optimisation steps before τ is allowed to grow.
    """

    def __init__(
        self,
        d_model: int,
        tau_init: float = 0.0,
        tau_max: float = 0.01,
        warmup_steps: int = 1000,
        hidden_ratio: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if not 0.0 <= tau_init <= tau_max:
            raise ValueError(
                f"tau_init ({tau_init}) must lie in [0, tau_max ({tau_max})]"
            )
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.d_model = d_model
        self.tau_init = float(tau_init)
        self.tau_max = float(tau_max)
        self.warmup_steps = int(warmup_steps)

        hidden = max(1, d_model // hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Initialise final layer to bias toward "continue":
        # σ(-2) ≈ 0.119 → ρ starts small.
        with torch.no_grad():
            last_linear = self.net[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.constant_(last_linear.bias, -2.0)

        # Mutable tau (not a Parameter — scheduled by the trainer).
        self.register_buffer("_tau", torch.tensor(self.tau_init, dtype=torch.float32))
        self.register_buffer(
            "_step_counter", torch.tensor(0, dtype=torch.long),
            persistent=False,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute ρ ∈ (0, 1) for one loop.

        Parameters
        ----------
        hidden_state:
            ``[B, D]`` per-sample loop summary. If a 3-D ``[B, T, D]``
            tensor is supplied it is mean-pooled over ``T`` (so the
            caller can pass a sequence in directly).

        Returns
        -------
        rho:
            ``[B]`` tensor of halting probabilities in ``(0, 1)``.
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)
        elif hidden_state.dim() != 2:
            raise ValueError(
                f"hidden_state must be [B, D] or [B, T, D], "
                f"got shape {tuple(hidden_state.shape)}"
            )
        logit = self.net(hidden_state.float()).squeeze(-1)
        rho = torch.sigmoid(logit)
        return rho

    # ──────────────────────────────────────────────────────────────────────────
    # Ponder cost
    # ──────────────────────────────────────────────────────────────────────────

    def compute_ponder_cost(
        self, rho_history: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Ponder cost ``τ · Σ ρ`` over the loops already executed.

        Parameters
        ----------
        rho_history:
            Iterable of ``[B]`` tensors, one per executed loop, in
            chronological order.

        Returns
        -------
        cost:
            Scalar ``Tensor`` (differentiable w.r.t. ``rho_history``).
        """
        rhos = list(rho_history)
        if not rhos:
            return torch.zeros((), device=self._tau.device, dtype=torch.float32)
        stacked = torch.stack([r.float() for r in rhos], dim=0)        # [L, B]
        per_sample = stacked.sum(dim=0)                                 # [B]
        return self._tau * per_sample.mean()

    # ──────────────────────────────────────────────────────────────────────────
    # Halting decision
    # ──────────────────────────────────────────────────────────────────────────

    def should_halt(
        self,
        cumulative_rho: torch.Tensor | float,
        epsilon: float = 0.01,
    ) -> bool:
        """Test whether the cumulative halting probability has saturated.

        Parameters
        ----------
        cumulative_rho:
            Scalar or ``[B]`` tensor — sum of ρ over loops so far.
        epsilon:
            Residual mass tolerated after halting. Default 0.01 (ACT
            convention).

        Returns
        -------
        halt:
            ``True`` iff *every* sample in the batch satisfies
            ``cumulative_rho ≥ 1 - ε``.
        """
        if isinstance(cumulative_rho, torch.Tensor):
            sat = cumulative_rho >= (1.0 - epsilon)
            if sat.dim() == 0:
                return bool(sat.item())
            return bool(sat.all().item())
        return float(cumulative_rho) >= (1.0 - epsilon)

    # ──────────────────────────────────────────────────────────────────────────
    # Adaptive τ scheduler
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_tau(
        self,
        step: int,
        answer_acc: float,
        baseline_acc: float,
        n_stable_steps: int = 200,
        plateau_patience: int = 3,
        acc_tolerance: float = 0.005,
        growth: float = 1.5,
    ) -> float:
        """Linearly grow τ once answer accuracy plateaus near ``baseline_acc``.

        Parameters
        ----------
        step:
            Current global training step.
        answer_acc:
            Current answer-token accuracy (in ``[0, 1]``).
        baseline_acc:
            Target accuracy at which the model is "competent" enough to
            start paying rent. Typically the SFT recovery target.
        n_stable_steps:
            Sliding window over which plateau detection runs.
        plateau_patience:
            Number of consecutive ``n_stable_steps`` windows in which the
            rolling accuracy stayed within ``acc_tolerance`` of
            ``baseline_acc`` before τ is grown.
        acc_tolerance:
            Maximum deviation from ``baseline_acc`` that still counts as
            "stable". Default 0.5 percentage points.
        growth:
            Multiplicative growth applied to τ on every plateau tick.

        Returns
        -------
        tau:
            The current value of τ after the update.
        """
        self._step_counter += 1
        # Reset everything before the warmup window.
        if step < self.warmup_steps:
            self._tau.fill_(self.tau_init)
            return float(self._tau.item())

        # Maintain a small rolling buffer of (step, acc).
        if not hasattr(self, "_acc_buffer"):
            self._acc_buffer: list[tuple[int, float]] = []
        self._acc_buffer.append((int(step), float(answer_acc)))
        # Drop entries older than n_stable_steps × plateau_patience.
        horizon = n_stable_steps * plateau_patience
        if self._acc_buffer and (step - self._acc_buffer[0][0]) > horizon:
            self._acc_buffer = self._acc_buffer[-horizon:]

        # Split the buffer into plateau_patience slices; check stability.
        if len(self._acc_buffer) < horizon:
            return float(self._tau.item())

        slice_size = n_stable_steps
        slices = [
            self._acc_buffer[i : i + slice_size]
            for i in range(0, len(self._acc_buffer), slice_size)
            if len(self._acc_buffer[i : i + slice_size]) == slice_size
        ]
        if len(slices) < plateau_patience:
            return float(self._tau.item())

        stable = True
        for s in slices[-plateau_patience:]:
            mean_acc = sum(a for _, a in s) / slice_size
            if abs(mean_acc - baseline_acc) > acc_tolerance:
                stable = False
                break

        if stable:
            cur_tau = float(self._tau.item())
            # Seed event: τ is still at the (possibly-zero) init value.
            # Multiplying by ``growth`` from zero would leave τ at zero
            # forever, so the first stable plateau *seeds* τ at
            # ``tau_init + growth`` (capped at ``tau_max``).
            if cur_tau <= self.tau_init + 1e-12:
                new_tau = min(self.tau_max, self.tau_init + growth * 1e-3)
            else:
                new_tau = min(self.tau_max, cur_tau * growth)
            self._tau.fill_(new_tau)
        return float(self._tau.item())

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def tau(self) -> float:
        return float(self._tau.item())

    def initial_bias_check(self) -> float:
        """Return the sigmoid of the bias of the final layer — should be ≈0.119."""
        last_linear = self.net[-1]
        if isinstance(last_linear, nn.Linear):
            return float(torch.sigmoid(last_linear.bias).item())
        return float("nan")

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, tau={self.tau:.4f}, "
            f"tau_max={self.tau_max}, warmup_steps={self.warmup_steps}"
        )

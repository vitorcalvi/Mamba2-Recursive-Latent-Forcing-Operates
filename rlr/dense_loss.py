"""
rlr/dense_loss.py — F1 Fix: Dense Per-Loop Supervision
=======================================================

Reference
---------
Adaptive Computation Time (ACT) / step-wise supervision is discussed in
Graves (2016). The "predicting HALT early costs intermediate-token CE at
remaining positions" trick is the per-step dense supervision from SELR
(Selective Early-exit for Latent Reasoning, arXiv:2608.13570 §3.2).

Problem
-------
The original ``rlf_engine_1_4b.py`` training branch computes
``avg_loss = mean(step_losses)``. When ``HALT_ID`` is predicted at loop 0
the remaining loops still fire ``F.cross_entropy`` against ``HALT_ID`` (the
chain target fallback ``chain_targets[b][min(loop_i, len-1)]``) which
yields *zero* gradient for the intermediate symbols the model never got
a chance to emit. The result is a per-sample supervision signal that
collapses to "predict § fast" rather than "predict the right intermediate
symbols".

Fix
---
``DensePerLoopLoss`` takes the full per-loop logit stack and the per-sample
*ground-truth* chain ``chain_targets[b]``. It then assigns each loop a
*specific* target token (intermediate symbol at loop 0..k-2, ``HALT_ID`` at
loop k-1) and accrues CE loss for *every* loop position. Predicting §
early means losing intermediate-symbol CE at the loops that were skipped
— the model can no longer free-ride on the HALT shortcut.

Usage
-----
>>> from rlr.dense_loss import DensePerLoopLoss
>>> criterion = DensePerLoopLoss(ignore_index=-100)
>>> loss, acc, halt_acc = criterion(logits_stacked, chain_targets, ans_starts,
...                                 loop_index=k, halt_id=7803)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class DensePerLoopLoss(nn.Module):
    """Per-loop cross-entropy with a hard HALT shortcut penalty.

    Parameters
    ----------
    ignore_index:
        Label id that is masked out of the loss / accuracy (mirrors
        ``F.cross_entropy`` semantics). Default ``-100``.
    reduction:
        ``"mean"`` averages across (valid_sample × loop). ``"sum"``
        accumulates raw totals. ``"none"`` returns a ``[N_valid]`` vector.
    label_smoothing:
        Forwarded to ``F.cross_entropy``. Default ``0.0``.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        logits: torch.Tensor,
        chain_targets: Sequence[Sequence[int]],
        ans_starts: Sequence[int],
        loop_index: int,
        halt_id: int,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute dense per-loop CE.

        Parameters
        ----------
        logits:
            ``[B, T, V]`` — model logits at the *answer-start* position for
            a *single* loop pass. The dense supervision loop is expected to
            be driven externally: caller calls ``forward`` once per loop
            with ``loop_index`` set to the current loop counter.
        chain_targets:
            ``list[list[int]]`` — per-sample ground-truth chain of token
            ids. ``chain_targets[b][i]`` is the symbol the model must emit
            at loop ``i``. The list length ``k`` is the loop budget for
            sample ``b``; loop ``k`` (== ``len(chain_targets[b])``) is the
            implicit HALT position.
        ans_starts:
            ``list[int]`` — answer start column ``as_`` for each sample.
            CE is taken at ``logits[b, as_ - 1, :]``.
        loop_index:
            Current loop counter ``i`` in ``0 .. k``. The target assigned
            at loop ``i`` is ``chain_targets[b][min(i, k-1)]``; at loop
            ``i == k`` the caller must invoke ``forward`` once more with
            ``loop_index=k`` and the target is forced to ``halt_id``.
        halt_id:
            Token id used as HALT (default 7803 = §).

        Returns
        -------
        (loss, accuracy, halt_accuracy):
            - ``loss``: scalar ``Tensor`` under the configured reduction
            - ``accuracy``: float in ``[0, 1]`` — token-level accuracy at
              the supervised position
            - ``halt_accuracy``: float in ``[0, 1]`` — accuracy *only* on
              samples where the assigned target is ``halt_id``
        """
        if logits.dim() != 3:
            raise ValueError(
                f"logits must be [B, T, V], got shape {tuple(logits.shape)}"
            )
        B, T, V = logits.shape
        if len(chain_targets) != B:
            raise ValueError(
                f"chain_targets length {len(chain_targets)} != batch size {B}"
            )
        if len(ans_starts) != B:
            raise ValueError(
                f"ans_starts length {len(ans_starts)} != batch size {B}"
            )

        device = logits.device
        dtype = logits.dtype

        # Per-sample targets for this loop.
        targets_list: list[int] = []
        valid_mask:    list[bool] = []
        for b in range(B):
            chain = chain_targets[b]
            k = len(chain)
            if k == 0:
                valid_mask.append(False)
                targets_list.append(self.ignore_index)
                continue
            if loop_index < k:
                tid = int(chain[loop_index])
            else:
                # We're past the chain → HALT must be emitted.
                tid = int(halt_id)
            as_ = ans_starts[b]
            if as_ < 1 or as_ >= T or tid >= V:
                valid_mask.append(False)
                targets_list.append(self.ignore_index)
            else:
                valid_mask.append(True)
                targets_list.append(tid)

        targets = torch.tensor(targets_list, device=device, dtype=torch.long)

        # Gather the supervised rows: logits[b, ans_starts[b]-1, :].
        rows = torch.tensor(
            [max(0, ans_starts[b] - 1) for b in range(B)],
            device=device,
            dtype=torch.long,
        )
        batch_idx = torch.arange(B, device=device, dtype=torch.long)
        selected = logits[batch_idx, rows, :]                          # [B, V]
        # Promote to float32 for numerical stability of CE in bf16/fp16.
        ce_per_sample = F.cross_entropy(
            selected.float(),
            targets,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )                                                              # [B]

        # Mask out invalid positions.
        valid = torch.tensor(valid_mask, device=device, dtype=torch.bool)
        n_valid = int(valid.sum().item())

        # ── Reduction ────────────────────────────────────────────────────────
        if self.reduction == "mean":
            if n_valid == 0:
                loss = torch.zeros((), device=device, dtype=dtype)
            else:
                loss = ce_per_sample[valid].sum() / max(n_valid, 1)
        elif self.reduction == "sum":
            loss = ce_per_sample[valid].sum() if n_valid > 0 else torch.zeros(
                (), device=device, dtype=dtype
            )
        else:  # "none"
            loss = ce_per_sample

        # ── Accuracy stats ───────────────────────────────────────────────────
        with torch.no_grad():
            preds = selected.argmax(dim=-1)                            # [B]
            if n_valid == 0:
                accuracy = 0.0
            else:
                accuracy = float(
                    (preds[valid] == targets[valid]).float().mean().item()
                )
            halt_mask = valid & (targets == halt_id)
            n_halt = int(halt_mask.sum().item())
            if n_halt == 0:
                halt_accuracy = 0.0
            else:
                halt_accuracy = float(
                    (preds[halt_mask] == halt_id).float().mean().item()
                )

        return loss, accuracy, halt_accuracy

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: dense loss over a *stack* of per-loop logits
    # ──────────────────────────────────────────────────────────────────────────

    def over_loops(
        self,
        logits_per_loop: Sequence[torch.Tensor],
        chain_targets: Sequence[Sequence[int]],
        ans_starts: Sequence[int],
        halt_id: int,
    ) -> tuple[torch.Tensor, float, float]:
        """Aggregate ``forward`` over an externally provided loop stack.

        Parameters
        ----------
        logits_per_loop:
            ``Sequence`` of length ``L``; each entry is ``[B, T, V]``. The
            loop index passed to ``forward`` is the position in the
            sequence — *not* the loop counter inside the RLF engine.
        chain_targets, ans_starts, halt_id:
            Forwarded to :meth:`forward`.

        Returns
        -------
        (avg_loss, avg_accuracy, halt_accuracy):
            ``avg_loss`` is the mean of per-loop scalar losses (matches
            the ``mean(step_losses)`` semantics of the original engine
            but with the F1 fix applied). ``avg_accuracy`` is the mean
            over loops of the per-loop token accuracy.
        """
        if not logits_per_loop:
            raise ValueError("logits_per_loop is empty")

        losses, accs, halt_accs = [], [], []
        for i, lg in enumerate(logits_per_loop):
            loss, acc, halt_acc = self(lg, chain_targets, ans_starts, i, halt_id)
            losses.append(loss)
            accs.append(acc)
            halt_accs.append(halt_acc)

        loss_stack = torch.stack([l for l in losses if l.requires_grad or True])
        # Filter out zero-tensor placeholders without grads by detecting NaN-safe.
        valid_losses = [l for l in losses if torch.is_tensor(l) and l.numel() == 1]
        if valid_losses:
            avg_loss = torch.stack(valid_losses).mean()
        else:
            avg_loss = torch.zeros((), device=logits_per_loop[0].device)
        avg_acc = sum(accs) / max(len(accs), 1)
        avg_halt_acc = sum(halt_accs) / max(len(halt_accs), 1)
        return avg_loss, avg_acc, avg_halt_acc

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, reduction={self.reduction!r}, "
            f"label_smoothing={self.label_smoothing}"
        )

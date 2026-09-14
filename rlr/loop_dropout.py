"""
rlr/loop_dropout.py — Per-Sample Loop Budget Regulariser
========================================================

Background
----------
The RLF engine has ``MAX_LOOPS = 6``. Empirically we observe that the
model learns to *key* on the loop index: by loop 4 it has memorised the
"shape" of iteration-4 hidden states and can short-circuit the reasoning
chain. This breaks the F2 fix because the ACT head sees a degenerate
distribution.

Fix
---
For every training batch, sample a per-sample loop budget
``k_b ~ Uniform(min_loops, max_loops)``. The training loop then runs
exactly ``k_b`` iterations for sample ``b``, and the model is forced to
solve the task in *any* number of loops in the allowed range. This is
the loop-index analogue of dropout: at train time the "depth channel"
is randomly corrupted, at eval time it is left intact.

The class is a plain ``nn.Module`` so it integrates with ``.to(device)``
and ``.train() / .eval()`` automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class LoopDropout(nn.Module):
    """Per-sample loop budget sampler with training/eval split.

    Parameters
    ----------
    max_loops:
        Upper bound of the loop budget (also returned at eval time).
    min_loops:
        Lower bound of the loop budget at train time. Default 1.
    deterministic:
        If ``True`` (and ``self.training`` is ``True``) sample every
        sample in the batch with the same budget, drawn once per call.
        Useful for unit tests. Default ``False``.
    """

    def __init__(
        self,
        max_loops: int,
        min_loops: int = 1,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if max_loops < 1:
            raise ValueError(f"max_loops must be >= 1, got {max_loops}")
        if min_loops < 1:
            raise ValueError(f"min_loops must be >= 1, got {min_loops}")
        if min_loops > max_loops:
            raise ValueError(
                f"min_loops ({min_loops}) cannot exceed max_loops ({max_loops})"
            )

        self.max_loops = int(max_loops)
        self.min_loops = int(min_loops)
        self.deterministic = bool(deterministic)

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return per-sample loop budget.

        Parameters
        ----------
        batch_size:
            Number of samples ``B`` in the current batch.

        Returns
        -------
        max_loops_per_sample:
            ``LongTensor[B]`` with values in ``[min_loops, max_loops]``
            at train time, and ``[max_loops] * B`` at eval time.

        Notes
        -----
        The sampler ignores ``torch.Generator`` for portability — if you
        need reproducibility pass a generator via :meth:`sample` and
        call it yourself.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        device = self._device_hint()

        if self.training:
            if self.deterministic:
                budget = int(
                    torch.randint(
                        low=self.min_loops,
                        high=self.max_loops + 1,
                        size=(1,),
                    ).item()
                )
                return torch.full(
                    (batch_size,), budget, dtype=torch.long, device=device,
                )
            return torch.randint(
                low=self.min_loops,
                high=self.max_loops + 1,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
        # Eval → every sample gets the full budget.
        return torch.full(
            (batch_size,), self.max_loops, dtype=torch.long, device=device,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Explicitly stochastic sample (always uses the train-time range)."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        device = self._device_hint()
        return torch.randint(
            low=self.min_loops,
            high=self.max_loops + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

    def max_for_eval(self, batch_size: int) -> torch.Tensor:
        """Return the eval-time budget (always ``max_loops``)."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        device = self._device_hint()
        return torch.full(
            (batch_size,), self.max_loops, dtype=torch.long, device=device,
        )

    def _device_hint(self) -> torch.device:
        """Best-effort device lookup from the module parameters."""
        for p in self.parameters():
            return p.device
        # No parameters — fall back to CPU.
        return torch.device("cpu")

    def extra_repr(self) -> str:
        return (
            f"min_loops={self.min_loops}, max_loops={self.max_loops}, "
            f"deterministic={self.deterministic}, "
            f"training={self.training}"
        )

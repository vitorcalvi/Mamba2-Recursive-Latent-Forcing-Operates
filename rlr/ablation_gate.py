"""
rlr/ablation_gate.py — F3 Fix: Scratchpad Utilisation Gate
==========================================================

Problem
-------
The RLF engine's gain comes from the *prefix latent scratchpad* — the
``[1, 8, 2048]`` ``latent_memory`` parameter that is prepended to every
forward pass. If the model learns to route information through the
backbone *without ever touching the scratchpad*, the F1/F2 fixes are
irrelevant: there is no reasoning happening, only a parametric shortcut
from the prompt to the answer.

Fix
---
``ScratchpadAblationGate`` runs the model twice on a held-out batch:

  1. **Full pass**:    ``forward(input_ids, …)`` returning normal metrics.
  2. **Ablated pass**: same forward but with ``latent_memory`` temporarily
     zeroed in-place (under ``torch.no_grad``).

If the metric drop ``Δ = score_full − score_ablated`` is below
``threshold_delta`` the scratchpad is being ignored and the gate *fails*.
A passing gate is a precondition for promoting the model out of phase 3b.

The gate is intentionally implemented as a plain Python class (no
``nn.Module``) because it is not a differentiable component of the
training graph — it is a release check.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ────────────────────────────────────────────────────────────────────────────
# Result container
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Outcome of one ``ScratchpadAblationGate.evaluate`` invocation."""

    score_full: float
    score_ablated: float
    delta: float
    passed: bool
    threshold_delta: float
    n_batches: int
    per_batch_full: list[float] = field(default_factory=list)
    per_batch_ablated: list[float] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _zeroed_parameter(p: nn.Parameter):
    """Temporarily replace a parameter's data with zeros (no grad)."""
    if not isinstance(p, nn.Parameter):
        raise TypeError(f"expected nn.Parameter, got {type(p).__name__}")
    saved = p.data
    p.data = torch.zeros_like(saved)
    try:
        yield
    finally:
        p.data = saved


def _default_metric_extractor(
    forward_fn: Callable[..., Any],
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> float:
    """Forward the model and extract a scalar accuracy.

    This handles three return shapes used in the RLF codebase:

      * ``(avg_loss, avg_acc, ans_acc, halt_acc)`` — training branch
      * ``(n_loops, trace, last)``                  — inference branch
      * plain scalar/tensor                        — anything else

    The RLF engine's ``forward(self, input_ids, chain_targets=None,
    ans_starts=None)`` takes the *batch* as its first (and only) input.
    The bound method ``model.forward`` already includes ``self``, so the
    forward call is simply ``forward_fn(batch)``. We also support the
    raw-form ``forward_fn(model, batch, device)`` for advanced users
    who pass an unbound method or a custom callable.
    """
    try:
        out = forward_fn(batch)
    except TypeError:
        # Unbound method or custom signature — fall back to the 3-arg form.
        out = forward_fn(model, batch, device)
    if isinstance(out, tuple):
        if len(out) >= 4 and isinstance(out[2], (int, float, torch.Tensor)):
            # Training tuple — index 2 is ``ans_acc``.
            acc = float(out[2].item() if isinstance(out[2], torch.Tensor) else out[2])
            return acc
        if len(out) == 3 and isinstance(out[0], int):
            # Inference tuple — use 1 / max(1, n_loops) as a smoothness proxy
            # so that a model that halts in 1 loop scores 1.0 and a model
            # that always uses MAX_LOOPS scores 1/MAX_LOOPS.
            return 1.0 / max(1, int(out[0]))
        if len(out) == 1:
            return float(out[0].item() if isinstance(out[0], torch.Tensor) else out[0])
    if isinstance(out, torch.Tensor):
        return float(out.item())
    return float(out)


def _move_batch(batch: Any, device: torch.device) -> Any:
    """Move a batch's tensors to ``device`` (in-place is impossible for tuples)."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_move_batch(v, device) for v in batch]
        return type(batch)(moved)
    return batch


# ────────────────────────────────────────────────────────────────────────────
# Gate
# ────────────────────────────────────────────────────────────────────────────


class ScratchpadAblationGate:
    """Release check that the prefix scratchpad is actually being used.

    Parameters
    ----------
    threshold_delta:
        Minimum acceptable accuracy drop when the scratchpad is zeroed.
        Default 10.0 (10 percentage points). Below this, the gate fails.
    max_batches:
        Hard cap on the number of eval batches to consume. ``None`` means
        drain the dataloader.
    """

    def __init__(
        self,
        threshold_delta: float = 10.0,
        max_batches: Optional[int] = None,
    ) -> None:
        if threshold_delta < 0:
            raise ValueError(
                f"threshold_delta must be >= 0, got {threshold_delta}"
            )
        self.threshold_delta = float(threshold_delta)
        self.max_batches = max_batches

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model: nn.Module,
        eval_dataloader: DataLoader | Iterable[Any],
        device: Optional[torch.device | str] = None,
        metric_fn: Optional[Callable[..., float]] = None,
    ) -> dict[str, Any]:
        """Run the ablation evaluation.

        Parameters
        ----------
        model:
            The RLF model. Must expose ``forward`` and either a
            ``latent_memory`` parameter *or* a parameter whose name
            contains ``"latent_memory"`` or ``"scratchpad"``.
        eval_dataloader:
            Any iterable yielding batches. ``DataLoader`` is the common
            case but the gate accepts plain iterables for testing.
        device:
            Target device. If ``None`` the model's device is used.
        metric_fn:
            Optional ``Callable(model, batch, device) -> float``.
            Defaults to :func:`_default_metric_extractor` which knows
            about the RLF training and inference return shapes.

        Returns
        -------
        result_dict:
            :class:`GateResult` serialised to a dict.
        """
        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)
        metric_fn = metric_fn or _default_metric_extractor

        latent_param = self._find_latent_memory(model)
        if latent_param is None:
            return GateResult(
                score_full=0.0,
                score_ablated=0.0,
                delta=0.0,
                passed=False,
                threshold_delta=self.threshold_delta,
                n_batches=0,
                notes="no latent_memory parameter found on model",
            ).to_dict()

        was_training = model.training
        model.eval()
        scores_full:    list[float] = []
        scores_ablated: list[float] = []

        forward_fn = model.forward

        with torch.no_grad():
            for b_idx, batch in enumerate(eval_dataloader):
                if self.max_batches is not None and b_idx >= self.max_batches:
                    break
                batch = _move_batch(batch, device)

                # Full pass.
                score_full = metric_fn(forward_fn, model, batch, device)
                scores_full.append(float(score_full))

                # Ablated pass.
                with _zeroed_parameter(latent_param):
                    score_ablate = metric_fn(forward_fn, model, batch, device)
                scores_ablated.append(float(score_ablate))

        if was_training:
            model.train()

        if not scores_full:
            return GateResult(
                score_full=0.0,
                score_ablated=0.0,
                delta=0.0,
                passed=False,
                threshold_delta=self.threshold_delta,
                n_batches=0,
                notes="dataloader yielded no batches",
            ).to_dict()

        mean_full = sum(scores_full) / len(scores_full)
        mean_ablate = sum(scores_ablated) / len(scores_ablated)
        # ``delta`` is reported in percentage points (× 100).
        delta_pp = (mean_full - mean_ablate) * 100.0
        passed = delta_pp >= self.threshold_delta
        notes = (
            "PASS: scratchpad contributes at least "
            f"{self.threshold_delta:.2f}pp of accuracy"
            if passed else
            "FAIL: scratchpad contribution below threshold "
            f"({delta_pp:.2f}pp < {self.threshold_delta:.2f}pp) — "
            "model may be ignoring latent_memory"
        )

        self._print_table(scores_full, scores_ablated, mean_full, mean_ablate,
                          delta_pp, passed)

        return GateResult(
            score_full=mean_full,
            score_ablated=mean_ablate,
            delta=delta_pp,
            passed=passed,
            threshold_delta=self.threshold_delta,
            n_batches=len(scores_full),
            per_batch_full=scores_full,
            per_batch_ablated=scores_ablated,
            notes=notes,
        ).to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_latent_memory(model: nn.Module) -> Optional[nn.Parameter]:
        """Locate the latent scratchpad parameter."""
        # Direct attribute first.
        if hasattr(model, "latent_memory") and isinstance(
            model.latent_memory, nn.Parameter
        ):
            return model.latent_memory
        # Fallback: named-parameter scan.
        for name, p in model.named_parameters():
            lname = name.lower()
            if "latent_memory" in lname or "scratchpad" in lname:
                return p
        return None

    @staticmethod
    def _print_table(
        scores_full: list[float],
        scores_ablated: list[float],
        mean_full: float,
        mean_ablate: float,
        delta_pp: float,
        passed: bool,
    ) -> None:
        """Emit a readable comparison table to stdout."""
        n = len(scores_full)
        if n == 0:
            return
        header = f"{'batch':>6} | {'full':>10} | {'ablated':>10} | {'Δ (pp)':>10}"
        sep    = "-" * len(header)
        print(f"\n{sep}\n ScratchpadAblationGate\n{sep}")
        print(header)
        print(sep)
        # Show at most 16 rows to avoid flooding the log.
        show = min(16, n)
        for i in range(show):
            f = scores_full[i]
            a = scores_ablated[i]
            print(
                f"{i:>6d} | {f * 100:>9.3f}% | {a * 100:>9.3f}% | "
                f"{(f - a) * 100:>+9.3f}"
            )
        if n > show:
            print(f"  ... ({n - show} more batches omitted)")
        print(sep)
        print(
            f"{'mean':>6} | {mean_full * 100:>9.3f}% | "
            f"{mean_ablate * 100:>9.3f}% | {delta_pp:>+9.3f}"
        )
        print(sep)
        verdict = "✅ PASS" if passed else "❌ FAIL"
        print(f" Verdict: {verdict}\n")

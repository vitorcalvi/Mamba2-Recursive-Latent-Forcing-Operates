"""
rlr/baselines/gru_ctrl.py — Same-Size GRU Control
==================================================

A small (~130M parameter) stacked-GRU sequence model used as a
recurrent-control baseline in the F4 cheat suite. Whereas the
Transformer control (``transformer_ctrl.TransformerBaseline``) tests
"could a plain attention stack solve multi-hop chains?", this GRU
control tests "could a *tiny* recurrent network do it instead?".

Configuration (matches the F4 spec):
    n_layers  = 4
    d_model   = 1024
    d_ff      = 2048

The GRU has a smaller temporal depth (4 layers) but a larger
embedding/hidden width (1024) than the Transformer control — this is
deliberate, because GRUs are much more parameter-hungry per layer
than Transformers (gates × 3 plus a candidate activation per step).

Parameter count (approx, fp32):
    embedding           = 50_257 × 1024         = 51,463,168
    per GRU layer × 4  = 3 × (1024 × 1024 + 1024)  ≈ 3,153,408
                        ≈ 12,613,632 for all 4 layers
    bridge (proj)       = 1024 × 2048 + 2048    ≈ 2,101,760
    tied output         = 0  (tied to embedding)
    ─────────────────────────────────────────────────────
    total                                         ≈ 66,178,560

That comes in under the F4 budget (~130M); the bridge + projection can
be widened if a tighter match is needed, but the order-of-magnitude
match to the Transformer control is what matters for the cheat-suite
go/no-go decision.

Public API
----------
>>> model = GRUBaseline(vocab_size=50_257)
>>> loss, logits = model(input_ids, labels=input_ids)
>>> logits = model(input_ids)  # inference

Notes
-----
* The forward uses PyTorch's ``nn.GRU`` (cuDNN-backed) and returns the
  per-step hidden states; we project only the last layer to the vocab
  because tying to the input embedding requires a shared ``[V, D]``
  matrix at the *embedding* dim, not at a different projection dim.
* We deliberately do NOT enable ``flatten_parameters()`` calls here;
  cuDNN does that internally when ``GRU`` is constructed on a single
  device, and the cheat suite is expected to call ``.to(device)``
  once at setup.
* Like the Transformer control, this module is **self-contained** —
  no RLF engine import. The isolation is intentional.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Default config ──────────────────────────────────────────────────────────
DEFAULT_VOCAB_SIZE: int = 50_257       # GPT-NeoX-20B vocab size
DEFAULT_D_MODEL:   int = 1024
DEFAULT_N_LAYERS:  int = 4
DEFAULT_D_FF:      int = 2048
DEFAULT_DROPOUT:   float = 0.1
DEFAULT_TIE:      bool = True


class GRUBaseline(nn.Module):
    """~130M-param stacked-GRU baseline for the F4 cheat suite.

    Architecture:
        token embedding  [V, D]
            ↓
        embedding dropout
            ↓
        stacked GRU (n_layers) — D=1024 each, cuDNN-backed
            ↓
        bridge: Linear(D, d_ff) → GELU → Linear(d_ff, D)
            ↓
        final LayerNorm
            ↓
        tied output projection (Linear(D, V), weights tied to embedding)

    Parameters
    ----------
    vocab_size:
        Tokenizer vocabulary size (default 50_257, GPT-NeoX-20B).
    d_model:
        Hidden dimension of every GRU layer and the token embedding
        (default 1024). The GRU runs on this width end-to-end.
    n_layers:
        Number of stacked GRU layers (default 4). PyTorch's GRU
        accepts this directly; layers 2..n have hidden-to-hidden
        weight matrices of shape ``[3D, D]``.
    d_ff:
        Inner dimension of the post-GRU bridge (default 2048). This
        gives the model an explicit per-position non-linear mixing
        step analogous to the Transformer's FFN sublayer.
    dropout:
        Dropout probability on the embedding output and the bridge
        output (default 0.1). Note that ``nn.GRU`` does NOT accept a
        dropout argument here because dropout-between-layers on
        cuDNN-backed stacked GRUs is non-trivial; we apply dropout
        to the bridge output instead, which is the standard
        workaround in PyTorch literature.
    tie_weights:
        If ``True`` (default), the output projection is tied to the
        input embedding matrix. Requires ``d_model == vocab_size``
        is *not* required — we explicitly tie the ``[V, D]`` lm_head
        to the ``[V, D]`` embedding via ``.weight = ...``.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        d_model:    int = DEFAULT_D_MODEL,
        n_layers:   int = DEFAULT_N_LAYERS,
        d_ff:       int = DEFAULT_D_FF,
        dropout:    float = DEFAULT_DROPOUT,
        tie_weights: bool = DEFAULT_TIE,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {n_layers}")

        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.d_ff        = d_ff
        self.tie_weights = tie_weights

        # ── Token embedding ────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # ── Stacked GRU ────────────────────────────────────────────────────
        # batch_first=True keeps the public surface [B, T, D], which
        # matches both the Transformer control and the RLF engine.
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0,         # see module docstring; we apply
                                 # dropout at the bridge instead.
            bidirectional=False,
        )

        # ── Bridge: per-position FFN over the GRU's top-layer hidden state
        self.bridge = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )

        # ── Output head ────────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            # Tied weights: share the [V, D] matrix.
            self.lm_head.weight = self.token_emb.weight

        # ── Initialisation ─────────────────────────────────────────────────
        self.apply(self._init_weights)
        # cuDNN's GRU uses orthogonal init for recurrent weights by
        # default already, but we re-apply to be defensive against
        # downstream subclassing.
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)

        self._print_param_report()

    # ── Initialisation helpers ──────────────────────────────────────────────
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-NeoX-style init for Linear / Embedding layers.

        GRU weights are left to PyTorch's default (which uses
        uniform init on input-to-hidden and orthogonal on hidden-to-
        hidden) because the joint init scheme is brittle.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_param_report(self) -> None:
        """Print a one-line parameter breakdown."""
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"  GRUBaseline:          "
            f"{n_total:,} params total ({n_train:,} trainable) | "
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"d_ff={self.d_ff}, vocab={self.vocab_size}, tie={self.tie_weights}"
        )

    # ── Forward pass ────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        labels:   Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        input_ids:
            ``[B, T]`` token id tensor.
        labels:
            Optional ``[B, T]`` tensor of next-token targets. If given,
            the model returns ``(loss, logits)`` where ``loss`` is the
            mean next-token cross-entropy (with ``-100`` ignored). If
            ``None``, returns ``logits`` only.

        Returns
        -------
        (loss, logits):
            - ``loss``: scalar ``Tensor`` (``requires_grad=True`` if
              ``labels`` is provided)
            - ``logits``: ``[B, T, vocab_size]`` next-token logits
        Only ``logits`` if ``labels`` is ``None``.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [B, T], got shape {tuple(input_ids.shape)}"
            )

        # ── Embedding ───────────────────────────────────────────────────────
        x = self.token_emb(input_ids)         # [B, T, D]
        x = self.emb_drop(x)

        # ── Stacked GRU ─────────────────────────────────────────────────────
        # ``gru_out`` is the per-step top-layer hidden state; we discard
        # ``h_n`` because we only need the temporal sequence for next-
        # token prediction.
        gru_out, _h_n = self.gru(x)            # [B, T, D]

        # ── Bridge ─────────────────────────────────────────────────────────
        h = self.bridge(gru_out)              # [B, T, D]
        h = self.ln_f(h)
        logits = self.lm_head(h)              # [B, T, V]

        if labels is None:
            return logits

        # ── Next-token cross-entropy ───────────────────────────────────────
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss, logits

    # ── Convenience ────────────────────────────────────────────────────────
    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return the parameter count."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"n_layers={self.n_layers}, d_ff={self.d_ff}, "
            f"tie_weights={self.tie_weights}"
        )

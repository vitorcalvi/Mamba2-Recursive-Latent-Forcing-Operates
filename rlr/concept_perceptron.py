"""
rlr/concept_perceptron.py — V4 ConceptPerceptron (Scratchpad Fallback)
=====================================================================

Background
----------
When the F3 ablation gate *keeps failing* (Δ < 10pp repeatedly) the
upstream V4 path triggers this fallback: drop the raw
``nn.Parameter`` ``latent_memory`` and replace it with a *learned,
prompt-conditioned* concept generator.

Architecture
------------
::

    input_ids → embedding_layer → mean-pool → MLP → n_concepts × d_model
                                                   │
                                                   └─ L2Norm × learnable_scale

The original V4 had a 58× norm-explosion issue because the MLP output was
left in raw scale. This module fixes that by:

    1. L2-normalising each concept token (unit-norm columns).
    2. Multiplying by a ``nn.Parameter(torch.tensor(2.0))`` learnable
       scale that starts at 2 — small enough to be stable, large enough
       to give gradient room.

The output ``[B, n_concepts, d_model]`` slots into the same slice as the
original ``latent_memory.expand(B, -1, -1)``, so swapping it in is a
single-line change in the RLF engine:

::

    mem = concept_perceptron(input_ids, self.embedding)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConceptPerceptron(nn.Module):
    """Prompt-conditioned concept-token generator.

    Parameters
    ----------
    d_model:
        Hidden size of the target model (Mamba-1.4B ⇒ 2048).
    n_concepts:
        Number of concept tokens to emit (``PREFIX_M`` × budget). The
        upstream RLF engine uses 8 — this is the default.
    vocab_size:
        Vocabulary size (GPT-NeoX ⇒ 50277). Used only for typing the
        internal scratch buffer; the real embedding is supplied per-call.
    hidden_ratio:
        Width of the MLP bottleneck as a fraction of ``d_model``.
        Default 2 (i.e. ``d_model // 2``).
    initial_scale:
        Initial value of the learnable L2-norm scale. Default 2.0.
    """

    def __init__(
        self,
        d_model: int,
        n_concepts: int = 8,
        vocab_size: int = 50277,
        hidden_ratio: int = 2,
        initial_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if n_concepts <= 0:
            raise ValueError(f"n_concepts must be > 0, got {n_concepts}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if hidden_ratio <= 0:
            raise ValueError(f"hidden_ratio must be > 0, got {hidden_ratio}")
        if initial_scale <= 0:
            raise ValueError(f"initial_scale must be > 0, got {initial_scale}")

        self.d_model = int(d_model)
        self.n_concepts = int(n_concepts)
        self.vocab_size = int(vocab_size)

        hidden = max(1, d_model // hidden_ratio)

        # Project the mean-pooled prompt embedding into a (n_concepts × d_model)
        # concept tensor, then refine with an MLP.
        self.prompt_pool = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.concept_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        # Learnable per-concept seeds — act as a "concept dictionary" that
        # conditions the MLP output. Shape: [n_concepts, d_model].
        self.concept_seed = nn.Parameter(
            torch.randn(n_concepts, d_model, dtype=torch.float32) * 0.02
        )

        # Projection from the prompt-pool hidden state → n_concepts gates
        # (one scalar per concept). Allows the model to *mix* the static
        # dictionary with prompt-conditioned deltas. The input dim is
        # ``hidden`` (the bottleneck width) — projecting from full
        # ``d_model`` would re-introduce the 58× norm explosion by
        # passing un-pooled high-norm features through the gate.
        self.gate_proj = nn.Linear(hidden, n_concepts)

        # Learnable norm scale. Starts at 2.0 (≈ 2 × unit norm).
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

        # Parameters stay in **float32** — bf16 L2-norm followed by a
        # learnable scale loses ~3 decimal digits of gradient signal
        # and re-introduces the upstream 58× norm explosion. The *output*
        # is cast to bf16 inside ``forward`` for parity with the rest of
        # the RLF stack's hidden states.

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        embedding_layer: nn.Embedding,
    ) -> torch.Tensor:
        """Generate ``[B, n_concepts, d_model]`` concept tokens.

        Parameters
        ----------
        input_ids:
            ``[B, T]`` token ids of the prompt. The module does not move
            the tensor — pass a tensor already on the right device.
        embedding_layer:
            The token embedding layer of the host model. Required so
            this module does not need a second copy of the 50k×2048
            embedding table.

        Returns
        -------
        concept_tokens:
            ``[B, n_concepts, d_model]`` bfloat16 tensor. Concept columns
            are L2-normalised and scaled by ``self.scale``.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [B, T], got shape {tuple(input_ids.shape)}"
            )
        if not isinstance(embedding_layer, nn.Embedding):
            raise TypeError(
                f"embedding_layer must be nn.Embedding, got {type(embedding_layer).__name__}"
            )
        if embedding_layer.embedding_dim != self.d_model:
            raise ValueError(
                f"embedding dim mismatch: got {embedding_layer.embedding_dim}, "
                f"expected {self.d_model}"
            )

        # Promote to float32 for stable mean / matmul, then cast back to
        # bfloat16 at the end (matches the rest of the RLF stack).
        x_embed = embedding_layer(input_ids).to(torch.float32)        # [B, T, D]
        prompt_summary = x_embed.mean(dim=1)                          # [B, D]

        hidden = self.prompt_pool(prompt_summary)                     # [B, H]

        # Per-concept mixing coefficients in [0, 1] (sigmoid-gated).
        gates = torch.sigmoid(self.gate_proj(hidden))                  # [B, C]
        # [B, D] → [B, C, D] via additive broadcast of the concept seed.
        seed = self.concept_seed.unsqueeze(0)                          # [1, C, D]
        # Project the prompt summary into a per-concept delta.
        delta = self.concept_mlp(prompt_summary).unsqueeze(1).expand(
            -1, self.n_concepts, -1
        )                                                             # [B, C, D]

        # Gate the *delta*, not the seed — keeps the seed acting as the
        # default concept dictionary when the prompt is uninformative.
        gated_delta = gates.unsqueeze(-1) * delta                     # [B, C, D]
        raw = seed + gated_delta                                      # [B, C, D]

        # L2-normalise the concept dimension (per-column unit norm), then
        # apply the learnable scale. This is the F4 fix for the upstream
        # 58× norm explosion.
        norms = raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit = raw / norms                                            # [B, C, D]
        scaled = unit * self.scale.to(unit.dtype)                     # [B, C, D]

        return scaled.to(torch.bfloat16)

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def norm_stats(self, input_ids: torch.Tensor, embedding_layer: nn.Embedding) -> dict:
        """Return per-concept L2-norm statistics (useful for debugging F3)."""
        with torch.no_grad():
            tokens = self.forward(input_ids, embedding_layer)
            flat_norms = tokens.float().norm(dim=-1)                  # [B, C]
        return {
            "mean": float(flat_norms.mean().item()),
            "std":  float(flat_norms.std(unbiased=False).item()),
            "min":  float(flat_norms.min().item()),
            "max":  float(flat_norms.max().item()),
            "scale": float(self.scale.item()),
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_concepts={self.n_concepts}, "
            f"vocab_size={self.vocab_size}, scale={self.scale.item():.4f}"
        )

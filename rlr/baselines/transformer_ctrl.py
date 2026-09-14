"""
rlr/baselines/transformer_ctrl.py — Same-Size Transformer Control
==================================================================

A small (~130M parameter) vanilla causal Transformer used as the
"could a tiny transformer also solve this?" control in the F4 cheat
suite. The baseline has:

  * n_layers = 12, d_model = 768, n_heads = 12, d_ff = 3072
  * sinusoidal positional encodings
  * pre-LayerNorm, GELU activations
  * tied input/output embeddings (matches GPT-NeoX tokenizer vocab)
  * autoregressive next-token cross-entropy
  * NO loops, NO scratchpad, NO recursion

The class deliberately exposes a ``forward(input_ids, labels=None)``
signature that mirrors the RLF engine so the cheat suite can drive
it through a single entry point.

Parameter count (approx, fp32):
    embedding     = 50_257 × 768         = 38,597,376
    per block × 12 = (Q+K+V+O proj = 4 × 768² = 2,359,296) +
                     (2 × 768 × 3072   = 4,718,592 FFN) +
                     (2 × 768 LN       = 1,536       LN)
                   ≈ 7,079,424 per block × 12 = 84,953,088
    tied output                               = 0
    final LN                                  = 1,536
    ─────────────────────────────────────────────────────
    total                                     ≈ 123,552,000

This is ~10× smaller than the 1.4B RLF engine, which is the point:
we are testing whether architectural inductive bias (Mamba's selective
SSM + RLF recursion) buys *anything* that a plain Transformer of
comparable scale cannot match.

Public API
----------
>>> model = TransformerBaseline(vocab_size=50_257)
>>> loss, logits = model(input_ids, labels=input_ids)
>>> logits = model(input_ids)  # inference

Notes
-----
* This module is **self-contained** — it does NOT import the RLF
  engine. That isolation is intentional: the control must remain
  reachable even if the RLF training stack is broken.
* The ``HALT_ID`` constant is not hard-coded here; the cheat suite
  decides what counts as a "halt" by inspecting the predicted token id.
* Causal masking uses an additive ``-inf`` mask on the attention scores
  (the standard decoder-only pattern), which is numerically stable in
  fp16/bf16. We do NOT use scaled-dot-product with a precomputed mask
  tensor because recomputing it per forward pass keeps the module
  agnostic to ``seq_len`` at call time.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Default config ──────────────────────────────────────────────────────────
# These defaults match the F4 spec in the cheat-suite brief.
DEFAULT_VOCAB_SIZE: int = 50_257       # GPT-NeoX-20B vocab size
DEFAULT_D_MODEL:   int = 768
DEFAULT_N_LAYERS:  int = 12
DEFAULT_N_HEADS:   int = 12
DEFAULT_D_FF:      int = 3072
DEFAULT_MAX_SEQ:   int = 2048          # positional-encoding table size
DEFAULT_DROPOUT:   float = 0.1


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017).

    A non-trainable ``[max_seq_len, d_model]`` buffer is registered and
    added to the embedding output at forward time. We pick sinusoidal
    over learned positional embeddings to avoid a trainable
    ``max_seq_len × d_model`` matrix that would inflate the parameter
    count by ~1.5M and bias the budget tally.
    """

    def __init__(self, d_model: int, max_seq_len: int = DEFAULT_MAX_SEQ) -> None:
        super().__init__()
        # Precompute the (sin, cos) table — shape [max_seq_len, d_model].
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        # Standard 10_000 base, 2*idx pairs share the same frequency.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        # Register as a non-trainable buffer; broadcast over batch dim.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape ``[B, T, D]``."""
        return x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention block (decoder-only).

    Uses an additive ``-inf`` mask built on the fly to keep the module
    agnostic to ``seq_len`` and to support arbitrary batch shapes.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Single fused QKV projection — fewer parameters than three
        # separate ``nn.Linear`` layers and lets cuBLAS pick a single
        # GEMM kernel for all three.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention. ``x`` shape: ``[B, T, D]``."""
        B, T, D = x.shape
        # [B, T, 3D] → [B, T, 3, H, d_h] → [3, B, H, T, d_h]
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)        # [B, H, T, d_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # F.scaled_dot_product_attention handles causal masking
        # internally, runs in fused kernels (Flash/Memory-Efficient)
        # when available, and avoids materialising an explicit mask
        # tensor of shape [T, T] — keeps memory flat in T.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.resid_dropout(self.out_proj(out))


class _TransformerBlock(nn.Module):
    """Pre-LN Transformer block: ``LN → Attn → residual → LN → FFN → residual``.

    Pre-LN is the GPT-2/GPT-Neo convention; it removes the need for a
    careful learning-rate warm-up because the residual stream is
    already normalised before each sublayer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)
        # GELU FFN — slightly better calibration than ReLU on small models.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Transformer block. ``x`` shape: ``[B, T, D]``."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class TransformerBaseline(nn.Module):
    """~130M-param causal Transformer baseline for the F4 cheat suite.

    The model is trained on the *same* chain data as the RLF engine
    (``rlf_dataset.RLFDataset``) but with no loops, no scratchpad, and
    no Mamba state-space recurrence. It serves as the "could a tiny
    Transformer also solve multi-hop variable chains?" control.

    Parameters
    ----------
    vocab_size:
        Tokenizer vocabulary size (default 50_257, GPT-NeoX-20B).
    d_model:
        Embedding / hidden dimension (default 768).
    n_layers:
        Number of Transformer blocks (default 12).
    n_heads:
        Number of attention heads per block (default 12). Must divide
        ``d_model``.
    d_ff:
        Feed-forward inner dimension (default 3072 = 4 × d_model).
    max_seq_len:
        Positional-encoding table size (default 2048).
    dropout:
        Dropout probability on attention weights, residual outputs,
        and FFN outputs (default 0.1).
    tie_weights:
        If ``True`` (default), the output projection is tied to the
        input embedding matrix. This saves ``vocab_size × d_model``
        parameters and is the GPT-NeoX convention.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        d_model:   int = DEFAULT_D_MODEL,
        n_layers:  int = DEFAULT_N_LAYERS,
        n_heads:   int = DEFAULT_N_HEADS,
        d_ff:      int = DEFAULT_D_FF,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        dropout:   float = DEFAULT_DROPOUT,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size   = vocab_size
        self.d_model      = d_model
        self.n_layers     = n_layers
        self.n_heads      = n_heads
        self.d_ff         = d_ff
        self.max_seq_len  = max_seq_len
        self.tie_weights  = tie_weights

        # ── Embedding stack ────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = _SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.emb_drop  = nn.Dropout(dropout)

        # ── Stack of Transformer blocks ────────────────────────────────────
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Output head ────────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        # Build the output projection as a separate Linear so we can
        # optionally tie its weight to ``token_emb``. Tying is the
        # GPT-NeoX convention and saves ~38M params at the default
        # vocab_size.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # Initialise weights with a small GPT-NeoX-style scheme.
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2·n_layers) — GPT-2 init.
        scale = 1.0 / math.sqrt(2.0 * n_layers)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("ffn.2.weight"):
                with torch.no_grad():
                    p.mul_(scale)

        # Report parameter count at construction time — useful for the
        # cheat-suite report ("Transformer ctrl: 123M params").
        self._print_param_report()

    # ── Initialisation helpers ──────────────────────────────────────────────
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-NeoX-style weight init for Linear layers and embeddings."""
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
            f"  TransformerBaseline: "
            f"{n_total:,} params total ({n_train:,} trainable) | "
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}, d_ff={self.d_ff}, "
            f"vocab={self.vocab_size}, tie={self.tie_weights}"
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
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        # ── Embedding + positional encoding ────────────────────────────────
        x = self.token_emb(input_ids)         # [B, T, D]
        x = self.pos_enc(x)
        x = self.emb_drop(x)

        # ── Transformer blocks ─────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)

        # ── Output head ────────────────────────────────────────────────────
        x = self.ln_f(x)
        logits = self.lm_head(x)              # [B, T, V]

        if labels is None:
            return logits

        # ── Next-token cross-entropy ───────────────────────────────────────
        # Shift: predict logits[:, :-1, :] against labels[:, 1:, :].
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
        """Return the parameter count.

        Parameters
        ----------
        only_trainable:
            If ``True``, count only parameters with ``requires_grad``.
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"d_ff={self.d_ff}, max_seq_len={self.max_seq_len}, "
            f"tie_weights={self.tie_weights}"
        )

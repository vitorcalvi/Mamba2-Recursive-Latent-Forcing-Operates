"""
rlr.baselines — Same-Size Control Models for the RLR Cheat Suite
================================================================

Why these baselines exist
-------------------------
The RLF engine (~1.4B parameters, Mamba1 backbone, 48 layers, d_model=2048)
solves multi-hop variable-pointer chains with a recursive latent scratchpad
that performs up to ``MAX_LOOPS=6`` inner iterations per token. The F4
cheat-detection suite (``rlr.cheat_suite``) needs *controls* to test the
claim that the latent loop is doing cognitive work rather than something
that any comparably-sized sequence model could do.

These baselines are intentionally *small* (~130M parameters, ~10× smaller
than the RLF engine) and *standard*:

  * ``TransformerBaseline`` — vanilla causal Transformer with sinusoidal
    positional encodings, pre-LN, GELU, trained with plain
    next-token cross-entropy on the same chain data. No loops, no
    scratchpad, no recursion. This is the "could a tiny transformer
    also solve this?" control.

  * ``GRUBaseline`` — stacked GRU sequence model, trained with the same
    next-token objective. Tests whether a tiny recurrent baseline is
    sufficient. GRU's hidden-state recurrence is fundamentally different
    from Mamba's selective SSM, but at ~130M it acts as a cheap
    recurrent sanity-check.

If these baselines *also* solve multi-hop chains at high accuracy, the
RLF result is not impressive; if they *fail* (accuracy ≪ RLF engine),
that is strong evidence that the latent loop is doing real cognitive
work and is not merely a learnable artifact of Mamba's state-space
recurrence.

Conventions
-----------
All baselines mirror the public surface of the RLF engine so the cheat
suite can drive them through a single ``forward(input_ids, labels=None)``
call:

  * training: ``forward`` returns ``(loss, logits)``
  * inference: ``forward`` returns ``logits`` only

The token budget is ``vocab_size = 50_257`` (GPT-NeoX-20B vocab, same
as ``rlf_engine_1_4b.tokenizer``); both baselines share an
``nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)`` so the
input distribution matches the engine's.

Author: Track 1 evaluation harness, F4 fix.
"""

from .transformer_ctrl import TransformerBaseline
from .gru_ctrl import GRUBaseline

__all__ = [
    "TransformerBaseline",
    "GRUBaseline",
]

__version__ = "0.1.0"

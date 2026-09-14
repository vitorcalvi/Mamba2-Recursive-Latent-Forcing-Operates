"""
rlr — Recursive Latent Reasoner (Track 1 core modules)
======================================================

Public classes exposed by this package:

    DensePerLoopLoss       (rlr.dense_loss)        — F1 fix
    ACTHaltingHead         (rlr.act_halting)       — F2 fix
    LoopDropout            (rlr.loop_dropout)      — regularizer
    ScratchpadAblationGate (rlr.ablation_gate)     — F3 fix
    ConceptPerceptron      (rlr.concept_perceptron)— V4 fallback

All modules are self-contained. They only depend on ``torch`` (and on each
other where the package docstring makes the dependency explicit). The classes
are designed to plug into the existing RLF engine in
``rlf_engine_1_4b.py`` whose conventions they mirror:

    - HALT_ID  = 7803        (§ token, GPT-NeoX vocab)
    - PREFIX_M = 8           (latent scratchpad width)
    - MAX_LOOPS = 6          (maximum RLF iterations per token)
    - d_model = 2048         (Mamba-1.4B hidden size)
"""

from .dense_loss import DensePerLoopLoss
from .act_halting import ACTHaltingHead
from .loop_dropout import LoopDropout
from .ablation_gate import ScratchpadAblationGate
from .concept_perceptron import ConceptPerceptron

__all__ = [
    "DensePerLoopLoss",
    "ACTHaltingHead",
    "LoopDropout",
    "ScratchpadAblationGate",
    "ConceptPerceptron",
]

__version__ = "0.1.0"

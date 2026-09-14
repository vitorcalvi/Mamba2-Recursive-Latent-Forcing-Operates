"""
bridge — Track 1 (RLR) ↔ Track 2 (Dreamer/T2) Bridge Layer
==========================================================

This package couples the Recursive Latent Reasoner (RLR) latent
scratchpad (Track 1) to the Mamba-2 DreamerV3-style world model in
``t2``. The bridge:

1. Projects the RSSM recurrent state ``h_t`` (Mamba-2 SSM output) into
   the RLR latent space.
2. Runs up to ``max_loops`` recursive latent refinements, with an
   :class:`ACTHaltingHead` deciding when to stop (adaptive computation
   time à la Graves 2016).
3. Returns the augmented feature vector and the number of loops used,
   so downstream callers can charge the correct ponder cost.

The bridge is *ablation-aware*: it can run in three modes that
:func:`BridgeEvaluation.check_gate_g3` compares head-to-head against a
matched-parameter baseline.

Public surface
--------------

.. code-block:: text

    RLRDreamerBridge   (bridge.rlr_dreamer_bridge)
    BridgeEvaluation   (bridge.bridge_eval)

Design references
-----------------
- Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural
  Networks.* arXiv:1603.08983 — ACT halting + ponder cost.
- Hafner et al. (2024). *Mastering Diverse Domains through World
  Models.* (DreamerV3) — RSSM interface conventions.
- DRAMA (arXiv:2410.08893) — first Mamba-2 MBRL agent.
"""

from __future__ import annotations

from .rlr_dreamer_bridge import RLRDreamerBridge
from .bridge_eval import BridgeEvaluation

__all__ = [
    "RLRDreamerBridge",
    "BridgeEvaluation",
]

__version__ = "0.1.0"

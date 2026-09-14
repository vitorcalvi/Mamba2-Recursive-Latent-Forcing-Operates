"""
t2.baselines — Same-Size Control Models for the Mamba-2 World Model
====================================================================

Why this baseline exists
------------------------
The Mamba-2 RSSM world model in :mod:`t2.world_model` is the centerpiece of
the Mamba-2 Recursive-Latent-Forcing-Operates pipeline. To make a fair
ablation claim that the *Mamba-2 selective SSM dynamics core* (rather than
the surrounding DreamerV3 heads) is responsible for any observed gains on
POPGym, we need a **matched-parameter baseline** that swaps only the
recurrent dynamics kernel:

    Mamba-2 stack  (selective SSM)  →  stacked nn.GRU  (gated RNN)

Everything else — encoder, prior/post heads, decoder, reward head,
continue head, categorical latent of size ``stoch_dim × n_categories``,
``WorldModelState`` / ``ImaginedTrajectory`` containers, the
``initial_state`` / ``step`` / ``forward`` / ``imagine`` API, and the
KL/recon/reward/continue loss decomposition — is *byte-identical* to the
Mamba2 version. Only ``_DynamicsCore`` differs.

This is the standard control used in DRAMA (arXiv:2410.08893) and
DreamerV3 (Hafner et al. 2024): a multi-layer GRU replaces the recurrent
state-space model and every other knob is held constant.

Public surface
--------------
``GRURSSMWorldModel`` — same forward signature and state shapes as
``Mamba2WorldModel``; can be dropped in via::

    from t2.baselines.gru_rssm import GRURSSMWorldModel
    wm = GRURSSMWorldModel(obs_dim, act_dim, d_model=256, n_layers=4, ...)

The module ``t2.baselines.__init__`` re-exports it under the name
``GRURSSMWorldModel``; ``t2`` also re-exports it lazily via
``__getattr__`` so ``from t2 import GRURSSMWorldModel`` works.
"""

from __future__ import annotations

from .gru_rssm import GRURSSMWorldModel

__all__ = [
    "GRURSSMWorldModel",
]

__version__ = "0.1.0"

"""
t2 — Track 2: SSM World-Model Reinforcement Learning
====================================================

This package implements a DreamerV3-style model-based RL agent whose
Recurrent State-Space Model (RSSM) uses a **Mamba-2 selective state-space
core** instead of the canonical GRU. It targets the POPGym benchmark suite
(Morad et al. 2023) and follows the design of the DRAMA paper
(arXiv:2410.08893), the first Mamba-2 MBRL agent.

Public surface
--------------

.. code-block:: text

    t2.world_model      → Mamba2WorldModel             (SSM dynamics core)
    t2.actor_critic     → DreamerActorCritic           (V3 AC + EMA target)
    t2.popgym_wrapper   → POPGymWrapper               (Autoencode / etc.)
    t2.dreamer_trainer  → DreamerTrainer              (full MBRL loop)
    t2.dreamer_eval     → DreamerEvalGrid             (vs GRU control)
    t2.baselines.gru_rssm → GRURSSMWorldModel         (matched-param GRU)

Graceful imports
----------------
``mamba_ssm`` is **optional**; when missing the world model falls back to a
pure-PyTorch selective scan. ``popgym`` is also optional; if absent the
wrapper returns deterministic toy environments so the rest of the
codebase stays import-clean. Both behaviours are signalled via
:data:`MAMBA_AVAILABLE` and :data:`POPGYM_AVAILABLE`.

Design references
-----------------
- DreamerV3 (Hafner et al. 2024) — symlog value transform, EMA target
  critic, lambda returns, categorical latents.
- DRAMA (arXiv:2410.08893) — 7M-param Mamba-2 world model, scheduled
  imagination horizon, POPGym evaluation.
- MAMBA meta-RL (arXiv:2403.09859) — augmented observation
  ``[o_t, r_t, t]`` so the SSM can condition on reward and timestep.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optional-dependency flags
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when mamba_ssm is installed
    import mamba_ssm  # type: ignore

    MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover
    MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = _exc

try:  # pragma: no cover
    import popgym  # type: ignore

    POPGYM_AVAILABLE = True
    _POPGYM_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover
    POPGYM_AVAILABLE = False
    _POPGYM_IMPORT_ERROR = _exc

# ---------------------------------------------------------------------------
# Re-exports (delayed to avoid forcing torch at package import time)
# ---------------------------------------------------------------------------
__all__ = [
    "MAMBA_AVAILABLE",
    "POPGYM_AVAILABLE",
    "Mamba2WorldModel",
    "DreamerActorCritic",
    "POPGymWrapper",
    "DreamerTrainer",
    "DreamerEvalGrid",
    "GRURSSMWorldModel",
]


def __getattr__(name: str):
    """Lazy attribute access so ``import t2`` does not require torch."""
    if name == "Mamba2WorldModel":
        from .world_model import Mamba2WorldModel

        return Mamba2WorldModel
    if name == "DreamerActorCritic":
        from .actor_critic import DreamerActorCritic

        return DreamerActorCritic
    if name == "POPGymWrapper":
        from .popgym_wrapper import POPGymWrapper

        return POPGymWrapper
    if name == "DreamerTrainer":
        from .dreamer_trainer import DreamerTrainer

        return DreamerTrainer
    if name == "DreamerEvalGrid":
        from .dreamer_eval import DreamerEvalGrid

        return DreamerEvalGrid
    if name == "GRURSSMWorldModel":
        from .baselines.gru_rssm import GRURSSMWorldModel

        return GRURSSMWorldModel
    raise AttributeError(f"module 't2' has no attribute {name!r}")


__version__ = "0.1.0"

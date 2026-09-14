"""bridge/bridge_eval.py — 3-Arm Ablation Evaluation Grid for Bridge (Gate G3).

Evaluates:
- Scratchpad ON (active gradient-updated latent looping)
- Scratchpad FROZEN (fixed-weight random recurrence control)
- Scratchpad ABSENT (standard Dreamer policy without latent loop)

Sweeps loop budgets (1, 4, 8) and ponder cost penalties tau (0.0, 0.01) across seeds.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from bridge.rlr_dreamer_bridge import RLRDreamerBridge
from t2.actor_critic import DreamerActorCritic
from t2.popgym_wrapper import POPGymWrapper
from t2.world_model import Mamba2WorldModel


@dataclass
class BridgeAblationResult:
    mode: str
    max_loops: int
    tau: float
    seed: int
    mean_return: float
    std_return: float
    avg_loops_used: float
    wall_clock_sec: float


class BridgeEvaluation:
    """Orchestrator for the 3-arm bridge ablation study."""

    def __init__(
        self,
        env_name: str = "RepeatPrevious",
        n_seeds: int = 5,
        n_episodes: int = 5,
        device: str = "cpu",
    ):
        self.env_name = env_name
        self.n_seeds = n_seeds
        self.n_episodes = n_episodes
        self.device = torch.device(device)

    def evaluate_configuration(
        self,
        mode: str,
        max_loops: int,
        tau: float,
        seed: int,
    ) -> BridgeAblationResult:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = POPGymWrapper(env_name=self.env_name, augment_obs=True, seed=seed)
        wm = Mamba2WorldModel(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            d_model=64,
            d_state=16,
            n_layers=2,
            stoch_dim=8,
            n_categories=8,
        ).to(self.device)

        bridge = RLRDreamerBridge(
            rssm_state_dim=wm.d_model,
            rlr_d_model=32,
            max_loops=max_loops,
            tau=tau,
            ablation_mode=mode,
        ).to(self.device)

        feat_dim = wm.d_model + wm.flat_z
        ac = DreamerActorCritic(state_dim=feat_dim, act_dim=env.act_dim, hidden=64).to(self.device)

        t0 = time.perf_counter()
        returns = []
        loops_used_list = []

        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_ret = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    embed = wm.encoder(obs_t)
                    # Pass recurrent state through RLR bridge
                    aug_embed, n_lp = bridge(embed)
                    loops_used_list.append(n_lp)

                    z_dummy = torch.zeros(1, wm.flat_z, device=self.device)
                    feat = torch.cat([aug_embed, z_dummy], dim=-1)
                    action = ac.actor(feat).argmax(dim=-1).item()

                obs, reward, term, trunc, _ = env.step(action)
                ep_ret += reward
                done = term or trunc
            returns.append(ep_ret)

        elapsed = time.perf_counter() - t0
        return BridgeAblationResult(
            mode=mode,
            max_loops=max_loops,
            tau=tau,
            seed=seed,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            avg_loops_used=float(np.mean(loops_used_list)) if loops_used_list else 0.0,
            wall_clock_sec=elapsed,
        )

    def run_ablation_grid(
        self,
        loop_configs: Optional[List[int]] = None,
        tau_configs: Optional[List[float]] = None,
    ) -> List[BridgeAblationResult]:
        loops = loop_configs or [1, 4, 8]
        taus = tau_configs or [0.0, 0.01]
        results: List[BridgeAblationResult] = []

        # 1. Baseline: Absent mode (loops=0, tau=0)
        for s in range(self.n_seeds):
            results.append(self.evaluate_configuration("absent", 0, 0.0, s))

        # 2. Frozen mode sweep
        for lp in loops:
            for s in range(self.n_seeds):
                results.append(self.evaluate_configuration("frozen", lp, 0.0, s))

        # 3. Full mode sweep
        for lp in loops:
            for t in taus:
                for s in range(self.n_seeds):
                    results.append(self.evaluate_configuration("full", lp, t, s))

        return results

    def check_gate_g3(
        self,
        results: List[BridgeAblationResult],
        min_improvement_pct: float = 0.15,
    ) -> Tuple[bool, float, Dict[str, float]]:
        absent_scores = [r.mean_return for r in results if r.mode == "absent"]
        full_scores = [r.mean_return for r in results if r.mode == "full"]
        frozen_scores = [r.mean_return for r in results if r.mode == "frozen"]

        mean_absent = float(np.mean(absent_scores)) if absent_scores else 1.0
        mean_full = float(np.mean(full_scores)) if full_scores else 0.0
        mean_frozen = float(np.mean(frozen_scores)) if frozen_scores else 0.0

        # Percentage improvement of full vs absent baseline
        improvement = (mean_full - mean_absent) / (abs(mean_absent) + 1e-6)
        passed = (improvement >= min_improvement_pct) and (mean_full > mean_frozen)

        breakdown = {
            "mean_absent": mean_absent,
            "mean_frozen": mean_frozen,
            "mean_full": mean_full,
            "improvement_pct": float(improvement),
        }
        return passed, float(improvement), breakdown

    def generate_report(self, results: List[BridgeAblationResult]) -> str:
        lines = [
            "### Bridge 3-Arm Ablation Grid (Gate G3)",
            "",
            "| Mode | Loops | Tau | Mean Return | Std | Avg Loops Used |",
            "|:---|:---|:---|:---|:---|:---|",
        ]

        grouped: Dict[Tuple[str, int, float], List[BridgeAblationResult]] = {}
        for r in results:
            key = (r.mode, r.max_loops, r.tau)
            grouped.setdefault(key, []).append(r)

        for (mode, lp, tau), group in grouped.items():
            scores = [g.mean_return for g in group]
            loops_used = [g.avg_loops_used for g in group]
            lines.append(
                f"| **{mode.upper()}** | {lp} | {tau} | {np.mean(scores):.2f} | ±{np.std(scores):.2f} | {np.mean(loops_used):.1f} |"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    evaluator = BridgeEvaluation(env_name="RepeatPrevious", n_seeds=2, n_episodes=2)
    print("Running mini ablation grid...")
    res = evaluator.run_ablation_grid(loop_configs=[2], tau_configs=[0.01])
    passed, imp, b = evaluator.check_gate_g3(res)
    print(evaluator.generate_report(res))
    print(f"\nGate G3 Check: {'PASSED' if passed else 'FAILED'} (Improvement: {imp:.2%})")

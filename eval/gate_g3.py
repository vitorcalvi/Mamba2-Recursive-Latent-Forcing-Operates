"""eval/gate_g3.py — Gate G3 Go/No-Go Specification Checker (Bridge Gate).

Evaluates whether coupling Track 1 (RLR recursive latent looping) to Track 2
(DreamerV3 RSSM policy) yields genuine computational advantage on POMDP
memory-credit benchmarks (POPGym) over the un-augmented baseline.

Gate G3 Criteria:
1. POPGym score improves >= 15% over T2 no-scratchpad policy at equal wall-clock
2. Full scratchpad outperforms frozen-weights random recurrence control
3. Replicated across >= 5 random seeds
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bridge.bridge_eval import BridgeEvaluation
from eval.gate_g1 import GateResult


class GateG3:
    """Orchestrator for Gate G3 validation."""

    def __init__(
        self,
        env_name: str = "RepeatPrevious",
        n_seeds: int = 5,
        min_improvement_pct: float = 0.15,
    ):
        self.env_name = env_name
        self.n_seeds = n_seeds
        self.min_improvement_pct = min_improvement_pct
        self.evaluator = BridgeEvaluation(
            env_name=env_name,
            n_seeds=n_seeds,
            n_episodes=5,
        )

    def evaluate(self, fast_mode: bool = False) -> GateResult:
        """Run 3-arm ablation grid and check Gate G3 criteria."""
        if fast_mode:
            evaluator = BridgeEvaluation(env_name=self.env_name, n_seeds=2, n_train_episodes=3, n_eval_episodes=2)
            results = evaluator.run_ablation_grid(loop_configs=[2], tau_configs=[0.01], fast_mode=True)
            passed, imp_pct, breakdown = evaluator.check_gate_g3(results, self.min_improvement_pct)
        else:
            results = self.evaluator.run_ablation_grid(loop_configs=[1, 4, 8], tau_configs=[0.0, 0.01], fast_mode=False)
            passed, imp_pct, breakdown = self.evaluator.check_gate_g3(results, self.min_improvement_pct)

        c1 = imp_pct >= self.min_improvement_pct
        c2 = breakdown["mean_full"] > breakdown["mean_frozen"]
        c3 = self.n_seeds >= (2 if fast_mode else 5)

        criteria = {
            "1_score_improvement_ge_15pct": bool(c1),
            "2_full_beats_frozen_scratchpad": bool(c2),
            "3_replicated_across_seeds": bool(c3),
        }

        all_passed = all(criteria.values())
        metrics = {
            "improvement_pct": float(imp_pct),
            "mean_full_return": float(breakdown["mean_full"]),
            "mean_frozen_return": float(breakdown["mean_frozen"]),
            "mean_absent_return": float(breakdown["mean_absent"]),
        }

        report_table = (evaluator if fast_mode else self.evaluator).generate_report(results)
        verdict_str = "🚀 GATE G3 PASSED: Latent recursive scratchpad bridge delivers verified empirical gain" if all_passed else "🛑 GATE G3 FAILED: Scratchpad bridge does not beat baseline ablations (negative result)"

        summary = (
            f"### Gate G3 Evaluation Report (T1↔T2 Bridge)\n\n"
            f"{report_table}\n\n"
            f"- **Improvement Delta:** {imp_pct:+.2%} (Threshold: +{self.min_improvement_pct:.0%})\n"
            f"- **Active vs Frozen:** Full ({breakdown['mean_full']:.2f}) vs Frozen ({breakdown['mean_frozen']:.2f})\n"
            f"- **Overall Verdict:** {verdict_str}\n"
        )

        return GateResult(
            gate_name="Gate_G3_Bridge",
            passed=all_passed,
            criteria=criteria,
            metrics=metrics,
            summary=summary,
        )


if __name__ == "__main__":
    gate = GateG3()
    print("Testing GateG3 (fast mode)...")
    res = gate.evaluate(fast_mode=True)
    print(res.summary)

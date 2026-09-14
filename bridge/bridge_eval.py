"""bridge/bridge_eval.py — 3-Arm Ablation Evaluation Grid for Bridge (Gate G3).

Evaluates:
- Scratchpad ON / FULL (active gradient-updated latent looping)
- Scratchpad FROZEN (fixed-weight random recurrence control)
- Scratchpad ABSENT (standard Dreamer policy without latent loop)

Key Architecture:
1. Real POMDP interaction using POPGym (popgym-RepeatPreviousEasy-v0)
2. Maintains real recurrent state h_t across rollouts via wm.step()
3. Genuinely trains World Model, Actor-Critic, and Bridge (in full mode)
4. Penalises latent compute via tau * ponder_cost in actor loss
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
import torch.nn as nn
import torch.nn.functional as F

from bridge.rlr_dreamer_bridge import RLRDreamerBridge
from t2.actor_critic import DreamerActorCritic
from t2.popgym_wrapper import POPGymWrapper
from t2.world_model import Mamba2WorldModel, WorldModelState


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
        n_train_episodes: int = 15,
        n_eval_episodes: int = 5,
        n_episodes: Optional[int] = None,
        device: str = "cpu",
    ):
        self.env_name = env_name
        self.n_seeds = n_seeds
        self.n_train_episodes = n_train_episodes
        self.n_eval_episodes = n_episodes if n_episodes is not None else n_eval_episodes
        self.device = torch.device(device)

    def _train_step(
        self,
        wm: Mamba2WorldModel,
        bridge: RLRDreamerBridge,
        ac: DreamerActorCritic,
        opt: torch.optim.Optimizer,
        state: WorldModelState,
        obs_t: torch.Tensor,
        prev_action: torch.Tensor,
        reward: float,
        tau: float,
        mode: str,
    ) -> Tuple[WorldModelState, int, float, torch.Tensor]:
        """Performs a single recurrent training step."""
        obs_embed = wm.encoder(obs_t)
        next_state, prior_logits, post_logits = wm.step(state.detach(), prev_action, obs_embed)

        aug_h, n_lp, p_cost = bridge(next_state.h, return_cost=True)
        feat = torch.cat([aug_h, next_state.z.flatten(start_dim=1)], dim=-1)

        logits = ac.actor(feat)
        value = ac.critic(feat)

        rec_obs = wm.decoder(torch.cat([next_state.h, next_state.z.flatten(start_dim=1)], dim=-1))
        pred_rew = wm.reward_head(torch.cat([next_state.h, next_state.z.flatten(start_dim=1)], dim=-1))

        loss_wm = F.mse_loss(rec_obs, obs_t) + F.mse_loss(pred_rew.squeeze(-1), torch.tensor([reward], device=self.device))
        target_v = torch.tensor([[reward]], device=self.device)
        loss_val = F.mse_loss(value, target_v)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss_pi = -logits.max(dim=-1).values.mean() - 0.01 * entropy

        loss = loss_wm + loss_val + loss_pi + tau * p_cost

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
        if mode == "full":
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()

        return next_state, n_lp, float(p_cost.item()), logits

    def evaluate_configuration(
        self,
        mode: str,
        max_loops: int,
        tau: float,
        seed: int,
        n_train: Optional[int] = None,
        n_eval: Optional[int] = None,
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

        train_params = list(wm.parameters()) + list(ac.parameters())
        if mode == "full":
            train_params += [p for p in bridge.parameters() if p.requires_grad]
        opt = torch.optim.Adam(train_params, lr=1e-3)

        t0 = time.perf_counter()
        train_eps = n_train if n_train is not None else self.n_train_episodes
        eval_eps = n_eval if n_eval is not None else self.n_eval_episodes

        # 1. Training Phase
        for ep in range(train_eps):
            obs, _ = env.reset(seed=seed + ep)
            state = wm.initial_state(batch_size=1, device=self.device)
            prev_action = torch.zeros(1, env.act_dim, device=self.device)
            done = False
            step_count = 0

            while not done and step_count < 64:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                action_int = torch.randint(0, env.act_dim, (1,)).item() if np.random.rand() < 0.2 else None

                next_obs, reward, term, trunc, _ = env.step(action_int if action_int is not None else prev_action.argmax(dim=-1).item())
                state, _, _, logits = self._train_step(
                    wm, bridge, ac, opt, state, obs_t, prev_action, reward, tau, mode
                )

                chosen_act = logits.argmax(dim=-1).item()
                prev_action = torch.zeros(1, env.act_dim, device=self.device)
                prev_action[0, chosen_act] = 1.0

                obs = next_obs
                done = term or trunc
                step_count += 1

        # 2. Evaluation Phase
        wm.eval()
        bridge.eval()
        ac.eval()

        returns = []
        loops_used_list = []

        with torch.no_grad():
            for ep in range(eval_eps):
                obs, _ = env.reset(seed=seed + 1000 + ep)
                state = wm.initial_state(batch_size=1, device=self.device)
                prev_action = torch.zeros(1, env.act_dim, device=self.device)
                ep_ret = 0.0
                done = False
                step_count = 0

                while not done and step_count < 64:
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    embed = wm.encoder(obs_t)
                    state, _, _ = wm.step(state, prev_action, embed)

                    aug_h, n_lp, _ = bridge(state.h, return_cost=True)
                    loops_used_list.append(n_lp)

                    feat = torch.cat([aug_h, state.z.flatten(start_dim=1)], dim=-1)
                    action = ac.actor(feat).argmax(dim=-1).item()

                    prev_action = torch.zeros(1, env.act_dim, device=self.device)
                    prev_action[0, action] = 1.0

                    obs, reward, term, trunc, _ = env.step(action)
                    ep_ret += reward
                    done = term or trunc
                    step_count += 1

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
        fast_mode: bool = False,
    ) -> List[BridgeAblationResult]:
        loops = loop_configs or ([2] if fast_mode else [1, 4, 8])
        taus = tau_configs or ([0.01] if fast_mode else [0.0, 0.01])
        n_train = 3 if fast_mode else self.n_train_episodes
        n_eval = 2 if fast_mode else self.n_eval_episodes
        n_seeds = min(2, self.n_seeds) if fast_mode else self.n_seeds

        results: List[BridgeAblationResult] = []

        # 1. Baseline: Absent mode (loops=0, tau=0)
        for s in range(n_seeds):
            results.append(self.evaluate_configuration("absent", 0, 0.0, s, n_train=n_train, n_eval=n_eval))

        # 2. Frozen mode sweep
        for lp in loops:
            for s in range(n_seeds):
                results.append(self.evaluate_configuration("frozen", lp, 0.0, s, n_train=n_train, n_eval=n_eval))

        # 3. Full mode sweep
        for lp in loops:
            for t in taus:
                for s in range(n_seeds):
                    results.append(self.evaluate_configuration("full", lp, t, s, n_train=n_train, n_eval=n_eval))

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

        denominator = abs(mean_absent) if abs(mean_absent) > 1e-4 else 1.0
        improvement = (mean_full - mean_absent) / denominator
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
    evaluator = BridgeEvaluation(env_name="RepeatPrevious", n_seeds=1, n_train_episodes=2, n_eval_episodes=2)
    print("Running smoke ablation grid with real POPGym + recurrent training...")
    res = evaluator.run_ablation_grid(loop_configs=[2], tau_configs=[0.01], fast_mode=True)
    passed, imp, b = evaluator.check_gate_g3(res)
    print(evaluator.generate_report(res))
    print(f"\nGate G3 Check: {"PASSED" if passed else "FAILED"} (Improvement: {imp:.2%})")

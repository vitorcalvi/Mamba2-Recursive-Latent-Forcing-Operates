"""t2/dreamer_eval.py — DreamerV3 Comparison Grid: Mamba-2 vs GRU-RSSM.

Compares Mamba2WorldModel against parameter-matched GRURSSMWorldModel across
POPGym memory tasks (Autoencode, Battleship, RepeatPrevious) and random seeds.

Key Architecture:
1. Real POPGym environment interaction
2. Carries real recurrent state h_t across rollouts via wm.step()
3. Genuinely trains world model and actor-critic before evaluation
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
import torch.nn.functional as F

from t2.actor_critic import DreamerActorCritic
from t2.baselines.gru_rssm import GRURSSMWorldModel
from t2.popgym_wrapper import POPGymWrapper
from t2.world_model import Mamba2WorldModel


@dataclass
class ModelEvalResult:
    model_name: str
    task_name: str
    seed: int
    mean_return: float
    std_return: float
    wall_clock_sec: float
    param_count: int


class DreamerEvalGrid:
    """Orchestrates head-to-head evaluation between Mamba-2 and GRU world models."""

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        n_seeds: int = 5,
        n_train_episodes: int = 10,
        n_eval_episodes: int = 5,
        device: str = "cpu",
    ):
        self.tasks = tasks or ["Autoencode", "Battleship", "RepeatPrevious"]
        self.n_seeds = n_seeds
        self.n_train_episodes = n_train_episodes
        self.n_eval_episodes = n_eval_episodes
        self.device = torch.device(device)

    def evaluate_model_on_task(
        self,
        model_cls: Any,
        task: str,
        seed: int,
        model_name: str,
        n_train: Optional[int] = None,
        n_eval: Optional[int] = None,
    ) -> ModelEvalResult:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = POPGymWrapper(env_name=task, augment_obs=True, seed=seed)
        wm = model_cls(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            d_model=64,
            d_state=16,
            n_layers=2,
            stoch_dim=8,
            n_categories=8,
        ).to(self.device)

        param_count = sum(p.numel() for p in wm.parameters())
        feat_dim = wm.d_model + wm.flat_z
        ac = DreamerActorCritic(state_dim=feat_dim, act_dim=env.act_dim, hidden=64).to(self.device)

        opt = torch.optim.Adam(list(wm.parameters()) + list(ac.parameters()), lr=1e-3)
        train_eps = n_train if n_train is not None else self.n_train_episodes
        eval_eps = n_eval if n_eval is not None else self.n_eval_episodes

        t0 = time.perf_counter()

        # 1. Training Phase
        for ep in range(train_eps):
            obs, _ = env.reset(seed=seed + ep)
            state = wm.initial_state(batch_size=1, device=self.device)
            prev_action = torch.zeros(1, env.act_dim, device=self.device)
            done = False
            step_count = 0

            while not done and step_count < 64:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                obs_embed = wm.encoder(obs_t)
                state, prior, post = wm.step(state.detach(), prev_action, obs_embed)
                feat = torch.cat([state.h, state.z.flatten(start_dim=1)], dim=-1)

                logits = ac.actor(feat)
                value = ac.critic(feat)

                if np.random.rand() < 0.2:
                    action = torch.randint(0, env.act_dim, (1,)).item()
                else:
                    action = logits.argmax(dim=-1).item()

                next_obs, reward, term, trunc, _ = env.step(action)

                rec_obs = wm.decoder(feat)
                pred_rew = wm.reward_head(feat)
                loss_wm = F.mse_loss(rec_obs, obs_t) + F.mse_loss(pred_rew.squeeze(-1), torch.tensor([reward], device=self.device))
                loss_v = F.mse_loss(value, torch.tensor([[reward]], device=self.device))

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)[0, action]
                entropy = -(probs * F.log_softmax(logits, dim=-1)).sum()
                loss_pi = -(reward - value.detach().item()) * log_prob - 0.01 * entropy

                loss = loss_wm + loss_v + loss_pi
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()

                prev_action = torch.zeros(1, env.act_dim, device=self.device)
                prev_action[0, action] = 1.0
                obs = next_obs
                done = term or trunc
                step_count += 1

        # 2. Evaluation Phase
        wm.eval()
        ac.eval()
        returns = []

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
                    obs_embed = wm.encoder(obs_t)
                    state, _, _ = wm.step(state, prev_action, obs_embed)
                    feat = torch.cat([state.h, state.z.flatten(start_dim=1)], dim=-1)
                    action = ac.actor(feat).argmax(dim=-1).item()

                    prev_action = torch.zeros(1, env.act_dim, device=self.device)
                    prev_action[0, action] = 1.0

                    obs, reward, term, trunc, _ = env.step(action)
                    ep_ret += reward
                    done = term or trunc
                    step_count += 1

                returns.append(ep_ret)

        elapsed = time.perf_counter() - t0
        return ModelEvalResult(
            model_name=model_name,
            task_name=task,
            seed=seed,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            wall_clock_sec=elapsed,
            param_count=param_count,
        )

    def run_grid(self, tasks: Optional[List[str]] = None, fast_mode: bool = False) -> Dict[str, List[ModelEvalResult]]:
        target_tasks = tasks or self.tasks
        n_train = 2 if fast_mode else self.n_train_episodes
        n_eval = 2 if fast_mode else self.n_eval_episodes
        seeds_count = min(2, self.n_seeds) if fast_mode else self.n_seeds

        results: Dict[str, List[ModelEvalResult]] = {"mamba2": [], "gru": []}

        for task in target_tasks:
            for s in range(seeds_count):
                mamba_res = self.evaluate_model_on_task(
                    Mamba2WorldModel, task, s, "Mamba-2", n_train=n_train, n_eval=n_eval
                )
                results["mamba2"].append(mamba_res)

                gru_res = self.evaluate_model_on_task(
                    GRURSSMWorldModel, task, s, "GRU-RSSM", n_train=n_train, n_eval=n_eval
                )
                results["gru"].append(gru_res)

        return results

    def check_gate_g2(
        self,
        results: Dict[str, List[ModelEvalResult]],
    ) -> Tuple[bool, Dict[str, bool]]:
        task_verdicts: Dict[str, bool] = {}

        tasks = list({r.task_name for r in results["mamba2"]})
        for task in tasks:
            mamba_scores = [r.mean_return for r in results["mamba2"] if r.task_name == task]
            gru_scores = [r.mean_return for r in results["gru"] if r.task_name == task]

            mean_mamba = float(np.mean(mamba_scores))
            mean_gru = float(np.mean(gru_scores))

            # Mamba-2 must match or exceed GRU (within 5% tolerance margin)
            passed = mean_mamba >= (mean_gru - 0.05 * abs(mean_gru))
            task_verdicts[task] = passed

        n_passed = sum(task_verdicts.values())
        overall_passed = n_passed >= (len(tasks) * 2 // 3)

        return overall_passed, task_verdicts

    def generate_report(self, results: Dict[str, List[ModelEvalResult]]) -> str:
        lines = [
            "### SSM World-Model RL Comparison Grid (POPGym Benchmark)",
            "",
            "| Task | Model | Mean Return | Std | Wall-Clock (s) | Param Count |",
            "|:---|:---|:---|:---|:---|:---|",
        ]

        tasks = list({r.task_name for r in results["mamba2"]})
        for task in sorted(tasks):
            for model_key, display_name in [("mamba2", "**Mamba-2**"), ("gru", "GRU-RSSM")]:
                entries = [r for r in results[model_key] if r.task_name == task]
                scores = [e.mean_return for e in entries]
                stds = [e.std_return for e in entries]
                times = [e.wall_clock_sec for e in entries]
                params = entries[0].param_count if entries else 0

                lines.append(
                    f"| {task} | {display_name} | {np.mean(scores):.2f} | ±{np.std(scores):.2f} | {np.mean(times):.1f}s | {params:,} |"
                )

        return "\n".join(lines)


if __name__ == "__main__":
    grid = DreamerEvalGrid(tasks=["RepeatPrevious"], n_seeds=1, n_train_episodes=2, n_eval_episodes=2)
    print("Running smoke evaluation on real POPGym...")
    res = grid.run_grid(fast_mode=True)
    passed, verdicts = grid.check_gate_g2(res)
    print(grid.generate_report(res))
    print(f"\nGate G2 Check: {"PASSED" if passed else "FAILED"} (Verdicts: {verdicts})")

"""t2/dreamer_trainer.py — DreamerV3-style Model-Based RL Trainer with SSM Core.

Integrates:
- POPGymWrapper (POMDP memory benchmark environments)
- Mamba2WorldModel (Mamba-2 / SSM RSSM dynamics core)
- DreamerActorCritic (Symlog critic, EMA target network, lambda returns)
- ReplayBuffer (subsequence sampling over full episodes)
"""

from __future__ import annotations

import sys
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
from t2.popgym_wrapper import POPGymWrapper
from t2.world_model import Mamba2WorldModel, WorldModelState


class ReplayBuffer:
    """Stores full episodes and samples random subsequences."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.episodes: List[List[Dict[str, np.ndarray]]] = []
        self.total_steps = 0

    def add_episode(self, episode: List[Dict[str, np.ndarray]]) -> None:
        self.episodes.append(episode)
        self.total_steps += len(episode)
        while self.total_steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed)

    def is_ready(self, batch_size: int, batch_length: int) -> bool:
        if len(self.episodes) < batch_size:
            return False
        valid = [ep for ep in self.episodes if len(ep) >= batch_length]
        return len(valid) >= batch_size

    def sample(self, batch_size: int, batch_length: int) -> Dict[str, torch.Tensor]:
        valid_eps = [ep for ep in self.episodes if len(ep) >= batch_length]
        indices = np.random.choice(len(valid_eps), size=batch_size, replace=True)

        batch_obs, batch_act, batch_rew, batch_cont = [], [], [], []
        for idx in indices:
            ep = valid_eps[idx]
            start = np.random.randint(0, len(ep) - batch_length + 1)
            subseq = ep[start : start + batch_length]

            batch_obs.append(np.stack([s["obs"] for s in subseq]))
            batch_act.append(np.stack([s["act"] for s in subseq]))
            batch_rew.append(np.array([s["rew"] for s in subseq], dtype=np.float32))
            batch_cont.append(np.array([s["cont"] for s in subseq], dtype=np.float32))

        # [B, T, ...] -> [T, B, ...]
        obs_t = torch.from_numpy(np.stack(batch_obs)).transpose(0, 1).float()
        act_t = torch.from_numpy(np.stack(batch_act)).transpose(0, 1).float()
        rew_t = torch.from_numpy(np.stack(batch_rew)).transpose(0, 1).float()
        cont_t = torch.from_numpy(np.stack(batch_cont)).transpose(0, 1).float()

        return {"obs": obs_t, "act": act_t, "rew": rew_t, "cont": cont_t}


class DreamerTrainer:
    """End-to-end DreamerV3 MBRL trainer with SSM dynamics."""

    def __init__(
        self,
        env_name: str = "Autoencode",
        buffer_size: int = 50000,
        batch_size: int = 4,
        batch_length: int = 8,
        horizon: int = 5,
        lr_wm: float = 1e-4,
        lr_ac: float = 3e-5,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.horizon = horizon

        self.env = POPGymWrapper(env_name=env_name, augment_obs=True, seed=seed)
        self.obs_dim = self.env.obs_dim
        self.act_dim = self.env.act_dim

        self.world_model = Mamba2WorldModel(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            d_model=64,
            d_state=16,
            n_layers=2,
            stoch_dim=8,
            n_categories=8,
        ).to(self.device)

        feat_dim = self.world_model.d_model + self.world_model.flat_z
        self.actor_critic = DreamerActorCritic(
            state_dim=feat_dim,
            act_dim=self.act_dim,
            hidden=64,
            act_type="discrete",
        ).to(self.device)

        self.replay = ReplayBuffer(capacity=buffer_size)
        self.opt_wm = torch.optim.Adam(self.world_model.parameters(), lr=lr_wm)
        self.opt_ac = torch.optim.Adam(self.actor_critic.parameters(), lr=lr_ac)
        self.step_count = 0

    def collect_episode(self) -> float:
        obs, _ = self.env.reset()
        episode = []
        total_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                embed = self.world_model.encoder(obs_t)
                z_dummy = torch.zeros(1, self.world_model.flat_z, device=self.device)
                feat = torch.cat([embed, z_dummy], dim=-1)
                logits = self.actor_critic.actor(feat)
                action = torch.distributions.Categorical(logits=logits).sample().item()

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Action vector: one-hot
            act_vec = np.zeros(self.act_dim, dtype=np.float32)
            act_vec[action] = 1.0

            episode.append({
                "obs": obs,
                "act": act_vec,
                "rew": np.float32(reward),
                "cont": np.float32(0.0 if terminated else 1.0),
                "done": np.float32(done),
            })
            obs = next_obs

        self.replay.add_episode(episode)
        self.step_count += len(episode)
        return total_reward

    def train_step(self) -> Dict[str, float]:
        if not self.replay.is_ready(self.batch_size, self.batch_length):
            self.collect_episode()
            return {"wm_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0}

        batch = self.replay.sample(self.batch_size, self.batch_length)
        obs = batch["obs"].to(self.device)
        act = batch["act"].to(self.device)
        rew = batch["rew"].to(self.device)

        # 1. Update World Model
        self.opt_wm.zero_grad()
        wm_out = self.world_model(obs, act, rew)
        wm_loss = wm_out["total"]
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.opt_wm.step()

        # 2. Update Actor-Critic via imagination
        self.opt_ac.zero_grad()
        with torch.no_grad():
            s0 = self.world_model.initial_state(self.batch_size, self.device)
            def policy_fn(state: WorldModelState) -> torch.Tensor:
                feat = torch.cat([state.h, state.z.flatten(start_dim=1)], dim=-1)
                logits = self.actor_critic.actor(feat)
                idx = torch.distributions.Categorical(logits=logits).sample()
                return F.one_hot(idx, num_classes=self.act_dim).float()

            traj = self.world_model.imagine(s0, policy_fn, horizon=self.horizon)

        actor_l = self.actor_critic.actor_loss(traj)
        critic_l = self.actor_critic.critic_loss(traj)
        ac_loss = actor_l + critic_l
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 100.0)
        self.opt_ac.step()
        self.actor_critic.update_target()

        return {
            "wm_loss": float(wm_loss.detach().item()),
            "actor_loss": float(actor_l.detach().item()),
            "critic_loss": float(critic_l.detach().item()),
        }

    def evaluate(self, n_episodes: int = 3) -> float:
        returns = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ret = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    embed = self.world_model.encoder(obs_t)
                    z_dummy = torch.zeros(1, self.world_model.flat_z, device=self.device)
                    feat = torch.cat([embed, z_dummy], dim=-1)
                    action = self.actor_critic.actor(feat).argmax(dim=-1).item()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ret += reward
                done = terminated or truncated
            returns.append(ret)
        return float(np.mean(returns))


if __name__ == "__main__":
    trainer = DreamerTrainer(env_name="Autoencode", buffer_size=1000, batch_size=2, batch_length=4, horizon=3)
    print("Collecting initial warmup episodes...")
    for _ in range(3):
        r = trainer.collect_episode()
        print(f"  Warmup episode return: {r:.2f}")

    print("Executing train steps...")
    for step in range(3):
        metrics = trainer.train_step()
        print(f"  Step {step}: wm_loss={metrics['wm_loss']:.4f} actor_loss={metrics['actor_loss']:.4f}")

    mean_ret = trainer.evaluate(n_episodes=2)
    print(f"Evaluation mean return: {mean_ret:.2f}")
    print("DreamerTrainer self-test complete.")

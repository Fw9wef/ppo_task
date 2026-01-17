"""
PPO Agent for MuJoCo Half Cheetah environment using Gymnasium.
"""
from typing import Tuple
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

from model import ActorCritic
from buffer import RolloutBuffer


class Agent:
    """Proximal Policy Optimisation (PPO) Agent."""
    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        batch_size: int,
        ppo_epochs: int,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        total_updates: int,
        device: torch.device | None = None,
    ) -> None:
        self.lr = lr
        self.init_lr = self.lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.total_updates = total_updates
        self.device = device

        self.policy = ActorCritic(
            state_dim,
            action_dim,
            hidden_dim,
        ).to(self.device)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.buffer = RolloutBuffer(
            n_steps, num_envs, state_dim, action_dim, device
        )

    def select_action(
        self, state: np.ndarray, deterministic: bool = False,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action based on current policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor, deterministic)

        return action, log_prob, value.squeeze(-1)

    def calculate_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate Generalised Advantage Estimation (GAE)."""
        n_steps, num_envs = rewards.shape
        advantages = torch.zeros(n_steps, num_envs).to(self.device)

        gae = torch.zeros(num_envs).to(self.device)

        for t in reversed(range(n_steps)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

            next_value = values[t]

        return advantages

    def learn(self, next_state: np.ndarray, test_run: bool = False) -> None:
        """Update policy and value networks using PPO."""
        eps = 1e-8

        self.optimiser.param_groups[0]["lr"] = self.lr
        self.lr -= self.init_lr / self.total_updates

        states, actions, old_log_probs, rewards_2d, old_values_flat, dones_2d = self.buffer.get()

        old_values_2d = old_values_flat.reshape(rewards_2d.shape)

        with torch.no_grad():
            next_state_tensor = torch.tensor(
                next_state, dtype=torch.float32
            ).to(self.device)
            _, _, last_value_tensor = self.policy.forward(next_state_tensor)
            next_value = last_value_tensor.squeeze()

        advantages = self.calculate_gae(rewards_2d, old_values_2d, dones_2d, next_value)

        returns = advantages + old_values_2d

        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        rollout_size = len(states)

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(rollout_size)

            for start in range(0, rollout_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                batch_old_values = old_values_flat[batch_idx]

                current_log_probs, current_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )

                current_values = current_values.squeeze()

                ratios = torch.exp(current_log_probs - batch_old_log_probs)

                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) *
                    batch_advantages
                )

                actor_loss = - torch.min(surr1, surr2).mean()

                values_clipped = batch_old_values + torch.clamp(
                    current_values - batch_old_values,
                    - self.clip_epsilon,
                    self.clip_epsilon,
                )
                loss_v1 = F.mse_loss(current_values, batch_returns, reduction="none")
                loss_v2 = F.mse_loss(values_clipped, batch_returns, reduction="none")
                critic_loss = torch.max(loss_v1, loss_v2).mean()

                entropy_loss = - entropy.mean()

                loss = (
                    actor_loss +
                    self.value_coef * critic_loss +
                    self.entropy_coef * entropy_loss
                )

                self.optimiser.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) 

                self.optimiser.step()

                if test_run:
                    break
            
            if test_run:
                break

        self.buffer.clear()
        return (
            actor_loss.cpu().item(),
            critic_loss.cpu().item(),
            entropy_loss.cpu().item(),
        )

    def save_model(self, filepath: Path) -> None:
        """Save the model to the specified filepath."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "policy_state_dict": self.policy.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
            }
            torch.save(checkpoint, filepath)
            print(f"Model saved successfully to {filepath}")
        except IOError as e:
            print(f"I/O error({e.errno}) while saving model at {filepath}: {e.strerror}")

    def load_model(self, filepath: Path) -> None:
        """Load the model from the specified filepath."""
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device
            )
            policy_state = checkpoint.get("policy_state_dict")
            optimiser_state = checkpoint.get("optimiser_state_dict")

            self.policy.load_state_dict(policy_state, strict=False)
            self.optimiser.load_state_dict(optimiser_state)
            print(f"Model loaded successfully from {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")


class SimpleAgent:
    """Simple Agent for enjoying a trained PPO model."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | None = None,
    ):
        self.device = device

        self.policy = ActorCritic(
            state_dim, action_dim, hidden_dim
        ).to(self.device)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        """Select action based on the policy."""
        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.act(state_tensor, deterministic)

        return action.squeeze(0).cpu().numpy()

    def load_model(self, filepath: Path) -> None:
        """Load the model from the specified filepath."""
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device
            )
            policy_state = checkpoint.get("policy_state_dict")

            self.policy.load_state_dict(policy_state, strict=False)
            print(f"Model loaded successfully from {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")

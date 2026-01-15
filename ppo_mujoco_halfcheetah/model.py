"""
Actor-Critic Network for PPO agent in MuJoCo Half Cheetah environment.
"""
from typing import Tuple

import torch
from torch import nn
from torch.distributions.normal import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 3


class ActorCritic(nn.Module):
    """Actor-Critic Network for continuous action space."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_value_heads: int = 2,
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.value_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_value_heads)]
        )

        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
            self,
            state: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.body(state)

        action_mean = self.actor_head(x)

        action_log_std = self.log_std_head(x)
        action_log_std = torch.clamp(
            action_log_std, min=LOG_STD_MIN, max=LOG_STD_MAX
        )
        action_std = action_log_std.exp()

        state_values = torch.cat([head(x) for head in self.value_heads], dim=-1)

        return action_mean, action_std, state_values

    def act(
        self,
        state: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, state_values = self.forward(state)

        dist = Normal(action_mean, action_std)

        action = action_mean if deterministic else dist.sample()

        action_log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach(), action_log_prob.detach(), state_values.detach()

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, state_values = self.forward(state)

        dist = Normal(action_mean, action_std)

        action_log_prob = dist.log_prob(action).sum(dim=-1)

        dist_entropy = dist.entropy().sum(dim=-1)

        return action_log_prob, state_values, dist_entropy

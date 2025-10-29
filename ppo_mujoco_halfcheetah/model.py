from typing import Tuple

import torch
from torch import nn
from torch.distributions.normal import Normal


class ActorCritic(nn.Module):
    """Actor-Critic Network for continuous action space."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            state: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.critic(state)
        action_mean = self.actor(state)
        action_std = self.log_std.exp().expand_as(action_mean)

        return action_mean, action_std, value

    def get_action_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        action_log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, action_log_prob, value, entropy

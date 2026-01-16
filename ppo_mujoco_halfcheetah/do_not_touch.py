import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from pathlib import Path
from utils import load_config
from agent import Agent
import torch
from torch import nn


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )
        self.action_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        observation = np.zeros((2,), dtype=np.float32)
        info = {}
        return observation, info
    
    def step(self, action):
        self.step_count += 1
        observation = np.zeros((2,), dtype=np.float32)
        reward = np.array([float(self.step_count/2), float(self.step_count/5)])
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


def check_math(config_filename: Path = Path("config.yaml")):
    env_fns = [lambda: DummyEnv() for _ in range(2)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    
    config = load_config(config_filename)
    num_updates = 4
    n_steps = 10
    agent_config = {
        "n_steps": n_steps,
        "num_envs": 2,
        "state_dim": 2,
        "action_dim": 2,
        "hidden_dim": config["hidden_dim"],
        "lr": config["learning_rate"],
        "gamma": config["gamma"],
        "gae_lambda": config["gae_lambda"],
        "clip_epsilon": config["clip_epsilon"],
        "batch_size": config["batch_size"],
        "ppo_epochs": config["ppo_epochs"],
        "value_coef": config["value_coef"],
        "entropy_coef": config["entropy_coef"],
        "max_grad_norm": config["max_grad_norm"],
        "total_updates": num_updates,
    }
    agent = Agent(**agent_config)
    
    def init_ones(module):
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                module.weight.data.fill_(1e-3)
            if module.bias is not None:
                module.bias.data.fill_(1e-3)
    
    agent.policy.apply(init_ones)

    state, _ = envs.reset()

    for update in range(1, num_updates + 1):

        for step in range(n_steps):
            action = np.array([0., 0.], dtype=np.float32)
            action_tensor = torch.from_numpy(action)
            log_prob = torch.tensor([(1 + step) / (n_steps + 10)], dtype=torch.float32)
            value = torch.tensor([step * 1e-1], dtype=torch.float32)
            next_state, reward, terminated, _, _ = envs.step(action)

            agent.buffer.add(
                torch.tensor(state),
                action_tensor,
                log_prob,
                reward,
                value,
                terminated,
            )

            state = next_state

        agent.learn(next_state)
    
    # Compare the weights between the new network and the reference
    reference_agent = Agent(**agent_config)
    reference_agent.load_model(Path("reference_model.pt"))

    agent_state = agent.policy.state_dict()
    reference_state = reference_agent.policy.state_dict()
    
    all_equal = True
    for key in agent_state.keys():
        agent_weight = agent_state[key]
        reference_weight = reference_state[key]
        
        if not torch.allclose(agent_weight, reference_weight, rtol=1e-5, atol=1e-8):
            all_equal = False

    envs.close()
    return all_equal


if __name__ == "__main__":
    correct = check_math()
    print(f"-=-={correct}=-=-")

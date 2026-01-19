import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from pathlib import Path
from utils import load_config
from agent import Agent
from train import train
from enjoy import enjoy
import torch
from torch import nn
import numpy as np
import random
import os


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")


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
        reward = np.array([float(1e-2 * self.step_count/2), float(1e-2 * self.step_count/5)])
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


def get_config(config_filename: Path = Path("config.yaml")):
    config = load_config(config_filename)
    agent_config = {
        "n_steps": 10,
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
        "total_updates": 4,
        "num_reward_components": 2,
        "reward_components": [1.0, 0.1]
    }
    return agent_config


def check_math():
    set_seed(123)
    env_fns = [lambda: DummyEnv() for _ in range(2)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    
    agent_config = get_config()
    agent = Agent(**agent_config)
    
    def init_ones(module):
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                module.weight.data.fill_(1e-3)
            if module.bias is not None:
                module.bias.data.fill_(1e-3)
    
    agent.policy.apply(init_ones)

    state, _ = envs.reset()
    num_updates, n_steps = agent_config["total_updates"], agent_config["n_steps"]
    debug_losses = None
    for _ in range(1, num_updates + 1):

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

        losses = agent.learn(next_state)
        debug_losses = losses if debug_losses is None else debug_losses
    
    print(debug_losses)

    for loss, ref_loss, l_name in zip(
            debug_losses,
            [0.303911, 0.270159, -2.828396],
            ["policy", "value", "entropy"]
        ):
        if not np.allclose(loss, ref_loss, rtol=1e-2, atol=1e-3):
            print(f"{l_name} loss does not match with reference")
        
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


def check_run():
    try:
        model_path = train(Path("config.yaml"), True)
        enjoy(model_path, 1)

        # let's make sure that the trained model is compatible with the two-component config
        # this will tell us that train and enjoy were adapted and work with multi-component rewards
        agent_config = get_config()
        agent_config["num_envs"] = 8
        agent_config["state_dim"] = 17  # actual HalfCheetah env parameters
        agent_config["action_dim"] = 6
        agent = Agent(**agent_config)  # this model has a multihead value function
        agent.load_model(model_path)  # the newly trained model must be compatible
        return True
    except Exception as e:
        print(f"Error: Run check failed: {e}")
        return False


if __name__ == "__main__":
    correct_math = check_math()
    correct_run = check_run()
    print(f"-=-={correct_math and correct_run}=-=-")

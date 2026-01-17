"""
Training script for PPO agent in MuJoCo Half Cheetah environment.
"""
from pathlib import Path
from datetime import datetime

import torch

from agent import Agent
from environment import make_vec_env
from utils import set_seed, load_config, save_config


def train(config_filename: Path = Path("config.yaml"), test_run: bool = False):
    """Train the PPO agent in the MuJoCo Half Cheetah environment."""
    config = load_config(config_filename)
    set_seed(config["seed"])

    run_name = "ppo_" + config["env_id"] + "_" + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    log_dir = Path(config["log_dir"])
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config.copy(), run_dir / "config.yaml")

    env_id = config["env_id"]
    num_envs = config["num_envs"]
    n_steps = config["n_steps"]

    envs, state_dim, action_dim = make_vec_env(
        env_id,
        num_envs,
        render_mode=None,
    )

    print(f"| Environment: {env_id} (x{num_envs})"
          f"| State space: {state_dim}, Action space: {action_dim}")

    rollout_size = n_steps * num_envs
    num_updates = config["total_timesteps"] // rollout_size

    agent = Agent(
        n_steps=n_steps,
        num_envs=num_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        lr=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=config["clip_epsilon"],
        batch_size=config["batch_size"],
        ppo_epochs=config["ppo_epochs"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
        max_grad_norm=config["max_grad_norm"],
        total_updates=num_updates,
    )

    print(f"Total timesteps: {config['total_timesteps']}, Total updates: {num_updates}")

    state, _ = envs.reset()

    for update in range(1, num_updates + 1):

        for _ in range(n_steps):
            action_tensor, log_prob, value = agent.select_action(state)

            action = action_tensor.cpu().numpy()

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

            if test_run:
                break

        agent.learn(next_state, test_run)

        print(f"Update {update}/{num_updates} completed.")

        if test_run:
            break

    agent.save_model(run_dir / "final_model.pt")

    envs.close()
    print("Train completed")


if __name__ == "__main__":
    train()

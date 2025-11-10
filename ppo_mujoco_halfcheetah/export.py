"""
Push a trained PPO agent to the Hugging Face Hub.
"""
from typing import Tuple
import argparse
from pathlib import Path
from datetime import datetime
import json
import tempfile

import torch
import numpy as np
import gymnasium as gym
from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from agent import SimpleAgent
from environment import make_env
from utils import load_config, record_movie


def evaluate_agent(
    agent: SimpleAgent,
    env: gym.Env,
    n_eval: int,
    seed: int,
) -> Tuple[np.float64, np.float64]:
    """Evaluate the agent over a number of episodes."""
    scores = []

    for episode in range(n_eval):
        state, _ = env.reset(seed=seed+episode)
        score = 0.
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += float(reward)

        scores.append(score)

    env.close()

    avg_score = np.mean(scores, dtype=np.float64)
    std_score = np.std(scores, dtype=np.float64)

    return avg_score, std_score


def push_to_hub(
    username: str,
    repo_name: str,
    artifact_path: Path,
    movie_fps: int,
    n_eval: int,
    ):
    """Push a trained PPO agent to the Hugging Face Hub."""
    api = HfApi()

    repo_id = f"{username}/{repo_name}"
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)
    print(f"Repository created at {repo_url}")

    config_path = artifact_path.parent / "config.yaml"
    config = load_config(config_path)

    env_id = config["env_id"]
    hidden_dim = config["hidden_dim"]
    seed = config["seed"]

    env, state_dim, action_dim = make_env(
        env_id,
        render_mode="rgb_array",
    )

    agent = SimpleAgent(state_dim, action_dim, hidden_dim)
    agent.load_model(artifact_path)

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)

        torch.save(agent.policy.state_dict(), local_directory / "model.pt")

        with open(local_directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        mean_reward, std_reward = evaluate_agent(
            env=env,
            agent=agent,
            seed=seed,
            n_eval=n_eval,
        )

        evaluate_data = {
            "env_id": env_id,
            "mean_reward": mean_reward,
            "n_eval_episodes": n_eval,
            "eval_datetime": datetime.now().isoformat(),
        }

        with open(local_directory / "results.json", "w", encoding="utf-8") as f:
            json.dump(evaluate_data, f)

        env_name = env_id

        metadata = {
            "tags": [
                env_name,
                "reinforcement-learning",
                "ppo",
                "halfcheetah",
                "mujoco",
                "gymnasium",
                "pytorch",
            ]
        }

        eval_metadata = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
        )

        metadata = {**metadata, **eval_metadata}

        model_card = f"""
# Proximal Policy Optimization (PPO) Agent playing {env_name}

This is a trained Proximal Policy Optimization (PPO) agent for the MuJoCo {env_name} environment.

## Model Details

The model was trained using the code available [here](https://github.com/giansimone/ppo-mujoco-halfcheetah/).

## Usage
To load and use this model for inference:

```python
import torch
import json
import gymnasium as gym

from agent import SimpleAgent
from environment import make_env

#Load the configuration
with open("config.json", "r") as f:
    config = json.load(f)

env_id = config["env_id"]
hidden_dim = config["hidden_dim"]

# Create environment. Get action and space dimensions
env, state_size, action_size = make_env(
    env_id,
    render_mode="human",
)

# Instantiate the agent and load the trained policy network
agent = SimpleAgent(state_size, action_size, hidden_dim)
agent.policy.load_state_dict(torch.load("model.pt"))

# Enjoy the agent!
state, _ = env.reset()
done = False

while not done:
    action = agent.select_action(state, deterministic=True)
    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    env.render()

env.close()
```
"""
        readme_path = local_directory / "README.md"
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(model_card)

        metadata_save(readme_path, metadata)

        print("Recording movie...")
        movie_path = local_directory / "replay.mp4"
        record_movie(env, agent, movie_path, movie_fps)

        print("Uploading to Hugging Face Hub...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_directory,
            path_in_repo=".",
        )

        print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a trained PPO agent to Hugging Face Hub.")
    parser.add_argument(
        "--username",
        "-u",
        type=str,
        required=True,
        help="Your Hugging Face username.",
    )
    parser.add_argument(
        "--repo-name",
        "-r",
        type=str,
        required=True,
        help="The name of the repository to create on the Hub.",
    )
    parser.add_argument(
        "--artifact-path",
        "-a",
        type=str,
        required=True,
        help="Path to the trained model artifact (.pt or .pth file).",
    )
    parser.add_argument(
        "--movie-fps",
        "-f",
        type=int,
        default=30,
        help="The fps value to record the movie.",
    )
    parser.add_argument(
        "--n-eval",
        "-n",
        type=int,
        default=10,
        help="The number of episodes to evaluate for metrics.",
    )

    args = parser.parse_args()

    push_to_hub(
        username=args.username,
        repo_name=args.repo_name,
        artifact_path=Path(args.artifact_path),
        movie_fps=args.movie_fps,
        n_eval=args.n_eval,
    )

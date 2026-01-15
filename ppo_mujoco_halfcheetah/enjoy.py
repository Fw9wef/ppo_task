"""
Module to enjoy a trained PPO agent playing MuJoCo Half Cheetah environment.
"""
import argparse
from pathlib import Path

from agent import SimpleAgent
from environment import make_env
from utils import load_config, combine_reward_components


def enjoy(artifact_path: Path, n_episodes: int) -> None:
    """Enjoy a trained PPO agent on the MuJoCo Half Cheetah environment."""
    config_path = artifact_path.parent / "config.yaml"
    config = load_config(config_path)

    env_id = config["env_id"]
    hidden_dim = config["hidden_dim"]

    env, state_dim, action_dim = make_env(
        env_id,
        render_mode="human",
    )

    agent = SimpleAgent(state_dim, action_dim, hidden_dim)
    agent.load_model(artifact_path)

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.

        while not done:
            action = agent.select_action(state, deterministic=True)

            next_state, reward_components, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            state = next_state
            episode_reward += float(combine_reward_components(reward_components))

        print(f"MuJoCo Half Cheetah Episode {episode} | Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        "-a",
        type=str,
        required=True,
        help="The path to the trained model artifact to play MuJoCo Half Cheetah.",
    )
    parser.add_argument(
        "--num-episodes",
        "-n",
        type=int,
        default=10,
        help="The number of MuJoCo Half Cheetah episodes to enjoy.",
    )
    args = parser.parse_args()

    artifact = Path(args.artifact)
    num_episodes = args.num_episodes

    enjoy(artifact_path=artifact, n_episodes=num_episodes)

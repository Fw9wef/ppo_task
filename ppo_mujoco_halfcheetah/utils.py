"""
Module for utility functions and configuration file handling.
"""
import random
from pathlib import Path

import torch
import numpy as np
import yaml
import gymnasium as gym
import imageio.v3 as iio

from agent import SimpleAgent


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the available device (CPU, CUDA, or MPS)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(filepath: Path) -> dict:
    """Load the YAML configuration file into a standard Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, filepath: Path) -> None:
    """Save a standard Python dictionary into a YAML configuration file."""
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def record_movie(env: gym.Env, agent: SimpleAgent, filepath: Path, fps: int):
    """Record a movie of the agent interacting with the environment."""
    images = []
    done = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)

    while not done:
        action = agent.select_action(state, deterministic=True)

        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state

        img = env.render()
        images.append(img)

    iio.imwrite(
        filepath,
        images,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )
    print("done")

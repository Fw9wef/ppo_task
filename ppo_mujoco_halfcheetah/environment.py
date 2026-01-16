"""
Module to create and manage the MuJoCo Half Cheetah environment.
"""
from typing import Tuple

import numpy as np
import gymnasium as gym


REWARD_COMPONENT_COUNT = 2
FORWARD_REWARD_WEIGHT = 1.0
CTRL_COST_WEIGHT = 0.1


def _get_info_value(info: dict, keys: tuple[str, ...], default):
    for key in keys:
        if key in info:
            return info[key]
    return default


class HalfCheetahRewardComponentWrapper(gym.Wrapper):
    """Wrap a single env to return reward components instead of scalar reward."""
    def _extract_components(self, reward, info: dict) -> np.ndarray:
        forward = _get_info_value(info, ("reward_forward", "reward_run"), reward)
        ctrl = _get_info_value(info, ("reward_ctrl",), np.zeros_like(reward))

        forward_weight = getattr(self.env.unwrapped, "forward_reward_weight", 1.0)
        ctrl_weight = getattr(self.env.unwrapped, "ctrl_cost_weight", 1.0)

        if forward_weight != 0:
            forward = np.asarray(forward) / forward_weight
        if ctrl_weight != 0:
            ctrl = np.asarray(ctrl) / ctrl_weight

        reward_components = np.stack([forward, ctrl]).astype(np.float32)
        return reward_components

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_components = self._extract_components(reward, info)
        return obs, reward_components, terminated, truncated, info


class HalfCheetahRewardComponentVectorWrapper(gym.vector.VectorWrapper):
    """Wrap a vector env to return reward components instead of scalar reward."""
    def _get_weight(self, name: str, default: float) -> float:
        if hasattr(self.env, "get_attr"):
            values = self.env.get_attr(name)
            if values:
                return values[0]
        return getattr(self.env, name, default)

    def _extract_components(self, reward, info: dict) -> np.ndarray:
        forward = _get_info_value(info, ("reward_forward", "reward_run"), reward)
        ctrl = _get_info_value(info, ("reward_ctrl",), np.zeros_like(reward))

        forward_weight = self._get_weight("forward_reward_weight", 1.0)
        ctrl_weight = self._get_weight("ctrl_cost_weight", 1.0)

        if forward_weight != 0:
            forward = np.asarray(forward) * forward_weight
        if ctrl_weight != 0:
            ctrl = np.asarray(ctrl) * ctrl_weight

        reward_components = np.stack([forward, ctrl], axis=-1).astype(np.float32)
        return reward_components

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        reward_components = self._extract_components(reward, info)
        return obs, reward_components, terminated, truncated, info


def make_env(
    env_id: str,
    render_mode: str | None = None,
) -> Tuple[gym.Env, int, int]:
    "Create and wrap environment."
    env = gym.make(
        env_id,
        render_mode=render_mode,
        camera_name="track",
        forward_reward_weight=FORWARD_REWARD_WEIGHT,
        ctrl_cost_weight=CTRL_COST_WEIGHT,
    )

    env = gym.wrappers.ClipAction(env)
    env = HalfCheetahRewardComponentWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    return env, state_dim, action_dim


def make_vec_env(
    env_id: str,
    num_envs: int,
    render_mode: str | None = None,
) -> Tuple[gym.vector.VectorEnv, int, int]:
    "Create and wrap environment."
    envs = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="async",
        render_mode=render_mode,
        forward_reward_weight=FORWARD_REWARD_WEIGHT,
        ctrl_cost_weight=CTRL_COST_WEIGHT,
    )

    envs = gym.wrappers.vector.ClipAction(envs)
    envs = HalfCheetahRewardComponentVectorWrapper(envs)

    env_state_dim = envs.single_observation_space.shape[0]
    env_action_dim = envs.single_action_space.shape[0]

    return envs, env_state_dim, env_action_dim

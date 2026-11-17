"""Fixed wall-color visual shift for matched MiniGrid adaptation experiments."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from env_wrappers import AddInverseRGBChannelsWrapper, make_wrapper


CANONICAL_WALL_RGB = (100, 100, 100)


def validate_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
        raise ValueError(f"Expected an RGB triplet in [0, 255], got {rgb}")
    return tuple(int(value) for value in rgb)


class FixedWallColorWrapper(gym.Wrapper):
    """Repaint exact wall interiors while preserving geometry and transition logic."""

    def __init__(
        self,
        env: gym.Env,
        *,
        wall_rgb: tuple[int, int, int] | None,
    ) -> None:
        super().__init__(env)
        space = env.observation_space
        if (
            not isinstance(space, gym.spaces.Box)
            or len(space.shape) != 3
            or space.shape[-1] != 3
            or space.dtype != np.uint8
        ):
            raise ValueError("FixedWallColorWrapper expects HWC uint8 RGB")
        self.wall_rgb = None if wall_rgb is None else validate_rgb(wall_rgb)
        raw = self.unwrapped
        height, width = space.shape[:2]
        if height % raw.height or width % raw.width:
            raise ValueError("Rendered dimensions must divide the MiniGrid grid")
        self._tile_h = height // raw.height
        self._tile_w = width // raw.width
        if self._tile_h != self._tile_w:
            raise ValueError("Wall recoloring requires square rendered tiles")
        self._empty_tile = Grid.render_tile(
            None,
            agent_dir=None,
            highlight=False,
            tile_size=self._tile_h,
        )
        self.current_wall_mask = np.zeros((height, width), dtype=np.float32)

    def _wall_mask(self) -> np.ndarray:
        raw = self.unwrapped
        mask = np.zeros(self.observation_space.shape[:2], dtype=np.float32)
        for gx in range(raw.width):
            for gy in range(raw.height):
                cell = raw.grid.get(gx, gy)
                if getattr(cell, "type", None) != "wall":
                    continue
                rendered = Grid.render_tile(
                    cell,
                    agent_dir=None,
                    highlight=False,
                    tile_size=self._tile_h,
                )
                footprint = np.any(rendered != self._empty_tile, axis=-1)
                rows = slice(gy * self._tile_h, (gy + 1) * self._tile_h)
                cols = slice(gx * self._tile_w, (gx + 1) * self._tile_w)
                mask[rows, cols][footprint] = 1.0
        return mask

    def _apply_wall_colour(self, observation: np.ndarray) -> np.ndarray:
        wall_mask = self._wall_mask()
        self.current_wall_mask = wall_mask
        if self.wall_rgb is None:
            return observation
        result = observation.copy()
        result[wall_mask > 0] = np.asarray(self.wall_rgb, dtype=np.uint8)
        return result

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        observation = self._apply_wall_colour(observation)
        info["attn_mask"] = self.current_wall_mask.copy()
        info["fixed_wall_rgb"] = self.wall_rgb
        return observation, info

    def step(self, action):
        previous_mask = self.current_wall_mask.copy()
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self._apply_wall_colour(observation)
        # The transition stored by PPO contains the pre-action observation.
        info["attn_mask"] = previous_mask
        info["fixed_wall_rgb"] = self.wall_rgb
        return observation, reward, terminated, truncated, info


def make_fixed_wall_color_vec_env(
    *,
    env_id: str,
    n_envs: int,
    seed: int,
    wall_rgb: tuple[int, int, int] | None,
    observation_channels: int,
    open_door_reward: float = 0.0,
    pick_key_reward: float = 0.0,
):
    if observation_channels not in (3, 6):
        raise ValueError("Fixed wall-color adaptation supports RGB3 or RGB6")
    base_wrapper = make_wrapper(
        use_rgb_img_obs=True,
        invert_rgb_channels=False,
        open_door_reward=open_door_reward,
        pick_key_reward=pick_key_reward,
    )

    def wrap(env: gym.Env) -> gym.Env:
        wrapped = FixedWallColorWrapper(base_wrapper(env), wall_rgb=wall_rgb)
        if observation_channels == 6:
            wrapped = AddInverseRGBChannelsWrapper(wrapped)
        return wrapped

    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"highlight": False},
        wrapper_class=wrap,
    )
    return VecTransposeImage(env)


def restore_canonical_walls(
    observations: np.ndarray,
    wall_masks: np.ndarray,
    canonical_rgb: tuple[int, int, int] = CANONICAL_WALL_RGB,
) -> np.ndarray:
    """Restore NCHW RGB3/RGB6 wall pixels for evaluation and tests."""
    canonical_rgb = validate_rgb(canonical_rgb)
    if observations.ndim != 4 or observations.shape[1] not in (3, 6):
        raise ValueError("Expected NCHW RGB3 or RGB6 observations")
    if wall_masks.shape != observations.shape[0:1] + observations.shape[2:]:
        raise ValueError("Wall masks must match batch and spatial dimensions")
    result = observations.copy()
    for index in range(len(result)):
        mask = wall_masks[index] > 0
        for channel, value in enumerate(canonical_rgb):
            result[index, channel][mask] = value
            if result.shape[1] == 6:
                result[index, channel + 3][mask] = 255 - value
    return result

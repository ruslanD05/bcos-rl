from __future__ import annotations


import collections
import random

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from minigrid.core.constants import COLOR_TO_IDX, DIR_TO_VEC, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from stable_baselines3.common.monitor import Monitor

class FloatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if isinstance(space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=space.low.astype(np.float32),
                high=space.high.astype(np.float32),
                dtype=np.float32,
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32)


class AddInverseRGBChannelsWrapper(gym.ObservationWrapper):
    """Expose RGB and inverse-RGB channels as the actual model observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 3:
            raise ValueError(
                "AddInverseRGBChannelsWrapper expects an HWC Box observation"
            )
        if space.shape[-1] != 3 or space.dtype != np.uint8:
            raise ValueError(
                "AddInverseRGBChannelsWrapper expects uint8 RGB observations, "
                f"got shape={space.shape}, dtype={space.dtype}"
            )
        height, width, channels = space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, channels * 2),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate((obs, 255 - obs), axis=-1)


class WallAppearanceWrapper(gym.ObservationWrapper):
    """Change rendered walls without changing their collision behavior."""

    MODES = ("normal", "hidden", "recolored")

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str = "normal",
        color: str = "yellow",
    ) -> None:
        super().__init__(env)
        if mode not in self.MODES:
            raise ValueError(f"wall appearance must be one of {self.MODES}, got {mode}")
        if color not in COLOR_TO_IDX:
            raise ValueError(f"Unknown MiniGrid wall color: {color}")
        space = env.observation_space
        if (
            not isinstance(space, gym.spaces.Box)
            or len(space.shape) != 3
            or space.shape[-1] != 3
            or space.dtype != np.uint8
        ):
            raise ValueError("WallAppearanceWrapper expects an HWC uint8 RGB image")

        raw = env.unwrapped
        height, width = space.shape[:2]
        if height % raw.height or width % raw.width:
            raise ValueError(
                "Rendered observation dimensions must be divisible by grid size"
            )
        self.mode = mode
        self.color = color
        self._tile_h = height // raw.height
        self._tile_w = width // raw.width
        if self._tile_h != self._tile_w:
            raise ValueError("Wall appearance replacement requires square tiles")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "normal":
            return obs

        result = obs.copy()
        replacement_cell = None if self.mode == "hidden" else Wall(self.color)
        replacement = Grid.render_tile(
            replacement_cell,
            agent_dir=None,
            highlight=False,
            tile_size=self._tile_h,
        )
        raw = self.unwrapped
        for gx in range(raw.width):
            for gy in range(raw.height):
                cell = raw.grid.get(gx, gy)
                if getattr(cell, "type", None) != "wall":
                    continue
                rows = slice(gy * self._tile_h, (gy + 1) * self._tile_h)
                cols = slice(gx * self._tile_w, (gx + 1) * self._tile_w)
                result[rows, cols] = replacement
        return result


class RandomWallColorWrapper(WallAppearanceWrapper):
    """Render physically unchanged walls in one sampled color per episode."""

    def __init__(self, env: gym.Env, *, colors: tuple[str, ...]) -> None:
        if not colors:
            raise ValueError("RandomWallColorWrapper requires at least one color")
        normalized = tuple(str(color).lower() for color in colors)
        unknown = [color for color in normalized if color not in COLOR_TO_IDX]
        if unknown:
            raise ValueError(f"Unknown MiniGrid wall colors: {unknown}")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Random wall colors must not contain duplicates")
        self.colors = normalized
        self._rng = np.random.default_rng()
        super().__init__(env, mode="recolored", color=self.colors[0])

    @property
    def current_wall_color(self) -> str:
        return self.color

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        index = int(self._rng.integers(0, len(self.colors)))
        self.color = self.colors[index]
        return super().reset(seed=seed, options=options)


class RandomFloorColorWrapper(gym.ObservationWrapper):
    """Recolor rendered floor interiors using one random color per episode.

    Grid lines, walls, and rendered object/agent silhouettes are preserved.
    The intervention is visual only and does not modify MiniGrid's symbolic
    grid, transitions, rewards, or collision rules.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        colors: tuple[tuple[int, int, int], ...],
    ) -> None:
        super().__init__(env)
        space = env.observation_space
        if (
            not isinstance(space, gym.spaces.Box)
            or len(space.shape) != 3
            or space.shape[-1] != 3
            or space.dtype != np.uint8
        ):
            raise ValueError("RandomFloorColorWrapper expects an HWC uint8 RGB image")
        if not colors:
            raise ValueError("RandomFloorColorWrapper requires at least one color")
        normalized = []
        for color in colors:
            if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
                raise ValueError(f"Invalid RGB floor color: {color}")
            normalized.append(tuple(int(channel) for channel in color))

        raw = env.unwrapped
        height, width = space.shape[:2]
        if height % raw.height or width % raw.width:
            raise ValueError("Rendered observation dimensions must divide the grid")
        self.colors = tuple(normalized)
        self._tile_h = height // raw.height
        self._tile_w = width // raw.width
        if self._tile_h != self._tile_w:
            raise ValueError("Floor recoloring requires square rendered tiles")
        empty = Grid.render_tile(
            None,
            agent_dir=None,
            highlight=False,
            tile_size=self._tile_h,
        )
        self._empty_tile = empty
        self._empty_interior = np.all(empty == 0, axis=-1)
        self._rng = np.random.default_rng()
        self._current_color = self.colors[0]

    @property
    def current_floor_color(self) -> tuple[int, int, int]:
        return self._current_color

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        index = int(self._rng.integers(0, len(self.colors)))
        self._current_color = self.colors[index]
        return super().reset(seed=seed, options=options)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        result = obs.copy()
        raw = self.unwrapped
        for gx in range(raw.width):
            for gy in range(raw.height):
                cell = raw.grid.get(gx, gy)
                if getattr(cell, "type", None) == "wall":
                    continue
                rows = slice(gy * self._tile_h, (gy + 1) * self._tile_h)
                cols = slice(gx * self._tile_w, (gx + 1) * self._tile_w)
                block = result[rows, cols]
                semantic_pixels = np.any(block != self._empty_tile, axis=-1)
                floor_pixels = self._empty_interior & ~semantic_pixels
                block[floor_pixels] = self._current_color
        return result


class OneHotGridObsWrapper(gym.ObservationWrapper):
    """Encode a full MiniGrid grid as a channels-last one-hot spatial tensor."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        raw = env.unwrapped
        height, width = raw.height, raw.width
        self.object_offset = 0
        self.color_offset = len(OBJECT_TO_IDX)
        self.state_offset = self.color_offset + len(COLOR_TO_IDX)
        # FullyObsWrapper stores agent direction (0..3) in the same field used
        # for ordinary object state (open/closed/locked, 0..2).
        self.num_state_values = max(len(STATE_TO_IDX), len(DIR_TO_VEC))
        self.num_features = self.state_offset + self.num_state_values
        one_hot_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(height, width, self.num_features),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            "image": one_hot_space,
        })

    def observation(self, obs: dict) -> dict:
        # MiniGrid's Grid.encode() is indexed [x, y, field]. Convert it to the
        # conventional image/mask layout [row=y, column=x, field].
        image = obs["image"].transpose(1, 0, 2)
        height, width = image.shape[:2]
        encoded = np.zeros((height, width, self.num_features), dtype=np.float32)
        rows, cols = np.indices((height, width))
        encoded[rows, cols, self.object_offset + image[..., 0]] = 1.0
        encoded[rows, cols, self.color_offset + image[..., 1]] = 1.0
        encoded[rows, cols, self.state_offset + image[..., 2]] = 1.0
        return {**obs, "image": encoded}

class RandomShortcutWrapper(gym.Wrapper):
    """
    Creates a 'hole' in the DoorKey wall in X% of episodes.
    """
    def __init__(self, env: gym.Env, shortcut_prob: float = 0.5):
        super().__init__(env)
        self.shortcut_prob = shortcut_prob

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if random.random() < self.shortcut_prob:
            env = self.unwrapped
            
            wall_col = None
            for x in range(env.width):
                if (x == 0) or (x == env.width - 1):
                    continue
                tile = env.grid.get(x, 1)
                if tile is not None and tile.type == "wall":
                    wall_col = x
                    break
            
            if wall_col is not None:
                hole_y = random.randint(1, env.height - 2)
                env.grid.set(wall_col, hole_y, None)
                obs = env.gen_obs()
                
        return obs, info
class MiniGridGenericStatsWrapper(gym.Wrapper):
    """
    Generic wrapper that safely tracks keys and doors if they exist, and always tracks general success.
    Works on any MiniGrid environment.
    """
    def __init__(self, env: gym.Env, open_door_reward: float = 0.0, pick_key_reward: float = 0.0):
        super().__init__(env)
        self.open_door_reward = open_door_reward
        self.pick_key_reward = pick_key_reward
        self._picked_key = False
        self._opened_door = False
        self._door_positions: list[tuple[int, int]] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._picked_key = False
        self._opened_door = False
        self._door_positions = []
        
        env = self.unwrapped
        if hasattr(env, "grid") and env.grid is not None:
            for x in range(env.width):
                for y in range(env.height):
                    obj = env.grid.get(x, y)
                    if obj is not None and getattr(obj, "type", None) == "door":
                        self._door_positions.append((x, y))
                        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env = self.unwrapped

        # Check for key pickup
        carrying = getattr(env, "carrying", None)
        if carrying is not None and getattr(carrying, "type", None) == "key" and not self._picked_key:
            self._picked_key = True
            reward += self.pick_key_reward

        # Check for door opening
        for dx, dy in self._door_positions:
            door = env.grid.get(dx, dy)
            if (door is not None and getattr(door, "type", None) == "door" 
                and getattr(door, "is_open", False) and not self._opened_door):
                self._opened_door = True
                reward += self.open_door_reward

        # Always output these keys with defaults so Monitor wrapper never fails
        info["picked_key"] = float(self._picked_key)
        info["opened_door"] = float(self._opened_door)
        
        # Success is defined as reaching the goal (terminated) with a positive reward
        info["is_success"] = float(terminated and reward > 0)

        return obs, reward, terminated, truncated, info
class MiniGridInventoryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _update_inventory_slot(self):
        env = self.unwrapped
        env.grid.set(0, 0, None)
        if env.carrying is not None:
            env.grid.set(0, 0, env.carrying)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_inventory_slot()
        return obs, info  # FullyObsWrapper/RGBImgObsWrapper above rebuilds from grid

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_inventory_slot()
        if self.unwrapped.render_mode == "human":
            self.env.render()
        return obs, reward, terminated, truncated, info
class MiniGridStatsCallback(BaseCallback):
    """
    Dynamically logs custom metrics tracked by the generic wrapper.
    """
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        # A single rolling window of recent episode records. All rates below
        # (aggregate and per-mission) are derived from this one window so
        # they stay mutually consistent — a per-key deque-per-metric scheme
        # would have each metric's window cover a different span of training
        # (e.g. a per-mission key only appends on ~half of episodes, so its
        # window would span ~2x as many timesteps as the aggregate's).
        self.episode_history: collections.deque = collections.deque(maxlen=window_size)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if infos is None or dones is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue
            record = {
                key: float(info[key])
                for key in ["picked_key", "opened_door", "is_success"]
                if key in info
            }
            if "mission" in info:
                record["mission"] = info["mission"]
            self.episode_history.append(record)

        if not self.episode_history:
            return True

        for key in ["picked_key", "opened_door", "is_success"]:
            values = [r[key] for r in self.episode_history if key in r]
            if values:
                self.logger.record(f"custom/{key}_rate", float(np.mean(values)))

        # Per-mission breakdown (only present on mission-conditioned envs,
        # e.g. MissionRoomEnv) — lets us tell "ignoring the mission cue"
        # (one mission ~100%, the other ~0%) apart from "undertrained"
        # (both missions similarly low/high).
        missions = {r["mission"] for r in self.episode_history if "mission" in r}
        for mission in missions:
            values = [
                r["is_success"] for r in self.episode_history
                if r.get("mission") == mission and "is_success" in r
            ]
            if values:
                self.logger.record(f"custom/is_success_{mission}_rate", float(np.mean(values)))

        return True
    
class ExplanationMaskWrapper(gym.Wrapper):
    """
    Adds ``info['attn_mask']`` — a float32 (H, W) array — to every step and
    reset, encoding the sequential attention target for DoorKey-style tasks.

    Phase logic
    ───────────
    0  Hunt key    : agent not carrying → highlight key tile.
    1  Unlock door : carrying key, door still locked → highlight door tile.
    2  Reach goal  : door open (or absent) → highlight goal tile.

    The rendered agent silhouette is included in the positive region in every
    phase. Its surrounding floor and grid lines remain neutral.

    For rendered RGB observations, targets follow the exact object silhouette
    produced by MiniGrid rather than filling the object's complete tile. For
    symbolic observations, one observation cell remains one target cell.

    The mask is built from the *current* environment grid state before the
    action is executed (stored in ``_prev_mask`` and returned in the subsequent
    ``step()`` info dict), so the mask aligns with ``self._last_obs`` in
    ``collect_rollouts``.

    Must be placed after ``ImgObsWrapper`` in the wrapper stack. The first two
    observation dimensions are spatial, so the same wrapper supports rendered
    RGB images (one grid tile spans several pixels) and symbolic one-hot grids
    (one grid tile spans one observation cell).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        include_walls_as_negative: bool = False,
        include_grid_lines_as_negative: bool = False,
    ) -> None:
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        raw = env.unwrapped
        self._tile_h = h // raw.height
        self._tile_w = w // raw.width
        self._obs_h  = h
        self._obs_w  = w
        self.include_walls_as_negative = include_walls_as_negative
        self.include_grid_lines_as_negative = include_grid_lines_as_negative
        self._prev_mask: np.ndarray = np.zeros((h, w), dtype=np.float32)

    # ----------------------------------------------------------------- gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_mask = self._compute_mask()
        info["attn_mask"] = self._prev_mask.copy()
        return obs, info

    def step(self, action):
        # Return the mask for the state BEFORE this action (= self._last_obs).
        mask_for_current_state = self._prev_mask
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Pre-compute mask for the new state; used on the next step.
        self._prev_mask = self._compute_mask()
        info["attn_mask"] = mask_for_current_state
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- internals
    def _tile_slices(self, gx: int, gy: int) -> tuple[slice, slice]:
        r0 = gy * self._tile_h
        c0 = gx * self._tile_w
        return (
            slice(r0, r0 + self._tile_h),
            slice(c0, c0 + self._tile_w),
        )

    def _fill_tile(self, mask: np.ndarray, gx: int, gy: int) -> None:
        """Set the complete observation cell at (gx, gy) to 1."""
        rows, cols = self._tile_slices(gx, gy)
        mask[rows, cols] = 1.0

    def _rendered_footprint(
        self,
        cell,
        *,
        agent_dir: int | None = None,
        empty_interior: bool = False,
    ) -> np.ndarray:
        """Return the pixels representing an object, agent, or empty cell.

        RGB masks follow MiniGrid's own renderer. Object footprints are the
        pixels that differ from an empty rendered tile, while an agent
        footprint is differenced against the same underlying cell without the
        agent. Symbolic observations have one pixel per cell and therefore
        retain the original full-cell target.
        """
        if self._tile_h == 1 and self._tile_w == 1:
            return np.ones((1, 1), dtype=bool)
        if self._tile_h != self._tile_w:
            raise ValueError(
                "Rendered MiniGrid masks require square tiles, got "
                f"{self._tile_h}x{self._tile_w}"
            )

        tile_size = self._tile_h
        base_cell = cell if agent_dir is not None else None
        background = Grid.render_tile(
            base_cell,
            agent_dir=None,
            highlight=False,
            tile_size=tile_size,
        )
        if empty_interior:
            # Empty RGB tiles are black except for the grey top/left grid
            # lines. Those shared grid lines are not evidence of emptiness.
            return np.all(background == 0, axis=-1)

        rendered = Grid.render_tile(
            cell,
            agent_dir=agent_dir,
            highlight=False,
            tile_size=tile_size,
        )
        return np.any(rendered != background, axis=-1)

    def _fill_semantic_target(
        self,
        mask: np.ndarray,
        gx: int,
        gy: int,
        value: float,
        *,
        cell=None,
        agent_dir: int | None = None,
        empty_interior: bool = False,
    ) -> None:
        footprint = self._rendered_footprint(
            cell,
            agent_dir=agent_dir,
            empty_interior=empty_interior,
        )
        rows, cols = self._tile_slices(gx, gy)
        block = mask[rows, cols]
        block[footprint] = value

    def _fill_grid_line_targets(
        self,
        mask: np.ndarray,
        value: float,
    ) -> None:
        """Label only MiniGrid's rendered top/left cell separator pixels."""
        if self._tile_h == 1 and self._tile_w == 1:
            return
        empty_tile = Grid.render_tile(
            None,
            agent_dir=None,
            highlight=False,
            tile_size=self._tile_h,
        )
        line_pixels = np.any(empty_tile != 0, axis=-1)
        for gx in range(self.unwrapped.width):
            for gy in range(self.unwrapped.height):
                rows, cols = self._tile_slices(gx, gy)
                block = mask[rows, cols]
                block[line_pixels] = value

    def _compute_mask(self) -> np.ndarray:
        env  = self.unwrapped
        mask = np.zeros((self._obs_h, self._obs_w), dtype=np.float32)

        if self.include_grid_lines_as_negative:
            self._fill_grid_line_targets(mask, -1.0)

        has_key = getattr(env, "carrying", None) is not None

        key_pos = door_pos = goal_pos = None
        door_open = False
        for gx in range(env.width):
            for gy in range(env.height):
                cell = env.grid.get(gx, gy)
                if cell is None:
                    continue
                t = cell.type
                if t == "wall" and self.include_walls_as_negative:
                    self._fill_semantic_target(
                        mask, gx, gy, -1.0, cell=cell
                    )
                if t == "key" and not has_key:
                    key_pos = (gx, gy)
                elif t == "door":
                    door_pos = (gx, gy)
                    door_open = getattr(cell, "is_open", False)
                elif t == "goal":
                    goal_pos = (gx, gy)

        # Phase determination
        if not has_key:
            phase = 0
        elif door_pos is not None and not door_open:
            phase = 1
        else:
            phase = 2

        if phase == 0 and key_pos is not None:
            self._fill_semantic_target(
                mask, *key_pos, 1.0, cell=env.grid.get(*key_pos)
            )
        elif phase == 1:
            if door_pos is not None:
                self._fill_semantic_target(
                    mask, *door_pos, 1.0, cell=env.grid.get(*door_pos)
                )
        elif phase == 2 and goal_pos is not None:
            self._fill_semantic_target(
                mask, *goal_pos, 1.0, cell=env.grid.get(*goal_pos)
            )

        ax, ay = (int(v) for v in env.agent_pos)
        self._fill_semantic_target(
            mask,
            ax,
            ay,
            1.0,
            cell=env.grid.get(ax, ay),
            agent_dir=int(env.agent_dir),
        )

        return mask


SequentialMaskWrapper = ExplanationMaskWrapper  # backward-compat alias


class FlippedExplanationMaskWrapper(ExplanationMaskWrapper):
    """Swap P and N while preserving the sequential mask's neutral region."""

    def _compute_mask(self) -> np.ndarray:
        return -super()._compute_mask()


class AllNegativeSequentialMaskWrapper(ExplanationMaskWrapper):
    """Sequential mask with only the agent and current objective in P.

    Every pixel outside those exact rendered footprints is N. Unlike the
    original sequential mask, this profile has no neutral region.
    """

    def _compute_mask(self) -> np.ndarray:
        mask = super()._compute_mask()
        return np.where(mask > 0.0, 1.0, -1.0).astype(np.float32)


class FlippedAllNegativeSequentialMaskWrapper(AllNegativeSequentialMaskWrapper):
    """Swap P and N in the strict sequential mask."""

    def _compute_mask(self) -> np.ndarray:
        return -super()._compute_mask()


class WallGridNegativeFloorPositiveMaskWrapper(ExplanationMaskWrapper):
    """
    Produces info['attn_mask'] with:
      -1.0 (Negative) on all perimeter/dividing wall pixels and grid-line borders.
      +1.0 (Positive) everywhere else (all room floor interiors, items, and agent).
    """

    def _compute_mask(self) -> np.ndarray:
        mask = np.ones((self._obs_h, self._obs_w), dtype=np.float32)
        self._fill_grid_line_targets(mask, -1.0)
        env = self.unwrapped
        for gx in range(env.width):
            for gy in range(env.height):
                cell = env.grid.get(gx, gy)
                if cell is not None and getattr(cell, "type", None) == "wall":
                    rows, cols = self._tile_slices(gx, gy)
                    mask[rows, cols] = -1.0
        return mask


class StaticMultiObjectMaskWrapper(ExplanationMaskWrapper):
    """
    Adds ``info['attn_mask']`` — a float32 (H, W) array — highlighting all
    four core semantic entities (Agent, Door, Key, Goal) simultaneously,
    instead of the single active sub-goal used by ``ExplanationMaskWrapper``.

    Non-sequential / static: the composite mask does not depend on task
    phase. Per-object silhouette masks are combined via logical OR (equivalently,
    an elementwise clamp of the summed masks) into one binary target that is
    applied uniformly at every step.
    """

    def _compute_mask(self) -> np.ndarray:
        env  = self.unwrapped
        mask = np.zeros((self._obs_h, self._obs_w), dtype=np.float32)

        has_key = getattr(env, "carrying", None) is not None

        # Agent tile
        ax, ay = env.agent_pos
        self._fill_semantic_target(
            mask,
            ax,
            ay,
            1.0,
            cell=env.grid.get(ax, ay),
            agent_dir=int(env.agent_dir),
        )

        # Door, Key (while still on the grid), Goal tiles
        for gx in range(env.width):
            for gy in range(env.height):
                cell = env.grid.get(gx, gy)
                if cell is None:
                    continue
                if cell.type in ("door", "key", "goal"):
                    self._fill_semantic_target(mask, gx, gy, 1.0, cell=cell)

        # Key picked up → highlighted via the inventory slot instead, since
        # it no longer occupies a grid cell.
        if has_key:
            inventory_cell = env.grid.get(0, 0)
            if inventory_cell is not None:
                self._fill_semantic_target(
                    mask, 0, 0, 1.0, cell=inventory_cell
                )

        # Logical OR across the per-object masks (clamp keeps it binary;
        # _fill_tile already writes 1.0, so this is a no-op safeguard).
        return np.clip(mask, 0.0, 1.0)


class SignedActionMaskWrapper(ExplanationMaskWrapper):
    """
    Produces ``info['attn_mask']`` as a *signed* (H, W) float32 array with values
    in {-1, 0, +1}, conditioned on BOTH the current state and the action taken
    this step (unlike the binary, state-only masks above).

        +1  the front object/interior should positively support the action,
        -1  the front object/interior should negatively affect the action,
         0  unsupervised (no gradient from the explanation loss).

    The mask is computed from the grid state *before* the action executes and the
    action itself, so it aligns exactly with the ``(obs, action)`` pair stored in
    the rollout buffer (obs = state before the action; action = the action whose
    logit is explained in ``BcosPPO.train``).

    Conservative rules (everything not listed stays neutral):

    forward  front traversable → +1 ; front blocked (wall/closed door/etc.) → -1
    pickup   front pickable & hands empty → +1 ; otherwise → -1
    toggle   openable door in front → +1 ; locked door w/o matching key →
             neutral (or -1 if ``neg_on_locked_door_without_key``) ;
             non-toggleable object in front → -1
    left/right/drop/done  → neutral

    ``enable_negative=False`` suppresses all -1 assignments (positive-only
    ablation). ``enabled_actions`` restricts which actions are supervised.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        enabled_actions: tuple[str, ...] = ("forward", "pickup", "toggle"),
        enable_negative: bool = True,
        neg_on_locked_door_without_key: bool = False,
        forward_traversable_positive: bool = True,
    ) -> None:
        super().__init__(env)
        self.enabled_actions = set(enabled_actions)
        self.enable_negative = enable_negative
        self.neg_on_locked_door_without_key = neg_on_locked_door_without_key
        self.forward_traversable_positive = forward_traversable_positive

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # No action has been taken yet → fully neutral mask.
        info["attn_mask"] = np.zeros((self._obs_h, self._obs_w), dtype=np.float32)
        return obs, info

    def step(self, action):
        # Compute the signed mask from the pre-step state + the taken action,
        # BEFORE stepping, so it matches (self._last_obs, action).
        mask = self._compute_signed_mask(int(action))
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["attn_mask"] = mask
        return obs, reward, terminated, truncated, info

    def _compute_signed_mask(self, action_idx: int) -> np.ndarray:
        env = self.unwrapped
        mask = np.zeros((self._obs_h, self._obs_w), dtype=np.float32)
        actions = env.actions

        fx, fy = (int(v) for v in env.front_pos)
        in_bounds = (0 <= fx < env.width) and (0 <= fy < env.height)
        front = env.grid.get(fx, fy) if in_bounds else None
        carrying = getattr(env, "carrying", None)

        def put(sign: int) -> None:
            if sign < 0 and not self.enable_negative:
                return
            if in_bounds:
                self._fill_semantic_target(
                    mask,
                    fx,
                    fy,
                    float(sign),
                    cell=front,
                    empty_interior=front is None,
                )

        if action_idx == int(actions.forward) and "forward" in self.enabled_actions:
            traversable = (front is None) or front.can_overlap()
            if traversable:
                if self.forward_traversable_positive:
                    put(+1)
            else:
                put(-1)

        elif action_idx == int(actions.pickup) and "pickup" in self.enabled_actions:
            if front is not None and front.can_pickup() and carrying is None:
                put(+1)
            else:  # empty, non-pickable, or hands already occupied
                put(-1)

        elif action_idx == int(actions.toggle) and "toggle" in self.enabled_actions:
            if front is not None and getattr(front, "type", None) == "door":
                if not getattr(front, "is_locked", False):
                    put(+1)  # unlocked door: toggle validly opens/closes it
                else:
                    has_key = (
                        carrying is not None
                        and getattr(carrying, "type", None) == "key"
                        and getattr(carrying, "color", None) == getattr(front, "color", None)
                    )
                    if has_key:
                        put(+1)                       # can unlock → supports toggle
                    elif self.neg_on_locked_door_without_key:
                        put(-1)                       # locked, no key → inhibit
                    # else: neutral (toggle is a no-op; conservative default)
            elif front is not None:  # non-toggleable object (wall, key, goal, ...)
                put(-1)
            # empty front cell → neutral

        # left / right / drop / done → all-neutral (zeros)
        return mask


class ActionCategoryMaskWrapper(ExplanationMaskWrapper):
    """Localize attribution magnitude according to the action category.

    Movement actions supervise the rendered agent plus the adjacent cell in
    the direction that action would face. Interaction actions supervise the
    agent plus the current front cell. The target is deliberately unsigned:
    it specifies where attribution magnitude should occur, not its polarity.
    ``done`` receives no supervision.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["attn_mask"] = np.zeros(
            (self._obs_h, self._obs_w), dtype=np.float32
        )
        return obs, info

    def step(self, action):
        mask = self._compute_action_category_mask(int(action))
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["attn_mask"] = mask
        return obs, reward, terminated, truncated, info

    def _put_agent(self, mask: np.ndarray) -> None:
        env = self.unwrapped
        ax, ay = (int(v) for v in env.agent_pos)
        self._fill_semantic_target(
            mask,
            ax,
            ay,
            1.0,
            cell=env.grid.get(ax, ay),
            agent_dir=int(env.agent_dir),
        )

    def _put_cell(self, mask: np.ndarray, gx: int, gy: int) -> None:
        env = self.unwrapped
        if not (0 <= gx < env.width and 0 <= gy < env.height):
            return
        cell = env.grid.get(gx, gy)
        self._fill_semantic_target(
            mask,
            gx,
            gy,
            1.0,
            cell=cell,
            empty_interior=cell is None,
        )

    def _compute_action_category_mask(self, action_idx: int) -> np.ndarray:
        env = self.unwrapped
        actions = env.actions
        mask = np.zeros((self._obs_h, self._obs_w), dtype=np.float32)

        movement_directions = {
            int(actions.forward): int(env.agent_dir),
            int(actions.left): (int(env.agent_dir) - 1) % 4,
            int(actions.right): (int(env.agent_dir) + 1) % 4,
        }
        interaction_actions = {
            int(actions.pickup),
            int(actions.toggle),
            int(actions.drop),
        }

        if action_idx in movement_directions:
            self._put_agent(mask)
            direction = movement_directions[action_idx]
            dx, dy = (int(v) for v in DIR_TO_VEC[direction])
            ax, ay = (int(v) for v in env.agent_pos)
            self._put_cell(mask, ax + dx, ay + dy)
        elif action_idx in interaction_actions:
            self._put_agent(mask)
            fx, fy = (int(v) for v in env.front_pos)
            self._put_cell(mask, fx, fy)

        return mask


class LocalActionAffordanceMaskWrapper(ExplanationMaskWrapper):
    """Action-conditioned P/N mask using only local affordance information.

    The mask deliberately has no DoorKey phase or global object knowledge.
    Movement actions use the agent and a square local neighborhood as P,
    except that interactive-object silhouettes are always neutral (D).
    Interaction actions use the agent and the front-cell affordance as P only
    when that action can currently have an effect. Every other pixel is N, so
    the masked-absolute-energy objective suppresses reliance on distant state.

    RGB6 observations are recommended: inverse RGB channels make empty black
    floor pixels non-zero and therefore explainable as traversability evidence.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        movement_radius: int = 1,
        emit_all_action_masks: bool = False,
    ) -> None:
        super().__init__(env)
        if movement_radius < 1:
            raise ValueError("movement_radius must be at least 1")
        self.movement_radius = int(movement_radius)
        self.emit_all_action_masks = bool(emit_all_action_masks)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["attn_mask"] = np.zeros(
            (self._obs_h, self._obs_w), dtype=np.float32
        )
        return obs, info

    def step(self, action):
        mask = self._compute_local_action_mask(int(action))
        all_action_masks = None
        if self.emit_all_action_masks:
            all_action_masks = np.stack(
                [
                    self._compute_local_action_mask(action_idx)
                    for action_idx in range(int(self.action_space.n))
                ]
            ).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["attn_mask"] = mask
        if all_action_masks is not None:
            info["all_action_attn_masks"] = all_action_masks
        return obs, reward, terminated, truncated, info

    def _put_agent(self, mask: np.ndarray) -> None:
        env = self.unwrapped
        ax, ay = (int(value) for value in env.agent_pos)
        self._fill_semantic_target(
            mask,
            ax,
            ay,
            1.0,
            cell=env.grid.get(ax, ay),
            agent_dir=int(env.agent_dir),
        )

    def _put_movement_neighborhood(self, mask: np.ndarray) -> None:
        env = self.unwrapped
        ax, ay = (int(value) for value in env.agent_pos)
        radius = self.movement_radius
        for gx in range(max(0, ax - radius), min(env.width, ax + radius + 1)):
            for gy in range(max(0, ay - radius), min(env.height, ay + radius + 1)):
                rows, cols = self._tile_slices(gx, gy)
                mask[rows, cols] = 1.0
        # Keep the target exact for the agent itself, rather than attributing
        # the whole agent tile to the rendered sprite.
        rows, cols = self._tile_slices(ax, ay)
        mask[rows, cols] = -1.0
        self._put_agent(mask)

    def _neutralize_interactive_objects(self, mask: np.ndarray) -> None:
        """Keep task-object appearance outside the movement objective."""
        env = self.unwrapped
        neutral_types = {"key", "door", "goal", "box", "ball"}
        for gx in range(env.width):
            for gy in range(env.height):
                cell = env.grid.get(gx, gy)
                if cell is None or getattr(cell, "type", None) not in neutral_types:
                    continue
                self._fill_semantic_target(
                    mask,
                    gx,
                    gy,
                    0.0,
                    cell=cell,
                )

    def _front_affordance_is_valid(self, action_idx: int) -> bool:
        env = self.unwrapped
        actions = env.actions
        front = env.grid.get(*(int(value) for value in env.front_pos))
        carrying = getattr(env, "carrying", None)

        if action_idx == int(actions.pickup):
            return (
                front is not None
                and bool(front.can_pickup())
                and carrying is None
            )
        if action_idx == int(actions.toggle):
            if front is None or not hasattr(front, "toggle"):
                return False
            if getattr(front, "type", None) != "door":
                return False
            if not bool(getattr(front, "is_locked", False)):
                return True
            return (
                carrying is not None
                and getattr(carrying, "type", None) == "key"
                and getattr(carrying, "color", None)
                == getattr(front, "color", None)
            )
        if action_idx == int(actions.drop):
            return carrying is not None and front is None
        return False

    def _put_valid_front_affordance(
        self,
        mask: np.ndarray,
        action_idx: int,
    ) -> None:
        if not self._front_affordance_is_valid(action_idx):
            return
        env = self.unwrapped
        fx, fy = (int(value) for value in env.front_pos)
        front = env.grid.get(fx, fy)
        self._fill_semantic_target(
            mask,
            fx,
            fy,
            1.0,
            cell=front,
            empty_interior=front is None,
        )

    def _compute_local_action_mask(self, action_idx: int) -> np.ndarray:
        env = self.unwrapped
        actions = env.actions
        mask = np.full((self._obs_h, self._obs_w), -1.0, dtype=np.float32)

        movement_actions = {
            int(actions.left),
            int(actions.right),
            int(actions.forward),
        }
        interaction_actions = {
            int(actions.pickup),
            int(actions.toggle),
            int(actions.drop),
        }

        if action_idx in movement_actions:
            self._put_movement_neighborhood(mask)
            self._neutralize_interactive_objects(mask)
            # If the agent overlaps a neutral object such as the goal, the
            # rendered agent itself remains positive.
            self._put_agent(mask)
        elif action_idx in interaction_actions:
            self._put_agent(mask)
            self._put_valid_front_affordance(mask, action_idx)
        else:
            # ``done`` is unused in DoorKey and receives no supervision.
            mask.fill(0.0)
        return mask


class LocalDirectionalActionMaskWrapper(LocalActionAffordanceMaskWrapper):
    """Strict local mask with one action-relative movement cell in P.

    For ``forward``, ``left``, and ``right``, P contains the rendered agent
    and respectively the front, left-adjacent, or right-adjacent cell. Every
    other pixel is N, including unrelated key, door, and goal pixels. This
    profile therefore requires no semantic object knowledge for movement.
    Interaction actions retain the valid-front-affordance behavior of
    :class:`LocalActionAffordanceMaskWrapper`; ``done`` remains unsupervised.
    """

    def _put_directional_cell(self, mask: np.ndarray, direction: int) -> None:
        env = self.unwrapped
        dx, dy = (int(value) for value in DIR_TO_VEC[direction])
        ax, ay = (int(value) for value in env.agent_pos)
        gx, gy = ax + dx, ay + dy
        if not (0 <= gx < env.width and 0 <= gy < env.height):
            return
        rows, cols = self._tile_slices(gx, gy)
        mask[rows, cols] = 1.0

    def _compute_local_action_mask(self, action_idx: int) -> np.ndarray:
        env = self.unwrapped
        actions = env.actions
        mask = np.full((self._obs_h, self._obs_w), -1.0, dtype=np.float32)

        movement_directions = {
            int(actions.forward): int(env.agent_dir),
            int(actions.left): (int(env.agent_dir) - 1) % 4,
            int(actions.right): (int(env.agent_dir) + 1) % 4,
        }
        interaction_actions = {
            int(actions.pickup),
            int(actions.toggle),
            int(actions.drop),
        }

        if action_idx in movement_directions:
            self._put_directional_cell(mask, movement_directions[action_idx])
            self._put_agent(mask)
        elif action_idx in interaction_actions:
            self._put_agent(mask)
            self._put_valid_front_affordance(mask, action_idx)
        else:
            mask.fill(0.0)
        return mask


class CombinedExplanationMaskWrapper(SignedActionMaskWrapper):
    """Emit aligned sequential and signed action masks for the same transition.

    The masks intentionally remain separate because they encode different
    statements. ``sequential_attn_mask`` identifies the active task object,
    while ``action_attn_mask`` assigns a local positive/negative affordance to
    the action that was actually taken.
    """

    def __init__(self, env: gym.Env, *, blocker_masks_only: bool = False, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.blocker_masks_only = bool(blocker_masks_only)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_mask = self._compute_mask()
        info["sequential_attn_mask"] = self._prev_mask.copy()
        info["action_attn_mask"] = np.zeros(
            (self._obs_h, self._obs_w), dtype=np.float32
        )
        return obs, info

    def step(self, action):
        sequential_mask = self._prev_mask
        action_mask = (
            self._compute_blocker_mask(int(action))
            if self.blocker_masks_only
            else self._compute_signed_mask(int(action))
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._prev_mask = self._compute_mask()
        info["sequential_attn_mask"] = sequential_mask
        info["action_attn_mask"] = action_mask
        return obs, reward, terminated, truncated, info

    def _compute_blocker_mask(self, action_idx: int) -> np.ndarray:
        """Return only negative evidence for currently invalid actions."""
        env = self.unwrapped
        actions = env.actions
        mask = np.zeros((self._obs_h, self._obs_w), dtype=np.float32)
        fx, fy = (int(v) for v in env.front_pos)
        in_bounds = (0 <= fx < env.width) and (0 <= fy < env.height)
        front = env.grid.get(fx, fy) if in_bounds else None
        carrying = getattr(env, "carrying", None)

        def mark(gx: int, gy: int, cell) -> None:
            self._fill_semantic_target(mask, gx, gy, -1.0, cell=cell)

        if action_idx == int(actions.forward):
            traversable = in_bounds and ((front is None) or front.can_overlap())
            if not traversable and in_bounds:
                mark(fx, fy, front)
            return mask

        if action_idx == int(actions.pickup):
            valid = (
                in_bounds
                and front is not None
                and front.can_pickup()
                and carrying is None
            )
            target_type = "key"
        elif action_idx == int(actions.toggle):
            valid = False
            if in_bounds and front is not None and getattr(front, "type", None) == "door":
                valid = not getattr(front, "is_locked", False) or (
                    carrying is not None
                    and getattr(carrying, "type", None) == "key"
                    and getattr(carrying, "color", None)
                    == getattr(front, "color", None)
                )
            target_type = "door"
        else:
            return mask

        if valid:
            return mask
        for gx in range(env.width):
            for gy in range(env.height):
                cell = env.grid.get(gx, gy)
                if cell is not None and getattr(cell, "type", None) == target_type:
                    mark(gx, gy, cell)
        return mask


def compose_sequential_action_mask(
    sequential_mask: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """Compose task relevance and action affordance by clipped signed addition.

    Agreement reinforces the same sign, while a +1/-1 conflict cancels to 0
    and becomes explicitly neutral for the combined objective.
    """
    return np.clip(
        sequential_mask.astype(np.float32) + action_mask.astype(np.float32),
        -1.0,
        1.0,
    )


class LiteralCombinedMaskWrapper(SignedActionMaskWrapper):
    """Emit one signed P/N/D mask combining sequential and action targets."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_mask = self._compute_mask()
        action_mask = np.zeros_like(self._prev_mask)
        info["sequential_attn_mask"] = self._prev_mask.copy()
        info["action_attn_mask"] = action_mask
        info["attn_mask"] = compose_sequential_action_mask(
            self._prev_mask, action_mask
        )
        return obs, info

    def step(self, action):
        sequential_mask = self._prev_mask
        action_mask = self._compute_signed_mask(int(action))
        combined_mask = compose_sequential_action_mask(
            sequential_mask, action_mask
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._prev_mask = self._compute_mask()
        info["sequential_attn_mask"] = sequential_mask
        info["action_attn_mask"] = action_mask
        info["attn_mask"] = combined_mask
        return obs, reward, terminated, truncated, info


def make_wrapper(
    *,
    use_rgb_img_obs: bool,
    invert_rgb_channels: bool = False,
    wall_appearance: str = "normal",
    wall_color: str = "yellow",
    random_wall_colors: tuple[str, ...] | None = None,
    random_floor_colors: tuple[tuple[int, int, int], ...] | None = None,
    use_one_hot_symbolic_obs: bool = False,
    use_inventory_obs: bool = False,
    shortcut_prob: float = 0.0,
    open_door_reward: float = 0.0,
    pick_key_reward: float = 0.0,
    use_sequential_mask: bool = False,
    use_sequential_all_negative_mask: bool = False,
    use_wall_grid_negative_floor_positive_mask: bool = False,
    flip_sequential_mask: bool = False,
    sequential_walls_as_negative: bool = False,
    sequential_grid_lines_as_negative: bool = False,
    use_static_multi_mask: bool = False,
    use_signed_action_mask: bool = False,
    use_action_category_mask: bool = False,
    use_local_action_mask: bool = False,
    use_local_directional_mask: bool = False,
    local_action_movement_radius: int = 1,
    local_action_emit_all_masks: bool = False,
    use_combined_explanation_masks: bool = False,
    combined_blocker_masks_only: bool = False,
    use_literal_combined_mask: bool = False,
    signed_enable_negative: bool = True,
    signed_enabled_actions: tuple[str, ...] = ("forward", "pickup", "toggle"),
    signed_neg_on_locked_door_without_key: bool = False,
    action_forward_traversable_positive: bool = True,
):
    mask_flags = [
        use_sequential_mask,
        use_sequential_all_negative_mask,
        use_wall_grid_negative_floor_positive_mask,
        use_static_multi_mask,
        use_signed_action_mask,
        use_action_category_mask,
        use_local_action_mask,
        use_local_directional_mask,
        use_combined_explanation_masks,
        use_literal_combined_mask,
    ]
    if sum(bool(f) for f in mask_flags) > 1:
        raise ValueError(
            "Mask options are mutually exclusive."
        )
    if flip_sequential_mask and not (
        use_sequential_mask or use_sequential_all_negative_mask
    ):
        raise ValueError(
            "flip_sequential_mask requires a sequential mask mode"
        )
    if wall_appearance != "normal" and not use_rgb_img_obs:
        raise ValueError("wall appearance changes require RGB observations")
    if random_wall_colors and not use_rgb_img_obs:
        raise ValueError("random wall colors require RGB observations")
    if random_wall_colors and wall_appearance != "normal":
        raise ValueError(
            "random_wall_colors cannot be combined with a fixed wall appearance"
        )
    def _wrap(env: gym.Env) -> gym.Env:
        env = MiniGridGenericStatsWrapper(env, open_door_reward=open_door_reward, pick_key_reward=pick_key_reward)
        if use_inventory_obs:
            env = MiniGridInventoryWrapper(env)
        env = RandomShortcutWrapper(env, shortcut_prob=shortcut_prob)

        if use_rgb_img_obs:
            env = RGBImgObsWrapper(env)
            env = ImgObsWrapper(env)
            if random_wall_colors:
                env = RandomWallColorWrapper(env, colors=random_wall_colors)
            else:
                env = WallAppearanceWrapper(
                    env,
                    mode=wall_appearance,
                    color=wall_color,
                )
            if random_floor_colors:
                env = RandomFloorColorWrapper(env, colors=random_floor_colors)
            if invert_rgb_channels:
                env = AddInverseRGBChannelsWrapper(env)
        else:
            env = FullyObsWrapper(env)
            if use_one_hot_symbolic_obs:
                env = OneHotGridObsWrapper(env)
            env = ImgObsWrapper(env)
            if not use_one_hot_symbolic_obs:
                env = FloatObsWrapper(env)

        if use_sequential_mask:
            sequential_wrapper = (
                FlippedExplanationMaskWrapper
                if flip_sequential_mask
                else ExplanationMaskWrapper
            )
            env = sequential_wrapper(
                env,
                include_walls_as_negative=sequential_walls_as_negative,
                include_grid_lines_as_negative=sequential_grid_lines_as_negative,
            )
        elif use_sequential_all_negative_mask:
            strict_wrapper = (
                FlippedAllNegativeSequentialMaskWrapper
                if flip_sequential_mask
                else AllNegativeSequentialMaskWrapper
            )
            env = strict_wrapper(env)
        elif use_wall_grid_negative_floor_positive_mask:
            env = WallGridNegativeFloorPositiveMaskWrapper(env)
        elif use_static_multi_mask:
            env = StaticMultiObjectMaskWrapper(env)
        elif use_signed_action_mask:
            env = SignedActionMaskWrapper(
                env,
                enabled_actions=signed_enabled_actions,
                enable_negative=signed_enable_negative,
                neg_on_locked_door_without_key=signed_neg_on_locked_door_without_key,
                forward_traversable_positive=action_forward_traversable_positive,
            )
        elif use_action_category_mask:
            env = ActionCategoryMaskWrapper(env)
        elif use_local_action_mask:
            env = LocalActionAffordanceMaskWrapper(
                env,
                movement_radius=local_action_movement_radius,
                emit_all_action_masks=local_action_emit_all_masks,
            )
        elif use_local_directional_mask:
            env = LocalDirectionalActionMaskWrapper(
                env,
                emit_all_action_masks=local_action_emit_all_masks,
            )
        elif use_combined_explanation_masks:
            env = CombinedExplanationMaskWrapper(
                env,
                blocker_masks_only=combined_blocker_masks_only,
                enabled_actions=signed_enabled_actions,
                enable_negative=signed_enable_negative,
                neg_on_locked_door_without_key=signed_neg_on_locked_door_without_key,
                forward_traversable_positive=action_forward_traversable_positive,
            )
        elif use_literal_combined_mask:
            env = LiteralCombinedMaskWrapper(
                env,
                enabled_actions=signed_enabled_actions,
                enable_negative=signed_enable_negative,
                neg_on_locked_door_without_key=signed_neg_on_locked_door_without_key,
                forward_traversable_positive=action_forward_traversable_positive,
            )

        env = Monitor(
            env, info_keywords=("picked_key", "opened_door", "is_success"),
        )
        return env
    return _wrap

"""Oracle-action visual shortcut environment for MiniGrid DoorKey.

The entire top wall row of the DoorKey grid is colored according to the optimal
action chosen by a deterministic symbolic DoorKey expert.

The explanation mask marks the entire top row as negative (N = -1.0) and the rest
of the playable grid as positive (P = +1.0).
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Literal

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from env_wrappers import make_wrapper


ShortcutMode = Literal["correlated", "removed", "random", "permuted"]
ShortcutMaskProfile = Literal[
    "border_negative_screen_positive",
    "sequential",
    "negative_hint_removal",
    "oracle_patch_polarity",
    "patch_only_negative",
]

ACTION_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),      # left
    1: (0, 255, 0),      # right
    2: (0, 0, 255),      # forward
    3: (255, 255, 0),    # pickup
    4: (0, 255, 255),    # drop
    5: (255, 0, 255),    # toggle
    6: (255, 255, 255),  # done
}

# A fixed derangement for the five actions used by the DoorKey expert.
PERMUTED_ACTION: dict[int, int] = {
    0: 1,
    1: 2,
    2: 0,
    3: 5,
    4: 6,
    5: 3,
    6: 4,
}


class DoorKeyOracle:
    """Deterministic shortest-action controller using the symbolic grid state."""

    @staticmethod
    def _objects(env) -> tuple[tuple[int, int] | None, object | None, tuple[int, int] | None]:
        key_pos = None
        door = None
        goal_pos = None
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                kind = getattr(cell, "type", None)
                if kind == "key":
                    key_pos = (x, y)
                elif kind == "door":
                    door = cell
                elif kind == "goal":
                    goal_pos = (x, y)
        return key_pos, door, goal_pos

    @staticmethod
    def _door_position(env, door: object | None) -> tuple[int, int] | None:
        if door is None:
            return None
        for x in range(env.width):
            for y in range(env.height):
                if env.grid.get(x, y) is door:
                    return (x, y)
        return None

    @staticmethod
    def _front(pose: tuple[int, int, int]) -> tuple[int, int]:
        x, y, direction = pose
        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
        return x + dx, y + dy

    @classmethod
    def _plan(
        cls,
        env,
        goal: Callable[[tuple[int, int, int]], bool],
    ) -> list[int]:
        actions = env.actions
        start = (
            int(env.agent_pos[0]),
            int(env.agent_pos[1]),
            int(env.agent_dir),
        )
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pose, plan = queue.popleft()
            if goal(pose):
                return plan

            x, y, direction = pose
            candidates = [
                ((x, y, (direction - 1) % 4), int(actions.left)),
                ((x, y, (direction + 1) % 4), int(actions.right)),
            ]
            fx, fy = cls._front(pose)
            if 0 <= fx < env.width and 0 <= fy < env.height:
                front = env.grid.get(fx, fy)
                if front is None or bool(front.can_overlap()):
                    candidates.insert(
                        0,
                        ((fx, fy, direction), int(actions.forward)),
                    )

            for next_pose, action in candidates:
                if next_pose in visited:
                    continue
                visited.add(next_pose)
                queue.append((next_pose, [*plan, action]))
        return []

    @classmethod
    def action(cls, env) -> int:
        actions = env.actions
        key_pos, door, goal_pos = cls._objects(env)
        carrying_key = getattr(getattr(env, "carrying", None), "type", None) == "key"

        if not carrying_key and key_pos is not None:
            if tuple(int(v) for v in env.front_pos) == key_pos:
                return int(actions.pickup)
            plan = cls._plan(env, lambda pose: cls._front(pose) == key_pos)
            return plan[0] if plan else int(actions.done)

        door_pos = cls._door_position(env, door)
        if door_pos is not None and not bool(getattr(door, "is_open", False)):
            if tuple(int(v) for v in env.front_pos) == door_pos:
                return int(actions.toggle)
            plan = cls._plan(env, lambda pose: cls._front(pose) == door_pos)
            return plan[0] if plan else int(actions.done)

        if goal_pos is not None:
            plan = cls._plan(
                env,
                lambda pose: (pose[0], pose[1]) == goal_pos,
            )
            return plan[0] if plan else int(actions.done)
        return int(actions.done)


class OracleActionShortcutWrapper(gym.Wrapper):
    """Make the entire top wall row of the DoorKey grid into an oracle-action cue."""

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: ShortcutMode = "correlated",
        correlation_probability: float = 1.0,
        border_height: int = 8,
        patch_size: int = 6,
        mask_profile: ShortcutMaskProfile = "border_negative_screen_positive",
    ) -> None:
        super().__init__(env)
        if mode not in ("correlated", "removed", "random", "permuted"):
            raise ValueError(f"Unknown shortcut mode: {mode!r}")
        if not 0.0 <= correlation_probability <= 1.0:
            raise ValueError("correlation_probability must be in [0, 1]")
        space = env.observation_space
        if (
            not isinstance(space, gym.spaces.Box)
            or len(space.shape) != 3
            or space.shape[-1] != 3
            or space.dtype != np.uint8
        ):
            raise ValueError("OracleActionShortcutWrapper expects HWC uint8 RGB")

        self.mode = mode
        self.correlation_probability = float(correlation_probability)
        self.mask_profile = mask_profile
        self._rng = np.random.default_rng()
        self.current_oracle_action = 6
        self.current_patch_action: int | None = None
        self.observation_space = space
        # Calculate tile height: observation height / grid height (48 / 6 = 8 px)
        grid_h = getattr(self.unwrapped, "height", 6)
        self.tile_h = space.shape[0] // grid_h

    def _cue_action(self, oracle_action: int) -> int | None:
        if self.mode == "removed":
            return None
        if self.mode == "permuted":
            return PERMUTED_ACTION[oracle_action]
        expert_actions = (0, 1, 2, 3, 5)
        if self.mode == "random":
            return int(self._rng.choice(expert_actions))
        if self._rng.random() < self.correlation_probability:
            return oracle_action
        alternatives = [action for action in expert_actions if action != oracle_action]
        return int(self._rng.choice(alternatives))

    def _apply_shortcut_cue(self, obs: np.ndarray) -> np.ndarray:
        result = obs.copy()
        oracle_action = DoorKeyOracle.action(self.unwrapped)
        cue_action = self._cue_action(oracle_action)
        if cue_action is not None:
            # Color the entire top wall row (row 0) in the oracle action color
            result[: self.tile_h, :] = ACTION_COLORS[cue_action]
        self.current_oracle_action = oracle_action
        self.current_patch_action = cue_action
        return result

    def _expand_mask(self, info: dict) -> None:
        expected = self.observation_space.shape[:2]
        # Top row of tiles = N (-1.0), all other rows = P (+1.0)
        expanded = np.ones(expected, dtype=np.float32)
        expanded[: self.tile_h, :] = -1.0
        info["attn_mask"] = expanded

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        cue_observation = self._apply_shortcut_cue(obs)
        self._expand_mask(info)
        info["shortcut_oracle_action"] = self.current_oracle_action
        info["shortcut_patch_action"] = self.current_patch_action
        return cue_observation, info

    def step(self, action):
        oracle_action = self.current_oracle_action
        patch_action = self.current_patch_action
        obs, reward, terminated, truncated, info = self.env.step(action)
        cue_observation = self._apply_shortcut_cue(obs)
        self._expand_mask(info)
        info["shortcut_oracle_action"] = oracle_action
        info["shortcut_patch_action"] = patch_action
        return cue_observation, reward, terminated, truncated, info


def make_shortcut_vec_env(
    *,
    env_id: str = "MiniGrid-DoorKey-6x6-v0",
    n_envs: int = 1,
    seed: int = 0,
    mode: ShortcutMode = "correlated",
    correlation_probability: float = 1.0,
    border_height: int = 8,
    patch_size: int = 6,
    mask_profile: ShortcutMaskProfile = "border_negative_screen_positive",
    open_door_reward: float = 0.1,
    pick_key_reward: float = 0.05,
):
    """Construct the fully observable RGB shortcut environment used by scripts."""
    base_wrapper = make_wrapper(
        use_rgb_img_obs=True,
        invert_rgb_channels=False,
        use_sequential_mask=True,
        open_door_reward=open_door_reward,
        pick_key_reward=pick_key_reward,
    )

    def wrap(env: gym.Env) -> gym.Env:
        return OracleActionShortcutWrapper(
            base_wrapper(env),
            mode=mode,
            correlation_probability=correlation_probability,
            border_height=border_height,
            patch_size=patch_size,
            mask_profile=mask_profile,
        )

    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"highlight": False},
        wrapper_class=wrap,
    )
    return VecTransposeImage(env)


def paint_shortcut_patch(
    observations: np.ndarray,
    oracle_actions: np.ndarray,
    *,
    mode: ShortcutMode,
    patch_size: int = 6,
    border_height: int = 8,
) -> np.ndarray:
    """Counterfactually repaint the top wall row in a batch of NCHW observations."""
    result = observations.copy()
    tile_h = border_height  # default 8 px for 6x6 (48x48)
    if mode == "removed":
        # Grey wall color [100, 100, 100] when shortcut is removed
        result[:, 0, :tile_h, :] = 100
        result[:, 1, :tile_h, :] = 100
        result[:, 2, :tile_h, :] = 100
        return result
    if mode not in ("correlated", "permuted"):
        raise ValueError("Paired repainting supports correlated, removed, or permuted")
    for index, oracle_action in enumerate(oracle_actions.astype(np.int64)):
        cue_action = int(oracle_action)
        if mode == "permuted":
            cue_action = PERMUTED_ACTION[cue_action]
        color = np.asarray(ACTION_COLORS[cue_action], dtype=np.uint8)
        result[index, 0, :tile_h, :] = color[0]
        result[index, 1, :tile_h, :] = color[1]
        result[index, 2, :tile_h, :] = color[2]
    return result

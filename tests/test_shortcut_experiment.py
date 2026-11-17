from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from env_wrappers import make_wrapper
from shortcut_experiment_env import (
    ACTION_COLORS,
    DoorKeyOracle,
    OracleActionShortcutWrapper,
    PERMUTED_ACTION,
    make_shortcut_vec_env,
    paint_shortcut_patch,
)


def make_env(mode: str = "correlated") -> OracleActionShortcutWrapper:
    base = make_wrapper(
        use_rgb_img_obs=True,
        invert_rgb_channels=False,
        use_sequential_mask=True,
    )(gym.make("MiniGrid-DoorKey-6x6-v0", highlight=False))
    return OracleActionShortcutWrapper(base, mode=mode)


class ShortcutExperimentTests(unittest.TestCase):
    def test_oracle_completes_deterministic_doorkey_instances(self):
        env = make_env()
        try:
            for seed in range(10):
                _, _ = env.reset(seed=seed)
                for _ in range(200):
                    _, _, terminated, truncated, info = env.step(
                        DoorKeyOracle.action(env.unwrapped)
                    )
                    if terminated or truncated:
                        self.assertEqual(info["is_success"], 1.0)
                        break
                else:
                    self.fail(f"Oracle did not finish seed {seed}")
        finally:
            env.close()

    def test_border_patch_and_mask_are_exactly_aligned(self):
        env = make_env("correlated")
        try:
            obs, info = env.reset(seed=3)
            oracle = info["shortcut_oracle_action"]
            color = np.asarray(ACTION_COLORS[oracle], dtype=np.uint8)
            self.assertEqual(obs.shape, (48, 48, 3))
            self.assertTrue(np.all(obs[:8, :] == color))
            mask = info["attn_mask"]
            self.assertEqual(mask.shape, (48, 48))
            self.assertTrue(np.all(mask[:8, :] == -1))
            self.assertGreater(np.count_nonzero(mask[8:] > 0), 0)
            self.assertEqual(np.count_nonzero(mask[:8] > 0), 0)
        finally:
            env.close()

    def test_step_diagnostics_describe_the_pre_action_observation(self):
        env = make_env("correlated")
        try:
            _, reset_info = env.reset(seed=8)
            old_oracle = int(reset_info["shortcut_oracle_action"])
            _, _, _, _, step_info = env.step(old_oracle)
            self.assertEqual(step_info["shortcut_oracle_action"], old_oracle)
            self.assertEqual(step_info["shortcut_patch_action"], old_oracle)
        finally:
            env.close()

    def test_counterfactual_patch_repainting(self):
        observations = np.zeros((2, 3, 48, 48), dtype=np.uint8)
        actions = np.asarray([0, 3])
        correlated = paint_shortcut_patch(
            observations, actions, mode="correlated"
        )
        permuted = paint_shortcut_patch(
            observations, actions, mode="permuted"
        )
        removed = paint_shortcut_patch(observations, actions, mode="removed")
        for index, action in enumerate(actions):
            expected = np.asarray(ACTION_COLORS[int(action)])[:, None, None]
            self.assertTrue(np.all(correlated[index, :, :8, :] == expected))
            permuted_expected = np.asarray(
                ACTION_COLORS[PERMUTED_ACTION[int(action)]]
            )[:, None, None]
            self.assertTrue(
                np.all(permuted[index, :, :8, :] == permuted_expected)
            )
        self.assertTrue(np.all(removed[:, :, :8, :] == 100))

    def test_vectorized_observation_and_mask_shapes_match(self):
        env = make_shortcut_vec_env(n_envs=2, seed=2)
        try:
            obs = env.reset()
            self.assertEqual(obs.shape, (2, 3, 48, 48))
            _, _, _, infos = env.step(np.asarray([0, 1]))
            for info in infos:
                self.assertEqual(info["attn_mask"].shape, (48, 48))
        finally:
            env.close()

    def test_current_protocol_uses_full_top_row_regardless_of_legacy_parameters(self):
        base = make_wrapper(
            use_rgb_img_obs=True,
            invert_rgb_channels=False,
            use_sequential_mask=True,
        )(gym.make("MiniGrid-DoorKey-6x6-v0", highlight=False))
        env = OracleActionShortcutWrapper(
            base,
            mode="correlated",
            correlation_probability=1.0,
            border_height=36,
            patch_size=32,
            mask_profile="negative_hint_removal",
        )
        try:
            obs, info = env.reset(seed=0)
            mask = info["attn_mask"]
            self.assertEqual(obs.shape, (48, 48, 3))
            self.assertEqual(mask.shape, (48, 48))
            self.assertEqual(np.count_nonzero(mask < 0), 8 * 48)
            self.assertEqual(np.count_nonzero(mask > 0), mask.size - 8 * 48)
            self.assertEqual(np.count_nonzero(mask == 0), 0)
            self.assertTrue(np.all(mask[:8, :] == -1))
        finally:
            env.close()

    def test_current_protocol_keeps_full_top_row_for_other_legacy_profiles(self):
        base = make_wrapper(
            use_rgb_img_obs=True,
            invert_rgb_channels=False,
            use_sequential_mask=True,
        )(gym.make("MiniGrid-DoorKey-6x6-v0", highlight=False))
        env = OracleActionShortcutWrapper(
            base,
            mode="correlated",
            correlation_probability=1.0,
            border_height=36,
            patch_size=32,
            mask_profile="oracle_patch_polarity",
        )
        try:
            _, info = env.reset(seed=0)
            mask = info["attn_mask"]
            self.assertEqual(np.count_nonzero(mask < 0), 8 * 48)
            self.assertEqual(np.count_nonzero(mask > 0), mask.size - 8 * 48)
            self.assertEqual(np.count_nonzero(mask == 0), 0)
            self.assertTrue(np.all(mask[:8, :] == -1))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

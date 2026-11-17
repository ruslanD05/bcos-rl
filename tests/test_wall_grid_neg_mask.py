import unittest
import numpy as np
import gymnasium as gym
from minigrid_part.env_wrappers import make_wrapper

class WallGridNegFloorPosMaskTest(unittest.TestCase):
    def test_mask_dimensions_and_polarity(self):
        env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array", highlight=False)
        wrapper = make_wrapper(
            use_rgb_img_obs=True,
            invert_rgb_channels=True,
            use_wall_grid_negative_floor_positive_mask=True,
        )
        wrapped = wrapper(env)
        obs, info = wrapped.reset(seed=42)
        mask = info["attn_mask"]

        self.assertEqual(mask.shape, (64, 64))
        # Unique values should only be -1.0 and 1.0 (no 0.0)
        unique_vals = set(np.unique(mask))
        self.assertEqual(unique_vals, {-1.0, 1.0})

        # Negative region should be wall tiles + grid lines
        n_pixels = np.sum(mask < 0)
        p_pixels = np.sum(mask > 0)
        self.assertEqual(n_pixels + p_pixels, 64 * 64)
        self.assertGreater(n_pixels, 2000)
        self.assertGreater(p_pixels, 1000)

        # Check step
        obs, rew, term, trunc, info = wrapped.step(0)
        mask_step = info["attn_mask"]
        self.assertEqual(set(np.unique(mask_step)), {-1.0, 1.0})

if __name__ == "__main__":
    unittest.main()

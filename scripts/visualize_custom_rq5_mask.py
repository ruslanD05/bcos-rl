#!/usr/bin/env python3
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("c:/RLProject/bachelors_thesis_ruslan")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "minigrid_part"))

import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid_part.env_wrappers import make_wrapper, ExplanationMaskWrapper

class WallGridNegativeFloorPositiveMaskWrapper(ExplanationMaskWrapper):
    """
    Mask wrapper where:
      N (-1): All perimeter and dividing wall pixels + grid-line borders.
      P (+1): Everywhere else (all navigable floor tiles, agent, key, door, goal).
      D (0) : None.
    """
    def _compute_mask(self) -> np.ndarray:
        # Start with all pixels as Positive (+1.0)
        mask = np.ones((self._obs_h, self._obs_w), dtype=np.float32)

        # 1. Fill grid lines as Negative (-1.0)
        self._fill_grid_line_targets(mask, -1.0)

        # 2. Fill all wall cells as Negative (-1.0)
        env = self.unwrapped
        for gx in range(env.width):
            for gy in range(env.height):
                cell = env.grid.get(gx, gy)
                if cell is not None and cell.type == "wall":
                    # Fill entire wall tile / silhouette
                    rows, cols = self._tile_slices(gx, gy)
                    mask[rows, cols] = -1.0

        return mask

# 1. Generate DoorKey state
raw_env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array", highlight=False)
wrapper_cls = make_wrapper(
    use_rgb_img_obs=True,
    invert_rgb_channels=True,
    use_sequential_mask=True,
    sequential_walls_as_negative=True,
    sequential_grid_lines_as_negative=True,
)
env_seq = wrapper_cls(raw_env)
obs_seq, info_seq = env_seq.reset(seed=10000)
mask_seq = info_seq["attn_mask"]

if obs_seq.ndim == 3 and obs_seq.shape[0] == 6:
    rgb = np.moveaxis(obs_seq[:3], 0, -1).copy()
else:
    rgb = obs_seq[..., :3].copy()

# 2. Generate Custom Mask (N on grid + wall, P everywhere else)
raw_env_custom = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array", highlight=False)
img_env = make_wrapper(use_rgb_img_obs=True, invert_rgb_channels=True)(raw_env_custom)
custom_env = WallGridNegativeFloorPositiveMaskWrapper(img_env)
_, info_custom = custom_env.reset(seed=10000)
mask_custom = info_custom["attn_mask"]

# Render RGBA colors
def to_rgb_mask(m):
    out = np.full((*m.shape, 3), 0.82, dtype=np.float32) # grey
    out[m > 0] = [0.267, 0.698, 0.306] # Green P=+1
    out[m < 0] = [0.816, 0.224, 0.204] # Red N=-1
    return out

mask_seq_rgb = to_rgb_mask(mask_seq)
mask_custom_rgb = to_rgb_mask(mask_custom)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=220)

# Top row: Sequential Semantic Mask (Current RQ5 mask)
axes[0, 0].imshow(rgb)
axes[0, 0].set_title("Observation", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_seq_rgb)
p_seq = np.sum(mask_seq > 0)
n_seq = np.sum(mask_seq < 0)
d_seq = np.sum(mask_seq == 0)
axes[0, 1].set_title(f"Sequential Mask ($M_{{\\mathrm{{seq}}}}$)\nGreen P: {p_seq}px (0.9%), Red N: {n_seq}px (62.9%), Grey D: {d_seq}px (36.2%)", fontsize=9.5, fontweight="bold")
axes[0, 1].axis("off")

axes[0, 2].imshow(rgb)
axes[0, 2].imshow(mask_seq_rgb, alpha=0.5)
axes[0, 2].set_title("Overlay: Sequential Mask", fontsize=11, fontweight="bold")
axes[0, 2].axis("off")

# Bottom row: Requested New Mask (N on grid & wall, P everywhere else)
axes[1, 0].imshow(rgb)
axes[1, 0].set_title("Observation", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask_custom_rgb)
p_cust = np.sum(mask_custom > 0)
n_cust = np.sum(mask_custom < 0)
d_cust = np.sum(mask_custom == 0)
axes[1, 1].set_title(f"New Requested Mask ($M_{{\\mathrm{{wall\\_neg\\_floor\\_pos}}}}$)\nGreen P: {p_cust}px (37.1%), Red N: {n_cust}px (62.9%), Grey D: {d_cust}px (0%)", fontsize=9.5, fontweight="bold")
axes[1, 1].axis("off")

axes[1, 2].imshow(rgb)
axes[1, 2].imshow(mask_custom_rgb, alpha=0.5)
axes[1, 2].set_title("Overlay: New Mask (P everywhere inside)", fontsize=11, fontweight="bold")
axes[1, 2].axis("off")

fig.suptitle("Comparison of Supervision Masks for RQ5:\nSequential Subgoal Mask vs. New 'N on Walls/Grid, P Everywhere Else' Mask", fontsize=12, fontweight="bold", y=0.98)
fig.tight_layout()

out_path = Path(__file__).resolve().parent.parent / "output" / "figures" / "rq5_custom_mask_comparison.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved custom mask visualization to {out_path}")

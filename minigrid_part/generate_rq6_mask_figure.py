#!/usr/bin/env python3
"""Generate a compact 1-row figure: observation with shortcut | supervision mask | removed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigrid_part.shortcut_experiment_env import make_shortcut_vec_env

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(r"C:\RLProject\bachelors_thesis_ruslan\Thesis_Template_DE_EN\BA_MA_English\sections\figures")


def main():
    env = make_shortcut_vec_env(
        env_id="MiniGrid-DoorKey-6x6-v0",
        n_envs=1,
        seed=42,
        mode="correlated",
        correlation_probability=1.0,
        border_height=0,
        patch_size=8,
    )
    obs = env.reset()
    # obs shape: (1, C, H, W)
    obs_np = obs[0]  # (C, H, W)
    if obs_np.shape[0] == 6:
        rgb = obs_np[:3].transpose(1, 2, 0)  # (H, W, 3)
    else:
        rgb = obs_np.transpose(1, 2, 0)

    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0, 1)

    H, W = rgb.shape[:2]
    tile_h = 8

    # Build mask visualization
    mask_vis = np.zeros((H, W, 3))
    mask_vis[:tile_h, :] = [0.85, 0.2, 0.2]   # Red = N
    mask_vis[tile_h:, :] = [0.2, 0.7, 0.3]     # Green = P

    # Build grey-wall (removed) version
    grey_obs = rgb.copy()
    grey_obs[:tile_h, :] = [100/255, 100/255, 100/255]

    # Create compact figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), dpi=300)

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("(a) Observation with\nShortcut Cue", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask_vis, interpolation="nearest")
    axes[1].set_title("(b) Supervision Mask", fontsize=10, fontweight="bold")
    axes[1].text(W // 2, tile_h // 2, "$N = -1$", ha="center", va="center",
                 fontsize=11, color="white", fontweight="bold")
    axes[1].text(W // 2, tile_h + (H - tile_h) // 2, "$P = +1$", ha="center", va="center",
                 fontsize=11, color="white", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(grey_obs, interpolation="nearest")
    axes[2].set_title("(c) Removed\n(Grey Wall)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    png_path = OUT_DIR / "rq6_shortcut_mask_design.png"
    pdf_path = FIG_DIR / "figure_rq6_shortcut_design.pdf"
    fig.savefig(str(png_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate visualization of all supervision masks used in RQ3:
1. Strict Sequential Mask (M_seq)
2. Directional Local Action Mask (M_local)
3. Flipped Control Mask (M_flip = -M_seq)
Across key decision states: State 22 (pickup key), State 10 (toggle door), State 00 (move forward).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(r"C:\RLProject\bachelors_thesis_ruslan\Thesis_Template_DE_EN\BA_MA_English\sections\figures")

DATASET_PATH = ROOT / "experiments" / "rq3_explanation_control" / "evaluation" / "seed_2" / "shared_states_rgb6.npz"

ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pickup key",
    4: "drop key",
    5: "toggle door",
    6: "done",
}

# Unified colormap for masks matching Figure 4.6:
# -1 (Negative / Suppressed): Vibrant Red (#d03934)
#  0 (Neutral / Don't Care): Mid Gray (#6b7280)
# +1 (Positive / Desired Target): Vibrant Green (#44b24e)
MASK_CMAP = ListedColormap(["#d03934", "#6b7280", "#44b24e"])
MASK_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], MASK_CMAP.N)


def main():
    data = np.load(DATASET_PATH)
    observations = data["observations"]
    actions = data["actions"]
    seq_masks = data["sequential_masks"]
    loc_masks = data["local_masks"]

    # Key representative states
    # 1. State 22: pickup key
    # 2. State 10: toggle door
    # 3. State 00: move forward (navigation)
    sample_indices = [22, 10, 0]

    # Panel 1: Comprehensive Multi-State Mask Overview (3 rows x 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 9.8), constrained_layout=True, dpi=300)

    col_titles = [
        "Observation",
        "Strict Sequential Mask\n($M_{\\mathrm{seq}}$)",
        "Directional Local Mask\n($M_{\\mathrm{local}}$)",
        "Flipped Control Mask\n($M_{\\mathrm{flip}} = -M_{\\mathrm{seq}}$)",
    ]

    for row_idx, state_idx in enumerate(sample_indices):
        obs = observations[state_idx]
        act = int(actions[state_idx])
        act_name = ACTION_NAMES[act]

        rgb_raw = obs[:3].transpose(1, 2, 0)
        if rgb_raw.max() <= 1.0:
            rgb_raw = (rgb_raw * 255).astype(np.uint8)

        m_seq = seq_masks[state_idx]
        m_loc = loc_masks[state_idx]
        m_flip = -m_seq

        # Column 0: Observation
        axes[row_idx, 0].imshow(rgb_raw, interpolation="nearest")
        axes[row_idx, 0].set_ylabel(
            f"State {state_idx}\n({act_name})", fontsize=11, fontweight="bold"
        )
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Column 1: Sequential Mask
        axes[row_idx, 1].imshow(m_seq, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Column 2: Local Action Mask
        axes[row_idx, 2].imshow(m_loc, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

        # Column 3: Flipped Mask
        axes[row_idx, 3].imshow(m_flip, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
        axes[row_idx, 3].set_xticks([])
        axes[row_idx, 3].set_yticks([])

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Custom legend for mask regions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#44b24e", edgecolor="black", label="Positive Region $P$ (+1: Desired Target / Unpenalized)"),
        Patch(facecolor="#d03934", edgecolor="black", label="Negative Region $N$ (-1: Penalized Background)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=10.5,
        frameon=True,
    )

    plt.suptitle(
        "MiniGrid Explanation Supervision Masks (RQ3)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    out_dirs = [
        FIG_DIR,
        ROOT / "overleaf_source_v2_6chapters" / "Thesis_Template_DE_EN" / "BA_MA_English" / "sections" / "figures",
        ROOT / "overleaf_source_v2_6chapters" / "Thesis_Template_DE_EN" / "BA_MA_English" / "sections" / "images",
    ]

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "figure_rq3_supervision_masks.pdf", bbox_inches="tight")
        fig.savefig(d / "figure_rq3_supervision_masks.png", bbox_inches="tight")

    plt.close(fig)
    print("Saved overview figures.")

    # Also generate individual 1-row figures for State 10 (Toggle) and State 22 (Pickup)
    for state_idx, name in [(10, "state10_toggle"), (22, "state22_pickup")]:
        fig_single, axes_single = plt.subplots(
            1, 4, figsize=(13.5, 3.5), constrained_layout=True, dpi=300
        )
        obs = observations[state_idx]
        act = int(actions[state_idx])
        act_name = ACTION_NAMES[act]

        rgb_raw = obs[:3].transpose(1, 2, 0)
        if rgb_raw.max() <= 1.0:
            rgb_raw = (rgb_raw * 255).astype(np.uint8)

        m_seq = seq_masks[state_idx]
        m_loc = loc_masks[state_idx]
        m_flip = -m_seq

        axes_single[0].imshow(rgb_raw, interpolation="nearest")
        axes_single[0].set_title("Observation", fontsize=10.5, fontweight="bold")
        axes_single[0].set_ylabel(f"State {state_idx}\n({act_name})", fontsize=10.5, fontweight="bold")
        axes_single[0].set_xticks([])
        axes_single[0].set_yticks([])

        axes_single[1].imshow(m_seq, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
        axes_single[1].set_title("Sequential Mask ($M_{\\mathrm{seq}}$)", fontsize=10.5, fontweight="bold")
        axes_single[1].set_xticks([])
        axes_single[1].set_yticks([])

        axes_single[2].imshow(m_loc, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
        axes_single[2].set_title("Local Mask ($M_{\\mathrm{local}}$)", fontsize=10.5, fontweight="bold")
        axes_single[2].set_xticks([])
        axes_single[2].set_yticks([])

        axes_single[3].imshow(m_flip, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
        axes_single[3].set_title("Flipped Mask ($M_{\\mathrm{flip}}$)", fontsize=10.5, fontweight="bold")
        axes_single[3].set_xticks([])
        axes_single[3].set_yticks([])

        for d in out_dirs:
            fig_single.savefig(d / f"figure_rq3_masks_{name}.pdf", bbox_inches="tight")
            fig_single.savefig(d / f"figure_rq3_masks_{name}.png", bbox_inches="tight")

        plt.close(fig_single)
        print(f"Saved single mask for {name}")


if __name__ == "__main__":
    main()

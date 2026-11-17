#!/usr/bin/env python3
"""Generate separate standalone 1-row compact attribution figures for Toggle (State 10) and Pickup (State 22)."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "minigrid_part"))

from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.explanation_supervision import compute_actor_explanation
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(r"C:\RLProject\bachelors_thesis_ruslan\Thesis_Template_DE_EN\BA_MA_English\sections\figures")

DATASET_PATH = ROOT / "experiments" / "rq3_explanation_control" / "evaluation" / "seed_2" / "shared_states_rgb6.npz"

MODELS = {
    "Pure PPO": ROOT / "artifacts" / "rq3-pure-seed2-zc4w6xrd-v0" / "final_model.zip",
    "Sequential mask": ROOT / "artifacts" / "rq3-sequential-seed2-uz595n5s-v0" / "final_model.zip",
    "Local action mask": ROOT / "artifacts" / "rq3-local-seed2-17omr0kt-v0" / "final_model.zip",
    "Flipped mask": ROOT / "artifacts" / "rq3-flipped-seed2-7ks51cn1-v0" / "final_model.zip",
}

ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pickup key",
    4: "drop key",
    5: "toggle door",
    6: "done",
}


def load_policy(path: Path) -> BcosPPO:
    return BcosPPO.load(
        str(path),
        device="cpu",
        custom_objects={
            "policy_class": ConfigurableActorCriticPolicy,
            "lr_schedule": lambda _: 0.0,
        },
    )


def compute_explanation(model: BcosPPO, obs: np.ndarray, action: int) -> np.ndarray:
    obs_t = th.as_tensor(obs[None], device="cpu").float()
    act_t = th.tensor([action], device="cpu").long()
    with th.enable_grad():
        e = compute_actor_explanation(
            model.policy,
            obs_t,
            act_t,
            create_graph=False,
            subtract_action_mean=True,
        )
    return e[0].detach().cpu().numpy()  # (6, H, W)


def render_compact_row(
    loaded_models: dict[str, BcosPPO],
    obs: np.ndarray,
    act: int,
    branches: list[str],
    state_label: str,
    output_stem: str,
):
    rgb_raw = obs[:3].transpose(1, 2, 0)
    if rgb_raw.max() <= 1.0:
        rgb_raw = (rgb_raw * 255).astype(np.uint8)

    n_cols = len(branches) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2), dpi=300, constrained_layout=True)
    axes = np.atleast_1d(axes)

    # Column 0: Observation
    axes[0].imshow(rgb_raw, interpolation="nearest")
    axes[0].set_title(f"Observation\n({state_label})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for col, branch in enumerate(branches, start=1):
        model = loaded_models[branch]
        e = compute_explanation(model, obs, act)
        u = np.abs(e).sum(axis=0)  # (H, W)

        im = axes[col].imshow(u, cmap="magma", interpolation="nearest")
        axes[col].set_title(f"{branch}", fontsize=11, fontweight="bold")
        axes[col].axis("off")

        cbar = fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("absolute energy", fontsize=8)

    png_path = OUT_DIR / f"{output_stem}.png"
    pdf_path = FIG_DIR / f"{output_stem}.pdf"

    fig.savefig(str(png_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


def main():
    data = np.load(DATASET_PATH)
    observations = data["observations"]
    actions = data["actions"]

    loaded_models = {name: load_policy(path) for name, path in MODELS.items()}

    branches_4 = ["Pure PPO", "Sequential mask", "Local action mask", "Flipped mask"]
    branches_3 = ["Pure PPO", "Sequential mask", "Local action mask"]

    # 1. State 10 (Toggle door) - 4 branches
    render_compact_row(
        loaded_models,
        observations[10],
        int(actions[10]),
        branches_4,
        "State 10: toggle door",
        "figure_rq3_compact_state10_toggle_4branches",
    )

    # 2. State 22 (Pickup key) - 4 branches
    render_compact_row(
        loaded_models,
        observations[22],
        int(actions[22]),
        branches_4,
        "State 22: pickup key",
        "figure_rq3_compact_state22_pickup_4branches",
    )

    # 3. State 10 (Toggle door) - 3 branches
    render_compact_row(
        loaded_models,
        observations[10],
        int(actions[10]),
        branches_3,
        "State 10: toggle door",
        "figure_rq3_compact_state10_toggle_3branches",
    )

    # 4. State 22 (Pickup key) - 3 branches
    render_compact_row(
        loaded_models,
        observations[22],
        int(actions[22]),
        branches_3,
        "State 22: pickup key",
        "figure_rq3_compact_state22_pickup_3branches",
    )


if __name__ == "__main__":
    main()

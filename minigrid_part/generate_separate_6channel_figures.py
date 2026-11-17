#!/usr/bin/env python3
"""Generate separate standalone 6-channel decomposed figures for Toggle (State 10) and Pickup (State 22)."""
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

CHANNEL_LABELS = ["R", "G", "B", "255-R", "255-G", "255-B"]


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


def render_standalone_figure(
    loaded_models: dict[str, BcosPPO],
    obs: np.ndarray,
    act: int,
    branches: list[str],
    title: str,
    output_stem: str,
):
    rgb_raw = obs[:3].transpose(1, 2, 0)
    if rgb_raw.max() <= 1.0:
        rgb_raw = (rgb_raw * 255).astype(np.uint8)

    # Precompute explanations and find max magnitude for each branch
    explanations = {}
    for branch in branches:
        e = compute_explanation(loaded_models[branch], obs, act)
        explanations[branch] = np.abs(e)

    fig, axes = plt.subplots(
        len(branches), 7, figsize=(14.5, 2.35 * len(branches) + 0.6), constrained_layout=True, dpi=300
    )
    axes = np.atleast_2d(axes)

    for row, branch in enumerate(branches):
        # Column 0: Observation
        axes[row, 0].imshow(rgb_raw, interpolation="nearest")
        axes[row, 0].set_ylabel(branch, fontsize=10.5, fontweight="bold")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        vmax = max(float(explanations[branch].max()), 1e-12)

        im = None
        for c in range(6):
            channel_mag = explanations[branch][c]
            im = axes[row, c + 1].imshow(
                channel_mag, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest"
            )
            axes[row, c + 1].set_xticks([])
            axes[row, c + 1].set_yticks([])

        cbar = plt.colorbar(im, ax=axes[row, 1:], fraction=0.012, pad=0.015)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_label("abs. contrib.", fontsize=7.5)

    axes[0, 0].set_title("Observation", fontsize=10.5, fontweight="bold")
    for c, label in enumerate(CHANNEL_LABELS):
        axes[0, c + 1].set_title(label, fontsize=10.5, fontweight="bold")

    plt.suptitle(title, fontsize=11.5, fontweight="bold")

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

    branches_3 = ["Pure PPO", "Sequential mask", "Local action mask"]
    branches_4 = ["Pure PPO", "Sequential mask", "Local action mask", "Flipped mask"]

    # 1. State 10 (Toggle Action) - 3 branches
    render_standalone_figure(
        loaded_models,
        observations[10],
        int(actions[10]),
        branches_3,
        "RGB6 channel contributions, State 10 (action = toggle)",
        "figure_rq3_6channels_state10_toggle_3branches",
    )

    # 2. State 22 (Pickup Action) - 3 branches
    render_standalone_figure(
        loaded_models,
        observations[22],
        int(actions[22]),
        branches_3,
        "RGB6 channel contributions, State 22 (action = pickup)",
        "figure_rq3_6channels_state22_pickup_3branches",
    )

    # 3. State 10 (Toggle Action) - 4 branches (with Flipped)
    render_standalone_figure(
        loaded_models,
        observations[10],
        int(actions[10]),
        branches_4,
        "RGB6 channel contributions, State 10 (action = toggle)",
        "figure_rq3_6channels_state10_toggle_4branches",
    )

    # 4. State 22 (Pickup Action) - 4 branches (with Flipped)
    render_standalone_figure(
        loaded_models,
        observations[22],
        int(actions[22]),
        branches_4,
        "RGB6 channel contributions, State 22 (action = pickup)",
        "figure_rq3_6channels_state22_pickup_4branches",
    )


if __name__ == "__main__":
    main()

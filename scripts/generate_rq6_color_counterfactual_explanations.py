#!/usr/bin/env python3
"""Explain one RQ6 state under grey and every trained shortcut colour."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th


ROOT = Path(__file__).resolve().parents[1]
MINIGRID = ROOT / "minigrid_part"
for path in (ROOT, MINIGRID):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bcos_rl_generic.explanation_supervision import compute_actor_explanation
from evaluate_explanation_faithfulness import explanations
from evaluate_explanation_metrics import load_model
from evaluate_shortcut_experiment import actor_outputs
from shortcut_experiment_env import ACTION_COLORS, make_shortcut_vec_env


DEFAULT_MODEL = (
    ROOT
    / "rq6-unified-complete-results/runs/corr-1p00/seed-001/models/baseline.zip"
)
DEFAULT_OUTPUT = ROOT / "presentation_assets/rq6_corr100_baseline_ig"
DEFAULT_THESIS_OUTPUT = (
    ROOT
    / "thesis/source/Thesis_Template_DE_EN/BA_MA_English/sections/figures"
    / "figure_rq6_posthoc_counterfactual.pdf"
)
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
VARIANTS: list[tuple[str, int | None]] = [
    ("Grey", None),
    ("Red", 0),
    ("Green", 1),
    ("Blue", 2),
    ("Yellow", 3),
    ("Magenta", 5),
]


def magnitude(attribution: np.ndarray) -> np.ndarray:
    return np.abs(attribution).sum(axis=1)


def show_map(axis, values: np.ndarray) -> None:
    scale = max(float(np.percentile(values, 99)), 1e-12)
    axis.imshow(values, cmap="magma", vmin=0.0, vmax=scale)
    axis.set_xticks([])
    axis.set_yticks([])


def collect_right_state(model, seed: int) -> np.ndarray:
    """Return the normal-grey observation paired with a correct right-cue state."""
    correlated_env = make_shortcut_vec_env(
        env_id="MiniGrid-DoorKey-6x6-v0",
        n_envs=1,
        seed=seed,
        mode="correlated",
        correlation_probability=1.0,
    )
    removed_env = make_shortcut_vec_env(
        env_id="MiniGrid-DoorKey-6x6-v0",
        n_envs=1,
        seed=seed,
        mode="removed",
        correlation_probability=1.0,
    )
    correlated = correlated_env.reset()
    removed = removed_env.reset()
    try:
        for _ in range(20_000):
            action, _ = model.predict(correlated, deterministic=True)
            paired_grey = removed[0].copy()
            correlated, _, correlated_done, infos = correlated_env.step(action)
            removed, _, removed_done, _ = removed_env.step(action)
            oracle = int(infos[0]["shortcut_oracle_action"])
            selected = int(action[0])
            if oracle == 1 and selected == 1:
                return paired_grey
            if correlated_done[0] != removed_done[0]:
                raise RuntimeError("Paired environments diverged under identical actions")
            if correlated_done[0]:
                correlated = correlated_env.reset()
                removed = removed_env.reset()
    finally:
        correlated_env.close()
        removed_env.close()
    raise RuntimeError("Could not collect a paired right-action state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--source-data",
        type=Path,
        help="Optional cached output from generate_rq6_gray_counterfactual.py.",
    )
    parser.add_argument("--seed", type=int, default=51_001)
    parser.add_argument("--ig-steps", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thesis-output", type=Path, default=DEFAULT_THESIS_OUTPUT)
    args = parser.parse_args()

    model = load_model(args.model, "cpu")
    if args.source_data is None:
        grey = collect_right_state(model, args.seed)
    else:
        with np.load(args.source_data) as data:
            actions = np.asarray(data["actions"]).astype(np.int64)
            removed = np.asarray(data["removed_observations"])
        right_indices = np.flatnonzero(actions == 1)
        if not len(right_indices):
            raise RuntimeError("Source data contain no right-action state")
        grey = removed[int(right_indices[0])].copy()

    observations = []
    for _, cue_action in VARIANTS:
        observation = grey.copy()
        if cue_action is not None:
            color = np.asarray(ACTION_COLORS[cue_action], dtype=observation.dtype)
            observation[0, :8, :] = color[0]
            observation[1, :8, :] = color[1]
            observation[2, :8, :] = color[2]
        observations.append(observation)
    observations = np.stack(observations)

    probabilities = actor_outputs(model, observations, 32)[0]
    selected_actions = probabilities.argmax(axis=1).astype(np.int64)
    selected_probabilities = probabilities[np.arange(len(observations)), selected_actions]
    ig = explanations(
        model,
        observations,
        selected_actions,
        batch_size=len(observations),
        method="integrated_gradients",
        integrated_gradients_steps=args.ig_steps,
    )
    with th.enable_grad():
        intrinsic = compute_actor_explanation(
            model.policy,
            th.as_tensor(observations, device=model.device),
            th.as_tensor(selected_actions, device=model.device),
            create_graph=False,
            subtract_action_mean=True,
        ).detach().cpu().numpy()

    ig_magnitude = magnitude(ig)
    bcos_magnitude = magnitude(intrinsic)
    ig_ratios = ig_magnitude[:, :8, :].sum(axis=(1, 2)) / np.maximum(
        ig_magnitude.sum(axis=(1, 2)), 1e-12
    )
    bcos_ratios = bcos_magnitude[:, :8, :].sum(axis=(1, 2)) / np.maximum(
        bcos_magnitude.sum(axis=(1, 2)), 1e-12
    )

    fig, axes = plt.subplots(3, len(VARIANTS), figsize=(15.3, 7.2), dpi=240)
    for index, (label, _) in enumerate(VARIANTS):
        rgb = np.moveaxis(observations[index, :3], 0, -1).astype(np.float32) / 255.0
        axes[0, index].imshow(rgb)
        axes[0, index].set_title(
            f"{label}\n{ACTION_NAMES[selected_actions[index]]}: "
            f"{100 * selected_probabilities[index]:.1f}%",
            fontsize=10,
        )
        show_map(axes[1, index], ig_magnitude[index])
        show_map(axes[2, index], bcos_magnitude[index])
        axes[1, index].set_title(f"Top-row energy: {100 * ig_ratios[index]:.1f}%", fontsize=9)
        axes[2, index].set_title(f"Top-row energy: {100 * bcos_ratios[index]:.1f}%", fontsize=9)
        for axis in axes[:, index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("Observation", fontsize=11)
    axes[1, 0].set_ylabel("Integrated Gradients", fontsize=11)
    axes[2, 0].set_ylabel("Intrinsic B-cos", fontsize=11)
    fig.suptitle(
        "Same state, different shortcut colour · explanations follow selected action",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "color_counterfactual_ig_vs_bcos"
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Compact manuscript panel: removed cue, correct cue, and a contradictory
    # cue on the same underlying state.  Keeping three columns preserves label
    # readability at thesis text width.
    thesis_indices = [0, 2, 4]  # Grey, Green/right, Yellow/pickup.
    thesis_fig, thesis_axes = plt.subplots(3, 3, figsize=(7.1, 6.7), dpi=240)
    for column, index in enumerate(thesis_indices):
        label, _ = VARIANTS[index]
        rgb = np.moveaxis(observations[index, :3], 0, -1).astype(np.float32) / 255.0
        thesis_axes[0, column].imshow(rgb)
        thesis_axes[0, column].set_title(
            f"{label}: {ACTION_NAMES[selected_actions[index]]} "
            f"({100 * selected_probabilities[index]:.1f}%)",
            fontsize=10,
        )
        show_map(thesis_axes[1, column], ig_magnitude[index])
        show_map(thesis_axes[2, column], bcos_magnitude[index])
        thesis_axes[1, column].set_title(
            f"Top row: {100 * ig_ratios[index]:.1f}%", fontsize=9
        )
        thesis_axes[2, column].set_title(
            f"Top row: {100 * bcos_ratios[index]:.1f}%", fontsize=9
        )
        for axis in thesis_axes[:, column]:
            axis.set_xticks([])
            axis.set_yticks([])
    thesis_axes[0, 0].set_ylabel("Observation", fontsize=10)
    thesis_axes[1, 0].set_ylabel("Integrated Gradients", fontsize=10)
    thesis_axes[2, 0].set_ylabel("Intrinsic B-cos", fontsize=10)
    thesis_fig.tight_layout(pad=0.6)
    args.thesis_output.parent.mkdir(parents=True, exist_ok=True)
    thesis_fig.savefig(args.thesis_output, bbox_inches="tight")
    thesis_fig.savefig(args.thesis_output.with_suffix(".png"), bbox_inches="tight")
    plt.close(thesis_fig)

    np.savez_compressed(
        args.output_dir / "color_counterfactual_source_data.npz",
        observations=observations,
        selected_actions=selected_actions,
        probabilities=probabilities,
        integrated_gradients=ig,
        intrinsic_bcos=intrinsic,
        ig_shortcut_ratios=ig_ratios,
        bcos_shortcut_ratios=bcos_ratios,
    )
    for index, (label, _) in enumerate(VARIANTS):
        print(
            f"{label}: {ACTION_NAMES[selected_actions[index]]} "
            f"{selected_probabilities[index]:.4f}; "
            f"IG cue={ig_ratios[index]:.3f}; B-cos cue={bcos_ratios[index]:.3f}"
        )
    print(f"Saved: {output.with_suffix('.png').resolve()}")
    print(f"Saved thesis figure: {args.thesis_output.resolve()}")


if __name__ == "__main__":
    main()

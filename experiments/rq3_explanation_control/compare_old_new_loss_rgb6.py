#!/usr/bin/env python3
"""Compare RGB6 explanations for pure, old-loss, and final-loss PPO."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "minigrid_part"))

from evaluate_explanation_faithfulness import explanations  # noqa: E402
from evaluate_explanation_metrics import MINIGRID_ACTION_NAMES, load_model  # noqa: E402


SIGNED = LinearSegmentedColormap.from_list(
    "blue_white_red", ["#1857d2", "#ffffff", "#e31a1c"]
)
ABSOLUTE = LinearSegmentedColormap.from_list(
    "white_magenta", ["#ffffff", "#ff00ff"]
)
CHANNELS = ("R", "G", "B", "255-R", "255-G", "255-B")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pure-model", type=Path, required=True)
    result.add_argument("--old-model", type=Path)
    result.add_argument("--new-model", type=Path)
    result.add_argument("--local-model", type=Path)
    result.add_argument("--pure-only", action="store_true")
    result.add_argument("--single-label", default="Pure PPO")
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--state-index", type=int, default=27)
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--display", choices=("signed", "absolute"), default="signed"
    )
    return result


def render(
    observation: np.ndarray,
    action_name: str,
    index: int,
    maps: tuple[tuple[str, np.ndarray], ...],
    output: Path,
    display: str,
) -> None:
    figure, axes = plt.subplots(
        len(maps), 8, figsize=(17.2, 2.4 * len(maps) + 0.9), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    rgb = np.transpose(observation[:3], (1, 2, 0)).astype(np.float32) / 255.0

    for row, (label, values) in enumerate(maps):
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(label, fontweight="bold", fontsize=10)

        if display == "absolute":
            shown = np.abs(values)
            summed = shown.sum(axis=0)
            # Normalize each observation/model independently. The six channels
            # share one scale; their sum has its own exact scale so it remains legible.
            channel_scale = max(float(shown.max()), 1e-12)
            sum_scale = max(float(summed.max()), 1e-12)
            channel_image = None
            for channel in range(6):
                channel_image = axes[row, channel + 1].imshow(
                    shown[channel], cmap=ABSOLUTE, vmin=0.0, vmax=channel_scale
                )
            sum_image = axes[row, 7].imshow(
                summed, cmap=ABSOLUTE, vmin=0.0, vmax=sum_scale
            )
            channel_bar = figure.colorbar(
                channel_image,
                ax=axes[row, 1:7],
                location="bottom",
                fraction=0.035,
                pad=0.02,
                aspect=45,
            )
            channel_bar.ax.tick_params(labelsize=7)
            channel_bar.set_label("absolute contribution (shared RGB6 scale)", fontsize=8)
            sum_bar = figure.colorbar(
                sum_image,
                ax=axes[row, 7],
                location="bottom",
                fraction=0.035,
                pad=0.02,
                aspect=8,
            )
            sum_bar.ax.tick_params(labelsize=7)
            sum_bar.set_label("absolute energy", fontsize=8)
        else:
            summed = values.sum(axis=0)
            # Normalize each model independently, sharing one scale across RGB6 and sum.
            scale = max(float(np.abs(values).max()), float(np.abs(summed).max()), 1e-12)
            image = None
            for channel in range(6):
                image = axes[row, channel + 1].imshow(
                    values[channel], cmap=SIGNED, vmin=-scale, vmax=scale
                )
            image = axes[row, 7].imshow(
                summed, cmap=SIGNED, vmin=-scale, vmax=scale
            )
            colorbar = figure.colorbar(
                image, ax=axes[row, 1:], location="right", fraction=0.012, pad=0.008
            )
            colorbar.ax.tick_params(labelsize=7)
            colorbar.set_label("signed contribution", fontsize=8)

        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_title("Observation")
    for channel, name in enumerate(CHANNELS):
        axes[0, channel + 1].set_title(name)
    if display == "absolute":
        axes[0, 7].set_title(r"Energy: $\sum_c |E_c|$")
        title = (
            f"Absolute RGB6 explanations, state {index}, action={action_name}\n"
            "white is zero; magenta is higher attribution magnitude"
        )
    else:
        axes[0, 7].set_title(r"Sum: $\sum_c E_c$")
        title = (
            f"Signed RGB6 explanations, state {index}, action={action_name}\n"
            "red supports the action; blue suppresses it"
        )
    figure.suptitle(title, fontsize=11)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parser().parse_args()
    with np.load(args.dataset) as data:
        observations = np.asarray(data["observations"])
        actions = np.asarray(data["actions"], dtype=np.int64).reshape(-1)

    index = args.state_index
    if not 0 <= index < len(observations):
        raise IndexError(f"State index {index} outside [0, {len(observations)})")
    observation = observations[index : index + 1]
    action = actions[index : index + 1]

    model_specs = [(args.single_label, args.pure_model)]
    if not args.pure_only:
        if args.old_model is None or args.new_model is None:
            raise ValueError("--old-model and --new-model are required without --pure-only")
        model_specs.extend(
            (("Old loss", args.old_model), ("Final loss", args.new_model))
        )
        if args.local_model is not None:
            model_specs.append(("Local action mask", args.local_model))

    maps = []
    for label, path in model_specs:
        model = load_model(path, args.device)
        maps.append((label, explanations(model, observation, action, 1)[0]))

    render(
        observation[0],
        MINIGRID_ACTION_NAMES[int(action[0])],
        index,
        tuple(maps),
        args.output,
        args.display,
    )
    print(f"State {index}; action={MINIGRID_ACTION_NAMES[int(action[0])]}")


if __name__ == "__main__":
    main()

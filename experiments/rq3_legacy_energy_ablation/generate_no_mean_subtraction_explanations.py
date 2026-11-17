#!/usr/bin/env python3
"""Generate exact seed-0 RQ3 explanations for raw selected action logits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch as th


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "minigrid_part"))

from bcos_rl_generic.explanation_supervision import compute_actor_explanation
from evaluate_explanation_metrics import load_model
from evaluate_local_mask_experiment import (
    collect_shared_dataset,
    representative_examples,
)


ENTITY = "ruslandontsov2005-saarland-informatics-campus"
PROJECT = "rq3-minigrid-ppo-explanation-control-final"
ARTIFACTS = {
    "pure": "rq3-pure-seed0-2fttulzu:v0",
    "sequential": "rq3-sequential-seed0-khgcp4ro:v0",
    "local": "rq3-local-seed0-mrucq5dx:v0",
    "flipped": "rq3-flipped-seed0-ur09jisg:v0",
}
LABELS = {
    "pure": "Pure PPO",
    "sequential": "Sequential",
    "local": "Local action",
    "flipped": "Flipped",
}


def model_paths(model_dir: Path) -> dict[str, Path]:
    import wandb

    api = wandb.Api()
    paths = {}
    for branch, artifact_name in ARTIFACTS.items():
        branch_dir = model_dir / branch
        candidates = sorted(branch_dir.rglob("*.zip")) if branch_dir.exists() else []
        if not candidates:
            artifact = api.artifact(f"{ENTITY}/{PROJECT}/{artifact_name}")
            artifact.download(root=branch_dir)
            candidates = sorted(branch_dir.rglob("*.zip"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one ZIP for {branch}, found {candidates}")
        paths[branch] = candidates[0]
    return paths


def actor_explanation(
    model,
    observation: np.ndarray,
    action: int,
    *,
    subtract_action_mean: bool,
) -> np.ndarray:
    obs = th.as_tensor(observation[None], device=model.device).float()
    actions = th.as_tensor([action], device=model.device).long()
    with th.enable_grad():
        attribution = compute_actor_explanation(
            model.policy,
            obs,
            actions,
            create_graph=False,
            subtract_action_mean=subtract_action_mean,
        )
    return attribution[0].detach().cpu().numpy().sum(axis=0)


def save_side_by_side(
    observation: np.ndarray,
    maps: dict[str, np.ndarray],
    stem: str,
    title: str,
    attribution_title: str,
    output_dir: Path,
) -> None:
    rgb = np.moveaxis(observation[:3], 0, -1).astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    absolute_maps = {name: np.abs(value) for name, value in maps.items()}
    scale = max(max(float(np.max(value)), 1e-8) for value in absolute_maps.values())

    fig, axes = plt.subplots(4, 2, figsize=(4.8, 7.4))
    for row, (branch, attribution) in enumerate(absolute_maps.items()):
        axes[row, 0].imshow(rgb)
        axes[row, 0].text(
            -0.08,
            0.5,
            LABELS[branch],
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )
        axes[row, 1].imshow(attribution, cmap="plasma", vmin=0.0, vmax=scale)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    axes[0, 0].set_title("Observation")
    axes[0, 1].set_title(attribution_title)
    fig.suptitle(title, y=0.995, fontsize=10)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / f"{stem}_absolute_side_by_side.png", dpi=300)
    fig.savefig(output_dir / f"{stem}_absolute_side_by_side.pdf")
    plt.close(fig)


def farthest_visible_key_index(dataset: dict[str, np.ndarray]) -> int:
    best_distance = -1.0
    best_index = -1
    for index, observation in enumerate(dataset["observations"]):
        rgb = np.moveaxis(observation[:3], 0, -1)
        agent = (rgb[..., 0] > 220) & (rgb[..., 1] < 40) & (rgb[..., 2] < 40)
        key = (rgb[..., 0] > 240) & (rgb[..., 1] > 240) & (rgb[..., 2] < 20)
        if not agent.any() or not key.any():
            continue
        distance = float(
            np.linalg.norm(np.argwhere(agent).mean(axis=0) - np.argwhere(key).mean(axis=0))
        )
        if distance > best_distance:
            best_distance = distance
            best_index = index
    if best_index < 0:
        raise RuntimeError("No state with a visible key was found")
    return best_index


def requested_examples(dataset: dict[str, np.ndarray]) -> dict[str, tuple[int, int]]:
    toggle_indices = np.flatnonzero(dataset["actions"] == 5)
    if not len(toggle_indices):
        raise RuntimeError("No toggle-at-door state was found")
    toggle_index = int(toggle_indices[0])
    far_index = farthest_visible_key_index(dataset)
    return {
        "toggle_door_with_key": (toggle_index, 5),
        "far_from_key": (far_index, int(dataset["actions"][far_index])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--subtract-action-mean", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir is None:
        mode_dir = "mean_subtracted" if args.subtract_action_mean else "no_mean_subtraction"
        args.output_dir = Path(__file__).resolve().parent / "thesis_outputs" / mode_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = model_paths(args.output_dir / "models")
    models = {name: load_model(path, args.device) for name, path in paths.items()}
    dataset_args = SimpleNamespace(
        dataset_seed=20_000,
        dataset_episodes=30,
        dataset_samples=256,
        dataset_stochastic=False,
        env_id="MiniGrid-DoorKey-8x8-v0",
        open_door_reward=0.1,
        pick_key_reward=0.05,
        movement_radius=1,
    )
    dataset = collect_shared_dataset(models["pure"], dataset_args)
    np.savez_compressed(args.output_dir / "seed0_shared_states_rgb6.npz", **dataset)

    metadata = {}
    examples = (
        requested_examples(dataset)
        if args.subtract_action_mean
        else {**representative_examples(dataset), **requested_examples(dataset)}
    )
    for category, (index, action) in examples.items():
        observation = dataset["observations"][index]
        maps = {
            name: actor_explanation(
                model,
                observation,
                action,
                subtract_action_mean=args.subtract_action_mean,
            )
            for name, model in models.items()
        }
        prefix = "rq3_relative" if args.subtract_action_mean else "rq3_no_mean"
        stem = f"{prefix}_{category}"
        title = {
            "toggle_door_with_key": "Toggle at the door while carrying the key",
            "far_from_key": "Agent far from the visible key",
        }.get(category, category.replace("_", " ").title())
        attribution_title = (
            "Absolute relative-logit attribution"
            if args.subtract_action_mean
            else "Absolute raw-logit attribution"
        )
        save_side_by_side(
            observation,
            maps,
            stem,
            title,
            attribution_title,
            args.output_dir,
        )
        np.savez_compressed(
            args.output_dir / f"{stem}_raw_attributions.npz",
            observation=observation,
            action=np.asarray(action),
            **maps,
        )
        metadata[category] = {"dataset_index": index, "action": action}
    target_description = (
        "relative selected-action logit z_a - mean_j(z_j)"
        if args.subtract_action_mean
        else "raw selected-action logit z_a"
    )
    (args.output_dir / "README.txt").write_text(
        f"Exact B-cos gradient x input explanations of the {target_description}. Figures "
        "show absolute attribution magnitude using a shared purple-yellow scale; "
        "the NPZ files retain the original signed tensors.\n"
        f"Representative states: {metadata}\n",
        encoding="utf-8",
    )
    mode = "mean-subtracted" if args.subtract_action_mean else "raw-logit"
    print(f"Saved {mode} RQ3 explanations to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

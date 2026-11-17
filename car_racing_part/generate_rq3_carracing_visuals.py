#!/usr/bin/env python3
"""Generate matched five-seed visuals for the CarRacing RQ3 support sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcos_rl_generic.explanation_supervision import compute_actor_explanation
from car_racing_part.explanation_masks import (
    CarRacingGrassColorWrapper,
    CarRacingMaskConfig,
    build_carracing_explanation_mask,
)
from car_racing_part.train_carracing_ppo import ACTION_NAMES, build_env
from car_racing_part.visualize_carracing import load_model, record_episode


TARGET_ACTIONS = (3, 2, 1)  # gas, steer_left, steer_right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-seed", type=int, default=1)
    parser.add_argument("--environment-seed", type=int, default=27_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-states", type=Path)
    parser.add_argument("--file-prefix", default="rq3_carracing")
    parser.add_argument("--condition-label", default="explanation-guided")
    parser.add_argument("--comparison-bundle", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(bundle_root: Path) -> dict:
    return json.loads((bundle_root / "model_manifest.json").read_text(encoding="utf-8"))


def model_paths(bundle_root: Path) -> dict[int, Path]:
    manifest = load_manifest(bundle_root)
    return {
        int(row["seed"]): bundle_root / row["relative_path"]
        for row in manifest["models"]
        if row["model_kind"] == "final"
    }


def verify_models(bundle_root: Path) -> dict[int, dict[str, str]]:
    manifest = load_manifest(bundle_root)
    expected = {
        int(row["seed"]): row["sha256"]
        for row in manifest["models"]
        if row["model_kind"] == "final"
    }
    verified = {}
    paths = model_paths(bundle_root)
    if sorted(paths) != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Expected seeds 0--4, found {sorted(paths)}")
    for seed, path in sorted(paths.items()):
        actual = sha256(path)
        if actual != expected[seed]:
            raise RuntimeError(f"SHA-256 mismatch for seed {seed}: {actual} != {expected[seed]}")
        verified[seed] = {"path": str(path.resolve()), "sha256": actual}
    return verified


def select_reference_frames(model_path_value: Path, base_seed: int, device: str):
    for offset in range(20):
        env_seed = base_seed + offset
        env = build_env(
            env_id="CarRacing-v3",
            n_envs=1,
            seed=env_seed,
            use_subprocesses=False,
            wrapper_class=partial(CarRacingGrassColorWrapper, color_name="normal"),
        )
        try:
            model = load_model(model_path_value, env, device)
            frames = record_episode(model, env, deterministic=True, max_steps=1_000)
        finally:
            env.close()
        chosen = []
        for action in TARGET_ACTIONS:
            candidates = [
                (index, frame)
                for index, frame in enumerate(frames)
                if frame.action == action and index >= 40
            ]
            if not candidates:
                break
            index, frame = max(
                candidates,
                key=lambda item: float(item[1].action_probabilities[action]),
            )
            chosen.append((index, frame))
        if len(chosen) == len(TARGET_ACTIONS):
            return env_seed, frames[-1].cumulative_reward, chosen
    raise RuntimeError("Could not find a deterministic episode containing gas, steer_left, and steer_right")


def attribution_energy(model, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    obs_tensor = th.as_tensor(observations, device=model.device)
    action_tensor = th.as_tensor(actions, device=model.device, dtype=th.long)
    with th.enable_grad():
        explanation = compute_actor_explanation(
            model.policy,
            obs_tensor,
            action_tensor,
            create_graph=False,
        )
    return explanation.detach().abs().sum(dim=1).cpu().numpy()


def add_energy_overlay(axis, observation: np.ndarray, energy: np.ndarray) -> None:
    rgb = np.moveaxis(observation, 0, -1)
    scale = max(float(np.percentile(energy, 99.0)), 1e-12)
    normalized = np.clip(energy / scale, 0.0, 1.0)
    axis.imshow(rgb)
    axis.imshow(
        normalized,
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        alpha=np.clip(normalized * 0.82, 0.0, 0.82),
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def save_five_seed_figure(
    bundle_root: Path,
    observations: np.ndarray,
    actions: np.ndarray,
    output_dir: Path,
    device: str,
    file_prefix: str,
    condition_label: str,
) -> dict[int, list[float]]:
    energies = {}
    env = build_env(
        env_id="CarRacing-v3",
        n_envs=1,
        seed=99_999,
        use_subprocesses=False,
    )
    try:
        for seed, path in sorted(model_paths(bundle_root).items()):
            model = load_model(path, env, device)
            values = attribution_energy(model, observations, actions)
            energies[seed] = values
            del model
    finally:
        env.close()

    figure, axes = plt.subplots(3, 6, figsize=(13.2, 7.0))
    for row, action in enumerate(actions):
        axes[row, 0].imshow(np.moveaxis(observations[row], 0, -1))
        axes[row, 0].set_ylabel(ACTION_NAMES[int(action)], fontsize=10)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for column, seed in enumerate(sorted(energies), start=1):
            add_energy_overlay(axes[row, column], observations[row], energies[seed][row])
    for axis, title in zip(axes[0], ("Observation", "Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4")):
        axis.set_title(title, fontsize=10)
    figure.suptitle(f"CarRacing: intrinsic B-cos attribution across five {condition_label} seeds", fontsize=12)
    figure.tight_layout()
    base = output_dir / f"{file_prefix}_five_seed_attributions"
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return {
        seed: [float(np.sum(row)) for row in values]
        for seed, values in energies.items()
    }


def save_mask_figure(
    bundle_root: Path,
    observations: np.ndarray,
    actions: np.ndarray,
    frame_indices: list[int],
    reference_seed: int,
    output_dir: Path,
    device: str,
    file_prefix: str,
    condition_label: str,
) -> None:
    env = build_env(
        env_id="CarRacing-v3",
        n_envs=1,
        seed=99_998,
        use_subprocesses=False,
    )
    try:
        model = load_model(model_paths(bundle_root)[reference_seed], env, device)
        energies = attribution_energy(model, observations, actions)
    finally:
        env.close()
    config = CarRacingMaskConfig(profile="boundary-dashboard", warmup_frames=0, boundary_radius=3)
    figure, axes = plt.subplots(3, 3, figsize=(7.7, 7.1))
    for row, (observation, action, frame_index) in enumerate(zip(observations, actions, frame_indices)):
        rgb = np.moveaxis(observation, 0, -1)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(ACTION_NAMES[int(action)], fontsize=10)
        mask = build_carracing_explanation_mask(observation, episode_step=frame_index, config=config)
        overlay = rgb.astype(np.float32) / 255.0
        positive = mask > 0
        negative = mask < 0
        overlay[positive] = 0.35 * overlay[positive] + 0.65 * np.array([0.0, 1.0, 0.0])
        overlay[negative] = 0.35 * overlay[negative] + 0.65 * np.array([1.0, 0.0, 0.0])
        axes[row, 1].imshow(overlay)
        add_energy_overlay(axes[row, 2], observation, energies[row])
        for axis in axes[row, :2]:
            axis.set_xticks([])
            axis.set_yticks([])
    mask_title = "RQ3 target mask (not used)" if "no-explanation" in condition_label else "Training mask"
    for axis, title in zip(axes[0], ("Observation", mask_title, f"Seed {reference_seed} attribution")):
        axis.set_title(title, fontsize=10)
    figure.suptitle(f"CarRacing mask and {condition_label} attribution", fontsize=12)
    figure.tight_layout()
    base = output_dir / f"{file_prefix}_mask_and_attribution"
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def save_paired_figure(
    baseline_root: Path,
    guided_root: Path,
    observations: np.ndarray,
    actions: np.ndarray,
    reference_seed: int,
    output_dir: Path,
    device: str,
) -> None:
    env = build_env(
        env_id="CarRacing-v3",
        n_envs=1,
        seed=99_997,
        use_subprocesses=False,
    )
    try:
        baseline = load_model(model_paths(baseline_root)[reference_seed], env, device)
        baseline_energy = attribution_energy(baseline, observations, actions)
        del baseline
        guided = load_model(model_paths(guided_root)[reference_seed], env, device)
        guided_energy = attribution_energy(guided, observations, actions)
    finally:
        env.close()
    mask_config = CarRacingMaskConfig(profile="boundary-dashboard", warmup_frames=0, boundary_radius=3)
    figure, axes = plt.subplots(3, 4, figsize=(10.1, 7.1))
    for row, (observation, action) in enumerate(zip(observations, actions)):
        rgb = np.moveaxis(observation, 0, -1)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(ACTION_NAMES[int(action)], fontsize=10)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        mask = build_carracing_explanation_mask(
            observation,
            episode_step=0,
            config=mask_config,
        )
        overlay = rgb.astype(np.float32) / 255.0
        positive = mask > 0
        negative = mask < 0
        overlay[positive] = 0.35 * overlay[positive] + 0.65 * np.array([0.0, 1.0, 0.0])
        overlay[negative] = 0.35 * overlay[negative] + 0.65 * np.array([1.0, 0.0, 0.0])
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        add_energy_overlay(axes[row, 2], observation, baseline_energy[row])
        add_energy_overlay(axes[row, 3], observation, guided_energy[row])
    for axis, title in zip(
        axes[0],
        ("Observation", "Target mask", "No explanation loss", "Explanation-guided"),
    ):
        axis.set_title(title, fontsize=10)
    figure.suptitle(f"CarRacing seed {reference_seed}: matched intrinsic B-cos attributions", fontsize=12)
    figure.tight_layout()
    base = output_dir / "rq3_carracing_paired_seed1_comparison"
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def save_reward_figure(
    bundle_root: Path,
    output_dir: Path,
    file_prefix: str,
    condition_label: str,
) -> None:
    manifest = load_manifest(bundle_root)
    rows = sorted(manifest["runs"], key=lambda row: int(row["seed"]))
    seeds = np.asarray([int(row["seed"]) for row in rows])
    means = np.asarray([float(row["final_eval_mean_reward"]) for row in rows])
    stds = np.asarray([float(row["final_eval_std_reward"]) for row in rows])
    aggregate = float(np.mean(means))
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    axis.bar(seeds, means, yerr=stds, capsize=4, color="#9b1b30", alpha=0.86)
    axis.axhline(aggregate, color="#252525", linestyle="--", linewidth=1.2, label=f"Five-seed mean: {aggregate:.1f}")
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Final episodic reward")
    axis.set_xticks(seeds)
    axis.set_title(
        f"CarRacing {condition_label} final stochastic evaluation\n"
        "Bars: 30-episode mean; whiskers: within-seed SD"
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    base = output_dir / f"{file_prefix}_final_rewards"
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verified = verify_models(args.bundle_root)
    if args.reference_states:
        reference = np.load(args.reference_states)
        observations = reference["observations"]
        actions = reference["actions"]
        frame_indices = reference["frame_indices"].astype(int).tolist()
        env_seed = int(reference["environment_seed"])
        episode_return = None
    else:
        env_seed, episode_return, chosen = select_reference_frames(
            model_paths(args.bundle_root)[args.reference_seed],
            args.environment_seed,
            args.device,
        )
        frame_indices = [index for index, _frame in chosen]
        observations = np.stack([frame.observation for _index, frame in chosen])
        actions = np.asarray([frame.action for _index, frame in chosen], dtype=np.int64)
    energy_totals = save_five_seed_figure(
        args.bundle_root, observations, actions, args.output_dir, args.device,
        args.file_prefix, args.condition_label,
    )
    save_mask_figure(
        args.bundle_root,
        observations,
        actions,
        frame_indices,
        args.reference_seed,
        args.output_dir,
        args.device,
        args.file_prefix,
        args.condition_label,
    )
    save_reward_figure(args.bundle_root, args.output_dir, args.file_prefix, args.condition_label)
    if args.comparison_bundle:
        save_paired_figure(
            args.bundle_root,
            args.comparison_bundle,
            observations,
            actions,
            args.reference_seed,
            args.output_dir,
            args.device,
        )
    np.savez_compressed(
        args.output_dir / "rq3_carracing_reference_states.npz",
        observations=observations,
        actions=actions,
        frame_indices=np.asarray(frame_indices),
        environment_seed=np.asarray(env_seed),
    )
    metadata = {
        "reference_model_seed": args.reference_seed,
        "environment_seed": env_seed,
        "deterministic_episode_return": episode_return,
        "frame_indices": frame_indices,
        "actions": [ACTION_NAMES[int(action)] for action in actions],
        "attribution_display": "sum_c abs(E_c), independently scaled to each map's 99th percentile; observation preserved with attribution-dependent red opacity",
        "condition_label": args.condition_label,
        "reference_states_source": str(args.reference_states.resolve()) if args.reference_states else "generated by this run",
        "models": verified,
        "energy_totals_for_displayed_states": energy_totals,
    }
    (args.output_dir / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

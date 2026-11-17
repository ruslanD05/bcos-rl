#!/usr/bin/env python3
"""Evaluate matched pure, sequential, local-action, and flipped-mask PPO models."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from env_wrappers import make_wrapper
from evaluate_explanation_faithfulness import (
    actor_outputs,
    empty_grid_substrate,
    evaluate_deletion_insertion_sensitivity,
    evaluate_model as evaluate_faithfulness,
    explanations,
    parse_fractions,
    region_metrics,
    save_plot as save_faithfulness_plot,
)
from evaluate_explanation_metrics import (
    MINIGRID_ACTION_NAMES,
    evaluate_actor_distribution,
    evaluate_channelwise_absolute_explanations,
    load_model,
)
from evaluate_wall_ood import parse_models


INTERACTION_ACTIONS = (3, 4, 5)
MOVEMENT_ACTIONS = (0, 1, 2)


def build_env(
    args: argparse.Namespace,
    *,
    seed: int,
    mask_kind: str = "none",
):
    if mask_kind not in {"none", "sequential", "local"}:
        raise ValueError(mask_kind)
    wrapper = make_wrapper(
        use_rgb_img_obs=True,
        invert_rgb_channels=True,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_all_negative_mask=mask_kind == "sequential",
        use_local_directional_mask=mask_kind == "local",
        local_action_movement_radius=args.movement_radius,
        local_action_emit_all_masks=mask_kind == "local",
    )
    env = make_vec_env(
        args.env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"highlight": False},
        wrapper_class=wrapper,
    )
    return VecTransposeImage(env)


def action_name(action: int) -> str:
    if 0 <= action < len(MINIGRID_ACTION_NAMES):
        return MINIGRID_ACTION_NAMES[action]
    return f"action_{action}"


def evaluate_policy_stochastic(
    model,
    args: argparse.Namespace,
) -> dict[str, Any]:
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    env = build_env(args, seed=args.seed, mask_kind="none")
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    action_counts: Counter[int] = Counter()
    episode_reward = 0.0
    episode_length = 0
    obs = env.reset()
    try:
        while len(rewards) < args.episodes:
            action, _ = model.predict(obs, deterministic=False)
            action_idx = int(np.asarray(action).reshape(-1)[0])
            action_counts[action_idx] += 1
            obs, reward, done, infos = env.step(action)
            episode_reward += float(reward[0])
            episode_length += 1
            if done[0]:
                rewards.append(episode_reward)
                lengths.append(episode_length)
                successes.append(float(infos[0].get("is_success", 0.0)))
                episode_reward = 0.0
                episode_length = 0
    finally:
        env.close()

    total_actions = sum(action_counts.values())
    action_rates = {
        action_name(index): action_counts[index] / max(total_actions, 1)
        for index in range(int(model.action_space.n))
    }
    return {
        "episodes": args.episodes,
        "action_mode": "stochastic",
        "seed": args.seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "std_episode_length": float(np.std(lengths)),
        "success_rate": float(np.mean(successes)),
        "action_rates": action_rates,
    }


def collect_shared_dataset(
    reference_model,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    np.random.seed(args.dataset_seed)
    th.manual_seed(args.dataset_seed)
    sequential_env = build_env(args, seed=args.dataset_seed, mask_kind="sequential")
    local_env = build_env(args, seed=args.dataset_seed, mask_kind="local")
    sequential_obs = sequential_env.reset()
    local_obs = local_env.reset()
    if not np.array_equal(sequential_obs, local_obs):
        raise RuntimeError("Synchronized mask environments disagree at reset")

    rng = np.random.default_rng(args.dataset_seed)
    reservoir: list[tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]] = []
    transitions_seen = 0
    completed_episodes = 0
    try:
        while completed_episodes < args.dataset_episodes:
            action, _ = reference_model.predict(
                sequential_obs,
                deterministic=not args.dataset_stochastic,
            )
            next_sequential, seq_reward, seq_done, seq_infos = sequential_env.step(action)
            next_local, local_reward, local_done, local_infos = local_env.step(action)
            if not np.array_equal(next_sequential, next_local):
                raise RuntimeError("Synchronized mask environments produced different observations")
            if not np.array_equal(seq_done, local_done) or not np.allclose(
                seq_reward, local_reward
            ):
                raise RuntimeError("Synchronized mask environments diverged")

            sequential_mask = seq_infos[0].get("attn_mask")
            local_mask = local_infos[0].get("attn_mask")
            all_local_masks = local_infos[0].get("all_action_attn_masks")
            if sequential_mask is None or local_mask is None or all_local_masks is None:
                raise RuntimeError("Evaluation mask wrapper omitted required masks")
            item = (
                sequential_obs[0].copy(),
                int(np.asarray(action).reshape(-1)[0]),
                np.asarray(sequential_mask, dtype=np.float32).copy(),
                np.asarray(local_mask, dtype=np.float32).copy(),
                np.asarray(all_local_masks, dtype=np.float32).copy(),
            )
            transitions_seen += 1
            if len(reservoir) < args.dataset_samples:
                reservoir.append(item)
            else:
                replacement = int(rng.integers(0, transitions_seen))
                if replacement < args.dataset_samples:
                    reservoir[replacement] = item

            if seq_done[0]:
                completed_episodes += 1
            sequential_obs = next_sequential
            local_obs = next_local
    finally:
        sequential_env.close()
        local_env.close()

    if len(reservoir) < args.dataset_samples:
        raise RuntimeError(
            f"Collected only {len(reservoir)} transitions; requested {args.dataset_samples}"
        )
    return {
        "observations": np.stack([item[0] for item in reservoir]),
        "actions": np.asarray([item[1] for item in reservoir], dtype=np.int64),
        "sequential_masks": np.stack([item[2] for item in reservoir]),
        "local_masks": np.stack([item[3] for item in reservoir]),
        "local_masks_all_actions": np.stack([item[4] for item in reservoir]),
    }


def load_or_collect_dataset(
    reference_model,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if args.dataset_in is not None:
        with np.load(args.dataset_in) as data:
            required = {
                "observations",
                "actions",
                "sequential_masks",
                "local_masks",
                "local_masks_all_actions",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
            return {name: np.asarray(data[name]) for name in required}

    dataset = collect_shared_dataset(reference_model, args)
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dataset_out, **dataset)
    return dataset


def pairwise_policy_metrics(
    outputs_by_model: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    eps = 1e-8
    for left_name, left in outputs_by_model.items():
        result[left_name] = {}
        p = np.clip(left["probabilities"], eps, 1.0)
        for right_name, right in outputs_by_model.items():
            q = np.clip(right["probabilities"], eps, 1.0)
            result[left_name][right_name] = {
                "mean_kl_left_to_right": float(
                    np.sum(p * (np.log(p) - np.log(q)), axis=1).mean()
                ),
                "argmax_disagreement_rate": float(
                    np.mean(left["argmax_actions"] != right["argmax_actions"])
                ),
                "mean_probability_l1": float(np.abs(p - q).sum(axis=1).mean()),
            }
    return result


def continuation_metadata(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path.parent / "manifest.json"
    if not manifest_path.exists():
        return {"manifest": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest": str(manifest_path.resolve()),
        "branch": manifest.get("branch"),
        "seed": manifest.get("seed"),
        "source_sha256": manifest.get("source_sha256"),
        "training_wall_seconds": manifest.get("training_wall_seconds"),
        "environment_steps_per_second": manifest.get(
            "environment_steps_per_second"
        ),
        "parameters": manifest.get("parameters"),
    }


def pnd_absolute_energy(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    masks_by_name: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, dict[str, float]]:
    attribution = explanations(model, observations, actions, batch_size)
    magnitude = np.abs(attribution).sum(axis=1)
    total = magnitude.sum(axis=(1, 2)) + 1e-8
    result: dict[str, dict[str, float]] = {}
    for name, mask in masks_by_name.items():
        p = (magnitude * (mask > 0)).sum(axis=(1, 2))
        n = (magnitude * (mask < 0)).sum(axis=(1, 2))
        d = (magnitude * (mask == 0)).sum(axis=(1, 2))
        result[name] = {
            "positive_absolute_energy": float(p.mean()),
            "negative_absolute_energy": float(n.mean()),
            "neutral_absolute_energy": float(d.mean()),
            "positive_total_energy_ratio": float(np.mean(p / total)),
            "negative_total_energy_ratio": float(np.mean(n / total)),
            "neutral_total_energy_ratio": float(np.mean(d / total)),
        }
    return result


def evaluate_regions_for_mask(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    substrate: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    original = actor_outputs(model, observations, actions, args.batch_size)
    attribution = explanations(model, observations, actions, args.batch_size)
    spatial = attribution.sum(axis=1)
    channelwise_magnitude = np.abs(attribution).sum(axis=1)
    return {
        "P": region_metrics(
            model,
            observations,
            actions,
            original,
            substrate,
            masks > 0,
            spatial,
            channelwise_magnitude,
            args.batch_size,
            args.seed + seed_offset,
            args.random_repeats,
            args.bootstrap_repeats,
        ),
        "N": region_metrics(
            model,
            observations,
            actions,
            original,
            substrate,
            masks < 0,
            spatial,
            channelwise_magnitude,
            args.batch_size,
            args.seed + seed_offset + 1000,
            args.random_repeats,
            args.bootstrap_repeats,
        ),
    }


def mask_overlay(observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(observation[:3], 0, -1).astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    overlay = rgb.copy()
    positive = mask > 0
    negative = mask < 0
    overlay[positive] = 0.35 * overlay[positive] + 0.65 * np.asarray([0.0, 1.0, 0.0])
    overlay[negative] = 0.35 * overlay[negative] + 0.65 * np.asarray([1.0, 0.0, 0.0])
    return np.clip(overlay, 0.0, 1.0)


def representative_examples(dataset: dict[str, np.ndarray]) -> dict[str, tuple[int, int]]:
    actions = dataset["actions"]
    all_masks = dataset["local_masks_all_actions"]
    movement_indices = np.flatnonzero(np.isin(actions, MOVEMENT_ACTIONS))
    if not len(movement_indices):
        raise RuntimeError("Dataset has no movement-action example")
    examples: dict[str, tuple[int, int]] = {
        "movement": (int(movement_indices[0]), int(actions[movement_indices[0]]))
    }

    valid = None
    invalid = None
    # Prefer an interaction the reference policy actually executed, since it
    # is the clearest qualitative example of an operative affordance.
    for index, stored_action in enumerate(actions):
        action = int(stored_action)
        if action not in INTERACTION_ACTIONS:
            continue
        counts = {
            candidate: int(np.count_nonzero(all_masks[index, candidate] > 0))
            for candidate in INTERACTION_ACTIONS
        }
        if counts[action] > min(counts.values()):
            valid = (index, action)
            break

    for index in range(len(actions)):
        counts = {
            action: int(np.count_nonzero(all_masks[index, action] > 0))
            for action in INTERACTION_ACTIONS
        }
        baseline = min(counts.values())
        for action in (3, 5, 4):
            count = counts[action]
            if valid is None and count > baseline:
                valid = (index, action)
            if invalid is None and count == baseline:
                invalid = (index, action)
        if valid is not None and invalid is not None:
            break
    if valid is None or invalid is None:
        raise RuntimeError("Could not identify both valid and invalid interaction examples")
    examples["valid_interaction"] = valid
    examples["invalid_interaction"] = invalid
    return examples


def save_representative_figures(
    models: dict[str, Any],
    dataset: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, Any]] = {}
    for category, (index, action) in representative_examples(dataset).items():
        observation = dataset["observations"][index]
        sequential_mask = dataset["sequential_masks"][index]
        local_mask = dataset["local_masks_all_actions"][index, action]
        per_model = {}
        for name, model in models.items():
            attribution = explanations(
                model,
                observation[None],
                np.asarray([action], dtype=np.int64),
                1,
            )[0].sum(axis=0)
            per_model[name] = attribution
        scale = max(float(np.max(np.abs(value))) for value in per_model.values())
        scale = max(scale, 1e-8)

        figure, axes = plt.subplots(
            len(models), 5, figsize=(17, 3.5 * len(models)), squeeze=False
        )
        rgb = np.moveaxis(observation[:3], 0, -1)
        for row, (name, attribution) in enumerate(per_model.items()):
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(f"{name}: observation")
            axes[row, 1].imshow(mask_overlay(observation, sequential_mask))
            axes[row, 1].set_title("sequential P/N mask")
            axes[row, 2].imshow(mask_overlay(observation, local_mask))
            axes[row, 2].set_title(f"local mask for {action_name(action)}")
            axes[row, 3].imshow(mask_overlay(observation, -sequential_mask))
            axes[row, 3].set_title("flipped sequential mask")
            axes[row, 4].imshow(rgb)
            axes[row, 4].imshow(
                attribution,
                cmap="seismic",
                vmin=-scale,
                vmax=scale,
                alpha=np.clip(np.abs(attribution) / scale, 0.0, 0.8),
            )
            axes[row, 4].set_title("signed B-cos attribution")
            for axis in axes[row]:
                axis.axis("off")
        figure.suptitle(
            f"{category.replace('_', ' ')} | state {index} | action {action_name(action)}"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = output_dir / f"{category}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        metadata[category] = {
            "dataset_index": index,
            "action": action,
            "action_name": action_name(action),
            "figure": str(path.resolve()),
        }
    return metadata


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--reference-model", default="pure")
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--dataset-seed", type=int, default=20_000)
    parser.add_argument("--dataset-episodes", type=int, default=30)
    parser.add_argument("--dataset-samples", type=int, default=256)
    parser.add_argument("--dataset-stochastic", action="store_true")
    parser.add_argument("--dataset-in", type=Path)
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("local_mask_shared_states_rgb6.npz"),
    )
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument("--movement-radius", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--fractions", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--success-tolerance", type=float, default=0.02)
    parser.add_argument("--episode-length-factor", type=float, default=1.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local_mask_experiment_metrics.json"),
    )
    parser.add_argument("--faithfulness-plot", type=Path)
    parser.add_argument("--figures-dir", type=Path, default=Path("local_mask_figures"))
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if args.episodes <= 0 or args.dataset_samples <= 0 or args.dataset_episodes <= 0:
        raise ValueError("Episode and dataset counts must be positive")
    if args.reference_model != "pure":
        print("Warning: a non-pure reference may bias dataset state coverage", flush=True)

    model_paths = parse_models(args.model)
    required_names = {"pure", "sequential", "local", "flipped"}
    missing = required_names.difference(model_paths)
    if missing:
        raise ValueError(f"Required model labels are missing: {sorted(missing)}")
    if args.reference_model not in model_paths:
        raise ValueError(f"Unknown reference model: {args.reference_model}")
    models = {name: load_model(path, args.device) for name, path in model_paths.items()}
    shapes = {name: tuple(model.observation_space.shape) for name, model in models.items()}
    if len(set(shapes.values())) != 1 or next(iter(shapes.values()))[0] != 6:
        raise ValueError(f"All models must use the same RGB6 observation shape: {shapes}")

    dataset = load_or_collect_dataset(models[args.reference_model], args)
    dataset = {
        name: values[: args.dataset_samples]
        for name, values in dataset.items()
    }
    observations = dataset["observations"][: args.dataset_samples]
    actions = dataset["actions"][: args.dataset_samples].astype(np.int64)
    sequential_masks = dataset["sequential_masks"][: args.dataset_samples]
    local_masks = dataset["local_masks"][: args.dataset_samples]
    flipped_masks = -sequential_masks
    expected_shape = next(iter(shapes.values()))
    if tuple(observations.shape[1:]) != expected_shape:
        raise ValueError(f"Dataset/model shape mismatch: {observations.shape[1:]} != {expected_shape}")

    substrate = empty_grid_substrate(tuple(observations.shape[1:]), grid_size=8)
    fractions = parse_fractions(args.fractions)
    result: dict[str, Any] = {
        "experiment": "matched RGB6 local-action mask comparison",
        "policy_evaluation": {
            "episodes": args.episodes,
            "action_mode": "stochastic",
            "identical_seed": args.seed,
        },
        "dataset": {
            "path": str((args.dataset_in or args.dataset_out).resolve()),
            "reference_model": args.reference_model,
            "action_mode": "stochastic" if args.dataset_stochastic else "deterministic",
            "samples": int(len(observations)),
            "seed": args.dataset_seed,
        },
        "mask_definitions": {
            "sequential": "P=agent+current key/door/goal objective; N=walls+grid lines",
            "local": "action-conditioned local affordance mask with no task phase or global-object selection",
            "flipped": "exact sequential mask with P and N exchanged",
        },
        "faithfulness_methods": {
            "deletion_insertion": "Petsiuk et al. (RISE), BMVC 2018",
            "region_comprehensiveness_sufficiency": "DeYoung et al. (ERASER), ACL 2020",
            "sensitivity_n": "Ancona et al., ICLR 2018",
            "infidelity": "Yeh et al., NeurIPS 2019",
            "substrate": "valid rendered empty RGB6 MiniGrid grid",
            "secondary_substrate": "all six channels zero; off-manifold sensitivity analysis only",
        },
        "models": {},
    }
    outputs_by_model = {}
    faithfulness_plot_result = {"models": {}}
    for model_index, (name, model) in enumerate(models.items()):
        print(f"Evaluating {name}", flush=True)
        policy = evaluate_policy_stochastic(model, args)
        distribution = evaluate_actor_distribution(model, observations, args.batch_size)
        outputs_by_model[name] = actor_outputs(model, observations, actions, args.batch_size)
        alignment = {
            "sequential": evaluate_channelwise_absolute_explanations(
                model, observations, actions, sequential_masks, args.batch_size
            ),
            "local": evaluate_channelwise_absolute_explanations(
                model, observations, actions, local_masks, args.batch_size
            ),
            "flipped": evaluate_channelwise_absolute_explanations(
                model, observations, actions, flipped_masks, args.batch_size
            ),
        }
        faithfulness = evaluate_faithfulness(
            model,
            observations,
            actions,
            sequential_masks,
            substrate,
            fractions,
            args.batch_size,
            args.seed + model_index * 100_000,
            args.random_repeats,
            args.bootstrap_repeats,
            "stored",
        )
        sequential_regions = faithfulness.pop("regions")
        faithfulness["regions_by_mask"] = {
            "sequential": sequential_regions,
            "local": evaluate_regions_for_mask(
                model,
                observations,
                actions,
                local_masks,
                substrate,
                args,
                70_000 + model_index * 10_000,
            ),
            "flipped": evaluate_regions_for_mask(
                model,
                observations,
                actions,
                flipped_masks,
                substrate,
                args,
                90_000 + model_index * 10_000,
            ),
        }
        result["models"][name] = {
            "path": str(model_paths[name].resolve()),
            "continuation": continuation_metadata(model_paths[name]),
            "saved_timesteps": int(model.num_timesteps),
            "saved_explanation_coef": float(getattr(model, "explanation_coef", 0.0)),
            "policy": policy,
            "fixed_dataset_actor_distribution": distribution,
            "mask_alignment": alignment,
            "pnd_absolute_energy": pnd_absolute_energy(
                model,
                observations,
                actions,
                {
                    "sequential": sequential_masks,
                    "local": local_masks,
                    "flipped": flipped_masks,
                },
                args.batch_size,
            ),
            "faithfulness": faithfulness,
            "zero_substrate_sensitivity": evaluate_deletion_insertion_sensitivity(
                model,
                observations,
                actions,
                np.zeros_like(substrate),
                fractions,
                args.batch_size,
                args.seed + 200_000 + model_index * 100_000,
                args.random_repeats,
                args.bootstrap_repeats,
                "stored",
            ),
        }
        faithfulness_plot_result["models"][name] = {
            "faithfulness": faithfulness,
            "zero_substrate_sensitivity": result["models"][name][
                "zero_substrate_sensitivity"
            ],
        }

    result["fixed_dataset_pairwise_policy_change"] = pairwise_policy_metrics(
        outputs_by_model
    )
    figures = save_representative_figures(models, dataset, args.figures_dir)
    result["representative_figures"] = figures

    pure_policy = result["models"]["pure"]["policy"]
    local_policy = result["models"]["local"]["policy"]
    local_summary = result["models"]["local"]["faithfulness"]["summary"]
    deletion_ci = local_summary["bootstrap_95_ci"]["deletion_advantage_over_random"]
    insertion_ci = local_summary["bootstrap_95_ci"]["insertion_advantage_over_random"]
    policy_preserved = (
        local_policy["success_rate"] >= pure_policy["success_rate"] - args.success_tolerance
        and local_policy["mean_episode_length"]
        <= pure_policy["mean_episode_length"] * args.episode_length_factor
    )
    causal_faithfulness = deletion_ci["low"] > 0 and insertion_ci["low"] > 0
    result["claim_test"] = {
        "claim": "local mask preserves policy quality and causal faithfulness without global phase information",
        "policy_preserved": bool(policy_preserved),
        "policy_criteria": {
            "success_rate_tolerance": args.success_tolerance,
            "maximum_episode_length_factor": args.episode_length_factor,
        },
        "causal_faithfulness": bool(causal_faithfulness),
        "faithfulness_criterion": "95% bootstrap lower bounds for deletion and insertion advantage over random are positive",
        "no_global_phase_information_by_construction": True,
        "supported": bool(policy_preserved and causal_faithfulness),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_path = args.faithfulness_plot or args.output.with_suffix(".png")
    save_faithfulness_plot(faithfulness_plot_result, plot_path)

    print("\nLocal-mask experiment summary:")
    for name, model_result in result["models"].items():
        policy = model_result["policy"]
        faith = model_result["faithfulness"]["summary"]
        print(
            f"- {name}: success={policy['success_rate']:.3f}, "
            f"reward={policy['mean_reward']:.3f}, steps={policy['mean_episode_length']:.2f}, "
            f"entropy={model_result['fixed_dataset_actor_distribution']['mean_entropy']:.3f}, "
            f"deletion_advantage={faith['deletion_advantage_over_random']:.3f}, "
            f"insertion_advantage={faith['insertion_advantage_over_random']:.3f}"
        )
    print(f"Claim supported: {result['claim_test']['supported']}")
    print(f"Saved metrics: {args.output}")
    print(f"Saved faithfulness curves: {plot_path}")
    print(f"Saved representative figures: {args.figures_dir}")


if __name__ == "__main__":
    main()

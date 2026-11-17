#!/usr/bin/env python3
"""Compare PPO policies and explanations under wall-appearance shifts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
from minigrid.core.constants import COLOR_TO_IDX

from evaluate_explanation_metrics import (
    evaluate_actor_distribution,
    evaluate_explanations,
    evaluate_policy,
    load_model,
)
from train_minigrid_ActorCritic import build_vec_env


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Model label and checkpoint path. Repeat for every model.",
    )
    parser.add_argument(
        "--reference-model",
        help="Model whose trajectories and actions define each shared dataset.",
    )
    parser.add_argument(
        "--variants",
        default="normal,hidden,yellow",
        help="Comma-separated wall variants: normal, hidden, or a MiniGrid color.",
    )
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument(
        "--grid-lines-as-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions. Stochastic evaluation is the default.",
    )
    parser.add_argument("--max-explanation-samples", type=int, default=512)
    parser.add_argument("--explanation-batch-size", type=int, default=32)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wall_ood_metrics.json"),
    )
    return parser.parse_args()


def parse_models(specs: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH for --model, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name or name in models:
            raise ValueError(f"Model labels must be non-empty and unique: {name!r}")
        models[name] = Path(raw_path)
    return models


def parse_variants(raw: str) -> list[str]:
    variants = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if not variants:
        raise ValueError("--variants must contain at least one variant")
    valid = {"normal", "hidden", *COLOR_TO_IDX.keys()}
    unknown = [variant for variant in variants if variant not in valid]
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}; use normal, hidden, or {sorted(COLOR_TO_IDX)}"
        )
    if len(set(variants)) != len(variants):
        raise ValueError("--variants must not contain duplicates")
    return variants


def wall_settings(variant: str) -> tuple[str, str]:
    if variant == "normal":
        return "normal", "yellow"
    if variant == "hidden":
        return "hidden", "yellow"
    return "recolored", variant


def make_env(
    args: argparse.Namespace,
    *,
    observation_channels: int,
    variant: str,
):
    appearance, color = wall_settings(variant)
    return build_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed,
        use_rgb_img_obs=True,
        invert_channels=observation_channels == 6,
        wall_appearance=appearance,
        wall_color=color,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=True,
        sequential_walls_as_negative=True,
        sequential_grid_lines_as_negative=args.grid_lines_as_negative,
    )


def evaluate_model_policy(
    args: argparse.Namespace,
    model,
    *,
    observation_channels: int,
    variant: str,
    collect: bool,
):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    env = make_env(
        args,
        observation_channels=observation_channels,
        variant=variant,
    )
    try:
        return evaluate_policy(
            model,
            env,
            episodes=args.episodes,
            sample_count=args.max_explanation_samples,
            seed=args.seed,
            collect=collect,
            deterministic=args.deterministic,
        )
    finally:
        env.close()


def main() -> None:
    args = make_args()
    if args.episodes <= 0 or args.max_explanation_samples <= 0:
        raise ValueError("Episode and explanation sample counts must be positive")
    if args.explanation_batch_size <= 0:
        raise ValueError("--explanation-batch-size must be positive")

    model_paths = parse_models(args.model)
    variants = parse_variants(args.variants)
    reference_name = args.reference_model or next(iter(model_paths))
    if reference_name not in model_paths:
        raise ValueError(f"Unknown --reference-model label: {reference_name}")

    models = {
        name: load_model(path, args.device)
        for name, path in model_paths.items()
    }
    shapes = {
        name: tuple(model.observation_space.shape)
        for name, model in models.items()
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All checkpoints must use the same observation shape: {shapes}")
    observation_shape = next(iter(shapes.values()))
    if len(observation_shape) != 3 or observation_shape[0] not in (3, 6):
        raise ValueError(
            "Expected channels-first RGB checkpoints with 3 or 6 channels, "
            f"got {observation_shape}"
        )
    observation_channels = int(observation_shape[0])

    result = {
        "reference_model": reference_name,
        "shared_explanation_actions": "reference model stored actions",
        "policy_action_mode": (
            "deterministic" if args.deterministic else "stochastic"
        ),
        "episodes": args.episodes,
        "seed": args.seed,
        "observation_channels": observation_channels,
        "grid_lines_as_negative": args.grid_lines_as_negative,
        "models": {
            name: {
                "path": str(path.resolve()),
                "saved_timesteps": int(models[name].num_timesteps),
                "saved_explanation_coef": float(
                    getattr(models[name], "explanation_coef", 0.0)
                ),
            }
            for name, path in model_paths.items()
        },
        "variants": {},
    }

    for variant in variants:
        print(f"Evaluating wall variant: {variant}")
        reference_policy, dataset = evaluate_model_policy(
            args,
            models[reference_name],
            observation_channels=observation_channels,
            variant=variant,
            collect=True,
        )
        assert dataset is not None
        observations, stored_actions, masks = dataset

        if args.dataset_dir:
            args.dataset_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.dataset_dir / f"{variant}_shared_states.npz",
                observations=observations,
                actions=stored_actions,
                masks=masks,
            )

        variant_result = {
            "wall_appearance": wall_settings(variant)[0],
            "wall_color": (
                variant if variant not in ("normal", "hidden") else None
            ),
            "models": {},
        }
        for name, model in models.items():
            if name == reference_name:
                policy_metrics = reference_policy
            else:
                policy_metrics, _ = evaluate_model_policy(
                    args,
                    model,
                    observation_channels=observation_channels,
                    variant=variant,
                    collect=False,
                )
            explanation_metrics = evaluate_explanations(
                model,
                observations,
                stored_actions,
                masks,
                args.explanation_batch_size,
            )
            actor_distribution = evaluate_actor_distribution(
                model,
                observations,
                args.explanation_batch_size,
            )
            variant_result["models"][name] = {
                "policy": policy_metrics,
                "explanation": explanation_metrics,
                "actor_distribution": actor_distribution,
            }
        result["variants"][variant] = variant_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved OOD comparison to: {args.output}")


if __name__ == "__main__":
    main()

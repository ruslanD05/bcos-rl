#!/usr/bin/env python3
"""Evaluate the five matched RQ3 policies on custom fixed wall colours."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from evaluate_explanation_metrics import load_model
from evaluate_rq3_wall_colors import evaluate_vector_policy
from fixed_wall_color_env import make_fixed_wall_color_vec_env
from run_rq3_evaluation import locate_model


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("pure", "sequential", "local")
DEFAULT_COLOURS = {
    "black": (20, 20, 20),
    "cyan": (0, 180, 180),
    "magenta": (180, 0, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "rq5_wall_colour" / "rq3_custom_wall_colors",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value]
    if sorted(values) != list(range(5)):
        raise ValueError("This matched evaluation requires exactly seeds 0,1,2,3,4")
    return values


def evaluate(args: argparse.Namespace, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for training_seed in seeds:
        evaluation_seed = args.eval_seed + training_seed
        for branch in BRANCHES:
            model_path = locate_model(args.models_root, training_seed, branch)
            model = load_model(model_path, args.device)
            if tuple(model.observation_space.shape) != (6, 64, 64):
                raise ValueError(f"Expected RGB6 64x64 checkpoint, got {model.observation_space.shape}")
            for colour, rgb in DEFAULT_COLOURS.items():
                np.random.seed(evaluation_seed)
                th.manual_seed(evaluation_seed)
                env = make_fixed_wall_color_vec_env(
                    env_id="MiniGrid-DoorKey-8x8-v0",
                    n_envs=args.n_envs,
                    seed=evaluation_seed,
                    wall_rgb=rgb,
                    observation_channels=6,
                    open_door_reward=args.open_door_reward,
                    pick_key_reward=args.pick_key_reward,
                )
                try:
                    metrics = evaluate_vector_policy(
                        model, env, episodes=args.episodes, deterministic=args.deterministic
                    )
                finally:
                    env.close()
                rows.append(
                    {
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "branch": branch,
                        "colour": colour,
                        "rgb": rgb,
                        "model_path": str(model_path.resolve()),
                        **metrics,
                    }
                )
                print(
                    f"seed={training_seed} branch={branch} colour={colour}: "
                    f"success={metrics['success_rate']:.3f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.n_envs <= 0:
        raise ValueError("--episodes and --n-envs must be positive")
    per_seed = evaluate(args, parse_seeds(args.seeds))
    summary = (
        per_seed.groupby(["branch", "colour"], as_index=False)
        .agg(
            mean_success_rate=("success_rate", "mean"),
            sd_success_rate=("success_rate", "std"),
            mean_episode_length=("mean_episode_length", "mean"),
            n_seeds=("training_seed", "count"),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "colours": DEFAULT_COLOURS,
                "episodes_per_model_colour": args.episodes,
                "action_mode": "deterministic" if args.deterministic else "stochastic",
                "evaluation_seed_base": args.eval_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()

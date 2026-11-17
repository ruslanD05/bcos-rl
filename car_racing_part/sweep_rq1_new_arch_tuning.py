#!/usr/bin/env python3
"""Focused CarRacing tuning for the final RGB6/uncentered-BN B-cos RQ1 actor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from car_racing_part.sweep_rq1_carracing_architecture import train_one
from minigrid_part.rq1_sweep_common import DEFAULT_OUTPUT_ROOT


def sweep_config() -> dict:
    return {
        "name": "rq1-carracing-rgb6-uncentered-bn-focused-tuning-v1",
        "method": "bayes",
        "metric": {"name": "final_eval/mean_reward", "goal": "maximize"},
        "parameters": {
            "variant": {"value": "bcos"},
            "seed": {"value": 100},
            "stable_bcos_norm": {"value": False},
            "bcos_complement_input": {"value": True},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 5e-5,
                "max": 2e-4,
            },
            "ent_coef": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 8e-3,
            },
        },
    }


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="rq1-carracing-ppo-new-architecture-tuning-v1")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--bcos-b", type=float, default=2.3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if args.create_sweep:
        sweep_id = wandb.sweep(sweep_config(), project=args.project, entity=args.entity)
        print(f"Created sweep: {sweep_id}")
        return
    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            function=lambda: train_one(args),
            count=args.count,
            project=args.project,
            entity=args.entity,
        )
        return
    raise SystemExit("Pass --create-sweep or --sweep-id.")


if __name__ == "__main__":
    main()

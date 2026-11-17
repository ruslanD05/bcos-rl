#!/usr/bin/env python3
"""Focused DoorKey tuning for the final RGB6/uncentered-BN B-cos RQ1 actor."""
from __future__ import annotations

import argparse

from rq1_sweep_common import common_agent_args, run_sweep_cli, train_minigrid_ppo


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_agent_args(
        parser,
        default_project="rq1-minigrid-ppo-new-architecture-tuning-v1",
        default_experiment="minigrid_ppo_rgb6_uncentered_bn_tuning",
    )
    parser.add_argument("--total-timesteps", type=int, default=750_000)
    parser.add_argument("--n-envs", type=int, default=8)
    return parser.parse_args()


def tuning_defaults(args: argparse.Namespace) -> dict:
    return {
        "experiment": args.experiment,
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "variant": "bcos_b2.3",
        "seed": 100,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "learning_rate": 0.00022569760177299956,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.995,
        "gae_lambda": 0.90,
        "clip_range": 0.1,
        "target_kl": 0.05,
        "ent_coef": 0.0003215278796080832,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "features_dim": 256,
        "pi_layers": "128,128",
        "vf_layers": "256,256",
        "activation": "tanh",
        "stable_bcos_norm": False,
        "open_door_reward": 0.1,
        "pick_key_reward": 0.05,
        "eval_freq": 50_000,
        "checkpoint_freq": 250_000,
        "periodic_eval_episodes": 20,
        "final_eval_episodes": 100,
        "tuning_protocol": "single-seed-search-then-held-out-validation",
    }


def sweep_config() -> dict:
    return {
        "name": "rq1-minigrid-rgb6-uncentered-bn-focused-tuning-v1",
        "method": "bayes",
        "metric": {"name": "final_eval/success_rate", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1.2e-4,
                "max": 3.0e-4,
            },
            "ent_coef": {
                "distribution": "log_uniform_values",
                "min": 1.5e-4,
                "max": 7.0e-4,
            },
        },
    }


def main() -> None:
    args = make_args()
    run_sweep_cli(args, sweep_config(), train_minigrid_ppo, tuning_defaults(args))


if __name__ == "__main__":
    main()

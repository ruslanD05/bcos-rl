#!/usr/bin/env python3
"""Five-seed validation of the three selected shaped-reward PPO configs."""
from __future__ import annotations

import argparse

from rq1_sweep_common import common_agent_args, run_sweep_cli, train_minigrid_ppo


PROFILES = {
    "revived": {
        "learning_rate": 0.00022916476161434724,
        "ent_coef": 0.0004112404740243896,
        "n_steps": 256,
        "n_epochs": 8,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "target_kl": 0.05,
        "max_grad_norm": 1.0,
    },
    "earthy": {
        "learning_rate": 0.00016879698835114228,
        "ent_coef": 0.0004310733175380074,
        "n_steps": 256,
        "n_epochs": 4,
        "gamma": 0.995,
        "gae_lambda": 0.90,
        "target_kl": 0.03,
        "max_grad_norm": 1.0,
    },
    "stellar": {
        "learning_rate": 0.00022569760177299956,
        "ent_coef": 0.0003215278796080832,
        "n_steps": 512,
        "n_epochs": 4,
        "gamma": 0.995,
        "gae_lambda": 0.90,
        "target_kl": 0.05,
        "max_grad_norm": 1.0,
    },
}


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_agent_args(
        parser,
        default_project="rq1-minigrid-ppo-shaped-validation",
        default_experiment="minigrid_ppo_rgb6_shaped_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = make_args()
    defaults = {
        "experiment": args.experiment,
        "config_profile": "revived",
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "variant": "bcos_b2.3",
        "seed": 0,
        "total_timesteps": 750_000,
        "n_envs": 8,
        "batch_size": 128,
        "clip_range": 0.1,
        "vf_coef": 1.0,
        "features_dim": 256,
        "pi_layers": "128,128",
        "vf_layers": "256,256",
        "activation": "tanh",
        "stable_bcos_norm": True,
        "open_door_reward": 0.1,
        "pick_key_reward": 0.05,
        "eval_freq": 50_000,
        "checkpoint_freq": 250_000,
        "periodic_eval_episodes": 20,
        "final_eval_episodes": 100,
        **PROFILES["revived"],
    }
    sweep = {
        "name": "rq1-ppo-top3-five-seed-validation",
        "method": "grid",
        "metric": {"name": "final_eval/success_rate", "goal": "maximize"},
        "parameters": {
            "config_profile": {"values": list(PROFILES)},
            "seed": {"values": [0, 1, 2, 3, 4]},
        },
    }
    train = lambda run_args, run_defaults: train_minigrid_ppo(
        run_args, run_defaults, PROFILES
    )
    run_sweep_cli(args, sweep, train, defaults)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""RQ1 MiniGrid DQN sweep: matched standard and B-cos Q-networks."""
from __future__ import annotations

import argparse

from rq1_sweep_common import (
    common_agent_args,
    run_sweep_cli,
    train_minigrid_dqn,
)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_agent_args(
        parser,
        default_project="rq1-minigrid-dqn-architecture-final",
        default_experiment="minigrid_dqn_architecture",
    )
    parser.add_argument("--total-timesteps", type=int, default=750_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--bcos-b", type=float, default=2.3)
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    defaults = {
        "experiment": args.experiment,
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "variant": "standard",
        "seed": 0,
        "bcos_b": args.bcos_b,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        # Fixed from calibration run 9d0vkesj (hearty-sweep-3): the only
        # sustained 100%-success configuration in the single-seed search.
        "learning_rate": 0.00020948656109281204,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 128,
        "gamma": 0.98,
        "train_freq": 4,
        "gradient_steps": 8,
        "tau": 1.0,
        "target_update_interval": 500,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.1,
        "max_grad_norm": 5.0,
        "features_dim": 256,
        "q_layers": "256,128",
        "activation": "relu",
        "stable_bcos_norm": True,
        "open_door_reward": args.open_door_reward,
        "pick_key_reward": args.pick_key_reward,
        "eval_freq": 50_000,
        "checkpoint_freq": 250_000,
        "periodic_eval_episodes": 20,
        "final_eval_episodes": 100,
    }
    sweep = {
        "name": "rq1-minigrid-dqn-architecture-final",
        "method": "grid",
        "metric": {"name": "final_eval/success_rate", "goal": "maximize"},
        "parameters": {
            "variant": {"values": ["standard", "bcos"]},
            "seed": {"values": [0, 1, 2, 3, 4]},
        },
    }
    run_sweep_cli(args, sweep, train_minigrid_dqn, defaults)


if __name__ == "__main__":
    main()

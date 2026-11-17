#!/usr/bin/env python3
"""RQ1 support sweep: matched PPO architectures on LavaCrossing."""
from __future__ import annotations

import argparse

from rq1_sweep_common import (
    common_agent_args,
    run_sweep_cli,
    train_minigrid_ppo,
)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_agent_args(
        parser,
        default_project="rq1-lavacrossing-ppo-architecture-final",
        default_experiment="lavacrossing_ppo_architecture",
    )
    parser.add_argument("--total-timesteps", type=int, default=750_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--bcos-b", type=float, default=2.3)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    bcos_variant = f"bcos_b{args.bcos_b:.1f}"
    defaults = {
        "experiment": args.experiment,
        "env_id": "MiniGrid-LavaCrossingS9N1-v0",
        "variant": "standard",
        "seed": 0,
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
        "stable_bcos_norm": True,
        "open_door_reward": 0.1,
        "pick_key_reward": 0.05,
        "eval_freq": 50_000,
        "checkpoint_freq": 250_000,
        "periodic_eval_episodes": 20,
        "final_eval_episodes": 100,
    }
    sweep = {
        "name": "rq1-lavacrossing-ppo-architecture-final",
        "method": "grid",
        "metric": {"name": "final_eval/success_rate", "goal": "maximize"},
        "parameters": {
            "env_id": {"value": "MiniGrid-LavaCrossingS9N1-v0"},
            "variant": {"values": ["standard", bcos_variant]},
            "seed": {"values": [0, 1, 2, 3, 4]},
        },
    }
    run_sweep_cli(args, sweep, train_minigrid_ppo, defaults)


if __name__ == "__main__":
    main()

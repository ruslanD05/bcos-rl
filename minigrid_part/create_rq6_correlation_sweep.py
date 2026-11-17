#!/usr/bin/env python3
"""Create W&B sweep for RQ6 Shortcut Correlation Sweep (50%, 60%, 70%, 80%, 90%, 100%)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import wandb


def parse_csv(value: str, cast: type = float) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Create W&B sweep for shortcut correlation analysis")
    parser.add_argument("--project", default="rq6-minigrid-shortcut-correlation-sweep")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--correlations", default="0.50,0.60,0.70,0.80,0.90,1.00")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    args = parser.parse_args()

    corrs = parse_csv(args.correlations, float)
    seeds = parse_csv(args.seeds, int)

    cfg = {
        "env_id": "MiniGrid-DoorKey-6x6-v0",
        "total_timesteps": args.total_timesteps,
        "n_envs": 4,
        "device": "cpu",
        "border_height": 8,
        "patch_size": 6,
        "mask_profile": "border_negative_screen_positive",
        "explanation_loss_type": "channelwise_masked_absolute_energy",
        "explanation_coef": 0.15,
        "explanation_max_grad_ratio": 0.1,
        "eval_episodes": 100,
        "max_paired_states": 128,
        "source_success_threshold": 0.90,
        "eval_stochastic": True,
        "learning_rate": 5.31e-4,
        "n_steps": 512,
        "batch_size": 512,
        "n_epochs": 8,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 1.0,
        "features_dim": 128,
        "pi_layers": "128,128",
        "vf_layers": "128,128",
        "bcos_b": 2.3,
        "actor_type": "bcos",
        "critic_type": "mlp",
        "bcos_max_out": 1,
        "stable_bcos_norm": True,
        "bias_free_bcos_norm": True,
    }

    parameters = {key: {"value": value} for key, value in cfg.items()}
    parameters["training_correlation"] = {"values": corrs}
    parameters["seed"] = {"values": seeds}

    sweep_definition = {
        "method": "grid",
        "name": "rq6_shortcut_correlation_sweep",
        "metric": {
            "name": "paired/permuted_success_drop_reduction",
            "goal": "maximize",
        },
        "parameters": parameters,
    }

    sweep_id = wandb.sweep(
        sweep_definition,
        project=args.project,
        entity=args.entity or None,
    )

    print(f"\n=======================================================")
    print(f"  CREATED W&B SWEEP ID: {sweep_id}")
    print(f"  Project: {args.project}")
    print(f"  Correlations: {corrs}")
    print(f"  Seeds: {seeds}")
    print(f"  Total Combinations: {len(corrs) * len(seeds)}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()

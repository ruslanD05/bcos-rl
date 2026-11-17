#!/usr/bin/env python3
"""Matched RGB6 PPO study: pure, semantic sequential, and local action masks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from wandb.integration.sb3 import WandbCallback

from env_wrappers import MiniGridStatsCallback, make_wrapper

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO, SuccessRateCheckpointCallback
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from sweep_minigrid_PPO import make_policy_kwargs


ROOT = Path(__file__).resolve().parent
VERSION = "rgb6-local-action-mask-v1"
CONDITIONS = ("pure", "sequential_semantic", "local_action")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integers")
    return seeds


def fixed_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_version": VERSION,
        "condition": args.condition,
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device": args.device,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "checkpoint_interval": args.checkpoint_interval,
        "open_door_reward": args.open_door_reward,
        "pick_key_reward": args.pick_key_reward,
        "explanation_coef": args.explanation_coef,
        "configured_explanation_coef": args.explanation_coef,
        "explanation_coef_schedule": "constant",
        "explanation_max_grad_ratio": args.explanation_max_grad_ratio,
        "movement_radius": args.movement_radius,
        "explanation_loss_type": "masked_absolute_energy",
        "learning_rate": 0.0006141598118159885,
        "n_steps": 512,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "target_kl": 0.05,
        "ent_coef": 0.0009960268117654178,
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "pi_layers": "128,128",
        "vf_layers": "128,128",
        "features_dim": 128,
        "bcos_b": 2.3,
        "bcos_max_out": 1,
        "actor_type": "bcos",
        "critic_type": "mlp",
        "critic_activation": "tanh",
        "stable_bcos_norm": True,
        "bias_free_bcos_norm": True,
        "use_rgb_img_obs": True,
        "invert_channels": True,
        "observation_channels": 6,
        "confidence_coef": 0.0,
    }


def sweep_config(args: argparse.Namespace) -> dict[str, Any]:
    parameters = {key: {"value": value} for key, value in fixed_config(args).items()}
    parameters["condition"] = {"values": list(CONDITIONS)}
    parameters["seed"] = {"values": parse_seeds(args.seeds)}
    return {
        "method": "grid",
        "metric": {"name": "eval/success_rate", "goal": "maximize"},
        "parameters": parameters,
    }


class FinalEvalWandbCallback(BaseCallback):
    def __init__(self, evaluation: EvalCallback):
        super().__init__(verbose=0)
        self.evaluation = evaluation
        self._last_call = -1

    def _on_step(self) -> bool:
        callback = self.evaluation
        if callback.n_calls == self._last_call or callback.n_calls == 0:
            return True
        if callback.n_calls % callback.eval_freq or not callback._is_success_buffer:
            return True
        self._last_call = callback.n_calls
        wandb.log(
            {
                "eval/success_rate": float(np.mean(callback._is_success_buffer)),
                "eval/mean_reward": float(callback.last_mean_reward),
            },
            step=self.num_timesteps,
        )
        return True


def build_env(cfg: dict[str, Any], *, n_envs: int, seed: int, condition: str):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    env = make_vec_env(
        env_id=str(cfg["env_id"]),
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"highlight": False},
        wrapper_class=make_wrapper(
            use_rgb_img_obs=True,
            invert_rgb_channels=True,
            open_door_reward=float(cfg["open_door_reward"]),
            pick_key_reward=float(cfg["pick_key_reward"]),
            use_sequential_mask=condition == "sequential_semantic",
            sequential_walls_as_negative=condition == "sequential_semantic",
            sequential_grid_lines_as_negative=condition == "sequential_semantic",
            use_local_action_mask=condition == "local_action",
            local_action_movement_radius=int(cfg["movement_radius"]),
        ),
    )
    return VecTransposeImage(env)


def train_one_run(args: argparse.Namespace) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        config=fixed_config(args),
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=False,
        tags=[item.strip() for item in args.tags.split(",") if item.strip()],
    )
    cfg = dict(wandb.config)
    condition = str(cfg["condition"])
    if condition not in CONDITIONS:
        raise ValueError(f"Invalid condition: {condition}")
    if int(cfg["total_timesteps"]) != 1_000_000:
        raise ValueError("This matched experiment requires 1,000,000 timesteps")
    effective_coef = 0.0 if condition == "pure" else float(cfg["explanation_coef"])
    if condition != "pure" and effective_coef <= 0:
        raise ValueError("Explanation conditions require a positive coefficient")

    descriptions = {
        "pure": "No explanation mask or explanation loss",
        "sequential_semantic": "P=agent+current global subgoal; N=walls+grid lines",
        "local_action": "Movement P=agent+local geometry with interactive objects D; interaction P=agent+valid front affordance; all other pixels N",
    }
    wandb.config.update(
        {
            "effective_explanation_coef": effective_coef,
            "mask_description": descriptions[condition],
            "uses_global_task_phase": condition == "sequential_semantic",
        },
        allow_val_change=True,
    )

    run_dir = ROOT / "wandb_runs" / "local_action_mask" / run.id
    best_dir = run_dir / "best_model"
    checkpoint_dir = run_dir / "checkpoints"
    model_dir = run_dir / "models"
    for directory in (best_dir, checkpoint_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    n_envs = int(cfg["n_envs"])
    seed = int(cfg["seed"])
    train_env = build_env(cfg, n_envs=n_envs, seed=seed, condition=condition)
    eval_env = build_env(cfg, n_envs=1, seed=seed + 10_000, condition="pure")
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=train_env,
        bce_coef=0.0,
        explanation_coef=effective_coef,
        explanation_loss_type="masked_absolute_energy",
        explanation_max_grad_ratio=float(cfg["explanation_max_grad_ratio"]),
        learning_rate=float(cfg["learning_rate"]),
        n_steps=int(cfg["n_steps"]),
        batch_size=int(cfg["batch_size"]),
        n_epochs=int(cfg["n_epochs"]),
        gamma=float(cfg["gamma"]),
        gae_lambda=float(cfg["gae_lambda"]),
        clip_range=float(cfg["clip_range"]),
        target_kl=float(cfg["target_kl"]),
        ent_coef=float(cfg["ent_coef"]),
        vf_coef=float(cfg["vf_coef"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        policy_kwargs=make_policy_kwargs(cfg),
        tensorboard_log=str(run_dir / "tensorboard"),
        gradient_diagnostics=effective_coef > 0,
        seed=seed,
        device=str(cfg["device"]),
        verbose=1,
    )
    if tuple(model.observation_space.shape)[0] != 6:
        raise RuntimeError(f"Expected RGB6 CHW observations, got {model.observation_space.shape}")

    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(int(cfg["eval_freq"]) // n_envs, 1),
        n_eval_episodes=int(cfg["n_eval_episodes"]),
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks = CallbackList(
        [
            MiniGridStatsCallback(window_size=100),
            SuccessRateCheckpointCallback(
                save_dir=str(checkpoint_dir),
                threshold=0.95,
                interval=int(cfg["checkpoint_interval"]),
            ),
            evaluation,
            FinalEvalWandbCallback(evaluation),
            WandbCallback(
                model_save_path=str(model_dir),
                model_save_freq=max(int(cfg["checkpoint_interval"]) // n_envs, 1),
                gradient_save_freq=0,
                verbose=1,
            ),
        ]
    )
    try:
        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=callbacks,
            tb_log_name=f"local_action_{run.id}",
        )
        final_path = run_dir / f"final_{condition}.zip"
        model.save(final_path)
        artifact = wandb.Artifact(
            name=f"local-action-mask-{condition}-{run.id}",
            type="model",
            metadata={"condition": condition, "seed": seed},
        )
        artifact.add_file(str(final_path))
        run.log_artifact(artifact)
    finally:
        train_env.close()
        eval_env.close()
        run.finish()


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="minigrid-rgb6-local-action-mask-v1")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--condition", choices=CONDITIONS, default="local_action")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument("--explanation-coef", type=float, default=0.2)
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1)
    parser.add_argument("--movement-radius", type=int, default=1)
    parser.add_argument(
        "--tags",
        default="minigrid,ppo,bcos,rgb6,action-conditioned-mask,matched-study",
    )
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if args.create_sweep:
        sweep_id = wandb.sweep(
            sweep_config(args), project=args.project, entity=args.entity or None
        )
        print(f"Created sweep: {sweep_id}")
    elif args.sweep_id:
        wandb.agent(
            sweep_id=args.sweep_id,
            function=lambda: train_one_run(args),
            count=args.count,
            project=args.project,
            entity=args.entity or None,
        )
    else:
        train_one_run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""W&B hyperparameter sweep for the strict no-auxiliary RGB B-cos PPO baseline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

import pointing_game_env
from env_wrappers import MiniGridStatsCallback, make_wrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO, SuccessRateCheckpointCallback
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    GenericCNNExtractor,
    activation_from_name,
    parse_layer_sizes,
    validate_layer_sizes,
)


ROOT = Path(__file__).resolve().parent
ARCHITECTURE_VERSION = "rgb6-bcos-actor-standard-critic-v1"

gym.register(
    id="MiniGrid-Composite-15x15-v0",
    entry_point="pointing_game_env:CompositeGridEnv",
)


class BaselineObjectiveCallback(BaseCallback):
    """Log success first, then mean reward as a small tie-breaker."""

    def __init__(self, eval_callback: EvalCallback):
        super().__init__(verbose=0)
        self.eval_callback = eval_callback
        self._last_eval_call = -1

    def _on_step(self) -> bool:
        evaluation_finished = (
            self.eval_callback.n_calls > 0
            and self.eval_callback.n_calls % self.eval_callback.eval_freq == 0
            and self.eval_callback.n_calls != self._last_eval_call
        )
        if evaluation_finished and self.eval_callback._is_success_buffer:
            self._last_eval_call = self.eval_callback.n_calls
            success_rate = float(np.mean(self.eval_callback._is_success_buffer))
            score = success_rate + 1e-3 * self.eval_callback.last_mean_reward
            wandb.log({"eval/baseline_score": score}, step=self.num_timesteps)
        return True


def baseline_sweep_config(
    seed: int,
    *,
    env_id: str = "MiniGrid-DoorKey-8x8-v0",
    total_timesteps: int = 600_000,
    n_envs: int = 8,
    device: str = "cpu",
    eval_freq: int = 50_000,
    n_eval_episodes: int = 20,
    model_save_freq: int = 150_000,
    checkpoint_threshold: float = 0.95,
    checkpoint_interval: int = 100_000,
    shortcut_prob: float = 0.0,
    open_door_reward: float = 0.0,
    pick_key_reward: float = 0.0,
    use_inventory_obs: bool = False,
) -> dict[str, Any]:
    """Search optimization choices while keeping the baseline architecture fixed."""
    return {
        "method": "bayes",
        "metric": {"name": "eval/baseline_score", "goal": "maximize"},
        "parameters": {
            # Search one seed first. Top configurations should be rerun with
            # --seed 1/2/... using separate sweeps after ranking.
            "seed": {"value": seed},
            "architecture_version": {"value": ARCHITECTURE_VERSION},
            "env_id": {"value": env_id},
            "total_timesteps": {"value": total_timesteps},
            "n_envs": {"value": n_envs},
            "device": {"value": device},
            "eval_freq": {"value": eval_freq},
            "n_eval_episodes": {"value": n_eval_episodes},
            "model_save_freq": {"value": model_save_freq},
            "checkpoint_threshold": {"value": checkpoint_threshold},
            "checkpoint_interval": {"value": checkpoint_interval},
            "shortcut_prob": {"value": shortcut_prob},
            "open_door_reward": {"value": open_door_reward},
            "pick_key_reward": {"value": pick_key_reward},
            "use_inventory_obs": {"value": use_inventory_obs},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 8e-4,
            },
            "n_steps": {"values": [128, 256, 512]},
            "batch_size": {"values": [128, 256, 512]},
            "n_epochs": {"values": [4, 8, 10]},
            "gamma": {"values": [0.98, 0.99, 0.995]},
            "gae_lambda": {"values": [0.90, 0.95]},
            "clip_range": {"values": [0.1, 0.2]},
            "target_kl": {"values": [0.02, 0.03, 0.05]},
            "ent_coef": {
                "distribution": "log_uniform_values",
                "min": 2e-4,
                "max": 1e-2,
            },
            "vf_coef": {"values": [0.25, 0.5, 1.0]},
            "max_grad_norm": {"values": [0.5, 1.0]},
            "pi_layers": {"values": ["64,64", "128,128"]},
            "vf_layers": {"values": ["64,64", "128,128", "256,256"]},
            "features_dim": {"values": [128, 256]},
            "bcos_b": {"values": [2.0, 2.1, 2.3]},
            # Fixed architecture and objective fields are logged in every run.
            "actor_type": {"value": "bcos"},
            "critic_type": {"value": "mlp"},
            "critic_activation": {"value": "tanh"},
            "bcos_max_out": {"value": 1},
            "stable_bcos_norm": {"value": True},
            "bias_free_bcos_norm": {"value": True},
            "use_rgb_img_obs": {"value": True},
            "invert_channels": {"value": True},
            "bce_coef": {"value": 0.0},
            "explanation_coef": {"value": 0.0},
        },
    }


def direct_run_defaults(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device": args.device,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "model_save_freq": args.model_save_freq,
        "checkpoint_threshold": args.checkpoint_threshold,
        "checkpoint_interval": args.checkpoint_interval,
        "open_door_reward": args.open_door_reward,
        "pick_key_reward": args.pick_key_reward,
        "shortcut_prob": args.shortcut_prob,
        "use_inventory_obs": args.use_inventory_obs,
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "target_kl": 0.03,
        "ent_coef": 1e-3,
        "vf_coef": 0.5,
        "max_grad_norm": 1.0,
        "pi_layers": "128,128",
        "vf_layers": "128,128",
        "features_dim": 256,
        "bcos_b": 2.0,
        "actor_type": "bcos",
        "critic_type": "mlp",
        "critic_activation": "tanh",
        "bcos_max_out": 1,
        "stable_bcos_norm": True,
        "bias_free_bcos_norm": True,
        "use_rgb_img_obs": True,
        "invert_channels": True,
        "bce_coef": 0.0,
        "explanation_coef": 0.0,
    }


def build_rgb_env(
    *,
    env_id: str,
    n_envs: int,
    seed: int,
    use_subproc: bool,
    shortcut_prob: float,
    open_door_reward: float,
    pick_key_reward: float,
    use_inventory_obs: bool,
):
    vec_env_cls = SubprocVecEnv if use_subproc and n_envs > 1 else None
    env = make_vec_env(
        env_id=env_id,
        env_kwargs={"highlight": False},
        n_envs=n_envs,
        seed=seed,
        wrapper_class=make_wrapper(
            use_rgb_img_obs=True,
            invert_rgb_channels=True,
            use_one_hot_symbolic_obs=False,
            use_inventory_obs=use_inventory_obs,
            shortcut_prob=shortcut_prob,
            open_door_reward=open_door_reward,
            pick_key_reward=pick_key_reward,
            use_sequential_mask=False,
            use_static_multi_mask=False,
            use_signed_action_mask=False,
        ),
        vec_env_cls=vec_env_cls,
    )
    return VecTransposeImage(env)


def validate_config(cfg: dict[str, Any], n_envs: int) -> None:
    fixed_values = {
        "actor_type": "bcos",
        "critic_type": "mlp",
        "stable_bcos_norm": True,
        "bias_free_bcos_norm": True,
        "use_rgb_img_obs": True,
        "invert_channels": True,
        "bce_coef": 0.0,
        "explanation_coef": 0.0,
    }
    for key, expected in fixed_values.items():
        if cfg.get(key) != expected:
            raise ValueError(
                f"Baseline sweep requires {key}={expected!r}, got {cfg.get(key)!r}"
            )

    pi_layers = parse_layer_sizes(str(cfg["pi_layers"]))
    vf_layers = parse_layer_sizes(str(cfg["vf_layers"]))
    validate_layer_sizes(pi_layers, arg_name="pi_layers")
    validate_layer_sizes(vf_layers, arg_name="vf_layers")

    rollout_size = int(cfg["n_steps"]) * n_envs
    batch_size = int(cfg["batch_size"])
    if batch_size > rollout_size or rollout_size % batch_size != 0:
        raise ValueError(
            "batch_size must divide n_steps*n_envs exactly; "
            f"got batch_size={batch_size}, rollout_size={rollout_size}"
        )
    if int(cfg["features_dim"]) <= 0:
        raise ValueError("features_dim must be positive")
    if float(cfg["target_kl"]) <= 0:
        raise ValueError("target_kl must be positive")


def make_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    pi_layers = parse_layer_sizes(str(cfg["pi_layers"]))
    vf_layers = parse_layer_sizes(str(cfg["vf_layers"]))
    features_dim = int(cfg["features_dim"])
    bcos_b = float(cfg["bcos_b"])
    bcos_max_out = int(cfg["bcos_max_out"])

    return {
        "actor_type": "bcos",
        "critic_type": "mlp",
        "net_arch": {"pi": pi_layers, "vf": vf_layers},
        "activation_fn": activation_from_name(str(cfg["critic_activation"])),
        "bcos_b": bcos_b,
        "bcos_max_out": bcos_max_out,
        "stable_bcos_norm": True,
        "share_features_extractor": False,
        "features_extractor_class": GenericCNNExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "use_bcos": True,
            "bcos_b": bcos_b,
            "bcos_max_out": bcos_max_out,
            # [RGB, inverse RGB] already comes from the environment.
            "invert_channels": False,
            "stable_bcos_norm": True,
            "bias_free_bcos_norm": True,
        },
        "critic_features_extractor_class": GenericCNNExtractor,
        "critic_features_extractor_kwargs": {
            "features_dim": features_dim,
            "use_bcos": False,
            "bcos_b": bcos_b,
            "bcos_max_out": bcos_max_out,
            "invert_channels": False,
            "stable_bcos_norm": False,
            "bias_free_bcos_norm": False,
        },
        "normalize_images": True,
    }


def train_one_run(args: argparse.Namespace) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        config=direct_run_defaults(args),
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=False,
        tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()],
    )
    cfg = dict(wandb.config)
    n_envs = int(cfg["n_envs"])
    validate_config(cfg, n_envs)

    run_dir = ROOT / "wandb_runs" / run.id
    tb_dir = run_dir / "tb"
    model_dir = run_dir / "models"
    best_model_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval_logs"
    checkpoint_dir = run_dir / "checkpoints"
    for directory in (model_dir, best_model_dir, eval_log_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    total_timesteps = int(cfg["total_timesteps"])
    train_env = build_rgb_env(
        env_id=str(cfg["env_id"]),
        n_envs=n_envs,
        seed=seed,
        use_subproc=args.use_subproc,
        shortcut_prob=float(cfg["shortcut_prob"]),
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
        use_inventory_obs=bool(cfg["use_inventory_obs"]),
    )
    eval_env = build_rgb_env(
        env_id=str(cfg["env_id"]),
        n_envs=1,
        seed=seed + 10_000,
        use_subproc=False,
        shortcut_prob=float(cfg["shortcut_prob"]),
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
        use_inventory_obs=bool(cfg["use_inventory_obs"]),
    )

    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=train_env,
        # Hard-coded to prevent sweep configuration mistakes from enabling aux.
        bce_coef=0.0,
        explanation_coef=0.0,
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
        tensorboard_log=str(tb_dir),
        policy_kwargs=make_policy_kwargs(cfg),
        seed=seed,
        device=str(cfg["device"]),
        verbose=1,
    )

    if model.bce_coef != 0.0 or model.explanation_coef != 0.0:
        raise RuntimeError("Baseline model unexpectedly enabled an auxiliary loss")

    stats_callback = MiniGridStatsCallback(window_size=100)
    checkpoint_callback = SuccessRateCheckpointCallback(
        save_dir=str(checkpoint_dir),
        threshold=float(cfg["checkpoint_threshold"]),
        interval=int(cfg["checkpoint_interval"]),
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=max(int(cfg["eval_freq"]) // max(n_envs, 1), 1),
        n_eval_episodes=int(cfg["n_eval_episodes"]),
        deterministic=True,
        render=False,
        verbose=1,
    )
    wandb_callback = WandbCallback(
        model_save_path=str(model_dir),
        model_save_freq=max(int(cfg["model_save_freq"]) // max(n_envs, 1), 1),
        gradient_save_freq=0,
        verbose=1,
    )
    objective_callback = BaselineObjectiveCallback(eval_callback)
    callbacks = CallbackList(
        [
            stats_callback,
            checkpoint_callback,
            eval_callback,
            objective_callback,
            wandb_callback,
        ]
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"baseline_{run.id}",
        )
        final_model_path = run_dir / "final_model.zip"
        model.save(str(final_model_path))

        artifact = wandb.Artifact(
            name=f"baseline-model-{run.id}",
            type="model",
            metadata={
                "run_id": run.id,
                "architecture_version": ARCHITECTURE_VERSION,
                "env_id": str(cfg["env_id"]),
                "total_timesteps": total_timesteps,
                "bce_coef": 0.0,
                "explanation_coef": 0.0,
            },
        )
        artifact.add_file(str(final_model_path))
        run.log_artifact(artifact)
    finally:
        train_env.close()
        eval_env.close()
        run.finish()


def create_sweep(args: argparse.Namespace) -> None:
    sweep_id = wandb.sweep(
        sweep=baseline_sweep_config(
            args.seed,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            device=args.device,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            model_save_freq=args.model_save_freq,
            checkpoint_threshold=args.checkpoint_threshold,
            checkpoint_interval=args.checkpoint_interval,
            shortcut_prob=args.shortcut_prob,
            open_door_reward=args.open_door_reward,
            pick_key_reward=args.pick_key_reward,
            use_inventory_obs=args.use_inventory_obs,
        ),
        project=args.project,
        entity=args.entity or None,
    )
    print(f"Created sweep: {sweep_id}")
    print(
        "Start an agent with:\n"
        f"python minigrid_part/sweep_minigrid_PPO.py --project {args.project} "
        f"{f'--entity {args.entity} ' if args.entity else ''}"
        f"--sweep-id {sweep_id}"
        f"{' --use-subproc' if args.use_subproc else ''}"
    )


def run_agent(args: argparse.Namespace) -> None:
    if not args.sweep_id:
        raise ValueError("--sweep-id is required when running an agent")
    wandb.agent(
        sweep_id=args.sweep_id,
        function=lambda: train_one_run(args),
        count=args.count,
        project=args.project,
        entity=args.entity or None,
    )


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W&B sweep for the strict no-aux RGB B-cos PPO baseline"
    )
    parser.add_argument("--project", default="minigrid-ppo-baseline-v1")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-subproc", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--model-save-freq", type=int, default=150_000)
    parser.add_argument("--checkpoint-threshold", type=float, default=0.95)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--shortcut-prob", type=float, default=0.0)
    parser.add_argument("--open-door-reward", type=float, default=0.0)
    parser.add_argument("--pick-key-reward", type=float, default=0.0)
    parser.add_argument("--use-inventory-obs", action="store_true")
    parser.add_argument("--tags", default="minigrid,ppo,bcos,baseline,no-aux")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if args.create_sweep:
        create_sweep(args)
    elif args.sweep_id:
        run_agent(args)
    else:
        train_one_run(args)


if __name__ == "__main__":
    main()

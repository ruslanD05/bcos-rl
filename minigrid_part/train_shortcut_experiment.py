#!/usr/bin/env python3
"""Train the controlled oracle-patch MiniGrid shortcut experiment."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

from stable_baselines3.common.callbacks import CallbackList, EvalCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.explanation_supervision import MaskedRolloutBuffer
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    GenericCNNExtractor,
)
from env_wrappers import MiniGridStatsCallback
from shortcut_experiment_env import make_shortcut_vec_env


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("baseline", "suppressed")


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=(*CONDITIONS, "both"), default="both")
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-correlation", type=float, default=0.9)
    parser.add_argument("--border-height", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument(
        "--mask-profile",
        choices=(
            "sequential",
            "negative_hint_removal",
            "oracle_patch_polarity",
            "patch_only_negative",
        ),
        default="sequential",
    )
    parser.add_argument(
        "--explanation-loss-type",
        choices=(
            "masked_absolute_energy",
            "channelwise_masked_absolute_energy",
            "channelwise_polarity_masked_energy",
            "patch_absolute_energy",
        ),
        default="masked_absolute_energy",
    )
    parser.add_argument("--explanation-coef", type=float, default=0.15)
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1)
    parser.add_argument(
        "--source-model",
        type=Path,
        default=None,
        help="Path to an initial/baseline checkpoint to finetune from.",
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--bcos-b", type=float, default=2.0)
    parser.add_argument("--pi-layers", default="128,128")
    parser.add_argument("--vf-layers", default="128,128")

    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--n-eval-episodes", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "shortcut_experiment",
    )
    parser.add_argument("--run-name", default="oracle_patch")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or any(width <= 0 for width in layers):
        raise ValueError("network layer sizes must be positive")
    return layers


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0 or args.n_envs <= 0:
        raise ValueError("timesteps and n_envs must be positive")
    if not 0 <= args.training_correlation <= 1:
        raise ValueError("--training-correlation must be in [0, 1]")
    if args.explanation_coef <= 0 or args.explanation_max_grad_ratio <= 0:
        raise ValueError("explanation coefficient and gradient ratio must be positive")
    rollout = args.n_steps * args.n_envs
    if args.batch_size > rollout or rollout % args.batch_size:
        raise ValueError("--batch-size must divide --n-steps * --n-envs")
    parse_layers(args.pi_layers)
    parse_layers(args.vf_layers)


def policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "features_dim": args.features_dim,
        "bcos_b": args.bcos_b,
        "bcos_max_out": 1,
        "invert_channels": False,
        "bias_free_bcos_norm": True,
    }
    return {
        "actor_type": "bcos",
        "critic_type": "mlp",
        "net_arch": {
            "pi": parse_layers(args.pi_layers),
            "vf": parse_layers(args.vf_layers),
        },
        "bcos_b": args.bcos_b,
        "bcos_max_out": 1,
        "stable_bcos_norm": True,
        "share_features_extractor": False,
        "features_extractor_class": GenericCNNExtractor,
        "features_extractor_kwargs": {
            **common,
            "use_bcos": True,
            "stable_bcos_norm": True,
        },
        "critic_features_extractor_class": GenericCNNExtractor,
        "critic_features_extractor_kwargs": {
            **common,
            "use_bcos": False,
            "stable_bcos_norm": False,
        },
        "normalize_images": True,
    }


def make_env(args: argparse.Namespace, *, n_envs: int, seed: int, training: bool):
    return make_shortcut_vec_env(
        env_id=args.env_id,
        n_envs=n_envs,
        seed=seed,
        mode="correlated",
        correlation_probability=args.training_correlation if training else 1.0,
        border_height=args.border_height,
        patch_size=args.patch_size,
        mask_profile=getattr(args, "mask_profile", "sequential"),
    )


def train_condition(
    args: argparse.Namespace,
    condition: str,
    run_dir: Path,
) -> Path:
    condition_dir = run_dir / condition
    best_dir = condition_dir / "best_model"
    eval_dir = condition_dir / "eval"
    for directory in (condition_dir, best_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = make_env(args, n_envs=args.n_envs, seed=args.seed, training=True)
    eval_env = make_env(
        args,
        n_envs=1,
        seed=args.seed + 10_000,
        training=False,
    )
    explanation_coef = args.explanation_coef if condition == "suppressed" else 0.0
    try:
        if args.source_model is not None and (
            condition == "suppressed" or args.condition == condition
        ):
            print(f"Loading initial model checkpoint from {args.source_model}...")
            model = BcosPPO.load(
                str(args.source_model),
                env=env,
                device=args.device,
                custom_objects={
                    "policy_class": ConfigurableActorCriticPolicy,
                    "lr_schedule": lambda _: args.learning_rate,
                },
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                confidence_coef=0.0,
                explanation_coef=explanation_coef,
                explanation_loss_type=args.explanation_loss_type,
                explanation_max_grad_ratio=args.explanation_max_grad_ratio,
                gradient_diagnostics=condition == "suppressed",
                tensorboard_log=str(run_dir / "tensorboard"),
                seed=args.seed,
            )
        else:
            model = BcosPPO(
                policy=ConfigurableActorCriticPolicy,
                env=env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                confidence_coef=0.0,
                explanation_coef=explanation_coef,
                explanation_loss_type=args.explanation_loss_type,
                explanation_max_grad_ratio=args.explanation_max_grad_ratio,
                gradient_diagnostics=condition == "suppressed",
                policy_kwargs=policy_kwargs(args),
                tensorboard_log=str(run_dir / "tensorboard"),
                seed=args.seed,
                device=args.device,
                verbose=1,
            )
        if condition == "suppressed" and not isinstance(
            model.rollout_buffer, MaskedRolloutBuffer
        ):
            raise RuntimeError("Suppressed condition did not create a masked buffer")

        callbacks = CallbackList(
            [
                MiniGridStatsCallback(window_size=100),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(eval_dir),
                    eval_freq=max(args.eval_freq // args.n_envs, 1),
                    n_eval_episodes=args.n_eval_episodes,
                    deterministic=True,
                    render=False,
                    verbose=1,
                ),
            ]
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=condition,
        )
        output = condition_dir / "final_model.zip"
        model.save(str(output))
        return output
    finally:
        env.close()
        eval_env.close()


def main() -> None:
    args = make_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.run_name}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "output_dir": str(args.output_dir.resolve())}
    config.update(
        {
            "positive_region": (
                "all non-patch pixels"
                if args.mask_profile in (
                    "negative_hint_removal",
                    "oracle_patch_polarity",
                )
                else "rendered agent and current task-object silhouettes"
            ),
            "negative_region": "artificial oracle-action patch only",
            "neutral_region": (
                "none"
                if args.mask_profile in (
                    "negative_hint_removal",
                    "oracle_patch_polarity",
                )
                else "all remaining pixels"
            ),
            "suppression_objective": args.explanation_loss_type,
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str),
        encoding="utf-8",
    )

    conditions = CONDITIONS if args.condition == "both" else (args.condition,)
    for condition in conditions:
        print(f"Training shortcut condition: {condition}")
        path = train_condition(args, condition, run_dir)
        print(f"Saved {condition}: {path}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

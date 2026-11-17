#!/usr/bin/env python3
"""Train matched standard or B-cos discrete PPO on CarRacing-v3."""
from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    GenericCNNExtractor,
    activation_from_name,
    parse_layer_sizes,
    validate_layer_sizes,
)
from car_racing_part.explanation_masks import (
    CarRacingExplanationMaskWrapper,
    CarRacingMaskConfig,
)


ROOT = Path(__file__).resolve().parent
ACTION_NAMES = ("noop", "steer_right", "steer_left", "gas", "brake")


class DiscreteActionDiagnosticsCallback(BaseCallback):
    """Log the action mixture over each complete rollout."""

    def __init__(self) -> None:
        super().__init__()
        self.counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        self.total = 0

    def _on_rollout_start(self) -> None:
        self.counts.fill(0)
        self.total = 0

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is None:
            return True
        actions = np.asarray(actions).reshape(-1).astype(np.int64)
        self.counts += np.bincount(actions, minlength=len(ACTION_NAMES))
        self.total += len(actions)
        return True

    def _on_rollout_end(self) -> None:
        if self.total == 0:
            return
        for index, name in enumerate(ACTION_NAMES):
            self.logger.record(
                f"actions/{name}_rate",
                float(self.counts[index] / self.total),
            )


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-type",
        choices=("standard", "bcos"),
        required=True,
        help="Standard CNN/MLP actor or B-cos CNN/MLP/categorical head.",
    )
    parser.add_argument("--env-id", default="CarRacing-v3")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--pi-layers", default="256,256")
    parser.add_argument("--vf-layers", default="256,256")
    parser.add_argument(
        "--activation",
        choices=("tanh", "relu", "elu", "leaky_relu"),
        default="tanh",
    )
    parser.add_argument("--bcos-b", type=float, default=2.0)
    parser.add_argument("--bcos-max-out", type=int, default=1)
    parser.add_argument(
        "--stable-bcos-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bcos-complement-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use [RGB, 1-RGB] in the B-cos actor by default. The actual "
            "environment/model observation remains one RGB frame."
        ),
    )

    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--explanation-coef",
        type=float,
        default=0.0,
        help="Weight of the visual explanation loss; requires --policy-type bcos.",
    )
    parser.add_argument(
        "--explanation-loss-type",
        choices=("masked_absolute_energy", "absolute_energy", "partial_energy"),
        default="masked_absolute_energy",
    )
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1)
    parser.add_argument(
        "--mask-profile",
        choices=("boundary", "boundary-dashboard", "road-positive"),
        default="boundary",
    )
    parser.add_argument("--mask-warmup-frames", type=int, default=0)
    parser.add_argument("--mask-boundary-radius", type=int, default=3)
    parser.add_argument("--mask-grass-as-negative", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0 or args.n_envs <= 0:
        raise ValueError("Timesteps and environment count must be positive")
    if args.n_steps <= 0 or args.batch_size <= 0 or args.n_epochs <= 0:
        raise ValueError("PPO rollout and optimization sizes must be positive")
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size or rollout_size % args.batch_size:
        raise ValueError("--batch-size must divide --n-steps * --n-envs")
    if args.eval_freq <= 0 or args.checkpoint_freq <= 0:
        raise ValueError("Evaluation and checkpoint frequencies must be positive")
    if args.n_eval_episodes <= 0:
        raise ValueError("--n-eval-episodes must be positive")
    validate_layer_sizes(parse_layer_sizes(args.pi_layers), arg_name="--pi-layers")
    validate_layer_sizes(parse_layer_sizes(args.vf_layers), arg_name="--vf-layers")
    if args.explanation_coef < 0 or args.explanation_max_grad_ratio <= 0:
        raise ValueError("Explanation coefficient must be non-negative and gradient ratio positive")
    if args.explanation_coef > 0 and args.policy_type != "bcos":
        raise ValueError("Explanation supervision requires --policy-type bcos")
    if args.mask_warmup_frames < 0 or args.mask_boundary_radius < 0:
        raise ValueError("Mask warmup and boundary radius must be non-negative")


def build_env(
    *,
    env_id: str,
    n_envs: int,
    seed: int,
    use_subprocesses: bool,
    wrapper_class=None,
):
    vec_env_class = SubprocVecEnv if use_subprocesses and n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"continuous": False, "domain_randomize": False},
        wrapper_class=wrapper_class,
        vec_env_cls=vec_env_class,
    )
    return VecTransposeImage(env)


def make_policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    use_bcos = args.policy_type == "bcos"
    common = {
        "features_dim": args.features_dim,
        "bcos_b": args.bcos_b,
        "bcos_max_out": args.bcos_max_out,
    }
    return {
        "actor_type": "bcos" if use_bcos else "mlp",
        "critic_type": "mlp",
        "net_arch": {
            "pi": parse_layer_sizes(args.pi_layers),
            "vf": parse_layer_sizes(args.vf_layers),
        },
        "activation_fn": activation_from_name(args.activation),
        "bcos_b": args.bcos_b,
        "bcos_max_out": args.bcos_max_out,
        "stable_bcos_norm": bool(use_bcos and args.stable_bcos_norm),
        "share_features_extractor": False,
        "features_extractor_class": GenericCNNExtractor,
        "features_extractor_kwargs": {
            **common,
            "use_bcos": use_bcos,
            "invert_channels": bool(use_bcos and args.bcos_complement_input),
            "stable_bcos_norm": bool(use_bcos and args.stable_bcos_norm),
            "bias_free_bcos_norm": use_bcos,
        },
        "critic_features_extractor_class": GenericCNNExtractor,
        "critic_features_extractor_kwargs": {
            **common,
            "use_bcos": False,
            "invert_channels": False,
            "stable_bcos_norm": False,
            "bias_free_bcos_norm": False,
        },
        "normalize_images": True,
    }


def main() -> None:
    args = make_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"carracing_{args.policy_type}_ppo"
    run_dir = args.output_dir / f"{run_name}_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    eval_dir = run_dir / "eval"
    for directory in (checkpoint_dir, best_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask_config = CarRacingMaskConfig(
        profile=args.mask_profile,
        warmup_frames=args.mask_warmup_frames,
        boundary_radius=args.mask_boundary_radius,
        grass_as_negative=args.mask_grass_as_negative,
    )
    mask_wrapper = (
        partial(CarRacingExplanationMaskWrapper, config=mask_config)
        if args.explanation_coef > 0
        else None
    )
    train_env = build_env(
        env_id=args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        use_subprocesses=args.vec_env == "subproc",
        wrapper_class=mask_wrapper,
    )
    eval_env = build_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed + 10_000,
        use_subprocesses=False,
    )
    try:
        algorithm_class = BcosPPO if args.policy_type == "bcos" else PPO
        auxiliary_kwargs: dict[str, Any] = {}
        if algorithm_class is BcosPPO:
            auxiliary_kwargs = {
                "confidence_coef": 0.0,
                "explanation_coef": args.explanation_coef,
                "explanation_loss_type": args.explanation_loss_type,
                "explanation_max_grad_ratio": args.explanation_max_grad_ratio,
            }
        model = algorithm_class(
            policy=ConfigurableActorCriticPolicy,
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=make_policy_kwargs(args),
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
            **auxiliary_kwargs,
        )
        if model.observation_space.shape != (3, 96, 96):
            raise RuntimeError(
                "Expected one channels-first RGB frame, got "
                f"{model.observation_space.shape}"
            )
        if getattr(model.action_space, "n", None) != len(ACTION_NAMES):
            raise RuntimeError(
                f"Expected five discrete actions, got {model.action_space}"
            )

        config = {
            **vars(args),
            "output_dir": str(args.output_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
            "observation": "single RGB frame, uint8, CHW 3x96x96",
            "actor_input_channels": (
                6 if args.policy_type == "bcos" and args.bcos_complement_input else 3
            ),
            "critic_input_channels": 3,
            "frame_stack": 1,
            "action_space": {
                str(index): name for index, name in enumerate(ACTION_NAMES)
            },
            "actor_cnn": "B-cos" if args.policy_type == "bcos" else "standard",
            "actor_mlp": "B-cos" if args.policy_type == "bcos" else "standard",
            "actor_action_head": (
                "B-cos categorical"
                if args.policy_type == "bcos"
                else "standard categorical"
            ),
            "critic": "standard CNN plus MLP",
            "auxiliary_losses": args.explanation_coef > 0,
            "explanation_coef": args.explanation_coef,
            "explanation_loss_type": (
                args.explanation_loss_type if args.explanation_coef > 0 else None
            ),
            "explanation_max_grad_ratio": args.explanation_max_grad_ratio,
            "mask": (
                "P=car+road-boundaries+full-dashboard; N=road+grass; D=other pixels"
                if args.explanation_coef > 0 and args.mask_profile == "boundary-dashboard"
                else (
                    "P=car+road-boundaries+active-dashboard; N=road; D=other pixels"
                    if args.explanation_coef > 0 and args.mask_profile == "boundary"
                    else (
                        "P=car+road; N=grass; D=dashboard+curbs+uncertain"
                        if args.explanation_coef > 0
                        else None
                    )
                )
            ),
            "mask_warmup_frames": args.mask_warmup_frames,
            "mask_boundary_radius": args.mask_boundary_radius,
            "domain_randomization": False,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )

        callbacks = CallbackList(
            [
                DiscreteActionDiagnosticsCallback(),
                CheckpointCallback(
                    save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                    save_path=str(checkpoint_dir),
                    name_prefix=f"{args.policy_type}_ppo",
                ),
                EvalCallback(
                    eval_env=eval_env,
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

        print(f"Run directory: {run_dir.resolve()}")
        print(f"Policy type: {args.policy_type}")
        print(f"Observation space: {model.observation_space}")
        print(f"Action space: {model.action_space} ({', '.join(ACTION_NAMES)})")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
        )
        final_path = run_dir / f"final_{args.policy_type}_ppo.zip"
        model.save(str(final_path))
        print(f"Saved final model: {final_path.resolve()}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train B-cos PPO from scratch with scheduled explanation supervision."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

from env_wrappers import MiniGridStatsCallback
from train_minigrid_ActorCritic import build_vec_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.explanation_supervision import MaskedRolloutBuffer
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    GenericCNNExtractor,
    activation_from_name,
    parse_layer_sizes,
    validate_layer_sizes,
)


ROOT = Path(__file__).resolve().parent


class ExplanationCurriculumCallback(BaseCallback):
    """Schedule explanation strength from time or rolling training success."""

    def __init__(
        self,
        *,
        mode: str,
        start_coef: float,
        end_coef: float,
        ramp_timesteps: int,
        checkpoint_dir: Path,
        checkpoint_freq: int,
        success_threshold: float = 0.5,
        success_window: int = 100,
        min_success_episodes: int = 100,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        if mode not in {"control", "linear", "success-gated"}:
            raise ValueError(f"Unsupported curriculum mode: {mode!r}")
        if not 0 <= start_coef <= end_coef:
            raise ValueError("Expected 0 <= start_coef <= end_coef")
        if ramp_timesteps <= 0:
            raise ValueError("ramp_timesteps must be positive")
        if checkpoint_freq <= 0:
            raise ValueError("checkpoint_freq must be positive")
        if not 0 <= success_threshold <= 1:
            raise ValueError("success_threshold must be in [0, 1]")
        if success_window <= 0 or min_success_episodes <= 0:
            raise ValueError("Success window sizes must be positive")
        if min_success_episodes > success_window:
            raise ValueError("min_success_episodes cannot exceed success_window")

        self.mode = mode
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.ramp_timesteps = ramp_timesteps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.success_threshold = success_threshold
        self.min_success_episodes = min_success_episodes
        self.success_history: collections.deque[float] = collections.deque(
            maxlen=success_window
        )
        self.gate_timestep: int | None = None
        self.next_checkpoint = checkpoint_freq

    def _coefficient(self, timestep: int) -> tuple[float, float]:
        if self.mode == "control":
            return 0.0, 0.0
        if self.mode == "success-gated":
            if self.gate_timestep is None:
                return self.start_coef, 0.0
            elapsed = max(0, timestep - self.gate_timestep)
        else:
            elapsed = max(0, timestep)
        progress = min(1.0, elapsed / self.ramp_timesteps)
        coefficient = self.start_coef + (
            self.end_coef - self.start_coef
        ) * progress
        return coefficient, progress

    def _save(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        self.model.save(str(path))
        if self.verbose:
            print(f"[Curriculum] Saved checkpoint: {path}")

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        coefficient, _ = self._coefficient(self.model.num_timesteps)
        self.model.explanation_coef = coefficient

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", []),
        ):
            if done and "is_success" in info:
                self.success_history.append(float(info["is_success"]))

        rolling_success = (
            float(np.mean(self.success_history)) if self.success_history else 0.0
        )
        if (
            self.mode == "success-gated"
            and self.gate_timestep is None
            and len(self.success_history) >= self.min_success_episodes
            and rolling_success >= self.success_threshold
        ):
            self.gate_timestep = self.model.num_timesteps
            self._save(
                f"before_explanation_gate_step_{self.gate_timestep:07d}.zip"
            )
            if self.verbose:
                print(
                    "[Curriculum] Success gate opened at "
                    f"{self.gate_timestep:,} steps "
                    f"(rolling success={rolling_success:.3f})"
                )

        coefficient, progress = self._coefficient(self.model.num_timesteps)
        self.model.explanation_coef = coefficient
        self.logger.record("curriculum/explanation_coef", coefficient)
        self.logger.record("curriculum/ramp_progress", progress)
        self.logger.record("curriculum/rolling_success_rate", rolling_success)
        self.logger.record(
            "curriculum/success_episodes",
            len(self.success_history),
        )
        self.logger.record(
            "curriculum/gate_triggered",
            float(self.mode == "linear" or self.gate_timestep is not None),
        )
        self.logger.record(
            "curriculum/gate_timestep",
            self.gate_timestep if self.gate_timestep is not None else -1,
        )

        while self.model.num_timesteps >= self.next_checkpoint:
            self._save(f"model_step_{self.next_checkpoint:07d}.zip")
            self.next_checkpoint += self.checkpoint_freq
        return True


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--total-timesteps", type=int, default=655_360)
    parser.add_argument(
        "--curriculum",
        choices=("control", "linear", "success-gated"),
        default="linear",
    )
    parser.add_argument("--start-explanation-coef", type=float, default=0.0)
    parser.add_argument("--explanation-coef", type=float, default=0.15)
    parser.add_argument(
        "--ramp-timesteps",
        type=int,
        help=(
            "Ramp duration. Defaults to the full run for linear mode and "
            "151,552 steps after activation for success-gated mode."
        ),
    )
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--success-window", type=int, default=100)
    parser.add_argument("--min-success-episodes", type=int, default=100)

    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument(
        "--six-channel-obs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--grid-lines-as-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--learning-rate", type=float, default=0.0006141598118159885)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--ent-coef", type=float, default=0.0009960268117654178)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--pi-layers", default="128,128")
    parser.add_argument("--vf-layers", default="128,128")
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--bcos-b", type=float, default=2.3)
    parser.add_argument("--bcos-max-out", type=int, default=1)
    parser.add_argument(
        "--stable-bcos-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "explanation_curriculum",
    )
    parser.add_argument("--run-name", default="bcos_ppo_explanation_curriculum")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if (
        args.curriculum != "control"
        and not 0 <= args.start_explanation_coef <= args.explanation_coef
    ):
        raise ValueError(
            "Expected 0 <= --start-explanation-coef <= --explanation-coef"
        )
    if args.curriculum != "control" and args.explanation_coef <= 0:
        raise ValueError("--explanation-coef must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.eval_freq <= 0 or args.checkpoint_freq <= 0:
        raise ValueError("Evaluation and checkpoint frequencies must be positive")
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size or rollout_size % args.batch_size:
        raise ValueError("--batch-size must divide --n-steps * --n-envs")
    validate_layer_sizes(
        parse_layer_sizes(args.pi_layers),
        arg_name="--pi-layers",
    )
    validate_layer_sizes(
        parse_layer_sizes(args.vf_layers),
        arg_name="--vf-layers",
    )


def resolve_ramp_timesteps(args: argparse.Namespace) -> int:
    if args.ramp_timesteps is not None:
        if args.ramp_timesteps <= 0:
            raise ValueError("--ramp-timesteps must be positive")
        return args.ramp_timesteps
    if args.curriculum in {"control", "linear"}:
        return args.total_timesteps
    return 151_552


def make_policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    extractor_kwargs = {
        "features_dim": args.features_dim,
        "bcos_b": args.bcos_b,
        "bcos_max_out": args.bcos_max_out,
        "invert_channels": False,
        "bias_free_bcos_norm": True,
    }
    return {
        "actor_type": "bcos",
        "critic_type": "mlp",
        "net_arch": {
            "pi": parse_layer_sizes(args.pi_layers),
            "vf": parse_layer_sizes(args.vf_layers),
        },
        "activation_fn": activation_from_name("tanh"),
        "bcos_b": args.bcos_b,
        "bcos_max_out": args.bcos_max_out,
        "stable_bcos_norm": args.stable_bcos_norm,
        "share_features_extractor": False,
        "features_extractor_class": GenericCNNExtractor,
        "features_extractor_kwargs": {
            **extractor_kwargs,
            "use_bcos": True,
            "stable_bcos_norm": args.stable_bcos_norm,
        },
        "critic_features_extractor_class": GenericCNNExtractor,
        "critic_features_extractor_kwargs": {
            **extractor_kwargs,
            "use_bcos": False,
            "stable_bcos_norm": False,
        },
        "normalize_images": True,
    }


def make_env(
    args: argparse.Namespace,
    *,
    n_envs: int,
    seed: int,
    with_explanation_mask: bool,
):
    return build_vec_env(
        env_id=args.env_id,
        n_envs=n_envs,
        seed=seed,
        use_rgb_img_obs=True,
        invert_channels=args.six_channel_obs,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=with_explanation_mask,
        sequential_walls_as_negative=with_explanation_mask,
        sequential_grid_lines_as_negative=(
            with_explanation_mask and args.grid_lines_as_negative
        ),
    )


def main() -> None:
    args = make_args()
    validate_args(args)
    ramp_timesteps = resolve_ramp_timesteps(args)
    uses_explanation = args.curriculum != "control"
    effective_start_coef = (
        args.start_explanation_coef if uses_explanation else 0.0
    )
    effective_end_coef = args.explanation_coef if uses_explanation else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.run_name}_{args.curriculum}_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    best_dir = run_dir / "best_model"
    for directory in (run_dir, checkpoint_dir, eval_dir, best_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["resolved_ramp_timesteps"] = ramp_timesteps
    config["effective_start_explanation_coef"] = effective_start_coef
    config["effective_end_explanation_coef"] = effective_end_coef
    config["objective"] = (
        "SB3 PPO only"
        if not uses_explanation
        else "SB3 PPO plus scheduled partial masked energy"
    )
    if uses_explanation:
        config["positive_region"] = "agent plus active sequential task object"
        config["negative_region"] = (
            "walls plus grid lines" if args.grid_lines_as_negative else "walls"
        )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    train_env = make_env(
        args,
        n_envs=args.n_envs,
        seed=args.seed,
        with_explanation_mask=uses_explanation,
    )
    eval_env = make_env(
        args,
        n_envs=1,
        seed=args.seed + 10_000,
        with_explanation_mask=uses_explanation,
    )
    try:
        # Explanation runs need a positive constructor value to create the
        # MaskedRolloutBuffer. The callback sets the effective coefficient
        # before the first rollout. Control uses the original SB3 buffer.
        model = BcosPPO(
            policy=ConfigurableActorCriticPolicy,
            env=train_env,
            bce_coef=0.0,
            explanation_coef=effective_end_coef,
            explanation_loss_type="partial_energy",
            partial_energy_alignment_margin=0.5,
            sequential_explanation_coef=0.0,
            action_explanation_coef=0.0,
            gradient_diagnostics=True,
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
            tensorboard_log=str(run_dir / "tensorboard"),
            policy_kwargs=make_policy_kwargs(args),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        if uses_explanation and not isinstance(
            model.rollout_buffer,
            MaskedRolloutBuffer,
        ):
            raise RuntimeError("Curriculum requires MaskedRolloutBuffer")
        if not uses_explanation and isinstance(
            model.rollout_buffer,
            MaskedRolloutBuffer,
        ):
            raise RuntimeError("Control unexpectedly constructed MaskedRolloutBuffer")
        model.explanation_coef = effective_start_coef
        model.save(str(run_dir / "initial_untrained.zip"))

        curriculum_callback = ExplanationCurriculumCallback(
            mode=args.curriculum,
            start_coef=effective_start_coef,
            end_coef=effective_end_coef,
            ramp_timesteps=ramp_timesteps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=args.checkpoint_freq,
            success_threshold=args.success_threshold,
            success_window=args.success_window,
            min_success_episodes=args.min_success_episodes,
        )
        callbacks = CallbackList(
            [
                MiniGridStatsCallback(window_size=100),
                curriculum_callback,
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

        print(
            f"Training {args.curriculum} curriculum for "
            f"{args.total_timesteps:,} steps"
        )
        if args.curriculum == "control":
            print("Control objective: pure SB3 PPO with no auxiliary losses")
        elif args.curriculum == "success-gated":
            print(
                "Explanation gate: rolling success >= "
                f"{args.success_threshold:.2f} over "
                f"{args.min_success_episodes}-{args.success_window} episodes"
            )
        if uses_explanation:
            print(
                f"Explanation coefficient: {effective_start_coef:g} -> "
                f"{effective_end_coef:g} over {ramp_timesteps:,} ramp steps"
            )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=f"curriculum_{args.curriculum}",
        )
        final_path = run_dir / "final_model.zip"
        model.save(str(final_path))
    finally:
        train_env.close()
        eval_env.close()

    print(f"Final model: {final_path}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

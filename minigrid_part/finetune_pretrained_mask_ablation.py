#!/usr/bin/env python3
"""Fine-tune a pretrained B-cos PPO model with a selected mask ablation."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.explanation_supervision import (
    DualMaskedRolloutBuffer,
    MaskedRolloutBuffer,
)
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from env_wrappers import MiniGridStatsCallback
from train_minigrid_ActorCritic import build_vec_env


ROOT = Path(__file__).resolve().parent
MASK_MODES = (
    "sequential-correct-masked",
    "sequential-all-negative",
    "sequential-wall-grid-legacy",
    "flipped-sequential",
    "flipped-sequential-legacy",
    "action-forward-neutral",
    "action-category-absolute",
    "local-action-masked",
    "local-directional-masked",
    "local-neighborhood-legacy",
    "dual-polarity-blockers",
)


class AdditionalStepsCheckpointCallback(BaseCallback):
    """Save checkpoints at fixed thresholds after fine-tuning starts."""

    def __init__(self, save_dir: Path, interval: int, mask_mode: str) -> None:
        super().__init__(verbose=1)
        self.save_dir = save_dir
        self.interval = interval
        self.mask_mode = mask_mode
        self.start_timesteps = 0
        self.next_checkpoint = interval

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.start_timesteps = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        elapsed = int(self.model.num_timesteps) - self.start_timesteps
        while elapsed >= self.next_checkpoint:
            path = self.save_dir / (
                f"{self.mask_mode}_after_{self.next_checkpoint:07d}_steps.zip"
            )
            self.model.save(str(path))
            print(
                f"Saved checkpoint after {elapsed:,} actual additional steps: {path}"
            )
            self.next_checkpoint += self.interval
        return True


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mask-mode", choices=MASK_MODES, required=True)
    parser.add_argument("--additional-timesteps", type=int, default=150_000)
    parser.add_argument("--explanation-coef", type=float, default=0.15)
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1)
    parser.add_argument(
        "--negative-explanation-coef",
        type=float,
        help="Negative blocker coefficient; defaults to --explanation-coef.",
    )
    parser.add_argument("--sequential-margin", type=float, default=0.001)
    parser.add_argument("--negative-margin", type=float, default=0.01)
    parser.add_argument(
        "--channelwise-absolute-energy",
        action="store_true",
        help=(
            "Use sum_c |E_c| before applying the masked P/N energy ratio, "
            "preventing cancellation between RGB6 channels."
        ),
    )
    parser.add_argument(
        "--raw-action-explanation",
        action="store_true",
        help=(
            "Supervise the raw selected-action logit explanation instead of "
            "subtracting the mean action logit before attribution."
        ),
    )
    parser.add_argument("--movement-radius", type=int, default=1)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument(
        "--grid-lines-as-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include grid lines in the sequential mask before flipping it.",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument(
        "--eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax actions for evaluation; stochastic evaluation is default.",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "pretrained_mask_ablation",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.additional_timesteps <= 0:
        raise ValueError("--additional-timesteps must be positive")
    if args.explanation_coef <= 0:
        raise ValueError("--explanation-coef must be positive")
    if args.explanation_max_grad_ratio <= 0:
        raise ValueError("--explanation-max-grad-ratio must be positive")
    if args.negative_explanation_coef is not None and args.negative_explanation_coef <= 0:
        raise ValueError("--negative-explanation-coef must be positive")
    if args.sequential_margin <= 0 or args.negative_margin <= 0:
        raise ValueError("--sequential-margin and --negative-margin must be positive")
    if args.movement_radius < 1:
        raise ValueError("--movement-radius must be at least 1")
    if args.n_envs <= 0 or args.n_eval_episodes <= 0:
        raise ValueError("--n-envs and --n-eval-episodes must be positive")
    if args.eval_freq <= 0 or args.checkpoint_freq <= 0:
        raise ValueError("--eval-freq and --checkpoint-freq must be positive")


def inspect_checkpoint(path: Path, device: str) -> tuple[int, int]:
    model = PPO.load(
        str(path),
        env=None,
        device=device,
        custom_objects={
            "policy_class": ConfigurableActorCriticPolicy,
            "lr_schedule": lambda _: 0.0,
        },
    )
    shape = tuple(model.observation_space.shape)
    if len(shape) != 3 or shape[0] not in (3, 6):
        raise ValueError(
            "Expected a channels-first RGB checkpoint with 3 or 6 channels, "
            f"got {shape}"
        )
    return int(model.num_timesteps), int(shape[0])


def make_env(
    args: argparse.Namespace,
    *,
    n_envs: int,
    seed: int,
    observation_channels: int,
):
    sequential_all_negative = args.mask_mode in {
        "sequential-correct-masked",
        "sequential-all-negative",
    }
    sequential_legacy = args.mask_mode == "sequential-wall-grid-legacy"
    flipped = args.mask_mode == "flipped-sequential"
    flipped_legacy = args.mask_mode == "flipped-sequential-legacy"
    action_neutral = args.mask_mode == "action-forward-neutral"
    action_category = args.mask_mode == "action-category-absolute"
    local_directional = args.mask_mode in {
        "local-action-masked",
        "local-directional-masked",
    }
    local_legacy = args.mask_mode == "local-neighborhood-legacy"
    dual_polarity = args.mask_mode == "dual-polarity-blockers"
    return build_vec_env(
        env_id=args.env_id,
        n_envs=n_envs,
        seed=seed,
        use_rgb_img_obs=True,
        invert_channels=observation_channels == 6,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=sequential_legacy or flipped_legacy,
        use_sequential_all_negative_mask=sequential_all_negative or flipped,
        flip_sequential_mask=flipped or flipped_legacy,
        sequential_walls_as_negative=sequential_legacy or flipped_legacy,
        sequential_grid_lines_as_negative=(
            (sequential_legacy or flipped_legacy) and args.grid_lines_as_negative
        ),
        use_signed_action_mask=action_neutral,
        use_action_category_mask=action_category,
        use_local_action_mask=local_legacy,
        use_local_directional_mask=local_directional,
        use_combined_explanation_masks=dual_polarity,
        combined_blocker_masks_only=dual_polarity,
        local_action_movement_radius=args.movement_radius,
        signed_enable_negative=True,
        signed_enabled_actions=("forward", "pickup", "toggle"),
        action_forward_traversable_positive=False,
    )


def main() -> None:
    args = make_args()
    validate_args(args)
    saved_timesteps, observation_channels = inspect_checkpoint(
        args.model_path, args.device
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or args.mask_mode
    run_dir = args.output_dir / f"{run_name}_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    best_dir = run_dir / "best_model"
    for path in (checkpoint_dir, eval_dir, best_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_env = make_env(
        args,
        n_envs=args.n_envs,
        seed=args.seed,
        observation_channels=observation_channels,
    )
    eval_env = make_env(
        args,
        n_envs=1,
        seed=args.seed + 10_000,
        observation_channels=observation_channels,
    )
    try:
        explanation_loss_type = {
            "action-category-absolute": "absolute_energy",
            "local-action-masked": "masked_absolute_energy",
            "local-directional-masked": "masked_absolute_energy",
            "local-neighborhood-legacy": "masked_absolute_energy",
            "sequential-correct-masked": "masked_absolute_energy",
            "sequential-all-negative": "masked_absolute_energy",
            "flipped-sequential": "masked_absolute_energy",
        }.get(args.mask_mode, "partial_energy")
        if args.channelwise_absolute_energy:
            if explanation_loss_type != "masked_absolute_energy":
                raise ValueError(
                    "--channelwise-absolute-energy requires a masked P/N mask mode"
                )
            explanation_loss_type = "channelwise_masked_absolute_energy"
        dual_polarity = args.mask_mode == "dual-polarity-blockers"
        negative_coef = (
            args.negative_explanation_coef
            if args.negative_explanation_coef is not None
            else args.explanation_coef
        )
        model = BcosPPO.load(
            str(args.model_path),
            env=train_env,
            device=args.device,
            custom_objects={"policy_class": ConfigurableActorCriticPolicy},
            bce_coef=0.0,
            explanation_coef=0.0 if dual_polarity else args.explanation_coef,
            explanation_loss_type=explanation_loss_type,
            explanation_subtract_action_mean=not args.raw_action_explanation,
            partial_energy_alignment_margin=0.5,
            sequential_explanation_coef=args.explanation_coef if dual_polarity else 0.0,
            action_explanation_coef=negative_coef if dual_polarity else 0.0,
            sequential_margin=args.sequential_margin,
            signed_margin=args.negative_margin,
            signed_pos_weight=0.0 if dual_polarity else 1.0,
            signed_neg_weight=1.0,
            dual_masks_exclusive=dual_polarity,
            explanation_max_grad_ratio=args.explanation_max_grad_ratio,
            gradient_diagnostics=True,
        )
        expected_buffer = DualMaskedRolloutBuffer if dual_polarity else MaskedRolloutBuffer
        if not isinstance(model.rollout_buffer, expected_buffer):
            raise RuntimeError("Fine-tuning did not construct a masked rollout buffer")
        model.tensorboard_log = str(run_dir / "tensorboard")

        rollout_size = model.n_steps * args.n_envs
        actual_additional = (
            math.ceil(args.additional_timesteps / rollout_size) * rollout_size
        )
        config = {
            **vars(args),
            "model_path": str(args.model_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "saved_timesteps": saved_timesteps,
            "observation_channels": observation_channels,
            "rollout_size": rollout_size,
            "actual_additional_timesteps": actual_additional,
            "actual_final_timesteps": saved_timesteps + actual_additional,
            "explanation_target": (
                "raw_selected_action_logit"
                if args.raw_action_explanation
                else "selected_action_logit_minus_action_mean"
            ),
            "channel_polarity_aggregation": (
                "positive=sum_c(max(E_c,0)); negative=sum_c(max(-E_c,0))"
                if dual_polarity
                else None
            ),
            "objective": (
                "PPO plus channel-polarity-separated positive sequential and negative invalid-action margins"
                if dual_polarity
                else "PPO plus local action-conditioned P/N absolute-energy localization"
                if args.mask_mode in {"local-action-masked", "local-directional-masked"}
                else (
                    "PPO plus sequential P/N absolute-energy localization"
                    if args.mask_mode in {"sequential-correct-masked", "sequential-all-negative"}
                    else (
                        "PPO plus unsigned action-conditioned absolute-energy localization"
                        if args.mask_mode == "action-category-absolute"
                        else "PPO plus partial masked energy"
                    )
                )
            ),
            "forward_traversable_target": (
                0 if args.mask_mode == "action-forward-neutral" else None
            ),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str),
            encoding="utf-8",
        )

        callbacks = CallbackList(
            [
                MiniGridStatsCallback(window_size=100),
                AdditionalStepsCheckpointCallback(
                    checkpoint_dir,
                    args.checkpoint_freq,
                    args.mask_mode,
                ),
                EvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(eval_dir),
                    eval_freq=max(args.eval_freq // args.n_envs, 1),
                    n_eval_episodes=args.n_eval_episodes,
                    deterministic=args.eval_deterministic,
                    render=False,
                    verbose=1,
                ),
            ]
        )

        print(f"Source checkpoint: {args.model_path}")
        print(f"Source timesteps: {saved_timesteps:,}")
        print(f"Observation channels: {observation_channels}")
        print(f"Mask mode: {args.mask_mode}")
        print(f"Explanation coefficient: {args.explanation_coef:g}")
        print(
            "Explanation target: "
            f"{'raw selected-action logit' if args.raw_action_explanation else 'relative selected-action logit'}"
        )
        print(f"Requested additional steps: {args.additional_timesteps:,}")
        print(f"Actual complete-rollout steps: {actual_additional:,}")
        print(
            "Evaluation actions: "
            f"{'deterministic' if args.eval_deterministic else 'stochastic'}"
        )

        model.learn(
            total_timesteps=args.additional_timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            tb_log_name=args.mask_mode,
        )
        expected_timesteps = saved_timesteps + actual_additional
        if model.num_timesteps != expected_timesteps:
            raise RuntimeError(
                f"Unexpected final timestep {model.num_timesteps}; "
                f"expected {expected_timesteps}"
            )

        final_path = run_dir / f"final_{args.mask_mode}.zip"
        model.save(str(final_path))
        print(f"Saved final checkpoint: {final_path}")
        print(f"Run directory: {run_dir}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

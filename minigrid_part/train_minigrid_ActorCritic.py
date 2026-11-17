#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import os
import sys
from typing import Any

import gymnasium as gym
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.torch_layers import FlattenExtractor

import pointing_game_env

from env_wrappers import (
    make_wrapper, MiniGridStatsCallback
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bcos_rl_generic.train_model import (
    SaveVecNormalizeOnBest, ConfigurableActorCriticPolicy,
    GenericCNNExtractor, SymbolicGridCNNExtractor, parse_layer_sizes,
    validate_layer_sizes, activation_from_name,
)
from bcos_rl_generic.bcosPPO import (
    ConfidenceCoefficientRampCallback,
    BcosPPO,
    SuccessRateCheckpointCallback,
)

ROOT = Path(__file__).resolve().parent

gym.register(
    id="MiniGrid-Composite-15x15-v0",
    entry_point="pointing_game_env:CompositeGridEnv", 
)



def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on MiniGrid DoorKey with B-cos or MLP actor.")
    parser.add_argument("--env-id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=None, help="Stop PPO epochs early when the approximate KL exceeds this value.")
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--actor-type", type=str, default="bcos", choices=["bcos", "mlp"])
    parser.add_argument("--critic-type", type=str, default="mlp", choices=["bcos", "mlp"])
    parser.add_argument("--pi-layers", type=str, default="128,128")
    parser.add_argument("--vf-layers", type=str, default="128,128")
    parser.add_argument("--critic-activation", type=str, default="tanh", choices=["tanh", "relu", "elu", "leaky_relu"])
    parser.add_argument("--bcos-b", type=float, default=2.0)
    parser.add_argument("--bcos-max-out", type=int, default=1)
    
    parser.add_argument("--use-rgb-img-obs", action="store_true", help="Use fully observable RGB image observations with a CNN feature extractor.")
    parser.add_argument(
        "--invert-channels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Expose [RGB, 1-RGB] as six observation channels. Defaults to on "
            "for an RGB B-cos actor and off otherwise."
        ),
    )
    parser.add_argument(
        "--stable-bcos-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bias-free GroupNorm/RMSNorm in B-cos networks.",
    )
    parser.add_argument("--symbolic-extractor", choices=["cnn", "flat"], default="cnn", help="Feature extractor for one-hot symbolic observations. 'flat' retains the legacy flattened baseline.")
    parser.add_argument("--symbolic-features-dim", type=int, default=128, help="Output size of the symbolic CNN feature extractor.")
    parser.add_argument("--use-inventory-obs", action="store_true", help="Include the agent's inventory in the observation (if the env has an inventory).")

    parser.add_argument("--shortcut-prob", type=float, default=0.0, help="Probability of creating a shortcut hole in the wall each episode (for testing generalization).")
    parser.add_argument("--open-door-reward", type=float, default=0.0, help="Additional reward for opening the door (if the env has a door).")
    parser.add_argument("--pick-key-reward", type=float, default=0.0, help="Additional reward for picking up the key (if the env has a key).")

    parser.add_argument("--confidence-coef-start", "--bce-coef-start", "--aux-alpha-start", dest="confidence_coef_start", type=float, default=0.0, help="Initial coefficient for positive-advantage action-confidence sharpening.")
    parser.add_argument("--confidence-coef-end", "--bce-coef-end", "--aux-alpha-end", dest="confidence_coef_end", type=float, default=0.0, help="Final coefficient for positive-advantage action-confidence sharpening.")
    parser.add_argument("--confidence-coef-schedule", "--bce-coef-schedule", "--aux-schedule", dest="confidence_coef_schedule", type=str, default="linear", choices=["linear", "exponential"], help="Schedule for the confidence coefficient.")
    parser.add_argument("--bce-temperature", "--aux-temp", dest="deprecated_bce_temperature", type=float, default=1.0, help="Deprecated compatibility option; categorical confidence uses the policy probability directly.")

    parser.add_argument("--use-sequential-mask", action="store_true", help="Add ExplanationMaskWrapper to produce phase-based (single active sub-goal) attention masks.")
    parser.add_argument("--sequential-walls-as-negative", action="store_true", help="Add rendered wall pixels as the N region of the sequential P/N/D mask.")
    parser.add_argument("--sequential-grid-lines-as-negative", action="store_true", help="Add rendered grey cell-separator pixels as the N region of the sequential P/N/D mask.")
    parser.add_argument("--use-static-multi-mask", action="store_true", help="Add StaticMultiObjectMaskWrapper to produce a static composite mask over Agent+Door+Key+Goal simultaneously. Mutually exclusive with --use-sequential-mask.")
    parser.add_argument("--use-signed-action-mask", action="store_true", help="Add SignedActionMaskWrapper to produce action-conditioned SIGNED masks in {-1,0,+1}. Use with --explanation-loss-type signed. Mutually exclusive with the other mask flags.")
    parser.add_argument("--use-action-category-mask", action="store_true", help="Use unsigned action-conditioned localization masks for movement and interaction actions.")
    parser.add_argument("--use-local-action-mask", action="store_true", help="Use the local action-affordance P/N mask: local geometry for movement and valid front-cell affordances for interaction.")
    parser.add_argument("--local-action-movement-radius", type=int, default=1, help="Chebyshev radius of the local movement neighborhood in grid cells.")
    parser.add_argument("--use-combined-explanation-masks", action="store_true", help="Emit separate sequential and signed action masks and optimize both explanation objectives.")
    parser.add_argument("--use-literal-combined-mask", action="store_true", help="Compose sequential and action targets into one signed P/N/D mask.")
    parser.add_argument("--explanation-coef", "--expl-alpha", dest="explanation_coef", type=float, default=0.0, help="Coefficient for the explanation-supervision loss. 0 disables it.")
    parser.add_argument("--sequential-explanation-coef", type=float, default=0.0, help="Coefficient for sequential positive-energy supervision in combined-mask mode.")
    parser.add_argument("--action-explanation-coef", type=float, default=0.0, help="Coefficient for signed action-affordance supervision in combined-mask mode.")
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1, help="Maximum explanation-gradient norm as a fraction of the PPO actor-gradient norm.")
    parser.add_argument("--sequential-margin", type=float, default=0.001, help="Required mean RMS-normalized attribution in the active sequential target region.")
    parser.add_argument("--use-original-explanation-energy-loss", "--use-original-energy-loss", dest="use_original_explanation_energy_loss", action="store_true", help="Deprecated compatibility flag; both energy paths use the corrected spatial ratio.")
    parser.add_argument("--explanation-loss-type", "--expl-loss-type", dest="explanation_loss_type", type=str, default="positive_margin", choices=["energy", "partial_energy", "absolute_energy", "masked_absolute_energy", "positive_margin", "signed"], help="Explanation loss family: literal P/N/D energy, unsigned absolute-energy localization, positive sequential margin, legacy energy, or signed action supervision.")
    parser.add_argument("--partial-energy-alignment-margin", type=float, default=0.5, help="Bounded positive-alignment margin used for P-only literal masks.")
    parser.add_argument("--signed-margin", type=float, default=0.01, help="Margin m for signed RMS-normalized attribution: wants S*A >= m per supervised cell.")
    parser.add_argument("--signed-pos-weight", type=float, default=1.0, help="Weight on the hinge for +1 (positively supervised) cells.")
    parser.add_argument("--signed-neg-weight", type=float, default=1.0, help="Weight on the hinge for -1 (negatively supervised) cells.")
    parser.add_argument("--signed-disable-negative", action="store_true", help="Positive-only ablation: the signed mask emits only +1 cells (no -1 supervision).")
    parser.add_argument("--signed-actions", type=str, default="forward,pickup,toggle", help="Comma-separated actions the signed mask supervises (subset of forward,pickup,toggle,left,right,drop,done).")
    parser.add_argument("--signed-neg-on-locked-door", action="store_true", help="Assign -1 to a locked door in front when the agent lacks the matching key (default: neutral).")
    parser.add_argument("--gradient-diagnostics", action="store_true", help="Log separate PPO/explanation actor gradients and per-update actor changes.")

    parser.add_argument("--checkpoint-threshold", type=float, default=0.95, help="Success rate that triggers the first milestone checkpoint.")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000, help="Steps between checkpoints after the milestone is hit.")

    parser.add_argument("--tensorboard-log", type=str, default=str(ROOT / "runs" / "minigrid_ppo"))
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "models"))
    parser.add_argument("--save-prefix", type=str, default="minigrid_ppo")
    parser.add_argument("--log-config", action="store_true")

    return parser.parse_args()


def build_vec_env(
    *, env_id: str, n_envs: int, seed: int, use_rgb_img_obs: bool,
    invert_channels: bool = False,
    wall_appearance: str = "normal", wall_color: str = "yellow",
    use_inventory_obs: bool = False, shortcut_prob: float = 0.0,
    open_door_reward: float = 0.0, pick_key_reward: float = 0.0,
    use_sequential_mask: bool = False, use_static_multi_mask: bool = False,
    use_sequential_all_negative_mask: bool = False,
    flip_sequential_mask: bool = False,
    sequential_walls_as_negative: bool = False,
    sequential_grid_lines_as_negative: bool = False,
    use_signed_action_mask: bool = False, signed_enable_negative: bool = True,
    use_action_category_mask: bool = False,
    use_local_action_mask: bool = False,
    use_local_directional_mask: bool = False,
    local_action_movement_radius: int = 1,
    use_combined_explanation_masks: bool = False,
    combined_blocker_masks_only: bool = False,
    use_literal_combined_mask: bool = False,
    signed_enabled_actions: tuple[str, ...] = ("forward", "pickup", "toggle"),
    signed_neg_on_locked_door_without_key: bool = False,
    action_forward_traversable_positive: bool = True,
):
    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"highlight": False},
        wrapper_class=make_wrapper(
            use_rgb_img_obs=use_rgb_img_obs,
            invert_rgb_channels=invert_channels,
            wall_appearance=wall_appearance,
            wall_color=wall_color,
            use_one_hot_symbolic_obs=True,
            use_inventory_obs=use_inventory_obs,
            shortcut_prob=shortcut_prob,
            open_door_reward=open_door_reward,
            pick_key_reward=pick_key_reward,
            use_sequential_mask=use_sequential_mask,
            use_sequential_all_negative_mask=use_sequential_all_negative_mask,
            flip_sequential_mask=flip_sequential_mask,
            sequential_walls_as_negative=sequential_walls_as_negative,
            sequential_grid_lines_as_negative=sequential_grid_lines_as_negative,
            use_static_multi_mask=use_static_multi_mask,
            use_signed_action_mask=use_signed_action_mask,
            use_action_category_mask=use_action_category_mask,
            use_local_action_mask=use_local_action_mask,
            use_local_directional_mask=use_local_directional_mask,
            local_action_movement_radius=local_action_movement_radius,
            use_combined_explanation_masks=use_combined_explanation_masks,
            combined_blocker_masks_only=combined_blocker_masks_only,
            use_literal_combined_mask=use_literal_combined_mask,
            signed_enable_negative=signed_enable_negative,
            signed_enabled_actions=signed_enabled_actions,
            signed_neg_on_locked_door_without_key=signed_neg_on_locked_door_without_key,
            action_forward_traversable_positive=action_forward_traversable_positive,
        ),
    )
    
    if use_rgb_img_obs:
        env = VecTransposeImage(env)
    return env


def main() -> None:
    args = make_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.invert_channels is None:
        args.invert_channels = args.use_rgb_img_obs and args.actor_type == "bcos"
    if args.invert_channels and not args.use_rgb_img_obs:
        raise ValueError("--invert-channels requires --use-rgb-img-obs")

    pi_layers = parse_layer_sizes(args.pi_layers)
    vf_layers = parse_layer_sizes(args.vf_layers)
    validate_layer_sizes(pi_layers, arg_name="--pi-layers")
    validate_layer_sizes(vf_layers, arg_name="--vf-layers")

    if args.bcos_max_out < 1:
        raise ValueError(f"--bcos-max-out must be >= 1, got {args.bcos_max_out}")
    if args.sequential_explanation_coef < 0 or args.action_explanation_coef < 0:
        raise ValueError("Combined explanation coefficients must be non-negative")
    if args.sequential_explanation_coef > 0 and args.sequential_margin <= 0:
        raise ValueError("--sequential-margin must be positive when enabled")
    mask_flags = [
        args.use_sequential_mask,
        args.use_static_multi_mask,
        args.use_signed_action_mask,
        args.use_action_category_mask,
        args.use_local_action_mask,
        args.use_combined_explanation_masks,
        args.use_literal_combined_mask,
    ]
    if sum(mask_flags) > 1:
        raise ValueError("Explanation mask modes are mutually exclusive")
    if args.sequential_walls_as_negative and not args.use_sequential_mask:
        raise ValueError(
            "--sequential-walls-as-negative requires --use-sequential-mask"
        )
    if args.sequential_grid_lines_as_negative and not args.use_sequential_mask:
        raise ValueError(
            "--sequential-grid-lines-as-negative requires --use-sequential-mask"
        )
    if args.explanation_loss_type == "signed" and not args.use_signed_action_mask and not args.use_combined_explanation_masks:
        raise ValueError("--explanation-loss-type signed requires --use-signed-action-mask")
    if args.use_signed_action_mask and args.explanation_loss_type != "signed":
        raise ValueError("--use-signed-action-mask requires --explanation-loss-type signed")
    if args.use_action_category_mask and args.explanation_loss_type != "absolute_energy":
        raise ValueError(
            "--use-action-category-mask requires "
            "--explanation-loss-type absolute_energy"
        )
    if args.use_local_action_mask and args.explanation_loss_type != "masked_absolute_energy":
        raise ValueError(
            "--use-local-action-mask requires "
            "--explanation-loss-type masked_absolute_energy"
        )
    if args.explanation_loss_type == "masked_absolute_energy" and not args.use_local_action_mask:
        raise ValueError(
            "--explanation-loss-type masked_absolute_energy requires "
            "--use-local-action-mask"
        )
    if args.explanation_loss_type == "absolute_energy" and not args.use_action_category_mask:
        raise ValueError(
            "--explanation-loss-type absolute_energy requires "
            "--use-action-category-mask"
        )
    if (
        args.explanation_loss_type == "partial_energy"
        and not (
            args.use_literal_combined_mask
            or (
                args.use_sequential_mask
                and (
                    args.sequential_walls_as_negative
                    or args.sequential_grid_lines_as_negative
                )
            )
        )
    ):
        raise ValueError(
            "--explanation-loss-type partial_energy requires "
            "--use-literal-combined-mask or a sequential mask with "
            "a negative-region flag"
        )
    if (
        args.use_literal_combined_mask
        and args.explanation_loss_type != "partial_energy"
    ):
        raise ValueError(
            "--use-literal-combined-mask requires "
            "--explanation-loss-type partial_energy"
        )
    if (
        args.explanation_coef > 0
        and args.explanation_loss_type == "positive_margin"
        and not args.use_sequential_mask
    ):
        raise ValueError(
            "--explanation-loss-type positive_margin requires --use-sequential-mask"
        )
    dual_coef_active = (
        args.sequential_explanation_coef > 0 or args.action_explanation_coef > 0
    )
    if dual_coef_active != args.use_combined_explanation_masks:
        raise ValueError(
            "Combined-mask mode and its sequential/action coefficients must be enabled together"
        )
    if args.use_combined_explanation_masks and args.explanation_coef > 0:
        raise ValueError("--explanation-coef is not used in combined-mask mode")
    signed_enabled_actions = tuple(a.strip() for a in args.signed_actions.split(",") if a.strip())
    critic_activation = activation_from_name(args.critic_activation)

    log_root = ROOT / "eval_logs" / f"{args.save_prefix}_{args.env_id}_{timestamp}"
    best_model_dir = log_root / "best_model"
    eval_log_dir = log_root / "eval_results"
    checkpoint_dir = log_root / "checkpoints"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(
        env_id=args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        use_rgb_img_obs=args.use_rgb_img_obs,
        invert_channels=args.invert_channels,
        use_inventory_obs=args.use_inventory_obs,
        shortcut_prob=args.shortcut_prob,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=args.use_sequential_mask,
        sequential_walls_as_negative=args.sequential_walls_as_negative,
        sequential_grid_lines_as_negative=args.sequential_grid_lines_as_negative,
        use_static_multi_mask=args.use_static_multi_mask,
        use_signed_action_mask=args.use_signed_action_mask,
        use_action_category_mask=args.use_action_category_mask,
        use_local_action_mask=args.use_local_action_mask,
        local_action_movement_radius=args.local_action_movement_radius,
        use_combined_explanation_masks=args.use_combined_explanation_masks,
        use_literal_combined_mask=args.use_literal_combined_mask,
        signed_enable_negative=not args.signed_disable_negative,
        signed_enabled_actions=signed_enabled_actions,
        signed_neg_on_locked_door_without_key=args.signed_neg_on_locked_door,
    )

    eval_env = build_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed + 10_000,
        use_rgb_img_obs=args.use_rgb_img_obs,
        invert_channels=args.invert_channels,
        use_inventory_obs=args.use_inventory_obs,
        shortcut_prob=args.shortcut_prob,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=args.use_sequential_mask,
        sequential_walls_as_negative=args.sequential_walls_as_negative,
        sequential_grid_lines_as_negative=args.sequential_grid_lines_as_negative,
        use_static_multi_mask=args.use_static_multi_mask,
        use_signed_action_mask=args.use_signed_action_mask,
        use_action_category_mask=args.use_action_category_mask,
        use_local_action_mask=args.use_local_action_mask,
        local_action_movement_radius=args.local_action_movement_radius,
        use_combined_explanation_masks=args.use_combined_explanation_masks,
        use_literal_combined_mask=args.use_literal_combined_mask,
        signed_enable_negative=not args.signed_disable_negative,
        signed_enabled_actions=signed_enabled_actions,
        signed_neg_on_locked_door_without_key=args.signed_neg_on_locked_door,
    )
    stable_bcos_norm = args.stable_bcos_norm and (
        args.actor_type == "bcos" or args.critic_type == "bcos"
    )
    policy_kwargs: dict[str, Any] = {
        "actor_type": args.actor_type,
        "critic_type": args.critic_type,
        "net_arch": {"pi": pi_layers, "vf": vf_layers},
        "activation_fn": critic_activation,
        "bcos_b": args.bcos_b,
        "bcos_max_out": args.bcos_max_out,
        "stable_bcos_norm": stable_bcos_norm,
        "share_features_extractor": False,
    }

    if args.use_rgb_img_obs:
        policy_kwargs.update({
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "use_bcos": args.actor_type == "bcos",
                "bcos_b": args.bcos_b,
                "bcos_max_out": args.bcos_max_out,
                # Inverse channels are part of the observation so that
                # gradient*input explanations include their contribution.
                "invert_channels": False,
                "stable_bcos_norm": stable_bcos_norm and args.actor_type == "bcos",
                "bias_free_bcos_norm": args.actor_type == "bcos",
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                "features_dim": 256,
                "use_bcos": args.critic_type == "bcos",
                "bcos_b": args.bcos_b,
                "bcos_max_out": args.bcos_max_out,
                "invert_channels": False,
                "stable_bcos_norm": stable_bcos_norm and args.critic_type == "bcos",
                "bias_free_bcos_norm": args.critic_type == "bcos",
            },
            "normalize_images": True,
        })
    elif args.symbolic_extractor == "cnn":
        policy_kwargs.update({
            "features_extractor_class": SymbolicGridCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": args.symbolic_features_dim,
                "use_bcos": args.actor_type == "bcos",
                "bcos_b": args.bcos_b,
                "bcos_max_out": args.bcos_max_out,
                "stable_bcos_norm": stable_bcos_norm and args.actor_type == "bcos",
                "bias_free_bcos_norm": args.actor_type == "bcos",
            },
            "critic_features_extractor_class": SymbolicGridCNNExtractor,
            "critic_features_extractor_kwargs": {
                "features_dim": args.symbolic_features_dim,
                "use_bcos": args.critic_type == "bcos",
                "bcos_b": args.bcos_b,
                "bcos_max_out": args.bcos_max_out,
                "stable_bcos_norm": stable_bcos_norm and args.critic_type == "bcos",
                "bias_free_bcos_norm": args.critic_type == "bcos",
            },
            "normalize_images": False,
        })
    else:
        policy_kwargs.update({
            "features_extractor_class": FlattenExtractor,
            "normalize_images": False,
        })

    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=train_env,
        confidence_coef=args.confidence_coef_start,
        explanation_coef=args.explanation_coef,
        sequential_explanation_coef=args.sequential_explanation_coef,
        action_explanation_coef=args.action_explanation_coef,
        sequential_margin=args.sequential_margin,
        partial_energy_alignment_margin=args.partial_energy_alignment_margin,
        use_original_explanation_energy_loss=args.use_original_explanation_energy_loss,
        explanation_loss_type=args.explanation_loss_type,
        signed_margin=args.signed_margin,
        signed_pos_weight=args.signed_pos_weight,
        signed_neg_weight=args.signed_neg_weight,
        explanation_max_grad_ratio=args.explanation_max_grad_ratio,
        gradient_diagnostics=args.gradient_diagnostics,
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
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    stats_callback = MiniGridStatsCallback(window_size=100)
    checkpoint_callback = SuccessRateCheckpointCallback(
        save_dir=str(checkpoint_dir),
        threshold=args.checkpoint_threshold,
        interval=args.checkpoint_interval,
    )
    confidence_coef_ramp_callback = ConfidenceCoefficientRampCallback(
        start_coef=args.confidence_coef_start,
        end_coef=args.confidence_coef_end,
        total_steps=args.total_timesteps,
        schedule=args.confidence_coef_schedule,
    )
    best_vecnorm_path = best_model_dir / "best_model.vecnormalize.pkl"
    save_vecnorm_on_best = SaveVecNormalizeOnBest(
        save_path=str(best_vecnorm_path), verbose=1,
    )

    total_eval_intervals = 5
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq = max(args.total_timesteps // (args.n_envs * total_eval_intervals), 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        callback_on_new_best=save_vecnorm_on_best,
    )

    callback = CallbackList([stats_callback, checkpoint_callback, confidence_coef_ramp_callback, eval_callback])

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.save_prefix}_{args.env_id}_{timestamp}.zip"
    model.save(str(save_path))

    vecnorm_path = save_dir / f"{args.save_prefix}_{args.env_id}_{timestamp}.vecnormalize.pkl"
    vecnorm_env = model.get_vec_normalize_env()
    if vecnorm_env is not None:
        vecnorm_env.save(str(vecnorm_path))

    if args.log_config:
        config_path = save_path.with_suffix(".config.json")
        config_path.write_text(json.dumps(vars(args), indent=2))

    train_env.close()
    eval_env.close()

    print(f"Saved trained model to: {save_path}")
    if vecnorm_env is not None:
        print(f"Saved VecNormalize stats to: {vecnorm_path}")
    print(f"Best eval model dir: {best_model_dir}")
    print(f"Eval logs dir: {eval_log_dir}")


if __name__ == "__main__":
    main()

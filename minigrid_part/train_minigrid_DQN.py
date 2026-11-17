#!/usr/bin/env python3
"""
Train PPO on MiniGrid DoorKey with:
- actor: selectable B-cos MLP or standard MLP
- critic: standard MLP
- optional B-cos action head for discrete actions
- two observation pipelines:
  1) symbolic/grid path: MiniGrid DoorKey -> optional FullyObsWrapper -> ImgObsWrapper -> float32 -> MLP policy
  2) RGB path: MiniGrid DoorKey -> RGBImgObsWrapper -> ImgObsWrapper -> VecTransposeImage -> CNN feature extractor -> B-cos CNN if actor_type == "bcos", otherwise standard CNN
- separate evaluation environment with EvalCallback
- VecNormalize only for non-RGB observations
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import os
import sys
from typing import Any

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage
from stable_baselines3.common.torch_layers import FlattenExtractor

from env_wrappers import (
    make_wrapper, MiniGridStatsCallback
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bcos_rl_generic.train_model import (
    SaveVecNormalizeOnBest, ConfigurableDQNPolicy, 
    GenericCNNExtractor, parse_layer_sizes, validate_layer_sizes, activation_from_name,
)
from bcos_rl_generic.bcosDQN import (
    BcosDQN, AuxAlphaRampCallback,
)

ROOT = Path(__file__).resolve().parent

def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on MiniGrid DoorKey with B-cos or MLP actor.")
    parser.add_argument("--env-id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # DQN Hyperparameters
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Size of the replay buffer.")
    parser.add_argument("--learning-starts", type=int, default=1_000, help="How many steps to collect before learning starts.")
    parser.add_argument("--train-freq", type=int, default=4, help="Update the model every X steps.")
    parser.add_argument("--gradient-steps", type=int, default=1, help="How many gradient steps to take per update.")
    parser.add_argument("--target-update-interval", type=int, default=1_000, help="Target network update frequency.")
    parser.add_argument("--exploration-fraction", type=float, default=0.6, help="Fraction of training period to reduce exploration rate.")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05, help="Final value of random action probability.")

    parser.add_argument("--aux-alpha-start", type=float, default=0.0, help="Initial aux loss coefficient (0 = no alignment pressure at start).")
    parser.add_argument("--aux-alpha-end", type=float, default=0.01, help="Final aux loss coefficient.")
    parser.add_argument("--temp", type=float, default=5.0, help="Initial temperature for scaling Q-values in aux loss (higher = softer).")

    parser.add_argument("--actor-type", type=str, default="bcos", choices=["bcos", "mlp"])
    parser.add_argument("--q-layers", type=str, default="128,128")
    parser.add_argument("--critic-activation", type=str, default="tanh", choices=["tanh", "relu", "elu", "leaky_relu"])
    parser.add_argument("--bcos-b", type=float, default=2.0)
    parser.add_argument("--bcos-max-out", type=int, default=1)
    
    parser.add_argument("--use-rgb-img-obs", action="store_true", help="Use fully observable RGB image observations with a CNN feature extractor.")
    parser.add_argument("--invert-channels", action="store_true", help="For B-cos CNN, apply the [x, 1-x] transform to get 2x channels.")

    parser.add_argument("--shortcut-prob", type=float, default=0.0, help="Probability of creating a shortcut hole in the wall each episode (for testing generalization).")
    parser.add_argument("--open-door-reward", type=float, default=0.0, help="Additional reward for opening the door (if the env has a door).")
    parser.add_argument("--pick-key-reward", type=float, default=0.0, help="Additional reward for picking up the key (if the env has a key).")

    parser.add_argument("--tensorboard-log", type=str, default=str(ROOT / "runs" / "minigrid_dqn"))
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "models"))
    parser.add_argument("--save-prefix", type=str, default="minigrid_dqn")
    parser.add_argument("--log-config", action="store_true")

    return parser.parse_args()

def build_vec_env(
    *, env_id: str, n_envs: int, seed: int, use_rgb_img_obs: bool, shortcut_prob: float = 0.0, open_door_reward: float = 0.0, pick_key_reward: float = 0.0
) -> gym.Env:
    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"highlight": False},
        wrapper_class=make_wrapper(
            use_rgb_img_obs=use_rgb_img_obs,
            shortcut_prob=shortcut_prob,
            open_door_reward=open_door_reward,
            pick_key_reward=pick_key_reward,
        ),
    )
    
    if use_rgb_img_obs:
        env = VecTransposeImage(env)
    else:
        env = VecNormalize(
            env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        )
    return env


def main() -> None:
    args = make_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    q_layers = parse_layer_sizes(args.q_layers)
    validate_layer_sizes(q_layers, arg_name="--q-layers")

    if args.bcos_max_out < 1:
        raise ValueError(f"--bcos-max-out must be >= 1, got {args.bcos_max_out}")
    critic_activation = activation_from_name(args.critic_activation)

    log_root = ROOT / "eval_logs" / f"{args.save_prefix}_{args.env_id}_{timestamp}"
    best_model_dir = log_root / "best_model"
    eval_log_dir = log_root / "eval_results"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(
    env_id=args.env_id,
    n_envs=args.n_envs,
    seed=args.seed,
    use_rgb_img_obs=args.use_rgb_img_obs,
    shortcut_prob=args.shortcut_prob,
    open_door_reward=args.open_door_reward,
    pick_key_reward=args.pick_key_reward,
    )
    eval_env = build_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed + 10_000,
        use_rgb_img_obs=args.use_rgb_img_obs,
        shortcut_prob=args.shortcut_prob,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
    )

    policy_kwargs: dict[str, Any] = {
        "actor_type": args.actor_type,
        "net_arch": q_layers, 
        "activation_fn": critic_activation,
        "bcos_b": args.bcos_b,
        "bcos_max_out": args.bcos_max_out,
    }

    if args.use_rgb_img_obs:
        policy_kwargs.update({
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "use_bcos": args.actor_type == "bcos",
                "bcos_b": args.bcos_b,
                "bcos_max_out": args.bcos_max_out,
                "invert_channels": args.invert_channels,
            },
            "normalize_images": True,
        })
    else:
        policy_kwargs.update({
            "features_extractor_class": FlattenExtractor,
            "normalize_images": False,
        })
    model = DQN(
        policy=ConfigurableDQNPolicy,
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model = BcosDQN(
        policy=ConfigurableDQNPolicy,
        env=train_env,
        aux_alpha=args.aux_alpha_start,
        temperature=args.temp,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    stats_callback = MiniGridStatsCallback(window_size=100)
    aux_alpha_ramp_callback = AuxAlphaRampCallback(
        start_alpha=args.aux_alpha_start,
        end_alpha=args.aux_alpha_end,
        total_steps=args.total_timesteps,
    )
    best_vecnorm_path = best_model_dir / "best_model.vecnormalize.pkl"
    save_vecnorm_on_best = SaveVecNormalizeOnBest(
        save_path=str(best_vecnorm_path), verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=False,
        render=False,
        callback_on_new_best=save_vecnorm_on_best,
    )

    callback = CallbackList([stats_callback, aux_alpha_ramp_callback, eval_callback])

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
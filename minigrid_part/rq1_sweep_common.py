#!/usr/bin/env python3
"""Shared, auxiliary-free training utilities for the final RQ1 sweeps."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from wandb.integration.sb3 import WandbCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from minigrid_part.env_wrappers import MiniGridStatsCallback, make_wrapper
except ModuleNotFoundError:  # Direct execution from minigrid_part/.
    from env_wrappers import MiniGridStatsCallback, make_wrapper

from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    ConfigurableDQNPolicy,
    GenericCNNExtractor,
    activation_from_name,
    parse_layer_sizes,
)

DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "rq1_architecture"


def parse_ppo_variant(variant: str) -> tuple[str, float]:
    if variant == "standard":
        return "standard", 2.0
    prefix = "bcos_b"
    if not variant.startswith(prefix):
        raise ValueError(f"Unknown PPO variant: {variant}")
    return "bcos", float(variant[len(prefix) :])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def count_parameters(model) -> dict[str, int]:
    all_parameters = list(model.policy.parameters())
    return {
        "total": int(sum(parameter.numel() for parameter in all_parameters)),
        "trainable": int(
            sum(parameter.numel() for parameter in all_parameters if parameter.requires_grad)
        ),
    }


def build_minigrid_env(
    *,
    env_id: str,
    n_envs: int,
    seed: int,
    rgb6: bool,
    open_door_reward: float,
    pick_key_reward: float,
):
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"highlight": False},
        wrapper_class=make_wrapper(
            use_rgb_img_obs=True,
            invert_rgb_channels=rgb6,
            open_door_reward=open_door_reward,
            pick_key_reward=pick_key_reward,
        ),
        vec_env_cls=DummyVecEnv,
    )
    return VecTransposeImage(env)


def final_stochastic_evaluation(model, env, episodes: int) -> dict[str, float]:
    observations = env.reset()
    returns = np.zeros(env.num_envs, dtype=np.float64)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    successes: list[float] = []

    while len(completed_returns) < episodes:
        actions, _ = model.predict(observations, deterministic=False)
        observations, rewards, dones, infos = env.step(actions)
        returns += rewards
        lengths += 1
        for index, done in enumerate(dones):
            if not done or len(completed_returns) >= episodes:
                continue
            info = infos[index]
            episode = info.get("episode", {})
            completed_returns.append(float(episode.get("r", returns[index])))
            completed_lengths.append(int(episode.get("l", lengths[index])))
            successes.append(float(info.get("is_success", 0.0)))
            returns[index] = 0.0
            lengths[index] = 0

    return {
        "episodes": float(episodes),
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(completed_returns)),
        "std_reward": float(np.std(completed_returns, ddof=1)),
        "mean_episode_length": float(np.mean(completed_lengths)),
        "std_episode_length": float(np.std(completed_lengths, ddof=1)),
    }


def ppo_policy_kwargs(*, policy_type: str, bcos_b: float, cfg: dict[str, Any]) -> dict:
    use_bcos = policy_type == "bcos"
    stable_norm = bool(use_bcos and cfg["stable_bcos_norm"])
    common = {
        "features_dim": int(cfg["features_dim"]),
        "bcos_b": bcos_b,
        "bcos_max_out": 1,
    }
    return {
        "actor_type": "bcos" if use_bcos else "mlp",
        "critic_type": "mlp",
        "net_arch": {
            "pi": parse_layer_sizes(str(cfg["pi_layers"])),
            "vf": parse_layer_sizes(str(cfg["vf_layers"])),
        },
        "activation_fn": activation_from_name(str(cfg["activation"])),
        "bcos_b": bcos_b,
        "bcos_max_out": 1,
        "stable_bcos_norm": stable_norm,
        "share_features_extractor": False,
        "features_extractor_class": GenericCNNExtractor,
        "features_extractor_kwargs": {
            **common,
            "use_bcos": use_bcos,
            "invert_channels": False,
            "stable_bcos_norm": stable_norm,
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


def dqn_policy_kwargs(*, policy_type: str, bcos_b: float, cfg: dict[str, Any]) -> dict:
    use_bcos = policy_type == "bcos"
    return {
        "actor_type": "bcos" if use_bcos else "mlp",
        "net_arch": parse_layer_sizes(str(cfg["q_layers"])),
        "activation_fn": activation_from_name(str(cfg["activation"])),
        "bcos_b": bcos_b,
        "bcos_max_out": 1,
        "features_extractor_class": GenericCNNExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(cfg["features_dim"]),
            "use_bcos": use_bcos,
            "bcos_b": bcos_b,
            "bcos_max_out": 1,
            "invert_channels": False,
            "stable_bcos_norm": bool(use_bcos and cfg["stable_bcos_norm"]),
            "bias_free_bcos_norm": use_bcos,
        },
        "normalize_images": True,
    }


def make_run_dir(output_root: Path, experiment: str, variant: str, seed: int, run_id: str) -> Path:
    path = output_root / experiment / variant / f"seed_{seed}" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_slug(env_id: str) -> str:
    return env_id.lower().replace("-v0", "").replace("-v3", "").replace("-", "_")


def _save_run_package(
    *,
    run,
    model,
    run_dir: Path,
    config: dict[str, Any],
    elapsed: float,
    total_timesteps: int,
    final_eval: dict[str, float],
) -> None:
    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    parameters = count_parameters(model)
    manifest = {
        "model": final_path.name,
        "sha256": sha256(final_path),
        "git_revision": git_revision(),
        "parameters": parameters,
        "training_wall_seconds": elapsed,
        "environment_steps_per_second": total_timesteps / max(elapsed, 1e-12),
        "final_stochastic_evaluation": final_eval,
        "config": config,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    run.log(
        {
            "experiment/parameter_count": parameters["total"],
            "experiment/trainable_parameter_count": parameters["trainable"],
            "experiment/training_wall_seconds": elapsed,
            "experiment/environment_steps_per_second": total_timesteps / max(elapsed, 1e-12),
            **{f"final_eval/{key}": value for key, value in final_eval.items()},
        }
    )
    artifact = wandb.Artifact(
        name=f"{config['experiment']}-{run.id}",
        type="model",
        metadata={
            "variant": config["variant"],
            "seed": config["seed"],
            "timesteps": total_timesteps,
            "sha256": manifest["sha256"],
        },
    )
    artifact.add_file(str(final_path))
    artifact.add_file(str(run_dir / "manifest.json"))
    artifact.add_file(str(run_dir / "config.json"))
    run.log_artifact(artifact)


def train_minigrid_ppo(
    args,
    defaults: dict[str, Any],
    config_profiles: dict[str, dict[str, Any]] | None = None,
) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config=defaults,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        tags=["rq1", "minigrid", "ppo", "rgb6", "no-aux"],
    )
    cfg = dict(wandb.config)
    if config_profiles is not None:
        profile_name = str(cfg.get("config_profile", ""))
        if profile_name not in config_profiles:
            raise ValueError(f"Unknown PPO config profile: {profile_name!r}")
        profile = config_profiles[profile_name]
        cfg.update(profile)
        run.config.update(profile, allow_val_change=True)
    policy_type, bcos_b = parse_ppo_variant(str(cfg["variant"]))
    seed = int(cfg["seed"])
    run_dir = make_run_dir(
        args.output_root,
        str(cfg["experiment"]),
        f"{environment_slug(str(cfg['env_id']))}/{cfg['variant']}",
        seed,
        run.id,
    )
    cfg.update(
        {
            "policy_type": policy_type,
            "bcos_b": bcos_b,
            "observation": "RGB6 [RGB, 255-RGB], uint8 CHW",
            "auxiliary_coefficients": 0.0,
            "vec_env": "DummyVecEnv",
            "run_dir": str(run_dir.resolve()),
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8"
    )

    train_env = build_minigrid_env(
        env_id=str(cfg["env_id"]),
        n_envs=int(cfg["n_envs"]),
        seed=seed,
        rgb6=True,
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
    )
    eval_env = build_minigrid_env(
        env_id=str(cfg["env_id"]),
        n_envs=1,
        seed=seed + 10_000,
        rgb6=True,
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
    )
    try:
        algorithm = BcosPPO if policy_type == "bcos" else PPO
        auxiliary = (
            {"confidence_coef": 0.0, "explanation_coef": 0.0}
            if policy_type == "bcos"
            else {}
        )
        model = algorithm(
            policy=ConfigurableActorCriticPolicy,
            env=train_env,
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
            policy_kwargs=ppo_policy_kwargs(policy_type=policy_type, bcos_b=bcos_b, cfg=cfg),
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=seed,
            device=args.device,
            verbose=1,
            **auxiliary,
        )
        if model.observation_space.shape[0] != 6:
            raise RuntimeError(f"Expected RGB6 input, got {model.observation_space.shape}")
        callbacks = CallbackList(
            [
                MiniGridStatsCallback(window_size=100),
                CheckpointCallback(
                    save_freq=max(int(cfg["checkpoint_freq"]) // int(cfg["n_envs"]), 1),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix="model",
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(run_dir / "best_model"),
                    log_path=str(run_dir / "eval"),
                    eval_freq=max(int(cfg["eval_freq"]) // int(cfg["n_envs"]), 1),
                    n_eval_episodes=int(cfg["periodic_eval_episodes"]),
                    deterministic=False,
                    verbose=1,
                ),
                WandbCallback(verbose=1),
            ]
        )
        started = time.perf_counter()
        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=callbacks,
            tb_log_name=f"{cfg['experiment']}_{cfg['variant']}_{seed}",
        )
        elapsed = time.perf_counter() - started
        final_eval = final_stochastic_evaluation(
            model, eval_env, int(cfg["final_eval_episodes"])
        )
        _save_run_package(
            run=run,
            model=model,
            run_dir=run_dir,
            config=cfg,
            elapsed=elapsed,
            total_timesteps=int(cfg["total_timesteps"]),
            final_eval=final_eval,
        )
    finally:
        train_env.close()
        eval_env.close()
        run.finish()


def train_minigrid_dqn(args, defaults: dict[str, Any]) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config=defaults,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        tags=["rq1", "minigrid", "dqn", "rgb6", "no-aux"],
    )
    cfg = dict(wandb.config)
    policy_type = str(cfg["variant"])
    if policy_type not in {"standard", "bcos"}:
        raise ValueError(f"Unknown DQN variant: {policy_type}")
    seed = int(cfg["seed"])
    bcos_b = float(cfg["bcos_b"])
    run_dir = make_run_dir(
        args.output_root,
        str(cfg["experiment"]),
        f"{environment_slug(str(cfg['env_id']))}/{policy_type}",
        seed,
        run.id,
    )
    cfg.update(
        {
            "policy_type": policy_type,
            "observation": "RGB6 [RGB, 255-RGB], uint8 CHW",
            "auxiliary_coefficients": 0.0,
            "vec_env": "DummyVecEnv",
            "run_dir": str(run_dir.resolve()),
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8"
    )
    train_env = build_minigrid_env(
        env_id=str(cfg["env_id"]),
        n_envs=int(cfg["n_envs"]),
        seed=seed,
        rgb6=True,
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
    )
    eval_env = build_minigrid_env(
        env_id=str(cfg["env_id"]),
        n_envs=1,
        seed=seed + 10_000,
        rgb6=True,
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
    )
    try:
        model = DQN(
            policy=ConfigurableDQNPolicy,
            env=train_env,
            learning_rate=float(cfg["learning_rate"]),
            buffer_size=int(cfg["buffer_size"]),
            learning_starts=int(cfg["learning_starts"]),
            batch_size=int(cfg["batch_size"]),
            gamma=float(cfg["gamma"]),
            train_freq=int(cfg["train_freq"]),
            gradient_steps=int(cfg["gradient_steps"]),
            tau=float(cfg["tau"]),
            target_update_interval=int(cfg["target_update_interval"]),
            exploration_fraction=float(cfg["exploration_fraction"]),
            exploration_initial_eps=1.0,
            exploration_final_eps=float(cfg["exploration_final_eps"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            policy_kwargs=dqn_policy_kwargs(policy_type=policy_type, bcos_b=bcos_b, cfg=cfg),
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=seed,
            device=args.device,
            verbose=1,
        )
        if model.observation_space.shape[0] != 6:
            raise RuntimeError(f"Expected RGB6 input, got {model.observation_space.shape}")
        callbacks = CallbackList(
            [
                MiniGridStatsCallback(window_size=100),
                CheckpointCallback(
                    save_freq=max(int(cfg["checkpoint_freq"]) // int(cfg["n_envs"]), 1),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix="model",
                    save_replay_buffer=False,
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(run_dir / "best_model"),
                    log_path=str(run_dir / "eval"),
                    eval_freq=max(int(cfg["eval_freq"]) // int(cfg["n_envs"]), 1),
                    n_eval_episodes=int(cfg["periodic_eval_episodes"]),
                    deterministic=False,
                    verbose=1,
                ),
                WandbCallback(verbose=1),
            ]
        )
        started = time.perf_counter()
        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=callbacks,
            tb_log_name=f"{cfg['experiment']}_{policy_type}_{seed}",
        )
        elapsed = time.perf_counter() - started
        final_eval = final_stochastic_evaluation(
            model, eval_env, int(cfg["final_eval_episodes"])
        )
        replay_buffer_path = run_dir / "final_replay_buffer.pkl"
        model.save_replay_buffer(str(replay_buffer_path))
        cfg["replay_buffer_file"] = replay_buffer_path.name
        cfg["replay_buffer_uploaded_to_wandb"] = False
        (run_dir / "config.json").write_text(
            json.dumps(cfg, indent=2, default=str), encoding="utf-8"
        )
        _save_run_package(
            run=run,
            model=model,
            run_dir=run_dir,
            config=cfg,
            elapsed=elapsed,
            total_timesteps=int(cfg["total_timesteps"]),
            final_eval=final_eval,
        )
    finally:
        train_env.close()
        eval_env.close()
        run.finish()


def common_agent_args(parser, *, default_project: str, default_experiment: str):
    parser.add_argument("--project", default=default_project)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment", default=default_experiment)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def run_sweep_cli(args, sweep_config: dict, train_fn, defaults: dict) -> None:
    if args.create_sweep:
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Created sweep: {sweep_id}")
        return
    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            function=lambda: train_fn(args, defaults),
            count=args.count,
            project=args.project,
            entity=args.entity,
        )
        return
    train_fn(args, defaults)

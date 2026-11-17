#!/usr/bin/env python3
"""RQ1 support sweep: matched standard and B-cos PPO on discrete CarRacing."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from car_racing_part.train_carracing_ppo import (
    DiscreteActionDiagnosticsCallback,
    build_env,
    make_policy_kwargs,
)
from minigrid_part.rq1_sweep_common import (
    DEFAULT_OUTPUT_ROOT,
    _save_run_package,
    final_stochastic_evaluation,
    make_run_dir,
)


def sweep_config() -> dict:
    return {
        "name": "rq1-carracing-ppo-architecture-final-v2",
        "method": "grid",
        "metric": {"name": "final_eval/mean_reward", "goal": "maximize"},
        "parameters": {
            "variant": {"values": ["standard", "bcos"]},
            "seed": {"values": [0, 1, 2, 3, 4]},
        },
    }


def policy_kwargs(cfg: dict) -> dict:
    policy_args = SimpleNamespace(
        policy_type=str(cfg["variant"]),
        features_dim=int(cfg["features_dim"]),
        pi_layers=str(cfg["pi_layers"]),
        vf_layers=str(cfg["vf_layers"]),
        activation=str(cfg["activation"]),
        bcos_b=float(cfg["bcos_b"]),
        bcos_max_out=1,
        stable_bcos_norm=bool(cfg["stable_bcos_norm"]),
        bcos_complement_input=False,
    )
    return make_policy_kwargs(policy_args)


def train_one(args: argparse.Namespace) -> None:
    defaults = {
        "experiment": "carracing_ppo_architecture",
        "env_id": "CarRacing-v3",
        "variant": "standard",
        "seed": 0,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 512,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "target_kl": 0.03,
        "ent_coef": 0.003,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "features_dim": 256,
        "pi_layers": "256,256",
        "vf_layers": "256,256",
        "activation": "tanh",
        "bcos_b": args.bcos_b,
        "stable_bcos_norm": True,
        "eval_freq": 100_000,
        "checkpoint_freq": 250_000,
        "periodic_eval_episodes": 10,
        "final_eval_episodes": 30,
    }
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config=defaults,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        tags=["rq1", "carracing", "ppo", "rgb3", "discrete", "no-aux"],
    )
    cfg = dict(wandb.config)
    variant = str(cfg["variant"])
    seed = int(cfg["seed"])
    if variant not in {"standard", "bcos"}:
        raise ValueError(f"Unknown policy variant: {variant}")
    run_dir = make_run_dir(
        args.output_root,
        str(cfg["experiment"]),
        f"carracing/{variant}",
        seed,
        run.id,
    )
    cfg.update(
        {
            "observation": "single RGB3 frame, uint8 CHW 3x96x96",
            "action_space": "five discrete actions",
            "auxiliary_coefficients": 0.0,
            "domain_randomization": False,
            "vec_env": "DummyVecEnv",
            "run_dir": str(run_dir.resolve()),
        }
    )
    run.config.update(cfg)
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8"
    )
    train_env = build_env(
        env_id="CarRacing-v3",
        n_envs=int(cfg["n_envs"]),
        seed=seed,
        use_subprocesses=False,
    )
    eval_env = build_env(
        env_id="CarRacing-v3", n_envs=1, seed=seed + 10_000, use_subprocesses=False
    )
    try:
        algorithm = BcosPPO if variant == "bcos" else PPO
        auxiliary = (
            {"confidence_coef": 0.0, "explanation_coef": 0.0}
            if variant == "bcos"
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
            policy_kwargs=policy_kwargs(cfg),
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=seed,
            device=args.device,
            verbose=1,
            **auxiliary,
        )
        callbacks = CallbackList(
            [
                DiscreteActionDiagnosticsCallback(),
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
            tb_log_name=f"rq1_carracing_{variant}_{seed}",
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


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="rq1-carracing-ppo-architecture-final-v2")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--bcos-b", type=float, default=2.3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if args.create_sweep:
        sweep_id = wandb.sweep(sweep_config(), project=args.project, entity=args.entity)
        print(f"Created sweep: {sweep_id}")
        return
    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            function=lambda: train_one(args),
            count=args.count,
            project=args.project,
            entity=args.entity,
        )
        return
    train_one(args)


if __name__ == "__main__":
    main()

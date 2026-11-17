#!/usr/bin/env python3
"""W&B sweep for the CarRacing explanation-supervision support experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import wandb
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from wandb.integration.sb3 import WandbCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from car_racing_part.explanation_masks import CarRacingExplanationMaskWrapper, CarRacingMaskConfig
from car_racing_part.train_carracing_ppo import (
    DiscreteActionDiagnosticsCallback,
    build_env,
    make_policy_kwargs,
)
from minigrid_part.rq1_sweep_common import count_parameters, final_stochastic_evaluation, git_revision


class LinearExplanationRampCallback(BaseCallback):
    """Increase explanation supervision evenly over the training run."""

    def __init__(self, *, end_coef: float, ramp_timesteps: int) -> None:
        super().__init__()
        if end_coef <= 0:
            raise ValueError("end_coef must be positive")
        if ramp_timesteps <= 0:
            raise ValueError("ramp_timesteps must be positive")
        self.end_coef = end_coef
        self.ramp_timesteps = ramp_timesteps

    def _set_coefficient(self) -> None:
        progress = min(1.0, self.model.num_timesteps / self.ramp_timesteps)
        coefficient = self.end_coef * progress
        self.model.explanation_coef = coefficient
        self.logger.record("curriculum/explanation_coef", coefficient)
        self.logger.record("curriculum/ramp_progress", progress)

    def _on_training_start(self) -> None:
        self._set_coefficient()

    def _on_step(self) -> bool:
        self._set_coefficient()
        return True


def sweep_config() -> dict:
    return {
        "name": "rq3-carracing-boundary-dashboard-support-v1",
        "method": "grid",
        "metric": {"name": "final_eval/mean_reward", "goal": "maximize"},
        "parameters": {"seed": {"values": [0, 1, 2, 3, 4]}},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="thesis-rq3-carracing-boundary-support-v1")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "rq3_carracing_support" / "runs",
    )
    return parser.parse_args()


def defaults() -> dict:
    return {
        "experiment": "rq3_carracing_boundary_dashboard_support_v1",
        "env_id": "CarRacing-v3",
        "seed": 0,
        "total_timesteps": 1_000_000,
        "n_envs": 8,
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
        "bcos_b": 2.3,
        "stable_bcos_norm": True,
        "eval_freq": 100_000,
        "checkpoint_freq": 250_000,
        "periodic_eval_episodes": 30,
        "final_eval_episodes": 30,
        "explanation_coef": 0.15,
        "explanation_schedule": "linear",
        "ramp_timesteps": 1_000_000,
        "explanation_loss_type": "masked_absolute_energy",
        "explanation_max_grad_ratio": 0.1,
        "mask_profile": "boundary-dashboard",
        "mask_warmup_frames": 0,
        "mask_boundary_radius": 3,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_one(args: argparse.Namespace) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config=defaults(),
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        tags=["rq3", "carracing", "bcos", "explanation-supervision", "support"],
    )
    cfg = dict(wandb.config)
    seed = int(cfg["seed"])
    run_dir = args.output_root / f"seed_{seed}" / run.id
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    eval_dir = run_dir / "eval"
    for directory in (checkpoint_dir, best_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=False)

    cfg.update(
        {
            "run_dir": str(run_dir.resolve()),
            "observation": "single RGB frame, uint8, CHW 3x96x96",
            "action_space": "five discrete actions",
            "mask": "P=car+road-boundaries+full-dashboard; N=grass+road-interior; D=curbs+uncertain",
        }
    )
    run.config.update(cfg, allow_val_change=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    mask_config = CarRacingMaskConfig(
        profile=str(cfg["mask_profile"]),
        warmup_frames=int(cfg["mask_warmup_frames"]),
        boundary_radius=int(cfg["mask_boundary_radius"]),
    )
    train_env = build_env(
        env_id=str(cfg["env_id"]),
        n_envs=int(cfg["n_envs"]),
        seed=seed,
        use_subprocesses=False,
        wrapper_class=lambda env: CarRacingExplanationMaskWrapper(env, config=mask_config),
    )
    eval_env = build_env(
        env_id=str(cfg["env_id"]),
        n_envs=1,
        seed=seed + 10_000,
        use_subprocesses=False,
    )
    final_eval_env = build_env(
        env_id=str(cfg["env_id"]),
        n_envs=1,
        seed=seed + 20_000,
        use_subprocesses=False,
    )
    try:
        policy_args = argparse.Namespace(
            policy_type="bcos",
            features_dim=int(cfg["features_dim"]),
            pi_layers=str(cfg["pi_layers"]),
            vf_layers=str(cfg["vf_layers"]),
            activation=str(cfg["activation"]),
            bcos_b=float(cfg["bcos_b"]),
            bcos_max_out=1,
            stable_bcos_norm=bool(cfg["stable_bcos_norm"]),
            bcos_complement_input=False,
        )
        model = BcosPPO(
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
            policy_kwargs=make_policy_kwargs(policy_args),
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=seed,
            device=args.device,
            verbose=1,
            confidence_coef=0.0,
            explanation_coef=float(cfg["explanation_coef"]),
            explanation_loss_type=str(cfg["explanation_loss_type"]),
            explanation_max_grad_ratio=float(cfg["explanation_max_grad_ratio"]),
        )
        # The positive constructor value creates the masked rollout buffer.
        # The linear callback sets the actual coefficient to zero before the
        # first rollout, then raises it to 0.15 across training.
        model.explanation_coef = 0.0
        callbacks = CallbackList(
            [
                LinearExplanationRampCallback(
                    end_coef=float(cfg["explanation_coef"]),
                    ramp_timesteps=int(cfg["ramp_timesteps"]),
                ),
                DiscreteActionDiagnosticsCallback(),
                CheckpointCallback(
                    save_freq=max(int(cfg["checkpoint_freq"]) // int(cfg["n_envs"]), 1),
                    save_path=str(checkpoint_dir),
                    name_prefix="model",
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(eval_dir),
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
            tb_log_name=f"rq3_carracing_boundary_seed{seed}",
        )
        elapsed = time.perf_counter() - started
        final_eval = final_stochastic_evaluation(
            model, final_eval_env, int(cfg["final_eval_episodes"])
        )
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        manifest = {
            "final_model": final_path.name,
            "final_model_sha256": sha256(final_path),
            "git_revision": git_revision(),
            "parameters": count_parameters(model),
            "training_wall_seconds": elapsed,
            "final_stochastic_evaluation": final_eval,
            "config": cfg,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        run.log({f"final_eval/{key}": value for key, value in final_eval.items()})

        artifact = wandb.Artifact(
            name=f"rq3-carracing-boundary-seed{seed}-{run.id}",
            type="training-run",
            metadata={"seed": seed, "timesteps": int(cfg["total_timesteps"]), "final_model_sha256": manifest["final_model_sha256"]},
        )
        artifact.add_dir(str(run_dir))
        run.log_artifact(artifact).wait()
    finally:
        train_env.close()
        eval_env.close()
        final_eval_env.close()
        run.finish()


def main() -> None:
    args = parse_args()
    if args.create_sweep:
        sweep_id = wandb.sweep(sweep_config(), project=args.project, entity=args.entity)
        print(sweep_id)
    elif args.sweep_id:
        wandb.agent(args.sweep_id, function=lambda: train_one(args), count=args.count, project=args.project, entity=args.entity)
    else:
        raise ValueError("Pass --create-sweep or --sweep-id")


if __name__ == "__main__":
    main()

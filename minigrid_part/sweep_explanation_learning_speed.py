#!/usr/bin/env python3
"""Matched RGB6 sweep comparing explanation-loss training curricula."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback

from env_wrappers import MiniGridStatsCallback
from sweep_local_action_mask import build_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO, SuccessRateCheckpointCallback
from bcos_rl_generic.explanation_supervision import MaskedRolloutBuffer
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from ppo_experiment_profiles import apply_profile, profile_names
from sweep_minigrid_PPO import make_policy_kwargs


ROOT = Path(__file__).resolve().parent
VERSION = "rgb6-explanation-curriculum-speed-v2"
CHANNELWISE_VERSION = "rgb6-explanation-curriculum-speed-channelwise-v4"
ARMS = ("pure", "linear", "success_gated_50", "constant")
TRAIN_THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 1.00)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integers")
    return seeds


def parse_arms(raw: str) -> list[str]:
    arms = [item.strip() for item in raw.split(",") if item.strip()]
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("--arms must contain unique curriculum arms")
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"Unknown curriculum arms: {unknown}")
    return arms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_coefficient(arm: str, target: float) -> float:
    return target if arm == "constant" else 0.0


def scheduled_coefficient(
    arm: str,
    *,
    target: float,
    timestep: int,
    total_timesteps: int,
    gate_timestep: int | None,
) -> float:
    if arm == "pure":
        return 0.0
    if arm == "constant":
        return target
    if arm == "linear":
        return target * min(max(timestep / max(total_timesteps, 1), 0.0), 1.0)
    if arm == "success_gated_50":
        return target if gate_timestep is not None else 0.0
    raise ValueError(f"Unknown curriculum arm: {arm}")


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    target = 0.0 if args.arm == "pure" else args.explanation_coef
    channelwise = bool(args.channelwise_absolute_energy)
    config = {
        "experiment_version": CHANNELWISE_VERSION if channelwise else VERSION,
        "arm": args.arm,
        "condition": "pure" if args.arm == "pure" else "sequential_explanation",
        "configured_explanation_coef": target,
        "target_explanation_coef": target,
        "shared_explanation_coef": args.explanation_coef,
        "initial_explanation_coef": initial_coefficient(args.arm, target),
        "explanation_schedule": args.arm,
        "success_gate_threshold": args.success_gate_threshold,
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device": args.device,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "checkpoint_interval": args.checkpoint_interval,
        "success_window": args.success_window,
        "minimum_success_episodes": args.minimum_success_episodes,
        "open_door_reward": args.open_door_reward,
        "pick_key_reward": args.pick_key_reward,
        "explanation_loss_type": (
            "channelwise_masked_absolute_energy"
            if channelwise
            else "masked_absolute_energy"
        ),
        "channelwise_absolute_energy": channelwise,
        "explanation_max_grad_ratio": args.explanation_max_grad_ratio,
        "mask_p": "exact agent silhouette plus current key/door/goal objective",
        "mask_n": "exact wall silhouettes plus grid-line pixels",
        "movement_radius": 1,
        "learning_rate": 0.0006141598118159885,
        "n_steps": 512,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "target_kl": 0.05,
        "ent_coef": 0.0009960268117654178,
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "pi_layers": "128,128",
        "vf_layers": "128,128",
        "features_dim": 128,
        "bcos_b": 2.3,
        "bcos_max_out": 1,
        "actor_type": "bcos",
        "critic_type": "mlp",
        "critic_activation": "tanh",
        "stable_bcos_norm": True,
        "bias_free_bcos_norm": True,
        "use_rgb_img_obs": True,
        "invert_channels": True,
        "observation_channels": 6,
        "confidence_coef": 0.0,
    }
    return apply_profile(args.ppo_profile, config)


def sweep_config(args: argparse.Namespace) -> dict[str, Any]:
    derived = {
        "condition",
        "configured_explanation_coef",
        "target_explanation_coef",
        "initial_explanation_coef",
        "explanation_schedule",
    }
    parameters = {
        key: {"value": value}
        for key, value in base_config(args).items()
        if key not in derived
    }
    parameters["arm"] = {"values": parse_arms(args.arms)}
    parameters["seed"] = {"values": parse_seeds(args.seeds)}
    return {
        "method": "grid",
        "metric": {
            "name": "learning_speed/first_eval_step_success_100",
            "goal": "minimize",
        },
        "parameters": parameters,
    }


class CurriculumAndSpeedCallback(BaseCallback):
    """Apply the curriculum and record unsmoothed training-success crossings."""

    def __init__(
        self,
        *,
        arm: str,
        target: float,
        window: int,
        minimum_episodes: int,
        gate_threshold: float,
        budget: int,
    ):
        super().__init__(verbose=0)
        self.arm = arm
        self.target = target
        self.history: collections.deque[float] = collections.deque(maxlen=window)
        self.minimum_episodes = minimum_episodes
        self.gate_threshold = gate_threshold
        self.budget = budget
        self.gate_timestep: int | None = None
        self.first_steps: dict[float, int | None] = {
            threshold: None for threshold in TRAIN_THRESHOLDS
        }

    @staticmethod
    def _suffix(threshold: float) -> str:
        return str(int(round(100 * threshold)))

    def _set_coefficient(self) -> None:
        coefficient = scheduled_coefficient(
            self.arm,
            target=self.target,
            timestep=int(self.num_timesteps),
            total_timesteps=self.budget,
            gate_timestep=self.gate_timestep,
        )
        self.model.explanation_coef = coefficient
        self.logger.record("curriculum/explanation_coef", coefficient)
        self.logger.record("curriculum/gate_triggered", float(self.gate_timestep is not None))
        self.logger.record(
            "curriculum/gate_timestep",
            -1 if self.gate_timestep is None else self.gate_timestep,
        )

    def _on_training_start(self) -> None:
        self._set_coefficient()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                self.history.append(float(info.get("is_success", 0.0)))

        if len(self.history) >= self.minimum_episodes:
            rate = float(np.mean(self.history))
            self.logger.record("learning_speed/rolling_success_rate", rate)
            if (
                self.arm == "success_gated_50"
                and self.gate_timestep is None
                and rate >= self.gate_threshold
            ):
                self.gate_timestep = int(self.num_timesteps)
                wandb.run.summary["curriculum/gate_timestep"] = self.gate_timestep
            for threshold in TRAIN_THRESHOLDS:
                if self.first_steps[threshold] is None and rate >= threshold:
                    step = int(self.num_timesteps)
                    self.first_steps[threshold] = step
                    suffix = self._suffix(threshold)
                    self.logger.record(f"learning_speed/first_train_step_success_{suffix}", step)
                    wandb.run.summary[f"learning_speed/first_train_step_success_{suffix}"] = step
                    wandb.run.summary[f"learning_speed/reached_train_success_{suffix}"] = True

        self._set_coefficient()
        return True

    def _on_training_end(self) -> None:
        for threshold, step in self.first_steps.items():
            suffix = self._suffix(threshold)
            reached = step is not None
            wandb.run.summary[f"learning_speed/first_train_step_success_{suffix}"] = (
                step if reached else self.budget + 1
            )
            wandb.run.summary[f"learning_speed/reached_train_success_{suffix}"] = reached


class EvalSpeedCallback(BaseCallback):
    """Record the first deterministic evaluation with 100% success."""

    def __init__(self, evaluation: EvalCallback, *, budget: int):
        super().__init__(verbose=0)
        self.evaluation = evaluation
        self.budget = budget
        self.last_evaluation_call = -1
        self.first_perfect_step: int | None = None

    def _on_step(self) -> bool:
        callback = self.evaluation
        if (
            callback.n_calls == 0
            or callback.n_calls == self.last_evaluation_call
            or callback.n_calls % callback.eval_freq
            or not callback._is_success_buffer
        ):
            return True
        self.last_evaluation_call = callback.n_calls
        success = float(np.mean(callback._is_success_buffer))
        self.logger.record("learning_speed/eval_success_rate", success)
        self.logger.record("learning_speed/eval_mean_reward", callback.last_mean_reward)
        if self.first_perfect_step is None and np.isclose(success, 1.0):
            self.first_perfect_step = int(self.num_timesteps)
            self.logger.record(
                "learning_speed/first_eval_step_success_100", self.first_perfect_step
            )
            wandb.run.summary["learning_speed/first_eval_step_success_100"] = (
                self.first_perfect_step
            )
            wandb.run.summary["learning_speed/reached_eval_success_100"] = True
        return True

    def _on_training_end(self) -> None:
        reached = self.first_perfect_step is not None
        value = self.first_perfect_step if reached else self.budget + 1
        wandb.run.summary["learning_speed/first_eval_step_success_100"] = value
        wandb.run.summary["learning_speed/reached_eval_success_100"] = reached


def train_one_run(args: argparse.Namespace) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        config=base_config(args),
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=False,
        tags=[item.strip() for item in args.tags.split(",") if item.strip()],
    )
    cfg = dict(wandb.config)
    arm = str(cfg["arm"])
    if arm not in ARMS:
        raise ValueError(f"Unknown experiment arm: {arm}")
    target = 0.0 if arm == "pure" else float(cfg["shared_explanation_coef"])
    explanation_loss_type = str(cfg["explanation_loss_type"])
    if explanation_loss_type not in {
        "masked_absolute_energy",
        "channelwise_masked_absolute_energy",
    }:
        raise ValueError(f"Unsupported explanation loss: {explanation_loss_type}")
    initial = initial_coefficient(arm, target)
    run.config.update(
        {
            "condition": "pure" if arm == "pure" else "sequential_explanation",
            "configured_explanation_coef": target,
            "target_explanation_coef": target,
            "initial_explanation_coef": initial,
            "explanation_schedule": arm,
        },
        allow_val_change=True,
    )
    run.summary["experiment_arm"] = arm
    run.summary["configured_explanation_coef"] = target
    cfg.update(dict(run.config))
    if int(cfg["total_timesteps"]) not in {750_000, 1_000_000}:
        raise ValueError(
            "This learning-speed study supports the 750,000-step unified "
            "protocol or the legacy 1,000,000-step protocol"
        )

    run_dir = ROOT / "wandb_runs" / "explanation_curriculum_speed" / run.id
    best_dir = run_dir / "best_model"
    checkpoint_dir = run_dir / "checkpoints"
    model_dir = run_dir / "models"
    for directory in (best_dir, checkpoint_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    n_envs = int(cfg["n_envs"])
    seed = int(cfg["seed"])
    train_condition = "pure" if arm == "pure" else "sequential_semantic"
    train_env = build_env(cfg, n_envs=n_envs, seed=seed, condition=train_condition)
    eval_env = build_env(cfg, n_envs=1, seed=seed + 10_000, condition="pure")
    # Non-pure arms must be constructed with a non-zero coefficient so PPO
    # allocates MaskedRolloutBuffer. The callback then applies the true initial value.
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=train_env,
        bce_coef=0.0,
        explanation_coef=target,
        explanation_loss_type=explanation_loss_type,
        explanation_max_grad_ratio=float(cfg["explanation_max_grad_ratio"]),
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
        policy_kwargs=make_policy_kwargs(cfg),
        tensorboard_log=str(run_dir / "tensorboard"),
        gradient_diagnostics=arm != "pure",
        seed=seed,
        device=str(cfg["device"]),
        verbose=1,
    )
    model.explanation_coef = initial
    if tuple(model.observation_space.shape)[0] != 6:
        raise RuntimeError(f"Expected RGB6 observations, got {model.observation_space.shape}")
    if arm != "pure" and not isinstance(model.rollout_buffer, MaskedRolloutBuffer):
        raise RuntimeError("Explanation curriculum requires MaskedRolloutBuffer")
    if arm == "pure" and isinstance(model.rollout_buffer, MaskedRolloutBuffer):
        raise RuntimeError("Pure control unexpectedly uses MaskedRolloutBuffer")

    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(int(cfg["eval_freq"]) // n_envs, 1),
        n_eval_episodes=int(cfg["n_eval_episodes"]),
        deterministic=True,
        render=False,
        verbose=1,
    )
    curriculum = CurriculumAndSpeedCallback(
        arm=arm,
        target=target,
        window=int(cfg["success_window"]),
        minimum_episodes=int(cfg["minimum_success_episodes"]),
        gate_threshold=float(cfg["success_gate_threshold"]),
        budget=int(cfg["total_timesteps"]),
    )
    eval_speed = EvalSpeedCallback(evaluation, budget=int(cfg["total_timesteps"]))
    callbacks = CallbackList(
        [
            MiniGridStatsCallback(window_size=int(cfg["success_window"])),
            curriculum,
            SuccessRateCheckpointCallback(
                save_dir=str(checkpoint_dir),
                threshold=0.95,
                interval=int(cfg["checkpoint_interval"]),
            ),
            evaluation,
            eval_speed,
            WandbCallback(
                model_save_path=str(model_dir),
                model_save_freq=max(int(cfg["checkpoint_interval"]) // n_envs, 1),
                gradient_save_freq=0,
                verbose=1,
            ),
        ]
    )
    try:
        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=callbacks,
            tb_log_name=f"curriculum_speed_{run.id}",
        )
        final_path = run_dir / f"final_{arm}.zip"
        model.save(final_path)

        statistics = {
            "schema_version": 1,
            "run_id": run.id,
            "project": run.project,
            "experiment_version": str(cfg["experiment_version"]),
            "config": cfg,
            "learning_speed": {
                "gate_timestep": curriculum.gate_timestep,
                "first_train_threshold_steps": {
                    CurriculumAndSpeedCallback._suffix(threshold): step
                    for threshold, step in curriculum.first_steps.items()
                },
                "first_eval_step_success_100": eval_speed.first_perfect_step,
            },
            "models": {
                "final": {
                    "artifact_path": "models/final.zip",
                    "sha256": sha256_file(final_path),
                }
            },
        }
        statistics_path = run_dir / "run_statistics.json"
        statistics_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
        config_path = run_dir / "resolved_config.json"
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        artifact = wandb.Artifact(
            name=f"rq4-channelwise-{arm.replace('_', '-')}-seed-{seed:03d}",
            type="model",
            description=(
                "RQ4 channelwise curriculum model with learning-speed statistics "
                "and deterministic evaluation history."
            ),
            metadata={
                "arm": arm,
                "target_explanation_coef": target,
                "seed": seed,
                "explanation_loss_type": explanation_loss_type,
                "experiment_version": str(cfg["experiment_version"]),
            },
        )
        artifact.add_file(str(final_path), name="models/final.zip")
        best_path = best_dir / "best_model.zip"
        if best_path.exists():
            artifact.add_file(str(best_path), name="models/best.zip")
        evaluations_path = run_dir / "eval" / "evaluations.npz"
        if evaluations_path.exists():
            artifact.add_file(
                str(evaluations_path), name="statistics/evaluations.npz"
            )
        artifact.add_file(str(statistics_path), name="statistics/run_statistics.json")
        artifact.add_file(str(config_path), name="metadata/resolved_config.json")
        run.summary["artifacts/model_name"] = artifact.name
        run.log_artifact(artifact, aliases=["latest", CHANNELWISE_VERSION])
    finally:
        train_env.close()
        eval_env.close()
        run.finish()


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="minigrid-rgb6-explanation-curriculum-speed-v2")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--arm", choices=ARMS, default="pure")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ppo-profile",
        choices=profile_names(),
        default="current",
        help="Frozen PPO profile; use rq1-rq3-stellar for unified reruns.",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--success-window", type=int, default=100)
    parser.add_argument("--minimum-success-episodes", type=int, default=100)
    parser.add_argument("--success-gate-threshold", type=float, default=0.5)
    parser.add_argument("--explanation-coef", type=float, default=0.15)
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1)
    parser.add_argument(
        "--channelwise-absolute-energy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sum_c |E_c| so RGB6 channels cannot cancel.",
    )
    parser.add_argument(
        "--tags",
        default="minigrid,ppo,bcos,rgb6,explanation-loss,curriculum,learning-speed,matched-seeds",
    )
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if args.create_sweep:
        sweep_id = wandb.sweep(
            sweep_config(args), project=args.project, entity=args.entity or None
        )
        print(f"Created sweep: {sweep_id}")
    elif args.sweep_id:
        wandb.agent(
            sweep_id=args.sweep_id,
            function=lambda: train_one_run(args),
            count=args.count,
            project=args.project,
            entity=args.entity or None,
        )
    else:
        train_one_run(args)


if __name__ == "__main__":
    main()

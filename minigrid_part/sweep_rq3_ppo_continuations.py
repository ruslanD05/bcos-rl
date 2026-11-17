#!/usr/bin/env python3
"""RQ3: matched pure, sequential, local, and flipped PPO continuations."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wandb
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback

from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.explanation_supervision import MaskedRolloutBuffer
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from finetune_pretrained_mask_ablation import (
    AdditionalStepsCheckpointCallback,
    inspect_checkpoint,
    make_env,
)
from rq1_sweep_common import (
    common_agent_args,
    count_parameters,
    final_stochastic_evaluation,
    git_revision,
    run_sweep_cli,
    sha256,
)
from train_minigrid_ActorCritic import build_vec_env
from env_wrappers import MiniGridStatsCallback

BRANCH_TO_MASK_MODE = {
    "pure": "pure",
    "sequential": "sequential-all-negative",
    "local": "local-directional-masked",
    "flipped": "flipped-sequential",
}


def parse_csv_choices(raw: str, *, allowed: set[str], name: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"--{name} must contain unique values")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown --{name} values: {unknown}")
    return values


def parse_sweep_seeds(raw: str) -> list[int]:
    values = parse_csv_choices(
        raw,
        allowed={str(seed) for seed in range(5)},
        name="seeds",
    )
    return [int(value) for value in values]


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_agent_args(
        parser,
        default_project="rq3-minigrid-ppo-strict-mask-control-v2",
        default_experiment="rq3_ppo_strict_mask_continuations_v2",
    )
    parser.set_defaults(
        output_root=(
            ROOT
            / "experiments"
            / "rq3_explanation_control"
            / "continuations"
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
            ROOT
            / "experiments"
            / "rq1_architecture"
            / "minigrid_ppo_architecture"
            / "minigrid_doorkey_8x8"
            / "bcos_b2.3"
        ),
    )
    parser.add_argument("--additional-timesteps", type=int, default=150_000)
    parser.add_argument("--branch", choices=tuple(BRANCH_TO_MASK_MODE), default="pure")
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--branches", default=",".join(BRANCH_TO_MASK_MODE))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--explanation-coef", type=float, default=0.15)
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1)
    parser.add_argument(
        "--channelwise-absolute-energy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sum absolute channel contributions before spatial P/N aggregation.",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--periodic-eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument(
        "--include-pure",
        action="store_true",
        help="Also rerun the pure continuation; normally reuse canonical RQ3 controls.",
    )
    return parser.parse_args()


def source_checkpoint(source_root: Path, seed: int) -> Path:
    candidates = sorted((source_root / f"seed_{seed}").glob("*/final_model.zip"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one RQ1 B=2.3 source checkpoint for seed {seed}, "
            f"found {len(candidates)} under {source_root}"
        )
    return candidates[0]


def branch_env_args(cfg: dict[str, Any], mask_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        mask_mode=mask_mode,
        env_id=str(cfg["env_id"]),
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
        grid_lines_as_negative=True,
        movement_radius=int(cfg["movement_radius"]),
    )


def plain_env(cfg: dict[str, Any], *, n_envs: int, seed: int):
    return build_vec_env(
        env_id=str(cfg["env_id"]),
        n_envs=n_envs,
        seed=seed,
        use_rgb_img_obs=True,
        invert_channels=True,
        open_door_reward=float(cfg["open_door_reward"]),
        pick_key_reward=float(cfg["pick_key_reward"]),
    )


def run_one(args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config=defaults,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        tags=["rq3", "ppo", "rgb6", "matched-continuation"],
    )
    cfg = dict(wandb.config)
    branch = str(cfg["branch"])
    if branch not in BRANCH_TO_MASK_MODE:
        raise ValueError(f"Unknown branch: {branch}")
    seed = int(cfg["seed"])
    mask_mode = BRANCH_TO_MASK_MODE[branch]
    source = source_checkpoint(args.source_root, seed)
    source_timesteps, channels = inspect_checkpoint(source, args.device)
    if channels != 6:
        raise ValueError(f"RQ3 requires RGB6 source checkpoints, got {channels} channels")

    run_dir = args.output_root / f"seed_{seed}" / branch / run.id
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    eval_dir = run_dir / "eval"
    for path in (checkpoint_dir, best_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=False)

    env_args = branch_env_args(cfg, mask_mode)
    train_env = (
        plain_env(cfg, n_envs=int(cfg["n_envs"]), seed=seed)
        if branch == "pure"
        else make_env(
            env_args,
            n_envs=int(cfg["n_envs"]),
            seed=seed,
            observation_channels=channels,
        )
    )
    periodic_eval_env = plain_env(cfg, n_envs=1, seed=seed + 10_000)
    final_eval_env = plain_env(cfg, n_envs=1, seed=seed + 20_000)
    try:
        explanation_coef = 0.0 if branch == "pure" else float(cfg["explanation_coef"])
        explanation_loss_type = (
            "channelwise_masked_absolute_energy"
            if branch != "pure" and bool(cfg["channelwise_absolute_energy"])
            else "masked_absolute_energy"
        )
        model = BcosPPO.load(
            str(source),
            env=train_env,
            device=args.device,
            custom_objects={"policy_class": ConfigurableActorCriticPolicy},
            bce_coef=0.0,
            confidence_coef=0.0,
            explanation_coef=explanation_coef,
            explanation_loss_type=explanation_loss_type,
            sequential_explanation_coef=0.0,
            action_explanation_coef=0.0,
            explanation_max_grad_ratio=float(cfg["explanation_max_grad_ratio"]),
            gradient_diagnostics=branch != "pure",
        )
        if branch == "pure" and isinstance(model.rollout_buffer, MaskedRolloutBuffer):
            raise RuntimeError("Pure continuation unexpectedly uses a masked buffer")
        if branch != "pure" and not isinstance(model.rollout_buffer, MaskedRolloutBuffer):
            raise RuntimeError("Explanation continuation did not create a masked buffer")

        model.tensorboard_log = str(run_dir / "tensorboard")
        rollout_size = model.n_steps * int(cfg["n_envs"])
        actual_additional = math.ceil(
            int(cfg["additional_timesteps"]) / rollout_size
        ) * rollout_size
        resolved_config = {
            **cfg,
            "source_checkpoint": str(source.resolve()),
            "source_sha256": sha256(source),
            "source_timesteps": source_timesteps,
            "mask_mode": mask_mode,
            "effective_explanation_coef": explanation_coef,
            "explanation_loss_type": explanation_loss_type,
            "channelwise_absolute_energy": bool(cfg["channelwise_absolute_energy"]),
            "actual_additional_timesteps": actual_additional,
            "expected_final_timesteps": source_timesteps + actual_additional,
            "observation": "RGB6 [RGB, 255-RGB], uint8 CHW",
            "git_revision": git_revision(),
        }
        run.config.update(resolved_config, allow_val_change=True)
        (run_dir / "config.json").write_text(
            json.dumps(resolved_config, indent=2, default=str), encoding="utf-8"
        )

        callbacks = CallbackList(
            [
                MiniGridStatsCallback(window_size=100),
                AdditionalStepsCheckpointCallback(
                    checkpoint_dir, int(cfg["checkpoint_freq"]), branch
                ),
                EvalCallback(
                    periodic_eval_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(eval_dir),
                    eval_freq=max(int(cfg["eval_freq"]) // int(cfg["n_envs"]), 1),
                    n_eval_episodes=int(cfg["periodic_eval_episodes"]),
                    deterministic=False,
                    render=False,
                    verbose=1,
                ),
                WandbCallback(verbose=1),
            ]
        )

        started = time.perf_counter()
        model.learn(
            total_timesteps=int(cfg["additional_timesteps"]),
            reset_num_timesteps=False,
            callback=callbacks,
            tb_log_name=f"rq3_{branch}_seed{seed}",
        )
        elapsed = time.perf_counter() - started
        expected = source_timesteps + actual_additional
        if int(model.num_timesteps) != expected:
            raise RuntimeError(f"Expected {expected} timesteps, got {model.num_timesteps}")

        final_eval = final_stochastic_evaluation(
            model, final_eval_env, int(cfg["final_eval_episodes"])
        )
        final_path = run_dir / "final_model.zip"
        model.save(str(final_path))
        parameters = count_parameters(model)
        manifest = {
            "model": final_path.name,
            "sha256": sha256(final_path),
            "source_checkpoint": str(source.resolve()),
            "source_sha256": resolved_config["source_sha256"],
            "git_revision": resolved_config["git_revision"],
            "branch": branch,
            "seed": seed,
            "parameters": parameters,
            "training_wall_seconds": elapsed,
            "environment_steps_per_second": actual_additional / max(elapsed, 1e-12),
            "final_stochastic_evaluation": final_eval,
            "config": resolved_config,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        run.log(
            {
                "experiment/parameter_count": parameters["total"],
                "experiment/training_wall_seconds": elapsed,
                "experiment/environment_steps_per_second": actual_additional
                / max(elapsed, 1e-12),
                **{f"final_eval/{key}": value for key, value in final_eval.items()},
            }
        )
        artifact = wandb.Artifact(
            f"rq3-{branch}-seed{seed}-{run.id}",
            type="model",
            metadata={
                "branch": branch,
                "seed": seed,
                "source_sha256": resolved_config["source_sha256"],
                "sha256": manifest["sha256"],
            },
        )
        for path in (final_path, run_dir / "config.json", run_dir / "manifest.json"):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    finally:
        train_env.close()
        periodic_eval_env.close()
        final_eval_env.close()
        run.finish()


def main() -> None:
    args = make_args()
    defaults = {
        "experiment": args.experiment,
        "branch": args.branch,
        "seed": args.seed,
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "additional_timesteps": args.additional_timesteps,
        "n_envs": args.n_envs,
        "explanation_coef": args.explanation_coef,
        "explanation_max_grad_ratio": args.explanation_max_grad_ratio,
        "channelwise_absolute_energy": args.channelwise_absolute_energy,
        "movement_radius": 1,
        "open_door_reward": 0.1,
        "pick_key_reward": 0.05,
        "eval_freq": args.eval_freq,
        "checkpoint_freq": args.checkpoint_freq,
        "periodic_eval_episodes": args.periodic_eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "mask_definition_version": "strict-v3",
    }
    branches = parse_csv_choices(
        args.branches,
        allowed=set(BRANCH_TO_MASK_MODE),
        name="branches",
    )
    if not args.include_pure:
        branches = [branch for branch in branches if branch != "pure"]
    if not branches:
        raise ValueError("No branches remain; use --include-pure or select supervised branches")
    seeds = parse_sweep_seeds(args.seeds)
    sweep = {
        "name": (
            "rq3-minigrid-ppo-channelwise-control-v3"
            if args.channelwise_absolute_energy
            else "rq3-minigrid-ppo-strict-mask-control-v2"
        ),
        "method": "grid",
        "metric": {"name": "final_eval/success_rate", "goal": "maximize"},
        "parameters": {
            "branch": {"values": branches},
            "seed": {"values": seeds},
        },
    }
    run_sweep_cli(args, sweep, run_one, defaults)


if __name__ == "__main__":
    main()

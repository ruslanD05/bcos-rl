#!/usr/bin/env python3
"""RQ6 Confirmatory Sweep: Oracle-Action Shortcut Suppression with Channelwise Absolute-Energy Loss.

Evaluates paired baseline vs. attribution-suppressed B-cos PPO policies
using the validated RQ3/RQ4 architecture (B=2.3, RMSNormUncentered1d, GroupNormUncentered2d)
and the corrected channelwise masked absolute-energy loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import wandb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MINIGRID_DIR = Path(__file__).resolve().parent
if str(MINIGRID_DIR) not in sys.path:
    sys.path.insert(0, str(MINIGRID_DIR))

import evaluate_shortcut_experiment as evaluator
from ppo_experiment_profiles import apply_profile, profile_names
from train_shortcut_experiment import train_condition

SWEEP_VERSION = "rq6-shortcut-suppression-channelwise-v4"
CONDITIONS = ("baseline", "suppressed")


def parse_csv(raw: str, cast=str) -> list[Any]:
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Comma-separated values must be non-empty and unique")
    return values


def fixed_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "sweep_version": SWEEP_VERSION,
        "seed": args.seed,
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "device": args.device,
        "training_correlation": args.training_correlation,
        "border_height": args.border_height,
        "patch_size": args.patch_size,
        "mask_profile": args.mask_profile,
        "explanation_loss_type": "channelwise_masked_absolute_energy",
        "explanation_coef": args.explanation_coef,
        "explanation_max_grad_ratio": args.explanation_max_grad_ratio,
        "eval_episodes": args.eval_episodes,
        "max_paired_states": args.max_paired_states,
        "source_success_threshold": args.source_success_threshold,
        "eval_stochastic": True,
        "learning_rate": 5.31e-4,
        "n_steps": 512,
        "batch_size": 512,
        "n_epochs": 8,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 1.0,
        "features_dim": 128,
        "pi_layers": "128,128",
        "vf_layers": "128,128",
        "bcos_b": 2.3,
        "actor_type": "bcos",
        "critic_type": "mlp",
        "bcos_max_out": 1,
        "stable_bcos_norm": True,
        "bias_free_bcos_norm": True,
    }
    return apply_profile(args.ppo_profile, config)


def sweep_config(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_csv(args.seeds, int)
    correlations = (
        parse_csv(args.training_correlations, float)
        if args.training_correlations
        else [float(args.training_correlation)]
    )
    invalid_correlations = [value for value in correlations if not 0.5 <= value <= 1.0]
    if invalid_correlations:
        raise ValueError(
            "Training correlations must be within [0.5, 1.0]: "
            f"{invalid_correlations}"
        )
    parameters = {key: {"value": value} for key, value in fixed_config(args).items()}
    parameters["seed"] = {"values": seeds}
    parameters["training_correlation"] = {"values": correlations}
    return {
        "method": "grid",
        "metric": {
            "name": "paired/permuted_success_drop_reduction",
            "goal": "maximize",
        },
        "parameters": parameters,
    }


def resolved_run_args(cli: argparse.Namespace, cfg: dict[str, Any], output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        condition="both",
        env_id=str(cfg["env_id"]),
        total_timesteps=int(cfg["total_timesteps"]),
        n_envs=int(cfg["n_envs"]),
        seed=int(cfg["seed"]),
        device=str(cfg["device"]),
        training_correlation=float(cfg["training_correlation"]),
        border_height=int(cfg["border_height"]),
        patch_size=int(cfg["patch_size"]),
        mask_profile=str(cfg["mask_profile"]),
        explanation_loss_type="channelwise_masked_absolute_energy",
        explanation_coef=float(cfg["explanation_coef"]),
        explanation_max_grad_ratio=float(cfg["explanation_max_grad_ratio"]),
        source_model=None,
        learning_rate=float(cfg["learning_rate"]),
        n_steps=int(cfg["n_steps"]),
        batch_size=int(cfg["batch_size"]),
        n_epochs=int(cfg["n_epochs"]),
        gamma=float(cfg["gamma"]),
        gae_lambda=float(cfg["gae_lambda"]),
        clip_range=float(cfg["clip_range"]),
        ent_coef=float(cfg["ent_coef"]),
        vf_coef=float(cfg["vf_coef"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        features_dim=int(cfg["features_dim"]),
        pi_layers=str(cfg["pi_layers"]),
        vf_layers=str(cfg["vf_layers"]),
        bcos_b=float(cfg["bcos_b"]),
        eval_freq=cli.training_eval_freq,
        n_eval_episodes=cli.training_eval_episodes,
        output_dir=output,
        run_name=f"rq6_channelwise_seed{cfg['seed']}",
    )


def evaluation_args(cfg: dict[str, Any], output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        env_id=str(cfg["env_id"]),
        episodes=int(cfg["eval_episodes"]),
        seed=20_000 + int(cfg["seed"]),
        device=str(cfg["device"]),
        stochastic=True,
        border_height=int(cfg["border_height"]),
        patch_size=int(cfg["patch_size"]),
        max_paired_states=int(cfg["max_paired_states"]),
        batch_size=32,
        output=output,
    )


def log_condition(name: str, result: dict[str, Any]) -> None:
    for mode, metrics in result["performance"].items():
        for metric, value in metrics.items():
            wandb.run.summary[f"{name}/{mode}_{metric}"] = value
    paired = result["paired_sensitivity"]
    wandb.run.summary[f"{name}/shortcut_region_attribution_ratio"] = paired[
        "shortcut_region_attribution"
    ][
        "mean_shortcut_region_attribution_ratio"
    ]
    for mode in ("permuted", "removed"):
        wandb.run.summary[f"{name}/{mode}_probability_l1"] = paired[mode][
            "mean_probability_l1"
        ]
        wandb.run.summary[f"{name}/{mode}_action_change_rate"] = 1.0 - paired[
            mode
        ]["greedy_action_agreement"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_performance_table(run: wandb.sdk.wandb_run.Run, results: dict[str, Any]) -> None:
    table = wandb.Table(
        columns=[
            "condition",
            "evaluation_mode",
            "success_rate",
            "mean_reward",
            "mean_episode_length",
        ]
    )
    for condition, result in results.items():
        for mode, metrics in result["performance"].items():
            table.add_data(
                condition,
                mode,
                metrics["success_rate"],
                metrics["mean_reward"],
                metrics["mean_episode_length"],
            )
    run.log({"evaluation/performance_table": table})


def train_one_run(args: argparse.Namespace) -> None:
    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        config=fixed_config(args),
        sync_tensorboard=True,
        save_code=True,
        tags=[item.strip() for item in args.tags.split(",") if item.strip()],
    )
    cfg = dict(wandb.config)
    run_dir = ROOT / "experiments" / "rq6_shortcut_suppression" / "runs_channelwise" / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_args = resolved_run_args(args, cfg, run_dir)
    eval_args = evaluation_args(cfg, run_dir / "paired_results.json")
    try:
        models = {
            condition: train_condition(train_args, condition, run_dir)
            for condition in CONDITIONS
        }
        results = {
            condition: evaluator.evaluate_model(path, eval_args)
            for condition, path in models.items()
        }
        for condition, result in results.items():
            log_condition(condition, result)

        baseline = results["baseline"]["performance"]
        suppressed = results["suppressed"]["performance"]
        threshold = float(cfg["source_success_threshold"])
        valid = (
            baseline["correlated"]["success_rate"] >= threshold
            and suppressed["correlated"]["success_rate"] >= threshold
        )
        run.summary["paired/valid_source_pair"] = int(valid)
        for mode in ("correlated", "permuted", "removed", "random"):
            run.summary[f"paired/suppressed_minus_baseline_{mode}_success"] = (
                suppressed[mode]["success_rate"] - baseline[mode]["success_rate"]
            )
        for mode in ("permuted", "removed", "random"):
            baseline_drop = (
                baseline["correlated"]["success_rate"] - baseline[mode]["success_rate"]
            )
            suppressed_drop = (
                suppressed["correlated"]["success_rate"] - suppressed[mode]["success_rate"]
            )
            run.summary[f"baseline/{mode}_success_drop"] = baseline_drop
            run.summary[f"suppressed/{mode}_success_drop"] = suppressed_drop
            run.summary[f"paired/{mode}_success_drop_reduction"] = (
                baseline_drop - suppressed_drop
            )

        baseline_sensitivity = results["baseline"]["paired_sensitivity"]
        suppressed_sensitivity = results["suppressed"]["paired_sensitivity"]
        shortcut_region_attribution_ratio_reduction = (
            baseline_sensitivity["shortcut_region_attribution"][
                "mean_shortcut_region_attribution_ratio"
            ]
            - suppressed_sensitivity["shortcut_region_attribution"][
                "mean_shortcut_region_attribution_ratio"
            ]
        )
        wandb.run.summary["paired/shortcut_region_attribution_ratio_reduction"] = (
            shortcut_region_attribution_ratio_reduction
        )

        log_performance_table(run, results)

        statistics = {
            "schema_version": 1,
            "run_id": run.id,
            "project": run.project,
            "sweep_version": SWEEP_VERSION,
            "config": cfg,
            "valid_source_pair": valid,
            "models": {
                condition: {
                    "artifact_path": f"models/{condition}.zip",
                    "sha256": sha256_file(path),
                }
                for condition, path in models.items()
            },
            "results": results,
            "paired": {
                "shortcut_region_attribution_ratio_reduction": shortcut_region_attribution_ratio_reduction,
                "suppressed_minus_baseline_success": {
                    mode: (
                        suppressed[mode]["success_rate"]
                        - baseline[mode]["success_rate"]
                    )
                    for mode in ("correlated", "permuted", "removed", "random")
                },
                "success_drop_reduction": {
                    mode: (
                        (
                            baseline["correlated"]["success_rate"]
                            - baseline[mode]["success_rate"]
                        )
                        - (
                            suppressed["correlated"]["success_rate"]
                            - suppressed[mode]["success_rate"]
                        )
                    )
                    for mode in ("permuted", "removed", "random")
                },
            },
        }
        statistics_path = run_dir / "run_statistics.json"
        statistics_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")

        config_path = run_dir / "resolved_config.json"
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        with open(run_dir / "paired_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "valid": valid,
                    "baseline_perf": baseline,
                    "suppressed_perf": suppressed,
                    "baseline_shortcut_region_ratio": baseline_sensitivity["shortcut_region_attribution"]["mean_shortcut_region_attribution_ratio"],
                    "suppressed_shortcut_region_ratio": suppressed_sensitivity["shortcut_region_attribution"]["mean_shortcut_region_attribution_ratio"],
                },
                f,
                indent=2,
            )

        correlation_tag = f"{float(cfg['training_correlation']):.2f}".replace(".", "p")
        artifact = wandb.Artifact(
            name=f"rq6-channelwise-pair-corr-{correlation_tag}-seed-{int(cfg['seed']):03d}",
            type="model",
            description=(
                "Paired baseline and explanation-suppressed RQ6 models with full "
                "counterfactual evaluation statistics."
            ),
            metadata={
                "sweep_version": SWEEP_VERSION,
                "seed": int(cfg["seed"]),
                "training_correlation": float(cfg["training_correlation"]),
                "explanation_loss_type": str(cfg["explanation_loss_type"]),
                "valid_source_pair": bool(valid),
                "eval_episodes": int(cfg["eval_episodes"]),
            },
        )
        artifact.add_file(str(models["baseline"]), name="models/baseline.zip")
        artifact.add_file(str(models["suppressed"]), name="models/suppressed.zip")
        artifact.add_file(str(statistics_path), name="statistics/run_statistics.json")
        artifact.add_file(str(config_path), name="metadata/resolved_config.json")
        artifact.add_file(
            str(run_dir / "paired_summary.json"),
            name="statistics/paired_summary.json",
        )
        run.summary["artifacts/model_pair_name"] = artifact.name
        run.log_artifact(artifact, aliases=["latest", SWEEP_VERSION])
    finally:
        run.finish()


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="rq6-minigrid-shortcut-channelwise-v3")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--create-sweep", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ppo-profile",
        choices=profile_names(),
        default="current",
        help="Frozen PPO profile; use rq1-rq3-stellar for unified reruns.",
    )
    parser.add_argument("--training-correlation", type=float, default=1.0)
    parser.add_argument(
        "--training-correlations",
        default="",
        help=(
            "Optional comma-separated correlation grid used when creating a W&B "
            "sweep, for example 0.5,0.6,0.7,0.8,0.9,1.0."
        ),
    )
    parser.add_argument("--border-height", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument(
        "--mask-profile",
        choices=(
            "border_negative_screen_positive",
            "sequential",
            "negative_hint_removal",
            "oracle_patch_polarity",
            "patch_only_negative",
        ),
        default="border_negative_screen_positive",
        help="Mask setup: 'border_negative_screen_positive' sets top border (patch + margins) to N (-1) and entire game screen to P (+1).",
    )
    parser.add_argument("--explanation-coef", type=float, default=0.15)
    parser.add_argument("--explanation-max-grad-ratio", type=float, default=0.1)
    parser.add_argument("--training-eval-freq", type=int, default=25_000)
    parser.add_argument("--training-eval-episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--max-paired-states", type=int, default=128)
    parser.add_argument("--source-success-threshold", type=float, default=0.90)
    parser.add_argument(
        "--tags",
        default="rq6,minigrid,ppo,bcos,rgb6,channelwise-loss,shortcut-suppression,paired",
    )
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if args.create_sweep:
        sweep_id = wandb.sweep(
            sweep_config(args), project=args.project, entity=args.entity or None
        )
        print(f"Created RQ6 W&B sweep: {sweep_id}")
        print(
            f"Run agents with:\n"
            f"python minigrid_part/sweep_rq6_shortcut_suppression_channelwise.py "
            f"--project {args.project} "
            f"{f'--entity {args.entity} ' if args.entity else ''}"
            f"--sweep-id {sweep_id} --count 1"
        )
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

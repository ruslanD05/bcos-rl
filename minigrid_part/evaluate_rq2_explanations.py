#!/usr/bin/env python3
"""Compare intrinsic B-cos and post-hoc explanations on shared MiniGrid inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from evaluate_explanation_faithfulness import (
    empty_grid_substrate,
    evaluate_model,
    load_dataset,
    parse_fractions,
)
from evaluate_explanation_metrics import (
    evaluate_policy,
    load_model,
    make_eval_env,
)
from evaluate_wall_ood import parse_models


METHODS = ("bcos", "saliency", "integrated_gradients", "smoothgrad")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_methods(raw: str) -> list[str]:
    methods = [value.strip() for value in raw.split(",") if value.strip()]
    invalid = sorted(set(methods).difference(METHODS))
    if invalid or not methods:
        raise ValueError(f"Methods must be selected from {METHODS}; got {invalid}")
    return list(dict.fromkeys(methods))


def architecture(model) -> str:
    return "bcos" if getattr(model.policy, "actor_type", None) == "bcos" else "standard"


def collect_dataset(args, models, model_paths):
    if args.reference_model not in models:
        raise ValueError(f"Unknown reference model: {args.reference_model}")
    reference = models[args.reference_model]
    shape = tuple(reference.observation_space.shape)
    if len(shape) != 3 or shape[0] not in (3, 6):
        raise ValueError(f"Expected CHW RGB/RGB6 reference model, got {shape}")
    env_args = SimpleNamespace(
        env_id=args.env_id,
        seed=args.seed,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        grid_lines_as_negative=args.grid_lines_as_negative,
    )
    env = make_eval_env(env_args, six_channel_obs=shape[0] == 6)
    try:
        policy, dataset = evaluate_policy(
            reference,
            env,
            episodes=args.dataset_episodes,
            sample_count=args.max_samples,
            seed=args.seed,
            collect=True,
            deterministic=not args.stochastic_dataset,
        )
    finally:
        env.close()
    if dataset is None:
        raise RuntimeError("Dataset collection returned no samples")
    observations, actions, masks = dataset
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dataset_out,
        observations=observations,
        actions=actions,
        masks=masks,
        reference_model=np.asarray(args.reference_model),
        reference_model_path=np.asarray(str(model_paths[args.reference_model].resolve())),
    )
    return observations, actions, masks, policy


def summary_row(
    model_name: str,
    model_architecture: str,
    method: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    summary = metrics["summary"]
    sensitivity = metrics["sensitivity_n_and_infidelity"]["aggregate"]
    completeness = metrics["completeness"]
    return {
        "model": model_name,
        "architecture": model_architecture,
        "method": method,
        "samples": metrics["samples"],
        "deletion_advantage": summary["deletion_advantage_over_random"],
        "insertion_advantage": summary["insertion_advantage_over_random"],
        "sensitivity_n_pearson": sensitivity["sensitivity_n_pearson"],
        "sensitivity_n_spearman_secondary": sensitivity["sensitivity_n_spearman"],
        "normalized_infidelity": sensitivity["normalized_infidelity"],
        "completeness_relative_error": completeness["relative_absolute_error"],
        "attribution_seconds_per_sample": metrics["attribution_runtime"][
            "seconds_per_sample"
        ],
        "p_deletion_advantage": metrics["regions"]["P"][
            "deletion_probability_drop_advantage_over_random"
        ],
        "n_deletion_advantage": metrics["regions"]["N"][
            "deletion_probability_drop_advantage_over_random"
        ],
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "deletion_advantage",
        "insertion_advantage",
        "sensitivity_n_pearson",
        "sensitivity_n_spearman_secondary",
        "normalized_infidelity",
        "completeness_relative_error",
        "attribution_seconds_per_sample",
        "p_deletion_advantage",
        "n_deletion_advantage",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["architecture"], row["method"]), []).append(row)
    result = {}
    for (arch, method), values in groups.items():
        key = f"{arch}/{method}"
        result[key] = {"models": [value["model"] for value in values], "n": len(values)}
        for metric in metrics:
            array = np.asarray([float(value[metric]) for value in values])
            result[key][metric] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "values": array.tolist(),
            }
    return result


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    entries = [
        (model_name, method, method_result["faithfulness"])
        for model_name, model_result in result["models"].items()
        for method, method_result in model_result["methods"].items()
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for model_name, method, metrics in entries:
        label = f"{model_name}:{method}"
        for axis, curve_name in zip(axes, ("deletion", "insertion")):
            points = metrics[curve_name]["high"]
            x = sorted(float(value) for value in points)
            y = [points[f"{value:g}"]["probability_retention"] for value in x]
            axis.plot(x, y, marker=".", linewidth=1, label=label)
    axes[0].set_title("Deletion")
    axes[1].set_title("Insertion")
    for axis in axes:
        axis.set_xlabel("spatial pixel fraction")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("selected-action probability retention")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("RQ2 explanation faithfulness on shared states and actions")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--dataset-in", type=Path)
    parser.add_argument(
        "--dataset-out", type=Path, default=Path("rq2_shared_states_rgb6.npz")
    )
    parser.add_argument("--dataset-episodes", type=int, default=30)
    parser.add_argument("--stochastic-dataset", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--smoothgrad-samples", type=int, default=8)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--fractions", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
    )
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument(
        "--grid-lines-as-negative", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("rq2_explanation_comparison.json")
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if min(
        args.max_samples,
        args.batch_size,
        args.ig_steps,
        args.smoothgrad_samples,
        args.random_repeats,
        args.bootstrap_repeats,
    ) <= 0:
        raise ValueError("All sample, step, repeat, and batch counts must be positive")
    methods = parse_methods(args.methods)
    fractions = parse_fractions(args.fractions)
    model_paths = parse_models(args.model)
    models = {name: load_model(path, args.device) for name, path in model_paths.items()}

    if args.dataset_in:
        observations, actions, masks = load_dataset(args.dataset_in, args.max_samples)
        dataset_path = args.dataset_in
        reference_policy = None
    else:
        observations, actions, masks, reference_policy = collect_dataset(
            args, models, model_paths
        )
        dataset_path = args.dataset_out

    expected_shape = tuple(observations.shape[1:])
    mismatched = {
        name: tuple(model.observation_space.shape)
        for name, model in models.items()
        if tuple(model.observation_space.shape) != expected_shape
    }
    if mismatched:
        raise ValueError(f"Model/dataset observation mismatch: {mismatched}")
    substrate = empty_grid_substrate(expected_shape, args.grid_size)

    result: dict[str, Any] = {
        "dataset": str(dataset_path.resolve()),
        "dataset_reference_model": args.reference_model,
        "dataset_policy": reference_policy,
        "samples": int(len(observations)),
        "actions": "stored reference-model actions shared by every method and model",
        "scalar_target": "selected action logit minus mean action logit",
        "spatial_aggregation": "signed channel sum; absolute value used for ranking",
        "primary_substrate": "valid rendered empty MiniGrid grid",
        "methods": methods,
        "fractions": fractions,
        "random_repeats": args.random_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "models": {},
    }
    rows: list[dict[str, Any]] = []
    for model_index, (name, model) in enumerate(models.items()):
        arch = architecture(model)
        result["models"][name] = {
            "path": str(model_paths[name].resolve()),
            "sha256": sha256(model_paths[name]),
            "architecture": arch,
            "saved_timesteps": int(model.num_timesteps),
            "methods": {},
        }
        for method_index, method in enumerate(methods):
            if method == "bcos" and arch != "bcos":
                continue
            print(f"Evaluating {name}: {method}", flush=True)
            started = time.perf_counter()
            metrics = evaluate_model(
                model,
                observations,
                actions,
                masks,
                substrate,
                fractions,
                args.batch_size,
                args.seed + model_index * 100_000 + method_index * 10_000,
                args.random_repeats,
                args.bootstrap_repeats,
                "stored",
                method,
                args.ig_steps,
                args.smoothgrad_samples,
            )
            result["models"][name]["methods"][method] = {
                "evaluation_wall_seconds": float(time.perf_counter() - started),
                "faithfulness": metrics,
            }
            rows.append(summary_row(name, arch, method, metrics))

    result["aggregate_by_architecture_and_method"] = aggregate_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = args.csv or args.output.with_suffix(".csv")
    plot_path = args.plot or args.output.with_suffix(".png")
    save_csv(rows, csv_path)
    save_plot(result, plot_path)

    print("\nRQ2 summary (larger deletion/insertion advantages are better):")
    for row in rows:
        print(
            f"- {row['model']}:{row['method']} "
            f"deletion={row['deletion_advantage']:.4f}, "
            f"insertion={row['insertion_advantage']:.4f}, "
            f"Sensitivity-n(Pearson)={row['sensitivity_n_pearson']:.4f}, "
            f"runtime={row['attribution_seconds_per_sample']:.4f}s/sample"
        )
    print(f"Saved metrics: {args.output}")
    print(f"Saved summary: {csv_path}")
    print(f"Saved curves: {plot_path}")


if __name__ == "__main__":
    main()

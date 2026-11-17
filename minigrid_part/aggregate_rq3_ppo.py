#!/usr/bin/env python3
"""Aggregate matched-seed RQ3 PPO evaluation files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


AVAILABLE_BRANCHES = ("pure", "sequential", "local", "flipped")
DEFAULT_BRANCHES = AVAILABLE_BRANCHES
METRICS = {
    "policy_success": ("policy", "success_rate"),
    "policy_reward": ("policy", "mean_reward"),
    "policy_episode_length": ("policy", "mean_episode_length"),
    "entropy": ("fixed_dataset_actor_distribution", "mean_entropy"),
    "sequential_mask_ratio": (
        "mask_alignment",
        "sequential",
        "target_energy_ratio",
    ),
    "local_mask_ratio": ("mask_alignment", "local", "target_energy_ratio"),
    "flipped_mask_ratio": (
        "mask_alignment",
        "flipped",
        "target_energy_ratio",
    ),
    "sequential_p_total_ratio": (
        "pnd_absolute_energy",
        "sequential",
        "positive_total_energy_ratio",
    ),
    "sequential_n_total_ratio": (
        "pnd_absolute_energy",
        "sequential",
        "negative_total_energy_ratio",
    ),
    "sequential_d_total_ratio": (
        "pnd_absolute_energy",
        "sequential",
        "neutral_total_energy_ratio",
    ),
    "local_p_total_ratio": (
        "pnd_absolute_energy",
        "local",
        "positive_total_energy_ratio",
    ),
    "local_n_total_ratio": (
        "pnd_absolute_energy",
        "local",
        "negative_total_energy_ratio",
    ),
    "local_d_total_ratio": (
        "pnd_absolute_energy",
        "local",
        "neutral_total_energy_ratio",
    ),
    "deletion_advantage": (
        "faithfulness",
        "summary",
        "deletion_advantage_over_random",
    ),
    "insertion_advantage": (
        "faithfulness",
        "summary",
        "insertion_advantage_over_random",
    ),
    "sensitivity_n": (
        "faithfulness",
        "sensitivity_n_and_infidelity",
        "aggregate",
        "sensitivity_n_spearman",
    ),
    "infidelity": (
        "faithfulness",
        "sensitivity_n_and_infidelity",
        "aggregate",
        "normalized_infidelity",
    ),
    "training_wall_seconds": ("continuation", "training_wall_seconds"),
    "environment_steps_per_second": (
        "continuation",
        "environment_steps_per_second",
    ),
}


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=AVAILABLE_BRANCHES,
        default=list(DEFAULT_BRANCHES),
        help="Branches to include (default: pure sequential local flipped).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/rq3_explanation_control/evaluation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/rq3_explanation_control/final/summary.json"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=30_000)
    return parser.parse_args()


def nested(mapping: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = mapping
    for key in path:
        value = value[key]
    if value is None:
        return float("nan")
    return float(value)


def bootstrap_mean(values: np.ndarray, repeats: int, rng: np.random.Generator) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    draws = rng.choice(finite, size=(repeats, len(finite)), replace=True).mean(axis=1)
    return {
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n": int(len(finite)),
    }


def main() -> None:
    args = make_args()
    paths = args.input or sorted(args.input_dir.glob("seed_*/evaluation.json"))
    if len(paths) < 2:
        raise RuntimeError(f"Expected at least two seed evaluations, found {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for record in records:
        missing = set(args.branches).difference(record["models"])
        if missing:
            raise RuntimeError(f"Evaluation is missing branches: {sorted(missing)}")

    rng = np.random.default_rng(args.seed)
    result: dict[str, Any] = {
        "experiment": "RQ3 matched PPO explanation-control continuations",
        "statistical_unit": "independent source/training seed",
        "seeds": len(records),
        "inputs": [str(path.resolve()) for path in paths],
        "branches": {},
        "paired_differences_vs_pure": {},
    }
    rows: list[dict[str, Any]] = []
    for branch in args.branches:
        result["branches"][branch] = {}
        result["paired_differences_vs_pure"][branch] = {}
        for metric, metric_path in METRICS.items():
            values = np.asarray(
                [nested(record["models"][branch], metric_path) for record in records]
            )
            result["branches"][branch][metric] = bootstrap_mean(
                values, args.bootstrap_repeats, rng
            )
            pure = np.asarray(
                [nested(record["models"]["pure"], metric_path) for record in records]
            )
            result["paired_differences_vs_pure"][branch][metric] = bootstrap_mean(
                values - pure, args.bootstrap_repeats, rng
            )
            for index, value in enumerate(values):
                rows.append(
                    {
                        "seed": records[index]["models"][branch]["continuation"]["seed"],
                        "branch": branch,
                        "metric": metric,
                        "value": value,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"RQ3 aggregate summary (mean across {len(records)} seeds):")
    for branch in args.branches:
        branch_result = result["branches"][branch]
        print(
            f"- {branch}: success={branch_result['policy_success']['mean']:.3f}, "
            f"sequential={branch_result['sequential_mask_ratio']['mean']:.3f}, "
            f"local={branch_result['local_mask_ratio']['mean']:.3f}, "
            f"flipped={branch_result['flipped_mask_ratio']['mean']:.3f}, "
            f"deletion={branch_result['deletion_advantage']['mean']:.3f}, "
            f"insertion={branch_result['insertion_advantage']['mean']:.3f}"
        )
    print(f"Saved: {args.output}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

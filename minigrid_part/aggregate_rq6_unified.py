#!/usr/bin/env python3
"""Aggregate the complete 30-pair RQ6 evaluation set and render thesis figures."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PAIR_NAME = re.compile(r"corr-(?P<corr>\d+p\d+)_seed-(?P<seed>\d{3})\.json$")
MODELS = ("baseline", "suppressed")
CORRELATIONS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/rq6_shortcut_suppression/evaluations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rq6_shortcut_suppression/aggregate"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("thesis/source/Thesis_Template_DE_EN/BA_MA_English/sections/figures"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_823)
    return parser.parse_args()


def parse(path: Path) -> tuple[float, int]:
    match = PAIR_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected RQ6 result filename: {path.name}")
    return float(match.group("corr").replace("p", ".")), int(match.group("seed"))


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def paired_bootstrap(values: np.ndarray, *, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def read_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("corr-*_seed-*.json")):
        correlation, seed = parse(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["config"]["episodes"] != 100:
            raise ValueError(f"Unexpected episode count in {path}")
        row: dict[str, Any] = {"correlation": correlation, "seed": seed, "source_file": str(path)}
        for model in MODELS:
            data = payload["models"][model]
            attribution = data["paired_sensitivity"]["shortcut_region_attribution"]
            if attribution.get("aggregation") != "sum_c_abs_E_c":
                raise ValueError(f"RQ6 shortcut-region metric is not channelwise in {path}")
            row[f"{model}_permuted_success"] = data["performance"]["permuted"]["success_rate"]
            row[f"{model}_permuted_episode_length"] = data["performance"]["permuted"]["mean_episode_length"]
            row[f"{model}_shortcut_region_ratio"] = attribution[
                "mean_shortcut_region_attribution_ratio"
            ]
        records.append(row)
    expected = {(corr, seed) for corr in CORRELATIONS for seed in range(5)}
    observed = {(row["correlation"], row["seed"]) for row in records}
    if observed != expected:
        raise RuntimeError(f"Incomplete/duplicate RQ6 design: expected {expected - observed}; unexpected {observed - expected}")
    return records


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for correlation in CORRELATIONS:
        selected = sorted((row for row in records if row["correlation"] == correlation), key=lambda row: row["seed"])
        result: dict[str, Any] = {"correlation": correlation, "n_seeds": len(selected)}
        for metric in ("permuted_success", "permuted_episode_length", "shortcut_region_ratio"):
            for model in MODELS:
                values = np.asarray([row[f"{model}_{metric}"] for row in selected], dtype=float)
                mean, sd = mean_sd(values)
                result[f"{model}_{metric}_mean"] = mean
                result[f"{model}_{metric}_sd"] = sd
        difference = np.asarray(
            [row["suppressed_permuted_success"] - row["baseline_permuted_success"] for row in selected], dtype=float
        )
        result["permuted_success_difference_mean"] = float(difference.mean())
        result["permuted_success_difference_ci95_low"], result["permuted_success_difference_ci95_high"] = paired_bootstrap(
            difference, repeats=args.bootstrap_repeats, seed=args.seed + round(correlation * 1000)
        )
        baseline = result["baseline_shortcut_region_ratio_mean"]
        suppressed = result["suppressed_shortcut_region_ratio_mean"]
        result["shortcut_region_ratio_reduction_percent"] = 100 * (1 - suppressed / baseline)
        rows.append(result)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_success_difference(records: list[dict[str, Any]], summary: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.1))
    x = np.arange(len(CORRELATIONS))
    rng = np.random.default_rng(20_260_823)
    for index, correlation in enumerate(CORRELATIONS):
        selected = [row for row in records if row["correlation"] == correlation]
        values = np.asarray([100 * (row["suppressed_permuted_success"] - row["baseline_permuted_success"]) for row in selected])
        axis.scatter(index + rng.uniform(-0.09, 0.09, len(values)), values, color="#6c757d", alpha=0.9, s=30, zorder=2)
        stat = summary[index]
        mean = 100 * stat["permuted_success_difference_mean"]
        low = 100 * stat["permuted_success_difference_ci95_low"]
        high = 100 * stat["permuted_success_difference_ci95_high"]
        axis.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="#c1121f", markersize=6, capsize=4, linewidth=1.6, zorder=3)
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(x, [f"{int(c * 100)}%" for c in CORRELATIONS])
    axis.set_xlabel("Training shortcut reliability")
    axis.set_ylabel("Suppressed − baseline permuted success (pp)")
    axis.set_title("RQ6: effect of explanation supervision under a wrong shortcut cue")
    axis.grid(axis="y", alpha=0.28)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    records = read_records(args.input_dir)
    summary = summarize(records, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, args.output_dir / "per_pair.csv")
    write_csv(summary, args.output_dir / "summary.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "pairs": 30,
                    "seeds_per_reliability": 5,
                    "evaluation_episodes": 100,
                    "policy_action_mode": "stochastic",
                    "shortcut_region_energy_aggregation": "sum_c_abs_E_c",
                    "shortcut_region": "full top wall row (rows 0--7, all columns)",
                    "paired_bootstrap_repeats": args.bootstrap_repeats,
                },
                "summary": summary,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    plot_success_difference(records, summary, args.figure_dir / "figure_rq6_permuted_success_difference.pdf")
    print(f"Saved RQ6 summary to {args.output_dir}")


if __name__ == "__main__":
    main()

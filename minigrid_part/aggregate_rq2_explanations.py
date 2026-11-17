#!/usr/bin/env python3
"""Aggregate final RQ2 metrics and render representative explanations."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_explanation_faithfulness import explanations, load_dataset
from evaluate_explanation_metrics import load_model


ROOT = Path(__file__).resolve().parents[1]

METRICS = (
    "deletion_advantage",
    "insertion_advantage",
    "sensitivity_n_pearson",
    "normalized_infidelity",
    "completeness_relative_error",
    "attribution_seconds_per_sample",
)
METHOD_LABELS = {
    "bcos": "Intrinsic B-cos",
    "saliency": "Saliency",
    "integrated_gradients": "Integrated Gradients",
    "smoothgrad": "SmoothGrad",
}
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")


def resolve_model_path(raw_path: str) -> Path:
    """Resolve paths stored by an earlier Windows evaluation on this checkout."""
    direct = Path(raw_path)
    if direct.is_file():
        return direct
    parts = raw_path.replace("\\", "/").split("/")
    try:
        start = parts.index("experiments")
    except ValueError as error:
        raise FileNotFoundError(f"Cannot resolve recorded model path: {raw_path}") from error
    resolved = ROOT.joinpath(*parts[start:])
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Recorded model path is unavailable locally: {raw_path}\n"
            f"Expected restored path: {resolved}"
        )
    return resolved


def read_result(path: Path, profile: str) -> list[dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for model_name, model in result["models"].items():
        for method, entry in model["methods"].items():
            metrics = entry["faithfulness"]
            summary = metrics["summary"]
            sensitivity_results = metrics["sensitivity_n_and_infidelity"]
            fixed_fraction_results = [
                value
                for fraction, value in sensitivity_results.items()
                if fraction != "aggregate"
            ]
            sensitivity = sensitivity_results["aggregate"]
            rows.append(
                {
                    "profile": profile,
                    "model": model_name,
                    "architecture": model["architecture"],
                    "method": method,
                    "model_path": model["path"],
                    "deletion_advantage": summary["deletion_advantage_over_random"],
                    "insertion_advantage": summary["insertion_advantage_over_random"],
                    # Sensitivity-n is defined for a fixed subset size. Average
                    # the PCCs for the prespecified 10%-90% subset fractions
                    # instead of pooling pairs from different subset sizes.
                    "sensitivity_n_pearson": float(
                        np.mean(
                            [
                                value["sensitivity_n_pearson"]
                                for value in fixed_fraction_results
                            ]
                        )
                    ),
                    "normalized_infidelity": sensitivity["normalized_infidelity"],
                    "completeness_relative_error": metrics["completeness"][
                        "relative_absolute_error"
                    ],
                    "attribution_seconds_per_sample": metrics["attribution_runtime"][
                        "seconds_per_sample"
                    ],
                }
            )
    return rows


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "n": int(len(values)),
    }


def seed_from_model(model: str) -> int:
    match = re.search(r"_s(\d+)$", model)
    if not match:
        raise ValueError(f"Model name does not end in a seed identifier: {model}")
    return int(match.group(1))


def clustered_bootstrap_mean_ci(
    values: np.ndarray,
    seeds: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    unique_seeds = np.unique(seeds)
    sampled = rng.choice(unique_seeds, size=(repeats, len(unique_seeds)), replace=True)
    boot = np.asarray(
        [
            np.concatenate([values[seeds == seed] for seed in draw]).mean()
            for draw in sampled
        ]
    )
    low, high = np.percentile(boot, (2.5, 97.5))
    return float(low), float(high)


def aggregate(
    rows: list[dict[str, Any]], bootstrap_repeats: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"groups": {}, "paired_bcos_differences": {}}
    for architecture in ("standard", "bcos"):
        for method in METHOD_LABELS:
            selected = [
                row
                for row in rows
                if row["architecture"] == architecture and row["method"] == method
            ]
            if not selected:
                continue
            key = f"{architecture}/{method}"
            result["groups"][key] = {}
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                seeds = np.asarray([seed_from_model(row["model"]) for row in selected])
                stats = summarize(values)
                stats["ci95_low"], stats["ci95_high"] = clustered_bootstrap_mean_ci(
                    values, seeds, rng, bootstrap_repeats
                )
                result["groups"][key][metric] = stats

    bcos_rows = [row for row in rows if row["architecture"] == "bcos"]
    by_model = {
        (profile, model): {
            row["method"]: row
            for row in bcos_rows
            if row["profile"] == profile and row["model"] == model
        }
        for profile, model in sorted(
            {(row["profile"], row["model"]) for row in bcos_rows}
        )
    }
    for alternative in ("saliency", "integrated_gradients", "smoothgrad"):
        comparison = {}
        for metric in METRICS:
            differences = np.asarray(
                [
                    methods["bcos"][metric] - methods[alternative][metric]
                    for methods in by_model.values()
                    if "bcos" in methods and alternative in methods
                ],
                dtype=np.float64,
            )
            difference_seeds = np.asarray(
                [
                    seed_from_model(model)
                    for (_profile, model), methods in by_model.items()
                    if "bcos" in methods and alternative in methods
                ]
            )
            stats = summarize(differences)
            stats["ci95_low"], stats["ci95_high"] = clustered_bootstrap_mean_ci(
                differences, difference_seeds, rng, bootstrap_repeats
            )
            comparison[metric] = stats
        result["paired_bcos_differences"][f"bcos_minus_{alternative}"] = comparison
    return result


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(aggregate_result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    methods = ("bcos", "integrated_gradients", "saliency", "smoothgrad")
    metrics = (
        ("deletion_advantage", "Deletion advantage", True),
        ("insertion_advantage", "Insertion advantage", True),
        ("sensitivity_n_pearson", "Mean fixed-size Sensitivity-n (PCC)", True),
        ("attribution_seconds_per_sample", "Seconds per sample", False),
    )
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for axis, (metric, title, _higher_better) in zip(axes, metrics):
        means, low, high, labels = [], [], [], []
        for method in methods:
            stats = aggregate_result["groups"][f"bcos/{method}"][metric]
            means.append(stats["mean"])
            low.append(stats["mean"] - stats["ci95_low"])
            high.append(stats["ci95_high"] - stats["mean"])
            labels.append(METHOD_LABELS[method])
        positions = np.arange(len(methods))
        axis.bar(positions, means, color=("#2A6F97", "#61A5C2", "#89C2D9", "#A9D6E5"))
        axis.errorbar(positions, means, yerr=(low, high), fmt="none", color="black", capsize=3)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if metric == "attribution_seconds_per_sample":
            axis.set_yscale("log")
    figure.suptitle("RQ2: explanation methods on 15 B-cos PPO models")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def representative_indices(actions: np.ndarray, count: int) -> list[int]:
    preferred = (2, 3, 5, 0, 1)
    chosen: list[int] = []
    for action in preferred:
        matches = np.flatnonzero(actions == action)
        if len(matches):
            chosen.append(int(matches[len(matches) // 2]))
        if len(chosen) == count:
            return chosen
    for index in range(len(actions)):
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == count:
            break
    return chosen


def save_qualitative_figure(
    model_path: Path,
    dataset_path: Path,
    output: Path,
    device: str,
    samples: int,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    observations, actions, _masks = load_dataset(dataset_path, 10_000)
    indices = representative_indices(actions, samples)
    selected_obs = observations[indices]
    selected_actions = actions[indices]
    model = load_model(model_path, device)
    methods = ("bcos", "saliency", "integrated_gradients", "smoothgrad")
    maps = {
        method: explanations(
            model,
            selected_obs,
            selected_actions,
            batch_size=len(indices),
            method=method,
            integrated_gradients_steps=32,
            smoothgrad_samples=8,
        ).sum(axis=1)
        for method in methods
    }

    figure, axes = plt.subplots(len(indices), 5, figsize=(14, 2.9 * len(indices)))
    axes = np.atleast_2d(axes)
    for row, (obs, action) in enumerate(zip(selected_obs, selected_actions)):
        rgb = np.moveaxis(obs[:3], 0, -1).astype(np.float32) / 255.0
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(f"Action: {ACTION_NAMES[int(action)]}", fontsize=11)
        for column, method in enumerate(methods, start=1):
            attribution = np.abs(maps[method][row])
            scale = max(float(np.percentile(attribution, 99)), 1e-12)
            normalized = np.clip(attribution / scale, 0.0, 1.0)
            axes[row, column].imshow(rgb)
            axes[row, column].imshow(
                normalized,
                cmap="Reds",
                vmin=0,
                vmax=1,
                alpha=np.clip(normalized * 0.8, 0.0, 0.8),
            )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    titles = ("Observation", *(METHOD_LABELS[method] for method in methods))
    for axis, title in zip(axes[0], titles):
        axis.set_title(title)
    figure.suptitle("Representative RQ2 absolute attribution magnitudes")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return {
        "model_path": str(model_path.resolve()),
        "dataset_indices": indices,
        "actions": [ACTION_NAMES[int(action)] for action in selected_actions],
        "normalization": "absolute channel-summed attribution, independently scaled to each map's 99th percentile; red overlay with maximum opacity 80%",
    }


def save_markdown(result: dict[str, Any], path: Path) -> None:
    groups = result["aggregate"]["groups"]
    lines = [
        "# RQ2 Final Statistical Report",
        "",
        "All attribution methods explain the same stored action on the same 256 RGB6 states. "
        "B-cos statistics use three validated PPO configurations and five matched training seeds per configuration.",
        "",
        "## B-cos Model Results",
        "",
        "| Method | Deletion advantage | Insertion advantage | Mean fixed-size Sensitivity-n (PCC) | Infidelity | Completeness error | Seconds/sample |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("bcos", "integrated_gradients", "saliency", "smoothgrad"):
        group = groups[f"bcos/{method}"]
        value = lambda metric: group[metric]["mean"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {value('deletion_advantage'):.3f} | "
            f"{value('insertion_advantage'):.3f} | {value('sensitivity_n_pearson'):.3f} | "
            f"{value('normalized_infidelity'):.3f} | {value('completeness_relative_error'):.3g} | "
            f"{value('attribution_seconds_per_sample'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Intrinsic B-cos attribution is approximately tied with Integrated Gradients on random-normalised deletion, "
            "is stronger on insertion and mean fixed-size Sensitivity-n, has exact output completeness, and is much faster. "
            "The evidence therefore supports an overall faithfulness-and-efficiency advantage, not universal "
            "superiority on every individual metric.",
            "",
            "Error bars in `final_summary.png` are 95% cluster-bootstrap confidence intervals over the five training seeds, retaining all configuration runs within each sampled seed. "
            "Complete machine-readable estimates and paired differences are in `final_report.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--earthy", type=Path, required=True)
    parser.add_argument("--revived", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qualitative-samples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        read_result(args.comparison, "stellar")
        + read_result(args.earthy, "earthy")
        + read_result(args.revived, "revived")
    )
    # Standard controls occur only in comparison.json; the three B-cos profiles are all retained.
    unique = {
        (row["profile"], row["model"], row["architecture"], row["method"]): row
        for row in rows
    }
    rows = list(unique.values())
    aggregate_result = aggregate(rows, args.bootstrap_repeats, args.seed)
    result = {
        "protocol": {
            "dataset": str(args.dataset.resolve()),
            "statistical_unit": "training seed; the three fixed configuration runs are retained as a within-seed cluster",
            "bcos_models": 15,
            "standard_models": 5,
            "bootstrap_repeats": args.bootstrap_repeats,
            "scalar_target": "stored action logit minus mean action logit",
            "deletion_insertion": "AUC advantage over an equal-area random ordering on a rendered-empty-grid substrate",
            "sensitivity_n": "Adapted Sensitivity-n summary: PCC is computed separately at each prespecified subset fraction from 10% to 90%, then the nine fixed-size PCCs are averaged; Spearman is retained only in source evaluations as a secondary diagnostic",
        },
        "aggregate": aggregate_result,
    }
    json_path = args.output_dir / "final_report.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_rows(rows, args.output_dir / "final_per_model.csv")
    save_summary_plot(aggregate_result, args.output_dir / "final_summary.png")
    save_markdown(result, args.output_dir / "FINAL_REPORT.md")

    stellar = [
        row
        for row in rows
        if row["profile"] == "stellar"
        and row["architecture"] == "bcos"
        and row["method"] == "bcos"
    ]
    qualitative_model = resolve_model_path(
        sorted(stellar, key=lambda row: row["model"])[0]["model_path"]
    )
    qualitative = save_qualitative_figure(
        qualitative_model,
        args.dataset,
        args.output_dir / "qualitative_explanations.png",
        args.device,
        args.qualitative_samples,
    )
    result["qualitative_figure"] = qualitative
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved final report: {json_path}")
    print(f"Saved summary figure: {args.output_dir / 'final_summary.png'}")
    print(f"Saved qualitative figure: {args.output_dir / 'qualitative_explanations.png'}")


if __name__ == "__main__":
    main()

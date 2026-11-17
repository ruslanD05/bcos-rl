#!/usr/bin/env python3
"""Generate a compact review figure from the downloaded unified RQ4 artifact."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "rq4-consolidated"
OUTPUT = ROOT / "thesis" / "rq4_unified_review"
THESIS_FIGURES = ROOT / "thesis" / "source" / "Thesis_Template_DE_EN" / "BA_MA_English" / "sections" / "figures"
CONDITIONS = ("pure", "linear", "success_gated_50", "constant")
LABELS = ("B-cos PPO", "Linear", "Success-Gated", "Constant")
COLORS = ("#30343f", "#277da1", "#43aa8b", "#d1495b")
BUDGET = 750_000


def load() -> dict[str, dict[str, list[int]]]:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    values = {condition: {"50": [], "100": []} for condition in CONDITIONS}
    for run in manifest["runs"]:
        condition = run["condition"]
        stats_path = ARTIFACTS / "runs" / run["wandb_run_id"] / "statistics" / "run_statistics.json"
        thresholds = json.loads(stats_path.read_text())["learning_speed"]["first_train_threshold_steps"]
        for threshold in ("50", "100"):
            value = thresholds[threshold]
            # A PPO rollout can log a threshold just after the 750k budget;
            # these are right-censored at the prespecified budget.
            values[condition][threshold].append(BUDGET if value is None or value > BUDGET else value)
    return values


def main() -> None:
    values = load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    rng = np.random.default_rng(20260823)
    for axis, threshold in zip(axes, ("50", "100")):
        for index, (condition, label, color) in enumerate(zip(CONDITIONS, LABELS, COLORS)):
            points = np.asarray(values[condition][threshold]) / 1_000
            jitter = rng.uniform(-0.10, 0.10, len(points))
            censored = points >= BUDGET / 1_000
            axis.scatter(index + jitter[~censored], points[~censored], s=48, color=color, zorder=3)
            axis.scatter(index + jitter[censored], points[censored], s=52, marker="x", linewidths=1.8, color=color, zorder=3)
            axis.errorbar(index, points.mean(), yerr=points.std(ddof=1), color="black", capsize=5, linewidth=1.4, zorder=4)
            axis.scatter(index, points.mean(), s=42, color="white", edgecolor="black", zorder=5)
        axis.axhline(BUDGET / 1_000, color="#777777", linestyle="--", linewidth=1)
        axis.text(3.48, BUDGET / 1_000 + 9, "750k censoring limit", ha="right", va="bottom", fontsize=8, color="#555555")
        axis.set_title(f"First {threshold}% training-success threshold")
        axis.set_xticks(range(4), LABELS, rotation=20, ha="right")
        axis.set_xlim(-0.45, 3.55)
        axis.set_ylim(0, 790)
        axis.grid(axis="y", alpha=0.28)
    axes[0].set_ylabel("Environment steps (thousands)")
    fig.suptitle("Unified RQ4: raw seed thresholds; open circles show mean $\\pm$ sample SD", y=1.02, fontsize=11)
    fig.tight_layout()
    THESIS_FIGURES.mkdir(parents=True, exist_ok=True)
    for directory in (OUTPUT, THESIS_FIGURES):
        for suffix in ("pdf", "png"):
            fig.savefig(directory / f"rq4_unified_thresholds.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    fig, axis = plt.subplots(figsize=(7.4, 4.3))
    for condition, label, color in zip(CONDITIONS, LABELS, COLORS):
        series = []
        for run in manifest["runs"]:
            if run["condition"] != condition:
                continue
            data = np.load(ARTIFACTS / "runs" / run["wandb_run_id"] / "statistics" / "evaluations.npz")
            series.append(np.asarray(data["successes"], dtype=float).mean(axis=1))
            steps = np.asarray(data["timesteps"], dtype=float)
        values = np.asarray(series)
        axis.plot(steps / 1_000, values.mean(axis=0), label=label, color=color, linewidth=2)
        axis.fill_between(steps / 1_000, values.mean(axis=0) - values.std(axis=0, ddof=1), values.mean(axis=0) + values.std(axis=0, ddof=1), color=color, alpha=0.16)
    axis.set(xlabel="Environment steps (thousands)", ylabel="Deterministic evaluation success", xlim=(0, 750), ylim=(-0.02, 1.02), title="Unified RQ4 evaluation-success trajectories")
    axis.grid(alpha=0.28)
    axis.legend(loc="lower right")
    fig.tight_layout()
    for directory in (OUTPUT, THESIS_FIGURES):
        for suffix in ("pdf", "png"):
            fig.savefig(directory / f"rq4_unified_learning_curves.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(OUTPUT / "rq4_unified_thresholds.pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the final RQ1 thesis figures and compact result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STANDARD = "#2878B5"
BCOS = "#D95319"
POINT = "#303030"
BOOTSTRAP_SAMPLES = 10_000
RNG_SEED = 20260812


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    return tuple(np.percentile(draws.mean(axis=1), [2.5, 97.5]))


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_ppo_learning_curves(history: pd.DataFrame, output_dir: Path) -> None:
    data = history[history["analysis"] == "ppo_architecture"].copy()
    labels = {"standard": "Standard PPO", "bcos_b2.3": "B-cos PPO"}
    colors = {"standard": STANDARD, "bcos_b2.3": BCOS}
    metrics = [
        ("eval/success_rate", "Evaluation success rate", (0.0, 1.03)),
        ("eval/mean_reward", "Evaluation return", None),
    ]
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for variant in ("standard", "bcos_b2.3"):
            subset = data[data["variant"] == variant]
            rows = []
            for step, group in subset.groupby("global_step"):
                values = group[metric].dropna().to_numpy(dtype=float)
                low, high = bootstrap_ci(values, rng)
                rows.append((float(step), values.mean(), low, high))
            curve = np.asarray(sorted(rows), dtype=float)
            ax.plot(curve[:, 0], curve[:, 1], color=colors[variant], lw=1.8, label=labels[variant])
            ax.fill_between(curve[:, 0], curve[:, 2], curve[:, 3], color=colors[variant], alpha=0.18)
        ax.set_xlabel("Environment transitions")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, output_dir, "rq1_ppo_learning_curves")


def plot_b_power(export_dir: Path, output_dir: Path) -> None:
    runs = pd.read_csv(export_dir / "b_power_per_run_learning_metrics.csv")
    rng = np.random.default_rng(RNG_SEED)
    metrics = [
        ("final_success", "Final success rate", (0.0, 1.04)),
        ("learning_curve_auc_normalized", "Normalized success AUC", (0.0, 1.04)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    jitter = np.linspace(-0.065, 0.065, 5)

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        means, lows, highs = [], [], []
        b_values = sorted(runs["B_value"].dropna().unique())
        for b in b_values:
            values = runs.loc[runs["B_value"] == b, metric].dropna().to_numpy(dtype=float)
            offsets = jitter[: len(values)]
            ax.scatter(np.full(len(values), b) + offsets, values, color=POINT, alpha=0.65, s=14, zorder=2)
            low, high = bootstrap_ci(values, rng)
            means.append(values.mean())
            lows.append(low)
            highs.append(high)
        means = np.asarray(means)
        errors = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        ax.errorbar(b_values, means, yerr=errors, color=BCOS, marker="o", ms=4, lw=1.5, capsize=2.5, zorder=3)
        ax.set_xlabel("B-cos power B")
        ax.set_ylabel(ylabel)
        ax.set_xticks(b_values)
        ax.set_ylim(*ylim)

    fig.tight_layout()
    save_figure(fig, output_dir, "rq1_b_power_sensitivity")


def plot_dqn_paired(per_run: pd.DataFrame, output_dir: Path) -> None:
    data = per_run[per_run["project"] == "rq1-minigrid-dqn-architecture-final"].copy()
    pivot = data.pivot(index="seed", columns="variant", values="final_success").sort_index()
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    offsets = np.linspace(-0.035, 0.035, len(pivot))
    for offset, (_, row) in zip(offsets, pivot.iterrows()):
        ax.plot([offset, 1 + offset], [row["standard"], row["bcos"]], color="#A0A0A0", lw=1.0, zorder=1)
        ax.scatter(offset, row["standard"], color=STANDARD, s=28, zorder=2)
        ax.scatter(1 + offset, row["bcos"], color=BCOS, s=28, zorder=2)
    ax.set_xticks([0, 1], ["Standard DQN", "B-cos DQN"])
    ax.set_ylabel("Final success rate")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title("Matched seeds (B-cos-calibrated configuration)")
    fig.tight_layout()
    save_figure(fig, output_dir, "rq1_dqn_paired_success")


def plot_support_learning_curves(history: pd.DataFrame, output_dir: Path) -> None:
    specifications = [
        ("lavacrossing", "eval/success_rate", "LavaCrossing success rate", "bcos_b2.3"),
        ("carracing", "eval/mean_reward", "CarRacing evaluation return", "bcos"),
    ]
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

    for ax, (analysis, metric, ylabel, bcos_variant) in zip(axes, specifications):
        data = history[history["analysis"] == analysis]
        for variant, label, color in (
            ("standard", "Standard PPO", STANDARD),
            (bcos_variant, "B-cos PPO", BCOS),
        ):
            subset = data[data["variant"] == variant]
            rows = []
            for step, group in subset.groupby("global_step"):
                values = group[metric].dropna().to_numpy(dtype=float)
                low, high = bootstrap_ci(values, rng)
                rows.append((float(step), values.mean(), low, high))
            curve = np.asarray(sorted(rows), dtype=float)
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.8, label=label)
            ax.fill_between(curve[:, 0], curve[:, 2], curve[:, 3], color=color, alpha=0.18)
        ax.set_xlabel("Environment transitions")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_ylim(0.0, 1.03)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_dir, "rq1_support_learning_curves")


def mean_sd(values: pd.Series, digits: int = 3) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return f"{values.mean():.{digits}f} $\\pm$ {values.std(ddof=1):.{digits}f}"


def write_latex_table(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    columns = "l" + "c" * (len(frame.columns) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(frame.columns) + " \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(map(str, row)) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def create_tables(per_run: pd.DataFrame, output_dir: Path) -> None:
    ppo = per_run[per_run["project"] == "rq1-minigrid-ppo-architecture-final"]
    rows = []
    for variant, label in (("standard", "Standard PPO"), ("bcos_b2.3", "B-cos PPO")):
        group = ppo[ppo["variant"] == variant]
        rows.append(
            {
                "Model": label,
                "Success": mean_sd(group["final_success"]),
                "Return": mean_sd(group["final_reward"]),
                "Episode length": mean_sd(group["final_episode_length"], 1),
                "Wall time (min)": mean_sd(group["training_wall_time_seconds"] / 60.0, 1),
                "FPS": mean_sd(group["fps"], 1),
            }
        )
    main = pd.DataFrame(rows)
    main.to_csv(output_dir / "rq1_main_results_table.csv", index=False)
    write_latex_table(
        main,
        output_dir / "rq1_main_results_table.tex",
        "DoorKey PPO performance and compute cost over five seeds (mean $\\pm$ sample SD).",
        "tab:rq1-main-results",
    )

    support_rows = []
    projects = [
        ("rq1-lavacrossing-ppo-architecture-final", "LavaCrossing", "bcos_b2.3"),
        ("rq1-carracing-ppo-architecture-final-v2", "CarRacing", "bcos"),
    ]
    for project, environment, bcos_variant in projects:
        data = per_run[per_run["project"] == project]
        for variant, label in (("standard", "Standard PPO"), (bcos_variant, "B-cos PPO")):
            group = data[data["variant"] == variant]
            success = "--" if environment == "CarRacing" else mean_sd(group["final_success"])
            support_rows.append(
                {
                    "Environment": environment,
                    "Model": label,
                    "Success": success,
                    "Return": mean_sd(group["final_reward"], 1),
                    "Episode length": mean_sd(group["final_episode_length"], 1),
                }
            )
    support = pd.DataFrame(support_rows)
    support.to_csv(output_dir / "rq1_support_results_table.csv", index=False)
    write_latex_table(
        support,
        output_dir / "rq1_support_results_table.tex",
        "Support-environment results over five seeds (mean $\\pm$ sample SD). CarRacing has no binary success criterion.",
        "tab:rq1-support-results",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "statistical_export" / "v0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "thesis_outputs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    style()
    per_run = pd.read_csv(args.export_dir / "per_run.csv")
    history = pd.read_csv(args.export_dir / "learning_history.csv")
    plot_ppo_learning_curves(history, args.output_dir)
    plot_b_power(args.export_dir, args.output_dir)
    plot_dqn_paired(per_run, args.output_dir)
    plot_support_learning_curves(history, args.output_dir)
    create_tables(per_run, args.output_dir)
    print(f"Saved RQ1 thesis outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

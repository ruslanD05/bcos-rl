#!/usr/bin/env python3
"""Aggregate all 18 RQ6 sweep paired_summary.json files into a master table + publication plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RUNS_DIR = Path(r"C:\RLProject\bachelors_thesis_ruslan\experiments\rq6_shortcut_suppression\runs_channelwise")
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping from W&B run_id -> (correlation, seed) based on the sweep matrix
RUN_MAP = {
    # 50% correlation
    "pjy9i25x": (0.50, 0), "a5w3lf7t": (0.50, 1), "zcvf69e8": (0.50, 2),
    # 60% correlation
    "cikmsjio": (0.60, 0), "0k58z14s": (0.60, 1), "7yesdlp5": (0.60, 2),
    # 70% correlation
    "k09mrra3": (0.70, 0), "qg1muw9o": (0.70, 1), "dm8xa3ji": (0.70, 2),
    # 80% correlation
    "spoxhchk": (0.80, 0), "kxdqv327": (0.80, 1), "74gyunrw": (0.80, 2),
    # 90% correlation
    "tfwyg23i": (0.90, 0), "osf3kpdv": (0.90, 1), "lyo7krly": (0.90, 2),
    # 100% correlation
    "rv89ibc4": (1.00, 0), "ee50amlx": (1.00, 1), "pj95m60d": (1.00, 2),
}


def load_all() -> pd.DataFrame:
    rows = []
    for run_id, (corr, seed) in RUN_MAP.items():
        summary_path = RUNS_DIR / run_id / "paired_summary.json"
        if not summary_path.exists():
            print(f"WARNING: Missing {summary_path}")
            continue
        with open(summary_path) as f:
            data = json.load(f)

        row = {"run_id": run_id, "correlation": corr, "seed": seed}
        for cond in ("baseline", "suppressed"):
            perf = data[f"{cond}_perf"]
            for mode in ("correlated", "removed", "permuted", "random"):
                row[f"{cond}_{mode}_succ"] = perf[mode]["success_rate"]
                row[f"{cond}_{mode}_len"] = perf[mode]["mean_episode_length"]
                row[f"{cond}_{mode}_rew"] = perf[mode]["mean_reward"]
            row[f"{cond}_patch_ratio"] = data[f"{cond}_patch_ratio"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["correlation", "seed"]).reset_index(drop=True)
    return df


def print_full_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("  RQ6 SHORTCUT CORRELATION SWEEP — FULL RESULTS (18 RUNS)")
    print("=" * 120)
    header = f"{'Corr%':>5} {'Seed':>4} | {'B.Perm%':>7} {'S.Perm%':>7} {'Δ':>6} | {'B.Rem%':>6} {'S.Rem%':>6} | {'B.PLen':>6} {'S.PLen':>6} | {'B.Patch%':>8} {'S.Patch%':>8} {'Red%':>5}"
    print(header)
    print("-" * 120)
    for _, r in df.iterrows():
        bp = r["baseline_permuted_succ"] * 100
        sp = r["suppressed_permuted_succ"] * 100
        delta = sp - bp
        br = r["baseline_removed_succ"] * 100
        sr = r["suppressed_removed_succ"] * 100
        bpl = r["baseline_permuted_len"]
        spl = r["suppressed_permuted_len"]
        bpr = r["baseline_patch_ratio"] * 100
        spr = r["suppressed_patch_ratio"] * 100
        red = ((bpr - spr) / max(bpr, 0.001)) * 100
        print(f"{r['correlation']*100:>5.0f} {r['seed']:>4.0f} | {bp:>7.1f} {sp:>7.1f} {delta:>+6.1f} | {br:>6.1f} {sr:>6.1f} | {bpl:>6.1f} {spl:>6.1f} | {bpr:>8.3f} {spr:>8.3f} {red:>5.1f}")


def print_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 120)
    print("  RQ6 AGGREGATED (MEAN ± STD OVER 3 SEEDS)")
    print("=" * 120)

    agg_rows = []
    for corr in sorted(df["correlation"].unique()):
        sub = df[df["correlation"] == corr]
        row = {"correlation": corr}
        for key in ["baseline_permuted_succ", "suppressed_permuted_succ",
                     "baseline_removed_succ", "suppressed_removed_succ",
                     "baseline_permuted_len", "suppressed_permuted_len",
                     "baseline_correlated_succ", "suppressed_correlated_succ",
                     "baseline_correlated_len", "suppressed_correlated_len",
                     "baseline_patch_ratio", "suppressed_patch_ratio"]:
            row[f"{key}_mean"] = sub[key].mean()
            row[f"{key}_std"] = sub[key].std()
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)

    header = f"{'Corr%':>5} | {'B.Perm Succ':>18} | {'S.Perm Succ':>18} | {'Δ Perm':>8} | {'B.Perm Len':>12} {'S.Perm Len':>12} | {'B.Patch%':>12} {'S.Patch%':>12} {'Red%':>6}"
    print(header)
    print("-" * 140)
    for _, r in agg.iterrows():
        bp_m, bp_s = r["baseline_permuted_succ_mean"] * 100, r["baseline_permuted_succ_std"] * 100
        sp_m, sp_s = r["suppressed_permuted_succ_mean"] * 100, r["suppressed_permuted_succ_std"] * 100
        delta = sp_m - bp_m
        bpl_m = r["baseline_permuted_len_mean"]
        spl_m = r["suppressed_permuted_len_mean"]
        bpr_m = r["baseline_patch_ratio_mean"] * 100
        spr_m = r["suppressed_patch_ratio_mean"] * 100
        red = ((bpr_m - spr_m) / max(bpr_m, 0.001)) * 100
        print(f"{r['correlation']*100:>5.0f} | {bp_m:>6.1f} ± {bp_s:>4.1f}%     | {sp_m:>6.1f} ± {sp_s:>4.1f}%     | {delta:>+7.1f} | {bpl_m:>10.1f}  {spl_m:>10.1f}  | {bpr_m:>10.3f}  {spr_m:>10.3f}  {red:>5.1f}")

    return agg


def plot_master_curves(agg: pd.DataFrame) -> None:
    corrs = agg["correlation"].values * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle("RQ6: Shortcut Correlation Sweep — Baseline vs. Explanation-Suppressed",
                 fontsize=14, fontweight="bold", y=0.98)

    c_base = "#d95f02"
    c_supp = "#1b9e77"

    # Panel 1: Permuted Success Rate
    ax = axes[0, 0]
    ax.errorbar(corrs, agg["baseline_permuted_succ_mean"] * 100,
                yerr=agg["baseline_permuted_succ_std"] * 100,
                fmt="o--", color=c_base, label="Baseline (Pure B-cos)", capsize=4, linewidth=2, markersize=7)
    ax.errorbar(corrs, agg["suppressed_permuted_succ_mean"] * 100,
                yerr=agg["suppressed_permuted_succ_std"] * 100,
                fmt="s-", color=c_supp, label=r"Suppressed ($\mathcal{L}_{expl}$)", capsize=4, linewidth=2, markersize=7)
    ax.set_title("Adversarial Robustness (Permuted Cue)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Shortcut Reliability (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Removed Success Rate
    ax = axes[0, 1]
    ax.errorbar(corrs, agg["baseline_removed_succ_mean"] * 100,
                yerr=agg["baseline_removed_succ_std"] * 100,
                fmt="o--", color=c_base, label="Baseline (Pure B-cos)", capsize=4, linewidth=2, markersize=7)
    ax.errorbar(corrs, agg["suppressed_removed_succ_mean"] * 100,
                yerr=agg["suppressed_removed_succ_std"] * 100,
                fmt="s-", color=c_supp, label=r"Suppressed ($\mathcal{L}_{expl}$)", capsize=4, linewidth=2, markersize=7)
    ax.set_title("Removal Invariance (Grey Wall)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Shortcut Reliability (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Episode Length under Permuted
    ax = axes[1, 0]
    ax.plot(corrs, agg["baseline_permuted_len_mean"], "o--", color=c_base,
            label="Baseline (Pure B-cos)", linewidth=2, markersize=7)
    ax.plot(corrs, agg["suppressed_permuted_len_mean"], "s-", color=c_supp,
            label=r"Suppressed ($\mathcal{L}_{expl}$)", linewidth=2, markersize=7)
    ax.set_title("Episode Length Under Adversarial Cue (Lower = Faster)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Shortcut Reliability (%)")
    ax.set_ylabel("Mean Steps")
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Patch Attribution Ratio
    ax = axes[1, 1]
    ax.plot(corrs, agg["baseline_patch_ratio_mean"] * 100, "o--", color=c_base,
            label="Baseline (Pure B-cos)", linewidth=2, markersize=7)
    ax.plot(corrs, agg["suppressed_patch_ratio_mean"] * 100, "s-", color=c_supp,
            label=r"Suppressed ($\mathcal{L}_{expl}$)", linewidth=2, markersize=7)
    ax.set_title("Shortcut Attribution Energy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Shortcut Reliability (%)")
    ax.set_ylabel("Attribution on Top Row (%)")
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = OUT_DIR / "rq6_correlation_sweep_curves.png"
    pdf_path = OUT_DIR / "rq6_correlation_sweep_curves.pdf"
    fig.savefig(str(png_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    df = load_all()

    # Save raw CSV
    csv_path = OUT_DIR / "rq6_correlation_sweep_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw CSV: {csv_path}")

    print_full_table(df)
    agg = print_aggregated(df)

    # Save aggregated CSV
    agg_csv = OUT_DIR / "rq6_correlation_sweep_aggregated.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"\nSaved aggregated CSV: {agg_csv}")

    plot_master_curves(agg)


if __name__ == "__main__":
    main()

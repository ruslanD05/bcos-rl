#!/usr/bin/env python3
"""Generate publication-ready thesis figures for RQ4 (Explanation Supervision Timing)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "rq4_explanation_curriculum" / "analysis"
OUTPUT_DIRS = [
    DATA_DIR,
    ROOT / "thesis_drafts" / "figures",
    ROOT / "Thesis_Template_DE_EN" / "BA_MA_English" / "sections" / "figures",
]

# Consistent styling & palette
CONDITIONS = ["pure", "constant", "linear", "success-gated"]
CONDITION_LABELS = {
    "pure": "Pure PPO",
    "constant": "Constant",
    "linear": "Linear",
    "success-gated": "Success-Gated",
}
COLORS = {
    "pure": "#2B2D42",          # Dark Slate / Charcoal
    "constant": "#D90429",      # Crimson Red
    "linear": "#0077B6",        # Ocean Blue
    "success-gated": "#2A9D8F", # Persian Green / Emerald
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {stem}.pdf and {stem}.png")


def plot_learning_curves(history: pd.DataFrame) -> None:
    """Plot evaluation and rolling rollout success curves with bootstrap CIs."""
    fig, (ax_eval, ax_roll) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    # Common evaluation grid
    eval_steps = np.arange(50_000, 1_000_001, 50_000)

    for cond in CONDITIONS:
        cond_data = history[history["condition"] == cond]
        seeds = cond_data["seed"].unique()
        
        # 1. Evaluation Success
        eval_matrix = []
        for seed in seeds:
            s_data = cond_data[cond_data["seed"] == seed].dropna(subset=["evaluation_success"])
            if len(s_data) > 0:
                s_data = s_data.sort_values("global_step")
                # Interp to standard grid
                vals = np.interp(eval_steps, s_data["global_step"], s_data["evaluation_success"])
                eval_matrix.append(vals)
        
        if eval_matrix:
            eval_mat = np.array(eval_matrix)
            mean_eval = np.mean(eval_mat, axis=0)
            
            # Bootstrap 95% CI
            n_boot = 2000
            boot_means = []
            rng = np.random.default_rng(20250308)
            for _ in range(n_boot):
                idx = rng.choice(len(eval_matrix), size=len(eval_matrix), replace=True)
                boot_means.append(np.mean(eval_mat[idx], axis=0))
            ci_low = np.percentile(boot_means, 2.5, axis=0)
            ci_high = np.percentile(boot_means, 97.5, axis=0)
            
            ax_eval.plot(eval_steps / 1e3, mean_eval, label=CONDITION_LABELS[cond], color=COLORS[cond], lw=2)
            ax_eval.fill_between(eval_steps / 1e3, ci_low, ci_high, color=COLORS[cond], alpha=0.15)

        # 2. Rolling Rollout Success
        roll_steps = np.arange(20_000, 1_000_001, 20_000)
        roll_matrix = []
        for seed in seeds:
            s_data = cond_data[cond_data["seed"] == seed].dropna(subset=["rolling_rollout_success"])
            if len(s_data) > 0:
                s_data = s_data.sort_values("global_step")
                vals = np.interp(roll_steps, s_data["global_step"], s_data["rolling_rollout_success"])
                roll_matrix.append(vals)
        
        if roll_matrix:
            roll_mat = np.array(roll_matrix)
            mean_roll = np.mean(roll_mat, axis=0)
            boot_means = []
            rng = np.random.default_rng(20250308)
            for _ in range(n_boot):
                idx = rng.choice(len(roll_matrix), size=len(roll_matrix), replace=True)
                boot_means.append(np.mean(roll_mat[idx], axis=0))
            ci_low = np.percentile(boot_means, 2.5, axis=0)
            ci_high = np.percentile(boot_means, 97.5, axis=0)
            
            ax_roll.plot(roll_steps / 1e3, mean_roll, label=CONDITION_LABELS[cond], color=COLORS[cond], lw=2)
            ax_roll.fill_between(roll_steps / 1e3, ci_low, ci_high, color=COLORS[cond], alpha=0.15)

    ax_eval.set_title("(a) Evaluation Success Rate")
    ax_eval.set_xlabel("Environment Steps (Thousands)")
    ax_eval.set_ylabel("Success Rate")
    ax_eval.set_ylim(-0.02, 1.02)
    ax_eval.set_xlim(0, 1000)
    ax_eval.legend(loc="upper left", frameon=True)

    ax_roll.set_title("(b) Rolling Rollout Success Rate (Window=100)")
    ax_roll.set_xlabel("Environment Steps (Thousands)")
    ax_roll.set_xlim(0, 1000)

    fig.tight_layout()
    save_figure(fig, "figure_success_learning_curves")


def plot_steps_to_threshold(thresholds: pd.DataFrame) -> None:
    """Plot environment transitions to 50%, 80%, 90%, and 95% success thresholds."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True)
    target_threshs = [0.5, 0.8, 0.9, 0.95]
    
    for ax, th_val in zip(axes, target_threshs):
        th_data = thresholds[(thresholds["threshold"] == th_val) & (thresholds["success_source"] == "rollout_rolling")]
        
        for x_idx, cond in enumerate(CONDITIONS):
            c_data = th_data[th_data["condition"] == cond]
            reached = c_data[c_data["reached"] == True]
            censored = c_data[c_data["censored"] == True]
            
            # Scatter reached points
            if len(reached) > 0:
                y_vals = reached["first_step"].values / 1e3
                jitter = np.random.default_rng(x_idx).uniform(-0.12, 0.12, size=len(y_vals))
                ax.scatter(x_idx + jitter, y_vals, color=COLORS[cond], s=40, zorder=3, edgecolors="none")
                
                # Median bar
                med = np.median(y_vals)
                ax.plot([x_idx - 0.25, x_idx + 0.25], [med, med], color="black", lw=2.5, zorder=4)
            
            # Mark censored points with 'x' at budget (1000k)
            if len(censored) > 0:
                y_cens = np.full(len(censored), 1003.52)
                jitter = np.random.default_rng(x_idx + 10).uniform(-0.12, 0.12, size=len(y_cens))
                ax.scatter(x_idx + jitter, y_cens, color=COLORS[cond], marker="x", s=55, lw=2, zorder=3)
            
            # Annotation n/5
            n_reached = len(reached)
            ax.text(x_idx, 1070, f"{n_reached}/5", ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

        ax.set_title(f"{int(th_val*100)}% Success")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(["Pure", "Const.", "Linear", "Gated"], rotation=25)
        ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
        ax.set_ylim(0, 1150)

    axes[0].set_ylabel("Environment Steps (Thousands)")
    fig.suptitle(r"Steps to Rolling-Rollout Success Thresholds (Horizontal bars: observed medians; $\times$ denotes censored)", fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure_steps_to_threshold")


def plot_localization_vs_success(per_run: pd.DataFrame) -> None:
    """Plot Final Target-Energy Localization Ratio vs Policy Success."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for cond in ["constant", "linear", "success-gated"]:
        c_data = per_run[per_run["condition"] == cond]
        x = c_data["final_localization_target_energy_ratio"].values
        y = c_data["final_evaluation_success"].values
        seeds = c_data["seed"].values
        
        ax.scatter(x, y, color=COLORS[cond], label=CONDITION_LABELS[cond], s=65, zorder=3, edgecolors="white", lw=0.8)
        for xi, yi, s in zip(x, y, seeds):
            ax.annotate(f"{s}", (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8.5, color="#333333")

    ax.set_xlabel(r"Final Target-Energy Localization Ratio $P / (P + N)$")
    ax.set_ylabel("Final Evaluation Success Rate")
    ax.set_title("Final Explanation Localization vs. Policy Success")
    ax.set_ylim(0.40, 1.02)
    ax.set_xlim(0.988, 0.997)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    save_figure(fig, "figure_localization_vs_success")


def plot_gradient_ratio(history: pd.DataFrame) -> None:
    """Plot actual explanation-to-PPO actor gradient ratio over training."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    
    steps_grid = np.arange(50_000, 1_000_001, 20_000)

    for cond in ["constant", "linear", "success-gated"]:
        cond_data = history[history["condition"] == cond]
        seeds = cond_data["seed"].unique()
        matrix = []
        for seed in seeds:
            s_data = cond_data[cond_data["seed"] == seed].dropna(subset=["actual_explanation_to_ppo_actor_gradient_ratio"])
            if len(s_data) > 0:
                s_data = s_data.sort_values("global_step")
                vals = np.interp(steps_grid, s_data["global_step"], s_data["actual_explanation_to_ppo_actor_gradient_ratio"])
                matrix.append(vals)
        
        if matrix:
            mat = np.array(matrix)
            mean_val = np.mean(mat, axis=0)
            ax.plot(steps_grid / 1e3, mean_val, label=CONDITION_LABELS[cond], color=COLORS[cond], lw=2)

    # 10% cap reference line
    ax.axhline(0.10, color="gray", linestyle=":", label=r"10% Cap ($\gamma_{\max}=0.10$)", lw=1.5)

    ax.set_title("Explanation-to-PPO Actor Gradient Ratio throughout Training")
    ax.set_xlabel("Environment Steps (Thousands)")
    ax.set_ylabel(r"Gradient Norm Ratio $\|\nabla_{\theta}\mathcal{L}_{\mathrm{expl}}\| \,/\, \|\nabla_{\theta}\mathcal{L}_{\mathrm{PPO}}\|$")
    ax.set_ylim(0.0, 0.12)
    ax.set_xlim(0, 1000)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    save_figure(fig, "figure_explanation_gradient_ratio")


def main() -> None:
    setup_style()
    history = pd.read_csv(DATA_DIR / "learning_history.csv")
    thresholds = pd.read_csv(DATA_DIR / "threshold_steps.csv")
    per_run = pd.read_csv(DATA_DIR / "per_run.csv")

    print("Generating RQ4 figures...")
    plot_learning_curves(history)
    plot_steps_to_threshold(thresholds)
    plot_localization_vs_success(per_run)
    plot_gradient_ratio(history)
    print("All figures successfully generated and saved.")


if __name__ == "__main__":
    main()

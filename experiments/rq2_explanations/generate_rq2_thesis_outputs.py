#!/usr/bin/env python3
"""Generate thesis tables and qualitative figures for RQ2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "minigrid_part"))

from bcos_rl_generic.bcosPPO import BcosPPO
from car_racing_part.train_carracing_ppo import ACTION_NAMES as CAR_ACTION_NAMES, build_env
from car_racing_part.visualize_carracing import record_episode
from minigrid_part.aggregate_rq2_explanations import ACTION_NAMES, METHOD_LABELS
from minigrid_part.evaluate_explanation_faithfulness import explanations, load_dataset
from minigrid_part.evaluate_explanation_metrics import load_model


METHODS = ("bcos", "saliency", "integrated_gradients", "smoothgrad")
ARCHITECTURE_METHODS = {
    "standard": ("saliency", "integrated_gradients", "smoothgrad"),
    "bcos": METHODS,
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def metric_cell(stats: dict, digits: int = 3) -> str:
    half_width = (stats["ci95_high"] - stats["ci95_low"]) / 2.0
    return f"{stats['mean']:.{digits}f} $\\pm$ {half_width:.{digits}f}"


def write_table(report: dict, output_dir: Path) -> None:
    groups = report["aggregate"]["groups"]
    metric_columns = {
        "Deletion": ("deletion_advantage", "max"),
        "Insertion": ("insertion_advantage", "max"),
        "Mean Sens.-n": ("sensitivity_n_pearson", "max"),
        "Infidelity": ("normalized_infidelity", "min"),
        "Completeness": ("completeness_relative_error", "min"),
        "Seconds/sample": ("attribution_seconds_per_sample", "min"),
    }
    rows = []
    for architecture in ("standard", "bcos"):
        for method in ARCHITECTURE_METHODS[architecture]:
            group = groups[f"{architecture}/{method}"]
            rows.append(
                {
                    "Architecture": "Standard" if architecture == "standard" else "B-cos",
                    "Method": METHOD_LABELS[method],
                    "Deletion": metric_cell(group["deletion_advantage"]),
                    "Insertion": metric_cell(group["insertion_advantage"]),
                    "Mean Sens.-n": metric_cell(group["sensitivity_n_pearson"]),
                    "Infidelity": metric_cell(group["normalized_infidelity"]),
                    "Completeness": f"{group['completeness_relative_error']['mean']:.2e}",
                    "Seconds/sample": f"{group['attribution_seconds_per_sample']['mean']:.6f}",
                }
            )

    import csv

    csv_path = output_dir / "rq2_explanation_methods_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{3.5pt}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{llcccccc}",
        "        \\toprule",
        "        Architecture & Method & Del. advantage $\\uparrow$ & Ins. advantage $\\uparrow$ & Mean Sens.-$n$ (PCC) $\\uparrow$ & Infidelity $\\downarrow$ & Completeness $\\downarrow$ & Seconds/sample $\\downarrow$ \\\\",
        "        \\midrule",
    ]
    for index, row in enumerate(rows):
        if index == 3:
            lines.append("        \\midrule")
        architecture = "standard" if row["Architecture"] == "Standard" else "bcos"
        method = next(
            key for key, label in METHOD_LABELS.items() if label == row["Method"]
        )
        cells = []
        for column, display_value in row.items():
            if column not in metric_columns:
                cells.append(str(display_value))
                continue
            metric, direction = metric_columns[column]
            candidates = [
                groups[f"{architecture}/{candidate}"][metric]["mean"]
                for candidate in ARCHITECTURE_METHODS[architecture]
            ]
            best = max(candidates) if direction == "max" else min(candidates)
            current = groups[f"{architecture}/{method}"][metric]["mean"]
            value = str(display_value)
            cells.append(f"\\textbf{{{value}}}" if np.isclose(current, best) else value)
        lines.append(
            "        "
            + " & ".join(cells)
            + " \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            "    \\caption{RQ2 explanation results. Deletion and insertion are AUC advantages over equal-area random perturbations. The adapted Sensitivity-$n$ summary is the mean of Pearson correlation coefficients (PCCs) computed separately at subset fractions from 10\\% to 90\\%. Values for deletion, insertion, mean Sensitivity-$n$, and infidelity are means $\\pm$ half-widths of the 95\\% cluster-bootstrap intervals. Higher is better for deletion, insertion, and mean Sensitivity-$n$; lower is better for infidelity, completeness error, and runtime.}",
            "    \\label{tab:rq2_explanation_methods}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "rq2_explanation_methods_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def absolute_spatial(attribution: np.ndarray) -> np.ndarray:
    return np.abs(attribution.sum(axis=0))


def overlay(axis, rgb: np.ndarray, attribution: np.ndarray | None, title: str) -> None:
    if attribution is None:
        axis.imshow(rgb)
    else:
        scale = max(float(np.percentile(attribution, 99)), 1e-12)
        normalized = np.clip(attribution / scale, 0.0, 1.0)
        axis.imshow(rgb)
        axis.imshow(
            normalized,
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=np.clip(normalized * 0.8, 0.0, 0.8),
        )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def select_minigrid_indices(actions: np.ndarray) -> list[int]:
    selected = []
    for action in (2, 3, 5):  # forward, pickup, toggle
        matches = np.flatnonzero(actions == action)
        if len(matches):
            selected.append(int(matches[len(matches) // 2]))
    if len(selected) < 3:
        selected.extend(index for index in range(len(actions)) if index not in selected)
    return selected[:3]


def minigrid_models(per_model_csv: Path) -> tuple[Path, Path]:
    import csv

    with per_model_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    standard = next(
        Path(row["model_path"])
        for row in rows
        if row["profile"] == "stellar"
        and row["architecture"] == "standard"
        and row["model"] == "standard_s0"
    )
    bcos = next(
        Path(row["model_path"])
        for row in rows
        if row["profile"] == "stellar"
        and row["architecture"] == "bcos"
        and row["model"] == "bcos_s0"
    )
    return standard, bcos


def minigrid_figures(
    dataset: Path, per_model_csv: Path, output_dir: Path, device: str
) -> None:
    observations, actions, _ = load_dataset(dataset, 10_000)
    indices = select_minigrid_indices(actions)
    standard_path, bcos_path = minigrid_models(per_model_csv)
    models = {
        "Standard PPO": load_model(standard_path, device),
        "B-cos PPO": load_model(bcos_path, device),
    }

    selected_obs = observations[indices]
    selected_actions = actions[indices]
    maps = {}
    for model_label, model in models.items():
        methods = ARCHITECTURE_METHODS["standard" if model_label.startswith("Standard") else "bcos"]
        maps[model_label] = {
            method: explanations(
                model,
                selected_obs,
                selected_actions,
                batch_size=len(indices),
                method=method,
                integrated_gradients_steps=32,
                smoothgrad_samples=8,
            )
            for method in methods
        }

    columns = ("Observation", "Intrinsic B-cos", "Saliency", "Integrated Gradients", "SmoothGrad")
    for sample, (obs, action, dataset_index) in enumerate(
        zip(selected_obs, selected_actions, indices)
    ):
        rgb = np.moveaxis(obs[:3], 0, -1).astype(np.float32) / 255.0
        fig, axes = plt.subplots(2, 5, figsize=(10.4, 4.25))
        for row, model_label in enumerate(("Standard PPO", "B-cos PPO")):
            overlay(axes[row, 0], rgb, None, columns[0] if row == 0 else "")
            axes[row, 0].set_ylabel(model_label)
            for column, method in enumerate(METHODS, start=1):
                if method not in maps[model_label]:
                    axes[row, column].axis("off")
                    axes[row, column].text(0.5, 0.5, "Not available", ha="center", va="center")
                    if row == 0:
                        axes[row, column].set_title(columns[column])
                    continue
                attribution = absolute_spatial(maps[model_label][method][sample])
                overlay(
                    axes[row, column],
                    rgb,
                    attribution,
                    columns[column] if row == 0 else "",
                )
        fig.suptitle(
            f"MiniGrid action: {ACTION_NAMES[int(action)]} (shared state {dataset_index})"
        )
        fig.tight_layout()
        save(fig, output_dir, f"rq2_minigrid_{ACTION_NAMES[int(action)]}")


def select_carracing_frames(frames, count: int = 3) -> list[int]:
    candidates = np.linspace(60, max(len(frames) - 1, 60), 12, dtype=int)
    selected = []
    used_actions = set()
    for index in candidates:
        index = min(int(index), len(frames) - 1)
        action = frames[index].action
        if action not in used_actions:
            selected.append(index)
            used_actions.add(action)
        if len(selected) == count:
            break
    for index in candidates:
        index = min(int(index), len(frames) - 1)
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def carracing_figure(model_path: Path, output_dir: Path, device: str) -> None:
    env = build_env(
        env_id="CarRacing-v3", n_envs=1, seed=10_000, use_subprocesses=False
    )
    try:
        model = BcosPPO.load(model_path, env=env, device=device)
        frames = record_episode(model, env, deterministic=True, max_steps=1_000)
    finally:
        env.close()
    indices = select_carracing_frames(frames)
    observations = np.stack([frames[index].observation for index in indices])
    actions = np.asarray([frames[index].action for index in indices], dtype=np.int64)
    maps = {
        method: explanations(
            model,
            observations,
            actions,
            batch_size=len(indices),
            method=method,
            integrated_gradients_steps=32,
            smoothgrad_samples=8,
        )
        for method in METHODS
    }

    columns = ("Observation", "Intrinsic B-cos", "Saliency", "Integrated Gradients", "SmoothGrad")
    fig, axes = plt.subplots(len(indices), 5, figsize=(10.4, 6.2))
    for row, (observation, action, frame_index) in enumerate(
        zip(observations, actions, indices)
    ):
        rgb = np.moveaxis(observation[:3], 0, -1).astype(np.float32) / 255.0
        overlay(axes[row, 0], rgb, None, columns[0] if row == 0 else "")
        axes[row, 0].set_ylabel(f"{CAR_ACTION_NAMES[int(action)]}\nframe {frame_index}")
        for column, method in enumerate(METHODS, start=1):
            overlay(
                axes[row, column],
                rgb,
                absolute_spatial(maps[method][row]),
                columns[column] if row == 0 else "",
            )
    fig.suptitle("CarRacing explanations on one B-cos PPO policy")
    fig.tight_layout()
    save(fig, output_dir, "rq2_carracing_methods")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rq2-dir", type=Path, default=ROOT / "experiments" / "rq2_explanations"
    )
    parser.add_argument(
        "--carracing-model",
        type=Path,
        default=ROOT
        / "experiments"
        / "carracing_explanation_mask_comparison"
        / "models"
        / "pure_bcos_ppo_1269760.zip",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.rq2_dir / "thesis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    style()
    report = json.loads((args.rq2_dir / "final" / "final_report.json").read_text())
    write_table(report, output_dir)
    if args.table_only:
        print(f"Saved RQ2 thesis table to: {output_dir.resolve()}")
        return
    minigrid_figures(
        args.rq2_dir / "shared_states_rgb6.npz",
        args.rq2_dir / "final" / "final_per_model.csv",
        output_dir,
        args.device,
    )
    carracing_figure(args.carracing_model, output_dir, args.device)
    print(f"Saved RQ2 thesis outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

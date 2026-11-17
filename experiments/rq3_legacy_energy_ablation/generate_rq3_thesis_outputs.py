#!/usr/bin/env python3
"""Generate final RQ3 thesis tables, figures, and provenance metadata."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image


BRANCHES = ("pure", "sequential", "local", "flipped")
LABELS = {
    "pure": "Pure PPO",
    "sequential": "Sequential",
    "local": "Local action",
    "flipped": "Flipped",
}
COLORS = {
    "pure": "#2878B5",
    "sequential": "#D95319",
    "local": "#2A9D55",
    "flipped": "#7B4AB5",
}
MASK_METRICS = ("sequential_mask_ratio", "local_mask_ratio", "flipped_mask_ratio")
MASK_LABELS = ("Sequential mask", "Local-action mask", "Flipped mask")


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
        }
    )


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pm(stats: dict, digits: int = 3) -> str:
    half_width = (float(stats["ci95_high"]) - float(stats["ci95_low"])) / 2.0
    return f"{float(stats['mean']):.{digits}f} $\\pm$ {half_width:.{digits}f}"


def write_results_table(summary: dict, output_dir: Path) -> None:
    rows = []
    for branch in BRANCHES:
        metrics = summary["branches"][branch]
        rows.append(
            {
                "Branch": LABELS[branch],
                "Success": pm(metrics["policy_success"]),
                "Return": pm(metrics["policy_reward"]),
                "Episode length": pm(metrics["policy_episode_length"], 2),
                "Deletion": pm(metrics["deletion_advantage"]),
                "Insertion": pm(metrics["insertion_advantage"]),
                "Sensitivity-$n$": pm(metrics["sensitivity_n"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "rq3_results_table.csv", index=False)

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{lcccccc}",
        "        \\toprule",
        "        Branch & Success & Return & Episode length & Deletion & Insertion & Sensitivity-$n$ \\\\",
        "        \\midrule",
    ]
    for row in rows:
        lines.append("        " + " & ".join(row.values()) + " \\\\")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            "    \\caption{RQ3 policy quality and attribution faithfulness over five matched seeds. Values are means $\\pm$ half-widths of 95\\% bootstrap confidence intervals. Higher is better for all reported faithfulness metrics.}",
            "    \\label{tab:rq3-results}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "rq3_results_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_localization_table(summary: dict, output_dir: Path) -> None:
    rows = []
    for branch in BRANCHES:
        metrics = summary["branches"][branch]
        rows.append(
            {
                "Branch": LABELS[branch],
                "Sequential mask": pm(metrics["sequential_mask_ratio"]),
                "Local-action mask": pm(metrics["local_mask_ratio"]),
                "Flipped mask": pm(metrics["flipped_mask_ratio"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "rq3_localization_table.csv", index=False)

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\small",
        "    \\begin{tabular}{lccc}",
        "        \\toprule",
        "        Branch & Sequential mask & Local-action mask & Flipped mask \\\\",
        "        \\midrule",
    ]
    for row in rows:
        lines.append("        " + " & ".join(row.values()) + " \\\\")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\caption{Cross-mask attribution-energy ratios over five matched seeds. Values are means $\\pm$ half-widths of 95\\% bootstrap confidence intervals. Each row evaluates one trained branch under all three masks.}",
            "    \\label{tab:rq3-localization}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "rq3_localization_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_paired_differences(summary: dict, output_dir: Path) -> None:
    rows = []
    own_mask = {
        "sequential": "sequential_mask_ratio",
        "local": "local_mask_ratio",
        "flipped": "flipped_mask_ratio",
    }
    for branch in ("sequential", "local", "flipped"):
        metrics = summary["paired_differences_vs_pure"][branch]
        for metric in (
            "policy_success",
            own_mask[branch],
            "deletion_advantage",
            "insertion_advantage",
        ):
            stats = metrics[metric]
            rows.append(
                {
                    "branch": branch,
                    "metric": metric,
                    "mean_difference": float(stats["mean"]),
                    "ci95_low": float(stats["ci95_low"]),
                    "ci95_high": float(stats["ci95_high"]),
                    "n": int(stats["n"]),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "rq3_paired_differences_vs_pure.csv", index=False
    )


def write_result_summary(summary: dict, output_dir: Path) -> None:
    branches = summary["branches"]
    paired = summary["paired_differences_vs_pure"]

    def ci(branch: str, metric: str) -> str:
        value = paired[branch][metric]
        return (
            f"{value['mean']:+.3f} "
            f"[{value['ci95_low']:+.3f}, {value['ci95_high']:+.3f}]"
        )

    text = f"""# Strict-mask RQ3 results

Statistical unit: independent source/training seed, n={summary['seeds']}.
Every policy was evaluated over 100 stochastic episodes. Explanation metrics
use 256 shared state-action samples per seed.

## Policy quality

- Pure success: {branches['pure']['policy_success']['mean']:.3f}
- Sequential success: {branches['sequential']['policy_success']['mean']:.3f}
- Local success: {branches['local']['policy_success']['mean']:.3f}
- Flipped success: {branches['flipped']['policy_success']['mean']:.3f}

## Explanation control

- Sequential alignment: {branches['pure']['sequential_mask_ratio']['mean']:.3f}
  -> {branches['sequential']['sequential_mask_ratio']['mean']:.3f};
  paired difference {ci('sequential', 'sequential_mask_ratio')}.
- Local alignment: {branches['pure']['local_mask_ratio']['mean']:.3f}
  -> {branches['local']['local_mask_ratio']['mean']:.3f};
  paired difference {ci('local', 'local_mask_ratio')}.
- Flipped alignment: {branches['pure']['flipped_mask_ratio']['mean']:.3f}
  -> {branches['flipped']['flipped_mask_ratio']['mean']:.3f};
  paired difference {ci('flipped', 'flipped_mask_ratio')}.

## Faithfulness

- Sequential deletion difference versus pure: {ci('sequential', 'deletion_advantage')}.
- Sequential insertion difference versus pure: {ci('sequential', 'insertion_advantage')}.
- Local deletion difference versus pure: {ci('local', 'deletion_advantage')}.
- Local insertion difference versus pure: {ci('local', 'insertion_advantage')}.
- Flipped deletion difference versus pure: {ci('flipped', 'deletion_advantage')}.
- Flipped insertion difference versus pure: {ci('flipped', 'insertion_advantage')}.

## Supported conclusion

The strict sequential and local masks changed attribution in their intended
directions while preserving task performance. The exact inverse control
produced the strongest alignment with the deliberately flipped target.
Faithfulness changed by metric: deletion improved for the sequential branch,
while insertion did not show a uniform improvement. Explanation localization
therefore remains distinct from universal causal faithfulness.
"""
    (output_dir / "rq3_strict_results_summary.md").write_text(
        text, encoding="utf-8"
    )


def save_aggregate_figure(summary: dict, output_dir: Path) -> None:
    matrix = np.asarray(
        [
            [summary["branches"][branch][metric]["mean"] for metric in MASK_METRICS]
            for branch in BRANCHES
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    image = axes[0].imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_xticks(range(3), MASK_LABELS, rotation=20, ha="right")
    axes[0].set_yticks(range(4), [LABELS[branch] for branch in BRANCHES])
    axes[0].set_title("Attribution energy inside each mask")
    axes[0].grid(False)
    for row in range(4):
        for column in range(3):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > 0.58 else "black",
            )
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)

    x = np.arange(len(BRANCHES))
    width = 0.34
    for offset, metric, label in (
        (-width / 2, "deletion_advantage", "Deletion"),
        (width / 2, "insertion_advantage", "Insertion"),
    ):
        means = np.asarray([summary["branches"][branch][metric]["mean"] for branch in BRANCHES])
        lows = np.asarray([summary["branches"][branch][metric]["ci95_low"] for branch in BRANCHES])
        highs = np.asarray([summary["branches"][branch][metric]["ci95_high"] for branch in BRANCHES])
        axes[1].bar(x + offset, means, width, label=label, alpha=0.88)
        axes[1].errorbar(
            x + offset,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            color="black",
            capsize=2,
            lw=0.8,
        )
    axes[1].set_xticks(x, [LABELS[branch] for branch in BRANCHES], rotation=20, ha="right")
    axes[1].set_ylabel("Advantage over random perturbation")
    axes[1].set_title("Perturbation faithfulness")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0.0, 0.55)

    fig.tight_layout()
    fig.savefig(output_dir / "rq3_localization_and_faithfulness.png")
    fig.savefig(output_dir / "rq3_localization_and_faithfulness.pdf")
    plt.close(fig)


def panel_cell(image_path: Path, column: int, row: int) -> Image.Image:
    """Extract one image panel from the existing 5-column, 4-row diagnostic."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    x0 = int(column * width / 5)
    x1 = int((column + 1) * width / 5)
    content_top = int(height * 0.035)
    row_height = (height - content_top) / 4
    y0 = int(content_top + row * row_height)
    y1 = int(content_top + (row + 1) * row_height)
    cell = image[y0:y1, x0:x1]

    nonwhite = np.any(cell < 245, axis=2)
    def longest_run(indices: np.ndarray) -> tuple[int, int] | None:
        if not len(indices):
            return None
        splits = np.flatnonzero(np.diff(indices) > 1) + 1
        runs = np.split(indices, splits)
        best = max(runs, key=len)
        return int(best[0]), int(best[-1] + 1)

    panel_rows = np.flatnonzero(nonwhite.mean(axis=1) > 0.5)
    if not len(panel_rows):
        return Image.fromarray(cell)
    top, bottom = longest_run(panel_rows)
    panel_columns = np.flatnonzero(nonwhite[top:bottom].mean(axis=0) > 0.5)
    if not len(panel_columns):
        return Image.fromarray(cell[top:bottom])
    left, right = longest_run(panel_columns)
    return Image.fromarray(cell[top:bottom, left:right])


def aligned_rgb(panel: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(panel.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def register_overlay(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    import cv2

    template = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    source = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        cv2.findTransformECC(template, source, warp, cv2.MOTION_AFFINE, criteria)
        return cv2.warpAffine(
            overlay,
            warp,
            (rgb.shape[1], rgb.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        return overlay


def explanation_components(image_path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    observation = panel_cell(image_path, 0, 0)
    size = observation.size
    rgb = aligned_rgb(observation, size)
    overlays = [
        register_overlay(rgb, aligned_rgb(panel_cell(image_path, 4, row), size))
        for row in range(4)
    ]
    return rgb, overlays


def signed_visualizations(
    rgb: np.ndarray, overlays: list[np.ndarray]
) -> list[np.ndarray]:
    signed_maps = []
    for overlay in overlays:
        delta = overlay - rgb
        signed_maps.append(delta[..., 0] - delta[..., 2])
    scale = max(float(np.max(np.abs(value))) for value in signed_maps)
    scale = max(scale, 1e-8)
    return [np.sign(value) * np.sqrt(np.abs(value) / scale) for value in signed_maps]


def attribution_magnitudes(
    rgb: np.ndarray, overlays: list[np.ndarray]
) -> list[np.ndarray]:
    return [
        np.abs((overlay - rgb)[..., 0] - (overlay - rgb)[..., 2])
        for overlay in overlays
    ]


def save_faded_overlay_figure(image_path: Path, stem: str, output_dir: Path) -> None:
    rgb, overlays = explanation_components(image_path)
    labels = ("Observation", "Pure PPO", "Sequential", "Local action", "Flipped")
    faded = 0.30 * rgb + 0.70
    panels = [rgb]
    cmap = plt.get_cmap("seismic")
    for signed_map in signed_visualizations(rgb, overlays):
        color = cmap((signed_map + 1.0) / 2.0)[..., :3]
        alpha = (0.90 * np.abs(signed_map))[..., None]
        panels.append(np.clip((1.0 - alpha) * faded + alpha * color, 0.0, 1.0))

    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.45))
    for axis, panel, label in zip(axes, panels, labels):
        axis.imshow(panel)
        axis.set_title(label)
        axis.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / f"{stem}_faded_overlay.png")
    fig.savefig(output_dir / f"{stem}_faded_overlay.pdf")
    plt.close(fig)


def save_side_by_side_figure(
    image_path: Path, stem: str, output_dir: Path
) -> None:
    rgb, overlays = explanation_components(image_path)
    branch_labels = ("Pure PPO", "Sequential", "Local action", "Flipped")
    magnitude_maps = [
        magnitude / max(float(np.max(magnitude)), 1e-8)
        for magnitude in attribution_magnitudes(rgb, overlays)
    ]
    black_red = LinearSegmentedColormap.from_list(
        "absolute_black_red", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )

    fig, axes = plt.subplots(4, 2, figsize=(4.8, 7.4))
    for row, (label, magnitude) in enumerate(zip(branch_labels, magnitude_maps)):
        axes[row, 0].imshow(rgb)
        axes[row, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )
        axes[row, 1].imshow(magnitude, cmap=black_red, vmin=0.0, vmax=1.0)
        for axis in axes[row]:
            axis.axis("off")
    axes[0, 0].set_title("Observation")
    axes[0, 1].set_title("Absolute attribution")
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / f"{stem}_absolute_red_side_by_side.png")
    fig.savefig(output_dir / f"{stem}_absolute_red_side_by_side.pdf")
    plt.close(fig)


def save_state_mask_figure(image_path: Path, stem: str, output_dir: Path) -> None:
    panels = [panel_cell(image_path, column, 0) for column in range(4)]
    labels = ("Observation", "Sequential mask", "Local-action mask", "Flipped mask")
    fig, axes = plt.subplots(1, 4, figsize=(8.7, 2.45))
    for axis, panel, label in zip(axes, panels, labels):
        axis.imshow(panel)
        axis.set_title(label)
        axis.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def save_representative_figure(
    experiment_dir: Path, output_dir: Path, *, side_by_side_only: bool = False
) -> None:
    figure_dir = experiment_dir / "evaluation" / "seed_0" / "figures"
    figure_specs = (
        ("movement.png", "Movement state: forward", "rq3_explanations_movement"),
        (
            "valid_interaction.png",
            "Valid interaction state: pickup",
            "rq3_explanations_valid_interaction",
        ),
        (
            "invalid_interaction.png",
            "Invalid interaction state",
            "rq3_explanations_invalid_interaction",
        ),
    )
    for filename, title, stem in figure_specs:
        image_path = figure_dir / filename
        save_side_by_side_figure(image_path, stem, output_dir)
        save_state_mask_figure(
            image_path,
            stem.replace("rq3_explanations_", "rq3_masks_"),
            output_dir,
        )
        if not side_by_side_only:
            save_faded_overlay_figure(image_path, stem, output_dir)


def run_ids(experiment_dir: Path) -> list[dict]:
    rows = []
    for evaluation in sorted((experiment_dir / "evaluation").glob("seed_*/evaluation.json")):
        result = json.loads(evaluation.read_text(encoding="utf-8"))
        seed = int(evaluation.parent.name.split("_")[-1])
        for branch in BRANCHES:
            model = result["models"][branch]
            run_id = Path(model["path"]).parent.name
            continuation = model["continuation"]
            rows.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "run_id": run_id,
                    "source_sha256": continuation["source_sha256"],
                    "saved_timesteps": model["saved_timesteps"],
                    "explanation_coef": model["saved_explanation_coef"],
                    "parameters": continuation["parameters"]["total"],
                    "model_path_on_cluster": model["path"],
                }
            )
    return rows


def fetch_wandb(rows: list[dict], entity: str, project: str, output_dir: Path) -> None:
    import wandb

    api = wandb.Api()
    diagnostics = []
    archive = []
    for row in rows:
        run = api.run(f"{entity}/{project}/{row['run_id']}")
        raw_history = list(
            run.scan_history(
                keys=["diagnostics/explanation_to_ppo_actor_grad_ratio"]
            )
        )
        capped_history = list(
            run.scan_history(keys=["train/explanation_actor_grad_ratio"])
        )
        raw_ratio = np.asarray(
            [
                item.get("diagnostics/explanation_to_ppo_actor_grad_ratio")
                for item in raw_history
                if item.get("diagnostics/explanation_to_ppo_actor_grad_ratio") is not None
            ],
            dtype=float,
        )
        capped_ratio = np.asarray(
            [
                item.get("train/explanation_actor_grad_ratio")
                for item in capped_history
                if item.get("train/explanation_actor_grad_ratio") is not None
            ],
            dtype=float,
        )
        artifacts = list(run.logged_artifacts())
        artifact = next((value for value in artifacts if value.type == "model"), None)
        metadata = artifact.metadata if artifact is not None else {}
        final_sha256 = metadata.get("sha256", "")
        artifact_name = artifact.name if artifact is not None else ""
        record = {
            **row,
            "final_model_sha256": final_sha256,
            "artifact_name": artifact_name,
                "raw_explanation_to_ppo_grad_ratio_mean": (
                    float(raw_ratio.mean()) if len(raw_ratio) else np.nan
                ),
                "raw_explanation_to_ppo_grad_ratio_median": (
                    float(np.median(raw_ratio)) if len(raw_ratio) else np.nan
                ),
                "applied_explanation_actor_grad_ratio_mean": (
                    float(capped_ratio.mean()) if len(capped_ratio) else np.nan
                ),
                "logged_updates": max(len(raw_ratio), len(capped_ratio)),
        }
        diagnostics.append(record)
        archive.append(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "branch": row["branch"],
                "artifact_name": artifact_name,
                "source_sha256": row["source_sha256"],
                "final_model_sha256": final_sha256,
                "config": dict(run.config),
            }
        )
    pd.DataFrame(diagnostics).to_csv(output_dir / "rq3_provenance_and_gradients.csv", index=False)
    (output_dir / "rq3_run_configs_and_hashes.json").write_text(
        json.dumps(archive, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_readme(output_dir: Path, fetched: bool) -> None:
    text = """# RQ3 Thesis Outputs

Generate the table and figures from an evaluated experiment with:

```powershell
python experiments/rq3_legacy_energy_ablation/generate_rq3_thesis_outputs.py --experiment-dir experiments/rq3_explanation_control --side-by-side-only
```

Add `--fetch-wandb` to refresh model artifact hashes and explanation-gradient
diagnostics without downloading model ZIPs.

`rq3_results_table` reports policy quality and faithfulness. `rq3_localization_table`
reports cross-mask attribution alignment. `rq3_paired_differences_vs_pure.csv`
contains paired-seed effect estimates and bootstrap intervals, and
`rq3_strict_results_summary.md` gives the compact statistical interpretation.

`rq3_localization_and_faithfulness` contains the primary aggregate evidence.
The heatmap evaluates every branch under the same three masks. The bar chart
reports deletion and insertion advantage over equal-area random perturbations.

The `*_side_by_side` figures place standalone absolute attribution magnitude in
red to the right of the observation for every branch. Each attribution panel is
normalized by its own absolute maximum, without percentile clipping. P/N/D
targets are provided separately in
the corresponding `rq3_masks_*` figures. `rq3_results_section.tex` is the
self-contained thesis text using these strict-mask results.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_latex_include(output_dir: Path) -> None:
    text = r"""% Copy to the thesis project or adapt the paths as needed.
\input{figures/rq3/rq3_results_table.tex}
\input{figures/rq3/rq3_localization_table.tex}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/rq3/rq3_localization_and_faithfulness.pdf}
    \caption{Cross-mask localization and perturbation faithfulness for the four RQ3 continuation branches. Error bars denote 95\% bootstrap confidence intervals over five matched seeds.}
    \label{fig:rq3-localization-faithfulness}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/rq3/rq3_masks_movement.pdf}
    \caption{Sequential, local-action, and flipped P/N/D masks for the representative movement state.}
    \label{fig:rq3-movement-masks}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.72\textwidth]{figures/rq3/rq3_explanations_movement_absolute_red_side_by_side.pdf}
    \caption{Representative absolute B-cos attribution magnitudes for a movement state. Darker red denotes stronger attribution. Each attribution map is normalized by its own absolute maximum without percentile clipping.}
    \label{fig:rq3-movement-explanations}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/rq3/rq3_masks_valid_interaction.pdf}
    \caption{Sequential, local-action, and flipped P/N/D masks for the representative valid-interaction state.}
    \label{fig:rq3-interaction-masks}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.72\textwidth]{figures/rq3/rq3_explanations_valid_interaction_absolute_red_side_by_side.pdf}
    \caption{Representative absolute B-cos attribution magnitudes for a valid interaction state. Darker red denotes stronger attribution. Each attribution map is normalized by its own absolute maximum without percentile clipping.}
    \label{fig:rq3-interaction-explanations}
\end{figure}
"""
    (output_dir / "rq3_thesis_include.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--fetch-wandb", action="store_true")
    parser.add_argument("--side-by-side-only", action="store_true")
    parser.add_argument(
        "--entity", default="ruslandontsov2005-saarland-informatics-campus"
    )
    parser.add_argument("--project", default="rq3-minigrid-ppo-explanation-control-final")
    args = parser.parse_args()
    output_dir = args.experiment_dir / "thesis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    style()
    summary = load_summary(args.experiment_dir / "final" / "summary.json")
    write_results_table(summary, output_dir)
    write_localization_table(summary, output_dir)
    write_paired_differences(summary, output_dir)
    write_result_summary(summary, output_dir)
    save_aggregate_figure(summary, output_dir)
    save_representative_figure(
        args.experiment_dir,
        output_dir,
        side_by_side_only=args.side_by_side_only,
    )
    provenance = run_ids(args.experiment_dir)
    pd.DataFrame(provenance).to_csv(output_dir / "rq3_local_provenance.csv", index=False)
    if args.fetch_wandb:
        fetch_wandb(provenance, args.entity, args.project, output_dir)
    write_readme(output_dir, args.fetch_wandb)
    write_latex_include(output_dir)
    print(f"Saved RQ3 thesis outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

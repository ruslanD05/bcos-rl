#!/usr/bin/env python3
"""Discover RQ3 continuation models, evaluate every seed, then aggregate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AVAILABLE_BRANCHES = ("pure", "sequential", "local", "flipped")
DEFAULT_BRANCHES = AVAILABLE_BRANCHES
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=ROOT / "artifacts",
        help=(
            "Root containing flat W&B artifact directories named "
            "rq3-<branch>-seed<seed>-*/final_model.zip, or the legacy "
            "seed_<seed>/<branch>/*/final_model.zip layout."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "experiments"
            / "rq3_explanation_control"
            / "evaluation"
        ),
    )
    parser.add_argument(
        "--pure-models-root",
        type=Path,
        help="Optional continuation root containing the matched pure controls.",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=AVAILABLE_BRANCHES,
        default=list(DEFAULT_BRANCHES),
        help="Model branches to evaluate (default: pure sequential local flipped).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Matched training seeds to evaluate (default: 0 1 2 3 4).",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--dataset-episodes", type=int, default=30)
    parser.add_argument("--dataset-samples", type=int, default=256)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def locate_model(root: Path, seed: int, branch: str) -> Path:
    seed_root = root / f"seed_{seed}"
    candidates = sorted((seed_root / branch).glob("*/final_model.zip"))
    if branch == "pure" and not candidates:
        direct = seed_root / "final_model.zip"
        candidates = [direct] if direct.is_file() else []
    if not candidates:
        candidates = sorted(
            root.glob(f"rq3-{branch}-seed{seed}-*/final_model.zip")
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one model for seed={seed}, branch={branch}; found "
            f"{len(candidates)} under {root}"
        )
    return candidates[0]


def main() -> None:
    args = make_args()
    evaluator = Path(__file__).with_name("evaluate_local_mask_experiment.py")
    aggregator = Path(__file__).with_name("aggregate_rq3_ppo.py")
    args.output_root.mkdir(parents=True, exist_ok=True)

    evaluation_paths: list[Path] = []
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")

    for seed in args.seeds:
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        output = seed_dir / "evaluation.json"
        evaluation_paths.append(output)
        command = [
            sys.executable,
            str(evaluator),
            "--reference-model",
            "pure",
            "--episodes",
            str(args.episodes),
            "--seed",
            str(10_000 + seed),
            "--dataset-seed",
            str(20_000 + seed),
            "--dataset-episodes",
            str(args.dataset_episodes),
            "--dataset-samples",
            str(args.dataset_samples),
            "--dataset-out",
            str(seed_dir / "shared_states_rgb6.npz"),
            "--bootstrap-repeats",
            str(args.bootstrap_repeats),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--output",
            str(output),
            "--faithfulness-plot",
            str(seed_dir / "faithfulness.png"),
            "--figures-dir",
            str(seed_dir / "figures"),
        ]
        for branch in args.branches:
            model_root = (
                args.pure_models_root
                if branch == "pure" and args.pure_models_root is not None
                else args.models_root
            )
            command.extend(
                ["--model", f"{branch}={locate_model(model_root, seed, branch)}"]
            )
        print(f"Evaluating RQ3 seed {seed}", flush=True)
        subprocess.run(command, check=True)

    final_dir = args.output_root.parent / "final"
    aggregate_command = [
        sys.executable,
        str(aggregator),
        "--output",
        str(final_dir / "summary.json"),
    ]
    for path in evaluation_paths:
        aggregate_command.extend(["--input", str(path)])
    subprocess.run(aggregate_command, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute final channelwise localization for the downloaded unified RQ4 sweep.

Training thresholds and deterministic success are already stored in the artifacts.
For each matched seed, the B-cos PPO baseline generates one fixed semantic-mask state set. Each
schedule's final checkpoint is then scored only on that same stored state set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("rq4-consolidated"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rq4_supervision_timing/evaluations"),
    )
    parser.add_argument("--reference-episodes", type=int, default=30)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return "channelwise_explanation" in result
    except (json.JSONDecodeError, TypeError):
        return False


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.artifact_dir / "manifest.json").read_text())
    by_seed: dict[int, dict[str, Path]] = {}
    for entry in manifest["runs"]:
        by_seed.setdefault(int(entry["seed"]), {})[entry["condition"]] = (
            args.artifact_dir / "runs" / entry["wandb_run_id"] / "models" / "final.zip"
        )
    expected = {"pure", "linear", "success_gated_50", "constant"}
    if set(by_seed) != set(range(5)) or any(set(models) != expected for models in by_seed.values()):
        raise RuntimeError("Expected all four RQ4 schedules for each seed 0--4")
    evaluator = Path(__file__).with_name("evaluate_explanation_metrics.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed, models in sorted(by_seed.items()):
        seed_dir = args.output_dir / f"seed-{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        dataset = seed_dir / "pure_reference_states.npz"
        for condition in ("pure", "linear", "success_gated_50", "constant"):
            output = seed_dir / f"{condition}.json"
            if not args.force and complete(output):
                print(f"seed {seed}: {condition} already complete", flush=True)
                continue
            print(f"seed {seed}: evaluating {condition}", flush=True)
            command = [
                sys.executable, str(evaluator), "--model-path", str(models[condition]),
                "--env-id", "MiniGrid-DoorKey-8x8-v0",
                "--seed", str(10_000 + seed), "--device", "cpu",
                "--open-door-reward", "0.1", "--pick-key-reward", "0.05",
                "--max-explanation-samples", str(args.samples),
                "--explanation-batch-size", "32", "--output", str(output),
            ]
            if condition == "pure" and not dataset.is_file():
                command.extend(["--episodes", str(args.reference_episodes), "--dataset-out", str(dataset)])
            elif dataset.is_file():
                command.extend([
                    "--dataset-in", str(dataset), "--dataset-actions", "stored",
                    "--skip-policy-evaluation",
                ])
            else:
                raise RuntimeError("The pure reference dataset must be created first")
            run(command)
    print(f"Completed RQ4 final evaluation batch: {args.output_dir}")


if __name__ == "__main__":
    main()

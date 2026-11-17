#!/usr/bin/env python3
"""Re-evaluate every downloaded paired checkpoint from the unified RQ6 sweep.

This deliberately invokes the established evaluator with the original sweep's
evaluation protocol: 100 stochastic episodes per patch condition and 128 paired
states. Existing valid JSON results are skipped, making interrupted batches safe
to resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("rq6-unified-complete-results/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rq6_shortcut_suppression/evaluations"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-paired-states", type=int, default=128)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N pairs.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def result_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"baseline", "suppressed"} <= set(payload["models"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def main() -> None:
    args = arguments()
    evaluator = Path(__file__).with_name("evaluate_shortcut_experiment.py")
    pairs = sorted(args.artifacts_dir.glob("corr-*/seed-*"))
    if len(pairs) != 30:
        raise RuntimeError(f"Expected 30 downloaded RQ6 pairs, found {len(pairs)}")
    if args.limit is not None:
        pairs = pairs[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, pair in enumerate(pairs, start=1):
        correlation = pair.parent.name.removeprefix("corr-")
        seed_label = pair.name.removeprefix("seed-")
        train_seed = int(seed_label)
        output = args.output_dir / f"corr-{correlation}_seed-{seed_label}.json"
        if not args.force and result_is_complete(output):
            print(f"[{index:02d}/{len(pairs):02d}] already complete: {pair}", flush=True)
            continue
        print(f"[{index:02d}/{len(pairs):02d}] evaluating: {pair}", flush=True)
        command = [
            sys.executable,
            str(evaluator),
            "--baseline-model", str(pair / "models/baseline.zip"),
            "--suppressed-model", str(pair / "models/suppressed.zip"),
            "--env-id", "MiniGrid-DoorKey-6x6-v0",
            "--episodes", str(args.episodes),
            "--seed", str(20_000 + train_seed),
            "--device", "cpu",
            "--stochastic",
            "--border-height", "8",
            "--patch-size", "6",
            "--max-paired-states", str(args.max_paired_states),
            "--batch-size", "32",
            "--output", str(output),
        ]
        subprocess.run(command, check=True)
    print(f"Completed RQ6 evaluation batch: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

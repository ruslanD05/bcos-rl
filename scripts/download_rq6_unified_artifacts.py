#!/usr/bin/env python3
"""Download the completed unified RQ6 model-pair artifacts from W&B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


ENTITY = "ruslandontsov2005-saarland-informatics-campus"
PROJECT = "thesis-unified-rq6-shortcut-v1"
SWEEP_ID = "v0jkk8mv"
REQUIRED_FILES = (
    "models/baseline.zip",
    "models/suppressed.zip",
    "metadata/resolved_config.json",
    "statistics/paired_summary.json",
    "statistics/run_statistics.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rq6_shortcut_suppression/artifacts"),
    )
    parser.add_argument(
        "--correlations",
        type=float,
        nargs="+",
        help="Download only these training-correlation values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=120)
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
    runs = sorted(
        list(sweep.runs),
        key=lambda run: (
            float(run.config["training_correlation"]),
            int(run.config["seed"]),
        ),
    )
    if len(runs) != 30 or any(run.state != "finished" for run in runs):
        states = {run.state for run in runs}
        raise RuntimeError(
            f"Expected 30 finished runs, found {len(runs)} with states {states}"
        )

    if args.correlations:
        wanted = set(args.correlations)
        runs = [
            run for run in runs
            if float(run.config["training_correlation"]) in wanted
        ]

    print(f"Downloading {len(runs)} completed RQ6 model-pair artifacts...", flush=True)
    manifest: list[dict[str, object]] = []
    for index, run in enumerate(runs, start=1):
        correlation = float(run.config["training_correlation"])
        seed = int(run.config["seed"])
        correlation_label = f"{correlation:.2f}".replace(".", "p")
        artifact_name = (
            f"rq6-channelwise-pair-corr-{correlation_label}-seed-{seed:03d}:v0"
        )
        artifact = api.artifact(f"{ENTITY}/{PROJECT}/{artifact_name}")
        target = args.output_dir / run.id
        for name in REQUIRED_FILES:
            destination = target / name
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            print(f"[{index:02d}/{len(runs):02d}] {run.id}: {name}", flush=True)
            artifact.get_path(name).download(root=str(target))
        missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
        if missing:
            raise RuntimeError(f"Artifact download incomplete for {run.id}: {missing}")
        manifest.append(
            {
                "run_id": run.id,
                "artifact": artifact_name,
                "training_correlation": correlation,
                "seed": seed,
                "valid_source_pair": bool(run.summary.get("paired/valid_source_pair")),
            }
        )
        print(f"[{index:02d}/{len(runs):02d}] complete: {run.id}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Downloaded 30 paired artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

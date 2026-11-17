#!/usr/bin/env python3
"""Aggregate matched multi-seed outputs from evaluate_wall_ood.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "success_rate": ("policy", "success_rate"),
    "mean_reward": ("policy", "mean_reward"),
    "mean_episode_length": ("policy", "mean_episode_length"),
    "masked_energy_ratio": ("explanation", "masked_energy_ratio"),
    "explanation_loss": ("explanation", "loss"),
    "positive_alignment": ("explanation", "positive_alignment"),
    "mean_entropy": ("actor_distribution", "mean_entropy"),
    "mean_max_probability": ("actor_distribution", "mean_max_probability"),
}


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=50_000)
    return parser.parse_args()


def nested(mapping: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = mapping
    for key in path:
        value = value[key]
    return float(value)


def summarize(values: list[float], repeats: int, rng: np.random.Generator) -> dict:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n": int(len(array)),
        "values": array.tolist(),
    }


def main() -> None:
    args = make_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    if len(records) < 2:
        raise ValueError("At least two matched seed evaluations are required")
    variants = tuple(records[0]["variants"])
    branches = tuple(records[0]["models"])
    if "normal" not in variants or "pure" not in branches:
        raise ValueError("Expected normal variant and pure reference branch")
    for record in records[1:]:
        if tuple(record["variants"]) != variants or tuple(record["models"]) != branches:
            raise ValueError("All inputs must contain identical variants and branches")

    rng = np.random.default_rng(args.seed)
    result: dict[str, Any] = {
        "inputs": [str(path.resolve()) for path in args.input],
        "seeds": len(records),
        "variants": list(variants),
        "branches": list(branches),
        "aggregate": {},
        "paired_variant_minus_normal": {},
        "paired_branch_minus_pure": {},
    }
    for branch in branches:
        result["aggregate"][branch] = {}
        result["paired_variant_minus_normal"][branch] = {}
        for variant in variants:
            result["aggregate"][branch][variant] = {}
            result["paired_variant_minus_normal"][branch][variant] = {}
            for metric, path in METRICS.items():
                values = [
                    nested(record["variants"][variant]["models"][branch], path)
                    for record in records
                ]
                normal = [
                    nested(record["variants"]["normal"]["models"][branch], path)
                    for record in records
                ]
                result["aggregate"][branch][variant][metric] = summarize(
                    values, args.bootstrap_repeats, rng
                )
                result["paired_variant_minus_normal"][branch][variant][metric] = summarize(
                    (np.asarray(values) - np.asarray(normal)).tolist(),
                    args.bootstrap_repeats,
                    rng,
                )

    for branch in branches:
        result["paired_branch_minus_pure"][branch] = {}
        for variant in variants:
            result["paired_branch_minus_pure"][branch][variant] = {}
            for metric, path in METRICS.items():
                values = [
                    nested(record["variants"][variant]["models"][branch], path)
                    - nested(record["variants"][variant]["models"]["pure"], path)
                    for record in records
                ]
                result["paired_branch_minus_pure"][branch][variant][metric] = summarize(
                    values, args.bootstrap_repeats, rng
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Aggregated {len(records)} seeds: {', '.join(variants)}")
    for branch in branches:
        values = [
            f"{variant}={result['aggregate'][branch][variant]['success_rate']['mean']:.3f}"
            for variant in variants
        ]
        print(f"- {branch}: " + ", ".join(values))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Matched multi-seed evaluation of oracle-patch baseline and suppression models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_shortcut_experiment as single


def parse_seed_paths(values: list[str], label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        raw_seed, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"{label} must use SEED=PATH, got {value!r}")
        seed = int(raw_seed)
        path = Path(raw_path)
        if seed in result:
            raise ValueError(f"Duplicate {label} seed: {seed}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[seed] = path
    return result


def discover_models(root: Path, condition: str) -> dict[int, Path]:
    candidates: dict[int, list[Path]] = {}
    for path in root.glob(f"**/{condition}/final_model.zip"):
        seed = None
        for parent in path.parents:
            for token in parent.name.replace("-", "_").split("_"):
                if token.startswith("seed") and token[4:].isdigit():
                    seed = int(token[4:])
                    break
            if seed is not None:
                break
        if seed is not None:
            candidates.setdefault(seed, []).append(path)
    return {
        seed: max(paths, key=lambda item: item.stat().st_mtime)
        for seed, paths in candidates.items()
    }


def bootstrap_mean(values: list[float], repeats: int, seed: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(repeats, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "seeds": int(len(data)),
    }


def seed_metrics(model: dict[str, Any]) -> dict[str, float]:
    performance = model["performance"]
    paired = model["paired_sensitivity"]
    correlated = performance["correlated"]["success_rate"]
    return {
        "correlated_success": correlated,
        "permuted_success": performance["permuted"]["success_rate"],
        "removed_success": performance["removed"]["success_rate"],
        "random_success": performance["random"]["success_rate"],
        "permuted_success_drop": correlated - performance["permuted"]["success_rate"],
        "removed_success_drop": correlated - performance["removed"]["success_rate"],
        "random_success_drop": correlated - performance["random"]["success_rate"],
        "patch_attribution_ratio": paired["attribution"]["mean_patch_attribution_ratio"],
        "permuted_action_change_rate": 1.0 - paired["permuted"]["greedy_action_agreement"],
        "permuted_probability_l1": paired["permuted"]["mean_probability_l1"],
    }


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--suppressed", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=20_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--border-height", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--max-paired-states", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("shortcut_multiseed.json"))
    return parser.parse_args()


def main() -> None:
    args = make_args()
    baseline = parse_seed_paths(args.baseline, "baseline")
    suppressed = parse_seed_paths(args.suppressed, "suppressed")
    if args.models_root:
        baseline.update(discover_models(args.models_root, "baseline"))
        suppressed.update(discover_models(args.models_root, "suppressed"))
    if not baseline or set(baseline) != set(suppressed):
        raise ValueError(
            f"Baseline and suppressed seeds must match and be non-empty: "
            f"baseline={sorted(baseline)}, suppressed={sorted(suppressed)}"
        )

    results: dict[str, Any] = {
        "config": {
            "episodes_per_variant": args.episodes,
            "policy_action_mode": "stochastic" if args.stochastic else "deterministic",
            "matched_training_seeds": sorted(baseline),
            "bootstrap_unit": "training seed",
            "bootstrap_repeats": args.bootstrap_repeats,
        },
        "seeds": {},
    }
    condition_values = {"baseline": [], "suppressed": []}
    paired_deltas: dict[str, list[float]] = {}
    for training_seed in sorted(baseline):
        print(f"Evaluating matched seed {training_seed}", flush=True)
        single_args = argparse.Namespace(
            env_id=args.env_id,
            episodes=args.episodes,
            seed=args.evaluation_seed + training_seed * 1000,
            stochastic=args.stochastic,
            device=args.device,
            border_height=args.border_height,
            patch_size=args.patch_size,
            max_paired_states=args.max_paired_states,
            batch_size=args.batch_size,
        )
        models = {
            "baseline": single.evaluate_model(baseline[training_seed], single_args),
            "suppressed": single.evaluate_model(suppressed[training_seed], single_args),
        }
        compact = {name: seed_metrics(model) for name, model in models.items()}
        results["seeds"][str(training_seed)] = {
            "models": models,
            "summary": compact,
        }
        for name in condition_values:
            condition_values[name].append(compact[name])
        for metric in compact["baseline"]:
            paired_deltas.setdefault(metric, []).append(
                compact["suppressed"][metric] - compact["baseline"][metric]
            )

    results["condition_summary"] = {
        condition: {
            metric: bootstrap_mean(
                [seed[metric] for seed in values],
                args.bootstrap_repeats,
                args.evaluation_seed + index,
            )
            for index, metric in enumerate(values[0])
        }
        for condition, values in condition_values.items()
    }
    results["paired_suppressed_minus_baseline"] = {
        metric: bootstrap_mean(
            values,
            args.bootstrap_repeats,
            args.evaluation_seed + 10_000 + index,
        )
        for index, (metric, values) in enumerate(paired_deltas.items())
    }
    results["interpretation"] = {
        "robustness": "negative paired success-drop deltas favor suppression",
        "attribution": "negative patch-attribution delta means suppression reduced patch reliance",
        "behavior": "negative action-change and probability-L1 deltas mean less counterfactual sensitivity",
        "scope": "bootstrap intervals quantify variation across matched training seeds, not episode-level independence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["paired_suppressed_minus_baseline"], indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

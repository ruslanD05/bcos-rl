#!/usr/bin/env python3
"""Recompute full-shortcut RQ6 attribution and paired-sensitivity metrics.

The original RQ6 evaluations already contain the behavioral measurements.  This
script deterministically recreates each model's 128 correlated reference
states and replaces paired sensitivity using the complete top-row shortcut
region and ``sum_c |E_c|``. It never changes success, reward, or length.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

from evaluate_shortcut_experiment import (
    load_model,
    paired_sensitivity,
)
from shortcut_experiment_env import make_shortcut_vec_env


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/rq6_shortcut_suppression/evaluations"),
    )
    parser.add_argument("--max-paired-states", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def train_seed(path: Path) -> int:
    return int(path.stem.rsplit("seed-", maxsplit=1)[1])


def collect_correlated_states(
    model, args: SimpleNamespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect exactly the reference state/action protocol, without 100 episodes."""
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    env = make_shortcut_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed,
        mode="correlated",
        correlation_probability=1.0,
        border_height=args.border_height,
        patch_size=args.patch_size,
    )
    observations: list[np.ndarray] = []
    oracle_actions: list[int] = []
    actions: list[int] = []
    current = env.reset()
    try:
        while len(observations) < args.max_paired_states:
            action, _ = model.predict(current, deterministic=not args.stochastic)
            next_observation, _reward, _done, infos = env.step(action)
            observations.append(current[0].copy())
            oracle_actions.append(int(infos[0]["shortcut_oracle_action"]))
            actions.append(int(action[0]))
            current = next_observation
    finally:
        env.close()
    return (
        np.stack(observations),
        np.asarray(oracle_actions, dtype=np.int64),
        np.asarray(actions, dtype=np.int64),
    )


def main() -> None:
    args = arguments()
    paths = sorted(args.input_dir.glob("corr-*_seed-*.json"))
    if len(paths) != 30:
        raise RuntimeError(f"Expected 30 RQ6 evaluation files, found {len(paths)}")
    for index, path in enumerate(paths, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = [
            payload["models"][model]["paired_sensitivity"].get(
                "shortcut_region_attribution"
            )
            for model in ("baseline", "suppressed")
        ]
        if all(item is not None for item in existing) and not args.force:
            print(f"[{index:02d}/30] already recomputed: {path.name}", flush=True)
            continue
        eval_args = SimpleNamespace(
            env_id=payload["config"]["env_id"],
            episodes=int(payload["config"]["episodes"]),
            seed=20_000 + train_seed(path),
            device=args.device,
            stochastic=payload["config"]["policy_action_mode"] == "stochastic",
            border_height=int(payload["config"]["border_height"]),
            patch_size=int(payload["config"]["patch_size"]),
            max_paired_states=args.max_paired_states,
            batch_size=args.batch_size,
        )
        for model_name in ("baseline", "suppressed"):
            model_data = payload["models"][model_name]
            model = load_model(Path(model_data["model_path"]), args.device)
            observations, oracle_actions, selected_actions = collect_correlated_states(
                model, eval_args
            )
            model_data["paired_sensitivity"] = paired_sensitivity(
                model,
                observations,
                oracle_actions,
                selected_actions,
                eval_args,
            )
            # Persist each expensive model computation; this makes the script
            # safe to resume under short batch-job time limits.
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[{index:02d}/30] updated: {path.name}", flush=True)


if __name__ == "__main__":
    main()

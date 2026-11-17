#!/usr/bin/env python3
"""Audit normalization and B-cos completeness on thesis checkpoints.

The audit uses fixed observations collected from the corresponding evaluation
environments and checks whether input-times-gradient contributions reconstruct
the selected action-relative logit or Q-value in explanation mode.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import DQN, PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MINIGRID_PART = ROOT / "minigrid_part"
if str(MINIGRID_PART) not in sys.path:
    sys.path.insert(0, str(MINIGRID_PART))

from bcos_rl_generic.explanation_supervision import (
    compute_actor_explanation,
    compute_q_explanation,
    relative_action_outputs,
)
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    ConfigurableDQNPolicy,
)
from minigrid_part.shortcut_experiment_env import make_shortcut_vec_env


NORMALIZATION_NAMES = {
    "AllNorm2d",
    "BatchNorm2d",
    "BatchNormUncentered2d",
    "DetachableGroupNorm2d",
    "DetachableLayerNorm",
    "DetachablePositionNorm2d",
    "GNInstanceNormUncentered2d",
    "GNLayerNormUncentered2d",
    "GroupNorm",
    "GroupNormUncentered2d",
    "LayerNorm",
    "RMSNormUncentered1d",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "audits" / "bcos_completeness" / "report.json",
    )
    return parser.parse_args()


def discover_models() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    rq1_root = ROOT / "experiments/rq1_architecture/minigrid_ppo_architecture/minigrid_doorkey_8x8/bcos_b2.3"
    for path in sorted(rq1_root.glob("seed_*/*/final_model.zip")):
        entries.append({"group": "RQ1 PPO", "algorithm": "PPO", "path": path})

    entries.append(
        {
            "group": "RQ2 DQN qualitative",
            "algorithm": "DQN",
            "path": ROOT / "models/rq2_dqn_proof_of_concept/j1tv9wkx/final_model.zip",
        }
    )

    for path in sorted((ROOT / "artifacts").glob("rq3-*/final_model.zip")):
        entries.append({"group": "RQ3 PPO", "algorithm": "PPO", "path": path})

    for path in sorted((ROOT / "rq4-consolidated/runs").glob("*/models/final.zip")):
        entries.append({"group": "RQ4 PPO", "algorithm": "PPO", "path": path})

    rq6_root = ROOT / "rq6-unified-complete-results/runs"
    for path in sorted(rq6_root.glob("corr-*/seed-*/models/*.zip")):
        entries.append({"group": "RQ6 PPO", "algorithm": "PPO", "path": path})

    return entries


def load_ppo(path: Path, device: str) -> PPO:
    return PPO.load(
        str(path),
        env=None,
        device=device,
        custom_objects={
            "policy_class": ConfigurableActorCriticPolicy,
            "lr_schedule": lambda _: 0.0,
        },
    )


def load_dqn(path: Path, device: str) -> DQN:
    return DQN.load(
        str(path),
        env=None,
        device=device,
        custom_objects={
            "policy_class": ConfigurableDQNPolicy,
            "lr_schedule": lambda _: 0.0,
        },
    )


def load_rgb6_observations(samples: int) -> np.ndarray:
    dataset = ROOT / "experiments/rq3_explanation_control/evaluation/seed_0/shared_states_rgb6.npz"
    with np.load(dataset) as payload:
        observations = payload["observations"]
    if len(observations) < samples:
        raise ValueError(f"RGB6 dataset has {len(observations)} observations, need {samples}")
    return observations[:samples].copy()


def collect_shortcut_observations(model: PPO, samples: int) -> np.ndarray:
    env = make_shortcut_vec_env(
        env_id="MiniGrid-DoorKey-6x6-v0",
        n_envs=1,
        seed=20_000,
        mode="correlated",
        correlation_probability=1.0,
        border_height=8,
        patch_size=6,
    )
    observations: list[np.ndarray] = []
    obs = env.reset()
    try:
        while len(observations) < samples:
            observations.append(obs[0].copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, _done, _info = env.step(action)
    finally:
        env.close()
    return np.stack(observations)


def actor_modules(policy) -> list[tuple[str, th.nn.Module]]:
    roots = [
        ("features_extractor", policy.pi_features_extractor),
        ("policy_net", policy.mlp_extractor.policy_net),
        ("action_net", policy.action_net),
    ]
    result: list[tuple[str, th.nn.Module]] = []
    seen: set[int] = set()
    for prefix, root in roots:
        for suffix, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            name = prefix if not suffix else f"{prefix}.{suffix}"
            result.append((name, module))
    return result


def q_modules(policy) -> list[tuple[str, th.nn.Module]]:
    return [("q_net" if not name else f"q_net.{name}", module) for name, module in policy.q_net.named_modules()]


def normalization_inventory(policy, algorithm: str) -> list[dict[str, Any]]:
    modules = actor_modules(policy) if algorithm == "PPO" else q_modules(policy)
    inventory = []
    for name, module in modules:
        class_name = module.__class__.__name__.removesuffix("NoBias")
        if class_name not in NORMALIZATION_NAMES:
            continue
        bias = getattr(module, "bias", None)
        inventory.append(
            {
                "path": name,
                "class": module.__class__.__name__,
                "bias_absent": bias is None,
                "eps": getattr(module, "eps", None),
            }
        )
    return inventory


def actor_target(policy, observations: th.Tensor, actions: th.Tensor, *, explanation_mode: bool) -> th.Tensor:
    context = policy.explanation_mode() if explanation_mode else th.enable_grad()
    with context:
        features = policy.extract_features(observations.float())
        if isinstance(features, tuple):
            features = features[0]
        latent = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent)
        return relative_action_outputs(logits).gather(1, actions[:, None]).squeeze(1)


def q_target(policy, observations: th.Tensor, actions: th.Tensor, *, explanation_mode: bool) -> th.Tensor:
    context = policy.explanation_mode() if explanation_mode else th.enable_grad()
    with context:
        values = policy.q_net(observations.float())
        return relative_action_outputs(values).gather(1, actions[:, None]).squeeze(1)


def summarize_errors(errors: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    scaled = errors / np.maximum(targets, 1e-8)
    return {
        "median_absolute_residual": float(np.median(errors)),
        "mean_absolute_residual": float(np.mean(errors)),
        "p99_absolute_residual": float(np.quantile(errors, 0.99)),
        "max_absolute_residual": float(np.max(errors)),
        "max_scaled_residual": float(np.max(scaled)),
    }


def audit_model(entry: dict[str, Any], observations: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    path: Path = entry["path"]
    algorithm = entry["algorithm"]
    model = load_ppo(path, args.device) if algorithm == "PPO" else load_dqn(path, args.device)
    model.policy.eval()
    residuals: list[np.ndarray] = []
    target_magnitudes: list[np.ndarray] = []
    forward_differences: list[np.ndarray] = []

    for start in range(0, len(observations), args.batch_size):
        stop = min(start + args.batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
        with th.no_grad():
            if algorithm == "PPO":
                actions = th.as_tensor(
                    model.predict(observations[start:stop], deterministic=True)[0],
                    device=model.device,
                ).long()
            else:
                actions = model.policy.q_net(obs).argmax(dim=1)

        with th.enable_grad():
            if algorithm == "PPO":
                explanation = compute_actor_explanation(
                    model.policy, obs, actions, create_graph=False
                )
                target_explanation = actor_target(
                    model.policy, obs, actions, explanation_mode=True
                )
                target_ordinary = actor_target(
                    model.policy, obs, actions, explanation_mode=False
                )
            else:
                explanation = compute_q_explanation(
                    model.policy, obs, actions, create_graph=False
                )
                target_explanation = q_target(
                    model.policy, obs, actions, explanation_mode=True
                )
                target_ordinary = q_target(
                    model.policy, obs, actions, explanation_mode=False
                )

        reconstruction = explanation.flatten(start_dim=1).sum(dim=1)
        residuals.append((reconstruction - target_explanation).abs().detach().cpu().numpy())
        target_magnitudes.append(target_explanation.abs().detach().cpu().numpy())
        forward_differences.append(
            (target_explanation - target_ordinary).abs().detach().cpu().numpy()
        )

    errors = np.concatenate(residuals)
    targets = np.concatenate(target_magnitudes)
    forward_difference = np.concatenate(forward_differences)
    inventory = normalization_inventory(model.policy, algorithm)
    norm_counts = Counter(item["class"] for item in inventory)
    inferred_stable = "RMSNormUncentered1d" in norm_counts
    recorded_stable = getattr(model.policy, "stable_bcos_norm", None)
    return {
        "group": entry["group"],
        "algorithm": algorithm,
        "path": str(path.relative_to(ROOT)),
        "observation_shape": list(observations.shape[1:]),
        "samples": len(observations),
        "actor_type": getattr(model.policy, "actor_type", None),
        "stable_bcos_norm": recorded_stable if recorded_stable is not None else inferred_stable,
        "stable_bcos_norm_source": "policy_attribute" if recorded_stable is not None else "inferred_from_modules",
        "normalization_counts": dict(norm_counts),
        "normalization_bias_absent": all(item["bias_absent"] for item in inventory),
        "normalization_modules": inventory,
        "max_forward_value_difference_explanation_vs_ordinary": float(np.max(forward_difference)),
        **summarize_errors(errors, targets),
    }


def main() -> None:
    args = arguments()
    entries = discover_models()
    missing = [str(item["path"].relative_to(ROOT)) for item in entries if not item["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing discovered checkpoints: {missing}")

    rgb6 = load_rgb6_observations(args.samples)
    rq6_entry = next(item for item in entries if item["group"] == "RQ6 PPO")
    rq6_collector = load_ppo(rq6_entry["path"], args.device)
    shortcut = collect_shortcut_observations(rq6_collector, args.samples)

    results = []
    for index, entry in enumerate(entries, start=1):
        observations = shortcut if entry["group"] == "RQ6 PPO" else rgb6
        print(f"[{index:03d}/{len(entries):03d}] {entry['group']}: {entry['path'].name}", flush=True)
        try:
            results.append(audit_model(entry, observations, args))
        except Exception as error:  # Keep a complete record of any load/evaluation failures.
            results.append(
                {
                    "group": entry["group"],
                    "algorithm": entry["algorithm"],
                    "path": str(entry["path"].relative_to(ROOT)),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    successful = [item for item in results if "error" not in item]
    failures = [item for item in results if "error" in item]
    report = {
        "criterion": "abs(sum(input_times_gradient) - selected_relative_output)",
        "explanation_target": "action output minus mean action output",
        "dataset": {
            "DoorKey RGB6": "RQ3 seed-0 shared real evaluation states",
            "RQ6 RGB3": "fresh correlated-cue DoorKey-6x6 evaluation rollout",
        },
        "requested_samples_per_checkpoint": args.samples,
        "checkpoint_count": len(entries),
        "successful_audits": len(successful),
        "failed_audits": len(failures),
        "group_counts": dict(Counter(item["group"] for item in results)),
        "all_normalization_biases_absent": all(
            item["normalization_bias_absent"] for item in successful
        ),
        "overall_max_absolute_residual": max(
            item["max_absolute_residual"] for item in successful
        ),
        "overall_max_forward_value_difference": max(
            item["max_forward_value_difference_explanation_vs_ordinary"]
            for item in successful
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

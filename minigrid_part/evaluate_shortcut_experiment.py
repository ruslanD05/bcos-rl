#!/usr/bin/env python3
"""Evaluate shortcut performance, shortcut-region attribution, and sensitivity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.explanation_supervision import compute_actor_explanation
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from shortcut_experiment_env import make_shortcut_vec_env, paint_shortcut_patch


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--suppressed-model", type=Path)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions; deterministic evaluation is the default.",
    )
    parser.add_argument("--border-height", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--max-paired-states", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("shortcut_results.json"))
    return parser.parse_args()


def load_model(path: Path, device: str) -> PPO:
    return PPO.load(
        str(path),
        env=None,
        device=device,
        custom_objects={
            "policy_class": ConfigurableActorCriticPolicy,
            "lr_schedule": lambda _: 0.0,
        },
    )


def evaluate_performance(
    model: PPO,
    args: argparse.Namespace,
    mode: str,
    *,
    collect_states: bool = False,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    env = make_shortcut_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed,
        mode=mode,
        correlation_probability=1.0,
        border_height=args.border_height,
        patch_size=args.patch_size,
    )
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    successes: list[float] = []
    oracle_matches: list[float] = []
    sampled_obs: list[np.ndarray] = []
    sampled_oracles: list[int] = []
    sampled_actions: list[int] = []
    reward_sum = 0.0
    length = 0
    obs = env.reset()
    try:
        while len(episode_rewards) < args.episodes:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            old_obs = obs
            obs, reward, done, infos = env.step(action)
            info = infos[0]
            oracle = int(info["shortcut_oracle_action"])
            chosen = int(action[0])
            oracle_matches.append(float(chosen == oracle))
            reward_sum += float(reward[0])
            length += 1
            if collect_states and len(sampled_obs) < args.max_paired_states:
                sampled_obs.append(old_obs[0].copy())
                sampled_oracles.append(oracle)
                sampled_actions.append(chosen)
            if done[0]:
                episode_rewards.append(reward_sum)
                episode_lengths.append(length)
                successes.append(float(info.get("is_success", reward_sum > 0)))
                reward_sum = 0.0
                length = 0
    finally:
        env.close()

    metrics = {
        "episodes": float(args.episodes),
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "oracle_action_agreement": float(np.mean(oracle_matches)),
    }
    if not collect_states:
        return metrics, None
    dataset = (
        np.stack(sampled_obs),
        np.asarray(sampled_oracles, dtype=np.int64),
        np.asarray(sampled_actions, dtype=np.int64),
    )
    return metrics, dataset


def actor_outputs(
    model: PPO,
    observations: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = []
    logits = []
    entropies = []
    model.policy.eval()
    for start in range(0, len(observations), batch_size):
        obs = th.as_tensor(
            observations[start : start + batch_size],
            device=model.device,
        )
        captured: list[th.Tensor] = []
        hook = model.policy.action_net.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        try:
            with th.no_grad():
                distribution = model.policy.get_distribution(obs).distribution
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one actor forward, captured {len(captured)}")
        probabilities.append(distribution.probs.detach().cpu().numpy())
        logits.append(captured[0].detach().cpu().numpy())
        entropies.append(distribution.entropy().detach().cpu().numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(logits),
        np.concatenate(entropies),
    )


def shortcut_region_attribution_metrics(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    border_height: int,
    batch_size: int,
) -> dict[str, float]:
    patch_energy = []
    total_energy = []
    ratios = []
    model.policy.eval()
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
        action = th.as_tensor(actions[start:stop], device=model.device)
        with th.enable_grad():
            explanation = compute_actor_explanation(
                model.policy,
                obs,
                action,
                create_graph=False,
            )
        with th.no_grad():
            # Match the channelwise supervision objective.  Summing signed
            # channels before taking the absolute value would allow RGB and
            # complement-channel contributions to cancel.
            magnitude = explanation.abs().sum(dim=1)
            shortcut_region = magnitude[:, :border_height, :].sum(
                dim=(1, 2)
            )
            total = magnitude.sum(dim=(1, 2))
            patch_energy.extend(shortcut_region.cpu().tolist())
            total_energy.extend(total.cpu().tolist())
            ratios.extend((shortcut_region / (total + 1e-8)).cpu().tolist())
    return {
        "samples": float(len(observations)),
        "aggregation": "sum_c_abs_E_c",
        "region": "full_top_row_shortcut",
        "region_height_pixels": float(border_height),
        "mean_shortcut_region_absolute_energy": float(np.mean(patch_energy)),
        "mean_total_absolute_energy": float(np.mean(total_energy)),
        "mean_shortcut_region_attribution_ratio": float(np.mean(ratios)),
    }


def paired_sensitivity(
    model: PPO,
    observations: np.ndarray,
    oracle_actions: np.ndarray,
    selected_actions: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variants = {
        mode: paint_shortcut_patch(
            observations,
            oracle_actions,
            mode=mode,
            patch_size=args.patch_size,
        )
        for mode in ("correlated", "removed", "permuted")
    }
    outputs = {
        mode: actor_outputs(model, obs, args.batch_size)
        for mode, obs in variants.items()
    }
    corr_prob, corr_logits, corr_entropy = outputs["correlated"]
    indices = np.arange(len(selected_actions))
    corr_relative = corr_logits[indices, selected_actions] - corr_logits.mean(axis=1)
    result: dict[str, Any] = {
        "states": float(len(observations)),
        "correlated_mean_entropy": float(corr_entropy.mean()),
    }
    corr_greedy = corr_prob.argmax(axis=1)
    for mode in ("removed", "permuted"):
        probabilities, logits, entropy = outputs[mode]
        greedy = probabilities.argmax(axis=1)
        relative = logits[indices, selected_actions] - logits.mean(axis=1)
        result[mode] = {
            "mean_probability_l1": float(
                np.abs(corr_prob - probabilities).sum(axis=1).mean()
            ),
            "greedy_action_agreement": float(np.mean(corr_greedy == greedy)),
            "mean_selected_relative_logit_change": float(
                np.abs(corr_relative - relative).mean()
            ),
            "mean_entropy": float(entropy.mean()),
        }
    result["shortcut_region_attribution"] = shortcut_region_attribution_metrics(
        model,
        variants["correlated"],
        selected_actions,
        border_height=args.border_height,
        batch_size=args.batch_size,
    )
    return result


def evaluate_model(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = load_model(path, args.device)
    performance: dict[str, dict[str, float]] = {}
    correlated_dataset = None
    for mode in ("correlated", "removed", "permuted", "random"):
        metrics, dataset = evaluate_performance(
            model,
            args,
            mode,
            collect_states=mode == "correlated",
        )
        performance[mode] = metrics
        if dataset is not None:
            correlated_dataset = dataset
    assert correlated_dataset is not None
    observations, oracle_actions, selected_actions = correlated_dataset
    return {
        "model_path": str(path.resolve()),
        "performance": performance,
        "paired_sensitivity": paired_sensitivity(
            model,
            observations,
            oracle_actions,
            selected_actions,
            args,
        ),
    }


def main() -> None:
    args = make_args()
    models = {
        name: path
        for name, path in (
            ("baseline", args.baseline_model),
            ("suppressed", args.suppressed_model),
        )
        if path is not None
    }
    if not models:
        raise ValueError("Provide --baseline-model and/or --suppressed-model")
    if args.episodes <= 0 or args.max_paired_states <= 0 or args.batch_size <= 0:
        raise ValueError("episode, state, and batch counts must be positive")
    results = {
        "config": {
            "env_id": args.env_id,
            "episodes": args.episodes,
            "seed": args.seed,
            "patch_size": args.patch_size,
            "border_height": args.border_height,
            "policy_action_mode": "stochastic" if args.stochastic else "deterministic",
        },
        "models": {
            name: evaluate_model(path, args) for name, path in models.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

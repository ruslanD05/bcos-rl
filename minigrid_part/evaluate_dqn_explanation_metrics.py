#!/usr/bin/env python3
"""Evaluate DQN policy performance and B-cos mask alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import DQN

from evaluate_explanation_metrics import (
    MINIGRID_ACTION_NAMES,
    evaluate_policy,
)
from train_minigrid_ActorCritic import build_vec_env

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.explanation_supervision import (
    compute_q_explanation,
    partial_masked_energy_loss,
)
from bcos_rl_generic.train_model import ConfigurableDQNPolicy


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument(
        "--grid-lines-as-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-explanation-samples", type=int, default=512)
    parser.add_argument("--explanation-batch-size", type=int, default=32)
    parser.add_argument("--dataset-in", type=Path)
    parser.add_argument("--dataset-out", type=Path)
    parser.add_argument(
        "--dataset-actions",
        choices=("model", "stored"),
        default="stored",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_model(path: Path, device: str) -> DQN:
    return DQN.load(
        str(path),
        env=None,
        device=device,
        custom_objects={
            "policy_class": ConfigurableDQNPolicy,
            "lr_schedule": lambda _: 0.0,
        },
    )


def make_env(args: argparse.Namespace):
    return build_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed,
        use_rgb_img_obs=True,
        invert_channels=False,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=True,
        sequential_walls_as_negative=True,
        sequential_grid_lines_as_negative=args.grid_lines_as_negative,
    )


def evaluate_explanations(
    model: DQN,
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "masked_energy_ratio": 0.0,
        "positive_alignment": 0.0,
        "negative_positive_fraction": 0.0,
        "positive_energy": 0.0,
        "negative_positive_energy": 0.0,
        "positive_absolute_energy": 0.0,
        "negative_absolute_energy": 0.0,
    }
    model.policy.eval()
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
        action = th.as_tensor(actions[start:stop], device=model.device).long()
        mask = th.as_tensor(masks[start:stop], device=model.device).float()
        with th.enable_grad():
            explanation = compute_q_explanation(
                model.policy,
                obs,
                action,
                create_graph=False,
            )
        loss, stats = partial_masked_energy_loss(explanation, mask)
        with th.no_grad():
            attribution = explanation.sum(dim=1)
            positive_attribution = attribution.clamp(min=0)
            positive = (mask > 0).to(attribution.dtype)
            negative = (mask < 0).to(attribution.dtype)
            p_energy = (positive_attribution * positive).sum(dim=(1, 2))
            n_energy = (positive_attribution * negative).sum(dim=(1, 2))
            p_absolute = (attribution.abs() * positive).sum(dim=(1, 2))
            n_absolute = (attribution.abs() * negative).sum(dim=(1, 2))
        count = stop - start
        totals["loss"] += loss.item() * count
        for key in (
            "masked_energy_ratio",
            "positive_alignment",
            "negative_positive_fraction",
        ):
            totals[key] += stats[key] * count
        totals["positive_energy"] += p_energy.sum().item()
        totals["negative_positive_energy"] += n_energy.sum().item()
        totals["positive_absolute_energy"] += p_absolute.sum().item()
        totals["negative_absolute_energy"] += n_absolute.sum().item()
    return {
        "samples": float(len(observations)),
        **{key: value / len(observations) for key, value in totals.items()},
    }


def evaluate_q_distribution(
    model: DQN,
    observations: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    q_batches = []
    model.policy.eval()
    for start in range(0, len(observations), batch_size):
        obs = th.as_tensor(
            observations[start : start + batch_size],
            device=model.device,
        )
        with th.no_grad():
            q_batches.append(model.q_net(obs).cpu())
    q_values = th.cat(q_batches)
    action_count = int(model.action_space.n)
    names = [
        MINIGRID_ACTION_NAMES[index]
        if index < len(MINIGRID_ACTION_NAMES)
        else f"action_{index}"
        for index in range(action_count)
    ]
    preferred = q_values.argmax(dim=1)
    rates = th.bincount(preferred, minlength=action_count).float() / len(q_values)
    top_two = q_values.topk(k=min(2, action_count), dim=1).values
    margins = top_two[:, 0] - top_two[:, 1] if action_count >= 2 else th.zeros(len(q_values))
    dominant = int(rates.argmax().item())
    return {
        "samples": int(len(q_values)),
        "mean_q_values": {
            name: float(q_values[:, index].mean())
            for index, name in enumerate(names)
        },
        "std_q_values": {
            name: float(q_values[:, index].std(unbiased=False))
            for index, name in enumerate(names)
        },
        "argmax_action_rate": {
            name: float(rates[index]) for index, name in enumerate(names)
        },
        "dominant_argmax_action": names[dominant],
        "dominant_argmax_rate": float(rates[dominant]),
        "mean_top1_top2_q_margin": float(margins.mean()),
    }


def main() -> None:
    args = make_args()
    if args.episodes <= 0 or args.max_explanation_samples <= 0:
        raise ValueError("Episode and sample counts must be positive")
    if args.explanation_batch_size <= 0:
        raise ValueError("Explanation batch size must be positive")
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    model = load_model(args.model_path, args.device)
    if tuple(model.observation_space.shape) != (3, 64, 64):
        raise ValueError(
            f"Expected an RGB3 DQN checkpoint, got {model.observation_space.shape}"
        )
    env = make_env(args)
    try:
        policy_metrics, collected = evaluate_policy(
            model,
            env,
            episodes=args.episodes,
            sample_count=args.max_explanation_samples,
            seed=args.seed,
            collect=args.dataset_in is None,
            deterministic=not args.stochastic,
        )
    finally:
        env.close()

    if args.dataset_in:
        with np.load(args.dataset_in) as data:
            observations = data["observations"]
            stored_actions = data["actions"]
            masks = data["masks"]
        if args.dataset_actions == "model":
            actions, _ = model.predict(
                observations,
                deterministic=not args.stochastic,
            )
            actions = np.asarray(actions).reshape(-1)
        else:
            actions = stored_actions
        dataset_source = str(args.dataset_in)
    else:
        assert collected is not None
        observations, actions, masks = collected
        dataset_source = "current model trajectories"

    if args.dataset_out:
        args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset_out,
            observations=observations,
            actions=actions,
            masks=masks,
        )

    result = {
        "model_path": str(args.model_path.resolve()),
        "saved_timesteps": int(model.num_timesteps),
        "observation_channels": 3,
        "grid_lines_as_negative": args.grid_lines_as_negative,
        "policy_action_mode": "stochastic" if args.stochastic else "deterministic",
        "dataset_source": dataset_source,
        "dataset_actions": args.dataset_actions,
        "policy": policy_metrics,
        "explanation": evaluate_explanations(
            model,
            observations,
            actions,
            masks,
            args.explanation_batch_size,
        ),
        "q_distribution": evaluate_q_distribution(
            model,
            observations,
            args.explanation_batch_size,
        ),
    }
    output = args.output or args.model_path.with_suffix(".metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved metrics to: {output}")


if __name__ == "__main__":
    main()

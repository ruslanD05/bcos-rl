#!/usr/bin/env python3
"""Evaluate policy performance and B-cos mask alignment for a PPO checkpoint."""
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
from bcos_rl_generic.explanation_supervision import (
    channelwise_masked_absolute_attribution_energy_loss,
    compute_actor_explanation,
    partial_masked_energy_loss,
)
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy
from train_minigrid_ActorCritic import build_vec_env


MINIGRID_ACTION_NAMES = (
    "left",
    "right",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of taking deterministic actions.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--open-door-reward", type=float, default=0.0)
    parser.add_argument("--pick-key-reward", type=float, default=0.0)
    parser.add_argument(
        "--grid-lines-as-negative",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-explanation-samples", type=int, default=512)
    parser.add_argument("--explanation-batch-size", type=int, default=32)
    parser.add_argument(
        "--dataset-in",
        type=Path,
        help="Use a fixed observation/mask dataset saved by another evaluation.",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        help="Save sampled observations, masks, and actions for comparisons.",
    )
    parser.add_argument(
        "--dataset-actions",
        choices=("model", "stored"),
        default="model",
        help="Explain this checkpoint's actions or the dataset's stored actions.",
    )
    parser.add_argument(
        "--skip-policy-evaluation",
        action="store_true",
        help="With --dataset-in, compute attribution metrics only.",
    )
    parser.add_argument("--output", type=Path)
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


def make_eval_env(args: argparse.Namespace, *, six_channel_obs: bool):
    return build_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed,
        use_rgb_img_obs=True,
        invert_channels=six_channel_obs,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=True,
        sequential_walls_as_negative=True,
        sequential_grid_lines_as_negative=args.grid_lines_as_negative,
    )


def evaluate_policy(
    model: PPO,
    env,
    *,
    episodes: int,
    sample_count: int,
    seed: int,
    collect: bool,
    deterministic: bool,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[np.ndarray, int, np.ndarray]] = []
    transitions_seen = 0
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    episode_reward = 0.0
    episode_length = 0
    obs = env.reset()

    while len(rewards) < episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, done, infos = env.step(action)
        info = infos[0]
        episode_reward += float(reward[0])
        episode_length += 1

        if collect:
            mask = info.get("attn_mask")
            if mask is None:
                raise RuntimeError("Evaluation environment did not emit attn_mask")
            item = (obs[0].copy(), int(action[0]), np.asarray(mask).copy())
            transitions_seen += 1
            if len(reservoir) < sample_count:
                reservoir.append(item)
            else:
                index = int(rng.integers(0, transitions_seen))
                if index < sample_count:
                    reservoir[index] = item

        if done[0]:
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(float(info.get("is_success", 0.0)))
            episode_reward = 0.0
            episode_length = 0
        obs = next_obs

    policy_metrics = {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "std_episode_length": float(np.std(lengths)),
        "success_rate": float(np.mean(successes)),
    }
    if not collect:
        return policy_metrics, None
    dataset = (
        np.stack([item[0] for item in reservoir]),
        np.asarray([item[1] for item in reservoir], dtype=np.int64),
        np.stack([item[2] for item in reservoir]).astype(np.float32),
    )
    return policy_metrics, dataset


def evaluate_explanations(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    if len(observations) == 0:
        raise ValueError("Explanation dataset is empty")
    model.policy.eval()
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

    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
        action = th.as_tensor(actions[start:stop], device=model.device).long()
        mask = th.as_tensor(masks[start:stop], device=model.device).float()
        with th.enable_grad():
            explanation = compute_actor_explanation(
                model.policy,
                obs,
                action,
                create_graph=False,
                subtract_action_mean=bool(
                    getattr(model, "explanation_subtract_action_mean", True)
                ),
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


def evaluate_channelwise_absolute_explanations(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Measure P/N alignment using sum_c |E_c|, irrespective of sign."""
    if len(observations) == 0:
        raise ValueError("Explanation dataset is empty")
    model.policy.eval()
    totals = {
        "loss": 0.0,
        "target_energy_ratio": 0.0,
        "negative_energy_ratio": 0.0,
        "positive_absolute_energy": 0.0,
        "negative_absolute_energy": 0.0,
    }
    selected = 0
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
        action = th.as_tensor(actions[start:stop], device=model.device).long()
        mask = th.as_tensor(masks[start:stop], device=model.device).float()
        with th.enable_grad():
            explanation = compute_actor_explanation(
                model.policy,
                obs,
                action,
                create_graph=False,
                subtract_action_mean=bool(
                    getattr(model, "explanation_subtract_action_mean", True)
                ),
            )
        loss, stats = channelwise_masked_absolute_attribution_energy_loss(
            explanation, mask
        )
        with th.no_grad():
            magnitude = explanation.abs().sum(dim=1)
            positive = (mask > 0).to(magnitude.dtype)
            negative = (mask < 0).to(magnitude.dtype)
            valid = positive.flatten(start_dim=1).any(dim=1) & negative.flatten(
                start_dim=1
            ).any(dim=1)
            p_energy = (magnitude * positive).sum(dim=(1, 2))
            n_energy = (magnitude * negative).sum(dim=(1, 2))
        count = int(valid.sum().item())
        if count == 0:
            continue
        selected += count
        totals["loss"] += float(loss.item()) * count
        totals["target_energy_ratio"] += stats["target_energy_ratio"] * count
        totals["negative_energy_ratio"] += stats["negative_energy_ratio"] * count
        totals["positive_absolute_energy"] += p_energy[valid].sum().item()
        totals["negative_absolute_energy"] += n_energy[valid].sum().item()
    denominator = max(selected, 1)
    return {
        "samples": float(len(observations)),
        "selected_samples": float(selected),
        "selected_transition_rate": float(selected / len(observations)),
        "aggregation": "sum_c_abs_E_c",
        **{key: value / denominator for key, value in totals.items()},
    }


def evaluate_actor_distribution(
    model: PPO,
    observations: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Measure final actor logits and action preferences on fixed states."""
    if len(observations) == 0:
        raise ValueError("Actor-distribution dataset is empty")
    if not hasattr(model.action_space, "n"):
        raise ValueError("Actor-distribution metrics require a discrete action space")

    model.policy.eval()
    logits_batches = []
    probability_batches = []
    entropy_batches = []
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
        actor_logits: list[th.Tensor] = []
        hook = model.policy.action_net.register_forward_hook(
            lambda _module, _inputs, output: actor_logits.append(output)
        )
        try:
            with th.no_grad():
                distribution = model.policy.get_distribution(obs).distribution
        finally:
            hook.remove()
        if len(actor_logits) != 1:
            raise RuntimeError(
                "Expected one actor action_net call, "
                f"captured {len(actor_logits)}"
            )
        logits_batches.append(actor_logits[0].detach().cpu())
        probability_batches.append(distribution.probs.detach().cpu())
        entropy_batches.append(distribution.entropy().detach().cpu())

    logits = th.cat(logits_batches, dim=0)
    probabilities = th.cat(probability_batches, dim=0)
    entropies = th.cat(entropy_batches, dim=0)
    action_count = int(model.action_space.n)
    action_names = [
        MINIGRID_ACTION_NAMES[index]
        if index < len(MINIGRID_ACTION_NAMES)
        else f"action_{index}"
        for index in range(action_count)
    ]
    preferred = probabilities.argmax(dim=1)
    preferred_counts = th.bincount(preferred, minlength=action_count).float()
    preferred_rates = preferred_counts / len(observations)
    top_two_logits = logits.topk(k=min(2, action_count), dim=1).values
    if action_count >= 2:
        logit_margins = top_two_logits[:, 0] - top_two_logits[:, 1]
    else:
        logit_margins = th.zeros(len(observations))
    dominant_index = int(preferred_rates.argmax().item())

    return {
        "samples": int(len(observations)),
        "mean_logits": {
            name: float(logits[:, index].mean().item())
            for index, name in enumerate(action_names)
        },
        "std_logits": {
            name: float(logits[:, index].std(unbiased=False).item())
            for index, name in enumerate(action_names)
        },
        "mean_probabilities": {
            name: float(probabilities[:, index].mean().item())
            for index, name in enumerate(action_names)
        },
        "argmax_action_rate": {
            name: float(preferred_rates[index].item())
            for index, name in enumerate(action_names)
        },
        "dominant_argmax_action": action_names[dominant_index],
        "dominant_argmax_rate": float(preferred_rates[dominant_index].item()),
        "mean_entropy": float(entropies.mean().item()),
        "mean_max_probability": float(probabilities.max(dim=1).values.mean().item()),
        "mean_top1_top2_logit_margin": float(logit_margins.mean().item()),
    }


def main() -> None:
    args = make_args()
    if args.episodes <= 0 or args.max_explanation_samples <= 0:
        raise ValueError("Episode and explanation sample counts must be positive")
    if args.explanation_batch_size <= 0:
        raise ValueError("--explanation-batch-size must be positive")

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    model = load_model(args.model_path, args.device)
    observation_shape = tuple(model.observation_space.shape)
    if len(observation_shape) != 3 or observation_shape[0] not in (3, 6):
        raise ValueError(
            "Expected a channels-first RGB checkpoint with 3 or 6 channels, "
            f"got {observation_shape}"
        )
    six_channel_obs = observation_shape[0] == 6
    if args.skip_policy_evaluation and args.dataset_in is None:
        raise ValueError("--skip-policy-evaluation requires --dataset-in")
    collected = None
    if args.skip_policy_evaluation:
        policy_metrics = None
    else:
        env = make_eval_env(args, six_channel_obs=six_channel_obs)
        try:
            if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
                raise ValueError(
                    "Checkpoint/environment observation mismatch: "
                    f"{model.observation_space.shape} != {env.observation_space.shape}"
                )
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
        action_mode = "stochastic" if args.stochastic else "deterministic"
        dataset_source = f"current model {action_mode} trajectories"

    if args.dataset_out:
        args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset_out,
            observations=observations,
            actions=actions,
            masks=masks,
        )

    explanation_metrics = evaluate_explanations(
        model,
        observations,
        actions,
        masks,
        args.explanation_batch_size,
    )
    channelwise_explanation_metrics = evaluate_channelwise_absolute_explanations(
        model,
        observations,
        actions,
        masks,
        args.explanation_batch_size,
    )
    actor_distribution = evaluate_actor_distribution(
        model,
        observations,
        args.explanation_batch_size,
    )
    result: dict[str, Any] = {
        "model_path": str(args.model_path.resolve()),
        "saved_timesteps": int(model.num_timesteps),
        "saved_explanation_coef": float(getattr(model, "explanation_coef", 0.0)),
        "observation_channels": int(observation_shape[0]),
        "grid_lines_as_negative": args.grid_lines_as_negative,
        "policy_action_mode": "stochastic" if args.stochastic else "deterministic",
        "policy_evaluation_skipped": args.skip_policy_evaluation,
        "dataset_source": dataset_source,
        "dataset_actions": args.dataset_actions,
        "policy": policy_metrics,
        "explanation": explanation_metrics,
        "channelwise_explanation": channelwise_explanation_metrics,
        "actor_distribution": actor_distribution,
    }
    output = args.output or args.model_path.with_suffix(".metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved metrics to: {output}")


if __name__ == "__main__":
    main()

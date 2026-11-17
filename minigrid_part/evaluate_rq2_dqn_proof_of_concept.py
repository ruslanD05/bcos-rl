#!/usr/bin/env python3
"""Compact RQ2 DQN explanation proof of concept on successful B-cos seeds."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bcos_rl_generic.explanation_supervision import (
    compute_q_explanation,
    relative_action_outputs,
)
from evaluate_dqn_explanation_metrics import load_model
from evaluate_explanation_faithfulness import (
    correlation,
    empty_grid_substrate,
    perturb_by_order,
    ranking_order,
)
from train_minigrid_ActorCritic import build_vec_env


METHODS = ("bcos", "saliency", "integrated_gradients", "smoothgrad")
METHOD_LABELS = {
    "bcos": "Intrinsic B-cos",
    "saliency": "Saliency",
    "integrated_gradients": "Integrated Gradients",
    "smoothgrad": "SmoothGrad",
}
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")


def collect_successful_states(model, args, *, seed: int):
    channels = int(model.observation_space.shape[0])
    env = build_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=seed,
        use_rgb_img_obs=True,
        invert_channels=channels == 6,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
        use_sequential_mask=False,
    )
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[np.ndarray, int]] = []
    successful_transitions = 0
    successful_episodes = 0
    attempted_episodes = 0
    episode: list[tuple[np.ndarray, int]] = []
    obs = env.reset()
    try:
        while successful_episodes < args.successful_episodes:
            action, _ = model.predict(obs, deterministic=True)
            episode.append((obs[0].copy(), int(action[0])))
            obs, _reward, done, infos = env.step(action)
            if not done[0]:
                continue
            attempted_episodes += 1
            if float(infos[0].get("is_success", 0.0)) > 0.5:
                successful_episodes += 1
                for item in episode:
                    successful_transitions += 1
                    if len(reservoir) < args.samples_per_seed:
                        reservoir.append(item)
                    else:
                        index = int(rng.integers(0, successful_transitions))
                        if index < args.samples_per_seed:
                            reservoir[index] = item
            episode = []
            if attempted_episodes >= args.max_attempted_episodes:
                break
    finally:
        env.close()
    if len(reservoir) < args.samples_per_seed:
        raise RuntimeError(
            f"Collected only {len(reservoir)} states from {successful_episodes} "
            f"successful episodes in {attempted_episodes} attempts"
        )
    observations = np.stack([item[0] for item in reservoir])
    actions = np.asarray([item[1] for item in reservoir], dtype=np.int64)
    return observations, actions, {
        "successful_episodes": successful_episodes,
        "attempted_episodes": attempted_episodes,
        "successful_transitions_seen": successful_transitions,
    }


def relative_q_tensor(model, observations: th.Tensor) -> th.Tensor:
    return relative_action_outputs(model.q_net(observations))


def selected_targets(model, observations: np.ndarray, actions: np.ndarray, batch_size: int):
    parts = []
    model.policy.eval()
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
        action = th.as_tensor(actions[start:stop], device=model.device).long()
        with th.no_grad():
            values = relative_q_tensor(model, obs)
            parts.append(values.gather(1, action[:, None]).squeeze(1).cpu().numpy())
    return np.concatenate(parts)


def explanations(model, observations, actions, method, args):
    parts = []
    model.policy.eval()
    for start in range(0, len(observations), args.batch_size):
        stop = min(start + args.batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device).float()
        action = th.as_tensor(actions[start:stop], device=model.device).long()
        with th.enable_grad():
            if method == "bcos":
                attribution = compute_q_explanation(
                    model.policy, obs, action, create_graph=False
                )
            elif method == "saliency":
                obs = obs.detach().requires_grad_(True)
                attribution = Saliency(
                    lambda value: relative_q_tensor(model, value)
                ).attribute(obs, target=action, abs=False)
            elif method == "integrated_gradients":
                obs = obs.detach().requires_grad_(True)
                attribution = IntegratedGradients(
                    lambda value: relative_q_tensor(model, value)
                ).attribute(
                    obs,
                    baselines=th.zeros_like(obs),
                    target=action,
                    n_steps=args.ig_steps,
                    method="gausslegendre",
                )
            elif method == "smoothgrad":
                obs = obs.detach().requires_grad_(True)
                attribution = NoiseTunnel(
                    Saliency(lambda value: relative_q_tensor(model, value))
                ).attribute(
                    obs,
                    target=action,
                    nt_type="smoothgrad",
                    nt_samples=args.smoothgrad_samples,
                    stdevs=0.02 * 255.0,
                    abs=False,
                )
            else:
                raise ValueError(method)
        parts.append(attribution.detach().cpu().numpy())
    return np.concatenate(parts)


def target_retention(changed, baseline, original):
    denominator = float(np.mean(original - baseline))
    if abs(denominator) < 1e-8:
        raise RuntimeError("Mean baseline-corrected relative Q-value is too small")
    return float(np.mean(changed - baseline) / denominator)


def curve_auc(model, observations, actions, baseline, original, substrate, magnitude,
              fractions, mode, insertion, args, seed):
    repeats = args.random_repeats if mode == "random" else 1
    curves = []
    for repeat in range(repeats):
        order = ranking_order(
            magnitude,
            mode,
            np.random.default_rng(seed + repeat * 1009),
        )
        values = []
        for fraction in fractions:
            changed_obs = perturb_by_order(
                observations, substrate, order, fraction, insertion=insertion
            )
            changed = selected_targets(model, changed_obs, actions, args.batch_size)
            values.append(target_retention(changed, baseline, original))
        curves.append(values)
    mean_curve = np.mean(np.asarray(curves), axis=0)
    return float(np.trapz(mean_curve, np.asarray(fractions)))


def sensitivity_infidelity(model, observations, actions, spatial_attribution, args, seed):
    zero = np.zeros(observations.shape[1:], dtype=observations.dtype)
    original = selected_targets(model, observations, actions, args.batch_size)
    predicted_all, actual_all = [], []
    magnitude = np.abs(spatial_attribution)
    for fraction_index, fraction in enumerate(args.sensitivity_fractions):
        for repeat in range(args.random_repeats):
            order = ranking_order(
                magnitude,
                "random",
                np.random.default_rng(seed + fraction_index * 10007 + repeat * 1009),
            )
            changed_obs = perturb_by_order(
                observations, zero, order, fraction, insertion=False
            )
            changed = selected_targets(model, changed_obs, actions, args.batch_size)
            count = int(round(fraction * order.shape[1]))
            flat = spatial_attribution.reshape(len(observations), -1)
            predicted_all.append(
                np.take_along_axis(flat, order[:, :count], axis=1).sum(axis=1)
            )
            actual_all.append(original - changed)
    predicted = np.concatenate(predicted_all)
    actual = np.concatenate(actual_all)
    squared_error = np.square(predicted - actual)
    return (
        correlation(predicted, actual, ranked=True),
        float(squared_error.mean() / (np.square(actual).mean() + 1e-8)),
    )


def evaluate_method(model, observations, actions, method, args, seed):
    started = time.perf_counter()
    attribution = explanations(model, observations, actions, method, args)
    runtime = (time.perf_counter() - started) / len(observations)
    spatial = attribution.sum(axis=1)
    magnitude = np.abs(spatial)

    substrate = empty_grid_substrate(tuple(observations.shape[1:]), args.grid_size)
    substrate_batch = np.broadcast_to(substrate, observations.shape).copy()
    original = selected_targets(model, observations, actions, args.batch_size)
    baseline = selected_targets(model, substrate_batch, actions, args.batch_size)
    fractions = args.curve_fractions
    deletion_high = curve_auc(
        model, observations, actions, baseline, original, substrate, magnitude,
        fractions, "high", False, args, seed + 1000,
    )
    deletion_random = curve_auc(
        model, observations, actions, baseline, original, substrate, magnitude,
        fractions, "random", False, args, seed + 2000,
    )
    insertion_high = curve_auc(
        model, observations, actions, baseline, original, substrate, magnitude,
        fractions, "high", True, args, seed + 3000,
    )
    insertion_random = curve_auc(
        model, observations, actions, baseline, original, substrate, magnitude,
        fractions, "random", True, args, seed + 4000,
    )

    sensitivity, infidelity = sensitivity_infidelity(
        model, observations, actions, spatial, args, seed + 5000
    )
    zero_batch = np.zeros_like(observations)
    zero_target = selected_targets(model, zero_batch, actions, args.batch_size)
    target_difference = original - zero_target
    attribution_sum = spatial.sum(axis=(1, 2))
    completeness = float(
        np.abs(attribution_sum - target_difference).mean()
        / (np.abs(target_difference).mean() + 1e-8)
    )
    return {
        "deletion_advantage": deletion_random - deletion_high,
        "insertion_advantage": insertion_high - insertion_random,
        "sensitivity_n": sensitivity,
        "normalized_infidelity": infidelity,
        "completeness_relative_error": completeness,
        "seconds_per_sample": runtime,
    }, attribution


def aggregate(rows):
    metrics = (
        "deletion_advantage",
        "insertion_advantage",
        "sensitivity_n",
        "normalized_infidelity",
        "completeness_relative_error",
        "seconds_per_sample",
    )
    result = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        entry = {"method": method, "n": len(selected)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_sd"] = float(values.std(ddof=1))
        result.append(entry)
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_value(mean, sd, *, scientific=False):
    if scientific:
        return f"{mean:.2e} $\\pm$ {sd:.2e}"
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def write_latex_table(path, aggregated, selected_seed_count):
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{3.5pt}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        Method & Deletion $\uparrow$ & Insertion $\uparrow$ & Sensitivity-$n$ $\uparrow$ & Infidelity $\downarrow$ & Completeness $\downarrow$ & Seconds/sample $\downarrow$ \\",
        r"        \midrule",
    ]
    for row in aggregated:
        cells = [
            METHOD_LABELS[row["method"]],
            format_value(row["deletion_advantage_mean"], row["deletion_advantage_sd"]),
            format_value(row["insertion_advantage_mean"], row["insertion_advantage_sd"]),
            format_value(row["sensitivity_n_mean"], row["sensitivity_n_sd"]),
            format_value(row["normalized_infidelity_mean"], row["normalized_infidelity_sd"]),
            format_value(row["completeness_relative_error_mean"], row["completeness_relative_error_sd"], scientific=True),
            format_value(row["seconds_per_sample_mean"], row["seconds_per_sample_sd"], scientific=True),
        ]
        lines.append("        " + " & ".join(cells) + r" \\")
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        f"    \\caption{{Descriptive DQN explanation results on the {selected_seed_count} retained B-cos seeds. Values are means $\\pm$ standard deviations across the selected seeds. The experiment is a proof of concept and is not used for statistical architecture comparisons.}}",
        r"    \label{tab:rq2_dqn_proof_of_concept}",
        r"\end{table}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_examples(examples, output_base):
    figure, axes = plt.subplots(len(examples), 5, figsize=(12.5, 2.35 * len(examples)))
    axes = np.atleast_2d(axes)
    for row, (observation, action, attributions) in enumerate(examples):
        rgb = np.moveaxis(observation[:3], 0, -1).astype(np.float32) / 255.0
        axes[row, 0].imshow(rgb)
        name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
        axes[row, 0].set_ylabel(name, fontsize=10)
        for axis, method in zip(axes[row, 1:], METHODS):
            magnitude = np.abs(attributions[method].sum(axis=0))
            scale = max(float(np.percentile(magnitude, 99)), 1e-12)
            normalized = np.clip(magnitude / scale, 0.0, 1.0)
            axis.imshow(rgb)
            axis.imshow(
                normalized,
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                alpha=np.clip(normalized * 0.8, 0.0, 0.8),
            )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(
        axes[0], ("Observation", *(METHOD_LABELS[method] for method in METHODS))
    ):
        axis.set_title(title)
    figure.suptitle("DoorKey explanations on one B-cos DQN policy", fontsize=11)
    figure.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    parser.add_argument("--successful-episodes", type=int, default=30)
    parser.add_argument("--max-attempted-episodes", type=int, default=100)
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--smoothgrad-samples", type=int, default=8)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--evaluation-seed-offset", type=int, default=10_000)
    args = parser.parse_args()
    args.curve_fractions = [value / 10 for value in range(11)]
    args.sensitivity_fractions = [0.1, 0.25, 0.5, 0.75, 0.9]
    return args


def main():
    args = parse_args()
    model_specs = []
    for value in args.model:
        seed_text, path_text = value.split("=", 1)
        model_specs.append((int(seed_text), Path(path_text)))
    rows = []
    provenance = []
    examples = None
    for model_index, (training_seed, path) in enumerate(sorted(model_specs)):
        print(f"Loading seed {training_seed}: {path}", flush=True)
        model = load_model(path, args.device)
        observations, actions, collection = collect_successful_states(
            model, args, seed=args.evaluation_seed_offset + training_seed
        )
        dataset_path = args.output_dir / f"seed_{training_seed}_successful_states.npz"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, observations=observations, actions=actions)
        provenance.append({
            "training_seed": training_seed,
            "model_path": str(path.resolve()),
            "reported_final_success": {0: 1.00, 1: 0.99, 2: 0.96}[training_seed],
            **collection,
        })
        method_attributions = {}
        for method_index, method in enumerate(METHODS):
            print(f"  Evaluating {METHOD_LABELS[method]}", flush=True)
            metrics, attribution = evaluate_method(
                model,
                observations,
                actions,
                method,
                args,
                args.seed + model_index * 100000 + method_index * 10000,
            )
            rows.append({"training_seed": training_seed, "method": method, **metrics})
            method_attributions[method] = attribution
        if examples is None:
            examples = []
            for action in (2, 3, 5):
                candidates = np.flatnonzero(actions == action)
                if not len(candidates):
                    raise RuntimeError(f"No {ACTION_NAMES[action]} example in seed {training_seed}")
                index = int(candidates[0])
                examples.append((
                    observations[index],
                    int(actions[index]),
                    {method: values[index] for method, values in method_attributions.items()},
                ))

    aggregated = aggregate(rows)
    write_csv(args.output_dir / "per_seed_metrics.csv", rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregated)
    write_latex_table(
        args.output_dir / "rq2_dqn_proof_of_concept_table.tex",
        aggregated,
        len(model_specs),
    )
    assert examples is not None
    plot_examples(examples, args.output_dir / "rq2_dqn_explanation_example")
    result = {
        "scope": "descriptive proof of concept on B-cos DQN seeds 0 and 1",
        "selection_note": (
            "Seed 2 had 0.96 success in the original stochastic final evaluation "
            "but was excluded because it did not provide stable greedy successful "
            "rollouts for the explanation-data protocol."
        ),
        "methods": list(METHODS),
        "scalar_target": "selected Q-value minus mean Q-value over actions",
        "curve_score": "baseline-corrected relative-Q retention",
        "dataset_actions": "deterministic greedy DQN actions on successful trajectories",
        "samples_per_seed": args.samples_per_seed,
        "provenance": provenance,
        "per_seed": rows,
        "aggregate": aggregated,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregated, indent=2), flush=True)


if __name__ == "__main__":
    main()

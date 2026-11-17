#!/usr/bin/env python3
"""Evaluate B-cos explanation faithfulness on shared MiniGrid states.

The evaluator measures whether attribution rank predicts actual actor-output
changes. It reports deletion/insertion curves, matched-random controls, P/N
region interventions, attribution-effect correlations, and B-cos completeness.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from minigrid.core.grid import Grid

from evaluate_explanation_metrics import load_model
from evaluate_wall_ood import parse_models
from bcos_rl_generic.explanation_supervision import compute_actor_explanation
from bcos_rl_generic.explanation_supervision import relative_action_outputs


def parse_fractions(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("Fractions must include 0 and 1")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Fractions must lie in [0, 1]")
    if not any(0.0 < value < 1.0 for value in values):
        raise ValueError("Fractions must include at least one interior point")
    return values


def load_dataset(path: Path, max_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        required = {"observations", "actions", "masks"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
        observations = np.asarray(data["observations"][:max_samples])
        actions = np.asarray(data["actions"][:max_samples]).reshape(-1).astype(np.int64)
        masks = np.asarray(data["masks"][:max_samples]).astype(np.float32)
    if not (len(observations) == len(actions) == len(masks)) or not len(observations):
        raise ValueError("Dataset arrays must have the same positive length")
    if observations.ndim != 4 or observations.shape[1] not in (3, 6):
        raise ValueError(f"Expected NCHW RGB/RGB6 observations, got {observations.shape}")
    if masks.shape != (len(observations), *observations.shape[-2:]):
        raise ValueError(f"Mask/observation shape mismatch: {masks.shape}")
    return observations, actions, masks


def actor_outputs(model, observations: np.ndarray, actions: np.ndarray, batch_size: int):
    logits_parts, probabilities_parts = [], []
    model.policy.eval()
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device)
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
            raise RuntimeError(f"Expected one action_net call, captured {len(captured)}")
        logits_parts.append(captured[0].detach().cpu().numpy())
        probabilities_parts.append(distribution.probs.detach().cpu().numpy())
    logits = np.concatenate(logits_parts)
    relative_logits = logits - logits.mean(axis=1, keepdims=True)
    probabilities = np.concatenate(probabilities_parts)
    rows = np.arange(len(observations))
    return {
        "logits": logits,
        "relative_logits": relative_logits,
        "probabilities": probabilities,
        "selected_logits": relative_logits[rows, actions],
        "raw_selected_logits": logits[rows, actions],
        "selected_probabilities": probabilities[rows, actions],
        "argmax_actions": probabilities.argmax(axis=1),
    }


def actor_relative_logits(policy, observations: th.Tensor) -> th.Tensor:
    """Return relative actor logits while preserving input gradients."""
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        features = features[0]
    latent_pi = policy.mlp_extractor.forward_actor(features)
    logits = policy.action_net(latent_pi)
    if isinstance(logits, tuple):
        logits = logits[0]
    return relative_action_outputs(logits)


def explanations(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    method: str = "bcos",
    *,
    integrated_gradients_steps: int = 32,
    smoothgrad_samples: int = 8,
):
    parts = []
    model.policy.eval()
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        obs = th.as_tensor(observations[start:stop], device=model.device).float()
        action = th.as_tensor(actions[start:stop], device=model.device).long()
        with th.enable_grad():
            if method == "bcos":
                attribution = compute_actor_explanation(
                    model.policy,
                    obs,
                    action,
                    create_graph=False,
                    subtract_action_mean=bool(
                        getattr(model, "explanation_subtract_action_mean", True)
                    ),
                )
            elif method == "saliency":
                obs = obs.detach().requires_grad_(True)
                attribution = Saliency(
                    lambda value: actor_relative_logits(model.policy, value)
                ).attribute(obs, target=action, abs=False)
            elif method == "integrated_gradients":
                obs = obs.detach().requires_grad_(True)
                attribution = IntegratedGradients(
                    lambda value: actor_relative_logits(model.policy, value)
                ).attribute(
                    obs,
                    baselines=th.zeros_like(obs),
                    target=action,
                    n_steps=integrated_gradients_steps,
                    method="gausslegendre",
                )
            elif method == "smoothgrad":
                obs = obs.detach().requires_grad_(True)
                attribution = NoiseTunnel(
                    Saliency(lambda value: actor_relative_logits(model.policy, value))
                ).attribute(
                    obs,
                    target=action,
                    nt_type="smoothgrad",
                    nt_samples=smoothgrad_samples,
                    stdevs=0.02 * 255.0,
                    abs=False,
                )
            else:
                raise ValueError(f"Unsupported explanation method: {method}")
        parts.append(attribution.detach().cpu().numpy())
    return np.concatenate(parts)


def empty_grid_substrate(shape: tuple[int, int, int], grid_size: int) -> np.ndarray:
    channels, height, width = shape
    if channels not in (3, 6) or height % grid_size or width % grid_size:
        raise ValueError(f"Cannot construct empty-grid substrate for {shape}")
    tile_h, tile_w = height // grid_size, width // grid_size
    if tile_h != tile_w:
        raise ValueError("MiniGrid substrate requires square rendered tiles")
    tile = Grid.render_tile(
        None, agent_dir=None, highlight=False, tile_size=tile_h
    ).astype(np.uint8)
    rgb = np.tile(tile, (grid_size, grid_size, 1))
    hwc = np.concatenate((rgb, 255 - rgb), axis=-1) if channels == 6 else rgb
    return np.moveaxis(hwc, -1, 0)


def ranking_order(magnitude: np.ndarray, mode: str, rng: np.random.Generator):
    flat = magnitude.reshape(len(magnitude), -1)
    if mode == "high":
        return np.argsort(-flat, axis=1, kind="stable")
    if mode == "low":
        return np.argsort(flat, axis=1, kind="stable")
    if mode == "random":
        return np.argsort(rng.random(flat.shape), axis=1)
    raise ValueError(mode)


def perturb_by_order(
    observations: np.ndarray,
    substrate: np.ndarray,
    order: np.ndarray,
    fraction: float,
    *,
    insertion: bool,
) -> np.ndarray:
    result = (
        np.broadcast_to(substrate, observations.shape).copy()
        if insertion
        else observations.copy()
    )
    count = int(round(fraction * order.shape[1]))
    if count == 0:
        return result
    for sample in range(len(observations)):
        indices = order[sample, :count]
        target = result[sample].reshape(result.shape[1], -1)
        source = (
            observations[sample].reshape(observations.shape[1], -1)
            if insertion
            else substrate.reshape(substrate.shape[0], -1)
        )
        target[:, indices] = source[:, indices]
    return result


def perturb_by_mask(
    observations: np.ndarray,
    substrate: np.ndarray,
    spatial_masks: np.ndarray,
    *,
    retain: bool,
) -> np.ndarray:
    result = (
        np.broadcast_to(substrate, observations.shape).copy()
        if retain
        else observations.copy()
    )
    for sample, spatial_mask in enumerate(spatial_masks):
        indices = np.flatnonzero(spatial_mask.reshape(-1))
        target = result[sample].reshape(result.shape[1], -1)
        source = (
            observations[sample].reshape(observations.shape[1], -1)
            if retain
            else substrate.reshape(substrate.shape[0], -1)
        )
        target[:, indices] = source[:, indices]
    return result


def output_change(original: dict, changed: dict, actions: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    p = np.clip(original["probabilities"], eps, 1.0)
    q = np.clip(changed["probabilities"], eps, 1.0)
    probability_drop = (
        original["selected_probabilities"] - changed["selected_probabilities"]
    )
    logit_drop = original["selected_logits"] - changed["selected_logits"]
    return {
        "selected_probability": float(changed["selected_probabilities"].mean()),
        "probability_retention": float(
            changed["selected_probabilities"].mean()
            / (original["selected_probabilities"].mean() + eps)
        ),
        "selected_probability_drop": float(probability_drop.mean()),
        "selected_logit_drop": float(logit_drop.mean()),
        "mean_absolute_selected_logit_change": float(np.abs(logit_drop).mean()),
        "policy_kl_from_original": float(np.sum(p * (np.log(p) - np.log(q)), axis=1).mean()),
        "target_action_argmax_rate": float(np.mean(changed["argmax_actions"] == actions)),
    }


def average_dicts(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([value[key] for value in values]))
        for key in values[0]
    }


def curve(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    original: dict,
    substrate: np.ndarray,
    magnitude: np.ndarray,
    fractions: list[float],
    mode: str,
    batch_size: int,
    seed: int,
    random_repeats: int,
    *,
    insertion: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    repeats = random_repeats if mode == "random" else 1
    per_repeat = []
    per_repeat_probabilities = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 1009 * repeat)
        order = ranking_order(magnitude, mode, rng)
        points = {}
        selected_probabilities = {}
        for fraction in fractions:
            changed_obs = perturb_by_order(
                observations, substrate, order, fraction, insertion=insertion
            )
            changed = actor_outputs(model, changed_obs, actions, batch_size)
            key = f"{fraction:g}"
            points[key] = output_change(original, changed, actions)
            selected_probabilities[key] = changed["selected_probabilities"]
        per_repeat.append(points)
        per_repeat_probabilities.append(selected_probabilities)
    averaged_points = {
        key: average_dicts([repeat[key] for repeat in per_repeat])
        for key in per_repeat[0]
    }
    averaged_probabilities = {
        key: np.mean(
            np.stack([repeat[key] for repeat in per_repeat_probabilities]), axis=0
        )
        for key in per_repeat_probabilities[0]
    }
    return averaged_points, averaged_probabilities


def auc(curve_points: dict[str, dict[str, float]], key: str) -> float:
    x = np.asarray(sorted(float(value) for value in curve_points))
    y = np.asarray([curve_points[f"{value:g}"][key] for value in x])
    return float(np.trapz(y, x))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool = False) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2:
        return 0.0
    if ranked:
        left, right = rankdata(left), rankdata(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_interval(
    sample_count: int,
    statistic,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    estimate = float(statistic(np.arange(sample_count)))
    if repeats <= 0:
        return {"estimate": estimate, "low": estimate, "high": estimate}
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, sample_count, size=sample_count)
        values[index] = statistic(sample)
    return {
        "estimate": estimate,
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
    }


def sample_curve_auc(
    probabilities: dict[str, np.ndarray],
    original_probabilities: np.ndarray,
    indices: np.ndarray,
) -> float:
    x = np.asarray(sorted(float(value) for value in probabilities))
    denominator = original_probabilities[indices].mean() + 1e-8
    y = np.asarray(
        [probabilities[f"{value:g}"][indices].mean() / denominator for value in x]
    )
    return float(np.trapz(y, x))


def bootstrap_curve_advantages(
    deletion: dict[str, dict[str, np.ndarray]],
    insertion: dict[str, dict[str, np.ndarray]],
    original_probabilities: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    count = len(original_probabilities)
    return {
        "deletion_advantage_over_random": bootstrap_interval(
            count,
            lambda indices: sample_curve_auc(
                deletion["random"], original_probabilities, indices
            )
            - sample_curve_auc(deletion["high"], original_probabilities, indices),
            repeats=repeats,
            seed=seed,
        ),
        "insertion_advantage_over_random": bootstrap_interval(
            count,
            lambda indices: sample_curve_auc(
                insertion["high"], original_probabilities, indices
            )
            - sample_curve_auc(insertion["random"], original_probabilities, indices),
            repeats=repeats,
            seed=seed + 1,
        ),
    }


def matched_random_masks(
    masks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = np.zeros_like(masks, dtype=bool)
    pixel_count = masks.shape[1] * masks.shape[2]
    for sample, mask in enumerate(masks):
        count = int(mask.sum())
        if count:
            selected = rng.choice(pixel_count, size=count, replace=False)
            result[sample].reshape(-1)[selected] = True
    return result


def region_metrics(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    original: dict,
    substrate: np.ndarray,
    region: np.ndarray,
    spatial_attribution: np.ndarray,
    spatial_magnitude: np.ndarray | None,
    batch_size: int,
    seed: int,
    random_repeats: int,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    deleted_obs = perturb_by_mask(observations, substrate, region, retain=False)
    retained_obs = perturb_by_mask(observations, substrate, region, retain=True)
    deleted = actor_outputs(model, deleted_obs, actions, batch_size)
    retained = actor_outputs(model, retained_obs, actions, batch_size)
    deleted_change = output_change(original, deleted, actions)
    retained_change = output_change(original, retained, actions)

    random_deleted_changes, random_retained_changes = [], []
    random_deleted_probabilities, random_retained_probabilities = [], []
    for repeat in range(random_repeats):
        random_region = matched_random_masks(
            region, np.random.default_rng(seed + 2027 * repeat)
        )
        random_deleted = actor_outputs(
            model,
            perturb_by_mask(observations, substrate, random_region, retain=False),
            actions,
            batch_size,
        )
        random_retained = actor_outputs(
            model,
            perturb_by_mask(observations, substrate, random_region, retain=True),
            actions,
            batch_size,
        )
        random_deleted_changes.append(output_change(original, random_deleted, actions))
        random_retained_changes.append(output_change(original, random_retained, actions))
        random_deleted_probabilities.append(random_deleted["selected_probabilities"])
        random_retained_probabilities.append(random_retained["selected_probabilities"])

    attribution_sum = (spatial_attribution * region).sum(axis=(1, 2))
    if spatial_magnitude is None:
        spatial_magnitude = np.abs(spatial_attribution)
    attribution_magnitude = (spatial_magnitude * region).sum(axis=(1, 2))
    actual_effect = original["selected_logits"] - deleted["selected_logits"]
    random_deletion = average_dicts(random_deleted_changes)
    random_retention = average_dicts(random_retained_changes)
    original_probability = original["selected_probabilities"]
    deletion_drop = original_probability - deleted["selected_probabilities"]
    retention_gap = original_probability - retained["selected_probabilities"]
    random_deleted_probability = np.mean(
        np.stack(random_deleted_probabilities), axis=0
    )
    random_retained_probability = np.mean(
        np.stack(random_retained_probabilities), axis=0
    )
    random_deletion_drop = original_probability - random_deleted_probability
    random_retention_gap = original_probability - random_retained_probability
    deletion_advantage = deletion_drop - random_deletion_drop
    retention_advantage = random_retention_gap - retention_gap
    count = len(observations)
    return {
        "pixel_fraction": float(region.mean()),
        "deletion": deleted_change,
        "retention_only": retained_change,
        "matched_random_deletion": random_deletion,
        "matched_random_retention_only": random_retention,
        "deletion_probability_drop_advantage_over_random": float(
            deleted_change["selected_probability_drop"]
            - random_deletion["selected_probability_drop"]
        ),
        "retention_probability_gap_advantage_over_random": float(
            random_retention["selected_probability_drop"]
            - retained_change["selected_probability_drop"]
        ),
        "signed_attribution_vs_logit_effect_pearson": correlation(
            attribution_sum, actual_effect
        ),
        "signed_attribution_vs_logit_effect_spearman": correlation(
            attribution_sum, actual_effect, ranked=True
        ),
        "magnitude_vs_absolute_logit_effect_spearman": correlation(
            attribution_magnitude, np.abs(actual_effect), ranked=True
        ),
        "bootstrap_95_ci": {
            "deletion_probability_drop_advantage_over_random": bootstrap_interval(
                count,
                lambda indices: deletion_advantage[indices].mean(),
                repeats=bootstrap_repeats,
                seed=seed + 101,
            ),
            "retention_probability_gap_advantage_over_random": bootstrap_interval(
                count,
                lambda indices: retention_advantage[indices].mean(),
                repeats=bootstrap_repeats,
                seed=seed + 102,
            ),
            "signed_attribution_vs_logit_effect_spearman": bootstrap_interval(
                count,
                lambda indices: correlation(
                    attribution_sum[indices], actual_effect[indices], ranked=True
                ),
                repeats=bootstrap_repeats,
                seed=seed + 103,
            ),
            "magnitude_vs_absolute_logit_effect_spearman": bootstrap_interval(
                count,
                lambda indices: correlation(
                    attribution_magnitude[indices],
                    np.abs(actual_effect[indices]),
                    ranked=True,
                ),
                repeats=bootstrap_repeats,
                seed=seed + 104,
            ),
        },
    }


def sensitivity_n_and_infidelity(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    original: dict,
    spatial_attribution: np.ndarray,
    fractions: list[float],
    batch_size: int,
    seed: int,
    random_repeats: int,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    """Evaluate random-subset Sensitivity-n and zero-baseline infidelity.

    Sensitivity-n correlates summed attribution with the selected-logit change
    after deleting the same feature subset. Pearson correlation is the primary
    metric, matching Ancona et al.; Spearman is retained as a secondary robust
    diagnostic. Infidelity is the squared mismatch between those two quantities.
    A zero substrate is used because B-cos contributions are defined relative
    to zero input.
    """
    zero_substrate = np.zeros(observations.shape[1:], dtype=observations.dtype)
    magnitude = np.abs(spatial_attribution)
    results: dict[str, Any] = {}
    all_predicted, all_actual = [], []
    for fraction_index, fraction in enumerate(fractions):
        if fraction <= 0.0 or fraction >= 1.0:
            continue
        predicted_parts, actual_parts = [], []
        for repeat in range(random_repeats):
            rng = np.random.default_rng(
                seed + fraction_index * 10_007 + repeat * 1009
            )
            order = ranking_order(magnitude, "random", rng)
            changed_obs = perturb_by_order(
                observations,
                zero_substrate,
                order,
                fraction,
                insertion=False,
            )
            changed = actor_outputs(model, changed_obs, actions, batch_size)
            count = int(round(fraction * order.shape[1]))
            flat_attr = spatial_attribution.reshape(len(observations), -1)
            selected = np.take_along_axis(flat_attr, order[:, :count], axis=1)
            predicted_parts.append(selected.sum(axis=1))
            actual_parts.append(
                original["selected_logits"] - changed["selected_logits"]
            )
        predicted = np.concatenate(predicted_parts)
        actual = np.concatenate(actual_parts)
        squared_error = np.square(predicted - actual)
        normalized_infidelity = squared_error / (np.square(actual).mean() + 1e-8)
        pair_count = len(predicted)
        results[f"{fraction:g}"] = {
            "sensitivity_n_pearson": correlation(predicted, actual),
            "sensitivity_n_spearman": correlation(predicted, actual, ranked=True),
            "infidelity_mse": float(squared_error.mean()),
            "normalized_infidelity": float(normalized_infidelity.mean()),
            "bootstrap_95_ci": {
                "sensitivity_n_pearson": bootstrap_interval(
                    pair_count,
                    lambda indices: correlation(predicted[indices], actual[indices]),
                    repeats=bootstrap_repeats,
                    seed=seed + fraction_index * 31,
                ),
                "sensitivity_n_spearman": bootstrap_interval(
                    pair_count,
                    lambda indices: correlation(
                        predicted[indices], actual[indices], ranked=True
                    ),
                    repeats=bootstrap_repeats,
                    seed=seed + fraction_index * 31 + 1,
                ),
                "normalized_infidelity": bootstrap_interval(
                    pair_count,
                    lambda indices: normalized_infidelity[indices].mean(),
                    repeats=bootstrap_repeats,
                    seed=seed + fraction_index * 31 + 2,
                ),
            },
        }
        all_predicted.append(predicted)
        all_actual.append(actual)

    predicted = np.concatenate(all_predicted)
    actual = np.concatenate(all_actual)
    squared_error = np.square(predicted - actual)
    normalized = squared_error / (np.square(actual).mean() + 1e-8)
    fixed_fraction_results = [
        value for fraction, value in results.items() if fraction != "aggregate"
    ]
    results["aggregate"] = {
        "sensitivity_n_pearson": float(
            np.mean(
                [value["sensitivity_n_pearson"] for value in fixed_fraction_results]
            )
        ),
        "sensitivity_n_spearman": float(
            np.mean(
                [value["sensitivity_n_spearman"] for value in fixed_fraction_results]
            )
        ),
        "pooled_across_fractions_pearson_diagnostic": correlation(predicted, actual),
        "pooled_across_fractions_spearman_diagnostic": correlation(
            predicted, actual, ranked=True
        ),
        "infidelity_mse": float(squared_error.mean()),
        "normalized_infidelity": float(normalized.mean()),
        "interpretation": (
            "Sensitivity-n PCC is computed separately for each fixed subset "
            "fraction and averaged across fractions; higher is better. Lower "
            "infidelity is better; Spearman and correlations pooled across "
            "fractions are secondary diagnostics"
        ),
    }
    return results


def evaluate_model(
    model,
    observations: np.ndarray,
    stored_actions: np.ndarray,
    masks: np.ndarray,
    substrate: np.ndarray,
    fractions: list[float],
    batch_size: int,
    seed: int,
    random_repeats: int,
    bootstrap_repeats: int,
    action_source: str,
    method: str = "bcos",
    integrated_gradients_steps: int = 32,
    smoothgrad_samples: int = 8,
) -> dict[str, Any]:
    if action_source == "model":
        actions = np.asarray(model.predict(observations, deterministic=True)[0]).reshape(-1)
    else:
        actions = stored_actions
    actions = actions.astype(np.int64)
    original = actor_outputs(model, observations, actions, batch_size)
    attribution_started = time.perf_counter()
    attribution = explanations(
        model,
        observations,
        actions,
        batch_size,
        method,
        integrated_gradients_steps=integrated_gradients_steps,
        smoothgrad_samples=smoothgrad_samples,
    )
    attribution_seconds = time.perf_counter() - attribution_started
    spatial_attribution = attribution.sum(axis=1)
    magnitude = np.abs(spatial_attribution)

    deletion, insertion = {}, {}
    deletion_samples, insertion_samples = {}, {}
    for mode in ("high", "random", "low"):
        deletion[mode], deletion_samples[mode] = curve(
            model, observations, actions, original, substrate, magnitude,
            fractions, mode, batch_size, seed, random_repeats, insertion=False,
        )
        insertion[mode], insertion_samples[mode] = curve(
            model, observations, actions, original, substrate, magnitude,
            fractions, mode, batch_size, seed, random_repeats, insertion=True,
        )

    deletion_auc = {
        mode: auc(points, "probability_retention") for mode, points in deletion.items()
    }
    insertion_auc = {
        mode: auc(points, "probability_retention") for mode, points in insertion.items()
    }
    positive = masks > 0
    negative = masks < 0
    selected_logits = original["selected_logits"]
    zero_outputs = actor_outputs(
        model, np.zeros_like(observations), actions, batch_size
    )
    target_difference = selected_logits - zero_outputs["selected_logits"]
    attribution_sum = spatial_attribution.sum(axis=(1, 2))
    completeness_error = attribution_sum - target_difference
    curve_bootstrap = bootstrap_curve_advantages(
        deletion_samples,
        insertion_samples,
        original["selected_probabilities"],
        repeats=bootstrap_repeats,
        seed=seed + 20_000,
    )
    return {
        "samples": int(len(observations)),
        "actions": action_source,
        "method": method,
        "attribution_runtime": {
            "total_seconds": float(attribution_seconds),
            "seconds_per_sample": float(attribution_seconds / len(observations)),
        },
        "mean_original_selected_probability": float(
            original["selected_probabilities"].mean()
        ),
        "completeness": {
            "mean_attribution_sum": float(attribution_sum.mean()),
            "mean_selected_logit": float(selected_logits.mean()),
            "mean_relative_selected_logit": float(selected_logits.mean()),
            "mean_zero_baseline_target_difference": float(target_difference.mean()),
            "mean_absolute_error": float(np.abs(completeness_error).mean()),
            "relative_absolute_error": float(
                np.abs(completeness_error).mean()
                / (np.abs(target_difference).mean() + 1e-8)
            ),
            "pearson": correlation(attribution_sum, target_difference),
            "guaranteed_by_method": method in ("bcos", "integrated_gradients"),
        },
        "deletion": deletion,
        "insertion": insertion,
        "summary": {
            "deletion_auc_probability_retention": deletion_auc,
            "deletion_advantage_over_random": deletion_auc["random"] - deletion_auc["high"],
            "insertion_auc_probability_retention": insertion_auc,
            "insertion_advantage_over_random": insertion_auc["high"] - insertion_auc["random"],
            "bootstrap_95_ci": curve_bootstrap,
        },
        "regions": {
            "P": region_metrics(
                model, observations, actions, original, substrate, positive,
                spatial_attribution, None, batch_size, seed + 30000, random_repeats,
                bootstrap_repeats,
            ),
            "N": region_metrics(
                model, observations, actions, original, substrate, negative,
                spatial_attribution, None, batch_size, seed + 40000, random_repeats,
                bootstrap_repeats,
            ),
        },
        "sensitivity_n_and_infidelity": sensitivity_n_and_infidelity(
            model,
            observations,
            actions,
            original,
            spatial_attribution,
            fractions,
            batch_size,
            seed + 50_000,
            random_repeats,
            bootstrap_repeats,
        ),
    }


def evaluate_deletion_insertion_sensitivity(
    model,
    observations: np.ndarray,
    stored_actions: np.ndarray,
    substrate: np.ndarray,
    fractions: list[float],
    batch_size: int,
    seed: int,
    random_repeats: int,
    bootstrap_repeats: int,
    action_source: str,
    method: str = "bcos",
    integrated_gradients_steps: int = 32,
    smoothgrad_samples: int = 8,
) -> dict[str, Any]:
    """Evaluate deletion/insertion under an explicitly secondary substrate."""
    if action_source == "model":
        actions = np.asarray(model.predict(observations, deterministic=True)[0]).reshape(-1)
    else:
        actions = stored_actions
    actions = actions.astype(np.int64)
    original = actor_outputs(model, observations, actions, batch_size)
    attribution = explanations(
        model,
        observations,
        actions,
        batch_size,
        method,
        integrated_gradients_steps=integrated_gradients_steps,
        smoothgrad_samples=smoothgrad_samples,
    )
    magnitude = np.abs(attribution.sum(axis=1))

    deletion, insertion = {}, {}
    deletion_samples, insertion_samples = {}, {}
    for mode in ("high", "random", "low"):
        deletion[mode], deletion_samples[mode] = curve(
            model, observations, actions, original, substrate, magnitude,
            fractions, mode, batch_size, seed, random_repeats, insertion=False,
        )
        insertion[mode], insertion_samples[mode] = curve(
            model, observations, actions, original, substrate, magnitude,
            fractions, mode, batch_size, seed, random_repeats, insertion=True,
        )

    deletion_auc = {
        mode: auc(points, "probability_retention") for mode, points in deletion.items()
    }
    insertion_auc = {
        mode: auc(points, "probability_retention") for mode, points in insertion.items()
    }
    return {
        "samples": int(len(observations)),
        "actions": action_source,
        "method": method,
        "deletion": deletion,
        "insertion": insertion,
        "summary": {
            "deletion_auc_probability_retention": deletion_auc,
            "deletion_advantage_over_random": deletion_auc["random"] - deletion_auc["high"],
            "insertion_auc_probability_retention": insertion_auc,
            "insertion_advantage_over_random": insertion_auc["high"] - insertion_auc["random"],
            "bootstrap_95_ci": bootstrap_curve_advantages(
                deletion_samples,
                insertion_samples,
                original["selected_probabilities"],
                repeats=bootstrap_repeats,
                seed=seed + 20_000,
            ),
        },
    }


def save_plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    model_names = list(result["models"])
    include_zero = any(
        "zero_substrate_sensitivity" in model
        for model in result["models"].values()
    )
    curve_specs = [
        ("faithfulness", "deletion", "valid-grid deletion"),
        ("faithfulness", "insertion", "valid-grid insertion"),
    ]
    if include_zero:
        curve_specs.extend(
            [
                ("zero_substrate_sensitivity", "deletion", "all-zero deletion"),
                ("zero_substrate_sensitivity", "insertion", "all-zero insertion"),
            ]
        )
    figure, axes = plt.subplots(
        len(curve_specs),
        len(model_names),
        figsize=(5 * len(model_names), 4 * len(curve_specs)),
        sharex=True,
    )
    if len(model_names) == 1:
        axes = np.asarray(axes).reshape(len(curve_specs), 1)
    for column, name in enumerate(model_names):
        for row, (section, curve_name, label) in enumerate(curve_specs):
            axis = axes[row, column]
            model_result = result["models"][name]
            if section not in model_result:
                axis.set_visible(False)
                continue
            metrics = model_result[section]
            for mode in ("high", "random", "low"):
                points = metrics[curve_name][mode]
                x = sorted(float(value) for value in points)
                y = [points[f"{value:g}"]["probability_retention"] for value in x]
                axis.plot(x, y, marker="o", label=mode)
            axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
            axis.set_title(f"{name}: {label}")
            axis.set_xlabel("spatial pixel fraction")
            axis.set_ylabel("selected-action probability retention")
            axis.grid(alpha=0.25)
            axis.legend()
    figure.suptitle("MiniGrid B-cos explanation faithfulness and substrate sensitivity")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--dataset-in", type=Path, default=Path("evaluation_states_3.npz"))
    parser.add_argument("--actions", choices=("model", "stored"), default="model")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--fractions", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--secondary-zero-substrate",
        action="store_true",
        help="Also run deletion/insertion with every observation channel set to zero.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("minigrid_faithfulness_comparison.json")
    )
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = make_args()
    if (
        args.max_samples <= 0
        or args.batch_size <= 0
        or args.random_repeats <= 0
        or args.bootstrap_repeats <= 0
    ):
        raise ValueError("Sample, batch, random-repeat, and bootstrap counts must be positive")
    fractions = parse_fractions(args.fractions)
    model_paths = parse_models(args.model)
    observations, stored_actions, masks = load_dataset(args.dataset_in, args.max_samples)
    substrate = empty_grid_substrate(tuple(observations.shape[1:]), args.grid_size)
    models = {name: load_model(path, args.device) for name, path in model_paths.items()}
    expected_shape = tuple(observations.shape[1:])
    mismatched = {
        name: tuple(model.observation_space.shape)
        for name, model in models.items()
        if tuple(model.observation_space.shape) != expected_shape
    }
    if mismatched:
        raise ValueError(f"Model/dataset observation mismatch: {mismatched}")

    result: dict[str, Any] = {
        "dataset": str(args.dataset_in.resolve()),
        "samples": int(len(observations)),
        "action_source": args.actions,
        "substrate": "valid rendered empty MiniGrid tiles",
        "substrates": {
            "primary": "valid rendered empty MiniGrid tiles",
            "secondary": (
                "all observation channels zero (off-manifold sensitivity analysis)"
                if args.secondary_zero_substrate
                else None
            ),
        },
        "random_repeats": args.random_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "fractions": fractions,
        "methods": {
            "deletion_insertion": "Petsiuk et al. (RISE), BMVC 2018",
            "comprehensiveness_sufficiency": "DeYoung et al. (ERASER), ACL 2020",
            "sensitivity_n": "Ancona et al., ICLR 2018",
            "infidelity": "Yeh et al., NeurIPS 2019",
        },
        "interpretation": {
            "deletion": "lower high-attribution AUC and positive random-minus-high advantage are better",
            "insertion": "higher high-attribution AUC and positive high-minus-random advantage are better",
            "regions": "compare P/N interventions with their equal-area random controls",
            "limitation": "pixel interventions remain counterfactual inputs; interpret them with matched random controls and alongside environment-level OOD tests",
            "substrate_scope": "primary claims use the valid rendered-grid substrate; the all-zero substrate is secondary sensitivity analysis only",
        },
        "models": {},
    }
    for name, model in models.items():
        print(f"Evaluating faithfulness for {name}", flush=True)
        metrics = evaluate_model(
            model,
            observations,
            stored_actions,
            masks,
            substrate,
            fractions,
            args.batch_size,
            args.seed,
            args.random_repeats,
            args.bootstrap_repeats,
            args.actions,
        )
        result["models"][name] = {
            "path": str(model_paths[name].resolve()),
            "saved_timesteps": int(model.num_timesteps),
            "saved_explanation_coef": float(getattr(model, "explanation_coef", 0.0)),
            "faithfulness": metrics,
        }
        if args.secondary_zero_substrate:
            zero_substrate = np.zeros_like(substrate)
            result["models"][name]["zero_substrate_sensitivity"] = (
                evaluate_deletion_insertion_sensitivity(
                    model,
                    observations,
                    stored_actions,
                    zero_substrate,
                    fractions,
                    args.batch_size,
                    args.seed + 100_000,
                    args.random_repeats,
                    args.bootstrap_repeats,
                    args.actions,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_path = args.plot or args.output.with_suffix(".png")
    save_plot(result, plot_path)
    print("\nFaithfulness summary (larger advantages are better):")
    for name, model_result in result["models"].items():
        faithfulness = model_result["faithfulness"]
        summary = faithfulness["summary"]
        deletion_ci = summary["bootstrap_95_ci"]["deletion_advantage_over_random"]
        insertion_ci = summary["bootstrap_95_ci"]["insertion_advantage_over_random"]
        sensitivity = faithfulness["sensitivity_n_and_infidelity"]["aggregate"]
        print(
            f"- {name}: deletion_advantage="
            f"{summary['deletion_advantage_over_random']:.4f} "
            f"[{deletion_ci['low']:.4f}, {deletion_ci['high']:.4f}], "
            f"insertion_advantage={summary['insertion_advantage_over_random']:.4f} "
            f"[{insertion_ci['low']:.4f}, {insertion_ci['high']:.4f}], "
            f"Sensitivity-n(Pearson)={sensitivity['sensitivity_n_pearson']:.4f}, "
            f"normalized_infidelity={sensitivity['normalized_infidelity']:.4f}"
        )
        if "zero_substrate_sensitivity" in model_result:
            zero = model_result["zero_substrate_sensitivity"]["summary"]
            print(
                f"  secondary all-zero: deletion_advantage="
                f"{zero['deletion_advantage_over_random']:.4f}, "
                f"insertion_advantage={zero['insertion_advantage_over_random']:.4f}"
            )
    print(f"Saved metrics to: {args.output}")
    print(f"Saved curves to: {plot_path}")


if __name__ == "__main__":
    main()

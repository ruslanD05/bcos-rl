#!/usr/bin/env python3
"""Evaluate strict RQ3 PPO branches under matched wall-color shifts."""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

from evaluate_explanation_metrics import load_model
from evaluate_wall_ood import wall_settings
from run_rq3_evaluation import locate_model
from train_minigrid_ActorCritic import build_vec_env


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("pure", "sequential", "local")


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=ROOT / "experiments" / "rq3_explanation_control" / "continuations",
    )
    parser.add_argument(
        "--pure-models-root",
        type=Path,
        default=ROOT / "experiments" / "rq3_explanation_control" / "pure_artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "rq5_wall_colour" / "wall_color_evaluation",
    )
    parser.add_argument("--variants", default="normal,red,blue,purple,yellow")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-seed", type=int, default=30_000)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Recompute aggregates from output-dir/per_seed_results.csv without evaluation.",
    )
    parser.add_argument("--open-door-reward", type=float, default=0.1)
    parser.add_argument("--pick-key-reward", type=float, default=0.05)
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--seeds must contain unique comma-separated integers")
    return values


def parse_variants(raw: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    allowed = {"normal", "red", "blue", "purple", "yellow", "grey", "hidden"}
    if not values or len(values) != len(set(values)) or not set(values) <= allowed:
        raise ValueError(f"Invalid wall variants: {values}")
    if "normal" not in values:
        raise ValueError("--variants must include normal for matched robustness drops")
    return values


def bootstrap_mean_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def stable_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def make_env(args: argparse.Namespace, *, seed: int, variant: str):
    appearance, color = wall_settings(variant)
    return build_vec_env(
        env_id="MiniGrid-DoorKey-8x8-v0",
        n_envs=args.n_envs,
        seed=seed,
        use_rgb_img_obs=True,
        invert_channels=True,
        wall_appearance=appearance,
        wall_color=color,
        open_door_reward=args.open_door_reward,
        pick_key_reward=args.pick_key_reward,
    )


def evaluate_vector_policy(model, env, *, episodes: int, deterministic: bool) -> dict[str, float]:
    rewards = np.zeros(env.num_envs, dtype=np.float64)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []
    successes: list[float] = []
    observations = env.reset()

    while len(completed_rewards) < episodes:
        actions, _ = model.predict(observations, deterministic=deterministic)
        observations, step_rewards, dones, infos = env.step(actions)
        rewards += np.asarray(step_rewards, dtype=np.float64)
        lengths += 1
        for index in np.flatnonzero(dones):
            if len(completed_rewards) >= episodes:
                break
            completed_rewards.append(float(rewards[index]))
            completed_lengths.append(int(lengths[index]))
            successes.append(float(infos[index].get("is_success", 0.0)))
            rewards[index] = 0.0
            lengths[index] = 0

    return {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(completed_rewards)),
        "std_reward": float(np.std(completed_rewards)),
        "mean_episode_length": float(np.mean(completed_lengths)),
        "std_episode_length": float(np.std(completed_lengths)),
        "success_rate": float(np.mean(successes)),
    }


def evaluate(args: argparse.Namespace, seeds: list[int], variants: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for training_seed in seeds:
        for branch in BRANCHES:
            root = args.pure_models_root if branch == "pure" else args.models_root
            model_path = locate_model(root, training_seed, branch)
            model = load_model(model_path, args.device)
            shape = tuple(model.observation_space.shape)
            if len(shape) != 3 or shape[0] != 6:
                raise ValueError(f"Expected RGB6 model, got {shape} for {model_path}")

            for variant in variants:
                evaluation_seed = args.eval_seed + training_seed
                np.random.seed(evaluation_seed)
                th.manual_seed(evaluation_seed)
                env = make_env(args, seed=evaluation_seed, variant=variant)
                try:
                    metrics = evaluate_vector_policy(
                        model, env,
                        episodes=args.episodes,
                        deterministic=args.deterministic,
                    )
                finally:
                    env.close()
                rows.append(
                    {
                        "training_seed": training_seed,
                        "branch": branch,
                        "variant": variant,
                        "model_path": str(model_path.resolve()),
                        "saved_timesteps": int(model.num_timesteps),
                        "evaluation_seed": evaluation_seed,
                        **metrics,
                    }
                )
                print(
                    f"seed={training_seed} branch={branch} wall={variant}: "
                    f"success={metrics['success_rate']:.3f}, "
                    f"steps={metrics['mean_episode_length']:.2f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def aggregate(per_run: pd.DataFrame, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, object]] = []
    for (branch, variant), group in per_run.groupby(["branch", "variant"], sort=False):
        for metric in ("success_rate", "mean_reward", "mean_episode_length"):
            values = group[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_ci(
                values, repeats, seed=stable_seed(branch, variant, metric)
            )
            aggregate_rows.append(
                {
                    "branch": branch,
                    "variant": variant,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": len(values),
                }
            )

    normal = per_run[per_run["variant"] == "normal"][
        ["training_seed", "branch", "success_rate"]
    ].rename(columns={"success_rate": "normal_success"})
    shifted = per_run[per_run["variant"] != "normal"].merge(
        normal, on=["training_seed", "branch"], validate="many_to_one"
    )
    shifted["success_drop_from_normal"] = (
        shifted["normal_success"] - shifted["success_rate"]
    )
    drop_rows: list[dict[str, object]] = []
    for (branch, variant), group in shifted.groupby(["branch", "variant"], sort=False):
        values = group["success_drop_from_normal"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(
            values, repeats, seed=stable_seed(branch, variant, "drop")
        )
        drop_rows.append(
            {
                "branch": branch,
                "variant": variant,
                "mean_success_drop": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "n_seeds": len(values),
            }
        )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(drop_rows)


def paired_branch_differences(per_run: pd.DataFrame, repeats: int) -> pd.DataFrame:
    pivot = per_run.pivot(
        index=["training_seed", "variant"], columns="branch", values="success_rate"
    ).reset_index()
    rows: list[dict[str, object]] = []
    for branch in ("sequential", "local"):
        for variant, group in pivot.groupby("variant", sort=False):
            differences = (group[branch] - group["pure"]).to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_ci(
                differences, repeats, seed=stable_seed(branch, variant, "minus_pure")
            )
            rows.append(
                {
                    "branch": branch,
                    "variant": variant,
                    "mean_success_difference_vs_pure": float(differences.mean()),
                    "sample_sd": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                    "improved_seeds": int((differences > 0).sum()),
                    "tied_seeds": int((differences == 0).sum()),
                    "worsened_seeds": int((differences < 0).sum()),
                    "n_seeds": len(differences),
                }
            )
    return pd.DataFrame(rows)


def save_figure(aggregates: pd.DataFrame, variants: list[str], output_dir: Path) -> None:
    success = aggregates[aggregates["metric"] == "success_rate"]
    labels = {"pure": "Pure PPO", "sequential": "Sequential", "local": "Local action"}
    colors = {"pure": "#4c78a8", "sequential": "#f58518", "local": "#54a24b"}
    x = np.arange(len(variants))
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    for branch in BRANCHES:
        rows = success[success["branch"] == branch].set_index("variant").loc[variants]
        mean = rows["mean"].to_numpy()
        errors = np.vstack((mean - rows["ci95_low"].to_numpy(), rows["ci95_high"].to_numpy() - mean))
        axis.errorbar(
            x,
            mean,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=3,
            label=labels[branch],
            color=colors[branch],
        )
    axis.set_xticks(x, ["Normal" if value == "normal" else value.title() for value in variants])
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Success rate")
    axis.set_xlabel("Wall rendering")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="lower left")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"rq3_wall_color_success.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = make_args()
    if args.episodes <= 0 or args.bootstrap_repeats <= 0 or args.n_envs <= 0:
        raise ValueError("Episodes, environments, and bootstrap repeats must be positive")
    seeds = parse_ints(args.seeds)
    variants = parse_variants(args.variants)
    th.set_num_threads(1)
    th.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.output_dir / "per_seed_results.csv"
    if args.reuse_results:
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing reusable results: {result_path}")
        per_run = pd.read_csv(result_path)
    else:
        per_run = evaluate(args, seeds, variants)
    aggregates, drops = aggregate(per_run, args.bootstrap_repeats)
    comparisons = paired_branch_differences(per_run, args.bootstrap_repeats)
    per_run.to_csv(result_path, index=False)
    aggregates.to_csv(args.output_dir / "aggregate_statistics.csv", index=False)
    drops.to_csv(args.output_dir / "success_drops_from_normal.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_success_differences_vs_pure.csv", index=False)
    save_figure(aggregates, variants, args.output_dir)

    payload = {
        "protocol": {
            "environment": "MiniGrid-DoorKey-8x8-v0",
            "observation": "RGB6",
            "episodes_per_model_variant": args.episodes,
            "in_process_environments": args.n_envs,
            "policy_action_mode": "deterministic" if args.deterministic else "stochastic",
            "training_seeds": seeds,
            "wall_variants": variants,
            "evaluation_seed_base": args.eval_seed,
            "bootstrap_repeats": args.bootstrap_repeats,
        },
        "aggregate_statistics": aggregates.to_dict(orient="records"),
        "success_drops_from_normal": drops.to_dict(orient="records"),
        "paired_success_differences_vs_pure": comparisons.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved wall-color evaluation to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

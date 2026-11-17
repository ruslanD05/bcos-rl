#!/usr/bin/env python3
"""Build the curated, reproducible model registry used by the thesis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "thesis_model_registry"


def entry(
    model_id: str,
    source: str,
    *,
    domain: str,
    experiment: str,
    role: str,
    rewards: dict[str, float] | None = None,
    config: str | None = None,
    notes: str = "",
) -> dict:
    return {
        "model_id": model_id,
        "source": source,
        "domain": domain,
        "experiment": experiment,
        "role": role,
        "rewards": rewards or {},
        "config_source": config,
        "notes": notes,
    }


DOORKEY_REWARDS = {"goal": 1.0, "open_door": 0.1, "pick_key": 0.05}
DOORKEY_SPARSE_REWARDS = {"goal": 1.0, "open_door": 0.0, "pick_key": 0.0}

MODELS = [
    entry(
        "minigrid_ppo_rgb3_baseline_503808",
        "experiments/wall_mask_ood_EXPL_LOSS/models/baseline_ppo_503808.zip",
        domain="minigrid-doorkey-8x8",
        experiment="wall-mask-faithfulness",
        role="shared-pretraining-baseline",
        rewards=DOORKEY_REWARDS,
        config="experiments/wall_mask_ood_EXPL_LOSS/configs/experiment_config.json",
    ),
    entry(
        "minigrid_ppo_rgb3_pure_655360",
        "experiments/minigrid_faithfulness_evaluation/models/pure_bcos_ppo_rgb3_655360steps.zip",
        domain="minigrid-doorkey-8x8",
        experiment="wall-mask-faithfulness",
        role="pure-control",
        rewards=DOORKEY_REWARDS,
        config="experiments/wall_mask_ood_EXPL_LOSS/configs/experiment_config.json",
    ),
    entry(
        "minigrid_ppo_rgb3_correct_mask_c015_655360",
        "experiments/minigrid_faithfulness_evaluation/models/correct_mask_bcos_ppo_rgb3_coef0.15_655360steps.zip",
        domain="minigrid-doorkey-8x8",
        experiment="wall-mask-faithfulness",
        role="correct-sequential-mask",
        rewards=DOORKEY_REWARDS,
        config="experiments/wall_mask_ood_EXPL_LOSS/configs/experiment_config.json",
    ),
    entry(
        "minigrid_ppo_rgb3_flipped_mask_c015_655360",
        "experiments/minigrid_faithfulness_evaluation/models/flipped_mask_bcos_ppo_rgb3_coef0.15_655360steps.zip",
        domain="minigrid-doorkey-8x8",
        experiment="wall-mask-faithfulness",
        role="flipped-mask-negative-control",
        rewards=DOORKEY_REWARDS,
        config="experiments/wall_mask_ood_EXPL_LOSS/configs/flipped_mask_config.json",
    ),
    entry(
        "minigrid_ppo_rgb3_curriculum_pure_655360",
        "experiments/explanation_curriculum_comparison/models/pure_ppo_control_655360.zip",
        domain="minigrid-doorkey-8x8",
        experiment="explanation-curriculum",
        role="pure-control",
        rewards=DOORKEY_REWARDS,
        config="experiments/explanation_curriculum_comparison/configs/control_config.json",
    ),
    entry(
        "minigrid_ppo_rgb3_curriculum_linear_c015_655360",
        "experiments/explanation_curriculum_comparison/models/linear_from_scratch_ppo_655360_coef_0.15.zip",
        domain="minigrid-doorkey-8x8",
        experiment="explanation-curriculum",
        role="linear-from-scratch",
        rewards=DOORKEY_REWARDS,
        config="experiments/explanation_curriculum_comparison/configs/linear_config.json",
    ),
    entry(
        "minigrid_ppo_rgb3_curriculum_success_gated_c015_655360",
        "experiments/explanation_curriculum_comparison/models/success_gated_ppo_655360_coef_0.15.zip",
        domain="minigrid-doorkey-8x8",
        experiment="explanation-curriculum",
        role="success-gated",
        rewards=DOORKEY_REWARDS,
        config="experiments/explanation_curriculum_comparison/configs/success_gated_config.json",
    ),
    entry(
        "minigrid_ppo_rgb3_confidence_pure_655360",
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/ExplLoss/pure_ppo_655360_control.zip",
        domain="minigrid-doorkey-8x8",
        experiment="confidence-sharpening",
        role="pure-control",
        rewards=DOORKEY_REWARDS,
        notes="Configuration is recoverable from the SB3 archive; no separate run config was preserved.",
    ),
    entry(
        "minigrid_ppo_rgb3_confidence_gated_c0003_655360",
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/AuxLossAdv/final_success_gated_confidence.zip",
        domain="minigrid-doorkey-8x8",
        experiment="confidence-sharpening",
        role="positive-advantage-confidence",
        rewards=DOORKEY_REWARDS,
        notes="Success-gated confidence coefficient 0.003.",
    ),
    entry(
        "minigrid_ppo_rgb6_pure_1001472",
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/6Chan/model (29).zip",
        domain="minigrid-doorkey-8x8",
        experiment="rgb6-local-action-mask",
        role="shared-pretraining-baseline",
        rewards=DOORKEY_SPARSE_REWARDS,
        notes="Six-channel [RGB, inverse RGB] checkpoint trained without intermediate rewards.",
    ),
    entry(
        "minigrid_ppo_rgb6_absolute_energy_c020_1003520",
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/6Chan/model (30).zip",
        domain="minigrid-doorkey-8x8",
        experiment="rgb6-floor-robustness",
        role="sequential-explanation",
        rewards=DOORKEY_REWARDS,
    ),
    entry(
        "minigrid_ppo_rgb6_floor_study_pure_1003520",
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/6Chan/model (31).zip",
        domain="minigrid-doorkey-8x8",
        experiment="rgb6-floor-robustness",
        role="pure-control",
        rewards=DOORKEY_REWARDS,
    ),
    entry(
        "minigrid_ppo_rgb6_local_action_c015_1153024",
        "minigrid_part/runs/pretrained_mask_ablation/local_action_mask_coef015_20260801_124821/final_local-action-masked.zip",
        domain="minigrid-doorkey-8x8",
        experiment="rgb6-local-action-mask",
        role="local-action-mask-final",
        rewards=DOORKEY_REWARDS,
        config="minigrid_part/runs/pretrained_mask_ablation/local_action_mask_coef015_20260801_124821/config.json",
    ),
    *[
        entry(
            f"minigrid_ppo_rgb6_local_action_c015_after_{steps}",
            "minigrid_part/runs/pretrained_mask_ablation/"
            "local_action_mask_coef015_20260801_124821/checkpoints/"
            f"local-action-masked_after_{steps:07d}_steps.zip",
            domain="minigrid-doorkey-8x8",
            experiment="rgb6-local-action-mask",
            role="local-action-temporal-checkpoint",
            rewards=DOORKEY_REWARDS,
            config="minigrid_part/runs/pretrained_mask_ablation/local_action_mask_coef015_20260801_124821/config.json",
        )
        for steps in (50_000, 100_000, 150_000)
    ],
    entry(
        "minigrid_dqn_rgb3_baseline_500000",
        "experiments/dqn_advantage_mask_ablation/models/baseline_500k/baseline_0500000.zip",
        domain="minigrid-doorkey-8x8",
        experiment="dqn-advantage-mask-ablation",
        role="shared-pretraining-baseline",
        rewards=DOORKEY_REWARDS,
    ),
    *[
        entry(
            f"minigrid_dqn_rgb3_{name}_650000",
            f"experiments/dqn_advantage_mask_ablation/models/{folder}/model_0650000_final.zip",
            domain="minigrid-doorkey-8x8",
            experiment="dqn-advantage-mask-ablation",
            role=role,
            rewards=DOORKEY_REWARDS,
            config=f"experiments/dqn_advantage_mask_ablation/configs/{folder}.json",
        )
        for name, folder, role in (
            ("pure", "pure_control", "pure-control"),
            ("all_transitions", "all_transitions", "explanation-all-transitions"),
            ("flip_positive", "flip_positive", "flip-positive-on-negative-advantage"),
            ("positive_only", "positive_only", "positive-advantage-only"),
        )
    ],
    *[
        entry(
            model_id,
            f"experiments/carracing_explanation_mask_comparison/models/{filename}",
            domain="carracing-v3-discrete",
            experiment="carracing-mask-comparison",
            role=role,
            rewards={"environment_default": 1.0},
            config=f"experiments/carracing_explanation_mask_comparison/configs/{config}",
        )
        for model_id, filename, role, config in (
            ("carracing_ppo_bcos_source_1015808", "source_bcos_ppo_1015808.zip", "shared-pretraining-baseline", "source_baseline.json"),
            ("carracing_ppo_bcos_pure_1269760", "pure_bcos_ppo_1269760.zip", "pure-control", "pure_control.json"),
            ("carracing_ppo_bcos_road_negative_c010_1269760", "road_negative_mask_bcos_ppo_1269760_coef_0.1.zip", "road-negative-mask", "road_negative_mask.json"),
            ("carracing_ppo_bcos_road_grass_negative_c030_1269760", "road_grass_negative_mask_bcos_ppo_1269760_coef_0.3.zip", "road-grass-negative-mask", "road_grass_negative_mask.json"),
            ("carracing_ppo_bcos_car_road_positive_c050_1269760", "car_road_positive_mask_bcos_ppo_1269760_coef_0.5.zip", "car-road-positive-mask", "car_road_positive_mask.json"),
        )
    ],
]


def sha256(path: Path) -> str:
    for attempt in range(5):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise AssertionError("unreachable")


def sb3_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    policy_kwargs = data.get("policy_kwargs", {})
    observation_space = data.get("observation_space", {})
    shape = observation_space.get("_shape", []) if isinstance(observation_space, dict) else []
    algorithm = "dqn" if "buffer_size" in data else "ppo" if "n_steps" in data else "unknown"
    scalar_keys = (
        "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
        "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
        "buffer_size", "learning_starts", "train_freq", "gradient_steps",
        "target_update_interval", "exploration_fraction", "exploration_final_eps",
        "explanation_coef", "explanation_loss_type", "explanation_max_grad_ratio",
        "confidence_coef", "bce_coef",
    )
    hyperparameters = {
        key: data[key]
        for key in scalar_keys
        if key in data and isinstance(data[key], (str, int, float, bool, type(None)))
    }
    return {
        "algorithm": algorithm,
        "timesteps": int(data.get("num_timesteps", 0) or 0),
        "seed": data.get("seed", "unknown"),
        "observation_shape": list(shape) if isinstance(shape, list) else [],
        "actor_type": policy_kwargs.get("actor_type", "unknown") if isinstance(policy_kwargs, dict) else "unknown",
        "critic_type": policy_kwargs.get("critic_type", "unknown") if isinstance(policy_kwargs, dict) else "unknown",
        "bcos_b": policy_kwargs.get("bcos_b", "unknown") if isinstance(policy_kwargs, dict) else "unknown",
        "hyperparameters": hyperparameters,
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def main() -> None:
    models_dir = OUTPUT / "models"
    configs_dir = OUTPUT / "configs"
    models_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    missing = []
    for specification in MODELS:
        source = ROOT / specification["source"]
        if not source.exists():
            missing.append(specification["source"])
            continue
        destination = models_dir / specification["domain"] / f"{specification['model_id']}.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        os.chmod(destination, stat.S_IREAD | stat.S_IWRITE)
        source_hash = sha256(source)
        copied_hash = sha256(destination)
        if source_hash != copied_hash:
            raise RuntimeError(f"Hash mismatch after copying {source}")

        source_config_destination = None
        config_hash = None
        if specification["config_source"]:
            config_source = ROOT / specification["config_source"]
            if config_source.exists():
                source_config_destination = (
                    configs_dir
                    / f"{specification['model_id']}.source{config_source.suffix}"
                )
                shutil.copyfile(config_source, source_config_destination)
                os.chmod(source_config_destination, stat.S_IREAD | stat.S_IWRITE)
                config_hash = sha256(source_config_destination)

        archive_metadata = sb3_metadata(source)
        normalized_config_destination = configs_dir / f"{specification['model_id']}.json"
        normalized_config = {
            "model_id": specification["model_id"],
            "domain": specification["domain"],
            "experiment": specification["experiment"],
            "role": specification["role"],
            "rewards": specification["rewards"],
            "archive_metadata": archive_metadata,
            "source_model": specification["source"],
            "source_config": specification["config_source"],
            "notes": specification["notes"],
        }
        normalized_config_destination.write_text(
            json.dumps(normalized_config, indent=2), encoding="utf-8"
        )
        record = {
            **specification,
            **archive_metadata,
            "canonical_path": destination.relative_to(ROOT).as_posix(),
            "sha256": copied_hash,
            "bytes": destination.stat().st_size,
            "normalized_config_path": normalized_config_destination.relative_to(
                ROOT
            ).as_posix(),
            "copied_source_config_path": (
                source_config_destination.relative_to(ROOT).as_posix()
                if source_config_destination else None
            ),
            "config_sha256": config_hash,
            "metadata_provenance": {
                "timesteps_observation_architecture_hyperparameters": "SB3 archive data entry",
                "rewards_role_experiment": "curated experiment records",
                "config": (
                    "normalized archive metadata plus copied source file"
                    if source_config_destination
                    else "normalized archive metadata; separate source file not preserved"
                ),
            },
        }
        manifest.append(record)

    if missing:
        raise FileNotFoundError("Missing curated model sources:\n" + "\n".join(missing))

    revision = {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "working_tree_dirty": bool(git_output("status", "--porcelain")),
        "working_tree_status": git_output("status", "--short").splitlines(),
        "registry_builder": "tools/build_thesis_model_registry.py",
    }
    (OUTPUT / "code_revision.json").write_text(
        json.dumps(revision, indent=2), encoding="utf-8"
    )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    checksum_lines = [
        f"{row['sha256']}  {Path(row['canonical_path']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()}"
        for row in manifest
    ]
    (OUTPUT / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )

    lines = [
        "# Thesis Model Registry",
        "",
        "Curated checkpoints that directly support thesis experiments. The registry",
        "is generated by `python tools/build_thesis_model_registry.py`.",
        "",
        f"- Models: {len(manifest)}",
        f"- Unique SHA-256 hashes: {len({row['sha256'] for row in manifest})}",
        f"- Git commit: `{revision['git_commit']}`",
        f"- Working tree dirty when generated: `{revision['working_tree_dirty']}`",
        "",
        "## Integrity",
        "",
        "Every copied archive is hashed after copying. `manifest.json` is the",
        "authoritative machine-readable index. Unknown metadata is left unknown",
        "rather than inferred from filenames.",
        "",
        "## Models",
        "",
    ]
    for row in manifest:
        lines.extend(
            [
                f"### {row['model_id']}",
                f"- Experiment: `{row['experiment']}`",
                f"- Role: `{row['role']}`",
                f"- Algorithm: `{row['algorithm']}`",
                f"- Timesteps: `{row['timesteps']}`",
                f"- Observation: `{row['observation_shape']}`",
                f"- Seed: `{row['seed']}`",
                f"- SHA-256: `{row['sha256']}`",
                f"- Model: `{row['canonical_path']}`",
                f"- Normalized config: `{row['normalized_config_path']}`",
                f"- Source config copy: `{row['copied_source_config_path'] or 'not separately preserved'}`",
                "",
            ]
        )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Built registry with {len(manifest)} models at {OUTPUT}")


if __name__ == "__main__":
    main()

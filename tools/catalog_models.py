#!/usr/bin/env python3
"""Build a deduplicated, metadata-rich catalog of SB3 model archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile


EXCLUDED_PARTS = {".git", ".venv", ".uv-cache", "model_catalog"}

FEATURED_MODELS = {
    # MiniGrid PPO floor-color randomization.
    "minigrid/doorkey/ppo/floor_color_randomization/pure_bcos_ppo_rgb6_1003520steps.zip":
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/6Chan/model (31).zip",
    "minigrid/doorkey/ppo/floor_color_randomization/absolute_energy_bcos_ppo_rgb6_coef0.20_1003520steps.zip":
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/6Chan/model (30).zip",
    # MiniGrid PPO correct/flipped wall-mask experiment.
    "minigrid/doorkey/ppo/wall_mask_ablation/baseline_bcos_ppo_rgb3_503808steps.zip":
        "experiments/wall_mask_ood_EXPL_LOSS/models/baseline_ppo_503808.zip",
    "minigrid/doorkey/ppo/wall_mask_ablation/pure_control_bcos_ppo_rgb3_655360steps.zip":
        "experiments/wall_mask_ood_EXPL_LOSS/models/pure_ppo_655360_control.zip",
    "minigrid/doorkey/ppo/wall_mask_ablation/correct_mask_bcos_ppo_rgb3_coef0.15_655360steps.zip":
        "experiments/wall_mask_ood_EXPL_LOSS/models/correct_mask_ppo_655360_coef_0.15.zip",
    "minigrid/doorkey/ppo/wall_mask_ablation/flipped_mask_bcos_ppo_rgb3_coef0.15_655360steps.zip":
        "experiments/wall_mask_ood_EXPL_LOSS/models/flipped_mask_ppo_655360_coef_0.15.zip",
    # Explanation timing/curriculum experiment.
    "minigrid/doorkey/ppo/explanation_curriculum/pure_control_bcos_ppo_rgb3_655360steps.zip":
        "experiments/explanation_curriculum_comparison/models/pure_ppo_control_655360.zip",
    "minigrid/doorkey/ppo/explanation_curriculum/linear_from_start_coef0.15_655360steps.zip":
        "experiments/explanation_curriculum_comparison/models/linear_from_scratch_ppo_655360_coef_0.15.zip",
    "minigrid/doorkey/ppo/explanation_curriculum/success_gated_coef0.15_655360steps.zip":
        "experiments/explanation_curriculum_comparison/models/success_gated_ppo_655360_coef_0.15.zip",
    # Advantage-conditioned confidence experiment.
    "minigrid/doorkey/ppo/confidence_loss/pure_control_bcos_ppo_rgb3_655360steps.zip":
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/ExplLoss/pure_ppo_655360_control.zip",
    "minigrid/doorkey/ppo/confidence_loss/success_gated_confidence_coef0.003_655360steps.zip":
        "minigrid_part/models/PPO/DoorKey/ImgObsFull/AuxLossAdv/final_success_gated_confidence.zip",
    # DQN explanation/advantage-mask branches.
    "minigrid/doorkey/dqn/advantage_mask_ablation/baseline_pure_bcos_dqn_rgb3_500000steps.zip":
        "experiments/dqn_advantage_mask_ablation/models/baseline_500k/baseline_0500000.zip",
    "minigrid/doorkey/dqn/advantage_mask_ablation/pure_control_bcos_dqn_rgb3_650000steps.zip":
        "experiments/dqn_advantage_mask_ablation/models/pure_control/model_0650000_final.zip",
    "minigrid/doorkey/dqn/advantage_mask_ablation/all_transitions_explanation_coef0.10_650000steps.zip":
        "experiments/dqn_advantage_mask_ablation/models/all_transitions/model_0650000_final.zip",
    "minigrid/doorkey/dqn/advantage_mask_ablation/flip_positive_on_negative_advantage_coef0.10_650000steps.zip":
        "experiments/dqn_advantage_mask_ablation/models/flip_positive/model_0650000_final.zip",
    "minigrid/doorkey/dqn/advantage_mask_ablation/positive_advantage_only_coef0.10_650000steps.zip":
        "experiments/dqn_advantage_mask_ablation/models/positive_only/model_0650000_final.zip",
    # CarRacing PPO mask comparison.
    "carracing/ppo/mask_comparison/source_pure_bcos_ppo_1015808steps.zip":
        "experiments/carracing_explanation_mask_comparison/models/source_bcos_ppo_1015808.zip",
    "carracing/ppo/mask_comparison/pure_control_bcos_ppo_1269760steps.zip":
        "experiments/carracing_explanation_mask_comparison/models/pure_bcos_ppo_1269760.zip",
    "carracing/ppo/mask_comparison/road_negative_mask_coef0.10_1269760steps.zip":
        "experiments/carracing_explanation_mask_comparison/models/road_negative_mask_bcos_ppo_1269760_coef_0.1.zip",
    "carracing/ppo/mask_comparison/road_and_grass_negative_mask_coef0.30_1269760steps.zip":
        "experiments/carracing_explanation_mask_comparison/models/road_grass_negative_mask_bcos_ppo_1269760_coef_0.3.zip",
    "carracing/ppo/mask_comparison/car_and_road_positive_mask_coef0.50_1269760steps.zip":
        "experiments/carracing_explanation_mask_comparison/models/car_road_positive_mask_bcos_ppo_1269760_coef_0.5.zip",
}


def slug(value: object, *, limit: int = 48) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return (text or "unknown")[:limit].rstrip("-")


def numeric(data: dict, key: str) -> float:
    value = data.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sb3_data(path: Path) -> tuple[dict, str | None]:
    try:
        with zipfile.ZipFile(path) as archive:
            if "data" not in archive.namelist():
                return {}, "missing SB3 data entry"
            return json.loads(archive.read("data")), None
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        return {}, str(error)


def classify_path(relative: Path) -> tuple[str, str]:
    value = relative.as_posix().lower()
    if "car_racing" in value or "carracing" in value:
        return "carracing", "carracing-v3"
    if "pgtg" in value:
        return "pgtg", "pgtg"
    if "pointing" in value:
        return "minigrid", "pointing-game"
    if "holeenv" in value or "hole_env" in value:
        return "minigrid", "hole-env"
    if "composite" in value:
        return "minigrid", "composite"
    if "doorkey" in value or "minigrid" in value or "dqn_advantage" in value:
        return "minigrid", "doorkey-8x8"
    return "unknown", "unknown"


def metadata(path: Path, root: Path) -> dict:
    relative = path.relative_to(root)
    data, error = read_sb3_data(path)
    policy_kwargs = data.get("policy_kwargs", {})
    if not isinstance(policy_kwargs, dict):
        policy_kwargs = {}
    observation_space = data.get("observation_space", {})
    shape = observation_space.get("_shape", []) if isinstance(observation_space, dict) else []
    shape = [int(value) for value in shape] if isinstance(shape, list) else []

    if "buffer_size" in data:
        algorithm = "dqn"
    elif "n_steps" in data:
        algorithm = "ppo"
    else:
        algorithm = "unknown"
    actor = policy_kwargs.get("actor_type") or policy_kwargs.get("net_type")
    if not actor:
        extractor = policy_kwargs.get("features_extractor_kwargs", {})
        actor = "bcos" if isinstance(extractor, dict) and extractor.get("use_bcos") else "standard"

    explanation_coef = max(
        numeric(data, "explanation_coef"),
        numeric(data, "action_explanation_coef"),
        numeric(data, "sequential_explanation_coef"),
    )
    confidence_coef = max(
        numeric(data, "confidence_coef"),
        numeric(data, "bce_coef"),
        numeric(data, "aux_alpha"),
    )
    if explanation_coef > 0 and confidence_coef > 0:
        role = "combined"
    elif explanation_coef > 0:
        role = "explanation"
    elif confidence_coef > 0:
        role = "confidence"
    else:
        role = "pure"

    domain, task = classify_path(relative)
    source_parts = [slug(part, limit=24) for part in relative.with_suffix("").parts[-3:]]
    return {
        "source_path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "is_sb3": error is None,
        "metadata_error": error or "",
        "domain": domain,
        "task": task,
        "algorithm": algorithm,
        "actor": slug(actor),
        "critic": slug(policy_kwargs.get("critic_type", "unknown")),
        "observation_shape": "x".join(str(value) for value in shape) or "unknown",
        "timesteps": int(data.get("num_timesteps", 0) or 0),
        "seed": data.get("seed") if isinstance(data.get("seed"), int) else "unknown",
        "role": role,
        "explanation_coef": explanation_coef,
        "explanation_loss_type": str(data.get("explanation_loss_type", "")),
        "confidence_coef": confidence_coef,
        "bcos_b": policy_kwargs.get("bcos_b", ""),
        "source_label": "-".join(source_parts),
    }


def preferred_source(item: dict) -> tuple[int, int, str]:
    path = item["source_path"].lower()
    if "/experiments/" in f"/{path}":
        rank = 0
    elif "/models/" in f"/{path}":
        rank = 1
    elif "final" in path:
        rank = 2
    elif "best_model" in path:
        rank = 3
    else:
        rank = 4
    return rank, len(path), path


def coefficient(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def canonical_name(item: dict) -> str:
    details = []
    if item["explanation_coef"] > 0:
        loss = slug(item["explanation_loss_type"] or "unknown", limit=24)
        details.append(f"expl-{loss}-c{coefficient(item['explanation_coef'])}")
    if item["confidence_coef"] > 0:
        details.append(f"conf-c{coefficient(item['confidence_coef'])}")
    detail = "_" + "_".join(details) if details else ""
    name = (
        f"{item['algorithm']}_{item['actor']}_obs-{item['observation_shape']}"
        f"_t{item['timesteps']}_{item['role']}{detail}_seed-{item['seed']}"
        f"_{item['source_label']}_{item['sha256'][:8]}.zip"
    )
    return slug(name.removesuffix(".zip"), limit=170) + ".zip"


def write_manifest(output: Path, items: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(items[0]) if items else []
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)
    (output / "manifest.json").write_text(json.dumps(items, indent=2), encoding="utf-8")


def link_models(root: Path, output: Path, items: list[dict]) -> tuple[int, int]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        if item["is_sb3"]:
            groups.setdefault(item["sha256"], []).append(item)

    linked = copied = 0
    for digest, aliases in groups.items():
        source_item = min(aliases, key=preferred_source)
        source = root / source_item["source_path"]
        destination = (
            output
            / "models"
            / source_item["domain"]
            / source_item["task"]
            / source_item["algorithm"]
            / source_item["role"]
            / canonical_name(source_item)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
                linked += 1
            except OSError:
                shutil.copy2(source, destination)
                copied += 1
        canonical = destination.relative_to(root).as_posix()
        for alias in aliases:
            alias["canonical_path"] = canonical
            alias["duplicate_count"] = len(aliases)
    return linked, copied


def write_readme(output: Path, items: list[dict], unique_count: int) -> None:
    domains: dict[str, int] = {}
    roles: dict[str, int] = {}
    for item in items:
        domains[item["domain"]] = domains.get(item["domain"], 0) + 1
        roles[item["role"]] = roles.get(item["role"], 0) + 1
    text = f"""# Model Catalog

Generated by `tools/catalog_models.py`.

- Source archives: {len(items)}
- Unique SB3 checkpoints: {unique_count}
- Original files are preserved in place.
- Canonical files are hard links where supported, so they consume no duplicate model storage.

## Layout

`models/<domain>/<task>/<algorithm>/<role>/`

Roles are inferred from saved coefficients: `pure`, `explanation`, `confidence`, or `combined`.
Every canonical filename includes algorithm, actor type, observation shape, timesteps,
loss/coefficient information, seed, source context, and an eight-character SHA-256 suffix.

`manifest.csv` and `manifest.json` map every original archive to its canonical checkpoint
and identify byte-identical duplicates.

## Counts

Domains: {json.dumps(domains, sort_keys=True)}

Roles: {json.dumps(roles, sort_keys=True)}
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def create_featured(root: Path, output: Path) -> None:
    rows = []
    for destination_value, source_value in FEATURED_MODELS.items():
        source = root / source_value
        if not source.is_file():
            raise FileNotFoundError(f"Featured model source is missing: {source}")
        destination = output / "featured" / destination_value
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
                link_mode = "hardlink"
            except OSError:
                shutil.copy2(source, destination)
                link_mode = "copy"
        else:
            link_mode = "existing"
        rows.append(
            {
                "featured_path": destination.relative_to(root).as_posix(),
                "source_path": source.relative_to(root).as_posix(),
                "sha256": sha256(source),
                "link_mode": link_mode,
            }
        )
    featured = output / "featured"
    with (featured / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        relative = Path(row["featured_path"]).relative_to(Path("model_catalog") / "featured")
        group = relative.parent.as_posix()
        grouped.setdefault(group, []).append(row)
    sections = []
    for group, group_rows in grouped.items():
        entries = "\n".join(
            f"- `{Path(row['featured_path']).name}`\n"
            f"  Source: `{row['source_path']}`"
            for row in group_rows
        )
        sections.append(f"## {group}\n\n{entries}")
    (featured / "README.md").write_text(
        "# Featured Thesis Models\n\n"
        "This directory contains short, explicitly named aliases for final models used in "
        "the main thesis experiments. Files are hard links to their preserved source "
        "checkpoints. See `manifest.csv` for provenance and SHA-256 hashes.\n\n"
        "The source checkpoints remain in their original locations so existing experiment "
        "commands and training provenance are not broken.\n\n"
        + "\n\n".join(sections)
        + "\n",
        encoding="utf-8",
    )


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("model_catalog"))
    parser.add_argument("--featured-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = make_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.featured_only:
        create_featured(root, output)
        print(f"Featured catalog: {output / 'featured'}")
        return
    candidates = sorted(
        path
        for path in root.rglob("*.zip")
        if not (set(path.relative_to(root).parts) & EXCLUDED_PARTS)
    )
    print(f"Inspecting {len(candidates)} archives...")
    items = [metadata(path, root) for path in candidates]
    linked, copied = link_models(root, output, items)
    unique_count = len({item["sha256"] for item in items if item["is_sb3"]})
    write_manifest(output, items)
    write_readme(output, items, unique_count)
    create_featured(root, output)
    print(
        f"Cataloged {len(items)} sources as {unique_count} unique SB3 checkpoints "
        f"({linked} hard links, {copied} copies)."
    )
    print(f"Catalog: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Restore the ten RQ1 DoorKey PPO checkpoints from their W&B artifacts.

Only the run IDs used by the committed paired statistical export are restored.
Each download is verified against the SHA-256 recorded in the original local
manifest before it is placed in its canonical run directory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "minigrid_ppo_architecture"
ENTITY = "ruslandontsov2005-saarland-informatics-campus"
PROJECT = "rq1-minigrid-ppo-architecture-final"
RUNS = {
    "bcos_b2.3": ["jf3kqwlr", "k3q5a4e7", "fqw5hyn9", "ufe7rnq3", "1mgfx45e"],
    "standard": ["5yl3iohu", "hz6bih6o", "j8uwewk8", "vd8cpzm4", "7w02e9i4"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    api = wandb.Api()
    base = ROOT / "experiments" / "rq1_architecture" / EXPERIMENT / "minigrid_doorkey_8x8"
    restored = 0
    for variant, run_ids in RUNS.items():
        for seed, run_id in enumerate(run_ids):
            target = base / variant / f"seed_{seed}" / run_id / "final_model.zip"
            manifest_path = target.with_name("manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest["sha256"]
            if target.exists():
                actual = sha256(target)
                if actual != expected:
                    raise RuntimeError(f"Existing file has wrong hash: {target}")
                print(f"Already verified: {target.relative_to(ROOT)}")
                continue

            artifact_name = f"{ENTITY}/{PROJECT}/{EXPERIMENT}-{run_id}:v0"
            with tempfile.TemporaryDirectory(prefix=f"rq1-{run_id}-") as temporary:
                download_dir = Path(temporary)
                api.artifact(artifact_name).download(root=str(download_dir))
                downloaded = download_dir / "final_model.zip"
                actual = sha256(downloaded)
                if actual != expected:
                    raise RuntimeError(
                        f"Hash mismatch for {run_id}: expected {expected}, got {actual}"
                    )
                shutil.move(str(downloaded), str(target))
            print(f"Restored and verified: {target.relative_to(ROOT)}")
            restored += 1
    print(f"Completed: {restored} checkpoint(s) restored; all targets verified.")


if __name__ == "__main__":
    main()

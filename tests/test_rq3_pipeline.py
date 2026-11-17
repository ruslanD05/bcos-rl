from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import aggregate_rq3_ppo as aggregate
import evaluate_local_mask_experiment as evaluation
from run_rq3_evaluation import DEFAULT_BRANCHES, DEFAULT_SEEDS, locate_model
from sweep_rq3_ppo_continuations import BRANCH_TO_MASK_MODE, source_checkpoint


class RQ3PipelineTests(unittest.TestCase):
    def test_final_rq3_uses_completed_channelwise_seeds(self) -> None:
        self.assertEqual(DEFAULT_SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(DEFAULT_BRANCHES, ("pure", "sequential", "local", "flipped"))

    def test_locate_model_supports_flat_wandb_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "rq3-local-seed2-example-v0" / "final_model.zip"
            expected.parent.mkdir()
            expected.write_bytes(b"model")
            self.assertEqual(locate_model(root, 2, "local"), expected)

    def test_rq3_branch_grid_is_complete(self) -> None:
        self.assertEqual(
            BRANCH_TO_MASK_MODE,
            {
                "pure": "pure",
                "sequential": "sequential-all-negative",
                "local": "local-directional-masked",
                "flipped": "flipped-sequential",
            },
        )

    def test_source_checkpoint_requires_one_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_dir = root / "seed_2" / "run"
            seed_dir.mkdir(parents=True)
            expected = seed_dir / "final_model.zip"
            expected.write_bytes(b"model")
            self.assertEqual(source_checkpoint(root, 2), expected)

            other = root / "seed_2" / "other" / "final_model.zip"
            other.parent.mkdir()
            other.write_bytes(b"model")
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                source_checkpoint(root, 2)

    def test_pnd_energy_partitions_total(self) -> None:
        attribution = np.ones((1, 2, 2, 2), dtype=np.float32)
        mask = np.asarray([[[1, -1], [0, 0]]], dtype=np.float32)
        with patch.object(evaluation, "explanations", return_value=attribution):
            result = evaluation.pnd_absolute_energy(
                object(),
                np.zeros((1, 6, 2, 2), dtype=np.float32),
                np.asarray([0]),
                {"test": mask},
                1,
            )["test"]
        self.assertAlmostEqual(result["positive_total_energy_ratio"], 0.25)
        self.assertAlmostEqual(result["negative_total_energy_ratio"], 0.25)
        self.assertAlmostEqual(result["neutral_total_energy_ratio"], 0.5)

    def test_seed_bootstrap_reports_independent_seed_count(self) -> None:
        rng = np.random.default_rng(0)
        result = aggregate.bootstrap_mean(np.asarray([1.0, 2.0, 3.0]), 100, rng)
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertEqual(result["n"], 3)


if __name__ == "__main__":
    unittest.main()

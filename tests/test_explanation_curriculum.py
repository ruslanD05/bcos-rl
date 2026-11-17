from argparse import Namespace
import os
from pathlib import Path
import sys
import unittest

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from train_explanation_curriculum_minigrid import (
    ExplanationCurriculumCallback,
    resolve_ramp_timesteps,
)


def make_callback(mode: str) -> ExplanationCurriculumCallback:
    return ExplanationCurriculumCallback(
        mode=mode,
        start_coef=0.0,
        end_coef=0.2,
        ramp_timesteps=100,
        checkpoint_dir=Path("unused"),
        checkpoint_freq=50,
        success_threshold=0.5,
        success_window=10,
        min_success_episodes=5,
        verbose=0,
    )


class ExplanationCurriculumTests(unittest.TestCase):
    def assert_coefficient(
        self,
        actual: tuple[float, float],
        expected: tuple[float, float],
    ) -> None:
        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])

    def test_linear_curriculum_uses_global_training_progress(self):
        callback = make_callback("linear")

        self.assert_coefficient(callback._coefficient(0), (0.0, 0.0))
        self.assert_coefficient(callback._coefficient(50), (0.1, 0.5))
        self.assert_coefficient(callback._coefficient(150), (0.2, 1.0))

    def test_control_curriculum_remains_zero(self):
        callback = ExplanationCurriculumCallback(
            mode="control",
            start_coef=0.0,
            end_coef=0.0,
            ramp_timesteps=100,
            checkpoint_dir=Path("unused"),
            checkpoint_freq=50,
            verbose=0,
        )

        self.assert_coefficient(callback._coefficient(0), (0.0, 0.0))
        self.assert_coefficient(callback._coefficient(1_000), (0.0, 0.0))

    def test_success_curriculum_stays_zero_until_gate_then_ramps(self):
        callback = make_callback("success-gated")

        self.assert_coefficient(callback._coefficient(1_000), (0.0, 0.0))
        callback.gate_timestep = 1_000
        self.assert_coefficient(callback._coefficient(1_000), (0.0, 0.0))
        self.assert_coefficient(callback._coefficient(1_050), (0.1, 0.5))
        self.assert_coefficient(callback._coefficient(1_200), (0.2, 1.0))

    def test_schedule_specific_default_ramp_lengths(self):
        linear = Namespace(
            ramp_timesteps=None,
            curriculum="linear",
            total_timesteps=655_360,
        )
        gated = Namespace(
            ramp_timesteps=None,
            curriculum="success-gated",
            total_timesteps=655_360,
        )
        control = Namespace(
            ramp_timesteps=None,
            curriculum="control",
            total_timesteps=655_360,
        )

        self.assertEqual(resolve_ramp_timesteps(linear), 655_360)
        self.assertEqual(resolve_ramp_timesteps(gated), 151_552)
        self.assertEqual(resolve_ramp_timesteps(control), 655_360)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bcos_rl_generic.bcosDQN import ExplanationBcosDQN
from bcos_rl_generic.explanation_supervision import (
    MaskedReplayBuffer,
    compute_q_explanation,
)
from bcos_rl_generic.train_model import (
    ConfigurableDQNPolicy,
    GenericCNNExtractor,
)


class MaskImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(8, 8, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(3)
        self.steps = 0
        self.mask = np.zeros((8, 8), dtype=np.float32)
        self.mask[2:4, 2:4] = 1
        self.mask[0, :] = -1

    def observation(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[2:4, 2:4, 1] = 255
        image[0, :, :] = 128
        return image

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {"attn_mask": self.mask.copy()}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= 4
        reward = float(action == 1)
        return (
            self.observation(),
            reward,
            terminated,
            False,
            {"attn_mask": self.mask.copy()},
        )


class MaskedReplayBufferTests(unittest.TestCase):
    def make_buffer(self) -> MaskedReplayBuffer:
        return MaskedReplayBuffer(
            buffer_size=12,
            observation_space=gym.spaces.Box(
                low=0,
                high=255,
                shape=(1, 2, 2),
                dtype=np.uint8,
            ),
            action_space=gym.spaces.Discrete(2),
            n_envs=2,
            mask_shape=(2, 2),
            device="cpu",
        )

    def test_masks_remain_aligned_with_sampled_observations(self):
        buffer = self.make_buffer()
        for step in range(3):
            codes = np.asarray(
                [step * 2 + 1, step * 2 + 2],
                dtype=np.uint8,
            )
            observations = np.stack(
                [np.full((1, 2, 2), code, dtype=np.uint8) for code in codes]
            )
            next_observations = observations + 1
            masks = [
                np.full((2, 2), int(code), dtype=np.float32)
                for code in codes
            ]
            buffer.add(
                observations,
                next_observations,
                np.asarray([0, 1]),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=bool),
                [{"attn_mask": mask} for mask in masks],
            )

        samples = buffer.sample(100)
        observation_codes = samples.observations[:, 0, 0, 0].float()
        mask_codes = samples.masks[:, 0, 0]
        th.testing.assert_close(observation_codes, mask_codes)

    def test_missing_mask_is_rejected(self):
        buffer = self.make_buffer()
        observations = np.zeros((2, 1, 2, 2), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            buffer.add(
                observations,
                observations,
                np.asarray([0, 1]),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=bool),
                [{}, {}],
            )


class DQNExplanationTests(unittest.TestCase):
    @staticmethod
    def policy_kwargs():
        return {
            "actor_type": "bcos",
            "bcos_b": 2.0,
            "net_arch": [8],
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": 8,
                "use_bcos": True,
                "bcos_b": 2.0,
                "stable_bcos_norm": True,
            },
            "normalize_images": True,
        }

    def test_relative_q_explanation_is_differentiable(self):
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 8, 8),
            dtype=np.uint8,
        )
        policy = ConfigurableDQNPolicy(
            observation_space,
            gym.spaces.Discrete(3),
            lambda _: 1e-3,
            actor_type="bcos",
            bcos_b=2.0,
            net_arch=[8],
            features_extractor_class=GenericCNNExtractor,
            features_extractor_kwargs={
                "features_dim": 8,
                "use_bcos": True,
                "bcos_b": 2.0,
                "stable_bcos_norm": True,
            },
            normalize_images=True,
        )
        observations = th.randint(
            0,
            256,
            (2, 3, 8, 8),
            dtype=th.uint8,
        )
        actions = th.tensor([0, 2])

        explanation = compute_q_explanation(policy, observations, actions)

        self.assertEqual(tuple(explanation.shape), (2, 3, 8, 8))
        self.assertTrue(explanation.requires_grad)
        explanation.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in policy.q_net.parameters()
            )
        )

    def test_standard_checkpoint_can_continue_with_explanation_loss(self):
        env = VecTransposeImage(DummyVecEnv([MaskImageEnv]))
        try:
            baseline = DQN(
                ConfigurableDQNPolicy,
                env,
                learning_rate=1e-4,
                buffer_size=32,
                learning_starts=0,
                batch_size=2,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=10,
                replay_buffer_class=MaskedReplayBuffer,
                replay_buffer_kwargs={"mask_shape": (8, 8)},
                policy_kwargs=self.policy_kwargs(),
                seed=0,
            )
            baseline.learn(total_timesteps=4)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "baseline.zip"
                replay_path = Path(directory) / "baseline_replay.pkl"
                baseline.save(path)
                baseline.save_replay_buffer(replay_path)

                control = DQN.load(path, env=env)
                control.load_replay_buffer(replay_path)
                control.learn(total_timesteps=4, reset_num_timesteps=False)

                models = []
                for mode in ("all", "positive_only", "flip_positive"):
                    model = ExplanationBcosDQN.load(
                        path,
                        env=env,
                        replay_buffer_class=MaskedReplayBuffer,
                        replay_buffer_kwargs={"mask_shape": (8, 8)},
                        explanation_coef=0.1,
                        explanation_max_grad_ratio=0.1,
                        explanation_advantage_mode=mode,
                        explanation_update_frequency=2,
                        explanation_batch_size=1,
                        min_replay_transitions=0,
                    )
                    model.load_replay_buffer(replay_path)
                    model.learning_starts = 0
                    model.learn(
                        total_timesteps=4,
                        reset_num_timesteps=False,
                    )
                    models.append(model)

            self.assertEqual(control.num_timesteps, 8)
            for model in models:
                self.assertEqual(model.num_timesteps, 8)
                self.assertGreater(model._n_updates, baseline._n_updates)
                self.assertIsInstance(
                    model.replay_buffer,
                    MaskedReplayBuffer,
                )
            self.assertIsInstance(control.replay_buffer, MaskedReplayBuffer)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

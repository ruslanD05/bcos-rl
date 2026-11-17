from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from bcos.modules import BcosLinear
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import ReplayBufferSamples

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.explanation_supervision import (
    compute_actor_explanation,
    compute_q_explanation,
    relative_action_outputs,
)
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    ConfigurableDQNPolicy,
    GenericCNNExtractor,
)
from env_wrappers import make_wrapper


class TinyImageEnv(gym.Env):
    def __init__(self, channels: int = 6):
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(channels, 8, 8), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(3)
        self._step = 0

    def _observation(self) -> np.ndarray:
        return self.np_random.integers(
            0, 256, self.observation_space.shape, dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._observation(), {}

    def step(self, action):
        self._step += 1
        return self._observation(), float(action == 0), self._step >= 3, False, {}


class RecordingExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 1):
        super().__init__(observation_space, features_dim)
        self.last_input: th.Tensor | None = None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        self.last_input = observations.detach().clone()
        return observations.flatten(start_dim=1).mean(dim=1, keepdim=True)


def actor_policy(actor_type: str = "bcos") -> ConfigurableActorCriticPolicy:
    common = {
        "features_dim": 8,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    return ConfigurableActorCriticPolicy(
        gym.spaces.Box(0, 255, shape=(6, 8, 8), dtype=np.uint8),
        gym.spaces.Discrete(3),
        lr_schedule=lambda _: 3e-4,
        actor_type=actor_type,
        critic_type="mlp",
        net_arch={"pi": [8], "vf": [8]},
        features_extractor_class=GenericCNNExtractor,
        features_extractor_kwargs={
            **common,
            "use_bcos": actor_type == "bcos",
            "stable_bcos_norm": actor_type == "bcos",
            "bias_free_bcos_norm": actor_type == "bcos",
        },
        critic_features_extractor_class=GenericCNNExtractor,
        critic_features_extractor_kwargs={
            **common,
            "use_bcos": False,
            "stable_bcos_norm": False,
            "bias_free_bcos_norm": False,
        },
        stable_bcos_norm=actor_type == "bcos",
        share_features_extractor=False,
    )


def dqn_policy_kwargs(actor_type: str = "bcos") -> dict:
    return {
        "actor_type": actor_type,
        "bcos_b": 2.0,
        "net_arch": [8],
        "features_extractor_class": GenericCNNExtractor,
        "features_extractor_kwargs": {
            "features_dim": 8,
            "use_bcos": actor_type == "bcos",
            "bcos_b": 2.0,
            "stable_bcos_norm": actor_type == "bcos",
            "bias_free_bcos_norm": actor_type == "bcos",
        },
        "normalize_images": True,
    }


def make_dqn(actor_type: str = "bcos") -> DQN:
    return DQN(
        ConfigurableDQNPolicy,
        TinyImageEnv(),
        learning_rate=1e-4,
        buffer_size=32,
        learning_starts=0,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        gamma=0.9,
        target_update_interval=10,
        policy_kwargs=dqn_policy_kwargs(actor_type),
        seed=0,
        verbose=0,
    )


def actor_relative_logits(policy, observations: th.Tensor) -> th.Tensor:
    with policy.explanation_mode():
        features = policy.extract_features(observations.float())
        if isinstance(features, tuple):
            features = features[0]
        latent = policy.mlp_extractor.forward_actor(features)
        return relative_action_outputs(policy.action_net(latent))


def test_zero_auxiliary_ppo_delegates_to_sb3_train():
    algorithm = object.__new__(BcosPPO)
    algorithm.confidence_coef = 0.0
    algorithm.explanation_coef = 0.0
    algorithm.sequential_explanation_coef = 0.0
    algorithm.action_explanation_coef = 0.0

    with patch.object(PPO, "train") as base_train, patch.object(
        BcosPPO, "_record_rollout_logits"
    ) as record_logits:
        BcosPPO.train(algorithm)

    base_train.assert_called_once_with()
    record_logits.assert_called_once_with()


def test_dqn_td_loss_matches_sb3_target_equation():
    model = make_dqn()
    model.set_logger(configure(folder=None, format_strings=[]))
    observations = th.randint(0, 256, (2, 6, 8, 8), dtype=th.uint8)
    next_observations = th.randint(0, 256, (2, 6, 8, 8), dtype=th.uint8)
    actions = th.tensor([[0], [2]])
    rewards = th.tensor([[1.0], [-0.5]])
    dones = th.tensor([[0.0], [1.0]])
    samples = ReplayBufferSamples(
        observations, actions, next_observations, dones, rewards
    )

    with th.no_grad():
        next_q = model.q_net_target(next_observations).max(dim=1).values[:, None]
        target = rewards + (1.0 - dones) * model.gamma * next_q
        current = model.q_net(observations).gather(1, actions)
        expected = F.smooth_l1_loss(current, target).item()

    model.replay_buffer.sample = lambda *_args, **_kwargs: samples
    model.train(gradient_steps=1, batch_size=2)
    assert abs(model.logger.name_to_value["train/loss"] - expected) < 1e-7


def test_online_and_target_q_networks_are_distinct_bcos_architectures():
    model = make_dqn()
    assert any(isinstance(module, BcosLinear) for module in model.q_net.modules())
    assert any(
        isinstance(module, BcosLinear) for module in model.q_net_target.modules()
    )
    online = list(model.q_net.parameters())
    target = list(model.q_net_target.parameters())
    assert online and len(online) == len(target)
    assert all(left is not right for left, right in zip(online, target))
    for left, right in zip(online, target):
        th.testing.assert_close(left, right)


def test_sb3_normalizes_rgb6_exactly_once():
    observation_space = gym.spaces.Box(
        0, 255, shape=(6, 8, 8), dtype=np.uint8
    )
    policy = ConfigurableDQNPolicy(
        observation_space,
        gym.spaces.Discrete(2),
        lambda _: 1e-3,
        actor_type="mlp",
        net_arch=[],
        features_extractor_class=RecordingExtractor,
        features_extractor_kwargs={"features_dim": 1},
        normalize_images=True,
    )
    observation = th.zeros((1, 6, 8, 8), dtype=th.uint8)
    observation[:, 0] = 255
    observation[:, 1] = 128
    policy.q_net(observation)
    captured = policy.q_net.features_extractor.last_input
    assert captured is not None
    th.testing.assert_close(captured[:, 0], th.ones_like(captured[:, 0]))
    th.testing.assert_close(
        captured[:, 1], th.full_like(captured[:, 1], 128.0 / 255.0)
    )


def test_actor_bcos_explanation_reconstructs_relative_logit():
    policy = actor_policy()
    policy.eval()
    observations = th.randint(1, 255, (2, 6, 8, 8), dtype=th.uint8)
    actions = th.tensor([0, 2])
    explanation = compute_actor_explanation(
        policy, observations, actions, create_graph=False
    )
    target = actor_relative_logits(policy, observations).gather(
        1, actions[:, None]
    ).squeeze(1)
    reconstructed = explanation.flatten(start_dim=1).sum(dim=1)
    th.testing.assert_close(reconstructed, target, rtol=2e-4, atol=2e-4)


def test_q_bcos_explanation_reconstructs_relative_q_value():
    model = make_dqn()
    model.policy.eval()
    observations = th.randint(1, 255, (2, 6, 8, 8), dtype=th.uint8)
    actions = th.tensor([1, 2])
    explanation = compute_q_explanation(
        model.policy, observations, actions, create_graph=False
    )
    with model.policy.explanation_mode():
        relative_q = relative_action_outputs(model.q_net(observations.float()))
    target = relative_q.gather(1, actions[:, None]).squeeze(1)
    reconstructed = explanation.flatten(start_dim=1).sum(dim=1)
    th.testing.assert_close(reconstructed, target, rtol=2e-4, atol=2e-4)


def test_batched_actor_explanations_match_single_sample_results():
    policy = actor_policy()
    policy.eval()
    observations = th.randint(1, 255, (3, 6, 8, 8), dtype=th.uint8)
    actions = th.tensor([0, 1, 2])
    batched = compute_actor_explanation(
        policy, observations, actions, create_graph=False
    )
    singles = th.cat(
        [
            compute_actor_explanation(
                policy,
                observations[index : index + 1],
                actions[index : index + 1],
                create_graph=False,
            )
            for index in range(len(actions))
        ]
    )
    th.testing.assert_close(batched, singles, rtol=1e-5, atol=1e-6)


def test_create_graph_changes_gradient_tracking_not_explanation_values():
    policy = actor_policy()
    policy.eval()
    observations = th.randint(1, 255, (2, 6, 8, 8), dtype=th.uint8)
    actions = th.tensor([0, 1])
    training = compute_actor_explanation(
        policy, observations, actions, create_graph=True
    )
    evaluation = compute_actor_explanation(
        policy, observations, actions, create_graph=False
    )
    th.testing.assert_close(training.detach(), evaluation.detach())
    gradients = th.autograd.grad(
        training.square().mean(),
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        allow_unused=True,
    )
    assert any(gradient is not None for gradient in gradients)


def test_relative_action_target_is_shared_and_offset_invariant():
    outputs = th.tensor([[2.0, -1.0, 4.0], [0.5, 1.5, -2.0]])
    shifted = outputs + th.tensor([[100.0], [-37.0]])
    relative = relative_action_outputs(outputs)
    th.testing.assert_close(relative, relative_action_outputs(shifted))
    th.testing.assert_close(relative.mean(dim=1), th.zeros(2))


def test_fixed_seed_reproduces_minigrid_rgb6_observations_and_layouts():
    factory = make_wrapper(use_rgb_img_obs=True, invert_rgb_channels=True)
    first = factory(gym.make("MiniGrid-DoorKey-8x8-v0", highlight=False))
    second = factory(gym.make("MiniGrid-DoorKey-8x8-v0", highlight=False))
    try:
        first_obs, _ = first.reset(seed=123)
        second_obs, _ = second.reset(seed=123)
        np.testing.assert_array_equal(first_obs, second_obs)
        np.testing.assert_array_equal(
            first.unwrapped.grid.encode(), second.unwrapped.grid.encode()
        )
        for action in (0, 2, 1, 2):
            first_obs, first_reward, first_done, first_truncated, _ = first.step(action)
            second_obs, second_reward, second_done, second_truncated, _ = second.step(action)
            np.testing.assert_array_equal(first_obs, second_obs)
            assert (first_reward, first_done, first_truncated) == (
                second_reward,
                second_done,
                second_truncated,
            )
    finally:
        first.close()
        second.close()


def test_dqn_save_load_preserves_q_outputs():
    model = make_dqn()
    observations = th.randint(0, 256, (2, 6, 8, 8), dtype=th.uint8)
    with th.no_grad():
        expected = model.q_net(observations).clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.zip"
        model.save(path)
        loaded = DQN.load(path, device="cpu")
        with th.no_grad():
            actual = loaded.q_net(observations)
    th.testing.assert_close(actual, expected)


def test_conventional_and_bcos_policies_receive_identical_rgb6_tensors():
    observation_space = gym.spaces.Box(
        0, 255, shape=(6, 8, 8), dtype=np.uint8
    )
    observation = th.randint(0, 256, (2, 6, 8, 8), dtype=th.uint8)
    captured = []
    for actor_type in ("mlp", "bcos"):
        policy = ConfigurableDQNPolicy(
            observation_space,
            gym.spaces.Discrete(2),
            lambda _: 1e-3,
            actor_type=actor_type,
            net_arch=[],
            features_extractor_class=RecordingExtractor,
            features_extractor_kwargs={"features_dim": 1},
            normalize_images=True,
        )
        policy.q_net(observation)
        value = policy.q_net.features_extractor.last_input
        assert value is not None
        captured.append(value)
    th.testing.assert_close(captured[0], captured[1])

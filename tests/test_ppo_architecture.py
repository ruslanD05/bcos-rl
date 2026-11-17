from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from bcos.modules import BcosConv2d
from bcos.modules.norms import GroupNormUncentered2d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from bcos_rl_generic.bcosPPO import (
    BcosPPO,
    explanation_supervision_selection,
    partial_energy_supervision_selection,
    positive_advantage_confidence_loss,
    positive_margin_supervision_selection,
)
from bcos_rl_generic.explanation_supervision import (
    DualMaskedRolloutBuffer,
    MaskedRolloutBuffer,
)
from bcos_rl_generic.train_model import (
    ConfigurableActorCriticPolicy,
    GenericCNNExtractor,
)
from env_wrappers import AddInverseRGBChannelsWrapper
from sweep_minigrid_PPO import (
    baseline_sweep_config,
    direct_run_defaults,
    make_policy_kwargs,
    validate_config,
)


class _TinyImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(6, 64, 64),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(7)
        self._step = 0

    def _observation(self) -> np.ndarray:
        rng = np.random.default_rng(self._step)
        return rng.integers(0, 256, self.observation_space.shape, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._observation(), {}

    def step(self, action):
        self._step += 1
        terminated = self._step >= 2
        return self._observation(), float(action == 0), terminated, False, {}


class _TinyDualMaskEnv(_TinyImageEnv):
    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        sequential_mask = np.zeros((64, 64), dtype=np.float32)
        action_mask = np.zeros((64, 64), dtype=np.float32)
        sequential_mask[0, 0] = 1.0
        action_mask[1, 1] = 1.0
        return observation, reward, terminated, truncated, {
            "sequential_attn_mask": sequential_mask,
            "action_attn_mask": action_mask,
        }


class _TinyLiteralMaskEnv(_TinyImageEnv):
    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[0, 0] = 1.0
        mask[1, 1] = -1.0
        return observation, reward, terminated, truncated, {"attn_mask": mask}


class _TinySequentialMaskEnv(_TinyImageEnv):
    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[0:2, 0:2] = 1.0
        return observation, reward, terminated, truncated, {"attn_mask": mask}


def test_confidence_loss_sharpens_only_positive_advantage_actions():
    logits = th.tensor(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        requires_grad=True,
    )
    actions = th.tensor([0, 1])
    advantages = th.tensor([1.0, -1.0])
    log_prob = F.log_softmax(logits, dim=1).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    loss, selected = positive_advantage_confidence_loss(log_prob, advantages)
    th.testing.assert_close(loss, -log_prob[0])
    assert selected.tolist() == [True, False]

    shifted_log_prob = F.log_softmax(logits + 100.0, dim=1).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)
    shifted_loss, _ = positive_advantage_confidence_loss(
        shifted_log_prob, advantages
    )
    th.testing.assert_close(loss, shifted_loss)

    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0
    assert th.all(logits.grad[1] == 0)


def test_signed_selection_keeps_negative_advantage_transitions():
    masks = th.zeros(3, 4, 4)
    masks[0, 0, 0] = -1
    masks[1, 1, 1] = 1
    advantages = th.tensor([-2.0, 2.0, 2.0])

    signed = explanation_supervision_selection(masks, advantages, signed=True)
    energy = explanation_supervision_selection(masks, advantages, signed=False)

    assert signed.tolist() == [True, True, False]
    assert energy.tolist() == [False, True, False]


def test_positive_margin_selection_uses_normalized_relative_advantage():
    masks = th.ones(4, 2, 2)
    raw_advantages = th.tensor([-4.0, -3.0, -2.0, -1.0])
    normalized = (
        raw_advantages - raw_advantages.mean()
    ) / (raw_advantages.std() + 1e-8)

    selected = positive_margin_supervision_selection(masks, normalized)
    assert selected.tolist() == [False, False, True, True]


def test_partial_energy_selection_keeps_negative_masks_and_good_positive_masks():
    masks = th.zeros(3, 2, 2)
    masks[0, 0, 0] = 1.0
    masks[1, 0, 0] = 1.0
    masks[2, 0, 0] = -1.0
    normalized_advantages = th.tensor([-1.0, 1.0, -1.0])

    selected = partial_energy_supervision_selection(
        masks, normalized_advantages
    )
    assert selected.tolist() == [False, True, True]


def test_inverse_rgb_channels_are_part_of_observation():
    env = gym.Env()
    env.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(2, 2, 3),
        dtype=np.uint8,
    )
    wrapper = AddInverseRGBChannelsWrapper(env)
    rgb = np.array(
        [[[0, 64, 255], [10, 20, 30]], [[1, 2, 3], [4, 5, 6]]],
        dtype=np.uint8,
    )

    observation = wrapper.observation(rgb)
    assert wrapper.observation_space.shape == (2, 2, 6)
    np.testing.assert_array_equal(observation[..., :3], rgb)
    np.testing.assert_array_equal(observation[..., 3:], 255 - rgb)


def test_bcos_actor_and_mlp_critic_use_distinct_cnn_types():
    observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(6, 64, 64),
        dtype=np.uint8,
    )
    action_space = gym.spaces.Discrete(7)
    common = {
        "features_dim": 32,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    policy = ConfigurableActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 3e-4,
        actor_type="bcos",
        critic_type="mlp",
        net_arch={"pi": [32], "vf": [32]},
        features_extractor_class=GenericCNNExtractor,
        features_extractor_kwargs={
            **common,
            "use_bcos": True,
            "stable_bcos_norm": True,
            "bias_free_bcos_norm": True,
        },
        critic_features_extractor_class=GenericCNNExtractor,
        critic_features_extractor_kwargs={
            **common,
            "use_bcos": False,
            "stable_bcos_norm": False,
            "bias_free_bcos_norm": False,
        },
        stable_bcos_norm=True,
        share_features_extractor=False,
    )

    assert any(isinstance(module, BcosConv2d) for module in policy.pi_features_extractor.modules())
    assert not any(isinstance(module, BcosConv2d) for module in policy.vf_features_extractor.modules())
    assert any(isinstance(module, th.nn.Conv2d) for module in policy.vf_features_extractor.modules())

    bcos_norms = [
        module
        for module in policy.pi_features_extractor.modules()
        if isinstance(module, GroupNormUncentered2d)
    ]
    assert bcos_norms
    assert all(module.bias is None for module in bcos_norms)

    optimizer_ids = {
        id(parameter)
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimizer_ids for parameter in policy.parameters())

    observation = th.randint(0, 256, (2, 6, 64, 64), dtype=th.uint8)
    actions, values, log_prob = policy(observation)
    assert actions.shape == (2,)
    assert values.shape == (2, 1)
    assert log_prob.shape == (2,)
    assert th.isfinite(values).all()


def test_no_auxiliary_bcos_ppo_completes_an_sb3_update():
    common = {
        "features_dim": 16,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=_TinyImageEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        bce_coef=0.0,
        explanation_coef=0.0,
        policy_kwargs={
            "actor_type": "bcos",
            "critic_type": "mlp",
            "net_arch": {"pi": [16], "vf": [16]},
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                **common,
                "use_bcos": True,
                "stable_bcos_norm": True,
                "bias_free_bcos_norm": True,
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                **common,
                "use_bcos": False,
                "stable_bcos_norm": False,
                "bias_free_bcos_norm": False,
            },
            "stable_bcos_norm": True,
            "share_features_extractor": False,
        },
        seed=0,
        verbose=0,
    )

    model.learn(total_timesteps=2)
    assert model._n_updates == 1
    assert model.bce_coef == 0.0
    assert model.explanation_coef == 0.0


def test_combined_explanation_ppo_completes_a_differentiable_update():
    common = {
        "features_dim": 8,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=_TinyDualMaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        explanation_coef=0.0,
        sequential_explanation_coef=1e-4,
        action_explanation_coef=1e-4,
        signed_margin=1e-3,
        policy_kwargs={
            "actor_type": "bcos",
            "critic_type": "mlp",
            "net_arch": {"pi": [8], "vf": [8]},
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                **common,
                "use_bcos": True,
                "stable_bcos_norm": True,
                "bias_free_bcos_norm": True,
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                **common,
                "use_bcos": False,
                "stable_bcos_norm": False,
                "bias_free_bcos_norm": False,
            },
            "stable_bcos_norm": True,
            "share_features_extractor": False,
        },
        seed=0,
        verbose=0,
    )

    model.learn(total_timesteps=2)
    assert model._n_updates == 1
    assert isinstance(model.rollout_buffer, DualMaskedRolloutBuffer)
    assert model.action_explanation_coef == 1e-4
    assert (
        model.logger.name_to_value["train/explanation_actor_grad_ratio"]
        <= model.explanation_max_grad_ratio + 1e-6
    )


def test_literal_partial_energy_ppo_completes_an_update():
    common = {
        "features_dim": 8,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=_TinyLiteralMaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        explanation_coef=1e-4,
        explanation_loss_type="partial_energy",
        partial_energy_alignment_margin=0.5,
        policy_kwargs={
            "actor_type": "bcos",
            "critic_type": "mlp",
            "net_arch": {"pi": [8], "vf": [8]},
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                **common,
                "use_bcos": True,
                "stable_bcos_norm": True,
                "bias_free_bcos_norm": True,
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                **common,
                "use_bcos": False,
                "stable_bcos_norm": False,
                "bias_free_bcos_norm": False,
            },
            "stable_bcos_norm": True,
            "share_features_extractor": False,
        },
        seed=0,
        verbose=0,
    )

    model.learn(total_timesteps=2)
    assert model._n_updates == 1
    assert isinstance(model.rollout_buffer, MaskedRolloutBuffer)
    assert model.explanation_loss_type == "partial_energy"


def test_action_category_absolute_energy_ppo_completes_an_update():
    common = {
        "features_dim": 8,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=_TinySequentialMaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        explanation_coef=1e-4,
        explanation_loss_type="absolute_energy",
        policy_kwargs={
            "actor_type": "bcos",
            "critic_type": "mlp",
            "net_arch": {"pi": [8], "vf": [8]},
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                **common,
                "use_bcos": True,
                "stable_bcos_norm": True,
                "bias_free_bcos_norm": True,
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                **common,
                "use_bcos": False,
                "stable_bcos_norm": False,
                "bias_free_bcos_norm": False,
            },
            "stable_bcos_norm": True,
            "share_features_extractor": False,
        },
        seed=0,
        verbose=0,
    )

    model.learn(total_timesteps=2)
    assert model._n_updates == 1
    assert isinstance(model.rollout_buffer, MaskedRolloutBuffer)
    assert model.explanation_loss_type == "absolute_energy"
    assert "absolute_energy/target_energy_ratio" in model.logger.name_to_value


def test_local_action_masked_absolute_energy_ppo_completes_an_update():
    common = {
        "features_dim": 8,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=_TinyLiteralMaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        explanation_coef=1e-4,
        explanation_loss_type="masked_absolute_energy",
        policy_kwargs={
            "actor_type": "bcos",
            "critic_type": "mlp",
            "net_arch": {"pi": [8], "vf": [8]},
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                **common,
                "use_bcos": True,
                "stable_bcos_norm": True,
                "bias_free_bcos_norm": True,
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                **common,
                "use_bcos": False,
                "stable_bcos_norm": False,
                "bias_free_bcos_norm": False,
            },
            "stable_bcos_norm": True,
            "share_features_extractor": False,
        },
        seed=0,
        verbose=0,
    )

    model.learn(total_timesteps=2)
    assert model._n_updates == 1
    assert isinstance(model.rollout_buffer, MaskedRolloutBuffer)
    assert model.explanation_loss_type == "masked_absolute_energy"
    assert "masked_absolute_energy/target_energy_ratio" in model.logger.name_to_value


def test_channelwise_masked_absolute_energy_ppo_completes_an_update():
    common = {
        "features_dim": 8,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=_TinyLiteralMaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        explanation_coef=1e-4,
        explanation_loss_type="channelwise_masked_absolute_energy",
        policy_kwargs={
            "actor_type": "bcos",
            "critic_type": "mlp",
            "net_arch": {"pi": [8], "vf": [8]},
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                **common,
                "use_bcos": True,
                "stable_bcos_norm": True,
                "bias_free_bcos_norm": True,
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                **common,
                "use_bcos": False,
                "stable_bcos_norm": False,
                "bias_free_bcos_norm": False,
            },
            "stable_bcos_norm": True,
            "share_features_extractor": False,
        },
        seed=0,
        verbose=0,
    )

    model.learn(total_timesteps=2)
    assert model._n_updates == 1
    assert isinstance(model.rollout_buffer, MaskedRolloutBuffer)
    assert model.explanation_loss_type == "channelwise_masked_absolute_energy"
    assert (
        "channelwise_masked_absolute_energy/target_energy_ratio"
        in model.logger.name_to_value
    )


def test_sequential_positive_margin_ppo_completes_a_diagnostic_update():
    common = {
        "features_dim": 8,
        "bcos_b": 2.0,
        "bcos_max_out": 1,
        "invert_channels": False,
    }
    model = BcosPPO(
        policy=ConfigurableActorCriticPolicy,
        env=_TinySequentialMaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        explanation_coef=0.1,
        explanation_loss_type="positive_margin",
        sequential_margin=1e-4,
        gradient_diagnostics=True,
        policy_kwargs={
            "actor_type": "bcos",
            "critic_type": "mlp",
            "net_arch": {"pi": [8], "vf": [8]},
            "features_extractor_class": GenericCNNExtractor,
            "features_extractor_kwargs": {
                **common,
                "use_bcos": True,
                "stable_bcos_norm": True,
                "bias_free_bcos_norm": True,
            },
            "critic_features_extractor_class": GenericCNNExtractor,
            "critic_features_extractor_kwargs": {
                **common,
                "use_bcos": False,
                "stable_bcos_norm": False,
                "bias_free_bcos_norm": False,
            },
            "stable_bcos_norm": True,
            "share_features_extractor": False,
        },
        seed=0,
        verbose=0,
    )

    model.learn(total_timesteps=2)
    assert model._n_updates == 1
    assert isinstance(model.rollout_buffer, MaskedRolloutBuffer)
    assert model.explanation_loss_type == "positive_margin"
    assert (
        model.logger.name_to_value["train/explanation_actor_grad_ratio"]
        <= model.explanation_max_grad_ratio + 1e-6
    )


def test_baseline_sweep_cannot_enable_auxiliary_losses():
    args = SimpleNamespace(
        env_id="MiniGrid-DoorKey-8x8-v0",
        total_timesteps=600_000,
        n_envs=8,
        seed=3,
        device="cpu",
        eval_freq=50_000,
        n_eval_episodes=20,
        model_save_freq=150_000,
        checkpoint_threshold=0.95,
        checkpoint_interval=100_000,
        open_door_reward=0.0,
        pick_key_reward=0.0,
        shortcut_prob=0.0,
        use_inventory_obs=False,
    )
    config = direct_run_defaults(args)
    validate_config(config, n_envs=args.n_envs)
    assert config["bce_coef"] == 0.0
    assert config["explanation_coef"] == 0.0
    assert config["use_rgb_img_obs"] is True
    assert config["invert_channels"] is True

    sweep_config = baseline_sweep_config(seed=3)
    assert sweep_config["metric"]["name"] == "eval/baseline_score"
    sweep_parameters = sweep_config["parameters"]
    assert sweep_parameters["seed"]["value"] == 3
    assert sweep_parameters["bce_coef"]["value"] == 0.0
    assert sweep_parameters["explanation_coef"]["value"] == 0.0
    assert "signed_margin" not in sweep_parameters

    policy_kwargs = make_policy_kwargs(config)
    assert policy_kwargs["features_extractor_kwargs"]["use_bcos"] is True
    assert policy_kwargs["critic_features_extractor_kwargs"]["use_bcos"] is False


def _run_all() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()

from __future__ import annotations

import torch as th
import torch.nn as nn

from bcos.modules import BcosLinear
from bcos.modules.common import BcosUtilMixin
from bcos.modules.norms import BatchNormUncentered2d

from bcos_rl_generic.explanation_supervision import (
    attribution_ratio_loss,
    channelwise_masked_absolute_attribution_energy_loss,
    channelwise_polarity_masked_energy_loss,
    compute_actor_explanation,
    masked_absolute_attribution_energy_loss,
)
from bcos_rl_generic.train_model import BcosInputTransform


def test_ratio_uses_channel_summed_spatial_contribution() -> None:
    explanation = th.zeros(1, 2, 2, 2)
    explanation[0, 0, 0, 0] = 2.0
    explanation[0, 1, 0, 0] = -1.0
    mask = th.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0
    assert attribution_ratio_loss(explanation, mask).item() < 1e-5


def test_ratio_gradient_redistributes_background_relevance() -> None:
    explanation = th.zeros(1, 1, 1, 2, requires_grad=True)
    with th.no_grad():
        explanation[0, 0, 0, 0] = 1.0  # target
        explanation[0, 0, 0, 1] = 1.0  # background
    mask = th.tensor([[[1.0, 0.0]]])
    attribution_ratio_loss(explanation, mask).backward()
    assert explanation.grad is not None
    assert explanation.grad[0, 0, 0, 0] < 0  # descent increases target relevance
    assert explanation.grad[0, 0, 0, 1] > 0  # descent decreases background relevance


def test_channelwise_masked_energy_prevents_channel_cancellation() -> None:
    explanation = th.tensor([[[[1.0, 1.0]], [[0.0, -1.0]]]])
    mask = th.tensor([[[1.0, -1.0]]])

    summed_loss, _ = masked_absolute_attribution_energy_loss(explanation, mask)
    channelwise_loss, stats = channelwise_masked_absolute_attribution_energy_loss(
        explanation, mask
    )

    assert summed_loss.item() < 1e-6
    assert th.allclose(channelwise_loss, th.tensor(2.0 / 3.0), atol=1e-6)
    assert abs(stats["negative_energy_ratio"] - 2.0 / 3.0) < 1e-6


def test_channelwise_polarity_mask_rewards_negative_patch_positive_elsewhere() -> None:
    mask = th.tensor([[[1.0, -1.0]]])
    aligned = th.tensor([[[[1.0, -1.0]], [[0.5, -0.5]]]])
    wrong = -aligned

    aligned_loss, aligned_stats = channelwise_polarity_masked_energy_loss(
        aligned, mask
    )
    wrong_loss, wrong_stats = channelwise_polarity_masked_energy_loss(wrong, mask)

    assert aligned_loss.item() < 1e-6
    assert wrong_loss.item() > 0.999
    assert aligned_stats["p_positive_fraction"] > 0.999
    assert aligned_stats["n_negative_fraction"] > 0.999
    assert wrong_stats["wrong_polarity_ratio"] > 0.999


class _ActorExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = BcosInputTransform()
        self.norm = BatchNormUncentered2d(6)
        self.flatten = nn.Flatten()
        self.linear = BcosLinear(24, 8)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(self.flatten(self.norm(self.transform(x))))


class _ActorMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_net = BcosLinear(8, 5)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)


class _Policy(BcosUtilMixin, nn.Module):
    actor_type = "bcos"

    def __init__(self) -> None:
        super().__init__()
        self.pi_features_extractor = _ActorExtractor()
        self.mlp_extractor = _ActorMlp()
        self.action_net = BcosLinear(5, 3)

    def extract_features(self, x: th.Tensor):
        # Match an SB3 policy with separate actor/critic extractors.
        return self.pi_features_extractor(x), th.zeros(x.shape[0], 8, device=x.device)


def test_complement_explanation_is_canonical_and_does_not_update_batchnorm() -> None:
    th.manual_seed(0)
    policy = _Policy().train()
    obs = th.rand(2, 3, 2, 2)
    actions = th.tensor([0, 2])
    tracked_before = policy.pi_features_extractor.norm.num_batches_tracked.clone()

    explanation = compute_actor_explanation(policy, obs, actions)

    assert explanation.shape == (2, 6, 2, 2)
    assert policy.pi_features_extractor.norm.training
    assert th.equal(
        policy.pi_features_extractor.norm.num_batches_tracked, tracked_before
    )
    assert explanation.requires_grad
    explanation.square().mean().backward()
    assert policy.action_net.linear.weight.grad is not None


def test_raw_action_explanation_does_not_subtract_other_actions() -> None:
    th.manual_seed(1)
    policy = _Policy().eval()
    observation = th.rand(1, 3, 2, 2)
    observations = observation.repeat(3, 1, 1, 1)
    actions = th.arange(3)

    raw = compute_actor_explanation(
        policy,
        observations,
        actions,
        create_graph=False,
        subtract_action_mean=False,
    )
    relative = compute_actor_explanation(
        policy,
        observations,
        actions,
        create_graph=False,
        subtract_action_mean=True,
    )

    assert th.allclose(relative, raw - raw.mean(dim=0, keepdim=True), atol=1e-5)
    assert not th.allclose(raw, relative)


def test_non_bcos_actor_is_rejected() -> None:
    policy = _Policy()
    policy.actor_type = "mlp"
    try:
        compute_actor_explanation(policy, th.rand(1, 3, 2, 2), th.tensor([0]))
    except ValueError as error:
        assert "B-cos" in str(error)
    else:
        raise AssertionError("Expected non-B-cos policy to be rejected")

"""Differentiable explanation supervision utilities for BcosPPO.

Three components
────────────────
MaskedRolloutBuffer / DualMaskedRolloutBuffer
    Subclass of SB3's RolloutBuffer that stores a float32 (H, W) attention
    mask, or separate sequential/action masks, alongside every transition and
    shuffles them in sync with all other fields.

compute_actor_explanation(policy, obs, actions) → (B, C, H, W)
    Signed gradient × input computed inside policy.explanation_mode().
    Positive values = pixel pushed taken-action logit up; negative = suppressed.
    Uses create_graph=True so loss.backward() updates policy parameters.

attribution_ratio_loss(explanation, mask) → scalar
    Positive-attribution localization on the channel-summed spatial map. Both
    numerator and denominator remain differentiable, so relevance is moved from
    background to target instead of merely inflated.

signed_attribution_margin_loss(explanation, signed_mask, margin, ...) → (loss, stats)
    Action-conditioned *signed* supervision. signed_mask ∈ {-1, 0, +1} says each
    cell should support (+1), inhibit (-1) or be left neutral (0) for the
    explained action. Per supervised cell the hinge is max(0, m − S·A), zero once
    S·A ≥ m; unsupervised (S=0) cells are excluded from both the sum and its
    normaliser, so they receive no gradient. Unlike the positive-only energy
    loss above, this can *require* genuinely negative attribution (e.g. a wall in
    front should suppress `forward`).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterator, NamedTuple, Optional

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class MaskedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    masks: th.Tensor
    discounts: Optional[th.Tensor] = None


class MaskedReplayBuffer(ReplayBuffer):
    """Replay buffer that keeps one signed spatial mask per observation."""

    def __init__(
        self,
        *args,
        mask_shape: tuple[int, ...] = (64, 64),
        **kwargs,
    ) -> None:
        self.mask_shape = mask_shape
        super().__init__(*args, **kwargs)
        # MiniGrid masks only contain -1, 0, and +1. int8 avoids adding a
        # second image-sized float32 tensor to an already large replay buffer.
        self.attn_masks = np.zeros(
            (self.buffer_size, self.n_envs, *mask_shape),
            dtype=np.int8,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict],
    ) -> None:
        masks = [info.get("attn_mask") for info in infos]
        if any(mask is None for mask in masks):
            raise RuntimeError(
                "MaskedReplayBuffer requires info['attn_mask'] for every transition"
            )
        stacked = np.stack(masks)
        if stacked.shape != (self.n_envs, *self.mask_shape):
            raise ValueError(
                "Unexpected replay mask shape "
                f"{stacked.shape}; expected {(self.n_envs, *self.mask_shape)}"
            )
        self.attn_masks[self.pos] = stacked.astype(np.int8, copy=False)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskedReplayBufferSamples:
        env_indices = np.random.randint(
            0,
            high=self.n_envs,
            size=(len(batch_inds),),
        )
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[
                    (batch_inds + 1) % self.buffer_size,
                    env_indices,
                    :,
                ],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :],
                env,
            )

        observations = self._normalize_obs(
            self.observations[batch_inds, env_indices, :],
            env,
        )
        dones = (
            self.dones[batch_inds, env_indices]
            * (1 - self.timeouts[batch_inds, env_indices])
        ).reshape(-1, 1)
        rewards = self._normalize_reward(
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
            env,
        )
        return MaskedReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(dones),
            rewards=self.to_torch(rewards),
            masks=self.to_torch(
                self.attn_masks[batch_inds, env_indices, :].astype(np.float32)
            ),
            discounts=None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Masked rollout buffer
# ──────────────────────────────────────────────────────────────────────────────

class MaskedRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer extended with a float32 (H, W) attention mask per transition.

    The mask is stored in ``self.attn_masks`` and flattened / shuffled in the
    same order as observations, actions, etc., so mini-batches returned by
    ``get()`` are always aligned.  The current mini-batch's masks are available
    on ``self._current_masks`` immediately after each ``yield`` in ``get()``.

    Usage in collect_rollouts::

        buffer.add(obs, actions, rewards, starts, values, log_probs,
                   attn_mask=np.stack([info["attn_mask"] for info in infos]))

    Usage in train::

        for rollout_data in self.rollout_buffer.get(batch_size):
            masks = self.rollout_buffer._current_masks  # (B, H, W)
    """

    def __init__(self, *args, mask_shape: tuple[int, ...] = (64, 64), **kwargs):
        self._mask_shape = mask_shape
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.attn_masks = np.zeros(
            (self.buffer_size, self.n_envs, *self._mask_shape), dtype=np.float32
        )
        self._flat_masks: Optional[np.ndarray] = None
        self._current_masks: Optional[th.Tensor] = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        attn_mask: Optional[np.ndarray] = None,
    ) -> None:
        if attn_mask is not None:
            self.attn_masks[self.pos] = attn_mask  # (n_envs, H, W)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        # Flatten masks with the same swap-axes order as observations etc.
        self._flat_masks = self.swap_and_flatten(self.attn_masks)
        yield from super().get(batch_size)

    def _get_samples(
        self, batch_inds: np.ndarray, env=None
    ) -> RolloutBufferSamples:
        samples = super()._get_samples(batch_inds, env)
        # Stash the mini-batch masks so BcosPPO.train() can read them.
        if self._flat_masks is not None:
            self._current_masks = self.to_torch(self._flat_masks[batch_inds])
        return samples


class DualMaskedRolloutBuffer(RolloutBuffer):
    """Rollout buffer carrying separate sequential and action masks."""

    def __init__(self, *args, mask_shape: tuple[int, ...] = (64, 64), **kwargs):
        self._mask_shape = mask_shape
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs, *self._mask_shape)
        self.sequential_attn_masks = np.zeros(shape, dtype=np.float32)
        self.action_attn_masks = np.zeros(shape, dtype=np.float32)
        self._flat_sequential_masks: Optional[np.ndarray] = None
        self._flat_action_masks: Optional[np.ndarray] = None
        self._current_sequential_masks: Optional[th.Tensor] = None
        self._current_action_masks: Optional[th.Tensor] = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        sequential_attn_mask: Optional[np.ndarray] = None,
        action_attn_mask: Optional[np.ndarray] = None,
    ) -> None:
        if sequential_attn_mask is not None:
            self.sequential_attn_masks[self.pos] = sequential_attn_mask
        if action_attn_mask is not None:
            self.action_attn_masks[self.pos] = action_attn_mask
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        self._flat_sequential_masks = self.swap_and_flatten(
            self.sequential_attn_masks
        )
        self._flat_action_masks = self.swap_and_flatten(self.action_attn_masks)
        yield from super().get(batch_size)

    def _get_samples(
        self, batch_inds: np.ndarray, env=None
    ) -> RolloutBufferSamples:
        samples = super()._get_samples(batch_inds, env)
        if self._flat_sequential_masks is not None:
            self._current_sequential_masks = self.to_torch(
                self._flat_sequential_masks[batch_inds]
            )
        if self._flat_action_masks is not None:
            self._current_action_masks = self.to_torch(
                self._flat_action_masks[batch_inds]
            )
        return samples


# ──────────────────────────────────────────────────────────────────────────────
# Differentiable B-cos explanation
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _frozen_batch_norm(module: nn.Module) -> Iterator[None]:
    """Use inference statistics without disabling affine-parameter gradients."""
    batch_norms = [
        child
        for child in module.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]
    states = [child.training for child in batch_norms]
    try:
        for child in batch_norms:
            child.training = False
        yield
    finally:
        for child, was_training in zip(batch_norms, states):
            child.training = was_training


def relative_action_outputs(outputs: th.Tensor) -> th.Tensor:
    """Remove action-independent offsets from batched logits or Q-values."""
    if outputs.ndim != 2:
        raise ValueError(
            "Action outputs must have shape (batch, actions), got "
            f"{tuple(outputs.shape)}"
        )
    return outputs - outputs.mean(dim=1, keepdim=True)


def compute_actor_explanation(
    policy,
    obs: th.Tensor,
    actions: th.Tensor,
    *,
    create_graph: bool = True,
    subtract_action_mean: bool = True,
) -> th.Tensor:
    """
    B-cos explanation (gradient × input) for the actor head, differentiable
    w.r.t. policy parameters.

    Concretely: forward the observation through the actor inside
    ``policy.explanation_mode()``, then compute
    ``∂(taken_logit − mean_logit) / ∂input`` with ``create_graph=True``.
    For internal complement encoding, input is ``[x, 1-x]``. The element-wise
    product ``grad × input`` is the signed B-cos contribution.

    Args:
        policy:  ConfigurableActorCriticPolicy (must have .explanation_mode()).
        obs:     RGB (B, C, H, W) or symbolic one-hot (B, H, W, C).
        actions: (B,) long tensor of taken actions.

    Returns:
        explanation: (B, C, H, W), or (B, 2C, H, W) with internal complement
            encoding; signed and differentiable w.r.t. policy parameters.
    """
    if getattr(policy, "actor_type", None) != "bcos":
        raise ValueError("Explanation supervision requires a B-cos policy actor")

    # Keep observations in their original image scale. policy.extract_features()
    # applies SB3 preprocessing, including the single required division by 255.
    obs_f = obs.float().detach().requires_grad_(True)

    # Most current MiniGrid runs expose [RGB, inverse-RGB] directly as the
    # observation. Also support the extractor's internal [x, 1-x] transform:
    # canonical B-cos attribution must then be z*dlogit/dz, not x*dlogit/dx.
    transformed_inputs: list[th.Tensor] = []
    transform_hooks = []
    actor_extractor = getattr(policy, "pi_features_extractor", None)
    if actor_extractor is not None:
        for module in actor_extractor.modules():
            if (
                getattr(module, "is_bcos_input_transform", False)
                or module.__class__.__name__ == "BcosInputTransform"
            ):
                transform_hooks.append(
                    module.register_forward_hook(
                        lambda _module, _inputs, output: transformed_inputs.append(output)
                    )
                )

    try:
        with _frozen_batch_norm(policy), policy.explanation_mode():
            features = policy.extract_features(obs_f)
            if isinstance(features, tuple):
                features = features[0]
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            if isinstance(logits, tuple):
                logits = logits[0]
    finally:
        for hook in transform_hooks:
            hook.remove()

    # Relative logit keeps the gradient well-conditioned and matches the
    # sign convention used in visualize_minigrid.get_bcos_explanation_Actor.
    explanation_outputs = (
        relative_action_outputs(logits) if subtract_action_mean else logits
    )
    target = explanation_outputs.gather(
        1, actions.long().unsqueeze(1)
    ).squeeze(1)

    # create_graph=True is essential: it records the dependence of `grad` on
    # policy parameters so that loss.backward() can update the weights.
    contribution_input = transformed_inputs[-1] if transformed_inputs else obs_f
    grad = th.autograd.grad(
        target.sum(), contribution_input, create_graph=create_graph
    )[0]

    # grad × input — signed NCHW tensor.  Positive values mean the pixel pushed
    # the taken action's relative logit up; negative means it suppressed it.
    # The loss handles sign internally (positive-inside vs abs-total).
    explanation = grad * contribution_input
    channels_last_input = isinstance(
        policy.pi_features_extractor, FlattenExtractor
    ) or getattr(policy.pi_features_extractor, "channels_last_input", False)
    if explanation.ndim == 4 and channels_last_input and not transformed_inputs:
        # Symbolic one-hot observations use channels-last (B, H, W, C), while
        # all explanation losses consume channels-first tensors.
        explanation = explanation.permute(0, 3, 1, 2)
    return explanation


def compute_q_explanation(
    policy,
    obs: th.Tensor,
    actions: th.Tensor,
    *,
    create_graph: bool = True,
) -> th.Tensor:
    """Differentiable B-cos explanation of a taken action's relative Q-value."""
    if getattr(policy, "actor_type", None) != "bcos":
        raise ValueError("Explanation supervision requires a B-cos Q-network")

    obs_f = obs.float().detach().requires_grad_(True)
    transformed_inputs: list[th.Tensor] = []
    transform_hooks = []
    extractor = policy.q_net.features_extractor
    for module in extractor.modules():
        if (
            getattr(module, "is_bcos_input_transform", False)
            or module.__class__.__name__ == "BcosInputTransform"
        ):
            transform_hooks.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output: transformed_inputs.append(
                        output
                    )
                )
            )

    try:
        with _frozen_batch_norm(policy), policy.explanation_mode():
            q_values = policy.q_net(obs_f)
    finally:
        for hook in transform_hooks:
            hook.remove()

    relative_q_values = relative_action_outputs(q_values)
    target = relative_q_values.gather(
        1, actions.long().unsqueeze(1)
    ).squeeze(1)
    contribution_input = transformed_inputs[-1] if transformed_inputs else obs_f
    grad = th.autograd.grad(
        target.sum(),
        contribution_input,
        create_graph=create_graph,
    )[0]
    explanation = grad * contribution_input
    channels_last_input = isinstance(extractor, FlattenExtractor) or getattr(
        extractor,
        "channels_last_input",
        False,
    )
    if explanation.ndim == 4 and channels_last_input and not transformed_inputs:
        explanation = explanation.permute(0, 3, 1, 2)
    return explanation


# ──────────────────────────────────────────────────────────────────────────────
# Attribution ratio loss
# ──────────────────────────────────────────────────────────────────────────────

def _spatial_contribution(explanation: th.Tensor) -> th.Tensor:
    """Canonical signed B-cos contribution map."""
    if explanation.ndim != 4:
        raise ValueError(
            f"Expected explanation with shape (B,C,H,W), got {tuple(explanation.shape)}"
        )
    return explanation.sum(dim=1)


def attribution_ratio_loss(explanation: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Localize positive spatial contribution with a true ratio gradient."""
    attr = _spatial_contribution(explanation)
    target = mask.to(dtype=attr.dtype).clamp(0.0, 1.0)
    valid = target.sum(dim=(1, 2)) > 0
    if not valid.any():
        return attr.sum() * 0.0

    attr = attr[valid]
    target = target[valid]
    smooth_eps = 1e-6
    positive = 0.5 * (attr + th.sqrt(attr.square() + smooth_eps**2))
    inside = (positive * target).sum(dim=(1, 2))
    total = positive.sum(dim=(1, 2))
    return (1.0 - inside / (total + 1e-8)).mean()


def attribution_ratio_loss_original(explanation: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Backward-compatible alias for the corrected spatial ratio."""
    return attribution_ratio_loss(explanation, mask)


def positive_attribution_margin_loss(
    explanation: th.Tensor,
    positive_mask: th.Tensor,
    margin: float = 0.001,
) -> tuple[th.Tensor, dict[str, float]]:
    """Require positive attribution in a target region and ignore its exterior.

    Attribution channels are summed before spatial aggregation, matching the
    contribution map used by visualization and signed action supervision. A
    single hinge is applied to each sample's mean target-region attribution;
    this encourages attention to the region without requiring every rendered
    pixel in a MiniGrid tile to carry the margin independently.
    """
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")

    attr = _spatial_contribution(explanation)
    mask = positive_mask > 0
    counts = mask.sum(dim=(1, 2))
    valid = counts > 0

    if not valid.any():
        loss = attr.sum() * 0.0
        return loss, {
            "mean_target_attribution": 0.0,
            "margin_satisfied_rate": 0.0,
            "n_supervised_samples": 0.0,
        }

    mask_f = mask.to(attr.dtype)
    # Normalize only over supervised pixels. This removes arbitrary actor-logit
    # scale while keeping the exterior truly neutral (zero loss gradient).
    region_square = (attr.square() * mask_f).sum(dim=(1, 2))
    region_rms = th.sqrt(region_square / counts.clamp_min(1).to(attr.dtype) + 1e-4)
    normalized_attr = attr / region_rms[:, None, None]
    region_sums = (normalized_attr * mask_f).sum(dim=(1, 2))
    region_means = region_sums[valid] / counts[valid].to(attr.dtype)
    loss = th.clamp(margin - region_means, min=0.0).mean()
    with th.no_grad():
        stats = {
            "mean_target_attribution": region_means.mean().item(),
            "margin_satisfied_rate": (region_means >= margin).float().mean().item(),
            "n_supervised_samples": float(valid.sum().item()),
        }
    return loss, stats


def polarity_separated_attribution_margin_loss(
    explanation: th.Tensor,
    target_mask: th.Tensor,
    *,
    polarity: str,
    margin: float,
) -> tuple[th.Tensor, dict[str, float]]:
    """Apply a margin to one attribution polarity without channel cancellation.

    Positive and negative attribution are separated per channel before channels
    are summed. The desired polarity is normalized against total attribution
    magnitude in the supervised region. The forward values use an exact hard
    polarity split. A straight-through derivative on the inactive side lets a
    completely wrong-polarity region receive a gradient and cross zero.
    """
    if polarity not in {"positive", "negative"}:
        raise ValueError(f"polarity must be 'positive' or 'negative', got {polarity!r}")
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")

    def positive_energy(values: th.Tensor) -> th.Tensor:
        hard = values.clamp(min=0.0)
        straight_through = (values - values.detach()) * (values < 0).to(values.dtype)
        return (hard + straight_through).sum(dim=1)

    positive = positive_energy(explanation)
    negative = positive_energy(-explanation)
    desired = positive if polarity == "positive" else negative
    total_magnitude = explanation.abs().sum(dim=1)

    mask = target_mask > 0
    counts = mask.sum(dim=(1, 2))
    valid = counts > 0
    if not valid.any():
        loss = explanation.sum() * 0.0
        return loss, {
            "mean_target_energy": 0.0,
            "margin_satisfied_rate": 0.0,
            "n_supervised_samples": 0.0,
            "n_supervised_cells": 0.0,
        }

    mask_f = mask.to(explanation.dtype)
    region_square = (total_magnitude.square() * mask_f).sum(dim=(1, 2))
    region_rms = th.sqrt(
        region_square / counts.clamp_min(1).to(explanation.dtype) + 1e-4
    )
    normalized_desired = desired / region_rms[:, None, None]
    region_means = (
        (normalized_desired * mask_f).sum(dim=(1, 2))[valid]
        / counts[valid].to(explanation.dtype)
    )
    loss = th.clamp(margin - region_means, min=0.0).mean()
    with th.no_grad():
        stats = {
            "mean_target_energy": region_means.mean().item(),
            "margin_satisfied_rate": (region_means >= margin).float().mean().item(),
            "n_supervised_samples": float(valid.sum().item()),
            "n_supervised_cells": float(mask.sum().item()),
        }
    return loss, stats


def partial_masked_energy_loss(
    explanation: th.Tensor,
    signed_mask: th.Tensor,
    alignment_margin: float = 0.5,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Fable-style energy loss over literal positive/negative/neutral regions.

    ``signed_mask`` uses +1 for P, -1 for N and 0 for D. When both P and N
    exist, the loss is ``1 - E_P / (E_P + E_N + eps)`` using positive spatial
    attribution energy. P-only samples use a bounded signed-alignment fallback;
    N-only samples minimize the bounded fraction of positive attribution in N.
    D never appears in any numerator or denominator.
    """
    if not 0.0 <= alignment_margin <= 1.0:
        raise ValueError(
            f"alignment_margin must be in [0, 1], got {alignment_margin}"
        )

    attr = _spatial_contribution(explanation)
    positive_attr = attr.clamp(min=0)
    absolute_attr = attr.abs()
    positive_region = signed_mask > 0
    negative_region = signed_mask < 0
    has_positive = positive_region.flatten(start_dim=1).any(dim=1)
    has_negative = negative_region.flatten(start_dim=1).any(dim=1)
    valid = has_positive | has_negative

    p = positive_region.to(attr.dtype)
    n = negative_region.to(attr.dtype)
    p_energy = (positive_attr * p).sum(dim=(1, 2))
    n_energy = (positive_attr * n).sum(dim=(1, 2))
    p_signed = (attr * p).sum(dim=(1, 2))
    p_absolute = (absolute_attr * p).sum(dim=(1, 2))
    n_absolute = (absolute_attr * n).sum(dim=(1, 2))

    p_alignment = p_signed / (p_absolute + eps)
    n_positive_fraction = n_energy / (n_absolute + eps)
    ratio_loss = 1.0 - p_energy / (p_energy + n_energy + eps)
    positive_only_loss = th.clamp(alignment_margin - p_alignment, min=0.0)

    both = has_positive & has_negative
    positive_only = has_positive & ~has_negative
    negative_only = has_negative & ~has_positive
    per_sample = th.zeros_like(p_energy)
    per_sample = th.where(both, ratio_loss, per_sample)
    per_sample = th.where(positive_only, positive_only_loss, per_sample)
    per_sample = th.where(negative_only, n_positive_fraction, per_sample)

    if not valid.any():
        loss = attr.sum() * 0.0
    else:
        loss = per_sample[valid].mean()

    with th.no_grad():
        stats = {
            "masked_energy_ratio": (
                (1.0 - ratio_loss[both]).mean().item() if both.any() else 0.0
            ),
            "positive_alignment": (
                p_alignment[has_positive].mean().item()
                if has_positive.any() else 0.0
            ),
            "negative_positive_fraction": (
                n_positive_fraction[has_negative].mean().item()
                if has_negative.any() else 0.0
            ),
            "both_regions_rate": both.float().mean().item(),
            "positive_only_rate": positive_only.float().mean().item(),
            "negative_only_rate": negative_only.float().mean().item(),
        }
    return loss, stats


def absolute_attribution_energy_loss(
    explanation: th.Tensor,
    target_mask: th.Tensor,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Localize unsigned spatial attribution inside an action target.

    The per-sample objective is ``1 - |A|_P / (|A|_all + eps)``. Unlike the
    signed and partial-energy objectives, this only states where attribution
    magnitude should occur. Attribution polarity is unconstrained, and every
    pixel outside P competes in the denominator.
    """
    attr = _spatial_contribution(explanation)
    magnitude = attr.abs()
    target = target_mask > 0
    valid = target.flatten(start_dim=1).any(dim=1)
    target_f = target.to(magnitude.dtype)
    inside = (magnitude * target_f).sum(dim=(1, 2))
    total = magnitude.sum(dim=(1, 2))
    ratio = inside / (total + eps)
    per_sample = 1.0 - ratio
    loss = per_sample[valid].mean() if valid.any() else attr.sum() * 0.0

    with th.no_grad():
        stats = {
            "target_energy_ratio": (
                ratio[valid].mean().item() if valid.any() else 0.0
            ),
            "outside_energy_ratio": (
                (1.0 - ratio[valid]).mean().item() if valid.any() else 0.0
            ),
            "mask_coverage": target_f.mean().item(),
            "n_supervised_samples": float(valid.sum().item()),
        }
    return loss, stats


def masked_absolute_attribution_energy_loss(
    explanation: th.Tensor,
    signed_mask: th.Tensor,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Move unsigned attribution magnitude from N to P while excluding D.

    The objective is ``1 - |A|_P / (|A|_P + |A|_N + eps)``. Pixels marked
    zero are genuinely neutral and receive no direct gradient from this loss.
    """
    attr = _spatial_contribution(explanation)
    magnitude = attr.abs()
    positive = signed_mask > 0
    negative = signed_mask < 0
    valid = positive.flatten(start_dim=1).any(dim=1) & negative.flatten(
        start_dim=1
    ).any(dim=1)
    p = positive.to(magnitude.dtype)
    n = negative.to(magnitude.dtype)
    p_energy = (magnitude * p).sum(dim=(1, 2))
    n_energy = (magnitude * n).sum(dim=(1, 2))
    ratio = p_energy / (p_energy + n_energy + eps)
    loss = (1.0 - ratio[valid]).mean() if valid.any() else attr.sum() * 0.0

    with th.no_grad():
        return loss, {
            "target_energy_ratio": ratio[valid].mean().item() if valid.any() else 0.0,
            "negative_energy_ratio": (
                (1.0 - ratio[valid]).mean().item() if valid.any() else 0.0
            ),
            "positive_mask_coverage": p.mean().item(),
            "negative_mask_coverage": n.mean().item(),
            "selected_transition_rate": valid.float().mean().item(),
        }


def channelwise_masked_absolute_attribution_energy_loss(
    explanation: th.Tensor,
    signed_mask: th.Tensor,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Move channelwise absolute attribution energy from N to P.

    Unlike :func:`masked_absolute_attribution_energy_loss`, this computes
    ``sum_c |E_c|`` at each spatial location. Opposite RGB6 channel
    contributions therefore cannot cancel before the P/N objective is applied.
    """
    if explanation.ndim != 4:
        raise ValueError(
            f"Expected explanation with shape (B,C,H,W), got {tuple(explanation.shape)}"
        )
    magnitude = explanation.abs().sum(dim=1)
    positive = signed_mask > 0
    negative = signed_mask < 0
    valid = positive.flatten(start_dim=1).any(dim=1) & negative.flatten(
        start_dim=1
    ).any(dim=1)
    p = positive.to(magnitude.dtype)
    n = negative.to(magnitude.dtype)
    p_energy = (magnitude * p).sum(dim=(1, 2))
    n_energy = (magnitude * n).sum(dim=(1, 2))
    ratio = p_energy / (p_energy + n_energy + eps)
    loss = (1.0 - ratio[valid]).mean() if valid.any() else explanation.sum() * 0.0

    with th.no_grad():
        return loss, {
            "target_energy_ratio": ratio[valid].mean().item() if valid.any() else 0.0,
            "negative_energy_ratio": (
                (1.0 - ratio[valid]).mean().item() if valid.any() else 0.0
            ),
            "positive_mask_coverage": p.mean().item(),
            "negative_mask_coverage": n.mean().item(),
            "selected_transition_rate": valid.float().mean().item(),
        }


def channelwise_polarity_masked_energy_loss(
    explanation: th.Tensor,
    signed_mask: th.Tensor,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Align attribution polarity with a signed P/N mask channelwise.

    P receives positive attribution energy and N receives negative attribution
    energy. Opposite RGB6 channels are separated before summing, so they cannot
    cancel before the P/N objective is evaluated.
    """
    if explanation.ndim != 4:
        raise ValueError(
            f"Expected explanation with shape (B,C,H,W), got {tuple(explanation.shape)}"
        )
    positive_attr = explanation.clamp(min=0.0).sum(dim=1)
    negative_attr = (-explanation).clamp(min=0.0).sum(dim=1)
    positive = signed_mask > 0
    negative = signed_mask < 0
    valid = positive.flatten(start_dim=1).any(dim=1) & negative.flatten(
        start_dim=1
    ).any(dim=1)
    p = positive.to(explanation.dtype)
    n = negative.to(explanation.dtype)

    p_positive = (positive_attr * p).sum(dim=(1, 2))
    p_negative = (negative_attr * p).sum(dim=(1, 2))
    n_positive = (positive_attr * n).sum(dim=(1, 2))
    n_negative = (negative_attr * n).sum(dim=(1, 2))
    desired = p_positive + n_negative
    wrong = p_negative + n_positive
    ratio = desired / (desired + wrong + eps)
    loss = (1.0 - ratio[valid]).mean() if valid.any() else explanation.sum() * 0.0

    with th.no_grad():
        p_total = p_positive + p_negative
        n_total = n_positive + n_negative
        return loss, {
            "target_energy_ratio": ratio[valid].mean().item() if valid.any() else 0.0,
            "wrong_polarity_ratio": (
                (1.0 - ratio[valid]).mean().item() if valid.any() else 0.0
            ),
            "p_positive_fraction": (
                (p_positive[valid] / (p_total[valid] + eps)).mean().item()
                if valid.any()
                else 0.0
            ),
            "n_negative_fraction": (
                (n_negative[valid] / (n_total[valid] + eps)).mean().item()
                if valid.any()
                else 0.0
            ),
            "positive_mask_coverage": p.mean().item(),
            "negative_mask_coverage": n.mean().item(),
            "selected_transition_rate": valid.float().mean().item(),
        }


def patch_absolute_energy_loss(
    explanation: th.Tensor,
    signed_mask: th.Tensor,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Directly penalize absolute attribution energy in the N region across all channels.

    No ratio normalization is applied: the loss is the mean absolute attribution
    on pixels where signed_mask < 0, evaluated independently across channels and
    averaged.
    """
    if explanation.ndim != 4:
        raise ValueError(
            f"Expected explanation with shape (B,C,H,W), got {tuple(explanation.shape)}"
        )
    negative = signed_mask < 0
    valid = negative.flatten(start_dim=1).any(dim=1)
    n = negative.unsqueeze(1).to(explanation.dtype)  # (B, 1, H, W)

    patch_channel_energy = (explanation.abs() * n).sum(dim=(2, 3))  # (B, C)
    total_channel_energy = explanation.abs().sum(dim=(2, 3))         # (B, C)

    # Per-channel patch energy fraction: in [0, 1] per channel, isolated from other channels
    channel_patch_ratio = patch_channel_energy / (total_channel_energy + eps)  # (B, C)
    sample_loss = channel_patch_ratio.mean(dim=1)                              # (B,)
    loss = sample_loss[valid].mean() if valid.any() else explanation.sum() * 0.0

    with th.no_grad():
        total_energy = explanation.abs().sum(dim=(1, 2, 3))
        patch_energy = (explanation.abs() * n).sum(dim=(1, 2, 3))
        ratio = patch_energy / (total_energy + eps)
        return loss, {
            "target_energy_ratio": (
                (1.0 - ratio[valid]).mean().item() if valid.any() else 0.0
            ),
            "negative_energy_ratio": (
                ratio[valid].mean().item() if valid.any() else 0.0
            ),
            "patch_absolute_loss": loss.item(),
            "mean_patch_abs_energy": (
                patch_energy[valid].mean().item() if valid.any() else 0.0
            ),
            "mean_total_abs_energy": (
                total_energy[valid].mean().item() if valid.any() else 0.0
            ),
            "patch_energy_fraction": (
                ratio[valid].mean().item() if valid.any() else 0.0
            ),
            "negative_mask_coverage": negative.float().mean().item(),
            "selected_transition_rate": valid.float().mean().item(),
        }


def attribution_map_consistency_loss(
    explanation: th.Tensor,
    counterfactual_explanation: th.Tensor,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Match normalized spatial attribution magnitude across a paired nuisance edit.

    Each map is channel-summed, converted to magnitude, and L1-normalized per
    sample.  The resulting half-L1 distance lies in [0, 1]. Normalization keeps
    the objective focused on explanation structure rather than arbitrary actor
    logit scale.
    """
    attr = _spatial_contribution(explanation).abs()
    counterfactual_attr = _spatial_contribution(
        counterfactual_explanation
    ).abs()
    if attr.shape != counterfactual_attr.shape:
        raise ValueError(
            "Paired explanations must have identical shapes, got "
            f"{tuple(attr.shape)} and {tuple(counterfactual_attr.shape)}"
        )
    attr_norm = attr / (attr.sum(dim=(1, 2), keepdim=True) + eps)
    counterfactual_norm = counterfactual_attr / (
        counterfactual_attr.sum(dim=(1, 2), keepdim=True) + eps
    )
    per_sample = 0.5 * (
        attr_norm - counterfactual_norm
    ).abs().sum(dim=(1, 2))
    loss = per_sample.mean()
    with th.no_grad():
        return loss, {
            "mean_map_l1": per_sample.mean().item(),
            "max_map_l1": per_sample.max().item(),
            "original_total_absolute_energy": attr.sum(dim=(1, 2)).mean().item(),
            "counterfactual_total_absolute_energy": (
                counterfactual_attr.sum(dim=(1, 2)).mean().item()
            ),
        }


def advantage_conditioned_partial_energy_loss(
    explanation: th.Tensor,
    signed_mask: th.Tensor,
    polarity: th.Tensor,
    alignment_margin: float = 0.5,
    sign_weight: float = 1.0,
    eps: float = 1e-8,
) -> tuple[th.Tensor, dict[str, float]]:
    """Localize sign-conditioned support in P and align P's polarity.

    ``polarity`` is +1 for a relatively good action and -1 for a relatively
    bad action. The original N region is always suppressed; it is never
    converted into a positive target when polarity is negative.
    """
    if not 0.0 <= alignment_margin <= 1.0:
        raise ValueError("alignment_margin must be in [0, 1]")
    if sign_weight < 0:
        raise ValueError("sign_weight must be non-negative")
    attr = _spatial_contribution(explanation)
    positive = (signed_mask > 0).to(attr.dtype)
    negative = (signed_mask < 0).to(attr.dtype)
    has_positive = positive.flatten(start_dim=1).any(dim=1)
    has_negative = negative.flatten(start_dim=1).any(dim=1)
    valid = has_positive
    absolute = attr.abs()
    p_absolute = (absolute * positive).sum(dim=(1, 2))
    n_absolute = (absolute * negative).sum(dim=(1, 2))
    p_signed = (attr * positive).sum(dim=(1, 2))
    p_alignment = p_signed / (p_absolute + eps)
    desired_alignment = polarity.to(attr.dtype) * p_alignment
    desired_positive = (
        polarity.to(attr.dtype).view(-1, 1, 1) * attr
    ).clamp(min=0)
    p_energy = (desired_positive * positive).sum(dim=(1, 2))
    n_energy = (attr.clamp(min=0) * negative).sum(dim=(1, 2))
    localization_loss = 1.0 - p_energy / (p_energy + n_energy + eps)
    sign_loss = th.clamp(
        alignment_margin - desired_alignment,
        min=0.0,
    )
    per_sample = localization_loss + sign_weight * sign_loss
    loss = per_sample[valid].mean() if valid.any() else attr.sum() * 0.0
    with th.no_grad():
        stats = {
            "masked_energy_ratio": (
                (1.0 - localization_loss[valid]).mean().item()
                if valid.any()
                else 0.0
            ),
            "positive_alignment": (
                p_alignment[valid].mean().item() if valid.any() else 0.0
            ),
            "desired_sign_alignment": (
                desired_alignment[valid].mean().item()
                if valid.any()
                else 0.0
            ),
            "sign_margin_satisfied_rate": (
                (desired_alignment[valid] >= alignment_margin)
                .float()
                .mean()
                .item()
                if valid.any()
                else 0.0
            ),
            "negative_absolute_fraction": (
                (
                    n_absolute[valid]
                    / (p_absolute[valid] + n_absolute[valid] + eps)
                )
                .mean()
                .item()
                if valid.any()
                else 0.0
            ),
            "negative_positive_fraction": (
                (n_energy[has_negative] / (n_absolute[has_negative] + eps))
                .mean()
                .item()
                if has_negative.any()
                else 0.0
            ),
        }
    return loss, stats


# ──────────────────────────────────────────────────────────────────────────────
# Signed action-conditioned attribution supervision
# ──────────────────────────────────────────────────────────────────────────────

def signed_attribution_margin_loss(
    explanation: th.Tensor,
    signed_mask: th.Tensor,
    margin: float = 0.01,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> tuple[th.Tensor, dict]:
    """
    Action-conditioned signed attribution supervision via a margin/hinge loss.

    Args:
        explanation: (B, C, H, W) signed attribution (grad × input) for the
            explained action. NOT detached — gradients must flow into the policy.
        signed_mask: (B, H, W) with values in {-1, 0, +1}:
            +1 → cell should positively support the action  (want A ≥ +margin)
            -1 → cell should negatively affect the action    (want A ≤ -margin)
             0 → unsupervised; contributes no gradient and is excluded from the
                 normaliser.
        margin: desired margin on RMS-normalized signed attribution (> 0).
        pos_weight: weight applied to the hinge on +1 cells.
        neg_weight: weight applied to the hinge on -1 cells.

    Per supervised cell (S ≠ 0) the loss is ``max(0, margin − S · A)``, which is
    zero once ``S · A ≥ margin`` and grows linearly as the attribution moves the
    wrong way. Channel aggregation is a signed sum over channels
    (``explanation.sum(dim=1)``), matching the B-cos contribution map and the
    repo's ``visualize_minigrid`` heatmaps.

    Returns:
        (loss, stats) where ``loss`` is a scalar tensor (graph-attached, so
        ``backward`` always works — 0.0 when no cells are supervised) and
        ``stats`` is a dict of detached floats for logging.

    The loss is finite for an all-zero mask (returns 0.0) and never divides by
    zero.
    """
    # Signed per-cell attribution — matches visualize_minigrid.explanation.sum(0)
    # and the bcos library's contribution_map = (input * grad).sum(1).
    attr = _spatial_contribution(explanation)            # (B, H, W), signed, grad-attached
    S = signed_mask

    pos_cells = S > 0
    neg_cells = S < 0
    omega = S != 0                                      # supervised cells
    per_sample_count = omega.sum(dim=(1, 2), keepdim=True).clamp_min(1)
    supervised_square = (attr.square() * omega).sum(dim=(1, 2), keepdim=True)
    rms = th.sqrt(supervised_square / per_sample_count + 1e-4)
    attr = attr / rms
    n = omega.sum()

    # Detached diagnostics — always finite, even for an empty mask.
    with th.no_grad():
        n_pos = pos_cells.sum()
        n_neg = neg_cells.sum()
        mean_attr_pos = attr[pos_cells].mean().item() if n_pos > 0 else 0.0
        mean_attr_neg = attr[neg_cells].mean().item() if n_neg > 0 else 0.0
        # Region hinge means (unweighted), for logging only.
        pos_loss_val = (
            (th.clamp(margin - attr, min=0.0) * pos_cells).sum() / n_pos
        ).item() if n_pos > 0 else 0.0
        neg_loss_val = (
            (th.clamp(margin + attr, min=0.0) * neg_cells).sum() / n_neg
        ).item() if n_neg > 0 else 0.0

    if n == 0:
        # Keep the result attached to the graph (× 0) so callers can always
        # call backward() unconditionally.
        loss = attr.sum() * 0.0
        stats = {
            "signed_loss": 0.0,
            "signed_pos_loss": 0.0,
            "signed_neg_loss": 0.0,
            "n_pos_cells": 0.0,
            "n_neg_cells": 0.0,
            "mean_attr_pos": mean_attr_pos,
            "mean_attr_neg": mean_attr_neg,
        }
        return loss, stats

    hinge = th.clamp(margin - S * attr, min=0.0)        # (B, H, W)
    weight = th.where(
        pos_cells,
        th.as_tensor(pos_weight, device=attr.device, dtype=attr.dtype),
        th.as_tensor(neg_weight, device=attr.device, dtype=attr.dtype),
    )
    # Zero out unsupervised cells (S=0); normalise by the number of supervised
    # cells so neutral cells influence neither the numerator nor the denominator.
    per_cell = hinge * weight * omega.to(attr.dtype)
    loss = per_cell.sum() / n.to(attr.dtype)

    stats = {
        "signed_loss": loss.item(),
        "signed_pos_loss": pos_loss_val,
        "signed_neg_loss": neg_loss_val,
        "n_pos_cells": float(n_pos.item()),
        "n_neg_cells": float(n_neg.item()),
        "mean_attr_pos": mean_attr_pos,
        "mean_attr_neg": mean_attr_neg,
    }
    return loss, stats

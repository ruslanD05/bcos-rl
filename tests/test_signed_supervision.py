#!/usr/bin/env python3
"""Tests for action-conditioned signed attribution supervision.

Covers (a) the margin loss ``signed_attribution_margin_loss`` and (b) the
``SignedActionMaskWrapper`` mask construction.

Run with pytest::

    uv run pytest tests/test_signed_supervision.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch as th

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from bcos_rl_generic.explanation_supervision import (
    absolute_attribution_energy_loss,
    advantage_conditioned_partial_energy_loss,
    partial_masked_energy_loss,
    polarity_separated_attribution_margin_loss,
    positive_attribution_margin_loss,
    signed_attribution_margin_loss,
)
from minigrid.core.world_object import Wall, Door, Key, Goal, Box
MARGIN = 0.5
ATOL = 1e-5


def test_absolute_energy_localizes_magnitude_without_prescribing_sign():
    mask = th.tensor([[[1.0, 0.0]]])
    mostly_inside = th.tensor([[[[-3.0, 1.0]]]], requires_grad=True)
    mostly_outside = th.tensor([[[[-1.0, 3.0]]]], requires_grad=True)

    inside_loss, stats = absolute_attribution_energy_loss(mostly_inside, mask)
    outside_loss, _ = absolute_attribution_energy_loss(mostly_outside, mask)

    assert inside_loss.item() < outside_loss.item()
    assert abs(stats["target_energy_ratio"] - 0.75) < ATOL
    inside_loss.backward()
    assert mostly_inside.grad is not None


def test_advantage_conditioned_loss_flips_only_positive_target_sign():
    mask = th.tensor([[[1.0, -1.0]]])
    matching = th.tensor([[[[-2.0, 0.0]]]])
    wrong = th.tensor([[[[2.0, 0.0]]]])
    negative_polarity = th.tensor([-1.0])
    matching_loss, stats = advantage_conditioned_partial_energy_loss(
        matching,
        mask,
        negative_polarity,
        alignment_margin=0.5,
    )
    wrong_loss, _ = advantage_conditioned_partial_energy_loss(
        wrong,
        mask,
        negative_polarity,
        alignment_margin=0.5,
    )
    assert matching_loss.item() < wrong_loss.item()
    assert stats["desired_sign_alignment"] > 0.99


def test_advantage_conditioned_loss_keeps_negative_region_suppressed():
    mask = th.tensor([[[1.0, -1.0]]])
    clean = th.tensor([[[[2.0, 0.0]]]])
    leaked = th.tensor([[[[2.0, 2.0]]]])
    polarity = th.tensor([1.0])
    clean_loss, _ = advantage_conditioned_partial_energy_loss(
        clean, mask, polarity
    )
    leaked_loss, _ = advantage_conditioned_partial_energy_loss(
        leaked, mask, polarity
    )
    assert clean_loss.item() < leaked_loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_expl_and_mask(H=4, W=4, C=3):
    """Zero explanation (B=1) + zero signed mask, ready to be edited per-test."""
    expl = th.zeros(1, C, H, W, requires_grad=False)
    mask = th.zeros(1, H, W)
    return expl, mask


def _set_cell_attr(expl, h, w, per_channel_value):
    """Set the attribution at pixel (h, w) so its channel-sum equals target.

    signed_attribution_margin_loss aggregates channels via sum, so to get a
    per-cell signed attribution A we split it evenly across channels.
    """
    C = expl.shape[1]
    expl[:, :, h, w] = per_channel_value / C


def _cellsum(expl, h, w):
    return expl[:, :, h, w].sum().item()


# ──────────────────────────────────────────────────────────────────────────────
# Loss: forward into a free cell  (S = +1)
# ──────────────────────────────────────────────────────────────────────────────

def test_forward_free_cell_attr_above_margin_zero_loss():
    expl, mask = _make_expl_and_mask()
    mask[0, 1, 1] = +1.0
    _set_cell_attr(expl, 1, 1, per_channel_value=2 * MARGIN)  # A = 2m > m
    loss, stats = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert abs(loss.item()) < ATOL, loss.item()
    assert stats["n_pos_cells"] == 1.0


def test_forward_free_cell_attr_below_margin_positive_loss():
    expl, mask = _make_expl_and_mask()
    mask[0, 1, 1] = +1.0
    _set_cell_attr(expl, 1, 1, per_channel_value=0.0)  # A = 0 < m
    loss, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert loss.item() > 0.0
    # hinge = m - A = m
    assert abs(loss.item() - MARGIN) < ATOL


# ──────────────────────────────────────────────────────────────────────────────
# Loss: forward into a wall  (S = -1)
# ──────────────────────────────────────────────────────────────────────────────

def test_forward_wall_attr_below_neg_margin_zero_loss():
    expl, mask = _make_expl_and_mask()
    mask[0, 2, 2] = -1.0
    _set_cell_attr(expl, 2, 2, per_channel_value=-2 * MARGIN)  # A = -2m <= -m
    loss, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert abs(loss.item()) < ATOL, loss.item()


def test_forward_wall_attr_positive_high_loss():
    expl, mask = _make_expl_and_mask()
    mask[0, 2, 2] = -1.0
    _set_cell_attr(expl, 2, 2, per_channel_value=+MARGIN)  # A = +m, want A <= -m
    loss, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert loss.item() > MARGIN


# ──────────────────────────────────────────────────────────────────────────────
# Loss: pickup with key ahead (S=+1) vs invalid object ahead (S=-1)
# ──────────────────────────────────────────────────────────────────────────────

def test_pickup_key_positive_attr_low_loss():
    expl, mask = _make_expl_and_mask()
    mask[0, 0, 3] = +1.0
    _set_cell_attr(expl, 0, 3, per_channel_value=+2 * MARGIN)
    loss, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert abs(loss.item()) < ATOL


def test_pickup_key_negative_attr_high_loss():
    expl, mask = _make_expl_and_mask()
    mask[0, 0, 3] = +1.0
    _set_cell_attr(expl, 0, 3, per_channel_value=-MARGIN)  # supports the opposite
    loss, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert loss.item() > MARGIN


def test_signed_loss_is_nearly_scale_invariant():
    expl, mask = _make_expl_and_mask()
    mask[0, 1, 1] = +1.0
    mask[0, 2, 2] = -1.0
    _set_cell_attr(expl, 1, 1, per_channel_value=+0.2)
    _set_cell_attr(expl, 2, 2, per_channel_value=+0.1)
    base, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    scaled, _ = signed_attribution_margin_loss(expl * 100.0, mask, margin=MARGIN)
    assert abs(base.item() - scaled.item()) < 0.01


def test_pickup_invalid_object_positive_attr_high_low():
    # Front object non-pickable → S = -1.
    expl, mask = _make_expl_and_mask()
    mask[0, 3, 3] = -1.0
    _set_cell_attr(expl, 3, 3, per_channel_value=+MARGIN)  # positive on invalid → high
    high, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert high.item() > MARGIN

    _set_cell_attr(expl, 3, 3, per_channel_value=-2 * MARGIN)  # sufficiently negative → low
    low, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert abs(low.item()) < ATOL


# ──────────────────────────────────────────────────────────────────────────────
# Loss: neutral cells & empty mask
# ──────────────────────────────────────────────────────────────────────────────

def test_neutral_cells_do_not_affect_loss():
    expl, mask = _make_expl_and_mask()
    mask[0, 1, 1] = +1.0
    _set_cell_attr(expl, 1, 1, per_channel_value=0.3)
    base, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)

    # Change attribution on a bunch of S=0 cells arbitrarily.
    expl[:, :, 0, 0] = 5.0
    expl[:, :, 3, 3] = -9.0
    expl[:, :, 2, 0] = 42.0
    after, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert abs(base.item() - after.item()) < ATOL


def test_empty_mask_zero_and_finite():
    expl, mask = _make_expl_and_mask()
    expl[:] = th.randn_like(expl)  # arbitrary attribution, but no supervised cells
    loss, stats = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    assert np.isfinite(loss.item())
    assert abs(loss.item()) < ATOL
    assert stats["n_pos_cells"] == 0.0 and stats["n_neg_cells"] == 0.0


def test_loss_is_differentiable():
    expl, mask = _make_expl_and_mask()
    expl.requires_grad_(True)
    mask[0, 1, 1] = +1.0
    mask[0, 2, 2] = -1.0
    loss, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    loss.backward()
    assert expl.grad is not None and th.isfinite(expl.grad).all()


def test_empty_mask_backward_is_safe():
    expl, mask = _make_expl_and_mask()
    expl.requires_grad_(True)
    loss, _ = signed_attribution_margin_loss(expl, mask, margin=MARGIN)
    loss.backward()  # must not raise even though no cell is supervised
    assert expl.grad is not None


def test_positive_margin_ignores_pixels_outside_target():
    explanation = th.zeros(1, 2, 3, 3, requires_grad=True)
    mask = th.zeros(1, 3, 3)
    mask[0, 1, 1] = 1.0

    loss, stats = positive_attribution_margin_loss(
        explanation, mask, margin=0.5
    )
    loss.backward()

    assert abs(loss.item() - 0.5) < ATOL
    assert stats["mean_target_attribution"] == 0.0
    assert explanation.grad is not None
    assert th.all(explanation.grad[:, :, 1, 1] < 0)
    outside = explanation.grad.clone()
    outside[:, :, 1, 1] = 0
    assert th.count_nonzero(outside) == 0


def test_positive_margin_sums_channels_before_positive_hinge():
    explanation = th.zeros(1, 2, 2, 2)
    mask = th.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0
    explanation[0, 0, 0, 0] = 2.0
    explanation[0, 1, 0, 0] = -2.0

    loss, _ = positive_attribution_margin_loss(
        explanation, mask, margin=0.5
    )
    assert abs(loss.item() - 0.5) < ATOL


def test_polarity_separated_positive_does_not_cancel_channels():
    explanation = th.zeros(1, 2, 2, 2)
    mask = th.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0
    explanation[0, 0, 0, 0] = 2.0
    explanation[0, 1, 0, 0] = -2.0

    loss, stats = polarity_separated_attribution_margin_loss(
        explanation, mask, polarity="positive", margin=0.4
    )
    assert loss.item() < ATOL
    assert stats["mean_target_energy"] > 0.49


def test_polarity_separated_negative_does_not_cancel_channels():
    explanation = th.zeros(1, 2, 2, 2)
    mask = th.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0
    explanation[0, 0, 0, 0] = 2.0
    explanation[0, 1, 0, 0] = -2.0

    loss, stats = polarity_separated_attribution_margin_loss(
        explanation, mask, polarity="negative", margin=0.4
    )
    assert loss.item() < ATOL
    assert stats["mean_target_energy"] > 0.49


def test_polarity_separated_wrong_sign_has_nonzero_gradient():
    explanation = th.ones(1, 2, 2, 2, requires_grad=True)
    mask = th.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0

    loss, _ = polarity_separated_attribution_margin_loss(
        explanation, mask, polarity="negative", margin=0.4
    )
    loss.backward()
    assert loss.item() > 0.39
    assert explanation.grad is not None
    assert th.count_nonzero(explanation.grad[:, :, 0, 0]) > 0


def test_positive_margin_is_nearly_scale_invariant():
    explanation = th.zeros(1, 1, 2, 2)
    explanation[0, 0, 0, 0] = -0.2
    mask = th.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0
    base, _ = positive_attribution_margin_loss(explanation, mask, margin=0.5)
    scaled, _ = positive_attribution_margin_loss(
        explanation * 100.0, mask, margin=0.5
    )
    assert abs(base.item() - scaled.item()) < 0.01


def test_partial_energy_prefers_positive_over_negative_region():
    mask = th.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0
    mask[0, 0, 1] = -1.0

    good = th.zeros(1, 1, 2, 2)
    good[0, 0, 0, 0] = 2.0
    good_loss, good_stats = partial_masked_energy_loss(good, mask)

    bad = th.zeros(1, 1, 2, 2)
    bad[0, 0, 0, 1] = 2.0
    bad_loss, _ = partial_masked_energy_loss(bad, mask)

    assert good_loss.item() < ATOL
    assert abs(good_stats["masked_energy_ratio"] - 1.0) < ATOL
    assert abs(bad_loss.item() - 1.0) < ATOL


def test_partial_energy_handles_positive_and_negative_only_masks():
    positive_mask = th.zeros(1, 2, 2)
    positive_mask[0, 0, 0] = 1.0
    positive_attr = th.zeros(1, 1, 2, 2)
    positive_attr[0, 0, 0, 0] = 1.0
    positive_loss, _ = partial_masked_energy_loss(
        positive_attr, positive_mask, alignment_margin=0.5
    )
    assert positive_loss.item() < ATOL

    negative_mask = -positive_mask
    negative_attr = th.zeros(1, 1, 2, 2)
    negative_attr[0, 0, 0, 0] = -1.0
    negative_loss, _ = partial_masked_energy_loss(
        negative_attr, negative_mask
    )
    assert negative_loss.item() < ATOL


def test_partial_energy_neutral_region_has_zero_gradient():
    explanation = th.zeros(1, 1, 2, 3, requires_grad=True)
    mask = th.zeros(1, 2, 3)
    mask[0, 0, 0] = 1.0
    mask[0, 0, 1] = -1.0
    loss, _ = partial_masked_energy_loss(explanation, mask)
    loss.backward()
    assert explanation.grad is not None
    assert th.count_nonzero(explanation.grad[:, :, :, 2]) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Mask construction: SignedActionMaskWrapper
# ──────────────────────────────────────────────────────────────────────────────

def _make_doorkey_wrapper(
    enable_negative=True,
    neg_on_locked_door=False,
    forward_traversable_positive=True,
):
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import SignedActionMaskWrapper

    env = gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = SignedActionMaskWrapper(
        env, enable_negative=enable_negative,
        neg_on_locked_door_without_key=neg_on_locked_door,
        forward_traversable_positive=forward_traversable_positive,
    )
    return env


def _make_symbolic_doorkey_wrapper(enable_negative=True):
    import gymnasium as gym
    from env_wrappers import make_wrapper

    return make_wrapper(
        use_rgb_img_obs=False,
        use_one_hot_symbolic_obs=True,
        use_signed_action_mask=True,
        signed_enable_negative=enable_negative,
    )(gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False))


def _front_tile_value(env, mask):
    """Return the unique non-zero target and the complete front-tile block."""
    raw = env.unwrapped
    fx, fy = (int(v) for v in raw.front_pos)
    th_, tw = env._tile_h, env._tile_w
    block = mask[fy * th_:(fy + 1) * th_, fx * tw:(fx + 1) * tw]
    nonzero = np.unique(block[block != 0])
    if nonzero.size == 0:
        return 0.0, block
    assert nonzero.size == 1, nonzero
    return float(nonzero[0]), block


def test_wrapper_forward_into_wall_is_negative():
    env = _make_doorkey_wrapper()
    env.reset(seed=0)
    raw = env.unwrapped
    actions = raw.actions
    # Place a wall directly in front, agent facing it.
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close(); return
    raw.grid.set(fx, fy, Wall())
    mask = env._compute_signed_mask(int(actions.forward))
    val, block = _front_tile_value(env, mask)
    assert val == -1.0
    assert np.any(block == -1.0)
    # MiniGrid walls visually occupy the complete rendered tile.
    assert np.all(block == -1.0)
    env.close()


def test_wrapper_forward_into_free_is_positive():
    env = _make_doorkey_wrapper()
    env.reset(seed=0)
    raw = env.unwrapped
    actions = raw.actions
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close(); return
    raw.grid.set(fx, fy, None)  # empty → traversable
    mask = env._compute_signed_mask(int(actions.forward))
    val, block = _front_tile_value(env, mask)
    assert val == +1.0
    assert np.any(block == 0.0)  # grid lines are not evidence of emptiness
    env.close()


def test_wrapper_forward_into_free_can_be_neutral():
    env = _make_doorkey_wrapper(forward_traversable_positive=False)
    env.reset(seed=0)
    raw = env.unwrapped
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close()
        return
    raw.grid.set(fx, fy, None)
    mask = env._compute_signed_mask(int(raw.actions.forward))
    assert np.all(mask == 0.0)
    env.close()


def test_wrapper_pickup_key_ahead_is_positive():
    env = _make_doorkey_wrapper()
    env.reset(seed=0)
    raw = env.unwrapped
    actions = raw.actions
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close(); return
    raw.grid.set(fx, fy, Key("yellow"))
    raw.carrying = None
    mask = env._compute_signed_mask(int(actions.pickup))
    val, block = _front_tile_value(env, mask)
    assert val == +1.0
    assert np.any(block == 0.0) and np.any(block == +1.0)
    env.close()


def test_wrapper_pickup_nonpickable_ahead_is_negative():
    env = _make_doorkey_wrapper()
    env.reset(seed=0)
    raw = env.unwrapped
    actions = raw.actions
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close(); return
    raw.grid.set(fx, fy, Wall())  # non-pickable
    mask = env._compute_signed_mask(int(actions.pickup))
    val, _ = _front_tile_value(env, mask)
    assert val == -1.0
    env.close()


def test_wrapper_pickup_empty_ahead_is_negative():
    env = _make_doorkey_wrapper()
    env.reset(seed=0)
    raw = env.unwrapped
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close()
        return
    raw.grid.set(fx, fy, None)
    value, _ = _front_tile_value(
        env, env._compute_signed_mask(int(raw.actions.pickup))
    )
    assert value == -1.0
    env.close()


def test_wrapper_left_right_are_neutral():
    env = _make_doorkey_wrapper()
    env.reset(seed=0)
    raw = env.unwrapped
    actions = raw.actions
    for a in (actions.left, actions.right, actions.done):
        mask = env._compute_signed_mask(int(a))
        assert np.all(mask == 0.0), a
    env.close()


def test_wrapper_disable_negative_suppresses_minus_one():
    env = _make_doorkey_wrapper(enable_negative=False)
    env.reset(seed=0)
    raw = env.unwrapped
    actions = raw.actions
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close(); return
    raw.grid.set(fx, fy, Wall())
    mask = env._compute_signed_mask(int(actions.forward))
    assert np.all(mask >= 0.0)          # no -1 written
    assert not np.any(mask == -1.0)
    env.close()


def test_symbolic_one_hot_observation_and_mask_are_grid_aligned():
    from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

    env = _make_symbolic_doorkey_wrapper()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5, 5, 21)
    assert obs.dtype == np.float32
    assert np.all(obs.sum(axis=-1) == 3.0)
    assert info["attn_mask"].shape == (5, 5)
    assert env.env._tile_h == 1 and env.env._tile_w == 1

    raw = env.unwrapped
    ax, ay = (int(v) for v in raw.agent_pos)
    color_offset = len(OBJECT_TO_IDX)
    state_offset = color_offset + len(COLOR_TO_IDX)
    assert obs[ay, ax, OBJECT_TO_IDX["agent"]] == 1.0
    assert obs[ay, ax, state_offset + raw.agent_dir] == 1.0

    for gx in range(raw.width):
        for gy in range(raw.height):
            if raw.grid.get(gx, gy) is None:
                assert obs[gy, gx, OBJECT_TO_IDX["empty"]] == 1.0
                env.close()
                return
    env.close()
    raise AssertionError("Expected at least one empty grid cell")


def test_symbolic_bcos_cnn_preserves_batch_and_gradient():
    from bcos_rl_generic.train_model import SymbolicGridCNNExtractor

    env = _make_symbolic_doorkey_wrapper()
    obs, _ = env.reset(seed=0)
    extractor = SymbolicGridCNNExtractor(
        env.observation_space,
        features_dim=32,
        use_bcos=True,
        stable_bcos_norm=True,
    )
    features = extractor(th.as_tensor(obs[None]))
    assert features.shape == (1, 32)
    features.square().mean().backward()
    assert any(
        parameter.grad is not None and th.isfinite(parameter.grad).all()
        for parameter in extractor.parameters()
    )
    env.close()


def test_wrapper_toggle_unlocked_door_positive_locked_neutral():
    # Unlocked door in front → +1.
    env = _make_doorkey_wrapper(neg_on_locked_door=False)
    env.reset(seed=0)
    raw = env.unwrapped
    actions = raw.actions
    fx, fy = (int(v) for v in raw.front_pos)
    if not (0 <= fx < raw.width and 0 <= fy < raw.height):
        env.close(); return
    raw.grid.set(fx, fy, Door("yellow", is_locked=False))
    val, _ = _front_tile_value(env, env._compute_signed_mask(int(actions.toggle)))
    assert val == +1.0

    # Locked door, no key, default policy → neutral (0).
    raw.grid.set(fx, fy, Door("yellow", is_locked=True))
    raw.carrying = None
    val2, _ = _front_tile_value(env, env._compute_signed_mask(int(actions.toggle)))
    assert val2 == 0.0
    env.close()


def test_combined_wrapper_emits_both_pre_action_masks():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import CombinedExplanationMaskWrapper

    env = gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = CombinedExplanationMaskWrapper(env)
    env.reset(seed=0)

    raw = env.unwrapped
    fx, fy = (int(v) for v in raw.front_pos)
    for gx in range(raw.width):
        for gy in range(raw.height):
            if getattr(raw.grid.get(gx, gy), "type", None) == "key":
                raw.grid.set(gx, gy, None)
    raw.grid.set(fx, fy, Key("yellow"))
    raw.carrying = None
    env._prev_mask = env._compute_mask()

    _, _, _, _, info = env.step(int(raw.actions.pickup))
    assert set(("sequential_attn_mask", "action_attn_mask")) <= info.keys()
    assert info["sequential_attn_mask"].shape == info["action_attn_mask"].shape

    action_value, _ = _front_tile_value(env, info["action_attn_mask"])
    assert action_value == +1.0
    tile_h, tile_w = env._tile_h, env._tile_w
    key_block = info["sequential_attn_mask"][
        fy * tile_h:(fy + 1) * tile_h,
        fx * tile_w:(fx + 1) * tile_w,
    ]
    assert np.any(key_block == 1.0)
    assert np.any(key_block == 0.0)
    ax, ay = (int(v) for v in raw.agent_pos)
    agent_block = info["sequential_attn_mask"][
        ay * tile_h:(ay + 1) * tile_h,
        ax * tile_w:(ax + 1) * tile_w,
    ]
    assert np.any(agent_block == 1.0)
    assert np.any(agent_block == 0.0)
    env.close()


def test_rgb_sequential_key_mask_matches_rendered_object_footprint():
    import gymnasium as gym
    from minigrid.core.grid import Grid
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import ExplanationMaskWrapper

    env = gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
    env = ExplanationMaskWrapper(ImgObsWrapper(RGBImgObsWrapper(env)))
    _, info = env.reset(seed=0)
    raw = env.unwrapped
    key_pos = next(
        (x, y)
        for x in range(raw.width)
        for y in range(raw.height)
        if getattr(raw.grid.get(x, y), "type", None) == "key"
    )
    key = raw.grid.get(*key_pos)
    rows, cols = env._tile_slices(*key_pos)
    block = info["attn_mask"][rows, cols]
    rendered = Grid.render_tile(key, highlight=False, tile_size=env._tile_h)
    empty = Grid.render_tile(None, highlight=False, tile_size=env._tile_h)
    expected = np.any(rendered != empty, axis=-1)

    assert np.array_equal(block > 0, expected)
    assert expected.any() and (~expected).any()
    env.close()


def test_rgb_sequential_negative_mask_uses_rendered_walls_and_grid_lines():
    import gymnasium as gym
    from minigrid.core.grid import Grid
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import ExplanationMaskWrapper

    env = gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
    env = ExplanationMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(env)),
        include_walls_as_negative=True,
        include_grid_lines_as_negative=True,
    )
    _, info = env.reset(seed=0)
    raw = env.unwrapped
    mask = info["attn_mask"]

    wall_pos = next(
        (x, y)
        for x in range(raw.width)
        for y in range(raw.height)
        if getattr(raw.grid.get(x, y), "type", None) == "wall"
    )
    wall_rows, wall_cols = env._tile_slices(*wall_pos)
    assert np.all(mask[wall_rows, wall_cols] == -1.0)

    empty_pos = next(
        (x, y)
        for x in range(raw.width)
        for y in range(raw.height)
        if raw.grid.get(x, y) is None
        and (x, y) != tuple(int(v) for v in raw.agent_pos)
    )
    empty_rows, empty_cols = env._tile_slices(*empty_pos)
    empty_block = mask[empty_rows, empty_cols]
    rendered_empty = Grid.render_tile(
        None, highlight=False, tile_size=env._tile_h
    )
    expected_lines = np.any(rendered_empty != 0, axis=-1)
    assert np.array_equal(empty_block < 0, expected_lines)
    assert np.all(empty_block[~expected_lines] == 0.0)

    key_pos = next(
        (x, y)
        for x in range(raw.width)
        for y in range(raw.height)
        if getattr(raw.grid.get(x, y), "type", None) == "key"
    )
    key_rows, key_cols = env._tile_slices(*key_pos)
    key_block = mask[key_rows, key_cols]
    key = raw.grid.get(*key_pos)
    rendered_key = Grid.render_tile(
        key, highlight=False, tile_size=env._tile_h
    )
    expected_key = np.any(rendered_key != rendered_empty, axis=-1)
    assert np.array_equal(key_block > 0, expected_key)
    assert np.array_equal(key_block < 0, expected_lines & ~expected_key)

    ax, ay = (int(v) for v in raw.agent_pos)
    agent_rows, agent_cols = env._tile_slices(ax, ay)
    agent_block = mask[agent_rows, agent_cols]
    agent_cell = raw.grid.get(ax, ay)
    rendered_agent = Grid.render_tile(
        agent_cell,
        agent_dir=int(raw.agent_dir),
        highlight=False,
        tile_size=env._tile_h,
    )
    rendered_without_agent = Grid.render_tile(
        agent_cell,
        highlight=False,
        tile_size=env._tile_h,
    )
    expected_agent = np.any(
        rendered_agent != rendered_without_agent, axis=-1
    )
    assert np.array_equal(agent_block > 0, expected_agent)
    assert expected_agent.any() and (~expected_agent).any()
    assert np.array_equal(agent_block < 0, expected_lines & ~expected_agent)
    env.close()


def test_action_category_movement_masks_use_action_direction_not_global_walls():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import ActionCategoryMaskWrapper

    env = ActionCategoryMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-Empty-5x5-v0", highlight=False)
        ))
    )
    env.reset(seed=0)
    raw = env.unwrapped
    raw.agent_pos = np.array((2, 2))
    raw.agent_dir = 0

    expected_cells = {
        int(raw.actions.forward): (3, 2),
        int(raw.actions.left): (2, 1),
        int(raw.actions.right): (2, 3),
    }
    for action, expected_cell in expected_cells.items():
        mask = env._compute_action_category_mask(action)
        rows, cols = env._tile_slices(*expected_cell)
        assert np.any(mask[rows, cols] > 0)
        unrelated_rows, unrelated_cols = env._tile_slices(1, 1)
        assert not np.any(mask[unrelated_rows, unrelated_cols] > 0)

    env.close()


def test_all_negative_sequential_mask_has_no_neutral_pixels():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import AllNegativeSequentialMaskWrapper

    env = AllNegativeSequentialMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
        ))
    )
    _, info = env.reset(seed=0)
    mask = info["attn_mask"]
    assert np.any(mask > 0)
    assert np.any(mask < 0)
    assert not np.any(mask == 0)
    env.close()


def test_flipped_all_negative_sequential_mask_is_exact_inverse():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import (
        AllNegativeSequentialMaskWrapper,
        FlippedAllNegativeSequentialMaskWrapper,
    )

    normal = AllNegativeSequentialMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
        ))
    )
    flipped = FlippedAllNegativeSequentialMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
        ))
    )
    _, normal_info = normal.reset(seed=0)
    _, flipped_info = flipped.reset(seed=0)
    assert np.array_equal(flipped_info["attn_mask"], -normal_info["attn_mask"])
    assert not np.any(flipped_info["attn_mask"] == 0)
    normal.close()
    flipped.close()


def test_local_directional_movement_uses_exact_action_relative_cell():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import LocalDirectionalActionMaskWrapper

    env = LocalDirectionalActionMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-Empty-8x8-v0", highlight=False)
        ))
    )
    env.reset(seed=0)
    raw = env.unwrapped
    raw.agent_pos = np.array((3, 3))
    raw.agent_dir = 0

    expected_cells = {
        int(raw.actions.forward): (4, 3),
        int(raw.actions.left): (3, 2),
        int(raw.actions.right): (3, 4),
    }
    other_cells = {(4, 3), (3, 2), (3, 4)}
    for action, expected_cell in expected_cells.items():
        mask = env._compute_local_action_mask(action)
        rows, cols = env._tile_slices(*expected_cell)
        assert np.all(mask[rows, cols] > 0)
        for other_cell in other_cells - {expected_cell}:
            other_rows, other_cols = env._tile_slices(*other_cell)
            assert np.all(mask[other_rows, other_cols] < 0)

    env.close()


def test_local_directional_movement_does_not_neutralize_semantic_objects():
    import gymnasium as gym
    from minigrid.core.world_object import Goal, Key
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import LocalDirectionalActionMaskWrapper

    env = LocalDirectionalActionMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-Empty-8x8-v0", highlight=False)
        ))
    )
    env.reset(seed=0)
    raw = env.unwrapped
    raw.agent_pos = np.array((3, 3))
    raw.agent_dir = 0
    raw.grid.set(4, 3, Key("yellow"))
    raw.grid.set(5, 4, Goal())

    mask = env._compute_local_action_mask(int(raw.actions.forward))
    selected_rows, selected_cols = env._tile_slices(4, 3)
    unrelated_rows, unrelated_cols = env._tile_slices(5, 4)
    assert np.all(mask[selected_rows, selected_cols] > 0)
    assert np.all(mask[unrelated_rows, unrelated_cols] < 0)
    assert not np.any(mask == 0)
    env.close()


def test_action_category_interaction_targets_front_object_and_done_is_empty():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import ActionCategoryMaskWrapper

    env = ActionCategoryMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-Empty-5x5-v0", highlight=False)
        ))
    )
    env.reset(seed=0)
    raw = env.unwrapped
    raw.agent_pos = np.array((2, 2))
    raw.agent_dir = 0
    raw.grid.set(3, 2, Key("yellow"))

    pickup_mask = env._compute_action_category_mask(int(raw.actions.pickup))
    rows, cols = env._tile_slices(3, 2)
    assert np.any(pickup_mask[rows, cols] > 0)

    done_mask = env._compute_action_category_mask(int(raw.actions.done))
    assert not np.any(done_mask)
    env.close()


def test_local_action_movement_mask_is_strictly_local():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import LocalActionAffordanceMaskWrapper

    env = LocalActionAffordanceMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-Empty-8x8-v0", highlight=False)
        )),
        movement_radius=1,
    )
    env.reset(seed=0)
    raw = env.unwrapped
    raw.agent_pos = np.array((3, 3))
    raw.agent_dir = 0
    mask = env._compute_local_action_mask(int(raw.actions.forward))

    local_rows, local_cols = env._tile_slices(4, 4)
    distant_rows, distant_cols = env._tile_slices(5, 5)
    assert np.all(mask[local_rows, local_cols] > 0)
    assert np.all(mask[distant_rows, distant_cols] < 0)
    assert np.any(mask > 0) and np.any(mask < 0)
    env.close()


def test_local_action_movement_leaves_interactive_objects_neutral():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import LocalActionAffordanceMaskWrapper

    env = LocalActionAffordanceMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-Empty-8x8-v0", highlight=False)
        )),
        movement_radius=1,
    )
    env.reset(seed=0)
    raw = env.unwrapped
    raw.agent_pos = np.array((3, 3))
    raw.agent_dir = 0
    raw.grid.set(4, 3, Key("yellow"))

    mask = env._compute_local_action_mask(int(raw.actions.forward))
    rows, cols = env._tile_slices(4, 3)
    block = mask[rows, cols]
    footprint = env._rendered_footprint(raw.grid.get(4, 3))
    assert np.all(block[footprint] == 0)
    assert np.any(block[~footprint] > 0)
    env.close()


def test_local_action_interactions_require_a_valid_front_affordance():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import LocalActionAffordanceMaskWrapper

    env = LocalActionAffordanceMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-Empty-5x5-v0", highlight=False)
        ))
    )
    env.reset(seed=0)
    raw = env.unwrapped
    raw.agent_pos = np.array((2, 2))
    raw.agent_dir = 0
    front_rows, front_cols = env._tile_slices(3, 2)

    empty_pickup = env._compute_local_action_mask(int(raw.actions.pickup))
    assert np.all(empty_pickup[front_rows, front_cols] < 0)

    raw.grid.set(3, 2, Key("yellow"))
    valid_pickup = env._compute_local_action_mask(int(raw.actions.pickup))
    assert np.any(valid_pickup[front_rows, front_cols] > 0)
    assert np.any(valid_pickup[front_rows, front_cols] < 0)

    done = env._compute_local_action_mask(int(raw.actions.done))
    assert not np.any(done)
    env.close()


def test_flipped_sequential_mask_swaps_positive_and_negative_regions():
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import (
        ExplanationMaskWrapper,
        FlippedExplanationMaskWrapper,
    )

    normal = ExplanationMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
        )),
        include_walls_as_negative=True,
        include_grid_lines_as_negative=True,
    )
    flipped = FlippedExplanationMaskWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
        )),
        include_walls_as_negative=True,
        include_grid_lines_as_negative=True,
    )
    _, normal_info = normal.reset(seed=0)
    _, flipped_info = flipped.reset(seed=0)
    assert np.array_equal(
        flipped_info["attn_mask"],
        -normal_info["attn_mask"],
    )
    normal.close()
    flipped.close()


def test_wall_appearance_changes_pixels_but_preserves_wall_cells():
    import gymnasium as gym
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Wall
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import WallAppearanceWrapper

    hidden = WallAppearanceWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
        )),
        mode="hidden",
    )
    recolored = WallAppearanceWrapper(
        ImgObsWrapper(RGBImgObsWrapper(
            gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
        )),
        mode="recolored",
        color="yellow",
    )
    hidden_obs, _ = hidden.reset(seed=0)
    recolored_obs, _ = recolored.reset(seed=0)
    raw = hidden.unwrapped
    wall_pos = next(
        (x, y)
        for x in range(raw.width)
        for y in range(raw.height)
        if getattr(raw.grid.get(x, y), "type", None) == "wall"
    )
    gx, gy = wall_pos
    rows = slice(gy * hidden._tile_h, (gy + 1) * hidden._tile_h)
    cols = slice(gx * hidden._tile_w, (gx + 1) * hidden._tile_w)
    empty_tile = Grid.render_tile(
        None, tile_size=hidden._tile_h
    ).astype(hidden_obs.dtype)
    yellow_wall = Grid.render_tile(
        Wall("yellow"), tile_size=hidden._tile_h
    ).astype(recolored_obs.dtype)

    assert np.array_equal(hidden_obs[rows, cols], empty_tile)
    assert np.array_equal(recolored_obs[rows, cols], yellow_wall)
    assert getattr(hidden.unwrapped.grid.get(gx, gy), "type", None) == "wall"
    assert getattr(recolored.unwrapped.grid.get(gx, gy), "type", None) == "wall"
    hidden.close()
    recolored.close()


def test_rgb_static_agent_mask_matches_rendered_agent_footprint():
    import gymnasium as gym
    from minigrid.core.grid import Grid
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from env_wrappers import StaticMultiObjectMaskWrapper

    env = gym.make("MiniGrid-DoorKey-5x5-v0", highlight=False)
    env = StaticMultiObjectMaskWrapper(ImgObsWrapper(RGBImgObsWrapper(env)))
    _, info = env.reset(seed=0)
    raw = env.unwrapped
    ax, ay = (int(v) for v in raw.agent_pos)
    cell = raw.grid.get(ax, ay)
    rows, cols = env._tile_slices(ax, ay)
    block = info["attn_mask"][rows, cols]
    rendered = Grid.render_tile(
        cell,
        agent_dir=int(raw.agent_dir),
        highlight=False,
        tile_size=env._tile_h,
    )
    without_agent = Grid.render_tile(
        cell, highlight=False, tile_size=env._tile_h
    )
    expected = np.any(rendered != without_agent, axis=-1)

    assert np.array_equal(block > 0, expected)
    assert expected.any() and (~expected).any()
    env.close()


def test_literal_mask_cancels_opposing_labels():
    from env_wrappers import compose_sequential_action_mask

    sequential = np.zeros((2, 3), dtype=np.float32)
    action = np.zeros((2, 3), dtype=np.float32)
    sequential[0, 0] = 1.0
    sequential[0, 1] = 1.0
    action[0, 0] = -1.0
    action[1, 2] = 1.0

    combined = compose_sequential_action_mask(sequential, action)
    assert combined[0, 0] == 0.0
    assert combined[0, 1] == 1.0
    assert combined[1, 2] == 1.0
    assert combined[1, 1] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

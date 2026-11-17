from __future__ import annotations

import collections
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, is_vecenv_wrapped

from bcos_rl_generic.explanation_supervision import (
    absolute_attribution_energy_loss,
    attribution_map_consistency_loss,
    DualMaskedRolloutBuffer,
    MaskedRolloutBuffer,
    attribution_ratio_loss,
    attribution_ratio_loss_original,
    compute_actor_explanation,
    channelwise_masked_absolute_attribution_energy_loss,
    channelwise_polarity_masked_energy_loss,
    masked_absolute_attribution_energy_loss,
    patch_absolute_energy_loss,
    partial_masked_energy_loss,
    polarity_separated_attribution_margin_loss,
    positive_attribution_margin_loss,
    signed_attribution_margin_loss,
)


def positive_advantage_confidence_loss(
    action_log_prob: th.Tensor,
    advantages: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Sharpen categorical confidence only for positive-advantage actions.

    Minimizing ``-log pi(a_taken | s)`` is cross-entropy with the taken action
    as a one-hot target. Unlike BCE on one raw logit, this is invariant to a
    common shift of all action logits and therefore matches the categorical
    policy probability used by PPO.
    """
    selected = advantages > 0
    if selected.any():
        loss = -action_log_prob[selected].mean()
    else:
        loss = action_log_prob.sum() * 0.0
    return loss, selected


def explanation_supervision_selection(
    masks: th.Tensor,
    raw_advantages: th.Tensor,
    *,
    signed: bool,
) -> th.Tensor:
    """Select valid explanation targets without dropping signed negatives."""
    has_mask = masks.flatten(start_dim=1).ne(0).any(dim=1)
    return has_mask if signed else has_mask & (raw_advantages > 0)


def positive_margin_supervision_selection(
    masks: th.Tensor,
    normalized_advantages: th.Tensor,
) -> th.Tensor:
    """Select relatively good transitions using PPO-normalized advantages."""
    has_mask = masks.flatten(start_dim=1).ne(0).any(dim=1)
    return has_mask & (normalized_advantages >= 0)


def partial_energy_supervision_selection(
    masks: th.Tensor,
    normalized_advantages: th.Tensor,
) -> th.Tensor:
    """Keep all negative targets and relatively good positive-only targets."""
    has_positive = masks.flatten(start_dim=1).gt(0).any(dim=1)
    has_negative = masks.flatten(start_dim=1).lt(0).any(dim=1)
    return has_negative | (has_positive & (normalized_advantages >= 0))


class SuccessRateCheckpointCallback(BaseCallback):
    """Save the model when success rate first crosses `threshold`, then every `interval` steps."""

    def __init__(
        self,
        save_dir: str,
        threshold: float = 0.95,
        interval: int = 100_000,
        window_size: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.interval = interval
        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._milestone_hit = False
        self._next_save_at = 0

    def _save(self, tag: str) -> None:
        path = self.save_dir / f"model_{tag}"
        self.model.save(str(path))
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(str(self.save_dir / f"model_{tag}.vecnormalize.pkl"))
        if self.verbose:
            print(f"[Checkpoint] Saved {path}  (step {self.num_timesteps})")

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done and "is_success" in info:
                self._history.append(float(info["is_success"]))

        if not self._history:
            return True

        success_rate = float(np.mean(self._history))

        if not self._milestone_hit and success_rate >= self.threshold:
            self._milestone_hit = True
            self._save(f"succ{success_rate:.2f}_step{self.num_timesteps}")
            self._next_save_at = self.num_timesteps + self.interval

        if self._milestone_hit and self.num_timesteps >= self._next_save_at:
            self._save(f"succ{success_rate:.2f}_step{self.num_timesteps}")
            self._next_save_at += self.interval

        return True


class ConfidenceCoefficientRampCallback(BaseCallback):
    """Ramp the positive-advantage confidence coefficient during training."""

    def __init__(
        self,
        start_coef: float = 0.0,
        end_coef: float = 0.1,
        total_steps: int = 100_000,
        schedule: str = "linear",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if schedule not in ("linear", "exponential"):
            raise ValueError(f"schedule must be 'linear' or 'exponential', got {schedule!r}")
        if schedule == "exponential" and start_coef <= 0.0:
            raise ValueError("start_coef must be > 0 for an exponential schedule")
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.total_steps = total_steps
        self.schedule = schedule

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_steps)
        if self.schedule == "exponential":
            new_coef = self.start_coef * (self.end_coef / self.start_coef) ** progress
        else:
            new_coef = self.start_coef + (self.end_coef - self.start_coef) * progress
        self.model.confidence_coef = new_coef
        self.logger.record("train/confidence_coef", new_coef)
        return True


# Backward-compatible import name for older scripts.
BceCoefficientRampCallback = ConfidenceCoefficientRampCallback


class BcosPPO(PPO):
    def __init__(
        self,
        *args,
        confidence_coef: float | None = None,
        bce_coef: float | None = None,
        bce_temperature: float | None = None,
        explanation_coef: float | None = None,
        sequential_explanation_coef: float = 0.0,
        action_explanation_coef: float = 0.0,
        use_original_explanation_energy_loss: bool | None = None,
        explanation_loss_type: str | None = None,
        explanation_subtract_action_mean: bool = True,
        # Deprecated aliases retained for existing scripts and saved configs.
        aux_alpha: float | None = None,
        temperature: float | None = None,
        expl_alpha: float | None = None,
        use_original_energy_loss: bool | None = None,
        expl_loss_type: str | None = None,
        sequential_margin: float = 0.001,
        partial_energy_alignment_margin: float = 0.5,
        signed_margin: float = 0.01,
        signed_pos_weight: float = 1.0,
        signed_neg_weight: float = 1.0,
        dual_masks_exclusive: bool = False,
        explanation_max_grad_ratio: float = 0.1,
        explanation_counterfactual_rgb: tuple[int, int, int] = (100, 100, 100),
        gradient_diagnostics: bool = False,
        **kwargs,
    ):
        resolved_explanation_loss_type = (
            explanation_loss_type
            if explanation_loss_type is not None
            else (expl_loss_type if expl_loss_type is not None else "energy")
        )
        if resolved_explanation_loss_type not in (
            "energy", "partial_energy", "absolute_energy", "masked_absolute_energy", "channelwise_masked_absolute_energy", "channelwise_polarity_masked_energy", "patch_absolute_energy", "positive_margin", "signed", "explanation_consistency", "teacher_explanation_consistency"
        ):
            raise ValueError(
                "explanation_loss_type must be 'energy', 'partial_energy', "
                "'absolute_energy', 'masked_absolute_energy', "
                "'channelwise_masked_absolute_energy', "
                "'channelwise_polarity_masked_energy', "
                "'patch_absolute_energy', "
                "'positive_margin', 'signed', 'explanation_consistency' or "
                "'teacher_explanation_consistency', "
                f"got {resolved_explanation_loss_type!r}"
            )
        self.confidence_coef = (
            confidence_coef
            if confidence_coef is not None
            else (bce_coef if bce_coef is not None else (aux_alpha or 0.0))
        )
        self.bce_temperature = (
            bce_temperature
            if bce_temperature is not None
            else (temperature if temperature is not None else 1.0)
        )
        self.explanation_coef = (
            explanation_coef
            if explanation_coef is not None
            else (expl_alpha or 0.0)
        )
        if sequential_explanation_coef < 0 or action_explanation_coef < 0:
            raise ValueError("Combined explanation coefficients must be non-negative")
        if self.explanation_coef > 0 and (
            sequential_explanation_coef > 0 or action_explanation_coef > 0
        ):
            raise ValueError(
                "explanation_coef cannot be combined with separate sequential/action "
                "explanation coefficients"
            )
        self.sequential_explanation_coef = sequential_explanation_coef
        self.action_explanation_coef = action_explanation_coef
        if sequential_explanation_coef > 0 and sequential_margin <= 0:
            raise ValueError("sequential_margin must be positive when enabled")
        self.sequential_margin = sequential_margin
        if not 0 <= partial_energy_alignment_margin <= 1:
            raise ValueError("partial_energy_alignment_margin must be in [0, 1]")
        self.partial_energy_alignment_margin = partial_energy_alignment_margin
        self.use_original_explanation_energy_loss = (
            use_original_explanation_energy_loss
            if use_original_explanation_energy_loss is not None
            else bool(use_original_energy_loss)
        )
        self.explanation_loss_type = resolved_explanation_loss_type
        self.explanation_subtract_action_mean = bool(
            explanation_subtract_action_mean
        )
        self.signed_margin = signed_margin
        self.signed_pos_weight = signed_pos_weight
        self.signed_neg_weight = signed_neg_weight
        self.dual_masks_exclusive = bool(dual_masks_exclusive)
        if explanation_max_grad_ratio <= 0:
            raise ValueError("explanation_max_grad_ratio must be positive")
        self.explanation_max_grad_ratio = explanation_max_grad_ratio
        if len(explanation_counterfactual_rgb) != 3 or any(
            channel < 0 or channel > 255
            for channel in explanation_counterfactual_rgb
        ):
            raise ValueError("explanation_counterfactual_rgb must be an RGB triplet")
        self.explanation_counterfactual_rgb = tuple(
            int(channel) for channel in explanation_counterfactual_rgb
        )
        # Runtime-only frozen policy used by teacher explanation distillation.
        # It is deliberately excluded from model checkpoints below.
        self.explanation_teacher_policy = None
        self.gradient_diagnostics = gradient_diagnostics
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["explanation_teacher_policy"]

    def set_explanation_teacher(self, policy) -> None:
        """Attach and freeze the source policy used as an explanation teacher."""
        if getattr(policy, "actor_type", None) != "bcos":
            raise ValueError("Explanation teacher must use a B-cos actor")
        policy.to(self.device)
        policy.set_training_mode(False)
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        self.explanation_teacher_policy = policy

    def _counterfactual_wall_observations(
        self,
        observations: th.Tensor,
        wall_masks: th.Tensor,
    ) -> th.Tensor:
        """Restore exact wall pixels to canonical RGB, including RGB6 inverse channels."""
        if observations.ndim != 4 or observations.shape[1] not in (3, 6):
            raise ValueError(
                "Wall counterfactuals require NCHW RGB3 or RGB6 observations, "
                f"got {tuple(observations.shape)}"
            )
        result = observations.clone()
        wall_mask = wall_masks > 0
        for channel, value in enumerate(self.explanation_counterfactual_rgb):
            canonical = th.as_tensor(
                value,
                dtype=result.dtype,
                device=result.device,
            )
            result[:, channel] = th.where(
                wall_mask,
                canonical,
                result[:, channel],
            )
            if result.shape[1] == 6:
                inverse = th.as_tensor(
                    255 - value,
                    dtype=result.dtype,
                    device=result.device,
                )
                result[:, channel + 3] = th.where(
                    wall_mask,
                    inverse,
                    result[:, channel + 3],
                )
        return result

    @property
    def aux_alpha(self) -> float:
        """Deprecated alias for `confidence_coef`."""
        return self.confidence_coef

    @aux_alpha.setter
    def aux_alpha(self, value: float) -> None:
        self.confidence_coef = value

    @property
    def bce_coef(self) -> float:
        """Deprecated alias for `confidence_coef`."""
        return self.confidence_coef

    @bce_coef.setter
    def bce_coef(self, value: float) -> None:
        self.confidence_coef = value

    @property
    def temperature(self) -> float:
        """Deprecated alias for `bce_temperature`."""
        return self.bce_temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.bce_temperature = value

    @property
    def expl_alpha(self) -> float:
        """Deprecated alias for `explanation_coef`."""
        return self.explanation_coef

    @expl_alpha.setter
    def expl_alpha(self, value: float) -> None:
        self.explanation_coef = value

    @property
    def use_original_energy_loss(self) -> bool:
        """Deprecated alias for `use_original_explanation_energy_loss`."""
        return self.use_original_explanation_energy_loss

    @use_original_energy_loss.setter
    def use_original_energy_loss(self, value: bool) -> None:
        self.use_original_explanation_energy_loss = value

    @property
    def expl_loss_type(self) -> str:
        """Deprecated alias for `explanation_loss_type`."""
        return self.explanation_loss_type

    @expl_loss_type.setter
    def expl_loss_type(self, value: str) -> None:
        self.explanation_loss_type = value

    # ------------------------------------------------------------------
    # Replace the default RolloutBuffer with MaskedRolloutBuffer when
    # explanation supervision is active (explanation_coef > 0).
    # ------------------------------------------------------------------
    def _setup_model(self) -> None:
        super()._setup_model()
        use_dual_masks = (
            self.sequential_explanation_coef > 0
            or self.action_explanation_coef > 0
        )
        if self.explanation_coef > 0 or use_dual_masks:
            if getattr(self.policy, "actor_type", None) != "bcos":
                raise ValueError("Explanation supervision requires a B-cos actor")
            actor_extractor = getattr(self.policy, "pi_features_extractor", None)
            if (
                actor_extractor is not None
                and hasattr(actor_extractor, "use_bcos")
                and not actor_extractor.use_bcos
            ):
                raise ValueError(
                    "Explanation supervision requires a B-cos actor feature extractor"
                )
            obs_shape = self.observation_space.shape
            if len(obs_shape) != 3:
                mask_shape: tuple[int, ...] = (64, 64)
            elif is_vecenv_wrapped(self.env, VecTransposeImage) or (
                obs_shape[0] < obs_shape[1] and obs_shape[0] < obs_shape[2]
            ):
                # RGB observations may already be channels-first even without
                # an explicit VecTransposeImage wrapper.
                mask_shape = obs_shape[1:]
            else:
                # Channels-last (default non-RGB obs path): obs is (H, W, C).
                mask_shape = obs_shape[:2]
            buffer_class = DualMaskedRolloutBuffer if use_dual_masks else MaskedRolloutBuffer
            self.rollout_buffer = buffer_class(
                self.n_steps,
                self.observation_space,
                self.action_space,
                mask_shape=mask_shape,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

    def _freeze_batch_norm_stats(self) -> None:
        """Use rollout-time BatchNorm statistics while keeping affine gradients."""
        batch_norm_types = (
            th.nn.BatchNorm1d,
            th.nn.BatchNorm2d,
            th.nn.BatchNorm3d,
            th.nn.SyncBatchNorm,
        )
        for module in self.policy.modules():
            if isinstance(module, batch_norm_types):
                module.eval()

    def _actor_parameters(self) -> list[th.nn.Parameter]:
        modules = [
            self.policy.pi_features_extractor,
            self.policy.mlp_extractor.policy_net,
            self.policy.action_net,
        ]
        parameters: list[th.nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return parameters

    def _actor_logits(self, observations: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(observations)
        actor_features = features[0] if isinstance(features, tuple) else features
        latent_pi = self.policy.mlp_extractor.forward_actor(actor_features)
        return self.policy.action_net(latent_pi)

    def _balance_explanation_term(
        self,
        base_loss: th.Tensor,
        weighted_explanation_term: th.Tensor,
        actor_parameters: list[th.nn.Parameter],
    ) -> tuple[th.Tensor, float, float]:
        """Cap explanation-gradient norm relative to the PPO actor gradient."""
        base_grads = th.autograd.grad(
            base_loss, actor_parameters, retain_graph=True, allow_unused=True
        )
        explanation_grads = th.autograd.grad(
            weighted_explanation_term,
            actor_parameters,
            retain_graph=True,
            allow_unused=True,
        )

        def tensor_norm(grads) -> th.Tensor:
            terms = [grad.detach().square().sum() for grad in grads if grad is not None]
            if not terms:
                return th.zeros((), device=self.device)
            return th.sqrt(th.stack(terms).sum())

        base_norm = tensor_norm(base_grads)
        explanation_norm = tensor_norm(explanation_grads)
        if not th.isfinite(explanation_norm) or explanation_norm.item() == 0.0:
            return weighted_explanation_term * 0.0, 0.0, 0.0
        cap = self.explanation_max_grad_ratio * base_norm / (explanation_norm + 1e-12)
        scale = min(1.0, float(cap.item()))
        applied_ratio = float(
            (scale * explanation_norm / (base_norm + 1e-12)).item()
        )
        return weighted_explanation_term * scale, scale, applied_ratio

    def _record_rollout_logits(self, max_samples: int = 1024) -> None:
        """Track actor logits without changing the PPO optimization path."""
        if not isinstance(self.action_space, spaces.Discrete):
            return
        observations = self.rollout_buffer.observations.reshape(
            -1, *self.observation_space.shape
        )
        actions = self.rollout_buffer.actions.reshape(-1)
        if len(observations) > max_samples:
            indices = np.linspace(
                0, len(observations) - 1, max_samples, dtype=np.int64
            )
            observations = observations[indices]
            actions = actions[indices]

        was_training = self.policy.training
        self.policy.set_training_mode(False)
        try:
            with th.no_grad():
                logits = self._actor_logits(obs_as_tensor(observations, self.device))
                action_tensor = th.as_tensor(actions, device=self.device).long()
                taken_logits = logits.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        finally:
            self.policy.set_training_mode(was_training)

        self.logger.record("train/logits_mean", logits.mean().item())
        self.logger.record("train/logits_std", logits.std().item())
        self.logger.record("train/logits_min", logits.min().item())
        self.logger.record("train/logits_max", logits.max().item())
        self.logger.record("train/taken_logit_mean", taken_logits.mean().item())
        self.logger.record("train/taken_logit_std", taken_logits.std().item())
        self.logger.record(
            "train/taken_logit_pos_frac",
            (taken_logits > 0).float().mean().item(),
        )

    @staticmethod
    def _tensor_list_norm(tensors: tuple[th.Tensor | None, ...] | list[th.Tensor]) -> float:
        values = [tensor.detach().square().sum() for tensor in tensors if tensor is not None]
        if not values:
            return 0.0
        return th.stack(values).sum().sqrt().item()

    # ------------------------------------------------------------------
    # collect_rollouts: identical to OnPolicyAlgorithm's implementation
    # except that attn_masks are extracted from infos and forwarded to
    # MaskedRolloutBuffer.add().
    # ------------------------------------------------------------------
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if not isinstance(
            rollout_buffer, (MaskedRolloutBuffer, DualMaskedRolloutBuffer)
        ):
            return super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )

        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # ── Extract attn_masks from infos ────────────────────────────────
            add_kwargs: dict = {}
            if isinstance(rollout_buffer, MaskedRolloutBuffer):
                mask_list = [info.get("attn_mask") for info in infos]
                if any(m is not None for m in mask_list):
                    default = np.zeros(rollout_buffer._mask_shape, dtype=np.float32)
                    add_kwargs["attn_mask"] = np.stack(
                        [m if m is not None else default for m in mask_list]
                    )
            elif isinstance(rollout_buffer, DualMaskedRolloutBuffer):
                default = np.zeros(rollout_buffer._mask_shape, dtype=np.float32)
                sequential_masks = [
                    info.get("sequential_attn_mask") for info in infos
                ]
                action_masks = [info.get("action_attn_mask") for info in infos]
                add_kwargs["sequential_attn_mask"] = np.stack(
                    [m if m is not None else default for m in sequential_masks]
                )
                add_kwargs["action_attn_mask"] = np.stack(
                    [m if m is not None else default for m in action_masks]
                )

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                **add_kwargs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        if (
            self.confidence_coef <= 0
            and self.explanation_coef <= 0
            and self.sequential_explanation_coef <= 0
            and self.action_explanation_coef <= 0
        ):
            # Keep the no-auxiliary baseline identical to the installed SB3
            # PPO implementation. B-cos remains a policy architecture choice,
            # but rollout collection and optimization are delegated unchanged.
            super().train()
            self._record_rollout_logits()
            return

        self.policy.set_training_mode(True)
        # Rollouts are collected with the policy in eval mode. Keeping
        # BatchNorm in eval mode here ensures the first recomputed log-prob is
        # from the same policy that produced rollout_data.old_log_prob.
        # BatchNorm weight/bias parameters still receive gradients in eval mode.
        self._freeze_batch_norm_stats()
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, confidence_losses, expl_losses = [], [], [], []
        confidence_selected_rates, confidence_selected_probabilities = [], []
        sequential_expl_losses, action_expl_losses = [], []
        sequential_mean_attrs, sequential_margin_rates = [], []
        combined_expl_losses = []
        raw_positive_advantage_rates = []
        normalized_positive_advantage_rates = []
        sequential_mask_rates, sequential_selected_rates = [], []
        action_mask_rates, action_selected_rates = [], []
        partial_energy_ratios, partial_positive_alignments = [], []
        partial_negative_positive_fractions = []
        partial_positive_mask_rates, partial_negative_mask_rates = [], []
        partial_selected_rates = []
        absolute_energy_ratios, absolute_mask_coverages = [], []
        absolute_selected_rates = []
        masked_absolute_ratios, masked_absolute_negative_ratios = [], []
        masked_absolute_selected_rates = []
        consistency_map_l1s, consistency_max_map_l1s = [], []
        consistency_selected_rates = []
        mask_raw_batch_losses, mask_weighted_batch_losses = [], []
        mask_loss_to_ppo_ratios, ppo_plus_mask_objectives = [], []
        ppo_original_losses = []
        clip_fractions = []
        all_logits_mean, all_logits_std, all_logits_min, all_logits_max = [], [], [], []
        taken_logit_mean, taken_logit_std, taken_logit_pos_frac = [], [], []
        expl_inside_ratios, mask_coverages, expl_pos_fracs = [], [], []
        total_abs_energies, inside_energies = [], []
        logit_variances, mean_abs_logits = [], []
        # Signed-supervision diagnostics (only populated when expl_loss_type == "signed").
        signed_pos_losses, signed_neg_losses = [], []
        signed_n_pos_cells, signed_n_neg_cells = [], []
        signed_mean_attr_pos, signed_mean_attr_neg = [], []
        actor_grad_norms, policy_grad_norms = [], []
        ppo_actor_grad_norms, explanation_actor_grad_norms = [], []
        actor_param_max_changes, actor_logit_max_changes = [], []
        explanation_graph_attached = []
        explanation_balance_scales, explanation_applied_grad_ratios = [], []

        actor_parameters = self._actor_parameters()
        optimizer_parameter_ids = {
            id(parameter)
            for group in self.policy.optimizer.param_groups
            for parameter in group["params"]
        }
        actor_parameters_missing_optimizer = sum(
            id(parameter) not in optimizer_parameter_ids for parameter in actor_parameters
        )

        loss_fn = (
            attribution_ratio_loss_original
            if self.use_original_explanation_energy_loss
            else attribution_ratio_loss
        )
        use_signed_expl = self.explanation_loss_type == "signed"
        use_positive_margin_expl = self.explanation_loss_type == "positive_margin"
        use_partial_energy_expl = self.explanation_loss_type == "partial_energy"
        use_absolute_energy_expl = self.explanation_loss_type == "absolute_energy"
        use_masked_absolute_expl = (
            self.explanation_loss_type == "masked_absolute_energy"
        )
        use_channelwise_masked_absolute_expl = (
            self.explanation_loss_type == "channelwise_masked_absolute_energy"
        )
        use_channelwise_polarity_masked_expl = (
            self.explanation_loss_type == "channelwise_polarity_masked_energy"
        )
        use_patch_absolute_expl = (
            self.explanation_loss_type == "patch_absolute_energy"
        )
        use_consistency_expl = (
            self.explanation_loss_type == "explanation_consistency"
        )
        use_teacher_consistency_expl = (
            self.explanation_loss_type == "teacher_explanation_consistency"
        )
        if (
            self.explanation_coef > 0
            and use_teacher_consistency_expl
            and self.explanation_teacher_policy is None
        ):
            raise RuntimeError(
                "teacher_explanation_consistency requires set_explanation_teacher()"
            )

        use_expl = (
            self.explanation_coef > 0
            and isinstance(self.rollout_buffer, MaskedRolloutBuffer)
        )
        use_dual_expl = isinstance(
            self.rollout_buffer, DualMaskedRolloutBuffer
        ) and (
            self.sequential_explanation_coef > 0
            or self.action_explanation_coef > 0
        )
        use_confidence = self.confidence_coef > 0

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                run_diagnostics = self.gradient_diagnostics and not ppo_actor_grad_norms
                explanation_for_diagnostics = None
                explanation_loss_for_diagnostics = None
                explanation_coef_for_diagnostics = self.explanation_coef
                mask_raw_for_batch = 0.0
                mask_weighted_for_batch = 0.0
                applied_explanation_term_for_batch = 0.0
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Capture actor logits for free during the existing forward pass —
                # no duplicate feature extraction needed.
                actor_logits_cache: list[th.Tensor] = []
                hook = self.policy.action_net.register_forward_hook(
                    lambda m, inp, out: actor_logits_cache.append(out)
                )
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                hook.remove()
                actor_logits = actor_logits_cache[0]  # (B, n_actions), grad attached
                values = values.flatten()

                # Keep raw GAE advantages for auxiliary targets. PPO's
                # mini-batch normalization can change their sign.
                raw_advantages = rollout_data.advantages
                raw_positive_advantage_rates.append(
                    (raw_advantages > 0).float().mean().item()
                )
                advantages = raw_advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                normalized_positive_advantage_rates.append(
                    (advantages >= 0).float().mean().item()
                )

                # --- Standard PPO losses (identical to SB3) ---
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # Vanilla PPO objective, before confidence/explanation auxiliary
                # terms below are added — lets us see how much those terms
                # are perturbing training relative to plain PPO.
                ppo_original_losses.append(loss.item())

                # --- Positive-advantage categorical confidence sharpening ---
                taken_logits = actor_logits.gather(1, actions.unsqueeze(1)).squeeze(1)
                if use_confidence:
                    loss_confidence, confidence_selected = (
                        positive_advantage_confidence_loss(
                            log_prob,
                            raw_advantages,
                        )
                    )
                    confidence_losses.append(loss_confidence.item())
                    confidence_selected_rates.append(
                        confidence_selected.float().mean().item()
                    )
                    if confidence_selected.any():
                        confidence_selected_probabilities.append(
                            log_prob[confidence_selected].detach().exp().mean().item()
                        )
                    loss = loss + self.confidence_coef * loss_confidence

                # --- Attribution ratio loss (explanation supervision) --------
                if use_expl:
                    masks = self.rollout_buffer._current_masks

                    if masks is not None:
                        # Partial energy keeps negative affordance targets and
                        # gates positive-only targets by normalized advantage.
                        if use_consistency_expl or use_teacher_consistency_expl:
                            selected = (
                                masks.flatten(start_dim=1).ne(0).any(dim=1)
                                & (raw_advantages > 0)
                            )
                            consistency_selected_rates.append(
                                selected.float().mean().item()
                            )
                        elif (
                            use_masked_absolute_expl
                            or use_channelwise_masked_absolute_expl
                            or use_channelwise_polarity_masked_expl
                        ):
                            flat_masks = masks.flatten(start_dim=1)
                            selected = flat_masks.gt(0).any(dim=1) & flat_masks.lt(
                                0
                            ).any(dim=1)
                            masked_absolute_selected_rates.append(
                                selected.float().mean().item()
                            )
                        elif use_absolute_energy_expl:
                            selected = masks.flatten(start_dim=1).gt(0).any(dim=1)
                            absolute_selected_rates.append(
                                selected.float().mean().item()
                            )
                        elif use_partial_energy_expl:
                            selected = partial_energy_supervision_selection(
                                masks, advantages
                            )
                            flat_masks = masks.flatten(start_dim=1)
                            partial_positive_mask_rates.append(
                                flat_masks.gt(0).any(dim=1).float().mean().item()
                            )
                            partial_negative_mask_rates.append(
                                flat_masks.lt(0).any(dim=1).float().mean().item()
                            )
                            partial_selected_rates.append(
                                selected.float().mean().item()
                            )
                        elif use_positive_margin_expl:
                            selected = positive_margin_supervision_selection(
                                masks, advantages
                            )
                            flat_masks = masks.flatten(start_dim=1)
                            sequential_mask_rates.append(
                                flat_masks.ne(0).any(dim=1).float().mean().item()
                            )
                            sequential_selected_rates.append(
                                selected.float().mean().item()
                            )
                        else:
                            selected = explanation_supervision_selection(
                                masks,
                                raw_advantages,
                                signed=use_signed_expl,
                            )

                        if selected.any():
                            selected_obs = rollout_data.observations[selected]
                            selected_actions = actions[selected]
                            selected_masks = masks[selected]

                            explanation = compute_actor_explanation(
                                self.policy,
                                selected_obs,
                                selected_actions,
                                subtract_action_mean=self.explanation_subtract_action_mean,
                            )
                            explanation_for_diagnostics = explanation

                            if use_consistency_expl or use_teacher_consistency_expl:
                                counterfactual_obs = (
                                    self._counterfactual_wall_observations(
                                        selected_obs,
                                        selected_masks,
                                    )
                                )
                                if use_teacher_consistency_expl:
                                    self.explanation_teacher_policy.set_training_mode(
                                        False
                                    )
                                    counterfactual_explanation = (
                                        compute_actor_explanation(
                                            self.explanation_teacher_policy,
                                            counterfactual_obs,
                                            selected_actions,
                                            create_graph=False,
                                            subtract_action_mean=self.explanation_subtract_action_mean,
                                        ).detach()
                                    )
                                else:
                                    counterfactual_explanation = (
                                        compute_actor_explanation(
                                            self.policy,
                                            counterfactual_obs,
                                            selected_actions,
                                            subtract_action_mean=self.explanation_subtract_action_mean,
                                        )
                                    )
                                loss_expl, consistency_stats = (
                                    attribution_map_consistency_loss(
                                        explanation,
                                        counterfactual_explanation,
                                    )
                                )
                                expl_losses.append(loss_expl.item())
                                balanced_term, balance_scale, applied_ratio = (
                                    self._balance_explanation_term(
                                        loss,
                                        self.explanation_coef * loss_expl,
                                        actor_parameters,
                                    )
                                )
                                loss = loss + balanced_term
                                applied_explanation_term_for_batch = (
                                    balanced_term.detach().item()
                                )
                                explanation_balance_scales.append(balance_scale)
                                explanation_applied_grad_ratios.append(applied_ratio)
                                explanation_loss_for_diagnostics = loss_expl
                                explanation_coef_for_diagnostics = (
                                    self.explanation_coef * balance_scale
                                )
                                consistency_map_l1s.append(
                                    consistency_stats["mean_map_l1"]
                                )
                                consistency_max_map_l1s.append(
                                    consistency_stats["max_map_l1"]
                                )
                            elif (
                                use_masked_absolute_expl
                                or use_channelwise_masked_absolute_expl
                                or use_channelwise_polarity_masked_expl
                                or use_patch_absolute_expl
                            ):
                                loss_expl, masked_absolute_stats = (
                                    (
                                        patch_absolute_energy_loss
                                        if use_patch_absolute_expl
                                        else (
                                            channelwise_polarity_masked_energy_loss
                                            if use_channelwise_polarity_masked_expl
                                            else (
                                                channelwise_masked_absolute_attribution_energy_loss
                                                if use_channelwise_masked_absolute_expl
                                                else masked_absolute_attribution_energy_loss
                                            )
                                        )
                                    )(explanation, selected_masks)
                                )
                                expl_losses.append(loss_expl.item())
                                balanced_term, balance_scale, applied_ratio = (
                                    self._balance_explanation_term(
                                        loss,
                                        self.explanation_coef * loss_expl,
                                        actor_parameters,
                                    )
                                )
                                loss = loss + balanced_term
                                applied_explanation_term_for_batch = (
                                    balanced_term.detach().item()
                                )
                                explanation_balance_scales.append(balance_scale)
                                explanation_applied_grad_ratios.append(applied_ratio)
                                explanation_loss_for_diagnostics = loss_expl
                                explanation_coef_for_diagnostics = (
                                    self.explanation_coef * balance_scale
                                )
                                masked_absolute_ratios.append(
                                    masked_absolute_stats.get("target_energy_ratio", 0.0)
                                )
                                masked_absolute_negative_ratios.append(
                                    masked_absolute_stats.get(
                                        "negative_energy_ratio",
                                        masked_absolute_stats.get(
                                            "wrong_polarity_ratio", 0.0
                                        ),
                                    )
                                )
                            elif use_absolute_energy_expl:
                                loss_expl, absolute_stats = (
                                    absolute_attribution_energy_loss(
                                        explanation, selected_masks
                                    )
                                )
                                expl_losses.append(loss_expl.item())
                                balanced_term, balance_scale, applied_ratio = (
                                    self._balance_explanation_term(
                                        loss,
                                        self.explanation_coef * loss_expl,
                                        actor_parameters,
                                    )
                                )
                                loss = loss + balanced_term
                                applied_explanation_term_for_batch = (
                                    balanced_term.detach().item()
                                )
                                explanation_balance_scales.append(balance_scale)
                                explanation_applied_grad_ratios.append(applied_ratio)
                                explanation_loss_for_diagnostics = loss_expl
                                explanation_coef_for_diagnostics = (
                                    self.explanation_coef * balance_scale
                                )
                                absolute_energy_ratios.append(
                                    absolute_stats["target_energy_ratio"]
                                )
                                absolute_mask_coverages.append(
                                    absolute_stats["mask_coverage"]
                                )
                            elif use_partial_energy_expl:
                                loss_expl, partial_stats = partial_masked_energy_loss(
                                    explanation,
                                    selected_masks,
                                    alignment_margin=(
                                        self.partial_energy_alignment_margin
                                    ),
                                )
                                expl_losses.append(loss_expl.item())
                                balanced_term, balance_scale, applied_ratio = (
                                    self._balance_explanation_term(
                                        loss,
                                        self.explanation_coef * loss_expl,
                                        actor_parameters,
                                    )
                                )
                                loss = loss + balanced_term
                                applied_explanation_term_for_batch = balanced_term.detach().item()
                                explanation_balance_scales.append(balance_scale)
                                explanation_applied_grad_ratios.append(applied_ratio)
                                explanation_loss_for_diagnostics = loss_expl
                                explanation_coef_for_diagnostics = (
                                    self.explanation_coef * balance_scale
                                )
                                partial_energy_ratios.append(
                                    partial_stats["masked_energy_ratio"]
                                )
                                partial_positive_alignments.append(
                                    partial_stats["positive_alignment"]
                                )
                                partial_negative_positive_fractions.append(
                                    partial_stats["negative_positive_fraction"]
                                )
                            elif use_signed_expl:
                                loss_expl, sstats = signed_attribution_margin_loss(
                                    explanation, selected_masks,
                                    margin=self.signed_margin,
                                    pos_weight=self.signed_pos_weight,
                                    neg_weight=self.signed_neg_weight,
                                )
                                expl_losses.append(loss_expl.item())
                                balanced_term, balance_scale, applied_ratio = (
                                    self._balance_explanation_term(
                                        loss,
                                        self.explanation_coef * loss_expl,
                                        actor_parameters,
                                    )
                                )
                                loss = loss + balanced_term
                                applied_explanation_term_for_batch = balanced_term.detach().item()
                                explanation_balance_scales.append(balance_scale)
                                explanation_applied_grad_ratios.append(applied_ratio)
                                explanation_loss_for_diagnostics = loss_expl
                                explanation_coef_for_diagnostics = (
                                    self.explanation_coef * balance_scale
                                )

                                signed_pos_losses.append(sstats["signed_pos_loss"])
                                signed_neg_losses.append(sstats["signed_neg_loss"])
                                signed_n_pos_cells.append(sstats["n_pos_cells"])
                                signed_n_neg_cells.append(sstats["n_neg_cells"])
                                signed_mean_attr_pos.append(sstats["mean_attr_pos"])
                                signed_mean_attr_neg.append(sstats["mean_attr_neg"])
                            elif use_positive_margin_expl:
                                loss_expl, sequential_stats = positive_attribution_margin_loss(
                                    explanation,
                                    selected_masks,
                                    margin=self.sequential_margin,
                                )
                                expl_losses.append(loss_expl.item())
                                sequential_expl_losses.append(loss_expl.item())
                                sequential_mean_attrs.append(
                                    sequential_stats["mean_target_attribution"]
                                )
                                sequential_margin_rates.append(
                                    sequential_stats["margin_satisfied_rate"]
                                )
                                balanced_term, balance_scale, applied_ratio = (
                                    self._balance_explanation_term(
                                        loss,
                                        self.explanation_coef * loss_expl,
                                        actor_parameters,
                                    )
                                )
                                loss = loss + balanced_term
                                applied_explanation_term_for_batch = balanced_term.detach().item()
                                explanation_balance_scales.append(balance_scale)
                                explanation_applied_grad_ratios.append(applied_ratio)
                                explanation_loss_for_diagnostics = loss_expl
                                explanation_coef_for_diagnostics = (
                                    self.explanation_coef * balance_scale
                                )
                            else:
                                loss_expl = loss_fn(explanation, selected_masks)
                                expl_losses.append(loss_expl.item())
                                balanced_term, balance_scale, applied_ratio = (
                                    self._balance_explanation_term(
                                        loss,
                                        self.explanation_coef * loss_expl,
                                        actor_parameters,
                                    )
                                )
                                loss = loss + balanced_term
                                applied_explanation_term_for_batch = balanced_term.detach().item()
                                explanation_balance_scales.append(balance_scale)
                                explanation_applied_grad_ratios.append(applied_ratio)
                                explanation_loss_for_diagnostics = loss_expl
                                explanation_coef_for_diagnostics = (
                                    self.explanation_coef * balance_scale
                                )

                                with th.no_grad():
                                    mask_4d = selected_masks.unsqueeze(1)
                                    abs_exp  = explanation.abs()
                                    total_d  = abs_exp.sum(dim=(1, 2, 3)) + 1e-8
                                    inside_d = (explanation.clamp(min=0) * mask_4d).sum(dim=(1, 2, 3))
                                    expl_inside_ratios.append((inside_d / total_d).mean().item())
                                    mask_coverages.append(selected_masks.mean().item())
                                    expl_pos_fracs.append((explanation > 0).float().mean().item())

                                    # Diagnostics for logit/explanation collapse hypothesis
                                    total_abs_energies.append(abs_exp.sum(dim=(1, 2, 3)).mean().item())
                                    inside_energies.append(inside_d.mean().item())
                                    selected_logits = actor_logits[selected]
                                    logit_variances.append(selected_logits.var().item())
                                    mean_abs_logits.append(selected_logits.abs().mean().item())
                        else:
                            expl_losses.append(0.0)
                    else:
                        expl_losses.append(0.0)

                if use_expl and explanation_loss_for_diagnostics is not None:
                    mask_raw_for_batch = (
                        explanation_loss_for_diagnostics.detach().item()
                    )
                    mask_weighted_for_batch = applied_explanation_term_for_batch
                # ─────────────────────────────────────────────────────────────

                if use_dual_expl:
                    sequential_masks = self.rollout_buffer._current_sequential_masks
                    action_masks = self.rollout_buffer._current_action_masks
                    if sequential_masks is not None and action_masks is not None:
                        sequential_has_mask = sequential_masks.flatten(
                            start_dim=1
                        ).ne(0).any(dim=1)
                        action_has_mask = action_masks.flatten(
                            start_dim=1
                        ).ne(0).any(dim=1)
                        sequential_selected = positive_margin_supervision_selection(
                            sequential_masks, advantages
                        )
                        action_selected = explanation_supervision_selection(
                            action_masks, raw_advantages, signed=True
                        )
                        if self.dual_masks_exclusive:
                            has_negative_blocker = action_masks.flatten(
                                start_dim=1
                            ).lt(0).any(dim=1)
                            sequential_selected = (
                                sequential_selected & ~has_negative_blocker
                            )
                        sequential_mask_rates.append(
                            sequential_has_mask.float().mean().item()
                        )
                        sequential_selected_rates.append(
                            sequential_selected.float().mean().item()
                        )
                        action_mask_rates.append(action_has_mask.float().mean().item())
                        action_selected_rates.append(
                            action_selected.float().mean().item()
                        )
                        if self.sequential_explanation_coef <= 0:
                            sequential_selected = th.zeros_like(sequential_selected)
                        if self.action_explanation_coef <= 0:
                            action_selected = th.zeros_like(action_selected)

                        union_selected = sequential_selected | action_selected
                        if union_selected.any():
                            explanation = compute_actor_explanation(
                                self.policy,
                                rollout_data.observations[union_selected],
                                actions[union_selected],
                                subtract_action_mean=self.explanation_subtract_action_mean,
                            )
                            explanation_for_diagnostics = explanation
                            combined_explanation_loss = explanation.new_zeros(())

                            sequential_in_union = sequential_selected[union_selected]
                            if sequential_in_union.any():
                                loss_sequential, sequential_stats = polarity_separated_attribution_margin_loss(
                                    explanation[sequential_in_union],
                                    sequential_masks[union_selected][sequential_in_union],
                                    polarity="positive",
                                    margin=self.sequential_margin,
                                )
                                sequential_expl_losses.append(loss_sequential.item())
                                sequential_mean_attrs.append(
                                    sequential_stats["mean_target_energy"]
                                )
                                sequential_margin_rates.append(
                                    sequential_stats["margin_satisfied_rate"]
                                )
                                combined_explanation_loss = combined_explanation_loss + (
                                    self.sequential_explanation_coef * loss_sequential
                                )

                            action_in_union = action_selected[union_selected]
                            if action_in_union.any():
                                negative_masks = -action_masks[union_selected][action_in_union]
                                loss_action, sstats = polarity_separated_attribution_margin_loss(
                                    explanation[action_in_union],
                                    negative_masks,
                                    polarity="negative",
                                    margin=self.signed_margin,
                                )
                                action_expl_losses.append(loss_action.item())
                                combined_explanation_loss = combined_explanation_loss + (
                                    self.action_explanation_coef * loss_action
                                )
                                signed_pos_losses.append(0.0)
                                signed_neg_losses.append(loss_action.item())
                                signed_n_pos_cells.append(0.0)
                                signed_n_neg_cells.append(sstats["n_supervised_cells"])
                                signed_mean_attr_pos.append(0.0)
                                signed_mean_attr_neg.append(-sstats["mean_target_energy"])

                            balanced_term, balance_scale, applied_ratio = (
                                self._balance_explanation_term(
                                    loss,
                                    combined_explanation_loss,
                                    actor_parameters,
                                )
                            )
                            loss = loss + balanced_term
                            explanation_balance_scales.append(balance_scale)
                            explanation_applied_grad_ratios.append(applied_ratio)
                            combined_expl_losses.append(
                                balanced_term.detach().item()
                            )
                            explanation_loss_for_diagnostics = combined_explanation_loss
                            explanation_coef_for_diagnostics = balance_scale

                if use_expl:
                    # Keep every minibatch, including those with no selected
                    # transition, aligned with its corresponding PPO objective.
                    ppo_objective = ppo_original_losses[-1]
                    mask_raw_batch_losses.append(mask_raw_for_batch)
                    mask_weighted_batch_losses.append(mask_weighted_for_batch)
                    mask_loss_to_ppo_ratios.append(
                        abs(mask_weighted_for_batch) / (abs(ppo_objective) + 1e-8)
                    )
                    ppo_plus_mask_objectives.append(
                        ppo_objective + mask_weighted_for_batch
                    )

                # --- Logit tracking (detached, no grad) ---
                with th.no_grad():
                    al = actor_logits.detach()
                    all_logits_mean.append(al.mean().item())
                    all_logits_std.append(al.std().item())
                    all_logits_min.append(al.min().item())
                    all_logits_max.append(al.max().item())
                    tl = taken_logits.detach()
                    taken_logit_mean.append(tl.mean().item())
                    taken_logit_std.append(tl.std().item())
                    taken_logit_pos_frac.append((tl > 0).float().mean().item())

                # KL divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                diagnostics_applied = (
                    run_diagnostics
                    and explanation_loss_for_diagnostics is not None
                )
                if diagnostics_applied:
                    ppo_actor_objective = policy_loss + self.ent_coef * entropy_loss
                    ppo_grads = th.autograd.grad(
                        ppo_actor_objective,
                        actor_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    ppo_actor_grad_norms.append(self._tensor_list_norm(ppo_grads))

                    explanation_grads = th.autograd.grad(
                        explanation_coef_for_diagnostics * explanation_loss_for_diagnostics,
                        actor_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    explanation_actor_grad_norms.append(
                        self._tensor_list_norm(explanation_grads)
                    )
                    explanation_graph_attached.append(float(
                        explanation_for_diagnostics is not None
                        and explanation_for_diagnostics.requires_grad
                        and explanation_loss_for_diagnostics.requires_grad
                    ))

                    actor_parameters_before = [
                        parameter.detach().clone() for parameter in actor_parameters
                    ]
                    actor_logits_before = actor_logits.detach().clone()

                self.policy.optimizer.zero_grad()
                loss.backward()
                actor_grad_norms.append(self._tensor_list_norm([
                    parameter.grad for parameter in actor_parameters
                    if parameter.grad is not None
                ]))
                policy_grad_norm = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                policy_grad_norms.append(float(policy_grad_norm))
                self.policy.optimizer.step()

                if diagnostics_applied:
                    actor_param_max_changes.append(max(
                        (parameter.detach() - before).abs().max().item()
                        for parameter, before in zip(actor_parameters, actor_parameters_before)
                    ))
                    with th.no_grad():
                        actor_logits_after = self._actor_logits(rollout_data.observations)
                    actor_logit_max_changes.append(
                        (actor_logits_after - actor_logits_before).abs().max().item()
                    )

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if confidence_losses:
            self.logger.record("train/confidence_loss", np.mean(confidence_losses))
        self.logger.record("train/confidence_coef", self.confidence_coef)
        if confidence_selected_rates:
            self.logger.record(
                "confidence/selected_transition_rate",
                np.mean(confidence_selected_rates),
            )
        if confidence_selected_probabilities:
            self.logger.record(
                "confidence/mean_selected_action_probability",
                np.mean(confidence_selected_probabilities),
            )
        if expl_losses:
            self.logger.record("train/explanation_loss", np.mean(expl_losses))
        self.logger.record("train/explanation_coef", self.explanation_coef)
        if explanation_balance_scales:
            self.logger.record(
                "train/explanation_balance_scale",
                np.mean(explanation_balance_scales),
            )
            self.logger.record(
                "train/explanation_actor_grad_ratio",
                np.mean(explanation_applied_grad_ratios),
            )
        self.logger.record(
            "train/explanation_max_grad_ratio",
            self.explanation_max_grad_ratio,
        )
        if use_consistency_expl or use_teacher_consistency_expl:
            consistency_prefix = (
                "teacher_explanation_consistency"
                if use_teacher_consistency_expl
                else "explanation_consistency"
            )
            self.logger.record(
                f"{consistency_prefix}/selected_transition_rate",
                np.mean(consistency_selected_rates)
                if consistency_selected_rates
                else 0.0,
            )
            self.logger.record(
                f"{consistency_prefix}/mean_map_l1",
                np.mean(consistency_map_l1s) if consistency_map_l1s else 0.0,
            )
            self.logger.record(
                f"{consistency_prefix}/max_map_l1",
                np.mean(consistency_max_map_l1s)
                if consistency_max_map_l1s
                else 0.0,
            )
        if use_partial_energy_expl:
            self.logger.record(
                "partial_energy/alignment_margin",
                self.partial_energy_alignment_margin,
            )
            self.logger.record(
                "partial_energy/masked_energy_ratio",
                np.mean(partial_energy_ratios) if partial_energy_ratios else 0.0,
            )
            self.logger.record(
                "partial_energy/positive_alignment",
                np.mean(partial_positive_alignments)
                if partial_positive_alignments else 0.0,
            )
            self.logger.record(
                "partial_energy/negative_positive_fraction",
                np.mean(partial_negative_positive_fractions)
                if partial_negative_positive_fractions else 0.0,
            )
            self.logger.record(
                "partial_energy/positive_mask_rate",
                np.mean(partial_positive_mask_rates)
                if partial_positive_mask_rates else 0.0,
            )
            self.logger.record(
                "partial_energy/negative_mask_rate",
                np.mean(partial_negative_mask_rates)
                if partial_negative_mask_rates else 0.0,
            )
            self.logger.record(
                "partial_energy/selected_transition_rate",
                np.mean(partial_selected_rates) if partial_selected_rates else 0.0,
            )
        if use_absolute_energy_expl:
            self.logger.record(
                "absolute_energy/target_energy_ratio",
                np.mean(absolute_energy_ratios) if absolute_energy_ratios else 0.0,
            )
            self.logger.record(
                "absolute_energy/outside_energy_ratio",
                1.0 - np.mean(absolute_energy_ratios)
                if absolute_energy_ratios
                else 0.0,
            )
            self.logger.record(
                "absolute_energy/mask_coverage",
                np.mean(absolute_mask_coverages)
                if absolute_mask_coverages
                else 0.0,
            )
            self.logger.record(
                "absolute_energy/selected_transition_rate",
                np.mean(absolute_selected_rates)
                if absolute_selected_rates
                else 0.0,
            )
        if (
            use_masked_absolute_expl
            or use_channelwise_masked_absolute_expl
            or use_channelwise_polarity_masked_expl
            or use_patch_absolute_expl
        ):
            masked_log_prefix = (
                "patch_absolute_energy"
                if use_patch_absolute_expl
                else (
                    "channelwise_polarity_masked_energy"
                    if use_channelwise_polarity_masked_expl
                    else (
                        "channelwise_masked_absolute_energy"
                        if use_channelwise_masked_absolute_expl
                        else "masked_absolute_energy"
                    )
                )
            )
            self.logger.record(
                f"{masked_log_prefix}/target_energy_ratio",
                np.mean(masked_absolute_ratios) if masked_absolute_ratios else 0.0,
            )
            self.logger.record(
                f"{masked_log_prefix}/negative_energy_ratio",
                np.mean(masked_absolute_negative_ratios)
                if masked_absolute_negative_ratios
                else 0.0,
            )
            self.logger.record(
                f"{masked_log_prefix}/selected_transition_rate",
                np.mean(masked_absolute_selected_rates)
                if masked_absolute_selected_rates
                else 0.0,
            )
        if use_expl:
            self.logger.record(
                "train/weighted_explanation_loss",
                np.mean(mask_weighted_batch_losses)
                if mask_weighted_batch_losses else 0.0,
            )
            self.logger.record(
                "loss_comparison/ppo_objective",
                np.mean(ppo_original_losses),
            )
            self.logger.record(
                "loss_comparison/mask_raw",
                np.mean(mask_raw_batch_losses)
                if mask_raw_batch_losses else 0.0,
            )
            self.logger.record(
                "loss_comparison/mask_weighted",
                np.mean(mask_weighted_batch_losses)
                if mask_weighted_batch_losses else 0.0,
            )
            self.logger.record(
                "loss_comparison/mask_to_ppo_abs_ratio",
                np.mean(mask_loss_to_ppo_ratios)
                if mask_loss_to_ppo_ratios else 0.0,
            )
            self.logger.record(
                "loss_comparison/ppo_plus_mask_objective",
                np.mean(ppo_plus_mask_objectives)
                if ppo_plus_mask_objectives else np.mean(ppo_original_losses),
            )
        use_single_sequential_expl = use_expl and use_positive_margin_expl
        if self.sequential_explanation_coef > 0 or use_single_sequential_expl:
            self.logger.record(
                "train/sequential_explanation_loss",
                np.mean(sequential_expl_losses) if sequential_expl_losses else 0.0,
            )
        if self.action_explanation_coef > 0:
            self.logger.record(
                "train/action_explanation_loss",
                np.mean(action_expl_losses) if action_expl_losses else 0.0,
            )
        if use_dual_expl:
            self.logger.record(
                "train/combined_explanation_loss",
                np.mean(combined_expl_losses) if combined_expl_losses else 0.0,
            )
        effective_sequential_coef = (
            self.explanation_coef
            if use_single_sequential_expl
            else self.sequential_explanation_coef
        )
        self.logger.record(
            "train/sequential_explanation_coef", effective_sequential_coef
        )
        self.logger.record("train/sequential_margin", self.sequential_margin)
        if self.sequential_explanation_coef > 0 or use_single_sequential_expl:
            self.logger.record(
                "sequential/mean_target_attribution",
                np.mean(sequential_mean_attrs) if sequential_mean_attrs else 0.0,
            )
            self.logger.record(
                "sequential/margin_satisfied_rate",
                np.mean(sequential_margin_rates) if sequential_margin_rates else 0.0,
            )
            self.logger.record(
                "sequential/loss_applied",
                float(bool(sequential_expl_losses)),
            )
        if raw_positive_advantage_rates:
            self.logger.record(
                "explanation/raw_positive_advantage_rate",
                np.mean(raw_positive_advantage_rates),
            )
        if normalized_positive_advantage_rates:
            self.logger.record(
                "explanation/normalized_nonnegative_advantage_rate",
                np.mean(normalized_positive_advantage_rates),
            )
        if sequential_mask_rates:
            self.logger.record(
                "sequential/mask_available_rate", np.mean(sequential_mask_rates)
            )
            self.logger.record(
                "sequential/selected_transition_rate",
                np.mean(sequential_selected_rates),
            )
        if action_mask_rates:
            self.logger.record("action/mask_available_rate", np.mean(action_mask_rates))
            self.logger.record(
                "action/selected_transition_rate", np.mean(action_selected_rates)
            )
        self.logger.record("train/action_explanation_coef", self.action_explanation_coef)
        self.logger.record(
            "diagnostics/actor_parameters_missing_optimizer",
            actor_parameters_missing_optimizer,
        )
        if actor_grad_norms:
            self.logger.record("diagnostics/actor_grad_norm", np.mean(actor_grad_norms))
            self.logger.record("diagnostics/policy_grad_norm", np.mean(policy_grad_norms))
        if ppo_actor_grad_norms:
            self.logger.record("diagnostics/ppo_actor_grad_norm", np.mean(ppo_actor_grad_norms))
        if explanation_actor_grad_norms:
            self.logger.record(
                "diagnostics/explanation_actor_grad_norm",
                np.mean(explanation_actor_grad_norms),
            )
        if ppo_actor_grad_norms and explanation_actor_grad_norms:
            self.logger.record(
                "diagnostics/explanation_to_ppo_actor_grad_ratio",
                np.mean(explanation_actor_grad_norms)
                / (np.mean(ppo_actor_grad_norms) + 1e-12),
            )
        if actor_param_max_changes:
            self.logger.record(
                "diagnostics/actor_param_max_change",
                np.mean(actor_param_max_changes),
            )
            self.logger.record(
                "diagnostics/actor_logit_max_change",
                np.mean(actor_logit_max_changes),
            )
        if explanation_graph_attached:
            self.logger.record(
                "diagnostics/explanation_graph_attached",
                np.mean(explanation_graph_attached),
            )
        if signed_pos_losses:
            # Signed action-conditioned supervision diagnostics.
            signed_totals = action_expl_losses if action_expl_losses else expl_losses
            self.logger.record("signed/loss_total", np.mean(signed_totals))
            self.logger.record("signed/loss_pos_region", np.mean(signed_pos_losses))
            self.logger.record("signed/loss_neg_region", np.mean(signed_neg_losses))
            self.logger.record("signed/n_pos_cells", np.mean(signed_n_pos_cells))
            self.logger.record("signed/n_neg_cells", np.mean(signed_n_neg_cells))
            self.logger.record("signed/mean_attr_pos", np.mean(signed_mean_attr_pos))
            self.logger.record("signed/mean_attr_neg", np.mean(signed_mean_attr_neg))
        if expl_inside_ratios:
            self.logger.record("train/expl_inside_ratio", np.mean(expl_inside_ratios))
            self.logger.record("train/mask_coverage", np.mean(mask_coverages))
            self.logger.record("train/expl_pos_frac", np.mean(expl_pos_fracs))
        if total_abs_energies:
            self.logger.record("explanation/total_abs_energy", np.mean(total_abs_energies))
            self.logger.record("explanation/inside_energy", np.mean(inside_energies))
            self.logger.record("explanation/logit_variance", np.mean(logit_variances))
            self.logger.record("explanation/mean_absolute_logit", np.mean(mean_abs_logits))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/logits_mean", np.mean(all_logits_mean))
        self.logger.record("train/logits_std", np.mean(all_logits_std))
        self.logger.record("train/logits_min", np.mean(all_logits_min))
        self.logger.record("train/logits_max", np.mean(all_logits_max))
        self.logger.record("train/taken_logit_mean", np.mean(taken_logit_mean))
        self.logger.record("train/taken_logit_std", np.mean(taken_logit_std))
        self.logger.record("train/taken_logit_pos_frac", np.mean(taken_logit_pos_frac))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/loss_ppo_original", np.mean(ppo_original_losses))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

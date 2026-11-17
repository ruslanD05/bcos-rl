from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import torch.nn.functional as F
import numpy as np

from bcos_rl_generic.explanation_supervision import (
    MaskedReplayBuffer,
    advantage_conditioned_partial_energy_loss,
    compute_q_explanation,
    partial_masked_energy_loss,
)


class AuxAlphaRampCallback(BaseCallback):
    def __init__(
        self,
        start_alpha: float = 0.0,
        end_alpha: float = 0.1,
        total_steps: int = 100_000,
        schedule: str = "linear",  # "linear", "exponential", or "constant"
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if schedule not in ("linear", "exponential", "constant"):
            raise ValueError(f"schedule must be 'linear', 'exponential', or 'constant', got '{schedule}'")
        if schedule == "exponential" and start_alpha <= 0.0:
            raise ValueError("start_alpha must be > 0 for exponential schedule")
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.total_steps = total_steps
        self.schedule = schedule

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_steps)
        if self.schedule == "constant":
            new_alpha = self.end_alpha
        elif self.schedule == "exponential":
            new_alpha = self.start_alpha * (self.end_alpha / self.start_alpha) ** progress
        else:
            new_alpha = self.start_alpha + (self.end_alpha - self.start_alpha) * progress
        self.model.aux_alpha = new_alpha
        self.logger.record("train/aux_alpha", new_alpha)
        return True

class BcosDQN(DQN):
    def __init__(self, *args, aux_alpha: float = 0.1, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_alpha = aux_alpha
        self.temperature = temperature

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses, losses_td, losses_bce = [], [], []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # --- Standard SB3 TD Loss Calculation ---
            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values_gathered = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            
            loss_td = F.smooth_l1_loss(current_q_values_gathered, target_q_values)

            # --- Auxiliary BCE Loss: confidence that each action has positive advantage ---
            with th.no_grad():
                target_q_all = self.q_net_target(replay_data.observations)          # (B, n_actions)
                v_target = target_q_all.mean(dim=1, keepdim=True)                   # (B, 1) mean-Q baseline
                advantages = target_q_all - v_target                               # (B, n_actions)
                # Soft target in (0, 1): how confidently positive is each action's advantage
                soft_targets = th.sigmoid(advantages / self.temperature)            # (B, n_actions)

            # Train sigmoid(Q_online / T) to match soft_targets
            loss_bce = F.binary_cross_entropy_with_logits(
                current_q_values / self.temperature, soft_targets
            )

            # Combine
            loss = loss_td + (self.aux_alpha * loss_bce)
            losses.append(loss.item())
            losses_td.append(loss_td.item())
            losses_bce.append(loss_bce.item())

            # --- Standard SB3 Optimization Step ---
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/loss_td", np.mean(losses_td))
        self.logger.record("train/loss_bce", np.mean(losses_bce))


class ExplanationBcosDQN(DQN):
    """Original DQN TD objective plus spatial B-cos explanation supervision."""

    def __init__(
        self,
        *args,
        explanation_coef: float = 0.15,
        explanation_max_grad_ratio: float = 0.1,
        partial_energy_alignment_margin: float = 0.5,
        min_replay_transitions: int = 10_000,
        explanation_advantage_mode: str = "all",
        explanation_advantage_threshold: float = 0.0,
        explanation_update_frequency: int = 1,
        explanation_batch_size: int = 0,
        explanation_sign_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if explanation_coef <= 0:
            raise ValueError("explanation_coef must be positive")
        if explanation_max_grad_ratio <= 0:
            raise ValueError("explanation_max_grad_ratio must be positive")
        if not 0 <= partial_energy_alignment_margin <= 1:
            raise ValueError("partial_energy_alignment_margin must be in [0, 1]")
        if min_replay_transitions < 0:
            raise ValueError("min_replay_transitions must be non-negative")
        if explanation_advantage_mode not in (
            "all",
            "positive_only",
            "flip_positive",
        ):
            raise ValueError(
                "explanation_advantage_mode must be all, positive_only, "
                "or flip_positive"
            )
        if explanation_advantage_threshold < 0:
            raise ValueError("explanation_advantage_threshold must be non-negative")
        if explanation_update_frequency <= 0:
            raise ValueError("explanation_update_frequency must be positive")
        if explanation_batch_size < 0:
            raise ValueError("explanation_batch_size must be non-negative")
        if explanation_sign_weight < 0:
            raise ValueError("explanation_sign_weight must be non-negative")
        self.explanation_coef = explanation_coef
        self.explanation_max_grad_ratio = explanation_max_grad_ratio
        self.partial_energy_alignment_margin = partial_energy_alignment_margin
        self.min_replay_transitions = min_replay_transitions
        self.explanation_advantage_mode = explanation_advantage_mode
        self.explanation_advantage_threshold = explanation_advantage_threshold
        self.explanation_update_frequency = explanation_update_frequency
        self.explanation_batch_size = explanation_batch_size
        self.explanation_sign_weight = explanation_sign_weight

    @staticmethod
    def _gradient_norm(grads) -> th.Tensor:
        terms = [grad.detach().square().sum() for grad in grads if grad is not None]
        if not terms:
            return th.zeros(())
        return th.stack(terms).sum().sqrt()

    def _balance_explanation_term(
        self,
        td_loss: th.Tensor,
        weighted_explanation: th.Tensor,
        parameters: list[th.nn.Parameter],
    ) -> tuple[th.Tensor, float, float]:
        td_grads = th.autograd.grad(
            td_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        explanation_grads = th.autograd.grad(
            weighted_explanation,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        td_norm = self._gradient_norm(td_grads).to(self.device)
        explanation_norm = self._gradient_norm(explanation_grads).to(self.device)
        if not th.isfinite(explanation_norm) or explanation_norm.item() == 0.0:
            return weighted_explanation * 0.0, 0.0, 0.0
        cap = (
            self.explanation_max_grad_ratio
            * td_norm
            / (explanation_norm + 1e-12)
        )
        scale = min(1.0, float(cap.item()))
        applied_ratio = float(
            (scale * explanation_norm / (td_norm + 1e-12)).item()
        )
        return weighted_explanation * scale, scale, applied_ratio

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if not isinstance(self.replay_buffer, MaskedReplayBuffer):
            raise RuntimeError(
                "ExplanationBcosDQN requires MaskedReplayBuffer"
            )
        replay_transitions = self.replay_buffer.size() * self.n_envs
        self.logger.record("train/replay_transitions", replay_transitions)
        if replay_transitions < self.min_replay_transitions:
            self.logger.record("train/explanation_warmup", 1.0)
            return

        self.logger.record("train/explanation_warmup", 0.0)
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        total_losses, td_losses, explanation_losses = [], [], []
        weighted_losses, balance_scales, applied_grad_ratios = [], [], []
        masked_energy_ratios, positive_alignments = [], []
        negative_positive_fractions, selected_rates = [], []
        desired_sign_alignments, sign_margin_rates = [], []
        relative_advantage_positive_rates = []
        explanation_update_rates = []
        q_parameters = [
            parameter
            for parameter in self.q_net.parameters()
            if parameter.requires_grad
        ]

        for local_update in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            with th.no_grad():
                next_q_values = self.q_net_target(
                    replay_data.next_observations
                )
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                discounts = (
                    replay_data.discounts
                    if replay_data.discounts is not None
                    else self.gamma
                )
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            actions = replay_data.actions.long().flatten()
            selected_q_values = current_q_values.gather(
                1,
                actions.unsqueeze(1),
            )
            td_loss = F.smooth_l1_loss(
                selected_q_values,
                target_q_values,
            )

            update_index = self._n_updates + local_update
            apply_explanation = (
                update_index % self.explanation_update_frequency == 0
            )
            explanation_update_rates.append(float(apply_explanation))
            masks = replay_data.masks
            selected = masks.flatten(start_dim=1).ne(0).any(dim=1)
            relative_advantage = None
            if apply_explanation and self.explanation_advantage_mode != "all":
                with th.no_grad():
                    target_current_q = self.q_net_target(
                        replay_data.observations
                    )
                    relative_advantage = target_current_q.gather(
                        1,
                        actions.unsqueeze(1),
                    ).squeeze(1) - target_current_q.mean(dim=1)
                relative_advantage_positive_rates.append(
                    (relative_advantage >= 0).float().mean().item()
                )
                if self.explanation_advantage_mode == "positive_only":
                    selected = selected & (
                        relative_advantage
                        >= self.explanation_advantage_threshold
                    )
                else:
                    selected = selected & (
                        relative_advantage.abs()
                        >= self.explanation_advantage_threshold
                    )
            if not apply_explanation:
                selected = th.zeros_like(selected)
            if apply_explanation:
                selected_rates.append(selected.float().mean().item())
            selected_indices = selected.nonzero(as_tuple=False).flatten()
            if (
                self.explanation_batch_size > 0
                and len(selected_indices) > self.explanation_batch_size
            ):
                order = th.randperm(
                    len(selected_indices),
                    device=selected_indices.device,
                )[: self.explanation_batch_size]
                selected_indices = selected_indices[order]
            if len(selected_indices) > 0:
                explanation = compute_q_explanation(
                    self.policy,
                    replay_data.observations[selected_indices],
                    actions[selected_indices],
                )
                if self.explanation_advantage_mode == "flip_positive":
                    assert relative_advantage is not None
                    polarity = th.where(
                        relative_advantage[selected_indices] >= 0,
                        1.0,
                        -1.0,
                    )
                    explanation_loss, stats = (
                        advantage_conditioned_partial_energy_loss(
                            explanation,
                            masks[selected_indices],
                            polarity,
                            alignment_margin=(
                                self.partial_energy_alignment_margin
                            ),
                            sign_weight=self.explanation_sign_weight,
                        )
                    )
                    desired_sign_alignments.append(
                        stats["desired_sign_alignment"]
                    )
                    sign_margin_rates.append(
                        stats["sign_margin_satisfied_rate"]
                    )
                else:
                    explanation_loss, stats = partial_masked_energy_loss(
                        explanation,
                        masks[selected_indices],
                        alignment_margin=(
                            self.partial_energy_alignment_margin
                        ),
                    )
                weighted_explanation = (
                    self.explanation_coef * explanation_loss
                )
                balanced_explanation, balance_scale, applied_ratio = (
                    self._balance_explanation_term(
                        td_loss,
                        weighted_explanation,
                        q_parameters,
                    )
                )
                masked_energy_ratios.append(stats["masked_energy_ratio"])
                positive_alignments.append(stats["positive_alignment"])
                if "negative_positive_fraction" in stats:
                    negative_positive_fractions.append(
                        stats["negative_positive_fraction"]
                    )
            else:
                explanation_loss = td_loss.detach() * 0.0
                balanced_explanation = td_loss * 0.0
                balance_scale = 0.0
                applied_ratio = 0.0

            loss = td_loss + balanced_explanation
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.q_net.parameters(),
                self.max_grad_norm,
            )
            self.policy.optimizer.step()

            total_losses.append(loss.item())
            td_losses.append(td_loss.item())
            explanation_losses.append(explanation_loss.item())
            weighted_losses.append(balanced_explanation.item())
            balance_scales.append(balance_scale)
            applied_grad_ratios.append(applied_ratio)

        self._n_updates += gradient_steps
        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )
        self.logger.record("train/loss", np.mean(total_losses))
        self.logger.record("train/loss_td", np.mean(td_losses))
        self.logger.record(
            "train/explanation_loss",
            np.mean(explanation_losses),
        )
        self.logger.record(
            "train/weighted_explanation_loss",
            np.mean(weighted_losses),
        )
        self.logger.record("train/explanation_coef", self.explanation_coef)
        self.logger.record(
            "train/explanation_balance_scale",
            np.mean(balance_scales),
        )
        self.logger.record(
            "train/explanation_actor_grad_ratio",
            np.mean(applied_grad_ratios),
        )
        self.logger.record(
            "explanation/selected_transition_rate",
            np.mean(selected_rates) if selected_rates else 0.0,
        )
        self.logger.record(
            "explanation/update_rate",
            np.mean(explanation_update_rates),
        )
        self.logger.record(
            "explanation/relative_advantage_positive_rate",
            (
                np.mean(relative_advantage_positive_rates)
                if relative_advantage_positive_rates
                else 0.0
            ),
        )
        self.logger.record(
            "explanation/desired_sign_alignment",
            (
                np.mean(desired_sign_alignments)
                if desired_sign_alignments
                else 0.0
            ),
        )
        self.logger.record(
            "explanation/sign_margin_satisfied_rate",
            np.mean(sign_margin_rates) if sign_margin_rates else 0.0,
        )
        self.logger.record(
            "partial_energy/masked_energy_ratio",
            np.mean(masked_energy_ratios) if masked_energy_ratios else 0.0,
        )
        self.logger.record(
            "partial_energy/positive_alignment",
            np.mean(positive_alignments) if positive_alignments else 0.0,
        )
        self.logger.record(
            "partial_energy/negative_positive_fraction",
            (
                np.mean(negative_positive_fractions)
                if negative_positive_fractions
                else 0.0
            ),
        )

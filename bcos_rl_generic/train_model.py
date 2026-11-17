# shared_lib.py
from __future__ import annotations
from functools import partial
from typing import Any
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import ObservationWrapper, spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from bcos.modules import BcosLinear
from bcos.modules.common import BcosSequential, BcosUtilMixin, DetachableModule
from bcos.modules import BcosConv2d
from bcos.modules.norms import (
    BatchNormUncentered2d,
    GroupNormUncentered2d,
    NoBias,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


def parse_layer_sizes(layer_spec: str) -> list[int]:
    cleaned = layer_spec.strip()
    if cleaned == "": return []
    return [int(x.strip()) for x in cleaned.split(",") if x.strip()]

def validate_layer_sizes(layer_sizes: list[int], *, arg_name: str) -> None:
    if any(size <= 0 for size in layer_sizes):
        raise ValueError(f"{arg_name} must contain only positive integers, got: {layer_sizes}")

def activation_from_name(name: str) -> type[nn.Module]:
    table: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported activation '{name}'. Choose one of: {', '.join(table)}")
    return table[key]

class SaveVecNormalizeOnBest(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def _on_event(self) -> bool:
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            if self.verbose > 0:
                print(self.save_path)
            vec_env.save(self.save_path)
        return True
    
class ConfigurableQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        actor_type: str = "bcos",
        bcos_b: float = 2.0,
        bcos_max_out: int = 1,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )
        
        action_dim = int(self.action_space.n)
        modules: list[nn.Module] = []
        prev_dim = self.features_dim
        
        for hidden_dim in self.net_arch:
            if actor_type == "bcos":
                modules.append(BcosLinear(prev_dim, hidden_dim, b=bcos_b, max_out=bcos_max_out))
            else:
                modules.append(nn.Linear(prev_dim, hidden_dim))
                modules.append(activation_fn())
            prev_dim = hidden_dim
            
        if actor_type == "bcos":
            modules.append(BcosLinear(prev_dim, action_dim, b=bcos_b, max_out=bcos_max_out))
            self.q_net = BcosSequential(*modules)
        else:
            modules.append(nn.Linear(prev_dim, action_dim))
            self.q_net = nn.Sequential(*modules)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.q_net(features)


class ConfigurableDQNPolicy(BcosUtilMixin, DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        lr_schedule: th.optim.lr_scheduler._LRScheduler,
        actor_type: str = "bcos",
        bcos_b: float = 2.0,
        bcos_max_out: int = 1,
        **kwargs: Any,
    ) -> None:
        self.actor_type = actor_type
        self.bcos_b = bcos_b
        self.bcos_max_out = bcos_max_out
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_q_net(self) -> ConfigurableQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return ConfigurableQNetwork(
            **net_args,
            actor_type=self.actor_type,
            bcos_b=self.bcos_b,
            bcos_max_out=self.bcos_max_out,
        ).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update({
            "actor_type": self.actor_type,
            "bcos_b": self.bcos_b,
            "bcos_max_out": self.bcos_max_out
        })
        return data
class ConfigurableActorCriticExtractor(nn.Module):
    latent_dim_pi: int
    latent_dim_vf: int

    def __init__(
        self, feature_dim: int, net_arch: list[int] | dict[str, list[int]], 
        activation_fn: type[nn.Module], *, actor_type: str = "bcos", critic_type: str = "mlp",
        bcos_b: float = 2.0, bcos_max_out: int = 1,
        stable_bcos_norm: bool = False, device: th.device | str = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        pi_layers = net_arch.get("pi", []) if isinstance(net_arch, dict) else net_arch
        vf_layers = net_arch.get("vf", []) if isinstance(net_arch, dict) else net_arch

        if actor_type == "bcos":
            self.policy_net, self.latent_dim_pi = self._build_bcos_mlp(
                feature_dim, pi_layers, bcos_b, bcos_max_out, stable_bcos_norm
            )
        elif actor_type == "mlp":
            self.policy_net, self.latent_dim_pi = self._build_standard_mlp(
                feature_dim, pi_layers, activation_fn
            )
        else:
            raise ValueError(f"Unsupported actor_type '{actor_type}'")

        if critic_type == "bcos":
            self.value_net, self.latent_dim_vf = self._build_bcos_mlp(
                feature_dim, vf_layers, bcos_b, bcos_max_out, stable_bcos_norm
            )
        elif critic_type == "mlp":
            self.value_net, self.latent_dim_vf = self._build_standard_mlp(
                feature_dim, vf_layers, activation_fn
            )
        else:
            raise ValueError(f"Unsupported critic_type '{critic_type}'")
        self.to(device)

    @staticmethod
    def _build_bcos_mlp(
        input_dim: int,
        layers: list[int],
        bcos_b: float,
        bcos_max_out: int,
        stable_bcos_norm: bool,
    ) -> tuple[nn.Module, int]:
        if not layers: return nn.Identity(), input_dim
        modules: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in layers:
            modules.append(BcosLinear(prev_dim, hidden_dim, b=bcos_b, max_out=bcos_max_out))
            if stable_bcos_norm:
                modules.append(RMSNormUncentered1d(hidden_dim))
            prev_dim = hidden_dim
        return BcosSequential(*modules), prev_dim

    @staticmethod
    def _build_standard_mlp(input_dim: int, layers: list[int], activation_fn: type[nn.Module]) -> tuple[nn.Module, int]:
        if not layers: return nn.Identity(), input_dim
        modules: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in layers:
            modules.append(nn.Linear(prev_dim, hidden_dim))
            modules.append(activation_fn())
            prev_dim = hidden_dim
        return nn.Sequential(*modules), prev_dim

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class ConfigurableActorCriticPolicy(BcosUtilMixin, ActorCriticPolicy):
    def __init__(
        self, *args: Any, actor_type: str = "bcos", critic_type: str = "mlp",
        bcos_b: float = 2.0, bcos_max_out: int = 1,
        stable_bcos_norm: bool = False,
        critic_features_extractor_class: type[BaseFeaturesExtractor] | None = None,
        critic_features_extractor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.actor_type = actor_type
        self.critic_type = critic_type
        self.bcos_b = bcos_b
        self.bcos_max_out = bcos_max_out
        self.stable_bcos_norm = stable_bcos_norm
        self.critic_features_extractor_class = critic_features_extractor_class
        self.critic_features_extractor_kwargs = critic_features_extractor_kwargs or {}
        self._building_initial_feature_extractors = True
        self._feature_extractor_build_count = 0
        super().__init__(*args, **kwargs)
        self._building_initial_feature_extractors = False
        self._feature_extractor_build_count = 0

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Build a separately configurable critic extractor when requested."""
        use_critic_extractor = (
            self._building_initial_feature_extractors
            and self._feature_extractor_build_count == 1
            and self.critic_features_extractor_class is not None
        )
        if use_critic_extractor:
            extractor = self.critic_features_extractor_class(
                self.observation_space,
                **self.critic_features_extractor_kwargs,
            )
            if extractor.features_dim != self.features_dim:
                raise ValueError(
                    "Actor and critic feature extractors must have the same "
                    f"features_dim, got {self.features_dim} and {extractor.features_dim}"
                )
        else:
            extractor = super().make_features_extractor()
        self._feature_extractor_build_count += 1
        return extractor

    def _build(self, lr_schedule: th.optim.lr_scheduler._LRScheduler) -> None:
        super()._build(lr_schedule)
        if self.critic_type == "bcos":
            self.value_net = BcosLinear(
                self.mlp_extractor.latent_dim_vf, 1, 
                b=self.bcos_b, max_out=self.bcos_max_out,
            ).to(self.device)
            if self.ortho_init:
                self.value_net.apply(partial(self.init_weights, gain=1.0))

            # ActorCriticPolicy._build() creates its optimizer before this
            # custom B-cos value head exists. Recreate it so the new head is
            # trainable. For an MLP critic, retain SB3's original initialized
            # value head and optimizer unchanged.
            self.optimizer = self.optimizer_class(
                self.parameters(),
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

    """
    def _evaluate_full(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        q_values = self.value_net(latent_vf)                          # [B, n_actions]
        action_probs = distribution.distribution.probs                # [B, n_actions]
        v_s = (action_probs * q_values).sum(dim=-1, keepdim=True)    # [B, 1]

        adv_pred_logits = self.advantage_pred_net(latent_pi)          # [B, n_actions]

        return v_s, log_prob, entropy, q_values, adv_pred_logits

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        q_values = self.value_net(latent_vf)                          # [B, n_actions]
        action_probs = distribution.distribution.probs                # [B, n_actions]
        v_s = (action_probs * q_values).sum(dim=-1, keepdim=True)    # [B, 1]

        return actions, v_s, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        
        v_s, log_prob, entropy, _, _ = self._evaluate_full(obs, actions)
        return v_s, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        vf_features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        q_values = self.value_net(latent_vf)                        # [B, n_actions]

        pi_features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        action_probs = F.softmax(self.action_net(latent_pi), dim=-1)  # [B, n_actions]

        return (action_probs * q_values).sum(dim=-1, keepdim=True)   # [B, 1]

    """

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ConfigurableActorCriticExtractor(
            feature_dim=self.features_dim, net_arch=self.net_arch,
            activation_fn=self.activation_fn, actor_type=self.actor_type, critic_type=self.critic_type,
            bcos_b=self.bcos_b, bcos_max_out=self.bcos_max_out,
            stable_bcos_norm=self.stable_bcos_norm, device=self.device,
        )
        if self.actor_type == "bcos" and isinstance(self.action_dist, CategoricalDistribution):
            def bcos_proba_net(latent_dim: int) -> BcosLinear:
                return BcosLinear(
                    latent_dim, 
                    self.action_dist.action_dim,
                    b=self.bcos_b, 
                    max_out=self.bcos_max_out,
                )
            self.action_dist.proba_distribution_net = bcos_proba_net

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update({
            "actor_type": self.actor_type,
            "critic_type": self.critic_type,
            "bcos_b": self.bcos_b,
            "bcos_max_out": self.bcos_max_out,
            "stable_bcos_norm": self.stable_bcos_norm,
            "critic_features_extractor_class": self.critic_features_extractor_class,
            "critic_features_extractor_kwargs": self.critic_features_extractor_kwargs,
        })
        return data

class BcosInputTransform(nn.Module):
    """Internal complement encoding marker used by explanation extraction."""

    is_bcos_input_transform = True

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = th.clamp(x, 0.0, 1.0)
        return th.cat([x, 1.0 - x], dim=1)


class RMSNormUncentered1d(DetachableModule):
    """Per-sample RMS normalization that remains linear in explanation mode."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(th.ones(num_features))
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        stats_input = x.detach() if self.detach else x
        rms = stats_input.square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class GenericCNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        *,
        use_bcos: bool = False,
        bcos_b: float = 2.0,
        bcos_max_out: int = 1,
        invert_channels: bool = False,
        stable_bcos_norm: bool = False,
        bias_free_bcos_norm: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        self.use_bcos = use_bcos
        self.invert_channels = invert_channels
        shape = observation_space.shape
        n_input_channels = 1 if len(shape) == 2 else shape[0]
        h, w = shape if len(shape) == 2 else shape[1:]

        # Define configurations based on input size
        is_small = h < 32 or w < 32
        in_ch = n_input_channels * 2 if (use_bcos and invert_channels) else n_input_channels
        cfg = [(in_ch, 32, 4, 2, 1), (32, 64, 4, 2, 1)] if is_small else [(in_ch, 32, 8, 4, 0), (32, 64, 4, 2, 0), (64, 64, 3, 1, 0)]

        # Dynamic factory setup to eliminate structure duplication
        if use_bcos:
            seq_cls = BcosSequential
            layers = [BcosInputTransform()] if invert_channels else []
            def make_block(c_in, c_out, k, s, p):
                norm_factory = (
                    GroupNormUncentered2d
                    if stable_bcos_norm
                    else BatchNormUncentered2d
                )
                if bias_free_bcos_norm:
                    norm_factory = NoBias(norm_factory)
                norm = (
                    norm_factory(num_groups=8, num_channels=c_out)
                    if stable_bcos_norm
                    else norm_factory(c_out)
                )
                return [
                    BcosConv2d(c_in, c_out, k, stride=s, padding=p, b=bcos_b),
                    norm,
                ]
        else:
            seq_cls = nn.Sequential
            layers = []
            def make_block(c_in, c_out, k, s, p):
                return [nn.Conv2d(c_in, c_out, k, stride=s, padding=p), nn.ReLU()]

        # Assemble CNN backbone
        for params in cfg:
            layers.extend(make_block(*params))
        layers.append(nn.Flatten())
        self.cnn = seq_cls(*layers)

        # Infer flatten dimensions without updating BatchNorm running stats.
        # PPO collects rollouts in eval mode, so those statistics must not be
        # initialized from this arbitrary observation-space sample.
        cnn_was_training = self.cnn.training
        self.cnn.eval()
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample = sample.unsqueeze(1) if sample.ndim == 3 else sample
            n_flatten = self.cnn(sample).shape[1]
        self.cnn.train(cnn_was_training)

        # Assemble linear head
        self.linear = (
            BcosSequential(
                BcosLinear(n_flatten, features_dim, b=bcos_b, max_out=bcos_max_out),
                RMSNormUncentered1d(features_dim),
            )
            if use_bcos and stable_bcos_norm
            else BcosLinear(n_flatten, features_dim, b=bcos_b, max_out=bcos_max_out)
            if use_bcos
            else nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float()
        if x.ndim == 3:  # Dynamically identifies a batched 2D flat environment (B, H, W)
            x = x.unsqueeze(1)
        return self.linear(self.cnn(x))


class SymbolicGridCNNExtractor(GenericCNNExtractor):
    """Spatial CNN for channels-last one-hot MiniGrid observations."""

    channels_last_input = True

    def __init__(self, observation_space: spaces.Box, *args: Any, **kwargs: Any) -> None:
        if len(observation_space.shape) != 3:
            raise ValueError(
                "SymbolicGridCNNExtractor expects an (H, W, C) observation, "
                f"got {observation_space.shape}"
            )
        chw_space = spaces.Box(
            low=np.moveaxis(observation_space.low, -1, 0),
            high=np.moveaxis(observation_space.high, -1, 0),
            dtype=observation_space.dtype,
        )
        super().__init__(chw_space, *args, **kwargs)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.ndim != 4:
            raise ValueError(
                "SymbolicGridCNNExtractor expects batched (B, H, W, C) input, "
                f"got {tuple(observations.shape)}"
            )
        return super().forward(observations.permute(0, 3, 1, 2))

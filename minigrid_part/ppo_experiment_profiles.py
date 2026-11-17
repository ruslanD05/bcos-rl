"""Frozen PPO hyperparameter profiles for comparable MiniGrid studies."""
from __future__ import annotations

from typing import Any


CURRENT = "current"
RQ1_RQ3_STELLAR = "rq1-rq3-stellar"


_PROFILES: dict[str, dict[str, Any]] = {
    CURRENT: {},
    # Stable profile used by the RQ1 architecture comparison and the RQ3
    # source checkpoints. Environment-specific settings stay in each sweep.
    RQ1_RQ3_STELLAR: {
        "learning_rate": 0.00022569760177299956,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.995,
        "gae_lambda": 0.90,
        "clip_range": 0.1,
        "target_kl": 0.05,
        "ent_coef": 0.0003215278796080832,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "features_dim": 256,
        "pi_layers": "128,128",
        "vf_layers": "256,256",
        "bcos_b": 2.3,
        "stable_bcos_norm": True,
        "bias_free_bcos_norm": True,
    },
}


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)


def apply_profile(name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* updated with the named frozen profile."""
    if name not in _PROFILES:
        raise ValueError(f"Unknown PPO profile: {name!r}")
    resolved = dict(config)
    resolved.update(_PROFILES[name])
    resolved["ppo_profile"] = name
    return resolved

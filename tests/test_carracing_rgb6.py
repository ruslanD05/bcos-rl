"""CarRacing actor encoding, standard-control, and checkpoint regressions."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn

from bcos.modules import BcosConv2d
from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.train_model import ConfigurableActorCriticPolicy, RMSNormUncentered1d
from car_racing_part import sweep_rq1_carracing_architecture as sweep
from car_racing_part.train_carracing_ppo import build_env, make_args, make_policy_kwargs


def policy_args(variant="bcos", complement=True):
    return SimpleNamespace(
        policy_type=variant, features_dim=256, pi_layers="256,256",
        vf_layers="256,256", activation="tanh", bcos_b=2.3,
        bcos_max_out=1, stable_bcos_norm=False, bcos_complement_input=complement,
    )


def make_model(env, variant="bcos", complement=True):
    cls = BcosPPO if variant == "bcos" else PPO
    aux = dict(confidence_coef=0.0, explanation_coef=0.0) if variant == "bcos" else {}
    return cls(
        ConfigurableActorCriticPolicy, env, n_steps=8, batch_size=8, n_epochs=1,
        policy_kwargs=make_policy_kwargs(policy_args(variant, complement)),
        seed=17, device="cpu", verbose=0, **aux,
    )


class CarRacingRGB6Tests(unittest.TestCase):
    def test_cli_and_rgb3_rq1_configuration(self):
        with patch.object(sys, "argv", ["train", "--policy-type", "bcos"]):
            self.assertTrue(make_args().bcos_complement_input)
        with patch.object(sys, "argv", ["train", "--policy-type", "bcos", "--no-bcos-complement-input"]):
            self.assertFalse(make_args().bcos_complement_input)
        params = sweep.sweep_config()["parameters"]
        self.assertNotIn("bcos_complement_input", params)
        self.assertNotIn("stable_bcos_norm", params)
        cfg = vars(policy_args()) | {
            "variant": "bcos",
            "stable_bcos_norm": True,
            "bcos_complement_input": False,
        }
        kwargs = sweep.policy_kwargs(cfg)["features_extractor_kwargs"]
        self.assertFalse(kwargs["invert_channels"])
        self.assertTrue(kwargs["stable_bcos_norm"])

    def test_standard_control_is_unchanged(self):
        env = build_env(env_id="CarRacing-v3", n_envs=1, seed=17, use_subprocesses=False)
        try:
            old = make_model(env, "standard", False)
            new = make_model(env, "standard", True)
            self.assertEqual(old.policy_kwargs, new.policy_kwargs)
            for key, tensor in old.policy.state_dict().items():
                torch.testing.assert_close(tensor, new.policy.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(new.policy.pi_features_extractor.cnn[0].in_channels, 3)
        finally:
            env.close()

    def test_rgb6_preprocessing_update_and_checkpoint(self):
        env = build_env(env_id="CarRacing-v3", n_envs=1, seed=17, use_subprocesses=False)
        try:
            model = make_model(env)
            actor = model.policy.pi_features_extractor
            conv = next(m for m in actor.modules() if isinstance(m, BcosConv2d))
            captured = []
            hook = conv.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach().clone()))
            # Include exact endpoints and intermediate values to detect double scaling.
            obs = np.tile(np.arange(96, dtype=np.uint8)[None, None, None, :], (1, 3, 96, 1))
            obs[:, :, :, -1] = 255
            model.policy.set_training_mode(False)
            model.predict(obs, deterministic=True)
            hook.remove()
            rgb = torch.as_tensor(obs, dtype=torch.float32) / 255
            torch.testing.assert_close(captured[0], torch.cat([rgb, 1-rgb], dim=1))
            self.assertEqual(tuple(captured[0].shape), (1, 6, 96, 96))
            self.assertEqual(model.policy.vf_features_extractor.cnn[0].in_channels, 3)
            self.assertEqual(model.observation_space.shape, (3, 96, 96))
            self.assertFalse(any(isinstance(m, RMSNormUncentered1d) for m in model.policy.modules()))
            norms = [m for m in actor.modules() if isinstance(m, nn.BatchNorm2d)]
            self.assertEqual(len(norms), 3)
            self.assertTrue(all(m.bias is None for m in norms))
            model.learn(total_timesteps=16)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            model.policy.set_training_mode(False)
            tensor_obs = torch.as_tensor(obs)
            with torch.no_grad():
                expected = model.policy.get_distribution(tensor_obs).distribution.probs
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "rgb6.zip"
                model.save(path)
                loaded = BcosPPO.load(path, env=env, device="cpu")
                loaded.policy.set_training_mode(False)
                with torch.no_grad():
                    actual = loaded.policy.get_distribution(tensor_obs).distribution.probs
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(loaded.policy.pi_features_extractor.invert_channels)
                legacy = make_model(env, complement=False)
                legacy.save(Path(folder) / "rgb3.zip")
                restored = BcosPPO.load(Path(folder) / "rgb3.zip", env=env, device="cpu")
                self.assertFalse(restored.policy.pi_features_extractor.invert_channels)
                restored.predict(obs, deterministic=True)
        finally:
            env.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

#!/usr/bin/env python3
"""Interactively inspect exact CarRacing observations and actor explanations."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common.save_util import load_from_zip_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bcos_rl_generic.bcosPPO import BcosPPO
from bcos_rl_generic.explanation_supervision import compute_actor_explanation
from car_racing_part.explanation_masks import (
    build_carracing_explanation_mask,
    CarRacingGrassColorWrapper,
    CarRacingMaskConfig,
    GRASS_COLORS,
)
from car_racing_part.train_carracing_ppo import ACTION_NAMES, build_env


@dataclass
class EpisodeFrame:
    observation: np.ndarray
    action: int
    action_probabilities: np.ndarray
    value: float
    reward: float
    cumulative_reward: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument(
        "--initial-frame",
        type=int,
        default=0,
        help="Zero-based frame initially displayed after recording the episode.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use greedy actions; pass --no-deterministic to sample the policy.",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Add a panel showing the proposed P/N/D CarRacing training mask.",
    )
    parser.add_argument(
        "--mask-grass-as-negative",
        action="store_true",
        help="Mark distant grass as N instead of leaving it neutral.",
    )
    parser.add_argument(
        "--mask-profile",
        choices=("boundary", "boundary-dashboard", "road-positive"),
        default="boundary",
        help="Use the existing boundary mask or P=car+road, N=grass.",
    )
    parser.add_argument(
        "--grass-color",
        choices=tuple(GRASS_COLORS),
        default="normal",
        help="Visually recolor grass while leaving CarRacing dynamics unchanged.",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=None,
        help="Save the first visualization frame to this image path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window (useful with --save-frame).",
    )
    return parser.parse_args()


def load_model(model_path: Path, env, device: str):
    data, _, _ = load_from_zip_file(model_path, device=device)
    policy_kwargs = data.get("policy_kwargs", {}) if data else {}
    actor_type = policy_kwargs.get("actor_type", "unknown")
    if actor_type != "bcos":
        raise ValueError(
            f"This viewer requires a B-cos actor, but checkpoint actor_type={actor_type!r}"
        )
    return BcosPPO.load(model_path, env=env, device=device)


def policy_outputs(model, observation: np.ndarray) -> tuple[np.ndarray, float]:
    obs_tensor = th.as_tensor(observation, device=model.device)
    with th.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probabilities = distribution.distribution.probs[0]
        value = model.policy.predict_values(obs_tensor).flatten()[0]
    return probabilities.cpu().numpy(), float(value.cpu())


def record_episode(model, env, *, deterministic: bool, max_steps: int) -> list[EpisodeFrame]:
    observation = env.reset()
    frames: list[EpisodeFrame] = []
    cumulative_reward = 0.0

    for _ in range(max_steps):
        action_array, _ = model.predict(observation, deterministic=deterministic)
        action = int(np.asarray(action_array).reshape(-1)[0])
        probabilities, value = policy_outputs(model, observation)
        next_observation, rewards, dones, _infos = env.step(
            np.asarray([action], dtype=np.int64)
        )
        reward = float(rewards[0])
        cumulative_reward += reward
        frames.append(
            EpisodeFrame(
                observation=np.asarray(observation[0]).copy(),
                action=action,
                action_probabilities=probabilities.copy(),
                value=value,
                reward=reward,
                cumulative_reward=cumulative_reward,
            )
        )
        observation = next_observation
        if bool(dones[0]):
            break

    return frames


class CarRacingExplanationViewer:
    def __init__(
        self,
        model,
        frames: list[EpisodeFrame],
        *,
        deterministic: bool,
        show_mask: bool,
        initial_frame: int,
        mask_grass_as_negative: bool,
        mask_profile: str,
    ) -> None:
        self.model = model
        self.frames = frames
        self.deterministic = deterministic
        self.index = min(max(initial_frame, 0), len(frames) - 1)
        self.playing = False
        self.show_mask = show_mask
        self.mask_config = CarRacingMaskConfig(
            profile=mask_profile,
            grass_as_negative=mask_grass_as_negative
        )
        self.explanation_cache: dict[int, np.ndarray] = {}

        panel_count = 3 if show_mask else 2
        self.figure, axes = plt.subplots(
            1, panel_count, figsize=(6.3 * panel_count, 6.5), constrained_layout=True
        )
        self.obs_axis = axes[0]
        self.explanation_axis = axes[1]
        self.mask_axis = axes[2] if show_mask else None
        self.timer = self.figure.canvas.new_timer(interval=120)
        self.timer.add_callback(self._advance)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw()

    def _explanation(self, index: int) -> np.ndarray:
        if index not in self.explanation_cache:
            frame = self.frames[index]
            observation = th.as_tensor(
                frame.observation[None], device=self.model.device
            )
            action = th.as_tensor([frame.action], device=self.model.device)
            explanation = compute_actor_explanation(
                self.model.policy,
                observation,
                action,
                create_graph=False,
            )
            spatial = explanation.detach()[0].sum(dim=0).cpu().numpy()
            self.explanation_cache[index] = spatial
        return self.explanation_cache[index]

    def draw(self) -> None:
        frame = self.frames[self.index]
        rgb = np.moveaxis(frame.observation, 0, -1)
        spatial = self._explanation(self.index)
        scale = max(float(np.percentile(np.abs(spatial), 99.0)), 1e-12)

        self.obs_axis.clear()
        self.explanation_axis.clear()
        self.obs_axis.imshow(rgb)
        self.obs_axis.set_title("Exact RGB observation supplied to the model")
        self.obs_axis.axis("off")

        self.explanation_axis.imshow(rgb)
        self.explanation_axis.imshow(
            spatial,
            cmap="seismic",
            vmin=-scale,
            vmax=scale,
            alpha=0.58,
            interpolation="nearest",
        )
        self.explanation_axis.set_title(
            "Signed B-cos attribution: red supports, blue suppresses\n"
            f"relative {ACTION_NAMES[frame.action]} logit; scale={scale:.3g}"
        )
        self.explanation_axis.axis("off")

        if self.mask_axis is not None:
            mask = build_carracing_explanation_mask(
                frame.observation,
                episode_step=self.index,
                config=self.mask_config,
            )
            mask_overlay = rgb.astype(np.float32) / 255.0
            positive = mask > 0
            negative = mask < 0
            mask_overlay[positive] = (
                0.25 * mask_overlay[positive] + 0.75 * np.array([0.0, 1.0, 0.0])
            )
            mask_overlay[negative] = (
                0.25 * mask_overlay[negative] + 0.75 * np.array([1.0, 0.0, 0.0])
            )
            self.mask_axis.clear()
            self.mask_axis.imshow(mask_overlay)
            self.mask_axis.set_title(
                "Proposed training mask: green=P, red=N, unchanged=D\n"
                f"P={positive.mean():.1%}, N={negative.mean():.1%}"
            )
            self.mask_axis.axis("off")

        probabilities = ", ".join(
            f"{name}={probability:.2f}"
            for name, probability in zip(ACTION_NAMES, frame.action_probabilities)
        )
        mode = "deterministic" if self.deterministic else "stochastic"
        self.figure.suptitle(
            f"Frame {self.index + 1}/{len(self.frames)} | {mode} action: "
            f"{ACTION_NAMES[frame.action]} (p={frame.action_probabilities[frame.action]:.3f}) "
            f"| value={frame.value:.2f} | reward={frame.reward:+.2f} "
            f"| return={frame.cumulative_reward:.2f}\n{probabilities}\n"
            "A/D or arrows: frame | Home/End: boundary | Space: play/pause | Q: close",
            fontsize=10,
        )
        self.figure.canvas.draw_idle()

    def _advance(self) -> None:
        if not self.playing:
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.timer.stop()
            return
        self.index += 1
        self.draw()

    def _on_key(self, event) -> None:
        if event.key in ("right", "d"):
            self.index = min(self.index + 1, len(self.frames) - 1)
        elif event.key in ("left", "a"):
            self.index = max(self.index - 1, 0)
        elif event.key == "home":
            self.index = 0
        elif event.key == "end":
            self.index = len(self.frames) - 1
        elif event.key in (" ", "space"):
            self.playing = not self.playing
            if self.playing:
                self.timer.start()
            else:
                self.timer.stop()
            return
        elif event.key in ("q", "escape"):
            plt.close(self.figure)
            return
        else:
            return
        self.draw()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    env = build_env(
        env_id="CarRacing-v3",
        n_envs=1,
        seed=args.seed,
        use_subprocesses=False,
        wrapper_class=partial(
            CarRacingGrassColorWrapper,
            color_name=args.grass_color,
        ),
    )
    try:
        model = load_model(args.model_path, env, args.device)
        if model.observation_space.shape != (3, 96, 96):
            raise ValueError(
                "Expected a single RGB observation with shape (3, 96, 96), got "
                f"{model.observation_space.shape}"
            )
        frames = record_episode(
            model,
            env,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    if not frames:
        raise RuntimeError("The environment produced no frames")
    print(
        f"Recorded {len(frames)} frames, return={frames[-1].cumulative_reward:.2f}, "
        f"mode={'deterministic' if args.deterministic else 'stochastic'}"
    )
    viewer = CarRacingExplanationViewer(
        model,
        frames,
        deterministic=args.deterministic,
        show_mask=args.show_mask,
        initial_frame=args.initial_frame,
        mask_grass_as_negative=args.mask_grass_as_negative,
        mask_profile=args.mask_profile,
    )
    if args.save_frame is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        viewer.figure.savefig(args.save_frame, dpi=160)
        print(f"Saved visualization: {args.save_frame.resolve()}")
    if args.no_show:
        plt.close(viewer.figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()

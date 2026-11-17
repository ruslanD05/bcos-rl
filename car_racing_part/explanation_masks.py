"""Visual explanation masks for RGB CarRacing observations."""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


GRASS_COLORS: dict[str, tuple[int, int, int] | None] = {
    "normal": None,
    "purple": (150, 35, 200),
    "yellow": (225, 210, 25),
    "blue": (35, 95, 220),
}


@dataclass(frozen=True)
class CarRacingMaskConfig:
    """Configuration for a CarRacing visual explanation target mask."""

    profile: str = "boundary"
    warmup_frames: int = 50
    boundary_radius: int = 3
    road_channel_tolerance: int = 14
    road_min_value: int = 65
    road_max_value: int = 150
    grass_green_margin: int = 35
    grass_min_green: int = 145
    grass_as_negative: bool = False

    def __post_init__(self) -> None:
        if self.profile not in {"boundary", "boundary-dashboard", "road-positive"}:
            raise ValueError(
                "Mask profile must be 'boundary', 'boundary-dashboard', or "
                f"'road-positive', got "
                f"{self.profile!r}"
            )


def _as_hwc_rgb(observation: np.ndarray) -> np.ndarray:
    image = np.asarray(observation)
    if image.shape == (3, 96, 96):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image.astype(np.uint8, copy=False)


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary.copy()
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    result = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def carracing_visual_regions(
    observation: np.ndarray,
    *,
    config: CarRacingMaskConfig = CarRacingMaskConfig(),
) -> dict[str, np.ndarray]:
    """Segment semantically useful regions from an exact RGB observation."""
    image = _as_hwc_rgb(observation)
    height, width, _ = image.shape
    rgb = image.astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    dashboard_start = int(round(height * 0.875))
    scene = np.zeros((height, width), dtype=bool)
    scene[:dashboard_start] = True

    channel_spread = np.maximum.reduce(
        [np.abs(red - green), np.abs(red - blue), np.abs(green - blue)]
    )
    intensity = (red + green + blue) / 3.0
    road = (
        scene
        & (channel_spread <= config.road_channel_tolerance)
        & (intensity >= config.road_min_value)
        & (intensity <= config.road_max_value)
    )
    grass = (
        scene
        & (green >= config.grass_min_green)
        & ((green - red) >= config.grass_green_margin)
        & ((green - blue) >= config.grass_green_margin)
    )
    # Preserve semantic grass masks when inspecting one of our exact visual
    # interventions after the recoloring has already been applied.
    for color in GRASS_COLORS.values():
        if color is not None:
            grass |= scene & np.all(image == np.asarray(color, dtype=np.uint8), axis=-1)

    # Keep only a narrow geometry-bearing band instead of rewarding all grass.
    road_side = road & _dilate(grass, config.boundary_radius)
    grass_side = grass & _dilate(road, config.boundary_radius)
    boundary = _dilate(road_side | grass_side, 1) & scene

    # The follow camera keeps the car near the lower center. Restricting the
    # red threshold to this ROI prevents red/white track curbs becoming "car".
    yy, xx = np.indices((height, width))
    car_roi = (
        (xx >= int(width * 0.34))
        & (xx < int(width * 0.66))
        & (yy >= int(height * 0.53))
        & (yy < dashboard_start)
    )
    car = car_roi & (red >= 145) & (green <= 90) & (blue <= 90)

    # Identify visible indicators separately. The boundary-dashboard profile
    # below intentionally targets the complete telemetry strip, including its
    # dark background, so that the desired region is visually explicit.
    active = np.maximum.reduce([red, green, blue]) >= 45
    dashboard = yy >= dashboard_start
    speed = dashboard & (xx >= int(width * 0.115)) & (xx < int(width * 0.16))
    abs_bars = dashboard & (xx >= int(width * 0.165)) & (xx < int(width * 0.29))
    steering = dashboard & (xx >= int(width * 0.30)) & (xx < int(width * 0.66))
    gyro = dashboard & (xx >= int(width * 0.60)) & (xx < int(width * 0.98))
    indicators = active & (speed | abs_bars | steering | gyro)

    if config.profile == "road-positive":
        # Test whether the policy can concentrate evidence on the vehicle and
        # drivable surface while explicitly suppressing grass dependence.
        positive = car | road
        negative = grass & ~positive
    else:
        positive = car | boundary
        if config.profile == "boundary-dashboard":
            positive |= dashboard
        else:
            positive |= indicators
        # Do not assign conflicting signs near P; all other pixels are D.
        negative_candidates = road.copy()
        if config.profile == "boundary-dashboard" or config.grass_as_negative:
            negative_candidates |= grass
        negative = negative_candidates & ~_dilate(positive, 1)
    return {
        "positive": positive,
        "negative": negative,
        "road": road,
        "grass": grass,
        "boundary": boundary,
        "car": car,
        "indicators": indicators,
    }


def build_carracing_explanation_mask(
    observation: np.ndarray,
    *,
    episode_step: int,
    config: CarRacingMaskConfig = CarRacingMaskConfig(),
) -> np.ndarray:
    """Return a float32 P/N/D mask: +1 target, -1 suppressed, 0 neutral."""
    image = _as_hwc_rgb(observation)
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    if episode_step < config.warmup_frames:
        return mask
    regions = carracing_visual_regions(image, config=config)
    mask[regions["negative"]] = -1.0
    mask[regions["positive"]] = 1.0
    return mask


def recolor_carracing_grass(
    observation: np.ndarray,
    color_name: str,
    *,
    config: CarRacingMaskConfig = CarRacingMaskConfig(),
) -> np.ndarray:
    """Recolor segmented grass without changing any environment dynamics."""
    if color_name not in GRASS_COLORS:
        raise ValueError(
            f"Unknown grass color {color_name!r}; choose from {sorted(GRASS_COLORS)}"
        )
    image = _as_hwc_rgb(observation)
    color = GRASS_COLORS[color_name]
    if color is None:
        return image.copy()
    grass = carracing_visual_regions(image, config=config)["grass"]
    recolored = image.copy()
    recolored[grass] = np.asarray(color, dtype=np.uint8)
    return recolored


class CarRacingGrassColorWrapper(gym.ObservationWrapper):
    """Apply a visual-only grass-color intervention to RGB observations."""

    def __init__(self, env: gym.Env, *, color_name: str) -> None:
        super().__init__(env)
        if color_name not in GRASS_COLORS:
            raise ValueError(
                f"Unknown grass color {color_name!r}; choose from {sorted(GRASS_COLORS)}"
            )
        self.color_name = color_name

    def observation(self, observation):
        return recolor_carracing_grass(observation, self.color_name)


class CarRacingExplanationMaskWrapper(gym.Wrapper):
    """Attach the pre-action CarRacing mask to ``info['attn_mask']``."""

    def __init__(
        self,
        env: gym.Env,
        *,
        config: CarRacingMaskConfig = CarRacingMaskConfig(),
    ) -> None:
        super().__init__(env)
        self.config = config
        self._observation: np.ndarray | None = None
        self._episode_step = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._observation = np.asarray(observation).copy()
        self._episode_step = 0
        info["attn_mask"] = np.zeros(
            self._observation.shape[:2], dtype=np.float32
        )
        return observation, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        mask = build_carracing_explanation_mask(
            self._observation,
            episode_step=self._episode_step,
            config=self.config,
        )
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["attn_mask"] = mask
        self._observation = np.asarray(observation).copy()
        self._episode_step += 1
        return observation, reward, terminated, truncated, info

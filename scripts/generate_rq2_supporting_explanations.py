#!/usr/bin/env python3
"""Generate qualitative explanation figures for supporting environments (LavaCrossing and CarRacing)."""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
import minigrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "minigrid_part"))

from bcos_rl_generic.bcosPPO import BcosPPO
from minigrid_part.env_wrappers import make_wrapper
from minigrid_part.evaluate_explanation_faithfulness import explanations
from minigrid_part.evaluate_explanation_metrics import load_model
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

METHODS = ("bcos", "saliency", "integrated_gradients", "smoothgrad")
ACTION_NAMES = {0: "left", 1: "right", 2: "forward"}

def style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def overlay(axis, rgb: np.ndarray, attribution: np.ndarray | None, title: str):
    if attribution is None:
        axis.imshow(rgb)
    else:
        scale = max(float(np.percentile(attribution, 99)), 1e-12)
        normalized = np.clip(attribution / scale, 0.0, 1.0)
        axis.imshow(rgb)
        axis.imshow(
            normalized,
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=np.clip(normalized * 0.85, 0.0, 0.85),
        )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])

def make_lava_env():
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="rgb_array", highlight=False)
    wrapper_cls = make_wrapper(use_rgb_img_obs=True, invert_rgb_channels=True)
    return wrapper_cls(env)

def generate_lavacrossing_figure(output_dir: Path):
    vec_env = VecTransposeImage(DummyVecEnv([make_lava_env]))
    bcos_model = BcosPPO.load("artifacts/lava_nolmfy7e/final_model.zip", env=vec_env, device="cpu")
    std_model = PPO.load("artifacts/lava_standard/final_model.zip", env=vec_env, device="cpu")
    
    # Run episode to collect frames
    obs = vec_env.reset()
    frames = []
    done = False
    step = 0
    while not done and step < 50:
        action, _ = bcos_model.predict(obs, deterministic=True)
        frames.append((obs[0].copy(), int(action[0])))
        obs, reward, done, _ = vec_env.step(action)
        step += 1
        if done[0]:
            break
            
    # Pick 3 key frames: start (0), navigating turn (6), final approach (9)
    indices = [0, 6, min(9, len(frames)-1)]
    selected_obs = np.stack([frames[i][0] for i in indices])
    selected_actions = np.asarray([frames[i][1] for i in indices], dtype=np.int64)
    
    models = {
        "Standard PPO": std_model,
        "B-cos PPO": bcos_model,
    }
    
    maps = {}
    for label, model in models.items():
        arch_methods = ("saliency", "integrated_gradients", "smoothgrad") if label.startswith("Standard") else METHODS
        maps[label] = {
            m: explanations(
                model,
                selected_obs,
                selected_actions,
                batch_size=len(indices),
                method=m,
                integrated_gradients_steps=32,
                smoothgrad_samples=8,
            )
            for m in arch_methods
        }
        
    columns = ("Observation", "Intrinsic B-cos", "Saliency", "Integrated Gradients", "SmoothGrad")
    
    for idx_in_sub, frame_idx in enumerate(indices):
        obs_raw = selected_obs[idx_in_sub]
        action = selected_actions[idx_in_sub]
        rgb = np.moveaxis(obs_raw[:3], 0, -1).astype(np.float32) / 255.0
        
        fig, axes = plt.subplots(2, 5, figsize=(10.4, 4.25))
        for row, (model_label, model) in enumerate(models.items()):
            overlay(axes[row, 0], rgb, None, columns[0] if row == 0 else "")
            axes[row, 0].set_ylabel(model_label)
            for col, method in enumerate(METHODS, start=1):
                if method not in maps[model_label]:
                    axes[row, col].axis("off")
                    axes[row, col].text(0.5, 0.5, "Not available", ha="center", va="center")
                    if row == 0:
                        axes[row, col].set_title(columns[col])
                    continue
                attr = np.abs(maps[model_label][method][idx_in_sub].sum(axis=0))
                overlay(axes[row, col], rgb, attr, columns[col] if row == 0 else "")
        action_str = ACTION_NAMES.get(int(action), str(action))
        fig.suptitle(f"LavaCrossing decision: {action_str} (step {frame_idx})")
        fig.tight_layout()
        stem = f"rq2_lavacrossing_step{frame_idx}_{action_str}"
        fig.savefig(output_dir / f"{stem}.pdf")
        fig.savefig(output_dir / f"{stem}.png")
        plt.close(fig)
        print(f"Saved {stem}")

if __name__ == "__main__":
    style()
    out = ROOT / "overleaf_source_v2_6chapters" / "Thesis_Template_DE_EN" / "BA_MA_English" / "sections" / "images"
    out.mkdir(parents=True, exist_ok=True)
    generate_lavacrossing_figure(out)
    print("All supporting explanations generated successfully.")

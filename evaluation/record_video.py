"""Record a deterministic evaluation rollout to video, for the challenge's
required deliverable ("a video of your policy walking, or attempting to").

Uses MuJoCo's offscreen renderer (no on-screen viewer needed, works
headless). Kept separate from evaluation/evaluate.py's metrics run so
video rendering cost never affects the metrics protocol.

Usage:
    python -m evaluation.record_video --model experiments/runs/baseline0001/final_model.zip \
        --seed 1000 --out experiments/runs/baseline0001/rollout_seed1000.mp4
"""

from __future__ import annotations

import argparse

import imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = G1WalkEnv(cfg)
    model = PPO.load(args.model, device=cfg.ppo.device)

    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, camera)
    camera.distance = 3.0
    camera.azimuth = 140
    camera.elevation = -20

    obs, info = env.reset(seed=args.seed)
    print(f"command: {info['command']}")

    frames = []
    render_every = max(1, round(1 / (cfg.sim.control_dt * 30)))  # ~30fps output
    max_steps = int(args.max_seconds / cfg.sim.control_dt)

    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % render_every == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(renderer.render().copy())
        step += 1

    print(f"episode: {step} steps ({step * cfg.sim.control_dt:.2f}s), "
          f"terminated={terminated}, truncated={truncated}, reason={info.get('termination_reason')}")

    fps = round(1 / (cfg.sim.control_dt * render_every))
    imageio.mimsave(args.out, frames, fps=fps)
    print(f"wrote {args.out} ({len(frames)} frames @ {fps}fps)")

    env.close()


if __name__ == "__main__":
    main()

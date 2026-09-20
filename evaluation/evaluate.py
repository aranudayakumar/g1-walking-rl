"""Deterministic evaluation, kept separate from (stochastic) training per
CLAUDE.md. Runs a fixed set of seeds through the policy with
`deterministic=True` action selection and reports the metrics named in
CLAUDE.md's Evaluation section.

Usage:
    python -m evaluation.evaluate --model experiments/runs/smoke0001/final_model.zip --episodes 10
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv
from envs.observations import quat_to_yaw

EVAL_SEEDS = list(range(1000, 1000 + 20))  # fixed, disjoint from training seeds


def run_episode(model: PPO, env: G1WalkEnv, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    command = info["command"]
    pelvis_xy_start = env.data.xpos[env.pelvis_id][:2].copy()
    yaw_start = quat_to_yaw(env.data.xquat[env.pelvis_id])
    terminated = truncated = False
    total_reward = 0.0
    steps = 0
    vel_errors = []
    torque_costs = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        vel_errors.append(info.get("raw_vel_error", np.nan))
        torque_costs.append(-info.get("torque_penalty", 0.0))

    episode_length_s = steps * env.cfg.sim.control_dt
    yaw_end = quat_to_yaw(env.data.xquat[env.pelvis_id])
    # Wrap to (-180, 180] so a policy that happens to complete more than a
    # half-turn doesn't report a misleadingly small drift.
    yaw_drift_deg = float(np.degrees(np.arctan2(np.sin(yaw_end - yaw_start), np.cos(yaw_end - yaw_start))))
    pelvis_xy_end = env.data.xpos[env.pelvis_id][:2].copy()
    net_displacement_xy = pelvis_xy_end - pelvis_xy_start
    expected_displacement_xy = command * episode_length_s
    # "Survives without falling" is not the same as "actually walks" -- a
    # policy could in principle stand nearly still and still avoid the
    # termination conditions (see docs/FAILURE_ANALYSIS.md F-0002, which
    # this metric was added specifically to catch). Report distance
    # actually covered, not just survival, so that gap is visible.
    distance_traveled_m = float(np.linalg.norm(net_displacement_xy))
    expected_distance_m = float(np.linalg.norm(expected_displacement_xy))
    # distance_ratio is only meaningful once the episode has run long enough
    # for "expected distance" to be a stable denominator -- a short, failed
    # episode can produce a large or even >1 ratio from a few steps of
    # lurching before falling (observed in EXP-0008: 2.4s mean episode length
    # produced a spurious 1.91 ratio that looked better than a real walker's
    # 0.57). Flag short episodes so aggregates can exclude them explicitly
    # rather than silently averaging in a misleading number.
    reliable_ratio = episode_length_s >= 0.5 * env.cfg.sim.episode_seconds

    return {
        "seed": seed,
        "command": command.tolist(),
        "episode_length_s": episode_length_s,
        "episode_steps": steps,
        "total_reward": total_reward,
        "mean_vel_error": float(np.nanmean(vel_errors)) if vel_errors else float("nan"),
        "mean_torque_cost": float(np.mean(torque_costs)) if torque_costs else 0.0,
        "termination_reason": info.get("termination_reason"),
        "final_pelvis_height": info.get("pelvis_height"),
        "distance_traveled_m": distance_traveled_m,
        "expected_distance_m": expected_distance_m,
        "distance_ratio": distance_traveled_m / expected_distance_m if expected_distance_m > 1e-6 else float("nan"),
        "distance_ratio_reliable": reliable_ratio,
        "yaw_drift_deg": yaw_drift_deg,
        "yaw_drift_per_10s_deg": yaw_drift_deg / episode_length_s * 10.0 if episode_length_s > 1e-6 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = G1WalkEnv(cfg)
    model = PPO.load(args.model, device=cfg.ppo.device)

    seeds = EVAL_SEEDS[: args.episodes]
    results = [run_episode(model, env, seed) for seed in seeds]

    lengths = [r["episode_steps"] for r in results]
    rewards = [r["total_reward"] for r in results]
    vel_errors = [r["mean_vel_error"] for r in results]
    # Only average distance_ratio over episodes long enough for it to be a
    # meaningful number (see run_episode's "distance_ratio_reliable" note --
    # a handful of steps before an early fall can produce a spuriously large
    # ratio that isn't comparable to a full-episode walker's ratio).
    reliable_ratios = [
        r["distance_ratio"] for r in results if r["distance_ratio_reliable"] and np.isfinite(r["distance_ratio"])
    ]
    reasons = {}
    for r in results:
        reasons[r["termination_reason"]] = reasons.get(r["termination_reason"], 0) + 1

    summary = {
        "n_episodes": len(results),
        "mean_episode_length_steps": float(np.mean(lengths)),
        "mean_episode_length_s": float(np.mean(lengths)) * cfg.sim.control_dt,
        "mean_total_reward": float(np.mean(rewards)),
        "mean_vel_error": float(np.mean(vel_errors)),
        "mean_distance_traveled_m": float(np.mean([r["distance_traveled_m"] for r in results])),
        "mean_distance_ratio_reliable_only": float(np.mean(reliable_ratios)) if reliable_ratios else float("nan"),
        "n_reliable_ratio_episodes": len(reliable_ratios),
        # Mean absolute yaw drift per 10s -- normalized so episodes of
        # different lengths (e.g. a fall at 5s vs a full 20s) are
        # comparable (docs/FAILURE_ANALYSIS.md F-0003).
        "mean_abs_yaw_drift_per_10s_deg": float(np.mean([abs(r["yaw_drift_per_10s_deg"]) for r in results])),
        "termination_reasons": reasons,
    }

    print(json.dumps({"summary": summary, "episodes": results}, indent=2))
    env.close()


if __name__ == "__main__":
    main()

"""Quantitative gait diagnostic: does the policy actually take steps (feet
lift off the ground, alternating stance/swing) and walk in a straight
line (heading doesn't drift), or is it sliding/dragging and/or curving?

Built in response to a user report that the walk doesn't look like real
stepping / doesn't go straight -- measuring the mechanism before guessing
a reward fix (CLAUDE.md failure-analysis discipline: observation before
interpretation).

Usage:
    python -m diagnostics.gait_diagnostic --model experiments/runs/exp0009_longer_baseline/best_checkpoint_6M.zip --vx 0.3 --vy 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

from config.loader import load_config
from envs.contacts import LEFT_FOOT_GEOMS, RIGHT_FOOT_GEOMS, foot_in_contact
from envs.g1_walk_env import G1WalkEnv
from envs.observations import quat_to_yaw

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--vx", type=float, default=None, help="override commanded forward velocity")
    parser.add_argument("--vy", type=float, default=None, help="override commanded lateral velocity")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", default=None, help="output PNG path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = G1WalkEnv(cfg)
    model = PPO.load(args.model, device=cfg.ppo.device)

    floor_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_foot_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    obs, info = env.reset(seed=args.seed)
    if args.vx is not None or args.vy is not None:
        cmd = env._command.copy()
        if args.vx is not None:
            cmd[0] = args.vx
        if args.vy is not None:
            cmd[1] = args.vy
        env._command = cmd
    print(f"command: {env._command}")

    n_steps = int(args.seconds / cfg.sim.control_dt)
    t = np.zeros(n_steps)
    base_xy = np.zeros((n_steps, 2))
    base_yaw = np.zeros(n_steps)
    left_z = np.zeros(n_steps)
    right_z = np.zeros(n_steps)
    left_contact = np.zeros(n_steps, dtype=bool)
    right_contact = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        t[i] = i * cfg.sim.control_dt
        base_xy[i] = env.data.xpos[env.pelvis_id][:2]
        base_yaw[i] = quat_to_yaw(env.data.xquat[env.pelvis_id])
        left_z[i] = env.data.xpos[left_foot_body][2]
        right_z[i] = env.data.xpos[right_foot_body][2]
        left_contact[i] = foot_in_contact(env.data, LEFT_FOOT_GEOMS, floor_geom)
        right_contact[i] = foot_in_contact(env.data, RIGHT_FOOT_GEOMS, floor_geom)
        if terminated or truncated:
            t, base_xy, base_yaw = t[: i + 1], base_xy[: i + 1], base_yaw[: i + 1]
            left_z, right_z = left_z[: i + 1], right_z[: i + 1]
            left_contact, right_contact = left_contact[: i + 1], right_contact[: i + 1]
            print(f"episode ended at step {i} ({t[-1]:.2f}s): terminated={terminated} reason={info.get('termination_reason')}")
            break
    else:
        print(f"ran full {args.seconds}s without termination")

    # --- Metrics ---
    yaw_drift_deg = np.degrees(base_yaw[-1] - base_yaw[0])
    # Lateral deviation from the straight line defined by initial heading.
    heading0 = base_yaw[0]
    forward_dir = np.array([np.cos(heading0), np.sin(heading0)])
    lateral_dir = np.array([-np.sin(heading0), np.cos(heading0)])
    rel = base_xy - base_xy[0]
    forward_dist = rel @ forward_dir
    lateral_dist = rel @ lateral_dir
    max_lateral_deviation = float(np.max(np.abs(lateral_dist)))
    net_forward = float(forward_dist[-1])

    # Step counting: a "step" = a contact->no-contact->contact cycle (foot lifts and lands).
    def count_liftoffs(contact: np.ndarray) -> int:
        return int(np.sum((contact[:-1] == True) & (contact[1:] == False)))

    left_steps = count_liftoffs(left_contact)
    right_steps = count_liftoffs(right_contact)
    both_airborne_frac = float(np.mean(~left_contact & ~right_contact))
    both_grounded_frac = float(np.mean(left_contact & right_contact))
    left_airtime_frac = float(np.mean(~left_contact))
    right_airtime_frac = float(np.mean(~right_contact))

    print(f"\n--- Straightness ---")
    print(f"yaw drift: {yaw_drift_deg:.1f} deg over {t[-1]:.1f}s")
    print(f"net forward progress: {net_forward:.2f}m, max lateral deviation from initial heading: {max_lateral_deviation:.2f}m")

    print(f"\n--- Stepping ---")
    print(f"left foot: {left_steps} liftoffs, airborne {left_airtime_frac*100:.0f}% of time, z range [{left_z.min():.3f}, {left_z.max():.3f}]")
    print(f"right foot: {right_steps} liftoffs, airborne {right_airtime_frac*100:.0f}% of time, z range [{right_z.min():.3f}, {right_z.max():.3f}]")
    print(f"both feet grounded: {both_grounded_frac*100:.0f}% of time, both airborne: {both_airborne_frac*100:.0f}% of time")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    axes[0].plot(base_xy[:, 0], base_xy[:, 1], "-o", markersize=2)
    axes[0].plot(base_xy[0, 0], base_xy[0, 1], "go", label="start")
    axes[0].plot(base_xy[-1, 0], base_xy[-1, 1], "ro", label="end")
    axes[0].set_xlabel("world x (m)")
    axes[0].set_ylabel("world y (m)")
    axes[0].set_title(f"Base trajectory (top-down) -- yaw drift {yaw_drift_deg:.1f} deg")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, left_z, label="left foot z")
    axes[1].plot(t, right_z, label="right foot z")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("foot height (m)")
    axes[1].set_title("Foot height over time (peaks = steps, flat = dragging)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(t, 0, left_contact.astype(float), step="post", alpha=0.5, label="left contact")
    axes[2].fill_between(t, 1, 1 + right_contact.astype(float), step="post", alpha=0.5, label="right contact")
    axes[2].set_xlabel("time (s)")
    axes[2].set_yticks([0.5, 1.5])
    axes[2].set_yticklabels(["left", "right"])
    axes[2].set_title("Foot ground-contact state (gap = swing phase)")
    axes[2].legend()

    fig.tight_layout()
    out_path = args.out or str(REPO_ROOT / "diagnostics" / "reports" / "gait_diagnostic.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")
    env.close()


if __name__ == "__main__":
    main()

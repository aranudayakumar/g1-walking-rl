"""Evaluate every checkpoint saved during one run (plus its final model),
to see a trend over training budget rather than a single endpoint --
built for EXP-0009 (does more training alone help?).

Usage: python -m evaluation.evaluate_checkpoints --run exp0009_longer_baseline
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config.loader import load_config
from envs.g1_walk_env import G1WalkEnv
from evaluation.evaluate import EVAL_SEEDS, run_episode

REPO_ROOT = Path(__file__).resolve().parent.parent


def _steps_from_checkpoint_name(path: Path) -> int:
    m = re.search(r"_(\d+)_steps", path.stem)
    return int(m.group(1)) if m else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "experiments" / "runs" / args.run
    cfg = load_config(args.config)
    env = G1WalkEnv(cfg)
    seeds = EVAL_SEEDS[: args.episodes]

    checkpoints = sorted(
        (run_dir / "checkpoints").glob("*.zip"), key=_steps_from_checkpoint_name
    )
    final = run_dir / "final_model.zip"
    targets = [(_steps_from_checkpoint_name(p), p) for p in checkpoints]
    if final.exists():
        meta = json.loads((run_dir / "run_meta.json").read_text())
        targets.append((meta["timesteps"], final))

    results_by_step = {}
    for steps, path in targets:
        model = PPO.load(str(path), device=cfg.ppo.device)
        episodes = [run_episode(model, env, seed) for seed in seeds]
        lengths_s = [e["episode_length_s"] for e in episodes]
        reliable = [e["distance_ratio"] for e in episodes if e["distance_ratio_reliable"]]
        falls = sum(1 for e in episodes if e["termination_reason"] is not None)
        mean_yaw_drift = float(np.mean([abs(e["yaw_drift_per_10s_deg"]) for e in episodes]))

        results_by_step[steps] = {
            "mean_len_s": float(np.mean(lengths_s)),
            "mean_dist_ratio_reliable": float(np.mean(reliable)) if reliable else float("nan"),
            "n_reliable": len(reliable),
            "falls": falls,
            "n_episodes": len(episodes),
            "mean_abs_yaw_drift_per_10s_deg": mean_yaw_drift,
        }
        print(
            f"{steps:>10} steps: len={results_by_step[steps]['mean_len_s']:.1f}s  "
            f"dist_ratio={results_by_step[steps]['mean_dist_ratio_reliable']:.3f} "
            f"(n={results_by_step[steps]['n_reliable']})  falls={falls}/{len(episodes)}  "
            f"yaw_drift={mean_yaw_drift:.1f}deg/10s"
        )

    out_path = run_dir / "checkpoint_trend.json"
    out_path.write_text(json.dumps(results_by_step, indent=2))
    print(f"\nwrote {out_path}")
    env.close()


if __name__ == "__main__":
    main()

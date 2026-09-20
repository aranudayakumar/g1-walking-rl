"""Plot reward/episode-length curves from a training run's progress.csv,
for the challenge's required deliverable (docs/DELIVERABLES.md).

Usage: python -m results.plot_curves --run baseline0001
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "experiments" / "runs" / args.run
    csv_path = run_dir / "progress.csv"
    rows = list(csv.DictReader(open(csv_path)))

    steps = [int(r["time/total_timesteps"]) for r in rows]
    ep_len = [float(r["rollout/ep_len_mean"]) for r in rows]
    ep_rew = [float(r["rollout/ep_rew_mean"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(steps, ep_rew, color="#2563eb")
    axes[0].set_ylabel("mean episode reward")
    axes[0].set_title(f"{args.run}: training curves")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, ep_len, color="#16a34a")
    axes[1].set_ylabel("mean episode length (control steps)")
    axes[1].set_xlabel("training timesteps")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = run_dir / "curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

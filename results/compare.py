"""Compare all evaluated runs side by side, on the metric that actually
matters (distance traveled, not just reward/survival -- docs/DECISIONS.md
ADR-003), so a new experiment can't look like a win by reward/survival
alone without that being visible immediately.

Usage: python -m results.compare
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "runs"

COLUMNS = [
    ("run", 26),
    ("len_s", 7),
    ("reward", 8),
    ("vel_err", 8),
    ("dist_m", 8),
    ("dist_ratio", 18),
    ("falls", 6),
    ("yaw_drift", 12),
]


def main() -> None:
    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        summary_path = run_dir / "eval_summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text())["summary"]
        n = s["n_episodes"]
        falls = n - s["termination_reasons"].get("null", 0)
        # "mean_distance_ratio_reliable_only" replaced "mean_distance_ratio"
        # after EXP-0008 showed short-episode ratios can be spuriously large
        # (docs excludes episodes under half the max episode length); fall
        # back to the old key for eval_summary.json files saved before that.
        dist_ratio = s.get("mean_distance_ratio_reliable_only", s.get("mean_distance_ratio", float("nan")))
        n_reliable = s.get("n_reliable_ratio_episodes")
        ratio_note = f"{dist_ratio:.3f}" + (f" (n={n_reliable})" if n_reliable is not None else "")
        yaw_drift = s.get("mean_abs_yaw_drift_per_10s_deg")
        yaw_note = f"{yaw_drift:.1f} deg/10s" if yaw_drift is not None else "n/a"
        rows.append(
            {
                "run": run_dir.name,
                "len_s": f"{s['mean_episode_length_s']:.1f}",
                "reward": f"{s['mean_total_reward']:.0f}",
                "vel_err": f"{s['mean_vel_error']:.3f}",
                "dist_m": f"{s.get('mean_distance_traveled_m', float('nan')):.2f}",
                "dist_ratio": ratio_note,
                "falls": f"{falls}/{n}",
                "yaw_drift": yaw_note,
            }
        )

    if not rows:
        print("no experiments/runs/*/eval_summary.json found -- run evaluation.evaluate first")
        return

    header = "  ".join(name.ljust(w) for name, w in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[name]).ljust(w) for name, w in COLUMNS))

    print("\ndist_ratio = mean(distance actually traveled / commanded distance) -- the")
    print("metric that actually distinguishes real walking from cheap survival (ADR-003).")
    print("yaw_drift = mean |heading change| per 10s under whatever command each eval episode")
    print("sampled -- large values mean the walk curves/spirals rather than going straight")
    print("(docs/FAILURE_ANALYSIS.md F-0003). 'n/a' = evaluated before this metric existed.")


if __name__ == "__main__":
    main()

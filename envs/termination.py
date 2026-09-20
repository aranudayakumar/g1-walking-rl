"""Termination (episode-ending failure) vs. truncation (time limit).

Two termination conditions, both directly tied to "the robot has fallen":
pelvis too low, or base tipped too far from upright (projected-gravity z
too negative -- see envs/observations.py:projected_gravity; z=-1 means
perfectly upright, z=0 means tipped exactly 90 degrees). The configured
threshold `min_projected_gravity_z=-0.3` (config/default.yaml) triggers
at tilt = acos(0.3) =~ 72.5 degrees from upright, not at 90.
Truncation is handled separately by the env's own step counter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.loader import TerminationConfig


@dataclass(frozen=True)
class TerminationResult:
    terminated: bool
    reason: str | None


def check_termination(
    pelvis_height: float,
    projected_gravity_z: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    cfg: TerminationConfig,
) -> TerminationResult:
    if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
        return TerminationResult(True, "numerical_invalid")
    if pelvis_height < cfg.min_pelvis_height_m:
        return TerminationResult(True, "fell_height")
    if projected_gravity_z > cfg.min_projected_gravity_z:
        return TerminationResult(True, "fell_tilt")
    return TerminationResult(False, None)

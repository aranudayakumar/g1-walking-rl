"""Domain randomization (PROJECT_PLAN.md Phase 8), added only after basic
walking was confirmed working (canonical baseline `exp0011_gait_quality_v2`
-- CLAUDE.md: "Add domain randomization once basic walking works").

Randomizes, per episode: floor friction, a global body-mass scale, and a
motor-strength scale (applied as a torque multiplier). Also applies
randomly-timed external pushes (a velocity impulse to the pelvis) during
an episode. All four are the ones the challenge doc names explicitly
("friction, mass, motor strength, pushes").

Kept as a separate module from the reward/env logic so it can be
completely disabled (the default for every existing config) without
touching anything else -- one conceptual addition, isolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from config.loader import DomainRandomizationConfig


@dataclass
class EpisodeRandomization:
    motor_strength: float  # torque multiplier for this episode


class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, floor_geom_id: int):
        self.model = model
        self.floor_geom_id = floor_geom_id
        # Snapshot nominal values once, at construction, so repeated
        # episodes randomize around the model's *original* values rather
        # than compounding drift from a previous episode's randomization.
        self._nominal_floor_friction = model.geom_friction[floor_geom_id].copy()
        self._nominal_body_mass = model.body_mass.copy()
        self._nominal_body_inertia = model.body_inertia.copy()

    def randomize_episode(self, rng: np.random.Generator, cfg: DomainRandomizationConfig) -> EpisodeRandomization:
        """Apply friction/mass randomization directly to the model (mutates
        shared per-env MjModel state) and return the motor-strength scale
        for this episode (applied in the control loop, not the model)."""
        if not cfg.enabled:
            return EpisodeRandomization(motor_strength=1.0)

        friction_scale = rng.uniform(*cfg.floor_friction_range)
        self.model.geom_friction[self.floor_geom_id] = self._nominal_floor_friction * friction_scale

        mass_scale = rng.uniform(*cfg.mass_scale_range)
        self.model.body_mass[:] = self._nominal_body_mass * mass_scale
        self.model.body_inertia[:] = self._nominal_body_inertia * mass_scale

        motor_strength = float(rng.uniform(*cfg.motor_strength_range))
        return EpisodeRandomization(motor_strength=motor_strength)


class PushScheduler:
    """Applies a random horizontal velocity impulse to a free-joint body's
    qvel at randomly-sampled intervals during an episode."""

    def __init__(self) -> None:
        self.time_until_next_push = float("inf")

    def reset(self, rng: np.random.Generator, cfg: DomainRandomizationConfig) -> None:
        if cfg.enabled:
            self.time_until_next_push = rng.uniform(*cfg.push_interval_s_range)
        else:
            self.time_until_next_push = float("inf")

    def maybe_push(
        self,
        data: mujoco.MjData,
        control_dt: float,
        rng: np.random.Generator,
        cfg: DomainRandomizationConfig,
    ) -> bool:
        if not cfg.enabled:
            return False
        self.time_until_next_push -= control_dt
        if self.time_until_next_push > 0:
            return False

        speed = rng.uniform(*cfg.push_velocity_range)
        angle = rng.uniform(0, 2 * np.pi)
        data.qvel[0] += speed * np.cos(angle)
        data.qvel[1] += speed * np.sin(angle)
        self.time_until_next_push = rng.uniform(*cfg.push_interval_s_range)
        return True

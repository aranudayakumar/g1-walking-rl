"""Minimal typed config loader. One dataclass tree, one yaml file -- no
schema-validation framework, since the config is small and flat enough
that plain dataclasses give the same safety with far less code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SimConfig:
    physics_dt: float
    control_decimation: int
    episode_seconds: float

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.control_decimation

    @property
    def episode_control_steps(self) -> int:
        return int(round(self.episode_seconds / self.control_dt))


@dataclass
class ActionConfig:
    action_scale: float


@dataclass
class PDGainGroup:
    kp: float
    kd: float


@dataclass
class PDGainsConfig:
    hip_pitch: PDGainGroup
    hip_roll: PDGainGroup
    hip_yaw: PDGainGroup
    knee: PDGainGroup
    ankle_pitch: PDGainGroup
    ankle_roll: PDGainGroup


@dataclass
class CommandConfig:
    vx_range: tuple[float, float]
    vy_range: tuple[float, float]
    # Command is sampled once per episode (envs/reset.sample_command, called
    # only from G1WalkEnv.reset). There is no mid-episode resampling, so no
    # separate "resample interval" field exists here -- one was removed after
    # an audit found it unused and misleadingly implying that capability.


@dataclass
class RewardWeightsConfig:
    tracking_lin_vel: float
    tracking_sigma: float
    alive_bonus: float
    torque_penalty: float
    # Added after F-0003 (docs/FAILURE_ANALYSIS.md): penalize yaw rotation
    # (base_ang_vel_z^2) and jittery actions (sum((action-prev_action)^2)).
    # Default 0.0 so any config predating F-0003 still loads and behaves
    # exactly as before.
    ang_vel_penalty: float = 0.0
    action_rate_penalty: float = 0.0
    # Added after F-0004 (docs/FAILURE_ANALYSIS.md): reward longer swing
    # phases at touchdown (envs/contacts.AirTimeTracker), targeting the
    # chattery ~20-30ms steps left unresolved by action_rate_penalty alone.
    # Default 0.0 so any config predating F-0004 still loads unchanged.
    feet_air_time_bonus: float = 0.0
    feet_air_time_threshold_s: float = 0.2
    # Added after F-0004 found feet_air_time_bonus alone is gameable (a
    # foot that never leaves the ground never triggers its touchdown-only
    # reward -- "never step" is zero-cost under it). This term penalizes
    # continuous stance beyond max_stance_s every step it continues, which
    # cannot be dodged the same way (envs/contacts.GroundTimeTracker).
    stance_overrun_penalty: float = 0.0
    max_stance_s: float = 1.0
    # Added after ADR-009 (EXP-0018) found that pushing feet_air_time_bonus
    # harder degrades heading control, but ang_vel_penalty (instantaneous
    # rotation-rate squared) doesn't clearly track it: a slow, steady drift
    # and a larger but zero-mean oscillation can have the same mean
    # ang_vel_z^2, yet only the former accumulates into real heading loss.
    # This term penalizes net deviation from the heading at episode start
    # directly (envs/g1_walk_env, quat_to_yaw), targeting drift specifically
    # rather than rotation magnitude in general.
    heading_deviation_penalty: float = 0.0


@dataclass
class TerminationConfig:
    min_pelvis_height_m: float
    min_projected_gravity_z: float


@dataclass
class PPOConfig:
    algorithm: str
    policy: str
    device: str
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    policy_net: list[int]
    value_net: list[int]
    target_kl: float | None = None


@dataclass
class RewardScheduleTerm:
    # Linear ramp: weight = start for progress<=0, end for progress>=end_fraction,
    # linearly interpolated in between. progress = current_timestep / total_timesteps
    # (training.curriculum_callback.CurriculumProgressCallback), set on the env via
    # G1WalkEnv.set_curriculum_progress -- untouched (progress=0.0) means every
    # scheduled term stays at `start`, so a config with no reward_schedule section
    # behaves exactly as before (no callback is even attached, see training/train.py).
    start: float
    end: float
    end_fraction: float = 1.0


@dataclass
class DomainRandomizationConfig:
    # Phase 8 (PROJECT_PLAN.md): only added once basic walking worked
    # (confirmed on exp0011_gait_quality_v2). Disabled by default so every
    # existing config keeps its exact prior (deterministic-physics) behavior.
    enabled: bool = False
    floor_friction_range: tuple[float, float] = (1.0, 1.0)
    mass_scale_range: tuple[float, float] = (1.0, 1.0)
    motor_strength_range: tuple[float, float] = (1.0, 1.0)
    push_interval_s_range: tuple[float, float] = (1e9, 1e9)  # effectively never
    push_velocity_range: tuple[float, float] = (0.0, 0.0)


@dataclass
class Config:
    model_path: str
    sim: SimConfig
    action: ActionConfig
    pd_gains: PDGainsConfig
    command: CommandConfig
    reward_weights: RewardWeightsConfig
    termination: TerminationConfig
    ppo: PPOConfig
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    # Optional: {reward_weights field name -> RewardScheduleTerm}. Empty by
    # default, so every existing config is unaffected (see RewardScheduleTerm).
    reward_schedule: dict[str, RewardScheduleTerm] = field(default_factory=dict)


def load_config(path: str | Path = "config/default.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        model_path=raw["model_path"],
        sim=SimConfig(**raw["sim"]),
        action=ActionConfig(**raw["action"]),
        pd_gains=PDGainsConfig(**{k: PDGainGroup(**v) for k, v in raw["pd_gains"].items()}),
        command=CommandConfig(
            vx_range=tuple(raw["command"]["vx_range"]),
            vy_range=tuple(raw["command"]["vy_range"]),
        ),
        reward_weights=RewardWeightsConfig(**raw["reward_weights"]),
        termination=TerminationConfig(**raw["termination"]),
        ppo=PPOConfig(**raw["ppo"]),
        domain_randomization=_load_domain_randomization(raw.get("domain_randomization")),
        reward_schedule=_load_reward_schedule(raw.get("reward_schedule")),
    )


def _load_reward_schedule(raw: dict | None) -> dict[str, RewardScheduleTerm]:
    if raw is None:
        return {}
    return {name: RewardScheduleTerm(**term) for name, term in raw.items()}


def _load_domain_randomization(raw: dict | None) -> DomainRandomizationConfig:
    if raw is None:
        return DomainRandomizationConfig()
    return DomainRandomizationConfig(
        enabled=raw.get("enabled", False),
        floor_friction_range=tuple(raw["floor_friction_range"]),
        mass_scale_range=tuple(raw["mass_scale_range"]),
        motor_strength_range=tuple(raw["motor_strength_range"]),
        push_interval_s_range=tuple(raw["push_interval_s_range"]),
        push_velocity_range=tuple(raw["push_velocity_range"]),
    )

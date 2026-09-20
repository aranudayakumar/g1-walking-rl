# Simulation and Control

Owner: main agent + `sim-environment-engineer`.

## Robot/model source
`assets/g1/g1.xml` (upstream, unmodified), vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
`unitree_g1/`, commit `da76818e269b82289eba39808e2fb91d679d6994`, copied
from the already-verified `../standing-rl/assets/g1` (see
`assets/g1/PROVENANCE.md`, `docs/DECISIONS.md` ADR-001).

`assets/g1/g1_walk.xml` / `scene_walk.xml` are derived by
`scripts/build_walk_model.py`: the 12 leg `<position>` actuators become
`<motor>` (direct torque); everything else (geometry, mass, inertia,
joint ranges, waist/arm actuators) is untouched.

## Controlled joints / DoFs
Only the 12 leg joints (`control/joint_map.CANONICAL_ORDER`): per leg,
hip pitch/roll/yaw, knee, ankle pitch/roll. Waist (3) and arms (14) are
held at the upstream `g1.xml` "stand" keyframe ctrl values
(`control/default_pose.UPPER_BODY_TARGETS`, verified against the live
model in `scripts/smoke_test_model.py`) via MuJoCo's own position
actuators; they are not RL-controlled (docs/DECISIONS.md ADR-002).

## State indexing (verified via `scripts/smoke_test_model.py`)
- `nq=36, nv=35, nu=29, njnt=30` (29 hinge joints + 1 free/floating base).
- Floating base: `qpos[0:7]` (xyz + wxyz quat), `qvel[0:6]`.
- Leg joints: `qpos[7:19]`, `qvel[6:18]`, in `CANONICAL_ORDER` (left leg
  block, then right leg block, each hip_pitch/hip_roll/hip_yaw/knee/
  ankle_pitch/ankle_roll) -- `control/joint_map.build_joint_map` resolves
  and asserts this against the live model rather than assuming it.
- Actuator ids 0-11 are the 12 leg motors (same order as joints);
  12-28 are the untouched waist/arm position actuators.

## Physics timestep
`0.002s` (500 Hz), set by the upstream model (`<option
integrator="implicitfast"/>` with default timestep), asserted against
`config/default.yaml: sim.physics_dt` at env construction.

## Control timestep / decimation
`control_decimation=10` -> 50 Hz control (`control_dt=0.02s`). Episode
length: `episode_seconds=20.0` -> 1000 control steps (config/default.yaml
`sim.episode_seconds`, `Config.sim.episode_control_steps`).

## Action definition
12-dim, `Box[-1, 1]`. `target_q = clip(q_default + action_scale *
clip(action, -1, 1), joint_range)`, `action_scale=0.25`
(`control/pd_controller.action_to_target`). PD torque:
`torque = kp*(target_q - q) - kd*qd`, clipped to each joint's
`actuatorfrcrange` (`control/pd_controller.pd_torque`).

## Default pose and scaling
`control/default_pose.build_q_default()`: slight-crouch stance (hip_pitch
-0.10, knee 0.30, ankle_pitch -0.20 rad, mirrored both legs) rather than
the upstream fully-straight-leg keyframe. Starting point cited from the
sibling `../standing-rl` project's validated pose for the identical robot
(docs/RESEARCH_LOG.md R-0003); revisit if evidence says otherwise.

## PD / actuator mapping
Per-joint-group gains (`control/pd_controller.DEFAULT_KP_BY_GROUP/
DEFAULT_KD_BY_GROUP`, same starting point as the default pose): hip
100/2, knee 150/4, ankle 40/2 (kp, kd), applied identically to both legs.

## Reset behavior
`envs/reset.reset_state`: reset to the upstream "stand" keyframe, then
overwrite the 12 leg qpos with `q_default`, zero all velocities,
`mj_forward`. Command velocity `(vx, vy)` is resampled once per episode
(`envs/reset.sample_command`, ranges in `config/default.yaml: command`).

Domain randomization (Phase 8, `envs/domain_randomization.py`, added
after basic walking was confirmed on `exp0011_gait_quality_v2`) runs
*before* `reset_state`'s `mj_forward` call so each episode's randomized
physics take effect from the first observation: floor friction, a global
body-mass+inertia scale, and a per-episode motor-strength multiplier
(applied in the control loop, not the model). Disabled by default
(`domain_randomization.enabled: false`) -- every config predating this
feature is unaffected and produces bit-identical behavior (verified,
`tests/test_domain_randomization.py`).

## Contact handling
Foot ground-contact is detected via 4 corner collision spheres per foot
(`envs/contacts.py`: `LEFT_FOOT_GEOMS`/`RIGHT_FOOT_GEOMS`, verified
against the live model at env construction via `verify_foot_geoms`).
Used by: `envs/domain_randomization.PushScheduler` (none directly, no
contact dependency there), `diagnostics/gait_diagnostic.py` (foot-height/
contact-state plots), and briefly by two reward terms
(`feet_air_time_bonus`, `stance_overrun_penalty`) added and then reverted
in EXP-0012/0013 (docs/FAILURE_ANALYSIS.md F-0004) -- the contact
detection itself is retained (it's correct and useful for diagnostics),
only the two reward terms built on top of it were reverted.

No foot-contact-state *observation* is fed to the policy, and no
foot-slip reward term exists currently (F-0004 found real foot slip
during nominal "contact" -- see that entry -- but a dedicated slip
penalty was not attempted; the two air-time-based terms that were tried
targeted swing duration, not slip velocity, directly).

## Verified smoke tests
- `scripts/smoke_test_model.py` (2026-09-18): model loads; 500 zero-control
  physics steps stay finite; reset-to-keyframe is repeatable.
- `scripts/smoke_test_pd_hold.py` (2026-09-18): PD controller produces
  bounded torque (max 39.9% of any joint's force limit, no saturation)
  holding `q_default` with zero action. The robot is **not**
  self-stabilizing under pure PD (expected -- see below): pelvis height
  crosses the 0.5m termination threshold at t=1.30s, falling forward as
  ankle-pitch tracking error grows past -0.4 rad around t=1s. This is
  physically expected, not a bug: a free-standing biped with joint-space
  PD and no active balance feedback is an unstable inverted-pendulum-like
  system; balancing is exactly what the RL policy must learn. Confirmed
  by re-tracing per-step joint error (not just final state) before
  accepting this conclusion, per failure-analysis discipline.
- `scripts/smoke_test_env.py` (2026-09-18): `gymnasium.utils.env_checker
  .check_env` passes; `reset(seed=42)` is exactly repeatable (obs and
  sampled command both match bit-for-bit); zero-action and random-action
  rollouts both terminate at ~1.3s via `fell_height`, consistent with the
  PD-hold finding above; all observations finite and within the declared
  space throughout.

## Known simulator issues
None found yet. SB3 cannot reliably use Apple MPS
(`docs/RESEARCH_LOG.md` R-0003) -- PPO device is pinned to `cpu`
(`config/default.yaml: ppo.device`).

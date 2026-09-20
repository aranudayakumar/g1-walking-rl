# G1 model provenance

Vendored from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
`unitree_g1/` subdirectory.

```text
source:  https://github.com/google-deepmind/mujoco_menagerie
commit:  da76818e269b82289eba39808e2fb91d679d6994
path:    unitree_g1/
fetched: 2026-08-26
```

`LICENSE`, `CHANGELOG.md`, and upstream `README.md` are the original
upstream files, unmodified. `g1.xml`, `g1_mjx.xml`, `g1_with_hands.xml`,
`scene*.xml`, and `assets/` (meshes) are all unmodified upstream files.

The model is `g1_29dof_rev_1_0`: 29 actuated joints (12 leg, 3 waist, 14
arm/wrist) plus a 6-DOF floating base (`floating_base_joint`).

## `g1_stand.xml` — standing-rl derived model

`g1_stand.xml` is a fork of `g1.xml` with exactly one class of change: the
12 lower-body leg joint actuators were changed from MuJoCo `<position>`
actuators (upstream default: `kp=500 dampratio=1`, one gain for every
joint) to `<motor>` (direct torque) actuators.

Reason: `CLAUDE_PROMPT.md` §2 specifies distinct PD gains per leg-joint
group (hip pitch/roll/yaw, knee, ankle pitch/roll), applied explicitly in
`control/pd_controller.py`. That requires the environment to command
torque directly; MuJoCo's built-in position actuator would apply its own
internal (uniform) gains instead of ours.

Per-joint `ctrlrange`/`forcerange` on each new motor actuator equals that
joint's original `actuatorfrcrange` from `g1.xml` — no torque limits were
changed, only the control mode.

Waist and arm actuators are untouched (`<position>`, upstream gains):
they are held at a fixed target and are not RL-controlled in the first
training stage, so MuJoCo's internal position PD is sufficient for them.

No geometry, mass, inertia, joint range, or collision property was
changed anywhere in the file. `scene_stand.xml` is `scene.xml` pointed at
`g1_stand.xml` instead of `g1.xml`, with the floor `friction` made
explicit (`1.0 0.005 0.0001`) so ground friction is a known, versioned
quantity rather than the MuJoCo default.

Regenerate `g1_stand.xml` from a newer upstream `g1.xml` by re-running the
transform documented in the file's own header comment (12 named
`<position>` → `<motor>` substitutions); do not hand-edit it independently
of `g1.xml`, or the two will silently drift.

# PPO Locomotion Gait Evaluation — Unitree G1

## Objective

Evaluate the existing PPO locomotion policy for the Unitree G1 in MuJoCo, focusing on stability, forward velocity tracking, and foot movement. Identify gait limitations and propose three targeted reward-function improvements.

## Baseline

*Policy:* EXP-0026 (`best_checkpoint_2M.zip`)  
*Configuration:* `config/exp0012_air_time.yaml`  
*Evaluation:* 20 deterministic episodes, seeds 1000–1019, 20 seconds each.

The following results come from the repository's previously recorded evaluation, not a newly executed rollout.

| Metric | Recorded result |
|---|---|
| Falls | 0 / 20 |
| Mean episode duration | 20.0 s |
| Mean velocity error | 0.0361 m/s |
| Mean net displacement | 4.28 m |
| Mean reliable distance ratio | 0.918 |
| Mean absolute endpoint yaw drift (normalized to 10 s) | 17.62° |

## Observations

### 1. Stability

The policy maintains upright locomotion throughout the recorded evaluation, with no falls across 20 episodes. However, its heading control remains imperfect. The 17.62° mean normalized yaw drift indicates unintended changes in orientation despite successful balance.

*Assessment:* Stable in the tested simulation conditions, but directional stability needs improvement.

### 2. Forward Velocity

The mean velocity-tracking error of 0.0361 m/s indicates that the policy follows commanded velocities reasonably well.

However, the tracking reward measures velocity in the pelvis frame. Therefore, the robot can track its local forward command while drifting away from its original world-frame direction.

*Assessment:* Good basic velocity tracking, but heading drift reduces the quality of straight-line walking.

### 3. Foot Movement

The repository's gait investigations document three recurring issues:

- Short foot-swing phases, producing a shuffling gait.
- Wide leg abduction, indicating excessive lateral foot placement.
- Intermittent loaded-foot slipping during stance.

These behaviors suggest that the policy has learned a viable locomotion strategy without consistently achieving desirable walking mechanics.

These are documented project-level observations, not new measurements from the canonical checkpoint.

*Assessment:* Foot movement is the primary area for improvement.

## Proposed Reward-Function Changes

### 1. Swing-Foot Clearance Reward

Introduce a reward that encourages moderate vertical foot clearance during swing.

*Motivation:* The policy exhibits short swing phases and shuffling behavior.

*Implementation concept:* Reward foot height near a target clearance only while the foot is airborne and a nonzero walking command is active.

*Expected effect:* More distinct steps and reduced foot dragging.

*Potential risk:* Excessive weighting could encourage exaggerated foot lifting.

### 2. Stance-Foot Slip Penalty

Penalize horizontal foot velocity when the corresponding foot is in stable ground contact.

*Motivation:* The repository identifies intermittent loaded-foot slip.

*Implementation concept:*

`r_slip = -w_slip * sum(contact_i * ||v_foot_xy_i||²)`

Use contact filtering to avoid penalizing normal touchdown and liftoff.

*Expected effect:* Reduced foot sliding and improved contact stability.

*Potential risk:* Excessive penalties may discourage stepping and forward motion.

### 3. Heading-Stabilization Reward Refinement

Refine the existing heading-related reward to discourage unintended yaw drift during straight-walking commands.

*Motivation:* The recorded evaluation shows significant residual heading drift.

*Implementation concept:* Penalize deviation from the initial heading for straight-walking episodes, with appropriate handling for commanded turning if turning commands are introduced later.

*Expected effect:* Improved directional consistency and straighter walking trajectories.

*Potential risk:* Overweighting heading control could cause unnatural compensatory movements or reduced forward progress.

## Recommended Validation

Warm-start from the canonical EXP-0026 checkpoint and test each reward modification separately.

Compare each candidate against the original checkpoint using identical evaluation conditions.

Track:

- Fall rate and episode duration
- Forward velocity-tracking error
- Net displacement and reliable distance ratio
- Endpoint yaw drift
- Foot clearance and swing duration
- Stance-foot horizontal slip

An improvement should demonstrate better gait mechanics without sacrificing survival or velocity tracking.

## Conclusion

The existing PPO policy successfully learns basic upright locomotion and accurate velocity tracking under the recorded simulation conditions. However, its gait remains limited by short swing phases, lateral leg movement, foot slip, and heading drift.

The most promising reward-function modifications are swing-foot clearance shaping, stance-foot slip penalization, and heading-stabilization refinement.

These recommendations are hypotheses requiring controlled evaluation, rather than experimentally verified improvements.

## Sources

- Repository README and canonical EXP-0026 evaluation summary
- `docs/FAILURE_ANALYSIS.md`
- `config/exp0012_air_time.yaml`
- Repository gait diagnostic and evaluation tooling

*Evaluation status:* Review of existing experimental evidence. Fresh MuJoCo rollout and reward ablation experiments have not yet been performed.

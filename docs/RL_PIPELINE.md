# RL Pipeline

Owner: main agent + `rl-training-engineer`.

## PPO implementation
`stable-baselines3` (`training/train.py`). We do not implement PPO/GAE/
value/clipping ourselves; SB3 owns all of it (CLAUDE.md). Device pinned
to `cpu` (`config/default.yaml: ppo.device`) -- SB3 does not reliably use
Apple MPS (docs/RESEARCH_LOG.md R-0003).

## Policy/value architecture
`MlpPolicy`, separate policy/value MLP heads, `[256, 256]` each
(`config/default.yaml: ppo.policy_net/value_net`). Default SB3
initialization and Gaussian action distribution (diagonal, learned
per-dimension std).

## Observation vector
44-dim `float32`, built by `envs/observations.build_observation`. In
order:

| Field | Dim | Meaning | Units | Frame | Source |
|---|---|---|---|---|---|
| leg joint pos (rel.) | 12 | `qpos[leg] - q_default` | rad | joint space | `data.qpos` via `JointMap.qpos_adr` |
| leg joint vel | 12 | `qvel[leg]` | rad/s | joint space | `data.qvel` via `JointMap.qvel_adr` |
| base angular velocity | 3 | pelvis angular velocity | rad/s | pelvis body frame | `mj_objectVelocity(..., flg_local=1)` |
| projected gravity | 3 | unit gravity direction | unitless | pelvis body frame | `R_wb^T @ [0,0,-1]` |
| commanded velocity | 2 | (vx, vy) target | m/s | pelvis body frame | resampled once per episode, `envs/reset.sample_command` |
| previous action | 12 | last policy output | unitless [-1,1] | n/a | env's own `_prev_action` |

Base **linear** velocity is deliberately excluded (used only internally
for the reward, not observed by the policy) -- see docs/DECISIONS.md
ADR-002 for why.

No normalization/`VecNormalize` yet -- fields are already small-scale
(joint angles/velocities in radians, unit gravity vector, velocities in
single-digit m/s). Revisit only if a measured training failure points at
observation scale (docs/FAILURE_ANALYSIS.md).

## Action vector
12-dim `float32`, `Box[-1, 1]`. Maps to a PD position target via
`control/pd_controller.action_to_target` (see docs/SIMULATION.md
"Action definition"). Applied for `control_decimation=10` physics
sub-steps per env `step()` call (i.e. the action is held constant for
0.02s of physics time, recomputing torque from fresh `q, qd` each
sub-step).

## Reward
Starts from the challenge doc's minimum, plus terms added after measured
gait-quality failures (F-0003, F-0004) (`rewards/walking_reward.compute_reward`):

```
tracking_lin_vel   = w_track * exp(-||v_cmd_xy - v_base_xy||^2 / sigma^2)
alive_bonus        = w_alive * 1.0
torque_penalty     = -w_torque * sum(torque^2)
ang_vel_penalty    = -w_ang_vel * base_ang_vel_z^2
action_rate_penalty= -w_action_rate * sum((action - prev_action)^2)
feet_air_time_bonus= w_air_time * sum_{feet at touchdown}(swing_time - threshold)
reward = tracking_lin_vel + alive_bonus + torque_penalty + ang_vel_penalty
       + action_rate_penalty + feet_air_time_bonus
```

Canonical weights (`config/exp0012_air_time.yaml: reward_weights`, unchanged
since EXP-0012 and used by the current canonical baseline
`exp0026_air_time_on_exp0023_long`, docs/DECISIONS.md ADR-011): `tracking_lin_vel
=1.0`, `tracking_sigma=0.25`, `alive_bonus=0.5`, `torque_penalty=0.0002`,
`ang_vel_penalty=0.1`, `action_rate_penalty=0.01`, `feet_air_time_bonus
=3.0`, `feet_air_time_threshold_s=0.2`. `config/default.yaml` keeps the
original three-term minimum (every added term defaults to `0.0` for
backward compatibility -- see `config/loader.py`).

`ang_vel_penalty`'s weight was **measured, not guessed**: a first attempt
at `1.0` (same order as `tracking_lin_vel`) made a real walking gait's
typical yaw-wobble cost (-0.44/step) exceed its tracking reward
(+0.80/step), collapsing the policy to standing still (EXP-0010). `0.1`
was chosen by replaying the working gait through the new reward math
offline and picking a weight that keeps its typical cost to ~5% of the
tracking reward (docs/DECISIONS.md ADR-005) -- verify any future
reward-weight choice the same way, not by feel.

`feet_air_time_bonus` (`envs/contacts.AirTimeTracker`) rewards a longer
swing phase at the moment each foot touches down, targeting the ~20ms
median swing duration measured in the pre-F-0004 baseline (an order of
magnitude below a deliberate stride). Trained *from scratch* (EXP-0012)
it was gamed: a foot that never lifts never triggers a touchdown-only
reward, so the policy learned to skate instead of step. A complementary
penalty for overstaying ground contact (EXP-0013) closed that loophole
but caused a 100% fall rate (the two terms fight over the same variable).
What worked (EXP-0016, docs/DECISIONS.md ADR-007): the *same, unmodified*
reward term, introduced by **fine-tuning from an already-good checkpoint
at a reduced learning rate** (`training/train.py --init-from ...
--init-learning-rate 5e-5`) instead of training from scratch. An
established policy has real, working reward to lose by switching to
skating; a freshly-initializing one does not. Result: median swing
duration doubled (20ms -> 40ms, directly measured, not inferred from
reward) with no regression elsewhere.

Every component is returned in `info` for per-episode logging
(`evaluation/evaluate.py`); do not add a term without a named measured
failure (docs/FAILURE_ANALYSIS.md) and a record in docs/DECISIONS.md.

## Termination / truncation
Termination (`envs/termination.check_termination`): pelvis height <
0.5m ("fell_height"), or projected-gravity z > -0.3 ("fell_tilt", ~>=70
deg from upright), or non-finite qpos/qvel ("numerical_invalid").
Truncation: `episode_seconds=20.0` -> 1000 control steps
(`config/default.yaml: sim.episode_seconds`, handled in
`G1WalkEnv.step`, separate from termination).

## Rollout configuration
`n_steps=1024` per env (`config/default.yaml: ppo.n_steps`).
`training/vec_env.make_vec_env`: `DummyVecEnv` for smoke tests
(`--no-subproc`), `SubprocVecEnv` for real runs -- validated working on
this macOS/arm64 machine with MuJoCo in EXP-0004 (9729 fps rollout
collection with 8 parallel envs, vs. ~2300 fps for 2 `DummyVecEnv` envs).

## PPO configuration
`batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
lr=3e-4, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5` -- SB3/PPO paper
defaults, not yet tuned for this task (`config/default.yaml: ppo`).
Change one at a time and log the reasoning (docs/DECISIONS.md /
experiments/EXPERIMENT_LOG.md) if evidence motivates it.

## Logging
SB3's own logger, configured to `stdout` + `csv` + `tensorboard`, written
to `experiments/runs/<run_name>/` (`training/train.py`). Per-iteration:
`rollout/ep_len_mean`, `rollout/ep_rew_mean`, and (from the 2nd iteration
onward -- SB3 flushes each iteration's `train/` stats at the *start* of
the next dump, so a 1-iteration run shows no `train/` block, see
EXP-0004) `train/{approx_kl, clip_fraction, entropy_loss,
explained_variance, loss, value_loss, policy_gradient_loss}`.

## Checkpoint / resume
`training/train.py --checkpoint-freq N` uses SB3's `CheckpointCallback`
to save periodic checkpoints under `experiments/runs/<run_name>/
checkpoints/`; the final model always saves to `.../final_model.zip`.
Each run also writes `run_meta.json` (config path, seeds, env count,
timesteps, elapsed time, and `init_from`/`init_learning_rate` when used)
for reproducibility.

`training/train.py --init-from <checkpoint.zip> [--init-learning-rate
<lr>]` (added for EXP-0016, docs/DECISIONS.md ADR-007) warm-starts PPO
from an existing checkpoint's weights instead of random initialization --
`PPO.load(path, env=vec_env)`, then optionally overriding the learning
rate (SB3 requires rebuilding the LR schedule via
`stable_baselines3.common.utils.get_schedule_fn`, not just setting the
attribute, for this to take effect). This is now the preferred way to
introduce a new reward term that risks an easy degenerate escape when
learned from scratch: fine-tuning from an established policy at a
reduced learning rate gives the policy real, working behavior to lose by
exploiting the new term, which a freshly-initializing policy does not
have. Also incidentally avoided the climb-crash-recover PPO oscillation
seen in every from-scratch run in this project (F-0002) -- not yet
confirmed as a general fix for that, but a promising, untested lead.

## Deterministic evaluation
`evaluation/evaluate.py`, always run separately from training
(CLAUDE.md). Fixed disjoint seed range `1000-1019` (never used for
training seeds), `model.predict(obs, deterministic=True)`. Reports per
episode: length (steps and seconds), total reward, mean velocity-tracking
error, mean torque cost, termination reason, final pelvis height; and in
aggregate: mean episode length/reward/vel-error and a termination-reason
histogram.

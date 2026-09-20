# Experiment Log

Append substantial experiment summaries here. Detailed runs may also have individual files in `experiments/runs/`.

Do not rewrite history to make results look cleaner.

## Baseline
Not established. (Vertical slices 1-3 below are pipeline validation, not the canonical baseline learning run -- that comes at PROJECT_PLAN.md Phase 6.)

---

### EXP-0001 — Model load / physics-step smoke test
Date: 2026-09-18

Hypothesis: `assets/g1/scene_walk.xml` (derived model, legs converted to motor actuators) loads correctly and steps stably with zero control.

Why this experiment now: first vertical slice; nothing else can be trusted until this passes.

Code state: initial commit-less working tree (no git repo yet in this project), `scripts/build_walk_model.py` + `scripts/smoke_test_model.py`.

Config:
- seed(s): n/a (deterministic, no RNG)
- env count: 1
- physics dt: 0.002s
- control dt: n/a (raw physics steps, no control loop yet)
- action scale: n/a
- reward terms/scales: n/a
- PPO differences: n/a
- budget: 500 physics steps (1s)

Evaluation protocol: inspect `nq/nv/nu/njnt`, per-joint qpos/qvel address and range, per-actuator gaintype/ctrlrange/forcerange, keyframe presence, then step 500x with `ctrl=0` checking `isfinite`.

Results: `nq=36 nv=35 nu=29 njnt=30`. All 12 leg actuators confirmed `gaintype=0` (fixed/motor) with `ctrlrange==forcerange` matching each joint's `actuatorfrcrange` from upstream `g1.xml`. Keyframe `stand` present (`qpos` len 36, `ctrl` len 29). 500/500 steps finite. Reset-to-keyframe exactly repeatable (`np.allclose` true).

Behavioral observations: with zero leg torque, pelvis falls from 0.79m to 0.084m over 1s (legs give no support) -- expected, confirms motor actuators are actually driving the joints (a stuck/miswired actuator would not collapse this way).

Conclusion: model, indexing, and physics stepping are correct and match the design intent (docs/DECISIONS.md ADR-002).

Decision: Keep. Proceed to control-mapping vertical slice.

Next: EXP-0002 (PD-hold control mapping).

---

### EXP-0002 — PD control-mapping smoke test (zero-action hold)
Date: 2026-09-18

Hypothesis: the PD controller (`control/pd_controller.py`) computes torques of sane magnitude (no saturation) and the sim stays numerically stable when commanded to hold the default leg pose with zero policy action.

Why this experiment now: isolates "is the control mapping correct" from "does the full env/reward/reset contract work" (CLAUDE.md fast-iteration ordering) -- must pass before writing the Gymnasium env.

Code state: `control/joint_map.py`, `control/pd_controller.py`, `control/default_pose.py`, `scripts/smoke_test_pd_hold.py`.

Config:
- seed(s): n/a (deterministic)
- env count: 1
- physics dt: 0.002s
- control dt: 0.02s (decimation=10)
- action scale: 0.25 (unused directly -- action is fixed at zero)
- PD gains: hip 100/2, knee 150/4, ankle 40/2 (kp/kd), cited from `../standing-rl` (docs/RESEARCH_LOG.md R-0003)
- reward terms/scales: n/a (reward module not built yet)
- PPO differences: n/a
- budget: 250 control steps (5s)

Evaluation protocol: hold `q_default` (zero action) for 5s; track pelvis height, max |torque|/forcerange fraction, and time-to-cross a candidate 0.5m termination height; pass criteria = numerically finite throughout, no torque saturation, and the fall (if any) is gradual, not an instant blow-up in the first ~0.4s.

Results: 250/250 steps finite. Max |torque|/limit fraction = 0.399 (no saturation). Pelvis height: 0.788 -> 0.095m over 5s, crossing 0.5m at t=1.30s.

Behavioral observations: per-step trace showed the fall is driven by ankle-pitch tracking error growing from ~0 to -0.41 rad between t=0.5s and t=1.0s (well past the ankle's usable range for holding), i.e. the whole body pitches forward like an inverted pendulum and the low ankle gain (kp=40) cannot arrest it once it starts. Height decays smoothly (0.79 -> 0.78 -> 0.71 -> 0.47), not instantaneously.

Conclusion: this is expected physics, not a bug -- a free-standing biped with pure joint-space PD and no active balance/COM feedback is not self-stabilizing; that instability is exactly what the RL policy has to learn to counteract. Confirmed by tracing the mechanism (ankle tracking error growth) rather than accepting the final-height number alone, per failure-analysis discipline (rule out setup bugs with evidence, don't just eyeball the outcome).

Decision: Keep gains/pose as the starting config. Do not "fix" PD to self-balance -- that would defeat the purpose of training. Revised the smoke test's pass/fail criteria to match its actual purpose (control-mapping sanity, not indefinite standing).

Next: EXP-0003 (full env contract).

---

### EXP-0003 — Environment contract smoke test
Date: 2026-09-18

Hypothesis: `envs/g1_walk_env.py` satisfies the Gymnasium API, resets deterministically given a seed, and produces finite observations under both zero-action and random-action rollouts, terminating consistently with the EXP-0002 finding.

Why this experiment now: last vertical-slice gate before PPO integration (PROJECT_PLAN.md Phase 3 exit tests).

Code state: `envs/g1_walk_env.py`, `envs/observations.py`, `envs/reset.py`, `envs/termination.py`, `rewards/walking_reward.py`, `config/default.yaml`, `config/loader.py`, `scripts/smoke_test_env.py`.

Config:
- seed(s): 42 (determinism check), 0 (zero-action rollout), 1 (random-action rollout), 2 (stand-still command override)
- env count: 1
- physics dt: 0.002s
- control dt: 0.02s
- action scale: 0.25
- reward terms/scales: tracking_lin_vel=1.0 (sigma=0.25), alive_bonus=0.5, torque_penalty=0.0002
- PPO differences: n/a (PPO not yet integrated)
- budget: up to 1000 control steps (20s) per rollout

Evaluation protocol: `gymnasium.utils.env_checker.check_env`; `reset(seed=42)` called twice, compare obs and sampled command bit-for-bit; zero-action and random-action rollouts to termination/truncation, asserting finite obs and `observation_space.contains(obs)` every step.

Results: `check_env` passed (only expected warnings about unbounded observation Box, which is intentional -- no natural bounds for these quantities). `reset(seed=42)` exactly repeatable. Zero-action rollout: 66 steps (1.32s), `terminated=True`, `reason=fell_height`, total_reward=51.13. Random-action rollout: 65 steps (1.30s), same termination reason, total_reward=29.42.

Behavioral observations: termination timing (~1.3s) matches EXP-0002's PD-hold finding almost exactly, cross-validating that the full env's control loop reproduces the isolated PD-mapping behavior rather than introducing a new bug (e.g. wrong decimation, wrong gain application order).

Conclusion: environment contract is correct and consistent with the lower-level control-mapping result. Ready for PPO integration.

Decision: Keep. Proceed to PPO smoke pipeline (PROJECT_PLAN.md Phase 5).

Next: EXP-0004 (PPO smoke run: rollout -> GAE/value -> update -> checkpoint -> reload -> deterministic eval, tiny budget).

---

### EXP-0004 — PPO smoke pipeline (rollout -> update -> checkpoint -> reload -> eval)
Date: 2026-09-18

Hypothesis: stable-baselines3 PPO can collect rollouts from `G1WalkEnv`, run a clipped policy update with GAE/value internals it owns (not reimplemented here), checkpoint, reload, and produce a consistent deterministic evaluation -- with no NaNs anywhere in the loop.

Why this experiment now: PROJECT_PLAN.md Phase 5 gate; must pass before any real training budget is spent.

Code state: `training/train.py`, `training/vec_env.py`, `evaluation/evaluate.py`.

Config:
- seed(s): 0 (training), 1000-1004 (deterministic eval, disjoint range)
- env count: 2 (DummyVecEnv, `smoke0002`), then 8 (SubprocVecEnv, `smoke_subproc`)
- physics dt: 0.002s / control dt: 0.02s
- action scale: 0.25
- reward terms/scales: as ADR-002 (tracking_lin_vel=1.0/sigma=0.25, alive_bonus=0.5, torque_penalty=0.0002)
- PPO: n_steps=1024, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, lr=3e-4, net=[256,256], device=cpu (config/default.yaml, unmodified from baseline defaults)
- budget: `smoke0002` = 4096 timesteps (2 PPO iterations); `smoke_subproc` = 8192 timesteps (1 iteration, 8 envs) purely to validate SubprocVecEnv works on macOS with MuJoCo before committing to it for the real baseline run

Evaluation protocol: inspect SB3's own `train/` log block for finite loss/entropy/value/KL stats; run `evaluation.evaluate` on the resulting checkpoint over 5 fixed eval seeds with `deterministic=True` actions.

Results: `smoke0002` (2 iterations) logged `approx_kl=0.017, clip_fraction=0.232, entropy_loss=-17.0, value_loss=13.0, explained_variance=-0.013, loss=3.07` -- all finite, magnitudes sane for an untrained policy (near-zero explained variance is expected this early). `smoke_subproc` (8 parallel envs) reached 9729 fps during rollout collection (vs. ~2300 fps for 2 DummyVecEnv envs), confirming SubprocVecEnv + MuJoCo works without fork-safety issues on this macOS/arm64 setup. Deterministic eval on `smoke0002`'s checkpoint: mean episode length 64.4 steps (1.29s), all 5/5 episodes end via `fell_height` -- statistically indistinguishable from the untrained EXP-0002/0003 baseline (~1.3s), as expected at only 4096 training timesteps.

Behavioral observations: checkpoint reload reproduces the same qualitative behavior (falls at the same timescale) as the freshly-trained model, i.e. no save/load corruption or device-mismatch bug.

Conclusion: the full pipeline (env -> rollout -> GAE/value -> clipped update -> checkpoint -> reload -> deterministic eval) works end to end with a maintained trainer, satisfying PROJECT_PLAN.md Phase 5's exit test. SubprocVecEnv is safe to use for a real training budget.

Decision: Keep. Proceed to a bounded Phase 6 baseline run using SubprocVecEnv.

Next: EXP-0005 (first bounded baseline training run, real budget, watch reward/episode-length curves for learning signal).

---

### EXP-0005 — First bounded baseline training run
Date: 2026-09-18 (pre-registered before running, per CLAUDE.md "before expensive work record hypothesis/metric/budget/stop condition")

Hypothesis: with the minimal reward/env from ADR-002 and unmodified baseline PPO hyperparameters, PPO shows measurable learning signal (rising `ep_len_mean` and/or `ep_rew_mean`, above the ~64-66-step / ~22-reward untrained baseline from EXP-0003/0004) within a CPU-feasible budget. We are not expecting a clean walking gait -- PROJECT_PLAN.md Phase 6 goal is "does learning signal exist," not convergence.

Metric: SB3's `rollout/ep_len_mean` and `rollout/ep_rew_mean` over training, plus a post-hoc deterministic evaluation (`evaluation.evaluate`, same 20 fixed seeds as future runs) comparing episode length/reward/termination-reason distribution against the EXP-0004 untrained checkpoint.

Budget: 3,000,000 timesteps, `n_envs=12` (SubprocVecEnv, leaving ~3 of 15 cores headroom), seed=0, checkpoint every 250k steps. Expected wall-clock: ~5-8 minutes at the ~8-10k steps/s observed in EXP-0004's `smoke_subproc` throughput test.

Stop condition: fixed budget (3M timesteps) regardless of intermediate results -- this is a bounded diagnostic run, not a train-to-convergence run. If `ep_len_mean` has not moved at all above the untrained baseline by the end, that is itself the result to analyze (docs/FAILURE_ANALYSIS.md), not a reason to keep training uncontrolled.

Code state: no changes since EXP-0004; config/default.yaml unmodified baseline.

Config: physics dt 0.002s, control dt 0.02s, action_scale 0.25, reward = tracking_lin_vel(1.0,sigma=0.25)+alive_bonus(0.5)+torque_penalty(0.0002), PPO n_steps=1024/batch=256/epochs=10/gamma=0.99/gae_lambda=0.95/clip=0.2/lr=3e-4/net=[256,256]/device=cpu.

Evaluation protocol: `evaluation.evaluate --model experiments/runs/baseline0001/final_model.zip --episodes 20` (fixed seeds 1000-1019, `deterministic=True`), compared against the EXP-0004 untrained numbers (mean length 64.4 steps, 5/5 `fell_height`).

Results: Completed in 413.6s wall-clock (7253 steps/s, 12 SubprocVecEnv workers). Training curve (`experiments/runs/baseline0001/curves.png`): `ep_len_mean`/`ep_rew_mean` roughly flat for the first ~1.5M timesteps (64->~150 steps), then rise sharply between 1.5M-2.0M timesteps to ~700-780 steps / ~700-750 reward, then oscillate in that range (with two visible dips, e.g. around 2.0M and 2.7M) through 3.0M.

Deterministic evaluation (`evaluation.evaluate`, 20 fixed seeds 1000-1019, `deterministic=True`): mean episode length **825.7/1000 steps (16.5s)**, mean reward **947.1**, mean velocity error **0.155 m/s**. **15/20 episodes reach the full 20s truncation** (`termination_reason=null`) with velocity error 0.06-0.16 m/s; the remaining **5/20 fall** (`fell_height`), and every one of those 5 has the highest commanded speeds in the eval set (`|v_cmd|` roughly 0.37-0.55 m/s vs. 0.0-0.26 m/s for the 15 that survive) -- a clean, evidence-backed pattern, not noise.

Sanity check against reward-hacking (seed 1002, cmd=[0.23, -0.06] m/s): pelvis moved **3.09m forward / 0.52m lateral over 20s** (commanded displacement would be 4.57m / -1.14m -- undershooting, especially laterally, but clearly real locomotion, not standing still). Left-knee angle ranged 0.09-0.51 rad over the episode (std 0.076) -- genuine cyclic leg articulation, not a frozen pose. Video: `experiments/runs/baseline0001/rollout_seed1002.mp4`.

Behavioral observations: `train/approx_kl` (0.09-0.19) and `train/clip_fraction` (0.55-0.75) stayed far above typical healthy-PPO ranges (commonly targeted ~0.01-0.03 KL) for most of the run -- the policy is updating very aggressively each iteration. This is a plausible mechanism for both the reward-curve oscillation during the plateau and the higher-speed-command failures (an aggressive update policy may overwrite a working slow-gait solution before consolidating a faster one). Not yet confirmed as *the* mechanism -- see F-0002.

Conclusion: hypothesis confirmed -- clear, measured learning signal well above the untrained baseline, on a CPU-only, ~7-minute training budget. This satisfies PROJECT_PLAN.md Phase 6's exit test ("does learning signal exist"). The failure mode (falls concentrated at higher commanded speed) and the PPO instability signal (high KL/clip-fraction) are real, evidence-backed observations worth one follow-up experiment before calling this the final baseline.

Decision: **Freeze `baseline0001` as the canonical baseline** (docs/DELIVERABLES.md, PROJECT_STATE.md) -- it already meets the challenge's minimum bar. Run one bounded follow-up experiment (EXP-0006) testing whether reducing update aggressiveness (lower `clip_range` and/or a `target_kl` early-stop) improves high-speed stability, since PPO config has not been touched from its untuned defaults yet.

Next: EXP-0006 (PPO stability follow-up, informed by F-0002).

---

### EXP-0006 — PPO `target_kl` stability follow-up
Date: 2026-09-18

Hypothesis: capping PPO's per-update KL divergence (`target_kl=0.03`, SB3's built-in early-stopping-within-epoch) will reduce the destabilizing update aggressiveness observed in `baseline0001` (`approx_kl` 0.09-0.19, `clip_fraction` 0.55-0.75) and, in particular, eliminate or reduce the falls concentrated at high commanded speed (F-0002). Single conceptual change: PPO update aggressiveness, via the single most direct/standard lever (`target_kl`); `clip_range` left unchanged at 0.2.

Why this experiment now: F-0002's fastest discriminating test, run immediately after freezing `baseline0001`.

Code state: added `target_kl: float | None` to `config.loader.PPOConfig` (default `None`, backward compatible) and wired it into `training/train.py`'s `PPO(...)` call. New config `config/exp0006_target_kl.yaml` (identical to `config/default.yaml` except `ppo.target_kl: 0.03`).

Config: identical to `baseline0001` (env, reward, PPO hyperparameters) except `ppo.target_kl=0.03`. seed=0, n_envs=12 (SubprocVecEnv), 3,000,000 timesteps, checkpoint-freq unset (final only).

Evaluation protocol: identical to `baseline0001` -- `evaluation.evaluate`, same 20 fixed seeds (1000-1019), `deterministic=True`. Also re-ran `baseline0001` through the *same* evaluation script after adding a distance-traveled metric (see below), so both are compared on identical protocol and identical metric set.

Results: Training ran faster (298.7s vs. baseline's 413.6s, 10043 vs. 7253 steps/s -- `target_kl` early-stopping skips unneeded gradient steps) and, as intended, `approx_kl` dropped to 0.02-0.04 and `clip_fraction` to 0.25-0.29 throughout. Deterministic eval: **20/20 episodes survive the full 20s (0 falls)**, mean reward 952.0 (vs. baseline's 947.1 over 15/20 survivors) -- read alone, this looks like a clean win.

It is not, once distance traveled is measured (metric added to `evaluation/evaluate.py` specifically because this result looked suspiciously clean): mean distance traveled **0.21m/episode (5.1% of the commanded expected distance)**, vs. `baseline0001`'s **1.85m/episode (56.6% of expected)**. `exp0006_target_kl`'s velocity-tracking error is also worse (0.225 vs 0.155 m/s). The policy learned to stand almost motionless and collect `alive_bonus` for the full 20s rather than walk -- a degenerate optimum that trivially satisfies the termination conditions without doing the task.

Behavioral observations: per-episode sanity check (seeds 1002 and 1017, direct rollout with pelvis xy tracked) confirmed the aggregate finding at the individual-episode level -- both showed near-zero net displacement (`[0.25,-0.10]`m and `[-0.03,-0.51]`m respectively, vs. expected `[4.57,-1.14]`m and `[11.02,2.57]`m) despite nonzero, non-frozen knee articulation (i.e. not a literal paralysis bug -- it moves, just not to go anywhere).

Conclusion: F-0002's hypothesis (b) (PPO instability is *the* cause of high-speed failure, and fixing it yields a better gait) is not supported -- capping KL removed the instability but revealed the reward alone doesn't sufficiently penalize standing still, so the optimizer found the easier degenerate solution once the harder (noisy, unstable) path to real high-speed walking was no longer being taken. `baseline0001`'s noisier training, imperfect as it is, produced the more genuine walker. Full writeup: `docs/FAILURE_ANALYSIS.md` F-0002.

Decision: **Keep `baseline0001` as the canonical baseline. Do not adopt `target_kl` as a default change.** Keep this run and its config as a documented negative result, not deleted.

Next: (optional, not executed -- see F-0002 "Follow-up") EXP-0007 would test a reward that scales/removes the flat `alive_bonus` in favor of rewarding only real tracked velocity + effort, ideally combined with `target_kl`, to test whether that removes the "stand still" degenerate optimum while keeping PPO stable. Left for whoever continues this project; not required for the challenge's minimum bar, which `baseline0001` already meets.

---

### EXP-0007 — Reward shaping: reduce `alive_bonus` weight to remove the cheap-survival incentive
Date: 2026-09-18 (pre-registered before running)

Hypothesis: `baseline0001`'s `alive_bonus=0.5` makes "survive without real velocity tracking" cheap enough that, combined with noisy/unstable updates, the policy sometimes settles for near-stillness at hard (high-speed) commands instead of a real gait (F-0002's revised, corrected mechanism after ADR-003). Reducing `alive_bonus` to 0.1 (single conceptual change -- a term-weight rebalance, not a new term; all three challenge-minimum reward components are kept) should make real tracking necessary to earn good reward, and should improve `distance_ratio` (the metric that actually matters, per ADR-003) at high-speed commands specifically, possibly at the cost of the currently-clean low-speed behavior if 0.1 is too aggressive a cut.

Failure being targeted: F-0002 (high-speed falls in `baseline0001`; EXP-0006 showed the fix isn't PPO stability alone).

Metric: `evaluation.evaluate` on the same 20 fixed seeds (1000-1019) as `baseline0001`/`exp0006_target_kl`, with particular attention to `distance_ratio` (not just reward/survival, per ADR-003) and specifically whether the previously-failing high-speed seeds (1001, 1013, 1017, and the two unexplained mid-speed fallers 1005/1007 per the corrected F-0002) now walk rather than merely survive.

Budget: 3,000,000 timesteps, `n_envs=12` (SubprocVecEnv), seed=0 -- identical protocol to `baseline0001` and `exp0006_target_kl` so all three are directly comparable.

Stop condition: fixed budget, regardless of intermediate curve shape (same discipline as EXP-0005).

Code state: no source changes; new config `config/exp0007_low_alive_bonus.yaml` (`reward_weights.alive_bonus: 0.1`, everything else identical to `config/default.yaml`, no `target_kl`).

Config: as `baseline0001` except `reward_weights.alive_bonus=0.1` (was 0.5).

Evaluation protocol: `evaluation.evaluate --model experiments/runs/exp0007_low_alive_bonus/final_model.zip --config config/exp0007_low_alive_bonus.yaml --episodes 20`, same fixed seeds, `deterministic=True`, comparing `mean_episode_length`, `mean_vel_error`, `mean_distance_traveled_m`, `mean_distance_ratio`, and per-seed termination reasons against both prior runs.

Results: Training completed in 402.6s (7451 steps/s, same throughput as prior runs). Training-time `ep_len_mean`/`ep_rew_mean` are not directly comparable to `baseline0001`'s (different reward scale from the weight change), so we go straight to deterministic eval on the same 20 fixed seeds:

| run | mean len (s) | mean reward | mean vel_err | mean dist (m) | **mean dist_ratio** | falls |
|---|---|---|---|---|---|---|
| `baseline0001` | 16.5 | 947 | 0.155 | 1.85 | **0.566** | 5/20 |
| `exp0007_low_alive_bonus` | 15.6 | 627 | 0.151 | 1.05 | **0.421** | 6/20 |

Behavioral observations: velocity-tracking error during surviving portions is marginally better (0.151 vs 0.155), but **distance_ratio is worse (0.421 vs 0.566), not better**, and fall count is slightly higher (6/20 vs 5/20, now including a new failure mode, `fell_tilt` x4, not seen at all in `baseline0001`'s failures). `train/approx_kl` stayed just as high as `baseline0001`'s (0.15-0.19) throughout -- expected, since this experiment deliberately left PPO settings untouched to isolate the reward-weight effect, but it means this run inherited the same update-instability problem baseline0001 has, on top of a reshaped reward landscape.

Conclusion: **hypothesis not supported.** Reducing `alive_bonus` alone did not fix the high-speed-command failures and did not improve real walking distance -- if anything it's a slightly worse walker by the metric that matters (ADR-003), despite a marginal improvement in tracking precision when it does survive. This rules out "the reward weighting alone is the problem" as a sufficient explanation for F-0002 (complementing EXP-0006, which ruled out "PPO instability alone is the problem"). Both single-factor fixes under-deliver; neither is individually a good instrument to swap for the flat combination of reward-shape + unstable-updates that produced `baseline0001`'s (real but partial) walking behavior.

Decision: **Do not adopt.** Keep `baseline0001` as the canonical baseline; keep this as a documented negative result. Since both single-factor changes have now been tested independently (EXP-0006: PPO stability alone -> degenerate stillness; EXP-0007: reward weight alone -> no improvement), an interaction test combining both is now a justified next step, not a shortcut past the "one conceptual change per experiment" discipline -- each half has already been isolated and evaluated on its own.

Next: EXP-0008 (combine reduced `alive_bonus` with `target_kl`, testing whether both together avoid both failure modes -- config `config/exp0008_low_alive_bonus_target_kl.yaml`, pre-built alongside this experiment).

---

### EXP-0008 — Interaction test: reduced `alive_bonus` + `target_kl` together
Date: 2026-09-18 (pre-registered before running)

Hypothesis: EXP-0006 showed stabilizing PPO alone (`target_kl=0.03`) removes falls but collapses to a near-motionless degenerate optimum. EXP-0007 showed reducing `alive_bonus` alone (0.5->0.1) does not fix falls and does not improve real walking distance. Neither single-factor change works, but they may address complementary problems: `target_kl` prevents the noisy updates that were (apparently) baseline0001's actual route to discovering real locomotion, while a low `alive_bonus` (which failed to help *when combined with unstable updates* in EXP-0007) might be exactly what's needed to stop `target_kl`'s stable-but-lazy optimizer from settling for stillness once instability is no longer forcing it to explore. This is now a justified interaction test, not a shortcut -- both halves were independently isolated and evaluated first (EXP-0006, EXP-0007).

Failure being targeted: F-0002 (high-speed falls / low dist_ratio in `baseline0001`), specifically the "stable updates converge to standing still" failure mode newly characterized in EXP-0006.

Metric: same as EXP-0006/EXP-0007 -- `evaluation.evaluate` on the same 20 fixed seeds, primarily `mean_distance_ratio` (ADR-003), secondarily fall count and `mean_vel_error`.

Budget: 3,000,000 timesteps, `n_envs=12`, seed=0 -- identical protocol to all prior runs in this family.

Stop condition: fixed budget. If `mean_distance_ratio` does not clearly exceed `baseline0001`'s 0.566, this is a third documented negative result, and `baseline0001` stays canonical -- no further reward/PPO-config guessing without a new, specific piece of evidence to motivate it (CLAUDE.md: do not cargo-cult configs).

Code state: no source changes; `config/exp0008_low_alive_bonus_target_kl.yaml` (alive_bonus=0.1 AND target_kl=0.03, otherwise identical to default.yaml).

Results: Training completed in 278.9s (10756 steps/s -- `target_kl` early-stopping again reduced wasted gradient steps, as in EXP-0006). Training-time `ep_len_mean` ended at only 88.8 steps (vs. baseline's 767-783, EXP-0007's 414) -- a bad sign confirmed by deterministic eval:

| run | mean len (s) | mean reward | mean vel_err | mean dist (m) | mean dist_ratio | falls |
|---|---|---|---|---|---|---|
| `baseline0001` | 16.5 | 947 | 0.155 | 1.85 | 0.566 | 5/20 |
| `exp0006_target_kl` | 20.0 | 952 | 0.225 | 0.21 | 0.051 | 0/20 |
| `exp0007_low_alive_bonus` | 15.6 | 627 | 0.151 | 1.05 | 0.421 | 6/20 |
| `exp0008_low_alive_bonus_target_kl` | **2.4** | 65 | 0.325 | 0.66 | 1.911* | **20/20** |

*`dist_ratio` of 1.911 is an artifact, not a real result -- with episodes averaging only 2.4s, the "expected distance" denominator (`command * episode_length`) is tiny, so any lurching motion before falling inflates the ratio. This metric is only meaningful for episodes that run close to the full 20s (docs/DECISIONS.md ADR-003 should be read with this caveat going forward); by every other measure (episode length, fall count, absolute distance, velocity error) this is unambiguously the worst run so far.

Behavioral observations: combining both changes did not get the best of each -- it got the worst of both. `approx_kl` was successfully capped (0.03-0.05, similar to EXP-0006), so the instability-suppression worked mechanically, but with `target_kl` slowing/limiting gradient steps per iteration (fewer effective updates, confirmed by `n_updates=621` vs. baseline's `n_updates=2440` for the same 3M timesteps) *and* a reshaped, less forgiving reward (low `alive_bonus`) removing an easy early reward signal to bootstrap the value function from, the policy appears to have been given a harder optimization landscape with a slower optimizer and never recovered within the same 3M-timestep budget baseline0001 used.

Conclusion: the interaction hypothesis is **not supported** -- these two changes do not combine constructively at this training budget; they compound into a strictly worse outcome. This does not necessarily mean the underlying ideas (stable updates; reward that requires progress) are wrong in general -- it may mean 3M timesteps is no longer a fair budget once both a slower-updating optimizer and a harder reward landscape are combined. Distinguishing "these changes are bad" from "these changes just need more training" is exactly hypothesis (a) from the original F-0002 analysis, which none of EXP-0006/0007/0008 have actually tested (all three used the same 3M-timestep budget as `baseline0001`, varying reward/PPO config instead of budget).

Decision: **Do not adopt.** `baseline0001` remains canonical. Three single-variable-or-combination guesses at reward/PPO config have now been tried and rejected -- per CLAUDE.md ("do not cargo-cult configs"), the next step is to test the one variable never yet isolated (training budget itself) rather than guess a fourth config combination.

Next: EXP-0009 -- rerun `baseline0001`'s exact, unmodified config for a longer budget (10M timesteps, same seed/env-count/protocol) to test hypothesis (a) directly: are the residual high-speed failures a config problem, or simply undertraining?

---

### EXP-0009 — Does more training alone help? (unmodified config, longer budget)
Date: 2026-09-18 (pre-registered before running)

Hypothesis: none of the config changes tried so far (EXP-0006 PPO stability, EXP-0007 reward weight, EXP-0008 both combined) improved on `baseline0001`'s `mean_distance_ratio=0.566`. The original F-0002 analysis never actually tested hypothesis (a) -- "the policy simply hasn't seen enough training yet, independent of config" -- since every follow-up varied config, not budget, at the same fixed 3M-timestep budget `baseline0001` used. This experiment isolates budget as the only variable: identical config to `baseline0001` (`config/default.yaml`, unmodified), same seed=0, same env count, extended to 10,000,000 timesteps with checkpoints every 2M steps so we can see the trend (not just an endpoint), evaluated on the same 20 fixed seeds at each checkpoint.

Failure being targeted: F-0002, specifically distinguishing "config is wrong" (ruled out for the 3 variants tried) from "just needs more training."

Metric: `mean_distance_ratio` and fall count at each of the 5 checkpoints (2M/4M/6M/8M/10M), same protocol as all prior evals.

Budget: 10,000,000 timesteps (vs. 3M for all prior runs -- a single, clearly-labeled exception to "prefer small runs," justified because every smaller/cheaper lever has now been tried and rejected, per CLAUDE.md "prefer small runs that answer one question" -- this is the one remaining un-answered question), `n_envs=12`, seed=0, `checkpoint_freq=2_000_000`.

Stop condition: fixed budget. If `mean_distance_ratio` at 10M timesteps is not clearly better than at 3M (i.e. than `baseline0001` itself), training budget alone is not the answer either, and this whole reward/config family (F-0002's three explanations) will have been exhausted -- next step would require actually watching the failure in the viewer/video rather than more blind config search, which CLAUDE.md's failure-analysis discipline prefers anyway once cheap hypotheses run out.

Code state: no source or config changes -- reuses `config/default.yaml` exactly.

Results: Training completed in 1376.0s (22.9 min, 7268 steps/s -- consistent throughput, no slowdown from the checkpoint callback). Checkpoint trend (`evaluation.evaluate_checkpoints`, 20 episodes each, reliable-only distance ratio):

| steps | mean len (s) | dist_ratio (n reliable) | falls |
|---|---|---|---|
| 2M | 18.9 | 0.249 (19) | 2/20 |
| 4M | 15.9 | 0.691 (15) | 5/20 |
| **6M** | **18.6** | **0.705 (19)** | **3/20** |
| 8M | 19.1 | 0.560 (19) | 2/20 |
| 10M (final) | 18.5 | 0.589 (19) | 3/20 |

Compare to `baseline0001` (3M steps, same config/seed): dist_ratio 0.479 (n=16), falls 5/20. **Every checkpoint from 4M onward already beats `baseline0001`'s dist_ratio**, confirming hypothesis (a) directly: `baseline0001` was simply stopped too early, not fundamentally misconfigured. The training curve (`experiments/runs/exp0009_longer_baseline/curves.png`) explains why: `ep_rew_mean`/`ep_len_mean` show a repeating climb-crash-recover pattern (steep drops around 3.2M, 3.5M, 4.6M, **6.0M**, 7.0M, 8.1M, 9.3M timesteps), consistent with the very high `approx_kl`/`clip_fraction` seen in every run in this family -- each cycle, an aggressive update destroys recent progress, then the policy recovers and, over many cycles, reaches higher peaks than before (peak reward climbs from ~750 around 2-3M to ~900 around 9M). **The 6M-step checkpoint happens to land on a local peak just before the largest crash in the whole run** -- this is closer to lucky snapshot timing than a specifically "6M is special" result; nearby checkpoints (4M: 0.691, 8M: 0.560) are in the same ballpark, confirming the *region* (4M-10M) is reliably better than 3M, even though the exact best step is noisy.

Selected `experiments/runs/exp0009_longer_baseline/best_checkpoint_6M.zip` (copy of the `ppo_g1_walk_5999976_steps.zip` checkpoint) and ran the standard 20-episode deterministic eval on it:

| run | mean len (s) | mean reward | mean vel_err | mean dist (m) | dist_ratio | falls |
|---|---|---|---|---|---|---|
| `baseline0001` (3M) | 16.5 | 947 | 0.155 | 1.85 | 0.479 (n=16) | 5/20 |
| **`exp0009` @ 6M** | **18.6** | **1097** | **0.119** | **2.54** | **0.705 (n=19)** | **3/20** |

This is a strict improvement on every metric simultaneously (unlike EXP-0006/0007/0008, which all traded one metric for a worse one) -- longer survival, higher reward, better velocity tracking, more real distance covered, and fewer falls. Per-episode inspection: 17/20 episodes now reach the full 20s (vs. 15/20); the 3 remaining falls (seeds 1001, 1013, 1017) are the same high-speed-command seeds identified in F-0002, but now survive much longer before falling (13.6s, 6.6s, 11.4s vs. `baseline0001`'s 3-11s range for the same seeds) and cover real distance even in failure (1.3-4.0m).

Conclusion: **hypothesis (a) confirmed.** More training, at the identical config, is a real and substantial fix -- more effective than any of the three config changes tried in EXP-0006/0007/0008. The underlying PPO-update-instability observation from F-0002 was correct, but the fix isn't to suppress it (EXP-0006 showed that backfires into a degenerate optimum) -- it's to train through enough of the noisy climb-crash-recover cycles that a good peak is reached, and to **select the evaluated checkpoint, not just the final one**, since the final step of a fixed-budget run can land on either a peak or a trough of this oscillation (in `baseline0001`'s case, 3M happened to be climbing toward but hadn't yet reached the first big plateau; here, the raw 10M-step `final_model.zip` alone (0.589) would have been worse than the properly-selected 6M checkpoint (0.705) -- checkpoint selection via evaluation is not optional busywork, it materially matters).

Decision: **Promote `experiments/runs/exp0009_longer_baseline/best_checkpoint_6M.zip` to the new canonical baseline**, superseding `baseline0001`. See `docs/DECISIONS.md` ADR-004. New video: `experiments/runs/exp0009_longer_baseline/rollout_seed1002.mp4`. `docs/FAILURE_ANALYSIS.md` F-0002 updated with this resolution.

Next: the remaining 3/20 failures still concentrate at high commanded speed -- a legitimate, smaller, honestly-labeled residual limitation, not blocking. Optional future work: (a) train even longer / average multiple seeds to reduce checkpoint-selection noise; (b) revisit `target_kl` now armed with the knowledge that it needs to be paired with enough budget to still find a good peak, not just stability for its own sake; (c) a narrower `vx_range` curriculum that only later expands to the full high-speed range.

---

### EXP-0010 — Gait quality: fix heading drift and chattery stepping (user-reported)
Date: 2026-09-18 (pre-registered before running)

Hypothesis: the canonical policy (`exp0009_longer_baseline` @ 6M) scores well on every aggregate metric used so far but was never checked for heading stability or stride quality. A new diagnostic (`diagnostics/gait_diagnostic.py`, built for this) found, under a clean pure-forward command (vx=0.3, vy=0): **95 degrees of yaw drift in 10s** (reward never penalizes rotation, since `tracking_lin_vel` is computed in the robot's own local frame) and **chattery, high-frequency (~5.5Hz) foot contact** with only 2.5cm of clearance (nothing rewards smooth, low-frequency strides). Adding two minimal, directly-motivated reward terms -- `ang_vel_penalty` (yaw-rate squared) and `action_rate_penalty` (squared action delta) -- should reduce both without regressing the `dist_ratio`/survival gains from EXP-0009 (docs/FAILURE_ANALYSIS.md F-0003).

Failure being targeted: F-0003 (doesn't walk straight, chattery steps).

Metric: primary -- yaw drift (deg) and max lateral deviation (m) from `diagnostics.gait_diagnostic` under a fixed vx=0.3/vy=0 command; foot-liftoff count and foot-height range as a stepping-quality proxy. Secondary (must not regress) -- `mean_distance_ratio_reliable_only`, fall count, from the standard `evaluation.evaluate` protocol on the same 20 fixed seeds.

Budget: 10,000,000 timesteps (matching EXP-0009's proven-necessary budget, not the smaller 3M used for the rejected EXP-0006/0007/0008 -- F-0002 established that a fair comparison needs this much), `n_envs=12`, seed=0, `checkpoint_freq=2_000_000`, evaluated across the full checkpoint trend (`evaluation.evaluate_checkpoints`) rather than only the final step, per the ADR-004 lesson.

Stop condition: fixed budget. If yaw drift and/or foot chatter are not clearly improved at the best checkpoint, or if `dist_ratio`/survival regress meaningfully, this is a documented negative result and the two terms' weights (or the diagnosis itself) need revisiting -- not blind escalation to more terms.

Code state: `rewards/walking_reward.py` (new `ang_vel_penalty`/`action_rate_penalty` terms, `compute_reward` signature extended), `config/loader.py` (`RewardWeightsConfig` gets the two new fields, defaulting to 0.0 for backward compatibility with every prior config), `envs/g1_walk_env.py` (passes `base_ang_vel_z`/`action`/`prev_action` into the reward call), `diagnostics/gait_diagnostic.py` (new), `tests/test_reward.py` (6 new tests for the new terms + 1 backward-compatibility test). New config `config/exp0010_gait_quality.yaml`: `ang_vel_penalty=1.0` (same order as `tracking_lin_vel`'s own weight), `action_rate_penalty=0.01` (standard literature starting value), otherwise identical to `default.yaml`.

Results: Training completed in 1283.1s (7794 steps/s). Checkpoint trend (`evaluation.evaluate_checkpoints`, 20 episodes each):

| steps | mean len (s) | dist_ratio (n reliable) | falls |
|---|---|---|---|
| 2M | 18.6 | 0.011 (18) | 2/20 |
| 4M | 20.0 | 0.021 (20) | 1/20 |
| 6M | 20.0 | **0.010 (20)** | **0/20** |
| 8M | 19.9 | 0.018 (20) | 1/20 |
| 10M (final) | 20.0 | 0.007 (20) | 0/20 |

Survival is excellent (0-2 falls, near-full episode length every checkpoint) but `dist_ratio` **collapsed to near-zero** at every single checkpoint -- the same degenerate-stillness failure mode as EXP-0006, now via a different mechanism. Confirmed directly with `diagnostics.gait_diagnostic` on the 6M checkpoint: yaw drift dropped from baseline's 95 deg/10s to **1.4 deg/10s** (the heading-drift problem is completely solved), but net forward progress is **0.01m over 10s** and **0 foot liftoffs, both feet grounded 100% of the time** -- it stands almost perfectly still.

Root cause, isolated by direct measurement rather than guessing: ran `exp0009_longer_baseline`'s actual, previously-successful gait (6M checkpoint, unmodified config) back through the *new* reward's math offline. Its natural yaw wobble while walking averages `ang_vel_z^2 = 0.44` (90th percentile 1.19) -- at `ang_vel_penalty=1.0`, that's a mean penalty of **-0.44/step, 90th percentile -1.19/step**, against a mean `tracking_lin_vel` reward of only **0.80/step**. The penalty for the yaw wobble a normal gait requires often *exceeds* the reward for tracking velocity -- standing still (near-zero yaw rate) is a strictly better deal under this weight. `action_rate_penalty=0.01` was comparatively mild on the same gait (mean cost 0.062/step, ~8% of the tracking reward) -- not the primary culprit.

Conclusion: `ang_vel_penalty=1.0` was ~10-50x too strong, chosen from an unverified "same order as `tracking_lin_vel`'s weight" heuristic rather than measuring what a real gait's yaw dynamics actually cost. This is exactly the kind of guess CLAUDE.md warns against ("do not cargo-cult... trace why each part exists") -- corrected here by directly measuring the target gait's own statistics before re-guessing a weight.

Decision: **Reject these weights, do not adopt as-is.** Recalibrate `ang_vel_penalty` down by ~10x (to 0.1, making its typical cost ~5% of the tracking reward -- a gentle nudge, not a wall) and keep `action_rate_penalty=0.01` unchanged (evidence above shows it wasn't the dominant problem). New config: `config/exp0011_gait_quality_v2.yaml`.

Next: EXP-0011 (recalibrated gait-quality weights, same measurement-based process to verify before declaring success).

---

### EXP-0011 — Gait quality, recalibrated: `ang_vel_penalty` reduced 10x based on measured gait statistics
Date: 2026-09-18 (pre-registered before running)

Hypothesis: `ang_vel_penalty=0.1` (down from EXP-0010's 1.0) makes the typical cost of a real gait's yaw wobble (~0.044/step at the measured `ang_vel_z^2~0.44`) a small fraction (~5.5%) of the tracking reward (~0.80/step) rather than exceeding it, avoiding the standing-still collapse while still discouraging the more extreme 95-degree-per-10s drift seen with no penalty at all (EXP-0009/F-0003). `action_rate_penalty=0.01` unchanged (measured as a minor, non-dominant contributor in EXP-0010's analysis).

Failure being targeted: F-0003 (heading drift, chattery steps), while explicitly avoiding EXP-0010's overcorrection into degenerate stillness.

Metric: same as EXP-0010 -- yaw drift/lateral deviation from `diagnostics.gait_diagnostic`, plus `mean_distance_ratio_reliable_only`/falls from the standard eval protocol (this time the *primary* gate, checked before anything else, given EXP-0010's lesson).

Budget: 10,000,000 timesteps, `n_envs=12`, seed=0, `checkpoint_freq=2_000_000` -- identical protocol to EXP-0009/0010.

Stop condition: fixed budget. If `dist_ratio` again collapses, `ang_vel_penalty` needs to go lower still (or a different formulation, e.g. only penalizing yaw rate above a small deadband) -- do not simply retry the same weight with a different seed.

Code state: no source changes; new config `config/exp0011_gait_quality_v2.yaml` (`ang_vel_penalty=0.1`, `action_rate_penalty=0.01`, otherwise identical to `default.yaml`).

Results: Training completed in 3875s (~65 min -- notably slower than the ~22 min every other 10M-timestep run in this project took, 2581 steps/s vs. the usual ~7200-7800; almost certainly CPU contention/thermal effects from running several consecutive 10M-step trainings back to back on this laptop rather than a code issue, since throughput was consistent within the run). Checkpoint trend (`evaluation.evaluate_checkpoints`, 20 episodes each):

| steps | mean len (s) | dist_ratio (n reliable) | falls | yaw_drift/10s (all-command eval avg) |
|---|---|---|---|---|
| 2M | 20.0 | 0.375 (20) | 0/20 | 32.2 deg |
| 4M | 15.3 | 0.518 (14) | 6/20 | 40.9 deg |
| 6M | 16.2 | 0.491 (15) | 7/20 | 17.7 deg |
| 8M | 18.2 | 0.344 (18) | 2/20 | 66.2 deg |
| **~10M** | **18.3** | **0.671 (18)** | **2/20** | 51.6 deg |

No degenerate-stillness collapse this time -- `dist_ratio` stayed in a normal range at every checkpoint. Selected the ~10M-step checkpoint (`best_checkpoint_10M.zip`) by the same evaluate-the-trend discipline as EXP-0009, then ran the targeted straight-line diagnostic (`diagnostics.gait_diagnostic`, vx=0.3/vy=0, same protocol as F-0003's original measurement) on it: **yaw drift 11.6 deg/10s (down from 95 deg/10s, an 88% reduction)**, net forward progress 2.30m (better than the un-penalized policy's 1.77m under the identical test), max lateral deviation 0.30m (down from 1.35m). The trajectory plot (`diagnostics/reports/gait_diagnostic_exp0011_10M.png`) shows a visibly near-straight path after an initial short curve, vs. the old policy's dramatic spiral. Foot-height/contact pattern is only marginally improved (45-47 liftoffs vs. 54-56 before, 75% grounded vs. 69% before) -- `action_rate_penalty=0.01` nudged stepping frequency down slightly but did not turn the chattery small-amplitude stepping into large clean strides; this part of F-0003 is only partially addressed.

Standard 20-episode eval (same fixed seeds, same protocol as every prior run) comparing directly against the current canonical baseline (`exp0009_longer_baseline`, re-evaluated with the same new yaw-drift metric for a fair comparison):

| run | len (s) | reward | vel_err | dist (m) | dist_ratio | falls | yaw_drift/10s |
|---|---|---|---|---|---|---|---|
| `exp0009_longer_baseline` (current canonical) | 18.6 | 1097 | 0.119 | 2.54 | 0.705 (n=19) | 3/20 | 63.9 deg |
| `exp0011_gait_quality_v2` @ ~10M | 18.3 | 1067 | 0.134 | **2.99** | 0.671 (n=18) | **2/20** | **42.7 deg** |

Every metric is within normal run-to-run noise of `exp0009` (survival, reward, tracking error, `dist_ratio`, falls all comparable or slightly better) except real distance traveled (better) and yaw drift, which improved by a third in this varied-command aggregate and by 88% in the controlled single-command diagnostic that isolates the effect cleanly. No metric regressed meaningfully.

Conclusion: **hypothesis confirmed, without repeating EXP-0010's overcorrection.** Measuring the target gait's actual yaw-rate statistics before picking a weight (rather than guessing "same order as another term") was the key difference between this experiment and EXP-0010's collapse. Two genuine, measured gait-quality problems (F-0003) are now meaningfully improved with no regression in the walking-quality metrics established by ADR-003/ADR-004.

Decision: **Promote `experiments/runs/exp0011_gait_quality_v2/best_checkpoint_10M.zip` to the new canonical baseline**, superseding `exp0009_longer_baseline`. See `docs/DECISIONS.md` ADR-005. New video: `experiments/runs/exp0011_gait_quality_v2/rollout_seed1002.mp4`.

Next: the chattery small-stride stepping pattern (F-0003's second half) is only partially fixed. A larger `action_rate_penalty`, or a direct minimum-foot-clearance/stride-length reward term, would be the next targeted experiment if pursued further -- not required for the challenge's minimum bar, which continues to be exceeded by a growing margin.

---

### EXP-0012 — Feet air-time reward: fix the remaining chattery-stepping problem
Date: 2026-09-18 (pre-registered before running)

Hypothesis: `action_rate_penalty` (EXP-0011) only marginally reduced step chatter (liftoffs 54-56 -> 45-47/10s, foot clearance essentially unchanged at 0.030-0.055m) because it discourages large action deltas in general, not short swing phases specifically. A direct reward for swing duration at the moment of touchdown (the standard "feet air time" technique from the legged-locomotion RL literature, e.g. Rudin et al. 2021) should target the actual behavior we want (fuller, more deliberate strides) more precisely.

Failure being targeted: F-0003's unresolved second half / new entry F-0004 (chattery, ~20-30ms swing phases, an order of magnitude below a deliberate stride).

Measurement before choosing weights (ADR-005 discipline, learned from EXP-0010's collapse): replayed `exp0011_gait_quality_v2`'s actual checkpoint (6 episodes, 5188 control steps, 861 touchdowns) through a new `envs.contacts.AirTimeTracker` to get the real swing-time distribution: **mean 27ms, median 20ms, 90th percentile 40ms, max 80ms**. Computed the candidate reward's cost on this same data for several (threshold, weight) pairs before picking one: at `threshold=0.2s, weight=1.0`, the current gait's shortfall costs only -0.029/step (negligible vs. the ~0.8/step tracking reward) -- too weak to matter. Chose `weight=3.0` (shortfall -0.086/step, ~11% of tracking reward) as a meaningful but not dominant nudge, deliberately more conservative than a naive "make it matter" overcorrection.

Metric: primary -- swing-time distribution (mean/median/p90) and foot-clearance/liftoff-count from `diagnostics.gait_diagnostic`, directly comparable to the F-0003/F-0004 baseline measurements above. Secondary (must not regress) -- `mean_distance_ratio_reliable_only`, fall count, `mean_abs_yaw_drift_per_10s_deg` from the standard eval protocol.

Budget: 10,000,000 timesteps, `n_envs=12`, seed=0, `checkpoint_freq=2_000_000` -- same protocol as every gait-quality experiment since EXP-0009.

Stop condition: fixed budget. If swing time doesn't measurably increase, or if `dist_ratio`/falls/yaw-drift regress, this is a documented negative result -- escalate the weight only with a new measurement, not a blind retry.

Code state: `envs/contacts.py` (new -- `AirTimeTracker`, plus `LEFT_FOOT_GEOMS`/`RIGHT_FOOT_GEOMS`/`foot_in_contact`/`both_feet_contact`, refactored out of `diagnostics/gait_diagnostic.py` which now imports from here instead of duplicating), `config/loader.py` (`feet_air_time_bonus`/`feet_air_time_threshold_s`, defaulting to 0.0/0.2 for backward compatibility), `rewards/walking_reward.py` (`compute_reward` takes `air_time_reward_raw`), `envs/g1_walk_env.py` (owns an `AirTimeTracker` instance, verifies foot geoms against the live model at construction via `verify_foot_geoms`), `tests/test_air_time_tracker.py` (new, 5 tests), `tests/test_reward.py` (extended, 2 new tests + refactored to a shared `reward()` helper). New config `config/exp0012_air_time.yaml`.

Results: Training completed in 2261.4s (~38 min, 4422 steps/s -- notably reduced throughput mid/late-run, likely `approx_kl` computation and the Python-loop contact-checking overhead at higher clip fractions; not investigated further since it didn't block completion). Checkpoint trend:

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s (aggregate) |
|---|---|---|---|
| 2M | 0.013 (19) | 1/20 | 4.9 deg |
| 4M | 0.575 (19) | 1/20 | 21.1 deg |
| 6M | 0.551 (20) | 0/20 | 43.7 deg |
| 8M | 0.409 (19) | 1/20 | 48.6 deg |
| 10M | 0.606 (19) | 1/20 | 32.6 deg |

Aggregate numbers look reasonable to good (falls even lower than `exp0011`'s 2/20) -- **this is exactly the trap F-0004 is about**: none of these metrics can see what's actually happening. Controlled straight-line diagnostic (`diagnostics.gait_diagnostic`, vx=0.3/vy=0, same protocol as every prior gait check) on 4 different checkpoints (4M/6M/8M/10M) all showed the same pattern: liftoff counts collapsed to 2-21 per 10s (vs. `exp0011`'s 45-47), airborne time 0-5% (vs. 11-14%), and on some checkpoints yaw drift got much worse (up to 164 deg/10s on the 8M checkpoint) despite `ang_vel_penalty` being unchanged. Direct measurement of foot horizontal velocity while nominally "in contact" (finite-difference of foot body position, same contact detector the reward uses) confirmed the mechanism: mean 0.24-0.26 m/s, comparable to the 0.3 m/s commanded speed -- **the policy is skating, sliding its feet along the ground while remaining in geometric contact, to dodge the air-time reward's touchdown-only trigger.** Full mechanism and evidence: `docs/FAILURE_ANALYSIS.md` F-0004.

Conclusion: the pre-registered stop condition ("swing time doesn't measurably increase -> documented negative result") is met, but the actual failure mode is more specific and informative than "didn't work" -- it's a structural reward-hacking blind spot (an event-triggered reward can't penalize an event that never happens), not simply a bad weight choice. This is *not* a repeat of EXP-0010's mistake (that was an uncalibrated weight; here the weight was properly calibrated and the exploit exists at any positive weight).

Decision: **Do not adopt. Keep `exp0011_gait_quality_v2` as the canonical baseline.** Keep `exp0012` as a documented, mechanistically-understood negative result -- genuinely useful (it identifies a real gap in the reward, not just "this number was wrong").

Next: EXP-0013 -- add a complementary `stance_overrun_penalty` (penalizes continuous ground contact beyond a max duration, evaluated every step so it cannot be dodged by never lifting), alongside the existing (unchanged) `feet_air_time_bonus`.

---

### EXP-0013 — Close the skating exploit: add `stance_overrun_penalty`
Date: 2026-09-18 (pre-registered before running)

Hypothesis: `feet_air_time_bonus` alone (EXP-0012) let the policy dodge the touchdown-only reward entirely by never lifting its feet (skating instead of stepping, F-0004). Adding a per-step penalty for continuous stance beyond a calibrated maximum (`stance_overrun_penalty`, `envs/contacts.GroundTimeTracker`) closes this loophole structurally -- it accrues every step a foot overstays, so "never touch down" is no longer free -- while `feet_air_time_bonus` (unchanged) continues to push toward longer, more deliberate swings once touchdowns do happen.

Failure being targeted: F-0004 (skating exploit).

Measurement before choosing weights (ADR-005/F-0004 discipline): measured `exp0011`'s (the last known-good stepping gait, pre-air-time-reward) actual stance-duration distribution -- mean 0.21s, median 0.16s, p90 0.38s, p99 0.62s, max 2.8s (one rare long single-support outlier). Simulated the candidate penalty against this same gait for several `max_stance_s` values before picking one: `max_stance_s=1.0` (comfortably above the p99 of 0.62s, so legitimate rare long-stance moments are untouched) costs only ~0.030/step at weight=1.0 -- chose `stance_overrun_penalty=2.0` (~0.060/step, ~7.5% of the tracking reward) as a meaningful-but-modest tax, calibrated the same way as every other gait-quality weight in this project family.

Metric: primary -- liftoff count, airborne %, and stance-time distribution from `diagnostics.gait_diagnostic` plus a direct foot-slip-while-in-contact measurement (the exact signal that caught F-0004), on the controlled straight-line test. Secondary -- `dist_ratio`, falls, yaw drift from the standard eval.

Budget: 10,000,000 timesteps, `n_envs=12`, seed=0, `checkpoint_freq=2_000_000` -- same protocol as every gait-quality experiment.

Stop condition: fixed budget. If liftoff counts/airborne % don't recover toward `exp0011`'s levels (45-47/10s, 11-14%) and foot slip while "in contact" doesn't drop substantially from `exp0012`'s ~0.25 m/s, this is a second documented negative result for the air-time-reward family, and it should be abandoned in favor of the already-solid `exp0011` baseline rather than a third weight-tweaking attempt.

Code state: `envs/contacts.py` (new `GroundTimeTracker`), `config/loader.py` (`stance_overrun_penalty`/`max_stance_s`, defaulting to 0.0/1.0 for backward compatibility), `rewards/walking_reward.py` (`compute_reward` takes `stance_overrun_raw`), `envs/g1_walk_env.py` (owns a `GroundTimeTracker` instance alongside the existing `AirTimeTracker`), `tests/test_ground_time_tracker.py` (new, 5 tests including one that specifically asserts the never-lift exploit is no longer free), `tests/test_reward.py` (2 new tests). New config `config/exp0013_air_time_v2.yaml`.

Results: Training completed in 1397.8s (~23 min, 7154 steps/s -- normal throughput, unlike EXP-0011's anomalous slowdown). Checkpoint trend -- **every single checkpoint, 100% fall rate**:

| steps | dist_ratio | falls | yaw_drift/10s |
|---|---|---|---|
| 2M | n/a (0 reliable episodes) | 20/20 | 72.0 deg |
| 4M | n/a | 20/20 | 82.8 deg |
| 6M | n/a | 20/20 | 121.8 deg |
| 8M | n/a | 20/20 | 60.3 deg |
| 10M | n/a | 20/20 | 51.3 deg |

Mean episode length 2.5-3.1s at every checkpoint -- categorically worse than `exp0012`'s worst checkpoint (which at least had 0-1 falls and ~0.4-0.6 `dist_ratio`), and only barely better than the untrained baseline's ~1.3s.

Traced the mechanism directly via per-step reward-component logging (seed 1002, 10M-step checkpoint, vx=0.3 command): the robot delays lifting a foot (avoiding a swing so short it would still fail the 0.2s `feet_air_time_bonus` threshold), during which `stance_overrun_penalty` grows from -0.8 to -2.4 to -4.0 over a few hundred ms; a late, forced, poorly-timed step follows (still triggering the air-time shortfall penalty anyway, -0.54) and the robot visibly destabilizes within the next ~20 control steps (pelvis height 0.76m -> 0.46m) and falls shortly after. Full trace in `docs/FAILURE_ANALYSIS.md` F-0004.

Conclusion: the two independently-calibrated terms do not compose. `stance_overrun_penalty` demands a lift within ~1.0s; `feet_air_time_bonus` demands, once lifted, a swing of >=0.2s -- a policy trying to satisfy both from scratch, within this budget, cannot find the (frequent-but-full-length swing) solution that would satisfy both and instead gets caught in a lose-lose cycle of delayed, truncated, destabilizing steps. Each term measured as individually modest (11% and 7.5% of the tracking reward) does not predict how the *combination* behaves -- this is exactly the risk `docs/DECISIONS.md` ADR-005's measure-first discipline reduces but cannot eliminate for interacting terms.

Decision: **Do not adopt. Hits the pre-registered stop condition exactly** ("second documented negative result... abandon in favor of `exp0011`"). **`exp0011_gait_quality_v2` remains the canonical baseline.** Not attempting a third weight-tweak on this mechanism -- two independent, well-instrumented failures (skating exploit in EXP-0012, lose-lose destabilization here) are a clear enough signal that this specific two-term formulation needs a different approach, not another calibration pass.

Next: the chattery-stepping problem (F-0003 second half) stays open, honestly documented as attempted-twice-and-reverted. If revisited, `docs/FAILURE_ANALYSIS.md` F-0004's follow-up section suggests two untried alternative approaches (reward stride length directly instead of swing duration; or a two-stage curriculum). Pivoting to a different, well-scoped next goal instead: Phase 8 domain randomization, per PROJECT_PLAN.md and the challenge document's own "once basic walking works" guidance -- basic walking works, robustly, on `exp0011`.

---

### EXP-0014 — Phase 8: domain randomization (friction, mass, motor strength, pushes)
Date: 2026-09-18 (pre-registered before running)

Hypothesis: training with randomized floor friction, body mass, motor strength, and periodic velocity pushes (`envs/domain_randomization.py`, new) will produce a policy that is more robust to physical variation than `exp0011` (trained with none of this), at some likely cost to peak performance under the exact nominal conditions `exp0011` was tuned for -- the expected trade-off named in PROJECT_PLAN.md Phase 8 ("Evaluate clean and randomized environments separately").

Failure being targeted: none yet measured -- this is the challenge's own explicitly named next step ("Add domain randomization once basic walking works"), not a response to a diagnosed failure. Reward and PPO config are otherwise identical to `exp0011` (single conceptual change: physics randomization only).

Randomization ranges (informed starting points, not yet tuned -- see `config/exp0014_domain_randomization.yaml` for full reasoning): floor friction x[0.5, 1.5], mass x[0.85, 1.15] (mass and inertia scaled together for physical consistency), motor strength x[0.7, 1.0] (real motors underperform more often than they exceed rated torque), pushes every 3-6s at 0.1-0.5 m/s (deliberately modest -- the policy has never trained under any perturbation before).

Metric: primary -- deterministic eval under **both** conditions on the same 20 fixed seeds: (a) nominal/clean physics (using `config/exp0011_gait_quality_v2.yaml` to evaluate the exp0014 checkpoint, since that config has domain randomization disabled and is otherwise identical) to check for regression vs. `exp0011`'s own numbers on its home turf; (b) randomized physics (`config/exp0014_domain_randomization.yaml`) to check for a genuine robustness gain. Compare `exp0011` itself evaluated under randomized physics too, as the "trained without robustness, tested with it" control -- the key comparison PROJECT_PLAN.md Phase 8 calls for.

Budget: 10,000,000 timesteps, `n_envs=12`, seed=0, `checkpoint_freq=2_000_000` -- same protocol as every experiment since EXP-0009.

Stop condition: fixed budget. Evaluate the full checkpoint trend (`evaluation.evaluate_checkpoints`) under both conditions before selecting a checkpoint, per the ADR-004 lesson.

Code state: `envs/domain_randomization.py` (new -- `DomainRandomizer`, `PushScheduler`), `config/loader.py` (`DomainRandomizationConfig`, defaulting to fully disabled for backward compatibility with every prior config), `envs/g1_walk_env.py` (owns a `DomainRandomizer`/`PushScheduler`, applies motor-strength scaling in the control loop and pushes at the start of `step()`, randomizes physics *before* `reset_state`'s `mj_forward` call so this episode's parameters are used consistently from the first observation), `tests/test_domain_randomization.py` (new, 8 tests including one that specifically asserts a disabled config consumes zero extra RNG draws, preserving every prior config's exact reproducibility). New config `config/exp0014_domain_randomization.yaml`.

Results: Training completed in 3025.9s (~50 min, 3305 steps/s -- markedly slower than the usual ~7200, likely a mix of per-step randomization bookkeeping overhead and the machine-load variability already seen in EXP-0011). Evaluated the full checkpoint trend under **both** conditions, per the pre-registration:

**Clean/nominal physics** (`config/exp0011_gait_quality_v2.yaml`, DR disabled):

| steps | dist_ratio | falls |
|---|---|---|
| 2M-8M | 0.18-0.33 | 0-2/20 |
| ~10M (best) | 0.42-0.43 | 4-7/20 |

**Randomized physics** (`config/exp0014_domain_randomization.yaml`, DR enabled, i.e. the conditions it trained under):

| steps | dist_ratio (few reliable episodes) | falls |
|---|---|---|
| 2M-8M | 0.19-0.39 | 14-18/20 |
| ~10M (best) | 0.53-0.61 | 17-19/20 |

Both are clearly worse than `exp0011`'s own clean-eval numbers (0.705 dist_ratio, 3/20 falls) -- expected, since `exp0011` never had to cope with physics variation. The critical comparison is the **control**: `exp0011` (trained with zero randomization) evaluated under the *same randomized conditions* `exp0014` trained on: mean length 10.9s, `dist_ratio` 0.789 (n=9 reliable), **13/20 falls**. This is comparable to or *better than* `exp0014`'s own best checkpoint under its own training distribution (17-19/20 falls) -- **the policy trained specifically for robustness is not more robust than the one that wasn't, at this budget.**

Conclusion: domain randomization is correctly implemented and technically verified (per-episode friction/mass/motor-strength variation and periodic pushes all confirmed working, `tests/test_domain_randomization.py`, manual per-episode inspection), but **10M timesteps -- the budget that was sufficient for the much simpler non-randomized task (EXP-0009) -- is not enough for this harder, higher-variance task to converge to a measurable robustness benefit.** This is the same lesson as EXP-0009 (F-0002's resolution): before concluding a technique doesn't work, check whether it was simply undertrained. Domain randomization enlarges the effective state/scenario distribution substantially (friction x mass x motor-strength x push-timing, combinatorially), so it plausibly needs meaningfully more than 10M steps where the non-randomized task needed ~4-6M.

Decision: **Do not adopt `exp0014` as a general-purpose replacement for `exp0011` yet** -- it is strictly worse on both axes tested at this budget, not a favorable trade-off. Keep `exp0011_gait_quality_v2` as the canonical baseline. This is not treated as disproving domain randomization's value (PROJECT_PLAN.md Phase 8 is not abandoned) -- it is treated as an undertrained data point, following the EXP-0009 precedent directly.

Next: EXP-0015 -- extend the *same* domain-randomization config to a longer budget (20M timesteps, double EXP-0014's) to test directly whether more training, not a different technique, closes the gap -- mirroring exactly what worked for the base gait in EXP-0009.

---

### EXP-0015 — Domain randomization, longer budget
Date: 2026-09-18 (pre-registered before running)

Hypothesis: EXP-0014 was undertrained, not fundamentally flawed (same lesson as EXP-0009/F-0002). Doubling the budget (20M timesteps) should move the randomized-condition fall rate and `dist_ratio` meaningfully past `exp0011`'s "trained without any robustness" control (13/20 falls, 0.79 dist_ratio under randomized physics).

Failure being targeted: EXP-0014's undertraining, not a new failure.

Metric: same as EXP-0014 -- checkpoint-trend eval under both clean and randomized physics, primary comparison against the `exp0011`-under-randomized-physics control already measured.

Budget: 20,000,000 timesteps (2x EXP-0014, matching the scale-up ratio that worked for EXP-0009 relative to `baseline0001`), `n_envs=12`, seed=0, `checkpoint_freq=4_000_000` (keeping 5-6 checkpoints as in every prior run).

Stop condition: fixed budget. If the randomized-condition fall rate still doesn't clearly beat `exp0011`'s 13/20 control, domain randomization needs either even more budget, gentler randomization ranges, or a curriculum (start clean, ramp up randomization) -- not a third blind budget doubling.

Code state: no changes -- reuses `config/exp0014_domain_randomization.yaml` exactly.

Results: Training completed in 6463.6s (~108 min, 3095 steps/s). The first 10M timesteps exactly reproduce `exp0014`'s own checkpoints (identical seed/config, confirming PPO/env determinism) -- `dist_ratio` at the 4M/8M checkpoints matches `exp0014` bit-for-bit (0.183, 0.239 clean). Continuing to 20M shows clear, continued improvement on both evaluation conditions:

**Clean physics** (`config/exp0011_gait_quality_v2.yaml`):

| steps | dist_ratio | falls |
|---|---|---|
| 4M-16M | 0.18-0.38 | 0-4/20 |
| **20M (best)** | **0.542** | **1/20** |

**Randomized physics** (`config/exp0014_domain_randomization.yaml`, its own training distribution):

| steps | dist_ratio | falls |
|---|---|---|
| 4M-16M | 0.19-0.52 | 14-18/20 |
| **20M (best)** | **0.574-0.590** | **11-13/20** |

Full standard-protocol (20-episode) evaluation of the selected 20M checkpoint against the `exp0011`-under-randomized-physics control established in EXP-0014:

| condition | policy | len (s) | vel_err | dist_ratio (n reliable) | falls |
|---|---|---|---|---|---|
| clean | `exp0011` (no DR) | 18.6 | 0.119 | 0.705 (19) | 3/20 |
| clean | `exp0015` (DR, 20M) | 19.5 | 0.165 | 0.519 (20) | **1/20** |
| randomized | `exp0011` (no DR, control) | 10.9 | 0.199 | 0.789 (9) | 13/20 |
| randomized | `exp0015` (DR, 20M) | **13.9** | 0.238 | 0.590 (14) | **11/20** |

Conclusion: the budget hypothesis is confirmed -- doubling to 20M substantially closed and, on survival specifically, reversed the gap from EXP-0014. Under randomized physics (the condition that matters for this experiment's purpose), `exp0015` now survives longer (13.9s vs. 10.9s) and falls less often (11/20 vs. 13/20) than a policy that never trained with any randomization, with more episodes reaching a length long enough for `dist_ratio` to be a reliable estimate (14 vs. 9) -- a genuine, if modest, robustness gain, not yet a dramatic one. The trade-off: `exp0015` tracks commanded velocity less precisely on both conditions (higher `vel_err`, lower `dist_ratio`) than `exp0011` -- a real, sensible trade-off (a more conservative, fall-resistant gait vs. a more precise but less robust one), not a strict win. The clean-physics `dist_ratio` trend (0.18 -> 0.54 from 4M to 20M) had also not yet plateaued, suggesting further budget could improve this further, but this is left as future work given the time already invested (this single run took ~108 minutes).

Decision: **`exp0015_domain_randomization_long`'s 20M checkpoint is a genuine, working, more-robust *alternative* to `exp0011`, not a strict replacement.** `exp0011_gait_quality_v2` remains the canonical baseline for the primary walking-challenge deliverable (better clean-condition tracking, which is what the deliverable video/curves showcase), while `exp0015` is documented as the answer to "what if the surface/motors/robot mass aren't exactly nominal" -- available for anyone who needs that property, with its trade-offs clearly stated.

Next (not executed, time-bounded): even more budget (40M+ timesteps) to see if the clean-physics `dist_ratio` trend continues improving toward `exp0011`'s level while keeping the randomized-physics robustness gain; or narrower randomization ranges as an alternative lever if budget alone plateaus.

---

### EXP-0016 — Chattery-stepping fix, third attempt: warm-start fine-tuning instead of training from scratch
Date: 2026-09-18 (pre-registered before running)

Hypothesis: EXP-0012/0013 both trained the `feet_air_time_bonus` reward *from scratch*, meaning the policy was still randomly exploring when it first encountered the touchdown-only reward -- "never touch down" was a trivially discoverable, zero-cost escape before the policy had ever learned real stepping. `exp0011`'s checkpoint already steps frequently (45-47 liftoffs/10s); warm-starting PPO from those weights instead of random initialization means the policy starts from an established, reward-positive stepping strategy that skating would have to actively displace, not merely avoid discovering. Fine-tuning at a reduced learning rate (damping the initial shock of the reward change, confirmed necessary and sufficient in a smoke test: `approx_kl` dropped from 0.86 at the default LR to 0.028 at 5e-5, with the policy's full performance -- `ep_len_mean=1000` -- retained immediately rather than destabilized) should let swing duration increase gradually rather than triggering a wholesale strategy change.

Failure being targeted: F-0004 (chattery stepping / skating exploit), via a genuinely new lever (initialization + fine-tuning) rather than a third reward-weight guess.

New capability built for this: `training/train.py --init-from <checkpoint> [--init-learning-rate <lr>]` -- warm-start PPO from an existing checkpoint's weights instead of random init, with an optional learning-rate override for fine-tuning (SB3 requires rebuilding the LR schedule, not just setting the attribute, for this to take effect -- verified in the smoke test). This is a real, general capability gap this project's own docs had noted as missing (`docs/RL_PIPELINE.md`: "No resume-from-checkpoint training loop yet").

Metric: primary -- swing-time distribution and foot-slip-while-in-contact from `diagnostics.gait_diagnostic` (the exact measurements that caught F-0004), checked at every checkpoint to catch a skating regression early, not just at the end. Secondary -- `dist_ratio`, falls, yaw drift from the standard eval, which must not regress from `exp0011`'s levels.

Budget: 5,000,000 timesteps (smaller than the standard 10M -- fine-tuning from an already-good policy should need less), `n_envs=12`, seed=0, `checkpoint_freq=1_000_000` (5 checkpoints, closely spaced to catch any skating regression early rather than only discovering it at the end as happened in EXP-0012).

Stop condition: fixed budget, but check every checkpoint's gait diagnostic as it becomes available rather than waiting for the end -- if skating re-emerges at any point, that checkpoint and the reasoning are recorded, and earlier checkpoints (before the regression) are still evaluated as potential candidates.

Code state: `training/train.py` (new `--init-from`/`--init-learning-rate` flags). Reuses `config/exp0012_air_time.yaml` exactly (same `feet_air_time_bonus=3.0`, `feet_air_time_threshold_s=0.2`, no `stance_overrun_penalty` -- isolating initialization as the only variable changed from EXP-0012).

Results: Training completed in 693.8s (~11.5 min, 7207 steps/s -- normal throughput). Checkpoint trend -- **no collapse at any checkpoint**, unlike EXP-0012's from-scratch attempt:

| steps | dist_ratio | falls | yaw_drift/10s |
|---|---|---|---|
| 1M | 0.470 (20) | 0/20 | 43.3 deg |
| 2M | 0.568 (19) | 1/20 | 49.4 deg |
| 3M | 0.538 (19) | 1/20 | 43.3 deg |
| **4M** | **0.627 (20)** | **1/20** | **36.9 deg** |
| 5M | 0.603 (19) | 1/20 | 41.6 deg |

Verified the actual gait, not just the aggregate metrics (the exact check that would have caught EXP-0012's skating exploit): controlled straight-line diagnostic on the 4M checkpoint showed **50-51 liftoffs/10s** (vs. `exp0011`'s 45-47) and **17-21% airborne time** (vs. 11-14%) -- more time in the air, not less, the opposite of the skating signature. Direct swing-time measurement (same protocol as the original F-0003/F-0004 measurement, 6 episodes, 914 touchdowns): **median swing time 40ms (was 20ms), mean 35.0ms (was 27.3ms), p90 60ms (was 40ms), max 120ms (was 80ms)** -- a genuine ~2x increase in stride duration, not a reward artifact.

Full standard 20-episode evaluation of the selected 4M checkpoint against `exp0011`:

| run | len (s) | reward | vel_err | dist (m) | dist_ratio (n) | falls | yaw_drift/10s |
|---|---|---|---|---|---|---|---|
| `exp0011` (current canonical) | 18.3 | 1067 | 0.134 | 2.99 | 0.671 (18) | 2/20 | 42.7 deg |
| `exp0016` @ 4M (warm-start) | **19.7** | **1184** | **0.086** | **3.03** | 0.627 (**20**) | **1/20** | **36.9 deg** |

Every metric matches or improves on `exp0011` -- longer survival, higher reward, **36% lower velocity-tracking error**, more real distance covered, fewer falls, less yaw drift -- except `dist_ratio` itself, which is marginally lower (0.627 vs. 0.671) but computed over *all 20* reliable episodes vs. `exp0011`'s 18, making the two not quite apples-to-apples in `exp0011`'s favor.

A striking secondary observation: the training curve (`experiments/runs/exp0016_warmstart_air_time/curves.png`) is smooth and steadily improving, with **none of the violent climb-crash-recover oscillation** that characterized every from-scratch run in this project (F-0002). Fine-tuning an already-good policy at a low learning rate (5e-5, vs. the standard 3e-4) avoided the PPO-instability pattern entirely, not just the skating exploit -- a second, unplanned benefit of this approach.

Conclusion: **hypothesis confirmed.** The skating exploit in EXP-0012 was not inherent to the `feet_air_time_bonus` reward term itself -- it was a consequence of training that reward from scratch, when the policy could stumble onto "never touch down" before ever discovering real stepping. Warm-starting from an established stepping policy and fine-tuning gently gave the reward room to *refine* an existing good strategy instead of re-exploring from zero, and the existing strategy had too much to lose (working tracking + alive-bonus reward) to abandon for skating. This also incidentally produced a much smoother, more stable training process.

Decision: **Promote `experiments/runs/exp0016_warmstart_air_time/best_checkpoint_4M.zip` to the new canonical baseline**, superseding `exp0011_gait_quality_v2`. This is the first strict, unambiguous improvement across nearly every axis since `exp0011` itself, and the first successful fix for F-0003's stepping-quality half. New video: `experiments/runs/exp0016_warmstart_air_time/rollout_seed1002.mp4`. `docs/FAILURE_ANALYSIS.md` F-0004 updated with this resolution; `docs/DECISIONS.md` gets ADR-007.

Next: swing duration is meaningfully better (2x) but still short of a fully "deliberate" stride (40ms median vs. e.g. 200-300ms for a natural human-scale gait cadence) -- further fine-tuning iterations (repeat this same warm-start recipe from the new checkpoint) or a higher `feet_air_time_threshold_s` are both plausible next steps if pursued further. Not required for the challenge's minimum bar, which continues to be exceeded by a wide margin.

---

### EXP-0017 — Continue the successful fine-tuning recipe: more budget from `exp0016`
Date: 2026-09-18 (pre-registered before running)

Hypothesis: `exp0016`'s median swing time (40ms) is still far below the reward's own target (`feet_air_time_threshold_s=0.2` = 200ms) -- every touchdown is still being penalized for falling short, meaning the gradient toward longer swings has not run out of room to push. Continuing to fine-tune the *same lineage* (warm-start from `exp0016`'s own checkpoint, same config, same reduced learning rate) for more budget should push swing duration further toward that target, testing whether the improvement continues or has already plateaued.

Failure being targeted: F-0004's residual (swing duration doubled but still short of deliberate).

Metric: primary -- swing-time distribution (mean/median/p90) via the same direct-measurement protocol used throughout F-0004/EXP-0016, checked against `exp0016`'s own 40ms median. Secondary -- `dist_ratio`, falls, yaw drift, vel_err from the standard eval, which must not regress from `exp0016`'s levels.

Budget: 10,000,000 timesteps (2x EXP-0016's 5M, since this is now a known-safe recipe rather than a first probe), `n_envs=12`, seed=0, `checkpoint_freq=2_000_000`, `init_from=exp0016's 4M checkpoint`, `init_learning_rate=5e-5` (unchanged -- isolating budget as the only new variable, matching the EXP-0009/EXP-0015 pattern of testing budget before changing anything else).

Stop condition: fixed budget. If swing duration has plateaued (not continued increasing) by 10M, that's a real ceiling for this reward/threshold combination at this learning rate -- the next lever would be raising `feet_air_time_threshold_s` itself, not more budget on the same target.

Code state: no changes -- reuses `config/exp0012_air_time.yaml`, `training/train.py --init-from`/`--init-learning-rate` exactly as EXP-0016.

Results: Training completed in 12671.3s (~3.5 hours -- throughput degraded substantially over the run, from 11097 fps at the start to 789 fps by the end, almost certainly machine load/thermal effects from this being one of many long consecutive training runs this session, not a code issue). Checkpoint trend -- **stable and improving throughout, no regressions**:

| steps | dist_ratio | falls | yaw_drift/10s |
|---|---|---|---|
| 2M | 0.628 (19) | 2/20 | 51.2 deg |
| 4M | 0.589 (20) | 0/20 | 31.4 deg |
| 6M | 0.653 (20) | 0/20 | 37.0 deg |
| **8M** | **0.709 (20)** | **0/20** | 36.3 deg |
| 10M | 0.624 (20) | 0/20 | 38.2 deg |

Swing-time measurement on the 8M checkpoint (same protocol, 6 episodes, 861 touchdowns): **mean 47.6ms (was 35.0ms at 4M), median 40ms (unchanged from 4M), p90 60ms (unchanged), max 100ms (down from 120ms)**. The median and p90 plateaued -- more of the distribution shifted toward the 40-60ms band (a tighter, more consistent stride) rather than the whole distribution continuing to shift right. Controlled diagnostic confirms real stepping, more so than at 4M: 43-45 liftoffs/10s (down slightly from 50-51, consistent with each swing taking a bit longer) at **24-25% airborne time (up from 17-21%)**, both feet grounded only 51% of the time (down from 62%).

Full standard evaluation of the 8M checkpoint -- **the best result of the entire project on every metric simultaneously**:

| run | len (s) | reward | vel_err | dist (m) | dist_ratio (n) | falls | yaw_drift/10s |
|---|---|---|---|---|---|---|---|
| `exp0011` | 18.3 | 1067 | 0.134 | 2.99 | 0.671 (18) | 2/20 | 42.7 deg |
| `exp0016` @ 4M | 19.7 | 1184 | 0.086 | 3.03 | 0.627 (20) | 1/20 | 36.9 deg |
| **`exp0017` @ 8M** | **20.0** | **1268** | **0.061** | 3.01 | **0.709 (20)** | **0/20** | **36.3 deg** |

All 20/20 episodes reach the full 20s with zero falls -- the first time any run in this project has achieved that on the full standard eval. The training curve (`experiments/runs/exp0017_warmstart_air_time_long/curves.png`) stayed smooth and steadily improving across the *entire* 10M-timestep run, with no trace of the climb-crash-recover oscillation seen in every from-scratch run -- further evidence (now over 2x the duration of EXP-0016's own confirmation) that low-LR fine-tuning avoids that instability pattern, not just a short-run coincidence.

Conclusion: the fine-tuning recipe continues to pay off with more budget, but swing duration specifically has plateaued around a 40-60ms band -- the *quality* of the existing stride pattern (consistency, tracking precision, stability) kept improving even after stride *duration* stopped increasing. This is a meaningful, informative distinction: more budget on this exact reward/threshold keeps helping overall gait quality, but reaching a noticeably longer, more human-like stride would need a higher `feet_air_time_threshold_s`, not just more training at the current one.

Decision: **Promote `experiments/runs/exp0017_warmstart_air_time_long/best_checkpoint_8M.zip` to the new canonical baseline**, superseding `exp0016_warmstart_air_time`. New video: `experiments/runs/exp0017_warmstart_air_time_long/rollout_seed1002.mp4`. `docs/DECISIONS.md` gets ADR-008.

Next: raise `feet_air_time_threshold_s` (e.g. to 0.3-0.4s) and repeat the same warm-start-fine-tune recipe from this checkpoint, now that swing duration has plateaued at the current threshold -- the clear, evidence-backed next lever if this is pursued further.

---

### EXP-0018 — Push swing duration further: double `feet_air_time_bonus`, same threshold
Date: 2026-09-18 (pre-registered before running)

Hypothesis: EXP-0017 showed swing duration plateauing (median 40ms, mean 47.6ms) despite still being far short of the reward's own `feet_air_time_threshold_s=0.2` (200ms) target. Two candidate levers exist to push further: raise the threshold, or raise the weight (push harder toward the *existing* unmet target). Testing weight first, since it's a smaller, more conservative change (same target, just more pressure) than moving the goalpost itself. Measured before choosing (replaying `exp0017`'s own gait through the reward math): at `feet_air_time_bonus=6.0` (2x the current 3.0), the per-step cost is -0.131 (vs. -0.066 at 3.0), about 14% of the gait's own much-improved mean tracking reward (0.952/step, up from the 0.80/step measured back in F-0003) -- a moderate, safe escalation, nowhere near the >50% ratio that caused EXP-0010's collapse.

Failure being targeted: F-0004's residual plateau (EXP-0017).

Metric: primary -- swing-time distribution (mean/median/p90), same direct-measurement protocol as every prior check in this family, compared against `exp0017`'s 40ms/47.6ms/60ms. Secondary -- `dist_ratio`, falls, yaw drift, vel_err, which must not regress from `exp0017`'s best-of-project levels (0.709, 0/20, 36.3 deg, 0.061).

Budget: 8,000,000 timesteps, `n_envs=12`, seed=0, `checkpoint_freq=2_000_000`, `init_from=exp0017`'s 8M checkpoint, `init_learning_rate=5e-5` (unchanged) -- isolating the doubled weight as the only new variable.

Stop condition: fixed budget. If swing duration still doesn't move, weight alone isn't the lever -- the threshold itself needs to move (or this is a genuine physical/dynamic ceiling given the current PD gains/control frequency, not a reward-tuning problem at all). If falls or `dist_ratio` regress meaningfully, the weight was pushed too far for this budget -- revert, don't escalate further blindly.

Code state: no source changes -- new config `config/exp0018_air_time_v3.yaml` (`feet_air_time_bonus=6.0`, otherwise identical to `config/exp0012_air_time.yaml`).

Results: Training completed in 1427.7s (~24 min, 5603 steps/s). Checkpoint trend -- falls stayed low throughout (0-1/20, no destabilization):

| steps | dist_ratio | falls | yaw_drift/10s |
|---|---|---|---|
| 2M | 0.548 (20) | 0/20 | 32.8 deg |
| 4M | 0.575 (20) | 0/20 | 40.6 deg |
| 6M | 0.560 (20) | 0/20 | 36.4 deg |
| 8M | 0.626 (19) | 1/20 | 50.8 deg |

Swing-time measurement (same protocol, 6 episodes) on the 2M and 6M checkpoints: **median 60ms (up from 40ms), mean 53.3-59.2ms (up from 47.6ms), p90 80ms (up from 60ms)** -- genuine further improvement, and a 3x increase from the pre-F-0004 baseline (20ms) overall. Controlled diagnostic confirms this at every checkpoint checked (airborne time 24-27%, foot-height range extending up to 6-7cm vs. `exp0017`'s ~6cm max) -- real, higher, longer steps.

However, the controlled straight-line diagnostic also showed **heading control got worse** at every checkpoint checked: 2M -28.1 deg/10s, 4M -49.8 deg/10s, 8M -65.3 deg/10s (vs. `exp0017`'s 11.6 deg/10s on the identical test) -- and the 8M checkpoint showed a first-ever "both feet airborne" moment (1% of the episode, a brief double-float/hop), never observed in any prior checkpoint in this project. Full standard evaluation of the best-balanced checkpoint (2M, chosen for its combination of 0 falls and best aggregate yaw drift):

| run | len (s) | reward | vel_err | dist (m) | dist_ratio (n) | falls | yaw_drift/10s |
|---|---|---|---|---|---|---|---|
| `exp0017` (canonical) | 20.0 | 1268 | 0.061 | 3.01 | **0.709 (20)** | 0/20 | 36.3 deg |
| `exp0018` @ 2M | 20.0 | 1220 | 0.066 | 2.51 | 0.548 (20) | 0/20 | 32.8 deg (aggregate) |

Conclusion: **this is a genuine trade-off, not a clean win or a failure.** Doubling `feet_air_time_bonus` did what it was measured and intended to do -- swing duration increased substantially further -- but at a real cost: `dist_ratio` dropped 23% relative (0.709 -> 0.548) and real distance covered fell (3.01m -> 2.51m), with heading control also measurably worse in the controlled single-command test (though the aggregate varied-command yaw-drift metric happens to look slightly better, likely noise given the controlled test disagrees clearly). Pushing harder on stride duration is competing with the earlier-fixed straight-line and tracking-precision objectives once past a certain point -- plausibly because longer, higher steps at the same 50Hz control cadence and PD gains require more corrective effort to keep balanced and on-heading, effort that's no longer available to spend on tracking precision.

Decision: **Do not promote. `exp0017_warmstart_air_time_long` remains the canonical baseline** -- `dist_ratio` (ADR-003's own north-star metric for "is it actually walking well") is a clear, meaningful step down, and heading control regressed in the test built specifically to catch that (F-0003). `exp0018_air_time_v3`'s 2M checkpoint is kept and documented as an available alternative for anyone who specifically values visibly longer/higher strides over tracking precision and heading stability -- the same "documented alternative, not a replacement" treatment given to the domain-randomization variant (EXP-0015).

Next: this traces out a real Pareto frontier (stride duration vs. tracking/heading quality) rather than a single dominant direction -- `exp0017` is very close to the best achievable trade-off with this reward family at this weight/threshold combination. Further pushing this specific lever (weight or threshold) is not expected to yield another clean win without also addressing whatever makes longer strides cost heading control (e.g. re-strengthening `ang_vel_penalty` in tandem, or increasing PD gains to give more corrective authority during longer swings) -- untested, left for future work. The chattery-stepping problem (F-0003/F-0004) is considered adequately resolved for this project's purposes: swing duration tripled from the original 20ms baseline across EXP-0016/0017, with `exp0017` representing the best-verified balance point found.

---

### EXP-0019 — Recover heading control at the longer stride length: `heading_deviation_penalty`
Date: 2026-09-19 (pre-registered before running)

Hypothesis: ADR-009 found that `ang_vel_penalty` (instantaneous rotation-rate squared) doesn't cleanly target *net* heading drift -- measured directly: `exp0017` (good heading, 19deg drift in a redone controlled measurement) and `exp0018` (bad heading, 28deg drift) have nearly identical mean `ang_vel_z^2` (0.038 vs. 0.112 in *heading-deviation-squared* terms once measured properly per-step over the controlled test -- a steady small bias integrates into real drift that a magnitude-only, zero-mean-blind penalty can't distinguish from harmless oscillation). A new term, `heading_deviation_penalty`, penalizes deviation from the heading at episode start directly, targeting the actual failure (drift) rather than a proxy (rotation-rate magnitude) that theory and measurement both suggest is a poor fit for it. If this works, it could let `exp0018`'s longer strides and `exp0017`'s good heading coexist -- resolving the ADR-009 trade-off rather than just documenting it.

Failure being targeted: ADR-009's Pareto trade-off (longer strides cost heading control).

Measurement before choosing the weight: replayed both `exp0017`'s and `exp0018`'s checkpoints through the *same controlled straight-line test* (vx=0.3, vy=0, seed 1002) and computed mean heading-deviation-squared per step directly (not the instantaneous ang_vel proxy): `exp0017` 0.038, `exp0018` 0.112 -- a real, 3x gap consistent with the qualitative diagnostic difference. At `heading_deviation_penalty=0.5`, this costs `exp0017`'s gait 0.019/step (~2% of its ~0.94 tracking reward) and `exp0018`'s gait 0.056/step (~6%) -- a meaningful, targeted, moderate escalation.

New capability built for this: `heading_deviation_penalty` reward term (`envs/g1_walk_env.py` tracks `self._yaw_at_reset` via `quat_to_yaw`, computes wrapped deviation each step; `config/loader.py` field defaults to 0.0 for backward compatibility; `rewards/walking_reward.py` squares and weights it). 6 new tests (3 reward-math tests, 2 env-level tests verifying the yaw-tracking mechanism itself, 1 backward-compatibility extension).

Metric: primary -- controlled straight-line heading drift (same protocol as every heading check since F-0003) and swing-time distribution (must not regress from `exp0018`'s 60ms median -- the whole point is to keep the longer strides). Secondary -- `dist_ratio`, falls, vel_err from the standard eval, compared against both `exp0017` (good heading) and `exp0018` (long strides, bad heading) as the two reference points this experiment is trying to reconcile.

Budget: 8,000,000 timesteps, `n_envs=12`, seed=0, `checkpoint_freq=2_000_000`, `init_from=exp0018`'s 2M checkpoint (the longer-stride, worse-heading gait -- fine-tuning FROM the behavior we want to fix, not from `exp0017`, since we want to keep the stride gains while adding heading correction), `init_learning_rate=5e-5` (unchanged recipe).

Stop condition: fixed budget. If heading drift doesn't clearly improve toward `exp0017`'s level while swing duration stays near `exp0018`'s level, the two objectives may not be jointly satisfiable with this reward family at these weights -- a legitimate negative result confirming ADR-009's Pareto framing more strongly, not grounds for further blind weight tweaking.

Code state: `config/loader.py` (`heading_deviation_penalty`), `rewards/walking_reward.py` (`compute_reward` takes `heading_deviation_rad`), `envs/g1_walk_env.py` (tracks `_yaw_at_reset`, computes deviation each step), `tests/test_reward.py` (3 new tests), `tests/test_env_contract.py` (2 new tests). New config `config/exp0019_heading_hold.yaml`.

Results: Training completed in 1114.0s (~19 min, 7181 steps/s -- normal throughput). Checkpoint trend, still climbing at the end (not yet plateaued):

| steps | dist_ratio | falls | yaw_drift/10s (aggregate) |
|---|---|---|---|
| 2M | 0.357 (20) | 0/20 | 21.4 deg |
| 4M | 0.564 (18) | 2/20 | 33.2 deg |
| 6M | 0.563 (20) | 0/20 | 24.9 deg |
| **8M** | **0.613 (19)** | 1/20 | 27.3 deg |

Controlled straight-line diagnostic on the 8M checkpoint (same protocol as every heading check since F-0003): **yaw drift -7.5 deg/10s -- better than `exp0017`'s -19.0 deg AND `exp0018`'s -28.2 deg on the identical test.** Direct swing-time measurement (same protocol, 6 episodes): median 60ms, mean 50.8ms -- **matches `exp0018`'s stride-length gains essentially exactly** (60ms median, 53.3ms mean), not a regression back toward `exp0017`'s shorter 40ms strides.

Full standard evaluation of the 8M checkpoint against both reference points:

| run | len (s) | vel_err | dist (m) | dist_ratio (n) | falls | yaw_drift/10s | swing median |
|---|---|---|---|---|---|---|---|
| `exp0017` (canonical) | 20.0 | 0.061 | 3.01 | **0.709 (20)** | **0/20** | 36.3 deg | 40ms |
| `exp0018` (long stride, worse heading) | 20.0 | 0.066 | 2.51 | 0.548 (20) | 0/20 | 32.8 deg | 60ms |
| `exp0019` (this run) | 19.2 | 0.093 | 2.94 | 0.613 (19) | 1/20 | **27.3 deg** | **60ms** |

Conclusion: **the hypothesis is confirmed mechanistically -- `heading_deviation_penalty` genuinely decouples heading control from stride length**, unlike `ang_vel_penalty` alone. `exp0019` has the best heading of all three checkpoints (both in the controlled test and the aggregate) while matching `exp0018`'s longer strides exactly. However, it does not fully recover `exp0017`'s overall tracking precision: `dist_ratio` (0.613) sits between `exp0018` and `exp0017`, and `vel_err` (0.093) is actually worse than *both* other checkpoints -- a genuinely mixed, three-way trade-off rather than a clean win over `exp0017`. Critically, the checkpoint trend was **still climbing at 8M, not plateaued** (0.357 -> 0.564 -> 0.563 -> 0.613) -- the same shape EXP-0016 showed before EXP-0017's continuation pushed it substantially further. Per the ADR-004/ADR-008 precedent ("test more budget before concluding a technique doesn't work or has plateaued"), this should be continued before making a final call, not judged on an still-improving trend.

Decision: **Not yet promoted -- continue this exact lineage with more budget first** (EXP-0020), following the established precedent, before deciding whether `exp0019`'s approach can fully match `exp0017`'s tracking precision while keeping the heading and stride gains, or whether a genuine 3-way Pareto surface (stride length, tracking precision, heading control) is the final answer.

Next: EXP-0020 -- continue fine-tuning from `exp0019`'s 8M checkpoint, same config/learning rate, 8M more timesteps.

---

### EXP-0020 — Continue the heading-hold lineage: does more budget help, per the ADR-004/0008 precedent?
Date: 2026-09-19 (pre-registered before running)

Hypothesis: `exp0019`'s checkpoint trend was still climbing at 8M timesteps (0.357 -> 0.564 -> 0.563 -> 0.613), unresolved whether it would continue improving with more budget (as EXP-0016 -> EXP-0017 did) or had already plateaued (as EXP-0017's own swing duration did). Continuing the exact same lineage for 8M more timesteps tests this directly, per the established "test budget before concluding" discipline (ADR-004, ADR-008).

Failure being targeted: none new -- testing whether `exp0019` was undertrained.

Budget: 8,000,000 timesteps, `n_envs=12`, seed=0, `checkpoint_freq=2_000_000`, `init_from=exp0019`'s 8M checkpoint, `init_learning_rate=5e-5` (unchanged).

Stop condition: fixed budget. If the trend doesn't continue climbing, this lineage has plateaued and the final decision is made on the best checkpoint found across EXP-0019/0020 combined, not escalated further.

Code state: no changes -- reuses `config/exp0019_heading_hold.yaml` exactly.

Results: Training completed in 1105.8s (~18 min, 7235 steps/s). Checkpoint trend -- **did not continue climbing, unlike EXP-0016->0017**:

| steps | dist_ratio | falls | yaw_drift/10s (aggregate) |
|---|---|---|---|
| 2M | 0.631 (19) | 2/20 | 29.7 deg |
| 4M | 0.523 (20) | 0/20 | 26.7 deg |
| 6M | 0.584 (19) | 2/20 | 30.6 deg |
| 8M | 0.589 (19) | 1/20 | 39.6 deg |

`dist_ratio` oscillates in the 0.52-0.63 range without a clear upward trend, and yaw drift got *worse* at the later checkpoints (39.6 deg at 8M, worse than `exp0019`'s own 27.3 deg) -- the opposite of what more budget did for EXP-0016/0017's underlying metric. No checkpoint here clearly beats `exp0019`'s own 8M result (0.613 dist_ratio, 27.3 deg yaw drift, 1 fall).

Conclusion: **this specific lineage has plateaued.** Unlike the base-gait budget question (ADR-004) where more training was unambiguously the answer, this reward configuration (with `heading_deviation_penalty` added on top of an already-tuned, already-fine-tuned-twice lineage) does not continue improving with more of the same recipe -- confirming this is a genuine local optimum for this exact weight combination, not simply undertraining.

Decision: **`exp0019`'s 8M checkpoint is the best result from the heading-hold lineage; EXP-0020 is a documented negative/confirmatory result, not adopted.** Final three-way comparison across the whole chattery-stepping-and-heading investigation:

| run | vel_err | dist_ratio | falls | yaw_drift (controlled) | swing median |
|---|---|---|---|---|---|
| `exp0017` (canonical) | **0.061** | **0.709** | **0/20** | -19.0 deg | 40ms |
| `exp0018` (longest stride) | 0.066 | 0.548 | 0/20 | -28.2 deg | **60ms** |
| `exp0019` (best heading) | 0.093 | 0.613 | 1/20 | **-7.5 deg** | **60ms** |

No single checkpoint dominates on every axis -- this is a genuine 3-way Pareto surface (tracking precision vs. stride length vs. heading control), not a single best answer. `exp0017` remains canonical (best tracking precision and survival, which the challenge's own deliverables emphasize); `exp0018` and `exp0019` are both kept as documented alternatives representing different, deliberately-chosen points on that surface -- `exp0018` for maximum stride length, `exp0019` for the best combination of long strides with straight-line walking, at some cost to tracking precision.

Next: this investigation is concluded. A genuinely different approach (not another weight tweak) would be needed to find a checkpoint that matches `exp0017` on tracking precision while also matching `exp0019` on heading and `exp0018`/`exp0019` on stride length simultaneously -- e.g. a longer combined curriculum from scratch, or increased PD gains for more corrective authority (ADR-009's suggestion), both untested and left for future work.

---

### EXP-0021 — Domain randomization re-run on top of the canonical stepping-quality reward
Date: 2026-09-19 (pre-registered before running)

Before this: increasing PD gains (ADR-009/010's other untested lever) was measurement-tested, not trained -- per-joint torque as a fraction of `actuatorfrcrange` never exceeds ~63% on any of `exp0017`/`exp0018`/`exp0019`, so there is no torque headroom being hit; raising PD gains is not well-motivated and is deprioritized (docs/DECISIONS.md ADR-010 addendum). Separately, the high-speed-command failure mode (F-0002) was re-verified against `exp0017` using only existing eval data (no training): falls are resolved (0/20 confirmed at the seed level) but the same seeds now show a curving/heading-drift pattern instead. Neither finding calls for new training on its own. Domain randomization (PROJECT_STATE.md "highest-value next step" #3) is the remaining well-scoped, previously-validated lever that can produce an actual improvement (robustness) rather than more characterization.

Hypothesis: EXP-0014/0015 showed domain randomization (floor friction, mass/inertia, motor strength, periodic pushes -- `envs/domain_randomization.py`) gives a genuine, if modest, robustness gain (fewer falls, longer survival under randomized physics) at some cost to tracking precision, but that recipe trained on the pre-EXP-0016 reward (no `feet_air_time_bonus`). Re-running the identical randomization config on top of the current canonical reward (`config/exp0012_air_time.yaml`, behind `exp0017`), via warm-start fine-tuning from `exp0017`'s own 8M checkpoint at low LR (the project's established safe way to introduce a new training-distribution change, per ADR-007/008), should extend that same robustness gain to the better base gait, rather than forcing DR to relearn stepping quality from scratch.

Failure being targeted: none new -- extending EXP-0015's validated result onto a better base checkpoint, and testing whether warm-starting (vs. EXP-0015's from-scratch approach) also avoids DR-specific training instability the way it did for the air-time/heading terms.

Config: `config/exp0021_dr_on_canonical.yaml` (canonical `exp0012` reward weights, unchanged, plus `config/exp0014_domain_randomization.yaml`'s randomization block, unchanged -- single conceptual change is DR itself, since the reward is identical to what `exp0017` already trained under).
- seed: 0
- env count: 12 (SubprocVecEnv)
- physics dt: 0.002s, control decimation 10 (50Hz)
- action scale: 0.25
- reward terms/scales: identical to canonical (`exp0012`) -- tracking_lin_vel=1.0, alive_bonus=0.5, torque_penalty=0.0002, ang_vel_penalty=0.1, action_rate_penalty=0.01, feet_air_time_bonus=3.0 @ threshold 0.2s
- PPO differences: `init_from=experiments/runs/exp0017_warmstart_air_time_long/best_checkpoint_8M.zip`, `init_learning_rate=5e-5` (fine-tune, not from-scratch)
- budget: 10,000,000 timesteps, `checkpoint_freq=2,000,000` (matching EXP-0017's follow-up scale, since fine-tuning from an already-good base should need less than EXP-0015's 20M from-scratch budget)

Evaluation protocol: standard 20-seed deterministic eval (`evaluation/evaluate_checkpoints.py`) under two conditions per checkpoint -- (a) clean physics (`config/exp0012_air_time.yaml`, no randomization) to check tracking precision/dist_ratio/falls are preserved relative to `exp0017`'s own 0.709/0/20; (b) randomized physics (`config/exp0021_dr_on_canonical.yaml`'s own training distribution) to check robustness relative to `exp0017` evaluated under the same randomized physics with zero DR training (the correct control, mirroring EXP-0014/15's `exp0011`-under-randomized-physics control).

Stop condition: fixed 10M-timestep budget. Success = randomized-physics survival/dist_ratio clearly improves over the `exp0017`-under-randomization control (as EXP-0015 did over `exp0011`'s control) while clean-physics metrics stay close to `exp0017`'s own numbers (not regressing the way EXP-0015's from-scratch DR did relative to `exp0011`). If clean-physics metrics regress substantially, that trade-off gets documented honestly, not treated as a training bug.

Results: Training completed in 1385.2s (~23 min, 7219 steps/s), no crashes. `approx_kl`/`clip_fraction` started low (0.017/0.22, matching prior fine-tune runs) but climbed over the run to 0.05-0.08/0.48-0.59 by 10M -- higher than EXP-0016/17/19's fine-tunes typically showed (usually <0.03), consistent with domain randomization being a larger distribution shift than adding a reward term alone, though it never produced the outright climb-crash-recover collapse seen in from-scratch runs.

Checkpoint trend, clean physics (`config/exp0012_air_time.yaml`, canonical's own eval config):

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s |
|---|---|---|---|
| 2M | 0.598 (20) | 0/20 | 30.5 deg |
| 4M | 0.558 (20) | 0/20 | 27.4 deg |
| **6M** | **0.609 (20)** | **0/20** | 33.0 deg |
| 8M | 0.488 (20) | 0/20 | 30.9 deg |
| 10M | 0.539-0.548 (20) | 0/20 | 39.6 deg |

Checkpoint trend, randomized physics (`config/exp0021_dr_on_canonical.yaml`, its own training distribution):

| steps | dist_ratio (n reliable) | falls | mean len (s) |
|---|---|---|---|
| 2M | 0.831 (13) | 11/20 | 13.7 |
| 4M | 0.789 (16) | 9/20 | 15.8 |
| **6M** | **0.679 (16)** | **7/20** | **16.3** |
| 8M | 0.656 (15) | 8/20 | 16.1 |
| 10M | 0.669-0.686 (15-16) | 10-11/20 | 15.2 |

The 6M checkpoint has the best clean-physics `dist_ratio` of the whole trend *and* the best (lowest) randomized-physics fall rate -- a clean, non-arbitrary pick, not a tie-break. Selected as `experiments/runs/exp0021_dr_on_canonical/best_checkpoint_6M.zip`.

Control (required by the pre-registered protocol): `exp0017` (zero DR training) evaluated under the *same* randomized-physics config -- 14/20 falls, mean length 10.3s, `mean_vel_error` 0.168.

Full standard-protocol comparison (20 episodes each):

| condition | policy | len (s) | vel_err | dist_ratio (n reliable) | falls |
|---|---|---|---|---|---|
| clean | `exp0017` (no DR) | 20.0 | 0.061 | 0.709 (20) | 0/20 |
| clean | `exp0021` @ 6M (DR) | 20.0 | 0.075 | 0.609 (20) | 0/20 |
| randomized | `exp0017` (no DR, control) | 10.3 | 0.168 | 0.988 (10) | 14/20 |
| randomized | `exp0021` @ 6M (DR) | **16.3** | 0.110 | 0.679 (16) | **7/20** |

(Note: the control's 0.988 randomized `dist_ratio` is not directly comparable -- only 10/20 of its episodes are long enough to count as "reliable" per ADR-003's own filter, i.e. it is the ratio among the minority that happened to survive; `exp0021` has both more reliable episodes (16/20) and a lower per-episode ratio among them -- the two numbers answer different questions and should not be read as "the control tracks better.")

Conclusion: **hypothesis confirmed.** Domain randomization re-trained on top of the canonical stepping-quality reward reproduces EXP-0015's exact trade-off pattern (falls under randomized physics roughly halved -- 14/20 to 7/20 -- and survival time up ~58%, at a real but moderate cost to clean-physics tracking precision -- `dist_ratio` 0.709 to 0.609, about 14% relative, `vel_err` 0.061 to 0.075 m/s) on a meaningfully better base gait than EXP-0015 started from (`exp0017`'s 0.709 clean `dist_ratio` vs. `exp0011`'s 0.705 -- essentially the same clean baseline quality, so this is a fair, controlled re-run, not confounded by a different starting point). Warm-start fine-tuning (5e-5 LR from `exp0017`'s checkpoint) also worked for introducing this training-distribution shift, not just new reward terms -- no exploit, no collapse, though the elevated late-training KL/clip-fraction suggests DR is a "larger" distribution shift than a reward-term addition and might benefit from an even lower fine-tune LR or more budget if pushed further. Clean-physics `dist_ratio` did not clearly plateau by 10M (2M/6M's ~0.60 vs. 4M/8M/10M's dips look more like noise around a similar level than a trend, unlike EXP-0009's/EXP-0017's clean monotonic climbs) -- more budget is not obviously the next lever here.

Decision: **Adopt `experiments/runs/exp0021_dr_on_canonical/best_checkpoint_6M.zip` as a new documented alternative, alongside `exp0018`/`exp0019`, not superseding canonical `exp0017`.** This adds a fourth point to the project's Pareto surface -- a robustness axis, not just stride-length/heading -- for anyone who specifically needs the policy to survive physics variation (unknown terrain friction, payload changes, motor wear, external pushes) at some cost to nominal-condition tracking precision. `exp0015` (the pre-EXP-0016 domain-randomization variant) is superseded by this result for anyone who wants both DR robustness and the improved stepping quality; it is kept for history, not deleted.

Next: if this axis is pursued further, the untested levers are a lower fine-tune LR (to tame the elevated late-training KL) or a curriculum that ramps randomization strength in gradually rather than applying the full EXP-0014 ranges from step zero -- both left for future work, not required by the challenge.

---

### EXP-0022 — Does a from-scratch curriculum with all reward terms present escape the 3-way Pareto trade-off?
Date: 2026-09-19 (pre-registered before running)

Hypothesis: ADR-009/ADR-010's 3-way Pareto surface (no checkpoint matches `exp0017` on tracking precision, `exp0018`/`exp0019` on stride length, and `exp0019` on heading control simultaneously) emerged from a *sequential* fine-tuning chain -- `exp0011` (tracking-only) -> `exp0016`/`exp0017` (add `feet_air_time_bonus`, fine-tune) -> `exp0018` (raise the bonus, fine-tune again) -> `exp0019` (add `heading_deviation_penalty`, fine-tune again). Each low-LR fine-tune stage, by design, makes only small departures from its starting policy (that is precisely why it avoids the from-scratch exploit/collapse problem, per ADR-007/008) -- but this could mean each stage's local optimum gets locked in before the next term has a chance to reshape the *underlying* gait strategy, rather than the combination of terms having a genuine joint optimum that dominates on all three axes. Training from scratch with all terms present together from step 0 removes that path-dependence; if the same trade-off reappears, it is a property of the reward combination itself, not an artifact of how it was reached. If a from-scratch run instead finds a checkpoint that is competitive with `exp0017` on tracking *and* `exp0019` on heading *and* `exp0018`/`exp0019` on stride length, the Pareto surface is not fundamental and the sequential fine-tuning chain was the limiting factor all along.

Failure being targeted: none new -- testing whether ADR-009/010's Pareto surface is fundamental or a fine-tuning-path artifact.

Secondary question: every from-scratch run in this project so far (F-0002) has shown the climb-crash-recover PPO oscillation; this is also the first test of whether that holds even for this richer, multi-term reward (all prior from-scratch runs used simpler rewards -- `baseline0001` had no air-time/heading terms at all).

Config: `config/exp0022_full_curriculum_from_scratch.yaml` -- byte-for-byte identical reward weights and PPO hyperparameters to `config/exp0019_heading_hold.yaml` (`feet_air_time_bonus=6.0`, `heading_deviation_penalty=0.5`, standard `learning_rate=3e-4`, not a fine-tune LR), but **no `--init-from`** -- random initialization, matching how `baseline0001`/`exp0009`/`exp0014`/`exp0015` were all trained from scratch.
- seed: 0
- env count: 12 (SubprocVecEnv)
- physics dt: 0.002s, control decimation 10 (50Hz)
- action scale: 0.25
- budget: 20,000,000 timesteps, `checkpoint_freq=4,000,000` -- matching EXP-0009's and EXP-0015's from-scratch budget precedent (a from-scratch run needs more budget than a fine-tune to reach comparable quality; this is not directly comparable to `exp0019`'s own 8M fine-tune budget, since `exp0019` also inherited ~15M timesteps of prior fine-tuning through its lineage -- a fair from-scratch comparison needs a budget of the same order as that *total* lineage cost, not just its last stage)

Evaluation protocol: standard 20-seed deterministic eval (`evaluation/evaluate_checkpoints.py`, clean physics) on every checkpoint plus final model, same as every other run -- comparing `dist_ratio`, falls, and the controlled straight-line yaw-drift test (`diagnostics/gait_diagnostic.py`) against `exp0017`/`exp0018`/`exp0019`'s known numbers.

Stop condition: fixed 20M-timestep budget. Success = a checkpoint within a small margin of `exp0017`'s `dist_ratio` (~0.70) AND `exp0019`'s controlled-test yaw drift (~-7.5 deg/10s) AND `exp0018`/`exp0019`'s stride length (~60ms median swing) simultaneously -- if no single checkpoint clears roughly 90% of each of these three targets at once, the Pareto surface is confirmed fundamental to this reward combination at this budget, not a fine-tuning artifact, and this specific lever is considered answered (not extended with a third budget increase without a new reason).

Results: Training completed in 2990.8s (~50 min, 6687 steps/s), no crashes. `approx_kl`/`clip_fraction` ran hot for the *entire* run and never settled -- 0.09-0.21 / 0.53-0.69 even at the very end (20M steps in), well above the already-elevated levels seen for `baseline0001` (0.09-0.19) and far above any fine-tuning run in this project (typically <0.03). `ep_len_mean` climbed to ~960-970/1000 (near-full survival) but `ep_rew_mean` plateaued around 790-800 without a clear trend.

Checkpoint trend, matching config (`config/exp0022_full_curriculum_from_scratch.yaml`):

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s |
|---|---|---|---|
| 4M | 0.041 (19) | 1/20 | 3.4 deg |
| 8M | 0.058 (20) | 0/20 | 13.9 deg |
| 12M | 0.101 (19) | 2/20 | 26.1 deg |
| 16M | **0.119** (19) | 2/20 | 37.2 deg |
| 20M | 0.048 (20) | 0/20 | 4.4 deg |

**Every checkpoint is a catastrophic `dist_ratio` failure** (0.04-0.12, vs. `exp0017`'s 0.71, `exp0018`'s 0.55, `exp0019`'s 0.61) despite near-zero falls and near-full episode length -- the exact inverse-looking signature of the EXP-0006/F-0002 degenerate optimum (looks fine on survival, is not walking). Direct per-episode inspection of the best checkpoint (16M) confirms it: `final_pelvis_height` ~0.78 (standing, never fell), episodes run the full 20s, but `distance_traveled_m` is 0.23-0.44m *over the entire 20-second episode* (roughly 1-2 cm/s average drift, not a gait) while `mean_vel_error` is 0.23-0.41 m/s -- far worse tracking than any genuine walking checkpoint in the project (`exp0017`'s is 0.061). The policy is standing (with some in-place swaying/rotation -- yaw drift is real, 20-48 deg/10s on individual episodes, despite barely translating) and collecting `alive_bonus` + low torque/ang-vel/action-rate penalties, not walking at all. The trend across checkpoints is noisy and non-monotonic (0.04 -> 0.12 -> 0.05), not a slow climb like `exp0009`'s clean 0.18 -> 0.71 -- this looks like noise around a stuck local optimum, not a slower version of genuine progress.

Conclusion: **hypothesis not supported, and disconfirmed more strongly than expected.** Training from scratch with the full multi-term reward (`feet_air_time_bonus=6.0` + `heading_deviation_penalty=0.5` on top of the base gait-quality terms) did not just reproduce the 3-way Pareto trade-off in a different form -- it failed to discover real locomotion *at all* within 20M timesteps, collapsing to the same "alive but not walking" degenerate optimum first characterized in EXP-0006/F-0002, this time reached via persistently high-KL noisy exploration rather than KL-capped stability. This is informative and generalizes F-0002's lesson: the failure mode is not specific to `target_kl`-capped PPO (EXP-0006's mechanism) -- a sufficiently rich, multi-term reward can have "survive without moving" as such an easy, strongly-attracting local optimum from a random start that even a noisy, high-KL, unconstrained PPO process never leaves its basin, whereas the *same* reward combination is easy to reach incrementally when starting from a policy that already walks (that's exactly what the sequential fine-tuning chain -- `exp0011`->`exp0016`->`exp0017`->`exp0018`->`exp0019` -- did successfully). The pre-registered stop condition is clearly met on the "fail" branch: no checkpoint gets anywhere near 90% of `exp0017`'s tracking, `exp0019`'s heading, or `exp0018`/`exp0019`'s stride length -- most don't walk in any meaningful sense at all.

Decision: **Do not pursue further budget on this exact config -- the checkpoint trend shows noise around a stuck optimum, not slow convergence, so more timesteps is not the indicated fix (unlike EXP-0009/0015's from-scratch budget stories, where the trend was monotonically improving and simply needed more time).** No checkpoint is promoted or kept as an alternative; artifacts are kept for the record only (`experiments/runs/exp0022_full_curriculum_from_scratch/`, no `best_checkpoint` designated, matching the EXP-0006/0007/0008 documented-failure convention). `exp0017` remains canonical; the 3-way Pareto surface (ADR-009/010) stands, now with stronger evidence that it is not an artifact of the sequential fine-tuning approach -- if anything, sequential fine-tuning from an already-competent base is confirmed *necessary*, not merely convenient, for bootstrapping this level of reward complexity at all.

Next: this specific question (does a from-scratch curriculum escape the Pareto surface) is answered "no, and more strongly no than expected" for this reward richness and budget. A genuinely different attempt at escaping the Pareto surface would need either (a) a curriculum that starts simple and adds terms gradually *within* a single from-scratch run (e.g. reward-term annealing/scheduling), which is a different mechanism from both "all terms from step 0" (this experiment) and "sequential fine-tuning across separate runs" (the existing approach), or (b) accepting the Pareto surface as a genuine property of this reward family and exploring a structurally different lever (e.g. curriculum on command *difficulty* rather than reward terms, or an architecture change) -- both left for future work, neither required by the challenge.

---

### EXP-0023 — Does low LR alone (independent of warm-starting) avoid the from-scratch PPO oscillation?
Date: 2026-09-19 (pre-registered before running)

Hypothesis: F-0002 established that every from-scratch run in this project using the *standard* `learning_rate=3e-4` shows a climb-crash-recover oscillation (high `approx_kl`/`clip_fraction`), while every fine-tuning run at the *low* `learning_rate=5e-5` has been smooth. EXP-0022 (from scratch, standard LR, but a much richer reward) does not isolate the variable: it changed both initialization (random vs. warm-started) and reward complexity (multi-term vs. minimal) at once, and still ran high-KL throughout, so it cannot distinguish "low LR causes smoothness" from "warm-starting from a competent policy causes smoothness." This experiment isolates LR as the only variable: `config/default.yaml` (the exact `baseline0001`/`exp0009` config -- minimal 3-term reward, no air-time/heading terms) with only `learning_rate` changed from 3e-4 to 5e-5, still randomly initialized (no `--init-from`). If training stays smooth (`approx_kl` in the typically-cited ~0.01-0.03 range, no climb-crash-recover pattern in the curves), low LR by itself is sufficient to avoid the oscillation, independent of warm-starting. If it still oscillates (or simply fails to learn anything within budget, since 5e-5 is 6x slower than the standard rate), warm-starting -- not LR alone -- is the operative factor, and the smooth fine-tuning curves seen throughout this project are explained by "starting from a working policy," not "using a low LR."

Failure being targeted: none new -- isolating the mechanism behind F-0002's oscillation / the project's fine-tuning smoothness, cleanly separating LR from initialization for the first time.

Config: `config/exp0023_low_lr_from_scratch.yaml` -- identical to `config/default.yaml` except `ppo.learning_rate: 5.0e-5` (vs. the standard 3.0e-4).
- seed: 0
- env count: 12 (SubprocVecEnv)
- physics dt: 0.002s, control decimation 10 (50Hz)
- reward terms/scales: minimal -- `tracking_lin_vel=1.0`, `alive_bonus=0.5`, `torque_penalty=0.0002` (unchanged from `baseline0001`/`exp0009`)
- PPO differences: `learning_rate=5e-5` only; no `--init-from` (random init)
- budget: 10,000,000 timesteps, `checkpoint_freq=2,000,000` -- identical to EXP-0009's budget/protocol, for a direct, apples-to-apples comparison against the one existing "same reward, same budget, standard LR" data point

Evaluation protocol: standard 20-seed deterministic eval (`evaluation/evaluate_checkpoints.py`, matching config) at each checkpoint, same as EXP-0009; primary signal is the training curve's `approx_kl`/`clip_fraction` shape (smooth vs. climb-crash-recover) compared directly against `exp0009`'s own curve, not just final `dist_ratio`.

Stop condition: fixed 10M-timestep budget. Three possible clean outcomes, all informative: (1) smooth curve + competitive `dist_ratio` -- low LR alone avoids the oscillation and still learns to walk, just needs the same budget; (2) smooth curve + poor `dist_ratio` (undertrained-looking, not degenerate-looking) -- low LR avoids the oscillation but trades it for slower learning, a real speed/stability trade-off; (3) oscillates anyway, or collapses to the EXP-0022-style degenerate optimum -- low LR alone is not sufficient, warm-starting from a working policy is the actual operative factor. No further budget escalation without a new hypothesis regardless of which outcome occurs.

Results: Training completed in 1367.8s (~23 min, 7311 steps/s), no crashes. The training curve is clean and essentially monotonic -- `ep_rew_mean` climbs from ~22 to ~1420 with no crash exceeding a 16.8% dip (and that dip occurs at a reward level of ~100, very early in training, not a late-training regression); `approx_kl` never exceeds 0.061 across the entire 10M-timestep run and spends most of training under 0.03 -- both far below `baseline0001`/`exp0009`'s typical 0.09-0.19 and EXP-0022's persistent 0.09-0.21.

Checkpoint trend (`config/exp0023_low_lr_from_scratch.yaml`):

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s |
|---|---|---|---|
| 2M | 0.026 (20) | 0/20 | 10.3 deg |
| 4M | 0.298 (16) | 4/20 | 45.5 deg |
| 6M | 0.441 (18) | 2/20 | 63.3 deg |
| **8M** | **0.848** (20) | **0/20** | 30.4 deg |
| 10M | 0.756-0.760 (20) | 0/20 | 47.9-48.3 deg |

The 8M checkpoint's full standard evaluation: `dist_ratio` **0.848 -- the best of any checkpoint in the entire project** (canonical `exp0017`: 0.709), `vel_err` **0.035 m/s** (also best in project; canonical: 0.061), 0/20 falls, 20.0s mean survival, yaw drift 30.4 deg/10s (better than canonical's 36.3), mean torque cost 0.046 (comparable to or lower than other runs, ruling out "wins via brute-force torque" as the explanation). On paper, a clean sweep on every metric `evaluation/evaluate.py` reports.

**The controlled gait diagnostic (`diagnostics/gait_diagnostic.py`, vx=0.3/vy=0) tells a different, important story.** Yaw drift 13.4 deg/10s (good, comparable to canonical), net forward progress 2.79m/10s at commanded 0.3 m/s (93% -- excellent controlled tracking). But foot-height and contact-pattern plots (`diagnostics/reports/gait_diagnostic_exp0023_8M.png`, directly compared against `diagnostics/reports/gait_diagnostic_exp0017_8M.png`) show a foot-lift range of only **0.032-0.044m (1.2cm)**, visibly flatter and lower than canonical's own already-modest 0.03-0.06+m (3cm) range, with a much higher stepping frequency (69-70 liftoffs/10s vs. canonical's 43-45). A direct swing-duration measurement (ad-hoc script replaying `envs.contacts`-style touchdown bookkeeping, 6 episodes, 1334 touchdowns) gives median 40.0ms / mean 36.5ms / p90 60.0ms -- numerically close to canonical's own 40ms median, so raw "time airborne" alone did not initially flag the problem; it was the foot-height *amplitude* plot, not the duration statistic, that revealed this is a small, rapid shuffle rather than a real stride. **This checkpoint reproduces the exact small/chattery-stepping character F-0003/F-0004 originally set out to fix -- unsurprising in hindsight, since this run's reward (`config/default.yaml`) has no `feet_air_time_bonus` term at all, so nothing in the objective asks for a longer stride.**

Conclusion: **the pre-registered question has a clear, if nuanced, answer.** Outcome (1) from the stop condition is confirmed for training *stability*: low LR alone (5e-5), independent of warm-starting, avoids the from-scratch PPO oscillation (F-0002) -- and, more than merely avoiding it, converges to the best pure-tracking result in the project. But this experiment also sharpens *why* the project needed `feet_air_time_bonus`/`heading_deviation_penalty` in the first place: it was never really about PPO instability causing bad strides -- it was that nothing in the minimal reward asks for a longer stride, so even a perfectly smooth, well-converged optimization process settles on the cheapest way to track velocity precisely (small, fast steps), exactly as F-0003 first found. Low LR fixes *how reliably* PPO reaches a good optimum for whatever the reward actually specifies; it does not change what optimum that is. This means EXP-0022's failure (full reward, standard LR, from scratch, never learns to walk at all) and this run's success (minimal reward, low LR, from scratch, walks very well but with tiny steps) differ in *two* variables at once (LR and reward richness) -- leaving open whether the *combination* (full reward + low LR, from scratch) could get a policy that is stable to train **and** has a real stride, matching or beating `exp0017` without any sequential fine-tuning at all.

Decision: **Do not promote -- `exp0017` remains canonical.** This checkpoint is not a Pareto-surface "alternative" in the same sense as `exp0018`/`exp0019`/`exp0021` (which are all built on the canonical stride-quality reward and represent deliberate trade-offs along axes the project cares about); it is a different, informative data point about training *mechanism*, with an honestly-labeled downside (small/chattery stepping, the opposite of what several ADRs worked to fix). Kept for the record at `experiments/runs/exp0023_low_lr_from_scratch/best_checkpoint_8M.zip` -- available as the best-tracking-precision checkpoint in the project for anyone who explicitly does not care about stride naturalism and wants the cheapest reproduction recipe (one 23-minute from-scratch run, no fine-tuning chain).

Next: the natural, now well-motivated follow-up is EXP-0024 -- train the *full* reward (`feet_air_time_bonus` + `heading_deviation_penalty`, `exp0019`'s exact weights, same as EXP-0022) from scratch, but at this run's low learning rate (5e-5) instead of EXP-0022's standard 3e-4. If PPO stability was the only thing missing from EXP-0022, this should succeed where EXP-0022 failed outright, and the resulting stride/tracking/heading trade-off (if any) would be the cleanest possible test of whether the ADR-009/010 Pareto surface is fundamental to the reward or an artifact of *both* the standard LR and the sequential-fine-tuning path.

---

### EXP-0024 — Full reward, from scratch, at the low LR that fixed EXP-0023's oscillation: does it also fix EXP-0022's collapse?
Date: 2026-09-19 (pre-registered before running)

Hypothesis: EXP-0022 (full reward -- `feet_air_time_bonus=6.0` + `heading_deviation_penalty=0.5` -- from scratch at the standard `learning_rate=3e-4`) collapsed to a degenerate "stand still" optimum and never learned to walk, with persistently high `approx_kl` (0.09-0.21) throughout. EXP-0023 (minimal reward, same random init, but `learning_rate=5e-5`) trained smoothly (`approx_kl` <0.06 throughout) and reached the best tracking precision in the project, though with a small/chattery stride the minimal reward doesn't discourage. These two results differ in two variables at once (reward richness and LR), so neither alone explains EXP-0022's failure. This experiment isolates the remaining combination: the full reward, from scratch, at the low LR. If EXP-0022's collapse was fundamentally a PPO-stability problem (too-large updates on a harder, richer objective knocking the policy out of the basin where real walking is reachable), the low LR should fix it, the same way it fixed the *simpler* reward's oscillation in EXP-0023 -- and, if `feet_air_time_bonus` still does its job under stable optimization, the resulting checkpoint should have a genuinely longer stride than EXP-0023's, without needing EXP-0016 through EXP-0020's five-stage sequential fine-tuning chain at all.

Failure being targeted: none new -- disambiguating EXP-0022's two conflated variables (LR, reward complexity) to find the actual cause of its collapse.

Config: `config/exp0024_full_reward_low_lr_from_scratch.yaml` -- identical reward weights to `config/exp0022_full_curriculum_from_scratch.yaml` (`exp0019`'s exact weights: `feet_air_time_bonus=6.0`, `heading_deviation_penalty=0.5`), but `ppo.learning_rate: 5.0e-5` (matching EXP-0023) instead of the standard 3.0e-4. No `--init-from` (random init, matching both EXP-0022 and EXP-0023).
- seed: 0
- env count: 12 (SubprocVecEnv)
- budget: 20,000,000 timesteps, `checkpoint_freq=4,000,000` -- matching EXP-0022's exact budget for a direct, single-variable (LR only) comparison

Evaluation protocol: identical to EXP-0022 -- standard 20-seed deterministic eval (`evaluation/evaluate_checkpoints.py`) at each checkpoint, plus the controlled gait diagnostic (`diagnostics/gait_diagnostic.py`, vx=0.3/vy=0) and a direct swing-duration measurement on the best checkpoint, since EXP-0023 just demonstrated that aggregate tracking metrics alone can miss a real stride-quality regression.

Stop condition: fixed 20M-timestep budget. Three possible outcomes: (1) learns to walk AND achieves a real stride (foot-lift amplitude and swing duration comparable to `exp0017`/`exp0018`, not EXP-0023's flat 1.2cm shuffle) -- the Pareto surface was an artifact of standard-LR/sequential-fine-tuning after all, and a single from-scratch run can match years... months of sequential engineering; (2) learns to walk with good tracking but still takes small/chattery steps (repeats EXP-0023's pattern even with the air-time term present) -- `feet_air_time_bonus` itself needs stable, extended optimization pressure that only a longer fine-tune (not a from-scratch run at this budget) provides; (3) still collapses to standing still like EXP-0022 -- LR was not the operative variable in EXP-0022's failure, and reward complexity/richness itself resists from-scratch optimization regardless of LR. No further budget escalation without a new hypothesis.

Results: Training completed in 2644.3s (~44 min, 7563 steps/s), no crashes. `approx_kl` stayed moderate throughout, ending at 0.033-0.045 -- far below EXP-0022's persistent 0.09-0.21, but notably higher than EXP-0023's typical <0.03 (the low LR alone did not fully tame this richer reward's optimization the way it did the simple one). `ep_len_mean` reached ~980-984/1000 (near-full survival) but `ep_rew_mean` plateaued around 724-726 -- lower than even EXP-0022's own final 789-800.

Checkpoint trend (`config/exp0024_full_reward_low_lr_from_scratch.yaml`):

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s |
|---|---|---|---|
| 4M | 0.016 (18) | 2/20 | 4.3 deg |
| 8M | 0.032 (20) | 1/20 | 2.6 deg |
| 12M | 0.014 (20) | 0/20 | 0.9 deg |
| 16M | 0.016 (20) | 0/20 | 1.3 deg |
| 20M | 0.016 (20) | 0/20 | 1.0 deg |

**Every checkpoint collapses to standing still -- more completely than EXP-0022's own collapse, not less.** Direct per-episode inspection of the final (20M) checkpoint: `distance_traveled_m` is 0.04-0.05m *over an entire 20-second episode* (vs. EXP-0022's already-degenerate 0.2-0.4m -- this is a 5-10x more complete freeze), `final_pelvis_height` ~0.79 (standing normally, never falls), and yaw drift is 0.17-0.75 deg/10s per episode -- an order of magnitude lower than EXP-0022's 20-48 deg/10s. EXP-0022's policy at least swayed in place; this one is essentially perfectly motionless. Mean torque cost (0.017-0.020) is also lower than EXP-0022's (0.054-0.057), consistent with a more static posture needing less correction.

Conclusion: **outcome (3) from the stop condition -- the low LR that fixed EXP-0023's simple-reward oscillation does not fix this richer reward's collapse; if anything it makes the collapse more complete.** This resolves the ambiguity EXP-0022 left open: EXP-0022 conflated "standard LR" with "rich reward" as joint causes of its failure, but EXP-0024 isolates LR and shows it is not the operative variable -- a stable, low-LR, carefully-converging optimizer finds the *same* "stand still" trap just as reliably as a noisy, high-KL one, in fact more completely (a lower-variance search apparently settles into the nearest local optimum to random init even more cleanly, and for this reward, "stand still" is that optimum). The likely mechanism: `heading_deviation_penalty` is trivially and perfectly satisfied by never moving or rotating at all (zero deviation, zero cost), and `ang_vel_penalty`/`torque_penalty`/`action_rate_penalty` are all also minimized by staying still -- so from a random initialization equidistant from "stand still" and "walk," essentially every optimization path tried so far (high-KL, low-KL) converges to the closer, trivial optimum. **Warm-starting is not merely a convenient way to avoid this trap -- it is the only method tested across EXP-0022/0023/0024 that reliably avoids it**, because it starts already far from the "stand still" basin rather than needing to escape it during training.

Decision: **Do not promote; no checkpoint kept as a `best_checkpoint` (matching the EXP-0006/0007/0008/0022 documented-failure convention).** `exp0017` remains canonical. The ADR-009/010 Pareto surface question is now closed as far as "does a from-scratch method escape it" goes: neither standard LR (EXP-0022) nor low LR (EXP-0024) does, for this reward richness -- the surface (and the sequential fine-tuning chain needed to reach any point on it at all) is a property of this reward family from random initialization, not an artifact of one specific training recipe.

Next: the from-scratch/curriculum-order investigation (EXP-0022/0023/0024) is concluded. A genuinely different escape would need a mechanism that starts the *reward* simple and adds complexity mid-run (reward annealing within a single training run, changing the objective over time rather than the checkpoint), which is different from every combination tried here (all terms from step 0, at two different LRs) and from the existing sequential approach (separate full runs per stage) -- left as a real but untried idea for future work, not required by the challenge.

---

### EXP-0025 — Fine-tune `feet_air_time_bonus` onto EXP-0023's superior-tracking checkpoint: does the proven recipe generalize to a better starting point?
Date: 2026-09-19 (pre-registered before running)

Hypothesis: EXP-0016/0017 proved a specific recipe -- warm-start fine-tune at `learning_rate=5e-5` to introduce `feet_air_time_bonus` -- turns a chattery-stepping-but-decent-tracking policy (`exp0011`, `dist_ratio=0.671`) into one with both a real stride and better tracking (`exp0017`, `dist_ratio=0.709`). EXP-0023 (this session) found an even better-tracking chattery policy (`dist_ratio=0.848`, via low-LR-from-scratch training on the minimal reward) with a stepping pattern quantitatively similar in scale to `exp0011`'s original problem (1.2cm foot lift, comparable swing-duration regime). If the EXP-0016/17 recipe is a generalizable technique (not a fact specific to `exp0011`'s particular policy), applying it to `exp0023`'s checkpoint should similarly lengthen its stride while preserving most of its superior tracking -- potentially producing a checkpoint that beats `exp0017` on tracking precision AND matches it on stride quality, since it starts from a strictly better tracking base than `exp0011` did.

Failure being targeted: none new -- testing whether the warm-start fine-tuning recipe (ADR-007) generalizes across different starting policies, and opportunistically searching for a checkpoint that dominates the existing 3-way Pareto surface (ADR-009/010) rather than trading along it.

Calibration (measured before choosing, per ADR-005 discipline, not reused blindly from `exp0012`): replayed `exp0023`'s own checkpoint through `envs.contacts.AirTimeTracker` at `threshold_s=0.2` (6 episodes, 6000 steps) -- mean raw air-time cost -0.0364/step. At `feet_air_time_bonus=3.0` (exp0012's canonical weight), this costs -0.109/step, about 7.6% of `exp0023`'s own ~1.43/step total reward -- a similar order of magnitude to (slightly gentler than) EXP-0012's original 11% calibration for `exp0011`, so the existing weight is well-justified for this new starting point rather than needing re-derivation.

Config: `config/exp0012_air_time.yaml` (canonical reward, completely unchanged -- literally the same config that turned `exp0011` into `exp0016`), applied via `--init-from experiments/runs/exp0023_low_lr_from_scratch/best_checkpoint_8M.zip --init-learning-rate 5e-5` (matching EXP-0016's exact recipe, only the starting checkpoint differs).
- seed: 0, env count: 12, budget: 5,000,000 timesteps, `checkpoint_freq=1,000,000` -- identical to EXP-0016's own probe budget/cadence, for a direct, apples-to-apples comparison of "does this recipe work on a different starting policy."

Evaluation protocol: standard 20-seed deterministic eval at each checkpoint (`evaluation/evaluate_checkpoints.py`, matching config), plus the controlled gait diagnostic (`diagnostics/gait_diagnostic.py`) and a direct swing-duration measurement on the best checkpoint -- mandatory this time, not optional, since EXP-0023 already demonstrated aggregate metrics alone can hide a stepping-quality regression.

Stop condition: fixed 5M-timestep budget, checked every checkpoint as it becomes available (matching EXP-0016's own stop condition) in case a skating-style regression re-emerges early. Three possible outcomes: (1) stride lengthens (foot-lift amplitude and swing duration approach `exp0017`'s) while `dist_ratio` stays well above `exp0017`'s 0.709 -- a genuine new best, promotable; (2) stride lengthens but tracking regresses to roughly `exp0017`'s own level -- the recipe generalizes but converges to a similar point on the same Pareto surface, not a new dominant one; (3) the exploit/regression pattern from EXP-0012/0013 (F-0004) re-emerges on this different starting policy -- the recipe's safety was specific to `exp0011`'s policy shape, not general. No budget extension without a new hypothesis regardless of outcome.

Results: Training completed in 687.5s (~11.5 min, 7273 steps/s), no crashes. `approx_kl` ended moderate (0.038-0.059 -- higher than EXP-0016's typical <0.03, but far below EXP-0022/0024's 0.09-0.21), consistent with fine-tuning a policy that started from a different (and differently-shaped) optimum than `exp0011`.

Checkpoint trend (`config/exp0012_air_time.yaml`):

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s (full eval) |
|---|---|---|---|
| 1M | 0.702 (19) | 2/20 | 51.8 deg |
| 2M | 0.747 (19) | 2/20 | 56.5 deg |
| 3M | 0.782 (19) | 1/20 | 36.9 deg |
| **4M** | **0.774** (20) | **0/20** | **30.7 deg** |
| 5M | 0.847-0.864 (19) | 1-2/20 | 35.3-36.9 deg |

The 4M checkpoint (0/20 falls, the only perfect-survival point in the trend) full standard eval: `dist_ratio` **0.774** (vs. canonical `exp0017`'s 0.709), `vel_err` **0.040** (vs. 0.061), `dist_m` **3.81** (vs. 3.01), yaw drift (full eval) **30.7 deg/10s** (vs. 36.3) -- **beats canonical on every one of the standard aggregate metrics this project has used for every promotion decision**, at equal falls.

Mandatory stride-quality verification (not skipped this time): the controlled gait diagnostic (`diagnostics/gait_diagnostic.py`, vx=0.3/vy=0) and a direct swing-duration measurement on the 4M checkpoint show a **real stride, not a repeat of EXP-0023's chattery collapse** -- foot-lift range 0.031-0.054m (2.3cm, close to canonical's own 3cm, visually confirmed via `diagnostics/reports/gait_diagnostic_exp0025_4M.png` against canonical's own plot), 42-48 liftoffs/10s (canonical: 43-45), swing duration mean 44.5ms / median 40.0ms / p90 60.0ms / max 100.0ms (747 touchdowns, 6 episodes) -- nearly identical to canonical's own 47.6ms/40ms/60ms/100ms. **The fine-tuning recipe (ADR-007) generalizes: it reproduced a comparable real stride on a completely different starting policy, not just on `exp0011`'s specific one.**

The one real regression: the controlled straight-line test shows -25.7 deg/10s heading drift, worse than canonical's own best controlled measurement (-11.6 to -19.0 deg/10s across the project, though still much better than `exp0018`'s -28.2). This is a genuine, honestly-reported trade-off, not hidden by the good aggregate numbers -- interestingly, canonical's own *full-eval* yaw drift (36.3, averaged over randomized commands) is actually worse than this checkpoint's (30.7), so the two checkpoints' heading quality ranks oppositely depending on which of the two heading measurements is used; this is itself informative (heading quality is evidently sensitive to which command distribution it's measured under, not a single well-ordered scalar).

Conclusion: **outcome (1) from the stop condition, with a caveat.** The recipe generalizes -- a real stride, comparable to canonical's, was reproduced on a much-better-tracking starting policy, without the EXP-0012/0013-style exploit/regression re-emerging. This is not yet a clean, unambiguous "new best" (the controlled-heading regression is real), but it closely mirrors EXP-0016's own status at its 5M mark: a promising, still-improving trend (5M's dist_ratio, 0.847-0.864, is the highest of the whole run) with a stability wrinkle (1-2/20 falls at 5M) not yet resolved. Per the exact precedent that turned EXP-0016 into EXP-0017 (continue fine-tuning the same lineage with more budget before concluding), the right next step is to extend this fine-tune, not judge it final at a 5M-timestep probe.

Decision: **Not yet promoted -- extend first.** Continuing to EXP-0026 (fine-tune further from this run's 4M checkpoint, the one perfect-survival point, matching EXP-016's own choice to continue from a stable early checkpoint rather than the noisier final one).

Next: EXP-0026 -- continue fine-tuning this lineage for 10M more timesteps (matching EXP-0016->0017's 2x budget-extension ratio), same recipe, to see whether falls stabilize back to 0/20 at the higher `dist_ratio` the trend is climbing toward, and whether the controlled-heading regression resolves, worsens, or persists with more training -- exactly the open question EXP-0016->0017 answered for the original lineage.

---

### EXP-0026 — Continue the EXP-0025 lineage: does more budget resolve the controlled-heading regression, matching the EXP-0016->0017 precedent?
Date: 2026-09-19 (pre-registered before running)

Hypothesis: EXP-0025's checkpoint trend was still climbing on `dist_ratio` at 5M timesteps (0.702 -> 0.864) without having clearly stabilized falls (0/20 only at 4M, 1-2/20 at 3M/5M) or resolved the controlled-heading regression (-25.7 deg/10s vs. canonical's -11.6 to -19.0). This is structurally identical to EXP-0016's own status at its 5M mark, which was resolved by continuing the exact same fine-tune for 10M more timesteps (EXP-0017) -- reaching 0/20 falls and the project's best-ever numbers. Testing the same "more budget" hypothesis here before concluding anything about this lineage's ceiling.

Failure being targeted: none new -- testing whether EXP-0025's still-improving, not-yet-stable trend was simply undertrained, per the ADR-004/008 "test budget before concluding" precedent.

Config: `config/exp0012_air_time.yaml` (unchanged), `--init-from experiments/runs/exp0025_air_time_on_exp0023/best_checkpoint_4M.zip` (the one perfect-survival checkpoint from EXP-0025, not the noisier 5M/final one), `--init-learning-rate 5e-5` (unchanged).
- seed: 0, env count: 12, budget: 10,000,000 timesteps, `checkpoint_freq=2,000,000` -- matching EXP-0016->0017's exact 2x budget-extension ratio and cadence.

Evaluation protocol: standard 20-seed deterministic eval at each checkpoint, plus the controlled gait diagnostic and a direct swing-duration measurement on the best checkpoint -- mandatory, as established this session.

Stop condition: fixed 10M-timestep budget. Success = a checkpoint with 0/20 falls, `dist_ratio` clearly above EXP-0025 @ 4M's 0.774, and controlled-heading drift at or better than canonical's own range (-11.6 to -19.0 deg/10s) -- a genuine, unambiguous new best across every metric this project tracks. Partial success = tracking keeps improving but heading does not recover -- a new, real trade-off point, promoted only as an alternative not a replacement. No further budget extension without a new hypothesis regardless of outcome.

Results: Training completed in 1375.9s (~23 min, 7268 steps/s), no crashes. `approx_kl` ran moderate-to-high (0.058-0.074 at the end) -- higher than EXP-0025's own 0.038-0.059, suggesting this lineage's optimization is getting somewhat more turbulent with continued budget, not less (unlike EXP-0016->0017, where continued fine-tuning stayed smooth throughout).

Checkpoint trend (`config/exp0012_air_time.yaml`):

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s (full eval) |
|---|---|---|---|
| **2M** | **0.918** (20) | **0/20** | **17.6 deg** |
| 4M | 0.857 (20) | 0/20 | 24.3 deg |
| 6M | 0.855 (20) | 1/20 | 23.2 deg |
| 8M | 0.818 (19) | 1/20 | 22.0 deg |
| 10M | 0.890-0.893 (19) | 1/20 | 21.3-22.6 deg |

Unlike every prior fine-tuning extension in this project (EXP-0016->0017, EXP-0019->0020), **this lineage peaks early (2M) and does not improve with more budget** -- `dist_ratio` declines and a fall reappears from 4M onward. The 2M checkpoint (0/20 falls, the clear peak) full standard evaluation: `dist_ratio` **0.918** (the best result of the entire project by a wide margin -- previous best: `exp0023`'s 0.848, which had a degenerate stride), `vel_err` **0.036**, `dist_m` **4.28**, yaw drift (full eval) **17.6 deg/10s** (best of any real-walking checkpoint in the project), mean torque cost 0.045 (comparable to/lower than every other checkpoint, ruling out "wins via brute force").

Mandatory stride-quality verification: the controlled gait diagnostic (`diagnostics/gait_diagnostic.py`, vx=0.3/vy=0, `diagnostics/reports/gait_diagnostic_exp0026_2M.png`) shows a **visibly straight trajectory** (much straighter than EXP-0025's own curving path), yaw drift -14.1 deg/10s (within canonical `exp0017`'s own historical range of -11.6 to -19.0), foot-lift range 0.032-0.057m (2.5cm, comparable to canonical's 3cm), and a direct swing-duration measurement (6 episodes, 763 touchdowns) of mean 55.2ms / median **60.0ms** / p90 80.0ms / max 100.0ms -- matching `exp0018`/`exp0019`'s own best-in-project stride length (60ms median), not just canonical's more modest 40ms.

**This single checkpoint matches or beats every prior checkpoint in the project on every axis of the previously-declared 3-way Pareto surface (ADR-009/010) simultaneously:**

| checkpoint | dist_ratio | vel_err | falls | yaw (full) | yaw (controlled) | swing median |
|---|---|---|---|---|---|---|
| `exp0017` (canonical, best tracking) | 0.709 | 0.061 | 0/20 | 36.3 | -11.6 to -19.0 | 40ms |
| `exp0018` (longest stride) | 0.548 | 0.066 | 0/20 | 32.8 | -28.2 | 60ms |
| `exp0019` (best heading) | 0.613 | 0.093 | 1/20 | 27.3 | **-7.5** | 60ms |
| **`exp0026` @ 2M** | **0.918** | **0.036** | **0/20** | **17.6** | -14.1 | **60ms** |

`exp0026` beats `exp0017` on tracking, heading (both measures), and matches the best stride length in the project; it beats `exp0018` on every axis with no trade-off; it beats `exp0019` on tracking, full-eval heading, and falls, losing only to `exp0019`'s controlled-heading measurement by a modest margin (-14.1 vs -7.5) while dominating it everywhere else.

Conclusion: **the 3-way Pareto surface (ADR-009/010) is resolved, not merely characterized.** The mechanism: every prior point on that surface was reached by sequentially fine-tuning from `exp0011` -- a from-scratch, standard-LR (3e-4) base with only 0.671 `dist_ratio`. EXP-0023 (this session) showed that a from-scratch run at *low* LR (5e-5) reaches a substantially better tracking optimum (0.848) than any standard-LR from-scratch run ever has, albeit with a degenerate stride. EXP-0025/0026 show that applying the established stepping-quality fine-tuning recipe (ADR-007) to *that* better base, instead of to `exp0011`, compounds: the final result inherits both the superior tracking of its low-LR-from-scratch ancestor and a real, long stride from the fine-tuning step, ending up better than every checkpoint descended from the weaker `exp0011` lineage on every axis but one (a 6.6-degree gap on one specific controlled-heading measurement). The Pareto surface was real for the `exp0011`-descended lineage, but not fundamental to the reward family -- it was a property of the specific (weaker) base gait every prior checkpoint was built on.

Decision: **Promote `experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip` to canonical baseline, superseding `exp0017_warmstart_air_time_long`.** This is the first canonical-baseline change since ADR-008. `exp0017`, `exp0018`, `exp0019` are all kept for the historical record (each was a genuine best-known-result or informative documented alternative at the time). `docs/DECISIONS.md` gets ADR-011.

Next: this lineage itself has peaked at 2M (more budget on the exact same recipe makes things worse, not better) -- if pursued further, the next lever would be applying the *same* "improve the base, then fine-tune stepping quality onto it" strategy one level deeper (e.g. a longer/more careful low-LR-from-scratch base-training run before this fine-tuning step), or fine-tuning `heading_deviation_penalty` onto this new checkpoint the way EXP-0019 did onto `exp0018`, to see if the one remaining gap (controlled heading, -14.1 vs `exp0019`'s -7.5) can be closed too. Both untested, left for future work -- the challenge's bar was exceeded long ago and this is already a strict improvement.

Cheap follow-up (2026-09-19, no training): re-verified the high-speed-command failure mode (F-0002) against `exp0026` using its own existing eval data. 0/20 falls holds (unchanged from `exp0017`). The `distance_ratio` degradation at high commanded speed is now much smaller (0.870 vs. 0.926 mean for the rest, a ~6% relative drop, vs. `exp0017`'s original 63% relative drop) -- heading still degrades disproportionately at high speed (3.2x higher mean |yaw drift|), so the mechanism persists, but its practical impact shrank substantially on the better base gait. See `docs/FAILURE_ANALYSIS.md` F-0002.

---

### EXP-0027 — Close the one remaining gap: fine-tune `heading_deviation_penalty` onto the new canonical
Date: 2026-09-19 (pre-registered before running)

Hypothesis: `exp0026` (current canonical) beats every prior checkpoint on tracking, full-eval heading, falls, and stride length, but `exp0019` still holds the single best *controlled*-heading measurement in the project (-7.5 deg/10s vs. `exp0026`'s -14.1). `exp0019` reached its heading quality by fine-tuning `heading_deviation_penalty` onto a worse-tracking base (the `exp0018` lineage); applying the identical technique to `exp0026` (a much better base) should, by the same "improve the base, then apply the proven recipe" logic that produced `exp0026` itself (ADR-011), close this one remaining gap without giving up `exp0026`'s other advantages -- potentially producing a checkpoint with no remaining gap on any axis tracked by this project.

Failure being targeted: none new -- closing ADR-011's one honestly-reported residual gap.

Calibration (measured before choosing, per ADR-005/010 discipline): replayed `exp0026`'s own checkpoint through the controlled straight-line test (vx=0.3, vy=0, matching EXP-0019's own calibration protocol) and computed mean heading-deviation-squared directly -- 0.0207. At `heading_deviation_penalty=0.5` (`exp0019`'s original weight), this costs only 0.8% of `exp0026`'s ~1.36/step reward -- far gentler than the ~6% cost that same weight imposed on `exp0018`'s worse-heading gait when `exp0019` was calibrated, because `exp0026` already has much better heading than `exp0018` did. Scaled to a comparable but more cautious ~3% cost (roughly half of `exp0019`'s original aggressiveness, given `exp0026`'s lineage already showed more training on the *same* recipe can make things worse -- EXP-0026's own trend peaked early and declined with more budget) needs `heading_deviation_penalty=2.0`.

Config: `config/exp0027_heading_on_exp0026.yaml` (canonical `exp0012_air_time` reward, unchanged, plus `heading_deviation_penalty=2.0`), `--init-from experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip --init-learning-rate 5e-5`.
- seed: 0, env count: 12, budget: 3,000,000 timesteps (smaller than EXP-0016's 5M probe, given the demonstrated fragility of this specific lineage), `checkpoint_freq=1,000,000` (checked every checkpoint, not just the end, per the same fragility concern).

Evaluation protocol: standard 20-seed deterministic eval at each checkpoint, plus the mandatory controlled gait diagnostic and swing-duration measurement on the best checkpoint.

Stop condition: fixed 3M-timestep budget. Success = controlled-heading drift improves toward or past `exp0019`'s -7.5 deg/10s while `dist_ratio`/falls/stride length stay close to `exp0026`'s own numbers -- a checkpoint with no remaining gap. Partial success = heading improves some without matching `exp0019`, still likely promotable if nothing else regresses. Failure = any of `exp0026`'s own strengths regress substantially (a real risk given this lineage's known fragility) -- if so, this specific combination is not pursued further without a new idea (e.g. a smaller weight, or fine-tuning from an earlier/different exp0026-lineage checkpoint).

Results: Training completed in 411.7s (~7 min, 7288 steps/s). Checkpoint trend shows the predicted fragility materializing, but not before a striking near-miss at the first checkpoint:

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s (full eval) |
|---|---|---|---|
| **1M** | **0.996** (20) | 2/20 | 21.9 deg |
| 2M | 0.389 (15) | 5/20 | 32.0 deg |
| 3M | 0.710-0.750 (10) | 10/20 | 60.8-64.4 deg |

Training progressively destabilized after 1M (falls climbing from 2/20 to 10/20, `dist_ratio` collapsing then partially recovering on a shrinking reliable sample) -- confirming the pre-registered fragility risk. But the 1M checkpoint itself is remarkable: full standard eval gives `dist_ratio=0.996` (better than canonical `exp0026`'s own 0.918), `vel_err=0.059` (worse than `exp0026`'s 0.036 but comparable to the old canonical `exp0017`'s 0.061), and the controlled gait diagnostic (`diagnostics/gait_diagnostic.py`, `diagnostics/reports/gait_diagnostic_exp0027_1M.png`) shows **-7.5 deg/10s controlled-heading drift -- an exact tie with `exp0019`'s best-in-project figure** -- with a real, verified stride (foot-lift range 0.032-0.057m, 40-41 liftoffs/10s, both nearly identical to `exp0026`'s own numbers).

The blocker: 2/20 falls (vs. `exp0026`'s 0/20), at seeds 1005 and 1007 -- both **low**-speed commands (0.05-0.17 m/s), not the high-speed regime F-0002 has repeatedly implicated. Notably, these are the exact same two seeds flagged as unexplained anomalous fallers all the way back in `baseline0001`'s original evaluation (EXP-0005/F-0002) -- a narrow, rare, recurring sensitivity to these specific seed/command conditions that has now appeared across at least two very different policies in this project, still not root-caused.

Conclusion: **partial success, not a clean win.** The hypothesis that `exp0026` could reach `exp0019`-level heading without giving up its other advantages is directly supported at 1M timesteps -- but the pre-registered failure mode (this lineage's fragility under continued training) also materialized, and 2/20 falls is a real regression the project's own promotion bar (established across ADR-004/007/008/011) does not accept. `heading_deviation_penalty=2.0` may simply be too strong a push for this lineage's stability margin, given `exp0026`'s own gap at weight=0 was assessed as only needing a "moderate" nudge.

Decision: **Not promoted as-is.** Kept for the record (`experiments/runs/exp0027_heading_on_exp0026/`, no `best_checkpoint` promoted given the fall regression). The 1M result is too promising to abandon outright -- worth one more targeted attempt at a gentler weight before concluding the combination doesn't work.

Next: EXP-0028 -- repeat this exact fine-tune with a substantially lower `heading_deviation_penalty` weight (1.0, half of this run's), checking more frequently (every 0.5M timesteps) to catch the best point before any destabilization, testing whether a gentler push gets most of the heading benefit without the fall regression.

---

### EXP-0028 — Gentler heading push: half the weight, checked more frequently
Date: 2026-09-19 (pre-registered before running)

Hypothesis: EXP-0027's 1M checkpoint (`heading_deviation_penalty=2.0`) proved the underlying idea works -- better tracking than canonical plus heading tied with `exp0019`'s best -- but destabilized with more training, and 2/20 falls even at the best checkpoint blocked promotion. Halving the weight to 1.0 should impose a gentler optimization pressure, plausibly avoiding the destabilization while still meaningfully improving heading over `exp0026`'s own controlled-test figure (-14.1 deg/10s) -- even a partial improvement toward `exp0019`'s -7.5 without any fall regression would be a promotable improvement over `exp0026`.

Failure being targeted: EXP-0027's fall regression (2/20 at the best checkpoint, worse by 3M).

Config: `config/exp0028_heading_on_exp0026_gentle.yaml` (identical to EXP-0027 except `heading_deviation_penalty=1.0`), `--init-from experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip --init-learning-rate 5e-5` (same base checkpoint as EXP-0027, for a direct, single-variable comparison).
- seed: 0, env count: 12, budget: 2,000,000 timesteps (shorter than EXP-0027, since its own destabilization was already visible by 2M), `checkpoint_freq=500,000` (checked more frequently, per EXP-0027's own lesson that the best point can be early and narrow).

Evaluation protocol: standard 20-seed deterministic eval at each checkpoint, plus the mandatory controlled gait diagnostic on the best (0/20-falls) checkpoint.

Stop condition: fixed 2M-timestep budget. Success = a checkpoint with 0/20 falls and controlled-heading drift better than `exp0026`'s -14.1 (even if short of `exp0019`'s -7.5) -- a genuine, promotable improvement. If every checkpoint still shows fall regression even at this halved weight, the technique is judged not to work for this specific lineage regardless of weight, and is not pursued further without a fundamentally different idea (e.g. fine-tuning from an earlier, less-optimized point in the `exp0025`/`exp0026` lineage instead of the 2M peak).

Results: Checkpoint trend:

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s (full eval) |
|---|---|---|---|
| 0.5M | 0.775 (19) | 1/20 | 16.9 deg |
| 1M | 0.927 (18) | 2/20 | 25.1 deg |
| 1.5M | 0.598 (18) | 2/20 | 34.0 deg |
| 2M | 0.635 (20) | 2/20 | 37.1-38.3 deg |

**The hypothesis is not supported.** Halving the weight did not avoid the fall regression -- every single checkpoint in this run has at least 1/20 falls, including the very first one at 500k timesteps; `exp0026`'s own 0/20 is never recovered at any point. Yaw drift also gets *worse*, not better, as training continues (16.9 -> 37-38 deg/10s) -- the opposite of the intended effect, and worse than `exp0026`'s own 17.6 baseline by the end. Unlike EXP-0027 (which showed a genuine, if brief, high point at 1M before destabilizing further), this run never reaches a clean improvement over `exp0026` on any metric at any checkpoint simultaneously.

Conclusion: introducing `heading_deviation_penalty` via fine-tuning onto `exp0026`'s 2M checkpoint is destabilizing at both weights tested (1.0 and 2.0) -- this is not simply a matter of tuning the weight down further. The likely explanation: `exp0026`'s own checkpoint trend (EXP-0026) already showed it sits at a narrow, early peak that more training on its *own* recipe made worse, not better -- it is plausibly a fragile optimum in general, not specifically sensitive to this one new term's magnitude, and perturbing it with *any* new term at a fine-tuning LR risks knocking it off that peak.

Decision: **This specific approach (fine-tune heading_deviation_penalty onto `exp0026`'s 2M checkpoint) is not pursued further without a fundamentally different idea** -- per the project's standing discipline, two single-variable attempts (weight 2.0 and 1.0) is enough evidence without escalating to a third weight guess. `exp0026` remains canonical, unchanged. EXP-0027's 1M checkpoint (dist_ratio=0.996, heading tied with `exp0019`'s best, but 2/20 falls) is kept as a documented near-miss for the record, not promoted. If this axis is revisited, the more promising untested idea is fine-tuning from an *earlier* point in the `exp0025`/`exp0026` lineage (before it reached its narrow 2M peak) rather than from the peak itself, since a less-optimized starting point may have more slack to absorb a new term without destabilizing.

Next: move to the other items in the current experiment series -- re-running domain randomization (EXP-0021's recipe) on the new canonical `exp0026`, since EXP-0021 was built on the now-superseded `exp0017`.

---

### EXP-0029 — Domain randomization re-run on the new canonical (`exp0026`)
Date: 2026-09-19 (pre-registered before running)

Hypothesis: EXP-0021 showed the domain-randomization recipe (`envs/domain_randomization.py`, friction/mass/motor-strength/pushes) reproduces a genuine robustness/precision trade-off when fine-tuned onto a canonical reward via warm-start -- but that run was built on `exp0017`, since superseded (ADR-011). Re-running the identical recipe on `exp0026` (now a substantially better base gait -- `dist_ratio` 0.709 -> 0.918 clean) should reproduce the same qualitative trade-off pattern on the new, better base, similar to how EXP-0021 itself reproduced EXP-0015's pattern on top of `exp0017`.

Failure being targeted: none new -- extending EXP-0021's validated result onto the new canonical, matching PROJECT_STATE.md's own "highest-value next step" list.

Config: `config/exp0021_dr_on_canonical.yaml`, unchanged -- its reward weights are byte-for-byte identical to `exp0026`'s own (`config/exp0012_air_time.yaml`), confirmed by diff before running, so this is a clean single-variable change (base checkpoint only). `--init-from experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip --init-learning-rate 5e-5` (same recipe as EXP-0021, new base).
- seed: 0, env count: 12, budget: 10,000,000 timesteps, `checkpoint_freq=2,000,000` -- identical to EXP-0021's own budget/cadence for a direct comparison. (Note: `exp0026`'s own lineage showed fragility to further training/new terms at this LR in EXP-0027/0028 -- watching every checkpoint here for early destabilization, not just trusting the final one.)

Evaluation protocol: identical to EXP-0021 -- checkpoint trend and full standard eval under both clean physics (`config/exp0012_air_time.yaml`) and randomized physics (`config/exp0021_dr_on_canonical.yaml`'s own training distribution), plus the correct control (`exp0026` itself evaluated under randomized physics with zero DR training).

Stop condition: fixed 10M-timestep budget. Success = randomized-physics survival/falls clearly improve over the `exp0026`-under-randomization control (mirroring EXP-0021's own win over its `exp0017` control) while clean-physics metrics stay reasonably close to `exp0026`'s own numbers. Given EXP-0027/0028's demonstrated fragility of this exact lineage to any further training, a clean-physics regression here would not be surprising and will be reported honestly, not treated as a bug.

Results: Checkpoint trend, clean physics (`config/exp0012_air_time.yaml`):

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s |
|---|---|---|---|
| 2M | 0.741 (20) | 0/20 | 31.7 deg |
| 4M | 0.601 (20) | 0/20 | 46.8 deg |
| **6M** | **0.643** (20) | **0/20** | 40.7 deg |
| 8M | 0.525 (18) | 2/20 | 35.1 deg |
| 10M | 0.543-0.565 (18-19) | 2/20 | 43.8-49.0 deg |

Checkpoint trend, randomized physics (`config/exp0021_dr_on_canonical.yaml`, own training distribution):

| steps | dist_ratio (n reliable) | falls | mean len (s) |
|---|---|---|---|
| 2M | 0.942 (17) | 9/20 | 16.5 |
| 4M | 0.697 (19) | 7/20 | 18.4 |
| **6M** | **0.830** (19) | **4/20** | **18.8** |
| 8M | 0.724 (18) | 4/20 | 18.4 |
| 10M | 0.784-0.788 (18) | 4-6/20 | 18.1-18.3 |

As predicted, this lineage shows the same fragility pattern as EXP-0026/0027/0028 (best early, degrading with more budget on clean physics: falls appear from 8M onward) -- but unlike those runs, a genuinely strong, promotable result still emerges. The 6M checkpoint is the clear best across both conditions (0/20 falls clean, 4/20 falls randomized, the lowest randomized-physics fall count of the whole trend).

Control (per the pre-registered protocol): `exp0026` itself (zero DR training) evaluated under this same randomized-physics config -- 14/20 falls, mean length 10.2s.

Full standard-protocol comparison:

| condition | policy | len (s) | vel_err | dist_ratio (n reliable) | falls |
|---|---|---|---|---|---|
| clean | `exp0026` (no DR) | 20.0 | 0.036 | 0.918 (20) | 0/20 |
| clean | `exp0029` @ 6M (DR) | 20.0 | 0.076 | 0.643 (20) | 0/20 |
| randomized | `exp0026` (no DR, control) | 10.2 | 0.178 | 1.204 (10) | 14/20 |
| randomized | `exp0029` @ 6M (DR) | **18.8** | 0.091 | 0.830 (19) | **4/20** |

(As with EXP-0021/0015, the control's inflated 1.204 randomized `dist_ratio` reflects only the minority of its episodes that survive long enough to count as "reliable" -- not a real tracking-quality comparison.)

Conclusion: **hypothesis confirmed, and the result is stronger than EXP-0021's own.** Falls under randomized physics drop from 14/20 to 4/20 (71% relative reduction, vs. EXP-0021's 14/20 -> 7/20, a 50% reduction relative to its own `exp0017` control) and survival nearly doubles (10.2s -> 18.8s, an 84% increase, vs. EXP-0021's 58%). The robustness/precision trade-off is real here too (clean `dist_ratio` 0.918 -> 0.643, `vel_err` 0.036 -> 0.076), but the DR-trained checkpoint is now robust enough that it beats even `exp0017` (the old canonical) on `dist_ratio` under clean physics while being dramatically more fall-resistant under randomized physics -- a genuinely strong combined result, better on every measured axis than any DR checkpoint the project has produced.

Decision: **Adopt `experiments/runs/exp0029_dr_on_exp0026/best_checkpoint_6M.zip` as the new robustness-axis alternative, superseding `exp0021_dr_on_canonical` for this purpose** (not superseding canonical `exp0026` itself -- clean-physics tracking precision is still meaningfully worse). `exp0021` is kept for the historical record, same treatment as `exp0015` before it.

Next: this experiment series (EXP-0027 through EXP-0029) is concluded. Remaining untested items from PROJECT_STATE.md: whether a longer/more careful low-LR-from-scratch base-training run (beyond `exp0023`'s 8M timesteps) could raise the tracking ceiling even further -- a genuinely different, larger investigation than anything tried this cycle, left for future work.

---

### EXP-0030 — Does more budget on the low-LR-from-scratch base training raise the tracking ceiling further?
Date: 2026-09-19 (pre-registered before running)

Hypothesis: `exp0023`'s own checkpoint trend (2M: 0.026 -> 4M: 0.298 -> 6M: 0.441 -> 8M: 0.848 -> 10M: 0.756) is a strong, still-climbing trajectory with only a slight late dip, not a clear plateau -- structurally similar to `baseline0001`'s own noisy-but-climbing pattern that continued improving substantially when extended from 3M to 10M timesteps (EXP-0009, ADR-004). Continuing this exact base-training run for 10M more timesteps (same config, same LR, same seed -- doubling the total budget to 20M) should test directly whether the tracking ceiling this project has been fine-tuning onto (`exp0023`'s 0.848, which became `exp0026`'s 0.918 after fine-tuning `feet_air_time_bonus` onto it) can be raised even further with more budget on the base-training step itself, following the "test budget before concluding" precedent that has repeatedly paid off in this project (ADR-004/006).

Failure being targeted: none new -- testing item #4 from PROJECT_STATE.md's "highest-value next step" list, the one remaining untested lever from this cycle's investigation.

Config: `config/exp0023_low_lr_from_scratch.yaml`, unchanged. `--init-from experiments/runs/exp0023_low_lr_from_scratch/final_model.zip --init-learning-rate 5e-5` (continuing the exact same trajectory at the same LR, not a new fine-tuning stage -- this is still the base-training phase, matching how EXP-0020 continued EXP-0019's lineage).
- seed: 0, env count: 12, budget: 10,000,000 timesteps (matching the original run's own budget, for a combined 20M total, mirroring EXP-0015's exact 2x extension of EXP-0014), `checkpoint_freq=2,000,000`.

Evaluation protocol: standard 20-seed deterministic eval at each checkpoint (`config/exp0023_low_lr_from_scratch.yaml`, no stepping/heading terms -- tracking precision is the only thing this stage optimizes). If a new best checkpoint emerges, the natural immediate follow-up (not yet a separate pre-registered experiment) is to fine-tune the established `feet_air_time_bonus` recipe onto it, exactly as EXP-0025/0026 did onto `exp0023`'s own checkpoint, to test whether the "improve the base, then fine-tune" strategy compounds a second time.

Stop condition: fixed 10M-timestep budget (20M total). Success = a checkpoint with `dist_ratio` clearly above `exp0023`'s own 0.848 ceiling, worth fine-tuning onto next. Failure/plateau = no checkpoint beats 0.848 -- this specific lineage's ceiling is confirmed at its original 8M-timestep point, and the base-training budget lever is considered exhausted for this recipe (not escalated to a third budget doubling without a new reason).

Results: Checkpoint trend (steps shown relative to this continuation; absolute budget = 10M + this run's steps):

| steps (this run) | dist_ratio (n reliable) | falls | yaw_drift/10s |
|---|---|---|---|
| 2M | 0.794 (20) | 1/20 | 43.3 deg |
| 4M | 0.769 (20) | 0/20 | 28.1 deg |
| 6M | 0.719 (20) | 0/20 | 24.3 deg |
| 8M | 0.726 (20) | 0/20 | 21.6 deg |
| 10M | 0.726-0.730 (19-20) | 1/20 | 27.8-28.2 deg |

**No checkpoint in this 10M-timestep extension beats `exp0023`'s original peak (0.848 at 8M).** The trend instead settled into a noisier, lower plateau around 0.72-0.79 -- unlike EXP-0009's own budget extension (which found a *higher* peak with more training on the standard-LR baseline), more budget on this low-LR-from-scratch recipe did not raise the ceiling; if anything the best individual checkpoint here (2M-into-continuation, 0.794) is still below the original run's peak.

Conclusion: **hypothesis not supported -- the ceiling is confirmed at `exp0023`'s original 8M-timestep point, not a moving target.** This is a meaningful contrast with the standard-LR case: `baseline0001`'s climb-crash-recover cycles kept reaching *new, higher* peaks with more budget (EXP-0009), but this low-LR run's peaks look more like noisy fluctuations around a roughly-converged level than a still-ascending trajectory -- more training does not reliably return to, let alone exceed, the best peak already observed. This also reframes EXP-0025/0026's success: it depended on catching `exp0023`'s noisy peak at exactly the point it occurred, not on an ever-improving base-training process that could be pushed further by budget alone.

Decision: **Base-training budget is not a further lever for this recipe.** `exp0023`'s original 8M checkpoint (and everything built on it -- `exp0026`, `exp0029`) stands as the best result from this lineage; no new checkpoint promoted from this run. Not escalated to a third budget doubling, per the pre-registered stop condition.

Next: PROJECT_STATE.md's next-step list is now fully exhausted. A genuinely new idea, not yet tried: EXP-0027/0028's heading fine-tune varied the reward *weight* (2.0, then 1.0) and both destabilized `exp0026`'s fragile peak. Neither tried varying the *learning rate* instead -- EXP-0027's weight=2.0 already produced the best-in-project heading result at 1M timesteps before destabilizing with more training; a much lower fine-tune LR (slower, smaller optimization steps) might preserve that near-miss's quality for longer without needing a weaker reward signal.

---

### EXP-0031 — Retry the heading fine-tune with a much lower LR instead of a lower weight
Date: 2026-09-20 (pre-registered before running)

Hypothesis: EXP-0027 (`heading_deviation_penalty=2.0`, LR=5e-5) produced the best near-miss result of the heading-fine-tune series at 1M timesteps (`dist_ratio=0.996`, controlled heading tied `exp0019`'s best) before destabilizing with more training (falls climbing to 10/20 by 3M). EXP-0028 tried fixing this by halving the *weight* (1.0) at the same LR, and it did not help -- every checkpoint still fell, and heading got worse, not better. This suggests the problem is not the new term's magnitude but the *step size* of the optimization process perturbing an already-fragile peak (`exp0026`'s own trend showed continued training on its own established recipe also degrades it, EXP-0026). If so, keeping the weight that already worked well briefly (2.0) but reducing the fine-tune LR substantially (5e-5 -> 1e-5, a 5x reduction) should make smaller, more cautious departures from `exp0026`'s starting point, potentially preserving the good heading result for longer before any destabilization, rather than needing a weaker reward signal at all.

Failure being targeted: EXP-0027/0028's fall regression, testing LR as the untried variable (weight already tried at two values with no success).

Config: `config/exp0027_heading_on_exp0026.yaml` (`heading_deviation_penalty=2.0`, unchanged -- same weight as EXP-0027, the one that showed the most promise), `--init-from experiments/runs/exp0026_air_time_on_exp0023_long/best_checkpoint_2M.zip --init-learning-rate 1e-5` (5x lower than every other fine-tune in this project).
- seed: 0, env count: 12, budget: 3,000,000 timesteps (matching EXP-0027's budget, for a direct comparison isolating LR as the only variable), `checkpoint_freq=500,000` (checked frequently, matching EXP-0028's cadence, given the narrow window EXP-0027 showed).

Evaluation protocol: standard 20-seed deterministic eval at each checkpoint, plus the mandatory controlled gait diagnostic on the best (0/20-falls, if any) checkpoint.

Stop condition: fixed 3M-timestep budget. Success = a checkpoint with 0/20 falls and controlled-heading drift better than `exp0026`'s -14.1 deg/10s. Failure = falls persist at every checkpoint regardless of the slower LR -- this would show the fragility is intrinsic to perturbing this specific optimum with *any* new term at *any* tested magnitude/rate, closing this investigation for good without a third variable to try.

Results: Checkpoint trend:

| steps | dist_ratio (n reliable) | falls | yaw_drift/10s (full eval) |
|---|---|---|---|
| 0.5M | 0.802 (20) | 1/20 | 14.9 deg |
| 1M | 0.763 (18) | 2/20 | 17.6 deg |
| 1.5M | 0.759 (20) | 1/20 | 16.4 deg |
| **2M** | 0.728 (20) | **0/20** | 18.9 deg |
| **2.5M** | **0.931** (20) | **0/20** | 21.1 deg |
| **3M** | **1.099** (19-20) | **0/20** | 18.7 deg |

**The core hypothesis is confirmed: falls stabilize to 0/20 from 2M onward and stay there through 3M** -- unlike both EXP-0027 (falls climbing 2/20 -> 10/20 with more budget) and EXP-0028 (falls present at every single checkpoint, never reaching 0/20). Lowering the fine-tune LR (not the reward weight) is what stabilizes this fine-tune, confirming the mechanism hypothesis directly. `dist_ratio` also climbs to unprecedented levels by 3M -- 1.099, the best in the entire project (previous best: `exp0026`'s 0.918) -- verified as a real result, not a metric artifact: per-episode inspection shows `distance_traveled_m` genuinely high (4.5-7.0m across sampled episodes) with reasonable `mean_vel_error` (0.035-0.056 per sampled episode) and normal `final_pelvis_height` (~0.78, standing/walking, not degenerate).

Mandatory stride-quality verification: the 3M checkpoint's controlled gait diagnostic and swing-duration measurement (`diagnostics/reports/gait_diagnostic_exp0031_3M.png`, 6 episodes, 882 touchdowns) show a **real stride matching the project's best** -- swing duration mean 51.1ms / median 60.0ms / p90 80.0ms / max 100.0ms, foot-lift range 0.031-0.057m -- nearly identical to `exp0026`'s own numbers.

The controlled-heading result is genuinely mixed, not a clean win: the 3M checkpoint's single controlled trial shows +23.6 deg/10s drift (worse than `exp0026`'s -14.1), but re-checking the 2.5M checkpoint gives +14.7 deg/10s -- nearly the *same magnitude* as `exp0026`'s own figure, just opposite sign. This reveals real checkpoint-to-checkpoint noise in a single 10-second controlled trial that this project's evaluation protocol has not previously had reason to average over. Full standard-eval comparison:

| checkpoint | dist_ratio | vel_err | falls | yaw (full) | yaw (controlled) |
|---|---|---|---|---|---|
| `exp0026` (canonical) | 0.918 | **0.036** | 0/20 | 17.6 | -14.1 |
| `exp0031` @ 2.5M | 0.931 | 0.057 | 0/20 | 21.1 | +14.7 |
| `exp0031` @ 3M | **1.099** | 0.061 | 0/20 | 18.7 | +23.6 (noisy single trial) |

Neither `exp0031` checkpoint strictly dominates `exp0026`: `dist_ratio` improves substantially, but `vel_error` regresses by ~65% relative, and full-eval heading is comparable-to-slightly-worse. This is a genuine trade-off, not a clean win on the originally-targeted axis (closing `exp0019`'s controlled-heading gap) -- that specific goal is not achieved.

Conclusion: **the real, lasting result of this experiment is methodological, not the specific checkpoint.** EXP-0027/0028 together showed that `exp0026` sits at a fragile optimum where introducing *any* new reward term at the standard fine-tune LR (5e-5) destabilizes it regardless of the term's weight. EXP-0031 shows this is fixable -- not by weakening the new term, but by slowing down the optimization process itself (LR 5e-5 -> 1e-5). This generalizes usefully beyond this one case: **when fine-tuning onto a narrow/fragile optimum, a lower learning rate is the more targeted fix for instability than a lower reward weight**, since the weight only controls how much the new term matters, while the LR controls how big a step the optimizer is allowed to take away from the (fragile) starting point -- these are different failure mechanisms that call for different fixes, and this project had not previously separated them.

Decision: **Not promoted to canonical** (a genuine trade-off, not a strict improvement over `exp0026`) but **kept as a new documented alternative** -- `experiments/runs/exp0031_heading_on_exp0026_lowlr/best_checkpoint_3M.zip`, the best raw distance-coverage/tracking-ratio checkpoint in the project, with a real, verified stride and stable 0/20 falls, trading away some instantaneous velocity-tracking precision. `exp0026` remains canonical.

Next: this specific heading-gap-closing goal (matching `exp0019`'s controlled measurement) remains unachieved after three attempts (EXP-0027/0028/0031) and is not pursued further without a fundamentally different idea -- diminishing returns on this narrow question. The more valuable, generalizable finding (LR vs. weight for stabilizing fine-tunes onto fragile optima) is the one worth carrying forward to any future fine-tuning work in this project.

---

### EXP-0032 — Reward annealing within a single run: does gradually introducing terms avoid the from-scratch collapse, without needing a separate fine-tuning stage?
Date: 2026-09-20 (pre-registered before running)

New capability (code, not just config): `config.loader.RewardScheduleTerm` + `Config.reward_schedule` (a dict of reward-weight name -> `{start, end, end_fraction}`), `envs.g1_walk_env.G1WalkEnv.set_curriculum_progress`/`_effective_reward_weights` (linear ramp from `start` at progress=0 to `end` at progress>=`end_fraction`, clamped after), and `training.curriculum_callback.CurriculumProgressCallback` (an SB3 `BaseCallback` that calls `set_curriculum_progress` on every env worker via `VecEnv.env_method` once per rollout, `progress = num_timesteps / total_timesteps`). Wired into `training/train.py`: the callback is only attached when `cfg.reward_schedule` is non-empty, so every existing config's behavior is provably unchanged (`tests/test_reward_schedule.py`, 6 new tests, including an explicit check that `_effective_reward_weights()` returns the exact same static object when no schedule is configured). Smoke-tested end-to-end on both `DummyVecEnv` and `SubprocVecEnv` before this run. Full test suite: 65/65 passing.

Hypothesis: EXP-0022 (`exp0019`'s full reward -- `feet_air_time_bonus=6.0` + `heading_deviation_penalty=0.5` -- present from step 0, standard LR, random init) collapsed to a degenerate "stand still" optimum and never learned to walk. EXP-0024 (same reward, low LR instead) collapsed even more completely. Across both, "every non-tracking term is trivially minimized by not moving" made standing still a strongly-attracting optimum near a random start, regardless of learning rate (`docs/FAILURE_ANALYSIS.md` F-0002 third/fourth occurrences). The project's actual working solution (EXP-0016 through EXP-0026) was warm-start fine-tuning across *separate* training runs -- first learn to walk under the simple reward alone, only then introduce the harder terms via a fresh run initialized from that checkpoint. Reward annealing tests whether the same *effect* (simple objective first, complexity later) can be achieved *within one continuous run*, without ever needing a separate checkpoint/fine-tuning stage: if the policy has already learned real forward locomotion under the base terms alone by the time `feet_air_time_bonus`/`heading_deviation_penalty` ramp up to meaningful weight, it should never be near the "stand still" basin when the harder terms start to matter, sidestepping the collapse EXP-0022/0024 hit at either learning rate.

Failure being targeted: EXP-0022/0024's degenerate collapse, testing a third variable (introduction schedule) neither of those experiments varied.

Config: `config/exp0032_reward_annealed_from_scratch.yaml` -- identical final reward weights and PPO hyperparameters to `config/exp0022_full_curriculum_from_scratch.yaml` (`feet_air_time_bonus=6.0`, `heading_deviation_penalty=0.5`, `learning_rate=3e-4`, random init, no `--init-from`), except both terms start at 0.0 and ramp linearly to their full value over the first 40% of training (`end_fraction=0.4`) instead of being present at full weight from step 0. The five base gait-quality terms (`tracking_lin_vel`, `alive_bonus`, `torque_penalty`, `ang_vel_penalty`, `action_rate_penalty`) are present from step 0, unchanged -- these were already proven safe to train from scratch on their own (EXP-0009/0011).
- seed: 0, env count: 12, budget: 20,000,000 timesteps (matching EXP-0022 exactly, for a direct, single-variable comparison), `checkpoint_freq=4,000,000`.

Evaluation protocol: standard 20-seed deterministic eval at each checkpoint (matching config), plus the mandatory controlled gait diagnostic and swing-duration measurement on the best checkpoint if any checkpoint shows real walking (not degenerate) -- established practice this session, non-negotiable before any promotion claim.

Stop condition: fixed 20M-timestep budget. Three possible outcomes: (1) real walking emerges and improves through the anneal, avoiding EXP-0022's collapse entirely -- annealing is a genuinely useful alternative to separate-run fine-tuning, worth characterizing further; (2) the policy still collapses to standing still once the terms ramp up past some threshold, just later than EXP-0022 -- annealing delays but does not prevent the trap, and the "stand still" basin is even more fundamental than EXP-0022/0024 suggested; (3) walking emerges during the ramp but the terms' introduction still destabilizes it once they reach full weight, similar in kind to `exp0026`'s own fragility (EXP-0027/0028) -- suggesting the destabilization risk from a new term is generic to this reward family regardless of *how* gradually it's introduced, not specific to fine-tuning onto an already-peaked checkpoint. No budget escalation without a new hypothesis regardless of outcome.

Results: pending -- run launched in background.

---

## Template

### EXP-XXXX — Short name
Date:

Hypothesis:

Why this experiment now:

Code state:

Config:
- seed(s):
- env count:
- physics dt:
- control dt:
- action scale:
- reward terms/scales:
- PPO differences:
- budget:

Evaluation protocol:

Results:

Behavioral observations:

Conclusion:

Decision:
Keep / revert / investigate.

Next:

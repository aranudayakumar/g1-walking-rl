# Independent Audit — Locomotion New-Member Challenge

Audit date: 2026-09-18. Performed by an independent pass that re-read the challenge
document, re-read source (not just docs), re-ran the test suite, recomputed
claimed statistics from raw JSON, and independently re-executed one deterministic
evaluation episode against the frozen checkpoint to check for fabrication.

Overall method: nothing below was taken on the project's own docs' word alone —
every claim in this file was checked against code, a live model, an actual test
run, or a raw data recomputation.

---

## Findings

### F1 — Checkpoint reload and deterministic-eval numbers are genuinely reproducible (not fabricated)
**Checked:** Independently re-ran seed 1002 through `experiments/runs/baseline0001/final_model.zip`
via `PPO.load` + `G1WalkEnv`, outside of `evaluation/evaluate.py`, and compared to the
seed-1002 entry already recorded in `experiments/runs/baseline0001/eval_summary.json`.

**Found:** Bit-for-bit match: `episode_steps=1000`, `total_reward=1189.3657441268122`,
`distance_traveled_m=3.130177114743428`, `termination_reason=None` — all match the
stored JSON to full float precision. Also recomputed every aggregate in
`eval_summary.json` (mean length, mean reward, mean distance, mean distance ratio)
directly from its own `episodes` array for both `baseline0001` and `exp0006_target_kl`;
all match the stored `summary` block exactly.

**Severity:** N/A (positive finding). This is strong evidence the checkpoint, environment,
and reported numbers are real and self-consistent, not invented after the fact.

---

### F2 — Model/joint indexing, PD control, observation, reward, termination all match their doc claims and are unit-tested
**Checked:** `control/joint_map.py` against a live load of `assets/g1/g1_walk.xml` (joint
types, qpos/qvel addresses, actuator biastype); `control/pd_controller.py`'s
`action_to_target`/`pd_torque` math; `envs/observations.py`'s 44-dim layout;
`envs/reset.py` / `envs/termination.py`; `rewards/walking_reward.py`'s 3-term reward;
`envs/g1_walk_env.py`'s control-decimation loop and physics-dt assertion.

**Found:** All match their prose descriptions in `docs/SIMULATION.md` / `docs/RL_PIPELINE.md`
exactly. `mujoco.MjModel` inspection confirms joint ids 1–12 are exactly `CANONICAL_ORDER`
in that order, all hinge joints, qpos[7:19]/qvel[6:18], `nu=29` (12 leg motors + 17 upper-body
position actuators), matching the claimed layout. `check_termination` really does check
`numerical_invalid` before `fell_height` before `fell_tilt`, and this ordering is explicitly
unit-tested (`test_numerical_check_takes_priority_over_height`). The action-hold loop in
`G1WalkEnv.step` really does recompute PD torque from fresh `q`/`qd` every physics sub-step
while holding the same `target_q` for `control_decimation=10` steps, matching the claimed
"held for 0.02s, recomputed each sub-step" behavior. F-0001 (the `actuator_gaintype` vs
`actuator_biastype` bug) is real and correctly diagnosed: both `<position>` and `<motor>`
actuators do report `gaintype=FIXED`, and only `biastype` distinguishes them — verified
directly against both `g1_walk.xml` and `g1.xml`.

**Severity:** N/A (positive finding).

---

### F3 — 27/27 tests pass, and are substantive rather than vacuous
**Checked:** `PYTHONPATH=. python -m pytest tests/ -q` from a clean shell with the
project's own `.venv`.

**Found:** `27 passed, 2 warnings` (only expected `check_env` warnings about unbounded
observation space, which is intentional given no natural bounds exist for these
quantities). No skips, no xfails hiding anything. Read all 5 test files: they assert
concrete numeric behavior (e.g. `pd_torque` sign/clipping/saturation, reward-component
non-domination, termination-priority ordering, joint-map rejection of the wrong model),
not `assert True`-style padding.

**Severity:** N/A (positive finding).

---

### F4 — MEDIUM: F-0002's "clean, evidence-backed pattern" for high-speed falls is overstated
**Checked:** Recomputed commanded-speed magnitude (`sqrt(vx²+vy²)`) for all 20 eval
episodes and split by termination reason, directly from `eval_summary.json`.

**Found:** Fall speeds: `[0.133, 0.179, 0.415, 0.506, 0.566]`. Survivor speeds range
`[0.059 .. 0.357]` (median ≈0.19). Three of the five falls (0.415/0.506/0.566) are
clearly above the entire survivor range and support the claimed pattern. But two falls
(seed 1005 at 0.179 m/s, seed 1007 at 0.133 m/s) sit *inside* the bulk of the survivor
distribution — well below its median and below several surviving episodes' commanded
speeds — yet those two episodes fell almost immediately (155 and 248 steps, i.e. 3.1s
and 5.0s, vs. 1000-step survival for comparable-speed survivors). `docs/FAILURE_ANALYSIS.md`
F-0002 describes this as "a clean, evidence-backed pattern, not noise" and frames the
failure mode purely as "concentrated at the top of the sampled speed range." That framing
is true for 3/5 falls but glosses over 2/5 falls that contradict a pure speed-threshold
story and are left unexplained (some other seed-specific instability, e.g. a bad initial
command/lateral-component combination, is a more likely explanation than "high speed").

**Severity:** MEDIUM. Not a code bug — it's an interpretive overstatement in the failure
analysis. It doesn't invalidate the overall conclusion (baseline0001 does show real
learning and a real speed-dependent trend), but a reader taking F-0002 at face value would
believe the correlation is cleaner and more mechanistically understood than the raw data
supports.

---

### F5 — LOW/MEDIUM: `command.resample_seconds` is a dead config field implying unimplemented behavior
**Checked:** `grep -rn "resample_seconds"` across all Python source.

**Found:** `resample_seconds` is declared in `CommandConfig` (`config/loader.py`) and
present in both `config/default.yaml` and `config/exp0006_target_kl.yaml` with a comment
("resample once per episode (episode == 20s)"), but it is never read anywhere outside
`config/loader.py` itself. `envs/reset.sample_command` is only ever called from
`G1WalkEnv.reset()`; there is no mid-episode resampling logic in `G1WalkEnv.step()` at all.
Because `resample_seconds == episode_seconds == 20.0` in every config actually used, this
currently produces no observable discrepancy (the command is sampled once per 20s episode
either way) — but the field's name and comment imply a general "resample every N seconds"
capability that does not exist in code. If someone later set `resample_seconds` to
something other than `episode_seconds` expecting mid-episode command changes, nothing
would happen.

**Severity:** LOW-MEDIUM. Vestigial/misleading config surface — not a functional bug today,
but a real "claims a capability the code doesn't have" gap of the kind the audit was asked
to look for.

---

### F6 — MEDIUM: termination-angle documentation is internally inconsistent, and one of the numbers is arithmetically wrong
**Checked:** Independently recomputed, from the actual `projected_gravity` rotation math in
`envs/observations.py`, what tilt angle (degrees from upright) corresponds to the
termination threshold `min_projected_gravity_z = -0.3` used in `config/default.yaml` /
`envs/termination.py`.

**Found:** The threshold `grav_z > -0.3` triggers at a tilt angle of **≈72.5°** from
upright (`-cos(72.5°) ≈ -0.3`), not the three different values documents give for it:
- `docs/RL_PIPELINE.md` / `docs/DECISIONS.md` ADR-002 say "~70 deg" — correct, close enough.
- `config/default.yaml`'s own inline comment says **"~107 deg from upright"** — this is
  wrong by about 35 degrees (107° would correspond to `grav_z ≈ +0.29`, a very different,
  much more tipped-over threshold than what the code actually enforces).
- `envs/termination.py`'s docstring says "z near 0 or positive means tipped >=90 degrees" —
  vague and, taken literally, also inconsistent with the actual ~72.5° trigger point (the
  episode terminates well before `grav_z` reaches 0/90°).

**Severity:** MEDIUM. Purely a documentation/comment-accuracy problem — the executable
termination check itself is simple, correct, and unit-tested (`tests/test_termination.py`)
against the literal `-0.3` numeric threshold, so this cannot cause a runtime bug. But it
means nobody actually verified the degree-conversion arithmetic before writing it down in
three places with three different answers, which is exactly the kind of unverified,
uncross-checked claim CLAUDE.md's "know units/why it exists" principle is meant to prevent.

---

### F7 — MEDIUM: `PROJECT_PLAN.md` is stale relative to `PROJECT_STATE.md` / `docs/DELIVERABLES.md` about which phase the project is in
**Checked:** Cross-read `PROJECT_PLAN.md`, `PROJECT_STATE.md`, `docs/DECISIONS.md`,
`docs/DELIVERABLES.md`, `docs/FAILURE_ANALYSIS.md` for consistency about run/phase status,
per the audit's explicit ask.

**Found:** `PROJECT_PLAN.md`'s progress line says: *"Phase 7 in progress (EXP-0006 ...) ...
Phase 9 (deliverables) drafted in docs/DELIVERABLES.md, **pending EXP-0006's outcome** for
the final writeup."* But `docs/DELIVERABLES.md`, `docs/FAILURE_ANALYSIS.md` F-0002, and
`docs/DECISIONS.md` ADR-003 all already contain EXP-0006's finished outcome (the "stands
nearly still" negative result) and describe `baseline0001` as the confirmed canonical
baseline. `PROJECT_STATE.md` separately states "Phases 0-7 complete." These three files
disagree about whether Phase 7 (and therefore the deliverables writeup) is finished or
still pending on EXP-0006 — `PROJECT_PLAN.md` was evidently not updated after EXP-0006
concluded and after `DELIVERABLES.md` was finalized around it.

**Severity:** MEDIUM (doc-only). Doesn't affect the shipped pipeline or its results, but
it's a real contradiction about project status of exactly the kind the audit was asked to
check for, and would confuse a new reader of this repo about what's actually done.

---

### F8 — LOW: `PROJECT_STATE.md`'s own footer is stale relative to its own body
**Checked:** Same file, header vs. footer.

**Found:** The file's own "Current phase" section (top) says "Phases 0-7 complete" and its
"Latest experiment" section documents EXP-0006 as "complete, documented negative result."
But the file's final line says **"Last updated: 2026-09-18, mid-EXP-0005."** EXP-0005 is
the run that produced `baseline0001` itself, i.e. the file's own timestamp claims it was
last touched a full experiment iteration earlier than the content it now contains.

**Severity:** LOW. Cosmetic — a stale trailing timestamp inside an otherwise-current file,
not a substantive contradiction, but sloppy for a file whose explicit purpose is being "the
handoff file for every Claude session."

---

### F9 — LOW: challenge-doc coverage is otherwise genuinely complete
**Checked:** Re-read `../Locomotion New Member Document.md` Part 2 line by line against the
implementation.

**Found:** Every explicit requirement is met and traceable to real code/artifacts:
G1 + MuJoCo renders/steps (`scripts/smoke_test_model.py`, confirmed live); observation/action
space defined explicitly and documented field-by-field; reward starts minimal
(velocity tracking + survival + torque penalty) and is not padded with untested extra terms;
rollout collection uses a real vectorized env (`SubprocVecEnv`, 12 workers, confirmed
9729 fps in EXP-0004 / 7253 steps/s in the real baseline run); PPO update is entirely
SB3-owned (no hand-rolled GAE/clipping found anywhere in the repo); training and
deterministic "play" evaluation are cleanly separated (`training/train.py` vs.
`evaluation/evaluate.py`, disjoint seed ranges 0–11 vs. 1000–1019, `deterministic=True`
used only in eval/video); reward/episode-length curves exist and are a real, non-trivial
1200×900 PNG generated from the actual `progress.csv`; a real 440KB MP4 exists and was
generated by `evaluation/record_video.py` from the same frozen checkpoint (offscreen
MuJoCo renderer, not a screen recording or placeholder). Domain randomization (challenge's
"once basic walking works") is correctly *not yet done* and honestly labeled as deferred,
which is consistent with the challenge document explicitly treating that as a later step,
not a requirement for this deliverable.

**Severity:** N/A (positive finding, included for completeness of the audit).

---

### F10 — LOW: reproduction instructions are plausible but not fully hands-off
**Checked:** Whether the exact commands in `docs/DELIVERABLES.md` "Reproduction instructions"
would actually run as written on a fresh checkout.

**Found:** All referenced scripts/modules exist (`scripts/smoke_test_model.py`,
`scripts/smoke_test_pd_hold.py`, `scripts/smoke_test_env.py`, `training.train`,
`evaluation.evaluate`, `results.plot_curves`, `evaluation.record_video`), their CLI
argument names match their actual `argparse` definitions, and `python3.12` is genuinely
available on this machine at `/opt/homebrew/bin/python3.12`, so `python3.12 -m venv .venv`
would work as written. One caveat not mentioned in the doc: the *exact* baseline numbers
(825.7-step mean, etc.) depend on CPU-only SB3/PyTorch determinism across machines/versions,
which is not guaranteed identical on a different machine even with the same seed — the doc
doesn't caveat that "reproduction" here means "same pipeline, same protocol," not
"bit-identical numbers on any machine."

**Severity:** LOW. Reasonable expectation for a CPU RL pipeline; just worth naming
explicitly since the doc doesn't.

---

## No BLOCKER-severity findings

Nothing found rises to BLOCKER: there is no fabricated result, no test that's silently
vacuous, no claimed artifact that doesn't exist, no checkpoint that fails to reload, and no
documented number that fails to recompute from its own raw data.

## Overall verdict

This project genuinely meets the new-member locomotion challenge's bar. The pipeline is
real and end-to-end (MuJoCo → PD control → Gymnasium env → SB3 PPO → checkpoint →
deterministic evaluation), every stage is independently unit-tested (27/27 passing, not
vacuous), and the headline claims were independently re-derived rather than taken on faith:
re-running the frozen checkpoint against a fresh environment instance reproduced the
stored per-episode reward and distance-traveled numbers bit-for-bit, and every aggregate
in both `eval_summary.json` files recomputes correctly from its own raw per-episode data.
The project's single most important piece of intellectual honesty — catching that
`exp0006_target_kl`'s "improvement" (better reward, zero falls) was actually a degenerate
near-motionless policy once distance-traveled was checked, and correctly keeping the
noisier-but-actually-walking `baseline0001` as canonical instead — is real, well-reasoned,
and exactly the kind of failure analysis the challenge and CLAUDE.md ask for. The
weaknesses found are all in the documentation layer, not the implementation: one
interpretive overstatement in a failure-analysis writeup (F4), one dead/vestigial config
field (F5), three mutually-inconsistent and one arithmetically-wrong description of a
termination angle (F6), and stale/contradictory status text between planning docs (F7, F8).
None of these change whether the pipeline works, is understood, or is honestly documented
at the level the challenge asks for — they are polish items, not correctness or integrity
problems.

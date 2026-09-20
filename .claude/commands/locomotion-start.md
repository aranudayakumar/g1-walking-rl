Read `CLAUDE.md`, the locomotion challenge document, `PROJECT_PLAN.md`, and `PROJECT_STATE.md`.

Bootstrap this challenge now.

1. Extract the exact required goal/deliverables from the challenge and update `PROJECT_STATE.md`.
2. Audit this machine and repository: OS, architecture, Python, package managers, CPU/GPU/CUDA availability, memory if easy to determine, current files, git status, existing code, and existing model assets.
3. Use `locomotion-researcher` subagents in parallel for genuinely independent questions, especially current Unitree/MuJoCo framework behavior and any unclear challenge-linked repository assumptions.
4. Compare implementation routes and record the chosen architecture plus fallback in `docs/DECISIONS.md`.
5. Rewrite `PROJECT_PLAN.md` if research changes the sensible phases.
6. Implement the smallest vertical slice immediately after planning: ideally model discovery/load and a verified physics smoke test.
7. Use the `sim-environment-engineer` where it speeds up isolated simulator work.
8. Run the smallest test proving the slice works.
9. Update `docs/SIMULATION.md`, `PROJECT_STATE.md`, and any research/decision logs.
10. Continue to the next small slice if the first succeeds and no real blocker remains.

Do not stop after producing a plan.
Do not start a long RL training run during bootstrap.

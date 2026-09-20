Perform an independent audit of this project against the locomotion challenge.

Do not assume existing claims are correct.

Audit:
- challenge requirement coverage;
- model/control indexing;
- physics dt vs control dt;
- action semantics;
- observation correctness and normalization;
- reset/termination behavior;
- reward implementation and component scales;
- PPO interface/config;
- rollout/update execution;
- train/eval separation;
- checkpoint reload;
- deterministic evaluation;
- experiment reproducibility;
- whether claimed improvements are supported by comparable metrics;
- deliverable readiness.

Use specialized subagents for independent reviews when helpful.

Run small verification commands/tests where safe.
Write findings to `docs/AUDIT.md` with severity:
BLOCKER / HIGH / MEDIUM / LOW.

Fix clear BLOCKER/HIGH issues that can be repaired safely within the current task, then rerun relevant tests and update the audit.
Update `PROJECT_STATE.md` with remaining blockers.

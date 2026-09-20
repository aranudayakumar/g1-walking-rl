Read `CLAUDE.md`, `PROJECT_STATE.md`, `PROJECT_PLAN.md`, the latest experiment entries, and relevant subsystem docs.

Identify the single highest-value next step toward an end-to-end, explainable challenge completion.

Then execute it rather than merely suggesting it.

Requirements:
- use a subagent only if the task is independent/parallel or context-heavy;
- run a targeted test;
- log the result;
- update `PROJECT_STATE.md`;
- revise the plan/decision docs if evidence changed;
- if the step naturally exposes a clear immediate next smoke test, perform that too.

Avoid long training unless the pipeline is already validated and the run has a written hypothesis, budget, metrics, and stop condition.

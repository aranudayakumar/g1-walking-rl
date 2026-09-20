Turn the user's experiment idea (`$ARGUMENTS`) into one bounded, reproducible locomotion experiment.

Read the canonical baseline and latest failure analysis first.

Before changing/running anything:
- state the hypothesis;
- identify the failure being targeted;
- define the exact changed variable(s);
- define the baseline;
- define evaluation metrics;
- define the training/run budget and stop condition;
- create an experiment record from `templates/EXPERIMENT_TEMPLATE.md`.

Use `experiment-runner` for execution if the run is self-contained.
Use `failure-analyst` afterward when interpretation is nontrivial.

Run deterministic evaluation using the same protocol as baseline.
Log the result in `experiments/EXPERIMENT_LOG.md`.
Keep/revert/investigate based on evidence.
Update `PROJECT_STATE.md`.

Do not change multiple conceptual factors unless the experiment specifically requires an interaction test.

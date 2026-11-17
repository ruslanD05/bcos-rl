# CarRacing experiments

This package contains the discrete-action `CarRacing-v3` pipelines used as a
support environment in the thesis. The final experiments compare standard PPO
with a B-cos actor and evaluate boundary-focused attribution supervision.

## Shared setup

- observations are `96 x 96` RGB frames;
- the archived RQ1 comparison and RQ3 support experiment use RGB3 actors;
- standalone B-cos training also supports optional internal
  `[RGB, 1 - RGB]` complement encoding;
- actions are no-op, steer right, steer left, accelerate, and brake;
- standard and B-cos variants share the PPO configuration and critic design;
- explanation supervision applies only to the B-cos actor.

## Entry points

```text
train_carracing_ppo.py                    Standalone PPO training
sweep_rq1_carracing_architecture.py       RQ1 matched architecture sweep
sweep_rq1_new_arch_tuning.py              Recorded architecture tuning sweep
sweep_rq3_carracing_boundary_support.py   RQ3 boundary-support sweep
generate_rq3_carracing_visuals.py         RQ3 explanation figures
visualize_carracing.py                    Episode and attribution viewer
explanation_masks.py                      CarRacing mask construction
```

Inspect all command-line options before launching a run:

```bash
uv run python car_racing_part/train_carracing_ppo.py --help
uv run python car_racing_part/sweep_rq1_carracing_architecture.py --help
uv run python car_racing_part/sweep_rq3_carracing_boundary_support.py --help
```

The RQ3 Condor launch file and recorded visualization data are under
`experiments/rq3_carracing_support/`. Model binaries and run logs are stored in
Weights & Biases or local run directories and are not part of the review
branch.

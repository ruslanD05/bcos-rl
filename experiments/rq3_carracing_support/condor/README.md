# CarRacing RQ3 support run

This W\&B grid sweep trains five B-cos PPO seeds on discrete `CarRacing-v3` for one
million transitions. It uses the RQ1 CarRacing PPO configuration and adds only
explanation supervision. Its coefficient rises linearly from 0 to 0.15 over the
one million training transitions.

The mask is active from the first frame after reset:

- positive: car, road boundaries, and the full bottom dashboard;
- negative: grass and road surface away from the boundaries;
- neutral: curbs and uncertain pixels.

Create the sweep ID on a machine with W\&B authentication:

```bash
uv run python car_racing_part/sweep_rq3_carracing_boundary_support.py --entity ruslandontsov2005-saarland-informatics-campus --create-sweep
```

Then submit from the repository root:

```bash
condor_submit experiments/rq3_carracing_support/condor/rq3_carracing_boundary_support.sub SWEEP_ID=<sweep-id>
```

Each Condor process trains one seed (`0`--`4`) and writes its checkpoint, evaluation
files, TensorBoard logs, and configuration under `experiments/rq3_carracing_support/runs/`.

At the end of each run, W\&B receives one `training-run` artifact containing the full
run directory recursively: the final model, all periodic checkpoints, the best model,
evaluations, TensorBoard logs, configuration, and manifest.

# RQ1 Final Training Sweeps

This package trains every final RQ1 model from scratch. All MiniGrid runs use
RGB6 observations, `DummyVecEnv`, the original PPO/DQN objectives, and zero
auxiliary coefficients. DoorKey runs use intermediate rewards of `0.05` for
key pickup and `0.1` for opening the door; these events do not exist in
LavaCrossing.

## Experiment Matrix

1. `sweep_rq1_ppo_architecture.py`: 10 runs.
   - standard PPO and independently validated B-cos B=2.3, 5 seeds each.
2. `sweep_rq1_ppo_b_power.py`: 40 runs.
   - B = 1.5, 2.0, ..., 5.0, 5 seeds each.
3. `sweep_rq1_dqn_architecture.py`: 10 runs.
   - standard and B-cos DQN, 5 matched seeds each.
4. `sweep_rq1_minigrid_support.py`: 10 runs.
   - standard and selected-B B-cos PPO on LavaCrossing S9N1, 5 seeds each.
5. `sweep_rq1_carracing_architecture.py`: 10 runs.
   - standard and selected-B B-cos PPO on discrete CarRacing, 5 seeds.

The PPO architecture sweep uses the `stellar` B=2.3 configuration selected by
an independent 3-profile, 5-seed validation. It achieved mean final success
0.998 and at least 0.99 success on every seed. The requested B-power grid uses
half-step values and is trained independently; no models are reused.

All shaped MiniGrid final-training runs use a budget of `750,000` environment
steps.

## Output Layout

Each run writes to:

```text
experiments/rq1_architecture/<experiment>/<environment>/<variant>/seed_<seed>/<run_id>/
  config.json
  manifest.json
  final_model.zip
  final_replay_buffer.pkl  # DQN only; kept local, not uploaded to W&B
  checkpoints/
  best_model/
  eval/
  tensorboard/
```

`manifest.json` contains the model SHA-256, Git revision, parameter counts,
training wall time, throughput, and the final stochastic evaluation. The final
model and manifest are also uploaded as a W&B artifact.

## Launch Order

On the cluster login node, export the W&B key before submitting. The submit
files pass only this variable into each Docker job; the key is not stored in
the repository:

```bash
export WANDB_API_KEY='your-current-wandb-api-key'
mkdir -p experiments/rq1_architecture/condor/logs
```

Submit all jobs from the same shell. Confirm the variable is present with
`test -n "$WANDB_API_KEY" && echo "W&B key is set"`.

Create all five W&B sweeps locally with:

```bash
bash experiments/rq1_architecture/create_all_sweeps.sh 2.3
```

The IDs are saved to `created_sweep_ids.txt`. The argument is the B value used
by the two support sweeps.

### 1. DoorKey PPO architecture

```bash
condor_submit experiments/rq1_architecture/condor/01_ppo_architecture.sub
```

### 2. DoorKey PPO B-power curve

```bash
condor_submit experiments/rq1_architecture/condor/02_ppo_b_power.sub
```

Treat the B-power curve as a sensitivity analysis. The primary DoorKey
architecture comparison remains standard versus independently validated
B=2.3, regardless of the curve's result.

### 3. DoorKey DQN

The default B=2.3 reflects the existing DQN pilot results. Change it explicitly
if the final design chooses another value.

```bash
condor_submit experiments/rq1_architecture/condor/03_dqn_architecture.sub
```

### 4. LavaCrossing support environment

```bash
condor_submit experiments/rq1_architecture/condor/04_minigrid_support.sub
```

### 5. CarRacing support environment

```bash
condor_submit experiments/rq1_architecture/condor/05_carracing_architecture.sub
```

The five submit files use the tested Docker image and scratch-backed `uv`
environment. Each Condor job consumes exactly one W&B run. No sweep uses
subprocess vector environments.

## Tests Reusing These Models

No additional training is required for:

- conventional versus validated B=2.3 B-cos final policy quality;
- the PPO B-power curve;
- time to 80%, 90%, and 95% success from periodic evaluations;
- stability and variation across seeds;
- parameter-count, throughput, and wall-clock comparisons;
- RQ2 post-hoc versus intrinsic explanation comparisons on fixed datasets;
- explanation runtime and qualitative heatmaps;
- perturbation faithfulness tests using the selected final PPO/DQN models.

For RQ2, use the same fixed states and stored actions for every explanation
method. Do not retrain copies merely to generate saliency, Integrated Gradients,
or intrinsic B-cos explanations.

## Thesis Figures and Tables

The verified W&B export is archived under `statistical_export/v0`. Generate
the final RQ1 figures and compact tables with:

```powershell
python experiments/rq1_architecture/generate_rq1_thesis_outputs.py
```

Outputs are written to `thesis_outputs/` as PDF and PNG figures and as CSV and
LaTeX tables. The DQN figure is explicitly labeled as using a B-cos-calibrated
configuration; replace it only if the independent 1M-step DQN confirmation is
adopted as the final result.

# Learning Inherently Interpretable RL Policies Using B-cos Networks

Code and experiment records for Ruslan Dontsov's bachelor's
thesis at Saarland University. The project studies B-cos policies trained with
PPO and DQN, evaluates their inherent explanations, and tests
action-conditioned attribution supervision.

## Repository map

```text
bcos_rl_generic/      Core B-cos PPO/DQN policies and supervision losses
minigrid_part/        MiniGrid training, evaluation, and figure pipelines
car_racing_part/      CarRacing training, evaluation, and figure pipelines
experiments/          RQ-specific launch files and machine-readable results
scripts/              Cross-experiment audits and artifact utilities
tools/                Result-registry and thesis-figure utilities
tests/                Unit and integration tests
```

The experiment directories follow the thesis research questions:

```text
experiments/
├── rq1_architecture/             Architecture and B-power comparisons
├── rq2_explanations/             Explanation quality and faithfulness
├── rq3_explanation_control/      Action-conditioned attribution guidance
├── rq3_carracing_support/        CarRacing support experiment
├── rq3_legacy_energy_ablation/   Earlier loss retained for the RQ3 comparison
├── rq4_supervision_timing/       Supervision curriculum experiments
├── rq5_wall_colour/              Wall-colour robustness evaluation
├── rq6_shortcut_suppression/     Controlled shortcut experiment
└── audits/bcos_completeness/     Numerical B-cos decomposition audit
```

Each experiment directory contains its own launch notes and recorded outputs.
Large checkpoint archives are stored externally and are intentionally not
included in this review branch. Run identifiers and available artifact hashes
are kept in the experiment-level configuration and manifest files.

## Setup

Python 3.10--3.12 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
uv sync --group dev
```

Run the automated checks with:

```bash
uv run pytest
uv run python verify_submission_repo.py
```

## Main entry points

The final training sweeps are named `sweep_rq*.py` in the two environment
packages. For example:

```bash
uv run python minigrid_part/sweep_rq1_ppo_architecture.py --help
uv run python minigrid_part/sweep_rq3_ppo_continuations.py --help
uv run python car_racing_part/sweep_rq3_carracing_boundary_support.py --help
```

Evaluation and aggregation scripts use descriptive `evaluate_*`,
`aggregate_*`, and `generate_*` names. Exact commands for cluster runs are
archived beside the corresponding results under `experiments/`.

## Reproducibility note

Recorded CSV, JSON, and NPZ outputs are retained where they support the thesis
tables, figures, or statistical analyses. Generated plot copies, logs,
presentation material, model binaries, and superseded exploratory pipelines
are excluded from this branch. Weights & Biases run identifiers remain in the
recorded configuration and manifest files.

## License

The source code is released under the [MIT License](LICENSE).

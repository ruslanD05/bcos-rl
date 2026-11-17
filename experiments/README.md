# Experiment archive

This directory is organized by the research questions in the thesis. It keeps
launch configurations and machine-readable outputs needed to trace reported
results. Large model checkpoints, W&B caches, logs, and duplicate generated
figures are excluded from the review branch.

| Directory | Contents |
| --- | --- |
| `rq1_architecture/` | PPO/DQN architecture comparisons, B-power sensitivity, and support environments |
| `rq2_explanations/` | Shared evaluation states, method-level measurements, and final aggregates |
| `rq3_explanation_control/` | Final channelwise action-conditioned supervision experiment |
| `rq3_legacy_energy_ablation/` | Earlier loss formulation used in the old-versus-final comparison |
| `rq3_carracing_support/` | CarRacing boundary-support configuration and recorded visualization data |
| `rq4_supervision_timing/` | Curriculum launch files, evaluations, and aggregates |
| `rq5_wall_colour/` | Wall-colour distribution-shift evaluations |
| `rq6_shortcut_suppression/` | Shortcut-correlation evaluations and final aggregates |
| `audits/bcos_completeness/` | Numerical reconstruction audit for B-cos models |

Experiment launchers live in `minigrid_part/` and `car_racing_part/`. Paths in
older recorded JSON and CSV files describe the location used when those files
were created; current executable defaults use the directory names above.

# Unified MiniGrid PPO reruns

The RQ4 and RQ6 reruns use the frozen `rq1-rq3-stellar` PPO profile from
`minigrid_part/ppo_experiment_profiles.py`. The profile is the stable RQ1
architecture-comparison configuration and the source configuration for RQ3.

| Study | W&B project | Sweep ID | Grid | Transition budget |
|---|---|---|---:|---:|
| RQ4 | `thesis-unified-rq4-curriculum-v2` | `mh5ches2` | 4 schedules x 5 seeds = 20 runs | 750,000 per run |
| RQ6 | `thesis-unified-rq6-shortcut-v1` | `v0jkk8mv` | 6 correlations x 5 seeds = 30 paired runs | 150,000 per model |

The Condor submission files are:

- `experiments/rq4_supervision_timing/condor/rq4_unified_curriculum.sub`
- `experiments/rq6_shortcut_suppression/condor/rq6_unified_correlation.sub`

No Condor jobs have been submitted.

The earlier empty RQ4 sweep (`thesis-unified-rq4-curriculum-v1`, `y5crfs09`)
used an unintended one-million-transition budget and must not be used.

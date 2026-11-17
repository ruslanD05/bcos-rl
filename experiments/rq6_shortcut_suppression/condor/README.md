# Unified RQ6 shortcut-correlation rerun

This reruns the six shortcut correlations (0.50 to 1.00) over five seeds. Each
job trains a matched Baseline/Suppressed pair for 150,000 transitions, producing
30 paired jobs and 60 models. All PPO hyperparameters use the frozen RQ1/RQ3
`stellar` profile; the 6x6 shortcut environment and correlation are the only
task-specific differences.

After the W&B sweep ID has been inserted into `rq6_unified_correlation.sub`,
submit the 30 jobs on Condor:

```bash
mkdir -p /home/rudo00002/bachelor_thesis_RL/bachelors_thesis_ruslan/experiments/rq6_shortcut_suppression/condor/logs
condor_submit /home/rudo00002/bachelor_thesis_RL/bachelors_thesis_ruslan/experiments/rq6_shortcut_suppression/condor/rq6_unified_correlation.sub
```

Do not combine these results with the earlier RQ6 sweeps.

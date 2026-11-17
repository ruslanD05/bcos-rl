# Unified RQ4 curriculum rerun

This reruns all four schedules (Pure, Linear, Success-Gated, Constant) over five
seeds. Every run uses 750,000 transitions and the frozen RQ1/RQ3 `stellar`
PPO profile. The only experimental difference is the supervision schedule.

After the W&B sweep ID has been inserted into `rq4_unified_curriculum.sub`, submit
the 20 jobs on Condor:

```bash
mkdir -p /home/rudo00002/bachelor_thesis_RL/bachelors_thesis_ruslan/experiments/rq4_supervision_timing/condor/logs
condor_submit /home/rudo00002/bachelor_thesis_RL/bachelors_thesis_ruslan/experiments/rq4_supervision_timing/condor/rq4_unified_curriculum.sub
```

Do not combine these results with the older RQ4 v4 runs.

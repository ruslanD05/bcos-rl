# RQ3 seed-0 recovery

The original channelwise-v3 sweep completed all seeds except seed 0. These four
jobs reproduce the original pure, sequential, local, and flipped seed-0 configs.

First ensure the restored LFS checkpoint is present on the cluster:

```bash
git pull origin main
git lfs pull --include="experiments/rq1_architecture/minigrid_ppo_architecture/minigrid_doorkey_8x8/bcos_b2.3/seed_0/jf3kqwlr/final_model.zip"
test -f experiments/rq1_architecture/minigrid_ppo_architecture/minigrid_doorkey_8x8/bcos_b2.3/seed_0/jf3kqwlr/final_model.zip
```

Create a four-run recovery sweep:

```bash
uv run python minigrid_part/sweep_rq3_ppo_continuations.py \
  --create-sweep \
  --project rq3-minigrid-ppo-channelwise-control-v3 \
  --experiment rq3_ppo_channelwise_continuations_v3 \
  --output-root experiments/rq3_explanation_control/continuations \
  --channelwise-absolute-energy \
  --additional-timesteps 300000 \
  --branches pure,sequential,local,flipped \
  --seeds 0 \
  --include-pure
```

Replace `REPLACE_WITH_RECOVERY_SWEEP_ID` in `rq3_seed0_recovery.sub`, then run:

```bash
mkdir -p experiments/rq3_explanation_control/condor/logs
condor_submit experiments/rq3_explanation_control/condor/rq3_seed0_recovery.sub
```

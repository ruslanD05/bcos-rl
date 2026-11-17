# B-cos normalization and completeness audit

This audit checks whether the intrinsic input-times-gradient explanation
reconstructs the selected action-relative output in B-cos explanation mode:

\[
\left|\sum_i x_i\frac{\partial f_{\mathrm{rel}}(x)}{\partial x_i}
- f_{\mathrm{rel}}(x)\right|.
\]

## Audited checkpoints

The local audit covered 101 thesis-relevant checkpoints:

| Group | Checkpoints | Largest median residual | Largest maximum residual |
|---|---:|---:|---:|
| RQ1 B-cos PPO | 5 | 9.54e-7 | 4.29e-6 |
| RQ2 qualitative B-cos DQN | 1 | 6.94e-8 | 3.02e-7 |
| RQ3 B-cos PPO | 15 | 1.91e-6 | 8.58e-6 |
| RQ4 B-cos PPO | 20 | 1.55e-6 | 9.54e-6 |
| RQ6 B-cos PPO | 60 | 2.86e-6 | 2.86e-6 |

Each checkpoint was evaluated on 128 genuine environment observations, for
12,928 reconstruction checks in total. DoorKey RGB6 checkpoints used the saved
RQ3 shared-state evaluation dataset. RQ6 checkpoints used a fresh rollout from
the correlated-cue DoorKey-6x6 evaluation environment.

All 101 audits completed successfully. Every explanation-path normalization
layer was bias-free, and the largest difference between ordinary-mode and
explanation-mode forward values was exactly zero. The overall largest
reconstruction residual was 9.54e-6, consistent with float32 numerical error.

All audited PPO actors contained three `GroupNormUncentered2d` and three
`RMSNormUncentered1d` modules. The audited DQN online Q-network contained three
`GroupNormUncentered2d` and one `RMSNormUncentered1d` module. All checkpoints
used `stable_bcos_norm=True`.

## Scope limitation

This is a complete audit of the thesis-relevant checkpoints currently available
in the local canonical experiment caches selected by the script. The clean
submission registry references additional RQ1/RQ2/support-task checkpoints that
are not present at their registered local paths. Auditing those exact archives
would require restoring them from the external W&B artifact store.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python \
  scripts/audit_bcos_completeness.py --samples 128 --batch-size 32
```

Per-checkpoint module inventories and residuals are stored in `report.json`.

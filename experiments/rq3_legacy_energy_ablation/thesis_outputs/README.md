# RQ3 Thesis Outputs

Generate the table and figures with:

```powershell
python experiments/rq3_legacy_energy_ablation/generate_rq3_thesis_outputs.py
```

Add `--fetch-wandb` to refresh model artifact hashes and explanation-gradient
diagnostics without downloading model ZIPs.

`rq3_run_configs_and_hashes.json` archives the exact W&B configuration and
source/final model hashes for all 20 matched continuation runs.

`rq3_localization_and_faithfulness` contains the primary aggregate evidence.
The heatmap evaluates every branch under the same three masks. The bar chart
reports deletion and insertion advantage over equal-area random perturbations.

Each state has two explanation layouts. The `*_faded_overlay` version renders
the observation at 30% opacity beneath a strengthened attribution overlay. The
`*_side_by_side` version places standalone absolute attribution magnitude in
red to the right of the observation for every branch. Each map is normalized
independently to its 99th percentile. P/N/D targets are provided
separately in the corresponding `rq3_masks_*` figures.

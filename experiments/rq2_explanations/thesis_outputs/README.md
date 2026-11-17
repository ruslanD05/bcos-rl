# RQ2 Thesis Outputs

Generate all outputs from the stored RQ2 evaluation and local checkpoints:

```powershell
python experiments/rq2_explanations/generate_rq2_thesis_outputs.py --device cpu
```

## Quantitative table

`rq2_explanation_methods_table.tex` compares intrinsic B-cos, Saliency,
Integrated Gradients, and SmoothGrad. The main controlled explanation-method
comparison is within the B-cos rows because those methods explain the same
policies. Standard-PPO rows are architecture controls and should not be read as
paired explanation-method comparisons against different B-cos policies.
The adapted Sensitivity-\(n\) summary computes a Pearson correlation separately
for each prespecified subset fraction from 10% to 90% and then averages the nine
fixed-size correlations. It does not pool state--subset pairs across fractions.

## MiniGrid figures

The forward, pickup, and toggle figures use the same stored RGB6 observation
and action for standard PPO and B-cos PPO. Intrinsic B-cos is unavailable for
the conventional network. Integrated Gradients uses 32 Gauss-Legendre steps;
SmoothGrad uses eight samples with noise standard deviation `0.02 * 255`.

## CarRacing figure

`rq2_carracing_methods` uses the pure B-cos checkpoint
`experiments/carracing_explanation_mask_comparison/models/pure_bcos_ppo_1269760.zip`.
All methods explain the same selected action on each frame from one
deterministic episode.

All qualitative maps use signed channel summation. Red supports and blue
suppresses the selected relative action logit. Each panel is independently
scaled to its 99th absolute-attribution percentile, so color intensity must not
be compared quantitatively across methods; use the table for that comparison.

# RQ2 Final Statistical Report

All attribution methods explain the same stored action on the same 256 RGB6 states. B-cos statistics use three validated PPO configurations and five matched training seeds per configuration.

## B-cos Model Results

| Method | Deletion advantage | Insertion advantage | Mean fixed-size Sensitivity-n (PCC) | Infidelity | Completeness error | Seconds/sample |
|---|---:|---:|---:|---:|---:|---:|
| Intrinsic B-cos | 0.221 | 0.187 | 0.422 | 0.453 | 1.62e-07 | 0.000434 |
| Integrated Gradients | 0.223 | 0.172 | 0.350 | 1.733 | 0.658 | 0.014209 |
| Saliency | 0.188 | 0.174 | 0.054 | 1.000 | 1 | 0.000613 |
| SmoothGrad | 0.187 | 0.174 | 0.048 | 1.000 | 1 | 0.004439 |

## Interpretation

Intrinsic B-cos attribution is approximately tied with Integrated Gradients on random-normalised deletion, is stronger on insertion and mean fixed-size Sensitivity-n, has exact output completeness, and is much faster. The evidence therefore supports an overall faithfulness-and-efficiency advantage, not universal superiority on every individual metric.

Error bars in `final_summary.png` are 95% cluster-bootstrap confidence intervals over the five training seeds, retaining all configuration runs within each sampled seed. Complete machine-readable estimates and paired differences are in `final_report.json`.

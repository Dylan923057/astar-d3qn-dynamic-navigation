# Dynamic prediction cross-map mechanism analysis

## Scope

This analysis uses the frozen train and validation scenario definitions plus existing validation curves. It does not use the test split and does not rerun training.

## Environment controls

| Map | Obstacles | Causal obstacles | Conflict progress | Risk positions / 100 steps | Speed | Route length |
|---|---:|---:|---:|---:|---:|---:|
| irregular_workcell_91701 | 4.00 | 1.50 | 0.322 | 2.273 | 1.00 | 4.30 |
| irregular_workcell_91702 | 4.00 | 1.50 | 0.334 | 2.273 | 1.00 | 4.33 |
| irregular_workcell_91703 | 4.00 | 1.50 | 0.324 | 2.273 | 1.00 | 4.41 |

Obstacle-count distribution, causal-obstacle count, speed, designed conflict frequency, and early/middle/late allocation are matched by construction across the three maps. Therefore the existing evidence cannot support the claim that map03 is simply a lower-complexity dynamic environment.

## Prediction and navigation

| Map | Prediction F1 | Decision-zone F1 | B-A AUC | Improved seeds | A threshold | B threshold |
|---|---:|---:|---:|---:|---:|---:|
| irregular_workcell_91701 | 0.309 | 0.457 | +0.0382 | 2/3 | 96667 | 76667 |
| irregular_workcell_91702 | 0.443 | 0.601 | +0.0319 | 3/3 | 33333 | 36667 |
| irregular_workcell_91703 | 0.406 | 0.527 | -0.0306 | 1/3 | 43333 | 60000 |

Map-level prediction-F1/effect correlation is -0.332; with only three maps this is descriptive, not inferential.
Across nine paired seeds, prediction-F1/effect correlation is 0.241.
Map03 prediction F1 and decision-zone F1 are both higher than map01, yet its navigation AUC effect is negative. This rejects a simple 'prediction head failed on map03' explanation.
The evidence is most consistent with the prediction task being learned while its representation benefit is not reliably converted into improved Q-policy learning under every map geometry and conflict structure.

## Paper interpretation

- Do not claim universal cross-map improvement or that the method is proven to work only in more complex dynamic scenes.
- Report positive learning-efficiency evidence on map01 and map02, with map03 as a formal boundary/negative case.
- Frame the contribution as a conditional auxiliary-learning effect and explicitly discuss representation-to-control transfer limitations.
- The current three maps do not vary obstacle count or speed independently, so complexity causality remains untested.

## Next action

Write the method, mechanism analysis, and limitation sections now. If one additional experiment is later required, pre-register a small dynamic-complexity sweep on a newly frozen validation set; do not tune prediction parameters on the current validation results.

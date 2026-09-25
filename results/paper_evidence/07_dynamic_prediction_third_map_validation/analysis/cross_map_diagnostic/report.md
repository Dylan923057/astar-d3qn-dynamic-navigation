# Dynamic prediction auxiliary cross-map diagnostic

This diagnostic reads validation curves only. It does not open test files.

| Map | A mean AUC | B mean AUC | B-A | Improved seeds | A threshold | B threshold |
|---|---:|---:|---:|---:|---:|---:|
| irregular_workcell_91701 | 0.7042 | 0.7424 | +0.0382 | 2/3 | 96667 | 76667 |
| irregular_workcell_91702 | 0.9271 | 0.9590 | +0.0319 | 3/3 | 33333 | 36667 |
| irregular_workcell_91703 | 0.8674 | 0.8368 | -0.0306 | 1/3 | 43333 | 60000 |

- Positive maps: 2/3.
- Positive paired seeds: 6/9.
- Equal-map mean AUC effect: +0.0132.
- Map 3 is not a terminal-safety collapse: both methods finish with safe success 1.0.
- On map 3, B reaches the validation threshold later for all seeds and has larger late-stage variation.
- Therefore the auxiliary prediction benefit is map-dependent and does not support a robust three-map generalization claim.
- Do not unlock test data, tune prediction parameters, or expand seeds based on this result.

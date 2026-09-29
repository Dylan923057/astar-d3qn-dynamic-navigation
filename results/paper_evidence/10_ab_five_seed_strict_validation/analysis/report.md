# Strict three-map A/B five-seed validation

## Protocol

All 30 adaptation runs use the same frozen code/config/manifest provenance and paired per-seed foundations. Only validation artifacts were read; test artifacts were forbidden.

## Per-map effects

- irregular_workcell_91701: A=0.6758±0.1435, B=0.7417±0.0564, Δ=+0.0658, 95% bootstrap CI=[-0.0838, +0.2154], B>A=2/5.
- irregular_workcell_91702: A=0.8879±0.0633, B=0.8825±0.0505, Δ=-0.0054, 95% bootstrap CI=[-0.0908, +0.0612], B>A=3/5.
- irregular_workcell_91703: A=0.8996±0.0617, B=0.8671±0.0552, Δ=-0.0325, 95% bootstrap CI=[-0.0558, -0.0092], B>A=0/5.

## Direct answers

1. map01 positive-gain assessment: positive_direction_but_uncertain.
   map02 positive-gain assessment: negative_direction_but_uncertain.
2. map03 negative-gain assessment: stable_negative.
3. Cross-map assessment: scenario_dependent.

## Cross-map equal-weight result

- A equal-weight mean AUC: 0.8211
- B equal-weight mean AUC: 0.8304
- Equal-weight paired effect: +0.0093
- Hierarchical 95% bootstrap CI: [-0.0471, +0.0694]
- Classification: scenario_dependent

Map-level effects are weighted equally. Seeds from different maps are not pooled as one sample. Exact paired sign-flip tests are supplementary; effect sizes and bootstrap intervals remain primary.

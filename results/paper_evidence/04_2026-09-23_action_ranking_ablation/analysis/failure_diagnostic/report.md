# Time-decay final-model validation failure diagnosis

This report is evaluation-only. It loads final weights, uses the frozen validation split, disables exploration, performs no updates, and never reads the test split.

## Consistency

- Re-evaluated scenarios: 72
- Exact matches to saved final validation rows: 72/72

## Outcomes

| Seed | Condition | Success | Dynamic collision | Static collision | Timeout |
|---:|---|---:|---:|---:|---:|
| 0 | control | 12/12 | 0 | 0 | 0 |
| 0 | conflict | 12/12 | 0 | 0 | 0 |
| 1 | control | 12/12 | 0 | 0 | 0 |
| 1 | conflict | 7/12 | 5 | 0 | 0 |
| 2 | control | 12/12 | 0 | 0 | 0 |
| 2 | conflict | 12/12 | 0 | 0 | 0 |

## Failure localization

- Total failures: 5
- Dynamic collisions: 5
- Timeouts: 0
- Collision actions already flagged risky before execution: 5/5
- Colliding obstacle visible in current local frame: 5/5
- Colliding obstacle represented in the three-frame local history: 5/5
- Most frequent collision cells: [34, 35] (5)
- Timeout patterns: none

## Cross-seed critical-step comparison

All five failures occur in seed 1 at step 63. Seed 0 or seed 2 succeeds on every identical frozen scenario, so these cases are not intrinsically unsolvable.

| Scenario | Seed | Outcome | Position before step 63 | Action |
|---|---:|---|---|---|
| pair_018_n5_000_conflict | 0 | success | [33, 35] | right |
| pair_018_n5_000_conflict | 1 | dynamic_collision | [33, 35] | down |
| pair_018_n5_000_conflict | 2 | success | [39, 29] | right |
| pair_019_n5_001_conflict | 0 | success | [33, 35] | right |
| pair_019_n5_001_conflict | 1 | dynamic_collision | [33, 35] | down |
| pair_019_n5_001_conflict | 2 | success | [39, 27] | right |
| pair_021_n5_003_conflict | 0 | success | [29, 33] | right |
| pair_021_n5_003_conflict | 1 | dynamic_collision | [33, 35] | down |
| pair_021_n5_003_conflict | 2 | success | [39, 29] | right |
| pair_022_n3_004_conflict | 0 | success | [28, 34] | right |
| pair_022_n3_004_conflict | 1 | dynamic_collision | [33, 35] | down |
| pair_022_n3_004_conflict | 2 | success | [39, 29] | right |
| pair_022_n5_004_conflict | 0 | success | [33, 35] | right |
| pair_022_n5_004_conflict | 1 | dynamic_collision | [33, 35] | down |
| pair_022_n5_004_conflict | 2 | success | [39, 27] | right |

## Evidence-guided next target

The failed state already contains three consistent dynamic frames, and the chosen action is flagged as an immediate collision risk. Seed 0 avoids the same critical state by moving right in several matched scenarios, while seed 2 takes an earlier detour. The next single modification should therefore target unstable safe-versus-risky action ranking from existing temporal observations, not add more obstacle channels, maps, rewards, or risk-sampling ratios.

## Representative figures

- `figures/seed_1__pair_022_n3_004_conflict.png`: seed 1, pair_022_n3_004_conflict, dynamic_collision.
- `figures/seed_1__pair_018_n5_000_conflict.png`: seed 1, pair_018_n5_000_conflict, dynamic_collision.
- `figures/seed_1__pair_019_n5_001_conflict.png`: seed 1, pair_019_n5_001_conflict, dynamic_collision.
- `figures/seed_1__pair_021_n5_003_conflict.png`: seed 1, pair_021_n5_003_conflict, dynamic_collision.
- `figures/seed_1__pair_022_n5_004_conflict.png`: seed 1, pair_022_n5_004_conflict, dynamic_collision.

## Scope

The diagnosis describes only the fixed-budget final checkpoints. It must not be used to infer all earlier learning failures or to select a test result.

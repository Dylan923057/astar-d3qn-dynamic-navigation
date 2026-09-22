# Migration Notes

The new project keeps only the reusable algorithmic core from `D:\Astar-DRL`.

## Reused and corrected

- Four-connected A* structure from `core/astar.py`, rewritten with stale-entry
  checks and a randomized optimal tie-break variant.
- Grid conversion helpers from `core/grid.py`, with planner moves separated
  from the five-action execution space.
- Dueling CNN and Double-DQN update ideas from `agents/`, refactored so the
  agent no longer owns a specific replay implementation.
- Uniform replay persistence, YAML loading, CSV/JSON I/O, transition budgets,
  and path metrics.
- Deterministic random-map construction and selected path-progress helpers.

## Deliberately excluded

- Keypoint curricula and keypoint rewards.
- Cross-map generalization protocols and logical V3 generators.
- BFS behavior cloning, prioritized replay, map-balanced replay, and PER.
- Historical checkpoints, results, diagnostics, and formal-report scripts.

## Newly implemented

- Persistent A* demonstration replay with a registered batch fraction.
- Standardized five-action execution semantics including stay.
- A 20x20 fixed-count random benchmark with deterministic manifests.
- A* demonstration transition collection using the same environment reward as
  online D3QN training.


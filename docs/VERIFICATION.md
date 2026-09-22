# Verification Record

Date: 2026-08-07

- Python compilation completed for `src/`, `scripts/`, and `tests/`.
- 33 unit tests passed.
- Standard A* matched BFS shortest-path lengths on 30 seeded random grids.
- Randomized A* tie-breaking retained optimal path lengths.
- Dueling aggregation and Double-DQN selection/evaluation were tested directly.
- Terminal transitions were verified not to bootstrap.
- Uniform, one-time prefill, and persistent-demo pipelines all completed gradient
  updates in short smoke runs.
- Five fixed 20x20 training maps were generated with distinct SHA-256 grid hashes.
- Twenty A* demonstration episodes produced 720 environment transitions.
- Local observations use a padded 11x11 obstacle window plus two normalized
  signed goal-displacement scalars. Demonstrations use format version 2.
- Training progress callbacks report rolling metrics and greedy map evaluation
  at the configured interval without changing replay or network parameters.

Smoke-run success rates are not research results; smoke runs exist solely to
validate the execution paths.

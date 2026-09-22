# Research Protocol 0.1

Status: implementation baseline, not a formal result.

## Research question

Does A* demonstration replay improve the learning efficiency and frozen-policy
success rate of Dueling Double DQN on reproducible grid-navigation tasks?

## Initial controlled comparison

1. Uniform online replay only.
2. One-time A* replay prefill followed by ordinary online replay.
3. A persistent A* demonstration partition sampled throughout training.

All three methods must share maps, observations, actions, rewards, network,
optimizer, transition budget, random seeds, and evaluation episodes.

## Random benchmark

- 20x20 square occupancy grids.
- Fixed-count random static obstacles and fixed endpoints.
- Every accepted map is A* reachable and passes registered path-length and
  turn-count gates.
- The five fixed training maps are shared by all three replay strategies. No
  held-out map or unseen-map generalization claim is part of this comparison.
- Five execution actions: up, down, left, right, stay.
- The initial benchmark uses an agent-centered 11x11 local static-grid
  observation. The spatial input is the local obstacle occupancy; normalized
  signed row and column displacement to the goal are auxiliary scalar inputs.
  This phase validates replay integration and is not presented as a complete
  dynamic-obstacle environment.
- Twenty A* demonstration episodes are collected with randomized optimal
  tie-breaking and the same environment reward function used by D3QN.
- Prefill and persistent-demo training load the exact same saved demonstration
  dataset; only replay retention and sampling behavior differ.

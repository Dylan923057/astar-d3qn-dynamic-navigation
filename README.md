# Astar-D3QN-Workshop

Current independent protocol: [持续回放与动态适应代价实验 v2（交互位置均衡）](docs/REPLAY_ADAPTATION_V2.zh-CN.md).
Irregular workcell maps, phase-paired single-obstacle scenarios, and matched
foundation checkpoints compare continued demo replay at 0%, 10%, and 25%.
Prepare with `python scripts/prepare_replay_adaptation.py`; this does not train.
The default v2 dataset assigns equal early/middle/late interaction quotas and
distinct decision positions within each split. Legacy v1 data remain separate.

Clean research code for studying A* demonstration replay with a canonical
Dueling Double DQN (D3QN). The benchmark uses reproducible 20x20 random
occupancy grids and an agent-centered 11x11 local observation with a normalized
goal-direction vector. A fixed workshop environment will be added after the
replay ablation is validated.

The initial comparison is deliberately limited to:

1. D3QN with uniform online replay.
2. D3QN with one-time A* replay prefill.
3. D3QN with a persistent A* demonstration partition.

Optional comparison baselines are also available through the same training
entry point: proportional prioritized replay (`--strategy per`) and a
controlled DQfD-style demonstration margin baseline (`--strategy dqfd`). See
`docs/replay_baselines.md` for the exact scope and commands.

## Algorithm contracts

- A* expands only four movement neighbors and uses Manhattan distance. Its
  randomized variant changes only tie-breaking, so shortest-path optimality is
  preserved.
- The execution action space has five actions: up, down, left, right, and stay.
- D3QN uses the dueling aggregation `Q = V + A - mean(A)`.
- Double DQN selects the next action with the policy network and evaluates it
  with the target network.
- The local observation contains one spatial obstacle channel and two normalized
  goal-displacement scalars. The CNN encodes the local map and the scalars are
  concatenated before the dueling value and advantage heads.
- Demonstrations are environment transitions, not coordinate lists.

## Quick start

```powershell
python scripts/generate_random_benchmark.py --config configs/random_benchmark.yaml
python scripts/collect_astar_demos.py --config configs/random_benchmark.yaml
python -m unittest discover -s tests -q
python scripts/train_random_benchmark.py --config configs/random_benchmark.yaml --smoke
```

Generated experiment artifacts belong under `outputs/`. The five fixed training
maps are stored together in `maps/random_benchmark/train/maps.json` and are
reproducible from their registered seeds. Prefill and persistent-demo training
both load the same registered A* dataset from `data/demonstrations/`.
Training prints a rolling progress line every 100 episodes by default. Each
line includes recent reward, success and collision counts, followed by a
greedy evaluation over the five training maps. Set
`training.progress_interval` to `0` to disable these reports.

## Multi-obstacle risk handover

The independent candidate method in [多动态障碍风险覆盖示范交接实验 v1](docs/RISK_HANDOVER_V1.zh-CN.md)
keeps the frozen adaptation maps but composes paired 3/5-obstacle training
scenes and a 1/3/5/7-obstacle test sweep.  Its three-part replay progressively
replaces the fixed A* demonstration slots with short pre-conflict online
sequences as distinct risk positions are covered.  Generate and audit the
dataset with `python scripts/prepare_risk_handover.py`; this command never
starts training.

## References

- Hart, Nilsson, and Raphael (1968), A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths.
- van Hasselt, Guez, and Silver (2016), Deep Reinforcement Learning with Double
  Q-learning.
- Wang et al. (2016), Dueling Network Architectures for Deep Reinforcement
  Learning.
- Schaul et al. (2016), Prioritized Experience Replay.
- Hester et al. (2018), Deep Q-learning from Demonstrations.

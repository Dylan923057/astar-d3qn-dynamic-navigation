# Replay Baselines

The training entry point supports the original controlled replay strategies and
two optional comparison baselines:

```powershell
python scripts/train_random_benchmark.py --config configs/structured_calibration_40x40.yaml --map-index 0 --strategy per --seed 0
python scripts/train_random_benchmark.py --config configs/structured_calibration_40x40.yaml --map-index 0 --strategy dqfd --seed 0
```

`per` is proportional prioritized replay. New transitions receive the current
maximum priority, samples carry normalized importance-sampling weights, and
priorities are updated from the per-transition absolute TD errors. The default
parameters are `alpha=0.6`, beta annealing from `0.4` to `1.0`, and priority
epsilon `1e-6`.

`dqfd` is a DQfD-style demonstration baseline built on the workshop's
persistent demonstration partition. It uses one-step Double-DQN TD loss and a
large-margin loss on demonstration transitions. It intentionally does not claim
to reproduce every component of Hester et al.'s DQfD implementation: n-step
returns, a separate demonstration pretraining phase, and the original full
loss schedule are not included. Use it as a controlled margin-loss baseline;
call it `DQfD-style` in experiment reports.

Both strategies use the configured total replay capacity. For `dqfd`, the
demonstration partition is included in that total, as it is for
`persistent_demo`.

# Dynamic spatial generalization v2 protocol

This protocol preserves the completed v1 runs. V2 always selects a checkpoint
using validation data only. The untouched test split is evaluated once after the
selected weights are loaded.

## Phase A: diagnostic experiments on the existing manifest

Run the three replay baselines with identical validation-checkpoint selection:

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'; foreach ($strategy in @('uniform','prefill','persistent_demo')) { foreach ($seed in 0..4) { python -u .\scripts\train_dynamic_spatial_generalization.py --config .\configs\dynamic_spatial_generalization_map01_checkpoint_v2.yaml --strategy $strategy --seed $seed; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } }
```

Run only Persistent Demo with a 25% to 0% decay from episode 200 to 600:

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'; foreach ($seed in 0..4) { python -u .\scripts\train_dynamic_spatial_generalization.py --config .\configs\dynamic_spatial_generalization_map01_demo_decay_v2.yaml --strategy persistent_demo --seed $seed; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
```

Run only Persistent Demo with a fixed 10% demonstration fraction:

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'; foreach ($seed in 0..4) { python -u .\scripts\train_dynamic_spatial_generalization.py --config .\configs\dynamic_spatial_generalization_map01_demo10_v2.yaml --strategy persistent_demo --seed $seed; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
```

Choose the Persistent Demo schedule from validation summaries only. Primary
metric: `safe_success_rate`; tie breakers: success rate, mean dynamic collision
count, and mean steps. Do not use the v1 test results for this choice.

## Phase B: one confirmatory run on Map 2

The confirmatory experiment moves to a new static map so none of its spatial
cells were used to design the v2 protocol. Its frozen manifest is
`data/dynamic_scenarios/map02_spatial_generalization_confirmatory_v2.json`.
Run Uniform and Prefill with the confirmatory base config:

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'; foreach ($strategy in @('uniform','prefill')) { foreach ($seed in 0..4) { python -u .\scripts\train_dynamic_spatial_generalization.py --config .\configs\dynamic_spatial_generalization_map02_confirmatory_v2.yaml --strategy $strategy --seed $seed; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } }
```

Then run exactly one Persistent Demo command, selected from Phase A:

- Fixed 25%: use `dynamic_spatial_generalization_map02_confirmatory_v2.yaml`.
- Decay 25% to 0%: use `dynamic_spatial_generalization_map02_confirmatory_decay_v2.yaml`.
- Fixed 10%: use `dynamic_spatial_generalization_map02_confirmatory_demo10_v2.yaml`.

All three confirmatory strategies write to the same confirmatory output root, so
do not run more than one Persistent Demo variant there.

## Diagnostic visualizations

Render a hard dynamic test scenario from the completed v1 final models:

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'; python .\scripts\render_spatial_replay_comparison.py --config .\configs\dynamic_spatial_generalization_map01.yaml --training-seed 0 --scenario-id 20023 --model final
```

Render the seed 3 static-retention failure from the completed v1 models:

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'; python .\scripts\render_static_retention_comparison.py --config .\configs\dynamic_spatial_generalization_map01.yaml --training-seed 3 --model final
```

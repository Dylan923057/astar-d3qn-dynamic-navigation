# Capability-Adaptive Handover v1 Results

Formal fixed-budget results for three maps and seeds 0--4 (15 runs total).

- Status: all 15 runs are `validation_complete`.
- Adaptation budget: 200,000 environment steps per run.
- Runtime: PyTorch `2.13.0+cu126`, CUDA.
- Test split: deferred; no test data were read or generated.
- Monitor: 12 deterministic conflict scenes selected from the train split.
- Controller: `rho_max=0.25`, `beta=0.3`, `tau=0.90`, `K=2`, `C_0=0.0`.

Each run contains `training.csv`, `episodes.csv`, `validation_final.csv`,
`result.json`, and `run_audit.json`. The frozen effective configuration is
included at the root of this directory.

Final model weights are intentionally omitted from Git because the repository
ignores `*.pth`; the original `outputs/` tree remains unchanged and retains all
15 `model_final.pth` files.

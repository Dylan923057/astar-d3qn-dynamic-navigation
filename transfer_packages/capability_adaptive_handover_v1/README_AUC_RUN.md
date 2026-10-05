# Capability-Adaptive Handover v1 AUC Extension

This add-on reruns the unchanged capability-adaptive method in the independent
`outputs/capability_adaptive_handover_v1_auc/` namespace. It adds greedy
epsilon=0 validation observations at step 0 and every 10k environment steps.
Validation is never passed to the competence controller, replay allocation,
expert exit, stopping logic, optimizer, or any other training decision. Test is
still deferred and is never read or generated.

The copied `baselines/time_decay/` directory contains the 15 matched A-branch
validation curves and result provenance needed for paired analysis. Each
adaptive run is required to match the same map, seed, and foundation snapshot.

## Files to copy to the other computer

Copy these paths into the existing `capability_adaptive_handover_v1` package,
preserving their relative paths:

- `configs/capability_adaptive_handover_v1_auc.yaml`
- `src/astar_d3qn/training/capability_adaptive_auc.py`
- `scripts/run_capability_adaptive_handover_auc.py`
- `scripts/analyze_capability_adaptive_handover_auc.py`
- `tests/test_capability_adaptive_auc.py`
- `baselines/time_decay/` and `baselines/time_decay_inventory.json`
- `README_AUC_RUN.md`

Do not recopy or replace `foundations/`, `data/`, the original source files, or
either completed output directory.

## Preflight only

```powershell
python scripts/run_capability_adaptive_handover_auc.py --stage preflight --device cuda --threads 1 --deep-foundation-check
python -m unittest discover -s tests -v
```

The preflight requires the runtime PyTorch version to match the foundation
metadata exactly and validates all 15 time-decay baseline/foundation pairings.

## Formal run

```powershell
python scripts/run_capability_adaptive_handover_auc.py --stage run --device cuda --threads 1
```

## Paired analysis after all 15 runs finish

```powershell
python scripts/analyze_capability_adaptive_handover_auc.py
```

Analysis is written once to
`outputs/capability_adaptive_handover_v1_auc/analysis/`. Existing analysis
output is never overwritten. The three-map total is computed from equal-weight
map summaries; the 15 runs are not pooled as exchangeable samples.

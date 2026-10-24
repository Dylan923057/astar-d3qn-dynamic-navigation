"""Extend the frozen consistency ablation to seeds3/4; preserve original code/results."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import run_execution_consistency as entry

ROOT=entry.ROOT
CONFIG=ROOT/'configs/whole_map_91701_execution_consistency_seeds3to4_v1.yaml'
PARENT_CONFIG=entry.CONFIG
PARENT_FREEZE=ROOT/'data/whole_map_91701_execution_consistency_v1/freeze_manifest.json'
SEEDS=(3,4)
PREFIX='whole_map_91701_execution_consistency_seeds3to4_v1'
parent_run_one=entry.run_one


def source_hashes():
    frozen=entry.common.read_json(PARENT_FREEZE)
    sources=dict(frozen['source_sha256'])
    for relative,digest in sources.items():
        if entry.common.sha256(ROOT/relative)!=digest:
            raise ValueError(f'Original frozen source changed: {relative}')
    for path in (CONFIG,Path(__file__)):
        sources[str(path.relative_to(ROOT))]=entry.common.sha256(path)
    return sources


def configure():
    original=entry.legacy.load_config(PARENT_CONFIG)
    frozen=entry.common.read_json(PARENT_FREEZE)
    if original!=frozen['config']:
        raise ValueError('Original frozen configuration changed.')
    config=entry.legacy.load_config(CONFIG)
    expected=deepcopy(original)
    expected['experiment'].update(name=PREFIX,
        protocol='one_factor_remove_executed_teacher_match_seeds3to4_v1',
        output_root='outputs/'+PREFIX,check_root='results/'+PREFIX)
    expected['execution_consistency'].update(seeds=list(SEEDS),registration_root='data/'+PREFIX)
    if config!=expected:
        raise ValueError('Only seed list and independent registration/output directories may change.')
    source_hashes()
    return config,entry.legacy.validate_config(entry.legacy.load_config(entry.legacy.CONFIG))


def run_one(seed,resume=False):
    try:
        return parent_run_one(seed,resume)
    except BaseException:
        status=ROOT/'outputs'/PREFIX/entry.METHOD/f'seed_{seed}'/'status.json'
        if status.exists():
            record=entry.common.read_json(status)
            if not record.get('complete'):
                record['resume_command']=f'python scripts/run_execution_consistency_seeds3to4.py --train --seeds {seed} --resume'
                entry.write_json(record,status)
        raise


# Reuse the unchanged learner, training loop, validation, diagnostics and recovery.
# Only this process's entry bindings change; the original frozen files stay intact.
entry.CONFIG=CONFIG
entry.SEEDS=SEEDS
entry.configure=configure
entry.source_hashes=source_hashes
entry.run_one=run_one

if __name__=='__main__':
    entry.main()

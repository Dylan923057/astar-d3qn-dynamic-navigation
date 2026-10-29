"""Independent external baselines; the completed C/D pilot is read-only."""
from __future__ import annotations

from pathlib import Path
import json

import multitask_pilot_common as pilot

ROOT=pilot.ROOT
NAME='asvc_multitask_external_v1'
CONFIG=ROOT/'configs'/f'{NAME}.yaml'
REG,OUT,RESULTS=(ROOT/p/NAME for p in ('data','outputs','results'))
RUNS=(('d3qn',0),('dqfd',0),('d3qn',1),('dqfd',1))
METHODS=('C','D','d3qn','dqfd')
read,write,immutable,sha,csv_write=pilot.read,pilot.write,pilot.immutable,pilot.sha,pilot.csv_write


def config():
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def data():
    return pilot.load_data()


def directory(method,seed):
    if method in ('C','D'):
        return pilot.run_directory(method,seed)
    if (method,seed) not in RUNS:raise ValueError('Only four registered external pilot runs are allowed')
    return OUT/method/f'seed_{seed}'


def sources():
    paths=[CONFIG,*sorted((ROOT/'scripts').glob('*multitask_external*.py')),
           ROOT/'scripts/diagnose_multitask_pairs.py',ROOT/'scripts/start_multitask_external.ps1',
           ROOT/'tests/test_multitask_external.py']
    legacy=[ROOT/'src/astar_d3qn/agents/dqfd.py',ROOT/'src/astar_d3qn/replay/dqfd.py',
            ROOT/'configs/whole_map_91701_dqfd_comparison_v1.yaml']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths+legacy}


def shared_identity():
    pilot.verify_freeze()
    old=read(pilot.REG/'frozen_protocol.json')
    return dict(pilot_freeze_sha256=sha(pilot.REG/'frozen_protocol.json'),
                inputs_sha256=old['input_sha256'],source_sha256=old['source_sha256'])


def verify():
    frozen=read(REG/'frozen_protocol.json')
    if frozen['shared_identity']!=shared_identity() or frozen['source_sha256']!=sources():
        raise ValueError('Frozen external baseline or shared pilot identity changed')
    for name,digest in frozen['artifact_sha256'].items():
        if sha(REG/name)!=digest:raise ValueError('Frozen demonstration artifact changed: '+name)
    return frozen


def model_info(method,seed):
    path=directory(method,seed)
    if not (path/'completion.json').exists():return None
    result=read(path/'result.json');model=path/'model_final.pth'
    if not result.get('complete') or result.get('step')!=500000:raise ValueError('Incomplete/wrong-budget final model')
    if sha(model)!=result['final_model_sha256']:raise ValueError('Final model digest differs')
    return dict(method=method,seed=seed,directory=path,model=model,model_sha256=result['final_model_sha256'],result=result)

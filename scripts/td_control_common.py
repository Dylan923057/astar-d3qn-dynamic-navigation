"""Independent two-seed TD-only control; upstream experiments remain read-only."""
from __future__ import annotations

from pathlib import Path
import yaml
import multitask_external_common as previous

ROOT=previous.ROOT
NAME='asvc_multitask_td_control_v1'
CONFIG=ROOT/'configs'/f'{NAME}.yaml'
REG,OUT,RESULTS=(ROOT/p/NAME for p in ('data','outputs','results'))
METHOD='d3qn_bound';METHODS=(*previous.METHODS,METHOD);SEEDS=(0,1)
read,write,immutable,sha,csv_write=previous.read,previous.write,previous.immutable,previous.sha,previous.csv_write


def config():return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def shared():return previous.data()


def directory(seed):
    if seed not in SEEDS:raise ValueError('Only seeds 0 and 1 are authorized')
    return OUT/METHOD/f'seed_{seed}'


def training_sources():
    names=['configs/asvc_multitask_td_control_v1.yaml','scripts/td_control_common.py',
        'scripts/td_control_engine.py','scripts/run_td_control.py','scripts/start_td_control.ps1','tests/test_td_control.py']
    return {name:sha(ROOT/name) for name in names}


def upstream_identity():
    previous.verify()
    return dict(shared_identity=previous.shared_identity(),external_freeze_sha256=sha(previous.REG/'frozen_protocol.json'),
                external_live_sources_sha256=previous.sources(),
                paired_groups_sha256=sha(previous.REG/'paired_reference_groups.json'),
                validation_scenes_sha256=sha(previous.pilot.REG/'validation_scenes.json'))


def verify(check_results=False):
    frozen=read(REG/'frozen_protocol.json')
    if frozen['upstream_identity']!=upstream_identity() or frozen['training_source_sha256']!=training_sources():
        raise ValueError('Frozen TD control or shared protocol changed')
    if check_results:
        for name,digest in frozen['protected_upstream_results_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Protected old run artifact changed: '+name)
    return frozen


def model_info(method,seed):
    if method!=METHOD:return previous.model_info(method,seed)
    path=directory(seed)
    if not (path/'completion.json').exists():return None
    result=read(path/'result.json');model=path/'model_final.pth'
    if not result['complete'] or result['step']!=500000 or sha(model)!=result['final_model_sha256']:
        raise ValueError('TD control final model/budget differs')
    return dict(method=method,seed=seed,directory=path,model=model,model_sha256=result['final_model_sha256'],result=result)

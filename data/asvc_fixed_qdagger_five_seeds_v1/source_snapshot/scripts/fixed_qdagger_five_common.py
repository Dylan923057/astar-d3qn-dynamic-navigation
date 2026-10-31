"""Independent five-seed registration; the frozen two-seed learner stays unchanged."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import fixed_qdagger_common as original

c=original.c;ROOT=c.ROOT
NAME='asvc_fixed_qdagger_five_seeds_v1'
DATA,OUT,RESULTS=(ROOT/folder/NAME for folder in ('data','outputs','results'))
NEW_SEEDS=(2,3,4);ALL_SEEDS=(0,1,2,3,4);REUSED_SEEDS=(0,1)
METHODS=('margin','ce','d3qn','dqfd','qdagger')
CONFIG=ROOT/f'configs/{NAME}.yaml'
METADATA={'experiment','protocol','seeds','maximum_new_training_runs','final500_evaluation','qdagger_five_extension'}
CODE=(f'configs/{NAME}.yaml','scripts/fixed_qdagger_five_common.py','scripts/run_fixed_qdagger_five.py',
    'scripts/evaluate_fixed_qdagger_five.py','scripts/analyze_fixed_qdagger_five.py',
    'scripts/start_fixed_qdagger_five.ps1','tests/test_fixed_qdagger_five.py','docs/ASVC_FIXED_QDAGGER_FIVE_SEEDS.zh-CN.md')


def configure():
    parent,shared,inputs=original.configure();expected=deepcopy(parent)
    expected.update(experiment=NAME,protocol='unchanged_fixed_qdagger_seeds234_and_frozen500_v1',
        seeds=list(NEW_SEEDS),maximum_new_training_runs=3,final500_evaluation=True,
        qdagger_five_extension=dict(pooled_seeds=list(ALL_SEEDS),reused_seeds=list(REUSED_SEEDS),new_seeds=list(NEW_SEEDS),
            original_registration='data/asvc_fixed_qdagger_v1/freeze.json',original_output='outputs/asvc_fixed_qdagger_v1',
            additional_online_environment_steps=600000,additional_offline_updates=60000,additional_online_updates=598503,
            final500_scenes='data/asvc_fixed_ce_final500_v1/frozen500.json',final500_episodes_per_seed=500,
            final500_epsilon=0,final500_teacher=False,final500_dynamic_protection=False,final500_static_mask=True,
            final500_device='cpu',final500_checkpoint='final_200000_only',final500_used_for_tuning=False,
            learning_parameter_changes='none',maximum_training_processes=1))
    config=c.legacy.load_config(CONFIG)
    if config!=expected:raise ValueError('Only seed/path/evaluation registration metadata may change')
    return config,shared,inputs


def algorithm_config(config):return {k:v for k,v in config.items() if k not in METADATA}


def make_agent(seed,device=None):
    if seed not in ALL_SEEDS:raise ValueError('Unregistered seed')
    return original.make_agent(seed,device)


load_demos=original.load_demos


def training_directory(seed):
    if seed not in ALL_SEEDS:raise ValueError('Unregistered seed')
    return (original.OUT if seed in REUSED_SEEDS else OUT)/f'seed_{seed}'


def baseline_directory(method,seed):
    if method=='qdagger':return training_directory(seed)
    if method=='ce':return ROOT/'outputs'/('asvc_fixed_ce_v1' if seed<2 else 'asvc_fixed_ce_five_seeds_v1')/f'seed_{seed}'
    return original.baseline_directory(method,seed)


def analysis_directory(method,seed):
    if method=='qdagger':return DATA/'reused_qdagger_snapshots'/f'seed_{seed}' if seed in REUSED_SEEDS else OUT/f'seed_{seed}'
    return DATA/'baseline_snapshots'/method/f'seed_{seed}'


def complete(seed):
    directory=training_directory(seed)
    if not (directory/'completion.json').exists():return False
    result=c.read(directory/'result.json');run=c.read(directory/'run_manifest.json')
    config,shared,_=configure()
    identity=original.DATA if seed in REUSED_SEEDS else DATA
    assert result['complete'] and c.read(directory/'completion.json')['complete']
    assert result['seed']==run['seed']==seed and result['environment_steps']==200000
    assert result['offline_updates']==20000 and result['online_updates']==199501 and result['total_updates']==219501
    assert result['freeze_sha256']==c.sha(identity/'freeze.json') and run['freeze_sha256']==result['freeze_sha256']
    assert result['final_model_sha256']==c.sha(directory/'model_final.pth')
    assert algorithm_config(run['config'])==algorithm_config(config) and run['shared_config']==shared
    assert result['teacher_queries']==119999 and result['teacher_exit']==120000 and result['student_replay_only']
    assert not result['teacher_advice'] and not result['td_clip'] and result['supervision_floor']==0
    return True


def source_hashes():
    result=c.read(original.DATA/'freeze.json')['source_sha256'].copy()
    result.update(c.file_map([ROOT/name for name in CODE]));return result


def verify(check_protected=True):
    original.verify(check_protected)
    frozen=c.read(DATA/'freeze.json');config,shared,_=configure()
    assert frozen['config']==config and frozen['shared_config']==shared
    assert frozen['source_sha256']==source_hashes()
    c.check_files(frozen['source_sha256']);c.check_files(frozen['input_sha256'])
    for name,value in frozen['copied_files_sha256'].items():assert c.sha(DATA/name)==value
    if check_protected:c.check_files(frozen['protected_sha256'])
    return frozen

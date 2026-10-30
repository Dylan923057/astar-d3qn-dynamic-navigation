"""Supplement seeds 3/4 without changing or rerunning the original three seeds."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import unittest

import torch

import run_persistent_teaching as parent
import supervision_tail_common as tail

ROOT, common, legacy = parent.ROOT, parent.common, parent.legacy
NAME = 'asvc_persistent_supervision_five_seeds_v1'
CONFIG = ROOT/'configs'/f'{NAME}.yaml'
PARENT_CONFIG = ROOT/'configs/asvc_persistent_teaching_v1.yaml'
PARENT_FREEZE = ROOT/'data/asvc_persistent_teaching_v1/freeze_manifest.json'
METHOD = 'supervision_floor'
NEW_SEEDS = (3, 4)
ALL_SEEDS = tuple(range(5))
OUT = ROOT/'outputs'/NAME
RESULTS = ROOT/'results'/NAME
REG = ROOT/'data'/NAME
ORIGINAL_RUN = parent.run_one


def configure():
    original = legacy.load_config(PARENT_CONFIG)
    if original != common.read_json(PARENT_FREEZE)['config']:
        raise ValueError('Original three-seed configuration changed.')
    config = legacy.load_config(CONFIG)
    expected = deepcopy(original)
    expected['experiment'].update(name=NAME, protocol='unchanged_supervision_floor_extend_seeds34_v1',
        output_root='outputs/'+NAME, check_root='results/'+NAME)
    expected['persistent_teaching'].update(seeds=list(NEW_SEEDS),registration_root='data/'+NAME,
        baseline_root='outputs/whole_map_91701_execution_consistency_seeds3to4_v1/advice_bound_margin_all_eligible')
    expected['five_seed_review'] = dict(variant=METHOD,pooled_seeds=list(ALL_SEEDS),reused_seeds=[0,1,2],
        reused_root='outputs/asvc_persistent_teaching_v1/supervision_floor',
        final500_config='configs/whole_map_91701_supervision_tail_v1.yaml',candidate_name='ASVC-D3QN-PS',
        promotion_criteria=dict(both_new_seeds_late_aulc_not_lower=True,both_new_seeds_late_minimum_not_lower=True,
            five_seed_mean_late_aulc_higher=True,five_seed_mean_late_minimum_higher=True,
            final500_maximum_mean_success_drop=.01),
        selection_note='descriptive_acceptance_criteria_not_statistical_significance')
    if config != expected: raise ValueError('Only additional seeds, metadata and frozen review criteria may change.')
    return config, legacy.validate_config(legacy.load_config(legacy.CONFIG))


def source_hashes():
    sources = dict(common.read_json(PARENT_FREEZE)['source_sha256'])
    for name,digest in sources.items():
        if common.sha256(ROOT/name) != digest: raise ValueError(f'Frozen parent source changed: {name}')
    for path in (CONFIG, Path(__file__)):
        sources[str(path.relative_to(ROOT))] = common.sha256(path)
    return sources


def new_directory(seed):
    if seed not in ALL_SEEDS: raise ValueError('Expected seed0-4.')
    return (ROOT/'outputs/asvc_persistent_teaching_v1' if seed<3 else OUT)/METHOD/f'seed_{seed}'


def baseline_directory(seed):
    name = 'whole_map_91701_execution_consistency_v1' if seed<3 else 'whole_map_91701_execution_consistency_seeds3to4_v1'
    return ROOT/'outputs'/name/'advice_bound_margin_all_eligible'/f'seed_{seed}'


def preflight():
    config,inputs = configure()
    sources = source_hashes()
    tests = unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_persistent_teaching.py')
    tested = unittest.TextTestRunner(verbosity=1).run(tests)
    if not tested.wasSuccessful(): raise ValueError('Guidance tests failed.')
    pairs, reused = [], []
    for seed in ALL_SEEDS:
        reference = baseline_directory(seed)
        manifest = common.read_json(reference/'run_manifest.json')
        for name,digest in manifest['source_sha256'].items():
            if common.sha256(ROOT/name)!=digest: raise ValueError('Historical baseline source changed.')
        if not common.read_json(reference/'completion.json')['complete']: raise ValueError('Incomplete baseline.')
        agent = parent.make_agent(config,inputs[0],seed,METHOD,'cpu')
        digest = legacy.runtime.state_digest(agent.training_state_dict())
        if digest!=manifest['initial_state_sha256']: raise ValueError('Full initial learning state mismatch.')
        old_agent = parent.make_agent(legacy.load_config(PARENT_CONFIG),inputs[0],seed,METHOD,'cpu')
        for step in (0,80000,90000,100000,150000,200000):
            agent.training_action_steps = old_agent.training_action_steps = step
            if agent.imitation_weight()!=old_agent.imitation_weight() or agent.advice_probability()!=old_agent.advice_probability():
                raise ValueError('Teaching schedule changed between seed batches.')
        record = dict(seed=seed,initial_state_sha256=digest,
            baseline_manifest_sha256=common.sha256(reference/'run_manifest.json'),
            baseline_curve_sha256=common.sha256(reference/'validation_curve.csv'))
        if seed in NEW_SEEDS: pairs.append(record)
        else:
            path = new_directory(seed)
            result = common.read_json(path/'result.json')
            if not result['complete'] or result['environment_steps']!=200000 or result['gradient_updates']!=199501:
                raise ValueError('Reused run incomplete.')
            if result['initial_state_sha256']!=digest or common.sha256(path/'model_final.pth')!=result['final_model_sha256']:
                raise ValueError('Reused model/state mismatch.')
            reused.append(dict(seed=seed,directory=str(path.relative_to(ROOT)),
                result_sha256=common.sha256(path/'result.json'),curve_sha256=common.sha256(path/'validation_curve.csv'),
                model_sha256=common.sha256(path/'model_final.pth')))
    validation,_ = legacy.runtime.check(config,inputs,list(NEW_SEEDS))
    scenes,_,scene_freeze = tail.load_frozen(legacy.load_config(tail.CONFIG))
    report = dict(passed=True,tests=tested.testsRun,initializations=pairs,reused_runs=reused,validation=validation,
        planned_new_runs=2,pooled_seeds=list(ALL_SEEDS),unchanged_supervision_floor=.1,unchanged_advice_floor=0.,
        final500_scene_sha256=scene_freeze['files_sha256']['independent_final_scenarios.json'],
        final500_scene_count=len(scenes),final500_evaluated_before_training=False,
        final500_uses_final_checkpoint_only=True,promotion_criteria=config['five_seed_review']['promotion_criteria'])
    frozen = dict(config=config,checks=report,source_sha256=sources,
        validation_sha256=common.sha256(ROOT/config['dataset']['validation_reference']))
    path = REG/'freeze_manifest.json'
    if path.exists():
        if common.read_json(path)!=frozen: raise ValueError('Extension registration changed.')
    else:
        parent.write_json(frozen,path)
        for name in sources:
            target = REG/'source_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    parent.write_json(report,RESULTS/'preflight.json')
    print('Five-seed extension preflight passed: '+str(path),flush=True)


def install():
    # Process-local bindings reuse the exact frozen training implementation.
    parent.CONFIG=CONFIG;parent.SEEDS=NEW_SEEDS;parent.VARIANTS=(METHOD,)
    parent.configure=configure;parent.source_hashes=source_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seeds',nargs='+',type=int,choices=NEW_SEEDS,default=list(NEW_SEEDS))
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):parser.error('Duplicate seeds.')
    torch.set_num_threads(1)
    if not args.train:preflight();return
    install()
    for seed in args.seeds:ORIGINAL_RUN(METHOD,seed,args.resume)


if __name__=='__main__':main()

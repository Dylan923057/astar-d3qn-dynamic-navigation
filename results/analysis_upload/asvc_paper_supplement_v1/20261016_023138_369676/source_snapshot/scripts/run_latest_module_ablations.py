"""Frozen latest-method module removals; --train executes exactly ten new runs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import unittest

import torch

import run_execution_consistency as base
import analyze_execution_consistency_five_seeds as complete
from astar_d3qn.agents.all_eligible_module_ablation import AllEligibleModuleAblationAgent,METHODS
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.utils.seed import seed_everything

ROOT,common,legacy=base.ROOT,base.common,base.legacy
NAME='latest_all_eligible_modules_v1'
CONFIGS={METHODS[0]:ROOT/'configs/latest_all_eligible_no_td_v1.yaml',
         METHODS[1]:ROOT/'configs/latest_all_eligible_no_override_v1.yaml'}
PARENT_CONFIG=ROOT/'configs/whole_map_91701_execution_consistency_v1.yaml'
PARENT_RUN_ONE=base.run_one
ACTIVE={}
SEEDS=tuple(range(5))


def configure(method):
    if method not in METHODS:raise ValueError('Only the two fixed removals are allowed.')
    original=legacy.load_config(PARENT_CONFIG);expected=deepcopy(original)
    no_td=method==METHODS[0]
    expected['experiment'].update(name=NAME,protocol='latest_all_eligible_remove_'+('td' if no_td else 'override')+'_v1',
        output_root='outputs/'+NAME,check_root='results/'+NAME)
    expected['execution_consistency'].update(method=method,seeds=list(SEEDS),registration_root='data/'+NAME+'/'+method)
    expected['module_ablation']=dict(remove='td_target_clip' if no_td else 'action_override',
        target_clip=not no_td,action_override=no_td,execution_match_required=False,compensate_label_count=False,
        supervision_exit=100000,offline_demonstrations=0,pretraining_updates=0)
    config=legacy.load_config(CONFIGS[method])
    if config!=expected:raise ValueError('Only the named removal and independent metadata may change.')
    return config,legacy.validate_config(legacy.load_config(legacy.CONFIG))


def source_hashes():
    result={}
    for seed in (0,3):result.update(complete.registered_sources(seed))
    for path in (*CONFIGS.values(),Path(__file__),ROOT/'src/astar_d3qn/agents/all_eligible_module_ablation.py',
                 ROOT/'tests/test_latest_module_ablations.py'):
        result[str(path.relative_to(ROOT))]=common.sha256(path)
    return result


def make_agent(config,problem,seed,device=None):
    seed_everything(seed);values=dict(config['agent'])
    if device:values['device']=device
    ad,repair=config['training_advice'],config['value_repair']
    a=AllEligibleModuleAblationAgent(D3QNConfig((5,15,15),4,5,seed=seed,**values),problem,
        repair_method='advice_bound_margin',ablation_method=config['execution_consistency']['method'],
        lower=repair['target_lower'],upper=repair['target_upper'],margin=repair['margin'],margin_weight=repair['margin_weight'],
        probability=ad['probability'],decay_steps=ad['decay_environment_steps'],risk_radius=ad['risk_radius'],rng_offset=ad['rng_seed_offset'])
    ACTIVE['agent']=a
    return a


def install(method):
    base.CONFIG=CONFIGS[method];base.METHOD=method;base.SEEDS=SEEDS
    base.configure=lambda:configure(method)
    base.source_hashes=source_hashes;base.make_agent=make_agent


def preflight(method):
    install(method);config,inputs=configure(method);initial=[]
    control_config=legacy.load_config(PARENT_CONFIG)
    for seed in SEEDS:
        _,freeze=complete.protocol(seed)
        reference=next(r for r in freeze['checks']['initializations'] if r['seed']==seed)
        agent=make_agent(config,inputs[0],seed,'cpu')
        digest=legacy.runtime.state_digest(agent.training_state_dict())
        if digest!=reference['initial_state_sha256']:raise ValueError('Complete initial learner/target/optimizer/RNG differs.')
        baseline=legacy.make_agent(control_config,inputs[0],seed,'advice_bound_margin','cpu')
        for t in range(200001):
            agent.training_action_steps=baseline.training_action_steps=t
            if agent.imitation_weight()!=baseline.imitation_weight():raise ValueError('Supervision clock changed.')
            expected=baseline.advice_probability(t) if method==METHODS[0] else 0.
            if agent.advice_probability(t)!=expected:raise ValueError('Override schedule changed.')
        if legacy.effective_training(config,seed,'formal')!=legacy.effective_training(control_config,seed,'formal'):
            raise ValueError('Environment budget/replay/exploration/update settings changed.')
        factories=[legacy.runtime.make_factory(c,*inputs,method='unguided',seed=seed) for c in (config,control_config)]
        for episode in range(3):
            envs=[f(inputs[0],max_steps=6,window_size=15,reward_config=legacy._reward_config(config),terminate_on_collision=True) for f in factories]
            if envs[0].dynamic_obstacles!=envs[1].dynamic_obstacles or envs[0].scenario_id!=envs[1].scenario_id:
                raise ValueError('Episode-index physical scene generation differs.')
        initial.append(dict(seed=seed,initial_state_sha256=digest,latest_complete_paired=True,original_initialization_paired=True))
    validation,_=legacy.runtime.check(config,inputs,list(SEEDS))
    import supervision_tail_common as tail
    scenes,_,scene_freeze=tail.load_frozen(legacy.load_config(tail.CONFIG))
    report=dict(passed=True,initializations=initial,validation=validation,small_recovery_check=base.recovery_check(config,inputs),
        raw_td_real_action=True,supervision_independent_teacher_action=True,execution_match_required=False,
        online_environment_steps=200000,online_updates=199501,offline_demonstrations=0,pretraining_updates=0,
        random_alignment='Same per-seed initial network/target/optimizer and learner/advice RNG; same physical scenario generation by episode index. Divergent policies need not collect identical trajectories.',
        independent500_used=False,final500_scene_sha256=scene_freeze['files_sha256']['independent_final_scenarios.json'],
        final500_scene_count=len(scenes),nonfinite_policy='Record exact failing environment step; retain seed; do not change config or restart numerical failures.')
    frozen=dict(config=config,source_sha256=source_hashes(),checks=report,
        validation_sha256=common.sha256(ROOT/config['dataset']['validation_reference']))
    path=ROOT/config['execution_consistency']['registration_root']/'freeze_manifest.json'
    if path.exists() and common.read_json(path)!=frozen:raise ValueError('Frozen module-removal protocol changed.')
    if not path.exists():
        write=base.write_json;write(frozen,path)
        for name,digest in frozen['source_sha256'].items():
            target=path.parent/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
            if common.sha256(target)!=digest:raise ValueError('Source snapshot mismatch.')
    directory=legacy.runtime.unique_check_dir(config,'check_'+method)
    base.write_json(report,directory/'check_report.json');print(directory,flush=True)
    return frozen


def run_one(method,seed,resume=False):
    install(method)
    directory=ROOT/'outputs'/NAME/method/f'seed_{seed}'
    failure=directory/'nonfinite_failure.json'
    if failure.exists():
        print(f'[{method} seed{seed}] retained numerical failure; no restart.',flush=True)
        return False
    try:
        PARENT_RUN_ONE(seed,resume)
        return True
    except BaseException as error:
        status_path=directory/'status.json'
        status=common.read_json(status_path) if status_path.exists() else dict(complete=False,running=False)
        agent=ACTIVE.get('agent')
        status.update(method=method,seed=seed,environment_steps=agent.training_action_steps if agent else None,
            gradient_updates=agent.update_steps if agent else None,error=repr(error),
            resume_command=f'python scripts/run_latest_module_ablations.py --train --methods {method} --seeds {seed} --resume')
        if isinstance(error,FloatingPointError):
            status.update(numerical_failure=True,resume_allowed=False)
            base.write_json(status,failure);base.write_json(status,status_path)
            print(f'[{method} seed{seed}] NONFINITE at step={status["environment_steps"]}; retained, no tuning/restart.',flush=True)
            return False
        if directory.exists():base.write_json(status,status_path)
        raise


def write_batch_status():
    records=[]
    for method in METHODS:
        for seed in SEEDS:
            path=ROOT/'outputs'/NAME/method/f'seed_{seed}'/'status.json'
            records.append(dict(method=method,seed=seed,**(common.read_json(path) if path.exists() else dict(complete=False,running=False,environment_steps=0))))
    base.write_json(dict(planned_runs=10,runs=records),ROOT/'outputs'/NAME/'batch_status.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS));args=parser.parse_args()
    if len(set(args.methods))!=len(args.methods) or len(set(args.seeds))!=len(args.seeds):parser.error('No duplicate runs.')
    torch.set_num_threads(1)
    if not args.train:
        suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p)
            for p in ('test_execution_consistency.py','test_latest_module_ablations.py')])
        if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():raise ValueError('Module-removal checks failed.')
        for method in args.methods:preflight(method)
        return
    for seed in args.seeds:
        for method in args.methods:
            run_one(method,seed,args.resume);write_batch_status()


if __name__=='__main__':main()

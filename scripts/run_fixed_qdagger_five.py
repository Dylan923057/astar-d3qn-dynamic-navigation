"""Extend unchanged QDagger to seeds 2/3/4, then confirm all five final models."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import shutil
import traceback
import unittest
from unittest.mock import patch
import torch
import fixed_qdagger_five_common as f
import run_fixed_qdagger as original_runner
import analyze_fixed_qdagger as original_analysis
c=f.c


@contextmanager
def installed():
    from analyze_fixed_qdagger_five import analyze
    with patch.object(original_runner,'q',f),patch.object(original_analysis,'analyze',analyze):
        yield


def run_one(seed,resume=False):
    if seed not in f.NEW_SEEDS:raise ValueError('Only new seeds 2/3/4 may train; preserve seeds 0/1')
    with installed():original_runner.run_one(seed,resume)


def short_trial(seed,device,directory,extension):
    def execute():
        exp=original_runner.Experiment(seed,directory,unit=True,device=device)
        exp.identity=dict(study='seed_extension_parity_unit',seed=seed,freeze_sha256='unit')
        demos,_=f.load_demos();exp.offline(demos,end=3);exp.begin_online()
        for _ in range(12):exp.step(budget=12,learning_starts=6,batch_size=4)
        exp.flush_bin();return original_runner.behavioral_fingerprint(exp)
    if extension:
        with installed():return execute()
    return execute()


def recovery_check(root,device,demos):
    with installed():
        def execute(path,resume):
            exp=original_runner.Experiment(2,path,unit=True,device=device);exp.offline(demos,end=3)
            if resume:
                exp.checkpoint();exp=original_runner.Experiment(2,path,unit=True,device=device,resume=True)
            exp.offline(demos,end=6);exp.begin_online()
            for _ in range(12):exp.step(budget=24,learning_starts=6,batch_size=4)
            while exp.env is None or not exp.collector.pending:
                assert exp.total<23;exp.step(budget=24,learning_starts=6,batch_size=4)
            if resume:
                exp.checkpoint();exp=original_runner.Experiment(2,path,unit=True,device=device,resume=True)
            while exp.total<24:exp.step(budget=24,learning_starts=6,batch_size=4)
            exp.flush_bin();return original_runner.behavioral_fingerprint(exp)
        assert execute(root/'plain',False)==execute(root/'resumed',True),'Extended offline/mid-episode recovery differs'
    return dict(passed=True,seed=2,device=device,offline_updates=6,online_steps=24,formal_training=False)


def safe_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert c.sha(source)==c.sha(target),'Existing registration draft differs'
    else:shutil.copyfile(source,target)


def prepare():
    if (f.DATA/'freeze.json').exists():f.verify();print('Existing five-seed QDagger freeze verified; preserved.',flush=True);return
    config,shared,inputs=f.configure();f.original.verify()
    import run_fixed_ce_final500 as ce
    ce_frozen=ce.verify();scenes,scene_sha=c.final_scenes()
    assert scene_sha==ce_frozen['scene_sha256'] and [s['scenario_id'] for s in scenes]==ce_frozen['scene_order']
    assert c.read(ce.DATA/'frozen500.json')['scenarios']==scenes
    assert not any((f.OUT/f'seed_{s}').exists() for s in f.NEW_SEEDS),'Existing unregistered training preserved'
    assert all(f.complete(s) for s in f.REUSED_SEEDS)
    attempt=f.RESULTS/'preflight_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');attempt.mkdir(parents=True,exist_ok=False)
    try:
        suite=unittest.defaultTestLoader.discover(str(f.ROOT/'tests'),pattern='test_fixed_qdagger_five.py')
        with (attempt/'unit_tests.log').open('w',encoding='utf-8') as stream:
            tested=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        assert tested.wasSuccessful(),'Five-seed QDagger preflight tests failed; see retained log'
        demos,audit=f.load_demos();recovery=[]
        for device in (('cpu','cuda') if torch.cuda.is_available() else ('cpu',)):
            for seed in f.NEW_SEEDS:
                a=f.make_agent(seed,device);b=c.legacy.runtime.make_agent(shared,seed,device)
                assert c.digest(a.training_state_dict())==c.digest(b.training_state_dict())
            recovery.append(recovery_check(attempt/device,device,demos))
            print('QDagger extension initialization and full recovery PASS on '+device,flush=True)
        baseline=[];protected_paths=[f.original.DATA,f.original.OUT,f.original.RESULTS,
            c.ROOT/'data/asvc_current_method.json',ce.DATA,ce.OUT,ce.RESULTS]
        for method in f.METHODS[:-1]:
            for seed in f.ALL_SEEDS:
                directory=f.baseline_directory(method,seed);result=c.read(directory/'result.json');run=c.read(directory/'run_manifest.json')
                assert c.read(directory/'completion.json')['complete'] and result['environment_steps']==200000
                for key in ('dataset','environment','agent','reward','dynamic_route_pool','guidance'):
                    assert run['config'][key]==shared[key],f'Existing {method} seed{seed}: shared {key} differs'
                for name in ('result.json','run_manifest.json','completion.json','validation_curve.csv','validation_details.csv','training.csv','model_final.pth'):
                    protected_paths.append(directory/name)
                baseline.append(dict(method=method,seed=seed,directory=directory.relative_to(f.ROOT).as_posix(),
                    model_sha256=c.sha(directory/'model_final.pth'),curve_sha256=c.sha(directory/'validation_curve.csv')))
        baseline500=[]
        for method in ('margin','ce'):
            for seed in f.ALL_SEEDS:
                directory=ce.OUT/f'seed_{seed}' if method=='ce' else f.ROOT/'results/asvc_persistent_supervision_five_seeds_v1/final500'/f'seed_{seed}'
                reg=c.read(directory/'registration.json');assert c.read(directory/'completion.json')['complete']
                for key,value in dict(seed=seed,model_sha256=c.sha(f.baseline_directory(method,seed)/'model_final.pth'),
                    scene_sha256=scene_sha,scene_count=500,checkpoint_steps=200000,epsilon=0,
                    teacher=False,static_mask=True,dynamic_shield=False,device='cpu').items():assert reg[key]==value
                c.scene_order(c.rows(directory/'scene_details.csv'),scenes);protected_paths.append(directory)
                baseline500.append(dict(method=method,seed=seed,directory=directory.relative_to(f.ROOT).as_posix(),
                    model_sha256=reg['model_sha256'],scene_sha256=scene_sha))
        # Save source and raw curve snapshots after every preflight invariant passes.
        f.DATA.mkdir(parents=True,exist_ok=True)
        hashes=f.source_hashes()
        for name in hashes:safe_copy(f.ROOT/name,f.DATA/'source_snapshot'/name)
        keep=('result.json','run_manifest.json','completion.json','validation_curve.csv','validation_details.csv','training.csv','update_diagnostics.csv')
        for method in f.METHODS:
            seeds=f.REUSED_SEEDS if method=='qdagger' else f.ALL_SEEDS
            for seed in seeds:
                source=f.baseline_directory(method,seed);target=f.analysis_directory(method,seed)
                for name in keep:
                    if (source/name).is_file():safe_copy(source/name,target/name)
        safe_copy(f.original.DATA/'frozen50.json',f.DATA/'frozen50.json')
        safe_copy(ce.DATA/'frozen500.json',f.DATA/'frozen500.json')
        c.write(audit,f.DATA/'demonstration_audit.json')
        planned=[dict(seed=s,model_path=(f.training_directory(s)/'model_final.pth').relative_to(f.ROOT).as_posix(),
            model_sha256=c.sha(f.training_directory(s)/'model_final.pth') if s in f.REUSED_SEEDS else None,
            new_training=(s in f.NEW_SEEDS),selection='final_200000_only') for s in f.ALL_SEEDS]
        c.write(planned,f.DATA/'planned_final_models.json')
        report=dict(passed=True,tests=tested.testsRun,original_mechanism_tests_reused=True,
            original_preflight_sha256=c.sha(f.original.RESULTS/'preflight.json'),initialization_seeds=list(f.NEW_SEEDS),
            recovery=recovery,original_learner_and_runner_unchanged=True,demonstration_transitions=7260,
            teacher_mismatches=0,new_training_seeds=list(f.NEW_SEEDS),reused_training_seeds=list(f.REUSED_SEEDS),
            additional_online_steps=600000,additional_offline_updates=60000,additional_online_updates=598503,
            final500_scene_sha256=scene_sha,final500_order_checked=True,final500_models=5,final500_episodes=2500,
            independent500_used_for_tuning=False,multitask_used=False)
        c.write(report,attempt/'passed.json');c.write(report,f.RESULTS/'preflight.json')
        copied=c.file_map([f.DATA])
        copied={Path(name).relative_to(f.DATA.relative_to(f.ROOT)).as_posix():value for name,value in copied.items()}
        c.write(dict(created=datetime.now().isoformat(),config=config,shared_config=shared,source_sha256=hashes,
            protected_sha256=c.file_map(protected_paths),copied_files_sha256=copied,
            input_sha256=c.file_map([f.original.DATA/'freeze.json',ce.DATA/'freeze.json',ce.DATA/'frozen500.json',
                f.ROOT/shared['dataset']['validation_reference']]),
            final500_scene_sha256=scene_sha,baseline_audit=baseline,baseline500=baseline500,checks=report),f.DATA/'freeze.json')
        f.verify()
        from analyze_fixed_qdagger_five import analyze
        analyze()
        print(f'PREFLIGHT PASS: {tested.testsRun} extension tests; original seeds preserved; exactly three new training runs and 2500 final500 episodes authorized.',flush=True)
    except BaseException as error:
        c.write(dict(passed=False,error=repr(error),traceback=traceback.format_exc(),formal_training_started=False),attempt/'failure.json');raise


def final_audit():
    from evaluate_fixed_qdagger_five import verify_completed
    f.verify();models=[]
    for seed in f.ALL_SEEDS:
        assert f.complete(seed) and verify_completed(seed)
        directory=f.training_directory(seed)
        saved=torch.load(directory/'recovery_latest.pth',map_location='cpu',weights_only=False)
        assert saved['loop']['phase']=='complete' and saved['loop']['total']==200000
        assert saved['loop']['offline_updates']==20000 and saved['loop']['online_updates']==199501
        assert saved['agent']['update_steps']==219501 and len(saved['loop']['replay'])==10000
        assert not saved['loop']['collector'].pending and saved['environment'] is None
        assert all(r.teacher_action==-1 for r in saved['loop']['replay'])
        models.append(dict(seed=seed,model_path=(directory/'model_final.pth').relative_to(f.ROOT).as_posix(),
            model_sha256=c.sha(directory/'model_final.pth'),new_training=(seed in f.NEW_SEEDS),
            online_steps=200000,offline_updates=20000,online_updates=199501,full_recovery_verified=True,final500_complete=True))
        del saved
    c.csv_write(models,f.RESULTS/'final_model_index.csv')
    c.write(dict(passed=True,completed=datetime.now().isoformat(),models=models,new_training_runs=3,
        reused_training_runs=2,final500_episodes=2500,old_results_unchanged=True),f.RESULTS/'completion_audit.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--train',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    if args.train and args.evaluate_only:parser.error('Select training pipeline or evaluation only')
    torch.set_num_threads(1)
    if not args.train and not args.evaluate_only:prepare();return
    f.verify();f.OUT.mkdir(parents=True,exist_ok=True);lock=f.OUT/'runner.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    from evaluate_fixed_qdagger_five import evaluate_seed
    from analyze_fixed_qdagger_five import analyze
    def status(phase,seed=None):
        c.write(dict(running=True,complete=False,pid=os.getpid(),phase=phase,active_seed=seed,new_seeds=list(f.NEW_SEEDS),
            reused_seeds=list(f.REUSED_SEEDS),maximum_new_runs=3,updated=datetime.now().isoformat()),f.RESULTS/'queue_status.json')
    try:
        for seed in f.ALL_SEEDS:
            if not f.complete(seed):
                if args.evaluate_only:continue
                status('training',seed);run_one(seed,args.resume)
            status('final500',seed);evaluate_seed(seed);analyze()
        trained=[s for s in f.ALL_SEEDS if f.complete(s)]
        if len(trained)==5:final_audit()
        analyze()
        complete=len(trained)==5
        c.write(dict(running=False,complete=complete,training_complete_seeds=trained,
            new_seeds=list(f.NEW_SEEDS),reused_seeds=list(f.REUSED_SEEDS),final500_episodes=2500 if complete else None,
            updated=datetime.now().isoformat()),f.RESULTS/'queue_status.json')
    except BaseException as error:
        c.write(dict(running=False,complete=False,error=repr(error),traceback=traceback.format_exc(),
            updated=datetime.now().isoformat()),f.RESULTS/'queue_status.json');raise
    finally:lock.unlink()


if __name__=='__main__':main()

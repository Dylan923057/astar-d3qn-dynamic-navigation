"""Frozen500 confirmation, final weights only; no training or teacher decisions."""
from __future__ import annotations
from datetime import datetime
from time import perf_counter
from unittest.mock import patch
import fixed_qdagger_five_common as f
from fixed_qdagger_agent import ObservableTeacher,QDaggerAgent
c=f.c


def directory(seed):return f.OUT/'final500'/f'seed_{seed}'


def registration(seed):
    frozen=f.verify(check_protected=False)
    if not f.complete(seed):raise ValueError('Final200000 model is not complete')
    source=f.training_directory(seed);result=c.read(source/'result.json')
    return dict(seed=seed,model_path=(source/'model_final.pth').relative_to(f.ROOT).as_posix(),
        model_sha256=result['final_model_sha256'],training_freeze_sha256=result['freeze_sha256'],
        extension_freeze_sha256=c.sha(f.DATA/'freeze.json'),scene_sha256=frozen['final500_scene_sha256'],
        copied_scene_sha256=c.sha(f.DATA/'frozen500.json'),scene_count=500,checkpoint_steps=200000,
        epsilon=0,teacher=False,dynamic_shield=False,static_mask=True,updates=0,device='cpu',
        max_steps=300,used_for_tuning=False,source_sha256=c.sha(f.ROOT/'scripts/evaluate_fixed_qdagger_five.py'),
        rollout_sha256=c.sha(f.ROOT/'src/astar_d3qn/evaluation/rollout.py'))


def verify_completed(seed):
    target=directory(seed)
    if not (target/'completion.json').exists():return False
    reg=registration(seed)
    assert c.read(target/'registration.json')==reg and c.read(target/'completion.json')['complete']
    scenes=c.read(f.DATA/'frozen500.json')['scenarios'];details=c.rows(target/'scene_details.csv')
    c.scene_order(details,scenes);failed={r['scenario_id']:r for r in details if not c.flag(r['safe_success'])}
    trajectories=c.read(target/'failure_trajectories.json')
    assert len(trajectories)==len(failed) and {t['scenario_id'] for t in trajectories}==set(failed)
    for t in trajectories:
        row=failed[t['scenario_id']]
        assert len(t['steps'])==int(row['steps']) and t['termination_reason']==row['termination_reason']
    summary=c.read(target/'summary.json');expected=c.summarize(details)
    for key,value in expected.items():
        assert summary[key] is None if value is None else abs(summary[key]-value)<1e-10
    assert c.sha(f.ROOT/reg['model_path'])==reg['model_sha256']
    return True


def evaluate_seed(seed):
    if verify_completed(seed):print(f'[QDagger500 seed{seed}] existing complete evaluation reused.',flush=True);return
    frozen=f.verify(check_protected=False);reg=registration(seed);target=directory(seed)
    target.mkdir(parents=True,exist_ok=True)
    if (target/'registration.json').exists():assert c.read(target/'registration.json')==reg
    else:c.write(reg,target/'registration.json')
    resolved=f.DATA/'final500_registrations'/f'seed_{seed}.json'
    if resolved.exists():assert c.read(resolved)==reg
    else:c.write(reg,resolved)
    _,shared,inputs=f.configure();scenes=c.read(f.DATA/'frozen500.json')['scenarios']
    agent=c.legacy.runtime.make_agent(shared,seed,'cpu');agent.load_weights(f.ROOT/reg['model_path'])
    for start in range(0,500,50):
        path=target/f'batch_{start:03d}.json'
        if path.exists():
            cached=c.read(path);assert cached['registration']==reg
            c.scene_order(cached['details'],scenes[start:start+50]);continue
        clock=perf_counter()
        with patch.object(ObservableTeacher,'query',side_effect=AssertionError('Teacher queried during final500')), \
             patch.object(QDaggerAgent,'train_update',side_effect=AssertionError('Update during final500')):
            _,details,failures=c.evaluate(agent,scenes[start:start+50],shared,inputs)
        c.write(dict(registration=reg,details=details,failures=failures,seconds=perf_counter()-clock),path)
        c.write(dict(running=True,seed=seed,completed_scenes=start+50,updated=datetime.now().isoformat()),
            f.RESULTS/'final500_progress.json')
        print(f'[QDagger500 seed{seed}] {start+50}/500',flush=True)
    batches=[c.read(target/f'batch_{start:03d}.json') for start in range(0,500,50)]
    details=[r for b in batches for r in b['details']];failures=[r for b in batches for r in b['failures']]
    c.scene_order(details,scenes);assert c.sha(f.ROOT/reg['model_path'])==reg['model_sha256']
    c.csv_write(details,target/'scene_details.csv');c.write(failures,target/'failure_trajectories.json')
    summary=c.summarize(details);c.write(summary,target/'summary.json')
    c.write(dict(complete=True,episodes=500,failures=len(failures),seconds=sum(b['seconds'] for b in batches),
        model_sha256=reg['model_sha256'],scene_sha256=reg['scene_sha256']),target/'completion.json')
    verify_completed(seed)
    print(f'[QDagger500 seed{seed}] COMPLETE safe={summary["safe_success_rate"]:.2%} collision={summary["dynamic_collision_rate"]:.2%} timeout={summary["timeout_rate"]:.2%}',flush=True)

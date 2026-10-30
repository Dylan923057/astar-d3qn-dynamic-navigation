"""Confirm five frozen CE final models; reuse the registered margin final500."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback
import torch
import fixed_supplement_common as c

NAME = 'asvc_fixed_ce_final500_v1'
DATA, OUT, RESULTS = (c.ROOT/folder/NAME for folder in ('data','outputs','results'))
KEYS = ('safe_success_rate','dynamic_collision_rate','timeout_rate','mean_steps','mean_wait_steps',
    'mean_wait_fraction','pooled_wait_fraction','mean_path_efficiency_success_only','safe_success_weighted_path_efficiency')


def ce_directory(seed):
    return c.ROOT/'outputs'/('asvc_fixed_ce_v1' if seed<2 else 'asvc_fixed_ce_five_seeds_v1')/f'seed_{seed}'


def prepare():
    if (DATA/'freeze.json').exists(): return verify()
    c.ce.verify_freeze()
    scenes, scene_sha = c.final_scenes(); assert len(scenes) == 500
    registry = c.read(c.ROOT/'data/asvc_current_method.json')
    protected = [c.ROOT/'data/asvc_current_method.json']
    sources = [Path(__file__),c.ROOT/'scripts/fixed_supplement_common.py',c.ROOT/f'configs/{NAME}.yaml']
    planned = []
    for seed in range(5):
        model = c.ROOT/registry['final_models'][str(seed)]
        baseline = c.ROOT/'results/asvc_persistent_supervision_five_seeds_v1/final500'/f'seed_{seed}'
        reg = c.read(baseline/'registration.json')
        for key, value in dict(seed=seed,model_sha256=c.sha(model),scene_sha256=scene_sha,scene_count=500,
            checkpoint_steps=200000,epsilon=0,teacher=False,static_mask=True,dynamic_shield=False,device='cpu').items():
            assert reg[key] == value, f'Margin cache protocol differs: {seed}/{key}'
        assert reg['rollout_sha256'] == c.sha(c.ROOT/'src/astar_d3qn/evaluation/rollout.py')
        assert c.read(baseline/'completion.json')['complete']
        details = c.rows(baseline/'scene_details.csv'); c.scene_order(details, scenes)
        summary = c.summarize(details)
        for key in KEYS[:3]: assert abs(summary[key]-float(c.read(baseline/'summary.json')[key]))<1e-12
        failures = c.ROOT/'outputs/asvc_persistent_supervision_five_seeds_v1/final500'/f'seed_{seed}'/'failure_trajectories.json'
        assert failures.exists() and len(c.read(failures)) == sum(not c.flag(r['safe_success']) for r in details)
        ce_dir = ce_directory(seed); result = c.read(ce_dir/'result.json')
        assert c.read(ce_dir/'completion.json')['complete'] and result['environment_steps']==200000
        assert result['gradient_updates']==199501 and result['final_model_sha256']==c.sha(ce_dir/'model_final.pth')
        assert result['seed']==seed
        protected.extend([model,baseline,failures])
        protected.extend(ce_dir/n for n in ('model_final.pth','run_manifest.json','result.json','completion.json','validation_curve.csv','validation_details.csv'))
        planned.append(dict(seed=seed,model_path=(ce_dir/'model_final.pth').relative_to(c.ROOT).as_posix(),
            model_sha256=result['final_model_sha256'],reused_margin=baseline.relative_to(c.ROOT).as_posix(),
            margin_model_sha256=c.sha(model),margin_failure_path=failures.relative_to(c.ROOT).as_posix()))
    assert not any((OUT/f'seed_{s}').exists() for s in range(5)), 'Refusing unregistered existing outputs'
    freeze = dict(created=datetime.now().isoformat(),config=c.legacy.load_config(c.ROOT/f'configs/{NAME}.yaml'),
        scene_sha256=scene_sha,scene_order=[s['scenario_id'] for s in scenes],models=planned,
        source_sha256=c.file_map(sources),protected_sha256=c.file_map(protected),training_runs=0,
        new_evaluation_episodes=2500,reused_margin_episodes=2500,parameter_tuning=False)
    DATA.mkdir(parents=True,exist_ok=True)
    for name in freeze['source_sha256']:
        target=DATA/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(c.ROOT/name,target)
    c.write(dict(scenarios=scenes),DATA/'frozen500.json')
    freeze['copied_scene_sha256']=c.sha(DATA/'frozen500.json')
    c.write(freeze,DATA/'freeze.json'); c.write(dict(passed=True,models_checked=10,margin_reused=2500,
        model_and_scene_sha_order_verified=True,training_runs=0),RESULTS/'preflight.json')
    print('CE final500 preflight PASS: five final CE models; 2500 margin episodes reused.',flush=True)
    return freeze


def verify():
    frozen=c.read(DATA/'freeze.json');c.check_files(frozen['source_sha256']);c.check_files(frozen['protected_sha256'])
    assert c.sha(DATA/'frozen500.json')==frozen['copied_scene_sha256']
    return frozen


def report():
    frozen=verify(); records=[]; all_details=[]; pairs=[]; per_scene=[]
    for item in frozen['models']:
        seed=item['seed']; selected={}
        for method in ('margin','ce'):
            directory=c.ROOT/item['reused_margin'] if method=='margin' else OUT/f'seed_{seed}'
            if not (directory/'completion.json').exists(): continue
            details=c.rows(directory/'scene_details.csv'); selected[method]=details
            records.append(dict(method=method,seed=seed,reused=(method=='margin'),**c.summarize(details)))
            all_details.extend(dict(method=method,seed=seed,**row) for row in details)
        if len(selected)==2:
            left,right=records[-2:]
            pairs.append(dict(method='CE_minus_margin',seed=seed,**{k:right[k]-left[k] if right[k] is not None and left[k] is not None else None for k in KEYS}))
            for a,b in zip(selected['margin'],selected['ce']):
                assert a['scenario_id']==b['scenario_id']
                per_scene.append(dict(seed=seed,scenario_id=a['scenario_id'],
                    delta_safe_success=int(c.flag(b['safe_success']))-int(c.flag(a['safe_success'])),
                    delta_dynamic_collision=int(c.flag(b['dynamic_collision']))-int(c.flag(a['dynamic_collision'])),
                    delta_timeout=int(b['termination_reason']=='timeout')-int(a['termination_reason']=='timeout'),
                    delta_steps=float(b['steps'])-float(a['steps']),delta_wait_steps=float(b['wait_steps'])-float(a['wait_steps']),
                    delta_path_efficiency_success_in_both=float(b['path_efficiency'])-float(a['path_efficiency']) if a['path_efficiency'] and b['path_efficiency'] else None))
    c.csv_write(records,RESULTS/'per_seed.csv');c.csv_write(c.summary_table(records,KEYS),RESULTS/'mean_sample_sd.csv')
    c.csv_write(pairs,RESULTS/'paired_per_seed.csv');c.csv_write(c.summary_table(pairs,KEYS),RESULTS/'paired_mean_sample_sd.csv')
    c.csv_write(all_details,RESULTS/'all_scene_details.csv');c.csv_write(per_scene,RESULTS/'paired_scene_differences.csv')
    validation=[]
    for seed in range(5):
        for method in ('margin','ce'):
            directory=c.ce.baseline_directory(seed) if method=='margin' else ce_directory(seed)
            row=c.rows(directory/'validation_curve.csv')[-1]
            validation.append(dict(method=method,seed=seed,split='repeated_frozen50_validation',**row))
    c.csv_write(validation,RESULTS/'final50_validation_separate.csv')
    complete=len(pairs)==5
    lines=['# 固定任务CE最终500确认评估','',f'状态：{len(pairs)}/5配对seed完成。CE新增{len(pairs)*500}回合；大间隔原2500回合复用。',
        '', '最终模型固定200000步。epsilon=0，静态合法动作掩码，无教师代选/动态保护/更新。500确认与反复使用的50验证分别报告；未据500成绩调参或挑检查点。',
        '', '|方法|seed|安全成功|动态碰撞|超时|平均步数|平均等待|成功路径效率|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['method']}|{r['seed']}|{r['safe_success_rate']:.2%}|{r['dynamic_collision_rate']:.2%}|{r['timeout_rate']:.2%}|{r['mean_steps']:.3f}|{r['mean_wait_steps']:.3f}|{r['mean_path_efficiency_success_only']:.4f}|")
    lines+=['','均值与样本SD见mean_sample_sd.csv，配对差及其样本SD见paired_*。统计单位为训练seed，500重复场景不是500个独立训练seed。',
        '路径效率继承原定义L*/执行步数，仅成功任务汇总；另报安全成功加权效率（失败置0），避免只看成功子集。步数和等待包含失败回合。逐场景与全部失败轨迹保留。',
        '', '[原50最终验证，分开报告](final50_validation_separate.csv)。QDagger两seed结果在独立目录，未在500确认池选参数。']
    (RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(dict(complete=complete,completed_ce_seeds=[p['seed'] for p in pairs],new_episodes=len(pairs)*500,
        reused_margin_episodes=2500,updated=datetime.now().isoformat()),RESULTS/'status.json')


def run():
    frozen=prepare();scenes=c.read(DATA/'frozen500.json')['scenarios'];config,inputs=c.configuration()
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'runner.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        for item in frozen['models']:
            seed=item['seed'];directory=OUT/f'seed_{seed}'
            registration=dict(seed=seed,model_sha256=item['model_sha256'],scene_sha256=frozen['scene_sha256'],
                freeze_sha256=c.sha(DATA/'freeze.json'),scene_count=500,checkpoint_steps=200000,
                epsilon=0,teacher=False,static_mask=True,dynamic_shield=False,device='cpu')
            if (directory/'completion.json').exists():
                assert c.read(directory/'registration.json')==registration;print(f'CE500 seed{seed} reused.',flush=True);continue
            directory.mkdir(parents=True,exist_ok=True)
            if (directory/'registration.json').exists():assert c.read(directory/'registration.json')==registration
            else:c.write(registration,directory/'registration.json')
            agent=c.legacy.runtime.make_agent(config,seed,'cpu');agent.load_weights(c.ROOT/item['model_path'])
            for start in range(0,500,50):
                part=directory/f'batch_{start:03d}.json'
                if part.exists():
                    saved=c.read(part);c.scene_order(saved['details'],scenes[start:start+50]);continue
                clock=perf_counter();_,details,failures=c.evaluate(agent,scenes[start:start+50],config,inputs)
                c.write(dict(details=details,failures=failures,seconds=perf_counter()-clock),part)
                c.write(dict(running=True,pid=os.getpid(),seed=seed,completed_scenes=start+50,
                    updated=datetime.now().isoformat()),RESULTS/'status.json')
                print(f'[CE500 seed{seed}] {start+50}/500',flush=True)
            batches=[c.read(directory/f'batch_{start:03d}.json') for start in range(0,500,50)]
            details=[r for b in batches for r in b['details']];failures=[r for b in batches for r in b['failures']]
            c.scene_order(details,scenes);assert c.sha(c.ROOT/item['model_path'])==item['model_sha256']
            c.csv_write(details,directory/'scene_details.csv');c.write(failures,directory/'failure_trajectories.json')
            c.write(c.summarize(details),directory/'summary.json');c.write(dict(complete=True,episodes=500,
                failure_trajectories=len(failures),seconds=sum(b['seconds'] for b in batches)),directory/'completion.json')
            report()
        report();verify()
    except BaseException as error:
        c.write(dict(complete=False,running=False,error=repr(error),traceback=traceback.format_exc()),RESULTS/'failure.json');raise
    finally:lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--report',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.run:run()
    elif args.report:report()
    else:prepare()

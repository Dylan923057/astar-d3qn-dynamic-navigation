"""Audit the completed seed0 pilot and existing paired control. No training or final500."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import torch

import run_staged_withdrawal_pilot as entry
import diagnose_comparison_evidence as diagnostic
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc
from astar_d3qn.envs.types import Observation
from astar_d3qn.utils.io import write_json, write_records_csv


def main():
    torch.set_num_threads(1)
    common=entry.common; config=common.legacy.load_config(entry.CONFIG); inputs=entry.validate(config)
    problem=inputs[0]; registered=common.read_json(common.ROOT/config['staged_withdrawal']['artifact_root']/'freeze_manifest.json')
    new=common.ROOT/config['experiment']['output_root']/entry.METHOD/'seed_0'
    run=common.read_json(new/'result.json'); completion=common.read_json(new/'completion.json')
    if (not completion['complete'] or completion['environment_steps']!=200000 or run['environment_steps']!=200000
            or run['gradient_updates']!=199501 or run['source_sha256']!=entry.source_hashes()
            or run['source_sha256']!=registered['source_sha256'] or run['config']!=config
            or run['initial_state_sha256']!=registered['paired_initial_state_sha256']
            or run['freeze_manifest_sha256']!=common.sha256(common.ROOT/config['staged_withdrawal']['artifact_root']/'freeze_manifest.json')
            or run['test_data_used'] or run['evaluation_advice'] or run['independent500_used']
            or common.sha256(new/'model_final.pth')!=run['final_model_sha256']):
        raise ValueError('Incomplete, unpaired, modified or incompatible pilot.')
    old,oldrun=common.audit_run(common.legacy.load_config(common.CONFIG),'advice_bound_margin',0)
    if run['initial_state_sha256']!=oldrun['initial_state_sha256']:
        raise ValueError('Unpaired full initial training state.')
    scenes=common.read_json(common.ROOT/config['dataset']['validation_reference'])['scenarios']
    expected_ids=[s['scenario_id'] for s in scenes]
    distances=static_distances(problem); diagnostic.static_distances_cached=distances
    output=common.ROOT/config['experiment']['check_root']/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    metrics=[]; checkpoints=[]; phases=[]; curves={}; fingerprints={}; initialized_model_checks=0
    for method,directory in (('advice_bound_margin',old),(entry.METHOD,new)):
        curve=load_csv(directory/'validation_curve.csv'); curves[method]=curve
        details=load_csv(directory/'validation_details.csv'); all_behaviors=[]
        if len(curve)!=21 or len(details)!=1050: raise ValueError('Missing validation observations.')
        for point in curve:
            step=int(point['environment_steps_total']); entries=[r for r in details if int(r['environment_steps_total'])==step]
            if [r['scenario_id'] for r in entries]!=expected_ids:
                raise ValueError('Validation scenes or order differs.')
            for key,expected in (
                ('safe_success_rate',np.mean([float(r['safe_success']) for r in entries])),
                ('dynamic_collision_rate',np.mean([float(r['dynamic_collision']) for r in entries])),
                ('timeout_rate',np.mean([r['termination_reason']=='timeout' for r in entries]))):
                if not np.isclose(float(point[key]),expected): raise ValueError('Validation aggregation mismatch.')
            traces=common.read_json(directory/f'validation_failures_{step:06d}.json')
            if {t['scenario_id'] for t in traces}!={r['scenario_id'] for r in entries if not float(r['safe_success'])}:
                raise ValueError('Missing or extra failure traces.')
            behavior=[trace_behavior(t,distances) for t in traces]
            checkpoints.append(dict(method=method,seed=0,checkpoint=step,success=float(point['safe_success_rate']),
                **failure_counts(behavior)))
            all_behaviors.extend(dict(checkpoint=step,**b) for b in behavior)
        x=np.asarray([int(r['environment_steps_total']) for r in curve]); y=np.asarray([float(r['safe_success_rate']) for r in curve])
        if x[0]!=0 or x[-1]!=200000 or np.any(np.diff(x)<=0): raise ValueError('Incorrect evaluation clocks.')
        row=dict(method=method,seed=0,full_aulc=aulc(x,y),last_five_mean=float(y[-5:].mean()),
            final_success=y[-1],final_collision=float(curve[-1]['dynamic_collision_rate']),final_timeout=float(curve[-1]['timeout_rate']))
        for left,right in ((0,60000),(60000,80000),(80000,100000),(100000,120000),(100000,200000),(140000,200000)):
            row[f'aulc_{left}_{right}']=aulc(x,y,left,right)
        for left,right in ((80000,100500),(100000,120500),(80000,160500),(100000,200000)):
            indices=np.flatnonzero((x>=left)&(x<=right)); index=indices[np.argmin(y[indices])]
            row[f'min_{left}_{right}']=float(y[index]); row[f'min_step_{left}_{right}']=int(x[index])
            selected=[b for b in all_behaviors if left<=b['checkpoint']<=right]
            for k,v in failure_counts(selected).items(): row[f'{left}_{right}_{k}']=v
        reached=np.flatnonzero(y>=.9); triples=[i for i in range(len(y)-2) if np.all(y[i:i+3]>=.9)]
        post=[i for i in range(len(y)-2) if x[i]>=100000 and np.all(y[i:i+3]>=.9)]
        row.update(first90=int(x[reached[0]]) if len(reached) else None,
            first_three90_confirmed=int(x[triples[0]+2]) if triples else None,
            post_exit_first90=int(x[np.flatnonzero((x>=100000)&(y>=.9))[0]]) if np.any((x>=100000)&(y>=.9)) else None,
            post_exit_three90_confirmed=int(x[post[0]+2]) if post else None)
        metrics.append(row)
        learning=load_csv(directory/'value_learning.csv'); advice=load_csv(directory/'advice_budget.csv')
        if sum(int(r['updates']) for r in learning)!=199501: raise ValueError('Incomplete update log.')
        for left,right in ((60000,80000),(80000,100000),(100000,120000),(120000,200000)):
            chosen=[r for r in learning if left<int(r['bin_end_step'])<=right]; updates=sum(int(r['updates']) for r in chosen)
            action=[r for r in advice if left<int(r['bin_end_step'])<=right]
            p=dict(method=method,seed=0,left=left,right=right,updates=updates,
                sampled_supervised_fraction=sum(int(r['teacher_label_samples']) for r in chosen)/(updates*64) if left<100000 else None,
                q_abs_max=max(float(r['q_abs_max']) for r in chosen),
                teacher_execution_rate=sum(int(r['applied']) for r in action)/sum(int(r['steps']) for r in action))
            for key in ('td_loss_mean','demo_margin_loss_mean','teacher_margin_weight_mean'):
                p[key]=sum(float(r[key])*int(r['updates']) for r in chosen)/updates
            if directory==new:
                census=[r for r in load_csv(new/'collection_label_census.csv') if left<int(r['bin_end_step'])<=right]
                td=[r for r in load_csv(new/'true_training_td.csv') if left<int(r['bin_end_step'])<=right]
                p['eligible_collected_fraction']=sum(int(r['eligible_labels']) for r in census)/sum(int(r['collected']) for r in census)
                p['true_training_td_abs_mean']=sum(float(r['td_abs_mean'])*int(r['updates']) for r in td)/updates
                p['true_training_td_abs_max']=max(float(r['td_abs_max']) for r in td)
            else:
                p['eligible_collected_fraction']=None; p['true_training_td_abs_mean']=None; p['true_training_td_abs_max']=None
            phases.append(p)
        final=directory/'model_final.pth'; fingerprints[str(final.relative_to(common.ROOT))]=common.sha256(final)
        agent=common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG),problem,0,'advice_bound_margin','cpu')
        agent.load_weights(final)
        s,rows,_=diagnostic.rollout(config,problem,agent,scenes,'autonomous')
        for key in ('safe_success_rate','dynamic_collision_rate','timeout_rate'):
            if not np.isclose(s[key],float(curve[-1][key])): raise ValueError('Final autonomous replay mismatch.')
        write_records_csv(rows,output/f'{method}_final_episodes.csv')
    # Verify the recorded weights, not just seed integers: pre-intervention checkpoints must coincide.
    for a,b in zip(curves['advice_bound_margin'][:6],curves[entry.METHOD][:6]):
        if int(a['environment_steps_total'])!=int(b['environment_steps_total']): raise ValueError('Early clocks differ.')
        step=int(a['environment_steps_total'])
        weights1=torch.load(old/f'model_step_{step:06d}.pth',map_location='cpu',weights_only=True)
        weights2=torch.load(new/f'model_step_{step:06d}.pth',map_location='cpu',weights_only=True)
        if any(not torch.equal(weights1[k],weights2[k]) for k in weights1): raise ValueError('Models diverged before intervention.')
        initialized_model_checks+=1
    # Same historical failure observations for both methods. Diagnostic labels only; never added to replay.
    state_root=common.ROOT/'results/teaching_withdrawal_diagnosis/20261011_012947_389779'
    data=np.load(state_root/'seed0_observations.npz'); index=load_csv(state_root/'seed0_state_index.csv')
    states=[Observation(s,c) for s,c in zip(data['spatial'],data['scalars'])]; masks=data['masks']
    labels=np.asarray([int(r['teacher_action']) for r in index]); clear=np.asarray([r['observed_teacher_risk']=='False' for r in index])
    ranking=[]
    for method,directory in (('advice_bound_margin',old),(entry.METHOD,new)):
        agent=common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG),problem,0,'advice_bound_margin','cpu')
        for anchor in (80000,90000,100000,110000,120000):
            point=min(curves[method],key=lambda r:abs(int(r['environment_steps_total'])-anchor)); step=int(point['environment_steps_total'])
            path=directory/f'model_step_{step:06d}.pth'; fingerprints[str(path.relative_to(common.ROOT))]=common.sha256(path)
            agent.load_weights(path); q=diagnostic.q_batch(agent,states)
            greedy=np.where(masks,q,-np.inf).argmax(axis=1)
            other=masks.copy(); other[np.arange(len(states)),labels]=False
            advantage=q[np.arange(len(states)),labels]-np.where(other,q,-np.inf).max(axis=1)
            ranking.append(dict(method=method,seed=0,anchor=anchor,checkpoint=step,state_count=len(states),clear_count=int(clear.sum()),
                teacher_agreement_clear=float((greedy[clear]==labels[clear]).mean()),
                teacher_advantage_clear_mean=float(advantage[clear].mean()),wait_rate=float((greedy==4).mean())))
    for name,rows in (('metrics',metrics),('checkpoint_behavior',checkpoints),('training_phases',phases),('same_state_rankings',ranking)):
        write_records_csv(rows,output/(name+'.csv'))
    write_records_csv([dict(metric=k,new_minus_old=metrics[1][k]-metrics[0][k]) for k in metrics[0]
                       if k not in ('method','seed') and metrics[0][k] is not None and metrics[1][k] is not None],output/'paired_difference.csv')
    if any(common.sha256(common.ROOT/p)!=sha for p,sha in fingerprints.items()): raise ValueError('Source model mutated.')
    write_json(dict(complete=True,training_started=False,independent500_used=False,statistical_unit='seed',n=1,
        initial_state_paired=True,identical_early_checkpoints=initialized_model_checks,
        same_validation_scenes=50,verified_detail_rows=2100,final_autonomous_episodes_reproduced=100,
        model_sha256=fingerprints,new_training_seconds=run['online_training_seconds'],total_seconds=run['total_seconds'],
        limitations=['no seed-level sample SD or general superiority inference from n=1',
            'same-state audit uses selected old failure observations; teacher agreement is not dynamic safety',
            'scheduled advice probability area reduced 8%; not equal teacher count',
            'original actual training TD errors and per-state replay coverage unavailable']),output/'verification.json')
    text=['seed0分阶段撤教结果','',
        '原组合 vs 提前撤代选（8万代选退出、10万监督退出）：']
    for key in ('full_aulc','aulc_100000_200000','aulc_140000_200000','min_80000_160500','last_five_mean','final_success'):
        text.append(f'{key}: {metrics[0][key]:.2%} → {metrics[1][key]:.2%}')
    text.extend(['', 'seed0表现为低谷减轻：最低点仍在10万附近，8万附近没有出现同样深的低谷，后半程成绩提高。',
        '仍有10万步88%的低谷，以及13万/15万94%的回退；不能称为已解决撤教不稳。',
        '初始完整状态相同，0–5万的六个checkpoint参数完全一致；未改奖励/地图/网络/监督日程。',
        '此结果支持提前自主采集的单seed可行性，不足以证明因果或五seed稳定增益。',
        '8%较少的代选日程面积是对比边界。旧方法没有真实TD误差分布，不能用旧Huber损失冒充。',
        '下一步最少的训练扩展是固定该日程验证seed1/2，尤其seed2原低谷58%；不先调参。',
        '当前入口仅允许seed0，本轮没有更改冻结入口或启动其他seed。'])
    (output/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method,curve in curves.items():
        label='Original combination' if method=='advice_bound_margin' else 'Override exits at 80k'
        axes[0].plot([int(r['environment_steps_total'])/1000 for r in curve],[float(r['safe_success_rate'])*100 for r in curve],marker='o',label=label)
        rows=[r for r in ranking if r['method']==method]
        axes[1].plot([r['checkpoint']/1000 for r in rows],[r['teacher_advantage_clear_mean'] for r in rows],marker='o',label=label)
    axes[0].set(ylabel='Autonomous success (%)',ylim=(0,105),title='Paired seed0, original 50 validation scenes')
    axes[1].set(ylabel='Q(A*) - max Q(other legal)',title='Same selected old failure observations')
    axes[1].axhline(0,color='gray',linestyle=':')
    for ax in axes:
        for x in (80,100): ax.axvline(x,color='gray',linestyle='--',alpha=.6)
        ax.set_xlabel('Environment steps (thousands)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(output/f'paired_seed0.{ext}',dpi=160)
    plt.close(fig); print(output)


if __name__=='__main__': main()

"""Audit completed seed0/1/2 replications; validation only, no training or test tuning."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import run_replication_seeds12 as entry
from analyze_runtime_path_pilot import load_csv, trace_behavior, static_distances
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc, mean_records
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT, common, legacy = entry.ROOT, entry.common, entry.legacy


def directories(arm, seed):
    cfg,old,inputs=entry.configure(arm)
    if arm=='staged':
        reference=ROOT/'outputs/whole_map_91701_value_repair_v1/advice_bound_margin'/f'seed_{seed}'
        method='staged_withdrawal'
    else:
        reference=ROOT/'outputs/whole_map_91701_dqfd_comparison_v1/dqfd'/f'seed_{seed}'
        method='dqfd_no_pretrain'
    new=(ROOT/old['experiment']['output_root'] if seed==0 else ROOT/cfg['experiment']['output_root'])/entry.METHODS[arm]/f'seed_{seed}'
    return cfg,inputs,reference,new,method


def fmt(value):
    return '未达到' if value is None else str(value)


def analyze():
    torch.set_num_threads(1)
    output=ROOT/'results/replication_seeds012_v1'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # No partial-summary substitution for the four required runs.
    for arm in ('staged','dqfd'):
        for seed in (0,1,2):
            _,_,old,new,_=directories(arm,seed)
            for directory in (old,new):
                if not common.read_json(directory/'completion.json')['complete']:
                    raise ValueError(f'Incomplete run: {directory}')
    output.mkdir(parents=True,exist_ok=False)
    metrics=[]; points=[]; phases=[]; all_curves=[]; audits=[]
    source_files=entry.source_hashes()
    for arm in ('staged','dqfd'):
        for seed in (0,1,2):
            cfg,inputs,old,new,new_method=directories(arm,seed)
            distances=static_distances(inputs[0])
            ids=[s['scenario_id'] for s in common.read_json(ROOT/cfg['dataset']['validation_reference'])['scenarios']]
            for method,directory in (('advice_bound_margin' if arm=='staged' else 'dqfd',old),(new_method,new)):
                run=common.read_json(directory/'result.json')
                if run['environment_steps']!=200000 or (run.get('final_model_sha256') is not None and
                        common.sha256(directory/'model_final.pth')!=run['final_model_sha256']):
                    raise ValueError(f'Budget/final model changed: {directory}')
                final_weights=torch.load(directory/'model_final.pth',map_location='cpu',weights_only=True)
                step_weights=torch.load(directory/'model_step_200000.pth',map_location='cpu',weights_only=True)
                if final_weights.keys()!=step_weights.keys() or any(not torch.equal(final_weights[k],step_weights[k]) for k in final_weights):
                    raise ValueError('Final model differs from the 200000-step checkpoint.')
                if directory==new and seed>0 and run['source_sha256']!=source_files:
                    raise ValueError('New replication sources changed.')
                curve=load_csv(directory/'validation_curve.csv'); detail=load_csv(directory/'validation_details.csv')
                x=np.asarray([int(r['environment_steps_total']) for r in curve]);y=np.asarray([float(r['safe_success_rate']) for r in curve])
                if len(curve)!=21 or len(detail)!=1050 or x[0]!=0 or x[-1]!=200000 or np.any(np.diff(x)<=0):
                    raise ValueError('Incomplete or unordered curves/details.')
                all_behaviors=[]
                for point in curve:
                    step=int(point['environment_steps_total'])
                    selected=[r for r in detail if int(r['environment_steps_total'])==step]
                    if [r['scenario_id'] for r in selected]!=ids:raise ValueError('Fixed validation IDs/order differ.')
                    expected=dict(safe_success_rate=np.mean([float(r['safe_success']) for r in selected]),
                        dynamic_collision_rate=np.mean([float(r['dynamic_collision']) for r in selected]),
                        timeout_rate=np.mean([r['termination_reason']=='timeout' for r in selected]))
                    if any(not np.isclose(float(point[k]),v) for k,v in expected.items()):raise ValueError('Validation aggregates disagree.')
                    traces=common.read_json(directory/f'validation_failures_{step:06d}.json')
                    if len(traces)!=sum(not float(r['safe_success']) for r in selected) or {t['scenario_id'] for t in traces}!={r['scenario_id'] for r in selected if not float(r['safe_success'])}:
                        raise ValueError('Missing/duplicate failure traces.')
                    behaviors=[trace_behavior(t,distances) for t in traces]
                    counts=failure_counts(behaviors)
                    row=dict(method=method,seed=seed,step=step,**expected,**counts)
                    points.append(row)
                    all_behaviors.extend(dict(step=step,**b) for b in behaviors)
                    all_curves.append(dict(method=method,seed=seed,**point))
                row=dict(method=method,seed=seed,full_aulc=aulc(x,y),
                    aulc_0_60000=aulc(x,y,0,60000),aulc_0_80000=aulc(x,y,0,80000),
                    aulc_0_10000=aulc(x,y,0,10000),aulc_10000_80000=aulc(x,y,10000,80000),
                    aulc_60000_80000=aulc(x,y,60000,80000),aulc_80000_100000=aulc(x,y,80000,100000),
                    aulc_100000_200000=aulc(x,y,100000,200000),aulc_140000_200000=aulc(x,y,140000,200000),
                    final_success=float(y[-1]),final_collision=float(curve[-1]['dynamic_collision_rate']),
                    final_timeout=float(curve[-1]['timeout_rate']),last_five_mean=float(y[-5:].mean()),
                    online_updates=run.get('online_updates',run['gradient_updates']),
                    total_updates=run['gradient_updates'],pretraining_updates=run.get('pretraining_updates',0),
                    online_training_seconds=run.get('online_training_seconds'))
                final_counts=next(p for p in points if p['method']==method and p['seed']==seed and p['step']==200000)
                for k in ('collision_count','timeout_count','waiting_timeout_count','looping_timeout_count','other_timeout_count'):
                    row['final_'+k]=final_counts[k]
                reached=np.flatnonzero(y>=.9)
                triple=[i for i in range(len(y)-2) if np.all(y[i:i+3]>=.9)]
                row.update(first90=int(x[reached[0]]) if len(reached) else None,
                    first_three90_start=int(x[triple[0]]) if triple else None,
                    first_three90_confirmed=int(x[triple[0]+2]) if triple else None)
                for left,right in ((60000,80500),(80000,100500),(100000,120500),(80000,160500),
                                   (60000,200000),(160000,200000)):
                    indexes=np.flatnonzero((x>=left)&(x<=right));i=indexes[np.argmin(y[indexes])]
                    row[f'min_{left}_{right}']=float(y[i]);row[f'min_step_{left}_{right}']=int(x[i])
                    counts=failure_counts([b for b in all_behaviors if left<=b['step']<=right])
                    row.update({f'{left}_{right}_{k}':v for k,v in counts.items()})
                trough_step=row['min_step_80000_160500']
                after=np.flatnonzero((x>trough_step)&(y>=.9))
                post_triples=[i for i in range(len(y)-2) if x[i]>trough_step and np.all(y[i:i+3]>=.9)]
                row.update(recovery_first90_after_min=int(x[after[0]]) if len(after) else None,
                    recovery_three90_start=int(x[post_triples[0]]) if post_triples else None,
                    recovery_three90_confirmed=int(x[post_triples[0]+2]) if post_triples else None)
                metrics.append(row)
                if arm=='staged':
                    advice=load_csv(directory/'advice_budget.csv');learning=load_csv(directory/'value_learning.csv')
                    if sum(int(r['updates']) for r in learning)!=199501:raise ValueError('Incomplete learning log.')
                    if any(int(r['teacher_label_samples']) or float(r['teacher_margin_weight_mean']) for r in learning if int(r['bin_start_step'])>=100001):
                        raise ValueError('Supervision persists after exit.')
                    if method==new_method and any(int(r['requested']) or int(r['applied']) for r in advice if int(r['bin_start_step'])>=80001):
                        raise ValueError('Advice persists after 80k exit.')
                    for left,right in ((0,60000),(60000,80000),(80000,100000),(100000,200000)):
                        chosen=[r for r in advice if left<int(r['bin_end_step'])<=right]
                        steps=sum(int(r['steps']) for r in chosen)
                        phases.append(dict(method=method,seed=seed,left=left,right=right,steps=steps,
                            advice_requested=sum(int(r['requested']) for r in chosen),
                            advice_applied=sum(int(r['applied']) for r in chosen),
                            actual_teacher_execution_rate=sum(int(r['applied']) for r in chosen)/steps,
                            planned_probability_sum=sum(float(r['probability_sum']) for r in chosen)))
                # Re-evaluate saved final weights on the same 50 validation scenes, never pick a checkpoint.
                agent=entry.make_stage(cfg,inputs[0],seed,'cpu') if arm=='staged' else entry.dq.make_agent(cfg,inputs[0],seed,'dqfd','cpu')
                agent.load_weights(directory/'model_final.pth')
                summary,episode_rows,_=legacy.old.evaluate(cfg,inputs,seed,agent)
                if any(not np.isclose(summary[k],float(curve[-1][k])) for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate')):
                    raise ValueError('Final autonomous re-evaluation mismatch.')
                write_records_csv(episode_rows,output/f'{method}_seed{seed}_final50.csv')
                audits.append(dict(method=method,seed=seed,final_model_sha256=common.sha256(directory/'model_final.pth'),
                    validation_points=21,validation_episodes=1050,final50_replayed=True))
            if arm=='staged':
                a=load_csv(old/'validation_curve.csv');b=load_csv(new/'validation_curve.csv')
                for r,s in zip(a[:6],b[:6]):
                    if r['environment_steps_total']!=s['environment_steps_total']:raise ValueError('Pre-intervention clocks differ.')
                    step=int(r['environment_steps_total'])
                    w1=torch.load(old/f'model_step_{step:06d}.pth',map_location='cpu',weights_only=True)
                    w2=torch.load(new/f'model_step_{step:06d}.pth',map_location='cpu',weights_only=True)
                    if w1.keys()!=w2.keys() or any(not torch.equal(w1[k],w2[k]) for k in w1):
                        raise ValueError(f'seed{seed}: pre-60k checkpoint differs at {step}.')
    numerical=[k for k in metrics[0] if k not in ('method','seed')]
    # Threshold/censored steps are reported individually, never imputed as 200k or silently averaged over achievers.
    summary_metrics=[k for k in numerical if 'first' not in k and 'recovery' not in k and 'min_step' not in k]
    means=mean_records(metrics,summary_metrics)
    pairs=[]
    for old,new in (('advice_bound_margin','staged_withdrawal'),('dqfd','dqfd_no_pretrain')):
        for seed in (0,1,2):
            before=next(r for r in metrics if r['method']==old and r['seed']==seed)
            after=next(r for r in metrics if r['method']==new and r['seed']==seed)
            pairs.append(dict(method=new+'_minus_'+old,seed=seed,
                **{k:after[k]-before[k] if after[k] is not None and before[k] is not None else None for k in numerical}))
    write_records_csv(metrics,output/'per_seed_metrics.csv')
    write_records_csv(means,output/'mean_sample_sd.csv')
    write_records_csv(pairs,output/'paired_differences.csv')
    write_records_csv(mean_records(pairs,summary_metrics),output/'paired_mean_sample_sd.csv')
    write_records_csv(points,output/'checkpoint_failures.csv')
    write_records_csv(phases,output/'actual_teaching_rates.csv')
    write_records_csv(all_curves,output/'validation_curves.csv')
    write_json(dict(passed=True,statistical_unit='seed',n=3,source_sha256=source_files,
        audits=audits,old_results_preserved=True,seed0_reused=True,independent500_used=False,
        scheduled_advice_area_reduction=.08,notes=[
            'AULC integrates actual episode-end checkpoint positions using linear interpolation.',
            '80k exit window: 80k–100500; 100k exit window: 100k–120500; overall trough: 80k–160500.',
            '500-step right tolerance includes episode-end validation jitter, not an extra scheduled validation.',
            'Recovery first90 and three consecutive points are distinct; neither asserts permanent stability.',
            'Waiting/looping timeout groups are mutually exclusive; checkpoint sums reuse the same 50 scenes.',
            'No-pretrain still retains and samples all 7260 offline demonstrations; it is not demonstration-free D3QN.',
            'DQfD step0 is after 20000 offline updates; control step0 is random initialization. Report 10k–80k AULC separately as well.',
            'Recovery preserves learning state; static A* action cache may be deterministically rebuilt on restart, affecting attributed planner search cost.',
        ]),output/'verification.json')
    lines=['三seed配对复核（统计单位seed；样本标准差ddof=1；仅原50验证场景）','',
        '|方法|seed|全程AULC%|10–20万AULC%|8–16万最低%（步）|最终成功%|最终碰撞/等待/绕圈超时|',
        '|---|---:|---:|---:|---|---:|---|']
    for r in metrics:
        lines.append(f'|{r["method"]}|{r["seed"]}|{100*r["full_aulc"]:.2f}|{100*r["aulc_100000_200000"]:.2f}|{100*r["min_80000_160500"]:.0f} ({r["min_step_80000_160500"]})|{100*r["final_success"]:.0f}|{r["final_collision_count"]}/{r["final_waiting_timeout_count"]}/{r["final_looping_timeout_count"]}|')
    lines.extend(['','|方法|0–8万AULC%|全程AULC%|10–20万AULC%|最后五次均值%|最终成功%|最终碰撞%|最终超时%|',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in means:
        lines.append('|'+r['method']+'|'+ '|'.join(f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_sample_sd"]:.2f}' for k in (
            'aulc_0_80000','full_aulc','aulc_100000_200000','last_five_mean','final_success','final_collision','final_timeout'))+'|')
    lines.extend(['','8–16万九次验证的失败累计（同50场景重复验证，不是独立测试样本）：',
        '|方法|seed|碰撞|等待超时|绕圈超时|', '|---|---:|---:|---:|---:|'])
    for r in metrics:
        if r['method'] in ('advice_bound_margin','staged_withdrawal'):
            lines.append(f'|{r["method"]}|{r["seed"]}|{r["80000_160500_collision_count"]}|{r["80000_160500_waiting_timeout_count"]}|{r["80000_160500_looping_timeout_count"]}|')
    lines.extend(['','实际教师代选率：',
        '|方法|seed|6–8万%|8–10万%|', '|---|---:|---:|---:|'])
    for method in ('advice_bound_margin','staged_withdrawal'):
        for seed in (0,1,2):
            chosen=[r for r in phases if r['method']==method and r['seed']==seed]
            values=[next(r['actual_teacher_execution_rate'] for r in chosen if r['left']==left) for left in (60000,80000)]
            lines.append(f'|{method}|{seed}|{100*values[0]:.3f}|{100*values[1]:.3f}|')
    lines.extend(['','逐seed撤教低谷和恢复：'])
    for r in metrics:
        if r['method'] in ('advice_bound_margin','staged_withdrawal'):
            lines.append(f'{r["method"]} seed{r["seed"]}: 8万窗口最低{100*r["min_80000_100500"]:.0f}%; 10万窗口最低{100*r["min_100000_120500"]:.0f}%; 低谷后首次90%={fmt(r["recovery_first90_after_min"])}; 连续三点起点/确认={fmt(r["recovery_three90_start"])}/{fmt(r["recovery_three90_confirmed"])}。')
    lines.extend(['','同seed配对差（百分点）：'])
    for r in pairs:
        lines.append(f'{r["method"]} seed{r["seed"]}: 全程{100*r["full_aulc"]:+.2f}, 0–8万{100*r["aulc_0_80000"]:+.2f}, 10–20万{100*r["aulc_100000_200000"]:+.2f}, 8–16万最低{100*r["min_80000_160500"]:+.0f}, 最终{100*r["final_success"]:+.0f}。')
    paired_summary=mean_records(pairs,summary_metrics)
    lines.extend(['','|比较|全程配对差pp|0–8万配对差pp|10–20万配对差pp|', '|---|---:|---:|---:|'])
    for r in paired_summary:
        lines.append('|'+r['method']+'|'+ '|'.join(f'{100*r[k+"_mean"]:+.2f}±{100*r[k+"_sample_sd"]:.2f}' for k in ('full_aulc','aulc_0_80000','aulc_100000_200000'))+'|')
    s2old=next(r for r in metrics if r['method']=='advice_bound_margin' and r['seed']==2)
    s2new=next(r for r in metrics if r['method']=='staged_withdrawal' and r['seed']==2)
    trough=next(r for r in points if r['method']=='staged_withdrawal' and r['seed']==2 and r['step']==s2new['min_step_80000_160500'])
    lines.extend(['','|问题|本轮证据与结论|', '|---|---|',
        '|分阶段学习效率收益是否重复|三seed全程AULC均增加；不等于撤教后稳定性均改善。|',
        '|撤教低谷是否解决|seed0、1最低点均提高18pp；seed2最低点反而下降22pp。尚未解决。|',
        f'|seed2低谷是否移动|{s2old["min_step_80000_160500"]}步的58%变为{s2new["min_step_80000_160500"]}步的36%，后移且加深；新低谷{trough["collision_count"]}碰撞、{trough["waiting_timeout_count"]}等待超时、{trough["looping_timeout_count"]}绕圈超时。|',
        '|取消DQfD预训练是否有一致影响|三seed全程AULC均下降；10–20万配对差有正有负，最终seed2反而提高2pp。不能称为长期必然变差。|',
        '|DQfD前期差异边界|原DQfD的0步点已经预训练，对照0步是随机网络。0–8万差均值−5.27pp；另算1–8万为+0.52±1.10pp（逐seed见CSV），不能把全程差解读为全程学习能力差。|',
        '|后续安排|固定当前两方案，各补seed3、4共四次20万步；现入口仍限制seed1、2，扩展须另建冻结入口。随后四方法×五seed×同一冻结500场景=10000回合最终确认；不选最佳模型、不按测试结果调参。|'])
    lines.extend(['','边界：只确认三个seed；分阶段预设代选概率总量减少8%，不能单独归因为错开退出时钟。',
        'DQfD比较同为20万在线交互；原方法总219501梯度更新，对照199501，且对照仍使用永久离线示范。',
        '成本：原完整组合缺少训练耗时；现有耗时跨运行时段，只作描述，不据此宣称计算速度更快。各seed更新次数及可得耗时见CSV。',
        '后续：不据本轮结果改日程；补齐同设置seed3、4，再对五seed各方法20万最终模型统一使用已有冻结独立500场景确认。',
        '本轮未启动seed3、4，也未使用独立500场景。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(output,all_curves)
    print('\n'.join(lines));print(output)
    return output


def plot(output,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True)
    for row,methods in enumerate((('advice_bound_margin','staged_withdrawal'),('dqfd','dqfd_no_pretrain'))):
        for seed in (0,1,2):
            ax=axes[row,seed]
            for method in methods:
                points=[r for r in curves if r['method']==method and r['seed']==seed]
                ax.plot([int(r['environment_steps_total'])/1000 for r in points],
                    [100*float(r['safe_success_rate']) for r in points],marker='.',label=method)
            ax.axvline(80,color='gray',linestyle=':',linewidth=1)
            ax.axvline(100,color='gray',linestyle='--',linewidth=1)
            ax.set(title=f'seed {seed}',ylim=(0,103),xlabel='Online environment steps (k)')
            ax.grid(alpha=.2)
            if seed==0:ax.set_ylabel('Autonomous success (%)');ax.legend(fontsize=8)
    figure.tight_layout();figure.savefig(output/'paired_validation_curves.png',dpi=180);plt.close(figure)


if __name__=='__main__':analyze()

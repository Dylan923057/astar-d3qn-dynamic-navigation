"""Audit completed seed0-4 replications; validation only, no training or test tuning."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import run_replication_seeds12 as entry
import run_replication_seeds34 as entry34
from analyze_runtime_path_pilot import load_csv, trace_behavior, static_distances
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc, mean_records
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT, common, legacy = entry.ROOT, entry.common, entry.legacy


def directories(arm, seed):
    protocol = entry34 if seed >= 3 else entry
    cfg,old,inputs=protocol.configure(arm)
    if arm=='staged':
        reference=ROOT/'outputs/whole_map_91701_value_repair_v1/advice_bound_margin'/f'seed_{seed}'
        method='staged_withdrawal'
    else:
        reference=ROOT/'outputs/whole_map_91701_dqfd_comparison_v1/dqfd'/f'seed_{seed}'
        method='dqfd_no_pretrain'
    new=(ROOT/old['experiment']['output_root'] if seed==0 else ROOT/cfg['experiment']['output_root'])/protocol.METHODS[arm]/f'seed_{seed}'
    return cfg,inputs,reference,new,method


def analyze():
    torch.set_num_threads(1)
    output=ROOT/'results/replication_seeds01234_v1'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # No partial-summary substitution for the four required runs.
    for arm in ('staged','dqfd'):
        for seed in (0,1,2,3,4):
            _,_,old,new,_=directories(arm,seed)
            for directory in (old,new):
                if not common.read_json(directory/'completion.json')['complete']:
                    raise ValueError(f'Incomplete run: {directory}')
    output.mkdir(parents=True,exist_ok=False)
    metrics=[]; points=[]; phases=[]; all_curves=[]; audits=[]
    source_files=dict(entry.source_hashes()); source_files.update(entry34.source_hashes())
    for arm in ('staged','dqfd'):
        for seed in (0,1,2,3,4):
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
                if directory==new and seed>0 and run['source_sha256']!=(entry34 if seed>=3 else entry).source_hashes():
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
        for seed in (0,1,2,3,4):
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
    write_json(dict(passed=True,statistical_unit='seed',n=5,source_sha256=source_files,
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
    write_report(output,metrics,means,pairs,points,phases)
    plot(output,all_curves)
    print(output)
    return output


def write_report(output,metrics,means,pairs,points,phases):
    names=dict(advice_bound_margin='原完整组合',staged_withdrawal='分阶段撤教',dqfd='原DQfD',dqfd_no_pretrain='DQfD无预训练')
    lines=['# 五seed配对复核：分阶段撤教与DQfD取消预训练','',
        '统计单位为训练seed（0–4），±为样本标准差ddof=1。全部来自原50固定验证场景；验证关闭教师、路线引导和动态保护，保留共有静态合法动作掩码。',
        '新增两方法共十次20万步训练已完成，seed0复用单seed试验，seed1、2及seed3、4分别使用独立冻结入口；原两方法十次训练直接复用。没有重训或挑选最佳检查点。独立500场景尚未用于这两个新对照的最终确认。','',
        '|方法|seed|全程AULC%|0–8万AULC%|10–20万AULC%|8–16万最低%（步）|最终成功%|最终碰撞/等待/绕圈超时|',
        '|---|---:|---:|---:|---:|---|---:|---|']
    for r in metrics:
        lines.append(f'|{names[r["method"]]}|{r["seed"]}|{100*r["full_aulc"]:.2f}|{100*r["aulc_0_80000"]:.2f}|{100*r["aulc_100000_200000"]:.2f}|{100*r["min_80000_160500"]:.0f} ({r["min_step_80000_160500"]})|{100*r["final_success"]:.0f}|{r["final_collision_count"]}/{r["final_waiting_timeout_count"]}/{r["final_looping_timeout_count"]}|')
    lines.extend(['','|方法|全程AULC%|0–8万AULC%|1–8万AULC%|10–20万AULC%|最后五次均值%|最终成功%|碰撞%|超时%|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in means:
        lines.append('|'+names[r['method']]+'|'+ '|'.join(f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_sample_sd"]:.2f}' for k in (
            'full_aulc','aulc_0_80000','aulc_10000_80000','aulc_100000_200000','last_five_mean','final_success','final_collision','final_timeout'))+'|')
    lines.extend(['','同seed配对差（新−原，百分点）：','',
        '|比较|seed|全程|0–8万|1–8万|10–20万|8–16万最低|最终成功|',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in pairs:
        lines.append('|'+r['method']+'|'+str(r['seed'])+'|'+ '|'.join(f'{100*r[k]:+.2f}' for k in (
            'full_aulc','aulc_0_80000','aulc_10000_80000','aulc_100000_200000','min_80000_160500','final_success'))+'|')
    lines.extend(['','退出窗口与恢复（实际环境步）：','',
        '|方法|seed|8万窗口最低%|10万窗口最低%|低谷后首次≥90%|连续三点起点/确认|8–16万碰撞/等待/绕圈超时|',
        '|---|---:|---:|---:|---|---|---|'])
    for r in metrics:
        if r['method'] not in ('advice_bound_margin','staged_withdrawal'):continue
        show=lambda v:'未达到' if v is None else str(v)
        lines.append(f'|{names[r["method"]]}|{r["seed"]}|{100*r["min_80000_100500"]:.0f}|{100*r["min_100000_120500"]:.0f}|{show(r["recovery_first90_after_min"])}|{show(r["recovery_three90_start"])}/{show(r["recovery_three90_confirmed"])}|{r["80000_160500_collision_count"]}/{r["80000_160500_waiting_timeout_count"]}/{r["80000_160500_looping_timeout_count"]}|')
    lines.extend(['','8万窗口为80000–100500，10万窗口为100000–120500；8–16万范围包含160500以内的第16万步验证点。500步容差容纳回合结束验证的时钟偏移。累计失败数重复使用同50场景，不能当作独立样本。首次恢复、连续三点及长期稳定是不同概念。','',
        '|方法|seed|6–8万实际代选率%|8–10万实际代选率%|', '|---|---:|---:|---:|'])
    for method in ('advice_bound_margin','staged_withdrawal'):
        for seed in range(5):
            rows=[r for r in phases if r['method']==method and r['seed']==seed]
            rates=[next(r['actual_teacher_execution_rate'] for r in rows if r['left']==t) for t in (60000,80000)]
            lines.append(f'|{names[method]}|{seed}|{100*rates[0]:.3f}|{100*rates[1]:.3f}|')
    paired_means=mean_records(pairs,['full_aulc','aulc_0_80000','aulc_10000_80000','aulc_100000_200000','final_success'])
    lines.extend(['','配对差的五seed均值±样本标准差：','',
        '|比较|全程pp|0–8万pp|1–8万pp|10–20万pp|最终成功pp|', '|---|---:|---:|---:|---:|---:|'])
    for r in paired_means:
        lines.append('|'+r['method']+'|'+ '|'.join(f'{100*r[k+"_mean"]:+.2f}±{100*r[k+"_sample_sd"]:.2f}' for k in (
            'full_aulc','aulc_0_80000','aulc_10000_80000','aulc_100000_200000','final_success'))+'|')
    staged=[r for r in pairs if r['method'].startswith('staged_')]
    control=[r for r in pairs if r['method'].startswith('dqfd_')]
    lines.extend(['','|问题|证据及结论|', '|---|---|',
        f'|分阶段全程效率收益是否重复|{sum(r["full_aulc"]>0 for r in staged)}/5 seed提高。撤教最低点提高{sum(r["min_80000_160500"]>0 for r in staged)}/5、降低{sum(r["min_80000_160500"]<0 for r in staged)}/5；效率改善不能替代稳定性证据。|',
        f'|取消DQfD预训练|全程AULC降低{sum(r["full_aulc"]<0 for r in control)}/5；10–20万提高{sum(r["aulc_100000_200000"]>0 for r in control)}/5、降低{sum(r["aulc_100000_200000"]<0 for r in control)}/5。原DQfD的0步点已经离线预训练，对照从随机网络开始；另列1–8万AULC。|',
        '|seed2低谷|原组100089步58%，新组110052步36%，后移且加深；新低谷9次等待、23次绕圈超时，无碰撞。尚不能声称已解决撤教低谷。|',
        '|比较预算|两者在线环境交互均200000步、在线更新199501次；原DQfD另有20000离线更新，总219501，对照总199501。取消预训练仍保留7260永久示范、PER、多步TD、margin及L2。|',
        '|代选边界|分阶段预设代选日程总量减少8%；不是等教师使用量实验，不能把收益单独归因为错开退出时间。|',
        '|seed4全程差值|分阶段相对原组仅+0.0027个百分点，属于几乎持平；不能把数学上的正值包装成明显改善。|',
        '|计算成本|原完整组合缺少耗时记录；其余可得耗时及梯度更新见CSV。跨运行时段耗时只作描述，不宣称速度公平优胜。|',
        '|下一步|四方法×五seed的20万步最终模型，统一在已有冻结独立500场景评估，共10000回合；原方法优先复用同场景、同最终模型SHA匹配的已有评估。此次仅整理验证结果，未运行独立500、未调参或选检查点。|','',
        '详细数据：[逐seed指标](per_seed_metrics.csv)、[五seed均值](mean_sample_sd.csv)、[配对差](paired_differences.csv)、[配对均值](paired_mean_sample_sd.csv)、[所有验证曲线](validation_curves.csv)、[逐点失败分类](checkpoint_failures.csv)、[实际代选](actual_teaching_rates.csv)、[审核记录](verification.json)、[曲线图](paired_validation_curves.png)。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(output,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,5,figsize=(20,7),sharex=True,sharey=True)
    for row,methods in enumerate((('advice_bound_margin','staged_withdrawal'),('dqfd','dqfd_no_pretrain'))):
        for seed in (0,1,2,3,4):
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

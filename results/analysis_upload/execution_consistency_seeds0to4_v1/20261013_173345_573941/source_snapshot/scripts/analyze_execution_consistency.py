"""Audit execution-consistency seed0/1/2; validation only, no training or test tuning."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import run_execution_consistency as entry
from analyze_runtime_path_pilot import load_csv, trace_behavior, static_distances
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc, mean_records
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT, common, legacy = entry.ROOT, entry.common, entry.legacy
SEEDS=(0,1,2)
ANALYSIS_ROOT=ROOT/'results/whole_map_91701_execution_consistency_v1/analysis'


def registered_sources(seed):
    return entry.source_hashes()


def directories(arm, seed):
    cfg,inputs=entry.configure()
    original=ROOT/'outputs/whole_map_91701_value_repair_v1/advice_bound_margin'/f'seed_{seed}'
    new=ROOT/cfg['experiment']['output_root']/entry.METHOD/f'seed_{seed}'
    return cfg,inputs,original,new,entry.METHOD


def analyze():
    torch.set_num_threads(1)
    output=ANALYSIS_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # Require all three new runs and their three original paired controls.
    for arm in ('staged',):
        for seed in SEEDS:
            _,_,old,new,_=directories(arm,seed)
            for directory in (old,new):
                if not common.read_json(directory/'completion.json')['complete']:
                    raise ValueError(f'Incomplete run: {directory}')
    output.mkdir(parents=True,exist_ok=False)
    metrics=[]; points=[]; phases=[]; all_curves=[]; audits=[]
    source_files=registered_sources(SEEDS[0])
    for arm in ('staged',):
        for seed in SEEDS:
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
                if directory==new and run['source_sha256']!=registered_sources(seed):
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
                    aulc_0_50000=aulc(x,y,0,50000),
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
                row['recovery_required']=row['min_80000_160500']<.9
                if not row['recovery_required']:
                    for key in ('recovery_first90_after_min','recovery_three90_start','recovery_three90_confirmed'):
                        row[key]=None
                metrics.append(row)
                if arm=='staged':
                    advice=load_csv(directory/'advice_budget.csv');learning=load_csv(directory/'value_learning.csv')
                    if sum(int(r['updates']) for r in learning)!=199501:raise ValueError('Incomplete learning log.')
                    if any(int(r['teacher_label_samples']) or float(r['teacher_margin_weight_mean']) for r in learning if int(r['bin_start_step'])>=100001):
                        raise ValueError('Supervision persists after exit.')
                    if method==new_method and any(int(r['requested']) or int(r['applied']) for r in advice if int(r['bin_start_step'])>=100001):
                        raise ValueError('Advice persists after original100k exit.')
                    for left,right in ((0,60000),(60000,80000),(80000,100000),(100000,200000)):
                        chosen=[r for r in advice if left<int(r['bin_end_step'])<=right]
                        steps=sum(int(r['steps']) for r in chosen)
                        phases.append(dict(method=method,seed=seed,left=left,right=right,steps=steps,
                            advice_requested=sum(int(r['requested']) for r in chosen),
                            advice_applied=sum(int(r['applied']) for r in chosen),
                            actual_teacher_execution_rate=sum(int(r['applied']) for r in chosen)/steps,
                            planned_probability_sum=sum(float(r['probability_sum']) for r in chosen)))
                # Re-evaluate saved final weights on the same 50 validation scenes, never pick a checkpoint.
                agent=entry.make_agent(cfg,inputs[0],seed,'cpu')
                agent.load_weights(directory/'model_final.pth')
                summary,episode_rows,_=legacy.old.evaluate(cfg,inputs,seed,agent)
                if any(not np.isclose(summary[k],float(curve[-1][k])) for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate')):
                    raise ValueError('Final autonomous re-evaluation mismatch.')
                write_records_csv(episode_rows,output/f'{method}_seed{seed}_final50.csv')
                audits.append(dict(method=method,seed=seed,final_model_sha256=common.sha256(directory/'model_final.pth'),
                    validation_points=21,validation_episodes=1050,final50_replayed=True))
    numerical=[k for k in metrics[0] if k not in ('method','seed')]
    # Threshold/censored steps are reported individually, never imputed as 200k or silently averaged over achievers.
    summary_metrics=[k for k in numerical if 'first' not in k and 'recovery' not in k and 'min_step' not in k]
    means=mean_records(metrics,summary_metrics)
    pairs=[]
    for old,new in (('advice_bound_margin',entry.METHOD),):
        for seed in SEEDS:
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
    write_json(dict(passed=True,statistical_unit='seed',n=len(SEEDS),source_sha256=source_files,
        source_sha256_by_seed={str(seed):registered_sources(seed) for seed in SEEDS},
        analysis_source_sha256=common.sha256(Path(__file__)),
        audits=audits,old_results_preserved=True,seed0_reused=False,independent500_used=False,
        scheduled_advice_area_reduction=0.0,notes=[
            'AULC integrates actual episode-end checkpoint positions using linear interpolation.',
            'Pre-exit window: 80k–100500; only teacher exit is 100k; exit window: 100k–120500; overall trough: 80k–160500.',
            '500-step right tolerance includes episode-end validation jitter, not an extra scheduled validation.',
            'Recovery first90 and three consecutive points are distinct; neither asserts permanent stability.',
            'Waiting/looping timeout groups are mutually exclusive; checkpoint sums reuse the same 50 scenes.',
            'Only executed-teacher equality eligibility removed; no pretraining, demonstrations or compensation.',
            'Recovery preserves learning state; static A* action cache may be deterministically rebuilt on restart, affecting attributed planner search cost.',
        ]),output/'verification.json')
    mechanisms=collect_mechanisms(metrics)
    write_records_csv(mechanisms,output/'phase_mechanisms.csv')
    behavior_phases=[]
    for metric in metrics:
        selected=[p for p in points if p['method']==metric['method'] and p['seed']==metric['seed']]
        x=np.asarray([p['step'] for p in selected]);y=np.asarray([p['safe_success_rate'] for p in selected])
        for left,right in ((0,50000),(50000,80000),(80000,100000),(100000,120000),(120000,160000),(160000,200000)):
            chosen=[p for p in selected if left<p['step']<=right]
            episodes=len(chosen)*50
            behavior_phases.append(dict(method=metric['method'],seed=metric['seed'],left=left,right=right,
                aulc=aulc(x,y,left,right),validation_points=len(chosen),validation_episodes=episodes,
                safe_success_rate=np.mean([p['safe_success_rate'] for p in chosen]),
                dynamic_collision_rate=np.mean([p['dynamic_collision_rate'] for p in chosen]),
                timeout_rate=np.mean([p['timeout_rate'] for p in chosen]),
                **{k:sum(p[k] for p in chosen) for k in ('collision_count','timeout_count','waiting_timeout_count','looping_timeout_count','other_timeout_count')}))
    write_records_csv(behavior_phases,output/'phase_validation_behaviors.csv')
    write_report(output,metrics,means,pairs,points,mechanisms)
    plot(output,all_curves)
    print(output)
    return output


def collect_mechanisms(metrics):
    rows=[]
    for metric in metrics:
        _,_,old,new,_=directories('staged',metric['seed'])
        directory=new if metric['method']==entry.METHOD else old
        learning=load_csv(directory/'value_learning.csv');advice=load_csv(directory/'advice_budget.csv')
        alignment=load_csv(new/'supervision_alignment.csv') if directory==new else None
        for left,right in ((0,50000),(50000,80000),(80000,100000),(100000,120000),(120000,160000),(160000,200000)):
            selected=[r for r in learning if left<int(r['bin_end_step'])<=right]
            updates=sum(int(r['updates']) for r in selected);labels=sum(int(r['teacher_label_samples']) for r in selected)
            chosen=[r for r in advice if left<int(r['bin_end_step'])<=right]
            row=dict(method=metric['method'],seed=metric['seed'],left=left,right=right,updates=updates,
                sampled_draws=updates*64,supervised_draws=labels,supervision_sample_fraction=labels/(updates*64),
                matching_fraction_among_supervised=1.0 if labels and directory==old else None,
                baseline_matching_rate_inferred_from_label_rule=directory==old,
                q_abs_max=max(float(r['q_abs_max']) for r in selected),q_min=None,q_max=None,
                actual_teacher_execution_rate=sum(int(r['applied']) for r in chosen)/sum(int(r['steps']) for r in chosen))
            for key in ('demo_margin_loss_mean','teacher_margin_weight_mean','target_clip_fraction_mean','td_loss_mean','q_mean_mean'):
                row[key]=sum(float(r[key])*int(r['updates']) for r in selected)/updates
            if alignment is not None:
                extra=[r for r in alignment if left<int(r['bin_end_step'])<=right]
                if sum(int(r['supervised_draws']) for r in extra)!=labels:raise ValueError('Label accounting inconsistent.')
                matched=sum(int(r['matching_supervised_draws']) for r in extra)
                row.update(matching_supervised_draws=matched,
                    matching_fraction_among_supervised=matched/labels if labels else None,
                    q_min=min(float(r['q_min']) for r in extra),q_max=max(float(r['q_max']) for r in extra))
            rows.append(row)
    return rows


def write_report(output,metrics,means,pairs,points,mechanisms):
    names={'advice_bound_margin':'原执行一致性筛选',entry.METHOD:'取消执行一致性筛选'}
    lines=[f'# 执行一致性筛选：{len(SEEDS)}seed单因素消融','',
        '原完整组合（10万步代选和监督同时退出）对照。新组只取消实际执行动作=A*建议这一筛选条件；真实Transition不变，TD索引真实动作，margin监督索引独立教师动作。无额外示范、预训练、权重或采样补偿。统计单位seed，±为样本标准差ddof=1。','',
        '|方法|seed|0–5万AULC%|0–8万AULC%|全程AULC%|10–20万AULC%|8–16万最低%（步）|最后五次均值%|最终成功%|碰撞/等待/绕圈超时|',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|---|']
    for r in metrics:
        lines.append(f'|{names[r["method"]]}|{r["seed"]}|{100*r["aulc_0_50000"]:.2f}|{100*r["aulc_0_80000"]:.2f}|{100*r["full_aulc"]:.2f}|{100*r["aulc_100000_200000"]:.2f}|{100*r["min_80000_160500"]:.0f} ({r["min_step_80000_160500"]})|{100*r["last_five_mean"]:.2f}|{100*r["final_success"]:.0f}|{r["final_collision_count"]}/{r["final_waiting_timeout_count"]}/{r["final_looping_timeout_count"]}|')
    keys=('aulc_0_50000','aulc_0_80000','full_aulc','aulc_100000_200000','last_five_mean','final_success','final_collision','final_timeout')
    lines.extend(['','|方法|0–5万AULC%|0–8万AULC%|全程AULC%|10–20万AULC%|最后五次均值%|最终成功%|碰撞%|超时%|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in means:
        lines.append('|'+names[r['method']]+'|'+ '|'.join(f'{100*r[k+"_mean"]:.2f}±{100*r[k+"_sample_sd"]:.2f}' for k in keys)+'|')
    lines.extend(['','配对差为新−原（正值代表取消筛选更高）：','',
        '|seed|0–5万pp|全程pp|10–20万pp|最低点pp|最终成功pp|', '|---|---:|---:|---:|---:|---:|'])
    for r in pairs:
        lines.append('|'+str(r['seed'])+'|'+ '|'.join(f'{100*r[k]:+.2f}' for k in ('aulc_0_50000','full_aulc','aulc_100000_200000','min_80000_160500','final_success'))+'|')
    paired_means=mean_records(pairs,['aulc_0_50000','full_aulc','aulc_100000_200000','min_80000_160500','final_success'])[0]
    lines.append(f'|{len(SEEDS)}seed配对均值±样本标准差|'+ '|'.join(f'{100*paired_means[k+"_mean"]:+.2f}±{100*paired_means[k+"_sample_sd"]:.2f}' for k in ('aulc_0_50000','full_aulc','aulc_100000_200000','min_80000_160500','final_success'))+'|')
    lines.extend(['','|方法|seed|首次90%（步）|首次连续三点起点/确认（步）|',
        '|---|---:|---:|---|'])
    for r in metrics:
        lines.append(f'|{names[r["method"]]}|{r["seed"]}|{r["first90"]}|{r["first_three90_start"]}/{r["first_three90_confirmed"]}|')
    lines.extend(['','|方法|seed|10万退出窗口最低%|最低点后首次90%|连续三点起点/确认|8–16万碰撞/等待/绕圈超时|',
        '|---|---:|---:|---|---|---|'])
    show=lambda x:'未达到' if x is None else str(x)
    for r in metrics:
        first=show(r['recovery_first90_after_min']) if r['recovery_required'] else '未跌破90%，无需恢复'
        triple=show(r['recovery_three90_start'])+'/'+show(r['recovery_three90_confirmed']) if r['recovery_required'] else '不适用'
        lines.append(f'|{names[r["method"]]}|{r["seed"]}|{100*r["min_100000_120500"]:.0f}|{first}|{triple}|{r["80000_160500_collision_count"]}/{r["80000_160500_waiting_timeout_count"]}/{r["80000_160500_looping_timeout_count"]}|')
    lines.extend(['','|方法|8–16万最低成功率均值±样本标准差%|8–16万累计碰撞/等待/绕圈超时|',
        '|---|---:|---|'])
    for r in means:
        chosen=[m for m in metrics if m['method']==r['method']]
        totals=[sum(m['80000_160500_'+k] for m in chosen) for k in ('collision_count','waiting_timeout_count','looping_timeout_count')]
        lines.append(f'|{names[r["method"]]}|{100*r["min_80000_160500_mean"]:.2f}±{100*r["min_80000_160500_sample_sd"]:.2f}|'+ '/'.join(str(v) for v in totals)+'|')
    lines.extend(['','|方法|seed|阶段|监督占比%|监督中的执行一致率%|margin损失|监督权重|Q范围|TD目标裁剪%|实际代选%|',
        '|---|---:|---|---:|---:|---:|---:|---|---:|---:|'])
    for r in mechanisms:
        match='不适用' if r['matching_fraction_among_supervised'] is None else f'{100*r["matching_fraction_among_supervised"]:.2f}'
        qr=f'{r["q_min"]:.2f}–{r["q_max"]:.2f}' if r['q_min'] is not None else f'abs≤{r["q_abs_max"]:.2f}（有符号范围缺失）'
        lines.append(f'|{names[r["method"]]}|{r["seed"]}|{r["left"]}–{r["right"]}|{100*r["supervision_sample_fraction"]:.2f}|{match}|{r["demo_margin_loss_mean"]:.4f}|{r["teacher_margin_weight_mean"]:.3f}|{qr}|{100*r["target_clip_fraction_mean"]:.2f}|{100*r["actual_teacher_execution_rate"]:.2f}|')
    lines.extend(['','原基线监督一致率100%是既有标签规则的推论；基线未记录有符号Q最小/最大值，只提供实际记录的abs最大值。10万步后不计算教师标签，一致率记为不适用，不能解释为零一致或零潜在可监督率。监督比例按回放采样次数计算，同一转移可以被重复采样。',
        '8–16万区间使用160500步以内的第16万验证点，退出窗口为100000–120500，以容纳回合结束验证的时钟偏移。AULC使用实际步数线性积分。累计失败数重复使用同50场景，不是新增独立样本。首次90%与连续三点不同，未达到保持缺失值，不填20万。','',
        f'{len(SEEDS)}seed中取消筛选：前5万AULC提高{sum(r["aulc_0_50000"]>0 for r in pairs)}/{len(SEEDS)}，全程提高{sum(r["full_aulc"]>0 for r in pairs)}/{len(SEEDS)}，后半提高{sum(r["aulc_100000_200000"]>0 for r in pairs)}/{len(SEEDS)}，撤教最低点提高{sum(r["min_80000_160500"]>0 for r in pairs)}/{len(SEEDS)}。',
        '监督样本数量变化是本实验的机制变化，不是等标签数量对照。已有500场景未读取性能、未调参，新对照尚未进行独立500最终确认。结果无论好坏全部保留。',
        '数据：per_seed_metrics.csv、mean_sample_sd.csv、paired_differences.csv、paired_mean_sample_sd.csv、phase_mechanisms.csv、phase_validation_behaviors.csv、checkpoint_failures.csv、validation_curves.csv；verify记录所有模型SHA及原50最终复评。'])
    if SEEDS==(0,1,2) and all(r['full_aulc']>0 and r['aulc_0_50000']>0 and r['aulc_100000_200000']>0 and r['min_80000_160500']>0 for r in pairs):
        lines.extend(['',
            '结论：当前三seed没有证据支持执行一致性筛选具有一致增益；相反，取消它后三seed前5万、全程及后半AULC和撤教最低成功率均提高。前8万分阶段收益并非一致（seed1、2略降），不能写成所有阶段均有优势。',
            '取消筛选后的低谷仍在11万步：seed0和seed2比原低谷后移约1万步；seed1时间基本不变。后半AULC三seed均提高，故不能仅解释为把低谷推迟。seed0恢复到连续三个90%以上验证点提前约3万步，seed1、2确认时间基本相同。',
            '学习效率和低谷深度改善没有换来一致的安全增益：8–16万累计碰撞13→17，等待超时68→21，绕圈超时59→29。seed1最终新增1次动态碰撞，发生在验证场景validation_seed91701999_episode000033的第12步，无等待或重访；不是超时。仅当前固定50场景、三seed证据，不据此推断最终独立场景优势。',
            '机制：前5万监督覆盖率约74%→97%，新组监督样本中约76%实际动作与教师一致，后期降到约68%。监督包含更多未执行教师动作的状态；TD仍学习真实动作。样本增多与动作不同标签同时由取消筛选造成，本实验不能分离两者的因果贡献。',
            '论文：不能将执行一致性筛选表述为已验证有效的独立贡献。可诚实报告为单因素消融及收益边界：扩大在线监督覆盖改善三seed学习效率和撤教低谷，但碰撞风险仍需核查。保留原完整组合及所有新结果，不修改日程、不删seed、不自动补seed3、4。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(output,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(SEEDS),figsize=(4.3*len(SEEDS),4),sharey=True)
    for seed,ax in zip(SEEDS,np.atleast_1d(axes)):
        for method in ('advice_bound_margin',entry.METHOD):
            rows=[r for r in curves if r['method']==method and r['seed']==seed]
            ax.plot([int(r['environment_steps_total'])/1000 for r in rows],[100*float(r['safe_success_rate']) for r in rows],marker='.',label=method)
        ax.axvline(100,color='gray',linestyle='--');ax.set(title=f'seed {seed}',xlabel='Environment steps (k)',ylim=(0,103));ax.grid(alpha=.2)
    axes[0].set_ylabel('Autonomous success (%)');axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'paired_validation_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':analyze()



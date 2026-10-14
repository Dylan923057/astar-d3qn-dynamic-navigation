"""Combine fixed50 learning evidence and same-frozen500 final confirmation, without retraining."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

import latest_module_common as protocol
from analyze_runtime_path_pilot import load_csv,static_distances,trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc
from astar_d3qn.utils.io import write_json,write_records_csv

ROOT,common=protocol.ROOT,protocol.common
METRICS=('aulc_0_50000','full_aulc','aulc_100000_200000','min_80000_160500','min_90000_120500','min_100000_200000',
    'last_five_mean','validation50_final_success','independent500_success','independent500_collision','independent500_timeout',
    'final500_collision_count','final500_waiting_timeout_count','final500_looping_timeout_count','final500_other_timeout_count')


def behaviors(rows):
    return [dict(r,waiting_dominated=str(r['waiting_dominated']).lower()=='true',
        repeated_movement=str(r['repeated_movement']).lower()=='true',
        tail_period=int(r['tail_period']) if r['tail_period'] else None) for r in rows]


def mechanism(p):
    directory=p['directory'];value=directory/'value_learning.csv';advice=directory/'advice_budget.csv'
    if not value.is_file() or not advice.is_file():return []
    learning=load_csv(value);rates=load_csv(advice)
    alignment=load_csv(directory/'supervision_alignment.csv') if (directory/'supervision_alignment.csv').is_file() else []
    if not p['failed'] and p['method'] in protocol.core.METHODS:
        if sum(int(r['updates']) for r in learning)!=199501:raise ValueError('Incomplete update diagnostics.')
        if any(int(r['teacher_label_samples']) or float(r['teacher_margin_weight_mean']) for r in learning if int(r['bin_start_step'])>=100001):raise ValueError('Supervision after exit.')
        if p['method']==protocol.core.METHODS[1] and any(int(r['requested']) or int(r['applied']) for r in rates):raise ValueError('No-override arm used advice.')
        if p['method']==protocol.core.METHODS[0] and any(float(r['target_clip_fraction_mean']) for r in learning):raise ValueError('No-TD arm clipped targets.')
        if not any(int(r['teacher_label_samples'])>0 for r in learning if int(r['bin_start_step'])<100000):raise ValueError('Supervision stopped with override removal.')
    result=[]
    for left,right in ((0,50000),(50000,80000),(80000,100000),(100000,120000),(120000,160000),(160000,200000)):
        values=[r for r in learning if left<int(r['bin_end_step'])<=right]
        if not values:continue
        updates=sum(int(r['updates']) for r in values);labels=sum(int(r['teacher_label_samples']) for r in values)
        selected=[r for r in rates if left<int(r['bin_end_step'])<=right];steps=sum(int(r['steps']) for r in selected)
        # 10k collection bins match these stage boundaries; learning bins are 1k.
        extra=[r for r in alignment if left<int(r['bin_end_step'])<=right]
        matched=sum(int(r['matching_supervised_draws']) for r in extra) if extra else None
        row=dict(method=p['method'],seed=p['seed'],left=left,right=right,updates=updates,sampled_draws=updates*64,
            supervised_draws=labels,supervision_sample_fraction=labels/(updates*64),
            matching_fraction_among_supervised=matched/labels if matched is not None and labels else None,
            actual_override_rate=sum(int(r['applied']) for r in selected)/steps if steps else None,
            q_min=min(float(r['q_min']) for r in extra) if extra else None,
            q_max=max(float(r['q_max']) for r in extra) if extra else None,
            q_abs_max=max(float(r['q_abs_max']) for r in values),
            raw_target_min=min(float(r['raw_target_min']) for r in extra) if extra and all('raw_target_min' in r for r in extra) else None,
            raw_target_max=max(float(r['raw_target_max']) for r in values))
        for key in ('demo_margin_loss_mean','teacher_margin_weight_mean','td_loss_mean','target_clip_fraction_mean','raw_target_outside_fraction_mean'):
            row[key]=sum(float(r[key])*int(r['updates']) for r in values)/updates
        result.append(row)
    return result


def analyze():
    evaluation=ROOT/common.read_json(ROOT/'results/latest_all_eligible_modules_final500_v1/latest.json')['evaluation']
    final=common.read_json(evaluation/'verification.json')
    if not final['complete'] or final['frozen_scene_sha256']!=common.read_json(ROOT/'data'/protocol.core.NAME/'final500_registration.json')['frozen_scene_sha256']:
        raise ValueError('Final500 confirmation incomplete/changed.')
    test=load_csv(evaluation/'summary.csv')
    output=ROOT/'results/latest_all_eligible_modules_analysis_v1'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    rows=[];curves=[];failures=[];mechanisms=[];checks=[];costs=[]
    for method in protocol.ALL:
        for seed in range(5):
            p=protocol.run(method,seed);directory=p['directory']
            row=dict(method=method,seed=seed,numerical_failure=p['failed'],**{key:None for key in METRICS})
            curve=load_csv(directory/'validation_curve.csv')
            detail=load_csv(directory/'validation_details.csv')
            behavior_file=directory/'failure_behaviors.csv'
            if behavior_file.is_file():behavior=load_csv(behavior_file)
            else:
                # DQfD kept full failures instead of the newer behavior CSV.
                behavior=[]
                distances=static_distances(p['inputs'][0])
                for point in curve:
                    step=int(point['environment_steps_total'])
                    for trace in common.read_json(directory/f'validation_failures_{step:06d}.json'):
                        behavior.append(dict(environment_steps_total=step,**trace_behavior(trace,distances)))
                write_records_csv(behavior,output/f'derived_failure_behaviors_{method}_seed{seed}.csv')
            ids=[s['scenario_id'] for s in common.read_json(ROOT/p['config']['dataset']['validation_reference'])['scenarios']]
            x=np.asarray([int(r['environment_steps_total']) for r in curve]);y=np.asarray([float(r['safe_success_rate']) for r in curve])
            if x[0]!=0 or np.any(np.diff(x)<=0) or (not p['failed'] and (len(curve)!=21 or x[-1]!=200000)):
                raise ValueError('Validation clock/budget differs.')
            for point in curve:
                step=int(point['environment_steps_total'])
                selected=[r for r in detail if int(r['environment_steps_total'])==step]
                if len(selected)!=50 or [r['scenario_id'] for r in selected]!=ids:raise ValueError('Fixed50 scene identity/order differs.')
                expected=dict(safe_success_rate=np.mean([float(r['safe_success']) for r in selected]),
                    dynamic_collision_rate=np.mean([float(r['dynamic_collision']) for r in selected]),
                    timeout_rate=np.mean([r['termination_reason']=='timeout' for r in selected]))
                if any(not np.isclose(float(point[k]),value) for k,value in expected.items()):raise ValueError('Curve and scene results disagree.')
                failed=[r for r in behavior if int(r['environment_steps_total'])==step]
                if len(failed)!=sum(not float(r['safe_success']) for r in selected) or {r['scenario_id'] for r in failed}!={r['scenario_id'] for r in selected if not float(r['safe_success'])}:
                    raise ValueError('Missing failure behavior records.')
                failures.append(dict(method=method,seed=seed,step=step,**expected,**failure_counts(behaviors(failed))))
                curves.append(dict(point,method=method,seed=seed))
            for key,left,right in (('aulc_0_50000',0,50000),('full_aulc',0,200000),('aulc_100000_200000',100000,200000)):
                if x[-1]>=right:row[key]=aulc(x,y,left,right)
            for left,right in ((80000,160500),(90000,120500),(100000,200000)):
                chosen=np.flatnonzero((x>=left)&(x<=right))
                if len(chosen) and (not p['failed'] or x[-1]>=right):
                    i=chosen[np.argmin(y[chosen])];row[f'min_{left}_{right}']=float(y[i]);row[f'min_step_{left}_{right}']=int(x[i])
                else:row[f'min_step_{left}_{right}']=None
            reached=np.flatnonzero(y>=.9);triples=[i for i in range(len(y)-2) if np.all(y[i:i+3]>=.9)]
            row.update(first90=int(x[reached[0]]) if len(reached) else None,
                first_three90_start=int(x[triples[0]]) if triples else None,
                first_three90_confirmed=int(x[triples[0]+2]) if triples else None)
            trough=row.get('min_step_100000_200000');needed=row.get('min_100000_200000') is not None and row['min_100000_200000']<.9
            after=np.flatnonzero((x>trough)&(y>=.9)) if needed else []
            recovery=[i for i in range(len(y)-2) if x[i]>trough and np.all(y[i:i+3]>=.9)] if needed else []
            row.update(recovery_required=needed,recovery_first90=int(x[after[0]]) if len(after) else None,
                recovery_three90_start=int(x[recovery[0]]) if recovery else None,
                recovery_three90_confirmed=int(x[recovery[0]+2]) if recovery else None,
                complete_environment_steps=p['failure']['environment_steps'] if p['failed'] else 200000)
            if not p['failed']:
                row.update(last_five_mean=float(y[-5:].mean()),validation50_final_success=float(y[-1]))
                cost=p['result'];costs.append(dict(method=method,seed=seed,
                    online_updates=cost.get('online_updates',199501),total_gradient_updates=cost['gradient_updates'],
                    pretraining_updates=cost.get('pretraining_updates',20000 if method=='dqfd' else 0),
                    online_training_seconds=cost.get('online_training_seconds'),
                    budget_boundary='DQfD retains offline demos/PER and 20000 pretraining updates' if method=='dqfd' else 'No offline demos/pretraining'))
                # Check full saved failures at trough and final; other checkpoints reuse generated behavior CSV.
                for step in sorted({200000,row['min_step_80000_160500']}):
                    traces=common.read_json(directory/f'validation_failures_{step:06d}.json')
                    counted=failure_counts([trace_behavior(t,static_distances(p['inputs'][0])) for t in traces])
                    recorded=next(r for r in failures if r['method']==method and r['seed']==seed and r['step']==step)
                    if any(counted[k]!=recorded[k] for k in counted):raise ValueError('Saved trough/final trace classification differs.')
            final_row=next(r for r in test if r['method']==method and int(r['seed'])==seed)
            for source,target in (('safe_success_rate','independent500_success'),('dynamic_collision_rate','independent500_collision'),
                ('timeout_rate','independent500_timeout'),('collision_count','final500_collision_count'),
                ('waiting_timeout_count','final500_waiting_timeout_count'),('looping_timeout_count','final500_looping_timeout_count'),('other_timeout_count','final500_other_timeout_count')):
                row[target]=float(final_row[source]) if final_row.get(source) else None
            rows.append(row);mechanisms.extend(mechanism(p))
            checks.append(dict(method=method,seed=seed,completed=not p['failed'],validation_points=len(curve),
                validation_detail_rows=len(detail),source=protocol.relative(directory)))
    means=protocol.finite_five_means(rows,list(METRICS));pairs=[]
    for control in protocol.ALL[1:]:
        for seed in range(5):
            a=next(r for r in rows if r['method']==protocol.FULL and r['seed']==seed)
            b=next(r for r in rows if r['method']==control and r['seed']==seed)
            pairs.append(dict(method='full_minus_'+control,seed=seed,
                **{key:a[key]-b[key] if a[key] is not None and b[key] is not None else None for key in METRICS}))
    pair_means=protocol.finite_five_means(pairs,list(METRICS),methods=tuple('full_minus_'+m for m in protocol.ALL[1:]))
    for name,data in (('per_seed_metrics',rows),('method_means',means),('paired_differences',pairs),('paired_means',pair_means),
        ('validation_curves',curves),('checkpoint_failures',failures),('phase_mechanisms',mechanisms),('training_costs',costs)):
        write_records_csv(data,output/f'{name}.csv')
    write_json(dict(passed=True,statistical_unit='seed',planned_training_seeds=5,checks=checks,
        new_training_runs=10,old_methods_reused=True,final500_evaluation=protocol.relative(evaluation),
        final500_verification_sha256=common.sha256(evaluation/'verification.json'),
        analysis_source_sha256=common.sha256(Path(__file__)),common_source_sha256=common.sha256(Path(protocol.__file__)),
        failed_training_runs=final['failed_training_runs'],threshold_imputation=False,used_for_tuning=False),output/'verification.json')
    report(output,rows,means,pairs,pair_means);plot(output,curves)
    write_json(dict(analysis=protocol.relative(output)),ROOT/'results/latest_all_eligible_modules_analysis_v1/latest.json')
    print(output,flush=True);return output


def report(output,rows,means,pairs,pair_means):
    def number(value,scale=1):return '缺失/失败' if value is None else f'{scale*value:.2f}'
    lines=['# 最新完整方法：模块消融证据','',
        '只新增去TD裁剪、去A*代选两组，各五seed、200000在线步。最新版完整组、advice_bound、原始D3QN复用，原筛选完整组与DQfD作为背景对照保留。不调整日程、奖励或参数。统计单位seed，±为样本标准差ddof=1。','',
        '原50验证：自主ε=0，无教师、路线引导、动态保护或Q输出裁剪，共同静态合法动作掩码。AULC使用真实环境步数；首次90%不等于连续三点90%，未达到不填20万。D3QN等无教师方法的同时间窗口最低值不解释为撤教因果效应。','',
        '|方法|seed|前5万AULC%|全程AULC%|10–20万AULC%|8–16万最低%（步）|最后五次%|原50最终%|',
        '|---|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:
        lines.append(f'|{r["method"]}|{r["seed"]}|{number(r["aulc_0_50000"],100)}|{number(r["full_aulc"],100)}|{number(r["aulc_100000_200000"],100)}|{number(r["min_80000_160500"],100)} ({r.get("min_step_80000_160500")})|{number(r["last_five_mean"],100)}|{number(r["validation50_final_success"],100)}|')
    lines.extend(['','|方法|前5万AULC%|全程AULC%|10–20万AULC%|独立500成功%|独立500碰撞%|独立500超时%|', '|---|---:|---:|---:|---:|---:|---:|'])
    keys=('aulc_0_50000','full_aulc','aulc_100000_200000','independent500_success','independent500_collision','independent500_timeout')
    for r in means:
        values=[f'{number(r[k+"_mean"],100)}±{number(r[k+"_sample_sd"],100)} (n={r[k+"_observed_seed_count"]}/5)' for k in keys]
        lines.append('|'+r['method']+'|'+'|'.join(values)+'|')
    lines.extend(['','独立500逐seed最终确认（与原50曲线分开）：','',
        '|方法|seed|成功%|动态碰撞%|碰撞次数|等待超时|绕圈超时|其他超时|', '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in rows:
        lines.append(f'|{r["method"]}|{r["seed"]}|{number(r["independent500_success"],100)}|{number(r["independent500_collision"],100)}|{number(r["final500_collision_count"])}|{number(r["final500_waiting_timeout_count"])}|{number(r["final500_looping_timeout_count"])}|{number(r["final500_other_timeout_count"])}|')
    lines.extend(['','|方法|seed|首次90%|首次三点起点/确认|10–20万最低%（步）|低谷后首次90%|低谷后三点起点/确认|', '|---|---:|---|---|---|---|---|'])
    for r in rows:
        recovery=str(r['recovery_first90']) if r['recovery_required'] else '未跌破90%/不适用'
        lines.append(f'|{r["method"]}|{r["seed"]}|{r["first90"]}|{r["first_three90_start"]}/{r["first_three90_confirmed"]}|{number(r["min_100000_200000"],100)} ({r.get("min_step_100000_200000")})|{recovery}|{r["recovery_three90_start"]}/{r["recovery_three90_confirmed"]}|')
    lines.extend(['','完整方法减消融方法的同seed配对差（百分点；正值代表完整组该率/AULC更高，碰撞/超时则越低越好）：','',
        '|对比|seed|前5万|全程|10–20万|8–16万最低|独立500成功|独立500碰撞|', '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in pairs:lines.append('|'+r['method']+'|'+str(r['seed'])+'|'+'|'.join(number(r[k],100) for k in ('aulc_0_50000','full_aulc','aulc_100000_200000','min_80000_160500','independent500_success','independent500_collision'))+'|')
    lines.extend(['','模块作用的跨seed重复（只用预注册对比；不挑seed）：',''])
    for control,label in ((protocol.core.METHODS[0],'TD目标约束'),(protocol.core.METHODS[1],'在线A*动作代选'),('advice_bound','动作监督')):
        selected=[r for r in pairs if r['method']=='full_minus_'+control]
        counts={k:sum(r[k] is not None and r[k]>0 for r in selected) for k in ('aulc_0_50000','full_aulc','aulc_100000_200000','independent500_success')}
        summary=next(r for r in pair_means if r['method']=='full_minus_'+control)
        lines.append(f'{label}：完整组前5万/全程/后半AULC提高{counts["aulc_0_50000"]}/5、{counts["full_aulc"]}/5、{counts["aulc_100000_200000"]}/5 seed；500成功率提高{counts["independent500_success"]}/5；全程配对差{number(summary["full_aulc_mean"],100)}±{number(summary["full_aulc_sample_sd"],100)}个百分点。')
    lines.extend(['',
        '监督使用独立教师动作，TD仍使用真实动作/奖励/下一观测/终止标记；碰撞排除、静态合法与观测风险筛选不变。去代选组监督时钟按真实步数前进，代选率始终为0，10万后监督为0。去TD组完整目标不裁剪，Smooth L1及梯度范数裁剪保留。',
        'phase_mechanisms.csv记录阶段监督覆盖、执行一致率、margin、监督权重、Q范围、原始TD范围、裁剪及实际代选；旧方法未记录有符号Q/原始TD下界的字段保持缺失，不用当前同步目标网络伪造历史训练量。计数按回放抽样次数，非独立转移数。',
        '原筛选完整组与DQfD是背景对照，不能冒充本轮两个模块消融。DQfD保留冻结示范和20000离线更新，虽然在线步数相同，总训练预算不同。',
        '论文只保留数据支持的学习效率与模块组合贡献；若模块增益随seed改变，明确写成条件性收益。500测试只做最终确认，不能据此调参或选检查点；不主张首次融合、保证安全、地图泛化或解决撤教低谷。数值失败seed保留，并将缺失计入n/5，不能把剩余样本宣称五seed均值。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(output,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(22,4),sharey=True)
    for seed,ax in enumerate(axes):
        for method in protocol.MAIN:
            selected=[r for r in curves if r['method']==method and r['seed']==seed]
            ax.plot([int(r['environment_steps_total'])/1000 for r in selected],[100*float(r['safe_success_rate']) for r in selected],label=method)
        ax.axvline(100,color='gray',linestyle='--');ax.set(title=f'seed {seed}',xlabel='Environment steps (k)',ylim=(-2,102));ax.grid(alpha=.2)
    axes[0].set_ylabel('Autonomous success (%)');axes[0].legend(fontsize=5)
    fig.tight_layout();fig.savefig(output/'paired_validation_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':analyze()

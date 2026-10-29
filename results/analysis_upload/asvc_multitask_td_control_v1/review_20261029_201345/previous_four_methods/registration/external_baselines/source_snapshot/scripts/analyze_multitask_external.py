"""Regenerate observed evidence only; missing runs and missing traces stay missing."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from statistics import mean,stdev

import numpy as np
import multitask_external_common as c

LABELS={'C':'C 持续弱监督','D':'D 风险软权重','d3qn':'原始 D3QN','dqfd':'DQfD'}
PHASES=((0,100000),(100000,250000),(250000,500000),(400000,500000))
BEHAVIOR=('wait_fraction','revisit_move_ratio','periodic_movement_ratio','final_static_distance')


def rows(path):
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def number(value):return None if value is None or value=='' else float(value)


def aulc(curve,start,end):
    x=np.array([int(r['environment_steps_total']) for r in curve]);y=np.array([float(r['success_rate']) for r in curve])
    if not len(x) or x[0]>start or x[-1]<end:return None
    inside=x[(x>start)&(x<end)];xx=np.r_[start,inside,end];yy=np.interp(xx,x,y)
    return float(np.trapz(yy,xx)/(end-start))


def aggregate(records,keys,group_keys):
    result=[]
    groups=sorted({tuple(r[k] for k in group_keys) for r in records})
    for group in groups:
        selected=[r for r in records if tuple(r[k] for k in group_keys)==group]
        row=dict(zip(group_keys,group));row['seed_count']=len(selected)
        for key in keys:
            values=[number(r.get(key)) for r in selected];values=[v for v in values if v is not None]
            row[key+'_n']=len(values);row[key+'_mean']=mean(values) if values else None
            row[key+'_sample_sd']=stdev(values) if len(values)>1 else None
        result.append(row)
    return result


def curve_metrics(method,seed):
    directory=c.directory(method,seed);curve=rows(directory/'validation_curve.csv')
    if not curve:return None,[]
    curve=sorted(curve,key=lambda r:int(r['environment_steps_total']))
    steps=[int(r['environment_steps_total']) for r in curve]
    if len(set(steps))!=len(steps) or steps[0]!=0 or any(s%10000 for s in steps):raise ValueError('Invalid validation checkpoint sequence')
    complete=(directory/'completion.json').exists()
    if complete and steps!=list(range(0,500001,10000)):raise ValueError('Completed run lacks frozen validation checkpoints')
    final=curve[-1];record=dict(method=method,seed=seed,complete=complete,last_validation_step=steps[-1],
        final_success_rate=float(final['success_rate']) if complete else None,
        final_collision_rate=float(final['collision_rate']) if complete else None,
        final_timeout_rate=float(final['timeout_rate']) if complete else None,
        latest_success_rate=float(final['success_rate']),aulc_0_500000=aulc(curve,0,500000))
    for start,end in PHASES:record[f'aulc_{start}_{end}']=aulc(curve,start,end)
    late=[float(r['success_rate']) for r in curve if 460000<=int(r['environment_steps_total'])<=500000]
    record['last_five_success_mean']=mean(late) if len(late)==5 else None
    details=rows(directory/'validation_details.csv')
    for checkpoint in curve:
        subset=[r for r in details if int(r['environment_steps_total'])==int(checkpoint['environment_steps_total'])]
        if len(subset)!=50:raise ValueError('Checkpoint does not contain exactly 50 validation tasks')
        for metric in ('success','collision','timeout'):
            if abs(mean(float(r[metric]) for r in subset)-float(checkpoint[metric+'_rate']))>1e-12:raise ValueError('Validation task/summary mismatch')
    return record,[dict(method=method,seed=seed,**r) for r in curve]


def paired_metrics(method,seed):
    records=[];tasks=[]
    for condition in ('dynamic','static'):
        directory=c.RESULTS/'paired'/method/f'seed_{seed}'/condition
        if not (directory/'completion.json').exists():continue
        data=rows(directory/'per_task.csv')
        for band in ('all','short','medium','long'):
            for group in ('all','reference_clear','reference_conflict'):
                part=[r for r in data if (band=='all' or r['band']==band) and (group=='all' or r['reference_group']==group)]
                if not part:continue
                record=dict(method=method,seed=seed,condition=condition,band=band,reference_group=group,tasks=len(part),
                    trace_tasks=sum(r['trace_metrics_available']=='True' for r in part))
                for metric in ('success','collision','timeout'):
                    record[metric+'_rate']=mean(float(r[metric]) for r in part)
                for metric in BEHAVIOR:
                    values=[number(r.get(metric)) for r in part];values=[v for v in values if v is not None]
                    record[metric]=mean(values) if values else None;record[metric+'_n']=len(values)
                records.append(record)
        tasks.extend(dict(method=method,seed=seed,condition=condition,**r) for r in data)
    by={(r['condition'],r['task_id']):r for r in tasks};deltas=[]
    for (condition,task_id),dynamic in by.items():
        if condition!='dynamic' or ('static',task_id) not in by:continue
        static=by['static',task_id]
        row=dict(method=method,seed=seed,task_id=task_id,band=static['band'],reference_group=static['reference_group'])
        for metric in ('success','collision','timeout',*BEHAVIOR):
            x,y=number(static.get(metric)),number(dynamic.get(metric))
            row[metric+'_static_minus_dynamic']=x-y if x is not None and y is not None else None
        row['static_success_dynamic_failure']=int(float(static['success'])==1 and float(dynamic['success'])==0)
        row['static_failure_dynamic_success']=int(float(static['success'])==0 and float(dynamic['success'])==1)
        deltas.append(row)
    changes=[]
    for band in ('all','short','medium','long'):
        part=[r for r in deltas if band=='all' or r['band']==band]
        if not part:continue
        record=dict(method=method,seed=seed,band=band,tasks=len(part))
        for key in part[0]:
            if key.endswith('_static_minus_dynamic'):
                values=[r[key] for r in part if r[key] is not None]
                record[key]=mean(values) if values else None;record[key+'_paired_tasks']=len(values)
        for key in ('static_success_dynamic_failure','static_failure_dynamic_success'):record[key]=sum(r[key] for r in part)
        changes.append(record)
    return records,tasks,changes,deltas


def learning_audit(method,seed):
    path=c.directory(method,seed);logs=rows(path/'update_log.csv');offline=rows(path/'offline_update_log.csv')
    if not logs and not offline:return None,[]
    record=dict(method=method,seed=seed,online_logged_bins=len(logs),offline_logged_bins=len(offline))
    for phase,data in (('online',logs),('offline',offline)):
        if not data:continue
        for key,operation in (('q_min',min),('q_max',max),('q_abs_max',max),('raw_target_min',min),('raw_target_max',max),
            ('raw_one_target_min',min),('raw_one_target_max',max),('raw_n_target_min',min),('raw_n_target_max',max)):
            values=[number(r.get(key)) for r in data];values=[v for v in values if v is not None]
            record[phase+'_'+key]=operation(values) if values else None
        previous=0;total=0.;weight=0
        for row in data:
            updates=int(row.get('bin_updates') or int(row.get('updates',0))-previous)
            previous=int(row.get('updates',previous))
            value=number(row.get('target_clip_fraction',row.get('one_target_clip_fraction')))
            if value is not None:total+=value*updates;weight+=updates
        record[phase+'_target_clip_fraction']=total/weight if weight else None
    risk=[]
    if (path/'supervision_statistics.json').exists():
        stats=c.read(path/'supervision_statistics.json');actual=stats['replay_actual_statistics'];advice=stats['advice_statistics']
        record.update(environment_label_availability=stats['environment_label_availability'],replay_supervision_coverage=stats['replay_supervision_coverage'],
            replay_eligible_draws=actual['eligible_draws'],effective_supervision_coefficient_sum=actual['effective_coefficient_sum'],
            base_coefficient_sum=actual['base_coefficient_sum'],reduced_draws=actual['reduced_draws'],
            reduced_fraction_of_eligible=actual['reduced_draws']/max(1,actual['eligible_draws']),
            effective_to_base_ratio=actual['effective_coefficient_sum']/max(1,actual['base_coefficient_sum']),
            advice_applied_rate=advice['applied']/advice['steps'],advice_requested_rate=advice['requested']/advice['steps'])
        for index,entry in sorted(stats['risk_bins'].items()):
            risk.append(dict(method=method,seed=seed,risk_bin=index,draws=entry['draws'],
                mean_risk=entry['risk_sum']/max(1,entry['draws']),mean_coefficient=entry['coefficient_sum']/max(1,entry['draws']),
                mean_base=entry['base_sum']/max(1,entry['draws']),coefficient_sum=entry['coefficient_sum'],base_sum=entry['base_sum']))
    if (path/'sample_statistics.json').exists():
        stats=c.read(path/'sample_statistics.json');record.update(offline_updates=stats['offline_updates'],online_updates=stats['online_updates'],**stats['counts'])
    return record,risk


def legacy_evidence():
    source=c.ROOT/'results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719'
    names=('summary.csv','method_means.csv','phase_metrics.csv','all_validation_checkpoints.csv','all_value_learning.csv','REPORT.zh-CN.md')
    evidence=dict(source_directory=str(source.relative_to(c.ROOT)),setting='fixed start and goal, 200000 steps, seeds0..4; never mix with current multi-task',
        input_sha256={name:c.sha(source/name) for name in names if (source/name).exists()},new_training=0)
    c.write(c.RESULTS/'legacy_unguided_bound_provenance.json',evidence)
    seeds=[r for r in rows(source/'summary.csv') if r['method']=='unguided_bound']
    c.csv_write(c.RESULTS/'legacy_unguided_bound_seeds.csv',seeds)
    means=[r for r in rows(source/'method_means.csv') if r['method'] in ('unguided_raw','unguided_bound')]
    c.csv_write(c.RESULTS/'legacy_raw_vs_bound.csv',means)
    return means


def plots(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'C':'#237bb5','D':'#cc4e23','d3qn':'#587848','dqfd':'#9366a2'}
    figure,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for method in c.METHODS:
        by_seed=[]
        for seed in (0,1):
            data=[r for r in curves if r['method']==method and r['seed']==seed]
            if not data:continue
            x=np.array([int(r['environment_steps_total']) for r in data]);y=np.array([float(r['success_rate']) for r in data])
            axes[0].plot(x,y,color=colors[method],linestyle='-' if seed==0 else '--',label=f'{method} seed{seed}',alpha=.8)
            by_seed.append(dict(zip(x,y)))
        if not by_seed:continue
        common=sorted(set.intersection(*(set(d) for d in by_seed)))
        values=np.array([[d[x] for x in common] for d in by_seed]);avg=values.mean(0)
        axes[1].plot(common,avg,label=f'{method} (n={len(by_seed)})',color=colors[method])
        if len(by_seed)>1:
            sd=values.std(0,ddof=1);axes[1].fill_between(common,avg-sd,avg+sd,color=colors[method],alpha=.12)
    for axis in axes:axis.set(xlabel='Online environment steps',ylabel='Frozen validation success',ylim=(-.03,1.03));axis.grid(alpha=.2);axis.legend(fontsize=8)
    axes[0].set_title('Every seed, no smoothing');axes[1].set_title('Mean and sample SD across available seeds')
    figure.savefig(c.RESULTS/'learning_curves.png',dpi=150);figure.savefig(c.RESULTS/'learning_curves.svg');plt.close(figure)


def fmt(value,percent=False):
    if value is None:return 'NA'
    return f'{100*float(value):.1f}%' if percent else f'{float(value):.4f}'


def report(metrics,means,paired,changes,audits,legacy):
    lines=['# 多任务外部基线与静态/动态配对诊断',f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。',
        '仅 seed 0、1 预实验；验证池为同一冻结 50 场景，未见组合与未见目标测试池均未评估。D 保持候选方法身份。',
        '## 运行与学习曲线','全程/阶段 AULC 使用原始验证成功率曲线的梯形积分除以区间长度；阶段为 0–10万、10–25万、25–50万及末段40–50万。未覆盖完整区间时记 NA。末五点均值固定为46–50万。DQfD 的0步成绩在离线预训练之后，横轴不含示范和离线成本。',
        '| 方法 | seed | 最后验证步 | 完成 | 最终成功 | 碰撞 | 超时 | 全程AULC | 末五点 |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    by={(r['method'],r['seed']):r for r in metrics}
    for method in c.METHODS:
        for seed in (0,1):
            row=by.get((method,seed))
            if row is None:lines.append(f'| {LABELS[method]} | {seed} | 尚无验证 | 否 | NA | NA | NA | NA | NA |');continue
            lines.append(f'| {LABELS[method]} | {seed} | {row["last_validation_step"]} | {row["complete"]} | {fmt(row["final_success_rate"],True)} | {fmt(row["final_collision_rate"],True)} | {fmt(row["final_timeout_rate"],True)} | {fmt(row["aulc_0_500000"])} | {fmt(row["last_five_success_mean"],True)} |')
    lines+=['','逐 seed 全部阶段见 `run_comparison.csv`；均值和样本标准差见 `method_comparison.csv`。样本标准差按 ddof=1 在 seed 间计算，不能把50个任务当50个独立训练seed。','',
        '| 方法 | 已有曲线seed数 | 最终成功均值 ± 样本SD | 最终碰撞均值 ± 样本SD | 最终超时均值 ± 样本SD |',
        '|---|---:|---:|---:|---:|']
    for row in means:
        values=[f'{fmt(row[k+"_mean"],True)} ± {fmt(row[k+"_sample_sd"],True)}' for k in ('final_success_rate','final_collision_rate','final_timeout_rate')]
        lines.append(f'| {LABELS[row["method"]]} | {row["seed_count"]} | '+ ' | '.join(values)+' |')
    lines+=['','## 配对诊断','循环检测只统计真实位置变化，周期2–12且至少3次重复；等待不直接算移动循环。重复访问比例按已访问的成功移动终点/全部成功移动定义。指标定义、固定参考冲突组和逐任务轨迹见登记与 paired 子目录。',
        '动态成功/碰撞/超时/等待等旧成绩直接复用。旧 C/D 全轨迹不足时，循环、终止距离只对留存轨迹可见，缺失为 NA；这部分子集不得代表全部50任务。只有得到单独授权才补录缺失动态轨迹，且必须逐任务核对原成绩。',
        '| 方法/seed | 静态成功 | 动态成功 | 静态−动态成功率 | 静态循环任务数/50 | 动态完整轨迹 |',
        '|---|---:|---:|---:|---:|---:|']
    conclusions=[]
    for method in c.METHODS:
        for seed in (0,1):
            base=c.RESULTS/'paired'/method/f'seed_{seed}'
            static=rows(base/'static/per_task.csv');dynamic=rows(base/'dynamic/per_task.csv')
            if not static or not dynamic:continue
            ss=mean(float(r['success']) for r in static);ds=mean(float(r['success']) for r in dynamic)
            loops=sum(r['periodic_movement']=='True' for r in static);traces=sum(r['trace_metrics_available']=='True' for r in dynamic)
            lines.append(f'| {method}/{seed} | {fmt(ss,True)} | {fmt(ds,True)} | {fmt(ss-ds,True)} | {loops}/50 | {traces}/50 |')
            failed=sum(not float(r['success']) for r in static)
            if failed:conclusions.append(f'{method} seed{seed} 在静态条件仍失败 {failed}/50，其中出现移动循环的任务 {loops}/50。它直接说明去除动态障碍后仍有导航缺口；尚不能把该缺口归因于某一种监督损失。')
            else:conclusions.append(f'{method} seed{seed} 静态50任务全部成功；动态成功率比静态低 {100*(ss-ds):.1f} 个百分点。该差值反映动态障碍对这一模型和验证集的净影响。')
    lines+=['','逐任务成对变化见 `paired_task_changes.csv`；按seed/长短/统一参考冲突组汇总见 `paired_seed_summary.csv`，跨seed均值/样本SD见 `paired_method_summary.csv`。正的静态−动态成功率表示动态条件较差，但不等于全部动态失败都由障碍独立造成。',
        '代表性轨迹见 `representative_cases.csv` 与各条件的 `offline_cases.csv`：记录是否与教师一致、是否等待、观测风险与真实后续冲突；全部在执行完成后离线计算。选择规则是固定的可解释案例选择，不用于重新筛选总体成绩。','',
        '## 已观察到什么，尚不能解释什么',*conclusions,
        'C 的动态最终成功率为76%/72%，D为74%/64%；D碰撞更少（0%/6% 对 C 的6%/20%），但超时更多（26%/30% 对18%/8%）。这些是结果权衡，不能将碰撞减少直接称为整体避障能力提高。',
        '是否是“监督减少导致循环”仍需要成对循环轨迹与监督受控对照；即使补齐轨迹，也只是相关证据。四方法未全部完成前，不能把整体瓶颈定为监督方式；静态失败说明基础多目标导航仍有缺口，动态额外差值提供动态交互影响的证据。',
        '50万步是否足够应结合末段AULC、末五点和完整基线曲线判断；不能仅据末次检查点自动延长训练或改参。','',
        '## Q值、TD目标与监督审计','`learning_audit.csv` 给出在线/离线Q和原始TD目标极值、按实际更新数加权的裁剪比例。C/D的目标范围约束并不强制Q值本身处于[-6,10]；原始D3QN/DQfD裁剪比例必须为0。DQfD单步与多步目标分别记录。',
        '`risk_weight_audit.csv` 按实际回放合格抽样记录风险、权重和系数总量。环境标签可用率、回放监督覆盖率分母不同；C/D有效系数总量及风险降权比例见 learning_audit，不以环境步数代替监督更新量。']
    for row in audits:
        if 'effective_to_base_ratio' in row:
            lines.append(f'- {row["method"]}/{row["seed"]}：环境标签率 {fmt(row["environment_label_availability"],True)}，回放覆盖 {fmt(row["replay_supervision_coverage"],True)}，有效系数总量 {row["effective_supervision_coefficient_sum"]:.1f}，实际/基础系数 {fmt(row["effective_to_base_ratio"],True)}，合格样本降权比例 {fmt(row["reduced_fraction_of_eligible"],True)}。')
    if (c.REG/'demonstration_report.json').exists():
        demo=c.read(c.REG/'demonstration_report.json')
        lines+=['','## DQfD单列成本',f'固定480次示范尝试，成功保留 {demo["retained_episodes"]} 条完整轨迹、{demo["transitions"]} 转移；尝试消耗 {demo["attempted_environment_steps"]} 环境步。尝试240训练任务、40目标，成功覆盖 {demo["retained_training_tasks"]} 任务、{demo["retained_goals"]} 目标；失败 {demo["failed_episodes"]} 次。没有使用旧固定目标示范、验证或测试任务，没有临时加预算。',
            '该示范覆盖有限，是解释DQfD成绩的明确限制。示范生成成本由两个seed共享并只统计一次；每seed另有20000离线更新+499501在线更新+500000在线环境步，永久示范3472条额外存放，在线回放容量20000。不能宣称DQfD与其他方法具有相同总数据或更新成本。']
    lines+=['','## 旧 unguided_bound 证据与后续方案',
        '旧5-seed、20万步固定起终点结果单独见 `legacy_unguided_bound_seeds.csv`、`legacy_raw_vs_bound.csv` 和SHA来源登记。unguided_bound为D3QN+TD约束，无A*代选或监督；已有证据可直接用于旧设置中的数值稳定性和模块对照，不因重画图训练。']
    for row in legacy:
        lines.append(f'- 旧 {row["method"]}：最终成功均值 {fmt(row["safe_success_rate_mean"],True)} ± {fmt(row["safe_success_rate_sample_sd"],True)}，全程AULC {float(row["safe_success_aulc_mean"]):.4f}。固定任务协议，不能混入本轮多任务表。')
    lines+=['','未来CE/大间隔对照和监督下限敏感性方案见 `docs/MULTITASK_EXTERNAL_PROTOCOL.zh-CN.md`，本轮均未启动。现有C可作为同协议floor=0.1的候选复用结果；floor=0、0.05、0.2无严格可比多任务运行。旧固定任务敏感性只能作为旧设置证据。多任务D3QN+TD约束控制组仍缺，需另行授权，不由本轮4次基线替代。','',
        '## 审稿意见—已有证据—剩余缺口—下一步',
        '| 审稿关注 | 已有证据 | 剩余缺口 | 下一步 |','|---|---|---|---|',
        '| 外部基线公平性 | 同冻结任务/观测/网络/奖励/验证；独立原始D3QN和完整DQfD队列 | 未完成运行的全程结果 | 仅完成本轮4次，不重复C/D |',
        '| 动态避障或基础导航 | 同50任务静态/动态配对，统一参考冲突组 | 旧动态全轨迹不足时循环指标为NA | 只在明确授权后补轨迹；否则保留缺失 |',
        '| 风险软权重贡献 | C/D相同硬筛查/代选；实际回放风险/系数总量 | 仅两个seed，监督总量与策略访问状态均有差异 | 不升级D；保留受控对照方案 |',
        '| CE是否同样有效 | 已准备同教师和同预算CE方案 | 无多任务CE对照 | 另行批准后等额调参，不称完整QDagger |',
        '| 监督下限选择 | 多任务C=0.1；旧固定任务敏感性独立保留 | 多任务0/0.05/0.2 | 单参数最小敏感性，尚不启动 |',
        '| TD约束是否必要 | 旧unguided_raw/bound各5seed可复用 | 新多任务D3QN+TD约束缺失 | 单列缺口，不自动训练 |',
        '| 泛化与显著性 | 两seed冻结验证池预实验 | 未见组合/目标未测试；统计功效有限 | 不宣称正式显著性或未见目标泛化 |','',
        '## 实际命令',
        '在仓库根目录运行：','```powershell',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_multitask_external.ps1 -Action Status',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_multitask_external.ps1 -Action Resume -Workers 2',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_multitask_external.ps1 -Action Pairs',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_multitask_external.ps1 -Action Report',
        '```','Resume需先确认原协调器和训练进程已退出，锁仍会拒绝重复任务。续训加载完整checkpoint_latest.pth（优化器、目标网络、PER或均匀回放、n步待处理队列、各类随机状态和环境），不以model_final权重冒充完整恢复。非有限失败保留，不自动换seed或重跑。',
        '首次启动命令为 Prepare 后 Start -Workers 2；Start拒绝重复协调器登记。CE、参数扫描、五seed和正式消融没有启动入口。']
    rendered=[];in_code=False
    for line in lines:
        if line.startswith('```'):
            if not in_code and rendered and rendered[-1]:rendered.append('')
            rendered.append(line);in_code=not in_code
            if not in_code:rendered.append('')
        elif in_code or line.startswith('|'):rendered.append(line)
        else:
            if rendered and rendered[-1]:rendered.append('')
            rendered.append(line)
    (c.RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(rendered)+'\n',encoding='utf-8')


def main():
    c.RESULTS.mkdir(parents=True,exist_ok=True)
    metrics=[];curves=[];paired=[];tasks=[];changes=[];deltas=[];audits=[];risk=[];cases=[];status=[]
    for method in c.METHODS:
        for seed in (0,1):
            path=c.directory(method,seed);record,curve=curve_metrics(method,seed)
            if record is not None:metrics.append(record);curves.extend(curve)
            p,t,ch,de=paired_metrics(method,seed);paired.extend(p);tasks.extend(t);changes.extend(ch);deltas.extend(de)
            audit,rr=learning_audit(method,seed)
            if audit is not None:audits.append(audit);risk.extend(rr)
            for condition in ('static','dynamic'):
                directory=c.RESULTS/'paired'/method/f'seed_{seed}'/condition
                if not (directory/'representatives.json').exists():continue
                selected=c.read(directory/'representatives.json')['task_ids']
                cases.extend(dict(method=method,seed=seed,condition=condition,**r) for r in rows(directory/'per_task.csv') if r['task_id'] in selected)
            state=c.read(path/'status.json') if (path/'status.json').exists() else dict(running=False,complete=False,step=0)
            status.append(dict(method=method,seed=seed,**state))
    keys=['final_success_rate','final_collision_rate','final_timeout_rate','aulc_0_500000','last_five_success_mean',*[f'aulc_{a}_{b}' for a,b in PHASES]]
    means=aggregate(metrics,keys,['method'])
    pmeans=aggregate(paired,['success_rate','collision_rate','timeout_rate',*BEHAVIOR],['method','condition','band','reference_group'])
    change_keys=[k for r in changes[:1] for k in r if k.endswith('_static_minus_dynamic')]
    cmeans=aggregate(changes,change_keys,['method','band'])
    for name,data in [('run_comparison.csv',metrics),('method_comparison.csv',means),('all_validation_curves.csv',curves),
        ('paired_seed_summary.csv',paired),('paired_method_summary.csv',pmeans),('paired_all_tasks.csv',tasks),
        ('paired_seed_changes.csv',changes),('paired_method_changes.csv',cmeans),('paired_task_changes.csv',deltas),
        ('learning_audit.csv',audits),('risk_weight_audit.csv',risk),('representative_cases.csv',cases)]:c.csv_write(c.RESULTS/name,data)
    legacy=legacy_evidence();report(metrics,means,paired,changes,audits,legacy)
    if curves:plots(curves)
    c.write(c.RESULTS/'comparison_status.json',dict(updated=datetime.now().astimezone().isoformat(timespec='seconds'),runs=status,
        all_eight_complete=all(r.get('complete') for r in status),new_training_runs_allowed=4,test_pool_evaluations=0,
        dynamic_trace_metrics_complete=all(r.get('trace_tasks')==50 for r in paired if r['band']=='all' and r['reference_group']=='all'),
        future_training_started=0,report=str((c.RESULTS/'REPORT.zh-CN.md').relative_to(c.ROOT))))
    print('Updated comparison:',c.RESULTS/'REPORT.zh-CN.md',flush=True)


if __name__=='__main__':main()

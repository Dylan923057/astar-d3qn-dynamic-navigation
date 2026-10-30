"""Paired two-seed fixed-task summaries; incomplete intervals stay missing."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import statistics

import fixed_ce_common as c


def rows(path):
    path=Path(path)
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def interpolate(curve,step,metric):
    xs=[int(r['environment_steps_total']) for r in curve]
    if not xs or step<xs[0] or step>xs[-1]:raise ValueError('Do not extrapolate incomplete learning curves')
    for i,x in enumerate(xs):
        if x==step:return float(curve[i][metric])
        if x>step:
            y0,y1=float(curve[i-1][metric]),float(curve[i][metric])
            return y0+(y1-y0)*(step-xs[i-1])/(x-xs[i-1])
    raise AssertionError('Missing interpolation interval')


def aulc(curve,start,end,metric='safe_success_rate'):
    if not curve or int(curve[0]['environment_steps_total'])>start or int(curve[-1]['environment_steps_total'])<end:return None
    points=[start]+[int(r['environment_steps_total']) for r in curve if start<int(r['environment_steps_total'])<end]+[end]
    values=[interpolate(curve,x,metric) for x in points]
    return sum((b-a)*(y0+y1)/2 for a,b,y0,y1 in zip(points,points[1:],values,values[1:]))/(end-start)


def metrics(curve):
    if not curve:return {}
    late=[r for r in curve if 100000<=int(r['environment_steps_total'])<=200000]
    complete=int(curve[-1]['environment_steps_total'])==200000
    result={f'aulc_{a}_{b}':aulc(curve,a,b) for a,b in ((0,50000),(0,200000),(100000,200000))}
    result.update(last_validation_step=int(curve[-1]['environment_steps_total']),curve_complete=complete,
        late_min_safe_success_rate=min(float(r['safe_success_rate']) for r in late) if complete and late else None,
        late_validation_points=len(late),validation_points=len(curve))
    for name in ('safe_success_rate','collision_rate','timeout_rate'):
        result['final_'+name]=float(curve[-1][name]) if complete else None
    for name in ('collision_rate','timeout_rate'):
        result['late_'+name+'_aulc']=aulc(curve,100000,200000,name)
        result['late_'+name+'_point_mean']=statistics.mean(float(r[name]) for r in late) if complete and late else None
        result['late_'+name+'_point_max']=max(float(r[name]) for r in late) if complete and late else None
    return result


MEASUREMENTS=['aulc_0_50000','aulc_0_200000','aulc_100000_200000','late_min_safe_success_rate',
    'final_safe_success_rate','final_collision_rate','final_timeout_rate',
    'late_collision_rate_aulc','late_timeout_rate_aulc','late_collision_rate_point_mean','late_timeout_rate_point_mean',
    'late_collision_rate_point_max','late_timeout_rate_point_max']


def analyze(plot=True):
    frozen=c.verify_freeze(check_baselines=False)
    records=[];curves={}
    for seed in c.SEEDS:
        for method in ('margin','ce'):
            directory=(c.DATA/'baseline_snapshots'/f'seed_{seed}') if method=='margin' else (c.OUT/f'seed_{seed}')
            curve=rows(directory/'validation_curve.csv');curves[method,seed]=curve
            if curve:
                points=[int(r['environment_steps_total']) for r in curve]
                if points!=sorted(set(points)) or points[-1]>200000:raise ValueError('Invalid validation clock')
                record=dict(method=method,seed=seed,source_reused=(method=='margin'),**metrics(curve))
                if method=='ce':
                    completion=directory/'completion.json'
                    record['run_complete']=completion.exists() and c.read(completion)['complete']
                else:record['run_complete']=True
                records.append(record)
            else:records.append(dict(method=method,seed=seed,source_reused=False,run_complete=False,
                                     curve_complete=False,validation_points=0,last_validation_step=0))
    c.write_csv(records,c.RESULTS/'per_seed_metrics.csv')
    summary=[];paired=[];paired_summary=[]
    for method in ('margin','ce'):
        for metric in MEASUREMENTS:
            values=[r[metric] for r in records if r['method']==method and r.get(metric) is not None]
            summary.append(dict(method=method,metric=metric,available_training_seeds=len(values),
                mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None,
                exploratory_only=True))
    for seed in c.SEEDS:
        left=next(r for r in records if r['method']=='ce' and r['seed']==seed)
        right=next(r for r in records if r['method']=='margin' and r['seed']==seed)
        paired.append(dict(seed=seed,contrast='CE_minus_margin',**{metric:left[metric]-right[metric]
            if left.get(metric) is not None and right.get(metric) is not None else None for metric in MEASUREMENTS}))
    for metric in MEASUREMENTS:
        values=[r[metric] for r in paired if r[metric] is not None]
        paired_summary.append(dict(contrast='CE_minus_margin',metric=metric,available_paired_seeds=len(values),
            mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None))
    c.write_csv(summary,c.RESULTS/'mean_sample_sd.csv');c.write_csv(paired,c.RESULTS/'paired_seed_differences.csv')
    c.write_csv(paired_summary,c.RESULTS/'paired_mean_sample_sd.csv')
    full=all(r['run_complete'] for r in records)
    c.write_json(dict(all_two_ce_runs_complete=full,formal_test_used=False,baseline_rerun=False,
        statistics_unit='training_seed',interpolation='piecewise linear in actual step; no extrapolation',
        late_minimum='actual validation points in [100000,200000]',runs=records),c.RESULTS/'analysis_status.json')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for ax,name in zip(axes,('safe_success_rate','collision_rate','timeout_rate')):
            for (method,seed),curve in curves.items():
                if curve:ax.plot([int(r['environment_steps_total'])/1000 for r in curve],
                    [float(r[name]) for r in curve],linestyle='--' if method=='margin' else '-',
                    color=('#3478b1','#d67b30')[seed],label=f'{method} seed{seed}')
            ax.set(xlabel='Environment steps (thousands)',ylabel=name,xlim=(0,200),ylim=(-.02,1.02));ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Fixed-task same-teacher CE vs reused margin; '+('complete' if full else 'IN PROGRESS'))
        fig.tight_layout();fig.savefig(c.RESULTS/'learning_curves.png',dpi=220);fig.savefig(c.RESULTS/'learning_curves.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for method in ('margin','ce'):
            for seed in c.SEEDS:
                directory=c.DATA/'baseline_snapshots'/f'seed_{seed}' if method=='margin' else c.OUT/f'seed_{seed}'
                log=rows(directory/'value_learning.csv')
                if not log:continue
                rawkey='demo_margin_loss_mean' if method=='margin' else 'supervision_ce_loss_mean'
                for ax,key in zip(axes,('q_abs_max','td_loss_mean',rawkey)):
                    ax.plot([int(r['bin_end_step'])/1000 for r in log],[float(r[key]) for r in log],
                        linestyle='--' if method=='margin' else '-',color=('#3478b1','#d67b30')[seed],label=f'{method} seed{seed}')
        for ax,label in zip(axes,('Q absolute maximum','TD loss','Raw eligible supervision loss')):
            ax.set(xlabel='Environment steps (thousands)',ylabel=label,xlim=(0,200));ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(c.RESULTS/'value_loss_diagnostics.png',dpi=220)
        fig.savefig(c.RESULTS/'value_loss_diagnostics.pdf');plt.close(fig)
    lines=['# 固定起终点：同教师在线 CE 对照','',
        '状态：两次 CE 训练全部完成。' if full else '状态：训练尚未全部完成；空白指标表示对应区间未完成，不能当作零或最终结论。','',
        '原大间隔组 seed0、1 直接复用；CE 从头训练，各 200000 环境步。仅把合格样本大间隔替换为静态合法动作上的 T=1 交叉熵。未使用最终500场景。','',
        '| 方法 | seed | 验证到达步数 | 前5万 AULC | 全程 AULC | 后10万 AULC | 后半最低成功 | 最终成功 | 最终碰撞 | 最终超时 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'未完成' if v is None else f'{100*v:.2f}%'
    for r in records:
        keys=('aulc_0_50000','aulc_0_200000','aulc_100000_200000','late_min_safe_success_rate','final_safe_success_rate','final_collision_rate','final_timeout_rate')
        lines.append(f"| {r['method']} | {r['seed']} | {r['last_validation_step']} | "+' | '.join(fmt(r.get(k)) for k in keys)+' |')
    lines += ['', 'AULC 用实际环境步数作横轴，在 50000/100000 边界线性插值，并按区间长度归一化；不补造原组整数步检查点。后半最低值只取实际验证点。',
        '', '逐 seed、均值/样本 SD、CE−margin 配对差见同目录 CSV。只有两个训练 seed，属于探索性对照；原50场景被反复使用，不是独立测试。原始逐场景成绩和失败轨迹在 outputs/asvc_fixed_ce_v1/seed_*/。',
        '', '训练日志保留 Q 范围、TD、原始 CE 与实际加权 CE、标签覆盖率及代选预算。CE 和 margin 的损失数值尺度不同，不能凭哪一种标量损失较小判定优劣。']
    if full:
        changes={r['metric']:r['mean'] for r in paired_summary}
        lines += ['', f"观测：两 seed 平均全程 AULC 的 CE−margin 差为 {100*changes['aulc_0_200000']:+.2f} 个百分点；后10万差 {100*changes['aulc_100000_200000']:+.2f} 个百分点；最终成功差 {100*changes['final_safe_success_rate']:+.2f} 个百分点。",
            '', '原因假设：任何差异只能先归于本协议中监督损失替换所引起的学习和采样轨迹变化；不能单凭两个 seed 证明普遍优势、收敛性或具体梯度原因。创新性判断：本项是同教师监督形式控制实验，不以采用交叉熵本身宣称创新。']
    (c.RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Paired analysis:', 'complete' if full else 'partial; missing intervals left blank',flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-plots',action='store_true');args=parser.parse_args()
    analyze(not args.no_plots)

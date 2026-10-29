"""Validation-only pilot report; no model evaluation and no training launch."""
from __future__ import annotations

from datetime import datetime
import os
import statistics

import numpy as np
import multitask_pilot_common as c


def read_csv(path):
    import csv
    return list(csv.DictReader(path.open(encoding='utf-8-sig',newline=''))) if path.exists() else []


def integral(rows,start,end):
    x=np.array([int(r['environment_steps_total']) for r in rows]);y=np.array([float(r['success_rate']) for r in rows])
    if not len(x) or x[-1]<end:return None
    xs=np.r_[start,x[(x>start)&(x<end)],end];ys=np.interp(xs,x,y)
    return float(np.sum(np.diff(xs)*(ys[:-1]+ys[1:])/2)/(end-start))


def main():
    cfg=c.config();c.RESULTS.mkdir(parents=True,exist_ok=True)
    curves=[];runs=[];risk_rows=[];pairs=[];curves_by_run={}
    for group,seed in c.RUNS:
        directory=c.run_directory(group,seed)
        status=c.read(directory/'status.json') if (directory/'status.json').exists() else dict(running=False,step=0)
        complete=(directory/'completion.json').exists();failure=c.read(directory/'failure.json') if (directory/'failure.json').exists() else None
        rows=read_csv(directory/'validation_curve.csv');curves_by_run[group,seed]=rows
        curves.extend(dict(group=group,seed=seed,**r) for r in rows)
        row=dict(group=group,seed=seed,status='complete' if complete else 'failed' if failure else 'running' if status.get('running') else 'not_started',
                 environment_steps=status.get('step',0),complete=complete,error=failure['error'] if failure else None)
        if rows:
            last=rows[-1];row.update(last_validation_step=int(last['environment_steps_total']),last_success=float(last['success_rate']),
                last_collision=float(last['collision_rate']),last_timeout=float(last['timeout_rate']),
                early_aulc=integral(rows,0,100000),full_aulc=integral(rows,0,500000),late_aulc=integral(rows,250000,500000))
            late=[float(r['success_rate']) for r in rows if int(r['environment_steps_total'])>=250000]
            row['late_min_success']=min(late) if late else None
            final=[float(r['success_rate']) for r in rows if int(r['environment_steps_total'])>=400000]
            row['final_window_success_sd']=statistics.stdev(final) if len(final)>1 else None
            passing=[int(r['environment_steps_total']) for r in rows if float(r['success_rate'])>=.8]
            row['first_80_percent_step']=min(passing) if passing else None
            row['multitask_descriptive_threshold_met']=bool(complete and float(last['success_rate'])>=.8 and
                all(float(last[f'{band}_success_rate'])>=.7 for band in cfg['tasks']['length_bands']))
            row['last_100k_success_change']=float(last['success_rate'])-float(next((r['success_rate'] for r in rows if int(r['environment_steps_total'])==400000),last['success_rate'])) if complete else None
        runs.append(row)
        if (directory/'supervision_statistics.json').exists():
            stats=c.read(directory/'supervision_statistics.json')
            for band,values in stats['risk_bins'].items():
                risk_rows.append(dict(group=group,seed=seed,risk_bin=band,**values,
                    mean_coefficient=values['coefficient_sum']/max(1,values['draws']),
                    replay_supervision_coverage=stats['replay_supervision_coverage'],
                    environment_label_availability=stats['environment_label_availability']))
    for seed in (0,1):
        left={int(r['environment_steps_total']):r for r in curves_by_run['C',seed]}
        right={int(r['environment_steps_total']):r for r in curves_by_run['D',seed]}
        for step in sorted(left.keys()&right.keys()):
            pairs.append(dict(seed=seed,environment_steps_total=step,
                success_difference=float(right[step]['success_rate'])-float(left[step]['success_rate']),
                collision_difference=float(right[step]['collision_rate'])-float(left[step]['collision_rate']),
                timeout_difference=float(right[step]['timeout_rate'])-float(left[step]['timeout_rate'])))
    c.csv_write(c.RESULTS/'run_results.csv',runs);c.csv_write(c.RESULTS/'validation_curves_all_seeds.csv',curves)
    c.csv_write(c.RESULTS/'paired_D_minus_C_curves.csv',pairs);c.csv_write(c.RESULTS/'supervision_by_risk.csv',risk_rows)
    finished=all(r['complete'] for r in runs)
    conclusions=[]
    if not finished:conclusions.append('四次预实验尚未全部完成；当前曲线仅描述进度，不作完整预算或方法优劣结论。')
    else:
        established=[f"{r['group']}/seed{r['seed']}" for r in runs if r.get('multitask_descriptive_threshold_met')]
        conclusions.append('达到预登记多任务描述性门槛的运行：'+(', '.join(established) or '无')+'。门槛仅针对冻结验证分布。')
        for seed in (0,1):
            a=next(r for r in runs if r['group']=='C' and r['seed']==seed);b=next(r for r in runs if r['group']=='D' and r['seed']==seed)
            conclusions.append(f"seed {seed}: D-C 全程AULC差={b['full_aulc']-a['full_aulc']:.4f}；最终成功率差={b['last_success']-a['last_success']:.4f}；后半程最低成功率差={b['late_min_success']-a['late_min_success']:.4f}。")
        conclusions.append('预算判断同时查看最后10万步变化、分档成功率与波动；仍在上升或任务未建立时，50万步充分性没有得到支持。不得据此改变本轮预算。')
    conclusions.append('两个训练seed属于预实验；未见组合和未见目标测试池未评估，不能声称通过未见目标泛化。')
    report=dict(updated=datetime.now().astimezone().isoformat(timespec='seconds'),complete=finished,
                completed_runs=sum(r['complete'] for r in runs),total_runs=4,runs=runs,conclusions=conclusions,
                test_pool_evaluations=0,formal_ablation_runs=0)
    c.write(c.RESULTS/'REPORT.json',report)
    if curves:
        os.environ.setdefault('MPLCONFIGDIR',str(c.RESULTS/'matplotlib_cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
        for (group,seed),rows in curves_by_run.items():
            if not rows:continue
            xs=[int(r['environment_steps_total']) for r in rows]
            for ax,key in zip(axes,('success_rate','collision_rate','timeout_rate')):
                ax.plot(xs,[float(r[key]) for r in rows],label=f'{group} seed {seed}');ax.set_ylabel(key);ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
        axes[0].legend();axes[-1].set_xlabel('Environment steps');fig.tight_layout();fig.savefig(c.RESULTS/'individual_learning_curves.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        for seed in (0,1):
            selected=[r for r in pairs if r['seed']==seed]
            for ax,key in zip(axes,('success_difference','collision_difference','timeout_difference')):
                ax.plot([r['environment_steps_total'] for r in selected],[r[key] for r in selected],label=f'seed {seed}');ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('D-C '+key);ax.grid(alpha=.2)
        axes[0].legend();axes[-1].set_xlabel('Environment steps');fig.tight_layout();fig.savefig(c.RESULTS/'paired_differences.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
        for (group,seed),rows in curves_by_run.items():
            for ax,band in zip(axes,cfg['tasks']['length_bands']):
                ax.plot([int(r['environment_steps_total']) for r in rows],[float(r[f'{band}_success_rate']) for r in rows],label=f'{group} seed {seed}');ax.set_ylabel(band+' success');ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
        axes[0].legend();axes[-1].set_xlabel('Environment steps');fig.tight_layout();fig.savefig(c.RESULTS/'length_band_learning_curves.png',dpi=160);plt.close(fig)
    print(f'Pilot results: {sum(r["complete"] for r in runs)}/4 complete; test evaluations: 0',flush=True)


if __name__=='__main__':main()

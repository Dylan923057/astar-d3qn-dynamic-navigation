"""Read-only learning-curve review. Never starts or resumes a training process."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('C','D','d3qn','dqfd')
LABELS={'C':'C: persistent supervision','D':'D: risk weighting','d3qn':'Original D3QN','dqfd':'DQfD'}


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    output=ROOT/'results/asvc_multitask_external_v1'/datetime.now().strftime('learning_trend_review_%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    curves={};hashes={};windows=[];recent=[]
    for method in METHODS:
        for seed in (0,1):
            experiment='asvc_multitask_risk_pilot_v1' if method in ('C','D') else 'asvc_multitask_external_v1'
            path=ROOT/'outputs'/experiment/method/f'seed_{seed}/validation_curve.csv'
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
            x=np.array([int(r['environment_steps_total']) for r in rows])
            assert x.tolist()==list(range(0,500001,10000))
            values={name:np.array([float(r[name+'_rate']) for r in rows]) for name in ('success','collision','timeout')}
            assert np.allclose(sum(values.values()),1)
            curves[method,seed]=(x,values)
            def area(name,start,end):
                mask=(x>=start)&(x<=end)
                return float(np.trapz(values[name][mask],x[mask])/(end-start))
            for start in range(0,500000,100000):
                windows.append(dict(method=method,seed=seed,start=start,end=start+100000,
                    **{name:area(name,start,start+100000) for name in values}))
            recent.append(dict(method=method,seed=seed,success_300k_400k=area('success',300000,400000),
                success_400k_500k=area('success',400000,500000),success_400k_450k=area('success',400000,450000),
                success_450k_500k=area('success',450000,500000),
                collision_400k_450k=area('collision',400000,450000),collision_450k_500k=area('collision',450000,500000),
                last_five_success_mean=float(values['success'][-5:].mean()),
                last_five_success_min=float(values['success'][-5:].min()),last_five_success_max=float(values['success'][-5:].max())))
    averages=[]
    for method in METHODS:
        for start in range(0,500000,100000):
            selected=[r for r in windows if r['method']==method and r['start']==start]
            averages.append(dict(method=method,start=start,end=start+100000,seeds=len(selected),
                success_mean=float(np.mean([r['success'] for r in selected])),
                success_sample_sd=float(np.std([r['success'] for r in selected],ddof=1))))
    save_csv(output/'window_metrics_per_seed.csv',windows)
    save_csv(output/'window_metrics_mean_sd.csv',averages)
    save_csv(output/'late_window_metrics.csv',recent)
    for minimum,name in ((0,'learning_trends_full'),(300000,'learning_trends_last_200k')):
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
        for ax,method in zip(axes.flat,METHODS):
            for seed,color in ((0,'#2877b5'),(1,'#d66c27')):
                x,values=curves[method,seed];y=100*values['success'];smooth=np.convolve(y,np.ones(5)/5,mode='valid')
                ax.plot(x/10000,y,color=color,linewidth=1,marker='o',markersize=2,alpha=.4,label=f'seed {seed}: raw')
                ax.plot(x[4:]/10000,smooth,color=color,linewidth=2.2,label=f'seed {seed}: 5-checkpoint trailing mean')
            ax.set_title(LABELS[method]);ax.set(xlim=(minimum/10000,50),ylim=(-2,102))
            ax.grid(alpha=.22);ax.set_xlabel('Online environment steps (x10^4)');ax.set_ylabel('Frozen validation success (%)')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
        fig.suptitle('Two-seed pilot: actual validation scores and visualization-only trailing means\nSame frozen 50 tasks; no confidence intervals or formal convergence tests',fontsize=11)
        fig.savefig(output/f'{name}.png',dpi=150);fig.savefig(output/f'{name}.pdf');plt.close(fig)
    record=dict(created=datetime.now().astimezone().isoformat(timespec='seconds'),input_sha256=hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        windows='Normalized trapezoid area per stated window; not a single checkpoint score.',
        smoothing='Five existing checkpoints, trailing only; raw curves retained. No fitting or extrapolation.',
        same_50_validation_tasks=True,seeds=2,formal_convergence_test=False,new_training_runs=0,
        inference='A rising tail does not prove insufficient budget is the only cause; a noisy flat tail does not prove optimal convergence.')
    (output/'provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(output,flush=True)
    for method in METHODS:
        print(method,'10^5-step window means (%):',[round(100*r['success_mean'],2) for r in averages if r['method']==method])


if __name__=='__main__':main()

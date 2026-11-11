"""Partial or completed reports; never turn missing or failed runs into zero scores."""
from __future__ import annotations
import csv
from collections import defaultdict
from datetime import datetime
import statistics
import numpy as np
import multimap_layout_common as c

LABELS=dict(d3qn='D3QN',astar_prefill_td='A* prefill + TD constraint',dqfd='DQfD',
    qdagger='QDagger (task adaptation)',ce='Online CE',asvc='ASVC',asvc_no_td='ASVC w/o TD constraint',asvc_withdraw='ASVC w/o persistence')
COLORS=dict(d3qn='#66717A',astar_prefill_td='#9A6BAE',dqfd='#DA8B1C',qdagger='#228E80',ce='#4CA1C4',asvc='#C94B45',asvc_no_td='#C87955',asvc_withdraw='#A99083')

def rows(path):
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def auc(curve,start,end):
    x=np.array([float(r['environment_steps_total']) for r in curve]);y=np.array([float(r['safe_success_rate']) for r in curve])
    if not len(x) or x[0]>start or x[-1]<end:return None
    inside=(x>start)&(x<end);xx=np.r_[start,x[inside],end];yy=np.r_[np.interp(start,x,y),y[inside],np.interp(end,x,y)]
    return float(np.trapezoid(yy,xx)/(end-start))

def aggregate(per_seed,groups):
    keys=['final_safe_success_rate','final_dynamic_collision_rate','final_timeout_rate','early_aulc','full_aulc','late_aulc',
        'late_min_success','offline_seconds','online_seconds','evaluation_seconds','inference_latency_ms','online_astar_calls','online_astar_seconds']
    bins=defaultdict(list)
    for r in per_seed:bins[tuple(r[k] for k in groups)].append(r)
    result=[]
    for group,selected in bins.items():
        for metric in keys:
            values=[float(r[metric]) for r in selected if r.get(metric) is not None]
            if values:result.append(dict(zip(groups,group),metric=metric,n_seeds=len(values),mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None))
    return result

def analyze():
    from run_multimap_layout import progress,run_directory
    progress();c.RESULTS.mkdir(parents=True,exist_ok=True)
    jobs=c.read(c.DATA/'queue.json')['jobs'];complete=[];curves=[];failures=[]
    for job in jobs:
        root=run_directory(job)
        for row in rows(root/'validation_curve.csv'):curves.append(dict(**job,**row))
        if (root/'failure.json').exists():failures.append(dict(**job,**c.read(root/'failure.json')))
        if not (root/'completion.json').exists():continue
        value=c.read(root/'result.json');curve=rows(root/'validation_curve.csv');test=value['final500']
        selected=[float(r['safe_success_rate']) for r in curve if float(r['environment_steps_total'])>=100000]
        timing=value['astar_timing']
        complete.append(dict(**job,final_safe_success_rate=test['safe_success_rate'],final_dynamic_collision_rate=test['dynamic_collision_rate'],
            final_timeout_rate=test['timeout_rate'],early_aulc=auc(curve,0,50000),full_aulc=auc(curve,0,200000),late_aulc=auc(curve,100000,200000),
            late_min_success=min(selected),offline_updates=value['offline_updates'],online_updates=value['online_updates'],
            demo_transitions=value['demo_registration']['transitions'] if value['demo_registration'] else 0,
            offline_seconds=value['offline_seconds'],online_seconds=value['online_seconds'],evaluation_seconds=value['evaluation_seconds'],
            inference_latency_ms=test['mean_action_latency_ms'],online_astar_calls=timing.get('astar_lookups',0)+value['online_teacher_queries'],
            online_astar_seconds=timing.get('astar_seconds',0)+value['online_teacher_seconds']))
    c.csv_write(complete,c.RESULTS/'per_seed.csv');c.csv_write(curves,c.RESULTS/'validation_curves.csv')
    stats=aggregate(complete,['map_seed','method']);c.csv_write(stats,c.RESULTS/'per_map_mean_sample_sd.csv')
    # Cross-map macro average is paired by training seed and only published with all three layouts present.
    macro=[]
    for method in c.settings()['methods']:
        for seed in range(5):
            selected=[r for r in complete if r['method']==method and r['train_seed']==seed]
            if len(selected)==3:
                row=dict(method=method,train_seed=seed,maps=3)
                for key,value in selected[0].items():
                    if key not in ('method','map_seed','train_seed') and isinstance(value,(int,float)):row[key]=statistics.mean(float(r[key]) for r in selected)
                macro.append(row)
    c.csv_write(macro,c.RESULTS/'cross_map_macro_per_seed.csv');c.csv_write(aggregate(macro,['method']),c.RESULTS/'cross_map_macro_mean_sample_sd.csv')
    paired=[]
    for r in complete:
        reference=next((s for s in complete if s['map_seed']==r['map_seed'] and s['train_seed']==r['train_seed'] and s['method']=='asvc'),None)
        if reference and r['method']!='asvc':
            for metric in ['final_safe_success_rate','final_dynamic_collision_rate','final_timeout_rate','early_aulc','full_aulc','late_aulc']:
                paired.append(dict(map_seed=r['map_seed'],train_seed=r['train_seed'],comparison='asvc_minus_'+r['method'],metric=metric,difference=reference[metric]-r[metric]))
    c.csv_write(paired,c.RESULTS/'paired_differences.csv')
    paired_stats=[];bins=defaultdict(list)
    for r in paired:bins[(r['map_seed'],r['comparison'],r['metric'])].append(r['difference'])
    for (map_seed,comparison,metric),values in bins.items():paired_stats.append(dict(map_seed=map_seed,comparison=comparison,metric=metric,n_seeds=len(values),mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None))
    c.csv_write(paired_stats,c.RESULTS/'paired_mean_sample_sd.csv');c.write(failures,c.RESULTS/'failures.json')
    plot(curves)
    status=c.read(c.RESULTS/'progress.json');maps=c.read(c.DATA/'maps_frozen.json')
    text=['# ASVC 多地图独立训练补实验','',f'更新时间：{datetime.now().isoformat()}。计划100次，已完成训练及最终500评估 {len(complete)}/100。',
        f'状态：{status["counts"]}。未完成项不计零分，失败 seed 不剔除或替换。','',
        '原图91701的30次主训练及已有500/密度/消融记录保留，来源见 existing_runs.csv 和 data/asvc_multimap_layout_v1/existing_evidence.json；本轮统计单列三张新图。',
        '',f'新图：{[r["map_seed"] for r in maps["maps"]]}；消融图：{maps["ablation_map_seed"]}（仅按冻结几何规则选择）。',
        '', '每次固定200000在线步；验证点沿回合边界每约4000步记录实际步数，最终500测试只用最终模型。AULC用验证成功率的分段线性积分除以区间长度。早期0–50k、全程0–200k、后半100–200k。后半最低值只取实际验证点。',
        '', '表格：per_seed.csv、per_map_mean_sample_sd.csv、paired_differences.csv、paired_mean_sample_sd.csv。跨图宏平均仅在同方法同训练seed三张图全部完成后计算。样本SD的独立单位是训练seed；500个场景不是500次训练。',
        '', '训练/离线预训练/验证与最终测试/断点写入耗时分开。两个训练进程使用同一GPU、各1个CPU线程；这是受控并发吞吐下的计时，不是单进程独占设备性能。A*查询耗时含缓存查询，搜索次数另列；自主推理计时来自评估动作延迟。',
        '', '结论边界：同一地图生成规则下，不同静态布局的独立训练验证；不能解释为未训练地图的零样本迁移。当前缺项尚未完成时不比较完整排名、不预设ASVC更好。']
    if complete:
        text+=['','|map_seed|方法|train_seed|最终安全成功|动态碰撞|超时|全程AULC|','|---|---|---:|---:|---:|---:|---:|']
        for r in complete:text.append(f'|{r["map_seed"]}|{r["method"]}|{r["train_seed"]}|{r["final_safe_success_rate"]:.2%}|{r["final_dynamic_collision_rate"]:.2%}|{r["final_timeout_rate"]:.2%}|{r["full_aulc"]:.4f}|')
    (c.RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(f'REPORT: complete={len(complete)}/100; failures={len(failures)}',flush=True)

def plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not curves:return
    maps=[r['map_seed'] for r in c.read(c.DATA/'maps_frozen.json')['maps']]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax,map_seed in zip(axes,maps):
        for method in c.settings()['methods']:
            selected=[r for r in curves if r['map_seed']==map_seed and r['method']==method]
            for seed in sorted({r['train_seed'] for r in selected}):
                rows_=[r for r in selected if r['train_seed']==seed]
                ax.plot([float(r['environment_steps_total']) for r in rows_],[100*float(r['safe_success_rate']) for r in rows_],c=COLORS[method],alpha=.45,lw=1,label=LABELS[method] if seed==min(r['train_seed'] for r in selected) else None)
        ax.set_title(f'map_seed {map_seed}');ax.set_xlabel('Online environment steps');ax.set_xlim(0,200000);ax.set_ylim(-1,101);ax.grid(alpha=.15)
    axes[0].set_ylabel('Safe success rate (%)');handles={}
    for ax in axes:
        for h,label in zip(*ax.get_legend_handles_labels()):handles[label]=h
    fig.legend(handles.values(),handles.keys(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.13,1,1))
    fig.savefig(c.RESULTS/'learning_curves.png',dpi=250);fig.savefig(c.RESULTS/'learning_curves.pdf');plt.close(fig)

if __name__=='__main__':analyze()

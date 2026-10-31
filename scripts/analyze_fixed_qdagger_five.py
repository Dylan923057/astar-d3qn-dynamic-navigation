"""Five-seed fixed-task learning and final500 confirmation, reported separately."""
from __future__ import annotations
import argparse
from datetime import datetime
import statistics
import fixed_qdagger_five_common as f
from analyze_fixed_ce import metrics,MEASUREMENTS,interpolate
c=f.c
FINAL_KEYS=('safe_success_rate','dynamic_collision_rate','timeout_rate','mean_steps','mean_wait_steps',
    'mean_wait_fraction','pooled_wait_fraction','mean_path_efficiency_success_only','safe_success_weighted_path_efficiency')


def final500_tables(frozen):
    scenes=c.read(f.DATA/'frozen500.json')['scenarios'];records=[];details=[];selected={};pairs=[];scene_pairs=[]
    for method in ('margin','ce','qdagger'):
        for seed in f.ALL_SEEDS:
            if method=='qdagger':directory=f.OUT/'final500'/f'seed_{seed}'
            else:directory=f.ROOT/next(r['directory'] for r in frozen['baseline500'] if r['method']==method and r['seed']==seed)
            if not (directory/'completion.json').exists():continue
            part=c.rows(directory/'scene_details.csv');c.scene_order(part,scenes);selected[method,seed]=part
            records.append(dict(method=method,seed=seed,reused=(method!='qdagger'),**c.summarize(part)))
            details.extend(dict(method=method,seed=seed,**r) for r in part)
    matched=[s for s in f.ALL_SEEDS if all((m,s) in selected for m in ('margin','ce','qdagger'))]
    for seed in matched:
        adapted=next(r for r in records if r['method']=='qdagger' and r['seed']==seed)
        for method in ('margin','ce'):
            reference=next(r for r in records if r['method']==method and r['seed']==seed)
            pairs.append(dict(method='qdagger_minus_'+method,seed=seed,
                **{k:adapted[k]-reference[k] if adapted[k] is not None and reference[k] is not None else None for k in FINAL_KEYS}))
            for a,b in zip(selected[method,seed],selected['qdagger',seed]):
                assert a['scenario_id']==b['scenario_id']
                scene_pairs.append(dict(contrast='qdagger_minus_'+method,seed=seed,scenario_id=a['scenario_id'],
                    delta_safe_success=int(c.flag(b['safe_success']))-int(c.flag(a['safe_success'])),
                    delta_collision=int(c.flag(b['dynamic_collision']))-int(c.flag(a['dynamic_collision'])),
                    delta_timeout=int(b['termination_reason']=='timeout')-int(a['termination_reason']=='timeout'),
                    delta_steps=float(b['steps'])-float(a['steps']),delta_wait_steps=float(b['wait_steps'])-float(a['wait_steps']),
                    delta_success_path_efficiency=float(b['path_efficiency'])-float(a['path_efficiency']) if a['path_efficiency'] and b['path_efficiency'] else None))
    c.csv_write(records,f.RESULTS/'final500_per_seed.csv');c.csv_write(details,f.RESULTS/'final500_scene_details.csv')
    c.csv_write(c.summary_table([r for r in records if r['seed'] in matched],FINAL_KEYS),f.RESULTS/'final500_mean_sample_sd.csv')
    c.csv_write(pairs,f.RESULTS/'final500_paired_per_seed.csv');c.csv_write(c.summary_table(pairs,FINAL_KEYS),f.RESULTS/'final500_paired_mean_sample_sd.csv')
    c.csv_write(scene_pairs,f.RESULTS/'final500_paired_scenes.csv')
    return matched,records


def analyze(plot=True):
    frozen=f.verify(check_protected=False);records=[];curves={};points=[];cost=[]
    for method in f.METHODS:
        for seed in f.ALL_SEEDS:
            directory=f.analysis_directory(method,seed);curve=c.rows(directory/'validation_curve.csv');curves[method,seed]=curve
            complete=(directory/'completion.json').exists() and c.read(directory/'completion.json')['complete']
            if curve:
                xs=[int(r['environment_steps_total']) for r in curve]
                assert xs==sorted(set(xs)) and xs[0]==0 and xs[-1]<=200000
                row=dict(method=method,seed=seed,run_complete=complete,new_training_this_extension=(method=='qdagger' and seed in f.NEW_SEEDS),**metrics(curve))
                points.extend(dict(method=method,seed=seed,**r) for r in curve)
            else:row=dict(method=method,seed=seed,run_complete=False,curve_complete=False,last_validation_step=0,validation_points=0)
            records.append(row)
            if not complete:continue
            result=c.read(directory/'result.json');offline=20000 if method in ('qdagger','dqfd') else 0
            cost.append(dict(method=method,seed=seed,new_training_this_extension=(method=='qdagger' and seed in f.NEW_SEEDS),
                online_environment_steps=200000,offline_updates=offline,online_updates=199501,total_updates=offline+199501,
                demonstration_transitions=7260 if offline else 0,teacher_queries=result.get('teacher_queries'),
                teacher_seconds=result.get('teacher_seconds'),offline_seconds=result.get('offline_seconds',result.get('pretraining_seconds')),
                online_seconds=result.get('online_seconds',result.get('online_training_seconds')),
                validation_seconds=result.get('evaluation_seconds'),cross_run_timing_comparable=False))
    matched=[s for s in f.ALL_SEEDS if all(next(r for r in records if r['method']==m and r['seed']==s).get('run_complete') and
        next(r for r in records if r['method']==m and r['seed']==s).get('curve_complete') for m in f.METHODS)]
    differences=[]
    for seed in matched:
        adapted=next(r for r in records if r['method']=='qdagger' and r['seed']==seed)
        for method in f.METHODS[:-1]:
            baseline=next(r for r in records if r['method']==method and r['seed']==seed)
            differences.append(dict(method='qdagger_minus_'+method,seed=seed,**{k:adapted[k]-baseline[k] for k in MEASUREMENTS}))
    c.csv_write(records,f.RESULTS/'per_seed_metrics.csv');c.csv_write(points,f.RESULTS/'all_validation_points.csv')
    c.csv_write(c.summary_table([r for r in records if r['seed'] in matched],MEASUREMENTS),f.RESULTS/'mean_sample_sd.csv')
    c.csv_write(differences,f.RESULTS/'paired_per_seed.csv');c.csv_write(c.summary_table(differences,MEASUREMENTS),f.RESULTS/'paired_mean_sample_sd.csv')
    c.csv_write(cost,f.RESULTS/'training_cost.csv')
    final_matched,final_records=final500_tables(frozen)
    lines=['# 固定QDagger五seed扩展与最终500确认','',
        f'更新时间{datetime.now().isoformat(timespec="seconds")}。完整学习曲线配对seed={matched}；完整500确认配对seed={final_matched}。',
        '仅新增QDagger seed2、3、4；原seed0、1与其他方法全部复用。每seed20000离线更新＋200000在线步。原两seed报告与500确认前的协议保留，本轮独立登记。',
        '', '## 原50重复验证：逐seed',
        '', '|方法|seed|状态|前5万AULC|全程AULC|后10万AULC|后半最低成功|最终成功|碰撞|超时|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=('aulc_0_50000','aulc_0_200000','aulc_100000_200000','late_min_safe_success_rate','final_safe_success_rate','final_collision_rate','final_timeout_rate')
    for r in records:
        values=['—' if r.get(k) is None else f'{r[k]:.2%}' for k in keys]
        lines.append(f"|{r['method']}|{r['seed']}|{'完成' if r['run_complete'] else '待完成'}|"+'|'.join(values)+'|')
    lines+=['','均值/样本SD仅用所有方法共同完成的相同seed，详见mean_sample_sd.csv；n未满5时不称五seed完成。AULC按真实步数积分、不外推；在线0点是预训练后网络，预训练前50单列。',
        '', '## 冻结500确认：逐seed（与50验证分开）','',
        '|方法|seed|安全成功|动态碰撞|超时|平均步数|平均等待步数|成功路径效率|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in final_records:
        eff='—' if r['mean_path_efficiency_success_only'] is None else f"{r['mean_path_efficiency_success_only']:.4f}"
        lines.append(f"|{r['method']}|{r['seed']}|{r['safe_success_rate']:.2%}|{r['dynamic_collision_rate']:.2%}|{r['timeout_rate']:.2%}|{r['mean_steps']:.3f}|{r['mean_wait_steps']:.3f}|{eff}|")
    if final_matched:
        lines+=['','|500确认（共同完成seed）|安全成功均值±样本SD|动态碰撞|超时|','|---|---:|---:|---:|']
        for method in ('margin','ce','qdagger'):
            selected=[r for r in final_records if r['method']==method and r['seed'] in final_matched];cells=[]
            for key in ('safe_success_rate','dynamic_collision_rate','timeout_rate'):
                values=[r[key] for r in selected];sd=f'{statistics.stdev(values):.2%}' if len(values)>1 else '—'
                cells.append(f'{statistics.mean(values):.2%} ± {sd}')
            lines.append('|'+method+'|'+'|'.join(cells)+'|')
    lines+=['','逐场景、失败及同seed配对差保留，步数/等待含失败；路径效率同时提供成功子集和失败置0的安全成功加权版本。大间隔/CE各2500原确认回合核对后复用；QDagger五最终模型新评2500回合，不选择中间模型。',
        '这是看到原两seed成绩后的授权扩展，训练参数未改，未筛seed；500只作确认，不据此调参。旧两seed模型/日志/源码不变。',
        'QDagger额外预训练和原7260条示范成本单列，不增加示范；λ于120000步归零。完整恢复沿用原学习器和环境/RNG恢复机制。',
        '本任务A*教师、Double-DQN、短预算与退出日程适配，不冒称完整Atari复现、正式显著性或单项机制因果。所有结果属于固定起终点，不混入50万步多任务成绩；没有自动启动其他训练。',
        '', '[配置、扩展边界与命令](../../docs/ASVC_FIXED_QDAGGER_FIVE_SEEDS.zh-CN.md)。']
    f.RESULTS.mkdir(parents=True,exist_ok=True);(f.RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(dict(complete=len(matched)==5 and len(final_matched)==5,completed_training_seed_ids=matched,
        completed_final500_seed_ids=final_matched,final500_qdagger_episodes=500*len(final_matched),
        updated=datetime.now().isoformat()),f.RESULTS/'analysis_status.json')
    if plot:plots(curves,matched)
    return matched,final_matched


def plots(curves,matched):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(margin='#0072b2',ce='#cc79a7',d3qn='#777777',dqfd='#d55e00',qdagger='#009e73')
    fig,axes=plt.subplots(5,3,figsize=(13,16),sharex=True)
    for seed in f.ALL_SEEDS:
        for ax,key in zip(axes[seed],('safe_success_rate','collision_rate','timeout_rate')):
            for method in f.METHODS:
                curve=curves[method,seed]
                if curve:ax.plot([int(r['environment_steps_total'])/1000 for r in curve],[float(r[key]) for r in curve],label=method,color=colors[method])
            ax.set(title=f'seed {seed}: {key}',xlabel='Online steps (thousands)',ylim=(-.02,1.02));ax.grid(alpha=.2)
            ax.axvline(120,color='#999',linestyle=':',linewidth=.8)
    axes[0,0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(f.RESULTS/f'individual_seeds.{ext}',dpi=160)
    plt.close(fig)
    if not matched:return
    grid=list(range(0,200001,10000));fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,('safe_success_rate','collision_rate','timeout_rate')):
        for method in f.METHODS:
            values=[[interpolate(curves[method,s],step,key) for s in matched] for step in grid]
            means=[statistics.mean(v) for v in values];sd=[statistics.stdev(v) if len(v)>1 else 0 for v in values]
            ax.plot([v/1000 for v in grid],means,color=colors[method],label=method)
            ax.fill_between([v/1000 for v in grid],[m-d for m,d in zip(means,sd)],[m+d for m,d in zip(means,sd)],color=colors[method],alpha=.12)
        ax.set(title=key,xlabel='Online steps (thousands)',ylim=(-.02,1.02));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle(f'Fixed task: mean +/- sample SD, n={len(matched)} matched seeds');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(f.RESULTS/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-plot',action='store_true');args=parser.parse_args();analyze(not args.no_plot)

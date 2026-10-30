"""Fixed-task four-method curves and seed-paired descriptive comparisons only."""
from __future__ import annotations
import argparse
import statistics
import fixed_qdagger_common as q
from analyze_fixed_ce import metrics,MEASUREMENTS,interpolate
c=q.c
METHODS=('margin','d3qn','dqfd','qdagger')


def analyze(plot=True):
    frozen=q.verify(check_protected=False);records=[];curves={};points=[];cost=[]
    for method in METHODS:
        for seed in (0,1):
            directory=q.OUT/f'seed_{seed}' if method=='qdagger' else q.DATA/'baseline_snapshots'/method/f'seed_{seed}'
            curve=c.rows(directory/'validation_curve.csv');curves[method,seed]=curve
            if not curve:continue
            xs=[int(r['environment_steps_total']) for r in curve]
            assert xs==sorted(set(xs)) and xs[0]==0 and xs[-1]<=200000
            done=(directory/'completion.json').exists() and c.read(directory/'completion.json')['complete']
            row=dict(method=method,seed=seed,run_complete=done,reused=(method!='qdagger'),**metrics(curve))
            records.append(row);points.extend(dict(method=method,seed=seed,**r) for r in curve)
            if done:
                result=c.read(directory/'result.json')
                cost.append(dict(method=method,seed=seed,online_environment_steps=200000,
                    offline_updates=20000 if method in ('qdagger','dqfd') else 0,
                    online_updates=199501,total_updates=219501 if method in ('qdagger','dqfd') else 199501,
                    demonstration_transitions=7260 if method in ('qdagger','dqfd') else 0,
                    original_shared_demonstration_seconds=frozen['checks']['demonstration_transitions'] and c.read(q.DATA/'demonstration_audit.json')['original_demo_generation_seconds'] if method in ('qdagger','dqfd') else 0,
                    teacher_queries=result.get('teacher_queries'),teacher_seconds=result.get('teacher_seconds'),
                    offline_seconds=result.get('offline_seconds',result.get('pretraining_seconds')),
                    online_seconds=result.get('online_seconds',result.get('online_training_seconds')),
                    evaluation_seconds=result.get('evaluation_seconds'),timing_cross_run_comparable=False))
    complete_seeds=[s for s in (0,1) if all(any(r['method']==m and r['seed']==s and r['run_complete'] and r['curve_complete'] for r in records) for m in METHODS)]
    completed=[r for r in records if r['seed'] in complete_seeds]
    stats=c.summary_table(completed,MEASUREMENTS);differences=[]
    for seed in complete_seeds:
        adapted=next(r for r in records if r['method']=='qdagger' and r['seed']==seed)
        for method in METHODS[:-1]:
            baseline=next(r for r in records if r['method']==method and r['seed']==seed)
            differences.append(dict(method='qdagger_minus_'+method,seed=seed,**{k:adapted[k]-baseline[k] for k in MEASUREMENTS}))
    c.csv_write(records,q.RESULTS/'per_seed_metrics.csv');c.csv_write(stats,q.RESULTS/'mean_sample_sd.csv')
    c.csv_write(points,q.RESULTS/'all_validation_points.csv');c.csv_write(differences,q.RESULTS/'paired_per_seed.csv')
    c.csv_write(c.summary_table(differences,MEASUREMENTS),q.RESULTS/'paired_mean_sample_sd.csv')
    c.csv_write(cost,q.RESULTS/'training_cost.csv')
    lines=['# 固定起终点QDagger任务适配版：两seed预实验','',f'四方法共同完成的seed：{complete_seeds}，计划seed0、1。',
        '复用当前持续弱监督大间隔、原始D3QN、DQfD固定任务200000步模型与全部原50曲线。本轮只新增QDagger两个seed。尚未完整覆盖的AULC不外推。',
        '', '|方法|seed|状态|前5万AULC|全程AULC|后10万AULC|后半最低成功|最终成功|碰撞|超时|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=('aulc_0_50000','aulc_0_200000','aulc_100000_200000','late_min_safe_success_rate','final_safe_success_rate','final_collision_rate','final_timeout_rate')
    for row in records:
        values=['—' if row.get(k) is None else f'{row[k]:.2%}' for k in keys]
        lines.append(f"|{row['method']}|{row['seed']}|{'完成' if row['run_complete'] else '运行中'}|"+'|'.join(values)+'|')
    lines+=['','均值±样本SD与同seed配对差分别保存在mean_sample_sd.csv和paired_*；仅用四方法共同完成的seed计算汇总。',
        '0步采用20000次预训练后的网络；预训练前50场景单独保存。AULC按真实环境步线性积分，在线回合边界验证的偏移不伪装成整数万步。',
        'QDagger额外20000离线更新/seed，原DQfD同样20000；7260示范生成10756环境步、107/128成功回合与原生成耗时单列，未重新增加示范。每个在线步开始500更新，199501在线更新、共219501更新。',
        '查询只服务蒸馏。确定性教师每个采集状态只标注一次，实际查询与回放蒸馏样本使用量分别记录；120000步起监督严格为0。Q和完整多步目标均不裁剪。',
        '这是A*规则教师、Double-DQN骨干、短预算和固定退出日程的QDagger任务适配，不是原Atari完整复现。与主方法的差值包含预训练、教师标签/筛查、多步TD、代选、TD约束及退出方式的组合差异，不能解释为单项因果。',
        '两seed仅作预实验；50池反复验证，不宣称正式显著性或跨地图泛化。QDagger不使用最终500场景选参数，CE500确认另见../asvc_fixed_ce_final500_v1/REPORT.zh-CN.md。',
        '当前观察只描述成功/碰撞/超时与价值日志，不仅凭低成功归因于Q爆炸，不自动加裁剪、换seed或调整日程。']
    if len(complete_seeds)==2:
        for method in METHODS:
            selected=[r for r in records if r['method']==method]
            lines.append(f"\n{method}：最终安全成功均值{statistics.mean(r['final_safe_success_rate'] for r in selected):.2%}，全程AULC均值{statistics.mean(r['aulc_0_200000'] for r in selected):.2%}。")
    q.RESULTS.mkdir(parents=True,exist_ok=True)
    (q.RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors=dict(margin='#0072b2',d3qn='#777777',dqfd='#d55e00',qdagger='#009e73')
        fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
        for seed in (0,1):
            for ax,key in zip(axes[seed],('safe_success_rate','collision_rate','timeout_rate')):
                for method in METHODS:
                    curve=curves[method,seed]
                    if curve:ax.plot([int(r['environment_steps_total'])/1000 for r in curve],[float(r[key]) for r in curve],label=method,color=colors[method])
                ax.set(title=f'seed {seed}: {key}',xlabel='Online steps (thousands)',ylim=(-.02,1.02));ax.grid(alpha=.2)
                ax.axvline(120,color='#999',linestyle=':',linewidth=.8)
        axes[0,0].legend(fontsize=8);fig.tight_layout()
        for extension in ('png','pdf'):fig.savefig(q.RESULTS/f'learning_curves.{extension}',dpi=180)
        plt.close(fig)
    return complete_seeds


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-plot',action='store_true');args=parser.parse_args();analyze(not args.no_plot)

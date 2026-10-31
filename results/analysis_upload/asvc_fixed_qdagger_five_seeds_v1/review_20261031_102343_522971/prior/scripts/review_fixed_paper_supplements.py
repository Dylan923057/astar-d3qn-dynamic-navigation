"""Read-only postprocessing of CE500 confirmation and the fixed QDagger pilot."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import statistics
import fixed_supplement_common as c
import fixed_qdagger_common as q
import run_fixed_ce_final500 as final

RESULTS=c.ROOT/'results/asvc_fixed_paper_supplements_v1'


def mean_sd(values):
    return statistics.mean(values),statistics.stdev(values) if len(values)>1 else None


def make_report():
    RESULTS.mkdir(parents=True,exist_ok=True)
    final.verify();q.verify()
    ce=c.rows(final.RESULTS/'per_seed.csv')
    assert len(ce)==10
    ce_stats=c.rows(final.RESULTS/'mean_sample_sd.csv');pair_stats=c.rows(final.RESULTS/'paired_mean_sample_sd.csv')
    lookup={(r['method'],r['metric']):r for r in ce_stats}
    pair_lookup={r['metric']:r for r in pair_stats}
    lines=['# 固定起终点小论文补实验：CE最终确认与QDagger任务适配',
        '',f'本报告更新时间：{datetime.now().isoformat(timespec="seconds")}。所有结果属于固定起终点200000在线步协议，不包含50万步多任务研究。',
        '', '## CE最终500确认：已完成',
        '', 'CE seed0–4共新增2500回合；当前大间隔组依data/asvc_current_method.json核对并复用原2500回合。最终200000步模型、场景SHA/顺序、epsilon=0、静态掩码、无教师/保护/更新均已核对。未选择最佳检查点或据500成绩调整参数。',
        '', '|方法|seed0成功|seed1|seed2|seed3|seed4|', '|---|---:|---:|---:|---:|---:|']
    for method in ('margin','ce'):
        selected=sorted((r for r in ce if r['method']==method),key=lambda r:int(r['seed']))
        lines.append('|'+method+'|'+'|'.join(f"{float(r['safe_success_rate']):.2%}" for r in selected)+'|')
    lines+=['','|500确认指标|大间隔均值±样本SD|CE均值±样本SD|CE−大间隔配对均值±样本SD|',
        '|---|---:|---:|---:|']
    names=dict(safe_success_rate='安全成功率',dynamic_collision_rate='动态碰撞率',timeout_rate='超时率',
        mean_steps='每回合步数（含失败）',mean_wait_steps='每回合等待步数（含失败）',
        mean_wait_fraction='回合等待比例的均值',mean_path_efficiency_success_only='成功路径效率',
        safe_success_weighted_path_efficiency='安全成功加权路径效率（失败置0）')
    for key,label in names.items():
        cells=[]
        for method in ('margin','ce'):
            row=lookup[method,key];scale=100 if 'rate' in key or 'fraction' in key else 1
            cells.append(f"{scale*float(row['mean']):.4f} ± {scale*float(row['sample_sd']):.4f}"+('%' if scale==100 else ''))
        row=pair_lookup[key];scale=100 if 'rate' in key or 'fraction' in key else 1
        cells.append(f"{scale*float(row['mean']):+.4f} ± {scale*float(row['sample_sd']):.4f}"+(' 个百分点' if scale==100 else ''))
        lines.append('|'+label+'|'+'|'.join(cells)+'|')
    validation=c.rows(final.RESULTS/'final50_validation_separate.csv')
    lines+=['','## 最终50验证：与500确认分开',
        '', '|方法|最终安全成功均值±样本SD|碰撞|超时|','|---|---:|---:|---:|']
    for method in ('margin','ce'):
        selected=[r for r in validation if r['method']==method];cells=[]
        for key in ('safe_success_rate','collision_rate','timeout_rate'):
            mean,sd=mean_sd([float(r[key]) for r in selected]);cells.append(f'{mean:.2%} ± {sd:.2%}')
        lines.append('|'+method+'|'+'|'.join(cells)+'|')
    lines+=['','直接观察：CE在500确认中有seed0–2高于大间隔、seed3–4低于大间隔的配对差；平均安全成功低0.96个百分点，配对差样本SD约1.99个百分点。CE seed4的8次超时单列在全部失败轨迹中。',
        '不能据这些成绩断言某一种监督损失普遍占优，也不能把失败原因直接归结为损失尺度或梯度冲突。成功子集路径效率容易受成功集合变化影响，另报失败置0的安全成功加权效率。',
        '统计单位为训练seed（n=5），50/500均是当前固定地图与动态生成分布；500是冻结确认集，不是未见地图泛化或无限制独立外部测试。',
        '', '## QDagger任务适配：状态与完整成本',
        '', '[机制—实现—适配差异与命令](../../docs/ASVC_FIXED_QDAGGER.zh-CN.md)。原文、官方配置/源码按提交固定并保留许可证，代码与完整配置在data/asvc_fixed_qdagger_v1冻结。']
    queue=q.RESULTS/'queue_status.json';status=c.read(queue) if queue.exists() else {}
    lines.append('\n当前队列：'+('两seed完成' if status.get('complete') else f"尚未完成；active_seed={status.get('active_seed','未启动')}。"))
    for seed in (0,1):
        path=q.OUT/f'seed_{seed}'/'status.json'
        if path.exists():
            s=c.read(path)
            lines.append(f"\nseed{seed}：phase={s.get('phase',s.get('method','未知'))}，在线步={s.get('online_environment_steps',s.get('environment_steps',0))}，离线更新={s.get('offline_updates',0)}，在线更新={s.get('online_updates',0)}。")
    report=q.RESULTS/'REPORT.zh-CN.md'
    if report.exists():
        lines+=['','[四方法逐seed、均值/样本SD及曲线](../asvc_fixed_qdagger_v1/REPORT.zh-CN.md)。数据只比较同seed0、1，原主方法/D3QN/DQfD直接复用；未完成时不外推AULC。']
    if status.get('complete'):
        metrics=c.rows(q.RESULTS/'per_seed_metrics.csv')
        lines+=['','|方法（仅seed0/1）|前5万AULC均值±SD|全程AULC|后10万AULC|后半最低成功|最终成功|最终碰撞|最终超时|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        keys=('aulc_0_50000','aulc_0_200000','aulc_100000_200000','late_min_safe_success_rate',
            'final_safe_success_rate','final_collision_rate','final_timeout_rate')
        for method in ('margin','d3qn','dqfd','qdagger'):
            selected=[r for r in metrics if r['method']==method];assert len(selected)==2
            values=[]
            for key in keys:
                mean,sd=mean_sd([float(r[key]) for r in selected]);values.append(f'{mean:.2%} ± {sd:.2%}')
            lines.append('|'+method+'|'+'|'.join(values)+'|')
        adapted=[r for r in metrics if r['method']=='qdagger']
        baseline=[r for r in metrics if r['method']=='margin']
        delta=statistics.mean(float(r['final_safe_success_rate']) for r in adapted)-statistics.mean(float(r['final_safe_success_rate']) for r in baseline)
        lines.append(f'\n直接观察：QDagger适配版相对当前大间隔组的最终50验证安全成功均值差为{100*delta:+.2f}个百分点。具体逐seed差和AULC见配对表；这是组合协议对比，原因不能只归于蒸馏形式。')
    lines+=['','每seed：冻结7260示范、20000离线更新、200000在线步/199501在线更新；原示范生成10756环境步、107/128成功回合，未新增示范。预训练前与预训练后在线0步分别保存；曲线0点使用预训练后模型。',
        '教师仅用于蒸馏，不代选；学生在线回放不永久混入示范。λ(t)=3max(1−t/120000,0)，T=0.1，无T²附加系数、无TD/Q裁剪、无持续下限、无PER或DQfD额外损失。',
        'QDagger不参与500场景评估或调参。两seed只作任务适配预实验；与主方法的差异包含预训练、三步TD、教师标签、代选、TD约束和退出机制，不能当作单项监督因果对照。',
        '', '## 证据入口',
        '', '- [CE逐seed](../asvc_fixed_ce_final500_v1/per_seed.csv)、[均值样本SD](../asvc_fixed_ce_final500_v1/mean_sample_sd.csv)、[配对差](../asvc_fixed_ce_final500_v1/paired_per_seed.csv)、[逐场景配对](../asvc_fixed_ce_final500_v1/paired_scene_differences.csv)。',
        '- CE新失败轨迹：outputs/asvc_fixed_ce_final500_v1/seed_*/failure_trajectories.json；原组失败轨迹按登记保留在旧目录，未重跑。',
        '- QDagger逐次验证与失败：outputs/asvc_fixed_qdagger_v1/seed_*；完整恢复必须使用recovery_latest.pth，不能把加载model_final.pth称为续训。',
        '- 预检：results/asvc_fixed_ce_final500_v1/preflight.json、results/asvc_fixed_qdagger_v1/preflight.json；所有未完成/失败预检保留在preflight_attempts，不计为正式训练。',
        '', '```powershell',
        'python scripts/run_fixed_ce_final500.py --report',
        'python scripts/analyze_fixed_qdagger.py',
        'python scripts/review_fixed_paper_supplements.py',
        '# 仅在中断后、确认原进程退出时完整续跑；完成seed直接复用',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_fixed_qdagger.ps1 -Resume',
        '```',
        '', '本轮没有更改当前主方法登记，也未启动额外seed、CE重训、参数扫描、消融或多任务实验。']
    (RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(dict(created=datetime.now().isoformat(),ce500_complete=True,new_ce_episodes=2500,reused_margin_episodes=2500,
        qdagger_complete=status.get('complete',False),source_sha256=c.sha(Path(__file__)),
        ce_freeze_sha256=c.sha(final.DATA/'freeze.json'),qdagger_freeze_sha256=c.sha(q.DATA/'freeze.json'),
        no_additional_training=True),RESULTS/'review.json')


def plot_ce():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=c.rows(final.RESULTS/'per_seed.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key in zip(axes,('safe_success_rate','dynamic_collision_rate','timeout_rate')):
        for seed in range(5):
            values=[100*float(next(r for r in records if r['method']==m and int(r['seed'])==seed)[key]) for m in ('margin','ce')]
            ax.plot((0,1),values,'o-',alpha=.65,label=f'seed {seed}')
        ax.set(xticks=(0,1),xticklabels=('Margin','CE'),ylabel=key+' (%)',xlim=(-.25,1.25));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Frozen 500-scene confirmation: paired final 200k models; n=5 seeds');fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(RESULTS/f'ce_final500_paired.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');args=parser.parse_args()
    make_report()
    if args.plot:plot_ce()

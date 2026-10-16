"""Summarize the read-only withdrawal audit; never trains or reads final-test scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analyze_runtime_path_pilot import load_csv
from astar_d3qn.utils.io import write_records_csv


def summarize(directory):
    behavior=load_csv(directory/'checkpoint_behavior.csv')
    ranking=load_csv(directory/'same_state_rankings.csv')
    changes=load_csv(directory/'same_state_changes.csv')
    phases=load_csv(directory/'training_phases.csv')
    means=[]
    for table,rows,key,metrics in (
        ('behavior',behavior,'anchor',('success','waiting_timeout_count','looping_timeout_count','collision_count')),
        ('ranking',ranking,'anchor',('teacher_agreement_clear','teacher_advantage_clear_mean','wait_selection_rate')),
        ('phase',phases,'left',('sampled_supervised_fraction','margin_loss_mean','td_loss_mean','teacher_execution_rate'))):
        for value in sorted({int(r[key]) for r in rows}):
            subset=[r for r in rows if int(r[key])==value]
            for metric in metrics:
                values=[float(r[metric]) for r in subset if r[metric]!='']
                means.append(dict(table=table,anchor=value,metric=metric,seed_count=len(values),
                    mean=float(np.mean(values)) if values else None,
                    sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    write_records_csv(means,directory/'seed_mean_sample_sd.csv')
    def fmt(value,percent=False):
        return f'{float(value)*100:.1f}%' if percent else f'{float(value):.3f}'
    text=['撤教低谷诊断（已有五seed；未使用独立500场景）','',
        '逐seed：相同失败观测上，9万步→10万步的动作排名变化。',
        'seed | 9万成功率→10万成功率 | 原选A*后改选其他（风险筛选通过状态） | A*相对最佳其他动作Q差（9万→10万）']
    for seed in sorted({int(r['seed']) for r in behavior}):
        def at(rows,anchor): return next(r for r in rows if int(r['seed'])==seed and int(r['anchor'])==anchor)
        b9,b10=at(behavior,90000),at(behavior,100000)
        r9,r10=at(ranking,90000),at(ranking,100000)
        c=next(r for r in changes if int(r['seed'])==seed and int(r['before'])==90000)
        text.append(f'{seed} | {fmt(b9["success"],True)}→{fmt(b10["success"],True)} | '
            f'{fmt(c["clear_teacher_to_other"],True)} | '
            f'{fmt(r9["teacher_advantage_clear_mean"])}→{fmt(r10["teacher_advantage_clear_mean"])}')
    b10=[r for r in behavior if int(r['anchor'])==100000]
    waits=sum(int(r['waiting_timeout_count']) for r in b10)
    loops=sum(int(r['looping_timeout_count']) for r in b10)
    collisions=sum(int(r['collision_count']) for r in b10)
    text.extend(['',f'10万附近250个验证回合：等待超时{waits}，绕圈超时{loops}，碰撞{collisions}。这些回合不是250个独立seed。',
        '6–8万监督回放抽样标签占比65.1%–65.6%；8–10万为66.1%–66.4%。',
        '未加权margin损失从约0.004升至约0.19；同期监督权重均值从0.30降至0.10，TD损失下降。',
        ' sampled Q绝对值最大约11–13，未观察到DDQNA式Q爆炸。',
        '', '有证据：同一观测上原有动作排名发生退化，主要表现为等待和绕圈，而非大量碰撞。',
        '不支持：把低谷直接归因于没有监督标签、评估仍带教师、或Q值爆炸。',
        '仍未确认：监督强制形成的动作间隔是否在TD校准时被压缩、探索/回放分布改变各占多少作用。',
        '监督margin=0.8，而TD自然动作差可能更小，是待检验解释；不能仅由相关性证明梯度冲突或遗忘。',
        '', '证据边界：',
        '同状态集合是各模型失败轨迹尾段的固定抽样，不代表全部导航状态；A*一致不等于动态安全。',
        '历史回放、优化器及目标网络快照未保存，无法恢复真实TD误差或关键状态的历史训练覆盖。',
        '撤教后label_samples=0是停止计算标签，不是经验中没有符合规则的标签。',
        '', '下一步预注册单seed0：',
        '从头200000步，与原完整组合配对初始化。6万步以前代选完全一致；6–8万从p=0.32线性降至0。',
        '监督仍为1-t/100000，10万步归零，网络/奖励/回放/风险筛选/标签规则/TD约束/epsilon全部保持原样。',
        '只改变代选日程；代选概率日程面积减少8%，不是教师用量严格相等的实验。',
        '目的：检验8–10万步先自主采集、仍有监督，是否缓解动作排名退化；尚未证明能解决低谷。',
        '同时检查8万及10万两个退出窗口、8–16万整体最低点、10–20万AULC及最终模型，避免只把低谷前移。',
        '不改奖励，不新增监督标签定义，不保留精选回放，不通过独立500场景调参。',
        '', '运行命令：', 'python scripts/run_staged_withdrawal_pilot.py',
        'python scripts/run_staged_withdrawal_pilot.py --train',
        'python scripts/run_staged_withdrawal_pilot.py --analyze'])
    (directory/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for seed in sorted({int(r['seed']) for r in behavior}):
        b=sorted([r for r in behavior if int(r['seed'])==seed],key=lambda r:int(r['anchor']))
        q=sorted([r for r in ranking if int(r['seed'])==seed],key=lambda r:int(r['anchor']))
        axes[0].plot([int(r['checkpoint'])/1000 for r in b],[float(r['success'])*100 for r in b],marker='o',label=f'seed {seed}')
        axes[1].plot([int(r['checkpoint'])/1000 for r in q],[float(r['teacher_advantage_clear_mean']) for r in q],marker='o')
    axes[0].set(ylabel='Autonomous success (%)',ylim=(0,105),title='Original 50 validation scenes')
    axes[1].set(ylabel='Q(A*) - max Q(other legal actions)',title='Same selected failure observations')
    axes[1].axhline(0,color='gray',linestyle=':')
    for ax in axes:
        ax.axvline(100,color='gray',linestyle='--'); ax.set_xlabel('Environment steps (thousands)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(directory/f'withdrawal_same_state.{suffix}',dpi=160)
    plt.close(fig)
    return directory/'REPORT.txt'


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    args=parser.parse_args(); print(summarize(args.directory))


if __name__=='__main__': main()

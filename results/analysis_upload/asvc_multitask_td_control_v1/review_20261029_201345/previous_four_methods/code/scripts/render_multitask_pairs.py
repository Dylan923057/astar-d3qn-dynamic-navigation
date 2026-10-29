"""Offline case figures and readable diagnostic supplement; no environment actions."""
from __future__ import annotations

import csv
from datetime import datetime
from statistics import mean,stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import multitask_external_common as c


def rows(path):
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    cfg,data,planner=c.data();table=[];cases=[];input_hashes={}
    grid=np.zeros((data['size'],data['size']))
    for x,y in data['obstacles']:grid[x,y]=1
    tasks={t['task_id']:t for t in data['pools']['validation']}
    for method in c.METHODS:
        for seed in (0,1):
            base=c.RESULTS/'paired'/method/f'seed_{seed}'
            static=rows(base/'static/per_task.csv');dynamic=rows(base/'dynamic/per_task.csv')
            if not static or not dynamic:continue
            input_hashes[str((base/'static/per_task.csv').relative_to(c.ROOT))]=c.sha(base/'static/per_task.csv')
            input_hashes[str((base/'dynamic/per_task.csv').relative_to(c.ROOT))]=c.sha(base/'dynamic/per_task.csv')
            row=dict(method=method,seed=seed,static_success=mean(float(r['success']) for r in static),
                dynamic_success=mean(float(r['success']) for r in dynamic),
                static_cycle_tasks=sum(r['periodic_movement']=='True' for r in static),
                static_failed_cycle_tasks=sum(r['periodic_movement']=='True' and r['success']=='0' for r in static),
                static_wait_dominated_tasks=sum(float(r['wait_fraction'])>.5 for r in static),
                dynamic_trace_tasks=sum(r['trace_metrics_available']=='True' for r in dynamic))
            row['static_minus_dynamic_success']=row['static_success']-row['dynamic_success'];table.append(row)
            for condition in ('static','dynamic'):
                directory=base/condition
                selected=c.read(directory/'representatives.json')['task_ids']
                details=rows(directory/'per_task.csv');traces={r['task_id']:r for r in c.read(directory/'trajectories.json')}
                for record in details:
                    if record['task_id'] in selected:cases.append(dict(method=method,seed=seed,condition=condition,record=record,trace=traces[record['task_id']]))
    c.csv_write(c.RESULTS/'diagnostic_observations.csv',table)
    if cases:
        columns=3;count=len(cases);figure,axes=plt.subplots((count+columns-1)//columns,columns,figsize=(13,3.8*((count+columns-1)//columns)))
        for axis,case in zip(np.ravel(axes),cases):
            row=case['record'];task=tasks[row['task_id']];path=np.array(case['trace']['positions']);teacher=np.array(planner.path(task['start'],task['goal']))
            axis.imshow(grid,cmap='Greys',vmin=0,vmax=1,origin='upper',interpolation='nearest')
            axis.plot(teacher[:,1],teacher[:,0],'--',color='#999999',linewidth=.9,label='Static reference (offline)')
            color='#25945b' if row['success']=='1' else '#c54c38'
            axis.plot(path[:,1],path[:,0],color=color,linewidth=1.5,alpha=.8,label='Executed trace')
            axis.scatter([task['start'][1]],[task['start'][0]],color='#2079b6',marker='o',s=28,zorder=5)
            axis.scatter([task['goal'][1]],[task['goal'][0]],color='#e1ae29',marker='*',s=90,zorder=5)
            axis.scatter(path[-1,1],path[-1,0],color=color,marker='x',s=40,zorder=6)
            all_points=np.r_[teacher,path];low=np.maximum(0,all_points.min(0)-3);high=np.minimum(data['size']-1,all_points.max(0)+3)
            axis.set_xlim(low[1]-.5,high[1]+.5);axis.set_ylim(high[0]+.5,low[0]-.5)
            outcome='success' if row['success']=='1' else 'collision' if row['collision']=='1' else 'timeout'
            axis.set_title(f'{case["method"]}/{case["seed"]} {case["condition"]} {row["task_id"]}\n{outcome}; moves={row["successful_moves"]}; cycle={float(row["periodic_movement_ratio"]):.2f}; dist={row["final_static_distance"]}',fontsize=9)
            axis.tick_params(labelsize=7)
        for axis in np.ravel(axes)[len(cases):]:axis.axis('off')
        figure.suptitle('Saved executed trajectories; reference paths drawn only offline\nDynamic obstacles omitted from static background; dynamics remain in saved offline logs',fontsize=12)
        figure.tight_layout(rect=(0,0,1,.97));figure.savefig(c.RESULTS/'representative_trajectories.png',dpi=130);plt.close(figure)
    lines=['# 当前配对诊断的直接观察','',f'生成：{datetime.now().astimezone().isoformat(timespec="seconds")}。仅分析已经完成且保存的执行结果。','',
        '| 方法/seed | 静态成功 | 原动态成功 | 静态−动态 | 静态移动循环任务/50 | 动态轨迹覆盖 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f'| {row["method"]}/{row["seed"]} | {100*row["static_success"]:.0f}% | {100*row["dynamic_success"]:.0f}% | {100*row["static_minus_dynamic_success"]:+.0f}个百分点 | {row["static_cycle_tasks"]}/50 | {row["dynamic_trace_tasks"]}/50 |')
    lines+=['','当前C的静态失败主要表现为移动循环：seed0的13个失败中12个出现循环，seed1为13/13；D分别为19/20、17/17。等待主导的静态失败各有至多1例。它直接支持“基础多目标导航尚有缺口”，但不能单凭这张表判定是监督减少造成。',
        '', 'C两seed平均静态成功74%，原动态成功74%；D静态63%，原动态69%。D seed0去掉动态障碍后成功率下降14个百分点。动态障碍既可能造成冲突，也可能改变局部输入和决策，帮助策略离开循环；后一个解释是待核查的推测，不能把净差值当成纯避障损失。应查看逐任务的静态成功/动态失败和反向配对，而非只比较均值。',
        '', '复用C/D全程曲线：C归一化AULC为0.4588/0.4840，D为0.4422/0.4412；D−C逐seed为−0.0166/−0.0428，均值−0.0297。当前两个seed没有显示风险自适应学得更快。D的AULC跨seed波动更小，但最终成功率波动更大（样本SD：D 7.07个百分点、C 2.83个百分点），不能笼统宣称更稳定。最终动态碰撞均值从C的13%降为D的3%，超时从13%升为28%，成功从74%降为69%；整体收益未得到支持。',
        '', '旧C/D动态循环只覆盖每模型4/50任务，其中留存规则包含固定任务和首个失败，存在选择性，不能用这些任务推断D总体动态循环更多。完整静态循环比较是可观察证据；动态总体循环比较仍缺轨迹。',
        '', '## 已保存的代表性案例','']
    for case in cases:
        r=case['record'];outcome='成功' if r['success']=='1' else '碰撞' if r['collision']=='1' else '超时'
        lines.append(f'- {case["method"]}/{case["seed"]} {case["condition"]} {r["task_id"]}：{outcome}，{r["steps"]}步，等待{float(r["wait_fraction"])*100:.1f}%，移动重复访问{float(r["revisit_move_ratio"])*100:.1f}%，循环覆盖{float(r["periodic_movement_ratio"])*100:.1f}%，最终静态距离{r["final_static_distance"]}。')
    completed=all((c.directory(method,seed)/'completion.json').exists() for method in c.METHODS for seed in (0,1))
    completion_text='四个新增外部基线均已完成，完整比较与剩余缺口见 FINAL_SUMMARY.zh-CN.md。' if completed else '四方法总体结论须等四个新增外部基线完成。'
    lines+=['','图见 `representative_trajectories.png`。图中动态障碍没有画在静态背景上；真实时序冲突见各条件 offline_cases.csv，不能据图中未画障碍推断执行时没有障碍。这些是按冻结规则选出的解释案例，不替代全部50任务统计。',
        '', completion_text+'当前数据没有支持将D升级为主方法，也不能宣称两seed正式显著性、未见目标泛化或50万步已经足够。']
    (c.RESULTS/'DIAGNOSIS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(c.RESULTS/'diagnostic_supplement_provenance.json',dict(source_sha256=c.sha(__file__),input_sha256=input_hashes,
        offline_only=True,new_policy_evaluations=0,created=datetime.now().astimezone().isoformat(timespec='seconds')))
    print('Saved diagnostic supplement and offline trajectory figure.',flush=True)


if __name__=='__main__':main()

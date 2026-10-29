"""Five-method evidence from completed data; pending controls remain NA."""
from __future__ import annotations

import csv
from datetime import datetime
from statistics import mean,stdev

import numpy as np
import td_control_common as c
from analyze_multitask_external import aggregate,aulc,number,learning_audit

LABELS={'C':'C 持续弱监督','D':'D 风险自适应','d3qn':'原始D3QN','dqfd':'DQfD',c.METHOD:'D3QN＋TD约束'}
PHASES=((0,500000),(0,100000),(100000,250000),(250000,500000),(400000,500000))
METRICS=('final_success_rate','final_collision_rate','final_timeout_rate',*[f'aulc_{a}_{b}' for a,b in PHASES],'last_five_success_mean')


def rows(path):
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def run_metrics(method,seed):
    path=c.directory(seed) if method==c.METHOD else c.previous.directory(method,seed)
    curve=rows(path/'validation_curve.csv');complete=(path/'completion.json').exists()
    if not curve:return dict(method=method,seed=seed,complete=False,last_validation_step=0,**{k:None for k in METRICS}),[],None
    if [int(r['environment_steps_total']) for r in curve]!=list(range(0,int(curve[-1]['environment_steps_total'])+1,10000)):
        raise ValueError('Validation checkpoint sequence differs')
    record=dict(method=method,seed=seed,complete=complete,last_validation_step=int(curve[-1]['environment_steps_total']))
    for name in ('success','collision','timeout'):record['final_'+name+'_rate']=float(curve[-1][name+'_rate']) if complete else None
    for a,b in PHASES:record[f'aulc_{a}_{b}']=aulc(curve,a,b)
    tail=[float(r['success_rate']) for r in curve if int(r['environment_steps_total'])>=460000]
    record['last_five_success_mean']=mean(tail) if len(tail)==5 else None
    if complete:
        info=c.model_info(method,seed);result=info['result'];manifest=c.read(path/'run_manifest.json')
        record.update(online_environment_steps=500000,online_updates=result.get('online_updates',499501),
            offline_updates=result.get('offline_updates',0),total_updates=result['updates'],device=manifest['device'],
            recorded_session_seconds=result.get('session_seconds'),model_sha256=info['model_sha256'])
    if method!=c.METHOD:
        audit,_=learning_audit(method,seed)
        if audit:
            for key in ('q_min','q_max','q_abs_max','raw_target_min','raw_target_max','target_clip_fraction'):
                audit[key]=audit.get('online_'+key)
            lows=[audit.get(k) for k in ('online_raw_target_min','online_raw_one_target_min','online_raw_n_target_min') if audit.get(k) is not None]
            if lows:audit['raw_target_min']=min(lows)
            if method in ('C','D'):
                audit['clipped_target_min']=max(-6.,audit['raw_target_min']);audit['clipped_target_max']=min(10.,audit['raw_target_max'])
            else:audit['clipped_target_min']=audit['raw_target_min'];audit['clipped_target_max']=audit['raw_target_max']
            log=rows(path/'update_log.csv')
            audit['gradient_norm_before_clip_peak']=max(float(r.get('gradient_norm_before_clip',r.get('gradient_norm'))) for r in log)
    else:
        log=rows(path/'update_log.csv');audit=dict(method=method,seed=seed)
        for key,operation in (('q_min',min),('q_max',max),('q_abs_max',max),('raw_target_min',min),('raw_target_max',max),('clipped_target_min',min),('clipped_target_max',max)):
            audit[key]=operation(float(r[key]) for r in log) if log else None
        total=sum(int(r['bin_updates']) for r in log)
        audit['target_clip_fraction']=sum(float(r['target_clip_fraction'])*int(r['bin_updates']) for r in log)/total if total else None
        audit.update(logged_updates=total,offline_updates=0,teacher_loss=False,advice=False,uniform_replay=True,q_output_clip=False)
        audit['gradient_norm_before_clip_peak']=max(float(r['gradient_norm_before_clip']) for r in log) if log else None
    return record,[dict(method=method,seed=seed,**r) for r in curve],audit


def loop_metrics():
    pointer=c.REG/'loop_latest.json'
    if not pointer.exists():return [],[],[],[],[]
    root=c.ROOT/c.read(pointer)['results_path'];seed_rows=[];all_tasks=[];cases=[];changes=[];buffers=[]
    for method in c.METHODS:
        for seed in c.SEEDS:
            by={}
            for condition in ('dynamic','static'):
                directory=root/method/f'seed_{seed}'/condition
                if not (directory/'completion.json').exists():continue
                data=rows(directory/'per_task.csv');by[condition]={r['task_id']:r for r in data}
                all_tasks.extend(dict(method=method,seed=seed,condition=condition,**r) for r in data)
                for band in ('all','short','medium','long'):
                    for group in ('all','reference_clear','reference_conflict'):
                        part=[r for r in data if (band=='all' or r['band']==band) and (group=='all' or r['reference_group']==group)]
                        if not part:continue
                        failed=[r for r in part if int(r['timeout'])];steps=sum(int(r['steps']) for r in failed)
                        row=dict(method=method,seed=seed,condition=condition,band=band,reference_group=group,tasks=len(part),trace_tasks=len(part),
                            **{name+'_rate':mean(int(r[name]) for r in part) for name in ('success','collision','timeout')},
                            periodic_movement_task_rate=mean(r['periodic_movement']=='True' for r in part),
                            loop_timeout_task_rate=sum(int(r['timeout']) and r['periodic_movement']=='True' for r in part)/len(part),
                            mean_wait_fraction=mean(float(r['wait_fraction']) for r in part),
                            mean_revisit_move_ratio=mean(float(r['revisit_move_ratio']) for r in part),
                            mean_final_static_distance=mean(float(r['final_static_distance']) for r in part),
                            timeout_steps=steps,
                            timeout_teacher_not_top_fraction=sum(int(r['teacher_not_top_rank']) for r in failed)/steps if steps else None,
                            timeout_teacher_loses_to_retreat_fraction=sum(int(r['teacher_loses_to_retreat']) for r in failed)/steps if steps else None,
                            goal_input_mismatches=sum(int(r['goal_input_mismatch']) for r in part),
                            teacher_label_mismatches=sum(int(r['teacher_label_mismatch']) for r in part),
                            saved_action_greedy_mismatches=sum(int(r['saved_action_not_current_greedy']) for r in part))
                        seed_rows.append(row)
                selected=c.read(directory/'representatives.json')['task_ids']
                for r in data:
                    if r['task_id'] in selected:cases.append(dict(method=method,seed=seed,condition=condition,
                        diagnostic_directory=str(directory.relative_to(c.ROOT)),**r))
            if set(by)=={'dynamic','static'}:
                for task_id,static in by['static'].items():
                    dynamic=by['dynamic'][task_id]
                    row=dict(method=method,seed=seed,task_id=task_id,band=static['band'],reference_group=static['reference_group'])
                    for name in ('success','collision','timeout','wait_fraction','revisit_move_ratio','periodic_movement_ratio','final_static_distance'):
                        row[name+'_static_minus_dynamic']=float(static[name])-float(dynamic[name])
                    changes.append(row)
            path=root/'replay_audit'/f'{method}_seed_{seed}.json'
            if path.exists():
                record=c.read(path);buffers.append(dict(method=method,seed=seed,**record['counts'],scope=record['scope']))
    return seed_rows,all_tasks,cases,changes,buffers


def plots(curves,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'C':'#237bb5','D':'#cf5b28','d3qn':'#668c48','dqfd':'#9467bd',c.METHOD:'#8c564b'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    for method in c.METHODS:
        seeds=[]
        for seed in c.SEEDS:
            part=[r for r in curves if r['method']==method and r['seed']==seed]
            if not part:continue
            x=[int(r['environment_steps_total']) for r in part];y=[float(r['success_rate']) for r in part]
            axes[0].plot(x,y,color=colors[method],linestyle='-' if seed==0 else '--',label=f'{method}/{seed}',alpha=.85)
            seeds.append(dict(zip(x,y)))
        if not seeds:continue
        shared=sorted(set.intersection(*(set(s) for s in seeds)));array=np.array([[s[x] for x in shared] for s in seeds]);avg=array.mean(0)
        axes[1].plot(shared,avg,color=colors[method],label=f'{method} n={len(seeds)}')
        if len(seeds)>1:
            sd=array.std(0,ddof=1);axes[1].fill_between(shared,avg-sd,avg+sd,color=colors[method],alpha=.12)
    for ax in axes:ax.set(xlabel='Online environment steps',ylabel='Frozen validation success',ylim=(-.03,1.03));ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_title('Every seed, raw values');axes[1].set_title('Mean and training-seed sample SD')
    fig.savefig(c.RESULTS/'five_method_learning_curves.png',dpi=150);fig.savefig(c.RESULTS/'five_method_learning_curves.pdf');plt.close(fig)
    _,data,planner=c.shared();tasks={r['task_id']:r for r in data['pools']['validation']}
    grid=np.zeros((data['size'],data['size']))
    for position in data['obstacles']:grid[tuple(position)]=1
    destination=c.RESULTS/'typical_cases';destination.mkdir(exist_ok=True)
    # One deterministic representative per method/seed/condition; all selected
    # cases and all timeout Q vectors remain available in the CSVs.
    seen=set()
    for case in cases:
        key=case['method'],case['seed'],case['condition']
        if key in seen:continue
        seen.add(key);directory=c.ROOT/case['diagnostic_directory'];task_id=case['task_id'];task=tasks[task_id]
        trace=next(r for r in c.read(directory/'trajectories.json') if r['task_id']==task_id)
        steps=[r for r in rows(directory/'timeout_and_reference_steps.csv') if r['task_id']==task_id]
        path=np.array(trace['positions']);reference=np.array(planner.path(task['start'],task['goal']))
        fig,(a,b)=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        a.imshow(grid,cmap='Greys',origin='upper',vmin=0,vmax=1)
        a.plot(reference[:,1],reference[:,0],'--',color='#999999',linewidth=1,label='Static teacher (offline)')
        a.plot(path[:,1],path[:,0],color='#c44e39',linewidth=1.6,label='Executed trace')
        a.scatter(task['start'][1],task['start'][0],color='#2079b6',s=30);a.scatter(task['goal'][1],task['goal'][0],marker='*',color='#dbb126',s=95)
        points=np.r_[path,reference];low=np.maximum(points.min(0)-3,0);high=np.minimum(points.max(0)+3,data['size']-1)
        a.set_xlim(low[1]-.5,high[1]+.5);a.set_ylim(high[0]+.5,low[0]-.5);a.legend(fontsize=7)
        for action in range(5):b.plot([int(r['tick']) for r in steps],[float(r[f'q_{action}']) for r in steps],linewidth=.9,alpha=.7,label=f'Q action{action}')
        b.plot([int(r['tick']) for r in steps],[float(r['teacher_q']) for r in steps],'k--',linewidth=2,label='Teacher Q (offline)')
        b.set(xlabel='Executed tick',ylabel='Final network Q');b.grid(alpha=.2);b.legend(fontsize=7)
        fig.suptitle(f'{key[0]}/seed{key[1]} {key[2]} {task_id}: timeout; executed supervision=0\nDynamic obstacles omitted from background; true conflicts are offline CSV columns',fontsize=10)
        filename=f'{key[0]}_seed{key[1]}_{key[2]}_{task_id}'
        fig.savefig(destination/(filename+'.png'),dpi=130);plt.close(fig)


def display(value,percentage=False):
    if value is None or value=='':return 'NA'
    return f'{float(value)*100:.2f}%' if percentage else f'{float(value):.4f}'


def main():
    c.RESULTS.mkdir(parents=True,exist_ok=True);runs=[];curves=[];audits=[]
    for method in c.METHODS:
        for seed in c.SEEDS:
            result,curve,audit=run_metrics(method,seed);runs.append(result);curves.extend(curve)
            if audit:audits.append(audit)
    averages=aggregate(runs,METRICS,['method']);lookup={(r['method'],r['seed']):r for r in runs}
    effects=[]
    for seed in c.SEEDS:
        for name,left,right,interpretation in (('TD_only_minus_raw',c.METHOD,'d3qn','TD target bound alone'),
            ('C_minus_TD_only','C',c.METHOD,'Combined Astar advice plus persistent supervision; not persistent supervision alone')):
            a,b=lookup[left,seed],lookup[right,seed]
            effects.append(dict(contrast=name,seed=seed,interpretation=interpretation,
                **{key:float(a[key])-float(b[key]) if a.get(key) is not None and b.get(key) is not None else None for key in METRICS}))
    effect_means=aggregate(effects,METRICS,['contrast'])
    loops,tasks,cases,changes,buffers=loop_metrics()
    observation_evidence=dict(repeated_states=0,closed_cycles=0)
    if (c.REG/'loop_latest.json').exists():
        from cycle_observation_evidence import review
        observation_evidence=review(c.ROOT/c.read(c.REG/'loop_latest.json')['results_path'])
    loop_keys=('success_rate','collision_rate','timeout_rate','periodic_movement_task_rate','loop_timeout_task_rate',
        'mean_wait_fraction','mean_revisit_move_ratio','mean_final_static_distance','timeout_teacher_not_top_fraction','timeout_teacher_loses_to_retreat_fraction')
    loop_means=aggregate(loops,loop_keys,['method','condition','band','reference_group']) if loops else []
    for name,values in (('five_method_per_seed.csv',runs),('five_method_mean_sample_sd.csv',averages),('all_validation_curves.csv',curves),
        ('controlled_effects_per_seed.csv',effects),('controlled_effects_mean_sample_sd.csv',effect_means),('learning_and_cost_audit.csv',audits),
        ('loop_seed_summary.csv',loops),('loop_method_mean_sample_sd.csv',loop_means),('all_paired_tasks.csv',tasks),
        ('paired_task_changes.csv',changes),('typical_cases.csv',cases),('replay_goal_teacher_audit.csv',buffers)):
        c.csv_write(c.RESULTS/name,values)
    complete=all(r['complete'] for r in runs)
    lines=['# TD约束控制与循环诊断：五方法比较','',f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}。',
        '',f'新增控制完成 {sum(r["complete"] for r in runs if r["method"]==c.METHOD)}/2；原四方法8次运行全部复用。未完成项保留NA。',
        '', '| 方法 | seed | 验证步数 | 最终成功 | 碰撞 | 超时 | 全程AULC | 末10万AULC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:lines.append(f'| {LABELS[r["method"]]} | {r["seed"]} | {r["last_validation_step"]} | {display(r["final_success_rate"],True)} | {display(r["final_collision_rate"],True)} | {display(r["final_timeout_rate"],True)} | {display(r["aulc_0_500000"])} | {display(r["aulc_400000_500000"])} |')
    lines+=['','各seed及均值/样本SD、全程/分阶段AULC、末五点和成本见five_method_*.csv。所有统计单位为训练seed，只有两个，反复验证同50任务不增加训练重复数。',
        '', '## TD单独贡献与教学组合增益','',
        'TD-only−原始D3QN比较仅改变完整TD目标裁剪，衡量该设置下TD约束单独影响；C−TD-only比较对应“A*代选＋持续动作监督”的教学组合，不能归为持续监督单项。两者数值可以相加，但不能把其比例当作相互独立的因果贡献比例。',
        '', '| 对照 | seed | 成功率差 | 全程AULC差 | 超时率差 |','|---|---:|---:|---:|---:|']
    for r in effects:lines.append(f'| {r["contrast"]} | {r["seed"]} | {display(r["final_success_rate"],True)} | {display(r["aulc_0_500000"])} | {display(r["final_timeout_rate"],True)} |')
    if not complete:lines+=['','控制训练尚未全部完成，以上效应暂不作最终判断；不按中途成绩停训、换seed或改参数。']
    else:
        for r in effect_means:
            lines.append(f'\n{r["contrast"]}：最终成功差均值 {display(r["final_success_rate_mean"],True)}，样本SD {display(r["final_success_rate_sample_sd"],True)}；全程AULC差均值 {display(r["aulc_0_500000_mean"])}。应结合碰撞、超时、末段曲线共同解释。')
    lines+=['','## 循环、教师标签和目标输入','',
        '新动态补录只针对C/D缺失轨迹，逐任务核对原成功/碰撞/超时/步数及等待/交互；原成绩和日志不修改。旧四方法可用静态/动态轨迹直接复用，新控制最终动态只用原定50万步验证，并新增移除动态障碍的静态评估。',
        '', '| 方法/seed | 条件 | 成功 | 超时 | 循环超时任务占比 | 超时步中教师非首位 | 超时步中回退Q更高 | 目标/教师错误 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in loops:
        if r['band']=='all' and r['reference_group']=='all':
            lines.append(f'| {r["method"]}/{r["seed"]} | {r["condition"]} | {display(r["success_rate"],True)} | {display(r["timeout_rate"],True)} | {display(r["loop_timeout_task_rate"],True)} | {display(r["timeout_teacher_not_top_fraction"],True)} | {display(r["timeout_teacher_loses_to_retreat_fraction"],True)} | {r["goal_input_mismatches"]}/{r["teacher_label_mismatches"]} |')
    lines+=['','这些是执行后观测。教师Q长期低于回退动作可以解释当时网络为何选择循环，却不能单独确定这种排序是怎样训练出来的。没有标签/目标错误证据时不把失败归因于标签错配；静态教师正确也不表示动态环境中该动作总安全。',
        '', '实际评估监督系数始终0。timeout_and_reference_steps.csv中的C/D权重均为事后假设：状态硬筛查版本和额外排除碰撞转移的回放资格版本分别记录，不冒充真实梯度更新。五动作Q、合法掩码、教师独立BFS校验、Q排名/差值、风险及真实未来冲突均在同表，风险只从当前/历史局部观测计算。',
        '', 'C/D最终回放各20000条的目标、教师和风险核验见replay_goal_teacher_audit.csv。该审计只覆盖最终保留经验，不能据此证明每一次历史更新均无错配。',
        '', f'补充的观测复现审计检出 {observation_evidence["repeated_states"]} 个至少出现3次的完全相同观测状态、{observation_evidence["closed_cycles"]} 个观测后继唯一且至少重复3次的闭环。repeated_observation_states.csv保留状态SHA、访问次数、五动作Q与教师/回退差；recurrent_observation_cycles.csv保留闭环状态、位置和动作。等待自环不算移动闭环；同观测存在多个后继时不声称闭环。这些数据可解释固定网络为何反复选同一动作，不能单独确定训练阶段形成该Q排序的原因。',
        '', '静态/动态配对同时改变动态占据和网络输入分布，不等价于单独静态训练。不同方法实际交互率不必相同，冲突分组复用共同静态参考规则，不能按策略是否遇障碍分别筛选。',
        '', '## 价值尺度、成本与结论边界','',
        'learning_and_cost_audit.csv与原始update_log记录Q和原始/裁剪后TD范围、按真实更新数加权的裁剪比例、梯度及训练更新成本。新控制只裁剪完整TD目标，不裁剪Q输出，不增加损失，无示范/离线更新。TD原始/裁剪样本每1000更新导出整批64条，首批也记录。',
        '', 'DQfD仍披露每seed额外20000离线更新、永久3472条示范，仅成功覆盖70/240训练任务及31/40目标；生成15107环境步由两个seed共享。其低成绩不直接归因于Q爆炸，本轮外部基线未见千级价值膨胀或非有限失败。',
        '', 'D保持探索方法身份。仅2seed冻结验证结果，不宣称正式显著性或未见目标泛化。不自动延长预算、不启动CE、监督下限扫描、五seed或完整消融。',
        '', '## 复现与后续协议','',
        '独立协议和命令：docs/MULTITASK_TD_CONTROL.zh-CN.md；同教师CE/大间隔候选日程：docs/TEACHER_LOSS_COMPARISON_PLAN.zh-CN.md。本轮仅整理该计划，未启动CE训练。',
        '', '```powershell',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Status',
        '# 只有当前进程退出后才续跑；完整恢复优化器/目标网络/回放/RNG/环境：',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Resume -Workers 1',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Diagnose',
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Report',
        '```']
    (c.RESULTS/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plots(curves,cases)
    c.write(c.RESULTS/'comparison_status.json',dict(updated=datetime.now().astimezone().isoformat(timespec='seconds'),
        all_ten_runs_complete=complete,new_training_runs=2,completed_new_runs=sum(r['complete'] for r in runs if r['method']==c.METHOD),
        paired_task_records=len(tasks),replay_audited_transitions=sum(int(r['transitions']) for r in buffers),
        test_pool_evaluations=0,extra_training_started=0))
    print('Updated five-method report:',c.RESULTS/'REPORT.zh-CN.md',flush=True)


if __name__=='__main__':main()

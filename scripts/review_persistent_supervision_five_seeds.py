"""Five-seed audit, final500 evaluation, figures and conditional current-method selection."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

import run_persistent_supervision_five_seeds as study
from analyze_persistent_teaching import read_csv, auc
from astar_d3qn.evaluation.rollout import evaluate_agent

ROOT, common, legacy = study.ROOT, study.common, study.legacy
SEEDS = study.ALL_SEEDS
METHODS = ('original_asvc','persistent_asvc','dqfd','d3qn','astar_prefill')
LABELS = dict(original_asvc='ASVC (original withdrawal)',persistent_asvc='ASVC (persistent supervision)',
              dqfd='DQfD',d3qn='D3QN',astar_prefill='A* prefill + TD constraint')
COLORS = dict(original_asvc='#777777',persistent_asvc='#0072B2',dqfd='#D55E00',d3qn='#999933',astar_prefill='#882255')
CN = dict(original_asvc='原ASVC',persistent_asvc='持续弱监督ASVC',dqfd='DQfD',d3qn='D3QN',astar_prefill='A*初始化＋TD约束')
EVAL = study.RESULTS/'final500'
write_json,write_records_csv = study.parent.write_json,study.parent.write_records_csv


def directory(method,seed):
    if method=='persistent_asvc':return study.new_directory(seed)
    if method=='original_asvc':return study.baseline_directory(seed)
    names = dict(dqfd='whole_map_91701_dqfd_comparison_v1/dqfd',d3qn='whole_map_91701_value_repair_v1/unguided_raw',
                 astar_prefill='asvc_paper_supplement_v1/astar_prefill_bound')
    return ROOT/'outputs'/names[method]/f'seed_{seed}'


def run_info(method,seed):
    path=directory(method,seed)
    result=common.read_json(path/'result.json')
    done=common.read_json(path/'completion.json')
    if not done['complete'] or result['environment_steps']!=200000:raise ValueError(f'Incomplete run: {path}')
    model=path/'model_final.pth';digest=common.sha256(model)
    if result.get('final_model_sha256',digest)!=digest:raise ValueError('Final model changed.')
    if method=='persistent_asvc':
        if result['gradient_updates']!=199501:raise ValueError('Wrong update budget.')
        reference=common.read_json(study.baseline_directory(seed)/'run_manifest.json')
        if result['initial_state_sha256']!=reference['initial_state_sha256']:raise ValueError('Unpaired seed.')
        for name,value in result['source_sha256'].items():
            if common.sha256(ROOT/name)!=value:raise ValueError(f'Changed training source: {name}')
        prefix=common.read_json(path/'prefix_parity.json')
        if len(prefix)<9 or not all(r['exact_weights_match'] for r in prefix):raise ValueError('Pre-intervention parity failed.')
        for filename,field,expected in (('advice_budget.csv','mean_requested_probability',0.),
                                        ('value_learning.csv','teacher_margin_weight_mean',.1)):
            values=[float(r[field]) for r in read_csv(path/filename) if int(r['bin_start_step'])>=100001]
            if not values or not np.allclose(values,expected,atol=1e-10,rtol=0):raise ValueError('Teaching floor audit failed.')
    return path,result,model,digest


def evaluate_final500():
    config,inputs=study.configure()
    frozen=common.read_json(study.REG/'freeze_manifest.json')
    if frozen['config']!=config or frozen['source_sha256']!=study.source_hashes():raise ValueError('Extension freeze mismatch.')
    planned=[(seed,*run_info('persistent_asvc',seed)) for seed in SEEDS]
    scenes,_,scene_freeze=study.tail.load_frozen(legacy.load_config(study.tail.CONFIG))
    scene_digest=scene_freeze['files_sha256']['independent_final_scenarios.json']
    if scene_digest!=frozen['checks']['final500_scene_sha256']:raise ValueError('Scene freeze changed.')
    all_rows,all_summaries=[],[]
    for seed,path,result,model,digest in planned:
        target=EVAL/f'seed_{seed}'
        registration=dict(seed=seed,model_sha256=digest,scene_sha256=scene_digest,scene_count=500,
            checkpoint_steps=200000,epsilon=0,teacher=False,static_mask=True,dynamic_shield=False,
            used_for_schedule_tuning=False,device='cpu',source_sha256=common.sha256(Path(__file__)),
            rollout_sha256=common.sha256(ROOT/'src/astar_d3qn/evaluation/rollout.py'))
        if (target/'completion.json').exists():
            if common.read_json(target/'registration.json')!=registration:raise ValueError('Cached final500 registration differs.')
            summary=common.read_json(target/'summary.json');rows=read_csv(target/'scene_details.csv')
        else:
            agent=legacy.runtime.make_agent(config,seed,'cpu');agent.load_weights(model)
            factory=study.tail.FrozenFactory(inputs[0],scenes,trace=True)
            started=perf_counter()
            with legacy.runtime.preserved_evaluation(agent):
                summary,rows,_=evaluate_agent(agent,[inputs[0]]*500,max_steps=300,window_size=15,
                    reward_config=legacy._reward_config(config),terminate_on_collision=True,
                    environment_factory=factory,mask_static_invalid_actions=True)
            if [r['scenario_id'] for r in rows]!=[s['scenario_id'] for s in scenes]:raise ValueError('Scene order/identity mismatch.')
            summary=dict(seed=seed,scene_count=500,safe_success_rate=summary['safe_success_rate'],
                dynamic_collision_rate=summary['dynamic_collision_rate'],
                timeout_rate=sum(r['termination_reason']=='timeout' for r in rows)/500)
            failures=[dict(scenario_id=r['scenario_id'],safe_success=False,termination_reason=r['termination_reason'],steps=e.trace)
                for r,e in zip(rows,factory.environments) if not r['safe_success']]
            write_json(failures,study.OUT/'final500'/f'seed_{seed}'/'failure_trajectories.json')
            write_json(registration,target/'registration.json');write_records_csv(rows,target/'scene_details.csv')
            write_json(summary,target/'summary.json')
            if common.sha256(model)!=digest:raise ValueError('Evaluation changed a source model.')
            write_json(dict(complete=True,seconds=perf_counter()-started),target/'completion.json')
        all_summaries.append(summary);all_rows.extend(dict(seed=seed,**r) for r in rows)
        print(f'[final500 seed{seed}] success={summary["safe_success_rate"]:.2%}',flush=True)
    write_records_csv(all_summaries,EVAL/'summary.csv');write_records_csv(all_rows,EVAL/'scene_details.csv')
    write_json(dict(complete=True,models=5,episodes=2500,scene_sha256=scene_digest,epsilon=0,
        source_models={str(m.relative_to(ROOT)):d for _,_,_,m,d in planned}),EVAL/'verification.json')


def cached_final500(method,seed,model,digest,scene_digest):
    if method=='persistent_asvc':return common.read_json(EVAL/f'seed_{seed}'/'summary.json')
    if method=='astar_prefill':
        path=ROOT/'results/asvc_paper_supplement_v1/evaluation_v2/original500/astar_prefill_bound'/f'seed_{seed}'
        reg=common.read_json(path/'registration.json')
        if reg['model_sha256']!=digest or reg['scene_sha256']!=scene_digest:raise ValueError('Prefill final500 cache mismatch.')
        return common.read_json(path/'summary.json')
    path=ROOT/common.read_json(ROOT/'results/latest_all_eligible_modules_final500_v1/latest.json')['evaluation']
    audit=common.read_json(path/'verification.json')
    if audit['frozen_scene_sha256']!=scene_digest or audit['source_model_sha256'][model.relative_to(ROOT).as_posix()]!=digest:
        raise ValueError('Historical final500 model/scenes differ.')
    key=dict(original_asvc='advice_bound_margin_all_eligible',dqfd='dqfd',d3qn='unguided_raw')[method]
    return next(r for r in read_csv(path/'summary.csv') if r['method']==key and int(r['seed'])==seed)


def collect():
    _,_,scene_freeze=study.tail.load_frozen(legacy.load_config(study.tail.CONFIG))
    scene_digest=scene_freeze['files_sha256']['independent_final_scenarios.json']
    metrics,curves,evidence=[],[],{}
    for method in METHODS:
        for seed in SEEDS:
            path,result,model,digest=run_info(method,seed)
            rows=read_csv(path/'validation_curve.csv')
            x=np.array([int(r['environment_steps_total']) for r in rows])
            y=np.array([float(r['safe_success_rate']) for r in rows])
            if len(x)!=21 or x[0]!=0 or x[-1]!=200000 or np.any(np.diff(x)<=0):raise ValueError('Expected all 21 checkpoints.')
            late=x>=100000; near=(x>=90000)&(x<=120500)
            final=cached_final500(method,seed,model,digest,scene_digest)
            metrics.append(dict(method=method,seed=seed,full_aulc=auc(x,y,0,200000),
                early_aulc=auc(x,y,0,50000),late_aulc=auc(x,y,100000,200000),
                withdrawal_min=float(y[near].min()),late_min=float(y[late].min()),
                final_validation=float(y[-1]),final500_success=float(final['safe_success_rate']),
                final500_collision=float(final['dynamic_collision_rate']),final500_timeout=float(final['timeout_rate'])))
            for r in rows:
                curves.append(dict(method=method,seed=seed,step=int(r['environment_steps_total']),
                    success=float(r['safe_success_rate']),timeout=float(r['timeout_rate']),
                    wait=float(r['mean_wait_steps'])))
            for name in ('result.json','validation_curve.csv','model_final.pth'):
                p=path/name;evidence[p.relative_to(ROOT).as_posix()]=common.sha256(p)
    return metrics,curves,evidence


def figures(output,metrics,curves):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for suffix in ('png','pdf','svg'):fig.savefig(output/f'{name}.{suffix}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    def curve(method,seed,key='success'):
        rows=[r for r in curves if r['method']==method and r['seed']==seed]
        return np.array([r['step'] for r in rows]),np.array([r[key] for r in rows])
    def band(ax,method,start,end,key='success'):
        points=np.linspace(start,end,201)
        ys=np.array([np.interp(points,*curve(method,s,key))*100 for s in SEEDS])
        mean=ys.mean(axis=0);sd=ys.std(axis=0,ddof=1)
        ax.plot(points/1000,mean,label=LABELS[method],color=COLORS[method],lw=1.8)
        ax.fill_between(points/1000,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=COLORS[method],alpha=.10)
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6))
    for method in METHODS:
        band(axes[0],method,0,200000)
    for method in METHODS[:3]:band(axes[1],method,80000,200000)
    for ax,title in zip(axes,('Full training','Late-stage stability')):
        ax.set(title=title,xlabel='Environment steps (thousands)',ylabel='Autonomous safe success (%)',ylim=(0,102))
        ax.axvline(100,color='#999999',ls=':',lw=1);ax.grid(alpha=.18);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Five-seed mean ± sample SD | fixed 50-scenario validation')
    fig.tight_layout();save(fig,'01_learning_curves')
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharex=True,sharey=True)
    for seed,ax in zip(SEEDS,axes.flat):
        for method in METHODS[:3]:
            x,y=curve(method,seed);ax.plot(x/1000,y*100,color=COLORS[method],label=LABELS[method],lw=1.6)
        ax.set(title=f'Seed {seed}',xlim=(80,200),ylim=(0,102),xlabel='Steps (thousands)',ylabel='Safe success (%)')
        ax.grid(alpha=.2);ax.axvline(100,color='#999999',ls=':',lw=1)
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',frameon=False,fontsize=10)
    fig.suptitle('Individual seeds: every scheduled checkpoint retained')
    fig.tight_layout();save(fig,'02_individual_seeds')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,('late_aulc','late_min','final500_success'),
        ('100k–200k AULC','100k–200k minimum','Final 500-scene success')):
        for seed in SEEDS:
            values=[100*next(r[key] for r in metrics if r['method']==m and r['seed']==seed) for m in METHODS[:2]]
            ax.plot((0,1),values,marker='o',lw=1.3,label=f'Seed {seed}')
        ax.set(title=title,xticks=(0,1),xticklabels=('Original','Persistent'),ylabel='%');ax.grid(axis='y',alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout();save(fig,'03_paired_changes')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in METHODS[:3]:band(axes[0],method,80000,200000,key='timeout')
    axes[0].set(title='Validation timeouts',xlabel='Steps (thousands)',ylabel='Timeout (%)',ylim=(0,100))
    positions=np.arange(len(METHODS))
    for offset,key,color,label in ((-.22,'final500_collision','#D55E00','Dynamic collision'),(.22,'final500_timeout','#777777','Timeout')):
        values=[[100*r[key] for r in metrics if r['method']==m] for m in METHODS]
        axes[1].bar(positions+offset,[np.mean(v) for v in values],width=.42,
            yerr=[np.std(v,ddof=1) for v in values],color=color,label=label,capsize=3)
    axes[1].set(title='Final frozen 500 scenes',ylabel='Rate (%)',xticks=positions,
        xticklabels=('Original','Persistent','DQfD','D3QN','A* prefill'))
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'04_failure_rates')
    steps=np.linspace(0,200000,401)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].plot(steps/1000,.8*np.maximum(0,1-steps/100000),color=COLORS['persistent_asvc'])
    axes[0].set(title='Action advice (unchanged)',ylabel='Requested probability',xlabel='Steps (thousands)')
    original=np.maximum(0,1-steps/100000)
    axes[1].plot(steps/1000,original,color=COLORS['original_asvc'],label='Original')
    axes[1].plot(steps/1000,np.maximum(.1,original),color=COLORS['persistent_asvc'],label='Persistent')
    axes[1].set(title='Action supervision',ylabel='Auxiliary loss weight',xlabel='Steps (thousands)');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'05_teaching_schedule')
    fig,ax=plt.subplots(figsize=(13,6));ax.set(xlim=(0,13),ylim=(0,6));ax.axis('off')
    boxes=[(.2,4.2,2.1,1.1,'Observation\nlocal static map\n3 dynamic frames + goal'),
        (3,4.2,2.1,1.1,'Dueling Double DQN\nstatic legal mask\nepsilon-greedy candidate'),
        (5.8,4.2,2.3,1.1,'Training action advice\nprobability 0.8 to 0\nOFF from 100k'),
        (8.8,4.2,2.4,1.1,'Real environment step\nuniform FIFO replay\nactual action retained'),
        (3,2.4,2.1,1.1,'Static-map A* teacher\nobserved-history\nrisk filter'),
        (5.8,2.4,2.8,1.1,'Independent teacher label\nmargin loss, margin 0.8\nweight max(0.1, 1 - t/100k)'),
        (9.3,2.4,2.5,1.1,'TD loss on actual action\nfull TD target clip\n[-6, 10]'),
        (7,.6,4.2,1,'Online optimization\nTD loss + persistent weak supervision'),
        (.2,.4,5.7,1.2,'Autonomous evaluation: final 200k model\nepsilon = 0; static legal mask only\nno A* advice or dynamic action shield')]
    for i,(x,y,w,h,label) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc='#E8F4ED' if i in (5,8) else '#EAF3F8',ec='#35566A'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for start,end in [((2.3,4.75),(3,4.75)),((5.1,4.75),(5.8,4.75)),((8.1,4.75),(8.8,4.75)),
        ((1.25,4.2),(3,3.1)),((5.1,3.2),(5.8,4.25)),((5.1,2.95),(5.8,2.95)),
        ((10,4.2),(10.55,3.5)),((8.9,4.2),(8.2,3.5)),((7.2,2.4),(8,1.6)),
        ((10.55,2.4),(10.1,1.6))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#35566A',lw=1.4))
    ax.set_title('ASVC with persistent weak action supervision',fontsize=13)
    save(fig,'06_framework')


def analyze():
    config,_=study.configure()
    metrics,curves,evidence=collect()
    keys=[k for k in metrics[0] if k not in ('method','seed')]
    means=[];pairs=[]
    for method in METHODS:
        selected=[r for r in metrics if r['method']==method]
        if len(selected)!=5:raise ValueError('Do not drop a seed.')
        row=dict(method=method)
        for key in keys:
            values=[r[key] for r in selected];row[key+'_mean']=float(np.mean(values));row[key+'_sd']=float(np.std(values,ddof=1))
        means.append(row)
    for seed in SEEDS:
        old=next(r for r in metrics if r['method']=='original_asvc' and r['seed']==seed)
        new=next(r for r in metrics if r['method']=='persistent_asvc' and r['seed']==seed)
        pairs.append(dict(seed=seed,**{k:new[k]-old[k] for k in keys}))
    before,after=means[:2];new_pairs=[r for r in pairs if r['seed'] in study.NEW_SEEDS]
    allowance=config['five_seed_review']['promotion_criteria']['final500_maximum_mean_success_drop']
    checks=dict(both_new_seeds_late_aulc_not_lower=all(r['late_aulc']>=-1e-10 for r in new_pairs),
        both_new_seeds_late_minimum_not_lower=all(r['late_min']>=-1e-10 for r in new_pairs),
        five_seed_mean_late_aulc_higher=after['late_aulc_mean']>before['late_aulc_mean'],
        five_seed_mean_late_minimum_higher=after['late_min_mean']>before['late_min_mean'],
        final500_within_registered_one_point_margin=after['final500_success_mean']>=before['final500_success_mean']-allowance)
    decision=dict(recommended_as_current=all(checks.values()),criteria=checks,
        final500_allowed_drop_percentage_points=100*allowance,
        interpretation='Predeclared descriptive screening criteria, not statistical significance or proof of generalization.')
    output=study.RESULTS/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));output.mkdir(parents=True)
    for name,rows in (('per_seed.csv',metrics),('means.csv',means),('paired_differences.csv',pairs),('curves.csv',curves)):
        write_records_csv(rows,output/name)
    write_json(decision,output/'decision.json')
    write_json(dict(complete=True,new_training_runs=2,reused_training_runs=3,seeds=list(SEEDS),source_sha256=evidence,
        analysis_source_sha256=common.sha256(Path(__file__)),final500_verification=common.read_json(EVAL/'verification.json'),
        earlier_three_seeds_used_for_method_selection=True,added_seeds_are_replication=True,
        raw_curves_unmodified=True,chart_bands='Sample SD across 5 seeds; linear timestamp alignment only; no smoothing.',
        historical_ablations_and_density_results_not_relabelled=True),output/'evidence.json')
    figures(output,metrics,curves)
    lines=['# 持续弱监督：五个seed复核与主结果图','',
        '复用seed0–2，只新增seed3、4。各20万环境步；监督权重下限0.1，A*代选在10万步退出。评估由网络独立行动。','',
        '前三个seed参与了方法选择，后两个seed用于复核；合并五seed不是五个全新、盲选的重复实验。','',
        '|方法|全程AULC|后半程AULC|后半程最低成功率|最终500安全成功率|',
        '|---|---:|---:|---:|---:|']
    for r in means:
        lines.append('|'+CN[r['method']]+'|'+'|'.join(f'{r[k+"_mean"]*100:.2f}±{r[k+"_sd"]*100:.2f}%' for k in ('full_aulc','late_aulc','late_min','final500_success'))+'|')
    lines+=['','各指标均为五seed均值±样本标准差；先逐seed取后半程最低点，再平均。最终500成绩与固定50验证曲线分开报告。','',
        '## 新版相对原版的同seed差值（百分点）','',
        '|seed|后半程AULC变化|后半程最低值变化|最终500成功率变化|','|---|---:|---:|---:|']
    for r in pairs:lines.append(f'|{r["seed"]}|'+'|'.join(f'{100*r[k]:+.2f}' for k in ('late_aulc','late_min','final500_success'))+'|')
    lines+=['','## 采用判定','',
        '通过预先登记的描述性门槛，建议采用持续弱监督作为当前主方法。' if decision['recommended_as_current'] else
        '没有全部通过预先登记的描述性门槛；保留候选状态，不能直接宣布替换当前主方法。','',
        '门槛为新增两个seed的后半程AULC和最低值均不降低、五seed平均两项均提高，且最终500平均成功率下降不超过1个百分点。这个容许幅度用于工程选择，不是显著性检验。','']
    for key,value in checks.items():lines.append(f'- {key}: {value}')
    lines+=['','## 图表','',
        '![主对比曲线](01_learning_curves.png)','',
        '![每个seed](02_individual_seeds.png)','',
        '图表同时提供300dpi PNG、PDF和SVG；还包括配对变化、超时/碰撞、教学日程。绘图只对齐验证时间，不平滑原始曲线。','',
        'DQfD额外使用2万次离线更新、永久示范和PER；相同在线交互步数不代表相同总训练预算。',
        '固定地图及起终点；500场景与旧方法共用冻结集合，不是跨地图测试。旧消融、障碍密度和轨迹图仍属于原版，不能改名当作新版结果。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (study.RESULTS/'LATEST.md').write_text(f'# 五seed持续弱监督结果\n\n[报告]({output.name}/REPORT.md) · [主曲线]({output.name}/01_learning_curves.png) · [采用判定]({output.name}/decision.json)\n',encoding='utf-8')
    write_json(dict(complete=True,report=str(output/'REPORT.md'),recommended_as_current=decision['recommended_as_current']),study.RESULTS/'review_status.json')
    select_current(output,decision)
    print(output,flush=True)
    print(decision,flush=True)
    return output,decision


def select_current(output,decision):
    """Apply the user's conditional selection only after the complete registered review."""
    if not decision['recommended_as_current']:return
    selected=dict(method='asvc_persistent_supervision',display_name='ASVC-D3QN（持续弱监督）',
        selected_at=datetime.now().isoformat(timespec='seconds'),source_review=str(output.relative_to(ROOT)),
        decision_sha256=common.sha256(output/'decision.json'),seeds=list(SEEDS),
        advice_probability_floor=0.,supervision_weight_floor=.1,advice_exit_step=100000,
        supervision_floor_start_step=90000,environment_steps=200000,
        training_config_seed012='configs/asvc_persistent_teaching_v1.yaml',
        training_config_seed34=str(study.CONFIG.relative_to(ROOT)),
        final_models={str(seed):str((study.new_directory(seed)/'model_final.pth').relative_to(ROOT)) for seed in SEEDS},
        historical_method='advice_bound_margin_all_eligible',historical_ablation_and_density_still_old_method=True)
    target=ROOT/'data/asvc_current_method.json'
    if target.exists():
        previous=common.read_json(target)
        if previous.get('source_review')==selected['source_review']:return
        write_json(previous,ROOT/'data/asvc_current_method_history'/f'{datetime.now():%Y%m%d_%H%M%S}.json')
    write_json(selected,target)
    rel='../'+str(output.relative_to(ROOT)).replace('\\','/')
    lines=['# 当前主方法：ASVC持续弱监督版','',
        '五seed及冻结500场景复核通过预先登记的描述性门槛，按本轮用户授权采用为当前主方法。原版实验、模型和图表保留为历史版本。','',
        '动作代选按原计划在10万步退出；动作监督权重下降到0.1后持续保留。奖励、网络、观测、TD约束与风险筛选沿用原版。评估只使用网络。','',
        f'[五seed报告]({rel}/REPORT.md) · [主对比图]({rel}/01_learning_curves.png) · [逐seed曲线]({rel}/02_individual_seeds.png)','',
        '## 代码和训练入口','',
        '- 实现：[persistent_teaching.py](../src/astar_d3qn/agents/persistent_teaching.py)。',
        '- seed0–2：[run_persistent_teaching.py](../scripts/run_persistent_teaching.py)，使用`--variants supervision_floor`。',
        '- seed3–4：[run_persistent_supervision_five_seeds.py](../scripts/run_persistent_supervision_five_seeds.py)。',
        '- 完整模型路径及选择证据：[当前方法登记](../data/asvc_current_method.json)。',
        '- 五个种子已经完成，不要为重画图重复训练。','',
        '重画当前五seed主图（复用已完成的最终500评估）：','',
        '```powershell',
        '& "D:\\softwares\\Anaconda3\\envs\\astar-drl\\python.exe" scripts/review_persistent_supervision_five_seeds.py',
        '```','',
        '## 论文图表边界','',
        '本次更新学习曲线、逐seed稳定性、配对差、最终500碰撞/超时及教学日程，包含PNG、PDF和SVG。',
        '旧版去TD、去代选等消融，以及障碍密度和轨迹图，不能改名当作新版结果。论文使用新版作为主方法时，这些对应实验仍需另行补齐或明确标为原版。',
        '三seed已参与方案选择，后两seed用于复核；结论限于当前固定地图及场景分布，不是跨地图泛化保证。','']
    (ROOT/'docs/ASVC_CURRENT.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
    begin='<!-- ASVC_CURRENT_SELECTION -->';end='<!-- /ASVC_CURRENT_SELECTION -->'
    for path,link in ((ROOT/'README.md','docs/ASVC_CURRENT.zh-CN.md'),
                      (ROOT/'docs/WORKSPACE_GUIDE.zh-CN.md','ASVC_CURRENT.zh-CN.md'),
                      (ROOT/'docs/ASVC_FIRST_TRAINING_STEP.zh-CN.md','ASVC_CURRENT.zh-CN.md')):
        text=path.read_text(encoding='utf-8')
        block=begin+'\n'+f'2026-10-23更新：当前主方法已采用**持续弱动作监督版ASVC**，详见[当前方法与新图]({link})。下文原有10万步撤监督的说明及历史消融结果对应原版。'+'\n'+end
        if begin in text and end in text:
            left=text.index(begin);right=text.index(end,left)+len(end);text=text[:left]+block+text[right:]
        else:
            title,separator,rest=text.partition('\n');text=title+'\n\n'+block+'\n'+rest
        if path.name=='WORKSPACE_GUIDE.zh-CN.md':
            text=text.replace('正在补seed3、4并准备同一冻结500场景评估和新版主图','已完成seed3、4补跑、同一冻结500场景评估和新版主图')
            text=text.replace('当前论文方法叫 **ASVC-D3QN**','原版论文方法叫 **ASVC-D3QN**')
            text=text.replace('## 当前主线从哪个文件启动','## 原版主线从哪个文件启动')
        path.write_text(text,encoding='utf-8')
    print('CURRENT METHOD SELECTED: '+str(target),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.evaluate:evaluate_final500()
    output,decision=analyze()
    select_current(output,decision)


if __name__=='__main__':main()

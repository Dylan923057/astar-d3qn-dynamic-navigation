"""Read completed 50k probes; autonomous diagnostics use original50, never final500."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import random

import numpy as np
import torch

import ddqna_adapter_common as common
import diagnose_comparison_evidence as diagnosis
from analyze_runtime_path_pilot import load_csv, static_distances
from analyze_teaching_efficiency import aulc
from astar_d3qn.utils.io import write_json,write_records_csv

ARMS=('huber','mse','huber_bound')


def uniform_index_schedule_hash():
    # Protocol-derived audit, not an additional replay/gradient training run.
    rng=random.Random(0);digest=hashlib.sha256()
    for step in range(500,50001):
        indices=rng.sample(range(min(step,10000)),64)
        digest.update(np.asarray(indices,dtype='<i4').tobytes())
    return digest.hexdigest()


def probe_aulc(curve):
    x=np.array([int(r['environment_steps_total']) for r in curve])
    y=np.array([float(r['safe_success_rate']) for r in curve])
    if x[0]!=0 or x[-1]!=50000 or np.any(np.diff(x)<=0):raise ValueError('Probe AULC requires actual0–50k data.')
    return aulc(x,y,0,50000)


def update_windows(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(int(r['environment_step'])-1)//10000].append(r)
    return [dict(left=i*10000,right=(i+1)*10000,updates=len(group),
        predicted_min=min(float(r['predicted_min']) for r in group),
        predicted_max=max(float(r['predicted_max']) for r in group),
        actual_target_min=min(float(r['actual_target_min']) for r in group),
        actual_target_max=max(float(r['actual_target_max']) for r in group),
        raw_target_max=max(float(r['raw_target_max']) for r in group),
        td_abs_mean=float(np.mean([float(r['td_abs_mean']) for r in group])),
        per_batch_td_p95_mean=float(np.mean([float(r['td_abs_p95']) for r in group])),
        td_abs_max=max(float(r['td_abs_max']) for r in group),
        raw_td_abs_mean=float(np.mean([float(r['raw_td_abs_mean']) for r in group])),
        outside_return_fraction=sum(float(r['predicted_outside_return_count']) for r in group)/sum(float(r['sample_count']) for r in group),
        target_clip_fraction=sum(float(r['actual_target_clipped_count']) for r in group)/sum(float(r['sample_count']) for r in group))
        for i,group in sorted(grouped.items())]


def paper_evidence(evidence,output):
    summaries=load_csv(evidence/'evidence/summary.csv')
    curves=load_csv(evidence/'evidence/all_curves.csv')
    rows=[]
    for seed in common.SEEDS:
        c=[r for r in curves if r['method']=='advice_bound_margin' and int(r['seed'])==seed and int(r['environment_steps_total'])>=100000]
        minimum=min(c,key=lambda r:float(r['safe_success_rate']))
        after=[r for r in c if int(r['environment_steps_total'])>int(minimum['environment_steps_total'])]
        recovered=next((r for r in after if float(r['safe_success_rate'])>=.9),None)
        confirmed=next((after[i+2] for i in range(len(after)-2) if all(float(r['safe_success_rate'])>=.9 for r in after[i:i+3])),None)
        summary=next(r for r in summaries if r['method']=='advice_bound_margin' and int(r['seed'])==seed)
        rows.append(dict(seed=seed,minimum_step=int(minimum['environment_steps_total']),
            minimum_success=float(minimum['safe_success_rate']),
            first_recovery_90_step=int(recovered['environment_steps_total']) if recovered else None,
            recovery_three_point_confirmation_step=int(confirmed['environment_steps_total']) if confirmed else None,
            final_success=float(summary['final_success']),final_collision=float(summary['final_collision']),
            full_aulc=float(summary['full_aulc'])))
    write_records_csv(rows,output/'withdrawal_recovery.csv')
    paired=load_csv(evidence/'evidence/paired_means.csv')
    relevant=[r for r in paired if r['method'] in ('added_margin','added_action_override','added_supervision','td_bound_with_override_and_margin')]
    write_records_csv(relevant,output/'paper_module_evidence.csv')
    return rows,relevant


def analyze(root,evidence):
    output=common.ROOT/'results/ddqna_stability_probe_v1'/datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f')
    # Audit all completed arms before creating a result directory.
    runs={a:common.read_json(root/a/'seed_0/result.json') for a in ARMS}
    if any(not r['complete'] or r['environment_steps']!=50000 or r['gradient_updates']!=49501 for r in runs.values()):
        raise ValueError('All three paired50k probes must be complete.')
    if len({r['initial_learner_sha256'] for r in runs.values()})!=1 or len({common.canonical_hash(r['initial_rng_sha256']) for r in runs.values()})!=1:
        raise ValueError('Probe learner/RNG initial states differ.')
    scene_rows={a:load_csv(root/a/'seed_0/training_scenes.csv') for a in ARMS}
    shared_count=min(map(len,scene_rows.values()))
    if any(len({scene_rows[a][i]['scene_sha256'] for a in ARMS})!=1 for i in range(shared_count)):
        raise ValueError('Episode-index scene sequence changed between arms.')
    output.mkdir(parents=True,exist_ok=False)
    config=runs['huber']['config'];problem=common.validate_config(common.legacy.load_config(common.CONFIG))[0]
    common.register(common.legacy.load_config(common.CONFIG))
    scenes=common.read_json(common.ROOT/config['dataset']['validation_reference'])['scenarios']
    diagnosis.static_distances_cached=static_distances(problem)
    diagnostics=[];windows=[];finals=[];onsets=[];fingerprints={}
    for arm in ARMS:
        directory=root/arm/'seed_0';updates=load_csv(directory/'updates.csv')
        if len(updates)!=49501 or [int(r['environment_step']) for r in updates]!=list(range(500,50001)):
            raise ValueError('Missing real TD update records or different update frequency.')
        windows.extend(dict(arm=arm,**r) for r in update_windows(updates))
        def first(predicate):
            return next((int(r['environment_step']) for r in updates if predicate(r)),None)
        onsets.append(dict(arm=arm,first_sampled_q_outside_return_step=first(lambda r:float(r['predicted_outside_return_count'])>0),
            first_sampled_abs_q_gt20_step=first(lambda r:float(r['predicted_abs_gt20_count'])>0),
            first_batch_td_p95_gt10_step=first(lambda r:float(r['td_abs_p95'])>10),
            first_target_clip_step=first(lambda r:float(r['actual_target_clipped_count'])>0)))
        curve=load_csv(directory/'validation_curve.csv')
        for row in curve:
            step=int(row['environment_steps_total']);model=directory/f'model_step_{step:06d}.pth'
            fingerprints[str(model.relative_to(common.ROOT))]=common.sha256(model)
            agent=common.make_agent(config,problem,0);agent.load_weights(model)
            s,episodes,decisions=diagnosis.rollout(config,problem,agent,scenes,'autonomous')
            if s.get('original_api_mismatches',0):raise ValueError('Batched diagnostic action differs from original API.')
            if not all(np.isclose(s[k],float(row[k])) for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate')):
                raise ValueError('Probe autonomous results not reproduced.')
            diagnostics.append(dict(arm=arm,environment_step=step,**s))
            write_records_csv(episodes,output/f'{arm}_{step:06d}_episodes.csv')
            write_records_csv(decisions,output/f'{arm}_{step:06d}_q_decisions.csv')
            print(f'[{arm}] step={step} autonomous={s["safe_success_rate"]:.1%} Qmax={s["valid_q_max"]:.2f}',flush=True)
        s=diagnostics[-1];branches=load_csv(directory/'action_branches.csv');n=sum(int(r['steps']) for r in branches)
        requests=sum(int(r['teacher_branch']) for r in branches);applied=sum(int(r['teacher_applied']) for r in branches)
        finals.append(dict(arm=arm,seed=0,online_steps=50000,online_updates=49501,
            full_50k_aulc=probe_aulc(curve),
            autonomous_success=s['safe_success_rate'],collision_count=s['collision_count'],
            waiting_timeout_count=s['waiting_timeout_count'],looping_timeout_count=s['looping_timeout_count'],
            other_timeout_count=s['other_timeout_count'],network_teacher_agreement=s['network_teacher_agreement'],
            valid_q_min=s['valid_q_min'],valid_q_max=s['valid_q_max'],
            teacher_request_rate=requests/n,teacher_execution_rate=applied/n,
            requested_suggestion_valid_rate=applied/requests,
            **{k:runs[arm][k] for k in ('online_training_seconds','validation_seconds','total_seconds')}))
        if any(common.sha256(common.ROOT/p)!=v for p,v in fingerprints.items()):raise ValueError('Model changed during diagnosis.')
    write_records_csv(diagnostics,output/'autonomous_diagnosis.csv')
    write_records_csv(windows,output/'training_td_windows.csv')
    write_records_csv(finals,output/'summary.csv');write_records_csv(onsets,output/'onsets.csv')
    recovery,paired=paper_evidence(evidence,output)
    previous_root=common.ROOT/'outputs/ddqna_stability_probe_v1/20261010_185732_145439'
    reproduced={}
    for arm in ARMS:
        previous=previous_root/arm/'seed_0/model_probe_final.pth'
        current=root/arm/'seed_0/model_probe_final.pth'
        if previous.exists():
            reproduced[arm]=common.legacy.runtime.state_digest(torch.load(previous,map_location='cpu',weights_only=True))==common.legacy.runtime.state_digest(torch.load(current,map_location='cpu',weights_only=True))
    early_huber_bound_parameters={}
    for step in (0,10154,20059):
        models=[root/a/'seed_0'/f'model_step_{step:06d}.pth' for a in ('huber','huber_bound')]
        early_huber_bound_parameters[str(step)]=len({common.legacy.runtime.state_digest(torch.load(m,map_location='cpu',weights_only=True)) for m in models})==1
    write_json(dict(probe_root=str(root),complete=True,seed_count=1,
        initial_learner_sha256=runs['huber']['initial_learner_sha256'],initial_rng_identical=True,
        prior_completed_probe_final_parameters_match=reproduced,
        early_huber_bound_parameters_equal=early_huber_bound_parameters,
        episode_index_scene_sha256_matches=shared_count,scene_counts={a:len(s) for a,s in scene_rows.items()},
        random_alignment=runs['huber']['random_alignment'],model_sha256=fingerprints,
        uniform_replay_relative_index_schedule_sha256=uniform_index_schedule_hash(),
        uniform_replay_index_audit='Derived from registered uniform buffer: one addition per environment step, same sizes and one sample per update; contents differ by policy',
        original50_only=True,final500_used=False,thresholds_descriptive_only=dict(return_bound=[-6,10],large_sampled_abs_q=20,per_batch_td_p95=10),
        td_records='True pre-update target minus predicted Q, including actual online target-network lag; bound arm also records raw targets/errors',
        cost_note='Instrumentation overhead included; not a clean timing benchmark against older formal runs'),output/'verification.json')
    render(output,finals,onsets,windows,diagnostics,recovery,paired)
    print(output);return output


def rerender(root,evidence,previous):
    """Correct derived summaries while retaining checked rollouts and old reports."""
    verification=common.read_json(previous/'verification.json')
    if Path(verification['probe_root']).resolve()!=root or not verification['complete'] or verification['final500_used']:
        raise ValueError('Only reuse the completed original50 diagnosis of this probe.')
    for name,sha in verification['model_sha256'].items():
        if common.sha256(common.ROOT/name)!=sha:raise ValueError('Diagnosed checkpoint changed.')
    def numeric_rows(path):
        rows=load_csv(path)
        for r in rows:
            for k,v in r.items():
                if not v:r[k]=None;continue
                try:
                    f=float(v);r[k]=int(f) if f.is_integer() else f
                except ValueError:pass
        return rows
    finals=numeric_rows(previous/'summary.csv')
    for r in finals:r['full_50k_aulc']=probe_aulc(load_csv(root/r['arm']/'seed_0/validation_curve.csv'))
    windows=numeric_rows(previous/'training_td_windows.csv');onsets=numeric_rows(previous/'onsets.csv')
    diagnostics=numeric_rows(previous/'autonomous_diagnosis.csv')
    output=common.ROOT/'results/ddqna_stability_probe_v1'/datetime.now().strftime('analysis_corrected_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (('summary.csv',finals),('training_td_windows.csv',windows),('onsets.csv',onsets),('autonomous_diagnosis.csv',diagnostics)):
        write_records_csv(rows,output/name)
    recovery,paired=paper_evidence(evidence,output)
    write_json(dict(verification,derived_aulc_window=[0,50000],
        readonly_episode_and_q_diagnosis_directory=str(previous),
        supersedes_previous_probe_aulc_only=True,
        analysis_script_sha256=common.sha256(Path(__file__))),output/'verification.json')
    render(output,finals,onsets,windows,diagnostics,recovery,paired)
    print(output);return output


def render(output,finals,onsets,windows,diagnostics,recovery,paired):
    lines=['# DDQNA数值稳定性探针与论文证据','',
        '三组仅seed0，各50000环境步、49501更新；固定p=0.5，ε仍按150000步时钟。当前D3QN适配诊断，不代表原版DDQNA优劣，不使用独立500场景。','',
        'huber_bound仅将完整单步TD目标裁剪至[-6,10]，不裁剪网络Q输出。实际目标TD误差和裁剪前原始目标误差分别记录，避免把目标被限制本身当成导航已改善。','',
        '网络/教师一致率按自主轨迹的逐步决策统计；三组访问状态可能不同，不将其解释为相同状态集上的一致率。教师执行率来自训练分支日志，评价时教师执行为0。','',
        '沿用已有失败分类：等待动作占回合至少50%为等待超时；其余超时中，移动至少20次且至少半数进入已访问格，或轨迹尾段呈周期，归为绕圈超时。分类按轨迹计算，不从成功率推断。','',
        '|组|5万步自主成功|碰撞|等待超时|绕圈超时|Q范围|自主动作/教师一致|实际教师执行|5万步AULC|',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in finals:lines.append(f'|{r["arm"]}|{r["autonomous_success"]:.0%}|{r["collision_count"]}|{r["waiting_timeout_count"]}|{r["looping_timeout_count"]}|{r["valid_q_min"]:.2f}～{r["valid_q_max"]:.2f}|{r["network_teacher_agreement"]:.2%}|{r["teacher_execution_rate"]:.2%}|{r["full_50k_aulc"]:.2%}|')
    lines+=['','|组|在线训练秒|循环内验证秒|含初始/末尾评估与写盘总秒|','|---|---:|---:|---:|']
    for r in finals:lines.append(f'|{r["arm"]}|{r["online_training_seconds"]:.1f}|{r["validation_seconds"]:.1f}|{r["total_seconds"]:.1f}|')
    lines+=['','## 数值与行为时序','',
        '训练Q仅统计实际采样的动作Q；验证Q统计访问状态的所有静态合法动作。真实TD误差使用更新前policy与当时target网络。下面TD P95是每批P95的均值，不是全部样本的全局P95。超过回报边界的轻微误差不等同于发散；abs(Q)>20及TD P95>10仅作固定描述指标，不用于选模型/调参。','',
        '|组|环境步窗口|采样Q范围|真实TD绝对均值|裁剪前原始TD绝对均值|每批TD P95均值|TD绝对峰值|目标裁剪比例|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in windows:lines.append(f'|{r["arm"]}|{r["left"]}–{r["right"]}|{r["predicted_min"]:.2f}～{r["predicted_max"]:.2f}|{r["td_abs_mean"]:.3f}|{r["raw_td_abs_mean"]:.3f}|{r["per_batch_td_p95_mean"]:.3f}|{r["td_abs_max"]:.2f}|{r["target_clip_fraction"]:.2%}|')
    lines+=['','|组|首个采样Q越过[-6,10]|首个采样abs(Q)>20|首批TD P95>10|首次目标裁剪|','|---|---:|---:|---:|---:|']
    for r in onsets:lines.append('|'+r['arm']+'|'+'|'.join(str(r[k]) if r[k] is not None else '未出现' for k in ('first_sampled_q_outside_return_step','first_sampled_abs_q_gt20_step','first_batch_td_p95_gt10_step','first_target_clip_step'))+'|')
    lines+=['','|组|实际验证步|自主成功|等待超时|绕圈超时|合法Q最大|自主动作/教师一致|','|---|---:|---:|---:|---:|---:|---:|']
    for r in diagnostics:lines.append(f'|{r["arm"]}|{r["environment_step"]}|{r["safe_success_rate"]:.0%}|{r["waiting_timeout_count"]}|{r["looping_timeout_count"]}|{r["valid_q_max"]:.2f}|{r["network_teacher_agreement"]:.2%}|')
    lookup={r['arm']:r for r in finals};tail={r['arm']:r for r in windows if r['left']==40000}
    lines+=['','## 本探针结论','',
        f'Huber与MSE在4–5万步的采样Q峰值分别为{tail["huber"]["predicted_max"]:.2f}和{tail["mse"]["predicted_max"]:.2f}；真实TD绝对均值分别为{tail["huber"]["td_abs_mean"]:.3f}和{tail["mse"]["td_abs_mean"]:.3f}。MSE改变了数值增长速度，但两者最终自主成功均为{lookup["huber"]["autonomous_success"]:.0%}/{lookup["mse"]["autonomous_success"]:.0%}，不支持仅换损失即可获得可靠导航。','',
        f'TD约束组4–5万步采样Q峰值为{tail["huber_bound"]["predicted_max"]:.2f}，真实TD绝对均值为{tail["huber_bound"]["td_abs_mean"]:.3f}，最终自主成功率为{lookup["huber_bound"]["autonomous_success"]:.0%}。数值控制与导航效果分别判断，少数场景成功不作单seed优越性证据。','',
        '早期Q与TD误差较小时，自主导航已经很差；后段的数值增长与行为退化有时同时发生，但不是所有失败的必要条件或唯一原因。原版DDQNA网络/输入/动作等与当前适配不同，不能从MSE/Huber结果推断原版优劣。5万步的数值受控不能宣布长期失稳已解决。']
    lines+=['','## 随机过程对齐','',
        '三组完整初始网络/目标网络/优化器及学习器随机状态一致，重新设定Python/NumPy/PyTorch及CUDA种子，要求确定性算子。探索与回放种子0，教师种子730000，场景种子91701000。场景按回合序号逐项校验物理场景SHA。','',
        '不是全程相同轨迹：策略分化后回合长度与合法动作集合不同；探索RNG的choice消耗及教师条件请求可能分化。Uniform回放每步加入一条、同一步池长度相同、同更新序号采样RNG消耗相同，因此抽样相对序号按更新对齐；但经验内容随策略改变。验证使用原50冻结场景并保留训练随机状态。实际验证位于回合结束点，存在少量步数偏移。','',
        '训练耗时包含新增诊断记录开销，不能直接和旧正式训练比较纯计算效率。','',
        '## 论文证据：保留强基线与交接局限','',
        '|条件性增益，五seed配对|全程AULC差值，百分点±样本标准差|','|---|---:|']
    names={'added_margin':'完整组合－A*代选＋TD约束','added_action_override':'完整组合－无代选＋监督＋TD约束','added_supervision':'监督＋TD约束－仅TD约束','td_bound_with_override_and_margin':'完整组合－A*代选＋动作监督'}
    for r in paired:lines.append(f'|{names[r["method"]]}|{float(r["full_aulc_mean"])*100:.2f}±{float(r["full_aulc_sample_sd"])*100:.2f}|')
    lines+=['','|seed|撤教后最低值/步数|首次恢复≥90%|恢复后三点确认步数|最终成功|最终碰撞|','|---|---|---:|---:|---:|---:|']
    for r in recovery:lines.append(f'|{r["seed"]}|{r["minimum_success"]:.0%}/{r["minimum_step"]}|{r["first_recovery_90_step"]}|{r["recovery_three_point_confirmation_step"]}|{r["final_success"]:.0%}|{r["final_collision"]:.0%}|')
    lines+=['','三点恢复不等于后续永久稳定；最低值是已记录验证点的最低值。完整组合仍有退出低谷，seed3最终4/50动态碰撞。延期监督的14–20万步AULC五seed均下降，不能称已解决低谷。','',
        'DQfD保留为强基线：同为200000在线交互，但额外20000离线更新、永久7260示范及PER/多步TD/监督。完整组合与DQfD全程AULC分别91.37±1.05%和93.24±0.42%，不主张我方普遍更优。独立500结果单列于原报告，未参与本探针。','',
        '下一对照仅取消DQfD离线更新，保留原冻结示范、永久分区、PER、多步TD、大间隔监督、L2及原在线条件；已配对核验seed0–4，零训练更新。删除离线阶段也意味着没有离线更新产生的优化器状态、target同步、PER优先级与RNG消耗，这些是该单因素干预的后果，不人为补造。','',
        '贡献围绕监督与TD约束的组合增益、动作代选的额外贡献、撤教后低谷及恢复。DDQNA全0不是主要优势证据；风险筛选不保证安全。5万步稳定也不能证明20万步或多seed长期稳定。']
    lines+=['','## 可复核命令','',
        'DQfD取消离线预训练的零更新配置检查：','',
        '```powershell','python scripts/run_dqfd_no_pretrain.py --seeds 0 1 2 3 4','```','',
        '后续先运行seed0单因素对照的命令（本轮未执行，会启动200000步在线训练）：','',
        '```powershell','python scripts/run_dqfd_no_pretrain.py --train --seeds 0','```']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(8,9),sharex=True)
    for arm in ARMS:
        d=[r for r in diagnostics if r['arm']==arm];w=[r for r in windows if r['arm']==arm]
        axes[0].plot([r['environment_step'] for r in d],[max(abs(r['valid_q_min']),abs(r['valid_q_max'])) for r in d],marker='o',label=arm)
        axes[1].plot([(r['left']+r['right'])/2 for r in w],[r['per_batch_td_p95_mean'] for r in w],marker='o')
        axes[2].plot([r['environment_step'] for r in d],[r['safe_success_rate']*100 for r in d],marker='o')
    axes[0].set_yscale('log');axes[0].axhline(10,color='black',linestyle=':',alpha=.4);axes[0].set_ylabel('Visited legal |Q| max');axes[0].legend()
    axes[1].set_ylabel('Mean batch |TD| P95');axes[2].set_ylabel('Autonomous success (%)');axes[2].set_xlabel('Online environment steps')
    axes[2].set_ylim(0,100)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/'probe_timing.png',dpi=180);fig.savefig(output/'probe_timing.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('probe_root',type=Path)
    p.add_argument('--evidence',type=Path,default=common.ROOT/'results/comparison_evidence_diagnosis/20261010_162403_833725')
    p.add_argument('--reuse-diagnosis',type=Path,help='Recompute summaries without rerunning checked model evaluations.')
    a=p.parse_args();torch.set_num_threads(1)
    if a.reuse_diagnosis:rerender(a.probe_root.resolve(),a.evidence.resolve(),a.reuse_diagnosis.resolve())
    else:analyze(a.probe_root.resolve(),a.evidence.resolve())


if __name__=='__main__':main()

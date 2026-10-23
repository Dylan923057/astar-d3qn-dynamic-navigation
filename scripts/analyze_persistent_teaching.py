"""Read all 12 completed paired runs; compare autonomous validation trajectories."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run_persistent_teaching as study

ROOT = study.ROOT
NAMES = dict(baseline='Original withdrawal', supervision_floor='Supervision floor 0.1',
             advice_floor='Advice floor 5%', both_floors='Both floors')
COLORS = dict(baseline='#555555', supervision_floor='#0072B2', advice_floor='#D55E00', both_floors='#009E73')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def auc(x, y, start, end):
    if x[0] > start or x[-1] < end: raise ValueError('Incomplete curve; no extrapolation allowed.')
    points = np.r_[start, x[(x>start)&(x<end)], end]
    values = np.interp(points, x, y)
    return float(np.sum(np.diff(points)*(values[:-1]+values[1:])*.5)/(end-start))


def analyze():
    config, _ = study.configure()
    freeze = study.common.read_json(ROOT/config['persistent_teaching']['registration_root']/'freeze_manifest.json')
    if freeze['source_sha256'] != study.source_hashes(): raise ValueError('Training sources changed.')
    root = ROOT/config['experiment']['output_root']
    output = ROOT/config['experiment']['check_root']/('analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    metrics, curves, pairs, audit, schedules = [], [], [], [], []
    hashes = {}
    datasets = {}
    for method in ('baseline', *study.VARIANTS):
        for seed in study.SEEDS:
            directory = ROOT/config['persistent_teaching']['baseline_root']/f'seed_{seed}' if method=='baseline' else root/method/f'seed_{seed}'
            if not (directory/'completion.json').is_file(): raise ValueError(f'Incomplete run: {directory}')
            result = study.common.read_json(directory/'result.json')
            manifest = study.common.read_json(directory/'run_manifest.json')
            if not result['complete'] or result['environment_steps']!=200000 or result['gradient_updates']!=199501:
                raise ValueError('Incomplete training/update budget.')
            expected = next(r for r in freeze['checks']['initializations'] if r['seed']==seed)
            if manifest['initial_state_sha256'] != expected['initial_state_sha256']: raise ValueError('Unpaired initialization.')
            path = directory/'validation_curve.csv'
            if method=='baseline' and study.common.sha256(path)!=expected['baseline_curve_sha256']:
                raise ValueError('Historical curve changed.')
            rows = read_csv(path)
            x = np.array([int(r['environment_steps_total']) for r in rows])
            y = np.array([float(r['safe_success_rate']) for r in rows])
            if len(rows)!=21 or len(set(x))!=21 or np.any(np.diff(x)<=0) or x[-1]!=200000:
                raise ValueError('Expected all 21 fixed checkpoints.')
            datasets[method,seed] = rows
            near = (x>=90000)&(x<=120500)
            late = x>=100000
            index = np.flatnonzero(late)[np.argmin(y[late])]
            final = rows[-1]
            metrics.append(dict(method=method, seed=seed, full_aulc=auc(x,y,0,200000),
                post100k_aulc=auc(x,y,100000,200000), withdrawal_min=float(y[near].min()),
                post100k_min=float(y[late].min()), post100k_min_step=int(x[index]),
                final_success=float(final['safe_success_rate']), final_collision=float(final['collision_rate']),
                final_timeout=float(final['timeout_rate']), final_mean_wait=float(final['mean_wait_steps']),
                final_mean_steps=float(final['mean_steps']), last5_mean=float(y[-5:].mean())))
            for nominal, row in zip(range(0,200001,10000),rows):
                curves.append(dict(method=method,seed=seed,nominal_step=nominal,
                    actual_step=int(row['environment_steps_total']),success=float(row['safe_success_rate']),
                    timeout=float(row['timeout_rate']),collision=float(row['collision_rate']),
                    wait=float(row['mean_wait_steps'])))
            if method!='baseline':
                prefix = study.common.read_json(directory/'prefix_parity.json')
                if not prefix or not all(r['exact_weights_match'] for r in prefix): raise ValueError('Prefix audit failed.')
                audit.append(dict(method=method,seed=seed,exact_prefix_checkpoints=len(prefix),
                    last_exact_prefix_step=max(r['step'] for r in prefix)))
                for file, field in (('advice_budget.csv','mean_requested_probability'),
                                    ('value_learning.csv','teacher_margin_weight_mean')):
                    late_rows = [r for r in read_csv(directory/file) if int(r['bin_start_step'])>=100001]
                    actual = [float(r[field]) for r in late_rows]
                    expected_floor = config['persistent_teaching']['variants'][method]['advice_floor' if file.startswith('advice') else 'supervision_floor']
                    if not actual or not np.allclose(actual,expected_floor,atol=1e-10,rtol=0):
                        raise ValueError('Recorded teaching floor differs from protocol.')
                    schedules.append(dict(method=method,seed=seed,quantity=field,minimum=min(actual),maximum=max(actual)))
            for filename in ('validation_curve.csv','validation_details.csv','run_manifest.json','result.json',
                             'advice_budget.csv','value_learning.csv','model_final.pth'):
                p = directory/filename
                hashes[str(p.relative_to(ROOT))] = study.common.sha256(p)
    fields = ('full_aulc','post100k_aulc','withdrawal_min','post100k_min','final_success','final_collision','final_timeout','final_mean_wait','last5_mean')
    means = []
    for method in ('baseline',*study.VARIANTS):
        selected = [r for r in metrics if r['method']==method]
        row = dict(method=method,seed_count=len(selected))
        for field in fields:
            values = [r[field] for r in selected]
            row[field+'_mean'] = float(np.mean(values)); row[field+'_sd'] = float(np.std(values,ddof=1))
        means.append(row)
    for method in study.VARIANTS:
        for seed in study.SEEDS:
            old = next(r for r in metrics if r['method']=='baseline' and r['seed']==seed)
            new = next(r for r in metrics if r['method']==method and r['seed']==seed)
            pairs.append(dict(method=method,seed=seed,**{f:new[f]-old[f] for f in fields}))
    output.mkdir(parents=True)
    for filename, rows in (('per_seed.csv',metrics),('method_means.csv',means),('paired_differences.csv',pairs),
                           ('curves.csv',curves),('schedule_audit.csv',schedules),('prefix_audit.csv',audit)):
        study.write_records_csv(rows,output/filename)
    evidence = dict(complete_new_runs=9,reused_baselines=3,seeds=list(study.SEEDS),
        evaluation='50 fixed validation scenarios, epsilon=0, no teacher action or dynamic shield',
        independent_test_used=False,descriptive_not_significance_test=True,source_sha256=hashes,
        checkpoint_alignment='Per-seed AULC uses actual steps; figures align the 21 scheduled checkpoint indices.')
    study.write_json(evidence,output/'evidence.json')
    fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for method in ('baseline',*study.VARIANTS):
        values = np.array([[float(r['safe_success_rate'])*100 for r in datasets[method,s]] for s in study.SEEDS])
        nominal = np.arange(21)*10
        axes.flat[0].plot(nominal,values.mean(axis=0),label=NAMES[method],color=COLORS[method],lw=2)
        for i,seed in enumerate(study.SEEDS,1):
            axes.flat[i].plot(nominal,values[i-1],color=COLORS[method],lw=1.6)
    for ax,title in zip(axes.flat,('Three-seed mean','Seed 0','Seed 1','Seed 2')):
        ax.set(title=title,xlabel='Environment steps (thousands)',ylabel='Autonomous safe success (%)',ylim=(-2,102))
        ax.axvline(100,color='#999999',ls=':',lw=1);ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Persistent teaching: autonomous validation, 50 fixed scenarios')
    fig.tight_layout();fig.savefig(output/'success_curves.png',dpi=170);plt.close(fig)
    labels = dict(baseline='原方案',supervision_floor='持续弱监督',advice_floor='持续少量代选',both_floors='两者都保留')
    lines = ['# 持续指导实验：三种子结果','',
        '四组均为20万环境步；新增9次训练，复用原方案3次。评估关闭A*代选，使用固定50验证场景。奖励保持原版。','',
        '监督权重下限0.1（9万步起），代选概率下限0.05（93750步起）；保持原观测风险筛选与TD目标约束。','',
        '下表为3个训练种子的均值；最低成功率先逐种子取检查点最低值再平均。AULC为按训练步数平均的曲线面积，不是独立场景成功率。','',
        '|方案|9–12万最低成功率|10–20万最低成功率|10–20万AULC|最终成功率|最终超时率|',
        '|---|---:|---:|---:|---:|---:|']
    for r in means:
        lines.append('|'+labels[r['method']]+'|'+'|'.join(f'{r[k+"_mean"]:.2%}' for k in ('withdrawal_min','post100k_min','post100k_aulc','final_success','final_timeout'))+'|')
    lines += ['','## 配对差值（新方案减原方案，百分点）','',
        '|方案|种子|原退化窗口最低值变化|10–20万最低值变化|10–20万AULC变化|最终成功率变化|',
        '|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        lines.append(f'|{labels[r["method"]]}|{r["seed"]}|'+ '|'.join(f'{100*r[k]:+.2f}' for k in ('withdrawal_min','post100k_min','post100k_aulc','final_success'))+'|')
    lines += ['','![完整学习曲线](success_curves.png)','',
        '证据范围：3个种子、固定地图及起终点、已使用过的50验证场景；未使用独立500测试集，不宣称跨地图泛化或统计显著性。',
        '保留全部种子及全部检查点，不能仅根据最终成功率判定退化消失。指导比例不根据本批结果改动。',
        '数值、配对差、指导日程核查及训练前段逐参数一致性分别见同目录CSV；原始来源SHA见evidence.json。','']
    lines += ['## 逐种子改善情况','']
    for method in study.VARIANTS:
        selected = [r for r in pairs if r['method']==method]
        better_window = sum(r['withdrawal_min']>1e-10 for r in selected)
        better_late = sum(r['post100k_aulc']>1e-10 for r in selected)
        worse_min = sum(r['post100k_min'] < -1e-10 for r in selected)
        lines.append(f'- {labels[method]}：{better_window}/3个种子的原退化窗口最低值提高；'
            f'{better_late}/3个种子的后半程AULC提高；{worse_min}/3个种子的后半程最低值降低。')
    lines += ['','这些是配对描述统计。只有后半程曲线也支持改善，才不能简单解释为低谷后移；3个种子与离散检查点仍不足以证明退化永久消失。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    latest = output.parent/'LATEST.md'
    latest.write_text(f'# 持续指导实验结果\n\n[完整报告]({output.name}/REPORT.md) · [曲线]({output.name}/success_curves.png) · [逐种子指标]({output.name}/per_seed.csv)\n',encoding='utf-8')
    print(output,flush=True)
    print('\n'.join(lines[8:15]),flush=True)
    return output


if __name__ == '__main__': analyze()

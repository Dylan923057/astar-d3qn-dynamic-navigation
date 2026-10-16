"""Audit and package completed eleven-method results for GitHub; never trains."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import dqfd_comparison_common as common
from analyze_runtime_path_pilot import static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import aulc, mean_records


ROOT = common.ROOT


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def relative(path):
    return path.relative_to(ROOT).as_posix()


def latest(prefix):
    choices = sorted((ROOT/'results/whole_map_91701_dqfd_comparison_v1').glob(prefix+'_*'))
    complete = [p for p in choices if (p/'verification.json').is_file()
                and common.read_json(p/'verification.json').get('complete')]
    if not complete:
        raise FileNotFoundError(f'No completed {prefix}; run summarize_dqfd_comparison.py first.')
    return complete[-1]


def audit_final(config, analysis, evaluation):
    av, ev = (common.read_json(p/'verification.json') for p in (analysis, evaluation))
    if (av['checkpoint_count'] != 1155 or av['methods'] != list(common.ALL_METHODS)
            or ev['models'] != 55 or ev['scene_rows'] != 27500
            or ev['checkpoint_selection'] != 'final_200000_step_only'
            or ev['evaluation_epsilon'] != 0 or ev['action_override']
            or ev['future_obstacles_used_for_actions'] or ev['training_started']):
        raise ValueError('Incomplete unified final protocol.')
    scenes, _, frozen = common.tail.load_frozen(common.legacy.load_config(common.tail.CONFIG))
    if ev['frozen_scene_sha256'] != frozen['files_sha256']['independent_final_scenarios.json']:
        raise ValueError('Incorrect final scene set.')
    raw = ROOT/ev['full_failure_trace_root']
    if common.read_json(raw/'completion.json') != ev:
        raise ValueError('Raw final evaluation has no matching completion audit.')
    groups = defaultdict(list)
    for row in read_csv(evaluation/'scene_details.csv'):
        groups[row['method'], int(row['seed'])].append(row)
    summaries = {(r['method'], int(r['seed'])): r for r in read_csv(evaluation/'summary.csv')}
    behaviors = {(r['method'], int(r['seed']), r['scenario_id']): r
                 for r in read_csv(evaluation/'failure_behaviors.csv')}
    distances = static_distances(common.validate_config(config)[0])
    hashes, inspected, selected = {}, 0, []
    for method in common.ALL_METHODS:
        for seed in common.SEEDS:
            directory, result = common.audit_run(config, method, seed)
            model = directory/'model_final.pth'
            recorded = ev['source_model_sha256'].get(str(model.relative_to(ROOT)))
            if recorded != common.sha256(model) or recorded != result.get('final_model_sha256', recorded):
                raise ValueError('Final model fingerprint mismatch.')
            rows, summary = groups[method, seed], summaries[method, seed]
            if [r['scenario_id'] for r in rows] != [s['scenario_id'] for s in scenes]:
                raise ValueError('Final scenes not paired or in registered order.')
            if any(r['dynamic_route_ids'].split(';') != s['route_ids'] for r, s in zip(rows, scenes)):
                raise ValueError('Final route identities changed.')
            if any(int(r['dynamic_obstacle_count']) != len(s['obstacles']) for r,s in zip(rows,scenes)):
                raise ValueError('Final dynamic obstacle counts changed.')
            if any(float(r['static_collision']) for r in rows):
                raise ValueError('Static masking violated.')
            expected = dict(safe_success_rate=np.mean([float(r['safe_success']) for r in rows]),
                dynamic_collision_rate=np.mean([float(r['dynamic_collision']) for r in rows]),
                timeout_rate=np.mean([r['termination_reason'] == 'timeout' for r in rows]))
            if any(not np.isclose(float(summary[k]), v) for k, v in expected.items()):
                raise ValueError('Final scene results disagree with summary.')
            path = raw/method/f'seed_{seed}'/'failure_trajectories.json'
            traces = common.read_json(path)
            hashes[relative(path)] = common.sha256(path)
            failed = {r['scenario_id']: r for r in rows if not float(r['safe_success'])}
            if len(traces) != len(failed) or {t['scenario_id'] for t in traces} != set(failed):
                raise ValueError('Missing or duplicate full final failure trajectories.')
            recomputed, categories = [], set()
            for trace in traces:
                row = failed[trace['scenario_id']]
                if trace['termination_reason'] != row['termination_reason']:
                    raise ValueError('Failure reason mismatch.')
                b = trace_behavior(trace, distances)
                if b['steps'] != int(row['steps']) or b['steps'] > 300:
                    raise ValueError('Final trace length disagrees with episode result.')
                if b['termination_reason']=='timeout' and b['steps'] != 300:
                    raise ValueError('Timeout occurred before the fixed 300-step limit.')
                saved = behaviors[method, seed, trace['scenario_id']]
                for k in ('steps', 'wait_steps', 'repeat_move_count', 'unique_visited_cells',
                          'closest_static_remaining_steps', 'final_static_remaining_steps'):
                    if int(saved[k]) != b[k]:
                        raise ValueError(f'Failure behavior differs: {method}/{seed}/{k}.')
                if saved['termination_reason'] != b['termination_reason']:
                    raise ValueError('Failure behavior reason differs.')
                if (int(saved['tail_period']) if saved['tail_period'] else None) != b['tail_period']:
                    raise ValueError('Failure loop period differs.')
                for k in ('wait_fraction', 'repeat_move_fraction'):
                    if not np.isclose(float(saved[k]), b[k]):
                        raise ValueError('Failure behavior fraction differs.')
                for k in ('waiting_dominated', 'repeated_movement'):
                    if (saved[k] == 'True') != b[k]:
                        raise ValueError('Failure behavior category differs.')
                recomputed.append(b)
                category = ('collision' if b['termination_reason'] == 'collision' else
                            'waiting_timeout' if b['waiting_dominated'] else
                            'looping_timeout' if b['repeated_movement'] or b['tail_period'] else 'other_timeout')
                if category not in categories:
                    selected.append((method, seed, 200000, 'new500', category, path, trace))
                    categories.add(category)
            if any(int(summary[k]) != v for k, v in failure_counts(recomputed).items()):
                raise ValueError('Final collision/timeout counts disagree with raw trajectories.')
            inspected += len(traces)
    if len(behaviors) != inspected:
        raise ValueError('Extra or duplicated failure behavior rows.')
    return dict(complete=True, final_models=55, final_scene_rows=27500,
        full_final_failures_recomputed=inspected, raw_failure_file_sha256=hashes,
        source_model_sha256=ev['source_model_sha256'], frozen_scene_sha256=ev['frozen_scene_sha256'],
        training_checkpoint_count=1155, training_failures_inspected=av['inspected_failures'],
        training_started=False, uploaded_weights=False), selected


def figures(analysis, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_csv(analysis/'all_validation_checkpoints.csv')
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['method'], int(r['seed'])].append(r)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
    for method, ax in zip(common.ALL_METHODS, axes.flat):
        for seed in common.SEEDS:
            data = grouped[method, seed]
            ax.plot([int(r['environment_steps_total'])/1000 for r in data],
                    [float(r['safe_success_rate'])*100 for r in data], label=f'seed {seed}', lw=1.2)
        ax.set_title(method, fontsize=9)
        ax.axvline(100, color='gray', ls=':', lw=1)
        ax.axvline(140, color='gray', ls='--', lw=1)
        ax.set_ylim(-2, 102); ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc='center')
    fig.supxlabel('Online environment steps (thousands)')
    fig.supylabel('Autonomous safe success on original 50 scenes (%)')
    fig.tight_layout()
    fig.savefig(output/'all_seed_learning_curves.png', dpi=160)
    fig.savefig(output/'all_seed_learning_curves.pdf')
    plt.close(fig)
    selected = ('advice_bound_margin', common.tail.METHOD, *common.METHODS)
    fig, ax = plt.subplots(figsize=(10, 5))
    sensitivity = []
    grid = np.arange(0, 200001, 10000)
    for method in selected:
        ys = []
        for seed in common.SEEDS:
            data = grouped[method, seed]
            x = np.array([int(r['environment_steps_total']) for r in data])
            y = np.array([float(r['safe_success_rate']) for r in data])
            ys.append(np.interp(grid, x, y)*100)
            sensitivity.append(dict(method=method, seed=seed, posthoc_aulc_10000_200000=aulc(x,y,10000,200000)))
        ys = np.array(ys); mean, sd = ys.mean(axis=0), ys.std(axis=0, ddof=1)
        line, = ax.plot(grid/1000, mean, label=method)
        ax.fill_between(grid/1000, mean-sd, mean+sd, alpha=.12, color=line.get_color())
    ax.axvline(100, color='gray', ls=':'); ax.axvline(140, color='gray', ls='--')
    ax.set(xlabel='Online environment steps (thousands)', ylabel='Autonomous safe success (%)', ylim=(0,105))
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output/'main_comparison_mean_sample_sd.png', dpi=180)
    fig.savefig(output/'main_comparison_mean_sample_sd.pdf'); plt.close(fig)
    write_csv(output/'posthoc_10k_200k_sensitivity.csv', sensitivity)
    metric = 'posthoc_aulc_10000_200000'
    write_csv(output/'posthoc_sensitivity_means.csv', mean_records(sensitivity, [metric]))
    baseline = {r['seed']: r[metric] for r in sensitivity if r['method']=='advice_bound_margin'}
    paired = [dict(method=r['method']+'_minus_advice_bound_margin', seed=r['seed'],
                   **{metric: r[metric]-baseline[r['seed']]}) for r in sensitivity
              if r['method']!='advice_bound_margin']
    write_csv(output/'posthoc_sensitivity_paired_differences.csv', paired)
    write_csv(output/'posthoc_sensitivity_paired_means.csv', mean_records(paired, [metric]))


def supervision_coverage(config, output):
    rows = []
    for method in common.ALL_METHODS:
        for seed in common.SEEDS:
            directory = common.run_directory(config, method, seed)
            row = dict(method=method, seed=seed, online_supervised_sample_occurrences=0,
                online_sample_occurrences=199501*64, online_supervised_sample_fraction=0.,
                offline_supervised_sample_occurrences=0, collected_transitions=None,
                active_collected_transitions=None, eligible_collected_labels=None,
                collected_label_fraction_of_all=None, collected_label_fraction_while_active=None)
            if method.startswith('dqfd'):
                logs = read_csv(directory/'dqfd_learning.csv')
                row['online_supervised_sample_occurrences'] = sum(int(r['demo_samples']) for r in logs if r['phase']=='online')
                row['offline_supervised_sample_occurrences'] = sum(int(r['demo_samples']) for r in logs if r['phase']=='pretrain')
                if sum(int(r['sample_count']) for r in logs if r['phase']=='online') != row['online_sample_occurrences']:
                    raise ValueError('Online DQfD sample count mismatch.')
                if row['offline_supervised_sample_occurrences'] != 20000*64:
                    raise ValueError('Offline DQfD sample count mismatch.')
            else:
                logs = read_csv(directory/'value_learning.csv')
                row['online_supervised_sample_occurrences'] = sum(int(r['teacher_label_samples']) for r in logs)
            row['online_supervised_sample_fraction'] = row['online_supervised_sample_occurrences']/row['online_sample_occurrences']
            path = directory/'collection_coverage.csv'
            if path.is_file():
                data = read_csv(path)
                for key in ('collected_transitions','active_collected_transitions','eligible_collected_labels'):
                    row[key] = sum(int(r[key]) for r in data)
                if row['collected_transitions'] != 200000:
                    raise ValueError('Collection coverage has incomplete environment steps.')
                row['collected_label_fraction_of_all'] = row['eligible_collected_labels']/200000
                row['collected_label_fraction_while_active'] = row['eligible_collected_labels']/row['active_collected_transitions']
            rows.append(row)
    write_csv(output/'supervision_label_coverage.csv', rows)


def package(analysis, evaluation):
    config = common.legacy.load_config(common.CONFIG)
    common.frozen(config)
    audit, selected = audit_final(config, analysis, evaluation)
    name = 'whole_map_91701_eleven_methods_seeds0to4_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT/'results/analysis_upload'/name
    output.mkdir(parents=True, exist_ok=False)
    external = {}
    def record_external(path):
        external[relative(path)] = common.sha256(path)
    def copy(path, target):
        dest = output/target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for directory in (analysis, evaluation):
        for p in directory.iterdir():
            if p.is_file():
                record_external(p)
    # Historical 35 runs already have audited reading packages; copy only the 20 new runs.
    for method in (common.tail.METHOD, *common.METHODS):
        for seed in common.SEEDS:
            directory = common.run_directory(config, method, seed)
            for path in sorted(directory.iterdir()):
                if path.suffix not in ('.csv', '.json') or path.name.startswith('validation_failures_'):
                    continue
                copy(path, Path('run_logs')/method/f'seed_{seed}'/path.name)
    # Training-frozen sources and current support code are explicitly distinguished.
    registered = common.source_hashes()
    registered.update(common.tail.source_hashes())
    for method in common.ALL_METHODS:
        registered.update(common.read_json(common.run_directory(config,method,0)/'result.json')['source_sha256'])
    for base in ('src/astar_d3qn', 'scripts'):
        for path in (ROOT/base).rglob('*.py'):
            copy(path, Path('source_snapshot')/path.relative_to(ROOT))
    for path in (ROOT/'configs').glob('whole_map_91701*.yaml'):
        copy(path, Path('source_snapshot')/path.relative_to(ROOT))
    for path in (ROOT/'tests').glob('test_*comparison.py'):
        copy(path, Path('source_snapshot')/path.relative_to(ROOT))
    for filename in ('test_supervision_tail.py', 'test_value_repair.py', 'test_teaching_efficiency.py'):
        copy(ROOT/'tests'/filename, Path('source_snapshot/tests')/filename)
    for directory in ('data/whole_map_91701_teaching_efficiency_v1',
                      'data/whole_map_91701_supervision_tail_v1', 'data/whole_map_91701_dqfd_comparison_v1'):
        for path in (ROOT/directory).iterdir():
            if path.is_file():
                record_external(path)
    frozen = common.read_json(ROOT/'data/whole_map_91701_supervision_tail_v1/freeze_manifest.json')
    for name in frozen['source_data_sha256']:
        record_external(ROOT/name)
    prior_review = ROOT/'results/whole_map_91701_teaching_efficiency_v1/review_20261008_113945_236854'
    for name in ('REPORT.md','verification.json','reconstructed_failure_q.csv','critical_training_coverage.csv'):
        record_external(prior_review/name)
    # Preserve all original seed3 final collisions plus representative withdrawal failures.
    summaries = read_csv(analysis/'summary.csv')
    distances = static_distances(common.validate_config(config)[0])
    for row in summaries:
        method, seed = row['method'], int(row['seed'])
        steps = {200000}
        if method in ('advice_bound_margin', common.tail.METHOD, *common.METHODS):
            steps.update(int(row[k]) for k in ('original_window_min_step', 'new_window_min_step'))
        directory = common.run_directory(config, method, seed)
        for step in sorted(steps):
            path = directory/f'validation_failures_{step:06d}.json'
            traces = common.read_json(path)
            categories = set()
            for trace in traces:
                b = trace_behavior(trace, distances)
                category = ('collision' if b['termination_reason'] == 'collision' else
                            'waiting_timeout' if b['waiting_dominated'] else
                            'looping_timeout' if b['repeated_movement'] or b['tail_period'] else 'other_timeout')
                critical = method == 'advice_bound_margin' and seed == 3 and step == 200000
                if critical or category not in categories:
                    selected.append((method, seed, step, 'original50', category, path, trace))
                    categories.add(category)
    index = []
    trace_source_hashes = dict(audit['raw_failure_file_sha256'])
    for number, (method, seed, step, dataset, category, source, trace) in enumerate(selected):
        path = Path('trajectories')/f'{number:03d}_{method}_seed{seed}_{dataset}_{step}_{category}.json'
        write_json(output/path, trace)
        source_name = relative(source)
        if source_name not in trace_source_hashes:
            trace_source_hashes[source_name] = common.sha256(source)
        index.append(dict(file=path.as_posix(), method=method, seed=seed, dataset=dataset,
            environment_steps=step, scenario_id=trace['scenario_id'], category=category,
            selection_reason='all baseline seed3 final collisions or first per failure category in final/worst window',
            local_raw_source=source_name, source_sha256=trace_source_hashes[source_name]))
    write_csv(output/'trajectory_index.csv', index)
    figures(analysis, output)
    supervision_coverage(config, output)
    demo = np.load(common.artifact_root(config)/'demonstrations.npz', allow_pickle=False)
    actions, counts = np.unique(demo['action'], return_counts=True)
    demo_manifest = common.frozen(config)
    write_json(output/'demonstration_summary.json', dict(
        transition_count=int(len(demo['action'])), attempted_episodes=demo_manifest['attempted_episodes'],
        retained_successful_episodes=demo_manifest['retained_episodes'],
        action_counts={str(int(a)): int(n) for a,n in zip(actions,counts)},
        action_names={'0':'UP','1':'DOWN','2':'LEFT','3':'RIGHT','4':'WAIT'},
        source_sha256=demo_manifest['files_sha256']['demonstrations.npz'],
        teacher_policy='static A*; WAIT when observed three-frame risk veto rejects next cell',
        future_obstacle_information_used=False))
    audit['selected_full_trajectories'] = len(index)
    write_json(output/'verification.json', audit)
    write_json(output/'training_registered_source_sha256.json', registered)
    write_json(output/'external_artifact_sha256.json', external)
    def link(path, label):
        return f'[{label}](../../../{relative(path)})'
    report = ['# 十一方法×五seed正式实验阅读包', '',
        '55次训练全部完成，每次200000在线环境步；保持固定地图、起终点和每回合3–5动态障碍。原结果保留。', '',
        link(analysis/'REPORT.md', '原50场景完整学习曲线与逐seed汇总')+'；'+
        link(evaluation/'REPORT.md', '新冻结500场景最终确认结果')+'。', '',
        '统一评估55份20万步最终模型，共27500次自主评估；epsilon=0，无A*代选，不选最佳检查点。'+
        f'重新核对全部{audit["full_final_failures_recomputed"]}条最终失败轨迹。新500仅作最终确认，未调参。', '',
        '| 方法 | 全程AULC（原50） | 新500最终安全成功 | 新500动态碰撞 | 新500超时 |',
        '|---|---:|---:|---:|---:|']
    am = {r['method']: r for r in read_csv(analysis/'method_means.csv')}
    em = {r['method']: r for r in read_csv(evaluation/'method_means.csv')}
    def metric(row, key):
        return f'{float(row[key+"_mean"])*100:.2f}±{float(row[key+"_sample_sd"])*100:.2f}%'
    for method in common.ALL_METHODS:
        report.append('| '+method+' | '+metric(am[method], 'safe_success_aulc')+' | '+
                      ' | '.join(metric(em[method], k) for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate'))+' |')
    report += ['', '统计单位为5个配对训练seed；±为样本标准差，不能把500场景当成500个独立训练重复。'+
        'CSV中的率/AULC按0–1存储，配对差值乘100得到百分点。', '',
        '判断边界：DQfD在0在线步之前已用7260条永久安全示范完成20000次离线更新；每seed在线更新仍为199501次。'+
        '因此在线AULC对比不能代替等示范预算、等计算成本对比。示范只保留成功回合，是训练信息优势，但不使用验证或确认场景。'+
        '这批示范还包含198个WAIT标签（DOWN/RIGHT各3531），由当前和历史观测风险拒绝后等待产生；'+
        '原组合遇到风险时拒绝代选/辅助标签，二者教师处理方式也不同，不能将差异全归因于学习算法。'+
        '示范生成成本5.46秒是A/B共享的一次成本，成本表中的shared字段不能逐seed重复计费；C不使用这批示范。'+
        '原方法未记录的实测耗时保留空值，不能据此宣称DQfD实际更快或更慢。', '',
        '监督尾段延期的全程AULC提高，五seed的14–20万步AULC全部下降；四seed存在低谷后移迹象。'+
        '监督权重积分也从50000增至54000，不能把结果只归因于退出时钟。'+
        '取消风险筛选使平均学习效率降低且波动增大，最终50场景成功率较高不能证明筛选无用。', '',
        '新500结果用于最终自主表现确认，不能反馈调参。研究仍限于同地图路线组合与相位，不能声称跨地图泛化。'+
        '新500与旧50、旧500不重复，并从DQfD示范生成中排除；原随机在线训练没有逐回合记录路线组合，不能进一步声称已核实它与每次在线训练完全不重复。', '',
        link(ROOT/'docs/DQFD_COMPARISON_V1.zh-CN.md', 'DQfD适配差异与风险消融协议')+'；'+
        link(ROOT/'docs/SUPERVISION_TAIL_V1.zh-CN.md', '尾段延期日程与判据')+'。', '',
        link(prior_review/'reconstructed_failure_q.csv', '原seed3碰撞等关键失败状态的Q值诊断')+'；'+
        link(prior_review/'critical_training_coverage.csv', '原关键状态训练经历覆盖')+'。'+
        '这些是已完成的历史诊断；新500轨迹不能反馈教师日程或监督强度。', '',
        '文件导航：', '',
        '- `run_logs/`：尾段5次及DQfD/目标限制/无风险筛选15次完整训练、验证、监督及成本日志。',
        '- `source_snapshot/`：源码和配置；`training_registered_source_sha256.json`区分训练登记源码与整理时支持源码。',
        '- `trajectory_index.csv`与`trajectories/`：明确选择依据的完整失败轨迹；全部失败分类见原汇总CSV。',
        '- `all_seed_learning_curves.png/pdf`：55条自主学习曲线；`main_comparison_mean_sample_sd.png/pdf`：各seed按实际步数线性插值至同一1万步网格后绘制均值±样本标准差。',
        '- `posthoc_10k_200k_sensitivity.csv`：剔除前1万在线步的事后敏感性检查，不能替代预先约定的全程AULC。',
        '- `supervision_label_coverage.csv`：在线回放中监督样本出现次数/占比、DQfD离线监督样本次数，以及实际采集标签覆盖率（只有已记录的方法）；重复采样次数不是独立示范数。',
        '- `demonstration_summary.json`：冻结示范量、成功回合筛选及动作分布，明确DQfD教师中的WAIT标签来源。',
        '- `verification.json`：最终模型、场景及原始失败核对；`manifest.json`：包内文件SHA；`external_artifact_sha256.json`：链接的报告和冻结数据SHA。', '',
        link(ROOT/'results/analysis_upload/whole_map_91701_teaching_efficiency_seeds0to4_20261008_113945_236854/README.md', '历史七方法/35模型日志入口')+'；'+
        link(ROOT/'results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/README.md', '原六方法/30模型日志入口')+'。'+
        '这些历史包的旧500结果与本包的新500须分别引用。', '',
        '权重及批量原始轨迹保留在本地outputs，上传包记录路径和SHA；未提交或推送GitHub，未启动额外训练。']
    (output/'README.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    prompt = '''请从本包README开始，结合所链接的统一学习汇总、新500最终确认、逐seed配对差值、训练成本、源码与失败轨迹，分析A*教学如何加速D3QN自主学习。
请分别回答：1. 原组合相对各消融的收益及交互；2. DQfD的优势是否主要来自预训练起点、永久示范及额外计算，TD clip对DQfD是否有稳定增益；3. 风险筛选影响效率、稳定性还是最终避障；4. 尾段延期是否仅推迟低谷（两退出窗、合并窗、14–20万AULC及监督积分差）；5. 论文可支持的结论、缺少的公平对照及下一步最小实验。
统计单位是5个配对seed，报告均值±样本标准差和逐seed差值，不能以场景扩充训练重复。主指标为预定全程AULC；剔除前1万步仅是事后敏感性检查。DQfD先用7260条安全训练示范做20000次离线更新，0在线步已预训练，不能把在线步公平等同总训练成本公平。DQfD示范包含198个观测风险触发的WAIT标签，而原组合遇风险拒绝标签；教学量和教师政策均不完全匹配。旧500与本次新500分开，不挑最佳检查点，不据新500调参。
等待、循环与碰撞必须依据轨迹；静态可达不代表动态策略成功。未来障碍仅用于事后诊断。固定地图、同起终点、3–5动态障碍不能支持跨地图或普遍优于现有方法的主张。历史未记录的成本留空，不推测。
'''
    (output/'GPT_ANALYSIS_PROMPT.md').write_text(prompt, encoding='utf-8')
    file_hashes = {p.relative_to(output).as_posix(): common.sha256(p)
                   for p in output.rglob('*') if p.is_file()}
    write_json(output/'manifest.json', dict(complete=True, files_sha256=file_hashes,
        analysis=relative(analysis), final_evaluation=relative(evaluation),
        statistical_unit='five paired training seeds', source_weights_uploaded=False))
    verify(output)
    print(output)
    return output


def verify(output):
    manifest = common.read_json(output/'manifest.json')
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual != set(manifest['files_sha256']):
        raise ValueError('Package files differ from manifest.')
    for name, digest in manifest['files_sha256'].items():
        path = output/name
        if common.sha256(path) != digest or path.stat().st_size >= 100*1024**2:
            raise ValueError('Package hash mismatch or GitHub oversized file.')
        if path.suffix in ('.pt', '.pth', '.zip'):
            raise ValueError('Weights/archives must stay local.')
    for name, digest in common.read_json(output/'external_artifact_sha256.json').items():
        if common.sha256(ROOT/name) != digest:
            raise ValueError('Linked artifact checksum changed.')
    import re
    for path in (output/'README.md', output/'GPT_ANALYSIS_PROMPT.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not target.startswith(('https://', 'http://')) and not (path.parent/target).is_file():
                raise ValueError(f'Broken reading-package link: {target}')
    print(json.dumps(dict(verified=True, files=len(manifest['files_sha256']),
        size_mib=sum((output/p).stat().st_size for p in manifest['files_sha256'])/2**20)))


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
    else:
        package((args.analysis or latest('analysis')).resolve(),
                (args.evaluation or latest('independent_eval')).resolve())


if __name__ == '__main__':
    main()

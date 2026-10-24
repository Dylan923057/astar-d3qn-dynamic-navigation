"""Build and verify the completed persistent-ASVC paper evidence bundle; no training or evaluation."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from pathlib import Path
import shutil
import statistics

from package_persistent_supervision import ROOT, read, write, rows, write_csv, sha, require, local, relative

NAME = 'asvc_persistent_paper_completion_v1'
LIVE = ROOT / 'results' / NAME
BASE = ROOT / 'results/analysis_upload' / NAME
MODULES = ('persistent_asvc', 'no_td', 'no_advice', 'no_supervision')
DENSITY = ('persistent_asvc', 'original_asvc', 'dqfd', 'd3qn', 'astar_prefill')
RATES = ('safe_success_rate', 'dynamic_collision_rate', 'timeout_rate')
LEARNING = ('full_aulc', 'late_aulc', 'late_min', 'final_validation', 'final500_success', 'final500_collision', 'final500_timeout')
ALLOW = {'.json', '.csv', '.md', '.py', '.yaml', '.png', '.pdf', '.svg', '.ps1'}


def close(a, b, context):
    require(abs(float(a)-float(b)) < 1e-10, f'Statistics mismatch: {context}')


def aulc(curve, start, end):
    xs = [int(r['environment_steps_total']) for r in curve]
    ys = [float(r['safe_success_rate']) for r in curve]
    require(xs[0] <= start < end <= xs[-1], 'No extrapolation of learning curves.')
    def at(t):
        i = bisect_left(xs, t)
        if xs[i] == t:
            return ys[i]
        return ys[i-1] + (ys[i]-ys[i-1])*(t-xs[i-1])/(xs[i]-xs[i-1])
    points = [start] + [x for x in xs if start < x < end] + [end]
    return sum((b-a)*(at(a)+at(b))*.5 for a, b in zip(points, points[1:]))/(end-start)


def check_means(per_seed, summaries, methods, keys, density=None):
    require({r['method'] for r in summaries} == set(methods) and len(summaries) == len(methods), 'Method means incomplete.')
    for method in methods:
        selected = [r for r in per_seed if r['method'] == method]
        require(len(selected) == 5 and {int(r['seed']) for r in selected} == set(range(5)), 'Keep five seeds.')
        mean = next(r for r in summaries if r['method'] == method)
        for key in keys:
            values = [float(r[key]) for r in selected]
            close(statistics.mean(values), mean[key + '_mean'], (method, density, key, 'mean'))
            close(statistics.stdev(values), mean[key + '_sd'], (method, density, key, 'sample SD'))


def verify(package, check_sources=False):
    package = package.resolve()
    manifest = read(package / 'manifest.json')
    expected = manifest['files_sha256']
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual == set(expected) | {'manifest.json'}, 'Bundle file list differs.')
    for name, digest in expected.items():
        path = (package / name).resolve()
        require(path.is_relative_to(package) and path.suffix.lower() in ALLOW, f'Invalid bundle file: {name}')
        require(path.stat().st_size < 50*1024**2 and sha(path) == digest, f'File size/checksum mismatch: {name}')
    require(manifest['complete'] and manifest['new_training_runs'] == 10 and not manifest['weights_included'], 'Wrong package scope.')
    model_rows = rows(package / 'model_index.csv')
    require(len(model_rows) == 40 and len({(r['method'], int(r['seed'])) for r in model_rows}) == 40, 'Expected 40 distinct final models.')
    models = {(r['method'], int(r['seed'])): r for r in model_rows}
    scenes = {'final500': read(package / 'inputs/independent_final_scenarios.json')['scenarios']}
    scene_sha = {'final500': sha(package / 'inputs/independent_final_scenarios.json')}
    for n in (1, 3, 5, 7):
        p = package / 'inputs/density' / f'density_{n}.json'
        scenes[f'density_{n}'] = read(p)['scenes']
        scene_sha[f'density_{n}'] = sha(p)
        require(all(len(r['obstacles']) == n for r in scenes[f'density_{n}']), 'Wrong density.')
        require(read(package / 'inputs/density/density_registration.json')['files_sha256'][p.name] == sha(p), 'Density freeze mismatch.')
    for value in scenes.values():
        require(len(value) == len({r['scenario_id'] for r in value}) == 500, 'Expected 500 unique scenarios per suite.')
    for n in (1, 3, 5):
        require(all(a['obstacles'] == b['obstacles'][:n] for a, b in zip(scenes[f'density_{n}'], scenes['density_7'])), 'Density pairing changed.')
    index = rows(package / 'evaluation_index.csv')
    expected_suites = {('final500', m, s) for m in MODULES for s in range(5)} | {
        (f'density_{n}', m, s) for n in (1, 3, 5, 7) for m in DENSITY for s in range(5)}
    require(len(index) == 120 and {(r['suite'], r['method'], int(r['seed'])) for r in index} == expected_suites, 'Missing/duplicate evaluation suites.')
    evaluated = {}
    new_suites = 0
    for record in index:
        suite, method, seed = record['suite'], record['method'], int(record['seed'])
        path = package / 'evaluation' / suite / method / f'seed_{seed}'
        reg, done, summary = (read(path / name) for name in ('registration.json', 'completion.json', 'summary.json'))
        require(done['complete'] and reg['scene_sha256'] == scene_sha[suite], 'Evaluation scene/completion mismatch.')
        require(reg['model_sha256'] == models[(method, seed)]['model_sha256'], 'Evaluation uses another model.')
        require(reg['epsilon'] == 0 and reg['static_mask'] and reg['device'] == 'cpu' and reg['checkpoint_steps'] == 200000, 'Evaluation protocol differs.')
        require(reg['max_steps'] == 300 and reg['window_size'] == 15, 'Evaluation episode/window differs.')
        require(not any(reg[k] for k in ('teacher', 'route_guidance', 'dynamic_shield', 'q_output_clip', 'used_for_tuning')), 'Evaluation intervention/tuning detected.')
        details = rows(path / 'scene_details.csv')
        require([r['scenario_id'] for r in details] == [r['scenario_id'] for r in scenes[suite]], 'Scene IDs/order differ.')
        rates = dict(safe_success_rate=sum(float(r['safe_success']) for r in details)/500,
                     dynamic_collision_rate=sum(float(r['dynamic_collision']) for r in details)/500,
                     timeout_rate=sum(r['termination_reason'] == 'timeout' for r in details)/500)
        for key, value in rates.items():
            close(summary[key], value, (suite, method, seed, key))
        evaluated[(suite, method, seed)] = rates
        is_new = (suite == 'final500' and method in ('no_td', 'no_advice')) or (suite != 'final500' and method == 'persistent_asvc')
        require(done['reused'] == (not is_new), 'New/reused evaluation counts differ.')
        new_suites += int(is_new)
    require(new_suites == 30, 'Expected 30 new suites (15000 episodes), 90 reused suites.')
    metrics = rows(package / 'summary/ablation_per_seed.csv')
    require(len(metrics) == 20 and {(r['method'], int(r['seed'])) for r in metrics} == {(m, s) for m in MODULES for s in range(5)}, 'Ablation rows differ.')
    validation_total = 0
    for metric in metrics:
        method, seed = metric['method'], int(metric['seed'])
        path = package / 'run_logs' / method / f'seed_{seed}'
        done, result = read(path / 'completion.json'), read(path / 'result.json')
        require(done['complete'] and result['environment_steps'] == 200000 and result['gradient_updates'] == 199501, 'Incomplete/wrong-budget ablation run.')
        curve, details = rows(path / 'validation_curve.csv'), rows(path / 'validation_details.csv')
        steps = [int(r['environment_steps_total']) for r in curve]
        require(len(steps) == 21 and steps[0] == 0 and steps[-1] == 200000 and steps == sorted(set(steps)), 'Incomplete validation curve.')
        require(len(details) == 1050, 'Expected 21 x 50 validation rows.')
        for point in curve:
            selected = [r for r in details if int(r['environment_steps_total']) == int(point['environment_steps_total'])]
            require(len(selected) == len({r['scenario_id'] for r in selected}) == 50, 'Validation scenarios incomplete.')
            close(sum(float(r['safe_success']) for r in selected)/50, point['safe_success_rate'], 'Validation success')
        validation_total += len(details)
        close(aulc(curve, 0, 200000), metric['full_aulc'], 'Full AULC')
        close(aulc(curve, 100000, 200000), metric['late_aulc'], 'Late AULC')
        close(min(float(r['safe_success_rate']) for r in curve if int(r['environment_steps_total']) >= 100000), metric['late_min'], 'Late minimum')
        close(curve[-1]['safe_success_rate'], metric['final_validation'], 'Final validation')
        for key, evaluation_key in zip(('final500_success', 'final500_collision', 'final500_timeout'), RATES):
            close(metric[key], evaluated[('final500', method, seed)][evaluation_key], key)
        if method in ('no_td', 'no_advice'):
            frozen = read(package / 'registration' / method / 'freeze_manifest.json')
            require(result['source_sha256'] == frozen['source_sha256'] and result['config'] == frozen['config'], 'Training freeze differs.')
            require(result['freeze_manifest_sha256'] == sha(package / 'registration' / method / 'freeze_manifest.json'), 'Training registration digest differs.')
            require(result['final_model_sha256'] == models[(method, seed)]['model_sha256'], 'Final model digest differs.')
            initial = next(r for r in frozen['checks']['initializations'] if r['seed'] == seed)
            require(result['initial_state_sha256'] == initial['initial_state_sha256'], 'Initial pairing differs.')
            prefix = read(path / 'prefix_parity.json')
            require(len(prefix) == 9 and all(r['exact_weights_match'] for r in prefix), 'Historical early-checkpoint pairing failed.')
            advice, values = rows(path / 'advice_budget.csv'), rows(path / 'value_learning.csv')
            require(all(abs(float(r['teacher_margin_weight_mean'])-.1) < 1e-10 for r in values if int(r['bin_start_step']) >= 100001), 'Supervision floor differs.')
            if method == 'no_td':
                require(all(float(r['target_clip_fraction_mean']) == 0 for r in values), 'TD clipping not removed.')
            else:
                require(all(int(r['requested']) == int(r['applied']) == 0 for r in advice), 'Advice not removed.')
    check_means(metrics, rows(package / 'summary/ablation_means.csv'), MODULES, LEARNING)
    for pair in rows(package / 'summary/ablation_paired_differences.csv'):
        seed, method = int(pair['seed']), pair['method']
        full = next(r for r in metrics if r['method'] == 'persistent_asvc' and int(r['seed']) == seed)
        removed = next(r for r in metrics if r['method'] == method and int(r['seed']) == seed)
        for key in ('late_aulc', 'late_min', 'final500_success'):
            close(float(full[key])-float(removed[key]), pair[key + '_full_minus_removed'], 'Paired differences')
    density = rows(package / 'summary/density_per_seed.csv')
    require(len(density) == 100, 'Expected 100 density/seed rows.')
    for record in density:
        key = (f'density_{record["density"]}', record['method'], int(record['seed']))
        for rate in RATES:
            close(record[rate], evaluated[key][rate], key)
    density_means = rows(package / 'summary/density_means.csv')
    for n in (1, 3, 5, 7):
        check_means([r for r in density if int(r['density']) == n],
            [r for r in density_means if int(r['density']) == n], DENSITY, RATES, density=n)
    if check_sources:
        for name, digest in (manifest['source_files_sha256'] | manifest['models_sha256']).items():
            require(sha(local(name)) == digest, f'Local source changed after packaging: {name}')
    total = sum((package / name).stat().st_size for name in expected)
    print(f'PASS: {len(expected)} SHA-verified files; 10 new complete trainings; 120 evaluation suites / 60000 episodes; {validation_total} validation rows.')
    print(f'Bundle: {total/1024**2:.2f} MiB. New evaluations: 15000 episodes; reused: 45000. No training/evaluation/push performed.')
    return manifest


def build():
    pipeline = read(LIVE / 'pipeline_status.json')
    require(pipeline['complete'] and pipeline['all_ten_trainings_succeeded'] and not pipeline['numerical_failures'], 'This package requires all ten completed runs.')
    analysis = local(pipeline['report']).parent
    destination = BASE / analysis.name
    if destination.exists():
        verify(destination, check_sources=True)
        return destination
    evidence = read(analysis / 'evidence.json')
    require(evidence['complete'] and evidence['suite'] == 'all', 'Require both ablation and density analyses.')
    require(sha(ROOT / 'scripts/analyze_persistent_paper.py') == evidence['analysis_source_sha256'], 'Analysis source changed.')
    require(sha(ROOT / 'scripts/evaluate_persistent_paper.py') == evidence['evaluation_source_sha256'], 'Evaluation source changed.')
    reuse = read(ROOT / 'data' / NAME / 'reuse_registration.json')
    hashes = dict(evidence['sources_sha256']) | reuse['source_sha256'] | reuse['reused_model_sha256']
    models = {}
    for name, digest in hashes.items():
        path = local(name)
        require(sha(path) == digest, f'Registered evidence changed: {name}')
        if path.suffix == '.pth':
            models[relative(path)] = digest
    require(len(models) == 40, 'Expected forty distinct models, including reused comparisons.')
    destination.mkdir(parents=True)
    sources = {}
    def copy(source, target):
        source = Path(source)
        require(source.suffix.lower() in ALLOW, f'Unsupported upload file: {source}')
        sources[relative(source)] = sha(source)
        target = destination / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for path in sorted(analysis.iterdir()):
        if path.is_file():
            copy(path, Path('summary') / path.name)
    for path in sorted((LIVE / 'evaluation').rglob('*')):
        if path.is_file():
            copy(path, Path('evaluation') / path.relative_to(LIVE / 'evaluation'))
    for name in ('pipeline_status.json', 'evaluation_all_status.json', 'preflight.json'):
        copy(LIVE / name, Path('registration') / name)
    frozen_sources = {}
    for method in ('no_td', 'no_advice'):
        path = ROOT / 'data' / NAME / method / 'freeze_manifest.json'
        frozen = read(path)
        copy(path, Path('registration') / method / path.name)
        for name, digest in frozen['source_sha256'].items():
            name = name.replace('\\', '/')
            require(name not in frozen_sources or frozen_sources[name] == digest, 'Frozen source versions disagree.')
            frozen_sources[name] = digest
    copy(ROOT / 'data' / NAME / 'reuse_registration.json', 'registration/reuse_registration.json')
    for name, digest in frozen_sources.items():
        require(sha(ROOT / name) == digest, f'Frozen training source changed: {name}')
        copy(ROOT / name, Path('source_snapshot') / name)
    extras = ['scripts/evaluate_persistent_paper.py', 'scripts/analyze_persistent_paper.py',
        'scripts/package_persistent_paper_completion.py', 'scripts/package_persistent_supervision.py',
        'scripts/sync_persistent_paper_github.ps1']
    for name in extras:
        copy(ROOT / name, Path('source_snapshot') / name)
    copy(ROOT / 'data/whole_map_91701_supervision_tail_v1/independent_final_scenarios.json', 'inputs/independent_final_scenarios.json')
    copy(ROOT / frozen['config']['dataset']['validation_reference'], 'inputs/fixed_validation_scenarios.json')
    for key in ('source_manifest', 'route_pool_manifest'):
        path = ROOT / frozen['config']['dataset'][key]
        copy(path, Path('inputs') / path.parent.name / path.name)
    for name in ('density_registration.json', 'density_1.json', 'density_3.json', 'density_5.json', 'density_7.json'):
        copy(ROOT / 'data/asvc_paper_density_v2' / name, Path('inputs/density') / name)
    aliases = dict(supervision_floor='persistent_asvc', no_td='no_td', no_advice='no_advice', advice_bound='no_supervision',
        advice_bound_margin_all_eligible='original_asvc', dqfd='dqfd', unguided_raw='d3qn', astar_prefill_bound='astar_prefill')
    model_index, diagnostics = [], []
    for name, digest in sorted(models.items()):
        path = ROOT / name
        directory = path.parent
        seed = int(directory.name.split('_')[-1])
        method = aliases[directory.parent.name]
        result = read(directory / 'result.json')
        model_index.append(dict(method=method, seed=seed, model_path=name, model_sha256=digest,
            environment_steps=result['environment_steps'], online_updates=result.get('online_updates', result['gradient_updates']),
            pretraining_updates=result.get('pretraining_updates', 0), new_training=method in ('no_td', 'no_advice')))
        if method not in MODULES:
            continue
        required = ['run_manifest.json', 'result.json', 'completion.json', 'validation_curve.csv', 'validation_details.csv']
        optional = ['advice_budget.csv', 'value_learning.csv', 'supervision_alignment.csv', 'failure_behaviors.csv', 'prefix_parity.json']
        if method in ('no_td', 'no_advice'):
            required += ['training.csv', 'prefix_parity.json']
        for item in dict.fromkeys(required + optional):
            if item in required or (directory / item).exists():
                copy(directory / item, Path('run_logs') / method / f'seed_{seed}' / item)
        values = rows(directory / 'value_learning.csv')
        has_behaviors = (directory / 'failure_behaviors.csv').exists()
        behaviors = rows(directory / 'failure_behaviors.csv') if has_behaviors else []
        timed_out = [r for r in behaviors if int(r['environment_steps_total']) == 200000 and r['termination_reason'] == 'timeout']
        diagnostics.append(dict(method=method, seed=seed, training_q_abs_peak=max(float(r['q_abs_max']) for r in values),
            training_raw_target_peak=max(float(r['raw_target_max']) for r in values),
            final50_timeouts=len(timed_out) if has_behaviors else None,
            final50_repeated_movement_timeouts=sum(r['repeated_movement'] == 'True' for r in timed_out) if has_behaviors else None,
            final50_waiting_dominated_timeouts=sum(r['waiting_dominated'] == 'True' for r in timed_out) if has_behaviors else None))
    write_csv(destination / 'model_index.csv', model_index)
    write_csv(destination / 'diagnostics_per_seed.csv', diagnostics)
    evaluations = []
    for path in sorted((destination / 'evaluation').glob('*/*/seed_*')):
        reg, done = read(path / 'registration.json'), read(path / 'completion.json')
        evaluations.append(dict(suite=reg['suite'], method=reg['method'], seed=reg['seed'],
            reused=done['reused'], scene_count=500, model_sha256=reg['model_sha256'], scene_sha256=reg['scene_sha256'],
            package_path=path.relative_to(destination).as_posix()))
    write_csv(destination / 'evaluation_index.csv', evaluations)
    scope = sorted(set(list(frozen_sources) + extras + ['.gitattributes', 'docs/GITHUB_SYNC.zh-CN.md',
        'docs/ASVC_CURRENT.zh-CN.md', 'docs/ASVC_PERSISTENT_PAPER_COMPLETION.zh-CN.md',
        'data/' + NAME, relative(BASE), relative(analysis), 'results/' + NAME + '/LATEST.md',
        'results/' + NAME + '/preflight.json', 'results/' + NAME + '/pipeline_status.json',
        'results/' + NAME + '/PROGRESS.md', 'results/' + NAME + '/evaluation_all_status.json']))
    write(destination / 'upload_scope.json', scope)
    copy(ROOT / 'docs/ASVC_PERSISTENT_PAPER_COMPLETION.zh-CN.md', 'protocol_and_commands.md')
    protocol_copy = destination / 'protocol_and_commands.md'
    protocol_copy.write_text(protocol_copy.read_text(encoding='utf-8').replace('](../', '](../../../../'), encoding='utf-8')
    (destination / 'README.md').write_text(README, encoding='utf-8')
    write(destination / 'manifest.json', dict(complete=True, new_training_runs=10, new_environment_steps=2000000,
        ablation_methods=4, ablation_seeds=5, density_methods=5, density_levels=[1, 3, 5, 7],
        evaluated_model_suites=120, evaluation_episodes=60000, new_evaluation_episodes=15000,
        reused_evaluation_episodes=45000, weights_included=False, statistical_unit='training_seed',
        source_files_sha256=sources, models_sha256=models,
        files_sha256={p.relative_to(destination).as_posix(): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}))
    verify(destination, check_sources=True)
    write(BASE / 'latest.json', dict(package=relative(destination)))
    (BASE / 'LATEST.md').write_text(f'# 新版模块消融与密度结果\n\n[阅读说明]({destination.name}/README.md) · [详细报告]({destination.name}/summary/REPORT.md) · [模块图]({destination.name}/summary/01_module_learning_curves.png) · [密度图]({destination.name}/summary/04_density_comparison.png)\n', encoding='utf-8')
    print(destination)
    return destination


README = '''# 持续弱监督ASVC：新版模块消融与密度测试

[原始报告](summary/REPORT.md) · [模块逐seed数据](summary/ablation_per_seed.csv) · [密度逐seed数据](summary/density_per_seed.csv) · [诊断数据](diagnostics_per_seed.csv)

2026-10-24全部完成：新训练10次，各20万环境步，无数值非有限值中止；新增评估30组共15000回合，复用90组45000回合。合计120组60000条评估记录。没有为本次打包重新训练或评估。

## 先看结论

|方法|全程AULC|后半程AULC|后半程最低成功率|最终500成功率|
|---|---:|---:|---:|---:|
|新版完整方法|95.63±0.38%|99.54±0.34%|96.80±1.79%|99.28±0.73%|
|去TD目标约束|50.65±17.98%|39.73±31.00%|4.00±8.94%|61.24±38.08%|
|去A*动作代选|86.93±1.38%|99.28±0.13%|96.80±1.10%|98.72±0.39%|
|去动作监督（复用）|60.44±2.30%|97.52±1.08%|87.60±5.90%|98.92±1.32%|

这些是五个训练seed的均值±样本标准差。AULC是按环境步数积分的曲线平均成功率；最终500与反复使用的50场景验证分开统计。

- TD约束对稳定性贡献最大：去掉后五seed后半程均劣于完整方法。最终500逐seed成功率为98.2%、80.8%、40.4%、4.8%、82.0%，不能只展示恢复较好的seed0。训练Q绝对值峰值约38.21、125.34、278.98、1829.33、56.16；这是价值估计膨胀的诊断证据，不等同于浮点数值崩溃。
- A*动作代选主要改善早期学习：去掉后全程AULC降低8.70个百分点，但后半程仅低0.26个百分点，最低成功率均值相同，最终500低0.56个百分点。不能据此声称没有代选就无法学会，或每个seed的后期表现都更差。
- 动作监督主要改善学习速度，并提高后期稳定性：去掉后全程AULC降低35.19个百分点，后半程低2.02个百分点，最低值均值低9.20个百分点；最终500只低0.36个百分点。部分seed最终成绩更好，不能把差异包装成所有指标都大幅领先。

![模块曲线](summary/01_module_learning_curves.png)

## 障碍密度结果与取舍

|方法|1个障碍|3个障碍|5个障碍|7个障碍|
|---|---:|---:|---:|---:|
|新版完整方法|100.00±0.00%|99.32±0.18%|98.12±0.39%|95.68±0.83%|
|原版ASVC|99.96±0.09%|99.68±0.30%|98.32±0.73%|96.64±1.52%|
|DQfD|99.60±0.51%|98.52±1.94%|97.04±2.70%|95.36±3.24%|
|D3QN|99.44±0.55%|97.80±1.70%|95.72±2.76%|92.20±4.15%|
|A*初始化＋TD约束|99.28±1.00%|98.48±1.10%|96.68±1.95%|94.80±2.03%|

新版四档密度的10000回合没有观察到超时；但3、5、7障碍成功率均低于原版。7障碍时新版碰撞率4.32%、原版2.96%，超时率分别0%和0.40%，最终成功率相差−0.96个百分点。新版主要解决训练过程的回落，没有证明密度泛化优于旧版。

7障碍时新版与DQfD成功率仅相差+0.32个百分点；DQfD碰撞率更低、超时率更高。这里只报告描述性差异，不据五seed均值宣称统计显著或全面更安全。

![密度图](summary/04_density_comparison.png)

## 实验版本与复用边界

完整方法A*代选在10万步退出，动作监督权重在9万步降到0.1后保留。两个新消融都保持这一监督日程：去TD只取消目标裁剪，去代选只取消动作替换，仍用风险筛选后的教师标签计算监督损失。奖励、观测、网络、回放和训练预算不变。

去动作监督复用旧advice_bound五seed；训练配置、初始化及关闭监督后的网络/目标网络/优化器更新等价性已核对。新版完整方法训练及最终500结果复用此前结果。旧版ASVC、DQfD、D3QN和A*初始化密度结果按模型、场景和评估协议校验后复用。每个新消融在0至8万步的九个检查点均与对应旧消融的参数完全一致。

固定使用20万步最终模型，不挑最佳检查点；评估epsilon=0、仅静态非法动作屏蔽，无教师/A*代选、动态安全屏蔽或未来障碍信息。所有seed和检查点保留。统计单位是训练seed，500场景不是500个独立训练重复。固定地图和起终点的密度测试不代表跨地图泛化。DQfD另有2万次离线更新、永久示范及PER，在线步数相同不等于总计算预算相同。

## 文件导航与校验

- `summary/`：原始报告、五seed均值/样本标准差、逐seed配对差和5组PNG/PDF/SVG图。
- `run_logs/`：四个模块方法共20次运行的420个验证点、21000条验证明细；10次新训练含完整训练CSV，另有监督/代选/Q值诊断。
- `evaluation/`：模块最终500的20组、密度的100组，总计60000条场景明细；每组保留完成标记、模型/场景SHA及新评估/复用来源。
- `registration/`：两个新消融的训练前冻结协议、复用核对、完成状态。
- `inputs/`：原50验证、原500最终场景、四档密度v2场景、地图及动态路线登记。
- `source_snapshot/`：对应训练/评估/分析代码、配置与测试快照。
- `model_index.csv`：40份最终模型路径、SHA、在线及离线更新预算；模型权重和恢复文件不包含在本包。
- `evaluation_index.csv`：120组评估的来源索引。
- `diagnostics_per_seed.csv`：Q值峰值和最终50验证的超时行为，仅用于诊断。
- `manifest.json`：完整文件清单与SHA；`upload_scope.json`：本次Git提交范围。

仓库根目录校验：`python scripts/package_persistent_paper_completion.py --verify`。该命令只需标准库，克隆后不需要模型文件。原工作空间可加`--check-sources`核对本地原始数据和权重。阅读和校验不训练、不评估、不推送；重新执行模型评估仍需本地权重。

上传命令与运行说明见仓库`docs/GITHUB_SYNC.zh-CN.md`和`docs/ASVC_PERSISTENT_PAPER_COMPLETION.zh-CN.md`。旧版报告、模型和原始轨迹继续保留，没有覆盖或改名冒充新版结果。
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--check-sources', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(local(read(BASE / 'latest.json')['package']), args.check_sources)
    else:
        build()

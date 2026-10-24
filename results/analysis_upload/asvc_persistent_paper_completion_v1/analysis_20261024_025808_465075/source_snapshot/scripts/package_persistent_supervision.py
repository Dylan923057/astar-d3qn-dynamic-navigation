"""Package completed ASVC results for GitHub; standard library only, no training/evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
NAME = 'asvc_persistent_supervision_five_seeds_v1'
BASE = ROOT / 'results/analysis_upload' / NAME
SELECTION = 'results/asvc_persistent_teaching_v1/analysis_20261023_034314_462299'
METHODS = ('original_asvc', 'persistent_asvc', 'dqfd', 'd3qn', 'astar_prefill')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def local(name):
    path = (ROOT / name.replace('\\', '/')).resolve()
    require(path.is_relative_to(ROOT), f'Path outside workspace: {name}')
    return path


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def verify(package, check_sources=False):
    package = package.resolve()
    manifest = read(package / 'manifest.json')
    expected = manifest['files_sha256']
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual == set(expected) | {'manifest.json'}, 'Package file list changed.')
    for name, digest in expected.items():
        path = (package / name).resolve()
        require(path.is_relative_to(package), f'Invalid package path: {name}')
        require(sha(path) == digest, f'Package checksum mismatch: {name}')
        require(path.suffix.lower() in {'.json', '.csv', '.md', '.py', '.yaml', '.png', '.pdf', '.svg', '.ps1'}, name)
        require(path.stat().st_size < 50 * 1024**2, f'File too large for this bundle: {name}')
    scene_ids = [r['scenario_id'] for r in read(package / 'inputs/independent_final_scenarios.json')['scenarios']]
    require(len(scene_ids) == len(set(scene_ids)) == 500, 'Expected 500 unique frozen scenarios.')
    metrics = rows(package / 'summary/per_seed.csv')
    require({(r['method'], int(r['seed'])) for r in metrics} == {(m, s) for m in METHODS for s in range(5)}, 'Missing method/seed.')
    for record in metrics:
        method, seed = record['method'], int(record['seed'])
        details = rows(package / f'final500/{method}/seed_{seed}/scene_details.csv')
        require([r['scenario_id'] for r in details] == scene_ids, f'Scene mismatch: {method}/{seed}')
        rates = dict(final500_success=sum(float(r['safe_success']) for r in details)/500,
                     final500_collision=sum(float(r['dynamic_collision']) for r in details)/500,
                     final500_timeout=sum(r['termination_reason'] == 'timeout' for r in details)/500)
        require(all(abs(float(record[k])-v) < 1e-12 for k, v in rates.items()), f'Final500 statistics mismatch: {method}/{seed}')
        path = package / f'run_logs/{method}/seed_{seed}'
        require(read(path / 'completion.json')['complete'], f'Incomplete run: {method}/{seed}')
        require(read(path / 'result.json')['environment_steps'] == 200000, 'Wrong training budget.')
        curve = rows(path / 'validation_curve.csv')
        require(len(curve) == 21 and int(curve[-1]['environment_steps_total']) == 200000, 'Incomplete curve.')
        minimum = min(float(r['safe_success_rate']) for r in curve if int(r['environment_steps_total']) >= 100000)
        require(abs(minimum-float(record['late_min'])) < 1e-12, 'Minimum differs from raw curve.')
        if method == 'persistent_asvc':
            parity = read(path / 'prefix_parity.json')
            require(len(parity) == 9 and all(r['exact_weights_match'] for r in parity), 'Initial training parity failed.')
            for filename, key, floor in [('advice_budget.csv', 'mean_requested_probability', 0.),
                                         ('value_learning.csv', 'teacher_margin_weight_mean', .1)]:
                tail = [float(r[key]) for r in rows(path / filename) if int(r['bin_start_step']) >= 100001]
                require(tail and all(abs(v-floor) < 1e-10 for v in tail), 'Teaching schedule mismatch.')
    for mean in rows(package / 'summary/means.csv'):
        values = [float(r['final500_success']) for r in metrics if r['method'] == mean['method']]
        require(abs(statistics.mean(values)-float(mean['final500_success_mean'])) < 1e-12, 'Mean differs from scene data.')
        require(abs(statistics.stdev(values)-float(mean['final500_success_sd'])) < 1e-12, 'Sample SD differs from scene data.')
    if check_sources:
        for name, digest in manifest['source_files_sha256'].items():
            require(sha(local(name)) == digest, f'Source changed after packaging: {name}')
    total = sum((package / name).stat().st_size for name in expected)
    print(f'PASS: {len(expected)} SHA-verified files; 25 model/seed final evaluations, 12500 scene rows; {total/1024**2:.2f} MiB.')
    print('Includes five candidate seeds and three-seed selection history. No weights, training or push.')
    return manifest


def package_results():
    current = read(ROOT / 'data/asvc_current_method.json')
    review = local(current['source_review'])
    evidence = read(review / 'evidence.json')
    require(evidence['complete'] and evidence['seeds'] == list(range(5)), 'Five-seed review incomplete.')
    require(sha(review / 'decision.json') == current['decision_sha256'], 'Current-method decision changed.')
    require(sha(ROOT / 'scripts/review_persistent_supervision_five_seeds.py') == evidence['analysis_source_sha256'], 'Review code changed.')
    destination = BASE / review.name
    if destination.exists():
        verify(destination, check_sources=True)
        return destination
    sources = {}
    models = {}
    for report in (review / 'evidence.json', ROOT / SELECTION / 'evidence.json'):
        for name, digest in read(report)['source_sha256'].items():
            path = local(name)
            require(sha(path) == digest, f'Experiment evidence changed: {name}')
            if path.suffix == '.pth':
                models[relative(path)] = digest
    destination.mkdir(parents=True)

    def copy(source, target):
        source = Path(source)
        sources[relative(source)] = sha(source)
        target = destination / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    for origin, target in ((review, 'summary'), (ROOT / SELECTION, 'selection_three_seeds')):
        for path in sorted(origin.iterdir()):
            if path.is_file():
                copy(path, Path(target) / path.name)
    copy(ROOT / 'data/asvc_current_method.json', 'current_method.json')
    frozen_sources = {}
    for study in ('asvc_persistent_teaching_v1', NAME):
        path = ROOT / 'data' / study / 'freeze_manifest.json'
        frozen = read(path)
        copy(path, f'registration/{study}/freeze_manifest.json')
        frozen_sources.update({name.replace('\\', '/'): digest for name, digest in frozen['source_sha256'].items()})
    for name, digest in frozen_sources.items():
        require(sha(local(name)) == digest, f'Frozen training source changed: {name}')
        copy(local(name), Path('source_snapshot') / name)
    extras = ['scripts/analyze_persistent_teaching.py', 'scripts/review_persistent_supervision_five_seeds.py',
              'scripts/watch_persistent_teaching.py', 'scripts/watch_persistent_supervision_five_seeds.py',
              'scripts/package_persistent_supervision.py', 'scripts/sync_github.ps1']
    for name in extras:
        copy(ROOT / name, Path('source_snapshot') / name)
    validation = local(frozen['config']['dataset']['validation_reference'])
    require(sha(validation) == frozen['validation_sha256'], 'Frozen validation changed.')
    copy(validation, 'inputs/fixed_validation_scenarios.json')
    scene_file = ROOT / 'data/whole_map_91701_supervision_tail_v1/independent_final_scenarios.json'
    require(sha(scene_file) == evidence['final500_verification']['scene_sha256'], 'Frozen final500 changed.')
    copy(scene_file, 'inputs/independent_final_scenarios.json')
    for name in ('source_manifest', 'route_pool_manifest'):
        source = local(frozen['config']['dataset'][name])
        copy(source, Path('inputs') / source.parent.name / source.name)

    # Keep complete scalar logs for new runs, validation details for every comparison.
    index = []
    for model_name, digest in models.items():
        source = local(model_name).parent
        seed = int(source.name.split('_')[-1])
        arm = source.parent.name
        method = dict(advice_bound_margin_all_eligible='original_asvc', supervision_floor='persistent_asvc',
                      dqfd='dqfd', unguided_raw='d3qn', astar_prefill_bound='astar_prefill',
                      advice_floor='selection_advice_floor', both_floors='selection_both_floors')[arm]
        target = Path('run_logs') / method / f'seed_{seed}'
        required = ['completion.json', 'result.json', 'run_manifest.json', 'validation_curve.csv', 'validation_details.csv']
        optional = ['advice_budget.csv', 'value_learning.csv', 'supervision_alignment.csv',
                    'failure_behaviors.csv', 'dqfd_learning.csv', 'prefix_parity.json']
        if method == 'persistent_asvc' or method.startswith('selection_'):
            required += ['training.csv', 'prefix_parity.json']
        for name in dict.fromkeys(required + optional):
            if name in required or (source / name).exists():
                copy(source / name, target / name)
        index.append(dict(method=method, seed=seed, model_path=model_name, model_sha256=digest,
                          run_logs=target.as_posix(), environment_steps=200000))
    require(len(index) == 31, 'Expected 25 main runs and 6 additional selection runs.')
    write_csv(destination / 'model_index.csv', index)

    old = local(read(ROOT / 'results/latest_all_eligible_modules_final500_v1/latest.json')['evaluation'])
    old_audit = read(old / 'verification.json')
    require(old_audit['frozen_scene_sha256'] == sha(scene_file), 'Historical scenes differ.')
    copy(old / 'verification.json', 'final500/historical_verification.json')
    historical_rows = rows(old / 'scene_details.csv')
    historical_summaries = rows(old / 'summary.csv')
    # The combined historical file contains unrelated ablations; only publish selected rows.
    sources[relative(old / 'scene_details.csv')] = sha(old / 'scene_details.csv')
    sources[relative(old / 'summary.csv')] = sha(old / 'summary.csv')
    keys = dict(original_asvc='advice_bound_margin_all_eligible', dqfd='dqfd', d3qn='unguided_raw')
    for record in index:
        method, seed = record['method'], record['seed']
        if method.startswith('selection_'):
            continue
        target = Path('final500') / method / f'seed_{seed}'
        if method in keys:
            require(old_audit['source_model_sha256'][record['model_path']] == record['model_sha256'], 'Old model mismatch.')
            selected = [r for r in historical_rows if r['method'] == keys[method] and int(r['seed']) == seed]
            write_csv(destination / target / 'scene_details.csv', selected)
            summary = next(r for r in historical_summaries if r['method'] == keys[method] and int(r['seed']) == seed)
            write(destination / target / 'summary.json', summary)
            write(destination / target / 'provenance.json', dict(source=relative(old),
                  scene_details_sha256=sha(old / 'scene_details.csv'), model_sha256=record['model_sha256']))
        else:
            origin = (ROOT / 'results' / NAME / 'final500' / f'seed_{seed}' if method == 'persistent_asvc'
                      else ROOT / 'results/asvc_paper_supplement_v1/evaluation_v2/original500/astar_prefill_bound' / f'seed_{seed}')
            reg = read(origin / 'registration.json')
            require(reg['model_sha256'] == record['model_sha256'] and reg['scene_sha256'] == sha(scene_file), 'Final500 registration mismatch.')
            require(read(origin / 'completion.json')['complete'], 'Incomplete final evaluation.')
            for name in ('scene_details.csv', 'summary.json', 'registration.json', 'completion.json'):
                copy(origin / name, target / name)

    scope = sorted(set(list(frozen_sources) + extras + [
        '.gitattributes', 'docs/GITHUB_SYNC.zh-CN.md', 'docs/ASVC_CURRENT.zh-CN.md',
        'docs/ASVC_PERSISTENT_TEACHING.zh-CN.md', 'docs/ASVC_PERSISTENT_SUPERVISION_FIVE_SEEDS.zh-CN.md',
        'data/asvc_current_method.json', 'data/asvc_persistent_teaching_v1', 'data/' + NAME,
        relative(review), SELECTION, 'results/' + NAME + '/LATEST.md', 'results/' + NAME + '/PROGRESS.md',
        relative(BASE)]))
    write(destination / 'upload_scope.json', scope)
    (destination / 'README.md').write_text('''# ASVC持续弱监督：五seed结果上传包

[结果报告](summary/REPORT.md) · [主对比图](summary/01_learning_curves.png) · [逐seed曲线](summary/02_individual_seeds.png) · [逐seed数据](summary/per_seed.csv) · [三seed方案选择历史](selection_three_seeds/REPORT.md)

五个seed各训练20万步：seed0–2复用先前结果，seed3–4新增复核。A*代选在10万步退出，监督权重降到0.1后保持；评估由网络独立行动。
后半程AULC由94.58%提高到99.54%，各seed后半程最低成功率的平均由73.60%提高到96.80%；最终500成功率由99.36%变为99.28%。主要改善是训练稳定性，最终成功率略低0.08个百分点。

## 文件说明

- `summary/`：五方法×五seed的报告、CSV及六组PNG/PDF/SVG图。
- `selection_three_seeds/`：三个候选方案的原三seed完整比较，保留选择依据。
- `run_logs/`：31次运行的曲线、逐场景验证和诊断；11次持续教学运行另含完整训练CSV。
- `final500/`：五方法×五seed共12500条评估明细，新方法2500回合新评估、四个对照10000回合复用已核对结果。
- `registration/`、`inputs/`、`source_snapshot/`：冻结协议、50/500场景与对应代码；`model_index.csv`记录最终模型路径和SHA。
- `manifest.json`：逐文件SHA及来源；`upload_scope.json`限定此次Git提交范围。

统计单位为训练seed。前三个seed参与了方案选择，后两个用于复核；固定地图及同一冻结500场景不代表跨地图泛化。旧消融和密度实验仍属于原版。
权重、回放恢复文件和批量失败轨迹保留本地。克隆仓库后可阅读全部图表并用下面命令校验；重新评估需要原权重，不能把缺权重当成自动重训的理由。

仓库根目录只读校验：`python scripts/package_persistent_supervision.py --verify`
本地从原始输出重画：`python scripts/review_persistent_supervision_five_seeds.py`（需要原始outputs及模型）。
上传命令见仓库`docs/GITHUB_SYNC.zh-CN.md`的持续弱监督条目。原训练冻结覆盖全部src文件，因此也保留其中两个奖励实验模块的源码字节；本包不包含奖励实验结果，也没有改变本方法奖励。
''', encoding='utf-8')
    write(destination / 'manifest.json', dict(complete=True, seeds=list(range(5)), main_runs=25,
          selection_extra_runs=6, final500_episodes=12500, candidate_final500_episodes=2500,
          source_files_sha256=sources, model_sha256=models, weights_included=False,
          files_sha256={p.relative_to(destination).as_posix(): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}))
    verify(destination, check_sources=True)
    write(BASE / 'latest.json', dict(package=relative(destination)))
    (BASE / 'LATEST.md').write_text(f'# 持续弱监督五seed阅读入口\n\n[说明]({destination.name}/README.md) · [报告]({destination.name}/summary/REPORT.md) · [曲线]({destination.name}/summary/01_learning_curves.png)\n', encoding='utf-8')
    print(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify the latest bundle without model files.')
    parser.add_argument('--check-sources', action='store_true', help='Also compare original local source files.')
    args = parser.parse_args()
    if args.verify:
        verify(local(read(BASE / 'latest.json')['package']), check_sources=args.check_sources)
    else:
        package_results()

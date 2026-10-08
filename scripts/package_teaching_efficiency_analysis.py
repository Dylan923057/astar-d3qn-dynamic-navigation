"""Prepare a small GitHub reading package from completed runs; no training or evaluation."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import teaching_efficiency_common as common
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from astar_d3qn.utils.io import write_json, write_records_csv


def package(review):
    config = common.legacy.load_config(common.CONFIG)
    problem = common.validate_config(config)[0]
    common.load_frozen(config)
    audit = common.read_json(review / 'verification.json')
    if (not audit['complete'] or audit['models'] != 35 or audit['paired_seeds'] != 5
            or audit['independent_scene_rows'] != 17500 or audit['full_raw_failure_trajectories_checked'] != 3496
            or audit['training_started'] or audit['suite_evaluation_started']):
        raise ValueError('Use the completed seven-method, five-seed review.')
    analysis = common.ROOT / audit['input_analysis']
    evaluation = common.ROOT / audit['input_evaluation']
    evaluated = common.read_json(evaluation / 'verification.json')
    raw_root = common.ROOT / evaluated['full_failure_trace_root']
    if common.read_json(raw_root / 'completion.json') != dict(complete=True, **evaluated):
        raise ValueError('Incomplete independent evaluation.')
    runs = {(method, seed): common.audit_run(config, method, seed)
            for method in common.ALL_METHODS for seed in common.SEEDS}
    weights = {str((directory / 'model_final.pth').relative_to(common.ROOT)): common.sha256(directory / 'model_final.pth')
               for directory, _ in runs.values()}
    if weights != audit['source_model_sha256'] or weights != evaluated['source_model_sha256']:
        raise ValueError('Reviewed and evaluated weights differ from local completed runs.')
    destination = common.ROOT / 'results/analysis_upload' / (
        'whole_map_91701_teaching_efficiency_seeds0to4_' + review.name.removeprefix('review_'))
    if destination.exists():
        verify(destination)
        print(f'Existing package verified; preserved: {destination}')
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    provenance, index = {}, []

    def copy(source, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if relative == 'review/REPORT.md':
            content = target.read_text(encoding='utf-8')
            for folder in (analysis, evaluation):
                content = content.replace('../' + folder.name + '/',
                                          '../../../../' + folder.relative_to(common.ROOT).as_posix() + '/')
            target.write_text(content, encoding='utf-8')
        provenance[Path(relative).as_posix()] = source.relative_to(common.ROOT).as_posix()

    # Avoid duplicating the large 17500-row table; link its tracked report directory below.
    for source in review.iterdir():
        if source.is_file():
            copy(source, 'review/' + source.name)
    log_names = ('training.csv', 'validation_curve.csv', 'validation_details.csv', 'value_learning.csv',
                 'advice_budget.csv', 'collection_coverage.csv', 'supervision_coverage.csv',
                 'failure_behaviors.csv', 'result.json', 'run_manifest.json', 'completion.json', 'static_final.json')
    for seed in common.SEEDS:
        directory, _ = runs[common.METHOD, seed]
        for name in log_names:
            copy(directory / name, f'runs/{common.METHOD}/seed_{seed}/{name}')
    registered = set()
    for _, result in runs.values():
        registered.update(result['source_sha256'])
    support = ['scripts/analyze_teaching_efficiency.py', 'scripts/evaluate_teaching_efficiency.py',
               'scripts/prepare_teaching_efficiency.py', 'scripts/review_teaching_efficiency_results.py',
               'scripts/package_teaching_efficiency_analysis.py', 'scripts/run_runtime_path_guidance.py',
               'scripts/analyze_runtime_path_pilot.py', 'tests/test_teaching_efficiency.py',
               'docs/TEACHING_EFFICIENCY_V1.zh-CN.md']
    sources = {common.ROOT / name for name in registered | set(support)}
    sources.update((common.ROOT / 'src/astar_d3qn').rglob('*.py'))
    for source in sorted(sources):
        copy(source, 'snapshot/' + source.relative_to(common.ROOT).as_posix())
    # Include the four original failure records used to freeze local critical observations.
    frozen = common.read_json(common.artifact_root(config) / 'freeze_manifest.json')
    for name in frozen['source_data_sha256']:
        if name.startswith('outputs'):
            source = common.ROOT / name
            copy(source, 'snapshot/' + source.relative_to(common.ROOT).as_posix())
    distances = static_distances(problem)

    def store(source, method, seed, step, trace, reason):
        relative = f'trajectories/{method}_seed{seed}_step{step}_{trace["scenario_id"]}.json'
        if any(r['file'] == relative for r in index):
            return
        digest = common.sha256(source)
        write_json(dict(method=method, seed=seed, environment_steps=step, selection_reason=reason,
                        source_file=source.relative_to(common.ROOT).as_posix(), source_sha256=digest,
                        trace=trace), destination / relative)
        index.append(dict(method=method, seed=seed, environment_steps=step, scenario_id=trace['scenario_id'],
                          termination_reason=trace['termination_reason'], selection_reason=reason, file=relative))

    for seed in common.SEEDS:
        source = runs[common.METHOD, seed][0] / 'validation_failures_200000.json'
        for trace in common.read_json(source):
            store(source, common.METHOD, seed, 200000, trace, 'All new-ablation final failures on the original 50 scenes.')
    selections = ((2, 100085, 'waiting_dominated'), (2, 100085, 'collision'),
                  (2, 110173, 'waiting_dominated'), (2, 110173, 'repeated_movement'),
                  (0, 140031, 'repeated_movement'), (1, 140028, 'repeated_movement'))
    for seed, step, phenotype in selections:
        source = runs[common.METHOD, seed][0] / f'validation_failures_{step:06d}.json'
        candidates = sorted(common.read_json(source), key=lambda t: t['scenario_id'])
        chosen = next(t for t in candidates if (t['termination_reason'] == 'collision' if phenotype == 'collision'
                      else trace_behavior(t, distances)[phenotype]))
        store(source, common.METHOD, seed, step, chosen, f'First scenario ID with {phenotype} at the specified low checkpoint.')
    for method, seed in (('advice_bound_margin', 3), (common.METHOD, 0), (common.METHOD, 3)):
        source = raw_root / method / f'seed_{seed}' / 'failure_trajectories.json'
        for trace in common.read_json(source):
            if method == common.METHOD and seed == 0 and trace['termination_reason'] != 'timeout':
                continue
            store(source, method, seed, 200000, trace,
                  'All independent final failures for this seed.' if seed == 3 else
                  'All 12 independent final waiting timeouts for the new ablation seed0.')
    write_records_csv(index, destination / 'trajectory_index.csv')
    older = 'whole_map_91701_value_repair_seeds0to4_20261007_234317_595719'
    root_link = '../../..'
    relative_review = review.relative_to(common.ROOT).as_posix()
    relative_analysis = analysis.relative_to(common.ROOT).as_posix()
    relative_evaluation = evaluation.relative_to(common.ROOT).as_posix()
    readme = f'''# A*教学效率：七方法 × seed0–4，GitHub分析入口

最新结果：无代选＋动作监督＋TD限制AULC为83.02%±4.07%，比无A*＋TD限制提高34.66个百分点；完整组合91.37%±1.05%，再提高8.35个百分点，主要来自前5万步。两项配对比较均五seed同向提高。500场景最终成功率依次为新组97.84%±1.75%、组合98.68%±1.08%、代选＋TD限制99.12%±1.17%。论文定位为学习效率，不主张最终避障最好。

固定地图91701、起点(3,3)、终点(36,36)，每回合3–5个动态障碍；各200000环境步。所有评估epsilon=0，由D3QN自主选动作，无A*代选或动态保护。网络输入没有A*路径；教学是训练时的动作代选和/或辅助动作监督，不能与此前运行时路径输入实验混淆。本轮使用配对随机初始化，没有加载旧静态foundation。

阅读顺序：

1. [最终复核报告]({root_link}/{relative_review}/REPORT.md)：逐seed、均值±样本标准差、全程/阶段AULC、最后五次均值、退出低谷、监督覆盖和碰撞Q诊断。
2. [七组完整学习曲线与原50场景明细]({root_link}/{relative_analysis}/REPORT.md)、[全部检查点CSV]({root_link}/{relative_analysis}/all_validation_checkpoints.csv)。
3. [独立500场景最终评估]({root_link}/{relative_evaluation}/REPORT.md)、[17500条场景明细]({root_link}/{relative_evaluation}/scene_details.csv)、[全部3496条失败行为]({root_link}/{relative_evaluation}/failure_behaviors.csv)。
4. [新增五seed训练日志](runs/supervision_only_bound/seed_0/result.json)，其余seed目录并列；原六组30次训练日志复用[此前包](../{older}/README.md)。
5. [精选完整失败轨迹索引](trajectory_index.csv)、[训练源码及支持文件快照](snapshot/scripts/run_teaching_efficiency.py)、[独立配置快照](snapshot/configs/whole_map_91701_teaching_efficiency_v1.yaml)。
6. [可复制给GPT的分析请求](GPT_ANALYSIS_PROMPT.md)。

包内38条精选完整轨迹有明确选择规则：新组原50场景全部5次最终失败；退出/后期低谷6个分类代表；组合seed3独立场景全部9次碰撞；新组seed0全部12次独立等待超时；新组seed3全部6次独立碰撞。全部3496条失败的行为统计仍在链接的CSV中，精选轨迹不是所有失败样本。数据和证据保留成功及失败，未筛选有利seed。

独立500场景在新增训练前冻结，与旧50场景的物理场景及路线组合不重复，仅评估200000步最终模型，不用于调参、教学调度或检查点选择。它们来自同地图同路线池，不代表跨地图泛化，也不保证与随机训练场景完全不重合。统计单位是5个配对训练seed，不能把500场景当500次独立训练。

已核对35次完成状态、源码和最终权重SHA、配对初始状态，735个原50场景检查点、17500条独立结果，以及全部3496条独立失败轨迹。本次打包仅读取并复制已有结果，没有训练或重新评估。future障碍信息仅用于已保存前缀的事后诊断。

`manifest.json`记录包文件SHA、外部报告文件SHA及35份未上传权重路径/SHA。快照中的登记训练源文件可与run result中的source_sha256核对；其他支持文件是整理时快照。包内及冻结数据保留原始字节，避免换行转换破坏指纹。

本地权重、原始outputs和约1.65GB完整独立失败轨迹不上传；包及链接的CSV足以阅读分析。只有克隆仓库不能再次运行依赖本地权重的审核或评估命令，不能因此自动重训。使用已生成的报告、日志、数据和源码快照即可。
'''
    (destination / 'README.md').write_text(readme, encoding='utf-8')
    prompt = '''请基于本目录README的链接、训练日志及对应源码分析最新七方法seed0–4正式实验。请检查全程与分阶段AULC、最后五次均值、90000–120500步内最低成功率，以及500冻结独立场景的最终安全成功/动态碰撞/超时率；均逐seed报告，使用均值及样本标准差。

研究范围固定地图、相同起终点、每回合3–5动态障碍，重点是A*教学能否加速D3QN自主学习。新增supervision_only_bound没有A*动作代选，但保留匹配实际执行、观测风险筛选、非碰撞标签的动作监督与TD目标限制；它和原六组均从配对随机初始化开始。不要混淆静态foundation、离线25%示范回放或运行时路径输入实验。

重点回答：动作监督在没有代选时是否有效？完整组合的额外收益在哪些阶段？43.28%与69.71%的回放标签覆盖率能支持什么，不能证明什么？新组seed2退出低谷中的等待与后期seed0/1循环有何区别？原组合seed3在独立场景9次碰撞中6次精确复现原局部观测，为什么其向右/向下Q差只有0.02仍会撞？新增seed3正确选择向下及精确训练经历覆盖能提供何种证据，为什么仍不能确定碰撞因果？

明确区分观察证据、合理推断、未记录的旧训练经历。不要仅凭成功率猜失败原因，不要把回放抽样次数当唯一经验数，不要用加载后同步的目标网络冒充历史TD目标。原组合在500场景最终成功率98.68%，代选＋TD限制99.12%；不要夸大最终效果或显著性。500场景不是500个训练重复，且此评估不是跨地图泛化。

围绕学习效率提出可信的论文主线、结论边界和后续实验优先级。后续调度或监督改动应基于训练数据/原50验证场景；不要据已查看的500独立场景调参，若使用其中的发现来调整方法，后续确认需新冻结独立场景。不要自动运行训练。
'''
    (destination / 'GPT_ANALYSIS_PROMPT.md').write_text(prompt, encoding='utf-8')
    external = {}
    for folder in (review, analysis, evaluation, common.artifact_root(config)):
        for source in folder.iterdir():
            if source.is_file():
                external[source.relative_to(common.ROOT).as_posix()] = common.sha256(source)
    manifest = dict(complete=True, training_started=False, evaluation_started=False,
                    review_source=relative_review, registered_training_source_files=sorted(registered),
                    file_provenance=provenance, external_repo_files_sha256=external,
                    report_links_adjusted_for_package_location=['review/REPORT.md'],
                    excluded_model_fingerprints=weights, selected_full_trace_count=len(index),
                    reused_baseline_package=older,
                    files_sha256={p.relative_to(destination).as_posix(): common.sha256(p)
                                  for p in destination.rglob('*') if p.is_file()})
    if len(index) != 38:
        raise ValueError('The declared complete/representative trace selection changed.')
    write_json(manifest, destination / 'manifest.json')
    verify(destination)
    print(destination)
    return destination


def verify(destination):
    manifest = common.read_json(destination / 'manifest.json')
    for name, digest in manifest['files_sha256'].items():
        if common.sha256(destination / name) != digest:
            raise ValueError(f'Package file changed: {name}')
    for name, digest in manifest['external_repo_files_sha256'].items():
        if common.sha256(common.ROOT / name) != digest:
            raise ValueError(f'Linked repository evidence changed: {name}')
    files = [p for p in destination.rglob('*') if p.is_file()]
    if any(p.suffix in ('.pth', '.pt', '.zip') or p.stat().st_size >= 100 * 1024**2 for p in files):
        raise ValueError('Unexpected model, archive or oversized file in reading package.')
    print(f'Package verified: {len(files)} files, {sum(p.stat().st_size for p in files)/1024**2:.2f} MiB; '
          f'{manifest["selected_full_trace_count"]} selected complete trajectories.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    root = common.ROOT / 'results/whole_map_91701_teaching_efficiency_v1'
    candidates = sorted(p for p in root.glob('review_*') if (p / 'verification.json').exists())
    if not args.review and not candidates:
        raise FileNotFoundError('No completed review.')
    package((args.review or candidates[-1]).resolve())


if __name__ == '__main__':
    main()

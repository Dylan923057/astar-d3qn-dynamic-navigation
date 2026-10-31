"""Build an immutable five-seed CE reading bundle; verify with the Python standard library."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import gzip
import hashlib
from pathlib import Path
import re
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import package_fixed_ce_evidence as base

NAME = 'asvc_fixed_ce_five_seeds_v1'
BASE = ROOT / 'results/analysis_upload' / NAME
OLD_NAME = 'asvc_fixed_ce_v1'
require, normal, local, sha = base.require, base.normal, base.local, base.sha
read, write, rows, write_csv, close, integral = base.read, base.write, base.rows, base.write_csv, base.close, base.integral
CODE = [f'configs/{NAME}.yaml', 'scripts/fixed_ce_five_common.py', 'scripts/run_fixed_ce_five.py',
        'scripts/analyze_fixed_ce_five.py', 'scripts/start_fixed_ce_five.ps1', 'tests/test_fixed_ce_five.py',
        'docs/ASVC_FIXED_CE_FIVE_SEEDS.zh-CN.md', 'scripts/package_fixed_ce_five_evidence.py',
        'scripts/sync_fixed_ce_five_github.ps1', 'docs/ASVC_FIXED_CE_FIVE_GITHUB_UPLOAD.zh-CN.md']
ATTRIBUTES = ['configs/.gitattributes', 'scripts/.gitattributes', 'tests/.gitattributes', 'docs/.gitattributes',
              f'data/{NAME}/.gitattributes', f'results/analysis_upload/{NAME}/.gitattributes']


def verify(package, check_sources=False):
    package = Path(package).resolve()
    manifest = read(package / 'manifest.json')
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual == set(manifest['files_sha256']) | {'manifest.json'}, 'Archive file list differs')
    patterns = (r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
                r'AKIA[0-9A-Z]{16}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    for name, digest in manifest['files_sha256'].items():
        path = local(package, name)
        require(path.suffix.lower() in base.ALLOW or path.name == '.gitattributes', 'Unexpected artifact: ' + name)
        require(path.stat().st_size < 50 * 1024**2, 'Oversized artifact: ' + name)
        require(sha(path) == digest, 'Archive hash changed: ' + name)
        if path.suffix.lower() not in ('.gz', '.pdf', '.png'):
            require(not any(re.search(p, path.read_text(encoding='utf-8-sig')) for p in patterns), 'Potential credential: ' + name)
    require(manifest['new_training_seeds'] == [2,3,4] and manifest['reused_ce_seeds'] == [0,1], 'Seed scope differs')
    require(manifest['new_training_runs'] == 3 and manifest['new_environment_steps'] == 600000 and
            manifest['new_gradient_updates'] == 598503, 'Extension budget differs')
    require(manifest['cumulative_ce_environment_steps'] == 1000000 and not manifest['baseline_retrained'] and
            not manifest['independent500_used'] and not manifest['weights_included'], 'Study scope differs')
    for source, destination in manifest['source_to_archive'].items():
        path = local(package, destination)
        if destination.endswith('.json.gz'):
            with gzip.open(path, 'rb') as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
        else:
            digest = sha(path)
        require(digest == manifest['source_files_sha256'][source], 'Original scientific bytes differ: ' + source)
    for study in (NAME, OLD_NAME):
        frozen = read(package / f'data/{study}/freeze.json')
        for name, digest in frozen['source_sha256'].items():
            require(sha(local(package, name)) == digest, 'Frozen code changed: ' + name)
        for name, digest in frozen['copied_input_sha256'].items():
            require(sha(local(package / f'data/{study}', name)) == digest, 'Frozen copied input changed: ' + name)
        for name, digest in frozen['protected_files_sha256'].items():
            name = normal(name)
            require(manifest['source_files_sha256'].get(name, manifest['models_sha256'].get(name)) == digest,
                    'Protected historical artifact changed: ' + name)
    results = package / f'results/{NAME}'
    require(read(results / 'queue_status.json')['all_five_complete'] and
            read(results / 'analysis_status.json')['all_five_ce_runs_complete'], 'Five-seed runs incomplete')
    preflight = read(results / 'preflight.json')
    require(preflight['passed'] and preflight['tests'] == 13, 'Thirteen-check preflight missing')
    scenes = read(package / f'data/{NAME}/fixed_validation_scenarios.json')['scenarios']
    order = [s['scenario_id'] for s in scenes]
    require(len(order) == len(set(order)) == 50, 'Expected original50 scenario order')
    metrics = rows(results / 'per_seed_metrics.csv')
    expected = {(m, str(s)) for m in ('margin','ce') for s in range(5)}
    require(len(metrics) == 10 and {(r['method'],r['seed']) for r in metrics} == expected, 'Expected ten run summaries')
    run_map = {(r['method'],str(r['seed'])): r for r in manifest['runs']}
    require(set(run_map) == expected, 'Run provenance is incomplete')
    combined_details, combined_curves = [], []
    total_episodes = total_failures = 0
    for record in metrics:
        method, seed = record['method'], record['seed']
        run = run_map[method,seed]
        directory = local(package, run['directory'])
        result = read(directory / 'result.json')
        require(read(directory / 'completion.json')['complete'] and result['complete'] and
                result['environment_steps'] == 200000 and result['gradient_updates'] == 199501, 'Run budget incomplete')
        require(manifest['models_sha256'][run['directory']+'/model_final.pth'] == result['final_model_sha256'], 'Final model identity differs')
        curve, details = rows(directory / 'validation_curve.csv'), rows(directory / 'validation_details.csv')
        steps = [int(r['environment_steps_total']) for r in curve]
        require(steps == sorted(set(steps)) and len(steps) == 21 and steps[0] == 0 and steps[-1] == 200000,
                'Expected 21 actual validation checkpoints per run')
        require(len(details) == 1050, 'Expected 1050 validation rows per run')
        grouped = defaultdict(list)
        for detail in details:
            grouped[int(detail['environment_steps_total'])].append(detail)
        for point in curve:
            step = int(point['environment_steps_total'])
            part = grouped[step]
            require([r['scenario_id'] for r in part] == order, 'Frozen validation scene order differs')
            for field, raw in [('safe_success_rate','safe_success'), ('collision_rate','collision')]:
                close(point[field], statistics.mean(float(r[raw]) for r in part), field)
            close(point['timeout_rate'], sum(r['termination_reason']=='timeout' for r in part)/50, 'timeout rate')
            with gzip.open(directory / f'validation_failures_{step:06d}.json.gz', 'rt', encoding='utf-8-sig') as stream:
                traces = base.json.load(stream)
            failed = {r['scenario_id']:r for r in part if float(r['safe_success'])!=1.}
            require(len(traces)==len(failed) and {t['scenario_id'] for t in traces}==set(failed), 'Missing or duplicate failure traces')
            for trace in traces:
                detail = failed[trace['scenario_id']]
                require(len(trace['steps'])==int(detail['steps']) and trace['termination_reason']==detail['termination_reason'],
                        'Failure trace length/outcome differs')
            total_failures += len(traces)
        total_episodes += len(details)
        combined_details.extend(dict(method=method,seed=seed,**r) for r in details)
        combined_curves.extend(dict(method=method,seed=seed,**r) for r in curve)
        for start, end in [(0,50000),(0,200000),(100000,200000)]:
            close(record[f'aulc_{start}_{end}'], integral(curve,start,end,'safe_success_rate'), 'actual-step AULC')
        late = [r for r in curve if int(r['environment_steps_total'])>=100000]
        close(record['late_min_safe_success_rate'], min(float(r['safe_success_rate']) for r in late), 'late minimum')
        for field in ('safe_success_rate','collision_rate','timeout_rate'):
            close(record['final_'+field], curve[-1][field], 'final outcome')
        for field in ('collision_rate','timeout_rate'):
            close(record['late_'+field+'_aulc'], integral(curve,100000,200000,field), 'late outcome AULC')
            close(record['late_'+field+'_point_mean'], statistics.mean(float(r[field]) for r in late), 'late point mean')
            close(record['late_'+field+'_point_max'], max(float(r[field]) for r in late), 'late point maximum')
        training = rows(directory / 'training.csv')
        close(training[-1]['environment_steps_total'],200000,'training budget')
        close(training[-1]['gradient_updates_total'],199501,'update budget')
        if method=='ce':
            margin_dir = local(package,run_map['margin',seed]['directory'])
            margin = read(margin_dir / 'run_manifest.json')
            require(result['initial_state_sha256']==margin['initial_state_sha256'] and
                    result['effective_training']==margin['effective_training'], 'Paired seed initialization/settings differ')
            require(result['baseline_model_sha256']==read(margin_dir/'result.json')['final_model_sha256'] and
                    result['baseline_curve_sha256']==sha(margin_dir/'validation_curve.csv'), 'Paired margin provenance differs')
            require(result['temperature']==1 and result['pretraining_updates']==0 and
                    not any(result[k] for k in ('evaluation_advice','independent500_used','route_input','soft_risk_weight')), 'CE scope changed')
            for row in rows(directory/'value_learning.csv'):
                require(row['supervision_loss_type']=='CE_T1','Incorrect loss log')
                require(all(base.math.isfinite(float(row[k])) for k in ('q_abs_max','td_loss_mean','supervision_ce_loss_mean','supervision_weighted_loss_mean')),
                        'Nonfinite training diagnostic')
                require(base.math.isclose(float(row['td_loss_mean'])+float(row['supervision_weighted_loss_mean']),
                        float(row['total_loss_mean']),rel_tol=2e-7,abs_tol=1e-8),'Float32 CE loss sum differs')
                if int(row['bin_start_step'])>100000:
                    close(row['supervision_weight_mean'],.1,'Persistent supervision floor')
            for row in rows(directory/'advice_budget.csv'):
                if int(row['bin_start_step'])>100000:
                    close(row['applied'],0,'Advice after exit')
    require(total_episodes==10500, 'Validation record count differs')
    require(rows(results/'validation_details.csv')==combined_details, 'Combined scenario CSV differs from raw logs')
    require(rows(results/'learning_curve_points.csv')==combined_curves, 'Combined curve CSV differs from raw logs')
    for row in rows(results/'mean_sample_sd.csv'):
        values = [float(r[row['metric']]) for r in metrics if r['method']==row['method']]
        close(row['completed_paired_training_seeds'],5,'Five training seeds required')
        require(row['paired_seed_ids']=='0;1;2;3;4' and row['all_five_complete']=='True','Unmatched/missing seed')
        close(row['mean'],statistics.mean(values),'mean over training seeds')
        close(row['sample_sd'],statistics.stdev(values),'sample SD')
    paired = rows(results/'paired_seed_differences.csv')
    require(len(paired)==5 and all(r['paired_complete']=='True' for r in paired),'Incomplete paired differences')
    for row in paired:
        part = {r['method']:r for r in metrics if r['seed']==row['seed']}
        for key in row.keys()-{'seed','contrast','paired_complete'}:
            close(row[key],float(part['ce'][key])-float(part['margin'][key]),'paired difference')
    for row in rows(results/'paired_mean_sample_sd.csv'):
        values=[float(r[row['metric']]) for r in paired]
        close(row['completed_paired_training_seeds'],5,'paired count')
        close(row['mean'],statistics.mean(values),'paired mean')
        close(row['sample_sd'],statistics.stdev(values),'paired sample SD')
    costs = rows(results/'training_cost.csv')
    require(len(costs)==10 and sum(int(r['new_steps_in_this_extension']) for r in costs)==600000,'New cost differs')
    for row in costs:
        run = run_map[row['method'],row['seed']]
        result = read(local(package,run['directory'])/'result.json')
        require((row['source_reused']=='True')==run['reused_in_extension'],'Cost reuse classification differs')
        for key in ('environment_steps','gradient_updates','online_training_seconds','session_seconds'):
            close(row[key],result[key],'cost '+key)
    model_rows = rows(package/'model_index.csv')
    require({r['path']:r['sha256'] for r in model_rows}==manifest['models_sha256'],'Model index differs')
    if check_sources:
        for name,digest in (manifest['source_files_sha256']|manifest['models_sha256']).items():
            require(sha(local(ROOT,name))==digest,'Local source/model changed: '+name)
    size=sum(local(package,n).stat().st_size for n in actual)
    print(f'PASS: {len(actual)} files / {size/1024**2:.2f} MiB; 5 CE vs 5 margin seeds, all complete.')
    print(f'10500 repeated-validation rows, 210 checkpoints, {total_failures} failure trajectories; statistics and hashes verified.')
    print('New extension: seeds 2/3/4, 600000 steps / 598503 updates; final500 unused; no model binaries.')
    return manifest


README = '''# 固定任务同教师 CE 对照：五 seed 完整阅读包

**CE 与原持续弱监督大间隔方法各五个 seed，均完成 200000 环境步。** 本轮只新增 CE seed2、3、4，合计600000环境步、598503次更新；原CE seed0、1及原大间隔五seed直接复用。不是把本轮新增训练写成五次。

## 核心结果

| 指标（均值±样本SD，n=5训练seed） | 大间隔 m=0.8 | CE T=1 |
|---|---:|---:|
| 前5万步AULC | 85.64% ± 1.25% | 84.26% ± 2.38% |
| 全程AULC | 95.63% ± 0.38% | 94.03% ± 0.95% |
| 后10万步AULC | 99.54% ± 0.34% | 98.14% ± 0.93% |
| 后半程最低成功率 | 96.80% ± 1.79% | 92.40% ± 4.34% |
| 最终安全成功率 | 100.00% ± 0.00% | 99.60% ± 0.89% |
| 后半程碰撞AULC | 0.22% ± 0.15% | 0.64% ± 0.53% |
| 后半程超时AULC | 0.24% ± 0.29% | 1.22% ± 0.55% |

观测：五个seed的全程及后半程AULC，CE均低于配对大间隔组。均值差分别为−1.59、−1.40个百分点（由未四舍五入值计算）。CE seed4后半程最低86%，最终恢复100%；seed2的后半最低值98%高于原组96%，所以不能声称每个稳定性指标都更差。CE最终五seed合计249/250成功、1/250碰撞、无超时；原组250/250成功。这些是同50场景的五模型重复验证，不是独立250任务测试。

结论限于本协议：没有得到将当前主方法换成CE的依据，保留大间隔监督，CE作为同教师监督形式对照。波动差异的具体梯度原因尚未确定，不宣称普遍优势、显著性或收敛证明，也不以交叉熵本身宣称创新。

## 阅读顺序

1. [详细报告](results/asvc_fixed_ce_five_seeds_v1/REPORT.zh-CN.md)、[五seed协议](docs/ASVC_FIXED_CE_FIVE_SEEDS.zh-CN.md)、[原两seed协议](docs/ASVC_FIXED_CE.zh-CN.md)。
2. [逐seed指标](results/asvc_fixed_ce_five_seeds_v1/per_seed_metrics.csv)、[均值/样本SD](results/asvc_fixed_ce_five_seeds_v1/mean_sample_sd.csv)、[逐seed配对差](results/asvc_fixed_ce_five_seeds_v1/paired_seed_differences.csv)、[配对差汇总](results/asvc_fixed_ce_five_seeds_v1/paired_mean_sample_sd.csv)。
3. [均值学习曲线](results/asvc_fixed_ce_five_seeds_v1/learning_curves.png)、[逐seed曲线](results/asvc_fixed_ce_five_seeds_v1/individual_seeds.png)、[Q/损失诊断](results/asvc_fixed_ce_five_seeds_v1/value_loss_diagnostics.png)。同目录保留PDF。
4. [合并210个验证点](results/asvc_fixed_ce_five_seeds_v1/learning_curve_points.csv)、[10500条逐场景结果](results/asvc_fixed_ce_five_seeds_v1/validation_details.csv)、[各组训练成本](results/asvc_fixed_ce_five_seeds_v1/training_cost.csv)。
5. [新增CE seed2–4原始日志](outputs/asvc_fixed_ce_five_seeds_v1/)、[复用CE seed0、1](outputs/asvc_fixed_ce_v1/)、[大间隔seed0–2](outputs/asvc_persistent_teaching_v1/supervision_floor/)、[大间隔seed3、4](outputs/asvc_persistent_supervision_five_seeds_v1/supervision_floor/)。全部训练CSV、Q范围、TD/监督损失、标签覆盖、代选预算及全部失败轨迹均保留。
6. [冻结及来源](data/asvc_fixed_ce_five_seeds_v1/freeze.json)、[冻结50场景](data/asvc_fixed_ce_five_seeds_v1/fixed_validation_scenarios.json)、[13项预检](results/asvc_fixed_ce_five_seeds_v1/preflight.json)、[完成状态](results/asvc_fixed_ce_five_seeds_v1/queue_status.json)、[全部模型身份](model_index.csv)。
7. [先前两seed报告](results/asvc_fixed_ce_v1/REPORT.zh-CN.md)、[其原上传清单](history/two_seed_manifest.json)及[本轮上传说明](docs/ASVC_FIXED_CE_FIVE_GITHUB_UPLOAD.zh-CN.md)。旧阅读包本身仍在仓库，未覆盖。

## 实验与统计边界

仅监督形式不同：合法动作上的CE，T=1；同一A*教师、风险筛选、碰撞标签排除、真实执行动作TD、代选100k退出、监督下限0.1、TD目标[-6,10]和epsilon日程。网络、观测、奖励、回放和更新预算相同，没有扫描参数、离线预训练或风险软权重。

所有方法共用原50冻结场景及顺序，每万步跨界后的回合末自主验证，无教师代选/风险保护/更新，使用200k最终模型。AULC按实际步数积分、边界线性插值。10500条记录=10个训练结果×21个验证点×50场景，统计单位是五个配对训练seed。均值曲线阴影是样本SD，不是置信区间。

扩展在查看原两seed后由用户授权，未依据结果调参或挑选seed。没有运行CE最终500场景，没有把重复验证冒充独立测试。与500k多任务CE提案分开。预检24步恢复单元不混入正式训练统计，历史与本轮成本分别列出；不同线程/负载下的墙钟耗时不作为效率优势证据。

## 字节保存与独立校验

原始文件按字节复制，所有失败轨迹采用 `.json.gz` 无损压缩，解压内容与原JSON的SHA256一致。本地原文件未删除。所有中间/最终模型和完整回放检查点保留本地，包内只登记路径、大小和哈希；阅读包不能代替模型重跑策略评估或续训。

进入本包目录，只需Python标准库，无PyTorch/GPU即可复核原始记录、全部失败轨迹、AULC、均值/样本SD、配对差与哈希：

```powershell
python tools/verify_bundle.py --verify --package .
```

在原工作区可以用 `python scripts/package_fixed_ce_five_evidence.py --verify --check-sources` 额外验证当前本地原文件和模型。原两seed包仍可独立校验；本轮仅对共享Git属性文件追加五seed路径规则，旧包的严格本地来源校验会反映这一元数据变化，旧科学记录和冻结训练代码未改。其他旧研究的文档链接可能需要完整仓库，本轮全部数据与图表均在本包。
'''


def build(new_revision=False):
    require((ROOT/'.git').exists(),'Build only in the original workspace')
    pointer=BASE/'latest.json'
    if pointer.exists() and not new_revision:
        package=local(ROOT,read(pointer)['package'])
        verify(package,True)
        print('Existing immutable bundle reused:',package.relative_to(ROOT))
        return package
    frozen=read(ROOT/f'data/{NAME}/freeze.json')
    require(read(ROOT/f'results/{NAME}/queue_status.json')['all_five_complete'],'Wait for all five runs')
    for group in ('source_sha256','protected_files_sha256'):
        for name,digest in frozen[group].items():
            require(sha(local(ROOT,name))==digest,'Frozen original changed: '+name)
    for name,digest in frozen['copied_input_sha256'].items():
        require(sha(local(ROOT/f'data/{NAME}',name))==digest,'Copied input changed: '+name)
    package=BASE/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    package.mkdir(parents=True,exist_ok=False)
    sources,destinations,models,index={},{},{},[]

    def copy(name,destination=None):
        name=normal(name);source=local(ROOT,name);digest=sha(source)
        if source.suffix in ('.pth','.pt'):
            if name not in models:
                models[name]=digest
                index.append(dict(path=name,bytes=source.stat().st_size,sha256=digest,included=False,
                    kind='full_recovery' if 'recovery' in source.name else 'policy_weights'))
            return
        destination=normal(destination or name)
        compressed=source.name.startswith('validation_failures_') and source.suffix=='.json'
        if compressed:destination+='.gz'
        if destinations.get(name)==destination:return
        target=local(package,destination);target.parent.mkdir(parents=True,exist_ok=True)
        if compressed:
            with target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped:
                with source.open('rb') as stream:shutil.copyfileobj(stream,zipped)
        else:
            require(not target.exists() or sha(target)==digest,'Conflicting destination')
            shutil.copyfile(source,target)
        sources[name],destinations[name]=digest,destination

    for name in base.dependency_files(list(frozen['source_sha256'])+CODE):copy(name)
    for name in frozen['protected_files_sha256']:copy(name)
    for folder in (f'data/{NAME}',f'outputs/{NAME}',f'results/{NAME}'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and 'matplotlib_cache' not in path.parts:
                copy(path.relative_to(ROOT).as_posix())
    for name in CODE+ATTRIBUTES+['pyproject.toml']:copy(name)
    old_pointer=read(ROOT/f'results/analysis_upload/{OLD_NAME}/latest.json')
    old_package=local(ROOT,old_pointer['package'])
    require(sha(old_package/'manifest.json')==old_pointer['manifest_sha256'],'Prior publication pointer changed')
    copy((old_package/'manifest.json').relative_to(ROOT).as_posix(),'history/two_seed_manifest.json')
    copy(f'results/analysis_upload/{OLD_NAME}/latest.json','history/two_seed_pointer.json')
    copy('scripts/package_fixed_ce_five_evidence.py','tools/verify_bundle.py')
    run_index=[]
    registration=read(ROOT/'data/asvc_current_method.json')
    for seed in range(5):
        for method in ('margin','ce'):
            directory=Path(normal(registration['final_models'][str(seed)])).parent if method=='margin' else \
                Path('outputs')/(OLD_NAME if seed<2 else NAME)/f'seed_{seed}'
            run_index.append(dict(method=method,seed=seed,directory=directory.as_posix(),
                reused_in_extension=(method=='margin' or seed<2)))
    write_csv(package/'model_index.csv',sorted(index,key=lambda r:r['path']))
    (package/'README.md').write_text(README,encoding='utf-8')
    (package/'.gitattributes').write_text('** -text\n',encoding='utf-8')
    relative=package.relative_to(ROOT).as_posix()
    scope=sorted(set(CODE+ATTRIBUTES+[p.relative_to(ROOT).as_posix() for p in (ROOT/f'data/{NAME}').rglob('*') if p.is_file()]+
        [relative,f'results/analysis_upload/{NAME}/latest.json',f'results/analysis_upload/{NAME}/LATEST.md']))
    write(package/'upload_scope.json',scope)
    write(package/'manifest.json',dict(created=datetime.now().isoformat(),name=NAME,
        files_sha256={p.relative_to(package).as_posix():sha(p) for p in sorted(package.rglob('*')) if p.is_file()},
        source_files_sha256=sources,source_to_archive=destinations,models_sha256=models,runs=run_index,
        new_training_runs=3,new_training_seeds=[2,3,4],reused_ce_seeds=[0,1],pooled_training_seeds=list(range(5)),
        new_environment_steps=600000,new_gradient_updates=598503,cumulative_ce_environment_steps=1000000,
        cumulative_ce_gradient_updates=997505,baseline_retrained=False,weights_included=False,independent500_used=False,
        validation_records=10500,new_ce_validation_records=3150,reused_ce_validation_records=2100,
        reused_margin_validation_records=5250,statistical_unit='five paired training seeds',
        prior_two_seed_manifest_sha256=old_pointer['manifest_sha256']))
    verify(package,True)
    write(pointer,dict(package=relative,manifest_sha256=sha(package/'manifest.json')))
    (BASE/'LATEST.md').write_text(f'# CE五seed完整阅读包\n\n[数据、图表与完整报告]({package.name}/README.md)\n\n'
        '本轮新增seed2、3、4各20万步，原CE seed0、1及大间隔五seed复用。五seed全程AULC：CE94.03%，大间隔95.63%；'
        '后半程98.14%对99.54%。10500条重复验证记录，不是独立测试；未使用最终500场景。\n',encoding='utf-8')
    print('Package:',relative)
    return package


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--check-sources',action='store_true')
    parser.add_argument('--new-revision',action='store_true');parser.add_argument('--package',type=Path)
    args=parser.parse_args()
    if args.verify:
        if args.package:package=args.package
        elif (ROOT/'manifest.json').exists():package=ROOT
        else:
            pointer=read(BASE/'latest.json');package=local(ROOT,pointer['package'])
            require(sha(package/'manifest.json')==pointer['manifest_sha256'],'Publication pointer differs')
        verify(package,args.check_sources)
    else:build(args.new_revision)


if __name__=='__main__':main()

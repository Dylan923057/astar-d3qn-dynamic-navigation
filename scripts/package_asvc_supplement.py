"""Build a GitHub reading bundle only after all five runs and 85 new evaluations finish."""
from datetime import datetime
import shutil
from asvc_supplement_common import *
from evaluate_asvc_supplement import run_info,EVAL,TRACES,DENSITY_REG
from plot_asvc_supplement import FIG,full_figures,design_figures

def package():
    for method in MAIN:
        for seed in SEEDS:
            p=run_info(method,seed)
            for density in ('1','3','5','7')+ (('original500',) if method==METHOD else ()):
                path=EVAL/density/method/f'seed_{seed}'
                assert common.read_json(path/'completion.json')['complete']
                registration=common.read_json(path/'registration.json')
                assert registration['model_sha256']==p['digest']
                if density!='original500':assert registration['scene_sha256']==common.sha256(DENSITY_REG/f'density_{density}.json')
                rows=__import__('analyze_runtime_path_pilot').load_csv(path/'scene_details.csv')
                assert len(rows)==500
    design_figures();full_figures()
    destination=ROOT/'results/analysis_upload'/NAME/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copytree(FIG,destination/'paper')
    shutil.copytree(EVAL,destination/'evaluation')
    shutil.copytree(REG,destination/'registration/training')
    shutil.copytree(DENSITY_REG,destination/'registration/density')
    sources=set(source_hashes())
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*asvc*.py'))
    # Include all code needed to resolve historical audits and regenerate plotting/evaluation.
    for base in ('scripts','src/astar_d3qn','configs'):
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.yaml','.json'):sources.add(str(p.relative_to(ROOT)))
    for name in sources:
        target=destination/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    for method in tuple(dict.fromkeys((*MAIN,*historical.MAIN))):
        for seed in SEEDS:
            p=run_info(method,seed)
            for name in ('run_manifest.json','result.json','completion.json','validation_curve.csv','validation_details.csv',
                         'training.csv','advice_budget.csv','value_learning.csv','supervision_alignment.csv','dqfd_learning.csv'):
                source=p['directory']/name
                if source.exists():
                    target=destination/'run_logs'/method/f'seed_{seed}'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    generation=common.read_json(REG/'freeze_manifest.json')['checks']['demonstrations']['original_generation_cost']
    write_json(generation,destination/'paper/shared_demonstration_generation_cost.json')
    (destination/'README.md').write_text('''# ASVC-D3QN 补实验阅读包

[实际结果报告](paper/REPORT.md) · [主表](paper/main_results.md) · [模块消融表](paper/module_ablation.md)

ASVC-D3QN仅为`advice_bound_margin_all_eligible`的论文显示名，旧ID、配置与结果路径不变。
新增五次20万在线步的A*初始化机制对照，不是A*-IDDQN完整复现。
本包仅在五次训练、原500场景的五模型评估及四档密度的80模型评估全部完成后生成。

paper内有PDF、SVG、300dpi PNG、实际验证时间的原始CSV、五seed均值/样本标准差及同seed配对差。
曲线阴影按实际时间线性对齐，未平滑。统计单位为训练seed。
registration保存冻结配置、场景ID、生成规则与SHA；run_logs保存所有方法的真实训练记录。
source_snapshot保存绘图与复现源码。权重和全部原始轨迹保留本地；paper/model_manifest.csv记录权重路径和SHA。
固定seed0及预注册的三个相同场景用于轨迹图，成功和失败均按实际结果展示。

DQfD的额外20000次离线更新、永久示范和PER成本单列；共享示范生成成本见paper/shared_demonstration_generation_cost.json。
历史没有记录的时间/设备/A*调用字段保持空白。密度试验是同地图适应评估，不用于调参，不是跨地图泛化。
首版密度场景因初始位置重叠无效而保留为历史；正式密度统计使用全部通过构造验证的v2。

复现命令（仓库根目录，PowerShell；已有完整结果不会自动重训）：

```powershell
python scripts/run_asvc_supplement.py
python scripts/prepare_asvc_density.py
python scripts/evaluate_asvc_supplement.py
python scripts/plot_asvc_supplement.py
python scripts/package_asvc_supplement.py
```

仅空输出目录的新训练：`python scripts/run_asvc_supplement.py --train`
恢复未完成训练：`python scripts/run_asvc_supplement.py --train --resume`
重新评估要求本地已有最终权重；缺权重不会自动训练。评估按已完成模型/密度恢复。
''',encoding='utf-8')
    manifest={p.relative_to(destination).as_posix():common.sha256(p) for p in destination.rglob('*') if p.is_file()}
    write_json(dict(complete=True,files_sha256=manifest,statistical_unit='training_seed',new_training_runs=5,
        new_evaluation_model_suites=85),destination/'manifest.json')
    pointer=destination.parent/'LATEST.md'
    pointer.write_text(f'# ASVC-D3QN 补实验阅读包\n\n[阅读入口]({destination.name}/README.md)\n',encoding='utf-8')
    write_json(dict(complete=True,reading_bundle=historical.relative(destination)),RESULTS/'completion.json')
    print(destination,flush=True)
    return destination

if __name__=='__main__':package()

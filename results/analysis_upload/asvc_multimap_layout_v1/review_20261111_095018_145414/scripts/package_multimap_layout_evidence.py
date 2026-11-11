"""Publish a self-contained audited snapshot; never edit frozen training sources."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import gzip
import json
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile
import multimap_evidence_tools as e

ROOT=Path(__file__).resolve().parents[1]
NAME=e.NAME;DATA=ROOT/'data'/NAME;OUT=ROOT/'outputs'/NAME
BASE=ROOT/'results/analysis_upload'/NAME
LABELS=dict(d3qn='D3QN',astar_prefill_td='A* prefill + TD constraint',dqfd='DQfD',qdagger='QDagger (task adaptation)',
    ce='Online CE',asvc='ASVC',asvc_no_td='ASVC w/o TD constraint',asvc_withdraw='ASVC w/o persistence')
COLORS=dict(d3qn='#66717A',astar_prefill_td='#9A6BAE',dqfd='#DA8B1C',qdagger='#228E80',ce='#4CA1C4',asvc='#C94B45')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def relative(path):return Path(path).relative_to(ROOT).as_posix()
def run_dir(job):return OUT/f'map_{job["map_seed"]}'/job['method']/f'seed_{job["train_seed"]}'

def completed_jobs():
    freeze_sha=e.sha(DATA/'freeze.json');done=[];missing=[]
    for job in read(DATA/'queue.json')['jobs']:
        root=run_dir(job);file=root/'completion.json'
        if file.exists():
            value=read(file);assert value['complete']
            assert value['identity']==dict(study=NAME,**job,freeze_sha256=freeze_sha)
            assert value['model_sha256']==e.sha(root/'model_final.pth')
            assert read(root/'final500/completion.json')['complete'];done.append(job)
        else:
            status=read(root/'status.json') if (root/'status.json').exists() else {}
            missing.append(dict(**job,observed_environment_steps=status.get('environment_steps',0),
                observed_phase=status.get('phase','not_started'),failure_record=(root/'failure.json').exists(),
                recovery_available=(root/'recovery_latest.pth').exists()))
    assert sum(j['method'] in e.MAIN for j in done)==90,'All 90 main comparisons must be complete before publication'
    return done,missing

class Archive:
    def __init__(self,path):self.path=path;self.sources={}
    def copy(self,source,destination=None,compress=False):
        source=Path(source);name=relative(source);destination=destination or name
        if name in self.sources:
            assert self.sources[name]['archive']==destination+('.gz' if compress else '')
            return
        target=self.path/(destination+('.gz' if compress else ''));target.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with source.open('rb') as src,target.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
        else:shutil.copyfile(source,target)
        self.sources[name]=dict(archive=target.relative_to(self.path).as_posix(),raw_sha256=e.sha(source),raw_bytes=source.stat().st_size,compressed=compress)
    def text(self,value,name):
        target=self.path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(value,encoding='utf-8')

def copy_yaml_ancestors(archive):
    import yaml
    pending=list(archive.path.glob('configs/*.yaml'));seen=set()
    while pending:
        file=pending.pop();name=file.relative_to(archive.path).as_posix()
        if name in seen:continue
        seen.add(name);cfg=yaml.safe_load(file.read_text(encoding='utf-8-sig')) or {}
        inherited=cfg.get('inherits')
        if not inherited:continue
        for parent in inherited if isinstance(inherited,list) else [inherited]:
            source=(ROOT/name).parent/parent
            if source.is_absolute() and not source.resolve().is_relative_to(ROOT):raise ValueError('External YAML parent')
            archive.copy(source);pending.append(archive.path/relative(source))

def add_model_index(archive,done):
    models=[];excluded=[]
    for job in done:
        directory=run_dir(job);model=directory/'model_final.pth';result=read(directory/'result.json')
        models.append(dict(scope='new_layout',**job,directory=relative(directory),model_path=relative(model),
            model_sha256=e.sha(model),model_bytes=model.stat().st_size,included=False,online_steps=200000,
            offline_updates=result['offline_updates']))
        for path in sorted(directory.glob('*.pth')):
            excluded.append(dict(path=relative(path),bytes=path.stat().st_size,sha256=e.sha(path),
                kind='final_weights' if path.name=='model_final.pth' else 'complete_recovery_or_pretraining_state',included=False))
    old=read(DATA/'existing_evidence.json')['fixed_task_training']
    for row in old:
        directory=ROOT/row['directory'];model=directory/'model_final.pth'
        assert e.sha(model)==row['final_model_sha256']
        models.append(dict(scope='original_map',map_seed=91701,train_seed=row['train_seed'],method=row['method'],directory=row['directory'],
            model_path=relative(model),model_sha256=row['final_model_sha256'],model_bytes=model.stat().st_size,included=False,online_steps=200000,
            offline_updates=20000 if row['method'] in ('dqfd','qdagger') else 0))
        archive.copy(directory/'validation_curve.csv',f'reference/original_map/{row["method"]}/seed_{row["train_seed"]}/validation_curve.csv',True)
        archive.copy(directory/'result.json',f'reference/original_map/{row["method"]}/seed_{row["train_seed"]}/result.json')
        archive.copy(directory/'completion.json',f'reference/original_map/{row["method"]}/seed_{row["train_seed"]}/completion.json')
    e.csv_write(models,archive.path/'model_index.csv');e.csv_write(excluded,archive.path/'excluded_model_artifacts.csv')
    return models,excluded

def plot(values,package):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    maps=[91705,91706,91707];per_seed=values['per_seed'];stats=values['per_map_mean_sample_sd'];curves=values['validation_curves']
    output=package/'figures';output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def statistic(map_seed,method,key):return next(r for r in stats if r['map_seed']==map_seed and r['method']==method and r['metric']==key)
    # Primary metric, all five actual seed values remain visible.
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True)
    for ax,map_seed in zip(axes,maps):
        for i,method in enumerate(e.MAIN):
            s=statistic(map_seed,method,'early_aulc');ax.bar(i,100*s['mean'],color=COLORS[method],width=.65,alpha=.85)
            ax.errorbar(i,100*s['mean'],yerr=100*s['sample_sd'],color='#30383E',capsize=3,lw=1)
            actual=[r['early_aulc']*100 for r in per_seed if r['map_seed']==map_seed and r['method']==method]
            ax.scatter(i+np.linspace(-.16,.16,len(actual)),actual,s=14,facecolor='white',edgecolor='#30383E',linewidth=.7,zorder=3)
        ax.set_title(f'map_seed {map_seed}');ax.set_xticks(range(6),['D3QN','A* init.','DQfD','QDagger','CE','ASVC'],rotation=35,ha='right')
        ax.set_ylim(0,104);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    axes[0].set_ylabel('Early AULC, 0–50k steps (%)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'01_primary_early_aulc.{ext}',dpi=300)
    plt.close(fig)
    # Common grid only for display; publication AULC is computed from actual points.
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.4),sharey=True);grid=np.arange(0,200001,4000)
    for ax,map_seed in zip(axes,maps):
        for method in e.MAIN:
            lines=[]
            for seed in range(5):
                selected=[r for r in curves if r['map_seed']==map_seed and r['method']==method and r['train_seed']==seed]
                x=[int(r['environment_steps_total']) for r in selected];y=[float(r['safe_success_rate'])*100 for r in selected]
                lines.append(np.interp(grid,x,y))
            mean=np.mean(lines,axis=0);sd=np.std(lines,axis=0,ddof=1)
            ax.plot(grid,mean,color=COLORS[method],lw=1.7,label=LABELS[method]);ax.fill_between(grid,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=COLORS[method],alpha=.10)
        ax.set_title(f'map_seed {map_seed}');ax.set_xlabel('Online environment steps');ax.set_xlim(0,200000);ax.set_ylim(0,102);ax.grid(alpha=.15)
        ax.set_xticks([0,50000,100000,150000,200000],['0','50k','100k','150k','200k'])
    axes[0].set_ylabel('Safe success rate (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.15,1,1))
    for ext in ('png','pdf'):fig.savefig(output/f'02_learning_curves.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1))
    for ax,key,title in zip(axes,['final_safe_success_rate','final_dynamic_collision_rate','final_timeout_rate'],['(a) Safe success','(b) Dynamic collisions','(c) Timeouts']):
        for method in e.MAIN:
            selected=[statistic(m,method,key) for m in maps];means=np.array([s['mean']*100 for s in selected]);sd=np.array([s['sample_sd']*100 for s in selected])
            ax.errorbar(range(3),means,yerr=sd,color=COLORS[method],marker='o',ms=4,lw=1.6,capsize=2,label=LABELS[method])
        ax.set_title(title);ax.set_xticks(range(3),[str(m) for m in maps]);ax.set_xlabel('Static map seed');ax.set_ylabel('Rate (%)');ax.grid(alpha=.15)
        if key!='final_safe_success_rate':ax.set_ylim(bottom=-.3)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.17,1,1))
    for ext in ('png','pdf'):fig.savefig(output/f'03_final500_outcomes.{ext}',dpi=300)
    plt.close(fig)

def report(package,values,snapshot):
    stats=values['per_map_mean_sample_sd'];jobs=snapshot['included_completed_jobs'];ablation=len(jobs)-90
    def s(m,method,key):return next(r for r in stats if r['map_seed']==m and r['method']==method and r['metric']==key)
    def fmt(row):return f'{row["mean"]*100:.2f}% ± {row["sample_sd"]*100:.2f}%' if row['sample_sd'] is not None else f'{row["mean"]*100:.2f}%'
    lines=['# 多地图补实验：早期样本效率与最终性能','',f'快照时间：{snapshot["created"]}。主对比90/90完成；关键消融{ablation}/10完成。总计{len(jobs)}/100完成，缺项见 `provenance/missing_jobs.csv`。',
        '', '核心目标是在不损害最终安全成功率、动态碰撞率和超时率的条件下，提高前5万步的样本效率。学习率0.0003是优化器超参数，与AULC不同。',
        '', '每图每方法train_seed=0–4均已完成200000在线步，采用固定最终模型，每seed测试500场景。每图6方法×5seed×500=15000回合，三图主对比共45000回合。各图独立从头训练，无模型迁移。',
        '', '以下均为五训练seed均值±样本SD；AULC按实际验证步数积分。','',
        '|地图|方法|前5万步AULC|全程AULC|最终安全成功|动态碰撞|超时|','|---|---|---:|---:|---:|---:|---:|']
    for m in (91705,91706,91707):
        for method in e.MAIN:lines.append('|'+str(m)+'|'+LABELS[method]+'|'+'|'.join(fmt(s(m,method,k)) for k in ('early_aulc','full_aulc','final_safe_success_rate','final_dynamic_collision_rate','final_timeout_rate'))+'|')
    lines+=['','直接观察：ASVC在三个新图的前5万步AULC均值均为六方法最高；91705和91707的最终三项指标均值均优于原始D3QN。91706虽然早期AULC提高，但最终安全成功率从D3QN的98.84%降为96.76%，超时从0.20%升为2.96%；动态碰撞从0.96%降为0.28%。因此只能支持部分布局同时达到早期增益和最终性能保持，不能宣称所有布局均无损。',
        '', '91706 ASVC seed0/3的最终安全成功率为91.2%/93.0%，超时8.0%/6.8%，其余三个seed为99.6%/100%/100%。失败轨迹全部保留；当前数据不能直接确定循环/等待形成的训练原因，也不能把低碰撞等同于整体导航更好。',
        '', '原图91701 ASVC最终500安全成功率99.28%±0.73%，三张新图分别99.36%±0.55%、96.76%±4.30%、99.72%±0.23%。原图每约10000步验证，新图每约4000步验证，不直接把不同采样频率的早期AULC差值当作布局因果效应。参考区保留原30条曲线及ASVC统一1万步插值网格的敏感性核对，插值不是新增评估。',
        '', 'DQfD/QDagger每seed额外20000次离线更新，示范生成成本单列；A*初始化无离线梯度更新，示范进入普通FIFO。不能把在线AULC直接当成总计算成本效率。三图示范保留6719/7573/6659条，均来自相同128次尝试预算；失败尝试已记录。',
        '', '跨图宏平均先对同方法同训练seed的三张图均值化，再按五seed计算样本SD；结果在cross_map_macro_*。配对差按同图同seed的ASVC减对照。所有 seed 均保留，不挑最佳中间模型，不将场景数当作训练样本数。',
        '', f'消融当前完成{ablation}/10，只在91707比较去TD目标约束、后期完全撤监督和复用完整ASVC；未满五seed的组不称为五seed结果，n_seeds列给出实际数量。部分消融结果不能用于正式完整排名。',
        '', '本轮未启用多起终点的两个测试池，未新增Office/Parcel/Warehouse，未启动新的参数扫描或其他训练。结论限定为同一地图生成规则下不同静态布局上的独立训练验证；不宣称零样本地图迁移、正式显著性或最终性能的统计非劣效。',
        '', '计时采用同一GPU且最多并发2个训练进程，各1个CPU线程。离线、在线、评估、断点写入分别记录；训练计时受后台任务及并发影响，不作为独占设备性能保证。推理耗时只来自自主评估。']
    (package/'summary/REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def build():
    BASE.mkdir(parents=True,exist_ok=True)
    frozen=read(DATA/'freeze.json')
    for name,digest in frozen['source_sha256'].items():assert e.sha(ROOT/name)==digest,name
    for name,digest in frozen['data_sha256'].items():assert e.sha(DATA/name)==digest,name
    done,missing=completed_jobs();stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print(f'PACKAGING verified snapshot: {len(done)}/100 completed; 90/90 main comparisons',flush=True)
    package=BASE/f'review_{stamp}';package.mkdir(exist_ok=False);archive=Archive(package)
    snapshot=dict(created=datetime.now().isoformat(),planned_runs=100,main_complete=True,ablation_complete=len(done)==100,
        included_completed_jobs=done,missing_jobs=missing,active_logs_not_copied=True,weights_included=False,
        protected_freeze_sha256=e.sha(DATA/'freeze.json'))
    e.write(snapshot,package/'provenance/snapshot.json');e.csv_write(missing,package/'provenance/missing_jobs.csv')
    for path in sorted(DATA.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.tmp','.lock'):archive.copy(path)
    for name in frozen['source_sha256']:archive.copy(ROOT/name)
    for name in ('scripts/multimap_layout_supervisor.py','scripts/start_multimap_layout_compatible.ps1',
        'scripts/package_multimap_layout_evidence.py','scripts/multimap_evidence_tools.py','pyproject.toml'):
        archive.copy(ROOT/name)
    copy_yaml_ancestors(archive)
    (package/'tools').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'scripts/multimap_evidence_tools.py',package/'tools/evidence.py')
    included_logs=('run_manifest.json','completion.json','result.json','training.csv','validation_curve.csv','validation_details.csv',
        'update_diagnostics.csv','advice.csv','supervision.csv','value_learning.csv','before_pretraining50.json')
    validation_failures=[]
    for job in done:
        directory=run_dir(job)
        for name in included_logs:
            path=directory/name
            if path.exists():archive.copy(path,compress=path.suffix=='.csv')
        for path in sorted((directory/'final500').iterdir()):
            if path.is_file() and path.suffix not in ('.tmp','.pth','.lock'):archive.copy(path,compress=path.suffix=='.csv')
        validation_failures.extend(sorted(directory.glob('validation_failures_*.json.gz')))
    print(f'Archived completed raw logs and final500 failures; {len(validation_failures)} repeated-validation failure files retained separately',flush=True)
    models,excluded=add_model_index(archive,done)
    for name in ('reused_original_final500.csv','reused_original_ablations.csv','existing_runs.csv','candidate_audit.csv','maps.csv','six_method_smoke.json',
        'new_static_layouts.png','new_static_layouts.pdf','map_91705_dynamic_preview.png','map_91706_dynamic_preview.png','map_91707_dynamic_preview.png',
        'REPORTING_COMPATIBILITY.zh-CN.md','launch.json'):
        path=ROOT/'results'/NAME/name
        if path.exists():archive.copy(path,'reference/'+name if name.startswith(('reused_','existing_','launch')) else 'figures/'+name if path.suffix in ('.png','.pdf') else 'provenance/'+name)
    for path in sorted((ROOT/'results'/NAME).glob('asvc_layout_comparison_*.json')):archive.copy(path,'reference/'+path.name)
    for name in ('six_method_per_seed.csv','six_method_mean_sample_sd.csv','independent_verification.json'):
        path=ROOT/'results/asvc_fixed_ce_qdagger_density_v1'/name
        if path.exists():archive.copy(path,'reference/original_density/'+name)
    # Supplemental validation trajectories remain byte-identical gzip members.
    attachment=BASE/f'validation_failures_{stamp}.zip';failure_index=[]
    with zipfile.ZipFile(attachment,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in validation_failures:
            item=dict(path=relative(path),bytes=path.stat().st_size,sha256=e.sha(path))
            failure_index.append(item);z.write(path,item['path'])
    e.csv_write(failure_index,package/'provenance/validation_failure_attachment_index.csv')
    print(f'Supplemental trajectory ZIP: {attachment.stat().st_size/1024**2:.1f} MiB',flush=True)
    e.write(dict(name=attachment.name,bytes=attachment.stat().st_size,sha256=e.sha(attachment),files=len(failure_index),
        purpose='All repeated-validation failure trajectories; final500 failures are already included in the main bundle',git_tracking=False),package/'provenance/supplemental_attachment.json')
    values=e.compute(package)
    print('Scientific counts, actual validation points, final500 scenes and metrics recomputed',flush=True)
    for name,value in values.items():
        if isinstance(value,list):e.csv_write(value,package/'summary'/(name+'.csv'))
    e.write(dict(scientific_audit=values['audit'],source_freeze_verified=True,model_hashes_checked_during_packaging=True,
        weights_included=False,source_to_archive_entries=len(archive.sources),created=datetime.now().isoformat()),package/'provenance/verification.json')
    report(package,values,snapshot);plot(values,package)
    archive.text('*.zip\n*.pth\n*.pt\n__pycache__/\n*.py[cod]\n*.tmp\n*.lock\n!**/demonstrations.npz\n', '.gitignore')
    archive.text('* -text\n','.gitattributes')
    contents=f'''# ASVC 多地图补实验：GitHub 证据包

快照时间：{snapshot['created']}。**六方法主对比90/90完成；消融{len(done)-90}/10完成；总计{len(done)}/100。** 缺项和中断不计零分，不用不完整组冒充五seed结果。

1. 先看 [主要结论与完整表格](summary/REPORT.zh-CN.md)，重点是**前5万步AULC及最终三项性能**。ASVC在三图的早期AULC均值最高，但91706有最终成功率/超时的折中，不能宣称全面无损。
2. 图表：[早期AULC](figures/01_primary_early_aulc.png)、[学习曲线](figures/02_learning_curves.png)、[最终500三指标](figures/03_final500_outcomes.png)、[三张新地图](figures/new_static_layouts.png)。PNG及PDF均保留，图表只由数据绘制，没有生成或补造评估点。
3. `summary/`保存逐seed、每图均值/样本SD、配对差和跨图宏平均。`outputs/asvc_multimap_layout_v1/`保留每次已完成运行的原始CSV无损gzip、最终500逐场景及**全部最终失败轨迹**。主对比45000次最终测试，另加当前已完成消融的测试。
4. `data/asvc_multimap_layout_v1/`保留冻结地图、候选与拒绝原因、完整路线池、50/500场景、程序式训练序列定义/前缀、本图示范及预算、95个冻结文件的源码快照。`configs/`、`scripts/`、`src/`、`tests/`保留相关实现及继承配置。原冻结完全不变。
5. `reference/`仅复用原图证据与密度汇总；不把旧回放、50万步多任务或零样本跨地图评估混入新布局训练统计。

## 校验和重算（只需Python标准库）

在本包根目录执行：

```powershell
python tools/evidence.py --verify
python tools/evidence.py --recompute --output ../recomputed_multimap
```

验证会检查完整文件列表/SHA、原冻结、运行身份、精确交互/更新预算、全部最终500场景顺序、失败轨迹、51个实际验证点及其50场景记录，并从原始记录重算AULC、均值、样本SD、配对差和跨图宏平均。重算写到包外，避免破坏归档。

CSV/JSON的无损gzip可直接用Python gzip模块打开；manifest同时登记原文件与压缩文件SHA。本目录`.gitattributes`保留字节，`.gitignore`明确允许冻结示范NPZ进入Git。**不要把压缩ZIP本身加入普通Git提交**；提交解压后的此目录即可。

## 模型、恢复与补充附件

`model_index.csv`登记{len(models)}个原图/新图最终模型，`excluded_model_artifacts.csv`登记已完成新运行的权重和完整恢复文件。二进制均留在原工作区，包中没有模型；仅凭本证据包不能重新做网络推断或完整续训。只加载权重不是完整恢复。

反复50验证的全部失败轨迹另存 `{attachment.name}`（本地同级目录，SHA和成员清单在provenance），可作为GitHub Release附件，避免普通仓库膨胀；最终500失败已经在主包内，没有抽样删除。未完成运行的可变日志不进入此快照。

原工作区完整恢复/继续缺项的实际命令：

```powershell
& D:\\softwares\\Anaconda3\\envs\\astar-drl\\python.exe -X utf8 -u scripts/multimap_layout_supervisor.py --run --workers 2
```

该命令只应在原工作区使用，自动跳过身份一致的完成项、完整恢复中断项；已有调度器运行时文件锁拒绝重复队列。训练源码不修改，NumPy1.26的梯形积分名称兼容仅发生在汇总进程。原9项预检包含旧模型保护核对，证据包未带这些模型；需要重做完整训练预检时须提供原模型与其历史路径。本包的标准库证据校验与统计重算不依赖它们。

结论限定为**同一生成规则下不同静态布局的独立训练验证**，每图固定起终点并从头训练；不代表未训练地图的零样本泛化，也不宣称统计非劣效或正式显著性。
'''
    archive.text(contents,'README.md')
    e.write(dict(paths=[relative(package),relative(BASE/'LATEST.md'),relative(BASE/'.gitignore')],
        excludes=['model weights','recovery states','ZIP attachments','pre-existing root README/.gitattributes/trajectory changes'],
        note='Review and add only this exact directory plus the latest pointer. Do not use git add .'),package/'upload_scope.json')
    patterns=(r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    for path in package.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md','.json','.yaml','.toml','.ps1','.csv'):
            text=path.read_text(encoding='utf-8-sig');assert not any(re.search(p,text) for p in patterns),'Potential credential: '+str(path)
    manifest=dict(study=NAME,created=snapshot['created'],planned_runs=100,included_completed_runs=len(done),main_runs=90,
        completed_ablation_runs=len(done)-90,weights_included=False,source_files=archive.sources,
        files_sha256={p.relative_to(package).as_posix():e.sha(p) for p in sorted(package.rglob('*')) if p.is_file()})
    e.write(manifest,package/'manifest.json')
    e.verify(package)
    bundle_zip=BASE/(package.name+'.zip')
    with zipfile.ZipFile(bundle_zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(package.parent).as_posix())
    record=dict(directory=relative(package),archive=relative(bundle_zip),archive_bytes=bundle_zip.stat().st_size,
        archive_sha256=e.sha(bundle_zip),supplemental_archive=relative(attachment),supplemental_bytes=attachment.stat().st_size,
        completed_runs=len(done),main_runs=90,ablation_runs=len(done)-90,created=snapshot['created'])
    e.write(record,BASE/(package.name+'_registration.json'))
    (BASE/'.gitignore').write_text('*.zip\n.build.lock\n',encoding='utf-8')
    (BASE/'LATEST.md').write_text(f'# 多地图布局补实验：最新上传包\n\n[阅读说明]({package.name}/README.md) · [结果报告]({package.name}/summary/REPORT.zh-CN.md) · [前期AULC图]({package.name}/figures/01_primary_early_aulc.png)\n\n主对比90/90完成；消融{len(done)-90}/10完成，总计{len(done)}/100。包中缺项单列，不冒充全部完成。\n\nGitHub提交解压后的 `{package.name}/`，ZIP及完整验证失败轨迹作为本地/Release附件。\n',encoding='utf-8')
    e.write(record,BASE/'latest.json');print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true',help='Build a new immutable final snapshot after all planned runs finish')
    args=parser.parse_args()
    if not args.watch:build();return
    while True:
        jobs=read(DATA/'queue.json')['jobs'];count=sum((run_dir(j)/'completion.json').exists() for j in jobs)
        if count==100:build();return
        if any((run_dir(j)/'failure.json').exists() for j in jobs):
            print('Recorded failure retained; no automatic final publication',flush=True);return
        time.sleep(30)

if __name__=='__main__':main()

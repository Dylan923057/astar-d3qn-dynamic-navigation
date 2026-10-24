# 持续弱监督新版：补齐模块消融与障碍密度证据

**2026-10-24已全部完成。** 两种新版消融共10次20万步训练，30组新评估15000回合，90组复用评估45000回合，报告及五组PNG/PDF/SVG图均已生成。

[五seed结果报告](../../../../results/asvc_persistent_paper_completion_v1/analysis_20261024_025808_465075/REPORT.md) · [上传包与详细解读](../../../../results/analysis_upload/asvc_persistent_paper_completion_v1/LATEST.md)。以下运行命令保留用于复现，查看或上传无需重新训练。

本轮只补与当前方法版本对应的证据。当前完整方法的动作监督权重从1衰减至0.1后持续保留，A*动作代选在10万步退出。奖励、观测、网络、地图、风险筛选、回放和训练预算不变。

|实验|内容|新增工作|
|---|---|---|
|新版去TD目标约束|取消TD目标裁剪，保留A*代选日程和持续0.1动作监督|5个seed，各20万步|
|新版去A*动作代选|始终由网络探索选动作，保留TD目标约束、监督标签及持续0.1动作监督|5个seed，各20万步|
|两个新消融的最终确认|使用各自20万步最终模型，评估原冻结500场景|10组，共5000回合|
|新版密度测试|已有新版5模型，测试原冻结v2的1、3、5、7障碍场景，每档500场景|20组，共10000回合；不训练|

完整方法、D3QN、DQfD、A*初始化对照均复用现有模型。新版最终500结果也复用；旧版ASVC密度成绩以原版标签保留。三个旧基线和原版ASVC的80组密度结果，以及新版完整方法和去监督对照的10组最终500结果，共90组缓存已按模型SHA、场景SHA、场景顺序、评估协议和逐场景统计核对。

去动作监督复用旧`advice_bound`的五seed。除监督外的训练配置、初始化、回放及预算一致；关闭监督后，在0、89999、90000、99999、100000、150000、200000训练时钟下，用相同样本执行更新，网络、目标网络、优化器及学习随机状态完全一致。这个核对不依赖结果是否好看。

## 一条命令运行全部

在PowerShell运行，默认并行2个训练任务。协调进程在当前终端显示进度，子进程日志写入文件，不弹出新窗口。

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 -u scripts/run_persistent_paper_completion.py --run --workers 2
```

顺序是检查/冻结 → 10次训练 → 新消融最终500评估（完整方法及去监督结果复用） → 新版密度评估（旧对照结果复用） → 报告和图表。不会重跑已完成的完整方法训练。

只有一个训练任务并行时，将`--workers 2`改成`--workers 1`。正式任务默认使用原配置的CUDA设备。评估统一使用CPU，与旧评估协议一致。

## 中断续跑

原进程停止后运行：

```powershell
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 -u scripts/run_persistent_paper_completion.py --run --workers 2 --resume
```

已完成任务会校验后跳过；中断任务恢复完整学习状态、回放、随机状态和环境生成器；未启动任务从头开始。恢复到最近一个已保存的回合边界，最多回退到上一次约1万步检查点。评估按模型和密度自动跳过已完成组。

若出现数值非有限值，保留该seed、失败步数和日志，继续其余预定任务；不更换seed、不调参偷偷重跑。图表将该seed标为失败，缺少有效seed的方法不会用其余种子冒充五seed平均。

## 单独执行

只检查，不开始正式训练或评估：

```powershell
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 scripts/run_persistent_paper_completion.py
```

只训练10次消融：

```powershell
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 -u scripts/run_persistent_paper_completion.py --train-only --workers 2
```

只用现有新版模型做密度评估并出图（不依赖新增消融训练完成）：

```powershell
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 -u scripts/run_persistent_paper_completion.py --density-only
```

已有全部训练后，只执行剩余评估和汇总：

```powershell
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 -u scripts/evaluate_persistent_paper.py --evaluate
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 scripts/analyze_persistent_paper.py
```

## 文件位置与验证

- [配置](../../../../configs/asvc_persistent_paper_completion_v1.yaml)：只登记两个模块移除和四档密度测试。
- [训练与运行入口](../../../../scripts/run_persistent_paper_completion.py)：检查、两个并行任务、续跑和后续步骤。
- [实验共用实现](../../../../scripts/persistent_paper_common.py)：复用原冻结训练循环，两个消融都保持监督下限0.1。
- [评估入口](../../../../scripts/evaluate_persistent_paper.py)：新模型自主评估和旧数据核对。
- [汇总与绘图](../../../../scripts/analyze_persistent_paper.py)：五seed均值、样本标准差、同seed差值、曲线和密度图。
- [检查结果](../../../../results/asvc_persistent_paper_completion_v1/preflight.json)：正式训练前的核对证据。
- `results/asvc_persistent_paper_completion_v1/PROGRESS.md`：运行时的十次训练进度。
- `outputs/asvc_persistent_paper_completion_v1/logs/`：逐方法/seed日志。
- `results/asvc_persistent_paper_completion_v1/LATEST.md`：运行完成后的报告入口。

新增9项测试覆盖动作代选移除、TD裁剪移除、持续监督及真实TD动作、风险筛选、自主评估、历史90k前一致性、状态恢复、失败保留和重复启动锁；两个消融分别做跨监督退出点的24步完整断点恢复检查。正式训练中，0至8万步的九个检查点必须与相应旧消融参数完全一致，9万步开始才允许因持续监督而分化。此约束验证的是新旧同一种消融的前缀，并非要求删模块后与完整方法保持相同轨迹。

模型选择固定为20万步最终检查点；所有seed和验证点保留，不平滑曲线、不依据500场景成绩调参。密度测试是同地图适应性证据，不是跨地图泛化。DQfD额外2万次离线更新和永久示范仍需在论文中注明。

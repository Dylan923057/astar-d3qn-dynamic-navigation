# GitHub同步范围

最新批次（2026-10-27）：[四方法多任务外部基线与静态/动态配对诊断](MULTITASK_COMPARISON_UPLOAD.zh-CN.md) · [完整阅读包](../results/analysis_upload/asvc_multitask_external_v1/LATEST.md)。新增D3QN/DQfD各2seed、复用C/D四次，共8条500000步曲线；全部完成，无失败。含408个检查点、20400条验证记录、AULC、配对诊断、失败轨迹、学习趋势、冻结输入/源码和示范覆盖限制。模型、回放及文档转换工具保留本地，D不升级，未见目标测试及追加训练均未启动。

本批使用独立命令，不混入范围外未提交改动：

```powershell
# 只读校验/预览，不训练、不评估、不暂存、不联网
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_multitask_comparison_github.ps1 -Preview
# 用户自行执行提交和推送；不使用代理时省略ProxyUrl
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_multitask_comparison_github.ps1 -ProxyUrl http://127.0.0.1:7897
```

克隆后只读校验：`python scripts/package_multitask_comparison.py --verify`，仅需标准库；原工作区增加`--check-sources`校验日志、模型及断点来源。此前C/D单独阅读包和旧命令保留如下。

最新批次（2026-10-26）：[同地图多起终点与风险自适应监督预实验](ASVC_MULTITASK_PILOT.zh-CN.md) · [阅读包](../results/analysis_upload/asvc_multitask_risk_pilot_v1/LATEST.md)。C/D 各 seed 0、1，共四次训练，全部完成 50 万环境步。D 的最终平均碰撞率更低，但成功率更低、超时更多，未显示学习速度或稳定性的持续增益；两个测试池和正式消融均未运行。

本批独立上传命令，不混入已有 README、旧实验和其他未提交改动：

```powershell
# 只读预览：校验阅读包及本地来源，不训练、不评估、不暂存
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_multitask_evidence_github.ps1 -Preview
# 由用户执行提交和推送；不使用代理时省略 ProxyUrl
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_multitask_evidence_github.ps1 -ProxyUrl http://127.0.0.1:7897
```

克隆后只读校验：`python scripts/package_multitask_evidence.py --verify`，只需 Python 标准库。原工作空间可增加 `--check-sources` 校验原始日志与模型 SHA。权重、完整回放断点、绘图缓存和实时输出保留本地；阅读包内包含冻结输入、源码、四次运行记录、场景明细、监督诊断和图表。以下为历史批次。

最新批次（2026-10-24）：[持续弱监督新版模块消融与密度阅读包](../results/analysis_upload/asvc_persistent_paper_completion_v1/LATEST.md)。10次新训练全部20万步完成；120组评估共60000回合，其中新增15000、复用45000。包含逐seed与逐场景数据、训练诊断、冻结输入及源码、五组PNG/PDF/SVG图和结果解读。

本次使用独立上传脚本，不改变上一批的上传入口。数据已经打包，以下命令只校验/提交/推送，不训练、不重跑评估：

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'
# 只读预览
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_persistent_paper_github.ps1 -Preview
# 提交并推送到现有origin/main；保持此前Clash代理开启
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_persistent_paper_github.ps1 -ProxyUrl http://127.0.0.1:7897
```

不用代理时删除`-ProxyUrl http://127.0.0.1:7897`。脚本自动找到本机Git和Python、验证数据包及原始来源、限定此次提交范围，并保护范围外的暂存内容。权重、恢复文件、批量失败轨迹和奖励实验结果留在本地。本次准备工作不执行commit或push。

只校验阅读包：`python scripts/package_persistent_paper_completion.py --verify`；原工作空间可增加`--check-sources`核对本地模型与原始文件。克隆后的只读校验无需PyTorch、模型权重或原始outputs。

结果解读需保留：TD消融支持稳定性作用，代选与监督主要改善学习速度；新版7障碍成功率95.68%，低于原版96.64%，不能写成所有指标全面提升。以下为历史批次记录。

最新批次（2026-10-23）：[持续弱监督五seed阅读包](../results/analysis_upload/asvc_persistent_supervision_five_seeds_v1/LATEST.md)。五个seed各20万步，包含五方法主对比、最终500场景明细、六组PNG/PDF/SVG图、三seed方案选择历史、训练日志和冻结代码。前三个seed复用，后两个补跑；新方法最终评估2500回合，四个对照复用10000回合。

数据包已经整理好，以下命令不重新训练或评估。PowerShell中先进入项目，再预览或上传：

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'
# 只读预览及SHA/统计校验
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -PersistentSupervisionOnly -Preview
# 提交并推送到已配置的origin/main
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -PersistentSupervisionOnly -ProxyUrl http://127.0.0.1:7897
```

最后一行沿用此前Clash代理；不用代理时删除`-ProxyUrl http://127.0.0.1:7897`。脚本自动定位本机Git，使用专用提交信息，不需要手动`git add .`。已暂存的范围外文件会触发停止，以免混入本次提交；不会清空这些暂存内容，也不会强制推送。

此范围上传相关代码、配置、测试、文档、冻结登记和整理后的结果。模型权重、回放恢复文件、原始批量轨迹、奖励实验结果和其他历史改动留在本地。训练冻结曾覆盖全部src模块，所以其源码字节也保留以便核对；不代表本次采用了奖励重设计。单独校验：`python scripts/package_persistent_supervision.py --verify`。只有本地原始文件齐全时才使用`--check-sources`。

本次整理不执行commit或push，由用户运行上面的上传命令。以下为历史批次记录，旧的`-AsvcOnly`不包含持续弱监督结果。

最新批次（2026-10-16）：[ASVC-D3QN补实验阅读包](../results/analysis_upload/asvc_paper_supplement_v1/LATEST.md)及[结果与图表说明](ASVC_PAPER_SUPPLEMENT.zh-CN.md)。五次初始化对照训练、85组最终评估、42500条场景记录全部完成；无需重新训练、评估或打包。使用本批专用范围，旧选项不包含本批。

```powershell
# 只读预览与校验
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -AsvcOnly -Preview
# 提交并推送；保持此前Clash代理开启，不使用代理时省略ProxyUrl
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -AsvcOnly -ProxyUrl http://127.0.0.1:7897
```

本批阅读包约98.84 MiB，上传范围含源码、冻结数据与文档约112 MiB；单文件最大约2.73 MiB。上传前自动核对包内SHA和完整实验记录，`.gitattributes`防止换行转换破坏冻结文件SHA。权重、恢复断点、全量原始轨迹及重复实时结果留在本地；不包含范围外历史轨迹改动。本次整理未创建提交或推送。以下为历史批次记录。

最新批次：[新版完整方法模块消融阅读包](../results/analysis_upload/latest_all_eligible_modules_v1/LATEST.md)及[结果说明](LATEST_MODULE_ABLATIONS.zh-CN.md)。新增两组五seed共10次200000步训练、10份新最终模型的500场景评估均完成，另25份历史最终评估按模型、场景与协议SHA复用。上传包括本批源码、两份配置、测试、冻结登记、检查报告、统计表、曲线和阅读包；权重、回放及原始outputs保留本地。旧脚本的 `-ExecutionConsistencyOnly` 仅覆盖上一批，不包含本批模块消融。

当前批次：[执行一致性筛选五seed阅读包](../results/analysis_upload/execution_consistency_seeds0to4_v1/LATEST.md)、[实验与结果说明](EXECUTION_CONSISTENCY.zh-CN.md)。新增五次200000步对照全部完成，原完整组合五seed复用；十个最终模型复核原50场景，210验证点、10500条逐场景记录。同一冻结500最终确认已完成，新增五模型2500回合，原组五seed按SHA与协议复用，合计5000条记录和62条失败轨迹均已审计，不重跑评估或训练。阅读包包含分开的训练/最终报告及合并配对表，脚本阻止上传缺少500确认的旧包。

已经完成分析和打包，在仓库根目录运行这一行即可提交推送，保持此前Clash代理开启：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ExecutionConsistencyOnly -ProxyUrl http://127.0.0.1:7897
```

只读预览将末尾ProxyUrl参数替换为`-Preview`。脚本验证最新包，限定本批源码、配置、冻结登记、报告和阅读包；权重、回放断点、重复报告及范围外历史轨迹不会上传。没有自动提交或推送；这条命令由用户执行。旧批次记录与命令保留如下。

本次同步目标为 `origin/main`：`Dylan923057/astar-d3qn-dynamic-navigation`。

最新批次：[分阶段撤教与DQfD无预训练五seed阅读包](../results/analysis_upload/replication_seeds0to4_v1/LATEST.md)，涵盖20次运行、全部420验证点、逐seed及配对统计、源码/配置/冻结登记和失败示例。四方法五seed的同一冻结500场景最终确认已完成，共10000条记录；10份原结果按SHA和协议复用，补评10份新最终模型。上传脚本会阻止提交缺少最终500确认的旧阅读包。

只同步本批源码与整理数据（不上传权重或回放断点，不混入范围外历史轨迹改动）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ReplicationOnly -Preview
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ReplicationOnly -ProxyUrl http://127.0.0.1:7897
```

第二条沿用此前Clash的7897端口，需保持代理开启；不用代理时省略`-ProxyUrl`。脚本先校验最新阅读包，再提交并推送；预览仅只读，不训练、不改索引。新报告及重算命令见[五seed说明](REPLICATION_FIVE_SEEDS.zh-CN.md)。

以下为此前单seed批次的同步记录，其中HEAD及origin/main是当时的缓存值。

2026-10-12本次整理：[对比诊断与单seed对照阅读包](../results/analysis_upload/diagnostics_withdrawal_update_v1/20261012_002943_168455/README.md)、[GPT分析请求](../results/analysis_upload/diagnostics_withdrawal_update_v1/20261012_002943_168455/GPT_ANALYSIS_PROMPT.md)。包内193个文件约9.21MiB，包含新增实现、配置与冻结登记、运行日志、曲线、配对表、模型路径/SHA和精选失败轨迹。DDQNA短探针仅发布更正后0–5万步AULC；重复诊断、旧错误AULC报告及全量逐动作Q明细保留本地。

本地HEAD与缓存origin/main均为`7deff74`，本次整理未联网核验远端新提交、未创建提交或推送。新增`-DiagnosticsOnly`范围只同步这批诊断/对照代码和报告，保留那份仅多空行的历史轨迹修改及其他范围外暂存工作；若范围外已有暂存内容，脚本停止，不自动清理。

保持既有Clash代理开启，在仓库根目录执行这一行（提交并推送，不训练）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -DiagnosticsOnly -ProxyUrl http://127.0.0.1:7897 -Message "Add diagnostics and paired seed0 withdrawal controls"
```

可选预览，不改索引、不提交、不推送：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -DiagnosticsOnly -Preview
```

本次阅读包校验命令（已经通过，无需重复生成包）：

```powershell
python scripts/package_diagnostic_updates.py --verify results/analysis_upload/diagnostics_withdrawal_update_v1/20261012_002943_168455
```

DQfD取消离线预训练及分阶段撤教都已完成seed0，seed1–4未运行；它们尚未做独立500最终评估。早期preflight的“未训练”仅说明检查发生时的状态，当前完成状态以阅读包run_logs下的result.json/completion.json为准。权重及全量更新/轨迹留在outputs，不能因克隆后缺失权重自动重新训练。详情见[新增工作说明](DIAGNOSTICS_WITHDRAWAL_UPDATE.zh-CN.md)。

以下是此前已上传的结果和历史同步命令。

2026-10-10此前补充：[DDQNA动作机制适配版五seed阅读包](../results/analysis_upload/whole_map_91701_ddqna_adapter_seeds0to4_20261010_112856_785796/README.md)、[GPT分析请求](../results/analysis_upload/whole_map_91701_ddqna_adapter_seeds0to4_20261010_112856_785796/GPT_ANALYSIS_PROMPT.md)。新增五次200000在线步训练全部完成，两方法十个最终模型完成5000次自主评估。210个学习检查点、5761条验证失败、2546条最终失败轨迹重新核对；阅读包约20.89MiB、123个文件，含完整五seed运行日志、源码/配置快照、配对结果、成本、图表和精选完整轨迹。

上传包含本次配置、代理、训练/分析/打包脚本、测试、冻结登记、文献核对和报告。`-ValueRepairOnly` 已扩充覆盖DDQNA适配版及此前十一方法这条研究线；范围外的历史实验未提交修改保留。权重与批量原始轨迹继续留在本地，不自动训练或重跑评估。本次的权重路径为 `outputs/whole_map_91701_ddqna_adapter_v1/ddqna_d3qn/seed_<0–4>/model_final.pth`。

本次已经整理完成。仓库根目录运行这一行上传，保持既有Clash代理开启：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ValueRepairOnly -ProxyUrl http://127.0.0.1:7897 -Message "Add DDQNA adaptation five-seed results and final evaluation"
```

只检查现成包、不训练、不推送：

```powershell
python scripts/package_ddqna_adapter_analysis.py --verify results/analysis_upload/whole_map_91701_ddqna_adapter_seeds0to4_20261010_112856_785796
```

以下为此前十一方法的同步记录和历史命令，继续保留。

最新入口：[十一方法五seed阅读包](../results/analysis_upload/whole_map_91701_eleven_methods_seeds0to4_20261009_183634_447770/README.md)、[可复制的GPT分析请求](../results/analysis_upload/whole_map_91701_eleven_methods_seeds0to4_20261009_183634_447770/GPT_ANALYSIS_PROMPT.md)。

55次200000在线步正式训练均已完成。1155个原50场景自主验证检查点、20286条训练期间验证失败轨迹已核对；55份最终模型统一完成新冻结500场景的27500次自主评估，全部3688条最终失败轨迹重新核对。均值与样本标准差按五个配对训练seed统计，不挑最佳检查点，不用新500调参。

最新阅读包约85.19MiB，包含尾段5次及DQfD/TD限制/无风险筛选15次新增完整运行日志、源码与配置快照、监督标签覆盖率、示范动作分布、55条学习曲线、194条精选完整失败轨迹、模型路径与SHA、文件SHA及完整报告链接。原35次日志复用历史阅读包：[七方法入口](../results/analysis_upload/whole_map_91701_teaching_efficiency_seeds0to4_20261008_113945_236854/README.md)、[原六方法入口](../results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/README.md)。旧500与新500的结果分别保留。

同步包含相关实现、配置、测试、分析及打包脚本、冻结500场景、7260条压缩A*示范、实验报告、CSV、图表及精选轨迹。训练前登记的27份源文件、冻结数据和审计阅读包均设置Git字节保护，避免Windows换行转换导致SHA变化。

55份最终权重、预训练权重、其他检查点和批量原始outputs保留在本地；不新增上传权重、缓存、重复ZIP或批量原始轨迹。历史已使用Git LFS的论文证据权重维持原存档方式。

本地权重位于以下目录的 `<method>/seed_<seed>/model_final.pth`：

- `outputs/whole_map_91701_value_repair_v1/`
- `outputs/whole_map_91701_teaching_efficiency_v1/`
- `outputs/whole_map_91701_supervision_tail_v1/`
- `outputs/whole_map_91701_dqfd_comparison_v1/`

原50场景入口 `outputs/foundation_dynamic_ratios_91701_20261002_v1/fixed_validation_scenarios.json` 是必要输入，继续单独保留。仅克隆仓库不会得到本地权重或完整原始轨迹；直接查看已生成报告即可，不应因权重缺失自动重训。

在项目根目录运行这一行，保持Clash代理开启；创建提交并推送main，不启动训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ValueRepairOnly -ProxyUrl http://127.0.0.1:7897 -Message "Add eleven-method five-seed results and frozen 500-scene evaluation"
```

沿用的 `-ValueRepairOnly` 选项现在覆盖价值修正、教学消融、尾段延期及DQfD十一方法这条研究线。其他历史实验的未提交修改保持不变；若范围外已有暂存文件，脚本停止，避免混入提交。脚本核对main及目标仓库、不强制推送，代理仅作用于本次Git调用。

只预览、不修改索引或推送：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ValueRepairOnly -Preview
```

若所选范围内存在已跟踪但现在被忽略的文件，脚本仅停止跟踪对应索引项，不删除本地文件、不重写历史。省略范围选项则同步整个项目所有未忽略改动。

需要重新整理时，依次运行下列非训练命令；每次生成独立输出，不覆盖原结果：

```powershell
python scripts/summarize_dqfd_comparison.py --analyze
python scripts/summarize_dqfd_comparison.py --evaluate
python scripts/package_dqfd_comparison_analysis.py
```

本次已完成这些整理和评估，无需上传前重复执行。校验现成阅读包：

```powershell
python scripts/package_dqfd_comparison_analysis.py --verify results/analysis_upload/whole_map_91701_eleven_methods_seeds0to4_20261009_183634_447770
```

# GitHub同步范围

本次同步目标：`origin/main`，仓库 `Dylan923057/astar-d3qn-dynamic-navigation`。

保留代码、配置、测试、地图、动态路线池、独立实验包中的代码与说明、实验报告、CSV指标、图表和选取的失败轨迹。
README已列出最新七组×seed0–4的20万步正式结果、500场景独立最终评估，以及此前实验。

最新入口：[七方法阅读包](../results/analysis_upload/whole_map_91701_teaching_efficiency_seeds0to4_20261008_113945_236854/README.md)、[GPT分析请求](../results/analysis_upload/whole_map_91701_teaching_efficiency_seeds0to4_20261008_113945_236854/GPT_ANALYSIS_PROMPT.md)。新包约21.84MiB，含新增五seed完整训练/验证/价值/监督覆盖日志、登记源码与支持源码快照、38条精选完整失败轨迹和SHA清单。原六组30次日志复用此前包；七组完整曲线、17500条独立场景结果、3496条失败行为及最新复核报告通过仓库内相对链接读取，避免重复复制整张场景表。

500冻结场景数据、独立配置、测试、实现/分析/打包脚本及相关源码一起上传。35份本地最终权重只记录路径和SHA，不上传；约1.65GB独立原始失败轨迹继续保留在outputs。当前另一个历史实验轨迹的修改不在本次范围内。

在项目根目录运行下列一行，保持Clash代理开启；脚本创建提交并推送main，不启动训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ValueRepairOnly -ProxyUrl http://127.0.0.1:7897 -Message "Add seven-method five-seed teaching efficiency results and 500-scene evaluation"
```

此前供GPT阅读的入口：[六方法五seed结果包README](../results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/README.md)，以及[可复制分析请求](../results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/GPT_ANALYSIS_PROMPT.md)。包约70MiB，含30次运行的全部训练日志、630次验证检查点、31500条场景明细、17539条失败行为、价值日志、曲线、代码及配置快照、精选完整轨迹。包含组合组seed3的全部四次最终碰撞，旧两seed分析包继续保留。

包内`manifest.json`记录文件SHA256。7份训练时登记源文件在30次运行间一致且与当前文件相符；其他支持源码是整理时快照，不能冒充训练时已记录指纹。失败行为CSV只删去重复的位置尾段列，保留每条失败及所有统计指标；精选轨迹保留完整步骤。
新包使用独立Git属性保留原始文件字节，避免Windows换行转换导致克隆后SHA不一致。

不新增上传本地训练权重和原始outputs、缓存、临时PDF解析依赖、文献PDF、重复ZIP，以及分析上传包中的大段`failure_trajectories.json`副本。
新增加忽略旧交接实验`final_failures_*.json`和价值修正分析的批量`navigation_examples_*.json`副本；原文件继续留在本地。新分析包的`trajectories/`保留有明确选择理由的完整示例，全部失败行为分类仍可读取。
`results/paper_evidence`内原本纳入版本控制的精选权重继续使用已有Git LFS规则；不改变这些历史实验的存档策略。

以下文件是当前训练入口的必要输入，单独保留：

`outputs/foundation_dynamic_ratios_91701_20261002_v1/fixed_validation_scenarios.json`

它记录50个冻结验证场景，不是训练权重，也不包含test；保留原路径以便入口核对同一批验证场景。

本次训练权重仍在本地`outputs/whole_map_91701_value_repair_v1/<method>/seed_<seed>/model_final.pth`；分析包仅记录模型路径和SHA。
旧foundation仍需从本地原始outputs恢复。仅克隆代码与报告不会得到这些未上传的训练权重，脚本不能因此自动重训foundation。
`analyze_runtime_path_formal.py`依赖本地原始训练日志、轨迹和权重；GitHub直接查看已经生成的分析报告、CSV和图表即可。

已有部分缓存和文献文件早已被跟踪，单加`.gitignore`不会移除它们。
同步脚本使用`git rm --cached`停止跟踪所选范围内当前被忽略的文件，仅操作Git索引，不删除本地文件，不重写历史。
随后加入项目改动、创建提交、推送当前`main`到`origin/main`；任何一步失败立即停止，不使用强制推送。

本次使用`-ValueRepairOnly`，只提交价值修正及教学效率相关结果、分析包、代码和文档，保留其他历史实验的未提交修改。已有范围外暂存文件时脚本停止，避免混入提交。省略此选项则同步整个项目所有未忽略的改动。

在项目根目录运行这一行，保持Clash代理开启。沿用已验证可访问GitHub的本地代理，设置仅作用于这次Git命令，不修改全局配置，不启动训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ValueRepairOnly -ProxyUrl http://127.0.0.1:7897
```

只检查待上传清单、不修改索引、不提交或推送：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ValueRepairOnly -Preview
```

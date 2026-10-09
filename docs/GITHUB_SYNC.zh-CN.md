# GitHub同步范围

本次同步目标为 `origin/main`：`Dylan923057/astar-d3qn-dynamic-navigation`。

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

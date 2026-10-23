# 持续弱监督：五seed复核与新版候选

目标是确认持续弱监督能否作为新版ASVC。复用已完成的seed0–2，仅新增seed3、4，各训练20万环境步。配置和实现保持一致：A*代选在10万步退出，动作监督权重从1按原日程衰减到0.1后保持。奖励使用原版，不混入奖励重设计实验。

前三个seed已经参与方案选择；后两个seed是补充复核，不能把合并五seed写成五个全新、盲选实验。

## 判定口径

在看到新增种子结果之前登记以下描述性门槛：

- seed3和seed4的10–20万步AULC及最低成功率，相对原版均不降低。
- 五seed平均的后半程AULC及最低成功率提高。
- 五份20万步最终模型在同一冻结500场景上的平均安全成功率，相对原版下降不超过1个百分点。

这些是候选方案的工程选择标准，不是显著性检验。所有种子和检查点都保留；最终500只评估最终模型，不选最好检查点，不据此调节监督权重。若某条未通过，报告具体差异，保留候选状态，不自动宣布替换。

## 入口

- [补跑配置](../configs/asvc_persistent_supervision_five_seeds_v1.yaml)
- [补跑入口](../scripts/run_persistent_supervision_five_seeds.py)
- [最终评估、汇总和绘图](../scripts/review_persistent_supervision_five_seeds.py)
- [实时进度](../results/asvc_persistent_supervision_five_seeds_v1/PROGRESS.md)

```powershell
& "D:\softwares\Anaconda3\envs\astar-drl\python.exe" scripts/run_persistent_supervision_five_seeds.py
& "D:\softwares\Anaconda3\envs\astar-drl\python.exe" scripts/run_persistent_supervision_five_seeds.py --train --seeds 3 4
& "D:\softwares\Anaconda3\envs\astar-drl\python.exe" scripts/review_persistent_supervision_five_seeds.py --evaluate
```

本批已经后台启动，不要重复启动相同种子。中断后先确认进程已停止，再用`--resume`恢复完整训练状态。

输出目录为`outputs/asvc_persistent_supervision_five_seeds_v1`，分析目录为`results/asvc_persistent_supervision_five_seeds_v1`。自动汇总结束后，该分析目录的`LATEST.md`链接正式报告。

## 重画范围

主对比包括原ASVC、持续弱监督ASVC、DQfD、D3QN、A*初始化＋TD约束。输出五seed均值与样本标准差曲线、逐seed稳定性、配对差、最终500碰撞与超时、教学日程，均提供PNG、PDF和SVG。

已有消融、障碍密度和轨迹图属于原版方法，不能仅替换图例就当作新版证据。本轮先更新有实际新数据支撑的主对比和稳定性图。DQfD仍有额外预训练及永久示范，图表注明预算差异。

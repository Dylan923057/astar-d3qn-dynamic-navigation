# 当前主方法：ASVC持续弱监督版

五seed及冻结500场景复核通过预先登记的描述性门槛，按本轮用户授权采用为当前主方法。原版实验、模型和图表保留为历史版本。

动作代选按原计划在10万步退出；动作监督权重下降到0.1后持续保留。奖励、网络、观测、TD约束与风险筛选沿用原版。评估只使用网络。

[五seed报告](../results/asvc_persistent_supervision_five_seeds_v1/review_20261023_131739_114918/REPORT.md) · [主对比图](../results/asvc_persistent_supervision_five_seeds_v1/review_20261023_131739_114918/01_learning_curves.png) · [逐seed曲线](../results/asvc_persistent_supervision_five_seeds_v1/review_20261023_131739_114918/02_individual_seeds.png)

## 代码和训练入口

- 实现：[persistent_teaching.py](../src/astar_d3qn/agents/persistent_teaching.py)。
- seed0–2：[run_persistent_teaching.py](../scripts/run_persistent_teaching.py)，使用`--variants supervision_floor`。
- seed3–4：[run_persistent_supervision_five_seeds.py](../scripts/run_persistent_supervision_five_seeds.py)。
- 完整模型路径及选择证据：[当前方法登记](../data/asvc_current_method.json)。
- 五个种子已经完成，不要为重画图重复训练。

重画当前五seed主图（复用已完成的最终500评估）：

```powershell
& "D:\softwares\Anaconda3\envs\astar-drl\python.exe" scripts/review_persistent_supervision_five_seeds.py
```

## 论文图表边界

本次更新学习曲线、逐seed稳定性、配对差、最终500碰撞/超时及教学日程，包含PNG、PDF和SVG。
旧版去TD、去代选等消融，以及障碍密度和轨迹图，不能改名当作新版结果。论文使用新版作为主方法时，这些对应实验仍需另行补齐或明确标为原版。
三seed已参与方案选择，后两seed用于复核；结论限于当前固定地图及场景分布，不是跨地图泛化保证。

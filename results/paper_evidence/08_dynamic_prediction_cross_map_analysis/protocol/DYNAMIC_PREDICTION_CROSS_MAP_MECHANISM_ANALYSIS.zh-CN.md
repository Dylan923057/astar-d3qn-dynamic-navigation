# 动态占用预测辅助跨地图机制分析协议

## 目标

在不重新训练、不读取测试划分的前提下，解释 `global_prediction` 在 map01、map02 为正而在 map03 为负的原因，并约束论文可以提出的结论。

## 数据范围

- 环境统计仅使用 `risk_handover_v1` 的 train 和 validation 场景定义。
- 导航性能仅使用 A `time_decay` 与 B `global_prediction` 已冻结的 validation 曲线。
- 预测性能使用 B 验证曲线中已经保存的 `prediction_positive_f1` 和 `decision_zone_f1`。
- 不使用 test split，不重新训练，不调整方法参数。

## 环境指标

- 动态障碍数量及分布。
- 因果冲突障碍数量及占比。
- 首次冲突步数、归一化路径位置和 early/middle/late 分布。
- 每个冲突场景的风险位置数量。
- 障碍方向、`move_every`、格/步速度和路线长度。
- 配对设计下的 control/conflict 场景比例。

## 预测与导航指标

- 三 seed 平均 prediction F1 和 decision-zone F1。
- A/B 平均验证 AUC、B-A、提升 seed 数量。
- 达到连续安全阈值的平均步数。
- 最后 5 万步安全成功率波动。
- prediction F1 与导航 AUC 效应的描述性相关性；不把三地图相关系数解释为统计推断。

## 解释边界

- 只有在障碍数量、速度或冲突频率确实存在差异时，才讨论复杂度条件效应。
- 如果 map03 的 prediction F1 不低但导航收益为负，则不能归因为预测头失败，应解释为预测表示没有稳定转化为 Q 策略收益。
- 当前分析只用于机制、边界和论文叙事，不用于结果驱动调参。

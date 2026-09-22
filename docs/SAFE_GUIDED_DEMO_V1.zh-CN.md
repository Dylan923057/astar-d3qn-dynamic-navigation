# 状态级安全 A* 引导 v1

## 为什么修改

`conflict_adaptive_demo` 根据冲突 EMA 调整整个 replay 的示范比例，但正式结果中平均比例仍为约 24.7%–24.8%，只在 Office 改善，在 Parcel 和 Warehouse 下降。一个局部冲突不应同时削弱其他位置仍然正确的 A* 知识。

## 新方法

策略名为 `safe_guided_demo`，包含两个互相分离的部分：

1. 与 Prefill 相同，训练开始时把静态 A* transition 放入普通 replay；这些旧示范可以被在线经验自然淘汰，不设永久分区。
2. 每个真实在线状态若位于示范覆盖位置，则读取该位置的 A* 动作。只有动作既不进入动态障碍当前格、也不进入下一步预测格时，才把它保存为该在线 transition 的 `safe_demo_action`。

训练批次中的 `safe_demo_action` 使用 large-margin 辅助损失，默认 margin `0.8`、权重 `1.0`，与项目已有的 DQfD 默认量级一致。若该位置所有 A* 动作都危险，则不提供标签、不虚构等待或绕行动作，只使用真实 TD 目标学习。

## 与失败方法的区别

- 旧方法：某处发生冲突后，短暂降低所有位置的示范采样比例；
- 新方法：只跳过当前状态的危险 A* 标签，其他安全位置继续接受引导。

静态示范自身的动态通道仍然为零；动态安全判断只发生在带真实动态历史的在线状态上，不能表述成“给旧静态示范补上动态标签”。

## 日志与配置

训练 CSV 新增：`safe_guidance_candidate_count`、`safe_guidance_applied_count`、`safe_guidance_blocked_count`、`safe_guidance_coverage_rate`、`safe_guidance_application_rate`、`safe_guidance_mean_risky_fraction`、`safe_guidance_margin_loss`、`safe_guidance_batch_count_mean`。

三份配置为：

- `configs/dynamic_spatial_generalization_office_safe_guided_v1.yaml`；
- `configs/dynamic_spatial_generalization_parcel_safe_guided_v1.yaml`；
- `configs/dynamic_spatial_generalization_warehouse_safe_guided_v1.yaml`。

本轮通过 103 项单元测试和 Office seed 0 的 4-episode smoke，没有启动正式训练。由于 v4 test 已被用于方法分析，新方法在 v4 上的结果只能作为开发结果；论文最终结论需要另外冻结未见确认场景后只评估一次。

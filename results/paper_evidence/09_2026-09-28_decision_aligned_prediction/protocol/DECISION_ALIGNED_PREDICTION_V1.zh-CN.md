# Decision-Aligned Prediction V1 冻结协议

## 研究问题

动态占用预测可以学到，但原始 prediction loss 对共享 CNN 的梯度可能与 TD loss 冲突，并对 Q-learning 产生负迁移。`decision_aligned_prediction` 只修改共享表示的梯度合成方式，不修改 D3QN、time-decay 示范回放、奖励、探索率、经验池、网络规模或动态占用预测任务。

## 方法

每个训练 batch 分别计算 TD loss 与 prediction loss 对 `policy_network.encoder` 的梯度 `g_td` 和 `g_pred`。

- 当 `g_td · g_pred >= 0` 时，保留原预测梯度。
- 当 `g_td · g_pred < 0` 时，使用 `g_pred' = g_pred - (g_pred · g_td / ||g_td||²) g_td`。
- 共享 CNN 使用 `g_td + 0.1 g_pred'`；Q 分支只使用 TD 梯度；预测头继续使用完整的 `0.1 g_pred`。
- 投影前后都沿用原有全模型梯度裁剪。

固定参数为 prediction loss weight `0.1`、pos_weight `20`、预测头 hidden dim `128`。空间预测权重保持全局均匀 `1.0`。

## 第一阶段

- 地图：`irregular_workcell_91703`（`map-index 2`）。
- seed：`0 1 2`；每个分支固定训练 `200k`。
- A：既有 `time_decay`；B：既有 `global_prediction`；D：新增 `decision_aligned_prediction`。
- A/B 复用 `outputs/dynamic_prediction_third_map_v1`，D 写入 `outputs/decision_aligned_prediction_v1`。
- 只读取 validation，不生成或读取 test；不得根据结果调参。

主要指标为 validation conflict safe-success AUC，同时报告 threshold confirmation step、最终 safe success、dynamic collision、timeout、last 50k mean/std。机制指标包括投影前后 cosine、冲突 batch 比例、修正比例、移除梯度范数比例，以及同一 batch 更新前后的 TD/prediction loss。

## 预先冻结的继续条件

只有全部满足才进入独立新地图：

1. D 的平均 validation AUC 至少比 B 高 `0.03`，且至少 `2/3` seed 高于 B。
2. D 的平均 validation AUC 不低于 A，且至少 `2/3` seed 不低于 A。
3. 每个 seed 的最终 dynamic collision 与 timeout 均不高于 A。
4. D 的 last-50k 平均 safe success 不低于 A，平均标准差不高于 A。
5. 三个 seed 均实际记录到梯度对齐更新。

若不满足，停止调参并作为负结果或证据不足报告。若满足，冻结代码和参数后才创建一张未参与设计的新地图；在查看任何结果前冻结场景划分，再对 A/B/D 各跑 3 个 seed。第二阶段通过后才扩到 5 seed 和最终 test。

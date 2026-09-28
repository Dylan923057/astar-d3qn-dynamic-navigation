# Decision-Aligned Prediction 第一阶段结果

## 实验范围

- 地图：`irregular_workcell_91703`。
- A：既有 `time_decay`；B：既有 `global_prediction`；D：新增 `decision_aligned_prediction`。
- D 使用 seed 0/1/2，每次固定训练 20 万步，并复用与 A/B 相同的逐 seed foundation。
- prediction loss weight 固定为 0.1，pos_weight 固定为 20；没有调整奖励、探索率、回放、网络规模或其他辅助模块。
- 本阶段只使用 validation，未生成、读取或归档 test 结果。
- 为保证 GitHub 上可以完整复现，本目录保留三个 foundation checkpoint、三个最终 D3QN 权重和三个 prediction-head 权重，并通过 Git LFS 管理。

## 主要结果

- A 平均 validation AUC：0.8674。
- B 平均 validation AUC：0.8368。
- D 平均 validation AUC：0.8743，比 B 高 0.0375，比 A 高 0.0069。
- D 在 2/3 seed 上高于 B，也在 2/3 seed 上不低于 A。
- 三种方法最终平均安全成功率均为 1.0，最终动态碰撞率和超时率均为 0。
- D 的 last-50k 平均安全成功率为 0.9213，低于 A 的 1.0；平均标准差为 0.1219，高于 A 的 0.0 和 B 的 0.0589。

## 机制结果

- TD 与 prediction 梯度冲突 batch 比例为 0.4878，修正比例同为 0.4878。
- 投影平均移除了 prediction 梯度范数的 0.0772。
- 平均 cosine similarity 从 0.0056 提高到 0.0828。
- 同批更新后的 TD loss 和 prediction loss 平均都下降，说明实现按预期工作。

## 冻结结论

梯度对齐消除了原始 global prediction 在平均 AUC 上的负迁移，并略高于基线，但没有满足预先登记的后期性能和稳定性条件。因此第一阶段判定为未通过：不进入新地图、不扩到 5 seed、不解锁 test，也不根据结果继续调参。该结果适合作为“机制有效但整体方法尚不稳定”的正式消融和失败分析。

## 目录

- `analysis/`：A/B/D 冻结判定、逐 seed 指标和报告。
- `config/`：实验配置快照。
- `dataset/`：冻结的完整场景 manifest 和审计表。
- `foundations/`：三个 seed 的完整 foundation checkpoint 与训练记录。
- `protocol/`：预先冻结的方法、指标和继续条件。
- `runs/`：D 组三次正式运行的训练、验证证据和最终权重，不含 test。

A/B 原始运行已归档在 `07_dynamic_prediction_third_map_validation/`，此处不重复复制。

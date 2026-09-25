# 动态占用预测辅助跨地图验证（2026-09-25）

## 实验内容

- 地图：`irregular_workcell_91702`。
- A：`time_decay`，seed 0/1/2。
- B：`global_prediction`，seed 0/1/2。
- 每次适应训练固定 20 万步；每个 seed 的 A/B 从同一基础快照分叉。
- B 固定 `prediction_loss_weight=0.1`、`prediction_pos_weight=20`，没有继续调参。
- 本阶段严格只归档验证证据，没有生成或读取测试集结果。
- foundation 和最终模型权重继续保留在本地 `outputs/`，归档只保存审计文件。

## 验证集结论

- A 的平均安全成功 AUC 为 0.9271。
- B 的平均安全成功 AUC 为 0.9590，相对 A 提高 0.0319。
- B 在 3/3 个 seed 上均优于对应 A。
- B 的最终碰撞率、超时率没有退化，后三 seed 后 5 万步波动均值与 A 相同。
- 预先登记的跨地图门槛全部通过，下一步为扩展到 5 个 seed，而不是立即查看测试集。

## 目录

- `analysis/validation_decision/`：冻结的跨地图验证判定和逐 seed 指标。
- `config/`：本轮配置快照。
- `protocol/`：实验设计、继续门槛和测试隔离规则。
- `foundations/`：三个基础模型的配置、训练和资格审计，不含权重。
- `runs/`：A/B 六次正式运行的训练与验证证据，不含权重和测试结果。

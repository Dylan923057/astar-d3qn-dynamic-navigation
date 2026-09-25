# 动态占用预测辅助第三地图验证（正式负结果）

## 实验内容

- 地图：`irregular_workcell_91703`。
- A：`time_decay`，seed 0/1/2。
- B：`global_prediction`，seed 0/1/2。
- 每次适应训练固定 20 万步；每个 seed 的 A/B 从同一基础快照分叉。
- B 固定 `prediction_loss_weight=0.1`、`prediction_pos_weight=20`，未进行结果驱动调参。
- 本阶段严格只归档验证证据，没有生成或读取测试集结果。
- foundation 和最终模型权重继续保留在本地 `outputs/`，归档只保存审计文件。

## 第三地图结论

- A 的平均安全成功 AUC 为 0.8674。
- B 的平均安全成功 AUC 为 0.8368，相对 A 下降 0.0306。
- B 仅在 1/3 个 seed 上略高于对应 A。
- 两组最终安全成功率均为 1.0，但 B 达到阈值更慢且后期波动更大。
- 预先冻结的第三地图继续门槛未通过，因此不解锁测试集、不扩 seed、不调参。

## 三地图诊断

- map01：B-A = +0.0382，2/3 seed 提升。
- map02：B-A = +0.0319，3/3 seed 提升。
- map03：B-A = -0.0306，1/3 seed 提升。
- 总计 2/3 地图、6/9 配对 seed 为正，等权地图平均效应为 +0.0132。
- 证据表明预测辅助收益具有地图依赖性，不能表述为稳健的三地图泛化改进。

## 目录

- `analysis/validation_decision/`：第三地图冻结判定和逐 seed 指标。
- `analysis/cross_map_diagnostic/`：三地图配对汇总与正式失败分析。
- `config/`：本轮配置快照。
- `protocol/`：实验设计、继续门槛和测试隔离规则。
- `foundations/`：三个基础模型的配置、训练和资格审计，不含权重。
- `runs/`：A/B 六次正式运行的训练与验证证据，不含权重和测试结果。

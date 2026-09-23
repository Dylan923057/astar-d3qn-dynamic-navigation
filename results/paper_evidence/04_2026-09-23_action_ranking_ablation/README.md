# 动态风险动作排序消融（2026-09-23）

## 内容

- A：阶段 02 已归档的 `time_decay`，本阶段不重复复制。
- B：`demo_action_margin`，seed 0/1/2。
- C：`all_action_margin`，seed 0/1/2。
- 每次训练使用第一张地图和 20 万动态交互步。
- 模型权重与大型逐步测试轨迹仍只保留在本地 `outputs/`。

## 验证集结论

- A 平均安全成功 AUC：0.7042。
- B 平均安全成功 AUC：0.5590。
- C 平均安全成功 AUC：0.4646。
- C-A：-0.2396；C-B：-0.0944；C 优于 A 的 seed 数为 0/3。
- 预登记继续条件未通过，本实现停止，不继续搜索间隔或损失权重。

测试文件是在验证集停止决策写入 `analysis/validation_decision/decision.json` 后归档，未参与方法选择。本阶段属于正式负结果，可用于消融和失败分析，不能表述为性能提升。

## 目录

- `analysis/validation_decision/`：严格基于验证曲线的 A/B/C 决策。
- `analysis/preflight/`：相同碰撞前观测的三个网络动作评分及标签覆盖。
- `analysis/failure_diagnostic/`：时间衰减最终模型的失败定位及代表图。
- `runs/`：B/C 六次正式运行的配置审计、训练、验证和固定最终测试结果。
- `branch_provenance.json`：区分复用基础模型的旧代码哈希与本轮实际分支代码哈希。

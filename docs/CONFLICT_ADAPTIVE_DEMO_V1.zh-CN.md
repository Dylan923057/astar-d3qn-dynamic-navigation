# 冲突感知 A* 示范回放 v1

## 方法定义

旧 A* 示范是在静态环境采集的，三个动态历史通道均为零。因此不能声称每条旧示范自身带有动态风险标签。v1 方法在真实在线状态中恢复该位置对应的 A* 示范动作分布，并调用动态环境的碰撞语义，测量这些示范动作与当前障碍位置是否冲突。

冲突率使用指数滑动平均：

`c_t = (1-alpha) * c_(t-1) + alpha * observed_conflict`

示范采样比例为：

`f_t = f_min + (f_max-f_min) * (1-c_t)^sensitivity`

正式默认值为 `f_max=0.25`、`f_min=0`、`alpha=0.2`、`sensitivity=1`。没有对应示范位置的在线状态不更新冲突率，避免将“没有证据”错误当成“安全”。

## 两个版本

- `conflict_predict_next: true`：同时检查动态障碍当前格和下一步预测格，与环境实际碰撞语义一致，是主方法；
- `conflict_predict_next: false`：只检查当前占用，是信息较弱的消融。

训练日志新增 `demo_fraction_mean`、`demo_action_conflict_rate`、`demo_conflict_observation_count` 和 `demo_conflict_ema`。策略名为 `conflict_adaptive_demo`，底层仍使用不可淘汰的示范分区和在线环形分区，只动态改变每批次的示范数量。

## 实验边界

该方法目前限定单地图训练，正好匹配现有三张 v4 地图各自训练的协议。它不使用 validation/test 信号控制训练，也不改变奖励、网络、epsilon、replay 总容量和 checkpoint 选择规则。

主方法配置为 `configs/dynamic_spatial_generalization_{office,parcel,warehouse}_conflict_adaptive_v1.yaml`；当前占用消融配置为 `configs/dynamic_spatial_generalization_{office,parcel,warehouse}_conflict_current_v1.yaml`。正式实验只需要新增这两种方法各 3 地图 × 5 seed，共 30 次；Uniform、Prefill、Persistent 25% 直接复用已经完成的 v4 结果。

本轮只运行了 Office seed 0 的两个 4-episode smoke，未启动正式训练。全量单元测试为 100 passed。

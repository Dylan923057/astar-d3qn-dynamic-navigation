# 碰撞终止协议 v1

本协议只修改碰撞语义，不同时修改地图、动态障碍物、网络、示范数据或示范使用方式。

- 碰撞即时奖励保持 `-1.0`，不使用过大的负奖励。
- 首次静态或动态碰撞立即结束当前回合，碰撞回合无法继续获得目标奖励。
- 到达目标奖励保持 `10.0`，步进、进度和等待奖励保持原值。
- 验证和测试使用相同的碰撞终止语义；碰撞即本次安全导航失败。
- 旧 `balanced_v4` 配置及输出不修改，新实验写入独立输出目录。
- 三种方法均训练恰好 `400000` 个环境交互步，不再用完成回合数决定停止时间。
- epsilon 在前 `340000` 个环境步从 `1.0` 线性衰减到 `0.05`，后 `60000` 步保持 `0.05`。
- 每跨过 `25000` 个训练环境步执行一次验证；验证本身不计入训练预算。

当前尚未启动训练。训练阶段是否需要随机起点仍未确定；应先记录碰撞终止后的位置覆盖情况，再决定是否加入，避免无证据地增加新变量。

对应配置：

- `configs/dynamic_spatial_generalization_office_terminal_collision_v1.yaml`
- `configs/dynamic_spatial_generalization_parcel_terminal_collision_v1.yaml`
- `configs/dynamic_spatial_generalization_warehouse_terminal_collision_v1.yaml`

# Office A* 示范衰减实验 v2

## 为什么现在做这一步

对已完成的 6 种方法、5 个随机种子和每次训练保存的 16 个检查点进行了离线冲突探针分析，共检查 480 个模型。探针包含 905 个训练分布内的 A* 首次冲突状态和 20 个必须等待的 oracle 状态；模型只推理，不继续学习。

结果不支持“Persistent 在训练后期必然退化”这一强假设。Persistent 的冲突安全动作率从 225k 的 85.1% 上升到 400k 的 97.3%，5 个种子的变化均为正。A* 持续示范确实加快了中前期学习：Persistent 的 validation-safe AUC 为 0.553，高于 Uniform 的 0.440 和 Prefill 的 0.443。

但旧自适应机制几乎没有产生调节。最终实际示范比例为：Persistent 25.00%、CA-current 24.71%、CA-predictive 24.55%。因此旧 CA-predictive 与 Persistent 在机制上过于接近，不能用于验证“自适应衰减是否更好”。

## 新增的两个方法

### Time-decay

这是纯时间消融，不使用冲突信号：

- 0–200k environment steps：示范比例保持 25%；
- 200k–340k：从 25% 线性降到 0；
- 340k–400k：保持 0。

它回答“只要后期移除 A* 是否就足够”。衰减使用 environment-step 时钟，所有种子的切换时点一致。

### CA-decay-v2

这是事件触发的自适应方法，不预设按时间下降：

- 初始示范比例 25%，最低 5%；
- 在智能体到达静态 A* 覆盖的位置时，检查 A* 动作当前或下一时刻是否会与动态障碍冲突；
- 任意冲突作为一个事件，冲突压力以 0.5 的系数快速上升；
- 无冲突时以 0.005 的系数缓慢恢复；
- 示范比例为 `0.05 + 0.20 * (1 - conflict_pressure)`。

用已有 Local-conflict 日志中的实际事件频率估算，四个 curriculum 阶段的稳态示范比例约为 25.0%、20.4%、14.1% 和 10.5%。因此新版会随冲突密度明显变化，同时在无冲突静态阶段保留完整的 A* 加速作用。

## 比较与判定

原 6 种方法不重训，只训练 Time-decay 和 CA-decay-v2，各 5 个种子。训练结束后同一条命令会自动生成正式测试汇总、近堵路因果诊断、全部检查点冲突曲线以及 8 种策略的地图和 GIF。

主要判定顺序：

1. validation collision-free success AUC：是否保留前期学习加速；
2. 905 个冲突状态上的 safe-action AUC：是否更早摆脱被堵塞的 A* 动作；
3. 正式同分布测试的 safe success 和 required behavior：不能以牺牲主任务为代价；
4. 近堵路诊断的 safe success、目标碰撞率和 exact-state safe action；
5. 实际示范比例曲线：确认算法确实发生了调节。

如果 CA-decay-v2 优于 Persistent 且优于 Time-decay，才支持“冲突自适应”本身有效；如果 Time-decay 与其相当或更好，只能说明后期减少示范有效；如果 Persistent 仍最好，则应否定当前自适应衰减方案，而不是继续调参寻找正结果。

## 运行

```powershell
python scripts\run_office_controlled_behavior_study.py --device cuda
```

命令可恢复执行：已有 30 个正式 run 会自动跳过，只运行新增的 10 个 run，之后执行 v2 汇总与可视化。

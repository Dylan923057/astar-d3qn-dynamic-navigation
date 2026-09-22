# 多动态障碍风险覆盖示范交接实验 v1

## 研究问题

现有单障碍同源分叉实验用于隔离 A* 示范回放比例的总作用，但不代表密集动态导航。其训练中动态障碍虽然经常可见，真正阻塞参考 A* 动作的风险步却低于千分之二。v1 因此同时改变两个相互配套的部分：建立多障碍风险基准，并加入风险覆盖驱动的三分区回放。旧 `replay_adaptation_v2` 数据、代码入口和结果全部保留。

## 多障碍场景

- 静态地图、起终点和单障碍物理路线来自冻结的 `replay_adaptation_v2`。
- 训练和验证各自包含 3/5 障碍场景；测试分别包含 1/3/5/7 障碍密度。
- 3 障碍场景有 1 个因果冲突障碍，5/7 障碍场景有 2 个；其余障碍使用对参考 A* 安全的控制相位。
- control/conflict 配对保持完整路线不变，只把因果障碍切换为安全或冲突相位。
- 同一场景内动态路线格互不重叠；所有 control 使用原 A* 路径实际回放，所有 conflict 使用位置—时间 A* 求得的安全路径实际回放。
- 当前不规则地图存在等长替代通路，所以 `conflict_oracle_steps - control_oracle_steps` 可以为 0。该数据检验的是参考动作受阻后的安全换路，不宣称每场都严格要求等待或增加最短路代价。

每张地图的 paired scene 数量为 train/validation/test=`36/12/48`。测试每种密度有 12 对。逐场审计位于 `data/risk_handover_v1/scenario_audit.csv`，位置预览位于 `maps/previews/risk_handover_v1/`。

## 风险覆盖示范交接

回放由三个分区组成：

1. 不可淘汰的静态 A* 示范；
2. 普通在线 FIFO 经验；
3. 容量 3000 的动态风险经验。

风险事件包括参考 A* 动作有当前/下一相位碰撞风险、实际选择动作有风险或发生动态碰撞。风险事件及其前三个转移写入风险分区，同时继续保留在普通在线池。覆盖键为“障碍密度 + 决策位置”；重复访问同一键不会虚增覆盖。

batch 固定为 64，其中普通在线配额始终为 48。剩余 16 个指导槽初始全部采样静态示范；随着独立风险键覆盖数达到 18，线性替换为风险样本，最终为 0 示范 + 16 风险 + 48 普通在线。方法不读取 validation/test 成绩，不增加 batch，不改变学习率、奖励或网络。

## 比较顺序

正式比较：固定 0%、固定 10%、固定 25%、时间衰减和 `risk_handover`。建议先跑 Map 1 的 seed 0–2；只有风险池实际触发、风险采样比例达到非零且安全 AUC 方向一致时，再扩到三地图五 seed。PER 和 DQfD-style 可在主方法通过 pilot 后加入强基线。

主要指标仍为等交互预算 validation conflict safe-success AUC，同时按 1/3/5/7 障碍密度报告最终安全成功、动态碰撞和超时。训练日志额外记录 `risk_buffer_size`、`risk_coverage_count`、`handover_progress` 和累计风险采样数。

## 命令

只重新生成和查看数据，不训练：

```powershell
python scripts/prepare_risk_handover.py
```

三种固定比例 Map 1 pilot：

```powershell
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2 --schedule fixed --device cuda
```

时间衰减和风险交接 pilot：

```powershell
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2 --schedule decay --device cuda
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2 --schedule risk_handover --device cuda
```

不同 schedule 会复用同一 seed 的合格 foundation；同一分支已经完成时自动跳过。不要并行启动会同时创建同一个 foundation 的命令。

五种方法全部完成后汇总 Map 1：

```powershell
python scripts/summarize_risk_handover.py --map-index 0 --seeds 0 1 2
```

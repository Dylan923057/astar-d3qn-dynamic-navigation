# 动态风险动作排序消融 v1

## 研究问题

在保持观测、奖励、动作执行、回放容量和示范衰减计划不变的条件下，检验训练阶段的即时风险动作排序约束，能否提高安全成功 AUC，并降低不同训练 seed 之间的动态避障波动。

## 三组设计

- A：`schedule_decay`，已有时间衰减结果，代码等价性核对后直接复用。
- B：`demo_action_margin`，仅在在线状态位于 A* 参考路径且 A* 动作立即危险时添加排序标签。
- C：`all_action_margin`，在所有在线状态标记全部静态合法、但会立即动态碰撞的动作。

B、C 均使用相同的 `margin=0.8`、`loss_weight=1.0`。排序损失只要求最佳安全动作的 Q 值比最高风险动作至少高出给定间隔；它不指定必须向哪个方向走，也不替换实际执行动作。

## 数据边界

- 风险标签由模拟器在动作执行前生成，只写入在线经验。
- 标签不作为网络输入，验证和测试时不进行动态动作屏蔽。
- 保存的 5 个验证失败仅用于只读预检，不写入训练池。
- 第一轮只使用地图索引 0 和 seed 0、1、2，共新增 B/C 六次正式训练。

## 正式训练

PowerShell 中依次执行：

```powershell
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2 --schedule demo_action_margin --stage adapt --device auto --threads 1 --reuse-foundation
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2 --schedule all_action_margin --stage adapt --device auto --threads 1 --reuse-foundation
```

## 验证集决策

两组训练全部结束后，先运行不读取测试结果的汇总：

```powershell
python scripts/summarize_action_ranking_ablation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2
```

汇总脚本默认只读取 `validation_curve.csv`，不会打开包含测试指标的 `result.json`。它还会核对复用的 A 组平均 AUC 是否为预登记的 0.7042，防止引用错实验。

继续投入的工程门槛为：C 相对 A 的平均验证安全成功 AUC 至少提高 0.03，至少 2/3 个 seed 改善，C 相对 B 有正收益，且冲突场景的最终碰撞率、超时率和静态保持没有明显恶化。这不是统计显著性检验。

只有冻结验证决策后，才能使用 `--include-test` 生成固定最终模型的测试汇总。若未通过，停止本实现，不继续搜索更多间隔或损失权重。

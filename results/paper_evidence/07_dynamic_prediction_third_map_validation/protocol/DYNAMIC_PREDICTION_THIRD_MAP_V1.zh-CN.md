# 动态占用预测辅助第三地图独立验证 v1

## 当前目标

在第三张冻结地图 `irregular_workcell_91703` 上独立比较 A `time_decay` 与 B `global_prediction`，验证前两张地图的正向结果能否继续复现。当前阶段不再修改算法，也不运行 C、测试集或消融实验。

## 冻结设置

- seed：0、1、2。
- 每个 A/B 分支固定运行 20 万动态交互步。
- 每个 seed 的 A/B 必须从同一个合格 foundation 快照分叉。
- B 固定 `prediction_loss_weight=0.1`、`prediction_pos_weight=20`、`prediction_mode=global_prediction`。
- replay、reward、epsilon、batch size、数据划分和其余训练设置保持不变。
- A/B 均设置 `--defer-test`；冻结验证结论前不得生成或读取测试结果。

## 验证指标

- 主指标：`validation_conflict_auc`，报告 A/B 三 seed 均值及 B-A。
- 学习效率：`threshold_confirmation_step`，安全成功率达到 90% 且连续满足两次验证的首次步数；未达到时标记右删失。
- 最终性能：安全成功率、动态碰撞率和超时率。
- 后期稳定性：最后 5 万步安全成功率的均值和标准差。

## 预先冻结的继续门槛

第三地图同时满足以下条件才认为跨地图验证通过：

1. B 的三 seed 平均验证 AUC 高于 A。
2. B 至少在 2/3 个 seed 上的验证 AUC 高于对应 A。
3. 每个 seed 的最终动态碰撞率和超时率相对 A 的增幅均不超过 0.05。

最后 5 万步稳定性作为辅助证据完整报告，但不作为本轮硬门槛。通过后冻结方法并准备三张地图的最终测试；未通过则停止扩跑并诊断，不根据结果修改门槛。

## 严格执行顺序

1. 先训练 map03 的三个 foundation 和三次 A。
2. A 全部完成并通过状态、步数、测试隔离和 foundation 资格审计后，才给出 B 的运行指令。
3. B 全部完成并通过审计后，才给出验证汇总指令。
4. 第三地图验证通过后，才准备最终测试集评估。
5. 最终测试完成后，再执行 `B_no_prediction` 消融。
6. 完成结果归档到 `results/paper_evidence/07_dynamic_prediction_third_map_validation/`，不上传权重、大型轨迹或 smoke 结果。

## 当前阶段命令

当前只执行 foundation 与 A，不连续启动 B：

```powershell
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 2 --seeds 0 1 2 --schedule decay --stage all --device auto --threads 1 --defer-test --output-root outputs/dynamic_prediction_third_map_v1
```

后续命令必须等待当前阶段完成并检查结果后再确定。

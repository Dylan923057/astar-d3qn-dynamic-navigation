# 动态占用预测辅助跨地图验证 v1

## 研究问题

验证第一张地图上 `global_prediction` 的提升是否能够迁移到第二张地图 `irregular_workcell_91702`，而不是继续修改方法或搜索参数。

## 冻结设计

- A：`time_decay`。
- B：`global_prediction`。
- seed：0、1、2。
- 每组每个 seed 固定运行 20 万动态交互步，共 6 次适应训练。
- A/B 从相同 seed 的合格基础快照分叉。
- B 固定 `prediction_loss_weight=0.1`、`prediction_pos_weight=20`、全局预测权重 1。
- time_decay、epsilon、奖励、回放配额、数据划分和其余超参数保持不变。
- 不运行 `decision_weighted_prediction`，不再调整预测参数。
- A/B 均只生成训练与验证结果；冻结验证决策前不得生成或读取测试结果。

## 预先确定的判定门槛

同时满足以下条件才认为 B 具有初步跨地图有效性：

1. B 的三 seed 平均验证安全成功 AUC 高于 A。
2. B 至少在 2/3 个 seed 上的验证 AUC 高于对应 A。
3. 每个 seed 最终验证碰撞率和超时率均不高于对应 A。
4. B 的三 seed 后 5 万步安全成功率标准差均值不高于 A。

通过后才扩展至 5 个 seed，随后冻结方法并进入最终测试；未通过则停止扩跑并做失败诊断，不根据结果调整上述门槛。

## PowerShell 运行顺序

先运行 A。若 map02 的基础模型尚不存在，该命令会先为三个 seed 训练合格基础模型，再运行三次 A：

```powershell
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 1 --seeds 0 1 2 --schedule decay --stage all --device auto --threads 1 --defer-test --output-root outputs/dynamic_prediction_generalization_v1
```

A 成功后运行三次 B，复用完全相同的基础快照：

```powershell
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 1 --seeds 0 1 2 --schedule global_prediction --stage adapt --device auto --threads 1 --defer-test --prediction-loss-weight 0.1 --prediction-pos-weight 20 --output-root outputs/dynamic_prediction_generalization_v1 --foundation-source-root outputs/risk_handover_v1
```

六次适应训练完成后，只读取验证集并冻结决策：

```powershell
python scripts/summarize_dynamic_prediction_generalization.py --config configs/risk_handover_v1.yaml --map-index 1 --seeds 0 1 2
```

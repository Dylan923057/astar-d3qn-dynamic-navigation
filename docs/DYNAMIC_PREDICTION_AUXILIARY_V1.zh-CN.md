# 动态障碍预测辅助 D3QN v1

## 单一研究变量

保留现有 A* 基础模型、time_decay 示范计划、20 万步预算、epsilon、奖励、回放、地图、场景、障碍运动和 D3QN TD loss。预测头只在训练时提供辅助监督，验证和测试动作仍只由 Q 值决定。

## 网络与标签

- 共享原 `DuelingQNetwork` 的三层 CNN encoder。
- 预测头输入为共享 CNN 特征与已执行动作的 one-hot 拼接。
- 预测头为 `Linear -> ReLU -> Linear`，隐藏维度 128，输出完整 15×15 局部网格。
- 标签严格取 `transition.next_state.spatial[current_dynamic_channel]`。
- 环境显式声明 `current_dynamic_channel=1`；目标采用 next_state 自己的机器人中心坐标。
- 预测头不读取 simulator future oracle，不参与 action mask、奖励或动作选择。

## 两个新增组

- B `global_prediction`：完整 15×15 网格权重均为 1。
- C `decision_weighted_prediction`：完整网格仍计算损失，仅中心 5×5 权重为 3，其余为 1。

B/C 使用同一网络容量、`lambda_pred=0.1`、`pos_weight=20`、基础模型、回放配额和训练协议。二者唯一差异是 prediction loss 的空间权重。

## 损失

`L_total = L_D3QN + lambda_pred * mean(weight_map * BCEWithLogits(prediction, next_dynamic_target, pos_weight=20))`

静态 A* 示范样本不参与预测 loss；预测监督只作用于在线动态训练 transition。`lambda_pred=0` 时不构建预测 loss 计算图，D3QN 参数更新与原 TD-only 路径逐元素一致。

## 正式训练前证据

- 训练场景 4752 个真实 transition 坐标检查全部通过。
- 动态正像素比例为 0.3653%，原始负正比约 272.7；首轮固定截断为 `pos_weight=20`。
- B/C 各完成 3000 步 smoke，无 NaN，预测损失下降，辅助项未淹没 TD loss。
- smoke 不生成或读取测试结果。
- 详细报告：`outputs/dynamic_prediction_preflight_v1/report.md`。

## 正式实验命令

正式实验尚未自动启动。通过人工确认后，在 PowerShell 中依次运行：

```powershell
python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2 --schedule global_prediction --stage adapt --device auto --threads 1 --prediction-loss-weight 0.1 --prediction-pos-weight 20 --output-root outputs/dynamic_prediction_auxiliary_v1 --foundation-source-root outputs/risk_handover_v1 --reuse-foundation
if ($LASTEXITCODE -eq 0) { python scripts/run_replay_adaptation.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2 --schedule decision_weighted_prediction --stage adapt --device auto --threads 1 --prediction-loss-weight 0.1 --prediction-pos-weight 20 --output-root outputs/dynamic_prediction_auxiliary_v1 --foundation-source-root outputs/risk_handover_v1 --reuse-foundation }
```

六次运行只生成训练和验证结果，状态为 `validation_complete`，不会生成 `test_evaluation.csv`。完成后运行：

```powershell
python scripts/summarize_dynamic_prediction_auxiliary.py --config configs/risk_handover_v1.yaml --map-index 0 --seeds 0 1 2
```

该脚本只读取验证曲线并写入冻结决策。测试阶段必须等待验证决策完成。

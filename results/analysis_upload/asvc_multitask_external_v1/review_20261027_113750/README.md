# 四方法多任务外部基线与静态/动态配对诊断

本轮已于2026-10-27 04:47完成。新增原始D3QN、DQfD各seed0/1，复用此前多任务C/D四次运行，共8条曲线、每次500000在线环境步。D3QN/DQfD均在相同240任务训练池重新训练，没有拿旧固定任务模型直接验证。

## 先看结果

| 方法 | 成功率seed0/1 | 平均成功 | 平均碰撞 | 平均超时 | 全程AULC均值±样本SD |
|---|---:|---:|---:|---:|---:|
| C 持续弱监督 | 76% / 72% | 74% | 13% | 13% | 47.14% ± 1.78% |
| D 风险自适应 | 74% / 64% | 69% | 3% | 28% | 44.17% ± 0.07% |
| 原始D3QN | 2% / 26% | 14% | 20% | 66% | 5.88% ± 4.95% |
| DQfD | 12% / 20% | 16% | 16% | 68% | 14.94% ± 1.02% |

当前相同50万步预算下，C有明显数值优势，D尚无整体收益；仅2seed，不宣称正式显著性或未见目标泛化。去掉动态障碍后的平均成功率为C 74%、D 63%、D3QN 0%、DQfD 18%，基础多目标导航仍有缺口。不能将超时增加直接归因为监督减少。

## 阅读顺序

1. [最终结论、成本和审稿缺口](results/asvc_multitask_external_v1/FINAL_SUMMARY.zh-CN.md)。
2. [逐seed全程/分阶段AULC](results/asvc_multitask_external_v1/run_comparison.csv) 与 [均值/样本SD](results/asvc_multitask_external_v1/method_comparison.csv)。
3. [完整学习曲线](results/asvc_multitask_external_v1/learning_trend_review_20261027_104518/learning_trends_full.png)、[最后20万步](results/asvc_multitask_external_v1/learning_trend_review_20261027_104518/learning_trends_last_200k.png)、同目录窗口统计CSV/PDF。细线是原始成绩，粗线为仅供显示的5点后向平均，不作正式收敛检验。
4. [配对诊断和代表性失败](results/asvc_multitask_external_v1/DIAGNOSIS.zh-CN.md)、[轨迹图](results/asvc_multitask_external_v1/representative_trajectories.png)、[全部配对任务成绩](results/asvc_multitask_external_v1/paired_all_tasks.csv)。
5. [Q/TD及实际监督审计](results/asvc_multitask_external_v1/learning_audit.csv)、[风险区间系数](results/asvc_multitask_external_v1/risk_weight_audit.csv)。
6. [冻结协议、启动/完整续训与未启动方案](docs/MULTITASK_EXTERNAL_PROTOCOL.zh-CN.md)。

## 数据范围与复用

- `runs/<方法>/seed_0或1/`：8次运行的完整51点曲线、各2550条验证明细、实际多任务训练回合、更新/监督/PER成本日志、登记与完成记录；C/D历史逐步离线案例也保留。
- 总计408个检查点、20400条训练期验证记录，反复使用相同50场景，不能当成20400个独立测试任务。其中C/D的10200条复用，新增基线10200条。
- 静态/动态配对表800条，其中400条动态成绩复用原最终验证，400条静态是新增无训练评估，不把800条都当新增评估。包内616份可用完整配对轨迹，旧C/D动态仅各4/50；缺少的循环/终止距离保持NA，未擅自补跑。
- `registration/shared_pilot/`：共同240训练/50验证/两个60测试任务池、50个冻结场景、旧冻结、原始代码快照与旧结果保护登记。保存测试池定义不代表做过测试。
- `registration/external_baselines/`：新冻结、示范计划、480次成功/失败尝试、3472条示范NPZ、覆盖报告、原始快照与单独登记的报表修正。示范成功77次、覆盖70/240任务、31/40目标；没有使用旧固定起终点示范或验证/测试任务，也没有临时增加预算。
- 每个方法499501次在线更新；DQfD额外20000离线更新/seed，并永久保留示范。示范生成15107环境步由两个seed共享，成本另列，不能宣称总成本相同。
- `code/`：原始训练代码、当前通过登记修正后的分析代码、配置、测试和只读工具。训练源字节与原冻结核对；两个报表/校验修正保留原快照，不改训练身份。
- `model_index.csv`：8个最终模型及完整断点的本地路径/SHA。权重和回放本体留在本地；没有权重就不能重新执行网络评估，只有权重也不等于完整续训。已有结果的只读复核无需它们。
- 旧固定任务unguided_bound证据单列于results下legacy_*表，不能混入新多任务成绩。

## 校验命令

下载整个目录后，只需Python标准库：

```powershell
python tools/verify_bundle.py --verify --package .
```

仓库根目录：

```powershell
python scripts/package_multitask_comparison.py --verify
# 仅原工作空间具有模型、完整断点和原始日志时：
python scripts/package_multitask_comparison.py --verify --check-sources
```

校验包括全部文件SHA、原冻结/报表修正、多任务训练任务和场景序列、20400条验证、全程与阶段AULC、seed间样本SD、800条配对成绩与缺失轨迹、示范预算和来源。没有PyTorch也能校验已有结果；不训练、不跑网络、不改Git索引。

## 结论边界与后续

C仍在改善但后期放缓，部分seed已横盘；D3QN有seed长期低水平，DQfD末段不持续上升。50万步预算不足可能是因素，延长收益尚未验证。追加预算、CE监督、监督下限0/0.05/0.1/0.2、五seed和正式消融均未启动。另开预算扩展时必须保留原50万步结果，同预算比较并完整恢复断点，不能挑失败seed或重设日程。

GitHub上传范围由upload_scope.json限定。阅读包不包含账户配置、凭据、文档转换工具、模型、回放、缓存或其他未提交实验；原始日志中本地路径作为来源保留。SHA保证内容一致，不代表结论已独立复现。

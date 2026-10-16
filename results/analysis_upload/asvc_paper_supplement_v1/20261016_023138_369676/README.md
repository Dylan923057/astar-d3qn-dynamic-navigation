# ASVC-D3QN 补实验阅读包

[实际结果报告](paper/REPORT.md) · [主表](paper/main_results.md) · [模块消融表](paper/module_ablation.md)

ASVC-D3QN仅为`advice_bound_margin_all_eligible`的论文显示名，旧ID、配置与结果路径不变。
新增五次20万在线步的A*初始化机制对照，不是A*-IDDQN完整复现。
本包仅在五次训练、原500场景的五模型评估及四档密度的80模型评估全部完成后生成。

paper内有PDF、SVG、300dpi PNG、实际验证时间的原始CSV、五seed均值/样本标准差及同seed配对差。
曲线阴影按实际时间线性对齐，未平滑。统计单位为训练seed。
registration保存冻结配置、场景ID、生成规则与SHA；run_logs保存所有方法的真实训练记录。
source_snapshot保存绘图与复现源码。权重和全部原始轨迹保留本地；paper/model_manifest.csv记录权重路径和SHA。
固定seed0及预注册的三个相同场景用于轨迹图，成功和失败均按实际结果展示。

DQfD的额外20000次离线更新、永久示范和PER成本单列；共享示范生成成本见paper/shared_demonstration_generation_cost.json。
历史没有记录的时间/设备/A*调用字段保持空白。密度试验是同地图适应评估，不用于调参，不是跨地图泛化。
首版密度场景因初始位置重叠无效而保留为历史；正式密度统计使用全部通过构造验证的v2。

复现命令（仓库根目录，PowerShell；已有完整结果不会自动重训）：

```powershell
python scripts/run_asvc_supplement.py
python scripts/prepare_asvc_density.py
python scripts/evaluate_asvc_supplement.py
python scripts/plot_asvc_supplement.py
python scripts/package_asvc_supplement.py
```

仅空输出目录的新训练：`python scripts/run_asvc_supplement.py --train`
恢复未完成训练：`python scripts/run_asvc_supplement.py --train --resume`
重新评估要求本地已有最终权重；缺权重不会自动训练。评估按已完成模型/密度恢复。

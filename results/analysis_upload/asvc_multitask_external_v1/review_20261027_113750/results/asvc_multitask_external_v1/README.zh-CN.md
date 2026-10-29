# 本轮交付入口

本轮四次外部基线及静态配对评估已于2026-10-27 04:47完成，无失败，队列为空。先看 [最终结论与剩余缺口](FINAL_SUMMARY.zh-CN.md)、[完整比较表](REPORT.zh-CN.md)、[配对诊断和失败案例](DIAGNOSIS.zh-CN.md)、[冻结协议与后续方案](../../docs/MULTITASK_EXTERNAL_PROTOCOL.zh-CN.md)。REPORT由冻结模板生成，审稿表中“未完成运行”等通用占位以FINAL_SUMMARY的完成审计为准。

实际推荐命令（仓库根目录）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Status
# 仅在旧协调器和训练均已退出后续跑：
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Resume -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Report
```

首次启动为 `-Action Prepare` 后 `-Action Start -Workers 2`；本轮已经完成，勿重复。`start_external_baselines.ps1` 为非科研逻辑的薄包装：初始化 PowerShell 的 LASTEXITCODE，修复冻结旧启动器在新shell中成功启动后误报退出码的显示问题。冻结的启动器、源快照、实验协议和运行身份保留不变。冻结报告模板中的旧启动器命令请使用此包装执行。

主要文件：

- `run_comparison.csv`：每方法/seed全程、分阶段AULC及末段表现。
- `method_comparison.csv`：跨seed均值和样本标准差，缺失结果为NA。
- `paired_seed_summary.csv` / `paired_method_summary.csv`：静态/动态、长短分层、统一参考冲突组；动态循环统计必须同时查看轨迹覆盖数。
- `paired_task_changes.csv` / `paired_seed_changes.csv` / `paired_method_changes.csv`：逐任务和跨seed配对变化。
- `paired/`：逐任务成绩、完整可用轨迹、代表性成功/失败、执行后离线教师/风险/真实冲突。
- `learning_audit.csv` / `risk_weight_audit.csv`：Q/原始TD/裁剪比例，以及实际回放监督覆盖、降权和系数总量。
- `legacy_unguided_bound_seeds.csv` / `legacy_raw_vs_bound.csv` / `legacy_unguided_bound_provenance.json`：旧固定任务证据与来源SHA，不混入新多任务比较。
- `learning_curves.png` / `.svg`：逐seed原始学习曲线，以及跨seed均值和样本SD。
- `comparison_status.json` 与 `outputs/asvc_multitask_external_v1/pipeline_status.json`：结果完整性和实际运行队列。

旧C/D每模型只留4个完整动态任务轨迹。动态成功/碰撞/超时和等待对全部50任务可用；额外循环和终止距离仅有4/50，其他留空而非0。未获得用户明确授权前，不补跑缺失动态策略轨迹。新基线最终例行动态验证直接保存50/50轨迹，不额外评估动态场景。

报表读取新增基线统计时的重复字段错误已修复，并单独登记于 `data/asvc_multitask_external_v1/nontraining_corrections/001_report_counter_collision.json`。原冻结文件和源快照均保留，所有训练、环境、示范生成及配置源码未改变；验证器只允许这两项报表/验证修正，仍拒绝未登记变动。当前训练、checkpoint和原C/D结果不受影响。

本轮不启动CE、敏感性、五seed、正式消融或任何测试池。D不得预先升级为主方法。

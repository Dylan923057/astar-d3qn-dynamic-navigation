# 本轮入口：TD单独控制与循环证据

2026-10-29 18:21两次新增控制与全部诊断已完成，无失败。先看 [最终发现](FINAL_FINDINGS.zh-CN.md)：TD-only平均成功14%→31%，C为74%；TD单独有效，教学组合仍有额外收益，不能归为持续监督单项。

从 [五方法报告](REPORT.zh-CN.md) 和 [完成状态](comparison_status.json) 查看当前进度。训练未完成的控制组指标明确为NA；协调器仅运行seed0、1，各500000步，之后自动补静态配对与五方法报告，不追加其他训练。

- [逐seed结果、各阶段AULC与成本](five_method_per_seed.csv)、[均值和样本SD](five_method_mean_sample_sd.csv)。
- [TD单独改善与教学组合差值](controlled_effects_per_seed.csv)、[差值均值/样本SD](controlled_effects_mean_sample_sd.csv)。C−TD-only只能解释为A*代选＋持续监督组合，不能归于持续监督单项。
- [五方法学习曲线](five_method_learning_curves.png)、[全部原始曲线](all_validation_curves.csv)。
- [循环/长度/参考冲突组逐seed统计](loop_seed_summary.csv)、[跨seed统计](loop_method_mean_sample_sd.csv)。
- [逐任务静态/动态结果](all_paired_tasks.csv)、[成对变化](paired_task_changes.csv)。静态评估会改变输入分布，不等于单独静态训练。
- [典型循环索引](typical_cases.csv) 和 `typical_cases/`：执行轨迹、五动作Q与教师Q。所有超时的详细五动作Q、合法掩码、教师排名、硬筛查、局部风险、真实未来冲突在版本化诊断目录的timeout_and_reference_steps.csv。
- [完全相同观测的重复状态](repeated_observation_states.csv)、[观测闭环](recurrent_observation_cycles.csv)：状态SHA、访问次数、五动作Q与回退差。它们解释网络执行时为什么重复选动作，不能单独确定形成该Q排序的训练原因。
- [最终保留回放的目标/教师/风险审计](replay_goal_teacher_audit.csv)：C/D共80000条经验，不代表所有历史抽样都已核验。
- [Q/原始与裁剪后TD/梯度/成本](learning_and_cost_audit.csv)。新控制原始TD及裁剪目标样本在其outputs下td_target_samples.csv。

版本化轨迹/逐步证据位置见 [loop_latest.json](../../data/asvc_multitask_td_control_v1/loop_latest.json)。实际评估没有动作监督，监督系数恒为0；所有C/D权重是执行后计算的假设值，不能写成评估时真的使用了监督。

协议和可用命令：[MULTITASK_TD_CONTROL](../../docs/MULTITASK_TD_CONTROL.zh-CN.md)。CE/大间隔只读作业计划：[TEACHER_LOSS_COMPARISON_PLAN](../../docs/TEACHER_LOSS_COMPARISON_PLAN.zh-CN.md)，实际CE训练入口仍未实现，本轮未执行。

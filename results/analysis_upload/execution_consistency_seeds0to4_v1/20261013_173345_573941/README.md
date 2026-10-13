# 执行一致性筛选：五seed阅读包

先读[五seed报告](summary/REPORT.md)、[逐seed结果](summary/per_seed_metrics.csv)、[配对差](summary/paired_differences.csv)、[配对均值与样本标准差](summary/paired_mean_sample_sd.csv)及[曲线](summary/paired_validation_curves.png)。统计单位seed，样本标准差ddof=1。

原完整组合advice_bound_margin五seed复用；新组只取消辅助监督的实际执行动作=A*建议条件。TD使用真实动作、真实奖励/下一状态/终止标记，margin使用独立教师动作。不改其他标签规则、风险筛选、权重、采样、奖励或日程；代选与监督在10万步同时退出。不是分阶段撤教方案。

五次新训练均200000环境步、199501在线更新，没有离线示范或预训练。全部10个最终模型复评原50场景，210个验证点、10500条场景记录全部核对。不选择最佳检查点。新对照尚未评估独立500场景，本包只支持原50验证协议下的结论。

`run_logs/`保留全部曲线、逐场景结果、已有机制日志和新组训练CSV；`source_snapshot/`保留对应源码/配置；`registration/`保留原三seed和新增两seed冻结登记。监督比例按回放采样次数计算，不能解释为不同转移的覆盖率。原组监督一致率由标签规则推得100%，有符号Q范围未记录，缺失不能填零。

`trajectories/`保留最终和撤教低谷的首个各类失败示例，完整字段仅压缩JSON空白；选择规则不筛选模型。全部失败计数见summary/checkpoint_failures.csv，原始完整轨迹路径及SHA见raw_trace_index.csv。未来障碍信息仅用于事后诊断。

权重和回放断点留在本地，准确最终权重路径/SHA见manifest.json。克隆后缺失权重不能自动重训。旧结果及三seed历史报告保留；五seed结论优先看本包，不沿用三seed结论。

完整性检查：`python scripts/package_execution_consistency.py --verify <本包路径>`。给GPT的分析请求见[GPT_ANALYSIS_PROMPT.md](GPT_ANALYSIS_PROMPT.md)。

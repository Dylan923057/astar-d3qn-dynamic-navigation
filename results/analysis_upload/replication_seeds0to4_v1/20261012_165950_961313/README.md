# 分阶段撤教与DQfD无预训练：五seed阅读包

先读[五seed报告](summary/REPORT.md)、[均值和样本标准差](summary/mean_sample_sd.csv)、[同seed配对差](summary/paired_differences.csv)及[曲线](summary/paired_validation_curves.png)。统计单位为seed，不是50验证回合。

两种新增方法各seed0–4全部完成200000在线步；原完整组合与原DQfD五seed复用。共有20次运行、420验证点、21000条逐场景验证记录；20个最终模型重新复核原50场景。四方法同一冻结500场景最终确认已完成，原两方法按SHA及协议核对复用，新两方法补评5000回合。没有新训练或最佳模型选择。

分阶段日程：0–60000与原组合一致；60000–80000代选概率由0.32线性降至0；80000后代选为0；监督按原日程在100000完全退出。TD约束、风险筛选、标签规则及训练设置不变。预设代选日程总量减少8%，不是等教师量对照。

DQfD对照只取消20000次离线更新。仍使用同一7260条冻结示范、永久示范分区、PER、多步TD、margin、L2及原在线条件，无TD裁剪。原DQfD总219501次更新，对照199501次。原DQfD的0步验证已经预训练；另外报告1–8万AULC。

`run_logs/`包含所有五seed曲线、场景明细、已有诊断与成本日志，以及十次新增对照的训练CSV；原方法训练CSV此前已上传，不重复复制。`source_snapshot/`保存对应实现和配置；`registration/`保留冻结登记；`trajectories/`按固定规则保存失败示例，仅移除JSON空白、不删字段。未来障碍轨迹仅用于事后诊断。全部失败分类在summary/checkpoint_failures.csv，示例不代表独立样本或模型筛选。

新增权重、回放恢复文件、批量失败轨迹和临时文件留在本地。最终模型准确路径和SHA在manifest.json；仅克隆仓库不能重新执行模型评估，缺权重不得自动重训。示范仍在仓库data/whole_map_91701_dqfd_comparison_v1/demonstrations.npz，不重复复制。

独立500最终确认见[最终报告](final500/REPORT.md)、[逐seed结果](final500/summary.csv)、[均值与样本标准差](final500/method_means.csv)和[配对差](final500/paired_difference_means.csv)。同批500场景合并10000条明细，不与50验证成功率混比，也不据此调参。summary中的50场景报告保存了评估前的“待评估”历史状态，当前完成状态以final500/verification.json为准。三seed报告作为历史保留。

[给GPT的分析请求](GPT_ANALYSIS_PROMPT.md)。文件完整性：`python scripts/package_replication_five_seeds.py --verify <本包目录>`，不训练、不推送。

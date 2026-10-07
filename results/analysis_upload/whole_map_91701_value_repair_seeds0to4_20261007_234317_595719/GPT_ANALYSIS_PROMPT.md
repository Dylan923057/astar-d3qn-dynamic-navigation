请读取这个仓库的最新五seed完整结果，结合相关代码分析，不要把旧两seed包、20000步pilot或此前运行时路径输入实验当成最新结果。

仓库：https://github.com/Dylan923057/astar-d3qn-dynamic-navigation

最新入口：https://github.com/Dylan923057/astar-d3qn-dynamic-navigation/blob/main/results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/README.md

请先读同目录README.md、REPORT.md、summary.csv、method_means.csv、paired_differences.csv、phase_metrics.csv和均值曲线，再检查implementation/代码、all_value_learning.csv、all_failure_behaviors.csv、trajectory_index.csv和seed3_collision_action_inspection.json。模型权重留在本地，不能声称你已经加载或重跑权重。

六组分别是：无A*原始D3QN、无A*且限制TD目标、A*探索、A*探索且限制TD目标、A*探索且动作监督、三者组合。全部随机初始化、沿用相同目标曼哈顿距离奖励；没有静态foundation、25%示范回放或运行时A*路径输入。训练200000步，A*和辅助监督100000步退出。验证只有网络、epsilon=0，同50个固定场景，不用test。

请回答：

1. 按配对seed和全部检查点，这三个设计分别带来什么收益？区分早期效率、后期稳定性和最终安全，不只看最后一次或最好检查点。
2. 第3、5组为何失败或退化？用价值日志与真实失败轨迹支持解释，区分已观察到的事实、合理推测和还未排除的其他原因。
3. 第6组为何学得快，却退出附近下降并在seed3最终发生四次迎面碰撞？第4组在同状态的不同Q排序说明什么，又不能说明什么？
4. 控制变量是否充分？风险筛选、监督衰减、TD目标限制、普通epsilon探索之间，哪些作用尚未单独识别？核查辅助动作监督的条件、奖励、截断bootstrap及非法动作屏蔽。
5. 现有证据最适合支持哪种论文主张？哪些创新说法不成立？如与已有论文比较，请查阅原论文并引用来源，不跨不同地图与协议直接比较成功率。
6. 推荐最少且有明确用途的下一批实验，优先解决退出稳定性、同条件已有方法对照和泛化；说明每个实验能排除什么解释。不要为了更高结果随意改奖励后与旧组混比，也不要自动启动训练。

统计单位是5个训练seed；重复验证场景和检查点相关。当前组合组AULC最高，但第4组最终五seed均100%，第6组为[100,100,100,92,100]%。请保持这个结论边界，不预设组合方案全面胜出。回答先给简单明确的结论，再给证据与实验建议。

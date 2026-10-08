# A*教学效率：七方法 × seed0–4，GitHub分析入口

最新结果：无代选＋动作监督＋TD限制AULC为83.02%±4.07%，比无A*＋TD限制提高34.66个百分点；完整组合91.37%±1.05%，再提高8.35个百分点，主要来自前5万步。两项配对比较均五seed同向提高。500场景最终成功率依次为新组97.84%±1.75%、组合98.68%±1.08%、代选＋TD限制99.12%±1.17%。论文定位为学习效率，不主张最终避障最好。

固定地图91701、起点(3,3)、终点(36,36)，每回合3–5个动态障碍；各200000环境步。所有评估epsilon=0，由D3QN自主选动作，无A*代选或动态保护。网络输入没有A*路径；教学是训练时的动作代选和/或辅助动作监督，不能与此前运行时路径输入实验混淆。本轮使用配对随机初始化，没有加载旧静态foundation。

阅读顺序：

1. [最终复核报告](../../../results/whole_map_91701_teaching_efficiency_v1/review_20261008_113945_236854/REPORT.md)：逐seed、均值±样本标准差、全程/阶段AULC、最后五次均值、退出低谷、监督覆盖和碰撞Q诊断。
2. [七组完整学习曲线与原50场景明细](../../../results/whole_map_91701_teaching_efficiency_v1/analysis_20261008_101153_586224/REPORT.md)、[全部检查点CSV](../../../results/whole_map_91701_teaching_efficiency_v1/analysis_20261008_101153_586224/all_validation_checkpoints.csv)。
3. [独立500场景最终评估](../../../results/whole_map_91701_teaching_efficiency_v1/independent_eval_20261008_101801_911391/REPORT.md)、[17500条场景明细](../../../results/whole_map_91701_teaching_efficiency_v1/independent_eval_20261008_101801_911391/scene_details.csv)、[全部3496条失败行为](../../../results/whole_map_91701_teaching_efficiency_v1/independent_eval_20261008_101801_911391/failure_behaviors.csv)。
4. [新增五seed训练日志](runs/supervision_only_bound/seed_0/result.json)，其余seed目录并列；原六组30次训练日志复用[此前包](../whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/README.md)。
5. [精选完整失败轨迹索引](trajectory_index.csv)、[训练源码及支持文件快照](snapshot/scripts/run_teaching_efficiency.py)、[独立配置快照](snapshot/configs/whole_map_91701_teaching_efficiency_v1.yaml)。
6. [可复制给GPT的分析请求](GPT_ANALYSIS_PROMPT.md)。

包内38条精选完整轨迹有明确选择规则：新组原50场景全部5次最终失败；退出/后期低谷6个分类代表；组合seed3独立场景全部9次碰撞；新组seed0全部12次独立等待超时；新组seed3全部6次独立碰撞。全部3496条失败的行为统计仍在链接的CSV中，精选轨迹不是所有失败样本。数据和证据保留成功及失败，未筛选有利seed。

独立500场景在新增训练前冻结，与旧50场景的物理场景及路线组合不重复，仅评估200000步最终模型，不用于调参、教学调度或检查点选择。它们来自同地图同路线池，不代表跨地图泛化，也不保证与随机训练场景完全不重合。统计单位是5个配对训练seed，不能把500场景当500次独立训练。

已核对35次完成状态、源码和最终权重SHA、配对初始状态，735个原50场景检查点、17500条独立结果，以及全部3496条独立失败轨迹。本次打包仅读取并复制已有结果，没有训练或重新评估。future障碍信息仅用于已保存前缀的事后诊断。

`manifest.json`记录包文件SHA、外部报告文件SHA及35份未上传权重路径/SHA。快照中的登记训练源文件可与run result中的source_sha256核对；其他支持文件是整理时快照。包内及冻结数据保留原始字节，避免换行转换破坏指纹。

本地权重、原始outputs和约1.65GB完整独立失败轨迹不上传；包及链接的CSV足以阅读分析。只有克隆仓库不能再次运行依赖本地权重的审核或评估命令，不能因此自动重训。使用已生成的报告、日志、数据和源码快照即可。

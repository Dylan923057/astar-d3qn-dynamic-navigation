# 执行一致性筛选：同一冻结500场景最终确认

只评估200000步最终模型，ε=0，关闭教师、路线引导和动态保护，仅共同静态合法动作掩码。新增五模型共2500回合，原组五seed同场景/SHA/协议结果复用。统计单位seed，样本标准差ddof=1，不用于调参或选择检查点。

|方法|成功%|动态碰撞%|超时%|
|---|---:|---:|---:|
|advice_bound_margin|98.16±1.82|1.84±1.82|0.00±0.00|
|advice_bound_margin_all_eligible|99.36±0.43|0.48±0.41|0.16±0.36|

|方法|seed|成功%|碰撞次数|等待超时|绕圈超时|其他超时|
|---|---:|---:|---:|---:|---:|---:|
|advice_bound_margin|0|98.2|9|0|0|0|
|advice_bound_margin_all_eligible|0|99.0|1|3|1|0|
|advice_bound_margin|1|99.8|1|0|0|0|
|advice_bound_margin_all_eligible|1|99.0|5|0|0|0|
|advice_bound_margin|2|97.6|12|0|0|0|
|advice_bound_margin_all_eligible|2|100.0|0|0|0|0|
|advice_bound_margin|3|95.4|23|0|0|0|
|advice_bound_margin_all_eligible|3|99.2|4|0|0|0|
|advice_bound_margin|4|99.8|1|0|0|0|
|advice_bound_margin_all_eligible|4|99.6|2|0|0|0|

配对差见paired_differences.csv及paired_difference_means.csv；全部5000条场景记录见scene_details.csv。失败完整轨迹路径见result_sources.csv，未来障碍信息仅用于事后诊断。
该确认只研究同地图独立动态场景；学习效率与撤教低谷仍以原50验证曲线判断，不主张地图泛化或保证安全。

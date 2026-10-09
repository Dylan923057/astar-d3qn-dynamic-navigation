# 统一配对seed0–4对比

尾段实验新冻结的同一批500场景，仅200000在线步最终模型，epsilon=0普通D3QN动作，无A*/动态风险保护，不选择最佳检查点，不调参。

统计单位为五个训练seed，均值±样本标准差（ddof=1）；配对差值按同seed相减，不以场景数扩大样本量。

| 方法 | safe_success_rate | dynamic_collision_rate | timeout_rate |
|---|---:|---:|---:|
| unguided_raw | 97.48±1.68% | 2.36±1.72% | 0.16±0.17% |
| unguided_bound | 97.56±2.54% | 2.40±2.54% | 0.04±0.09% |
| advice_raw | 19.52±43.65% | 2.60±3.41% | 77.88±43.22% |
| advice_bound | 98.92±1.32% | 1.08±1.32% | 0.00±0.00% |
| advice_margin | 50.32±46.32% | 2.56±2.81% | 47.12±45.70% |
| advice_bound_margin | 98.16±1.82% | 1.84±1.82% | 0.00±0.00% |
| supervision_only_bound | 97.76±1.44% | 1.76±1.17% | 0.48±1.07% |
| advice_bound_margin_tail140k | 96.84±2.43% | 1.76±1.71% | 1.40±1.81% |
| dqfd | 97.88±2.09% | 0.88±1.04% | 1.24±1.49% |
| dqfd_bound | 99.20±0.77% | 0.40±0.58% | 0.40±0.79% |
| advice_bound_margin_no_risk | 98.84±1.32% | 0.60±0.42% | 0.56±1.25% |

逐seed、分阶段AULC、窗口最低、最终碰撞、等待/循环/其他超时见summary.csv；配对逐seed差值及均值±样本标准差见paired_differences.csv与paired_difference_means.csv。
等待/循环分类沿用原轨迹分析，窗口累计次数包含重复场景检查，不能当成独立失败场景总数。

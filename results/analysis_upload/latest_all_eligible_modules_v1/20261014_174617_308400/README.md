# 最新完整方法：模块消融阅读包

[学习过程与模块报告](learning50/REPORT.md)、[逐seed](learning50/per_seed_metrics.csv)、[配对差](learning50/paired_differences.csv)、[配对均值/样本标准差](learning50/paired_means.csv)、[曲线](learning50/paired_validation_curves.png)。独立500结果在final500/summary.csv和method_means.csv，与原50训练曲线分开报告。

主对比：最新all-eligible完整组、新去TD组、新去A*代选组、已有advice_bound、原始D3QN（unguided_raw）。原筛选完整组和DQfD仅作背景并保留；DQfD有永久示范/PER和20000离线更新，总训练预算不同。

只新增两组各seed0–4，固定200000在线环境步；监督独立教师动作、TD真实动作。关闭代选仍推进监督时钟；去TD仅关闭完整目标[-6,10]裁剪；保留Smooth L1、梯度范数裁剪和全部其他设置。监督与有代选组在100000退出，不补偿标签数量，不增加示范或预训练。冻结源码及配置见registration/和source_snapshot/。

全部seed保留，数值失败写明准确发生步数并报告缺失，不能删seed或用剩余样本冒充五seed均值。所有最终评估仅200000模型、ε=0、关闭教师/路线引导/动态保护/Q输出裁剪，共同静态合法动作掩码；同一冻结500不用于调参。旧结果仅模型/场景/协议一致时复用。

阶段监督覆盖、执行一致率、margin、权重、Q、原始TD目标范围、裁剪及实际代选率在learning50/phase_mechanisms.csv；旧未记录字段保持缺失。监督比例分母为回放抽样次数，非独立转移数。首达90%和连续三点90%分开，未达到保持缺失。未来障碍信息仅用于事后轨迹诊断。

失败示例是低谷/最终处每类别首个保存场景，不选择模型；原始轨迹和权重/回放断点留在本地，模型精确路径/SHA见manifest.json。克隆缺权重不自动重训。旧实验与旧阅读包保留，没有参数搜索或额外方法训练。

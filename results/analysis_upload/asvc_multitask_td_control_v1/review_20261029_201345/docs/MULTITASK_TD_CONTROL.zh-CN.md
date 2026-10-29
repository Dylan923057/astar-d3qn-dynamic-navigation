# 多任务D3QN＋TD目标约束与循环诊断

本轮授权仅新增 `d3qn_bound` 的seed0、1，每次500000在线环境步、499501在线更新，无离线预训练、示范或预填。其他8次多任务运行直接复用。旧配置、冻结、模型、检查点、原始日志及已有上传包均保持原样。

## 单一算法变化

实现于 `scripts/td_control_engine.py`。TDControlAgent继承现有ExternalD3QNAgent；原始Double/Dueling Q、Smooth L1、Adam、静态动作掩码、均匀回放、真实执行动作与真实转移均沿用。唯一算法变化是完整目标 `raw=r+gamma*(1-terminated)*Q_target(next,argmax Q_policy)` 计算后裁剪到[-6,10]，不裁剪Q输出，不增加损失，不调用教师、监督或风险软权重。

共享旧240训练任务、50冻结验证任务、场景/任务生成序列、历史可见性修正后的观测、零路线通道及原网络/奖励。replay20000、batch64、lr0.0003、gamma0.99、目标同步250更新、梯度范数10。epsilon1→0.05在150000步内线性衰减，500步开始更新。超时max(100,4L*+20)，到达/碰撞不自举，超时仍自举。

每10000步同50任务贪心网络验证，最终固定500000步模型。原始/裁剪TD范围、裁剪比例、Q与梯度每次更新参与日志统计，每1000环境步输出区间统计；首批及每1000更新整批64个TD目标导出到td_target_samples.csv，含任务ID、实际动作、奖励、terminated/truncated及原始/裁剪目标。

完整恢复保存policy、target、optimizer、均匀回放及其RNG、全部全局/设备RNG、当前任务/场景、环境历史/动态内部状态、计数器、日志和审计采样进度。CPU及CUDA batch64验证恢复后动作、更新数值与网络状态精确一致。发生非有限值保留failure.json、失败状态和最近完整检查点，不自动重试、换seed或调参。

## 循环诊断

用户2026-10-29明确授权补齐C/D四个最终模型缺失动态轨迹。仅补每模型缺少46任务，共184次推断，逐任务核对成功/碰撞/超时/步数及等待/交互。原始成绩不改；不一致则将候选轨迹和差异另存reconciliation_failures并排查，不把新成绩覆盖旧成绩。

原四方法已有静态/动态完整轨迹直接复用，新控制动态复用原定最终验证、只新增50任务静态评估。诊断代码/输入有独立版本化协议，所有结果写新目录，不改旧协议。移除动态障碍也改变输入分布，不能替代单独静态训练。

所有动作先由原网络贪心执行并固定，之后重放保存动作重建相同观测，计算五动作Q、静态合法掩码、教师动作、独立BFS教师校验、目标位移一致性、教师合法动作Q排名/差值、回退动作Q偏好、硬筛查与局部观测4步风险。真实未来仅用于事后冲突误差分析，不传入风险估计或动作选择。

实际评估监督系数恒为0。假设状态权重C=.1*z、D=.1*z*(1-R)^2；另外记录排除该实际碰撞转移后的假设回放资格权重，不能把它们解释为评估中真的有监督。超时全部逐步导出，固定参考成功任务0/17/34作背景；典型循环按周期覆盖最大的三个超时以及等待最多的一个选择。完整五动作Q导出不使用角度比较。

另审计C/D四个最终回放各20000条转移，核对当前/后继目标位移、零路线通道、教师动作和存储风险重算。这不能覆盖所有历史抽样；未发现错误也不能写成全部训练过程已被证明无错配。

## 五方法比较与归因边界

结果位于 `results/asvc_multitask_td_control_v1/REPORT.zh-CN.md`。包括各seed、均值/样本SD、全程/阶段AULC、末10万步、末五点、短中长、静态/动态配对、循环与Q排序、原始/裁剪目标及成本。

`TD_only_minus_raw`表示TD约束单独影响；`C_minus_TD_only`表示A*代选与持续动作监督的教学组合增益，不能归于持续监督单项，也不把代数差分比例当独立因果贡献。DQfD继续单列每seed20000离线更新与3472示范仅成功覆盖70/240任务、31/40目标的限制，低成绩不直接归因于Q爆炸。

D维持探索方法；两seed冻结验证不构成正式显著性或未见目标泛化证据。CE、下限扫描、五seed、完整消融与两个测试池均不启动。同教师CE/大间隔待执行计划见 `docs/TEACHER_LOSS_COMPARISON_PLAN.zh-CN.md`。

## 实际命令

在仓库根目录和astar-drl环境中运行：

```powershell
# 首次预检；本轮已执行，已有登记会校验复用
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Prepare
# 首次启动仅两个seed，已有队列时拒绝重复
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Start -Workers 1
# 查看状态
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Status
# 原协调器/训练退出后，完整断点恢复；完成运行直接复用
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Resume -Workers 1
# 单个已有seed完整续训
python scripts/run_td_control.py train --seed 0 --resume
# 诊断/汇总；不会训练，已有同协议诊断复用
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Diagnose
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_td_control.ps1 -Action Report
```

默认1个训练进程；最多2个，并行第二个需至少4GiB可用内存。新入口只允许本控制seed0、1；不会据成绩改变预算。报告在控制未完成时写NA，完成后协调器自动补最终模型配对与五方法比较。

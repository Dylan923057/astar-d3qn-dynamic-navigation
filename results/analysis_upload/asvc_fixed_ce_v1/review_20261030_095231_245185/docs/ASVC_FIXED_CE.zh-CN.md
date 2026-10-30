# 固定起终点 20 万步：同教师在线 CE 对照

本轮按 2026-10-30 用户要求补固定起终点小论文。依据 `ASVC_CURRENT.zh-CN.md` 的当前持续弱监督方法，与 50 万步多任务 CE 提案独立。

## 两组与唯一变化

- 大间隔组：复用 `outputs/asvc_persistent_teaching_v1/supervision_floor/seed_0,seed_1`，m=0.8。冻结原配置、源码、曲线、日志、模型和结果哈希；不重新训练。
- CE 组：只新增 seed0、1，从头各训练 200000 环境步、199501 次更新。新配置 `configs/asvc_fixed_ce_v1.yaml`，独立 `data/outputs/results/asvc_fixed_ce_v1/`。
- 同一教师、同一静态合法性检查、三帧观测风险半径 1 筛选、碰撞标签排除规则；实际实现继承 `PersistentTeachingAgent` / `AStarAllEligibleSupervisionAgent`。继承配置中旧 `value_repair.labels` 描述是历史标签，当前有效规则由 `execution_consistency.require_executed_teacher_match=false` 和该类实现决定，合格教师标签不要求与执行动作相同。

\[
L=L_{TD}+\lambda(t)\,|E|^{-1}\sum_{i\in E}-\log\operatorname{softmax}(Q_{\mathrm{valid}}/1)_{a^*},
\quad\lambda(t)=\max(1-t/100000,0.1).
\]

非法静态动作以 -∞ 掩蔽，不参与 softmax；只在合格行计算 CE，E 为空返回 0。TD 仍读取原始不可变 Transition 的实际动作、奖励和下一观测。没有修改采样、回放、教师标签或额外风险软权重。

为保留历史源码哈希，新实现位于 `scripts/fixed_ce_agent.py`：AST 只替换原 D3QN 的监督损失块，保留 TD 目标、真实动作索引、优化器、梯度裁剪、目标网络更新等其余语句；教师包装函数只替换被调用的损失函数。生成后的完整代码文本与 AST 审计存入本轮冻结目录，便于检查。旧 `src/` 文件不改。

## 保持不变的协议

地图 irregular_workcell_91701，40×40，起点 (3,3)、终点 (36,36)；每回合原生成器的 3–5 动态障碍。5 动作，15×15 观测，5 空间通道/4 标量；原路线相关通道和额外标量保持零，观测历史按原实现保留。

普通步 -0.01、进展系数 0.05、额外等待项 0、碰撞 -1、目标 10；gamma=0.99；原 CNN/dueling 网络、hidden=256、Adam lr=0.0003、batch=64、uniform replay=10000、learning_starts=500、每步一次更新、目标同步每 250 次更新、全局梯度裁剪 10。epsilon 1→0.05，150000 步衰减。

A* 代选概率从 0.8 衰减，100000 步退出；监督 90000 步达到 0.1 下限并保留；TD **目标**裁剪 [-6,10]，不裁剪 Q 输出。无离线预训练，无预填示范。相同 seed 的初始化策略/目标网络/优化器/学习器与代选 RNG 必须与原 margin 记录一致；策略分化后，回合长度、采集经验和采样内容可以不同，不声称后续轨迹相同。

## 验证与统计

使用同一冻结 50 场景。与旧训练器完全一致，每跨过 10000 环境步后的回合边界验证，最多偏移一个回合；日志记录真实步数，例如旧组第一个点是 10062。不是将旧检查点伪装为精确整数步。终点固定为 200000，不用最佳验证模型，不使用最终 500 场景选择参数。

每次验证 epsilon=0，静态掩码；教师、风险保护和训练更新函数设为禁止调用，并在前后核验完整学习状态与全局 RNG 不变。首次预检还运行两个 seed 的原 50 场景初始化评估，与 margin 同 seed 第 0 步成绩核对。

AULC 按真实步数分段线性积分，边界 50000/100000 插值，除以区间长度，分别报告 [0,50000]、[0,200000]、[100000,200000]。后半程最低安全成功率只取实际位于 [100000,200000] 的验证点；另报后半碰撞/超时 AULC、验证点均值/最大值及最终值。未覆盖完整区间时留缺失，绝不外推。

统计单位为训练 seed：逐 seed、两 seed 均值/样本 SD、同 seed CE−margin 配对差。两 seed 只作探索性对照；重复使用 50 场景不等于增加独立训练次数。损失曲线分别标注 margin/CE，数值尺度不同，不凭损失大小宣称方法优劣。

## 产物及执行

- `data/asvc_fixed_ce_v1/freeze.json`：完整解析配置、原组保护哈希、初始状态配对、预检、源码哈希。
- `data/asvc_fixed_ce_v1/baseline_snapshots/`：原 margin seed0、1 的配置/曲线/逐场景/诊断日志按原字节复用。
- `outputs/asvc_fixed_ce_v1/seed_*/`：CE 配置身份、原始训练、逐场景曲线、失败轨迹、每次验证模型、最终模型及完整回放恢复点。
- `value_learning.csv`：Q 绝对范围、TD、原始 CE、实际逐更新系数×CE 的平均值及总损失；`supervision_alignment.csv`：Q 最小/最大值、标签覆盖率、标签与执行动作的一致比例；`advice_budget.csv`：实际代选预算。旧兼容检查点里的 margin 命名键在输出 CSV 中明确改为 CE 名称。
- `results/asvc_fixed_ce_v1/`：逐 seed、均值/样本 SD、配对差、学习曲线和价值/损失图、自动更新报告；未完成部分明确标注。
- `preflight_attempts/`：单元测试、24 步 CPU 恢复单元（不是正式 200k 运行）、初始化自主评估及任何预检失败。所有原组和本轮失败均保留。

原工作区 PowerShell：

```powershell
# 仅预检/复核冻结；不启动正式训练
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -u scripts/run_fixed_ce.py
# 后台启动两个顺序运行的 seed；仅操作本轮新进程
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_fixed_ce.ps1
# 查看进度或重新汇总现有数据
Get-Content results/asvc_fixed_ce_v1/queue_status.json
Get-Content outputs/asvc_fixed_ce_v1/seed_0/status.json
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' scripts/analyze_fixed_ce.py
```

已有完整运行不会被重训。意外中断后只能显式使用 `--resume`（或启动脚本 `-Resume`），恢复完整学习器、回放、工厂计数和随机状态；非有限失败不得自动恢复、换 seed 或调参。队列锁阻止重复运行；启动脚本检测其他 Python 进程时停止，不去关闭或重排其他任务。两个 seed 顺序执行，CPU 单线程；新后台进程使用隐藏窗口和较低优先级。

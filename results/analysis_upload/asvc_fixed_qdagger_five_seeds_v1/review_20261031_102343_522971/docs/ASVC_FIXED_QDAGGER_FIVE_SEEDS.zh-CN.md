# 固定起终点QDagger：补齐五seed并确认最终500场景

用户在查看原两seed预实验后授权本扩展。原seed0、1的冻结、模型、全量断点、曲线和日志保留；只新增seed2、3、4各20000离线更新＋200000在线环境步/199501在线更新，新增预算600000在线步、60000离线更新、598503在线更新。五seed累计1000000在线步和100000离线更新。与50万步多任务研究分开，不按结果改参数、删seed或选最佳检查点。

## 设置与实现

完整继承[原QDagger任务适配协议](ASVC_FIXED_QDAGGER.zh-CN.md)和configs/asvc_fixed_qdagger_v1.yaml。旧配置及代码不改，新配置为configs/asvc_fixed_qdagger_five_seeds_v1.yaml；只改变种子、目录和500评估登记元数据。训练仍由原run_fixed_qdagger.Experiment与fixed_qdagger_agent.QDaggerAgent执行；新入口仅在本进程内绑定独立登记/输出目录与新分析回调，不复制或替换更新、采集、回放或恢复语句。

继续复用原7260条DQfD训练示范（107/128成功回合，原生成10756环境步），不新增示范。三步Double-DQN＋Huber，教师one-hot，T=0.1、离线蒸馏权重3，在线λ(t)=3max(1−t/120000,0)。在线学生epsilon-greedy采集，batch64、uniform replay10000、lr0.0003、gamma0.99、250更新同步、梯度范数裁剪10，500在线步起每步更新，epsilon1→0.05/150000步。无A*代选、TD/Q裁剪、持续监督下限、风险软权重、PER或DQfD额外损失。

共用原静态地图、(3,3)→(36,36)、3–5动态障碍、观测/奖励、路线输入置零、静态动作掩码与300步超时。到达/碰撞不自举，超时/预算截断自举，三步尾部使用实际步长。复用原回合边界每万步50验证，记录真实步数；预训练前50单列，曲线0点采用预训练后网络。

## 冻结与预检

data/asvc_fixed_qdagger_five_seeds_v1保存新冻结、旧两seed只读曲线副本、四个既有方法各五seed的原始对照表、源码快照、50/500冻结输入及SHA。原两seed全过程和已有CE/大间隔500结果有SHA保护。

复用原预检中经源码SHA核对的教师/三步/真实动作TD检查，另检查：仅元数据改变、原入口与扩展入口同seed初始化和短程实际训练逐字节一致、CPU/CUDA离线与回合中断完整恢复、评估隔离、命名空间恢复、旧两seed拒绝重训、全部500场景及顺序相同。所有预检尝试单列，不算正式训练。非有限失败保留断点和日志，不自动换seed、加裁剪或调参。

## 最终500确认

五个最终模型各500回合，共2500回合；seed0、1也使用原200000步最终模型。相同场景取自data/asvc_fixed_ce_final500_v1/frozen500.json，核对原始场景SHA、复制文件SHA、任务/路线/相位及顺序。模型SHA在执行前按各自完成记录登记，不能选择中间模型。先评估现有seed0、1，随后每个新增seed完成后评估；评估顺序不影响冻结训练参数。

epsilon=0，仅网络贪心，关闭教师和动态保护，保留共同静态掩码；CPU、max_steps300、奖励/碰撞规则与原CE/大间隔500完全一致，无参数更新，学习状态和随机状态前后核对。50场景批次原子保存并支持中断接续；已完成模型/场景/协议一致的缓存复用，不重复运行。

逐场景成绩及全部失败轨迹保存在outputs/asvc_fixed_qdagger_five_seeds_v1/final500/seed_*。报告安全成功、动态碰撞、超时、步数、等待、成功子集路径效率与失败置0效率，逐seed、均值±样本SD和同seed对大间隔/CE配对差，复用这两组已完成的各2500回合。只用共同完成的seed汇总，未完成时明确n，不把不同样本数混成五seed比较。原50验证的AULC和最终500确认分开。

## 运行与续跑

```powershell
# 仅冻结/预检
python scripts/run_fixed_qdagger_five.py --prepare
# 顺序运行三个新增seed，并自动完成五个最终模型的500确认和汇总
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_fixed_qdagger_five.ps1
# 确认原进程退出后显式完整恢复；已完成seed和评估批次直接复用
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_fixed_qdagger_five.ps1 -Resume
# 只汇总既有结果
python scripts/analyze_fixed_qdagger_five.py
# 仅评估已完成的最终模型；禁止与队列同时运行
python scripts/run_fixed_qdagger_five.py --evaluate-only
```

最多一个训练进程，后台隐藏窗口、较低优先级；发现其他Python工作时不抢占、不结束它。完整续训复用原优化器、目标网络、回放、三步待决队列、环境、工厂计数、RNG与日志；仅加载权重不是完整续训。

这是查看原两seed结果后的授权扩展，不冒称最初即预登记的五seed实验。仍是A*教师与本任务预算的QDagger适配，不宣称原Atari完整复现、正式统计显著性或单项机制因果；不用500调参，不混入固定/多任务成绩，也不自动启动其他训练。

# 多任务外部基线与静态/动态配对诊断

本轮只允许原始 D3QN、DQfD 各 seed 0、1 四次新运行，每次 500000 在线环境步。不修改已完成的 C/D，不开启未见组合或未见目标测试池，不启动交叉熵、敏感性、五 seed 或正式消融。D 继续作为候选方法，结论限于两个 seed 预实验。

## 来源与冻结

共享协议来自 `configs/asvc_multitask_risk_pilot_v1.yaml`、`data/asvc_multitask_risk_pilot_v1/frozen_protocol.json`。新配置为 `configs/asvc_multitask_external_v1.yaml`；新输入、输出和分析分别位于 `data/`、`outputs/`、`results/asvc_multitask_external_v1/`。首次 Prepare 校验旧冻结、登记新源文件 SHA256、冻结示范和配对协议，随后拒绝源代码或输入改变。保留全部旧文件。

运行后发现报表读取新增sample_statistics时把online_updates重复作为关键字，已在 `data/asvc_multitask_external_v1/nontraining_corrections/001_report_counter_collision.json` 单独登记非训练修正：仅修复汇总器读取，并让验证器校验登记的修正SHA。原frozen_protocol.json、原source_snapshot、所有训练/环境/示范/配置源码保持不变，修正版本另存快照；运行身份不变。该修正不改变任何目标、数据、超参数、动作或更新。新的 `start_external_baselines.ps1` 薄包装还初始化PowerShell退出码，避免旧冻结启动器在新shell中成功启动后误报。

相同 60×60 irregular_workcell_91701 地图、240训练任务、50验证任务，短/中/长长度12–29/30–49/50–79，训练按三回合1:1:1分层。每回合3/4/5个动态障碍，场景兼顾参考路径、替代通路、背景，种子和回合序列与C/D共用。不同策略导致回合长度和实际访问状态不同，不宣称实际交互率相同。

输入沿用5×15×15局部网格和4个原有标量，历史动态通道只保留各时刻真实可见的障碍并按当前原点对齐。目标仅通过原有相对位移提供；规划路径不输入网络。网络骨干、动作（四邻接加等待）、奖励、gamma=0.99、碰撞及交换规则、静态动作掩码不变。超时固定 max(100,4×静态距离+20)。到达/碰撞为 terminated，无TD自举；超时为 truncated，仍从末状态自举，结束回合并清空n步待处理队列，防止跨目标串联。

在线回放容量20000、batch64、学习率0.0003、目标同步250次更新、梯度范数裁剪10。500步开始更新，在线499501次更新。epsilon从1到0.05、150000步线性衰减。每10000步同一50场景贪心网络验证，不做参数选择或按成绩提前结束。

## 两个外部基线

原始D3QN直接复用原学习器的Double/Dueling Q与Smooth L1，无A*代选、动作监督或TD范围裁剪。额外任务元数据仅留在回放，不参与标签生成。DQfD直接复用仓库原目标：平方单步TD、多步TD(n=10)、永久示范分区、PER、大间隔专家监督和L2；不添加TD约束。诊断包装增加原始目标/Q/梯度日志，数值单元检查验证和原实现逐次更新完全一致。

DQfD参数：20000离线更新，n步权重1、专家权重1、margin0.8、L2=1e-5；PER alpha0.4，beta离线0.6、在线从0.6线性到1；agent bonus0.001、demo bonus1，优先级为绝对单步TD误差。在线永久示范仍参与抽样，离线/在线更新与示范抽样数单独记录。DQfD使用平方TD是原实现要求，不替换成C/D的Smooth L1。

示范在任何基线训练前冻结：按任务ID排序，240个训练任务各尝试2次，共480次；seed=926100000+10×任务下标+尝试下标，障碍数量按尝试序号循环3/4/5。使用与C/D相同的静态最优动作场和局部可见硬筛查，标签不合格则等待；只保留完整成功且无碰撞轨迹，所有失败也登记，不增补预算。训练场景生成规则相同，示范场景种子独立且仅来自训练任务。两seed共用同一示范文件。

实际生成：77成功/403失败、3472转移；尝试15107环境步、240任务/40目标，成功70任务/31目标。示范覆盖较低是限制，不能隐去或追加生成来追求成绩。示范成本不算入500000在线步，但在比较中单列。永久示范3472条额外于20000在线回放。示范数据生成器及原预算登记保留；使用独立reader对压缩数组一次解压，避免逐条读取重复保留大数组。

## 配对诊断

先完成C/D，基线完成后自动补齐。动态条件复用每个最终模型的原50任务成绩，静态条件仅移除全部动态障碍，其他问题和推断设置相同。策略执行调用基础网络贪心入口，不经过采集/教师/风险/动态保护，检查前后优化器、网络和随机状态不变。

旧C/D只留3–4个完整任务的动作记录；可以离线重放已保存动作恢复这些轨迹，不重新运行策略。其余动态循环和终止距离记NA。额外动态策略重放须用户明确回复后另写授权记录，且逐任务核对成功/碰撞/超时/步数/等待/交互和原成绩一致。原成绩与源日志保持不变。新基线在原定500000步最终验证时保存全部轨迹，无额外动态评估。

等待比例=等待动作/总动作；重复访问=到达已访问格子的成功移动次数/成功移动次数；周期检测仅使用真实移动后的位置，周期2–12、至少重复3遍，避免把等待称为移动循环。记录终止位置的静态最短距离。短/中/长分别汇总；同一任务的静态−动态变化成对记录，再跨seed计算均值/样本SD。

统一冲突分组使用冻结最优动作场的名义路径，在名义时刻查询场景旧/新动态占据，虚拟参考不因碰撞提前终止；与四种网络实际是否遇见障碍无关。真实路线、后续冲突只在离线分组和案例阶段读取，网络动作决策不读取。案例选择规则：最低ID成功、最低ID碰撞、循环占比最高的超时（同分按ID）。案例每一步离线计算教师一致性、是否等待、局部观测风险R、最终阶段假设C/D权重、真实下一步及4步教师冲突；不把风险称为概率或安全保证。

## 尚不启动的CE/大间隔对照

主对照以C（无风险软权重）为公共设置，仅替换监督损失。共享教师动作场、混合目标回放正确标签、硬筛查z、代选日程与风险门控、TD约束[-6,10]、网络、回放、任务场景、超时、50万步和验证协议。禁用风险软权重以免同时变更多个机制。

当前大间隔为 max_a[Q(s,a)+m·1(a≠a*)]−Q(s,a*)，屏蔽非法静态动作；CE为 −log softmax(Q(s,a)/T)[a*]，同样屏蔽非法动作。两者均用 sum(z×lambda_base×loss)/max(1,sum z)，不除权重和。真实TD和碰撞经验仍保留。

建议冻结后给予两种损失相同调参预算：每种3个候选、每候选seed0/1各100000在线步（均含相同代选与基础监督日程）；大间隔m∈{0.4,0.8,1.6}，CE T∈{0.5,1,2}，其余权重均不动，两个方法均新做同等预算，不能只调CE。按冻结验证池0–100000 AULC的两seed均值选型，同分选中间值；正式各2seed、500000步另行批准。已有C的m=0.8只能作为参考，若未来调参后仍选0.8且所有冻结身份一致，才复用它的正式预算运行。该方案是同教师在线CE控制，不是完整QDagger复现。

## 尚不启动的监督下限敏感性

仅对C扫描floor∈{0,0.05,0.1,0.2}，其他设置完全固定，无风险软权重。基础监督从1在90000步线性衰减到所选floor并保持；不能同时扫描风险指数、代选概率、奖励或CE温度。每项seed0/1、500000步，无按成绩停训。

已核查可复用：本轮来源C/seed0、1是同多任务协议floor=0.1，完整500000步，可作该项结果。严格可比floor0、0.05、0.2尚无运行，最少需另行批准6次。D的有效权重受风险影响不能当作固定floor结果。`asvc_persistent_teaching_v1`和其五seed扩展是20万步固定任务，监督/代选下限组合也不同，不能混入多任务敏感性。旧消融中监督衰减到0同样不是严格可比多任务floor0。

## 旧TD约束证据

`results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/` 已有unguided_raw和unguided_bound各seed0–4、200000步固定任务结果、完整学习曲线和Q/TD日志。bound不含A*和动作监督，仅TD目标[-6,10]，可直接整理重画图。最终成功：raw 94.4%±5.18%，bound 97.6%±3.29%（样本SD）；归一化全程AULC分别0.29080和0.48356。只支持旧任务设置，当前多任务D3QN+TD约束控制组仍缺，单列待批准，不为重画图或补齐表格自动训练。

## 命令、恢复与交付

在仓库根目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Prepare
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Start -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Status
# 原协调器和训练进程均退出后使用：
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Resume -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Pairs
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_external_baselines.ps1 -Action Report
```

单个已有基线的完整续训（其他方法/seed同理）：

```powershell
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 -u scripts/run_multitask_external.py train --method dqfd --seed 0 --resume
```

队列只含4次，最多2个训练进程，内存不足自动降为1；完成的运行直接复用。恢复文件包含策略/目标/优化器、在线与永久示范回放及PER树/优先级/RNG、n步待处理记录、各类随机状态、当前任务/场景/环境历史和动态计数器、日志和日程进度。CPU与CUDA batch64恢复检查验证恢复后动作、更新结果与网络状态精确一致。只有模型权重不称完整恢复。

每10000步写完整checkpoint，DQfD离线每1000更新写checkpoint。非有限失败写failure.json、保留日志和最近完整状态、不自动重试或换seed。完成后队列自动补静态配对并重建四方法比较、曲线、样本SD、阶段AULC、案例和审稿证据表。中途报告明确保留待完成项，不能把队列启动写成结果已完成。

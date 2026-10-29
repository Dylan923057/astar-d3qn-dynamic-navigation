# 同教师在线CE与大间隔：待执行方案

状态：仅整理协议与作业请求，本轮不启动。它是同一静态A*教师的在线交叉熵监督控制，不是完整QDagger复现。配置为 `configs/multitask_teacher_loss_plan_v1.yaml`，训练开关明确false；实际训练入口尚未实现，不能把请求生成命令称为已就绪的训练命令。

与原C共享240训练任务、50冻结验证任务、场景序列、观测、网络、奖励、超时、自举、静态动作掩码、回放和更新预算。教师为同一按目标缓存的最优动作场，硬筛查和碰撞转移标签排除规则相同，A*代选日程/风险门控不变，完整TD目标约束[-6,10]不变，不使用D的风险软权重。

唯一对照项为合格样本的监督损失：当前大间隔 `max_a[Q(a)+m*1(a!=teacher)]-Q(teacher)`，CE为合法动作上的 `-log_softmax(Q/T)[teacher]`。都乘相同lambda_base并除以合格样本数，不除权重和；真实执行动作、真实转移和碰撞经验的TD不变。

沿用现有未执行提案：两种损失各3候选、每候选seed0/1各100000在线步，每种损失共6次、600000在线步。大间隔m为0.4/0.8/1.6；CE温度T为0.5/1/2，不再同时扫描监督权重等参数。按两seed的0–100000验证AULC均值选型，同分取中间值。两组调参预算相同，不能只调CE。

该提案的明确局限：10万步调参仅覆盖很短的弱监督保留阶段，不能据此选出后半程最稳定的参数。是否将两边调参预算同步延长，应在下一轮启动前另行冻结；本轮不擅自增加任何调参运行。调参和最终复用seed0/1也不是独立盲测重复。

最终预算拟各seed0/1、500000在线步。若选择的大间隔仍为0.8，且所有任务/源代码/超参数身份严格一致，现有C两次500000步可复用；其他候选需要对应新运行。未见组合和未见目标测试池仍不用于选择参数。

当前实际可执行命令仅校验/生成待批准请求，不训练：

```powershell
python scripts/plan_teacher_loss_control.py --check
python scripts/plan_teacher_loss_control.py --export
python scripts/plan_teacher_loss_control.py --loss cross_entropy --candidate 1 --seed 0 --phase tuning --export
python scripts/plan_teacher_loss_control.py --loss large_margin --candidate 0.8 --seed 0 --phase final --export
```

作业请求标明训练入口尚未实现。真正开训前仍需实现损失切换、验证相同筛查/代选/TD、预检和新的独立登记；不得借用本轮TD控制入口启动CE或扫描。

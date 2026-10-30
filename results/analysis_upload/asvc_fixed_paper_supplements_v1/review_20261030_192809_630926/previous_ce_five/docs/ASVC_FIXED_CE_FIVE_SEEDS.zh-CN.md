# 固定起终点 CE 对照：扩展至五 seed

用户在查看原两 seed 对照后，授权补跑 CE seed2、3、4。每个 seed 从头训练 200000 环境步、199501 次更新；本轮新增预算合计 600000 环境步、598503 次更新。CE seed0、1 以及持续弱监督大间隔方法 seed0–4 直接复用。原两 seed 配置、模型、数据、阅读包均保留。

继承 [原 CE 协议](ASVC_FIXED_CE.zh-CN.md)，唯一扩展是种子和汇总范围：T=1 的静态合法动作 CE、独立 A* 标签、原风险筛选和碰撞标签排除、TD 使用真实执行动作、监督下限0.1、100000步后代选退出、TD目标[-6,10]、epsilon 1→0.05/150000步，网络、观测、奖励、回放和更新预算均保持不变。不预训练，不加路径输入或风险软权重，不扫描参数。

本批与 500000 步多任务 CE 提案分开。仍使用原50冻结场景，每跨过10000环境步后的回合边界自主验证，epsilon=0、无代选、无风险保护、无参数更新，最终使用200000步模型；不选择最优中间模型，不自动增加最终500评估。

## 独立目录与复用

- 配置：`configs/asvc_fixed_ce_five_seeds_v1.yaml`。
- 冻结、来源哈希和只读副本：`data/asvc_fixed_ce_five_seeds_v1/`。
- 新训练：`outputs/asvc_fixed_ce_five_seeds_v1/seed_2`、`seed_3`、`seed_4`。
- 五seed汇总：`results/asvc_fixed_ce_five_seeds_v1/REPORT.zh-CN.md`。
- 原CE seed0、1：`outputs/asvc_fixed_ce_v1/`。
- 原大间隔seed0–2：`outputs/asvc_persistent_teaching_v1/supervision_floor/`。
- 原大间隔seed3、4：`outputs/asvc_persistent_supervision_five_seeds_v1/supervision_floor/`，按照当前方法登记核对身份及实际训练设置。

复用原CE runner、学习器和恢复代码，只在本进程内绑定新配置/路径与五seed分析入口，不修改原文件。预检核对各seed完整初始化及CPU/CUDA一致性、同seed原CE与扩展入口的动作/转移/优化器更新一致性、训练场景生成、原50自主评估隔离与完整断点恢复。原始文件及新冻结副本均有哈希保护。

## 命令

预检（不启动正式训练）：

```powershell
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' scripts/run_fixed_ce_five.py
```

后台依次启动 seed2、3、4，低优先级、每次一个进程；若已有 Python 工作则停止启动，避免抢占其他任务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_fixed_ce_five.ps1
```

原进程确认退出且有完整恢复检查点时可显式加 `-Resume`；不重跑已完成seed、不自动重试非有限失败。进程异常留下的锁需先核查，脚本不自动删除。

```powershell
Get-Content results/asvc_fixed_ce_five_seeds_v1/queue_status.json
Get-Content results/asvc_fixed_ce_five_seeds_v1/REPORT.zh-CN.md -Encoding UTF8
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' scripts/analyze_fixed_ce_five.py
```

训练到每个验证点自动更新CSV/报告，每个seed完成及队列结束更新PNG/PDF，不需要另行启动分析队列。

## 统计与判断

报告逐seed、配对差和均值±样本SD：前5万、全程、后10万AULC，后半最低安全成功、最终安全成功/碰撞/超时、后半碰撞/超时AULC。以实际环境步数积分，边界线性插值，不外推未完成区间。尚未完成时，汇总只用两组共同完成的相同seed，并明确标注n，不把“原组5个、CE不足5个”当成五seed比较。

学习曲线包含均值±样本SD及逐seed曲线，另保留Q/损失诊断图、全部原始验证记录、失败轨迹、标签覆盖和训练成本。曲线可视化在共同步数网格插值，阴影不是置信区间。不同时间运行的耗时不作为效率优势证据。

这是看到前两seed后进行的扩展，未调整参数、筛选seed或追加训练预算。原50场景是重复验证池，不冒充独立测试。结果区分实际观测与原因假设，不预设CE一定改善，不自动更换已登记主方法。

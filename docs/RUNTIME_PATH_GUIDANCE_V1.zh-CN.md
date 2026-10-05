# 实时 A* 路线引导接入 v1

该入口已完成seed0–4的正式公平对照：每seed的两组均训练200000环境步。
旧配置、旧输出、旧权重均保留。入口默认仅检查；必须显式传入 `--train` 才会开始正式训练。
完整结果见[五seed分析](../results/whole_map_91701_runtime_path_v1/analysis_20261005_094812_262464/REPORT.md)。
最终平均安全成功率为无引导94.4%、有引导95.6%；学习速度收益因seed而异，尚未证明稳定优势。

## 方法

- `unguided`：原四通道动态观测与最终目标方向，新增输入全部置零。
- `path_guided`：同一观测上增加一个局部静态 A* 路线通道，以及固定前视子目标的两个相对坐标。
- `astar_replan_wait`：已知静态地图，每步仅根据当前局部观测到的动态占据重新做 A*；当前观测图无路则等待。没有读取完整动态路线、相位或未来位置。它可能被即将进入下一格的障碍撞到，没有安全保证。

两组 D3QN 均为 `(5,15,15)` 空间输入与 4 个标量，模型结构和参数数量相同。
前两个标量仍是最终目标相对方向，后两个是子目标相对方向。控制组只将新增通道和新增标量置零。
同 seed 两组从完全相同的随机网络、优化器和随机数状态开始；两组均不采样 A* 示范。

前视目标：找当前位置距离名义路线最近的格（Manhattan 距离，相同时选最早索引），再向前取 4 条路径边，末尾截断到最终目标。
每步从当前位置重新投影，不保留隐藏的进度游标。整个静态名义路线在当前局部窗口内绘制为二值通道。
子目标只作为观测，不覆盖动作、不替换最终终点，也不增加路径跟随奖励。
等待、偏离路线和重回路线都由原动作实现。暂不加入自适应子目标、进度锁定或障碍预测。

## 保持的条件

- 地图 `irregular_workcell_91701`，原起点 `(3,3)`、终点 `(36,36)`。
- grid SHA：`789e02a19e945736e2ae78fa54163dd58e069df568d777f76577f6b8ccede0ff`。
- route pool v2 design SHA：`d9a57b3af3fb88bdef4601b07115a0ea8d7f92c040de3a9797b1439d768de3be`。
- 全图路线池、每回合 3–5 个动态障碍、每步移动、局部动态历史观测、五个动作、静态动作掩码、奖励和碰撞结束规则均沿用已有实现。
- 正式入口登记：每组 200000 环境步，online replay 10000，batch 64，learning starts 500，epsilon 1.0→0.05、150000 步衰减；不加载 foundation，不使用 Adaptive。
- 50 个固定验证场景由原验证采样种子生成，两组共用。同 seed 共用按回合索引的训练场景序列；行为不同会使回合长度不同，不声称逐环境步动态经历相同。
- 只使用 validation，不读取或生成 test。正式训练将记录 step 0、周期验证、最终验证以及失败逐步轨迹。

本配置保留原地图和起终点，是接入与第一轮机制检查版本，不能据此宣称多目标或跨地图泛化。
多个起终点需要另行登记任务并检查动态路线对新起终点的保护条件，不能直接改变原路线池工厂的任务对象。

## 权重兼容性

旧网络为四空间通道、两个标量，新网络为五空间通道、四个标量，旧 foundation 权重和旧 replay 不能直接完整加载。
本轮不自动重训 foundation，也不做隐式补零权重迁移。已有实验可作背景参考，但训练起点和输入不同，不能直接拿旧成绩证明新输入的收益。
新公平对照使用本入口的 `unguided` 与 `path_guided` 两组。

## PowerShell 命令

在项目根目录运行。以下命令均不会启动正式训练：

```powershell
python scripts/run_runtime_path_guidance.py --check --seeds 0 1
python scripts/run_runtime_path_guidance.py --smoke --seeds 0 1
python scripts/run_runtime_path_guidance.py --evaluate-rules
```

`--check` 不做任何梯度更新，检查地图 SHA、50 个相同验证场景、原环境行为一致性、新观测和相同网络初值。
`--smoke` 每 seed 每组只跑 8 个 CPU 环境步，batch/learning starts 临时设为 4，epsilon 衰减预算临时设为 8 步以满足原训练器的预算约束，验证真实回放采样和优化器接入；输出属于 integration，不能作为训练效果证据。result.json 同时保存正式登记配置和实际检查配置。
`--evaluate-rules` 只评估 A* 重规划/无路等待规则，保存逐场景指标和完整轨迹。
所有检查使用带时间戳的新目录，位于 `results/whole_map_91701_runtime_path_v1/`。

下面是以后显式启动正式训练的命令，当前不执行：

```powershell
python scripts/run_runtime_path_guidance.py --train --seeds 0 --methods unguided path_guided
python scripts/run_runtime_path_guidance.py --train --seeds 1 --methods unguided path_guided
```

正式输出为 `outputs/whole_map_91701_runtime_path_v1/<method>/seed_<seed>/`，已有 run 目录会拒绝覆盖。
入口支持seed0–4；本次补跑seed2–4使用 `--train --seeds 2 3 4 --methods unguided path_guided`。
五seed结果均已完成，此命令在当前本地运行会因已有目录而拒绝覆盖。
评价报告安全成功率、动态碰撞率、超时率、等待、重复格、步数与逐步失败轨迹。短期检查不用于宣称优越性。

## 5 万步短期对照入口

新增独立配置 `configs/whole_map_91701_runtime_path_pilot_v1.yaml`，每 seed 两组均训练 50000 步。
为完成一次短期探索过程，epsilon 1.0→0.05 在 37500 步衰减，保持衰减占总预算 75% 的比例。
两组其他条件相同，但该短期试验不能视为原 200000 步设置的精确前 50000 步，也不能直接续作原正式实验。
不使用示范或旧 foundation。默认只检查；显式 `--pilot` 才会训练，两组依次执行。

```powershell
python scripts/run_runtime_path_guidance.py --pilot --seeds 0 --methods unguided path_guided
python scripts/run_runtime_path_guidance.py --pilot --seeds 1 --methods unguided path_guided
```

短期训练输出在 `outputs/whole_map_91701_runtime_path_pilot_v1/<method>/seed_<seed>/`；已存在的 run 拒绝覆盖。
只使用原 50 个 validation，不使用 test。两个 seed 是探索性试验，不能据此确认稳定优势或宣告方向失败。

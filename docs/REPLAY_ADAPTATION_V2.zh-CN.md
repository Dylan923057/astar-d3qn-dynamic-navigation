# 动态交互位置均衡版 v2

v2 修正 v1 动态路线集中于任务前中段、后段没有交互的问题。三张静态地图、起终点、示范和训练参数均保持不变，更新动态路线候选、场景配额、预览和审计。旧 manifest、预览和输出保留，v2 使用独立目录。

## 必须满足的位置覆盖

以完整参考路径的步数为分母划分三段，**不是按现有候选位置的分位数重新命名**。三张地图均为66步，因此决策步骤0–21为early，22–43为middle，44–65为late。

| 集合 | 前段配对 | 中段配对 | 后段配对 | 不重复决策位置 | 场景数 |
|---|---:|---:|---:|---:|---:|
| train | 6 | 6 | 6 | 18 | 36 |
| validation | 2 | 2 | 2 | 6 | 12 |
| test | 4 | 4 | 4 | 12 | 24 |

每个集合内同一位置只选一对，不允许改变相位或路线长度后重复占用位置名额。跨集合仍允许位置/路线几何共享，但完整运动场景不重复。这是同地图机制实验，不冒充未见位置泛化。

位置选择优先拉开参考步数间隔，并同时分配三集合；配额不足时明确失败，不由别的分段补齐。硬性间隔为至少1个参考步，即位置不重复；没有声称所有路线或相邻观察窗口相互不重叠。生成时曾检查更强的2步间隔，但第二张图中段在保留所有行为见证的条件下无法满足6个位置，因此没有把这条额外要求作为协议条件。

## 路线与安全条件

候选增加3格短横穿、交汇处转弯和穿过交汇点后转弯的路线，使参考路径拐角附近也能被覆盖；原来的5/7/9格横穿继续作为候选。每场仍只有一个匀速往返障碍。3格只是候选长度，不保证入选：例如保护障碍旧/新位置的碰撞规则可能使三格直线的中点始终无法安全通过，这类候选仍会被剔除。

沿用以下验收：

- control中参考路径安全，conflict中参考路径冲突；
- conflict对全部20条实际示范的冲突数高于control；
- 存在可见后等待、局部绕行后重接原路的安全见证；
- 绕行仍限制在决策位置的局部范围内，最多在10个参考步后重接，额外代价不超过8步，无重复格；
- 所有见证与时空最短路径均由实际环境逐步回放检查；
- 场景接受不依赖任何学习策略的表现。

静态障碍没有为了补齐配额而移动，地图身份通过grid SHA-256与v1核对。

## 查看与运行

当前三个脚本默认读取 `configs/replay_adaptation_v2.yaml`，原先两条训练/汇总命令不需要加参数即可使用新版：

```powershell
python scripts/run_replay_adaptation.py --map-index 0 --seeds 0 1 --device cuda
```

等待训练正常结束、终端重新出现输入提示后，再运行：

```powershell
python scripts/summarize_replay_adaptation.py --map-index 0 --seeds 0 1
```

如需重新输出预览，不会启动训练：

```powershell
python scripts/prepare_replay_adaptation.py
```

主要文件：

- 配置：`configs/replay_adaptation_v2.yaml`
- 数据：`data/replay_adaptation_v2/manifest.json`
- 逐位置审计：`data/replay_adaptation_v2/coverage_audit.csv`
- 地图总览：`maps/previews/replay_adaptation_v2/map_overview.png`
- 第一张图路线：`maps/previews/replay_adaptation_v2/irregular_workcell_91701_routes.png`
- 配对示例：`maps/previews/replay_adaptation_v2/irregular_workcell_91701_examples.png`
- 正式结果：`outputs/replay_adaptation_v2/formal/`

路线图用橙/紫/青区分前/中/后段，叠加显示整个场景池，不能理解为每场同时存在全部障碍。示例图依次展示训练前段、验证中段、测试后段。

同源分叉、静态合格门槛、固定交互预算、评估指标和结论边界沿用 `REPLAY_ADAPTATION_V1.zh-CN.md`。正式训练不由生成脚本或单元测试启动；数据覆盖通过不等于已经证明静态可学性或固定示范有害。

若明确查看历史v1，需给脚本附加 `--config configs/replay_adaptation_v1.yaml`；不要把v1、v2结果合并。

## 本次验收结果

三张图全部达到上表配额，训练/验证/测试各有18/6/12个不同决策位置。第一张图训练位置从原来的参考第14–39步扩大到第2–64步。14项单元测试通过，包括v1/v2全部见证回放、三段边界、位置去重、缺失后段必须失败，以及原有同源分叉检查。新版CPU短流程及统计输出检查通过。尚未启动正式训练，这些结果只说明场景与代码验收通过。

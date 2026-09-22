# 动态空间场景冲突审计与补充压力评估

更新时间：2026-09-01

本分析用于回答：固定 25% Persistent Demo 是否会在静态 A* 路径与动态障碍发生强冲突时系统性退化，以及现有证据是否支持立即实现冲突感知自适应示范比例。

结论先行：**当前结果不支持立即开展自适应训练。** 固定 25% 在两张地图的低、中、高冲突压力档中均未被固定 10% 或 Prefill 稳定反超。Map 1 存在很大的训练 seed 分化，但它不是固定 25% 独有；Map 2 在强相位对齐压力下仍然接近满分。自适应只能保留为以后在更多、难度匹配地图上重新检验的候选假设。

## 一、执行内容

新增：

- `src/astar_d3qn/evaluation/conflict.py`：障碍轨迹、几何相交和 nominal A* 到达时序冲突指标；
- `scripts/run_spatial_conflict_stress.py`：manifest 审计、正式 test 冲突分层复分析、配对相位压力场景生成及 selected-model 评估；
- `scripts/evaluate_spatial_conflict_agreement.py`：在“照静态 A* 下一步走会立即动态碰撞”的状态中测量策略是否仍执行 A* 动作；
- `tests/test_conflict_analysis.py`：障碍反弹时序、直接冲突和 off-path 情形的单元测试。

主要输出目录：`outputs/spatial_conflict_stress_v1/`。

本次没有重新训练模型。使用的是 Map 1、Map 2 已有的 `model_selected.pth`。压力评估共执行 2400 个 rollout：

- Map 1：Uniform、Prefill、Persistent 25%、Persistent 10%、25%→0，5 个训练 seed；
- Map 2：Uniform、Prefill、Persistent 25%，5 个训练 seed；
- 每张地图 20 组 test 路线组合，每组构造 low/medium/high 三种相位，共 60 个场景。

同一个 block 的 low/medium/high 使用完全相同的 3 条 corridor 和 2 条 background 路线，只修改 `start_index` 和 `direction`，所以主要操纵的是动态相位，而不是路线位置。

## 二、冻结 manifest 冲突审计

“直接路线”表示动态路线至少有一个格子与注册的 nominal A* 路径重合。“精确时序冲突”表示假设智能体每步沿 nominal A* 前进时，进入下一格会与障碍当前或下一位置冲突。

| Map | Split | 场景数 | 无直接相交路线 | 至少2条直接路线 | 存在精确时序冲突 | 平均冲突分数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Map 1 | train | 100 | 48% | 7% | 34% | 23.02 |
| Map 1 | validation | 20 | 40% | 0% | 25% | 15.29 |
| Map 1 | test | 50 | 52% | 10% | 36% | 14.27 |
| Map 2 | train | 100 | 33% | 23% | 34% | 29.17 |
| Map 2 | validation | 20 | 25% | 15% | 50% | 42.09 |
| Map 2 | test | 50 | 60% | 2% | 24% | 13.15 |

这说明：

1. 原 manifest 的空间隔离有效，但没有同时约束 split 难度分布；
2. Map 2 validation 明显比 test 更高冲突，Map 2 test 偏容易；
3. Map 1 validation 没有任何包含至少两条直接相交路线的场景，但 test 有 10%；
4. 以后生成新地图时，不能只平衡路线数量，还要平衡直接相交数、相位冲突率和冲突分数。

## 三、原正式 test 的冲突分层复分析

把每张地图原来的 50 个 test 场景按 intrinsic conflict score 排序并分成三等份。以下为 5 个训练 seed 的安全成功率均值：

| Map | 方法 | Low | Medium | High |
| --- | --- | ---: | ---: | ---: |
| Map 1 | Uniform | 92.9% | 83.5% | 73.8% |
| Map 1 | Prefill | 95.3% | 88.2% | 70.0% |
| Map 1 | Persistent 10% | 95.3% | 85.9% | 70.0% |
| Map 1 | Persistent 25% | 77.6% | 72.9% | 78.8% |
| Map 1 | 25%→0 | 72.9% | 75.3% | 73.8% |
| Map 2 | Uniform | 92.9% | 94.1% | 93.8% |
| Map 2 | Prefill | 92.9% | 94.1% | 95.0% |
| Map 2 | Persistent 25% | 98.8% | 100.0% | 100.0% |

Map 1 中 Uniform、Prefill 和固定 10% 随冲突档位升高而下降，但固定 25% 没有出现这种单调下降；Map 2 固定 25% 在三档都最好。因此原 test 不支持“冲突越强，固定 25% 相对越差”。

直接按有无 nominal 路线相交重新分组也得到相似结果：Map 1 中 Uniform、Prefill、固定 10% 在 direct 场景下降明显，而固定 25% 从无直接相交的 78.5% 变为直接相交的 74.2%，下降较小。

## 四、配对相位压力评估

压力场景的 intrinsic 难度如下：

| Map | 档位 | 平均直接路线数 | 存在精确冲突 | 平均精确冲突数 | 平均冲突分数 |
| --- | --- | ---: | ---: | ---: | ---: |
| Map 1 | Low | 1.4 | 0% | 0.0 | 7.43 |
| Map 1 | Medium | 1.4 | 100% | 1.4 | 43.83 |
| Map 1 | High | 1.4 | 100% | 2.2 | 64.63 |
| Map 2 | Low | 1.2 | 100% | 1.0 | 41.51 |
| Map 2 | Medium | 1.2 | 100% | 1.0 | 60.71 |
| Map 2 | High | 1.2 | 100% | 5.2 | 144.71 |

Map 2 的部分 test 路线与 nominal path 重合多个格子，所有相位都至少产生一次精确冲突，所以其 Low 是“相对低”，不是零冲突。

Selected-model 压力结果：

| Map | 方法 | Low safe | Medium safe | High safe |
| --- | --- | ---: | ---: | ---: |
| Map 1 | Uniform | 18% | 46% | 34% |
| Map 1 | Prefill | 51% | 49% | 32% |
| Map 1 | Persistent 10% | 35% | 50% | 33% |
| Map 1 | Persistent 25% | **59%** | **71%** | **47%** |
| Map 1 | 25%→0 | 47% | 59% | 29% |
| Map 2 | Uniform | 97% | 94% | 98% |
| Map 2 | Prefill | 97% | 96% | 94% |
| Map 2 | Persistent 25% | **99%** | **100%** | **100%** |

Map 1 的 95% 区间很宽，原因是训练 seed 分化非常严重。例如固定 25% 的总体 60 场景安全率按 seed 为 86.7%、41.7%、63.3%、88.3%、15.0%。压力场景不是导致这种差异的唯一原因：seed 4 在三档中的 success 都只有约 20%，但动态碰撞率约 5%，主要是无法到达而不是频繁碰撞；seed 0 和 3 则保持较高安全率。

另一个重要限制是：Low/Medium/High 是相对于 nominal A* 到达时序定义的。学习策略可能绕行、等待或根本不走 nominal 路径，因此 Low 不保证对每个已训练策略都最容易。这也是 Map 1 部分指标不单调的主要原因之一。

## 五、冲突状态下的 A* 动作一致率

诊断状态定义为：

- 策略当前正位于注册的 nominal A* 路径；
- 如果执行静态 A* 的下一步动作，该目标格会被动态障碍的当前或下一位置占用。

主要观察：

- Map 1 High 档中，固定 25% 有 4/5 个 seed 实际遇到这种状态；其 A* 动作一致率分别为 0%、34.8%、100%、无该状态、0%；
- 同档 Uniform 的非空 seed 一致率为 0%、0%、50%、25%；固定 10% 为 50% 和 0%；Prefill 也出现 32% 等非零值；
- 一旦在该定义的状态继续执行 A* 动作，记录到的情况基本都会动态碰撞，说明诊断定义有效；
- 固定 25% 的部分 seed 确实存在 A* 冲突盲从，但另外一些 seed 能完全避开，且相同问题并非 Persistent 独有；
- Map 2 几乎没有策略真正到达这种即时冲突状态，说明策略实际路径和到达相位已经避开了 nominal 假设中的冲突。

因此，存在“个别 seed 的静态示范动作偏置”信号，但没有证据表明全局降低 demo fraction 能稳定解决它。固定 10% 在压力结果中没有优于固定 25%，时间衰减也没有优势。

## 六、决策

当前不执行 CA-ADR 或其他全局示范比例自适应训练。原因是预先提出的决策条件没有满足：

- 没有出现固定 25% 从 Low 到 High 被其他方法稳定反超；
- Map 2 固定 25% 在强相位压力下仍是最好；
- Map 1 的主要不确定性是训练 seed 和空间路线泛化，不是一个单调的冲突强度效应；
- A* 冲突盲从存在于部分 seed 和多种 replay 方法中，不能归因为 Persistent 25% 独有。

下一步应改为：

1. 修改未来地图的场景生成协议，使 train/validation/test 在空间隔离的同时匹配冲突难度分布；
2. 新增至少 3 张结构不同的地图，预先冻结每个 split 的直接相交数、精确相位冲突率和难度范围；
3. 在新地图上至少比较 Uniform、Prefill、Persistent 10%、Persistent 25%，不要继续使用“25% 单向降到 0”作为自适应代表；
4. 只有在多张新地图上重复出现“低冲突 25% 好、高冲突 10% 好”的交叉，才实现可逆的冲突感知比例；
5. 如果仍然是固定 25% 更稳，应放弃全局自适应主线，转向更有针对性的状态级风险过滤、强基线和多地图统计。

## 七、结果文件

- `REPORT.md`：自动生成的英文汇总；
- `manifest_split_summary.csv`：两张地图各 split 的冲突分布；
- `manifest_route_conflicts.csv`：逐路线几何及可实现时序冲突；
- `manifest_scenario_conflicts.csv`：逐正式场景 intrinsic 指标；
- `formal_test_by_conflict.csv`、`formal_test_conflict_summary.csv`：原正式 test 分层结果；
- `stress_scenarios.json`：60 个/地图的配对压力场景；
- `stress_evaluation.csv`、`stress_summary.csv`：2400 个 rollout 和聚合结果；
- `astar_conflict_agreement.csv`、`astar_conflict_agreement_summary.csv`：冲突状态下的静态 A* 动作一致率。

所有压力结果都是 post-hoc supplemental evidence，不能替换或回头修改 Map 1/Map 2 已冻结的正式 test 结论。

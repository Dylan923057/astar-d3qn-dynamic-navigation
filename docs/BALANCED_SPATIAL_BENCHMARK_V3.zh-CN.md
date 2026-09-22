# 三地图冲突均衡空间泛化基准 v3

更新时间：2026-09-01

## 目的

Map 1/Map 2 已经实现训练、验证、测试动态路线的空间隔离，但三个 split 的冲突难度没有被同时控制，尤其 Map 2 的 test 整体偏容易。因此，Map 1/Map 2 的结果适合保留为历史证据，却不足以单独判断示范回放比例为何在不同地图上出现排名变化。

v3 不覆盖旧地图和旧结果，而是新增三张结构不同的 40×40 地图，并同时控制：

- 静态拓扑；
- train/validation/test 动态路线位置隔离；
- 每个 split 中低、中、高冲突场景的数量和主要冲突指标；
- 每个场景的障碍物数量、路线类别和运动速度。

本次只完成地图、场景清单、审计和测试，没有运行训练。

## 三张地图

| 地图 | 场景结构 | 静态障碍格 | nominal A* 步数 | 转弯数 |
| --- | --- | ---: | ---: | ---: |
| `office_40x40` | 房间、墙体、门洞和主走廊 | 236 | 70 | 5 |
| `parcel_station_40x40` | 中央分拣岛、装卸通道和外围工作区 | 382 | 70 | 7 |
| `warehouse_40x40` | 平行货架、横向通道和多处瓶颈 | 388 | 70 | 9 |

三张地图的起点均为 `(2,2)`，终点均为 `(37,37)`。它们不是同一张地图换障碍物 seed，而是三种确定性的结构拓扑。

静态总览：`maps/previews/spatial_generalization_balanced_v3_static_overview.png`

## 数据集与动态障碍物

每张地图仍包含 100 个训练场景、20 个验证场景和 50 个测试场景。每个场景固定有 5 个动态障碍物：3 条 corridor 路线和 2 条 background 路线，全部每个环境步移动一格。

路线池规模保持三个 split 不同，但在三张地图之间完全一致：

| split | corridor 路线池 | background 路线池 | 场景数 |
| --- | ---: | ---: | ---: |
| train | 9 | 7 | 100 |
| validation | 5 | 4 | 20 |
| test | 5 | 6 | 50 |

不同 split 的任何路线格之间至少保留一格空白缓冲，即 Chebyshev 距离至少为 2。background 路线不得与注册的 nominal A* 路径相交。路线组合最多重复两次，完整的“路线＋初始位置＋方向”配置不重复。

## 冲突分层

每个 split 都按 40%/30%/30% 划分：

| 层级 | train | validation | test | 直接穿过 nominal A* 的路线数 | 时间语义 |
| --- | ---: | ---: | ---: | ---: | --- |
| low | 40 | 8 | 20 | 0 | 无直接交叉、无近时冲突 |
| medium | 30 | 6 | 15 | 1 | 无即时冲突，但在 ±2 步窗口内恰有一次近时冲突 |
| high | 30 | 6 | 15 | 2 | 保证存在即时冲突 |

同一张地图内，train/validation/test 的 low 和 medium 平均即时冲突数均为 0；high 平均即时冲突数在三个 split 中完全相同：Office 为 2，Parcel Station 为 2，Warehouse 为 3。Warehouse 的 high 数值更高是地图路径与货架通道形成连续交叉所致，不应把不同地图的绝对冲突分数当作相同难度。

冲突定义基于“智能体沿注册 nominal A* 每步前进一格”时的障碍物时序，是场景的内在描述，不是对学习策略真实轨迹的难度保证。策略绕开 nominal 路径后，low/medium/high 不一定严格单调。

完整审计位于：`outputs/spatial_generalization_balanced_v3_design/`

- `scenario_conflicts.csv`：每个场景的冲突分量；
- `split_stratum_summary.csv`：每张地图、每个 split、每个层级的均值；
- `manifests.json`：清单路径和 SHA-256。

## 冻结清单与配置

| 地图 | 配置 | frozen manifest | SHA-256 前 12 位 |
| --- | --- | --- | --- |
| Office | `configs/dynamic_spatial_generalization_office_balanced_v3.yaml` | `data/dynamic_scenarios/office_40x40_balanced_v3.json` | `d4b3e5f7909e` |
| Parcel Station | `configs/dynamic_spatial_generalization_parcel_balanced_v3.yaml` | `data/dynamic_scenarios/parcel_station_40x40_balanced_v3.json` | `99d92dd8f639` |
| Warehouse | `configs/dynamic_spatial_generalization_warehouse_balanced_v3.yaml` | `data/dynamic_scenarios/warehouse_40x40_balanced_v3.json` | `d6561604cc9f` |

若重新运行生成器，必须把 manifest hash 变化视为实验协议变化；已经开始正式训练后不要再重生成清单。

## 推荐实验顺序

第一阶段只比较 `uniform`、`prefill`、`persistent_demo` 固定 25%，每张地图 5 个训练 seed。三种方法在同一地图必须共享同一个 frozen manifest。共计 `3 地图 × 3 方法 × 5 seed = 45` 次训练。

先用 validation 规则选定 checkpoint，再一次性读取 test；不要根据 test 结果换 checkpoint 或调比例。主要报告：

- test `safe_success_rate`；
- `success_rate`；
- 动态碰撞率和动态碰撞次数；
- 路径步数/效率；
- 按 low、medium、high 分层的上述指标；
- 5 seed 均值、标准差、95% bootstrap 置信区间和配对 seed 差值。

只有固定 25% 在至少部分地图或冲突层上出现可重复的反转或明显代价，才有充分理由继续设计冲突感知自适应比例。不要先假设自适应一定更好。

## 命令

重新生成清单和图片（正式训练前才允许执行）：

```powershell
python -u .\scripts\generate_balanced_spatial_benchmark.py
```

运行三地图、三策略、五 seed 正式训练：

```powershell
foreach ($config in @('.\configs\dynamic_spatial_generalization_office_balanced_v3.yaml','.\configs\dynamic_spatial_generalization_parcel_balanced_v3.yaml','.\configs\dynamic_spatial_generalization_warehouse_balanced_v3.yaml')) { foreach ($strategy in @('uniform','prefill','persistent_demo')) { foreach ($seed in 0..4) { python -u .\scripts\train_dynamic_spatial_generalization.py --config $config --strategy $strategy --seed $seed; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } } }
```


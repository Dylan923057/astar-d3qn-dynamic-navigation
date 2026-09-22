# Static Local Observation Baseline v1

冻结日期：2026-08-07

## 1. 基线目的

本基线用于比较 Dueling Double DQN 在固定静态栅格地图和局部观测条件下的三种经验回放策略：

1. `uniform`：仅使用在线均匀经验回放；
2. `prefill`：训练开始前将 A* 示范一次性写入普通 FIFO replay；
3. `persistent_demo`：A* 示范保存在不可淘汰的独立分区中，并在训练期间持续采样。

该基线只研究 A* 示范回放对学习效率和最终静态路径性能的影响，不包含动态障碍，也不声明跨地图泛化能力。

## 2. 冻结文件

- 配置：`configs/static_local_baseline_v1.yaml`
- 地图：`maps/random_benchmark/train/maps.json`
- A* 示范：`data/demonstrations/train_astar_demos.npz`
- 示范元数据：`data/demonstrations/train_astar_demos.json`
- 结果根目录：`outputs/random_benchmark/Static Local Observation Baseline v1/`

正式运行目录：

- `20260807_132000_uniform`
- `20260807_133358_prefill`
- `20260807_134641_persistent_demo`

每个运行目录包含最终模型、`training.csv`、冻结 greedy 评估、训练曲线和最终路径图。

## 3. 地图与任务

- 地图数量：5
- 地图大小：20x20
- 静态障碍密度：0.20
- 地图种子：2000、2001、2002、2003、2004
- 起点：`(1, 1)`
- 终点：`(18, 18)`
- 动作：上、下、左、右、stay
- 最大步数：350
- 碰撞后保持原位置，episode 不终止
- episode 仅在到达目标或达到最大步数时结束

地图均可由四连通 A* 到达，具体网格哈希保存在 `maps.json` 和示范元数据中。

## 4. 观测结构

空间输入：

```text
(1, 11, 11)
```

唯一空间通道为以智能体为中心的局部静态障碍占据图：

- 自由位置：0
- 障碍位置：1
- 地图外填充：1
- 智能体固定在窗口中心，不单独设置位置通道

标量输入：

```text
(2,)
```

两个标量分别为归一化的目标相对行、列位移：

```text
delta_row = (goal_row - agent_row) / (map_size - 1)
delta_col = (goal_col - agent_col) / (map_size - 1)
```

取值范围为 `[-1, 1]`。CNN 编码局部障碍图后，将这两个标量与空间特征拼接，再输入 Dueling value 和 advantage 分支。

## 5. 网络与训练参数

- 算法：Dueling Double DQN
- 卷积编码器：`1 -> 32 -> 64 -> 64`
- Dueling 隐藏维度：256
- 学习率：0.0003
- 折扣因子：0.99
- target 同步间隔：250 次梯度更新
- 梯度裁剪：10.0
- batch size：64
- replay capacity：10000
- learning starts：64 环境步
- 每环境步更新次数：1
- 训练 episode：1500
- epsilon：1.0 线性衰减至 0.05
- epsilon 衰减长度：1200 episode
- epsilon 在同一 episode 内保持不变
- 训练种子：0
- 训练进度与 greedy 检查间隔：100 episode

## 6. 奖励函数

```text
step      = -0.01
progress  =  0.05 * (old_distance - new_distance)
stay      = -0.01
collision = -1.00
goal      = 10.00
```

距离使用当前位置到目标的 Manhattan distance。

## 7. A* 示范

- A* 示范 episode：20
- 示范 transition：720
- 每张地图分配 4 个示范 episode
- A* 使用随机最优 tie-breaking，保持最短路径长度
- 示范和在线训练使用相同环境奖励
- 数据格式版本：2
- 数据 SHA-256：`b2c4d2b791cd7fa1061252263a328b9de171af514c924052d591b2bc0896aced`

Replay 差异：

- `uniform` 不加载示范；
- `prefill` 将 720 条示范写入容量 10000 的普通 FIFO replay，允许被在线数据覆盖；
- `persistent_demo` 独立保留 720 条示范，每个 batch 固定采样 25% 示范和 75% 在线数据。

## 8. 正式结果

最终冻结 greedy 评估：

| 策略 | 成功地图 | 平均步数 | 路径效率 | 碰撞 | 重复访问 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 5/5 | 36 | 1.0 | 0 | 0 |
| Prefill | 5/5 | 36 | 1.0 | 0 | 0 |
| Persistent Demo | 5/5 | 36 | 1.0 | 0 | 0 |

训练过程成功率：

| Episode 范围 | Uniform | Prefill | Persistent Demo |
| --- | ---: | ---: | ---: |
| 1-100 | 1.0% | 0.0% | 0.0% |
| 101-200 | 3.0% | 10.0% | 45.0% |
| 201-300 | 15.0% | 16.0% | 92.0% |
| 301-500 | 81.0% | 89.5% | 100.0% |
| 501-1000 | 99.8% | 100.0% | 99.8% |
| 1001-1500 | 100.0% | 99.8% | 100.0% |

汇总：

| 策略 | 首次成功 episode | 成功 episode 总数 | 环境步 | 梯度更新 |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 81 | 1180/1500 | 195901 | 195838 |
| Prefill | 118 | 1204/1500 | 189469 | 189406 |
| Persistent Demo | 124 | 1336/1500 | 167867 | 167804 |

首次成功受 epsilon 随机探索影响较大，不作为主要结论。更稳定的证据是 Persistent Demo 在 episode 101-500 的分段成功率明显高于另外两种策略。

## 9. 基线结论

在这组固定静态地图和单一训练种子下：

1. 三种 replay 经过充分训练后都能生成无碰撞的 A* 等长最优路径；
2. Persistent Demo 的主要优势不是最终路径更短，而是训练中前期成功率更高、达到稳定策略所需环境步更少；
3. Prefill 相比 Uniform 有一定中期优势，但不如持续保留示范稳定；
4. 当前结果支持“持久 A* 示范改善局部观测下的样本效率”这一研究假设，但不能单独证明统计显著性。

## 10. 适用边界

本基线仍存在以下限制：

- 仅使用一个训练 seed；
- 仅使用五张20x20随机静态地图；
- 不包含动态障碍、历史帧或动态速度信息；
- 不包含办公室、快递站和仓库等结构化场景；
- 不评估跨地图泛化；
- Persistent Demo 使用固定 25% 示范比例，尚未实现自适应采样。

后续动态研究应使用独立配置和独立环境，不覆盖本基线文件与结果。

## 11. 复现指令

重新生成示范：

```powershell
python scripts/collect_astar_demos.py --config configs/static_local_baseline_v1.yaml
```

依次运行三种正式实验：

```powershell
$strategies=@('uniform','prefill','persistent_demo'); foreach($strategy in $strategies){ python scripts/train_random_benchmark.py --config configs/static_local_baseline_v1.yaml --strategy $strategy; if($LASTEXITCODE -ne 0){throw "training failed: $strategy"} }
```

运行测试：

```powershell
python -m unittest discover -s tests -q
```

冻结时共有 33 个单元测试通过。

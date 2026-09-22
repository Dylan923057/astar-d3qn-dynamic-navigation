# 三地图时空难度均衡基准 v4

## 为什么从 v3 升级

v3 按静态 A* 路线与动态障碍物的名义时序冲突数分层。这个指标不能完全表示智能体真正需要付出的避障代价：障碍物可能在智能体到达交叉点前已经离开，也可能造成更长的绕行。因此 v4 增加了时空安全路径 oracle，用于测量在障碍物按环境规则运动时，从起点到终点的最短无碰撞路径。

## Oracle 定义

每个时间步先推进动态障碍物，再检查智能体动作；智能体动作集合包含四个移动动作和等待动作。BFS 的状态为“智能体位置 + 障碍物运动周期相位”，同时禁止动作前后两端的动态障碍物位置，以匹配 `DynamicGridNavigationEnv` 的碰撞语义。输出包括：

- `minimum_safe_path_steps`：最短无碰撞步数；
- `safe_detour_steps = minimum_safe_path_steps - nominal_path_steps`；
- 无解场景会被标记为 unreachable，而不是静默纳入训练集。

## v4 冻结结果

| 地图 | generation seed | 高冲突 detour 均值（train / validation / test） |
|---|---:|---:|
| Office | 20260909 | 0.000 / 0.000 / 0.000 |
| Parcel station | 20260915 | 1.800 / 2.000 / 2.000 |
| Warehouse | 20260948 | 1.033 / 1.000 / 1.067 |

低、中冲突层的 detour 均值均为 0；三张地图的全部 510 个场景均可找到时空无碰撞路径。Parcel 的高冲突层仍有 0.2 步的 split 均值差异，这是当前路线池约束下的最优候选，不应宣称为完全相同难度。

## 生成与审计文件

- `src/astar_d3qn/evaluation/conflict.py`：时空安全路径 oracle；
- `scripts/audit_spatiotemporal_difficulty.py`：输出逐场景审计 CSV；
- `scripts/generate_balanced_spatial_benchmark_v4.py`：搜索并冻结路线池种子；
- `configs/*balanced_v4.yaml` 与 `data/dynamic_scenarios/*balanced_v4.json`：正式基准；
- `outputs/spatial_generalization_balanced_v4_design/scenario_spatiotemporal_difficulty.csv`：完整审计结果。

v4 只改变基准场景生成与难度审计，没有改变网络、奖励函数、动作策略或经验回放实现。已有 v3 训练结果应继续标记为 v3；冻结 v4 后再进行三地图、三策略、五 seed 的正式复现实验。

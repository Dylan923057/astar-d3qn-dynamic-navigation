# Office 行为验证场景 v6

## 为什么替换 v5

v5 训练集的动态安全最短路平均额外步数和最大额外步数均为 0，严格等待场景为 0/100。它虽然能阻断部分静态 A* 示范，但没有证明训练中的最优行为需要等待，因此不适合研究“静态 A* 示范与动态避障行为的冲突”。

## v6 的三类行为

- `wait`：允许原地等待时的最短安全路线，严格短于禁止等待时的最短安全路线，并且重建出的最优路线至少含一次等待。
- `avoidance`：等待没有收益；静态标称 A* 路线会碰撞；继续使用标称门组合仍是全局最优，但必须在门内或通道内局部偏离标称路线。
- `reroute`：等待没有收益，并且至少一组替代门组合严格短于标称门组合。

训练、验证、测试分别按 40/30/30、8/6/6、20/15/15 分配等待、局部避让和全局改道场景。所有判定均由与环境运动顺序一致的周期时间扩展最短路完成，不靠人工看图分类。

## 五个动态障碍物怎样才算有效

这里分成两个层次，不能混为一谈：

1. 每个障碍都必须在正确时间与 20 条实际静态 A* 示范中的至少一条发生碰撞。做不到这一点的候选不会进入场景清单。
2. 每个完整场景至少有两个障碍通过删除实验被证明具有决定作用：删除后，被阻断示范的并集减少，或四种上下门组合中至少一种的最短安全代价改变。

不能强制五个障碍在同一条最优轨迹上都产生决定作用。Office 的左右门互斥，智能体一次只能选择其中一扇；放在未选择门上的障碍仍在影响备选路线，但删除它未必改变当前最优轨迹。把“五个都能实际阻断示范”与“至少两个改变全局决策”分开，既能排除无关障碍，又不会设置一个与地图拓扑矛盾的条件。

## 集合间的位置关系

Office 只有四个真正的跨墙门洞。若强制训练、验证、测试连门洞格子都不能重复，就无法让三个集合都包含同一种等待行为，只能把障碍放回无关的开放区域。因此 v6 允许共享少量物理门洞，但使用不同的完整运动路线、初始相位和组合。

Office v6 检验的是未见路线组合和时序下的行为泛化。最终空间泛化应由不参与 Office 方法开发的 Parcel 和 Warehouse 地图检验，不能把这两个问题混为一谈。

## 文件

- 配置：`configs/dynamic_spatial_generalization_office_behavior_v6_terminal_v1.yaml`
- 生成器：`scripts/generate_office_behavior_scenarios_v6.py`
- 行为最短路：`src/astar_d3qn/evaluation/behavior_oracle.py`
- 清单：`data/dynamic_scenarios/office_40x40_behavior_v6.json`
- 图片：`maps/previews/office_40x40_behavior_v6_*.png`
- 审计：`outputs/office_behavior_v6_design/`

## 训练结果可视化

`train_dynamic_spatial_generalization.py` 现在会在每个策略-seed 目录中额外保存：

- `test_trajectories.json`：50 个冻结测试场景的完整路径、碰撞位置、等待事件和终止原因；
- `plots/training_curves.png`：该次运行的训练曲线；
- `plots/validation_curves.png`：该次运行在 20 个验证场景上的选模曲线。

三种策略各 5 个 seed 全部完成后，运行 `scripts/plot_office_behavior_v6_results.py`。它会先检查 15 次运行是否完整且是否使用同一个 frozen manifest，然后在实验输出根目录的 `analysis/` 下生成：

- `validation_learning_curves.png`：三策略验证曲线，横轴为统一的环境交互步数，阴影为 5 seed 的 95% 置信区间；
- `test_overall_metrics.png`：50 个最终测试场景的总体安全成功、碰撞、等待和路径效率；
- `test_behavior_metrics.png`：按等待、局部避让、全局改道分层的结果；
- `test_50_scenario_heatmaps.png`：每个策略在全部 50 个测试场景上的逐场热力图；
- `paths/seed_*/test_paths_page_*.png`：每个训练 seed 的全部 50 场路径，10 场一页、每场三策略并排，共 25 页，不挑选单一“好看”seed；
- 对应的逐 seed、逐行为、逐场 CSV 和 `path_gallery_index.csv`。

注意：20 个场景是 validation，用于选 checkpoint；50 个场景是只在训练结束后使用的 test。正式报告不能把 50 个 test 场景称为 validation。

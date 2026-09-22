# Astar-D3QN-Workshop 项目交接文档

更新时间：2026-09-01

工作区：`D:\Asatr-D3QN-Workshop`

本文件是新会话的当前事实来源。开始工作前应完整阅读本文件，并核对相关代码和输出。第 23 节以后的内容是 2026-09-01 的最新状态，若与前面“尚未完成”或“下一步”类表述冲突，以第 23 节以后为准。不要根据早期实验记录自行恢复已经否定的方案，也不要自行启动长时间正式训练；用户要求修改代码时可以直接修改，但修改完成后不要代替用户运行训练，应提供可直接粘贴到 PowerShell 的单行命令。

本文件编码为 UTF-8。Windows PowerShell 5 读取时应使用 `Get-Content HANDOFF.md -Encoding UTF8`，否则可能显示中文乱码。

关键补充文档：

- `docs/EXPERIMENT_INVENTORY.zh-CN.md`：截至 2026-09-01 的实验时间线、用途分级和结果目录总表；
- `docs/dynamic_spatial_v2_protocol.md`：动态空间泛化 v2 的正式协议；
- `docs/replay_baselines.md`：PER 和 DQfD-style 的实现边界；
- `docs/LITERATURE_REVIEW.zh-CN.md`：已有文献核对；
- `docs/RESEARCH_PROTOCOL.md`：replay 公平对照原则；
- `README.md`：当前项目入口和命令。

## 1. 当前研究主线

基础版本的论文表述：

> 提出不可淘汰的 A* 示范分区，通过受控实验揭示一次性 Prefill 的数据淘汰问题，并验证持续示范能否改善 D3QN 的样本效率与策略稳定性。

增强版本的论文表述：

> 在示范遗忘分析基础上，提出性能门控的自适应 A* 示范采样机制，在早期利用专家经验、后期增加在线经验，并在性能退化时恢复示范支持。

不要声称“首次将 A* 经验加入 D3QN”。应强调：

- 示范生命周期管理；
- 数据遗忘与策略遗忘的区分；
- 固定示范比例与自适应比例的比较；
- 严格控制变量的 replay 消融；
- 静态环境中的样本效率，以及分布变化后示范支持的作用和局限。

当前论文目标大致是 SCI 三、四区级别的小论文。现有单地图结果仍不足以支持完整论文结论，但已经形成可靠的静态校准证据。

## 2. 当前必须遵守的实验边界

主线实验中暂时不要修改：

- 地图、起终点和静态障碍；
- 奖励函数；
- epsilon 衰减方案；
- D3QN 网络结构；
- replay 总容量；
- batch size、学习率、target 更新周期等训练参数。

原因是当前要把性能变化归因于 replay 中示范经验的生命周期和采样方式，而不是其他同时变化的因素。

当前也不要：

- 重新生成全部地图；
- 立即扩展到大规模多地图训练；
- 加入转弯惩罚或路径平滑奖励；
- 为了制造遗忘而随意改变奖励或网络；
- 把一次短暂 greedy 失败直接称为灾难性遗忘；
- 把 `dqfd` 当前实现称为完整 DQfD；
- 把已有动态配置未经检查就当作公平的正式实验场景。

主线顺序已经确定：

1. 先冻结 40x40 静态 Map 1 的多 seed replay 基线；
2. 再在同一张地图上构造并冻结所有策略共享的动态障碍轨迹；
3. 先测试固定比例 `persistent_demo` 是否在动态分布变化下退化；
4. 只有观察到固定比例方法的真实局限后，才实现性能门控的自适应示范比例。

## 3. 三种主 replay 的精确定义

### Uniform

- 只存储在线训练产生的经验；
- 容量为 10000 的 FIFO uniform replay；
- 不加载 A* 示范。

### Prefill

- 训练开始前把同一份 A* 示范加入普通 FIFO replay；
- 后续在线经验和示范经验处在同一队列；
- 示范会随在线经验写入而被覆盖；
- 示范被覆盖后，训练过程不再能直接采样这些示范。

### Persistent Demo

- A* 示范存放在独立、不可淘汰的分区；
- 在线经验进入独立 FIFO 分区；
- replay 总容量仍固定为 10000，不允许把示范分区额外叠加到 10000 之外；
- 40x40 Map 1 有 1520 条示范，因此分区为 1520 条示范加 8480 条在线经验；
- `batch_size=64`、`demo_fraction=0.25` 时，每次更新固定抽取 16 条示范和 48 条在线经验，不是 5 条示范；
- 示范用于普通 TD 学习，不是直接做行为克隆标签训练。

三种方法必须共享地图、奖励、状态、网络、训练预算和评估流程。Prefill 与 Persistent Demo 必须加载完全相同的 A* 示范。

## 4. 当前静态实验配置

配置：`configs/structured_calibration_40x40.yaml`

训练入口：`scripts/train_random_benchmark.py`

当前正式校准对象：

- 地图：`calibration_40x40_map_01`；
- `--map-index 0`；
- 地图 seed：1500，所有训练 seed 共享同一地图；
- 起点：`(1, 1)`；
- 终点：`(39, 39)`；
- 环境：静态局部观测；
- observation window：15；
- 最大步数：600；
- action 数：5，包含 `stay`；
- 训练 episode：1500；
- replay capacity：10000；
- batch size：64；
- learning starts：64；
- updates per step：1；
- epsilon：1.0 线性衰减到 0.05；
- epsilon decay：1200 episode；
- learning rate：0.0003；
- gamma：0.99；
- target sync：250 次 gradient update；
- hidden dim：256；
- `progress_interval=100`；
- `diagnostic_interval=10`。

奖励保持不变：

```text
goal       = +10.0
collision  = -1.0
step       = -0.01
progress   = 0.05 * Manhattan distance improvement
stay       = -0.05 extra
```

20 条随机 tie-breaking A* 示范，每条最短路径为 76 个 transition，总计 1520 条。

## 5. 已完成的代码修改

### 示范生命周期和诊断

- `training.csv` 已增加：
  - `environment_steps_total`；
  - `gradient_updates_total`；
  - `replay_size`；
  - `demo_retained_count`；
  - `demo_retention_rate`。
- 新增 `diagnostics.csv`，每 10 episode 记录：
  - `demo_optimal_action_agreement`，当前 greedy 动作是否属于该状态的最优动作集合；
  - `demo_exact_action_agreement`，当前 greedy 动作是否与保存的具体示范动作一致。
- `--seed` 已支持独立训练随机种子。
- 每 100 episode 执行一次 frozen greedy 评估。
- 每 100 episode 保存周期 checkpoint 到 `checkpoints/`。
- 每 100 episode 保存 greedy 路径图到 `paths/periodic/`。
- 最佳和最终模型仍保存为 `model_best.pth` 和 `model_final.pth`。

重要语义：`diagnostics.csv` 的 `demo_transition_count` 是原始示范数据集的大小，不是 replay 当前保留的示范数量。分析 replay 中的数据淘汰必须使用 `training.csv` 的 `demo_retained_count` 和 `demo_retention_rate`。

### Persistent replay 容量修正

`src/astar_d3qn/training/trainer.py` 已修正 persistent replay 容量语义。40x40 Map 1 现在是 1520 demo + 8480 online = 10000 total，而不是 1520 + 10000。

### PER 和 DQfD-style 基线

- PER：`src/astar_d3qn/replay/prioritized.py`，CLI 使用 `--strategy per`。
- DQfD-style：CLI 使用 `--strategy dqfd`。
- 文档：`docs/replay_baselines.md`。

当前 `dqfd` 仅包含：

- persistent demonstration partition；
- one-step Double-DQN TD loss；
- demonstration large-margin loss。

当前没有完整实现原始 DQfD 的：

- n-step return；
- demo-only pretraining；
- 完整 loss schedule；
- 原论文完整的 priority handling。

因此论文和图表中只能称为 `DQfD-style`，除非后续补齐并验证完整机制。PER 和 DQfD-style smoke 已通过，但尚未进入正式多 seed 对比。

### 路径可视化和转折指标

用户之前要求每 100 episode 能直观看到 greedy 路径，功能已经完成。

评估代码也被动记录：

- `movement_steps`；
- `turn_count`；
- `turn_rate`；
- 对应 aggregate 均值。

这些字段只进入评估 CSV 和日志，没有进入奖励、状态、网络、动作选择或 replay 采样。因此没有污染主线控制变量。

当前已经决定回到 replay 主线。路径平滑和路径优化暂缓，不增加转弯惩罚。若未来增加转弯惩罚，必须把上一动作等必要信息加入状态以维护 Markov 性，并作为独立扩展/消融重新训练，不能混入当前主实验。

### 静态/动态碰撞分类指标

2026-08-15 已增加统一的 `CollisionTracker`，位置为 `src/astar_d3qn/core/collisions.py`。它只扩展记录，不改变碰撞判定、奖励或训练行为。

从下一次训练开始，`training.csv`、frozen greedy 评估明细和 held-out 明细会增加：

- `scenario_id`；
- `dynamic_obstacle_count`；
- `static_collision`、`static_collision_count`；
- `dynamic_collision`、`dynamic_collision_count`；
- `first_static_collision_step`、`first_dynamic_collision_step`；
- `dynamic_obstacle_contact_count`；
- `unique_dynamic_obstacles_hit`；
- `dynamic_obstacle_indices_hit`；
- `safe_success`；
- `dynamic_collision_free_success`。

评估 summary 和周期 checkpoint 指标会增加静态/动态碰撞率、平均次数、无碰撞成功率、首次碰撞步、平均路径效率、等待事件和 reward 等汇总。`scenario_id` 用来把碰撞结果关联回动态场景 manifest。

`DynamicGridNavigationEnv` 运行时原本已经返回 `collision_type` 和 `dynamic_collision_indices`，但旧训练器只累计总碰撞。现在训练和评估会保存分类结果。旧动态输出无法仅凭已有 CSV 可靠恢复静态/动态碰撞类型，必须使用新代码重新评估或重跑。

## 6. 当前验证状态

最近一次验证：

```powershell
python -m unittest discover -s tests -q
```

结果：75 项测试全部通过。

当前 Python 环境没有安装 `pytest`，所以 `python -m pytest -q` 会失败；使用 `unittest discover`。

Uniform、Prefill、Persistent Demo smoke 均已通过。PER 和 DQfD-style smoke 也已通过。

## 7. 最新 5-seed 静态正式实验

运行命令：

```powershell
$seeds=0..4; $strategies=@('uniform','prefill','persistent_demo'); foreach($seed in $seeds){ foreach($strategy in $strategies){ python scripts/train_random_benchmark.py --config configs/structured_calibration_40x40.yaml --map-index 0 --strategy $strategy --seed $seed; if($LASTEXITCODE -ne 0){ throw "training failed: seed=$seed strategy=$strategy" } } }
```

输出根目录：`outputs/structured_calibration_40x40/`

本次正式 cohort 的 15 个目录：

```text
20260814_203024_map_01_seed_1500_trainseed_0_uniform
20260814_210818_map_01_seed_1500_trainseed_0_prefill
20260814_214557_map_01_seed_1500_trainseed_0_persistent_demo
20260814_221918_map_01_seed_1500_trainseed_1_uniform
20260814_224947_map_01_seed_1500_trainseed_1_prefill
20260814_232856_map_01_seed_1500_trainseed_1_persistent_demo
20260815_000155_map_01_seed_1500_trainseed_2_uniform
20260815_004039_map_01_seed_1500_trainseed_2_prefill
20260815_012053_map_01_seed_1500_trainseed_2_persistent_demo
20260815_015416_map_01_seed_1500_trainseed_3_uniform
20260815_022855_map_01_seed_1500_trainseed_3_prefill
20260815_030959_map_01_seed_1500_trainseed_3_persistent_demo
20260815_034330_map_01_seed_1500_trainseed_4_uniform
20260815_041726_map_01_seed_1500_trainseed_4_prefill
20260815_050126_map_01_seed_1500_trainseed_4_persistent_demo
```

这 15 次都完整运行到 episode 1500。以下两个目录只有 4 个 episode，是中断 smoke/临时运行，严禁纳入统计：

```text
20260814_201453_map_01_seed_1500_trainseed_0_persistent_demo
20260814_201531_map_01_seed_1500_trainseed_0_persistent_demo
```

更早的单 seed 完整输出属于旧 cohort，不要与本次 15 次结果混合统计。本次分析只使用上面明确列出的目录。

## 8. 5-seed 逐次结果

“首次 greedy 成功”定义为 `progress_evaluation.csv` 中第一次 `success_rate=1` 的周期评估，不是训练期 epsilon-greedy episode 的偶然成功。

| Strategy        | Seed | First greedy success ep | Env steps at first success | Final env steps | Final greedy steps | Best checkpoint ep |
| --------------- | ---: | ----------------------: | -------------------------: | --------------: | -----------------: | -----------------: |
| Uniform         |    0 |                     800 |                     387620 |          457250 |                 76 |                800 |
| Uniform         |    1 |                     500 |                     257262 |          369747 |                 76 |                500 |
| Uniform         |    2 |                     100 |                      60000 |          458593 |                 76 |                100 |
| Uniform         |    3 |                     600 |                     321249 |          414112 |                 76 |                600 |
| Uniform         |    4 |                     600 |                     306961 |          399405 |                 76 |                600 |
| Prefill         |    0 |                     600 |                     295212 |          388918 |                 76 |                600 |
| Prefill         |    1 |                     600 |                     314580 |          407607 |                 76 |                600 |
| Prefill         |    2 |                     600 |                     317622 |          415052 |                 76 |                600 |
| Prefill         |    3 |                     700 |                     347970 |          425325 |                 76 |                700 |
| Prefill         |    4 |                     700 |                     374376 |          459158 |                 78 |                700 |
| Persistent Demo |    0 |                     200 |                     118655 |          333293 |                 76 |                200 |
| Persistent Demo |    1 |                     200 |                     118227 |          334206 |                 76 |                200 |
| Persistent Demo |    2 |                     200 |                     118285 |          334150 |                 78 |                200 |
| Persistent Demo |    3 |                     100 |                      60000 |          336950 |                 76 |                100 |
| Persistent Demo |    4 |                     200 |                     118539 |          333880 |                 76 |                300 |

Aggregate：

| Strategy        | First success ep mean +/- SD | Median ep | Env steps to first success mean +/- SD | Median env steps | Final total env steps mean +/- SD |
| --------------- | ---------------------------: | --------: | -------------------------------------: | ---------------: | --------------------------------: |
| Uniform         |                520 +/- 258.8 |       600 |                      266618 +/- 124531 |           306961 |                  419821 +/- 38279 |
| Prefill         |                 640 +/- 54.8 |       600 |                       329952 +/- 31198 |           317622 |                  419212 +/- 25994 |
| Persistent Demo |                 180 +/- 44.7 |       200 |                       106741 +/- 26130 |           118285 |                   334496 +/- 1419 |

所有 15 次运行最终 frozen greedy 都成功。Uniform 的最终路径均为 76 步；Prefill 和 Persistent Demo 各有一个 seed 为 78 步，其余为 76 步。

当前可支持的结论：

> 在固定的 40x40 静态 Map 1 上，持续示范 replay 显著改善了 D3QN 的早期样本效率，并降低了达到稳定表现所需的环境交互量。该结论目前是单地图、5 个训练 seed 的证据，不能外推为跨地图普遍结论。

Uniform 的方差明显更大，尤其 seed 2 在 episode 100 偶然形成成功 greedy 策略，导致均值被明显拉低。因此同时报告均值、标准差和中位数，不要只报告均值。

## 9. 数据遗忘结果

训练过程中的 replay 保留情况在 5 个 seed 上完全一致：

### Prefill

- episode 10：replay size 7520，1520 条示范仍全部保留，retention=1.0；
- episode 20：replay size 10000，示范已经全部被在线经验覆盖，retention=0.0；
- episode 20 至 1500：retention 始终为 0.0。

更细的早期旧检查曾显示示范大约在 episode 14 至 17 之间快速从 1520 降到 0；正式 cohort 的 10-episode 快照明确确认 episode 10 到 20 之间完成淘汰。

### Persistent Demo

- episode 10 至 1500：始终保留全部 1520 条示范；
- replay 总大小最终为 10000；
- retention 始终为 1.0。

因此已经可靠观察到 Prefill 的“数据遗忘/数据淘汰”，但这不自动等于神经网络策略已经遗忘示范知识。

## 10. 策略一致率和策略遗忘结果

episode 1500 时：

- Prefill 的最优动作集合一致率约为 0.9958，精确示范动作一致率均值为 0.6566；
- Persistent Demo 的最优动作集合一致率约为 0.9982，精确示范动作一致率均值为 0.6553。

精确动作一致率低于最优动作集合一致率是合理的，因为同一状态可能有多个等价最优动作，而随机 tie-breaking A* 只保存其中一个具体动作。

Prefill 在示范样本已经从 replay 完全消失后，最终最优动作集合一致率仍接近 1。这说明当前静态、确定性环境中观察到的是：

> 明确的数据遗忘，但没有对应的持续性策略遗忘。网络可能已经把示范引导形成的价值结构保存在参数中，之后也能通过在线经验继续维持。

因此不能用这组静态结果声称 Prefill 已发生灾难性策略遗忘。

## 11. 阶段性 greedy 退化

每 100 episode 的 frozen greedy 评估中，三种策略各有 1 个 seed 出现过 1 -> 0 的短暂回退，之后均恢复：

- Uniform seed 2：episode 100 成功，episode 200 至 600 失败，episode 700 起恢复；
- Prefill seed 4：episode 700 起成功，episode 1300 短暂失败，episode 1400 恢复；
- Persistent Demo seed 3：episode 100 至 200 成功，episode 300 至 400 失败，episode 500 恢复。

其余 12 次运行一旦首次成功，后续周期 greedy 评估都保持成功。

结论：当前只存在阶段性策略波动，没有持续性灾难性遗忘，而且这种波动并非 Prefill 独有。

Uniform 也会退化的原因包括：

- DQN 的函数逼近会让一次更新同时改变许多状态的 Q 值；
- bootstrapping target 随网络变化；
- replay 数据分布持续变化；
- 某条刚学到的决策边界可能被后续更新暂时破坏。

Prefill 的后期回退不能主要归因于“epsilon 仍然很高”。例如 episode 1300/1400 时 epsilon 已经接近下限 0.05。训练时的探索行为会影响 replay 分布，但 frozen greedy 回退本质上仍是学习网络的价值估计发生变化。

Persistent Demo 持续抽取 A* 经验通常能稳定关键状态的价值排序，但不能数学上保证完全不回退。seed 3 已证明它也可能发生短暂波动。

## 12. 路径平滑问题的当前结论

A* nominal path：76 moves、17 turns。

此前单 seed 检查发现 Persistent Demo 的某些早期 checkpoint 虽然成功很快，但转折较多。原因不是实现错误：

- 当前奖励只优化到达、步数、进度、碰撞和 stay，没有转弯成本；
- 所有 76 步最短路径在当前奖励下基本等价；
- A* 示范使用随机等价 tie-breaking；
- 20 条示范本身有 20 至 40 次转弯，均值约 27.45；
- D3QN 学的是高回报可达策略，不会自动选择最少转弯路径。

最新 5-seed 最终转折均值仅作为被动记录：Uniform 24.8、Prefill 24.8、Persistent Demo 19.2。不要据此启动路径优化，也不要把它作为当前论文主贡献。当前先完成 replay 生命周期和动态适应主线。

## 13. 动态障碍假设的严谨表述

不能直接断言 Persistent Demo 在动态障碍下必然比 Uniform 更容易退化。

合理假设是：

- 静态 A* 示范会强化 nominal path 上的动作价值；
- 当动态障碍临时阻断 nominal path 时，过高且固定的示范采样比例可能与新的在线避障经验竞争；
- Uniform 没有示范约束，可能更自由地适应新分布，但早期学习效率和稳定性可能更差；
- Persistent Demo 也可能因为示范提供全局前进结构而更稳定，并不一定退化。

动态环境下的碰撞 transition 有负奖励，动态 observation 也包含历史障碍通道，因此固定示范不一定阻止网络学习避障。是否冲突必须通过受控实验验证，不能预先当作结论。

“灾难性遗忘”至少需要：

- 先建立明确、稳定的旧任务能力；
- 分布变化后旧能力显著下降；
- 下降持续多个评估周期，而非单点失败；
- 最好同时报告旧任务保持率和新任务适应率。

如果只在动态环境中从头训练，更准确的术语可能是“示范偏置导致的适应受限”或“固定示范比例下的策略干扰”，而不是严格的连续学习灾难性遗忘。新会话需要继续注意术语边界。

## 14. 自适应示范比例的预期设计

增强版本尚未实现。预期机制是性能门控的示范采样比例：

- 训练早期使用较高示范比例，提升样本效率；
- 性能稳定后逐步降低示范比例，增加在线经验占比；
- 当 frozen greedy 成功率、碰撞率或路径效率明显恶化时，暂时恢复示范支持；
- 使用平滑窗口、迟滞和上下界，避免比例频繁振荡。

正式比较至少应包含：

- Uniform；
- Prefill；
- fixed Persistent Demo，例如 0.25；
- Adaptive Persistent Demo；
- PER；
- DQfD-style 或后续完整 DQfD。

但不要现在实现 adaptive。先测试 fixed persistent 在冻结的动态场景中是否存在可解释的局限。

如果 `demo_fraction=0.25` 在动态场景中仍然没有退化，可以设计预先说明的压力消融，例如固定比例 0.05、0.10、0.25、0.50，研究固定比例的稳定性和适应性权衡。不要通过任意修改奖励或网络来人为制造结果。

## 15. 动态实验的下一步

下一项正式工作不是换更复杂静态地图，也不是改路径平滑，而是审计并冻结 40x40 Map 1 的共享动态障碍场景。

现有相关配置：`configs/dynamic_calibration_40x40_map01.yaml`

现有动态代码可能包含基于不同策略历史 final path 选取障碍交互位置的逻辑。虽然三个策略可能最终使用同一个组合场景，但在正式训练前必须再次确认：

- 三种 replay 使用完全相同的障碍数量、路线、初始相位、速度和随机种子；
- 障碍轨迹不根据当前训练策略变化；
- 每个 seed/strategy 的训练和 frozen evaluation 都能复现同一时序；
- 不存在策略专属障碍布置；
- A* 示范文件、动态 observation 和奖励语义一致；
- 动态示范如何生成和是否包含碰撞/等待必须明确记录。

完成审计和 smoke 后，先运行 fixed replay 动态实验，不要直接运行 adaptive。

建议动态阶段仍从 Map 1、5 个训练 seed 开始。需要的核心指标：

- frozen greedy success rate；
- collision rate/count；
- wait steps/event count；
- path length/efficiency；
- 首次适应成功所需 episode 和 environment steps；
- 静态旧任务能力保持率；
- 示范动作一致率；
- 发生退化后的恢复时间。

当前没有提供正式动态训练命令，因为必须先完成共享障碍轨迹审计。

## 16. 论文层面的当前证据与缺口

已经拥有：

- 受控的 Uniform / Prefill / Persistent Demo 实现；
- replay 中示范保留率的逐 episode 记录；
- 最优动作集合一致率和精确示范动作一致率；
- 每 100 episode frozen greedy checkpoint 和路径图；
- 40x40 静态 Map 1 的 5-seed 正式结果；
- 明确的数据淘汰证据；
- Persistent Demo 的静态样本效率优势；
- PER 和 DQfD-style 的可运行代码。

仍然缺少：

- 多张固定地图上的复现；
- 公平冻结的动态障碍实验；
- 固定示范比例在动态变化中的局限证据；
- adaptive 比例机制和对应消融；
- PER/DQfD-style 正式多 seed 对照；
- 统计检验、置信区间和完整论文图表；
- 若要直接声称 DQfD 对比，则需要完整复现 DQfD 或明确限定为 DQfD-style。

静态结果当前最稳妥的论文表述：

> 在固定的 40x40 静态地图上，持续保留并固定比例采样 A* 示范显著提高了 D3QN 的早期样本效率。一次性 Prefill 的示范在 replay 填满后被快速淘汰，但策略层面的最优动作一致率仍可恢复并保持较高，说明 replay 数据遗忘与策略遗忘需要分别测量。

## 17. 动态泛化脚本的当前状态

新增入口：`scripts/train_dynamic_generalization.py`

配置：`configs/dynamic_generalization_map01.yaml`

该实验仍使用静态基线中的 40x40 `calibration_40x40_map_01`，但每个 episode 叠加 3 个动态障碍。训练场景 seed 为 0 至 99，held-out seed 为 100 至 199。训练每个 episode 切换到下一个场景，100 个场景循环使用，1500 episode 共循环 15 次。

当前生成器只找到 5 条长度为 5 的候选障碍轨迹。每个场景从中选择 3 条互不重叠的轨迹，并随机设置初始格和方向。因此 100 个训练场景只有 10 种空间路线组合；训练和 held-out 都覆盖相同的 10 种组合，主要差异是运动相位和方向。

所以该脚本验证的是 seen-location / unseen-phase 泛化，不是 unseen-location 空间泛化。当前结果不能声称对全新动态障碍位置的泛化。

已经完成 seed 0、1 的三策略运行。初步结果：

- Persistent Demo 在训练场景上稳定达到 100% 成功的 episode 分别为 100、300；
- Uniform 分别为 500、800；
- Prefill 分别为 700、700；
- 六次运行最终 held-out success 都为 100%；
- held-out 路径约 76 步，碰撞接近 0。

这表明当前相位泛化任务可能偏简单，初步支持 Persistent Demo 的样本效率，但不支持固定示范导致动态灾难性遗忘。旧输出中的 collision 是静态/动态混合统计；分类碰撞指标加入后，后续运行才会输出独立类型。

该脚本目前不生成路径 PNG，因为它评估多个动态场景且尚未选择代表性场景，也没有调用 `render_final_path`。正式论文可视化前需要固定代表性 held-out scenario，并绘制智能体路径和动态障碍轨迹。

## 18. 受控动态瓶颈实验（已实现并验证）

正式配置：`configs/dynamic_controlled_bottleneck_40x40_map01.yaml`

训练入口：`scripts/train_random_benchmark.py`

输出根目录：`outputs/dynamic_controlled_bottleneck_40x40_map01/`

实验继续使用 40x40 `calibration_40x40_map_01`，且 Uniform、Prefill、Persistent Demo 共享完全相同的 3 个确定性动态障碍。路线、初始相位、方向和速度均不依赖策略或训练 seed：

- `mandatory_bottleneck`：路线 `(6,5)` 到 `(6,9)`；其中 `(6,5)` 是删除后会使起点与终点不连通的图割点，也是所有可行路径的必经格。静态 nominal A* 在第 9 个动作直接进入该格会动态碰撞，提前等待一步可以安全通过；
- `nominal_crossing_middle`：横穿 nominal path 的 `(22,5)` 到 `(22,9)`，参考路径索引 27，相位对齐到考虑第一次等待后的环境步 28；
- `nominal_crossing_late`：横穿 nominal path 的 `(35,8)` 到 `(35,12)`，参考路径索引 43，相位对齐到考虑前两次等待后的环境步 45；
- 三条路线互不重叠，障碍均每一步移动并在路线端点反弹；
- 环境不在碰撞时终止，最大 600 步，网络、奖励、epsilon、replay 容量等训练设置保持不变。

示范文件：`data/demonstrations/dynamic_controlled_bottleneck_40x40_map01/map_01_seed_1500_astar_demos.npz`

示范语义必须表述为：20 条静态 nominal A* 路径，共 1520 个 transition；输入仍为动态网络需要的 4 个空间通道，但 3 个动态历史通道全部补零。metadata 的环境标识是 `static_nominal_zero_dynamic_channels_v1`。它不是会避让动态障碍的动态专家，这正是用来受控检验“旧静态示范与新动态在线经验是否冲突”的设置。Prefill 和 Persistent Demo 加载同一份示范，Uniform 不使用示范。

正式训练会从头训练三种 replay，不加载 `train_dynamic_generalization.py` 或静态实验的模型权重。它能比较：

- 静态 A* 示范是否继续提高动态任务的早期样本效率；
- Persistent Demo 是否更快学会在必经瓶颈前等待；
- 固定 `demo_fraction=0.25` 是否增加动态碰撞，或限制在线动态经验的适应；
- Prefill 示范淘汰、Persistent 示范保留与策略表现之间的关系。

若只是在动态环境中从头训练，出现的性能差异应称为“固定示范比例下的策略干扰/适应受限”，不能直接称为灾难性遗忘。严格的灾难性遗忘还需要先建立旧静态能力，再在分布变化后持续评估旧任务保持率。

已完成验证：77 项单元测试通过，`compileall` 通过，3 个 replay 的 smoke 通过。Smoke 输出确认包含分类型碰撞字段、`scenario_id=map01_controlled_bottleneck`、周期 checkpoint、周期 greedy 路径图和 best/final 路径图。

正式 5-seed、3-strategy PowerShell 单行命令：

```powershell
$seeds=0..4; $strategies=@('uniform','prefill','persistent_demo'); foreach($seed in $seeds){ foreach($strategy in $strategies){ python scripts/train_random_benchmark.py --config configs/dynamic_controlled_bottleneck_40x40_map01.yaml --map-index 0 --strategy $strategy --seed $seed; if($LASTEXITCODE -ne 0){ throw "training failed: seed=$seed strategy=$strategy" } } }
```

## 19. 受控动态实验结束后的决策

先只分析上述 15 次运行，不立即修改奖励、epsilon、网络架构或 replay 容量。主要读取各运行目录的 `training.csv`、`progress_evaluation.csv`、`diagnostics.csv`、`checkpoints/` 和 `paths/periodic/`。

比较每个策略的首次连续 greedy 成功 episode/environment steps、动态碰撞率和次数、无动态碰撞成功率、首次动态碰撞步、等待步/等待事件、成功路径效率、示范保留率和示范动作一致率，并报告 5-seed 均值、标准差、中位数和单 seed 轨迹。

- 若 Persistent Demo 更快且最终同样安全：结论仍是持续示范提高样本效率，本场景没有证实固定比例的动态适应代价；
- 若 Persistent Demo 早期更快但后期动态碰撞更多或成功率更低：再做固定比例 `0.05/0.10/0.25/0.50` 压力测试，然后据此设计自适应比例；
- 若三策略都学不会：先审计任务可学习性和观察/碰撞时序，不通过任意改奖励来制造结果；
- 若要研究严格的灾难性遗忘：另建“静态预训练 -> 动态适应 -> 同时评估静态保持与动态适应”的连续学习协议，不把当前从头训练实验冒充为遗忘实验。

## 20. 六障碍高密度受控条件

新增配置：`configs/dynamic_controlled_six_obstacles_40x40_map01.yaml`

输出根目录：`outputs/dynamic_controlled_six_obstacles_40x40_map01/`

上一次动态泛化实验实际是从 5 条候选 crossing 路线中每个 episode 抽取 3 条，并不是同时放置 5 个障碍。当前三障碍受控场景已经包含其中 2 条，所以不能直接相加为 8 个。六障碍版本使用：

- 1 条不可绕过的 `(6,5)` 必经瓶颈路线；
- 上一次生成器找到的全部 5 条 crossing 路线；
- A* 路径交互索引为 `9/13/27/28/36/43`；
- 计划交互环境步为 `9/14/29/31/40/48`；
- 每个交互点均具有统一语义：盲目进入会碰撞，提前等待 1 步后可以安全进入；
- 已用确定性 oracle 验证：静态 A* 的 76 个移动动作加 6 次等待，共 82 步、零碰撞到达终点。

六条路线、速度、方向和初始相位对所有 replay 和训练 seed 固定。该配置复用相同的 1520 条静态 nominal A* 示范，奖励、网络、epsilon、replay 容量及其他训练参数不变。77 项测试、编译、动画渲染和三策略 smoke 均已通过。

预览动画：`maps/previews/calibration_40x40_map_01_map01_controlled_six_obstacles.gif`

应先进行三策略单 seed 可学习性试验。若至少有策略能在 1500 episode 内稳定学会，再扩展为 5 seed；若三者都失败，不应把失败解释为 replay 差异。

## 21. 混合路径/空旷区域动态场景

用户要求避免把所有动态障碍都放在 A* 路径上，新增配置：`configs/dynamic_controlled_mixed_six_obstacles_40x40_map01.yaml`

该配置包含 6 个障碍：原受控场景的 3 个路径交互障碍，加上 3 个不与 nominal A* 路径相交的背景障碍：

- `(11,19)` 到 `(15,19)`，纵向往返；
- `(24,32)` 到 `(24,36)`，横向往返；
- `(35,19)` 到 `(39,19)`，纵向往返。前两条是 off-path 空旷区域路线；最后一条按用户指定坐标保留，但会经过 nominal path 的 `(38,19)`，因此实际场景是 4 条路径交互路线 + 2 条 off-path 路线。

两条 off-path 路线均通过静态障碍校验、与 nominal path 无交集、彼此及路径障碍互不重叠。它们不会直接阻断 A* 路径，但会增加动态观测背景，并可能影响策略绕行时的局部感知；第三条按用户指定坐标作为末段 nominal crossing 路线处理。

动画预览：`maps/previews/calibration_40x40_map_01_map01_controlled_mixed_six_obstacles.gif`

78 项测试、编译和三策略 smoke 已通过。混合配置是当前推荐的下一步实验；纯路径六障碍配置 `dynamic_controlled_six_obstacles_40x40_map01.yaml` 仍保留用于高密度对照。

## 22. 碰撞后成功与可视化指标

碰撞不终止 episode 的规则保持不变，`success=1` 仍只表示最终到达终点。为避免把“撞过但最后到达”误认为安全成功，训练和 greedy 验证现在额外输出：

- `safe_success` / `safe_success_rate`：到达终点且全程零碰撞；
- `collision_success` / `collision_success_rate`：到达终点但至少发生一次碰撞；
- `static_collision_success`、`dynamic_collision_success`：碰撞后成功按碰撞类型细分；
- `static_collision`、`dynamic_collision` 及对应次数/首次碰撞步：静态和动态碰撞始终分开统计。

每个运行目录现在包含：

- `training.csv`：逐 episode 探索训练指标；
- `progress_evaluation.csv`：每个周期 frozen greedy 验证指标；
- `training_map_evaluation.csv`、`best_map_evaluation.csv`：最终/最佳 greedy 的逐地图明细；
- `plots/training_curves.png`：训练成功率、安全成功率、碰撞后成功率、静态/动态碰撞次数、奖励和步数；
- `plots/greedy_evaluation_curves.png`：周期 greedy 的总成功率、无碰撞成功率、碰撞后成功率、静态/动态碰撞率和平均步数；
- `paths/periodic/`、`paths/*_best_path.png`、`paths/*_final_path.png`：路径、等待点、碰撞点和动态路线图。

正式结果分析应至少同时报告 `success_rate`、`safe_success_rate`、`collision_success_rate`、`static_collision_rate` 和 `dynamic_collision_rate`，不能只报告 `success`。

## 23. 2026-09-01 最新状态摘要

前面第 1 至 22 节保留了项目从静态校准到早期动态场景的历史。当前已经完成更严格的动态空间泛化 v2 实验，旧章节中“还要先审计动态场景”“先跑单 seed”等下一步已经执行完毕，不应重复作为当前任务。

当前最重要的事实是：

- Map 1 已完成三策略、5 seed 的 checkpoint v2 正式对照，以及固定 25%、线性衰减 25% 到 0、固定 10% 三种示范比例的消融；
- Map 2 已完成三策略、5 seed 的独立确认实验；
- 所有正式方法在同一地图内共享冻结的场景 manifest，训练、验证和测试空间路线池互不重叠；
- checkpoint 只依据 validation 指标选择，test 只在选好模型后运行一次，避免用测试集调参；
- Map 1 和 Map 2 的策略排名不完全一致，因此不能写成“Persistent Demo 在所有地图上普遍最好”；
- 当前更有研究价值的主线不是继续堆固定比例实验，而是研究“静态 A* 示范与动态避障任务之间的冲突”，并设计冲突感知的自适应示范回放。

完整实验清单和哪些数据能使用，以 `docs/EXPERIMENT_INVENTORY.zh-CN.md` 为准。

## 24. 动态空间泛化 v2 协议与场景含义

### 数据划分

每张地图的正式 manifest 包含：

- 100 个训练场景；
- 20 个验证场景，每 100 个 episode 用于 checkpoint 评估和选择；
- 50 个测试场景，仅在训练结束且 checkpoint 已由验证集选定后评估。

这可以口语化理解为 170 个“动态障碍组合场景”，每个场景里有 5 个动态障碍，所以总共有 850 个障碍实例；但不能说成 850 个独立场景。不同场景会复用本 split 的候选路线，只是路线组合、初始相位和运动方向等设置不同。

每个场景由 3 条 `corridor` 路线和 2 条 `background` 路线组成。两类路线的区别不是运行速度：

- `corridor` 表示更可能与智能体通行区域或 nominal A* 路径产生交互的路线；
- `background` 表示主要位于空旷或非 nominal 通行区域、用于增加动态观测背景和绕行风险的路线。

Map 1 当前路线池数量为：train 25 条 corridor、7 条 background；validation 10 条 corridor、4 条 background；test 15 条 corridor、6 条 background。用户确认过的 validation 修改已经写入 manifest：中心约为 `(14,26)` 的路线改为纵向，原中心约为 `(37,30)` 的路线移动到约 `(11,13)` 并改为纵向。

训练 seed 主要改变网络初始化、epsilon 探索动作、replay 抽样及优化随机性。正式 v2 中障碍物位置由冻结 manifest 决定，相同地图、相同 split 的三种策略共享同一组场景；seed 不应偷偷生成策略专属的障碍位置。

Map 1 正式 manifest 的 SHA-256 前 12 位为 `790e4f31b58e`。Map 2 正式 manifest 为 `data/dynamic_scenarios/map02_spatial_generalization_confirmatory_v2.json`，SHA-256 前 12 位为 `e800f6c8a9fa`。

### Checkpoint 选择规则

按以下顺序从 validation checkpoint 中选模型：

1. 最大化 `safe_success_rate`；
2. 若相同，最大化 `success_rate`；
3. 再最小化 `mean_dynamic_collision_count`；
4. 再最小化 `mean_steps`。

每次正式运行保存 `model_final.pth`、`model_best_validation.pth` 和 `model_selected.pth`。测试必须加载 `model_selected.pth`，不能看过 test 后换 checkpoint。

主要入口和配置：

- `scripts/train_dynamic_spatial_generalization.py`；
- `configs/dynamic_spatial_generalization_map01_checkpoint_v2.yaml`；
- `configs/dynamic_spatial_generalization_map01_demo_decay_v2.yaml`；
- `configs/dynamic_spatial_generalization_map01_demo10_v2.yaml`；
- `configs/dynamic_spatial_generalization_map02_confirmatory_v2.yaml`；
- `scripts/render_spatial_replay_comparison.py`；
- `scripts/render_static_retention_comparison.py`。

Map 2 的 decay 和 demo10 配置文件虽然存在，但尚未形成正式结果，不能当作已经做完的数据。

## 25. Map 1 正式结果

### 三策略 checkpoint v2

结果根目录：`outputs/dynamic_spatial_generalization_map01_checkpoint_v2/`。共 15 次完整运行，即 3 个策略乘 5 个训练 seed，每次有 15 个 validation checkpoint。

| 指标                                           |       Uniform |        Prefill | Persistent 25% |
| ---------------------------------------------- | ------------: | -------------: | -------------: |
| 训练 MA50 success 首次达到 80%（episode 均值） |        约 420 |         约 384 |         约 266 |
| 全 checkpoint validation safe-success 均值     |         66.4% |          69.3% |          89.8% |
| 最终 validation safe-success                   |  91% ± 10.8% |    91% ± 8.9% |    96% ± 4.2% |
| validation 选模后的 test safe-success          | 83.6% ± 7.3% | 84.8% ± 11.7% |  76.4% ± 6.2% |

严谨结论：Persistent 25% 在 Map 1 上明显加快训练并提高验证曲线，但所选模型在空间隔离的 test 路线池上并不是最好。它表现出“样本效率/验证稳定性收益”和“跨路线测试泛化可能受限”之间的权衡。

### 示范比例消融

结果目录：

- 固定 25%：`outputs/dynamic_spatial_generalization_map01_checkpoint_v2/` 中 Persistent Demo；
- 25% 到 0 的线性衰减：`outputs/dynamic_spatial_generalization_map01_demo_decay_v2/`；
- 固定 10%：`outputs/dynamic_spatial_generalization_map01_demo10_v2/`。

| Persistent 方案       | 最终 validation safe-success | 15 个 checkpoint 中完美 checkpoint 数 | 最佳 checkpoint episode 均值 | Map 1 test safe-success |
| --------------------- | ---------------------------: | ------------------------------------: | ---------------------------: | ----------------------: |
| 固定 25%              |                  96% ± 4.2% |                                   7.0 |                          680 |                   76.4% |
| 25%→0（ep 200–600） |                  88% ± 5.7% |                                   4.2 |                          960 |                   74.0% |
| 固定 10%              |                  93% ± 8.4% |                                   6.2 |                          860 |                   84.0% |

预先规定的 validation-only 规则会选择固定 25%。固定 10% 在 Map 1 test 上更好是值得解释的探索性信号，提示较高静态示范比例可能形成最短路径偏置；不能因为已经看过 test 就倒过来把固定 10% 宣布为正式赢家。

## 26. Map 2 独立确认结果

结果根目录：`outputs/dynamic_spatial_generalization_map02_confirmatory_v2/`。共 15 次完整运行，3 个策略乘 5 seed；每个策略共有 250 个 test rollout（5 seed × 50 场景）。

| 策略           | Success |            Safe success | 动态接触总数 | 静态碰撞 | Static retention |
| -------------- | ------: | ----------------------: | -----------: | -------: | ---------------: |
| Uniform        | 250/250 | 234/250 = 93.6% ± 7.8% |           18 |        0 |             100% |
| Prefill        | 250/250 | 235/250 = 94.0% ± 5.1% |           16 |        0 |             100% |
| Persistent 25% | 250/250 | 249/250 = 99.6% ± 0.9% |            1 |        0 |             100% |

学习过程补充：

- MA50 success 首次达到 80%：Uniform 423.8、Prefill 450.6、Persistent 257.8 episode；
- 首次完美 validation：Uniform 平均 620 且 5/5 seed 达到，Prefill 平均 1066.7 且仅 3/5 达到，Persistent 平均 420 且 5/5 达到；
- 全 checkpoint validation safe-success 均值：Uniform 63.5%、Prefill 58.5%、Persistent 83.5%；
- 最终 episode 的 validation safe-success 只有 Uniform 82%、Prefill 73%、Persistent 81%，说明 validation 选模非常必要，不能直接拿最后一轮模型做结论。

Map 2 支持“固定 25% Persistent Demo 在这一确认分布上兼具更快学习和更高动态安全性”。但 Map 1 与 Map 2 的 test 排名不同，所以不能声称普遍优越。Map 2 的 50 个 test 场景中，有 30 个没有动态路线直接穿过 nominal A* 路径，只有 1 个存在两条直接相交路线，整体偏容易。

Map 2 实验是在 Map 2 的 train 场景训练、validation 选模、Map 2 的 unseen-location test 场景测试，不是把 Map 1 模型零样本迁移到 Map 2。

## 27. 定性可视化

代表性结果位于：`outputs/dynamic_spatial_generalization_map02_confirmatory_v2/visualizations/scenario_20032_seed_1_selected/`。

- Uniform：80 步、2 次碰撞、0 次等待；
- Prefill：82 步、1 次碰撞、3 次等待；
- Persistent：78 步、0 碰撞、0 次等待。

目录内有 `replay_paths.png`、`replay_rollouts.gif` 和 metadata。该案例只能作定性说明，不能用一个案例代替 250 次 rollout 的统计结论。

## 28. 为什么静态 A* 示范可能有效，以及它为什么也可能造成冲突

当前 `persistent_demo` 不是行为克隆。它把示范 transition 与在线 transition 一起做普通 one-step TD 学习；large-margin imitation loss 只用于当前的 `dqfd` 策略。`batch_size=64`、`demo_fraction=0.25` 时，每次更新抽 16 条示范和 48 条在线经验。

所谓“A* 固定路径”也不是只有一条完全相同的路线。Map 1 的 20 个 demo episode 使用 randomized tie-breaking A*；在已检查的 seed 7400 下得到 20 条互不相同但同为 76 步的等长最短路径，共 1520 条 transition。示范状态使用 4 通道输入，但 3 个动态历史通道为零；在线动态状态则包含真实历史动态障碍。

可能有效的原因：

- 示范提供稳定的到达目标和静态拓扑价值传播，减少早期随机探索浪费；
- 不可淘汰分区保证成功转移不会被大量失败在线经验覆盖；
- 在线动态 transition 仍占 batch 多数，可以学习等待、绕行和碰撞惩罚；
- 网络能利用动态历史通道区分“静态示范状态”和“当前存在动态风险的状态”。

可能产生冲突的原因：

- 静态示范没有等待、动态绕行或动态碰撞经验；
- 固定抽取示范会减少每个 batch 中在线动态经验的比例；
- 示范会持续强化较短的 nominal 路径，当障碍恰好阻断这些位置时可能产生错误动作偏好；
- Map 1 固定 10% test 好于固定 25% 是这种偏置的线索，但不是机制证明；Map 2 又表明该偏置在部分分布中能被动态信息和在线学习纠正。

因此现有结果足以说明“当前协议下持续静态示范有时明显有益”，但不足以证明具体因果机制，也不足以证明它不会在高冲突动态场景中伤害策略。

## 29. 当前创新性和发表价值的严谨判断

已经形成的实质改进包括：

- 4 通道局部观测中加入动态障碍历史；
- 冻结且三策略共享的动态场景 manifest；
- train/validation/test 空间路线池隔离；
- 配对 seed 和 validation-only checkpoint selection；
- 测试集隔离，以及动态/静态碰撞、等待、safe success、static retention 等指标；
- Map 2 独立确认，而不是只在一张地图上挑最好结果。

这些是可靠的系统和实验协议改进，但 D3QN、A* 示范、Prefill/Persistent replay、局部占据栅格历史本身都不是新的算法。不能声称“首次把 A* 加入 D3QN”或“首次用示范回放做导航”。

当前最有价值的研究问题是：

> 静态规划专家用于动态避障时，何时能提高样本效率，何时会由于 demonstration-task mismatch 产生最短路径偏置或适应冲突，以及如何根据在线冲突程度自动调整示范使用量。

以当前结果作为硕士论文实验基础是足够的，但论文还没有闭环。若不再增加方法和证据，算法创新性不足以稳妥支撑较强的机器人/RL 期刊或会议；补足严谨统计和对照后，可以考虑应用型或层级较低的论文。若目标是更强论文，需要一个真正的方法改进和更广泛验证。

推荐题目方向：

> When Static Planner Demonstrations Meet Dynamic Obstacles: Conflict-Aware Adaptive Demonstration Replay for Sample-Efficient Navigation

中文可表述为：静态规划示范在动态避障中的收益、冲突与自适应回放。

## 30. 候选方法：冲突感知自适应示范回放

2026-09-01 的冲突审计和 2400 次补充压力 rollout 已完成，结果见第 36 节和 `docs/SPATIAL_CONFLICT_STRESS_ANALYSIS.zh-CN.md`。预先设定的“高冲突时固定 25% 被较低比例稳定反超”条件没有满足，因此当前**不要直接实现或训练 CA-ADR**。以下公式只保留为以后多地图证据出现性能交叉时的候选设计，不是已确定的下一步。

Conflict-Aware Adaptive Demonstration Replay（CA-ADR）的原候选思想不是按时间机械衰减，而是同时观察“导航还没学会”和“动态冲突正在增加”两类信号：

```text
rho_t = clip(
    rho_0
    + alpha * (1 - recent_navigation_success)
    - beta  * recent_dynamic_collision_or_conflict,
    rho_min,
    rho_max
)
```

- 导航成功率低时提高示范比例，保留 A* 的全局引导；
- 动态碰撞或冲突高时降低示范比例，让在线动态经验主导；
- 必须加入滑动平均、迟滞、更新滞后及上下界，防止比例振荡；
- 固定 10%、固定 25% 和时间衰减 25%→0 都是该方法必须比较的基线。

更强但工作量更大的可选方案是把静态 A* prior 和动态 residual 分成两个 Q 分支，再用动态风险 gate 融合。现阶段两者都不应先于多地图、难度匹配的固定比例复现。

## 31. 下一步实验和毕业闭环

优先级建议：

1. 已完成不重训的 A* conflict stress evaluation、正式 test 冲突分层复分析和冲突状态 A* action agreement 诊断；结果不支持立即训练自适应。
2. 修改后续场景生成协议：在保持空间隔离的同时，匹配 train/validation/test 的直接路线数、精确时序冲突率和冲突分数。
3. 扩展到至少 5 张结构差异明显的地图；新地图预先冻结后比较 `uniform / prefill / fixed10 / fixed25`。
4. 只有多地图重复出现“低冲突 fixed25 好、高冲突 fixed10 好”的性能交叉，才实现可逆 adaptive，并加入 `no history`、`no conflict signal` 消融。
5. 加入强基线：PER、严格限定名称的现有 DQfD-style、后续尽可能完整的 DQfD，以及 classical/time-aware planner 或 A*+DWA/反应式局部规划器。当前 `dqfd` 缺少 n-step、demo-only pretraining 等机制，不能称为完整 DQfD。
6. 做正式统计：均值、SD、95% CI、配对效应量、hierarchical bootstrap 或 performance profile。250 个 rollout 共享 5 个训练模型，不能假设它们是 250 个完全独立训练样本。
7. 固化可复现材料：manifest hash、配置、seed、选模规则、代码版本、环境依赖和失败运行排除标准。
8. 若学位方向偏机器人/控制，应补运动学、传感器噪声和高保真仿真，最好再做实机；若偏计算机/软件，可依据学校要求用更严格的大规模仿真替代部分实机工作。

当前限制必须在论文中主动承认：只有两张正式动态地图；Map 2 偏容易；动态障碍是确定性往返运动；观测较理想；运动为离散栅格；缺少完整 DQfD 和强经典局部规划基线；尚未完成正式置信区间与高保真/实机验证。

## 32. 实验结果目录的使用分级

### 正式主结果

- `outputs/dynamic_spatial_generalization_map02_confirmatory_v2/`：Map 2 独立确认，当前最重要的正式动态结果。

### 正式选择和消融

- `outputs/dynamic_spatial_generalization_map01_checkpoint_v2/`；
- `outputs/dynamic_spatial_generalization_map01_demo_decay_v2/`；
- `outputs/dynamic_spatial_generalization_map01_demo10_v2/`。

### 静态背景证据

- `outputs/random_benchmark/Static Local Observation Baseline v1/`；
- `outputs/structured_calibration_40x40/` 中正式 5-map 和最终 Map 1 15-run cohort，使用时按 `docs/EXPERIMENT_INVENTORY.zh-CN.md` 指定目录筛选。

### 支持性 pilot / 探索性结果

- `outputs/dynamic_controlled_mixed_six_obstacles_40x40_map01/`：仅 3 个完整 `trainseed_0` 目录，可作 pilot，不可伪装成 5-seed 正式结果；
- `outputs/dynamic_spatial_generalization_map01/`：早期 exploratory，排除 `_smoke`。

### 不纳入论文主统计

- 所有名称带 `smoke` 的目录；
- `outputs/dynamic_calibration_map01/`；
- 不完整的 controlled bottleneck、controlled six、hard smoke 等运行；
- PER/DQfD smoke；
- 旧的重复静态 seed 0；
- 只有配置、没有正式输出的 Map 2 decay/demo10。

整理数据时不要凭文件夹名猜测，先查看 `docs/EXPERIMENT_INVENTORY.zh-CN.md`。

## 33. 术语和结论边界

- Map 1/Map 2 的 test 是同一静态地图内、未见动态路线位置的空间泛化，不是跨地图 zero-shot transfer；
- Prefill 示范被 FIFO 淘汰是“数据遗忘”，不自动等于“策略灾难性遗忘”；
- 当前 dynamic-from-scratch 实验更适合称“静态示范偏置”“示范—任务冲突”或“动态适应受限”；
- 只有先学会旧任务、再适应新任务、并持续评估旧任务保持率，才能严谨讨论灾难性遗忘；
- `success=1` 只表示最终到达，安全结论必须看 `safe_success` 和分类碰撞；
- 单案例 GIF 只能用于解释行为，主结论必须来自多 seed 聚合；
- Map 2 证明的是定义好的确认分布上的优势，不能外推为所有地图、所有动态密度和所有障碍运动模型上的优势。

## 34. 相关文献入口

后续写作至少需要与以下方向进行对比，不能只与本项目早期版本比较：

- DQfD：[https://arxiv.org/abs/1704.03732](https://arxiv.org/abs/1704.03732)；
- Reinforced Imitation for end-to-end navigation：[https://arxiv.org/abs/1805.07095](https://arxiv.org/abs/1805.07095)；
- demonstration-guided UAV local planning：[https://arxiv.org/abs/2008.02521](https://arxiv.org/abs/2008.02521)；
- map-based DRL navigation and real-robot transfer：[https://arxiv.org/abs/2002.04349](https://arxiv.org/abs/2002.04349)；
- DWA-RL：[https://ieeexplore.ieee.org/document/9561462/](https://ieeexplore.ieee.org/document/9561462/)；
- Deep RL reproducibility：[https://arxiv.org/abs/1709.06560](https://arxiv.org/abs/1709.06560)；
- Statistical Precipice in RL evaluation：[https://arxiv.org/abs/2108.13264](https://arxiv.org/abs/2108.13264)；
- failure-aware demonstration navigation（2026）：[https://arxiv.org/abs/2604.23360](https://arxiv.org/abs/2604.23360)。

这些只是当前已核对的入口，不代表已经完成穷尽式新颖性检索。正式投稿前仍需按目标期刊/会议和近三年关键词更新综述。

## 35. 最近验证记录

最近一次完整测试记录是 2026-09-01：加入冲突审计和压力评估代码后，91 项 `unittest` 全部通过。本轮没有重新训练模型，但运行了现有 selected models 的补充评估和 A* 冲突动作诊断。

## 36. 2026-09-01 冲突审计与压力评估结果

新增完整报告：`docs/SPATIAL_CONFLICT_STRESS_ANALYSIS.zh-CN.md`。输出目录：`outputs/spatial_conflict_stress_v1/`。

已完成工作：

- 审计 Map 1/Map 2 的 train/validation/test 几何与 nominal 到达时序冲突；
- 用各地图 test 路线池构造20组相同路线组合，每组 low/medium/high 三种相位；
- 使用现有 selected models 完成 2400 个压力 rollout，没有重新训练；
- 复分析原正式 test 在冲突分数三等份上的结果；
- 重放策略并记录“照静态 A* 下一步走会立即碰撞”状态中的 A* 动作一致率。

manifest 审计发现：Map 1 test 52%、Map 2 test 60% 的场景没有动态路线直接覆盖 nominal A*；Map 2 validation 的平均冲突分数 42.09，test 只有 13.15，说明空间隔离有效但 split 难度没有匹配。

压力评估 safe-success：

| Map   | 方法           | Low | Medium | High |
| ----- | -------------- | --: | -----: | ---: |
| Map 1 | Uniform        | 18% |    46% |  34% |
| Map 1 | Prefill        | 51% |    49% |  32% |
| Map 1 | Persistent 10% | 35% |    50% |  33% |
| Map 1 | Persistent 25% | 59% |    71% |  47% |
| Map 1 | 25%→0         | 47% |    59% |  29% |
| Map 2 | Uniform        | 97% |    94% |  98% |
| Map 2 | Prefill        | 97% |    96% |  94% |
| Map 2 | Persistent 25% | 99% |   100% | 100% |

Map 1 方差很大，主要是训练 seed 分化和未见路线到达失败；Low/Medium/High 是相对于 nominal A* 时序定义，不保证对偏离 nominal 的学习策略单调变难。部分 Persistent 25% seed 在即时冲突状态会继续执行 A* 动作并碰撞，但其他 seed 能完全避开，Uniform、Prefill、固定10%也出现类似行为。

决策：当前不实现 CA-ADR。现有证据没有显示 fixed25 在高冲突档被 fixed10/Prefill 稳定反超。下一步先做新增地图的冲突难度匹配和固定比例多地图复现；只有出现可重复的性能交叉，才恢复自适应主线。

## 37. 2026-09-01 三地图冲突均衡基准 v3

已按第 36 节的决策新增三张结构不同的正式候选地图，没有覆盖 Map 1/Map 2 或旧结果：

- `office_40x40`：房间、门洞、主走廊；
- `parcel_station_40x40`：中央分拣岛、装卸通道；
- `warehouse_40x40`：平行货架、横向通道、多瓶颈。

每张地图均冻结 100 train、20 validation、50 test 场景；每场景仍为 3 条 corridor 加 2 条 background。三个 split 的路线格彼此至少间隔一格，且同时匹配 low/medium/high 比例 40%/30%/30%。low 为 0 条直接交叉，medium 为 1 条直接交叉且只有一次 ±2 步近时冲突，high 为 2 条直接交叉并保证即时冲突。三张地图的 split 主要冲突指标已对齐，路线组合最多重复两次，完整相位配置不重复。

配置：

- `configs/dynamic_spatial_generalization_office_balanced_v3.yaml`；
- `configs/dynamic_spatial_generalization_parcel_balanced_v3.yaml`；
- `configs/dynamic_spatial_generalization_warehouse_balanced_v3.yaml`。

frozen manifest SHA-256 前 12 位分别为 `d4b3e5f7909e`、`99d92dd8f639`、`d6561604cc9f`。设计审计输出在 `outputs/spatial_generalization_balanced_v3_design/`，完整说明见 `docs/BALANCED_SPATIAL_BENCHMARK_V3.zh-CN.md`。本轮只运行生成器和测试，没有启动训练；正式训练应比较三地图上的 Uniform、Prefill、Persistent 25%，每种 5 seed，共 45 次运行。只有观察到可重复的跨地图或跨冲突层性能交叉，才继续设计自适应示范比例。

## 38. 2026-09-02 时空难度均衡基准 v4

针对 v3 只匹配名义时序冲突、没有匹配真实避障代价的问题，新增了时空安全路径 oracle：按环境的障碍物先移动语义，在“位置 + 障碍物周期相位”状态空间中用 BFS 求最短无碰撞路径，并计算 `safe_detour_steps`。新增文件为 `src/astar_d3qn/evaluation/conflict.py`、`scripts/audit_spatiotemporal_difficulty.py`、`scripts/generate_balanced_spatial_benchmark_v4.py`，以及 `tests/test_balanced_spatial_benchmark_v4.py`。

已冻结三张 v4 基准：`configs/dynamic_spatial_generalization_office_balanced_v4.yaml`、`configs/dynamic_spatial_generalization_parcel_balanced_v4.yaml`、`configs/dynamic_spatial_generalization_warehouse_balanced_v4.yaml`。generation seed 分别为 `20260909`、`20260915`、`20260948`；manifest SHA-256 完整值分别为 `A9CD5A058770C01920DC60FBCD0C390A5F8C8BD3B0D76B45A5430FEC5BD8114B`、`3FAFF73CB3614FDF099360E57CB26AFA50C4D9AA0EED57E2A7D3AF6836B34B4D`、`A26163108D390C7DF5236F8C270FDE0C7007D687E6EDC89DCF9DD27B13F366E1`。完整审计位于 `outputs/spatial_generalization_balanced_v4_design/scenario_spatiotemporal_difficulty.csv`：Office 高冲突 detour 均值 train/validation/test 为 0/0/0，Parcel 为 1.8/2.0/2.0，Warehouse 为 1.033/1.0/1.067；低、中冲突层均为 0，510 个场景全部可达。

v4 仅改变场景基准和难度审计，不改变网络、奖励、动作或 replay。尚未启动 v4 训练；下一步应在三张 v4 地图上比较 Uniform、Prefill、Persistent 25%，每种 5 个 seed，共 45 次，并将 v3 结果与 v4 结果分开报告。

## 39. 2026-09-03 v4 正式结果与关键位置阻塞诊断

三张 v4 地图的 Uniform、Prefill、Persistent 25% 各 5 seed 已全部完成，共 45 次运行。test safe-success（5 seed 均值 ± SD）为：Office `59.6% ± 11.3% / 62.0% ± 22.9% / 58.4% ± 13.4%`，Parcel `93.2% ± 10.5% / 93.2% ± 4.1% / 82.0% ± 16.6%`，Warehouse `64.4% ± 25.5% / 58.0% ± 17.3% / 84.4% ± 11.1%`。Persistent 平均训练更快，但最终安全性呈明显地图依赖；5 seed 下各单地图差异的双侧检验均未达到 0.05，不能写成普遍优越。

为解释该交互，新增 `scripts/run_v4_critical_blockage_diagnostic.py`，只评估冻结的 selected models，不训练、不重新选模。每张地图选择 early/middle/late 三个高 A* 路径中心性格，每格两个障碍离开方向；每个场景含 1 个精确阻塞下一步 A* 动作的目标障碍和 4 个 off-path 背景障碍，共 18 个可安全到达的诊断场景。输出在 `outputs/v4_critical_blockage_diagnostic/`，说明见 `docs/V4_CRITICAL_BLOCKAGE_DIAGNOSTIC.zh-CN.md`。

强制冲突状态探针中，选择被阻塞 A* 动作的比例为：Office `Uniform 26.7% / Prefill 40.0% / Persistent 46.7%`；Parcel `23.3% / 40.0% / 40.0%`；Warehouse `46.7% / 23.3% / 26.7%`。相应自主安全成功为 Office `80.0% / 80.0% / 70.0%`，Parcel `86.7% / 80.0% / 66.7%`，Warehouse `53.3% / 66.7% / 70.0%`。45 个地图-策略-seed 观测中，冲突状态继续选择 A* 与目标碰撞率呈正相关（Spearman rho `0.566`，探索性 `p=5.09e-5`），但存在聚类，不能冒充因果证明。

当前主线因此从“按 episode 线性衰减示范比例”改为“冲突感知示范回放”：安全示范保留，当前/下一步动态占用与示范动作冲突时降低或取消该 transition 的采样权重。下一步先实现该方法及当前占用、一步预测两个消融，再用冻结基准与固定 25% 比较。

上述方法已经实现为 replay strategy `conflict_adaptive_demo`。由于静态示范的动态通道为零，实际实现没有伪造逐示范动态标签，而是在在线状态中恢复同一位置的示范动作分布，按环境真实碰撞语义计算当前/下一步动态冲突率的 EMA，再将示范比例在 0 到 25% 之间调整。默认 `alpha=0.2`、`sensitivity=1.0`；没有示范覆盖的在线位置不更新 EMA。主方法检查当前与下一步动态格，消融只看当前占用。

新增六份配置：三份 `dynamic_spatial_generalization_*_conflict_adaptive_v1.yaml` 为主方法，三份 `dynamic_spatial_generalization_*_conflict_current_v1.yaml` 为当前占用消融。完整定义见 `docs/CONFLICT_ADAPTIVE_DEMO_V1.zh-CN.md`。Office seed 0 的两个 4-episode smoke 已通过，全量测试为 100 passed；没有启动正式自适应训练。正式新增工作量为两种方法 × 三地图 × 五 seed，共 30 次，旧 v4 三种基线不必重跑。

## 40. 2026-09-04 全局自适应负结果与状态级安全引导

两种全局自适应各三地图 × 5 seed 的 30 次正式训练已完成。test safe-success：预测版 Office/Parcel/Warehouse 为 `71.6%/71.6%/69.6%`，只看当前占用版为 `50.4%/64.0%/66.4%`；对照 Fixed 25% 为 `58.4%/82.0%/84.4%`。预测版只改善 Office，在 Parcel 和 Warehouse 下降；Warehouse 相对 Fixed 25% 的 5 个 seed 全部下降，均值差 `-14.8` 个百分点。三地图宏平均为 Uniform `72.4%`、Prefill `71.1%`、Fixed 25% `74.9%`、预测自适应 `70.9%`、当前位置版 `60.3%`。当前全局 EMA 方法必须作为负结果，不能宣称优于基线。

机制原因：预测版训练时平均 demo fraction 仍约 `24.7%–24.8%`，冲突观测约低于 1%，低于 24% 的 episode 仅约 `2.6%–5.5%`；它接近 Fixed 25%，且局部冲突会错误地削弱所有位置的示范。结果总览为 `outputs/conflict_adaptive_v1_analysis/conflict_adaptive_v1_summary.png`。关键阻塞复测已扩展到五种方法：预测版危险 A* 选择率 Office/Parcel/Warehouse 为 `23.3%/16.7%/53.3%`，说明它在前两图局部有效、Warehouse 反而恶化。

已新增替代方法 `safe_guided_demo`。它以可淘汰 Prefill 启动；在线状态中只有 A* 动作不与动态障碍当前格和下一步预测格冲突时，才把该动作写入 transition 的 `safe_demo_action` 并施加 large-margin 引导；全部 A* 动作危险时不提供标签，由 TD 学习等待或绕行。它不再调整全局示范比例。实现涉及 `src/astar_d3qn/replay/transition.py`、`src/astar_d3qn/agents/d3qn.py`、`src/astar_d3qn/training/trainer.py` 和动态空间训练入口。三份 `dynamic_spatial_generalization_*_safe_guided_v1.yaml` 已添加，完整说明见 `docs/SAFE_GUIDED_DEMO_V1.zh-CN.md`。103 项测试和一次 4-episode smoke 通过，尚未正式训练。

## 41. 2026-09-05 Office 行为验证场景 v6

v5 Office 关键位置场景经精确动态最短路复查后被判定不适合正式训练：训练集严格等待场景为 `0/100`，无法支撑“学习等待/避让”的研究问题，因此不要训练 v5。

已新增 `configs/dynamic_spatial_generalization_office_behavior_v6_terminal_v1.yaml`、`scripts/generate_office_behavior_scenarios_v6.py`、`src/astar_d3qn/evaluation/behavior_oracle.py` 和 `tests/test_behavior_oracle.py`。v6 不再按视觉位置人工判定，而是用与环境碰撞时序一致的周期时间扩展最短路，将场景严格分为：等待、局部避让、全局改道。

冻结清单为 `data/dynamic_scenarios/office_40x40_behavior_v6.json`，SHA-256 前 12 位为 `ce47ee4ef71f`。训练/验证/测试分别为 100/20/50 场，行为数量为 `40/30/30`、`8/6/6`、`20/15/15`。每场固定 5 个动态障碍，5 个都必须在正确时刻阻断至少一条实际 A* 示范；每场至少 2 个障碍还必须通过删除反事实改变示范冲突并集或门组合安全代价。三个集合的完整运动路线互不重复，但允许共享少量物理门洞格，因为 Office 只有四个真实门洞；该实验检验时序/组合/行为迁移，不应冒充跨地图空间泛化。

审计结果：三个集合的五障碍有效率均为 100%；平均反事实决定障碍数为 `2.80/2.30/3.16`；等待场景的最优路线均恰有一次等待。预览图在 `maps/previews/office_40x40_behavior_v6_*.png`，逐场审计在 `outputs/office_behavior_v6_design/`，方法说明见 `docs/OFFICE_BEHAVIOR_VERIFIED_V6.zh-CN.md`。行为 oracle 与场景加载共 8 项针对性测试通过，并已独立核对 170 个场景 ID 唯一、跨集合完整路线不重复、每场五障碍全部有效且至少两个反事实决定。尚未启动 v6 正式训练。

结果可视化已经补齐，但尚无正式训练数据可画。修复了 `evaluate_agent` 在同一地图重复评价 50 个场景时用 `map_id` 覆盖前 49 条轨迹的问题；现在使用唯一 `trajectory_key`。每次训练会保存 `test_trajectories.json`、单 seed 训练曲线和验证曲线。全部 15 次运行完成后，`scripts/plot_office_behavior_v6_results.py` 会严格检查三策略 × 5 seed 和 manifest hash，并生成 5-seed 95% CI 验证曲线、50-test 总体图、按 wait/avoidance/reroute 分层图、50 场热力图，以及每个 seed 全部 50 场三策略并排路径图（共 25 页）。测试全集为 113 passed，另修复了一个使用精确相等比较浮点 epsilon 的脆弱旧测试。20 场是 validation，50 场是最终 test，后续表述不能混淆。

## 42. 2026-09-05 当前统一研究方向、方法与结果摘要

本节是下一次恢复会话时应优先采用的统一结论；它不删除前面的实验历史，但取代旧章节中已经过时的“当前方向”和“尚未训练”描述。

### 42.1 已经明确的研究问题

最初“用 A* 示范初始化或填充 D3QN，以加快路径规划学习”可以作为工程方法和基线，但单独作为论文创新不够。当前真正要研究的问题是：

> 静态 A* 专家示范在动态障碍导航中，什么时候能提高样本效率和全局到达能力，什么时候会因为缺少等待、动态绕行和碰撞经验而产生示范—任务不匹配，以及这种影响在等待、局部避让和全局改道三类行为上有何差异。

当前阶段先做清楚“静态示范到底带来了什么影响”，而不是继续无目的增加自适应模块。Office v6 首先比较三个基础策略；只有结果出现稳定、可解释的差异后，才决定是否设计新算法并扩展到其他地图。

目前可以主张的潜在贡献主要是：行为可验证的动态场景构建与 oracle、严格的训练/验证/测试协议，以及静态规划示范在动态任务中的收益与冲突的系统实证。算法层面的新方法尚未成功建立；全局自适应和状态级安全指导都已有负结果，不能包装成有效创新。

### 42.2 当前 Office v6 的正式比较对象

本轮只比较以下三种策略，每种 5 个训练 seed，共 15 次：

1. `uniform`：不使用 A* 示范，只从在线动态环境经验学习；
2. `prefill`：训练前用 20 条静态 randomized-tie A* 示范填充普通 FIFO replay，之后示范可以被在线经验逐渐淘汰；
3. `persistent_demo`：示范保存在不可淘汰分区中，每个 batch 固定 25% 示范、75% 在线动态经验。它仍是 one-step Double-DQN TD 学习，不等于行为克隆或完整 DQfD。

公平性控制已经固定：三策略共享同一 v6 manifest；训练 seed 均为 0–4；每次最多 400,000 个环境交互步；每 25,000 步在 20 个 validation 场景上评价；checkpoint 只按 validation 的安全成功、成功率、动态碰撞次数和步数选择；50 个 test 场景只在选模后评价一次。环境含上、下、左、右、原地等待 5 个动作，局部观察为 4×15×15（静态地图加三帧动态历史）和目标方向标量；碰撞奖励为 -1、目标为 +10、普通步为 -0.01、等待额外为 -0.05，训练和评价均为碰撞后立即结束该 episode。

“什么才叫好”不能只看最终是否到达。主指标是 `safe_success_rate` 和 `dynamic_collision_rate/count`；同时报告普通 success、达到同一安全水平所需环境交互步数、等待次数、路径效率、转弯/重复访问，以及 wait/avoidance/reroute 三类场景的分层结果。多 seed 是统计单位，不能把 5 个模型在 50 场上的 250 次 rollout 当成 250 个独立训练样本。

### 42.3 Office v6 场景为什么比旧地图有效

旧 v5 被精确 oracle 证明训练集严格等待场景为 0/100，因此已经否决，不能再训练或拿来证明智能体学会等待。v6 使用与环境碰撞时序一致的周期时间扩展最短路，预先把场景验证为：

- `wait`：允许 STAY 的最短安全路径严格短于禁止 STAY 的路径；
- `avoidance`：标称 A* 会冲突，但原门组合仍最优，需要局部偏移；
- `reroute`：替代上下门组合严格优于标称门组合。

v6 冻结 100 train、20 validation、50 test，三类数量分别为 40/30/30、8/6/6、20/15/15。每场 5 个动态障碍都必须在正确时刻阻断至少一条真实 A* 示范，每场至少 2 个障碍还要通过删除反事实改变示范冲突集合或门组合安全代价。完整路线跨 split 不重复，但允许共享 Office 数量有限的物理门洞；所以 v6 检验的是同一 Office 拓扑中的时序、组合和行为迁移，不是跨地图零样本泛化。manifest 为 `data/dynamic_scenarios/office_40x40_behavior_v6.json`，SHA-256 前 12 位 `ce47ee4ef71f`。

### 42.4 已使用方法和已经得到的结果

1. **Map 1/Map 2 v2 基础实验**：Persistent 25% 通常更快达到较高训练/验证成功，但最终 test 并非处处最好。Map 1 test safe-success 为 Uniform 83.6%、Prefill 84.8%、Persistent 76.4%；Map 2 为 93.6%、94.0%、99.6%。固定 10% 在 Map 1 test 为 84.0%，高于固定 25% 的 76.4%，而 25%→0 为 74.0%。这些结果证明策略排名依赖地图/分布，但旧 split 难度不匹配，不能直接据此宣布最优比例。
2. **三地图 v4 时空难度实验**：Office/Parcel/Warehouse 的 test safe-success（Uniform/Prefill/Persistent 25%）分别为 59.6/62.0/58.4%、93.2/93.2/82.0%、64.4/58.0/84.4%。Persistent 平均训练更快，但最终安全性存在明显地图交互，单地图 5 seed 的双侧检验均未达到 0.05，不能声称它普遍最好。
3. **关键阻塞与动作一致性诊断**：在 45 个地图-策略-seed 观测中，冲突状态继续执行静态 A* 动作与目标碰撞率呈正相关，探索性 Spearman rho=0.566、p=5.09e-5。该结果支持“静态示范可能形成冲突动作偏好”的机制线索，但观测有 seed/地图聚类，不是独立样本，不能写成因果证明。
4. **全局冲突自适应 `conflict_adaptive_demo`**：正式三地图×5 seed 结果为 Office/Parcel/Warehouse test safe-success 71.6/71.6/69.6%，只看当前位置的消融为 50.4/64.0/66.4%，对照 Persistent 25% 为 58.4/82.0/84.4%。预测版只改善 Office，三地图宏平均 70.9%，低于固定 25% 的 74.9%；当前位置版仅 60.3%。其平均示范比例仍约 24.7%–24.8%，冲突信号低于约 1%，方法几乎没有真正调节，属于明确负结果。
5. **状态级 `safe_guided_demo`**：后来已完成 Office/Parcel/Warehouse 各 5 seed，旧第 40 节“尚未正式训练”已经过时。test safe-success（5 seed 均值±SD）分别为 69.2%±17.3%、50.0%±8.7%、27.6%±20.6%；相对同一 v4 的 Persistent 25%，Office +10.8 个百分点、Parcel -32.0、Warehouse -56.8，三地图宏平均仅 49.0%。训练中候选指导动作被实际放行的比例约为 Office 99.6%、Parcel 99.2%、Warehouse 98.6%，所谓风险过滤几乎从不拦截，因此该方法也不能作为成功主方法。更具体的因果解释仍需消融，不能只根据相关结果断言。
6. **场景版本教训**：v3 只按名义冲突分层；v4 增加动态安全最短路，但 Office 的高冲突层真实 detour 仍为 0；v5 虽把障碍移到关键区，训练仍没有严格等待；v6 才第一次在训练前用行为 oracle 明确保证等待、局部避让和改道。这说明此前主要问题首先是场景定义和验收标准不足，不只是奖励函数或网络结构问题。

### 42.5 当前决策和接下来的判断规则

Office v6 三策略实验是当前唯一优先任务，暂不加入 `conflict_adaptive_demo` 或 `safe_guided_demo`。它的目的不是再次笼统证明“有 A* 比没有 A* 好”，而是在确实要求动态行为的环境中区分三件事：学习速度、最终安全性、以及不同动态行为类型上的偏差。

- 如果 Persistent 25% 学得更快，但在 wait/avoidance 场景碰撞更多或 safe-success 更低，而 Prefill/Uniform 更安全，则得到较强的 demonstration-task mismatch 证据，下一步才有理由研究局部、动作级或时间感知的示范使用机制。
- 如果 Persistent 25% 在学习速度和三类最终安全性上都更好，则当前数据不支持“静态示范伤害动态适应”；研究应转为静态示范的样本效率与鲁棒性，并在新地图和强基线中确认，不能为了创新硬造自适应方法。
- 如果三策略差异小且 5 seed 区间大量重叠，则不能凭均值选赢家；应先检查统计功效、失败类型和跨地图复现，而不是立即调奖励或再加模块。

Office v6 是在看过 v4 结果和多次调整场景后建立的开发基准，因此它本身不能作为完全独立的最终确认集。方法与假设在 Office v6 上冻结后，最终结论至少还要在预先冻结、行为同样可验证的 Parcel/Warehouse 或新 held-out 地图上确认，并补充更强基线（至少 PER、严格定义的 DQfD-style、时间感知/经典局部规划器）。若论文目标较高，还需连续/运动学仿真、传感器噪声，最好有实机或更高保真验证。

### 42.6 当前运行与可视化状态

截至 2026-09-05 17:32，用户已经启动 Office v6 命令，当前正在运行 `uniform seed=0`；该 run 目录已创建，但尚无 `run_metadata.json`、`model_selected.pth` 或 `test_trajectories.json`，因此 v6 还没有任何可报告的正式策略结果，不能提前下结论，也不要重复启动第二套相同训练。

训练脚本已经修复 50 个同地图测试轨迹被同一 `map_id` 覆盖的问题，并会为每次完成的运行保存唯一 `trajectory_key`、完整 `test_trajectories.json`、单 seed 训练曲线和 validation 曲线。15 次完整运行后，`scripts/plot_office_behavior_v6_results.py` 会检查所有 run 与 manifest hash，输出三策略 5-seed 95% CI 曲线、50-test 总体和行为分层图、逐场热力图，以及 5 seed×50 test×3 策略的全部路径分页图。20 场是 validation，50 场是最终 test，报告中必须严格区分。当前代码完整测试为 113 passed。

重要实验边界：v4 test 已经用于观察结果和设计新方法，因此新方法在 v4 test 上只能算开发实验，不能冒充全新确认结果。方法冻结后需另建未见确认场景，并对保存的旧基线模型和新模型只评估一次。

## 43. 2026-09-08 起 Office v10 配对因果场景（开发验收中）

2026-09-09 当前决策（最高优先级，取代本节下文“先继续生成v10”的指示）：用户明确要求停止无边界的地图/诊断生成，已取消运行`--avoidance-approaches`，不能再让用户先生成4对。训练基本验收与严格机制诊断分开。只读盘点发现v9/v10正式manifest不存在，最近产物均为开发test小批量；最新完整Office三集为v6，100/20/50，170个场景、38条完整路线无重复、静态路线/路径和运动参数未发现问题。全部保存70～75步oracle路径，本轮未重新动态回放。v7已完成run的manifest hash与现存v6一致。v6有三种行为但无normal；不把严格配对或不能回退等诊断条件强加为训练门槛。

用户要求继续后新增独立配置`configs/dynamic_spatial_generalization_office_v6_development_v1.yaml`，继承v8并显式锁定v6三集数量，stay=0等已确定奖励、碰撞终止、静态mask、40万步、34万步epsilon衰减、2.5万步验证。课程0/6万/14万/24万步启用0/1/3/5障碍，不筛选v6不存在的control/easy/medium/hard标签，阶段名改为subset，不声称数量必然对应难度。只规划三个策略各seed=0的开发试跑，非统计确认；验证/test始终5障碍。旧test已被查看，仅作为开发评估，不能冒充独立确认。

独立输出`outputs/dynamic_spatial_generalization_office_v6_development_v1/`，旧结果/配置不动。现有`plot_office_behavior_v6_results.py --config <新配置>`会生成三策略50场图和路径分页；修正其单seed标题/metadata，不声称可估计跨seed置信区间。兼容性测试已补但未执行，未启动程序或训练。可直接训练加汇总的单行命令和边界见`docs/OFFICE_V6_DEVELOPMENT_PILOT.zh-CN.md`。下一步用户运行3个seed0训练后读取真实结果，不再要求先完成v10特殊诊断或更换地图。

2026-09-09 门前/门后局部避让修改（当前最新）：用户复查 `route_expansion_pilots/20260909_220204_737133400/avoidance_failure_diagnostics/20260909_231226_446893000/` 的10个留存样例中，9个因固定门对下永久排除首冲突格而静态断开，另1个在(8,9)可静态绕行但达到50,000展开状态上限（不是5秒超时）；不能把留存样例比例外推至全部26个候选。静态必经格为(9..12,9)、(24..27,17)，坐标均为(row,col)。这证实部分候选位置与严格空间绕行见证不相容，不表示真实机器人不能等待/回退后通过。

已为 `scripts/pilot_office_v10_route_expansion.py` 增加 `--avoidance-approaches`：只做左上、左下两组avoidance，每组目标2对，共4对；不重跑等待/换路/右侧/终点。左上锚点(7,9)/(14,9)，左下(23,14)/(29,17)，分别取周围±1格、水平/纵向长度3/5/7，按两侧各长度/方向至多3个几何候选（总计最多72）。保留原test路线作协变量，新增主障碍只从本轮入选approach路线挑选；每区最多6条，优先平衡两侧，仍须实际通过相位配对必要条件和静态可绕检查。没有合格新路线则停止该组，不偷偷退回旧终点案例。位置/相位多样性仍待检查，4对不是训练集。

所有空间avoidance试生成在行为oracle前按具体相位过滤：找到参考首次主冲突格，检查保持原门对且删除该格后静态能否连通；缓存每格结果，不整条删除有其他可行相位的路线。生成`spatial_proposal_precheck.json`，等待/换路提案与接受规则不变。动态碰撞、决策可见性、无STAY/无重复格、删除主障碍反事实、相位对照等验收均保留，资源耗尽仍未知。仅开发test池新输出，不改变已冻结三集、墙体、起终点、示范、奖励和训练算法，不覆盖已有6对。

单行命令：`Set-Location 'D:\Asatr-D3QN-Workshop'; python -u .\scripts\pilot_office_v10_route_expansion.py --avoidance-approaches`。输出 `outputs/office_behavior_v10_dataset_design/avoidance_approach_pilots/<时间>/`，包括`avoidance_approaches.png`（候选图非已认证场景）、`expanded_route_pools.png`、相位过滤记录、逐对路径PNG/JSON和`batch_summary.json/png`。每组最多300候选/30秒协作预算，预处理、重验和绘图有额外耗时。仅静态复核并补回归测试，按用户要求未运行程序/测试。下一步检查四对是否实际产生及空间/路径多样性，再决定是否进入三集设计；不要宣布训练已就绪。

2026-09-09 区域扩展结果与失败复查（当前最新，优先于下文）：用户批次 `outputs/office_behavior_v10_dataset_design/route_expansion_pilots/20260909_220204_737133400/` 完成运行但15个区域×行为目标只得到6对：wait左上/左下/终点各1，reroute左上/左下各1，avoidance终点1。不是旧18对实验的退步，任务配额不同。右侧六组预检查无合格主障碍提案、实际搜索0次；当前固定参考走左门，不能由此宣称右侧区域对所有策略无关。左侧avoidance的26个可见可行候选有15个空间约束下无解、11个资源耗尽未知；旧统计没保存具体solver停止原因，不能直接断言15个全是门洞几何问题。

本轮新增 `scripts/diagnose_office_v10_avoidance_failures.py --batch-dir <上述批次>`，只复查失败avoidance组留存的可见性样例（上限每组5例，并非全部26例），不随机生成、不训练。输出 `avoidance_failure_diagnostics/<时间>/reference_cell_geometry.png/json`、逐例路径对比PNG/JSON及`summary.json`。静态四连通检查比较保持原门对时删除冲突格前后的连通性；绿色仅代表静态可绕，不能当作动态安全、局部可见或能学会的证明。空间搜索保留5秒单例预算，具体区分静态断开、完整约束搜索无解、资源耗尽未知和成功见证；重算标签/碰撞回放/绘图另计时间，已完成结果在异常/中断时保留。

生成器新增空间搜索详细停止原因统计及每种原因最多3份完整失败场景留存，不再把静态断开和动态/可见性约束失败混为一谈；接受条件、静态地图、动态路线池、奖励和训练配置均未修改。补充回归测试但按用户要求未执行任何程序或测试。下一步先读复查证据再确定局部避让路线，不要再次原样扩大15组搜索；已有6对保留但不足以批准正式数据集。开发过程中反复查看的test池不能再当最终独立确认集。

2026-09-09 区域路线扩展（最新）：空间复查 `.../spatial_bypass_diagnostics/20260909_213429_569824800/` 六例均找到72步简单空间绕行路径，与旧72步往返解等长、决策距离7、绕开(37,26)。证明这些场景允许空间绕行，不证明必须绕行或已学会；六例空间路径仍相同。

用户要求继续后新增 `scripts/pilot_office_v10_route_expansion.py`。保留静态地图/旧结果，开发test池在四门洞附近和终点通路附近枚举长3/5/7的横纵路线（每区至多36条预查、添加至多6条，不复制其他split完整路线），train/validation未改。新入口按5主障碍区域×3行为各找1对，共15组，区域索引作为生成seed偏移，不是三seed重复实验；每组300次或30秒。主区单独配额，不拿左上补缺项，factorized采样避免路线组合爆炸。avoidance同时要求已验证可见路径和无重复格空间绕行证据，原标签/配对/反事实不变。空间预算未知不接受且单独计数，候选接受后独立回放空间路径。

单行指令：`Set-Location 'D:\Asatr-D3QN-Workshop'; python -u .\scripts\pilot_office_v10_route_expansion.py`。结果到 `outputs/office_behavior_v10_dataset_design/route_expansion_pilots/<时间>/`，含扩展路线图/完整候选审计、逐区域失败原因、数量与区域图、原可见/空间路径指标。任何组缺项退出2，保留全部已完成结果，不等于地图不可行。未运行生成器或测试，仅静态复核。下一步用户运行再看实际路线多样性，暂不训练。详见设计文档第14节。

2026-09-09 空间绕行复查（最新）：用户批次 `visible_avoidance_batch_pilots/20260909_211848_318101000/` 已得到18对、36场，配对字段检查无误。但6个avoidance可见路径完全相同，72步中的2步为 `(37,19)→(36,19)→(37,19)` 往返，并非已证实的空间绕行。wait/reroute仍全部左上、avoidance全部终点附近。不能把“无STAY”当作空间绕行成功，也不能以18对凑齐宣布训练数据合格。

用户要求继续后新增 `scripts/diagnose_office_v10_spatial_bypass.py --batch-dir .\outputs\office_behavior_v10_dataset_design\visible_avoidance_batch_pilots\20260909_211848_318101000`，只复查已有6例，不改地图/路线/验收。单独记录STAY、立即往返、一般闭环，另用有预算A*寻找可见后首次偏离、同门组合、无重复格且绕开原首个主障碍冲突格的安全路径。保存所有访问历史避免错误合并；找到证明此严格类别可行，预算耗尽是未知，穷尽无解也仅限该类别。默认15秒/例、展开50k/生成150k状态，不含源数据复验和绘图。输出到该批次 `spatial_bypass_diagnostics/<时间>/`，有原闭环与新空间路径左右图、步数、逐步碰撞/可见性、summary。按用户要求未运行程序或测试；下一步用户执行并反馈结果。详见设计文档第13节。

2026-09-09 可见避让小批量v1（最新，优先于下文）：用户已完成可见性诊断 `batch_pilots/20260909_172002_590211200/visibility_diagnostics/20260909_205805_765544700/`。3例均是终点接近区，原全知70步、决策距离9；可见后偏离72步、决策距离7。并非这3例存在等长可见路径，不能写成平局误选已证实。

按用户批准的“可行性与效率分开”新增 `--pair-batch-pilot --visible-avoidance --diagnostic-attempts 300 --pilot-seconds-per-group 60`。只在此新小批量模式里将avoidance的可见性验收改为存在无等待、同门组合、首次偏离前可见的安全路径，不强求与全知最优等长，不设额外步数上限。原三行为标签/最优路径/最优步数、删除反事实、示范冲突和相位配对规则保留；wait/reroute不变。可见路径/额外步数独立存 `visible_avoidance`，原不可见决策不伪造为可见，产物重新求解校验两套指标。新目录 `outputs/office_behavior_v10_dataset_design/visible_avoidance_batch_pilots/<时间>/`，含配对图（绿色为可见路径）、区域汇总、`visible_avoidance_costs.png`和CSV。未改地图、不覆盖旧数据、不写正式manifest。按用户要求没有运行程序或测试。下一步用户执行新小批量命令，再依据区域分布及额外代价决定是否扩展其他split，不能直接训练。详见设计文档第12节。

2026-09-09 可见性诊断更新（最新）：已读取批次 `batch_pilots/20260909_172002_590211200/`。18对目标只得到12对：wait6、reroute6、avoidance0，主障碍全部为左上门口同一路线。24场配对字段检查无误。avoidance共460次，401次不满足行为，59次通过行为及删除反事实但未通过可见性，尚未检查配对。当前不能训练，也不能直接断言地图无解。

用户要求继续后新增 `scripts/diagnose_office_v10_visibility.py --batch-dir .\outputs\office_behavior_v10_dataset_design\batch_pilots\20260909_172002_590211200`。直接复查现有3个失败例，不重搜数据。oracle新增默认关闭的“首次偏离 nominal 时当前帧主障碍可见”约束，诊断同时限制无等待、原门组合，求此约束下最短安全路径，区别等长可见证据/可见性额外代价/此约束下无解；不能外推到其他参考路线或学习策略。旧正式验收没有切换，只新增证据记录。结果会存该批次下 `visibility_diagnostics/<时间>/`，含summary、左右路径与观察窗口图、完整路径和碰撞回放。未来生成的可见性失败每组额外保留前5例的主障碍索引及决策路径。代码与回归测试已补，按用户要求尚未运行；下一步用户执行上述命令，读实际结果再决定改判定还是动路线。详见设计文档第11节。

2026-09-09 小批量更新（优先于下文旧状态）：用户单对入口已成功，结果在 `outputs/office_behavior_v10_dataset_design/pair_pilots/20260909_115215_557449300/`。已读取 JSON 和 PNG：10 次尝试、搜索 4 秒，wait 最短 71 步且等待 1 次，no-wait 72 步；normal 和删除主障碍后均 70 步。仅改变主障碍 `test_critical_z1_01` 初始索引 4→2。主障碍 demo 冲突 9→0，全场景 demo 冲突 20→15，不能称对照为全部示范安全，也不能据此断言学习策略受示范伤害。

用户要求继续后新增 `--pair-batch-pilot --diagnostic-attempts 300 --pilot-seconds-per-group 60`：在同一测试路线池检查三行为×3生成 seed×2对=18对，使用单对成功的构造式提案，不改地图/路线/语义条件。逐组有界搜索，不足时保留成功候选、重新验证并生成图片，继续其余组；非全部完成退出码2。输出到 `batch_pilots/<时间>/`，含 `batch_summary.png/json`、`verified_pairs.json`、`pair_metrics.csv` 和逐对路径图，保留拒绝原因、尝试区域及接受区域。跨 seed 重复单独审计；不是正式训练数据集，不写 manifest，旧全量搜索尚未替换。按用户要求未运行程序或测试，只做静态检查；下一步让用户运行此小批量命令，再读取真实结果决定是否扩展 train/validation。详见设计文档第10节。

2026-09-09 更新（优先于本节以下旧状态）：已读取用户完成的诊断 `diagnostics/20260908_230601_720718600/test_wait_search.json`。2,000 次尝试耗时 709.73 秒，136 个满足 wait，其中 33 个无因果主障碍、19 个主障碍不可见、84 个配对失败；84 个检查了 336 个替代相位，199 个示范冲突未下降，其余 137 个标称路径仍碰撞，没有合格正常对照。

用户要求继续后，新增 `--find-first-wait-pair --diagnostic-attempts 2000` 单对开发入口：先枚举现有测试路线的相位必要条件，再从全部测试路线组合中构建五障碍场景，仍严格验证 wait、删除主障碍恢复 normal、可见性和相位配对，不再在此入口固定两个等待 anchor。成功只生成一对配对 JSON 和左右预览图，保存到 `outputs/office_behavior_v10_dataset_design/pair_pilots/<运行时间>/`，不生成正式训练清单。每个成功候选立即留存，配对失败前已验证的冲突候选保留前五个；路线预览、预检查和搜索诊断在失败时也留存。按用户要求尚未运行新入口或测试，下一步用户执行此单对命令，先查看产物，再讨论批量扩展。旧全量搜索尚未切换此新提案方式。详见设计文档第 9 节。

最新进展：用户已中止 test wait 至少 25,000 次搜索仍 0/20 的运行。当前下一步是有界诊断，不是重新运行全量生成。已加入 `--diagnose-test-wait --diagnostic-attempts 2000`，仅复现测试等待候选搜索；按场景拒绝、主障碍验证、相位验证分层计数，保存失败样例、耗时、配置和路线池到 `outputs/office_behavior_v10_dataset_design/diagnostics/<运行时间>/`。正式生成先检查测试配对，连续 3,000 次未接受即停止并保存审计。回归测试补充了零进展停止与拒绝原因落盘检查，按用户要求未运行。下一步读取 `test_wait_search.json` 定位到底是删除反事实、可见性还是相位对照条件卡住；不得凭旧混合拒绝数直接放宽设计条件。详见 `docs/OFFICE_BEHAVIOR_PAIRED_V10.zh-CN.md` 第 8 节。

本节取代第 42.5 节中“v6 是当前唯一优先任务”的旧状态；v6/v7/v8/v9 均保留为开发历史，当前先验收 v10 数据集，不直接复用旧结论。

在查阅导航基准和相关文献后，当前地图方案不再二选一地采用“完全随机栅格”或“完全手工固定场景”，而采用：**结构化 Office 静态拓扑 + 受约束随机化的动态路线、方向、速度与相位 + 行为 oracle 验收**。Office 是机制开发图；方法冻结后仍须在其他结构化地图及随机地图族上做外部确认。

新增配置 `configs/dynamic_spatial_generalization_office_behavior_v10_paired_v1.yaml` 和生成器 `scripts/generate_office_behavior_scenarios_v10.py`，保留 v9 作为历史版本。v10 的预定 split 为 100 train、20 validation、60 test：训练为 normal/wait/avoidance/reroute 各 25，验证各 5；测试为 wait/avoidance/reroute 各 10 个冲突场景，并各配一个 normal 对照，共 30 个冲突加 30 个对照。

每场仍有 5 个动态障碍，分别覆盖 5 个拓扑关键区，但不再假设五个障碍都必须同时造成冲突。每个行为场景声明一个主障碍：完整场景必须满足指定的等待、局部避让或全局改道；删除主障碍后，其他四个障碍不变且场景必须恢复为严格 normal；做出首次行为决策时主障碍必须处于 15×15 局部观察范围。主障碍统一存到索引 2，使原课程阶段 `[2] -> [2,3,4] -> [0..4]` 先呈现真正的行为触发障碍，而不是无关障碍。

最终 test 使用相位配对：冲突和对照保持五条路线、四个协变量障碍、运动方向和速度全部相同，只改变主障碍的 `start_index`；对照必须被 oracle 判为 normal，且主障碍对静态 A* 示范的冲突数严格下降。这种成对设计用于估计动态冲突的条件效应，避免把不同地图布局的总体难度误当成 A* 示范影响。它仍不能单独证明 replay 策略的因果机制，结论必须来自三策略、多个训练 seed 的配对差值。

第一次用户运行时，train 三类候选池均生成 50 个，但候选主障碍区域仅出现 `upper_left_gate` 和 `lower_left_gate`，触发全五区覆盖限制。随后将门槛降到两个区域，第二次运行又在 validation 失败：三类各 11 个候选仅出现 `upper_left_gate`。这两条日志只证明有限候选池的分布，不能证明未出现区域不可行。先前将其解释成拓扑必然性的表述撤回。

第二次报错后移除了缺少依据的主障碍区域数量硬门槛。行为、难度配额和逐场 oracle/配对验收继续强制执行；区域覆盖在配额允许时作为选择偏好，候选/选中计数写入 manifest 和 split_summary，并在少于两个区域时标记 `limited_primary_zone_coverage`。该标记要求训练前审查区域集中和 split 差异；生成成功不代表多样性合格。

静态复核同时发现并修复真实索引错误：主障碍尝试过程中交换到索引 2 后，若可见性或配对检查失败，原实现会将重排后的 scenario/masks 传给下一次尝试，而路线选项仍用原始索引。现在每次主障碍尝试都从不可变原始场景与 masks 开始，并校验输出的主障碍索引、路线和区域元数据。新增 `tests/test_office_behavior_v10_generation.py` 覆盖单区域配额选择、可见性失败后重试和配对失败后重试；按用户要求未执行测试或生成器。

生成器会写出 manifest、三 split 全景图、18 幅代表性冲突/对照图，以及 `scenario_dataset_audit.csv`、`paired_test_audit.csv`、`split_summary.json`。训练后的聚合绘图脚本已兼容 60 个测试场景，并新增逐配对/逐 seed 的差值表和 `test_paired_phase_effects.png`。截至本节记录时，修正规则后的生成器尚未再次运行，因此 v10 还没有 manifest hash，也不能声称数据集已经通过审计。应先由用户重新运行生成命令并人工查看预览，再决定是否启动一个 seed 的三策略 pilot；不要直接开始全量训练。

## 43. 2026-09-10 Office practical v11（当前最新，取代 v6/v10 训练入口）

用户明确否决回退到 v6；v6、v9、v10 只保留为开发历史。当前建立新配置 `configs/dynamic_spatial_generalization_office_practical_v11.yaml`、生成器 `scripts/generate_office_practical_scenarios_v11.py`、结果入口 `scripts/plot_office_practical_v11_results.py`、回归检查 `tests/test_office_practical_v11_generation.py` 和设计说明 `docs/OFFICE_PRACTICAL_V11.zh-CN.md`。静态 Office 墙体、起点 `(2,2)`、终点 `(37,37)` 不变；新建的是动态障碍场景数据集，正式 manifest 目标为 `data/dynamic_scenarios/office_40x40_practical_v11.json`，不覆盖任何旧数据。

v11 不再要求每个场景的五个障碍同时满足强因果反事实。每场仍有五个功能路线障碍，分别位于左上门、右上门、左下门、右下门和终点接近段；目标场景只声明一个真正触发行为的主障碍并固定在索引 0，其余四个上下文障碍必须对本场指定 A* 参考路径安全。门洞瓶颈承担 wait/reroute，门前后静态可绕的开阔段承担 avoidance。参考路径从训练实际使用、同 seed 生成的 20 条 randomized-tie A* 示范中逐场选择，不再固定左门 nominal，因此右门只有在真实示范经过时才成为主位置。行为标签只证明存在可观察、可回放的合理选择，不声称该动作是唯一最优策略。

数据配额为 train 100=`normal20/wait30/avoidance30/reroute20`，validation 20=`4/6/6/4`。test 共 50：普通 held-out 场景 30=`6/8/8/8`，另有 wait4、avoidance3、reroute3 共十组冲突/相位对照（20场）；每组只改变主障碍 `start_index`，其他路线、方向、速度和四个上下文障碍不变，且主障碍对20条示范的冲突数必须严格下降。三 split 的完整路线几何和五障碍配置不重复，但允许共享有限的物理门洞；这是同一 Office 拓扑内泛化，不是跨地图泛化。

验收定义：normal 的指定示范原样安全；wait 在同一门组合中使用 STAY 至少节省1步且主障碍在15×15观察窗内；avoidance 要求冲突格静态可绕，并存在可见后偏离、无STAY、同门组合、不重复格的安全路径，至多比全知最优多4步；reroute 要求无等待替代门组合至少改善1步且达到全局最短安全代价。清单会保存指定参考路径、reference index、安全 witness、冲突/观察/代价指标，生成后重新求解并逐步回放校验。配置阈值不再在代码中写死。

为避免 v10 式无界失败，连续2000次没有新增场景即停止；每500次打印进度，具体拒绝原因写入 `outputs/office_practical_v11_design/runs/<时间>/search/`。每次生成运行使用独立时间戳目录，失败证据不会被下一次覆盖。只有全部场景校验、审计和预览成功后才最后写正式 manifest；失败不会留下可被训练误用的半成品。最新状态镜像在 `outputs/office_practical_v11_design/generation_status.json`，只有 `status=complete` 且 `training_ready=true` 才表示机器验收完成。

课程固定为40万环境步：0–6万无动态障碍，6–14万仅索引0，14–24万索引0/1/2，24万后五个全开；验证和测试始终五个。奖励为 step=-0.01、progress=0.05、stay=0、collision=-1、goal=10，碰撞终止，静态非法动作 mask。三策略仍是 uniform/prefill/persistent_demo(25%)，配置 seed 为0/1/2。开始训练前必须人工查看 v11 的路线池、train/validation/test 全景图和分布图；生成成功不等于人工认可。若图不合理应否决数据集；不得因为训练后某策略没有领先而事后改图。

本轮遵照用户要求没有运行生成器、测试或训练，因此当前还没有 v11 manifest、预览、hash 或训练结果，也不能声称地图已经生成成功。下一步只运行一次 v11 生成命令并查看五张预览，不运行 v6、不运行 v10，也不直接开始训练。

2026-09-10 10:51 用户首次运行 v11，在路线池分配阶段立即失败：`No non-overlapping train route for upper_left_gate:approach_before`。失败状态已留存在 `outputs/office_practical_v11_design/runs/20260910_105139_504998100/generation_status.json`，正式 v11 manifest 没有产生。原因不是没有门前候选，而是旧 `_select_route_pools` 按 anchor 贪心：先选出的左上门瓶颈路线占用了门前路线的格子，后续没有回退机制。现已改为跨 train/validation/test 的联合回溯分配：一个区域的瓶颈/门前/门后路线整体检查，保证 split 内路线不重叠且完整路线几何跨 split 不重复；优先长度5，但允许3/7作为兼容替代。新增回归用例覆盖“首选瓶颈挡住门前路线时必须回退”。按用户要求修复后未自行运行程序或测试；当前仍无 v11 manifest，下一步由用户重新执行同一条生成命令。

用户 10:59:20 第二次运行后终端长期无输出。11:04:40 只读检查确认系统中已无 Python 进程；该次目录 `runs/20260910_105920_539071400/` 只有停留在 `running` 的状态文件，没有 placement audit、图片或 manifest，说明进程在正常异常处理前被外部终止。结合实现，最可能原因是联合回溯先把候选笛卡尔积全部物化，造成内存/组合爆炸；这是实现缺陷而非地图无解，但由于没有系统退出记录，不把 OOM 写成已证实事实。现已改为惰性逐组合 DFS，不再保存全组合列表；路线选择最多250,000次候选试探，每深入一个组立即打印，之后每10,000次打印耗时，超过上限正常报错并写 error 状态。修复后未运行程序或测试，正式 v11 manifest 仍不存在。

第三次用户运行完成了有界诊断：258个候选、15组，搜索1.1秒达到250,000次上限，最深进入validation第5区域但反复回退，最终报在 `validation:lower_right_gate:approach_before`；状态在 `runs/20260910_111205_115199500/`，没有manifest。日志证明不是性能卡死，而是全局DFS把空间互不相关的区域做了笛卡尔积回溯。现将路线分配分解为五个独立功能区：每区内部联合求 train/test/validation，仍强制split内路线不重叠、跨split完整几何不复用；区间结束后再次检查所有split的跨区格子不重叠及全局几何唯一。该分解利用anchor区域本来就空间分离的设计，不是放宽标准。修改后未运行程序或测试，下一步仍由用户执行同一生成命令。

第四次用户运行给出可解释的真实不可满足条件：前四区均完成，`goal_approach` 在1114次尝试后报告候选数 `goal_west=1, goal_middle=19, goal_east=27`。旧要求让train/test/validation都使用goal_west，同时又禁止跨split复用完整几何，因此必然无解。现将唯一goal_west路线只保留给train；train仍为west/middle/east各1，validation为middle/east各1，test为middle×2/east×1，所有完整路线继续跨split唯一。该调整让训练覆盖最完整，held-out集合使用充足的不同几何变体，不复制唯一西侧路线，也未取消隔离标准。修改后未运行程序或测试，正式v11 manifest仍不存在。

第五次用户运行已证明路线分配问题解决：258个候选、五个功能区仅用85次试探即完成train/test/validation联合分配。随后train normal生成到8/20后连续2000次全部报`no_reference_safe_combination`。静态复核确认这不是地图或验收条件再次无解，而是普通场景抽样器的顺序错误：它先按全局使用次数选路线，再检查该路线对本场A*参考是否存在安全相位；一个从未被使用但对该参考不可行的路线会永久占据最小使用次数，令有安全相位的其他路线永远没有被检查。现新增“先筛安全相位存在的路线，再在可行路线内按使用次数均衡”的选择器，并同时用于normal五区选择和目标场景的四个上下文障碍。normal搜索前还会输出并限制为“所有五区均至少有一个安全相位”的eligible reference集合；若确实不存在会立即给出明确错误，不再无意义空转。新增回归用例覆盖“低使用但不可行路线不能阻塞高使用可行路线”。地图、路线池、行为oracle和配额均未放宽；按用户要求修改后未运行程序或测试，正式v11 manifest仍不存在。

第六次用户运行中normal阶段已经通过，train wait连续2000次为0/30；拒绝构成为759次同门含等待也无解、502次同门禁止等待无解、464次最优同门路径不含STAY、256次全局无安全路、13次决策不可见、6次同门等待路径非全局最短。关键实现错误是把502个“允许STAY可达但禁止STAY无解”的候选全部拒绝；这与v9既有定义相反，也不符合等待必要性的逻辑。现只在无等待路径存在时检查至少1步优势；无等待无解视为更强的等待证据，但仍要求含STAY见证存在、达到全局最短且决策时主障碍可见。记录新增同门无等待可达布尔值和步数，配套回归测试与文档已更新。没有改地图、路线池、配额或其余验收门槛；按用户要求未运行程序/测试，正式v11 manifest仍不存在。

第七次用户运行说明第六次修复生效但生成策略仍不充分：路线分配85次完成，normal 20/20零拒绝，wait达到2/30后连续2000次无新增；总尝试2119次、1451.891秒，两个合格主障碍都来自`train_functional_01`（左上门）。新的根因是上下文相位只保证不撞原始A*参考，没有保证不撞主障碍导致STAY之后发生时间错位的wait witness，因此会把主障碍本来提供的安全行为路径再次封死或使其他门更优。现对所有target行为引入主障碍单独验收与缓存：只有单个主障碍就能触发wait/avoidance/reroute的相位/参考才保留，无效候选永久移除；四个上下文除了对参考安全，还必须对该单障碍oracle witness安全，完整五障碍场景最后仍按原标准复验。拒绝候选的规划改为按行为按需求解，不再一律计算全部八个门组合，并打印primary proposal数量。该修改增强主障碍因果解释、避免背景破坏见证及重复昂贵尝试，不改地图/配额且未放宽oracle。配套路径安全回归测试和设计文档已更新；按用户要求未运行程序或测试，正式v11 manifest仍不存在。

第八次用户运行首次给出了决定性的地图机制证据：train wait共793个单主障碍候选全部不合格，其中400个同门含等待仍无解、200个同门等待不具全局最优性、193个最优路径不含STAY。这说明旧的“沿一格宽门洞直线上下往返”障碍不会真正离开通道，智能体无法与其错身；它适合长期封门/触发reroute，不适合wait。之前五障碍组合偶然出现的2个wait来自上下文共同堵路，因果上不应保留。现修改的是动态路线蓝图而非静态Office：保留直线门洞路线专供reroute，另为四个门新增“穿过两格门洞后转入门后横向开阔区”的四连通turning路线，使门周期性完全腾空，专供wait；open approach继续供avoidance。训练/验证/测试路线池计数改为19/14/15。同一功能区的这些路线是互斥备选，可共享门洞但每场只选一条；完整路线几何仍不跨split复用。通用空间manifest验证新增turning方向和`alternative_group`支持，并禁止一场同时选同组路线。单主障碍因果预检、witness安全上下文和完整场景复验均保留。未改静态墙体、场景数量或oracle门槛；按用户要求未运行程序或测试，正式v11 manifest仍不存在。

第八次修图后的下一步不应再次直接执行完整生成。生成器原有`--preview-only`模式会只完成路线分配并输出新的`placement_blueprint.png`/公共route pool预览，不搜索场景、不写manifest；应先由用户运行该入口并人工确认点划线wait路线穿门后横向离开。当前公共预览仍可能是上一轮旧图，必须以新preview-only运行产物为准。

用户第一次运行新路线的preview-only时，四个门的wait_clearance候选全部缺失。原因已由静态Office定义直接确认：横墙分别占10/11行与25/26行，第二个门格左右仍是墙；初版turning路线在(11,c)/(26,c)立即横移，必然被`_valid_route`过滤。现路线改为`入口门格 -> 出口门格 -> 墙外净空格(row 12/27) -> 横向侧袋`，尾长2/3/4，总长5/6/7；四连通与静态合法性要求不变。修改后未运行程序或测试，下一步仍只运行preview-only，不执行完整场景生成。

用户第二次preview-only成功：`runs/20260910_134106_684871600/`状态为`preview_complete`，候选279条，route pool计数train/validation/test=19/14/15。每个split均含4条wait_gate和4条reroute bottleneck；open approach为11/6/7。已读取placement audit并目视检查blueprint：所有wait点划线都穿过两格门洞、到达row12/27净空格后横向离开，四个门完整覆盖；三个split的完整路线无重复，有限物理门格共享属预期。preview-only没有生成manifest，`training_ready=false`正确；下一步可以执行完整v11生成，但只有全部170场oracle复验及图片完成、状态变为complete/true后才能训练。

## 44. 2026-09-17：不规则地图与同源分叉适应实验（当前新增主线）

用户要求基于讨论重新设计地图和代码，先检验“持续回放是否增加动态适应代价”，再决定是否改进回放。新增独立协议 `docs/REPLAY_ADAPTATION_V1.zh-CN.md`、配置 `configs/replay_adaptation_v1.yaml`，保留全部旧实验；不要按早期 Office 待办自动恢复旧生成器或训练。

三张 40×40 不规则设备岛地图已生成，map seeds=91701/91702/91703，每图 train/validation/test=18/6/12 对，共216个单动态障碍场景。每对仅改变同一周期运动的相位；control 中参考 A* 原路安全，conflict 中参考原路碰撞且整组20条示范冲突数更高。conflict 具有实际环境回放验证的可见等待与局部绕行后重接参考路径的见证；安全规划不进入训练。三集合完整场景无重复，允许路线几何共享；不能宣称未见路线泛化。地图预览在 `maps/previews/replay_adaptation_v1/`，manifest 在 `data/replay_adaptation_v1/manifest.json`。

新入口 `scripts/run_replay_adaptation.py` 为每个 seed 先训练静态 foundation，合格后完整恢复网络/target/Adam/在线经验/示范/所有 RNG，分叉0%、10%、25%持续回放。0%保留相同在线容量，不是从零训练Uniform。静态最多30万步，每个动态分支固定20万步。保存内容指纹验证各分支起点相同，不再采用动态难度课程。新增固定冲突状态探针与实际动态接触量，区分学习避让与换路未接触。主指标为等步数的冲突安全率AUC；不预设负结果，且即使有代价也不等于已证明梯度干扰。

已完成10项 unittest，包括216场景见证/划分、运动时钟与环境一致、完整恢复、同分支确定性续训、评估不污染训练状态、精确交互/更新预算、失败门控、配对统计方向。CPU 和 CUDA 的16步短流程及汇总均通过，输出严格标 smoke_only。CPU中间版短流程移至 `outputs/replay_adaptation_v1/development_smoke_before_final_cli/` 保留；最终CPU检查在 `outputs/replay_adaptation_v1/smoke/`，CUDA检查在 `outputs/replay_adaptation_v1_cuda_check/smoke/`。

尚未启动正式训练，也尚未证明地图可学习性或持续示范有适应代价。按此前用户偏好提供命令由用户执行。建议先在第一张图跑两seed pilot：`python scripts/run_replay_adaptation.py --map-index 0 --seeds 0 1 --device cuda`；汇总：`python scripts/summarize_replay_adaptation.py --map-index 0 --seeds 0 1`。代码/配置/数据或设备改变时拒绝混用已有结果；具体分叉、输出和判读规则见新协议文档。

## 45. 2026-09-17：动态位置覆盖修正 v2（当前默认）

用户指出第一张图训练动态路线过于集中；审计确认v1三个集合均缺少参考路径后1/3交互，训练18对的前/中/后段计数为5/13/0，位置仅第14–39步。用户要求修改后，新增 `configs/replay_adaptation_v2.yaml`、独立v2数据/预览/输出目录，三张静态地图与训练参数保持不变。准备、训练、汇总脚本现默认v2，原两条终端命令保持不变；显式附加v1配置才能访问旧版。

候选支持短横穿、交汇点转弯和交汇后转弯，并纳入终点接近区域但不占用起终点。以完整66步参考路径划分early=0–21、middle=22–43、late=44–65，三集合分别强制6/6/6、2/2/2、4/4/4对。同一集合决策位置不得重复，位置选择优先拉开距离，硬间隔1步；若任意段配额不可行就失败，不以其他段补数。仍要求control参考安全、conflict参考碰撞且20条示范冲突数增加；等待、局部绕行和时空最短路径全部实际回放。候选类型按位置分别限额，避免短转弯候选挤掉可用直线候选。

三张图均已生成成功，共216场景；训练/验证/测试各18/6/12个不重复位置，第一张训练决策跨度2–64步。预览用橙/紫/青显示前/中/后段，位于 `maps/previews/replay_adaptation_v2/`。详细设计见 `docs/REPLAY_ADAPTATION_V2.zh-CN.md`；逐位置表在 `data/replay_adaptation_v2/coverage_audit.csv`。

14项unit tests通过，新版CPU16步分叉短流程与汇总通过，输出仅在 `outputs/replay_adaptation_v2/smoke/`。未启动正式训练，不将验收或smoke成功解释为学习成功或固定示范有害。旧v1数据、预览和结果保留，不混合统计。

## 46. 2026-09-21：多障碍风险覆盖示范交接 v1

基于两张图正式结果，研究方向改为解决“动态障碍可见但真正高风险经验稀疏”的问题，而不是继续调整全局示范比例。新增独立配置 `configs/risk_handover_v1.yaml`、协议 `docs/RISK_HANDOVER_V1.zh-CN.md`、生成器 `scripts/prepare_risk_handover.py`、回放实现 `RiskCoverageHandoverReplay` 和密度分层汇总 `scripts/summarize_risk_handover.py`。旧 `replay_adaptation_v2` 代码、数据和结果不覆盖。

冻结数据位于 `data/risk_handover_v1/manifest.json`，SHA-256 为 `34E62AE1F01F0D6D7F53AAF8DB663F4D7CF3FD49A5D074EB693F7BBFF00E55B3`。三张不规则地图每图 train/validation/test 为36/12/48对：训练和验证均衡使用3/5个障碍，测试按1/3/5/7个障碍各12对。3障碍含1个因果冲突障碍，5/7障碍含2个，其余使用控制相位；同场路线格互不重叠。288条逐地图/集合审计记录位于 `data/risk_handover_v1/scenario_audit.csv`，预览位于 `maps/previews/risk_handover_v1/`。所有control参考路径和conflict时空安全oracle都已在真实环境逐步回放。当前地图存在等长替代通路，oracle额外步数为0，因此本数据检验参考阻塞后的换路，不宣称严格等待必要性。

新 replay 保持batch=64和普通在线配额48不变。其余16个指导槽初始全部来自静态A*示范；风险事件及此前3步进入容量3000的风险分区，覆盖键为“障碍密度+决策位置”；独立覆盖达到18个键时线性完成从16示范到16风险的交接。训练日志保存风险池大小、覆盖数、交接进度和风险采样量。固定0/10/25、时间衰减和新方法共用同源foundation。

已完成数据生成、CPU 16步 smoke 和完整221项unittest，全部通过；只运行了smoke，没有启动正式训练。正式顺序先运行Map 1 seeds 0/1/2的五方法pilot，并用 `scripts/summarize_risk_handover.py` 汇总；只有风险机制确实触发且配对seed方向一致，才扩展三地图五seed。可直接使用的命令见新协议文档。

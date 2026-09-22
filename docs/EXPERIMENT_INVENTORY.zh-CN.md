# 实验时间线与数据使用索引

更新日期：2026-09-01

本文档只建立索引，不移动、不改名、不删除任何原始结果。正式整理数据时，
应按下文的“正式使用 / 辅助使用 / 排除”标记取数，不能把不同实验协议的
CSV直接合并。

## 一、最终建议保留的证据链

| 层次 | 实验 | 用途 | 结果目录 | 使用级别 |
| --- | --- | --- | --- | --- |
| 静态基础基线 | Static Local Observation Baseline v1 | 证明三种 replay 最终都能完成静态任务，Persistent Demo 主要提高样本效率 | `outputs/random_benchmark/Static Local Observation Baseline v1/` | 正式背景结果（单训练 seed） |
| 静态放大校准 | 40x40 五张结构化地图 | 验证放大地图上三种 replay 的收敛和路径能力 | `outputs/structured_calibration_40x40/` 中 2026-08-11 至 08-12 的前15个正式目录 | 正式辅助结果（每张地图一个训练） |
| 静态多 seed | Map 1、三策略、5个训练 seed | 比较静态任务样本效率的 seed 稳定性 | `outputs/structured_calibration_40x40/` 中 `20260814_203024...` 至 `20260815_050126...` 的最后15个正式目录 | 正式辅助结果 |
| 固定动态场景 | 混合六障碍 Map 1 | 证明固定强交互场景中 Persistent Demo 学得更快；检查等待、碰撞和路径 | `outputs/dynamic_controlled_mixed_six_obstacles_40x40_map01/` 中3个 `trainseed_0` 正式目录 | 辅助案例（只有一个训练 seed） |
| 空间泛化探索 | Map 1 spatial v1 | 首次使用100/20/50场景和空间不重叠路线；暴露最终模型不稳定问题 | `outputs/dynamic_spatial_generalization_map01/`，排除名称含 `_smoke` 的3个目录 | 探索结果，不作为最终主表 |
| 方法选择 | Map 1 checkpoint v2 | 三策略、5 seed；使用验证集选择 checkpoint | `outputs/dynamic_spatial_generalization_map01_checkpoint_v2/` | 正式方法选择结果 |
| 示范比例消融 | Map 1 固定10%与25%到0% | 判断 Persistent Demo 的示范比例是否需要降低或衰减 | `outputs/dynamic_spatial_generalization_map01_demo10_v2/` 与 `outputs/dynamic_spatial_generalization_map01_demo_decay_v2/` | 正式消融结果 |
| 独立确认 | Map 2 confirmatory v2 | 在未参与方法选择的新地图上确认三策略结果 | `outputs/dynamic_spatial_generalization_map02_confirmatory_v2/` | 最重要的正式确认结果 |
| 冲突机制诊断 | Map 1/Map 2 配对相位压力评估 | 审计 split 难度并检验固定25%是否在高冲突下被反超 | `outputs/spatial_conflict_stress_v1/` | Post-hoc 补充证据，不替代正式 test |

## 二、按时间顺序的完整实验记录

### 1. 2026-08-06 至 08-08：20x20随机静态地图原型

- 目的：验证局部观测D3QN、A*示范、Uniform/Prefill/Persistent Demo回放实现。
- 主要改动：20x20、5张随机地图、单通道静态局部观测、1500轮训练。
- 结果：早期目录混有smoke、失败重跑和旧协议；冻结版本另存为
  `Static Local Observation Baseline v1`。冻结版本三策略最终均为5/5成功、
  平均36步、零碰撞；Persistent Demo在101至500轮明显更快。
- 原始目录：`outputs/random_benchmark/`
- 使用规则：只正式使用
  `outputs/random_benchmark/Static Local Observation Baseline v1/`；同级时间戳
  目录均视为开发记录。

### 2. 2026-08-08 至 08-09：20x20单动态障碍原型

- 目的：加入动态障碍、历史帧和四通道观测，验证环境可以训练。
- 主要改动：每张地图1个长度5的往返动态障碍，A*规划忽略动态障碍。
- 结果：三策略最终均5/5成功、平均38.4步、零碰撞，任务过于简单，无法
  区分策略。
- 结果目录：`outputs/dynamic_benchmark/`
- 使用规则：只作为环境实现验证，不放入最终算法对比主表。

### 3. 2026-08-09：单张40x40随机静态地图

- 目的：把地图从20x20放大到40x40。
- 主要改动：窗口仍为11，最大350步。
- 结果：Uniform和Prefill失败，Persistent Demo成功；结果受到单张随机地图
  结构和旧checkpoint协议影响。
- 结果目录：`outputs/single_map_40x40_static/`
- 使用规则：开发诊断，已被结构化40x40实验替代。

### 4. 2026-08-09 至 08-10：五张40x40随机静态地图

- 目的：检查上一步失败是否是单地图偶然现象。
- 主要改动：扩展为5张独立随机地图，并逐步加入best checkpoint。
- 结果：不同地图和重跑之间成功情况不稳定，目录混有smoke和不同checkpoint
  协议；说明随机地图不适合做可控校准集。
- 结果目录：`outputs/five_map_40x40_static/`
- 使用规则：不进入最终主表，只说明为何改用结构化固定地图。

### 5. 2026-08-10 至 08-11：learning starts=2000消融

- 目的：检查Uniform失败是否由过早学习造成。
- 主要改动：只把`learning_starts`从64改为2000，只运行Uniform。
- 结果：5张地图仅1张成功，没有解决问题。
- 结果目录：`outputs/five_map_40x40_static_ls2000/`
- 使用规则：失败诊断，可在方法开发过程简述，不进入策略比较。

### 6. 2026-08-11：20x20结构化校准地图

- 目的：用固定、可解释、可复现的地图替代随机地图。
- 主要改动：5张结构化20x20地图；当时只完整运行Uniform。
- 结果：5/5成功、平均36步，证明结构化地图可用。
- 结果目录：`outputs/structured_calibration/`
- 使用规则：地图校准记录，不是三策略正式比较。

### 7. 2026-08-11 至 08-12：40x40结构化五地图静态比较

- 目的：建立后续动态实验所用的40x40固定地图基线。
- 主要改动：地图40x40、窗口15、最大600步、5张固定地图、三策略。
- 结果：三策略在5张地图上最终全部成功。最佳checkpoint平均轮数约为：
  Uniform 580、Prefill 500、Persistent Demo 220，Persistent Demo学习最快。
- 结果目录：`outputs/structured_calibration_40x40/`
- 正式目录：从`20260811_192414_map_01_seed_1500_uniform`到
  `20260812_022059_map_05_seed_1504_persistent_demo`的前15个目录。
- 使用规则：可作为跨地图静态辅助结果；每张地图只有一次训练，不能当作
  5个同分布训练seed。

### 8. 2026-08-12：Map 1动态障碍校准

- 目的：从静态任务过渡到固定动态任务。
- 第一组只有smoke：`outputs/dynamic_calibration_40x40_map01/`。
- 第二组加入3条基于三策略静态路径的共享穿越路线：
  `outputs/dynamic_calibration_40x40_map01_three_crossing/`。
- 结果：第二组三策略最终均76步、零碰撞，场景仍过于容易，而且路线设计
  参考了各策略已有路径。
- 使用规则：辅助开发记录，不作为公平的最终动态比较。

### 9. 2026-08-13：20x20较难结构化校准

- 目的：增加结构障碍和绕行难度，测试诊断指标与checkpoint逻辑。
- 主要改动：较难20x20地图、每10轮诊断。
- 结果：目录以smoke和重复调试为主；唯一完整三策略组只有Map 1、seed 0。
- 结果目录：`outputs/structured_calibration_hard/`
- 使用规则：开发诊断，不进入最终主表。

### 10. 2026-08-13 至 08-15：Map 1静态多训练seed比较

- 目的：把静态结论从单次训练扩展到5个训练seed。
- 主要改动：固定同一张40x40 Map 1，仅改变网络初始化、探索和replay随机性。
- 结果：三策略最终都成功。最佳checkpoint平均轮数约为Uniform 520、
  Prefill 640、Persistent Demo 200，再次显示Persistent Demo的静态样本效率
  优势。
- 结果目录：仍在`outputs/structured_calibration_40x40/`。
- 正式目录：`20260814_203024...`、`20260814_210818...`、
  `20260814_214557...`以及其后的trainseed 1至4，共15个目录。
- 排除：更早的重复seed 0目录，以及`per`、`dqfd`和名称相近的smoke目录。

### 11. 2026-08-15：随机seed动态“泛化”v1

- 目的：训练100个动态场景，再测试100个新seed场景。
- 主要改动：Map 1、每场景3个动态障碍，训练和测试使用不同生成seed。
- 结果：三策略held-out成功率均100%，但生成器只在同一小组空间路线中改变
  相位和组合，训练/测试共享障碍路线位置；它测试的不是严格未见空间位置。
- 结果目录：`outputs/dynamic_generalization_map01/`
- 使用规则：用于说明“只换seed不等于空间泛化”，不进入最终性能主表。

### 12. 2026-08-15 至 08-16：固定强交互动态场景开发

- `outputs/dynamic_controlled_bottleneck_40x40_map01/`：只有smoke，排除。
- `outputs/dynamic_controlled_six_obstacles_40x40_map01/`：只有smoke，排除。
- `outputs/dynamic_controlled_mixed_six_obstacles_40x40_map01/`：最终完成一组
  trainseed 0正式三策略比较；场景包含3条主路线交互和3条背景路线。
- 混合六障碍结果：训练成功率MA50达到80%所需轮数约为Uniform 306、
  Prefill 489、Persistent Demo 276；首次持续零碰撞greedy约为400、600、
  200轮；最终三策略均安全成功。
- 使用规则：该结果能说明固定场景中的学习速度，但只有一个训练seed，作为
  辅助案例，不能承担统计结论。

### 13. 2026-08-18 至 08-19：Map 1严格空间拆分v1

- 目的：让训练、验证、测试使用空间不重叠且留有一格缓冲的动态路线。
- 主要改动：每场景5个障碍，3条corridor加2条background；100个训练、
  20个验证、50个测试场景；5个训练seed。
- 结果：最终模型测试安全成功率均值为Uniform 76.4%、Prefill 81.6%、
  Persistent Demo 76.4%；Persistent Demo seed 3还出现静态保持失败。
- 结果目录：`outputs/dynamic_spatial_generalization_map01/`
- 排除：3个名称以`_smoke`结尾的目录。
- 使用规则：这是重要的探索结果，但当时测试的是`model_final.pth`，没有用
  验证集选择checkpoint，不能与v2数据直接合并。

### 14. 2026-08-19：Map 1 checkpoint v2

- 目的：修正v1只看最终模型的问题，并统一三策略的模型选择协议。
- 主要改动：每100轮保存checkpoint；按验证集安全成功率、成功率、动态碰撞
  次数和步数选择`model_selected.pth`；测试集只在选模完成后评估一次。
- 结果：训练成功率MA50达到80%约为Uniform 420、Prefill 384、Persistent
  Demo 266轮；整条验证曲线平均安全成功率为66.4%、69.3%、89.8%。Map 1
  测试安全成功率为83.6%、84.8%、76.4%。
- 结果目录：`outputs/dynamic_spatial_generalization_map01_checkpoint_v2/`
- 使用规则：这是正式方法选择数据。选择示范比例时只使用验证结果，Map 1
  测试结果不能反过来调参。

### 15. 2026-08-19 至 08-20：Persistent Demo比例消融

- 固定25%：包含在`dynamic_spatial_generalization_map01_checkpoint_v2/`。
- 25%线性降到0%（200至600轮）：
  `outputs/dynamic_spatial_generalization_map01_demo_decay_v2/`。
- 固定10%：`outputs/dynamic_spatial_generalization_map01_demo10_v2/`。
- 结果：验证末期安全成功率分别约96%、88%、93%；100%安全checkpoint平均
  数量分别为7.0、4.2、6.2；按照预注册验证规则选择固定25%。
- 使用规则：正式消融。测试安全率虽为76.4%、74.0%、84.0%，但不能用已
  查看过的Map 1测试集改选参数。

### 16. 2026-08-21 至 08-22：Map 2独立确认v2

- 目的：在没有参与方法和比例选择的新静态地图上确认结论。
- 主要改动：Map 2重新生成空间不重叠的100/20/50场景；仍使用固定25%
  Persistent Demo；三策略各5个训练seed。
- 结果：每种策略共250次测试。Uniform安全成功234/250（93.6%），Prefill
  235/250（94.0%），Persistent Demo 249/250（99.6%）；三者到达目标均为
  250/250，动态碰撞总次数为18、16、1，静态碰撞均为0。
- 结果目录：`outputs/dynamic_spatial_generalization_map02_confirmatory_v2/`
- 定性案例：
  `outputs/dynamic_spatial_generalization_map02_confirmatory_v2/visualizations/scenario_20032_seed_1_selected/`；
  Uniform 80步2次碰撞，Prefill 82步1次碰撞，Persistent Demo 78步0碰撞。
- 使用规则：这是当前最重要的独立确认结果。

### 17. 2026-09-01：空间冲突审计与配对相位压力评估

- 目的：检验静态 A* 示范是否会在高动态冲突下使固定25% Persistent Demo
  系统性退化，并决定是否值得实现冲突感知自适应比例。
- 主要改动：不重新训练；量化Map 1/Map 2各split的直接路线、nominal到达
  时序冲突和冲突分数；从test路线池选择20组路线组合，每组只改变相位构造
  low/medium/high三档；重放现有selected models。
- 规模：压力评估2400个rollout，另有2400个rollout用于记录“照静态A*下一步
  走会立即碰撞”状态中的A*动作一致率。
- 结果：Map 1固定25%的low/medium/high安全率为59%/71%/47%，均高于同档
  固定10%、Prefill和Uniform；Map 2为99%/100%/100%。没有出现支持全局
  自适应的稳定性能交叉。
- 额外发现：Map 2 validation平均冲突分数42.09、test仅13.15；现有manifest
  保证空间隔离，但没有保证split难度匹配。Map 1压力结果存在很大的训练seed
  分化，部分失败属于未见路线到达失败而非动态碰撞。
- 结果目录：`outputs/spatial_conflict_stress_v1/`
- 中文报告：`docs/SPATIAL_CONFLICT_STRESS_ANALYSIS.zh-CN.md`
- 使用规则：仅作为post-hoc机制诊断和后续协议设计依据，不替换原Map 1/Map 2
  test，不用于回头改选checkpoint或示范比例。

## 三、现代空间泛化实验中每个文件的含义

适用于Map 1 v1、Map 1 v2、比例消融和Map 2确认目录：

- `run_metadata.json`：本次运行的配置、场景清单哈希、汇总指标和选模信息。
- `training.csv`：1500个训练episode逐轮数据；用于学习曲线和样本效率。
- `validation_summary.csv`：每100轮一次的验证汇总；v2中用于checkpoint选择。
- `validation_evaluation.csv`：每个验证checkpoint在20个场景上的逐场景记录。
- `test_evaluation.csv`：选中模型在50个测试场景上的逐场景记录。
- `static_retention_evaluation.csv`：动态训练后的静态路径保持结果。
- `model_final.pth`：第1500轮模型。
- `model_best_validation.pth`：验证规则选出的最佳模型。
- `model_selected.pth`：实际用于测试的模型；v2分析应以它为准。
- `checkpoints/episode_XXXX.pth`：每100轮保存的模型，仅v2提供。

## 四、最终论文/报告取数建议

### 主结果表

只使用：

1. `outputs/dynamic_spatial_generalization_map02_confirmatory_v2/`
2. 每个策略5个seed的`run_metadata.json`和`test_evaluation.csv`

主结论：Persistent Demo在Map 2独立确认集上安全成功率最高且碰撞最少。

### 方法选择与消融表

使用：

1. `outputs/dynamic_spatial_generalization_map01_checkpoint_v2/`
2. `outputs/dynamic_spatial_generalization_map01_demo_decay_v2/`
3. `outputs/dynamic_spatial_generalization_map01_demo10_v2/`

只用验证集解释为何选择固定25%；Map 1测试列可以作为探索结果单列，但不能
写成调参依据。

### 学习效率图

优先使用v2各目录的`training.csv`和`validation_summary.csv`。静态背景图可以
补充使用`Static Local Observation Baseline v1`和40x40 Map 1五seed结果。

### 定性路径图

使用Map 2场景20032的`replay_paths.png`与`replay_rollouts.gif`。单个案例只能
解释行为差异，不能代替250次测试的统计结果。

## 五、明确排除清单

整理正式数据时排除：

- 任何目录名含`_smoke`，或`summary.json`中`"smoke": true`的运行。
- `dynamic_calibration_40x40_map01/`全部运行。
- `dynamic_controlled_bottleneck_40x40_map01/`全部运行。
- `dynamic_controlled_six_obstacles_40x40_map01/`全部运行。
- `structured_calibration_hard/`中的smoke、残缺和重复目录。
- `structured_calibration_40x40/`中的PER/DQfD smoke及重复seed 0调试目录。
- Map 1 spatial v1中的三个`_smoke`目录。
- Map 2的`confirmatory_decay`和`confirmatory_demo10`配置没有正式运行，不得
  当作已有实验结果。

## 六、目前仍需注意的限制

- Map 2的50个测试场景中有30个没有动态路线直接覆盖nominal A*路径，整体
  比Map 1容易；当前结论适用于已定义的空间泛化分布。
- Map 1与Map 2的策略排名不同，不能写成Persistent Demo在所有地图必然最好。
- Map 2是在Map 2训练集上训练、Map 2测试集上评估，证明的是新地图上的
  “地图内未见动态位置泛化”，不是从Map 1直接零样本迁移到Map 2。
- 高交互压力测试已经完成，并明确标记为补充探索实验；它不能替换或回头
  修改已完成的Map 2确认结论。
- 当前压力结果不支持立即训练CA-ADR。下一步应先在新增地图中同时约束空间
  隔离和split冲突难度匹配，并预先比较固定10%与固定25%。

# Astar-D3QN-Workshop

最新完成正式实验：[A*探索、动作学习与TD目标限制，六组×seed0–4，各200000步](results/whole_map_91701_value_repair_v1/analysis_20261007_234317_595719/REPORT.md)。
组合组最终安全成功率[100%,100%,100%,92%,100%]，均值98.4%；“A*探索＋目标限制”组五seed最终均100%。
组合组全程AULC为91.37%±1.05%，相对后者五seed均提高，平均差+30.93个百分点；优势主要是早期效率，不是最终全面胜出。退出附近曾降至58%后恢复，seed3最终仍有四次迎面碰撞。
所有630次检查点epsilon=0，关闭A*，共用50个固定验证场景，不使用test。见[均值与退出期曲线](results/whole_map_91701_value_repair_v1/analysis_20261007_234317_595719/mean_validation_curves.png)。
完整五seed的[GitHub分析包](results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/README.md)包含30次运行日志、全部验证明细、代码快照和失败轨迹；可复制[GPT分析请求](results/analysis_upload/whole_map_91701_value_repair_seeds0to4_20261007_234317_595719/GPT_ANALYSIS_PROMPT.md)。
此前seed0、1的[正式分析](results/whole_map_91701_value_repair_v1/analysis_20261007_110936_677575/REPORT.md)和[GPT分析包](results/analysis_upload/whole_map_91701_value_repair_seeds01_20261007_110936_677575/README.md)继续保留，它们不包含新增seed2–4；[原分析请求](results/analysis_upload/whole_map_91701_value_repair_seeds01_20261007_110936_677575/GPT_ANALYSIS_PROMPT.md)。

此前[六组各20000步的独立pilot](results/whole_map_91701_value_repair_pilot_v1/analysis_20261006_195243_463151/REPORT.md)未覆盖10万步引导退出；其结果与正式实验分开保存。

此前完成的20万步实验：全图路线池、每回合3–5个动态障碍，比较运行时A*路线输入与无引导D3QN。
seed0–4共10组，每组200000环境步；两组均从随机网络开始，不使用示范经验。
最终平均安全成功率为无引导94.4%、有引导95.6%，引导的学习速度收益随seed变化，不能宣称稳定优势。
方法与命令见[路线引导说明](docs/RUNTIME_PATH_GUIDANCE_V1.zh-CN.md)，
完整曲线、失败诊断和核对记录见[五seed分析](results/whole_map_91701_runtime_path_v1/analysis_20261005_094812_262464/REPORT.md)。

GitHub同步范围与本地权重说明见[同步说明](docs/GITHUB_SYNC.zh-CN.md)。

后续[A*训练引导与动态风险交接](docs/TRAINING_HANDOVER_V1.zh-CN.md)已完成seed0：无引导最终安全成功率92%，
持续引导0%、时间退出2%、风险交接0%、匹配介入次数对照0%。seed1未完成，用户已停止该批训练。
训练时的A*代选成功不等于网络自主学会导航；失败轨迹及价值诊断保留在独立结果目录。

当前[价值目标限制与A*动作学习对照](docs/VALUE_REPAIR_V1.zh-CN.md)，保留原物理场景、奖励及探索规则，
分别检查异常Q值与缺少明确动作学习的问题。`python scripts/run_value_repair.py`默认只做检查；
短诊断输出与20万步正式训练分开保存，正式训练必须显式`--train`。短诊断不能证明最终性能优势。

2000步短诊断：seed0、1的动作监督组与组合修正组均92%安全成功、8%动态碰撞、0%超时，
静态任务均66步到达；同样的四个冲突场景仍沿静态路线撞上障碍。见[短诊断分析](results/whole_map_91701_value_repair_v1/analysis_20261005_213205_512226/REPORT.md)。
六组各20000步试验完成seed0、1，200000步正式实验已完成seed0–4。已完成目录拒绝覆盖；没有自动启动额外训练。

以下为之前的实验协议与基础实现介绍。

Earlier independent protocol: [持续回放与动态适应代价实验 v2（交互位置均衡）](docs/REPLAY_ADAPTATION_V2.zh-CN.md).
Irregular workcell maps, phase-paired single-obstacle scenarios, and matched
foundation checkpoints compare continued demo replay at 0%, 10%, and 25%.
Prepare with `python scripts/prepare_replay_adaptation.py`; this does not train.
The default v2 dataset assigns equal early/middle/late interaction quotas and
distinct decision positions within each split. Legacy v1 data remain separate.

Clean research code for studying A* demonstration replay with a canonical
Dueling Double DQN (D3QN). The benchmark uses reproducible 20x20 random
occupancy grids and an agent-centered 11x11 local observation with a normalized
goal-direction vector. A fixed workshop environment will be added after the
replay ablation is validated.

The initial comparison is deliberately limited to:

1. D3QN with uniform online replay.
2. D3QN with one-time A* replay prefill.
3. D3QN with a persistent A* demonstration partition.

Optional comparison baselines are also available through the same training
entry point: proportional prioritized replay (`--strategy per`) and a
controlled DQfD-style demonstration margin baseline (`--strategy dqfd`). See
`docs/replay_baselines.md` for the exact scope and commands.

## Algorithm contracts

- A* expands only four movement neighbors and uses Manhattan distance. Its
  randomized variant changes only tie-breaking, so shortest-path optimality is
  preserved.
- The execution action space has five actions: up, down, left, right, and stay.
- D3QN uses the dueling aggregation `Q = V + A - mean(A)`.
- Double DQN selects the next action with the policy network and evaluates it
  with the target network.
- The local observation contains one spatial obstacle channel and two normalized
  goal-displacement scalars. The CNN encodes the local map and the scalars are
  concatenated before the dueling value and advantage heads.
- Demonstrations are environment transitions, not coordinate lists.

## Quick start

```powershell
python scripts/generate_random_benchmark.py --config configs/random_benchmark.yaml
python scripts/collect_astar_demos.py --config configs/random_benchmark.yaml
python -m unittest discover -s tests -q
python scripts/train_random_benchmark.py --config configs/random_benchmark.yaml --smoke
```

Generated experiment artifacts belong under `outputs/`. The five fixed training
maps are stored together in `maps/random_benchmark/train/maps.json` and are
reproducible from their registered seeds. Prefill and persistent-demo training
both load the same registered A* dataset from `data/demonstrations/`.
Training prints a rolling progress line every 100 episodes by default. Each
line includes recent reward, success and collision counts, followed by a
greedy evaluation over the five training maps. Set
`training.progress_interval` to `0` to disable these reports.

## Multi-obstacle risk handover

The independent candidate method in [多动态障碍风险覆盖示范交接实验 v1](docs/RISK_HANDOVER_V1.zh-CN.md)
keeps the frozen adaptation maps but composes paired 3/5-obstacle training
scenes and a 1/3/5/7-obstacle test sweep.  Its three-part replay progressively
replaces the fixed A* demonstration slots with short pre-conflict online
sequences as distinct risk positions are covered.  Generate and audit the
dataset with `python scripts/prepare_risk_handover.py`; this command never
starts training.

## References

- Hart, Nilsson, and Raphael (1968), A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths.
- van Hasselt, Guez, and Silver (2016), Deep Reinforcement Learning with Double
  Q-learning.
- Wang et al. (2016), Dueling Network Architectures for Deep Reinforcement
  Learning.
- Schaul et al. (2016), Prioritized Experience Replay.
- Hester et al. (2018), Deep Q-learning from Demonstrations.

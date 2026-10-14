# Astar-D3QN-Workshop

最新：[新版完整方法模块消融阅读包](results/analysis_upload/latest_all_eligible_modules_v1/LATEST.md)及[实验与结果说明](docs/LATEST_MODULE_ABLATIONS.zh-CN.md)。新增去TD目标约束、去A*动作代选两组，五seed共10次200000步训练及同一冻结500最终评估全部完成。完整方法全程AULC为92.99±1.36%，两组消融分别为34.86±8.23%、84.97±1.34%；动作监督与代选的前期学习增益均在五seed重复，TD约束明显改善本实现的学习稳定性。撤教低谷仍存在，不主张保证安全或最终性能全面优于对照。源码、配置、逐seed及配对统计、曲线、失败示例和SHA清单均已整理，权重与原始回放留在本地。

最新：[执行一致性筛选五seed阅读包](results/analysis_upload/execution_consistency_seeds0to4_v1/LATEST.md)及[实验说明](docs/EXECUTION_CONSISTENCY.zh-CN.md)。新对照五seed均完成20万步，复用原完整组合，十个最终模型复评原50场景。取消筛选的全程AULC为92.99±1.36%，原组91.37±1.05%，但seed3退化且低谷后移，不能宣称跨seed一致改善。独立500最终确认已完成：原组成功率98.16±1.82%，新组99.36±0.43%，新组仍有12次碰撞和4次超时。包内包含源码、冻结登记、全部曲线、原50与独立500分开的配对统计及失败示例；权重留在本地。

最新：[分阶段撤教与DQfD无预训练的五seed阅读包](results/analysis_upload/replication_seeds0to4_v1/LATEST.md)。两种新增对照的seed0–4均完成200000在线步，原两方法五seed直接复用；包内包含20次运行的曲线、逐场景验证明细、诊断日志、配对差、源码快照、冻结登记及失败示例。[协议及上传说明](docs/REPLICATION_FIVE_SEEDS.zh-CN.md)。

原完整组合全程AULC91.37±1.05%，分阶段92.06±1.41%；分阶段seed3下降、seed4几乎持平，且seed2、3、4撤教最低点更低，不能称撤教稳定改善。原DQfD全程93.24±0.42%，取消预训练90.75±0.47%，五seed全程均下降；两组10–20万AULC分别97.11±0.69%和96.92±0.51%，差异方向随seed变化。无预训练仍保留7260条永久示范，总梯度预算减少20000次。

同一冻结500场景最终确认已完成：原完整组合成功98.16±1.82%，分阶段97.00±4.26%；原DQfD97.88±2.09%，无预训练95.96±2.91%。分阶段seed2仅89.6%，主要是超时，不支持最终稳定优势。复用10份场景/模型SHA及协议一致的原评估，新增10份最终模型评估，共10000条场景记录。[最终报告](results/replication_final500_v1/20261012_164953_660389/REPORT.md)。没有重训、挑最佳检查点或据500结果调参。

以下单seed及三seed材料保留为历史，新增对照的当前结论以上述五seed报告为准。

2026-10-12此前：[对比诊断、稳定性探针与单seed对照阅读包](results/analysis_upload/diagnostics_withdrawal_update_v1/20261012_002943_168455/README.md)，包含已有12方法五seed的证据梳理、DDQNA适配版seed0/2诊断、三个seed0稳定性探针、撤教同状态Q值检查、DQfD取消离线预训练seed0和分阶段撤教seed0的完整运行日志、源码/冻结登记及精选失败轨迹。[GPT分析请求](results/analysis_upload/diagnostics_withdrawal_update_v1/20261012_002943_168455/GPT_ANALYSIS_PROMPT.md)、[该批状态与边界](docs/DIAGNOSTICS_WITHDRAWAL_UPDATE.zh-CN.md)、[上传命令](docs/GITHUB_SYNC.zh-CN.md)。

分阶段撤教（8万步代选退出、10万步监督退出）seed0全程AULC由89.69%提高至93.48%，10–20万步由92.68%提高至97.79%，最低成功率70%→88%；最终两组均100%。只支持该seed低谷减轻，尚未解决撤教不稳，代选概率日程面积也减少了8%。DQfD取消20000次离线更新、其余机制不变的seed0全程AULC为91.08%，原DQfD对应seed0为92.80%；最终50验证成功率96%与100%。两个新对照均仅seed0完成，尚无独立500最终确认评估。

DDQNA探针中TD约束抑制数值增长，但三组5万步自主成功率仅0%、0%、2%，不构成导航稳定改善或原论文优劣证据。短探针AULC已更正为0–5万步积分：1.60%、1.20%、0.20%；先前误外推到20万步的短探针汇总不再作为有效结果。

以下保留此前DDQNA及十一方法五seed实验结果。

此前[DDQNA动作选择机制适配版五seed阅读包](results/analysis_upload/whole_map_91701_ddqna_adapter_seeds0to4_20261010_112856_785796/README.md)，包含新增五次200000步训练、两方法十个最终模型的同批500场景自主评估、运行日志、源码快照、成本和失败轨迹。[GPT分析请求](results/analysis_upload/whole_map_91701_ddqna_adapter_seeds0to4_20261010_112856_785796/GPT_ANALYSIS_PROMPT.md)、[机制核对与适配差异](docs/DDQNA_ADAPTER_EXPERIMENT.md)。

完整组合全程AULC91.37±1.05%，DDQNA动作机制的D3QN适配版0.94±1.19%；最终500场景自主成功率分别98.16±1.82%和0%。适配版失败主要是绕圈/等待超时；这不是DDQNA原版完整复现，不能将本任务的适配结果推广为原论文无效。逐seed及配对差值见阅读包；不据最终500调参或选择检查点。上传方式见[同步说明](docs/GITHUB_SYNC.zh-CN.md)。

以下保留此前十一方法及历史实验结果。

最新主线十一方法×seed0–4全部完成，共55次200000在线步正式训练。原50场景的1155个自主验证检查点及20286条失败轨迹已核对；55份最终模型完成同一批新冻结500场景的27500次自主评估，全部3688条最终失败轨迹重新核对。epsilon=0，无A*代选，不选最佳检查点；新500仅作最终确认，未调参。

最新[GitHub阅读包](results/analysis_upload/whole_map_91701_eleven_methods_seeds0to4_20261009_183634_447770/README.md)包含20次新增运行日志、源码与配置快照、监督覆盖率、成本表、55条学习曲线、194条精选完整失败轨迹及SHA清单。可复制[GPT分析请求](results/analysis_upload/whole_map_91701_eleven_methods_seeds0to4_20261009_183634_447770/GPT_ANALYSIS_PROMPT.md)。逐seed、均值±样本标准差及配对差值见[完整训练汇总](results/whole_map_91701_dqfd_comparison_v1/analysis_20261009_173936_175389/REPORT.md)和[新500最终确认](results/whole_map_91701_dqfd_comparison_v1/independent_eval_20261009_174147_641239/REPORT.md)。

原组合全程AULC91.37%±1.05%；DQfD93.24%±0.42%，DQfD＋TD限制93.17%±0.60%，监督延期93.38%±1.90%，取消风险筛选90.42%±3.86%。DQfD多了20000次离线更新和7260条永久安全示范，包含198个风险等待标签；在线教学量与原组也不同，不能声称等总成本下胜出。新500最终成功率最高均值为DQfD＋TD限制99.20%±0.77%，原组合98.16%±1.82%。

监督延期组新500最终成功率96.84%±2.43%，五seed后6万步AULC均降低、四seed存在低谷后移迹象；不能称交接问题已解决。取消风险筛选组新500成功率98.84%±1.32%，不能主张筛选提高最终安全性。研究继续限定为固定地图、同起终点、3–5动态障碍中的自主学习效率；没有跨地图或最终全面最优结论。新增[协议与DQfD适配说明](docs/DQFD_COMPARISON_V1.zh-CN.md)、[尾段协议](docs/SUPERVISION_TAIL_V1.zh-CN.md)和[上传命令](docs/GITHUB_SYNC.zh-CN.md)。原结果保留，权重与批量原始轨迹留在本地。

此前七方法结果（旧500场景）：[七方法五seed复核报告](results/whole_map_91701_teaching_efficiency_v1/review_20261008_113945_236854/REPORT.md)。新增`supervision_only_bound`的seed0–4各训练200000步，复用原六组，35份最终模型完成同一批冻结500场景评估；源码、配对初始化、模型SHA与全部3496条独立失败轨迹通过复核。无代选＋动作监督＋TD限制的全程AULC为83.02%±4.07%，相对无A*＋TD限制提高34.66个百分点；完整组合组91.37%±1.05%，再提高8.35个百分点，主要收益集中在前5万步。500场景最终安全成功率分别为97.84%±1.75%和98.68%±1.08%，A*代选＋TD限制为99.12%±1.17%。退出低谷、等待超时和局部动态碰撞仍存在，继续定位为学习效率提升。协议与历史命令见[实验说明](docs/TEACHING_EFFICIENCY_V1.zh-CN.md)。

此前七方法[GitHub阅读包](results/analysis_upload/whole_map_91701_teaching_efficiency_seeds0to4_20261008_113945_236854/README.md)包含新增五seed日志、源码快照、38条精选完整失败轨迹及完整结果链接；可复制[GPT分析请求](results/analysis_upload/whole_map_91701_teaching_efficiency_seeds0to4_20261008_113945_236854/GPT_ANALYSIS_PROMPT.md)。

原六组正式实验（保留对照）：[A*探索、动作学习与TD目标限制，六组×seed0–4，各200000步](results/whole_map_91701_value_repair_v1/analysis_20261007_234317_595719/REPORT.md)。
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

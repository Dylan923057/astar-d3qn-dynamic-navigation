# 五方法多任务比较：TD约束单独控制与循环诊断

本轮2026-10-29 18:21完成。新增D3QN＋TD约束seed0/1，每次500000在线步、499501更新，无预训练/示范/教师；旧四方法8次运行复用。没有重复训练，没有启动CE、监督下限扫描、五seed、正式消融或两个测试池。

| 方法 | 成功率seed0/1 | 平均成功 | 平均碰撞 | 平均超时 | 全程AULC均值±样本SD |
|---|---:|---:|---:|---:|---:|
| 原始D3QN | 2% / 26% | 14% | 20% | 66% | 5.88% ± 4.95% |
| D3QN＋TD约束 | 26% / 36% | 31% | 11% | 58% | 15.99% ± 2.50% |
| DQfD | 12% / 20% | 16% | 16% | 68% | 14.94% ± 1.02% |
| C 持续弱监督 | 76% / 72% | 74% | 13% | 13% | 47.14% ± 1.78% |
| D 风险自适应 | 74% / 64% | 69% | 3% | 28% | 44.17% ± 0.07% |

TD约束单独令平均成功提高17个百分点，AULC提高10.11个百分点；在其基础上C再提高43个百分点、31.15个百分点。后者只能解释为A*代选＋持续动作监督的教学组合增益，不能归于持续监督单项，也不能把代数差分比例当成独立因果贡献。

## 阅读顺序

1. [最终发现](results/asvc_multitask_td_control_v1/FINAL_FINDINGS.zh-CN.md)、[完整报告](results/asvc_multitask_td_control_v1/REPORT.zh-CN.md)。
2. [逐seed与分阶段AULC](results/asvc_multitask_td_control_v1/five_method_per_seed.csv)、[均值/样本SD](results/asvc_multitask_td_control_v1/five_method_mean_sample_sd.csv)、[TD/教学组合差值](results/asvc_multitask_td_control_v1/controlled_effects_per_seed.csv)。
3. [五方法学习曲线](results/asvc_multitask_td_control_v1/five_method_learning_curves.png)、[PDF](results/asvc_multitask_td_control_v1/five_method_learning_curves.pdf)。
4. [循环与长度分档](results/asvc_multitask_td_control_v1/loop_seed_summary.csv)、[完整1000条配对成绩](results/asvc_multitask_td_control_v1/all_paired_tasks.csv)、[典型案例索引](results/asvc_multitask_td_control_v1/typical_cases.csv)和typical_cases图。
5. [重复观测状态](results/asvc_multitask_td_control_v1/repeated_observation_states.csv)、[观测闭环](results/asvc_multitask_td_control_v1/recurrent_observation_cycles.csv)、[回放目标/教师审计](results/asvc_multitask_td_control_v1/replay_goal_teacher_audit.csv)、[Q/TD及成本](results/asvc_multitask_td_control_v1/learning_and_cost_audit.csv)。
6. [独立冻结与复现命令](docs/MULTITASK_TD_CONTROL.zh-CN.md)、[CE/大间隔未执行方案](docs/TEACHER_LOSS_COMPARISON_PLAN.zh-CN.md)。

## 数据口径与文件

- `runs/d3qn_bound/seed_0或1/`：新增两次完整51点曲线、各2550条验证记录、训练回合、原始/裁剪TD和Q/梯度日志、各32000条TD目标样本、运行登记与完成记录。
- `previous_four_methods/`：上一轮完整阅读包按原字节嵌入，含原8次运行、共同输入、3472条DQfD示范、原始源码及冻结。其manifest不改。旧报告中C/D动态缺失描述是旧轮记录，本轮以新的50/50轨迹为准。
- 本包合计10条训练曲线、510个检查点、25500条验证记录；反复使用同一50任务，不是25500个独立任务或训练seed。
- 新诊断版本目录包含全部20条件、1000份完整轨迹、所有超时及固定参考任务的逐步五动作Q、合法掩码、教师排名/Q差、硬筛查、风险和事后真实冲突。动态原成绩保留，C/D补录184个缺失轨迹，新增控制静态100回合；不是1000次新增评估。
- C/D最终80000条保留回放未发现目标/静态教师/风险存储错配，仅覆盖最终缓冲区。反复出现相同观测、Q及贪心动作的闭环解释执行行为，不能单独确定训练原因。实际评估监督系数始终0，所有C/D权重是事后假设，真实未来只作离线误差分析。
- 静态配对也改变输入分布，不等于单独静态训练。D保持探索方法，不因碰撞低就升级为主方法。
- `data/asvc_multitask_td_control_v1/`：新冻结、原始训练源快照、补录授权、版本化诊断协议及清单。`code/`是本轮代码/测试/禁用CE计划；共同网络/环境源码在previous_four_methods/code，完整仓库中均在原位置。
- `model_index.csv`记录10个最终模型和完整断点的路径/SHA，本体保留本地。无权重不能重跑网络评估，只有权重也不等于完整续训。

DQfD额外20000离线更新/seed、永久3472示范仅成功覆盖70/240训练任务和31/40目标，生成15107环境步成本单列；不因为成绩低就断言Q爆炸。仅两seed验证，不宣称正式显著性或未见目标泛化。

## 只读校验

仅下载本资料目录后，进入目录运行（只需Python标准库，无PyTorch、GPU或权重）：

```powershell
python tools/verify_bundle.py --verify --package .
```

完整仓库根目录：

```powershell
python scripts/package_td_control_evidence.py --verify
# 仅原工作区具有原日志、模型和完整断点时：
python scripts/package_td_control_evidence.py --verify --check-sources
```

校验SHA、原冻结、10次预算、25500条验证、均值/样本SD、AULC及受控差值、64000个原始/裁剪目标、20组完整配对与离线权重语义。不会训练、运行策略、联网、暂存或推送。

## 上传范围

upload_scope.json限定本批代码、输入、文档和阅读包；旧包保留。模型、回放、缓存、无关README/历史轨迹、论文草稿、转换工具和其他未提交实验不混入。upload脚本只在用户自行执行时提交/推送，不强推，不清空范围外暂存。具体命令见docs/TD_CONTROL_GITHUB_UPLOAD.zh-CN.md。

# 固定任务同教师 CE 对照：五 seed 完整阅读包

**CE 与原持续弱监督大间隔方法各五个 seed，均完成 200000 环境步。** 本轮只新增 CE seed2、3、4，合计600000环境步、598503次更新；原CE seed0、1及原大间隔五seed直接复用。不是把本轮新增训练写成五次。

## 核心结果

| 指标（均值±样本SD，n=5训练seed） | 大间隔 m=0.8 | CE T=1 |
|---|---:|---:|
| 前5万步AULC | 85.64% ± 1.25% | 84.26% ± 2.38% |
| 全程AULC | 95.63% ± 0.38% | 94.03% ± 0.95% |
| 后10万步AULC | 99.54% ± 0.34% | 98.14% ± 0.93% |
| 后半程最低成功率 | 96.80% ± 1.79% | 92.40% ± 4.34% |
| 最终安全成功率 | 100.00% ± 0.00% | 99.60% ± 0.89% |
| 后半程碰撞AULC | 0.22% ± 0.15% | 0.64% ± 0.53% |
| 后半程超时AULC | 0.24% ± 0.29% | 1.22% ± 0.55% |

观测：五个seed的全程及后半程AULC，CE均低于配对大间隔组。均值差分别为−1.59、−1.40个百分点（由未四舍五入值计算）。CE seed4后半程最低86%，最终恢复100%；seed2的后半最低值98%高于原组96%，所以不能声称每个稳定性指标都更差。CE最终五seed合计249/250成功、1/250碰撞、无超时；原组250/250成功。这些是同50场景的五模型重复验证，不是独立250任务测试。

结论限于本协议：没有得到将当前主方法换成CE的依据，保留大间隔监督，CE作为同教师监督形式对照。波动差异的具体梯度原因尚未确定，不宣称普遍优势、显著性或收敛证明，也不以交叉熵本身宣称创新。

## 阅读顺序

1. [详细报告](results/asvc_fixed_ce_five_seeds_v1/REPORT.zh-CN.md)、[五seed协议](docs/ASVC_FIXED_CE_FIVE_SEEDS.zh-CN.md)、[原两seed协议](docs/ASVC_FIXED_CE.zh-CN.md)。
2. [逐seed指标](results/asvc_fixed_ce_five_seeds_v1/per_seed_metrics.csv)、[均值/样本SD](results/asvc_fixed_ce_five_seeds_v1/mean_sample_sd.csv)、[逐seed配对差](results/asvc_fixed_ce_five_seeds_v1/paired_seed_differences.csv)、[配对差汇总](results/asvc_fixed_ce_five_seeds_v1/paired_mean_sample_sd.csv)。
3. [均值学习曲线](results/asvc_fixed_ce_five_seeds_v1/learning_curves.png)、[逐seed曲线](results/asvc_fixed_ce_five_seeds_v1/individual_seeds.png)、[Q/损失诊断](results/asvc_fixed_ce_five_seeds_v1/value_loss_diagnostics.png)。同目录保留PDF。
4. [合并210个验证点](results/asvc_fixed_ce_five_seeds_v1/learning_curve_points.csv)、[10500条逐场景结果](results/asvc_fixed_ce_five_seeds_v1/validation_details.csv)、[各组训练成本](results/asvc_fixed_ce_five_seeds_v1/training_cost.csv)。
5. [新增CE seed2–4原始日志](outputs/asvc_fixed_ce_five_seeds_v1/)、[复用CE seed0、1](outputs/asvc_fixed_ce_v1/)、[大间隔seed0–2](outputs/asvc_persistent_teaching_v1/supervision_floor/)、[大间隔seed3、4](outputs/asvc_persistent_supervision_five_seeds_v1/supervision_floor/)。全部训练CSV、Q范围、TD/监督损失、标签覆盖、代选预算及全部失败轨迹均保留。
6. [冻结及来源](data/asvc_fixed_ce_five_seeds_v1/freeze.json)、[冻结50场景](data/asvc_fixed_ce_five_seeds_v1/fixed_validation_scenarios.json)、[13项预检](results/asvc_fixed_ce_five_seeds_v1/preflight.json)、[完成状态](results/asvc_fixed_ce_five_seeds_v1/queue_status.json)、[全部模型身份](model_index.csv)。
7. [先前两seed报告](results/asvc_fixed_ce_v1/REPORT.zh-CN.md)、[其原上传清单](history/two_seed_manifest.json)及[本轮上传说明](docs/ASVC_FIXED_CE_FIVE_GITHUB_UPLOAD.zh-CN.md)。旧阅读包本身仍在仓库，未覆盖。

## 实验与统计边界

仅监督形式不同：合法动作上的CE，T=1；同一A*教师、风险筛选、碰撞标签排除、真实执行动作TD、代选100k退出、监督下限0.1、TD目标[-6,10]和epsilon日程。网络、观测、奖励、回放和更新预算相同，没有扫描参数、离线预训练或风险软权重。

所有方法共用原50冻结场景及顺序，每万步跨界后的回合末自主验证，无教师代选/风险保护/更新，使用200k最终模型。AULC按实际步数积分、边界线性插值。10500条记录=10个训练结果×21个验证点×50场景，统计单位是五个配对训练seed。均值曲线阴影是样本SD，不是置信区间。

扩展在查看原两seed后由用户授权，未依据结果调参或挑选seed。没有运行CE最终500场景，没有把重复验证冒充独立测试。与500k多任务CE提案分开。预检24步恢复单元不混入正式训练统计，历史与本轮成本分别列出；不同线程/负载下的墙钟耗时不作为效率优势证据。

## 字节保存与独立校验

原始文件按字节复制，所有失败轨迹采用 `.json.gz` 无损压缩，解压内容与原JSON的SHA256一致。本地原文件未删除。所有中间/最终模型和完整回放检查点保留本地，包内只登记路径、大小和哈希；阅读包不能代替模型重跑策略评估或续训。

进入本包目录，只需Python标准库，无PyTorch/GPU即可复核原始记录、全部失败轨迹、AULC、均值/样本SD、配对差与哈希：

```powershell
python tools/verify_bundle.py --verify --package .
```

在原工作区可以用 `python scripts/package_fixed_ce_five_evidence.py --verify --check-sources` 额外验证当前本地原文件和模型。原两seed包仍可独立校验；本轮仅对共享Git属性文件追加五seed路径规则，旧包的严格本地来源校验会反映这一元数据变化，旧科学记录和冻结训练代码未改。其他旧研究的文档链接可能需要完整仓库，本轮全部数据与图表均在本包。

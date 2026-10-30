# 固定起终点：同教师在线 CE 对照阅读包

**两次 CE 训练已完成，seed 0、1 各 200000 环境步，从头训练。原持续弱监督大间隔组直接复用。** 本项是固定起终点小论文对照，与 500000 步多任务 CE 提案及动态冲突预测预实验分开。

## 结果

以下是两个训练 seed 的均值 ± 样本标准差，均来自反复使用的原冻结 50 场景验证池，不是独立最终测试。

| 指标 | 原大间隔 m=0.8 | CE，T=1 |
|---|---:|---:|
| 前 5 万步 AULC | 85.56% ± 0.88% | 83.68% ± 1.85% |
| 全程 AULC | 95.54% ± 0.06% | 93.72% ± 0.46% |
| 后 10 万步 AULC | 99.75% ± 0.07% | 98.00% ± 0.99% |
| 后半程最低成功率 | 98.00% ± 0.00% | 93.00% ± 1.41% |
| 最终安全成功率 | 100.00% ± 0.00% | 99.00% ± 1.41% |
| 后半程碰撞率 AULC | 0.10% ± 0.14% | 0.70% ± 0.28% |
| 后半程超时率 AULC | 0.15% ± 0.21% | 1.30% ± 0.70% |

CE seed0 最终成功 49/50、碰撞 1/50；seed1 最终成功 50/50。后半程最低成功分别为 94%、92%。原组两个 seed 最终均为 50/50，后半程最低均为 98%。所有失败证据均保留，没有只选成功检查点。

**观测：本协议下 CE 的全程、后半程表现更低，后半程碰撞及超时更多。暂不替换当前大间隔主方法。**

原因假设：损失形式的替换会改变更新及后续采样轨迹，但现有两个 seed 不能确定具体梯度原因，也不能证明普遍优势、显著性或收敛性。CE 与 margin 的损失尺度不同，不能凭标量损失大小评价优劣。创新性判断：这是同教师监督形式的控制实验，采用交叉熵本身不作为创新。

## 阅读顺序

1. [实验报告](results/asvc_fixed_ce_v1/REPORT.zh-CN.md)、[协议与运行命令](docs/ASVC_FIXED_CE.zh-CN.md)。
2. [逐 seed 指标](results/asvc_fixed_ce_v1/per_seed_metrics.csv)、[均值/样本 SD](results/asvc_fixed_ce_v1/mean_sample_sd.csv)、[配对差](results/asvc_fixed_ce_v1/paired_seed_differences.csv)、[配对差均值/样本 SD](results/asvc_fixed_ce_v1/paired_mean_sample_sd.csv)。
3. [学习曲线](results/asvc_fixed_ce_v1/learning_curves.png)、[Q 与损失诊断](results/asvc_fixed_ce_v1/value_loss_diagnostics.png)，同目录含 PDF。
4. [CE 原始日志](outputs/asvc_fixed_ce_v1/)、[复用原组完整日志](outputs/asvc_persistent_teaching_v1/supervision_floor/)。每组每 seed 有 21 次验证、1050 行逐场景记录；合计 4200 行不是 4200 个独立训练样本。训练 CSV、Q 范围、TD/监督损失、标签覆盖和代选记录均在对应 seed 目录。
5. [冻结 50 场景](data/asvc_fixed_ce_v1/fixed_validation_scenarios.json)、[冻结配置与源码身份](data/asvc_fixed_ce_v1/freeze.json)、[预检结果](results/asvc_fixed_ce_v1/preflight.json)、[训练成本](training_cost.csv)、[模型和检查点索引](model_index.csv)。
6. [上传说明](docs/ASVC_FIXED_CE_GITHUB_UPLOAD.zh-CN.md)、[最终队列状态](results/asvc_fixed_ce_v1/queue_status.json)。

监督仅把合格标签上的大间隔换为静态合法动作上的 CE（T=1）。教师、风险筛选、碰撞标签排除、独立教师标签/真实执行动作 TD、监督下限 0.1、100k 后代选退出、TD 目标 [-6,10]、epsilon 日程保持原样。没有离线预训练、风险软权重、路径输入、额外 seed、参数扫描或最终 500 场景选参。

AULC 用真实环境步数积分，50k/100k 边界线性插值；验证沿用历史协议，在跨越每 10000 步后的回合边界执行。没有把旧检查点伪装成恰好整数万步。

## 保存与复核

原始科学文件按字节复制。所有失败轨迹以 `.json.gz` 无损压缩，解压内容与原文件 SHA256 一致，本地原文件未删除。全部模型（含中间步骤）及完整优化器/回放/RNG 检查点只登记路径、大小、SHA256，不上传二进制；阅读包足够独立核查统计，不能代替这些本地模型重跑策略评估或续训。

`training_cost.csv` 分列原组历史成本与 CE 本轮成本，预检的 24 步恢复单元检查不混入正式统计。两组墙钟耗时受线程及当时机器负载影响，不据此宣称计算效率优势。旧文档中其他研究/模型的链接依赖完整仓库；本轮图表、原始记录、冻结输入、代码快照与失败轨迹均包含在包内。

进入本包目录，仅需 Python 标准库（无需 PyTorch/GPU/模型）即可核对哈希、4200 行记录、全部失败轨迹、AULC、均值/样本 SD 和配对差：

```powershell
python tools/verify_bundle.py --verify --package .
```

在原仓库根目录可运行 `python scripts/package_fixed_ce_evidence.py --verify --check-sources`，额外核对本地所有原文件和模型身份。下载包时只运行上面的独立校验，不运行训练脚本。复现训练需要完整仓库及原组登记的本地模型/检查点，原命令禁止覆盖完成结果。

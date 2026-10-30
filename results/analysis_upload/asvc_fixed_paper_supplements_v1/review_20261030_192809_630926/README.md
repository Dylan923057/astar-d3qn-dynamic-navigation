# 固定起终点补实验：CE最终500确认与QDagger任务适配

两项补实验均于2026-10-30完成，源结果和旧冻结不覆盖。本轮只有QDagger seed0、1两次新训练，各20000离线更新＋200000在线环境步；CE五个已完成最终模型新增2500回合确认，原大间隔组2500回合确认直接复用。没有50万步多任务研究或其他新增训练。

## 核心结果与阅读顺序

|500场景确认，5训练seed|大间隔|CE|
|---|---:|---:|
|安全成功率（均值±样本SD）|99.28%±0.73%|98.32%±1.48%|
|动态碰撞率|0.72%±0.73%|1.36%±1.34%|
|超时率|0|0.32%±0.72%|

CE−大间隔安全成功配对差−0.96±1.99个百分点；seed0–2更高、seed3–4更低，不能说每个seed都由大间隔胜出。CE共有42次失败（34碰撞、8超时），原组18次碰撞。

|50场景重复验证，仅seed0/1|全程AULC均值±样本SD|最终安全成功|
|---|---:|---:|
|当前持续弱监督大间隔|95.54%±0.06%|100%|
|DQfD|93.10%±0.43%|100%|
|原始D3QN|31.56%±5.76%|91%±1.41%|
|QDagger任务适配版|75.90%±3.79%|90%±5.66%|

QDagger最终seed0/1成功86%/94%，碰撞均2%，超时12%/4%；全程和后半段波动明显。早期0%后来恢复，退出教师附近再回落；时间相邻不是因果证明。数值日志有限，不将成绩低直接归于Q爆炸。

1. [合并总报告](results/asvc_fixed_paper_supplements_v1/REPORT.zh-CN.md)、[CE500详细报告](results/asvc_fixed_ce_final500_v1/REPORT.zh-CN.md)、[QDagger四方法报告](results/asvc_fixed_qdagger_v1/REPORT.zh-CN.md)。
2. CE的[逐seed](results/asvc_fixed_ce_final500_v1/per_seed.csv)、[均值/样本SD](results/asvc_fixed_ce_final500_v1/mean_sample_sd.csv)、[逐场景配对](results/asvc_fixed_ce_final500_v1/paired_scene_differences.csv)；5000条原/新模型逐场景记录与全部失败可核验。
3. QDagger的[逐seed](results/asvc_fixed_qdagger_v1/per_seed_metrics.csv)、[均值/样本SD](results/asvc_fixed_qdagger_v1/mean_sample_sd.csv)、[配对差](results/asvc_fixed_qdagger_v1/paired_per_seed.csv)、[四方法学习曲线](results/asvc_fixed_qdagger_v1/learning_curves.png)，保留PDF和所有真实步数。
4. [原文机制—本次实现—适配差异及复现命令](docs/ASVC_FIXED_QDAGGER.zh-CN.md)、data/asvc_fixed_qdagger_v1中的冻结、官方源码/许可证、107回合示范审计与原7260条训练示范。
5. outputs/asvc_fixed_qdagger_v1中的两个完整训练日志、220个更新诊断区间/seed、三步样本、预训练前和在线0步验证、全部失败轨迹；outputs/asvc_fixed_ce_final500_v1为新确认的全部原始结果。
6. [完成及完整恢复审计](results/asvc_fixed_paper_supplements_v1/evidence_audit.json)、[五seed训练/评估身份补核](results/asvc_fixed_paper_supplements_v1/extended_identity_audit.json)、[16个最终模型身份](model_index.csv)。权重和完整回放断点留在原工作区；另有excluded_model_artifacts.csv记录排除的模型/恢复文件SHA。

## 复用与统计边界

previous_ce_five是上一轮完整CE五seed阅读包，按原字节嵌入，其历史“500尚未评估”仅描述上一轮，本轮以新的CE500报告为准。旧两seed/五seed包本身不覆盖。原始D3QN/DQfD各seed0、1的完整CSV和全部验证失败轨迹也已纳入，不重新训练。

共16条200k学习曲线、336个检查点、16800条50场景重复验证记录；另有5000条500确认记录，及QDagger预训练前100条结果单列。不是16800个独立任务或训练seed。CE是5训练seed配对确认；QDagger是2训练seed预实验，没有运行QDagger最终500，不混合两种统计口径。

QDagger复用原DQfD的7260条示范（107/128成功回合，生成10756环境步），不增加示范。每seed额外20000离线更新，在线199501更新，总219501；教师查询、损失/Q/TD范围和耗时单列。学生三步Double-DQN＋Huber，T=0.1，确定性教师one-hot，λ(t)=3max(1−t/120000,0)。不加本方法的代选、TD裁剪、监督下限、风险软权重或DQfD附加损失。这是任务适配，不能声称原Atari完整复现；与主方法的差值是组合协议差，不能归于单一模块。

预检与中断恢复单元单列，保留未完成预检尝试，不算正式训练。无非有限数值失败，没有换seed或按结果改参数，不宣称正式统计显著性或跨地图泛化。原50最终验证与独立冻结500确认分开；500不是训练/调参依据。

## 数据压缩与独立校验

所有validation_failures、failure_trajectories、CE批次缓存、预训练前详情JSON都按原字节无损gzip。manifest同时记录原始SHA与压缩文件SHA；没有抽样删失败。可用Python标准库gzip解压。权重不能凭本包重建；重跑网络评估需要对应原模型，完整续训需要原完整断点。

仅下载本目录，进入目录后验证（只需Python标准库，无PyTorch/GPU/权重）：

```powershell
python tools/verify_bundle.py --verify --package .
```

完整仓库根目录：

```powershell
python scripts/package_fixed_paper_supplements.py --verify
# 仅保存全部原始文件/模型的原工作区：
python scripts/package_fixed_paper_supplements.py --verify --check-sources
```

验证核对SHA、原冻结、所有逐场景和失败、实际步数AULC、均值/样本SD、配对差、更新预算、蒸馏退出与恢复审计记录，不运行训练、策略评估或联网。upload_scope.json限定本批上传范围；旧根README、其他研究、模型/回放/缓存不混入。具体上传命令见[上传说明](docs/ASVC_FIXED_SUPPLEMENTS_UPLOAD.zh-CN.md)。

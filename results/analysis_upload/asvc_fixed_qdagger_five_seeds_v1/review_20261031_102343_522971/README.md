# 固定任务QDagger五seed与最终500确认：完整证据包

全部于2026-10-31 00:59完成。本轮仅新增QDagger seed2、3、4，原seed0、1及其他方法复用。原冻结、模型、原始日志和旧阅读包不覆盖；不混入50万步多任务研究。

## 结果与阅读顺序

|同一500场景，5训练seed|安全成功均值±样本SD|动态碰撞|超时|
|---|---:|---:|---:|
|持续弱监督大间隔|99.28%±0.73%|0.72%±0.73%|0|
|CE|98.32%±1.48%|1.36%±1.34%|0.32%±0.72%|
|QDagger任务适配|94.04%±3.92%|1.28%±0.58%|4.68%±4.03%|

QDagger五个seed成功率88.0%、92.8%、98.2%、94.8%、96.4%，2351/2500安全成功、32次动态碰撞、117次超时。相对大间隔安全成功配对差−5.24±3.43个百分点，相对CE−4.28±4.37个百分点。主要失败是超时；不能仅据这些成绩确定循环/等待的训练原因。

|原50重复验证，5训练seed|全程AULC均值±样本SD|后10万步AULC|
|---|---:|---:|
|持续弱监督大间隔|95.63%±0.38%|99.54%±0.34%|
|CE|94.03%±0.95%|98.14%±0.93%|
|DQfD|93.24%±0.42%|97.11%±0.69%|
|QDagger任务适配|78.92%±4.43%|85.06%±6.53%|
|原始D3QN|29.08%±17.45%|55.57%±32.21%|

本固定任务与适配配置下，大间隔学习过程和最终表现较好；不是单项机制因果结论，也不宣称正式统计显著性或原Atari完整复现。

1. [完整报告](results/asvc_fixed_qdagger_five_seeds_v1/REPORT.zh-CN.md)、[五seed均值曲线](results/asvc_fixed_qdagger_five_seeds_v1/learning_curves.png)、[逐seed曲线](results/asvc_fixed_qdagger_five_seeds_v1/individual_seeds.png)，同目录保留PDF。
2. [25个逐seed指标](results/asvc_fixed_qdagger_five_seeds_v1/per_seed_metrics.csv)、[均值/样本SD](results/asvc_fixed_qdagger_five_seeds_v1/mean_sample_sd.csv)、[学习配对差](results/asvc_fixed_qdagger_five_seeds_v1/paired_per_seed.csv)。
3. [500确认逐seed](results/asvc_fixed_qdagger_five_seeds_v1/final500_per_seed.csv)、[500均值/样本SD](results/asvc_fixed_qdagger_five_seeds_v1/final500_mean_sample_sd.csv)、[500配对差](results/asvc_fixed_qdagger_five_seeds_v1/final500_paired_per_seed.csv)、[7500条原/新逐场景](results/asvc_fixed_qdagger_five_seeds_v1/final500_scene_details.csv)、[5000条逐场景配对差](results/asvc_fixed_qdagger_five_seeds_v1/final500_paired_scenes.csv)。
4. [仅种子/路径/评估元数据变化的扩展协议](docs/ASVC_FIXED_QDAGGER_FIVE_SEEDS.zh-CN.md)、[预检](results/asvc_fixed_qdagger_five_seeds_v1/preflight.json)、[完成/完整恢复审计](results/asvc_fixed_qdagger_five_seeds_v1/completion_audit.json)、[25个最终模型身份](model_index.csv)。
5. data/asvc_fixed_qdagger_five_seeds_v1保留冻结配置、112个源码文件/快照、50/500输入、旧组只读副本与原7260条示范成本。新seed2–4的训练、Q/TD/蒸馏、三步样本及全部失败在outputs/asvc_fixed_qdagger_five_seeds_v1；原seed0/1和旧四组在prior或对应原目录。

## 口径与复用

25条训练曲线=5方法×5seed，525个实际验证点、26250条反复使用同50场景的记录。另有QDagger预训练前250条单列；7500条500确认=3方法×5模型×500场景，其中本轮新增QDagger2500回合，原大间隔/CE5000回合复用。样本SD以训练seed计算，不把场景数当成独立训练次数；50验证与500确认分开。

本轮新增预算为600000在线环境步、60000离线更新、598503在线更新。五个QDagger累计1000000在线步、100000离线更新、997505在线更新。每seed严格使用200000步最终模型，不选最佳中间模型。原7260示范（107/128成功回合，生成10756环境步）不增加；目标裁剪/代选/持续监督下限不引入QDagger，λ在120000步退出，T=0.1。

扩展是在看到原两seed成绩后由用户授权，未据500调参、筛seed或改预算。原两seed和所有旧结果保留。prior按原字节保留上一轮资料，其中“两seed”及“QDagger不跑500”的文字是旧轮边界；本轮以当前五seed协议和报告为准。

## 压缩、完整性与复现

大体积失败、批次缓存和预训练前JSON按原字节无损gzip，不抽样删除失败；manifest记录原始SHA和压缩SHA。权重、完整回放/优化器断点及恢复测试二进制留在原工作区，excluded_model_artifacts.csv登记路径/大小/SHA。只加载权重不等于完整续训，缺少原模型无法重跑网络评估。

只下载本目录，进入目录后即可标准库校验，无PyTorch/GPU/权重：

```powershell
python tools/verify_bundle.py --verify --package .
```

完整仓库根目录：

```powershell
python scripts/package_fixed_qdagger_five_evidence.py --verify
# 原工作区仍保存全部原文件、模型和断点时：
python scripts/package_fixed_qdagger_five_evidence.py --verify --check-sources
```

校验从原始逐场景数据复算成功/碰撞/超时、实际步数AULC、训练seed均值/样本SD、配对差，核对所有失败长度和结局、冻结/预算/教师退出及恢复审计；不训练、不运行策略、不联网。来源映射记录复用的历史文件位置，根目录提供完整当前源码与新实验数据。

GitHub仅提交upload_scope.json列出的本批内容，模型/回放/缓存、根README及其他研究改动不混入。见[限定上传命令](docs/ASVC_FIXED_QDAGGER_FIVE_UPLOAD.zh-CN.md)。

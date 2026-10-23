# ASVC持续弱监督：五seed结果上传包

[结果报告](summary/REPORT.md) · [主对比图](summary/01_learning_curves.png) · [逐seed曲线](summary/02_individual_seeds.png) · [逐seed数据](summary/per_seed.csv) · [三seed方案选择历史](selection_three_seeds/REPORT.md)

五个seed各训练20万步：seed0–2复用先前结果，seed3–4新增复核。A*代选在10万步退出，监督权重降到0.1后保持；评估由网络独立行动。
后半程AULC由94.58%提高到99.54%，各seed后半程最低成功率的平均由73.60%提高到96.80%；最终500成功率由99.36%变为99.28%。主要改善是训练稳定性，最终成功率略低0.08个百分点。

## 文件说明

- `summary/`：五方法×五seed的报告、CSV及六组PNG/PDF/SVG图。
- `selection_three_seeds/`：三个候选方案的原三seed完整比较，保留选择依据。
- `run_logs/`：31次运行的曲线、逐场景验证和诊断；11次持续教学运行另含完整训练CSV。
- `final500/`：五方法×五seed共12500条评估明细，新方法2500回合新评估、四个对照10000回合复用已核对结果。
- `registration/`、`inputs/`、`source_snapshot/`：冻结协议、50/500场景与对应代码；`model_index.csv`记录最终模型路径和SHA。
- `manifest.json`：逐文件SHA及来源；`upload_scope.json`限定此次Git提交范围。

统计单位为训练seed。前三个seed参与了方案选择，后两个用于复核；固定地图及同一冻结500场景不代表跨地图泛化。旧消融和密度实验仍属于原版。
权重、回放恢复文件和批量失败轨迹保留本地。克隆仓库后可阅读全部图表并用下面命令校验；重新评估需要原权重，不能把缺权重当成自动重训的理由。

仓库根目录只读校验：`python scripts/package_persistent_supervision.py --verify`
本地从原始输出重画：`python scripts/review_persistent_supervision_five_seeds.py`（需要原始outputs及模型）。
上传命令见仓库`docs/GITHUB_SYNC.zh-CN.md`的持续弱监督条目。原训练冻结覆盖全部src文件，因此也保留其中两个奖励实验模块的源码字节；本包不包含奖励实验结果，也没有改变本方法奖励。

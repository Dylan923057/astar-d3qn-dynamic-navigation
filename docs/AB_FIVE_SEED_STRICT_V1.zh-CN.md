# A/B 三地图严格 5-seed 验证协议

## 目的

本实验只比较 A=`time_decay` 与 B=`global_prediction`，验证 B 在三张冻结地图上的增益是否稳定。实验保持 validation-only，不生成、读取或使用 test 数据进行方法选择。

## 冻结范围

- 地图：`irregular_workcell_91701`、`irregular_workcell_91702`、`irregular_workcell_91703`。
- seeds：`0/1/2/3/4`，A/B 使用同一逐-seed foundation 快照。
- foundation 最大预算：300k；动态适应预算：200k；验证间隔：10k。
- time-decay 示范比例：1–50k 为 25%，50,001–100k 为 10%，100,001–200k 为 0%。
- batch size=64，learning rate=0.0003，gamma=0.99，target sync=250，hidden dim=256。
- adaptation epsilon：0.30→0.05，衰减 150k。
- B：prediction loss weight=0.1，pos weight=20，global spatial weight=1.0，prediction head hidden dim=128。
- reward、网络、replay 容量、场景划分和随机种子规则均由冻结配置与代码指纹锁定。
- 不运行 `decision_aligned_prediction`，不增加新地图或其他模块。

既有 seed 0/1/2 的 A/B 运行来自不同代码/配置指纹，不能与本协议混用，因此本次在独立输出根目录中严格重跑全部 5 seeds。预检会登记代码、配置和数据清单哈希；后续任一项变化都会中止。

## 执行顺序

1. `preflight`：只登记并检查冻结协议，不训练。
2. `foundation`：为三张地图生成当前冻结版本的 5 个 foundation。
3. `time_decay`：完成 A 的 15 次 validation-only 适应训练。
4. `global_prediction`：只有 A 全部通过协议检查后才运行 B 的 15 次 validation-only 适应训练。
5. 分析归档：严格检查 30 次运行后生成 raw data、论文图、统计表和报告。

每一步可安全重新执行：完整且指纹一致的结果会被识别；不完整目录或指纹不一致会直接停止，禁止静默覆盖或混用。

## 统计与产物

- 每张地图分别计算 5 个 paired seeds 的 validation conflict safe-success AUC。
- 报告 A/B mean±sample std、逐 seed AUC、Δ=B−A、B>A 数量、平均 paired difference、100,000 次 bootstrap 95% CI，并附精确 paired sign-flip test。
- 跨地图结果采用地图等权汇总；bootstrap 只在每张地图内部重采样 seed，不把三张地图的 seeds 混成一个大样本。
- 轨迹图的普通代表场景固定为 seed 0、每图 early/late 中按 ID 排序的第一个 5 障碍 conflict 场景；map03 退化图使用预先写入代码的确定性规则，并输出选择审计文件。
- 归档目录：`results/paper_evidence/10_ab_five_seed_strict_validation/`，其下明确区分 `raw_data/`、`figures/`、`tables/`、`analysis/`。
- 图片同时输出 350 dpi PNG 与矢量 PDF；表格输出 CSV、XLSX、Markdown 与 LaTeX。

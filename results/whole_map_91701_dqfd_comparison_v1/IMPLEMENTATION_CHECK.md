# DQfD与风险消融实现检查

本次仅冻结示范、固定配置并完成必要检查，没有启动正式训练，没有评估新500场景表现，正在运行的尾段训练未被停止或修改。

## 冻结数据

- 128次固定训练场景尝试，107次安全到达，保留7260条转移；另21次完整轨迹丢弃。未按模型验证结果调整尝试次数或教师。
- A/B所有seed使用同一份`data/whole_map_91701_dqfd_comparison_v1/demonstrations.npz`，SHA256 `8863cf5fdea2520cd78fa391856c129dabf3cbb5eedc5bc46ecc9290bb42545b`，222777字节；二值观测转uint8前逐元素检查，无损恢复float32。
- 示例生成10756次A*查询、66次实际缓存未命中搜索；生成耗时5.463258秒，A*查询计时0.051154秒。共享成本只计一次，不重复加到各seed总成本。
- 新500场景沿用尾段实验训练前已冻结的批次，SHA256 `051c083a8a185ac1738ad71debd3508cca7a20020319009eabe132062c130318`；7个示范候选与已冻结验证集合的组合/物理身份冲突，仅做身份排除，不查看表现。
- 独立配置、实现源码、示范和最终评估场景摘要于新增正式训练前注册。旧源码及原实验配置和结果未修改。

## 检查通过

- 新方法19项标准库unittest测试；原尾段17项与原value-repair16项回归，共52项通过，无新增pytest依赖。
- 15个预训练前基础D3QN状态（三方法×五seed）与历史配对初始化一致，包括online/target网络、Adam、更新数和探索RNG；预检本身0个梯度更新。
- 检查单步和多步完整TD目标先相加再clip，包含终止回报；A没有clip，网络Q不裁剪；静态mask下Double-DQN选择与target评估分离。
- 检查永久示范不会被FIFO覆盖、PER来源bonus/采样概率/IS权重、重复索引优先级、确定性采样、多步终止/timeout/回合边界及尾部刷新。
- 检查在线自采样没有专家监督，离线示范有单步、多步、大间隔和L2四项损失。
- 检查C代选和标签同时取消观测风险判据，执行匹配和碰撞标签排除保留，原10万步退出日程及epsilon=0无教师调用保留。
- A/B各使用冻结真实示范进行2次离线更新及8在线环境步（5次在线更新）短接入；C通过原trainer/CollectionDiagnostics进行8环境步、5次更新。均为CPU独立临时对象，无正式输出run、无正式训练/预训练任务、无500场景性能评估。
- 汇总入口复用现有原50场景明细和失败轨迹，已核对原组合与尾段seed0的21次检查点和全程AULC，未重新评估模型。
- 最终评估检查保证55个完整最终模型审计发生在创建输出和rollout之前；模型缺失时拒绝评估。

详细机器记录：`check_20261008_165522_338811/check_report.json`、`bounded_integration.json`与`bounded_no_risk_integration.json`。

## 运行

尾段正在运行的原任务继续原命令，避免重复启动。完成后依次执行：

```powershell
python scripts/prepare_dqfd_comparison.py
python scripts/run_dqfd_comparison.py --seeds 0 1 2 3 4
python scripts/run_dqfd_comparison.py --train --seeds 0 1 2 3 4 --methods dqfd dqfd_bound advice_bound_margin_no_risk
python scripts/analyze_supervision_tail.py
python scripts/summarize_dqfd_comparison.py --analyze
python scripts/summarize_dqfd_comparison.py --evaluate
```

固定A/B每seed20000次离线预训练+199501次在线更新、200000在线环境步；C无离线预训练。A/B共用冻结示范，但不承诺预训练后权重相同。DQfD适配差异及参数依据完整写于`docs/DQFD_COMPARISON_V1.zh-CN.md`。

## 待完成/数据限制

检查时尾段seed3/4尚未全部完成；新增A/B/C共15个正式run尚未启动，统一分析和最终确认需等待全部55模型齐全。历史方法没有保存的耗时/A*调用数无法准确补回，成本表相应项留空；不重新训练旧方法来补齐。

# 四方法多任务结果：GitHub资料说明

本批次对应2026-10-27完成的多任务外部基线与配对诊断，入口为[最新完整阅读包](../results/analysis_upload/asvc_multitask_external_v1/LATEST.md)。原C/D阅读包保留，不覆盖。当前主方法仍为持续弱监督ASVC，D不升级；所有追加训练都未启动。

## 已整理的内容

- 8条多任务训练曲线：新增原始D3QN、DQfD各seed0/1，复用C/D各seed0/1，均500000在线环境步。
- 408个验证检查点、20400条逐任务验证记录、逐seed/均值/样本SD、全程与分阶段AULC、末段与长度分档表现。
- 同一50任务的静态/动态配对成绩、固定参考冲突组、全部可用轨迹和执行后离线教师/风险/真实冲突案例。配对800条中的动态400条复用，静态400条新增；旧C/D动态全轨迹仅各4/50，缺失循环指标为NA。
- 后期学习趋势、原始与5点后向平均曲线、每10万步窗口统计，以及尚不能证明50万步充分或加步数必然改善的说明。
- 新旧冻结、注册源码与原始快照、报表修正登记、全部示范尝试和成功覆盖、140923字节示范NPZ、运行/更新/监督日志及模型SHA索引。
- 旧固定任务unguided_bound证据单列，CE/大间隔和监督下限方案禁用，仅说明剩余缺口。

主比较的最终成功均值为C 74%、D 69%、D3QN 14%、DQfD 16%；全程AULC为47.14%、44.17%、5.88%、14.94%。C在当前预算有明显数值优势，D降低碰撞同时增加超时；两seed验证不能当正式显著性或未见目标泛化。

## 校验、预览和上传

在仓库根目录执行。前两条只读，不训练、不跑策略、不改索引：

```powershell
python scripts/package_multitask_comparison.py --verify --check-sources
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_multitask_comparison_github.ps1 -Preview
```

由用户执行提交推送，不使用代理时省略ProxyUrl：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_multitask_comparison_github.ps1 -ProxyUrl http://127.0.0.1:7897
```

脚本核对现有origin/main、完整包与来源SHA，只处理upload_scope.json列出的路径；范围外已暂存文件会触发停止，保持原样，不清空、不强推。范围外的README改动、旧轨迹、论文草稿、Gemini包、奖励/风险迁移实验和文档转换工具均不混入。

GitHub上传可直接采用这个资料目录，也可运行限定范围脚本；不建议git add整个工作区。无须重新打包、训练或评估。仅下载资料目录后，运行`python tools/verify_bundle.py --verify --package .`即可用标准库校验全部数据统计，无需PyTorch或模型。

## 本地保留与复现边界

模型、完整回放断点、缓存和其他旧实验不上传也不删除。model_index.csv保留8个最终模型及完整checkpoint的路径/SHA；源码与已有统计可以审阅，读取结果无需权重。重新运行网络评估需要对应模型；完整续训需要策略、目标、优化器、回放/PER、n步队列、随机状态和当前环境状态，不能只加载权重。

离线示范每DQfD seed多20000更新，生成15107环境步成本另列；成功仅覆盖70/240训练任务、31/40目标，不能隐去。上传冻结测试池定义不代表已经评估它们。资料整理和上传脚本均不会触发新训练、策略评估或预算扩展。

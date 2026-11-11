# ASVC 多地图补实验：GitHub 证据包

快照时间：2026-11-11T09:50:18.145414。**六方法主对比90/90完成；消融7/10完成；总计97/100。** 缺项和中断不计零分，不用不完整组冒充五seed结果。

1. 先看 [主要结论与完整表格](summary/REPORT.zh-CN.md)，重点是**前5万步AULC及最终三项性能**。ASVC在三图的早期AULC均值最高，但91706有最终成功率/超时的折中，不能宣称全面无损。
2. 图表：[早期AULC](figures/01_primary_early_aulc.png)、[学习曲线](figures/02_learning_curves.png)、[最终500三指标](figures/03_final500_outcomes.png)、[三张新地图](figures/new_static_layouts.png)。PNG及PDF均保留，图表只由数据绘制，没有生成或补造评估点。
3. `summary/`保存逐seed、每图均值/样本SD、配对差和跨图宏平均。`outputs/asvc_multimap_layout_v1/`保留每次已完成运行的原始CSV无损gzip、最终500逐场景及**全部最终失败轨迹**。主对比45000次最终测试，另加当前已完成消融的测试。
4. `data/asvc_multimap_layout_v1/`保留冻结地图、候选与拒绝原因、完整路线池、50/500场景、程序式训练序列定义/前缀、本图示范及预算、95个冻结文件的源码快照。`configs/`、`scripts/`、`src/`、`tests/`保留相关实现及继承配置。原冻结完全不变。
5. `reference/`仅复用原图证据与密度汇总；不把旧回放、50万步多任务或零样本跨地图评估混入新布局训练统计。

## 校验和重算（只需Python标准库）

在本包根目录执行：

```powershell
python tools/evidence.py --verify
python tools/evidence.py --recompute --output ../recomputed_multimap
```

验证会检查完整文件列表/SHA、原冻结、运行身份、精确交互/更新预算、全部最终500场景顺序、失败轨迹、51个实际验证点及其50场景记录，并从原始记录重算AULC、均值、样本SD、配对差和跨图宏平均。重算写到包外，避免破坏归档。

CSV/JSON的无损gzip可直接用Python gzip模块打开；manifest同时登记原文件与压缩文件SHA。本目录`.gitattributes`保留字节，`.gitignore`明确允许冻结示范NPZ进入Git。**不要把压缩ZIP本身加入普通Git提交**；提交解压后的此目录即可。

## 模型、恢复与补充附件

`model_index.csv`登记127个原图/新图最终模型，`excluded_model_artifacts.csv`登记已完成新运行的权重和完整恢复文件。二进制均留在原工作区，包中没有模型；仅凭本证据包不能重新做网络推断或完整续训。只加载权重不是完整恢复。

反复50验证的全部失败轨迹另存 `validation_failures_20261111_095018_145414.zip`（本地同级目录，SHA和成员清单在provenance），可作为GitHub Release附件，避免普通仓库膨胀；最终500失败已经在主包内，没有抽样删除。未完成运行的可变日志不进入此快照。

原工作区完整恢复/继续缺项的实际命令：

```powershell
& D:\softwares\Anaconda3\envs\astar-drl\python.exe -X utf8 -u scripts/multimap_layout_supervisor.py --run --workers 2
```

该命令只应在原工作区使用，自动跳过身份一致的完成项、完整恢复中断项；已有调度器运行时文件锁拒绝重复队列。训练源码不修改，NumPy1.26的梯形积分名称兼容仅发生在汇总进程。原9项预检包含旧模型保护核对，证据包未带这些模型；需要重做完整训练预检时须提供原模型与其历史路径。本包的标准库证据校验与统计重算不依赖它们。

结论限定为**同一生成规则下不同静态布局的独立训练验证**，每图固定起终点并从头训练；不代表未训练地图的零样本泛化，也不宣称统计非劣效或正式显著性。

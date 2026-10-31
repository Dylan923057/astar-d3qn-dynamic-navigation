# 上传固定任务QDagger五seed与最终500结果

[最新完整阅读包](../results/analysis_upload/asvc_fixed_qdagger_five_seeds_v1/LATEST.md)汇总五方法各五seed的25条训练曲线、525个实际验证点、26250条原50场景重复验证结果，以及大间隔/CE/QDagger三方法的7500条500确认记录。原两seed、旧冻结和旧资料包保留。

本轮仅新增QDagger seed2、3、4，各20000离线更新＋200000在线步/199501在线更新；seed0、1复用。新增在线预算600000步、离线60000更新、在线598503更新。五个最终模型的2500回合确认全部完成，大间隔和CE原5000回合核对模型/场景SHA后复用，没有新增其他训练或多任务实验。

500确认安全成功：大间隔99.28%±0.73%，CE98.32%±1.48%，QDagger94.04%±3.92%。QDagger比大间隔低5.24个百分点，149次失败包括117次超时与32次碰撞。原50验证全程AULC：大间隔95.63%，CE94.03%，DQfD93.24%，QDagger78.92%，D3QN29.08%。两类口径分开，结论限于当前固定任务适配配置，不宣称正式显著性或单项原因。

## 上传

在仓库根目录先做只读预览，不联网、暂存、提交、训练或运行策略：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_qdagger_five_github.ps1 -Preview
```

由用户执行提交和推送到现有origin/main：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_qdagger_five_github.ps1
```

默认直连；仅开启代理时加`-ProxyUrl http://127.0.0.1:实际端口`。脚本自动寻找Git/Python，只暂存upload_scope.json指定路径，不使用git add整个工作区。根README、根.gitattributes的原有修改及其他研究不混入；范围外暂存内容原样保留并触发停止。远端不一致时停止，不自动合并或强推。

只为本包的冻结示范NPZ增加Git忽略例外，上传脚本仅接受明确列出的原7260条示范路径。模型、完整回放/优化器断点、恢复测试二进制和压缩ZIP留本地，不纳入Git提交；model_index.csv与excluded_model_artifacts.csv保留其路径/大小/SHA。

## 完整性与复现

全部失败轨迹、批次缓存和预训练前JSON无损gzip，不抽样删失败。manifest记录原字节SHA、压缩SHA和来源映射。prior内保留上一轮原字节资料；其中“两seed/不跑QDagger500”描述旧轮，本轮以新的五seed报告为准。

```powershell
# 完整仓库或克隆后的标准库校验
python scripts/package_fixed_qdagger_five_evidence.py --verify
# 仅具备全部原始文件、模型与断点的原工作区
python scripts/package_fixed_qdagger_five_evidence.py --verify --check-sources
# 仅下载阅读包：进入review_...目录，无PyTorch/GPU/权重
python tools/verify_bundle.py --verify --package .
```

校验复算逐场景成功/碰撞/超时、失败长度与结局、真实步数AULC、训练seed均值/样本SD及配对差，核对冻结、更新预算、教师退出和完整恢复审计记录。不重新训练或评估。重新网络评估需要原模型；完整续训必须恢复原完整断点，不是仅加载权重。

局部Git属性仅追加本批规则，保留冻结字节；旧包内部未改，旧包--check-sources可能提示本地Git属性的后续追加，科学数据和训练源码不因此变化。本次整理不执行commit或push。

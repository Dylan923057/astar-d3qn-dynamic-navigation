# 上传固定任务 CE 五 seed 结果

本批包含两组各五seed完整对比：本轮新增CE seed2、3、4各200000环境步；CE seed0、1及原大间隔五seed复用。原两seed包保留，不覆盖，不重新训练。

[五seed阅读包入口](../results/analysis_upload/asvc_fixed_ce_five_seeds_v1/LATEST.md)包含：冻结配置/50场景、完整代码快照、13项预检及完成状态、原始训练和诊断CSV、10500条逐场景验证记录、210个验证点的全部失败轨迹、配对统计、PNG/PDF、训练成本和模型身份。所有失败轨迹无损压缩为`.json.gz`；模型/完整回放检查点留在本地，登记路径、大小、SHA256。

结果如实保留：五seed全程AULC，CE94.03%对大间隔95.63%；后半程98.14%对99.54%。CE没有显示替换主方法的依据。50场景反复验证不能当作独立测试，未运行CE最终500场景。

在PowerShell执行预览（不联网、不暂存、不提交）：

```powershell
Set-Location D:\Asatr-D3QN-Workshop
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_ce_five_github.ps1 -Preview
```

预览通过后，提交并推送至现有仓库`origin/main`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_ce_five_github.ps1
```

默认直接连接GitHub，不再默认指定之前不可用的7897代理。仅当你明确启用了代理时，才加`-ProxyUrl http://127.0.0.1:实际端口`。脚本自动寻找Git和astar-drl Python，无需先激活conda。

只暂存`upload_scope.json`列出的本轮扩展代码、数据、阅读包，以及为其追加的局部Git属性规则，不使用`git add .`。根目录README、根`.gitattributes`的已有修改、其他研究和旧轨迹修改不混入。范围外已暂存内容原样保留并触发停止；远端与本地有冲突时停止，不自动合并或强推。推送因网络失败可在远端仍为本次提交父节点时用同一命令重试。

只读校验：

```powershell
python scripts/package_fixed_ce_five_evidence.py --verify
# 原工作区还保存全部原文件和模型时：
python scripts/package_fixed_ce_five_evidence.py --verify --check-sources
# 只下载阅读包，进入review_...目录，无需PyTorch/GPU：
python tools/verify_bundle.py --verify --package .
```

校验从原始逐场景数据复算成功/碰撞/超时，核对全部失败轨迹，按真实步数复算AULC及五训练seed的均值/样本SD、配对差，并核对10500行合并表、模型身份与哈希。不会训练或运行策略。

本轮仅对`configs/scripts/docs/tests`下现有`.gitattributes`追加五seed文件规则，确保Windows下载后冻结哈希仍可验证；旧两seed阅读包内部字节保持不变且可独立校验。旧包的`--check-sources`还检查当时的Git属性文件，因此当前工作区追加规则后会报告该元数据变化；这不代表历史训练文件变化。原训练冻结和科学记录均保持原样。

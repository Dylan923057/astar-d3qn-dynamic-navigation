# 固定起终点 CE 对照：上传 GitHub

本批只整理已完成的 200000 步同教师在线 CE 对照：CE seed0、1 从头训练，原持续弱监督大间隔 seed0、1 直接复用。不启动任何训练，不改变 ASVC 主方法登记，与 500000 步多任务 CE 提案分开。

[完整阅读包入口](../results/analysis_upload/asvc_fixed_ce_v1/LATEST.md)包含代码及上游依赖快照、冻结配置/50 场景、原始训练和诊断 CSV、四条完整验证曲线、4200 条逐场景记录、全部失败轨迹、均值/样本 SD、配对差、PNG/PDF、预检、完成状态及模型身份。失败轨迹采用 `.json.gz` 无损压缩，解压后与原文件字节一致。所有模型和完整回放检查点保留本地，仅上传路径/大小/SHA256。

结果标题应如实写为“固定任务同教师 CE 两 seed 对照”：CE 全程 AULC 为 93.72%，原组 95.54%；后半程为 98.00% 对 99.75%。本轮没有得到替换主方法的证据。两 seed 为探索性结果，原 50 场景是重复验证池，未使用最终 500 场景。

在 PowerShell 中执行（脚本自动查找现有 Git 和 astar-drl Python，无需先激活 conda）：

```powershell
Set-Location D:\Asatr-D3QN-Workshop
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_ce_github.ps1 -Preview
```

预览通过后上传至现有 `origin/main`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_ce_github.ps1 -ProxyUrl http://127.0.0.1:7897
```

不使用本机 7897 代理时，省略 `-ProxyUrl http://127.0.0.1:7897`。命令会提交和推送，本次整理阶段只运行预览，尚未提交或推送。

脚本仅暂存阅读包 `upload_scope.json` 列出的本轮代码、数据和属性文件，不使用 `git add .`。工作区现有 README 修改、其他研究文件、旧轨迹修改和论文文件不会混入。范围外已暂存内容会保留并触发停止，不清空暂存；远端有新提交或本地有无关未推送提交时停止，不自动合并或强推。推送因网络失败后，可以在远端仍为本次提交父节点时重试同一命令。

为了保留原冻结文件的换行字节，本批使用局部 `.gitattributes`；根目录既有 `.gitattributes` 及旧阅读包不修改。下载后的阅读包支持独立标准库校验：

```powershell
# 原仓库，检查包内统计及哈希：
python scripts/package_fixed_ce_evidence.py --verify
# 原工作区还保留全部来源/模型时，额外核对它们：
python scripts/package_fixed_ce_evidence.py --verify --check-sources
# 只下载阅读包：进入 review_... 目录运行，无需 PyTorch/GPU：
python tools/verify_bundle.py --verify --package .
```

校验会从逐场景结果复算每次成功/碰撞/超时率，核对 84 个检查点全部失败轨迹，用实际步数重新积分 AULC，并核对配对差与训练 seed 均值/样本 SD。4200 行评估记录不是 4200 次独立训练。

阅读包可独立复核记录和统计，未包含模型，因此不能直接重跑策略评估或从完整状态续训。原训练命令和本地模型身份仍保留；不要为上传而重新训练、换 seed 或覆盖旧结果。

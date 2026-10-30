# 上传固定任务CE500确认与QDagger两seed结果

[最新阅读包](../results/analysis_upload/asvc_fixed_paper_supplements_v1/LATEST.md)包含本轮完整结果、原始表格、学习曲线、所有失败轨迹、冻结和复现代码。此前CE两seed/五seed资料包保留；五seed包按原字节嵌入本包，历史“尚未运行CE500”的文字只描述上一轮。

本轮仅新增QDagger seed0、1：每seed20000次离线更新＋200000在线步/199501次在线更新；14个既有训练结果复用。CE新增2500个冻结确认回合，原大间隔组2500回合按模型SHA/场景SHA/顺序/协议核对后复用。50重复验证和500确认分开，未混入50万步多任务研究。

CE500安全成功均值98.32%±1.48%，大间隔99.28%±0.73%；配对差−0.96±1.99个百分点。QDagger最终50验证seed0/1成功86%/94%、全程AULC均值75.90%；本次低于当前大间隔和DQfD。仅两seed适配，不宣称完整Atari复现、正式显著性或单一机制因果。

包内共16条训练曲线、336个检查点、16800条50场景重复验证记录、5000条500确认记录，QDagger预训练前100回合单列。所有失败和大体积JSON无损gzip，原字节SHA和压缩SHA分别保存；不抽样删失败，不上传模型、完整回放断点或恢复测试文件。模型路径/大小/SHA和完整恢复审计记录保留。官方QDagger源码按提交固定，保留Apache许可证。

## 上传命令

在仓库根目录先预览；只读校验，不联网、训练、运行策略或暂存：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_paper_supplements_github.ps1 -Preview
```

由用户执行下面命令，提交并推送到现有origin/main：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_fixed_paper_supplements_github.ps1
```

默认直连。仅启用了代理时加`-ProxyUrl http://127.0.0.1:实际端口`。脚本自动寻找Git和astar-drl Python，核对资料及本地来源，只暂存upload_scope.json指定的本批文件，不使用git add整个工作区。根README、根.gitattributes的原有修改、其他研究和旧轨迹不混入；范围外已暂存文件原样保留并触发停止。远端不一致则停止，不自动合并或强推。

根.gitignore只增加本包原7260条示范NPZ的上传例外，上传脚本仅接受这一明确路径；其他模型、回放和NPZ仍按原规则处理。

## 校验与复现边界

```powershell
# 仓库克隆后的标准库校验
python scripts/package_fixed_paper_supplements.py --verify
# 仅原工作区具备全部原文件、模型和断点时
python scripts/package_fixed_paper_supplements.py --verify --check-sources
# 仅下载阅读包，进入review_...目录，无需PyTorch/GPU
python tools/verify_bundle.py --verify --package .
```

标准库校验复算逐场景成功/碰撞/超时、全部失败长度与结局、实际步数AULC、训练seed均值/样本SD和配对差；检查冻结、预算、日志中的教师退出、多步尾折扣及已完成的完整恢复审计。无权重不能重跑网络评估，仅载入权重也不等于完整续训。训练/续跑命令见ASVC_FIXED_QDAGGER.zh-CN.md；已完成的两seed应直接复用，无需再次启动。

为保持冻结SHA，configs/scripts/docs/tests及新data目录仅追加局部.gitattributes规则。旧包本身字节不变；旧包--check-sources可能报告局部Git属性的后续追加，这属于元数据变化，不代表训练代码或旧结果改变。此次准备不执行commit或push。

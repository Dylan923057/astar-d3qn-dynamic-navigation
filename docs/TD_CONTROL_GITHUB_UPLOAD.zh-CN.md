# TD控制与循环诊断：GitHub上传说明

本批对应2026-10-29完成的两次新TD-only控制与循环诊断。[阅读包入口](../results/analysis_upload/asvc_multitask_td_control_v1/LATEST.md)保留完整五方法结果，原C/D包和四方法包不覆盖。

新增D3QN＋TD目标[-6,10]各seed0/1、500000在线步、499501更新，唯一算法变化是完整目标裁剪。平均成功14%→31%，C为74%；C−TD-only是A*代选＋持续监督组合差，不归于持续监督单项。只有2seed固定验证，不宣称显著性或未见目标泛化。

包内包含十次模型身份、510个检查点、25500条验证记录、完整1000条静态/动态配对及轨迹、所有超时的五动作Q/合法掩码/教师排名和风险、64000条原始/裁剪目标样本、80000条最终回放审计、350条观测闭环记录、冻结/原始源码/测试和典型案例。

其中只有2次新训练；C/D缺失动态补录184回合、TD控制新增静态100回合，其余成绩明确复用，不能把全部配对当新增评估。实际评估监督为0，风险/教师/假设权重和真实未来均为事后分析。旧四方法完整阅读包在previous_four_methods下按原字节保留，其旧轨迹缺失描述不代表本轮状态。

## 命令

在仓库根目录执行，只读预览不会训练、跑策略、暂存、提交、联网或推送：

```powershell
python scripts/package_td_control_evidence.py --verify --check-sources
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_td_control_github.ps1 -Preview
```

用户自行执行下面命令提交推送；不使用代理时省略ProxyUrl：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync_td_control_github.ps1 -ProxyUrl http://127.0.0.1:7897
```

脚本核对origin/main和数据SHA，只提交upload_scope.json限定的本批文件；范围外已暂存改动会触发停止并保持原样。不强推，不清空暂存，不使用git add整个工作区。旧README/历史轨迹、论文草稿、Gemini包、风险迁移/奖励实验和转换工具均不混入。

下载本资料目录后可直接在该目录运行`python tools/verify_bundle.py --verify --package .`，只需标准库，不依赖PyTorch/GPU/模型。完整仓库克隆后用`python scripts/package_td_control_evidence.py --verify`。只有原工作区具备全部来源、模型和完整断点时才加--check-sources。

## 保留与边界

模型、全量回放断点、预检恢复文件和缓存不上传也不删除；model_index.csv保存十个模型和断点路径/SHA。读取已有数据无需权重，重新网络评估需要对应模型，完整续训必须恢复优化器/目标/回放/RNG和环境，不能只加载权重。

原四方法的冻结和模型不变；本包SHA与原始源文件对应。DQfD额外20000离线更新/seed，3472示范仅成功覆盖70/240训练任务，成本与限制继续披露。D保持探索方法。CE只有禁用协议与请求命令，实际训练入口尚未实现；本轮未启动任何其他训练或两个测试池。

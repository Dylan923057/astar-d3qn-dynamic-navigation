# ASVC 持续弱指导实验

本实验检查：将训练后期的 A* 动作代选或动作监督降到较低水平并持续保留，能否改善网络自主评估的退化。

使用当前新版 ASVC（独立教师动作标签）。沿用原奖励、40×40地图、15×15观测、三帧动态占据、TD目标范围、回放和训练参数。每组种子0、1、2，各20万环境步；原组复用已有完整结果，新增9次训练。

|方案|动作代选概率下限|动作监督权重下限|
|---|---:|---:|
|原方案|0|0|
|supervision_floor|0|0.1|
|advice_floor|0.05|0|
|both_floors|0.05|0.1|

实际日程为 `max(原日程, 下限)`，不是从训练开始恒定使用下限。监督在9万步达到0.1；代选在93750步达到0.05。风险筛选仍可取消A*代选，所以实际代选比例可能低于5%。监督权重是辅助损失的系数，不是动作概率。

评估始终关闭教师代选，使用网络自主动作。固定50验证场景用于比较全程曲线、9–12万低谷、10–20万低谷与曲线面积、最终碰撞和超时。保留所有种子与检查点，不使用独立500场景调参或选择模型。

新增入口及实现：

- [配置](../configs/asvc_persistent_teaching_v1.yaml)
- [指导下限实现](../src/astar_d3qn/agents/persistent_teaching.py)
- [运行入口](../scripts/run_persistent_teaching.py)
- [结果分析](../scripts/analyze_persistent_teaching.py)
- [功能测试](../tests/test_persistent_teaching.py)

训练前核查完整初始化、历史源码、恢复一致性和自主评估隔离。9万步以前的各检查点还会与历史权重逐参数比较；不一致则停止并检查对照条件。新实验独立保存到 `outputs/asvc_persistent_teaching_v1`，不覆盖旧训练。

```powershell
& "D:\softwares\Anaconda3\envs\astar-drl\python.exe" scripts/run_persistent_teaching.py
& "D:\softwares\Anaconda3\envs\astar-drl\python.exe" scripts/run_persistent_teaching.py --train --seeds 0 1 2
& "D:\softwares\Anaconda3\envs\astar-drl\python.exe" scripts/analyze_persistent_teaching.py
```

默认命令只检查；`--train`训练。中断后，确认原进程已停止，再添加`--resume`使用完整恢复状态；已完成的组会保留跳过。不要同时启动相同种子和方案。

本轮后台工作进程及输出日志记录在 `outputs/asvc_persistent_teaching_v1/logs`。待9次训练全部完成后分析；部分进度不能当作完整实验结论。

后台分析进程会等待全部训练完成并核验数据，再自动生成 `results/asvc_persistent_teaching_v1/LATEST.md`。实时汇总为该目录的 `progress.json`；若训练失败，`analysis_status.json`记录失败信息，保留原始文件。

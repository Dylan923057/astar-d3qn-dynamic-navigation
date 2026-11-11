# 统计入口兼容记录

原冻结 `data/asvc_multimap_layout_v1/freeze.json` 及其95个源码/配置/协议文件保持原样。正式训练已启动后，额外检查发现本机NumPy为1.26.4，没有2.x的`numpy.trapezoid`名称。梯形积分在1.26中的同义实现是`numpy.trapz`，数值定义不变。

新增独立 `scripts/multimap_layout_supervisor.py` 仅在调度/汇总进程建立该名称别名。训练worker继续以原冻结 `run_multimap_layout.py --worker ...` 在独立进程运行，没有别名注入或学习代码修改。原调度进程74200停止，训练进程62720和60700不中断；兼容调度器先接管并等待它们完成，再续跑剩余98项。原调度日志与全部训练状态保留；不重复已执行步数。

实际推荐启动/续跑命令（自动恢复完整状态、跳过完成项）：

```powershell
.\scripts\start_multimap_layout_compatible.ps1 -Workers 2
# 只汇总，不训练
& D:\softwares\Anaconda3\envs\astar-drl\python.exe -X utf8 scripts/multimap_layout_supervisor.py --analyze
# 只读当前进度
Get-Content results/asvc_multimap_layout_v1/progress.json -Encoding utf8
```

若已有兼容调度进程在运行，不另开第二个队列；OS锁会拒绝重复调度。新增兼容入口自身SHA和采纳的旧worker PID另存 `outputs/asvc_multimap_layout_v1/supervisor.json`，不覆盖原冻结。

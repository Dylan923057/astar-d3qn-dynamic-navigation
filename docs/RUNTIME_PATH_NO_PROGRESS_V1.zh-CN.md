# 无曼哈顿前进奖励：运行时A*引导对照 v1

补充两组训练：`unguided`和`path_guided`均将`reward.progress`从0.05设为0。
只改变这一个奖励项；不删除终点方向观测，不增加子目标奖励，不采样示范，不加载foundation。
普通步（接近终点、远离终点、等待）均为-0.01，到达终点为+10，碰撞为-1。
终点和碰撞仍使用原来的排他终局奖励，不叠加步成本。超时规则仍为每回合300步，无新增超时惩罚。

seed0–4，两组网络结构和同seed初始网络、优化器、随机数状态一致，并与原有距离奖励实验使用相同初始状态。
地图SHA、起终点、路线池、每回合3–5个动态障碍、按回合配对的训练场景序列、固定50个验证场景均保留。
每组200000环境步，epsilon 1.0到0.05、150000步衰减；online replay 10000、batch 64、learning starts 500。
只使用validation，评估epsilon=0，不使用test。

新配置为`configs/whole_map_91701_runtime_path_no_progress_v1.yaml`，原配置不变。
新输出为`outputs/whole_map_91701_runtime_path_no_progress_v1/<method>/seed_<seed>/`。
检查输出为`results/whole_map_91701_runtime_path_no_progress_v1/`。已有训练目录拒绝覆盖。

默认检查不会训练：

```powershell
python scripts/run_runtime_path_guidance.py --check --config configs/whole_map_91701_runtime_path_no_progress_v1.yaml --seeds 0 1 2 3 4
```

用户显式执行以下单行命令时，依次训练5个seed、每seed两组，共10组，每组20万步：

```powershell
python scripts/run_runtime_path_guidance.py --train --config configs/whole_map_91701_runtime_path_no_progress_v1.yaml --seeds 0 1 2 3 4 --methods unguided path_guided
```

本阶段是奖励消融：把新两组与已经完成的有距离奖励两组共同分析。
固定预算下无距离奖励可能改变学习难度；结果用于判断引导收益是否依赖奖励设置，不把零成功率直接解释为算法普遍无效。

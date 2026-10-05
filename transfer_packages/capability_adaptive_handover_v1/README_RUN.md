# capability_adaptive_handover_v1

此目录是独立实验包。它只在 200k-step 动态适应阶段改变 A* demo replay fraction；foundation、reward、epsilon、D3QN 网络、TD loss、batch size、replay capacity 和训练场景流均保持冻结条件。monitor 是 train split 中按 `pair_id` 排序后固定选出的 conflict scenes；这些场景仍保留在原 train 训练流中，因此除 replay fraction 外不改变训练数据。

## 1. 安装依赖（Windows / PowerShell）

```powershell
Set-Location capability_adaptive_handover_v1
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

15 个 foundation 均来自 CUDA 正式运行。最后一条命令必须显示 `True`；若普通安装未提供 CUDA，请按目标机 CUDA/驱动版本先安装 PyTorch 官方 CUDA wheel，再重新执行 `python -m pip install -e .`。

## 2. 预检

```powershell
python scripts/run_capability_adaptive_handover.py --stage preflight --device cuda --threads 1 --deep-foundation-check
python -m unittest discover -s tests -v
```

预检核对配置、数据 manifest、15 个 foundation 的文件 SHA-256、qualified/provenance 元数据；`--deep-foundation-check` 还核对 checkpoint 内部完整状态指纹。若某 foundation 缺失或校验失败，不要在本包内训练替代品：从原电脑 `outputs/ab_five_seed_strict_v1/formal/<map>/seed_<n>/foundation/foundation.pt` 只读重复制到本包对应 `foundations/<map>/seed_<n>/foundation.pt`，并再次预检。也可直接重复制完整实验包。

## 3. 运行 capability_adaptive_handover

全部三张地图、seed 0--4：

```powershell
python scripts/run_capability_adaptive_handover.py --stage run --device cuda --threads 1
```

按需运行单个任务（例如 map index 0、seed 0）：

```powershell
python scripts/run_capability_adaptive_handover.py --stage run --map-indices 0 --seeds 0 --device cuda --threads 1
```

运行固定 200k adaptation steps，不早停；validation 仅在训练结束后评估一次，test 不读取、不生成。结果目录已存在时不会覆盖不完整或不兼容结果。

## 4. 查看结果

```powershell
Get-ChildItem outputs\capability_adaptive_handover_v1\formal -Recurse -Filter result.json
Import-Csv outputs\capability_adaptive_handover_v1\formal\irregular_workcell_91701\seed_0\capability_adaptive_handover\training.csv | Format-Table
Get-Content outputs\capability_adaptive_handover_v1\formal\irregular_workcell_91701\seed_0\capability_adaptive_handover\result.json
```

`training.csv` 每 10k steps 记录 `environment_steps`、`monitor_safe_success`、`competence_ema`、`demo_fraction`、`demo_sample_count`、`online_sample_count` 和 `expert_exit_step`。参数位于 `configs/capability_adaptive_handover_v1.yaml`：`rho_max=0.25`、`beta=0.3`、`tau=0.90`、`consecutive_confirmations=2`、`monitor_scene_count=12`，且 `C_0=initial_competence=0.0`。

# Office 完整旧数据上的兼容性开发训练

## 当前入口

使用 `configs/dynamic_spatial_generalization_office_v6_development_v1.yaml`，不再等待v10特殊诊断案例生成。此配置从v8继承已有静态动作屏蔽与训练设置，显式指定v6数据、数量、奖励和课程阶段。没有创建新地图，没有重新生成三个集。

数据为 `data/dynamic_scenarios/office_40x40_behavior_v6.json`，SHA-256为 `ce47ee4ef71f9e110c69379b08669d0ab9c311afb10dcbc9d2d8f5a65aacda97`。训练100、验证20、测试50；分别有15、10、13条运动路线，场景每个5障碍。2026-09-09只读检查确认170个场景与38条完整路线无重复、静态几何与运动参数未发现问题；全部保存70～75步的安全路径，路径长度与记录一致。本次未重新执行动态回放或自动测试，不能称为全量运行验收已通过。

旧test已经用于开发，继续保留名称便于兼容代码，但本轮结果不是全新独立确认。没有normal类不等于数据不能训练：本轮已有静态起步阶段；同时不能把逐渐增加障碍数量当作已证明单调增加难度。

## 固定设置与范围

- `uniform`、`prefill`、`persistent_demo`各seed=0，共3次；每次40万真实环境交互步，epsilon按34万步衰减，每2.5万步验证选模型。单个seed只能检查流程、学习曲线与失败模式，不支持稳定排名或显著性结论。
- 课程沿用0/6万/14万/24万步启用0/1/3/5障碍。索引为`[]`、`[2]`、`[2,3,4]`、`[0,1,2,3,4]`；它们是v6存储顺序的子集，不是v10“主障碍始终索引2”的因果设置。
- 所有课程阶段使用全部100个train候选，不用`control/easy/medium/hard`筛选v6的行为标签。验证20场和test50场始终使用5障碍，不受课程影响。
- 保留碰撞终止和静态动作屏蔽，不屏蔽动态碰撞动作。奖励step=-0.01、progress=0.05、stay=0、collision=-1、goal=10；STAY仍有普通步进代价，不额外惩罚。相对v8唯一奖励差异是使用已在v9/v10确定的stay=0，不用新奖励试错。
- 示范仍为静态A*20条；Persistent示范比例25%，三个策略的replay逻辑不变。本轮不是原v6/v7实验的直接复现，也不能仅凭与历史结果的差别归因于课程或奖励中的某一项。

## 运行和输出

按用户要求未运行程序/训练/测试。下面单行依次完成三种策略，成功后自动汇总图片；任何训练失败立即停止，不进入下一策略或绘图。

```powershell
Set-Location 'D:\Asatr-D3QN-Workshop'; $officeConfig = '.\configs\dynamic_spatial_generalization_office_v6_development_v1.yaml'; foreach ($strategy in @('uniform','prefill','persistent_demo')) { python -u .\scripts\train_dynamic_spatial_generalization.py --config $officeConfig --strategy $strategy --seed 0; if ($LASTEXITCODE -ne 0) { throw 'Training failed; subsequent tasks stopped.' } }; python -u .\scripts\plot_office_behavior_v6_results.py --config $officeConfig
```

独立输出根目录为 `outputs/dynamic_spatial_generalization_office_v6_development_v1/`，不会覆盖旧版本。但不要重复运行同一配置、策略和seed来覆盖本轮结果。

- 各run：训练与验证记录、selected模型、test逐场指标及完整轨迹。
- `analysis/`：验证曲线、成功/碰撞等指标图、行为分层、50场逐场热力图及三策略路径分页图；`path_gallery_index.csv`用于定位图片。
- 单seed汇总图明确写“single-seed development result; no across-seed CI”，不把无误差条误解为结果稳定。

运行后先检查每个run是否达到40万步、课程是否推进到5障碍、各阶段是否产生成功经验、三策略的碰撞/超时构成。只有发现明确环境或配置错误才重开相关修改，不因某策略没有领先而重新设计地图。多seed和独立确认集是后续研究阶段，不是本轮启动前再加的门槛。

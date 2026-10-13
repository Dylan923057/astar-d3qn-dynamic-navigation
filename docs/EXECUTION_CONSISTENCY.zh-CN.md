# 执行一致性筛选：五seed单因素消融

[最新阅读包与报告](../results/analysis_upload/execution_consistency_seeds0to4_v1/LATEST.md)。

原完整组合advice_bound_margin（代选与辅助监督均10万步退出）五seed直接复用。新对照只取消辅助监督样本必须“实际执行动作=A*建议动作”这一条件；其余建议有效性、静态合法性、观测风险检查和碰撞排除保持不变。没有改奖励、margin损失形式、监督权重或采样比例；不使用分阶段撤教方案，不增加离线示范或预训练。

新接口使用独立teacher_actions，TD仍索引Transition.action及真实奖励、下一状态和终止标记；margin索引教师动作。原D3QN公共方法及旧完整组合代码未改。七项针对性检查、24步断点恢复校验、同seed完整初始训练状态、场景生成及0–20万日程一致性检查通过。冻结登记分别保留在data/whole_map_91701_execution_consistency_v1和data/whole_map_91701_execution_consistency_seeds3to4_v1。

五次新训练均完成200000在线环境步、199501在线更新。seed0–2输出在outputs/whole_map_91701_execution_consistency_v1，seed3–4在outputs/whole_map_91701_execution_consistency_seeds3to4_v1；每个目录下为advice_bound_margin_all_eligible/seed_X/model_final.pth。原组权重在outputs/whole_map_91701_value_repair_v1/advice_bound_margin/seed_X/model_final.pth。

全部十个最终权重复评原50冻结验证场景，ε=0、教师/路线引导/动态保护关闭，只保留共有静态合法动作掩码。使用真实环境步数计算AULC，未选择最佳检查点。独立500最终确认也已完成，不使用测试结果调参。

独立500成功率：原98.16±1.82%，新99.36±0.43%；动态碰撞率原1.84±1.82%，新0.48±0.41%；超时率原0%、新0.16±0.36%。新组2500回合有12次碰撞、3次等待超时、1次绕圈超时。成功率配对差+1.20±1.89个百分点，三个seed提高、seed1/4下降。最终平均改善不能掩盖seed3的学习过程退化，也不等于保证安全或地图泛化。

5000条结果及全部62条失败轨迹已与场景/模型SHA、终止原因和行为分类交叉核对，逐seed统计、均值及样本标准差和配对差重新计算一致。审计命令不执行新的回合：`python scripts/audit_execution_consistency_final500.py`。最新阅读包同时保留训练曲线与最终确认，原50报告中的“尚未评估”作为历史加注保留。

五seed全程AULC：原组91.37±1.05%，取消筛选92.99±1.36%；配对差+1.62±2.07个百分点（样本标准差）。取消筛选四seed全程提高，但seed3全程下降1.06个百分点、后半下降3.91个百分点，低谷由76%变为60%，移到15万步。不能沿用三seed“均改善”的初步结论，也不能将执行一致性筛选写成有一致增益的贡献。最终自主成功率为原98.40±3.58%、新99.20±1.10%，仅支持当前50验证场景的描述。

阅读包包含全部曲线/场景明细、配对统计、覆盖率及动作一致率、margin/权重/Q范围/TD裁剪/实际代选率、等待与绕圈失败分类、成本记录、源码快照和冻结登记。失败示例按最终与低谷处每类别首个场景确定，不筛选模型；批量完整轨迹SHA另存。原组没有有符号Q范围和实际训练耗时，缺失明确标记；监督样本增加与不同动作标签的作用不能分别归因。

仅重算分析、打包，不训练或上传：

```powershell
python -X utf8 scripts/analyze_execution_consistency_five_seeds.py
python -X utf8 scripts/package_execution_consistency.py
```

本地整理已执行以上两项，无需为了上传再跑。权重和回放断点留在outputs，克隆缺权重时不自动重训。

独立500最终确认已经执行完成，沿用此前冻结且与验证集不重复的500场景（SHA前缀051c083a8a185ac1），不重新生成或选择场景。新组seed0–4的200000步最终权重、输入场景及评估代码另存于data/execution_consistency_final500_v1/freeze_manifest.json。原完整组合五seed结果仅在模型/场景SHA、协议及逐场景明细一致时复用；新增2500回合统一自主评估，ε=0，关闭教师/路线引导/动态保护。以下保留入口命令，仅为了上传无需重新运行评估。

只读核验，不评估或训练：

```powershell
python -X utf8 scripts/evaluate_execution_consistency_final500.py
```

正式最终评估，不训练或覆盖旧结果：

```powershell
python -u -X utf8 scripts/evaluate_execution_consistency_final500.py --evaluate
```

结果写入results/execution_consistency_final500_v1的独立时间目录，完整失败轨迹留在对应outputs目录。报告包含逐seed、均值±样本标准差、配对差及碰撞/等待/绕圈超时；评估不用于改配置或选最佳模型。评估前的五seed阅读包保留其“尚未评估500”的历史状态，实际完成情况以新评估目录verification.json为准。

查看本批上传范围（只读）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ExecutionConsistencyOnly -Preview
```

提交并推送本批代码与数据：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -ExecutionConsistencyOnly -ProxyUrl http://127.0.0.1:7897
```

沿用之前Clash的7897端口，需保持代理运行；不使用代理时省略ProxyUrl参数。脚本先验证阅读包，再提交并推送，不触发训练。范围外历史轨迹及其他暂存改动不混入本批提交。

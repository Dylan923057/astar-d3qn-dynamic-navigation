"""Render the independent diagnostic artifacts into a Chinese evidence report."""
from pathlib import Path
import argparse

from analyze_runtime_path_pilot import load_csv
from astar_d3qn.utils.io import write_json
import ddqna_adapter_common as common


def render(root):
    if not common.read_json(root/'completion.json')['complete']:raise ValueError('Diagnosis not completed.')
    target=root/'REPORT.md'
    if target.exists():raise FileExistsError('Preserve the existing report.')
    evidence=load_csv(root/'evidence/method_means.csv')
    allseed=load_csv(root/'evidence/summary.csv')
    diagnostic=load_csv(root/'diagnosis/summary.csv')
    def ratio(row,key):
        m,sd=row[key+'_mean'],row[key+'_sample_sd']
        return '—' if not m else f'{100*float(m):.2f}±{100*float(sd):.2f}%' if sd else f'{100*float(m):.2f}%（SD不可估）'
    def step(row,key):
        n=int(row[key+'_observed_seed_count']);m=row[key+'_mean'];sd=row[key+'_sample_sd']
        if not n:return '未达到（0/5）'
        return (f'{float(m)/10000:.2f}±{float(sd)/10000:.2f}万步' if sd else f'{float(m)/10000:.2f}万步，SD不可估')+f'（{n}/5）'
    rows=['# 对比实验诊断与论文证据梳理','',
        '本次只读取已有模型、做固定场景诊断和复核已有统计；无新的正式训练、无500场景重新选法/调参、无地图或奖励修改。60次正式训练（12方法×5seed）全部完成，1260个原50场景自主验证点均有记录；配对初始化、预算、源码和最终权重通过原注册协议审计。',
        '', '## DDQNA：全0不是评估读错权重，但存在明显数值异常与教师依赖','',
        'DDQNA适配版训练全程p=0.5，10万步并未撤教。8/9/10/11万步检查点是按我方退出时钟对齐；其真正开关比较是同一checkpoint在评估时有无教师。',
        '动态诊断使用原50验证场景全部同序复用；自主ε=0、关闭教师/动态保护；辅助ε=0但诊断包装器明确启用p=0.5。后者只用于机制诊断，不加入自主性能主表。静态自主为同图确定性一次运行；静态辅助为固定20个随机试次，不当成20个训练seed。','',
        '| seed | checkpoint | 动态自主成功 | 动态辅助成功 | 静态自主成功 | 静态辅助成功（20次） | 自主动态Q最大值 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in diagnostic:
        if r['method']!='ddqna_d3qn' or r['static']!='False' or r['mode']!='autonomous':continue
        def find(mode,static):return next(d for d in diagnostic if d['method']==r['method'] and d['seed']==r['seed']
            and d['checkpoint']==r['checkpoint'] and d['mode']==mode and d['static']==static)
        values=[r,find('assisted','False'),find('autonomous','True'),find('assisted','True')]
        rows.append('| '+r['seed']+' | '+r['checkpoint']+' | '+' | '.join(f'{100*float(d["safe_success_rate"]):.0f}%' for d in values)
            +' | '+f'{float(r["valid_q_max"]):.2f}'+' |')
    rows+=['','所有12个自主动态检查点均复现原保存的成功/碰撞/超时统计；批量诊断与原select_action接口在抽查的每个访问格点动作一致，0次不一致。权重读入逐参数摘要匹配文件；每次诊断前后网络、目标网络、优化器及学习器RNG不变。',
        '每一步核对：5×15×15+4标量、额外A*输入为零、目标位移解码对应实际行列坐标、0/1/2/3/4动作是上/下/左/右/等待、掩码只排静态非法动作。A*建议均静态合法且独立BFS距离减少1。原碰撞终止和300步超时逻辑保持，超时按原规则继续自举。未发现这些项目的实现bug。','',
        '| 最终自主动态 | 碰撞 | 等待超时 | 绕圈超时 | 其他超时 | 网络/教师动作一致率 | 有效动作Q范围 | 冻结自举残差绝对值P95 |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in diagnostic:
        if r['checkpoint']!='200000' or r['static']!='False' or r['mode']!='autonomous':continue
        rows.append('| '+r['method']+' seed'+r['seed']+' | '+' | '.join(r[k] for k in
            ('collision_count','waiting_timeout_count','looping_timeout_count','other_timeout_count'))+' | '+
            f'{100*float(r["network_teacher_agreement"]):.2f}% | {float(r["valid_q_min"]):.2f}～{float(r["valid_q_max"]):.2f} | {float(r["frozen_residual_abs_p95"]):.2f} |')
    rows+=['','seed0静态最终在(9,3)与(10,3)之间反复移动，seed2静态最终主要等待；静态均无法自主到达，不能把失败只归于动态避障。纯A*诊断在静态66步到达、动态50场景92%安全到达；两模型均相同，说明教师路径方向有效，但静态合法不等于动态安全。',
        '普通非终止奖励在[-0.06,0.04]，γ=0.99，终止奖励为-1/10，因此真实折扣回报的保守范围为[-6,10]。seed0从2万步Q约2涨到8万步约1061、最终约10460；seed2从约2涨到最终约126。完整组合最终同协议Q最大约10.11/10.01。支持价值估计发生严重偏离，不是单纯的奖励尺度差异。',
        '这里的残差是r+γ max静态合法Q_loaded(s_next)(1-terminated)−Q_loaded(s,执行动作)：checkpoint只保存policy权重，加载后target同步为policy，原训练target快照及回放未保存，故不能冒称原训练逐样本TD误差。实际training.csv显示seed0尾段回合平均Huber损失均值76.83、最大回合均值147.93；seed2为0.81/1.50。这也不能把Huber数值直接与DQfD的MSE数值横比。',
        '本次所有决策、Q值、执行动作、标签、冻结残差保存在diagnosis/*_q_decisions.csv；逐场景等待/循环分类在*_episodes.csv。','',
        '| 指导统计 | seed0 | seed2 | 解释 |','|---|---:|---:|---|',
        '| 原训练0–8万步实际请求/执行率 | 12.80% | 12.52% | 不是总动作的50%；ε随机分支优先 |',
        '| 原训练10–20万步实际请求/执行率 | 43.37% | 43.52% | 教师一直存在 |',
        '| 原训练全程请求/执行率 | 29.61% | 29.60% | 所有已请求建议有效、无风险否决 |',
        '| 最终辅助诊断请求/执行率 | 50.13% | 50.23% | ε=0、p=0.5，仅用于诊断 |',
        '| 最终辅助诊断建议有效率 | 100% | 100% | 仅代表静态合法 |',
        '| 最终辅助路径网络/教师一致率 | 19.24% | 27.67% | 与自主路径的访问状态不同，不宜混算 |',
        '| 历史训练网络/教师一致率 | 未记录 | 未记录 | 分支日志不能补推出未请求时的教师动作 |',
        '', '### 原论文核对及未确认项','',
        '| 项目 | 原论文已核对 | 当前适配 |','|---|---|---|',
        '| 调度/测试 | ε随机优先，否则p=0.5教师；训练固定p；测试关A*、ε=0 | 概率顺序相符，自主开关相符 |',
        '| 损失 | MSE | 为共用D3QN而采用Huber；需要单独稳定性对照 |',
        '| 网络/观测/动作 | 全连接DDQN+Dropout；输入当前位置二维，8动作 | CNN+Dueling，局部动态三帧与目标位移，5动作 |',
        '| 环境/奖励/预算 | 多迷宫与作者动态过程/奖励，按回合衰减ε | 固定40×40，路线池3–5障碍，原项目奖励和150000步ε日程 |',
        '| 教师动态信息权限、缓存及所有优化参数源码细节 | 未核实 | 当前教师只看静态地图、当前位置/目标，有缓存 |',
        '', '依据：[DDQNA出版页](https://www.mdpi.com/2076-3417/16/6/2961)、[可访问的出版全文](https://www.researchgate.net/publication/402809951_A_Hybrid_Planning-Learning_Framework_for_Autonomous_Navigation_with_Dynamic_Obstacles) §3.5/3.8、算法1、表5、§4.2；官网直访429，通过全文文本核对。以上不是原版完整复现。',
        '结论分级：本轮未发现明确坐标/加载/评估实现bug；数值失稳、依赖教师纠正的行为有直接证据；由Huber、目标自举、Dueling函数近似、在线指导分布中哪项导致失稳尚未隔离，不能直接宣判DDQNA原论文差。其他三个seed仅复核已有全0统计，未做同等深入Q轨迹诊断。',
        '', '## 与DQfD的条件核对','',
        '| 条件 | 完整组合 | DQfD / DQfD+约束 |','|---|---|---|',
        '| 配对起点 | 相同seed随机网络、优化器初态 | 预训练之前与我方配对；在线开始时已经预训练 |',
        '| 离线示范 | 0 | 7260条动态实际执行转移；128次尝试保留107条成功非碰撞完整回合 |',
        '| 生成策略 | 无离线生成 | 静态A*建议，观测三帧风险否决后WAIT；不用未来动态信息；含198个WAIT动作 |',
        '| 离线梯度更新 | 0 | 每seed20000；永久示范、单步/10步TD、margin、L2、PER |',
        '| 在线交互/更新 | 200000 / 199501 | 200000 / 199501 |',
        '| 总梯度更新 | 199501 | 219501，另有共享示范生成成本 |',
        '| 回放 | 容量10000在线均匀回放；监督来自符合规则的执行动作 | 在线10000+永久7260示范分区；PER持续混合，非固定25% |',
        '| 示范抽样量 | 在线回放标签覆盖约34.77%，重复抽样不等于新示范 | 既有日志示范抽样覆盖约63.42%；不同教学量/状态分布 |',
        '| A*调用/退出 | 原始精确查询/搜索成本未记；动作代选及监督100000步归零 | 共享生成10756次查询、66次缓存外搜索，耗时5.46秒；在线调用0，永久示范/监督不退出 |',
        '| 验证/模型选择 | ε=0、无教师/动态保护；同50验证；最终模型固定200000步 | 同协议；验证不查询A*，不取最佳checkpoint |',
        '', '现有比较支持：相同在线交互步数下，DQfD是强基线，全程AULC93.24±0.42%略高于完整组合91.37±1.05%；不能声明我方普遍优于DQfD。它不是同总训练预算、同示范量或同教学退出机制比较。额外预训练、PER、多步TD、永久示范、WAIT标签与监督强度共同变化，不能将差值预先全归因于预训练。旧方法未记录的耗时/A*调用量保持缺失。',
        '最少补充因果对照：同冻结7260示范、同PER/多步/监督/L2/在线设置，DQfD仅去掉20000离线更新，以判断预训练在相同200000在线交互下的贡献。这一对照仍不是同总更新预算；若要声称等总计算优势，还需另行定义并匹配总梯度更新和教师数据预算，不能偷偷缩短在线预算。',
        '', '## 已有消融：五seed统计与稳定性','',
        '前期AULC固定0–80000步，全程0–200000步，沿用实际检查点线性插值/梯形积分。首次≥90%取首个实测检查点；三连≥90%另记第三点确认。未达到保留空值；阈值步数均值只计算达到者，括号是有效seed数。DQfD的0步是离线预训练之后。',
        '| 方法 | 前期AULC | 全程AULC | 首次90% | 三连90%确认 | 完全撤教后最低成功率 | 最终原50成功 |',
        '|---|---:|---:|---|---|---:|---:|']
    for r in evidence:
        rows.append('| '+r['method']+' | '+ratio(r,'early_aulc_0_80000')+' | '+ratio(r,'full_aulc')+' | '+
            step(r,'first_90_step')+' | '+step(r,'three_90_confirmation_step')+' | '+ratio(r,'post_teaching_min_success')+' | '+ratio(r,'final_success')+' |')
    rows+=['','完全撤教：原组合与相关消融100000步，尾段组140000步；DQfD永久示范、DDQNA固定教师、无教师组没有这个事件，表中为—，不能硬称退出。最低是每seed在退出至200000步已有检查点的最低值后再算均值/SD，并非全时刻真实最小值。尾段组同时另报告动作代选退出100000步后和监督退出140000步后的窗口，见逐seedCSV。',
        '三次连续达标只说明这段检查点达标，不保证之后稳定。margin无TD约束组首个90%很早，但只有1/5 seed三连达标且退出后最低均0；完整组5/5三连达标，退出后仍降至各seed70/62/58/76/62%。','',
        '| 原50全程AULC逐seed（0→4，%） | 数值 |','|---|---|']
    for r in evidence:
        selected=[s for s in allseed if s['method']==r['method']]
        rows.append('| '+r['method']+' | '+', '.join(f'{100*float(s["full_aulc"]):.2f}' for s in selected)+' |')
    rows+=['', '完整逐seed前期/全程AULC、首次与三连阈值步数、退出后最低值、最后五次均值、最终碰撞/超时见[evidence/summary.csv](evidence/summary.csv)；均值/SD/有效seed数见[method_means.csv](evidence/method_means.csv)，模块配对差值见[paired_differences.csv](evidence/paired_differences.csv)。','',
        '| 论文主张 | 已有支持 | 边界 |','|---|---|---|',
        '| 在线教学加快独立策略前期学习 | 完整前期88.87%，无A*+TD仅2.18%；每个验证点都是自主 | 是组合方案收益，不能说单独A*代选必然有效 |',
        '| 明确动作监督有效 | 代选+TD加入监督，全程+30.93个百分点；无代选的监督+TD相对仅TD +34.66 | 标签匹配/风险/非碰撞规则属于组合条件，非监督在任意设置都有效 |',
        '| 动作代选还有条件增量 | 完整相对监督+TD，全程+8.35±3.94个百分点、前期+21.82 | 需有监督和数值约束；单独A*组不能证明优势 |',
        '| TD约束提供稳定性收益 | 完整相对代选+监督，全程+55.63；仅TD相对原始D3QN也提高 | 当前DDQNA全0包含严重数值偏移，要补稳定版近邻再主张超越 |',
        '| 撤教后稳定保持 | 不成立：完整所有seed仍有低谷；延期全部seed14–20万AULC下降，四seed低谷后移 | 不能写已解决交接或保证安全 |',
        '| 动态风险筛选使最终更安全 | 尚不成立：配对全程差跨seed有正有负，去筛选最终500成功反而较高 | 描述性稳定性现象不是安全保证 |',
        '', '七个2×2×2动作代选/监督/TD约束组合已有五seed；缺“无代选、仅监督、无TD约束”一格，因此不是完整八格因子设计。现有配对足以支持条件增益，但不支持所有交互均已分离；这不是当前最优先补训项。',
        '', '### 已有最终500场景结果，单独报告','',
        '已有结果：完整98.16±1.82%，DQfD97.88±2.09%，DQfD+约束99.20±0.77%，尾段96.84±2.43%，无风险筛选98.84±1.32%，DDQNA适配0%。只作确认，不回流调参、不挑checkpoint；我方最终避障最好未成立。',
        '', '## 下一步最少补充与顺序','',
        '1. 先只做seed0的50000步数值探针：原Huber、原论文MSE、Huber+原完整TD目标约束三分支。从相同随机初始化开始；保持原地图、奖励、动态池、回放、学习率、150000步ε时钟及固定p=0.5，绝不压缩ε日程。记录真实训练TD误差/目标/Q范围及原50自主曲线。该探针只能诊断稳定性，不能作为五seed论文胜负证据；若5万步尚未触发问题，结论是证据不足，不据最终500自动延长或改参数。',
        '2. 先看探针能否消除数值爆大与静态失败，再决定是否需要注册稳定版DDQNA动作机制五seed（MSE对应原论文损失，clip版用于隔离数值稳定因素）。在稳定近邻存在前，当前全0不能作为核心胜出证据。',
        '3. 若论文要解释DQfD优势来自哪里，补DQfD去预训练、永久示范保留的对照；先做同seed小规模验证，再决定五seed。该对照不删除现有DQfD，也不代表同总更新预算。',
        '4. 暂不再扫退出时钟/监督强度。先用现有数据做退出前后状态、Q差和执行标签覆盖诊断；稳定性因果未查清前继续延迟退出可能只移动低谷。',
        '', '可支持的贡献表述：在固定地图动态障碍任务中，在线A*动作教学结合执行匹配的动作监督与TD目标约束，显著改善前期自主学习效率；配对消融说明各模块的条件增益，并揭示退出后的稳定性限制。不能主张首次融合、保证安全、最好最终避障、已解决退出低谷或优于DDQNA原版。',
        '', '## 命令','',
        '已完成的无训练诊断可重跑，每次写独立目录：','', '```powershell',
        'python scripts/diagnose_comparison_evidence.py --evidence --diagnose --seeds 0 2', '```','',
        '探针默认仅检查：','', '```powershell','python scripts/probe_ddqna_stability.py','```','',
        '以下会启动单seed三分支各50000步短探针；本轮未执行：','', '```powershell',
        'python scripts/probe_ddqna_stability.py --run --seed 0 --arms huber mse huber_bound','```','']
    target.write_text('\n'.join(rows),encoding='utf-8')
    write_json(dict(diagnosis_complete=True,formal_training_started=False,probe_run_started=False,
        existing_sources_preserved=True,report_sha256=common.sha256(target)),root/'report_verification.json')
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    render(parser.parse_args().directory.resolve())

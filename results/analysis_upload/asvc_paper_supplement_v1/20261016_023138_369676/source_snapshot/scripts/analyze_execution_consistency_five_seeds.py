"""Audit the completed five-seed consistency ablation; never starts training or test evaluation."""
from pathlib import Path

import analyze_execution_consistency as analysis

ROOT,entry,common=analysis.ROOT,analysis.entry,analysis.common
SEEDS=tuple(range(5))
PARENT_ROOT='whole_map_91701_execution_consistency_v1'
EXTENSION_ROOT='whole_map_91701_execution_consistency_seeds3to4_v1'


def protocol(seed):
    if seed not in SEEDS:raise ValueError('Expected seed0–4.')
    name=PARENT_ROOT if seed<3 else EXTENSION_ROOT
    config=entry.legacy.load_config(ROOT/'configs'/f'{name}.yaml')
    freeze=common.read_json(ROOT/config['execution_consistency']['registration_root']/'freeze_manifest.json')
    if freeze['config']!=config:raise ValueError('Frozen configuration changed.')
    for relative,digest in freeze['source_sha256'].items():
        if common.sha256(ROOT/relative)!=digest:raise ValueError(f'Frozen source changed: {relative}')
    return config,freeze


def registered_sources(seed):
    return protocol(seed)[1]['source_sha256']


def directories(arm,seed):
    config,freeze=protocol(seed)
    inputs=entry.legacy.validate_config(entry.legacy.load_config(entry.legacy.CONFIG))
    old=ROOT/'outputs/whole_map_91701_value_repair_v1/advice_bound_margin'/f'seed_{seed}'
    new=ROOT/config['experiment']['output_root']/entry.METHOD/f'seed_{seed}'
    result=common.read_json(new/'result.json')
    reference=next(r for r in freeze['checks']['initializations'] if r['seed']==seed)
    if result['initial_state_sha256']!=reference['initial_state_sha256'] or not reference['original_initialization_paired']:
        raise ValueError('Paired initialization changed.')
    return config,inputs,old,new,entry.METHOD


analysis.SEEDS=SEEDS
analysis.ANALYSIS_ROOT=ROOT/'results/execution_consistency_five_seeds_v1'
analysis.directories=directories
analysis.registered_sources=registered_sources


def conclusions(output):
    metrics=analysis.load_csv(output/'per_seed_metrics.csv')
    pairs=analysis.load_csv(output/'paired_differences.csv')
    means=analysis.load_csv(output/'mean_sample_sd.csv')
    old=next(r for r in means if r['method']=='advice_bound_margin')
    new=next(r for r in means if r['method']==entry.METHOD)
    lines=['# 五seed结论与证据边界','',
        '五次新对照均完成20万在线环境步，原完整组合五seed复用。全部十份最终权重均复评原50冻结验证场景；新对照未做独立500最终确认。三seed初步结论不能直接沿用。','',
        '|指标|原执行一致性筛选|取消筛选|', '|---|---:|---:|']
    for label,key in (('前5万AULC','aulc_0_50000'),('全程AULC','full_aulc'),('10–20万AULC','aulc_100000_200000'),('最终成功率','final_success')):
        lines.append(f'|{label}|{100*float(old[key+"_mean"]):.2f}±{100*float(old[key+"_sample_sd"]):.2f}%|{100*float(new[key+"_mean"]):.2f}±{100*float(new[key+"_sample_sd"]):.2f}%|')
    improvements=lambda key:sum(float(r[key])>0 for r in pairs)
    lines.extend(['',
        f'取消筛选：前5万AULC提高{improvements("aulc_0_50000")}/5，全程提高{improvements("full_aulc")}/5，10–20万提高{improvements("aulc_100000_200000")}/5，8–16万最低成功率提高{improvements("min_80000_160500")}/5。平均收益不能写成跨seed一致改善。',
        'seed3推翻“三seed都改善”的初步印象：取消筛选后全程AULC下降1.06个百分点，后半下降3.91个百分点，最低成功率由76%变为60%，低谷从11万步后移到15万步。最终成功率却从92%升至98%，说明只看最终点会遗漏学习过程退化。',
        'seed4取消筛选后全程小幅提高0.62个百分点，但前5万下降1.13个百分点、后半基本持平；8–16万最低90%，没有跌破90%，不能虚构恢复耗时。',
        '存在seed相关的早期效率/撤教表现权衡：seed3早期更好而撤教后更差；seed0–2早期和后半均更好。不能将取消筛选解释为只延后低谷，也不能宣称解决退出低谷。',''])
    for method,label in (('advice_bound_margin','原筛选'),(entry.METHOD,'取消筛选')):
        selected=[r for r in metrics if r['method']==method]
        totals=[sum(int(r['80000_160500_'+k]) for r in selected) for k in ('collision_count','waiting_timeout_count','looping_timeout_count')]
        final=sum(int(r['final_collision_count']) for r in selected)
        lines.append(f'{label}：8–16万各验证点累计碰撞/等待超时/绕圈超时={totals[0]}/{totals[1]}/{totals[2]}；最终五seed合计碰撞{final}次（250个验证回合）。区间累计重复同50场景，不是新增独立样本。')
    lines.extend(['',
        '机制：取消条件扩大监督样本数量，同时引入真实动作不同于教师动作的标签。这是同一消融带来的机制变化，未用权重或采样补偿；无法单独归因于标签数量或不同动作标签。TD仍使用真实执行经验，不是伪装的教师经验。Q、margin、裁剪和实际代选率见phase_mechanisms.csv。',
        '论文：当前证据不支持把执行一致性筛选作为有一致增益的独立贡献，也不支持取消筛选必然更优。可以报告平均学习效率收益及seed相关的撤教代价，保留全部seed和原完整组合。新对照未进行独立500确认，不能主张最终避障最优或地图泛化；不根据验证或测试结果回改本批冻结配置。'])
    text='\n'.join(lines)+'\n'
    (output/'CONCLUSIONS.md').write_text(text,encoding='utf-8')
    report=output/'REPORT.md'
    report.write_text(report.read_text(encoding='utf-8')+'\n'+text.replace('# 五seed结论与证据边界','五seed结论与证据边界',1),encoding='utf-8')


if __name__=='__main__':
    output=analysis.analyze()
    verification=common.read_json(output/'verification.json')
    verification['analysis_entry_sha256']=common.sha256(Path(__file__))
    analysis.write_json(verification,output/'verification.json')
    conclusions(output)

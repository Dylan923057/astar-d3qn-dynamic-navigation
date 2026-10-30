"""Fixed-start/goal 200k CE control; independent of 500k multitask proposals."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_persistent_teaching as prior
from fixed_ce_agent import FixedTeacherCEAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.utils.seed import seed_everything

NAME='asvc_fixed_ce_v1'
CONFIG=ROOT/'configs'/f'{NAME}.yaml'
DATA,OUT,RESULTS=(ROOT/folder/NAME for folder in ('data','outputs','results'))
SEEDS=(0,1)
read,sha=prior.common.read_json,prior.common.sha256
write_json,write_csv=prior.write_json,prior.write_records_csv
digest=prior.legacy.runtime.state_digest


def configure():
    config=prior.legacy.load_config(CONFIG)
    baseline,inputs=prior.configure()
    if {k:v for k,v in config.items() if k not in ('experiment','ce_control')} != {k:v for k,v in baseline.items() if k!='experiment'}:
        raise ValueError('CE changed a shared setting')
    expected=dict(seeds=[0,1],maximum_new_training_runs=2,environment_steps=200000,expected_gradient_updates=199501,
        registration_root=f'data/{NAME}',baseline_root='outputs/asvc_persistent_teaching_v1/supervision_floor',
        baseline_variant='supervision_floor',supervision_loss='masked_cross_entropy',temperature=1.,supervision_floor=.1,
        advice_floor=0.,initialization='fresh_random_paired_with_existing_margin_seed',
        evaluation='original_frozen50_every_10000_crossing_at_episode_end',evaluation_advice=False,
        independent500_used=False,pretraining_updates=0,soft_risk_weight=False,model_selection='final_200000_only',
        aulc_intervals=[[0,50000],[0,200000],[100000,200000]],
        aulc_rule='piecewise_linear_in_actual_environment_steps_normalized_by_interval_length',
        statistics='paired_two_seed_exploratory_mean_and_sample_sd')
    if config['ce_control']!=expected:raise ValueError('Use the two-seed T=1 fixed-task protocol; no automatic extension')
    if config['experiment']['output_root']!=f'outputs/{NAME}' or config['experiment']['check_root']!=f'results/{NAME}':
        raise ValueError('Independent output paths required')
    return config,inputs


def make_agent(config,problem,seed,device=None):
    if seed not in SEEDS:raise ValueError('Only seeds 0 and 1 are authorized')
    seed_everything(seed)
    values=dict(config['agent'])
    if device:values['device']=device
    ad,repair=config['training_advice'],config['value_repair']
    return FixedTeacherCEAgent(D3QNConfig((5,15,15),4,5,seed=seed,**values),problem,
        repair_method='advice_bound_margin',lower=repair['target_lower'],upper=repair['target_upper'],
        margin=repair['margin'],margin_weight=repair['margin_weight'],probability=ad['probability'],
        decay_steps=ad['decay_environment_steps'],risk_radius=ad['risk_radius'],rng_offset=ad['rng_seed_offset'],
        advice_floor=0.,supervision_floor=.1)


def sources():
    result=prior.source_hashes()
    paths=[CONFIG,ROOT/'tests/test_fixed_ce.py',ROOT/'docs/ASVC_FIXED_CE.zh-CN.md',
           ROOT/'scripts/start_fixed_ce.ps1',ROOT/'scripts/analyze_fixed_ce.py',ROOT/'scripts/run_fixed_ce.py',
           ROOT/'scripts/fixed_ce_agent.py',Path(__file__)]
    for path in paths:result[str(path.relative_to(ROOT))]=sha(path)
    return result


def audit_baseline(config):
    frozen=read(ROOT/'data/asvc_persistent_teaching_v1/freeze_manifest.json')
    if frozen['source_sha256']!=prior.source_hashes():raise ValueError('Historical margin training sources changed')
    current=read(ROOT/'data/asvc_current_method.json')
    protected={};pairs=[]
    for seed in SEEDS:
        directory=ROOT/config['ce_control']['baseline_root']/f'seed_{seed}'
        manifest=read(directory/'run_manifest.json');result=read(directory/'result.json')
        if not read(directory/'completion.json')['complete'] or result['environment_steps']!=200000 or result['gradient_updates']!=199501:
            raise ValueError('Historical margin run incomplete')
        if manifest['config']!=frozen['config'] or manifest['variant']!='supervision_floor' or manifest['seed']!=seed:
            raise ValueError('Historical method identity differs')
        if manifest['source_sha256']!=frozen['source_sha256'] or manifest['freeze_manifest_sha256']!=sha(ROOT/'data/asvc_persistent_teaching_v1/freeze_manifest.json'):
            raise ValueError('Historical provenance differs')
        model=ROOT/current['final_models'][str(seed)]
        if model.resolve()!=(directory/'model_final.pth').resolve() or sha(model)!=result['final_model_sha256']:
            raise ValueError('Current ASVC registration/model mismatch')
        if manifest['effective_training']!=asdict(prior.legacy.effective_training(config,seed,'formal')):
            raise ValueError('Replay/exploration/update budget differs from margin baseline')
        for path in directory.iterdir():
            if path.is_file():protected[str(path.relative_to(ROOT))]=sha(path)
        pairs.append(dict(seed=seed,baseline_directory=str(directory.relative_to(ROOT)),
            initial_state_sha256=manifest['initial_state_sha256'],baseline_model_sha256=result['final_model_sha256'],
            baseline_curve_sha256=sha(directory/'validation_curve.csv'),
            baseline_online_seconds=result['online_training_seconds']))
    for path in (ROOT/'docs/ASVC_CURRENT.zh-CN.md',ROOT/'data/asvc_current_method.json',
                 ROOT/'data/asvc_persistent_teaching_v1/freeze_manifest.json',
                 ROOT/config['dataset']['validation_reference'],ROOT/config['dataset']['source_manifest'],
                 ROOT/config['dataset']['route_pool_manifest']):
        protected[str(path.relative_to(ROOT))]=sha(path)
    return pairs,protected


def verify_freeze(check_baselines=True):
    frozen=read(DATA/'freeze.json');config,_=configure()
    if frozen['config']!=config or frozen['source_sha256']!=sources():raise ValueError('CE frozen configuration/source changed')
    if check_baselines:
        for name,value in frozen['protected_files_sha256'].items():
            if sha(ROOT/name)!=value:raise ValueError('Protected baseline/input changed: '+name)
    return frozen

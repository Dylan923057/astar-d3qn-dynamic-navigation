"""Read-only protocol and run resolution for the fixed two-removal suite."""
from pathlib import Path

import run_latest_module_ablations as core
import analyze_execution_consistency_five_seeds as complete

ROOT,common,legacy=core.ROOT,core.common,core.legacy
FULL='advice_bound_margin_all_eligible'
MAIN=(FULL,*core.METHODS,'advice_bound','unguided_raw')
CONTEXT=('advice_bound_margin','dqfd')
ALL=(*MAIN,*CONTEXT)
REGISTRATION_ROOT=ROOT/'data'/core.NAME


def relative(path):return Path(path).relative_to(ROOT).as_posix()


def run(method,seed):
    if method not in ALL or seed not in range(5):raise ValueError('Outside the registered comparison.')
    inputs=legacy.validate_config(legacy.load_config(legacy.CONFIG))
    if method in core.METHODS:
        config,_=core.configure(method)
        freeze=common.read_json(ROOT/config['execution_consistency']['registration_root']/'freeze_manifest.json')
        if freeze['source_sha256']!=core.source_hashes():raise ValueError('Frozen new training source changed.')
        directory=ROOT/config['experiment']['output_root']/method/f'seed_{seed}'
        if (directory/'nonfinite_failure.json').is_file():
            return dict(method=method,seed=seed,config=config,inputs=inputs,directory=directory,
                failed=True,failure=common.read_json(directory/'nonfinite_failure.json'))
    elif method==FULL:
        config,inputs,_,directory,_=complete.directories('staged',seed)
        _,freeze=complete.protocol(seed)
    elif method=='dqfd':
        config=legacy.load_config(ROOT/'configs/whole_map_91701_dqfd_comparison_v1.yaml')
        directory=ROOT/config['experiment']['output_root']/method/f'seed_{seed}'
        freeze=None
    else:
        config=legacy.load_config(legacy.CONFIG)
        directory=ROOT/config['experiment']['output_root']/method/f'seed_{seed}'
        freeze=None
    result=common.read_json(directory/'result.json');done=common.read_json(directory/'completion.json')
    if not done['complete'] or done['environment_steps']!=200000 or result['environment_steps']!=200000:
        raise ValueError(f'Unfinished run: {directory}')
    if method in core.METHODS:
        reference=next(r for r in freeze['checks']['initializations'] if r['seed']==seed)
        if result['source_sha256']!=freeze['source_sha256'] or result['initial_state_sha256']!=reference['initial_state_sha256']:
            raise ValueError('Unpaired new run or changed source.')
    model=directory/'model_final.pth'
    if not model.is_file():raise FileNotFoundError(f'Missing final weight; never retrain automatically: {model}')
    digest=common.sha256(model)
    if result.get('final_model_sha256',digest)!=digest:raise ValueError('Recorded final model changed.')
    return dict(method=method,seed=seed,config=config,inputs=inputs,directory=directory,
        result=result,failed=False,model=model,digest=digest)


def finite_five_means(records,keys,methods=ALL):
    """Never drop a failed/censored seed and call the remaining sample five-seed."""
    import numpy as np
    rows=[]
    for method in methods:
        chosen=[r for r in records if r['method']==method]
        if len(chosen)!=5 or {r['seed'] for r in chosen}!=set(range(5)):raise ValueError('Keep every seed.')
        row=dict(method=method,planned_seed_count=5)
        for key in keys:
            values=[r.get(key) for r in chosen]
            usable=[v for v in values if isinstance(v,(int,float,np.number)) and np.isfinite(v)]
            row[key+'_observed_seed_count']=len(usable)
            row[key+'_mean']=float(np.mean(usable)) if len(usable)==5 else None
            row[key+'_sample_sd']=float(np.std(usable,ddof=1)) if len(usable)==5 else None
        rows.append(row)
    return rows

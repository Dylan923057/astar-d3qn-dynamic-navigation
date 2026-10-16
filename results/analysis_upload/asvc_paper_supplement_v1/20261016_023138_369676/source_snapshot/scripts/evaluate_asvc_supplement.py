"""Final models only, fixed autonomous evaluation; resumable per model/density."""
import argparse
from asvc_supplement_common import *
from astar_d3qn.evaluation.rollout import evaluate_agent
from analyze_runtime_path_pilot import load_csv
from prepare_asvc_density import DENSITY_REG
EVAL=RESULTS/'evaluation_v2'
TRACES=OUT/'evaluation_v2'

def run_info(method,seed):
    if method!=METHOD:return historical.run(method,seed)
    config,inputs=configure();directory=OUT/METHOD/f'seed_{seed}'
    result=common.read_json(directory/'result.json')
    assert common.read_json(directory/'completion.json')['complete']
    assert result['environment_steps']==200000 and result['gradient_updates']==199501
    model=directory/'model_final.pth'
    assert common.sha256(model)==result['final_model_sha256']
    return dict(method=method,seed=seed,config=config,inputs=inputs,directory=directory,result=result,model=model,digest=common.sha256(model))

def evaluate(methods=MAIN):
    torch.set_num_threads(1)
    density=common.read_json(DENSITY_REG/'density_registration.json')
    suites={}
    for n in (1,3,5,7):
        path=DENSITY_REG/f'density_{n}.json';assert common.sha256(path)==density['files_sha256'][path.name]
        suites[str(n)]=(common.read_json(path)['scenes'],common.sha256(path))
    cfg=legacy.load_config(dq.tail.CONFIG)
    final,_,manifest=dq.tail.load_frozen(cfg)
    suites['original500']=(final,manifest['files_sha256']['independent_final_scenarios.json'])
    for method in methods:
        for seed in SEEDS:
            p=run_info(method,seed)
            weights=torch.load(p['model'],map_location='cpu',weights_only=True)
            finalweights=torch.load(p['directory']/'model_step_200000.pth',map_location='cpu',weights_only=True)
            assert weights.keys()==finalweights.keys() and all(torch.equal(weights[k],finalweights[k]) for k in weights)
            for name,(scenes,digest) in suites.items():
                if name=='original500' and method!=METHOD:continue
                target=EVAL/name/method/f'seed_{seed}'
                registration=dict(method=method,seed=seed,model_sha256=p['digest'],scene_sha256=digest,
                    epsilon=0,teacher=False,route_guidance=False,dynamic_shield=False,q_output_clip=False,
                    static_mask=True,checkpoint_steps=200000,used_for_tuning=False,
                    source_sha256=common.sha256(Path(__file__)),rollout_sha256=common.sha256(ROOT/'src/astar_d3qn/evaluation/rollout.py'))
                if (target/'completion.json').exists():
                    assert common.read_json(target/'registration.json')==registration
                    continue
                agent=legacy.runtime.make_agent(p['config'],seed,'cpu');agent.load_weights(p['model'])
                factory=dq.tail.FrozenFactory(p['inputs'][0],scenes,trace=True)
                clock=perf_counter()
                with legacy.runtime.preserved_evaluation(agent):
                    _,rows,_=evaluate_agent(agent,[p['inputs'][0]]*500,max_steps=300,window_size=15,
                        reward_config=legacy._reward_config(p['config']),terminate_on_collision=True,
                        environment_factory=factory,mask_static_invalid_actions=True)
                assert len(rows)==500 and [r['scenario_id'] for r in rows]==[s['scenario_id'] for s in scenes]
                assert common.sha256(p['model'])==p['digest']
                traces=[dict(scenario_id=r['scenario_id'],safe_success=r['safe_success'],
                    termination_reason=r['termination_reason'],steps=e.trace) for r,e in zip(rows,factory.environments)]
                summary=dict(method=method,seed=seed,density=name,scene_count=500,
                    safe_success_rate=float(np.mean([r['safe_success'] for r in rows])),
                    dynamic_collision_rate=float(np.mean([r['dynamic_collision'] for r in rows])),
                    timeout_rate=float(np.mean([r['termination_reason']=='timeout' for r in rows])))
                write_json(registration,target/'registration.json');write_records_csv(rows,target/'scene_details.csv')
                write_json(traces,TRACES/name/method/f'seed_{seed}'/'trajectories.json')
                write_json(summary,target/'summary.json')
                write_json(dict(complete=True,evaluation_seconds=perf_counter()-clock),target/'completion.json')
                print(f'{method} seed{seed} density={name}: {summary}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=MAIN,default=list(MAIN))
    args=parser.parse_args();evaluate(args.methods)

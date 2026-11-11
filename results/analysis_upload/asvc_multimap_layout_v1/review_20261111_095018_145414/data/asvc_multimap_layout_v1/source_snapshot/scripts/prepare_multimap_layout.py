"""Audit old evidence, preselect geometry, and freeze new map/scenario/demo inputs."""
from __future__ import annotations
import argparse
from collections import Counter,deque
from datetime import datetime
from dataclasses import asdict
import shutil
from time import perf_counter
import numpy as np
import multimap_layout_common as c
from astar_d3qn.maps.adaptation import irregular_problem
from astar_d3qn.maps.io import problem_record
from astar_d3qn.core.grid import move,in_bounds
from astar_d3qn.maps.problem import NavigationProblem
from astar_d3qn.replay.transition import Transition
from fixed_qdagger_agent import ObservableTeacher
import design_dynamic_route_pool_v2 as routes

def audit():
    registry=c.read(c.ROOT/'data/asvc_current_method.json')
    roots=dict(d3qn=['whole_map_91701_value_repair_v1/unguided_raw']*5,
        astar_prefill_td=['asvc_paper_supplement_v1/astar_prefill_bound']*5,
        dqfd=['whole_map_91701_dqfd_comparison_v1/dqfd']*5,
        qdagger=['asvc_fixed_qdagger_v1']*2+['asvc_fixed_qdagger_five_seeds_v1']*3,
        ce=['asvc_fixed_ce_v1']*2+['asvc_fixed_ce_five_seeds_v1']*3,
        asvc=['asvc_persistent_teaching_v1/supervision_floor']*3+['asvc_persistent_supervision_five_seeds_v1/supervision_floor']*2)
    rows=[];protected={}
    for method,dirs in roots.items():
        for seed,name in enumerate(dirs):
            directory=c.ROOT/'outputs'/name/f'seed_{seed}'
            completion=directory/'completion.json';result=directory/'result.json';model=directory/'model_final.pth'
            done=completion.exists() and c.read(completion).get('complete',False) and model.exists()
            record=c.read(result) if result.exists() else {}
            digest=c.sha(model) if model.exists() else None
            if record.get('final_model_sha256'): assert digest==record['final_model_sha256'],str(model)
            for p in (completion,result,model,directory/'validation_curve.csv',directory/'training.csv'):
                if p.exists(): protected[p.relative_to(c.ROOT).as_posix()]=c.sha(p)
            rows.append(dict(map_id='irregular_workcell_91701',method=method,train_seed=seed,
                training_complete=done,environment_steps=record.get('environment_steps'),
                directory=directory.relative_to(c.ROOT).as_posix(),final_model_sha256=digest,
                action='reuse_original_map_evidence_do_not_retrain'))
    assert len(rows)==30 and all(r['training_complete'] for r in rows),'Existing fixed-task audit incomplete'
    evidence=[]
    for p in sorted((c.ROOT/'results/analysis_upload').glob('*/LATEST.md'),key=lambda p:p.stat().st_mtime,reverse=True):
        if any(k in p.parent.name for k in ('persistent','fixed','multitask')):
            evidence.append(dict(path=p.relative_to(c.ROOT).as_posix(),sha256=c.sha(p),text=p.read_text(encoding='utf-8')))
    for p in (c.ROOT/'HANDOFF.md',c.ROOT/'docs/EXPERIMENT_INVENTORY.zh-CN.md',c.ROOT/'docs/ASVC_CURRENT.zh-CN.md',
              c.ROOT/'data/asvc_current_method.json'):
        protected[p.relative_to(c.ROOT).as_posix()]=c.sha(p)
    density=c.ROOT/'results/asvc_fixed_ce_qdagger_density_v1/six_method_per_seed.csv'
    assert density.exists()
    import csv
    with density.open(encoding='utf-8-sig',newline='') as f: density_rows=list(csv.DictReader(f))
    assert len(density_rows)==120
    protected[density.relative_to(c.ROOT).as_posix()]=c.sha(density)
    value=dict(time=datetime.now().isoformat(),fixed_task_training=rows,
        six_method_density_rows=len(density_rows),latest_completed_records=evidence,
        original_registry=registry,protected_sha256=protected,
        excluded_protocols=['old_demo_replay','500000_step_multitask','zero_shot_cross_map'],
        missing=dict(new_layout_main_runs=90,new_layout_ablations=10,new_final500_suites=100),
        old_results_reused_as_original_map_evidence_only=True)
    c.write(value,c.DATA/'existing_evidence.json');c.csv_write(rows,c.RESULTS/'existing_runs.csv')
    print('AUDIT: original-map 30/30 final training models; six-method density 120/120 rows',flush=True)

def audit_completed_evaluations():
    import csv
    def rows(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    evidence=c.read(c.DATA/'existing_evidence.json');protected=evidence['protected_sha256']
    for name,digest in protected.items():assert c.sha(c.ROOT/name)==digest,'Old protected result changed: '+name
    scenes=c.read(c.ROOT/'data/whole_map_91701_supervision_tail_v1/independent_final_scenarios.json')['scenarios']
    scene_sha=c.sha(c.ROOT/'data/whole_map_91701_supervision_tail_v1/independent_final_scenarios.json')
    old_root=c.ROOT/c.read(c.ROOT/'results/latest_all_eligible_modules_final500_v1/latest.json')['evaluation']
    old_details=rows(old_root/'scene_details.csv');verification=c.read(old_root/'verification.json')
    assert verification['complete'] and verification['frozen_scene_sha256']==scene_sha
    roots=dict(asvc='results/asvc_persistent_supervision_five_seeds_v1/final500',ce='outputs/asvc_fixed_ce_final500_v1',
        qdagger='outputs/asvc_fixed_qdagger_five_seeds_v1/final500',
        astar_prefill_td='results/asvc_paper_supplement_v1/evaluation_v2/original500/astar_prefill_bound')
    summary=[];files={}
    for run in evidence['fixed_task_training']:
        method,seed=run['method'],run['train_seed']
        if method in roots:
            directory=c.ROOT/roots[method]/f'seed_{seed}';registration=c.read(directory/'registration.json')
            assert c.read(directory/'completion.json')['complete']
            assert registration['model_sha256']==run['final_model_sha256'] and registration['scene_sha256']==scene_sha
            assert registration['epsilon']==0 and not registration['teacher'] and registration['static_mask']
            details=rows(directory/'scene_details.csv');source=directory/'scene_details.csv'
            for p in [source,directory/'registration.json',directory/'completion.json']:files[p.relative_to(c.ROOT).as_posix()]=c.sha(p)
        else:
            old_method={'d3qn':'unguided_raw','dqfd':'dqfd'}[method]
            assert verification['source_model_sha256'][run['directory']+'/model_final.pth']==run['final_model_sha256']
            details=[r for r in old_details if r['method']==old_method and int(r['seed'])==seed];source=old_root/'scene_details.csv'
        assert len(details)==500 and [r['scenario_id'] for r in details]==[str(s['scenario_id']) for s in scenes]
        assert [int(r['dynamic_obstacle_count']) for r in details]==[len(s['obstacles']) for s in scenes]
        flag=lambda x:str(x).lower() in ('1','true','1.0')
        summary.append(dict(method=method,train_seed=seed,episodes=500,safe_success_rate=sum(flag(r['safe_success']) for r in details)/500,
            dynamic_collision_rate=sum(flag(r['dynamic_collision']) for r in details)/500,timeout_rate=sum(r['termination_reason']=='timeout' for r in details)/500,
            source=source.relative_to(c.ROOT).as_posix(),model_sha256=run['final_model_sha256'],scene_sha256=scene_sha,reused=True))
    files[(old_root/'scene_details.csv').relative_to(c.ROOT).as_posix()]=c.sha(old_root/'scene_details.csv')
    files[(old_root/'verification.json').relative_to(c.ROOT).as_posix()]=c.sha(old_root/'verification.json')
    ablations=[]
    for method in ('no_td','no_advice'):
        for seed in range(5):
            root=c.ROOT/'outputs/asvc_persistent_paper_completion_v1'/method/f'seed_{seed}'
            assert c.read(root/'completion.json')['complete']
            result=c.read(root/'result.json');digest=c.sha(root/'model_final.pth')
            assert digest==result['final_model_sha256']
            ablations.append(dict(method=method,train_seed=seed,complete=True,model_sha256=digest,source=root.relative_to(c.ROOT).as_posix()))
            for p in [root/'model_final.pth',root/'completion.json',root/'result.json']:files[p.relative_to(c.ROOT).as_posix()]=c.sha(p)
    c.csv_write(summary,c.RESULTS/'reused_original_final500.csv');c.csv_write(ablations,c.RESULTS/'reused_original_ablations.csv')
    c.write(dict(final500_suites=30,final500_episodes=15000,all_model_scene_order_protocol_checks_passed=True,
        current_ablation_additional_training_runs=10,existing_density_suites=120,rerun_old_evaluations=False,protected_sha256=files),c.DATA/'completed_evaluation_audit.json')
    print('REUSE AUDIT: 30 final500 suites / 15000 scene rows; 10 current module ablation runs; density 120 suites',flush=True)

def geometry(problem):
    distances={};queue=deque()
    for cell in problem.obstacles: distances[cell]=0;queue.append(cell)
    while queue:
        p=queue.popleft()
        for a in range(4):
            q=move(p,a)
            if in_bounds(q,problem.size) and q not in distances:
                distances[q]=distances[p]+1;queue.append(q)
    degrees=Counter(sum(in_bounds(move(p,a),problem.size) and move(p,a) not in problem.obstacles for a in range(4))
        for p in distances if p not in problem.obstacles)
    path_clearance=[min(distances[p],p[0]+1,p[1]+1,problem.size-p[0],problem.size-p[1]) for p in problem.nominal_path]
    return dict(map_seed=problem.seed,map_id=problem.map_id,grid_sha256=problem.grid_sha256,
        obstacle_count=len(problem.obstacles),actual_obstacle_fraction=problem.density,static_shortest_steps=problem.astar_steps,
        free_cell_degree_counts=dict(degrees),dead_end_free_cells=degrees[1],
        nominal_min_clearance=min(path_clearance),nominal_mean_clearance=float(np.mean(path_clearance)),
        nominal_mean_inverse_clearance=float(np.mean(1/np.array(path_clearance,dtype=float))),
        corridor_definition='Manhattan distance to blocked cell or exterior; free-cell degree is four-neighbor degree',
        performance_used=False)

def freeze_maps():
    cfg=c.settings();registration=c.DATA/'candidate_registration.json'
    value=dict(candidate_map_seeds=cfg['candidate_map_seeds'],generator=cfg['generator'],
        selection='first_three_unique_reachable_maps_with_unmodified_v2_route_pool_and_scene_smoke',
        ablation_map_rule=cfg['ablation_map_rule'],registered=datetime.now().isoformat(),training_started=False)
    if not registration.exists(): c.write(value,registration)
    else:
        old=c.read(registration);assert old['candidate_map_seeds']==value['candidate_map_seeds'] and old['generator']==value['generator']
    if (c.DATA/'maps_frozen.json').exists(): return c.read(c.DATA/'maps_frozen.json')
    design=c.load_config(c.ROOT/cfg['route_design_config']);accepted=[];audit_rows=[]
    existing=c.read(c.ROOT/'data/risk_handover_v1/manifest.json')
    used={entry['problem']['grid_sha256'] for entry in existing['maps']}
    for seed in cfg['candidate_map_seeds']:
        root=c.DATA/f'map_{seed}';root.mkdir(parents=True,exist_ok=True)
        try:
            problem=irregular_problem(seed,size=40,demo_seed=7400)
            if problem.grid_sha256 in used: raise ValueError('duplicate_existing_or_selected_static_layout')
            paths,_,_=routes.v1.build_reasonable_path_ensemble(problem,seed=seed+design['path_ensemble']['seed_offset'],
                sample_count=design['path_ensemble']['sample_count'],candidate_attempts=design['path_ensemble']['candidate_attempts'],
                detour_budget_steps=design['path_ensemble']['detour_budget_steps'])
            pool,support,pool_audit=routes.build_refined_pool(problem,paths,design['route_pool_counts'],design['classification'],10)
            entry=dict(problem_reference=dict(map_id=problem.map_id,grid_sha256=problem.grid_sha256,start=list(problem.start),goal=list(problem.goal)),
                path_ensemble=dict(sample_count=len(paths),unique_path_count=len(set(paths)),sha256=routes._ensemble_sha256(paths)),
                path_support=support,route_pool=pool,route_pool_audit=pool_audit)
            c.write(problem_record(problem),root/'problem.json');c.write(entry,root/'route_pool.json')
            stream=c.SceneStream(seed,seed*1000,prefix='geometry_precheck')
            counts=Counter()
            for _ in range(128): counts[len(stream(**c.env_kwargs()).dynamic_obstacles)]+=1
            assert set(counts)=={3,4,5}
            info=geometry(problem);info['route_pool_counts']={k:len(v) for k,v in pool.items()};info['scene_smoke_count_histogram']=dict(counts)
            c.write(info,root/'geometry.json');accepted.append(info);used.add(problem.grid_sha256)
            audit_rows.append(dict(map_seed=seed,accepted=True,reason='all_geometry_and_scene_checks_passed',**{k:v for k,v in info.items() if k in ('grid_sha256','static_shortest_steps','actual_obstacle_fraction')}))
            print(f'ACCEPT map_seed={seed}: density={problem.density:.3f}; shortest={problem.astar_steps}',flush=True)
        except (RuntimeError,ValueError) as error:
            audit_rows.append(dict(map_seed=seed,accepted=False,reason=str(error)))
            c.write(dict(map_seed=seed,accepted=False,error=str(error)),root/'rejected_candidate.json')
            print(f'REJECT map_seed={seed}: {error}',flush=True)
        c.write(audit_rows,c.DATA/'candidate_audit.json');c.csv_write(audit_rows,c.RESULTS/'candidate_audit.csv')
        if len(accepted)==3: break
    if len(accepted)!=3: raise RuntimeError('Frozen candidates exhausted; do not relax map/pool criteria')
    chosen=max(accepted,key=lambda r:(r['nominal_mean_inverse_clearance'],-r['map_seed']))['map_seed']
    frozen=dict(maps=accepted,ablation_map_seed=chosen,selection_before_training=True,selected_by_geometry_only=True,
        candidates_not_examined=cfg['candidate_map_seeds'][len(audit_rows):],created=datetime.now().isoformat())
    c.write(frozen,c.DATA/'maps_frozen.json');c.csv_write(accepted,c.RESULTS/'maps.csv');preview(frozen)
    return frozen

def preview(frozen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,info in zip(axes,frozen['maps']):
        p=c.map_data(info['map_seed'])[0];grid=np.zeros((40,40))
        for r,col in p.obstacles: grid[r,col]=1
        ax.imshow(grid,cmap=ListedColormap(['#FFFFFF','#66717A']),vmin=0,vmax=1,interpolation='nearest')
        x,y=zip(*[(col,r) for r,col in p.nominal_path]);ax.plot(x,y,c='#4C7898',lw=1.3)
        ax.scatter([p.start[1]],[p.start[0]],c='#264F70',s=28);ax.scatter([p.goal[1]],[p.goal[0]],c='#C87955',marker='*',s=60)
        ax.set_title(f"map_seed {p.seed}\n{100*p.density:.1f}% blocked | shortest {p.astar_steps}",fontsize=10)
        ax.set_xlabel('Column');ax.set_ylabel('Row');ax.set_xticks([0,10,20,30,39]);ax.set_yticks([0,10,20,30,39])
        for spine in ax.spines.values(): spine.set_color('#9BA3A9')
    fig.legend(handles=[Line2D([0],[0],color='#4C7898',label='Static A* reference'),Line2D([0],[0],marker='o',lw=0,c='#264F70',label='Start'),Line2D([0],[0],marker='*',lw=0,c='#C87955',label='Goal')],loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(c.RESULTS/'new_static_layouts.png',dpi=300);fig.savefig(c.RESULTS/'new_static_layouts.pdf');plt.close(fig)

def freeze_scenes(seed):
    root=c.DATA/f'map_{seed}'
    if (root/'scenario_registration.json').exists(): return
    used=set();groups={};cfg=c.settings()
    for name,count,offset in [('validation50',50,700000000),('test500',500,800000000)]:
        stream=c.SceneStream(seed,offset+seed,exclude=used,prefix=name)
        scenes=[]
        while len(scenes)<count:
            env=stream(**c.env_kwargs());scene=c.scene_record(env);key=c.scene_key(scene)
            if key in used: continue
            used.add(key);scenes.append(scene)
        groups[name]=scenes;c.write(dict(map_seed=seed,scenarios=scenes,sampling_seed=offset+seed),root/f'{name}.json')
    prefixes={}
    for train_seed in cfg['train_seeds']:
        stream=c.training_stream(seed,train_seed)
        scenes=[c.scene_record(stream(**c.env_kwargs())) for _ in range(cfg['scene_stream']['audited_prefix_per_seed'])]
        prefixes[str(train_seed)]=scenes
    c.write(prefixes,root/'training_stream_prefixes.json')
    c.write(dict(map_seed=seed,train_seeds=cfg['train_seeds'],generator_source='src/astar_d3qn/envs/whole_map_route_pool.py',
        stream_spec=cfg['scene_stream'],validation_test_combinations_unique_and_disjoint=True,
        train_excludes_all_heldout_combinations=True,selection_used_policy_performance=False,
        representation_explanation='Complete deterministic procedural stream, not a finite scene pool. Frozen local RNG seeds, sampler source, route pool, rejection rule, and 128-scene prefixes per train_seed define every subsequent episode; full RNG state saved in recovery.'),root/'scenario_registration.json')
    # Representative dynamic preview, chosen by fixed index without a policy.
    p=c.map_data(seed)[0];scene=groups['validation50'][0]
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,5));routes.v1._draw_base(ax,p)
    colors={'high_interaction':'#C87955','alternative_branch':'#4C7898','background':'#729283'}
    for spec,cat in zip(scene['obstacles'],scene['categories']):
        points=np.array(spec['route']);ax.plot(points[:,1],points[:,0],c=colors[cat],lw=2)
        r,col=points[spec['start_index']];ax.scatter(col,r,c=colors[cat],s=45,marker='s')
    ax.set_title(f'Map {seed}: first frozen validation scene');fig.tight_layout();fig.savefig(c.RESULTS/f'map_{seed}_dynamic_preview.png',dpi=300);plt.close(fig)
    print(f'SCENES map={seed}: 50 validation, 500 independent test, 5 deterministic training streams',flush=True)

def freeze_demos(seed):
    root=c.DATA/f'map_{seed}'
    if (root/'demonstration_registration.json').exists():
        c.load_demos(seed,'qdagger');return
    cfg=c.settings()['demonstrations'];val,test=c.heldout(seed)
    stream=c.SceneStream(seed,cfg['seed'],exclude={c.scene_key(s) for s in val+test},prefix='demo')
    teacher=ObservableTeacher(c.map_data(seed)[0]);kept=[];episodes=[];clock=perf_counter();steps=0
    for ep in range(cfg['attempts']):
        env=stream(**c.env_kwargs());state=env.reset();transitions=[];waits=0
        while True:
            action=teacher.query(state);result=env.step(action);steps+=1;waits+=int(action==4)
            transitions.append(Transition(state,action,result.reward,result.observation,result.terminated,env.action_mask(True)))
            state=result.observation
            if result.done: break
        success=bool(result.info['reached']) and not bool(result.info['collision'])
        episodes.append(dict(**c.scene_record(env),steps=len(transitions),retained=success,
            retained_offset=len(kept) if success else None,termination_reason=result.info['termination_reason'],wait_steps=waits))
        if success: kept.extend(transitions)
    if len(kept)<64: raise RuntimeError('Fixed demonstration budget insufficient; preserve failure, no automatic budget increase')
    np.savez_compressed(root/'demonstrations.npz',
        spatial=np.stack([t.state.spatial for t in kept]).astype(np.uint8),scalars=np.stack([t.state.scalars for t in kept]),
        next_spatial=np.stack([t.next_state.spatial for t in kept]).astype(np.uint8),next_scalars=np.stack([t.next_state.scalars for t in kept]),
        action=np.array([t.action for t in kept],dtype=np.uint8),reward=np.array([t.reward for t in kept],dtype=np.float64),
        terminated=np.array([t.terminated for t in kept]),next_mask=np.stack([t.next_action_mask for t in kept]),episode_end=np.array([t.terminated for t in kept]))
    c.write(dict(episodes=episodes),root/'demonstration_episodes.json')
    c.write(dict(map_seed=seed,config=cfg,attempts=len(episodes),successful_attempts=sum(e['retained'] for e in episodes),
        failed_attempts=sum(not e['retained'] for e in episodes),transitions=len(kept),collection_environment_steps=steps,
        generation_seconds=perf_counter()-clock,teacher_queries=teacher.queries,astar_searches=teacher.searches,
        teacher_seconds=teacher.seconds,heldout_rejections=stream.rejected,uses_future=False,
        budget_equal_across_maps=True,transition_count_not_forced=True),root/'demonstration_registration.json')
    c.load_demos(seed,'qdagger');c.load_demos(seed,'dqfd')
    print(f'DEMOS map={seed}: {len(kept)} transitions; success={sum(e["retained"] for e in episodes)}/128',flush=True)

def freeze():
    if (c.DATA/'freeze.json').exists(): c.verify_freeze();print('Existing immutable freeze verified');return
    maps=c.read(c.DATA/'maps_frozen.json');config=c.settings();jobs=[]
    for info in maps['maps']:
        for seed in config['train_seeds']:
            for method in config['methods']:
                jobs.append(dict(map_seed=info['map_seed'],train_seed=seed,method=method))
    for seed in config['train_seeds']:
        for method in config['ablation_methods']:
            jobs.append(dict(map_seed=maps['ablation_map_seed'],train_seed=seed,method=method))
    assert len(jobs)==100
    c.write(dict(jobs=jobs,max_workers=2),c.DATA/'queue.json')
    source=c.source_files();data={p.relative_to(c.DATA).as_posix():c.sha(p) for p in c.DATA.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts}
    for name in source:
        dst=c.DATA/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(c.ROOT/name,dst)
    c.write(dict(config=config,shared_config=c.shared_config(),source_sha256=source,data_sha256=data,
        created=datetime.now().isoformat(),main_runs=90,ablation_runs=10,total_runs=100,
        original_results_preserved=True,performance_used_for_map_selection=False),c.DATA/'freeze.json')
    c.verify_freeze();print('FROZEN: 100 training identities, input hashes and complete source snapshot',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    c.DATA.mkdir(parents=True,exist_ok=True);c.RESULTS.mkdir(parents=True,exist_ok=True)
    if (c.DATA/'freeze.json').exists(): c.verify_freeze();return
    if not (c.DATA/'existing_evidence.json').exists(): audit()
    if not (c.DATA/'completed_evaluation_audit.json').exists():audit_completed_evaluations()
    maps=freeze_maps()
    for row in maps['maps']: freeze_scenes(row['map_seed']);freeze_demos(row['map_seed'])
    if args.freeze: freeze()

if __name__=='__main__': main()

"""Independent ASVC paper supplement; historical source files are read-only."""
from pathlib import Path
from dataclasses import replace, asdict
from time import perf_counter
import random
import numpy as np
import torch
from replication_recovery import Recovery as BaseRecovery, STAGE_LOCALS as BASE_LOCALS
import dqfd_comparison_common as dq
import latest_module_common as historical
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.trainer import build_replay
from astar_d3qn.replay.uniform import UniformReplayBuffer
from astar_d3qn.utils.io import write_json, write_records_csv

ROOT, common, legacy = historical.ROOT, historical.common, historical.legacy
NAME='asvc_paper_supplement_v1'
METHOD='astar_prefill_bound'
CONFIG=ROOT/'configs'/f'{NAME}.yaml'
REG=ROOT/'data'/NAME
OUT=ROOT/'outputs'/NAME
RESULTS=ROOT/'results'/NAME
SEEDS=tuple(range(5))
MAIN=(historical.FULL,'unguided_raw','dqfd',METHOD)
LABELS={historical.FULL:'ASVC-D3QN','unguided_raw':'D3QN','dqfd':'DQfD',METHOD:'A*-Prefill + TD constraint',
    'advice_margin_all_eligible':'ASVC without TD constraint',
    'supervision_only_bound_all_eligible':'ASVC without action advice',
    'advice_bound':'ASVC without action supervision','advice_bound_margin':'Historical execution-matched method'}

STAGE_LOCALS=(*BASE_LOCALS,'demonstration_ids')

class Recovery(BaseRecovery):
    """Preserve demo provenance by shared pickle references, never persisted Python IDs."""
    def checkpoint(self,context):
        objects=[t for t in context['replay'].snapshot() if id(t) in context['demonstration_ids']]
        original=self.names
        self.names=(*original,'_prefill_demo_objects')
        try:super().checkpoint(dict(context,_prefill_demo_objects=objects))
        finally:self.names=original

    def restore(self,context):
        values=super().restore(context)
        if values:
            values['demonstration_ids']={id(t) for t in values.pop('_prefill_demo_objects')}
        return values

def configure():
    config=legacy.load_config(CONFIG)
    old=legacy.load_config(legacy.CONFIG)
    for key in old:
        if key not in ('experiment','training') and config[key]!=old[key]:raise ValueError(f'Shared {key} changed')
    if config['training']!=dict(old['training'],replay_strategy='prefill'):raise ValueError('Only replay initialization may change')
    return config,legacy.validate_config(old)

def make_agent(config,problem,seed,device=None):
    return legacy.make_agent(config,problem,seed,'unguided_bound',device)

def source_hashes():
    paths=[CONFIG,ROOT/'scripts/asvc_supplement_common.py',ROOT/'scripts/asvc_prefill_training.py',
           ROOT/'scripts/replication_recovery.py',ROOT/'scripts/run_asvc_supplement.py']
    return {str(p.relative_to(ROOT)):common.sha256(p) for p in paths}|legacy.source_hashes()

def load_demonstrations():
    config=legacy.load_config(dq.CONFIG)
    manifest=dq.frozen(config)
    with np.load(dq.artifact_root(config)/'demonstrations.npz',allow_pickle=False) as d:
        result=[Transition(Observation(s.astype(np.float32),v.copy()),int(a),float(r),
            Observation(ns.astype(np.float32),nv.copy()),bool(t),mask.copy())
            for s,v,a,r,ns,nv,t,mask in zip(d['spatial'],d['scalars'],d['action'],d['reward'],
                d['next_spatial'],d['next_scalars'],d['terminated'],d['next_mask'])]
    assert len(result)==manifest['demo_transition_count']==7260
    return result

def audit_demonstrations(config,inputs):
    demos=load_demonstrations(); root=dq.artifact_root(legacy.load_config(dq.CONFIG))
    episodes=common.read_json(root/'demonstration_episodes.json')['episodes']
    retained=[e for e in episodes if e['retained']]
    factory=dq.tail.FrozenFactory(inputs[0],retained)
    teacher=legacy.make_agent(config,inputs[0],91702641,'advice_bound_margin','cpu')
    i=waits=0
    with np.load(root/'demonstrations.npz',allow_pickle=False) as data:
        for episode in retained:
            assert episode['retained_offset']==i
            env=factory(inputs[0],max_steps=300,window_size=15,reward_config=legacy._reward_config(config),terminate_on_collision=True)
            state=env.reset()
            for j in range(episode['steps']):
                t=demos[i]
                assert np.array_equal(t.state.spatial,state.spatial) and np.array_equal(t.state.scalars,state.scalars)
                action=teacher.static_advice(state)
                if action is None or not env.action_mask(True)[action] or teacher.observed_risk(state,action):action=4
                assert action==t.action
                waits+=int(action==4)
                result=env.step(t.action)
                assert result.reward==t.reward and result.terminated==t.terminated
                assert result.done==bool(data['episode_end'][i]) and not result.truncated
                assert np.array_equal(result.observation.spatial,t.next_state.spatial)
                assert np.array_equal(result.observation.scalars,t.next_state.scalars)
                assert np.array_equal(env.action_mask(True),t.next_action_mask)
                assert t.state.spatial.shape==(5,15,15) and t.state.scalars.shape==(4,)
                assert not np.any(t.state.spatial[-1]) and not np.any(t.state.scalars[2:])
                state=result.observation;i+=1
            assert result.terminated and result.info['reached']
    assert i==7260 and waits==198
    return dict(transitions=i,episodes=len(retained),observed_risk_waits=waits,
        all_transitions_reexecuted_exactly=True,terminated_is_bootstrap_mask=True,
        retained_dataset_truncations=0,timeout_and_budget_truncation_bootstraps=True,
        old_prefill_reusable=False,old_prefill_reason='1320 static transitions; different protocol and data',
        original_generation_cost=common.read_json(root/'freeze_manifest.json'))

def recovery_check(config,inputs):
    from tempfile import TemporaryDirectory
    from replication_recovery import instrument
    # Include demonstration identity set in recovery: diagnostics can otherwise lose provenance after pickle.
    training=replace(legacy.effective_training(config,0,'formal'),max_environment_steps=504,max_steps=6,
        progress_interval_environment_steps=None,progress_interval=0,allow_partial_epsilon_schedule=True)
    demos=load_demonstrations()
    def execute(path,stop=None,resume=False):
        agent=make_agent(config,inputs[0],0,'cpu')
        factory=legacy.runtime.make_factory(config,*inputs,method='unguided',seed=0)
        recovery=Recovery(path,agent,factory,STAGE_LOCALS,resume=resume,identity='prefill-parity',checkpoint_interval=250,stop_after=stop)
        result=instrument(legacy.train_d3qn,STAGE_LOCALS)([inputs[0]],agent,training,demonstrations=demos,
            reward_config=legacy._reward_config(config),environment_factory=factory,_recovery=recovery)
        assert result.gradient_updates==5 and agent.update_steps==5
        return legacy.runtime.state_digest(agent.training_state_dict()),result.episode_records
    with TemporaryDirectory(dir=OUT) as temporary:
        p=Path(temporary); plain=execute(p/'plain')
        try:execute(p/'resume',stop=252)
        except InterruptedError:pass
        else:raise AssertionError('No interruption')
        assert plain==execute(p/'resume',resume=True)
    replay=build_replay(training,demos,())
    assert isinstance(replay,UniformReplayBuffer) and len(replay)==7260
    marker=replace(demos[0],reward=123.)
    for _ in range(10000):replay.add(marker)
    assert len(replay)==10000 and all(t is marker for t in replay.snapshot())
    agent=make_agent(config,inputs[0],0,'cpu')
    assert agent.advice_probability()==0 and agent.imitation_weight()==0 and agent.bound_enabled
    r=torch.tensor([-100.,100.]); z=torch.zeros(2,5); m=torch.ones(2,5,dtype=torch.bool)
    assert torch.equal(agent._td_targets(r,torch.ones(2),z,z,m),torch.tensor([-6.,10.]))
    return dict(passed=True,steps=504,updates=5,learning_starts=500,exact_resume=True,full_fifo_overwrite=True)

def freeze_density(config,inputs):
    from astar_d3qn.envs.dynamic_grid import DynamicObstacleSpec
    cats=('high_interaction','alternative_branch','background')
    rng=random.Random(config['supplement']['density_seed'])
    move_every=legacy.runtime.make_factory(config,*inputs,method='unguided',seed=0).factory.move_every
    scenes={n:[] for n in (1,3,5,7)}
    # Rotating category order balances every prefix over 500 scenes (difference at most one).
    for i in range(500):
        order=[cats[(i+j)%3] for j in range(7)]
        chosen=[]
        for cat in order:
            candidates=[r for r in inputs[1]['route_pool'][cat] if r['route_id'] not in {v['route_id'] for v in chosen}]
            chosen.append(rng.choice(candidates))
        specs=[asdict(DynamicObstacleSpec(route=tuple(map(tuple,r['route'])),start_index=rng.randrange(len(r['route'])),
            direction=rng.choice((-1,1)),move_every=move_every,label=r['route_id'],
            reference_path_source='whole_map_reasonable_path_ensemble_v2')) for r in chosen]
        for n in scenes:
            scenes[n].append(dict(scenario_id=f'asvc_density_{n}_pair_{i:04d}',pair_id=i,
                start=list(inputs[0].start),goal=list(inputs[0].goal),route_ids=[r['route_id'] for r in chosen[:n]],
                categories=order[:n],obstacles=specs[:n]))
    hashes={}
    for n,rows in scenes.items():
        path=REG/f'density_{n}.json'
        write_json(dict(scenes=rows),path);hashes[path.name]=common.sha256(path)
    write_json(dict(seed=config['supplement']['density_seed'],scene_count_per_density=500,nested_prefixes=True,
        rule='Rotate H/A/B category order by pair index; sample distinct routes within each category; random uniform phase and direction; fixed original move_every. No policy-dependent rejection.',
        differs_from_training_sampler='Balanced category prefixes replace 3-5-only scene composition and acceptance quotas; physical motion rules unchanged.',
        move_every=move_every,grid=inputs[0].manifest(),files_sha256=hashes,
        representative_scene_ids=[f'asvc_density_5_pair_{i:04d}' for i in config['supplement']['representative_indices']],
        representative_seed=0,used_for_tuning=False),REG/'density_registration.json')

def preflight():
    torch.set_num_threads(1);config,inputs=configure();OUT.mkdir(parents=True,exist_ok=True)
    if (REG/'freeze_manifest.json').exists():
        frozen=common.read_json(REG/'freeze_manifest.json')
        assert frozen['config']==config and frozen['source_sha256']==source_hashes()
        return frozen
    audit=audit_demonstrations(config,inputs); recovery=recovery_check(config,inputs)
    initial=[]
    for seed in SEEDS:
        a=make_agent(config,inputs[0],seed,'cpu')
        assert dq.paired_digest(a)==dq.baseline_digest(config,inputs[0],seed)
        initial.append(dict(seed=seed,initial_state_sha256=legacy.runtime.state_digest(a.training_state_dict())))
    validation,_=legacy.runtime.check(legacy.load_config(legacy.CONFIG),inputs,list(SEEDS))
    REG.mkdir(parents=True,exist_ok=True)
    freeze_density(config,inputs)
    frozen=dict(config=config,source_sha256=source_hashes(),checks=dict(initializations=initial,
        random_alignment='Same base learner/target/optimizer/RNG and episode-index scene stream; replay initialization is the intervention.',
        demonstrations=audit,recovery=recovery,validation=validation),
        final_only=200000,online_updates=199501,offline_updates=0)
    write_json(frozen,REG/'freeze_manifest.json')
    print('PREFLIGHT PASSED: exact 7260 transitions, FIFO, step500 updates, resume, pairing, autonomous evaluation',flush=True)
    return frozen

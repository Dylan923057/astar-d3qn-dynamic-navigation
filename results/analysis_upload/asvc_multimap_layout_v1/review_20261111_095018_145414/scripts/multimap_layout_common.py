"""Independent map-aware adapters; historical learners and datasets are read-only."""
from __future__ import annotations
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from astar_d3qn.utils.config import load_config
from astar_d3qn.utils.seed import seed_everything
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.envs.path_guidance import PathGuidanceEnvironment
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent
from astar_d3qn.agents.persistent_teaching import PersistentTeachingAgent
from astar_d3qn.agents.dqfd import DQfDAgent
from astar_d3qn.replay.transition import Transition
from astar_d3qn.replay.dqfd import NStepCollector
from train_whole_map_route_pool_pilot import _factory, _reward_config, _training_config
from run_runtime_path_guidance import preserved_evaluation, state_digest
from teaching_efficiency_common import FrozenFactory
from fixed_qdagger_agent import QDaggerAgent, ObservableTeacher, ThreeStepCollector
from fixed_ce_agent import FixedTeacherCEAgent
from fixed_supplement_common import read, write, csv_write, sha, summarize

NAME='asvc_multimap_layout_v1'
CONFIG=ROOT/'configs'/f'{NAME}.yaml'
DATA,OUT,RESULTS=(ROOT/d/NAME for d in ('data','outputs','results'))

def settings(): return load_config(CONFIG)
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def shared_config():
    config=load_config(ROOT/settings()['reference_config'])
    config['training']['validation_interval_environment_steps']=settings()['validation_interval']
    return config
def map_data(map_seed):
    directory=DATA/f'map_{map_seed}'
    problem=problem_from_record(read(directory/'problem.json'))
    entry=read(directory/'route_pool.json')
    design=load_config(ROOT/settings()['route_design_config'])
    return problem,entry,dict(design=design)
def scene_key(scene): return tuple(sorted(scene['route_ids']))
def scene_record(env):
    inner=env.environment if isinstance(env,PathGuidanceEnvironment) else env
    return dict(scenario_id=inner.scenario_id,start=list(inner.problem.start),goal=list(inner.problem.goal),
        route_ids=list(inner.dynamic_route_ids),categories=list(inner.dynamic_route_categories),
        obstacles=[asdict(spec) for spec in inner.dynamic_obstacles])

class SceneStream:
    """Frozen procedural sequence, independent of agent actions and global RNGs.

    The kth accepted episode is the same for every method with this map/train
    seed. Episode durations can differ. Held-out route combinations are skipped
    without consulting outcomes. The complete local RNG state is checkpointed.
    """
    def __init__(self,map_seed,seed,*,exclude=(),prefix='train'):
        self.map_seed,self.seed=int(map_seed),int(seed)
        self.problem,entry,pool=map_data(map_seed)
        self.factory=_factory(shared_config(),self.problem,entry,pool,seed=seed,prefix=prefix)
        self.excluded=set(exclude);self.rejected=0;self.accepted=0
    def __call__(self,**kwargs):
        for _ in range(100000):
            env=self.factory(self.problem,**kwargs)
            if scene_key(scene_record(env)) in self.excluded:
                self.rejected+=1;continue
            self.accepted+=1
            return PathGuidanceEnvironment(env,enabled=False)
        raise RuntimeError('Frozen scene stream rejection budget exhausted')

def env_kwargs(): return dict(max_steps=300,window_size=15,reward_config=_reward_config(shared_config()),terminate_on_collision=True)
def heldout(map_seed):
    root=DATA/f'map_{map_seed}'
    return read(root/'validation50.json')['scenarios'],read(root/'test500.json')['scenarios']
def training_stream(map_seed,train_seed):
    val,test=heldout(map_seed)
    return SceneStream(map_seed,map_seed*1000+train_seed,exclude={scene_key(s) for s in val+test},prefix=f'train_map{map_seed}_train{train_seed}')

def effective_config(map_seed,method,train_seed):
    cfg=shared_config();cfg['reference_shared_config']=settings()['reference_config']
    cfg['experiment']=dict(name=NAME,output_root=f'outputs/{NAME}/map_{map_seed}/{method}/seed_{train_seed}',initialization='paired_random_no_foundation')
    cfg['dataset']=dict(map_id=f'irregular_workcell_{map_seed}',map_seed=map_seed,
        grid_sha256=map_data(map_seed)[0].grid_sha256,problem=f'data/{NAME}/map_{map_seed}/problem.json',
        route_pool=f'data/{NAME}/map_{map_seed}/route_pool.json',validation=f'data/{NAME}/map_{map_seed}/validation50.json',
        test=f'data/{NAME}/map_{map_seed}/test500.json',training_stream_seed=map_seed*1000+train_seed)
    cfg['training']['seeds']=[train_seed];cfg['training']['replay_strategy']='prefill' if method=='astar_prefill_td' else 'dqfd_per' if method=='dqfd' else 'student_three_step_uniform' if method=='qdagger' else 'uniform'
    cfg['dynamic_route_pool']['training_sampling_seed_offset']=map_seed*1000
    cfg['dynamic_route_pool']['validation_sampling_seed']=700000000+map_seed
    cfg['method']=dict(name=method,online_steps=200000,offline_updates=20000 if method in ('dqfd','qdagger') else 0,
        teacher_advice=method in ('ce','asvc','asvc_no_td','asvc_withdraw'),
        td_constraint=method in ('astar_prefill_td','ce','asvc','asvc_withdraw'),
        persistent_supervision_floor=.1 if method in ('ce','asvc','asvc_no_td') else 0.)
    if method=='dqfd':cfg['dqfd']=load_config(ROOT/settings()['dqfd_config'])['comparison']
    if method=='qdagger':cfg['qdagger']=dict(n_steps=3,temperature=.1,offline_weight=3,online_weight='3*max(1-t/120000,0)',permanent_demos=False)
    return cfg

def make_agent(method,map_seed,train_seed,device=None):
    cfg=shared_config();problem=map_data(map_seed)[0]
    seed_everything(train_seed);values=dict(cfg['agent']);values['device']=device or settings()['device']
    ac=D3QNConfig((5,15,15),4,5,seed=train_seed,**values)
    if method=='d3qn': return D3QNAgent(ac)
    if method=='dqfd': return DQfDAgent(ac,load_config(ROOT/settings()['dqfd_config'])['comparison'],bound=False)
    if method=='qdagger': return QDaggerAgent(ac)
    ad,repair=cfg['training_advice'],cfg['value_repair']
    kwargs=dict(lower=repair['target_lower'],upper=repair['target_upper'],margin=repair['margin'],
        margin_weight=repair['margin_weight'],probability=ad['probability'],decay_steps=ad['decay_environment_steps'],
        risk_radius=ad['risk_radius'],rng_offset=ad['rng_seed_offset'])
    if method=='astar_prefill_td': return AStarValueRepairAgent(ac,problem,repair_method='unguided_bound',**kwargs)
    cls=FixedTeacherCEAgent if method=='ce' else PersistentTeachingAgent
    agent=cls(ac,problem,repair_method='advice_bound_margin',advice_floor=0.,
        supervision_floor=0. if method=='asvc_withdraw' else .1,**kwargs)
    if method=='asvc_no_td': agent.bound_enabled=False
    return agent

def load_demos(map_seed,kind='plain'):
    root=DATA/f'map_{map_seed}'
    with np.load(root/'demonstrations.npz',allow_pickle=False) as z: data={k:z[k] for k in z.files}
    transitions=[Transition(
        __import__('astar_d3qn.envs.types',fromlist=['Observation']).Observation(data['spatial'][i].astype(np.float32),data['scalars'][i]),
        int(data['action'][i]),float(data['reward'][i]),
        __import__('astar_d3qn.envs.types',fromlist=['Observation']).Observation(data['next_spatial'][i].astype(np.float32),data['next_scalars'][i]),
        bool(data['terminated'][i]),data['next_mask'][i]) for i in range(len(data['action']))]
    if kind=='plain': return transitions
    collector=ThreeStepCollector() if kind=='qdagger' else NStepCollector(10,.99)
    teacher=ObservableTeacher(map_data(map_seed)[0]);records=[];ep=0
    for i,t in enumerate(transitions):
        end=bool(data['episode_end'][i])
        if kind=='qdagger':
            label=teacher.query(t.state)
            assert label==t.action,'Demonstration and online teacher differ'
            records.extend(collector.add(t,label,ep,episode_end=end))
        else: records.extend(collector.add(t,episode_end=end))
        ep+=int(end)
    assert not collector.pending and len(records)==len(transitions)
    return records

def evaluate(agent,map_seed,scenes,*,trace=True):
    from astar_d3qn.evaluation.rollout import evaluate_agent
    from unittest.mock import patch
    factory=FrozenFactory(map_data(map_seed)[0],scenes,trace=trace)
    before=state_digest(agent.training_state_dict());clock=perf_counter()
    with preserved_evaluation(agent),patch.object(agent,'train_batch',side_effect=AssertionError('Evaluation update')):
        _,details,_=evaluate_agent(agent,[factory.problem]*len(scenes),**env_kwargs(),environment_factory=factory,mask_static_invalid_actions=True)
    assert before==state_digest(agent.training_state_dict())
    assert [str(r['scenario_id']) for r in details]==[str(s['scenario_id']) for s in scenes]
    summary=summarize(details)
    summary['mean_action_latency_ms']=float(np.mean([r['mean_action_latency_ms'] for r in details]))
    failures=[dict(scenario_id=r['scenario_id'],termination_reason=r['termination_reason'],trace=e.trace)
        for r,e in zip(details,factory.environments) if not r['safe_success']] if trace else []
    return summary,details,failures,perf_counter()-clock

def source_files():
    import ast
    files=[p for p in (ROOT/'src/astar_d3qn').rglob('*.py') if '__pycache__' not in p.parts]
    names=['scripts/multimap_layout_common.py','scripts/prepare_multimap_layout.py','scripts/run_multimap_layout.py',
        'scripts/analyze_multimap_layout.py','scripts/start_multimap_layout.ps1','tests/test_multimap_layout.py',
        'scripts/fixed_ce_agent.py','scripts/fixed_qdagger_agent.py','scripts/fixed_supplement_common.py',
        'scripts/teaching_efficiency_common.py','scripts/run_runtime_path_guidance.py','scripts/train_whole_map_route_pool_pilot.py',
        'scripts/design_dynamic_route_pool_v1.py','scripts/design_dynamic_route_pool_v2.py',
        'configs/asvc_multimap_layout_v1.yaml','configs/dynamic_route_pool_v2.yaml','docs/ASVC_MULTIMAP_LAYOUT.zh-CN.md']
    files += [ROOT/name for name in names]
    # Freeze the transitive local script imports, including functions imported
    # inside the runner. All source package modules are already included above.
    pending=list(files);seen=set(files)
    while pending:
        p=pending.pop()
        if p.suffix!='.py':continue
        tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                candidate=ROOT/'scripts'/(module.split('.')[0]+'.py')
                if candidate.exists() and candidate not in seen:
                    seen.add(candidate);files.append(candidate);pending.append(candidate)
    files += [ROOT/settings()['reference_config'],ROOT/settings()['dqfd_config']]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
def verify_freeze():
    frozen=read(DATA/'freeze.json')
    assert frozen['config']==settings() and frozen['shared_config']==shared_config()
    assert frozen['source_sha256']==source_files(),'Source differs from frozen study; refuse mixed runs'
    for name,digest in frozen['data_sha256'].items(): assert sha(DATA/name)==digest,name
    return frozen

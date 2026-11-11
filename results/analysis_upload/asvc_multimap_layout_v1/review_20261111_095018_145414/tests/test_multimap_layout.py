"""Scientific invariants and short six-method recovery/parity, never formal scores."""
from __future__ import annotations
from collections import deque
from dataclasses import fields,is_dataclass,replace
import hashlib
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import multimap_layout_common as c
from run_multimap_layout import Experiment
from astar_d3qn.agents.d3qn import double_dqn_targets
from astar_d3qn.maps.adaptation import irregular_problem
from astar_d3qn.training.trainer import train_d3qn
from astar_d3qn.replay.transition import Transition
from fixed_qdagger_agent import ThreeStepCollector,coefficient

def fingerprint(exp):
    def visit(x):
        if isinstance(x,torch.Tensor):return ('tensor',str(x.dtype),tuple(x.shape),x.detach().cpu().numpy().tobytes())
        if isinstance(x,np.ndarray):return ('array',str(x.dtype),x.shape,x.tobytes())
        if isinstance(x,dict):return tuple((visit(k),visit(v)) for k,v in sorted(x.items(),key=lambda kv:str(kv[0])) if not isinstance(k,str) or ('seconds' not in k and k!='pid'))
        if isinstance(x,(list,tuple,deque)):return tuple(visit(v) for v in x)
        if isinstance(x,(set,frozenset)):return tuple(sorted((visit(v) for v in x),key=repr))
        if isinstance(x,random.Random):return visit(x.getstate())
        if isinstance(x,np.random.Generator):return visit(x.bit_generator.state)
        if is_dataclass(x):return (type(x).__name__,visit({f.name:getattr(x,f.name) for f in fields(x)}))
        if hasattr(x,'__dict__'):return (type(x).__name__,visit(vars(x)))
        return x
    return hashlib.sha256(repr(visit(exp.snapshot())).encode()).hexdigest()

class MultimapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        c.OUT.mkdir(parents=True,exist_ok=True)
        cls.maps=c.read(c.DATA/'maps_frozen.json');cls.map_seed=cls.maps['maps'][0]['map_seed']

    def test_original_generator_and_three_new_maps(self):
        reference=c.read(ROOT/'data/risk_handover_v1/manifest.json')['maps'][0]['problem']
        self.assertEqual(irregular_problem(91701).grid_sha256,reference['grid_sha256'])
        self.assertEqual(len(self.maps['maps']),3);self.assertEqual(len({m['grid_sha256'] for m in self.maps['maps']}),3)
        for m in self.maps['maps']:
            p=c.map_data(m['map_seed'])[0]
            self.assertEqual(p.start,(3,3));self.assertEqual(p.goal,(36,36));self.assertEqual(p.size,40)
            self.assertEqual(irregular_problem(p.seed).grid_sha256,p.grid_sha256)
        chosen=max(self.maps['maps'],key=lambda r:(r['nominal_mean_inverse_clearance'],-r['map_seed']))['map_seed']
        self.assertEqual(chosen,self.maps['ablation_map_seed'])

    def test_protected_old_results_unchanged(self):
        for filename in ('existing_evidence.json','completed_evaluation_audit.json'):
            for name,digest in c.read(c.DATA/filename)['protected_sha256'].items():
                self.assertEqual(c.sha(ROOT/name),digest,name)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_six_method_smoke(self):
        with TemporaryDirectory(dir=c.OUT) as temp:
            for method in c.settings()['methods']:
                e=Experiment(self.map_seed,method,0,Path(temp)/method,unit=True,device='cuda')
                if e.phase=='offline':e.offline(end=2);e.begin_online()
                for _ in range(12):e.step(budget=12,learning_starts=8,batch_size=4)
                self.assertEqual(e.online_updates,5)
                self.assertTrue(all(bool(torch.isfinite(p).all()) for p in e.agent.policy_network.parameters()))
                e.checkpoint();resumed=Experiment(self.map_seed,method,0,Path(temp)/method,resume=True,unit=True,device='cuda')
                self.assertEqual(fingerprint(e),fingerprint(resumed))

    def test_frozen_scene_sequence_and_split_isolation(self):
        for m in self.maps['maps']:
            map_seed=m['map_seed'];val,test=c.heldout(map_seed)
            self.assertEqual((len(val),len(test)),(50,500))
            keys=[c.scene_key(s) for s in val+test];self.assertEqual(len(set(keys)),550)
            prefixes=c.read(c.DATA/f'map_{map_seed}'/'training_stream_prefixes.json')
            for seed in range(5):
                streams=[c.training_stream(map_seed,seed) for _ in range(2)]
                for i in range(8):
                    records=[c.scene_record(stream(**c.env_kwargs())) for stream in streams]
                    self.assertEqual(c.canonical(records[0]),c.canonical(records[1]))
                    self.assertEqual(c.canonical(records[0]),c.canonical(prefixes[str(seed)][i]))
                    self.assertNotIn(c.scene_key(records[0]),set(keys))
                    cells=[set(map(tuple,s['route'])) for s in records[0]['obstacles']]
                    for j,left in enumerate(cells):
                        for right in cells[j+1:]:self.assertFalse(left&right)

    def test_method_flags_masks_and_schedules(self):
        seed=self.map_seed;agents={m:c.make_agent(m,seed,0,'cpu') for m in c.settings()['methods']+c.settings()['ablation_methods']}
        digests=[c.state_digest(a.policy_network.state_dict()) for a in agents.values()]
        self.assertEqual(len(set(digests)),1)
        self.assertFalse(hasattr(agents['d3qn'],'static_advice'));self.assertFalse(agents['dqfd'].bound_enabled)
        self.assertFalse(hasattr(agents['qdagger'],'static_advice'))
        self.assertTrue(agents['astar_prefill_td'].bound_enabled);self.assertEqual(agents['astar_prefill_td'].imitation_weight(),0)
        self.assertEqual(agents['astar_prefill_td'].advice_probability(),0)
        for t in [0,1,500,90000,99999,100000,120000,200000]:
            for a in agents.values():
                if hasattr(a,'training_action_steps'):a.training_action_steps=t
            self.assertEqual(agents['asvc'].advice_probability(),agents['ce'].advice_probability())
            self.assertEqual(agents['asvc'].imitation_weight(),agents['ce'].imitation_weight())
            self.assertEqual(agents['asvc_no_td'].imitation_weight(),agents['asvc'].imitation_weight())
        self.assertEqual(agents['asvc'].imitation_weight(),.1);self.assertEqual(agents['asvc_withdraw'].imitation_weight(),0)
        self.assertFalse(agents['asvc_no_td'].bound_enabled);self.assertTrue(agents['asvc_withdraw'].bound_enabled)
        self.assertEqual(coefficient(120000),0)

    def test_terminal_timeout_full_target_and_multistep(self):
        rewards=torch.tensor([10.,-1.,-.01]);terminal=torch.tensor([True,True,False])
        policy=torch.tensor([[2.,20.]]*3);target=torch.tensor([[7.,30.]]*3)
        raw=double_dqn_targets(rewards,terminal,.99,policy,target)
        np.testing.assert_allclose(raw.numpy(),[10.,-1.,29.69],rtol=1e-6)
        for method in ['asvc','asvc_no_td','astar_prefill_td']:
            a=c.make_agent(method,self.map_seed,0,'cpu');value=a._td_targets(rewards,terminal,policy,target,None)
            np.testing.assert_allclose(value.numpy(),raw.clamp(-6,10).numpy() if method!='asvc_no_td' else raw.numpy())
        env=c.training_stream(self.map_seed,0)(**c.env_kwargs());state=env.reset();mask=env.action_mask(True)
        collector=ThreeStepCollector();records=[]
        for i in range(5):
            records+=collector.add(Transition(state,4,1.,state,False,mask),4,0,episode_end=i==4)
        self.assertEqual([r.n_length for r in records],[3,3,3,2,1]);self.assertFalse(collector.pending)
        self.assertTrue(all(not r.n_terminated for r in records))

    def test_demonstrations_map_teacher_and_boundaries(self):
        for row in self.maps['maps']:
            seed=row['map_seed'];reg=c.read(c.DATA/f'map_{seed}'/'demonstration_registration.json')
            self.assertEqual(reg['attempts'],128);self.assertEqual(reg['successful_attempts']+reg['failed_attempts'],128)
            q=c.load_demos(seed,'qdagger');dq=c.load_demos(seed,'dqfd')
            self.assertEqual(len(q),reg['transitions']);self.assertEqual(len(q),len(dq))
            self.assertTrue(all(r.n_length<=3 for r in q));self.assertTrue(all(r.n_length<=10 for r in dq))

    def test_six_method_smoke_complete_recovery(self):
        reports=[]
        with TemporaryDirectory(dir=c.OUT) as temp:
            for method in c.settings()['methods']+c.settings()['ablation_methods']:
                def execute(path,interrupted):
                    e=Experiment(self.map_seed,method,0,path,unit=True,device='cpu')
                    if e.phase=='offline':
                        e.offline(end=3)
                        if interrupted:
                            e.checkpoint();e=Experiment(self.map_seed,method,0,path,resume=True,unit=True,device='cpu')
                        e.offline(end=6);e.begin_online()
                    for _ in range(12):e.step(budget=24,learning_starts=8,batch_size=4)
                    self.assertIsNotNone(e.env)
                    if interrupted:
                        e.checkpoint();e=Experiment(self.map_seed,method,0,path,resume=True,unit=True,device='cpu')
                    for _ in range(12):e.step(budget=24,learning_starts=8,batch_size=4)
                    self.assertEqual(e.online_updates,17);self.assertIsNone(e.env)
                    return e
                plain=execute(Path(temp)/method/'plain',False);resumed=execute(Path(temp)/method/'resume',True)
                self.assertEqual(fingerprint(plain),fingerprint(resumed),method)
                # Patches in validate forbid teacher and update calls, plus full state digest isolation.
                scenes=c.heldout(self.map_seed)[0][:2]
                before=c.state_digest(resumed.agent.training_state_dict());resumed.validate('unit',scenes=scenes)
                self.assertEqual(before,c.state_digest(resumed.agent.training_state_dict()))
                reports.append(dict(method=method,online_steps=24,online_updates=17,offline_updates=resumed.offline_updates,
                    full_resume_equal=True,autonomous_evaluation=True))
        c.write(reports,c.RESULTS/'six_method_smoke.json')

    def test_new_uniform_loop_matches_original_learner(self):
        # Same initialization, stream, epsilon, actions, replay draws and updates.
        map_seed=self.map_seed
        class Factory:
            def __init__(self):self.reset_schedule()
            def reset_schedule(self):self.stream=c.training_stream(map_seed,0)
            def __call__(self,problem,**kwargs):return self.stream(**kwargs)
        with TemporaryDirectory(dir=c.OUT) as temp:
            for method in ['d3qn','astar_prefill_td','asvc','ce']:
                a=c.make_agent(method,map_seed,0,'cpu');factory=Factory()
                config=replace(c._training_config(c.shared_config(),0),max_environment_steps=24,learning_starts=8,batch_size=4,
                    allow_partial_epsilon_schedule=True,progress_interval_environment_steps=None)
                demos=c.load_demos(map_seed) if method=='astar_prefill_td' else ()
                if demos:config=replace(config,replay_strategy='prefill')
                expected=train_d3qn([c.map_data(map_seed)[0]],a,config,demonstrations=demos,reward_config=c._reward_config(c.shared_config()),environment_factory=factory)
                e=Experiment(map_seed,method,0,Path(temp)/method,unit=True,device='cpu')
                for _ in range(24):e.step(budget=24,learning_starts=8,batch_size=4)
                self.assertEqual(expected.gradient_updates,e.online_updates)
                self.assertEqual(c.state_digest(a.training_state_dict()),c.state_digest(e.agent.training_state_dict()),method)

if __name__=='__main__':unittest.main()

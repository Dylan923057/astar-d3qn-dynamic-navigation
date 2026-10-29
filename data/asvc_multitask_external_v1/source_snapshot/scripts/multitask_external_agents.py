"""Unbounded original D3QN and original DQfD objective with additional diagnostics."""
from __future__ import annotations

import ast
from collections import Counter
import inspect
import textwrap

import numpy as np
import torch

from astar_d3qn.agents.d3qn import D3QNAgent,D3QNConfig
from astar_d3qn.agents.dqfd import DQfDAgent
from astar_d3qn.replay.transition import Transition
from astar_d3qn.replay.dqfd import DQfDReplay
from multitask_pilot_domain import static_mask


def dqfd_logged_update():
    original=DQfDAgent.train_dqfd
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0];clips=0
    for index,node in enumerate(function.body):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.nn.utils.clip_grad_norm_':
            function.body[index:index+1]=ast.parse('gradient_norm = '+ast.unparse(node.value)+'\nif not bool(torch.isfinite(gradient_norm)):\n    raise FloatingPointError("Nonfinite DQfD gradient norm")').body
            clips+=1;break
    if clips!=1:raise ValueError('Legacy DQfD update changed; inspect before adapting')
    returned=next(n for n in function.body if isinstance(n,ast.Return))
    if not isinstance(returned.value,ast.Call) or ast.unparse(returned.value.func)!='dict':raise ValueError('DQfD metric contract changed')
    for name,expression in dict(q_min='float(q.detach().min().item())',q_max='float(q.detach().max().item())',
        raw_one_target_min='float(raw_one.min().item())',raw_n_target_min='float(raw_n.min().item())',
        gradient_norm_before_clip='float(gradient_norm.item())').items():
        returned.value.keywords.append(ast.keyword(arg=name,value=ast.parse(expression,mode='eval').body))
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<unchanged-dqfd-objective-extra-metrics>','exec'),namespace)
    return namespace[function.name]


DQFD_UPDATE=dqfd_logged_update()


class ExternalD3QNAgent(D3QNAgent):
    def __init__(self,config):
        super().__init__(config);self.action_steps=0;self.sample_stats=Counter();self.last_targets={}

    def collect_action(self,state,task,position,epsilon):
        self.action_steps+=1
        return D3QNAgent.select_action(self,state,epsilon,np.flatnonzero(static_mask(state)).tolist())

    def _td_targets(self,*args):
        target=super()._td_targets(*args)
        if not bool(torch.isfinite(target).all()):raise FloatingPointError('Nonfinite raw D3QN target')
        self.last_targets=dict(raw_target_min=float(target.min()),raw_target_max=float(target.max()),target_clip_fraction=0.)
        return target

    def train_task_batch(self,batch):
        # Extra task metadata remains in replay, while the exact original learner
        # receives only the real transition fields it expects.
        transitions=[Transition(t.state,t.action,t.reward,t.next_state,t.terminated,t.next_action_mask) for t in batch]
        ranges={}
        def capture(module,args,value):
            if not ranges:ranges.update(q_min=float(value.detach().min()),q_max=float(value.detach().max()))
        hook=self.policy_network.register_forward_hook(capture)
        try:metrics=D3QNAgent.train_batch(self,transitions)
        finally:hook.remove()
        if not all(np.isfinite(metrics[k]) for k in ('loss','q_abs_max','gradient_norm_before_clip')):
            raise FloatingPointError('Nonfinite D3QN update; retain failed seed')
        self.sample_stats.update(updates=1,sampled_draws=len(batch),demonstration_draws=0)
        metrics.pop('td_errors',None)
        return {k:metrics[k] for k in ('loss','td_loss','q_abs_max','gradient_norm_before_clip')}|ranges|self.last_targets|dict(demonstration_samples=0,sample_count=len(batch))

    def training_state_dict(self):
        return super().training_state_dict()|dict(external=dict(action_steps=self.action_steps,sample_stats=self.sample_stats))

    def load_training_state_dict(self,state):
        super().load_training_state_dict(state);self.action_steps=state['external']['action_steps'];self.sample_stats=state['external']['sample_stats']


class ExternalDQfDAgent(DQfDAgent):
    def __init__(self,config,settings):
        super().__init__(config,settings,bound=False);self.action_steps=0;self.sample_stats=Counter()

    def collect_action(self,state,task,position,epsilon):
        self.action_steps+=1
        return D3QNAgent.select_action(self,state,epsilon,np.flatnonzero(static_mask(state)).tolist())

    def train_logged(self,records,weights,demonstrations):
        metrics=DQFD_UPDATE(self,records,weights,demonstrations)
        if not all(np.isfinite(metrics[k]) for k in ('loss','q_abs_max','gradient_norm_before_clip')):raise FloatingPointError('Nonfinite DQfD update')
        self.sample_stats.update(updates=1,sampled_draws=len(records),demonstration_draws=int(np.sum(demonstrations)))
        return metrics

    def training_state_dict(self):
        return super().training_state_dict()|dict(external=dict(action_steps=self.action_steps,sample_stats=self.sample_stats))

    def load_training_state_dict(self,state):
        super().load_training_state_dict(state);self.action_steps=state['external']['action_steps'];self.sample_stats=state['external']['sample_stats']


class StatefulDQfDReplay(DQfDReplay):
    def state_dict(self):
        return {k:v for k,v in vars(self).items() if k!='rng'}|dict(rng_state=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        for field in ('demo_count','online_capacity','alpha','agent_bonus','demo_bonus'):
            if state[field]!=getattr(self,field):raise ValueError('PER checkpoint configuration differs: '+field)
        vars(self).update({k:v for k,v in state.items() if k!='rng_state'})
        self.rng=np.random.default_rng();self.rng.bit_generator.state=state['rng_state']


def make_agent(cfg,settings,method,seed,device=None):
    values=dict(cfg['agent']);values['spatial_shape']=tuple(values['spatial_shape'])
    if device:values['device']=device
    config=D3QNConfig(seed=seed,**values)
    if method=='d3qn':return ExternalD3QNAgent(config)
    if method=='dqfd':return ExternalDQfDAgent(config,settings)
    raise ValueError('No extra training method is permitted')

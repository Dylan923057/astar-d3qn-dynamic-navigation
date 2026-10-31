"""One-factor ablation: remove executed-action equality from margin eligibility."""
from __future__ import annotations

from collections import Counter

import numpy as np

from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent
from astar_d3qn.agents.independent_teacher_actions import train_batch_with_teacher_actions


class AStarAllEligibleSupervisionAgent(AStarValueRepairAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.repair_method!='advice_bound_margin':raise ValueError('Use original complete combination, not staged withdrawal.')
        self.supervision_bins={}

    def supervision_action(self,transition):
        action=self.static_advice(transition.state)
        if action is None:return None
        if transition.terminated and transition.reward==-1.0:return None
        if not self.current_static_mask(transition.state)[action] or self.observed_risk(transition.state,action):return None
        return int(action)

    def teacher_label(self,transition):
        return self.supervision_action(transition) is not None

    def train_batch(self,batch,**kwargs):
        weight=self.imitation_weight()
        taught=[self.supervision_action(t) for t in batch] if weight>0 else None
        labels=[a is not None for a in taught] if taught is not None else None
        # Placeholder executed actions for ineligible rows are ignored by margin.
        # The original immutable transitions are passed unchanged to TD.
        teacher_actions=[int(t.action) if a is None else a for t,a in zip(batch,taught)] if taught is not None else None
        kwargs.update(demonstration_mask=labels,demo_margin=self.margin if weight>0 else 0.0,
            demo_loss_weight=weight,
            demo_competitor_mask=[self.current_static_mask(t.state) for t in batch] if weight>0 else None)
        metrics=train_batch_with_teacher_actions(self,batch,teacher_actions=teacher_actions,**kwargs)
        metrics.update(self._target_metrics)
        count=sum(labels) if labels is not None else 0
        matched=sum(a is not None and a==t.action for t,a in zip(batch,taught)) if taught is not None else 0
        metrics.update(teacher_label_count=count,teacher_margin_weight=weight,teacher_label_matching_count=matched)
        if not np.isfinite([metrics['q_mean'],metrics['q_abs_max'],metrics['loss']]).all():
            raise FloatingPointError('Nonfinite Q values/loss; stop rather than adjust settings.')
        index=max(0,self.training_action_steps-1)//1000
        row=self.repair_bins.setdefault(index,Counter())
        row['updates']+=1;row['q_abs_max']=max(row['q_abs_max'],metrics['q_abs_max'])
        row['raw_target_max']=max(row.get('raw_target_max',float('-inf')),metrics['raw_target_max'])
        row['teacher_label_samples']+=count
        for key in ('q_mean','target_mean','raw_target_mean','raw_target_outside_fraction',
            'target_clip_fraction','td_loss','demo_margin_loss','teacher_margin_weight'):
            row[key+'_sum']+=metrics[key]
        extra=self.supervision_bins.setdefault(index,Counter())
        extra['updates']+=1;extra['sampled_draws']+=len(batch)
        extra['supervised_draws']+=count;extra['matching_supervised_draws']+=matched
        extra['q_min']=min(extra.get('q_min',float('inf')),metrics['q_min'])
        extra['q_max']=max(extra.get('q_max',float('-inf')),metrics['q_max'])
        return metrics

    def supervision_records(self):
        return [dict(bin_start_step=i*1000+1,bin_end_step=min((i+1)*1000,self.training_action_steps),**dict(r),
            supervision_sample_fraction=r['supervised_draws']/r['sampled_draws'],
            matching_fraction_among_supervised=r['matching_supervised_draws']/r['supervised_draws'] if r['supervised_draws'] else None,
            labels_computed=i*1000<100000) for i,r in sorted(self.supervision_bins.items())]

    def training_state_dict(self):
        state=super().training_state_dict()
        if self.supervision_bins:state['supervision_alignment_bins']={i:dict(r) for i,r in self.supervision_bins.items()}
        return state

    def load_training_state_dict(self,state):
        super().load_training_state_dict(state)
        self.supervision_bins={int(i):Counter(r) for i,r in state.get('supervision_alignment_bins',{}).items()}

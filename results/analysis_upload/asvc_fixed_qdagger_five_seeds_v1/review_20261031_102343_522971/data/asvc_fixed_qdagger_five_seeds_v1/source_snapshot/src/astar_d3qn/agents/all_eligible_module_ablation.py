"""Two isolated removals from the frozen all-eligible complete method."""
from astar_d3qn.agents.astar_all_eligible_supervision import AStarAllEligibleSupervisionAgent

METHODS=('advice_margin_all_eligible','supervision_only_bound_all_eligible')


class AllEligibleModuleAblationAgent(AStarAllEligibleSupervisionAgent):
    def __init__(self,*args,ablation_method,**kwargs):
        if ablation_method not in METHODS:raise ValueError('Only the two registered removals are allowed.')
        super().__init__(*args,**kwargs)
        self.ablation_method=ablation_method
        if ablation_method=='advice_margin_all_eligible':
            self.bound_enabled=False
        else:
            self.method='unguided'  # Inherited selector advances real clock/RNG; margin stays enabled.

    def train_batch(self,batch,**kwargs):
        metrics=super().train_batch(batch,**kwargs)
        index=max(0,self.training_action_steps-1)//1000
        row=self.supervision_bins[index]
        row['raw_target_min']=min(row.get('raw_target_min',float('inf')),metrics['raw_target_min'])
        row['raw_target_max']=max(row.get('raw_target_max',float('-inf')),metrics['raw_target_max'])
        return metrics

    def training_state_dict(self):
        state=super().training_state_dict()
        if self._static_action_cache:state['module_static_action_cache']=dict(self._static_action_cache)
        return state

    def load_training_state_dict(self,state):
        super().load_training_state_dict(state)
        self._static_action_cache=dict(state.get('module_static_action_cache',{}))

"""Isolated same-teacher CE: preserve every original TD/update statement.

No historical source file is changed. AST construction replaces only the
demonstration loss block and supplies independent teacher indices. The original
teacher/risk/collision filtering and bookkeeping wrapper are reused verbatim
except its call to the loss implementation. Compatibility checkpoint log keys
retain 'margin'; published CE CSV columns are explicitly renamed by the runner.
"""
from __future__ import annotations
import ast
import inspect
import textwrap

import torch
import torch.nn.functional as F

from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.agents.astar_all_eligible_supervision import AStarAllEligibleSupervisionAgent
from astar_d3qn.agents.persistent_teaching import PersistentTeachingAgent


def masked_teacher_ce(q, teachers, eligible, valid):
    """T=1. Only eligible rows and statically legal columns enter softmax."""
    if q.ndim != 2 or teachers.shape != (q.shape[0],) or eligible.shape != teachers.shape or valid.shape != q.shape:
        raise ValueError('CE labels/masks must match batch and action dimensions')
    if not bool(eligible.any()):
        return q.new_zeros(())
    logits, labels, legal = q[eligible], teachers[eligible], valid[eligible]
    if not bool(((labels>=0)&(labels<q.shape[1])).all()) or not bool(legal.gather(1,labels[:,None]).all()):
        raise ValueError('Eligible teacher actions must be statically legal')
    return F.cross_entropy(logits.masked_fill(~legal,float('-inf')), labels, reduction='mean')


def build_ce_update():
    original=D3QNAgent.train_batch
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0]
    candidates=[(i,node) for i,node in enumerate(function.body) if isinstance(node,ast.If)
                and 'demonstration_mask' in ast.unparse(node.test) and 'demo_loss_weight' in ast.unparse(node.test)]
    if len(candidates)!=1:raise ValueError('Historical loss structure changed; audit before continuing')
    index,old_block=candidates[0]
    original_outside=[ast.dump(node) for i,node in enumerate(function.body) if i not in (index,len(function.body)-1)]
    function.args.kwonlyargs.append(ast.arg(arg='teacher_actions'))
    function.args.kw_defaults.append(ast.Constant(None))
    replacement=ast.parse('''if demonstration_mask is not None and demo_loss_weight > 0.0:
    if teacher_actions is None or demo_competitor_mask is None:
        raise ValueError("CE requires independent teacher labels and static action masks")
    ce_teachers = torch.as_tensor(teacher_actions, dtype=torch.long, device=self.device)
    ce_eligible = torch.as_tensor(demonstration_mask, dtype=torch.bool, device=self.device)
    ce_legal = torch.as_tensor(demo_competitor_mask, dtype=torch.bool, device=self.device)
    demo_loss = masked_teacher_ce(policy_q_values, ce_teachers, ce_eligible, ce_legal)
''').body[0]
    function.body[index]=replacement
    if original_outside != [ast.dump(node) for i,node in enumerate(function.body) if i not in (index,len(function.body)-1)]:
        raise AssertionError('A non-supervision TD/update statement changed')
    result=function.body[-1]
    if not isinstance(result,ast.Return) or not isinstance(result.value,ast.Dict):raise ValueError('Metrics schema changed')
    result.value.keys.extend([ast.Constant('q_min'),ast.Constant('q_max')])
    result.value.values.extend([ast.parse('float(policy_q_values.detach().min().item())',mode='eval').body,
                               ast.parse('float(policy_q_values.detach().max().item())',mode='eval').body])
    namespace=dict(original.__globals__,masked_teacher_ce=masked_teacher_ce)
    exec(compile(ast.fix_missing_locations(tree),'<fixed-start-goal-same-teacher-ce>','exec'),namespace)
    return namespace[function.name],ast.unparse(tree),dict(non_supervision_statements_identical=True,
        td_actions='unchanged Transition.action',new_temperature=1.,replaced_loss_blocks=1)


CE_UPDATE,GENERATED_UPDATE_SOURCE,AST_AUDIT=build_ce_update()


def build_teacher_wrapper():
    original=AStarAllEligibleSupervisionAgent.train_batch
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed=0
    class ReplaceCall(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal changed
            if isinstance(node.func,ast.Name) and node.func.id=='train_batch_with_teacher_actions':
                node.func.id='CE_UPDATE';changed+=1
            return self.generic_visit(node)
    tree=ReplaceCall().visit(tree)
    if changed!=1:raise ValueError('Teacher wrapper structure changed')
    namespace=dict(original.__globals__,CE_UPDATE=CE_UPDATE)
    exec(compile(ast.fix_missing_locations(tree),'<same-teacher-filter-wrapper-ce>','exec'),namespace)
    return namespace['train_batch'],ast.unparse(tree)


CE_WRAPPER,GENERATED_WRAPPER_SOURCE=build_teacher_wrapper()


class FixedTeacherCEAgent(PersistentTeachingAgent):
    def train_batch(self,batch,**kwargs):
        metrics=CE_WRAPPER(self,batch,**kwargs)
        metrics['supervision_ce_loss']=metrics['demo_margin_loss']
        metrics['supervision_weighted_loss']=metrics['teacher_margin_weight']*metrics['demo_margin_loss']
        row=self.repair_bins[max(0,self.training_action_steps-1)//1000]
        row['total_loss_sum']+=metrics['loss']
        row['supervision_weighted_loss_sum']+=metrics['supervision_weighted_loss']
        return metrics

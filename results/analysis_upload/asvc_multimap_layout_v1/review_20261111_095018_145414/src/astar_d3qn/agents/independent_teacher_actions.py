"""Independent optional teacher-action interface; registered D3QN stays untouched.

The historical function is reused verbatim except the four action references in
its existing margin block. TD, rewards, next states, masks and update operations
remain the original statements. Freeze includes this file and the source template.
"""
from __future__ import annotations

import ast
import copy
import inspect
import textwrap

from astar_d3qn.agents.d3qn import D3QNAgent


def _build():
    original=D3QNAgent.train_batch
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0]
    function.args.kwonlyargs.append(ast.arg(arg='teacher_actions'))
    function.args.kw_defaults.append(ast.Constant(value=None))
    candidates=[(i,n) for i,n in enumerate(function.body) if isinstance(n,ast.If)
        and 'demonstration_mask' in ast.unparse(n.test)]
    # The length-check also mentions demonstration_mask; select the loss gate.
    candidates=[(i,n) for i,n in candidates if 'demo_loss_weight' in ast.unparse(n.test)]
    if len(candidates)!=1:raise ValueError('Historical margin structure changed; inspect before adapting.')
    index,block=candidates[0]
    counts={'actions':0,'predicted_q':0}
    class TeacherOnly(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in counts:
                counts[node.id]+=1
                return ast.copy_location(ast.Name(id='supervision_actions' if node.id=='actions' else 'supervision_q',ctx=node.ctx),node)
            return node
    changed=TeacherOnly().visit(copy.deepcopy(block))
    if counts!={'actions':3,'predicted_q':1}:raise ValueError('Unexpected historical margin action references.')
    preparation=ast.parse('''if teacher_actions is None:
    supervision_actions = actions
    supervision_q = predicted_q
else:
    if len(teacher_actions) != len(batch):
        raise ValueError("teacher_actions must match batch length.")
    supervision_actions = torch.as_tensor(teacher_actions, dtype=torch.long, device=self.device)
    if supervision_actions.shape != actions.shape or not bool(((supervision_actions >= 0) & (supervision_actions < self.action_dim)).all()):
        raise ValueError("teacher_actions must contain valid action indices.")
    supervision_q = policy_q_values.gather(1, supervision_actions.unsqueeze(1)).squeeze(1)
''').body
    function.body[index:index+1]=preparation+[changed]
    result=function.body[-1]
    if not isinstance(result,ast.Return) or not isinstance(result.value,ast.Dict):
        raise ValueError('Historical train_batch return schema changed.')
    result.value.keys.extend([ast.Constant(value='q_min'),ast.Constant(value='q_max')])
    result.value.values.extend([ast.parse('float(policy_q_values.detach().min().item())',mode='eval').body,
        ast.parse('float(policy_q_values.detach().max().item())',mode='eval').body])
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<independent-teacher-action-margin>', 'exec'),namespace)
    return namespace[function.name]


_EXTENDED=_build()


def train_batch_with_teacher_actions(agent,batch,*,teacher_actions=None,**kwargs):
    """TD uses Transition.action; only margin uses independent teacher_actions.

    None preserves historical numerical behavior; extra q_min/q_max are passive
    diagnostics. No registered/public old method or Transition is modified.
    """
    return _EXTENDED(agent,batch,teacher_actions=teacher_actions,**kwargs)

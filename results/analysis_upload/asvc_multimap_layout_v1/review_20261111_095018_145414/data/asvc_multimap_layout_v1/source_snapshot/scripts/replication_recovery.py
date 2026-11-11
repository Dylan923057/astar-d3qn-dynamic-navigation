"""Episode-boundary recovery hooks around *unchanged* registered training loops.

AST instrumentation adds only restoration before the main loop and checkpointing
after its episode-end validation. No action/replay/update statement is replaced.
Checkpoints are trusted local pickle files, not suitable for untrusted downloads.
"""
from __future__ import annotations

import ast
import inspect
import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


STAGE_LOCALS = ('replay', 'rng', 'order', 'order_position', 'environment_steps',
    'gradient_updates', 'records', 'episode', 'demo_samples_total',
    'online_samples_total', 'coverage')
DQFD_LOCALS = ('replay', 'collector', 'total', 'updates', 'episode', 'records')


def instrument(function, names):
    """Reuse the exact historical function with two non-learning hooks."""
    tree = ast.parse(inspect.getsource(function))
    node = tree.body[0]
    node.args.kwonlyargs.append(ast.arg(arg='_recovery'))
    node.args.kw_defaults.append(ast.Constant(value=None))
    loops = [i for i, s in enumerate(node.body) if isinstance(s, ast.While)]
    if len(loops) != 1:
        raise ValueError('Historical loop structure changed; inspect before adapting.')
    index = loops[0]
    restore = ast.parse('if _recovery is not None:\n    _restored = _recovery.restore(locals())\n' +
        ''.join(f'    {n} = _restored.get("{n}", {n})\n' for n in names)).body
    hook = ast.parse('if _recovery is not None:\n    _clock = perf_counter()\n'
        '    _recovery.checkpoint(locals())\n').body[0]
    time_name = 'progress_callback_seconds' if 'environment_steps' in names else 'callback_seconds'
    hook.body.extend(ast.parse(f'{time_name} += perf_counter() - _clock').body)
    node.body[index].body.append(hook)
    node.body[index:index] = restore
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<registered-loop-recovery-hooks>', 'exec'), namespace)
    return namespace[function.__name__]


def factory_state(factory):
    result = []
    while factory is not None:
        result.append({k:v for k,v in vars(factory).items() if k not in ('factory', 'agent')})
        factory = getattr(factory, 'factory', None)
    return result


def restore_factory(factory, states):
    for values in states:
        vars(factory).update(values)
        factory = getattr(factory, 'factory', None)


class Recovery:
    def __init__(self, directory, agent, factory, names, diagnostics=None, resume=False,
                 identity=None, checkpoint_interval=10000, stop_after=None):
        self.directory = Path(directory)
        self.agent, self.factory, self.names = agent, factory, names
        self.diagnostics, self.identity = diagnostics, identity
        self.checkpoint_interval, self.stop_after = checkpoint_interval, stop_after
        self.path = self.directory / 'recovery_latest.pth'
        self.loaded = torch.load(self.path, map_location=agent.device, weights_only=False) if resume else None
        if self.loaded and self.loaded['identity'] != identity:
            raise ValueError('Recovery source/config/seed identity mismatch.')
        self.last_step = self.loaded['step'] if self.loaded else 0
        self.prior_training_seconds = self.loaded['training_seconds'] if self.loaded else 0.

    def restore(self, context):
        if not self.loaded:
            return {}
        s = self.loaded
        self.agent.load_training_state_dict(s['agent'])
        vars(self.agent).update(s['agent_extra'])
        restore_factory(self.factory, s['factory'])
        if self.diagnostics is not None:
            vars(self.diagnostics).update(s['diagnostics'])
        random.setstate(s['python_rng'])
        np.random.set_state(s['numpy_rng'])
        torch.set_rng_state(s['torch_rng'].cpu())
        if s['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
        return s['loop']

    def checkpoint(self, context):
        stage = 'environment_steps' in self.names
        step = context['environment_steps' if stage else 'total']
        if step // self.checkpoint_interval == self.last_step // self.checkpoint_interval and step != 200000:
            return
        start = context['started_at' if stage else 'start']
        callback = context['progress_callback_seconds' if stage else 'callback_seconds']
        extra = {k:vars(self.agent)[k] for k in ('planner_calls', 'td_bins', '_static_cache')
                 if k in vars(self.agent)}
        state = dict(version=1, episode_boundary=True, step=step, identity=self.identity,
            agent=self.agent.training_state_dict(), agent_extra=extra,
            loop={n:context[n] for n in self.names}, factory=factory_state(self.factory),
            diagnostics=vars(self.diagnostics) if self.diagnostics is not None else None,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()
                if torch.cuda.is_initialized() else None,
            training_seconds=self.prior_training_seconds + perf_counter()-start-callback)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        torch.save(state, temporary)
        os.replace(temporary, self.path)
        self.last_step = step
        if self.stop_after is not None and step >= self.stop_after:
            raise InterruptedError('Deliberate checkpoint-resume verification.')

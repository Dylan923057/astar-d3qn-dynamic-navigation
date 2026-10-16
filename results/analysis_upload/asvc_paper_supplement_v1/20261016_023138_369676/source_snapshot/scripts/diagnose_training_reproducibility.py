"""Read-only tracing around the production runner; stop without changing its budget."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import train_whole_map_route_pool_pilot as runner
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv
from astar_d3qn.replay.demo import PersistentDemoReplay


class DiagnosticStop(Exception):
    pass


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode()); h.update(x.tobytes())
        elif dataclasses.is_dataclass(x):
            for f in dataclasses.fields(x):
                add(f.name); add(getattr(x, f.name))
        elif isinstance(x, dict):
            for k in sorted(x):
                add(k); add(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(str(len(x)).encode())
            for item in x: add(item)
        else:
            h.update(repr(x).encode()); h.update(b'\0')
    add(value)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', default='adaptive', choices=['adaptive', 'fixed_25_full_capacity'])
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--deterministic', action='store_true', help='Diagnostic backend control, never an algorithm change.')
    parser.add_argument('--capture-first-update', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    config = runner.load_config(runner._resolve('configs/whole_map_route_pool_91701_3to5_capacity_controlled_v1.yaml'))
    config['experiment']['method'] = args.method
    config['experiment']['output_root'] = str(out / 'runner_output')
    config['training']['replay_strategy'] = 'persistent_demo'
    # Only the runtime stop differs: max_environment_steps and epsilon/schedule remain 200000.
    runner._validate_protocol(config, *runner._load_inputs(config))
    (out/'diagnostic_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    metadata = {'method': args.method, 'seed': 0, 'stop_steps': args.steps,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED')}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    trace = (out/'trace.jsonl').open('w', encoding='utf-8', buffering=1)
    context = {'step': 0, 'validation': False, 'agent': None, 'replay': None, 'factory': None}
    timing = {'train_batch_seconds': 0.0, 'validation_seconds': 0.0}
    started = perf_counter()
    transition_hashes = {}

    def emit(event, **fields):
        trace.write(json.dumps({'event': event, 'step': context['step'], **fields}, sort_keys=True) + '\n')

    def global_rng():
        return {'python': digest(random.getstate()), 'numpy': digest(np.random.get_state()),
                'torch_cpu': digest(torch.get_rng_state()),
                'torch_cuda': digest(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None}

    def training_rng():
        result = global_rng()
        agent, replay, factory = context['agent'], context['replay'], context['factory']
        if agent: result['agent'] = digest(agent._rng.getstate())
        if replay is not None:
            result['replay_demo'] = digest(replay._rng.getstate())
            result['replay_online'] = digest(replay.online._rng.getstate())
        if factory is not None:
            result['environment_route'] = digest(factory._rng.getstate())
            result['environment_count'] = digest(factory._count_rng.getstate())
        return result

    orig_init = runner.D3QNAgent.__init__
    def init(agent, cfg):
        orig_init(agent, cfg)
        context['agent'] = agent
        metadata.update(
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        )
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        emit('network_initialization', policy=digest(agent.policy_network.state_dict()),
             target=digest(agent.target_network.state_dict()), rng=training_rng())
        torch.save(agent.policy_network.state_dict(), out/'network_initial.pth')
    runner.D3QNAgent.__init__ = init

    orig_factory = runner.WholeMapRoutePoolEnvironmentFactory.__call__
    def factory(self, *a, **kw):
        env = orig_factory(self, *a, **kw)
        if self.scenario_prefix.startswith('train'):
            context['factory'] = self
            emit('environment_creation', scenario=env.scenario_id,
                 specs=digest(env.dynamic_obstacles), routes=[s.label for s in env.dynamic_obstacles],
                 route_rng=digest(self._rng.getstate()), count_rng=digest(self._count_rng.getstate()))
        return env
    runner.WholeMapRoutePoolEnvironmentFactory.__call__ = factory

    orig_select = runner.D3QNAgent.select_action
    def select(agent, state, epsilon, valid_actions=None):
        if context['validation']:
            return orig_select(agent, state, epsilon, valid_actions)
        if context['step'] >= args.steps:
            raise DiagnosticStop()
        context['step'] += 1
        clone = random.Random(); clone.setstate(agent._rng.getstate())
        draw = clone.random() if epsilon > 0 else None
        exploring = draw is not None and draw < epsilon
        candidates = list(range(agent.action_dim)) if valid_actions is None else list(valid_actions)
        expected_random_action = clone.choice(candidates) if exploring else None
        q = []
        hook = agent.policy_network.register_forward_hook(lambda m, inputs, output: q.append(output.detach().cpu().tolist()))
        before = digest(agent._rng.getstate())
        try: action = orig_select(agent, state, epsilon, valid_actions)
        finally: hook.remove()
        emit('action', observation=digest(state), epsilon=epsilon, draw=draw, exploring=exploring,
             expected_random_action=expected_random_action, action=action, candidates=candidates,
             rng_before=before, rng_after=digest(agent._rng.getstate()), q=q)
        return action
    runner.D3QNAgent.select_action = select

    orig_step = DynamicGridNavigationEnv.step
    def step(env, action):
        if context['validation']: return orig_step(env, action)
        before = {'position': env.position, 'dynamic': env.dynamic_positions,
                  'indices': env._indices.copy(), 'directions': env._directions.copy(),
                  'move_counters': env._move_counters.copy(), 'history': env._history.copy()}
        result = orig_step(env, action)
        emit('environment_step', scenario=env.scenario_id, action=int(action), before=before,
             position=env.position, dynamic=env.dynamic_positions, result=digest(result))
        return result
    DynamicGridNavigationEnv.step = step

    orig_replay_init = PersistentDemoReplay.__init__
    def replay_init(self, *a, **kw):
        orig_replay_init(self, *a, **kw); context['replay'] = self
        for transition in self._demonstrations:
            transition_hashes[id(transition)] = (transition, digest(transition))
        emit('replay_initialization', demos=digest([digest(t) for t in self._demonstrations]),
             capacity=self.online.capacity, rng=training_rng())
    PersistentDemoReplay.__init__ = replay_init
    orig_add = PersistentDemoReplay.add
    def add(self, transition):
        transition_hashes[id(transition)] = (transition, digest(transition))
        return orig_add(self, transition)
    PersistentDemoReplay.add = add
    orig_sample = PersistentDemoReplay.sample
    def sample(self, size):
        before = (digest(self._rng.getstate()), digest(self.online._rng.getstate()))
        batch = orig_sample(self, size)
        emit('replay_sample', counts=self.sample_counts(size),
             ordered_batch=digest([transition_hashes[id(t)][1] for t in batch]), rng_before=before,
             rng_after=(digest(self._rng.getstate()), digest(self.online._rng.getstate())))
        return batch
    PersistentDemoReplay.sample = sample

    orig_train = runner.D3QNAgent.train_batch
    def train(agent, *a, **kw):
        handles, captured = [], {}
        if args.capture_first_update and agent.update_steps == 0:
            torch.save({'batch': a[0], 'agent_config': dataclasses.asdict(agent.config)}, out/'first_batch.pth')
            def capture(module, inputs, output):
                if output.requires_grad:
                    captured['input'] = inputs[0].detach().cpu().clone()
                    output.register_hook(lambda grad: captured.update(grad_output=grad.detach().cpu().clone()))
            convolution = agent.policy_network.encoder[2]
            captured['weight'] = convolution.weight.detach().cpu().clone()
            handles.append(convolution.register_forward_hook(capture))
            handles.append(convolution.weight.register_hook(lambda grad: captured.update(weight_gradient=grad.detach().cpu().clone())))
        batch_started = perf_counter()
        updates = orig_train(agent, *a, **kw)
        timing['train_batch_seconds'] += perf_counter() - batch_started
        for handle in handles: handle.remove()
        if captured:
            torch.save(captured, out/'first_conv2_backward.pth')
        emit('network_update', update=agent.update_steps, metrics=updates,
             policy=digest(agent.policy_network.state_dict()), target=digest(agent.target_network.state_dict()),
             rng=global_rng())
        if agent.update_steps <= 5:
            torch.save(agent.policy_network.state_dict(), out/f'network_update_{agent.update_steps}.pth')
        return updates
    runner.D3QNAgent.train_batch = train

    orig_eval = runner.evaluate_agent
    def evaluate(*a, **kw):
        before = training_rng(); context['validation'] = True
        validation_started = perf_counter()
        try: result = orig_eval(*a, **kw)
        finally: context['validation'] = False
        timing['validation_seconds'] += perf_counter() - validation_started
        after = training_rng()
        # Wall-clock latency is measured separately, not a reproducible outcome.
        latency_keys = {'action_selection_seconds', 'mean_action_latency_ms'}
        comparable_summary = {k: v for k, v in result[0].items() if k not in latency_keys}
        comparable_details = [{k: v for k, v in row.items() if k not in latency_keys} for row in result[1]]
        emit('validation', rng_before=before, rng_after=after, training_rng_unchanged=before == after,
             summary=comparable_summary, details=digest(comparable_details),
             trajectories=digest(result[2]))
        return result
    runner.evaluate_agent = evaluate
    try:
        runner._run_seed(config, 0)
    except DiagnosticStop:
        emit('diagnostic_stop', rng=training_rng())
        print(f'DIAGNOSTIC_COMPLETE steps={context["step"]} output={out}', flush=True)
    finally:
        trace.close()
        timing.update(wall_seconds=perf_counter() - started, steps=context['step'],
                      gradient_updates=context['agent'].update_steps if context['agent'] else 0)
        (out/'timing.json').write_text(json.dumps(timing, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

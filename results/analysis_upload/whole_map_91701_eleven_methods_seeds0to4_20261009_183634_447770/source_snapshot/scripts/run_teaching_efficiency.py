"""Train only the new paired ablation when --train is explicit. Default: zero-update checks."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import teaching_efficiency_common as common
from astar_d3qn.agents.astar_supervision_only import AStarSupervisionOnlyAgent
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything


def make_agent(config, problem, seed, device=None):
    seed_everything(seed)
    values = dict(config['agent'])
    if device:
        values['device'] = device
    advice, repair = config['training_advice'], config['value_repair']
    window = config['environment']['window_size']
    return AStarSupervisionOnlyAgent(D3QNConfig((5, window, window), 4, 5, seed=seed, **values),
                                    problem, lower=repair['target_lower'], upper=repair['target_upper'],
                                    margin=repair['margin'], margin_weight=repair['margin_weight'],
                                    probability=advice['probability'], decay_steps=advice['decay_environment_steps'],
                                    risk_radius=advice['risk_radius'], rng_offset=advice['rng_seed_offset'])


class CollectionDiagnostics:
    """Observed states and real outcomes only; no obstacle future lookup or action replacement."""
    def __init__(self, factory, agent, critical):
        self.factory, self.agent = factory, agent
        self.exact = {row['observation_sha256'] for row in critical}
        self.positions = {tuple(row['position']) for row in critical}
        self.bins = {}

    def reset_schedule(self):
        self.factory.reset_schedule()
        self.bins = {}

    def __call__(self, problem, **kwargs):
        env = self.factory(problem, **kwargs)
        owner = self

        class ObservedEnvironment:
            def __getattr__(self, name):
                return getattr(env, name)

            def reset(self):
                self.state = env.reset()
                return self.state

            def step(self, action):
                state, position = self.state, tuple(env.position)
                result = env.step(action)
                step = owner.agent.training_action_steps
                row = owner.bins.setdefault((step - 1) // 1000, Counter())
                row['collected_transitions'] += 1
                active = owner.agent.imitation_weight() > 0
                row['active_collected_transitions'] += int(active)
                if active:
                    transition = Transition(state, int(action), result.reward, result.observation,
                                            result.terminated, env.action_mask(True))
                    row['eligible_collected_labels'] += int(owner.agent.teacher_label(transition))
                if position in owner.positions:
                    row['critical_position_visits'] += 1
                    row[f'critical_position_action_{action}'] += 1
                    row['critical_position_collisions'] += int(result.info['collision'])
                    radius = state.spatial.shape[1] // 2
                    # Current obstacle two cells right, previously three right: observed head-on motion.
                    head_on = state.spatial[1, radius, radius + 2] > .5 and state.spatial[2, radius, radius + 3] > .5
                    row['observed_head_on_position_visits'] += int(head_on)
                    row['observed_head_on_collisions'] += int(head_on and result.info['collision'])
                    if common.observation_hash(state) in owner.exact:
                        row['exact_critical_observation_visits'] += 1
                        row[f'exact_critical_action_{action}'] += 1
                        row['exact_critical_collisions'] += int(result.info['collision'])
                self.state = result.observation
                return result

        return ObservedEnvironment()

    def records(self):
        keys = ('eligible_collected_labels', 'critical_position_visits', 'critical_position_collisions',
                'observed_head_on_position_visits', 'observed_head_on_collisions',
                'exact_critical_observation_visits', 'exact_critical_collisions',
                *(f'critical_position_action_{a}' for a in range(5)),
                *(f'exact_critical_action_{a}' for a in range(5)))
        return [dict(bin_start_step=i * 1000 + 1,
                     bin_end_step=min((i + 1) * 1000, self.agent.training_action_steps),
                     **{k: row[k] for k in ('collected_transitions', 'active_collected_transitions', *keys)},
                     collected_label_coverage=row['eligible_collected_labels'] / row['collected_transitions'])
                for i, row in sorted(self.bins.items())]


def source_hashes():
    files = [Path(__file__), common.CONFIG, Path(common.__file__),
             common.ROOT / 'src/astar_d3qn/agents/astar_supervision_only.py',
             *[common.ROOT / name for name in common.legacy.source_hashes()]]
    return {str(p.relative_to(common.ROOT)): common.sha256(p) for p in files}


def check(config, inputs, seeds):
    scenes, critical, manifest = common.load_frozen(config)
    initializations = []
    for seed in seeds:
        agent = make_agent(config, inputs[0], seed, 'cpu')
        original = common.read_json(common.run_directory(config, 'advice_bound_margin', seed) / 'result.json')
        initial = common.legacy.runtime.state_digest(agent.training_state_dict())
        if initial != original['initial_state_sha256']:
            raise RuntimeError(f'Seed {seed} differs from recorded full paired initialization.')
        if agent.advice_probability(1) != 0 or agent.imitation_weight() != 1:
            raise RuntimeError('No-override ablation did not keep the original supervision schedule.')
        initializations.append(dict(seed=seed, initial_state_sha256=initial, matches_recorded_baseline=True))
    kwargs = dict(max_steps=300, window_size=15, reward_config=common.legacy._reward_config(config), terminate_on_collision=True)
    factory = common.FrozenFactory(inputs[0], scenes)
    for scene in scenes:
        env = factory(inputs[0], **kwargs)
        state = env.reset()
        if state.spatial.shape != (5, 15, 15) or state.scalars.shape != (4,) or np.any(state.spatial[4]) or np.any(state.scalars[2:]):
            raise RuntimeError('Independent scene observation compatibility failed.')
        if not 3 <= len(env.dynamic_obstacles) <= 5 or env.scenario_id != scene['scenario_id']:
            raise RuntimeError('Independent frozen scene loading failed.')
        for spec in env.dynamic_obstacles:
            for time, cell in enumerate(common.obstacle_cycle(asdict(spec))):
                if cell in inputs[0].obstacles or cell in (inputs[0].start, inputs[0].goal):
                    raise RuntimeError('Invalid obstacle route.')
    # Verify plain-network weight compatibility without measuring performance on the 500 scenes.
    agent = common.legacy.runtime.make_agent(config, 0, 'cpu')
    for seed in common.SEEDS:
        for method in common.legacy.METHODS:
            directory, _ = common.audit_run(config, method, seed)
            agent.load_weights(directory / 'model_final.pth')
    return dict(passed=True, gradient_updates=0, formal_training_started=False,
                original_sources_unchanged=True, baseline_final_weights_compatible=30,
                independent_scene_count=500, independent_scene_performance_evaluated=False,
                old_validation_overlap=0, paired_initializations=initializations,
                no_action_override=True, supervision_decay_steps=100000,
                critical_observation_count=len(critical),
                critical_unique_observation_count=len({r['observation_sha256'] for r in critical}),
                freeze_manifest_sha256=common.sha256(common.artifact_root(config) / 'freeze_manifest.json'))


def run_training(config, inputs, seeds):
    _, critical, freeze = common.load_frozen(config)  # Validate identity; independent scenes never enter training factories.
    root = common.ROOT / config['experiment']['output_root']
    planned = [common.run_directory(config, common.METHOD, seed) for seed in seeds]
    if any(p.exists() for p in planned):
        raise FileExistsError('Preserve existing runs; no automatic overwrite or resume.')
    for seed in seeds:
        for method in common.legacy.METHODS:
            common.audit_run(config, method, seed)
    comparisons = []
    distances = common.legacy.static_distances(inputs[0])
    for seed, directory in zip(seeds, planned):
        agent = make_agent(config, inputs[0], seed)
        agent.critical_observation_hashes = {r['observation_sha256'] for r in critical}
        agent.critical_positions = {tuple(r['position']) for r in critical}
        training = common.legacy.effective_training(config, seed, 'formal')
        initial_hash = common.legacy.runtime.state_digest(agent.training_state_dict())
        paired = common.read_json(common.run_directory(config, 'advice_bound_margin', seed) / 'result.json')
        if initial_hash != paired['initial_state_sha256']:
            raise RuntimeError('Recorded paired initialization mismatch; refusing to train.')
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(config=config, method=common.METHOD, seed=seed, training_mode='formal',
                        effective_training=asdict(training), initial_state_sha256=initial_hash,
                        source_sha256=source_hashes(), test_data_used=False, evaluation_advice=False,
                        action_override=False, independent_500_used_for_training_or_scheduling=False)
        write_json(metadata, directory / 'run_manifest.json')
        factory = CollectionDiagnostics(common.legacy.runtime.make_factory(config, *inputs, method='unguided', seed=seed), agent, critical)
        validations, details, behaviors = [], [], []

        def save_diagnostics():
            write_records_csv(agent.advice_records(), directory / 'advice_budget.csv')
            write_records_csv(agent.repair_records(), directory / 'value_learning.csv')
            write_records_csv(agent.coverage_records(), directory / 'supervision_coverage.csv')
            write_records_csv(factory.records(), directory / 'collection_coverage.csv')

        def record_validation(step):
            summary, rows, traces = common.legacy.old.evaluate(config, inputs, seed, agent)
            validations.append(dict(environment_steps_total=step, **summary))
            details.extend(dict(environment_steps_total=step, **r) for r in rows)
            failures = [t for t in traces if not t['safe_success']]
            behaviors.extend(dict(environment_steps_total=step, **common.legacy.trace_behavior(t, distances)) for t in failures)
            write_json(failures, directory / f'validation_failures_{step:06d}.json')
            write_records_csv(validations, directory / 'validation_curve.csv')
            write_records_csv(details, directory / 'validation_details.csv')
            write_records_csv(behaviors, directory / 'failure_behaviors.csv')
            save_diagnostics()
            agent.save_weights(directory / f'model_step_{step:06d}.pth')
            print(f'[{common.METHOD} seed={seed}] step={step} safe={summary["safe_success_rate"]:.1%} collision={summary["dynamic_collision_rate"]:.1%} timeout={summary["timeout_rate"]:.1%}', flush=True)

        record_validation(0)

        def progress(_episode, records, full_evaluation):
            if full_evaluation:
                record_validation(records[-1]['environment_steps_total'])
                write_records_csv(list(records), directory / 'training.csv')

        result = common.legacy.train_d3qn([inputs[0]], agent, training, demonstrations=(),
                                          reward_config=common.legacy._reward_config(config),
                                          environment_factory=factory, progress_callback=progress)
        if (result.environment_steps != 200000 or result.gradient_updates != 199501
                or agent.training_action_steps != 200000 or sum(r['collected_transitions'] for r in factory.records()) != 200000
                or any(r['applied'] for r in agent.advice_records())
                or any(r['supervised_samples'] for r in agent.coverage_records() if r['bin_start_step'] >= 100001)):
            raise RuntimeError('Budget, no-action-override or supervision-withdrawal audit failed.')
        if validations[-1]['environment_steps_total'] != 200000:
            record_validation(200000)
        save_diagnostics()
        write_records_csv(list(result.episode_records), directory / 'training.csv')
        agent.save_weights(directory / 'model_final.pth')
        _, final_static, trace = common.legacy.static_evaluation(config, inputs[0], 'unguided', seed, directory / 'model_final.pth')
        write_json(trace, directory / 'static_final_trace.json')
        write_json(dict(result=final_static, behavior=common.legacy.trace_behavior(trace, distances)), directory / 'static_final.json')
        write_json(dict(**metadata, environment_steps=200000, gradient_updates=199501,
                        map=inputs[0].manifest(), route_pool_design_sha256=inputs[2]['design_sha256'],
                        formal_training_started=True, integration_only=False, demo_transition_count=0,
                        online_replay_capacity=10000, final_validation=validations[-1], final_static=final_static,
                        final_model_sha256=common.sha256(directory / 'model_final.pth'),
                        advice_budget=agent.advice_records(), value_learning=agent.repair_records(),
                        freeze_manifest_sha256=common.sha256(common.artifact_root(config) / 'freeze_manifest.json')),
                   directory / 'result.json')
        write_json(dict(complete=True, environment_steps=200000), directory / 'completion.json')
        comparisons.append(dict(method=common.METHOD, seed=seed, **validations[-1]))
        write_records_csv(comparisons, root / 'comparison.csv')
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(common.SEEDS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(common.SEEDS):
        parser.error('Use unique paired seeds 0-4.')
    config = common.legacy.load_config(common.CONFIG)
    inputs = common.validate_config(config)
    if args.train:
        print(run_training(config, inputs, args.seeds))
    else:
        torch.set_num_threads(1)
        report = check(config, inputs, args.seeds)
        directory = common.legacy.runtime.unique_check_dir(config, 'check')
        write_json(report, directory / 'check_report.json')
        print(directory)


if __name__ == '__main__':
    main()

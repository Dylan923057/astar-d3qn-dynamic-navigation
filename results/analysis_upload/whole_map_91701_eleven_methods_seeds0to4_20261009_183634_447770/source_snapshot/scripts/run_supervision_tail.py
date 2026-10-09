"""Train five fresh paired tail models only with --train; default is a zero-update check."""
from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch

import supervision_tail_common as common
from run_teaching_efficiency import CollectionDiagnostics
from astar_d3qn.agents.astar_supervision_tail import supervision_weight
from astar_d3qn.utils.io import write_json, write_records_csv


def schedule_records(config):
    settings = config['supervision_tail']['schedule']
    steps = sorted(set(range(0, 200001, 1000)) | {79999, 80001, 99999, 100001, 139999, 140001})
    return [dict(environment_steps=t, baseline_supervision_weight=max(0., 1-t/100000),
                 new_supervision_weight=supervision_weight(t, **settings),
                 advice_probability_at_action_t=(.8 if t == 0 else .8*max(0., 1-t/100000)),
                 note='t=0 is before any action; actual advice uses the 1-based next action step.') for t in steps]


def check(config, inputs, seeds):
    scenes, _, manifest = common.load_frozen(config)
    paired = []
    for seed in seeds:
        _, baseline = common.audit_run(config, common.BASELINE, seed)
        agent = common.make_agent(config, inputs[0], seed, 'cpu')
        digest = common.legacy.runtime.state_digest(agent.training_state_dict())
        if digest != baseline['initial_state_sha256']:
            raise RuntimeError('Fresh initialization differs from the full recorded paired baseline state.')
        original = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG), inputs[0], seed,
                                            common.BASELINE, 'cpu')
        for step in range(80001):
            agent.training_action_steps = original.training_action_steps = step
            if agent.imitation_weight() != original.imitation_weight():
                raise RuntimeError('Supervision changed before or at 80000 steps.')
        for t in (1, 79999, 80000, 80001, 99999, 100000, 100001, 139999, 140000, 200000):
            if agent.advice_probability(t) != original.advice_probability(t):
                raise RuntimeError('Independent supervision schedule changed A* action override.')
        agent.training_action_steps = 140000
        if agent.imitation_weight() != 0 or agent.advice_probability(100000) != 0:
            raise RuntimeError('Withdrawal boundary is not exactly zero.')
        paired.append(dict(seed=seed, initial_state_sha256=digest, matches_recorded_baseline=True))
    kwargs = dict(max_steps=300, window_size=15, reward_config=common.legacy._reward_config(config),
                  terminate_on_collision=True)
    factory = common.FrozenFactory(inputs[0], scenes)
    for scene in scenes:
        env = factory(inputs[0], **kwargs)
        state = env.reset()
        if (state.spatial.shape != (5, 15, 15) or state.scalars.shape != (4,)
                or np.any(state.spatial[4]) or np.any(state.scalars[2:])
                or not 3 <= len(env.dynamic_obstacles) <= 5 or env.scenario_id != scene['scenario_id']):
            raise RuntimeError('Frozen scene observation/dynamics compatibility failed.')
        if any(tuple(cell) in inputs[0].obstacles or tuple(cell) in (inputs[0].start, inputs[0].goal)
               for spec in scene['obstacles'] for cell in spec['route']):
            raise RuntimeError('Frozen obstacle route is invalid.')
    network = common.legacy.runtime.make_agent(config, 0, 'cpu')
    for seed in common.SEEDS:
        directory, _ = common.audit_run(config, common.BASELINE, seed)
        network.load_weights(directory / 'model_final.pth')
    return dict(passed=True, gradient_updates=0, formal_training_started=False,
                independent_performance_evaluated=False, original_sources_unchanged=True,
                paired_initializations=paired, baseline_final_weights_compatible=5, frozen_scene_count=500,
                overlap_with_old_50=0, overlap_with_previous_500=0,
                equal_to_baseline_through_step=80000, action_override_zero_from_step=100000,
                supervision_zero_from_step=140000,
                supervision_weight_at_100000=supervision_weight(100000),
                boundary_continuity_absolute_error=abs(supervision_weight(80000)-.2),
                freeze_manifest_sha256=common.sha256(common.artifact_root(config) / 'freeze_manifest.json'))


def run_training(config, inputs, seeds):
    planned = [common.run_directory(config, common.METHOD, seed) for seed in seeds]
    if any(p.exists() for p in planned):
        raise FileExistsError('Preserve existing runs; no overwrite, resume, or automatic retraining.')
    _, critical, freeze = common.load_frozen(config)
    for seed in common.SEEDS:
        common.audit_run(config, common.BASELINE, seed)
    comparisons = []
    distances = common.legacy.static_distances(inputs[0])
    for seed, directory in zip(seeds, planned):
        agent = common.make_agent(config, inputs[0], seed)
        initial_hash = common.legacy.runtime.state_digest(agent.training_state_dict())
        _, control = common.audit_run(config, common.BASELINE, seed)
        if initial_hash != control['initial_state_sha256']:
            raise RuntimeError('Refusing unpaired initialization.')
        training = common.legacy.effective_training(config, seed, 'formal')
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(config=config, method=common.METHOD, seed=seed, training_mode='formal',
                        effective_training=asdict(training), initial_state_sha256=initial_hash,
                        source_sha256=common.source_hashes(), test_data_used=False, evaluation_advice=False,
                        independent_500_used_for_training_or_scheduling=False,
                        fresh_random_initialization=True, foundation_loaded=False,
                        freeze_manifest_sha256=common.sha256(common.artifact_root(config) / 'freeze_manifest.json'))
        if metadata['source_sha256'] != freeze['registered_training_source_sha256']:
            raise RuntimeError('Source implementation differs from pretraining registration.')
        write_json(metadata, directory / 'run_manifest.json')
        # This is exactly the baseline training sampler; neither held-out scene set enters it.
        factory = CollectionDiagnostics(common.legacy.runtime.make_factory(config, *inputs,
                                        method='unguided', seed=seed), agent, critical)
        validations, details, behaviors = [], [], []

        def save_diagnostics():
            write_records_csv(agent.advice_records(), directory / 'advice_budget.csv')
            write_records_csv(agent.repair_records(), directory / 'value_learning.csv')
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
            print(f'[{common.METHOD} seed={seed}] step={step} autonomous_safe={summary["safe_success_rate"]:.1%} '
                  f'collision={summary["dynamic_collision_rate"]:.1%} timeout={summary["timeout_rate"]:.1%}', flush=True)

        record_validation(0)

        def progress(_episode, records, full_evaluation):
            if full_evaluation:
                record_validation(records[-1]['environment_steps_total'])
                write_records_csv(list(records), directory / 'training.csv')

        result = common.legacy.train_d3qn([inputs[0]], agent, training, demonstrations=(),
                   reward_config=common.legacy._reward_config(config), environment_factory=factory, progress_callback=progress)
        if (result.environment_steps != 200000 or result.gradient_updates != 199501
                or agent.training_action_steps != 200000 or agent.imitation_weight() != 0
                or sum(r['collected_transitions'] for r in factory.records()) != 200000
                or any(r['applied'] or r['requested'] for r in agent.advice_records() if r['bin_start_step'] >= 100001)
                or any(r['teacher_label_samples'] or r['teacher_margin_weight_mean'] for r in agent.repair_records()
                       if r['bin_start_step'] >= 140001)):
            raise RuntimeError('Training budget or separate-clock withdrawal audit failed.')
        if validations[-1]['environment_steps_total'] != 200000:
            record_validation(200000)
        save_diagnostics()
        write_records_csv(list(result.episode_records), directory / 'training.csv')
        agent.save_weights(directory / 'model_final.pth')
        _, final_static, trace = common.legacy.static_evaluation(config, inputs[0], 'unguided', seed,
                                                               directory / 'model_final.pth')
        write_json(trace, directory / 'static_final_trace.json')
        write_json(dict(result=final_static, behavior=common.legacy.trace_behavior(trace, distances)), directory / 'static_final.json')
        write_json(dict(**metadata, environment_steps=200000, gradient_updates=199501, formal_training_started=True,
                        integration_only=False, demo_transition_count=0, online_replay_capacity=10000,
                        map=inputs[0].manifest(), route_pool_design_sha256=inputs[2]['design_sha256'],
                        final_validation=validations[-1], final_static=final_static,
                        final_model_sha256=common.sha256(directory / 'model_final.pth'),
                        advice_budget=agent.advice_records(), value_learning=agent.repair_records()), directory / 'result.json')
        write_json(dict(complete=True, environment_steps=200000), directory / 'completion.json')
        comparisons.append(dict(method=common.METHOD, seed=seed, **validations[-1]))
        write_records_csv(comparisons, common.ROOT / config['experiment']['output_root'] / 'comparison.csv')
    return common.ROOT / config['experiment']['output_root']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(common.SEEDS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(common.SEEDS):
        parser.error('Use unique paired seeds 0–4.')
    config = common.legacy.load_config(common.CONFIG)
    inputs = common.validate_config(config)
    if args.train:
        print(run_training(config, inputs, args.seeds))
    else:
        torch.set_num_threads(1)
        report = check(config, inputs, args.seeds)
        output = common.legacy.runtime.unique_check_dir(config, 'check')
        write_json(report, output / 'check_report.json')
        write_records_csv(schedule_records(config), output / 'weight_schedule.csv')
        print(output)


if __name__ == '__main__':
    main()

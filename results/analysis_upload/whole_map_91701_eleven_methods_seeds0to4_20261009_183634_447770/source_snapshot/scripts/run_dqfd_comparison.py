"""Three independent comparisons; defaults to zero-update preflight, --train is required."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import platform
from time import perf_counter

import torch

import dqfd_comparison_common as common
from run_teaching_efficiency import CollectionDiagnostics
from astar_d3qn.training.dqfd import new_replay, pretrain, train_online, synchronize
from astar_d3qn.utils.io import write_json, write_records_csv


class UpdateDiagnostics:
    def __init__(self):
        self.bins = {}

    def __call__(self, phase, step, metrics):
        row = self.bins.setdefault((phase, (step-1)//1000), Counter())
        row['updates'] += 1
        for key, value in metrics.items():
            if key.endswith('_max'):
                row[key] = max(row.get(key, -float('inf')), value)
            elif key in ('demo_samples', 'sample_count'):
                row[key] += value
            else:
                row[key+'_sum'] += value
        row['last_step'] = step

    def records(self):
        result = []
        for (phase, index), values in sorted(self.bins.items()):
            row = dict(phase=phase, bin_start_step=1000*index+1, bin_end_step=values['last_step'],
                       updates=values['updates'], demo_samples=values['demo_samples'], sample_count=values['sample_count'])
            row['demo_sample_coverage'] = row['demo_samples']/row['sample_count']
            row.update({(k[:-4]+'_mean' if k.endswith('_sum') else k): v/values['updates'] if k.endswith('_sum') else v
                        for k, v in values.items() if k not in ('last_step', 'updates', 'demo_samples', 'sample_count')})
            result.append(row)
        return result


def check(config, inputs, seeds):
    manifest = common.frozen(config)
    demos, _ = common.load_demonstrations(config)
    paired = []
    for seed in seeds:
        reference = common.baseline_digest(config, inputs[0], seed)
        for method in common.METHODS:
            agent = common.make_agent(config, inputs[0], seed, method, 'cpu')
            if common.paired_digest(agent) != reference:
                raise ValueError('Pretraining-before paired learner mismatch.')
            paired.append(dict(method=method, seed=seed, initial_learner_sha256=reference, matches_baseline=True))
    return dict(passed=True, gradient_updates=0, formal_training_started=False,
                independent_500_performance_evaluated=False, initializations=paired,
                demo_transition_count=len(demos), demo_artifact_sha256=manifest['files_sha256']['demonstrations.npz'],
                original_registered_sources_unchanged=True, existing_outputs_preserved=True)


def run(config, inputs, seeds, methods):
    manifest = common.frozen(config)
    planned = [(s, m, common.run_directory(config, m, s)) for s in seeds for m in methods]
    if any(d.exists() for _, _, d in planned):
        raise FileExistsError('Refusing overwrite/resume/retraining of existing runs; specify only unstarted seeds/methods.')
    # All protocol and paired checks must pass before the first new update.
    check(config, inputs, seeds)
    demos, _ = common.load_demonstrations(config)
    _, critical, _ = common.tail.load_frozen(common.legacy.load_config(common.tail.CONFIG))
    for seed, method, directory in planned:
        baseline = common.baseline_digest(config, inputs[0], seed)
        agent = common.make_agent(config, inputs[0], seed, method)
        if common.paired_digest(agent) != baseline:
            raise ValueError('Unpaired fresh initialization.')
        training = common.legacy.effective_training(config, seed, 'formal')
        directory.mkdir(parents=True, exist_ok=False)
        dqfd = method.startswith('dqfd')
        metadata = dict(config=config, method=method, seed=seed, training_mode='formal',
            effective_training=asdict(training), initial_learner_sha256=baseline,
            source_sha256=common.source_hashes(), test_data_used=False, evaluation_advice=False,
            independent_500_used_for_training_or_scheduling=False, foundation_loaded=False,
            freeze_manifest_sha256=common.sha256(common.artifact_root(config)/'freeze_manifest.json'),
            demonstration_sha256=manifest['files_sha256']['demonstrations.npz'] if dqfd else None,
            hardware=dict(platform=platform.platform(), torch=torch.__version__, device=str(agent.device),
                cuda_name=torch.cuda.get_device_name(agent.device) if agent.device.type == 'cuda' else None,
                torch_threads=torch.get_num_threads()))
        write_json(metadata, directory/'run_manifest.json')
        diagnostics = UpdateDiagnostics()
        start = perf_counter()
        pretraining_seconds, pretraining_updates = 0., 0
        replay = None
        if dqfd:
            replay = new_replay(demos, config['comparison'], seed, training.replay_capacity)
            pretraining_updates = config['comparison']['pretraining_updates']
            def offline_progress(phase, step, metrics):
                diagnostics(phase, step, metrics)
                if step % 1000 == 0:
                    write_records_csv(diagnostics.records(), directory/'dqfd_learning.csv')
                    print(f'[{method} seed={seed}] offline update={step}/{pretraining_updates}', flush=True)
            pretraining_seconds = pretrain(agent, replay, pretraining_updates, training.batch_size, offline_progress)
            agent.save_weights(directory/'model_pretrained.pth')
        factory = common.legacy.runtime.make_factory(config, *inputs, method='unguided', seed=seed)
        if not dqfd:
            factory = CollectionDiagnostics(factory, agent, critical)
        validations, details = [], []

        def save_diagnostics():
            if dqfd:
                write_records_csv(diagnostics.records(), directory/'dqfd_learning.csv')
            else:
                write_records_csv(agent.advice_records(), directory/'advice_budget.csv')
                write_records_csv(agent.repair_records(), directory/'value_learning.csv')
                write_records_csv(factory.records(), directory/'collection_coverage.csv')

        def record_validation(step):
            summary, rows, traces = common.legacy.old.evaluate(config, inputs, seed, agent)
            validations.append(dict(environment_steps_total=step, **summary))
            details.extend(dict(environment_steps_total=step, **r) for r in rows)
            write_json([t for t in traces if not t['safe_success']], directory/f'validation_failures_{step:06d}.json')
            write_records_csv(validations, directory/'validation_curve.csv')
            write_records_csv(details, directory/'validation_details.csv')
            save_diagnostics()
            agent.save_weights(directory/f'model_step_{step:06d}.pth')
            print(f'[{method} seed={seed}] step={step} autonomous_safe={summary["safe_success_rate"]:.1%} '
                  f'collision={summary["dynamic_collision_rate"]:.1%} timeout={summary["timeout_rate"]:.1%}', flush=True)

        record_validation(0)  # DQfD point zero is AFTER offline pretraining; offline cost is separately disclosed.
        def progress(_episode, records, full_evaluation):
            if full_evaluation:
                record_validation(records[-1]['environment_steps_total'])
                write_records_csv(list(records), directory/'training.csv')
        if dqfd:
            result = train_online(inputs[0], agent, training, replay, common.legacy._reward_config(config),
                                  factory, progress, diagnostics)
        else:
            synchronize(agent)
            result = common.legacy.train_d3qn([inputs[0]], agent, training, demonstrations=(),
                reward_config=common.legacy._reward_config(config), environment_factory=factory, progress_callback=progress)
            synchronize(agent)
        if result.environment_steps != 200000 or result.gradient_updates != 199501:
            raise ValueError('Formal online step/update budget mismatch.')
        if not dqfd and (agent.training_action_steps != 200000 or agent.imitation_weight() != 0
                or any(r['risk_veto'] for r in agent.advice_records())
                or any(r['requested'] or r['applied'] for r in agent.advice_records() if r['bin_start_step'] >= 100001)
                or any(r['teacher_label_samples'] for r in agent.repair_records() if r['bin_start_step'] >= 100001)):
            raise ValueError('No-risk ablation changed static/execution/collision constraints or original exit clock.')
        if validations[-1]['environment_steps_total'] != 200000:
            record_validation(200000)
        save_diagnostics()
        write_records_csv(list(result.episode_records), directory/'training.csv')
        agent.save_weights(directory/'model_final.pth')
        costs = dict(pretraining_seconds=pretraining_seconds, online_training_seconds=result.training_seconds,
            online_validation_and_callback_seconds=result.progress_callback_seconds,
            method_wall_seconds=perf_counter()-start,
            demo_generation_shared_seconds=manifest['demonstration_generation_seconds'],
            demo_generation_shared_astar_lookups=manifest['demo_astar_lookups'],
            demo_generation_shared_astar_searches=manifest['demo_astar_searches'],
            online_astar_lookups=0 if dqfd else agent.astar_lookups,
            online_astar_searches=0 if dqfd else agent.astar_searches,
            online_astar_seconds=0. if dqfd else agent.astar_seconds)
        write_json(dict(**metadata, **costs, environment_steps=200000,
            pretraining_updates=pretraining_updates, online_updates=result.gradient_updates,
            gradient_updates=pretraining_updates+result.gradient_updates,
            demo_transition_count=len(demos) if dqfd else 0,
            permanent_demo_retained_count=replay.demo_count if dqfd else 0,
            online_replay_capacity=training.replay_capacity, final_validation=validations[-1],
            label_check_count=None if dqfd else agent.label_checks,
            eligible_label_count=None if dqfd else agent.eligible_labels,
            final_model_sha256=common.sha256(directory/'model_final.pth')), directory/'result.json')
        write_json(dict(complete=True, environment_steps=200000), directory/'completion.json')
        print(f'Complete: {directory}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(common.SEEDS))
    parser.add_argument('--methods', choices=common.METHODS, nargs='+', default=list(common.METHODS))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(common.SEEDS) or len(set(args.methods)) != len(args.methods):
        parser.error('Use unique registered seeds and methods.')
    config = common.legacy.load_config(common.CONFIG)
    inputs = common.validate_config(config)
    if args.train:
        run(config, inputs, args.seeds, args.methods)
    else:
        torch.set_num_threads(1)
        report = check(config, inputs, args.seeds)
        output = common.legacy.runtime.unique_check_dir(config, 'check')
        write_json(report, output/'check_report.json')
        print(output)


if __name__ == '__main__':
    main()

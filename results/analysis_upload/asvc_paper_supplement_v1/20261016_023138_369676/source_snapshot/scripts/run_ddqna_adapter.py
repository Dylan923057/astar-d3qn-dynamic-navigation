"""Check by default; --train explicitly starts the five new 200k-step runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import platform
from time import perf_counter

import torch

import ddqna_adapter_common as common
from astar_d3qn.training.dqfd import synchronize
from astar_d3qn.utils.io import write_json, write_records_csv


def check(config, inputs, seeds):
    pairs = []
    for seed in seeds:
        reference = common.historical.baseline_digest(config, inputs[0], seed)
        agent = common.make_agent(config, inputs[0], seed, 'cpu')
        if common.historical.paired_digest(agent) != reference:
            raise ValueError('Initial network/optimizer/learner RNG mismatch.')
        if any(agent.advice_probability(t) != .5 for t in (1,80000,100000,100001,140000,200000)):
            raise ValueError('Teacher must remain .5 throughout training.')
        pairs.append(dict(seed=seed, initial_learner_sha256=reference, matches_original=True))
    return dict(passed=True, formal_training_started=False, gradient_updates=0,
        independent_500_performance_evaluated=False, paired_initializations=pairs,
        shared_settings_unchanged=True, registered_old_sources_unchanged=True,
        original_results_preserved=True)


def run(config, inputs, seeds):
    planned = [(s,common.run_directory(config,s)) for s in seeds]
    if any(d.exists() for _,d in planned):
        raise FileExistsError('Existing runs are preserved; select only unstarted seeds.')
    check(config, inputs, seeds)
    frozen = common.register(config)
    for seed, directory in planned:
        agent = common.make_agent(config, inputs[0], seed)
        initial = common.historical.paired_digest(agent)
        if initial != frozen['paired_initial_learner_sha256'][str(seed)]:
            raise ValueError('Unpaired fresh learner.')
        training = common.legacy.effective_training(config, seed, 'formal')
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(config=config, method=common.METHOD, seed=seed, training_mode='formal',
            effective_training=asdict(training), initial_learner_sha256=initial,
            source_sha256=frozen['source_sha256'], test_data_used=False, evaluation_advice=False,
            independent_500_used_for_training_or_scheduling=False, foundation_loaded=False,
            freeze_manifest_sha256=common.sha256(common.ROOT/config['ddqna_adapter']['artifact_root']/'freeze_manifest.json'),
            hardware=dict(platform=platform.platform(), torch=torch.__version__, device=str(agent.device),
                cuda_name=torch.cuda.get_device_name(agent.device) if agent.device.type=='cuda' else None,
                torch_threads=torch.get_num_threads()))
        write_json(metadata, directory/'run_manifest.json')
        factory = common.legacy.runtime.make_factory(config,*inputs,method='unguided',seed=seed)
        curves, details = [], []
        start = perf_counter()

        def record_validation(step):
            # The fixed-p teacher must be disabled explicitly as well as by epsilon=0.
            enabled = agent.teacher_enabled
            agent.teacher_enabled = False
            try:
                summary, rows, traces = common.legacy.old.evaluate(config,inputs,seed,agent)
            finally:
                agent.teacher_enabled = enabled
            curves.append(dict(environment_steps_total=step,**summary))
            details.extend(dict(environment_steps_total=step,**r) for r in rows)
            write_json([t for t in traces if not t['safe_success']],directory/f'validation_failures_{step:06d}.json')
            write_records_csv(curves,directory/'validation_curve.csv')
            write_records_csv(details,directory/'validation_details.csv')
            write_records_csv(agent.advice_records(),directory/'action_branches.csv')
            agent.save_weights(directory/f'model_step_{step:06d}.pth')
            print(f'[{common.METHOD} seed={seed}] step={step} autonomous_safe={summary["safe_success_rate"]:.1%} '
                  f'collision={summary["dynamic_collision_rate"]:.1%} timeout={summary["timeout_rate"]:.1%}',flush=True)

        record_validation(0)
        def progress(_episode, records, full_evaluation):
            if full_evaluation:
                record_validation(records[-1]['environment_steps_total'])
                write_records_csv(list(records),directory/'training.csv')
        synchronize(agent)
        result = common.legacy.train_d3qn([inputs[0]],agent,training,demonstrations=(),
            reward_config=common.legacy._reward_config(config),environment_factory=factory,progress_callback=progress)
        synchronize(agent)
        if (result.environment_steps != 200000 or result.gradient_updates != 199501 or
                agent.training_action_steps != 200000 or agent.advice_probability(200000) != .5):
            raise ValueError('Step/update/teacher schedule mismatch.')
        if curves[-1]['environment_steps_total'] != 200000:
            record_validation(200000)
        write_records_csv(list(result.episode_records),directory/'training.csv')
        write_records_csv(agent.advice_records(),directory/'action_branches.csv')
        agent.save_weights(directory/'model_final.pth')
        write_json(dict(**metadata, environment_steps=200000, gradient_updates=result.gradient_updates,
            online_updates=result.gradient_updates, pretraining_updates=0, demo_transition_count=0,
            online_replay_capacity=training.replay_capacity, final_validation=curves[-1],
            online_training_seconds=result.training_seconds,
            online_validation_and_callback_seconds=result.progress_callback_seconds,
            method_wall_seconds=perf_counter()-start, online_astar_lookups=agent.astar_lookups,
            online_astar_searches=agent.astar_searches, online_astar_seconds=agent.astar_seconds,
            final_model_sha256=common.sha256(directory/'model_final.pth')),directory/'result.json')
        write_json(dict(complete=True,environment_steps=200000),directory/'completion.json')
        print(f'Complete: {directory}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(common.SEEDS))
    args=parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(common.SEEDS):
        parser.error('Use unique paired seeds 0..4.')
    config=common.legacy.load_config(common.CONFIG)
    inputs=common.validate_config(config)
    if args.train:
        run(config,inputs,args.seeds)
    else:
        torch.set_num_threads(1)
        report=check(config,inputs,args.seeds)
        common.register(config)
        output=common.legacy.runtime.unique_check_dir(config,'check')
        write_json(report,output/'check_report.json')
        print(output)


if __name__=='__main__':
    main()

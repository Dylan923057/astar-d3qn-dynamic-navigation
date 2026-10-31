"""Reuse the frozen CE runner, adding seeds 2/3/4 only; no historical files are edited."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import os
import shutil
import traceback
import unittest
from unittest.mock import patch

import torch
import fixed_ce_five_common as c
import run_fixed_ce as original_runner
import analyze_fixed_ce as original_analysis


@contextmanager
def installed():
    from analyze_fixed_ce_five import analyze
    # Process-local dependency binding only; frozen train/evaluation/recovery bytecode stays unchanged.
    with patch.object(original_runner, 'c', c), patch.object(original_analysis, 'analyze', analyze):
        yield


def preflight():
    if (c.DATA / 'freeze.json').exists():
        c.verify_freeze()
        print('Existing five-seed CE extension freeze verified; preserved.', flush=True)
        return
    config, inputs = c.configure()
    pairs, reused, protected = c.audit_baseline(config)
    attempt = c.RESULTS / 'preflight_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        suite = unittest.TestSuite()
        for pattern in ('test_fixed_ce.py', 'test_fixed_ce_five.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(c.ROOT / 'tests'), pattern=pattern))
        with (attempt / 'unit_tests.log').open('w', encoding='utf-8') as stream:
            tested = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        if not tested.wasSuccessful():
            raise AssertionError('CE extension tests failed; see retained log')
        for pair in pairs:
            for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
                agent = c.make_agent(config, inputs[0], pair['seed'], device)
                if c.digest(agent.training_state_dict()) != pair['initial_state_sha256']:
                    raise AssertionError('Full paired initial learner state differs')
            pair['initialization_devices_verified'] = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        print('Paired initialization verified for all five seeds; checking original50 evaluation.', flush=True)
        validation, scenes = c.prior.legacy.runtime.check(config, inputs, list(c.NEW_SEEDS))
        if len(scenes) != 50:
            raise AssertionError('Expected frozen50')
        evaluation = []
        with installed():
            for seed in c.NEW_SEEDS:
                agent = c.make_agent(config, inputs[0], seed, 'cpu')
                summary, details, _ = original_runner.evaluate_isolated(config, inputs, seed, agent)
                old = original_analysis.rows(c.baseline_directory(seed) / 'validation_curve.csv')[0]
                for metric in ('safe_success_rate', 'collision_rate', 'timeout_rate'):
                    if abs(summary[metric] - float(old[metric])) > 1e-12:
                        raise AssertionError('Initial original50 validation differs')
                c.write_csv(details, attempt / f'initial50_seed_{seed}.csv')
                evaluation.append(dict(seed=seed, episodes=len(details), summary=summary,
                    teacher_calls=0, risk_calls=0, updates=0, learner_and_rng_unchanged=True))
                print(f'Original50 autonomous preflight seed{seed}: PASS', flush=True)
            recovery = original_runner.recovery_check(config, inputs, attempt / 'recovery')
        for name, value in protected.items():
            if c.sha(c.ROOT / name) != value:
                raise AssertionError('Preflight changed a protected historical artifact')
        report = dict(passed=True, tests=tested.testsRun, pairs=pairs, reused_ce=reused,
            validation=validation, autonomous_evaluation=evaluation, recovery=recovery,
            new_training_seeds=list(c.NEW_SEEDS), pooled_seeds=list(c.ALL_SEEDS),
            training_runs_authorized=3, steps_per_run=200000, updates_per_run=199501,
            previous_ce_retrained=False, margin_retrained=False, independent500_used=False,
            same_ce_runner=True, same_ce_loss=True, no_learning_parameter_changes=True)
        c.DATA.mkdir(parents=True, exist_ok=True)
        keep = ('run_manifest.json', 'result.json', 'completion.json', 'validation_curve.csv',
                'validation_details.csv', 'value_learning.csv', 'supervision_alignment.csv', 'advice_budget.csv')
        for method, seeds in [('margin', c.ALL_SEEDS), ('ce', c.REUSED_SEEDS)]:
            for seed in seeds:
                source = c.baseline_directory(seed) if method == 'margin' else c.original.OUT / f'seed_{seed}'
                target = c.analysis_directory(method, seed)
                target.mkdir(parents=True, exist_ok=False)
                for name in keep:
                    shutil.copyfile(source / name, target / name)
        shutil.copyfile(c.ROOT / config['dataset']['validation_reference'], c.DATA / 'fixed_validation_scenarios.json')
        hashes = c.sources()
        for name in hashes:
            target = c.DATA / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(c.ROOT / name, target)
        c.write_json(report, attempt / 'passed.json')
        c.write_json(dict(config=config, source_sha256=hashes, protected_files_sha256=protected,
            copied_input_sha256={str(p.relative_to(c.DATA)): c.sha(p) for p in c.DATA.rglob('*') if p.is_file()},
            checks=report, created=datetime.now().isoformat()), c.DATA / 'freeze.json')
        c.write_json(report, c.RESULTS / 'preflight.json')
        from analyze_fixed_ce_five import analyze
        analyze()
        print(f'PREFLIGHT PASS: {tested.testsRun} tests; only new seeds 2/3/4 authorized.', flush=True)
    except BaseException as error:
        c.write_json(dict(passed=False, error=repr(error), traceback=traceback.format_exc(), formal_training_runs=0),
                     attempt / 'failure.json')
        raise


def run_one(seed, resume=False):
    if seed not in c.NEW_SEEDS:
        raise ValueError('Only new seeds 2, 3, 4 may train in this extension')
    with installed():
        original_runner.run_one(seed, resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, choices=c.NEW_SEEDS, default=list(c.NEW_SEEDS))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate seeds are not allowed')
    torch.set_num_threads(1)
    if not args.train:
        preflight()
        return
    c.verify_freeze()
    c.OUT.mkdir(parents=True, exist_ok=True)
    lock = c.OUT / 'runner.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    active_seed = None
    try:
        for seed in args.seeds:
            active_seed = seed
            c.write_json(dict(running=True, complete=False, pid=os.getpid(), active_seed=seed,
                planned_new_seeds=args.seeds, reused_ce_seeds=list(c.REUSED_SEEDS), maximum_new_runs=3,
                steps_per_run=200000, updated=datetime.now().isoformat()), c.RESULTS / 'queue_status.json')
            run_one(seed, args.resume)
        c.verify_freeze()
        from analyze_fixed_ce_five import analyze
        analyze()
        complete = all((c.OUT / f'seed_{s}' / 'completion.json').exists() for s in c.NEW_SEEDS)
        c.write_json(dict(running=False, complete=complete, completed_queue_seeds=args.seeds,
            reused_ce_seeds=list(c.REUSED_SEEDS), all_five_complete=complete, updated=datetime.now().isoformat()),
            c.RESULTS / 'queue_status.json')
    except BaseException as error:
        c.write_json(dict(running=False, complete=False, error=repr(error), active_seed=active_seed,
            updated=datetime.now().isoformat()), c.RESULTS / 'queue_status.json')
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()

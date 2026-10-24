"""One command: ten version-matched ablations, cached final500, density tests and figures."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import unittest

import torch

import persistent_paper_common as core


def planned_jobs():
    return [(method, seed) for seed in core.SEEDS for method in core.METHODS]


def worker_command(method, seed, resume):
    command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()),
               '--worker', method, '--seed', str(seed)]
    if resume:
        command.append('--resume')
    return command


def status():
    records = []
    for method, seed in planned_jobs():
        path = core.directory(method, seed) / 'status.json'
        value = core.common.read_json(path) if path.exists() else dict(complete=False, environment_steps=0, running=False)
        records.append(dict(value, method=method, seed=seed))
    core.write_json(dict(planned_runs=10, runs=records), core.RESULTS / 'training_progress.json')
    lines = ['# 新版模块消融进度', '', '|方法|seed|环境步|状态|', '|---|---:|---:|---|']
    for row in records:
        label = '完成' if row.get('complete') else '数值失败，保留记录' if row.get('numerical_failure') else '运行中' if row.get('running') else '待运行/已中断'
        lines.append(f'|{row["method"]}|{row["seed"]}|{row.get("environment_steps", row.get("recoverable_environment_steps", 0))}|{label}|')
    (core.RESULTS / 'PROGRESS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return records


def train(workers=2, resume=False):
    jobs = iter(planned_jobs())
    active = []
    exhausted = False
    logs = core.OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with core.run_lock(core.OUT / 'locks/scheduler.lock'):
        try:
            while active or not exhausted:
                while len(active) < workers and not exhausted:
                    try:
                        method, seed = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    path = core.directory(method, seed)
                    if (path / 'completion.json').exists() or (path / 'nonfinite_failure.json').exists():
                        core.run_info(method, seed)
                        continue
                    if path.exists() and not resume:
                        raise FileExistsError(f'{path}: interrupted run exists; add --resume.')
                    log = (logs / f'{method}_seed_{seed}.log').open('ab' if resume else 'wb')
                    process = subprocess.Popen(worker_command(method, seed, resume), cwd=core.ROOT,
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    active.append((process, log, method, seed))
                    print(f'START {method} seed{seed}; PID={process.pid}; log={log.name}', flush=True)
                for job in active[:]:
                    process, log, method, seed = job
                    code = process.poll()
                    if code is not None:
                        log.close()
                        active.remove(job)
                        if code:
                            raise RuntimeError(f'{method} seed{seed} exited {code}; see {log.name}. Rerun with --resume after fixing the cause.')
                        print(f'FINISHED {method} seed{seed}', flush=True)
                records = status()
                progress = '; '.join(f'{m}/{s}: ' + str(next(r for r in records if r['method'] == m and r['seed'] == s).get('environment_steps', 0)) for _, _, m, s in active)
                if progress:
                    print(progress, flush=True)
                if active:
                    time.sleep(20)
        finally:
            # If the coordinator stops, retain existing recovery snapshots and stop its own children.
            for process, log, _, _ in active:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                log.close()
            status()


def preflight():
    suite = unittest.defaultTestLoader.discover(str(core.ROOT / 'tests'), pattern='test_persistent_paper_completion.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise ValueError('Persistent paper completion tests failed.')
    core.preflight(result.testsRun)
    from evaluate_persistent_paper import audit_reuse
    audit_reuse()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Train both removals, evaluate new models/densities, then report.')
    mode.add_argument('--train-only', action='store_true')
    mode.add_argument('--density-only', action='store_true', help='Evaluate existing full-method models and plot density results only.')
    mode.add_argument('--worker', choices=core.METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, choices=core.SEEDS, help=argparse.SUPPRESS)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worker:
        if args.seed is None:
            parser.error('--worker requires --seed')
        core.run_one(args.worker, args.seed, args.resume)
        return
    if args.density_only:
        from evaluate_persistent_paper import audit_reuse, evaluate
        from analyze_persistent_paper import analyze
        audit_reuse()
        evaluate('density')
        analyze('density')
        return
    preflight()
    if not (args.run or args.train_only):
        print('Checks and registration complete. No formal training/evaluation started. Add --run to execute.', flush=True)
        return
    core.write_json(dict(complete=False, phase='training'), core.RESULTS / 'pipeline_status.json')
    train(args.workers, args.resume)
    if args.train_only:
        return
    from evaluate_persistent_paper import evaluate
    from analyze_persistent_paper import analyze
    core.write_json(dict(complete=False, phase='evaluation'), core.RESULTS / 'pipeline_status.json')
    evaluate('all')
    report = analyze('all')
    failures = [r for r in status() if r.get('numerical_failure')]
    core.write_json(dict(complete=True, phase='complete', report=core.relative(report),
        numerical_failures=failures, all_ten_trainings_succeeded=not failures), core.RESULTS / 'pipeline_status.json')
    print(f'COMPLETE: {report}', flush=True)


if __name__ == '__main__':
    main()

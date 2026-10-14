"""Wait for the existing worker, then evaluate, analyze and package; never train."""
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = 'latest_all_eligible_modules_v1'
METHODS = ('advice_margin_all_eligible', 'supervision_only_bound_all_eligible')
COMMANDS = (
    ('evaluate_latest_module_ablations.py',),
    ('evaluate_latest_module_ablations.py', '--evaluate'),
    ('analyze_latest_module_ablations.py',),
    ('package_latest_module_ablations.py',),
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_sources():
    suite = read(ROOT / 'data' / NAME / 'suite_registration.json')
    for relative, digest in suite['core_registration_sha256'].items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Training registration changed: ' + relative)
        for source, expected in read(path)['source_sha256'].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source changed: ' + source)
    path = ROOT / 'scripts/run_latest_module_suite.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != suite['scheduler_source_sha256']:
        raise ValueError('Training scheduler changed.')
    for command in COMMANDS:
        if not (ROOT / 'scripts' / command[0]).is_file():
            raise FileNotFoundError(command[0])


def check_finished():
    for method in METHODS:
        for seed in range(5):
            directory = ROOT / 'outputs' / NAME / method / f'seed_{seed}'
            if (directory / 'nonfinite_failure.json').is_file():
                continue  # Reporting retains this seed as failed, without retry.
            done = read(directory / 'completion.json')
            if not done['complete'] or done['environment_steps'] != 200000:
                raise ValueError(f'Training unfinished: {method} seed{seed}')


def wait_existing_worker(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # Worker already exited; inspect files.
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        while True:
            state = kernel.WaitForSingleObject(handle, 15000)
            if state == 0:
                return
            if state != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--wait-for-pid', type=int)
    args = parser.parse_args()
    check_sources()
    if args.check:
        print('PASS: frozen training sources and four reporting commands; no training/evaluation started.')
        sys.exit(0)
    if not args.wait_for_pid or args.wait_for_pid <= 0:
        parser.error('Supply the existing training worker PID; this script never starts training.')
    output = ROOT / 'outputs' / NAME / 'automatic_finalization_v1'
    output.mkdir(parents=True, exist_ok=False)
    status = dict(complete=False, phase='waiting', worker_pid=args.wait_for_pid,
                  started=datetime.now().isoformat(), commands=COMMANDS, training_started=False)
    def save():
        (output / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        wait_existing_worker(args.wait_for_pid)
        check_sources()
        check_finished()
        with (output / 'pipeline.log').open('w', encoding='utf-8') as log:
            for command in COMMANDS:
                status['phase'] = command[0]
                save()
                subprocess.run([sys.executable, '-u', '-X', 'utf8', str(ROOT / 'scripts' / command[0]), *command[1:]],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(complete=True, phase='complete', finished=datetime.now().isoformat())
        save()
    except Exception as error:
        status.update(complete=False, phase='failed', error=repr(error), finished=datetime.now().isoformat())
        save()
        raise

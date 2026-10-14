"""Execute the frozen ten-run suite, preserving failures and batch status."""
import argparse
from pathlib import Path

import torch
import run_latest_module_ablations as core


def batch_status():
    records=[]
    for method in core.METHODS:
        for seed in core.SEEDS:
            path=core.ROOT/'outputs'/core.NAME/method/f'seed_{seed}'/'status.json'
            status=core.common.read_json(path) if path.exists() else dict(complete=False,running=False,environment_steps=0)
            records.append(dict(status,method=method,seed=seed))
    core.base.write_json(dict(planned_runs=10,runs=records),core.ROOT/'outputs'/core.NAME/'batch_status.json')


def freeze():
    references={}
    for method in core.METHODS:
        config,_=core.configure(method)
        path=core.ROOT/config['execution_consistency']['registration_root']/'freeze_manifest.json'
        registration=core.common.read_json(path)
        if registration['config']!=config or registration['source_sha256']!=core.source_hashes():
            raise ValueError('Core protocol/code changed after preflight.')
        references[str(path.relative_to(core.ROOT))]=core.common.sha256(path)
    value=dict(runs=[dict(method=m,seed=s) for s in core.SEEDS for m in core.METHODS],
        core_registration_sha256=references,scheduler_source_sha256=core.common.sha256(Path(__file__)),
        numerical_failure_policy='Retain exact step and seed; continue other pre-registered jobs; never retune or restart numerical failures.')
    path=core.ROOT/'data'/core.NAME/'suite_registration.json'
    if path.exists() and core.common.read_json(path)!=value:raise ValueError('Suite registration changed.')
    if not path.exists():core.base.write_json(value,path)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);freeze()
    if not args.train:print(dict(passed=True,planned_runs=10,training_started=False))
    else:
        for seed in core.SEEDS:
            for method in core.METHODS:
                core.run_one(method,seed,args.resume);batch_status()

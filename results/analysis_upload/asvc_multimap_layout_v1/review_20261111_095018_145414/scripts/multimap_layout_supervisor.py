"""Reporting-only NumPy 1.26 compatibility; preserve the immutable training core.

Can adopt already running workers without restarting them. New workers execute
the original frozen runner in fresh processes; this shim never enters learning.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from time import sleep
import numpy as np

# NumPy 2 renamed trapz to trapezoid without changing the integral definition.
# The installed study environment is NumPy 1.26.4. Only this coordinator/report
# process receives the alias; independent training workers are unchanged.
if not hasattr(np,'trapezoid'):np.trapezoid=np.trapz
import multimap_layout_common as c
import run_multimap_layout as runner
from analyze_multimap_layout import analyze,auc

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--analyze',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--adopt-pids',nargs='*',type=int,default=[]);args=parser.parse_args()
    c.verify_freeze()
    check=auc([dict(environment_steps_total=0,safe_success_rate=0),dict(environment_steps_total=200000,safe_success_rate=1)],0,200000)
    assert check==.5 and np.trapezoid([0.,1.],[0.,2.])==1.
    if args.analyze or not args.run:analyze();return
    with runner.run_lock(c.OUT/'supervisor.lock'):
        c.write(dict(pid=os.getpid(),started=datetime.now().isoformat(),adopted_worker_pids=args.adopt_pids,
            shim_sha256=c.sha(__file__),numpy=np.__version__,reporting_alias='trapezoid=trapz',
            frozen_training_sources_unchanged=True,freeze_sha256=c.sha(c.DATA/'freeze.json')),c.OUT/'supervisor.json')
        if args.adopt_pids:
            # Hold the ordinary scheduler lock while existing workers finish:
            # no second queue can submit their unfinished identities.
            with runner.run_lock(c.OUT/'scheduler.lock'):
                while any(runner.pid_alive(pid) for pid in args.adopt_pids):
                    runner.progress();sleep(15)
            analyze()
        runner.queue(resume=True,workers=args.workers)

if __name__=='__main__':main()

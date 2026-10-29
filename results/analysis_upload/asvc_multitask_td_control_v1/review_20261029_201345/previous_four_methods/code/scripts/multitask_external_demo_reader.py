"""Read each compressed demonstration array once; generator artifacts stay frozen."""
from __future__ import annotations

import numpy as np
import multitask_external_common as c
from multitask_pilot_domain import TaskTransition
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.dqfd import NStepCollector


def load_records(settings):
    report=c.read(c.REG/'demonstration_report.json')
    if c.sha(c.REG/'demonstrations.npz')!=report['demonstrations_sha256']:raise ValueError('Demonstration SHA mismatch')
    # NpzFile.__getitem__ decompresses on every access. Cache each array once so
    # per-transition views share backing storage rather than retaining N copies.
    with np.load(c.REG/'demonstrations.npz',allow_pickle=False) as archive:
        arrays={name:archive[name] for name in archive.files}
    collector=NStepCollector(settings['n_steps'],.99);records=[]
    for i in range(len(arrays['action'])):
        item=TaskTransition(Observation(arrays['spatial'][i],arrays['scalars'][i]),int(arrays['action'][i]),float(arrays['reward'][i]),
            Observation(arrays['next_spatial'][i],arrays['next_scalars'][i]),bool(arrays['terminated'][i]),arrays['next_mask'][i],
            str(arrays['task_id'][i]),tuple(map(int,arrays['position'][i])),tuple(map(int,arrays['next_position'][i])),
            bool(arrays['truncated'][i]),False,0.,{})
        records.extend(collector.add(item,episode_end=item.terminated or item.truncated))
    if collector.pending or len(records)!=report['transitions']:raise ValueError('Incomplete demonstration episode/n-step boundary')
    return records

"""Shared read-only fixed-task inputs for the 2026-10-30 supplements."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from unittest.mock import patch

import fixed_ce_five_common as ce
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.evaluation.rollout import evaluate_agent
from replication_recovery import factory_state, restore_factory

ROOT = ce.ROOT
legacy = ce.prior.legacy
read, sha, digest = ce.read, ce.sha, ce.digest


def write(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def rows(path):
    path = Path(path)
    if not path.exists(): return []
    with path.open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def csv_write(values, path):
    values = list(values); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not values: return
    keys = list(dict.fromkeys(k for row in values for k in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(values)
    os.replace(temporary, path)


def flag(value):
    return value is True or str(value).lower() in ('1', '1.0', 'true')


def configuration():
    return ce.configure()


def final_scenes():
    import supervision_tail_common as tail
    scenes, _, frozen = tail.load_frozen(legacy.load_config(tail.CONFIG))
    return scenes, frozen['files_sha256']['independent_final_scenarios.json']


def frozen50():
    config, _ = configuration()
    return read(ROOT / config['dataset']['validation_reference'])['scenarios']


def scene_order(rows_, scenes):
    assert len(rows_) == len(scenes)
    assert [str(r['scenario_id']) for r in rows_] == [str(s['scenario_id']) for s in scenes], 'Scene order differs'
    for row, scene in zip(rows_, scenes):
        assert int(row['dynamic_obstacle_count']) == len(scene['obstacles'])
        assert str(row['dynamic_route_ids']) == ';'.join(map(str, scene['route_ids']))


def summarize(records):
    n = len(records); assert n
    efficiencies = [float(r['path_efficiency']) for r in records if r.get('path_efficiency') not in ('', None)]
    success = sum(flag(r['safe_success']) for r in records)
    return dict(episodes=n, safe_success_rate=success/n,
        dynamic_collision_rate=sum(flag(r['dynamic_collision']) for r in records)/n,
        timeout_rate=sum(r['termination_reason'] == 'timeout' for r in records)/n,
        mean_steps=statistics.mean(float(r['steps']) for r in records),
        mean_wait_steps=statistics.mean(float(r['wait_steps']) for r in records),
        mean_wait_fraction=statistics.mean(float(r['wait_steps'])/max(1, float(r['steps'])) for r in records),
        pooled_wait_fraction=sum(float(r['wait_steps']) for r in records)/sum(float(r['steps']) for r in records),
        mean_path_efficiency_success_only=statistics.mean(efficiencies) if efficiencies else None,
        safe_success_weighted_path_efficiency=sum(float(r['path_efficiency'] or 0)*flag(r['safe_success']) for r in records)/n)


def evaluate(agent, scenes, config, inputs, *, trace=True):
    from teaching_efficiency_common import FrozenFactory
    factory = FrozenFactory(inputs[0], scenes, trace=trace)
    before = digest(agent.training_state_dict())
    with patch.object(agent, 'train_batch', side_effect=AssertionError('Evaluation update forbidden')), \
         legacy.runtime.preserved_evaluation(agent):
        summary, details, _ = evaluate_agent(agent, [inputs[0]]*len(scenes), max_steps=300,
            window_size=15, reward_config=legacy._reward_config(config), terminate_on_collision=True,
            environment_factory=factory, mask_static_invalid_actions=True)
    assert digest(agent.training_state_dict()) == before
    scene_order(details, scenes)
    failures = [dict(scenario_id=r['scenario_id'], safe_success=False,
        termination_reason=r['termination_reason'], steps=e.trace)
        for r, e in zip(details, factory.environments) if not flag(r['safe_success'])] if trace else []
    return dict(summary, timeout_rate=sum(r['termination_reason']=='timeout' for r in details)/len(details)), details, failures


def file_map(paths):
    result = {}
    for path in paths:
        path = Path(path)
        for p in path.rglob('*') if path.is_dir() else [path]:
            if p.is_file() and '__pycache__' not in p.parts: result[p.relative_to(ROOT).as_posix()] = sha(p)
    return result


def check_files(mapping):
    for name, expected in mapping.items():
        assert sha(ROOT/name) == expected, 'Protected file changed: ' + name


def summary_table(records, keys, group='method'):
    result = []
    for name in dict.fromkeys(r[group] for r in records):
        selected = [r for r in records if r[group] == name]
        for key in keys:
            values = [float(r[key]) for r in selected if r.get(key) is not None]
            result.append({group:name, 'metric':key, 'training_seeds':len(values),
                'mean':statistics.mean(values) if values else None,
                'sample_sd':statistics.stdev(values) if len(values)>1 else None})
    return result

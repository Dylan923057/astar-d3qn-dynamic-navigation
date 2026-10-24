"""Version-matched module removals; reuse the frozen persistent training loop."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import os
import shutil
from unittest.mock import patch

import numpy as np
import torch

import run_persistent_teaching as parent
import run_persistent_supervision_five_seeds as full
import supervision_tail_common as tail
from astar_d3qn.agents.all_eligible_module_ablation import AllEligibleModuleAblationAgent
from astar_d3qn.agents.d3qn import D3QNAgent, D3QNConfig
from astar_d3qn.utils.seed import seed_everything

ROOT, common, legacy = parent.ROOT, parent.common, parent.legacy
NAME = 'asvc_persistent_paper_completion_v1'
CONFIG = ROOT / 'configs' / (NAME + '.yaml')
REG, OUT, RESULTS = (ROOT / p / NAME for p in ('data', 'outputs', 'results'))
SEEDS = tuple(range(5))
METHODS = ('no_td', 'no_advice')
ABLATIONS = ('persistent_asvc', *METHODS, 'no_supervision')
DENSITY_METHODS = ('persistent_asvc', 'original_asvc', 'dqfd', 'd3qn', 'astar_prefill')
OLD_METHODS = dict(no_td='advice_margin_all_eligible', no_advice='supervision_only_bound_all_eligible')
write_json, write_csv = parent.write_json, parent.write_records_csv
ORIGINAL_RUN = parent.run_one
BASE_MAKE_AGENT = parent.base.make_agent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def read_csv(path):
    return parent.base.load_csv(path)


def protocol():
    config = legacy.load_config(CONFIG)
    original = common.read_json(full.PARENT_FREEZE)['config']
    require({k: v for k, v in config.items() if k not in ('experiment', 'paper_completion')} ==
            {k: v for k, v in original.items() if k != 'experiment'}, 'Shared training settings changed.')
    expected = dict(seeds=list(SEEDS), methods=list(METHODS), supervision_floor=.1, advice_floor=0.,
        registration_root='data/' + NAME, environment_steps=200000, final_checkpoint_only=True,
        density_registration_root='data/asvc_paper_density_v2', densities=[1, 3, 5, 7],
        scenes_per_density=500, evaluation_device='cpu', reuse_no_supervision='advice_bound',
        numerical_failure_policy='retain_seed_and_report_no_retraining_or_tuning')
    require(config['paper_completion'] == expected, 'Only the registered two removals and density evaluation are allowed.')
    require(config['experiment'] == dict(original['experiment'], name=NAME,
        protocol='persistent_supervision_module_removals_and_frozen_density_v1',
        output_root='outputs/' + NAME, check_root='results/' + NAME), 'Output metadata changed.')
    return config, legacy.validate_config(legacy.load_config(legacy.CONFIG))


def configure(method):
    require(method in METHODS, 'Only no_td and no_advice are new training runs.')
    config, inputs = protocol()
    config = deepcopy(config)
    config['persistent_teaching'].update(seeds=list(SEEDS), registration_root=f'data/{NAME}/{method}',
        baseline_root='outputs/latest_all_eligible_modules_v1/' + OLD_METHODS[method],
        variants={method: dict(advice_floor=0., supervision_floor=.1)})
    config['module_ablation'] = dict(remove='td_target_clip' if method == 'no_td' else 'action_override',
        target_clip=method != 'no_td', action_override=method != 'no_advice',
        supervision_floor=.1, supervision_floor_start=90000, advice_exit=100000,
        compensate_label_count=False, execution_match_required=False)
    return config, inputs


class PersistentModuleAgent(AllEligibleModuleAblationAgent):
    """The old single-factor removal, with the new persistent supervision clock."""
    def imitation_weight(self):
        return max(.1, super().imitation_weight())

    def supervision_records(self):
        records = super().supervision_records()
        for row in records:
            row['labels_computed'] = True
        return records


def make_agent(config, problem, seed, method, device=None):
    seed_everything(seed)
    values = dict(config['agent'])
    if device:
        values['device'] = device
    ad, repair = config['training_advice'], config['value_repair']
    return PersistentModuleAgent(D3QNConfig((5, 15, 15), 4, 5, seed=seed, **values), problem,
        repair_method='advice_bound_margin', ablation_method=OLD_METHODS[method],
        lower=repair['target_lower'], upper=repair['target_upper'], margin=repair['margin'],
        margin_weight=repair['margin_weight'], probability=ad['probability'],
        decay_steps=ad['decay_environment_steps'], risk_radius=ad['risk_radius'], rng_offset=ad['rng_seed_offset'])


def source_hashes():
    sources = dict(common.read_json(full.PARENT_FREEZE)['source_sha256'])
    for old in OLD_METHODS.values():
        frozen = common.read_json(ROOT / 'data/latest_all_eligible_modules_v1' / old / 'freeze_manifest.json')
        for name, digest in frozen['source_sha256'].items():
            require(name not in sources or sources[name] == digest, 'Historical source versions disagree.')
            sources[name] = digest
    for name, digest in sources.items():
        require(common.sha256(ROOT / name) == digest, f'Frozen historical source changed: {name}')
    for path in (CONFIG, Path(__file__), ROOT / 'scripts/run_persistent_paper_completion.py',
                 ROOT / 'tests/test_persistent_paper_completion.py'):
        sources[str(path.relative_to(ROOT))] = common.sha256(path)
    return sources


def directory(method, seed):
    if method in METHODS:
        return OUT / method / f'seed_{seed}'
    if method == 'persistent_asvc':
        return full.new_directory(seed)
    if method == 'original_asvc':
        return full.baseline_directory(seed)
    roots = dict(no_supervision='whole_map_91701_value_repair_v1/advice_bound',
        d3qn='whole_map_91701_value_repair_v1/unguided_raw',
        dqfd='whole_map_91701_dqfd_comparison_v1/dqfd',
        astar_prefill='asvc_paper_supplement_v1/astar_prefill_bound')
    return ROOT / 'outputs' / roots[method] / f'seed_{seed}'


def run_info(method, seed):
    path = directory(method, seed)
    if (path / 'nonfinite_failure.json').exists():
        return dict(method=method, seed=seed, failed=True, directory=path,
                    failure=common.read_json(path / 'nonfinite_failure.json'))
    done = common.read_json(path / 'completion.json')
    result = common.read_json(path / 'result.json')
    require(done['complete'] and result['environment_steps'] == 200000, f'Unfinished run: {path}')
    require(result.get('online_updates', result['gradient_updates']) == 199501, f'Unexpected online update budget: {path}')
    config = common.read_json(path / 'run_manifest.json')['config']
    expected, inputs = protocol()
    for key in ('dataset', 'environment', 'reward', 'guidance', 'dynamic_route_pool', 'agent'):
        require(config[key] == expected[key], f'Reuse settings differ: {method}/{seed}/{key}')
    model = path / 'model_final.pth'
    digest = common.sha256(model)
    require(result.get('final_model_sha256', digest) == digest, 'Final model checksum differs.')
    if method in METHODS:
        frozen = common.read_json(REG / method / 'freeze_manifest.json')
        require(frozen['config'] == config and frozen['source_sha256'] == result['source_sha256'] == source_hashes(), 'New training source/config differs.')
        prefix = common.read_json(path / 'prefix_parity.json')
        require(len(prefix) == 9 and all(r['exact_weights_match'] for r in prefix), 'Pre-90k paired ablation prefix differs.')
        for name, field, value in (('advice_budget.csv', 'mean_requested_probability', 0.),
                                   ('value_learning.csv', 'teacher_margin_weight_mean', .1)):
            values = [float(r[field]) for r in read_csv(path / name) if int(r['bin_start_step']) >= 100001]
            require(values and all(abs(v-value) < 1e-10 for v in values), f'Wrong teaching schedule: {method}')
        if method == 'no_advice':
            require(all(int(r['requested']) == int(r['applied']) == 0 for r in read_csv(path / 'advice_budget.csv')), 'Action advice was not fully removed.')
        if method == 'no_td':
            require(all(float(r['target_clip_fraction_mean']) == 0 for r in read_csv(path / 'value_learning.csv')), 'TD clipping was not removed.')
    return dict(method=method, seed=seed, failed=False, config=config, inputs=inputs,
                directory=path, model=model, digest=digest, result=result)


def assert_final_checkpoint(info):
    a = torch.load(info['model'], map_location='cpu', weights_only=True)
    b = torch.load(info['directory'] / 'model_step_200000.pth', map_location='cpu', weights_only=True)
    require(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), 'Use final200000, never a best checkpoint.')


def no_supervision_reuse():
    """Audit configs, initialization and exact learning updates with supervision disabled."""
    config, inputs = protocol()
    rows = []
    for seed in SEEDS:
        info = run_info('no_supervision', seed)
        manifest = common.read_json(info['directory'] / 'run_manifest.json')
        for name, digest in manifest['source_sha256'].items():
            require(common.sha256(ROOT / name) == digest, f'No-supervision source changed: {name}')
        require(asdict(legacy.effective_training(config, seed, 'formal')) == manifest['effective_training'], 'No-supervision training budget/replay differs.')
        original = legacy.make_agent(config, inputs[0], seed, 'advice_bound', 'cpu')
        new = parent.make_agent(config, inputs[0], seed, 'supervision_floor', 'cpu')
        new.margin_enabled = False
        new.supervision_floor = 0.
        require(legacy.runtime.state_digest(new.training_state_dict()) == manifest['initial_state_sha256'], 'No-supervision initial state differs.')
        factory = legacy.runtime.make_factory(config, *inputs, method='unguided', seed=seed)
        env = factory(inputs[0], max_steps=6, window_size=15, reward_config=legacy._reward_config(config), terminate_on_collision=True)
        state = env.reset()
        from astar_d3qn.replay.transition import Transition
        batch = []
        for _ in range(4):
            result = env.step(4)
            batch.append(Transition(state, 4, result.reward, result.observation, result.terminated, np.asarray(env.action_mask(True))))
            state = result.observation
            if result.done:
                break
        for step in (0, 89999, 90000, 99999, 100000, 150000, 200000):
            original.training_action_steps = new.training_action_steps = step
            require(new.imitation_weight() == original.imitation_weight() == 0, 'Supervision is not fully disabled.')
            require(new.advice_probability() == original.advice_probability(), 'No-supervision advice clock differs.')
            original.train_batch(batch)
            new.train_batch(batch)
            require(legacy.runtime.state_digest(D3QNAgent.training_state_dict(new)) ==
                    legacy.runtime.state_digest(D3QNAgent.training_state_dict(original)), 'No-supervision optimizer/network/target update differs.')
        assert_final_checkpoint(info)
        rows.append(dict(seed=seed, reusable=True, model_sha256=info['digest'],
            run_manifest_sha256=common.sha256(info['directory'] / 'run_manifest.json'),
            validation_sha256=common.sha256(info['directory'] / 'validation_curve.csv'),
            exact_updates_checked_at=[0, 89999, 90000, 99999, 100000, 150000, 200000]))
    return rows


@contextmanager
def run_lock(path):
    """An OS file lock is released even if a worker is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def run_one(method, seed, resume=False):
    path = directory(method, seed)
    with run_lock(OUT / 'locks' / f'{method}_seed_{seed}.lock'):
        if (path / 'nonfinite_failure.json').exists():
            print(f'{method} seed{seed}: retained numerical failure; no restart.', flush=True)
            return
        if (path / 'completion.json').exists():
            run_info(method, seed)
            print(f'{method} seed{seed}: already complete, preserved.', flush=True)
            return
        active = {}
        def construct(c, p, s, m, device=None):
            active['agent'] = make_agent(c, p, s, m, device)
            return active['agent']
        with ExitStack() as stack:
            for key, value in dict(CONFIG=CONFIG, SEEDS=SEEDS, VARIANTS=METHODS,
                    configure=lambda: configure(method), source_hashes=source_hashes, make_agent=construct).items():
                stack.enter_context(patch.object(parent, key, value))
            try:
                ORIGINAL_RUN(method, seed, resume=resume and path.exists())
            except FloatingPointError as error:
                agent = active.get('agent')
                status = dict(method=method, seed=seed, complete=False, numerical_failure=True,
                    environment_steps=agent.training_action_steps if agent else None,
                    gradient_updates=agent.update_steps if agent else None,
                    error=str(error), resume_allowed=False)
                write_json(status, path / 'nonfinite_failure.json')
                write_json(status, path / 'status.json')
                print(f'{method} seed{seed}: numerical failure retained at {status["environment_steps"]}.', flush=True)
    if (path / 'completion.json').exists():
        run_info(method, seed)


def preflight(test_count):
    config, inputs = protocol()
    sources = source_hashes()
    reuse = no_supervision_reuse()
    for method in METHODS:
        cfg, _ = configure(method)
        initials = []
        for seed in SEEDS:
            baseline = ROOT / cfg['persistent_teaching']['baseline_root'] / f'seed_{seed}'
            manifest = common.read_json(baseline / 'run_manifest.json')
            require(common.read_json(baseline / 'completion.json')['complete'], 'Historical paired ablation is incomplete.')
            agent = make_agent(cfg, inputs[0], seed, method, 'cpu')
            digest = legacy.runtime.state_digest(agent.training_state_dict())
            reference = common.read_json(full.new_directory(seed) / 'run_manifest.json')
            require(digest == manifest['initial_state_sha256'] == reference['initial_state_sha256'], 'Paired initialization differs.')
            require(legacy.effective_training(cfg, seed, 'formal') == legacy.effective_training(manifest['config'], seed, 'formal'), 'Historical ablation training budget differs.')
            for key in ('dataset', 'agent', 'environment', 'reward', 'guidance', 'training_advice', 'value_repair', 'dynamic_route_pool'):
                require(cfg[key] == manifest['config'][key], f'Historical ablation differs: {key}')
            for step in range(200001):
                agent.training_action_steps = step
                require(agent.imitation_weight() == max(.1, 1-step/100000), 'Wrong persistent supervision clock.')
                expected = .8*max(0, 1-step/100000) if method == 'no_td' else 0.
                require(agent.advice_probability(step) == expected, 'Wrong advice clock.')
            initials.append(dict(seed=seed, initial_state_sha256=digest,
                baseline_curve_sha256=common.sha256(baseline / 'validation_curve.csv')))
        small = deepcopy(cfg)
        small['training_advice']['decay_environment_steps'] = 12
        with patch.object(parent.base, 'make_agent', side_effect=lambda c, p, s, d=None: make_agent(c, p, s, method, d)):
            recovery = parent.base.recovery_check(small, inputs)
        checks = dict(passed=True, tests=test_count, initializations=initials,
            recovery_crosses_exit=recovery, no_supervision_reuse=reuse, fresh_random_initialization=True,
            full_run_environment_steps=200000, gradient_updates=199501, prefix_parity_before_step=90000,
            all_five_seeds_retained=True, evaluation_advice=False, pretraining_updates=0,
            random_alignment='Same initial learner and episode-index scenes; collected trajectories diverge after treatment.')
        frozen = dict(config=cfg, checks=checks, source_sha256=sources,
            validation_sha256=common.sha256(ROOT / cfg['dataset']['validation_reference']))
        target = REG / method / 'freeze_manifest.json'
        if target.exists():
            require(common.read_json(target) == frozen, 'Frozen preflight changed; preserve the existing protocol.')
        else:
            write_json(frozen, target)
            for name in sources:
                dest = target.parent / 'source_snapshot' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
    write_json(dict(passed=True, planned_new_training_runs=10, training_started=False,
        no_supervision_reuse=reuse, methods=list(METHODS), seeds=list(SEEDS)), RESULTS / 'preflight.json')
    print('PREFLIGHT PASSED: 10 new runs registered; old no-supervision control reusable.', flush=True)

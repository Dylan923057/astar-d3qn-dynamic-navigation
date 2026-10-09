"""Numerical/contract checks, not formal experiments or holdout evaluation."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import unittest
from functools import lru_cache
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import dqfd_comparison_common as common
from run_dqfd_comparison import UpdateDiagnostics
from astar_d3qn.agents.astar_value_repair import AStarValueRepairAgent
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.dqfd import NStepCollector, DQfDReplay
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.dqfd import new_replay, pretrain, train_online


def transition(reward=1., terminal=False, marker=0):
    spatial = np.zeros((5, 15, 15), dtype=np.float32)
    scalars = np.zeros(4, dtype=np.float32)
    scalars[0] = marker
    state = Observation(spatial, scalars)
    following = Observation(spatial.copy(), scalars.copy())
    return Transition(state, 4, reward, following, terminal, np.ones(5, dtype=bool))


def records(count=8):
    collector = NStepCollector(10, .99)
    rows = []
    for i in range(count):
        rows.extend(collector.add(transition(.04, i == count-1, marker=i), episode_end=i == count-1))
    return rows


@lru_cache(maxsize=1)
def protocol():
    torch.set_num_threads(1)
    config = common.legacy.load_config(common.CONFIG)
    return config, common.validate_config(config)


def test_terminal_n_step_return_and_no_episode_crossing():
    collector = NStepCollector(10, .5)
    assert collector.add(transition(2)) == []
    out = collector.add(transition(4, True), episode_end=True)
    assert [r.n_return for r in out] == [4, 4]
    assert [r.n_length for r in out] == [2, 1]
    assert all(r.n_terminated for r in out)
    assert not collector.pending
    out = collector.add(transition(100), episode_end=True)
    assert out[0].n_return == 100 and not out[0].n_terminated


def test_timeout_short_tail_bootstraps():
    collector = NStepCollector(3, .5)
    collector.add(transition(2))
    out = collector.add(transition(4), episode_end=True)
    assert out[0].n_length == 2 and out[0].n_return == 4
    assert not out[0].n_terminated


def test_full_nstep_sliding_window():
    collector = NStepCollector(3, .5)
    out = []
    for i in range(5):
        out.extend(collector.add(transition(i+1), episode_end=i == 4))
    assert len(out) == 5
    assert [r.n_length for r in out] == [3, 3, 3, 2, 1]
    assert out[0].n_return == 1+.5*2+.25*3


def test_permanent_demo_partition_and_fifo():
    demos = records(3)
    replay = DQfDReplay(demos, 2, seed=7)
    for r in records(9):
        replay.add(r)
    assert len(replay) == 5
    assert all(a is b for a, b in zip(replay.records[:3], demos))
    assert replay.records[3].transition.state.scalars[0] == 8
    assert replay.records[4].transition.state.scalars[0] == 7


def test_per_probability_bonus_and_global_is_weights():
    demo = records(1)[0]
    replay = DQfDReplay([demo], 1, alpha=1, demo_bonus=1, agent_bonus=.001, seed=9)
    replay.add(demo)
    replay.update_priorities([0, 1], [0, 0])
    batch, indices, weights, mask = replay.sample(30000, .6)
    assert mask.mean() > .995
    assert np.allclose(weights[indices == 0], .001**.6)
    assert np.allclose(weights[indices == 1], 1.)
    assert (weights > 0).all() and (weights <= 1).all()


def test_per_reproducible_sampling_and_duplicate_priorities():
    a = DQfDReplay(records(4), 2, seed=0)
    b = DQfDReplay(records(4), 2, seed=0)
    assert np.array_equal(a.sample(64, .6)[1], b.sample(64, .6)[1])
    a.update_priorities([0, 0], [1., 5.])
    assert np.isclose(a.sums[a.base], (5+1)**.4)
    with unittest.TestCase().assertRaises(ValueError):
        a.update_priorities([0], [float('nan')])


def test_paired_pretraining_initialization(protocol, seed):
    cfg, inputs = protocol
    baseline = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG), inputs[0], seed,
                                        'advice_bound_margin', 'cpu')
    for method in common.METHODS:
        agent = common.make_agent(cfg, inputs[0], seed, method, 'cpu')
        assert common.paired_digest(agent) == common.paired_digest(baseline)


def test_full_targets_clip_after_reward_and_bootstrap_only(protocol, discount):
    cfg, inputs = protocol
    a = common.make_agent(cfg, inputs[0], 0, 'dqfd', 'cpu')
    b = common.make_agent(cfg, inputs[0], 0, 'dqfd_bound', 'cpu')
    policy = torch.tensor([[0., 100., 1., 0., 0.]]*3)
    target = torch.tensor([[0., 30., 0., 0., 0.]]*3)
    returns = torch.tensor([5., -100., 12.])
    terminal = torch.tensor([0., 0., 1.])
    mask = torch.ones((3, 5), dtype=torch.bool)
    raw, _ = a.complete_target(returns, torch.full((3,), discount), terminal, policy, target, mask)
    clipped, original = b.complete_target(returns, torch.full((3,), discount), terminal, policy, target, mask)
    assert torch.equal(raw, original)
    assert torch.equal(clipped, torch.tensor([10., -6., 10.]))
    assert raw[0] == 5+discount*30 and raw[2] == 12
    assert target[0, 1] == 30  # Q outputs were not clamped.


def test_double_selection_and_static_mask(protocol):
    cfg, inputs = protocol
    a = common.make_agent(cfg, inputs[0], 0, 'dqfd', 'cpu')
    q, _ = a.complete_target(torch.tensor([0.]), torch.tensor([1.]), torch.tensor([0.]),
        torch.tensor([[100., 2., 3., 1., 0.]]), torch.tensor([[20., 80., 7., 0., 0.]]),
        torch.tensor([[False, True, True, True, True]]))
    assert q.item() == 7


def test_dqfd_all_losses_pretrain_and_online_contract(protocol):
    cfg, inputs = protocol
    agent = common.make_agent(cfg, inputs[0], 0, 'dqfd_bound', 'cpu')
    replay = new_replay(records(8), cfg['comparison'], 0)
    log = UpdateDiagnostics()
    assert pretrain(agent, replay, 2, 4, log) >= 0
    assert agent.update_steps == 2 and len(replay) == 8
    for row in log.records():
        assert row['one_td_loss_mean'] >= 0 and row['n_td_loss_mean'] >= 0
        assert row['expert_loss_mean'] >= 0 and row['l2_loss_mean'] > 0
        assert row['demo_sample_coverage'] == 1
        assert 'raw_one_target_max' in row
    training = replace(common.legacy.effective_training(cfg, 0, 'formal'), max_environment_steps=8,
        learning_starts=4, batch_size=4, progress_interval_environment_steps=4,
        allow_partial_epsilon_schedule=True)
    factory = common.legacy.runtime.make_factory(cfg, *inputs, method='unguided', seed=0)
    result = train_online(inputs[0], agent, training, replay, common.legacy._reward_config(cfg),
                          factory, lambda *args: None, log)
    assert result.environment_steps == 8 and result.gradient_updates == 5
    assert agent.update_steps == 7 and len(replay) == 16 and replay.demo_count == 8


def test_no_risk_predicate_applies_to_labels_and_override(protocol):
    cfg, inputs = protocol
    agent = common.make_agent(cfg, inputs[0], 0, common.METHODS[2], 'cpu')
    baseline = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG), inputs[0], 0,
                                        'advice_bound_margin', 'cpu')
    factory = common.legacy.runtime.make_factory(cfg, *inputs, method='unguided', seed=0)
    env = factory(inputs[0], max_steps=300, window_size=15,
                  reward_config=common.legacy._reward_config(cfg), terminate_on_collision=True)
    state = env.reset()
    teacher = baseline.static_advice(state)
    from astar_d3qn.core.grid import ACTION_DELTAS
    dr, dc = ACTION_DELTAS[teacher]
    state.spatial[1:4] = 0
    state.spatial[1, 7+dr, 7+dc] = 1
    t = Transition(state, teacher, .04, state, False, env.action_mask(True))
    assert baseline.observed_risk(state, teacher)
    assert not agent.observed_risk(state, teacher)
    assert not baseline.teacher_label(t) and agent.teacher_label(t)
    assert not agent.teacher_label(replace(t, action=(teacher+1)%5))
    assert not agent.teacher_label(replace(t, terminated=True, reward=-1.))
    agent.advice_rng.random = lambda: 0.
    assert agent.select_action(state, 1., list(np.flatnonzero(env.action_mask(True)))) == teacher
    assert agent.advice_records()[0]['risk_veto'] == 0
    for step in (99999, 100000, 140000, 200000):
        assert agent.advice_probability(step) == baseline.advice_probability(step)
        agent.training_action_steps = baseline.training_action_steps = step
        assert agent.imitation_weight() == baseline.imitation_weight()


def test_epsilon_zero_does_not_call_teacher(protocol):
    cfg, inputs = protocol
    agent = common.make_agent(cfg, inputs[0], 0, common.METHODS[2], 'cpu')
    state = transition().state
    before = common.legacy.runtime.state_digest(agent.training_state_dict())
    with common.legacy.runtime.preserved_evaluation(agent):
        agent.select_action(state, 0, [4])
    assert agent.astar_lookups == 0 and agent.training_action_steps == 0
    assert common.legacy.runtime.state_digest(agent.training_state_dict()) == before


def test_online_samples_have_no_expert_supervision(protocol):
    cfg, inputs = protocol
    agent = common.make_agent(cfg, inputs[0], 0, 'dqfd', 'cpu')
    rows = records(4)
    metrics = agent.train_dqfd(rows, np.ones(4), np.zeros(4, dtype=bool))
    assert metrics['expert_loss'] == 0 and metrics['demo_samples'] == 0
    assert metrics['one_td_loss'] >= 0 and metrics['n_td_loss'] >= 0 and metrics['l2_loss'] > 0


def test_final_evaluation_requires_all_models_before_any_output(protocol):
    from unittest.mock import patch
    import summarize_dqfd_comparison as summary
    cfg, _ = protocol
    with patch.object(common, 'frozen', return_value={}), patch.object(common.tail, 'load_frozen',
            return_value=([], [], {})), patch.object(common, 'audit_run', side_effect=FileNotFoundError('pending')), \
            patch.object(summary, 'evaluate_agent') as rollout, \
            patch.object(common.legacy.runtime, 'unique_check_dir') as create_output:
        with unittest.TestCase().assertRaises(FileNotFoundError):
            summary.evaluate(cfg, 'cpu')
        rollout.assert_not_called()
        create_output.assert_not_called()


def load_tests(loader, tests, pattern):
    import inspect
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if not name.startswith('test_') or not callable(function):
            continue
        parameters = inspect.signature(function).parameters
        arguments = [()]
        if 'protocol' in parameters:
            arguments = [(protocol(),)]
        if 'seed' in parameters:
            arguments = [(protocol(), seed) for seed in range(5)]
        if 'discount' in parameters:
            arguments = [(protocol(), discount) for discount in (.99, .99**10)]
        for args in arguments:
            suite.addTest(unittest.FunctionTestCase(lambda fn=function, values=args: fn(*values),
                                                    description=name))
    return suite


if __name__ == '__main__':
    unittest.main()

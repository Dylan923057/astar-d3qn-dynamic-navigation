"""Minimal DQfD loop using existing epsilon, episode-end validation, factory and result contracts."""
from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from astar_d3qn.replay.dqfd import DQfDReplay, NStepCollector
from astar_d3qn.replay.transition import Transition
from astar_d3qn.training.budget import crossed_interval
from astar_d3qn.training.trainer import TrainingResult, linear_epsilon_at_environment_step


def new_replay(demos, settings, seed, capacity=10000):
    return DQfDReplay(demos, capacity, alpha=settings['per_alpha'],
                      agent_bonus=settings['per_agent_bonus'], demo_bonus=settings['per_demo_bonus'], seed=seed)


def update(agent, replay, batch_size, beta):
    records, indices, weights, demo = replay.sample(batch_size, beta)
    metrics = agent.train_dqfd(records, weights, demo)
    replay.update_priorities(indices, metrics.pop('td_errors'))
    metrics['beta'] = beta
    return metrics


def synchronize(agent):
    if agent.device.type == 'cuda':
        torch.cuda.synchronize(agent.device)


def pretrain(agent, replay, updates, batch_size, on_update=None):
    synchronize(agent)
    start = perf_counter()
    for i in range(updates):
        metrics = update(agent, replay, batch_size, agent.settings['per_beta_start'])
        if on_update:
            on_update('pretrain', i+1, metrics)
    synchronize(agent)
    return perf_counter()-start


def train_online(problem, agent, config, replay, reward_config, factory, progress_callback, on_update=None):
    if config.updates_per_step != 1 or config.max_environment_steps is None:
        raise ValueError('Registered online budget requires one update per environment step.')
    factory.reset_schedule()
    collector = NStepCollector(agent.settings['n_steps'], agent.config.gamma)
    total = updates = episode = 0
    records, callback_seconds = [], 0.
    synchronize(agent)
    start = perf_counter()
    while total < config.max_environment_steps:
        before = total
        episode += 1
        env = factory(problem, max_steps=config.max_steps, window_size=config.window_size,
                      reward_config=reward_config, terminate_on_collision=config.terminate_on_collision)
        state = env.reset()
        episode_return, losses = 0., []
        while True:
            epsilon = linear_epsilon_at_environment_step(total, config)
            valid = list(np.flatnonzero(env.action_mask(True)))
            action = agent.select_action(state, epsilon, valid)
            result = env.step(action)
            total += 1
            end = result.done or total == config.max_environment_steps
            transition = Transition(state, action, result.reward, result.observation, result.terminated,
                                    env.action_mask(True))
            for item in collector.add(transition, episode_end=end):
                replay.add(item)
            if total >= config.learning_starts:
                beta = agent.settings['per_beta_start']+(agent.settings['per_beta_end']-
                    agent.settings['per_beta_start'])*min(1., total/config.max_environment_steps)
                metrics = update(agent, replay, config.batch_size, beta)
                losses.append(metrics['loss'])
                updates += 1
                if on_update:
                    on_update('online', total, metrics)
            state = result.observation
            episode_return += result.reward
            if end:
                break
        records.append(dict(episode=episode, environment_steps_total=total, gradient_updates_total=updates,
            epsilon=epsilon, steps=env.steps, episode_return=episode_return, safe_success=bool(result.info['reached']),
            termination_reason=result.info.get('termination_reason'), loss=float(np.mean(losses)) if losses else 0.,
            replay_size=len(replay), demo_retained_count=replay.demo_count))
        if crossed_interval(before, total, config.progress_interval_environment_steps):
            synchronize(agent)
            clock = perf_counter()
            progress_callback(episode, tuple(records), True)
            synchronize(agent)
            callback_seconds += perf_counter()-clock
    synchronize(agent)
    if collector.pending or replay.demo_count+min(total, replay.online_capacity) != len(replay):
        raise ValueError('Pending episode data or permanent partition lost at online completion.')
    return TrainingResult(tuple(records), total, updates, perf_counter()-start-callback_seconds, callback_seconds, False)

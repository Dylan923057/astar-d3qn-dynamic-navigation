"""Read-only same-observation action ranking audit around teaching withdrawal."""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import torch

import teaching_efficiency_common as common
from analyze_runtime_path_pilot import load_csv, static_distances, trace_behavior
from analyze_supervision_tail import failure_counts
from analyze_teaching_efficiency import mean_records
from diagnose_comparison_evidence import q_batch
from astar_d3qn.agents.d3qn import D3QNAgent
from astar_d3qn.utils.io import write_json, write_records_csv

ANCHORS = (80000, 90000, 100000, 110000, 120000)


def audit(seeds):
    config = common.legacy.load_config(common.CONFIG)
    inputs = common.validate_config(config)
    problem = inputs[0]
    scenes = common.read_json(common.ROOT / config['dataset']['validation_reference'])['scenarios']
    scene_lookup = {s['scenario_id']: s for s in scenes}
    output = common.ROOT / 'results/teaching_withdrawal_diagnosis' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    summaries, phases, rankings, pairs, model_hashes = [], [], [], [], {}
    replay_steps = api_checks = origin_checks = 0
    distances = static_distances(problem)
    for seed in seeds:
        directory, run = common.audit_run(config, 'advice_bound_margin', seed)
        curve = load_csv(directory / 'validation_curve.csv')
        selected = [min(curve, key=lambda r: abs(int(r['environment_steps_total']) - t)) for t in ANCHORS]
        states, masks, meta = [], [], []
        teacher = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG), problem, seed,
                                            'advice_bound_margin', 'cpu')
        for anchor, point in zip(ANCHORS, selected):
            step = int(point['environment_steps_total'])
            traces = common.read_json(directory / f'validation_failures_{step:06d}.json')
            behaviors = [trace_behavior(t, distances) for t in traces]
            summaries.append(dict(method='advice_bound_margin', seed=seed, anchor=anchor, checkpoint=step,
                success=float(point['safe_success_rate']), collision=float(point['dynamic_collision_rate']),
                timeout=float(point['timeout_rate']), **failure_counts(behaviors)))
            for trace in traces:
                scene = scene_lookup[trace['scenario_id']]
                factory = common.FrozenFactory(problem, [scene], trace=False)
                env = factory(problem, max_steps=300, window_size=15, terminate_on_collision=True,
                              reward_config=common.legacy._reward_config(config))
                state = env.reset()
                last = len(trace['steps']) - 1
                for index, saved in enumerate(trace['steps']):
                    if tuple(saved['before']['position']) != tuple(env.position):
                        raise ValueError('Saved failure position disagrees with frozen-scene replay.')
                    action = int(saved['action'])
                    # Fixed sampling rule: every tenth decision in the last 60 steps, plus the terminal decision.
                    keep = index == last or (index >= max(0, last - 59) and (last - index) % 10 == 0)
                    if keep:
                        label = teacher.static_advice(state)
                        mask = np.asarray(env.action_mask(True), dtype=bool)
                        risk = label is None or teacher.observed_risk(state, label)
                        states.append(state); masks.append(mask)
                        meta.append(dict(seed=seed, origin_anchor=anchor, origin_checkpoint=step,
                            scenario_id=trace['scenario_id'], decision=index + 1, position=list(env.position),
                            termination_reason=trace['termination_reason'], executed_action=action,
                            teacher_action=label, observed_teacher_risk=bool(risk),
                            static_remaining=distances[tuple(env.position)]))
                    outcome = env.step(action)
                    replay_steps += 1
                    if (tuple(saved['position']) != tuple(env.position) or
                            not np.isclose(float(saved['reward']), outcome.reward) or
                            saved['dynamic_positions'] != [list(p) for p in env.dynamic_positions]):
                        raise ValueError('Failure replay reward/dynamic positions mismatch.')
                    state = outcome.observation
                if not outcome.done or outcome.info['termination_reason'] != trace['termination_reason']:
                    raise ValueError('Failure replay termination mismatch.')
        write_records_csv(meta, output / f'seed{seed}_state_index.csv')
        if not states:
            continue
        np.savez_compressed(output / f'seed{seed}_observations.npz',
                            spatial=np.stack([s.spatial for s in states]), scalars=np.stack([s.scalars for s in states]),
                            masks=np.stack(masks))
        mask_array = np.stack(masks)
        labels = np.asarray([m['teacher_action'] if m['teacher_action'] is not None else 0 for m in meta])
        eligible = np.asarray([m['teacher_action'] is not None and not m['observed_teacher_risk'] for m in meta])
        predictions = {}
        for anchor, point in zip(ANCHORS, selected):
            step = int(point['environment_steps_total'])
            path = directory / f'model_step_{step:06d}.pth'
            digest = common.sha256(path); model_hashes[str(path.relative_to(common.ROOT))] = digest
            agent = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG), problem, seed,
                                              'advice_bound_margin', 'cpu')
            agent.load_weights(path)
            q = np.concatenate([q_batch(agent, states[i:i+128]) for i in range(0, len(states), 128)])
            greedy = np.where(mask_array, q, -np.inf).argmax(axis=1)
            valid_q = q[mask_array]
            competitors = mask_array.copy(); competitors[np.arange(len(states)), labels] = False
            advantage = q[np.arange(len(states)), labels] - np.where(competitors, q, -np.inf).max(axis=1)
            agree = greedy == labels
            predictions[anchor] = dict(greedy=greedy, agreement=agree, advantage=advantage)
            for i, state in enumerate(states):
                if meta[i]['origin_anchor'] == anchor:
                    origin_checks += 1
                    if greedy[i] != meta[i]['executed_action']:
                        raise ValueError('Loaded model does not reproduce its saved failure decisions.')
            for i in np.linspace(0, len(states)-1, min(20, len(states)), dtype=int):
                api_checks += 1
                if D3QNAgent.select_action(agent, states[i], 0, list(np.flatnonzero(masks[i]))) != greedy[i]:
                    raise ValueError('Batched action differs from original API.')
            rankings.append(dict(method='advice_bound_margin', seed=seed, checkpoint=step, anchor=anchor,
                shared_failure_state_count=len(states), observed_risk_clear_states=int(eligible.sum()),
                teacher_agreement_clear=float(agree[eligible].mean()) if eligible.any() else None,
                teacher_advantage_clear_mean=float(advantage[eligible].mean()) if eligible.any() else None,
                wait_selection_rate=float((greedy == 4).mean()), q_min=float(valid_q.min()), q_max=float(valid_q.max())))
            write_records_csv([dict(**m, evaluated_anchor=anchor, evaluated_checkpoint=step,
                greedy_action=int(greedy[i]), teacher_advantage=float(advantage[i]),
                q_values=q[i].tolist()) for i, m in enumerate(meta)], output / f'seed{seed}_q_anchor{anchor}.csv')
            if common.sha256(path) != digest:
                raise ValueError('Checkpoint changed during read-only audit.')
        for before, after in zip(ANCHORS, ANCHORS[1:]):
            a, b = predictions[before], predictions[after]
            pairs.append(dict(method='advice_bound_margin', seed=seed, before=before, after=after,
                state_count=len(states), clear_count=int(eligible.sum()),
                action_flip_rate=float((a['greedy'] != b['greedy']).mean()),
                clear_teacher_to_other=float((a['agreement'][eligible] & ~b['agreement'][eligible]).mean()),
                clear_other_to_teacher=float((~a['agreement'][eligible] & b['agreement'][eligible]).mean()),
                clear_advantage_change=float((b['advantage'][eligible] - a['advantage'][eligible]).mean())))
        learning = load_csv(directory / 'value_learning.csv'); advice = load_csv(directory / 'advice_budget.csv')
        for left, right in ((60000,80000),(80000,100000),(100000,120000)):
            rows = [r for r in learning if left < int(r['bin_end_step']) <= right]
            active = left < 100000
            updates = sum(int(r['updates']) for r in rows)
            a = [r for r in advice if left < int(r['bin_end_step']) <= right]
            phases.append(dict(method='advice_bound_margin', seed=seed, left=left, right=right, updates=updates,
                sampled_supervised_fraction=sum(int(r['teacher_label_samples']) for r in rows)/(updates*64) if active else None,
                post_exit_eligible_coverage='not logged' if not active else 'not available per state',
                supervision_weight_mean=sum(float(r['teacher_margin_weight_mean'])*int(r['updates']) for r in rows)/updates,
                margin_loss_mean=sum(float(r['demo_margin_loss_mean'])*int(r['updates']) for r in rows)/updates,
                td_loss_mean=sum(float(r['td_loss_mean'])*int(r['updates']) for r in rows)/updates,
                sampled_q_abs_max=max(float(r['q_abs_max']) for r in rows),
                teacher_execution_rate=sum(int(r['applied']) for r in a)/sum(int(r['steps']) for r in a)))
        print(f'[seed {seed}] audited failure states={len(states)}', flush=True)
    for name, rows in (('checkpoint_behavior', summaries), ('training_phases', phases),
                       ('same_state_rankings', rankings), ('same_state_changes', pairs)):
        write_records_csv(rows, output / f'{name}.csv')
    write_json(dict(complete=True, seeds=seeds, training_started=False, gradient_updates=0,
        validation_scenes=50, independent500_used=False, checkpoint_hashes=model_hashes,
        replay_steps_verified=replay_steps, source_action_checks=origin_checks, original_api_checks=api_checks,
        action_mismatches=0, future_information_used_for_actions=False,
        state_selection='all saved failures at five anchors; every 10th decision in tail60 plus terminal',
        limitations=['failure-selected states are diagnostic, not population accuracy',
            'A* agreement is not dynamic-safety ground truth',
            'historical replay/optimizer/target network and exact state coverage were not saved',
            'TD loss is logged Huber loss, not actual TD error distribution']), output / 'verification.json')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(common.SEEDS))
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(common.SEEDS):
        parser.error('Use unique seeds 0–4.')
    torch.set_num_threads(1)
    print(audit(args.seeds))


if __name__ == '__main__':
    main()

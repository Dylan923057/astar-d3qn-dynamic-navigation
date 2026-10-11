"""Single seed0 intervention; default preflight only, --train explicitly starts 200k steps."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import teaching_efficiency_common as common
from analyze_runtime_path_pilot import load_csv
from analyze_teaching_efficiency import aulc
from astar_d3qn.agents.astar_staged_withdrawal import AStarStagedWithdrawalAgent, staged_advice_probability
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.io import write_json, write_records_csv
from astar_d3qn.utils.seed import seed_everything

CONFIG = common.ROOT / 'configs/whole_map_91701_staged_withdrawal_pilot_v1.yaml'
METHOD = 'advice_bound_margin_advice80k'


def validate(config):
    reference = common.legacy.load_config(common.legacy.CONFIG)
    if config != common.legacy.load_config(CONFIG) or {
            k:v for k,v in config.items() if k not in ('experiment','staged_withdrawal')} != {
            k:v for k,v in reference.items() if k != 'experiment'}:
        raise ValueError('Map, rewards, epsilon, replay, optimization and supervision must remain unchanged.')
    if config['staged_withdrawal'] != dict(method=METHOD, seed=0, advice_tail_start_steps=60000,
            advice_end_steps=80000, supervision_end_steps=100000, environment_steps=200000,
            artifact_root='data/whole_map_91701_staged_withdrawal_pilot_v1',
            hypothesis='autonomous_collection_before_supervision_exit',
            use_independent500_for_diagnosis=False, auto_start_training=False):
        raise ValueError('Use the fixed single-seed pilot, not a new multi-seed sweep.')
    if config['experiment'] != dict(reference['experiment'],
            name='whole_map_91701_staged_withdrawal_pilot_v1',
            protocol='earlier_action_override_same_supervision_single_seed_v1',
            output_root='outputs/whole_map_91701_staged_withdrawal_pilot_v1',
            check_root='results/whole_map_91701_staged_withdrawal_pilot_v1'):
        raise ValueError('Independent pilot directories required.')
    return common.legacy.validate_config(reference)


def make_agent(config, problem, device=None):
    seed_everything(0)
    values = dict(config['agent'])
    if device: values['device'] = device
    advice, repair, schedule = config['training_advice'], config['value_repair'], config['staged_withdrawal']
    return AStarStagedWithdrawalAgent(D3QNConfig((5,15,15),4,5,seed=0,**values), problem,
        tail_start=schedule['advice_tail_start_steps'], advice_end=schedule['advice_end_steps'],
        lower=repair['target_lower'], upper=repair['target_upper'], margin=repair['margin'],
        margin_weight=repair['margin_weight'], probability=advice['probability'],
        decay_steps=advice['decay_environment_steps'], risk_radius=advice['risk_radius'],
        rng_offset=advice['rng_seed_offset'])


def source_hashes():
    files = [CONFIG, Path(__file__), common.ROOT/'src/astar_d3qn/agents/astar_staged_withdrawal.py',
             common.ROOT/'scripts/teaching_efficiency_common.py',
             *[common.ROOT/p for p in common.legacy.source_hashes()]]
    return {str(p.relative_to(common.ROOT)): common.sha256(p) for p in files}


def check(config, inputs):
    original_config = common.legacy.load_config(common.CONFIG)
    directory, original = common.audit_run(original_config,'advice_bound_margin',0)
    agent = make_agent(config, inputs[0], 'cpu')
    digest = common.legacy.runtime.state_digest(agent.training_state_dict())
    if digest != original['initial_state_sha256']:
        raise ValueError('Full initial network/target/optimizer/exploration/advice RNG is not paired.')
    baseline = common.legacy.make_agent(common.legacy.load_config(common.legacy.CONFIG),inputs[0],0,
                                       'advice_bound_margin','cpu')
    for t in range(200001):
        agent.training_action_steps = baseline.training_action_steps = t
        if agent.imitation_weight() != baseline.imitation_weight():
            raise ValueError('Auxiliary supervision changed.')
        if t <= 60000 and agent.advice_probability(t) != baseline.advice_probability(t):
            raise ValueError('Advice changed before 60000.')
        if t >= 80000 and agent.advice_probability(t) != 0:
            raise ValueError('Advice did not stop exactly at 80000.')
    return dict(passed=True, paired_seed=0, initial_state_sha256=digest,
        original_result_sha256=common.sha256(directory/'result.json'),
        full_budget=200000, gradients_in_preflight=0, training_started=False,
        original_sources_unchanged=True, identical_advice_through_step=60000,
        advice_off_from=80000, identical_supervision_all_steps=True, supervision_off_from=100000,
        supervision_weight_at_80000=max(0.,1-80000/100000),
        planned_advice_probability_sum=sum(staged_advice_probability(t) for t in range(1,200001)),
        baseline_advice_probability_sum=sum(.8*max(0.,1-t/100000) for t in range(1,200001)),
        caveat='Advice schedule area is 8% smaller; this is not an equal-teacher-count comparison.',
        evaluation=dict(epsilon=0,teacher=False,dynamic_protection=False,static_mask=True),
        final500_used=False)


def register(config, inputs):
    report = check(config,inputs)
    manifest = dict(config_sha256=common.canonical_hash(config),source_sha256=source_hashes(),
        validation_sha256=common.sha256(common.ROOT/config['dataset']['validation_reference']),
        map_sha256=config['dataset']['grid_sha256'],pool_sha256=config['dataset']['route_pool_design_sha256'],
        paired_initial_state_sha256=report['initial_state_sha256'],
        frozen_before_training=True, training_started=False)
    path = common.ROOT/config['staged_withdrawal']['artifact_root']/'freeze_manifest.json'
    if path.exists():
        if common.read_json(path) != manifest:
            raise ValueError('Registered pilot changed; preserve its manifest and output.')
    else: write_json(manifest,path)
    return report,manifest,path


class LabelCensusFactory:
    """Passive collection labels even after exit; no replay modification or extra gradient."""
    def __init__(self, factory, agent):
        self.factory,self.agent,self.bins,self.scenes = factory,agent,{},[]

    def reset_schedule(self):
        self.factory.reset_schedule(); self.bins={}; self.scenes=[]

    def __call__(self,problem,**kwargs):
        env=self.factory(problem,**kwargs); owner=self
        self.scenes.append(dict(episode_index=len(self.scenes),scenario_id=env.scenario_id,
            physical_scene_sha256=common.canonical_hash([asdict(o) for o in env.dynamic_obstacles])))
        class Observed:
            def __getattr__(self,name): return getattr(env,name)
            def reset(self):
                self.state=env.reset(); return self.state
            def step(self,action):
                state=self.state; result=env.step(action)
                row=owner.bins.setdefault((owner.agent.training_action_steps-1)//1000,Counter())
                row['collected']+=1
                owner.agent.diagnostic_planner=True
                try:
                    label=owner.agent.static_advice(state)
                    risk=label is not None and owner.agent.observed_risk(state,label)
                    transition=Transition(state,int(action),result.reward,result.observation,
                                          result.terminated,env.action_mask(True))
                    eligible=owner.agent.teacher_label(transition)
                finally: owner.agent.diagnostic_planner=False
                row['teacher_available']+=int(label is not None)
                row['executed_teacher_match']+=int(label==action)
                row['observed_teacher_risk']+=int(risk)
                row['eligible_labels']+=int(eligible)
                row['collisions']+=int(result.info['collision'])
                self.state=result.observation; return result
        return Observed()

    def records(self):
        return [dict(bin_start_step=i*1000+1,bin_end_step=min((i+1)*1000,self.agent.training_action_steps),
            **{k:r[k] for k in ('collected','teacher_available','executed_teacher_match',
                'observed_teacher_risk','eligible_labels','collisions')},
            eligible_collected_fraction=r['eligible_labels']/r['collected']) for i,r in sorted(self.bins.items())]


def run(config,inputs,manifest,manifest_path):
    directory=common.ROOT/config['experiment']['output_root']/METHOD/'seed_0'
    if directory.exists(): raise FileExistsError(f'Preserve existing run: {directory}')
    agent=make_agent(config,inputs[0])
    if common.legacy.runtime.state_digest(agent.training_state_dict())!=manifest['paired_initial_state_sha256']:
        raise ValueError('Runtime pairing mismatch.')
    training=common.legacy.effective_training(config,0,'formal')
    directory.mkdir(parents=True,exist_ok=False)
    started=perf_counter(); curves,details,behavior=[],[],[]
    distances=common.legacy.static_distances(inputs[0])
    factory=LabelCensusFactory(common.legacy.runtime.make_factory(config,*inputs,method='unguided',seed=0),agent)
    metadata=dict(config=config,method=METHOD,seed=0,pilot=True,training_mode='single_seed_pilot',
        effective_training=asdict(training),source_sha256=manifest['source_sha256'],
        initial_state_sha256=manifest['paired_initial_state_sha256'],
        freeze_manifest_sha256=common.sha256(manifest_path),test_data_used=False,evaluation_advice=False,
        model_selection='final_200000_only',independent500_used=False)
    write_json(metadata,directory/'run_manifest.json')
    def diagnostics():
        write_records_csv(agent.advice_records(),directory/'advice_budget.csv')
        write_records_csv(agent.repair_records(),directory/'value_learning.csv')
        write_records_csv(agent.td_records(),directory/'true_training_td.csv')
        write_records_csv(factory.records(),directory/'collection_label_census.csv')
        write_records_csv(factory.scenes,directory/'training_scenes.csv')
    def validate_model(step):
        summary,rows,traces=common.legacy.old.evaluate(config,inputs,0,agent)
        failures=[t for t in traces if not t['safe_success']]
        curves.append(dict(environment_steps_total=step,**summary))
        details.extend(dict(environment_steps_total=step,**r) for r in rows)
        behavior.extend(dict(environment_steps_total=step,**common.legacy.trace_behavior(t,distances)) for t in failures)
        write_records_csv(curves,directory/'validation_curve.csv')
        write_records_csv(details,directory/'validation_details.csv')
        write_records_csv(behavior,directory/'failure_behaviors.csv')
        write_json(failures,directory/f'validation_failures_{step:06d}.json')
        agent.save_weights(directory/f'model_step_{step:06d}.pth'); diagnostics()
        print(f'[{METHOD} seed0] step={step} autonomous success={summary["safe_success_rate"]:.1%}',flush=True)
    validate_model(0)
    def progress(_episode,records,full_evaluation):
        if full_evaluation:
            validate_model(records[-1]['environment_steps_total'])
            write_records_csv(list(records),directory/'training.csv')
    result=common.legacy.train_d3qn([inputs[0]],agent,training,demonstrations=(),
        reward_config=common.legacy._reward_config(config),environment_factory=factory,progress_callback=progress)
    if result.environment_steps!=200000 or result.gradient_updates!=199501 or agent.training_action_steps!=200000:
        raise ValueError('Training budget mismatch.')
    if any(r['requested'] or r['applied'] for r in agent.advice_records() if r['bin_start_step']>=80001):
        raise ValueError('Action override persisted after 80000.')
    if any(r['teacher_label_samples'] for r in agent.repair_records() if r['bin_start_step']>=100001):
        raise ValueError('Supervision persisted after 100000.')
    if curves[-1]['environment_steps_total']!=200000: validate_model(200000)
    diagnostics(); write_records_csv(list(result.episode_records),directory/'training.csv')
    agent.save_weights(directory/'model_final.pth')
    write_json(dict(**metadata,environment_steps=200000,gradient_updates=199501,
        online_replay_capacity=10000,demo_transition_count=0,final_validation=curves[-1],
        final_model_sha256=common.sha256(directory/'model_final.pth'),
        online_training_seconds=result.training_seconds,validation_seconds=result.progress_callback_seconds,
        total_seconds=perf_counter()-started,planner_calls=dict(agent.planner_calls),
        random_alignment='Identical initial learner/RNG; same physical scene at episode index. Policy divergence changes episode durations, candidate-action RNG consumption and replay contents.'),directory/'result.json')
    write_json(dict(complete=True,environment_steps=200000),directory/'completion.json')
    return directory


def analyze(config):
    new=common.ROOT/config['experiment']['output_root']/METHOD/'seed_0'
    if not (new/'completion.json').is_file(): raise FileNotFoundError('Pilot not complete; no analysis or retraining started.')
    result=common.read_json(new/'result.json')
    frozen=common.read_json(common.ROOT/config['staged_withdrawal']['artifact_root']/'freeze_manifest.json')
    if (result['config']!=config or result['source_sha256']!=source_hashes() or
            result['source_sha256']!=frozen['source_sha256'] or
            common.sha256(new/'model_final.pth')!=result['final_model_sha256']):
        raise ValueError('Pilot configuration/source/final model changed.')
    old,_=common.audit_run(common.legacy.load_config(common.CONFIG),'advice_bound_margin',0)
    rows=[]
    for method,directory in (('advice_bound_margin',old),(METHOD,new)):
        curve=load_csv(directory/'validation_curve.csv')
        x=np.asarray([int(r['environment_steps_total']) for r in curve]); y=np.asarray([float(r['safe_success_rate']) for r in curve])
        if x[0]!=0 or x[-1]!=200000 or len(x)!=21: raise ValueError('Incomplete validation curve.')
        entry=dict(method=method,seed=0,full_aulc=aulc(x,y),aulc_0_80000=aulc(x,y,0,80000),
            aulc_100000_200000=aulc(x,y,100000,200000),final_success=y[-1],
            final_collision=float(curve[-1]['dynamic_collision_rate']),final_timeout=float(curve[-1]['timeout_rate']),
            last_five_mean=float(y[-5:].mean()),
            min_80000_100500=float(y[(x>=80000)&(x<=100500)].min()),
            min_100000_120500=float(y[(x>=100000)&(x<=120500)].min()),
            min_80000_160500=float(y[(x>=80000)&(x<=160500)].min()))
        rows.append(entry)
    output=common.legacy.runtime.unique_check_dir(config,'analysis')
    write_records_csv(rows,output/'paired_seed0_metrics.csv')
    write_records_csv([dict(metric=k,new_minus_old=rows[1][k]-rows[0][k]) for k in rows[0] if k not in ('method','seed')],output/'paired_difference.csv')
    write_json(dict(training_started=False,final500_used=False,statistical_unit='seed',n=1,
        success_criterion='Inspect both exit windows, full80k–160k minimum and late AULC; improving one window alone may move the dip.'),output/'verification.json')
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(); actions.add_argument('--train',action='store_true'); actions.add_argument('--analyze',action='store_true')
    args=parser.parse_args(argv); config=common.legacy.load_config(CONFIG); inputs=validate(config)
    if args.analyze: print(analyze(config)); return
    if not args.train: torch.set_num_threads(1)
    report,manifest,path=register(config,inputs)
    if args.train: print(run(config,inputs,manifest,path))
    else:
        output=common.legacy.runtime.unique_check_dir(config,'check')
        write_json(report,output/'check_report.json')
        write_records_csv([dict(step=t,baseline_advice=.8*max(0,1-t/100000),
            staged_advice=staged_advice_probability(t),supervision=max(0,1-t/100000))
            for t in sorted(set(range(0,200001,1000))|{59999,60001,79999,80001,99999,100001})],output/'schedules.csv')
        print(output)


if __name__=='__main__': main()

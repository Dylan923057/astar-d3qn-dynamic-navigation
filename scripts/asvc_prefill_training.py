"""Isolated final-budget training with historical episode-boundary recovery."""
from asvc_supplement_common import *
from analyze_runtime_path_pilot import load_csv
from replication_recovery import instrument
from dataclasses import asdict
from time import perf_counter

def run_one(seed,resume=False):
    config,inputs=configure()
    frozen_path=ROOT/config['supplement']['registration_root']/'freeze_manifest.json'
    frozen=common.read_json(frozen_path)
    if frozen['config']!=config or frozen['source_sha256']!=source_hashes():raise ValueError('Frozen training source/config changed.')
    directory=ROOT/config['experiment']['output_root']/METHOD/f'seed_{seed}'
    if (directory/'completion.json').exists():
        if resume:print(f'[seed{seed}] already complete; preserved.',flush=True);return
        raise FileExistsError(f'Preserve existing result: {directory}')
    if directory.exists() and not resume:raise FileExistsError(f'Use explicit --resume for incomplete run: {directory}')
    if resume and not (directory/'recovery_latest.pth').is_file():raise FileNotFoundError('No full recovery checkpoint.')
    directory.mkdir(parents=True,exist_ok=resume)
    agent=make_agent(config,inputs[0],seed)
    initial=next(r['initial_state_sha256'] for r in frozen['checks']['initializations'] if r['seed']==seed)
    if legacy.runtime.state_digest(agent.training_state_dict())!=initial:raise ValueError('Runtime paired initialization mismatch.')
    factory=legacy.runtime.make_factory(config,*inputs,method='unguided',seed=seed)
    manager=Recovery(directory,agent,factory,STAGE_LOCALS,resume=resume,
        identity=dict(seed=seed,freeze_sha256=common.sha256(frozen_path)))
    step=manager.last_step
    curves=[r for r in load_csv(directory/'validation_curve.csv') if int(r['environment_steps_total'])<=step] if resume else []
    details=[r for r in load_csv(directory/'validation_details.csv') if int(r['environment_steps_total'])<=step] if resume else []
    behaviors=[r for r in load_csv(directory/'failure_behaviors.csv') if int(r['environment_steps_total'])<=step] if resume else []
    training=legacy.effective_training(config,seed,'formal');distances=legacy.static_distances(inputs[0])
    metadata=dict(config=config,method=METHOD,seed=seed,effective_training=asdict(training),
        source_sha256=frozen['source_sha256'],initial_state_sha256=initial,freeze_manifest_sha256=common.sha256(frozen_path),
        model_selection='final_200000_only',test_data_used=False,evaluation_advice=False,independent500_used=False,
        foundation_loaded=False,pretraining_updates=0,demo_transition_count=7260,online_replay_capacity=10000,
        teacher_action_separate_from_transition_action=False,random_alignment=frozen['checks']['random_alignment'])
    write_json(metadata,directory/'run_manifest.json')
    write_json(dict(complete=False,running=True,environment_steps=step),directory/'status.json')
    def diagnostics():
        write_records_csv(agent.advice_records(),directory/'advice_budget.csv')
        write_records_csv(agent.repair_records(),directory/'value_learning.csv')
        pass  # No action supervision in initialization control.
    def validate_model(current):
        summary,rows,traces=legacy.old.evaluate(config,inputs,seed,agent)
        failures=[t for t in traces if not t['safe_success']]
        curves.append(dict(environment_steps_total=current,**summary))
        details.extend(dict(environment_steps_total=current,**r) for r in rows)
        behaviors.extend(dict(environment_steps_total=current,**legacy.trace_behavior(t,distances)) for t in failures)
        write_records_csv(curves,directory/'validation_curve.csv');write_records_csv(details,directory/'validation_details.csv')
        write_records_csv(behaviors,directory/'failure_behaviors.csv')
        write_json(failures,directory/f'validation_failures_{current:06d}.json')
        agent.save_weights(directory/f'model_step_{current:06d}.pth');diagnostics()
        write_json(dict(complete=False,running=True,environment_steps=current,
            recoverable_environment_steps=manager.last_step),directory/'status.json')
        print(f'[{METHOD} seed{seed}] step={current} autonomous_success={summary["safe_success_rate"]:.1%}',flush=True)
    def progress(_episode,records,full):
        if full:validate_model(records[-1]['environment_steps_total']);write_records_csv(list(records),directory/'training.csv')
    init_clock=perf_counter(); demonstrations=load_demonstrations(); initialization_seconds=perf_counter()-init_clock
    loop=instrument(legacy.train_d3qn,STAGE_LOCALS)
    original_build=loop.__globals__['build_replay']
    replay_initialization_seconds=0.
    def measured_build(*args,**kwargs):
        nonlocal replay_initialization_seconds
        started=perf_counter(); replay=original_build(*args,**kwargs)
        replay_initialization_seconds+=perf_counter()-started
        return replay
    loop.__globals__['build_replay']=measured_build
    clock=perf_counter()
    if not resume:validate_model(0)
    try:
        result=loop([inputs[0]],agent,training,
            demonstrations=demonstrations, reward_config=legacy._reward_config(config),environment_factory=factory,progress_callback=progress,_recovery=manager)
        if result.environment_steps!=200000 or result.gradient_updates!=199501 or agent.update_steps!=199501 or agent.training_action_steps!=200000:
            raise ValueError('Formal budget mismatch.')
        if any(r['requested'] or r['applied'] for r in agent.advice_records() if r['bin_start_step']>=100001):raise ValueError('Advice persisted after100k.')
        if any(r['teacher_label_samples'] for r in agent.repair_records() if r['bin_start_step']>=100001):raise ValueError('Supervision persisted after100k.')
        if int(curves[-1]['environment_steps_total'])!=200000:validate_model(200000)
        agent.save_weights(directory/'model_final.pth');diagnostics()
        write_records_csv(list(result.episode_records),directory/'training.csv')
        write_json(dict(**metadata,complete=True,environment_steps=200000,gradient_updates=199501,online_updates=199501,
            initialization_seconds=initialization_seconds+replay_initialization_seconds,
            demonstration_load_seconds=initialization_seconds,replay_initialization_seconds=replay_initialization_seconds,
            device=str(agent.device), device_name=torch.cuda.get_device_name(agent.device) if agent.device.type=='cuda' else 'CPU', online_astar_lookups=0, online_astar_searches=0, online_astar_seconds=0., offline_updates=0,
            final_model_sha256=common.sha256(directory/'model_final.pth'),final_validation=curves[-1],
            online_training_seconds=manager.prior_training_seconds+result.training_seconds,
            validation_and_recovery_seconds=result.progress_callback_seconds,session_seconds=perf_counter()-clock),directory/'result.json')
        write_json(dict(complete=True,environment_steps=200000),directory/'completion.json')
        write_json(dict(complete=True,running=False,environment_steps=200000,
            recoverable_environment_steps=manager.last_step),directory/'status.json')
        print(f'[{METHOD} seed{seed}] COMPLETE {directory}',flush=True)
    except BaseException as error:
        write_json(dict(complete=False,running=False,recoverable_environment_steps=manager.last_step,error=repr(error),
            resume_command=f'python scripts/run_asvc_supplement.py --train --seeds {seed} --resume'),directory/'status.json')
        raise


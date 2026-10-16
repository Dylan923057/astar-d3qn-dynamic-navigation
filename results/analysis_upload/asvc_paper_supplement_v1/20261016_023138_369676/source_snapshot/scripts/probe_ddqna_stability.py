"""Independent short numerical probe. Default checks only; --run explicitly trains."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict,replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
import platform

import numpy as np
import torch
import torch.nn.functional as F

import ddqna_adapter_common as common
from astar_d3qn.agents.ddqna_adapter import DDQNAD3QNAgent
from astar_d3qn.agents.d3qn import D3QNAgent,D3QNConfig
from astar_d3qn.utils.seed import seed_everything
from astar_d3qn.utils.io import write_json,write_records_csv

CONFIG=common.ROOT/'configs/ddqna_stability_probe_v1.yaml'
ARMS=('huber','mse','huber_bound')


class SceneAuditFactory:
    """Observe episode-index scene pairing without consuming a random draw."""
    def __init__(self,factory):
        self.factory=factory;self.rows=[]
    def __getattr__(self,name):return getattr(self.factory,name)
    def reset_schedule(self):
        self.factory.reset_schedule();self.rows=[]
    def __call__(self,problem,**kwargs):
        env=self.factory(problem,**kwargs)
        self.rows.append(dict(episode=len(self.rows)+1,scenario_id=env.scenario_id,
            scene_sha256=common.canonical_hash([asdict(o) for o in env.dynamic_obstacles])))
        return env


class ProbeAgent(DDQNAD3QNAgent):
    def __init__(self,config,problem,arm):
        super().__init__(config,problem,probability=.5)
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.update_profiles=defaultdict(list)

    def train_batch(self,batch,sample_weights=None,demonstration_mask=None,**kwargs):
        if (sample_weights is not None or demonstration_mask is not None or
                any(kwargs.get(k,0) for k in ('demo_loss_weight','safe_guidance_loss_weight','conflict_margin_loss_weight'))):
            raise ValueError('This numerical probe permits ordinary uniform TD only.')
        spatial=torch.as_tensor(np.stack([t.state.spatial for t in batch]),device=self.device)
        scalars=torch.as_tensor(np.stack([t.state.scalars for t in batch]),device=self.device)
        actions=torch.tensor([t.action for t in batch],device=self.device)
        reward=torch.tensor([t.reward for t in batch],device=self.device,dtype=torch.float32)
        terminated=torch.tensor([t.terminated for t in batch],device=self.device)
        next_spatial=torch.as_tensor(np.stack([t.next_state.spatial for t in batch]),device=self.device)
        next_scalars=torch.as_tensor(np.stack([t.next_state.scalars for t in batch]),device=self.device)
        predicted=self.policy_network(spatial,scalars).gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            raw=self._td_targets(reward,terminated,self.policy_network(next_spatial,next_scalars),
                self.target_network(next_spatial,next_scalars),self._next_action_mask(batch))
            target=raw.clamp(-6,10) if self.arm=='huber_bound' else raw
        errors=target-predicted.detach()
        raw_errors=raw-predicted.detach()
        loss=F.mse_loss(predicted,target) if self.arm=='mse' else F.smooth_l1_loss(predicted,target)
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        if self.config.gradient_clip_norm>0:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.config.gradient_clip_norm)
        self.optimizer.step();self.update_steps+=1
        if self.update_steps%self.config.target_sync_interval==0:self.sync_target()
        self.update_profiles[(self.update_steps-1)//1000].append(dict(loss=loss.item(),
            environment_step=self.training_action_steps,update=self.update_steps,
            predicted_min=predicted.min().item(),predicted_max=predicted.max().item(),
            raw_target_min=raw.min().item(),raw_target_max=raw.max().item(),
            actual_target_min=target.min().item(),actual_target_max=target.max().item(),
            td_abs_mean=errors.abs().mean().item(),td_abs_max=errors.abs().max().item(),
            td_abs_p95=torch.quantile(errors.abs(),.95).item(),td_signed_mean=errors.mean().item(),
            raw_td_abs_mean=raw_errors.abs().mean().item(),raw_td_abs_max=raw_errors.abs().max().item(),
            predicted_outside_return_count=((predicted < -6)|(predicted > 10)).sum().item(),
            predicted_abs_gt20_count=(predicted.abs()>20).sum().item(),
            actual_target_clipped_count=(target!=raw).sum().item(),sample_count=len(batch)))
        return dict(loss=loss.item(),td_errors=errors.abs().cpu().numpy(),safe_guidance_margin_loss=0.,
            safe_guidance_batch_count=0,conflict_margin_loss=0.,conflict_margin_batch_count=0)

    def profiles(self):
        return [dict(update_start=i*1000+1,updates=len(rows),
            **{k+'_mean':float(np.mean([r[k] for r in rows])) for k in rows[0]},
            predicted_max_peak=max(r['predicted_max'] for r in rows),
            td_abs_max_peak=max(r['td_abs_max'] for r in rows)) for i,rows in sorted(self.update_profiles.items())]

    def update_records(self):
        return [r for _,rows in sorted(self.update_profiles.items()) for r in rows]


def make_agent(config,problem,seed,arm,device=None):
    seed_everything(seed);values=dict(config['agent'])
    if device:values['device']=device
    return ProbeAgent(D3QNConfig((5,15,15),4,5,seed=seed,**values),problem,arm)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--seed',type=int,choices=common.SEEDS,default=0)
    parser.add_argument('--arms',choices=ARMS,nargs='+',default=list(ARMS));args=parser.parse_args()
    if len(set(args.arms))!=len(args.arms):parser.error('Use unique arms.')
    original=common.legacy.load_config(common.CONFIG);common.register(original)
    config=common.legacy.load_config(CONFIG);problem,*other=common.validate_config(original)
    if {k:v for k,v in config.items() if k not in ('experiment','stability_probe')}!={
            k:v for k,v in original.items() if k!='experiment'}:raise ValueError('Shared conditions changed.')
    spec=config['stability_probe']
    if (spec['environment_steps']!=50000 or spec['teacher_probability']!=.5 or spec['use_final500'] or
            [spec['target_lower'],spec['target_upper']]!=[-6,10]):
        raise ValueError('Probe must be 50k online steps, constant teacher, no final500.')
    reference=common.historical.baseline_digest(original,problem,args.seed)
    for arm in args.arms:
        if common.historical.paired_digest(make_agent(config,problem,args.seed,arm,'cpu'))!=reference:
            raise ValueError('Initial learner mismatch.')
    if not args.run:
        print('PASS: paired initializations; unchanged map/reward/epsilon; 50k limit; no training started.');return
    output=common.ROOT/config['experiment']['output_root']/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    for arm in args.arms:
        directory=output/arm/f'seed_{args.seed}';directory.mkdir(parents=True,exist_ok=False)
        agent=make_agent(config,problem,args.seed,arm)
        if common.historical.paired_digest(agent)!=reference:raise ValueError('Runtime initialization mismatch.')
        started=perf_counter()
        initial_rng=dict(exploration=common.legacy.runtime.state_digest(agent._rng.getstate()),
            advice=common.legacy.runtime.state_digest(agent.advice_rng.getstate()),
            torch=common.legacy.runtime.state_digest(torch.get_rng_state()))
        training=replace(common.legacy.effective_training(config,args.seed,'formal'),max_environment_steps=50000,
            allow_partial_epsilon_schedule=True)
        curves=[]
        def validate(step):
            # Historical evaluator is autonomous; epsilon=0 bypasses fixed-p teacher.
            summary,rows,traces=common.legacy.old.evaluate(config,(problem,*other),args.seed,agent)
            curves.append(dict(environment_steps_total=step,**summary))
            write_records_csv(curves,directory/'validation_curve.csv')
            write_records_csv(agent.profiles(),directory/'update_profiles.csv')
            write_records_csv(agent.update_records(),directory/'updates.csv')
            agent.save_weights(directory/f'model_step_{step:06d}.pth')
            write_json([t for t in traces if not t['safe_success']],directory/f'failures_{step:06d}.json')
            print(f'[{arm}] step={step} autonomous success={summary["safe_success_rate"]:.1%}',flush=True)
        validate(0)
        def progress(_episode,records,full_evaluation):
            if full_evaluation:validate(records[-1]['environment_steps_total'])
        factory=SceneAuditFactory(common.legacy.runtime.make_factory(config,problem,*other,method='unguided',seed=args.seed))
        result=common.legacy.train_d3qn([problem],agent,training,demonstrations=(),
            reward_config=common.legacy._reward_config(config),environment_factory=factory,progress_callback=progress)
        if result.environment_steps!=50000 or result.gradient_updates!=49501:raise ValueError('Probe budget changed.')
        if curves[-1]['environment_steps_total']!=50000:validate(50000)
        agent.save_weights(directory/'model_probe_final.pth')
        write_records_csv(list(result.episode_records),directory/'training.csv')
        write_records_csv(agent.advice_records(),directory/'action_branches.csv')
        write_records_csv(agent.profiles(),directory/'update_profiles.csv')
        write_records_csv(agent.update_records(),directory/'updates.csv')
        write_records_csv(factory.rows,directory/'training_scenes.csv')
        write_json(dict(complete=True,arm=arm,seed=args.seed,config=config,effective_training=asdict(training),
            environment_steps=50000,gradient_updates=result.gradient_updates,initial_learner_sha256=reference,
            epsilon_clock_not_compressed=True,teacher_not_withdrawn=True,final500_used=False,
            inference_model_selection=False,purpose='Numerical diagnosis only; not five-seed final evidence',
            initial_rng_sha256=initial_rng,random_alignment=dict(network_global_seed=args.seed,
                exploration_seed=args.seed,advice_seed=args.seed+config['training_advice']['rng_seed_offset'],
                uniform_replay_seed=args.seed,scene_seed=args.seed+config['dynamic_route_pool']['training_sampling_seed_offset'],
                scene_pairing='episode index; environment-step alignment can diverge with episode duration',
                action_draw_pairing='same RNG initial states; conditional choice consumption can diverge after policy divergence',
                replay_pairing='same initial RNG, update frequency and capacity; sampled content changes with trajectories',
                deterministic_torch=True,evaluation_preserves_training_rng=True),
            online_training_seconds=result.training_seconds,validation_seconds=result.progress_callback_seconds,
            total_seconds=perf_counter()-started,hardware=dict(device=str(agent.device),torch=torch.__version__,
                platform=platform.platform(),torch_threads=torch.get_num_threads()),
            script_sha256=common.sha256(Path(__file__)),config_sha256=common.sha256(CONFIG),
            original_registration_sha256=common.sha256(common.ROOT/original['ddqna_adapter']['artifact_root']/'freeze_manifest.json')),
            directory/'result.json')
    print(output)


if __name__=='__main__':main()

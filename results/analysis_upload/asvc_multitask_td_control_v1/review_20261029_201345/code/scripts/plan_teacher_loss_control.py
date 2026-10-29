"""Validate/emit proposed teacher-loss jobs. Deliberately has no training action."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import yaml
import td_control_common as c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--export',action='store_true')
    parser.add_argument('--loss',choices=('large_margin','cross_entropy'));parser.add_argument('--candidate',type=float)
    parser.add_argument('--seed',type=int,choices=(0,1));parser.add_argument('--phase',choices=('tuning','final'),default='tuning')
    args=parser.parse_args();path=ROOT/'configs/multitask_teacher_loss_plan_v1.yaml'
    plan=yaml.safe_load(path.read_text(encoding='utf-8'));c.previous.verify()
    assert plan['training_enabled'] is False and plan['execution']['current_round_training_allowed'] is False
    assert plan['shared']['complete_td_target_clip']==[-6,10] and plan['shared']['risk_soft_weighting'] is False
    assert len(plan['tuning']['large_margin_candidates'])==len(plan['tuning']['cross_entropy_temperature_candidates'])==3
    jobs=[]
    for loss,key in (('large_margin','large_margin_candidates'),('cross_entropy','cross_entropy_temperature_candidates')):
        for candidate in plan['tuning'][key]:
            for seed in plan['tuning']['seeds']:
                jobs.append(dict(loss=loss,candidate=candidate,seed=seed,phase='tuning',online_steps=100000,
                    training_enabled=False,request_command=f'python scripts/plan_teacher_loss_control.py --loss {loss} --candidate {candidate} --seed {seed} --export'))
    if args.loss:
        if args.candidate is None or args.seed is None:parser.error('--loss requires --candidate and --seed')
        candidates=plan['tuning']['large_margin_candidates' if args.loss=='large_margin' else 'cross_entropy_temperature_candidates']
        if args.candidate not in candidates:parser.error('Candidate was not in the proposed equal-budget grid')
        jobs=[dict(loss=args.loss,candidate=args.candidate,seed=args.seed,phase=args.phase,
            online_steps=100000 if args.phase=='tuning' else 500000,training_enabled=False)]
    payload=dict(status='proposal_only_no_training',plan_sha256=c.sha(path),shared_identity=c.previous.shared_identity(),
        full_qdagger_replication=False,actual_training_runner_implemented=False,
        tuning_runs_per_loss=6,tuning_online_steps_per_loss=600000,jobs=jobs)
    if args.export:
        output=c.RESULTS/'teacher_loss_plan_requests.json';c.write(output,payload);print(output)
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
